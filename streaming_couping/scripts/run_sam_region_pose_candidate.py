#!/usr/bin/env python3
"""Evaluate MESA-style SAM-region filtering as an isolated V0 pose candidate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from streaming_couping.src.config import load_config
from streaming_couping.src.external_repos import maybe_add_repo_to_path
from streaming_couping.src.learned_pose.baseline_runtime import (
    camera_centers,
    decode_cached_poses,
    load_baseline_run_config,
    pose_metrics,
)
from streaming_couping.src.learned_pose.cache import cache_path, load_feature_cache
from streaming_couping.src.learned_pose.config import load_learned_pose_config
from streaming_couping.src.sam_region_pose_candidate import (
    build_match_pool,
    deployable_cache_view,
    extract_sift_features,
    load_sam_region_pose_config,
    scene_scale_from_cache,
    select_equal_count_correspondences,
    solve_pose_candidate,
)


REVISION = "v0_mesa_style_sam_region_sift_pointmap_pnp_r1"


def main() -> None:
    args = _parse_args()
    data = load_learned_pose_config(args.config)
    baseline = load_baseline_run_config(args.config)
    candidate = load_sam_region_pose_config(args.config)
    if not candidate.enabled:
        raise ValueError("sam_region_pose_candidate.enabled is false.")
    clip = next(item for item in data.clips if item.name == baseline.clip_name)
    path = cache_path(data, clip)
    full_payload = load_feature_cache(path)
    payload = deployable_cache_view(full_payload)
    expected_prompts = ("bed", "wardrobe", "picture", "mat", "chair")
    if tuple(str(value) for value in payload["instance_prompts"]) != expected_prompts:
        raise ValueError(
            "Candidate requires the five-prompt V0 cache; run "
            "BASELINE_REBUILD_CACHE=1 first."
        )
    if len(payload["instance_ids"]) != 16:
        raise ValueError("Candidate requires the 16-slot V0 registry cache.")

    recovery = load_config(data.recovery_config)
    maybe_add_repo_to_path(recovery.streamvggt_repo)
    from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri

    encoding = full_payload["baseline_pose_encoding"].unsqueeze(0).cpu()
    raw_pose, intrinsics = pose_encoding_to_extri_intri(
        encoding,
        image_size_hw=tuple(int(value) for value in full_payload["image_size"]),
    )
    frames = tuple(int(value) for value in payload["frame_indices"])
    positions = {frame: index for index, frame in enumerate(frames)}
    evaluation_indices = [positions[frame] for frame in baseline.evaluation_frames]
    generated = generate_candidates(
        payload=payload,
        raw_pose=raw_pose.cpu(),
        intrinsics=intrinsics.cpu(),
        evaluation_indices=evaluation_indices,
        frames=frames,
        config=candidate,
    )
    # Candidate generation has returned and is no longer allowed to mutate
    # poses.  Only now may the scoring-only path decode the GT encoding.
    _, target_pose = decode_cached_poses(
        full_payload,
        pose_decoder=pose_encoding_to_extri_intri,
        device="cpu",
    )
    output = score_and_write_candidate(
        payload=payload,
        scoring_metadata={
            "clip_name": full_payload["clip_name"],
            "reference_sequence_index": int(
                full_payload["reference_sequence_index"]
            ),
        },
        raw_pose=raw_pose.cpu(),
        target_pose=target_pose.cpu(),
        generated=generated,
        evaluation_indices=evaluation_indices,
        frames=frames,
        config=candidate,
        cache_path_value=path,
    )
    print(f"V0 SAM-region pose decision={output}")


def generate_candidates(
    *,
    payload: dict[str, object],
    raw_pose: torch.Tensor,
    intrinsics: torch.Tensor,
    evaluation_indices: list[int],
    frames: tuple[int, ...],
    config,
) -> dict[str, object]:
    """Generate immutable pose candidates from deployable fields only."""

    forbidden = [
        name
        for name in payload
        if name.startswith("target_") or "ground_truth" in name
    ]
    if forbidden:
        raise RuntimeError(f"Candidate generation received GT fields={forbidden}.")
    needed = set(evaluation_indices)
    for current in evaluation_indices:
        needed.update(
            current - offset
            for offset in config.anchor_offsets
            if current - offset >= 0
        )
    features = [None] * len(frames)
    for index in sorted(needed):
        features[index] = extract_sift_features(
            payload["stream_images"][index], config
        )
        print(
            f"V0 SAM-region SIFT frame={frames[index]} "
            f"features={features[index].points.shape[0]}"
        )

    scene_scale = scene_scale_from_cache(
        payload, config.point_confidence_threshold
    )
    method_poses = {
        method: raw_pose.detach().clone() for method in config.methods
    }
    method_active = {method: set() for method in config.methods}
    method_sam_active = {method: set() for method in config.methods}
    rows: list[dict[str, object]] = []
    image_size = tuple(int(value) for value in payload["stream_images"].shape[-2:])
    for current in evaluation_indices:
        pool = build_match_pool(
            payload=payload,
            current_index=current,
            features=features,
            raw_world_to_camera=raw_pose,
            intrinsics=intrinsics,
            config=config,
        )
        selections, selection_diagnostics = select_equal_count_correspondences(
            pool,
            config=config,
            image_size=image_size,
        )
        for method in config.methods:
            diagnostics = selection_diagnostics[method]
            sam_evidence_active = (
                method == "sam_region_identity"
                and diagnostics["selected_instance_correspondences"]
                >= int(config.min_instance_correspondences)
            )
            pose, solver = solve_pose_candidate(
                pool=pool,
                selected=selections[method],
                raw_pose=raw_pose[0, current],
                intrinsics=intrinsics[0, current],
                scene_scale=scene_scale,
                config=config,
                sam_evidence_active=sam_evidence_active,
            )
            if int(solver["optimized"]):
                method_poses[method][0, current] = pose
                method_active[method].add(current)
            if sam_evidence_active:
                method_sam_active[method].add(current)
            rows.append(
                {
                    "method": method,
                    "sequence_index": current,
                    "frame_index": frames[current],
                    "anchor_offsets": " ".join(
                        str(value)
                        for value in config.anchor_offsets
                        if current - value >= 0
                    ),
                    "candidate_pool": int(pool.current_points.shape[0]),
                    **diagnostics,
                    **solver,
                }
            )
        locked = selection_diagnostics[config.primary_method]["locked_equal_count"]
        instance = selection_diagnostics[config.primary_method][
            "selected_instance_correspondences"
        ]
        print(
            f"V0 SAM-region frame={frames[current]} pool={pool.current_points.shape[0]} "
            f"equal_count={locked} primary_instance={instance}"
        )

    return {
        "method_poses": method_poses,
        "method_active": method_active,
        "method_sam_active": method_sam_active,
        "frame_rows": rows,
        "scene_scale": scene_scale,
    }


def score_and_write_candidate(
    *,
    payload: dict[str, object],
    scoring_metadata: dict[str, object],
    raw_pose: torch.Tensor,
    target_pose: torch.Tensor,
    generated: dict[str, object],
    evaluation_indices: list[int],
    frames: tuple[int, ...],
    config,
    cache_path_value: Path,
) -> Path:
    """Score already-generated candidates; never generate or select a pose."""

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    method_poses = generated["method_poses"]
    method_active = generated["method_active"]
    method_sam_active = generated["method_sam_active"]
    rows = generated["frame_rows"]
    attach_frame_scores(
        rows,
        raw_pose=raw_pose,
        target_pose=target_pose,
        method_poses=method_poses,
    )
    fold_rows, decision = evaluate_methods(
        raw_pose=raw_pose,
        target_pose=target_pose,
        method_poses=method_poses,
        method_active=method_active,
        method_sam_active=method_sam_active,
        evaluation_indices=evaluation_indices,
        frames=frames,
        reference_index=int(scoring_metadata["reference_sequence_index"]),
        primary_method=config.primary_method,
    )
    write_csv(output_dir / "frame_diagnostics.csv", rows)
    write_csv(output_dir / "fold_decision.csv", fold_rows)
    summary = {
        "schema": 1,
        "baseline_version": "v0",
        "implementation_revision": REVISION,
        "config": str(config.source_path),
        "candidate_status": "offline_not_selected",
        "selected_pose_unchanged": True,
        "claim_level": "sam_region_matching_pose_candidate",
        "cache": str(cache_path_value),
        "clip": scoring_metadata["clip_name"],
        "frames": frames,
        "evaluation_frames": tuple(frames[index] for index in evaluation_indices),
        "configured_instance_prompts": tuple(payload["instance_prompts"]),
        "permanent_slot_capacity": len(payload["instance_ids"]),
        "max_pose_instances_per_frame": config.max_pose_instances,
        "discovered_track_count": sum(
            int(value) >= 0 for value in payload["sam_track_ids"]
        ),
        "methods": config.methods,
        "primary_method": config.primary_method,
        "anchor_offsets": config.anchor_offsets,
        "candidate_generation_gt_fields": 0,
        "gt_role": "scoring_only_after_candidate_generation",
        "pose_model_trained": False,
        "matcher_trained": False,
        "correspondence_source": "opencv_sift_mutual_ratio",
        "geometry_source": "frozen_streamvggt_world_pointmap",
        "sam_role": "persistent_region_identity_match_filter",
        "solver": "calibrated_pnp_ransac_epnp_lm_with_heldout_reprojection_gate",
        "scene_scale_native": generated["scene_scale"],
        "decision": decision,
    }
    result = output_dir / "candidate_summary.json"
    result.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf8",
    )
    torch.save(
        {
            "frame_indices": frames,
            "raw_world_to_camera": raw_pose,
            "candidate_world_to_camera": method_poses,
            "selected_pose_unchanged": True,
        },
        output_dir / "candidate_poses.pt",
    )
    report = output_dir / "copyable_result.txt"
    report.write_text(
        copyable_report(
            summary=summary,
            frame_rows=rows,
            fold_rows=fold_rows,
        ),
        encoding="utf8",
    )
    print(report.read_text(encoding="utf8"))
    return result


def attach_frame_scores(
    rows: list[dict[str, object]],
    *,
    raw_pose: torch.Tensor,
    target_pose: torch.Tensor,
    method_poses: dict[str, torch.Tensor],
) -> None:
    for row in rows:
        index = int(row["sequence_index"])
        raw_rotation, raw_center = frame_pose_errors(
            raw_pose[0, index], target_pose[0, index]
        )
        candidate_rotation, candidate_center = frame_pose_errors(
            method_poses[str(row["method"])][0, index], target_pose[0, index]
        )
        row.update(
            {
                "raw_rotation_degrees": raw_rotation,
                "candidate_rotation_degrees": candidate_rotation,
                "rotation_improved": int(candidate_rotation < raw_rotation),
                "raw_center_error_native": raw_center,
                "candidate_center_error_native": candidate_center,
                "center_improved": int(candidate_center < raw_center),
            }
        )


def evaluate_methods(
    *,
    raw_pose: torch.Tensor,
    target_pose: torch.Tensor,
    method_poses: dict[str, torch.Tensor],
    method_active: dict[str, set[int]],
    method_sam_active: dict[str, set[int]],
    evaluation_indices: list[int],
    frames: tuple[int, ...],
    reference_index: int,
    primary_method: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if len(evaluation_indices) != 12:
        raise ValueError("SAM-region pose audit requires twelve evaluation frames.")
    rows = []
    by_fold_method: dict[tuple[str, str], dict[str, object]] = {}
    for fold_index, fold in enumerate(("short", "medium", "long")):
        indices = evaluation_indices[4 * fold_index : 4 * (fold_index + 1)]
        raw = pose_metrics(
            raw_pose,
            target_pose,
            reference_index=reference_index,
            evaluation_indices=indices,
        )
        for method, poses in method_poses.items():
            candidate = pose_metrics(
                poses,
                target_pose,
                reference_index=reference_index,
                evaluation_indices=indices,
            )
            center_worse = 0
            rotation_worse = 0
            for index in indices:
                raw_rotation, raw_center = frame_pose_errors(
                    raw_pose[0, index], target_pose[0, index]
                )
                candidate_rotation, candidate_center = frame_pose_errors(
                    poses[0, index], target_pose[0, index]
                )
                center_worse += int(candidate_center > raw_center + 1e-9)
                rotation_worse += int(candidate_rotation > raw_rotation + 1e-6)
            center_gain = gain_percent(
                raw["center_error_native"], candidate["center_error_native"]
            )
            rotation_gain = gain_percent(
                raw["rotation_degrees"], candidate["rotation_degrees"]
            )
            active = sum(index in method_active[method] for index in indices)
            sam_active = sum(index in method_sam_active[method] for index in indices)
            pose_pass = int(
                active == 4
                and center_gain > 0.0
                and rotation_gain > 0.0
                and center_worse <= 1
                and rotation_worse <= 1
            )
            row = {
                "fold": fold,
                "method": method,
                "test_frames": " ".join(str(frames[index]) for index in indices),
                "active_frames": active,
                "sam_evidence_active_frames": sam_active,
                "raw_center_error_native": raw["center_error_native"],
                "candidate_center_error_native": candidate["center_error_native"],
                "center_gain_percent": center_gain,
                "center_worse_frames": center_worse,
                "raw_rotation_degrees": raw["rotation_degrees"],
                "candidate_rotation_degrees": candidate["rotation_degrees"],
                "rotation_gain_percent": rotation_gain,
                "rotation_worse_frames": rotation_worse,
                "pose_pass": pose_pass,
                "sam_beats_full": 0,
                "sam_beats_shuffled": 0,
                "causal_fold_pass": 0,
            }
            rows.append(row)
            by_fold_method[(fold, method)] = row

    causal_by_fold = {}
    for fold in ("short", "medium", "long"):
        primary = by_fold_method[(fold, primary_method)]
        full = by_fold_method[(fold, "full_image_match")]
        shuffled = by_fold_method[(fold, "shuffled_instance_identity")]
        beats_full = int(
            float(primary["candidate_center_error_native"])
            < float(full["candidate_center_error_native"])
            and float(primary["candidate_rotation_degrees"])
            < float(full["candidate_rotation_degrees"])
        )
        beats_shuffled = int(
            float(primary["candidate_center_error_native"])
            < float(shuffled["candidate_center_error_native"])
            and float(primary["candidate_rotation_degrees"])
            < float(shuffled["candidate_rotation_degrees"])
        )
        causal = int(
            int(primary["pose_pass"]) == 1
            and int(primary["sam_evidence_active_frames"]) == 4
            and beats_full == 1
            and beats_shuffled == 1
        )
        primary["sam_beats_full"] = beats_full
        primary["sam_beats_shuffled"] = beats_shuffled
        primary["causal_fold_pass"] = causal
        causal_by_fold[fold] = causal
    decision = {
        "primary_method": primary_method,
        "primary_all_fold_pose_pass": int(
            all(by_fold_method[(fold, primary_method)]["pose_pass"] for fold in causal_by_fold)
        ),
        "primary_all_fold_sam_causal_pass": int(all(causal_by_fold.values())),
        "causal_fold_pass": causal_by_fold,
        "selected_pose_modified": 0,
        "claim": "candidate_only_not_deployed_pose_improvement",
    }
    return rows, decision


def frame_pose_errors(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> tuple[float, float]:
    relative = predicted[:3, :3] @ target[:3, :3].T
    cosine = ((torch.trace(relative) - 1.0) * 0.5).clamp(-1, 1)
    rotation = float(torch.rad2deg(torch.acos(cosine)))
    center = float(
        torch.linalg.vector_norm(
            camera_centers(predicted[None]) - camera_centers(target[None]),
            dim=-1,
        )[0]
    )
    return rotation, center


def gain_percent(raw: float, candidate: float) -> float:
    return 100.0 * (float(raw) - float(candidate)) / max(float(raw), 1e-12)


def copyable_report(
    *,
    summary: dict[str, object],
    frame_rows: list[dict[str, object]],
    fold_rows: list[dict[str, object]],
) -> str:
    frame_columns = (
        "frame_index",
        "method",
        "candidate_pool",
        "available_correspondences",
        "locked_equal_count",
        "selected_instance_regions",
        "selected_instance_correspondences",
        "selected_background_correspondences",
        "optimized",
        "reason",
        "inliers",
        "inlier_ratio",
        "raw_validation_rmse_pixels",
        "candidate_validation_rmse_pixels",
        "validation_gain_fraction",
    )
    fold_columns = (
        "fold",
        "method",
        "test_frames",
        "active_frames",
        "sam_evidence_active_frames",
        "raw_center_error_native",
        "candidate_center_error_native",
        "center_gain_percent",
        "center_worse_frames",
        "raw_rotation_degrees",
        "candidate_rotation_degrees",
        "rotation_gain_percent",
        "rotation_worse_frames",
        "pose_pass",
        "sam_beats_full",
        "sam_beats_shuffled",
        "causal_fold_pass",
    )
    lines = [
        "===== COPYABLE_V0_SAM_REGION_POSE_RESULT_BEGIN =====",
        f"revision={summary['implementation_revision']}",
        f"clip={summary['clip']}",
        "methods=" + ",".join(summary["methods"]),
        "configured_prompts=" + ",".join(summary["configured_instance_prompts"]),
        f"discovered_tracks={summary['discovered_track_count']}",
        f"max_pose_instances_per_frame={summary['max_pose_instances_per_frame']}",
        "candidate_generation_gt_fields=0",
        "selected_pose_modified=0",
        "",
        ",".join(frame_columns),
    ]
    for row in frame_rows:
        lines.append(",".join(str(row.get(name, "")) for name in frame_columns))
    lines.extend(("", ",".join(fold_columns)))
    for row in fold_rows:
        lines.append(
            ",".join(str(row.get(name, "")) for name in fold_columns)
        )
    lines.extend(
        (
            "",
            "decision=" + json.dumps(summary["decision"], sort_keys=True),
            "===== COPYABLE_V0_SAM_REGION_POSE_RESULT_END =====",
            "",
        )
    )
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV {path}.")
    columns = tuple(dict.fromkeys(name for row in rows for name in row))
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="streaming_couping/configs/v0_baseline.yaml",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
