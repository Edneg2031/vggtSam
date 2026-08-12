#!/usr/bin/env python3
"""Evaluate causal StreamVGGT TrackHead factors as a V0 pose candidate."""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path

import torch

from streaming_couping.src.config import load_config
from streaming_couping.src.external_repos import maybe_add_repo_to_path
from streaming_couping.src.learned_pose.baseline_runtime import (
    camera_centers,
    load_baseline_run_config,
    load_track_ba_candidate_config,
    pose_metrics,
)
from streaming_couping.src.learned_pose.cache import (
    cache_path,
    load_feature_cache,
)
from streaming_couping.src.learned_pose.config import (
    load_learned_pose_config,
)
from streaming_couping.src.v0_track_ba import (
    build_track_window,
    compose_relative_candidate,
    load_frozen_track_head,
    optimize_track_window,
    region_coverage,
    scene_scale_from_cache,
    validate_track_cache,
)


REVISION = "causal_track_head_fixed_structure_ba_r2_validity_audit"


def main() -> None:
    args = _parse_args()
    data = load_learned_pose_config(args.config)
    baseline = load_baseline_run_config(args.config)
    candidate = load_track_ba_candidate_config(args.config)
    if args.device:
        from dataclasses import replace

        candidate = replace(candidate, device=str(args.device))
    if not candidate.enabled:
        raise ValueError("baseline.track_ba_candidate.enabled is false.")
    clip = next(
        item for item in data.clips if item.name == baseline.clip_name
    )
    path = cache_path(data, clip)
    payload = load_feature_cache(path)
    validate_track_cache(payload)
    recovery = load_config(data.recovery_config)
    maybe_add_repo_to_path(recovery.streamvggt_repo)
    from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri

    # Candidate generation decodes only deployable raw quantities. Target
    # pose stays on CPU and is not decoded/moved into the generation device.
    raw_pose, _ = pose_encoding_to_extri_intri(
        payload["baseline_pose_encoding"].unsqueeze(0).to(candidate.device),
        image_size_hw=tuple(int(v) for v in payload["image_size"]),
    )
    encoding = payload["baseline_pose_encoding"].unsqueeze(0).to(
        candidate.device
    )
    _, intrinsics = pose_encoding_to_extri_intri(
        encoding,
        image_size_hw=tuple(int(v) for v in payload["image_size"]),
    )
    frames = tuple(int(value) for value in payload["frame_indices"])
    positions = {frame: index for index, frame in enumerate(frames)}
    evaluation_indices = [
        positions[frame] for frame in baseline.evaluation_frames
    ]
    head = load_frozen_track_head(
        repo_path=recovery.streamvggt_repo,
        checkpoint_path=recovery.streamvggt_checkpoint,
        device=candidate.device,
    )
    scene_scale = scene_scale_from_cache(
        payload,
        candidate.point_confidence_threshold,
    )
    if args.validity_audit_only:
        output = run_validity_audit(
            payload=payload,
            raw_pose=raw_pose,
            intrinsics=intrinsics,
            evaluation_indices=evaluation_indices,
            frames=frames,
            head=head,
            candidate=candidate,
        )
        print(f"V0 Track-BA validity audit={output}")
        return
    target_pose = _decode_target_for_scoring(
        payload=payload,
        pose_decoder=pose_encoding_to_extri_intri,
    )
    output = run_candidate(
        payload=payload,
        raw_pose=raw_pose,
        target_pose=target_pose,
        intrinsics=intrinsics,
        evaluation_indices=evaluation_indices,
        frames=frames,
        reference_index=int(payload["reference_sequence_index"]),
        head=head,
        candidate=candidate,
        scene_scale=scene_scale,
        cache_path_value=path,
    )
    print(f"V0 Track-BA candidate decision={output}")


def run_candidate(
    *,
    payload: dict,
    raw_pose: torch.Tensor,
    target_pose: torch.Tensor,
    intrinsics: torch.Tensor,
    evaluation_indices: list[int],
    frames: tuple[int, ...],
    reference_index: int,
    head,
    candidate,
    scene_scale: float,
    cache_path_value: Path,
) -> Path:
    output_dir = candidate.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    method_poses: dict[str, torch.Tensor] = {}
    method_active: dict[str, set[int]] = {}
    method_feasible: dict[str, set[int]] = {}
    for method in candidate.methods:
        predicted = raw_pose.detach().clone().cpu()
        for current in evaluation_indices:
            window = build_track_window(
                payload=payload,
                head=head,
                current_index=current,
                method=method,
                config=candidate,
                raw_world_to_camera=raw_pose,
                intrinsics=intrinsics,
            )
            optimized_relative, diagnostics = optimize_track_window(
                raw_world_to_camera=raw_pose,
                intrinsics=intrinsics,
                window=window,
                scene_scale=scene_scale,
                config=candidate,
            )
            if int(diagnostics["optimized"]):
                predicted[0, current] = compose_relative_candidate(
                    raw_world_to_camera=raw_pose,
                    anchor_index=window.anchor_index,
                    candidate_relative=optimized_relative,
                )
            rows.append(
                {
                    "method": method,
                    "sequence_index": current,
                    "frame_index": frames[current],
                    "anchor_sequence_index": window.anchor_index,
                    "anchor_frame_index": frames[window.anchor_index],
                    "window_sequence_indices": " ".join(
                        str(value) for value in window.sequence_indices
                    ),
                    "window_frames": len(window.sequence_indices),
                    "query_count": int(window.query_points.shape[0]),
                    "region_coverage_fraction": region_coverage(
                        window.region_mask
                    ),
                    "equal_count_region_feasible": int(
                        window.equal_count_region_feasible
                    ),
                    **window.validity_diagnostics,
                    **diagnostics,
                }
            )
            if int(diagnostics["optimized"]):
                method_active.setdefault(method, set()).add(current)
            if window.equal_count_region_feasible:
                method_feasible.setdefault(method, set()).add(current)
        method_poses[method] = predicted
        print(f"V0 Track-BA completed method={method}")
    del head
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # All candidate trajectories are now frozen. GT scoring happens only
    # below and cannot alter support, optimization, fallback or method choice.
    _attach_frame_scores(
        rows,
        raw_pose=raw_pose.cpu(),
        target_pose=target_pose.cpu(),
        method_poses=method_poses,
    )
    fold_rows, decision = evaluate_methods(
        raw_pose=raw_pose.cpu(),
        target_pose=target_pose.cpu(),
        method_poses=method_poses,
        evaluation_indices=evaluation_indices,
        frames=frames,
        reference_index=reference_index,
        primary_method=candidate.primary_method,
        method_active=method_active,
        method_feasible=method_feasible,
    )
    _write_csv(output_dir / "frame_diagnostics.csv", rows)
    _write_csv(output_dir / "validity_audit.csv", rows)
    _write_csv(output_dir / "fold_decision.csv", fold_rows)
    summary = {
        "schema": 1,
        "baseline_version": "v0",
        "implementation_revision": REVISION,
        "candidate_status": "offline_not_selected",
        "claim_level": "frozen_track_correspondence_pose_candidate",
        "cache": str(cache_path_value),
        "clip": payload["clip_name"],
        "frames": frames,
        "evaluation_frames": tuple(frames[index] for index in evaluation_indices),
        "causal_window": True,
        "gt_used_for_candidate_generation": False,
        "gt_used_for_offline_scoring_only": True,
        "pose_model_trained": False,
        "sam_appearance_used": False,
        "sam_role": "dynamic_exclusion_and_spatial_stratification",
        "correspondence_source": "frozen_streamvggt_track_head",
        "geometry_source": "raw_streamvggt_depth_unprojected_in_anchor_camera",
        "pose_initializer_and_prior": "raw_streamvggt",
        "scene_scale_native": scene_scale,
        "primary_method_locked_before_scoring": candidate.primary_method,
        "methods": candidate.methods,
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
            "raw_world_to_camera": raw_pose.cpu(),
            "candidate_world_to_camera": method_poses,
            "primary_method": candidate.primary_method,
            "selected_pose_unchanged": True,
        },
        output_dir / "candidate_poses.pt",
    )
    return result


def run_validity_audit(
    *,
    payload: dict,
    raw_pose: torch.Tensor,
    intrinsics: torch.Tensor,
    evaluation_indices: list[int],
    frames: tuple[int, ...],
    head,
    candidate,
) -> Path:
    """Rerun only enough TrackHead calls to localize a zero-validity gate."""

    candidate.output_dir.mkdir(parents=True, exist_ok=True)
    methods = tuple(dict.fromkeys(("full_image", candidate.primary_method)))
    rows: list[dict[str, object]] = []
    for method in methods:
        for current in evaluation_indices:
            window = build_track_window(
                payload=payload,
                head=head,
                current_index=current,
                method=method,
                config=candidate,
                raw_world_to_camera=raw_pose,
                intrinsics=intrinsics,
            )
            rows.append(
                {
                    "method": method,
                    "sequence_index": current,
                    "frame_index": frames[current],
                    "anchor_sequence_index": window.anchor_index,
                    "anchor_frame_index": frames[window.anchor_index],
                    "window_sequence_indices": " ".join(
                        str(value) for value in window.sequence_indices
                    ),
                    "window_frames": len(window.sequence_indices),
                    "query_count": int(window.query_points.shape[0]),
                    "region_coverage_fraction": region_coverage(
                        window.region_mask
                    ),
                    "equal_count_region_feasible": int(
                        window.equal_count_region_feasible
                    ),
                    **window.validity_diagnostics,
                }
            )
        print(f"V0 Track-BA validity completed method={method}")
    del head
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    output = candidate.output_dir / "validity_audit.csv"
    _write_csv(output, rows)
    return output


def _decode_target_for_scoring(*, payload: dict, pose_decoder) -> torch.Tensor:
    target_encoding = payload.get("target_pose_encoding")
    if not torch.is_tensor(target_encoding):
        raise ValueError("V0 Track-BA cache lacks target_pose_encoding for scoring.")
    target, _ = pose_decoder(
        target_encoding.unsqueeze(0).cpu(),
        image_size_hw=tuple(int(value) for value in payload["image_size"]),
    )
    return target


def _attach_frame_scores(
    rows: list[dict[str, object]],
    *,
    raw_pose: torch.Tensor,
    target_pose: torch.Tensor,
    method_poses: dict[str, torch.Tensor],
) -> None:
    for row in rows:
        method = str(row["method"])
        current = int(row["sequence_index"])
        raw_rotation, raw_center = _frame_errors(
            raw_pose,
            target_pose,
            current,
        )
        candidate_rotation, candidate_center = _frame_errors(
            method_poses[method],
            target_pose,
            current,
        )
        row.update(
            {
                "raw_rotation_error_deg": raw_rotation,
                "candidate_rotation_error_deg": candidate_rotation,
                "raw_center_error_native": raw_center,
                "candidate_center_error_native": candidate_center,
                "rotation_improved": int(candidate_rotation < raw_rotation),
                "center_improved": int(candidate_center < raw_center),
            }
        )


def evaluate_methods(
    *,
    raw_pose: torch.Tensor,
    target_pose: torch.Tensor,
    method_poses: dict[str, torch.Tensor],
    evaluation_indices: list[int],
    frames: tuple[int, ...],
    reference_index: int,
    primary_method: str,
    method_active: dict[str, set[int]],
    method_feasible: dict[str, set[int]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows = []
    passes: dict[str, list[int]] = {method: [] for method in method_poses}
    for method, predicted in method_poses.items():
        for fold_index, fold in enumerate(("short", "medium", "long")):
            indices = evaluation_indices[4 * fold_index : 4 * (fold_index + 1)]
            raw_metrics = pose_metrics(
                raw_pose,
                target_pose,
                reference_index=reference_index,
                evaluation_indices=indices,
            )
            refined_metrics = pose_metrics(
                predicted,
                target_pose,
                reference_index=reference_index,
                evaluation_indices=indices,
            )
            center_worse = 0
            rotation_worse = 0
            active_frames = sum(
                index in method_active.get(method, set()) for index in indices
            )
            feasible_frames = sum(
                index in method_feasible.get(method, set()) for index in indices
            )
            for index in indices:
                raw_rotation, raw_center = _frame_errors(
                    raw_pose,
                    target_pose,
                    index,
                )
                refined_rotation, refined_center = _frame_errors(
                    predicted,
                    target_pose,
                    index,
                )
                center_worse += int(refined_center > raw_center + 1e-8)
                rotation_worse += int(refined_rotation > raw_rotation + 1e-8)
            center_gain = _gain(
                raw_metrics["center_error_native"],
                refined_metrics["center_error_native"],
            )
            rotation_gain = _gain(
                raw_metrics["rotation_degrees"],
                refined_metrics["rotation_degrees"],
            )
            # Mean camera-center is primary. Zero center-worse frames prevents
            # the short/long folds from hiding a local regression again.
            passed = int(
                active_frames == len(indices)
                and feasible_frames == len(indices)
                and center_gain > 0
                and center_worse == 0
            )
            passes[method].append(passed)
            rows.append(
                {
                    "fold": fold,
                    "method": method,
                    "test_frames": " ".join(str(frames[i]) for i in indices),
                    "raw_center_error_native": raw_metrics[
                        "center_error_native"
                    ],
                    "candidate_center_error_native": refined_metrics[
                        "center_error_native"
                    ],
                    "center_gain_percent": center_gain,
                    "active_frames": active_frames,
                    "equal_count_feasible_frames": feasible_frames,
                    "center_worse_frames": center_worse,
                    "raw_rotation_degrees": raw_metrics["rotation_degrees"],
                    "candidate_rotation_degrees": refined_metrics[
                        "rotation_degrees"
                    ],
                    "rotation_gain_percent": rotation_gain,
                    "rotation_worse_frames": rotation_worse,
                    "fold_pass": passed,
                }
            )
    all_fold = {
        method: int(all(values)) for method, values in passes.items()
    }
    primary_pass = int(all_fold[primary_method])
    sam_controls = (
        "sam_dynamic_excluded",
        "sam_instance_background_stratified",
    )
    sam_help_pass = int(
        any(all_fold.get(method, 0) for method in sam_controls)
        and not all_fold.get("full_image", 0)
    )
    return rows, {
        "all_fold_pass_by_method": all_fold,
        "primary_method": primary_method,
        "primary_all_fold_pass": primary_pass,
        "eligible_for_future_v0_revision": primary_pass,
        "current_v0_selected_pose_unchanged": True,
        "sam_mask_unique_pose_help_pass": sam_help_pass,
        "sam_mask_claim_requires_more_than_candidate_pass": True,
        "selection_policy": (
            "locked_primary_requires_positive_center_gain_and_zero_"
            "center_worse_frames_in_each_fold"
        ),
        "controls_are_diagnostic_never_selected_by_gt": True,
    }


def _frame_errors(
    predicted: torch.Tensor,
    target: torch.Tensor,
    index: int,
) -> tuple[float, float]:
    left = predicted[:, index]
    right = target[:, index]
    relative = left[..., :3, :3] @ right[..., :3, :3].transpose(-1, -2)
    cosine = (
        torch.diagonal(relative, dim1=-2, dim2=-1).sum(dim=-1) - 1.0
    ) * 0.5
    rotation = torch.rad2deg(torch.acos(cosine.clamp(-1, 1))).mean()
    center = torch.linalg.vector_norm(
        camera_centers(left) - camera_centers(right),
        dim=-1,
    ).mean()
    return float(rotation), float(center)


def _gain(raw: float, candidate: float) -> float:
    return 100.0 * (float(raw) - float(candidate)) / max(float(raw), 1e-8)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV {path}.")
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="streaming_couping/configs/v0_baseline.yaml",
    )
    parser.add_argument("--device")
    parser.add_argument(
        "--validity-audit-only",
        action="store_true",
        help=(
            "Run full-image and locked-primary TrackHead validity gates only; "
            "do not optimize, decode GT, score, or replace candidate outputs."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
