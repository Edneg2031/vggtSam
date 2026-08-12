#!/usr/bin/env python3
"""Audit frozen ALIKED-LightGlue correspondences with locked V0 PnP."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from streaming_couping.scripts.run_v0_track_ba_candidate import (
    _attach_frame_scores,
    _decode_target_for_scoring,
    evaluate_methods,
)
from streaming_couping.src.config import load_config
from streaming_couping.src.external_repos import maybe_add_repo_to_path
from streaming_couping.src.learned_pose.baseline_runtime import (
    load_baseline_run_config,
    load_lightglue_pnp_candidate_config,
)
from streaming_couping.src.learned_pose.cache import cache_path, load_feature_cache
from streaming_couping.src.learned_pose.config import load_learned_pose_config
from streaming_couping.src.v0_feature_pnp import (
    select_method_correspondences,
    solve_feature_pnp,
)
from streaming_couping.src.v0_lightglue_pnp import (
    build_lightglue_match_pool,
    extract_aliked_features,
    load_frozen_aliked_lightglue,
    spatial_hull_coverage,
)
from streaming_couping.src.v0_track_ba import (
    scene_scale_from_cache,
    validate_track_cache,
)

REVISION = "frozen_aliked_lightglue_native_all_causal_r7"
LIGHTGLUE_REVISION = "eb42fee2d71449efb0aa5c10549752b5d75384d8"
ALIKED_SHA256 = "5be8704840ed662d9d8c561bf7279c222092674e7eb05fd0feab94899e9d82f2"
LIGHTGLUE_SHA256 = "d975e965b105311a6143194852297dff4f02aea5cc2e10cecfed966ca0e22503"


def main() -> None:
    args = _parse_args()
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    data = load_learned_pose_config(args.config)
    baseline = load_baseline_run_config(args.config)
    candidate = load_lightglue_pnp_candidate_config(args.config)
    if not candidate.enabled:
        raise ValueError("baseline.lightglue_pnp_candidate.enabled is false.")
    clip = next(item for item in data.clips if item.name == baseline.clip_name)
    path = cache_path(data, clip)
    payload = load_feature_cache(path)
    validate_track_cache(payload)
    recovery = load_config(data.recovery_config)
    maybe_add_repo_to_path(recovery.streamvggt_repo)
    from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri

    encoding = payload["baseline_pose_encoding"].unsqueeze(0).cpu()
    raw_pose, intrinsics = pose_encoding_to_extri_intri(
        encoding,
        image_size_hw=tuple(int(value) for value in payload["image_size"]),
    )
    frames = tuple(int(value) for value in payload["frame_indices"])
    positions = {frame: index for index, frame in enumerate(frames)}
    evaluation_indices = [positions[frame] for frame in baseline.evaluation_frames]
    scene_scale = scene_scale_from_cache(
        payload,
        candidate.point_confidence_threshold,
    )
    output = run_candidate(
        payload=payload,
        raw_pose=raw_pose,
        intrinsics=intrinsics,
        evaluation_indices=evaluation_indices,
        frames=frames,
        reference_index=int(payload["reference_sequence_index"]),
        candidate=candidate,
        scene_scale=scene_scale,
        cache_path_value=path,
        pose_decoder=pose_encoding_to_extri_intri,
    )
    print(f"V0 LightGlue-PnP candidate decision={output}")


def run_candidate(
    *,
    payload: dict,
    raw_pose: torch.Tensor,
    intrinsics: torch.Tensor,
    evaluation_indices: list[int],
    frames: tuple[int, ...],
    reference_index: int,
    candidate,
    scene_scale: float,
    cache_path_value: Path,
    pose_decoder,
) -> Path:
    output_dir = candidate.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _validated_device(candidate.device)
    extractor, matcher = load_frozen_aliked_lightglue(
        repo_path=candidate.lightglue_repo,
        device=device,
        nfeatures=candidate.nfeatures,
        detection_threshold=candidate.detection_threshold,
    )
    needed = range(max(evaluation_indices) + 1)
    features = [None] * len(frames)
    for index in needed:
        features[index] = extract_aliked_features(
            payload["stream_images"][index],
            extractor=extractor,
            device=device,
        )
        print(
            f"V0 LightGlue-PnP ALIKED frame={frames[index]} "
            f"features={features[index].points.shape[0]}"
        )

    method_poses = {
        method: raw_pose.detach().clone().cpu()
        for method in candidate.methods
    }
    method_active: dict[str, set[int]] = {}
    method_feasible: dict[str, set[int]] = {}
    frame_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    height, width = tuple(int(value) for value in payload["stream_images"].shape[-2:])
    for current in evaluation_indices:
        pool, current_pair_rows = build_lightglue_match_pool(
            payload=payload,
            current_index=current,
            features=features,
            matcher=matcher,
            device=device,
            config=candidate,
        )
        for row in current_pair_rows:
            row.update(
                {
                    "anchor_frame_index": frames[int(row["anchor_sequence_index"])],
                    "current_frame_index": frames[current],
                }
            )
        pair_rows.extend(current_pair_rows)
        primary_available, _, _ = select_method_correspondences(
            pool,
            method=candidate.primary_method,
            count=candidate.max_correspondences,
        )
        locked_count = int(primary_available.shape[0])
        support_sufficient = locked_count >= int(candidate.min_correspondences)
        pair_raw = [int(row["raw_pair_matches"]) for row in current_pair_rows]
        pair_valid = [
            int(row["geometry_valid_pair_matches"])
            for row in current_pair_rows
        ]
        for method in candidate.methods:
            selected, feasible, support = select_method_correspondences(
                pool,
                method=method,
                count=locked_count,
            )
            fallback = None
            if not support_sufficient:
                fallback = "primary_support_below_min_correspondences"
            elif not feasible:
                fallback = "equal_count_infeasible"
            pose, diagnostics = solve_feature_pnp(
                pool=pool,
                selected=selected,
                raw_world_to_camera=raw_pose,
                intrinsics=intrinsics,
                current_index=current,
                scene_scale=scene_scale,
                config=candidate,
                force_fallback_reason=fallback,
            )
            if int(diagnostics["optimized"]):
                method_poses[method][0, current] = pose
                method_active.setdefault(method, set()).add(current)
            if feasible and support_sufficient:
                method_feasible.setdefault(method, set()).add(current)
            selected_scores = (
                1.0 - pool.distances.index_select(0, selected)
                if selected.shape[0]
                else torch.empty(0)
            )
            frame_rows.append(
                {
                    "method": method,
                    "sequence_index": current,
                    "frame_index": frames[current],
                    "history_frames": current,
                    "current_aliked_features": int(features[current].points.shape[0]),
                    "raw_pair_matches_min": min(pair_raw, default=0),
                    "raw_pair_matches_mean": _mean(pair_raw),
                    "raw_pair_matches_max": max(pair_raw, default=0),
                    "geometry_valid_pair_matches_min": min(pair_valid, default=0),
                    "geometry_valid_pair_matches_mean": _mean(pair_valid),
                    "geometry_valid_pair_matches_max": max(pair_valid, default=0),
                    "pooled_unique_current_matches": int(pool.current_points.shape[0]),
                    "locked_equal_count": locked_count,
                    "primary_support_sufficient": int(support_sufficient),
                    "equal_count_feasible": int(feasible),
                    "method_available_matches": support["available"],
                    "selected_match_score_mean": (
                        float(selected_scores.mean())
                        if selected_scores.numel()
                        else float("nan")
                    ),
                    "current_selected_hull_coverage_fraction": spatial_hull_coverage(
                        pool.current_points.index_select(0, selected),
                        image_size=(height, width),
                    ),
                    **diagnostics,
                }
            )
        print(
            f"V0 LightGlue-PnP frame={frames[current]} "
            f"history={current} pooled_unique={pool.current_points.shape[0]} "
            f"locked_count={locked_count} "
            f"support_sufficient={int(support_sufficient)}"
        )

    # Candidate construction is complete before target pose is decoded.
    del matcher, extractor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    target_pose = _decode_target_for_scoring(
        payload=payload,
        pose_decoder=pose_decoder,
    )
    _attach_frame_scores(
        frame_rows,
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
    all_fold = decision["all_fold_pass_by_method"]
    decision["sam_mask_unique_pose_help_pass"] = int(
        bool(all_fold.get(candidate.primary_method, 0))
        and not bool(all_fold.get("full_image", 0))
    )
    _write_csv(output_dir / "pair_diagnostics.csv", pair_rows)
    _write_csv(output_dir / "frame_diagnostics.csv", frame_rows)
    _write_csv(output_dir / "fold_decision.csv", fold_rows)
    summary = {
        "schema": 1,
        "baseline_version": "v0",
        "implementation_revision": REVISION,
        "candidate_status": "offline_not_selected",
        "claim_level": "frozen_correspondence_pnp_candidate",
        "cache": str(cache_path_value),
        "clip": payload["clip_name"],
        "frames": frames,
        "evaluation_frames": tuple(frames[index] for index in evaluation_indices),
        "causal_history_only": True,
        "gt_used_for_candidate_generation": False,
        "gt_used_for_offline_scoring_only": True,
        "pose_model_trained": False,
        "matcher_trained": False,
        "correspondence_source": "frozen_aliked_n16_plus_aliked_lightglue",
        "lightglue_repository_revision": LIGHTGLUE_REVISION,
        "aliked_weight_sha256": ALIKED_SHA256,
        "aliked_lightglue_weight_sha256": LIGHTGLUE_SHA256,
        "landmark_source": "native_streamvggt_world_pointmap",
        "history_scope": "all_causal",
        "solver": "calibrated_solvePnPRansac_epnp_lm_refine",
        "sam_appearance_used": False,
        "sam_role": "dynamic_exclusion_only",
        "primary_method_locked_before_scoring": candidate.primary_method,
        "methods": candidate.methods,
        "scene_scale_native": scene_scale,
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
    _write_markdown(output_dir / "decision.md", summary, fold_rows)
    return result


def _validated_device(device: str) -> str:
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("V0 LightGlue-PnP requested CUDA but CUDA is unavailable.")
    return str(device)


def _mean(values: list[int]) -> float:
    return sum(values) / max(len(values), 1)


def _write_markdown(
    path: Path,
    summary: dict[str, object],
    rows: list[dict[str, object]],
) -> None:
    decision = summary["decision"]
    lines = [
        "# V0 frozen ALIKED-LightGlue correspondence decision",
        "",
        "No matcher or pose model is trained. GT is offline scoring-only.",
        "",
        f"- locked primary: `{summary['primary_method_locked_before_scoring']}`",
        f"- primary all-fold pass: `{decision['primary_all_fold_pass']}`",
        "- selected V0 pose changed: `0`",
        "",
        "| fold | method | active | equal count | center gain | center worse | rotation gain | pass |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['fold']} | {row['method']} | {row['active_frames']}/4 | "
            f"{row['equal_count_feasible_frames']}/4 | "
            f"{float(row['center_gain_percent']):.6g} | "
            f"{row['center_worse_frames']} | "
            f"{float(row['rotation_gain_percent']):.6g} | {row['fold_pass']} |"
        )
    lines.extend(
        (
            "",
            "Interpretation:",
            "",
            "- coverage succeeds but pose fails: correspondence quantity was not the remaining bottleneck.",
            "- full-image and SAM-excluded both fail: do not tune this PnP route again.",
            "- SAM-excluded passes while equal-count full-image fails: SAM mask gating has candidate-level evidence only.",
            "- A pass remains one-scene offline evidence and does not automatically modify V0.",
        )
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV {path}.")
    fieldnames = tuple(dict.fromkeys(name for row in rows for name in row))
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
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
