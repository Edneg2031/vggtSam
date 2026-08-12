#!/usr/bin/env python3
"""Evaluate causal SIFT 2D-3D PnP as an independent V0 pose candidate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from streaming_couping.src.config import load_config
from streaming_couping.src.external_repos import maybe_add_repo_to_path
from streaming_couping.src.learned_pose.baseline_runtime import (
    load_baseline_run_config,
    load_feature_pnp_candidate_config,
)
from streaming_couping.src.learned_pose.cache import cache_path, load_feature_cache
from streaming_couping.src.learned_pose.config import load_learned_pose_config
from streaming_couping.src.v0_feature_pnp import (
    build_match_pool,
    extract_sift_features,
    select_method_correspondences,
    solve_feature_pnp,
)
from streaming_couping.src.v0_track_ba import (
    scene_scale_from_cache,
    validate_track_cache,
)
from streaming_couping.scripts.run_v0_track_ba_candidate import (
    _attach_frame_scores,
    _decode_target_for_scoring,
    evaluate_methods,
)


REVISION = "causal_sift_mutual_depth_pnp_r2"


def main() -> None:
    args = _parse_args()
    data = load_learned_pose_config(args.config)
    baseline = load_baseline_run_config(args.config)
    candidate = load_feature_pnp_candidate_config(args.config)
    if not candidate.enabled:
        raise ValueError("baseline.feature_pnp_candidate.enabled is false.")
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
    print(f"V0 Feature-PnP candidate decision={output}")


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
    needed = set(evaluation_indices)
    for current in evaluation_indices:
        needed.update(
            range(max(0, current - candidate.anchor_lookback), current)
        )
    features = [None] * len(frames)
    for index in sorted(needed):
        features[index] = extract_sift_features(
            payload["stream_images"][index],
            candidate,
        )
        print(
            f"V0 Feature-PnP SIFT frame={frames[index]} "
            f"features={features[index].points.shape[0]}"
        )

    method_poses = {
        method: raw_pose.detach().clone().cpu() for method in candidate.methods
    }
    method_active: dict[str, set[int]] = {}
    method_feasible: dict[str, set[int]] = {}
    rows: list[dict[str, object]] = []
    for current in evaluation_indices:
        pool = build_match_pool(
            payload=payload,
            current_index=current,
            features=features,
            raw_world_to_camera=raw_pose,
            intrinsics=intrinsics,
            config=candidate,
        )
        primary_selected, _, _ = select_method_correspondences(
            pool,
            method=candidate.primary_method,
            count=candidate.max_correspondences,
        )
        locked_count = int(primary_selected.shape[0])
        primary_support_sufficient = locked_count >= candidate.min_correspondences
        for method in candidate.methods:
            selected, feasible, support = select_method_correspondences(
                pool,
                method=method,
                count=locked_count,
            )
            force_fallback_reason = None
            if not primary_support_sufficient:
                force_fallback_reason = "primary_support_below_min_correspondences"
            elif not feasible:
                force_fallback_reason = "equal_count_infeasible"
            pose, diagnostics = solve_feature_pnp(
                pool=pool,
                selected=selected,
                raw_world_to_camera=raw_pose,
                intrinsics=intrinsics,
                current_index=current,
                scene_scale=scene_scale,
                config=candidate,
                force_fallback_reason=force_fallback_reason,
            )
            if int(diagnostics["optimized"]):
                method_poses[method][0, current] = pose
                method_active.setdefault(method, set()).add(current)
            if feasible and locked_count >= candidate.min_correspondences:
                method_feasible.setdefault(method, set()).add(current)
            anchor_start = max(0, current - candidate.anchor_lookback)
            rows.append(
                {
                    "method": method,
                    "sequence_index": current,
                    "frame_index": frames[current],
                    "anchor_sequence_indices": " ".join(
                        str(value) for value in range(anchor_start, current)
                    ),
                    "anchor_frame_indices": " ".join(
                        str(frames[value]) for value in range(anchor_start, current)
                    ),
                    "anchor_lookback": candidate.anchor_lookback,
                    "current_sift_features": int(features[current].points.shape[0]),
                    "pooled_unique_matches": int(pool.current_points.shape[0]),
                    "locked_equal_count": locked_count,
                    "primary_support_sufficient": int(primary_support_sufficient),
                    "equal_count_feasible": int(feasible),
                    "method_available_matches": support["available"],
                    "method_region_matches": support["region"],
                    "method_background_matches": support["background"],
                    **diagnostics,
                }
            )
        print(
            f"V0 Feature-PnP frame={frames[current]} "
            f"pool={pool.current_points.shape[0]} locked_count={locked_count} "
            f"support_sufficient={int(primary_support_sufficient)}"
        )

    # Candidate generation is complete before GT pose is even decoded.
    target_pose = _decode_target_for_scoring(
        payload=payload,
        pose_decoder=pose_decoder,
    )
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
    _write_csv(output_dir / "fold_decision.csv", fold_rows)
    summary = {
        "schema": 1,
        "baseline_version": "v0",
        "implementation_revision": REVISION,
        "candidate_status": "offline_not_selected",
        "claim_level": "frozen_local_feature_depth_pnp_pose_candidate",
        "cache": str(cache_path_value),
        "clip": payload["clip_name"],
        "frames": frames,
        "evaluation_frames": tuple(frames[index] for index in evaluation_indices),
        "causal_window": True,
        "gt_used_for_candidate_generation": False,
        "gt_used_for_offline_scoring_only": True,
        "pose_model_trained": False,
        "matcher_trained": False,
        "correspondence_source": "opencv_sift_mutual_ratio",
        "geometry_source": "raw_streamvggt_anchor_depth_k_pose",
        "solver": "calibrated_solvePnPRansac_epnp_lm_refine",
        "sam_appearance_used": False,
        "sam_role": "dynamic_exclusion_and_spatial_stratification",
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
    return result


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
