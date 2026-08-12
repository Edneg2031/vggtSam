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
    landmark_anchor_reprojection_stats,
    select_landmark_source,
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


REVISION = "causal_sift_landmark_history_factor_r3"


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
    if "all_causal" in candidate.history_scopes:
        needed.update(range(0, max(evaluation_indices) + 1))
    else:
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

    branch_specs = _branch_specs(candidate)
    primary_branch = _branch_name(
        candidate.primary_landmark_source,
        candidate.primary_history_scope,
        candidate.primary_method,
    )
    method_poses = {
        branch: raw_pose.detach().clone().cpu()
        for branch in branch_specs
    }
    method_active: dict[str, set[int]] = {}
    method_feasible: dict[str, set[int]] = {}
    rows: list[dict[str, object]] = []
    for current in evaluation_indices:
        for history_scope in candidate.history_scopes:
            pool = build_match_pool(
                payload=payload,
                current_index=current,
                features=features,
                raw_world_to_camera=raw_pose,
                intrinsics=intrinsics,
                config=candidate,
                history_scope=history_scope,
            )
            primary_selected, _, _ = select_method_correspondences(
                pool,
                method=candidate.primary_method,
                count=candidate.max_correspondences,
            )
            locked_count = int(primary_selected.shape[0])
            support_sufficient = locked_count >= candidate.min_correspondences
            start = (
                max(0, current - candidate.anchor_lookback)
                if history_scope == "recent"
                else 0
            )
            for branch, spec in branch_specs.items():
                landmark_source, branch_history, method = spec
                if branch_history != history_scope:
                    continue
                selected, feasible, support = select_method_correspondences(
                    pool,
                    method=method,
                    count=locked_count,
                )
                force_fallback_reason = None
                if not support_sufficient:
                    force_fallback_reason = (
                        "primary_support_below_min_correspondences"
                    )
                elif not feasible:
                    force_fallback_reason = "equal_count_infeasible"
                source_pool = select_landmark_source(pool, landmark_source)
                anchor_reprojection_mean, anchor_reprojection_max = (
                    landmark_anchor_reprojection_stats(
                        pool,
                        source=landmark_source,
                        selected=selected,
                    )
                )
                pose, diagnostics = solve_feature_pnp(
                    pool=source_pool,
                    selected=selected,
                    raw_world_to_camera=raw_pose,
                    intrinsics=intrinsics,
                    current_index=current,
                    scene_scale=scene_scale,
                    config=candidate,
                    force_fallback_reason=force_fallback_reason,
                )
                if int(diagnostics["optimized"]):
                    method_poses[branch][0, current] = pose
                    method_active.setdefault(branch, set()).add(current)
                if feasible and support_sufficient:
                    method_feasible.setdefault(branch, set()).add(current)
                rows.append(
                    {
                        "method": branch,
                        "support_method": method,
                        "landmark_source": landmark_source,
                        "history_scope": history_scope,
                        "is_locked_primary": int(branch == primary_branch),
                        "sequence_index": current,
                        "frame_index": frames[current],
                        "anchor_sequence_indices": " ".join(
                            str(value) for value in range(start, current)
                        ),
                        "anchor_frame_indices": " ".join(
                            str(frames[value]) for value in range(start, current)
                        ),
                        "anchor_count": current - start,
                        "anchor_lookback": candidate.anchor_lookback,
                        "current_sift_features": int(
                            features[current].points.shape[0]
                        ),
                        "pooled_unique_matches": int(pool.current_points.shape[0]),
                        "locked_equal_count": locked_count,
                        "primary_support_sufficient": int(support_sufficient),
                        "equal_count_feasible": int(feasible),
                        "method_available_matches": support["available"],
                        "method_region_matches": support["region"],
                        "method_background_matches": support["background"],
                        "landmark_anchor_reprojection_mean_epe_pixels": (
                            anchor_reprojection_mean
                        ),
                        "landmark_anchor_reprojection_max_epe_pixels": (
                            anchor_reprojection_max
                        ),
                        **diagnostics,
                    }
                )
            print(
                f"V0 Feature-PnP frame={frames[current]} "
                f"history={history_scope} pool={pool.current_points.shape[0]} "
                f"locked_count={locked_count} "
                f"support_sufficient={int(support_sufficient)}"
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
        primary_method=primary_branch,
        method_active=method_active,
        method_feasible=method_feasible,
    )
    full_control_branch = _branch_name(
        candidate.primary_landmark_source,
        candidate.primary_history_scope,
        "full_image",
    )
    all_fold = decision["all_fold_pass_by_method"]
    decision["sam_mask_unique_pose_help_pass"] = int(
        bool(all_fold.get(primary_branch, 0))
        and not bool(all_fold.get(full_control_branch, 0))
    )
    decision["factor_interpretation"] = {
        "all_causal_rescues_recent": (
            "history_coverage_was_the_support_bottleneck"
        ),
        "native_pointmap_beats_raw_depth": (
            "camera_pose_contamination_of_unprojected_landmarks_was_harmful"
        ),
        "native_pointmap_fails_center": (
            "frozen_pointmap_and_sift_pnp_do_not_supply_stable_center_correction"
        ),
    }
    for row in fold_rows:
        landmark_source, history_scope, support_method = branch_specs[
            str(row["method"])
        ]
        row.update(
            {
                "support_method": support_method,
                "landmark_source": landmark_source,
                "history_scope": history_scope,
                "is_locked_primary": int(str(row["method"]) == primary_branch),
            }
        )
    _write_csv(output_dir / "frame_diagnostics.csv", rows)
    _write_csv(output_dir / "fold_decision.csv", fold_rows)
    summary = {
        "schema": 2,
        "baseline_version": "v0",
        "implementation_revision": REVISION,
        "candidate_status": "offline_not_selected",
        "claim_level": "frozen_feature_pnp_landmark_history_factor_candidate",
        "cache": str(cache_path_value),
        "clip": payload["clip_name"],
        "frames": frames,
        "evaluation_frames": tuple(frames[index] for index in evaluation_indices),
        "causal_history_only": True,
        "gt_used_for_candidate_generation": False,
        "gt_used_for_offline_scoring_only": True,
        "pose_model_trained": False,
        "matcher_trained": False,
        "correspondence_source": "opencv_sift_mutual_ratio",
        "landmark_sources": candidate.landmark_sources,
        "history_scopes": candidate.history_scopes,
        "solver": "calibrated_solvePnPRansac_epnp_lm_refine",
        "sam_appearance_used": False,
        "sam_role": "dynamic_exclusion_and_spatial_stratification",
        "primary_method_locked_before_scoring": candidate.primary_method,
        "primary_landmark_source_locked_before_scoring": (
            candidate.primary_landmark_source
        ),
        "primary_history_scope_locked_before_scoring": (
            candidate.primary_history_scope
        ),
        "primary_branch_locked_before_scoring": primary_branch,
        "methods": candidate.methods,
        "evaluated_branches": branch_specs,
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
            "primary_branch": primary_branch,
            "selected_pose_unchanged": True,
        },
        output_dir / "candidate_poses.pt",
    )
    _write_markdown_decision(
        output_dir / "decision.md",
        summary=summary,
        fold_rows=fold_rows,
    )
    return result


def _branch_specs(candidate) -> dict[str, tuple[str, str, str]]:
    specs: dict[str, tuple[str, str, str]] = {}
    for landmark_source in candidate.landmark_sources:
        for history_scope in candidate.history_scopes:
            name = _branch_name(
                landmark_source,
                history_scope,
                candidate.primary_method,
            )
            specs[name] = (
                landmark_source,
                history_scope,
                candidate.primary_method,
            )
    for method in candidate.methods:
        if method == candidate.primary_method:
            continue
        name = _branch_name(
            candidate.primary_landmark_source,
            candidate.primary_history_scope,
            method,
        )
        specs[name] = (
            candidate.primary_landmark_source,
            candidate.primary_history_scope,
            method,
        )
    return specs


def _branch_name(
    landmark_source: str,
    history_scope: str,
    method: str,
) -> str:
    return f"{landmark_source}__{history_scope}__{method}"


def _write_markdown_decision(
    path: Path,
    *,
    summary: dict[str, object],
    fold_rows: list[dict[str, object]],
) -> None:
    decision = summary["decision"]
    lines = [
        "# V0 Feature-PnP landmark/history factor decision",
        "",
        "No matcher or pose model is trained. GT is used only after all causal candidates are frozen.",
        "",
        f"- locked primary: `{summary['primary_branch_locked_before_scoring']}`",
        f"- primary all-fold pass: `{decision['primary_all_fold_pass']}`",
        f"- selected V0 pose changed: `0`",
        "",
        "| landmark | history | support | fold | active | center gain | center worse | rotation gain | pass |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in fold_rows:
        lines.append(
            f"| {row['landmark_source']} | {row['history_scope']} | "
            f"{row['support_method']} | {row['fold']} | "
            f"{row['active_frames']}/4 | "
            f"{float(row['center_gain_percent']):.6g} | "
            f"{row['center_worse_frames']} | "
            f"{float(row['rotation_gain_percent']):.6g} | "
            f"{row['fold_pass']} |"
        )
    lines.extend(
        (
            "",
            "Interpretation:",
            "",
            "- all-causal rescues recent: recent history lacked correspondence coverage.",
            "- native pointmap beats raw-depth unprojection: raw camera bias contaminated landmarks.",
            "- native pointmap still fails center: stop this frozen SIFT-PnP route.",
            "- SAM support can be credited only if the locked SAM branch passes while equal-count full-image does not.",
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
