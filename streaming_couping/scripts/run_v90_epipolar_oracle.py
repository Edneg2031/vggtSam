#!/usr/bin/env python3
"""Run the V9 Stage-O visible-surface 2D epipolar oracle.

No network is trained.  SAM3.1 and StreamVGGT remain frozen and unloaded; the
runner consumes their retained V7.4 cache.  GT mesh/pose data is used only to
construct visible 2D correspondences and to score the fixed solver.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch
import yaml

from streaming_couping.src.config import load_config
from streaming_couping.src.external_repos import maybe_add_repo_to_path
from streaming_couping.src.learned_pose.cache import cache_path, load_feature_cache
from streaming_couping.src.learned_pose.config import load_learned_pose_config
from streaming_couping.src.v74_temporal_protocol import (
    EXPECTED_FRAMES,
    FOLDS,
    validate_folds,
)
from streaming_couping.src.v80_pose_geometry import (
    camera_centers,
    homogeneous,
    invert_rigid,
    rotation_error_degrees,
)
from streaming_couping.src.v90_epipolar_geometry import (
    EpipolarConfig,
    SurfaceCorrespondences,
    VisibilityConfig,
    causal_mask_history_indices,
    concatenate_correspondences,
    estimate_relative_epipolar_pose,
    recover_absolute_pose,
    relative_translation_direction_error_degrees,
    surface_reprojection_correspondences,
    translation_direction_error_degrees,
)


FRAME_COLUMNS = (
    "fold",
    "sequence_index",
    "frame_index",
    "test_position",
    "observed_slots",
    "mature_slots",
    "history_edges_attempted",
    "history_edges_solved",
    "participating_slots",
    "visible_correspondences",
    "effective_correspondences",
    "active",
    "fallback_reason",
    "ray_intersection_used",
    "raw_rotation_error_deg",
    "refined_rotation_error_deg",
    "rotation_improvement_deg",
    "raw_translation_direction_error_deg",
    "refined_translation_direction_error_deg",
    "translation_direction_improvement_deg",
    "raw_center_error_native",
    "refined_center_error_native",
    "center_improvement_native",
    "raw_aggregate_pose_deg",
    "refined_aggregate_pose_deg",
    "aggregate_improvement_deg",
    "raw_edge_rotation_error_deg",
    "oracle_edge_rotation_error_deg",
    "raw_edge_translation_direction_error_deg",
    "oracle_edge_translation_direction_error_deg",
    "mean_sampson_rmse",
    "mean_cheirality_fraction",
    "mean_design_condition",
    "fallback_exact",
    "rotation_improved",
    "translation_direction_improved",
    "aggregate_improved",
)

DIAGNOSTIC_COLUMNS = (
    "fold",
    "sequence_index",
    "frame_index",
    "history_sequence_index",
    "history_frame_index",
    "row_type",
    "slot",
    "sam_track_id",
    "sam_prompt",
    "sampled_queries",
    "projected_in_bounds",
    "visible_correspondences",
    "visibility_rate",
    "mean_depth_residual_metric",
    "p90_depth_residual_metric",
    "aggregated_slots",
    "effective_correspondences",
    "design_rank_ratio",
    "design_condition",
    "sampson_rmse",
    "inlier_ratio",
    "cheirality_fraction",
    "refinement_iterations",
    "edge_success",
    "edge_reason",
)

SUMMARY_COLUMNS = (
    "fold",
    "oracle_level",
    "correspondence_source",
    "pose_history_source",
    "intrinsics_source",
    "test_frames",
    "frames",
    "active_frames",
    "inactive_frames",
    "mean_visible_correspondences",
    "mean_effective_correspondences",
    "raw_rotation_error_deg",
    "refined_rotation_error_deg",
    "rotation_improvement_deg",
    "rotation_gain_percent",
    "raw_translation_direction_error_deg",
    "refined_translation_direction_error_deg",
    "translation_direction_improvement_deg",
    "translation_direction_gain_percent",
    "raw_center_error_native",
    "refined_center_error_native",
    "center_improvement_native",
    "raw_aggregate_pose_deg",
    "refined_aggregate_pose_deg",
    "aggregate_improvement_deg",
    "aggregate_gain_percent",
    "active_rotation_improved_frames",
    "active_translation_direction_improved_frames",
    "active_aggregate_improved_frames",
    "active_aggregate_worse_frames",
    "fallback_exact",
    "reference_exact",
    "fold_oracle_pass",
    "all_folds_oracle_pass",
)


@dataclass(frozen=True)
class V90Config:
    source_path: Path
    data_config: Path
    output_dir: Path
    clip_name: str
    max_history: int
    visibility: VisibilityConfig
    epipolar: EpipolarConfig


def main() -> None:
    args = _parse_args()
    config = load_v90_config(args.config)
    if args.output_dir:
        config = replace(
            config,
            output_dir=Path(args.output_dir).expanduser().resolve(),
        )
    result = run_v90_oracle(config)
    print(f"V9 Stage-O result={result}")


def run_v90_oracle(config: V90Config) -> Path:
    data = load_learned_pose_config(config.data_config)
    clip = next((item for item in data.clips if item.name == config.clip_name), None)
    if clip is None:
        raise ValueError(f"V9 clip={config.clip_name!r} is not configured.")
    path = cache_path(data, clip)
    if not path.is_file():
        raise FileNotFoundError(
            "V9 Stage O reuses the V7.4 observation cache and never silently "
            f"runs a backbone. Missing cache: {path}"
        )
    payload = load_feature_cache(path)
    frames = tuple(int(value) for value in payload["frame_indices"])
    if frames != EXPECTED_FRAMES:
        raise ValueError(f"V9 requires frames 90:15:525, got {frames}.")
    validate_folds(FOLDS, available_frames=set(frames))
    if int(payload["reference_sequence_index"]) != 0:
        raise ValueError("V9 requires frame 90 as the camera gauge.")
    _validate_payload(payload)

    recovery = load_config(data.recovery_config)
    maybe_add_repo_to_path(recovery.streamvggt_repo)
    from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri

    image_size = tuple(int(value) for value in payload["image_size"])
    baseline, _ = pose_encoding_to_extri_intri(
        torch.as_tensor(payload["baseline_pose_encoding"])[None].float(),
        image_size_hw=image_size,
    )
    target, gt_intrinsics = pose_encoding_to_extri_intri(
        torch.as_tensor(payload["target_pose_encoding"])[None].float(),
        image_size_hw=image_size,
    )
    baseline = homogeneous(baseline[0].double().cpu())
    target = homogeneous(target[0].double().cpu())
    gt_intrinsics = gt_intrinsics[0].double().cpu()
    masks = torch.as_tensor(payload["tracking_masks_stream"]).bool().cpu()
    world_points = torch.as_tensor(payload["target_world_points"]).double().cpu()
    depth = torch.as_tensor(payload["target_depth"]).double().cpu()
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    global_w2c = homogeneous(
        torch.as_tensor(payload["target_world_to_camera"]).double().cpu()
    )
    _validate_decoded_tensors(
        masks=masks,
        world_points=world_points,
        depth=depth,
        global_w2c=global_w2c,
        intrinsics=gt_intrinsics,
        baseline=baseline,
        target=target,
    )
    history_bank = causal_mask_history_indices(
        masks, max_history=config.max_history
    )
    positions = {frame: index for index, frame in enumerate(frames)}
    sam_track_ids = tuple(int(value) for value in payload.get("sam_track_ids", ()))
    sam_prompts = tuple(str(value) for value in payload.get("sam_track_prompts", ()))

    frame_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    for fold in FOLDS:
        for test_position, frame_index in enumerate(fold.test_frames):
            current = positions[int(frame_index)]
            frame_row, diagnostics = _evaluate_frame(
                fold=fold.name,
                test_position=test_position,
                current=current,
                frames=frames,
                masks=masks,
                history_bank=history_bank,
                world_points=world_points,
                depth=depth,
                global_w2c=global_w2c,
                intrinsics=gt_intrinsics,
                baseline=baseline,
                target=target,
                sam_track_ids=sam_track_ids,
                sam_prompts=sam_prompts,
                config=config,
            )
            frame_rows.append(frame_row)
            diagnostic_rows.extend(diagnostics)

    summary_rows = _summarize(frame_rows, baseline=baseline)
    all_pass = int(
        len(summary_rows) == len(FOLDS)
        and all(int(row["fold_oracle_pass"]) for row in summary_rows)
    )
    for row in summary_rows:
        row["all_folds_oracle_pass"] = all_pass

    config.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = config.output_dir / "v90_oracle_summary.csv"
    frames_path = config.output_dir / "v90_oracle_frames.csv"
    diagnostics_path = config.output_dir / "v90_correspondence_diagnostics.csv"
    _write_csv(summary_path, summary_rows, SUMMARY_COLUMNS)
    _write_csv(frames_path, frame_rows, FRAME_COLUMNS)
    _write_csv(
        diagnostics_path,
        diagnostic_rows,
        DIAGNOSTIC_COLUMNS,
        allow_empty=True,
    )
    (config.output_dir / "v90_decision.md").write_text(
        _decision_markdown(summary_rows), encoding="utf8"
    )
    _write_metadata(
        config.output_dir / "v90_metadata.json",
        config=config,
        cache_path_value=path,
        payload=payload,
        frames=frames,
        rows=frame_rows,
    )
    return summary_path


def _evaluate_frame(
    *,
    fold: str,
    test_position: int,
    current: int,
    frames: tuple[int, ...],
    masks: torch.Tensor,
    history_bank: torch.Tensor,
    world_points: torch.Tensor,
    depth: torch.Tensor,
    global_w2c: torch.Tensor,
    intrinsics: torch.Tensor,
    baseline: torch.Tensor,
    target: torch.Tensor,
    sam_track_ids: tuple[int, ...],
    sam_prompts: tuple[str, ...],
    config: V90Config,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    by_history: dict[int, list[SurfaceCorrespondences]] = defaultdict(list)
    diagnostics: list[dict[str, object]] = []
    observed = masks[current].flatten(1).any(dim=-1)
    mature_slots = 0
    for slot in range(masks.shape[1]):
        if not bool(observed[slot]):
            continue
        histories = [
            int(value)
            for value in history_bank[current, slot].tolist()
            if int(value) >= 0
        ]
        if histories:
            mature_slots += 1
        for history in histories:
            pair = surface_reprojection_correspondences(
                current_frame=current,
                history_frame=history,
                slot=slot,
                masks=masks,
                world_points_metric=world_points,
                depth_metric=depth,
                global_world_to_camera=global_w2c,
                intrinsics=intrinsics,
                config=config.visibility,
            )
            by_history[history].append(pair)
            diagnostics.append(
                _surface_diagnostic(
                    fold=fold,
                    frame_index=frames[current],
                    history_frame_index=frames[history],
                    pair=pair,
                    sam_track_ids=sam_track_ids,
                    sam_prompts=sam_prompts,
                )
            )

    edge_histories: list[int] = []
    edge_estimates = []
    edge_target_relative = []
    edge_l0_relative = []
    total_visible = 0
    participating_slots: set[int] = set()
    for history in sorted(by_history):
        pairs = by_history[history]
        current_uv, history_uv, weights = concatenate_correspondences(pairs)
        total_visible += int(current_uv.shape[0])
        participating_slots.update(pair.slot for pair in pairs if pair.count)
        l0_relative = baseline[history] @ invert_rigid(baseline[current])
        target_relative = target[history] @ invert_rigid(target[current])
        estimate = estimate_relative_epipolar_pose(
            current_uv,
            history_uv,
            weights,
            intrinsics[current],
            intrinsics[history],
            l0_relative,
            config=config.epipolar,
        )
        edge_histories.append(history)
        edge_estimates.append(estimate)
        edge_target_relative.append(target_relative)
        edge_l0_relative.append(l0_relative)
        diagnostics.append(
            _edge_diagnostic(
                fold=fold,
                sequence_index=current,
                frame_index=frames[current],
                history=history,
                history_frame_index=frames[history],
                pairs=pairs,
                estimate=estimate,
            )
        )

    absolute = recover_absolute_pose(
        current_index=current,
        baseline_world_to_camera=baseline,
        edge_history_indices=edge_histories,
        edge_estimates=edge_estimates,
        config=config.epipolar,
    )
    refined = absolute.world_to_camera
    fallback_exact = int(
        absolute.success or torch.equal(refined, baseline[current])
    )
    raw_rotation = float(rotation_error_degrees(baseline[current], target[current]))
    refined_rotation = float(rotation_error_degrees(refined, target[current]))
    raw_direction = _trajectory_direction_error(
        baseline, target, current=current
    )
    refined_sequence = baseline.clone()
    refined_sequence[current] = refined
    refined_direction = _trajectory_direction_error(
        refined_sequence, target, current=current
    )
    raw_centers = camera_centers(baseline)
    target_centers = camera_centers(target)
    refined_center = camera_centers(refined)
    raw_center_error = float(
        torch.linalg.vector_norm(raw_centers[current] - target_centers[current])
    )
    refined_center_error = float(
        torch.linalg.vector_norm(refined_center - target_centers[current])
    )
    raw_edge_rotation, oracle_edge_rotation = _edge_rotation_errors(
        edge_estimates, edge_l0_relative, edge_target_relative
    )
    raw_edge_direction, oracle_edge_direction = _edge_direction_errors(
        edge_estimates, edge_l0_relative, edge_target_relative
    )
    solved = [estimate for estimate in edge_estimates if estimate.success]
    raw_aggregate = raw_rotation + raw_direction
    refined_aggregate = refined_rotation + refined_direction
    row = {
        "fold": fold,
        "sequence_index": current,
        "frame_index": int(frames[current]),
        "test_position": int(test_position),
        "observed_slots": int(observed.sum()),
        "mature_slots": int(mature_slots),
        "history_edges_attempted": len(edge_estimates),
        "history_edges_solved": len(solved),
        "participating_slots": len(participating_slots),
        "visible_correspondences": total_visible,
        "effective_correspondences": _sum(
            estimate.effective_correspondences for estimate in solved
        ),
        "active": int(absolute.success),
        "fallback_reason": absolute.reason,
        "ray_intersection_used": int(absolute.used_ray_intersection),
        "raw_rotation_error_deg": raw_rotation,
        "refined_rotation_error_deg": refined_rotation,
        "rotation_improvement_deg": raw_rotation - refined_rotation,
        "raw_translation_direction_error_deg": raw_direction,
        "refined_translation_direction_error_deg": refined_direction,
        "translation_direction_improvement_deg": raw_direction - refined_direction,
        "raw_center_error_native": raw_center_error,
        "refined_center_error_native": refined_center_error,
        "center_improvement_native": raw_center_error - refined_center_error,
        "raw_aggregate_pose_deg": raw_aggregate,
        "refined_aggregate_pose_deg": refined_aggregate,
        "aggregate_improvement_deg": raw_aggregate - refined_aggregate,
        "raw_edge_rotation_error_deg": raw_edge_rotation,
        "oracle_edge_rotation_error_deg": oracle_edge_rotation,
        "raw_edge_translation_direction_error_deg": raw_edge_direction,
        "oracle_edge_translation_direction_error_deg": oracle_edge_direction,
        "mean_sampson_rmse": _finite_mean(
            estimate.sampson_rmse for estimate in solved
        ),
        "mean_cheirality_fraction": _finite_mean(
            estimate.cheirality_fraction for estimate in solved
        ),
        "mean_design_condition": _finite_mean(
            estimate.design_condition for estimate in solved
        ),
        "fallback_exact": fallback_exact,
        "rotation_improved": int(absolute.success and refined_rotation < raw_rotation),
        "translation_direction_improved": int(
            absolute.success and refined_direction < raw_direction
        ),
        "aggregate_improved": int(
            absolute.success and refined_aggregate < raw_aggregate
        ),
    }
    return row, diagnostics


def _surface_diagnostic(
    *,
    fold: str,
    frame_index: int,
    history_frame_index: int,
    pair: SurfaceCorrespondences,
    sam_track_ids: tuple[int, ...],
    sam_prompts: tuple[str, ...],
) -> dict[str, object]:
    return {
        "fold": fold,
        "sequence_index": pair.current_frame,
        "frame_index": int(frame_index),
        "history_sequence_index": pair.history_frame,
        "history_frame_index": int(history_frame_index),
        "row_type": "slot_surface_reprojection",
        "slot": pair.slot,
        "sam_track_id": sam_track_ids[pair.slot] if pair.slot < len(sam_track_ids) else -1,
        "sam_prompt": sam_prompts[pair.slot] if pair.slot < len(sam_prompts) else "",
        "sampled_queries": pair.sampled_queries,
        "projected_in_bounds": pair.projected_in_bounds,
        "visible_correspondences": pair.count,
        "visibility_rate": _ratio(pair.count, pair.sampled_queries),
        "mean_depth_residual_metric": _finite_mean(pair.depth_residual_metric.tolist()),
        "p90_depth_residual_metric": _quantile(pair.depth_residual_metric, 0.90),
        "aggregated_slots": 1,
        "effective_correspondences": _effective_count(pair.weights),
        "design_rank_ratio": "",
        "design_condition": "",
        "sampson_rmse": "",
        "inlier_ratio": "",
        "cheirality_fraction": "",
        "refinement_iterations": "",
        "edge_success": "",
        "edge_reason": "",
    }


def _edge_diagnostic(
    *,
    fold: str,
    sequence_index: int,
    frame_index: int,
    history: int,
    history_frame_index: int,
    pairs: list[SurfaceCorrespondences],
    estimate,
) -> dict[str, object]:
    sampled = sum(pair.sampled_queries for pair in pairs)
    projected = sum(pair.projected_in_bounds for pair in pairs)
    visible = sum(pair.count for pair in pairs)
    residuals = [pair.depth_residual_metric for pair in pairs if pair.count]
    depth_residual = (
        torch.cat(residuals) if residuals else torch.empty(0, dtype=torch.float64)
    )
    return {
        "fold": fold,
        "sequence_index": sequence_index,
        "frame_index": frame_index,
        "history_sequence_index": history,
        "history_frame_index": history_frame_index,
        "row_type": "history_edge_aggregate",
        "slot": -1,
        "sam_track_id": -1,
        "sam_prompt": "",
        "sampled_queries": sampled,
        "projected_in_bounds": projected,
        "visible_correspondences": visible,
        "visibility_rate": _ratio(visible, sampled),
        "mean_depth_residual_metric": _finite_mean(depth_residual.tolist()),
        "p90_depth_residual_metric": _quantile(depth_residual, 0.90),
        "aggregated_slots": sum(int(pair.count > 0) for pair in pairs),
        "effective_correspondences": estimate.effective_correspondences,
        "design_rank_ratio": estimate.design_rank_ratio,
        "design_condition": estimate.design_condition,
        "sampson_rmse": estimate.sampson_rmse,
        "inlier_ratio": estimate.inlier_ratio,
        "cheirality_fraction": estimate.cheirality_fraction,
        "refinement_iterations": estimate.refinement_iterations,
        "edge_success": int(estimate.success),
        "edge_reason": estimate.reason,
    }


def _summarize(
    frame_rows: Sequence[dict[str, object]], *, baseline: torch.Tensor
) -> list[dict[str, object]]:
    rows = []
    reference_exact = int(torch.equal(baseline[0], baseline[0].clone()))
    for fold in FOLDS:
        group = [row for row in frame_rows if row["fold"] == fold.name]
        active = [row for row in group if int(row["active"])]
        raw_rotation = _mean(float(row["raw_rotation_error_deg"]) for row in group)
        refined_rotation = _mean(
            float(row["refined_rotation_error_deg"]) for row in group
        )
        raw_direction = _mean(
            float(row["raw_translation_direction_error_deg"]) for row in group
        )
        refined_direction = _mean(
            float(row["refined_translation_direction_error_deg"]) for row in group
        )
        raw_center = _mean(float(row["raw_center_error_native"]) for row in group)
        refined_center = _mean(
            float(row["refined_center_error_native"]) for row in group
        )
        raw_aggregate = _mean(
            float(row["raw_aggregate_pose_deg"]) for row in group
        )
        refined_aggregate = _mean(
            float(row["refined_aggregate_pose_deg"]) for row in group
        )
        fallback_exact = int(all(int(row["fallback_exact"]) for row in group))
        worse = sum(
            int(
                int(row["active"])
                and float(row["refined_aggregate_pose_deg"])
                > float(row["raw_aggregate_pose_deg"]) + 1e-12
            )
            for row in group
        )
        passed = int(
            bool(active)
            and refined_rotation < raw_rotation
            and refined_direction < raw_direction
            and refined_aggregate < raw_aggregate
            and worse == 0
            and fallback_exact == 1
            and reference_exact == 1
        )
        rows.append(
            {
                "fold": fold.name,
                "oracle_level": "V9-O",
                "correspondence_source": "gt_visible_surface_reprojection",
                "pose_history_source": "frozen_streamvggt_l0",
                "intrinsics_source": "calibrated_gt_k",
                "test_frames": " ".join(str(value) for value in fold.test_frames),
                "frames": len(group),
                "active_frames": len(active),
                "inactive_frames": len(group) - len(active),
                "mean_visible_correspondences": _mean(
                    float(row["visible_correspondences"]) for row in group
                ),
                "mean_effective_correspondences": _mean(
                    float(row["effective_correspondences"]) for row in group
                ),
                "raw_rotation_error_deg": raw_rotation,
                "refined_rotation_error_deg": refined_rotation,
                "rotation_improvement_deg": raw_rotation - refined_rotation,
                "rotation_gain_percent": _gain(raw_rotation, refined_rotation),
                "raw_translation_direction_error_deg": raw_direction,
                "refined_translation_direction_error_deg": refined_direction,
                "translation_direction_improvement_deg": raw_direction - refined_direction,
                "translation_direction_gain_percent": _gain(raw_direction, refined_direction),
                "raw_center_error_native": raw_center,
                "refined_center_error_native": refined_center,
                "center_improvement_native": raw_center - refined_center,
                "raw_aggregate_pose_deg": raw_aggregate,
                "refined_aggregate_pose_deg": refined_aggregate,
                "aggregate_improvement_deg": raw_aggregate - refined_aggregate,
                "aggregate_gain_percent": _gain(raw_aggregate, refined_aggregate),
                "active_rotation_improved_frames": sum(
                    int(row["rotation_improved"]) for row in group
                ),
                "active_translation_direction_improved_frames": sum(
                    int(row["translation_direction_improved"]) for row in group
                ),
                "active_aggregate_improved_frames": sum(
                    int(row["aggregate_improved"]) for row in group
                ),
                "active_aggregate_worse_frames": worse,
                "fallback_exact": fallback_exact,
                "reference_exact": reference_exact,
                "fold_oracle_pass": passed,
                "all_folds_oracle_pass": 0,
            }
        )
    return rows


def _decision_markdown(rows: Sequence[dict[str, object]]) -> str:
    all_pass = int(bool(rows) and all(int(row["fold_oracle_pass"]) for row in rows))
    lines = [
        "# V9 Stage-O 2D epipolar oracle decision",
        "",
        "No model is trained. GT mesh/pose is used only for visible 2D correspondence labels and scoring.",
        "Pose recovery uses calibrated K, a fixed epipolar solver and frozen StreamVGGT L0 history.",
        "",
        f"- all-fold visible-surface oracle pass: `{all_pass}`",
        "",
        "| fold | active | rotation gain | direction gain | worse frames | pass |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['fold']} | {row['active_frames']}/{row['frames']} "
            f"| {float(row['rotation_gain_percent']):.6g} "
            f"| {float(row['translation_direction_gain_percent']):.6g} "
            f"| {row['active_aggregate_worse_frames']} "
            f"| {row['fold_oracle_pass']} |"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- pass=0: stop V9 before training a SAM matcher; inspect 2D support/epipolar degeneracy.",
            "- all-fold pass=1: Stage A may compare SAM local descriptors with UV and StreamVGGT appearance controls.",
            "- Center error is secondary because a monocular essential matrix does not recover metric edge scale.",
            "",
        ]
    )
    return "\n".join(lines)


def load_v90_config(path: str | Path) -> V90Config:
    source = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf8")) or {}
    visibility_raw = raw.get("visibility", {})
    solver_raw = raw.get("solver", {})
    repository = source.parents[2]
    data_config = Path(
        raw.get("data_config", "streaming_couping/configs/v74_temporal_data.yaml")
    )
    if not data_config.is_absolute():
        data_config = (repository / data_config).resolve()
    output = Path(
        raw.get(
            "output_dir",
            "outputs/streaming_couping_v90_epipolar_token_causality",
        )
    )
    if not output.is_absolute():
        output = (repository / output).resolve()
    config = V90Config(
        source_path=source,
        data_config=data_config,
        output_dir=output,
        clip_name=str(
            raw.get("clip_name", "00a231a370_90_525_step15_37_68_54")
        ),
        max_history=int(raw.get("max_history", 2)),
        visibility=VisibilityConfig(
            max_queries_per_instance=int(
                visibility_raw.get("max_queries_per_instance", 256)
            ),
            depth_tolerance_metric=float(
                visibility_raw.get("depth_tolerance_metric", 0.03)
            ),
            relative_depth_tolerance=float(
                visibility_raw.get("relative_depth_tolerance", 0.01)
            ),
        ),
        epipolar=EpipolarConfig(
            min_correspondences=int(solver_raw.get("min_correspondences", 8)),
            min_design_rank_ratio=float(
                solver_raw.get("min_design_rank_ratio", 1e-7)
            ),
            min_cheirality_fraction=float(
                solver_raw.get("min_cheirality_fraction", 0.50)
            ),
            refinement_iterations=int(
                solver_raw.get("refinement_iterations", 5)
            ),
            refinement_huber_delta=float(
                solver_raw.get("refinement_huber_delta", 0.002)
            ),
            refinement_damping=float(
                solver_raw.get("refinement_damping", 1e-6)
            ),
            refinement_step_epsilon=float(
                solver_raw.get("refinement_step_epsilon", 1e-6)
            ),
            max_refinement_step=float(
                solver_raw.get("max_refinement_step", 0.10)
            ),
            cheirality_max_points=int(
                solver_raw.get("cheirality_max_points", 128)
            ),
            ray_intersection_max_condition=float(
                solver_raw.get("ray_intersection_max_condition", 1e8)
            ),
        ),
    )
    _validate_config(config)
    return config


def _validate_config(config: V90Config) -> None:
    if config.max_history != 2:
        raise ValueError("V9 Stage O locks max_history=2; this is not a sweep.")
    if config.visibility.max_queries_per_instance < 8:
        raise ValueError("V9 needs at least eight queries per instance.")
    if config.epipolar.min_correspondences != 8:
        raise ValueError("V9 calibrated essential estimation locks min points to eight.")
    positive = (
        config.visibility.depth_tolerance_metric,
        config.visibility.relative_depth_tolerance,
        config.epipolar.min_design_rank_ratio,
        config.epipolar.refinement_huber_delta,
        config.epipolar.refinement_damping,
        config.epipolar.refinement_step_epsilon,
        config.epipolar.max_refinement_step,
        config.epipolar.ray_intersection_max_condition,
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in positive):
        raise ValueError("V9 positive numeric settings must be finite.")
    if not 0.0 < config.epipolar.min_cheirality_fraction <= 1.0:
        raise ValueError("V9 cheirality fraction must lie in (0,1].")


def _validate_payload(payload: dict[str, Any]) -> None:
    required = {
        "complete",
        "frame_indices",
        "reference_sequence_index",
        "image_size",
        "baseline_pose_encoding",
        "target_pose_encoding",
        "target_world_to_camera",
        "target_world_points",
        "target_depth",
        "tracking_masks_stream",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"V9 cache lacks fields={sorted(missing)}.")
    if not bool(payload["complete"]):
        raise ValueError("V9 refuses an incomplete feature cache.")
    if str(payload.get("sam_version", "")) != "sam3.1":
        raise ValueError("V9 requires the retained SAM3.1 tracking cache.")
    if str(payload.get("instance_source", "")) != "sam31_online":
        raise ValueError("V9 requires dynamic SAM3.1 online instance slots.")


def _validate_decoded_tensors(**values: torch.Tensor) -> None:
    masks = values["masks"]
    if masks.ndim != 4:
        raise ValueError("V9 decoded masks must be [S,K,H,W].")
    sequence, _, height, width = masks.shape
    expected = {
        "world_points": (sequence, height, width, 3),
        "depth": (sequence, height, width),
        "global_w2c": (sequence, 4, 4),
        "intrinsics": (sequence, 3, 3),
        "baseline": (sequence, 4, 4),
        "target": (sequence, 4, 4),
    }
    for name, shape in expected.items():
        if tuple(values[name].shape) != shape:
            raise ValueError(
                f"V9 decoded {name} shape={tuple(values[name].shape)}, expected={shape}."
            )
    if any(not torch.isfinite(values[name]).all() for name in ("global_w2c", "intrinsics", "baseline", "target")):
        raise ValueError("V9 camera tensors must be finite.")


def _trajectory_direction_error(
    predicted: torch.Tensor, target: torch.Tensor, *, current: int
) -> float:
    return float(
        translation_direction_error_degrees(
            predicted, target, reference_index=0
        )[current]
    )


def _edge_rotation_errors(estimates, l0_rows, target_rows) -> tuple[float, float]:
    raw = []
    oracle = []
    for estimate, l0, target in zip(estimates, l0_rows, target_rows):
        if not estimate.success:
            continue
        raw.append(float(rotation_error_degrees(l0, target)))
        candidate = torch.eye(4, dtype=torch.float64)
        candidate[:3, :3] = estimate.rotation_current_to_history
        oracle.append(float(rotation_error_degrees(candidate, target)))
    return _finite_mean(raw), _finite_mean(oracle)


def _edge_direction_errors(estimates, l0_rows, target_rows) -> tuple[float, float]:
    raw = []
    oracle = []
    for estimate, l0, target in zip(estimates, l0_rows, target_rows):
        if not estimate.success:
            continue
        raw.append(
            relative_translation_direction_error_degrees(
                l0[:3, :3], l0[:3, 3], target
            )
        )
        oracle.append(
            relative_translation_direction_error_degrees(
                estimate.rotation_current_to_history,
                estimate.translation_current_origin_in_history,
                target,
            )
        )
    return _finite_mean(raw), _finite_mean(oracle)


def _write_metadata(
    path: Path,
    *,
    config: V90Config,
    cache_path_value: Path,
    payload: dict[str, Any],
    frames: tuple[int, ...],
    rows: Sequence[dict[str, object]],
) -> None:
    config_hash = hashlib.sha256(config.source_path.read_bytes()).hexdigest()
    metadata = {
        "experiment": "V9 Stage-O 2D epipolar oracle",
        "configuration": {
            "source_path": str(config.source_path),
            "sha256": config_hash,
            "max_history": config.max_history,
            "visibility": asdict(config.visibility),
            "epipolar": asdict(config.epipolar),
        },
        "cache": {
            "path": str(cache_path_value),
            "size_bytes": cache_path_value.stat().st_size,
            "cache_version": payload.get("cache_version"),
            "clip_name": payload.get("clip_name"),
            "sam_version": payload.get("sam_version"),
            "sam_checkpoint": payload.get("sam_checkpoint"),
            "instance_source": payload.get("instance_source"),
        },
        "frames": list(frames),
        "folds": [
            {
                "name": fold.name,
                "train_frames": list(fold.train_frames),
                "test_frames": list(fold.test_frames),
            }
            for fold in FOLDS
        ],
        "gt_usage": [
            "surface correspondence construction",
            "z-buffer visibility",
            "calibrated intrinsics",
            "evaluation only",
        ],
        "forbidden_inputs_confirmed_absent": [
            "predicted depth",
            "predicted pointmap",
            "camera hidden",
            "frame index feature",
            "learned pose head",
            "GT-error fallback",
        ],
        "evaluated_frame_rows": len(rows),
    }
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf8")


def _write_csv(
    path: Path,
    rows: Sequence[dict[str, object]],
    columns: Sequence[str],
    *,
    allow_empty: bool = False,
) -> None:
    if not rows and not allow_empty:
        raise ValueError(f"Refusing to write empty V9 CSV: {path.name}")
    expected = set(columns)
    for index, row in enumerate(rows):
        if set(row) != expected:
            raise ValueError(
                f"V9 CSV {path.name} row={index} schema mismatch: "
                f"missing={sorted(expected - set(row))}, extra={sorted(set(row) - expected)}"
            )
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: Iterable[float]) -> float:
    rows = list(values)
    return float("nan") if not rows else sum(rows) / len(rows)


def _finite_mean(values: Iterable[float]) -> float:
    rows = [float(value) for value in values if math.isfinite(float(value))]
    return float("nan") if not rows else sum(rows) / len(rows)


def _sum(values: Iterable[float]) -> float:
    return float(sum(float(value) for value in values))


def _ratio(numerator: float, denominator: float) -> float:
    return 0.0 if denominator <= 0 else float(numerator) / float(denominator)


def _gain(initial: float, final: float) -> float:
    return 0.0 if initial <= 1e-12 else 100.0 * (initial - final) / initial


def _effective_count(weights: torch.Tensor) -> float:
    if not weights.numel():
        return 0.0
    return float(weights.sum().square() / weights.square().sum().clamp_min(1e-12))


def _quantile(values: torch.Tensor, value: float) -> float:
    finite = values[torch.isfinite(values)]
    return float("nan") if not finite.numel() else float(torch.quantile(finite, value))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="streaming_couping/configs/v90_epipolar_oracle.yaml",
    )
    parser.add_argument("--output-dir")
    return parser.parse_args()


if __name__ == "__main__":
    main()
