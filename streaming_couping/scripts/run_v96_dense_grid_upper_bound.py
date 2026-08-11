#!/usr/bin/env python3
"""Run V9.6 actual dense-grid coordinate/region upper bounds."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import torch
import yaml

from streaming_couping.scripts.run_v90_local_token_matcher import load_stage_a_config
from streaming_couping.scripts.run_v92_support_factorization import (
    SupportData,
    _build_support_records,
    _load_support_payload,
    _prepare_support_data,
    load_support_config,
)
from streaming_couping.scripts.run_v93_quantization_tolerance import (
    _select_continuous_fixed_edges,
    load_diagnostic_config,
)
from streaming_couping.scripts.run_v95_spatial_support_scope import (
    _frame_edge_map,
    _locked_edge_indices,
    load_spatial_scope_config,
)
from streaming_couping.src.learned_pose.config import load_learned_pose_config
from streaming_couping.src.v74_temporal_protocol import FOLDS
from streaming_couping.src.v80_pose_geometry import invert_rigid, rotation_error_degrees
from streaming_couping.src.v90_epipolar_geometry import (
    estimate_relative_epipolar_pose,
    local_token_reprojection_labels,
    relative_translation_direction_error_degrees,
)
from streaming_couping.src.v95_spatial_support import (
    concatenate_surface_rows,
    uv_hull_coverage,
)
from streaming_couping.src.v96_dense_grid_decoder import (
    GRID_DECODERS,
    GRID_SUPPORT_SCOPES,
    GridSupport,
    bbox_mask_stream,
    build_equal_count_grid_supports,
    decode_history_grid,
    dense_grid_normalized,
    filter_grid_labels_by_region,
    random_shifted_mask_stream,
)


SUMMARY_COLUMNS = (
    "fold",
    "support_scope",
    "decoder",
    "sam_grid",
    "fixed_history_edges",
    "frames",
    "active_frames",
    "inactive_frames",
    "mean_correspondences",
    "equal_count_exact",
    "mean_region_fraction",
    "mean_complement_fraction",
    "keys_per_query",
    "pck_threshold_pixels",
    "pck_accuracy",
    "mean_epe_pixels",
    "max_epe_pixels",
    "mean_current_hull_coverage",
    "mean_history_hull_coverage",
    "mean_log10_design_condition",
    "raw_rotation_error_deg",
    "refined_rotation_error_deg",
    "rotation_gain_percent",
    "raw_translation_direction_error_deg",
    "refined_translation_direction_error_deg",
    "translation_direction_gain_percent",
    "raw_relative_aggregate_deg",
    "refined_relative_aggregate_deg",
    "relative_aggregate_gain_percent",
    "relative_aggregate_worse_frames",
    "fold_pass",
    "all_folds_pass",
)

FRAME_COLUMNS = (
    "fold",
    "support_scope",
    "decoder",
    "sequence_index",
    "frame_index",
    "history_sequence_index",
    "history_frame_index",
    "correspondences",
    "region_rows",
    "complement_rows",
    "active",
    "solver_reason",
    "solver_initialization",
    "pck_accuracy",
    "mean_epe_pixels",
    "max_epe_pixels",
    "current_hull_coverage",
    "history_hull_coverage",
    "design_condition",
    "sampson_rmse",
    "raw_rotation_error_deg",
    "refined_rotation_error_deg",
    "raw_translation_direction_error_deg",
    "refined_translation_direction_error_deg",
    "relative_aggregate_worse",
)


@dataclass(frozen=True)
class DenseGridConfig:
    source_path: Path
    v95_config: Path
    output_dir: Path
    grid_size: tuple[int, int]
    support_scopes: tuple[str, ...]
    decoders: tuple[str, ...]
    complement_fraction: float
    random_mask_seed: int
    pck_threshold_pixels: float


def main() -> None:
    args = _parse_args()
    config = load_dense_grid_config(args.config)
    if args.output_dir:
        config = replace(
            config, output_dir=Path(args.output_dir).expanduser().resolve()
        )
    result = run_dense_grid_upper_bound(config)
    print(f"V9.6 dense-grid upper bound result={result}")


def run_dense_grid_upper_bound(config: DenseGridConfig) -> Path:
    v95_config = load_spatial_scope_config(config.v95_config)
    v93_config = load_diagnostic_config(v95_config.v93_config)
    support_config = load_support_config(v93_config.support_config)
    stage_config = load_stage_a_config(support_config.stage_a_config)
    stage_config = replace(
        stage_config,
        data_config=support_config.data_config,
        output_dir=config.output_dir,
    )
    learned_config = load_learned_pose_config(support_config.data_config)
    if tuple(learned_config.features.sam_grid) != config.grid_size:
        raise ValueError(
            "V9.6 configured grid does not match the SAM cache protocol: "
            f"{config.grid_size} vs {learned_config.features.sam_grid}."
        )
    if learned_config.features.sam_source != "detector_fpn2":
        raise ValueError("V9.6 requires the detector_fpn2 grid protocol.")

    payload, cache_file = _load_support_payload(support_config)
    data = _prepare_support_data(
        payload, config=support_config, stage_config=stage_config
    )
    del payload
    positions = {frame: index for index, frame in enumerate(data.frames)}
    all_records = {
        fold.name: _build_support_records(
            data,
            current_indices=[positions[value] for value in fold.test_frames],
            query_token_count=support_config.query_token_count,
            stage_config=stage_config,
        )
        for fold in FOLDS
    }
    derived_edges = {
        fold.name: _select_continuous_fixed_edges(
            fold_name=fold.name,
            test_frames=fold.test_frames,
            records=all_records[fold.name],
            data=data,
            stage_config=stage_config,
        )
        for fold in FOLDS
    }
    fixed_edges = _locked_edge_indices(v95_config, data)
    if derived_edges != fixed_edges:
        raise ValueError(
            "V9.6 fixed-edge lock differs from V9.3: "
            f"locked={_frame_edge_map(fixed_edges, data)} "
            f"derived={_frame_edge_map(derived_edges, data)}"
        )

    grid_uv = dense_grid_normalized(config.grid_size)
    grid_valid = torch.ones(len(grid_uv), dtype=torch.bool)
    union = data.masks.any(dim=1, keepdim=True)
    bbox = bbox_mask_stream(data.masks)
    random_control = random_shifted_mask_stream(
        data.masks, seed=config.random_mask_seed
    )
    full_masks = torch.ones_like(union)
    region_streams = {
        "sam_mask_balanced": union,
        "bbox_balanced": bbox,
        "random_shifted_mask_balanced": random_control,
    }
    supports: dict[str, dict[int, dict[str, GridSupport]]] = {}
    target_counts: dict[str, dict[int, int]] = {}
    for fold in FOLDS:
        fold_supports = {}
        fold_counts = {}
        for current, history in fixed_edges[fold.name].items():
            local_rows = [
                record.labels.visible_correspondences()
                for record in all_records[fold.name]
                if record.current == current and record.history == history
            ]
            instance_local32 = concatenate_surface_rows(
                local_rows, current_frame=current, history_frame=history
            )
            target_count = instance_local32.count
            fold_counts[current] = target_count
            labels = local_token_reprojection_labels(
                current_frame=current,
                history_frame=history,
                slot=0,
                local_uv_normalized=grid_uv,
                local_valid=grid_valid,
                masks=full_masks,
                world_points_metric=data.world_points,
                depth_metric=data.depth,
                global_world_to_camera=data.global_w2c,
                intrinsics=data.intrinsics,
                config=stage_config.visibility,
            )
            full = labels.visible_correspondences()
            regions = {}
            for scope, masks in region_streams.items():
                regions[scope] = (
                    filter_grid_labels_by_region(labels, masks),
                    filter_grid_labels_by_region(labels, ~masks),
                )
            fold_supports[current] = build_equal_count_grid_supports(
                full=full,
                regions=regions,
                target_count=target_count,
                image_size=data.image_size,
                complement_fraction=config.complement_fraction,
            )
        supports[fold.name] = fold_supports
        target_counts[fold.name] = fold_counts

    summaries: list[dict[str, object]] = []
    frames: list[dict[str, object]] = []
    for scope in config.support_scopes:
        for decoder in config.decoders:
            for fold in FOLDS:
                summary, rows = _evaluate_configuration(
                    fold_name=fold.name,
                    test_frames=fold.test_frames,
                    scope=scope,
                    decoder=decoder,
                    fixed_edges=fixed_edges[fold.name],
                    supports=supports[fold.name],
                    target_counts=target_counts[fold.name],
                    data=data,
                    stage_config=stage_config,
                    config=config,
                )
                summaries.append(summary)
                frames.extend(rows)
    _annotate_passes(summaries)
    _validate_outputs(summaries, frames, config=config)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = config.output_dir / "v96_dense_grid_summary.csv"
    frames_path = config.output_dir / "v96_dense_grid_frames.csv"
    decision_path = config.output_dir / "v96_dense_grid_decision.md"
    metadata_path = config.output_dir / "v96_dense_grid_metadata.json"
    _write_csv(summary_path, summaries, SUMMARY_COLUMNS)
    _write_csv(frames_path, frames, FRAME_COLUMNS)
    decision_path.write_text(_decision_markdown(summaries), encoding="utf8")
    metadata_path.write_text(
        json.dumps(
            {
                "experiment": "V9.6 actual dense-grid coordinate upper bound",
                "config": _jsonable_config(config),
                "cache": {
                    "path": str(cache_file),
                    "size_bytes": cache_file.stat().st_size,
                    "mtime_ns": cache_file.stat().st_mtime_ns,
                    "contains_dense_sam_descriptors": False,
                    "cached_local_source": learned_config.features.sam_source,
                    "configured_sam_grid": list(learned_config.features.sam_grid),
                },
                "fixed_edges": _frame_edge_map(fixed_edges, data),
                "trains_matcher": False,
                "trains_pose_model": False,
                "uses_pose_loss": False,
                "reads_gt_pose_error_for_selection": False,
                "support_selection": "current UV and current masks only",
                "gt_usage": "visibility, continuous history targets, decoding upper bound and scoring",
                "descriptor_claim_allowed": False,
                "soft_k4_uses_actual_grid_keys": True,
                "outputs": {
                    "summary": str(summary_path),
                    "frames": str(frames_path),
                    "decision": str(decision_path),
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf8",
    )
    print(f"V9.6 wrote summary={summary_path}")
    print(f"V9.6 wrote decision={decision_path}")
    return summary_path


def _evaluate_configuration(
    *,
    fold_name: str,
    test_frames: Sequence[int],
    scope: str,
    decoder: str,
    fixed_edges: dict[int, int],
    supports: dict[int, dict[str, GridSupport]],
    target_counts: dict[int, int],
    data: SupportData,
    stage_config,
    config: DenseGridConfig,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    output = []
    all_errors = []
    keys_per_query = None
    for frame_value in test_frames:
        current = data.frames.index(int(frame_value))
        history = fixed_edges[current]
        support = supports[current][scope]
        decoded = decode_history_grid(
            support.correspondences,
            mode=decoder,
            grid_size=config.grid_size,
            image_size=data.image_size,
        )
        keys_per_query = decoded.keys_per_query
        if decoded.errors_pixels.numel():
            all_errors.append(decoded.errors_pixels)
        row = decoded.correspondences
        l0_relative = data.baseline[history] @ invert_rigid(data.baseline[current])
        target_relative = data.target[history] @ invert_rigid(data.target[current])
        raw_r = float(rotation_error_degrees(l0_relative, target_relative))
        raw_t = relative_translation_direction_error_degrees(
            l0_relative[:3, :3], l0_relative[:3, 3], target_relative
        )
        estimate = estimate_relative_epipolar_pose(
            row.current_uv,
            row.history_uv,
            row.weights,
            data.intrinsics[current],
            data.intrinsics[history],
            l0_relative,
            config=stage_config.epipolar,
        )
        refined_r, refined_t = raw_r, raw_t
        if estimate.success:
            candidate = torch.eye(4, dtype=torch.float64)
            candidate[:3, :3] = estimate.rotation_current_to_history
            refined_r = float(rotation_error_degrees(candidate, target_relative))
            refined_t = relative_translation_direction_error_degrees(
                estimate.rotation_current_to_history,
                estimate.translation_current_origin_in_history,
                target_relative,
            )
        errors = decoded.errors_pixels
        raw_pose = raw_r + raw_t
        refined_pose = refined_r + refined_t
        output.append(
            {
                "fold": fold_name,
                "support_scope": scope,
                "decoder": decoder,
                "sequence_index": current,
                "frame_index": int(frame_value),
                "history_sequence_index": history,
                "history_frame_index": data.frames[history],
                "correspondences": row.count,
                "region_rows": support.region_rows,
                "complement_rows": support.complement_rows,
                "active": int(estimate.success),
                "solver_reason": estimate.reason,
                "solver_initialization": estimate.initialization,
                "pck_accuracy": _ratio(
                    float(errors.le(config.pck_threshold_pixels).sum()),
                    float(errors.numel()),
                ),
                "mean_epe_pixels": _finite_mean(errors.tolist()),
                "max_epe_pixels": _finite_max(errors.tolist()),
                "current_hull_coverage": uv_hull_coverage(
                    row.current_uv, data.image_size
                ),
                "history_hull_coverage": uv_hull_coverage(
                    row.history_uv, data.image_size
                ),
                "design_condition": estimate.design_condition,
                "sampson_rmse": estimate.sampson_rmse,
                "raw_rotation_error_deg": raw_r,
                "refined_rotation_error_deg": refined_r,
                "raw_translation_direction_error_deg": raw_t,
                "refined_translation_direction_error_deg": refined_t,
                "relative_aggregate_worse": int(
                    refined_pose > raw_pose + 1e-12
                ),
            }
        )
    errors = torch.cat(all_errors) if all_errors else torch.empty(0)
    raw_r = _finite_mean(float(row["raw_rotation_error_deg"]) for row in output)
    refined_r = _finite_mean(
        float(row["refined_rotation_error_deg"]) for row in output
    )
    raw_t = _finite_mean(
        float(row["raw_translation_direction_error_deg"]) for row in output
    )
    refined_t = _finite_mean(
        float(row["refined_translation_direction_error_deg"]) for row in output
    )
    raw_pose = raw_r + raw_t
    refined_pose = refined_r + refined_t
    summary = {
        "fold": fold_name,
        "support_scope": scope,
        "decoder": decoder,
        "sam_grid": f"{config.grid_size[0]}x{config.grid_size[1]}",
        "fixed_history_edges": " ".join(
            f"{data.frames[current]}:{data.frames[history]}"
            for current, history in sorted(fixed_edges.items())
        ),
        "frames": len(output),
        "active_frames": sum(int(row["active"]) for row in output),
        "inactive_frames": sum(not int(row["active"]) for row in output),
        "mean_correspondences": _finite_mean(
            float(row["correspondences"]) for row in output
        ),
        "equal_count_exact": int(
            all(
                int(row["correspondences"])
                == target_counts[int(row["sequence_index"])]
                for row in output
            )
        ),
        "mean_region_fraction": _finite_mean(
            _ratio(float(row["region_rows"]), float(row["correspondences"]))
            for row in output
        ),
        "mean_complement_fraction": _finite_mean(
            _ratio(float(row["complement_rows"]), float(row["correspondences"]))
            for row in output
        ),
        "keys_per_query": 0 if keys_per_query is None else keys_per_query,
        "pck_threshold_pixels": config.pck_threshold_pixels,
        "pck_accuracy": _ratio(
            float(errors.le(config.pck_threshold_pixels).sum()),
            float(errors.numel()),
        ),
        "mean_epe_pixels": _finite_mean(errors.tolist()),
        "max_epe_pixels": _finite_max(errors.tolist()),
        "mean_current_hull_coverage": _finite_mean(
            float(row["current_hull_coverage"]) for row in output
        ),
        "mean_history_hull_coverage": _finite_mean(
            float(row["history_hull_coverage"]) for row in output
        ),
        "mean_log10_design_condition": _finite_mean(
            math.log10(float(row["design_condition"]))
            for row in output
            if math.isfinite(float(row["design_condition"]))
            and float(row["design_condition"]) > 0.0
        ),
        "raw_rotation_error_deg": raw_r,
        "refined_rotation_error_deg": refined_r,
        "rotation_gain_percent": _gain(raw_r, refined_r),
        "raw_translation_direction_error_deg": raw_t,
        "refined_translation_direction_error_deg": refined_t,
        "translation_direction_gain_percent": _gain(raw_t, refined_t),
        "raw_relative_aggregate_deg": raw_pose,
        "refined_relative_aggregate_deg": refined_pose,
        "relative_aggregate_gain_percent": _gain(raw_pose, refined_pose),
        "relative_aggregate_worse_frames": sum(
            int(row["relative_aggregate_worse"]) for row in output
        ),
        "fold_pass": 0,
        "all_folds_pass": 0,
    }
    return summary, output


def _annotate_passes(rows: list[dict[str, object]]) -> None:
    for row in rows:
        row["fold_pass"] = int(_strict_pass(row))
    for scope in GRID_SUPPORT_SCOPES:
        for decoder in GRID_DECODERS:
            values = [
                row
                for row in rows
                if row["support_scope"] == scope and row["decoder"] == decoder
            ]
            if len(values) != len(FOLDS):
                raise ValueError(f"V9.6 scope={scope} decoder={decoder} lacks folds.")
            passed = int(all(int(row["fold_pass"]) for row in values))
            for row in values:
                row["all_folds_pass"] = passed


def _strict_pass(row: dict[str, object]) -> bool:
    return bool(
        int(row["equal_count_exact"])
        and int(row["active_frames"]) == int(row["frames"])
        and float(row["refined_rotation_error_deg"])
        < float(row["raw_rotation_error_deg"])
        and float(row["refined_translation_direction_error_deg"])
        < float(row["raw_translation_direction_error_deg"])
        and int(row["relative_aggregate_worse_frames"]) == 0
    )


def _decision_markdown(rows: Sequence[dict[str, object]]) -> str:
    groups = []
    for scope in GRID_SUPPORT_SCOPES:
        for decoder in GRID_DECODERS:
            values = [
                row
                for row in rows
                if row["support_scope"] == scope and row["decoder"] == decoder
            ]
            groups.append(
                {
                    "scope": scope,
                    "decoder": decoder,
                    "count": _finite_mean(
                        float(row["mean_correspondences"]) for row in values
                    ),
                    "equal": int(
                        all(int(row["equal_count_exact"]) for row in values)
                    ),
                    "pck": _finite_mean(float(row["pck_accuracy"]) for row in values),
                    "epe": _finite_mean(float(row["mean_epe_pixels"]) for row in values),
                    "max_epe": _finite_max(float(row["max_epe_pixels"]) for row in values),
                    "hull": _finite_mean(
                        float(row["mean_current_hull_coverage"]) for row in values
                    ),
                    "r_gain": _finite_mean(
                        float(row["rotation_gain_percent"]) for row in values
                    ),
                    "t_gain": _finite_mean(
                        float(row["translation_direction_gain_percent"])
                        for row in values
                    ),
                    "worse": sum(
                        int(row["relative_aggregate_worse_frames"]) for row in values
                    ),
                    "pass": int(all(int(row["all_folds_pass"]) for row in values)),
                }
            )
    positive = _group_pass(groups, "full_grid", "continuous_gt")
    hard_full = _group_pass(groups, "full_grid", "hard_nearest")
    soft_full = _group_pass(groups, "full_grid", "soft_bilinear_k4")
    soft_sam = _group_pass(groups, "sam_mask_balanced", "soft_bilinear_k4")
    soft_bbox = _group_pass(groups, "bbox_balanced", "soft_bilinear_k4")
    soft_random = _group_pass(
        groups, "random_shifted_mask_balanced", "soft_bilinear_k4"
    )
    sam_feasible = _group_equal(groups, "sam_mask_balanced", "soft_bilinear_k4")
    bbox_feasible = _group_equal(groups, "bbox_balanced", "soft_bilinear_k4")
    random_feasible = _group_equal(
        groups, "random_shifted_mask_balanced", "soft_bilinear_k4"
    )
    gate = int(positive and soft_full and soft_sam)
    sam_unique = int(
        sam_feasible
        and soft_sam
        and not soft_full
        and bbox_feasible
        and not soft_bbox
        and random_feasible
        and not soft_random
    )
    lines = [
        "# V9.6 actual dense-grid coordinate decision",
        "",
        "No dense descriptor is cached or evaluated, and no model is trained.",
        "Current queries and decoded history keys are restricted to the configured 72x72 SAM detector grid.",
        "Support selection uses current UV/masks only; GT supplies visibility and continuous history targets.",
        "",
        f"- full-grid continuous positive-control pass: `{positive}`",
        f"- full-grid hard-nearest all-fold pass: `{hard_full}`",
        f"- full-grid soft-bilinear-K4 all-fold pass: `{soft_full}`",
        f"- SAM-mask balanced soft-bilinear-K4 pass: `{soft_sam}`",
        f"- bbox balanced soft-bilinear-K4 pass: `{soft_bbox}`",
        f"- random-shifted-mask balanced soft-bilinear-K4 pass: `{soft_random}`",
        f"- SAM-mask balanced equal-count feasible: `{sam_feasible}`",
        f"- bbox balanced equal-count feasible: `{bbox_feasible}`",
        f"- random-shifted-mask balanced equal-count feasible: `{random_feasible}`",
        f"- SAM mask uniquely required by this oracle: `{sam_unique}`",
        f"- dense-descriptor stage allowed: `{gate}`",
        "",
        "| support | decoder | rows/edge | equal count | PCK@1 | EPE px | max EPE | current hull | mean R gain | mean t-dir gain | worse | pass |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in groups:
        lines.append(
            f"| {row['scope']} | {row['decoder']} | {row['count']:.6g} "
            f"| {row['equal']} | {row['pck']:.6g} "
            f"| {row['epe']:.6g} | {row['max_epe']:.6g} | {row['hull']:.6g} "
            f"| {row['r_gain']:.6g} | {row['t_gain']:.6g} "
            f"| {row['worse']} | {row['pass']} |"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- positive-control=0: actual-grid query placement broke the V9.5 upper bound; stop.",
            "- hard=1: Top-1 grid coordinates are already solver-feasible.",
            "- hard=0 and soft-K4=1: a learned sub-grid expectation decoder is required.",
            "- dense-descriptor-stage=1: coordinate/support expressivity is sufficient to justify caching and testing dense features.",
            "- A bbox/random row with equal-count=0 is an unavailable control, not evidence that SAM won.",
            "- bbox/random/full controls passing means SAM mask is not uniquely useful at the oracle-coordinate stage.",
            "- No result here is evidence that a SAM descriptor predicts correspondence or improves pose.",
            "",
        ]
    )
    return "\n".join(lines)


def _group_pass(groups, scope: str, decoder: str) -> int:
    values = [
        row
        for row in groups
        if row["scope"] == scope and row["decoder"] == decoder
    ]
    if len(values) != 1:
        raise ValueError(f"V9.6 lacks group scope={scope} decoder={decoder}.")
    return int(values[0]["pass"])


def _group_equal(groups, scope: str, decoder: str) -> int:
    values = [
        row
        for row in groups
        if row["scope"] == scope and row["decoder"] == decoder
    ]
    if len(values) != 1:
        raise ValueError(f"V9.6 lacks group scope={scope} decoder={decoder}.")
    return int(values[0]["equal"])


def load_dense_grid_config(path: str | Path) -> DenseGridConfig:
    source = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf8")) or {}
    repository = source.parents[2]
    config = DenseGridConfig(
        source_path=source,
        v95_config=_resolve_path(
            repository,
            raw.get(
                "v95_config",
                "streaming_couping/configs/v95_spatial_support_scope.yaml",
            ),
        ),
        output_dir=_resolve_path(
            repository,
            raw.get(
                "output_dir",
                "outputs/streaming_couping_v96_dense_grid_upper_bound",
            ),
        ),
        grid_size=tuple(int(value) for value in raw.get("grid_size", [72, 72])),
        support_scopes=tuple(str(value) for value in raw.get("support_scopes", [])),
        decoders=tuple(str(value) for value in raw.get("decoders", [])),
        complement_fraction=float(raw.get("complement_fraction", 0.5)),
        random_mask_seed=int(raw.get("random_mask_seed", 96)),
        pck_threshold_pixels=float(raw.get("pck_threshold_pixels", 1.0)),
    )
    _validate_config(config)
    return config


def _validate_config(config: DenseGridConfig) -> None:
    if config.grid_size != (72, 72):
        raise ValueError("V9.6 SAM grid is protocol-locked to 72x72.")
    if config.support_scopes != GRID_SUPPORT_SCOPES:
        raise ValueError("V9.6 support scopes are protocol-locked.")
    if config.decoders != GRID_DECODERS:
        raise ValueError("V9.6 grid decoders are protocol-locked.")
    if config.complement_fraction != 0.5:
        raise ValueError("V9.6 balanced scopes must use 50% complement.")
    if config.random_mask_seed != 96:
        raise ValueError("V9.6 random-mask seed changed.")
    if config.pck_threshold_pixels != 1.0:
        raise ValueError("V9.6 PCK threshold is protocol-locked to one pixel.")


def _validate_outputs(
    summaries: Sequence[dict[str, object]],
    frames: Sequence[dict[str, object]],
    *,
    config: DenseGridConfig,
) -> None:
    expected_summary = len(config.support_scopes) * len(config.decoders) * len(FOLDS)
    expected_frames = len(config.support_scopes) * len(config.decoders) * sum(
        len(fold.test_frames) for fold in FOLDS
    )
    if len(summaries) != expected_summary:
        raise ValueError(
            f"V9.6 summary rows={len(summaries)}, expected={expected_summary}."
        )
    if len(frames) != expected_frames:
        raise ValueError(f"V9.6 frame rows={len(frames)}, expected={expected_frames}.")


def _write_csv(path: Path, rows, columns) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty V9.6 CSV: {path.name}")
    expected = set(columns)
    for index, row in enumerate(rows):
        if set(row) != expected:
            raise ValueError(
                f"V9.6 CSV {path.name} row={index} mismatch: "
                f"missing={sorted(expected - set(row))}, "
                f"extra={sorted(set(row) - expected)}"
            )
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _ratio(numerator: float, denominator: float) -> float:
    return 0.0 if float(denominator) <= 0.0 else float(numerator) / float(denominator)


def _finite_mean(values: Iterable[float]) -> float:
    rows = [float(value) for value in values if math.isfinite(float(value))]
    return float("nan") if not rows else sum(rows) / len(rows)


def _finite_max(values: Iterable[float]) -> float:
    rows = [float(value) for value in values if math.isfinite(float(value))]
    return float("nan") if not rows else max(rows)


def _gain(initial: float, final: float) -> float:
    if not math.isfinite(initial) or not math.isfinite(final) or initial <= 1e-12:
        return 0.0
    return 100.0 * (initial - final) / initial


def _resolve_path(repository: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repository / path).resolve()


def _jsonable_config(config: DenseGridConfig) -> dict[str, object]:
    value = asdict(config)
    value["source_path"] = str(config.source_path)
    value["v95_config"] = str(config.v95_config)
    value["output_dir"] = str(config.output_dir)
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="streaming_couping/configs/v96_dense_grid_upper_bound.yaml",
    )
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
