#!/usr/bin/env python3
"""Run the V9.5 equal-count spatial-support feasibility diagnosis."""

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
    SupportRecord,
    _build_support_records,
    _load_support_payload,
    _prepare_support_data,
    load_support_config,
)
from streaming_couping.scripts.run_v93_quantization_tolerance import (
    _select_continuous_fixed_edges,
    load_diagnostic_config,
)
from streaming_couping.src.v74_temporal_protocol import FOLDS
from streaming_couping.src.v80_pose_geometry import invert_rigid, rotation_error_degrees
from streaming_couping.src.v90_epipolar_geometry import (
    SurfaceCorrespondences,
    estimate_relative_epipolar_pose,
    relative_translation_direction_error_degrees,
    surface_reprojection_correspondences,
)
from streaming_couping.src.v95_spatial_support import (
    SUPPORT_SCOPES,
    SpatialSupport,
    build_equal_count_supports,
    concatenate_surface_rows,
    perturb_history_uv,
    region_mask_streams,
    uv_hull_coverage,
)


SUMMARY_COLUMNS = (
    "fold",
    "support_scope",
    "noise_sigma_pixels",
    "replicates",
    "fixed_history_edges",
    "evaluation_rows",
    "active_rows",
    "inactive_rows",
    "mean_correspondences",
    "equal_count_exact",
    "mean_instance_fraction",
    "mean_background_fraction",
    "mean_current_hull_coverage",
    "mean_history_hull_coverage",
    "selected_mean_epe_pixels",
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
    "relative_aggregate_worse_rows",
    "fold_pass",
    "all_folds_pass",
)

FRAME_COLUMNS = (
    "fold",
    "support_scope",
    "noise_sigma_pixels",
    "replicate",
    "sequence_index",
    "frame_index",
    "history_sequence_index",
    "history_frame_index",
    "correspondences",
    "instance_rows",
    "background_rows",
    "active",
    "solver_reason",
    "solver_initialization",
    "current_hull_coverage",
    "history_hull_coverage",
    "selected_mean_epe_pixels",
    "design_condition",
    "sampson_rmse",
    "raw_rotation_error_deg",
    "refined_rotation_error_deg",
    "raw_translation_direction_error_deg",
    "refined_translation_direction_error_deg",
    "relative_aggregate_worse",
)


@dataclass(frozen=True)
class SpatialScopeConfig:
    source_path: Path
    v93_config: Path
    output_dir: Path
    support_scopes: tuple[str, ...]
    noise_sigmas_pixels: tuple[float, ...]
    noise_replicates: int
    noise_seed: int
    max_candidate_queries: int
    hybrid_background_fraction: float
    fixed_history_edges: dict[str, dict[int, int]]


def main() -> None:
    args = _parse_args()
    config = load_spatial_scope_config(args.config)
    if args.output_dir:
        config = replace(
            config, output_dir=Path(args.output_dir).expanduser().resolve()
        )
    result = run_spatial_support_scope(config)
    print(f"V9.5 spatial support result={result}")


def run_spatial_support_scope(config: SpatialScopeConfig) -> Path:
    v93_config = load_diagnostic_config(config.v93_config)
    support_config = load_support_config(v93_config.support_config)
    stage_config = load_stage_a_config(support_config.stage_a_config)
    stage_config = replace(
        stage_config,
        data_config=support_config.data_config,
        output_dir=config.output_dir,
    )
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
    fixed_edges = _locked_edge_indices(config, data)
    if derived_edges != fixed_edges:
        raise ValueError(
            "V9.5 fixed-edge lock differs from the deterministic V9.3 selector: "
            f"locked={_frame_edge_map(fixed_edges, data)} "
            f"derived={_frame_edge_map(derived_edges, data)}"
        )

    full_masks, background_masks = region_mask_streams(data.masks)
    visibility = replace(
        stage_config.visibility,
        max_queries_per_instance=config.max_candidate_queries,
    )
    supports: dict[str, dict[int, SpatialSupport]] = {}
    for fold in FOLDS:
        fold_supports = {}
        for current, history in fixed_edges[fold.name].items():
            edge_records = [
                record
                for record in all_records[fold.name]
                if record.current == current and record.history == history
            ]
            instance = concatenate_surface_rows(
                [record.labels.visible_correspondences() for record in edge_records],
                current_frame=current,
                history_frame=history,
            )
            full = surface_reprojection_correspondences(
                current_frame=current,
                history_frame=history,
                slot=0,
                masks=full_masks,
                world_points_metric=data.world_points,
                depth_metric=data.depth,
                global_world_to_camera=data.global_w2c,
                intrinsics=data.intrinsics,
                config=visibility,
            )
            background = surface_reprojection_correspondences(
                current_frame=current,
                history_frame=history,
                slot=0,
                masks=background_masks,
                world_points_metric=data.world_points,
                depth_metric=data.depth,
                global_world_to_camera=data.global_w2c,
                intrinsics=data.intrinsics,
                config=visibility,
            )
            fold_supports[current] = build_equal_count_supports(
                instance=instance,
                full_image_candidates=full,
                background_candidates=background,
                instance_union_mask=data.masks[current].any(dim=0),
                image_size=data.image_size,
                background_fraction=config.hybrid_background_fraction,
            )
        supports[fold.name] = fold_supports

    summaries: list[dict[str, object]] = []
    frames: list[dict[str, object]] = []
    for scope in config.support_scopes:
        for sigma in config.noise_sigmas_pixels:
            for fold in FOLDS:
                summary, rows = _evaluate_configuration(
                    fold_name=fold.name,
                    test_frames=fold.test_frames,
                    support_scope=scope,
                    sigma_pixels=sigma,
                    fixed_edges=fixed_edges[fold.name],
                    supports=supports[fold.name],
                    data=data,
                    stage_config=stage_config,
                    config=config,
                )
                summaries.append(summary)
                frames.extend(rows)
    _annotate_passes(summaries)
    _validate_outputs(summaries, frames, config=config)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = config.output_dir / "v95_spatial_support_summary.csv"
    frames_path = config.output_dir / "v95_spatial_support_frames.csv"
    decision_path = config.output_dir / "v95_spatial_support_decision.md"
    metadata_path = config.output_dir / "v95_spatial_support_metadata.json"
    _write_csv(summary_path, summaries, SUMMARY_COLUMNS)
    _write_csv(frames_path, frames, FRAME_COLUMNS)
    decision_path.write_text(_decision_markdown(summaries), encoding="utf8")
    metadata_path.write_text(
        json.dumps(
            {
                "experiment": "V9.5 equal-count spatial-support feasibility",
                "config": _jsonable_config(config),
                "v93_config": str(v93_config.source_path),
                "cache": {
                    "path": str(cache_file),
                    "size_bytes": cache_file.stat().st_size,
                    "mtime_ns": cache_file.stat().st_mtime_ns,
                },
                "fixed_edges": _frame_edge_map(fixed_edges, data),
                "edge_lock_matches_v93_selector": True,
                "trains_matcher": False,
                "trains_pose_model": False,
                "uses_pose_loss": False,
                "solver": "frozen O-R1",
                "gt_usage": (
                    "visibility, continuous correspondence, joint-view spatial "
                    "upper-bound selection, noise tolerance and scoring"
                ),
                "equal_count_against": "instance_local32 visible correspondence count",
                "noise_added_after_support_selection": True,
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
    print(f"V9.5 wrote summary={summary_path}")
    print(f"V9.5 wrote decision={decision_path}")
    return summary_path


def _evaluate_configuration(
    *,
    fold_name: str,
    test_frames: Sequence[int],
    support_scope: str,
    sigma_pixels: float,
    fixed_edges: dict[int, int],
    supports: dict[int, dict[str, SpatialSupport]],
    data: SupportData,
    stage_config,
    config: SpatialScopeConfig,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    replicates = 1 if float(sigma_pixels) == 0.0 else config.noise_replicates
    output = []
    epe_rows = []
    scope_index = config.support_scopes.index(support_scope)
    for replicate in range(replicates):
        for frame_value in test_frames:
            current = data.frames.index(int(frame_value))
            history = fixed_edges[current]
            support = supports[current][support_scope]
            prediction, errors = perturb_history_uv(
                support.correspondences,
                sigma_pixels=sigma_pixels,
                seed=(
                    config.noise_seed
                    + scope_index * 100_000_007
                    + replicate * 1_000_003
                    + current * 10_007
                    + history * 101
                ),
            )
            if errors.numel():
                epe_rows.append(errors)
            l0_relative = data.baseline[history] @ invert_rigid(data.baseline[current])
            target_relative = data.target[history] @ invert_rigid(data.target[current])
            raw_r = float(rotation_error_degrees(l0_relative, target_relative))
            raw_t = relative_translation_direction_error_degrees(
                l0_relative[:3, :3], l0_relative[:3, 3], target_relative
            )
            estimate = estimate_relative_epipolar_pose(
                prediction.current_uv,
                prediction.history_uv,
                prediction.weights,
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
            raw_pose = raw_r + raw_t
            refined_pose = refined_r + refined_t
            output.append(
                {
                    "fold": fold_name,
                    "support_scope": support_scope,
                    "noise_sigma_pixels": sigma_pixels,
                    "replicate": replicate,
                    "sequence_index": current,
                    "frame_index": int(frame_value),
                    "history_sequence_index": history,
                    "history_frame_index": data.frames[history],
                    "correspondences": prediction.count,
                    "instance_rows": support.instance_rows,
                    "background_rows": support.background_rows,
                    "active": int(estimate.success),
                    "solver_reason": estimate.reason,
                    "solver_initialization": estimate.initialization,
                    "current_hull_coverage": uv_hull_coverage(
                        prediction.current_uv, data.image_size
                    ),
                    "history_hull_coverage": uv_hull_coverage(
                        prediction.history_uv, data.image_size
                    ),
                    "selected_mean_epe_pixels": _finite_mean(errors.tolist()),
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
    instance_counts = {
        current: supports[current]["instance_local32"].count
        for current in fixed_edges
    }
    summary = {
        "fold": fold_name,
        "support_scope": support_scope,
        "noise_sigma_pixels": sigma_pixels,
        "replicates": replicates,
        "fixed_history_edges": " ".join(
            f"{data.frames[current]}:{data.frames[history]}"
            for current, history in sorted(fixed_edges.items())
        ),
        "evaluation_rows": len(output),
        "active_rows": sum(int(row["active"]) for row in output),
        "inactive_rows": sum(not int(row["active"]) for row in output),
        "mean_correspondences": _finite_mean(
            float(row["correspondences"]) for row in output
        ),
        "equal_count_exact": int(
            all(
                int(row["correspondences"])
                == instance_counts[int(row["sequence_index"])]
                for row in output
            )
        ),
        "mean_instance_fraction": _finite_mean(
            _ratio(float(row["instance_rows"]), float(row["correspondences"]))
            for row in output
        ),
        "mean_background_fraction": _finite_mean(
            _ratio(float(row["background_rows"]), float(row["correspondences"]))
            for row in output
        ),
        "mean_current_hull_coverage": _finite_mean(
            float(row["current_hull_coverage"]) for row in output
        ),
        "mean_history_hull_coverage": _finite_mean(
            float(row["history_hull_coverage"]) for row in output
        ),
        "selected_mean_epe_pixels": _finite_mean(
            torch.cat(epe_rows).tolist() if epe_rows else []
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
        "relative_aggregate_worse_rows": sum(
            int(row["relative_aggregate_worse"]) for row in output
        ),
        "fold_pass": 0,
        "all_folds_pass": 0,
    }
    return summary, output


def _annotate_passes(rows: list[dict[str, object]]) -> None:
    for row in rows:
        row["fold_pass"] = int(_strict_pass(row))
    keys = sorted(
        {
            (str(row["support_scope"]), float(row["noise_sigma_pixels"]))
            for row in rows
        }
    )
    for scope, sigma in keys:
        values = [
            row
            for row in rows
            if row["support_scope"] == scope
            and float(row["noise_sigma_pixels"]) == sigma
        ]
        if len(values) != len(FOLDS):
            raise ValueError(f"V9.5 scope={scope} sigma={sigma} lacks folds.")
        passed = int(all(int(row["fold_pass"]) for row in values))
        for row in values:
            row["all_folds_pass"] = passed


def _strict_pass(row: dict[str, object]) -> bool:
    return bool(
        int(row["equal_count_exact"])
        and int(row["active_rows"]) == int(row["evaluation_rows"])
        and float(row["refined_rotation_error_deg"])
        < float(row["raw_rotation_error_deg"])
        and float(row["refined_translation_direction_error_deg"])
        < float(row["raw_translation_direction_error_deg"])
        and int(row["relative_aggregate_worse_rows"]) == 0
    )


def _decision_markdown(rows: Sequence[dict[str, object]]) -> str:
    groups = []
    for scope in SUPPORT_SCOPES:
        for sigma in (0.0, 0.5, 1.0):
            values = [
                row
                for row in rows
                if row["support_scope"] == scope
                and float(row["noise_sigma_pixels"]) == sigma
            ]
            groups.append(
                {
                    "scope": scope,
                    "sigma": sigma,
                    "count": _finite_mean(
                        float(row["mean_correspondences"]) for row in values
                    ),
                    "current_hull": _finite_mean(
                        float(row["mean_current_hull_coverage"]) for row in values
                    ),
                    "history_hull": _finite_mean(
                        float(row["mean_history_hull_coverage"]) for row in values
                    ),
                    "condition": _finite_mean(
                        float(row["mean_log10_design_condition"]) for row in values
                    ),
                    "r_gain": _finite_mean(
                        float(row["rotation_gain_percent"]) for row in values
                    ),
                    "t_gain": _finite_mean(
                        float(row["translation_direction_gain_percent"])
                        for row in values
                    ),
                    "worse": sum(
                        int(row["relative_aggregate_worse_rows"]) for row in values
                    ),
                    "pass": int(all(int(row["all_folds_pass"]) for row in values)),
                }
            )
    positive = _group_pass(groups, "instance_local32", 0.0)
    full05 = _group_pass(groups, "full_image_equal_count", 0.5)
    hybrid05 = _group_pass(groups, "instance_background_balanced", 0.5)
    full10 = _group_pass(groups, "full_image_equal_count", 1.0)
    hybrid10 = _group_pass(groups, "instance_background_balanced", 1.0)
    feasible = int(positive and (full05 or hybrid05))
    lines = [
        "# V9.5 spatial-support scope decision",
        "",
        "No matcher or pose model is trained. All branches use the frozen V9.3 history edges and O-R1 solver.",
        "Every branch is matched to the same per-edge correspondence count; GT joint-view sampling is an oracle upper bound.",
        "",
        f"- instance-local32 continuous positive-control pass: `{positive}`",
        f"- full-image equal-count 0.5px-noise pass: `{full05}`",
        f"- instance/background balanced 0.5px-noise pass: `{hybrid05}`",
        f"- full-image equal-count 1.0px-noise pass: `{full10}`",
        f"- instance/background balanced 1.0px-noise pass: `{hybrid10}`",
        f"- expanded spatial support noise-feasible: `{feasible}`",
        "",
        "| support | noise px | rows/edge | current hull | history hull | log10 cond | mean R gain | mean t-dir gain | worse | pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in groups:
        lines.append(
            f"| {row['scope']} | {row['sigma']:.1f} | {row['count']:.6g} "
            f"| {row['current_hull']:.6g} | {row['history_hull']:.6g} "
            f"| {row['condition']:.6g} | {row['r_gain']:.6g} "
            f"| {row['t_gain']:.6g} | {row['worse']} | {row['pass']} |"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- positive-control=0: protocol regression; no spatial-support conclusion is valid.",
            "- expanded 0.5px=1: localized instance support, rather than pixel error alone, caused the V9.4 failure.",
            "- full-image=1 but hybrid=0: broad background coverage is required; instance evidence remains too localized.",
            "- hybrid=1: a globally conditioned matcher with SAM-based stratification has a viable upper bound.",
            "- both expanded 0.5px=0: terminate the fixed essential route entirely; do not train another matcher for it.",
            "- Any pass here is a GT spatial-support upper bound, never evidence that SAM descriptors improve pose.",
            "",
        ]
    )
    return "\n".join(lines)


def _group_pass(groups, scope: str, sigma: float) -> int:
    values = [
        row
        for row in groups
        if row["scope"] == scope and float(row["sigma"]) == float(sigma)
    ]
    if len(values) != 1:
        raise ValueError(f"V9.5 lacks group scope={scope} sigma={sigma}.")
    return int(values[0]["pass"])


def load_spatial_scope_config(path: str | Path) -> SpatialScopeConfig:
    source = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf8")) or {}
    repository = source.parents[2]
    edge_raw = raw.get("fixed_history_edges", {})
    config = SpatialScopeConfig(
        source_path=source,
        v93_config=_resolve_path(
            repository,
            raw.get(
                "v93_config",
                "streaming_couping/configs/v93_quantization_tolerance.yaml",
            ),
        ),
        output_dir=_resolve_path(
            repository,
            raw.get(
                "output_dir",
                "outputs/streaming_couping_v95_spatial_support_scope",
            ),
        ),
        support_scopes=tuple(str(value) for value in raw.get("support_scopes", [])),
        noise_sigmas_pixels=tuple(
            float(value) for value in raw.get("noise_sigmas_pixels", [])
        ),
        noise_replicates=int(raw.get("noise_replicates", 5)),
        noise_seed=int(raw.get("noise_seed", 95)),
        max_candidate_queries=int(raw.get("max_candidate_queries", 4096)),
        hybrid_background_fraction=float(
            raw.get("hybrid_background_fraction", 0.5)
        ),
        fixed_history_edges={
            str(fold): {int(current): int(history) for current, history in edges.items()}
            for fold, edges in edge_raw.items()
        },
    )
    _validate_config(config)
    return config


def _validate_config(config: SpatialScopeConfig) -> None:
    if config.support_scopes != SUPPORT_SCOPES:
        raise ValueError("V9.5 support scopes are protocol-locked.")
    if config.noise_sigmas_pixels != (0.0, 0.5, 1.0):
        raise ValueError("V9.5 noise sigmas are protocol-locked to 0/0.5/1px.")
    if config.noise_replicates != 5 or config.noise_seed != 95:
        raise ValueError("V9.5 noise replicate/seed protocol changed.")
    if config.max_candidate_queries != 4096:
        raise ValueError("V9.5 full-image candidate count is protocol-locked.")
    if config.hybrid_background_fraction != 0.5:
        raise ValueError("V9.5 hybrid support must be 50% background.")
    expected_folds = {fold.name for fold in FOLDS}
    if set(config.fixed_history_edges) != expected_folds:
        raise ValueError("V9.5 fixed edge lock does not cover every fold.")
    for fold in FOLDS:
        edges = config.fixed_history_edges[fold.name]
        if set(edges) != set(fold.test_frames):
            raise ValueError(f"V9.5 fixed edge lock differs for fold={fold.name}.")
        if any(history >= current for current, history in edges.items()):
            raise ValueError("V9.5 fixed history edges must be strictly causal.")


def _locked_edge_indices(
    config: SpatialScopeConfig, data: SupportData
) -> dict[str, dict[int, int]]:
    positions = {frame: index for index, frame in enumerate(data.frames)}
    return {
        fold: {
            positions[current_frame]: positions[history_frame]
            for current_frame, history_frame in edges.items()
        }
        for fold, edges in config.fixed_history_edges.items()
    }


def _frame_edge_map(
    values: dict[str, dict[int, int]], data: SupportData
) -> dict[str, dict[str, int]]:
    return {
        fold: {
            str(data.frames[current]): data.frames[history]
            for current, history in edges.items()
        }
        for fold, edges in values.items()
    }


def _validate_outputs(
    summaries: Sequence[dict[str, object]],
    frames: Sequence[dict[str, object]],
    *,
    config: SpatialScopeConfig,
) -> None:
    expected_summaries = (
        len(config.support_scopes) * len(config.noise_sigmas_pixels) * len(FOLDS)
    )
    repetitions = 1 + config.noise_replicates * (
        len(config.noise_sigmas_pixels) - 1
    )
    expected_frames = len(config.support_scopes) * repetitions * sum(
        len(fold.test_frames) for fold in FOLDS
    )
    if len(summaries) != expected_summaries:
        raise ValueError(
            f"V9.5 summary rows={len(summaries)}, expected={expected_summaries}."
        )
    if len(frames) != expected_frames:
        raise ValueError(f"V9.5 frame rows={len(frames)}, expected={expected_frames}.")


def _write_csv(path: Path, rows, columns) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty V9.5 CSV: {path.name}")
    expected = set(columns)
    for index, row in enumerate(rows):
        if set(row) != expected:
            raise ValueError(
                f"V9.5 CSV {path.name} row={index} mismatch: "
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


def _gain(initial: float, final: float) -> float:
    if not math.isfinite(initial) or not math.isfinite(final) or initial <= 1e-12:
        return 0.0
    return 100.0 * (initial - final) / initial


def _resolve_path(repository: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repository / path).resolve()


def _jsonable_config(config: SpatialScopeConfig) -> dict[str, object]:
    value = asdict(config)
    value["source_path"] = str(config.source_path)
    value["v93_config"] = str(config.v93_config)
    value["output_dir"] = str(config.output_dir)
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="streaming_couping/configs/v95_spatial_support_scope.yaml",
    )
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
