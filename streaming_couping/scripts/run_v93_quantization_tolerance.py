#!/usr/bin/env python3
"""Run V9.3 fixed-edge quantization/soft-decoder/noise diagnostics."""

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

from streaming_couping.scripts.run_v90_local_token_matcher import (
    A1_EDGE_SELECTION_POLICY,
    _pose_frame_rows,
    load_stage_a_config,
)
from streaming_couping.scripts.run_v92_support_factorization import (
    SupportData,
    SupportRecord,
    _build_support_records,
    _load_support_payload,
    _prepare_support_data,
    load_support_config,
)
from streaming_couping.src.v74_temporal_protocol import FOLDS
from streaming_couping.src.v93_quantization_tolerance import (
    OraclePrediction,
    PredictionDiagnostics,
    continuous_prediction,
    filter_prediction_by_oracle_error,
    hard_nearest_prediction,
    noisy_continuous_prediction,
    soft_knn_convex_prediction,
)


SUMMARY_COLUMNS = (
    "fold",
    "branch",
    "family",
    "parameter",
    "replicates",
    "fixed_edge_frames",
    "fixed_history_edges",
    "evaluation_rows",
    "active_rows",
    "inactive_rows",
    "visible_queries",
    "accepted_correspondences",
    "acceptance_fraction",
    "exact_interpolation_fraction",
    "pck_threshold_pixels",
    "visible_pck_accuracy",
    "visible_mean_epe_pixels",
    "selected_mean_epe_pixels",
    "selected_epe_p50_pixels",
    "selected_epe_p90_pixels",
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
    "relative_aggregate_worse_rows",
    "fold_pass",
    "all_folds_pass",
)

FRAME_COLUMNS = (
    "fold",
    "branch",
    "family",
    "parameter",
    "replicate",
    "sequence_index",
    "frame_index",
    "fixed_history_sequence_index",
    "fixed_history_frame_index",
    "observed_slots",
    "active",
    "visible_queries",
    "accepted_correspondences",
    "visible_pck_accuracy",
    "selected_mean_epe_pixels",
    "current_hull_coverage",
    "history_hull_coverage",
    "design_condition",
    "raw_rotation_error_deg",
    "refined_rotation_error_deg",
    "raw_translation_direction_error_deg",
    "refined_translation_direction_error_deg",
    "relative_aggregate_worse",
)


@dataclass(frozen=True)
class DiagnosticConfig:
    source_path: Path
    support_config: Path
    output_dir: Path
    soft_neighbors: tuple[int, ...]
    filter_thresholds_pixels: tuple[float, ...]
    noise_sigmas_pixels: tuple[float, ...]
    noise_replicates: int
    noise_seed: int


@dataclass(frozen=True)
class BranchSpec:
    name: str
    family: str
    parameter: float | int | str
    replicates: int = 1


BRANCH_BASE = (
    BranchSpec("continuous_gt", "continuous_positive_control", "exact"),
    BranchSpec("hard_nearest_k256", "hard_discrete", 256),
)


def main() -> None:
    args = _parse_args()
    config = load_diagnostic_config(args.config)
    if args.output_dir:
        config = replace(
            config, output_dir=Path(args.output_dir).expanduser().resolve()
        )
    result = run_quantization_tolerance(config)
    print(f"V9.3 quantization tolerance result={result}")


def run_quantization_tolerance(config: DiagnosticConfig) -> Path:
    support_config = load_support_config(config.support_config)
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
    records_by_fold = {
        fold.name: _build_support_records(
            data,
            current_indices=[positions[value] for value in fold.test_frames],
            query_token_count=support_config.query_token_count,
            stage_config=stage_config,
        )
        for fold in FOLDS
    }

    fixed_edges = {
        fold.name: _select_continuous_fixed_edges(
            fold_name=fold.name,
            test_frames=fold.test_frames,
            records=records_by_fold[fold.name],
            data=data,
            stage_config=stage_config,
        )
        for fold in FOLDS
    }
    branches = _branch_specs(config)
    summaries: list[dict[str, object]] = []
    frame_rows: list[dict[str, object]] = []
    for branch in branches:
        for fold in FOLDS:
            mapping = fixed_edges[fold.name]
            records = [
                record
                for record in records_by_fold[fold.name]
                if mapping.get(record.current) == record.history
            ]
            summary, frames = _evaluate_branch(
                fold_name=fold.name,
                test_frames=fold.test_frames,
                fixed_edges=mapping,
                records=records,
                branch=branch,
                data=data,
                support_config=support_config,
                stage_config=stage_config,
                config=config,
            )
            summaries.append(summary)
            frame_rows.extend(frames)
    _annotate_passes(summaries)
    _validate_outputs(summaries, frame_rows, branches=branches)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = config.output_dir / "v93_quantization_summary.csv"
    frames_path = config.output_dir / "v93_quantization_frames.csv"
    decision_path = config.output_dir / "v93_quantization_decision.md"
    metadata_path = config.output_dir / "v93_quantization_metadata.json"
    _write_csv(summary_path, summaries, SUMMARY_COLUMNS)
    _write_csv(frames_path, frame_rows, FRAME_COLUMNS)
    decision_path.write_text(_decision_markdown(summaries), encoding="utf8")
    metadata_path.write_text(
        json.dumps(
            {
                "experiment": "V9.3 fixed-edge quantization and tolerance diagnosis",
                "config": _jsonable_config(config),
                "support_config": str(support_config.source_path),
                "cache": {
                    "path": str(cache_file),
                    "size_bytes": cache_file.stat().st_size,
                    "mtime_ns": cache_file.stat().st_mtime_ns,
                },
                "fixed_edges": {
                    fold: {
                        str(data.frames[current]): data.frames[history]
                        for current, history in mapping.items()
                    }
                    for fold, mapping in fixed_edges.items()
                },
                "fixed_edge_selection": (
                    "continuous GT correspondence design condition only; "
                    "never GT pose error"
                ),
                "trains_matcher": False,
                "trains_pose_model": False,
                "uses_pose_loss": False,
                "soft_knn_is_deployable": False,
                "error_filter_is_deployable": False,
                "pose_solver": "frozen V9 O-R1 calibrated epipolar solver",
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
    print(f"V9.3 wrote summary={summary_path}")
    print(f"V9.3 wrote decision={decision_path}")
    return summary_path


def _select_continuous_fixed_edges(
    *,
    fold_name: str,
    test_frames: Sequence[int],
    records: Sequence[SupportRecord],
    data: SupportData,
    stage_config,
) -> dict[int, int]:
    predictions = {}
    match_stats = {}
    for record in records:
        prediction = continuous_prediction(
            record.labels,
            image_size=data.image_size,
            pck_threshold_pixels=stage_config.matcher.pck_threshold_pixels,
        )
        predictions[id(record)] = prediction.correspondences
        match_stats[id(record)] = prediction.diagnostics.as_pose_match_stats()
    frames, _ = _pose_frame_rows(
        stage="V9.3-edge-lock",
        fold_name=fold_name,
        architecture="continuous_gt_edge_selector",
        variant="best_design_condition_single",
        edge_selection_policy=A1_EDGE_SELECTION_POLICY,
        test_frames=test_frames,
        records=records,
        predictions=predictions,
        match_stats=match_stats,
        data=data,
        config=stage_config,
    )
    mapping = {}
    for row in frames:
        values = str(row["selected_history_sequence_indices"]).split()
        if len(values) > 1:
            raise ValueError("V9.3 fixed-edge selector returned multiple histories.")
        if values:
            mapping[int(row["sequence_index"])] = int(values[0])
    return mapping


def _evaluate_branch(
    *,
    fold_name: str,
    test_frames: Sequence[int],
    fixed_edges: dict[int, int],
    records: Sequence[SupportRecord],
    branch: BranchSpec,
    data: SupportData,
    support_config,
    stage_config,
    config: DiagnosticConfig,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    all_frames = []
    all_edges = []
    all_diagnostics: list[PredictionDiagnostics] = []
    all_errors = []
    compact_frames = []
    for replicate in range(branch.replicates):
        predictions = {}
        match_stats = {}
        per_record: dict[int, OraclePrediction] = {}
        for record in records:
            prediction = _make_prediction(
                record,
                branch=branch,
                replicate=replicate,
                data=data,
                stage_config=stage_config,
                config=config,
            )
            predictions[id(record)] = prediction.correspondences
            match_stats[id(record)] = prediction.diagnostics.as_pose_match_stats()
            per_record[id(record)] = prediction
            all_diagnostics.append(prediction.diagnostics)
            if int(prediction.selected_errors_pixels.numel()):
                all_errors.append(prediction.selected_errors_pixels)
        pose_frames, edge_rows = _pose_frame_rows(
            stage="V9.3",
            fold_name=fold_name,
            architecture=branch.family,
            variant=branch.name,
            edge_selection_policy=A1_EDGE_SELECTION_POLICY,
            test_frames=test_frames,
            records=records,
            predictions=predictions,
            match_stats=match_stats,
            data=data,
            config=stage_config,
        )
        all_frames.extend(pose_frames)
        all_edges.extend(edge_rows)
        compact_frames.extend(
            _compact_frames(
                fold_name=fold_name,
                branch=branch,
                replicate=replicate,
                pose_frames=pose_frames,
                edge_rows=edge_rows,
                records=records,
                predictions=per_record,
                fixed_edges=fixed_edges,
                data=data,
            )
        )

    totals = _sum_diagnostics(all_diagnostics)
    errors = torch.cat(all_errors) if all_errors else torch.empty(0)
    raw_r = _finite_mean(
        float(row["raw_edge_rotation_error_deg"]) for row in all_frames
    )
    refined_r = _finite_mean(
        float(row["refined_edge_rotation_error_deg"]) for row in all_frames
    )
    raw_t = _finite_mean(
        float(row["raw_edge_translation_direction_error_deg"]) for row in all_frames
    )
    refined_t = _finite_mean(
        float(row["refined_edge_translation_direction_error_deg"])
        for row in all_frames
    )
    raw_pose = raw_r + raw_t
    refined_pose = refined_r + refined_t
    fixed_text = " ".join(
        f"{data.frames[current]}:{data.frames[history]}"
        for current, history in sorted(fixed_edges.items())
    )
    summary = {
        "fold": fold_name,
        "branch": branch.name,
        "family": branch.family,
        "parameter": branch.parameter,
        "replicates": branch.replicates,
        "fixed_edge_frames": len(fixed_edges),
        "fixed_history_edges": fixed_text,
        "evaluation_rows": len(all_frames),
        "active_rows": sum(int(row["active"]) for row in all_frames),
        "inactive_rows": sum(not int(row["active"]) for row in all_frames),
        "visible_queries": totals.visible_queries,
        "accepted_correspondences": totals.accepted_correspondences,
        "acceptance_fraction": _ratio(
            totals.accepted_correspondences, totals.visible_queries
        ),
        "exact_interpolation_fraction": _ratio(
            totals.exact_interpolations, totals.accepted_correspondences
        ),
        "pck_threshold_pixels": stage_config.matcher.pck_threshold_pixels,
        "visible_pck_accuracy": _ratio(totals.pck_correct, totals.visible_queries),
        "visible_mean_epe_pixels": _ratio(
            totals.epe_sum_pixels, totals.visible_queries
        ),
        "selected_mean_epe_pixels": _ratio(
            totals.selected_epe_sum_pixels, totals.accepted_correspondences
        ),
        "selected_epe_p50_pixels": _quantile(errors, 0.50),
        "selected_epe_p90_pixels": _quantile(errors, 0.90),
        "mean_current_hull_coverage": _finite_mean(
            float(row["current_uv_hull_coverage_fraction"]) for row in all_edges
        ),
        "mean_history_hull_coverage": _finite_mean(
            float(row["history_uv_hull_coverage_fraction"]) for row in all_edges
        ),
        "mean_log10_design_condition": _finite_mean(
            math.log10(float(row["design_condition"]))
            for row in all_edges
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
            int(row["relative_aggregate_worse"]) for row in all_frames
        ),
        "fold_pass": 0,
        "all_folds_pass": 0,
    }
    return summary, compact_frames


def _make_prediction(
    record: SupportRecord,
    *,
    branch: BranchSpec,
    replicate: int,
    data: SupportData,
    stage_config,
    config: DiagnosticConfig,
) -> OraclePrediction:
    common = {
        "image_size": data.image_size,
        "pck_threshold_pixels": stage_config.matcher.pck_threshold_pixels,
    }
    history = {
        "history_uv_normalized": data.local_uv[record.history, record.slot, :256],
        "history_valid": data.local_valid[record.history, record.slot, :256],
        "match_radius_pixels": stage_config.matcher.target_radius_pixels,
    }
    if branch.family == "continuous_positive_control":
        return continuous_prediction(record.labels, **common)
    if branch.family == "hard_discrete":
        return hard_nearest_prediction(record.labels, **history, **common)
    if branch.family == "soft_convex_upper_bound":
        return soft_knn_convex_prediction(
            record.labels, neighbors=int(branch.parameter), **history, **common
        )
    if branch.family == "oracle_error_filter":
        hard = hard_nearest_prediction(record.labels, **history, **common)
        return filter_prediction_by_oracle_error(
            hard,
            record.labels,
            max_error_pixels=float(branch.parameter),
            **common,
        )
    if branch.family == "continuous_noise":
        seed = (
            int(config.noise_seed)
            + int(replicate) * 1_000_003
            + int(record.current) * 10_007
            + int(record.history) * 101
            + int(record.slot)
        )
        return noisy_continuous_prediction(
            record.labels,
            sigma_pixels=float(branch.parameter),
            seed=seed,
            **common,
        )
    raise ValueError(f"Unknown V9.3 branch family={branch.family!r}.")


def _compact_frames(
    *,
    fold_name: str,
    branch: BranchSpec,
    replicate: int,
    pose_frames: Sequence[dict[str, object]],
    edge_rows: Sequence[dict[str, object]],
    records: Sequence[SupportRecord],
    predictions: dict[int, OraclePrediction],
    fixed_edges: dict[int, int],
    data: SupportData,
) -> list[dict[str, object]]:
    output = []
    for pose in pose_frames:
        current = int(pose["sequence_index"])
        frame_records = [record for record in records if record.current == current]
        diagnostics = _sum_diagnostics(
            predictions[id(record)].diagnostics for record in frame_records
        )
        errors = [
            predictions[id(record)].selected_errors_pixels
            for record in frame_records
            if int(predictions[id(record)].selected_errors_pixels.numel())
        ]
        selected_errors = torch.cat(errors) if errors else torch.empty(0)
        edge = next(
            (row for row in edge_rows if int(row["sequence_index"]) == current),
            None,
        )
        history = fixed_edges.get(current)
        output.append(
            {
                "fold": fold_name,
                "branch": branch.name,
                "family": branch.family,
                "parameter": branch.parameter,
                "replicate": replicate,
                "sequence_index": current,
                "frame_index": int(pose["frame_index"]),
                "fixed_history_sequence_index": "" if history is None else history,
                "fixed_history_frame_index": (
                    "" if history is None else data.frames[history]
                ),
                "observed_slots": " ".join(
                    str(value) for value in sorted({r.slot for r in frame_records})
                ),
                "active": int(pose["active"]),
                "visible_queries": diagnostics.visible_queries,
                "accepted_correspondences": diagnostics.accepted_correspondences,
                "visible_pck_accuracy": _ratio(
                    diagnostics.pck_correct, diagnostics.visible_queries
                ),
                "selected_mean_epe_pixels": _finite_mean(selected_errors.tolist()),
                "current_hull_coverage": (
                    float("nan")
                    if edge is None
                    else float(edge["current_uv_hull_coverage_fraction"])
                ),
                "history_hull_coverage": (
                    float("nan")
                    if edge is None
                    else float(edge["history_uv_hull_coverage_fraction"])
                ),
                "design_condition": (
                    float("nan") if edge is None else float(edge["design_condition"])
                ),
                "raw_rotation_error_deg": float(
                    pose["raw_edge_rotation_error_deg"]
                ),
                "refined_rotation_error_deg": float(
                    pose["refined_edge_rotation_error_deg"]
                ),
                "raw_translation_direction_error_deg": float(
                    pose["raw_edge_translation_direction_error_deg"]
                ),
                "refined_translation_direction_error_deg": float(
                    pose["refined_edge_translation_direction_error_deg"]
                ),
                "relative_aggregate_worse": int(
                    pose["relative_aggregate_worse"]
                ),
            }
        )
    return output


def _branch_specs(config: DiagnosticConfig) -> tuple[BranchSpec, ...]:
    return (
        *BRANCH_BASE,
        *(
            BranchSpec(f"soft_convex_k{value}", "soft_convex_upper_bound", value)
            for value in config.soft_neighbors
        ),
        *(
            BranchSpec(
                f"hard_filter_epe_le_{int(value)}px",
                "oracle_error_filter",
                value,
            )
            for value in config.filter_thresholds_pixels
        ),
        *(
            BranchSpec(
                f"continuous_noise_sigma_{str(value).replace('.', 'p')}px",
                "continuous_noise",
                value,
                replicates=config.noise_replicates,
            )
            for value in config.noise_sigmas_pixels
        ),
    )


def _sum_diagnostics(rows: Iterable[PredictionDiagnostics]) -> PredictionDiagnostics:
    values = list(rows)
    return PredictionDiagnostics(
        **{
            name: sum(getattr(row, name) for row in values)
            for name in PredictionDiagnostics.__dataclass_fields__
        }
    )


def _annotate_passes(rows: list[dict[str, object]]) -> None:
    for row in rows:
        row["fold_pass"] = int(_strict_pass(row))
    for branch in sorted({str(row["branch"]) for row in rows}):
        group = [row for row in rows if row["branch"] == branch]
        if len(group) != len(FOLDS):
            raise ValueError(f"V9.3 branch={branch} lacks temporal folds.")
        passed = int(all(int(row["fold_pass"]) for row in group))
        for row in group:
            row["all_folds_pass"] = passed


def _strict_pass(row: dict[str, object]) -> bool:
    return bool(
        int(row["fixed_edge_frames"]) == len(FOLDS[0].test_frames)
        and int(row["active_rows"]) == int(row["evaluation_rows"])
        and float(row["refined_rotation_error_deg"])
        < float(row["raw_rotation_error_deg"])
        and float(row["refined_translation_direction_error_deg"])
        < float(row["raw_translation_direction_error_deg"])
        and int(row["relative_aggregate_worse_rows"]) == 0
    )


def _decision_markdown(rows: Sequence[dict[str, object]]) -> str:
    branches = []
    for name in dict.fromkeys(str(row["branch"]) for row in rows):
        group = [row for row in rows if row["branch"] == name]
        branches.append(
            {
                "name": name,
                "family": group[0]["family"],
                "parameter": group[0]["parameter"],
                "acceptance": _finite_mean(
                    float(row["acceptance_fraction"]) for row in group
                ),
                "pck": _finite_mean(
                    float(row["visible_pck_accuracy"]) for row in group
                ),
                "epe": _finite_mean(
                    float(row["selected_mean_epe_pixels"]) for row in group
                ),
                "r_gain": _finite_mean(
                    float(row["rotation_gain_percent"]) for row in group
                ),
                "t_gain": _finite_mean(
                    float(row["translation_direction_gain_percent"])
                    for row in group
                ),
                "worse": sum(
                    int(row["relative_aggregate_worse_rows"]) for row in group
                ),
                "pass": int(all(int(row["all_folds_pass"]) for row in group)),
            }
        )
    by_name = {row["name"]: row for row in branches}
    continuous_pass = int(by_name["continuous_gt"]["pass"])
    hard_pass = int(by_name["hard_nearest_k256"]["pass"])
    soft_pass = int(
        any(
            row["family"] == "soft_convex_upper_bound" and int(row["pass"])
            for row in branches
        )
    )
    filter_pass = int(
        any(
            row["family"] == "oracle_error_filter" and int(row["pass"])
            for row in branches
        )
    )
    passing_noise = [
        float(row["parameter"])
        for row in branches
        if row["family"] == "continuous_noise" and int(row["pass"])
    ]
    max_noise = max(passing_noise) if passing_noise else 0.0
    lines = [
        "# V9.3 fixed-edge quantization/tolerance decision",
        "",
        "No matcher or pose model is trained. Every branch uses the same per-frame causal history edge, frozen by continuous-GT design condition without reading GT pose error.",
        "Soft-convex and EPE-filter rows use GT and are diagnostic upper bounds only.",
        "",
        f"- continuous positive-control all-fold pass: `{continuous_pass}`",
        f"- hard nearest-256 all-fold pass: `{hard_pass}`",
        f"- any soft-convex representational upper bound pass: `{soft_pass}`",
        f"- any oracle EPE-filter upper bound pass: `{filter_pass}`",
        f"- maximum all-fold passing continuous-noise sigma: `{max_noise:g}px`",
        "",
        "| branch | acceptance | PCK@8 | selected EPE | mean R gain | mean t-dir gain | worse | all-fold pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in branches:
        lines.append(
            f"| {row['name']} | {float(row['acceptance']):.6g} "
            f"| {float(row['pck']):.6g} | {float(row['epe']):.6g} "
            f"| {float(row['r_gain']):.6g} | {float(row['t_gain']):.6g} "
            f"| {row['worse']} | {row['pass']} |"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- continuous=0: the earlier A0-H conclusion depended on branch-specific edge selection; repair the protocol before any token claim.",
            "- continuous=1 and soft=1 while hard=0: hard token quantization is the bottleneck; a sub-token soft coordinate decoder has a viable upper bound.",
            "- filter=1: a sufficiently precise subset works; a learned confidence/dustbin mechanism is required.",
            "- low noise tolerance: O-R1 plus localized instance support is intrinsically brittle to pixel error.",
            "- soft=0 despite near-zero EPE: spatial/epipolar conditioning, not token density or decoder expressivity, is the remaining bottleneck.",
            "- This remains a same-scene temporal oracle diagnosis and proves no SAM descriptor benefit.",
            "",
        ]
    )
    return "\n".join(lines)


def load_diagnostic_config(path: str | Path) -> DiagnosticConfig:
    source = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf8")) or {}
    repository = source.parents[2]
    config = DiagnosticConfig(
        source_path=source,
        support_config=_resolve_path(
            repository,
            raw.get(
                "support_config",
                "streaming_couping/configs/v92_support_factorization.yaml",
            ),
        ),
        output_dir=_resolve_path(
            repository,
            raw.get(
                "output_dir",
                "outputs/streaming_couping_v93_quantization_tolerance",
            ),
        ),
        soft_neighbors=tuple(int(v) for v in raw.get("soft_neighbors", (2, 4, 8))),
        filter_thresholds_pixels=tuple(
            float(v) for v in raw.get("filter_thresholds_pixels", (1, 2, 4, 8))
        ),
        noise_sigmas_pixels=tuple(
            float(v) for v in raw.get("noise_sigmas_pixels", (0.5, 1, 2, 4, 6))
        ),
        noise_replicates=int(raw.get("noise_replicates", 5)),
        noise_seed=int(raw.get("noise_seed", 93)),
    )
    if config.soft_neighbors != (2, 4, 8):
        raise ValueError("V9.3 locks soft K to 2/4/8.")
    if config.filter_thresholds_pixels != (1.0, 2.0, 4.0, 8.0):
        raise ValueError("V9.3 locks EPE filters to 1/2/4/8 px.")
    if config.noise_sigmas_pixels != (0.5, 1.0, 2.0, 4.0, 6.0):
        raise ValueError("V9.3 locks noise sigma to 0.5/1/2/4/6 px.")
    if config.noise_replicates != 5:
        raise ValueError("V9.3 locks five deterministic noise replicates.")
    return config


def _validate_outputs(
    summaries: Sequence[dict[str, object]],
    frames: Sequence[dict[str, object]],
    *,
    branches: Sequence[BranchSpec],
) -> None:
    expected_summary = len(branches) * len(FOLDS)
    expected_frames = sum(
        branch.replicates * len(fold.test_frames)
        for branch in branches
        for fold in FOLDS
    )
    if len(summaries) != expected_summary:
        raise ValueError(
            f"V9.3 summary rows={len(summaries)}, expected={expected_summary}."
        )
    if len(frames) != expected_frames:
        raise ValueError(f"V9.3 frame rows={len(frames)}, expected={expected_frames}.")
    keys = {(row["fold"], row["branch"]) for row in summaries}
    if len(keys) != expected_summary:
        raise ValueError("V9.3 summary contains duplicate experiment keys.")


def _write_csv(
    path: Path,
    rows: Sequence[dict[str, object]],
    columns: Sequence[str],
) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty V9.3 CSV: {path.name}")
    expected = set(columns)
    for index, row in enumerate(rows):
        if set(row) != expected:
            raise ValueError(
                f"V9.3 CSV {path.name} row={index} mismatch: "
                f"missing={sorted(expected - set(row))}, "
                f"extra={sorted(set(row) - expected)}"
            )
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _quantile(values: torch.Tensor, amount: float) -> float:
    if not int(values.numel()):
        return float("nan")
    return float(torch.quantile(values.double(), float(amount)))


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


def _jsonable_config(config: DiagnosticConfig) -> dict[str, object]:
    value = asdict(config)
    for name in ("source_path", "support_config", "output_dir"):
        value[name] = str(value[name])
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="streaming_couping/configs/v93_quantization_tolerance.yaml",
    )
    parser.add_argument("--output-dir")
    return parser.parse_args()


if __name__ == "__main__":
    main()
