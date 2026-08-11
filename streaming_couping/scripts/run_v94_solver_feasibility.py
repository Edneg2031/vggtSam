#!/usr/bin/env python3
"""Run V9.4 fixed-edge robust-solver feasibility diagnosis."""

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
    BranchSpec,
    _make_prediction,
    _select_continuous_fixed_edges,
    load_diagnostic_config,
)
from streaming_couping.src.v74_temporal_protocol import FOLDS
from streaming_couping.src.v80_pose_geometry import (
    invert_rigid,
    rotation_error_degrees,
)
from streaming_couping.src.v90_epipolar_geometry import (
    concatenate_correspondences,
    relative_translation_direction_error_degrees,
)
from streaming_couping.src.v93_quantization_tolerance import PredictionDiagnostics
from streaming_couping.src.v94_robust_epipolar import (
    ROBUST_SOLVERS,
    RobustEpipolarConfig,
    estimate_robust_relative_pose,
)


EVIDENCE_BRANCH_NAMES = (
    "continuous_gt",
    "soft_convex_k8",
    "continuous_noise_sigma_0p5px",
    "continuous_noise_sigma_1p0px",
)

SUMMARY_COLUMNS = (
    "fold",
    "evidence_branch",
    "evidence_family",
    "evidence_parameter",
    "solver",
    "replicates",
    "fixed_history_edges",
    "evaluation_rows",
    "active_rows",
    "inactive_rows",
    "visible_queries",
    "accepted_correspondences",
    "visible_pck_accuracy",
    "selected_mean_epe_pixels",
    "mean_correspondences",
    "mean_consensus_inliers",
    "mean_consensus_inlier_fraction",
    "mean_msac_objective",
    "mean_sampson_rmse",
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
    "evidence_branch",
    "solver",
    "replicate",
    "sequence_index",
    "frame_index",
    "history_sequence_index",
    "history_frame_index",
    "participating_slots",
    "correspondences",
    "consensus_inliers",
    "consensus_inlier_fraction",
    "msac_objective",
    "hypotheses_tested",
    "active",
    "solver_reason",
    "solver_initialization",
    "sampson_rmse",
    "raw_rotation_error_deg",
    "refined_rotation_error_deg",
    "raw_translation_direction_error_deg",
    "refined_translation_direction_error_deg",
    "relative_aggregate_worse",
)


@dataclass(frozen=True)
class SolverExperimentConfig:
    source_path: Path
    v93_config: Path
    output_dir: Path
    robust: RobustEpipolarConfig


def main() -> None:
    args = _parse_args()
    config = load_solver_config(args.config)
    if args.output_dir:
        config = replace(
            config, output_dir=Path(args.output_dir).expanduser().resolve()
        )
    result = run_solver_feasibility(config)
    print(f"V9.4 solver feasibility result={result}")


def run_solver_feasibility(config: SolverExperimentConfig) -> Path:
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
    fixed_edges = {
        fold.name: _select_continuous_fixed_edges(
            fold_name=fold.name,
            test_frames=fold.test_frames,
            records=all_records[fold.name],
            data=data,
            stage_config=stage_config,
        )
        for fold in FOLDS
    }
    branches = _evidence_branches(v93_config)
    summaries: list[dict[str, object]] = []
    frames: list[dict[str, object]] = []
    for branch in branches:
        for solver in ROBUST_SOLVERS:
            for fold in FOLDS:
                mapping = fixed_edges[fold.name]
                records = [
                    record
                    for record in all_records[fold.name]
                    if mapping.get(record.current) == record.history
                ]
                summary, rows = _evaluate_solver(
                    fold_name=fold.name,
                    test_frames=fold.test_frames,
                    branch=branch,
                    solver=solver,
                    records=records,
                    fixed_edges=mapping,
                    data=data,
                    stage_config=stage_config,
                    v93_config=v93_config,
                    config=config,
                )
                summaries.append(summary)
                frames.extend(rows)
    _annotate_passes(summaries)
    _validate_outputs(summaries, frames, branches=branches)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = config.output_dir / "v94_solver_summary.csv"
    frames_path = config.output_dir / "v94_solver_frames.csv"
    decision_path = config.output_dir / "v94_solver_decision.md"
    metadata_path = config.output_dir / "v94_solver_metadata.json"
    _write_csv(summary_path, summaries, SUMMARY_COLUMNS)
    _write_csv(frames_path, frames, FRAME_COLUMNS)
    decision_path.write_text(_decision_markdown(summaries), encoding="utf8")
    metadata_path.write_text(
        json.dumps(
            {
                "experiment": "V9.4 robust solver feasibility",
                "config": _jsonable_config(config),
                "v93_config": str(v93_config.source_path),
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
                "trains_matcher": False,
                "trains_pose_model": False,
                "uses_pose_loss": False,
                "reads_gt_pose_error_for_selection": False,
                "robust_hypotheses_deterministic": True,
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
    print(f"V9.4 wrote summary={summary_path}")
    print(f"V9.4 wrote decision={decision_path}")
    return summary_path


def _evaluate_solver(
    *,
    fold_name: str,
    test_frames: Sequence[int],
    branch: BranchSpec,
    solver: str,
    records: Sequence[SupportRecord],
    fixed_edges: dict[int, int],
    data: SupportData,
    stage_config,
    v93_config,
    config: SolverExperimentConfig,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    output_rows = []
    diagnostics: list[PredictionDiagnostics] = []
    errors = []
    by_current = {}
    for record in records:
        by_current.setdefault(record.current, []).append(record)
    for replicate in range(branch.replicates):
        predictions = {}
        for record in records:
            prediction = _make_prediction(
                record,
                branch=branch,
                replicate=replicate,
                data=data,
                stage_config=stage_config,
                config=v93_config,
            )
            predictions[id(record)] = prediction
            diagnostics.append(prediction.diagnostics)
            if int(prediction.selected_errors_pixels.numel()):
                errors.append(prediction.selected_errors_pixels)
        for frame_value in test_frames:
            current = data.frames.index(int(frame_value))
            frame_records = by_current.get(current, [])
            history = fixed_edges.get(current)
            pairs = [
                predictions[id(record)].correspondences for record in frame_records
            ]
            current_uv, history_uv, weights = concatenate_correspondences(pairs)
            raw_r = raw_t = float("nan")
            refined_r = refined_t = float("nan")
            result = None
            if history is not None:
                l0_relative = data.baseline[history] @ invert_rigid(
                    data.baseline[current]
                )
                target_relative = data.target[history] @ invert_rigid(
                    data.target[current]
                )
                raw_r = float(rotation_error_degrees(l0_relative, target_relative))
                raw_t = relative_translation_direction_error_degrees(
                    l0_relative[:3, :3], l0_relative[:3, 3], target_relative
                )
                result = estimate_robust_relative_pose(
                    current_uv,
                    history_uv,
                    weights,
                    data.intrinsics[current],
                    data.intrinsics[history],
                    l0_relative,
                    solver=solver,
                    epipolar_config=stage_config.epipolar,
                    robust_config=config.robust,
                    image_size=data.image_size,
                    seed_offset=(
                        current * 100_003
                        + history * 1_009
                        + replicate * 10_000_019
                    ),
                )
                refined_r, refined_t = raw_r, raw_t
                if result.estimate.success:
                    candidate = torch.eye(4, dtype=torch.float64)
                    candidate[:3, :3] = result.estimate.rotation_current_to_history
                    refined_r = float(rotation_error_degrees(candidate, target_relative))
                    refined_t = relative_translation_direction_error_degrees(
                        result.estimate.rotation_current_to_history,
                        result.estimate.translation_current_origin_in_history,
                        target_relative,
                    )
            active = int(result is not None and result.estimate.success)
            if result is None:
                inliers = hypotheses = 0
                fraction = 0.0
                objective = sampson = float("nan")
                reason = "missing_fixed_edge"
                initialization = "none"
            else:
                inliers = result.consensus.inliers
                hypotheses = result.consensus.hypotheses_tested
                fraction = result.consensus.inlier_fraction
                objective = result.consensus.msac_objective
                sampson = result.estimate.sampson_rmse
                reason = result.estimate.reason
                initialization = result.estimate.initialization
            raw_pose = raw_r + raw_t
            refined_pose = refined_r + refined_t
            output_rows.append(
                {
                    "fold": fold_name,
                    "evidence_branch": branch.name,
                    "solver": solver,
                    "replicate": replicate,
                    "sequence_index": current,
                    "frame_index": int(frame_value),
                    "history_sequence_index": "" if history is None else history,
                    "history_frame_index": (
                        "" if history is None else data.frames[history]
                    ),
                    "participating_slots": " ".join(
                        str(value)
                        for value in sorted(
                            {
                                record.slot
                                for record, pair in zip(frame_records, pairs)
                                if pair.count
                            }
                        )
                    ),
                    "correspondences": int(current_uv.shape[0]),
                    "consensus_inliers": inliers,
                    "consensus_inlier_fraction": fraction,
                    "msac_objective": objective,
                    "hypotheses_tested": hypotheses,
                    "active": active,
                    "solver_reason": reason,
                    "solver_initialization": initialization,
                    "sampson_rmse": sampson,
                    "raw_rotation_error_deg": raw_r,
                    "refined_rotation_error_deg": refined_r,
                    "raw_translation_direction_error_deg": raw_t,
                    "refined_translation_direction_error_deg": refined_t,
                    "relative_aggregate_worse": int(
                        math.isfinite(raw_pose)
                        and math.isfinite(refined_pose)
                        and refined_pose > raw_pose + 1e-12
                    ),
                }
            )
    totals = _sum_diagnostics(diagnostics)
    selected_errors = torch.cat(errors) if errors else torch.empty(0)
    raw_r = _finite_mean(float(row["raw_rotation_error_deg"]) for row in output_rows)
    refined_r = _finite_mean(
        float(row["refined_rotation_error_deg"]) for row in output_rows
    )
    raw_t = _finite_mean(
        float(row["raw_translation_direction_error_deg"]) for row in output_rows
    )
    refined_t = _finite_mean(
        float(row["refined_translation_direction_error_deg"]) for row in output_rows
    )
    raw_pose = raw_r + raw_t
    refined_pose = refined_r + refined_t
    fixed_text = " ".join(
        f"{data.frames[current]}:{data.frames[history]}"
        for current, history in sorted(fixed_edges.items())
    )
    summary = {
        "fold": fold_name,
        "evidence_branch": branch.name,
        "evidence_family": branch.family,
        "evidence_parameter": branch.parameter,
        "solver": solver,
        "replicates": branch.replicates,
        "fixed_history_edges": fixed_text,
        "evaluation_rows": len(output_rows),
        "active_rows": sum(int(row["active"]) for row in output_rows),
        "inactive_rows": sum(not int(row["active"]) for row in output_rows),
        "visible_queries": totals.visible_queries,
        "accepted_correspondences": totals.accepted_correspondences,
        "visible_pck_accuracy": _ratio(totals.pck_correct, totals.visible_queries),
        "selected_mean_epe_pixels": _finite_mean(selected_errors.tolist()),
        "mean_correspondences": _finite_mean(
            float(row["correspondences"]) for row in output_rows
        ),
        "mean_consensus_inliers": _finite_mean(
            float(row["consensus_inliers"]) for row in output_rows
        ),
        "mean_consensus_inlier_fraction": _finite_mean(
            float(row["consensus_inlier_fraction"]) for row in output_rows
        ),
        "mean_msac_objective": _finite_mean(
            float(row["msac_objective"]) for row in output_rows
        ),
        "mean_sampson_rmse": _finite_mean(
            float(row["sampson_rmse"]) for row in output_rows
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
            int(row["relative_aggregate_worse"]) for row in output_rows
        ),
        "fold_pass": 0,
        "all_folds_pass": 0,
    }
    return summary, output_rows


def _evidence_branches(v93_config) -> tuple[BranchSpec, ...]:
    candidates = (
        BranchSpec("continuous_gt", "continuous_positive_control", "exact"),
        BranchSpec("soft_convex_k8", "soft_convex_upper_bound", 8),
        BranchSpec(
            "continuous_noise_sigma_0p5px",
            "continuous_noise",
            0.5,
            replicates=v93_config.noise_replicates,
        ),
        BranchSpec(
            "continuous_noise_sigma_1p0px",
            "continuous_noise",
            1.0,
            replicates=v93_config.noise_replicates,
        ),
    )
    if tuple(branch.name for branch in candidates) != EVIDENCE_BRANCH_NAMES:
        raise RuntimeError("V9.4 evidence branch lock changed.")
    return candidates


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
    keys = sorted({(str(row["evidence_branch"]), str(row["solver"])) for row in rows})
    for evidence, solver in keys:
        group = [
            row
            for row in rows
            if row["evidence_branch"] == evidence and row["solver"] == solver
        ]
        if len(group) != len(FOLDS):
            raise ValueError(f"V9.4 evidence={evidence} solver={solver} lacks folds.")
        passed = int(all(int(row["fold_pass"]) for row in group))
        for row in group:
            row["all_folds_pass"] = passed


def _strict_pass(row: dict[str, object]) -> bool:
    return bool(
        int(row["active_rows"]) == int(row["evaluation_rows"])
        and float(row["refined_rotation_error_deg"])
        < float(row["raw_rotation_error_deg"])
        and float(row["refined_translation_direction_error_deg"])
        < float(row["raw_translation_direction_error_deg"])
        and int(row["relative_aggregate_worse_rows"]) == 0
    )


def _decision_markdown(rows: Sequence[dict[str, object]]) -> str:
    groups = []
    for evidence in EVIDENCE_BRANCH_NAMES:
        for solver in ROBUST_SOLVERS:
            values = [
                row
                for row in rows
                if row["evidence_branch"] == evidence and row["solver"] == solver
            ]
            groups.append(
                {
                    "evidence": evidence,
                    "solver": solver,
                    "inlier": _finite_mean(
                        float(row["mean_consensus_inlier_fraction"]) for row in values
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
    robust = [solver for solver in ROBUST_SOLVERS if solver != "o_r1"]
    continuous_control = _group_pass(groups, "continuous_gt", "o_r1")
    soft_pass = int(
        any(_group_pass(groups, "soft_convex_k8", solver) for solver in robust)
    )
    noise05_pass = int(
        any(
            _group_pass(groups, "continuous_noise_sigma_0p5px", solver)
            for solver in robust
        )
    )
    noise10_pass = int(
        any(
            _group_pass(groups, "continuous_noise_sigma_1p0px", solver)
            for solver in robust
        )
    )
    route_viable = int(continuous_control and soft_pass and noise05_pass)
    lines = [
        "# V9.4 robust-solver feasibility decision",
        "",
        "No matcher or pose model is trained. Evidence and per-frame history edges are frozen from V9.3.",
        "RANSAC hypotheses, spatial sampling and inlier refinement are deterministic and never read GT pose error.",
        "",
        f"- continuous O-R1 positive-control pass: `{continuous_control}`",
        f"- any robust solver soft-K8 all-fold pass: `{soft_pass}`",
        f"- any robust solver 0.5px-noise all-fold pass: `{noise05_pass}`",
        f"- any robust solver 1.0px-noise all-fold pass: `{noise10_pass}`",
        f"- instance-essential route solver-feasible: `{route_viable}`",
        "",
        "| evidence | solver | inlier fraction | mean R gain | mean t-dir gain | worse | all-fold pass |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in groups:
        lines.append(
            f"| {row['evidence']} | {row['solver']} "
            f"| {float(row['inlier']):.6g} | {float(row['r_gain']):.6g} "
            f"| {float(row['t_gain']):.6g} | {row['worse']} | {row['pass']} |"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- route=1: robust recovery tolerates the V9.3 upper-bound error; only then is a sub-pixel SAM matcher worth testing.",
            "- soft-K8=0 or noise-0.5=0 for every solver: terminate the localized-instance essential route rather than tune descriptors.",
            "- continuous control=0: implementation/protocol regression; no solver conclusion is valid.",
            "- Robust-oracle success would prove solver feasibility only, never SAM descriptor benefit.",
            "",
        ]
    )
    return "\n".join(lines)


def _group_pass(groups, evidence: str, solver: str) -> int:
    candidates = [
        row for row in groups if row["evidence"] == evidence and row["solver"] == solver
    ]
    if len(candidates) != 1:
        raise ValueError(f"V9.4 lacks group evidence={evidence} solver={solver}.")
    return int(candidates[0]["pass"])


def load_solver_config(path: str | Path) -> SolverExperimentConfig:
    source = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf8")) or {}
    repository = source.parents[2]
    robust = raw.get("robust_solver", {})
    config = SolverExperimentConfig(
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
                "outputs/streaming_couping_v94_solver_feasibility",
            ),
        ),
        robust=RobustEpipolarConfig(
            iterations=int(robust.get("iterations", 128)),
            threshold_pixels=float(robust.get("threshold_pixels", 1.5)),
            sample_size=int(robust.get("sample_size", 8)),
            spatial_candidate_pool=int(robust.get("spatial_candidate_pool", 32)),
            seed=int(robust.get("seed", 94)),
        ),
    )
    config.robust.validate()
    if config.robust != RobustEpipolarConfig():
        raise ValueError("V9.4 robust solver protocol is locked.")
    return config


def _validate_outputs(
    summaries: Sequence[dict[str, object]],
    frames: Sequence[dict[str, object]],
    *,
    branches: Sequence[BranchSpec],
) -> None:
    expected_summary = len(branches) * len(ROBUST_SOLVERS) * len(FOLDS)
    expected_frames = sum(
        branch.replicates * len(ROBUST_SOLVERS) * len(fold.test_frames)
        for branch in branches
        for fold in FOLDS
    )
    if len(summaries) != expected_summary:
        raise ValueError(
            f"V9.4 summary rows={len(summaries)}, expected={expected_summary}."
        )
    if len(frames) != expected_frames:
        raise ValueError(f"V9.4 frame rows={len(frames)}, expected={expected_frames}.")


def _write_csv(path: Path, rows, columns) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty V9.4 CSV: {path.name}")
    expected = set(columns)
    for index, row in enumerate(rows):
        if set(row) != expected:
            raise ValueError(
                f"V9.4 CSV {path.name} row={index} mismatch: "
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


def _jsonable_config(config: SolverExperimentConfig) -> dict[str, object]:
    value = asdict(config)
    for name in ("source_path", "v93_config", "output_dir"):
        value[name] = str(value[name])
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="streaming_couping/configs/v94_solver_feasibility.yaml",
    )
    parser.add_argument("--output-dir")
    return parser.parse_args()


if __name__ == "__main__":
    main()
