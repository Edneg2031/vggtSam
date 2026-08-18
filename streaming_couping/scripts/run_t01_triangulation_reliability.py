#!/usr/bin/env python3
"""Diagnose whether fixed non-GT gates remove T0 triangulation long tails."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
import yaml

from streaming_couping.src.triangulation_reliability import (
    FIXED_BIN_LABELS,
    ReliabilityGate,
    fixed_bin_label,
    paired_error_metrics,
    reliability_gate_mask,
    reliability_gate_pass,
)


REVISION = "t01_triangulation_long_tail_reliability_diagnostic_r1"
SOURCE_REVISION = "t0_sam_indexed_independent_3d_anchor_probe_r1"
GATE_NAMES = ("before_gate", "primary_gate", "primary_gate_views_ge_3")
BIN_METRICS = (
    "mean_reprojection_error_px",
    "maximum_ray_angle_degrees",
    "condition_number",
    "num_views",
)


@dataclass(frozen=True)
class T01Run:
    source_path: Path
    t0_output_dir: Path
    output_dir: Path
    branch: str
    primary_gate: ReliabilityGate
    strict_gate: ReliabilityGate
    minimum_anchor_count: int
    minimum_retained_ratio: float


def main() -> None:
    args = _parse_args()
    run = _load_run(args.config)
    artifact_path = run.t0_output_dir / "anchors_candidate.pt"
    candidate = torch.load(artifact_path, map_location="cpu", weights_only=False)
    anchors = _validate_and_select_candidates(candidate, run.branch)

    run.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = _candidate_metrics(anchors)
    gate_masks = {
        "before_gate": torch.ones(
            int(anchors["point_world"].shape[0]), dtype=torch.bool
        ),
        "primary_gate": reliability_gate_mask(metrics, run.primary_gate),
        "primary_gate_views_ge_3": reliability_gate_mask(metrics, run.strict_gate),
    }
    gate_signature = _gate_signature(run, artifact_path)
    frozen_path = run.output_dir / "frozen_gate_assignments.pt"
    manifest_path = run.output_dir / "frozen_gate_manifest.json"
    torch.save(
        {
            "schema": 1,
            "revision": REVISION,
            "artifact_role": "gt_blind_frozen_reliability_gate_assignments",
            "source_candidate_artifact": str(artifact_path),
            "source_candidate_revision": str(candidate["revision"]),
            "source_branch": run.branch,
            "candidate_generation_raw_pointmap_fields": 0,
            "candidate_generation_gt_fields": 0,
            "positive_depth_source": "inferred_from_t0_acceptance_contract",
            "reprojection_definition": "t0_mean_per_view_reprojection_error_px",
            "gate_thresholds_selected_with_gt": 0,
            "gate_signature": gate_signature,
            "anchor_index": torch.arange(
                int(anchors["point_world"].shape[0]), dtype=torch.long
            ),
            "metrics": metrics,
            "gate_masks": gate_masks,
        },
        frozen_path,
    )
    manifest = {
        "schema": 1,
        "revision": REVISION,
        "artifact_role": "gt_blind_gate_freeze_manifest",
        "source_candidate_artifact": str(artifact_path),
        "source_candidate_revision": str(candidate["revision"]),
        "source_branch": run.branch,
        "candidate_anchor_count": int(anchors["point_world"].shape[0]),
        "candidate_generation_raw_pointmap_fields": 0,
        "candidate_generation_gt_fields": 0,
        "positive_depth_source": "inferred_from_t0_acceptance_contract",
        "reprojection_definition": "t0_mean_per_view_reprojection_error_px",
        "gate_thresholds_selected_with_gt": 0,
        "primary_gate": asdict(run.primary_gate),
        "strict_gate": asdict(run.strict_gate),
        "gate_signature": gate_signature,
        "frozen_gate_artifact": str(frozen_path),
    }
    _write_json(manifest_path, manifest)
    print("T0.1 TRIANGULATION LONG-TAIL RELIABILITY DIAGNOSTIC")
    print("  source=frozen T0 candidates; matching/model/pose/pointmap unchanged")
    print("  candidate gate fields=non-GT reliability metrics only")
    print(
        "  gates frozen before T0 GT-derived anchor scores are opened "
        f"signature={gate_signature[:12]}"
    )

    # Only this scoring phase opens the T0 table containing raw/GT errors.
    t0_summary = _load_t0_summary(run.t0_output_dir / "summary.json", run.branch)
    scored_rows = _load_and_validate_scores(
        run.t0_output_dir / "anchor_scores.csv",
        branch=run.branch,
        anchors=anchors,
        metrics=metrics,
        gate_masks=gate_masks,
    )
    summary, gate_rows, bin_rows, concentration_rows = _score_reliability(
        run,
        candidate=candidate,
        t0_summary=t0_summary,
        artifact_path=artifact_path,
        frozen_path=frozen_path,
        manifest_path=manifest_path,
        gate_signature=gate_signature,
        scored_rows=scored_rows,
    )
    _write_outputs(
        run,
        summary=summary,
        gate_rows=gate_rows,
        bin_rows=bin_rows,
        concentration_rows=concentration_rows,
        anchor_rows=scored_rows,
    )
    primary = summary["gate_lookup"]["primary_gate"]
    strict = summary["gate_lookup"]["primary_gate_views_ge_3"]
    print(
        "  primary "
        f"anchors={primary['anchor_count']} "
        f"tri_gain={primary['tri_gain_vs_raw_percent']:.4f}% "
        f"pass={primary['gate_pass']}"
    )
    print(
        "  views>=3 "
        f"anchors={strict['anchor_count']} "
        f"tri_gain={strict['tri_gain_vs_raw_percent']:.4f}% "
        f"pass={strict['gate_pass']}"
    )
    print(f"  decision={summary['decision']['t01_decision']}")
    print(f"  result={run.output_dir / 'summary.json'}")


def _validate_and_select_candidates(
    payload: dict[str, Any], branch: str
) -> dict[str, torch.Tensor]:
    if int(payload.get("schema", -1)) != 1:
        raise ValueError("T0 candidate artifact schema is not supported.")
    if payload.get("revision") != SOURCE_REVISION:
        raise ValueError("T0.1 requires the frozen T0 r1 candidate revision.")
    if payload.get("artifact_role") != "frozen_independent_3d_anchor_candidates":
        raise ValueError("Input is not a frozen T0 candidate artifact.")
    if int(payload.get("candidate_generation_raw_pointmap_fields", -1)) != 0:
        raise ValueError("T0 candidate artifact used raw pointmap fields.")
    if int(payload.get("candidate_generation_gt_fields", -1)) != 0:
        raise ValueError("T0 candidate artifact used GT fields.")
    branches = payload.get("branches", {})
    if branch not in branches or not isinstance(branches[branch], dict):
        raise ValueError(f"T0 candidate artifact lacks branch {branch!r}.")
    anchors = branches[branch]
    required = {
        "point_world",
        "current_sequence_index",
        "current_frame_index",
        "slot",
        "sam_track_id",
        "query_patch_flat_index",
        "query_pixel_xy",
        "num_views",
        "condition_number",
        "maximum_ray_angle_degrees",
        "mean_reprojection_error_px",
    }
    missing = required.difference(anchors)
    if missing:
        raise ValueError(f"T0 candidate branch is missing {sorted(missing)}.")
    count = int(torch.as_tensor(anchors["point_world"]).shape[0])
    if count < 1:
        raise ValueError("T0 candidate branch contains no anchors.")
    for name in required:
        if int(torch.as_tensor(anchors[name]).shape[0]) != count:
            raise ValueError(f"T0 candidate field {name!r} has the wrong length.")
    return anchors


def _candidate_metrics(anchors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    count = int(anchors["point_world"].shape[0])
    metrics = {
        "mean_reprojection_error_px": anchors[
            "mean_reprojection_error_px"
        ].detach().float().cpu(),
        "maximum_ray_angle_degrees": anchors[
            "maximum_ray_angle_degrees"
        ].detach().float().cpu(),
        "condition_number": anchors["condition_number"].detach().float().cpu(),
        "num_views": anchors["num_views"].detach().long().cpu(),
        # T0 only packed anchors after positive_depth_rate == 1 passed.
        "positive_depth": torch.ones(count, dtype=torch.bool),
    }
    for name in (
        "mean_reprojection_error_px",
        "maximum_ray_angle_degrees",
        "condition_number",
    ):
        if not torch.isfinite(metrics[name]).all():
            raise ValueError(f"Frozen T0 metric {name!r} contains non-finite values.")
    return metrics


def _load_t0_summary(path: Path, branch: str) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf8"))
    if summary.get("revision") != SOURCE_REVISION:
        raise ValueError("T0 summary revision disagrees with T0.1.")
    if summary.get("baseline_status") != "frozen_unchanged":
        raise ValueError("T0 source did not preserve the frozen V0 baseline.")
    if branch not in summary.get("branch_lookup", {}):
        raise ValueError("T0 summary lacks the configured source branch.")
    return summary


def _load_and_validate_scores(
    path: Path,
    *,
    branch: str,
    anchors: dict[str, torch.Tensor],
    metrics: dict[str, torch.Tensor],
    gate_masks: dict[str, torch.Tensor],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf8") as handle:
        for raw in csv.DictReader(handle):
            if raw.get("branch") != branch:
                continue
            index = int(raw["anchor_index"])
            if not 0 <= index < int(anchors["point_world"].shape[0]):
                raise ValueError("T0 score row references an invalid anchor index.")
            expected = {
                "num_views": int(metrics["num_views"][index]),
                "mean_reprojection_error_px": float(
                    metrics["mean_reprojection_error_px"][index]
                ),
                "maximum_ray_angle_degrees": float(
                    metrics["maximum_ray_angle_degrees"][index]
                ),
                "condition_number": float(metrics["condition_number"][index]),
            }
            if int(raw["num_views"]) != expected["num_views"]:
                raise ValueError("T0 score/candidate num_views mismatch.")
            for name in (
                "mean_reprojection_error_px",
                "maximum_ray_angle_degrees",
                "condition_number",
            ):
                if not math.isclose(
                    float(raw[name]),
                    float(expected[name]),
                    rel_tol=1e-5,
                    abs_tol=1e-5,
                ):
                    raise ValueError(f"T0 score/candidate {name} mismatch.")
            tri_error = float(raw["tri_gt_error"])
            raw_error = float(raw["raw_gt_error"])
            if not math.isfinite(tri_error) or not math.isfinite(raw_error):
                raise ValueError("T0 score table contains non-finite paired errors.")
            rows.append(
                {
                    "anchor_index": index,
                    "current_sequence_index": int(raw["current_sequence_index"]),
                    "current_frame_index": int(raw["current_frame_index"]),
                    "slot": int(raw["slot"]),
                    "sam_track_id": int(raw["sam_track_id"]),
                    **expected,
                    "positive_depth": 1,
                    "tri_gt_error": tri_error,
                    "raw_gt_error": raw_error,
                    "delta_raw_minus_tri": raw_error - tri_error,
                    "tri_improved": int(tri_error < raw_error),
                    **{
                        name: int(mask[index]) for name, mask in gate_masks.items()
                    },
                }
            )
    if not rows:
        raise ValueError("T0 score table has no rows for the configured branch.")
    indices = [int(row["anchor_index"]) for row in rows]
    if len(indices) != len(set(indices)):
        raise ValueError("T0 score table contains duplicate anchor indices.")
    return sorted(rows, key=lambda row: int(row["anchor_index"]))


def _score_reliability(
    run: T01Run,
    *,
    candidate: dict[str, Any],
    t0_summary: dict[str, Any],
    artifact_path: Path,
    frozen_path: Path,
    manifest_path: Path,
    gate_signature: str,
    scored_rows: list[dict[str, Any]],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    tri_all = torch.tensor([float(row["tri_gt_error"]) for row in scored_rows])
    raw_all = torch.tensor([float(row["raw_gt_error"]) for row in scored_rows])
    severe_threshold = float(torch.quantile(raw_all, 0.90))
    for row in scored_rows:
        row["severe_outlier"] = int(
            float(row["tri_gt_error"]) > severe_threshold
        )

    gate_rows = []
    for name in GATE_NAMES:
        selected = [row for row in scored_rows if int(row[name]) == 1]
        paired = _paired_from_rows(selected)
        paired.update(
            {
                "gate": name,
                "retained_ratio": len(selected) / len(scored_rows),
                "severe_outlier_count": sum(
                    int(row["severe_outlier"]) for row in selected
                ),
                "severe_outlier_rate": sum(
                    int(row["severe_outlier"]) for row in selected
                )
                / max(len(selected), 1),
            }
        )
        paired["gate_pass"] = int(
            name != "before_gate"
            and reliability_gate_pass(
                paired,
                minimum_anchor_count=run.minimum_anchor_count,
                minimum_retained_ratio=run.minimum_retained_ratio,
            )
        )
        gate_rows.append(paired)

    bin_rows = []
    for metric in BIN_METRICS:
        for label in FIXED_BIN_LABELS[metric]:
            selected = [
                row
                for row in scored_rows
                if fixed_bin_label(metric, row[metric]) == label
            ]
            paired = _paired_from_rows(selected)
            paired.update(
                {
                    "metric": metric,
                    "bin": label,
                    "severe_outlier_count": sum(
                        int(row["severe_outlier"]) for row in selected
                    ),
                    "severe_outlier_rate": sum(
                        int(row["severe_outlier"]) for row in selected
                    )
                    / max(len(selected), 1),
                }
            )
            bin_rows.append(paired)

    indicators = {
        "high_reprojection_gt_2px": lambda row: float(
            row["mean_reprojection_error_px"]
        )
        > run.primary_gate.maximum_mean_reprojection_px,
        "small_angle_lt_2deg": lambda row: float(
            row["maximum_ray_angle_degrees"]
        )
        < run.primary_gate.minimum_triangulation_angle_degrees,
        "high_condition_gt_1e3": lambda row: float(row["condition_number"])
        > run.primary_gate.maximum_condition_number,
        "two_view_track": lambda row: int(row["num_views"]) == 2,
        "primary_gate_rejected": lambda row: int(row["primary_gate"]) == 0,
        "strict_gate_rejected": lambda row: int(
            row["primary_gate_views_ge_3"]
        )
        == 0,
    }
    severe_total = sum(int(row["severe_outlier"]) for row in scored_rows)
    concentration_rows = []
    for name, predicate in indicators.items():
        selected = [row for row in scored_rows if predicate(row)]
        severe = sum(int(row["severe_outlier"]) for row in selected)
        concentration_rows.append(
            {
                "indicator": name,
                "indicator_anchor_count": len(selected),
                "indicator_prevalence": len(selected) / len(scored_rows),
                "severe_outlier_count": severe,
                "severe_outlier_rate_within_indicator": severe
                / max(len(selected), 1),
                "share_of_all_severe_outliers": severe / max(severe_total, 1),
            }
        )

    lookup = {row["gate"]: row for row in gate_rows}
    passing = [
        name
        for name in GATE_NAMES[1:]
        if int(lookup[name]["gate_pass"]) == 1
    ]
    decision_name = "GO" if passing else "NO_GO"
    decision = {
        "t01_decision": decision_name,
        "passing_gates": passing,
        "primary_gate_pass": int(lookup["primary_gate"]["gate_pass"]),
        "strict_views_ge_3_gate_pass": int(
            lookup["primary_gate_views_ge_3"]["gate_pass"]
        ),
        "gate_frozen_before_gt_scoring": 1,
        "gate_thresholds_selected_with_gt": 0,
        "formal_v0_pose_modified": 0,
        "formal_v0_pointmap_modified": 0,
        "formal_v0_semantic_map_modified": 0,
        "next_gate": (
            "local_pointmap_residual_refinement_with_reliable_sparse_anchors"
            if passing
            else "stop_training_free_pointmap_refinement_keep_v0"
        ),
    }
    summary = {
        "schema": 1,
        "revision": REVISION,
        "experiment": "triangulation_long_tail_reliability_diagnostic",
        "baseline_version": "v0",
        "baseline_status": "frozen_unchanged",
        "clip": t0_summary["clip"],
        "frames": t0_summary["frames"],
        "source_branch": run.branch,
        "source_t0_revision": candidate["revision"],
        "source_t0_decision": t0_summary["decision"]["t0_decision"],
        "source_candidate_anchor_count": int(
            candidate["branches"][run.branch]["point_world"].shape[0]
        ),
        "evaluated_anchor_count": len(scored_rows),
        "reprojection_definition": "t0_mean_per_view_reprojection_error_px",
        "positive_depth_source": "inferred_from_t0_acceptance_contract",
        "primary_gate": asdict(run.primary_gate),
        "strict_gate": asdict(run.strict_gate),
        "go_criteria": {
            "minimum_anchor_count": run.minimum_anchor_count,
            "minimum_retained_ratio": run.minimum_retained_ratio,
            "tri_rmse_less_than_raw_rmse": 1,
            "tri_p90_not_greater_than_raw_p90": 1,
            "improved_anchor_ratio_greater_than": 0.5,
        },
        "severe_outlier_definition": "tri_gt_error > before_gate_raw_error_p90",
        "severe_outlier_threshold": severe_threshold,
        "gates": gate_rows,
        "gate_lookup": lookup,
        "bins": bin_rows,
        "outlier_concentration": concentration_rows,
        "candidate_generation_raw_pointmap_fields": 0,
        "candidate_generation_gt_fields": 0,
        "source_t0_gt_scores_read_after_gate_freeze": 1,
        "model_loaded_or_run": 0,
        "model_trained": 0,
        "pose_modified": 0,
        "pointmap_modified": 0,
        "gate_signature": gate_signature,
        "source_candidate_artifact": str(artifact_path),
        "frozen_gate_artifact": str(frozen_path),
        "frozen_gate_manifest": str(manifest_path),
        "decision": decision,
        "claim": (
            "fixed_non_gt_gate_is_sufficient_for_reliable_sparse_anchors"
            if passing
            else "fixed_non_gt_gates_do_not_remove_triangulation_long_tail"
        ),
    }
    return summary, gate_rows, bin_rows, concentration_rows


def _paired_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return paired_error_metrics(
        torch.tensor([float(row["tri_gt_error"]) for row in rows]),
        torch.tensor([float(row["raw_gt_error"]) for row in rows]),
    )


def _write_outputs(
    run: T01Run,
    *,
    summary: dict[str, Any],
    gate_rows: list[dict[str, Any]],
    bin_rows: list[dict[str, Any]],
    concentration_rows: list[dict[str, Any]],
    anchor_rows: list[dict[str, Any]],
) -> None:
    _write_json(run.output_dir / "summary.json", summary)
    _write_csv(run.output_dir / "gate_summary.csv", gate_rows)
    _write_csv(run.output_dir / "bin_summary.csv", bin_rows)
    _write_csv(
        run.output_dir / "outlier_concentration.csv", concentration_rows
    )
    _write_csv(run.output_dir / "anchor_reliability_scores.csv", anchor_rows)
    _write_copyable(run.output_dir / "copyable_result.txt", summary)


def _write_copyable(path: Path, summary: dict[str, Any]) -> None:
    primary = summary["primary_gate"]
    strict = summary["strict_gate"]
    lines = [
        "===== COPYABLE_T01_TRIANGULATION_RELIABILITY_BEGIN =====",
        f"revision={summary['revision']}",
        f"clip={summary['clip']}",
        f"frames={len(summary['frames'])}",
        f"source_branch={summary['source_branch']}",
        f"source_candidate_anchor_count={summary['source_candidate_anchor_count']}",
        f"evaluated_anchor_count={summary['evaluated_anchor_count']}",
        f"reprojection_definition={summary['reprojection_definition']}",
        f"positive_depth_source={summary['positive_depth_source']}",
        "model_loaded_or_run=0",
        "model_trained=0",
        "candidate_generation_raw_pointmap_fields=0",
        "candidate_generation_gt_fields=0",
        "gate_frozen_before_gt_scoring=1",
        "gate_thresholds_selected_with_gt=0",
        "formal_v0_modified=0",
        f"gate_signature={summary['gate_signature']}",
        "",
        "gate definitions:",
        (
            "primary_gate="
            f"positive_depth & mean_reprojection<={primary['maximum_mean_reprojection_px']}px "
            f"& max_ray_angle>={primary['minimum_triangulation_angle_degrees']}deg "
            f"& condition<={primary['maximum_condition_number']} "
            f"& views>={primary['minimum_views']}"
        ),
        (
            "primary_gate_views_ge_3="
            f"positive_depth & mean_reprojection<={strict['maximum_mean_reprojection_px']}px "
            f"& max_ray_angle>={strict['minimum_triangulation_angle_degrees']}deg "
            f"& condition<={strict['maximum_condition_number']} "
            f"& views>={strict['minimum_views']}"
        ),
        "",
        "gate,anchor_count,retained_ratio,tri_rmse,raw_rmse,tri_gain_vs_raw_percent,tri_median,raw_median,tri_p90,raw_p90,improved_anchor_ratio,severe_outlier_count,severe_outlier_rate,gate_pass",
    ]
    gate_fields = (
        "gate",
        "anchor_count",
        "retained_ratio",
        "tri_rmse",
        "raw_rmse",
        "tri_gain_vs_raw_percent",
        "tri_median",
        "raw_median",
        "tri_p90",
        "raw_p90",
        "improved_anchor_ratio",
        "severe_outlier_count",
        "severe_outlier_rate",
        "gate_pass",
    )
    for row in summary["gates"]:
        lines.append(",".join(str(row[field]) for field in gate_fields))
    lines.extend(
        (
            "",
            "metric,bin,anchor_count,tri_rmse,raw_rmse,tri_median,raw_median,tri_p90,raw_p90,improved_anchor_ratio,median_delta_raw_minus_tri,severe_outlier_count,severe_outlier_rate",
        )
    )
    bin_fields = (
        "metric",
        "bin",
        "anchor_count",
        "tri_rmse",
        "raw_rmse",
        "tri_median",
        "raw_median",
        "tri_p90",
        "raw_p90",
        "improved_anchor_ratio",
        "median_delta_raw_minus_tri",
        "severe_outlier_count",
        "severe_outlier_rate",
    )
    for row in summary["bins"]:
        lines.append(",".join(str(row[field]) for field in bin_fields))
    lines.extend(
        (
            "",
            "indicator,indicator_anchor_count,indicator_prevalence,severe_outlier_count,severe_outlier_rate_within_indicator,share_of_all_severe_outliers",
        )
    )
    concentration_fields = (
        "indicator",
        "indicator_anchor_count",
        "indicator_prevalence",
        "severe_outlier_count",
        "severe_outlier_rate_within_indicator",
        "share_of_all_severe_outliers",
    )
    for row in summary["outlier_concentration"]:
        lines.append(",".join(str(row[field]) for field in concentration_fields))
    lines.extend(
        (
            "",
            "decision=" + json.dumps(summary["decision"], sort_keys=True),
            f"claim={summary['claim']}",
            "outputs:",
            f"summary={path.with_name('summary.json')}",
            f"gate_csv={path.with_name('gate_summary.csv')}",
            f"bin_csv={path.with_name('bin_summary.csv')}",
            f"outlier_csv={path.with_name('outlier_concentration.csv')}",
            f"anchor_csv={path.with_name('anchor_reliability_scores.csv')}",
            f"frozen_gate_artifact={summary['frozen_gate_artifact']}",
            "===== COPYABLE_T01_TRIANGULATION_RELIABILITY_END =====",
        )
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def _load_run(path: str | Path) -> T01Run:
    source = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf8")) or {}
    gate = raw.get("gate", {})
    evaluation = raw.get("evaluation", {})
    primary = ReliabilityGate(
        name="primary_gate",
        maximum_mean_reprojection_px=float(
            gate.get("maximum_mean_reprojection_px", 2.0)
        ),
        minimum_triangulation_angle_degrees=float(
            gate.get("minimum_triangulation_angle_degrees", 2.0)
        ),
        maximum_condition_number=float(
            gate.get("maximum_condition_number", 1000.0)
        ),
        minimum_views=int(gate.get("minimum_views", 2)),
    )
    strict = ReliabilityGate(
        name="primary_gate_views_ge_3",
        maximum_mean_reprojection_px=primary.maximum_mean_reprojection_px,
        minimum_triangulation_angle_degrees=(
            primary.minimum_triangulation_angle_degrees
        ),
        maximum_condition_number=primary.maximum_condition_number,
        minimum_views=int(gate.get("strict_minimum_views", 3)),
    )
    run = T01Run(
        source_path=source,
        t0_output_dir=_resolve(
            raw.get("t0_output_dir", "outputs/streaming_couping_t0_anchor_probe")
        ),
        output_dir=_resolve(
            raw.get(
                "output_dir",
                "outputs/streaming_couping_t01_triangulation_reliability",
            )
        ),
        branch=str(raw.get("branch", "correct_persistent_id")),
        primary_gate=primary,
        strict_gate=strict,
        minimum_anchor_count=int(evaluation.get("minimum_anchor_count", 30)),
        minimum_retained_ratio=float(
            evaluation.get("minimum_retained_ratio", 0.10)
        ),
    )
    _validate_run(run)
    return run


def _validate_run(run: T01Run) -> None:
    if run.branch != "correct_persistent_id":
        raise ValueError("T0.1 is fixed to the correct_persistent_id branch.")
    run.primary_gate.validate()
    run.strict_gate.validate()
    if run.primary_gate.minimum_views != 2 or run.strict_gate.minimum_views != 3:
        raise ValueError("T0.1 only permits the predeclared 2-view/3-view gates.")
    if run.output_dir == run.t0_output_dir:
        raise ValueError("T0.1 cannot overwrite the T0 output directory.")
    if run.minimum_anchor_count < 1:
        raise ValueError("Minimum anchor count must be positive.")
    if not 0.0 < run.minimum_retained_ratio <= 1.0:
        raise ValueError("Minimum retained ratio must be in (0,1].")


def _gate_signature(run: T01Run, artifact_path: Path) -> str:
    source_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    payload = {
        "revision": REVISION,
        "source_artifact_sha256": source_hash,
        "source_branch": run.branch,
        "primary_gate": asdict(run.primary_gate),
        "strict_gate": asdict(run.strict_gate),
        "minimum_anchor_count": run.minimum_anchor_count,
        "minimum_retained_ratio": run.minimum_retained_ratio,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf8"
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf8")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="streaming_couping/configs/t01_triangulation_reliability.yaml",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
