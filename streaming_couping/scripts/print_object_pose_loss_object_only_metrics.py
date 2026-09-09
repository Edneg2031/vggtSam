#!/usr/bin/env python3
"""Compare raw/object-only metrics for the same exported SAM instances."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys
from typing import Any


METRICS = (
    "object_accuracy_m",
    "object_completeness_m",
    "fscore_5cm",
    "voxel_iou_5cm",
    "ghost_point_ratio",
)
LOWER_IS_BETTER = {
    "object_accuracy_m",
    "object_completeness_m",
    "ghost_point_ratio",
}


def main() -> None:
    if len(sys.argv) not in {3, 4}:
        raise SystemExit(
            "usage: print_object_pose_loss_object_only_metrics.py "
            "POSE_REFINEMENT_SUMMARY EVALUATION_SUMMARY [ALIGNED_BRANCH]"
        )
    refinement_path = Path(sys.argv[1]).expanduser().resolve()
    evaluation_path = Path(sys.argv[2]).expanduser().resolve()
    aligned_branch = (
        sys.argv[3] if len(sys.argv) == 4 else "object_pose_object_only"
    )
    refinement = _read_json(refinement_path)
    evaluation = _read_json(evaluation_path)
    rows = _read_csv(evaluation_path.parent / "map_objects.csv")

    raw = _matched_by_prediction(rows, "raw_pose")
    aligned = _matched_by_prediction(rows, aligned_branch)
    common_ids = sorted(set(raw).intersection(aligned))
    object_rows = [_compare_object(raw[instance_id], aligned[instance_id])
                   for instance_id in common_ids]
    raw_summary = _aggregate(object_rows, "raw")
    aligned_summary = _aggregate(object_rows, "aligned")
    delta_summary = {
        metric: _delta(aligned_summary.get(metric), raw_summary.get(metric))
        for metric in METRICS
    }

    lines = _render(
        refinement=refinement,
        evaluation=evaluation,
        raw_count=len(raw),
        aligned_count=len(aligned),
        aligned_branch=aligned_branch,
        object_rows=object_rows,
        raw_summary=raw_summary,
        aligned_summary=aligned_summary,
        delta_summary=delta_summary,
    )
    print("\n".join(lines))

    output_stem = (
        "object_pose_loss_object_only_metrics"
        if aligned_branch == "object_pose_object_only"
        else "object_pose_loss_object_per_instance_metrics"
    )
    output_csv = evaluation_path.parent / f"{output_stem}.csv"
    _write_csv(output_csv, object_rows)
    output_txt = evaluation_path.parent / f"{output_stem}.txt"
    output_txt.write_text("\n".join(lines) + "\n", encoding="utf8")
    print(f"metrics_csv={output_csv}")
    print(f"metrics_txt={output_txt}")


def _render(
    *,
    refinement: dict[str, Any],
    evaluation: dict[str, Any],
    raw_count: int,
    aligned_count: int,
    aligned_branch: str,
    object_rows: list[dict[str, Any]],
    raw_summary: dict[str, float | None],
    aligned_summary: dict[str, float | None],
    delta_summary: dict[str, float | None],
) -> list[str]:
    pose = evaluation.get("pose_evaluation", {})
    pose_branches = pose.get("branches", {}) if isinstance(pose, dict) else {}
    raw_pose = (
        pose_branches.get("raw_pose", {})
        if isinstance(pose_branches, dict)
        else {}
    )
    raw_pose_summary = (
        raw_pose.get("summary", {}) if isinstance(raw_pose, dict) else {}
    )
    correction_summary = refinement.get("object_point_correction", {})
    if not isinstance(correction_summary, dict):
        correction_summary = {}
    return [
        "===== OBJECT-ONLY SAME-INSTANCE METRICS =====",
        "comparison_key=predicted_instance_id from shared SAM tracking",
        "independent_instance_poses="
        + str(bool(refinement.get("independent_instance_poses", False))),
        f"raw_matched_objects={raw_count}",
        f"{aligned_branch}_matched_objects={aligned_count}",
        f"common_matched_objects={len(object_rows)}",
        f"pose_branches={sorted(pose_branches) if isinstance(pose_branches, dict) else []}",
        "pose=raw_pose "
        f"ATE_RMSE_m={raw_pose_summary.get('ate_rmse_m')} "
        f"RPE_t_RMSE_m={raw_pose_summary.get('rpe_translation_rmse_m')} "
        f"RPE_r_RMSE_deg={raw_pose_summary.get('rpe_rotation_rmse_deg')}",
        "accepted_frames=" + str(refinement.get("accepted_frame_count")),
        "object_point_correction_frames="
        + str(correction_summary.get("frame_count", 0)),
        "object_point_correction_instances="
        + str(correction_summary.get("instance_correction_count", 0)),
        "raw=" + _format_summary(raw_summary),
        f"{aligned_branch}=" + _format_summary(aligned_summary),
        "delta_aligned_minus_raw=" + _format_summary(delta_summary),
        "direction=" + _direction_summary(delta_summary),
        *[_format_object(row) for row in object_rows],
    ]


def _matched_by_prediction(
    rows: list[dict[str, str]], branch: str
) -> dict[int, dict[str, str]]:
    output: dict[int, dict[str, str]] = {}
    for row in rows:
        if str(row.get("variant", row.get("clip", ""))) != branch:
            continue
        if not _as_bool(row.get("matched")):
            continue
        instance_id = _as_int(row.get("predicted_instance_id"), -1)
        if instance_id >= 0:
            output.setdefault(instance_id, row)
    return output


def _compare_object(raw: dict[str, str], aligned: dict[str, str]) -> dict[str, Any]:
    instance_id = _as_int(raw.get("predicted_instance_id"), -1)
    result: dict[str, Any] = {
        "predicted_instance_id": instance_id,
        "object": aligned.get("gt_label") or raw.get("gt_label", "object"),
        "gt_instance_id_raw": _as_int(raw.get("gt_instance_id"), -1),
        "gt_instance_id_aligned": _as_int(aligned.get("gt_instance_id"), -1),
    }
    for metric in METRICS:
        raw_value = _as_float(_row_metric(raw, metric))
        aligned_value = _as_float(_row_metric(aligned, metric))
        result[f"raw_{metric}"] = raw_value
        result[f"aligned_{metric}"] = aligned_value
        result[f"delta_{metric}"] = _delta(aligned_value, raw_value)
    return result


def _row_metric(row: dict[str, str], metric: str) -> object:
    """Read per-object CSV names, including evaluator's voxel-IoU alias."""

    if metric in row:
        return row[metric]
    if metric == "voxel_iou_5cm":
        return row.get("voxel_iou")
    return None


def _aggregate(rows: list[dict[str, Any]], prefix: str) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for metric in METRICS:
        values = [
            row[f"{prefix}_{metric}"]
            for row in rows
            if row.get(f"{prefix}_{metric}") is not None
        ]
        result[metric] = sum(values) / len(values) if values else None
    return result


def _direction_summary(delta: dict[str, float | None]) -> str:
    improved: list[str] = []
    worsened: list[str] = []
    unchanged: list[str] = []
    for metric in METRICS:
        value = delta.get(metric)
        if value is None or abs(float(value)) < 1e-12:
            unchanged.append(metric)
        elif metric in LOWER_IS_BETTER and float(value) < 0.0:
            improved.append(metric)
        elif metric not in LOWER_IS_BETTER and float(value) > 0.0:
            improved.append(metric)
        else:
            worsened.append(metric)
    return (
        f"improved={improved} worsened={worsened} unchanged={unchanged}"
    )


def _format_object(row: dict[str, Any]) -> str:
    values = [
        f"{metric}_raw={_format(row.get(f'raw_{metric}'))} "
        f"{metric}_aligned={_format(row.get(f'aligned_{metric}'))} "
        f"delta={_format(row.get(f'delta_{metric}'))}"
        for metric in METRICS
    ]
    return (
        f"object={row['object']} predicted_instance_id="
        f"{row['predicted_instance_id']} " + " ".join(values)
    )


def _format_summary(summary: dict[str, float | None]) -> str:
    return " ".join(
        f"{metric}={_format(summary.get(metric))}" for metric in METRICS
    )


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _as_int(value: object, default: int) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return int(default)


def _as_float(value: object) -> float | None:
    try:
        parsed = float(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def _format(value: object) -> str:
    if value is None:
        return "None"
    return f"{float(value):.9g}"


if __name__ == "__main__":
    main()
