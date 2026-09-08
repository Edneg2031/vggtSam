#!/usr/bin/env python3
"""Compare only the object instances that received accepted V3 alignment."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Any


METRICS = (
    "object_accuracy_m",
    "object_completeness_m",
    "fscore_5cm",
    "voxel_iou",
    "ghost_point_ratio",
)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: print_instance_point_alignment_selected_metrics.py "
            "RUNTIME_OUTPUT_DIR EVALUATION_SUMMARY"
        )
    runtime_root = Path(sys.argv[1]).expanduser().resolve()
    evaluation_path = Path(sys.argv[2]).expanduser().resolve()
    comparison = _read_json(runtime_root / "comparison.json")
    evaluation = _read_json(evaluation_path)

    accepted_instance_ids = _accepted_instance_ids(comparison)
    map_objects_path = evaluation_path.parent / "map_objects.csv"
    rows = _read_csv(map_objects_path)
    by_branch: dict[str, dict[int, dict[str, str]]] = {"raw": {}, "instance_point_alignment": {}}
    for row in rows:
        branch = str(row.get("variant", row.get("clip", "")))
        if branch not in by_branch:
            continue
        try:
            gt_id = int(row["gt_instance_id"])
        except (KeyError, TypeError, ValueError):
            continue
        by_branch[branch][gt_id] = row

    selected_gt_ids = sorted(
        gt_id
        for gt_id, aligned in by_branch["instance_point_alignment"].items()
        if _as_int(aligned.get("predicted_instance_id"), -1)
        in accepted_instance_ids
        and _as_int(aligned.get("matched"), 0) == 1
        and gt_id in by_branch["raw"]
        and _as_int(by_branch["raw"][gt_id].get("matched"), 0) == 1
    )

    object_rows: list[dict[str, Any]] = []
    for gt_id in selected_gt_ids:
        raw = by_branch["raw"][gt_id]
        aligned = by_branch["instance_point_alignment"][gt_id]
        result: dict[str, Any] = {
            "gt_instance_id": gt_id,
            "gt_label": aligned.get("gt_label", raw.get("gt_label", "")),
            "predicted_instance_id": _as_int(
                aligned.get("predicted_instance_id"), -1
            ),
        }
        for metric in METRICS:
            raw_value = _as_float(raw.get(metric))
            aligned_value = _as_float(aligned.get(metric))
            result[f"raw_{metric}"] = raw_value
            result[f"aligned_{metric}"] = aligned_value
            result[f"delta_{metric}"] = (
                aligned_value - raw_value
                if raw_value is not None and aligned_value is not None
                else None
            )
        object_rows.append(result)

    raw_summary = _aggregate(object_rows, "raw")
    aligned_summary = _aggregate(object_rows, "aligned")
    delta_summary = {
        metric: _delta(aligned_summary.get(metric), raw_summary.get(metric))
        for metric in METRICS
    }

    print("===== SELECTED ALIGNED OBJECT METRICS =====")
    print(f"accepted_alignment_instance_ids={accepted_instance_ids}")
    print(f"selected_matched_objects={len(selected_gt_ids)}")
    for row in object_rows:
        print(
            f"object={row['gt_label']}_{row['gt_instance_id']} "
            f"predicted_instance_id={row['predicted_instance_id']} "
            + " ".join(
                f"{metric}_raw={_format(row[f'raw_{metric}'])} "
                f"{metric}_aligned={_format(row[f'aligned_{metric}'])} "
                f"delta={_format(row[f'delta_{metric}'])}"
                for metric in METRICS
            )
        )
    print("selected_raw=" + _format_summary(raw_summary))
    print("selected_aligned=" + _format_summary(aligned_summary))
    print("selected_delta_aligned_minus_raw=" + _format_summary(delta_summary))
    print(
        "selected_direction="
        + _direction_summary(delta_summary)
    )

    output_csv = evaluation_path.parent / "selected_aligned_object_metrics.csv"
    _write_csv(output_csv, object_rows)
    output_txt = evaluation_path.parent / "selected_aligned_object_metrics.txt"
    output_txt.write_text(
        "\n".join(
            [
                "===== SELECTED ALIGNED OBJECT METRICS =====",
                f"accepted_alignment_instance_ids={accepted_instance_ids}",
                f"selected_matched_objects={len(selected_gt_ids)}",
                "selected_raw=" + _format_summary(raw_summary),
                "selected_aligned=" + _format_summary(aligned_summary),
                "selected_delta_aligned_minus_raw=" + _format_summary(delta_summary),
                "selected_direction=" + _direction_summary(delta_summary),
            ]
        )
        + "\n",
        encoding="utf8",
    )
    print(f"selected_metrics_csv={output_csv}")
    print(f"selected_metrics_txt={output_txt}")


def _accepted_instance_ids(comparison: dict[str, Any]) -> list[int]:
    branch = comparison.get("branches", {}).get("instance_point_alignment", {})
    alignment = branch.get("instance_point_alignment", {}) if isinstance(branch, dict) else {}
    events = alignment.get("events", ()) if isinstance(alignment, dict) else ()
    slots = {
        _as_int(event.get("instance_id"), -1)
        for event in events
        if isinstance(event, dict) and _as_bool(event.get("accepted"))
    }
    return sorted(value for value in slots if value >= 0)


def _aggregate(rows: list[dict[str, Any]], prefix: str) -> dict[str, float | None]:
    output: dict[str, float | None] = {}
    for metric in METRICS:
        values = [
            row[f"{prefix}_{metric}"]
            for row in rows
            if row.get(f"{prefix}_{metric}") is not None
        ]
        output[metric] = sum(values) / len(values) if values else None
    return output


def _direction_summary(delta: dict[str, float | None]) -> str:
    lower_is_better = {
        "object_accuracy_m": "lower",
        "object_completeness_m": "lower",
        "ghost_point_ratio": "lower",
    }
    higher_is_better = {"fscore_5cm", "voxel_iou"}
    improved = []
    worsened = []
    unchanged = []
    for metric in METRICS:
        value = delta.get(metric)
        if value is None or abs(float(value)) < 1e-12:
            unchanged.append(metric)
        elif metric in lower_is_better and float(value) < 0.0:
            improved.append(metric)
        elif metric in higher_is_better and float(value) > 0.0:
            improved.append(metric)
        else:
            worsened.append(metric)
    return (
        f"improved={improved} worsened={worsened} unchanged={unchanged}"
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


def _as_int(value: object, default: int) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return int(default)


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off", "", "none", "null"}:
        return False
    return False


def _as_float(value: object) -> float | None:
    try:
        parsed = float(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed and abs(parsed) != float("inf") else None


def _delta(left: float | None, right: float | None) -> float | None:
    return None if left is None or right is None else float(left) - float(right)


def _format(value: object) -> str:
    if value is None:
        return "None"
    return f"{float(value):.9g}"


def _format_summary(summary: dict[str, float | None]) -> str:
    return " ".join(f"{metric}={_format(summary.get(metric))}" for metric in METRICS)


if __name__ == "__main__":
    main()
