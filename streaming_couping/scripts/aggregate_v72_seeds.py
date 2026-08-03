#!/usr/bin/env python3
"""Aggregate V7.2 local-token sweeps across random seeds."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


METRICS = (
    "development_score",
    "development_loss",
    "development_gain_vs_l0_percent",
    "validation_loss",
    "validation_gain_vs_l0_percent",
    "future_loss",
    "future_gain_vs_l0_percent",
    "cross_loss",
    "cross_gain_vs_l0_percent",
    "future_local_off_loss",
    "future_wrong_local_identity_loss",
    "future_shuffle_local_time_loss",
    "cross_local_off_loss",
    "cross_wrong_local_identity_loss",
    "cross_shuffle_local_time_loss",
)


def aggregate(inputs: list[str | Path], output: str | Path) -> Path:
    paths = [Path(value).expanduser().resolve() for value in inputs]
    if len(paths) < 2:
        raise ValueError("V7.2 seed aggregation requires at least two CSVs.")
    runs = []
    for path in paths:
        with path.open(newline="", encoding="utf8") as handle:
            rows = list(csv.DictReader(handle))
        current = {row["architecture"]: row for row in rows}
        if not rows or len(current) != len(rows):
            raise ValueError(f"Empty/duplicate V7.2 architecture CSV: {path}")
        runs.append((path, current))
    expected = set(runs[0][1])
    for path, current in runs[1:]:
        if set(current) != expected:
            raise ValueError(
                f"V7.2 architecture mismatch in {path}: "
                f"{sorted(set(current) ^ expected)}"
            )

    rows = []
    for architecture in sorted(
        expected,
        key=lambda value: float(runs[0][1][value]["development_score"]),
    ):
        first = runs[0][1][architecture]
        row: dict[str, object] = {
            "architecture": architecture,
            "token_count": first.get("token_count", ""),
            "mechanism": first.get("mechanism", ""),
            "instance_content": first.get("instance_content", ""),
            "runs": len(runs),
            "development_best_count": sum(
                int(current[architecture]["development_best"])
                for _, current in runs
            ),
            "causal_local_pass_count": sum(
                int(current[architecture]["causal_local_pass"])
                for _, current in runs
            ),
            "camera_control_win_count": sum(
                int(current[architecture]["beats_camera_controls_report_only"])
                for _, current in runs
            ),
            "source_csvs": " ".join(str(path) for path, _ in runs),
        }
        for metric in METRICS:
            values = [
                _optional_float(current[architecture].get(metric, ""))
                for _, current in runs
            ]
            finite = [value for value in values if value is not None]
            row[f"{metric}_mean"] = (
                _short(statistics.mean(finite)) if finite else ""
            )
            row[f"{metric}_std"] = (
                _short(statistics.pstdev(finite)) if len(finite) > 1 else ""
            )
        rows.append(row)
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return destination


def _optional_float(value: object) -> float | None:
    text = str(value).strip()
    return None if not text else float(text)


def _short(value: float) -> str:
    return f"{float(value):.8g}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    print(f"V7.2 seed aggregate={aggregate(args.inputs, args.output)}")

