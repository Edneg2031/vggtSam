#!/usr/bin/env python3
"""Aggregate repeated V7.1 seed CSVs into mean/std stability metrics."""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


METRICS = (
    "development_score",
    "development_loss",
    "development_gain_vs_frozen_l0_percent",
    "validation_loss",
    "validation_gain_vs_frozen_l0_percent",
    "future_loss",
    "future_gain_vs_frozen_l0_percent",
    "cross_loss",
    "cross_gain_vs_frozen_l0_percent",
    "future_wrong_geometry_damage_percent",
    "future_shuffle_time_damage_percent",
)


def aggregate(inputs: list[str | Path], output: str | Path) -> Path:
    paths = [Path(value).expanduser().resolve() for value in inputs]
    if len(paths) < 2:
        raise ValueError("V7.1 seed aggregation requires at least two CSVs.")
    runs = []
    for path in paths:
        with path.open("r", encoding="utf8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"Empty V7.1 seed CSV: {path}")
        current = {row["architecture"]: row for row in rows}
        if len(current) != len(rows):
            raise ValueError(f"Duplicate architecture in {path}")
        runs.append((path, current))
    expected = set(runs[0][1])
    for path, current in runs[1:]:
        if set(current) != expected:
            raise ValueError(
                f"Architecture mismatch in {path}: "
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
            "evidence": first["evidence"],
            "instance_content": first["instance_content"],
            "runs": len(runs),
            "source_csvs": " ".join(str(path) for path, _ in runs),
            "development_best_count": sum(
                int(current[architecture]["development_best"])
                for _, current in runs
            ),
            "causal_instance_pass_count": sum(
                int(current[architecture]["causal_instance_pass"])
                for _, current in runs
            ),
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
    with destination.open("w", encoding="utf8", newline="") as handle:
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
    path = aggregate(args.inputs, args.output)
    print(f"V7.1 seed aggregate={path}")
