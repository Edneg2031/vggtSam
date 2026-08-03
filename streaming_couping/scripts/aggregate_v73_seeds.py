#!/usr/bin/env python3
"""Aggregate completed V7.3 seed directories without reselecting on test splits."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


METRICS = (
    "development_score",
    "development_loss",
    "validation_loss",
    "future_loss",
    "cross_loss",
)


def main() -> None:
    args = _parse_args()
    root = Path(args.root).expanduser().resolve()
    seeds = tuple(int(v) for v in args.seeds.split())
    rows = aggregate(root, seeds)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(output, rows)
    print(output.read_text(encoding="utf8").rstrip())


def aggregate(root: Path, seeds: tuple[int, ...]) -> list[dict[str, object]]:
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("V7.3 seeds must be nonempty and unique.")
    grouped: dict[str, list[tuple[int, dict[str, str]]]] = defaultdict(list)
    for seed in seeds:
        path = root / f"seed_{seed}" / "v73_correspondence_ablation.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing V7.3 seed result: {path}")
        with path.open("r", encoding="utf8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        methods = [
            row
            for row in rows
            if row.get("architecture_family") == "v73_correspondence"
        ]
        if not methods:
            raise ValueError(f"No V7.3 method rows in {path}")
        for row in methods:
            grouped[row["architecture"]].append((seed, row))
    expected = set(seeds)
    output = []
    for architecture, entries in sorted(grouped.items()):
        actual = {seed for seed, _ in entries}
        if actual != expected:
            raise ValueError(
                f"Architecture {architecture} seed coverage {sorted(actual)} "
                f"does not match {sorted(expected)}."
            )
        rows = [row for _, row in sorted(entries)]
        result: dict[str, object] = {
            "architecture": architecture,
            "token_count": rows[0]["token_count"],
            "uses_sam": rows[0]["uses_sam"],
            "seeds": " ".join(str(v) for v in sorted(expected)),
            "seed_count": len(rows),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in rows]
            result[f"{metric}_mean"] = _short(sum(values) / len(values))
            result[f"{metric}_std"] = _short(_population_std(values))
        result["v73_development_best_count"] = sum(
            int(row["v73_development_best"]) for row in rows
        )
        result["beats_same_k_geometry_all_splits_count"] = sum(
            int(row["beats_same_k_geometry_all_splits"]) for row in rows
        )
        result["beats_retained_v72_geometry_all_splits_count"] = sum(
            int(row["beats_retained_v72_geometry_all_splits"]) for row in rows
        )
        result["beats_camera_controls_report_only_count"] = sum(
            int(row["beats_camera_controls_report_only"]) for row in rows
        )
        result["instance_off_exact_count"] = sum(
            int(row["instance_off_exact"]) for row in rows
        )
        result["causal_sam_pass_count"] = sum(
            int(row["causal_sam_pass"]) for row in rows
        )
        hurt = sum(int(row["sam_perturbation_hurt_count"]) for row in rows)
        total = sum(int(row["sam_perturbation_total"]) for row in rows)
        result["sam_perturbation_hurt_fraction"] = _short(
            hurt / total if total else 0.0
        )
        output.append(result)
    return output


def _population_std(values: list[float]) -> float:
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _short(value: float) -> str:
    return f"{float(value):.8g}"


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("No V7.3 seed rows to aggregate.")
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", default="outputs/streaming_couping_v73_multiseed"
    )
    parser.add_argument("--seeds", default="0 1 2")
    parser.add_argument(
        "--output",
        default=(
            "outputs/streaming_couping_v73_multiseed/"
            "v73_seed_aggregate.csv"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
