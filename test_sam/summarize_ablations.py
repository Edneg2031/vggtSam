#!/usr/bin/env python3
"""Collect final and best metrics from test_sam ablation runs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIELDS = [
    "experiment",
    "final_step",
    "final_loss",
    "final_cross_view_iou",
    "final_cross_view_recall",
    "final_absent_fp_ratio",
    "best_cross_view_iou",
    "best_cross_view_iou_step",
    "best_cross_view_recall",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summaries = []
    for history_path in sorted(args.root.glob("*/training_history.csv")):
        with history_path.open("r", encoding="utf8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            continue
        final = rows[-1]
        best_iou = max(rows, key=lambda row: float(row["cross_view_iou"]))
        summaries.append(
            {
                "experiment": history_path.parent.name,
                "final_step": final["step"],
                "final_loss": final["loss"],
                "final_cross_view_iou": final["cross_view_iou"],
                "final_cross_view_recall": final["cross_view_recall"],
                "final_absent_fp_ratio": final["absent_fp_ratio"],
                "best_cross_view_iou": best_iou["cross_view_iou"],
                "best_cross_view_iou_step": best_iou["step"],
                "best_cross_view_recall": max(
                    float(row["cross_view_recall"]) for row in rows
                ),
            }
        )

    if not summaries:
        raise RuntimeError(f"No training_history.csv files found under {args.root}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(summaries)

    print(f"ablation summary: {args.output}")
    for row in summaries:
        print(
            f"{row['experiment']}: "
            f"final_iou={float(row['final_cross_view_iou']):.4f} "
            f"best_iou={float(row['best_cross_view_iou']):.4f} "
            f"recall={float(row['final_cross_view_recall']):.4f}"
        )


if __name__ == "__main__":
    main()

