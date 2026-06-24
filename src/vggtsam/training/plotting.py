"""Plotting helpers for training metrics."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List


def plot_training_curves(
    metrics_path: Path,
    output_path: Path,
    *,
    title: str = "Training Curves",
) -> None:
    rows = read_metric_rows(metrics_path)
    if not rows:
        print(f"No metric rows found in {metrics_path}")
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Could not plot training curves: {exc}")
        return

    steps = [int(row["step"]) for row in rows]
    fieldnames = rows[0].keys()
    loss_names = ["loss"] + [
        name for name in fieldnames if name.endswith("_loss") and name != "loss"
    ]
    count_names = [
        name
        for name in fieldnames
        if name.startswith("num_") and all(row.get(name, "").strip() for row in rows)
    ]

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for name in loss_names:
        if name not in fieldnames:
            continue
        values = [float(row[name]) for row in rows]
        axes[0].plot(steps, values, label=name)
    axes[0].set_title(title)
    axes[0].set_ylabel("loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    for name in count_names:
        values = [int(float(row[name])) for row in rows]
        axes[1].plot(steps, values, label=name)
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("counts")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def read_metric_rows(metrics_path: Path) -> List[Dict[str, str]]:
    if not metrics_path.is_file():
        return []
    with metrics_path.open("r", newline="", encoding="utf8") as handle:
        return list(csv.DictReader(handle))
