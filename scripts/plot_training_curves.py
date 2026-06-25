#!/usr/bin/env python3
"""Plot training curves from a latent-fusion training history CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

from vggtsam.training.plotting import plot_training_curves


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("outputs/latent_fusion_debug/training_history.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/latent_fusion_debug/training_curves.png"),
    )
    parser.add_argument("--title", default="Latent Fusion Training")
    args = parser.parse_args()

    plot_training_curves(args.metrics, args.output, title=args.title)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
