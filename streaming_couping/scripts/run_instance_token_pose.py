#!/usr/bin/env python3
"""Cache, train, and evaluate persistent-instance StreamVGGT pose fusion."""

from __future__ import annotations

import argparse
from dataclasses import replace

from streaming_couping.src.learned_pose.cache import build_feature_caches
from streaming_couping.src.learned_pose.config import load_learned_pose_config
from streaming_couping.src.learned_pose.pipeline import (
    evaluate_all_modes,
    train_all_modes,
)


def main() -> None:
    args = _parse_args()
    config = load_learned_pose_config(args.config)
    if args.sam3_device is not None or args.geometry_device is not None:
        config = replace(
            config,
            sam3_device=args.sam3_device or config.sam3_device,
            geometry_device=args.geometry_device or config.geometry_device,
        )
    if args.training_device is not None:
        config = replace(
            config,
            training=replace(config.training, device=args.training_device),
        )
    if args.rebuild_cache:
        config = replace(
            config,
            features=replace(config.features, rebuild=True),
        )
    if args.stage in {"all", "cache"}:
        build_feature_caches(config)
    if args.stage in {"all", "train"}:
        train_all_modes(config)
    if args.stage in {"all", "eval"}:
        evaluate_all_modes(config)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="streaming_couping/configs/instance_token_pose.yaml",
    )
    parser.add_argument(
        "--stage",
        choices=("all", "cache", "train", "eval"),
        default="all",
    )
    parser.add_argument("--sam3-device")
    parser.add_argument("--geometry-device")
    parser.add_argument("--training-device")
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
