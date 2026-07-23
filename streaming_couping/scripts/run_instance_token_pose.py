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
from streaming_couping.src.learned_pose.export import export_final_ray_pose_outputs


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
    if args.stage == "ray":
        evaluate_all_modes(config, ray_pose_only=True)
    if args.stage == "export":
        path = export_final_ray_pose_outputs(
            config,
            variant=args.ray_variant,
            output_dir=args.export_output_dir,
        )
        print(f"exported final instance-ray pose and point clouds to {path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="streaming_couping/configs/instance_token_pose.yaml",
    )
    parser.add_argument(
        "--stage",
        choices=("all", "cache", "train", "eval", "ray", "export"),
        default="all",
    )
    parser.add_argument("--sam3-device")
    parser.add_argument("--geometry-device")
    parser.add_argument("--training-device")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument(
        "--ray-variant",
        help="Override evaluation.ray_pose.final_variant for export.",
    )
    parser.add_argument(
        "--export-output-dir",
        help="Override the final pose/PLY output directory.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
