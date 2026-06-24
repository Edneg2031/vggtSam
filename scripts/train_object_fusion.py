#!/usr/bin/env python3
"""Train object-level geometry/semantic fusion on processed ScanNet++."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from vggtsam.training.object_fusion import ObjectFusionTrainConfig, train_object_fusion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/object_fusion_train.yaml"))
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.iterations is not None:
        config["training"]["iterations"] = args.iterations
    if args.device is not None:
        config["training"]["device"] = args.device
    if args.output_dir is not None:
        config["training"]["output_dir"] = str(args.output_dir)

    train_object_fusion(build_train_config(config))


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf8") as handle:
        return yaml.safe_load(handle) or {}


def build_train_config(raw: dict) -> ObjectFusionTrainConfig:
    dataset = raw["dataset"]
    objects = raw["objects"]
    geometry = raw["geometry"]
    model = raw["model"]
    loss = raw["loss"]
    training = raw["training"]
    return ObjectFusionTrainConfig(
        manifest=Path(dataset["manifest"]),
        scene_id=dataset.get("scene_id"),
        sequence_length=int(dataset["sequence_length"]),
        frame_stride=int(dataset.get("frame_stride", 1)),
        min_pixels=int(objects["min_pixels"]),
        max_area_ratio=float(objects["max_area_ratio"]),
        min_visible_frames=int(objects["min_visible_frames"]),
        max_objects_per_frame=int(objects["max_objects_per_frame"]),
        semantic_ignore_label=int(objects["semantic_ignore_label"]),
        streamvggt_repo=Path(geometry["repo"]),
        streamvggt_checkpoint=Path(geometry["checkpoint"]),
        token_grid=tuple(int(v) for v in geometry["token_grid"]),
        d_fuse=int(model["d_fuse"]),
        num_heads=int(model["num_heads"]),
        num_classes=int(model["num_classes"]),
        semantic_weight=float(loss["semantic_weight"]),
        centroid_weight=float(loss["centroid_weight"]),
        contrastive_weight=float(loss["contrastive_weight"]),
        temperature=float(loss["temperature"]),
        device=training["device"],
        iterations=int(training["iterations"]),
        lr=float(training["lr"]),
        seed=int(training["seed"]),
        log_every=int(training["log_every"]),
        save_every=int(training["save_every"]),
        output_dir=Path(training["output_dir"]),
    )


if __name__ == "__main__":
    main()
