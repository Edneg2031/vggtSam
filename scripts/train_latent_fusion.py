#!/usr/bin/env python3
"""Train latent SAM3/StreamVGGT fusion on processed ScanNet++."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from vggtsam.training.latent_fusion import (
    LatentFusionTrainConfig,
    train_latent_fusion,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/latent_fusion_train.yaml"),
    )
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--visualize-every", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.iterations is not None:
        config["training"]["iterations"] = args.iterations
    if args.device is not None:
        config["training"]["device"] = args.device
    if args.output_dir is not None:
        config["training"]["output_dir"] = str(args.output_dir)
    if args.visualize_every is not None:
        config["training"]["visualize_every"] = args.visualize_every
    if args.prompt is not None:
        config["sam3"]["prompt"] = args.prompt
        config["sam3"]["prompt_mode"] = "fixed"
        if args.prompt.strip().lower() != "object":
            config["objects"]["target_object_labels"] = [args.prompt]

    train_latent_fusion(build_train_config(config))


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf8") as handle:
        return yaml.safe_load(handle) or {}


def build_train_config(raw: dict) -> LatentFusionTrainConfig:
    dataset = raw["dataset"]
    objects = raw["objects"]
    sam3 = raw["sam3"]
    geometry = raw["geometry"]
    model = raw["model"]
    loss = raw["loss"]
    training = raw["training"]
    return LatentFusionTrainConfig(
        manifest=Path(dataset["manifest"]),
        scene_id=dataset.get("scene_id"),
        sequence_length=int(dataset["sequence_length"]),
        frame_stride=int(dataset.get("frame_stride", 1)),
        min_pixels=int(objects["min_pixels"]),
        max_area_ratio=float(objects["max_area_ratio"]),
        min_visible_frames=int(objects["min_visible_frames"]),
        max_objects_per_frame=int(objects["max_objects_per_frame"]),
        ignore_instance_id=int(objects["ignore_instance_id"]),
        semantic_ignore_label=int(objects["semantic_ignore_label"]),
        excluded_semantic_labels=[
            int(label) for label in objects.get("excluded_semantic_labels", [])
        ],
        target_object_labels=[
            str(label) for label in objects.get("target_object_labels", [])
        ],
        excluded_object_labels=[
            str(label) for label in objects.get("excluded_object_labels", [])
        ],
        min_token_majority=float(objects["min_token_majority"]),
        min_tokens_per_instance=int(objects["min_tokens_per_instance"]),
        max_match_tokens=int(objects["max_match_tokens"]),
        sam3_repo=Path(sam3["repo"]),
        sam3_checkpoint=Path(sam3["checkpoint"]),
        sam3_prompt=str(sam3["prompt"]),
        sam3_prompt_mode=str(sam3.get("prompt_mode", "random_instance")),
        sam3_resolution=int(sam3["resolution"]),
        sam3_feature_source=str(sam3["feature_source"]),
        sam3_text_conditioning=str(sam3["text_conditioning"]),
        sam3_enable_inst_interactivity=bool(
            sam3.get("enable_inst_interactivity", False)
        ),
        streamvggt_repo=Path(geometry["repo"]),
        streamvggt_checkpoint=Path(geometry["checkpoint"]),
        token_grid=tuple(int(v) for v in geometry["token_grid"]),
        context_grid=tuple(int(v) for v in geometry["context_grid"]),
        streamvggt_layer_index=int(geometry["layer_index"]),
        streamvggt_image_mode=str(geometry["image_mode"]),
        point_target_source=str(geometry.get("point_target_source", "gt")),
        use_camera_tokens=bool(geometry.get("use_camera_tokens", False)),
        d_fuse=int(model["d_fuse"]),
        num_heads=int(model["num_heads"]),
        num_classes=int(model["num_classes"]),
        dropout=float(model.get("dropout", 0.0)),
        semantic_weight=float(loss["semantic_weight"]),
        point_weight=float(loss["point_weight"]),
        match_weight=float(loss["match_weight"]),
        mask_weight=float(loss.get("mask_weight", 1.0)),
        mask_dice_weight=float(loss.get("mask_dice_weight", 1.0)),
        temperature=float(loss["temperature"]),
        device=training["device"],
        iterations=int(training["iterations"]),
        lr=float(training["lr"]),
        seed=int(training["seed"]),
        log_every=int(training["log_every"]),
        save_every=int(training["save_every"]),
        visualize_every=int(training.get("visualize_every", 0)),
        visualize_threshold=float(training.get("visualize_threshold", 0.5)),
        output_dir=Path(training["output_dir"]),
    )


if __name__ == "__main__":
    main()
