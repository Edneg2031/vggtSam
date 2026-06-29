#!/usr/bin/env python3
"""Train dense SAM3/StreamVGGT fusion on processed ScanNet++."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from vggtsam.training.dense_fusion import (
    DenseFusionTrainConfig,
    train_dense_fusion,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/dense_fusion_train.yaml"),
    )
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--scene-id", default=None)
    parser.add_argument("--output-size", type=int, nargs=2, metavar=("H", "W"))
    parser.add_argument("--visualize-every", type=int, default=None)
    parser.add_argument("--target-mode", choices=["class", "instance"], default=None)
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--no-overfit", action="store_true")
    parser.add_argument("--window-index", type=int, default=None)
    parser.add_argument("--instance-id", type=int, default=None)
    args = parser.parse_args()

    raw = load_config(args.config)
    if args.iterations is not None:
        raw["training"]["iterations"] = args.iterations
    if args.device is not None:
        raw["training"]["device"] = args.device
    if args.output_dir is not None:
        raw["training"]["output_dir"] = str(args.output_dir)
    if args.visualize_every is not None:
        raw["training"]["visualize_every"] = args.visualize_every
    if args.scene_id is not None:
        raw["dataset"]["scene_id"] = args.scene_id
    if args.output_size is not None:
        raw["model"]["output_size"] = list(args.output_size)
    if args.target_mode is not None:
        raw["objects"]["target_mode"] = args.target_mode
    if args.overfit:
        raw["training"]["overfit"] = True
    if args.no_overfit:
        raw["training"]["overfit"] = False
    if args.window_index is not None:
        raw["training"]["overfit_window_index"] = args.window_index
    if args.instance_id is not None:
        raw["training"]["overfit_instance_id"] = args.instance_id
    if args.prompt is not None:
        raw["sam3"]["prompt"] = args.prompt
        raw["sam3"]["prompt_mode"] = "fixed"
        if args.prompt.strip().lower() != "object":
            raw["objects"]["target_object_labels"] = [args.prompt]

    train_dense_fusion(build_train_config(raw))


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf8") as handle:
        return yaml.safe_load(handle) or {}


def build_train_config(raw: dict) -> DenseFusionTrainConfig:
    dataset = raw["dataset"]
    objects = raw["objects"]
    sam3 = raw["sam3"]
    geometry = raw["geometry"]
    model = raw["model"]
    loss = raw["loss"]
    training = raw["training"]
    visualization = raw.get("visualization", {})
    return DenseFusionTrainConfig(
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
        target_mode=str(objects.get("target_mode", "class")),
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
        feature_grid=tuple(int(v) for v in geometry["feature_grid"]),
        context_grid=tuple(int(v) for v in geometry["context_grid"]),
        streamvggt_layer_index=int(geometry["layer_index"]),
        streamvggt_image_mode=str(geometry["image_mode"]),
        use_camera_tokens=bool(geometry.get("use_camera_tokens", False)),
        output_size=tuple(int(v) for v in model["output_size"]),
        d_fuse=int(model["d_fuse"]),
        num_heads=int(model["num_heads"]),
        embedding_dim=int(model["embedding_dim"]),
        num_classes=int(model["num_classes"]),
        dropout=float(model.get("dropout", 0.0)),
        mask_weight=float(loss["mask_weight"]),
        dice_weight=float(loss["dice_weight"]),
        point_weight=float(loss["point_weight"]),
        text_weight=float(loss["text_weight"]),
        aux_cls_weight=float(loss.get("aux_cls_weight", 0.0)),
        match_weight=float(loss["match_weight"]),
        temperature=float(loss["temperature"]),
        max_match_pixels=int(loss["max_match_pixels"]),
        negative_ratio=int(loss.get("negative_ratio", 8)),
        device=training["device"],
        iterations=int(training["iterations"]),
        lr=float(training["lr"]),
        seed=int(training["seed"]),
        log_every=int(training["log_every"]),
        save_every=int(training["save_every"]),
        visualize_every=int(training.get("visualize_every", 0)),
        visualize_threshold=float(training.get("visualize_threshold", 0.5)),
        overfit=bool(training.get("overfit", False)),
        overfit_window_index=int(training.get("overfit_window_index", 0)),
        overfit_instance_id=optional_int(training.get("overfit_instance_id")),
        max_visual_points=int(visualization.get("max_visual_points", 100_000)),
        output_dir=Path(training["output_dir"]),
    )


def optional_int(value) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


if __name__ == "__main__":
    main()
