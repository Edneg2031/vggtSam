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
    parser.add_argument("--sam3-device", default=None)
    parser.add_argument("--sam3-tracker-device", default=None)
    parser.add_argument("--sam3-tracker", action="store_true")
    parser.add_argument("--no-sam3-tracker", action="store_true")
    parser.add_argument("--no-sam3-tracker-box", action="store_true")
    parser.add_argument("--sam3-tracker-threshold", type=float, default=None)
    parser.add_argument("--geometry-device", default=None)
    geometry_streaming = parser.add_mutually_exclusive_group()
    geometry_streaming.add_argument("--geometry-streaming-cache", action="store_true")
    geometry_streaming.add_argument("--no-geometry-streaming-cache", action="store_true")
    parser.add_argument("--no-history", action="store_true")
    parser.add_argument(
        "--history-update-source",
        choices=["gt", "pred", "gt_or_pred", "sam3"],
        default=None,
    )
    parser.add_argument("--sam3-frame-chunk-size", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--scene-id", default=None)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=None)
    parser.add_argument("--frame-indices", type=int, nargs="+", default=None)
    parser.add_argument("--output-size", type=int, nargs=2, metavar=("H", "W"))
    parser.add_argument(
        "--point-decoder",
        choices=["simple", "stream_dpt"],
        default=None,
    )
    parser.add_argument(
        "--point-conditioning",
        choices=["none", "object_query"],
        default=None,
    )
    stream_dpt_pretrained = parser.add_mutually_exclusive_group()
    stream_dpt_pretrained.add_argument(
        "--stream-dpt-use-pretrained",
        action="store_true",
    )
    stream_dpt_pretrained.add_argument(
        "--no-stream-dpt-use-pretrained",
        action="store_true",
    )
    stream_dpt_freeze = parser.add_mutually_exclusive_group()
    stream_dpt_freeze.add_argument("--stream-dpt-freeze", action="store_true")
    stream_dpt_freeze.add_argument("--no-stream-dpt-freeze", action="store_true")
    parser.add_argument("--visualize-every", type=int, default=None)
    parser.add_argument("--mask-weight", type=float, default=None)
    parser.add_argument("--dice-weight", type=float, default=None)
    parser.add_argument("--point-weight", type=float, default=None)
    parser.add_argument(
        "--point-valid-source",
        choices=["gt", "pred"],
        default=None,
    )
    parser.add_argument("--point-valid-threshold", type=float, default=None)
    parser.add_argument("--chamfer-weight", type=float, default=None)
    parser.add_argument("--reprojection-weight", type=float, default=None)
    parser.add_argument("--text-weight", type=float, default=None)
    parser.add_argument("--aux-cls-weight", type=float, default=None)
    parser.add_argument("--match-weight", type=float, default=None)
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
    if args.sam3_device is not None:
        raw["sam3"]["device"] = args.sam3_device
    if args.sam3_tracker_device is not None:
        raw["sam3"]["tracker_device"] = args.sam3_tracker_device
    if args.sam3_tracker:
        raw["sam3"]["tracker_enabled"] = True
    if args.no_sam3_tracker:
        raw["sam3"]["tracker_enabled"] = False
    if args.no_sam3_tracker_box:
        raw["sam3"]["tracker_prompt_with_box"] = False
    if args.sam3_tracker_threshold is not None:
        raw["sam3"]["tracker_output_prob_thresh"] = args.sam3_tracker_threshold
    if args.geometry_device is not None:
        raw["geometry"]["device"] = args.geometry_device
    if args.geometry_streaming_cache:
        raw["geometry"]["streaming_cache"] = True
    if args.no_geometry_streaming_cache:
        raw["geometry"]["streaming_cache"] = False
    if args.no_history:
        raw.setdefault("history", {})["enabled"] = False
    if args.history_update_source is not None:
        raw.setdefault("history", {})["update_source"] = args.history_update_source
    if args.sam3_frame_chunk_size is not None:
        raw["sam3"]["frame_chunk_size"] = args.sam3_frame_chunk_size
    if args.output_dir is not None:
        raw["training"]["output_dir"] = str(args.output_dir)
    if args.visualize_every is not None:
        raw["training"]["visualize_every"] = args.visualize_every
    if args.mask_weight is not None:
        raw["loss"]["mask_weight"] = args.mask_weight
    if args.dice_weight is not None:
        raw["loss"]["dice_weight"] = args.dice_weight
    if args.point_weight is not None:
        raw["loss"]["point_weight"] = args.point_weight
    if args.point_valid_source is not None:
        raw["loss"]["point_valid_source"] = args.point_valid_source
    if args.point_valid_threshold is not None:
        raw["loss"]["point_valid_threshold"] = args.point_valid_threshold
    if args.chamfer_weight is not None:
        raw["loss"]["chamfer_weight"] = args.chamfer_weight
    if args.reprojection_weight is not None:
        raw["loss"]["reprojection_weight"] = args.reprojection_weight
    if args.text_weight is not None:
        raw["loss"]["text_weight"] = args.text_weight
    if args.aux_cls_weight is not None:
        raw["loss"]["aux_cls_weight"] = args.aux_cls_weight
    if args.match_weight is not None:
        raw["loss"]["match_weight"] = args.match_weight
    if args.scene_id is not None:
        raw["dataset"]["scene_id"] = args.scene_id
    if args.sequence_length is not None:
        raw["dataset"]["sequence_length"] = args.sequence_length
    if args.frame_stride is not None:
        raw["dataset"]["frame_stride"] = args.frame_stride
    if args.frame_indices is not None:
        raw["dataset"]["frame_indices"] = list(args.frame_indices)
    if args.output_size is not None:
        raw["model"]["output_size"] = list(args.output_size)
    if args.point_decoder is not None:
        raw["model"]["point_decoder"] = args.point_decoder
    if args.point_conditioning is not None:
        raw["model"]["point_conditioning"] = args.point_conditioning
    if args.stream_dpt_use_pretrained:
        raw["model"]["stream_dpt_use_pretrained"] = True
    if args.no_stream_dpt_use_pretrained:
        raw["model"]["stream_dpt_use_pretrained"] = False
    if args.stream_dpt_freeze:
        raw["model"]["stream_dpt_freeze"] = True
    if args.no_stream_dpt_freeze:
        raw["model"]["stream_dpt_freeze"] = False
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
    history = raw.get("history", {})
    training = raw["training"]
    visualization = raw.get("visualization", {})
    return DenseFusionTrainConfig(
        manifest=Path(dataset["manifest"]),
        scene_id=dataset.get("scene_id"),
        sequence_length=int(dataset["sequence_length"]),
        frame_stride=int(dataset.get("frame_stride", 1)),
        frame_indices=optional_int_list(dataset.get("frame_indices")),
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
        sam3_device=str(sam3.get("device") or training["device"]),
        sam3_frame_chunk_size=int(sam3.get("frame_chunk_size", 0)),
        sam3_tracker_enabled=bool(sam3.get("tracker_enabled", False)),
        sam3_tracker_device=str(
            sam3.get("tracker_device") or sam3.get("device") or training["device"]
        ),
        sam3_tracker_prompt_with_box=bool(
            sam3.get("tracker_prompt_with_box", True)
        ),
        sam3_tracker_output_prob_thresh=float(
            sam3.get("tracker_output_prob_thresh", 0.5)
        ),
        sam3_tracker_async_loading_frames=bool(
            sam3.get("tracker_async_loading_frames", False)
        ),
        streamvggt_repo=Path(geometry["repo"]),
        streamvggt_checkpoint=Path(geometry["checkpoint"]),
        geometry_device=str(geometry.get("device") or training["device"]),
        geometry_streaming_cache=bool(geometry.get("streaming_cache", False)),
        feature_grid=tuple(int(v) for v in geometry["feature_grid"]),
        context_grid=tuple(int(v) for v in geometry["context_grid"]),
        streamvggt_layer_index=int(geometry["layer_index"]),
        streamvggt_dpt_layer_indices=[
            int(v) for v in geometry.get("dpt_layer_indices", [4, 11, 17, 23])
        ],
        streamvggt_image_mode=str(geometry["image_mode"]),
        use_camera_tokens=bool(geometry.get("use_camera_tokens", False)),
        output_size=tuple(int(v) for v in model["output_size"]),
        d_fuse=int(model["d_fuse"]),
        num_heads=int(model["num_heads"]),
        embedding_dim=int(model["embedding_dim"]),
        num_classes=int(model["num_classes"]),
        dropout=float(model.get("dropout", 0.0)),
        point_decoder=str(model.get("point_decoder", "simple")),
        point_conditioning=str(model.get("point_conditioning", "object_query")),
        stream_dpt_use_pretrained=bool(
            model.get("stream_dpt_use_pretrained", True)
        ),
        stream_dpt_freeze=bool(model.get("stream_dpt_freeze", False)),
        mask_weight=float(loss["mask_weight"]),
        dice_weight=float(loss["dice_weight"]),
        point_weight=float(loss["point_weight"]),
        point_valid_source=str(loss.get("point_valid_source", "gt")),
        point_valid_threshold=float(loss.get("point_valid_threshold", 0.5)),
        chamfer_weight=float(loss.get("chamfer_weight", 0.0)),
        reprojection_weight=float(loss.get("reprojection_weight", 0.0)),
        text_weight=float(loss["text_weight"]),
        aux_cls_weight=float(loss.get("aux_cls_weight", 0.0)),
        match_weight=float(loss["match_weight"]),
        temperature=float(loss["temperature"]),
        max_match_pixels=int(loss["max_match_pixels"]),
        max_chamfer_points=int(loss.get("max_chamfer_points", 1024)),
        negative_ratio=int(loss.get("negative_ratio", 8)),
        history_enabled=bool(history.get("enabled", True)),
        history_update_source=str(history.get("update_source", "sam3")),
        history_pred_threshold=float(history.get("pred_threshold", 0.5)),
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


def optional_int_list(value) -> list[int] | None:
    if value is None or value == "":
        return None
    return [int(item) for item in value]


if __name__ == "__main__":
    main()
