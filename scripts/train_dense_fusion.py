#!/usr/bin/env python3
"""Train dense SAM3/StreamVGGT fusion on processed ScanNet++."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from vggtsam.training.dense_fusion import (
    DenseFusionTrainConfig,
    export_streamvggt_baseline,
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
    parser.add_argument("--geometry-device", default=None)
    geometry_streaming = parser.add_mutually_exclusive_group()
    geometry_streaming.add_argument("--geometry-streaming-cache", action="store_true")
    geometry_streaming.add_argument("--no-geometry-streaming-cache", action="store_true")
    parser.add_argument("--no-history", action="store_true")
    parser.add_argument(
        "--history-update-source",
        choices=["gt", "pred", "gt_or_pred", "fused_sam", "sam3_direct"],
        default=None,
    )
    parser.add_argument(
        "--fused-sam-prompt-source",
        choices=["none", "pred", "gt", "gt_or_pred", "sam3_direct"],
        default=None,
    )
    parser.add_argument("--fused-sam-mask-weight", type=float, default=None)
    parser.add_argument("--fused-sam-dice-weight", type=float, default=None)
    parser.add_argument(
        "--fused-sam-feature-mode",
        choices=["replace", "residual"],
        default=None,
        help=(
            "replace feeds adapter-generated SAM features to the decoder; residual "
            "adds adapter output to original SAM3 tracker features."
        ),
    )
    parser.add_argument("--sam3-direct-device", default=None)
    sam3_compare_direct = parser.add_mutually_exclusive_group()
    sam3_compare_direct.add_argument("--compare-sam3-direct", action="store_true")
    sam3_compare_direct.add_argument("--no-compare-sam3-direct", action="store_true")
    sam3_full_flow = parser.add_mutually_exclusive_group()
    sam3_full_flow.add_argument("--sam3-full-flow", action="store_true")
    sam3_full_flow.add_argument("--no-sam3-full-flow", action="store_true")
    parser.add_argument("--sam3-full-residual-scale", type=float, default=None)
    sam3_direct_box = parser.add_mutually_exclusive_group()
    sam3_direct_box.add_argument("--sam3-direct-box", action="store_true")
    sam3_direct_box.add_argument("--no-sam3-direct-box", action="store_true")
    parser.add_argument("--sam3-direct-threshold", type=float, default=None)
    sam3_direct_async = parser.add_mutually_exclusive_group()
    sam3_direct_async.add_argument(
        "--sam3-direct-async-loading-frames",
        action="store_true",
    )
    sam3_direct_async.add_argument(
        "--no-sam3-direct-async-loading-frames",
        action="store_true",
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
        "--point-mask-condition",
        choices=["none", "gt_soft"],
        default=None,
    )
    parser.add_argument(
        "--fusion-type",
        choices=["simple_cross_attn", "camera_guided"],
        default=None,
    )
    parser.add_argument(
        "--primary-mask-source",
        choices=["dense", "fused_sam", "sam3_direct", "sam3_full"],
        default=None,
        help=(
            "dense uses the lightweight mask_head; fused_sam routes the main "
            "prediction through SAM3 prompt/mask decoder; sam3_direct uses the "
            "original SAM3 video tracker mask provider; sam3_full uses the "
            "SAM3 full-flow tracker after injecting fused residuals."
        ),
    )
    camera_tokens = parser.add_mutually_exclusive_group()
    camera_tokens.add_argument("--use-camera-tokens", action="store_true")
    camera_tokens.add_argument("--no-use-camera-tokens", action="store_true")
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
        choices=["gt", "pred", "sam3_direct", "sam3_full"],
        default=None,
    )
    parser.add_argument("--point-valid-threshold", type=float, default=None)
    parser.add_argument("--chamfer-weight", type=float, default=None)
    parser.add_argument("--reprojection-weight", type=float, default=None)
    parser.add_argument("--text-weight", type=float, default=None)
    parser.add_argument("--aux-cls-weight", type=float, default=None)
    parser.add_argument("--match-weight", type=float, default=None)
    parser.add_argument(
        "--train-scope",
        choices=["all", "sam_adapter"],
        default=None,
        help=(
            "all trains the dense fusion baseline; sam_adapter freezes the model "
            "and trains only SAM/fusion adapter parameters."
        ),
    )
    parser.add_argument("--target-mode", choices=["class", "instance"], default=None)
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--no-overfit", action="store_true")
    parser.add_argument("--window-index", type=int, default=None)
    parser.add_argument("--instance-id", type=int, default=None)
    parser.add_argument(
        "--export-streamvggt-baseline",
        action="store_true",
        help="Run StreamVGGT once and export its pointmap with the selected GT mask.",
    )
    args = parser.parse_args()

    raw = load_config(args.config)
    if args.iterations is not None:
        raw["training"]["iterations"] = args.iterations
    if args.device is not None:
        raw["training"]["device"] = args.device
    if args.sam3_device is not None:
        raw["sam3"]["device"] = args.sam3_device
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
    if args.fused_sam_prompt_source is not None:
        raw.setdefault("history", {})[
            "fused_sam_prompt_source"
        ] = args.fused_sam_prompt_source
    if args.fused_sam_mask_weight is not None:
        raw.setdefault("fused_sam", {})["mask_weight"] = args.fused_sam_mask_weight
    if args.fused_sam_dice_weight is not None:
        raw.setdefault("fused_sam", {})["dice_weight"] = args.fused_sam_dice_weight
    if args.fused_sam_feature_mode is not None:
        raw.setdefault("fused_sam", {})["feature_mode"] = args.fused_sam_feature_mode
    if args.sam3_direct_device is not None:
        raw["sam3"]["direct_device"] = args.sam3_direct_device
    if args.compare_sam3_direct:
        raw["sam3"]["compare_direct"] = True
    if args.no_compare_sam3_direct:
        raw["sam3"]["compare_direct"] = False
    if args.sam3_full_flow:
        raw["sam3"]["full_flow"] = True
    if args.no_sam3_full_flow:
        raw["sam3"]["full_flow"] = False
    if args.sam3_full_residual_scale is not None:
        raw["sam3"]["full_residual_scale"] = args.sam3_full_residual_scale
    if args.sam3_direct_box:
        raw["sam3"]["direct_prompt_with_box"] = True
    if args.no_sam3_direct_box:
        raw["sam3"]["direct_prompt_with_box"] = False
    if args.sam3_direct_threshold is not None:
        raw["sam3"]["direct_output_prob_thresh"] = args.sam3_direct_threshold
    if args.sam3_direct_async_loading_frames:
        raw["sam3"]["direct_async_loading_frames"] = True
    if args.no_sam3_direct_async_loading_frames:
        raw["sam3"]["direct_async_loading_frames"] = False
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
    if args.train_scope is not None:
        raw["training"]["train_scope"] = args.train_scope
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
    if args.point_mask_condition is not None:
        raw["model"]["point_mask_condition"] = args.point_mask_condition
    if args.fusion_type is not None:
        raw["model"]["fusion_type"] = args.fusion_type
    if args.primary_mask_source is not None:
        raw["model"]["primary_mask_source"] = args.primary_mask_source
    if args.use_camera_tokens:
        raw["geometry"]["use_camera_tokens"] = True
    if args.no_use_camera_tokens:
        raw["geometry"]["use_camera_tokens"] = False
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

    config = build_train_config(raw)
    if args.export_streamvggt_baseline:
        export_streamvggt_baseline(config)
    else:
        train_dense_fusion(config)


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
    fused_sam = raw.get("fused_sam", {})
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
        sam3_direct_device=str(
            sam3.get("direct_device") or sam3.get("device") or training["device"]
        ),
        sam3_direct_prompt_with_box=bool(
            sam3.get("direct_prompt_with_box", True)
        ),
        sam3_direct_output_prob_thresh=float(
            sam3.get("direct_output_prob_thresh", 0.5)
        ),
        sam3_direct_async_loading_frames=bool(
            sam3.get("direct_async_loading_frames", False)
        ),
        sam3_compare_direct=bool(sam3.get("compare_direct", False)),
        sam3_full_flow=bool(sam3.get("full_flow", False)),
        sam3_full_residual_scale=float(sam3.get("full_residual_scale", 1.0)),
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
        point_mask_condition=str(model.get("point_mask_condition", "none")),
        fusion_type=str(model.get("fusion_type", "simple_cross_attn")),
        primary_mask_source=str(model.get("primary_mask_source", "dense")),
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
        history_update_source=str(history.get("update_source", "gt")),
        history_pred_threshold=float(history.get("pred_threshold", 0.5)),
        fused_sam_prompt_source=str(history.get("fused_sam_prompt_source", "pred")),
        fused_sam_feature_mode=str(fused_sam.get("feature_mode", "replace")),
        fused_sam_mask_weight=float(fused_sam.get("mask_weight", 0.0)),
        fused_sam_dice_weight=float(fused_sam.get("dice_weight", 0.0)),
        device=training["device"],
        iterations=int(training["iterations"]),
        lr=float(training["lr"]),
        train_scope=str(training.get("train_scope", "all")),
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
