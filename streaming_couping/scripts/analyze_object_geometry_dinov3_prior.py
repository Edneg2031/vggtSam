#!/usr/bin/env python3
"""Diagnose whether dense DINOv3 features contain useful geometry priors.

This command is intentionally an analysis-only stop point.  It consumes the
frozen StreamVGGT/SAM cache, frozen multi-layer DINOv3 dense features, and GT
only for diagnosis.  It does not train a head, alter a pointmap, or update any
model parameter.

The main comparison is spatial rather than object-embedding based:

* raw StreamVGGT point error is measured on GT object support;
* GT masks are split into boundary, near-boundary, and interior bands;
* DINOv3, StreamVGGT, and RGB spatial gradients are compared with that error;
* thresholds are estimated from train+validation only and then frozen for all
  splits, including the sealed test scene.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.nn import functional as F

from streaming_couping.src.dinov3_object_geometry import apply_similarity
from streaming_couping.src.learned_pose.cache import load_feature_cache
from streaming_couping.src.learned_pose.config import load_learned_pose_config
from streaming_couping.scripts.run_dinov3_object_geometry import (
    _load_clip_data,
    _load_ground_truth_bundle,
)


REVISION = "dinov3_object_geometry_prior_diagnostic_r1"
DEFAULT_STORAGE_ROOT = "/data184/open_source/vggtSam"
DEFAULT_PROTOCOL = "outputs/streaming_couping_multiclip/protocol.yaml"
DEFAULT_FEATURE_DIR = "outputs/streaming_couping_dinov3_object_features"
DEFAULT_OUTPUT_DIR = "outputs/streaming_couping_object_geometry_dinov3_prior"

BANDS = ("boundary", "near_boundary", "interior", "all")
SIZE_BINS = ("small", "medium", "large")
THINNESS_BINS = ("thin", "medium", "filled")
CONFIDENCE_BUCKETS = (
    (float("-inf"), 0.30, "<0.30"),
    (0.30, 0.50, "0.30-0.50"),
    (0.50, 0.70, "0.50-0.70"),
    (0.70, 0.90, "0.70-0.90"),
    (0.90, float("inf"), "0.90+")
)


@dataclass
class ClipInputs:
    """All frozen arrays needed for one clip's diagnosis."""

    data: Any
    ground_truth: Any
    gt_masks: torch.Tensor
    images: torch.Tensor
    aligned_raw: torch.Tensor
    error: torch.Tensor
    finite: torch.Tensor
    evaluation_valid: torch.Tensor
    gradients: dict[str, torch.Tensor]
    object_bands: tuple[dict[str, torch.Tensor], ...]
    regions: dict[str, torch.Tensor]
    dino_layer_ids: tuple[int, ...]
    dino_checkpoint: str


def main() -> None:
    args = _parse_args()
    storage_root = Path(
        os.environ.get("VGGT_SAM_STORAGE_ROOT", DEFAULT_STORAGE_ROOT)
    ).expanduser().resolve()
    protocol = _resolve_path(args.protocol, storage_root, DEFAULT_PROTOCOL)
    feature_dir = _resolve_path(args.feature_dir, storage_root, DEFAULT_FEATURE_DIR)
    output_dir = _resolve_path(args.output_dir, storage_root, DEFAULT_OUTPUT_DIR)
    if not protocol.is_file():
        raise FileNotFoundError(f"Multi-scene protocol is missing: {protocol}")
    if not feature_dir.is_dir():
        raise FileNotFoundError(f"DINOv3 feature cache directory is missing: {feature_dir}")

    config = load_learned_pose_config(protocol)
    clips_by_split = {
        split: tuple(clip for clip in config.clips if clip.split == split)
        for split in ("train", "validation", "test")
    }
    if any(not clips_by_split[split] for split in clips_by_split):
        raise ValueError(
            "The prior diagnostic requires train/validation/test clips; got "
            + ", ".join(f"{key}={len(value)}" for key, value in clips_by_split.items())
        )

    print("DINOv3 OBJECT GEOMETRY PRIOR DIAGNOSTIC")
    print(f"protocol={protocol}")
    print(f"feature_dir={feature_dir}")
    print(
        "diagnostic_only=1 frozen=StreamVGGT/SAM3/DINOv3; "
        "no training/no XYZ correction/no threshold tuning on test"
    )
    print(
        f"boundary_bands=0-5px,5-15px,>15px confidence_threshold={args.confidence_threshold} "
        f"high_confidence_threshold={args.high_confidence_threshold} "
        f"point_sample_limit={args.max_points_per_object_frame}"
    )

    # First pass: calibrate all object-size/thinness and point thresholds from
    # train+validation only.  The pass is deliberately separate so no test
    # statistic can influence a bin or a confidence-failure flag.
    calibration_descriptors: list[dict[str, Any]] = []
    calibration_groups: list[dict[str, Any]] = []
    calibration_finite_groups: list[dict[str, Any]] = []
    expected_layer_ids: tuple[int, ...] | None = None
    for split in ("train", "validation"):
        for clip in clips_by_split[split]:
            inputs = _load_clip_inputs(
                config,
                clip,
                feature_dir,
                confidence_threshold=float(args.confidence_threshold),
            )
            expected_layer_ids = _check_layer_ids(expected_layer_ids, inputs)
            descriptors = _describe_objects(inputs, split=split)
            calibration_descriptors.extend(descriptors)
            calibration_groups.append(_make_point_group(inputs, use_evaluation=True))
            calibration_finite_groups.append(_make_point_group(inputs, use_evaluation=False))
            print(
                f"  calibrated split={split} clip={clip.name} "
                f"objects={len(descriptors)} points={calibration_groups[-1]['count']}"
            )
            del inputs

    thresholds = _build_thresholds(
        calibration_descriptors,
        calibration_groups,
        calibration_finite_groups,
        high_confidence_threshold=float(args.high_confidence_threshold),
    )
    print(
        "calibration="
        f"size_q={thresholds['size_quantiles']} "
        f"thinness_q={thresholds['thinness_quantiles']} "
        f"error_q90={thresholds['error_q90_m']:.6f}"
    )

    all_groups: list[dict[str, Any]] = []
    finite_groups: list[dict[str, Any]] = []
    groups_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    finite_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    groups_by_region: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    groups_by_category: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    point_rows: list[dict[str, Any]] = []
    object_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    scene_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    gradient_bin_rows: list[dict[str, Any]] = []
    confidence_rows: list[dict[str, Any]] = []
    clip_summaries: list[dict[str, Any]] = []

    for split in ("train", "validation", "test"):
        for clip in clips_by_split[split]:
            inputs = _load_clip_inputs(
                config,
                clip,
                feature_dir,
                confidence_threshold=float(args.confidence_threshold),
            )
            expected_layer_ids = _check_layer_ids(expected_layer_ids, inputs)
            descriptors = _describe_objects(inputs, split=split)
            for row in descriptors:
                row["size_bin"] = _assign_bin(
                    float(row["median_area_px"]),
                    thresholds["size_quantiles"],
                    low_label="small",
                    high_label="large",
                )
                row["thinness_bin"] = _assign_bin(
                    float(row["median_thinness"]),
                    thresholds["thinness_quantiles"],
                    low_label="thin",
                    high_label="filled",
                )

            group = _make_point_group(inputs, use_evaluation=True)
            finite_group = _make_point_group(inputs, use_evaluation=False)
            all_groups.append(group)
            finite_groups.append(finite_group)
            groups_by_split[split].append(group)
            finite_by_split[split].append(finite_group)
            for band in BANDS:
                region_group = _make_point_group(
                    inputs,
                    use_evaluation=True,
                    region=band,
                )
                groups_by_region[(split, band)].append(region_group)
            for descriptor in descriptors:
                for dimension, category in (
                    ("size", str(descriptor["size_bin"])),
                    ("thinness", str(descriptor["thinness_bin"])),
                ):
                    category_group = _make_object_category_group(
                        inputs,
                        int(descriptor["gt_object"]),
                        category=dimension,
                        label=category,
                    )
                    if category_group["count"]:
                        groups_by_category[(split, dimension, category)].append(
                            category_group
                        )

            clip_objects = _build_object_rows(
                inputs,
                descriptors,
                thresholds=thresholds,
            )
            clip_boundary = _build_boundary_rows(inputs)
            clip_scene = _build_scene_rows(inputs, descriptors)
            clip_layers = _build_layer_rows(inputs, thresholds)
            clip_bins = _build_gradient_bin_rows(inputs, thresholds)
            clip_confidence = _build_confidence_rows(inputs, thresholds)
            point_rows.extend(
                _build_point_rows(
                    inputs,
                    descriptors,
                    max_points=int(args.max_points_per_object_frame),
                )
            )
            object_rows.extend(clip_objects)
            boundary_rows.extend(clip_boundary)
            scene_rows.extend(clip_scene)
            layer_rows.extend(clip_layers)
            gradient_bin_rows.extend(clip_bins)
            confidence_rows.extend(clip_confidence)
            clip_summaries.append(
                _clip_summary(
                    inputs,
                    descriptors,
                    clip_layers,
                    clip_boundary,
                    split=split,
                )
            )
            _save_visualizations(
                inputs,
                output_dir / "visualizations",
            )
            print(
                f"  analyzed split={split} clip={clip.name} "
                f"objects={len(descriptors)} points={group['count']} "
                f"sampled_rows={sum(1 for row in point_rows if row['clip'] == clip.name)}"
            )
            del inputs

    # Aggregate signal and region statistics from frozen point arrays.  These
    # aggregate rows are also the source for the copyable decision text.
    signal_summary_rows: list[dict[str, Any]] = []
    for split in ("train", "validation", "test", "ALL"):
        selected = all_groups if split == "ALL" else groups_by_split[split]
        combined = _concat_groups(selected)
        if combined["count"]:
            signal_summary_rows.extend(
                _signal_analysis_rows(
                    combined,
                    thresholds,
                    split=split,
                    clip="ALL",
                )
            )

    boundary_summary_rows: list[dict[str, Any]] = []
    for split in ("train", "validation", "test", "ALL"):
        for band in BANDS:
            selected = (
                groups_by_region[(split, band)]
                if split != "ALL"
                else [
                    group
                    for key, values in groups_by_region.items()
                    if key[1] == band
                    for group in values
                ]
            )
            combined = _concat_groups(selected)
            if combined["count"]:
                boundary_summary_rows.append(
                    _region_summary_row(combined, split=split, clip="ALL", band=band)
                )

    category_summary_rows: list[dict[str, Any]] = []
    for dimension, labels in (("size", SIZE_BINS), ("thinness", THINNESS_BINS)):
        for split in ("train", "validation", "test", "ALL"):
            for label in labels:
                selected = (
                    groups_by_category[(split, dimension, label)]
                    if split != "ALL"
                    else [
                        group
                        for key, values in groups_by_category.items()
                        if key[1] == dimension and key[2] == label
                        for group in values
                    ]
                )
                combined = _concat_groups(selected)
                if combined["count"]:
                    category_summary_rows.append(
                        _category_summary_row(
                            combined,
                            split=split,
                            dimension=dimension,
                            label=label,
                        )
                    )

    # Add aggregate rows to the requested CSVs while preserving per-clip rows.
    boundary_rows.extend(boundary_summary_rows)
    scene_rows.extend(_aggregate_scene_rows(groups_by_split))
    layer_rows.extend(signal_summary_rows)
    gradient_bin_rows.extend(
        _aggregate_gradient_bin_rows(
            all_groups,
            groups_by_split,
            thresholds,
        )
    )
    confidence_rows.extend(
        _aggregate_confidence_rows(
            finite_groups,
            finite_by_split,
            thresholds,
        )
    )

    interpretation = _interpret(
        boundary_summary_rows,
        category_summary_rows,
        signal_summary_rows,
        confidence_rows,
    )
    summary: dict[str, Any] = {
        "schema": 1,
        "revision": REVISION,
        "protocol": str(protocol),
        "feature_dir": str(feature_dir),
        "diagnostic_only": 1,
        "no_parameters_updated": 1,
        "no_xyz_correction": 1,
        "splits": {
            split: [clip.name for clip in clips]
            for split, clips in clips_by_split.items()
        },
        "dino": {
            "checkpoint": all_groups[0].get("dino_checkpoint", "") if all_groups else "",
            "layer_ids": list(
                all_groups[0]["dino_layer_ids"] if all_groups else ()
            ),
            "gradient": "mean 4-neighbor cosine discontinuity",
            "final_layer_alias": "dino_final",
            "multi_layer_alias": "dino_multilayer",
        },
        "geometry": {
            "raw_source": "frozen StreamVGGT baseline_world_points",
            "alignment": "frozen cache point_alignment_scale/rotation/translation",
            "point_error": "Euclidean distance between aligned raw point and target metric point",
            "evaluation_support": (
                f"finite GT object support and StreamVGGT confidence >= {args.confidence_threshold}"
            ),
        },
        "boundary_definition": {
            "distance_metric": "Chebyshev erosion on StreamVGGT processed grid",
            "boundary_px": "distance <= 5",
            "near_boundary_px": "5 < distance <= 15",
            "interior_px": "distance > 15",
        },
        "calibration": thresholds,
        "sampling": {
            "per_point_csv": "deterministic sample per GT object/frame",
            "max_points_per_object_frame": int(args.max_points_per_object_frame),
            "aggregate_statistics": "all eligible pixels, not CSV sample",
        },
        "clip_summaries": clip_summaries,
        "boundary_metrics": boundary_summary_rows,
        "category_metrics": category_summary_rows,
        "signal_metrics": signal_summary_rows,
        "interpretation": interpretation,
        "outputs": {},
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output_dir / "summary.json",
        "per_point_or_patch_metrics": output_dir / "per_point_or_patch_metrics.csv",
        "per_object_metrics": output_dir / "per_object_metrics.csv",
        "per_scene_metrics": output_dir / "per_scene_metrics.csv",
        "boundary_metrics": output_dir / "boundary_metrics.csv",
        "dino_layer_metrics": output_dir / "dino_layer_metrics.csv",
        "gradient_error_bins": output_dir / "gradient_error_bins.csv",
        "confidence_failure_metrics": output_dir / "confidence_failure_metrics.csv",
        "copyable_result": output_dir / "copyable_result.txt",
    }
    _write_json(paths["summary"], summary)
    _write_csv(paths["per_point_or_patch_metrics"], point_rows)
    _write_csv(paths["per_object_metrics"], object_rows)
    _write_csv(
        paths["per_scene_metrics"],
        scene_rows + boundary_summary_rows + category_summary_rows,
    )
    _write_csv(paths["boundary_metrics"], boundary_rows)
    _write_csv(paths["dino_layer_metrics"], layer_rows)
    _write_csv(paths["gradient_error_bins"], gradient_bin_rows)
    _write_csv(paths["confidence_failure_metrics"], confidence_rows)
    summary["outputs"] = {key: str(value) for key, value in paths.items()}
    _write_json(paths["summary"], summary)
    _write_copyable(paths["copyable_result"], summary)
    for key in (
        "summary",
        "per_point_or_patch_metrics",
        "per_object_metrics",
        "per_scene_metrics",
        "boundary_metrics",
        "dino_layer_metrics",
        "gradient_error_bins",
        "confidence_failure_metrics",
        "copyable_result",
    ):
        print(f"{key}={paths[key]}")
    print(f"decision={interpretation['decision']}")


def _load_clip_inputs(
    config: Any,
    clip: Any,
    feature_dir: Path,
    *,
    confidence_threshold: float,
) -> ClipInputs:
    data = _load_clip_data(
        config,
        clip,
        feature_dir,
        confidence_threshold=float(confidence_threshold),
    )
    ground_truth = _load_ground_truth_bundle(data)
    gt_masks = ground_truth.stream_masks.detach().bool().cpu()
    payload = load_feature_cache(data.cache_path)
    image_value = payload.get("stream_images")
    if image_value is None:
        raise ValueError(f"Frozen cache lacks stream_images: {data.cache_path}")
    images = torch.as_tensor(image_value).float().cpu()
    if images.ndim == 3:
        images = images.unsqueeze(0)
    if images.ndim != 4:
        raise ValueError(
            f"stream_images in {data.cache_path} must be [S,3,H,W], got {images.shape}"
        )
    if images.shape[1] not in {1, 3} and images.shape[-1] in {1, 3}:
        images = images.permute(0, 3, 1, 2).contiguous()
    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)
    expected_image = (len(data.frame_indices), *data.image_size)
    if tuple(images.shape) != (expected_image[0], 3, expected_image[1], expected_image[2]):
        raise ValueError(
            f"stream_images shape {tuple(images.shape)} disagrees with point grid "
            f"{expected_image} in {data.cache_path}"
        )

    dino_payload = torch.load(data.dino_path, map_location="cpu", weights_only=False)
    dino_layers, dino_layer_ids = _load_dense_layers(dino_payload, data.dino_path)
    dino_metadata = dino_payload.get("metadata", {})
    if not isinstance(dino_metadata, Mapping):
        dino_metadata = {}
    dino_checkpoint = str(
        dino_metadata.get("checkpoint", dino_payload.get("checkpoint", ""))
    )
    gradients: dict[str, torch.Tensor] = {}
    for layer_id in dino_layer_ids:
        dense = dino_layers[layer_id]
        gradient = _cosine_feature_gradient(dense)
        gradients[f"dino_layer_{layer_id}"] = _upsample_map(
            gradient,
            data.image_size,
        )
    final_name = f"dino_layer_{dino_layer_ids[-1]}"
    gradients["dino_final"] = gradients[final_name]
    gradients["dino_multilayer"] = torch.stack(
        [gradients[f"dino_layer_{layer_id}"] for layer_id in dino_layer_ids],
        dim=0,
    ).mean(dim=0)

    stream_features = data.features.detach().float().cpu()
    if stream_features.ndim != 4:
        raise ValueError(
            f"StreamVGGT geometry features must be [S,L,N,C], got {stream_features.shape}"
        )
    sequence, level_count, patch_count, channels = map(int, stream_features.shape)
    patch_height, patch_width = (int(value) for value in data.patch_shape)
    if sequence != len(data.frame_indices) or patch_height * patch_width != patch_count:
        raise ValueError(
            "StreamVGGT feature grid disagrees with frozen cache: "
            f"features={tuple(stream_features.shape)} patch_shape={data.patch_shape}"
        )
    stream_gradient_levels = []
    for level in range(level_count):
        dense = stream_features[:, level].reshape(
            sequence,
            patch_height,
            patch_width,
            channels,
        )
        stream_gradient_levels.append(_cosine_feature_gradient(dense))
    gradients["streamvggt"] = _upsample_map(
        torch.stack(stream_gradient_levels, dim=1).mean(dim=1),
        data.image_size,
    )
    gradients["rgb"] = _rgb_gradient(images)

    aligned_raw = apply_similarity(
        data.raw_native,
        scale=data.scale,
        rotation=data.rotation,
        translation=data.translation,
    ).float().cpu()
    error = torch.linalg.vector_norm(
        aligned_raw - data.target_metric.float().cpu(),
        dim=-1,
    )
    finite = (
        torch.isfinite(aligned_raw).all(dim=-1)
        & torch.isfinite(data.target_metric).all(dim=-1)
        & torch.isfinite(error)
        & torch.isfinite(data.confidence)
    )
    evaluation_valid = finite & (
        data.confidence.float().cpu() >= float(confidence_threshold)
    )
    if tuple(gt_masks.shape[0:1] + gt_masks.shape[2:]) != tuple(error.shape):
        raise ValueError(
            f"GT mask shape {tuple(gt_masks.shape)} disagrees with error {tuple(error.shape)}"
        )
    object_bands: list[dict[str, torch.Tensor]] = []
    region_masks = {
        "boundary": torch.zeros_like(error, dtype=torch.bool),
        "near_boundary": torch.zeros_like(error, dtype=torch.bool),
        "interior": torch.zeros_like(error, dtype=torch.bool),
        "all": torch.zeros_like(error, dtype=torch.bool),
    }
    for object_index in range(int(gt_masks.shape[1])):
        bands = _boundary_bands(gt_masks[:, object_index])
        bands["all"] = gt_masks[:, object_index]
        object_bands.append(bands)
        for band in BANDS:
            region_masks[band] |= bands[band]
    return ClipInputs(
        data=data,
        ground_truth=ground_truth,
        gt_masks=gt_masks,
        images=images,
        aligned_raw=aligned_raw,
        error=error,
        finite=finite,
        evaluation_valid=evaluation_valid,
        gradients=gradients,
        object_bands=tuple(object_bands),
        regions=region_masks,
        dino_layer_ids=dino_layer_ids,
        dino_checkpoint=dino_checkpoint,
    )


def _check_layer_ids(
    expected: tuple[int, ...] | None,
    inputs: ClipInputs,
) -> tuple[int, ...]:
    if expected is None:
        return inputs.dino_layer_ids
    if expected != inputs.dino_layer_ids:
        raise ValueError(
            "DINO dense layer IDs differ across clips: "
            f"expected={expected} clip={inputs.data.name} actual={inputs.dino_layer_ids}"
        )
    return expected


def _load_dense_layers(
    payload: Mapping[str, Any],
    path: Path,
) -> tuple[dict[int, torch.Tensor], tuple[int, ...]]:
    values = payload.get("dense_features_by_layer")
    ids = payload.get("dense_layer_ids")
    if not isinstance(values, Mapping) or not values:
        raise ValueError(
            f"DINO cache lacks dense_features_by_layer: {path}; "
            "run commands_analyze_object_geometry_dinov3_prior.txt"
        )
    if ids is None:
        ids = sorted(int(key) for key in values)
    layer_ids = tuple(int(value) for value in ids)
    if not layer_ids:
        raise ValueError(f"DINO cache has no dense layer IDs: {path}")
    output: dict[int, torch.Tensor] = {}
    for layer_id in layer_ids:
        value = values.get(str(layer_id), values.get(layer_id))
        if value is None:
            raise ValueError(f"DINO cache layer {layer_id} is missing: {path}")
        dense = torch.as_tensor(value).float().cpu()
        if dense.ndim != 4:
            raise ValueError(
                f"DINO layer {layer_id} in {path} must be [S,h,w,C], got {dense.shape}"
            )
        output[layer_id] = dense
    sequence = int(output[layer_ids[0]].shape[0])
    spatial = tuple(output[layer_ids[0]].shape[1:3])
    channels = int(output[layer_ids[0]].shape[-1])
    for layer_id, value in output.items():
        if int(value.shape[0]) != sequence or tuple(value.shape[1:3]) != spatial:
            raise ValueError(f"DINO layer grids disagree in {path}")
        if int(value.shape[-1]) != channels:
            raise ValueError(f"DINO layer channel dimensions disagree in {path}")
    return output, layer_ids


def _cosine_feature_gradient(dense: torch.Tensor) -> torch.Tensor:
    """Mean 4-neighbor cosine discontinuity for [S,H,W,C] features."""

    values = torch.as_tensor(dense).float()
    if values.ndim != 4:
        raise ValueError(f"Expected [S,H,W,C] features, got {values.shape}")
    values = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = F.normalize(values, dim=-1)
    sequence, height, width, _ = values.shape
    result = torch.zeros(sequence, height, width, dtype=torch.float32)
    counts = torch.zeros_like(result)

    horizontal = 1.0 - (values[:, :, 1:] * values[:, :, :-1]).sum(dim=-1)
    result[:, :, 1:] += horizontal
    result[:, :, :-1] += horizontal
    counts[:, :, 1:] += 1.0
    counts[:, :, :-1] += 1.0
    vertical = 1.0 - (values[:, 1:] * values[:, :-1]).sum(dim=-1)
    result[:, 1:] += vertical
    result[:, :-1] += vertical
    counts[:, 1:] += 1.0
    counts[:, :-1] += 1.0
    return result / counts.clamp_min(1.0)


def _rgb_gradient(images: torch.Tensor) -> torch.Tensor:
    """Mean 4-neighbor absolute RGB difference on the processed grid."""

    values = torch.as_tensor(images).float().cpu()
    if float(values.max()) > 1.5:
        values = values / 255.0
    values = values.clamp(0.0, 1.0)
    sequence, channels, height, width = values.shape
    result = torch.zeros(sequence, height, width, dtype=torch.float32)
    counts = torch.zeros_like(result)
    horizontal = (values[:, :, :, 1:] - values[:, :, :, :-1]).abs().mean(dim=1)
    result[:, :, 1:] += horizontal
    result[:, :, :-1] += horizontal
    counts[:, :, 1:] += 1.0
    counts[:, :, :-1] += 1.0
    vertical = (values[:, :, 1:] - values[:, :, :-1]).abs().mean(dim=1)
    result[:, 1:] += vertical
    result[:, :-1] += vertical
    counts[:, 1:] += 1.0
    counts[:, :-1] += 1.0
    return result / counts.clamp_min(1.0)


def _upsample_map(values: torch.Tensor, size: Sequence[int]) -> torch.Tensor:
    tensor = torch.as_tensor(values).float().cpu()
    if tensor.ndim != 3:
        raise ValueError(f"Map must be [S,H,W], got {tensor.shape}")
    return F.interpolate(
        tensor[:, None],
        size=(int(size[0]), int(size[1])),
        mode="bilinear",
        align_corners=False,
    )[:, 0]


def _boundary_bands(mask: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return mutually exclusive 5px/15px Chebyshev boundary bands."""

    value = torch.as_tensor(mask).bool().cpu()
    if value.ndim != 3:
        raise ValueError(f"Object masks must be [S,H,W], got {value.shape}")
    eroded5 = _erode(value, 5)
    eroded15 = _erode(value, 15)
    return {
        "boundary": value & ~eroded5,
        "near_boundary": eroded5 & ~eroded15,
        "interior": eroded15,
    }


def _erode(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if int(radius) <= 0:
        return mask.bool()
    value = (~mask.bool()).float()[:, None]
    padded = F.pad(
        value,
        (int(radius), int(radius), int(radius), int(radius)),
        mode="constant",
        value=1.0,
    )
    outside = F.max_pool2d(
        padded,
        kernel_size=2 * int(radius) + 1,
        stride=1,
    )
    return (outside[:, 0] == 0.0) & mask.bool()


def _describe_objects(inputs: ClipInputs, *, split: str) -> list[dict[str, Any]]:
    labels = tuple(str(value) for value in inputs.ground_truth.instances.labels)
    rows: list[dict[str, Any]] = []
    for object_index in range(int(inputs.gt_masks.shape[1])):
        areas: list[float] = []
        thinness: list[float] = []
        aspect: list[float] = []
        for frame in range(int(inputs.gt_masks.shape[0])):
            ys, xs = torch.nonzero(
                inputs.gt_masks[frame, object_index],
                as_tuple=True,
            )
            if ys.numel() == 0:
                continue
            area = float(ys.numel())
            bbox_height = float(int(ys.max()) - int(ys.min()) + 1)
            bbox_width = float(int(xs.max()) - int(xs.min()) + 1)
            bbox_area = max(1.0, bbox_height * bbox_width)
            areas.append(area)
            thinness.append(area / bbox_area)
            aspect.append(min(bbox_height, bbox_width) / max(bbox_height, bbox_width))
        if not areas:
            continue
        rows.append(
            {
                "split": split,
                "clip": inputs.data.name,
                "scene_id": inputs.data.scene_id,
                "gt_object": int(object_index),
                "gt_instance_id": int(inputs.ground_truth.instances.instance_ids[object_index]),
                "gt_label": labels[object_index] if object_index < len(labels) else "object",
                "observed_frames": len(areas),
                "median_area_px": _median_list(areas),
                "median_thinness": _median_list(thinness),
                "median_aspect": _median_list(aspect),
            }
        )
    return rows


def _build_thresholds(
    descriptors: Sequence[Mapping[str, Any]],
    groups: Sequence[Mapping[str, Any]],
    finite_groups: Sequence[Mapping[str, Any]],
    *,
    high_confidence_threshold: float,
) -> dict[str, Any]:
    size_values = torch.tensor(
        [float(row["median_area_px"]) for row in descriptors],
        dtype=torch.float32,
    )
    thinness_values = torch.tensor(
        [float(row["median_thinness"]) for row in descriptors],
        dtype=torch.float32,
    )
    if not descriptors:
        raise ValueError("No train/validation GT objects are available for calibration.")
    combined = _concat_groups(groups)
    finite_combined = _concat_groups(finite_groups)
    if not combined["count"] or not finite_combined["count"]:
        raise ValueError("No train/validation GT object points are available for calibration.")
    signal_quantiles: dict[str, list[float]] = {}
    for name, values in combined["signals"].items():
        signal_quantiles[name] = [
            _quantile(values, quantile)
            for quantile in (0.20, 0.40, 0.60, 0.80)
        ]
    stream_values = finite_combined["signals"].get("streamvggt")
    return {
        "calibration_splits": ["train", "validation"],
        "size_quantiles": [_quantile(size_values, 1.0 / 3.0), _quantile(size_values, 2.0 / 3.0)],
        "thinness_quantiles": [
            _quantile(thinness_values, 1.0 / 3.0),
            _quantile(thinness_values, 2.0 / 3.0),
        ],
        "error_q90_m": _quantile(finite_combined["error"], 0.90),
        "gradient_quantiles": signal_quantiles,
        "high_confidence_threshold": float(high_confidence_threshold),
        "stream_gradient_q50": (
            _quantile(stream_values, 0.50) if stream_values is not None else float("nan")
        ),
    }


def _make_point_group(
    inputs: ClipInputs,
    *,
    use_evaluation: bool,
    region: str | None = None,
) -> dict[str, Any]:
    if region is not None and region not in BANDS:
        raise ValueError(f"Unknown diagnostic region {region!r}")
    base = inputs.evaluation_valid if use_evaluation else inputs.finite
    base = base & inputs.regions["all"]
    if region is not None and region != "all":
        base = base & inputs.regions[region]
    return _group_from_mask(inputs, base, include_regions=True)


def _make_object_category_group(
    inputs: ClipInputs,
    object_index: int,
    *,
    category: str,
    label: str,
) -> dict[str, Any]:
    del category  # The label is already resolved by the caller.
    del label
    mask = inputs.object_bands[object_index]["all"] & inputs.evaluation_valid
    return _group_from_mask(inputs, mask, include_regions=True)


def _group_from_mask(
    inputs: ClipInputs,
    mask: torch.Tensor,
    *,
    include_regions: bool,
) -> dict[str, Any]:
    selected = torch.as_tensor(mask).bool().cpu()
    error = inputs.error[selected].float().cpu()
    output: dict[str, Any] = {
        "split": inputs.data.split,
        "clip": inputs.data.name,
        "scene_id": inputs.data.scene_id,
        "count": int(error.numel()),
        "error": error,
        "confidence": inputs.data.confidence.float().cpu()[selected],
        "signals": {
            name: values[selected].float().cpu()
            for name, values in inputs.gradients.items()
        },
        "dino_layer_ids": inputs.dino_layer_ids,
        "dino_checkpoint": inputs.dino_checkpoint,
    }
    if include_regions:
        output["boundary"] = inputs.regions["boundary"][selected]
        output["near_boundary"] = inputs.regions["near_boundary"][selected]
        output["interior"] = inputs.regions["interior"][selected]
    return output


def _concat_groups(groups: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not groups:
        return {
            "count": 0,
            "error": torch.empty(0),
            "confidence": torch.empty(0),
            "signals": {},
            "boundary": torch.empty(0, dtype=torch.bool),
            "near_boundary": torch.empty(0, dtype=torch.bool),
            "interior": torch.empty(0, dtype=torch.bool),
            "dino_layer_ids": (),
        }
    keys = sorted(
        set().union(*(set(group.get("signals", {})) for group in groups))
    )
    return {
        "count": sum(int(group.get("count", 0)) for group in groups),
        "error": _cat_field(groups, "error"),
        "confidence": _cat_field(groups, "confidence"),
        "signals": {
            key: _cat_values(
                [group.get("signals", {}).get(key, torch.empty(0)) for group in groups]
            )
            for key in keys
        },
        "boundary": _cat_field(groups, "boundary", dtype=torch.bool),
        "near_boundary": _cat_field(groups, "near_boundary", dtype=torch.bool),
        "interior": _cat_field(groups, "interior", dtype=torch.bool),
        "dino_layer_ids": tuple(groups[0].get("dino_layer_ids", ())),
    }


def _cat_field(
    groups: Sequence[Mapping[str, Any]],
    key: str,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return _cat_values(
        [torch.as_tensor(group.get(key, torch.empty(0)), dtype=dtype) for group in groups]
    )


def _cat_values(values: Sequence[torch.Tensor]) -> torch.Tensor:
    nonempty = [torch.as_tensor(value).reshape(-1) for value in values if torch.as_tensor(value).numel()]
    return torch.cat(nonempty) if nonempty else torch.empty(0)


def _build_object_rows(
    inputs: ClipInputs,
    descriptors: Sequence[Mapping[str, Any]],
    *,
    thresholds: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for descriptor in descriptors:
        object_index = int(descriptor["gt_object"])
        row = dict(descriptor)
        row["size_bin"] = _assign_bin(
            float(descriptor["median_area_px"]),
            thresholds["size_quantiles"],
            low_label="small",
            high_label="large",
        )
        row["thinness_bin"] = _assign_bin(
            float(descriptor["median_thinness"]),
            thresholds["thinness_quantiles"],
            low_label="thin",
            high_label="filled",
        )
        row["metric"] = "object"
        for band in BANDS:
            mask = inputs.object_bands[object_index][band] & inputs.evaluation_valid
            group = _group_from_mask(inputs, mask, include_regions=False)
            _add_error_stats(row, group["error"], prefix=f"{band}_")
            for signal in ("dino_final", "dino_multilayer", "streamvggt", "rgb"):
                row[f"{band}_{signal}_mean"] = _mean(group["signals"].get(signal, torch.empty(0)))
        row["boundary_to_interior_rmse"] = _ratio(
            row.get("boundary_rmse_m"), row.get("interior_rmse_m")
        )
        rows.append(row)
    return rows


def _build_boundary_rows(inputs: ClipInputs) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for band in BANDS:
        group = _make_point_group(inputs, use_evaluation=True, region=band)
        row = _region_summary_row(
            group,
            split=inputs.data.split,
            clip=inputs.data.name,
            band=band,
        )
        rows.append(row)
    return rows


def _region_summary_row(
    group: Mapping[str, Any],
    *,
    split: str,
    clip: str,
    band: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "split": split,
        "clip": clip,
        "metric": "boundary_band",
        "band": band,
    }
    _add_error_stats(row, group["error"])
    for signal in ("dino_final", "dino_multilayer", "streamvggt", "rgb"):
        row[f"{signal}_mean"] = _mean(group["signals"].get(signal, torch.empty(0)))
    return row


def _category_summary_row(
    group: Mapping[str, Any],
    *,
    split: str,
    dimension: str,
    label: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "split": split,
        "clip": "ALL",
        "metric": "object_category",
        "dimension": dimension,
        "label": label,
        "band": "all",
    }
    _add_error_stats(row, group["error"])
    for signal in ("dino_final", "dino_multilayer", "streamvggt", "rgb"):
        row[f"{signal}_mean"] = _mean(
            group["signals"].get(signal, torch.empty(0))
        )
    return row


def _build_scene_rows(
    inputs: ClipInputs,
    descriptors: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for band in BANDS:
        group = _make_point_group(inputs, use_evaluation=True, region=band)
        row = _region_summary_row(
            group,
            split=inputs.data.split,
            clip=inputs.data.name,
            band=band,
        )
        row["group_type"] = "band"
        row["group"] = band
        row["object_count"] = len(descriptors)
        rows.append(row)
    for dimension, labels in (("size", SIZE_BINS), ("thinness", THINNESS_BINS)):
        for label in labels:
            selected = [
                row for row in descriptors
                if str(row[f"{dimension}_bin"]) == label
            ]
            values = []
            for descriptor in selected:
                object_index = int(descriptor["gt_object"])
                mask = inputs.object_bands[object_index]["all"] & inputs.evaluation_valid
                values.append(_group_from_mask(inputs, mask, include_regions=False))
            group = _concat_groups(values)
            row = _region_summary_row(
                group,
                split=inputs.data.split,
                clip=inputs.data.name,
                band="all",
            )
            row["group_type"] = dimension
            row["group"] = label
            row["object_count"] = len(selected)
            rows.append(row)
    return rows


def _build_layer_rows(
    inputs: ClipInputs,
    thresholds: Mapping[str, Any],
) -> list[dict[str, Any]]:
    group = _make_point_group(inputs, use_evaluation=True)
    return _signal_analysis_rows(
        group,
        thresholds,
        split=inputs.data.split,
        clip=inputs.data.name,
    )


def _signal_analysis_rows(
    group: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    *,
    split: str,
    clip: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in sorted(group.get("signals", {})):
        row = _signal_analysis(group, name, thresholds)
        row.update(
            {
                "split": split,
                "clip": clip,
                "signal": name,
                "metric": "spatial_gradient_vs_geometry_error",
            }
        )
        if name.startswith("dino_layer_"):
            row["layer_id"] = int(name.rsplit("_", 1)[-1])
            row["source"] = "dinov3_intermediate_or_final"
        elif name == "dino_final":
            row["layer_id"] = int(group.get("dino_layer_ids", ([-1]))[-1])
            row["source"] = "dinov3_final_alias"
        elif name == "dino_multilayer":
            row["layer_id"] = "multi"
            row["source"] = "dinov3_multilayer_mean"
        else:
            row["layer_id"] = "control"
            row["source"] = name
        rows.append(row)
    return rows


def _signal_analysis(
    group: Mapping[str, Any],
    signal: str,
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    error = torch.as_tensor(group["error"]).float()
    values = torch.as_tensor(group["signals"].get(signal, torch.empty(0))).float()
    finite = torch.isfinite(error) & torch.isfinite(values)
    error = error[finite]
    values = values[finite]
    boundary = torch.as_tensor(group.get("boundary", torch.zeros_like(finite))).bool()[finite]
    interior = torch.as_tensor(group.get("interior", torch.zeros_like(finite))).bool()[finite]
    row: dict[str, Any] = {"count": int(error.numel())}
    _add_error_stats(row, error)
    row["spearman_all"] = _spearman(values, error)
    row["spearman_boundary"] = _spearman(values[boundary], error[boundary])
    row["spearman_interior"] = _spearman(values[interior], error[interior])
    row["gradient_mean_boundary"] = _mean(values[boundary])
    row["gradient_mean_interior"] = _mean(values[interior])
    row["boundary_to_interior_gradient"] = _ratio(
        row["gradient_mean_boundary"], row["gradient_mean_interior"]
    )
    quantiles = thresholds.get("gradient_quantiles", {}).get(signal, [])
    top80 = float(quantiles[-1]) if quantiles else float("nan")
    target = error >= float(thresholds["error_q90_m"])
    selected = values >= top80 if math.isfinite(top80) else torch.zeros_like(target)
    row["high_error_count"] = int(target.sum())
    row["top80_gradient_count"] = int(selected.sum())
    row["high_error_capture_top80"] = _fraction((target & selected).sum(), target.sum())
    row["high_error_precision_top80"] = _fraction((target & selected).sum(), selected.sum())

    confidence = torch.as_tensor(group.get("confidence", torch.empty(0))).float()
    if confidence.numel() == finite.numel():
        confidence = confidence[finite]
    else:
        confidence = torch.full_like(error, float("nan"))
    high_conf = confidence >= float(thresholds["high_confidence_threshold"])
    high_conf_error = high_conf & target
    row["high_confidence_count"] = int(high_conf.sum())
    row["high_confidence_high_error_count"] = int(high_conf_error.sum())
    row["high_confidence_high_error_rate"] = _fraction(
        high_conf_error.sum(), high_conf.sum()
    )
    row["high_confidence_high_error_capture_top80"] = _fraction(
        (high_conf_error & selected).sum(), high_conf_error.sum()
    )

    stream = torch.as_tensor(group["signals"].get("streamvggt", torch.empty(0))).float()
    if stream.numel() == finite.numel():
        stream = stream[finite]
    else:
        stream = torch.full_like(error, float("nan"))
    stream_q50 = float(thresholds.get("stream_gradient_q50", float("nan")))
    low_stream = torch.isfinite(stream) & (stream <= stream_q50)
    complementary = high_conf & target & low_stream
    row["complementary_high_error_count"] = int(complementary.sum())
    row["complementary_high_error_capture_top80"] = _fraction(
        (complementary & selected).sum(), complementary.sum()
    )
    return row


def _build_gradient_bin_rows(
    inputs: ClipInputs,
    thresholds: Mapping[str, Any],
) -> list[dict[str, Any]]:
    group = _make_point_group(inputs, use_evaluation=True)
    rows: list[dict[str, Any]] = []
    for signal, values in group["signals"].items():
        quantiles = thresholds["gradient_quantiles"].get(signal, [])
        if len(quantiles) != 4:
            continue
        values = values.float()
        for index, label in enumerate(("q1", "q2", "q3", "q4", "q5")):
            mask = _quantile_bin_mask(values, quantiles, index)
            row: dict[str, Any] = {
                "split": inputs.data.split,
                "clip": inputs.data.name,
                "scene_id": inputs.data.scene_id,
                "signal": signal,
                "gradient_bin": label,
            }
            _add_error_stats(row, group["error"][mask])
            row["gradient_mean"] = _mean(values[mask])
            rows.append(row)
    return rows


def _aggregate_gradient_bin_rows(
    all_groups: Sequence[Mapping[str, Any]],
    groups_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    thresholds: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, groups in [("train", groups_by_split["train"]), ("validation", groups_by_split["validation"]), ("test", groups_by_split["test"]), ("ALL", all_groups)]:
        group = _concat_groups(groups)
        if not group["count"]:
            continue
        for signal, values in group["signals"].items():
            quantiles = thresholds["gradient_quantiles"].get(signal, [])
            if len(quantiles) != 4:
                continue
            for index, label in enumerate(("q1", "q2", "q3", "q4", "q5")):
                mask = _quantile_bin_mask(values, quantiles, index)
                row = {
                    "split": split,
                    "clip": "ALL",
                    "scene_id": "ALL",
                    "signal": signal,
                    "gradient_bin": label,
                }
                _add_error_stats(row, group["error"][mask])
                row["gradient_mean"] = _mean(values[mask])
                rows.append(row)
    return rows


def _build_confidence_rows(
    inputs: ClipInputs,
    thresholds: Mapping[str, Any],
) -> list[dict[str, Any]]:
    group = _make_point_group(inputs, use_evaluation=False)
    rows: list[dict[str, Any]] = []
    confidence = group["confidence"]
    for low, high, label in CONFIDENCE_BUCKETS:
        mask = torch.isfinite(confidence) & (confidence >= low) & (confidence < high)
        row: dict[str, Any] = {
            "split": inputs.data.split,
            "clip": inputs.data.name,
            "scene_id": inputs.data.scene_id,
            "analysis": "confidence_bucket",
            "confidence_bucket": label,
        }
        _add_error_stats(row, group["error"][mask])
        row["confidence_mean"] = _mean(confidence[mask])
        row["high_error_rate"] = _fraction(
            (group["error"][mask] >= float(thresholds["error_q90_m"])).sum(),
            mask.sum(),
        )
        rows.append(row)

    high_conf_mask = confidence >= float(thresholds["high_confidence_threshold"])
    target = high_conf_mask & (group["error"] >= float(thresholds["error_q90_m"]))
    stream = group["signals"].get("streamvggt", torch.full_like(group["error"], float("nan")))
    low_stream = stream <= float(thresholds["stream_gradient_q50"])
    for signal, values in group["signals"].items():
        quantiles = thresholds["gradient_quantiles"].get(signal, [])
        top80 = float(quantiles[-1]) if quantiles else float("nan")
        selected = values >= top80 if math.isfinite(top80) else torch.zeros_like(target)
        row = {
            "split": inputs.data.split,
            "clip": inputs.data.name,
            "scene_id": inputs.data.scene_id,
            "analysis": "high_confidence_high_error",
            "signal": signal,
            "confidence_bucket": "high_confidence",
            "base_count": int(high_conf_mask.sum()),
            "target_count": int(target.sum()),
            "selected_count": int((high_conf_mask & selected).sum()),
            "target_rate": _fraction(target.sum(), high_conf_mask.sum()),
            "target_capture_top80": _fraction((target & selected).sum(), target.sum()),
            "target_precision_top80": _fraction(
                (target & selected).sum(), (high_conf_mask & selected).sum()
            ),
            "complementary_base_count": int((high_conf_mask & low_stream).sum()),
            "complementary_target_count": int((target & low_stream).sum()),
            "complementary_capture_top80": _fraction(
                (target & low_stream & selected).sum(), (target & low_stream).sum()
            ),
        }
        rows.append(row)
    return rows


def _aggregate_confidence_rows(
    finite_groups: Sequence[Mapping[str, Any]],
    finite_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    thresholds: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, groups in [("train", finite_by_split["train"]), ("validation", finite_by_split["validation"]), ("test", finite_by_split["test"]), ("ALL", finite_groups)]:
        group = _concat_groups(groups)
        if not group["count"]:
            continue
        confidence = group["confidence"]
        for low, high, label in CONFIDENCE_BUCKETS:
            mask = torch.isfinite(confidence) & (confidence >= low) & (confidence < high)
            row = {
                "split": split,
                "clip": "ALL",
                "scene_id": "ALL",
                "analysis": "confidence_bucket",
                "confidence_bucket": label,
            }
            _add_error_stats(row, group["error"][mask])
            row["confidence_mean"] = _mean(confidence[mask])
            row["high_error_rate"] = _fraction(
                (group["error"][mask] >= float(thresholds["error_q90_m"])).sum(),
                mask.sum(),
            )
            rows.append(row)
        high_conf_mask = confidence >= float(thresholds["high_confidence_threshold"])
        target = high_conf_mask & (group["error"] >= float(thresholds["error_q90_m"]))
        low_stream = group["signals"]["streamvggt"] <= float(thresholds["stream_gradient_q50"])
        for signal, values in group["signals"].items():
            quantiles = thresholds["gradient_quantiles"].get(signal, [])
            top80 = float(quantiles[-1]) if quantiles else float("nan")
            selected = values >= top80 if math.isfinite(top80) else torch.zeros_like(target)
            rows.append(
                {
                    "split": split,
                    "clip": "ALL",
                    "scene_id": "ALL",
                    "analysis": "high_confidence_high_error",
                    "signal": signal,
                    "confidence_bucket": "high_confidence",
                    "base_count": int(high_conf_mask.sum()),
                    "target_count": int(target.sum()),
                    "selected_count": int((high_conf_mask & selected).sum()),
                    "target_rate": _fraction(target.sum(), high_conf_mask.sum()),
                    "target_capture_top80": _fraction((target & selected).sum(), target.sum()),
                    "target_precision_top80": _fraction((target & selected).sum(), (high_conf_mask & selected).sum()),
                    "complementary_base_count": int((high_conf_mask & low_stream).sum()),
                    "complementary_target_count": int((target & low_stream).sum()),
                    "complementary_capture_top80": _fraction((target & low_stream & selected).sum(), (target & low_stream).sum()),
                }
            )
    return rows


def _build_point_rows(
    inputs: ClipInputs,
    descriptors: Sequence[Mapping[str, Any]],
    *,
    max_points: int,
) -> list[dict[str, Any]]:
    labels = {int(row["gt_object"]): str(row["gt_label"]) for row in descriptors}
    descriptor_by_object = {int(row["gt_object"]): row for row in descriptors}
    rows: list[dict[str, Any]] = []
    for object_index, bands in enumerate(inputs.object_bands):
        descriptor = descriptor_by_object.get(object_index)
        if descriptor is None:
            continue
        for frame in range(int(inputs.gt_masks.shape[0])):
            valid = bands["all"][frame] & inputs.evaluation_valid[frame]
            coordinates = torch.nonzero(valid, as_tuple=False)
            coordinates = _subsample_coordinates(coordinates, max_points)
            for coordinate in coordinates.tolist():
                y, x = int(coordinate[0]), int(coordinate[1])
                band = "interior"
                if bool(bands["boundary"][frame, y, x]):
                    band = "boundary"
                elif bool(bands["near_boundary"][frame, y, x]):
                    band = "near_boundary"
                row: dict[str, Any] = {
                    "split": inputs.data.split,
                    "clip": inputs.data.name,
                    "scene_id": inputs.data.scene_id,
                    "sequence_index": frame,
                    "frame_index": int(inputs.data.frame_indices[frame]),
                    "gt_object": object_index,
                    "gt_label": labels[object_index],
                    "size_bin": descriptor["size_bin"],
                    "thinness_bin": descriptor["thinness_bin"],
                    "band": band,
                    "y": y,
                    "x": x,
                    "raw_error_m": float(inputs.error[frame, y, x]),
                    "streamvggt_confidence": float(inputs.data.confidence[frame, y, x]),
                }
                for signal, values in inputs.gradients.items():
                    row[f"{signal}_gradient"] = float(values[frame, y, x])
                rows.append(row)
    return rows


def _clip_summary(
    inputs: ClipInputs,
    descriptors: Sequence[Mapping[str, Any]],
    layer_rows: Sequence[Mapping[str, Any]],
    boundary_rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
) -> dict[str, Any]:
    by_signal = {
        str(row["signal"]): {
            key: value
            for key, value in row.items()
            if key
            in {
                "spearman_all",
                "spearman_boundary",
                "spearman_interior",
                "boundary_to_interior_gradient",
                "high_error_capture_top80",
                "high_confidence_high_error_capture_top80",
            }
        }
        for row in layer_rows
    }
    boundary = {
        str(row["band"]): {
            key: value
            for key, value in row.items()
            if key in {"count", "rmse_m", "median_m", "p90_m"}
        }
        for row in boundary_rows
    }
    return {
        "split": split,
        "clip": inputs.data.name,
        "scene_id": inputs.data.scene_id,
        "frames": len(inputs.data.frame_indices),
        "gt_objects": len(descriptors),
        "eligible_object_points": int(_make_point_group(inputs, use_evaluation=True)["count"]),
        "boundary": boundary,
        "signals": by_signal,
    }


def _aggregate_scene_rows(
    groups_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for split in ("train", "validation", "test", "ALL"):
        selected_groups = (
            groups_by_split[split]
            if split != "ALL"
            else [group for values in groups_by_split.values() for group in values]
        )
        # The aggregate scene file carries all bands and both object-property
        # groupings; detailed rows remain in boundary/category CSVs too.
        if selected_groups:
            # Region masks are not retained in a point group, so the
            # aggregate band is represented by the overall object-support row;
            # boundary_metrics.csv is authoritative for actual bands.
            row = _region_summary_row(
                _concat_groups(selected_groups),
                split=split,
                clip="ALL",
                band="all_objects_support",
            )
            row["group_type"] = "aggregate_object_support"
            row["group"] = "all_objects"
            output.append(row)
    return output


def _interpret(
    boundary_rows: Sequence[Mapping[str, Any]],
    category_rows: Sequence[Mapping[str, Any]],
    signal_rows: Sequence[Mapping[str, Any]],
    confidence_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    def find_region(split: str, band: str) -> Mapping[str, Any] | None:
        return next(
            (
                row
                for row in boundary_rows
                if row.get("split") == split
                and row.get("clip") == "ALL"
                and row.get("band") == band
            ),
            None,
        )

    def find_category(split: str, dimension: str, label: str) -> Mapping[str, Any] | None:
        return next(
            (
                row
                for row in category_rows
                if row.get("split") == split
                and row.get("dimension") == dimension
                and row.get("label") == label
            ),
            None,
        )

    def find_signal(split: str, signal: str) -> Mapping[str, Any] | None:
        return next(
            (
                row
                for row in signal_rows
                if row.get("split") == split
                and row.get("clip") == "ALL"
                and row.get("signal") == signal
            ),
            None,
        )

    boundary_flags: dict[str, bool] = {}
    for split in ("validation", "test"):
        boundary = find_region(split, "boundary")
        interior = find_region(split, "interior")
        ratio = _ratio(
            boundary.get("rmse_m") if boundary else float("nan"),
            interior.get("rmse_m") if interior else float("nan"),
        )
        boundary_flags[split] = bool(math.isfinite(ratio) and ratio > 1.05)

    size_flags: dict[str, bool] = {}
    thin_flags: dict[str, bool] = {}
    for split in ("validation", "test"):
        small = find_category(split, "size", "small")
        large = find_category(split, "size", "large")
        thin = find_category(split, "thinness", "thin")
        filled = find_category(split, "thinness", "filled")
        size_flags[split] = bool(
            small and large and _ratio(small.get("rmse_m"), large.get("rmse_m")) > 1.05
        )
        thin_flags[split] = bool(
            thin and filled and _ratio(thin.get("rmse_m"), filled.get("rmse_m")) > 1.05
        )

    multi = find_signal("validation", "dino_multilayer") or find_signal("test", "dino_multilayer")
    final = find_signal("validation", "dino_final") or find_signal("test", "dino_final")
    rgb = find_signal("validation", "rgb") or find_signal("test", "rgb")
    stream = find_signal("validation", "streamvggt") or find_signal("test", "streamvggt")

    def value(row: Mapping[str, Any] | None, key: str) -> float:
        return float(row.get(key, float("nan"))) if row else float("nan")

    dino_error_relation = bool(value(multi, "spearman_all") > 0.05)
    dino_beyond_rgb = bool(
        value(multi, "spearman_all") > value(rgb, "spearman_all") + 0.02
        or value(multi, "high_error_capture_top80")
        > value(rgb, "high_error_capture_top80") + 0.03
    )
    dino_beyond_stream = bool(
        value(multi, "spearman_all") > value(stream, "spearman_all") + 0.02
        or value(multi, "complementary_high_error_capture_top80")
        > value(stream, "complementary_high_error_capture_top80") + 0.03
    )
    multilayer_better = bool(
        value(multi, "spearman_all") > value(final, "spearman_all") + 0.01
        or value(multi, "high_error_capture_top80")
        > value(final, "high_error_capture_top80") + 0.02
    )
    confidence_failure = any(
        str(row.get("analysis")) == "high_confidence_high_error"
        and row.get("split") == "validation"
        and row.get("clip") == "ALL"
        and row.get("signal") == "dino_multilayer"
        and float(row.get("target_rate", 0.0)) > 0.05
        for row in confidence_rows
    )
    dino_confidence_identifiable = False
    dino_confidence_row = next(
        (
            row
            for row in confidence_rows
            if row.get("analysis") == "high_confidence_high_error"
            and row.get("split") == "validation"
            and row.get("clip") == "ALL"
            and row.get("signal") == "dino_multilayer"
        ),
        None,
    )
    stream_confidence_row = next(
        (
            row
            for row in confidence_rows
            if row.get("analysis") == "high_confidence_high_error"
            and row.get("split") == "validation"
            and row.get("clip") == "ALL"
            and row.get("signal") == "streamvggt"
        ),
        None,
    )
    if dino_confidence_row and stream_confidence_row:
        dino_confidence_identifiable = bool(
            float(dino_confidence_row.get("target_rate", 0.0)) > 0.05
            and float(dino_confidence_row.get("target_capture_top80", float("nan")))
            > float(stream_confidence_row.get("target_capture_top80", float("nan"))) + 0.03
        )
    flags = {
        "boundary_failure": any(boundary_flags.values()),
        "small_or_thin_failure": any(size_flags.values()) or any(thin_flags.values()),
        "dino_gradient_relates_to_error": dino_error_relation,
        "dino_beyond_rgb": dino_beyond_rgb,
        "dino_beyond_streamvggt": dino_beyond_stream,
        "high_confidence_failure_exists": confidence_failure,
        "high_confidence_failure_dino_identifiable": dino_confidence_identifiable,
        "multilayer_beats_final": multilayer_better,
    }
    evidence_count = sum(int(value) for value in flags.values())
    # A dense fusion follow-up is justified only when the failure mode and a
    # non-redundant DINO signal both appear.  This remains a diagnostic GO, not
    # a claim that a decoder will improve geometry.
    decision = "GO" if flags["boundary_failure"] and evidence_count >= 4 else "NO_GO"
    return {
        "decision": decision,
        "evidence_count": evidence_count,
        "evidence_flags": flags,
        "boundary_failure_by_split": boundary_flags,
        "small_failure_by_split": size_flags,
        "thin_failure_by_split": thin_flags,
        "comparison": {
            "dino_multilayer": multi,
            "dino_final": final,
            "rgb": rgb,
            "streamvggt": stream,
        },
        "caveat": (
            "This is a frozen spatial diagnosis. A positive result means that "
            "DINO structure is associated with StreamVGGT failures; it does not "
            "prove that concatenation or XYZ regression will improve a map."
        ),
    }


def _write_copyable(path: Path, summary: Mapping[str, Any]) -> None:
    interpretation = summary["interpretation"]
    lines = [
        "===== DINOV3_OBJECT_GEOMETRY_PRIOR_BEGIN =====",
        f"revision={summary['revision']}",
        "diagnostic_only=1",
        "parameters_updated=0",
        "xyz_corrected=0",
        "gt_role=diagnosis_only",
        "",
        "QUESTIONS",
        "1_boundary_geometry_worse_than_interior="
        + str(interpretation["evidence_flags"]["boundary_failure"]),
        "2_small_or_thin_objects_worse="
        + str(interpretation["evidence_flags"]["small_or_thin_failure"]),
        "3_dino_gradient_correlates_with_geometry_error="
        + str(interpretation["evidence_flags"]["dino_gradient_relates_to_error"]),
        "4_dino_signal_beyond_rgb="
        + str(interpretation["evidence_flags"]["dino_beyond_rgb"]),
        "5_dino_signal_complementary_to_streamvggt="
        + str(interpretation["evidence_flags"]["dino_beyond_streamvggt"]),
        "6_high_confidence_streamvggt_failures_exist="
        + str(interpretation["evidence_flags"]["high_confidence_failure_exists"]),
        "7_high_confidence_failures_identifiable_by_dino="
        + str(interpretation["evidence_flags"]["high_confidence_failure_dino_identifiable"]),
        "8_multilayer_dino_beats_final_layer="
        + str(interpretation["evidence_flags"]["multilayer_beats_final"]),
        "",
        f"decision={interpretation['decision']}",
        f"evidence_count={interpretation['evidence_count']}/8",
        "boundary_failure_by_split=" + json.dumps(interpretation["boundary_failure_by_split"], sort_keys=True),
        "small_failure_by_split=" + json.dumps(interpretation["small_failure_by_split"], sort_keys=True),
        "thin_failure_by_split=" + json.dumps(interpretation["thin_failure_by_split"], sort_keys=True),
        "comparison=" + json.dumps(interpretation["comparison"], sort_keys=True, default=_json_default),
        "caveat=" + interpretation["caveat"],
        "",
        "outputs=" + json.dumps(summary.get("outputs", {}), sort_keys=True),
        "===== DINOV3_OBJECT_GEOMETRY_PRIOR_END =====",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def _save_visualizations(inputs: ClipInputs, output_dir: Path) -> None:
    """Write fixed first/middle/last-frame six-panel contact sheets."""

    output_dir.mkdir(parents=True, exist_ok=True)
    sequence = int(inputs.images.shape[0])
    frame_indices = tuple(sorted({0, sequence // 2, max(0, sequence - 1)}))
    union = inputs.regions["all"]
    for frame in frame_indices:
        rgb = _image_from_tensor(inputs.images[frame])
        mask_panel = _mask_image(union[frame])
        error_panel = _heatmap_image(
            inputs.error[frame],
            inputs.finite[frame] & inputs.regions["all"][frame],
        )
        dino_panel = _heatmap_image(
            inputs.gradients["dino_final"][frame],
            torch.isfinite(inputs.gradients["dino_final"][frame]),
        )
        stream_panel = _heatmap_image(
            inputs.gradients["streamvggt"][frame],
            torch.isfinite(inputs.gradients["streamvggt"][frame]),
        )
        band_panel = _band_image(inputs, frame)
        panels = [
            ("RGB", rgb),
            ("GT object union", mask_panel),
            ("raw geometry error", error_panel),
            ("DINO final gradient", dino_panel),
            ("StreamVGGT gradient", stream_panel),
            ("boundary bands", band_panel),
        ]
        sheet = _contact_sheet(panels)
        path = output_dir / f"{inputs.data.name}_frame_{frame:03d}.png"
        sheet.save(path)


def _image_from_tensor(value: torch.Tensor) -> Image.Image:
    tensor = torch.as_tensor(value).float().cpu()
    if tensor.ndim == 3 and tensor.shape[0] not in {1, 3} and tensor.shape[-1] in {1, 3}:
        tensor = tensor.permute(2, 0, 1)
    if tensor.ndim != 3:
        raise ValueError(f"RGB visualization tensor must be [3,H,W], got {tensor.shape}")
    if float(tensor.max()) > 1.5:
        tensor = tensor / 255.0
    array = (tensor[:3].clamp(0.0, 1.0).permute(1, 2, 0) * 255.0).byte().numpy()
    return Image.fromarray(array, mode="RGB")


def _mask_image(mask: torch.Tensor) -> Image.Image:
    value = torch.as_tensor(mask).bool().cpu()
    array = value.numpy().astype(np.uint8) * 255
    return Image.fromarray(array, mode="L").convert("RGB")


def _heatmap_image(values: torch.Tensor, valid: torch.Tensor) -> Image.Image:
    value = torch.as_tensor(values).float().cpu()
    mask = torch.as_tensor(valid).bool().cpu() & torch.isfinite(value)
    finite = value[mask]
    if finite.numel():
        upper = _quantile(finite, 0.95)
        lower = _quantile(finite, 0.05)
        if not math.isfinite(upper) or upper <= lower:
            lower = float(finite.min())
            upper = float(finite.max())
    else:
        lower, upper = 0.0, 1.0
    normalized = ((value - lower) / max(upper - lower, 1e-8)).clamp(0.0, 1.0)
    red = normalized * 255.0
    green = (1.0 - (normalized - 0.5).abs() * 2.0).clamp(0.0, 1.0) * 220.0
    blue = (1.0 - normalized) * 255.0
    array = torch.stack((red, green, blue), dim=-1).byte().numpy()
    array[~mask.numpy()] = 0
    return Image.fromarray(array, mode="RGB")


def _band_image(inputs: ClipInputs, frame: int) -> Image.Image:
    height, width = inputs.data.image_size
    array = np.zeros((height, width, 3), dtype=np.uint8)
    interior = inputs.regions["interior"][frame].numpy()
    near = inputs.regions["near_boundary"][frame].numpy()
    boundary = inputs.regions["boundary"][frame].numpy()
    array[interior] = (35, 150, 70)
    array[near] = (240, 190, 20)
    array[boundary] = (220, 35, 35)
    return Image.fromarray(array, mode="RGB")


def _contact_sheet(panels: Sequence[tuple[str, Image.Image]]) -> Image.Image:
    panel_width = 360
    resized: list[tuple[str, Image.Image]] = []
    for label, image in panels:
        ratio = panel_width / max(1, image.width)
        panel = image.resize(
            (panel_width, max(1, int(round(image.height * ratio)))),
            Image.Resampling.BILINEAR,
        ).convert("RGB")
        draw = ImageDraw.Draw(panel)
        draw.rectangle((0, 0, min(panel.width, 230), 24), fill=(0, 0, 0))
        draw.text((6, 5), label, fill=(255, 255, 255))
        resized.append((label, panel))
    panel_height = max(image.height for _, image in resized)
    sheet = Image.new("RGB", (panel_width * 3, panel_height * 2), (20, 20, 20))
    for index, (_, image) in enumerate(resized):
        x = (index % 3) * panel_width
        y = (index // 3) * panel_height
        sheet.paste(image, (x, y))
    return sheet


def _assign_bin(
    value: float,
    quantiles: Sequence[float],
    *,
    low_label: str,
    high_label: str,
) -> str:
    if len(quantiles) != 2 or not math.isfinite(value):
        return "medium"
    if value <= float(quantiles[0]):
        return low_label
    if value <= float(quantiles[1]):
        return "medium"
    return high_label


def _quantile(values: torch.Tensor, quantile: float) -> float:
    finite = torch.as_tensor(values).float().reshape(-1)
    finite = finite[torch.isfinite(finite)]
    if not finite.numel():
        return float("nan")
    return float(torch.quantile(finite, float(quantile)))


def _quantile_bin_mask(values: torch.Tensor, quantiles: Sequence[float], index: int) -> torch.Tensor:
    if index == 0:
        return values <= float(quantiles[0])
    if index == len(quantiles):
        return values > float(quantiles[-1])
    return (values > float(quantiles[index - 1])) & (values <= float(quantiles[index]))


def _add_error_stats(row: dict[str, Any], values: torch.Tensor, *, prefix: str = "") -> None:
    finite = torch.as_tensor(values).float().reshape(-1)
    finite = finite[torch.isfinite(finite)]
    row[f"{prefix}count"] = int(finite.numel())
    if not finite.numel():
        row[f"{prefix}mean_m"] = float("nan")
        row[f"{prefix}rmse_m"] = float("nan")
        row[f"{prefix}median_m"] = float("nan")
        row[f"{prefix}p90_m"] = float("nan")
        return
    row[f"{prefix}mean_m"] = float(finite.mean())
    row[f"{prefix}rmse_m"] = float(torch.sqrt(finite.square().mean()))
    row[f"{prefix}median_m"] = float(torch.quantile(finite, 0.50))
    row[f"{prefix}p90_m"] = float(torch.quantile(finite, 0.90))


def _mean(values: torch.Tensor) -> float:
    finite = torch.as_tensor(values).float()
    finite = finite[torch.isfinite(finite)]
    return float(finite.mean()) if finite.numel() else float("nan")


def _median_list(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(torch.tensor(list(values), dtype=torch.float32).median())


def _ratio(numerator: Any, denominator: Any) -> float:
    try:
        numerator = float(numerator)
        denominator = float(denominator)
    except (TypeError, ValueError):
        return float("nan")
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator <= 0.0:
        return float("nan")
    return numerator / denominator


def _fraction(numerator: Any, denominator: Any) -> float:
    try:
        numerator = float(numerator)
        denominator = float(denominator)
    except (TypeError, ValueError):
        return float("nan")
    if denominator <= 0.0:
        return float("nan")
    return numerator / denominator


def _spearman(first: torch.Tensor, second: torch.Tensor) -> float:
    x = torch.as_tensor(first).float().reshape(-1)
    y = torch.as_tensor(second).float().reshape(-1)
    valid = torch.isfinite(x) & torch.isfinite(y)
    x, y = x[valid], y[valid]
    if x.numel() < 2 or float(x.std()) == 0.0 or float(y.std()) == 0.0:
        return float("nan")
    rx = _average_ranks(x)
    ry = _average_ranks(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denominator = torch.sqrt(rx.square().sum() * ry.square().sum())
    return float((rx * ry).sum() / denominator) if float(denominator) > 0.0 else float("nan")


def _average_ranks(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values, stable=True)
    sorted_values = values[order]
    if sorted_values.numel() == 0:
        return torch.empty_like(values)
    group_start = torch.ones(
        sorted_values.shape,
        dtype=torch.bool,
        device=sorted_values.device,
    )
    if sorted_values.numel() > 1:
        group_start[1:] = sorted_values[1:] != sorted_values[:-1]
    group_ids = torch.cumsum(group_start.to(torch.long), dim=0) - 1
    group_count = torch.bincount(group_ids)
    positions = torch.arange(
        1,
        int(sorted_values.numel()) + 1,
        dtype=sorted_values.dtype,
        device=sorted_values.device,
    )
    group_sum = torch.bincount(group_ids, weights=positions)
    ranks_sorted = group_sum[group_ids] / group_count[group_ids].to(positions.dtype)
    ranks = torch.empty_like(ranks_sorted)
    ranks[order] = ranks_sorted
    return ranks


def _subsample_coordinates(coordinates: torch.Tensor, limit: int) -> torch.Tensor:
    if coordinates.shape[0] <= int(limit):
        return coordinates
    indices = torch.linspace(
        0,
        coordinates.shape[0] - 1,
        steps=max(1, int(limit)),
    ).round().long().unique()
    return coordinates.index_select(0, indices)


def _resolve_path(value: str | None, root: Path, default: str) -> Path:
    path = Path(value).expanduser() if value else root / default
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol")
    parser.add_argument("--feature-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--confidence-threshold", type=float, default=0.30)
    parser.add_argument("--high-confidence-threshold", type=float, default=0.70)
    parser.add_argument("--max-points-per-object-frame", type=int, default=256)
    return parser.parse_args()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf8")
        return
    clean_rows = [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in rows
    ]
    fields: list[str] = []
    for row in clean_rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(clean_rows)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot JSON encode {type(value)!r}")


if __name__ == "__main__":
    main()
