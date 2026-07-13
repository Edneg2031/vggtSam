#!/usr/bin/env python3
"""Analyze how SAM3 encoder/memory identity signals relate to StreamVGGT tokens."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import random
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import yaml

from test_sam.coordinates import resize_label_map, streamvggt_label_to_grid
from test_sam.data import load_mask_tracking_sequence
from test_sam.fusion import stream_tokens_to_maps
from test_sam.train_fusion_ablation import extract_sam_features, resize_target_masks
from vggtsam.adapters.streamvggt_latent import (
    StreamVGGTLatentAdapter,
    load_streamvggt_latent_model,
)
from vggtsam.training.dense_fusion import run_sam3_source_tracker_flow


@dataclass(frozen=True)
class Config:
    manifest: Path
    scene_id: str
    frame_indices: list[int]
    instance_id: int
    query_sequence_index: int
    sam3_repo: Path
    sam3_checkpoint: Path
    sam3_feature_device: str
    tracker_device: str
    sam3_resolution: int
    sam3_frame_chunk_size: int
    reference_prompt_mode: str
    streamvggt_repo: Path
    streamvggt_checkpoint: Path
    geometry_device: str
    geometry_streaming_cache: bool
    geometry_image_mode: str
    layer_indices: tuple[int, ...]
    output_size: tuple[int, int]
    topk: int
    output_dir: Path
    seed: int


def main() -> None:
    args = parse_args()
    raw = load_yaml(args.config)
    apply_overrides(raw, args)
    run(build_config(raw))


def run(config: Config) -> None:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(config.output_dir / "resolved_config.json", config_to_dict(config))

    sequence = load_mask_tracking_sequence(
        config.manifest,
        scene_id=config.scene_id,
        frame_indices=config.frame_indices,
        sequence_length=len(config.frame_indices),
        frame_stride=1,
        window_index=0,
        instance_id=config.instance_id,
        min_pixels=1,
        max_area_ratio=1.0,
        min_visible_frames=1,
        excluded_labels=[],
        seed=config.seed,
    )
    if not 0 <= config.query_sequence_index < len(sequence.frame_indices):
        raise ValueError("query_sequence_index is outside the selected sequence.")
    sequence = replace(sequence, reference_frame_idx=config.query_sequence_index)
    if not bool(sequence.target_masks[config.query_sequence_index].any()):
        raise RuntimeError("The configured query frame does not contain the target instance.")
    output_masks = resize_target_masks(sequence, config.output_size)

    sam_features, sam_input_images, sam_tracker = extract_sam_features(config, sequence)
    actual_logits, actual_aux = run_source_flow(
        config,
        sequence,
        output_masks,
        sam_features,
        sam_input_images,
        sam_tracker,
        oracle_visibility=False,
    )
    _, oracle_aux = run_source_flow(
        config,
        sequence,
        output_masks,
        sam_features,
        sam_input_images,
        sam_tracker,
        oracle_visibility=True,
    )
    stream_features, stream_meta = extract_stream_features(config, sequence.image_paths)

    feature_maps: dict[str, tuple[torch.Tensor, str]] = {
        "sam_encoder_fpn2": (sam_features["fpn2"].float().cpu(), "sam"),
    }
    actual_memory = actual_aux.get("spatial_memories")
    if actual_memory is not None:
        feature_maps["sam_spatial_memory_actual"] = (
            actual_memory.detach().float().cpu(),
            "sam",
        )
    oracle_memory = oracle_aux.get("spatial_memories")
    if oracle_memory is not None:
        feature_maps["sam_spatial_memory_oracle_visibility"] = (
            oracle_memory.detach().float().cpu(),
            "sam",
        )
    feature_maps.update(
        {name: (value, "stream") for name, value in stream_features.items()}
    )

    rows = []
    summary_rows = []
    similarity_maps: dict[str, list[np.ndarray]] = {}
    for feature_name, (features, coordinate_space) in feature_maps.items():
        validate_feature_map(features, frames=len(sequence.frame_indices), name=feature_name)
        label_grids = build_label_grids(
            sequence.instance_masks,
            features.shape[-2:],
            coordinate_space=coordinate_space,
            stream_mode=config.geometry_image_mode,
        )
        feature_rows, maps = evaluate_feature(
            feature_name,
            features,
            label_grids,
            target_instance=config.instance_id,
            query_index=config.query_sequence_index,
            frame_indices=sequence.frame_indices,
            topk=config.topk,
        )
        rows.extend(feature_rows)
        similarity_maps[feature_name] = maps
        summary_rows.append(
            summarize_feature(
                feature_name,
                features,
                feature_rows,
                query_index=config.query_sequence_index,
            )
        )
        save_feature_visualization(
            config.output_dir / "similarity_maps" / f"{feature_name}.png",
            feature_name=feature_name,
            image_paths=sequence.image_paths,
            instance_masks=sequence.instance_masks,
            target_instance=config.instance_id,
            similarity_maps=maps,
            frame_indices=sequence.frame_indices,
            output_size=config.output_size,
        )

    write_csv(config.output_dir / "feature_frame_metrics.csv", rows)
    write_csv(config.output_dir / "feature_summary.csv", summary_rows)
    print_feature_summary(summary_rows)
    pointer_rows = analyze_object_pointers(
        sequence.frame_indices,
        output_masks,
        actual_aux,
        oracle_aux,
        query_index=config.query_sequence_index,
    )
    write_csv(config.output_dir / "object_pointer_metrics.csv", pointer_rows)
    save_pointer_visualization(
        config.output_dir / "object_pointer_cosine.png",
        actual_aux.get("object_ptrs"),
        oracle_aux.get("object_ptrs"),
        sequence.frame_indices,
    )
    save_tracker_visualization(
        config.output_dir / "sam3_tracker_masks.png",
        sequence.image_paths,
        output_masks,
        actual_logits,
        actual_aux["object_score_logits"],
        config.output_size,
    )
    write_json(
        config.output_dir / "tensor_shapes.json",
        {
            "sam_features": {key: list(value.shape) for key, value in sam_features.items()},
            "actual_object_ptrs": optional_shape(actual_aux.get("object_ptrs")),
            "actual_spatial_memories": optional_shape(actual_aux.get("spatial_memories")),
            "oracle_object_ptrs": optional_shape(oracle_aux.get("object_ptrs")),
            "oracle_spatial_memories": optional_shape(oracle_aux.get("spatial_memories")),
            "stream_features": {key: list(value.shape) for key, value in stream_features.items()},
            "stream_meta": stream_meta,
        },
    )
    print(f"feature metrics: {config.output_dir / 'feature_frame_metrics.csv'}")
    print(f"feature summary: {config.output_dir / 'feature_summary.csv'}")
    print(f"object pointers: {config.output_dir / 'object_pointer_metrics.csv'}")


def run_source_flow(
    config,
    sequence,
    masks,
    sam_features,
    sam_input_images,
    sam_tracker,
    *,
    oracle_visibility,
):
    residuals = [
        torch.zeros_like(sam_features[f"fpn{level}"], dtype=torch.float32)
        for level in range(3)
    ]
    with torch.no_grad():
        logits, aux = run_sam3_source_tracker_flow(
            sam_tracker=sam_tracker,
            sam_tracker_features=sam_features,
            sam_input_images=sam_input_images,
            tracker_residuals=residuals,
            reference_mask=masks[config.query_sequence_index],
            reference_frame_idx=config.query_sequence_index,
            output_size=config.output_size,
            residual_scale=0.0,
            device=config.tracker_device,
            reference_prompt_mode=config.reference_prompt_mode,
            training_target_masks=(
                masks.to(config.tracker_device) if oracle_visibility else None
            ),
        )
    return logits.detach().float().cpu(), detach_aux(aux)


def extract_stream_features(config, image_paths):
    model = load_streamvggt_latent_model(
        repo_path=config.streamvggt_repo,
        checkpoint_path=config.streamvggt_checkpoint,
        device=config.geometry_device,
        strict=True,
    )
    model.requires_grad_(False)
    adapter = StreamVGGTLatentAdapter(
        model,
        device=config.geometry_device,
        token_grid=(72, 72),
        context_grid=(12, 12),
        layer_index=-1,
        dpt_layer_indices=config.layer_indices,
        image_mode=config.geometry_image_mode,
    )
    with torch.no_grad():
        output = adapter.extract_from_paths(
            image_paths,
            return_pointmap=False,
            streaming_cache=config.geometry_streaming_cache,
        )
    patch_shape = tuple(int(value) for value in output.aux["patch_shape"])
    levels = stream_tokens_to_maps(
        output.geometry.aux["stream_dpt_tokens"],
        patch_start_idx=int(output.geometry.aux["patch_start_idx"]),
        patch_shape=patch_shape,
        output_grid=patch_shape,
    )
    features = {
        f"stream_aggregator_layer_{layer_index}": value.detach().float().cpu()
        for layer_index, value in zip(config.layer_indices, levels)
    }
    meta = {
        "image_shape": list(output.aux["image_shape"]),
        "patch_shape": list(patch_shape),
        "patch_start_idx": int(output.geometry.aux["patch_start_idx"]),
        "streaming_cache": config.geometry_streaming_cache,
    }
    del output, adapter, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return features, meta


def build_label_grids(instance_masks, grid_size, *, coordinate_space, stream_mode):
    if coordinate_space == "stream":
        return [
            streamvggt_label_to_grid(mask, grid_size, mode=stream_mode)
            for mask in instance_masks
        ]
    if coordinate_space == "sam":
        return [resize_label_map(mask, grid_size) for mask in instance_masks]
    raise ValueError(f"Unknown coordinate space {coordinate_space!r}")


def evaluate_feature(feature_name, features, label_grids, *, target_instance, query_index, frame_indices, topk):
    normalized = F.normalize(features.float(), dim=1, eps=1e-6)
    query_mask = torch.from_numpy(label_grids[query_index] == target_instance)
    if not bool(query_mask.any()):
        raise RuntimeError(f"Target vanished from query grid for {feature_name}.")
    query_tokens = normalized[query_index].permute(1, 2, 0)[query_mask]
    prototype = F.normalize(query_tokens.mean(dim=0), dim=0, eps=1e-6)
    rows, maps = [], []
    for index, frame_idx in enumerate(frame_indices):
        similarity = torch.einsum("chw,c->hw", normalized[index], prototype)
        maps.append(similarity.cpu().numpy())
        labels = torch.from_numpy(label_grids[index].astype(np.int64))
        target = labels == int(target_instance)
        flat = similarity.flatten()
        k = min(int(topk), flat.numel())
        top_indices = torch.topk(flat, k=k).indices
        top_labels = labels.flatten()[top_indices]
        top1_index = int(top_indices[0])
        top_y, top_x = divmod(top1_index, similarity.shape[1])
        visible = bool(target.any())
        positive = similarity[target] if visible else similarity.new_empty(0)
        negative = similarity[~target] if visible else similarity.flatten()
        if visible:
            ys, xs = torch.where(target)
            centroid_x = xs.float().mean()
            centroid_y = ys.float().mean()
            anchor_error = float(
                torch.sqrt((centroid_x - top_x) ** 2 + (centroid_y - top_y) ** 2)
                / math.sqrt(similarity.shape[0] ** 2 + similarity.shape[1] ** 2)
            )
        else:
            anchor_error = float("nan")
        positive_mean = float(positive.mean()) if positive.numel() else float("nan")
        negative_mean = float(negative.mean()) if negative.numel() else float("nan")
        rows.append(
            {
                "feature_name": feature_name,
                "frame_index": frame_idx,
                "sequence_index": index,
                "visible": visible,
                "target_pixels_on_grid": int(target.sum()),
                "top1_inside_gt": bool(top_labels[0] == target_instance),
                "top5_inside_gt_rate": float((top_labels == target_instance).float().mean()),
                "top1_instance_id": int(top_labels[0]),
                "anchor_x": top_x,
                "anchor_y": top_y,
                "anchor_localization_error": anchor_error,
                "positive_cosine": positive_mean,
                "negative_cosine": negative_mean,
                "positive_negative_margin": positive_mean - negative_mean,
                "max_similarity": float(flat.max()),
            }
        )
    return rows, maps


def summarize_feature(name, features, rows, *, query_index):
    visible = [
        row
        for row in rows
        if row["visible"] and row["sequence_index"] != query_index
    ]
    values = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1]).float()
    sample = values[torch.linspace(0, values.shape[0] - 1, min(512, values.shape[0])).long()]
    effective_rank = approximate_effective_rank(sample)
    return {
        "feature_name": name,
        "feature_shape": list(features.shape),
        "feature_variance": float(values.var()),
        "effective_rank": effective_rank,
        "mean_top1_inside_gt": mean([float(row["top1_inside_gt"]) for row in visible]),
        "mean_top5_inside_gt_rate": mean([row["top5_inside_gt_rate"] for row in visible]),
        "mean_positive_cosine": mean([row["positive_cosine"] for row in visible]),
        "mean_negative_cosine": mean([row["negative_cosine"] for row in visible]),
        "mean_positive_negative_margin": mean([row["positive_negative_margin"] for row in visible]),
        "mean_anchor_localization_error": mean([row["anchor_localization_error"] for row in visible]),
    }


def analyze_object_pointers(frame_indices, masks, actual_aux, oracle_aux, *, query_index):
    rows = []
    for source, aux in (("actual", actual_aux), ("oracle_visibility", oracle_aux)):
        pointers = aux.get("object_ptrs")
        if pointers is None:
            continue
        normalized = F.normalize(pointers.float(), dim=-1, eps=1e-6)
        query = normalized[query_index]
        scores = aux["object_score_logits"].reshape(-1)
        for index, frame_idx in enumerate(frame_indices):
            rows.append(
                {
                    "source": source,
                    "frame_index": frame_idx,
                    "sequence_index": index,
                    "gt_visible": bool(masks[index].any()),
                    "object_score": float(scores[index]),
                    "pointer_norm": float(pointers[index].float().norm()),
                    "cosine_to_query_pointer": float(torch.dot(normalized[index], query)),
                }
            )
    return rows


def approximate_effective_rank(values):
    centered = values - values.mean(dim=0, keepdim=True)
    q = min(64, centered.shape[0] - 1, centered.shape[1])
    if q <= 1:
        return 1.0
    _, singular, _ = torch.pca_lowrank(centered, q=q, center=False)
    probability = singular.square()
    probability = probability / probability.sum().clamp_min(1e-12)
    entropy = -(probability * probability.clamp_min(1e-12).log()).sum()
    return float(entropy.exp())


def save_feature_visualization(path, *, feature_name, image_paths, instance_masks, target_instance, similarity_maps, frame_indices, output_size):
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(len(image_paths), 3, figsize=(12, 3.2 * len(image_paths)))
    for row, (image_path, labels, similarity, frame_idx) in enumerate(
        zip(image_paths, instance_masks, similarity_maps, frame_indices)
    ):
        with Image.open(image_path) as image:
            rgb = np.asarray(image.convert("RGB").resize((output_size[1], output_size[0])))
        gt = resize_label_map(labels, output_size) == target_instance
        similarity_tensor = torch.from_numpy(similarity)[None, None].float()
        similarity_up = F.interpolate(
            similarity_tensor,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )[0, 0].numpy()
        axes[row, 0].imshow(rgb)
        axes[row, 0].set_title(f"frame={frame_idx}")
        axes[row, 1].imshow(rgb)
        axes[row, 1].imshow(gt, alpha=0.5, cmap="Greens", vmin=0, vmax=1)
        axes[row, 1].set_title(f"GT instance={target_instance}")
        axes[row, 2].imshow(rgb)
        axes[row, 2].imshow(similarity_up, alpha=0.65, cmap="turbo", vmin=-1, vmax=1)
        axes[row, 2].set_title(f"similarity max={similarity.max():.3f}")
        for axis in axes[row]:
            axis.axis("off")
    figure.suptitle(feature_name)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_pointer_visualization(path, actual, oracle, frame_indices):
    matrices, names = [], []
    for name, pointers in (("actual", actual), ("oracle visibility", oracle)):
        if pointers is not None:
            normalized = F.normalize(pointers.float(), dim=-1, eps=1e-6)
            matrices.append((normalized @ normalized.T).cpu().numpy())
            names.append(name)
    if not matrices:
        return
    figure, axes = plt.subplots(1, len(matrices), figsize=(5 * len(matrices), 4))
    axes = np.atleast_1d(axes)
    for axis, matrix, name in zip(axes, matrices, names):
        image = axis.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
        axis.set_xticks(range(len(frame_indices)), frame_indices, rotation=45)
        axis.set_yticks(range(len(frame_indices)), frame_indices)
        axis.set_title(name)
        figure.colorbar(image, ax=axis, fraction=0.046)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_tracker_visualization(path, image_paths, masks, logits, scores, output_size):
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(len(image_paths), 3, figsize=(12, 3.2 * len(image_paths)))
    for index, image_path in enumerate(image_paths):
        with Image.open(image_path) as image:
            rgb = np.asarray(image.convert("RGB").resize((output_size[1], output_size[0])))
        gt = masks[index].numpy()
        pred = logits[index].sigmoid().numpy() >= 0.5
        axes[index, 0].imshow(rgb)
        axes[index, 0].set_title(f"RGB sequence_index={index}")
        axes[index, 1].imshow(rgb)
        axes[index, 1].imshow(gt, alpha=0.5, cmap="Greens")
        axes[index, 1].set_title(f"GT pixels={int(gt.sum())}")
        axes[index, 2].imshow(rgb)
        axes[index, 2].imshow(pred, alpha=0.5, cmap="Blues")
        axes[index, 2].set_title(
            f"SAM3 actual pixels={int(pred.sum())} score={float(scores[index]):.3f}"
        )
        for axis in axes[index]:
            axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def detach_aux(aux):
    return {
        key: value.detach().float().cpu() if torch.is_tensor(value) else value
        for key, value in aux.items()
    }


def validate_feature_map(value, *, frames, name):
    if value.ndim != 4 or value.shape[0] != frames:
        raise RuntimeError(f"{name} must be [T,C,H,W], got {tuple(value.shape)}")
    if not bool(torch.isfinite(value).all()):
        raise RuntimeError(f"{name} contains NaN or Inf.")


def optional_shape(value):
    return list(value.shape) if torch.is_tensor(value) else None


def mean(values):
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return sum(finite) / len(finite) if finite else float("nan")


def print_feature_summary(rows):
    print("\nCross-view identity summary (higher hit/margin, lower error is better):")
    for row in sorted(
        rows,
        key=lambda value: finite_or(value["mean_positive_negative_margin"], -math.inf),
        reverse=True,
    ):
        print(
            f"  {row['feature_name']}: "
            f"top1={row['mean_top1_inside_gt']:.3f} "
            f"top5={row['mean_top5_inside_gt_rate']:.3f} "
            f"margin={row['mean_positive_negative_margin']:.4f} "
            f"anchor_error={row['mean_anchor_localization_error']:.4f} "
            f"rank={row['effective_rank']:.1f}"
        )


def finite_or(value, fallback):
    value = float(value)
    return value if math.isfinite(value) else fallback


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as handle:
        json.dump(value, handle, indent=2)


def config_to_dict(config):
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in config.__dict__.items()
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("test_sam_for_streamvggt/config.yaml"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--frame-indices", type=int, nargs="+")
    parser.add_argument("--instance-id", type=int)
    parser.add_argument("--sam3-feature-device")
    parser.add_argument("--tracker-device")
    parser.add_argument("--geometry-device")
    return parser.parse_args()


def load_yaml(path):
    with path.open("r", encoding="utf8") as handle:
        return yaml.safe_load(handle)


def apply_overrides(raw, args):
    mappings = [
        ("output_dir", "analysis", "output_dir"),
        ("frame_indices", "dataset", "frame_indices"),
        ("instance_id", "dataset", "instance_id"),
        ("sam3_feature_device", "sam3", "feature_device"),
        ("tracker_device", "sam3", "tracker_device"),
        ("geometry_device", "streamvggt", "device"),
    ]
    for argument, section, key in mappings:
        value = getattr(args, argument)
        if value is not None:
            raw.setdefault(section, {})[key] = value


def build_config(raw):
    dataset, sam, stream, analysis = raw["dataset"], raw["sam3"], raw["streamvggt"], raw["analysis"]
    return Config(
        manifest=Path(dataset["manifest"]), scene_id=str(dataset["scene_id"]),
        frame_indices=[int(v) for v in dataset["frame_indices"]], instance_id=int(dataset["instance_id"]),
        query_sequence_index=int(dataset.get("query_sequence_index", 0)), sam3_repo=Path(sam["repo"]),
        sam3_checkpoint=Path(sam["checkpoint"]), sam3_feature_device=str(sam["feature_device"]),
        tracker_device=str(sam["tracker_device"]), sam3_resolution=int(sam.get("resolution", 1008)),
        sam3_frame_chunk_size=int(sam.get("frame_chunk_size", 1)),
        reference_prompt_mode=str(sam.get("reference_prompt_mode", "mask")),
        streamvggt_repo=Path(stream["repo"]), streamvggt_checkpoint=Path(stream["checkpoint"]),
        geometry_device=str(stream["device"]), geometry_streaming_cache=bool(stream.get("streaming_cache", True)),
        geometry_image_mode=str(stream.get("image_mode", "crop")),
        layer_indices=tuple(int(v) for v in stream.get("layer_indices", [4, 11, 17, 23])),
        output_size=tuple(int(v) for v in analysis.get("output_size", [256, 384])),
        topk=int(analysis.get("topk", 5)), output_dir=Path(analysis.get("output_dir", "outputs/sam_to_streamvggt_tokens")),
        seed=int(analysis.get("seed", 0)),
    )


if __name__ == "__main__":
    main()
