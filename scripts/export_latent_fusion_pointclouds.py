#!/usr/bin/env python3
"""Export GT, StreamVGGT, and latent-fusion pointmaps as PLY point clouds."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_latent_fusion import build_train_config, load_config
from vggtsam.adapters.sam3_intermediate import (
    SAM3IntermediateAdapter,
    load_sam3_image_model,
)
from vggtsam.adapters.streamvggt_latent import (
    StreamVGGTLatentAdapter,
    load_streamvggt_latent_model,
)
from vggtsam.data.scannetpp.object_sequence import (
    ObjectSamplingConfig,
    ObjectSequence,
    ScanNetPPObjectSequenceDataset,
)
from vggtsam.models.latent_fusion import LatentSAMVGGTModel
from vggtsam.training.latent_fusion import (
    build_latent_batch,
    majority_pool_mask,
    pool_pointmaps_to_grid,
    resolve_point_targets,
    select_object_prompt,
    should_request_streamvggt_pointmap,
    slice_camera_tokens,
    split_sequence_tokens,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/latent_fusion_train.yaml"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint to load. Defaults to the latest ckpt_step*.pt in training.output_dir.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--sequence-index", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--max-points", type=int, default=200_000)
    parser.add_argument("--max-abs-coordinate", type=float, default=1e5)
    parser.add_argument(
        "--export-all-tokens",
        action="store_true",
        help="Also export all finite tokens. By default object-supervision tokens are exported.",
    )
    args = parser.parse_args()

    raw = load_config(args.config)
    if args.device is not None:
        raw["training"]["device"] = args.device
    if args.output_dir is not None:
        raw["training"]["output_dir"] = str(args.output_dir)
    if args.seed is not None:
        raw["training"]["seed"] = args.seed
    if args.prompt is not None:
        raw["sam3"]["prompt"] = args.prompt
        raw["sam3"]["prompt_mode"] = "fixed"
        if args.prompt.strip().lower() != "object":
            raw["objects"]["target_object_labels"] = [args.prompt]
    config = build_train_config(raw)

    output_dir = config.output_dir / "pointclouds"
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(config.seed)
    dataset = build_dataset(config)
    sequence = (
        dataset[args.sequence_index]
        if args.sequence_index is not None
        else dataset.sample(rng)
    )
    prompt_selection = select_object_prompt(
        sequence.visible_instance_ids,
        sequence.object_labels,
        rng=rng,
        min_visible_frames=config.min_visible_frames,
        mode=config.sam3_prompt_mode,
        fallback_prompt=config.sam3_prompt,
        target_object_labels=config.target_object_labels,
        excluded_object_labels=config.excluded_object_labels,
    )
    if prompt_selection is None:
        raise RuntimeError("No valid prompt candidate found for the selected clip.")

    print(
        f"scene={sequence.scene_id} frames={sequence.frame_indices} "
        f"prompt={prompt_selection.prompt!r} instance={prompt_selection.sampled_instance_id}"
    )

    sam3_model = load_sam3_image_model(
        repo_path=config.sam3_repo,
        checkpoint_path=config.sam3_checkpoint,
        device=config.device,
        enable_inst_interactivity=config.sam3_enable_inst_interactivity,
    )
    sam3_model.requires_grad_(False)
    sam3 = SAM3IntermediateAdapter(
        sam3_model,
        device=config.device,
        resolution=config.sam3_resolution,
        source=config.sam3_feature_source,
        text_conditioning=config.sam3_text_conditioning,
        token_grid=config.token_grid,
    )

    streamvggt_model = load_streamvggt_latent_model(
        repo_path=config.streamvggt_repo,
        checkpoint_path=config.streamvggt_checkpoint,
        device=config.device,
        strict=True,
    )
    streamvggt_model.requires_grad_(False)
    geometry = StreamVGGTLatentAdapter(
        streamvggt_model,
        device=config.device,
        token_grid=config.token_grid,
        context_grid=config.context_grid,
        layer_index=config.streamvggt_layer_index,
        image_mode=config.streamvggt_image_mode,
    )

    with torch.no_grad():
        sam_out = sam3.extract_from_paths(
            sequence.image_paths,
            prompt=prompt_selection.prompt,
        )
        geo_out = geometry.extract_from_paths(
            sequence.image_paths,
            return_pointmap=True,
        )
        pointmap_grid = resolve_point_targets(
            sequence.pointmaps,
            geo_out.pointmap_grid,
            token_grid=config.token_grid,
            source=config.point_target_source,
        )

    batch = build_latent_batch(
        sequence.instance_masks,
        sequence.semantic_masks,
        sequence.visible_instance_ids,
        sequence.object_labels,
        pointmap_grid=pointmap_grid,
        token_grid=config.token_grid,
        min_visible_frames=config.min_visible_frames,
        ignore_instance_id=config.ignore_instance_id,
        semantic_ignore_label=config.semantic_ignore_label,
        excluded_semantic_labels=config.excluded_semantic_labels,
        target_object_labels=prompt_selection.target_object_labels,
        excluded_object_labels=config.excluded_object_labels,
        min_token_majority=config.min_token_majority,
        min_tokens_per_instance=config.min_tokens_per_instance,
        max_area_ratio=config.max_area_ratio,
        num_classes=config.num_classes,
        device=config.device,
    )
    if batch is None:
        raise RuntimeError("Selected clip has no valid object tokens after filtering.")

    token_rgb = pool_rgb_to_grid(sequence.image_paths, config.token_grid)
    gt_semantic = pool_label_sequence(sequence.semantic_masks, config.token_grid)
    gt_instance = pool_label_sequence(sequence.instance_masks, config.token_grid)
    object_mask = batch["mask_supervision_tokens"].detach().cpu().numpy().astype(bool)
    valid_mask = batch["valid_tokens"].detach().cpu().numpy().astype(bool)

    written: Dict[str, int] = {}
    gt_grid = (
        pool_pointmaps_to_grid(sequence.pointmaps, config.token_grid)
        if sequence.pointmaps is not None
        else None
    )
    if gt_grid is not None:
        written.update(
            export_pointmap_group(
                output_dir,
                prefix="gt",
                points=gt_grid,
                rgb=token_rgb,
                semantic_labels=gt_semantic,
                instance_labels=gt_instance,
                object_mask=object_mask,
                all_tokens=args.export_all_tokens,
                max_points=args.max_points,
                max_abs_coordinate=args.max_abs_coordinate,
            )
        )
    if geo_out.pointmap_grid is not None:
        written.update(
            export_pointmap_group(
                output_dir,
                prefix="streamvggt",
                points=geo_out.pointmap_grid.detach().cpu().numpy(),
                rgb=token_rgb,
                semantic_labels=gt_semantic,
                instance_labels=gt_instance,
                object_mask=object_mask,
                all_tokens=args.export_all_tokens,
                max_points=args.max_points,
                max_abs_coordinate=args.max_abs_coordinate,
            )
        )

    checkpoint = args.checkpoint or find_latest_checkpoint(config.output_dir)
    if checkpoint is None:
        print("No checkpoint found; exported GT/StreamVGGT point clouds only.")
    else:
        pred_pointmap, pred_logits = run_fusion_prediction(
            checkpoint,
            config,
            sam_tokens=sam_out.semantic.tokens.float(),
            geometry_tokens=geo_out.geometry.tokens.float(),
            camera_tokens=(
                geo_out.geometry.camera_tokens.float()
                if config.use_camera_tokens and geo_out.geometry.camera_tokens is not None
                else None
            ),
            num_frames=len(sequence.image_paths),
        )
        pred_semantic = pred_logits.argmax(dim=-1).cpu().numpy().reshape(
            len(sequence.image_paths),
            *config.token_grid,
        )
        written.update(
            export_pointmap_group(
                output_dir,
                prefix="pred",
                points=pred_pointmap.cpu().numpy().reshape(
                    len(sequence.image_paths),
                    *config.token_grid,
                    3,
                ),
                rgb=token_rgb,
                semantic_labels=pred_semantic,
                instance_labels=gt_instance,
                object_mask=object_mask,
                all_tokens=args.export_all_tokens,
                max_points=args.max_points,
                max_abs_coordinate=args.max_abs_coordinate,
            )
        )
        print(f"loaded checkpoint: {checkpoint}")

    summary = {
        "scene_id": sequence.scene_id,
        "frame_indices": sequence.frame_indices,
        "image_paths": [str(path) for path in sequence.image_paths],
        "prompt": prompt_selection.prompt,
        "sampled_instance_id": prompt_selection.sampled_instance_id,
        "sampled_label": prompt_selection.sampled_label,
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "pointcloud_counts": written,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf8")
    print(f"point clouds saved to: {output_dir}")
    print(f"summary: {summary_path}")


def build_dataset(config) -> ScanNetPPObjectSequenceDataset:
    object_config = ObjectSamplingConfig(
        min_pixels=config.min_pixels,
        max_area_ratio=config.max_area_ratio,
        min_visible_frames=config.min_visible_frames,
        max_objects_per_frame=config.max_objects_per_frame,
        ignore_instance_id=config.ignore_instance_id,
        semantic_ignore_label=config.semantic_ignore_label,
    )
    return ScanNetPPObjectSequenceDataset(
        config.manifest,
        scene_id=config.scene_id,
        sequence_length=config.sequence_length,
        frame_stride=config.frame_stride,
        object_config=object_config,
    )


def run_fusion_prediction(
    checkpoint_path: Path,
    config,
    *,
    sam_tokens: torch.Tensor,
    geometry_tokens: torch.Tensor,
    camera_tokens: torch.Tensor | None,
    num_frames: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens_per_frame = config.token_grid[0] * config.token_grid[1]
    context_tokens_per_frame = config.context_grid[0] * config.context_grid[1]
    sam_frame_tokens = split_sequence_tokens(
        sam_tokens,
        num_frames=num_frames,
        tokens_per_frame=tokens_per_frame,
        name="sam_tokens",
    )
    geometry_frame_tokens = split_sequence_tokens(
        geometry_tokens,
        num_frames=num_frames,
        tokens_per_frame=context_tokens_per_frame,
        name="geometry_tokens",
    )
    camera_dim = (
        int(camera_tokens.shape[-1])
        if config.use_camera_tokens and camera_tokens is not None
        else None
    )
    model = LatentSAMVGGTModel(
        sam_dim=int(sam_tokens.shape[-1]),
        geometry_dim=int(geometry_tokens.shape[-1]),
        camera_dim=camera_dim,
        d_fuse=config.d_fuse,
        num_heads=config.num_heads,
        num_classes=config.num_classes,
        dropout=config.dropout,
        token_grid=config.token_grid,
    ).to(config.device)

    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    state_dict = checkpoint.get("model", checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"checkpoint missing keys: {missing}")
    if unexpected:
        print(f"checkpoint unexpected keys: {unexpected}")
    model.eval()

    pred_points = []
    pred_logits = []
    with torch.no_grad():
        for frame_idx in range(num_frames):
            output = model(
                sam_tokens=sam_frame_tokens[frame_idx].to(config.device),
                geometry_tokens=geometry_frame_tokens[frame_idx].to(config.device),
                camera_tokens=slice_camera_tokens(
                    camera_tokens.to(config.device) if camera_tokens is not None else None,
                    frame_idx=frame_idx,
                    num_frames=num_frames,
                ),
            )
            pred_points.append(output.pointmap[0].detach().cpu())
            pred_logits.append(output.logits[0].detach().cpu())
    return torch.stack(pred_points, dim=0), torch.stack(pred_logits, dim=0)


def find_latest_checkpoint(output_dir: Path) -> Path | None:
    candidates = sorted(output_dir.glob("ckpt_step*.pt"))
    if not candidates:
        return None

    def step(path: Path) -> int:
        match = re.search(r"ckpt_step(\d+)\.pt$", path.name)
        return int(match.group(1)) if match else -1

    return max(candidates, key=step)


def pool_rgb_to_grid(
    image_paths: Sequence[Path],
    out_hw: tuple[int, int],
) -> np.ndarray:
    out_h, out_w = out_hw
    frames = []
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        image = image.resize((out_w, out_h), Image.BILINEAR)
        frames.append(np.asarray(image, dtype=np.uint8))
    return np.stack(frames, axis=0)


def pool_label_sequence(
    masks: Sequence[np.ndarray],
    out_hw: tuple[int, int],
) -> np.ndarray:
    labels = [majority_pool_mask(mask, out_hw)[0] for mask in masks]
    return np.stack(labels, axis=0)


def export_pointmap_group(
    output_dir: Path,
    *,
    prefix: str,
    points: np.ndarray,
    rgb: np.ndarray,
    semantic_labels: np.ndarray,
    instance_labels: np.ndarray,
    object_mask: np.ndarray,
    all_tokens: bool,
    max_points: int,
    max_abs_coordinate: float,
) -> Dict[str, int]:
    flat_points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    flat_rgb = np.asarray(rgb, dtype=np.uint8).reshape(-1, 3)
    flat_semantic = np.asarray(semantic_labels).reshape(-1)
    flat_instance = np.asarray(instance_labels).reshape(-1)
    finite = np.isfinite(flat_points).all(axis=1)
    finite &= np.abs(flat_points).max(axis=1) <= float(max_abs_coordinate)
    object_mask = np.asarray(object_mask, dtype=bool).reshape(-1) & finite
    counts: Dict[str, int] = {}

    exports = {
        f"{prefix}_objects_rgb.ply": (object_mask, flat_rgb),
        f"{prefix}_objects_semantic.ply": (
            object_mask,
            palette_from_labels(flat_semantic),
        ),
        f"{prefix}_objects_instance.ply": (
            object_mask,
            palette_from_labels(flat_instance),
        ),
    }
    if all_tokens:
        exports.update(
            {
                f"{prefix}_all_rgb.ply": (finite, flat_rgb),
                f"{prefix}_all_semantic.ply": (
                    finite,
                    palette_from_labels(flat_semantic),
                ),
            }
        )

    for filename, (mask, colors) in exports.items():
        path = output_dir / filename
        count = write_ply(
            path,
            flat_points[mask],
            colors[mask],
            max_points=max_points,
        )
        counts[filename] = count
    return counts


def palette_from_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    colors = np.zeros((labels.shape[0], 3), dtype=np.uint8)
    valid = labels >= 0
    values = labels[valid].astype(np.uint32)
    colors[valid, 0] = ((values * 37 + 17) % 255).astype(np.uint8)
    colors[valid, 1] = ((values * 67 + 71) % 255).astype(np.uint8)
    colors[valid, 2] = ((values * 97 + 131) % 255).astype(np.uint8)
    colors[~valid] = np.array([40, 40, 40], dtype=np.uint8)
    return colors


def write_ply(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    *,
    max_points: int,
) -> int:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    if points.shape[0] != colors.shape[0]:
        raise ValueError(
            f"points/colors length mismatch: {points.shape[0]} vs {colors.shape[0]}"
        )
    if points.shape[0] > max_points:
        indices = np.linspace(0, points.shape[0] - 1, int(max_points), dtype=np.int64)
        points = points[indices]
        colors = colors[indices]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {points.shape[0]}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write("end_header\n")
        for point, color in zip(points, colors):
            handle.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )
    return int(points.shape[0])


if __name__ == "__main__":
    main()
