#!/usr/bin/env python3
"""Run SAM3's original video predictor on a processed ScanNet++ sequence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

from vggtsam.adapters.sam3_video import (
    SAM3VideoTrackerAdapter,
    binary_iou,
    load_sam3_video_predictor,
)
from vggtsam.data.scannetpp.object_sequence import (
    ObjectSamplingConfig,
    ScanNetPPObjectSequenceDataset,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/dense_fusion_train.yaml"),
    )
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--instance-id", type=int, required=True)
    parser.add_argument("--frame-indices", type=int, nargs="+", required=True)
    parser.add_argument("--output-size", type=int, nargs=2, metavar=("H", "W"), default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompt-frame", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--box", dest="prompt_with_box", action="store_true")
    parser.add_argument("--no-box", dest="prompt_with_box", action="store_false")
    parser.set_defaults(prompt_with_box=None)
    parser.add_argument("--async-loading-frames", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    raw = load_yaml(args.config)
    dataset_cfg = raw["dataset"]
    object_cfg = raw["objects"]
    sam3_cfg = raw["sam3"]
    training_cfg = raw["training"]

    output_size = (
        tuple(int(v) for v in args.output_size)
        if args.output_size is not None
        else tuple(int(v) for v in raw["model"]["output_size"])
    )
    device = str(
        args.device
        or sam3_cfg.get("direct_device")
        or sam3_cfg.get("device")
        or training_cfg["device"]
    )
    threshold = float(
        args.threshold
        if args.threshold is not None
        else sam3_cfg.get("direct_output_prob_thresh", 0.5)
    )
    prompt_with_box = (
        bool(args.prompt_with_box)
        if args.prompt_with_box is not None
        else bool(sam3_cfg.get("direct_prompt_with_box", True))
    )

    object_config = ObjectSamplingConfig(
        min_pixels=int(object_cfg["min_pixels"]),
        max_area_ratio=float(object_cfg["max_area_ratio"]),
        min_visible_frames=1,
        max_objects_per_frame=int(object_cfg["max_objects_per_frame"]),
        ignore_instance_id=int(object_cfg["ignore_instance_id"]),
        semantic_ignore_label=int(object_cfg["semantic_ignore_label"]),
    )
    dataset = ScanNetPPObjectSequenceDataset(
        dataset_cfg["manifest"],
        scene_id=args.scene_id,
        sequence_length=len(args.frame_indices),
        frame_indices=list(args.frame_indices),
        object_config=object_config,
    )
    sequence = dataset[0]
    label = sequence.object_labels.get(int(args.instance_id), "")
    prompt = args.prompt or label or "object"
    gt_masks = [
        resize_instance_mask(mask, args.instance_id, output_size)
        for mask in sequence.instance_masks
    ]
    visible = [idx for idx, mask in enumerate(gt_masks) if mask.any()]
    if not visible:
        raise RuntimeError(
            f"instance_id={args.instance_id} is not visible in frame_indices={args.frame_indices}"
        )
    prompt_frame = int(args.prompt_frame) if args.prompt_frame is not None else visible[0]
    if prompt_frame < 0 or prompt_frame >= len(sequence.image_paths):
        raise ValueError(
            f"prompt_frame={prompt_frame} is out of range for {len(sequence.image_paths)} frames"
        )
    if not gt_masks[prompt_frame].any():
        raise RuntimeError(
            f"instance_id={args.instance_id} is not visible at prompt_frame={prompt_frame}"
        )

    predictor = load_sam3_video_predictor(
        repo_path=sam3_cfg["repo"],
        checkpoint_path=sam3_cfg["checkpoint"],
        device=device,
        async_loading_frames=bool(args.async_loading_frames),
    )
    tracker = SAM3VideoTrackerAdapter(
        predictor,
        output_prob_thresh=threshold,
        prompt_with_box=prompt_with_box,
    )
    track = tracker.track_from_paths(
        sequence.image_paths,
        prompt=prompt,
        output_size=output_size,
        prompt_frame_idx=prompt_frame,
        reference_mask=gt_masks[prompt_frame],
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = write_outputs(
        args.output_dir,
        sequence=sequence,
        gt_masks=gt_masks,
        pred_masks=list(track.masks),
        prompt=prompt,
        instance_id=args.instance_id,
        prompt_frame=prompt_frame,
        selected_obj_id=track.selected_obj_id,
        prompt_box_xywh=track.prompt_box_xywh,
        output_size=output_size,
    )
    summary = {
        "scene_id": sequence.scene_id,
        "frame_indices": sequence.frame_indices,
        "instance_id": int(args.instance_id),
        "label": label,
        "prompt": prompt,
        "prompt_frame": int(prompt_frame),
        "prompt_with_box": bool(prompt_with_box),
        "prompt_box_xywh": track.prompt_box_xywh,
        "selected_obj_id": track.selected_obj_id,
        "threshold": threshold,
        "mean_iou_visible": mean_visible_iou(metrics),
        "output_dir": str(args.output_dir),
        "sam3_aux": track.aux,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf8",
    )
    print(
        "sam3_sequence "
        f"prompt={prompt!r} instance={args.instance_id} "
        f"prompt_frame={prompt_frame} selected_obj_id={track.selected_obj_id} "
        f"mean_iou_visible={summary['mean_iou_visible']:.4f} "
        f"out={args.output_dir}"
    )


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf8") as handle:
        return yaml.safe_load(handle) or {}


def resize_instance_mask(
    instance_mask: np.ndarray,
    instance_id: int,
    output_size: tuple[int, int],
) -> torch.Tensor:
    mask = np.asarray(instance_mask) == int(instance_id)
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    image = image.resize((int(output_size[1]), int(output_size[0])), Image.NEAREST)
    return torch.from_numpy(np.asarray(image) > 0)


def load_rgb(path: Path, output_size: tuple[int, int]) -> Image.Image:
    return Image.open(path).convert("RGB").resize(
        (int(output_size[1]), int(output_size[0])),
        Image.BILINEAR,
    )


def write_outputs(
    output_dir: Path,
    *,
    sequence,
    gt_masks: Sequence[torch.Tensor],
    pred_masks: Sequence[torch.Tensor],
    prompt: str,
    instance_id: int,
    prompt_frame: int,
    selected_obj_id: int | None,
    prompt_box_xywh: tuple[float, float, float, float] | None,
    output_size: tuple[int, int],
) -> list[dict]:
    np.savez_compressed(
        output_dir / "sam3_masks.npz",
        masks=torch.stack([mask.bool().cpu() for mask in pred_masks], dim=0).numpy(),
        gt_masks=torch.stack([mask.bool().cpu() for mask in gt_masks], dim=0).numpy(),
    )
    metrics = []
    mask_dir = output_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    for frame_idx, (gt, pred) in enumerate(zip(gt_masks, pred_masks)):
        pred = pred.detach().bool().cpu()
        gt = gt.detach().bool().cpu()
        row = {
            "local_frame": frame_idx,
            "scene_frame": int(sequence.frame_indices[frame_idx]),
            "image": str(sequence.image_paths[frame_idx]),
            "gt_pixels": int(gt.sum().item()),
            "pred_pixels": int(pred.sum().item()),
            "iou": binary_iou(pred, gt),
        }
        metrics.append(row)
        Image.fromarray(pred.numpy().astype(np.uint8) * 255).save(
            mask_dir / f"frame_{frame_idx:03d}_pred.png"
        )
        Image.fromarray(gt.numpy().astype(np.uint8) * 255).save(
            mask_dir / f"frame_{frame_idx:03d}_gt.png"
        )

    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0].keys()))
        writer.writeheader()
        writer.writerows(metrics)

    save_visualization(
        output_dir / "sam3_sequence.png",
        sequence=sequence,
        gt_masks=gt_masks,
        pred_masks=pred_masks,
        metrics=metrics,
        prompt=prompt,
        instance_id=instance_id,
        prompt_frame=prompt_frame,
        selected_obj_id=selected_obj_id,
        prompt_box_xywh=prompt_box_xywh,
        output_size=output_size,
    )
    return metrics


def save_visualization(
    path: Path,
    *,
    sequence,
    gt_masks: Sequence[torch.Tensor],
    pred_masks: Sequence[torch.Tensor],
    metrics: Sequence[dict],
    prompt: str,
    instance_id: int,
    prompt_frame: int,
    selected_obj_id: int | None,
    prompt_box_xywh: tuple[float, float, float, float] | None,
    output_size: tuple[int, int],
) -> None:
    frames = len(sequence.image_paths)
    panel_w = 300
    panel_h = int(round(panel_w * output_size[0] / output_size[1]))
    title_h = 42
    row_label_w = 210
    margin = 8
    columns = 4
    width = row_label_w + columns * panel_w + (columns + 1) * margin
    height = title_h + frames * (panel_h + margin) + margin
    canvas = Image.new("RGB", (width, height), color=(18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (margin, 8),
        (
            f"SAM3 original video predictor | prompt={prompt!r} "
            f"instance={instance_id} prompt_frame={prompt_frame} "
            f"selected_obj={selected_obj_id} box={prompt_box_xywh}"
        ),
        fill=(245, 245, 245),
    )
    headings = ["RGB", "GT", "SAM3", "overlap"]
    for col, heading in enumerate(headings):
        x = row_label_w + margin + col * (panel_w + margin)
        draw.text((x, title_h - 18), heading, fill=(220, 220, 220))

    for frame_idx in range(frames):
        image = load_rgb(sequence.image_paths[frame_idx], output_size)
        gt = gt_masks[frame_idx].detach().bool().cpu().numpy()
        pred = pred_masks[frame_idx].detach().bool().cpu().numpy()
        panels = [
            image,
            overlay_mask(image, gt, (255, 90, 70), 0.55),
            overlay_mask(image, pred, (60, 220, 120), 0.55),
            overlap_panel(image, gt, pred),
        ]
        row_y = title_h + margin + frame_idx * (panel_h + margin)
        label = (
            f"local={frame_idx} scene={sequence.frame_indices[frame_idx]}\n"
            f"IoU={metrics[frame_idx]['iou']:.3f}\n"
            f"gt={metrics[frame_idx]['gt_pixels']} pred={metrics[frame_idx]['pred_pixels']}"
        )
        draw.text((margin, row_y + 6), label, fill=(230, 230, 230))
        for col, panel in enumerate(panels):
            panel = panel.resize((panel_w, panel_h), Image.BILINEAR)
            x = row_label_w + margin + col * (panel_w + margin)
            canvas.paste(panel, (x, row_y))
    canvas.save(path)


def overlay_mask(
    image: Image.Image,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float,
) -> Image.Image:
    base = np.asarray(image).astype(np.float32)
    mask_bool = np.asarray(mask).astype(bool)
    color_arr = np.asarray(color, dtype=np.float32)
    base[mask_bool] = base[mask_bool] * (1.0 - alpha) + color_arr * alpha
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def overlap_panel(image: Image.Image, gt: np.ndarray, pred: np.ndarray) -> Image.Image:
    base = np.asarray(image).astype(np.float32) * 0.45
    gt = np.asarray(gt).astype(bool)
    pred = np.asarray(pred).astype(bool)
    both = gt & pred
    gt_only = gt & ~pred
    pred_only = pred & ~gt
    base[gt_only] = np.asarray((255, 90, 70), dtype=np.float32)
    base[pred_only] = np.asarray((60, 220, 120), dtype=np.float32)
    base[both] = np.asarray((255, 220, 80), dtype=np.float32)
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def mean_visible_iou(metrics: Sequence[dict]) -> float:
    visible = [float(row["iou"]) for row in metrics if int(row["gt_pixels"]) > 0]
    if not visible:
        return 0.0
    return float(sum(visible) / len(visible))


if __name__ == "__main__":
    main()
