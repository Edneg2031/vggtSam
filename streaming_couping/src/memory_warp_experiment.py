"""Controlled ablation for StreamVGGT-aware SAM3 memory position warping."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw

from test_sam.data import load_mask_tracking_sequence

from .backbones.sam3_wrapper import SAM3Wrapper
from .backbones.streamvggt_wrapper import StreamVGGTWrapper
from .bridge.gating import binary_iou
from .bridge.memory_warp import GeometryMemoryPositionWarper
from .config import ExperimentConfig, load_config
from .pipeline import _resize_target_masks, summarize_masks
from .types import TrackingSequence


def main() -> None:
    args = _parse_args()
    overrides = {
        key: value
        for key, value in {
            "manifest": args.manifest,
            "scene_id": args.scene_id,
            "instance_id": args.instance_id,
            "frame_indices": args.frame_indices,
            "sam3_device": args.sam3_device,
            "geometry_device": args.geometry_device,
            "output_dir": args.output_dir,
        }.items()
        if value is not None
    }
    run_experiment(
        load_config(args.config, overrides),
        modes=tuple(args.modes),
        point_source=args.geometry_point_source,
        min_geometry_confidence=args.min_geometry_confidence,
        strict_identity=not args.allow_identity_mismatch,
    )


def run_experiment(
    config: ExperimentConfig,
    *,
    modes: tuple[str, ...],
    point_source: str,
    min_geometry_confidence: float,
    strict_identity: bool,
) -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    sequence = load_mask_tracking_sequence(
        config.manifest,
        scene_id=config.scene_id,
        frame_indices=config.frame_indices,
        sequence_length=len(config.frame_indices),
        frame_stride=1,
        window_index=0,
        instance_id=config.instance_id,
        min_pixels=config.min_pixels,
        max_area_ratio=config.max_area_ratio,
        min_visible_frames=1,
        excluded_labels=config.excluded_labels,
        seed=0,
    )
    target_masks = _resize_target_masks(sequence.target_masks, config.output_size)
    visible = [bool(mask.any()) for mask in target_masks]
    print(
        f"target scene={sequence.scene_id} frames={sequence.frame_indices} "
        f"instance={sequence.instance_id} label={sequence.label!r} "
        f"reference={sequence.reference_frame_idx} visible={visible}"
    )

    print("running frozen StreamVGGT once with causal caches...")
    geometry = StreamVGGTWrapper(
        repo_path=config.streamvggt_repo,
        checkpoint_path=config.streamvggt_checkpoint,
        device=config.geometry_device,
        image_mode=config.image_mode,
        streaming_cache=config.streaming_cache,
    ).load().extract(sequence.image_paths)
    print(
        f"geometry point_source={point_source} processed_size={geometry.processed_size}"
    )

    print("loading frozen SAM3 and running original memory baseline...")
    sam3 = SAM3Wrapper(
        repo_path=config.sam3_repo,
        checkpoint_path=config.sam3_checkpoint,
        device=config.sam3_device,
        output_threshold=config.sam3_output_threshold,
        prompt_with_box=config.prompt_with_box,
    ).load()
    original = _track_original(sam3, sequence, target_masks, config.output_size)
    results: dict[str, TrackingSequence] = {"original": original}
    warpers: dict[str, GeometryMemoryPositionWarper] = {}
    permutation = _cyclic_permutation(len(sequence.frame_indices))

    for mode in modes:
        torch.manual_seed(0)
        warper = GeometryMemoryPositionWarper(
            geometry,
            mode=mode,
            image_mode=config.image_mode,
            frame_permutation=permutation,
            min_geometry_confidence=min_geometry_confidence,
            point_source=point_source,
        )
        print(
            f"running SAM3 memory mode={mode} "
            f"geometry_permutation={list(permutation) if mode == 'shuffled' else 'aligned'}"
        )
        tracking = sam3.track_with_memory_position_warp(
            sequence.image_paths,
            prompt=sequence.label,
            output_size=config.output_size,
            reference_frame_idx=sequence.reference_frame_idx,
            reference_mask=target_masks[sequence.reference_frame_idx],
            warper=warper,
        )
        results[mode] = tracking
        warpers[mode] = warper

    identity_matches = True
    if "identity" in results:
        identity_matches = bool(
            torch.equal(original.masks, results["identity"].masks)
            and original.selected_obj_id == results["identity"].selected_obj_id
        )
        print(f"identity_matches_original={identity_matches}")

    summary_rows = _summary_rows(
        results,
        warpers=warpers,
        target_masks=target_masks,
        reference_frame_idx=sequence.reference_frame_idx,
        identity_matches=identity_matches,
    )
    frame_rows = _frame_rows(
        results,
        warpers=warpers,
        target_masks=target_masks,
        frame_indices=sequence.frame_indices,
    )
    _write_csv(config.output_dir / "summary.csv", summary_rows)
    _write_csv(config.output_dir / "frame_metrics.csv", frame_rows)
    pair_rows = []
    for mode, warper in warpers.items():
        pair_rows.extend({"mode": mode, **row} for row in warper.observation_rows())
    _write_csv(config.output_dir / "memory_warp_pairs.csv", pair_rows)
    _save_visualization(
        config.output_dir / "memory_warp_report.png",
        image_paths=sequence.image_paths,
        frame_indices=sequence.frame_indices,
        target_masks=target_masks,
        results=results,
        output_size=config.output_size,
    )
    metadata = {
        "experiment": "sam3_memory_position_warp",
        "scene_id": sequence.scene_id,
        "frame_indices": sequence.frame_indices,
        "instance_id": sequence.instance_id,
        "label": sequence.label,
        "reference_sequence_index": sequence.reference_frame_idx,
        "modes": ["original", *modes],
        "point_source": point_source,
        "min_geometry_confidence": float(min_geometry_confidence),
        "shuffled_geometry_permutation": list(permutation),
        "identity_matches_original": identity_matches,
        "scientific_controls": {
            "reference_gt_only": True,
            "later_gt_metrics_only": True,
            "geometry_fallback_disabled": True,
            "memory_writeback_disabled": True,
            "redetection_disabled": True,
            "persistent_obj_id_owned_by_sam3": True,
            "only_memory_spatial_position_encoding_changes": True,
        },
    }
    with (config.output_dir / "resolved_experiment.json").open(
        "w", encoding="utf8"
    ) as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    for row in summary_rows:
        print(
            f"mode={row['mode']:<9} cross_iou={row['cross_view_iou']:.4f} "
            f"recall={row['cross_view_recall']:.4f} "
            f"absent_fp={row['absent_fp_ratio']:.6f} "
            f"valid_warp={row['valid_warp_ratio']:.4f}"
        )
    print(f"summary: {config.output_dir / 'summary.csv'}")
    print(f"visualization: {config.output_dir / 'memory_warp_report.png'}")
    if strict_identity and not identity_matches:
        raise RuntimeError(
            "Identity memory hook changed SAM3 output. The aligned/shuffled comparison "
            "is not a valid single-variable ablation; inspect frame_metrics.csv."
        )


def _track_original(sam3, sequence, target_masks, output_size) -> TrackingSequence:
    torch.manual_seed(0)
    return sam3.track(
        sequence.image_paths,
        prompt=sequence.label,
        output_size=output_size,
        reference_frame_idx=sequence.reference_frame_idx,
        reference_mask=target_masks[sequence.reference_frame_idx],
    )


def _summary_rows(
    results: dict[str, TrackingSequence],
    *,
    warpers: dict[str, GeometryMemoryPositionWarper],
    target_masks: torch.Tensor,
    reference_frame_idx: int,
    identity_matches: bool,
) -> list[dict[str, Any]]:
    rows = []
    original_id = results["original"].selected_obj_id
    for mode, tracking in results.items():
        metrics = summarize_masks(
            tracking.masks,
            target_masks,
            reference_frame_idx=reference_frame_idx,
        )
        warp = warpers[mode].summary() if mode in warpers else {}
        rows.append(
            {
                "mode": mode,
                **metrics,
                "selected_obj_id": tracking.selected_obj_id,
                "same_obj_id_as_original": int(
                    tracking.selected_obj_id == original_id
                ),
                "identity_matches_original": int(identity_matches),
                "hook_calls": int(warp.get("hook_calls", 0)),
                "memory_pairs": int(warp.get("memory_pairs", 0)),
                "valid_warp_ratio": float(warp.get("valid_warp_ratio", 0.0)),
                "mean_warp_displacement_pixels": float(
                    warp.get("mean_warp_displacement_pixels", 0.0)
                ),
            }
        )
    return rows


def _frame_rows(
    results: dict[str, TrackingSequence],
    *,
    warpers: dict[str, GeometryMemoryPositionWarper],
    target_masks: torch.Tensor,
    frame_indices: Sequence[int],
) -> list[dict[str, Any]]:
    per_frame_warp: dict[tuple[str, int], list] = {}
    for mode, warper in warpers.items():
        for observation in warper.observations:
            per_frame_warp.setdefault((mode, observation.current_frame), []).append(
                observation
            )
    rows = []
    for mode, tracking in results.items():
        for sequence_index, (prediction, target, score) in enumerate(
            zip(tracking.masks, target_masks, tracking.scores)
        ):
            observations = per_frame_warp.get((mode, sequence_index), [])
            total = sum(item.total_tokens for item in observations)
            valid = sum(item.valid_tokens for item in observations)
            rows.append(
                {
                    "mode": mode,
                    "sequence_index": sequence_index,
                    "frame_index": int(frame_indices[sequence_index]),
                    "gt_visible": int(target.any()),
                    "score": float(score),
                    "iou": binary_iou(prediction, target),
                    "prediction_pixels": int(prediction.sum()),
                    "target_pixels": int(target.sum()),
                    "memory_pairs": len(observations),
                    "valid_warp_ratio": float(valid / max(total, 1)),
                    "mean_warp_displacement_pixels": float(
                        sum(
                            item.mean_displacement_pixels * item.valid_tokens
                            for item in observations
                        )
                        / max(valid, 1)
                    ),
                }
            )
    return rows


def _save_visualization(
    path: Path,
    *,
    image_paths: Sequence[Path],
    frame_indices: Sequence[int],
    target_masks: torch.Tensor,
    results: dict[str, TrackingSequence],
    output_size: tuple[int, int],
) -> None:
    height, width = output_size
    labels = ["RGB", "GT", *results.keys()]
    header_height = 24
    canvas = Image.new(
        "RGB",
        (width * len(labels), (height + header_height) * len(image_paths)),
        "white",
    )
    colors = {
        "GT": (255, 64, 64),
        "original": (64, 180, 255),
        "identity": (255, 190, 64),
        "aligned": (64, 220, 120),
        "shuffled": (190, 80, 255),
    }
    for row, image_path in enumerate(image_paths):
        with Image.open(image_path) as source:
            rgb = source.convert("RGB").resize(
                (width, height), Image.Resampling.BILINEAR
            )
        masks = {"GT": target_masks[row]}
        masks.update({mode: tracking.masks[row] for mode, tracking in results.items()})
        cells = {"RGB": rgb}
        for mode, mask in masks.items():
            cells[mode] = _overlay_mask(rgb, mask, colors[mode])
        for column, label in enumerate(labels):
            x = column * width
            y = row * (height + header_height)
            canvas.paste(cells[label], (x, y + header_height))
            draw = ImageDraw.Draw(canvas)
            title = f"{label} | frame={frame_indices[row]}"
            if label in results:
                title += f" | IoU={binary_iou(masks[label], target_masks[row]):.3f}"
            draw.text((x + 5, y + 5), title, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _overlay_mask(image: Image.Image, mask: torch.Tensor, color) -> Image.Image:
    mask_image = Image.fromarray((mask.cpu().numpy().astype(np.uint8) * 150), mode="L")
    overlay = Image.new("RGB", image.size, color)
    return Image.composite(overlay, image, mask_image)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _cyclic_permutation(num_frames: int) -> tuple[int, ...]:
    if num_frames < 2:
        raise ValueError("Shuffled geometry control requires at least two frames.")
    return tuple([*range(1, num_frames), 0])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ablate geometry-aware SAM3 memory position encoding."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--scene-id")
    parser.add_argument("--instance-id", type=int)
    parser.add_argument("--frame-indices", type=int, nargs="+")
    parser.add_argument("--sam3-device")
    parser.add_argument("--geometry-device")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("identity", "aligned", "shuffled"),
        default=("identity", "aligned", "shuffled"),
    )
    parser.add_argument(
        "--geometry-point-source",
        choices=("depth_camera", "point_head"),
        default="depth_camera",
        help="Use camera-consistent depth unprojection by default.",
    )
    parser.add_argument("--min-geometry-confidence", type=float, default=0.20)
    parser.add_argument("--allow-identity-mismatch", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
