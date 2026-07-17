"""Ablate geometry and StreamVGGT descriptors for SAM3 recovery selection."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw
import torch

from test_sam.data import load_mask_tracking_sequence

from .backbones.sam3_wrapper import SAM3Wrapper
from .backbones.streamvggt_wrapper import StreamVGGTWrapper
from .bridge.gating import binary_iou
from .bridge.segment_descriptor import SELECTION_MODES
from .config import ExperimentConfig, load_config
from .pipeline import _resize_target_masks, summarize_masks
from .sam3_mask_camera_refinement_experiment import (
    _run_recurrent_hard_memory,
)
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
        descriptor_layer=args.descriptor_layer,
        context_grid=(
            tuple(args.context_grid)
            if args.context_grid is not None
            else None
        ),
        descriptor_geometry_weight=args.descriptor_geometry_weight,
        descriptor_weight=args.descriptor_weight,
        reference_sequence_index=args.reference_sequence_index,
    )


def run_experiment(
    config: ExperimentConfig,
    *,
    modes: tuple[str, ...],
    descriptor_layer: int,
    context_grid: tuple[int, int] | None,
    descriptor_geometry_weight: float,
    descriptor_weight: float,
    reference_sequence_index: int,
) -> None:
    invalid_modes = sorted(set(modes) - set(SELECTION_MODES))
    if invalid_modes:
        raise ValueError(f"Unsupported selection modes: {invalid_modes}.")
    if not modes:
        raise ValueError("At least one candidate selection mode is required.")
    if descriptor_geometry_weight < 0.0 or descriptor_weight < 0.0:
        raise ValueError("Descriptor selection weights must be non-negative.")
    if descriptor_geometry_weight + descriptor_weight <= 0.0:
        raise ValueError("At least one candidate selection weight must be positive.")

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
    reference_sequence_index = int(reference_sequence_index)
    if not 0 <= reference_sequence_index < len(sequence.frame_indices):
        raise ValueError("reference_sequence_index is outside the sequence.")
    sequence = replace(
        sequence,
        reference_frame_idx=reference_sequence_index,
    )
    target_masks = _resize_target_masks(
        sequence.target_masks,
        config.output_size,
    )
    if not target_masks[reference_sequence_index].any():
        raise ValueError(
            "The selected reference frame does not contain the target instance."
        )
    print(
        f"target scene={sequence.scene_id} frames={sequence.frame_indices} "
        f"instance={sequence.instance_id} label={sequence.label!r} "
        f"reference={reference_sequence_index} "
        f"visible={target_masks.flatten(1).any(dim=1).tolist()}"
    )

    print(
        f"extracting frozen StreamVGGT layer {descriptor_layer} once with "
        "causal caches..."
    )
    stream = StreamVGGTWrapper(
        repo_path=config.streamvggt_repo,
        checkpoint_path=config.streamvggt_checkpoint,
        device=config.geometry_device,
        image_mode=config.image_mode,
        streaming_cache=config.streaming_cache,
    ).load()
    geometry, feature_levels = stream.extract_with_latents(
        sequence.image_paths,
        layer_indices=(int(descriptor_layer),),
        context_grid=context_grid,
    )
    del stream
    geometry_features = feature_levels[0]
    print(
        f"descriptor feature shape={tuple(geometry_features.shape)} "
        f"processed_size={geometry.processed_size}"
    )

    print("running frozen SAM3 original tracker once...")
    sam3 = SAM3Wrapper(
        repo_path=config.sam3_repo,
        checkpoint_path=config.sam3_checkpoint,
        device=config.sam3_device,
        output_threshold=config.sam3_output_threshold,
        prompt_with_box=config.prompt_with_box,
    ).load()
    original = sam3.track(
        sequence.image_paths,
        prompt=sequence.label,
        output_size=config.output_size,
        reference_frame_idx=reference_sequence_index,
        reference_mask=target_masks[reference_sequence_index],
    )
    original_metrics = summarize_masks(
        original.masks,
        target_masks,
        reference_frame_idx=reference_sequence_index,
    )
    print(
        f"original cross_iou={original_metrics['cross_view_iou']:.4f} "
        f"recall={original_metrics['cross_view_recall']:.4f}"
    )

    results: dict[str, TrackingSequence] = {"original": original}
    summary_rows = [
        {
            "mode": "original",
            "descriptor_layer": descriptor_layer,
            "geometry_weight": descriptor_geometry_weight,
            "descriptor_weight": descriptor_weight,
            **original_metrics,
            "cross_iou_gain_over_original": 0.0,
            "recovery_count": 0,
            "selected_candidate_events": 0,
            "mean_selected_geometry_iou": float("nan"),
            "mean_selected_descriptor_cosine": float("nan"),
            "mean_selected_candidate_gt_iou": float("nan"),
            "persistent_obj_id": original.selected_obj_id,
        }
    ]
    frame_rows = _tracking_frame_rows(
        "original",
        original,
        target_masks=target_masks,
        frame_indices=sequence.frame_indices,
    )
    selection_rows = []
    hard_memory_rows = []
    recoveries = {}
    text_candidate_cache = {}
    for mode in modes:
        torch.manual_seed(0)
        print(f"running recovery candidate selection mode={mode}...")
        tracking, recovery, mode_hard_rows = _run_recurrent_hard_memory(
            config,
            sequence=sequence,
            target_output_masks=target_masks,
            original_tracking=original,
            geometry=geometry,
            sam3=sam3,
            recovery_prompt_mode="global_text_select",
            geometry_features=geometry_features,
            candidate_selection_mode=mode,
            descriptor_geometry_weight=descriptor_geometry_weight,
            descriptor_weight=descriptor_weight,
            text_candidate_cache=text_candidate_cache,
        )
        results[mode] = tracking
        recoveries[mode] = {
            key: value
            for key, value in recovery.items()
            if key != "candidate_selection_rows"
        }
        mode_selection_rows = recovery["candidate_selection_rows"]
        selection_rows.extend(mode_selection_rows)
        hard_memory_rows.extend(
            {"mode": mode, **row}
            for row in mode_hard_rows
        )
        frame_rows.extend(
            _tracking_frame_rows(
                mode,
                tracking,
                target_masks=target_masks,
                frame_indices=sequence.frame_indices,
                hard_rows=mode_hard_rows,
            )
        )
        metrics = summarize_masks(
            tracking.masks,
            target_masks,
            reference_frame_idx=reference_sequence_index,
        )
        selected = [row for row in mode_selection_rows if row["selected"]]
        summary_rows.append(
            {
                "mode": mode,
                "descriptor_layer": descriptor_layer,
                "geometry_weight": descriptor_geometry_weight,
                "descriptor_weight": descriptor_weight,
                **metrics,
                "cross_iou_gain_over_original": (
                    metrics["cross_view_iou"]
                    - original_metrics["cross_view_iou"]
                ),
                "recovery_count": recovery["recovery_count"],
                "selected_candidate_events": len(selected),
                "mean_selected_geometry_iou": _finite_mean(
                    row["geometry_iou"] for row in selected
                ),
                "mean_selected_descriptor_cosine": _finite_mean(
                    row["descriptor_cosine"] for row in selected
                ),
                "mean_selected_candidate_gt_iou": _finite_mean(
                    row["candidate_gt_iou"] for row in selected
                ),
                "persistent_obj_id": tracking.selected_obj_id,
            }
        )
        print(
            f"mode={mode:<22} cross_iou={metrics['cross_view_iou']:.4f} "
            f"recall={metrics['cross_view_recall']:.4f} "
            f"absent_fp={metrics['absent_fp_ratio']:.6f} "
            f"recoveries={recovery['recovery_count']}"
        )

    _write_csv(config.output_dir / "summary.csv", summary_rows)
    _write_csv(config.output_dir / "frame_metrics.csv", frame_rows)
    _write_csv(
        config.output_dir / "candidate_selection_metrics.csv",
        selection_rows,
    )
    _write_csv(
        config.output_dir / "hard_memory_metrics.csv",
        hard_memory_rows,
    )
    _save_report(
        config.output_dir / "recovery_selection_report.png",
        image_paths=sequence.image_paths,
        frame_indices=sequence.frame_indices,
        target_masks=target_masks,
        results=results,
        output_size=config.output_size,
    )
    metadata = {
        "experiment": "recovery_candidate_selection_ablation",
        "scene_id": sequence.scene_id,
        "instance_id": sequence.instance_id,
        "instance_label": sequence.label,
        "frame_indices": sequence.frame_indices,
        "reference_sequence_index": reference_sequence_index,
        "modes": list(modes),
        "descriptor_layer": int(descriptor_layer),
        "descriptor_feature_shape": list(geometry_features.shape),
        "context_grid": list(geometry_features.shape[-2:]),
        "native_patch_grid": context_grid is None,
        "geometry_weight": float(descriptor_geometry_weight),
        "descriptor_weight": float(descriptor_weight),
        "recoveries": recoveries,
        "scientific_controls": {
            "sam3_candidate_generation_shared": True,
            "sam3_candidates_cached_by_recovery_frame": True,
            "streamvggt_geometry_shared": True,
            "hard_memory_gate_shared": True,
            "same_obj_id_writeback_shared": True,
            "later_gt_used_for_metrics_only": True,
            "shuffled_control_changes_descriptor_alignment_only": True,
        },
    }
    with (config.output_dir / "resolved_experiment.json").open(
        "w", encoding="utf8"
    ) as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    print(f"summary: {config.output_dir / 'summary.csv'}")
    print(
        "candidate diagnostics: "
        f"{config.output_dir / 'candidate_selection_metrics.csv'}"
    )
    print(
        f"visualization: {config.output_dir / 'recovery_selection_report.png'}"
    )


def _tracking_frame_rows(
    mode: str,
    tracking: TrackingSequence,
    *,
    target_masks: torch.Tensor,
    frame_indices: Sequence[int],
    hard_rows: list[dict] | None = None,
) -> list[dict]:
    hard_by_index = {
        int(row["sequence_index"]): row
        for row in (hard_rows or [])
    }
    rows = []
    for sequence_index, frame_index in enumerate(frame_indices):
        hard = hard_by_index.get(sequence_index, {})
        rows.append(
            {
                "mode": mode,
                "sequence_index": sequence_index,
                "frame_index": int(frame_index),
                "gt_visible": int(target_masks[sequence_index].any()),
                "iou": binary_iou(
                    tracking.masks[sequence_index],
                    target_masks[sequence_index],
                ),
                "score": float(tracking.scores[sequence_index]),
                "prediction_pixels": int(tracking.masks[sequence_index].sum()),
                "target_pixels": int(target_masks[sequence_index].sum()),
                "mask_source_choice": hard.get("mask_source_choice", "sam3_original"),
                "tracker_weak": int(hard.get("tracker_weak", 0)),
                "recovery_requested": int(hard.get("recovery_requested", 0)),
                "recovery_applied": int(hard.get("recovery_applied", 0)),
                "map_update": int(hard.get("map_update", 0)),
            }
        )
    return rows


def _save_report(
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
    colors = {
        "GT": (255, 64, 64),
        "original": (64, 170, 255),
        "geometry_only": (255, 180, 50),
        "descriptor_only": (60, 210, 130),
        "geometry_descriptor": (230, 70, 190),
        "shuffled_descriptor": (145, 90, 230),
    }
    header = 30
    canvas = Image.new(
        "RGB",
        (width * len(labels), (height + header) * len(image_paths)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for row_index, image_path in enumerate(image_paths):
        with Image.open(image_path) as source:
            rgb = source.convert("RGB").resize(
                (width, height),
                Image.Resampling.BILINEAR,
            )
        cells = {"RGB": rgb}
        cells["GT"] = _overlay(rgb, target_masks[row_index], colors["GT"])
        for mode, tracking in results.items():
            cells[mode] = _overlay(
                rgb,
                tracking.masks[row_index],
                colors[mode],
            )
        for column, label in enumerate(labels):
            x = column * width
            y = row_index * (height + header)
            canvas.paste(cells[label], (x, y + header))
            title = f"{label} | frame={frame_indices[row_index]}"
            if label in results:
                title += (
                    f" | IoU={binary_iou(results[label].masks[row_index], target_masks[row_index]):.3f}"
                )
            draw.text((x + 5, y + 7), title, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _overlay(
    image: Image.Image,
    mask: torch.Tensor,
    color: tuple[int, int, int],
) -> Image.Image:
    array = np.asarray(image).copy()
    selected = mask.detach().cpu().numpy().astype(bool)
    if selected.any():
        array[selected] = (
            0.45 * array[selected] + 0.55 * np.asarray(color)
        ).astype(np.uint8)
    return Image.fromarray(array)


def _finite_mean(values) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(finite.mean()) if finite.size else float("nan")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf8")
        return
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("streaming_couping/configs/default.yaml"),
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--scene-id")
    parser.add_argument("--instance-id", type=int)
    parser.add_argument("--frame-indices", type=int, nargs="+")
    parser.add_argument("--reference-sequence-index", type=int, default=0)
    parser.add_argument("--sam3-device")
    parser.add_argument("--geometry-device")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=SELECTION_MODES,
        default=SELECTION_MODES,
    )
    parser.add_argument("--descriptor-layer", type=int, default=17)
    parser.add_argument(
        "--context-grid",
        type=int,
        nargs=2,
        help="Optional descriptor grid override; default keeps the native patch grid.",
    )
    parser.add_argument(
        "--descriptor-geometry-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument("--descriptor-weight", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    main()
