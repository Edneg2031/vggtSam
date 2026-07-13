#!/usr/bin/env python3
"""Evaluate explicit StreamVGGT geometry recovery around frozen SAM3 tracking."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from test_sam.data import load_mask_tracking_sequence

from .backbones import run_frozen_sam3, run_frozen_streamvggt
from .config import ExperimentConfig, load_config
from .gates import decide_gates
from .geometry import (
    centroid_drift,
    mask_geometry_statistics,
    output_mask_to_stream,
    project_world_points,
    source_mask_to_stream,
)
from .metrics import binary_iou, summarize_masks
from .object_map import ObjectPointMap
from .types import GeometrySequence
from .visualize import save_report


def main() -> None:
    args = parse_args()
    overrides = {
        key: value
        for key, value in {
            "scene_id": args.scene_id,
            "frame_indices": args.frame_indices,
            "instance_id": args.instance_id,
            "sam3_device": args.sam3_device,
            "geometry_device": args.geometry_device,
            "geometry_modes": args.geometry_modes,
            "output_dir": args.output_dir,
        }.items()
        if value is not None
    }
    run(load_config(args.config, overrides))


def run(config: ExperimentConfig) -> None:
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
    target_masks = resize_target_masks(sequence.target_masks, config.output_size)
    print(
        f"target scene={sequence.scene_id} frames={sequence.frame_indices} "
        f"instance={sequence.instance_id} label={sequence.label!r} "
        f"reference_frame={sequence.reference_frame_idx} "
        f"visible={[bool(mask.any()) for mask in target_masks]}"
    )

    print("running frozen SAM3 video tracker...")
    sam_output = run_frozen_sam3(config, sequence, target_masks)
    sam_masks = sam_output.masks.cpu().bool()
    sam_scores = (
        sam_output.scores.cpu().float()
        if sam_output.scores is not None
        else sam_masks.flatten(1).any(dim=1).float()
    )
    original_metrics = summarize_masks(
        sam_masks,
        target_masks,
        reference_frame_idx=sequence.reference_frame_idx,
    )
    print(
        f"SAM3 original obj_id={sam_output.selected_obj_id} "
        f"cross_iou={original_metrics['cross_view_iou']:.4f} "
        f"cross_recall={original_metrics['cross_view_recall']:.4f}"
    )

    print("running frozen StreamVGGT with causal caches...")
    geometry = run_frozen_streamvggt(config, sequence)
    diagnostics = static_centroid_diagnostics(config, sequence, geometry)
    write_csv(config.output_dir / "static_centroid_diagnostics.csv", diagnostics)

    summaries = []
    for mode in config.geometry_modes:
        result = evaluate_mode(
            config,
            sequence=sequence,
            target_masks=target_masks,
            sam_masks=sam_masks,
            sam_scores=sam_scores,
            geometry=geometry,
            mode=mode,
        )
        mode_dir = config.output_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        write_csv(mode_dir / "frame_metrics.csv", result["rows"])
        save_object_map(
            mode_dir / "object_map.npz",
            result["object_map"],
            instance_id=sequence.instance_id,
            label=sequence.label,
        )
        save_report(
            mode_dir / "tracking_report.png",
            image_paths=sequence.image_paths,
            frame_indices=sequence.frame_indices,
            gt_masks=target_masks,
            sam_masks=sam_masks,
            priors=result["priors"],
            bridged_masks=result["bridged_masks"],
            scores=sam_scores,
            decisions=[row["gate_reason"] for row in result["rows"]],
            output_size=config.output_size,
            mode=mode,
        )
        summary = {
            "mode": mode,
            **{f"sam3_{key}": value for key, value in original_metrics.items()},
            **{f"bridge_{key}": value for key, value in result["metrics"].items()},
            "fallback_frames": sum(int(row["use_fallback"]) for row in result["rows"]),
            "map_initializations": sum(int(row["initialize_map"]) for row in result["rows"]),
            "map_updates": sum(int(row["update_map"]) for row in result["rows"]),
        }
        summaries.append(summary)
        print(
            f"mode={mode} bridge_cross_iou={result['metrics']['cross_view_iou']:.4f} "
            f"recall={result['metrics']['cross_view_recall']:.4f} "
            f"fallbacks={summary['fallback_frames']} updates={summary['map_updates']}"
        )

    write_csv(config.output_dir / "summary.csv", summaries)
    with (config.output_dir / "resolved_config.json").open("w", encoding="utf8") as handle:
        json.dump(config_as_json(config), handle, indent=2)
    print(f"summary: {config.output_dir / 'summary.csv'}")


def evaluate_mode(
    config: ExperimentConfig,
    *,
    sequence,
    target_masks: torch.Tensor,
    sam_masks: torch.Tensor,
    sam_scores: torch.Tensor,
    geometry: GeometrySequence,
    mode: str,
) -> dict:
    num_frames = len(sequence.frame_indices)
    permutation = geometry_permutation(num_frames, mode)
    object_map = ObjectPointMap(max_points_per_object=config.max_points_per_object)
    priors = torch.zeros_like(target_masks)
    bridged = torch.zeros_like(target_masks)
    rows = []
    persistence = 0

    for frame_idx in range(num_frames):
        geometry_idx = int(permutation[frame_idx])
        source_size = geometry.source_sizes[geometry_idx]
        points = geometry.world_points[geometry_idx]
        confidence = geometry.confidence[geometry_idx]
        is_reference = frame_idx == sequence.reference_frame_idx

        tracker_mask = sam_masks[frame_idx]
        tracker_score = float(sam_scores[frame_idx])
        update_output_mask = (
            target_masks[frame_idx]
            if config.map_update_source == "oracle"
            else tracker_mask
        )
        update_stream_mask = output_mask_to_stream(
            update_output_mask,
            source_size=source_size,
            processed_size=geometry.processed_size,
            image_mode=config.image_mode,
        )
        sampled_points, sampled_weights, region_confidence, centroid = mask_geometry_statistics(
            points,
            confidence,
            update_stream_mask,
            max_points=config.max_points_per_observation,
        )
        update_geometry_confidence = (
            region_confidence if update_stream_mask.any() else float(confidence.mean())
        )

        initialized_map = False
        if is_reference and mode != "zero":
            object_map.update(
                instance_id=sequence.instance_id,
                label=sequence.label,
                points=sampled_points,
                weights=sampled_weights,
                centroid=centroid,
                frame_idx=frame_idx,
            )
            persistence = 1
            initialized_map = object_map.has(sequence.instance_id)

        entry = object_map.get(sequence.instance_id)
        if entry is not None and mode != "zero":
            priors[frame_idx] = project_world_points(
                entry.points,
                world_to_camera=geometry.world_to_camera[geometry_idx],
                intrinsics=geometry.intrinsics[geometry_idx],
                source_size=source_size,
                processed_size=geometry.processed_size,
                output_size=config.output_size,
                image_mode=config.image_mode,
                splat_radius=config.splat_radius,
                observed_world_points=points,
                occlusion_depth_tolerance=config.occlusion_depth_tolerance,
                occlusion_relative_tolerance=config.occlusion_relative_tolerance,
            ).cpu()

        if priors[frame_idx].any():
            prior_stream_mask = output_mask_to_stream(
                priors[frame_idx],
                source_size=source_size,
                processed_size=geometry.processed_size,
                image_mode=config.image_mode,
            )
            _, _, fallback_geometry_confidence, _ = mask_geometry_statistics(
                points,
                confidence,
                prior_stream_mask,
                max_points=config.max_points_per_observation,
            )
        else:
            fallback_geometry_confidence = 0.0

        decision = decide_gates(
            track_confidence=tracker_score,
            update_geometry_confidence=update_geometry_confidence,
            fallback_geometry_confidence=fallback_geometry_confidence,
            persistence=persistence,
            has_object_map=entry is not None and bool(priors[frame_idx].any()),
            config=config.gates,
        )
        if is_reference:
            bridged[frame_idx] = tracker_mask
        elif decision.use_fallback:
            bridged[frame_idx] = priors[frame_idx]
        else:
            bridged[frame_idx] = tracker_mask

        prior_consistency_iou = (
            binary_iou(tracker_mask, priors[frame_idx])
            if tracker_mask.any() and priors[frame_idx].any()
            else 0.0
        )
        geometrically_consistent = (
            is_reference
            or prior_consistency_iou >= config.map_update_min_prior_iou
        )
        gate_reason = decision.reason
        if decision.update_map and not geometrically_consistent:
            gate_reason = "reject map update: tracker mask disagrees with 3D prior"
        updated_map = False
        if (
            not is_reference
            and decision.update_map
            and geometrically_consistent
            and mode != "zero"
            and sampled_points.numel() > 0
        ):
            object_map.update(
                instance_id=sequence.instance_id,
                label=sequence.label,
                points=sampled_points,
                weights=sampled_weights,
                centroid=centroid,
                frame_idx=frame_idx,
            )
            persistence += 1
            updated_map = True

        rows.append(
            {
                "sequence_index": frame_idx,
                "frame_index": sequence.frame_indices[frame_idx],
                "geometry_index": geometry_idx,
                "gt_visible": int(target_masks[frame_idx].any()),
                "sam3_score": tracker_score,
                "update_geometry_confidence": update_geometry_confidence,
                "fallback_geometry_confidence": fallback_geometry_confidence,
                "sam3_iou": binary_iou(tracker_mask, target_masks[frame_idx]),
                "prior_iou": binary_iou(priors[frame_idx], target_masks[frame_idx]),
                "bridge_iou": binary_iou(bridged[frame_idx], target_masks[frame_idx]),
                "prior_tracker_iou": prior_consistency_iou,
                "initialize_map": int(initialized_map),
                "update_map": int(updated_map),
                "use_fallback": int(decision.use_fallback),
                "map_points": int(entry.points.shape[0]) if entry is not None else 0,
                "gate_reason": gate_reason,
            }
        )
    return {
        "rows": rows,
        "priors": priors,
        "bridged_masks": bridged,
        "metrics": summarize_masks(
            bridged,
            target_masks,
            reference_frame_idx=sequence.reference_frame_idx,
        ),
        "object_map": object_map.get(sequence.instance_id),
    }


def static_centroid_diagnostics(
    config: ExperimentConfig,
    sequence,
    geometry: GeometrySequence,
) -> list[dict]:
    centroids = []
    rows = []
    for sequence_idx, instance_labels in enumerate(sequence.instance_masks):
        mask = source_mask_to_stream(
            instance_labels == sequence.instance_id,
            geometry.processed_size,
            image_mode=config.image_mode,
        )
        _, _, confidence, centroid = mask_geometry_statistics(
            geometry.world_points[sequence_idx],
            geometry.confidence[sequence_idx],
            mask,
            max_points=config.max_points_per_observation,
        )
        centroids.append(centroid)
        rows.append(
            {
                "sequence_index": sequence_idx,
                "frame_index": sequence.frame_indices[sequence_idx],
                "visible": int(mask.any()),
                "geometry_confidence": confidence,
                "centroid_x": float(centroid[0]),
                "centroid_y": float(centroid[1]),
                "centroid_z": float(centroid[2]),
            }
        )
    drift = centroid_drift(centroids)
    for row in rows:
        row["mean_reference_centroid_drift"] = drift
    return rows


def geometry_permutation(num_frames: int, mode: str) -> torch.Tensor:
    indices = torch.arange(num_frames)
    if mode == "shuffled" and num_frames > 1:
        return torch.roll(indices, shifts=1)
    return indices


def resize_target_masks(masks, output_size: tuple[int, int]) -> torch.Tensor:
    tensor = torch.from_numpy(np.stack(masks).astype(np.float32))[:, None]
    return F.interpolate(tensor, size=output_size, mode="nearest")[:, 0].bool()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_object_map(path: Path, entry, *, instance_id: int, label: str) -> None:
    if entry is None:
        return
    np.savez_compressed(
        path,
        points=entry.points.numpy(),
        confidence=entry.weights.numpy(),
        instance_id=np.asarray([int(instance_id)], dtype=np.int64),
        label=np.asarray([str(label)]),
        observations=np.asarray([int(entry.observations)], dtype=np.int64),
    )


def config_as_json(config: ExperimentConfig) -> dict:
    value = asdict(config)
    for key, item in list(value.items()):
        if isinstance(item, Path):
            value[key] = str(item)
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("test_dframework/config.yaml"))
    parser.add_argument("--scene-id")
    parser.add_argument("--instance-id", type=int)
    parser.add_argument("--frame-indices", type=int, nargs="+")
    parser.add_argument("--sam3-device")
    parser.add_argument("--geometry-device")
    parser.add_argument("--geometry-modes", nargs="+", choices=("aligned", "zero", "shuffled"))
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    main()
