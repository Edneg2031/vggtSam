"""Object-level metric-space evaluation for persistent semantic maps."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import torch

from .coordinates import streamvggt_label_to_grid
from .data import read_mask, resolve_manifest_path


@dataclass(frozen=True)
class SemanticMapMetricConfig:
    confidence_threshold: float = 0.30
    track_score_threshold: float = 0.50
    max_points_per_object: int = 4096
    distance_chunk_size: int = 512
    fscore_thresholds_m: tuple[float, ...] = (0.05, 0.10)
    voxel_size_m: float = 0.05
    ghost_distance_m: float = 0.10
    duplicate_voxel_iou: float = 0.25


def load_ground_truth_stream_masks(
    manifest_path: str | Path,
    *,
    scene_id: str,
    frame_indices: Sequence[int],
    instance_ids: Sequence[int],
    processed_size: tuple[int, int],
    image_mode: str,
) -> torch.Tensor:
    """Transform native GT instance labels onto the StreamVGGT point grid."""

    manifest_path = Path(manifest_path).expanduser().resolve()
    with manifest_path.open("r", encoding="utf8") as handle:
        manifest = json.load(handle)
    scene = next(
        (
            row
            for row in manifest.get("scenes", ())
            if str(row.get("scene_id")) == str(scene_id)
        ),
        None,
    )
    if scene is None:
        raise ValueError(f"Scene {scene_id!r} is absent from {manifest_path}.")
    frames = scene.get("frames", ())
    transformed = []
    for frame_index in frame_indices:
        value = frames[int(frame_index)].get("instance_mask")
        if not value:
            raise ValueError(
                f"Scene {scene_id} frame {frame_index} lacks an instance mask."
            )
        native = read_mask(resolve_manifest_path(value, manifest_path))
        transformed.append(
            torch.from_numpy(
                streamvggt_label_to_grid(
                    native,
                    processed_size,
                    mode=image_mode,
                ).copy()
            ).long()
        )
    labels = torch.stack(transformed)
    if instance_ids:
        return torch.stack(
            [labels == int(instance_id) for instance_id in instance_ids],
            dim=1,
        ).bool()
    height, width = (int(value) for value in processed_size)
    return torch.zeros(
        len(frame_indices),
        0,
        height,
        width,
        dtype=torch.bool,
    )


def apply_similarity(
    points: torch.Tensor,
    *,
    scale: float,
    rotation: torch.Tensor,
    translation: torch.Tensor,
) -> torch.Tensor:
    """Map native StreamVGGT points into the metric GT reference frame."""

    points = points.detach().float().cpu()
    rotation = rotation.detach().float().cpu()
    translation = translation.detach().float().cpu()
    if points.shape[-1] != 3 or tuple(rotation.shape) != (3, 3):
        raise ValueError("Invalid point or Sim(3) rotation shape.")
    if tuple(translation.shape) != (3,):
        raise ValueError("Sim(3) translation must have shape [3].")
    return float(scale) * (points @ rotation.T) + translation


def evaluate_semantic_object_map(
    *,
    scene_id: str,
    clip_name: str,
    variant: str,
    map_policy: str,
    aligned_world_points: torch.Tensor,
    target_world_points: torch.Tensor,
    confidence: torch.Tensor,
    predicted_masks: torch.Tensor,
    track_scores: torch.Tensor,
    gt_masks: torch.Tensor,
    gt_instance_ids: Sequence[int],
    gt_labels: Sequence[str],
    assignments: Sequence[Mapping[str, object]],
    map_write_mask: torch.Tensor | None = None,
    track_ids: Sequence[int] | None = None,
    config: SemanticMapMetricConfig = SemanticMapMetricConfig(),
) -> dict[str, object]:
    """Score one frozen tracking/map-write branch at object level."""

    points = aligned_world_points.detach().float().cpu()
    target_points = target_world_points.detach().float().cpu()
    confidence = confidence.detach().float().cpu()
    masks = predicted_masks.detach().cpu().bool()
    scores = track_scores.detach().cpu().float()
    gt_masks = gt_masks.detach().cpu().bool()
    if points.ndim != 4 or points.shape[-1] != 3:
        raise ValueError("aligned_world_points must have shape [S,H,W,3].")
    sequence, height, width = points.shape[:3]
    if tuple(target_points.shape) != tuple(points.shape):
        raise ValueError("Predicted and target pointmaps must share a grid.")
    if tuple(confidence.shape) != (sequence, height, width):
        raise ValueError("confidence must have shape [S,H,W].")
    if masks.ndim != 4 or masks.shape[0] != sequence or masks.shape[2:] != (
        height,
        width,
    ):
        raise ValueError("predicted_masks must have shape [S,K,H,W].")
    tracks = masks.shape[1]
    if tuple(scores.shape) != (sequence, tracks):
        raise ValueError("track_scores must have shape [S,K].")
    if tuple(gt_masks.shape) != (
        sequence,
        len(gt_instance_ids),
        height,
        width,
    ):
        raise ValueError("gt_masks do not match pointmap/object dimensions.")
    if len(gt_labels) != len(gt_instance_ids):
        raise ValueError("GT labels and IDs disagree.")
    if map_write_mask is None:
        writes = masks.flatten(2).any(dim=2)
    else:
        writes = map_write_mask.detach().cpu().bool()
        if tuple(writes.shape) != (sequence, tracks):
            raise ValueError("map_write_mask must have shape [S,K].")
    track_ids = tuple(range(tracks) if track_ids is None else track_ids)
    if len(track_ids) != tracks:
        raise ValueError("track_ids do not match prediction slots.")
    _validate_config(config)

    assignment_by_target = {
        int(row["gt_instance_id"]): int(row["slot"])
        for row in assignments
    }
    assigned_slots = set(assignment_by_target.values())
    finite_pred = torch.isfinite(points).all(dim=-1)
    finite_target = torch.isfinite(target_points).all(dim=-1)
    confidence_valid = confidence >= float(config.confidence_threshold)
    object_rows: list[dict[str, object]] = []
    target_voxels: dict[int, set[tuple[int, int, int]]] = {}
    for target_index, instance_id in enumerate(gt_instance_ids):
        slot = assignment_by_target.get(int(instance_id), -1)
        target_support = gt_masks[:, target_index] & finite_target
        target_object_points = deterministic_limit_points(
            target_points[target_support],
            config.max_points_per_object,
        )
        target_voxels[target_index] = voxel_keys(
            target_object_points,
            voxel_size=config.voxel_size_m,
        )
        if slot >= 0:
            pred_support = (
                masks[:, slot]
                & writes[:, slot, None, None]
                & (scores[:, slot, None, None]
                   >= float(config.track_score_threshold))
                & confidence_valid
                & finite_pred
            )
            predicted_object_points = deterministic_limit_points(
                points[pred_support],
                config.max_points_per_object,
            )
            paired_support = pred_support & gt_masks[:, target_index] & finite_target
            paired_pred, paired_target = deterministic_limit_pairs(
                points[paired_support],
                target_points[paired_support],
                config.max_points_per_object,
            )
        else:
            pred_support = torch.zeros(
                sequence,
                height,
                width,
                dtype=torch.bool,
            )
            predicted_object_points = torch.empty(0, 3)
            paired_pred = paired_target = torch.empty(0, 3)
        metrics = object_point_metrics(
            predicted_object_points,
            target_object_points,
            fscore_thresholds=config.fscore_thresholds_m,
            voxel_size=config.voxel_size_m,
            ghost_distance=config.ghost_distance_m,
            chunk_size=config.distance_chunk_size,
        )
        paired_rmse = (
            float(
                torch.sqrt(
                    (paired_pred - paired_target).square().sum(dim=1).mean()
                )
            )
            if paired_pred.numel()
            else float("nan")
        )
        row = {
            "scene_id": scene_id,
            "clip": clip_name,
            "variant": variant,
            "map_policy": map_policy,
            "gt_instance_id": int(instance_id),
            "gt_label": str(gt_labels[target_index]),
            "slot": int(slot),
            "track_id": int(track_ids[slot]) if slot >= 0 else -1,
            "matched": int(slot >= 0),
            "map_write_frames": int(writes[:, slot].sum()) if slot >= 0 else 0,
            "predicted_points": int(predicted_object_points.shape[0]),
            "target_points": int(target_object_points.shape[0]),
            "paired_points": int(paired_pred.shape[0]),
            "paired_rmse_m": paired_rmse,
            **metrics,
        }
        row["object_recall_iou25"] = int(
            float(row["voxel_iou"]) >= 0.25
        )
        row["object_recall_iou50"] = int(
            float(row["voxel_iou"]) >= 0.50
        )
        object_rows.append(row)

    duplicate_tracks = 0
    duplicate_rows: list[dict[str, object]] = []
    for slot in range(tracks):
        if slot in assigned_slots:
            continue
        pred_support = (
            masks[:, slot]
            & writes[:, slot, None, None]
            & (scores[:, slot, None, None]
               >= float(config.track_score_threshold))
            & confidence_valid
            & finite_pred
        )
        candidate_points = deterministic_limit_points(
            points[pred_support],
            config.max_points_per_object,
        )
        candidate_voxels = voxel_keys(
            candidate_points,
            voxel_size=config.voxel_size_m,
        )
        best_iou = 0.0
        best_target = -1
        for target, target_set in target_voxels.items():
            current = set_iou(candidate_voxels, target_set)
            if current > best_iou:
                best_iou = current
                best_target = target
        duplicate = (
            best_target >= 0
            and int(gt_instance_ids[best_target]) in assignment_by_target
            and best_iou >= float(config.duplicate_voxel_iou)
        )
        duplicate_tracks += int(duplicate)
        if candidate_points.numel():
            duplicate_rows.append(
                {
                    "scene_id": scene_id,
                    "clip": clip_name,
                    "variant": variant,
                    "map_policy": map_policy,
                    "slot": int(slot),
                    "track_id": int(track_ids[slot]),
                    "predicted_points": int(candidate_points.shape[0]),
                    "best_gt_instance_id": (
                        int(gt_instance_ids[best_target])
                        if best_target >= 0
                        else -1
                    ),
                    "best_voxel_iou": float(best_iou),
                    "duplicate_object": int(duplicate),
                }
            )

    summary = _summarize_rows(
        object_rows,
        scene_id=scene_id,
        clip_name=clip_name,
        variant=variant,
        map_policy=map_policy,
        duplicate_tracks=duplicate_tracks,
        unmatched_nonempty_tracks=len(duplicate_rows),
    )
    return {
        "summary": summary,
        "object_rows": object_rows,
        "duplicate_rows": duplicate_rows,
    }


def object_point_metrics(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    fscore_thresholds: Sequence[float],
    voxel_size: float,
    ghost_distance: float,
    chunk_size: int,
) -> dict[str, float]:
    predicted = predicted.detach().float().cpu()
    target = target.detach().float().cpu()
    output: dict[str, float] = {}
    if not predicted.numel() or not target.numel():
        output.update(
            {
                "object_accuracy_m": float("nan"),
                "object_completeness_m": float("nan"),
                "symmetric_distance_m": float("nan"),
                "ghost_point_ratio": (
                    1.0 if predicted.numel() and not target.numel() else 0.0
                ),
                "voxel_iou": 0.0,
            }
        )
        for threshold in fscore_thresholds:
            suffix = _distance_suffix(threshold)
            output[f"precision_{suffix}"] = 0.0
            output[f"recall_{suffix}"] = 0.0
            output[f"fscore_{suffix}"] = 0.0
        return output
    pred_to_target = nearest_distances(
        predicted,
        target,
        chunk_size=chunk_size,
    )
    target_to_pred = nearest_distances(
        target,
        predicted,
        chunk_size=chunk_size,
    )
    accuracy = float(pred_to_target.mean())
    completeness = float(target_to_pred.mean())
    output.update(
        {
            "object_accuracy_m": accuracy,
            "object_completeness_m": completeness,
            "symmetric_distance_m": 0.5 * (accuracy + completeness),
            "ghost_point_ratio": float(
                (pred_to_target > float(ghost_distance)).float().mean()
            ),
            "voxel_iou": set_iou(
                voxel_keys(predicted, voxel_size=voxel_size),
                voxel_keys(target, voxel_size=voxel_size),
            ),
        }
    )
    for threshold in fscore_thresholds:
        precision = float((pred_to_target <= float(threshold)).float().mean())
        recall = float((target_to_pred <= float(threshold)).float().mean())
        fscore = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0.0
            else 0.0
        )
        suffix = _distance_suffix(threshold)
        output[f"precision_{suffix}"] = precision
        output[f"recall_{suffix}"] = recall
        output[f"fscore_{suffix}"] = fscore
    return output


def nearest_distances(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    chunk_size: int,
) -> torch.Tensor:
    if not source.numel() or not target.numel():
        raise ValueError("Nearest distances require non-empty point sets.")
    output = []
    for start in range(0, source.shape[0], int(chunk_size)):
        distance = torch.cdist(
            source[start : start + int(chunk_size)],
            target,
        )
        output.append(distance.min(dim=1).values)
    return torch.cat(output)


def deterministic_limit_points(points: torch.Tensor, limit: int) -> torch.Tensor:
    values = points.detach().float().cpu().reshape(-1, 3)
    values = values[torch.isfinite(values).all(dim=1)]
    if values.shape[0] <= int(limit):
        return values
    indices = torch.linspace(
        0,
        values.shape[0] - 1,
        steps=int(limit),
    ).round().long()
    return values.index_select(0, indices)


def deterministic_limit_pairs(
    left: torch.Tensor,
    right: torch.Tensor,
    limit: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    left = left.detach().float().cpu().reshape(-1, 3)
    right = right.detach().float().cpu().reshape(-1, 3)
    valid = torch.isfinite(left).all(dim=1) & torch.isfinite(right).all(dim=1)
    left, right = left[valid], right[valid]
    if left.shape[0] <= int(limit):
        return left, right
    indices = torch.linspace(
        0,
        left.shape[0] - 1,
        steps=int(limit),
    ).round().long()
    return left.index_select(0, indices), right.index_select(0, indices)


def voxel_keys(
    points: torch.Tensor,
    *,
    voxel_size: float,
) -> set[tuple[int, int, int]]:
    if not points.numel():
        return set()
    quantized = torch.floor(points.detach().cpu() / float(voxel_size)).long()
    return {tuple(int(value) for value in row) for row in quantized.tolist()}


def set_iou(
    left: set[tuple[int, int, int]],
    right: set[tuple[int, int, int]],
) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _summarize_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    scene_id: str,
    clip_name: str,
    variant: str,
    map_policy: str,
    duplicate_tracks: int,
    unmatched_nonempty_tracks: int,
) -> dict[str, object]:
    def mean(name: str) -> float:
        values = [
            float(row[name])
            for row in rows
            if math.isfinite(float(row[name]))
        ]
        return sum(values) / len(values) if values else float("nan")

    return {
        "scene_id": scene_id,
        "clip": clip_name,
        "variant": variant,
        "map_policy": map_policy,
        "eligible_gt_objects": int(len(rows)),
        "matched_objects": int(sum(int(row["matched"]) for row in rows)),
        "object_accuracy_m": mean("object_accuracy_m"),
        "object_completeness_m": mean("object_completeness_m"),
        "symmetric_distance_m": mean("symmetric_distance_m"),
        "paired_rmse_m": mean("paired_rmse_m"),
        "fscore_5cm": mean("fscore_5cm"),
        "fscore_10cm": mean("fscore_10cm"),
        "ghost_point_ratio": mean("ghost_point_ratio"),
        "voxel_iou_5cm": mean("voxel_iou"),
        "object_recall_iou25": (
            sum(int(row["object_recall_iou25"]) for row in rows) / len(rows)
            if rows
            else float("nan")
        ),
        "object_recall_iou50": (
            sum(int(row["object_recall_iou50"]) for row in rows) / len(rows)
            if rows
            else float("nan")
        ),
        "duplicate_objects": int(duplicate_tracks),
        "unmatched_nonempty_tracks": int(unmatched_nonempty_tracks),
        "duplicate_object_rate": (
            duplicate_tracks / unmatched_nonempty_tracks
            if unmatched_nonempty_tracks
            else 0.0
        ),
    }


def _distance_suffix(value: float) -> str:
    centimeters = int(round(100.0 * float(value)))
    return f"{centimeters}cm"


def _validate_config(config: SemanticMapMetricConfig) -> None:
    if config.max_points_per_object < 1 or config.distance_chunk_size < 1:
        raise ValueError("Map metric point/chunk limits must be positive.")
    if config.voxel_size_m <= 0.0 or config.ghost_distance_m <= 0.0:
        raise ValueError("Map metric distances must be positive.")
    if not config.fscore_thresholds_m or any(
        value <= 0.0 for value in config.fscore_thresholds_m
    ):
        raise ValueError("Map metric F-score thresholds must be positive.")
    for name, value in (
        ("confidence_threshold", config.confidence_threshold),
        ("track_score_threshold", config.track_score_threshold),
        ("duplicate_voxel_iou", config.duplicate_voxel_iou),
    ):
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"Map metric {name} must be in [0,1].")
