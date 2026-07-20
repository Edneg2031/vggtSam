"""Evaluation-only metrics for tracker-conditioned StreamVGGT object maps.

The method never consumes GT geometry. For evaluation, one fixed Sim(3) is
estimated from paired full-scene points on the reference frame and then held
constant for every later frame, tracking mode, and map-update gate.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from test_sam.coordinates import streamvggt_image_transform
from test_sam.data import resolve_manifest_path
from vggtsam.data.scannetpp.object_sequence import read_pointmap

from .config import ExperimentConfig
from .recovery import mine_recovery, output_mask_to_stream
from .types import GeometrySequence, TrackingSequence


@dataclass(frozen=True)
class MapEvaluationContext:
    aligned_world_points: torch.Tensor
    gt_pointmaps: torch.Tensor
    sim3_scale: float
    sim3_rotation: torch.Tensor
    sim3_translation: torch.Tensor
    sim3_inliers: int
    sim3_rmse: float


def prepare_map_evaluation(
    config: ExperimentConfig,
    *,
    scene_id: str,
    frame_indices: Sequence[int],
    geometry: GeometrySequence,
    reference_frame_idx: int,
) -> MapEvaluationContext:
    """Load metric GT and align StreamVGGT once using the reference frame."""

    gt_pointmaps = _load_gt_pointmaps(
        config.manifest,
        scene_id=scene_id,
        frame_indices=frame_indices,
        processed_size=geometry.processed_size,
        image_mode=config.image_mode,
    )
    reference_frame_idx = int(reference_frame_idx)
    source = geometry.world_points[reference_frame_idx].reshape(-1, 3)
    target = gt_pointmaps[reference_frame_idx].reshape(-1, 3)
    confidence = geometry.confidence[reference_frame_idx].reshape(-1)
    valid = (
        torch.isfinite(source).all(dim=-1)
        & torch.isfinite(target).all(dim=-1)
        & torch.isfinite(confidence)
        & (confidence >= config.point_cloud_confidence_threshold)
    )
    source = source[valid]
    target = target[valid]
    source, target = _paired_limit(
        source,
        target,
        max_points=max(30_000, config.map_metric_max_points),
    )
    scale, rotation, translation, inliers, rmse = _robust_similarity(
        source,
        target,
        min_points=128,
    )
    aligned = scale * (
        geometry.world_points.float() @ rotation.T
    ) + translation
    return MapEvaluationContext(
        aligned_world_points=aligned,
        gt_pointmaps=gt_pointmaps,
        sim3_scale=scale,
        sim3_rotation=rotation,
        sim3_translation=translation,
        sim3_inliers=inliers,
        sim3_rmse=rmse,
    )


def evaluate_instance_maps(
    config: ExperimentConfig,
    *,
    context: MapEvaluationContext,
    sequence,
    target_masks: torch.Tensor,
    geometry: GeometrySequence,
    predictions: Mapping[str, TrackingSequence],
    event_policy: str,
) -> list[dict]:
    """Compare map purity/completeness for tracking and update-gate ablations."""

    gt_grid_masks = _masks_to_geometry_grid(
        target_masks,
        geometry=geometry,
        image_mode=config.image_mode,
    )
    gt_points, gt_observations, gt_before_limit = _collect_points(
        context.gt_pointmaps,
        gt_grid_masks,
        frame_keep=torch.ones(len(target_masks), dtype=torch.bool),
        confidence=None,
        confidence_threshold=0.0,
        max_points=config.map_metric_max_points,
    )
    rows = []
    prepared_tracking: dict[
        int,
        tuple[torch.Tensor, dict[str, torch.Tensor]],
    ] = {}
    evaluated_maps: dict[tuple[int, str], dict] = {}
    for mode, tracking in predictions.items():
        tracking_key = id(tracking)
        prepared = prepared_tracking.get(tracking_key)
        if prepared is None:
            grid_masks = _masks_to_geometry_grid(
                tracking.masks,
                geometry=geometry,
                image_mode=config.image_mode,
            )
            keep_variants = {
                "all_frames": torch.ones(
                    len(sequence.frame_indices),
                    dtype=torch.bool,
                ),
                "score_gate": (
                    tracking.scores.detach().cpu()
                    >= config.map_update_min_score
                ),
            }
            keep_variants["score_gate"][sequence.reference_frame_idx] = True
            joint = mine_recovery(
                config,
                sequence=sequence,
                target_masks=target_masks,
                original_masks=tracking.masks,
                original_scores=tracking.scores,
                geometry=geometry,
                map_update_policy="joint_reliable",
            )
            keep_variants["joint_gate"] = torch.tensor(
                [
                    index == int(sequence.reference_frame_idx)
                    or bool(row["map_updated"])
                    for index, row in enumerate(joint["rows"])
                ],
                dtype=torch.bool,
            )
            prepared_tracking[tracking_key] = (grid_masks, keep_variants)
        else:
            grid_masks, keep_variants = prepared
        for map_gate, keep in keep_variants.items():
            cache_key = (tracking_key, map_gate)
            cached = evaluated_maps.get(cache_key)
            if cached is None:
                cached = _evaluate_one_map(
                    config,
                    context=context,
                    sequence=sequence,
                    instance_label=sequence.label,
                    event_policy=event_policy,
                    tracking_mode=mode,
                    map_gate=map_gate,
                    grid_masks=grid_masks,
                    frame_keep=keep,
                    gt_points=gt_points,
                    gt_observations=gt_observations,
                    gt_before_limit=gt_before_limit,
                    geometry=geometry,
                )
                evaluated_maps[cache_key] = cached
                rows.append(cached)
            else:
                duplicate = dict(cached)
                duplicate["tracking_mode"] = mode
                rows.append(duplicate)

    rows.append(
        _evaluate_one_map(
            config,
            context=context,
            sequence=sequence,
            instance_label=sequence.label,
            event_policy=event_policy,
            tracking_mode="gt_mask_oracle",
            map_gate="all_frames",
            grid_masks=gt_grid_masks,
            frame_keep=torch.ones(len(target_masks), dtype=torch.bool),
            gt_points=gt_points,
            gt_observations=gt_observations,
            gt_before_limit=gt_before_limit,
            geometry=geometry,
        )
    )

    original = predictions["original"]
    shuffled_masks = original.masks.clone()
    movable = [
        index
        for index in range(len(shuffled_masks))
        if index != int(sequence.reference_frame_idx)
    ]
    if len(movable) > 1:
        source_indices = movable[1:] + movable[:1]
        shuffled_masks[movable] = original.masks[source_indices]
    shuffled_grid = _masks_to_geometry_grid(
        shuffled_masks,
        geometry=geometry,
        image_mode=config.image_mode,
    )
    rows.append(
        _evaluate_one_map(
            config,
            context=context,
            sequence=sequence,
            instance_label=sequence.label,
            event_policy=event_policy,
            tracking_mode="time_shuffled_original_masks",
            map_gate="negative_control",
            grid_masks=shuffled_grid,
            frame_keep=torch.ones(len(target_masks), dtype=torch.bool),
            gt_points=gt_points,
            gt_observations=gt_observations,
            gt_before_limit=gt_before_limit,
            geometry=geometry,
        )
    )
    return rows


def _evaluate_one_map(
    config: ExperimentConfig,
    *,
    context: MapEvaluationContext,
    sequence,
    instance_label: str,
    event_policy: str,
    tracking_mode: str,
    map_gate: str,
    grid_masks: torch.Tensor,
    frame_keep: torch.Tensor,
    gt_points: torch.Tensor,
    gt_observations: int,
    gt_before_limit: int,
    geometry: GeometrySequence,
) -> dict:
    points, observations, before_limit = _collect_points(
        context.aligned_world_points,
        grid_masks,
        frame_keep=frame_keep,
        confidence=geometry.confidence,
        confidence_threshold=config.point_cloud_confidence_threshold,
        max_points=config.map_metric_max_points,
    )
    metrics = _surface_metrics(
        points,
        gt_points,
        thresholds=config.map_metric_thresholds,
    )
    return {
        "scene_id": sequence.scene_id,
        "instance_id": int(sequence.instance_id),
        "instance_label": instance_label,
        "event_policy": event_policy,
        "tracking_mode": tracking_mode,
        "map_gate": map_gate,
        "kept_frame_indices": " ".join(
            str(sequence.frame_indices[index])
            for index in frame_keep.nonzero(as_tuple=False).flatten().tolist()
        ),
        "observation_frames": observations,
        "selected_points_before_limit": before_limit,
        "evaluated_points": int(points.shape[0]),
        "gt_observation_frames": gt_observations,
        "gt_points_before_limit": gt_before_limit,
        "evaluated_gt_points": int(gt_points.shape[0]),
        "reference_sim3_scale": context.sim3_scale,
        "reference_sim3_inliers": context.sim3_inliers,
        "reference_sim3_rmse": context.sim3_rmse,
        "gt_geometry_role": "evaluation_only",
        **metrics,
    }


def _masks_to_geometry_grid(
    masks: torch.Tensor,
    *,
    geometry: GeometrySequence,
    image_mode: str,
) -> torch.Tensor:
    return torch.stack(
        [
            output_mask_to_stream(
                mask,
                source_size=geometry.source_sizes[index],
                processed_size=geometry.processed_size,
                image_mode=image_mode,
            )
            for index, mask in enumerate(masks)
        ]
    ).bool()


def _collect_points(
    pointmaps: torch.Tensor,
    masks: torch.Tensor,
    *,
    frame_keep: torch.Tensor,
    confidence: torch.Tensor | None,
    confidence_threshold: float,
    max_points: int,
) -> tuple[torch.Tensor, int, int]:
    chunks = []
    observations = 0
    for index in range(pointmaps.shape[0]):
        if not bool(frame_keep[index]):
            continue
        valid = masks[index] & torch.isfinite(pointmaps[index]).all(dim=-1)
        if confidence is not None:
            valid &= (
                torch.isfinite(confidence[index])
                & (confidence[index] >= float(confidence_threshold))
            )
        selected = pointmaps[index][valid].detach().float().cpu()
        if selected.numel():
            chunks.append(selected)
            observations += 1
    points = (
        torch.cat(chunks, dim=0)
        if chunks
        else torch.empty((0, 3), dtype=torch.float32)
    )
    before_limit = int(points.shape[0])
    return (
        _limit_points(points, max_points=max_points),
        observations,
        before_limit,
    )


def _surface_metrics(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    thresholds: Sequence[float],
) -> dict[str, float]:
    if predicted.numel() == 0 or target.numel() == 0:
        result = {
            "pred_to_gt_mean": float("nan"),
            "gt_to_pred_mean": float("nan"),
            "chamfer_l1": float("nan"),
        }
        for threshold in thresholds:
            suffix = _threshold_suffix(threshold)
            result[f"precision_{suffix}"] = 0.0
            result[f"recall_{suffix}"] = 0.0
            result[f"fscore_{suffix}"] = 0.0
        return result
    pred_distance = _nearest_distances(predicted, target)
    target_distance = _nearest_distances(target, predicted)
    result = {
        "pred_to_gt_mean": float(pred_distance.mean()),
        "gt_to_pred_mean": float(target_distance.mean()),
        "chamfer_l1": float(
            0.5 * (pred_distance.mean() + target_distance.mean())
        ),
    }
    for threshold in thresholds:
        precision = float((pred_distance <= float(threshold)).float().mean())
        recall = float((target_distance <= float(threshold)).float().mean())
        fscore = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0.0
            else 0.0
        )
        suffix = _threshold_suffix(threshold)
        result[f"precision_{suffix}"] = precision
        result[f"recall_{suffix}"] = recall
        result[f"fscore_{suffix}"] = fscore
    return result


def _nearest_distances(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    chunk_size: int = 512,
) -> torch.Tensor:
    values = []
    for start in range(0, source.shape[0], chunk_size):
        distances = torch.cdist(source[start : start + chunk_size], target)
        values.append(distances.min(dim=1).values)
    return torch.cat(values)


def _load_gt_pointmaps(
    manifest_path: str | Path,
    *,
    scene_id: str,
    frame_indices: Sequence[int],
    processed_size: tuple[int, int],
    image_mode: str,
) -> torch.Tensor:
    manifest_path = Path(manifest_path).expanduser().resolve()
    with manifest_path.open("r", encoding="utf8") as handle:
        manifest = json.load(handle)
    scene = next(
        (
            item
            for item in manifest.get("scenes", [])
            if item.get("scene_id") == scene_id
        ),
        None,
    )
    if scene is None:
        raise ValueError(f"Scene {scene_id!r} is missing from {manifest_path}.")
    frames = scene.get("frames", [])
    pointmaps = []
    for index in frame_indices:
        value = frames[int(index)].get("pointmap")
        if not value:
            raise ValueError(
                "Map-quality evaluation requires mesh-rasterized GT pointmaps "
                "in the manifest."
            )
        path = resolve_manifest_path(value, manifest_path)
        pointmaps.append(
            _transform_dense_map(
                read_pointmap(path),
                processed_size,
                image_mode=image_mode,
            )
        )
    return torch.from_numpy(np.stack(pointmaps)).float()


def _transform_dense_map(
    values: np.ndarray,
    output_size: tuple[int, int],
    *,
    image_mode: str,
) -> np.ndarray:
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError(f"Expected pointmap [H,W,3], got {values.shape}.")
    transform = streamvggt_image_transform(values.shape[:2], mode=image_mode)
    output_height, output_width = (int(output_size[0]), int(output_size[1]))
    target_y, target_x = np.meshgrid(
        np.arange(output_height, dtype=np.float32),
        np.arange(output_width, dtype=np.float32),
        indexing="ij",
    )
    native_x = (
        (target_x + 0.5)
        * (transform.target_size[1] / float(output_width))
        - 0.5
    )
    native_y = (
        (target_y + 0.5)
        * (transform.target_size[0] / float(output_height))
        - 0.5
    )
    source_x = (
        (native_x - transform.offset_xy[0] + 0.5)
        / transform.scale_xy[0]
        - 0.5
    )
    source_y = (
        (native_y - transform.offset_xy[1] + 0.5)
        / transform.scale_xy[1]
        - 0.5
    )
    x_index = np.floor(source_x + 0.5).astype(np.int64)
    y_index = np.floor(source_y + 0.5).astype(np.int64)
    valid = (
        (x_index >= 0)
        & (x_index < values.shape[1])
        & (y_index >= 0)
        & (y_index < values.shape[0])
    )
    output = np.full(
        (output_height, output_width, 3),
        np.nan,
        dtype=np.float32,
    )
    output[valid] = values[y_index[valid], x_index[valid]].astype(np.float32)
    return output


def _robust_similarity(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    min_points: int,
    trim_fraction: float = 0.7,
    iterations: int = 4,
) -> tuple[float, torch.Tensor, torch.Tensor, int, float]:
    if source.shape[0] < int(min_points):
        raise ValueError(
            f"Reference Sim(3) needs {min_points} points, got {source.shape[0]}."
        )
    keep = torch.ones(source.shape[0], dtype=torch.bool)
    scale = 1.0
    rotation = torch.eye(3, dtype=source.dtype)
    translation = torch.zeros(3, dtype=source.dtype)
    for _ in range(int(iterations)):
        scale, rotation, translation = _umeyama(source[keep], target[keep])
        residual = torch.linalg.vector_norm(
            scale * (source @ rotation.T) + translation - target,
            dim=-1,
        )
        threshold = torch.quantile(residual, float(trim_fraction))
        next_keep = residual <= threshold
        if int(next_keep.sum()) < int(min_points) or torch.equal(next_keep, keep):
            break
        keep = next_keep
    scale, rotation, translation = _umeyama(source[keep], target[keep])
    residual = torch.linalg.vector_norm(
        scale * (source @ rotation.T) + translation - target,
        dim=-1,
    )
    rmse = float(torch.sqrt((residual[keep] ** 2).mean()))
    return scale, rotation, translation, int(keep.sum()), rmse


def _umeyama(
    source: torch.Tensor,
    target: torch.Tensor,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    source_mean = source.mean(dim=0)
    target_mean = target.mean(dim=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / source.shape[0]
    left, singular_values, right_t = torch.linalg.svd(covariance)
    signs = torch.ones(3, dtype=source.dtype)
    if torch.det(left @ right_t) < 0:
        signs[-1] = -1
    rotation = left @ torch.diag(signs) @ right_t
    variance = source_centered.square().sum(dim=1).mean().clamp_min(1e-12)
    scale = float((singular_values * signs).sum() / variance)
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def _paired_limit(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    max_points: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if source.shape[0] <= int(max_points):
        return source.float().cpu(), target.float().cpu()
    indices = torch.linspace(
        0,
        source.shape[0] - 1,
        steps=int(max_points),
    ).long()
    return (
        source.index_select(0, indices).float().cpu(),
        target.index_select(0, indices).float().cpu(),
    )


def _limit_points(points: torch.Tensor, *, max_points: int) -> torch.Tensor:
    if points.shape[0] <= int(max_points):
        return points
    indices = torch.linspace(
        0,
        points.shape[0] - 1,
        steps=int(max_points),
    ).long()
    return points.index_select(0, indices)


def _threshold_suffix(value: float) -> str:
    centimeters = float(value) * 100.0
    if abs(centimeters - round(centimeters)) < 1e-6:
        return f"{int(round(centimeters))}cm"
    return f"{value:g}".replace(".", "p")
