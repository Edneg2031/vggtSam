"""Evaluation-only fixed reference-frame pointmap alignment."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from test_sam.coordinates import streamvggt_image_transform
from test_sam.data import resolve_manifest_path
from vggtsam.data.scannetpp.object_sequence import read_pointmap

from .config import ExperimentConfig
from .types import GeometrySequence


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
    """Fit one reference-frame Sim(3); later frames remain evaluation-only."""

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
    source, target = _paired_limit(
        source[valid],
        target[valid],
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
                "Pointmap evaluation requires mesh-rasterized GT pointmaps."
            )
        pointmaps.append(
            _transform_dense_map(
                read_pointmap(
                    resolve_manifest_path(value, manifest_path)
                ),
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
    output_height, output_width = map(int, output_size)
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
    for _ in range(int(iterations)):
        scale, rotation, translation = _umeyama(
            source[keep],
            target[keep],
        )
        residual = torch.linalg.vector_norm(
            scale * (source @ rotation.T) + translation - target,
            dim=-1,
        )
        next_keep = residual <= torch.quantile(
            residual,
            float(trim_fraction),
        )
        if (
            int(next_keep.sum()) < int(min_points)
            or torch.equal(next_keep, keep)
        ):
            break
        keep = next_keep
    scale, rotation, translation = _umeyama(source[keep], target[keep])
    residual = torch.linalg.vector_norm(
        scale * (source @ rotation.T) + translation - target,
        dim=-1,
    )
    rmse = float(torch.sqrt(residual[keep].square().mean()))
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
