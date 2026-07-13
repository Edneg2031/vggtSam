"""Explicit 2D/3D transforms used by the dual-framework bridge."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from test_sam.coordinates import (
    output_mask_transform,
    streamvggt_image_transform,
    streamvggt_label_to_grid,
)

from .types import ProjectionResult


def normalize_confidence(confidence: torch.Tensor) -> torch.Tensor:
    """Robustly map StreamVGGT's unbounded expp1 confidence to [0, 1]."""

    confidence = torch.nan_to_num(confidence.float(), nan=0.0, posinf=0.0, neginf=0.0)
    if confidence.ndim == 4 and confidence.shape[-1] == 1:
        confidence = confidence[..., 0]
    flat = confidence.flatten(1)
    low = torch.quantile(flat, 0.05, dim=1, keepdim=True)
    high = torch.quantile(flat, 0.95, dim=1, keepdim=True)
    normalized = (flat - low) / (high - low).clamp_min(1e-6)
    return normalized.clamp(0.0, 1.0).reshape_as(confidence)


def source_mask_to_stream(
    mask: np.ndarray,
    processed_size: tuple[int, int],
    *,
    image_mode: str,
) -> torch.Tensor:
    labels = streamvggt_label_to_grid(
        mask.astype(np.uint8),
        processed_size,
        mode=image_mode,
    )
    return torch.from_numpy(labels > 0)


def output_mask_to_stream(
    mask: torch.Tensor,
    *,
    source_size: tuple[int, int],
    processed_size: tuple[int, int],
    image_mode: str,
) -> torch.Tensor:
    source = F.interpolate(
        mask.float()[None, None],
        size=source_size,
        mode="nearest",
    )[0, 0]
    return source_mask_to_stream(
        source.cpu().numpy() > 0.5,
        processed_size,
        image_mode=image_mode,
    )


def mask_geometry_statistics(
    points: torch.Tensor,
    confidence: torch.Tensor,
    mask: torch.Tensor,
    *,
    max_points: int,
) -> tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:
    valid = mask.bool() & torch.isfinite(points).all(dim=-1) & (confidence > 0.0)
    selected_points = points[valid]
    selected_weights = confidence[valid]
    if selected_points.numel() == 0:
        empty = points.new_zeros((0, 3))
        return empty, confidence.new_zeros((0,)), 0.0, points.new_full((3,), float("nan"))
    if selected_points.shape[0] > max_points:
        indices = torch.linspace(
            0,
            selected_points.shape[0] - 1,
            max_points,
            device=selected_points.device,
        ).long()
        selected_points = selected_points.index_select(0, indices)
        selected_weights = selected_weights.index_select(0, indices)
    weight_sum = selected_weights.sum().clamp_min(1e-6)
    centroid = (selected_points * selected_weights[:, None]).sum(dim=0) / weight_sum
    return selected_points, selected_weights, float(selected_weights.mean()), centroid


def project_world_points(
    points: torch.Tensor,
    *,
    world_to_camera: torch.Tensor,
    intrinsics: torch.Tensor,
    source_size: tuple[int, int],
    processed_size: tuple[int, int],
    output_size: tuple[int, int],
    image_mode: str,
    splat_radius: int,
    observed_world_points: torch.Tensor | None = None,
    occlusion_depth_tolerance: float = 0.02,
    occlusion_relative_tolerance: float = 0.05,
) -> torch.Tensor:
    """Project shared-world points through StreamVGGT camera into output pixels."""

    return project_world_points_with_stats(
        points,
        world_to_camera=world_to_camera,
        intrinsics=intrinsics,
        source_size=source_size,
        processed_size=processed_size,
        output_size=output_size,
        image_mode=image_mode,
        splat_radius=splat_radius,
        observed_world_points=observed_world_points,
        occlusion_depth_tolerance=occlusion_depth_tolerance,
        occlusion_relative_tolerance=occlusion_relative_tolerance,
    ).mask


def project_world_points_with_stats(
    points: torch.Tensor,
    *,
    world_to_camera: torch.Tensor,
    intrinsics: torch.Tensor,
    source_size: tuple[int, int],
    processed_size: tuple[int, int],
    output_size: tuple[int, int],
    image_mode: str,
    splat_radius: int,
    observed_world_points: torch.Tensor | None = None,
    occlusion_depth_tolerance: float = 0.02,
    occlusion_relative_tolerance: float = 0.05,
) -> ProjectionResult:
    """Project points and report whether the current pointmap supports them."""

    output = torch.zeros(output_size, dtype=torch.bool, device=points.device)
    input_points = int(points.shape[0])
    if points.numel() == 0:
        return ProjectionResult(output, 0, 0, 0, 0)
    rotation = world_to_camera[:3, :3]
    translation = world_to_camera[:3, 3]
    camera_points = points @ rotation.T + translation
    z = camera_points[:, 2]
    valid = torch.isfinite(camera_points).all(dim=-1) & (z > 1e-5)
    if not valid.any():
        return ProjectionResult(output, input_points, 0, 0, 0)
    camera_points = camera_points[valid]
    z = camera_points[:, 2]
    pixels = camera_points @ intrinsics.T
    x_processed = pixels[:, 0] / pixels[:, 2].clamp_min(1e-6)
    y_processed = pixels[:, 1] / pixels[:, 2].clamp_min(1e-6)

    depth_tested_points = 0
    depth_supported_points = 0
    if observed_world_points is not None:
        observed_camera = (
            observed_world_points.reshape(-1, 3) @ rotation.T + translation
        ).reshape(*observed_world_points.shape[:2], 3)
        observed_depth = observed_camera[..., 2]
        x_depth = x_processed.round().long()
        y_depth = y_processed.round().long()
        depth_inside = (
            (x_depth >= 0)
            & (x_depth < observed_depth.shape[1])
            & (y_depth >= 0)
            & (y_depth < observed_depth.shape[0])
        )
        depth_tested_points = int(depth_inside.sum())
        sampled_depth = z.new_full(z.shape, float("nan"))
        sampled_depth[depth_inside] = observed_depth[
            y_depth[depth_inside], x_depth[depth_inside]
        ]
        tolerance = float(occlusion_depth_tolerance) + (
            float(occlusion_relative_tolerance) * z.abs()
        )
        depth_consistent = (
            depth_inside
            & torch.isfinite(sampled_depth)
            & (sampled_depth > 0)
            & ((z - sampled_depth).abs() <= tolerance)
        )
        depth_supported_points = int(depth_consistent.sum())
        x_processed = x_processed[depth_consistent]
        y_processed = y_processed[depth_consistent]
        if x_processed.numel() == 0:
            return ProjectionResult(
                output,
                input_points,
                0,
                depth_supported_points,
                depth_tested_points,
            )

    stream_transform = streamvggt_image_transform(source_size, mode=image_mode)
    if stream_transform.target_size != tuple(processed_size):
        raise ValueError(
            "StreamVGGT projection size mismatch: "
            f"transform={stream_transform.target_size}, points={processed_size}"
        )
    output_transform = output_mask_transform(source_size, output_size)
    sx, sy = stream_transform.scale_xy
    ox, oy = stream_transform.offset_xy
    x_source = (x_processed - ox + 0.5) / sx - 0.5
    y_source = (y_processed - oy + 0.5) / sy - 0.5
    x_output, y_output = (
        (x_source + 0.5) * output_transform.scale_xy[0] - 0.5 + output_transform.offset_xy[0],
        (y_source + 0.5) * output_transform.scale_xy[1] - 0.5 + output_transform.offset_xy[1],
    )
    x_index = x_output.round().long()
    y_index = y_output.round().long()
    inside = (
        (x_index >= 0)
        & (x_index < output_size[1])
        & (y_index >= 0)
        & (y_index < output_size[0])
    )
    in_frame_points = int(inside.sum())
    output[y_index[inside], x_index[inside]] = True
    if splat_radius > 0:
        kernel = 2 * int(splat_radius) + 1
        output = F.max_pool2d(
            output.float()[None, None],
            kernel_size=kernel,
            stride=1,
            padding=int(splat_radius),
        )[0, 0] > 0
    if observed_world_points is None:
        depth_tested_points = in_frame_points
        depth_supported_points = in_frame_points
    return ProjectionResult(
        output,
        input_points,
        in_frame_points,
        depth_supported_points,
        depth_tested_points,
    )


def centroid_drift(centroids: Sequence[torch.Tensor]) -> float:
    finite = [value for value in centroids if bool(torch.isfinite(value).all())]
    if len(finite) < 2:
        return float("nan")
    reference = finite[0]
    return float(torch.stack([(value - reference).norm() for value in finite[1:]]).mean())
