"""Causal multi-prompt diagnostics for the frozen V0 temporal prior.

This module contains only geometry-side utilities.  It does not know about
SAM, ground truth, or the V0 cache format.  Callers can therefore freeze all
candidate prompts first and open annotations only in a separate evaluation
phase.

The intended prompt families are:

``center``
    One projected object center.
``surface_3`` / ``surface_5``
    Three or five front-surface candidates selected from a historical object
    point cloud with deterministic 2-D farthest-point sampling.
``box``
    A robust quantile bounding box around the projected historical object
    cloud.
``surface_5_depth_gate``
    ``surface_5`` candidates filtered by agreement with the current dense
    depth map.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class ProjectedPointSet:
    """Projection results before any candidate filtering."""

    uv: torch.Tensor
    depth: torch.Tensor
    camera_points: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class SpatialDispersion:
    """Distance statistics for one multi-point prompt event."""

    count: int
    mean_pairwise_px: float
    min_pairwise_px: float
    normalized_mean_pairwise: float


def project_world_points(
    world_points: torch.Tensor,
    world_to_camera: torch.Tensor,
    intrinsics: torch.Tensor,
    image_size: Sequence[int],
    *,
    min_depth: float = 1e-6,
) -> ProjectedPointSet:
    """Project world points with the standard column-vector pinhole model.

    ``world_to_camera`` is a ``[3,4]`` matrix and ``intrinsics`` is ``[3,3]``.
    The returned ``uv`` and ``depth`` retain one row per input point; callers
    can use ``valid_mask`` to preserve provenance while selecting candidates.
    """

    points = _as_float_tensor(world_points, "world_points")
    w2c = _as_float_tensor(world_to_camera, "world_to_camera")
    matrix = _as_float_tensor(intrinsics, "intrinsics")
    if points.ndim != 2 or tuple(points.shape[1:]) != (3,):
        raise ValueError(f"world_points must be [N,3], got {tuple(points.shape)}")
    if tuple(w2c.shape) != (3, 4):
        raise ValueError(f"world_to_camera must be [3,4], got {tuple(w2c.shape)}")
    if tuple(matrix.shape) != (3, 3):
        raise ValueError(f"intrinsics must be [3,3], got {tuple(matrix.shape)}")
    height, width = _image_size(image_size)
    if float(min_depth) <= 0.0:
        raise ValueError("min_depth must be positive.")

    dtype = torch.float64 if points.dtype == torch.float64 else torch.float32
    device = points.device
    points = points.to(device=device, dtype=dtype)
    w2c = w2c.to(device=device, dtype=dtype)
    matrix = matrix.to(device=device, dtype=dtype)
    homogeneous = torch.cat(
        (points, torch.ones_like(points[:, :1])),
        dim=1,
    )
    camera_points = (w2c @ homogeneous.T).T[:, :3]
    depth = camera_points[:, 2]
    projected = (matrix @ camera_points.T).T
    denominator = projected[:, 2:3]
    safe_denominator = torch.where(
        denominator.abs() > float(min_depth),
        denominator,
        torch.full_like(denominator, float(min_depth)),
    )
    uv = projected[:, :2] / safe_denominator
    valid = (
        torch.isfinite(points).all(dim=1)
        & torch.isfinite(camera_points).all(dim=1)
        & torch.isfinite(uv).all(dim=1)
        & (depth > float(min_depth))
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] < float(width))
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] < float(height))
    )
    return ProjectedPointSet(
        uv=uv,
        depth=depth,
        camera_points=camera_points,
        valid_mask=valid,
    )


def select_surface_indices(
    uv: torch.Tensor,
    depth: torch.Tensor,
    valid_mask: torch.Tensor,
    count: int,
    *,
    front_fraction: float = 0.35,
) -> torch.Tensor:
    """Select deterministic, front-surface, spatially spread point indices.

    Points are first restricted to the front ``front_fraction`` of projected
    depths.  Farthest-point sampling in image space then avoids selecting five
    nearly coincident pixels.  This is a causal visibility proxy, not a true
    renderer or an oracle visibility test.
    """

    uv = _as_float_tensor(uv, "uv")
    depth = _as_float_tensor(depth, "depth").reshape(-1)
    valid = valid_mask.bool().reshape(-1)
    if uv.ndim != 2 or tuple(uv.shape[1:]) != (2,):
        raise ValueError(f"uv must be [N,2], got {tuple(uv.shape)}")
    if depth.shape[0] != uv.shape[0] or valid.shape[0] != uv.shape[0]:
        raise ValueError("uv, depth, and valid_mask must share N.")
    if int(count) < 1:
        raise ValueError("count must be positive.")
    if not 0.0 < float(front_fraction) <= 1.0:
        raise ValueError("front_fraction must be in (0,1].")

    valid_indices = torch.nonzero(
        valid & torch.isfinite(uv).all(dim=1) & torch.isfinite(depth),
        as_tuple=False,
    ).flatten()
    if not valid_indices.numel():
        return torch.empty(0, dtype=torch.long, device=uv.device)
    ordered = valid_indices[torch.argsort(depth[valid_indices], stable=True)]
    front_count = max(int(count), int(torch.ceil(torch.tensor(
        float(front_fraction) * ordered.numel(),
        device=uv.device,
    )).item()))
    pool = ordered[: min(front_count, ordered.numel())]
    if pool.numel() <= int(count):
        return pool

    selected = [int(pool[0].item())]
    distances = torch.full(
        (pool.shape[0],),
        float("inf"),
        dtype=uv.dtype,
        device=uv.device,
    )
    for _ in range(1, min(int(count), pool.shape[0])):
        last = uv[selected[-1]]
        distance = (uv[pool] - last).square().sum(dim=1)
        distances = torch.minimum(distances, distance)
        # ``argmax`` is deterministic and the stable depth ordering breaks
        # ties in a reproducible way through the first occurrence.
        next_index = int(torch.argmax(distances).item())
        selected.append(int(pool[next_index].item()))
        distances[next_index] = float("-inf")
    return torch.tensor(selected, dtype=torch.long, device=uv.device)


def projected_bbox(
    uv: torch.Tensor,
    valid_mask: torch.Tensor,
    image_size: Sequence[int],
    *,
    quantile: float = 0.02,
    min_points: int = 1,
) -> torch.Tensor | None:
    """Return a robust ``[x0,y0,x1,y1]`` box for valid projected points."""

    values = _as_float_tensor(uv, "uv")
    valid = valid_mask.bool().reshape(-1)
    if values.ndim != 2 or tuple(values.shape[1:]) != (2,):
        raise ValueError(f"uv must be [N,2], got {tuple(values.shape)}")
    if valid.shape[0] != values.shape[0]:
        raise ValueError("uv and valid_mask must share N.")
    if not 0.0 <= float(quantile) < 0.5:
        raise ValueError("quantile must be in [0,0.5).")
    height, width = _image_size(image_size)
    good = valid & torch.isfinite(values).all(dim=1)
    if int(good.sum()) < int(min_points):
        return None
    points = values[good]
    low = torch.quantile(points, float(quantile), dim=0)
    high = torch.quantile(points, 1.0 - float(quantile), dim=0)
    x0 = low[0].clamp(0.0, float(width - 1))
    y0 = low[1].clamp(0.0, float(height - 1))
    x1 = high[0].clamp(0.0, float(width - 1))
    y1 = high[1].clamp(0.0, float(height - 1))
    if bool(x1 < x0) or bool(y1 < y0):
        return None
    return torch.stack((x0, y0, x1, y1))


def sample_depth_nearest(
    depth_map: torch.Tensor,
    uv: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a dense ``[H,W]`` depth map at ``(u,v)`` with nearest pixels."""

    depth = _as_float_tensor(depth_map, "depth_map")
    points = _as_float_tensor(uv, "uv")
    if depth.ndim != 2:
        raise ValueError(f"depth_map must be [H,W], got {tuple(depth.shape)}")
    if points.ndim != 2 or tuple(points.shape[1:]) != (2,):
        raise ValueError(f"uv must be [N,2], got {tuple(points.shape)}")
    height, width = depth.shape
    finite = torch.isfinite(points).all(dim=1)
    x = torch.round(points[:, 0]).long()
    y = torch.round(points[:, 1]).long()
    in_bounds = finite & (x >= 0) & (x < width) & (y >= 0) & (y < height)
    safe_x = x.clamp(0, width - 1)
    safe_y = y.clamp(0, height - 1)
    sampled = depth[safe_y, safe_x]
    valid = in_bounds & torch.isfinite(sampled) & (sampled > 0)
    return sampled, valid


def depth_consistency_gate(
    projected_depth: torch.Tensor,
    current_depth: torch.Tensor,
    current_valid: torch.Tensor,
    *,
    absolute_tolerance: float = 0.05,
    relative_tolerance: float = 0.10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gate projected points by current-depth agreement.

    A point is retained when

    ``abs(z_history - z_current) <= abs_tol + rel_tol * abs(z_current)``.

    The function returns ``(accepted, absolute_residual, relative_residual)``
    and never consults ground truth.
    """

    history = _as_float_tensor(projected_depth, "projected_depth").reshape(-1)
    current = _as_float_tensor(current_depth, "current_depth").reshape(-1)
    valid = current_valid.bool().reshape(-1)
    if history.shape != current.shape or valid.shape != history.shape:
        raise ValueError("Depth vectors must have the same shape.")
    if float(absolute_tolerance) < 0.0 or float(relative_tolerance) < 0.0:
        raise ValueError("Depth tolerances must be non-negative.")
    absolute = (history - current).abs()
    relative = absolute / current.abs().clamp_min(1e-6)
    accepted = (
        valid
        & torch.isfinite(history)
        & torch.isfinite(current)
        & (history > 0)
        & (current > 0)
        & (
            absolute
            <= float(absolute_tolerance)
            + float(relative_tolerance) * current.abs()
        )
    )
    return accepted, absolute, relative


def spatial_dispersion(
    uv: torch.Tensor,
    image_size: Sequence[int],
) -> SpatialDispersion:
    """Compute pairwise 2-D spacing, normalized by image diagonal."""

    points = _as_float_tensor(uv, "uv")
    if points.ndim != 2 or tuple(points.shape[1:]) != (2,):
        raise ValueError(f"uv must be [N,2], got {tuple(points.shape)}")
    height, width = _image_size(image_size)
    count = int(points.shape[0])
    if count < 2:
        return SpatialDispersion(
            count=count,
            mean_pairwise_px=0.0,
            min_pairwise_px=0.0,
            normalized_mean_pairwise=0.0,
        )
    distances = torch.pdist(points)
    diagonal = max((float(height) ** 2 + float(width) ** 2) ** 0.5, 1e-6)
    return SpatialDispersion(
        count=count,
        mean_pairwise_px=float(distances.mean()),
        min_pairwise_px=float(distances.min()),
        normalized_mean_pairwise=float(distances.mean()) / diagonal,
    )


def harmonic_mean(precision: float, coverage: float) -> float:
    """Return the precision/coverage harmonic mean with zero-safe handling."""

    first = float(precision)
    second = float(coverage)
    if first < 0.0 or second < 0.0:
        raise ValueError("precision and coverage must be non-negative.")
    if first + second <= 0.0:
        return 0.0
    return 2.0 * first * second / (first + second)


def _as_float_tensor(value: torch.Tensor, name: str) -> torch.Tensor:
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    if not value.is_floating_point():
        value = value.float()
    return value


def _image_size(value: Sequence[int]) -> tuple[int, int]:
    values = tuple(int(item) for item in value)
    if len(values) != 2 or values[0] <= 0 or values[1] <= 0:
        raise ValueError(f"image_size must be positive (H,W), got {value!r}.")
    return values[0], values[1]
