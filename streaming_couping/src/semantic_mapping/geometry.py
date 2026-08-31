"""Resolution and pose-neutral geometry helpers for semantic mapping."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F

from .contracts import GeometryFrame, SizeHW


def world_points_for_frame(
    frame: GeometryFrame,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``[H,W,3]`` world points and a finite/valid pixel mask.

    A backend that already predicts a world pointmap takes the fast path.  A
    backend that exposes only depth supplies the fallback path through the
    canonical camera-to-world pose and intrinsics.  No model-specific pose
    convention is handled here; adapters must convert into ``camera_to_world``
    before constructing ``GeometryFrame``.
    """

    frame.validate()
    height, width = frame.image_size
    if frame.world_points is not None:
        points = frame.world_points.detach().float().cpu()
        valid = torch.isfinite(points).all(dim=-1)
        if frame.valid is not None:
            valid &= frame.valid.detach().bool().cpu()
        return points, valid

    if frame.depth is None or frame.intrinsics is None or frame.camera_to_world is None:
        raise ValueError(
            "Depth backprojection requires depth, intrinsics, and camera_to_world."
        )
    depth = frame.depth.detach().float().cpu()
    if depth.ndim == 3:
        depth = depth[..., 0]
    intrinsics = frame.intrinsics.detach().float().cpu()
    pose = frame.camera_to_world.detach().float().cpu()
    if pose.shape == (3, 4):
        homogeneous = torch.eye(4, dtype=pose.dtype)
        homogeneous[:3] = pose
        pose = homogeneous

    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=depth.dtype),
        torch.arange(width, dtype=depth.dtype),
        indexing="ij",
    )
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    if float(fx.abs()) <= 1e-8 or float(fy.abs()) <= 1e-8:
        raise ValueError("Intrinsics must have non-zero focal lengths.")
    camera = torch.stack(
        (
            (xx - cx) / fx * depth,
            (yy - cy) / fy * depth,
            depth,
        ),
        dim=-1,
    )
    rotation = pose[:3, :3]
    translation = pose[:3, 3]
    points = torch.einsum("ij,hwj->hwi", rotation, camera) + translation
    valid = torch.isfinite(points).all(dim=-1) & torch.isfinite(depth) & (depth > 0.0)
    if frame.valid is not None:
        valid &= frame.valid.detach().bool().cpu()
    return points, valid


def geometry_confidence_for_frame(frame: GeometryFrame) -> torch.Tensor:
    """Return a finite confidence map in ``[0,1]`` for one geometry frame."""

    height, width = frame.image_size
    if frame.confidence is None:
        confidence = torch.ones(height, width, dtype=torch.float32)
    else:
        confidence = frame.confidence.detach().float().cpu()
    return torch.nan_to_num(confidence, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)


def rgb_for_frame(frame: GeometryFrame) -> torch.Tensor | None:
    """Normalize an optional RGB image to ``[H,W,3]`` in ``[0,1]``."""

    if frame.rgb is None:
        return None
    rgb = frame.rgb.detach().float().cpu()
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("Canonical GeometryFrame.rgb must have shape [H,W,3].")
    return rgb.clamp(0.0, 1.0)


def resize_bool_mask(mask: torch.Tensor, output_size: Sequence[int]) -> torch.Tensor:
    """Resize a binary mask with nearest-neighbor interpolation."""

    output_size = _size(output_size)
    value = torch.as_tensor(mask).detach().cpu()
    if value.ndim != 2:
        raise ValueError(f"Expected a [H,W] mask, got {tuple(value.shape)}.")
    if tuple(value.shape) == output_size:
        return value.bool()
    resized = F.interpolate(
        value.float()[None, None],
        size=output_size,
        mode="nearest",
    )[0, 0]
    return resized.bool()


def resize_rgb(rgb: torch.Tensor, output_size: Sequence[int]) -> torch.Tensor:
    """Resize an ``[H,W,3]`` RGB image with bilinear interpolation."""

    output_size = _size(output_size)
    value = torch.as_tensor(rgb).detach().float().cpu()
    if value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError(f"Expected an [H,W,3] image, got {tuple(value.shape)}.")
    if tuple(value.shape[:2]) == output_size:
        return value.clamp(0.0, 1.0)
    resized = F.interpolate(
        value.permute(2, 0, 1)[None],
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )[0]
    return resized.permute(1, 2, 0).clamp(0.0, 1.0)


def _size(value: Sequence[int]) -> SizeHW:
    if len(value) != 2:
        raise ValueError(f"Expected (height, width), got {value!r}.")
    height, width = int(value[0]), int(value[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"Image dimensions must be positive, got {(height, width)}.")
    return height, width
