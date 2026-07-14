"""Pointmap serialization helpers for geometry experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def save_pointmap_ply(
    path: Path,
    pointmap: torch.Tensor,
    colors: np.ndarray,
    *,
    mask: torch.Tensor | None = None,
    confidence: torch.Tensor | None = None,
    confidence_threshold: float = 0.0,
    max_points: int = 200_000,
) -> int:
    points = pointmap.detach().cpu().numpy().reshape(-1, 3)
    rgb = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    valid = np.isfinite(points).all(axis=-1)
    if mask is not None:
        valid &= mask.detach().cpu().numpy().reshape(-1).astype(bool)
    if confidence is not None:
        valid &= (
            confidence.detach().cpu().numpy().reshape(-1)
            >= float(confidence_threshold)
        )
    points = points[valid]
    rgb = rgb[valid]
    if points.shape[0] > int(max_points):
        indices = np.linspace(0, points.shape[0] - 1, int(max_points)).astype(np.int64)
        points = points[indices]
        rgb = rgb[indices]
    _write_binary_ply(path, points.astype(np.float32), rgb)
    return int(points.shape[0])


def save_aggregate_ply(
    path: Path,
    pointmaps: torch.Tensor,
    colors: np.ndarray,
    *,
    masks: torch.Tensor | None = None,
    confidence: torch.Tensor | None = None,
    confidence_threshold: float = 0.0,
    max_points: int = 400_000,
) -> int:
    points = pointmaps.detach().cpu().numpy().reshape(-1, 3)
    rgb = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    valid = np.isfinite(points).all(axis=-1)
    if masks is not None:
        valid &= masks.detach().cpu().numpy().reshape(-1).astype(bool)
    if confidence is not None:
        valid &= (
            confidence.detach().cpu().numpy().reshape(-1)
            >= float(confidence_threshold)
        )
    points = points[valid]
    rgb = rgb[valid]
    if points.shape[0] > int(max_points):
        indices = np.linspace(0, points.shape[0] - 1, int(max_points)).astype(np.int64)
        points = points[indices]
        rgb = rgb[indices]
    _write_binary_ply(path, points.astype(np.float32), rgb)
    return int(points.shape[0])


def _write_binary_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.empty(
        points.shape[0],
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    vertices["x"], vertices["y"], vertices["z"] = points.T
    vertices["red"], vertices["green"], vertices["blue"] = colors.T
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        vertices.tofile(handle)
