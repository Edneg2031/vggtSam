"""Shared, model-independent utilities for long-sequence baseline runs."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def natural_sort_key(path: str | Path) -> tuple[tuple[int, int | str], ...]:
    """Return a case-insensitive key that orders frame2 before frame10."""

    text = Path(path).name.casefold()
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part)
        for part in re.split(r"(\d+)", text)
        if part
    )


def discover_images(image_dir: str | Path, max_frames: int = 0) -> list[Path]:
    image_dir = Path(image_dir).expanduser().resolve()
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    images = sorted(
        (
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
        ),
        key=natural_sort_key,
    )
    if max_frames > 0:
        images = images[:max_frames]
    if not images:
        raise RuntimeError(f"No jpg/jpeg/png images found in {image_dir}")
    return images


def write_image_list(path: str | Path, images: Sequence[str | Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{Path(item).resolve()}\n" for item in images))


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default)
        + "\n"
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON")


def write_w2c_txt(
    path: str | Path,
    extrinsics: np.ndarray,
    frame_ids: Sequence[str | int],
) -> None:
    extrinsics = np.asarray(extrinsics)
    if extrinsics.shape != (len(frame_ids), 3, 4):
        raise ValueError(
            "Expected extrinsics [frames, 3, 4], got "
            f"{extrinsics.shape} for {len(frame_ids)} frame ids"
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write("# w2c\n")
        for frame_id, matrix in zip(frame_ids, extrinsics):
            values = [frame_id, *matrix[:3, :3].reshape(-1), *matrix[:3, 3]]
            handle.write(" ".join(str(value) for value in values) + "\n")


def write_intrinsics_txt(
    path: str | Path,
    intrinsics: np.ndarray,
    frame_ids: Sequence[str | int],
) -> None:
    intrinsics = np.asarray(intrinsics)
    if intrinsics.shape != (len(frame_ids), 3, 3):
        raise ValueError(
            "Expected intrinsics [frames, 3, 3], got "
            f"{intrinsics.shape} for {len(frame_ids)} frame ids"
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write("# fx fy cx cy\n")
        for frame_id, matrix in zip(frame_ids, intrinsics):
            handle.write(
                f"{frame_id} {float(matrix[0, 0])} {float(matrix[1, 1])} "
                f"{float(matrix[0, 2])} {float(matrix[1, 2])}\n"
            )


def write_binary_ply(
    path: str | Path,
    points: np.ndarray,
    colors: np.ndarray,
) -> None:
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    if points.ndim != 2 or points.shape[1] != 3 or colors.shape != points.shape:
        raise ValueError(
            f"PLY points/colors must both be [N, 3], got {points.shape}/{colors.shape}"
        )
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    colors = colors[finite]
    vertices = np.empty(
        len(points),
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    if len(vertices):
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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        vertices.tofile(handle)


class TemporalPointSampler:
    """Bounded deterministic point collection with coverage over every frame."""

    def __init__(self, max_points: int, total_frames: int) -> None:
        if max_points <= 0 or total_frames <= 0:
            raise ValueError("max_points and total_frames must be positive")
        self.max_points = int(max_points)
        self.per_frame = max(1, math.ceil(max_points / total_frames))
        self._points: list[np.ndarray] = []
        self._colors: list[np.ndarray] = []
        self.seen_points = 0

    def add(self, points: np.ndarray, colors: np.ndarray) -> None:
        points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
        if points.shape != colors.shape:
            raise ValueError(f"Point/color shape mismatch: {points.shape}/{colors.shape}")
        finite = np.isfinite(points).all(axis=1)
        points = points[finite]
        colors = colors[finite]
        self.seen_points += int(len(points))
        if not len(points):
            return
        count = min(self.per_frame, len(points))
        indices = np.linspace(0, len(points) - 1, num=count, dtype=np.int64)
        self._points.append(points[indices])
        self._colors.append(colors[indices])

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._points:
            return (
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.uint8),
            )
        points = np.concatenate(self._points, axis=0)
        colors = np.concatenate(self._colors, axis=0)
        if len(points) > self.max_points:
            indices = np.linspace(
                0, len(points) - 1, num=self.max_points, dtype=np.int64
            )
            points = points[indices]
            colors = colors[indices]
        return points, colors


def save_depth_visualization(path: str | Path, depth: np.ndarray) -> None:
    depth = np.asarray(depth, dtype=np.float32).squeeze()
    valid = np.isfinite(depth) & (depth > 0)
    normalized = np.zeros_like(depth, dtype=np.float32)
    if np.any(valid):
        low, high = np.percentile(depth[valid], [2.0, 98.0])
        if high <= low:
            high = low + 1e-6
        normalized[valid] = np.clip((depth[valid] - low) / (high - low), 0, 1)
    try:
        from matplotlib import colormaps

        rgb = (colormaps["plasma"](normalized)[..., :3] * 255).astype(np.uint8)
    except ImportError:
        rgb = np.repeat((normalized[..., None] * 255).astype(np.uint8), 3, axis=2)
    rgb[~valid] = 0
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path)


def save_rgb(path: str | Path, image_chw: np.ndarray) -> np.ndarray:
    image = np.asarray(image_chw)
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected RGB image [3, H, W], got {image.shape}")
    rgb = np.clip(image.transpose(1, 2, 0), 0, 1)
    rgb = (rgb * 255.0).round().astype(np.uint8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path)
    return rgb


def camera_centers_from_w2c(extrinsics: np.ndarray) -> np.ndarray:
    extrinsics = np.asarray(extrinsics, dtype=np.float64)
    rotations = extrinsics[:, :3, :3]
    translations = extrinsics[:, :3, 3]
    return -np.einsum("nij,nj->ni", rotations.transpose(0, 2, 1), translations)


def save_trajectory_plot(path: str | Path, extrinsics: np.ndarray, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    centers = camera_centers_from_w2c(extrinsics)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(10, 4.5))
    axis_3d = figure.add_subplot(1, 2, 1, projection="3d")
    axis_2d = figure.add_subplot(1, 2, 2)
    color = np.arange(len(centers))
    axis_3d.plot(centers[:, 0], centers[:, 1], centers[:, 2], linewidth=1)
    axis_3d.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c=color, s=8)
    axis_3d.set_xlabel("X")
    axis_3d.set_ylabel("Y")
    axis_3d.set_zlabel("Z")
    axis_3d.set_title("3D camera centers")
    axis_2d.plot(centers[:, 0], centers[:, 2], linewidth=1)
    axis_2d.scatter(centers[:, 0], centers[:, 2], c=color, s=8)
    axis_2d.set_xlabel("X")
    axis_2d.set_ylabel("Z")
    axis_2d.set_aspect("equal", adjustable="datalim")
    axis_2d.grid(alpha=0.25)
    axis_2d.set_title("Top view (X-Z)")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
