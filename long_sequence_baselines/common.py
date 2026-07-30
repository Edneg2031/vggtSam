"""Shared, model-independent utilities for long-sequence baseline runs."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

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
        self.seen_points += len(points)
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


def fit_similarity_transform(
    source_points: np.ndarray,
    target_points: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Fit ``target = scale * rotation @ source + translation`` by Umeyama."""

    source = np.asarray(source_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(
            f"Similarity points must share shape [N,3], got {source.shape}/{target.shape}."
        )
    if len(source) < 3 or not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("Similarity fitting needs at least three finite point pairs.")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    source_variance = float(np.square(source_centered).sum() / len(source))
    if source_variance <= np.finfo(np.float64).eps:
        raise ValueError("Similarity source points have zero variance.")
    covariance = target_centered.T @ source_centered / len(source)
    left, singular_values, right_transpose = np.linalg.svd(covariance)
    signs = np.ones(3, dtype=np.float64)
    if np.linalg.det(left @ right_transpose) < 0.0:
        signs[-1] = -1.0
    rotation = left @ np.diag(signs) @ right_transpose
    scale = float(np.dot(singular_values, signs) / source_variance)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"Similarity fit produced invalid scale {scale}.")
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def transform_w2c_world_similarity(
    extrinsics: np.ndarray,
    *,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """Express W2C poses in a world frame related by a fitted Sim(3)."""

    extrinsics = np.asarray(extrinsics, dtype=np.float64)
    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64)
    if extrinsics.ndim != 3 or extrinsics.shape[1:] != (3, 4):
        raise ValueError(f"Expected W2C poses [N,3,4], got {extrinsics.shape}.")
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("Similarity rotation/translation must be [3,3]/[3].")
    centers = camera_centers_from_w2c(extrinsics)
    transformed_centers = scale * (centers @ rotation.T) + translation
    transformed_rotations = extrinsics[:, :3, :3] @ rotation.T
    transformed_translations = -np.einsum(
        "nij,nj->ni",
        transformed_rotations,
        transformed_centers,
    )
    return np.concatenate(
        [transformed_rotations, transformed_translations[..., None]],
        axis=-1,
    )


def _set_equal_3d_axes(axis: Any, xyz: np.ndarray) -> None:
    """Use the same cubic 3-D bounds as HorizonStream's trajectory plot."""

    xyz = np.asarray(xyz, dtype=np.float64)
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(np.maximum(maxs - mins, 1e-6))
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)


def save_trajectory_plot(
    path: str | Path,
    extrinsics: np.ndarray,
    title: str,
    *,
    label: str = "prediction",
) -> None:
    """Draw one W2C trajectory with HorizonStream's prediction-only layout.

    Both long-sequence baselines call this exact function, so camera-center
    conversion, 3-D bounds, projections, colors, and figure size are identical.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    centers = camera_centers_from_w2c(extrinsics)
    if len(centers) < 2:
        raise ValueError("A trajectory plot requires at least two camera poses.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(16, 5), dpi=140)
    axis_3d = figure.add_subplot(1, 3, 1, projection="3d")
    axis_xy = figure.add_subplot(1, 3, 2)
    axis_xz = figure.add_subplot(1, 3, 3)
    color = "#1f77b4"

    axis_3d.plot(
        centers[:, 0],
        centers[:, 1],
        centers[:, 2],
        color=color,
        linewidth=2.0,
        label=label,
    )
    _set_equal_3d_axes(axis_3d, centers)
    axis_3d.set_xlabel("X")
    axis_3d.set_ylabel("Y")
    axis_3d.set_zlabel("Z")
    axis_3d.set_title(title)
    axis_3d.legend(loc="best")
    axis_3d.grid(True)

    axis_xy.plot(
        centers[:, 0], centers[:, 1], color=color, linewidth=2.0, label=label
    )
    axis_xy.set_xlabel("X")
    axis_xy.set_ylabel("Y")
    axis_xy.set_aspect("equal", adjustable="box")
    axis_xy.set_title("XY Projection")
    axis_xy.legend(loc="best")
    axis_xy.grid(True, alpha=0.4)

    axis_xz.plot(
        centers[:, 0], centers[:, 2], color=color, linewidth=2.0, label=label
    )
    axis_xz.set_xlabel("X")
    axis_xz.set_ylabel("Z")
    axis_xz.set_aspect("equal", adjustable="box")
    axis_xz.set_title("XZ Projection")
    axis_xz.legend(loc="best")
    axis_xz.grid(True, alpha=0.4)

    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def save_trajectory_overlay_plot(
    path: str | Path,
    trajectories: dict[str, np.ndarray],
    title: str,
) -> None:
    """Overlay same-gauge W2C trajectories with the Horizon three-panel layout."""

    if not trajectories:
        raise ValueError("At least one trajectory is required.")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    centers = {
        name: camera_centers_from_w2c(extrinsics)
        for name, extrinsics in trajectories.items()
    }
    lengths = {len(value) for value in centers.values()}
    if len(lengths) != 1 or next(iter(lengths)) < 2:
        raise ValueError("Overlay trajectories must share at least two frames.")
    all_centers = np.concatenate(list(centers.values()), axis=0)
    colors = (
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#9467bd",
        "#d62728",
        "#8c564b",
    )
    figure = plt.figure(figsize=(16, 5), dpi=140)
    axis_3d = figure.add_subplot(1, 3, 1, projection="3d")
    axis_xy = figure.add_subplot(1, 3, 2)
    axis_xz = figure.add_subplot(1, 3, 3)
    for index, (name, xyz) in enumerate(centers.items()):
        color = colors[index % len(colors)]
        axis_3d.plot(
            xyz[:, 0], xyz[:, 1], xyz[:, 2], color=color, linewidth=1.8, label=name
        )
        axis_xy.plot(xyz[:, 0], xyz[:, 1], color=color, linewidth=1.8, label=name)
        axis_xz.plot(xyz[:, 0], xyz[:, 2], color=color, linewidth=1.8, label=name)
    _set_equal_3d_axes(axis_3d, all_centers)
    axis_3d.set_title(title)
    axis_3d.set_xlabel("X")
    axis_3d.set_ylabel("Y")
    axis_3d.set_zlabel("Z")
    axis_3d.legend(loc="best")
    axis_3d.grid(True)
    for axis, projection_title, x_label, y_label in (
        (axis_xy, "XY Projection", "X", "Y"),
        (axis_xz, "XZ Projection", "X", "Z"),
    ):
        axis.set_title(projection_title)
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.set_aspect("equal", adjustable="box")
        axis.legend(loc="best")
        axis.grid(True, alpha=0.4)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
