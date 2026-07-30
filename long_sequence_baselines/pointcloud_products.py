"""Fair raw, confidence-filtered, and voxel-fused point-cloud products.

The same implementation is used for HorizonStream and StreamVGGT depth+pose
outputs.  StreamVGGT also feeds its directly-regressed world pointmaps through
the accumulator so point-head quality is not conflated with export filtering.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from .common import (
    TemporalPointSampler,
    natural_sort_key,
    write_binary_ply,
    write_json,
)


@dataclass(frozen=True)
class PointCloudProtocol:
    confidence_percentile: float = 50.0
    depth_percentile_low: float = 1.0
    depth_percentile_high: float = 99.0
    voxel_size_ratio: float = 0.01
    min_voxel_observations: int = 2
    max_points: int = 2_000_000

    def validate(self) -> None:
        if not 0.0 <= self.confidence_percentile <= 100.0:
            raise ValueError("confidence_percentile must be in [0, 100].")
        if not 0.0 <= self.depth_percentile_low < self.depth_percentile_high <= 100.0:
            raise ValueError("Depth percentiles must satisfy 0 <= low < high <= 100.")
        if self.voxel_size_ratio <= 0.0:
            raise ValueError("voxel_size_ratio must be positive.")
        if self.min_voxel_observations <= 0:
            raise ValueError("min_voxel_observations must be positive.")
        if self.max_points <= 0:
            raise ValueError("max_points must be positive.")


@dataclass
class _VoxelStats:
    coordinates: np.ndarray
    position_sum: np.ndarray
    color_sum: np.ndarray
    weight_sum: np.ndarray
    observations: np.ndarray


class WeightedVoxelFusion:
    """Confidence-weighted voxel averaging with per-frame observation counts."""

    def __init__(self, voxel_size: float) -> None:
        if voxel_size <= 0.0:
            raise ValueError("voxel_size must be positive.")
        self.voxel_size = float(voxel_size)
        self._levels: list[_VoxelStats | None] = []

    def add_frame(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        weights: np.ndarray,
    ) -> None:
        points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
        weights = np.asarray(weights, dtype=np.float32).reshape(-1)
        if points.shape != colors.shape or len(points) != len(weights):
            raise ValueError("Voxel points, colors, and weights disagree.")
        if not len(points):
            return
        finite = np.isfinite(points).all(axis=1) & np.isfinite(weights) & (weights > 0)
        if not np.any(finite):
            return
        stats = _aggregate_frame_voxels(
            points[finite],
            colors[finite],
            weights[finite],
            self.voxel_size,
        )
        self._insert(stats)

    def _insert(self, stats: _VoxelStats) -> None:
        level = 0
        while True:
            if level == len(self._levels):
                self._levels.append(stats)
                return
            existing = self._levels[level]
            if existing is None:
                self._levels[level] = stats
                return
            self._levels[level] = None
            stats = _merge_voxel_stats((existing, stats))
            level += 1

    def arrays(
        self,
        *,
        min_observations: int,
        max_points: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        active = [stats for stats in self._levels if stats is not None]
        if not active:
            return (
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.uint8),
                np.empty((0,), dtype=np.int32),
            )
        merged = _merge_voxel_stats(active)
        keep = merged.observations >= int(min_observations)
        weight = merged.weight_sum[keep].clip(min=1e-8)
        points = merged.position_sum[keep] / weight[:, None]
        colors = merged.color_sum[keep] / weight[:, None]
        observations = merged.observations[keep]
        if len(points) > int(max_points):
            indices = np.linspace(0, len(points) - 1, num=int(max_points), dtype=np.int64)
            points = points[indices]
            colors = colors[indices]
            observations = observations[indices]
        return (
            points.astype(np.float32, copy=False),
            np.clip(np.rint(colors), 0, 255).astype(np.uint8),
            observations.astype(np.int32, copy=False),
        )


class PointCloudProductAccumulator:
    """Collect bounded raw/filtered products and a weighted voxel map."""

    def __init__(
        self,
        *,
        total_frames: int,
        protocol: PointCloudProtocol,
        scale_reference: float,
    ) -> None:
        protocol.validate()
        if not np.isfinite(scale_reference) or scale_reference <= 0.0:
            raise ValueError("scale_reference must be finite and positive.")
        self.protocol = protocol
        self.scale_reference = float(scale_reference)
        self.voxel_size = self.scale_reference * float(protocol.voxel_size_ratio)
        self.raw = TemporalPointSampler(protocol.max_points, total_frames)
        self.filtered = TemporalPointSampler(protocol.max_points, total_frames)
        self.voxels = WeightedVoxelFusion(self.voxel_size)
        self.frames_added = 0
        self.filtered_points_seen = 0

    def add_frame(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        confidence: np.ndarray,
        *,
        raw_valid: np.ndarray | None = None,
        filtered_valid: np.ndarray | None = None,
    ) -> dict[str, float | int]:
        points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
        confidence = np.asarray(confidence, dtype=np.float32).reshape(-1)
        if points.shape != colors.shape or len(points) != len(confidence):
            raise ValueError("Point product inputs have inconsistent shapes.")
        finite = np.isfinite(points).all(axis=1) & np.isfinite(confidence)
        if raw_valid is not None:
            finite &= np.asarray(raw_valid, dtype=bool).reshape(-1)
        self.raw.add(points[finite], colors[finite])

        candidate = finite.copy()
        if filtered_valid is not None:
            candidate &= np.asarray(filtered_valid, dtype=bool).reshape(-1)
        if np.any(candidate):
            threshold = float(
                np.percentile(confidence[candidate], self.protocol.confidence_percentile)
            )
            selected = candidate & (confidence >= threshold)
            high = float(np.percentile(confidence[candidate], 99.0))
            denominator = max(high - threshold, 1e-6)
            weights = 0.1 + 0.9 * np.clip(
                (confidence[selected] - threshold) / denominator,
                0.0,
                1.0,
            )
        else:
            threshold = float("nan")
            selected = candidate
            weights = np.empty((0,), dtype=np.float32)

        self.filtered.add(points[selected], colors[selected])
        self.voxels.add_frame(points[selected], colors[selected], weights)
        self.frames_added += 1
        self.filtered_points_seen += int(selected.sum())
        return {
            "confidence_threshold": threshold,
            "raw_points": int(finite.sum()),
            "filtered_points": int(selected.sum()),
        }

    def write(self, points_dir: str | Path, prefix: str) -> dict[str, float | int | str]:
        points_dir = Path(points_dir)
        raw_points, raw_colors = self.raw.arrays()
        filtered_points, filtered_colors = self.filtered.arrays()
        fused_points, fused_colors, observations = self.voxels.arrays(
            min_observations=self.protocol.min_voxel_observations,
            max_points=self.protocol.max_points,
        )
        write_binary_ply(points_dir / f"{prefix}_raw.ply", raw_points, raw_colors)
        write_binary_ply(
            points_dir / f"{prefix}_conf.ply",
            filtered_points,
            filtered_colors,
        )
        write_binary_ply(
            points_dir / f"{prefix}_conf_voxel.ply",
            fused_points,
            fused_colors,
        )
        summary: dict[str, float | int | str] = {
            "prefix": prefix,
            "frames": self.frames_added,
            "confidence_percentile": self.protocol.confidence_percentile,
            "scale_reference": self.scale_reference,
            "voxel_size_ratio": self.protocol.voxel_size_ratio,
            "voxel_size": self.voxel_size,
            "min_voxel_observations": self.protocol.min_voxel_observations,
            "raw_points_seen": self.raw.seen_points,
            "filtered_points_seen": self.filtered_points_seen,
            "raw_ply_points": len(raw_points),
            "filtered_ply_points": len(filtered_points),
            "fused_ply_points": len(fused_points),
            "fused_observation_mean": (
                float(observations.mean()) if len(observations) else 0.0
            ),
        }
        write_json(points_dir / f"{prefix}_summary.json", summary)
        return summary


def unproject_depth(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    depth = _as_spatial_map(depth, dtype=np.float32)
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    if depth.ndim != 2 or intrinsics.shape != (3, 3):
        raise ValueError(f"Invalid depth/intrinsics shape: {depth.shape}/{intrinsics.shape}")
    height, width = depth.shape
    ys, xs = np.indices((height, width), dtype=np.float32)
    z = depth
    x = (xs - float(intrinsics[0, 2])) * z / float(intrinsics[0, 0])
    y = (ys - float(intrinsics[1, 2])) * z / float(intrinsics[1, 1])
    return np.stack((x, y, z), axis=-1)


def camera_to_world(points: np.ndarray, world_to_camera: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    world_to_camera = np.asarray(world_to_camera, dtype=np.float64)
    if world_to_camera.shape != (3, 4):
        raise ValueError(f"Expected W2C [3,4], got {world_to_camera.shape}")
    rotation = world_to_camera[:, :3]
    translation = world_to_camera[:, 3]
    world = (rotation.T @ (points.T - translation[:, None])).T
    return world.astype(np.float32, copy=False)


def rebuild_depth_pose_products(
    scene_dir: str | Path,
    *,
    protocol: PointCloudProtocol,
    prefix: str = "depthpose",
) -> dict[str, float | int | str]:
    """Rebuild comparable depth+pose products from one completed scene output."""

    protocol.validate()
    scene_dir = Path(scene_dir)
    depth_files = _sorted_files(scene_dir / "depth" / "dpt", ".npy")
    confidence_files = _sorted_files(scene_dir / "depth" / "conf", ".npy")
    rgb_files = _sorted_files(scene_dir / "images" / "rgb", ".png")
    frame_ids, extrinsics = read_w2c_txt(scene_dir / "poses" / "abs_pose.txt")
    intri_ids, intrinsics = read_intrinsics_txt(scene_dir / "poses" / "intri.txt")
    counts = {
        len(depth_files),
        len(confidence_files),
        len(rgb_files),
        len(frame_ids),
        len(intri_ids),
    }
    if len(counts) != 1 or not counts:
        raise ValueError(
            "Depth/confidence/RGB/pose frame counts disagree: "
            f"{len(depth_files)}/{len(confidence_files)}/{len(rgb_files)}/"
            f"{len(frame_ids)}/{len(intri_ids)}"
        )
    if frame_ids != intri_ids:
        raise ValueError("Pose and intrinsic frame ids disagree.")

    scale_reference = _depth_scale_reference(depth_files)
    print(
        f"[pointcloud] rebuilding {prefix}: frames={len(depth_files)} "
        f"scale={scale_reference:.6g} voxel={scale_reference * protocol.voxel_size_ratio:.6g}",
        flush=True,
    )
    accumulator = PointCloudProductAccumulator(
        total_frames=len(depth_files),
        protocol=protocol,
        scale_reference=scale_reference,
    )
    for index, (depth_path, confidence_path, rgb_path) in enumerate(
        zip(depth_files, confidence_files, rgb_files)
    ):
        depth = _as_spatial_map(np.load(depth_path), dtype=np.float32)
        confidence = _as_spatial_map(np.load(confidence_path), dtype=np.float32)
        rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
        if depth.shape != confidence.shape or rgb.shape[:2] != depth.shape:
            raise ValueError(
                f"Frame {index} depth/confidence/RGB shape mismatch: "
                f"{depth.shape}/{confidence.shape}/{rgb.shape}"
            )
        camera_points = unproject_depth(depth, intrinsics[index])
        world_points = camera_to_world(camera_points, extrinsics[index]).reshape(depth.shape + (3,))
        positive = np.isfinite(depth) & (depth > 0)
        filtered_depth = _depth_percentile_mask(
            depth,
            positive,
            protocol.depth_percentile_low,
            protocol.depth_percentile_high,
        )
        accumulator.add_frame(
            world_points,
            rgb,
            confidence,
            raw_valid=positive,
            filtered_valid=filtered_depth,
        )
        if (index + 1) % 50 == 0 or index + 1 == len(depth_files):
            print(
                f"[pointcloud] {prefix} accumulated {index + 1}/{len(depth_files)}",
                flush=True,
            )
    print(f"[pointcloud] finalizing {prefix} voxel products", flush=True)
    result = accumulator.write(scene_dir / "points", prefix)
    result.update(
        {
            "depth_percentile_low": protocol.depth_percentile_low,
            "depth_percentile_high": protocol.depth_percentile_high,
            "coordinate_source": "depth_intrinsics_online_w2c",
        }
    )
    write_json(scene_dir / "points" / f"{prefix}_summary.json", result)
    print(
        f"[pointcloud] {prefix} complete: raw={result['raw_ply_points']} "
        f"conf={result['filtered_ply_points']} fused={result['fused_ply_points']}",
        flush=True,
    )
    return result


def read_w2c_txt(path: str | Path) -> tuple[list[str], np.ndarray]:
    frame_ids: list[str] = []
    matrices: list[np.ndarray] = []
    for values in _read_numeric_rows(path):
        if len(values) != 13:
            raise ValueError(f"Expected 13 W2C fields in {path}, got {len(values)}")
        frame_ids.append(values[0])
        numeric = np.asarray(values[1:], dtype=np.float64)
        matrix = np.empty((3, 4), dtype=np.float64)
        matrix[:, :3] = numeric[:9].reshape(3, 3)
        matrix[:, 3] = numeric[9:12]
        matrices.append(matrix)
    return frame_ids, np.stack(matrices, axis=0)


def read_intrinsics_txt(path: str | Path) -> tuple[list[str], np.ndarray]:
    frame_ids: list[str] = []
    matrices: list[np.ndarray] = []
    for values in _read_numeric_rows(path):
        if len(values) != 5:
            raise ValueError(f"Expected 5 intrinsic fields in {path}, got {len(values)}")
        frame_ids.append(values[0])
        fx, fy, cx, cy = (float(value) for value in values[1:])
        matrices.append(
            np.asarray(((fx, 0.0, cx), (0.0, fy, cy), (0.0, 0.0, 1.0)))
        )
    return frame_ids, np.stack(matrices, axis=0)


def _read_numeric_rows(path: str | Path) -> list[list[str]]:
    rows = []
    for line in Path(path).read_text().splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            rows.append(stripped.split())
    if not rows:
        raise ValueError(f"No data rows in {path}")
    return rows


def _sorted_files(directory: Path, suffix: str) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    files = sorted(
        (path for path in directory.iterdir() if path.suffix.casefold() == suffix),
        key=natural_sort_key,
    )
    if not files:
        raise ValueError(f"No {suffix} files in {directory}")
    return files


def _depth_scale_reference(depth_files: Sequence[Path]) -> float:
    frame_medians = []
    for path in depth_files:
        depth = _as_spatial_map(np.load(path), dtype=np.float32)
        valid = np.isfinite(depth) & (depth > 0)
        if np.any(valid):
            frame_medians.append(float(np.median(depth[valid])))
    if not frame_medians:
        raise ValueError("No positive finite depth for voxel scale selection.")
    return float(np.median(frame_medians))


def _as_spatial_map(value: np.ndarray, *, dtype) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim == 2:
        return array
    if array.ndim == 3 and array.shape[-1] == 1:
        return array[..., 0]
    if array.ndim == 3 and array.shape[0] == 1:
        return array[0]
    raise ValueError(f"Expected a 2D spatial map, got {array.shape}")


def _depth_percentile_mask(
    depth: np.ndarray,
    valid: np.ndarray,
    low: float,
    high: float,
) -> np.ndarray:
    values = depth[valid]
    if not len(values):
        return valid.copy()
    lower, upper = np.percentile(values, (low, high))
    return valid & (depth >= lower) & (depth <= upper)


def _aggregate_frame_voxels(
    points: np.ndarray,
    colors: np.ndarray,
    weights: np.ndarray,
    voxel_size: float,
) -> _VoxelStats:
    coordinates = np.floor(points / float(voxel_size)).astype(np.int64)
    unique, inverse = np.unique(coordinates, axis=0, return_inverse=True)
    count = len(unique)
    weight_sum = np.bincount(inverse, weights=weights, minlength=count).astype(np.float64)
    position_sum = np.stack(
        [
            np.bincount(inverse, weights=points[:, axis] * weights, minlength=count)
            for axis in range(3)
        ],
        axis=1,
    )
    color_sum = np.stack(
        [
            np.bincount(inverse, weights=colors[:, axis] * weights, minlength=count)
            for axis in range(3)
        ],
        axis=1,
    )
    return _VoxelStats(
        coordinates=unique,
        position_sum=position_sum,
        color_sum=color_sum,
        weight_sum=weight_sum,
        observations=np.ones(count, dtype=np.int32),
    )


def _merge_voxel_stats(stats: Sequence[_VoxelStats]) -> _VoxelStats:
    if len(stats) == 1:
        return stats[0]
    coordinates = np.concatenate([item.coordinates for item in stats], axis=0)
    position = np.concatenate([item.position_sum for item in stats], axis=0)
    color = np.concatenate([item.color_sum for item in stats], axis=0)
    weight = np.concatenate([item.weight_sum for item in stats], axis=0)
    observations = np.concatenate([item.observations for item in stats], axis=0)
    unique, inverse = np.unique(coordinates, axis=0, return_inverse=True)
    count = len(unique)
    return _VoxelStats(
        coordinates=unique,
        position_sum=np.stack(
            [
                np.bincount(inverse, weights=position[:, axis], minlength=count)
                for axis in range(3)
            ],
            axis=1,
        ),
        color_sum=np.stack(
            [
                np.bincount(inverse, weights=color[:, axis], minlength=count)
                for axis in range(3)
            ],
            axis=1,
        ),
        weight_sum=np.bincount(inverse, weights=weight, minlength=count),
        observations=np.bincount(
            inverse,
            weights=observations,
            minlength=count,
        ).astype(np.int32),
    )
