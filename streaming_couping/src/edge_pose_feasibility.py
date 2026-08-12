"""Causal masked-edge pose feasibility diagnostics.

This module is intentionally independent from the retained V0 baseline.  It
reads a frozen V0 feature cache and measures whether the raw StreamVGGT pose,
depth, intrinsics, RGB edges, and SAM-derived masks form a coherent edge
reprojection objective.  It does not change selected poses and does not train
or instantiate a pose head.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
import yaml
from PIL import Image


EDGE_FEASIBILITY_REVISION = "e0_masked_edge_projection_audit_r1"


@dataclass(frozen=True)
class EdgeConfig:
    sobel_quantile: float = 0.88
    max_edges_per_frame: int = 12_000
    distance_truncate_px: int = 12


@dataclass(frozen=True)
class ProjectionConfig:
    source_offsets: tuple[int, ...] = (1, 2, 4)
    min_depth: float = 0.05
    max_depth: float = 25.0
    cycle_abs_depth_tolerance: float = 0.10
    cycle_rel_depth_tolerance: float = 0.10


@dataclass(frozen=True)
class EdgeFeasibilityConfig:
    source_path: Path
    base_config: Path
    output_dir: Path
    clip_name: str
    evaluation_frames: tuple[int, ...]
    branches: tuple[str, ...]
    edge: EdgeConfig
    projection: ProjectionConfig


VALID_BRANCHES = {
    "raw",
    "all_edge",
    "sam_static_edge",
    "shuffled_static_mask",
}


def load_edge_feasibility_config(path: str | Path) -> EdgeFeasibilityConfig:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf8") as handle:
        raw = yaml.safe_load(handle) or {}
    section = raw.get("e0", {})
    edge = section.get("edge", {})
    projection = section.get("projection", {})
    frames = section.get("frames", {})
    config = EdgeFeasibilityConfig(
        source_path=source,
        base_config=_path(section.get("base_config"), source.parent),
        output_dir=_path(
            section.get(
                "output_dir",
                "outputs/streaming_couping_e0_edge_feasibility",
            ),
            source.parent,
        ),
        clip_name=str(section.get("clip_name", "")),
        evaluation_frames=_int_tuple(frames.get("evaluation", ())),
        branches=tuple(str(value) for value in section.get("branches", ())),
        edge=EdgeConfig(
            sobel_quantile=float(edge.get("sobel_quantile", 0.88)),
            max_edges_per_frame=int(edge.get("max_edges_per_frame", 12_000)),
            distance_truncate_px=int(edge.get("distance_truncate_px", 12)),
        ),
        projection=ProjectionConfig(
            source_offsets=_int_tuple(
                projection.get("source_offsets", (1, 2, 4))
            ),
            min_depth=float(projection.get("min_depth", 0.05)),
            max_depth=float(projection.get("max_depth", 25.0)),
            cycle_abs_depth_tolerance=float(
                projection.get("cycle_abs_depth_tolerance", 0.10)
            ),
            cycle_rel_depth_tolerance=float(
                projection.get("cycle_rel_depth_tolerance", 0.10)
            ),
        ),
    )
    _validate_config(config)
    return config


def run_edge_feasibility(
    *,
    payload: dict,
    world_to_camera: torch.Tensor,
    intrinsics: torch.Tensor,
    config: EdgeFeasibilityConfig,
) -> dict[str, object]:
    """Evaluate raw-pose edge reprojection residuals for each branch."""

    frames = tuple(int(value) for value in payload["frame_indices"])
    frame_to_index = {frame: index for index, frame in enumerate(frames)}
    evaluation_indices = [frame_to_index[frame] for frame in config.evaluation_frames]
    images = _load_gray_images(
        payload["image_paths"],
        image_size=_image_size(payload),
    )
    depth = _tensor_field(payload, "baseline_depth").float()
    depth_confidence = _tensor_field(payload, "baseline_depth_confidence").float()
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth_confidence.ndim == 4 and depth_confidence.shape[-1] == 1:
        depth_confidence = depth_confidence[..., 0]
    if world_to_camera.ndim == 4:
        world_to_camera = world_to_camera[0]
    if intrinsics.ndim == 4:
        intrinsics = intrinsics[0]
    if world_to_camera.shape[0] != len(frames):
        raise ValueError("world_to_camera length does not match frame_indices.")
    if intrinsics.shape[0] != len(frames):
        raise ValueError("intrinsics length does not match frame_indices.")
    if depth.shape[-2:] != images.shape[-2:]:
        depth = _resize_scalar(depth[:, None], images.shape[-2:])[:, 0]
        depth_confidence = _resize_scalar(
            depth_confidence[:, None],
            images.shape[-2:],
        )[:, 0]

    edge_strength = sobel_magnitude(images)
    all_edges = threshold_edges(
        edge_strength,
        quantile=config.edge.sobel_quantile,
        max_edges_per_frame=config.edge.max_edges_per_frame,
    )
    exclusion_masks = _load_exclusion_masks(payload, size=images.shape[-2:])
    branch_edges = {
        branch: _branch_edges(branch, all_edges, exclusion_masks)
        for branch in config.branches
        if branch != "raw"
    }
    distance_fields = {
        branch: torch.stack(
            [
                truncated_distance_transform(
                    edge_map,
                    max_distance=config.edge.distance_truncate_px,
                )
                for edge_map in edges
            ],
            dim=0,
        )
        for branch, edges in branch_edges.items()
    }

    rows: list[dict[str, object]] = []
    for target_index in evaluation_indices:
        for offset in config.projection.source_offsets:
            source_index = target_index - int(offset)
            if source_index < 0:
                continue
            for branch in config.branches:
                if branch == "raw":
                    rows.append(
                        _raw_row(
                            frames=frames,
                            source_index=source_index,
                            target_index=target_index,
                            offset=offset,
                        )
                    )
                    continue
                row = pair_edge_projection_metrics(
                    frames=frames,
                    source_index=source_index,
                    target_index=target_index,
                    offset=offset,
                    branch=branch,
                    source_edges=branch_edges[branch][source_index],
                    target_distance=distance_fields[branch][target_index],
                    depth=depth,
                    depth_confidence=depth_confidence,
                    world_to_camera=world_to_camera,
                    intrinsics=intrinsics,
                    projection=config.projection,
                )
                rows.append(row)
    return {
        "revision": EDGE_FEASIBILITY_REVISION,
        "clip": payload["clip_name"],
        "frames": frames,
        "evaluation_frames": config.evaluation_frames,
        "branches": config.branches,
        "rows": rows,
        "summary": summarize_rows(rows),
    }


def pair_edge_projection_metrics(
    *,
    frames: Sequence[int],
    source_index: int,
    target_index: int,
    offset: int,
    branch: str,
    source_edges: torch.Tensor,
    target_distance: torch.Tensor,
    depth: torch.Tensor,
    depth_confidence: torch.Tensor,
    world_to_camera: torch.Tensor,
    intrinsics: torch.Tensor,
    projection: ProjectionConfig,
) -> dict[str, object]:
    h, w = source_edges.shape
    ys, xs = torch.nonzero(source_edges.bool(), as_tuple=True)
    total_edges = int(xs.numel())
    if total_edges == 0:
        return {
            **_row_base(frames, source_index, target_index, offset, branch),
            "source_edge_pixels": 0,
            "source_positive_depth_pixels": 0,
            "projected_in_bounds_pixels": 0,
            "depth_cycle_pass_pixels": 0,
            "valid_projected_pixels": 0,
            "in_bounds_rate": float("nan"),
            "depth_cycle_pass_rate": float("nan"),
            "mean_truncated_edge_distance_px": float("nan"),
            "median_truncated_edge_distance_px": float("nan"),
            "mean_depth_confidence": float("nan"),
        }

    source_depth = depth[source_index, ys, xs].float()
    source_conf = depth_confidence[source_index, ys, xs].float()
    positive = (
        torch.isfinite(source_depth)
        & (source_depth > float(projection.min_depth))
        & (source_depth < float(projection.max_depth))
    )
    if not bool(positive.any()):
        return {
            **_row_base(frames, source_index, target_index, offset, branch),
            "source_edge_pixels": total_edges,
            "source_positive_depth_pixels": 0,
            "projected_in_bounds_pixels": 0,
            "depth_cycle_pass_pixels": 0,
            "valid_projected_pixels": 0,
            "in_bounds_rate": 0.0,
            "depth_cycle_pass_rate": float("nan"),
            "mean_truncated_edge_distance_px": float("nan"),
            "median_truncated_edge_distance_px": float("nan"),
            "mean_depth_confidence": float("nan"),
        }

    xs = xs[positive].float()
    ys = ys[positive].float()
    source_depth = source_depth[positive]
    source_conf = source_conf[positive]
    uvz = project_depth_points(
        x=xs,
        y=ys,
        depth=source_depth,
        source_w2c=world_to_camera[source_index],
        target_w2c=world_to_camera[target_index],
        source_k=intrinsics[source_index],
        target_k=intrinsics[target_index],
    )
    u, v, z = uvz[:, 0], uvz[:, 1], uvz[:, 2]
    in_bounds = (
        torch.isfinite(u)
        & torch.isfinite(v)
        & torch.isfinite(z)
        & (z > float(projection.min_depth))
        & (u >= 0)
        & (v >= 0)
        & (u <= w - 1)
        & (v <= h - 1)
    )
    projected_count = int(in_bounds.sum())
    if projected_count == 0:
        return {
            **_row_base(frames, source_index, target_index, offset, branch),
            "source_edge_pixels": total_edges,
            "source_positive_depth_pixels": int(source_depth.numel()),
            "projected_in_bounds_pixels": 0,
            "depth_cycle_pass_pixels": 0,
            "valid_projected_pixels": 0,
            "in_bounds_rate": 0.0,
            "depth_cycle_pass_rate": float("nan"),
            "mean_truncated_edge_distance_px": float("nan"),
            "median_truncated_edge_distance_px": float("nan"),
            "mean_depth_confidence": float(source_conf.mean()),
        }

    u_i = u[in_bounds].round().long().clamp(0, w - 1)
    v_i = v[in_bounds].round().long().clamp(0, h - 1)
    z_in = z[in_bounds].float()
    target_depth = depth[target_index, v_i, u_i].float()
    target_valid = (
        torch.isfinite(target_depth)
        & (target_depth > float(projection.min_depth))
        & (target_depth < float(projection.max_depth))
    )
    tolerance = float(projection.cycle_abs_depth_tolerance) + float(
        projection.cycle_rel_depth_tolerance
    ) * torch.maximum(z_in.abs(), target_depth.abs())
    cycle = target_valid & ((target_depth - z_in).abs() <= tolerance)
    distances = target_distance[v_i, u_i].float()
    valid_distances = distances[cycle] if bool(cycle.any()) else distances
    return {
        **_row_base(frames, source_index, target_index, offset, branch),
        "source_edge_pixels": total_edges,
        "source_positive_depth_pixels": int(source_depth.numel()),
        "projected_in_bounds_pixels": projected_count,
        "depth_cycle_pass_pixels": int(cycle.sum()),
        "valid_projected_pixels": int(valid_distances.numel()),
        "in_bounds_rate": float(projected_count / max(int(source_depth.numel()), 1)),
        "depth_cycle_pass_rate": float(cycle.sum() / max(projected_count, 1)),
        "mean_truncated_edge_distance_px": float(valid_distances.mean()),
        "median_truncated_edge_distance_px": float(valid_distances.median()),
        "mean_depth_confidence": float(source_conf.mean()),
    }


def project_depth_points(
    *,
    x: torch.Tensor,
    y: torch.Tensor,
    depth: torch.Tensor,
    source_w2c: torch.Tensor,
    target_w2c: torch.Tensor,
    source_k: torch.Tensor,
    target_k: torch.Tensor,
) -> torch.Tensor:
    ones = torch.ones_like(x)
    pixels = torch.stack((x, y, ones), dim=-1).float()
    source_k_inv = torch.linalg.inv(source_k.float())
    source_camera = (pixels @ source_k_inv.T) * depth[:, None].float()
    source_rotation = source_w2c[:3, :3].float()
    source_translation = source_w2c[:3, 3].float()
    world = (source_camera - source_translation) @ source_rotation
    target_rotation = target_w2c[:3, :3].float()
    target_translation = target_w2c[:3, 3].float()
    target_camera = world @ target_rotation.T + target_translation
    projected = target_camera @ target_k.float().T
    z = target_camera[:, 2].clamp_min(1e-8)
    uv = projected[:, :2] / projected[:, 2:].clamp_min(1e-8)
    return torch.cat((uv, z[:, None]), dim=-1)


def sobel_magnitude(images: torch.Tensor) -> torch.Tensor:
    if images.ndim != 3:
        raise ValueError("images must have shape [T,H,W].")
    device = images.device
    dtype = images.dtype
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=dtype,
        device=device,
    )
    kernel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        dtype=dtype,
        device=device,
    )
    weight = torch.stack((kernel_x, kernel_y), dim=0)[:, None]
    gradients = F.conv2d(images[:, None], weight, padding=1)
    return torch.linalg.vector_norm(gradients, dim=1)


def threshold_edges(
    strength: torch.Tensor,
    *,
    quantile: float,
    max_edges_per_frame: int,
) -> torch.Tensor:
    if strength.ndim != 3:
        raise ValueError("edge strength must have shape [T,H,W].")
    output = []
    for frame in strength:
        flat = frame.flatten()
        finite = flat[torch.isfinite(flat)]
        if finite.numel() == 0:
            output.append(torch.zeros_like(frame, dtype=torch.bool))
            continue
        cutoff = torch.quantile(finite, float(quantile))
        mask = frame >= cutoff
        if int(mask.sum()) > int(max_edges_per_frame):
            indices = torch.topk(flat, k=int(max_edges_per_frame)).indices
            limited = torch.zeros_like(flat, dtype=torch.bool)
            limited[indices] = True
            mask = limited.reshape_as(frame)
        output.append(mask.bool())
    return torch.stack(output, dim=0)


def truncated_distance_transform(
    edge_mask: torch.Tensor,
    *,
    max_distance: int,
) -> torch.Tensor:
    """Return a small integer chamfer distance field truncated in pixels."""

    if edge_mask.ndim != 2:
        raise ValueError("edge_mask must have shape [H,W].")
    limit = int(max_distance)
    large = float(limit + 1)
    dist = torch.where(
        edge_mask.bool(),
        torch.zeros_like(edge_mask, dtype=torch.float32),
        torch.full_like(edge_mask, large, dtype=torch.float32),
    )
    for _ in range(limit):
        neighbor = torch.minimum(
            torch.minimum(_shift(dist, 1, 0, large), _shift(dist, -1, 0, large)),
            torch.minimum(_shift(dist, 0, 1, large), _shift(dist, 0, -1, large)),
        )
        dist = torch.minimum(dist, neighbor + 1.0)
    return dist.clamp_max(float(limit))


def summarize_rows(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        branch = str(row["branch"])
        if branch == "raw":
            continue
        grouped.setdefault(branch, []).append(row)
    summary = []
    for branch, branch_rows in sorted(grouped.items()):
        distances = [
            float(row["mean_truncated_edge_distance_px"])
            for row in branch_rows
            if math.isfinite(float(row["mean_truncated_edge_distance_px"]))
        ]
        cycle = [
            float(row["depth_cycle_pass_rate"])
            for row in branch_rows
            if math.isfinite(float(row["depth_cycle_pass_rate"]))
        ]
        in_bounds = [
            float(row["in_bounds_rate"])
            for row in branch_rows
            if math.isfinite(float(row["in_bounds_rate"]))
        ]
        summary.append(
            {
                "branch": branch,
                "pair_count": len(branch_rows),
                "mean_truncated_edge_distance_px": _mean(distances),
                "mean_depth_cycle_pass_rate": _mean(cycle),
                "mean_in_bounds_rate": _mean(in_bounds),
            }
        )
    return summary


def csv_columns(rows: Sequence[dict[str, object]]) -> tuple[str, ...]:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    return tuple(columns)


def _load_gray_images(paths: Sequence[str], *, image_size: tuple[int, int]) -> torch.Tensor:
    h, w = image_size
    frames = []
    for value in paths:
        image = Image.open(value).convert("L").resize((w, h), Image.BILINEAR)
        data = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
        frames.append(data.float().reshape(h, w) / 255.0)
    return torch.stack(frames, dim=0)


def _load_exclusion_masks(payload: dict, *, size: tuple[int, int]) -> torch.Tensor:
    candidates = (
        "associated_tracking_masks_output",
        "trusted_tracking_masks_output",
        "tracking_masks_output",
    )
    masks = None
    for key in candidates:
        value = payload.get(key)
        if torch.is_tensor(value):
            masks = value.bool()
            break
    if masks is None:
        frames = len(payload["frame_indices"])
        return torch.zeros((frames, *size), dtype=torch.bool)
    if masks.ndim != 4:
        raise ValueError(f"{key} must have shape [T,K,H,W].")
    merged = masks.any(dim=1)
    if merged.shape[-2:] != size:
        merged = _resize_mask(merged[:, None], size)[:, 0]
    return merged.bool()


def _branch_edges(
    branch: str,
    all_edges: torch.Tensor,
    exclusion_masks: torch.Tensor,
) -> torch.Tensor:
    if branch == "all_edge":
        return all_edges.bool()
    if branch == "sam_static_edge":
        return all_edges.bool() & ~exclusion_masks.bool()
    if branch == "shuffled_static_mask":
        shifted = torch.stack(
            [
                torch.roll(mask, shifts=(7 + index % 5, 11 + index % 7), dims=(-2, -1))
                for index, mask in enumerate(exclusion_masks.bool())
            ],
            dim=0,
        )
        return all_edges.bool() & ~shifted
    raise ValueError(f"Unsupported edge branch: {branch!r}.")


def _raw_row(
    *,
    frames: Sequence[int],
    source_index: int,
    target_index: int,
    offset: int,
) -> dict[str, object]:
    return {
        **_row_base(frames, source_index, target_index, offset, "raw"),
        "source_edge_pixels": 0,
        "source_positive_depth_pixels": 0,
        "projected_in_bounds_pixels": 0,
        "depth_cycle_pass_pixels": 0,
        "valid_projected_pixels": 0,
        "in_bounds_rate": float("nan"),
        "depth_cycle_pass_rate": float("nan"),
        "mean_truncated_edge_distance_px": float("nan"),
        "median_truncated_edge_distance_px": float("nan"),
        "mean_depth_confidence": float("nan"),
    }


def _row_base(
    frames: Sequence[int],
    source_index: int,
    target_index: int,
    offset: int,
    branch: str,
) -> dict[str, object]:
    return {
        "branch": branch,
        "source_sequence_index": int(source_index),
        "target_sequence_index": int(target_index),
        "source_frame_index": int(frames[source_index]),
        "target_frame_index": int(frames[target_index]),
        "source_offset": int(offset),
    }


def _tensor_field(payload: dict, key: str) -> torch.Tensor:
    value = payload.get(key)
    if not torch.is_tensor(value):
        raise ValueError(f"Feature cache lacks tensor field {key!r}.")
    return value


def _image_size(payload: dict) -> tuple[int, int]:
    raw = payload.get("image_size")
    if not raw or len(raw) != 2:
        raise ValueError("Feature cache lacks image_size=[H,W].")
    return int(raw[0]), int(raw[1])


def _resize_scalar(value: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(value.float(), size=size, mode="bilinear", align_corners=False)


def _resize_mask(value: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(value.float(), size=size, mode="nearest").bool()


def _shift(value: torch.Tensor, dy: int, dx: int, fill: float) -> torch.Tensor:
    h, w = value.shape
    output = torch.full_like(value, float(fill))
    y_src_start = max(0, -dy)
    y_src_end = min(h, h - dy)
    x_src_start = max(0, -dx)
    x_src_end = min(w, w - dx)
    y_dst_start = max(0, dy)
    y_dst_end = min(h, h + dy)
    x_dst_start = max(0, dx)
    x_dst_end = min(w, w + dx)
    output[y_dst_start:y_dst_end, x_dst_start:x_dst_end] = value[
        y_src_start:y_src_end,
        x_src_start:x_src_end,
    ]
    return output


def _mean(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _int_tuple(value) -> tuple[int, ...]:
    return tuple(int(item) for item in (value or ()))


def _path(value: str | Path | None, base: Path) -> Path:
    if value is None:
        raise ValueError("Missing required path value.")
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _validate_config(config: EdgeFeasibilityConfig) -> None:
    if not config.clip_name:
        raise ValueError("e0.clip_name is required.")
    if not config.evaluation_frames:
        raise ValueError("e0.frames.evaluation is required.")
    if not config.branches:
        raise ValueError("e0.branches must contain at least one branch.")
    unknown = set(config.branches) - VALID_BRANCHES
    if unknown:
        raise ValueError(f"Unsupported E0 branches: {sorted(unknown)}.")
    if not config.projection.source_offsets:
        raise ValueError("e0.projection.source_offsets must not be empty.")
    if config.edge.max_edges_per_frame <= 0:
        raise ValueError("e0.edge.max_edges_per_frame must be positive.")
    if not (0.0 < config.edge.sobel_quantile < 1.0):
        raise ValueError("e0.edge.sobel_quantile must be in (0,1).")
