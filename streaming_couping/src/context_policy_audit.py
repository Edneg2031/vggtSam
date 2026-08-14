"""Pure diagnostics for V0 context policies and pose/pointmap consistency."""

from __future__ import annotations

import math
import random
from typing import Sequence

import torch

from .semantic_map import normalize_confidence


def select_recent_history(
    frame_index: int,
    *,
    total_history_budget: int,
    anchor_frames: int,
) -> tuple[int, ...]:
    """Keep fixed early anchors and fill the budget with newest history."""

    anchors, available, remaining = _selection_parts(
        frame_index,
        total_history_budget=total_history_budget,
        anchor_frames=anchor_frames,
    )
    if remaining >= len(available):
        return tuple(range(int(frame_index)))
    return tuple(sorted((*anchors, *available[-remaining:])))


def select_random_history(
    frame_index: int,
    *,
    total_history_budget: int,
    anchor_frames: int,
    seed: int,
) -> tuple[int, ...]:
    """Select a deterministic causal random control with fixed anchors."""

    anchors, available, remaining = _selection_parts(
        frame_index,
        total_history_budget=total_history_budget,
        anchor_frames=anchor_frames,
    )
    if remaining >= len(available):
        return tuple(range(int(frame_index)))
    generator = random.Random((int(seed) + 1) * 1_000_003 + int(frame_index))
    sampled = generator.sample(list(available), k=remaining)
    return tuple(sorted((*anchors, *sampled)))


def pose_pointmap_consistency_rows(
    *,
    branch: str,
    frame_indices: Sequence[int],
    world_points: torch.Tensor,
    world_confidence: torch.Tensor,
    depth: torch.Tensor,
    depth_confidence: torch.Tensor,
    intrinsics: torch.Tensor,
    world_to_camera: torch.Tensor,
    confidence_threshold: float,
) -> list[dict[str, object]]:
    """Project shared-world pointmap pixels through a candidate camera pose."""

    points = world_points.detach().float().cpu()
    point_confidence = normalize_confidence(
        _squeeze_scalar_map(world_confidence, "world_confidence")
    )
    depth_value = _squeeze_scalar_map(depth, "depth")
    depth_confidence_value = normalize_confidence(
        _squeeze_scalar_map(depth_confidence, "depth_confidence")
    )
    k = intrinsics.detach().float().cpu()
    pose = _squeeze_pose(world_to_camera)
    sequence, height, width, channels = points.shape
    if channels != 3:
        raise ValueError("world_points must have shape [S,H,W,3].")
    expected_scalar = (sequence, height, width)
    if point_confidence.shape != expected_scalar:
        raise ValueError("world_confidence shape differs from world_points.")
    if depth_value.shape != expected_scalar:
        raise ValueError("depth shape differs from world_points.")
    if depth_confidence_value.shape != expected_scalar:
        raise ValueError("depth_confidence shape differs from world_points.")
    if k.shape != (sequence, 3, 3):
        raise ValueError("intrinsics must have shape [S,3,3].")
    if pose.shape != (sequence, 3, 4):
        raise ValueError("world_to_camera must have shape [S,3,4].")
    if len(frame_indices) != sequence:
        raise ValueError("frame_indices length differs from world_points.")

    y, x = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    rows: list[dict[str, object]] = []
    for index, frame in enumerate(frame_indices):
        current = points[index]
        observed_depth = depth_value[index]
        base = (
            torch.isfinite(current).all(dim=-1)
            & torch.isfinite(observed_depth)
            & (observed_depth > 1e-6)
            & torch.isfinite(point_confidence[index])
            & torch.isfinite(depth_confidence_value[index])
            & (point_confidence[index] >= float(confidence_threshold))
            & (depth_confidence_value[index] >= float(confidence_threshold))
        )
        base_points = int(base.sum())
        if not base_points:
            rows.append(
                _empty_consistency_row(
                    branch=branch,
                    sequence_index=index,
                    frame_index=int(frame),
                )
            )
            continue
        rotation = pose[index, :3, :3]
        translation = pose[index, :3, 3]
        camera = current @ rotation.T + translation
        camera_depth = camera[..., 2]
        positive = base & torch.isfinite(camera).all(dim=-1) & (
            camera_depth > 1e-6
        )
        projected = camera @ k[index].T
        projected_x = projected[..., 0] / projected[..., 2].clamp_min(1e-6)
        projected_y = projected[..., 1] / projected[..., 2].clamp_min(1e-6)
        finite_projection = torch.isfinite(projected_x) & torch.isfinite(
            projected_y
        )
        projectable = positive & finite_projection
        in_bounds = (
            projectable
            & (projected_x >= 0.0)
            & (projected_x <= width - 1)
            & (projected_y >= 0.0)
            & (projected_y <= height - 1)
        )
        reprojection = torch.sqrt(
            (projected_x - x).square() + (projected_y - y).square()
        )[projectable]
        relative_depth = (
            (camera_depth - observed_depth).abs()
            / observed_depth.clamp_min(1e-6)
        )[positive]
        projectable_points = int(projectable.sum())
        positive_points = int(positive.sum())
        rows.append(
            {
                "branch": str(branch),
                "sequence_index": index,
                "frame_index": int(frame),
                "support_points": base_points,
                "positive_z_rate": positive_points / base_points,
                "in_bounds_rate": int(in_bounds.sum())
                / max(projectable_points, 1),
                "reprojection_median_px": _quantile(reprojection, 0.50),
                "reprojection_p90_px": _quantile(reprojection, 0.90),
                "relative_depth_median": _quantile(relative_depth, 0.50),
                "relative_depth_p90": _quantile(relative_depth, 0.90),
            }
        )
    return rows


def summarize_consistency_rows(
    rows: Sequence[dict[str, object]],
) -> dict[str, object]:
    """Average per-frame consistency metrics without hiding bad frames."""

    if not rows:
        raise ValueError("Cannot summarize empty consistency rows.")
    branches = {str(row["branch"]) for row in rows}
    if len(branches) != 1:
        raise ValueError("Consistency rows must contain exactly one branch.")
    valid = [
        row
        for row in rows
        if int(row["support_points"]) > 0
        and math.isfinite(float(row["reprojection_median_px"]))
    ]
    if not valid:
        raise ValueError("Consistency audit has no valid frames.")
    return {
        "branch": next(iter(branches)),
        "frames": len(rows),
        "valid_frames": len(valid),
        "support_points": sum(int(row["support_points"]) for row in valid),
        "mean_positive_z_rate": _mean(valid, "positive_z_rate"),
        "mean_in_bounds_rate": _mean(valid, "in_bounds_rate"),
        "mean_frame_reprojection_median_px": _mean(
            valid, "reprojection_median_px"
        ),
        "mean_frame_reprojection_p90_px": _mean(
            valid, "reprojection_p90_px"
        ),
        "mean_frame_relative_depth_median": _mean(
            valid, "relative_depth_median"
        ),
        "mean_frame_relative_depth_p90": _mean(
            valid, "relative_depth_p90"
        ),
    }


def fixed_alignment_pointmap_rows(
    *,
    method: str,
    frame_indices: Sequence[int],
    reference_index: int,
    pointmap: torch.Tensor,
    raw_confidence: torch.Tensor,
    raw_world_points: torch.Tensor,
    target_world_points: torch.Tensor,
    scale: float,
    rotation: torch.Tensor,
    translation: torch.Tensor,
    confidence_threshold: float,
) -> list[dict[str, object]]:
    """Score a context policy with the raw-reference Sim(3) and support."""

    candidate = pointmap.detach().float().cpu()
    raw = raw_world_points.detach().float().cpu()
    target = target_world_points.detach().float().cpu()
    confidence = normalize_confidence(
        _squeeze_scalar_map(raw_confidence, "raw_confidence")
    )
    if candidate.shape != raw.shape or candidate.shape != target.shape:
        raise ValueError("Candidate, raw and target pointmaps must share shape.")
    if candidate.ndim != 4 or candidate.shape[-1] != 3:
        raise ValueError("pointmap must have shape [S,H,W,3].")
    if confidence.shape != candidate.shape[:3]:
        raise ValueError("raw_confidence shape differs from pointmaps.")
    if len(frame_indices) != candidate.shape[0]:
        raise ValueError("frame_indices length differs from pointmaps.")
    aligned = float(scale) * (
        candidate @ rotation.detach().float().cpu().T
    ) + translation.detach().float().cpu()
    raw_support = (
        torch.isfinite(raw).all(dim=-1)
        & torch.isfinite(target).all(dim=-1)
        & torch.isfinite(confidence)
        & (confidence >= float(confidence_threshold))
    )
    rows: list[dict[str, object]] = []
    for index, frame in enumerate(frame_indices):
        valid = raw_support[index] & torch.isfinite(aligned[index]).all(dim=-1)
        distances = torch.linalg.vector_norm(
            aligned[index][valid] - target[index][valid], dim=-1
        )
        weights = confidence[index][valid].clamp_min(1e-6)
        if not distances.numel():
            weighted_rmse = float("nan")
        else:
            weighted_rmse = float(
                torch.sqrt((weights * distances.square()).sum() / weights.sum())
            )
        rows.append(
            {
                "method": str(method),
                "sequence_index": index,
                "frame_index": int(frame),
                "is_reference": int(index == int(reference_index)),
                "paired_points": int(distances.numel()),
                "paired_distance_median": _quantile(distances, 0.50),
                "paired_distance_p90": _quantile(distances, 0.90),
                "paired_weighted_rmse": weighted_rmse,
            }
        )
    return rows


def summarize_pointmap_rows(
    rows: Sequence[dict[str, object]],
    *,
    exclude_reference: bool = True,
) -> dict[str, object]:
    if not rows:
        raise ValueError("Cannot summarize empty pointmap rows.")
    methods = {str(row["method"]) for row in rows}
    if len(methods) != 1:
        raise ValueError("Pointmap rows must contain exactly one method.")
    selected = [
        row
        for row in rows
        if not (exclude_reference and int(row["is_reference"]))
        and int(row["paired_points"]) > 0
        and math.isfinite(float(row["paired_weighted_rmse"]))
    ]
    if not selected:
        raise ValueError("Pointmap audit has no valid non-reference frames.")
    return {
        "method": next(iter(methods)),
        "evaluated_frames": len(selected),
        "paired_points": sum(int(row["paired_points"]) for row in selected),
        "mean_frame_paired_distance_median": _mean(
            selected, "paired_distance_median"
        ),
        "mean_frame_paired_distance_p90": _mean(
            selected, "paired_distance_p90"
        ),
        "mean_frame_paired_weighted_rmse": _mean(
            selected, "paired_weighted_rmse"
        ),
    }


def _selection_parts(
    frame_index: int,
    *,
    total_history_budget: int,
    anchor_frames: int,
) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    frame = int(frame_index)
    budget_value = int(total_history_budget)
    anchors_value = int(anchor_frames)
    if frame < 0:
        raise ValueError("frame_index must be non-negative.")
    if budget_value < 2:
        raise ValueError("total_history_budget must be at least two.")
    if not 1 <= anchors_value < budget_value:
        raise ValueError("anchor_frames must be in [1, total_history_budget).")
    budget = min(budget_value, frame)
    anchors = tuple(range(min(anchors_value, frame)))
    available = tuple(index for index in range(frame) if index not in anchors)
    return anchors, available, max(0, budget - len(anchors))


def _squeeze_scalar_map(value: torch.Tensor, name: str) -> torch.Tensor:
    output = value.detach().float().cpu()
    if output.ndim == 4 and output.shape[-1] == 1:
        output = output[..., 0]
    if output.ndim != 3:
        raise ValueError(f"{name} must have shape [S,H,W] or [S,H,W,1].")
    return output


def _squeeze_pose(value: torch.Tensor) -> torch.Tensor:
    output = value.detach().float().cpu()
    if output.ndim == 4 and output.shape[0] == 1:
        output = output[0]
    return output


def _quantile(values: torch.Tensor, quantile: float) -> float:
    finite = values[torch.isfinite(values)]
    if not finite.numel():
        return float("nan")
    return float(torch.quantile(finite.float(), float(quantile)))


def _mean(rows: Sequence[dict[str, object]], key: str) -> float:
    values = [
        float(row[key])
        for row in rows
        if math.isfinite(float(row[key]))
    ]
    return float(sum(values) / len(values)) if values else float("nan")


def _empty_consistency_row(
    *,
    branch: str,
    sequence_index: int,
    frame_index: int,
) -> dict[str, object]:
    return {
        "branch": str(branch),
        "sequence_index": int(sequence_index),
        "frame_index": int(frame_index),
        "support_points": 0,
        "positive_z_rate": float("nan"),
        "in_bounds_rate": float("nan"),
        "reprojection_median_px": float("nan"),
        "reprojection_p90_px": float("nan"),
        "relative_depth_median": float("nan"),
        "relative_depth_p90": float("nan"),
    }
