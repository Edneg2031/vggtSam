"""Training-free history policies for an isolated point-head replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .qk_pose_retrieval import rank_qk_history


@dataclass(frozen=True)
class GeometryHistoryPolicy:
    total_frame_budget: int = 5
    anchor_frames: int = 1
    overlap_pool_size: int = 8

    def validate(self) -> None:
        if int(self.total_frame_budget) < 2:
            raise ValueError("Geometry history needs at least two total frames.")
        if not 1 <= int(self.anchor_frames) < int(self.total_frame_budget):
            raise ValueError("anchor_frames must be in [1, total_frame_budget).")
        remaining = int(self.total_frame_budget) - int(self.anchor_frames)
        if int(self.overlap_pool_size) < remaining:
            raise ValueError("overlap_pool_size must cover the non-anchor budget.")


@dataclass(frozen=True)
class GeometryHistorySelection:
    selected_indices: tuple[int, ...]
    qk_pool_indices: tuple[int, ...]
    diagnostics: tuple[dict[str, Any], ...]


def select_geometry_history(
    frame_index: int,
    qk_scores: torch.Tensor,
    qk_world_to_camera: torch.Tensor,
    *,
    policy: GeometryHistoryPolicy,
) -> GeometryHistorySelection:
    """Select overlap first, then favor complementary camera viewpoints.

    The first-stage QK pool is an RGB-only overlap proxy.  Within that pool,
    camera-center distance and relative rotation are ranked independently and
    combined with equal weight.  Only ranks are used, so native pose scale does
    not introduce a tuned metric threshold.
    """

    policy.validate()
    frame = int(frame_index)
    if qk_scores.ndim != 1 or int(qk_scores.numel()) != frame:
        raise ValueError(
            f"Expected {frame} QK history scores, got {tuple(qk_scores.shape)}."
        )
    poses = _as_homogeneous(qk_world_to_camera)
    if frame < 0 or frame >= poses.shape[0]:
        raise ValueError("frame_index is outside the frozen QK pose sequence.")
    if not torch.isfinite(poses[: frame + 1]).all():
        raise ValueError("Frozen QK poses contain non-finite values.")

    anchors = tuple(range(min(int(policy.anchor_frames), frame)))
    available = tuple(index for index in range(frame) if index not in anchors)
    remaining = min(
        int(policy.total_frame_budget) - len(anchors),
        len(available),
    )
    if remaining <= 0:
        return GeometryHistorySelection(anchors, (), ())

    qk_ranked = rank_qk_history(qk_scores, indices=available)
    pool = tuple(qk_ranked[: min(int(policy.overlap_pool_size), len(qk_ranked))])
    baselines, rotations = _view_complementarity(
        poses,
        current_index=frame,
        history_indices=pool,
    )
    baseline_ranks = _normalized_ranks(baselines, pool)
    rotation_ranks = _normalized_ranks(rotations, pool)
    geometry_scores = 0.5 * (baseline_ranks + rotation_ranks)
    pool_positions = {history: position for position, history in enumerate(pool)}
    geometry_ranked = tuple(
        sorted(
            pool,
            key=lambda history: (
                -float(geometry_scores[pool_positions[history]]),
                -float(qk_scores[history]),
                history,
            ),
        )
    )
    chosen = tuple(geometry_ranked[:remaining])
    selected = tuple(sorted((*anchors, *chosen)))
    chosen_set = set(chosen)
    diagnostics = tuple(
        {
            "history_sequence_index": int(history),
            "qk_pool_rank": int(position),
            "qk_score": float(qk_scores[history]),
            "camera_center_baseline_native": float(baselines[position]),
            "relative_rotation_degrees": float(rotations[position]),
            "baseline_rank_normalized": float(baseline_ranks[position]),
            "rotation_rank_normalized": float(rotation_ranks[position]),
            "geometry_score": float(geometry_scores[position]),
            "selected": int(history in chosen_set),
        }
        for position, history in enumerate(pool)
    )
    return GeometryHistorySelection(selected, pool, diagnostics)


def _view_complementarity(
    poses: torch.Tensor,
    *,
    current_index: int,
    history_indices: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    if not history_indices:
        empty = torch.empty(0, dtype=torch.float32)
        return empty, empty
    rotations = torch.stack(
        [_project_rotation(matrix[:3, :3]) for matrix in poses]
    )
    translations = poses[:, :3, 3]
    centers = -torch.einsum(
        "sij,sj->si", rotations.transpose(-1, -2), translations
    )
    indices = torch.tensor(history_indices, dtype=torch.long)
    baselines = torch.linalg.vector_norm(
        centers.index_select(0, indices) - centers[int(current_index)],
        dim=-1,
    )
    current_rotation = rotations[int(current_index)]
    history_rotations = rotations.index_select(0, indices)
    relative = history_rotations @ current_rotation.T
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) / 2.0).clamp(
        -1.0, 1.0
    )
    angles = torch.rad2deg(torch.acos(cosine))
    return baselines.float(), angles.float()


def _normalized_ranks(
    values: torch.Tensor,
    history_indices: tuple[int, ...],
) -> torch.Tensor:
    if values.ndim != 1 or values.numel() != len(history_indices):
        raise ValueError("Rank values and history indices are inconsistent.")
    if values.numel() <= 1:
        return torch.ones_like(values, dtype=torch.float32)
    order = sorted(
        range(values.numel()),
        key=lambda position: (
            float(values[position]),
            int(history_indices[position]),
        ),
    )
    ranks = torch.empty(values.numel(), dtype=torch.float32)
    denominator = float(values.numel() - 1)
    for rank, position in enumerate(order):
        ranks[position] = float(rank) / denominator
    return ranks


def _as_homogeneous(value: torch.Tensor) -> torch.Tensor:
    poses = value.detach().float().cpu()
    if poses.ndim == 4 and poses.shape[0] == 1:
        poses = poses[0]
    if poses.ndim != 3 or tuple(poses.shape[-2:]) not in {(3, 4), (4, 4)}:
        raise ValueError("QK poses must have shape [S,3,4] or [S,4,4].")
    if tuple(poses.shape[-2:]) == (4, 4):
        return poses.clone()
    output = torch.eye(4).expand(poses.shape[0], 4, 4).clone()
    output[:, :3] = poses
    return output


def _project_rotation(rotation: torch.Tensor) -> torch.Tensor:
    left, _, right_t = torch.linalg.svd(rotation.float())
    projected = left @ right_t
    if torch.det(projected) < 0.0:
        left = left.clone()
        left[:, -1] *= -1.0
        projected = left @ right_t
    return projected
