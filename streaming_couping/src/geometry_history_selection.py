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
    overlap_pool_size: int = 12

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
    """Select QK overlap first, then greedily diversify camera viewpoints.

    The first-stage QK pool is an RGB-only overlap gate.  The best-QK candidate
    seeds the set.  Each later candidate maximizes its mean camera-center and
    rotation distance from the already selected set.  Distances are converted
    to ranks at every greedy step, avoiding a tuned translation/rotation unit
    conversion.
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
    pose_rotations, pose_centers = _camera_geometry(poses)
    current_baselines, current_rotations = _mean_pose_distances(
        pose_rotations,
        pose_centers,
        candidate_indices=pool,
        reference_indices=(frame,),
    )
    chosen = [pool[0]]
    selection_details: dict[int, dict[str, Any]] = {
        pool[0]: {
            "greedy_selection_step": 1,
            "selection_mean_baseline_to_prior_native": 0.0,
            "selection_mean_rotation_to_prior_degrees": 0.0,
            "selection_baseline_rank_normalized": 0.0,
            "selection_rotation_rank_normalized": 0.0,
            "selection_diversity_score": 0.0,
        }
    }
    while len(chosen) < remaining:
        candidates = tuple(history for history in pool if history not in chosen)
        baselines, rotations = _mean_pose_distances(
            pose_rotations,
            pose_centers,
            candidate_indices=candidates,
            reference_indices=tuple(chosen),
        )
        baseline_ranks = _normalized_ranks(baselines, candidates)
        rotation_ranks = _normalized_ranks(rotations, candidates)
        diversity_scores = 0.5 * (baseline_ranks + rotation_ranks)
        winner_position = sorted(
            range(len(candidates)),
            key=lambda position: (
                -float(diversity_scores[position]),
                -float(qk_scores[candidates[position]]),
                int(candidates[position]),
            ),
        )[0]
        winner = candidates[winner_position]
        chosen.append(winner)
        selection_details[winner] = {
            "greedy_selection_step": len(chosen),
            "selection_mean_baseline_to_prior_native": float(
                baselines[winner_position]
            ),
            "selection_mean_rotation_to_prior_degrees": float(
                rotations[winner_position]
            ),
            "selection_baseline_rank_normalized": float(
                baseline_ranks[winner_position]
            ),
            "selection_rotation_rank_normalized": float(
                rotation_ranks[winner_position]
            ),
            "selection_diversity_score": float(diversity_scores[winner_position]),
        }
    chosen_tuple = tuple(chosen)
    final_baselines = []
    final_rotations = []
    for history in pool:
        references = tuple(value for value in chosen_tuple if value != history)
        if not references:
            references = chosen_tuple
        baseline, rotation = _mean_pose_distances(
            pose_rotations,
            pose_centers,
            candidate_indices=(history,),
            reference_indices=references,
        )
        final_baselines.append(float(baseline[0]))
        final_rotations.append(float(rotation[0]))
    selected = tuple(sorted((*anchors, *chosen_tuple)))
    chosen_set = set(chosen_tuple)
    diagnostics = tuple(
        {
            "history_sequence_index": int(history),
            "qk_pool_rank": int(position),
            "qk_score": float(qk_scores[history]),
            "camera_center_baseline_to_current_native": float(
                current_baselines[position]
            ),
            "relative_rotation_to_current_degrees": float(
                current_rotations[position]
            ),
            "mean_baseline_to_final_selected_native": final_baselines[position],
            "mean_rotation_to_final_selected_degrees": final_rotations[position],
            **selection_details.get(
                history,
                {
                    "greedy_selection_step": -1,
                    "selection_mean_baseline_to_prior_native": None,
                    "selection_mean_rotation_to_prior_degrees": None,
                    "selection_baseline_rank_normalized": None,
                    "selection_rotation_rank_normalized": None,
                    "selection_diversity_score": None,
                },
            ),
            "selected": int(history in chosen_set),
        }
        for position, history in enumerate(pool)
    )
    return GeometryHistorySelection(selected, pool, diagnostics)


def _camera_geometry(poses: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rotations = torch.stack(
        [_project_rotation(matrix[:3, :3]) for matrix in poses]
    )
    translations = poses[:, :3, 3]
    centers = -torch.einsum(
        "sij,sj->si", rotations.transpose(-1, -2), translations
    )
    return rotations, centers


def _mean_pose_distances(
    rotations: torch.Tensor,
    centers: torch.Tensor,
    *,
    candidate_indices: tuple[int, ...],
    reference_indices: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    if not candidate_indices:
        empty = torch.empty(0, dtype=torch.float32)
        return empty, empty
    if not reference_indices:
        raise ValueError("Pose diversity needs at least one reference view.")
    candidates = torch.tensor(candidate_indices, dtype=torch.long)
    references = torch.tensor(reference_indices, dtype=torch.long)
    candidate_centers = centers.index_select(0, candidates)
    reference_centers = centers.index_select(0, references)
    baselines = torch.linalg.vector_norm(
        candidate_centers[:, None, :] - reference_centers[None, :, :],
        dim=-1,
    ).mean(dim=1)
    candidate_rotations = rotations.index_select(0, candidates)
    reference_rotations = rotations.index_select(0, references)
    relative = candidate_rotations[:, None] @ reference_rotations.transpose(-1, -2)[
        None
    ]
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1.0) / 2.0).clamp(
        -1.0, 1.0
    )
    angles = torch.rad2deg(torch.acos(cosine)).mean(dim=1)
    return baselines.float(), angles.float()


def select_recent_history(
    frame_index: int,
    *,
    total_frame_budget: int,
    anchor_frames: int,
) -> tuple[int, ...]:
    """Keep fixed anchors and fill the remaining budget with recent history."""

    frame = int(frame_index)
    total = int(total_frame_budget)
    anchors_count = int(anchor_frames)
    if total < 2 or not 1 <= anchors_count < total:
        raise ValueError("Invalid recent-history budget.")
    if frame <= total:
        return tuple(range(frame))
    anchors = tuple(range(min(anchors_count, frame)))
    available = tuple(index for index in range(frame) if index not in anchors)
    remaining = total - len(anchors)
    return tuple(sorted((*anchors, *available[-remaining:])))


def _normalized_ranks(
    values: torch.Tensor,
    history_indices: tuple[int, ...],
) -> torch.Tensor:
    if values.ndim != 1 or values.numel() != len(history_indices):
        raise ValueError("Rank values and history indices are inconsistent.")
    if values.numel() <= 1:
        return torch.ones_like(values, dtype=torch.float32)
    ranks = torch.empty(values.numel(), dtype=torch.float32)
    denominator = float(values.numel() - 1)
    for position in range(values.numel()):
        tied = torch.isclose(
            values,
            values[position],
            rtol=1e-5,
            atol=1e-7,
        )
        strictly_lower = (values < values[position]) & ~tied
        average_rank = float(strictly_lower.sum()) + 0.5 * float(tied.sum() - 1)
        ranks[position] = average_rank / denominator
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
