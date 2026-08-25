"""World-space consistency validation for V2.2 recovery candidates.

This module is deliberately independent of the foundation models.  It checks
the current StreamVGGT world points selected by a candidate mask against the
causal object memory accumulated from accepted observations.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch

from .aggregation.point_map_fusion import sample_masked_observation
from .types import RevisitCandidate
from .v2_geometry_recovery import V2ObjectMemoryState


@dataclass(frozen=True)
class V22GeometryValidationConfig:
    """Thresholds for centroid, point-overlap, and shape checks."""

    max_candidate_points: int = 2048
    max_historical_points: int = 2048
    min_candidate_points: int = 8
    min_historical_points: int = 16
    centroid_distance_threshold_m: float = 0.35
    centroid_distance_scale: float = 0.75
    point_overlap_distance_m: float = 0.10
    min_point_overlap_ratio: float = 0.20
    min_shape_score: float = 0.25
    alpha_2d_score: float = 0.55
    beta_3d_score: float = 0.45
    final_accept_threshold: float = 0.58
    distance_chunk_size: int = 512
    covariance_epsilon: float = 1.0e-6


def _finite_points(points: torch.Tensor) -> torch.Tensor:
    points = points.detach().float().cpu().reshape(-1, 3)
    return points[torch.isfinite(points).all(dim=-1)]


def _centroid(points: torch.Tensor) -> torch.Tensor:
    return points.mean(dim=0) if points.numel() else torch.zeros(3)


def _radius(points: torch.Tensor, center: torch.Tensor) -> float:
    if points.numel() == 0:
        return 0.0
    distance = torch.linalg.vector_norm(points - center[None], dim=-1)
    return float(torch.sqrt(torch.mean(distance.square())).item())


def _shape_signature(
    points: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Return normalized covariance eigenvalues in ascending order."""

    if int(points.shape[0]) < 2:
        return torch.full((3,), 1.0 / 3.0)
    centered = points - points.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(1, int(points.shape[0]) - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(float(epsilon))
    return eigenvalues / eigenvalues.sum().clamp_min(float(epsilon))


def _shape_score(
    candidate: torch.Tensor,
    historical: torch.Tensor,
    epsilon: float,
) -> float:
    left = _shape_signature(candidate, epsilon)
    right = _shape_signature(historical, epsilon)
    log_delta = torch.abs(torch.log(left.clamp_min(epsilon)) - torch.log(right.clamp_min(epsilon)))
    return float(torch.exp(-torch.mean(log_delta)).clamp(0.0, 1.0).item())


def _nearest_distances(
    source: torch.Tensor,
    target: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    if source.numel() == 0 or target.numel() == 0:
        return torch.empty(0)
    target = target.contiguous()
    rows: list[torch.Tensor] = []
    chunk_size = max(1, int(chunk_size))
    for start in range(0, int(source.shape[0]), chunk_size):
        chunk = source[start : start + chunk_size]
        distances = torch.cdist(chunk, target, p=2)
        rows.append(distances.min(dim=1).values)
    return torch.cat(rows, dim=0)


def validate_v22_candidate(
    *,
    candidate_row: dict[str, Any],
    candidate_stream_mask: torch.Tensor,
    current_world_points: torch.Tensor,
    current_confidence: torch.Tensor,
    memory: V2ObjectMemoryState,
    config: V22GeometryValidationConfig,
    **_: Any,
) -> dict[str, Any]:
    """Validate one already 2-D-approved V2.1 recovery candidate.

    The returned mapping is consumed by ``process_v21_sequence``.  A rejected
    result never modifies the final mask, score, or object memory.
    """

    historical = _finite_points(memory.world_points)
    if int(historical.shape[0]) > int(config.max_historical_points):
        # The memory is already confidence-pruned by V2ObjectMemoryState.  A
        # deterministic stride keeps validation bounded without introducing a
        # new learned or random selection policy.
        indices = torch.linspace(
            0,
            int(historical.shape[0]) - 1,
            steps=int(config.max_historical_points),
        ).round().long()
        historical = historical.index_select(0, indices)
    candidate_points, candidate_weights = sample_masked_observation(
        current_world_points,
        current_confidence,
        candidate_stream_mask.bool(),
        max_points=int(config.max_candidate_points),
    )
    candidate_points = _finite_points(candidate_points)
    historical_count = int(historical.shape[0])
    candidate_count = int(candidate_points.shape[0])
    base = {
        "geometry_validation_attempted": 1,
        "geometry_validation_accepted": 0,
        "candidate_world_point_count": candidate_count,
        "historical_world_point_count": historical_count,
        "candidate_2d_score": float(candidate_row.get("recovery_score", 0.0)),
        "centroid_distance": float("nan"),
        "centroid_threshold": float("nan"),
        "centroid_score": 0.0,
        "point_overlap_ratio": 0.0,
        "point_overlap_threshold": float(config.point_overlap_distance_m),
        "shape_score": 0.0,
        "geometry_consistency_score": 0.0,
        "final_geometry_score": 0.0,
        "geometry_validation_reason": "low_point_overlap",
    }
    if candidate_count < int(config.min_candidate_points):
        base["geometry_validation_reason"] = "low_point_overlap"
        return base
    if historical_count < int(config.min_historical_points):
        base["geometry_validation_reason"] = "low_point_overlap"
        return base

    candidate_center = _centroid(candidate_points)
    historical_center = _centroid(historical)
    center_distance = float(
        torch.linalg.vector_norm(candidate_center - historical_center).item()
    )
    historical_radius = _radius(historical, historical_center)
    candidate_radius = _radius(candidate_points, candidate_center)
    centroid_threshold = max(
        float(config.centroid_distance_threshold_m),
        float(config.centroid_distance_scale)
        * max(historical_radius, candidate_radius, 1.0e-3),
    )
    centroid_score = math.exp(
        -center_distance / max(centroid_threshold, float(config.covariance_epsilon))
    )

    nearest = _nearest_distances(
        candidate_points,
        historical,
        int(config.distance_chunk_size),
    )
    overlap_ratio = float(
        (nearest <= float(config.point_overlap_distance_m)).float().mean().item()
    )
    shape = _shape_score(
        candidate_points,
        historical,
        float(config.covariance_epsilon),
    )
    geometry_score = (
        0.40 * _clamp01(centroid_score)
        + 0.40 * _clamp01(overlap_ratio)
        + 0.20 * _clamp01(shape)
    )
    candidate_2d = _clamp01(float(candidate_row.get("recovery_score", 0.0)))
    weight_total = max(
        float(config.alpha_2d_score) + float(config.beta_3d_score),
        float(config.covariance_epsilon),
    )
    final_score = (
        float(config.alpha_2d_score) * candidate_2d
        + float(config.beta_3d_score) * geometry_score
    ) / weight_total

    base.update(
        {
            "centroid_distance": center_distance,
            "centroid_threshold": centroid_threshold,
            "centroid_score": _clamp01(centroid_score),
            "point_overlap_ratio": _clamp01(overlap_ratio),
            "shape_score": _clamp01(shape),
            "geometry_consistency_score": _clamp01(geometry_score),
            "final_geometry_score": _clamp01(final_score),
        }
    )
    if center_distance > centroid_threshold:
        base["geometry_validation_reason"] = "centroid_far"
        return base
    if overlap_ratio < float(config.min_point_overlap_ratio):
        base["geometry_validation_reason"] = "low_point_overlap"
        return base
    if shape < float(config.min_shape_score):
        base["geometry_validation_reason"] = "shape_inconsistent"
        return base
    if final_score < float(config.final_accept_threshold):
        # Preserve the requested coarse reason taxonomy for downstream counts.
        weakest = min(
            (
                (centroid_score, "centroid_far"),
                (overlap_ratio, "low_point_overlap"),
                (shape, "shape_inconsistent"),
            ),
            key=lambda item: item[0],
        )
        base["geometry_validation_reason"] = weakest[1]
        return base
    base["geometry_validation_accepted"] = 1
    base["accepted"] = 1
    base["geometry_validation_reason"] = "accepted"
    return base


def _clamp01(value: float) -> float:
    if not math.isfinite(float(value)):
        return 0.0
    return max(0.0, min(1.0, float(value)))


__all__ = [
    "V22GeometryValidationConfig",
    "validate_v22_candidate",
]
