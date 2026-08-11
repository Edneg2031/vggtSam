"""Discrete local-token support factorization for V9.2.

This module contains no learned matcher and no pose model.  GT reprojection is
used only to ask which *actual cached history key* is closest to each fixed
current query.  The returned correspondences can then be consumed by the
frozen V9 O-R1 epipolar solver.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from streaming_couping.src.v90_epipolar_geometry import (
    LocalTokenReprojection,
    SurfaceCorrespondences,
)
from streaming_couping.src.v91_token_evidence import normalized_uv_to_pixels


MATCH_STRATEGIES = ("nearest", "mutual", "greedy_unique")


@dataclass(frozen=True)
class SupportDiagnostics:
    """Additive diagnostics for one current/history/instance pair."""

    supervised_queries: int
    visible_queries: int
    coverage_at_8px: int
    coverage_at_12px: int
    coverage_at_16px: int
    accepted_correspondences: int
    unique_history_keys: int
    nearest_supported_assignments: int
    nearest_unique_history_keys: int
    pck_correct: int
    epe_sum_pixels: float
    selected_epe_sum_pixels: float

    @property
    def nearest_collisions(self) -> int:
        return max(
            int(self.nearest_supported_assignments)
            - int(self.nearest_unique_history_keys),
            0,
        )

    def as_pose_match_stats(self) -> dict[str, float]:
        """Return the precomputed fields expected by the V9 pose reporter."""

        dustbin = max(
            int(self.supervised_queries) - int(self.accepted_correspondences),
            0,
        )
        return {
            "supervised_queries": float(self.supervised_queries),
            "visible_queries": float(self.visible_queries),
            "visible_key_supported_queries": float(self.coverage_at_12px),
            "accepted_correspondences": float(self.accepted_correspondences),
            "dustbin_correct": float(dustbin),
            "dustbin_queries": float(dustbin),
            "pck_correct": float(self.pck_correct),
            "epe_sum_pixels": float(self.epe_sum_pixels),
            "metrics_precomputed": 1.0,
        }


@dataclass(frozen=True)
class SupportMatch:
    """One oracle match restricted to the supplied discrete key support."""

    correspondences: SurfaceCorrespondences
    diagnostics: SupportDiagnostics
    selected_query_indices: torch.Tensor
    selected_key_indices: torch.Tensor
    nearest_key_distance_pixels: torch.Tensor


def match_discrete_support(
    labels: LocalTokenReprojection,
    *,
    history_uv_normalized: torch.Tensor,
    history_valid: torch.Tensor,
    image_size: tuple[int, int],
    strategy: str,
    match_radius_pixels: float = 12.0,
    pck_threshold_pixels: float = 8.0,
) -> SupportMatch:
    """Match continuous GT targets to real history keys deterministically.

    ``nearest`` allows many current queries to select the same key. ``mutual``
    retains only bidirectional nearest pairs. ``greedy_unique`` sorts every
    eligible pair by distance and greedily enforces a one-to-one assignment.
    All strategies use the same fixed radius and never create an interpolated
    or continuous history point.
    """

    if strategy not in MATCH_STRATEGIES:
        raise ValueError(f"Unknown V9.2 support strategy={strategy!r}.")
    if history_uv_normalized.ndim != 2 or history_uv_normalized.shape[-1] != 2:
        raise ValueError("V9.2 history UV must have shape [R,2].")
    if history_valid.shape != history_uv_normalized.shape[:-1]:
        raise ValueError("V9.2 history UV/valid shapes disagree.")
    radius = float(match_radius_pixels)
    pck = float(pck_threshold_pixels)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("V9.2 match radius must be positive.")
    if not math.isfinite(pck) or pck <= 0.0 or pck > radius:
        raise ValueError("V9.2 PCK threshold must be in (0, match_radius].")

    key_uv = normalized_uv_to_pixels(history_uv_normalized, image_size)
    query_mask = (
        labels.query_valid.bool()
        & labels.target_visible.bool()
        & torch.isfinite(labels.history_target_uv).all(dim=-1)
    )
    key_mask = history_valid.bool() & torch.isfinite(key_uv).all(dim=-1)
    query_indices = torch.nonzero(query_mask, as_tuple=False).flatten()
    key_indices = torch.nonzero(key_mask, as_tuple=False).flatten()
    visible = int(query_indices.numel())
    supervised = int(labels.query_valid.sum())
    diagonal = float(math.hypot(*image_size))
    nearest_full = torch.full(
        (labels.current_uv.shape[0],),
        float("inf"),
        dtype=torch.float64,
    )

    if visible == 0 or int(key_indices.numel()) == 0:
        diagnostics = SupportDiagnostics(
            supervised_queries=supervised,
            visible_queries=visible,
            coverage_at_8px=0,
            coverage_at_12px=0,
            coverage_at_16px=0,
            accepted_correspondences=0,
            unique_history_keys=0,
            nearest_supported_assignments=0,
            nearest_unique_history_keys=0,
            pck_correct=0,
            epe_sum_pixels=diagonal * visible,
            selected_epe_sum_pixels=0.0,
        )
        empty = torch.empty(0, dtype=torch.long)
        return SupportMatch(
            correspondences=_empty_correspondences(labels),
            diagnostics=diagnostics,
            selected_query_indices=empty,
            selected_key_indices=empty.clone(),
            nearest_key_distance_pixels=nearest_full,
        )

    targets = labels.history_target_uv.index_select(0, query_indices).double()
    keys = key_uv.index_select(0, key_indices).double()
    distance = torch.cdist(targets, keys)
    nearest_distance, nearest_local_key = distance.min(dim=1)
    nearest_full[query_indices] = nearest_distance
    nearest_supported = nearest_distance.le(radius)
    nearest_supported_keys = nearest_local_key[nearest_supported]

    selected_local_queries, selected_local_keys = _select_pairs(
        distance,
        supported=nearest_supported,
        nearest_local_key=nearest_local_key,
        radius=radius,
        strategy=strategy,
    )
    selected_queries = query_indices.index_select(0, selected_local_queries)
    selected_keys = key_indices.index_select(0, selected_local_keys)
    selected_epe = distance[selected_local_queries, selected_local_keys]
    predicted_uv = key_uv.index_select(0, selected_keys).double()
    weights = labels.weights.index_select(0, selected_queries).double()
    finite_weights = torch.isfinite(weights) & weights.gt(0.0)
    weights = torch.where(finite_weights, weights, torch.ones_like(weights))

    accepted = int(selected_queries.numel())
    epe_sum = diagonal * max(visible - accepted, 0) + float(selected_epe.sum())
    diagnostics = SupportDiagnostics(
        supervised_queries=supervised,
        visible_queries=visible,
        coverage_at_8px=int(nearest_distance.le(8.0).sum()),
        coverage_at_12px=int(nearest_distance.le(12.0).sum()),
        coverage_at_16px=int(nearest_distance.le(16.0).sum()),
        accepted_correspondences=accepted,
        unique_history_keys=int(selected_keys.unique().numel()),
        nearest_supported_assignments=int(nearest_supported.sum()),
        nearest_unique_history_keys=int(nearest_supported_keys.unique().numel()),
        pck_correct=int(selected_epe.le(pck).sum()),
        epe_sum_pixels=epe_sum,
        selected_epe_sum_pixels=float(selected_epe.sum()),
    )
    correspondences = SurfaceCorrespondences(
        current_frame=labels.current_frame,
        history_frame=labels.history_frame,
        slot=labels.slot,
        current_uv=labels.current_uv.index_select(0, selected_queries).double(),
        history_uv=predicted_uv,
        weights=weights,
        depth_residual_metric=labels.depth_residual_metric.index_select(
            0, selected_queries
        ).double(),
        sampled_queries=labels.query_count,
        projected_in_bounds=labels.query_count,
        visible_queries=accepted,
    )
    return SupportMatch(
        correspondences=correspondences,
        diagnostics=diagnostics,
        selected_query_indices=selected_queries,
        selected_key_indices=selected_keys,
        nearest_key_distance_pixels=nearest_full,
    )


def _select_pairs(
    distance: torch.Tensor,
    *,
    supported: torch.Tensor,
    nearest_local_key: torch.Tensor,
    radius: float,
    strategy: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if strategy == "nearest":
        query = torch.nonzero(supported, as_tuple=False).flatten()
        return query, nearest_local_key.index_select(0, query)

    if strategy == "mutual":
        key_nearest_query = distance.min(dim=0).indices
        query_rows = torch.arange(distance.shape[0], dtype=torch.long)
        mutual = supported & key_nearest_query.index_select(
            0, nearest_local_key
        ).eq(query_rows)
        query = torch.nonzero(mutual, as_tuple=False).flatten()
        return query, nearest_local_key.index_select(0, query)

    if strategy != "greedy_unique":
        raise ValueError(f"Unknown V9.2 support strategy={strategy!r}.")
    eligible = torch.nonzero(distance.le(float(radius)), as_tuple=False)
    candidates = sorted(
        (
            float(distance[int(row[0]), int(row[1])]),
            int(row[0]),
            int(row[1]),
        )
        for row in eligible.tolist()
    )
    occupied_queries: set[int] = set()
    occupied_keys: set[int] = set()
    selected: list[tuple[int, int]] = []
    for _, query, key in candidates:
        if query in occupied_queries or key in occupied_keys:
            continue
        occupied_queries.add(query)
        occupied_keys.add(key)
        selected.append((query, key))
    selected.sort()
    if not selected:
        empty = torch.empty(0, dtype=torch.long)
        return empty, empty.clone()
    return (
        torch.tensor([row[0] for row in selected], dtype=torch.long),
        torch.tensor([row[1] for row in selected], dtype=torch.long),
    )


def _empty_correspondences(
    labels: LocalTokenReprojection,
) -> SurfaceCorrespondences:
    empty_uv = torch.empty((0, 2), dtype=torch.float64)
    empty = torch.empty(0, dtype=torch.float64)
    return SurfaceCorrespondences(
        current_frame=labels.current_frame,
        history_frame=labels.history_frame,
        slot=labels.slot,
        current_uv=empty_uv,
        history_uv=empty_uv.clone(),
        weights=empty,
        depth_residual_metric=empty.clone(),
        sampled_queries=labels.query_count,
        projected_in_bounds=labels.query_count,
        visible_queries=0,
    )
