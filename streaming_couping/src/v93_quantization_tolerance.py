"""Quantization and soft-coordinate upper bounds for V9.3.

All functions are diagnostic oracles.  They never read a descriptor, train a
matcher, or estimate pose.  GT visibility/target UV may be used to construct a
controlled correspondence row that is later consumed by the frozen V9 solver.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math

import torch

from streaming_couping.src.v90_epipolar_geometry import (
    LocalTokenReprojection,
    SurfaceCorrespondences,
)
from streaming_couping.src.v91_token_evidence import normalized_uv_to_pixels
from streaming_couping.src.v92_support_factorization import match_discrete_support


@dataclass(frozen=True)
class PredictionDiagnostics:
    supervised_queries: int
    visible_queries: int
    accepted_correspondences: int
    pck_correct: int
    epe_sum_pixels: float
    selected_epe_sum_pixels: float
    exact_interpolations: int

    def as_pose_match_stats(self) -> dict[str, float]:
        dustbin = max(
            int(self.supervised_queries) - int(self.accepted_correspondences), 0
        )
        return {
            "supervised_queries": float(self.supervised_queries),
            "visible_queries": float(self.visible_queries),
            "visible_key_supported_queries": float(
                self.accepted_correspondences
            ),
            "accepted_correspondences": float(self.accepted_correspondences),
            "dustbin_correct": float(dustbin),
            "dustbin_queries": float(dustbin),
            "pck_correct": float(self.pck_correct),
            "epe_sum_pixels": float(self.epe_sum_pixels),
            "metrics_precomputed": 1.0,
        }


@dataclass(frozen=True)
class OraclePrediction:
    correspondences: SurfaceCorrespondences
    diagnostics: PredictionDiagnostics
    selected_query_indices: torch.Tensor
    selected_errors_pixels: torch.Tensor


def continuous_prediction(
    labels: LocalTokenReprojection,
    *,
    image_size: tuple[int, int],
    pck_threshold_pixels: float = 8.0,
) -> OraclePrediction:
    """Return exact continuous visible GT UV as the positive control."""

    visible = labels.target_visible.bool() & labels.query_valid.bool()
    indices = torch.nonzero(visible, as_tuple=False).flatten()
    errors = torch.zeros(int(indices.numel()), dtype=torch.float64)
    return _prediction_from_rows(
        labels,
        selected_query_indices=indices,
        predicted_history_uv=labels.history_target_uv.index_select(
            0, indices
        ).double(),
        selected_errors=errors,
        visible_queries=int(visible.sum()),
        image_size=image_size,
        pck_threshold_pixels=pck_threshold_pixels,
        exact_interpolations=int(indices.numel()),
    )


def hard_nearest_prediction(
    labels: LocalTokenReprojection,
    *,
    history_uv_normalized: torch.Tensor,
    history_valid: torch.Tensor,
    image_size: tuple[int, int],
    match_radius_pixels: float = 12.0,
    pck_threshold_pixels: float = 8.0,
) -> OraclePrediction:
    """Wrap the V9.2 256-key hard-nearest oracle in V9.3 diagnostics."""

    matched = match_discrete_support(
        labels,
        history_uv_normalized=history_uv_normalized,
        history_valid=history_valid,
        image_size=image_size,
        strategy="nearest",
        match_radius_pixels=match_radius_pixels,
        pck_threshold_pixels=pck_threshold_pixels,
    )
    indices = matched.selected_query_indices
    errors = matched.nearest_key_distance_pixels.index_select(0, indices)
    return OraclePrediction(
        correspondences=matched.correspondences,
        diagnostics=_diagnostics(
            labels,
            visible_queries=matched.diagnostics.visible_queries,
            selected_errors=errors,
            image_size=image_size,
            pck_threshold_pixels=pck_threshold_pixels,
            exact_interpolations=int(errors.le(1e-6).sum()),
        ),
        selected_query_indices=indices,
        selected_errors_pixels=errors,
    )


def soft_knn_convex_prediction(
    labels: LocalTokenReprojection,
    *,
    history_uv_normalized: torch.Tensor,
    history_valid: torch.Tensor,
    image_size: tuple[int, int],
    neighbors: int,
    match_radius_pixels: float = 12.0,
    pck_threshold_pixels: float = 8.0,
) -> OraclePrediction:
    """Project each target onto the convex hull of its K nearest real keys.

    An attention/soft-expectation decoder can express any point in this convex
    hull.  GT chooses the optimal convex point, so this is a representational
    upper bound rather than a deployable matcher.
    """

    k = int(neighbors)
    if k not in {2, 4, 8}:
        raise ValueError("V9.3 soft convex K must be 2, 4 or 8.")
    radius = float(match_radius_pixels)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("V9.3 match radius must be positive.")
    key_uv = normalized_uv_to_pixels(history_uv_normalized, image_size)
    key_mask = history_valid.bool() & torch.isfinite(key_uv).all(dim=-1)
    key_indices = torch.nonzero(key_mask, as_tuple=False).flatten()
    visible = (
        labels.target_visible.bool()
        & labels.query_valid.bool()
        & torch.isfinite(labels.history_target_uv).all(dim=-1)
    )
    query_indices = torch.nonzero(visible, as_tuple=False).flatten()
    if not int(query_indices.numel()) or not int(key_indices.numel()):
        return _prediction_from_rows(
            labels,
            selected_query_indices=torch.empty(0, dtype=torch.long),
            predicted_history_uv=torch.empty((0, 2), dtype=torch.float64),
            selected_errors=torch.empty(0, dtype=torch.float64),
            visible_queries=int(query_indices.numel()),
            image_size=image_size,
            pck_threshold_pixels=pck_threshold_pixels,
            exact_interpolations=0,
        )

    targets = labels.history_target_uv.index_select(0, query_indices).double()
    keys = key_uv.index_select(0, key_indices).double()
    distance = torch.cdist(targets, keys)
    nearest = distance.min(dim=-1).values
    supported = nearest.le(radius)
    supported_rows = torch.nonzero(supported, as_tuple=False).flatten()
    selected_queries = query_indices.index_select(0, supported_rows)
    predicted = []
    errors = []
    for row in supported_rows.tolist():
        count = min(k, int(keys.shape[0]))
        local_indices = distance[row].topk(count, largest=False).indices
        point = project_to_convex_hull_2d(
            targets[row], keys.index_select(0, local_indices)
        )
        predicted.append(point)
        errors.append(torch.linalg.vector_norm(point - targets[row]))
    predicted_uv = (
        torch.stack(predicted)
        if predicted
        else torch.empty((0, 2), dtype=torch.float64)
    )
    error = (
        torch.stack(errors)
        if errors
        else torch.empty(0, dtype=torch.float64)
    )
    return _prediction_from_rows(
        labels,
        selected_query_indices=selected_queries,
        predicted_history_uv=predicted_uv,
        selected_errors=error,
        visible_queries=int(query_indices.numel()),
        image_size=image_size,
        pck_threshold_pixels=pck_threshold_pixels,
        exact_interpolations=int(error.le(1e-6).sum()),
    )


def filter_prediction_by_oracle_error(
    prediction: OraclePrediction,
    labels: LocalTokenReprojection,
    *,
    max_error_pixels: float,
    image_size: tuple[int, int],
    pck_threshold_pixels: float = 8.0,
) -> OraclePrediction:
    """Retain hard matches below a GT error threshold (diagnostic only)."""

    threshold = float(max_error_pixels)
    if threshold not in {1.0, 2.0, 4.0, 8.0}:
        raise ValueError("V9.3 filter threshold must be 1/2/4/8 pixels.")
    keep = prediction.selected_errors_pixels.le(threshold)
    rows = torch.nonzero(keep, as_tuple=False).flatten()
    query_indices = prediction.selected_query_indices.index_select(0, rows)
    predicted_uv = prediction.correspondences.history_uv.index_select(0, rows)
    errors = prediction.selected_errors_pixels.index_select(0, rows)
    return _prediction_from_rows(
        labels,
        selected_query_indices=query_indices,
        predicted_history_uv=predicted_uv,
        selected_errors=errors,
        visible_queries=prediction.diagnostics.visible_queries,
        image_size=image_size,
        pck_threshold_pixels=pck_threshold_pixels,
        exact_interpolations=int(errors.le(1e-6).sum()),
    )


def noisy_continuous_prediction(
    labels: LocalTokenReprojection,
    *,
    sigma_pixels: float,
    seed: int,
    image_size: tuple[int, int],
    pck_threshold_pixels: float = 8.0,
) -> OraclePrediction:
    """Add deterministic isotropic pixel noise to the continuous GT target."""

    sigma = float(sigma_pixels)
    if sigma not in {0.5, 1.0, 2.0, 4.0, 6.0}:
        raise ValueError("V9.3 noise sigma must be 0.5/1/2/4/6 pixels.")
    visible = labels.target_visible.bool() & labels.query_valid.bool()
    indices = torch.nonzero(visible, as_tuple=False).flatten()
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    noise = torch.randn(
        (int(indices.numel()), 2), generator=generator, dtype=torch.float64
    ) * sigma
    target = labels.history_target_uv.index_select(0, indices).double()
    predicted = target + noise
    errors = torch.linalg.vector_norm(noise, dim=-1)
    return _prediction_from_rows(
        labels,
        selected_query_indices=indices,
        predicted_history_uv=predicted,
        selected_errors=errors,
        visible_queries=int(indices.numel()),
        image_size=image_size,
        pck_threshold_pixels=pck_threshold_pixels,
        exact_interpolations=0,
    )


def project_to_convex_hull_2d(
    target: torch.Tensor, points: torch.Tensor
) -> torch.Tensor:
    """Return the Euclidean projection onto a small 2-D convex hull."""

    if target.shape != (2,) or points.ndim != 2 or points.shape[-1] != 2:
        raise ValueError("V9.3 convex projection expects target [2], points [K,2].")
    if int(points.shape[0]) < 1:
        raise ValueError("V9.3 convex projection requires at least one point.")
    target = target.double()
    points = points.double()
    best = points[torch.linalg.vector_norm(points - target, dim=-1).argmin()]
    best_error = float(torch.linalg.vector_norm(best - target))

    # By Caratheodory's theorem, any point in a 2-D convex hull is contained
    # in a triangle of its vertices.  Returning target is therefore exact.
    for first, second, third in itertools.combinations(range(points.shape[0]), 3):
        triangle = points[[first, second, third]]
        matrix = torch.stack(
            [triangle[0] - triangle[2], triangle[1] - triangle[2]], dim=1
        )
        determinant = float(torch.linalg.det(matrix))
        if abs(determinant) <= 1e-12:
            continue
        first_two = torch.linalg.solve(matrix, target - triangle[2])
        weights = torch.stack(
            [first_two[0], first_two[1], 1.0 - first_two.sum()]
        )
        if bool(weights.ge(-1e-10).all()):
            return target.clone()

    # Outside the hull, the closest point lies on a hull edge.  Enumerating
    # every segment also includes every hull edge and remains tiny for K<=8.
    for first, second in itertools.combinations(range(points.shape[0]), 2):
        start, end = points[first], points[second]
        direction = end - start
        denominator = float(direction.square().sum())
        if denominator <= 1e-12:
            continue
        amount = float(torch.dot(target - start, direction)) / denominator
        candidate = start + min(max(amount, 0.0), 1.0) * direction
        error = float(torch.linalg.vector_norm(candidate - target))
        if error < best_error:
            best, best_error = candidate, error
    return best.clone()


def _prediction_from_rows(
    labels: LocalTokenReprojection,
    *,
    selected_query_indices: torch.Tensor,
    predicted_history_uv: torch.Tensor,
    selected_errors: torch.Tensor,
    visible_queries: int,
    image_size: tuple[int, int],
    pck_threshold_pixels: float,
    exact_interpolations: int,
) -> OraclePrediction:
    indices = selected_query_indices.long().cpu()
    predicted = predicted_history_uv.double().cpu()
    errors = selected_errors.double().cpu()
    if predicted.shape != (int(indices.numel()), 2):
        raise ValueError("V9.3 predicted UV/query row shapes disagree.")
    if errors.shape != (int(indices.numel()),):
        raise ValueError("V9.3 selected error/query row shapes disagree.")
    weights = labels.weights.index_select(0, indices).double()
    weights = torch.where(
        torch.isfinite(weights) & weights.gt(0.0), weights, torch.ones_like(weights)
    )
    correspondences = SurfaceCorrespondences(
        current_frame=labels.current_frame,
        history_frame=labels.history_frame,
        slot=labels.slot,
        current_uv=labels.current_uv.index_select(0, indices).double(),
        history_uv=predicted,
        weights=weights,
        depth_residual_metric=labels.depth_residual_metric.index_select(
            0, indices
        ).double(),
        sampled_queries=labels.query_count,
        projected_in_bounds=labels.query_count,
        visible_queries=int(indices.numel()),
    )
    return OraclePrediction(
        correspondences=correspondences,
        diagnostics=_diagnostics(
            labels,
            visible_queries=visible_queries,
            selected_errors=errors,
            image_size=image_size,
            pck_threshold_pixels=pck_threshold_pixels,
            exact_interpolations=exact_interpolations,
        ),
        selected_query_indices=indices,
        selected_errors_pixels=errors,
    )


def _diagnostics(
    labels: LocalTokenReprojection,
    *,
    visible_queries: int,
    selected_errors: torch.Tensor,
    image_size: tuple[int, int],
    pck_threshold_pixels: float,
    exact_interpolations: int,
) -> PredictionDiagnostics:
    accepted = int(selected_errors.numel())
    diagonal = float(math.hypot(*image_size))
    epe_sum = float(selected_errors.sum()) + diagonal * max(
        int(visible_queries) - accepted, 0
    )
    return PredictionDiagnostics(
        supervised_queries=int(labels.query_valid.sum()),
        visible_queries=int(visible_queries),
        accepted_correspondences=accepted,
        pck_correct=int(selected_errors.le(float(pck_threshold_pixels)).sum()),
        epe_sum_pixels=epe_sum,
        selected_epe_sum_pixels=float(selected_errors.sum()),
        exact_interpolations=int(exact_interpolations),
    )
