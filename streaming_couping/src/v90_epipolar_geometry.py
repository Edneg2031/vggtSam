"""Fixed 2D epipolar geometry for the V9 SAM-token causality study.

This module deliberately contains no learned component and never consumes a
predicted depth or pointmap.  GT mesh points and GT camera poses are accepted
only by :func:`surface_reprojection_correspondences`, which constructs the
Stage-O diagnostic labels.  Pose recovery itself receives pixels, calibrated
intrinsics and frozen L0 camera poses.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from streaming_couping.src.v80_pose_geometry import (
    camera_centers,
    homogeneous,
    invert_rigid,
)


@dataclass(frozen=True)
class VisibilityConfig:
    max_queries_per_instance: int = 256
    depth_tolerance_metric: float = 0.03
    relative_depth_tolerance: float = 0.01


@dataclass(frozen=True)
class EpipolarConfig:
    min_correspondences: int = 8
    min_design_rank_ratio: float = 1e-7
    min_cheirality_fraction: float = 0.50
    refinement_iterations: int = 5
    refinement_huber_delta: float = 0.002
    refinement_damping: float = 1e-6
    refinement_step_epsilon: float = 1e-6
    max_refinement_step: float = 0.10
    cheirality_max_points: int = 128


@dataclass
class SurfaceCorrespondences:
    current_frame: int
    history_frame: int
    slot: int
    current_uv: torch.Tensor
    history_uv: torch.Tensor
    weights: torch.Tensor
    depth_residual_metric: torch.Tensor
    sampled_queries: int
    projected_in_bounds: int
    visible_queries: int

    @property
    def count(self) -> int:
        return int(self.current_uv.shape[0])


@dataclass
class EpipolarEstimate:
    rotation_current_to_history: torch.Tensor
    translation_current_origin_in_history: torch.Tensor
    essential: torch.Tensor
    sampson_rmse: float
    inlier_ratio: float
    cheirality_fraction: float
    design_rank_ratio: float
    design_condition: float
    correspondences: int
    effective_correspondences: float
    refinement_iterations: int
    initialization: str
    eight_point_sampson_rmse: float
    l0_local_sampson_rmse: float
    success: bool
    reason: str


@dataclass
class AbsolutePoseEstimate:
    world_to_camera: torch.Tensor
    edge_estimates: tuple[EpipolarEstimate, ...]
    edge_history_indices: tuple[int, ...]
    edge_weights: tuple[float, ...]
    used_ray_intersection: bool
    success: bool
    reason: str


@dataclass
class _RelativeCandidate:
    rotation: torch.Tensor
    translation: torch.Tensor
    essential: torch.Tensor
    sampson_rmse: float
    robust_objective: float
    inlier_ratio: float
    cheirality_fraction: float
    refinement_iterations: int
    initialization: str


def causal_mask_history_indices(
    masks: torch.Tensor,
    *,
    max_history: int = 2,
) -> torch.Tensor:
    """Return the most recent strictly-earlier non-empty mask per slot.

    A slot birth is therefore only a memory write.  It cannot be read until a
    later observation, and the current observation is written after its read.
    """

    if masks.ndim != 4:
        raise ValueError("V9 masks must have shape [S,K,H,W].")
    if int(max_history) < 1:
        raise ValueError("V9 max_history must be positive.")
    sequence, slots = masks.shape[:2]
    bank = torch.full(
        (slots, int(max_history)), -1, dtype=torch.long, device=masks.device
    )
    output = torch.full(
        (sequence, slots, int(max_history)),
        -1,
        dtype=torch.long,
        device=masks.device,
    )
    observed = masks.flatten(2).any(dim=-1)
    for frame in range(sequence):
        output[frame] = bank
        write = observed[frame]
        if bool(write.any()):
            shifted = torch.cat(
                [
                    torch.full(
                        (slots, 1),
                        frame,
                        dtype=torch.long,
                        device=masks.device,
                    ),
                    bank[:, :-1],
                ],
                dim=-1,
            )
            bank = torch.where(write[:, None], shifted, bank)
    return output


def surface_reprojection_correspondences(
    *,
    current_frame: int,
    history_frame: int,
    slot: int,
    masks: torch.Tensor,
    world_points_metric: torch.Tensor,
    depth_metric: torch.Tensor,
    global_world_to_camera: torch.Tensor,
    intrinsics: torch.Tensor,
    config: VisibilityConfig,
) -> SurfaceCorrespondences:
    """Reproject current visible mesh pixels into one historical slot mask."""

    _validate_surface_inputs(
        masks=masks,
        world_points_metric=world_points_metric,
        depth_metric=depth_metric,
        global_world_to_camera=global_world_to_camera,
        intrinsics=intrinsics,
    )
    sequence, slots, height, width = masks.shape
    current = int(current_frame)
    history = int(history_frame)
    slot_index = int(slot)
    if not (0 <= history < current < sequence):
        raise ValueError("V9 surface correspondence must be strictly causal.")
    if not 0 <= slot_index < slots:
        raise ValueError("V9 slot index is out of range.")

    world = world_points_metric[current]
    current_depth = depth_metric[current]
    valid = (
        masks[current, slot_index].bool()
        & torch.isfinite(world).all(dim=-1)
        & torch.isfinite(current_depth)
        & current_depth.gt(1e-8)
        & torch.linalg.vector_norm(world, dim=-1).gt(1e-8)
    )
    yx = torch.nonzero(valid, as_tuple=False)
    sampled_queries = min(int(yx.shape[0]), int(config.max_queries_per_instance))
    if sampled_queries == 0:
        return _empty_surface(current, history, slot_index, world.dtype)
    yx = _evenly_subsample_rows(yx, sampled_queries)
    points = world[yx[:, 0], yx[:, 1]].double()

    history_pose = homogeneous(global_world_to_camera[history]).double()
    camera = torch.einsum("ij,nj->ni", history_pose[:3, :3], points)
    camera = camera + history_pose[:3, 3]
    z = camera[:, 2]
    k = intrinsics[history].double()
    projected = torch.einsum("ij,nj->ni", k, camera)
    safe_z = torch.where(z.abs().gt(1e-12), z, torch.ones_like(z))
    uv = projected[:, :2] / safe_z[:, None]
    finite_projection = torch.isfinite(uv).all(dim=-1) & torch.isfinite(z)
    in_bounds = (
        finite_projection
        & z.gt(1e-8)
        & uv[:, 0].ge(0.0)
        & uv[:, 0].le(float(width - 1))
        & uv[:, 1].ge(0.0)
        & uv[:, 1].le(float(height - 1))
    )
    projected_in_bounds = int(in_bounds.sum())
    if not bool(in_bounds.any()):
        return _empty_surface(
            current,
            history,
            slot_index,
            world.dtype,
            sampled_queries=sampled_queries,
            projected_in_bounds=projected_in_bounds,
        )

    history_depth = _bilinear_sample(depth_metric[history].double(), uv)
    history_mask = _nearest_sample_mask(masks[history, slot_index], uv)
    tolerance = torch.maximum(
        torch.full_like(z, float(config.depth_tolerance_metric)),
        z.abs() * float(config.relative_depth_tolerance),
    )
    depth_residual = (history_depth - z).abs()
    visible = (
        in_bounds
        & history_mask
        & torch.isfinite(history_depth)
        & history_depth.gt(1e-8)
        & depth_residual.le(tolerance)
    )
    visible_queries = int(visible.sum())
    current_uv = torch.stack(
        [yx[:, 1].double(), yx[:, 0].double()], dim=-1
    )[visible]
    history_uv = uv[visible]
    residual = depth_residual[visible]
    visible_tolerance = tolerance[visible].clamp_min(1e-12)
    weights = torch.exp(-0.5 * (residual / visible_tolerance).square())
    return SurfaceCorrespondences(
        current_frame=current,
        history_frame=history,
        slot=slot_index,
        current_uv=current_uv,
        history_uv=history_uv,
        weights=weights,
        depth_residual_metric=residual,
        sampled_queries=sampled_queries,
        projected_in_bounds=projected_in_bounds,
        visible_queries=visible_queries,
    )


def concatenate_correspondences(
    rows: list[SurfaceCorrespondences],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    nonempty = [row for row in rows if row.count]
    if not nonempty:
        empty_uv = torch.empty(0, 2, dtype=torch.float64)
        return empty_uv, empty_uv.clone(), torch.empty(0, dtype=torch.float64)
    return (
        torch.cat([row.current_uv.double() for row in nonempty], dim=0),
        torch.cat([row.history_uv.double() for row in nonempty], dim=0),
        torch.cat([row.weights.double() for row in nonempty], dim=0),
    )


def estimate_relative_epipolar_pose(
    current_uv: torch.Tensor,
    history_uv: torch.Tensor,
    weights: torch.Tensor,
    current_intrinsics: torch.Tensor,
    history_intrinsics: torch.Tensor,
    l0_current_to_history: torch.Tensor,
    *,
    config: EpipolarConfig,
) -> EpipolarEstimate:
    """Estimate ``X_history = R X_current + t`` from calibrated pixels."""

    count = int(current_uv.shape[0])
    failure = _failed_epipolar(count)
    if (
        current_uv.ndim != 2
        or current_uv.shape[-1] != 2
        or history_uv.shape != current_uv.shape
        or weights.shape != (count,)
    ):
        raise ValueError("V9 epipolar inputs must be [N,2], [N,2], [N].")
    finite = (
        torch.isfinite(current_uv).all(dim=-1)
        & torch.isfinite(history_uv).all(dim=-1)
        & torch.isfinite(weights)
        & weights.gt(0.0)
    )
    current_uv = current_uv[finite].double()
    history_uv = history_uv[finite].double()
    weights = weights[finite].double()
    count = int(current_uv.shape[0])
    if count < int(config.min_correspondences):
        failure.correspondences = count
        failure.reason = "fewer_than_min_correspondences"
        return failure

    current = pixels_to_calibrated_points(current_uv, current_intrinsics)
    history = pixels_to_calibrated_points(history_uv, history_intrinsics)
    try:
        essential, ratio, condition = _weighted_eight_point(
            current, history, weights
        )
    except RuntimeError:
        failure.correspondences = count
        failure.reason = "svd_failed"
        return failure
    failure.design_rank_ratio = float(ratio)
    failure.design_condition = float(condition)
    failure.correspondences = count
    failure.effective_correspondences = _effective_count(weights)
    if (
        not math.isfinite(ratio)
        or ratio < float(config.min_design_rank_ratio)
    ):
        failure.reason = "degenerate_design_matrix"
        return failure

    l0 = homogeneous(l0_current_to_history).double()
    selected = _select_essential_solution(
        essential,
        current,
        history,
        l0,
        max_points=int(config.cheirality_max_points),
    )
    candidates: list[_RelativeCandidate] = []
    eight_point_rmse = float("nan")
    if selected is not None:
        rotation, translation, _ = selected
        eight_candidate = _build_relative_candidate(
            rotation,
            translation,
            current,
            history,
            weights,
            initialization="eight_point",
            config=config,
        )
        eight_point_rmse = eight_candidate.sampson_rmse
        if (
            math.isfinite(eight_candidate.robust_objective)
            and eight_candidate.cheirality_fraction
            >= float(config.min_cheirality_fraction)
        ):
            candidates.append(eight_candidate)

    l0_translation = l0[:3, 3]
    l0_local_rmse = float("nan")
    if (
        torch.isfinite(l0).all()
        and float(torch.linalg.vector_norm(l0_translation)) > 1e-12
    ):
        l0_candidate = _build_relative_candidate(
            l0[:3, :3],
            l0_translation,
            current,
            history,
            weights,
            initialization="l0_local",
            config=config,
        )
        l0_local_rmse = l0_candidate.sampson_rmse
        if (
            math.isfinite(l0_candidate.robust_objective)
            and l0_candidate.cheirality_fraction
            >= float(config.min_cheirality_fraction)
        ):
            candidates.append(l0_candidate)
    if not candidates:
        failure.reason = "no_cheirality_valid_initialization"
        failure.eight_point_sampson_rmse = eight_point_rmse
        failure.l0_local_sampson_rmse = l0_local_rmse
        return failure

    # This is solver-internal model selection on the observed epipolar
    # objective.  It never reads a GT pose error and introduces no accept
    # threshold.  L0 distance only provides a deterministic numerical tie.
    candidates.sort(
        key=lambda candidate: (
            candidate.robust_objective,
            _candidate_l0_distance(candidate, l0),
        )
    )
    candidate = candidates[0]
    return EpipolarEstimate(
        rotation_current_to_history=candidate.rotation,
        translation_current_origin_in_history=candidate.translation,
        essential=candidate.essential,
        sampson_rmse=candidate.sampson_rmse,
        inlier_ratio=candidate.inlier_ratio,
        cheirality_fraction=candidate.cheirality_fraction,
        design_rank_ratio=float(ratio),
        design_condition=float(condition),
        correspondences=count,
        effective_correspondences=_effective_count(weights),
        refinement_iterations=candidate.refinement_iterations,
        initialization=candidate.initialization,
        eight_point_sampson_rmse=eight_point_rmse,
        l0_local_sampson_rmse=l0_local_rmse,
        success=True,
        reason="ok",
    )


def recover_absolute_pose(
    *,
    current_index: int,
    baseline_world_to_camera: torch.Tensor,
    edge_history_indices: list[int],
    edge_estimates: list[EpipolarEstimate],
    config: EpipolarConfig,
) -> AbsolutePoseEstimate:
    """Compose relative edge solutions with historical frozen L0 poses."""

    baseline = homogeneous(baseline_world_to_camera).double()
    current = int(current_index)
    fallback = baseline[current].clone()
    valid = [
        (int(history), estimate)
        for history, estimate in zip(edge_history_indices, edge_estimates)
        if estimate.success and 0 <= int(history) < current
    ]
    if not valid:
        return AbsolutePoseEstimate(
            world_to_camera=fallback,
            edge_estimates=tuple(edge_estimates),
            edge_history_indices=tuple(int(value) for value in edge_history_indices),
            edge_weights=tuple(),
            used_ray_intersection=False,
            success=False,
            reason="no_solvable_history_edge",
        )

    centers = camera_centers(baseline)
    l0_current_center = centers[current]
    rotations = []
    edge_centers = []
    edge_weights = []
    for history, estimate in valid:
        history_rotation = baseline[history, :3, :3]
        absolute_rotation = (
            estimate.rotation_current_to_history.transpose(0, 1)
            @ history_rotation
        )
        direction = history_rotation.transpose(0, 1) @ (
            estimate.translation_current_origin_in_history
        )
        direction = direction / torch.linalg.vector_norm(direction).clamp_min(1e-12)
        edge_length = torch.linalg.vector_norm(
            l0_current_center - centers[history]
        )
        rotations.append(absolute_rotation)
        edge_centers.append(centers[history] + edge_length * direction)
        edge_weights.append(_absolute_edge_weight(estimate))

    weight_tensor = torch.tensor(edge_weights, dtype=torch.float64)
    rotation = _weighted_rotation_average(torch.stack(rotations), weight_tensor)
    normalized_weights = weight_tensor / weight_tensor.sum().clamp_min(1e-12)
    center = (normalized_weights[:, None] * torch.stack(edge_centers)).sum(dim=0)
    if not torch.isfinite(rotation).all() or not torch.isfinite(center).all():
        return AbsolutePoseEstimate(
            world_to_camera=fallback,
            edge_estimates=tuple(edge_estimates),
            edge_history_indices=tuple(int(value) for value in edge_history_indices),
            edge_weights=tuple(edge_weights),
            used_ray_intersection=False,
            success=False,
            reason="nonfinite_absolute_pose",
        )
    output = torch.eye(4, dtype=torch.float64)
    output[:3, :3] = rotation
    output[:3, 3] = -(rotation @ center)
    return AbsolutePoseEstimate(
        world_to_camera=output,
        edge_estimates=tuple(edge_estimates),
        edge_history_indices=tuple(int(value) for value in edge_history_indices),
        edge_weights=tuple(edge_weights),
        used_ray_intersection=False,
        success=True,
        reason="ok",
    )


def pixels_to_calibrated_points(
    uv: torch.Tensor, intrinsics: torch.Tensor
) -> torch.Tensor:
    if uv.ndim != 2 or uv.shape[-1] != 2:
        raise ValueError("Pixel coordinates must have shape [N,2].")
    k = torch.as_tensor(intrinsics).double()
    if k.shape != (3, 3) or not torch.isfinite(k).all():
        raise ValueError("Intrinsics must be a finite [3,3] matrix.")
    homogeneous_uv = torch.cat(
        [uv.double(), torch.ones(len(uv), 1, dtype=torch.float64)], dim=-1
    )
    calibrated = torch.linalg.solve(k, homogeneous_uv.transpose(0, 1)).transpose(0, 1)
    return calibrated / calibrated[:, 2:3].clamp_min(1e-12)


def translation_direction_error_degrees(
    predicted_world_to_camera: torch.Tensor,
    target_world_to_camera: torch.Tensor,
    *,
    reference_index: int = 0,
) -> torch.Tensor:
    """Angle between reference-to-camera center directions in one gauge."""

    predicted_center = camera_centers(predicted_world_to_camera)
    target_center = camera_centers(target_world_to_camera)
    predicted_direction = predicted_center - predicted_center[int(reference_index)]
    target_direction = target_center - target_center[int(reference_index)]
    predicted_norm = torch.linalg.vector_norm(predicted_direction, dim=-1)
    target_norm = torch.linalg.vector_norm(target_direction, dim=-1)
    denominator = predicted_norm * target_norm
    cosine = (predicted_direction * target_direction).sum(dim=-1) / denominator.clamp_min(1e-12)
    angle = torch.rad2deg(torch.acos(cosine.clamp(-1.0, 1.0)))
    return torch.where(denominator.gt(1e-12), angle, torch.zeros_like(angle))


def relative_translation_direction_error_degrees(
    rotation_current_to_history: torch.Tensor,
    translation_current_origin_in_history: torch.Tensor,
    target_current_to_history: torch.Tensor,
) -> float:
    del rotation_current_to_history  # Kept in the API to make the convention explicit.
    target = homogeneous(target_current_to_history).double()
    predicted = translation_current_origin_in_history.double()
    truth = target[:3, 3]
    denominator = (
        torch.linalg.vector_norm(predicted) * torch.linalg.vector_norm(truth)
    )
    if float(denominator) <= 1e-12:
        return 0.0
    cosine = torch.dot(predicted, truth) / denominator
    return float(torch.rad2deg(torch.acos(cosine.clamp(-1.0, 1.0))))


def _weighted_eight_point(
    current: torch.Tensor,
    history: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, float, float]:
    current_n, transform_current = _hartley_normalize(current, weights)
    history_n, transform_history = _hartley_normalize(history, weights)
    x1, y1 = current_n[:, 0], current_n[:, 1]
    x2, y2 = history_n[:, 0], history_n[:, 1]
    design = torch.stack(
        [
            x2 * x1,
            x2 * y1,
            x2,
            y2 * x1,
            y2 * y1,
            y2,
            x1,
            y1,
            torch.ones_like(x1),
        ],
        dim=-1,
    )
    normalized_weights = weights / weights.mean().clamp_min(1e-12)
    weighted = design * normalized_weights.sqrt()[:, None]
    _, singular, vh = torch.linalg.svd(weighted, full_matrices=True)
    if singular.numel() < 8 or vh.shape != (9, 9):
        raise RuntimeError("V9 eight-point SVD has an unexpected shape.")
    support_singular = singular[-1] if singular.numel() == 8 else singular[-2]
    ratio = float(support_singular / singular[0].clamp_min(1e-15))
    condition = float(singular[0] / support_singular.clamp_min(1e-15))
    essential_n = vh[-1].reshape(3, 3)
    essential = transform_history.transpose(0, 1) @ essential_n @ transform_current
    essential = _project_to_essential(essential)
    norm = torch.linalg.vector_norm(essential)
    if not torch.isfinite(norm) or float(norm) <= 1e-15:
        raise RuntimeError("V9 essential matrix is non-finite or zero.")
    return essential / norm, ratio, condition


def _hartley_normalize(
    points: torch.Tensor, weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    normalized_weights = weights / weights.sum().clamp_min(1e-12)
    center = (normalized_weights[:, None] * points[:, :2]).sum(dim=0)
    radius = torch.sqrt(
        (normalized_weights * (points[:, :2] - center).square().sum(dim=-1)).sum()
    )
    scale = math.sqrt(2.0) / float(radius.clamp_min(1e-12))
    transform = torch.tensor(
        [
            [scale, 0.0, -scale * float(center[0])],
            [0.0, scale, -scale * float(center[1])],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    output = torch.einsum("ij,nj->ni", transform, points)
    return output / output[:, 2:3].clamp_min(1e-12), transform


def _project_to_essential(matrix: torch.Tensor) -> torch.Tensor:
    u, singular, vh = torch.linalg.svd(matrix)
    if torch.det(u @ vh) < 0:
        u = u.clone()
        u[:, -1] *= -1.0
    value = 0.5 * (singular[0] + singular[1])
    return u @ torch.diag(torch.stack([value, value, value.new_zeros(())])) @ vh


def _select_essential_solution(
    essential: torch.Tensor,
    current: torch.Tensor,
    history: torch.Tensor,
    l0_current_to_history: torch.Tensor,
    *,
    max_points: int,
) -> tuple[torch.Tensor, torch.Tensor, float] | None:
    u, _, vh = torch.linalg.svd(essential)
    if torch.det(u) < 0:
        u = u.clone()
        u[:, -1] *= -1.0
    if torch.det(vh) < 0:
        vh = vh.clone()
        vh[-1] *= -1.0
    w = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    rotations = (u @ w @ vh, u @ w.transpose(0, 1) @ vh)
    translation = u[:, 2]
    candidates = []
    l0_rotation = l0_current_to_history[:3, :3]
    l0_translation = l0_current_to_history[:3, 3]
    for rotation in rotations:
        if torch.det(rotation) < 0:
            rotation = -rotation
        for sign in (1.0, -1.0):
            direction = translation * sign
            cheirality = _cheirality_fraction(
                rotation,
                direction,
                current,
                history,
                max_points=max_points,
            )
            rotation_distance = _rotation_distance_degrees(rotation, l0_rotation)
            direction_distance = _vector_angle_degrees(direction, l0_translation)
            candidates.append(
                (cheirality, rotation_distance + direction_distance, rotation, direction)
            )
    if not candidates:
        return None
    candidates.sort(key=lambda value: (-value[0], value[1]))
    cheirality, _, rotation, direction = candidates[0]
    return rotation, direction, float(cheirality)


def _build_relative_candidate(
    rotation: torch.Tensor,
    translation: torch.Tensor,
    current: torch.Tensor,
    history: torch.Tensor,
    weights: torch.Tensor,
    *,
    initialization: str,
    config: EpipolarConfig,
) -> _RelativeCandidate:
    translation = translation.double()
    translation = translation / torch.linalg.vector_norm(translation).clamp_min(1e-12)
    rotation, translation, iterations = _refine_bearing_pose(
        rotation.double(),
        translation,
        current,
        history,
        weights,
        config=config,
    )
    essential = _skew(translation) @ rotation
    residual = _signed_sampson_residual(essential, current, history)
    normalized_weights = weights / weights.sum().clamp_min(1e-12)
    rmse = torch.sqrt((normalized_weights * residual.square()).sum())
    inlier = residual.abs().le(float(config.refinement_huber_delta))
    inlier_ratio = float((normalized_weights * inlier.double()).sum())
    return _RelativeCandidate(
        rotation=rotation,
        translation=translation,
        essential=essential,
        sampson_rmse=float(rmse),
        robust_objective=float(
            _huber_objective(
                residual,
                weights,
                config.refinement_huber_delta,
            )
        ),
        inlier_ratio=inlier_ratio,
        cheirality_fraction=_cheirality_fraction(
            rotation,
            translation,
            current,
            history,
            max_points=int(config.cheirality_max_points),
        ),
        refinement_iterations=iterations,
        initialization=initialization,
    )


def _candidate_l0_distance(
    candidate: _RelativeCandidate, l0_current_to_history: torch.Tensor
) -> float:
    return _rotation_distance_degrees(
        candidate.rotation, l0_current_to_history[:3, :3]
    ) + _vector_angle_degrees(
        candidate.translation, l0_current_to_history[:3, 3]
    )


def _cheirality_fraction(
    rotation: torch.Tensor,
    translation: torch.Tensor,
    current: torch.Tensor,
    history: torch.Tensor,
    *,
    max_points: int,
) -> float:
    count = min(int(current.shape[0]), int(max_points))
    indices = torch.linspace(0, current.shape[0] - 1, count).round().long()
    x1 = current[indices]
    x2 = history[indices]
    p1 = torch.cat([torch.eye(3, dtype=torch.float64), torch.zeros(3, 1, dtype=torch.float64)], dim=1)
    p2 = torch.cat([rotation, translation[:, None]], dim=1)
    rows = []
    for first, second in zip(x1, x2):
        rows.append(
            torch.stack(
                [
                    first[0] * p1[2] - p1[0],
                    first[1] * p1[2] - p1[1],
                    second[0] * p2[2] - p2[0],
                    second[1] * p2[2] - p2[1],
                ]
            )
        )
    design = torch.stack(rows)
    try:
        _, _, vh = torch.linalg.svd(design)
    except RuntimeError:
        return 0.0
    homogeneous_points = vh[:, -1]
    divisor = homogeneous_points[:, 3]
    finite = divisor.abs().gt(1e-12) & torch.isfinite(homogeneous_points).all(dim=-1)
    safe_divisor = torch.where(
        divisor.abs().gt(1e-12), divisor, torch.ones_like(divisor)
    )
    points = homogeneous_points[:, :3] / safe_divisor[:, None]
    depth_current = points[:, 2]
    depth_history = (torch.einsum("ij,nj->ni", rotation, points) + translation)[:, 2]
    positive = finite & depth_current.gt(0.0) & depth_history.gt(0.0)
    return float(positive.double().mean())


def _refine_bearing_pose(
    rotation: torch.Tensor,
    translation: torch.Tensor,
    current: torch.Tensor,
    history: torch.Tensor,
    weights: torch.Tensor,
    *,
    config: EpipolarConfig,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    iterations = 0
    epsilon = float(config.refinement_step_epsilon)
    for _ in range(int(config.refinement_iterations)):
        essential = _skew(translation) @ rotation
        residual = _signed_sampson_residual(essential, current, history)
        if not torch.isfinite(residual).all():
            break
        jacobian_columns = []
        for axis in range(6):
            delta = torch.zeros(6, dtype=torch.float64)
            delta[axis] = epsilon
            candidate_rotation, candidate_translation = _apply_pose_delta(
                rotation, translation, delta
            )
            candidate_residual = _signed_sampson_residual(
                _skew(candidate_translation) @ candidate_rotation,
                current,
                history,
            )
            jacobian_columns.append((candidate_residual - residual) / epsilon)
        jacobian = torch.stack(jacobian_columns, dim=-1)
        robust = torch.where(
            residual.abs().le(float(config.refinement_huber_delta)),
            torch.ones_like(residual),
            float(config.refinement_huber_delta) / residual.abs().clamp_min(1e-12),
        )
        combined = weights * robust
        hessian = torch.einsum("ni,n,nj->ij", jacobian, combined, jacobian)
        gradient = torch.einsum("ni,n,n->i", jacobian, combined, residual)
        hessian = hessian + float(config.refinement_damping) * torch.eye(6, dtype=torch.float64)
        try:
            step = torch.linalg.solve(hessian, -gradient)
        except RuntimeError:
            break
        norm = torch.linalg.vector_norm(step)
        if float(norm) > float(config.max_refinement_step):
            step = step * (float(config.max_refinement_step) / norm)
        candidate_rotation, candidate_translation = _apply_pose_delta(
            rotation, translation, step
        )
        candidate_residual = _signed_sampson_residual(
            _skew(candidate_translation) @ candidate_rotation,
            current,
            history,
        )
        old_objective = _huber_objective(residual, weights, config.refinement_huber_delta)
        new_objective = _huber_objective(
            candidate_residual, weights, config.refinement_huber_delta
        )
        if not torch.isfinite(new_objective) or new_objective > old_objective:
            break
        rotation, translation = candidate_rotation, candidate_translation
        iterations += 1
        if float(torch.linalg.vector_norm(step)) < 1e-7:
            break
    return rotation, translation, iterations


def _apply_pose_delta(
    rotation: torch.Tensor, translation: torch.Tensor, delta: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    updated_rotation = _rotation_exp(delta[:3]) @ rotation
    updated_translation = translation + delta[3:]
    updated_translation = updated_translation / torch.linalg.vector_norm(
        updated_translation
    ).clamp_min(1e-12)
    return updated_rotation, updated_translation


def _rotation_exp(vector: torch.Tensor) -> torch.Tensor:
    angle = torch.linalg.vector_norm(vector)
    identity = torch.eye(3, dtype=torch.float64)
    if float(angle) < 1e-10:
        return identity + _skew(vector)
    axis = vector / angle
    skew = _skew(axis)
    return identity + torch.sin(angle) * skew + (1.0 - torch.cos(angle)) * (skew @ skew)


def _signed_sampson_residual(
    essential: torch.Tensor,
    current: torch.Tensor,
    history: torch.Tensor,
) -> torch.Tensor:
    ex1 = torch.einsum("ij,nj->ni", essential, current)
    etx2 = torch.einsum("ji,nj->ni", essential, history)
    numerator = (history * ex1).sum(dim=-1)
    denominator = torch.sqrt(
        ex1[:, :2].square().sum(dim=-1)
        + etx2[:, :2].square().sum(dim=-1)
    ).clamp_min(1e-12)
    return numerator / denominator


def _huber_objective(
    residual: torch.Tensor, weights: torch.Tensor, delta: float
) -> torch.Tensor:
    absolute = residual.abs()
    value = torch.where(
        absolute.le(float(delta)),
        0.5 * residual.square(),
        float(delta) * (absolute - 0.5 * float(delta)),
    )
    return (weights * value).sum() / weights.sum().clamp_min(1e-12)


def _absolute_edge_weight(estimate: EpipolarEstimate) -> float:
    """Bounded-fusion quality from correspondence evidence, never GT error."""

    support = max(float(estimate.effective_correspondences), 1e-12)
    inlier = max(float(estimate.inlier_ratio), 1e-6)
    residual = max(float(estimate.sampson_rmse), 1e-6)
    value = support * inlier / residual
    return value if math.isfinite(value) and value > 0.0 else 1e-12


def _weighted_rotation_average(
    rotations: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    matrix = (weights[:, None, None] * rotations).sum(dim=0)
    u, _, vh = torch.linalg.svd(matrix)
    correction = torch.eye(3, dtype=torch.float64)
    correction[-1, -1] = torch.sign(torch.det(u @ vh))
    return u @ correction @ vh


def _effective_count(weights: torch.Tensor) -> float:
    total = weights.sum()
    return float(total.square() / weights.square().sum().clamp_min(1e-12))


def _rotation_distance_degrees(first: torch.Tensor, second: torch.Tensor) -> float:
    relative = first @ second.transpose(0, 1)
    cosine = ((torch.trace(relative) - 1.0) * 0.5).clamp(-1.0, 1.0)
    return float(torch.rad2deg(torch.acos(cosine)))


def _vector_angle_degrees(first: torch.Tensor, second: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second)
    if float(denominator) <= 1e-12:
        return 180.0
    cosine = torch.dot(first, second) / denominator
    return float(torch.rad2deg(torch.acos(cosine.clamp(-1.0, 1.0))))


def _skew(vector: torch.Tensor) -> torch.Tensor:
    x, y, z = vector
    return torch.stack(
        [
            torch.stack([x.new_zeros(()), -z, y]),
            torch.stack([z, x.new_zeros(()), -x]),
            torch.stack([-y, x, x.new_zeros(())]),
        ]
    )


def _bilinear_sample(image: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    height, width = image.shape
    uv = uv.to(dtype=image.dtype, device=image.device)
    x = 2.0 * uv[:, 0] / max(width - 1, 1) - 1.0
    y = 2.0 * uv[:, 1] / max(height - 1, 1) - 1.0
    grid = torch.stack([x, y], dim=-1).reshape(1, 1, -1, 2)
    valid = torch.isfinite(image) & image.gt(1e-8)
    safe_image = torch.where(valid, image, torch.zeros_like(image))
    numerator = F.grid_sample(
        safe_image.reshape(1, 1, height, width),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    denominator = F.grid_sample(
        valid.to(dtype=image.dtype).reshape(1, 1, height, width),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    sampled = numerator.reshape(-1) / denominator.reshape(-1).clamp_min(1e-12)
    return torch.where(
        denominator.reshape(-1).gt(1e-12),
        sampled,
        torch.full_like(sampled, torch.nan),
    )


def _nearest_sample_mask(mask: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    height, width = mask.shape
    x = uv[:, 0].round().long().clamp(0, width - 1)
    y = uv[:, 1].round().long().clamp(0, height - 1)
    return mask[y, x].bool()


def _evenly_subsample_rows(rows: torch.Tensor, count: int) -> torch.Tensor:
    if rows.shape[0] <= int(count):
        return rows
    indices = torch.linspace(0, rows.shape[0] - 1, int(count)).round().long()
    return rows.index_select(0, indices)


def _validate_surface_inputs(
    *,
    masks: torch.Tensor,
    world_points_metric: torch.Tensor,
    depth_metric: torch.Tensor,
    global_world_to_camera: torch.Tensor,
    intrinsics: torch.Tensor,
) -> None:
    if masks.ndim != 4:
        raise ValueError("V9 masks must be [S,K,H,W].")
    sequence, _, height, width = masks.shape
    if world_points_metric.shape != (sequence, height, width, 3):
        raise ValueError("V9 world points must be [S,H,W,3] at mask resolution.")
    if depth_metric.shape != (sequence, height, width):
        raise ValueError("V9 depth must be [S,H,W] at mask resolution.")
    if tuple(global_world_to_camera.shape) not in {
        (sequence, 3, 4),
        (sequence, 4, 4),
    }:
        raise ValueError("V9 global W2C must be [S,3,4] or [S,4,4].")
    if intrinsics.shape != (sequence, 3, 3):
        raise ValueError("V9 intrinsics must be [S,3,3].")


def _empty_surface(
    current: int,
    history: int,
    slot: int,
    dtype: torch.dtype,
    *,
    sampled_queries: int = 0,
    projected_in_bounds: int = 0,
) -> SurfaceCorrespondences:
    del dtype
    return SurfaceCorrespondences(
        current_frame=current,
        history_frame=history,
        slot=slot,
        current_uv=torch.empty(0, 2, dtype=torch.float64),
        history_uv=torch.empty(0, 2, dtype=torch.float64),
        weights=torch.empty(0, dtype=torch.float64),
        depth_residual_metric=torch.empty(0, dtype=torch.float64),
        sampled_queries=int(sampled_queries),
        projected_in_bounds=int(projected_in_bounds),
        visible_queries=0,
    )


def _failed_epipolar(count: int) -> EpipolarEstimate:
    return EpipolarEstimate(
        rotation_current_to_history=torch.eye(3, dtype=torch.float64),
        translation_current_origin_in_history=torch.zeros(3, dtype=torch.float64),
        essential=torch.zeros(3, 3, dtype=torch.float64),
        sampson_rmse=float("nan"),
        inlier_ratio=0.0,
        cheirality_fraction=0.0,
        design_rank_ratio=0.0,
        design_condition=float("inf"),
        correspondences=int(count),
        effective_correspondences=0.0,
        refinement_iterations=0,
        initialization="none",
        eight_point_sampson_rmse=float("nan"),
        l0_local_sampson_rmse=float("nan"),
        success=False,
        reason="not_solved",
    )
