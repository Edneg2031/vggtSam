"""Robust similarity alignment and object-centric rigid registration."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SimilarityTransform:
    scale: float
    rotation: torch.Tensor
    translation: torch.Tensor
    inliers: int
    rmse: float


@dataclass(frozen=True)
class ICPResult:
    rotation: torch.Tensor
    translation: torch.Tensor
    inliers: int
    fitness: float
    rmse: float
    iterations: int
    accepted: bool
    reason: str


def estimate_similarity(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    trim_fraction: float = 0.7,
    robust_iterations: int = 4,
    min_points: int = 128,
) -> SimilarityTransform:
    """Estimate a robust Sim(3) from paired source and target points."""

    source, target = _paired_finite(source, target)
    if source.shape[0] < min_points:
        raise ValueError(
            f"Similarity alignment needs {min_points} points, got {source.shape[0]}."
        )
    keep = torch.ones(source.shape[0], dtype=torch.bool, device=source.device)
    scale = 1.0
    rotation = torch.eye(3, dtype=source.dtype, device=source.device)
    translation = torch.zeros(3, dtype=source.dtype, device=source.device)
    for _ in range(max(1, int(robust_iterations))):
        scale, rotation, translation = _umeyama(source[keep], target[keep])
        aligned = apply_similarity(source, scale, rotation, translation)
        residual = torch.linalg.vector_norm(aligned - target, dim=-1)
        finite = torch.isfinite(residual)
        finite_residual = residual[finite]
        if finite_residual.numel() < min_points:
            break
        threshold = torch.quantile(
            finite_residual,
            min(1.0, max(0.1, float(trim_fraction))),
        )
        next_keep = finite & (residual <= threshold)
        if next_keep.sum() < min_points or torch.equal(next_keep, keep):
            break
        keep = next_keep
    aligned = apply_similarity(source, scale, rotation, translation)
    residual = torch.linalg.vector_norm(aligned - target, dim=-1)
    rmse = float(torch.sqrt((residual[keep] ** 2).mean()).cpu())
    return SimilarityTransform(
        scale=float(scale),
        rotation=rotation.detach(),
        translation=translation.detach(),
        inliers=int(keep.sum()),
        rmse=rmse,
    )


def robust_icp(
    moving: torch.Tensor,
    fixed: torch.Tensor,
    *,
    moving_weights: torch.Tensor | None = None,
    max_points: int = 2048,
    iterations: int = 30,
    trim_fraction: float = 0.7,
    max_correspondence_distance: float = 0.20,
    min_inliers: int = 64,
    min_fitness: float = 0.10,
    max_rmse: float = 0.15,
    max_rotation_degrees: float = 45.0,
    max_translation: float = 1.0,
) -> ICPResult:
    """Align one current instance cloud to its persistent reference cloud."""

    moving, moving_weights = _finite_points(moving, moving_weights)
    fixed, _ = _finite_points(fixed, None)
    moving, moving_weights = _deterministic_subsample(
        moving, max_points=max_points, weights=moving_weights
    )
    fixed, _ = _deterministic_subsample(fixed, max_points=max_points, weights=None)
    identity = torch.eye(3, dtype=moving.dtype, device=moving.device)
    zero = torch.zeros(3, dtype=moving.dtype, device=moving.device)
    if moving.shape[0] < min_inliers or fixed.shape[0] < min_inliers:
        return ICPResult(
            identity,
            zero,
            0,
            0.0,
            float("inf"),
            0,
            False,
            "too few object points",
        )

    rotation = identity
    translation = zero
    completed_iterations = 0
    for iteration in range(max(1, int(iterations))):
        completed_iterations = iteration + 1
        transformed = apply_rigid(moving, rotation, translation)
        distances, indices = nearest_neighbors(transformed, fixed)
        valid = torch.isfinite(distances) & (
            distances <= float(max_correspondence_distance)
        )
        if int(valid.sum()) < min_inliers:
            break
        valid_distances = distances[valid]
        trim_threshold = torch.quantile(
            valid_distances,
            min(1.0, max(0.1, float(trim_fraction))),
        )
        inliers = valid & (distances <= trim_threshold)
        if int(inliers.sum()) < min_inliers:
            break
        weights = moving_weights[inliers] if moving_weights is not None else None
        delta_rotation, delta_translation = _weighted_rigid(
            transformed[inliers],
            fixed[indices[inliers]],
            weights=weights,
        )
        rotation = delta_rotation @ rotation
        translation = delta_rotation @ translation + delta_translation
        delta_angle = rotation_angle_degrees(delta_rotation)
        translation_delta = float(torch.linalg.vector_norm(delta_translation))
        if delta_angle < 1e-3 and translation_delta < 1e-5:
            break

    transformed = apply_rigid(moving, rotation, translation)
    distances, _ = nearest_neighbors(transformed, fixed)
    valid = torch.isfinite(distances) & (
        distances <= float(max_correspondence_distance)
    )
    inlier_count = int(valid.sum())
    fitness = inlier_count / max(1, moving.shape[0])
    rmse = (
        float(torch.sqrt((distances[valid] ** 2).mean()).cpu())
        if inlier_count
        else float("inf")
    )
    rotation_degrees = rotation_angle_degrees(rotation)
    translation_norm = float(torch.linalg.vector_norm(translation).cpu())
    checks = (
        (inlier_count >= min_inliers, "too few ICP inliers"),
        (fitness >= min_fitness, "ICP fitness below threshold"),
        (rmse <= max_rmse, "ICP RMSE above threshold"),
        (rotation_degrees <= max_rotation_degrees, "rotation correction too large"),
        (translation_norm <= max_translation, "translation correction too large"),
    )
    reason = next((message for passed, message in checks if not passed), "accepted")
    return ICPResult(
        rotation=rotation.detach(),
        translation=translation.detach(),
        inliers=inlier_count,
        fitness=float(fitness),
        rmse=float(rmse),
        iterations=completed_iterations,
        accepted=reason == "accepted",
        reason=reason,
    )


def apply_similarity(
    points: torch.Tensor,
    scale: float,
    rotation: torch.Tensor,
    translation: torch.Tensor,
) -> torch.Tensor:
    return float(scale) * (points @ rotation.T) + translation


def apply_rigid(
    points: torch.Tensor,
    rotation: torch.Tensor,
    translation: torch.Tensor,
) -> torch.Tensor:
    return points @ rotation.T + translation


def align_world_to_camera(
    world_to_camera: torch.Tensor,
    similarity: SimilarityTransform,
) -> torch.Tensor:
    """Express a predicted camera pose in the similarity-aligned world frame."""

    camera_to_world = invert_pose(world_to_camera)
    aligned = torch.eye(4, dtype=world_to_camera.dtype, device=world_to_camera.device)
    aligned[:3, :3] = similarity.rotation @ camera_to_world[:3, :3]
    aligned[:3, 3] = apply_similarity(
        camera_to_world[:3, 3][None],
        similarity.scale,
        similarity.rotation,
        similarity.translation,
    )[0]
    return invert_pose(aligned)


def apply_world_correction_to_pose(
    world_to_camera: torch.Tensor,
    rotation: torch.Tensor,
    translation: torch.Tensor,
) -> torch.Tensor:
    """Move a camera and its pointmap by the same world-space rigid correction."""

    camera_to_world = invert_pose(world_to_camera)
    corrected = torch.eye(4, dtype=world_to_camera.dtype, device=world_to_camera.device)
    corrected[:3, :3] = rotation @ camera_to_world[:3, :3]
    corrected[:3, 3] = apply_rigid(
        camera_to_world[:3, 3][None], rotation, translation
    )[0]
    return invert_pose(corrected)


def invert_pose(pose: torch.Tensor) -> torch.Tensor:
    if pose.shape == (3, 4):
        homogeneous = torch.eye(4, dtype=pose.dtype, device=pose.device)
        homogeneous[:3] = pose
        pose = homogeneous
    if pose.shape != (4, 4):
        raise ValueError(f"Expected [3,4] or [4,4] pose, got {tuple(pose.shape)}")
    output = torch.eye(4, dtype=pose.dtype, device=pose.device)
    output[:3, :3] = pose[:3, :3].T
    output[:3, 3] = -(pose[:3, :3].T @ pose[:3, 3])
    return output


def pose_errors(
    predicted_world_to_camera: torch.Tensor,
    target_world_to_camera: torch.Tensor,
) -> tuple[float, float]:
    predicted = invert_pose(predicted_world_to_camera)
    target = invert_pose(target_world_to_camera)
    relative_rotation = predicted[:3, :3].T @ target[:3, :3]
    rotation_error = rotation_angle_degrees(relative_rotation)
    translation_error = float(
        torch.linalg.vector_norm(predicted[:3, 3] - target[:3, 3]).cpu()
    )
    return rotation_error, translation_error


def rotation_angle_degrees(rotation: torch.Tensor) -> float:
    cosine = ((torch.trace(rotation) - 1.0) * 0.5).clamp(-1.0, 1.0)
    return float(torch.rad2deg(torch.acos(cosine)).cpu())


def nearest_neighbors(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    chunk_size: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    distances = []
    indices = []
    for start in range(0, source.shape[0], chunk_size):
        pairwise = torch.cdist(source[start : start + chunk_size], target)
        values, nearest = pairwise.min(dim=1)
        distances.append(values)
        indices.append(nearest)
    return torch.cat(distances), torch.cat(indices)


def symmetric_chamfer(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    max_points: int = 4096,
) -> float:
    first, _ = _finite_points(first, None)
    second, _ = _finite_points(second, None)
    first, _ = _deterministic_subsample(first, max_points=max_points, weights=None)
    second, _ = _deterministic_subsample(second, max_points=max_points, weights=None)
    if first.numel() == 0 or second.numel() == 0:
        return float("nan")
    first_distance, _ = nearest_neighbors(first, second)
    second_distance, _ = nearest_neighbors(second, first)
    return float((first_distance.mean() + second_distance.mean()).mul(0.5).cpu())


def _umeyama(
    source: torch.Tensor,
    target: torch.Tensor,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    source_mean = source.mean(dim=0)
    target_mean = target.mean(dim=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / source.shape[0]
    left, singular_values, right_t = torch.linalg.svd(covariance)
    signs = torch.ones(3, dtype=source.dtype, device=source.device)
    if torch.det(left @ right_t) < 0:
        signs[-1] = -1
    rotation = left @ torch.diag(signs) @ right_t
    variance = (source_centered.square().sum(dim=1)).mean().clamp_min(1e-12)
    scale = float((singular_values * signs).sum() / variance)
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def _weighted_rigid(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    weights: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if weights is None:
        weights = torch.ones(source.shape[0], dtype=source.dtype, device=source.device)
    weights = weights.clamp_min(1e-6)
    weights = weights / weights.sum()
    source_mean = (source * weights[:, None]).sum(dim=0)
    target_mean = (target * weights[:, None]).sum(dim=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = (target_centered * weights[:, None]).T @ source_centered
    left, _, right_t = torch.linalg.svd(covariance)
    signs = torch.ones(3, dtype=source.dtype, device=source.device)
    if torch.det(left @ right_t) < 0:
        signs[-1] = -1
    rotation = left @ torch.diag(signs) @ right_t
    translation = target_mean - rotation @ source_mean
    return rotation, translation


def _paired_finite(
    source: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = torch.isfinite(source).all(dim=-1) & torch.isfinite(target).all(dim=-1)
    return source[valid].float(), target[valid].float()


def _finite_points(
    points: torch.Tensor,
    weights: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    valid = torch.isfinite(points).all(dim=-1)
    if weights is not None:
        valid &= torch.isfinite(weights)
        weights = weights[valid].float()
    return points[valid].float(), weights


def _deterministic_subsample(
    points: torch.Tensor,
    *,
    max_points: int,
    weights: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if points.shape[0] <= int(max_points):
        return points, weights
    indices = torch.linspace(
        0,
        points.shape[0] - 1,
        steps=int(max_points),
        device=points.device,
    ).long()
    return points[indices], weights[indices] if weights is not None else None
