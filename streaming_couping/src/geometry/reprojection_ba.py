"""Pose-only reprojection refinement using frozen StreamVGGT point tracks."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from .registration import rotation_angle_degrees


@dataclass(frozen=True)
class PoseBAResult:
    world_to_camera: torch.Tensor
    rotation: torch.Tensor
    translation: torch.Tensor
    eligible_tracks: int
    initial_reprojection_rmse: float
    final_reprojection_rmse: float
    iterations: int
    accepted: bool
    reason: str


def sample_reference_query_points(
    instance_mask: torch.Tensor,
    world_points: torch.Tensor,
    confidence: torch.Tensor,
    *,
    confidence_threshold: float,
    max_points: int,
    erosion_radius: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample deterministic interior xy queries and their integer yx indices."""

    valid = (
        instance_mask.bool()
        & torch.isfinite(world_points).all(dim=-1)
        & (confidence >= float(confidence_threshold))
    )
    if int(erosion_radius) > 0:
        kernel = 2 * int(erosion_radius) + 1
        eroded = F.avg_pool2d(
            valid.float()[None, None],
            kernel_size=kernel,
            stride=1,
            padding=int(erosion_radius),
        )[0, 0] >= 1.0 - 1e-6
        if int(eroded.sum()) >= min(32, int(max_points)):
            valid = eroded
    pixel_yx = torch.nonzero(valid, as_tuple=False)
    if pixel_yx.shape[0] > int(max_points):
        selected = torch.linspace(
            0,
            pixel_yx.shape[0] - 1,
            int(max_points),
        ).round().long()
        pixel_yx = pixel_yx[selected]
    query_xy = pixel_yx[:, [1, 0]].float()
    return query_xy, pixel_yx


def points_inside_mask(mask: torch.Tensor, coordinates_xy: torch.Tensor) -> torch.Tensor:
    """Return whether rounded xy observations lie inside a binary image mask."""

    height, width = mask.shape
    coordinates = coordinates_xy.round().long()
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    result = torch.zeros(coordinates.shape[0], dtype=torch.bool)
    result[inside] = mask[y[inside], x[inside]].bool()
    return result


def refine_pose_with_tracks(
    world_points: torch.Tensor,
    observed_xy: torch.Tensor,
    weights: torch.Tensor,
    base_world_to_camera: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    optimization_mask: torch.Tensor | None = None,
    mode: str = "full_se3",
    iterations: int = 200,
    learning_rate: float = 0.03,
    robust_delta_pixels: float = 4.0,
    pose_prior_weight: float = 0.10,
    min_tracks: int = 24,
    max_rotation_degrees: float = 15.0,
    max_translation_depth_ratio: float = 0.25,
) -> PoseBAResult:
    """Optimize a small camera-frame delta around a frozen StreamVGGT pose."""

    if mode not in {"translation_only", "full_se3"}:
        raise ValueError(f"Unknown BA mode {mode!r}.")
    base = _homogeneous_pose(base_world_to_camera.detach().float().cpu())
    points = world_points.detach().float().cpu()
    observations = observed_xy.detach().float().cpu()
    track_weights = weights.detach().float().cpu()
    valid = (
        torch.isfinite(points).all(dim=-1)
        & torch.isfinite(observations).all(dim=-1)
        & torch.isfinite(track_weights)
        & (track_weights > 0.0)
    )
    if optimization_mask is not None:
        valid &= optimization_mask.detach().bool().cpu()
    points = points[valid]
    observations = observations[valid]
    track_weights = track_weights[valid].clamp_min(1e-6)
    identity = torch.eye(3, dtype=torch.float32)
    zero = torch.zeros(3, dtype=torch.float32)
    if points.shape[0] < int(min_tracks):
        return PoseBAResult(
            base,
            identity,
            zero,
            int(points.shape[0]),
            float("nan"),
            float("nan"),
            0,
            False,
            "too few valid tracks",
        )

    base_camera_points = _transform_world_points(points, base)
    positive_depth = base_camera_points[:, 2] > 1e-4
    points = points[positive_depth]
    observations = observations[positive_depth]
    track_weights = track_weights[positive_depth]
    base_camera_points = base_camera_points[positive_depth]
    if points.shape[0] < int(min_tracks):
        return PoseBAResult(
            base,
            identity,
            zero,
            int(points.shape[0]),
            float("nan"),
            float("nan"),
            0,
            False,
            "too few positive-depth tracks",
        )

    intrinsic = intrinsics.detach().float().cpu()
    initial_xy = _project_camera_points(base_camera_points, intrinsic)
    initial_rmse = _weighted_rmse(initial_xy - observations, track_weights)
    median_depth = base_camera_points[:, 2].median().abs().clamp_min(1e-3)
    image_diagonal = torch.sqrt(
        (2.0 * intrinsic[0, 2]) ** 2 + (2.0 * intrinsic[1, 2]) ** 2
    ).clamp_min(1.0)

    rotation_vector = torch.nn.Parameter(torch.zeros(3))
    translation = torch.nn.Parameter(torch.zeros(3))
    parameters = [translation]
    if mode == "full_se3":
        parameters.insert(0, rotation_vector)
    optimizer = torch.optim.Adam(parameters, lr=float(learning_rate))
    completed_iterations = 0
    for iteration in range(max(1, int(iterations))):
        completed_iterations = iteration + 1
        optimizer.zero_grad(set_to_none=True)
        delta_rotation = (
            torch.matrix_exp(_skew(rotation_vector))
            if mode == "full_se3"
            else identity
        )
        corrected_camera_points = (
            base_camera_points @ delta_rotation.T + translation
        )
        predicted_xy = _project_camera_points(
            corrected_camera_points,
            intrinsic,
            clamp_depth=True,
        )
        residual_pixels = torch.linalg.vector_norm(
            predicted_xy - observations,
            dim=-1,
        )
        scaled_residual = residual_pixels / image_diagonal
        scaled_delta = float(robust_delta_pixels) / float(image_diagonal)
        reprojection = _weighted_huber(
            scaled_residual,
            track_weights,
            delta=max(scaled_delta, 1e-6),
        )
        rotation_prior = (rotation_vector**2).mean()
        translation_prior = ((translation / median_depth) ** 2).mean()
        invalid_depth = F.relu(
            (1e-3 * median_depth - corrected_camera_points[:, 2]) / median_depth
        ).mean()
        loss = (
            reprojection
            + float(pose_prior_weight) * (rotation_prior + translation_prior)
            + invalid_depth
        )
        if not torch.isfinite(loss):
            break
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        delta_rotation = (
            torch.matrix_exp(_skew(rotation_vector))
            if mode == "full_se3"
            else identity
        )
        corrected_camera_points = (
            base_camera_points @ delta_rotation.T + translation
        )
        final_xy = _project_camera_points(
            corrected_camera_points,
            intrinsic,
            clamp_depth=True,
        )
        final_rmse = _weighted_rmse(final_xy - observations, track_weights)
        correction = torch.eye(4, dtype=torch.float32)
        correction[:3, :3] = delta_rotation
        correction[:3, 3] = translation
        corrected_pose = correction @ base
        rotation_degrees = rotation_angle_degrees(delta_rotation)
        translation_ratio = float(
            torch.linalg.vector_norm(translation) / median_depth
        )

    checks = (
        (math.isfinite(final_rmse), "non-finite optimized reprojection"),
        (final_rmse < initial_rmse, "reprojection did not improve"),
        (
            rotation_degrees <= float(max_rotation_degrees),
            "rotation delta exceeds safeguard",
        ),
        (
            translation_ratio <= float(max_translation_depth_ratio),
            "translation delta exceeds depth-relative safeguard",
        ),
    )
    reason = next((message for passed, message in checks if not passed), "accepted")
    accepted = reason == "accepted"
    return PoseBAResult(
        world_to_camera=corrected_pose.detach() if accepted else base,
        rotation=delta_rotation.detach(),
        translation=translation.detach(),
        eligible_tracks=int(points.shape[0]),
        initial_reprojection_rmse=float(initial_rmse),
        final_reprojection_rmse=float(final_rmse),
        iterations=completed_iterations,
        accepted=accepted,
        reason=reason,
    )


def identity_ba_result(
    world_to_camera: torch.Tensor,
    *,
    reason: str,
) -> PoseBAResult:
    pose = _homogeneous_pose(world_to_camera.detach().float().cpu())
    return PoseBAResult(
        world_to_camera=pose,
        rotation=torch.eye(3),
        translation=torch.zeros(3),
        eligible_tracks=0,
        initial_reprojection_rmse=float("nan"),
        final_reprojection_rmse=float("nan"),
        iterations=0,
        accepted=False,
        reason=reason,
    )


def _homogeneous_pose(pose: torch.Tensor) -> torch.Tensor:
    if pose.shape == (4, 4):
        return pose.clone()
    if pose.shape == (3, 4):
        output = torch.eye(4, dtype=pose.dtype)
        output[:3] = pose
        return output
    raise ValueError(f"Expected camera pose [3,4] or [4,4], got {tuple(pose.shape)}")


def _transform_world_points(points: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
    return points @ pose[:3, :3].T + pose[:3, 3]


def _project_camera_points(
    camera_points: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    clamp_depth: bool = False,
) -> torch.Tensor:
    depth = camera_points[:, 2]
    if clamp_depth:
        depth = depth.clamp_min(1e-6)
    projected = camera_points @ intrinsics.T
    return projected[:, :2] / depth[:, None]


def _skew(vector: torch.Tensor) -> torch.Tensor:
    x, y, z = vector.unbind()
    zero = torch.zeros((), dtype=vector.dtype, device=vector.device)
    return torch.stack(
        (
            torch.stack((zero, -z, y)),
            torch.stack((z, zero, -x)),
            torch.stack((-y, x, zero)),
        )
    )


def _weighted_huber(
    residual: torch.Tensor,
    weights: torch.Tensor,
    *,
    delta: float,
) -> torch.Tensor:
    delta_tensor = residual.new_tensor(float(delta))
    values = torch.where(
        residual <= delta_tensor,
        0.5 * residual.square() / delta_tensor,
        residual - 0.5 * delta_tensor,
    )
    return (values * weights).sum() / weights.sum().clamp_min(1e-6)


def _weighted_rmse(residual_xy: torch.Tensor, weights: torch.Tensor) -> float:
    squared = residual_xy.square().sum(dim=-1)
    value = (squared * weights).sum() / weights.sum().clamp_min(1e-6)
    return float(torch.sqrt(value))
