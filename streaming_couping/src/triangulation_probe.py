"""Training-free 2D matching and metric two-view triangulation primitives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from .pointmap_diagnosis import as_homogeneous_world_to_camera


@dataclass(frozen=True)
class PairMatches:
    current_xy: torch.Tensor
    history_xy: torch.Tensor
    query_indices: torch.Tensor
    confidence: torch.Tensor


@dataclass(frozen=True)
class TriangulatedPair:
    points: torch.Tensor
    condition: torch.Tensor
    ray_angle_degrees: torch.Tensor
    reprojection_mean_px: torch.Tensor
    reprojection_max_px: torch.Tensor
    positive_depth: torch.Tensor


def sample_mask_grid(
    mask: torch.Tensor,
    *,
    stride: int,
    max_points: int,
    margin: int,
) -> torch.Tensor:
    """Return deterministic integer ``(x,y)`` samples inside one mask."""

    mask = mask.detach().bool().cpu()
    if mask.ndim != 2 or stride < 1 or max_points < 1:
        raise ValueError("Invalid mask sampling arguments.")
    height, width = mask.shape
    y = torch.arange(margin, max(margin, height - margin), stride)
    x = torch.arange(margin, max(margin, width - margin), stride)
    if y.numel() == 0 or x.numel() == 0:
        return torch.empty(0, 2, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    xy = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)
    keep = mask[xy[:, 1], xy[:, 0]]
    xy = xy[keep]
    if xy.shape[0] > int(max_points):
        index = torch.linspace(0, xy.shape[0] - 1, int(max_points)).long()
        xy = xy.index_select(0, index)
    return xy.float()


def patch_descriptors(
    image: torch.Tensor,
    xy: torch.Tensor,
    *,
    radius: int,
) -> torch.Tensor:
    """Extract normalized frozen RGB patches at integer pixel locations."""

    image = image.detach().float().cpu()
    xy = xy.detach().float().cpu()
    if image.ndim != 3 or image.shape[0] != 3 or xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("image/xy must be [3,H,W] and [N,2].")
    if xy.shape[0] == 0:
        return torch.empty(0, 3 * (2 * radius + 1) ** 2)
    integer = xy.round().long()
    offsets = torch.arange(-int(radius), int(radius) + 1)
    dy, dx = torch.meshgrid(offsets, offsets, indexing="ij")
    sample_y = (integer[:, 1, None] + dy.reshape(1, -1)).clamp(
        0, image.shape[1] - 1
    )
    sample_x = (integer[:, 0, None] + dx.reshape(1, -1)).clamp(
        0, image.shape[2] - 1
    )
    descriptor = image[:, sample_y, sample_x].permute(1, 0, 2).reshape(
        integer.shape[0], -1
    )
    descriptor = descriptor - descriptor.mean(dim=1, keepdim=True)
    return functional.normalize(descriptor, dim=1, eps=1e-8)


def match_patch_descriptors(
    current_image: torch.Tensor,
    history_image: torch.Tensor,
    current_xy: torch.Tensor,
    history_mask: torch.Tensor,
    *,
    candidate_stride: int,
    max_history_points: int,
    patch_radius: int,
    ratio_threshold: float,
) -> PairMatches:
    """Match fixed current queries into one history ownership mask."""

    history_xy = sample_mask_grid(
        history_mask,
        stride=int(candidate_stride),
        max_points=int(max_history_points),
        margin=int(patch_radius),
    )
    empty = PairMatches(
        current_xy=torch.empty(0, 2),
        history_xy=torch.empty(0, 2),
        query_indices=torch.empty(0, dtype=torch.long),
        confidence=torch.empty(0),
    )
    if current_xy.shape[0] == 0 or history_xy.shape[0] < 2:
        return empty
    query = patch_descriptors(current_image, current_xy, radius=int(patch_radius))
    candidate = patch_descriptors(history_image, history_xy, radius=int(patch_radius))
    distance = torch.cdist(query, candidate)
    best_distance, best_index = torch.topk(
        distance, k=2, dim=1, largest=False, sorted=True
    )
    ratio = torch.where(
        best_distance[:, 1] > 1e-6,
        best_distance[:, 0] / best_distance[:, 1].clamp_min(1e-8),
        torch.ones_like(best_distance[:, 0]),
    )
    # Mutual nearest suppresses repeated texture without using any GT field.
    candidate_to_query = distance.argmin(dim=0)
    query_index = torch.arange(query.shape[0])
    mutual = candidate_to_query.index_select(0, best_index[:, 0]) == query_index
    keep = (ratio <= float(ratio_threshold)) & mutual
    selected = torch.nonzero(keep, as_tuple=False)[:, 0]
    if selected.numel() == 0:
        return empty
    return PairMatches(
        current_xy=current_xy.index_select(0, selected),
        history_xy=history_xy.index_select(
            0, best_index[:, 0].index_select(0, selected)
        ),
        query_indices=selected,
        confidence=(1.0 - ratio.index_select(0, selected)).clamp(0.0, 1.0),
    )


def equalize_query_support(
    correct: PairMatches, control: PairMatches
) -> tuple[PairMatches, PairMatches]:
    """Keep the exact same current query indices in two identity branches."""

    common = sorted(
        set(int(value) for value in correct.query_indices.tolist())
        & set(int(value) for value in control.query_indices.tolist())
    )
    if not common:
        empty = PairMatches(
            current_xy=torch.empty(0, 2),
            history_xy=torch.empty(0, 2),
            query_indices=torch.empty(0, dtype=torch.long),
            confidence=torch.empty(0),
        )
        return empty, empty
    return _select_query_ids(correct, common), _select_query_ids(control, common)


def triangulate_two_view(
    current_xy: torch.Tensor,
    history_xy: torch.Tensor,
    current_intrinsics: torch.Tensor,
    history_intrinsics: torch.Tensor,
    current_world_to_camera: torch.Tensor,
    history_world_to_camera: torch.Tensor,
) -> TriangulatedPair:
    """Linear DLT triangulation plus non-GT conditioning diagnostics."""

    current_xy = current_xy.detach().float().cpu()
    history_xy = history_xy.detach().float().cpu()
    if current_xy.shape != history_xy.shape or current_xy.ndim != 2:
        raise ValueError("Two-view coordinates must have matching [N,2] shapes.")
    count = current_xy.shape[0]
    if count == 0:
        empty = torch.empty(0)
        return TriangulatedPair(
            points=torch.empty(0, 3),
            condition=empty,
            ray_angle_degrees=empty,
            reprojection_mean_px=empty,
            reprojection_max_px=empty,
            positive_depth=torch.empty(0, dtype=torch.bool),
        )
    current_pose = as_homogeneous_world_to_camera(current_world_to_camera[None])[0]
    history_pose = as_homogeneous_world_to_camera(history_world_to_camera[None])[0]
    projection_current = current_intrinsics.float() @ current_pose[:3]
    projection_history = history_intrinsics.float() @ history_pose[:3]
    matrices = torch.stack(
        (
            current_xy[:, 0, None] * projection_current[2] - projection_current[0],
            current_xy[:, 1, None] * projection_current[2] - projection_current[1],
            history_xy[:, 0, None] * projection_history[2] - projection_history[0],
            history_xy[:, 1, None] * projection_history[2] - projection_history[1],
        ),
        dim=1,
    )
    _, singular, right_t = torch.linalg.svd(matrices)
    homogeneous = right_t[:, -1]
    denominator = homogeneous[:, 3:]
    denominator = torch.where(
        denominator.abs() >= 1e-12,
        denominator,
        torch.where(
            denominator >= 0.0,
            torch.full_like(denominator, 1e-12),
            torch.full_like(denominator, -1e-12),
        ),
    )
    points = homogeneous[:, :3] / denominator
    # The final singular value is the desired homogeneous null direction;
    # conditioning therefore uses sigma_max / sigma_(n-1).
    condition = singular[:, 0] / singular[:, -2].clamp_min(1e-12)
    projected_current, z_current = project_world_points(
        points, current_intrinsics, current_pose
    )
    projected_history, z_history = project_world_points(
        points, history_intrinsics, history_pose
    )
    residual_current = torch.linalg.vector_norm(projected_current - current_xy, dim=-1)
    residual_history = torch.linalg.vector_norm(projected_history - history_xy, dim=-1)
    residual = torch.stack((residual_current, residual_history), dim=1)
    angle = ray_angle_degrees(
        current_xy,
        history_xy,
        current_intrinsics,
        history_intrinsics,
        current_pose,
        history_pose,
    )
    return TriangulatedPair(
        points=points,
        condition=condition,
        ray_angle_degrees=angle,
        reprojection_mean_px=residual.mean(dim=1),
        reprojection_max_px=residual.max(dim=1).values,
        positive_depth=(z_current > 0.0) & (z_history > 0.0),
    )


def project_world_points(
    points: torch.Tensor,
    intrinsics: torch.Tensor,
    world_to_camera: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    pose = as_homogeneous_world_to_camera(world_to_camera[None])[0]
    camera = points.float() @ pose[:3, :3].T + pose[:3, 3]
    pixel_h = camera @ intrinsics.float().T
    pixel = pixel_h[:, :2] / pixel_h[:, 2:].clamp_min(1e-12)
    return pixel, camera[:, 2]


def ray_angle_degrees(
    current_xy: torch.Tensor,
    history_xy: torch.Tensor,
    current_intrinsics: torch.Tensor,
    history_intrinsics: torch.Tensor,
    current_world_to_camera: torch.Tensor,
    history_world_to_camera: torch.Tensor,
) -> torch.Tensor:
    current_ray = _world_rays(
        current_xy, current_intrinsics, current_world_to_camera
    )
    history_ray = _world_rays(
        history_xy, history_intrinsics, history_world_to_camera
    )
    cosine = (current_ray * history_ray).sum(dim=-1).abs().clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cosine))


def triangulation_gate(
    result: TriangulatedPair,
    *,
    min_ray_angle_degrees: float,
    max_ray_angle_degrees: float,
    max_condition: float,
    max_reprojection_px: float,
) -> torch.Tensor:
    return (
        torch.isfinite(result.points).all(dim=-1)
        & torch.isfinite(result.condition)
        & torch.isfinite(result.ray_angle_degrees)
        & torch.isfinite(result.reprojection_max_px)
        & result.positive_depth
        & (result.ray_angle_degrees >= float(min_ray_angle_degrees))
        & (result.ray_angle_degrees <= float(max_ray_angle_degrees))
        & (result.condition <= float(max_condition))
        & (result.reprojection_max_px <= float(max_reprojection_px))
    )


def project_oracle_correspondence(
    current_xy: torch.Tensor,
    current_target_world: torch.Tensor,
    history_target_world: torch.Tensor,
    history_mask: torch.Tensor,
    history_intrinsics: torch.Tensor,
    history_world_to_camera: torch.Tensor,
    *,
    depth_tolerance: float,
) -> PairMatches:
    """Generate evaluation-only visible-surface correspondences from GT."""

    if current_xy.shape[0] == 0:
        return PairMatches(
            current_xy=torch.empty(0, 2),
            history_xy=torch.empty(0, 2),
            query_indices=torch.empty(0, dtype=torch.long),
            confidence=torch.empty(0),
        )
    query_index = torch.arange(current_xy.shape[0])
    integer = current_xy.round().long()
    world = current_target_world[integer[:, 1], integer[:, 0]]
    projected, projected_z = project_world_points(
        world, history_intrinsics, history_world_to_camera
    )
    rounded = projected.round().long()
    height, width = history_mask.shape
    in_bounds = (
        torch.isfinite(projected).all(dim=-1)
        & (rounded[:, 0] >= 0)
        & (rounded[:, 0] < width)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < height)
        & (projected_z > 0.0)
    )
    safe_x = rounded[:, 0].clamp(0, width - 1)
    safe_y = rounded[:, 1].clamp(0, height - 1)
    history_world = history_target_world[safe_y, safe_x]
    history_camera_z = (
        history_world @ history_world_to_camera[:3, :3].T
        + history_world_to_camera[:3, 3]
    )[:, 2]
    visible = (
        in_bounds
        & history_mask[safe_y, safe_x]
        & torch.isfinite(world).all(dim=-1)
        & torch.isfinite(history_world).all(dim=-1)
        & ((history_camera_z - projected_z).abs() <= float(depth_tolerance))
    )
    selected = torch.nonzero(visible, as_tuple=False)[:, 0]
    return PairMatches(
        current_xy=current_xy.index_select(0, selected),
        history_xy=projected.index_select(0, selected),
        query_indices=query_index.index_select(0, selected),
        confidence=torch.ones(selected.numel()),
    )


def _world_rays(
    xy: torch.Tensor, intrinsics: torch.Tensor, world_to_camera: torch.Tensor
) -> torch.Tensor:
    homogeneous = torch.cat((xy.float(), torch.ones(xy.shape[0], 1)), dim=1)
    camera = homogeneous @ torch.linalg.inv(intrinsics.float()).T
    pose = as_homogeneous_world_to_camera(world_to_camera[None])[0]
    world = camera @ pose[:3, :3]
    return functional.normalize(world, dim=-1, eps=1e-8)


def _select_query_ids(matches: PairMatches, query_ids: list[int]) -> PairMatches:
    lookup = {int(value): index for index, value in enumerate(matches.query_indices.tolist())}
    index = torch.tensor([lookup[value] for value in query_ids], dtype=torch.long)
    return PairMatches(
        current_xy=matches.current_xy.index_select(0, index),
        history_xy=matches.history_xy.index_select(0, index),
        query_indices=matches.query_indices.index_select(0, index),
        confidence=matches.confidence.index_select(0, index),
    )
