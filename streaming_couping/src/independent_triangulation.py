"""Frozen 2D-feature matching and camera-ray triangulation primitives."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class MutualMatches:
    query_positions: torch.Tensor
    search_positions: torch.Tensor
    similarities: torch.Tensor
    margins: torch.Tensor


@dataclass(frozen=True)
class RayTriangulation:
    point_world: torch.Tensor
    condition_number: float
    minimum_ray_angle_degrees: float
    maximum_ray_angle_degrees: float
    mean_reprojection_error_px: float
    maximum_reprojection_error_px: float
    positive_depth_rate: float
    finite: bool


def extract_frame_patch_descriptors(
    token_levels: torch.Tensor,
    *,
    level_position: int,
    patch_start_idx: int,
    patch_shape: tuple[int, int],
    token_half: str = "frame",
) -> torch.Tensor:
    """Return normalized cached patch descriptors as ``[S,P,C]``.

    The V0 cache stores ``frame_tokens || global_tokens`` at every DPT level.
    T0 uses one half only and never reads point/depth-head output to form a
    correspondence.
    """

    value = token_levels.detach().float().cpu()
    if value.ndim != 5 or value.shape[1] != 1:
        raise ValueError(
            "token_levels must have shape [L,1,S,T,C], got "
            f"{tuple(value.shape)}."
        )
    level = int(level_position)
    if level < 0:
        level += int(value.shape[0])
    if not 0 <= level < int(value.shape[0]):
        raise ValueError("Descriptor level_position is outside token_levels.")
    tokens = value[level, 0]
    grid_height, grid_width = map(int, patch_shape)
    patch_count = grid_height * grid_width
    start = int(patch_start_idx)
    if tokens.shape[1] - start != patch_count:
        raise ValueError(
            "Cached patch tokens disagree with patch_shape: "
            f"tokens={tokens.shape[1] - start}, grid={patch_count}."
        )
    channels = int(tokens.shape[-1])
    if channels % 2:
        raise ValueError("Cached frame/global token channels must be even.")
    half = channels // 2
    mode = str(token_half).strip().lower()
    if mode == "frame":
        descriptors = tokens[:, start:, :half]
    elif mode == "global":
        descriptors = tokens[:, start:, half:]
    else:
        raise ValueError("token_half must be 'frame' or 'global'.")
    if not torch.isfinite(descriptors).all():
        raise ValueError("Cached patch descriptors contain non-finite values.")
    return F.normalize(descriptors, dim=-1, eps=1e-12)


def masks_to_patch_support(
    masks: torch.Tensor,
    scores: torch.Tensor,
    *,
    patch_shape: tuple[int, int],
    score_threshold: float,
    minimum_patch_coverage: float,
    erosion_pixels: int,
) -> torch.Tensor:
    """Convert dense SAM masks to conservative boolean patch support."""

    value = masks.detach().bool().cpu()
    confidence = scores.detach().float().cpu()
    if value.ndim != 4 or confidence.shape != value.shape[:2]:
        raise ValueError("masks/scores must have shapes [S,I,H,W] and [S,I].")
    if not 0.0 <= float(score_threshold) <= 1.0:
        raise ValueError("score_threshold must be in [0,1].")
    if not 0.0 < float(minimum_patch_coverage) <= 1.0:
        raise ValueError("minimum_patch_coverage must be in (0,1].")
    radius = int(erosion_pixels)
    if radius < 0:
        raise ValueError("erosion_pixels cannot be negative.")
    sequence, instances, height, width = value.shape
    flattened = value.reshape(sequence * instances, 1, height, width)
    if radius:
        background = (~flattened).float()
        background = F.pad(
            background,
            (radius, radius, radius, radius),
            mode="constant",
            value=1.0,
        )
        dilated_background = F.max_pool2d(
            background,
            kernel_size=2 * radius + 1,
            stride=1,
        )
        flattened = dilated_background < 0.5
    coverage = F.interpolate(
        flattened.float(),
        size=tuple(int(v) for v in patch_shape),
        mode="area",
    )
    support = coverage[:, 0] >= float(minimum_patch_coverage)
    support = support.reshape(sequence, instances, *patch_shape)
    return support & (confidence[:, :, None, None] >= float(score_threshold))


def patch_centers(
    patch_shape: tuple[int, int],
    image_size: tuple[int, int],
) -> torch.Tensor:
    """Return processed-image pixel centers as ``[P,2]`` in ``(x,y)`` order."""

    grid_height, grid_width = map(int, patch_shape)
    height, width = map(int, image_size)
    rows, columns = torch.meshgrid(
        torch.arange(grid_height, dtype=torch.float64),
        torch.arange(grid_width, dtype=torch.float64),
        indexing="ij",
    )
    x = (columns + 0.5) * (float(width) / grid_width) - 0.5
    y = (rows + 0.5) * (float(height) / grid_height) - 0.5
    return torch.stack((x, y), dim=-1).reshape(-1, 2)


def evenly_limited_indices(mask: torch.Tensor, *, limit: int) -> torch.Tensor:
    """Return deterministic, spatial-order samples from one flattened mask."""

    indices = torch.nonzero(mask.reshape(-1), as_tuple=False)[:, 0].long().cpu()
    maximum = int(limit)
    if maximum < 1:
        raise ValueError("Query limit must be positive.")
    if indices.numel() <= maximum:
        return indices
    positions = torch.linspace(0, indices.numel() - 1, steps=maximum).round().long()
    return indices.index_select(0, positions)


def mutual_nearest_matches(
    query_descriptors: torch.Tensor,
    search_descriptors: torch.Tensor,
    *,
    minimum_similarity: float,
    minimum_margin: float,
) -> MutualMatches:
    """Cosine mutual-nearest matching with a fixed forward margin gate."""

    query = query_descriptors.float()
    search = search_descriptors.float()
    if query.ndim != 2 or search.ndim != 2 or query.shape[1] != search.shape[1]:
        raise ValueError("Descriptor matrices must be [N,C] with equal C.")
    device = query.device
    if query.shape[0] == 0 or search.shape[0] == 0:
        empty_index = torch.empty(0, dtype=torch.long, device=device)
        empty_value = torch.empty(0, dtype=torch.float32, device=device)
        return MutualMatches(empty_index, empty_index.clone(), empty_value, empty_value)
    similarity = query @ search.T
    best_values, best_search = similarity.max(dim=1)
    if search.shape[0] >= 2:
        top_two = torch.topk(similarity, k=2, dim=1).values
        margins = top_two[:, 0] - top_two[:, 1]
    else:
        margins = torch.ones_like(best_values)
    reverse_query = similarity.argmax(dim=0)
    query_positions = torch.arange(query.shape[0], device=device)
    mutual = reverse_query.index_select(0, best_search) == query_positions
    valid = (
        mutual
        & torch.isfinite(best_values)
        & (best_values >= float(minimum_similarity))
        & (margins >= float(minimum_margin))
    )
    return MutualMatches(
        query_positions=query_positions[valid],
        search_positions=best_search[valid],
        similarities=best_values[valid],
        margins=margins[valid],
    )


def triangulate_camera_rays(
    pixels_xy: torch.Tensor,
    view_indices: torch.Tensor,
    world_to_camera: torch.Tensor,
    intrinsics: torch.Tensor,
) -> RayTriangulation:
    """Least-squares intersection of calibrated world-space camera rays."""

    pixels = pixels_xy.detach().double().cpu()
    views = view_indices.detach().long().cpu()
    poses = _homogeneous(world_to_camera)
    calibration = intrinsics.detach().double().cpu()
    if pixels.ndim != 2 or pixels.shape[1] != 2 or views.shape != pixels.shape[:1]:
        raise ValueError("pixels/view_indices must have shapes [V,2] and [V].")
    if pixels.shape[0] < 2 or torch.unique(views).numel() != views.numel():
        raise ValueError("Triangulation requires at least two unique views.")
    if views.min() < 0 or views.max() >= poses.shape[0]:
        raise ValueError("Triangulation view index is outside the camera sequence.")

    rotations = []
    centers = []
    directions = []
    for pixel, view in zip(pixels, views):
        pose = poses[int(view)]
        rotation = _project_rotation(pose[:3, :3])
        center = -(rotation.T @ pose[:3, 3])
        homogeneous_pixel = torch.tensor(
            [float(pixel[0]), float(pixel[1]), 1.0],
            dtype=torch.float64,
        )
        camera_direction = torch.linalg.solve(
            calibration[int(view)], homogeneous_pixel
        )
        world_direction = rotation.T @ camera_direction
        world_direction = world_direction / torch.linalg.vector_norm(
            world_direction
        ).clamp_min(1e-12)
        rotations.append(rotation)
        centers.append(center)
        directions.append(world_direction)
    centers_tensor = torch.stack(centers)
    directions_tensor = torch.stack(directions)
    identity = torch.eye(3, dtype=torch.float64)
    projectors = identity[None] - (
        directions_tensor[:, :, None] * directions_tensor[:, None, :]
    )
    system = projectors.sum(dim=0)
    right = torch.einsum("vij,vj->i", projectors, centers_tensor)
    condition = float(torch.linalg.cond(system))
    try:
        point = torch.linalg.solve(system, right)
    except RuntimeError:
        point = torch.full((3,), float("nan"), dtype=torch.float64)

    angle_values = []
    for first in range(directions_tensor.shape[0]):
        for second in range(first + 1, directions_tensor.shape[0]):
            cosine = torch.dot(
                directions_tensor[first], directions_tensor[second]
            ).clamp(-1.0, 1.0)
            angle_values.append(float(torch.rad2deg(torch.acos(cosine))))
    minimum_angle = min(angle_values) if angle_values else float("nan")
    maximum_angle = max(angle_values) if angle_values else float("nan")

    reprojection_errors = []
    positive = 0
    if torch.isfinite(point).all():
        for pixel, view, rotation in zip(pixels, views, rotations):
            pose = poses[int(view)]
            camera = rotation @ point + pose[:3, 3]
            positive += int(float(camera[2]) > 0.0)
            projected = calibration[int(view)] @ camera
            depth = float(projected[2])
            if abs(depth) < 1e-12:
                reprojection_errors.append(float("inf"))
            else:
                projected_xy = projected[:2] / projected[2]
                reprojection_errors.append(
                    float(torch.linalg.vector_norm(projected_xy - pixel))
                )
    else:
        reprojection_errors = [float("inf")] * int(pixels.shape[0])
    finite = bool(
        torch.isfinite(point).all()
        and math.isfinite(condition)
        and all(math.isfinite(value) for value in reprojection_errors)
    )
    return RayTriangulation(
        point_world=point.float(),
        condition_number=condition,
        minimum_ray_angle_degrees=minimum_angle,
        maximum_ray_angle_degrees=maximum_angle,
        mean_reprojection_error_px=(
            sum(reprojection_errors) / len(reprojection_errors)
        ),
        maximum_reprojection_error_px=max(reprojection_errors),
        positive_depth_rate=positive / int(pixels.shape[0]),
        finite=finite,
    )


def shift_patch_mask_exact(
    mask: torch.Tensor,
    *,
    shift_y: int,
    shift_x: int,
) -> torch.Tensor:
    """Area-exact cyclic shift used only by the spatial control branch."""

    if mask.ndim != 2:
        raise ValueError("Patch mask must be two-dimensional.")
    return torch.roll(mask.bool(), shifts=(int(shift_y), int(shift_x)), dims=(0, 1))


def _homogeneous(value: torch.Tensor) -> torch.Tensor:
    poses = value.detach().double().cpu()
    if poses.ndim == 4 and poses.shape[0] == 1:
        poses = poses[0]
    if poses.ndim != 3 or tuple(poses.shape[-2:]) not in {(3, 4), (4, 4)}:
        raise ValueError("Camera poses must have shape [S,3,4] or [S,4,4].")
    if tuple(poses.shape[-2:]) == (4, 4):
        return poses.clone()
    output = torch.eye(4, dtype=torch.float64).expand(poses.shape[0], 4, 4).clone()
    output[:, :3] = poses
    return output


def _project_rotation(rotation: torch.Tensor) -> torch.Tensor:
    left, _, right_t = torch.linalg.svd(rotation.double())
    projected = left @ right_t
    if torch.det(projected) < 0.0:
        left = left.clone()
        left[:, -1] *= -1.0
        projected = left @ right_t
    return projected
