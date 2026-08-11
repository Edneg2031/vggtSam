"""Spatial-support construction for the V9.5 epipolar feasibility audit.

The helpers in this module deliberately contain no learned component.  They
construct equal-count correspondence sets so that localized instance support,
full-image support and a balanced instance/background support can be compared
without confounding spatial extent with correspondence count.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch

from streaming_couping.src.v90_epipolar_geometry import SurfaceCorrespondences


SUPPORT_SCOPES = (
    "instance_local32",
    "full_image_equal_count",
    "instance_background_balanced",
)


@dataclass(frozen=True)
class SpatialSupport:
    scope: str
    correspondences: SurfaceCorrespondences
    instance_rows: int
    background_rows: int

    @property
    def count(self) -> int:
        return self.correspondences.count

    @property
    def instance_fraction(self) -> float:
        return _ratio(self.instance_rows, self.count)

    @property
    def background_fraction(self) -> float:
        return _ratio(self.background_rows, self.count)


def region_mask_streams(instance_masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return all-image and non-instance masks with one synthetic slot."""

    if instance_masks.ndim != 4:
        raise ValueError("V9.5 instance masks must be [S,K,H,W].")
    union = instance_masks.bool().any(dim=1, keepdim=True)
    full = torch.ones_like(union)
    background = ~union
    return full, background


def concatenate_surface_rows(
    rows: Sequence[SurfaceCorrespondences],
    *,
    current_frame: int,
    history_frame: int,
    slot: int = -1,
) -> SurfaceCorrespondences:
    """Concatenate correspondence rows while retaining a valid empty value."""

    valid = [row for row in rows if row.count]
    if not valid:
        return empty_surface(current_frame, history_frame, slot=slot)
    return SurfaceCorrespondences(
        current_frame=int(current_frame),
        history_frame=int(history_frame),
        slot=int(slot),
        current_uv=torch.cat([row.current_uv.double() for row in valid], dim=0),
        history_uv=torch.cat([row.history_uv.double() for row in valid], dim=0),
        weights=torch.cat([row.weights.double() for row in valid], dim=0),
        depth_residual_metric=torch.cat(
            [row.depth_residual_metric.double() for row in valid], dim=0
        ),
        sampled_queries=sum(int(row.sampled_queries) for row in rows),
        projected_in_bounds=sum(int(row.projected_in_bounds) for row in rows),
        visible_queries=sum(int(row.visible_queries) for row in rows),
    )


def build_equal_count_supports(
    *,
    instance: SurfaceCorrespondences,
    full_image_candidates: SurfaceCorrespondences,
    background_candidates: SurfaceCorrespondences,
    instance_union_mask: torch.Tensor,
    image_size: tuple[int, int],
    background_fraction: float = 0.5,
) -> dict[str, SpatialSupport]:
    """Build the three locked V9.5 support scopes at the instance row count.

    Selection is deterministic farthest-point sampling in joint current/history
    image coordinates.  It is a GT diagnostic upper bound: history UV is used
    only to make the support as spatially informative as possible.
    """

    if instance.current_frame != full_image_candidates.current_frame or (
        instance.history_frame != full_image_candidates.history_frame
    ):
        raise ValueError("V9.5 support candidates refer to different edges.")
    if background_candidates.current_frame != instance.current_frame or (
        background_candidates.history_frame != instance.history_frame
    ):
        raise ValueError("V9.5 background candidates refer to a different edge.")
    fraction = float(background_fraction)
    if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError("V9.5 background fraction must lie strictly inside (0,1).")
    target = instance.count
    if target == 0:
        empty = empty_surface(instance.current_frame, instance.history_frame)
        return {
            scope: SpatialSupport(scope, empty, 0, 0) for scope in SUPPORT_SCOPES
        }

    instance_fixed = take_surface(instance, spatial_farthest_indices(instance, target, image_size))
    full = take_surface(
        full_image_candidates,
        spatial_farthest_indices(full_image_candidates, target, image_size),
    )
    if full.count != target:
        raise ValueError(
            f"V9.5 full-image candidates={full.count}, required equal count={target}."
        )

    desired_background = min(target, int(round(target * fraction)))
    desired_instance = target - desired_background
    selected_instance = take_surface(
        instance_fixed,
        spatial_farthest_indices(instance_fixed, desired_instance, image_size),
    )
    selected_background = take_surface(
        background_candidates,
        spatial_farthest_indices(background_candidates, desired_background, image_size),
    )
    missing = target - selected_instance.count - selected_background.count
    if missing > 0:
        # A shortage in one region is filled deterministically from the other.
        if selected_background.count < desired_background:
            selected_instance = take_surface(
                instance_fixed,
                spatial_farthest_indices(
                    instance_fixed, selected_instance.count + missing, image_size
                ),
            )
        else:
            selected_background = take_surface(
                background_candidates,
                spatial_farthest_indices(
                    background_candidates,
                    selected_background.count + missing,
                    image_size,
                ),
            )
    hybrid = concatenate_surface_rows(
        [selected_instance, selected_background],
        current_frame=instance.current_frame,
        history_frame=instance.history_frame,
    )
    if hybrid.count != target:
        raise ValueError(
            f"V9.5 hybrid support={hybrid.count}, required equal count={target}."
        )

    full_instance_rows = count_rows_inside_mask(full.current_uv, instance_union_mask)
    output = {
        "instance_local32": SpatialSupport(
            "instance_local32", instance_fixed, instance_fixed.count, 0
        ),
        "full_image_equal_count": SpatialSupport(
            "full_image_equal_count",
            full,
            full_instance_rows,
            full.count - full_instance_rows,
        ),
        "instance_background_balanced": SpatialSupport(
            "instance_background_balanced",
            hybrid,
            selected_instance.count,
            selected_background.count,
        ),
    }
    if tuple(output) != SUPPORT_SCOPES:
        raise RuntimeError("V9.5 support scope order changed.")
    return output


def perturb_history_uv(
    row: SurfaceCorrespondences,
    *,
    sigma_pixels: float,
    seed: int,
) -> tuple[SurfaceCorrespondences, torch.Tensor]:
    """Add deterministic isotropic noise after support selection."""

    sigma = float(sigma_pixels)
    if sigma not in {0.0, 0.5, 1.0}:
        raise ValueError("V9.5 noise sigma must be 0/0.5/1 pixels.")
    if row.count == 0 or sigma == 0.0:
        errors = torch.zeros(row.count, dtype=torch.float64)
        return clone_surface(row), errors
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    noise = torch.randn(
        (row.count, 2), generator=generator, dtype=torch.float64
    ) * sigma
    output = clone_surface(row)
    output.history_uv = output.history_uv + noise
    return output, torch.linalg.vector_norm(noise, dim=-1)


def spatial_farthest_indices(
    row: SurfaceCorrespondences,
    count: int,
    image_size: tuple[int, int],
) -> torch.Tensor:
    """Select a deterministic joint-view farthest-point subset."""

    requested = max(0, min(int(count), row.count))
    if requested == 0:
        return torch.empty(0, dtype=torch.long)
    if requested == row.count:
        return torch.arange(row.count, dtype=torch.long)
    height, width = (int(value) for value in image_size)
    scale = torch.tensor(
        [max(width - 1, 1), max(height - 1, 1)], dtype=torch.float64
    )
    features = torch.cat(
        [row.current_uv.double() / scale, row.history_uv.double() / scale], dim=-1
    )
    finite = torch.isfinite(features).all(dim=-1)
    if int(finite.sum()) < requested:
        raise ValueError("V9.5 support contains too few finite UV rows.")
    valid_indices = torch.nonzero(finite, as_tuple=False).flatten()
    features = features.index_select(0, valid_indices)
    center = features.mean(dim=0)
    first = torch.linalg.vector_norm(features - center, dim=-1).argmax()
    selected = torch.empty(requested, dtype=torch.long)
    selected[0] = first
    minimum = torch.linalg.vector_norm(features - features[first], dim=-1)
    minimum[first] = -1.0
    for index in range(1, requested):
        choice = minimum.argmax()
        selected[index] = choice
        distance = torch.linalg.vector_norm(features - features[choice], dim=-1)
        minimum = torch.minimum(minimum, distance)
        minimum[choice] = -1.0
    return valid_indices.index_select(0, selected)


def take_surface(
    row: SurfaceCorrespondences, indices: torch.Tensor
) -> SurfaceCorrespondences:
    indices = indices.long().cpu()
    return SurfaceCorrespondences(
        current_frame=row.current_frame,
        history_frame=row.history_frame,
        slot=row.slot,
        current_uv=row.current_uv.index_select(0, indices).double(),
        history_uv=row.history_uv.index_select(0, indices).double(),
        weights=row.weights.index_select(0, indices).double(),
        depth_residual_metric=row.depth_residual_metric.index_select(0, indices).double(),
        sampled_queries=row.sampled_queries,
        projected_in_bounds=row.projected_in_bounds,
        visible_queries=row.visible_queries,
    )


def clone_surface(row: SurfaceCorrespondences) -> SurfaceCorrespondences:
    return SurfaceCorrespondences(
        current_frame=row.current_frame,
        history_frame=row.history_frame,
        slot=row.slot,
        current_uv=row.current_uv.double().clone(),
        history_uv=row.history_uv.double().clone(),
        weights=row.weights.double().clone(),
        depth_residual_metric=row.depth_residual_metric.double().clone(),
        sampled_queries=row.sampled_queries,
        projected_in_bounds=row.projected_in_bounds,
        visible_queries=row.visible_queries,
    )


def empty_surface(
    current_frame: int, history_frame: int, *, slot: int = -1
) -> SurfaceCorrespondences:
    empty_uv = torch.empty((0, 2), dtype=torch.float64)
    return SurfaceCorrespondences(
        current_frame=int(current_frame),
        history_frame=int(history_frame),
        slot=int(slot),
        current_uv=empty_uv,
        history_uv=empty_uv.clone(),
        weights=torch.empty(0, dtype=torch.float64),
        depth_residual_metric=torch.empty(0, dtype=torch.float64),
        sampled_queries=0,
        projected_in_bounds=0,
        visible_queries=0,
    )


def count_rows_inside_mask(uv: torch.Tensor, mask: torch.Tensor) -> int:
    if mask.ndim != 2:
        raise ValueError("V9.5 union mask must be [H,W].")
    if uv.numel() == 0:
        return 0
    height, width = mask.shape
    rounded = uv.double().round().long()
    x = rounded[:, 0].clamp(0, width - 1)
    y = rounded[:, 1].clamp(0, height - 1)
    finite = torch.isfinite(uv).all(dim=-1)
    return int((finite & mask.bool()[y, x]).sum())


def uv_hull_coverage(uv: torch.Tensor, image_size: tuple[int, int]) -> float:
    finite = uv[torch.isfinite(uv).all(dim=-1)].double().cpu().tolist()
    points = sorted({(float(row[0]), float(row[1])) for row in finite})
    if len(points) < 3:
        return 0.0

    def cross(origin, first, second) -> float:
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (
            first[1] - origin[1]
        ) * (second[0] - origin[0])

    lower = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    hull = lower[:-1] + upper[:-1]
    area = 0.5 * abs(
        sum(
            first[0] * second[1] - second[0] * first[1]
            for first, second in zip(hull, hull[1:] + hull[:1])
        )
    )
    height, width = (int(value) for value in image_size)
    return area / float(max(width - 1, 1) * max(height - 1, 1))


def _ratio(numerator: int, denominator: int) -> float:
    return 0.0 if int(denominator) <= 0 else float(numerator) / float(denominator)
