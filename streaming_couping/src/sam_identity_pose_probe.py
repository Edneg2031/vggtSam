"""Causal fixed-direction pose probe indexed by persistent SAM identities."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
import torch.nn.functional as F


@dataclass
class PointBank:
    """Bounded CPU point memory; observations are written only after probing."""

    points: torch.Tensor
    weights: torch.Tensor
    observations: int = 0

    @classmethod
    def empty(cls) -> "PointBank":
        return cls(
            points=torch.empty(0, 3, dtype=torch.float32),
            weights=torch.empty(0, dtype=torch.float32),
            observations=0,
        )

    def update(
        self,
        points: torch.Tensor,
        weights: torch.Tensor,
        *,
        max_points: int,
    ) -> None:
        valid = (
            torch.isfinite(points).all(dim=-1)
            & torch.isfinite(weights)
            & (weights > 0.0)
        )
        new_points = points[valid].detach().float().cpu()
        new_weights = weights[valid].detach().float().cpu()
        if not new_points.numel():
            return
        combined_points = torch.cat((self.points, new_points), dim=0)
        combined_weights = torch.cat((self.weights, new_weights), dim=0)
        self.points, self.weights = limit_weighted_points(
            combined_points,
            combined_weights,
            int(max_points),
        )
        self.observations += 1

    def sample(self, count: int) -> torch.Tensor:
        requested = int(count)
        if requested < 0 or requested > self.points.shape[0]:
            raise ValueError("PointBank sample count is outside the memory.")
        if requested == self.points.shape[0]:
            return self.points.clone()
        indices = torch.topk(
            self.weights,
            k=requested,
            largest=True,
            sorted=True,
        ).indices
        return self.points.index_select(0, indices)


@dataclass(frozen=True)
class ProjectionGroup:
    points: torch.Tensor
    ownership_mask: torch.Tensor | None


def fixed_pose_candidates(
    base_world_to_camera: torch.Tensor,
    *,
    rotation_step_degrees: float,
    translation_step: float,
) -> tuple[tuple[str, torch.Tensor], ...]:
    """Identity plus signed camera-frame perturbations along six DoF axes."""

    base = base_world_to_camera.detach().float().cpu()
    if base.shape != (3, 4):
        raise ValueError("base_world_to_camera must have shape [3,4].")
    if float(rotation_step_degrees) <= 0.0 or float(translation_step) <= 0.0:
        raise ValueError("Fixed pose steps must be positive.")
    output: list[tuple[str, torch.Tensor]] = [("identity", base.clone())]
    for axis, name in enumerate(("x", "y", "z")):
        for sign, suffix in ((1.0, "pos"), (-1.0, "neg")):
            angle = math.radians(float(rotation_step_degrees)) * sign
            delta_rotation = axis_angle_rotation(axis, angle)
            output.append(
                (
                    f"rotation_{name}_{suffix}",
                    left_perturb_pose(
                        base,
                        rotation=delta_rotation,
                        translation=torch.zeros(3),
                    ),
                )
            )
    for axis, name in enumerate(("x", "y", "z")):
        for sign, suffix in ((1.0, "pos"), (-1.0, "neg")):
            delta = torch.zeros(3)
            delta[axis] = sign * float(translation_step)
            output.append(
                (
                    f"translation_{name}_{suffix}",
                    left_perturb_pose(
                        base,
                        rotation=torch.eye(3),
                        translation=delta,
                    ),
                )
            )
    return tuple(output)


def evaluate_projection_groups(
    groups: Sequence[ProjectionGroup],
    *,
    world_to_camera: torch.Tensor,
    intrinsics: torch.Tensor,
    depth: torch.Tensor,
    normalized_depth_confidence: torch.Tensor,
    confidence_threshold: float,
    relative_depth_cap: float,
    mask_miss_weight: float,
) -> dict[str, float | int]:
    """Robust depth plus ownership loss on a fixed source-point support."""

    if not groups:
        raise ValueError("Projection loss requires at least one group.")
    pose = world_to_camera.detach().float().cpu()
    k = intrinsics.detach().float().cpu()
    depth_value = depth.detach().float().cpu()
    confidence = normalized_depth_confidence.detach().float().cpu()
    if pose.shape != (3, 4) or k.shape != (3, 3):
        raise ValueError("Pose/intrinsics shapes are invalid.")
    if depth_value.ndim != 2 or confidence.shape != depth_value.shape:
        raise ValueError("Depth/confidence must share shape [H,W].")
    cap = float(relative_depth_cap)
    mask_weight = float(mask_miss_weight)
    if cap <= 0.0 or mask_weight < 0.0:
        raise ValueError("Projection loss cap/weight is invalid.")
    height, width = depth_value.shape
    residual_parts: list[torch.Tensor] = []
    support_points = 0
    ownership_hits = 0
    ownership_valid_points = 0
    source_points = 0
    for group in groups:
        points = group.points.detach().float().cpu()
        if points.ndim != 2 or points.shape[-1] != 3 or not points.shape[0]:
            raise ValueError("Each projection group needs non-empty [N,3] points.")
        mask = group.ownership_mask
        if mask is not None:
            mask = mask.detach().bool().cpu()
            if mask.shape != (height, width):
                raise ValueError("Ownership mask shape differs from depth.")
        source_points += int(points.shape[0])
        camera = points @ pose[:3, :3].T + pose[:3, 3]
        camera_z = camera[:, 2]
        positive = torch.isfinite(camera).all(dim=-1) & (camera_z > 1e-6)
        projected = camera @ k.T
        u = projected[:, 0] / projected[:, 2].clamp_min(1e-6)
        v = projected[:, 1] / projected[:, 2].clamp_min(1e-6)
        in_bounds = (
            positive
            & torch.isfinite(u)
            & torch.isfinite(v)
            & (u >= 0.0)
            & (u <= width - 1)
            & (v >= 0.0)
            & (v <= height - 1)
        )
        pixel_x = u.round().long().clamp(0, width - 1)
        pixel_y = v.round().long().clamp(0, height - 1)
        observed_depth = depth_value[pixel_y, pixel_x]
        observed_confidence = confidence[pixel_y, pixel_x]
        supported = (
            in_bounds
            & torch.isfinite(observed_depth)
            & (observed_depth > 1e-6)
            & torch.isfinite(observed_confidence)
            & (observed_confidence >= float(confidence_threshold))
        )
        support_points += int(supported.sum())
        geometry = torch.full_like(camera_z, cap)
        relative = (
            (camera_z - observed_depth).abs()
            / observed_depth.clamp_min(1e-6)
        )
        geometry[supported] = relative[supported].clamp_max(cap)
        if mask is None:
            ownership = torch.zeros_like(geometry)
        else:
            hit = in_bounds & mask[pixel_y, pixel_x]
            ownership_hits += int(hit.sum())
            ownership_valid_points += int(in_bounds.sum())
            ownership = (~hit).float() * mask_weight
        residual_parts.append(geometry + ownership)
    residual = torch.cat(residual_parts)
    return {
        "loss": float(residual.mean()),
        "source_points": source_points,
        "depth_support_points": support_points,
        "depth_support_rate": support_points / max(source_points, 1),
        "ownership_hit_rate": (
            ownership_hits / max(ownership_valid_points, 1)
            if ownership_valid_points
            else 1.0
        ),
    }


def erode_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    value = mask.detach().bool().cpu()
    amount = int(radius)
    if value.ndim != 2:
        raise ValueError("Mask erosion expects [H,W].")
    if amount <= 0:
        return value.clone()
    inverse = (~value)[None, None].float()
    dilated_inverse = F.max_pool2d(
        inverse,
        kernel_size=2 * amount + 1,
        stride=1,
        padding=amount,
    )[0, 0].bool()
    return ~dilated_inverse


def shift_mask(
    mask: torch.Tensor,
    *,
    shift_y: int,
    shift_x: int,
) -> torch.Tensor:
    """Translate without circular wrap-around."""

    value = mask.detach().bool().cpu()
    if value.ndim != 2:
        raise ValueError("Mask shift expects [H,W].")
    height, width = value.shape
    dy, dx = int(shift_y), int(shift_x)
    output = torch.zeros_like(value)
    source_y0 = max(0, -dy)
    source_y1 = min(height, height - dy)
    source_x0 = max(0, -dx)
    source_x1 = min(width, width - dx)
    if source_y1 <= source_y0 or source_x1 <= source_x0:
        return output
    target_y0 = source_y0 + dy
    target_y1 = source_y1 + dy
    target_x0 = source_x0 + dx
    target_x1 = source_x1 + dx
    output[target_y0:target_y1, target_x0:target_x1] = value[
        source_y0:source_y1,
        source_x0:source_x1,
    ]
    return output


def masked_weighted_points(
    world_points: torch.Tensor,
    normalized_confidence: torch.Tensor,
    mask: torch.Tensor,
    *,
    confidence_threshold: float,
    max_points: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    points = world_points.detach().float().cpu()
    confidence = normalized_confidence.detach().float().cpu()
    region = mask.detach().bool().cpu()
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("world_points must have shape [H,W,3].")
    if confidence.shape != points.shape[:2] or region.shape != points.shape[:2]:
        raise ValueError("Confidence/mask shape differs from world_points.")
    valid = (
        region
        & torch.isfinite(points).all(dim=-1)
        & torch.isfinite(confidence)
        & (confidence >= float(confidence_threshold))
    )
    return limit_weighted_points(
        points[valid],
        confidence[valid],
        int(max_points),
    )


def limit_weighted_points(
    points: torch.Tensor,
    weights: torch.Tensor,
    limit: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError("Points must have shape [N,3].")
    if weights.ndim != 1 or weights.shape[0] != points.shape[0]:
        raise ValueError("Weights must have shape [N].")
    maximum = int(limit)
    if maximum <= 0:
        raise ValueError("Point limit must be positive.")
    if points.shape[0] <= maximum:
        return points.detach().float().cpu(), weights.detach().float().cpu()
    indices = torch.topk(weights, k=maximum, largest=True, sorted=True).indices
    return (
        points.index_select(0, indices).detach().float().cpu(),
        weights.index_select(0, indices).detach().float().cpu(),
    )


def left_perturb_pose(
    world_to_camera: torch.Tensor,
    *,
    rotation: torch.Tensor,
    translation: torch.Tensor,
) -> torch.Tensor:
    base = world_to_camera.detach().float().cpu()
    delta_rotation = rotation.detach().float().cpu()
    delta_translation = translation.detach().float().cpu()
    if base.shape != (3, 4):
        raise ValueError("world_to_camera must have shape [3,4].")
    if delta_rotation.shape != (3, 3) or delta_translation.shape != (3,):
        raise ValueError("SE3 perturbation shapes are invalid.")
    output = torch.empty_like(base)
    output[:3, :3] = delta_rotation @ base[:3, :3]
    output[:3, 3] = delta_rotation @ base[:3, 3] + delta_translation
    return output


def axis_angle_rotation(axis: int, angle: float) -> torch.Tensor:
    index = int(axis)
    if index not in (0, 1, 2):
        raise ValueError("Rotation axis must be 0, 1 or 2.")
    vector = torch.zeros(3)
    vector[index] = 1.0
    x, y, z = vector
    skew = torch.zeros(3, 3, dtype=torch.float32)
    skew[0, 1], skew[0, 2] = -z, y
    skew[1, 0], skew[1, 2] = z, -x
    skew[2, 0], skew[2, 1] = -y, x
    value = float(angle)
    return (
        torch.eye(3)
        + math.sin(value) * skew
        + (1.0 - math.cos(value)) * (skew @ skew)
    )
