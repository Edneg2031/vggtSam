"""Recover camera translation from learned world pointmaps and image rays.

The learned adapter is allowed to improve dense world geometry and camera
rotation.  Camera translation is then obtained from an explicit central-camera
constraint instead of another unconstrained regression head.  All operations
are frame-local except for the deployable reference-intrinsics stabilization.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from ..pose_evaluation import (
    _distance_statistics,
    _prepare_pose_sequence,
    _prepare_ray_inputs,
    _ray_residuals,
    _solve_weighted_ray_center,
)
from .config import RayPoseConfig


@dataclass(frozen=True)
class RayPoseResult:
    name: str
    role: str
    pose_encoding: torch.Tensor
    diagnostics: tuple[dict, ...]


@dataclass(frozen=True)
class _VariantSpec:
    pointmap_source: str
    pose_source: str
    intrinsics_source: str
    spatial_scope: str
    fit_method: str
    role: str


FINAL_RAY_POSE_NAME = (
    "ray_refined_pointmap_refined_rotation_reference_k_instances"
)
_FINAL_SPEC = _VariantSpec(
    "refined",
    "refined",
    "baseline_reference",
    "tracked_instances",
    "angular_huber",
    "deployable_v3_selected",
)


@torch.no_grad()
def recover_final_ray_pose(
    *,
    batch: dict,
    baseline_outputs: dict,
    refined_outputs: dict,
    config: RayPoseConfig,
) -> list[RayPoseResult]:
    """Recover final camera translation from refined instance-region rays."""

    if int(baseline_outputs["pose_encoding"].shape[0]) != 1:
        raise ValueError("Ray-pose recovery currently requires a single-clip batch.")
    if "world_points" not in refined_outputs or "world_confidence" not in refined_outputs:
        raise ValueError("Ray-pose recovery requires refined point-head outputs.")

    from streamvggt.utils.pose_enc import (
        extri_intri_to_pose_encoding,
        pose_encoding_to_extri_intri,
    )

    image_size = tuple(int(value) for value in batch["image_size"])
    frame_indices = [int(value) for value in batch["frame_indices"]]
    reference_index = int(batch["reference_sequence_index"])
    _, baseline_intrinsics = pose_encoding_to_extri_intri(
        baseline_outputs["pose_encoding"].float(),
        image_size_hw=image_size,
    )
    refined_w2c, _ = pose_encoding_to_extri_intri(
        refined_outputs["pose_encoding"].float(),
        image_size_hw=image_size,
    )
    refined_sequence = _prepare_pose_sequence(
        refined_w2c[0].detach().double().cpu(),
        frame_indices=frame_indices,
        source="instance_refined_camera_rotation",
    )

    baseline_points = _normalize_points(batch["baseline_world_points"])
    refined_points = _normalize_points(refined_outputs["world_points"])
    refined_confidence = _normalize_confidence(
        refined_outputs["world_confidence"],
        refined_points,
        name="refined_world_confidence",
    )
    # Preserve the selected experiment's exact common-finite support: the
    # instance mask is intersected with the frozen baseline pointmap validity,
    # while the fitted coordinates come from the refined pointmap.
    instance_mask = _tracked_instance_mask(batch, baseline_points)
    intrinsics = (
        baseline_intrinsics[0, reference_index]
        .detach()
        .double()
        .cpu()[None]
        .expand(len(frame_indices), -1, -1)
        .clone()
    )

    name = FINAL_RAY_POSE_NAME
    spec = _FINAL_SPEC
    centers = []
    diagnostic_rows = []
    for sequence_index, frame_index in enumerate(frame_indices):
        fallback_center = refined_sequence.camera_centers[sequence_index]
        rotation = refined_sequence.camera_to_world_rotation[sequence_index]
        if config.preserve_reference and sequence_index == reference_index:
            centers.append(fallback_center)
            diagnostic_rows.append(
                _preserved_reference_row(
                    name=name,
                    spec=spec,
                    sequence_index=sequence_index,
                    frame_index=frame_index,
                    center=fallback_center,
                    intrinsics=intrinsics[sequence_index],
                )
            )
            continue

        current_confidence = torch.where(
            instance_mask[sequence_index],
            refined_confidence[sequence_index],
            torch.full_like(refined_confidence[sequence_index], float("-inf")),
        )
        sampled_points, directions, weights, candidate_points = _prepare_ray_inputs(
            refined_points[sequence_index],
            current_confidence,
            intrinsics[sequence_index],
            rotation,
            confidence_threshold=float(config.confidence_threshold),
            max_points=int(config.max_points),
        )
        initial_point_stats = _distance_statistics(
            _ray_residuals(sampled_points, directions, fallback_center)
        )
        initial_angular_stats = _distance_statistics(
            _angular_residuals(sampled_points, directions, fallback_center)
        )
        fit = _fit_angular_huber_center(
            sampled_points,
            directions,
            weights,
            candidate_points=candidate_points,
            fallback_center=fallback_center,
            config=config,
        )

        proposed_shift = float(
            torch.linalg.vector_norm(fit["center"] - fallback_center)
        )
        accepted = bool(fit["solver_accepted"])
        rejection_reasons = []
        if not accepted:
            rejection_reasons.append(str(fit["status"]))
        if not math.isfinite(float(fit["point_residual_rmse"])):
            accepted = False
            rejection_reasons.append("non_finite_residual")
        elif float(fit["point_residual_rmse"]) > float(config.max_residual_rmse):
            accepted = False
            rejection_reasons.append("ray_residual_above_limit")
        if proposed_shift > float(config.max_center_shift):
            accepted = False
            rejection_reasons.append("center_shift_above_limit")
        if accepted:
            applied_center = fallback_center + float(config.blend) * (
                fit["center"] - fallback_center
            )
            policy_status = "accepted"
        else:
            applied_center = fallback_center
            policy_status = ";".join(dict.fromkeys(rejection_reasons))
        centers.append(applied_center)
        diagnostic_rows.append(
            {
                "variant": name,
                "variant_role": spec.role,
                "pointmap_source": spec.pointmap_source,
                "pose_source": spec.pose_source,
                "intrinsics_source": spec.intrinsics_source,
                "spatial_scope": spec.spatial_scope,
                "fit_method": spec.fit_method,
                "sequence_index": sequence_index,
                "frame_index": frame_index,
                "is_reference": 0,
                "solver_accepted": int(bool(fit["solver_accepted"])),
                "fit_accepted": int(accepted),
                "fit_status": policy_status,
                "solver_status": fit["status"],
                "candidate_points": int(candidate_points),
                "sampled_points": int(sampled_points.shape[0]),
                "retained_points": int(fit["retained_points"]),
                "solve_iterations": int(fit["solve_iterations"]),
                "condition_number": float(fit["condition_number"]),
                "initial_point_ray_rmse_native": initial_point_stats["rmse"],
                "fitted_point_ray_rmse_native": float(fit["point_residual_rmse"]),
                "initial_angular_rmse": initial_angular_stats["rmse"],
                "fitted_angular_rmse": float(fit["angular_residual_rmse"]),
                "proposed_center_shift_native": proposed_shift,
                "applied_center_shift_native": float(
                    torch.linalg.vector_norm(applied_center - fallback_center)
                ),
                "blend": float(config.blend),
                "fx": float(intrinsics[sequence_index, 0, 0]),
                "fy": float(intrinsics[sequence_index, 1, 1]),
                "cx": float(intrinsics[sequence_index, 0, 2]),
                "cy": float(intrinsics[sequence_index, 1, 2]),
                **_vector_fields("input_center_native", fallback_center),
                **_vector_fields("fitted_center_native", fit["center"]),
                **_vector_fields("applied_center_native", applied_center),
            }
        )

    centers_tensor = torch.stack(centers)
    c2w_rotation = refined_sequence.camera_to_world_rotation
    w2c_rotation = c2w_rotation.transpose(-1, -2)
    w2c_translation = -torch.einsum(
        "sij,sj->si",
        w2c_rotation,
        centers_tensor,
    )
    extrinsics = torch.cat(
        [w2c_rotation, w2c_translation[..., None]],
        dim=-1,
    )[None].to(
        device=refined_outputs["pose_encoding"].device,
        dtype=torch.float32,
    )
    encoded_intrinsics = intrinsics[None].to(
        device=extrinsics.device,
        dtype=torch.float32,
    )
    pose_encoding = extri_intri_to_pose_encoding(
        extrinsics,
        encoded_intrinsics,
        image_size_hw=image_size,
    )
    return [
        RayPoseResult(
            name=name,
            role=spec.role,
            pose_encoding=pose_encoding,
            diagnostics=tuple(diagnostic_rows),
        )
    ]


def _normalize_points(value: torch.Tensor) -> torch.Tensor:
    points = value.detach().double().cpu()
    if points.ndim != 5 or points.shape[0] != 1 or points.shape[-1] != 3:
        raise ValueError(
            "World points must have shape [1,S,H,W,3], got "
            f"{tuple(points.shape)}."
        )
    return points[0]


def _normalize_confidence(
    value: torch.Tensor,
    points: torch.Tensor,
    *,
    name: str,
) -> torch.Tensor:
    confidence = value.detach().double().cpu()
    if confidence.ndim == 5 and confidence.shape[-1] == 1:
        confidence = confidence[..., 0]
    if confidence.ndim != 4 or confidence.shape[0] != 1:
        raise ValueError(f"{name} must have shape [1,S,H,W] or [1,S,H,W,1].")
    confidence = confidence[0]
    if confidence.shape != points.shape[:3]:
        raise ValueError(
            f"{name} shape {tuple(confidence.shape)} disagrees with pointmap "
            f"{tuple(points.shape[:3])}."
        )
    return confidence


def _tracked_instance_mask(batch: dict, points: torch.Tensor) -> torch.Tensor:
    finite = torch.isfinite(points).all(dim=-1)
    masks = batch.get("tracking_masks_stream")
    if masks is None:
        instance_union = torch.zeros_like(finite)
    else:
        masks = masks.detach().bool().cpu()
        if masks.ndim != 5 or masks.shape[0] != 1:
            raise ValueError("tracking_masks_stream must have shape [1,S,K,H,W].")
        instance_union = masks[0].any(dim=1)
        if instance_union.shape != finite.shape:
            raise ValueError(
                "Tracking masks and ray pointmap resolution disagree: "
                f"{tuple(instance_union.shape)} vs {tuple(finite.shape)}."
            )
    return finite & instance_union


def _fit_angular_huber_center(
    points: torch.Tensor,
    directions: torch.Tensor,
    weights: torch.Tensor,
    *,
    candidate_points: int,
    fallback_center: torch.Tensor,
    config: RayPoseConfig,
) -> dict:
    sampled_points = int(points.shape[0])
    if sampled_points < int(config.min_points):
        return _angular_fallback(
            points,
            directions,
            fallback_center,
            status=f"fallback_insufficient_points:{sampled_points}<{config.min_points}",
        )
    center = fallback_center.double().cpu()
    condition_number = float("nan")
    iterations = 0
    try:
        for iteration in range(int(config.max_iterations)):
            offsets = points - center
            ranges = torch.linalg.vector_norm(offsets, dim=-1).clamp_min(
                float(config.angular_min_range)
            )
            angular = _angular_residuals(points, directions, center)
            if iteration == 0:
                robust = torch.ones_like(angular)
            else:
                delta = float(config.angular_huber_delta)
                robust = torch.where(
                    angular <= delta,
                    torch.ones_like(angular),
                    delta / angular.clamp_min(1e-12),
                )
            effective = weights * robust / ranges.square()
            effective = effective / effective.mean().clamp_min(1e-12)
            new_center, condition_number = _solve_weighted_ray_center(
                points,
                directions,
                effective,
            )
            iterations += 1
            if (
                not torch.isfinite(new_center).all()
                or not math.isfinite(condition_number)
                or condition_number > float(config.max_condition_number)
            ):
                return _angular_fallback(
                    points,
                    directions,
                    fallback_center,
                    status=f"fallback_ill_conditioned:{condition_number:.6g}",
                    iterations=iterations,
                    condition_number=condition_number,
                )
            step = torch.linalg.vector_norm(new_center - center)
            center = new_center
            if float(step) <= 1e-7:
                break
    except RuntimeError as error:
        return _angular_fallback(
            points,
            directions,
            fallback_center,
            status=f"fallback_linear_solve:{type(error).__name__}",
            iterations=iterations,
            condition_number=condition_number,
        )
    point_stats = _distance_statistics(_ray_residuals(points, directions, center))
    angular_stats = _distance_statistics(_angular_residuals(points, directions, center))
    return {
        "center": center,
        "solver_accepted": True,
        "status": "accepted_angular_huber",
        "candidate_points": int(candidate_points),
        "retained_points": sampled_points,
        "solve_iterations": iterations,
        "condition_number": condition_number,
        "point_residual_rmse": point_stats["rmse"],
        "angular_residual_rmse": angular_stats["rmse"],
    }


def _angular_fallback(
    points: torch.Tensor,
    directions: torch.Tensor,
    center: torch.Tensor,
    *,
    status: str,
    iterations: int = 0,
    condition_number: float = float("nan"),
) -> dict:
    point_stats = _distance_statistics(_ray_residuals(points, directions, center))
    angular_stats = _distance_statistics(_angular_residuals(points, directions, center))
    return {
        "center": center.double().cpu(),
        "solver_accepted": False,
        "status": status,
        "retained_points": int(points.shape[0]),
        "solve_iterations": int(iterations),
        "condition_number": float(condition_number),
        "point_residual_rmse": point_stats["rmse"],
        "angular_residual_rmse": angular_stats["rmse"],
    }


def _angular_residuals(
    points: torch.Tensor,
    directions: torch.Tensor,
    center: torch.Tensor,
) -> torch.Tensor:
    offsets = points - center
    ranges = torch.linalg.vector_norm(offsets, dim=-1)
    perpendicular = _ray_residuals(points, directions, center)
    return perpendicular / ranges.clamp_min(1e-12)


def _preserved_reference_row(
    *,
    name: str,
    spec: _VariantSpec,
    sequence_index: int,
    frame_index: int,
    center: torch.Tensor,
    intrinsics: torch.Tensor,
) -> dict:
    return {
        "variant": name,
        "variant_role": spec.role,
        "pointmap_source": spec.pointmap_source,
        "pose_source": spec.pose_source,
        "intrinsics_source": spec.intrinsics_source,
        "spatial_scope": spec.spatial_scope,
        "fit_method": spec.fit_method,
        "sequence_index": sequence_index,
        "frame_index": frame_index,
        "is_reference": 1,
        "solver_accepted": 0,
        "fit_accepted": 0,
        "fit_status": "preserved_reference",
        "solver_status": "not_requested",
        "candidate_points": 0,
        "sampled_points": 0,
        "retained_points": 0,
        "solve_iterations": 0,
        "condition_number": float("nan"),
        "initial_point_ray_rmse_native": float("nan"),
        "fitted_point_ray_rmse_native": float("nan"),
        "initial_angular_rmse": float("nan"),
        "fitted_angular_rmse": float("nan"),
        "proposed_center_shift_native": 0.0,
        "applied_center_shift_native": 0.0,
        "blend": 0.0,
        "fx": float(intrinsics[0, 0]),
        "fy": float(intrinsics[1, 1]),
        "cx": float(intrinsics[0, 2]),
        "cy": float(intrinsics[1, 2]),
        **_vector_fields("input_center_native", center),
        **_vector_fields("fitted_center_native", center),
        **_vector_fields("applied_center_native", center),
    }


def _vector_fields(prefix: str, value: torch.Tensor) -> dict[str, float]:
    vector = value.detach().double().cpu().reshape(3)
    return {
        f"{prefix}_x": float(vector[0]),
        f"{prefix}_y": float(vector[1]),
        f"{prefix}_z": float(vector[2]),
    }
