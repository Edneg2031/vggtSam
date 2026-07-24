"""Recover camera translation from learned world pointmaps and image rays.

The learned adapter is allowed to improve dense world geometry and camera
rotation.  Camera translation is then obtained from an explicit central-camera
constraint instead of another unconstrained regression head.  All operations
are frame-local except for the deployable reference-intrinsics stabilization.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
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
    "deployable_v4_strict_geometry",
)
_LEGACY_SPEC = _VariantSpec(
    "refined",
    "refined",
    "baseline_reference",
    "tracked_instances",
    "angular_huber",
    "deployable_v3_selected",
)
HISTORICAL_ANCHOR_POSE_NAME = (
    "ray_historical_anchor_refined_rotation"
)
CURRENT_RAW_POSE_NAME = "ray_current_raw_pointmap_refined_rotation"
REFERENCE_BLEND_ROLE = "reference_preserving_blend_ablation"


def reference_blend_pose_name(blend: float) -> str:
    """Stable CSV/checkpoint key for a reference-preserving blend policy."""

    percent = int(round(100.0 * float(blend)))
    if not math.isclose(float(blend), percent / 100.0, abs_tol=1e-8):
        raise ValueError(
            "Reference blend names require values representable as an "
            "integer percentage."
        )
    return (
        "ray_current_refined_preserve_reference_"
        f"blend_{percent:03d}"
    )


@torch.no_grad()
def recover_final_ray_pose(
    *,
    batch: dict,
    baseline_outputs: dict,
    refined_outputs: dict,
    config: RayPoseConfig,
) -> list[RayPoseResult]:
    """Evaluate configured analytic translation sources on fixed upstream outputs."""

    results: list[RayPoseResult] = []
    for solver_mode in config.solver_modes:
        if solver_mode == "historical_anchor":
            results.append(
                _recover_historical_anchor_pose(
                    batch=batch,
                    baseline_outputs=baseline_outputs,
                    refined_outputs=refined_outputs,
                    config=config,
                )
            )
        elif solver_mode in {"current_raw", "current_refined"}:
            results.append(
                _recover_current_pointmap_pose(
                    batch=batch,
                    baseline_outputs=baseline_outputs,
                    refined_outputs=refined_outputs,
                    config=config,
                    point_source=solver_mode,
                )
            )
        else:
            raise ValueError(f"Unknown ray-pose solver mode: {solver_mode!r}.")
    for blend in config.reference_blend_values:
        sweep_config = replace(
            config,
            preserve_reference=True,
            blend=float(blend),
            reference_blend_values=(),
        )
        results.append(
            _recover_current_pointmap_pose(
                batch=batch,
                baseline_outputs=baseline_outputs,
                refined_outputs=refined_outputs,
                config=sweep_config,
                point_source="current_refined",
                name_override=reference_blend_pose_name(blend),
                role_override=REFERENCE_BLEND_ROLE,
            )
        )
    return results


@torch.no_grad()
def _recover_current_pointmap_pose(
    *,
    batch: dict,
    baseline_outputs: dict,
    refined_outputs: dict,
    config: RayPoseConfig,
    point_source: str,
    name_override: str | None = None,
    role_override: str | None = None,
) -> RayPoseResult:
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
    if point_source == "current_refined":
        fitted_points = _normalize_points(refined_outputs["world_points"])
        fitted_confidence = _normalize_confidence(
            refined_outputs["world_confidence"],
            fitted_points,
            name="refined_world_confidence",
        )
    elif point_source == "current_raw":
        fitted_points = baseline_points
        fitted_confidence = _normalize_confidence(
            batch["baseline_world_confidence"],
            fitted_points,
            name="baseline_world_confidence",
        )
    else:
        raise ValueError(f"Unknown current point source: {point_source!r}.")
    # Preserve the selected experiment's exact common-finite support: the
    # instance mask is intersected with the frozen baseline pointmap validity,
    # while the fitted coordinates come from the refined pointmap.
    instance_mask = _tracked_instance_mask(batch, baseline_points)
    trusted_instance_valid = _trusted_instance_valid(batch)
    instance_ids = tuple(int(value) for value in batch.get("instance_ids", ()))
    intrinsics = (
        baseline_intrinsics[0, reference_index]
        .detach()
        .double()
        .cpu()[None]
        .expand(len(frame_indices), -1, -1)
        .clone()
    )

    name = (
        FINAL_RAY_POSE_NAME
        if point_source == "current_refined"
        else CURRENT_RAW_POSE_NAME
    )
    if name_override is not None:
        name = str(name_override)
    spec = (
        _FINAL_SPEC
        if bool(batch.get("strict_identity_gate", False))
        else _LEGACY_SPEC
    )
    if point_source == "current_raw":
        spec = _VariantSpec(
            "baseline",
            "refined",
            "baseline_reference",
            "tracked_instances",
            "angular_huber",
            "solver_source_ablation",
        )
    if role_override is not None:
        spec = replace(spec, role=str(role_override))
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
            fitted_confidence[sequence_index],
            torch.full_like(fitted_confidence[sequence_index], float("-inf")),
        )
        sampled_points, directions, weights, candidate_points = _prepare_ray_inputs(
            fitted_points[sequence_index],
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
        accepted, rejection_reasons = _accept_center_fit(
            fit,
            proposed_shift=proposed_shift,
            config=config,
        )
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
                "trusted_instance_ids": _instance_id_text(
                    instance_ids,
                    trusted_instance_valid[sequence_index],
                    selected=True,
                ),
                "rejected_instance_ids": _instance_id_text(
                    instance_ids,
                    trusted_instance_valid[sequence_index],
                    selected=False,
                ),
                "trusted_instance_count": int(
                    trusted_instance_valid[sequence_index].sum()
                ),
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
    return RayPoseResult(
        name=name,
        role=spec.role,
        pose_encoding=pose_encoding,
        diagnostics=tuple(diagnostic_rows),
    )


@torch.no_grad()
def _recover_historical_anchor_pose(
    *,
    batch: dict,
    baseline_outputs: dict,
    refined_outputs: dict,
    config: RayPoseConfig,
) -> RayPoseResult:
    """Solve centers from prior-frame instance anchors and current pixels."""

    from streamvggt.utils.pose_enc import (
        extri_intri_to_pose_encoding,
        pose_encoding_to_extri_intri,
    )

    image_size = tuple(int(value) for value in batch["image_size"])
    frame_indices = [int(value) for value in batch["frame_indices"]]
    reference_index = int(batch["reference_sequence_index"])
    baseline_w2c, baseline_intrinsics = pose_encoding_to_extri_intri(
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
        source="historical_anchor_refined_rotation",
    )
    baseline_w2c = baseline_w2c[0].detach().double().cpu()
    baseline_points = _normalize_points(batch["baseline_world_points"])
    baseline_confidence = _normalize_confidence(
        batch["baseline_world_confidence"],
        baseline_points,
        name="baseline_world_confidence",
    )
    masks = batch["trusted_tracking_masks_stream"].detach().bool().cpu()[0]
    trusted = _trusted_instance_valid(batch)
    instance_ids = tuple(int(value) for value in batch.get("instance_ids", ()))
    intrinsics = (
        baseline_intrinsics[0, reference_index]
        .detach()
        .double()
        .cpu()[None]
        .expand(len(frame_indices), -1, -1)
        .clone()
    )
    instance_count = masks.shape[1]
    object_maps = [torch.empty(0, 3, dtype=torch.float64) for _ in range(instance_count)]
    anchor_frames: list[list[int]] = [[] for _ in range(instance_count)]
    centers = []
    diagnostic_rows = []
    fit_config = replace(
        config,
        min_points=int(config.historical_min_correspondences),
    )

    for sequence_index, frame_index in enumerate(frame_indices):
        fallback_center = refined_sequence.camera_centers[sequence_index]
        rotation = refined_sequence.camera_to_world_rotation[sequence_index]
        matched_points = []
        matched_pixels = []
        matched_weights = []
        per_instance_counts = []
        source_frames: set[int] = set()
        current_samples: list[
            tuple[int, torch.Tensor, torch.Tensor]
        ] = []
        for slot in range(instance_count):
            current_points, pixels, weights = _sample_masked_world_pixels(
                baseline_points[sequence_index],
                baseline_confidence[sequence_index],
                masks[sequence_index, slot],
                confidence_threshold=float(config.confidence_threshold),
                max_points=int(
                    config.historical_max_points_per_instance
                ),
            )
            current_samples.append((slot, current_points, pixels))
            if (
                sequence_index == reference_index
                or not bool(trusted[sequence_index, slot])
                or object_maps[slot].shape[0]
                < int(config.historical_min_correspondences)
                or current_points.shape[0]
                < int(config.historical_min_correspondences)
            ):
                per_instance_counts.append(0)
                continue
            historical, current_pixel, current_weight = (
                _historical_correspondences(
                    current_points,
                    pixels,
                    weights,
                    object_maps[slot],
                    config=config,
                )
            )
            per_instance_counts.append(int(historical.shape[0]))
            if historical.numel():
                matched_points.append(historical)
                matched_pixels.append(current_pixel)
                matched_weights.append(current_weight)
                source_frames.update(anchor_frames[slot])

        if sequence_index == reference_index:
            applied_center = fallback_center
            fit = _angular_fallback(
                torch.empty(0, 3, dtype=torch.float64),
                torch.empty(0, 3, dtype=torch.float64),
                fallback_center,
                status="reference_initializes_historical_anchors",
            )
            accepted = False
            proposed_shift = 0.0
            policy_status = "reference_anchor_initialization"
            initial_point_stats = _distance_statistics(
                torch.empty(0, dtype=torch.float64)
            )
            initial_angular_stats = initial_point_stats
            sampled_points = torch.empty(0, 3, dtype=torch.float64)
            candidate_points = 0
        else:
            if matched_points:
                sampled_points = torch.cat(matched_points, dim=0)
                sampled_pixels = torch.cat(matched_pixels, dim=0)
                weights = torch.cat(matched_weights, dim=0)
            else:
                sampled_points = torch.empty(0, 3, dtype=torch.float64)
                sampled_pixels = torch.empty(0, 2, dtype=torch.float64)
                weights = torch.empty(0, dtype=torch.float64)
            candidate_points = int(sampled_points.shape[0])
            directions = _pixel_world_directions(
                sampled_pixels,
                intrinsics[sequence_index],
                rotation,
            )
            initial_point_stats = _distance_statistics(
                _ray_residuals(
                    sampled_points,
                    directions,
                    fallback_center,
                )
            )
            initial_angular_stats = _distance_statistics(
                _angular_residuals(
                    sampled_points,
                    directions,
                    fallback_center,
                )
            )
            fit = _fit_angular_huber_center(
                sampled_points,
                directions,
                weights,
                candidate_points=candidate_points,
                fallback_center=fallback_center,
                config=fit_config,
            )
            proposed_shift = float(
                torch.linalg.vector_norm(
                    fit["center"] - fallback_center
                )
            )
            accepted, rejection_reasons = _accept_center_fit(
                fit,
                proposed_shift=proposed_shift,
                config=config,
            )
            if accepted:
                applied_center = fallback_center + float(config.blend) * (
                    fit["center"] - fallback_center
                )
                policy_status = "accepted"
            else:
                applied_center = fallback_center
                policy_status = ";".join(
                    dict.fromkeys(rejection_reasons)
                )
        centers.append(applied_center)

        # Add only geometrically trusted observations. Non-reference frames
        # enter long-term anchors only after a successful historical fit.
        if sequence_index == reference_index or accepted:
            for slot, current_points, _ in current_samples:
                if (
                    not bool(trusted[sequence_index, slot])
                    or current_points.shape[0]
                    < int(config.historical_min_correspondences)
                ):
                    continue
                if sequence_index == reference_index:
                    reposed = current_points
                else:
                    camera_local = (
                        current_points @ baseline_w2c[
                            sequence_index, :3, :3
                        ].T
                        + baseline_w2c[sequence_index, :3, 3]
                    )
                    reposed = (
                        camera_local @ rotation.T + applied_center
                    )
                object_maps[slot] = _limit_points(
                    torch.cat([object_maps[slot], reposed], dim=0),
                    int(config.historical_max_points_per_instance),
                )
                if frame_index not in anchor_frames[slot]:
                    anchor_frames[slot].append(frame_index)

        diagnostic_rows.append(
            {
                "variant": HISTORICAL_ANCHOR_POSE_NAME,
                "variant_role": "historical_anchor_ablation",
                "pointmap_source": "historical_raw_instance_anchors",
                "pose_source": "refined",
                "intrinsics_source": "baseline_reference",
                "spatial_scope": "tracked_instances",
                "fit_method": "historical_anchor_angular_huber",
                "correspondence_source": (
                    "raw_pointmap_icp_to_prior_instance_map"
                ),
                "sequence_index": sequence_index,
                "frame_index": frame_index,
                "is_reference": int(
                    sequence_index == reference_index
                ),
                "solver_accepted": int(
                    bool(fit["solver_accepted"])
                ),
                "fit_accepted": int(accepted),
                "fit_status": policy_status,
                "solver_status": fit["status"],
                "candidate_points": candidate_points,
                "sampled_points": int(sampled_points.shape[0]),
                "retained_points": int(fit["retained_points"]),
                "solve_iterations": int(fit["solve_iterations"]),
                "condition_number": float(fit["condition_number"]),
                "initial_point_ray_rmse_native": initial_point_stats[
                    "rmse"
                ],
                "fitted_point_ray_rmse_native": float(
                    fit["point_residual_rmse"]
                ),
                "initial_angular_rmse": initial_angular_stats["rmse"],
                "fitted_angular_rmse": float(
                    fit["angular_residual_rmse"]
                ),
                "proposed_center_shift_native": proposed_shift,
                "applied_center_shift_native": float(
                    torch.linalg.vector_norm(
                        applied_center - fallback_center
                    )
                ),
                "blend": float(config.blend),
                "trusted_instance_ids": _instance_id_text(
                    instance_ids,
                    trusted[sequence_index],
                    selected=True,
                ),
                "rejected_instance_ids": _instance_id_text(
                    instance_ids,
                    trusted[sequence_index],
                    selected=False,
                ),
                "trusted_instance_count": int(
                    trusted[sequence_index].sum()
                ),
                "per_instance_correspondences": " ".join(
                    str(value) for value in per_instance_counts
                ),
                "anchor_source_frame_ids": " ".join(
                    str(value) for value in sorted(source_frames)
                ),
                "fx": float(intrinsics[sequence_index, 0, 0]),
                "fy": float(intrinsics[sequence_index, 1, 1]),
                "cx": float(intrinsics[sequence_index, 0, 2]),
                "cy": float(intrinsics[sequence_index, 1, 2]),
                **_vector_fields(
                    "input_center_native",
                    fallback_center,
                ),
                **_vector_fields(
                    "fitted_center_native",
                    fit["center"],
                ),
                **_vector_fields(
                    "applied_center_native",
                    applied_center,
                ),
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
    pose_encoding = extri_intri_to_pose_encoding(
        extrinsics,
        intrinsics[None].to(
            device=extrinsics.device,
            dtype=torch.float32,
        ),
        image_size_hw=image_size,
    )
    return RayPoseResult(
        name=HISTORICAL_ANCHOR_POSE_NAME,
        role="historical_anchor_ablation",
        pose_encoding=pose_encoding,
        diagnostics=tuple(diagnostic_rows),
    )


def _sample_masked_world_pixels(
    points: torch.Tensor,
    confidence: torch.Tensor,
    mask: torch.Tensor,
    *,
    confidence_threshold: float,
    max_points: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    valid = (
        mask.bool()
        & torch.isfinite(points).all(dim=-1)
        & torch.isfinite(confidence)
        & (confidence >= float(confidence_threshold))
    )
    pixels_yx = torch.nonzero(valid, as_tuple=False)
    if not pixels_yx.numel():
        return (
            torch.empty(0, 3, dtype=torch.float64),
            torch.empty(0, 2, dtype=torch.float64),
            torch.empty(0, dtype=torch.float64),
        )
    if pixels_yx.shape[0] > int(max_points):
        keep = torch.linspace(
            0,
            pixels_yx.shape[0] - 1,
            steps=int(max_points),
        ).round().long()
        pixels_yx = pixels_yx.index_select(0, keep)
    selected_points = points[
        pixels_yx[:, 0],
        pixels_yx[:, 1],
    ].double()
    selected_weights = confidence[
        pixels_yx[:, 0],
        pixels_yx[:, 1],
    ].double().clamp_min(1e-6)
    pixels_uv = pixels_yx[:, [1, 0]].double()
    return selected_points, pixels_uv, selected_weights


def _historical_correspondences(
    current_points: torch.Tensor,
    current_pixels: torch.Tensor,
    current_weights: torch.Tensor,
    historical_points: torch.Tensor,
    *,
    config: RayPoseConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    initialization = (
        torch.quantile(historical_points, 0.5, dim=0)
        - torch.quantile(current_points, 0.5, dim=0)
    )
    if float(torch.linalg.vector_norm(initialization)) > float(
        config.max_center_shift
    ):
        initialization = torch.zeros_like(initialization)
    shifted = current_points + initialization
    low = torch.quantile(historical_points, 0.05, dim=0)
    high = torch.quantile(historical_points, 0.95, dim=0)
    object_scale = float(torch.linalg.vector_norm(high - low))
    maximum_distance = max(
        float(config.historical_min_distance),
        float(config.historical_object_ratio) * object_scale,
    )
    # Avoid materializing a 4096 x 4096 double matrix for every instance.
    nearest_distance_rows = []
    nearest_index_rows = []
    chunk_size = 512
    for start in range(0, shifted.shape[0], chunk_size):
        distances = torch.cdist(
            shifted[start : start + chunk_size],
            historical_points,
        )
        distance, index = distances.min(dim=1)
        nearest_distance_rows.append(distance)
        nearest_index_rows.append(index)
    nearest_distance = torch.cat(nearest_distance_rows)
    nearest_index = torch.cat(nearest_index_rows)
    keep = nearest_distance <= maximum_distance
    if int(keep.sum()) < int(config.historical_min_correspondences):
        return (
            torch.empty(0, 3, dtype=torch.float64),
            torch.empty(0, 2, dtype=torch.float64),
            torch.empty(0, dtype=torch.float64),
        )
    return (
        historical_points.index_select(0, nearest_index[keep]),
        current_pixels[keep],
        current_weights[keep],
    )


def _pixel_world_directions(
    pixels_uv: torch.Tensor,
    intrinsics: torch.Tensor,
    camera_to_world_rotation: torch.Tensor,
) -> torch.Tensor:
    if not pixels_uv.numel():
        return torch.empty(0, 3, dtype=torch.float64)
    u, v = pixels_uv.unbind(dim=-1)
    camera = torch.stack(
        [
            (u - intrinsics[0, 2]) / intrinsics[0, 0],
            (v - intrinsics[1, 2]) / intrinsics[1, 1],
            torch.ones_like(u),
        ],
        dim=-1,
    )
    camera = torch.nn.functional.normalize(camera, dim=-1)
    world = camera @ camera_to_world_rotation.T
    return torch.nn.functional.normalize(world, dim=-1)


def _accept_center_fit(
    fit: dict,
    *,
    proposed_shift: float,
    config: RayPoseConfig,
) -> tuple[bool, list[str]]:
    accepted = bool(fit["solver_accepted"])
    reasons = []
    if not accepted:
        reasons.append(str(fit["status"]))
    residual = float(fit["point_residual_rmse"])
    if not math.isfinite(residual):
        accepted = False
        reasons.append("non_finite_residual")
    elif residual > float(config.max_residual_rmse):
        accepted = False
        reasons.append("ray_residual_above_limit")
    if proposed_shift > float(config.max_center_shift):
        accepted = False
        reasons.append("center_shift_above_limit")
    return accepted, reasons


def _limit_points(points: torch.Tensor, maximum: int) -> torch.Tensor:
    if points.shape[0] <= int(maximum):
        return points
    keep = torch.linspace(
        0,
        points.shape[0] - 1,
        steps=int(maximum),
    ).round().long()
    return points.index_select(0, keep)


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
    masks = (
        batch.get("trusted_tracking_masks_stream")
        if bool(batch.get("strict_identity_gate", False))
        else None
    )
    if masks is None:
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


def _trusted_instance_valid(batch: dict) -> torch.Tensor:
    valid = batch.get("trusted_instance_valid")
    if valid is not None:
        valid = valid.detach().bool().cpu()
        if valid.ndim != 3 or valid.shape[0] != 1:
            raise ValueError(
                "trusted_instance_valid must have shape [1,S,K]."
            )
        return valid[0]
    masks = batch.get("tracking_masks_stream")
    if masks is None:
        return torch.zeros(
            len(batch["frame_indices"]),
            len(batch.get("instance_ids", ())),
            dtype=torch.bool,
        )
    masks = masks.detach().bool().cpu()
    return masks[0].flatten(start_dim=2).any(dim=-1)


def _instance_id_text(
    instance_ids: tuple[int, ...],
    valid: torch.Tensor,
    *,
    selected: bool,
) -> str:
    if len(instance_ids) != int(valid.numel()):
        return ""
    return " ".join(
        str(instance_id)
        for instance_id, keep in zip(instance_ids, valid.tolist())
        if bool(keep) is selected
    )


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
