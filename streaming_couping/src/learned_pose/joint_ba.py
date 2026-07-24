"""Gauge-fixed joint refinement of camera poses and ray-constrained pointmaps.

This is an inference-only, low-dimensional pointmap bundle-adjustment layer.
It never reads ground truth.  Raw and learned point-head predictions are first
converted to camera-local depth candidates.  Camera poses and a small number of
raw/learned depth mixture coefficients are then optimized against robust
cross-view ray and depth constraints.  The final world pointmap is generated
from the final pose and depth, so the two outputs cannot disagree by design.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class JointBAConfig:
    outer_iterations: int = 3
    inner_steps: int = 60
    learning_rate: float = 0.03
    match_radius_patches: int = 2
    max_matches_per_edge_region: int = 128
    min_matches_per_edge_region: int = 12
    min_total_matches: int = 32
    feature_dim_limit: int = 256
    min_feature_cosine: float = 0.10
    min_point_confidence: float = 0.30
    max_log_depth_residual: float = 0.45
    max_forward_backward_patches: float = 3.0
    ray_huber_delta: float = 0.03
    depth_huber_delta: float = 0.10
    depth_residual_weight: float = 0.10
    rotation_prior_weight: float = 0.25
    translation_prior_weight: float = 0.25
    beta_prior_weight: float = 0.02
    max_rotation_update_degrees: float = 3.0
    max_translation_scene_ratio: float = 0.25
    initial_beta: float = 0.05
    dcs_depth_sigma: float = 0.12
    minimum_instance_switch: float = 0.15


@dataclass(frozen=True)
class JointBAVariant:
    name: str
    beta_mode: str
    instance_switches: bool


@dataclass(frozen=True)
class JointBAResult:
    name: str
    pose_encoding: torch.Tensor
    world_points: torch.Tensor
    depth: torch.Tensor
    diagnostics: dict[str, object]


JOINT_BA_VARIANTS = (
    JointBAVariant("gauge_ba_pose_only", "fixed_raw", False),
    JointBAVariant("gauge_ba_global_beta", "global", False),
    JointBAVariant("gauge_ba_instance_switch", "region", True),
)


@dataclass(frozen=True)
class _Matches:
    source_frame: torch.Tensor
    target_frame: torch.Tensor
    source_patch: torch.Tensor
    target_patch: torch.Tensor
    region: torch.Tensor
    weight: torch.Tensor
    edge_region: tuple[tuple[int, int, int, float, int], ...]

    @property
    def count(self) -> int:
        return int(self.source_frame.numel())


def run_joint_ba(
    *,
    raw_pose_encoding: torch.Tensor,
    learned_pose_encoding: torch.Tensor,
    raw_world_points: torch.Tensor,
    learned_world_points: torch.Tensor,
    raw_confidence: torch.Tensor,
    learned_confidence: torch.Tensor,
    token_levels: torch.Tensor,
    patch_start_idx: int,
    patch_shape: tuple[int, int],
    tracking_masks: torch.Tensor,
    trusted_tracking_masks: torch.Tensor,
    trusted_instance_valid: torch.Tensor,
    image_size: tuple[int, int],
    reference_index: int,
    scene_scale: float,
    variant: JointBAVariant,
    config: JointBAConfig = JointBAConfig(),
) -> JointBAResult:
    """Run one no-training joint pose/pointmap refinement variant."""

    _validate_variant(variant)
    device = raw_world_points.device
    dtype = torch.float32
    raw_pose_encoding = _pose_batch(raw_pose_encoding).to(
        device=device,
        dtype=dtype,
    )
    learned_pose_encoding = _pose_batch(learned_pose_encoding).to(
        device=device,
        dtype=dtype,
    )
    raw_world_points = _single_batch(raw_world_points).to(device=device, dtype=dtype)
    learned_world_points = _single_batch(learned_world_points).to(
        device=device,
        dtype=dtype,
    )
    raw_confidence = _confidence(raw_confidence).to(device=device, dtype=dtype)
    learned_confidence = _confidence(learned_confidence).to(
        device=device,
        dtype=dtype,
    )
    tracking_masks = _single_batch(tracking_masks).to(device=device).bool()
    trusted_tracking_masks = _single_batch(trusted_tracking_masks).to(
        device=device
    ).bool()
    trusted_instance_valid = _single_batch(trusted_instance_valid).to(
        device=device
    ).bool()

    from streamvggt.utils.pose_enc import (
        extri_intri_to_pose_encoding,
        pose_encoding_to_extri_intri,
    )

    raw_w2c, raw_intrinsics = pose_encoding_to_extri_intri(
        raw_pose_encoding,
        image_size_hw=image_size,
    )
    learned_w2c, _ = pose_encoding_to_extri_intri(
        learned_pose_encoding,
        image_size_hw=image_size,
    )
    raw_c2w = _w2c_to_c2w(raw_w2c[0])
    learned_c2w = _w2c_to_c2w(learned_w2c[0])
    initial_c2w = regauge_camera_to_world(
        learned_c2w,
        raw_c2w,
        reference_index=reference_index,
    )
    sequence = int(raw_c2w.shape[0])
    if raw_world_points.shape[:3] != learned_world_points.shape[:3]:
        raise ValueError("Raw and learned pointmaps must share [S,H,W].")
    if raw_world_points.shape[0] != sequence:
        raise ValueError("Pose and pointmap sequence lengths disagree.")
    height, width = (int(value) for value in raw_world_points.shape[1:3])
    if (height, width) != tuple(int(value) for value in image_size):
        raise ValueError(
            f"Pointmap/image size mismatch: {(height, width)} vs {image_size}."
        )

    # Use one calibrated camera model throughout the clip, matching the
    # existing stabilized ray solver and the fixed physical camera.
    intrinsics = (
        raw_intrinsics[0, int(reference_index)][None]
        .expand(sequence, -1, -1)
        .clone()
    )
    raw_depth = _world_to_camera_depth(raw_world_points, raw_c2w)
    learned_depth = _world_to_camera_depth(
        learned_world_points,
        learned_c2w,
    )
    learned_valid = (
        torch.isfinite(learned_depth)
        & (learned_depth > 1e-6)
        & torch.isfinite(learned_world_points).all(dim=-1)
    )
    raw_valid = (
        torch.isfinite(raw_depth)
        & (raw_depth > 1e-6)
        & torch.isfinite(raw_world_points).all(dim=-1)
    )
    learned_depth = torch.where(learned_valid, learned_depth, raw_depth)
    learned_confidence = torch.where(
        learned_valid,
        learned_confidence,
        torch.zeros_like(learned_confidence),
    )

    full_labels = _region_labels(
        tracking_masks,
        trusted_tracking_masks,
        trusted_instance_valid,
    )
    patch_features = _patch_features(
        token_levels,
        sequence=sequence,
        patch_start_idx=int(patch_start_idx),
        patch_shape=patch_shape,
        feature_dim_limit=int(config.feature_dim_limit),
        device=device,
    )
    patch_y, patch_x = _patch_sample_indices(
        image_size=image_size,
        patch_shape=patch_shape,
        device=device,
    )
    raw_depth_patch = raw_depth[:, patch_y, patch_x]
    learned_depth_patch = learned_depth[:, patch_y, patch_x]
    labels_patch = full_labels[:, patch_y, patch_x]
    raw_conf_patch = raw_confidence[:, patch_y, patch_x]
    learned_conf_patch = learned_confidence[:, patch_y, patch_x]
    confidence_patch = torch.sqrt(
        raw_conf_patch.clamp(0.0, 1.0)
        * torch.where(
            learned_conf_patch > 0,
            learned_conf_patch.clamp(0.0, 1.0),
            raw_conf_patch.clamp(0.0, 1.0),
        )
    )
    patch_uv = _patch_centers(
        image_size=image_size,
        patch_shape=patch_shape,
        device=device,
        dtype=dtype,
    )
    rays_patch = _pixel_rays(
        patch_uv,
        intrinsics,
    )
    region_count = int(trusted_tracking_masks.shape[1]) + 1
    scene_scale = max(float(scene_scale), 1e-6)
    reference_mask = torch.ones(sequence, 1, device=device, dtype=dtype)
    reference_mask[int(reference_index)] = 0.0

    rotation_parameter = torch.nn.Parameter(
        torch.zeros(sequence, 3, device=device, dtype=dtype)
    )
    translation_parameter = torch.nn.Parameter(
        torch.zeros(sequence, 3, device=device, dtype=dtype)
    )
    beta_parameter: torch.nn.Parameter | None = None
    if variant.beta_mode in {"global", "region"}:
        columns = 1 if variant.beta_mode == "global" else region_count
        initial_beta = min(max(float(config.initial_beta), 1e-4), 1.0 - 1e-4)
        initial_logit = math.log(initial_beta / (1.0 - initial_beta))
        beta_parameter = torch.nn.Parameter(
            torch.full(
                (sequence, columns),
                initial_logit,
                device=device,
                dtype=dtype,
            )
        )
    parameters = [rotation_parameter, translation_parameter]
    if beta_parameter is not None:
        parameters.append(beta_parameter)
    optimizer = torch.optim.Adam(parameters, lr=float(config.learning_rate))

    optimized_once = False
    first_ray_rmse = float("nan")
    first_depth_rmse = float("nan")
    last_matches: _Matches | None = None
    best_loss = float("inf")
    best_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None = None

    for _ in range(int(config.outer_iterations)):
        with torch.no_grad():
            c2w = _compose_c2w(
                initial_c2w,
                rotation_parameter,
                translation_parameter,
                reference_mask=reference_mask,
                scene_scale=scene_scale,
                config=config,
            )
            beta = _beta_values(
                beta_parameter,
                variant=variant,
                sequence=sequence,
                region_count=region_count,
                reference_mask=reference_mask,
                device=device,
                dtype=dtype,
            )
            depth_patch = _blend_depth(
                raw_depth_patch,
                learned_depth_patch,
                labels_patch,
                beta,
            )
            matches = _build_matches(
                c2w=c2w,
                depth_patch=depth_patch,
                confidence_patch=confidence_patch,
                labels_patch=labels_patch,
                features=patch_features,
                patch_uv=patch_uv,
                rays_patch=rays_patch,
                intrinsics=intrinsics,
                image_size=image_size,
                patch_shape=patch_shape,
                reference_index=int(reference_index),
                instance_switches=bool(variant.instance_switches),
                config=config,
            )
        if matches.count < int(config.min_total_matches):
            break
        last_matches = matches
        if not optimized_once:
            first_ray_rmse, first_depth_rmse = _residual_statistics(
                matches,
                c2w=c2w,
                beta=beta,
                raw_depth_patch=raw_depth_patch,
                learned_depth_patch=learned_depth_patch,
                rays_patch=rays_patch,
            )
        optimized_once = True
        stale = 0
        for _ in range(int(config.inner_steps)):
            optimizer.zero_grad(set_to_none=True)
            c2w = _compose_c2w(
                initial_c2w,
                rotation_parameter,
                translation_parameter,
                reference_mask=reference_mask,
                scene_scale=scene_scale,
                config=config,
            )
            beta = _beta_values(
                beta_parameter,
                variant=variant,
                sequence=sequence,
                region_count=region_count,
                reference_mask=reference_mask,
                device=device,
                dtype=dtype,
            )
            loss = _joint_objective(
                matches,
                c2w=c2w,
                beta=beta,
                raw_depth_patch=raw_depth_patch,
                learned_depth_patch=learned_depth_patch,
                rays_patch=rays_patch,
                rotation_parameter=rotation_parameter,
                translation_parameter=translation_parameter,
                config=config,
            )
            if not bool(torch.isfinite(loss)):
                break
            current = float(loss.detach())
            if current + 1e-8 < best_loss:
                best_loss = current
                best_state = (
                    rotation_parameter.detach().clone(),
                    translation_parameter.detach().clone(),
                    (
                        beta_parameter.detach().clone()
                        if beta_parameter is not None
                        else None
                    ),
                )
                stale = 0
            else:
                stale += 1
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
            if stale >= 12:
                break

    if not optimized_once or best_state is None:
        final_c2w = raw_c2w
        final_beta = torch.zeros(
            sequence,
            region_count,
            device=device,
            dtype=dtype,
        )
        status = "fallback_insufficient_cross_view_matches"
        best_loss = float("nan")
        used_exact_raw_fallback = True
    else:
        with torch.no_grad():
            rotation_parameter.copy_(best_state[0])
            translation_parameter.copy_(best_state[1])
            if beta_parameter is not None and best_state[2] is not None:
                beta_parameter.copy_(best_state[2])
            final_c2w = _compose_c2w(
                initial_c2w,
                rotation_parameter,
                translation_parameter,
                reference_mask=reference_mask,
                scene_scale=scene_scale,
                config=config,
            )
            final_beta = _beta_values(
                beta_parameter,
                variant=variant,
                sequence=sequence,
                region_count=region_count,
                reference_mask=reference_mask,
                device=device,
                dtype=dtype,
            )
        status = "accepted_joint_ba"
        used_exact_raw_fallback = False

    with torch.no_grad():
        if used_exact_raw_fallback:
            final_depth = raw_depth
            final_points = raw_world_points
            final_pose = raw_pose_encoding
        else:
            final_depth = _blend_depth(
                raw_depth,
                learned_depth,
                full_labels,
                final_beta,
            )
            final_depth = torch.where(raw_valid, final_depth, torch.nan)
            final_points = world_points_from_depth(
                final_depth,
                final_c2w,
                intrinsics,
                image_size=image_size,
            )
            final_w2c = _c2w_to_w2c(final_c2w)
            final_pose = extri_intri_to_pose_encoding(
                final_w2c[None],
                intrinsics[None],
                image_size_hw=image_size,
            )
            # Matrix-to-quaternion conversion is numerically equivalent but
            # not necessarily bit-identical (the quaternion sign is also
            # ambiguous).  Preserve the raw reference encoding explicitly so
            # the exported anchor is exact in both geometry and serialization.
            final_pose = final_pose.clone()
            final_pose[:, int(reference_index)] = raw_pose_encoding[
                :, int(reference_index)
            ]
        if last_matches is not None and optimized_once:
            final_ray_rmse, final_depth_rmse = _residual_statistics(
                last_matches,
                c2w=final_c2w,
                beta=final_beta,
                raw_depth_patch=raw_depth_patch,
                learned_depth_patch=learned_depth_patch,
                rays_patch=rays_patch,
            )
        else:
            final_ray_rmse = float("nan")
            final_depth_rmse = float("nan")
        edge_rows = last_matches.edge_region if last_matches is not None else ()
        active_instance_edges = sum(
            int(region > 0 and switch >= config.minimum_instance_switch)
            for _, _, region, switch, _ in edge_rows
        )
        rejected_instance_edges = sum(
            int(region > 0 and switch < config.minimum_instance_switch)
            for _, _, region, switch, _ in edge_rows
        )
        nonreference = torch.arange(sequence, device=device) != int(reference_index)
        beta_mean = (
            float(final_beta[nonreference].mean())
            if bool(nonreference.any())
            else 0.0
        )
        reference_error = float(
            (final_c2w[int(reference_index)] - raw_c2w[int(reference_index)])
            .abs()
            .max()
        )
        center_shift = torch.linalg.vector_norm(
            final_c2w[:, :3, 3] - initial_c2w[:, :3, 3],
            dim=-1,
        )
        diagnostics = {
            "status": status,
            "matches": last_matches.count if last_matches is not None else 0,
            "edge_regions": len(edge_rows),
            "active_instance_edges": active_instance_edges,
            "rejected_instance_edges": rejected_instance_edges,
            "initial_ray_rmse": first_ray_rmse,
            "final_ray_rmse": final_ray_rmse,
            "initial_log_depth_rmse": first_depth_rmse,
            "final_log_depth_rmse": final_depth_rmse,
            "beta_mean": beta_mean,
            "mean_pose_center_shift_native": float(center_shift.mean()),
            "reference_pose_max_abs_diff": reference_error,
            "objective": best_loss,
        }
    return JointBAResult(
        name=variant.name,
        pose_encoding=final_pose,
        world_points=final_points[None],
        depth=final_depth[None],
        diagnostics=diagnostics,
    )


def regauge_camera_to_world(
    learned_c2w: torch.Tensor,
    raw_c2w: torch.Tensor,
    *,
    reference_index: int,
) -> torch.Tensor:
    """Move learned relative poses into the exact raw reference gauge."""

    reference_index = int(reference_index)
    transform = raw_c2w[reference_index] @ torch.linalg.inv(
        learned_c2w[reference_index]
    )
    return transform[None] @ learned_c2w


def world_points_from_depth(
    depth: torch.Tensor,
    camera_to_world: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    image_size: tuple[int, int],
) -> torch.Tensor:
    """Backproject depth with the same poses used for the final camera output."""

    sequence, height, width = depth.shape
    if (height, width) != tuple(int(value) for value in image_size):
        raise ValueError("Depth and image size disagree.")
    y, x = torch.meshgrid(
        torch.arange(height, device=depth.device, dtype=depth.dtype),
        torch.arange(width, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    fx = intrinsics[:, 0, 0, None, None]
    fy = intrinsics[:, 1, 1, None, None]
    cx = intrinsics[:, 0, 2, None, None]
    cy = intrinsics[:, 1, 2, None, None]
    local = torch.stack(
        [
            (x[None] - cx) * depth / fx.clamp_min(1e-8),
            (y[None] - cy) * depth / fy.clamp_min(1e-8),
            depth,
        ],
        dim=-1,
    )
    rotation = camera_to_world[:, :3, :3]
    center = camera_to_world[:, :3, 3]
    return (
        torch.einsum("sij,shwj->shwi", rotation, local)
        + center[:, None, None, :]
    )


def dcs_switch(
    normalized_error_squared: torch.Tensor | float,
    *,
    minimum: float = 0.0,
) -> torch.Tensor:
    """Dynamic-covariance-style weight for an edge-level robust factor."""

    value = torch.as_tensor(normalized_error_squared)
    weight = torch.minimum(
        torch.ones_like(value),
        2.0 / (1.0 + value.clamp_min(0.0)),
    )
    return weight.clamp_min(float(minimum))


def _validate_variant(variant: JointBAVariant) -> None:
    if variant.beta_mode not in {"fixed_raw", "global", "region"}:
        raise ValueError(f"Unknown joint BA beta mode: {variant.beta_mode!r}.")


def _single_batch(value: torch.Tensor) -> torch.Tensor:
    if value.ndim >= 1 and value.shape[0] == 1:
        return value[0]
    return value


def _pose_batch(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 2:
        return value[None]
    if value.ndim != 3 or value.shape[0] != 1:
        raise ValueError(
            f"Pose encoding must be [S,D] or [1,S,D], got {tuple(value.shape)}."
        )
    return value


def _confidence(value: torch.Tensor) -> torch.Tensor:
    value = _single_batch(value)
    if value.ndim == 4 and value.shape[-1] == 1:
        value = value[..., 0]
    if value.ndim != 3:
        raise ValueError(f"Confidence must be [S,H,W], got {tuple(value.shape)}.")
    return value


def _w2c_to_c2w(extrinsics: torch.Tensor) -> torch.Tensor:
    rotation = extrinsics[:, :3, :3]
    translation = extrinsics[:, :3, 3]
    c2w_rotation = rotation.transpose(-1, -2)
    center = -torch.einsum("sij,sj->si", c2w_rotation, translation)
    bottom = torch.zeros(
        extrinsics.shape[0],
        1,
        4,
        device=extrinsics.device,
        dtype=extrinsics.dtype,
    )
    bottom[:, 0, 3] = 1.0
    return torch.cat(
        [torch.cat([c2w_rotation, center[..., None]], dim=-1), bottom],
        dim=-2,
    )


def _c2w_to_w2c(camera_to_world: torch.Tensor) -> torch.Tensor:
    rotation = camera_to_world[:, :3, :3].transpose(-1, -2)
    center = camera_to_world[:, :3, 3]
    translation = -torch.einsum("sij,sj->si", rotation, center)
    return torch.cat([rotation, translation[..., None]], dim=-1)


def _world_to_camera_depth(
    world_points: torch.Tensor,
    camera_to_world: torch.Tensor,
) -> torch.Tensor:
    rotation = camera_to_world[:, :3, :3]
    center = camera_to_world[:, :3, 3]
    local = torch.einsum(
        "sji,shwj->shwi",
        rotation,
        world_points - center[:, None, None, :],
    )
    return local[..., 2]


def _region_labels(
    tracking_masks: torch.Tensor,
    trusted_masks: torch.Tensor,
    trusted_valid: torch.Tensor,
) -> torch.Tensor:
    if tracking_masks.shape != trusted_masks.shape:
        raise ValueError("Raw and trusted tracking masks must have equal shape.")
    sequence, instances, height, width = tracking_masks.shape
    trusted_masks = trusted_masks & trusted_valid[:, :, None, None]
    labels = torch.zeros(
        sequence,
        height,
        width,
        device=tracking_masks.device,
        dtype=torch.long,
    )
    overlap = trusted_masks.sum(dim=1) > 1
    for slot in range(instances):
        labels = torch.where(
            trusted_masks[:, slot],
            torch.full_like(labels, slot + 1),
            labels,
        )
    # Pixels covered only by an untrusted/ambiguous instance must not silently
    # become background constraints.
    raw_union = tracking_masks.any(dim=1)
    trusted_union = trusted_masks.any(dim=1)
    labels = torch.where(raw_union & ~trusted_union, -torch.ones_like(labels), labels)
    return torch.where(overlap, -torch.ones_like(labels), labels)


def _patch_features(
    token_levels: torch.Tensor,
    *,
    sequence: int,
    patch_start_idx: int,
    patch_shape: tuple[int, int],
    feature_dim_limit: int,
    device: torch.device,
) -> torch.Tensor:
    value = token_levels
    if value.ndim == 5 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 4:
        raise ValueError(
            f"token_levels must be [L,S,N,C], got {tuple(value.shape)}."
        )
    value = value[-1, :, int(patch_start_idx) :]
    patches = int(patch_shape[0]) * int(patch_shape[1])
    if value.shape[:2] != (sequence, patches):
        raise ValueError(
            "Cached patch tokens disagree with patch_shape: "
            f"{tuple(value.shape[:2])} vs {(sequence, patches)}."
        )
    if value.shape[-1] > int(feature_dim_limit):
        indices = torch.linspace(
            0,
            value.shape[-1] - 1,
            int(feature_dim_limit),
            device=value.device,
        ).round().long()
        value = value.index_select(-1, indices)
    return F.normalize(value.to(device=device, dtype=torch.float32), dim=-1)


def _patch_sample_indices(
    *,
    image_size: tuple[int, int],
    patch_shape: tuple[int, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = (int(value) for value in image_size)
    patch_h, patch_w = (int(value) for value in patch_shape)
    y = (
        (torch.arange(patch_h, device=device) + 0.5) * height / patch_h - 0.5
    ).round().long().clamp(0, height - 1)
    x = (
        (torch.arange(patch_w, device=device) + 0.5) * width / patch_w - 0.5
    ).round().long().clamp(0, width - 1)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return yy.reshape(-1), xx.reshape(-1)


def _patch_centers(
    *,
    image_size: tuple[int, int],
    patch_shape: tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    height, width = (int(value) for value in image_size)
    patch_h, patch_w = (int(value) for value in patch_shape)
    y = (torch.arange(patch_h, device=device, dtype=dtype) + 0.5) * (
        height / patch_h
    ) - 0.5
    x = (torch.arange(patch_w, device=device, dtype=dtype) + 0.5) * (
        width / patch_w
    ) - 0.5
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)


def _pixel_rays(
    uv: torch.Tensor,
    intrinsics: torch.Tensor,
) -> torch.Tensor:
    sequence = intrinsics.shape[0]
    u = uv[:, 0][None].expand(sequence, -1)
    v = uv[:, 1][None].expand(sequence, -1)
    fx = intrinsics[:, 0, 0, None]
    fy = intrinsics[:, 1, 1, None]
    cx = intrinsics[:, 0, 2, None]
    cy = intrinsics[:, 1, 2, None]
    return torch.stack(
        [
            (u - cx) / fx.clamp_min(1e-8),
            (v - cy) / fy.clamp_min(1e-8),
            torch.ones_like(u),
        ],
        dim=-1,
    )


def _beta_values(
    parameter: torch.Tensor | None,
    *,
    variant: JointBAVariant,
    sequence: int,
    region_count: int,
    reference_mask: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if parameter is None or variant.beta_mode == "fixed_raw":
        return torch.zeros(sequence, region_count, device=device, dtype=dtype)
    beta = torch.sigmoid(parameter) * reference_mask
    if variant.beta_mode == "global":
        beta = beta.expand(-1, region_count)
    return beta


def _blend_depth(
    raw_depth: torch.Tensor,
    learned_depth: torch.Tensor,
    labels: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    safe_labels = labels.clamp_min(0)
    frame = torch.arange(
        raw_depth.shape[0],
        device=raw_depth.device,
    ).reshape((-1,) + (1,) * (raw_depth.ndim - 1))
    frame = frame.expand_as(safe_labels)
    selected_beta = beta[frame, safe_labels]
    selected_beta = torch.where(labels >= 0, selected_beta, torch.zeros_like(selected_beta))
    raw_log = raw_depth.clamp_min(1e-6).log()
    learned_log = learned_depth.clamp_min(1e-6).log()
    return torch.exp(raw_log + selected_beta * (learned_log - raw_log))


def _compose_c2w(
    initial: torch.Tensor,
    rotation_parameter: torch.Tensor,
    translation_parameter: torch.Tensor,
    *,
    reference_mask: torch.Tensor,
    scene_scale: float,
    config: JointBAConfig,
) -> torch.Tensor:
    max_rotation = math.radians(float(config.max_rotation_update_degrees))
    omega = max_rotation * torch.tanh(rotation_parameter) * reference_mask
    rotation = _so3_exp(omega) @ initial[:, :3, :3]
    max_translation = float(config.max_translation_scene_ratio) * float(scene_scale)
    center = initial[:, :3, 3] + (
        max_translation * torch.tanh(translation_parameter) * reference_mask
    )
    bottom = torch.zeros(
        initial.shape[0],
        1,
        4,
        device=initial.device,
        dtype=initial.dtype,
    )
    bottom[:, 0, 3] = 1.0
    return torch.cat(
        [torch.cat([rotation, center[..., None]], dim=-1), bottom],
        dim=-2,
    )


def _so3_exp(omega: torch.Tensor) -> torch.Tensor:
    theta2 = omega.square().sum(dim=-1, keepdim=True)
    theta = torch.sqrt(theta2 + 1e-12)
    # sinc-based coefficients avoid the inactive-branch NaN gradients that
    # torch.where(sin(theta) / theta, Taylor) can create at zero rotation.
    a = torch.sinc(theta / math.pi)
    b = 0.5 * torch.sinc(theta / (2.0 * math.pi)).square()
    x, y, z = omega.unbind(dim=-1)
    zero = torch.zeros_like(x)
    skew = torch.stack(
        [zero, -z, y, z, zero, -x, -y, x, zero],
        dim=-1,
    ).reshape(*omega.shape[:-1], 3, 3)
    identity = torch.eye(3, device=omega.device, dtype=omega.dtype).expand_as(skew)
    return identity + a[..., None] * skew + b[..., None] * (skew @ skew)


@torch.no_grad()
def _build_matches(
    *,
    c2w: torch.Tensor,
    depth_patch: torch.Tensor,
    confidence_patch: torch.Tensor,
    labels_patch: torch.Tensor,
    features: torch.Tensor,
    patch_uv: torch.Tensor,
    rays_patch: torch.Tensor,
    intrinsics: torch.Tensor,
    image_size: tuple[int, int],
    patch_shape: tuple[int, int],
    reference_index: int,
    instance_switches: bool,
    config: JointBAConfig,
) -> _Matches:
    sequence, patches = depth_patch.shape
    patch_h, patch_w = (int(value) for value in patch_shape)
    edges = {(index, index + 1) for index in range(sequence - 1)}
    edges.update(
        (min(reference_index, index), max(reference_index, index))
        for index in range(sequence)
        if index != reference_index
    )
    offsets = torch.stack(
        torch.meshgrid(
            torch.arange(
                -int(config.match_radius_patches),
                int(config.match_radius_patches) + 1,
                device=c2w.device,
            ),
            torch.arange(
                -int(config.match_radius_patches),
                int(config.match_radius_patches) + 1,
                device=c2w.device,
            ),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 2)
    source_frames = []
    target_frames = []
    source_patches = []
    target_patches = []
    regions = []
    weights = []
    edge_rows: list[tuple[int, int, int, float, int]] = []
    region_count = int(labels_patch.max().clamp_min(0)) + 1
    height, width = (int(value) for value in image_size)

    for source, target in sorted(edges):
        for region in range(region_count):
            valid_source = (
                (labels_patch[source] == region)
                & torch.isfinite(depth_patch[source])
                & (depth_patch[source] > 1e-6)
                & (
                    confidence_patch[source]
                    >= float(config.min_point_confidence)
                )
            )
            source_index = valid_source.nonzero(as_tuple=False).flatten()
            source_index = _deterministic_limit(
                source_index,
                int(config.max_matches_per_edge_region),
            )
            if source_index.numel() == 0:
                continue
            local = (
                rays_patch[source, source_index]
                * depth_patch[source, source_index, None]
            )
            world = (
                torch.einsum("ij,nj->ni", c2w[source, :3, :3], local)
                + c2w[source, :3, 3]
            )
            target_local = torch.einsum(
                "ji,nj->ni",
                c2w[target, :3, :3],
                world - c2w[target, :3, 3],
            )
            positive = target_local[:, 2] > 1e-6
            projected_u = (
                intrinsics[target, 0, 0]
                * target_local[:, 0]
                / target_local[:, 2].clamp_min(1e-6)
                + intrinsics[target, 0, 2]
            )
            projected_v = (
                intrinsics[target, 1, 1]
                * target_local[:, 1]
                / target_local[:, 2].clamp_min(1e-6)
                + intrinsics[target, 1, 2]
            )
            projected_x = (projected_u + 0.5) * patch_w / width - 0.5
            projected_y = (projected_v + 0.5) * patch_h / height - 0.5
            base = torch.stack(
                [projected_y.round().long(), projected_x.round().long()],
                dim=-1,
            )
            candidates = base[:, None, :] + offsets[None]
            candidate_valid = (
                positive[:, None]
                & (candidates[..., 0] >= 0)
                & (candidates[..., 0] < patch_h)
                & (candidates[..., 1] >= 0)
                & (candidates[..., 1] < patch_w)
            )
            candidate_flat = (
                candidates[..., 0].clamp(0, patch_h - 1) * patch_w
                + candidates[..., 1].clamp(0, patch_w - 1)
            )
            candidate_valid &= labels_patch[target, candidate_flat] == region
            candidate_valid &= torch.isfinite(depth_patch[target, candidate_flat])
            candidate_valid &= depth_patch[target, candidate_flat] > 1e-6
            candidate_valid &= (
                confidence_patch[target, candidate_flat]
                >= float(config.min_point_confidence)
            )
            similarity = (
                features[source, source_index, None]
                * features[target, candidate_flat]
            ).sum(dim=-1)
            similarity = torch.where(
                candidate_valid,
                similarity,
                torch.full_like(similarity, -torch.inf),
            )
            best_similarity, best_slot = similarity.max(dim=1)
            selected = candidate_flat.gather(1, best_slot[:, None])[:, 0]
            keep = (
                torch.isfinite(best_similarity)
                & (
                    best_similarity
                    >= float(config.min_feature_cosine)
                )
            )
            if not bool(keep.any()):
                continue
            source_kept = source_index[keep]
            target_kept = selected[keep]
            similarity_kept = best_similarity[keep]
            target_local_kept = target_local[keep]
            target_depth = depth_patch[target, target_kept]
            log_depth_error = (
                target_local_kept[:, 2].clamp_min(1e-6).log()
                - target_depth.clamp_min(1e-6).log()
            ).abs()

            # Forward/backward projection check using the target depth.
            target_world = (
                torch.einsum(
                    "ij,nj->ni",
                    c2w[target, :3, :3],
                    rays_patch[target, target_kept] * target_depth[:, None],
                )
                + c2w[target, :3, 3]
            )
            source_back = torch.einsum(
                "ji,nj->ni",
                c2w[source, :3, :3],
                target_world - c2w[source, :3, 3],
            )
            back_u = (
                intrinsics[source, 0, 0]
                * source_back[:, 0]
                / source_back[:, 2].clamp_min(1e-6)
                + intrinsics[source, 0, 2]
            )
            back_v = (
                intrinsics[source, 1, 1]
                * source_back[:, 1]
                / source_back[:, 2].clamp_min(1e-6)
                + intrinsics[source, 1, 2]
            )
            back_x = (back_u + 0.5) * patch_w / width - 0.5
            back_y = (back_v + 0.5) * patch_h / height - 0.5
            source_x = (source_kept % patch_w).float()
            source_y = torch.div(
                source_kept,
                patch_w,
                rounding_mode="floor",
            ).float()
            fb_error = torch.sqrt(
                (back_x - source_x).square()
                + (back_y - source_y).square()
            )
            keep_geometry = (
                (source_back[:, 2] > 1e-6)
                & (
                    log_depth_error
                    <= float(config.max_log_depth_residual)
                )
                & (
                    fb_error
                    <= float(config.max_forward_backward_patches)
                )
            )
            source_kept = source_kept[keep_geometry]
            target_kept = target_kept[keep_geometry]
            similarity_kept = similarity_kept[keep_geometry]
            log_depth_error = log_depth_error[keep_geometry]
            if (
                source_kept.numel()
                < int(config.min_matches_per_edge_region)
            ):
                continue
            if region > 0 and instance_switches:
                normalized = (
                    log_depth_error.median()
                    / max(float(config.dcs_depth_sigma), 1e-6)
                ).square()
                switch = float(dcs_switch(normalized))
            else:
                switch = 1.0
            confidence_weight = torch.sqrt(
                confidence_patch[source, source_kept]
                * confidence_patch[target, target_kept]
            ).clamp(0.0, 1.0)
            feature_weight = (
                (similarity_kept - float(config.min_feature_cosine))
                / max(1.0 - float(config.min_feature_cosine), 1e-6)
            ).clamp(0.0, 1.0)
            current_weight = confidence_weight * feature_weight * switch
            source_frames.append(
                torch.full_like(source_kept, source)
            )
            target_frames.append(
                torch.full_like(target_kept, target)
            )
            source_patches.append(source_kept)
            target_patches.append(target_kept)
            regions.append(torch.full_like(source_kept, region))
            weights.append(current_weight)
            edge_rows.append(
                (
                    source,
                    target,
                    region,
                    switch,
                    int(source_kept.numel()),
                )
            )
    if not source_frames:
        empty = torch.empty(0, device=c2w.device, dtype=torch.long)
        return _Matches(
            empty,
            empty,
            empty,
            empty,
            empty,
            empty.float(),
            tuple(edge_rows),
        )
    return _Matches(
        torch.cat(source_frames),
        torch.cat(target_frames),
        torch.cat(source_patches),
        torch.cat(target_patches),
        torch.cat(regions),
        torch.cat(weights),
        tuple(edge_rows),
    )


def _joint_objective(
    matches: _Matches,
    *,
    c2w: torch.Tensor,
    beta: torch.Tensor,
    raw_depth_patch: torch.Tensor,
    learned_depth_patch: torch.Tensor,
    rays_patch: torch.Tensor,
    rotation_parameter: torch.Tensor,
    translation_parameter: torch.Tensor,
    config: JointBAConfig,
) -> torch.Tensor:
    ray_forward, ray_backward, depth_forward, depth_backward = _match_residuals(
        matches,
        c2w=c2w,
        beta=beta,
        raw_depth_patch=raw_depth_patch,
        learned_depth_patch=learned_depth_patch,
        rays_patch=rays_patch,
    )
    ray_loss = 0.5 * (
        _huber(ray_forward, float(config.ray_huber_delta))
        + _huber(ray_backward, float(config.ray_huber_delta))
    )
    depth_loss = 0.5 * (
        _huber(depth_forward, float(config.depth_huber_delta))
        + _huber(depth_backward, float(config.depth_huber_delta))
    )
    weight = matches.weight.clamp_min(0.0)
    data = (
        weight
        * (
            ray_loss
            + float(config.depth_residual_weight) * depth_loss
        )
    ).sum() / weight.sum().clamp_min(1e-6)
    pose_prior = (
        float(config.rotation_prior_weight)
        * rotation_parameter.square().mean()
        + float(config.translation_prior_weight)
        * translation_parameter.square().mean()
    )
    beta_prior = float(config.beta_prior_weight) * beta.square().mean()
    return data + pose_prior + beta_prior


def _match_residuals(
    matches: _Matches,
    *,
    c2w: torch.Tensor,
    beta: torch.Tensor,
    raw_depth_patch: torch.Tensor,
    learned_depth_patch: torch.Tensor,
    rays_patch: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    source = matches.source_frame
    target = matches.target_frame
    region = matches.region
    source_patch = matches.source_patch
    target_patch = matches.target_patch
    source_beta = beta[source, region]
    target_beta = beta[target, region]
    source_depth = torch.exp(
        raw_depth_patch[source, source_patch].clamp_min(1e-6).log()
        + source_beta
        * (
            learned_depth_patch[source, source_patch].clamp_min(1e-6).log()
            - raw_depth_patch[source, source_patch].clamp_min(1e-6).log()
        )
    )
    target_depth = torch.exp(
        raw_depth_patch[target, target_patch].clamp_min(1e-6).log()
        + target_beta
        * (
            learned_depth_patch[target, target_patch].clamp_min(1e-6).log()
            - raw_depth_patch[target, target_patch].clamp_min(1e-6).log()
        )
    )
    source_local = rays_patch[source, source_patch] * source_depth[:, None]
    target_local = rays_patch[target, target_patch] * target_depth[:, None]
    source_world = (
        torch.bmm(
            c2w[source, :3, :3],
            source_local[..., None],
        )[..., 0]
        + c2w[source, :3, 3]
    )
    target_world = (
        torch.bmm(
            c2w[target, :3, :3],
            target_local[..., None],
        )[..., 0]
        + c2w[target, :3, 3]
    )
    source_in_target = torch.bmm(
        c2w[target, :3, :3].transpose(-1, -2),
        (source_world - c2w[target, :3, 3])[..., None],
    )[..., 0]
    target_in_source = torch.bmm(
        c2w[source, :3, :3].transpose(-1, -2),
        (target_world - c2w[source, :3, 3])[..., None],
    )[..., 0]
    target_ray = F.normalize(rays_patch[target, target_patch], dim=-1)
    source_ray = F.normalize(rays_patch[source, source_patch], dim=-1)
    ray_forward = torch.linalg.vector_norm(
        F.normalize(source_in_target, dim=-1) - target_ray,
        dim=-1,
    )
    ray_backward = torch.linalg.vector_norm(
        F.normalize(target_in_source, dim=-1) - source_ray,
        dim=-1,
    )
    depth_forward = (
        source_in_target[:, 2].clamp_min(1e-6).log()
        - target_depth.clamp_min(1e-6).log()
    ).abs()
    depth_backward = (
        target_in_source[:, 2].clamp_min(1e-6).log()
        - source_depth.clamp_min(1e-6).log()
    ).abs()
    return ray_forward, ray_backward, depth_forward, depth_backward


def _residual_statistics(
    matches: _Matches,
    *,
    c2w: torch.Tensor,
    beta: torch.Tensor,
    raw_depth_patch: torch.Tensor,
    learned_depth_patch: torch.Tensor,
    rays_patch: torch.Tensor,
) -> tuple[float, float]:
    values = _match_residuals(
        matches,
        c2w=c2w,
        beta=beta,
        raw_depth_patch=raw_depth_patch,
        learned_depth_patch=learned_depth_patch,
        rays_patch=rays_patch,
    )
    ray = torch.cat(values[:2])
    depth = torch.cat(values[2:])
    return (
        float(torch.sqrt(ray.square().mean())),
        float(torch.sqrt(depth.square().mean())),
    )


def _huber(value: torch.Tensor, delta: float) -> torch.Tensor:
    absolute = value.abs()
    delta = max(float(delta), 1e-8)
    return torch.where(
        absolute <= delta,
        0.5 * absolute.square(),
        delta * (absolute - 0.5 * delta),
    )


def _deterministic_limit(
    indices: torch.Tensor,
    maximum: int,
) -> torch.Tensor:
    if indices.numel() <= int(maximum):
        return indices
    selected = torch.linspace(
        0,
        indices.numel() - 1,
        int(maximum),
        device=indices.device,
    ).round().long()
    return indices.index_select(0, selected)
