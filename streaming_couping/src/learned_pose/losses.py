"""Losses for camera-only and all-token persistent-instance fusion."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import GEOMETRY_MODES, POSE_MODES, LossConfig


def compute_training_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict,
    config: LossConfig,
    *,
    mode: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    predicted_pose = outputs["pose_encoding"].float()
    target_pose = batch["target_pose_encoding"].float()
    zero = predicted_pose.new_zeros(())
    camera = zero
    relative_rotation = zero
    translation_direction = zero
    rigid = zero
    centroid = zero
    if mode in POSE_MODES:
        camera = camera_encoding_loss(predicted_pose, target_pose)
        relative_rotation = relative_rotation_loss(predicted_pose, target_pose)
        translation_direction = translation_direction_loss(predicted_pose, target_pose)
        rigid, centroid = instance_rigid_losses(
            predicted_pose,
            batch["instance_uvd"].float(),
            batch["instance_uvd_valid"].bool(),
            batch["instance_rigid_weight"].float(),
            image_size=tuple(int(v) for v in batch["image_size"]),
            scene_scale=float(batch["scene_scale"]),
            trim_quantile=config.rigid_trim_quantile,
        )
    residual = outputs.get(
        "residual_mean_square",
        predicted_pose.new_zeros(()),
    ).float()
    depth = zero
    depth_fixed = zero
    pointmap = zero
    if mode in GEOMETRY_MODES:
        if "depth" not in outputs or "world_points" not in outputs:
            raise RuntimeError(f"{mode} must return depth and world_points.")
        depth = scale_invariant_depth_loss(
            outputs["depth"].float(),
            batch["target_depth"].float(),
        )
        depth_fixed = fixed_reference_depth_loss(
            outputs["depth"].float(),
            batch["target_depth"].float(),
            baseline=batch["baseline_depth"].float(),
            reference_index=int(batch["reference_sequence_index"]),
        )
        pointmap = aligned_pointmap_loss(
            outputs["world_points"].float(),
            batch["target_world_points"].float(),
            scale=float(batch["point_alignment_scale"]),
            rotation=batch["point_alignment_rotation"].float(),
            translation=batch["point_alignment_translation"].float(),
        )
    terms = {
        "camera": camera,
        "relative_rotation": relative_rotation,
        "translation_direction": translation_direction,
        "rigid": rigid,
        "centroid": centroid,
        "residual": residual,
        "depth": depth,
        "depth_fixed": depth_fixed,
        "pointmap": pointmap,
    }
    total = (
        config.camera * camera
        + config.relative_rotation * relative_rotation
        + config.translation_direction * translation_direction
        + config.rigid * rigid
        + config.centroid * centroid
        + config.residual * residual
        + config.depth * depth
        + config.depth_fixed * depth_fixed
        + config.pointmap * pointmap
    )
    return total, {"total": total, **terms}


def camera_encoding_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    translation = F.smooth_l1_loss(predicted[..., :3], target[..., :3])
    pred_quat = F.normalize(predicted[..., 3:7], dim=-1, eps=1e-8)
    target_quat = F.normalize(target[..., 3:7], dim=-1, eps=1e-8)
    direct = (pred_quat - target_quat).abs().mean(dim=-1)
    flipped = (pred_quat + target_quat).abs().mean(dim=-1)
    rotation = torch.minimum(direct, flipped).mean()
    fov = F.smooth_l1_loss(predicted[..., 7:], target[..., 7:])
    return translation + rotation + 0.5 * fov


def relative_rotation_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_extrinsics, _ = _decode_pose(predicted, image_size=(2, 2))
    target_extrinsics, _ = _decode_pose(target, image_size=(2, 2))
    pred_rotation = pred_extrinsics[..., :3, :3]
    target_rotation = target_extrinsics[..., :3, :3]
    values = []
    for first in range(predicted.shape[1] - 1):
        for second in range(first + 1, predicted.shape[1]):
            pred_relative = pred_rotation[:, first] @ pred_rotation[:, second].transpose(-1, -2)
            target_relative = target_rotation[:, first] @ target_rotation[:, second].transpose(-1, -2)
            error = target_relative.transpose(-1, -2) @ pred_relative
            cosine = (
                (torch.diagonal(error, dim1=-2, dim2=-1).sum(-1) - 1.0)
                * 0.5
            ).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
            values.append(torch.acos(cosine))
    return torch.cat(values).mean() if values else predicted.new_zeros(())


def translation_direction_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_extrinsics, _ = _decode_pose(predicted, image_size=(2, 2))
    target_extrinsics, _ = _decode_pose(target, image_size=(2, 2))
    pred_h = _homogeneous(pred_extrinsics)
    target_h = _homogeneous(target_extrinsics)
    values = []
    for first in range(predicted.shape[1] - 1):
        for second in range(first + 1, predicted.shape[1]):
            pred_relative = pred_h[:, first] @ torch.linalg.inv(pred_h[:, second])
            target_relative = target_h[:, first] @ torch.linalg.inv(target_h[:, second])
            pred_t = pred_relative[:, :3, 3]
            target_t = target_relative[:, :3, 3]
            pred_norm = torch.linalg.vector_norm(pred_t, dim=-1)
            target_norm = torch.linalg.vector_norm(target_t, dim=-1)
            valid = (pred_norm > 1e-8) & (target_norm > 1e-8)
            if bool(valid.any()):
                cosine = F.cosine_similarity(pred_t[valid], target_t[valid], dim=-1).abs()
                values.append(1.0 - cosine.clamp(0.0, 1.0))
    return torch.cat(values).mean() if values else predicted.new_zeros(())


def instance_rigid_losses(
    pose_encoding: torch.Tensor,
    uvd: torch.Tensor,
    point_valid: torch.Tensor,
    instance_weight: torch.Tensor,
    *,
    image_size: tuple[int, int],
    scene_scale: float,
    trim_quantile: float,
    sequence_indices: list[int] | tuple[int, ...] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    extrinsics, intrinsics = _decode_pose(pose_encoding, image_size=image_size)
    world_points = _uvd_to_world(uvd, extrinsics, intrinsics)
    rigid_terms = []
    centroid_terms = []
    weights = []
    batch, sequence, instances = point_valid.shape[:3]
    frames = (
        list(range(sequence))
        if sequence_indices is None
        else [int(value) for value in sequence_indices]
    )
    if not frames:
        raise ValueError("sequence_indices must not be empty.")
    if len(set(frames)) != len(frames):
        raise ValueError("sequence_indices contain duplicates.")
    if any(frame < 0 or frame >= sequence for frame in frames):
        raise ValueError(f"sequence_indices must be in [0,{sequence}).")
    if frames != sorted(frames):
        raise ValueError("sequence_indices must be increasing.")
    for batch_index in range(batch):
        for instance in range(instances):
            previous = None
            for frame in frames:
                current_weight = instance_weight[batch_index, frame, instance]
                if float(current_weight.detach()) <= 0.0:
                    continue
                current_mask = point_valid[batch_index, frame, instance]
                if int(current_mask.sum()) < 8:
                    continue
                current = world_points[batch_index, frame, instance, current_mask]
                if previous is not None:
                    reference = previous.detach()
                    distances = torch.cdist(current, reference)
                    current_distance, current_index = distances.min(dim=1)
                    reference_distance = distances.min(dim=0).values
                    current_keep = _trimmed_keep(
                        current_distance,
                        trim_quantile=trim_quantile,
                    )
                    reference_keep = _trimmed_keep(
                        reference_distance,
                        trim_quantile=trim_quantile,
                    )
                    if int(current_keep.sum()) >= 4 and int(reference_keep.sum()) >= 4:
                        matched = reference.index_select(
                            0,
                            current_index[current_keep],
                        )
                        scale = max(float(scene_scale), 1e-6)
                        # Symmetric trimmed Chamfer.  The reference cloud is
                        # detached so the current frame is corrected toward
                        # causal history, never vice versa.
                        rigid_terms.append(
                            0.5
                            * (
                                current_distance[current_keep].mean()
                                + reference_distance[reference_keep].mean()
                            )
                            / scale
                        )
                        centroid_terms.append(
                            torch.linalg.vector_norm(
                                current[current_keep].mean(0) - matched.mean(0)
                            )
                            / scale
                        )
                        weights.append(current_weight.detach().clamp(0.0, 1.0))
                previous = current
    if not rigid_terms:
        zero = pose_encoding.new_zeros(())
        return zero, zero
    term_weights = torch.stack(weights)
    denominator = term_weights.sum().clamp_min(1e-6)
    rigid = (torch.stack(rigid_terms) * term_weights).sum() / denominator
    centroid = (torch.stack(centroid_terms) * term_weights).sum() / denominator
    return rigid, centroid


def _trimmed_keep(
    distances: torch.Tensor,
    *,
    trim_quantile: float,
) -> torch.Tensor:
    cutoff = torch.quantile(distances.detach(), float(trim_quantile))
    return distances <= cutoff


def scale_invariant_depth_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if predicted.ndim == 5 and predicted.shape[-1] == 1:
        predicted = predicted[..., 0]
    if target.ndim == 5 and target.shape[-1] == 1:
        target = target[..., 0]
    valid = (
        torch.isfinite(predicted)
        & torch.isfinite(target)
        & (predicted > 1e-6)
        & (target > 1e-6)
    )
    if not bool(valid.any()):
        return predicted.new_zeros(())
    difference = torch.log(predicted[valid].clamp_min(1e-6)) - torch.log(target[valid].clamp_min(1e-6))
    difference = difference - difference.mean()
    return torch.sqrt(difference.square().mean() + 1e-8)


def fixed_reference_depth_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    baseline: torch.Tensor,
    reference_index: int,
) -> torch.Tensor:
    """Log-depth loss under one scale fitted to the frozen reference frame."""

    predicted = _squeeze_depth(predicted)
    target = _squeeze_depth(target)
    baseline = _squeeze_depth(baseline)
    reference_predicted = baseline[:, int(reference_index)]
    reference_target = target[:, int(reference_index)]
    reference_valid = (
        torch.isfinite(reference_predicted)
        & torch.isfinite(reference_target)
        & (reference_predicted > 1e-6)
        & (reference_target > 1e-6)
    )
    if int(reference_valid.sum()) < 128:
        return predicted.new_zeros(())
    fixed_scale = (
        reference_target[reference_valid]
        / reference_predicted[reference_valid].clamp_min(1e-6)
    ).median().detach()
    valid = (
        torch.isfinite(predicted)
        & torch.isfinite(target)
        & (predicted > 1e-6)
        & (target > 1e-6)
    )
    if not bool(valid.any()):
        return predicted.new_zeros(())
    difference = (
        (predicted[valid] * fixed_scale).clamp_min(1e-6).log()
        - target[valid].clamp_min(1e-6).log()
    )
    if difference.numel() > 32768:
        indices = torch.linspace(
            0,
            difference.numel() - 1,
            32768,
            device=difference.device,
        ).round().long()
        difference = difference.index_select(0, indices)
    return F.smooth_l1_loss(difference, torch.zeros_like(difference))


def _squeeze_depth(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 5 and value.shape[-1] == 1:
        return value[..., 0]
    return value


def aligned_pointmap_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    scale: float,
    rotation: torch.Tensor,
    translation: torch.Tensor,
) -> torch.Tensor:
    aligned = float(scale) * (predicted @ rotation.transpose(-1, -2)) + translation
    valid = torch.isfinite(aligned).all(dim=-1) & torch.isfinite(target).all(dim=-1)
    if not bool(valid.any()):
        return predicted.new_zeros(())
    difference = torch.linalg.vector_norm(aligned[valid] - target[valid], dim=-1)
    if difference.numel() > 32768:
        indices = torch.linspace(0, difference.numel() - 1, 32768, device=difference.device).round().long()
        difference = difference.index_select(0, indices)
    cutoff = torch.quantile(difference.detach(), 0.90)
    return difference[difference <= cutoff].mean()


def _uvd_to_world(
    uvd: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
) -> torch.Tensor:
    u, v, depth = uvd.unbind(dim=-1)
    fx = intrinsics[..., 0, 0][..., None, None]
    fy = intrinsics[..., 1, 1][..., None, None]
    cx = intrinsics[..., 0, 2][..., None, None]
    cy = intrinsics[..., 1, 2][..., None, None]
    local = torch.stack(
        [
            (u - cx) * depth / fx.clamp_min(1e-6),
            (v - cy) * depth / fy.clamp_min(1e-6),
            depth,
        ],
        dim=-1,
    )
    homogeneous = _homogeneous(extrinsics)
    camera_to_world = torch.linalg.inv(homogeneous)
    rotation = camera_to_world[..., :3, :3]
    translation = camera_to_world[..., :3, 3]
    world = torch.einsum("bsij,bsknj->bskni", rotation, local)
    return torch.nan_to_num(world + translation[..., None, None, :])


def _decode_pose(
    pose_encoding: torch.Tensor,
    *,
    image_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri

    return pose_encoding_to_extri_intri(
        pose_encoding.float(),
        image_size_hw=image_size,
    )


def _homogeneous(extrinsics: torch.Tensor) -> torch.Tensor:
    result = torch.eye(4, dtype=extrinsics.dtype, device=extrinsics.device)
    result = result.expand(*extrinsics.shape[:-2], 4, 4).clone()
    result[..., :3, :4] = extrinsics
    return result
