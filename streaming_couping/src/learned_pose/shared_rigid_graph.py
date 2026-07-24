"""Conservative shared SE(3) correction for poses and world pointmaps.

Unlike ray-depth reconstruction, this module never replaces a learned world
pointmap with a reposed raw depth map.  It estimates a small per-frame rigid
world transform from robust cross-view point correspondences and applies that
same transform to both the camera and every point in the frame.  Local shape is
therefore preserved and the raw reference gauge remains fixed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from streaming_couping.src.learned_pose.joint_ba import (
    JointBAResult,
    _c2w_to_w2c,
    _confidence,
    _deterministic_limit,
    _huber,
    _patch_features,
    _patch_sample_indices,
    _pose_batch,
    _region_labels,
    _single_batch,
    _so3_exp,
    _w2c_to_c2w,
    _world_to_camera_depth,
    dcs_switch,
    regauge_camera_to_world,
)


@dataclass(frozen=True)
class SharedRigidConfig:
    inner_steps: int = 80
    learning_rate: float = 0.025
    match_radius_patches: int = 2
    max_matches_per_edge_region: int = 128
    min_matches_per_edge_region: int = 8
    min_total_matches: int = 32
    min_matches_per_frame: int = 16
    feature_dim_limit: int = 256
    min_feature_cosine: float = 0.10
    min_point_confidence: float = 0.30
    max_initial_point_distance_ratio: float = 0.35
    max_forward_backward_patches: float = 3.0
    point_huber_delta_ratio: float = 0.05
    rotation_prior_weight: float = 0.10
    translation_prior_weight: float = 0.10
    max_rotation_update_degrees: float = 1.5
    max_translation_scene_ratio: float = 0.05
    dcs_point_sigma_ratio: float = 0.08
    minimum_instance_switch: float = 0.15
    min_relative_residual_improvement: float = 0.01


@dataclass(frozen=True)
class SharedRigidVariant:
    name: str
    point_source: str
    instance_switches: bool


SHARED_RIGID_VARIANTS = (
    SharedRigidVariant("shared_se3_raw_dcs", "raw", True),
    SharedRigidVariant("shared_se3_learned", "learned", False),
    SharedRigidVariant("shared_se3_learned_dcs", "learned", True),
)


@dataclass(frozen=True)
class _PointMatches:
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


def run_shared_rigid_graph(
    *,
    raw_pose_encoding: torch.Tensor,
    initial_pose_encoding: torch.Tensor,
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
    variant: SharedRigidVariant,
    config: SharedRigidConfig = SharedRigidConfig(),
) -> JointBAResult:
    """Apply one small accepted shared transform to each supported frame."""

    if variant.point_source not in {"raw", "learned"}:
        raise ValueError(f"Unknown point source: {variant.point_source!r}.")
    device = raw_world_points.device
    dtype = torch.float32
    raw_pose_encoding = _pose_batch(raw_pose_encoding).to(device=device, dtype=dtype)
    initial_pose_encoding = _pose_batch(initial_pose_encoding).to(
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
    initial_w2c, _ = pose_encoding_to_extri_intri(
        initial_pose_encoding,
        image_size_hw=image_size,
    )
    raw_c2w = _w2c_to_c2w(raw_w2c[0])
    initial_c2w = regauge_camera_to_world(
        _w2c_to_c2w(initial_w2c[0]),
        raw_c2w,
        reference_index=int(reference_index),
    ).clone()
    initial_c2w[int(reference_index)] = raw_c2w[int(reference_index)]
    sequence = int(initial_c2w.shape[0])
    candidate_points = (
        raw_world_points
        if variant.point_source == "raw"
        else learned_world_points
    )
    candidate_confidence = (
        raw_confidence
        if variant.point_source == "raw"
        else learned_confidence
    )
    candidate_points = candidate_points.clone()
    candidate_confidence = candidate_confidence.clone()
    candidate_points[int(reference_index)] = raw_world_points[
        int(reference_index)
    ]
    candidate_confidence[int(reference_index)] = raw_confidence[
        int(reference_index)
    ]
    if candidate_points.shape != raw_world_points.shape:
        raise ValueError("Raw and learned pointmaps must share [S,H,W,3].")
    if candidate_points.shape[0] != sequence:
        raise ValueError("Pose and pointmap sequence lengths disagree.")
    height, width = (int(value) for value in candidate_points.shape[1:3])
    if (height, width) != tuple(int(value) for value in image_size):
        raise ValueError("Pointmap and image sizes disagree.")

    intrinsics = (
        raw_intrinsics[0, int(reference_index)][None]
        .expand(sequence, -1, -1)
        .clone()
    )
    labels = _region_labels(
        tracking_masks,
        trusted_tracking_masks,
        trusted_instance_valid,
    )
    features = _patch_features(
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
    points_patch = candidate_points[:, patch_y, patch_x]
    confidence_patch = candidate_confidence[:, patch_y, patch_x].clamp(0.0, 1.0)
    labels_patch = labels[:, patch_y, patch_x]
    scene_scale = max(float(scene_scale), 1e-6)
    matches = _build_point_matches(
        camera_to_world=initial_c2w,
        points_patch=points_patch,
        confidence_patch=confidence_patch,
        labels_patch=labels_patch,
        features=features,
        intrinsics=intrinsics,
        image_size=image_size,
        patch_shape=patch_shape,
        reference_index=int(reference_index),
        scene_scale=scene_scale,
        instance_switches=bool(variant.instance_switches),
        config=config,
    )

    initial_depth = _world_to_camera_depth(candidate_points, initial_c2w)
    if matches.count < int(config.min_total_matches):
        return _fallback_result(
            name=variant.name,
            pose=initial_pose_encoding,
            points=candidate_points,
            depth=initial_depth,
            raw_pose=raw_pose_encoding,
            reference_index=int(reference_index),
            matches=matches,
            status="fallback_insufficient_cross_view_matches",
            config=config,
        )

    incident = _incident_match_counts(matches, sequence=sequence)
    optimizable = (
        incident >= int(config.min_matches_per_frame)
    ).to(device=device, dtype=dtype)[:, None]
    optimizable[int(reference_index)] = 0.0
    if not bool(optimizable.bool().any()):
        return _fallback_result(
            name=variant.name,
            pose=initial_pose_encoding,
            points=candidate_points,
            depth=initial_depth,
            raw_pose=raw_pose_encoding,
            reference_index=int(reference_index),
            matches=matches,
            status="fallback_insufficient_frame_support",
            config=config,
        )

    rotation_parameter = torch.nn.Parameter(
        torch.zeros(sequence, 3, device=device, dtype=dtype)
    )
    translation_parameter = torch.nn.Parameter(
        torch.zeros(sequence, 3, device=device, dtype=dtype)
    )
    parameters = [rotation_parameter, translation_parameter]
    optimizer = torch.optim.Adam(parameters, lr=float(config.learning_rate))
    initial_distance = _point_distances(
        matches,
        points_patch=points_patch,
        initial_c2w=initial_c2w,
        rotation_parameter=rotation_parameter,
        translation_parameter=translation_parameter,
        update_mask=optimizable,
        scene_scale=scene_scale,
        config=config,
    )
    initial_rmse = _weighted_rmse(initial_distance, matches.weight)
    best_loss = float("inf")
    best_state: tuple[torch.Tensor, torch.Tensor] | None = None
    stale = 0
    for _ in range(int(config.inner_steps)):
        optimizer.zero_grad(set_to_none=True)
        distance = _point_distances(
            matches,
            points_patch=points_patch,
            initial_c2w=initial_c2w,
            rotation_parameter=rotation_parameter,
            translation_parameter=translation_parameter,
            update_mask=optimizable,
            scene_scale=scene_scale,
            config=config,
        )
        weight = matches.weight.clamp_min(0.0)
        data_loss = (
            weight
            * _huber(
                distance,
                float(config.point_huber_delta_ratio),
            )
        ).sum() / weight.sum().clamp_min(1e-6)
        prior = (
            float(config.rotation_prior_weight)
            * rotation_parameter.square().mean()
            + float(config.translation_prior_weight)
            * translation_parameter.square().mean()
        )
        loss = data_loss + prior
        if not bool(torch.isfinite(loss)):
            break
        current = float(loss.detach())
        if current + 1e-9 < best_loss:
            best_loss = current
            best_state = (
                rotation_parameter.detach().clone(),
                translation_parameter.detach().clone(),
            )
            stale = 0
        else:
            stale += 1
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 5.0)
        optimizer.step()
        if stale >= 15:
            break

    if best_state is None:
        return _fallback_result(
            name=variant.name,
            pose=initial_pose_encoding,
            points=candidate_points,
            depth=initial_depth,
            raw_pose=raw_pose_encoding,
            reference_index=int(reference_index),
            matches=matches,
            status="fallback_non_finite_optimization",
            config=config,
        )

    with torch.no_grad():
        rotation_parameter.copy_(best_state[0])
        translation_parameter.copy_(best_state[1])
        proposed_distance = _point_distances(
            matches,
            points_patch=points_patch,
            initial_c2w=initial_c2w,
            rotation_parameter=rotation_parameter,
            translation_parameter=translation_parameter,
            update_mask=optimizable,
            scene_scale=scene_scale,
            config=config,
        )
        accepted = _accepted_frames(
            matches,
            initial_distance=initial_distance,
            proposed_distance=proposed_distance,
            weight=matches.weight,
            sequence=sequence,
            reference_index=int(reference_index),
            minimum_matches=int(config.min_matches_per_frame),
            minimum_improvement=float(
                config.min_relative_residual_improvement
            ),
        ).to(device=device, dtype=dtype)[:, None]
        accepted = accepted * optimizable
        delta_rotation, delta_translation, final_c2w = _shared_transforms(
            initial_c2w,
            rotation_parameter,
            translation_parameter,
            update_mask=accepted,
            scene_scale=scene_scale,
            config=config,
        )
        final_distance = _transformed_match_distances(
            matches,
            points_patch=points_patch,
            initial_c2w=initial_c2w,
            delta_rotation=delta_rotation,
            delta_translation=delta_translation,
            scene_scale=scene_scale,
        )
        final_rmse = _weighted_rmse(final_distance, matches.weight)
        improvement = (
            (initial_rmse - final_rmse) / max(initial_rmse, 1e-8)
            if math.isfinite(initial_rmse) and math.isfinite(final_rmse)
            else float("-inf")
        )
        accepted_count = int(accepted.bool().sum())
        if (
            accepted_count == 0
            or improvement
            < float(config.min_relative_residual_improvement)
        ):
            return _fallback_result(
                name=variant.name,
                pose=initial_pose_encoding,
                points=candidate_points,
                depth=initial_depth,
                raw_pose=raw_pose_encoding,
                reference_index=int(reference_index),
                matches=matches,
                status="fallback_no_reliable_residual_improvement",
                config=config,
                initial_point_rmse=initial_rmse,
                final_point_rmse=final_rmse,
            )

        final_points = _transform_full_pointmap(
            candidate_points,
            initial_c2w=initial_c2w,
            delta_rotation=delta_rotation,
            delta_translation=delta_translation,
        )
        inactive = ~accepted[:, 0].bool()
        final_points[inactive] = candidate_points[inactive]
        final_depth = _world_to_camera_depth(final_points, final_c2w)
        final_w2c = _c2w_to_w2c(final_c2w)
        final_pose = extri_intri_to_pose_encoding(
            final_w2c[None],
            intrinsics[None],
            image_size_hw=image_size,
        )
        final_pose[:, inactive] = initial_pose_encoding[:, inactive]
        final_pose[:, int(reference_index)] = raw_pose_encoding[
            :, int(reference_index)
        ]
        center_shift = torch.linalg.vector_norm(
            delta_translation,
            dim=-1,
        )
        diagnostics = _diagnostics(
            matches,
            status="accepted_shared_se3",
            config=config,
            reference_exact=int(
                torch.equal(
                    final_pose[:, int(reference_index)],
                    raw_pose_encoding[:, int(reference_index)],
                )
            ),
            initial_point_rmse=initial_rmse,
            final_point_rmse=final_rmse,
            accepted_frames=accepted_count,
            mean_center_shift=float(center_shift[accepted[:, 0].bool()].mean()),
            objective=best_loss,
        )
    return JointBAResult(
        name=variant.name,
        pose_encoding=final_pose,
        world_points=final_points[None],
        depth=final_depth[None],
        diagnostics=diagnostics,
    )


@torch.no_grad()
def _build_point_matches(
    *,
    camera_to_world: torch.Tensor,
    points_patch: torch.Tensor,
    confidence_patch: torch.Tensor,
    labels_patch: torch.Tensor,
    features: torch.Tensor,
    intrinsics: torch.Tensor,
    image_size: tuple[int, int],
    patch_shape: tuple[int, int],
    reference_index: int,
    scene_scale: float,
    instance_switches: bool,
    config: SharedRigidConfig,
) -> _PointMatches:
    sequence = int(points_patch.shape[0])
    patch_h, patch_w = (int(value) for value in patch_shape)
    height, width = (int(value) for value in image_size)
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
                device=points_patch.device,
            ),
            torch.arange(
                -int(config.match_radius_patches),
                int(config.match_radius_patches) + 1,
                device=points_patch.device,
            ),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 2)
    source_frames: list[torch.Tensor] = []
    target_frames: list[torch.Tensor] = []
    source_patches: list[torch.Tensor] = []
    target_patches: list[torch.Tensor] = []
    regions: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    edge_rows: list[tuple[int, int, int, float, int]] = []
    region_count = int(labels_patch.max().clamp_min(0)) + 1

    for source, target in sorted(edges):
        for region in range(region_count):
            source_valid = (
                (labels_patch[source] == region)
                & torch.isfinite(points_patch[source]).all(dim=-1)
                & (
                    confidence_patch[source]
                    >= float(config.min_point_confidence)
                )
            )
            source_index = _deterministic_limit(
                source_valid.nonzero(as_tuple=False).flatten(),
                int(config.max_matches_per_edge_region),
            )
            if source_index.numel() == 0:
                continue
            source_world = points_patch[source, source_index]
            target_local = torch.einsum(
                "ji,nj->ni",
                camera_to_world[target, :3, :3],
                source_world - camera_to_world[target, :3, 3],
            )
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
            base = torch.stack(
                [
                    ((projected_v + 0.5) * patch_h / height - 0.5)
                    .round()
                    .long(),
                    ((projected_u + 0.5) * patch_w / width - 0.5)
                    .round()
                    .long(),
                ],
                dim=-1,
            )
            candidates = base[:, None] + offsets[None]
            candidate_valid = (
                (target_local[:, 2, None] > 1e-6)
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
            candidate_valid &= torch.isfinite(
                points_patch[target, candidate_flat]
            ).all(dim=-1)
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
            source_world = points_patch[source, source_kept]
            target_world = points_patch[target, target_kept]
            distance_ratio = (
                torch.linalg.vector_norm(
                    source_world - target_world,
                    dim=-1,
                )
                / scene_scale
            )
            source_local = torch.einsum(
                "ji,nj->ni",
                camera_to_world[source, :3, :3],
                target_world - camera_to_world[source, :3, 3],
            )
            back_u = (
                intrinsics[source, 0, 0]
                * source_local[:, 0]
                / source_local[:, 2].clamp_min(1e-6)
                + intrinsics[source, 0, 2]
            )
            back_v = (
                intrinsics[source, 1, 1]
                * source_local[:, 1]
                / source_local[:, 2].clamp_min(1e-6)
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
                (source_local[:, 2] > 1e-6)
                & (
                    distance_ratio
                    <= float(config.max_initial_point_distance_ratio)
                )
                & (
                    fb_error
                    <= float(config.max_forward_backward_patches)
                )
            )
            source_kept = source_kept[keep_geometry]
            target_kept = target_kept[keep_geometry]
            similarity_kept = similarity_kept[keep_geometry]
            distance_ratio = distance_ratio[keep_geometry]
            if (
                source_kept.numel()
                < int(config.min_matches_per_edge_region)
            ):
                continue
            if region > 0 and instance_switches:
                normalized = (
                    distance_ratio.median()
                    / max(float(config.dcs_point_sigma_ratio), 1e-6)
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
            source_frames.append(torch.full_like(source_kept, source))
            target_frames.append(torch.full_like(target_kept, target))
            source_patches.append(source_kept)
            target_patches.append(target_kept)
            regions.append(torch.full_like(source_kept, region))
            weights.append(confidence_weight * feature_weight * switch)
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
        empty = torch.empty(
            0,
            device=points_patch.device,
            dtype=torch.long,
        )
        return _PointMatches(
            empty,
            empty,
            empty,
            empty,
            empty,
            empty.float(),
            tuple(edge_rows),
        )
    return _PointMatches(
        torch.cat(source_frames),
        torch.cat(target_frames),
        torch.cat(source_patches),
        torch.cat(target_patches),
        torch.cat(regions),
        torch.cat(weights),
        tuple(edge_rows),
    )


def _shared_transforms(
    initial_c2w: torch.Tensor,
    rotation_parameter: torch.Tensor,
    translation_parameter: torch.Tensor,
    *,
    update_mask: torch.Tensor,
    scene_scale: float,
    config: SharedRigidConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    omega = (
        math.radians(float(config.max_rotation_update_degrees))
        * torch.tanh(rotation_parameter)
        * update_mask
    )
    delta_rotation = _so3_exp(omega)
    delta_translation = (
        float(config.max_translation_scene_ratio)
        * float(scene_scale)
        * torch.tanh(translation_parameter)
        * update_mask
    )
    rotation = delta_rotation @ initial_c2w[:, :3, :3]
    center = initial_c2w[:, :3, 3] + delta_translation
    bottom = torch.zeros(
        initial_c2w.shape[0],
        1,
        4,
        device=initial_c2w.device,
        dtype=initial_c2w.dtype,
    )
    bottom[:, 0, 3] = 1.0
    camera_to_world = torch.cat(
        [torch.cat([rotation, center[..., None]], dim=-1), bottom],
        dim=-2,
    )
    return delta_rotation, delta_translation, camera_to_world


def _point_distances(
    matches: _PointMatches,
    *,
    points_patch: torch.Tensor,
    initial_c2w: torch.Tensor,
    rotation_parameter: torch.Tensor,
    translation_parameter: torch.Tensor,
    update_mask: torch.Tensor,
    scene_scale: float,
    config: SharedRigidConfig,
) -> torch.Tensor:
    rotation, translation, _ = _shared_transforms(
        initial_c2w,
        rotation_parameter,
        translation_parameter,
        update_mask=update_mask,
        scene_scale=scene_scale,
        config=config,
    )
    return _transformed_match_distances(
        matches,
        points_patch=points_patch,
        initial_c2w=initial_c2w,
        delta_rotation=rotation,
        delta_translation=translation,
        scene_scale=scene_scale,
    )


def _transformed_match_distances(
    matches: _PointMatches,
    *,
    points_patch: torch.Tensor,
    initial_c2w: torch.Tensor,
    delta_rotation: torch.Tensor,
    delta_translation: torch.Tensor,
    scene_scale: float,
) -> torch.Tensor:
    source = matches.source_frame
    target = matches.target_frame
    source_point = _transform_selected_points(
        points_patch[source, matches.source_patch],
        frame=source,
        initial_c2w=initial_c2w,
        delta_rotation=delta_rotation,
        delta_translation=delta_translation,
    )
    target_point = _transform_selected_points(
        points_patch[target, matches.target_patch],
        frame=target,
        initial_c2w=initial_c2w,
        delta_rotation=delta_rotation,
        delta_translation=delta_translation,
    )
    return (
        torch.linalg.vector_norm(source_point - target_point, dim=-1)
        / max(float(scene_scale), 1e-6)
    )


def _transform_selected_points(
    points: torch.Tensor,
    *,
    frame: torch.Tensor,
    initial_c2w: torch.Tensor,
    delta_rotation: torch.Tensor,
    delta_translation: torch.Tensor,
) -> torch.Tensor:
    pivot = initial_c2w[frame, :3, 3]
    return (
        torch.bmm(
            delta_rotation[frame],
            (points - pivot)[..., None],
        )[..., 0]
        + pivot
        + delta_translation[frame]
    )


def _transform_full_pointmap(
    points: torch.Tensor,
    *,
    initial_c2w: torch.Tensor,
    delta_rotation: torch.Tensor,
    delta_translation: torch.Tensor,
) -> torch.Tensor:
    pivot = initial_c2w[:, :3, 3]
    return (
        torch.einsum(
            "sij,shwj->shwi",
            delta_rotation,
            points - pivot[:, None, None],
        )
        + pivot[:, None, None]
        + delta_translation[:, None, None]
    )


def _incident_match_counts(
    matches: _PointMatches,
    *,
    sequence: int,
) -> torch.Tensor:
    count = torch.zeros(
        sequence,
        device=matches.source_frame.device,
        dtype=torch.long,
    )
    count.scatter_add_(
        0,
        matches.source_frame,
        torch.ones_like(matches.source_frame),
    )
    count.scatter_add_(
        0,
        matches.target_frame,
        torch.ones_like(matches.target_frame),
    )
    return count


def _accepted_frames(
    matches: _PointMatches,
    *,
    initial_distance: torch.Tensor,
    proposed_distance: torch.Tensor,
    weight: torch.Tensor,
    sequence: int,
    reference_index: int,
    minimum_matches: int,
    minimum_improvement: float,
) -> torch.Tensor:
    accepted = torch.zeros(
        sequence,
        device=initial_distance.device,
        dtype=torch.bool,
    )
    for frame in range(sequence):
        if frame == int(reference_index):
            continue
        incident = (
            (matches.source_frame == frame)
            | (matches.target_frame == frame)
        )
        if int(incident.sum()) < int(minimum_matches):
            continue
        before = _weighted_rmse(initial_distance[incident], weight[incident])
        after = _weighted_rmse(proposed_distance[incident], weight[incident])
        if (
            math.isfinite(before)
            and math.isfinite(after)
            and after
            <= before * (1.0 - float(minimum_improvement))
        ):
            accepted[frame] = True
    return accepted


def _weighted_rmse(value: torch.Tensor, weight: torch.Tensor) -> float:
    value = value.detach()
    weight = weight.detach().clamp_min(0.0)
    if value.numel() == 0 or float(weight.sum()) <= 0.0:
        return float("nan")
    return float(
        torch.sqrt(
            (weight * value.square()).sum()
            / weight.sum().clamp_min(1e-8)
        )
    )


def _fallback_result(
    *,
    name: str,
    pose: torch.Tensor,
    points: torch.Tensor,
    depth: torch.Tensor,
    raw_pose: torch.Tensor,
    reference_index: int,
    matches: _PointMatches,
    status: str,
    config: SharedRigidConfig,
    initial_point_rmse: float = float("nan"),
    final_point_rmse: float = float("nan"),
) -> JointBAResult:
    final_pose = pose.clone()
    final_pose[:, int(reference_index)] = raw_pose[:, int(reference_index)]
    return JointBAResult(
        name=name,
        pose_encoding=final_pose,
        world_points=points[None],
        depth=depth[None],
        diagnostics=_diagnostics(
            matches,
            status=status,
            config=config,
            reference_exact=int(
                torch.equal(
                    final_pose[:, int(reference_index)],
                    raw_pose[:, int(reference_index)],
                )
            ),
            initial_point_rmse=initial_point_rmse,
            final_point_rmse=final_point_rmse,
            accepted_frames=0,
            mean_center_shift=0.0,
            objective=float("nan"),
        ),
    )


def _diagnostics(
    matches: _PointMatches,
    *,
    status: str,
    config: SharedRigidConfig,
    reference_exact: int,
    initial_point_rmse: float,
    final_point_rmse: float,
    accepted_frames: int,
    mean_center_shift: float,
    objective: float,
) -> dict[str, object]:
    active = sum(
        int(region > 0 and switch >= config.minimum_instance_switch)
        for _, _, region, switch, _ in matches.edge_region
    )
    rejected = sum(
        int(region > 0 and switch < config.minimum_instance_switch)
        for _, _, region, switch, _ in matches.edge_region
    )
    return {
        "status": status,
        "matches": matches.count,
        "edge_regions": len(matches.edge_region),
        "active_instance_edges": active,
        "rejected_instance_edges": rejected,
        "initial_point_rmse": initial_point_rmse,
        "final_point_rmse": final_point_rmse,
        "accepted_frames": int(accepted_frames),
        "mean_pose_center_shift_native": mean_center_shift,
        "reference_anchor_exact": int(reference_exact),
        "objective": objective,
    }
