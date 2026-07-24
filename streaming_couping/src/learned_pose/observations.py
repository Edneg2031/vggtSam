"""Build detached, causal instance observations for learned pose guidance."""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn.functional as F

from ..instance_observations import (
    InstanceRefinementConfig,
    TranslationProposal,
    deterministic_limit,
    merge_map_points,
    proposal_consensus,
    translation_icp,
)


GEOMETRY_FEATURE_NAMES = (
    "center_x",
    "center_y",
    "center_z",
    "log_cov_eigenvalue_0",
    "log_cov_eigenvalue_1",
    "log_cov_eigenvalue_2",
    "log_extent_x",
    "log_extent_y",
    "log_extent_z",
    "proposal_translation_x",
    "proposal_translation_y",
    "proposal_translation_z",
    "proposal_fitness",
    "proposal_rmse_ratio",
    "shape_similarity",
    "mean_point_confidence",
    "mask_area_ratio",
    "log_point_count",
    "consensus_residual_ratio",
    "normalized_frame_gap",
)

QUALITY_NAMES = (
    "track_confidence",
    "geometry_confidence",
    "static_score",
)

POSE_RESIDUAL_FEATURE_NAMES = (
    "observed_centroid_x",
    "observed_centroid_y",
    "projected_centroid_x",
    "projected_centroid_y",
    "centroid_residual_x",
    "centroid_residual_y",
    "observed_cov_xx",
    "observed_cov_yy",
    "observed_cov_xy",
    "projected_cov_xx",
    "projected_cov_yy",
    "projected_cov_xy",
    "cov_residual_xx",
    "cov_residual_yy",
    "cov_residual_xy",
    "observed_area_fraction",
    "projected_area_fraction",
    "projection_coverage",
    "projection_iou",
    "mean_ray_x",
    "mean_ray_y",
    "mean_ray_z",
    "ray_variance_x",
    "ray_variance_y",
    "ray_variance_z",
    "normalized_log_map_points",
    "normalized_object_spread",
)


@torch.no_grad()
def pool_sam_instance_features(
    spatial_features: torch.Tensor,
    masks: torch.Tensor,
) -> torch.Tensor:
    """Pool frozen SAM3 feature mean/std inside each recovered mask.

    Args:
        spatial_features: ``[S,C,Hf,Wf]`` frozen SAM3 FPN features.
        masks: ``[S,K,Hm,Wm]`` recovered same-ID masks.
    Returns:
        ``[S,K,2C]`` mean/std descriptors.
    """

    if spatial_features.ndim != 4 or masks.ndim != 4:
        raise ValueError("Expected SAM features [S,C,H,W] and masks [S,K,H,W].")
    if spatial_features.shape[0] != masks.shape[0]:
        raise ValueError("SAM feature and mask frame counts differ.")
    resized = F.interpolate(
        masks.to(device=spatial_features.device, dtype=torch.float32),
        size=spatial_features.shape[-2:],
        mode="nearest",
    ).bool()
    sequence, instances = resized.shape[:2]
    channels = spatial_features.shape[1]
    output = torch.zeros(
        sequence,
        instances,
        channels * 2,
        dtype=torch.float32,
        device=spatial_features.device,
    )
    for frame_index in range(sequence):
        feature = spatial_features[frame_index].float()
        for instance_index in range(instances):
            selected = feature[:, resized[frame_index, instance_index]]
            if selected.shape[1] == 0:
                continue
            mean = selected.mean(dim=1)
            variance = (selected - mean[:, None]).square().mean(dim=1)
            output[frame_index, instance_index] = torch.cat(
                [mean, torch.sqrt(variance.clamp_min(1e-8))]
            )
    return output.cpu()


@torch.no_grad()
def build_geometry_observations(
    *,
    world_points: torch.Tensor,
    confidence: torch.Tensor,
    masks: torch.Tensor,
    scores: torch.Tensor,
    instance_ids: Sequence[int],
    frame_indices: Sequence[int],
    reference_index: int,
    confidence_threshold: float,
    refinement: InstanceRefinementConfig,
    sampled_instance_points: int,
    hard_mismatch_min_points: int = 512,
    hard_mismatch_max_fitness: float = 0.02,
) -> dict[str, object]:
    """Create geometry descriptors and camera-local samples without GT gates.

    Geometry/static scores are deterministic and detached. They can weight a
    learned loss, but cannot collapse to zero through optimization.
    """

    world_points = world_points.detach().float().cpu()
    confidence = confidence.detach().float().cpu()
    masks = masks.detach().bool().cpu()
    scores = scores.detach().float().cpu()
    sequence, instances = masks.shape[:2]
    if world_points.shape[0] != sequence or scores.shape != (sequence, instances):
        raise ValueError("Geometry, masks, and scores do not share [S,K].")
    if len(instance_ids) != instances or len(frame_indices) != sequence:
        raise ValueError("instance_ids/frame_indices disagree with observation tensors.")

    origin, scene_scale = _reference_normalization(
        world_points[int(reference_index)],
        confidence[int(reference_index)],
        confidence_threshold=confidence_threshold,
    )
    geometry = torch.zeros(sequence, instances, len(GEOMETRY_FEATURE_NAMES))
    quality = torch.zeros(sequence, instances, len(QUALITY_NAMES))
    observed = torch.zeros(sequence, instances, dtype=torch.bool)
    identity_valid = torch.zeros(sequence, instances, dtype=torch.bool)
    identity_unknown = torch.zeros(sequence, instances, dtype=torch.bool)
    identity_mismatch = torch.zeros(sequence, instances, dtype=torch.bool)
    point_counts = torch.zeros(sequence, instances, dtype=torch.long)
    identity_rows: list[dict[str, object]] = []

    selected_points: list[list[torch.Tensor]] = [
        [torch.empty(0, 3) for _ in range(instances)]
        for _ in range(sequence)
    ]
    stats: list[list[dict[str, torch.Tensor | float | int]]] = [
        [{} for _ in range(instances)] for _ in range(sequence)
    ]
    for frame in range(sequence):
        for slot in range(instances):
            points, point_conf = _masked_points_and_confidence(
                world_points[frame],
                confidence[frame],
                masks[frame, slot],
                confidence_threshold=confidence_threshold,
                max_points=refinement.map_max_points,
            )
            selected_points[frame][slot] = points
            point_counts[frame, slot] = points.shape[0]
            stats[frame][slot] = _point_statistics(
                points,
                point_conf,
                origin=origin,
                scene_scale=scene_scale,
                mask_area=float(masks[frame, slot].float().mean()),
            )
            # Observation availability is a 2D tracking fact. Geometry-aware
            # modes separately reject insufficient 3D support through their
            # geometry confidence, while the SAM-only control must not receive
            # a hidden point-count gate.
            observed[frame, slot] = bool(masks[frame, slot].any())

    object_maps: dict[int, torch.Tensor] = {}
    reference_shapes: dict[int, torch.Tensor] = {}
    for slot, instance_id in enumerate(instance_ids):
        reference_points = selected_points[int(reference_index)][slot]
        if reference_points.shape[0] >= refinement.min_instance_points:
            object_maps[int(instance_id)] = reference_points
            reference_shapes[int(instance_id)] = stats[int(reference_index)][slot][
                "log_eigenvalues"
            ]

    previous_frame = int(frame_indices[int(reference_index)])
    for frame in range(sequence):
        proposals: list[TranslationProposal] = []
        proposal_by_slot: dict[int, TranslationProposal] = {}
        if frame != int(reference_index):
            for slot, instance_id in enumerate(instance_ids):
                current = selected_points[frame][slot]
                if int(instance_id) not in object_maps:
                    continue
                proposal = translation_icp(
                    current,
                    object_maps[int(instance_id)],
                    instance_id=int(instance_id),
                    config=refinement,
                )
                proposals.append(proposal)
                proposal_by_slot[slot] = proposal
        eligible = [
            proposal
            for proposal in proposals
            if proposal.accepted
            and float(scores[frame, list(instance_ids).index(proposal.instance_id)]) >= 0.5
        ]
        shared, participating, _ = proposal_consensus(
            eligible,
            min_instances=min(refinement.min_participating_instances, instances),
            max_distance=refinement.consensus_distance,
        )
        participating_set = set(participating)

        for slot, instance_id in enumerate(instance_ids):
            item = stats[frame][slot]
            if not item:
                continue
            proposal = proposal_by_slot.get(slot)
            shape_reference = reference_shapes.get(int(instance_id))
            shape_similarity = _shape_similarity(
                item["log_eigenvalues"],
                shape_reference,
            )
            proposal_translation = torch.zeros(3)
            fitness = 0.0
            rmse_ratio = 2.0
            consensus_ratio = 2.0
            geometry_confidence = 0.0
            static_score = 0.0
            if not bool(observed[frame, slot]):
                current_identity_valid = False
                current_identity_unknown = False
                current_identity_mismatch = False
                identity_state = "ABSENT"
                identity_reason = "absent_tracker_mask"
            elif frame == int(reference_index):
                current_identity_valid = int(instance_id) in object_maps
                current_identity_unknown = not current_identity_valid
                current_identity_mismatch = False
                identity_state = (
                    "MATCH" if current_identity_valid else "UNKNOWN"
                )
                identity_reason = (
                    "accepted_reference"
                    if current_identity_valid
                    else "unknown_reference_insufficient_points"
                )
            elif proposal is None:
                current_identity_valid = False
                current_identity_unknown = True
                current_identity_mismatch = False
                identity_state = "UNKNOWN"
                identity_reason = "unknown_missing_persistent_map"
            else:
                current_identity_valid = bool(proposal.accepted)
                current_identity_mismatch = (
                    not proposal.accepted
                    and int(proposal.current_points)
                    >= int(hard_mismatch_min_points)
                    and int(proposal.correspondences) == 0
                    and float(proposal.fitness)
                    <= float(hard_mismatch_max_fitness)
                )
                current_identity_unknown = (
                    not current_identity_valid
                    and not current_identity_mismatch
                )
                reason_suffix = proposal.reason.replace(
                    "; ", "_"
                ).replace(" ", "_")
                if current_identity_valid:
                    identity_state = "MATCH"
                    identity_reason = "accepted_bounded_3d_registration"
                elif current_identity_mismatch:
                    identity_state = "MISMATCH"
                    identity_reason = f"mismatch_strong_zero_support_{reason_suffix}"
                else:
                    identity_state = "UNKNOWN"
                    identity_reason = f"unknown_geometry_inconclusive_{reason_suffix}"
            identity_valid[frame, slot] = current_identity_valid
            identity_unknown[frame, slot] = current_identity_unknown
            identity_mismatch[frame, slot] = current_identity_mismatch
            if frame == int(reference_index):
                geometry_confidence = 1.0
                static_score = 1.0
                rmse_ratio = 0.0
                consensus_ratio = 0.0
            elif proposal is not None:
                proposal_translation = proposal.translation.float() / scene_scale
                fitness = float(proposal.fitness)
                if math.isfinite(proposal.rmse) and math.isfinite(proposal.correspondence_distance):
                    rmse_ratio = float(proposal.rmse) / max(
                        float(proposal.correspondence_distance), 1e-6
                    )
                if proposal.accepted:
                    geometry_confidence = (
                        fitness
                        * math.exp(-min(rmse_ratio, 10.0))
                        * shape_similarity
                        * float(item["mean_confidence"])
                    )
                    if shared is not None:
                        residual = float(
                            torch.linalg.vector_norm(
                                proposal.translation - shared
                            )
                        )
                        consensus_ratio = residual / max(
                            refinement.consensus_distance, 1e-6
                        )
                        static_score = math.exp(-min(consensus_ratio, 10.0))
                    else:
                        # Single-instance evidence is deliberately weaker than
                        # multi-instance consensus, but remains usable in a
                        # known-static indoor dataset.
                        static_score = 0.5 * shape_similarity

            frame_gap = (
                0.0
                if frame == int(reference_index)
                else min(
                    1.0,
                    abs(int(frame_indices[frame]) - previous_frame)
                    / max(1, refinement.temporal_max_frame_gap),
                )
            )
            vector = torch.cat(
                [
                    item["center"],
                    item["log_eigenvalues"],
                    item["log_extent"],
                    proposal_translation,
                    torch.tensor(
                        [
                            fitness,
                            min(rmse_ratio, 2.0),
                            shape_similarity,
                            float(item["mean_confidence"]),
                            float(item["mask_area"]),
                            math.log1p(int(item["point_count"])) / 12.0,
                            min(consensus_ratio, 2.0),
                            frame_gap,
                        ]
                    ),
                ]
            )
            geometry[frame, slot] = torch.nan_to_num(vector)
            quality[frame, slot] = torch.tensor(
                [
                    float(scores[frame, slot].clamp(0.0, 1.0)),
                    min(max(geometry_confidence, 0.0), 1.0),
                    min(max(static_score, 0.0), 1.0),
                ]
            )
            identity_rows.append(
                {
                    "sequence_index": frame,
                    "frame_index": int(frame_indices[frame]),
                    "instance_id": int(instance_id),
                    "is_reference": int(frame == int(reference_index)),
                    "mask_observed": int(observed[frame, slot]),
                    "track_confidence": float(scores[frame, slot]),
                    "point_count": int(point_counts[frame, slot]),
                    "identity_valid": int(current_identity_valid),
                    "identity_unknown": int(current_identity_unknown),
                    "identity_mismatch": int(current_identity_mismatch),
                    "identity_state": identity_state,
                    "identity_reason": identity_reason,
                    "registration_accepted": int(
                        proposal.accepted if proposal is not None else current_identity_valid
                    ),
                    "registration_fitness": (
                        float(proposal.fitness)
                        if proposal is not None
                        else (1.0 if current_identity_valid else 0.0)
                    ),
                    "registration_rmse_native": (
                        float(proposal.rmse)
                        if proposal is not None
                        else (0.0 if current_identity_valid else float("nan"))
                    ),
                    "registration_translation_native": (
                        float(torch.linalg.vector_norm(proposal.translation))
                        if proposal is not None
                        else 0.0
                    ),
                    "shape_similarity": float(shape_similarity),
                    "geometry_confidence": float(geometry_confidence),
                    "static_score": float(static_score),
                    "participates_in_consensus": int(
                        int(instance_id) in participating_set
                    ),
                }
            )

        if shared is not None:
            for slot, instance_id in enumerate(instance_ids):
                if int(instance_id) not in participating_set:
                    continue
                proposal = proposal_by_slot.get(slot)
                if proposal is None:
                    continue
                corrected = selected_points[frame][slot] + proposal.translation.float()
                object_maps[int(instance_id)] = merge_map_points(
                    object_maps[int(instance_id)],
                    corrected,
                    max_points=refinement.map_max_points,
                )
            previous_frame = int(frame_indices[frame])

    return {
        "geometry": geometry,
        "quality": quality,
        "observed": observed,
        "identity_valid": identity_valid,
        "identity_unknown": identity_unknown,
        "identity_mismatch": identity_mismatch,
        "identity_diagnostics": identity_rows,
        "point_counts": point_counts,
        "scene_origin": origin,
        "scene_scale": float(scene_scale),
        "geometry_feature_names": list(GEOMETRY_FEATURE_NAMES),
        "quality_names": list(QUALITY_NAMES),
    }


@torch.no_grad()
def build_pose_residual_observations(
    *,
    world_points: torch.Tensor,
    confidence: torch.Tensor,
    masks: torch.Tensor,
    world_to_camera: torch.Tensor,
    intrinsics: torch.Tensor,
    identity_valid: torch.Tensor,
    quality: torch.Tensor,
    geometry: torch.Tensor,
    confidence_threshold: float,
    min_instance_points: int,
    max_map_points: int,
    scene_scale: float,
    min_geometry_confidence: float,
    min_static_score: float,
) -> dict[str, object]:
    """Build causal mask-versus-history projection residuals for pose fusion."""

    world_points = world_points.detach().float().cpu()
    confidence = confidence.detach().float().cpu()
    masks = masks.detach().bool().cpu()
    world_to_camera = world_to_camera.detach().float().cpu()
    intrinsics = intrinsics.detach().float().cpu()
    identity_valid = identity_valid.detach().bool().cpu()
    quality = quality.detach().float().cpu()
    geometry = geometry.detach().float().cpu()
    sequence, instances, height, width = masks.shape
    expected = (sequence, instances)
    if identity_valid.shape != expected or quality.shape[:2] != expected:
        raise ValueError("Pose residual identity/quality tensors disagree with masks.")
    if world_to_camera.shape != (sequence, 3, 4):
        raise ValueError("world_to_camera must have shape [S,3,4].")
    if intrinsics.shape != (sequence, 3, 3):
        raise ValueError("intrinsics must have shape [S,3,3].")

    features = torch.zeros(
        sequence,
        instances,
        len(POSE_RESIDUAL_FEATURE_NAMES),
    )
    valid = torch.zeros(sequence, instances, dtype=torch.bool)
    object_maps: list[torch.Tensor] = [
        torch.empty(0, 3) for _ in range(instances)
    ]
    scale = max(float(scene_scale), 1e-6)
    for frame in range(sequence):
        for slot in range(instances):
            current_points, _ = _masked_points_and_confidence(
                world_points[frame],
                confidence[frame],
                masks[frame, slot],
                confidence_threshold=confidence_threshold,
                max_points=max_map_points,
            )
            if frame == 0 and current_points.shape[0] >= min_instance_points:
                object_maps[slot] = current_points

            observed_xy = _normalized_mask_coordinates(
                masks[frame, slot],
                width=width,
                height=height,
            )
            observed_moments = _moments_2d(observed_xy)
            rays = _mask_rays(
                masks[frame, slot],
                intrinsics[frame],
            )
            ray_mean, ray_variance = _ray_statistics(rays)
            historical = object_maps[slot]
            projected_xy, projected_mask = _project_world_points(
                historical,
                world_to_camera[frame],
                intrinsics[frame],
                height=height,
                width=width,
            )
            projected_moments = _moments_2d(projected_xy)
            observed_mask = masks[frame, slot]
            intersection = int((projected_mask & observed_mask).sum())
            projected_pixels = int(projected_mask.sum())
            union = int((projected_mask | observed_mask).sum())
            coverage = (
                float(intersection) / projected_pixels
                if projected_pixels
                else 0.0
            )
            iou = float(intersection) / union if union else 0.0
            if historical.shape[0] >= 2:
                spread = float(
                    torch.sqrt(
                        torch.linalg.eigvalsh(
                            torch.cov(historical.T)
                        ).clamp_min(0).sum()
                    )
                ) / scale
            else:
                spread = 0.0
            observed_centroid, observed_cov = observed_moments
            projected_centroid, projected_cov = projected_moments
            features[frame, slot] = torch.tensor(
                [
                    *observed_centroid.tolist(),
                    *projected_centroid.tolist(),
                    *(observed_centroid - projected_centroid).tolist(),
                    *observed_cov.tolist(),
                    *projected_cov.tolist(),
                    *(observed_cov - projected_cov).tolist(),
                    float(observed_mask.float().mean()),
                    float(projected_mask.float().mean()),
                    coverage,
                    iou,
                    *ray_mean.tolist(),
                    *ray_variance.tolist(),
                    math.log1p(int(historical.shape[0])) / 12.0,
                    min(spread, 2.0),
                ]
            )
            valid[frame, slot] = (
                observed_xy.shape[0] >= min_instance_points
                and historical.shape[0] >= min_instance_points
            )

            memory_write = (
                bool(identity_valid[frame, slot])
                and current_points.shape[0] >= min_instance_points
                and float(quality[frame, slot, 1])
                >= float(min_geometry_confidence)
                and float(quality[frame, slot, 2])
                >= float(min_static_score)
            )
            if memory_write and frame > 0:
                # The geometry descriptor stores the accepted ICP translation
                # normalized by scene_scale at indices 9:12.
                corrected = (
                    current_points
                    + geometry[frame, slot, 9:12] * scale
                )
                object_maps[slot] = deterministic_limit(
                    torch.cat([historical, corrected], dim=0),
                    max_map_points,
                )
    return {
        "pose_geometry": torch.nan_to_num(features),
        "pose_geometry_valid": valid,
        "pose_geometry_feature_names": list(
            POSE_RESIDUAL_FEATURE_NAMES
        ),
    }


def _normalized_mask_coordinates(
    mask: torch.Tensor,
    *,
    width: int,
    height: int,
) -> torch.Tensor:
    coordinates = torch.nonzero(mask, as_tuple=False).float()
    if not coordinates.numel():
        return torch.empty(0, 2)
    y = 2.0 * coordinates[:, 0] / max(height - 1, 1) - 1.0
    x = 2.0 * coordinates[:, 1] / max(width - 1, 1) - 1.0
    return torch.stack([x, y], dim=-1)


def _moments_2d(
    coordinates: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not coordinates.numel():
        return torch.zeros(2), torch.zeros(3)
    centroid = coordinates.mean(dim=0)
    centered = coordinates - centroid
    covariance = (centered.T @ centered) / max(
        int(coordinates.shape[0]),
        1,
    )
    packed = torch.stack(
        [covariance[0, 0], covariance[1, 1], covariance[0, 1]]
    )
    return centroid, packed


def _mask_rays(
    mask: torch.Tensor,
    intrinsics: torch.Tensor,
) -> torch.Tensor:
    coordinates = torch.nonzero(mask, as_tuple=False).float()
    if not coordinates.numel():
        return torch.empty(0, 3)
    y, x = coordinates.unbind(dim=-1)
    fx = intrinsics[0, 0].clamp_min(1e-6)
    fy = intrinsics[1, 1].clamp_min(1e-6)
    rays = torch.stack(
        [
            (x - intrinsics[0, 2]) / fx,
            (y - intrinsics[1, 2]) / fy,
            torch.ones_like(x),
        ],
        dim=-1,
    )
    return F.normalize(rays, dim=-1)


def _ray_statistics(
    rays: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not rays.numel():
        return torch.zeros(3), torch.zeros(3)
    mean = rays.mean(dim=0)
    variance = (rays - mean).square().mean(dim=0)
    return mean, variance


def _project_world_points(
    points: torch.Tensor,
    world_to_camera: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    height: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    support = torch.zeros(height, width, dtype=torch.bool)
    if not points.numel():
        return torch.empty(0, 2), support
    camera = (
        points @ world_to_camera[:3, :3].T
        + world_to_camera[:3, 3]
    )
    valid = torch.isfinite(camera).all(dim=-1) & (camera[:, 2] > 1e-6)
    camera = camera[valid]
    if not camera.numel():
        return torch.empty(0, 2), support
    u = intrinsics[0, 0] * camera[:, 0] / camera[:, 2]
    u = u + intrinsics[0, 2]
    v = intrinsics[1, 1] * camera[:, 1] / camera[:, 2]
    v = v + intrinsics[1, 2]
    visible = (
        (u >= 0)
        & (u <= width - 1)
        & (v >= 0)
        & (v <= height - 1)
    )
    u = u[visible]
    v = v[visible]
    if not u.numel():
        return torch.empty(0, 2), support
    x_normalized = 2.0 * u / max(width - 1, 1) - 1.0
    y_normalized = 2.0 * v / max(height - 1, 1) - 1.0
    coordinates = torch.stack([x_normalized, y_normalized], dim=-1)
    pixel_x = u.round().long().clamp(0, width - 1)
    pixel_y = v.round().long().clamp(0, height - 1)
    support[pixel_y, pixel_x] = True
    support = F.max_pool2d(
        support[None, None].float(),
        kernel_size=5,
        stride=1,
        padding=2,
    )[0, 0].bool()
    return coordinates, support


@torch.no_grad()
def sample_instance_uvd(
    depth: torch.Tensor,
    depth_confidence: torch.Tensor,
    masks: torch.Tensor,
    quality: torch.Tensor,
    *,
    max_points: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample mask pixels as ``(u,v,depth)`` for rigid consistency loss."""

    depth = depth.detach().float().cpu()
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    depth_confidence = depth_confidence.detach().float().cpu()
    if depth_confidence.ndim == 4 and depth_confidence.shape[-1] == 1:
        depth_confidence = depth_confidence[..., 0]
    masks = masks.detach().bool().cpu()
    sequence, instances, height, width = masks.shape
    uvd = torch.zeros(sequence, instances, max_points, 3)
    valid = torch.zeros(sequence, instances, max_points, dtype=torch.bool)
    weights = torch.zeros(sequence, instances)
    for frame in range(sequence):
        finite = torch.isfinite(depth[frame]) & (depth[frame] > 1e-6)
        finite &= torch.isfinite(depth_confidence[frame])
        for slot in range(instances):
            indices = torch.nonzero(masks[frame, slot] & finite, as_tuple=False)
            if indices.shape[0] == 0:
                continue
            indices = deterministic_limit(indices, max_points).long()
            count = indices.shape[0]
            y, x = indices[:, 0], indices[:, 1]
            uvd[frame, slot, :count, 0] = x.float()
            uvd[frame, slot, :count, 1] = y.float()
            uvd[frame, slot, :count, 2] = depth[frame, y, x]
            valid[frame, slot, :count] = True
            weights[frame, slot] = quality[frame, slot].prod()
    return uvd, valid, weights


def _reference_normalization(
    points: torch.Tensor,
    confidence: torch.Tensor,
    *,
    confidence_threshold: float,
) -> tuple[torch.Tensor, float]:
    valid = (
        torch.isfinite(points).all(dim=-1)
        & torch.isfinite(confidence)
        & (confidence >= float(confidence_threshold))
    )
    selected = points[valid]
    if selected.shape[0] < 128:
        selected = points[torch.isfinite(points).all(dim=-1)]
    if selected.shape[0] == 0:
        return torch.zeros(3), 1.0
    selected = deterministic_limit(selected, 32768)
    origin = torch.quantile(selected, 0.50, dim=0)
    radius = torch.linalg.vector_norm(selected - origin, dim=-1)
    scale = float(torch.quantile(radius, 0.75).clamp_min(1e-3))
    return origin, scale


def _masked_points_and_confidence(
    points: torch.Tensor,
    confidence: torch.Tensor,
    mask: torch.Tensor,
    *,
    confidence_threshold: float,
    max_points: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = (
        mask.bool()
        & torch.isfinite(points).all(dim=-1)
        & torch.isfinite(confidence)
        & (confidence >= float(confidence_threshold))
    )
    selected_points = points[valid]
    selected_confidence = confidence[valid]
    if selected_points.shape[0] > int(max_points):
        indices = torch.linspace(
            0,
            selected_points.shape[0] - 1,
            int(max_points),
        ).round().long()
        selected_points = selected_points.index_select(0, indices)
        selected_confidence = selected_confidence.index_select(0, indices)
    return selected_points, selected_confidence


def _point_statistics(
    points: torch.Tensor,
    confidence: torch.Tensor,
    *,
    origin: torch.Tensor,
    scene_scale: float,
    mask_area: float,
) -> dict[str, torch.Tensor | float | int]:
    if points.shape[0] == 0:
        return {
            "center": torch.zeros(3),
            "log_eigenvalues": torch.zeros(3),
            "log_extent": torch.zeros(3),
            "mean_confidence": 0.0,
            "mask_area": mask_area,
            "point_count": 0,
        }
    center_native = torch.quantile(points, 0.50, dim=0)
    centered = points - points.mean(dim=0)
    covariance = centered.T @ centered / max(1, points.shape[0] - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(1e-8)
    low = torch.quantile(points, 0.05, dim=0)
    high = torch.quantile(points, 0.95, dim=0)
    extent = (high - low).clamp_min(1e-6)
    return {
        "center": (center_native - origin) / float(scene_scale),
        "log_eigenvalues": torch.log(eigenvalues / (float(scene_scale) ** 2) + 1e-6),
        "log_extent": torch.log(extent / float(scene_scale) + 1e-6),
        "mean_confidence": float(confidence.mean()) if confidence.numel() else 0.0,
        "mask_area": mask_area,
        "point_count": int(points.shape[0]),
    }


def _shape_similarity(
    current: torch.Tensor,
    reference: torch.Tensor | None,
) -> float:
    if reference is None:
        return 0.0
    difference = float(torch.mean(torch.abs(current - reference)))
    return math.exp(-min(difference, 10.0))
