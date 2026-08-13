"""Causal static projective ICP candidate for StreamVGGT poses.

This is an isolated feasibility stage, not part of the retained V0 output
path.  It follows the standard RGB-D odometry construction: StreamVGGT depth
is unprojected into per-frame vertex/normal maps, a current frame is associated
projectively with several older frames, and only its world-to-camera SE(3) is
updated by robust point-to-plane LM.  SAM contributes exclusion masks only.

Candidate generation deliberately has no target/GT pose argument.  GT is
accepted only by :func:`score_projective_icp_candidates` after every candidate
pose has been materialised on CPU.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from .edge_directional_gt_audit import local_se3_pose
from .edge_pose_feasibility import (
    _homogeneous_pose,
    _load_exclusion_masks,
    _pose_errors,
)

G0_REVISION = "g0_causal_static_projective_point_to_plane_lm_r1"
VALID_BRANCHES = (
    "full_image",
    "sam_object_excluded",
    "shifted_object_mask_control",
)
FORBIDDEN_GT_FIELDS = {
    "target_pose_encoding",
    "target_world_to_camera",
    "target_depth",
    "target_world_points",
}


@dataclass(frozen=True)
class PyramidConfig:
    downsample_factors: tuple[int, ...] = (4, 2, 1)
    iterations: tuple[int, ...] = (8, 6, 5)


@dataclass(frozen=True)
class AssociationConfig:
    source_offsets: tuple[int, ...] = (1, 2, 4)
    min_depth: float = 0.05
    max_depth: float = 25.0
    min_depth_confidence: float = 0.30
    max_points_per_pair: int = 2048
    min_correspondences: int = 96
    depth_abs_scene_fraction: float = 0.01
    depth_rel_tolerance: float = 0.05
    normal_depth_abs_scene_fraction: float = 0.005
    normal_depth_rel_tolerance: float = 0.03
    min_normal_cosine: float = 0.50
    mask_dilation_px: int = 3


@dataclass(frozen=True)
class SolverConfig:
    huber_delta_scene_fraction: float = 0.005
    initial_damping: float = 1e-3
    damping_up: float = 10.0
    damping_down: float = 0.30
    max_lm_trials: int = 5
    rotation_prior_weight: float = 2e-3
    translation_prior_weight: float = 2e-3
    max_rotation_degrees: float = 2.0
    max_translation_scene_fraction: float = 0.02
    max_step_rotation_degrees: float = 0.50
    max_step_translation_scene_fraction: float = 0.005
    max_design_condition: float = 1e10
    minimum_energy_decrease: float = 1e-9


@dataclass(frozen=True)
class StaticProjectiveICPConfig:
    source_path: Path
    base_config: Path
    output_dir: Path
    clip_name: str
    evaluation_frames: tuple[int, ...]
    branches: tuple[str, ...]
    primary_branch: str
    pyramid: PyramidConfig
    association: AssociationConfig
    solver: SolverConfig
    device: str


@dataclass(frozen=True)
class ICPLevel:
    factor: int
    depth: torch.Tensor
    confidence: torch.Tensor
    intrinsics: torch.Tensor
    vertices: torch.Tensor
    normals: torch.Tensor
    geometry_valid: torch.Tensor
    branch_static: dict[str, torch.Tensor]


@dataclass(frozen=True)
class CorrespondenceSet:
    target_points: torch.Tensor
    source_vertices: torch.Tensor
    source_normals: torch.Tensor
    confidence_weights: torch.Tensor
    source_index: torch.Tensor
    projected_in_bounds: int
    depth_consistent: int


@dataclass(frozen=True)
class ProjectiveICPCandidate:
    target_sequence_index: int
    target_frame_index: int
    branch: str
    active: int
    failure_reason: str
    pose: torch.Tensor
    raw_pose: torch.Tensor
    source_pairs: int
    equal_count_target_points: int
    correspondences: int
    inlier_fraction: float
    initial_geometry_energy: float
    final_geometry_energy: float
    energy_decrease_fraction: float
    accepted_steps: int
    rejected_steps: int
    completed_levels: int
    design_condition: float
    rotation_correction_deg: float
    translation_correction_native: float


def load_static_projective_icp_config(
    path: str | Path,
) -> StaticProjectiveICPConfig:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf8") as handle:
        raw = yaml.safe_load(handle) or {}
    section = raw.get("g0", {})
    frames = section.get("frames", {})
    pyramid = section.get("pyramid", {})
    association = section.get("association", {})
    solver = section.get("solver", {})
    factors = tuple(
        int(value) for value in pyramid.get("downsample_factors", (4, 2, 1))
    )
    iterations = tuple(
        int(value) for value in pyramid.get("iterations", (8, 6, 5))
    )
    config = StaticProjectiveICPConfig(
        source_path=source,
        base_config=_path(section.get("base_config"), source.parent),
        output_dir=_path(
            section.get(
                "output_dir",
                "outputs/streaming_couping_g0_static_projective_icp",
            ),
            source.parent,
        ),
        clip_name=str(section.get("clip_name", "")),
        evaluation_frames=tuple(
            int(value) for value in frames.get("evaluation", ())
        ),
        branches=tuple(
            str(value) for value in section.get("branches", VALID_BRANCHES)
        ),
        primary_branch=str(
            section.get("primary_branch", "sam_object_excluded")
        ),
        pyramid=PyramidConfig(factors, iterations),
        association=AssociationConfig(
            source_offsets=tuple(
                int(value)
                for value in association.get("source_offsets", (1, 2, 4))
            ),
            min_depth=float(association.get("min_depth", 0.05)),
            max_depth=float(association.get("max_depth", 25.0)),
            min_depth_confidence=float(
                association.get("min_depth_confidence", 0.30)
            ),
            max_points_per_pair=int(
                association.get("max_points_per_pair", 2048)
            ),
            min_correspondences=int(
                association.get("min_correspondences", 96)
            ),
            depth_abs_scene_fraction=float(
                association.get("depth_abs_scene_fraction", 0.01)
            ),
            depth_rel_tolerance=float(
                association.get("depth_rel_tolerance", 0.05)
            ),
            normal_depth_abs_scene_fraction=float(
                association.get(
                    "normal_depth_abs_scene_fraction", 0.005
                )
            ),
            normal_depth_rel_tolerance=float(
                association.get("normal_depth_rel_tolerance", 0.03)
            ),
            min_normal_cosine=float(
                association.get("min_normal_cosine", 0.50)
            ),
            mask_dilation_px=int(association.get("mask_dilation_px", 3)),
        ),
        solver=SolverConfig(
            huber_delta_scene_fraction=float(
                solver.get("huber_delta_scene_fraction", 0.005)
            ),
            initial_damping=float(solver.get("initial_damping", 1e-3)),
            damping_up=float(solver.get("damping_up", 10.0)),
            damping_down=float(solver.get("damping_down", 0.30)),
            max_lm_trials=int(solver.get("max_lm_trials", 5)),
            rotation_prior_weight=float(
                solver.get("rotation_prior_weight", 2e-3)
            ),
            translation_prior_weight=float(
                solver.get("translation_prior_weight", 2e-3)
            ),
            max_rotation_degrees=float(
                solver.get("max_rotation_degrees", 2.0)
            ),
            max_translation_scene_fraction=float(
                solver.get("max_translation_scene_fraction", 0.02)
            ),
            max_step_rotation_degrees=float(
                solver.get("max_step_rotation_degrees", 0.50)
            ),
            max_step_translation_scene_fraction=float(
                solver.get("max_step_translation_scene_fraction", 0.005)
            ),
            max_design_condition=float(
                solver.get("max_design_condition", 1e10)
            ),
            minimum_energy_decrease=float(
                solver.get("minimum_energy_decrease", 1e-9)
            ),
        ),
        device=str(section.get("device", "cuda:0")),
    )
    _validate_config(config)
    return config


def objective_payload_from_cache(payload: dict) -> dict:
    """Copy only deployable fields and reject accidental GT propagation."""

    required = (
        "clip_name",
        "frame_indices",
        "image_size",
        "baseline_depth",
        "baseline_depth_confidence",
        "scene_scale",
    )
    output = {key: payload[key] for key in required}
    for key in (
        "associated_tracking_masks_output",
        "trusted_tracking_masks_output",
        "tracking_masks_output",
    ):
        if key in payload:
            output[key] = payload[key]
    if FORBIDDEN_GT_FIELDS & set(output):
        raise RuntimeError("G0 objective payload contains forbidden GT fields.")
    return output


def generate_projective_icp_candidates(
    *,
    payload: dict,
    raw_world_to_camera: torch.Tensor,
    intrinsics: torch.Tensor,
    config: StaticProjectiveICPConfig,
    device: str | torch.device | None = None,
) -> tuple[list[ProjectiveICPCandidate], dict[str, object]]:
    """Generate causal geometry candidates without accepting a GT tensor."""

    leaked = FORBIDDEN_GT_FIELDS & set(payload)
    if leaked:
        raise ValueError(
            f"G0 candidate generation received GT fields={sorted(leaked)}."
        )
    compute_device = torch.device(device or config.device)
    if compute_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"G0 requested {compute_device}, but CUDA is unavailable.")
    frames = tuple(int(value) for value in payload["frame_indices"])
    positions = {frame: index for index, frame in enumerate(frames)}
    try:
        evaluation_indices = [
            positions[frame] for frame in config.evaluation_frames
        ]
    except KeyError as error:
        raise ValueError(f"G0 evaluation frame is absent from cache: {error}")

    depth = _scalar_sequence(payload["baseline_depth"], "baseline_depth")
    confidence = _scalar_sequence(
        payload["baseline_depth_confidence"],
        "baseline_depth_confidence",
    )
    if depth.shape != confidence.shape:
        raise ValueError("G0 depth/confidence shapes differ.")
    raw = _pose_sequence(raw_world_to_camera).detach().float()
    k = _intrinsic_sequence(intrinsics).detach().float()
    if depth.shape[0] != len(frames) or raw.shape[0] != len(frames):
        raise ValueError("G0 cache tensor length does not match frame_indices.")
    if k.shape[0] != len(frames):
        raise ValueError("G0 intrinsics length does not match frame_indices.")
    exclusion, exclusion_source = _load_exclusion_masks(
        payload, size=tuple(int(value) for value in depth.shape[-2:])
    )
    exclusion = _dilate_mask(
        exclusion,
        radius=config.association.mask_dilation_px,
    )
    branch_static = build_branch_static_masks(
        exclusion,
        branches=config.branches,
    )
    area_counts = {
        branch: (~mask).sum(dim=(-2, -1))
        for branch, mask in branch_static.items()
    }
    mask_area_control_exact = bool(
        torch.equal(
            area_counts["sam_object_excluded"],
            area_counts["shifted_object_mask_control"],
        )
    )
    scene_scale = max(abs(float(payload["scene_scale"])), 1e-8)
    levels = build_icp_pyramid(
        depth=depth,
        confidence=confidence,
        intrinsics=k,
        branch_static=branch_static,
        scene_scale=scene_scale,
        config=config,
        device=compute_device,
    )
    raw = raw.to(compute_device)

    candidates: list[ProjectiveICPCandidate] = []
    for target_index in evaluation_indices:
        for branch in config.branches:
            candidates.append(
                optimize_target_pose(
                    target_index=target_index,
                    target_frame=frames[target_index],
                    branch=branch,
                    raw_world_to_camera=raw,
                    levels=levels,
                    scene_scale=scene_scale,
                    config=config,
                )
            )
    audit = {
        "candidate_generation_gt_fields": 0,
        "candidate_generation_target_argument": 0,
        "gt_role": "scoring_only_after_fixed_candidate_generation",
        "pose_output_role": "candidate_only_v0_selected_pose_unchanged",
        "history_direction": "causal_past_only",
        "source_offsets": list(config.association.source_offsets),
        "pyramid_factors": list(config.pyramid.downsample_factors),
        "pyramid_iterations": list(config.pyramid.iterations),
        "solver": "robust_point_to_plane_lm_with_raw_pose_prior",
        "exclusion_mask_source": exclusion_source,
        "exclusion_mask_semantics": (
            "prompted_object_region_not_complete_dynamic_truth"
        ),
        "sam_appearance_tokens_used": 0,
        "equal_target_support_across_branches": 1,
        "mask_area_control_exact": int(mask_area_control_exact),
        "scene_scale": scene_scale,
    }
    return candidates, audit


def build_branch_static_masks(
    exclusion_masks: torch.Tensor,
    *,
    branches: Sequence[str],
) -> dict[str, torch.Tensor]:
    """Build the SAM exclusion and location-matched control masks."""

    exclusion_masks = exclusion_masks.bool()
    # A 180-degree relocation preserves the exact excluded area and shape,
    # unlike a cropped translation.  It is a location control, not another
    # segmentation prediction.
    shifted = torch.flip(exclusion_masks, dims=(-2, -1))
    output = {}
    for branch in branches:
        if branch == "full_image":
            output[branch] = torch.ones_like(exclusion_masks)
        elif branch == "sam_object_excluded":
            output[branch] = ~exclusion_masks
        elif branch == "shifted_object_mask_control":
            output[branch] = ~shifted
        else:
            raise ValueError(f"Unsupported G0 branch {branch!r}.")
    return output


def build_icp_pyramid(
    *,
    depth: torch.Tensor,
    confidence: torch.Tensor,
    intrinsics: torch.Tensor,
    branch_static: dict[str, torch.Tensor],
    scene_scale: float,
    config: StaticProjectiveICPConfig,
    device: torch.device,
) -> tuple[ICPLevel, ...]:
    """Construct camera-space vertex and normal maps at every ICP scale."""

    original_h, original_w = depth.shape[-2:]
    levels = []
    for factor in config.pyramid.downsample_factors:
        h = max(3, round(original_h / factor))
        w = max(3, round(original_w / factor))
        size = (h, w)
        moved_depth = _resize_sequence(depth, size, "bilinear").to(device)
        moved_confidence = _resize_sequence(
            confidence, size, "bilinear"
        ).to(device)
        moved_k = intrinsics.to(device).clone()
        scale_x = w / original_w
        scale_y = h / original_h
        moved_k[:, 0, :] *= scale_x
        moved_k[:, 1, :] *= scale_y
        vertices = depth_to_vertex_map(moved_depth, moved_k)
        normals, normal_valid = vertex_normals(
            vertices,
            moved_depth,
            min_depth=config.association.min_depth,
            max_depth=config.association.max_depth,
            discontinuity_abs=(
                config.association.normal_depth_abs_scene_fraction
                * scene_scale
            ),
            discontinuity_rel=(
                config.association.normal_depth_rel_tolerance
            ),
        )
        depth_valid = (
            torch.isfinite(moved_depth)
            & torch.isfinite(moved_confidence)
            & (moved_depth > config.association.min_depth)
            & (moved_depth < config.association.max_depth)
            & (
                moved_confidence
                >= config.association.min_depth_confidence
            )
        )
        moved_static = {
            branch: _resize_sequence(mask.float(), size, "nearest")
            .to(device)
            .bool()
            for branch, mask in branch_static.items()
        }
        levels.append(
            ICPLevel(
                factor=factor,
                depth=moved_depth,
                confidence=moved_confidence,
                intrinsics=moved_k,
                vertices=vertices,
                normals=normals,
                geometry_valid=depth_valid & normal_valid,
                branch_static=moved_static,
            )
        )
    return tuple(levels)


def depth_to_vertex_map(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
) -> torch.Tensor:
    """Unproject z-depth into camera-space vertices [T,H,W,3]."""

    if depth.ndim != 3 or intrinsics.ndim != 3:
        raise ValueError("depth/K must have shapes [T,H,W] and [T,3,3].")
    _, h, w = depth.shape
    y, x = torch.meshgrid(
        torch.arange(h, device=depth.device, dtype=depth.dtype),
        torch.arange(w, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    x = x[None]
    y = y[None]
    fx = intrinsics[:, 0, 0, None, None]
    fy = intrinsics[:, 1, 1, None, None]
    cx = intrinsics[:, 0, 2, None, None]
    cy = intrinsics[:, 1, 2, None, None]
    return torch.stack(
        (
            (x - cx) * depth / fx.clamp_min(1e-8),
            (y - cy) * depth / fy.clamp_min(1e-8),
            depth,
        ),
        dim=-1,
    )


def vertex_normals(
    vertices: torch.Tensor,
    depth: torch.Tensor,
    *,
    min_depth: float,
    max_depth: float,
    discontinuity_abs: float,
    discontinuity_rel: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Central-difference normals with depth-discontinuity rejection."""

    if vertices.ndim != 4 or vertices.shape[-1] != 3:
        raise ValueError("vertices must have shape [T,H,W,3].")
    normals = torch.zeros_like(vertices)
    dx = vertices[:, 1:-1, 2:] - vertices[:, 1:-1, :-2]
    dy = vertices[:, 2:, 1:-1] - vertices[:, :-2, 1:-1]
    central = torch.linalg.cross(dx, dy, dim=-1)
    norm = torch.linalg.vector_norm(central, dim=-1)
    normal = central / norm[..., None].clamp_min(1e-12)
    normals[:, 1:-1, 1:-1] = normal

    centre = depth[:, 1:-1, 1:-1]
    neighbours = torch.stack(
        (
            depth[:, 1:-1, 2:],
            depth[:, 1:-1, :-2],
            depth[:, 2:, 1:-1],
            depth[:, :-2, 1:-1],
        ),
        dim=-1,
    )
    tolerance = float(discontinuity_abs) + float(discontinuity_rel) * centre.abs()
    valid_centre = (
        torch.isfinite(centre)
        & (centre > min_depth)
        & (centre < max_depth)
    )
    valid_neighbours = (
        torch.isfinite(neighbours).all(dim=-1)
        & (neighbours > min_depth).all(dim=-1)
        & (neighbours < max_depth).all(dim=-1)
        & ((neighbours - centre[..., None]).abs() <= tolerance[..., None]).all(
            dim=-1
        )
    )
    valid = torch.zeros_like(depth, dtype=torch.bool)
    valid[:, 1:-1, 1:-1] = (
        valid_centre
        & valid_neighbours
        & torch.isfinite(normal).all(dim=-1)
        & (norm > 1e-10)
    )
    return normals, valid


def optimize_target_pose(
    *,
    target_index: int,
    target_frame: int,
    branch: str,
    raw_world_to_camera: torch.Tensor,
    levels: Sequence[ICPLevel],
    scene_scale: float,
    config: StaticProjectiveICPConfig,
) -> ProjectiveICPCandidate:
    """Run coarse-to-fine re-associated point-to-plane LM for one frame."""

    raw_pose = _homogeneous_pose(raw_world_to_camera[target_index])
    candidate = raw_pose.clone()
    damping = float(config.solver.initial_damping)
    accepted_steps = 0
    rejected_steps = 0
    completed_levels = 0
    initial_energy = float("nan")
    final_energy = float("nan")
    last_correspondences = 0
    last_inlier_fraction = float("nan")
    last_condition = float("nan")
    last_target_count = 0
    last_pairs = 0
    failure_reason = "fewer_than_min_correspondences"

    for level, iterations in zip(levels, config.pyramid.iterations):
        equal_count = equal_target_support_count(
            level=level,
            target_index=target_index,
            branches=config.branches,
            maximum=config.association.max_points_per_pair,
        )
        last_target_count = equal_count
        if equal_count <= 0:
            continue
        level_completed = False
        for _ in range(iterations):
            correspondences = associate_projective_geometry(
                target_index=target_index,
                branch=branch,
                candidate_target_w2c=candidate,
                raw_world_to_camera=raw_world_to_camera,
                level=level,
                equal_target_count=equal_count,
                scene_scale=scene_scale,
                config=config,
            )
            count = int(correspondences.target_points.shape[0])
            possible = max(
                equal_count
                * min(
                    len(config.association.source_offsets),
                    target_index,
                ),
                1,
            )
            last_correspondences = count
            last_inlier_fraction = count / possible
            last_pairs = int(torch.unique(correspondences.source_index).numel())
            if count < config.association.min_correspondences:
                failure_reason = "fewer_than_min_correspondences"
                break

            residual, jacobian, confidence_weight = linearize_point_to_plane(
                candidate_target_w2c=candidate,
                raw_world_to_camera=raw_world_to_camera,
                correspondences=correspondences,
                target_index=target_index,
                scene_scale=scene_scale,
            )
            robust = huber_irls_weights(
                residual,
                delta=(
                    config.solver.huber_delta_scene_fraction * scene_scale
                ),
            )
            weights = (confidence_weight * robust).clamp_min(1e-8)
            normal_matrix, gradient = weighted_normal_equations(
                jacobian, residual, weights
            )
            state = correction_state(candidate, raw_pose, scene_scale)
            prior_diagonal = candidate.new_tensor(
                [config.solver.rotation_prior_weight] * 3
                + [config.solver.translation_prior_weight] * 3
            )
            normal_matrix = normal_matrix + torch.diag(prior_diagonal)
            gradient = gradient + prior_diagonal * state
            # Indoor point-to-plane geometry is often rank-deficient on a
            # single wall/floor.  Mature odometry keeps those directions tied
            # to the raw-pose prior; reject only if the regularised system is
            # still unusable.
            last_condition = _condition_number(normal_matrix)
            if not math.isfinite(last_condition) or (
                last_condition > config.solver.max_design_condition
            ):
                failure_reason = "ill_conditioned_regularized_system"
                break
            current_energy = point_to_plane_energy(
                candidate_target_w2c=candidate,
                raw_world_to_camera=raw_world_to_camera,
                target_index=target_index,
                correspondences=correspondences,
                scene_scale=scene_scale,
                raw_pose=raw_pose,
                config=config,
            )
            if not math.isfinite(initial_energy):
                initial_energy = float(current_energy)
            accepted = False
            trial_damping = damping
            for _ in range(config.solver.max_lm_trials):
                diagonal = torch.diagonal(normal_matrix).abs().clamp_min(1e-6)
                system = normal_matrix + trial_damping * torch.diag(diagonal)
                try:
                    normalized_step = torch.linalg.solve(system, -gradient)
                except RuntimeError:
                    trial_damping *= config.solver.damping_up
                    rejected_steps += 1
                    continue
                normalized_step = bound_normalized_step(
                    normalized_step,
                    scene_scale=scene_scale,
                    config=config,
                )
                physical_step = torch.cat(
                    (
                        normalized_step[:3],
                        normalized_step[3:] * scene_scale,
                    )
                )
                proposal = local_se3_pose(physical_step, candidate)
                if not within_total_trust_region(
                    proposal,
                    raw_pose,
                    scene_scale=scene_scale,
                    config=config,
                ):
                    trial_damping *= config.solver.damping_up
                    rejected_steps += 1
                    continue
                proposal_energy = point_to_plane_energy(
                    candidate_target_w2c=proposal,
                    raw_world_to_camera=raw_world_to_camera,
                    target_index=target_index,
                    correspondences=correspondences,
                    scene_scale=scene_scale,
                    raw_pose=raw_pose,
                    config=config,
                )
                if (
                    math.isfinite(proposal_energy)
                    and proposal_energy
                    < current_energy - config.solver.minimum_energy_decrease
                ):
                    candidate = proposal
                    final_energy = float(proposal_energy)
                    damping = max(
                        trial_damping * config.solver.damping_down,
                        1e-9,
                    )
                    accepted_steps += 1
                    accepted = True
                    level_completed = True
                    failure_reason = "ok"
                    break
                trial_damping *= config.solver.damping_up
                rejected_steps += 1
            if not accepted:
                damping = trial_damping
                break
        if level_completed:
            completed_levels += 1

    correction_rotation, correction_translation = _pose_errors(
        candidate, raw_pose
    )
    # Report a raw-versus-candidate energy on one frozen, finest-level
    # correspondence set.  Energies observed at different pyramid levels are
    # not directly comparable, so they are deliberately not exposed here.
    finest = levels[-1]
    audit_count = equal_target_support_count(
        level=finest,
        target_index=target_index,
        branches=config.branches,
        maximum=config.association.max_points_per_pair,
    )
    audit_correspondences = associate_projective_geometry(
        target_index=target_index,
        branch=branch,
        candidate_target_w2c=raw_pose,
        raw_world_to_camera=raw_world_to_camera,
        level=finest,
        equal_target_count=audit_count,
        scene_scale=scene_scale,
        config=config,
    )
    if int(audit_correspondences.target_points.shape[0]) >= (
        config.association.min_correspondences
    ):
        initial_energy = point_to_plane_energy(
            candidate_target_w2c=raw_pose,
            raw_world_to_camera=raw_world_to_camera,
            target_index=target_index,
            correspondences=audit_correspondences,
            scene_scale=scene_scale,
            raw_pose=raw_pose,
            config=config,
        )
        final_energy = point_to_plane_energy(
            candidate_target_w2c=candidate,
            raw_world_to_camera=raw_world_to_camera,
            target_index=target_index,
            correspondences=audit_correspondences,
            scene_scale=scene_scale,
            raw_pose=raw_pose,
            config=config,
        )
    else:
        # Do not compare energies inherited from different pyramid levels.
        initial_energy = float("nan")
        final_energy = float("nan")
    decrease = (
        (initial_energy - final_energy) / max(abs(initial_energy), 1e-12)
        if math.isfinite(initial_energy) and math.isfinite(final_energy)
        else float("nan")
    )
    deployable_energy_decreased = bool(
        math.isfinite(initial_energy)
        and math.isfinite(final_energy)
        and final_energy
        < initial_energy - config.solver.minimum_energy_decrease
    )
    active = int(accepted_steps > 0 and deployable_energy_decreased)
    if active:
        failure_reason = "ok"
    elif accepted_steps == 0 and failure_reason == "ok":
        failure_reason = "no_accepted_lm_step"
    elif accepted_steps > 0 and not deployable_energy_decreased:
        failure_reason = "fixed_raw_support_energy_not_decreased"
    return ProjectiveICPCandidate(
        target_sequence_index=target_index,
        target_frame_index=target_frame,
        branch=branch,
        active=active,
        failure_reason=failure_reason,
        pose=candidate.detach().cpu(),
        raw_pose=raw_pose.detach().cpu(),
        source_pairs=last_pairs,
        equal_count_target_points=last_target_count,
        correspondences=last_correspondences,
        inlier_fraction=last_inlier_fraction,
        initial_geometry_energy=initial_energy,
        final_geometry_energy=final_energy,
        energy_decrease_fraction=decrease,
        accepted_steps=accepted_steps,
        rejected_steps=rejected_steps,
        completed_levels=completed_levels,
        design_condition=last_condition,
        rotation_correction_deg=correction_rotation,
        translation_correction_native=correction_translation,
    )


def equal_target_support_count(
    *,
    level: ICPLevel,
    target_index: int,
    branches: Sequence[str],
    maximum: int,
) -> int:
    counts = [
        int(
            (
                level.geometry_valid[target_index]
                & level.branch_static[branch][target_index]
            ).sum()
        )
        for branch in branches
    ]
    return min(min(counts, default=0), int(maximum))


def associate_projective_geometry(
    *,
    target_index: int,
    branch: str,
    candidate_target_w2c: torch.Tensor,
    raw_world_to_camera: torch.Tensor,
    level: ICPLevel,
    equal_target_count: int,
    scene_scale: float,
    config: StaticProjectiveICPConfig,
) -> CorrespondenceSet:
    """Reassociate target vertices to source vertex/normal maps."""

    target_valid = (
        level.geometry_valid[target_index]
        & level.branch_static[branch][target_index]
    )
    selected = deterministic_spatial_indices(target_valid, equal_target_count)
    if selected.numel() == 0:
        return _empty_correspondences(level.depth.device)
    h, w = target_valid.shape
    y = torch.div(selected, w, rounding_mode="floor")
    x = selected.remainder(w)
    target_points = level.vertices[target_index, y, x]
    target_normals = level.normals[target_index, y, x]
    target_confidence = level.confidence[target_index, y, x]

    all_target_points = []
    all_source_vertices = []
    all_source_normals = []
    all_weights = []
    all_source_indices = []
    total_in_bounds = 0
    total_consistent = 0
    target_pose = _homogeneous_pose(candidate_target_w2c)
    target_rotation = target_pose[:3, :3]
    target_translation = target_pose[:3, 3]
    world_points = (
        target_points - target_translation
    ) @ target_rotation
    for offset in config.association.source_offsets:
        source_index = target_index - int(offset)
        if source_index < 0:
            continue
        source_pose = _homogeneous_pose(raw_world_to_camera[source_index])
        source_rotation = source_pose[:3, :3]
        source_translation = source_pose[:3, 3]
        source_points = world_points @ source_rotation.T + source_translation
        projected = source_points @ level.intrinsics[source_index].T
        z = source_points[:, 2]
        u = projected[:, 0] / z.clamp_min(1e-8)
        v = projected[:, 1] / z.clamp_min(1e-8)
        in_bounds = (
            torch.isfinite(source_points).all(dim=-1)
            & torch.isfinite(u)
            & torch.isfinite(v)
            & (z > config.association.min_depth)
            & (u >= 1.0)
            & (u <= w - 2.0)
            & (v >= 1.0)
            & (v <= h - 2.0)
        )
        total_in_bounds += int(in_bounds.sum())
        safe_u = u.clamp(0.0, w - 1.0)
        safe_v = v.clamp(0.0, h - 1.0)
        source_vertex = sample_map(
            level.vertices[source_index], safe_u, safe_v
        )
        source_normal = sample_map(
            level.normals[source_index], safe_u, safe_v
        )
        source_normal = source_normal / torch.linalg.vector_norm(
            source_normal, dim=-1, keepdim=True
        ).clamp_min(1e-12)
        source_depth = sample_map(
            level.depth[source_index], safe_u, safe_v
        )
        source_confidence = sample_map(
            level.confidence[source_index], safe_u, safe_v
        )
        source_valid = sample_map(
            level.geometry_valid[source_index].float(), safe_u, safe_v
        ) > 0.999
        source_static = sample_map(
            level.branch_static[branch][source_index].float(), safe_u, safe_v
        ) > 0.999
        relative_rotation = source_rotation @ target_rotation.T
        moved_target_normals = target_normals @ relative_rotation.T
        normal_cosine = (moved_target_normals * source_normal).sum(dim=-1)
        tolerance = (
            config.association.depth_abs_scene_fraction * scene_scale
            + config.association.depth_rel_tolerance
            * torch.maximum(z.abs(), source_depth.abs())
        )
        consistent = (
            in_bounds
            & source_valid
            & source_static
            & torch.isfinite(source_vertex).all(dim=-1)
            & torch.isfinite(source_normal).all(dim=-1)
            & torch.isfinite(source_confidence)
            & (source_confidence >= config.association.min_depth_confidence)
            & ((z - source_depth).abs() <= tolerance)
            & (normal_cosine >= config.association.min_normal_cosine)
        )
        total_consistent += int(consistent.sum())
        if not bool(consistent.any()):
            continue
        all_target_points.append(target_points[consistent])
        all_source_vertices.append(source_vertex[consistent])
        all_source_normals.append(source_normal[consistent])
        all_weights.append(
            (
                target_confidence[consistent].clamp(0.0, 1.0)
                * source_confidence[consistent].clamp(0.0, 1.0)
            ).sqrt()
        )
        all_source_indices.append(
            torch.full(
                (int(consistent.sum()),),
                source_index,
                dtype=torch.long,
                device=target_points.device,
            )
        )
    if not all_target_points:
        return CorrespondenceSet(
            target_points=target_points.new_empty((0, 3)),
            source_vertices=target_points.new_empty((0, 3)),
            source_normals=target_points.new_empty((0, 3)),
            confidence_weights=target_points.new_empty((0,)),
            source_index=torch.empty(
                0, dtype=torch.long, device=target_points.device
            ),
            projected_in_bounds=total_in_bounds,
            depth_consistent=total_consistent,
        )
    return CorrespondenceSet(
        target_points=torch.cat(all_target_points),
        source_vertices=torch.cat(all_source_vertices),
        source_normals=torch.cat(all_source_normals),
        confidence_weights=torch.cat(all_weights),
        source_index=torch.cat(all_source_indices),
        projected_in_bounds=total_in_bounds,
        depth_consistent=total_consistent,
    )


def deterministic_spatial_indices(
    mask: torch.Tensor,
    count: int,
) -> torch.Tensor:
    """Select a deterministic raster-stratified subset without randomness."""

    indices = torch.nonzero(mask.flatten(), as_tuple=False)[:, 0]
    count = min(int(count), int(indices.numel()))
    if count <= 0:
        return indices[:0]
    if count == indices.numel():
        return indices
    positions = (
        (torch.arange(count, device=indices.device, dtype=torch.float64) + 0.5)
        * indices.numel()
        / count
    ).floor().long().clamp_max(indices.numel() - 1)
    return indices.index_select(0, positions)


def linearize_point_to_plane(
    *,
    candidate_target_w2c: torch.Tensor,
    raw_world_to_camera: torch.Tensor,
    correspondences: CorrespondenceSet,
    target_index: int,
    scene_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Analytic Jacobian for a left target-w2c perturbation.

    Translation columns are expressed in units of ``scene_scale`` so the LM
    system is dimensionally balanced with rotation radians.
    """

    candidate = _homogeneous_pose(candidate_target_w2c)
    target_rotation = candidate[:3, :3]
    target_translation = candidate[:3, 3]
    residual_parts = []
    jacobian_parts = []
    for source_index in torch.unique(correspondences.source_index).tolist():
        selected = correspondences.source_index == int(source_index)
        points = correspondences.target_points[selected]
        vertices = correspondences.source_vertices[selected]
        normals = correspondences.source_normals[selected]
        source = _homogeneous_pose(raw_world_to_camera[int(source_index)])
        relative_rotation = source[:3, :3] @ target_rotation.T
        relative_translation = (
            source[:3, 3]
            - relative_rotation @ target_translation
        )
        transformed = points @ relative_rotation.T + relative_translation
        residual_parts.append(((transformed - vertices) * normals).sum(dim=-1))
        rotation_jacobian = torch.bmm(
            relative_rotation.expand(points.shape[0], -1, -1),
            skew_batch(points),
        )
        point_jacobian = torch.cat(
            (
                rotation_jacobian,
                -relative_rotation.expand(points.shape[0], -1, -1)
                * scene_scale,
            ),
            dim=-1,
        )
        jacobian_parts.append(
            torch.bmm(normals[:, None, :], point_jacobian)[:, 0]
        )
    return (
        torch.cat(residual_parts),
        torch.cat(jacobian_parts),
        correspondences.confidence_weights,
    )


def point_to_plane_energy(
    *,
    candidate_target_w2c: torch.Tensor,
    raw_world_to_camera: torch.Tensor,
    target_index: int,
    correspondences: CorrespondenceSet,
    scene_scale: float,
    raw_pose: torch.Tensor,
    config: StaticProjectiveICPConfig,
) -> float:
    residual, _, confidence = linearize_point_to_plane(
        candidate_target_w2c=candidate_target_w2c,
        raw_world_to_camera=raw_world_to_camera,
        correspondences=correspondences,
        target_index=target_index,
        scene_scale=scene_scale,
    )
    delta = max(
        config.solver.huber_delta_scene_fraction * scene_scale, 1e-8
    )
    absolute = residual.abs()
    robust = torch.where(
        absolute <= delta,
        0.5 * residual.square(),
        delta * (absolute - 0.5 * delta),
    )
    geometry = (confidence * robust).sum() / confidence.sum().clamp_min(1e-8)
    state = correction_state(candidate_target_w2c, raw_pose, scene_scale)
    prior = 0.5 * (
        config.solver.rotation_prior_weight * state[:3].square().sum()
        + config.solver.translation_prior_weight * state[3:].square().sum()
    )
    return float((geometry + prior).detach())


def weighted_normal_equations(
    jacobian: torch.Tensor,
    residual: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    denominator = weights.sum().clamp_min(1e-8)
    weighted_jacobian = jacobian * weights[:, None]
    normal = jacobian.T @ weighted_jacobian / denominator
    gradient = jacobian.T @ (weights * residual) / denominator
    return normal, gradient


def huber_irls_weights(
    residual: torch.Tensor,
    *,
    delta: float,
) -> torch.Tensor:
    delta = max(float(delta), 1e-8)
    absolute = residual.abs()
    return torch.where(
        absolute <= delta,
        torch.ones_like(absolute),
        delta / absolute.clamp_min(1e-12),
    )


def correction_state(
    candidate: torch.Tensor,
    raw_pose: torch.Tensor,
    scene_scale: float,
) -> torch.Tensor:
    correction = _homogeneous_pose(candidate) @ torch.linalg.inv(
        _homogeneous_pose(raw_pose)
    )
    omega = so3_log(correction[:3, :3])
    translation = correction[:3, 3] / max(float(scene_scale), 1e-8)
    return torch.cat((omega, translation))


def so3_log(rotation: torch.Tensor) -> torch.Tensor:
    cosine = ((torch.trace(rotation) - 1.0) * 0.5).clamp(-1.0, 1.0)
    angle = torch.acos(cosine)
    vee = torch.stack(
        (
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        )
    ) * 0.5
    small = vee
    regular = vee * (angle / torch.sin(angle).clamp_min(1e-8))
    return torch.where(angle < 1e-5, small, regular)


def bound_normalized_step(
    step: torch.Tensor,
    *,
    scene_scale: float,
    config: StaticProjectiveICPConfig,
) -> torch.Tensor:
    del scene_scale
    rotation = step[:3]
    translation = step[3:]
    max_rotation = math.radians(config.solver.max_step_rotation_degrees)
    max_translation = config.solver.max_step_translation_scene_fraction
    rotation_scale = min(1.0, max_rotation / max(float(rotation.norm()), 1e-12))
    translation_scale = min(
        1.0, max_translation / max(float(translation.norm()), 1e-12)
    )
    return torch.cat((rotation * rotation_scale, translation * translation_scale))


def within_total_trust_region(
    proposal: torch.Tensor,
    raw_pose: torch.Tensor,
    *,
    scene_scale: float,
    config: StaticProjectiveICPConfig,
) -> bool:
    rotation, translation = _pose_errors(proposal, raw_pose)
    return bool(
        math.isfinite(rotation)
        and math.isfinite(translation)
        and rotation <= config.solver.max_rotation_degrees + 1e-6
        and translation
        <= config.solver.max_translation_scene_fraction * scene_scale + 1e-8
    )


def sample_map(
    image: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Bilinear sample a scalar or channel-last image at pixel UV."""

    h, w = image.shape[:2]
    channels = 1 if image.ndim == 2 else image.shape[-1]
    nchw = (
        image.reshape(1, 1, h, w)
        if image.ndim == 2
        else image.permute(2, 0, 1)[None]
    )
    grid = torch.stack(
        (
            2.0 * u / max(w - 1, 1) - 1.0,
            2.0 * v / max(h - 1, 1) - 1.0,
        ),
        dim=-1,
    ).reshape(1, 1, -1, 2)
    sampled = F.grid_sample(
        nchw.float(),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).reshape(channels, -1).T
    return sampled[:, 0] if image.ndim == 2 else sampled


def skew_batch(points: torch.Tensor) -> torch.Tensor:
    x, y, z = points.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        (
            zero, -z, y,
            z, zero, -x,
            -y, x, zero,
        ),
        dim=-1,
    ).reshape(-1, 3, 3)


def score_projective_icp_candidates(
    candidates: Sequence[ProjectiveICPCandidate],
    *,
    target_world_to_camera: torch.Tensor,
    config: StaticProjectiveICPConfig,
) -> dict[str, object]:
    """Score immutable candidates with native/reference-aligned GT poses."""

    target = _pose_sequence(target_world_to_camera).detach().float().cpu()
    rows = []
    for candidate in candidates:
        target_pose = _homogeneous_pose(
            target[candidate.target_sequence_index]
        )
        raw_rotation, raw_center = _pose_errors(
            candidate.raw_pose, target_pose
        )
        candidate_rotation, candidate_center = _pose_errors(
            candidate.pose, target_pose
        )
        rows.append(
            {
                "target_sequence_index": candidate.target_sequence_index,
                "target_frame_index": candidate.target_frame_index,
                "branch": candidate.branch,
                "active": candidate.active,
                "failure_reason": candidate.failure_reason,
                "source_pairs": candidate.source_pairs,
                "equal_count_target_points": (
                    candidate.equal_count_target_points
                ),
                "correspondences": candidate.correspondences,
                "inlier_fraction": candidate.inlier_fraction,
                "initial_geometry_energy": (
                    candidate.initial_geometry_energy
                ),
                "final_geometry_energy": candidate.final_geometry_energy,
                "energy_decrease_fraction": (
                    candidate.energy_decrease_fraction
                ),
                "accepted_steps": candidate.accepted_steps,
                "rejected_steps": candidate.rejected_steps,
                "completed_levels": candidate.completed_levels,
                "design_condition": candidate.design_condition,
                "rotation_correction_deg": (
                    candidate.rotation_correction_deg
                ),
                "translation_correction_native": (
                    candidate.translation_correction_native
                ),
                "raw_rotation_error_deg": raw_rotation,
                "candidate_rotation_error_deg": candidate_rotation,
                "rotation_gain_deg": raw_rotation - candidate_rotation,
                "rotation_worse": int(candidate_rotation >= raw_rotation),
                "raw_center_error_native": raw_center,
                "candidate_center_error_native": candidate_center,
                "center_gain_native": raw_center - candidate_center,
                "center_worse": int(candidate_center >= raw_center),
                "frame_pass": int(
                    candidate.active
                    and candidate.final_geometry_energy
                    < candidate.initial_geometry_energy
                    and candidate_rotation < raw_rotation
                    and candidate_center < raw_center
                ),
            }
        )
    folds = summarize_folds(rows, config=config)
    decision = summarize_decision(folds, config=config)
    return {"rows": rows, "fold_summary": folds, "decision": decision}


def summarize_folds(
    rows: Sequence[dict[str, object]],
    *,
    config: StaticProjectiveICPConfig,
) -> list[dict[str, object]]:
    frame_to_fold = {
        frame: fold
        for fold, values in zip(
            ("short", "medium", "long"),
            (
                config.evaluation_frames[:4],
                config.evaluation_frames[4:8],
                config.evaluation_frames[8:12],
            ),
        )
        for frame in values
    }
    summary = []
    for branch in config.branches:
        for fold in ("short", "medium", "long"):
            selected = [
                row
                for row in rows
                if row["branch"] == branch
                and frame_to_fold[int(row["target_frame_index"])] == fold
            ]
            active = [row for row in selected if int(row["active"]) == 1]
            raw_rotation = _mean(active, "raw_rotation_error_deg")
            candidate_rotation = _mean(active, "candidate_rotation_error_deg")
            raw_center = _mean(active, "raw_center_error_native")
            candidate_center = _mean(active, "candidate_center_error_native")
            rotation_worse = sum(int(row["rotation_worse"]) for row in active)
            center_worse = sum(int(row["center_worse"]) for row in active)
            energy_pass = all(
                float(row["final_geometry_energy"])
                < float(row["initial_geometry_energy"])
                for row in active
            )
            fold_pass = bool(
                len(active) == 4
                and energy_pass
                and candidate_rotation < raw_rotation
                and candidate_center < raw_center
                and rotation_worse == 0
                and center_worse == 0
            )
            summary.append(
                {
                    "fold": fold,
                    "branch": branch,
                    "frames": " ".join(
                        str(row["target_frame_index"]) for row in selected
                    ),
                    "active_frames": len(active),
                    "mean_correspondences": _mean(active, "correspondences"),
                    "mean_inlier_fraction": _mean(active, "inlier_fraction"),
                    "mean_energy_decrease_fraction": _mean(
                        active, "energy_decrease_fraction"
                    ),
                    "mean_accepted_steps": _mean(active, "accepted_steps"),
                    "raw_rotation_error_deg": raw_rotation,
                    "candidate_rotation_error_deg": candidate_rotation,
                    "rotation_gain_deg": raw_rotation - candidate_rotation,
                    "rotation_worse_frames": rotation_worse,
                    "raw_center_error_native": raw_center,
                    "candidate_center_error_native": candidate_center,
                    "center_gain_native": raw_center - candidate_center,
                    "center_worse_frames": center_worse,
                    "geometry_energy_all_decreased": int(
                        bool(active) and energy_pass
                    ),
                    "fold_pass": int(fold_pass),
                }
            )
    return summary


def summarize_decision(
    folds: Sequence[dict[str, object]],
    *,
    config: StaticProjectiveICPConfig,
) -> dict[str, object]:
    branch_pass = {
        branch: int(
            len(selected := [row for row in folds if row["branch"] == branch])
            == 3
            and all(int(row["fold_pass"]) == 1 for row in selected)
        )
        for branch in config.branches
    }
    sam_unique = bool(branch_pass[config.primary_branch])
    primary_rows = [
        row for row in folds if row["branch"] == config.primary_branch
    ]
    shifted_rows = {
        row["fold"]: row
        for row in folds
        if row["branch"] == "shifted_object_mask_control"
    }
    full_rows = {
        row["fold"]: row
        for row in folds
        if row["branch"] == "full_image"
    }
    for primary in primary_rows:
        controls = (
            shifted_rows.get(primary["fold"]),
            full_rows.get(primary["fold"]),
        )
        sam_unique = bool(
            sam_unique
            and all(control is not None for control in controls)
            and all(
                float(primary["rotation_gain_deg"])
                > float(control["rotation_gain_deg"])
                and float(primary["center_gain_native"])
                > float(control["center_gain_native"])
                for control in controls
                if control is not None
            )
        )
    return {
        "branch_all_fold_pass": branch_pass,
        "any_geometry_branch_all_fold_pass": int(any(branch_pass.values())),
        "primary_branch": config.primary_branch,
        "primary_all_fold_pass": branch_pass[config.primary_branch],
        "sam_object_exclusion_unique_pass": int(sam_unique),
        "candidate_generation_gt_fields": 0,
        "gt_role": "scoring_only_after_fixed_candidate_generation",
        "selected_pose_modified": 0,
        "v0_selected_pose": "raw_streamvggt_unchanged",
        "claim": "candidate_feasibility_not_deployed_pose_improvement",
    }


def _empty_correspondences(device: torch.device) -> CorrespondenceSet:
    return CorrespondenceSet(
        target_points=torch.empty(0, 3, device=device),
        source_vertices=torch.empty(0, 3, device=device),
        source_normals=torch.empty(0, 3, device=device),
        confidence_weights=torch.empty(0, device=device),
        source_index=torch.empty(0, dtype=torch.long, device=device),
        projected_in_bounds=0,
        depth_consistent=0,
    )


def _condition_number(matrix: torch.Tensor) -> float:
    values = torch.linalg.eigvalsh(matrix.double())
    maximum = float(values[-1].abs())
    minimum = float(values[0].abs())
    return maximum / max(minimum, 1e-15)


def _scalar_sequence(value: object, name: str) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise ValueError(f"G0 cache lacks tensor field {name!r}.")
    output = value.detach().float().cpu()
    if output.ndim == 4 and output.shape[-1] == 1:
        output = output[..., 0]
    if output.ndim != 3:
        raise ValueError(f"{name} must have shape [T,H,W] or [T,H,W,1].")
    return output


def _pose_sequence(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 4:
        if value.shape[0] != 1:
            raise ValueError("G0 pose batch must have size one.")
        value = value[0]
    if value.ndim != 3 or value.shape[-2:] not in {(3, 4), (4, 4)}:
        raise ValueError("G0 poses must have shape [T,3,4] or [T,4,4].")
    return value


def _intrinsic_sequence(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 4:
        if value.shape[0] != 1:
            raise ValueError("G0 intrinsic batch must have size one.")
        value = value[0]
    if value.ndim != 3 or value.shape[-2:] != (3, 3):
        raise ValueError("G0 intrinsics must have shape [T,3,3].")
    return value


def _resize_sequence(
    value: torch.Tensor,
    size: tuple[int, int],
    mode: str,
) -> torch.Tensor:
    if value.shape[-2:] == size:
        return value.float()
    kwargs = {"size": size, "mode": mode}
    if mode in {"bilinear", "bicubic"}:
        kwargs["align_corners"] = False
    return F.interpolate(value[:, None].float(), **kwargs)[:, 0]


def _dilate_mask(mask: torch.Tensor, *, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask.bool()
    return F.max_pool2d(
        mask.float()[:, None],
        kernel_size=2 * radius + 1,
        stride=1,
        padding=radius,
    )[:, 0].bool()


def _mean(rows: Sequence[dict[str, object]], key: str) -> float:
    values = [
        float(row[key])
        for row in rows
        if math.isfinite(float(row[key]))
    ]
    return sum(values) / len(values) if values else float("nan")


def _path(value: str | Path | None, base: Path) -> Path:
    if value is None:
        raise ValueError("Missing required G0 path.")
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _validate_config(config: StaticProjectiveICPConfig) -> None:
    if not config.clip_name:
        raise ValueError("g0.clip_name is required.")
    if len(config.evaluation_frames) != 12:
        raise ValueError("G0 requires exactly twelve evaluation frames.")
    if tuple(config.branches) != VALID_BRANCHES:
        raise ValueError(
            "G0 branches must be full_image, sam_object_excluded, "
            "shifted_object_mask_control in that order."
        )
    if config.primary_branch != "sam_object_excluded":
        raise ValueError("G0 primary branch must be sam_object_excluded.")
    if len(config.pyramid.downsample_factors) != len(
        config.pyramid.iterations
    ):
        raise ValueError("G0 pyramid factors/iterations lengths differ.")
    if not config.pyramid.downsample_factors or any(
        value <= 0 for value in config.pyramid.downsample_factors
    ):
        raise ValueError("G0 pyramid factors must be positive.")
    if tuple(sorted(config.pyramid.downsample_factors, reverse=True)) != (
        config.pyramid.downsample_factors
    ):
        raise ValueError("G0 pyramid must run coarse to fine.")
    if any(value <= 0 for value in config.pyramid.iterations):
        raise ValueError("G0 pyramid iterations must be positive.")
    if not config.association.source_offsets or any(
        value <= 0 for value in config.association.source_offsets
    ):
        raise ValueError("G0 source offsets must be positive causal lags.")
    if config.association.min_correspondences < 6:
        raise ValueError("G0 needs at least six correspondences.")
    if config.association.max_points_per_pair < (
        config.association.min_correspondences
    ):
        raise ValueError("G0 max points must exceed min correspondences.")
    if not (0.0 <= config.association.min_normal_cosine <= 1.0):
        raise ValueError("G0 min_normal_cosine must be in [0,1].")
    if config.solver.max_rotation_degrees <= 0.0:
        raise ValueError("G0 rotation trust region must be positive.")
    if config.solver.max_translation_scene_fraction <= 0.0:
        raise ValueError("G0 translation trust region must be positive.")
    if config.solver.max_lm_trials <= 0:
        raise ValueError("G0 max_lm_trials must be positive.")
