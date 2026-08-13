"""Causal edge projection audit and pose-basin feasibility inputs.

This module is intentionally independent from the retained V0 baseline.  It
reads a frozen V0 feature cache and measures whether the raw StreamVGGT pose,
depth, intrinsics, RGB edges, and SAM-derived masks form a coherent edge
reprojection objective.  It does not change selected poses and does not train
or instantiate a pose head.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
import yaml
from PIL import Image


EDGE_FEASIBILITY_REVISION = "e0_differentiable_edge_pose_basin_r2"


@dataclass(frozen=True)
class EdgeConfig:
    sobel_quantile: float = 0.88
    max_edges_per_frame: int = 12_000
    distance_truncate_px: int = 12
    huber_delta_px: float = 2.0
    mask_dilation_px: int = 3


@dataclass(frozen=True)
class ProjectionConfig:
    source_offsets: tuple[int, ...] = (1, 2, 4)
    min_depth: float = 0.05
    max_depth: float = 25.0
    min_depth_confidence: float = 0.30
    depth_abs_tolerance: float = 0.05
    depth_rel_tolerance: float = 0.05
    boundary_penalty_weight: float = 1.0
    min_locked_points_per_direction: int = 64
    max_locked_points_per_direction: int = 2048


@dataclass(frozen=True)
class RecoveryConfig:
    device: str = "cuda:0"
    axes: tuple[str, ...] = ("x", "y", "z")
    signs: tuple[int, ...] = (-1, 1)
    rotation_degrees: tuple[float, ...] = (1.0,)
    translation_scene_fractions: tuple[float, ...] = (0.01,)
    optimizer_steps: int = 120
    learning_rate: float = 0.01
    gradient_probe_step: float = 0.001
    max_rotation_degrees: float = 3.0
    max_translation_scene_fraction: float = 0.03
    min_loss_order_pass_rate: float = 0.90
    min_mean_recovery_fraction: float = 0.50


@dataclass(frozen=True)
class EdgeFeasibilityConfig:
    source_path: Path
    base_config: Path
    output_dir: Path
    clip_name: str
    evaluation_frames: tuple[int, ...]
    branches: tuple[str, ...]
    edge: EdgeConfig
    projection: ProjectionConfig
    recovery: RecoveryConfig


VALID_BRANCHES = {
    "raw",
    "all_edge",
    "sam_object_excluded_edge",
    "shifted_object_mask_control",
}


def load_edge_feasibility_config(path: str | Path) -> EdgeFeasibilityConfig:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf8") as handle:
        raw = yaml.safe_load(handle) or {}
    section = raw.get("e0", {})
    edge = section.get("edge", {})
    projection = section.get("projection", {})
    recovery = section.get("recovery", {})
    frames = section.get("frames", {})
    config = EdgeFeasibilityConfig(
        source_path=source,
        base_config=_path(section.get("base_config"), source.parent),
        output_dir=_path(
            section.get(
                "output_dir",
                "outputs/streaming_couping_e0_edge_feasibility",
            ),
            source.parent,
        ),
        clip_name=str(section.get("clip_name", "")),
        evaluation_frames=_int_tuple(frames.get("evaluation", ())),
        branches=tuple(str(value) for value in section.get("branches", ())),
        edge=EdgeConfig(
            sobel_quantile=float(edge.get("sobel_quantile", 0.88)),
            max_edges_per_frame=int(edge.get("max_edges_per_frame", 12_000)),
            distance_truncate_px=int(edge.get("distance_truncate_px", 12)),
            huber_delta_px=float(edge.get("huber_delta_px", 2.0)),
            mask_dilation_px=int(edge.get("mask_dilation_px", 3)),
        ),
        projection=ProjectionConfig(
            source_offsets=_int_tuple(
                projection.get("source_offsets", (1, 2, 4))
            ),
            min_depth=float(projection.get("min_depth", 0.05)),
            max_depth=float(projection.get("max_depth", 25.0)),
            min_depth_confidence=float(
                projection.get("min_depth_confidence", 0.30)
            ),
            depth_abs_tolerance=float(
                projection.get("depth_abs_tolerance", 0.05)
            ),
            depth_rel_tolerance=float(
                projection.get("depth_rel_tolerance", 0.05)
            ),
            boundary_penalty_weight=float(
                projection.get("boundary_penalty_weight", 1.0)
            ),
            min_locked_points_per_direction=int(
                projection.get("min_locked_points_per_direction", 64)
            ),
            max_locked_points_per_direction=int(
                projection.get("max_locked_points_per_direction", 2048)
            ),
        ),
        recovery=RecoveryConfig(
            device=str(recovery.get("device", "cuda:0")),
            axes=tuple(str(value).lower() for value in recovery.get("axes", ("x", "y", "z"))),
            signs=tuple(int(value) for value in recovery.get("signs", (-1, 1))),
            rotation_degrees=tuple(
                float(value) for value in recovery.get("rotation_degrees", (1.0,))
            ),
            translation_scene_fractions=tuple(
                float(value)
                for value in recovery.get("translation_scene_fractions", (0.01,))
            ),
            optimizer_steps=int(recovery.get("optimizer_steps", 120)),
            learning_rate=float(recovery.get("learning_rate", 0.01)),
            gradient_probe_step=float(recovery.get("gradient_probe_step", 0.001)),
            max_rotation_degrees=float(recovery.get("max_rotation_degrees", 3.0)),
            max_translation_scene_fraction=float(
                recovery.get("max_translation_scene_fraction", 0.03)
            ),
            min_loss_order_pass_rate=float(
                recovery.get("min_loss_order_pass_rate", 0.90)
            ),
            min_mean_recovery_fraction=float(
                recovery.get("min_mean_recovery_fraction", 0.50)
            ),
        ),
    )
    _validate_config(config)
    return config


def run_edge_feasibility(
    *,
    payload: dict,
    world_to_camera: torch.Tensor,
    intrinsics: torch.Tensor,
    config: EdgeFeasibilityConfig,
) -> dict[str, object]:
    """Evaluate projection coherence and the local all-edge pose basin."""

    frames = tuple(int(value) for value in payload["frame_indices"])
    frame_to_index = {frame: index for index, frame in enumerate(frames)}
    evaluation_indices = [frame_to_index[frame] for frame in config.evaluation_frames]
    images = _load_gray_images(
        payload["image_paths"],
        image_size=_image_size(payload),
    )
    depth = _tensor_field(payload, "baseline_depth").float()
    depth_confidence = _tensor_field(payload, "baseline_depth_confidence").float()
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth_confidence.ndim == 4 and depth_confidence.shape[-1] == 1:
        depth_confidence = depth_confidence[..., 0]
    if world_to_camera.ndim == 4:
        world_to_camera = world_to_camera[0]
    if intrinsics.ndim == 4:
        intrinsics = intrinsics[0]
    if world_to_camera.shape[0] != len(frames):
        raise ValueError("world_to_camera length does not match frame_indices.")
    if intrinsics.shape[0] != len(frames):
        raise ValueError("intrinsics length does not match frame_indices.")
    if depth.shape[-2:] != images.shape[-2:]:
        depth = _resize_scalar(depth[:, None], images.shape[-2:])[:, 0]
        depth_confidence = _resize_scalar(
            depth_confidence[:, None],
            images.shape[-2:],
        )[:, 0]

    edge_strength = sobel_magnitude(images)
    all_edges = threshold_edges(
        edge_strength,
        quantile=config.edge.sobel_quantile,
        max_edges_per_frame=config.edge.max_edges_per_frame,
    )
    exclusion_masks, exclusion_source = _load_exclusion_masks(
        payload,
        size=images.shape[-2:],
    )
    if config.edge.mask_dilation_px:
        exclusion_masks = dilate_masks(
            exclusion_masks,
            radius=config.edge.mask_dilation_px,
        )
    branch_edges = equal_count_branch_edges(
        all_edges=all_edges,
        edge_strength=edge_strength,
        exclusion_masks=exclusion_masks,
        branches=tuple(branch for branch in config.branches if branch != "raw"),
    )
    distance_fields = {
        branch: torch.stack(
            [
                truncated_distance_transform(
                    edge_map,
                    max_distance=config.edge.distance_truncate_px,
                )
                for edge_map in edges
            ],
            dim=0,
        )
        for branch, edges in branch_edges.items()
    }

    rows: list[dict[str, object]] = []
    for target_index in evaluation_indices:
        for offset in config.projection.source_offsets:
            source_index = target_index - int(offset)
            if source_index < 0:
                continue
            for branch in config.branches:
                if branch == "raw":
                    rows.append(
                        _raw_row(
                            frames=frames,
                            source_index=source_index,
                            target_index=target_index,
                            offset=offset,
                        )
                    )
                    continue
                row = pair_edge_projection_metrics(
                    frames=frames,
                    source_index=source_index,
                    target_index=target_index,
                    offset=offset,
                    branch=branch,
                    source_edges=branch_edges[branch][source_index],
                    target_distance=distance_fields[branch][target_index],
                    depth=depth,
                    depth_confidence=depth_confidence,
                    world_to_camera=world_to_camera,
                    intrinsics=intrinsics,
                    projection=config.projection,
                )
                rows.append(row)
    recovery_rows = run_perturbation_recovery(
        frames=frames,
        evaluation_indices=evaluation_indices,
        all_edges=all_edges,
        depth=depth,
        depth_confidence=depth_confidence,
        world_to_camera=world_to_camera,
        intrinsics=intrinsics,
        scene_scale=float(payload.get("scene_scale", 1.0)),
        config=config,
    )
    return {
        "revision": EDGE_FEASIBILITY_REVISION,
        "clip": payload["clip_name"],
        "frames": frames,
        "evaluation_frames": config.evaluation_frames,
        "branches": config.branches,
        "rows": rows,
        "summary": summarize_rows(rows),
        "exclusion_mask_source": exclusion_source,
        "exclusion_mask_semantics": "prompted_object_region_not_dynamic_truth",
        "recovery_rows": recovery_rows,
        "recovery_summary": summarize_recovery(recovery_rows, config.recovery),
    }


def pair_edge_projection_metrics(
    *,
    frames: Sequence[int],
    source_index: int,
    target_index: int,
    offset: int,
    branch: str,
    source_edges: torch.Tensor,
    target_distance: torch.Tensor,
    depth: torch.Tensor,
    depth_confidence: torch.Tensor,
    world_to_camera: torch.Tensor,
    intrinsics: torch.Tensor,
    projection: ProjectionConfig,
) -> dict[str, object]:
    """Audit a raw-pose projection without claiming a true geometric cycle."""

    h, w = source_edges.shape
    ys, xs = torch.nonzero(source_edges.bool(), as_tuple=True)
    total_edges = int(xs.numel())
    if total_edges == 0:
        return _empty_projection_row(
            frames, source_index, target_index, offset, branch,
            source_edge_pixels=0,
        )

    source_depth = depth[source_index, ys, xs].float()
    source_conf = depth_confidence[source_index, ys, xs].float()
    positive = (
        torch.isfinite(source_depth)
        & torch.isfinite(source_conf)
        & (source_depth > float(projection.min_depth))
        & (source_depth < float(projection.max_depth))
        & (source_conf >= float(projection.min_depth_confidence))
    )
    if not bool(positive.any()):
        return _empty_projection_row(
            frames, source_index, target_index, offset, branch,
            source_edge_pixels=total_edges,
            in_bounds_rate=0.0,
        )

    xs = xs[positive].float()
    ys = ys[positive].float()
    source_depth = source_depth[positive]
    source_conf = source_conf[positive]
    uvz = project_depth_points(
        x=xs,
        y=ys,
        depth=source_depth,
        source_w2c=world_to_camera[source_index],
        target_w2c=world_to_camera[target_index],
        source_k=intrinsics[source_index],
        target_k=intrinsics[target_index],
    )
    u, v, z = uvz[:, 0], uvz[:, 1], uvz[:, 2]
    in_bounds = (
        torch.isfinite(u)
        & torch.isfinite(v)
        & torch.isfinite(z)
        & (z > float(projection.min_depth))
        & (u >= 0)
        & (v >= 0)
        & (u <= w - 1)
        & (v <= h - 1)
    )
    projected_count = int(in_bounds.sum())
    if projected_count == 0:
        row = _empty_projection_row(
            frames, source_index, target_index, offset, branch,
            source_edge_pixels=total_edges,
            in_bounds_rate=0.0,
        )
        row["source_positive_depth_pixels"] = int(source_depth.numel())
        row["mean_depth_confidence"] = float(source_conf.mean())
        return row

    u_in = u[in_bounds]
    v_in = v[in_bounds]
    z_in = z[in_bounds].float()
    target_depth = bilinear_sample_image(depth[target_index], u_in, v_in)
    target_conf = bilinear_sample_image(
        depth_confidence[target_index], u_in, v_in
    )
    target_valid = (
        torch.isfinite(target_depth)
        & torch.isfinite(target_conf)
        & (target_depth > float(projection.min_depth))
        & (target_depth < float(projection.max_depth))
        & (target_conf >= float(projection.min_depth_confidence))
    )
    tolerance = float(projection.depth_abs_tolerance) + float(
        projection.depth_rel_tolerance
    ) * torch.maximum(z_in.abs(), target_depth.abs())
    consistent = target_valid & ((target_depth - z_in).abs() <= tolerance)
    distances = bilinear_sample_image(target_distance, u_in, v_in)
    valid_distances = distances[consistent]
    mean_distance = (
        float(valid_distances.mean()) if valid_distances.numel() else float("nan")
    )
    median_distance = (
        float(valid_distances.median()) if valid_distances.numel() else float("nan")
    )
    return {
        **_row_base(frames, source_index, target_index, offset, branch),
        "source_edge_pixels": total_edges,
        "source_positive_depth_pixels": int(source_depth.numel()),
        "projected_in_bounds_pixels": projected_count,
        "depth_consistency_pass_pixels": int(consistent.sum()),
        "valid_projected_pixels": int(valid_distances.numel()),
        "in_bounds_rate": float(projected_count / max(int(source_depth.numel()), 1)),
        "depth_consistency_pass_rate": float(
            consistent.sum() / max(projected_count, 1)
        ),
        "mean_truncated_edge_distance_px": mean_distance,
        "median_truncated_edge_distance_px": median_distance,
        "mean_depth_confidence": float(source_conf.mean()),
    }


def bilinear_sample_image(
    image: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Sample a scalar image at pixel coordinates with UV gradients intact."""

    if image.ndim != 2:
        raise ValueError("image must have shape [H,W].")
    if u.shape != v.shape:
        raise ValueError("u and v must have identical shapes.")
    if u.numel() == 0:
        return image.new_empty(u.shape)
    h, w = image.shape
    x = 2.0 * u / max(w - 1, 1) - 1.0
    y = 2.0 * v / max(h - 1, 1) - 1.0
    grid = torch.stack((x, y), dim=-1).reshape(1, 1, -1, 2)
    sampled = F.grid_sample(
        image.float().reshape(1, 1, h, w),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled.reshape(-1).reshape(u.shape)


def run_perturbation_recovery(
    *,
    frames: Sequence[int],
    evaluation_indices: Sequence[int],
    all_edges: torch.Tensor,
    depth: torch.Tensor,
    depth_confidence: torch.Tensor,
    world_to_camera: torch.Tensor,
    intrinsics: torch.Tensor,
    scene_scale: float,
    config: EdgeFeasibilityConfig,
) -> list[dict[str, object]]:
    """Probe and optimize a locked, bidirectional all-edge objective.

    Raw StreamVGGT pose is used only as the centre of deterministic synthetic
    perturbations.  No GT pose, correspondence, or scoring field is read.
    """

    device = torch.device(config.recovery.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"E0 recovery requested {device}, but CUDA is unavailable."
        )
    recovery_edges = all_edges.detach().bool().cpu()
    distance_fields = torch.stack(
        [
            truncated_distance_transform(
                edge_map,
                max_distance=config.edge.distance_truncate_px,
            )
            for edge_map in recovery_edges
        ],
        dim=0,
    )
    depth = depth.detach().float().cpu()
    depth_confidence = depth_confidence.detach().float().cpu()
    world_to_camera = world_to_camera.detach().float().cpu()
    intrinsics = intrinsics.detach().float().cpu()
    translation_scene_scale = max(abs(float(scene_scale)), 1e-6)
    moved_distances = distance_fields.to(device)
    moved_w2c = world_to_camera.to(device)
    moved_k = intrinsics.to(device)
    rows: list[dict[str, object]] = []
    for target_index in evaluation_indices:
        locked_pairs = _lock_target_support(
            target_index=int(target_index),
            source_offsets=config.projection.source_offsets,
            all_edges=recovery_edges,
            depth=depth,
            depth_confidence=depth_confidence,
            world_to_camera=world_to_camera,
            intrinsics=intrinsics,
            projection=config.projection,
        )
        locked_directions = 2 * len(locked_pairs)
        locked_points = sum(
            int(pair[direction]["x"].numel())
            for pair in locked_pairs
            for direction in ("forward", "reverse")
        )
        common = {
            "target_sequence_index": int(target_index),
            "target_frame_index": int(frames[target_index]),
            "locked_pairs": len(locked_pairs),
            "locked_directions": locked_directions,
            "locked_points": locked_points,
        }
        trials = _perturbation_trials(config.recovery, translation_scene_scale)
        if not locked_pairs:
            for trial in trials:
                rows.append(
                    {
                        **common,
                        **trial,
                        **_inactive_recovery_metrics(
                            "fewer_than_min_locked_points"
                        ),
                    }
                )
            continue

        moved_pairs = _move_locked_pairs(locked_pairs, device)
        raw_target = _homogeneous_pose(moved_w2c[target_index])
        max_rotation = math.radians(config.recovery.max_rotation_degrees)
        max_translation = (
            config.recovery.max_translation_scene_fraction
            * translation_scene_scale
        )
        with torch.no_grad():
            raw_loss = float(
                _locked_target_loss(
                    raw_target,
                    target_index=target_index,
                    locked_pairs=moved_pairs,
                    distance_fields=moved_distances,
                    world_to_camera=moved_w2c,
                    intrinsics=moved_k,
                    projection=config.projection,
                    edge=config.edge,
                )
            )
        for trial in trials:
            initial_parameter = _trial_parameter(
                trial,
                max_rotation=max_rotation,
                max_translation=max_translation,
                device=device,
            )
            parameter = initial_parameter.clone().requires_grad_(True)
            perturbed_pose = _bounded_local_pose(
                parameter,
                raw_target,
                max_rotation=max_rotation,
                max_translation=max_translation,
            )
            perturbed_loss_tensor = _locked_target_loss(
                perturbed_pose,
                target_index=target_index,
                locked_pairs=moved_pairs,
                distance_fields=moved_distances,
                world_to_camera=moved_w2c,
                intrinsics=moved_k,
                projection=config.projection,
                edge=config.edge,
            )
            perturbed_loss_tensor.backward()
            gradient = parameter.grad.detach().clone()
            gradient_norm = float(torch.linalg.vector_norm(gradient))
            perturbed_loss = float(perturbed_loss_tensor.detach())
            with torch.no_grad():
                if gradient_norm > 0.0 and math.isfinite(gradient_norm):
                    probe_parameter = parameter.detach() - (
                        float(config.recovery.gradient_probe_step)
                        * gradient
                        / gradient.norm().clamp_min(1e-12)
                    )
                    probe_pose = _bounded_local_pose(
                        probe_parameter,
                        raw_target,
                        max_rotation=max_rotation,
                        max_translation=max_translation,
                    )
                    probe_loss = float(
                        _locked_target_loss(
                            probe_pose,
                            target_index=target_index,
                            locked_pairs=moved_pairs,
                            distance_fields=moved_distances,
                            world_to_camera=moved_w2c,
                            intrinsics=moved_k,
                            projection=config.projection,
                            edge=config.edge,
                        )
                    )
                else:
                    probe_loss = float("nan")

            parameter = initial_parameter.clone().requires_grad_(True)
            optimizer = torch.optim.Adam(
                [parameter], lr=float(config.recovery.learning_rate)
            )
            best_loss = perturbed_loss
            best_parameter = parameter.detach().clone()
            completed_steps = 0
            for step in range(int(config.recovery.optimizer_steps)):
                optimizer.zero_grad(set_to_none=True)
                candidate_pose = _bounded_local_pose(
                    parameter,
                    raw_target,
                    max_rotation=max_rotation,
                    max_translation=max_translation,
                )
                loss = _locked_target_loss(
                    candidate_pose,
                    target_index=target_index,
                    locked_pairs=moved_pairs,
                    distance_fields=moved_distances,
                    world_to_camera=moved_w2c,
                    intrinsics=moved_k,
                    projection=config.projection,
                    edge=config.edge,
                )
                if not bool(torch.isfinite(loss)):
                    break
                loss.backward()
                if parameter.grad is None or not bool(
                    torch.isfinite(parameter.grad).all()
                ):
                    break
                value = float(loss.detach())
                if value < best_loss:
                    best_loss = value
                    best_parameter = parameter.detach().clone()
                optimizer.step()
                completed_steps = step + 1
            with torch.no_grad():
                final_candidate_pose = _bounded_local_pose(
                    parameter,
                    raw_target,
                    max_rotation=max_rotation,
                    max_translation=max_translation,
                )
                final_candidate_loss = float(
                    _locked_target_loss(
                        final_candidate_pose,
                        target_index=target_index,
                        locked_pairs=moved_pairs,
                        distance_fields=moved_distances,
                        world_to_camera=moved_w2c,
                        intrinsics=moved_k,
                        projection=config.projection,
                        edge=config.edge,
                    )
                )
                if final_candidate_loss < best_loss:
                    best_loss = final_candidate_loss
                    best_parameter = parameter.detach().clone()
            with torch.no_grad():
                initial_pose = _bounded_local_pose(
                    initial_parameter,
                    raw_target,
                    max_rotation=max_rotation,
                    max_translation=max_translation,
                )
                best_pose = _bounded_local_pose(
                    best_parameter,
                    raw_target,
                    max_rotation=max_rotation,
                    max_translation=max_translation,
                )
                initial_rotation, initial_translation = _pose_errors(
                    initial_pose, raw_target
                )
                final_rotation, final_translation = _pose_errors(
                    best_pose, raw_target
                )
            rotation_unit = max(
                math.radians(min(config.recovery.rotation_degrees)), 1e-12
            )
            translation_unit = max(
                min(config.recovery.translation_scene_fractions)
                * translation_scene_scale,
                1e-12,
            )
            initial_joint_error = math.sqrt(
                (math.radians(initial_rotation) / rotation_unit) ** 2
                + (initial_translation / translation_unit) ** 2
            )
            final_joint_error = math.sqrt(
                (math.radians(final_rotation) / rotation_unit) ** 2
                + (final_translation / translation_unit) ** 2
            )
            recovery_fraction = (
                (initial_joint_error - final_joint_error)
                / max(initial_joint_error, 1e-12)
            )
            rows.append(
                {
                    **common,
                    **trial,
                    "active": 1,
                    "failure_reason": "ok",
                    "raw_loss": raw_loss,
                    "perturbed_loss": perturbed_loss,
                    "probe_loss": probe_loss,
                    "optimized_loss": best_loss,
                    "loss_order_pass": int(perturbed_loss > raw_loss + 1e-7),
                    "gradient_probe_pass": int(
                        math.isfinite(probe_loss)
                        and probe_loss < perturbed_loss - 1e-9
                    ),
                    "optimizer_active": int(best_loss < perturbed_loss - 1e-9),
                    "gradient_norm": gradient_norm,
                    "optimizer_steps_completed": completed_steps,
                    "initial_rotation_error_deg": initial_rotation,
                    "final_rotation_error_deg": final_rotation,
                    "initial_translation_error_native": initial_translation,
                    "final_translation_error_native": final_translation,
                    "initial_joint_normalized_error": initial_joint_error,
                    "final_joint_normalized_error": final_joint_error,
                    "recovery_fraction": recovery_fraction,
                }
            )
    return rows


def summarize_recovery(
    rows: Sequence[dict[str, object]],
    recovery: RecoveryConfig,
) -> dict[str, object]:
    active = [row for row in rows if int(row.get("active", 0)) == 1]
    rotation = [
        float(row["recovery_fraction"])
        for row in active
        if row["perturbation_family"] == "rotation"
    ]
    translation = [
        float(row["recovery_fraction"])
        for row in active
        if row["perturbation_family"] == "translation"
    ]
    active_frames = {
        int(row["target_frame_index"])
        for row in active
    }
    all_frames = {
        int(row["target_frame_index"])
        for row in rows
    }
    frame_summaries = []
    for frame in sorted(all_frames):
        frame_rows = [
            row for row in active if int(row["target_frame_index"]) == frame
        ]
        frame_rotation = [
            float(row["recovery_fraction"])
            for row in frame_rows
            if row["perturbation_family"] == "rotation"
        ]
        frame_translation = [
            float(row["recovery_fraction"])
            for row in frame_rows
            if row["perturbation_family"] == "translation"
        ]
        frame_ordering = _rate(frame_rows, "loss_order_pass")
        frame_probe = _rate(frame_rows, "gradient_probe_pass")
        frame_rotation_mean = _mean(frame_rotation)
        frame_translation_mean = _mean(frame_translation)
        frame_pass = bool(
            frame_rows
            and frame_ordering >= float(recovery.min_loss_order_pass_rate)
            and frame_probe >= float(recovery.min_loss_order_pass_rate)
            and frame_rotation_mean
            >= float(recovery.min_mean_recovery_fraction)
            and frame_translation_mean
            >= float(recovery.min_mean_recovery_fraction)
        )
        frame_summaries.append(
            {
                "target_frame_index": frame,
                "active_trials": len(frame_rows),
                "loss_order_pass_rate": frame_ordering,
                "gradient_probe_pass_rate": frame_probe,
                "mean_rotation_recovery_fraction": frame_rotation_mean,
                "mean_translation_recovery_fraction": frame_translation_mean,
                "frame_pass": int(frame_pass),
            }
        )
    ordering = _rate(active, "loss_order_pass")
    probe = _rate(active, "gradient_probe_pass")
    mean_rotation = _mean(rotation)
    mean_translation = _mean(translation)
    pose_basin_pass = bool(
        active
        and active_frames == all_frames
        and all(int(row["frame_pass"]) == 1 for row in frame_summaries)
    )
    return {
        "recovery_trials": len(rows),
        "active_trials": len(active),
        "evaluation_frames": len(all_frames),
        "active_frames": len(active_frames),
        "passing_frames": sum(int(row["frame_pass"]) for row in frame_summaries),
        "loss_order_pass_rate": ordering,
        "gradient_probe_pass_rate": probe,
        "optimizer_active_rate": _rate(active, "optimizer_active"),
        "mean_rotation_recovery_fraction": mean_rotation,
        "mean_translation_recovery_fraction": mean_translation,
        "minimum_frame_loss_order_pass_rate": _finite_min(
            row["loss_order_pass_rate"] for row in frame_summaries
        ),
        "minimum_frame_gradient_probe_pass_rate": _finite_min(
            row["gradient_probe_pass_rate"] for row in frame_summaries
        ),
        "minimum_frame_rotation_recovery_fraction": _finite_min(
            row["mean_rotation_recovery_fraction"] for row in frame_summaries
        ),
        "minimum_frame_translation_recovery_fraction": _finite_min(
            row["mean_translation_recovery_fraction"] for row in frame_summaries
        ),
        "frame_summaries": frame_summaries,
        "pose_basin_pass": int(pose_basin_pass),
        "claim": "local_all_edge_objective_feasibility_not_pose_improvement",
    }


def _lock_target_support(
    *,
    target_index: int,
    source_offsets: Sequence[int],
    all_edges: torch.Tensor,
    depth: torch.Tensor,
    depth_confidence: torch.Tensor,
    world_to_camera: torch.Tensor,
    intrinsics: torch.Tensor,
    projection: ProjectionConfig,
) -> list[dict[str, object]]:
    pairs: list[dict[str, object]] = []
    for offset in source_offsets:
        source_index = target_index - int(offset)
        if source_index < 0:
            continue
        forward = _lock_direction_support(
            edge_map=all_edges[source_index],
            source_depth=depth[source_index],
            source_confidence=depth_confidence[source_index],
            target_depth=depth[target_index],
            target_confidence=depth_confidence[target_index],
            source_w2c=world_to_camera[source_index],
            target_w2c=world_to_camera[target_index],
            source_k=intrinsics[source_index],
            target_k=intrinsics[target_index],
            projection=projection,
        )
        reverse = _lock_direction_support(
            edge_map=all_edges[target_index],
            source_depth=depth[target_index],
            source_confidence=depth_confidence[target_index],
            target_depth=depth[source_index],
            target_confidence=depth_confidence[source_index],
            source_w2c=world_to_camera[target_index],
            target_w2c=world_to_camera[source_index],
            source_k=intrinsics[target_index],
            target_k=intrinsics[source_index],
            projection=projection,
        )
        minimum = int(projection.min_locked_points_per_direction)
        if forward["x"].numel() < minimum or reverse["x"].numel() < minimum:
            continue
        pairs.append(
            {
                "source_index": source_index,
                "target_index": target_index,
                "forward": forward,
                "reverse": reverse,
            }
        )
    return pairs


def _lock_direction_support(
    *,
    edge_map: torch.Tensor,
    source_depth: torch.Tensor,
    source_confidence: torch.Tensor,
    target_depth: torch.Tensor,
    target_confidence: torch.Tensor,
    source_w2c: torch.Tensor,
    target_w2c: torch.Tensor,
    source_k: torch.Tensor,
    target_k: torch.Tensor,
    projection: ProjectionConfig,
) -> dict[str, torch.Tensor]:
    h, w = edge_map.shape
    y, x = torch.nonzero(edge_map.bool(), as_tuple=True)
    z_source = source_depth[y, x].float()
    confidence = source_confidence[y, x].float()
    valid_source = (
        torch.isfinite(z_source)
        & torch.isfinite(confidence)
        & (z_source > float(projection.min_depth))
        & (z_source < float(projection.max_depth))
        & (confidence >= float(projection.min_depth_confidence))
    )
    x = x[valid_source].float()
    y = y[valid_source].float()
    z_source = z_source[valid_source]
    if x.numel() == 0:
        return {"x": x, "y": y, "depth": z_source}
    projected = project_depth_points(
        x=x,
        y=y,
        depth=z_source,
        source_w2c=source_w2c,
        target_w2c=target_w2c,
        source_k=source_k,
        target_k=target_k,
    )
    u, v, z_target = projected.unbind(dim=-1)
    in_bounds = (
        torch.isfinite(projected).all(dim=-1)
        & (z_target > float(projection.min_depth))
        & (u >= 0.0)
        & (u <= w - 1.0)
        & (v >= 0.0)
        & (v <= h - 1.0)
    )
    sampled_depth = bilinear_sample_image(
        target_depth, u.clamp(0, w - 1), v.clamp(0, h - 1)
    )
    sampled_confidence = bilinear_sample_image(
        target_confidence, u.clamp(0, w - 1), v.clamp(0, h - 1)
    )
    tolerance = float(projection.depth_abs_tolerance) + float(
        projection.depth_rel_tolerance
    ) * torch.maximum(z_target.abs(), sampled_depth.abs())
    consistent = (
        in_bounds
        & torch.isfinite(sampled_depth)
        & torch.isfinite(sampled_confidence)
        & (sampled_depth > float(projection.min_depth))
        & (sampled_depth < float(projection.max_depth))
        & (sampled_confidence >= float(projection.min_depth_confidence))
        & ((sampled_depth - z_target).abs() <= tolerance)
    )
    x = x[consistent]
    y = y[consistent]
    z_source = z_source[consistent]
    maximum = int(projection.max_locked_points_per_direction)
    if x.numel() > maximum:
        selected = torch.linspace(0, x.numel() - 1, maximum).round().long()
        x = x.index_select(0, selected)
        y = y.index_select(0, selected)
        z_source = z_source.index_select(0, selected)
    return {"x": x, "y": y, "depth": z_source}


def _locked_target_loss(
    candidate_target_w2c: torch.Tensor,
    *,
    target_index: int,
    locked_pairs: Sequence[dict[str, object]],
    distance_fields: torch.Tensor,
    world_to_camera: torch.Tensor,
    intrinsics: torch.Tensor,
    projection: ProjectionConfig,
    edge: EdgeConfig,
) -> torch.Tensor:
    direction_losses = []
    for pair in locked_pairs:
        source_index = int(pair["source_index"])
        forward = pair["forward"]
        forward_uvz = project_depth_points(
            x=forward["x"],
            y=forward["y"],
            depth=forward["depth"],
            source_w2c=world_to_camera[source_index],
            target_w2c=candidate_target_w2c,
            source_k=intrinsics[source_index],
            target_k=intrinsics[target_index],
        )
        direction_losses.append(
            _distance_residual_loss(
                forward_uvz,
                distance_fields[target_index],
                projection=projection,
                edge=edge,
            )
        )
        reverse = pair["reverse"]
        reverse_uvz = project_depth_points(
            x=reverse["x"],
            y=reverse["y"],
            depth=reverse["depth"],
            source_w2c=candidate_target_w2c,
            target_w2c=world_to_camera[source_index],
            source_k=intrinsics[target_index],
            target_k=intrinsics[source_index],
        )
        direction_losses.append(
            _distance_residual_loss(
                reverse_uvz,
                distance_fields[source_index],
                projection=projection,
                edge=edge,
            )
        )
    return torch.stack(direction_losses).mean()


def _distance_residual_loss(
    uvz: torch.Tensor,
    distance_field: torch.Tensor,
    *,
    projection: ProjectionConfig,
    edge: EdgeConfig,
) -> torch.Tensor:
    h, w = distance_field.shape
    u, v, z = uvz.unbind(dim=-1)
    outside = (
        F.relu(-u)
        + F.relu(u - (w - 1.0))
        + F.relu(-v)
        + F.relu(v - (h - 1.0))
    )
    invalid_depth = F.relu(float(projection.min_depth) - z)
    sampled = bilinear_sample_image(
        distance_field,
        u.clamp(0.0, w - 1.0),
        v.clamp(0.0, h - 1.0),
    )
    residual = sampled + float(projection.boundary_penalty_weight) * (
        outside + invalid_depth
    )
    return F.smooth_l1_loss(
        residual,
        torch.zeros_like(residual),
        beta=float(edge.huber_delta_px),
        reduction="mean",
    )


def _perturbation_trials(
    recovery: RecoveryConfig,
    scene_scale: float,
) -> list[dict[str, object]]:
    trials = []
    for axis in recovery.axes:
        for sign in recovery.signs:
            for magnitude in recovery.rotation_degrees:
                trials.append(
                    {
                        "perturbation_family": "rotation",
                        "axis": axis,
                        "sign": int(sign),
                        "magnitude": float(magnitude),
                        "magnitude_unit": "degrees",
                    }
                )
            for fraction in recovery.translation_scene_fractions:
                trials.append(
                    {
                        "perturbation_family": "translation",
                        "axis": axis,
                        "sign": int(sign),
                        "magnitude": float(fraction) * scene_scale,
                        "magnitude_unit": "native",
                        "scene_fraction": float(fraction),
                    }
                )
    return trials


def _trial_parameter(
    trial: dict[str, object],
    *,
    max_rotation: float,
    max_translation: float,
    device: torch.device,
) -> torch.Tensor:
    parameter = torch.zeros(6, device=device)
    axis_index = {"x": 0, "y": 1, "z": 2}[str(trial["axis"])]
    sign = float(trial["sign"])
    if trial["perturbation_family"] == "rotation":
        value = sign * math.radians(float(trial["magnitude"]))
        parameter[axis_index] = _inverse_tanh_ratio(value, max_rotation)
    else:
        value = sign * float(trial["magnitude"])
        parameter[3 + axis_index] = _inverse_tanh_ratio(value, max_translation)
    return parameter


def _inverse_tanh_ratio(value: float, maximum: float) -> float:
    if maximum <= 0.0:
        raise ValueError("bounded correction maximum must be positive.")
    ratio = max(-0.999999, min(0.999999, value / maximum))
    return math.atanh(ratio)


def _bounded_local_pose(
    parameter: torch.Tensor,
    raw_target_w2c: torch.Tensor,
    *,
    max_rotation: float,
    max_translation: float,
) -> torch.Tensor:
    # Left-multiplication expresses the correction in the target-camera frame.
    # The full se(3) matrix exponential keeps rotation/translation coupled.
    tangent = torch.cat(
        (
            float(max_rotation) * torch.tanh(parameter[:3]),
            float(max_translation) * torch.tanh(parameter[3:]),
        )
    )
    top = torch.cat((_skew(tangent[:3]), tangent[3:, None]), dim=1)
    bottom = torch.zeros(1, 4, dtype=parameter.dtype, device=parameter.device)
    generator = torch.cat((top, bottom), dim=0)
    return torch.matrix_exp(generator) @ raw_target_w2c


def _skew(vector: torch.Tensor) -> torch.Tensor:
    x, y, z = vector.unbind()
    zero = torch.zeros((), dtype=vector.dtype, device=vector.device)
    return torch.stack(
        (
            torch.stack((zero, -z, y)),
            torch.stack((z, zero, -x)),
            torch.stack((-y, x, zero)),
        )
    )


def _homogeneous_pose(pose: torch.Tensor) -> torch.Tensor:
    if pose.shape == (4, 4):
        return pose
    if pose.shape != (3, 4):
        raise ValueError("pose must have shape [3,4] or [4,4].")
    bottom = pose.new_tensor([[0.0, 0.0, 0.0, 1.0]])
    return torch.cat((pose, bottom), dim=0)


def _pose_errors(
    candidate: torch.Tensor,
    reference: torch.Tensor,
) -> tuple[float, float]:
    relative_rotation = candidate[:3, :3] @ reference[:3, :3].T
    cosine = ((torch.trace(relative_rotation) - 1.0) / 2.0).clamp(-1.0, 1.0)
    rotation = float(torch.rad2deg(torch.acos(cosine)))
    candidate_center = -(candidate[:3, :3].T @ candidate[:3, 3])
    reference_center = -(reference[:3, :3].T @ reference[:3, 3])
    translation = float(torch.linalg.vector_norm(candidate_center - reference_center))
    return rotation, translation


def _move_locked_pairs(
    pairs: Sequence[dict[str, object]],
    device: torch.device,
) -> list[dict[str, object]]:
    return [
        {
            "source_index": pair["source_index"],
            "target_index": pair["target_index"],
            "forward": {
                key: value.to(device) for key, value in pair["forward"].items()
            },
            "reverse": {
                key: value.to(device) for key, value in pair["reverse"].items()
            },
        }
        for pair in pairs
    ]


def _inactive_recovery_metrics(reason: str) -> dict[str, object]:
    fields = {
        "active": 0,
        "failure_reason": reason,
        "loss_order_pass": 0,
        "gradient_probe_pass": 0,
        "optimizer_active": 0,
        "optimizer_steps_completed": 0,
    }
    for key in (
        "raw_loss",
        "perturbed_loss",
        "probe_loss",
        "optimized_loss",
        "gradient_norm",
        "initial_rotation_error_deg",
        "final_rotation_error_deg",
        "initial_translation_error_native",
        "final_translation_error_native",
        "initial_joint_normalized_error",
        "final_joint_normalized_error",
        "recovery_fraction",
    ):
        fields[key] = float("nan")
    return fields


def project_depth_points(
    *,
    x: torch.Tensor,
    y: torch.Tensor,
    depth: torch.Tensor,
    source_w2c: torch.Tensor,
    target_w2c: torch.Tensor,
    source_k: torch.Tensor,
    target_k: torch.Tensor,
) -> torch.Tensor:
    ones = torch.ones_like(x)
    pixels = torch.stack((x, y, ones), dim=-1).float()
    source_k_inv = torch.linalg.inv(source_k.float())
    source_camera = (pixels @ source_k_inv.T) * depth[:, None].float()
    source_rotation = source_w2c[:3, :3].float()
    source_translation = source_w2c[:3, 3].float()
    world = (source_camera - source_translation) @ source_rotation
    target_rotation = target_w2c[:3, :3].float()
    target_translation = target_w2c[:3, 3].float()
    target_camera = world @ target_rotation.T + target_translation
    projected = target_camera @ target_k.float().T
    z = target_camera[:, 2].clamp_min(1e-8)
    uv = projected[:, :2] / projected[:, 2:].clamp_min(1e-8)
    return torch.cat((uv, z[:, None]), dim=-1)


def sobel_magnitude(images: torch.Tensor) -> torch.Tensor:
    if images.ndim != 3:
        raise ValueError("images must have shape [T,H,W].")
    device = images.device
    dtype = images.dtype
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=dtype,
        device=device,
    )
    kernel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        dtype=dtype,
        device=device,
    )
    weight = torch.stack((kernel_x, kernel_y), dim=0)[:, None]
    gradients = F.conv2d(images[:, None], weight, padding=1)
    return torch.linalg.vector_norm(gradients, dim=1)


def threshold_edges(
    strength: torch.Tensor,
    *,
    quantile: float,
    max_edges_per_frame: int,
) -> torch.Tensor:
    if strength.ndim != 3:
        raise ValueError("edge strength must have shape [T,H,W].")
    output = []
    for frame in strength:
        flat = frame.flatten()
        finite = flat[torch.isfinite(flat)]
        if finite.numel() == 0:
            output.append(torch.zeros_like(frame, dtype=torch.bool))
            continue
        cutoff = torch.quantile(finite, float(quantile))
        mask = frame >= cutoff
        if int(mask.sum()) > int(max_edges_per_frame):
            indices = torch.topk(flat, k=int(max_edges_per_frame)).indices
            limited = torch.zeros_like(flat, dtype=torch.bool)
            limited[indices] = True
            mask = limited.reshape_as(frame)
        output.append(mask.bool())
    return torch.stack(output, dim=0)


def truncated_distance_transform(
    edge_mask: torch.Tensor,
    *,
    max_distance: int,
) -> torch.Tensor:
    """Return the Euclidean distance-to-edge field, truncated in pixels."""

    if edge_mask.ndim != 2:
        raise ValueError("edge_mask must have shape [H,W].")
    from scipy.ndimage import distance_transform_edt

    distance = distance_transform_edt(~edge_mask.detach().bool().cpu().numpy())
    return torch.from_numpy(distance).float().clamp_max(float(max_distance))


def summarize_rows(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        branch = str(row["branch"])
        if branch == "raw":
            continue
        grouped.setdefault(branch, []).append(row)
    summary = []
    for branch, branch_rows in sorted(grouped.items()):
        distances = [
            float(row["mean_truncated_edge_distance_px"])
            for row in branch_rows
            if math.isfinite(float(row["mean_truncated_edge_distance_px"]))
        ]
        consistency = [
            float(row["depth_consistency_pass_rate"])
            for row in branch_rows
            if math.isfinite(float(row["depth_consistency_pass_rate"]))
        ]
        in_bounds = [
            float(row["in_bounds_rate"])
            for row in branch_rows
            if math.isfinite(float(row["in_bounds_rate"]))
        ]
        summary.append(
            {
                "branch": branch,
                "pair_count": len(branch_rows),
                "mean_truncated_edge_distance_px": _mean(distances),
                "mean_depth_consistency_pass_rate": _mean(consistency),
                "mean_in_bounds_rate": _mean(in_bounds),
            }
        )
    return summary


def csv_columns(rows: Sequence[dict[str, object]]) -> tuple[str, ...]:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    return tuple(columns)


def _load_gray_images(paths: Sequence[str], *, image_size: tuple[int, int]) -> torch.Tensor:
    h, w = image_size
    frames = []
    for value in paths:
        image = Image.open(value).convert("L").resize((w, h), Image.BILINEAR)
        data = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
        frames.append(data.float().reshape(h, w) / 255.0)
    return torch.stack(frames, dim=0)


def _load_exclusion_masks(
    payload: dict,
    *,
    size: tuple[int, int],
) -> tuple[torch.Tensor, str]:
    candidates = (
        "associated_tracking_masks_output",
        "trusted_tracking_masks_output",
        "tracking_masks_output",
    )
    masks = None
    for key in candidates:
        value = payload.get(key)
        if torch.is_tensor(value):
            masks = value.bool()
            break
    if masks is None:
        frames = len(payload["frame_indices"])
        return torch.zeros((frames, *size), dtype=torch.bool), "none_zero_mask"
    if masks.ndim != 4:
        raise ValueError(f"{key} must have shape [T,K,H,W].")
    merged = masks.any(dim=1)
    if merged.shape[-2:] != size:
        merged = _resize_mask(merged[:, None], size)[:, 0]
    return merged.bool(), str(key)


def dilate_masks(masks: torch.Tensor, *, radius: int) -> torch.Tensor:
    if masks.ndim != 3:
        raise ValueError("masks must have shape [T,H,W].")
    radius = int(radius)
    if radius <= 0:
        return masks.bool()
    return F.max_pool2d(
        masks.float()[:, None],
        kernel_size=2 * radius + 1,
        stride=1,
        padding=radius,
    )[:, 0].bool()


def shifted_masks_no_wrap(masks: torch.Tensor) -> torch.Tensor:
    """Create the location control without wrap-around leakage."""

    h, w = masks.shape[-2:]
    dy = max(1, h // 4)
    dx = max(1, w // 4)
    directions = ((dy, dx), (dy, -dx), (-dy, dx), (-dy, -dx))
    return torch.stack(
        [
            _shift(mask.bool(), *directions[index % len(directions)], False)
            for index, mask in enumerate(masks)
        ],
        dim=0,
    )


def equal_count_branch_edges(
    *,
    all_edges: torch.Tensor,
    edge_strength: torch.Tensor,
    exclusion_masks: torch.Tensor,
    branches: Sequence[str],
) -> dict[str, torch.Tensor]:
    """Use the same per-frame edge count for every projection control."""

    if all_edges.shape != edge_strength.shape or all_edges.shape != exclusion_masks.shape:
        raise ValueError("edge candidates, strengths and masks must share [T,H,W].")
    shifted = shifted_masks_no_wrap(exclusion_masks.bool())
    outputs = {
        branch: torch.zeros_like(all_edges, dtype=torch.bool)
        for branch in branches
    }
    for frame_index in range(all_edges.shape[0]):
        candidates: dict[str, torch.Tensor] = {}
        for branch in branches:
            if branch == "all_edge":
                candidate = all_edges[frame_index]
            elif branch == "sam_object_excluded_edge":
                candidate = all_edges[frame_index] & ~exclusion_masks[frame_index]
            elif branch == "shifted_object_mask_control":
                candidate = all_edges[frame_index] & ~shifted[frame_index]
            else:
                raise ValueError(f"Unsupported edge branch: {branch!r}.")
            candidates[branch] = candidate.bool()
        locked_count = min(int(value.sum()) for value in candidates.values())
        for branch, candidate in candidates.items():
            if locked_count == 0:
                continue
            flat_indices = torch.nonzero(candidate.flatten(), as_tuple=False)[:, 0]
            scores = edge_strength[frame_index].flatten().index_select(0, flat_indices)
            order = torch.argsort(scores, descending=True, stable=True)
            selected = flat_indices.index_select(0, order[:locked_count])
            outputs[branch][frame_index].view(-1)[selected] = True
    return outputs


def _raw_row(
    *,
    frames: Sequence[int],
    source_index: int,
    target_index: int,
    offset: int,
) -> dict[str, object]:
    return {
        **_row_base(frames, source_index, target_index, offset, "raw"),
        "source_edge_pixels": 0,
        "source_positive_depth_pixels": 0,
        "projected_in_bounds_pixels": 0,
        "depth_consistency_pass_pixels": 0,
        "valid_projected_pixels": 0,
        "in_bounds_rate": float("nan"),
        "depth_consistency_pass_rate": float("nan"),
        "mean_truncated_edge_distance_px": float("nan"),
        "median_truncated_edge_distance_px": float("nan"),
        "mean_depth_confidence": float("nan"),
    }


def _empty_projection_row(
    frames: Sequence[int],
    source_index: int,
    target_index: int,
    offset: int,
    branch: str,
    *,
    source_edge_pixels: int,
    in_bounds_rate: float = float("nan"),
) -> dict[str, object]:
    return {
        **_row_base(frames, source_index, target_index, offset, branch),
        "source_edge_pixels": int(source_edge_pixels),
        "source_positive_depth_pixels": 0,
        "projected_in_bounds_pixels": 0,
        "depth_consistency_pass_pixels": 0,
        "valid_projected_pixels": 0,
        "in_bounds_rate": float(in_bounds_rate),
        "depth_consistency_pass_rate": float("nan"),
        "mean_truncated_edge_distance_px": float("nan"),
        "median_truncated_edge_distance_px": float("nan"),
        "mean_depth_confidence": float("nan"),
    }


def _row_base(
    frames: Sequence[int],
    source_index: int,
    target_index: int,
    offset: int,
    branch: str,
) -> dict[str, object]:
    return {
        "branch": branch,
        "source_sequence_index": int(source_index),
        "target_sequence_index": int(target_index),
        "source_frame_index": int(frames[source_index]),
        "target_frame_index": int(frames[target_index]),
        "source_offset": int(offset),
    }


def _tensor_field(payload: dict, key: str) -> torch.Tensor:
    value = payload.get(key)
    if not torch.is_tensor(value):
        raise ValueError(f"Feature cache lacks tensor field {key!r}.")
    return value


def _image_size(payload: dict) -> tuple[int, int]:
    raw = payload.get("image_size")
    if not raw or len(raw) != 2:
        raise ValueError("Feature cache lacks image_size=[H,W].")
    return int(raw[0]), int(raw[1])


def _resize_scalar(value: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(value.float(), size=size, mode="bilinear", align_corners=False)


def _resize_mask(value: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(value.float(), size=size, mode="nearest").bool()


def _shift(value: torch.Tensor, dy: int, dx: int, fill: float) -> torch.Tensor:
    h, w = value.shape
    output = torch.full_like(value, float(fill))
    y_src_start = max(0, -dy)
    y_src_end = min(h, h - dy)
    x_src_start = max(0, -dx)
    x_src_end = min(w, w - dx)
    y_dst_start = max(0, dy)
    y_dst_end = min(h, h + dy)
    x_dst_start = max(0, dx)
    x_dst_end = min(w, w + dx)
    output[y_dst_start:y_dst_end, x_dst_start:x_dst_end] = value[
        y_src_start:y_src_end,
        x_src_start:x_src_end,
    ]
    return output


def _mean(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _rate(rows: Sequence[dict[str, object]], key: str) -> float:
    if not rows:
        return float("nan")
    return float(sum(int(row[key]) for row in rows) / len(rows))


def _finite_min(values: Iterable[object]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return min(finite) if finite else float("nan")


def _int_tuple(value) -> tuple[int, ...]:
    return tuple(int(item) for item in (value or ()))


def _path(value: str | Path | None, base: Path) -> Path:
    if value is None:
        raise ValueError("Missing required path value.")
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _validate_config(config: EdgeFeasibilityConfig) -> None:
    if not config.clip_name:
        raise ValueError("e0.clip_name is required.")
    if not config.evaluation_frames:
        raise ValueError("e0.frames.evaluation is required.")
    if not config.branches:
        raise ValueError("e0.branches must contain at least one branch.")
    unknown = set(config.branches) - VALID_BRANCHES
    if unknown:
        raise ValueError(f"Unsupported E0 branches: {sorted(unknown)}.")
    if "all_edge" not in config.branches:
        raise ValueError("e0.branches must retain all_edge as the recovery branch.")
    if not config.projection.source_offsets:
        raise ValueError("e0.projection.source_offsets must not be empty.")
    if config.projection.min_locked_points_per_direction <= 0:
        raise ValueError("min_locked_points_per_direction must be positive.")
    if (
        config.projection.max_locked_points_per_direction
        < config.projection.min_locked_points_per_direction
    ):
        raise ValueError("max_locked_points_per_direction must be >= minimum.")
    if config.edge.max_edges_per_frame <= 0:
        raise ValueError("e0.edge.max_edges_per_frame must be positive.")
    if not (0.0 < config.edge.sobel_quantile < 1.0):
        raise ValueError("e0.edge.sobel_quantile must be in (0,1).")
    if set(config.recovery.axes) - {"x", "y", "z"}:
        raise ValueError("e0.recovery.axes only accepts x/y/z.")
    if set(config.recovery.signs) - {-1, 1}:
        raise ValueError("e0.recovery.signs only accepts -1/+1.")
    if not config.recovery.rotation_degrees:
        raise ValueError("e0.recovery.rotation_degrees must not be empty.")
    if not config.recovery.translation_scene_fractions:
        raise ValueError("e0.recovery.translation_scene_fractions must not be empty.")
    if config.recovery.max_rotation_degrees <= max(config.recovery.rotation_degrees):
        raise ValueError("max_rotation_degrees must exceed every test perturbation.")
    if config.recovery.max_translation_scene_fraction <= max(
        config.recovery.translation_scene_fractions
    ):
        raise ValueError(
            "max_translation_scene_fraction must exceed every test perturbation."
        )
