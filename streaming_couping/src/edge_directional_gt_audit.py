"""GT-scored fixed-step audit of edge-pose gradient directions.

This module is deliberately split into candidate generation and scoring.
`generate_directional_candidates` has no GT argument.  Ground truth is only
consumed later by `score_directional_candidates`; it never selects a branch,
step size, support point, gradient direction, or candidate pose.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Sequence

import torch
import yaml

from .edge_pose_feasibility import (
    EdgeFeasibilityConfig,
    _homogeneous_pose,
    _image_size,
    _load_exclusion_masks,
    _load_gray_images,
    _lock_target_support,
    _locked_target_loss,
    _move_locked_pairs,
    _pose_errors,
    _resize_scalar,
    _tensor_field,
    dilate_masks,
    equal_count_branch_edges,
    load_edge_feasibility_config,
    sobel_magnitude,
    threshold_edges,
    truncated_distance_transform,
)


E1_REVISION = "e1_fixed_step_edge_direction_gt_scoring_r1"


@dataclass(frozen=True)
class DirectionalConfig:
    rotation_step_degrees: float = 0.25
    translation_step_scene_fraction: float = 0.0025
    min_gradient_norm: float = 1e-10


@dataclass(frozen=True)
class EdgeDirectionalAuditConfig:
    source_path: Path
    base_config: Path
    e0_config_path: Path
    output_dir: Path
    clip_name: str
    evaluation_frames: tuple[int, ...]
    branches: tuple[str, ...]
    primary_branch: str
    directional: DirectionalConfig
    objective: EdgeFeasibilityConfig


@dataclass(frozen=True)
class DirectionalCandidate:
    target_sequence_index: int
    target_frame_index: int
    branch: str
    active: int
    failure_reason: str
    locked_pairs: int
    locked_points: int
    rotation_gradient_norm: float
    translation_gradient_norm: float
    raw_loss: float
    poses: dict[str, torch.Tensor]
    losses: dict[str, float]
    rotation_steps_deg: dict[str, float]
    translation_steps_native: dict[str, float]


VALID_DIRECTIONS = (
    "raw",
    "negative_joint",
    "positive_joint",
    "negative_rotation_only",
    "positive_rotation_only",
    "negative_translation_only",
    "positive_translation_only",
)


def load_edge_directional_audit_config(
    path: str | Path,
) -> EdgeDirectionalAuditConfig:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf8") as handle:
        raw = yaml.safe_load(handle) or {}
    section = raw.get("e1", {})
    frames = section.get("frames", {})
    directional = section.get("directional", {})
    base_config = _path(section.get("base_config"), source.parent)
    e0_config_path = _path(section.get("e0_config"), source.parent)
    objective = load_edge_feasibility_config(e0_config_path)
    config = EdgeDirectionalAuditConfig(
        source_path=source,
        base_config=base_config,
        e0_config_path=e0_config_path,
        output_dir=_path(
            section.get(
                "output_dir",
                "outputs/streaming_couping_e1_edge_directional_gt_audit",
            ),
            source.parent,
        ),
        clip_name=str(section.get("clip_name", "")),
        evaluation_frames=tuple(
            int(value) for value in frames.get("evaluation", ())
        ),
        branches=tuple(str(value) for value in section.get("branches", ())),
        primary_branch=str(section.get("primary_branch", "")),
        directional=DirectionalConfig(
            rotation_step_degrees=float(
                directional.get("rotation_step_degrees", 0.25)
            ),
            translation_step_scene_fraction=float(
                directional.get("translation_step_scene_fraction", 0.0025)
            ),
            min_gradient_norm=float(
                directional.get("min_gradient_norm", 1e-10)
            ),
        ),
        objective=objective,
    )
    _validate_config(config)
    return config


def objective_payload_from_cache(payload: dict) -> dict:
    """Strip every GT/scoring field before candidate generation."""

    required = (
        "clip_name",
        "frame_indices",
        "image_paths",
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
    forbidden = {
        "target_pose_encoding",
        "target_world_to_camera",
        "target_depth",
        "target_world_points",
    }
    if forbidden & set(output):
        raise RuntimeError("E1 objective payload contains forbidden GT fields.")
    return output


def generate_directional_candidates(
    *,
    payload: dict,
    raw_world_to_camera: torch.Tensor,
    intrinsics: torch.Tensor,
    config: EdgeDirectionalAuditConfig,
    device: str | torch.device,
) -> tuple[list[DirectionalCandidate], dict[str, object]]:
    """Generate every fixed candidate without accepting any GT tensor."""

    forbidden = {
        "target_pose_encoding",
        "target_world_to_camera",
        "target_depth",
        "target_world_points",
    }
    leaked = forbidden & set(payload)
    if leaked:
        raise ValueError(f"E1 candidate generation received GT fields={sorted(leaked)}.")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"E1 requested {device}, but CUDA is unavailable.")
    frames = tuple(int(value) for value in payload["frame_indices"])
    positions = {frame: index for index, frame in enumerate(frames)}
    evaluation_indices = [positions[frame] for frame in config.evaluation_frames]
    images = _load_gray_images(
        payload["image_paths"], image_size=_image_size(payload)
    )
    depth = _tensor_field(payload, "baseline_depth").float()
    confidence = _tensor_field(payload, "baseline_depth_confidence").float()
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if confidence.ndim == 4 and confidence.shape[-1] == 1:
        confidence = confidence[..., 0]
    if depth.shape[-2:] != images.shape[-2:]:
        depth = _resize_scalar(depth[:, None], images.shape[-2:])[:, 0]
        confidence = _resize_scalar(
            confidence[:, None], images.shape[-2:]
        )[:, 0]
    if raw_world_to_camera.ndim == 4:
        raw_world_to_camera = raw_world_to_camera[0]
    if intrinsics.ndim == 4:
        intrinsics = intrinsics[0]

    strength = sobel_magnitude(images)
    all_edges = threshold_edges(
        strength,
        quantile=config.objective.edge.sobel_quantile,
        max_edges_per_frame=config.objective.edge.max_edges_per_frame,
    )
    exclusion, exclusion_source = _load_exclusion_masks(
        payload, size=images.shape[-2:]
    )
    exclusion = dilate_masks(
        exclusion, radius=config.objective.edge.mask_dilation_px
    )
    branch_edges = equal_count_branch_edges(
        all_edges=all_edges,
        edge_strength=strength,
        exclusion_masks=exclusion,
        branches=config.branches,
    )
    distance_fields = {
        branch: torch.stack(
            [
                truncated_distance_transform(
                    edge_map,
                    max_distance=config.objective.edge.distance_truncate_px,
                )
                for edge_map in edges
            ]
        ).to(device)
        for branch, edges in branch_edges.items()
    }
    depth = depth.detach().float().cpu()
    confidence = confidence.detach().float().cpu()
    raw_cpu = raw_world_to_camera.detach().float().cpu()
    intrinsics_cpu = intrinsics.detach().float().cpu()
    raw_device = raw_cpu.to(device)
    intrinsics_device = intrinsics_cpu.to(device)
    scene_scale = max(abs(float(payload["scene_scale"])), 1e-8)
    rotation_step = math.radians(config.directional.rotation_step_degrees)
    translation_step = (
        config.directional.translation_step_scene_fraction * scene_scale
    )

    candidates: list[DirectionalCandidate] = []
    for target_index in evaluation_indices:
        for branch in config.branches:
            locked = _lock_target_support(
                target_index=target_index,
                source_offsets=config.objective.projection.source_offsets,
                all_edges=branch_edges[branch],
                depth=depth,
                depth_confidence=confidence,
                world_to_camera=raw_cpu,
                intrinsics=intrinsics_cpu,
                projection=config.objective.projection,
            )
            locked_points = sum(
                int(pair[direction]["x"].numel())
                for pair in locked
                for direction in ("forward", "reverse")
            )
            if not locked:
                candidates.append(
                    _inactive_candidate(
                        target_index,
                        frames[target_index],
                        branch,
                        locked_points,
                        "fewer_than_min_locked_points",
                    )
                )
                continue
            moved = _move_locked_pairs(locked, device)
            raw_pose = _homogeneous_pose(raw_device[target_index])
            tangent = torch.zeros(6, device=device, requires_grad=True)
            loss = _locked_target_loss(
                local_se3_pose(tangent, raw_pose),
                target_index=target_index,
                locked_pairs=moved,
                distance_fields=distance_fields[branch],
                world_to_camera=raw_device,
                intrinsics=intrinsics_device,
                projection=config.objective.projection,
                edge=config.objective.edge,
            )
            loss.backward()
            gradient = tangent.grad.detach()
            rotation_norm = float(gradient[:3].norm())
            translation_norm = float(gradient[3:].norm())
            rotation_direction = _unit_or_zero(
                gradient[:3], config.directional.min_gradient_norm
            )
            translation_direction = _unit_or_zero(
                gradient[3:], config.directional.min_gradient_norm
            )
            if not bool(torch.isfinite(gradient).all()) or (
                rotation_norm < config.directional.min_gradient_norm
                and translation_norm < config.directional.min_gradient_norm
            ):
                candidates.append(
                    _inactive_candidate(
                        target_index,
                        frames[target_index],
                        branch,
                        locked_points,
                        "nonfinite_or_zero_gradient",
                    )
                )
                continue
            negative_rotation = -rotation_step * rotation_direction
            negative_translation = -translation_step * translation_direction
            deltas = {
                "raw": torch.zeros(6, device=device),
                "negative_joint": torch.cat(
                    (negative_rotation, negative_translation)
                ),
                "positive_joint": torch.cat(
                    (-negative_rotation, -negative_translation)
                ),
                "negative_rotation_only": torch.cat(
                    (negative_rotation, torch.zeros(3, device=device))
                ),
                "positive_rotation_only": torch.cat(
                    (-negative_rotation, torch.zeros(3, device=device))
                ),
                "negative_translation_only": torch.cat(
                    (torch.zeros(3, device=device), negative_translation)
                ),
                "positive_translation_only": torch.cat(
                    (torch.zeros(3, device=device), -negative_translation)
                ),
            }
            poses: dict[str, torch.Tensor] = {}
            losses: dict[str, float] = {}
            rotation_steps: dict[str, float] = {}
            translation_steps: dict[str, float] = {}
            with torch.no_grad():
                for name in VALID_DIRECTIONS:
                    pose = local_se3_pose(deltas[name], raw_pose)
                    poses[name] = pose.detach().cpu()
                    losses[name] = float(
                        _locked_target_loss(
                            pose,
                            target_index=target_index,
                            locked_pairs=moved,
                            distance_fields=distance_fields[branch],
                            world_to_camera=raw_device,
                            intrinsics=intrinsics_device,
                            projection=config.objective.projection,
                            edge=config.objective.edge,
                        )
                    )
                    step_rotation, step_translation = _pose_errors(
                        pose, raw_pose
                    )
                    rotation_steps[name] = step_rotation
                    translation_steps[name] = step_translation
            candidates.append(
                DirectionalCandidate(
                    target_sequence_index=target_index,
                    target_frame_index=int(frames[target_index]),
                    branch=branch,
                    active=1,
                    failure_reason="ok",
                    locked_pairs=len(locked),
                    locked_points=locked_points,
                    rotation_gradient_norm=rotation_norm,
                    translation_gradient_norm=translation_norm,
                    raw_loss=float(loss.detach()),
                    poses=poses,
                    losses=losses,
                    rotation_steps_deg=rotation_steps,
                    translation_steps_native=translation_steps,
                )
            )
    audit = {
        "candidate_generation_gt_fields": 0,
        "candidate_generation_target_argument": 0,
        "fixed_rotation_step_degrees": config.directional.rotation_step_degrees,
        "fixed_translation_step_scene_fraction": (
            config.directional.translation_step_scene_fraction
        ),
        "scene_scale": scene_scale,
        "exclusion_mask_source": exclusion_source,
        "exclusion_mask_semantics": "prompted_object_region_not_dynamic_truth",
        "equal_edge_count_per_frame": int(_equal_counts(branch_edges)),
    }
    return candidates, audit


def score_directional_candidates(
    candidates: Sequence[DirectionalCandidate],
    *,
    target_world_to_camera: torch.Tensor,
    config: EdgeDirectionalAuditConfig,
) -> dict[str, object]:
    """Score already-materialized candidates; no candidate is changed here."""

    if target_world_to_camera.ndim == 4:
        target_world_to_camera = target_world_to_camera[0]
    target_world_to_camera = target_world_to_camera.detach().float().cpu()
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        base = {
            "target_sequence_index": candidate.target_sequence_index,
            "target_frame_index": candidate.target_frame_index,
            "branch": candidate.branch,
            "active": candidate.active,
            "failure_reason": candidate.failure_reason,
            "locked_pairs": candidate.locked_pairs,
            "locked_points": candidate.locked_points,
            "rotation_gradient_norm": candidate.rotation_gradient_norm,
            "translation_gradient_norm": candidate.translation_gradient_norm,
        }
        if not candidate.active:
            rows.append({**base, **_inactive_score_fields()})
            continue
        target_pose = _homogeneous_pose(
            target_world_to_camera[candidate.target_sequence_index]
        )
        errors = {
            name: _pose_errors(candidate.poses[name], target_pose)
            for name in VALID_DIRECTIONS
        }
        row = dict(base)
        for name in VALID_DIRECTIONS:
            row[f"{name}_edge_loss"] = candidate.losses[name]
            row[f"{name}_rotation_step_deg"] = (
                candidate.rotation_steps_deg[name]
            )
            row[f"{name}_translation_step_native"] = (
                candidate.translation_steps_native[name]
            )
            row[f"{name}_rotation_error_deg"] = errors[name][0]
            row[f"{name}_center_error_native"] = errors[name][1]
        raw_rotation, raw_center = errors["raw"]
        negative_rotation, negative_center = errors["negative_joint"]
        positive_rotation, positive_center = errors["positive_joint"]
        row.update(
            {
                "negative_rotation_gain_deg": raw_rotation - negative_rotation,
                "negative_center_gain_native": raw_center - negative_center,
                "negative_better_positive_rotation": int(
                    negative_rotation < positive_rotation
                ),
                "negative_better_positive_center": int(
                    negative_center < positive_center
                ),
                "negative_edge_loss_decreased": int(
                    candidate.losses["negative_joint"]
                    < candidate.losses["raw"]
                ),
                "positive_edge_loss_increased": int(
                    candidate.losses["positive_joint"]
                    > candidate.losses["raw"]
                ),
                "frame_pose_direction_pass": int(
                    negative_rotation < raw_rotation
                    and negative_center < raw_center
                    and negative_rotation < positive_rotation
                    and negative_center < positive_center
                ),
            }
        )
        rows.append(row)
    fold_summary = summarize_directional_folds(rows, config=config)
    decision = summarize_directional_decision(fold_summary, config=config)
    return {"rows": rows, "fold_summary": fold_summary, "decision": decision}


def summarize_directional_folds(
    rows: Sequence[dict[str, object]],
    *,
    config: EdgeDirectionalAuditConfig,
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
                and int(row["active"]) == 1
            ]
            raw_rotation = _mean_field(selected, "raw_rotation_error_deg")
            negative_rotation = _mean_field(
                selected, "negative_joint_rotation_error_deg"
            )
            positive_rotation = _mean_field(
                selected, "positive_joint_rotation_error_deg"
            )
            raw_center = _mean_field(selected, "raw_center_error_native")
            negative_center = _mean_field(
                selected, "negative_joint_center_error_native"
            )
            positive_center = _mean_field(
                selected, "positive_joint_center_error_native"
            )
            rotation_worse_frames = sum(
                int(
                    row["negative_joint_rotation_error_deg"]
                    >= row["raw_rotation_error_deg"]
                )
                for row in selected
            )
            center_worse_frames = sum(
                int(
                    row["negative_joint_center_error_native"]
                    >= row["raw_center_error_native"]
                )
                for row in selected
            )
            rotation_only = _mean_field(
                selected, "negative_rotation_only_rotation_error_deg"
            )
            positive_rotation_only = _mean_field(
                selected, "positive_rotation_only_rotation_error_deg"
            )
            rotation_only_worse_frames = sum(
                int(
                    row["negative_rotation_only_rotation_error_deg"]
                    >= row["raw_rotation_error_deg"]
                )
                for row in selected
            )
            translation_only = _mean_field(
                selected, "negative_translation_only_center_error_native"
            )
            positive_translation_only = _mean_field(
                selected, "positive_translation_only_center_error_native"
            )
            translation_only_worse_frames = sum(
                int(
                    row["negative_translation_only_center_error_native"]
                    >= row["raw_center_error_native"]
                )
                for row in selected
            )
            negative_loss_decrease_rate = _mean_field(
                selected, "negative_edge_loss_decreased"
            )
            positive_loss_increase_rate = _mean_field(
                selected, "positive_edge_loss_increased"
            )
            fold_pass = bool(
                len(selected) == 4
                and negative_rotation < raw_rotation
                and negative_center < raw_center
                and negative_rotation < positive_rotation
                and negative_center < positive_center
                and rotation_worse_frames == 0
                and center_worse_frames == 0
                and negative_loss_decrease_rate == 1.0
                and positive_loss_increase_rate == 1.0
            )
            rotation_only_pass = bool(
                len(selected) == 4
                and rotation_only < raw_rotation
                and rotation_only < positive_rotation_only
                and rotation_only_worse_frames == 0
            )
            translation_only_pass = bool(
                len(selected) == 4
                and translation_only < raw_center
                and translation_only < positive_translation_only
                and translation_only_worse_frames == 0
            )
            summary.append(
                {
                    "fold": fold,
                    "branch": branch,
                    "frames": " ".join(
                        str(row["target_frame_index"]) for row in selected
                    ),
                    "active_frames": len(selected),
                    "raw_rotation_error_deg": raw_rotation,
                    "negative_rotation_error_deg": negative_rotation,
                    "positive_rotation_error_deg": positive_rotation,
                    "rotation_gain_deg": raw_rotation - negative_rotation,
                    "raw_center_error_native": raw_center,
                    "negative_center_error_native": negative_center,
                    "positive_center_error_native": positive_center,
                    "center_gain_native": raw_center - negative_center,
                    "rotation_worse_frames": rotation_worse_frames,
                    "center_worse_frames": center_worse_frames,
                    "negative_edge_loss_decrease_rate": (
                        negative_loss_decrease_rate
                    ),
                    "positive_edge_loss_increase_rate": (
                        positive_loss_increase_rate
                    ),
                    "negative_rotation_only_gain_deg": (
                        raw_rotation - rotation_only
                    ),
                    "positive_rotation_only_error_deg": positive_rotation_only,
                    "rotation_only_worse_frames": rotation_only_worse_frames,
                    "rotation_only_fold_pass": int(rotation_only_pass),
                    "negative_translation_only_center_gain_native": (
                        raw_center - translation_only
                    ),
                    "positive_translation_only_center_error_native": (
                        positive_translation_only
                    ),
                    "translation_only_worse_frames": (
                        translation_only_worse_frames
                    ),
                    "translation_only_fold_pass": int(translation_only_pass),
                    "fold_pass": int(fold_pass),
                }
            )
    return summary


def summarize_directional_decision(
    fold_summary: Sequence[dict[str, object]],
    *,
    config: EdgeDirectionalAuditConfig,
) -> dict[str, object]:
    branch_passes = {
        branch: int(
            len(rows := [row for row in fold_summary if row["branch"] == branch])
            == 3
            and all(int(row["fold_pass"]) == 1 for row in rows)
        )
        for branch in config.branches
    }
    rotation_only_branch_passes = {
        branch: int(
            len(rows := [row for row in fold_summary if row["branch"] == branch])
            == 3
            and all(int(row["rotation_only_fold_pass"]) == 1 for row in rows)
        )
        for branch in config.branches
    }
    translation_only_branch_passes = {
        branch: int(
            len(rows := [row for row in fold_summary if row["branch"] == branch])
            == 3
            and all(int(row["translation_only_fold_pass"]) == 1 for row in rows)
        )
        for branch in config.branches
    }
    primary_rows = [
        row for row in fold_summary if row["branch"] == config.primary_branch
    ]
    controls = [
        branch for branch in config.branches if branch != config.primary_branch
    ]
    uniquely_better = bool(branch_passes[config.primary_branch])
    for primary in primary_rows:
        for control in controls:
            comparison = next(
                row
                for row in fold_summary
                if row["branch"] == control and row["fold"] == primary["fold"]
            )
            uniquely_better &= (
                primary["negative_rotation_error_deg"]
                < comparison["negative_rotation_error_deg"]
                and primary["negative_center_error_native"]
                < comparison["negative_center_error_native"]
            )
    return {
        "branch_all_fold_pass": branch_passes,
        "rotation_only_branch_all_fold_pass": rotation_only_branch_passes,
        "translation_only_branch_all_fold_pass": (
            translation_only_branch_passes
        ),
        "primary_branch": config.primary_branch,
        "primary_all_fold_pass": branch_passes[config.primary_branch],
        "primary_rotation_only_all_fold_pass": (
            rotation_only_branch_passes[config.primary_branch]
        ),
        "primary_translation_only_all_fold_pass": (
            translation_only_branch_passes[config.primary_branch]
        ),
        "prompted_object_exclusion_unique_pass": int(uniquely_better),
        "any_edge_direction_all_fold_pass": int(any(branch_passes.values())),
        "any_rotation_only_all_fold_pass": int(
            any(rotation_only_branch_passes.values())
        ),
        "any_translation_only_all_fold_pass": int(
            any(translation_only_branch_passes.values())
        ),
        "gt_role": "scoring_only_after_fixed_candidate_generation",
        "target_pose_source": "target_pose_encoding_native_reference_aligned",
        "selected_pose_modified": 0,
        "claim": "directional_pose_evidence_not_deployable_pose_improvement",
    }


def local_se3_pose(tangent: torch.Tensor, raw_w2c: torch.Tensor) -> torch.Tensor:
    """Left-compose an unbounded physical local se(3) correction."""

    omega = tangent[:3]
    x, y, z = omega.unbind()
    zero = torch.zeros((), dtype=tangent.dtype, device=tangent.device)
    skew = torch.stack(
        (
            torch.stack((zero, -z, y)),
            torch.stack((z, zero, -x)),
            torch.stack((-y, x, zero)),
        )
    )
    top = torch.cat((skew, tangent[3:, None]), dim=1)
    generator = torch.cat(
        (top, torch.zeros(1, 4, dtype=tangent.dtype, device=tangent.device))
    )
    return torch.matrix_exp(generator) @ raw_w2c


def _unit_or_zero(value: torch.Tensor, minimum: float) -> torch.Tensor:
    norm = value.norm()
    if not bool(torch.isfinite(norm)) or float(norm) < float(minimum):
        return torch.zeros_like(value)
    return value / norm


def _inactive_candidate(
    target_index: int,
    target_frame: int,
    branch: str,
    locked_points: int,
    reason: str,
) -> DirectionalCandidate:
    return DirectionalCandidate(
        target_sequence_index=target_index,
        target_frame_index=target_frame,
        branch=branch,
        active=0,
        failure_reason=reason,
        locked_pairs=0,
        locked_points=locked_points,
        rotation_gradient_norm=float("nan"),
        translation_gradient_norm=float("nan"),
        raw_loss=float("nan"),
        poses={},
        losses={},
        rotation_steps_deg={},
        translation_steps_native={},
    )


def _inactive_score_fields() -> dict[str, object]:
    output: dict[str, object] = {
        "negative_rotation_gain_deg": float("nan"),
        "negative_center_gain_native": float("nan"),
        "negative_better_positive_rotation": 0,
        "negative_better_positive_center": 0,
        "negative_edge_loss_decreased": 0,
        "positive_edge_loss_increased": 0,
        "frame_pose_direction_pass": 0,
    }
    for name in VALID_DIRECTIONS:
        for suffix in (
            "edge_loss",
            "rotation_step_deg",
            "translation_step_native",
            "rotation_error_deg",
            "center_error_native",
        ):
            output[f"{name}_{suffix}"] = float("nan")
    return output


def _equal_counts(branch_edges: dict[str, torch.Tensor]) -> bool:
    counts = [value.sum(dim=(-2, -1)) for value in branch_edges.values()]
    return bool(counts) and all(torch.equal(counts[0], value) for value in counts[1:])


def _mean_field(rows: Sequence[dict[str, object]], key: str) -> float:
    values = [float(row[key]) for row in rows if math.isfinite(float(row[key]))]
    return sum(values) / len(values) if values else float("nan")


def _path(value: str | Path | None, base: Path) -> Path:
    if value is None:
        raise ValueError("Missing required E1 path.")
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _validate_config(config: EdgeDirectionalAuditConfig) -> None:
    if not config.clip_name:
        raise ValueError("e1.clip_name is required.")
    if len(config.evaluation_frames) != 12:
        raise ValueError("E1 requires exactly twelve evaluation frames.")
    valid = {value for value in config.objective.branches if value != "raw"}
    unknown = set(config.branches) - valid
    if unknown:
        raise ValueError(f"E1 branches are absent from E0={sorted(unknown)}.")
    if config.primary_branch not in config.branches:
        raise ValueError("e1.primary_branch must be one of e1.branches.")
    if config.directional.rotation_step_degrees <= 0.0:
        raise ValueError("rotation_step_degrees must be positive.")
    if config.directional.translation_step_scene_fraction <= 0.0:
        raise ValueError("translation_step_scene_fraction must be positive.")
    if config.directional.min_gradient_norm <= 0.0:
        raise ValueError("min_gradient_norm must be positive.")
