"""Reusable pose, ray-center, and paired-pointmap evaluation primitives."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from test_sam.data import resolve_manifest_path


@dataclass(frozen=True)
class GroundTruthSequence:
    image_paths: tuple[Path, ...]
    world_to_camera: torch.Tensor
    intrinsics: torch.Tensor


@dataclass(frozen=True)
class PoseSequence:
    world_to_camera: torch.Tensor
    camera_to_world_rotation: torch.Tensor
    camera_centers: torch.Tensor
    rotation_quality_rows: tuple[dict, ...]


@dataclass(frozen=True)
class SimilarityAlignment:
    name: str
    scale: float
    rotation: torch.Tensor
    translation: torch.Tensor
    fit_source: str


@dataclass(frozen=True)
class RayFitConfig:
    trim_quantile: float = 0.80
    max_iterations: int = 4
    min_points: int = 1024
    max_points: int = 65536
    max_condition_number: float = 1e8


@dataclass(frozen=True)
class RayCenterFit:
    center: torch.Tensor
    fit_accepted: bool
    status: str
    candidate_points: int
    sampled_points: int
    retained_points: int
    solve_iterations: int
    condition_number: float
    all_residual_mean: float
    all_residual_median: float
    all_residual_rmse: float
    all_residual_p90: float
    retained_residual_mean: float
    retained_residual_median: float
    retained_residual_rmse: float
    retained_residual_p90: float


def _load_ground_truth_sequence(
    manifest_path: str | Path,
    *,
    scene_id: str,
    frame_indices: Sequence[int],
) -> GroundTruthSequence:
    manifest_path = Path(manifest_path).expanduser().resolve()
    with manifest_path.open("r", encoding="utf8") as handle:
        manifest = json.load(handle)
    scene = next(
        (
            item
            for item in manifest.get("scenes", [])
            if item.get("scene_id") == scene_id
        ),
        None,
    )
    if scene is None:
        raise ValueError(f"Scene {scene_id!r} is missing from {manifest_path}.")
    frames = scene.get("frames", [])
    image_paths = []
    world_to_camera = []
    intrinsics = []
    for frame_index in frame_indices:
        frame_index = int(frame_index)
        if not 0 <= frame_index < len(frames):
            raise ValueError(
                f"Frame {frame_index} is outside scene length {len(frames)}."
            )
        frame = frames[frame_index]
        image_paths.append(
            resolve_manifest_path(frame["image_path"], manifest_path)
        )
        world_to_camera.append(
            _read_matrix(
                frame.get("world_to_camera"),
                shape=(4, 4),
                field="world_to_camera",
                frame_index=frame_index,
            )
        )
        intrinsics.append(
            _read_matrix(
                frame.get("intrinsics"),
                shape=(3, 3),
                field="intrinsics",
                frame_index=frame_index,
            )
        )
    return GroundTruthSequence(
        image_paths=tuple(image_paths),
        world_to_camera=torch.from_numpy(np.stack(world_to_camera)).double(),
        intrinsics=torch.from_numpy(np.stack(intrinsics)).double(),
    )


def _read_matrix(
    value,
    *,
    shape: tuple[int, int],
    field: str,
    frame_index: int,
) -> np.ndarray:
    if value is None:
        raise ValueError(
            f"Frame {frame_index} has no manifest field {field!r}."
        )
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != shape or not np.isfinite(matrix).all():
        raise ValueError(
            f"Frame {frame_index} field {field!r} must be finite {shape}, "
            f"got {matrix.shape}."
        )
    return matrix


def _prepare_pose_sequence(
    world_to_camera: torch.Tensor,
    *,
    frame_indices: Sequence[int],
    source: str,
) -> PoseSequence:
    matrices = _homogeneous(world_to_camera).double().cpu()
    rotations = []
    centers = []
    quality_rows = []
    for sequence_index, matrix in enumerate(matrices):
        raw_rotation = matrix[:3, :3]
        rotation = _project_rotation(raw_rotation)
        center = -(rotation.T @ matrix[:3, 3])
        rotations.append(rotation.T)
        centers.append(center)
        quality_rows.append(
            {
                "source": source,
                "sequence_index": sequence_index,
                "frame_index": int(frame_indices[sequence_index]),
                "raw_rotation_determinant": float(
                    torch.det(raw_rotation)
                ),
                "raw_rotation_orthogonality_error": float(
                    torch.linalg.matrix_norm(
                        raw_rotation.T @ raw_rotation
                        - torch.eye(3, dtype=raw_rotation.dtype)
                    )
                ),
                "projection_change_frobenius": float(
                    torch.linalg.matrix_norm(rotation - raw_rotation)
                ),
            }
        )
    return _pose_sequence_from_centers(
        torch.stack(rotations),
        torch.stack(centers),
        rotation_quality_rows=tuple(quality_rows),
    )


def _prepare_ray_inputs(
    world_points: torch.Tensor,
    confidence: torch.Tensor,
    intrinsics: torch.Tensor,
    camera_to_world_rotation: torch.Tensor,
    *,
    confidence_threshold: float,
    max_points: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    height, width = world_points.shape[:2]
    rows, columns = torch.meshgrid(
        torch.arange(height, dtype=torch.float64),
        torch.arange(width, dtype=torch.float64),
        indexing="ij",
    )
    pixels = torch.stack(
        [columns, rows, torch.ones_like(columns)],
        dim=-1,
    )
    camera_directions = pixels @ torch.linalg.inv(intrinsics.double()).T
    world_directions = (
        camera_directions @ camera_to_world_rotation.double().T
    )
    norms = torch.linalg.vector_norm(world_directions, dim=-1)
    world_directions = world_directions / norms.clamp_min(1e-12)[..., None]
    valid = (
        torch.isfinite(world_points).all(dim=-1)
        & torch.isfinite(confidence)
        & (confidence >= float(confidence_threshold))
        & torch.isfinite(world_directions).all(dim=-1)
        & (norms > 1e-12)
    )
    flat_indices = torch.nonzero(
        valid.reshape(-1),
        as_tuple=False,
    )[:, 0]
    candidate_points = int(flat_indices.numel())
    if candidate_points > int(max_points):
        positions = torch.linspace(
            0,
            candidate_points - 1,
            steps=int(max_points),
            dtype=torch.float64,
        ).round().long()
        flat_indices = flat_indices.index_select(0, positions)
    points = world_points.reshape(-1, 3).double().index_select(
        0,
        flat_indices,
    )
    directions = world_directions.reshape(-1, 3).index_select(
        0,
        flat_indices,
    )
    weights = confidence.reshape(-1).double().index_select(
        0,
        flat_indices,
    ).clamp_min(1e-6)
    if weights.numel():
        weights = weights / weights.mean().clamp_min(1e-12)
    return points, directions, weights, candidate_points


def _fit_ray_center(
    points: torch.Tensor,
    directions: torch.Tensor,
    weights: torch.Tensor,
    *,
    candidate_points: int,
    fallback_center: torch.Tensor,
    robust_trim: bool,
    config: RayFitConfig,
) -> RayCenterFit:
    _validate_ray_fit_config(config)
    sampled_points = int(points.shape[0])
    if sampled_points < int(config.min_points):
        return _fallback_ray_center(
            fallback_center,
            points,
            directions,
            candidate_points=candidate_points,
            status=(
                f"fallback_insufficient_points:{sampled_points}"
                f"<{int(config.min_points)}"
            ),
        )
    active = torch.ones(sampled_points, dtype=torch.bool)
    center = fallback_center.double().cpu()
    condition_number = float("nan")
    solve_iterations = 0
    try:
        for solve_index in range(int(config.max_iterations)):
            center, condition_number = _solve_weighted_ray_center(
                points[active],
                directions[active],
                weights[active],
            )
            solve_iterations += 1
            if (
                not torch.isfinite(center).all()
                or not math.isfinite(condition_number)
                or condition_number > float(config.max_condition_number)
            ):
                return _fallback_ray_center(
                    fallback_center,
                    points,
                    directions,
                    candidate_points=candidate_points,
                    status=(
                        "fallback_ill_conditioned:"
                        f"{condition_number:.6g}"
                    ),
                    solve_iterations=solve_iterations,
                    condition_number=condition_number,
                )
            if not robust_trim:
                break
            residuals = _ray_residuals(points, directions, center)
            cutoff = torch.quantile(
                residuals,
                float(config.trim_quantile),
            )
            new_active = torch.isfinite(residuals) & (residuals <= cutoff)
            if (
                int(new_active.sum()) < int(config.min_points)
                or torch.equal(new_active, active)
                or solve_index + 1 >= int(config.max_iterations)
            ):
                break
            active = new_active
    except RuntimeError as error:
        return _fallback_ray_center(
            fallback_center,
            points,
            directions,
            candidate_points=candidate_points,
            status=f"fallback_linear_solve:{type(error).__name__}",
            solve_iterations=solve_iterations,
            condition_number=condition_number,
        )
    all_stats = _distance_statistics(
        _ray_residuals(points, directions, center)
    )
    retained_stats = _distance_statistics(
        _ray_residuals(points[active], directions[active], center)
    )
    return RayCenterFit(
        center=center,
        fit_accepted=True,
        status="accepted_trimmed" if robust_trim else "accepted_all",
        candidate_points=int(candidate_points),
        sampled_points=sampled_points,
        retained_points=int(active.sum()),
        solve_iterations=solve_iterations,
        condition_number=condition_number,
        all_residual_mean=all_stats["mean"],
        all_residual_median=all_stats["median"],
        all_residual_rmse=all_stats["rmse"],
        all_residual_p90=all_stats["p90"],
        retained_residual_mean=retained_stats["mean"],
        retained_residual_median=retained_stats["median"],
        retained_residual_rmse=retained_stats["rmse"],
        retained_residual_p90=retained_stats["p90"],
    )


def _validate_ray_fit_config(config: RayFitConfig) -> None:
    if not 0.0 < float(config.trim_quantile) < 1.0:
        raise ValueError("ray trim_quantile must be in (0,1).")
    if int(config.max_iterations) < 1:
        raise ValueError("ray max_iterations must be positive.")
    if int(config.min_points) < 3:
        raise ValueError("ray min_points must be at least 3.")
    if int(config.max_points) < int(config.min_points):
        raise ValueError("ray max_points must be at least ray min_points.")
    if float(config.max_condition_number) <= 1.0:
        raise ValueError("ray max_condition_number must be greater than 1.")


def _solve_weighted_ray_center(
    points: torch.Tensor,
    directions: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    identity = torch.eye(3, dtype=torch.float64)
    projectors = identity[None] - (
        directions[:, :, None] * directions[:, None, :]
    )
    normal_matrix = torch.einsum("n,nij->ij", weights, projectors)
    right_hand_side = torch.einsum(
        "n,nij,nj->i",
        weights,
        projectors,
        points,
    )
    condition_number = float(torch.linalg.cond(normal_matrix))
    return (
        torch.linalg.solve(normal_matrix, right_hand_side),
        condition_number,
    )


def _ray_residuals(
    points: torch.Tensor,
    directions: torch.Tensor,
    center: torch.Tensor,
) -> torch.Tensor:
    offsets = points - center
    along_ray = (offsets * directions).sum(dim=-1, keepdim=True)
    return torch.linalg.vector_norm(
        offsets - along_ray * directions,
        dim=-1,
    )


def _fallback_ray_center(
    center: torch.Tensor,
    points: torch.Tensor,
    directions: torch.Tensor,
    *,
    candidate_points: int,
    status: str,
    solve_iterations: int = 0,
    condition_number: float = float("nan"),
) -> RayCenterFit:
    stats = _distance_statistics(
        _ray_residuals(points, directions, center)
    )
    return RayCenterFit(
        center=center.double().cpu(),
        fit_accepted=False,
        status=status,
        candidate_points=int(candidate_points),
        sampled_points=int(points.shape[0]),
        retained_points=int(points.shape[0]),
        solve_iterations=int(solve_iterations),
        condition_number=float(condition_number),
        all_residual_mean=stats["mean"],
        all_residual_median=stats["median"],
        all_residual_rmse=stats["rmse"],
        all_residual_p90=stats["p90"],
        retained_residual_mean=stats["mean"],
        retained_residual_median=stats["median"],
        retained_residual_rmse=stats["rmse"],
        retained_residual_p90=stats["p90"],
    )


def _distance_statistics(values: torch.Tensor) -> dict[str, float]:
    finite = values[torch.isfinite(values)]
    if not finite.numel():
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "rmse": float("nan"),
            "p90": float("nan"),
        }
    return {
        "mean": float(finite.mean()),
        "median": float(finite.median()),
        "rmse": _rmse(finite),
        "p90": float(torch.quantile(finite, 0.90)),
    }


def _pose_sequence_from_centers(
    camera_to_world_rotation: torch.Tensor,
    camera_centers: torch.Tensor,
    *,
    rotation_quality_rows: tuple[dict, ...] = (),
) -> PoseSequence:
    rotations = camera_to_world_rotation.double().cpu()
    centers = camera_centers.double().cpu()
    world_to_camera = torch.eye(
        4,
        dtype=torch.float64,
    ).repeat(centers.shape[0], 1, 1)
    world_to_camera[:, :3, :3] = rotations.transpose(1, 2)
    world_to_camera[:, :3, 3] = -torch.einsum(
        "tij,tj->ti",
        world_to_camera[:, :3, :3],
        centers,
    )
    return PoseSequence(
        world_to_camera=world_to_camera,
        camera_to_world_rotation=rotations,
        camera_centers=centers,
        rotation_quality_rows=rotation_quality_rows,
    )


def _homogeneous(world_to_camera: torch.Tensor) -> torch.Tensor:
    if world_to_camera.ndim != 3:
        raise ValueError(
            "world_to_camera must have shape [T,3,4] or [T,4,4]."
        )
    if tuple(world_to_camera.shape[-2:]) == (4, 4):
        return world_to_camera
    if tuple(world_to_camera.shape[-2:]) != (3, 4):
        raise ValueError(
            "world_to_camera must have shape [T,3,4] or [T,4,4]."
        )
    result = torch.eye(
        4,
        dtype=world_to_camera.dtype,
        device=world_to_camera.device,
    ).repeat(world_to_camera.shape[0], 1, 1)
    result[:, :3] = world_to_camera
    return result


def _project_rotation(rotation: torch.Tensor) -> torch.Tensor:
    left, _, right_t = torch.linalg.svd(rotation)
    projected = left @ right_t
    if torch.det(projected) < 0:
        left = left.clone()
        left[:, -1] *= -1
        projected = left @ right_t
    return projected


def _reference_pose_alignment(
    predicted: PoseSequence,
    target: PoseSequence,
    *,
    reference_index: int,
    scale: float,
) -> SimilarityAlignment:
    rotation = _project_rotation(
        target.camera_to_world_rotation[reference_index]
        @ predicted.camera_to_world_rotation[reference_index].T
    )
    translation = (
        target.camera_centers[reference_index]
        - float(scale)
        * (rotation @ predicted.camera_centers[reference_index])
    )
    return SimilarityAlignment(
        name="reference_pose_point_scale",
        scale=float(scale),
        rotation=rotation,
        translation=translation,
        fit_source=(
            "reference pose center/orientation with scale from "
            "reference_point_sim3"
        ),
    )


def _evaluate_pose_alignment(
    alignment: SimilarityAlignment,
    *,
    predicted: PoseSequence,
    target: PoseSequence,
    frame_indices: Sequence[int],
    reference_index: int,
) -> tuple[dict, list[dict], list[dict]]:
    centers = float(alignment.scale) * (
        predicted.camera_centers @ alignment.rotation.T
    ) + alignment.translation
    rotations = torch.einsum(
        "ij,tjk->tik",
        alignment.rotation,
        predicted.camera_to_world_rotation,
    )
    translation_errors = torch.linalg.vector_norm(
        centers - target.camera_centers,
        dim=-1,
    )
    rotation_errors = torch.tensor(
        [
            _rotation_error_degrees(
                rotations[index],
                target.camera_to_world_rotation[index],
            )
            for index in range(len(frame_indices))
        ],
        dtype=torch.float64,
    )
    frame_rows = []
    for index, frame_index in enumerate(frame_indices):
        row = {
            "alignment": alignment.name,
            "alignment_fit_source": alignment.fit_source,
            "sequence_index": index,
            "frame_index": int(frame_index),
            "is_reference": int(index == int(reference_index)),
            "translation_error": float(translation_errors[index]),
            "rotation_error_degrees": float(rotation_errors[index]),
        }
        _add_vector(row, "predicted_center_raw", predicted.camera_centers[index])
        _add_vector(row, "predicted_center_aligned", centers[index])
        _add_vector(row, "gt_center", target.camera_centers[index])
        frame_rows.append(row)
    predicted_c2w = _camera_to_world_matrices(rotations, centers)
    target_c2w = _camera_to_world_matrices(
        target.camera_to_world_rotation,
        target.camera_centers,
    )
    rpe_rows = []
    for first in range(len(frame_indices) - 1):
        second = first + 1
        predicted_delta = (
            torch.linalg.inv(predicted_c2w[first])
            @ predicted_c2w[second]
        )
        target_delta = (
            torch.linalg.inv(target_c2w[first])
            @ target_c2w[second]
        )
        error = torch.linalg.inv(target_delta) @ predicted_delta
        rpe_rows.append(
            {
                "alignment": alignment.name,
                "first_sequence_index": first,
                "second_sequence_index": second,
                "first_frame_index": int(frame_indices[first]),
                "second_frame_index": int(frame_indices[second]),
                "source_frame_gap": int(
                    frame_indices[second] - frame_indices[first]
                ),
                "translation_error": float(
                    torch.linalg.vector_norm(error[:3, 3])
                ),
                "rotation_error_degrees": _rotation_angle_degrees(
                    error[:3, :3]
                ),
                "predicted_motion_translation": float(
                    torch.linalg.vector_norm(predicted_delta[:3, 3])
                ),
                "gt_motion_translation": float(
                    torch.linalg.vector_norm(target_delta[:3, 3])
                ),
                "predicted_motion_rotation_degrees": (
                    _rotation_angle_degrees(predicted_delta[:3, :3])
                ),
                "gt_motion_rotation_degrees": _rotation_angle_degrees(
                    target_delta[:3, :3]
                ),
            }
        )
    rpe_translation = torch.tensor(
        [row["translation_error"] for row in rpe_rows],
        dtype=torch.float64,
    )
    rpe_rotation = torch.tensor(
        [row["rotation_error_degrees"] for row in rpe_rows],
        dtype=torch.float64,
    )
    summary = {
        "alignment": alignment.name,
        "alignment_fit_source": alignment.fit_source,
        "alignment_scale": float(alignment.scale),
        "alignment_rotation": _flatten_matrix(alignment.rotation),
        "alignment_translation": _flatten_matrix(alignment.translation),
        "evaluated_frames": len(frame_indices),
        "ate_rmse": _rmse(translation_errors),
        "translation_error_mean": float(translation_errors.mean()),
        "translation_error_median": float(translation_errors.median()),
        "translation_error_max": float(translation_errors.max()),
        "rotation_error_mean_degrees": float(rotation_errors.mean()),
        "rotation_error_median_degrees": float(rotation_errors.median()),
        "rotation_error_max_degrees": float(rotation_errors.max()),
        "adjacent_rpe_pairs": len(rpe_rows),
        "rpe_translation_rmse": _rmse(rpe_translation),
        "rpe_translation_mean": (
            float(rpe_translation.mean())
            if rpe_translation.numel()
            else float("nan")
        ),
        "rpe_rotation_mean_degrees": (
            float(rpe_rotation.mean())
            if rpe_rotation.numel()
            else float("nan")
        ),
        "rpe_rotation_max_degrees": (
            float(rpe_rotation.max())
            if rpe_rotation.numel()
            else float("nan")
        ),
    }
    return summary, frame_rows, rpe_rows


def _camera_to_world_matrices(
    rotations: torch.Tensor,
    centers: torch.Tensor,
) -> torch.Tensor:
    result = torch.eye(4, dtype=torch.float64).repeat(
        rotations.shape[0],
        1,
        1,
    )
    result[:, :3, :3] = rotations
    result[:, :3, 3] = centers
    return result


def _rotation_error_degrees(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> float:
    return _rotation_angle_degrees(target.T @ predicted)


def _rotation_angle_degrees(rotation: torch.Tensor) -> float:
    cosine = ((torch.trace(rotation) - 1.0) * 0.5).clamp(-1.0, 1.0)
    return float(torch.rad2deg(torch.acos(cosine)))


def _all_pair_pose_metrics(
    predicted: PoseSequence,
    target: PoseSequence,
    *,
    frame_indices: Sequence[int],
) -> list[dict]:
    rows = []
    for first in range(len(frame_indices) - 1):
        for second in range(first + 1, len(frame_indices)):
            predicted_relative = (
                predicted.world_to_camera[first]
                @ torch.linalg.inv(predicted.world_to_camera[second])
            )
            target_relative = (
                target.world_to_camera[first]
                @ torch.linalg.inv(target.world_to_camera[second])
            )
            rotation_error = _rotation_error_degrees(
                predicted_relative[:3, :3],
                target_relative[:3, :3],
            )
            translation_error = _translation_direction_error_degrees(
                predicted_relative[:3, 3],
                target_relative[:3, 3],
            )
            rows.append(
                {
                    "first_sequence_index": first,
                    "second_sequence_index": second,
                    "first_frame_index": int(frame_indices[first]),
                    "second_frame_index": int(frame_indices[second]),
                    "source_frame_gap": int(
                        frame_indices[second] - frame_indices[first]
                    ),
                    "rotation_error_degrees": rotation_error,
                    "translation_direction_error_degrees": translation_error,
                    "max_pair_error_degrees": max(
                        rotation_error,
                        translation_error,
                    ),
                }
            )
    return rows


def _translation_direction_error_degrees(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> float:
    predicted_norm = torch.linalg.vector_norm(predicted)
    target_norm = torch.linalg.vector_norm(target)
    if float(predicted_norm) <= 1e-12 or float(target_norm) <= 1e-12:
        return float("nan")
    cosine = torch.dot(
        predicted / predicted_norm,
        target / target_norm,
    ).abs().clamp(0.0, 1.0)
    return float(torch.rad2deg(torch.acos(cosine)))


def _summarize_pose_pairs(rows: Sequence[dict]) -> list[dict]:
    valid = [
        row
        for row in rows
        if math.isfinite(float(row["rotation_error_degrees"]))
        and math.isfinite(
            float(row["translation_direction_error_degrees"])
        )
    ]
    result = {
        "protocol": "streamvggt_official_style_all_pairs",
        "translation_scale_ambiguity": "direction only, sign ambiguous",
        "pairs": len(rows),
        "valid_pairs": len(valid),
        "rotation_error_mean_degrees": _mean_rows(
            valid,
            "rotation_error_degrees",
        ),
        "rotation_error_median_degrees": _median_rows(
            valid,
            "rotation_error_degrees",
        ),
        "translation_direction_error_mean_degrees": _mean_rows(
            valid,
            "translation_direction_error_degrees",
        ),
        "translation_direction_error_median_degrees": _median_rows(
            valid,
            "translation_direction_error_degrees",
        ),
        "max_pair_error_mean_degrees": _mean_rows(
            valid,
            "max_pair_error_degrees",
        ),
    }
    for threshold in (5, 10, 30):
        result[f"rotation_accuracy_at_{threshold}deg"] = _accuracy(
            valid,
            "rotation_error_degrees",
            threshold,
        )
        result[f"translation_accuracy_at_{threshold}deg"] = _accuracy(
            valid,
            "translation_direction_error_degrees",
            threshold,
        )
        result[f"joint_accuracy_at_{threshold}deg"] = _accuracy(
            valid,
            "max_pair_error_degrees",
            threshold,
        )
    return [result]


def _pointmap_frame_metrics(
    *,
    aligned_points: torch.Tensor,
    gt_points: torch.Tensor,
    confidence: torch.Tensor,
    confidence_threshold: float,
    frame_indices: Sequence[int],
    reference_index: int,
) -> list[dict]:
    rows = []
    for index, frame_index in enumerate(frame_indices):
        predicted = aligned_points[index].reshape(-1, 3).double()
        target = gt_points[index].reshape(-1, 3).double()
        weights = confidence[index].reshape(-1).double()
        valid = (
            torch.isfinite(predicted).all(dim=-1)
            & torch.isfinite(target).all(dim=-1)
            & torch.isfinite(weights)
            & (weights >= float(confidence_threshold))
        )
        distances = torch.linalg.vector_norm(
            predicted[valid] - target[valid],
            dim=-1,
        )
        if not distances.numel():
            rows.append(
                {
                    "sequence_index": index,
                    "frame_index": int(frame_index),
                    "is_reference": int(index == int(reference_index)),
                    "paired_points": 0,
                    "paired_distance_mean": float("nan"),
                    "paired_distance_median": float("nan"),
                    "paired_distance_rmse": float("nan"),
                    "paired_distance_p90": float("nan"),
                }
            )
            continue
        rows.append(
            {
                "sequence_index": index,
                "frame_index": int(frame_index),
                "is_reference": int(index == int(reference_index)),
                "paired_points": int(distances.numel()),
                "paired_distance_mean": float(distances.mean()),
                "paired_distance_median": float(distances.median()),
                "paired_distance_rmse": _rmse(distances),
                "paired_distance_p90": float(
                    torch.quantile(distances, 0.90)
                ),
            }
        )
    return rows


def _summarize_pointmap_rows(rows: Sequence[dict]) -> list[dict]:
    groups = {
        "all_frames": list(rows),
        "reference_frame": [row for row in rows if row["is_reference"]],
        "nonreference_frames": [
            row for row in rows if not row["is_reference"]
        ],
    }
    result = []
    for group, selected in groups.items():
        valid = [
            row
            for row in selected
            if math.isfinite(float(row["paired_distance_rmse"]))
        ]
        result.append(
            {
                "group": group,
                "frames": len(selected),
                "valid_frames": len(valid),
                "paired_points": sum(
                    int(row["paired_points"]) for row in valid
                ),
                "mean_frame_distance_mean": _mean_rows(
                    valid,
                    "paired_distance_mean",
                ),
                "mean_frame_distance_median": _mean_rows(
                    valid,
                    "paired_distance_median",
                ),
                "mean_frame_distance_rmse": _mean_rows(
                    valid,
                    "paired_distance_rmse",
                ),
                "max_frame_distance_rmse": _max_rows(
                    valid,
                    "paired_distance_rmse",
                ),
                "mean_frame_distance_p90": _mean_rows(
                    valid,
                    "paired_distance_p90",
                ),
            }
        )
    return result


def _mean_rows(rows: Sequence[dict], key: str) -> float:
    values = [
        float(row[key])
        for row in rows
        if math.isfinite(float(row[key]))
    ]
    return float(np.mean(values)) if values else float("nan")


def _max_rows(rows: Sequence[dict], key: str) -> float:
    values = [
        float(row[key])
        for row in rows
        if math.isfinite(float(row[key]))
    ]
    return max(values) if values else float("nan")


def _median_rows(rows: Sequence[dict], key: str) -> float:
    values = [
        float(row[key])
        for row in rows
        if math.isfinite(float(row[key]))
    ]
    return float(np.median(values)) if values else float("nan")


def _accuracy(
    rows: Sequence[dict],
    key: str,
    threshold: float,
) -> float:
    values = [
        float(row[key])
        for row in rows
        if math.isfinite(float(row[key]))
    ]
    if not values:
        return float("nan")
    return sum(value < float(threshold) for value in values) / len(values)


def _rmse(values: torch.Tensor) -> float:
    if not values.numel():
        return float("nan")
    return float(torch.sqrt(values.square().mean()))


def _add_vector(row: dict, prefix: str, vector: torch.Tensor) -> None:
    row[f"{prefix}_x"] = float(vector[0])
    row[f"{prefix}_y"] = float(vector[1])
    row[f"{prefix}_z"] = float(vector[2])


def _flatten_matrix(value: torch.Tensor) -> str:
    return " ".join(f"{float(item):.9g}" for item in value.flatten())
