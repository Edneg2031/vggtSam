"""Evaluation-only StreamVGGT camera-pose and pointmap diagnostics.

The diagnostic does not consume SAM3 masks or modify StreamVGGT outputs. It
compares the frozen model with ScanNet++ COLMAP poses under explicit Sim(3)
gauge choices and reports paired pointmap errors after one fixed reference
frame alignment.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from test_sam.coordinates import streamvggt_image_transform
from test_sam.data import resolve_manifest_path

from .backbones.streamvggt_wrapper import StreamVGGTWrapper
from .config import ExperimentConfig, load_config
from .instance_map_evaluation import prepare_map_evaluation


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


def main() -> None:
    args = _parse_args()
    overrides = {
        key: value
        for key, value in {
            "manifest": args.manifest,
            "scene_id": args.scene_id,
            "frame_indices": args.frame_indices,
            "geometry_device": args.geometry_device,
            "output_dir": args.output_dir,
        }.items()
        if value is not None
    }
    config = load_config(args.config, overrides)
    run_diagnostics(
        config,
        reference_sequence_index=args.reference_sequence_index,
    )


def run_diagnostics(
    config: ExperimentConfig,
    *,
    reference_sequence_index: int,
) -> None:
    """Extract StreamVGGT once and write pose/pointmap evaluation tables."""

    torch.manual_seed(0)
    np.random.seed(0)
    reference_sequence_index = int(reference_sequence_index)
    if not 0 <= reference_sequence_index < len(config.frame_indices):
        raise ValueError(
            "reference_sequence_index must select one configured frame."
        )
    config.output_dir.mkdir(parents=True, exist_ok=True)

    ground_truth = _load_ground_truth_sequence(
        config.manifest,
        scene_id=config.scene_id,
        frame_indices=config.frame_indices,
    )
    geometry = StreamVGGTWrapper(
        repo_path=config.streamvggt_repo,
        checkpoint_path=config.streamvggt_checkpoint,
        device=config.geometry_device,
        image_mode=config.image_mode,
        streaming_cache=config.streaming_cache,
    ).load().extract(ground_truth.image_paths)
    if geometry.world_to_camera.shape[0] != len(config.frame_indices):
        raise RuntimeError(
            "StreamVGGT pose count does not match the selected sequence: "
            f"{geometry.world_to_camera.shape[0]} vs {len(config.frame_indices)}."
        )

    map_context = prepare_map_evaluation(
        config,
        scene_id=config.scene_id,
        frame_indices=config.frame_indices,
        geometry=geometry,
        reference_frame_idx=reference_sequence_index,
    )
    predicted = _prepare_pose_sequence(
        geometry.world_to_camera,
        frame_indices=config.frame_indices,
        source="streamvggt",
    )
    target = _prepare_pose_sequence(
        ground_truth.world_to_camera,
        frame_indices=config.frame_indices,
        source="scannetpp_colmap",
    )

    point_alignment = SimilarityAlignment(
        name="reference_point_sim3",
        scale=float(map_context.sim3_scale),
        rotation=map_context.sim3_rotation.double(),
        translation=map_context.sim3_translation.double(),
        fit_source="paired full-scene points from the reference frame only",
    )
    reference_pose_alignment = _reference_pose_alignment(
        predicted,
        target,
        reference_index=reference_sequence_index,
        scale=point_alignment.scale,
    )
    trajectory_alignment = _trajectory_alignment(predicted, target)
    alignments = (
        point_alignment,
        reference_pose_alignment,
        trajectory_alignment,
    )

    pose_summary_rows = []
    pose_frame_rows = []
    pose_rpe_rows = []
    for alignment in alignments:
        summary, frames, rpe = _evaluate_pose_alignment(
            alignment,
            predicted=predicted,
            target=target,
            frame_indices=config.frame_indices,
            reference_index=reference_sequence_index,
        )
        pose_summary_rows.append(summary)
        pose_frame_rows.extend(frames)
        pose_rpe_rows.extend(rpe)

    pose_pair_rows = _all_pair_pose_metrics(
        predicted,
        target,
        frame_indices=config.frame_indices,
    )
    pose_pair_summary_rows = _summarize_pose_pairs(pose_pair_rows)
    pointmap_frame_rows = _pointmap_frame_metrics(
        aligned_points=map_context.aligned_world_points,
        gt_points=map_context.gt_pointmaps,
        confidence=geometry.confidence,
        confidence_threshold=config.point_cloud_confidence_threshold,
        frame_indices=config.frame_indices,
        reference_index=reference_sequence_index,
    )
    pointmap_summary_rows = _summarize_pointmap_rows(pointmap_frame_rows)
    intrinsics_frame_rows = _intrinsics_frame_metrics(
        predicted=geometry.intrinsics,
        target=ground_truth.intrinsics,
        source_sizes=geometry.source_sizes,
        processed_size=geometry.processed_size,
        image_mode=config.image_mode,
        frame_indices=config.frame_indices,
    )
    intrinsics_summary_rows = _summarize_intrinsics(intrinsics_frame_rows)

    pose_summary_rows = _with_scene(config.scene_id, pose_summary_rows)
    pose_frame_rows = _with_scene(config.scene_id, pose_frame_rows)
    pose_rpe_rows = _with_scene(config.scene_id, pose_rpe_rows)
    pose_pair_rows = _with_scene(config.scene_id, pose_pair_rows)
    pose_pair_summary_rows = _with_scene(
        config.scene_id,
        pose_pair_summary_rows,
    )
    rotation_quality_rows = _with_scene(
        config.scene_id,
        [
            *predicted.rotation_quality_rows,
            *target.rotation_quality_rows,
        ],
    )
    pointmap_frame_rows = _with_scene(
        config.scene_id,
        pointmap_frame_rows,
    )
    pointmap_summary_rows = _with_scene(
        config.scene_id,
        pointmap_summary_rows,
    )
    intrinsics_frame_rows = _with_scene(
        config.scene_id,
        intrinsics_frame_rows,
    )
    intrinsics_summary_rows = _with_scene(
        config.scene_id,
        intrinsics_summary_rows,
    )

    _write_csv(config.output_dir / "pose_summary.csv", pose_summary_rows)
    _write_csv(config.output_dir / "pose_frame_metrics.csv", pose_frame_rows)
    _write_csv(config.output_dir / "pose_rpe.csv", pose_rpe_rows)
    _write_csv(config.output_dir / "pose_pair_metrics.csv", pose_pair_rows)
    _write_csv(
        config.output_dir / "pose_pair_summary.csv",
        pose_pair_summary_rows,
    )
    _write_csv(
        config.output_dir / "pose_rotation_quality.csv",
        rotation_quality_rows,
    )
    _write_csv(
        config.output_dir / "pointmap_frame_metrics.csv",
        pointmap_frame_rows,
    )
    _write_csv(
        config.output_dir / "pointmap_summary.csv",
        pointmap_summary_rows,
    )
    _write_csv(
        config.output_dir / "intrinsics_frame_metrics.csv",
        intrinsics_frame_rows,
    )
    _write_csv(
        config.output_dir / "intrinsics_summary.csv",
        intrinsics_summary_rows,
    )

    metadata = {
        "experiment": "raw_streamvggt_pose_pointmap_diagnostics",
        "evaluation_only": True,
        "sam3_or_instance_masks_used": False,
        "scene_id": config.scene_id,
        "frame_indices": list(config.frame_indices),
        "reference_sequence_index": reference_sequence_index,
        "reference_frame_index": config.frame_indices[
            reference_sequence_index
        ],
        "pose_convention": {
            "predicted": "world_to_camera, X_cam = R @ X_world + t",
            "target": "COLMAP world_to_camera, X_cam = R @ X_world + t",
            "camera_center": "C_world = -R.T @ t",
        },
        "alignment_policy": {
            "primary": "reference_point_sim3",
            "reference_point_sim3": (
                "one fixed Sim(3) fitted from paired full-scene reference-frame "
                "pointmaps; held fixed for all poses and later pointmaps"
            ),
            "reference_pose_point_scale": (
                "reference camera center/orientation fitted exactly; scale copied "
                "from reference_point_sim3"
            ),
            "trajectory_sim3": (
                "all camera centers fitted to GT; optimistic gauge-only diagnostic"
            ),
        },
        "reference_point_sim3": {
            "scale": float(map_context.sim3_scale),
            "rotation": map_context.sim3_rotation.tolist(),
            "translation": map_context.sim3_translation.tolist(),
            "inliers": int(map_context.sim3_inliers),
            "rmse": float(map_context.sim3_rmse),
        },
        "processed_size": list(geometry.processed_size),
        "point_confidence_threshold": float(
            config.point_cloud_confidence_threshold
        ),
        "outputs": [
            "pose_summary.csv",
            "pose_frame_metrics.csv",
            "pose_rpe.csv",
            "pose_pair_metrics.csv",
            "pose_pair_summary.csv",
            "pose_rotation_quality.csv",
            "pointmap_frame_metrics.csv",
            "pointmap_summary.csv",
            "intrinsics_frame_metrics.csv",
            "intrinsics_summary.csv",
        ],
    }
    with (config.output_dir / "metadata.json").open(
        "w",
        encoding="utf8",
    ) as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    print(f"pose summary: {config.output_dir / 'pose_summary.csv'}")
    print(
        "pointmap summary: "
        f"{config.output_dir / 'pointmap_summary.csv'}"
    )
    print(
        "intrinsics summary: "
        f"{config.output_dir / 'intrinsics_summary.csv'}"
    )


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
        world_to_camera=torch.from_numpy(
            np.stack(world_to_camera)
        ).double(),
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
        translation = matrix[:3, 3]
        center = -(rotation.T @ translation)
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
    camera_to_world_rotation = torch.stack(rotations)
    camera_centers = torch.stack(centers)
    projected_world_to_camera = torch.eye(
        4,
        dtype=torch.float64,
    ).repeat(len(matrices), 1, 1)
    projected_world_to_camera[:, :3, :3] = (
        camera_to_world_rotation.transpose(1, 2)
    )
    projected_world_to_camera[:, :3, 3] = -torch.einsum(
        "tij,tj->ti",
        projected_world_to_camera[:, :3, :3],
        camera_centers,
    )
    return PoseSequence(
        world_to_camera=projected_world_to_camera,
        camera_to_world_rotation=camera_to_world_rotation,
        camera_centers=camera_centers,
        rotation_quality_rows=tuple(quality_rows),
    )


def _homogeneous(world_to_camera: torch.Tensor) -> torch.Tensor:
    if world_to_camera.ndim != 3:
        raise ValueError(
            "world_to_camera must have shape [T,3,4] or [T,4,4], "
            f"got {tuple(world_to_camera.shape)}."
        )
    if tuple(world_to_camera.shape[-2:]) == (4, 4):
        return world_to_camera
    if tuple(world_to_camera.shape[-2:]) != (3, 4):
        raise ValueError(
            "world_to_camera must have shape [T,3,4] or [T,4,4], "
            f"got {tuple(world_to_camera.shape)}."
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
    rotation = (
        target.camera_to_world_rotation[reference_index]
        @ predicted.camera_to_world_rotation[reference_index].T
    )
    rotation = _project_rotation(rotation)
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


def _trajectory_alignment(
    predicted: PoseSequence,
    target: PoseSequence,
) -> SimilarityAlignment:
    scale, rotation, translation = _umeyama(
        predicted.camera_centers,
        target.camera_centers,
    )
    return SimilarityAlignment(
        name="trajectory_sim3",
        scale=scale,
        rotation=rotation,
        translation=translation,
        fit_source="all selected GT camera centers; optimistic diagnostic",
    )


def _umeyama(
    source: torch.Tensor,
    target: torch.Tensor,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("Umeyama inputs must both have shape [N,3].")
    if source.shape[0] < 3:
        raise ValueError("Trajectory Sim(3) alignment needs at least 3 poses.")
    source_mean = source.mean(dim=0)
    target_mean = target.mean(dim=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / source.shape[0]
    left, singular_values, right_t = torch.linalg.svd(covariance)
    signs = torch.ones(3, dtype=source.dtype)
    if torch.det(left @ right_t) < 0:
        signs[-1] = -1
    rotation = left @ torch.diag(signs) @ right_t
    variance = source_centered.square().sum(dim=1).mean().clamp_min(1e-12)
    scale = float((singular_values * signs).sum() / variance)
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


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
        "rpe_translation_mean": float(rpe_translation.mean()),
        "rpe_rotation_mean_degrees": float(rpe_rotation.mean()),
        "rpe_rotation_max_degrees": float(rpe_rotation.max()),
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
    """Match StreamVGGT's official scale-invariant relative-pose protocol."""

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


def _intrinsics_frame_metrics(
    *,
    predicted: torch.Tensor,
    target: torch.Tensor,
    source_sizes: Sequence[tuple[int, int]],
    processed_size: tuple[int, int],
    image_mode: str,
    frame_indices: Sequence[int],
) -> list[dict]:
    predicted = predicted.double().cpu()
    target = target.double().cpu()
    if tuple(predicted.shape[-2:]) != (3, 3):
        raise ValueError(
            f"Predicted intrinsics must be [T,3,3], got {predicted.shape}."
        )
    rows = []
    for index, frame_index in enumerate(frame_indices):
        transformed_target = _transform_intrinsics(
            target[index],
            source_size=source_sizes[index],
            processed_size=processed_size,
            image_mode=image_mode,
        )
        row = {
            "sequence_index": index,
            "frame_index": int(frame_index),
            "predicted_fx": float(predicted[index, 0, 0]),
            "gt_processed_fx": float(transformed_target[0, 0]),
            "fx_relative_error": _relative_error(
                predicted[index, 0, 0],
                transformed_target[0, 0],
            ),
            "predicted_fy": float(predicted[index, 1, 1]),
            "gt_processed_fy": float(transformed_target[1, 1]),
            "fy_relative_error": _relative_error(
                predicted[index, 1, 1],
                transformed_target[1, 1],
            ),
            "predicted_cx": float(predicted[index, 0, 2]),
            "gt_processed_cx": float(transformed_target[0, 2]),
            "cx_absolute_error_pixels": float(
                torch.abs(predicted[index, 0, 2] - transformed_target[0, 2])
            ),
            "predicted_cy": float(predicted[index, 1, 2]),
            "gt_processed_cy": float(transformed_target[1, 2]),
            "cy_absolute_error_pixels": float(
                torch.abs(predicted[index, 1, 2] - transformed_target[1, 2])
            ),
        }
        rows.append(row)
    return rows


def _transform_intrinsics(
    intrinsics: torch.Tensor,
    *,
    source_size: tuple[int, int],
    processed_size: tuple[int, int],
    image_mode: str,
) -> torch.Tensor:
    transform = streamvggt_image_transform(
        source_size,
        mode=image_mode,
    )
    if tuple(transform.target_size) != tuple(processed_size):
        raise RuntimeError(
            "StreamVGGT preprocessing transform disagrees with extracted image "
            f"size: {transform.target_size} vs {processed_size}."
        )
    scale_x, scale_y = transform.scale_xy
    offset_x, offset_y = transform.offset_xy
    result = intrinsics.clone().double()
    result[0, 0] *= scale_x
    result[1, 1] *= scale_y
    result[0, 2] = (
        (intrinsics[0, 2] + 0.5) * scale_x - 0.5 + offset_x
    )
    result[1, 2] = (
        (intrinsics[1, 2] + 0.5) * scale_y - 0.5 + offset_y
    )
    return result


def _summarize_intrinsics(rows: Sequence[dict]) -> list[dict]:
    return [
        {
            "frames": len(rows),
            "mean_fx_relative_error": _mean_rows(
                rows,
                "fx_relative_error",
            ),
            "max_fx_relative_error": _max_rows(
                rows,
                "fx_relative_error",
            ),
            "mean_fy_relative_error": _mean_rows(
                rows,
                "fy_relative_error",
            ),
            "max_fy_relative_error": _max_rows(
                rows,
                "fy_relative_error",
            ),
            "mean_cx_absolute_error_pixels": _mean_rows(
                rows,
                "cx_absolute_error_pixels",
            ),
            "mean_cy_absolute_error_pixels": _mean_rows(
                rows,
                "cy_absolute_error_pixels",
            ),
        }
    ]


def _relative_error(value: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.abs(value - target) / torch.abs(target).clamp_min(1e-12))


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


def _with_scene(scene_id: str, rows: Sequence[dict]) -> list[dict]:
    return [{"scene_id": str(scene_id), **row} for row in rows]


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="streaming_couping/configs/recovery_050_025.yaml",
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--scene-id")
    parser.add_argument("--frame-indices", type=int, nargs="+")
    parser.add_argument(
        "--reference-sequence-index",
        type=int,
        default=0,
    )
    parser.add_argument("--geometry-device")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
