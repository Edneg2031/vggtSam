"""Export the selected instance-ray pose and its consistent point clouds."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from vggtsam.utils.imports import maybe_add_repo_to_path

from ..config import load_config
from ..instance_point_cloud import (
    _limit_points,
    _write_binary_ply,
    load_processed_colors,
)
from ..pose_evaluation import _prepare_pose_sequence
from .cache import cache_path, load_feature_cache
from .config import LearnedPoseConfig


@torch.no_grad()
def export_final_ray_pose_outputs(
    config: LearnedPoseConfig,
    *,
    variant: str | None = None,
    output_dir: str | Path | None = None,
) -> Path:
    """Export native/development-metric poses and colored PLY files.

    Native coordinates are the deployable StreamVGGT point-head gauge.  The
    second coordinate system applies the cache's fixed reference-point Sim(3)
    and is strictly an evaluation convenience because that Sim(3) used GT
    point correspondences when the frozen cache was built.
    """

    ray_config = config.evaluation.ray_pose
    selected = str(variant or ray_config.final_variant)
    if selected not in ray_config.variants:
        raise ValueError(
            f"Export variant {selected!r} is not present in ray_pose.variants."
        )
    predictions_path = config.output_dir / "evaluation" / "ray_pose_predictions.pt"
    artifact = _torch_load(predictions_path)
    predictions = artifact.get("predictions")
    if not isinstance(predictions, dict):
        raise ValueError(f"Missing predictions mapping in {predictions_path}.")

    root = Path(output_dir) if output_dir is not None else (
        config.output_dir / "final_instance_ray_pose_v3"
    )
    root.mkdir(parents=True, exist_ok=True)
    recovery = load_config(config.recovery_config)
    # The training/evaluation paths add the external repository while loading
    # the StreamVGGT model.  Export intentionally loads no model, so register
    # the repository explicitly before importing its lightweight pose codec.
    maybe_add_repo_to_path(recovery.streamvggt_repo)
    all_pose_rows: list[dict] = []
    all_cloud_rows: list[dict] = []
    all_frame_rows: list[dict] = []

    for clip in config.clips:
        clip_prediction = predictions.get(clip.name)
        if not isinstance(clip_prediction, dict):
            raise ValueError(
                f"No saved ray-pose prediction for clip {clip.name!r}."
            )
        pose_encodings = clip_prediction.get("pose_encodings")
        if not isinstance(pose_encodings, dict) or selected not in pose_encodings:
            raise ValueError(
                f"Variant {selected!r} is missing for clip {clip.name!r}."
            )
        payload = load_feature_cache(cache_path(config, clip))
        _check_clip_payload(clip_prediction, payload, clip_name=clip.name)

        points = _world_points(clip_prediction["refined_world_points"])
        confidence = _world_confidence(
            clip_prediction["refined_world_confidence"],
            points,
        )
        masks = _tracking_masks(
            clip_prediction["tracking_masks_stream"],
            points,
            expected_instances=len(clip.instance_ids),
        )
        processed_size = (int(points.shape[1]), int(points.shape[2]))
        colors = load_processed_colors(
            clip_prediction["image_paths"],
            processed_size=processed_size,
            image_mode=recovery.image_mode,
        )
        if tuple(colors.shape[:3]) != tuple(points.shape[:3]):
            raise ValueError(
                f"RGB/pointmap shape mismatch for {clip.name}: "
                f"{tuple(colors.shape)} versus {tuple(points.shape)}."
            )

        native_c2w, native_w2c, intrinsics = _decode_pose(
            pose_encodings[selected],
            image_size=processed_size,
            frame_indices=clip.frame_indices,
        )
        scale = float(payload["point_alignment_scale"])
        rotation = payload["point_alignment_rotation"].detach().double().cpu()
        translation = payload["point_alignment_translation"].detach().double().cpu()
        metric_c2w, metric_w2c = _align_camera_pose(
            native_c2w,
            scale=scale,
            rotation=rotation,
            translation=translation,
        )
        all_pose_rows.extend(
            _pose_rows(
                clip_name=clip.name,
                scene_id=clip.scene_id,
                variant=selected,
                frame_indices=clip.frame_indices,
                reference_sequence_index=clip.reference_sequence_index,
                coordinate_system="streamvggt_point_head_native",
                c2w=native_c2w,
                w2c=native_w2c,
                intrinsics=intrinsics,
            )
        )
        all_pose_rows.extend(
            _pose_rows(
                clip_name=clip.name,
                scene_id=clip.scene_id,
                variant=selected,
                frame_indices=clip.frame_indices,
                reference_sequence_index=clip.reference_sequence_index,
                coordinate_system="fixed_reference_point_sim3_metric_evaluation_only",
                c2w=metric_c2w,
                w2c=metric_w2c,
                intrinsics=intrinsics,
            )
        )

        clip_root = root / clip.name
        cloud_root = clip_root / "pointclouds"
        cloud_root.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            clip_root / "camera_poses.npz",
            frame_indices=np.asarray(clip.frame_indices, dtype=np.int64),
            intrinsics=intrinsics.numpy(),
            c2w_native=native_c2w.numpy(),
            w2c_native=native_w2c.numpy(),
            c2w_metric_evaluation_only=metric_c2w.numpy(),
            w2c_metric_evaluation_only=metric_w2c.numpy(),
            point_alignment_scale=np.asarray(scale, dtype=np.float64),
            point_alignment_rotation=rotation.numpy(),
            point_alignment_translation=translation.numpy(),
        )

        valid = (
            torch.isfinite(points).all(dim=-1)
            & torch.isfinite(confidence)
            & (confidence >= float(ray_config.export_confidence_threshold))
        )
        scopes: list[tuple[str, int | None, torch.Tensor, int]] = [
            (
                "full_scene",
                None,
                valid,
                int(ray_config.export_max_full_scene_points),
            )
        ]
        scopes.extend(
            (
                "tracked_instance",
                int(instance_id),
                valid & masks[:, instance_index],
                int(ray_config.export_max_instance_points),
            )
            for instance_index, instance_id in enumerate(clip.instance_ids)
        )
        for scope, instance_id, selection, max_points in scopes:
            name = "full_scene" if instance_id is None else f"instance_{instance_id}"
            selected_points = points[selection].detach().float().cpu()
            selected_colors = colors[selection].detach().to(torch.uint8).cpu()
            before_limit = int(selected_points.shape[0])
            selected_points, selected_colors = _limit_points(
                selected_points,
                selected_colors,
                max_points=max_points,
            )
            native_path = cloud_root / f"{name}_native.ply"
            metric_path = cloud_root / f"{name}_metric_evaluation_only.ply"
            metric_points = scale * (selected_points @ rotation.T.float()) + (
                translation.float()
            )
            _write_binary_ply(native_path, selected_points, selected_colors)
            _write_binary_ply(metric_path, metric_points, selected_colors)
            for coordinate_system, path in (
                ("streamvggt_point_head_native", native_path),
                (
                    "fixed_reference_point_sim3_metric_evaluation_only",
                    metric_path,
                ),
            ):
                all_cloud_rows.append(
                    {
                        "clip": clip.name,
                        "scene_id": clip.scene_id,
                        "variant": selected,
                        "coordinate_system": coordinate_system,
                        "spatial_scope": scope,
                        "instance_id": "" if instance_id is None else instance_id,
                        "confidence_threshold": float(
                            ray_config.export_confidence_threshold
                        ),
                        "selected_points_before_limit": before_limit,
                        "exported_points": int(selected_points.shape[0]),
                        "ply_path": str(path),
                    }
                )
            all_frame_rows.extend(
                _selection_frame_rows(
                    clip_name=clip.name,
                    scene_id=clip.scene_id,
                    variant=selected,
                    frame_indices=clip.frame_indices,
                    scope=scope,
                    instance_id=instance_id,
                    selection=selection,
                    confidence=confidence,
                )
            )

    _write_csv(root / "camera_poses.csv", all_pose_rows)
    _write_csv(root / "pointcloud_summary.csv", all_cloud_rows)
    _write_csv(root / "pointcloud_frame_selection.csv", all_frame_rows)
    with (root / "metadata.json").open("w", encoding="utf8") as handle:
        json.dump(
            {
                "method": "instance-ray pose V3",
                "selected_variant": selected,
                "pose_source": "decoupled V2 learned rotation",
                "geometry_source": "decoupled V2 refined world pointmap",
                "translation_solver": (
                    "angular-Huber point-to-ray center fit restricted to "
                    "persistent tracked-instance masks"
                ),
                "intrinsics_source": "baseline reference-frame K",
                "native_coordinate_system": (
                    "Deployable StreamVGGT point-head gauge; scale is arbitrary."
                ),
                "metric_coordinate_system": (
                    "Evaluation only: fixed reference-frame pointmap Sim(3) "
                    "was fitted using GT point correspondences."
                ),
                "source_predictions": str(predictions_path),
                "config": str(config.source_path),
            },
            handle,
            indent=2,
        )
    return root


def _decode_pose(
    pose_encoding: torch.Tensor,
    *,
    image_size: tuple[int, int],
    frame_indices: Iterable[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri

    encoding = pose_encoding.detach().float().cpu()
    if encoding.ndim == 2:
        encoding = encoding[None]
    w2c, intrinsics = pose_encoding_to_extri_intri(
        encoding,
        image_size_hw=image_size,
    )
    sequence = _prepare_pose_sequence(
        w2c[0].detach().double().cpu(),
        frame_indices=tuple(int(value) for value in frame_indices),
        source="selected_instance_ray_pose_v3",
    )
    c2w = torch.eye(4, dtype=torch.float64).repeat(len(sequence.camera_centers), 1, 1)
    c2w[:, :3, :3] = sequence.camera_to_world_rotation
    c2w[:, :3, 3] = sequence.camera_centers
    return c2w, torch.linalg.inv(c2w), intrinsics[0].detach().double().cpu()


def _align_camera_pose(
    native_c2w: torch.Tensor,
    *,
    scale: float,
    rotation: torch.Tensor,
    translation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = torch.eye(4, dtype=torch.float64).repeat(native_c2w.shape[0], 1, 1)
    output[:, :3, :3] = torch.einsum(
        "ij,sjk->sik",
        rotation,
        native_c2w[:, :3, :3],
    )
    output[:, :3, 3] = float(scale) * (
        native_c2w[:, :3, 3] @ rotation.T
    ) + translation
    return output, torch.linalg.inv(output)


def _pose_rows(
    *,
    clip_name: str,
    scene_id: str,
    variant: str,
    frame_indices: Iterable[int],
    reference_sequence_index: int,
    coordinate_system: str,
    c2w: torch.Tensor,
    w2c: torch.Tensor,
    intrinsics: torch.Tensor,
) -> list[dict]:
    rows = []
    for sequence_index, frame_index in enumerate(frame_indices):
        center = c2w[sequence_index, :3, 3]
        rows.append(
            {
                "clip": clip_name,
                "scene_id": scene_id,
                "variant": variant,
                "coordinate_system": coordinate_system,
                "sequence_index": sequence_index,
                "frame_index": int(frame_index),
                "is_reference": int(sequence_index == int(reference_sequence_index)),
                "fx": float(intrinsics[sequence_index, 0, 0]),
                "fy": float(intrinsics[sequence_index, 1, 1]),
                "cx": float(intrinsics[sequence_index, 0, 2]),
                "cy": float(intrinsics[sequence_index, 1, 2]),
                "camera_center_x": float(center[0]),
                "camera_center_y": float(center[1]),
                "camera_center_z": float(center[2]),
                "camera_to_world_4x4": _flatten_matrix(c2w[sequence_index]),
                "world_to_camera_4x4": _flatten_matrix(w2c[sequence_index]),
            }
        )
    return rows


def _selection_frame_rows(
    *,
    clip_name: str,
    scene_id: str,
    variant: str,
    frame_indices: Iterable[int],
    scope: str,
    instance_id: int | None,
    selection: torch.Tensor,
    confidence: torch.Tensor,
) -> list[dict]:
    rows = []
    for sequence_index, frame_index in enumerate(frame_indices):
        current = selection[sequence_index]
        values = confidence[sequence_index][current]
        rows.append(
            {
                "clip": clip_name,
                "scene_id": scene_id,
                "variant": variant,
                "sequence_index": sequence_index,
                "frame_index": int(frame_index),
                "spatial_scope": scope,
                "instance_id": "" if instance_id is None else instance_id,
                "selected_points": int(current.sum()),
                "mean_selected_confidence": (
                    float(values.mean()) if values.numel() else float("nan")
                ),
            }
        )
    return rows


def _world_points(value: torch.Tensor) -> torch.Tensor:
    points = value.detach().float().cpu()
    if points.ndim == 5 and points.shape[0] == 1:
        points = points[0]
    if points.ndim != 4 or points.shape[-1] != 3:
        raise ValueError(f"Expected world points [S,H,W,3], got {tuple(points.shape)}.")
    return points


def _world_confidence(value: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    confidence = value.detach().float().cpu()
    if tuple(confidence.shape) == (1, *points.shape[:3]):
        confidence = confidence[0]
    elif tuple(confidence.shape) == (*points.shape[:3], 1):
        confidence = confidence[..., 0]
    elif tuple(confidence.shape) == (1, *points.shape[:3], 1):
        confidence = confidence[0, ..., 0]
    if tuple(confidence.shape) != tuple(points.shape[:3]):
        raise ValueError(
            f"Confidence/pointmap mismatch: {tuple(confidence.shape)} versus "
            f"{tuple(points.shape[:3])}."
        )
    return confidence


def _tracking_masks(
    value: torch.Tensor,
    points: torch.Tensor,
    *,
    expected_instances: int,
) -> torch.Tensor:
    masks = value.detach().bool().cpu()
    if masks.ndim == 5 and masks.shape[0] == 1:
        masks = masks[0]
    expected = (points.shape[0], expected_instances, points.shape[1], points.shape[2])
    if tuple(masks.shape) != expected:
        raise ValueError(
            f"Tracking-mask/pointmap mismatch: {tuple(masks.shape)} versus {expected}."
        )
    return masks


def _check_clip_payload(prediction: dict, payload: dict, *, clip_name: str) -> None:
    for field in ("frame_indices", "instance_ids", "image_paths"):
        if list(prediction.get(field, [])) != list(payload.get(field, [])):
            raise ValueError(f"Prediction/cache {field} mismatch for {clip_name}.")


def _flatten_matrix(value: torch.Tensor) -> str:
    return " ".join(f"{float(item):.10g}" for item in value.reshape(-1))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty export table: {path}.")
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _torch_load(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run run_instance_token_pose --stage ray first."
        )
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise ValueError(f"Unsupported ray-pose prediction artifact: {path}.")
    return value
