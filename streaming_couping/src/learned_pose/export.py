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
    all_camera_comparison_rows: list[dict] = []
    all_point_comparison_rows: list[dict] = []

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
        target_points = _world_points(payload["target_world_points"])
        if tuple(target_points.shape) != tuple(points.shape):
            raise ValueError(
                f"GT/predicted pointmap mismatch for {clip.name}: "
                f"{tuple(target_points.shape)} versus {tuple(points.shape)}."
            )
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
        _, _, target_intrinsics = _decode_pose(
            payload["target_pose_encoding"],
            image_size=processed_size,
            frame_indices=clip.frame_indices,
        )
        target_c2w, target_w2c = _camera_matrices_from_world_to_camera(
            payload["target_world_to_camera"],
            frame_indices=clip.frame_indices,
        )
        all_camera_comparison_rows.extend(
            _camera_comparison_rows(
                clip_name=clip.name,
                scene_id=clip.scene_id,
                variant=selected,
                frame_indices=clip.frame_indices,
                predicted_c2w=metric_c2w,
                target_c2w=target_c2w,
            )
        )
        native_pose_rows = _pose_rows(
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
        all_pose_rows.extend(native_pose_rows)
        all_pose_rows.extend(
            _pose_rows(
                clip_name=clip.name,
                scene_id=clip.scene_id,
                variant="scannetpp_ground_truth",
                frame_indices=clip.frame_indices,
                reference_sequence_index=clip.reference_sequence_index,
                coordinate_system="scannetpp_gt_world",
                c2w=target_c2w,
                w2c=target_w2c,
                intrinsics=target_intrinsics,
            )
        )
        all_pose_rows.extend(
            _pose_rows(
                clip_name=clip.name,
                scene_id=clip.scene_id,
                variant=selected,
                frame_indices=clip.frame_indices,
                reference_sequence_index=clip.reference_sequence_index,
                coordinate_system=(
                    "scannetpp_gt_world_via_fixed_reference_point_sim3"
                ),
                c2w=metric_c2w,
                w2c=metric_w2c,
                intrinsics=intrinsics,
            )
        )

        clip_root = root / clip.name
        deployable_root = clip_root / "deployable_native"
        deployable_root.mkdir(parents=True, exist_ok=True)
        _write_csv(deployable_root / "camera_poses.csv", native_pose_rows)
        np.savez_compressed(
            deployable_root / "camera_poses.npz",
            frame_indices=np.asarray(clip.frame_indices, dtype=np.int64),
            intrinsics=intrinsics.numpy(),
            c2w=native_c2w.numpy(),
            w2c=native_w2c.numpy(),
        )
        cloud_root = clip_root / "pointclouds"
        cloud_root.mkdir(parents=True, exist_ok=True)
        _remove_legacy_pointclouds(
            cloud_root,
            instance_ids=clip.instance_ids,
        )
        np.savez_compressed(
            clip_root / "camera_poses.npz",
            frame_indices=np.asarray(clip.frame_indices, dtype=np.int64),
            intrinsics=intrinsics.numpy(),
            c2w_native=native_c2w.numpy(),
            w2c_native=native_w2c.numpy(),
            c2w_metric_evaluation_only=metric_c2w.numpy(),
            w2c_metric_evaluation_only=metric_w2c.numpy(),
            gt_intrinsics=target_intrinsics.numpy(),
            gt_c2w=target_c2w.numpy(),
            gt_w2c=target_w2c.numpy(),
            point_alignment_scale=np.asarray(scale, dtype=np.float64),
            point_alignment_rotation=rotation.numpy(),
            point_alignment_translation=translation.numpy(),
        )

        valid = (
            torch.isfinite(points).all(dim=-1)
            & torch.isfinite(confidence)
            & (confidence >= float(ray_config.export_confidence_threshold))
        )
        finite_target = torch.isfinite(target_points).all(dim=-1)
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
            native_path = cloud_root / f"{name}_predicted_native.ply"
            metric_path = cloud_root / f"{name}_predicted_metric_gt_world.ply"
            metric_points = scale * (selected_points @ rotation.T.float()) + (
                translation.float()
            )
            _write_binary_ply(native_path, selected_points, selected_colors)
            _write_binary_ply(
                deployable_root / f"{name}.ply",
                selected_points,
                selected_colors,
            )
            _write_binary_ply(metric_path, metric_points, selected_colors)
            for coordinate_system, path, evaluation_mask in (
                (
                    "streamvggt_point_head_native",
                    native_path,
                    "all_finite_confident_prediction",
                ),
                (
                    "scannetpp_gt_world_via_fixed_reference_point_sim3",
                    metric_path,
                    "all_finite_confident_prediction",
                ),
            ):
                all_cloud_rows.append(
                    {
                        "clip": clip.name,
                        "scene_id": clip.scene_id,
                        "variant": selected,
                        "coordinate_system": coordinate_system,
                        "artifact_role": "prediction",
                        "evaluation_mask": evaluation_mask,
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

            scope_mask = (
                torch.ones_like(finite_target)
                if instance_id is None
                else masks[:, list(clip.instance_ids).index(int(instance_id))]
            )
            target_all_selection = finite_target & scope_mask
            target_all_points = (
                target_points[target_all_selection].detach().float().cpu()
            )
            target_all_colors = (
                colors[target_all_selection].detach().to(torch.uint8).cpu()
            )
            target_all_before_limit = int(target_all_points.shape[0])
            target_all_points, target_all_colors = _limit_points(
                target_all_points,
                target_all_colors,
                max_points=max_points,
            )
            target_all_path = (
                cloud_root / f"{name}_gt_visible_all_finite_metric_gt_world.ply"
            )
            _write_binary_ply(
                target_all_path,
                target_all_points,
                target_all_colors,
            )
            all_cloud_rows.append(
                {
                    "clip": clip.name,
                    "scene_id": clip.scene_id,
                    "variant": selected,
                    "coordinate_system": "scannetpp_gt_world",
                    "artifact_role": "ground_truth_visible_all_finite",
                    "evaluation_mask": "all_finite_gt_visible_in_selected_frames",
                    "spatial_scope": scope,
                    "instance_id": "" if instance_id is None else instance_id,
                    "confidence_threshold": "",
                    "selected_points_before_limit": target_all_before_limit,
                    "exported_points": int(target_all_points.shape[0]),
                    "ply_path": str(target_all_path),
                }
            )

            paired_selection = selection & finite_target
            paired_prediction = points[paired_selection].detach().float().cpu()
            paired_target = target_points[paired_selection].detach().float().cpu()
            paired_colors = colors[paired_selection].detach().to(torch.uint8).cpu()
            paired_before_limit = int(paired_prediction.shape[0])
            paired_prediction_metric_all = scale * (
                paired_prediction @ rotation.T.float()
            ) + translation.float()
            all_point_comparison_rows.append(
                {
                    "clip": clip.name,
                    "scene_id": clip.scene_id,
                    "variant": selected,
                    "alignment": "fixed_reference_point_sim3",
                    "coordinate_system": "scannetpp_gt_world",
                    "spatial_scope": scope,
                    "instance_id": "" if instance_id is None else instance_id,
                    "paired_points": paired_before_limit,
                    **_paired_distance_statistics(
                        paired_prediction_metric_all,
                        paired_target,
                    ),
                }
            )
            paired_prediction, paired_target, paired_colors = _limit_paired_points(
                paired_prediction,
                paired_target,
                paired_colors,
                max_points=max_points,
            )
            paired_prediction_metric = scale * (
                paired_prediction @ rotation.T.float()
            ) + translation.float()
            paired_prediction_path = (
                cloud_root / f"{name}_predicted_paired_metric_gt_world.ply"
            )
            target_path = cloud_root / f"{name}_gt_paired_metric_gt_world.ply"
            overlay_path = cloud_root / f"{name}_overlay_metric_gt_world.ply"
            _write_binary_ply(
                paired_prediction_path,
                paired_prediction_metric,
                paired_colors,
            )
            _write_binary_ply(target_path, paired_target, paired_colors)
            overlay_points = torch.cat(
                [paired_prediction_metric, paired_target],
                dim=0,
            )
            overlay_colors = torch.cat(
                [
                    _solid_colors(len(paired_prediction_metric), (255, 64, 64)),
                    _solid_colors(len(paired_target), (64, 255, 255)),
                ],
                dim=0,
            )
            _write_binary_ply(overlay_path, overlay_points, overlay_colors)
            for artifact_role, path, exported_points in (
                ("prediction_paired", paired_prediction_path, len(paired_prediction)),
                ("ground_truth_paired", target_path, len(paired_target)),
                ("overlay_pred_red_gt_cyan", overlay_path, len(overlay_points)),
            ):
                all_cloud_rows.append(
                    {
                        "clip": clip.name,
                        "scene_id": clip.scene_id,
                        "variant": selected,
                        "coordinate_system": "scannetpp_gt_world",
                        "artifact_role": artifact_role,
                        "evaluation_mask": "paired_finite_gt_and_confident_prediction",
                        "spatial_scope": scope,
                        "instance_id": "" if instance_id is None else instance_id,
                        "confidence_threshold": float(
                            ray_config.export_confidence_threshold
                        ),
                        "selected_points_before_limit": paired_before_limit,
                        "exported_points": int(exported_points),
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

        camera_overlay_path = clip_root / "camera_centers_overlay_metric_gt_world.ply"
        predicted_centers = metric_c2w[:, :3, 3].detach().float().cpu()
        target_centers = target_c2w[:, :3, 3].detach().float().cpu()
        _write_binary_ply(
            camera_overlay_path,
            torch.cat([predicted_centers, target_centers], dim=0),
            torch.cat(
                [
                    _solid_colors(len(predicted_centers), (255, 64, 64)),
                    _solid_colors(len(target_centers), (64, 255, 255)),
                ],
                dim=0,
            ),
        )
        (deployable_root / "README.txt").write_text(
            "Final deployable V3 reconstruction\n"
            "==================================\n\n"
            "full_scene.ply and instance_*.ply are the V2 refined world "
            "pointmaps. camera_poses.csv/.npz contain the selected V3 camera "
            "poses recovered directly from those same pointmaps. All files "
            "use one internally consistent StreamVGGT native gauge. No GT "
            "alignment or GT value is used in this directory.\n",
            encoding="utf8",
        )

    _write_csv(root / "camera_poses.csv", all_pose_rows)
    _write_csv(
        root / "camera_comparison_pointmap_sim3.csv",
        all_camera_comparison_rows,
    )
    _write_csv(root / "pointcloud_summary.csv", all_cloud_rows)
    _write_csv(root / "pointcloud_frame_selection.csv", all_frame_rows)
    _write_csv(root / "pointcloud_gt_comparison.csv", all_point_comparison_rows)
    (root / "HOW_TO_COMPARE.txt").write_text(
        "Final V3 export comparison\n"
        "==========================\n\n"
        "Do not overlay *_predicted_native.ply with GT: native is the arbitrary "
        "StreamVGGT gauge.\n\n"
        "Use *_overlay_metric_gt_world.ply for visual comparison. Prediction is "
        "red and ScanNet++ GT is cyan; both use exactly the same paired pixels "
        "and the same GT-world coordinate system.\n\n"
        "Use *_predicted_metric_gt_world.ply with "
        "*_gt_visible_all_finite_metric_gt_world.ply to compare the complete "
        "visible prediction and GT pointmaps.\n\n"
        "camera_poses.csv contains native prediction, GT-world prediction, and "
        "raw ScanNet++ GT poses. camera_comparison_pointmap_sim3.csv compares "
        "the latter two in the joint pointmap coordinate system.\n",
        encoding="utf8",
    )
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
                    "Prediction is transformed into the ScanNet++ GT world by "
                    "a fixed reference-frame pointmap Sim(3) fitted using GT "
                    "point correspondences. GT and overlay PLY files are in "
                    "this same coordinate system."
                ),
                "alignment_warning": (
                    "The joint pose/pointcloud comparison uses the cached fixed "
                    "reference-point Sim(3). The previously reported ATE uses "
                    "reference-pose orientation/translation with only the point "
                    "Sim(3) scale, so its numbers are not recomputed by "
                    "camera_comparison_pointmap_sim3.csv."
                ),
                "overlay_colors": "prediction=red, ground_truth=cyan",
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


def _camera_matrices_from_world_to_camera(
    value: torch.Tensor,
    *,
    frame_indices: Iterable[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    world_to_camera = value.detach().double().cpu()
    if world_to_camera.ndim == 4 and world_to_camera.shape[0] == 1:
        world_to_camera = world_to_camera[0]
    sequence = _prepare_pose_sequence(
        world_to_camera,
        frame_indices=tuple(int(item) for item in frame_indices),
        source="scannetpp_ground_truth",
    )
    c2w = torch.eye(4, dtype=torch.float64).repeat(len(sequence.camera_centers), 1, 1)
    c2w[:, :3, :3] = sequence.camera_to_world_rotation
    c2w[:, :3, 3] = sequence.camera_centers
    return c2w, torch.linalg.inv(c2w)


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


def _camera_comparison_rows(
    *,
    clip_name: str,
    scene_id: str,
    variant: str,
    frame_indices: Iterable[int],
    predicted_c2w: torch.Tensor,
    target_c2w: torch.Tensor,
) -> list[dict]:
    rows = []
    for sequence_index, frame_index in enumerate(frame_indices):
        predicted = predicted_c2w[sequence_index]
        target = target_c2w[sequence_index]
        center_error = torch.linalg.vector_norm(
            predicted[:3, 3] - target[:3, 3]
        )
        rotation_error = _rotation_error_degrees(
            predicted[:3, :3],
            target[:3, :3],
        )
        rows.append(
            {
                "clip": clip_name,
                "scene_id": scene_id,
                "variant": variant,
                "alignment": "fixed_reference_point_sim3",
                "coordinate_system": "scannetpp_gt_world",
                "sequence_index": sequence_index,
                "frame_index": int(frame_index),
                "camera_center_error_meters": float(center_error),
                "rotation_error_degrees": rotation_error,
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


def _limit_paired_points(
    predicted: torch.Tensor,
    target: torch.Tensor,
    colors: torch.Tensor,
    *,
    max_points: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if predicted.shape != target.shape or predicted.shape != colors.shape:
        raise ValueError(
            "Paired prediction, GT, and RGB arrays must all have shape [N,3]."
        )
    if predicted.shape[0] <= int(max_points):
        return predicted, target, colors
    indices = torch.linspace(
        0,
        predicted.shape[0] - 1,
        steps=int(max_points),
    ).long()
    return (
        predicted.index_select(0, indices),
        target.index_select(0, indices),
        colors.index_select(0, indices),
    )


def _solid_colors(count: int, rgb: tuple[int, int, int]) -> torch.Tensor:
    return torch.tensor(rgb, dtype=torch.uint8)[None].repeat(int(count), 1)


def _remove_legacy_pointclouds(
    cloud_root: Path,
    *,
    instance_ids: Iterable[int],
) -> None:
    names = ["full_scene", *(f"instance_{int(value)}" for value in instance_ids)]
    for name in names:
        for suffix in ("native", "metric_evaluation_only"):
            path = cloud_root / f"{name}_{suffix}.ply"
            if path.exists():
                path.unlink()


def _paired_distance_statistics(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    if predicted.shape != target.shape:
        raise ValueError("Predicted and GT paired point arrays must have equal shape.")
    if not predicted.numel():
        return {
            "paired_distance_mean": float("nan"),
            "paired_distance_median": float("nan"),
            "paired_distance_rmse": float("nan"),
            "paired_distance_p90": float("nan"),
        }
    distances = torch.linalg.vector_norm(predicted - target, dim=-1)
    return {
        "paired_distance_mean": float(distances.mean()),
        "paired_distance_median": float(distances.median()),
        "paired_distance_rmse": float(torch.sqrt(distances.square().mean())),
        "paired_distance_p90": float(torch.quantile(distances, 0.90)),
    }


def _rotation_error_degrees(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> float:
    relative = target.T @ predicted
    cosine = ((torch.trace(relative) - 1.0) * 0.5).clamp(-1.0, 1.0)
    return float(torch.rad2deg(torch.acos(cosine)))


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
