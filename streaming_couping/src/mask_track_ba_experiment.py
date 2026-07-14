"""Oracle mask-gated StreamVGGT track-head pose refinement experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from test_sam.data import load_mask_tracking_sequence

from .backbones.streamvggt_wrapper import (
    StreamVGGTWrapper,
    unproject_depth_to_world,
)
from .config import load_config
from .geometry.export import save_aggregate_ply, save_pointmap_ply
from .geometry.gt_data import load_gt_geometry_sequence
from .geometry.registration import (
    align_world_to_camera,
    apply_similarity,
    estimate_similarity,
    pose_errors,
    rotation_angle_degrees,
    symmetric_chamfer,
)
from .geometry.reprojection_ba import (
    identity_ba_result,
    points_inside_mask,
    refine_pose_with_tracks,
    sample_reference_query_points,
)
from .gt_mask_pose_experiment import pointmap_errors, select_causal_reference


def main() -> None:
    args = _parse_args()
    overrides = {
        key: value
        for key, value in {
            "manifest": args.manifest,
            "scene_id": args.scene_id,
            "instance_id": args.instance_id,
            "frame_indices": args.frame_indices,
            "geometry_device": args.geometry_device,
            "output_dir": args.output_dir,
        }.items()
        if value is not None
    }
    config = load_config(args.config, overrides)
    run_experiment(
        config,
        reference_sequence_index=args.reference_sequence_index,
        confidence_threshold=args.confidence_threshold,
        alignment_trim_fraction=args.alignment_trim_fraction,
        max_query_points=args.max_query_points,
        query_erosion_radius=args.query_erosion_radius,
        track_iterations=args.track_iterations,
        track_visibility_threshold=args.track_visibility_threshold,
        track_confidence_threshold=args.track_confidence_threshold,
        track_score_mode=args.track_score_mode,
        ba_mode=args.ba_mode,
        ba_iterations=args.ba_iterations,
        ba_learning_rate=args.ba_learning_rate,
        ba_robust_delta_pixels=args.ba_robust_delta_pixels,
        ba_pose_prior_weight=args.ba_pose_prior_weight,
        ba_min_tracks=args.ba_min_tracks,
        ba_max_rotation_degrees=args.ba_max_rotation_degrees,
        ba_max_translation_depth_ratio=args.ba_max_translation_depth_ratio,
    )


def run_experiment(
    config,
    *,
    reference_sequence_index: int | None,
    confidence_threshold: float,
    alignment_trim_fraction: float,
    max_query_points: int,
    query_erosion_radius: int,
    track_iterations: int,
    track_visibility_threshold: float,
    track_confidence_threshold: float,
    track_score_mode: str,
    ba_mode: str,
    ba_iterations: int,
    ba_learning_rate: float,
    ba_robust_delta_pixels: float,
    ba_pose_prior_weight: float,
    ba_min_tracks: int,
    ba_max_rotation_degrees: float,
    ba_max_translation_depth_ratio: float,
) -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    sequence = load_mask_tracking_sequence(
        config.manifest,
        scene_id=config.scene_id,
        frame_indices=config.frame_indices,
        sequence_length=len(config.frame_indices),
        frame_stride=1,
        window_index=0,
        instance_id=config.instance_id,
        min_pixels=config.min_pixels,
        max_area_ratio=config.max_area_ratio,
        min_visible_frames=1,
        excluded_labels=config.excluded_labels,
        seed=0,
    )
    reference = select_causal_reference(sequence, reference_sequence_index)
    print(
        f"target scene={sequence.scene_id} frames={sequence.frame_indices} "
        f"instance={sequence.instance_id} label={sequence.label!r} "
        f"reference={reference} ba_mode={ba_mode}"
    )
    print("running frozen StreamVGGT geometry with causal caches...")
    streamvggt = StreamVGGTWrapper(
        repo_path=config.streamvggt_repo,
        checkpoint_path=config.streamvggt_checkpoint,
        device=config.geometry_device,
        image_mode=config.image_mode,
        streaming_cache=config.streaming_cache,
    ).load()
    geometry = streamvggt.extract(sequence.image_paths)
    if (
        geometry.camera_world_points is None
        or geometry.depth is None
        or geometry.depth_confidence is None
    ):
        raise RuntimeError("StreamVGGT depth/camera geometry is unavailable.")
    gt = load_gt_geometry_sequence(
        config.manifest,
        scene_id=sequence.scene_id,
        frame_indices=sequence.frame_indices,
        instance_id=sequence.instance_id,
        processed_size=geometry.processed_size,
        image_mode=config.image_mode,
    )
    if tuple(gt.pointmaps.shape) != tuple(geometry.camera_world_points.shape):
        raise RuntimeError(
            "GT and depth-camera pointmap shapes disagree: "
            f"{tuple(gt.pointmaps.shape)} != "
            f"{tuple(geometry.camera_world_points.shape)}"
        )

    similarity = _estimate_reference_similarity(
        geometry.camera_world_points[reference],
        gt.pointmaps[reference],
        geometry.depth_confidence[reference],
        confidence_threshold=confidence_threshold,
        trim_fraction=alignment_trim_fraction,
    )
    print(
        f"reference depth-camera Sim3 scale={similarity.scale:.6f} "
        f"inliers={similarity.inliers} rmse={similarity.rmse:.6f}"
    )
    query_xy, query_yx = sample_reference_query_points(
        gt.instance_masks[reference],
        geometry.camera_world_points[reference],
        geometry.depth_confidence[reference],
        confidence_threshold=confidence_threshold,
        max_points=max_query_points,
        erosion_radius=query_erosion_radius,
    )
    if query_xy.shape[0] < int(ba_min_tracks):
        raise RuntimeError(
            f"Reference mask produced only {query_xy.shape[0]} valid query points."
        )
    query_world_points = geometry.camera_world_points[reference][
        query_yx[:, 0], query_yx[:, 1]
    ]
    query_depth_confidence = geometry.depth_confidence[reference][
        query_yx[:, 0], query_yx[:, 1]
    ]
    print(f"tracking {query_xy.shape[0]} reference instance points pairwise...")
    tracks = streamvggt.track_reference_points(
        sequence.image_paths,
        reference_index=reference,
        query_points=query_xy,
        iterations=track_iterations,
    )
    _print_track_diagnostics(
        frame_indices=sequence.frame_indices,
        masks=gt.instance_masks,
        tracks=tracks,
        visibility_threshold=track_visibility_threshold,
        confidence_threshold=track_confidence_threshold,
        score_mode=track_score_mode,
    )
    _save_track_visualizations(
        config.output_dir,
        frame_indices=sequence.frame_indices,
        colors=gt.colors,
        masks=gt.instance_masks,
        tracks=tracks,
        visibility_threshold=track_visibility_threshold,
        confidence_threshold=track_confidence_threshold,
        score_mode=track_score_mode,
    )

    unmasked_results = []
    gated_results = []
    for sequence_index, frame_index in enumerate(sequence.frame_indices):
        if sequence_index == reference:
            unmasked = identity_ba_result(
                geometry.world_to_camera[sequence_index],
                reason="reference frame",
            )
            gated = unmasked
        else:
            if track_score_mode == "ignore":
                track_weights = query_depth_confidence
            else:
                track_weights = (
                    query_depth_confidence
                    * tracks.visibility[sequence_index]
                    * tracks.confidence[sequence_index]
                )
            observed_xy = tracks.coordinates[sequence_index]
            reliable_tracks = _reliable_track_mask(
                observed_xy,
                tracks.visibility[sequence_index],
                tracks.confidence[sequence_index],
                image_size=gt.instance_masks[sequence_index].shape,
                visibility_threshold=track_visibility_threshold,
                confidence_threshold=track_confidence_threshold,
                score_mode=track_score_mode,
            )
            gated_tracks = reliable_tracks & points_inside_mask(
                gt.instance_masks[sequence_index],
                tracks.coordinates[sequence_index],
            )
            common = dict(
                world_points=query_world_points,
                observed_xy=tracks.coordinates[sequence_index],
                weights=track_weights,
                base_world_to_camera=geometry.world_to_camera[sequence_index],
                intrinsics=geometry.intrinsics[sequence_index],
                mode=ba_mode,
                iterations=ba_iterations,
                learning_rate=ba_learning_rate,
                robust_delta_pixels=ba_robust_delta_pixels,
                pose_prior_weight=ba_pose_prior_weight,
                min_tracks=ba_min_tracks,
                max_rotation_degrees=ba_max_rotation_degrees,
                max_translation_depth_ratio=ba_max_translation_depth_ratio,
            )
            unmasked = refine_pose_with_tracks(
                **common,
                optimization_mask=reliable_tracks,
            )
            gated = refine_pose_with_tracks(
                **common,
                optimization_mask=gated_tracks,
            )
        unmasked_results.append(unmasked)
        gated_results.append(gated)
        print(
            f"frame={frame_index} visible={int(gt.instance_masks[sequence_index].any())} "
            f"unmasked={unmasked.accepted}/{unmasked.eligible_tracks} "
            f"reproj={unmasked.initial_reprojection_rmse:.3f}->"
            f"{unmasked.final_reprojection_rmse:.3f} "
            f"gated={gated.accepted}/{gated.eligible_tracks} "
            f"reproj={gated.initial_reprojection_rmse:.3f}->"
            f"{gated.final_reprojection_rmse:.3f}"
        )

    raw_native_poses = torch.stack(
        [_homogeneous_pose(pose) for pose in geometry.world_to_camera]
    )
    unmasked_native_poses = torch.stack(
        [result.world_to_camera for result in unmasked_results]
    )
    gated_native_poses = torch.stack(
        [result.world_to_camera for result in gated_results]
    )
    raw_native_points = geometry.camera_world_points
    unmasked_native_points = unproject_depth_to_world(
        geometry.depth,
        unmasked_native_poses,
        geometry.intrinsics,
    )
    gated_native_points = unproject_depth_to_world(
        geometry.depth,
        gated_native_poses,
        geometry.intrinsics,
    )
    raw_points = _align_points(raw_native_points, similarity)
    unmasked_points = _align_points(unmasked_native_points, similarity)
    gated_points = _align_points(gated_native_points, similarity)
    raw_poses = _align_poses(raw_native_poses, similarity)
    unmasked_poses = _align_poses(unmasked_native_poses, similarity)
    gated_poses = _align_poses(gated_native_poses, similarity)

    rows = []
    for sequence_index, frame_index in enumerate(sequence.frame_indices):
        reliable_tracks = _reliable_track_mask(
            tracks.coordinates[sequence_index],
            tracks.visibility[sequence_index],
            tracks.confidence[sequence_index],
            image_size=gt.instance_masks[sequence_index].shape,
            visibility_threshold=track_visibility_threshold,
            confidence_threshold=track_confidence_threshold,
            score_mode=track_score_mode,
        )
        inside_instance = points_inside_mask(
            gt.instance_masks[sequence_index],
            tracks.coordinates[sequence_index],
        )
        row = {
            "sequence_index": sequence_index,
            "frame_index": frame_index,
            "gt_visible": int(gt.instance_masks[sequence_index].any()),
            "query_points": int(query_xy.shape[0]),
            "track_visibility_mean": _finite_tensor_mean(
                tracks.visibility[sequence_index]
            ),
            "track_visibility_max": _finite_tensor_max(
                tracks.visibility[sequence_index]
            ),
            "track_confidence_mean": _finite_tensor_mean(
                tracks.confidence[sequence_index]
            ),
            "track_confidence_max": _finite_tensor_max(
                tracks.confidence[sequence_index]
            ),
            "track_displacement_mean": _finite_tensor_mean(
                torch.linalg.vector_norm(
                    tracks.coordinates[sequence_index] - tracks.query_points,
                    dim=-1,
                )
            ),
            "reliable_tracks": int(reliable_tracks.sum()),
            "reliable_tracks_inside_instance": int(
                (reliable_tracks & inside_instance).sum()
            ),
        }
        _add_ba_fields(row, "unmasked_ba", unmasked_results[sequence_index])
        _add_ba_fields(row, "mask_gated_ba", gated_results[sequence_index])
        for name, poses, points in (
            ("raw", raw_poses, raw_points),
            ("unmasked_ba", unmasked_poses, unmasked_points),
            ("mask_gated_ba", gated_poses, gated_points),
        ):
            rotation, translation = pose_errors(
                poses[sequence_index],
                gt.world_to_camera[sequence_index],
            )
            full = pointmap_errors(points[sequence_index], gt.pointmaps[sequence_index])
            object_error = pointmap_errors(
                points[sequence_index],
                gt.pointmaps[sequence_index],
                mask=gt.instance_masks[sequence_index],
            )
            object_mask = (
                gt.instance_masks[sequence_index]
                & torch.isfinite(points[sequence_index]).all(dim=-1)
                & (
                    geometry.depth_confidence[sequence_index]
                    >= float(confidence_threshold)
                )
            )
            target_object = gt.pointmaps[sequence_index][
                gt.instance_masks[sequence_index]
            ]
            row[f"{name}_pose_rotation_degrees"] = rotation
            row[f"{name}_pose_translation"] = translation
            row[f"{name}_full_point_rmse"] = full["rmse"]
            row[f"{name}_object_point_rmse"] = object_error["rmse"]
            row[f"{name}_object_chamfer"] = symmetric_chamfer(
                points[sequence_index][object_mask],
                target_object,
            )
        rows.append(row)

    _export_pointmaps(
        config.output_dir,
        frame_indices=sequence.frame_indices,
        branches={
            "raw": raw_points,
            "unmasked_ba": unmasked_points,
            "mask_gated_ba": gated_points,
            "gt": gt.pointmaps,
        },
        masks=gt.instance_masks,
        confidence=geometry.depth_confidence,
        colors=gt.colors,
        confidence_threshold=confidence_threshold,
    )
    _write_csv(config.output_dir / "frame_metrics.csv", rows)
    selected = [
        row
        for row in rows
        if row["gt_visible"] and row["sequence_index"] != reference
    ]
    summary = {
        "scene_id": sequence.scene_id,
        "instance_id": sequence.instance_id,
        "label": sequence.label,
        "reference_sequence_index": reference,
        "reference_frame_index": sequence.frame_indices[reference],
        "ba_mode": ba_mode,
        "track_score_mode": track_score_mode,
        "visible_evaluation_frames": len(selected),
        "query_points": int(query_xy.shape[0]),
        "depth_camera_sim3_scale": similarity.scale,
        "depth_camera_sim3_inliers": similarity.inliers,
        "depth_camera_sim3_rmse": similarity.rmse,
        "accepted_unmasked_ba_frames": sum(
            row["unmasked_ba_accepted"] for row in selected
        ),
        "accepted_mask_gated_ba_frames": sum(
            row["mask_gated_ba_accepted"] for row in selected
        ),
    }
    metric_suffixes = (
        "pose_rotation_degrees",
        "pose_translation",
        "full_point_rmse",
        "object_point_rmse",
        "object_chamfer",
    )
    for branch in ("raw", "unmasked_ba", "mask_gated_ba"):
        for suffix in metric_suffixes:
            key = f"{branch}_{suffix}"
            summary[f"mean_{key}"] = _finite_mean(row[key] for row in selected)
    for branch in ("unmasked_ba", "mask_gated_ba"):
        for suffix in metric_suffixes:
            summary[f"mean_{branch}_{suffix}_improvement"] = (
                summary[f"mean_raw_{suffix}"]
                - summary[f"mean_{branch}_{suffix}"]
            )
    _write_csv(config.output_dir / "summary.csv", [summary])
    with (config.output_dir / "transforms.json").open("w", encoding="utf8") as handle:
        json.dump(
            {
                "settings": {
                    "confidence_threshold": confidence_threshold,
                    "max_query_points": max_query_points,
                    "query_erosion_radius": query_erosion_radius,
                    "track_iterations": track_iterations,
                    "track_visibility_threshold": track_visibility_threshold,
                    "track_confidence_threshold": track_confidence_threshold,
                    "track_score_mode": track_score_mode,
                    "ba_mode": ba_mode,
                    "ba_iterations": ba_iterations,
                    "ba_learning_rate": ba_learning_rate,
                    "ba_robust_delta_pixels": ba_robust_delta_pixels,
                    "ba_pose_prior_weight": ba_pose_prior_weight,
                    "ba_min_tracks": ba_min_tracks,
                    "ba_max_rotation_degrees": ba_max_rotation_degrees,
                    "ba_max_translation_depth_ratio": (
                        ba_max_translation_depth_ratio
                    ),
                },
                "similarity": {
                    "scale": similarity.scale,
                    "rotation": similarity.rotation.tolist(),
                    "translation": similarity.translation.tolist(),
                },
                "raw_native_world_to_camera": raw_native_poses.tolist(),
                "unmasked_native_world_to_camera": (
                    unmasked_native_poses.tolist()
                ),
                "mask_gated_native_world_to_camera": gated_native_poses.tolist(),
                "gt_world_to_camera": gt.world_to_camera.tolist(),
            },
            handle,
            indent=2,
        )
    print(f"summary: {config.output_dir / 'summary.csv'}")
    print(f"pointmaps: {config.output_dir / 'pointmaps'}")


def _estimate_reference_similarity(
    source: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
    *,
    confidence_threshold: float,
    trim_fraction: float,
):
    valid = (
        torch.isfinite(source).all(dim=-1)
        & torch.isfinite(target).all(dim=-1)
        & (confidence >= float(confidence_threshold))
    )
    source_points = source[valid]
    target_points = target[valid]
    if source_points.shape[0] > 30_000:
        indices = torch.linspace(0, source_points.shape[0] - 1, 30_000).long()
        source_points = source_points[indices]
        target_points = target_points[indices]
    return estimate_similarity(
        source_points,
        target_points,
        trim_fraction=trim_fraction,
    )


def _align_points(points: torch.Tensor, similarity) -> torch.Tensor:
    return apply_similarity(
        points,
        similarity.scale,
        similarity.rotation,
        similarity.translation,
    )


def _align_poses(poses: torch.Tensor, similarity) -> torch.Tensor:
    return torch.stack([align_world_to_camera(pose, similarity) for pose in poses])


def _homogeneous_pose(pose: torch.Tensor) -> torch.Tensor:
    if pose.shape == (4, 4):
        return pose.detach().float().cpu()
    output = torch.eye(4, dtype=torch.float32)
    output[:3] = pose.detach().float().cpu()
    return output


def _add_ba_fields(row: dict, prefix: str, result) -> None:
    row[f"{prefix}_accepted"] = int(result.accepted)
    row[f"{prefix}_reason"] = result.reason
    row[f"{prefix}_eligible_tracks"] = result.eligible_tracks
    row[f"{prefix}_initial_reprojection_rmse"] = (
        result.initial_reprojection_rmse
    )
    row[f"{prefix}_final_reprojection_rmse"] = result.final_reprojection_rmse
    row[f"{prefix}_delta_rotation_degrees"] = rotation_angle_degrees(
        result.rotation
    )
    row[f"{prefix}_delta_translation"] = float(
        torch.linalg.vector_norm(result.translation)
    )


def _reliable_track_mask(
    coordinates: torch.Tensor,
    visibility: torch.Tensor,
    confidence: torch.Tensor,
    *,
    image_size: tuple[int, int],
    visibility_threshold: float,
    confidence_threshold: float,
    score_mode: str,
) -> torch.Tensor:
    height, width = image_size
    spatially_valid = (
        torch.isfinite(coordinates).all(dim=-1)
        & (coordinates[:, 0] >= 0.0)
        & (coordinates[:, 0] <= width - 1)
        & (coordinates[:, 1] >= 0.0)
        & (coordinates[:, 1] <= height - 1)
    )
    if score_mode == "ignore":
        return spatially_valid
    if score_mode != "threshold":
        raise ValueError(f"Unknown track_score_mode={score_mode!r}.")
    return (
        spatially_valid
        & torch.isfinite(visibility)
        & torch.isfinite(confidence)
        & (visibility >= float(visibility_threshold))
        & (confidence >= float(confidence_threshold))
    )


def _print_track_diagnostics(
    *,
    frame_indices,
    masks: torch.Tensor,
    tracks,
    visibility_threshold: float,
    confidence_threshold: float,
    score_mode: str,
) -> None:
    for sequence_index, frame_index in enumerate(frame_indices):
        reliable = _reliable_track_mask(
            tracks.coordinates[sequence_index],
            tracks.visibility[sequence_index],
            tracks.confidence[sequence_index],
            image_size=masks[sequence_index].shape,
            visibility_threshold=visibility_threshold,
            confidence_threshold=confidence_threshold,
            score_mode=score_mode,
        )
        inside = points_inside_mask(
            masks[sequence_index],
            tracks.coordinates[sequence_index],
        )
        displacement = torch.linalg.vector_norm(
            tracks.coordinates[sequence_index] - tracks.query_points,
            dim=-1,
        )
        print(
            f"track frame={frame_index} "
            f"vis={_finite_tensor_mean(tracks.visibility[sequence_index]):.6f}/"
            f"{_finite_tensor_max(tracks.visibility[sequence_index]):.6f} "
            f"conf={_finite_tensor_mean(tracks.confidence[sequence_index]):.6f}/"
            f"{_finite_tensor_max(tracks.confidence[sequence_index]):.6f} "
            f"motion={_finite_tensor_mean(displacement):.3f}/"
            f"{_finite_tensor_max(displacement):.3f} "
            f"usable={int(reliable.sum())} "
            f"inside={int((reliable & inside).sum())}"
        )


def _save_track_visualizations(
    output_dir: Path,
    *,
    frame_indices,
    colors: np.ndarray,
    masks: torch.Tensor,
    tracks,
    visibility_threshold: float,
    confidence_threshold: float,
    score_mode: str,
) -> None:
    root = output_dir / "track_visualizations"
    root.mkdir(parents=True, exist_ok=True)
    for sequence_index, frame_index in enumerate(frame_indices):
        rgb = np.asarray(colors[sequence_index], dtype=np.uint8).copy()
        mask = masks[sequence_index].cpu().numpy().astype(bool)
        overlay = rgb.copy()
        overlay[mask] = np.array([30, 180, 255], dtype=np.uint8)
        rgb = np.where(mask[..., None], (0.65 * rgb + 0.35 * overlay), rgb)
        image = Image.fromarray(rgb.astype(np.uint8))
        draw = ImageDraw.Draw(image)
        reliable = _reliable_track_mask(
            tracks.coordinates[sequence_index],
            tracks.visibility[sequence_index],
            tracks.confidence[sequence_index],
            image_size=mask.shape,
            visibility_threshold=visibility_threshold,
            confidence_threshold=confidence_threshold,
            score_mode=score_mode,
        )
        inside = points_inside_mask(
            masks[sequence_index],
            tracks.coordinates[sequence_index],
        )
        for point, is_inside in zip(
            tracks.coordinates[sequence_index][reliable],
            inside[reliable],
        ):
            x, y = (float(point[0]), float(point[1]))
            color = (40, 220, 80) if bool(is_inside) else (235, 65, 55)
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
        draw.rectangle((0, 0, 330, 20), fill=(0, 0, 0))
        draw.text(
            (4, 4),
            f"frame={frame_index} reliable={int(reliable.sum())} "
            f"inside={int((reliable & inside).sum())}",
            fill=(255, 255, 255),
        )
        image.save(root / f"frame_{sequence_index:02d}_{frame_index}.png")


def _finite_tensor_mean(values: torch.Tensor) -> float:
    finite = values[torch.isfinite(values)]
    return float(finite.mean()) if finite.numel() else float("nan")


def _finite_tensor_max(values: torch.Tensor) -> float:
    finite = values[torch.isfinite(values)]
    return float(finite.max()) if finite.numel() else float("nan")


def _export_pointmaps(
    output_dir: Path,
    *,
    frame_indices,
    branches: dict[str, torch.Tensor],
    masks: torch.Tensor,
    confidence: torch.Tensor,
    colors: np.ndarray,
    confidence_threshold: float,
) -> None:
    root = output_dir / "pointmaps"
    for sequence_index, frame_index in enumerate(frame_indices):
        prefix = root / f"frame_{sequence_index:02d}_{frame_index}"
        for name, points in branches.items():
            branch_confidence = None if name == "gt" else confidence[sequence_index]
            save_pointmap_ply(
                prefix.with_name(prefix.name + f"_{name}.ply"),
                points[sequence_index],
                colors[sequence_index],
                confidence=branch_confidence,
                confidence_threshold=confidence_threshold,
            )
            save_pointmap_ply(
                prefix.with_name(prefix.name + f"_{name}_object.ply"),
                points[sequence_index],
                colors[sequence_index],
                mask=masks[sequence_index],
                confidence=branch_confidence,
                confidence_threshold=confidence_threshold,
            )
    for name, points in branches.items():
        branch_confidence = None if name == "gt" else confidence
        save_aggregate_ply(
            root / f"sequence_{name}.ply",
            points,
            colors,
            confidence=branch_confidence,
            confidence_threshold=confidence_threshold,
        )
        save_aggregate_ply(
            root / f"object_{name}.ply",
            points,
            colors,
            masks=masks,
            confidence=branch_confidence,
            confidence_threshold=confidence_threshold,
        )


def _finite_mean(values) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Oracle GT-mask-gated StreamVGGT track-head pose BA."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--scene-id")
    parser.add_argument("--instance-id", type=int)
    parser.add_argument("--frame-indices", type=int, nargs="+")
    parser.add_argument("--reference-sequence-index", type=int)
    parser.add_argument("--geometry-device")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--confidence-threshold", type=float, default=0.30)
    parser.add_argument("--alignment-trim-fraction", type=float, default=0.70)
    parser.add_argument("--max-query-points", type=int, default=512)
    parser.add_argument("--query-erosion-radius", type=int, default=2)
    parser.add_argument("--track-iterations", type=int, default=4)
    parser.add_argument("--track-visibility-threshold", type=float, default=0.05)
    parser.add_argument("--track-confidence-threshold", type=float, default=0.05)
    parser.add_argument(
        "--track-score-mode",
        choices=("threshold", "ignore"),
        default="threshold",
        help="Use track scores normally or ignore them only for coordinate diagnostics.",
    )
    parser.add_argument(
        "--ba-mode",
        choices=("translation_only", "full_se3"),
        default="full_se3",
    )
    parser.add_argument("--ba-iterations", type=int, default=200)
    parser.add_argument("--ba-learning-rate", type=float, default=0.03)
    parser.add_argument("--ba-robust-delta-pixels", type=float, default=4.0)
    parser.add_argument("--ba-pose-prior-weight", type=float, default=0.10)
    parser.add_argument("--ba-min-tracks", type=int, default=24)
    parser.add_argument("--ba-max-rotation-degrees", type=float, default=15.0)
    parser.add_argument(
        "--ba-max-translation-depth-ratio",
        type=float,
        default=0.25,
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
