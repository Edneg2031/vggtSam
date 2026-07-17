"""Use SAM3 instance masks to refine StreamVGGT pointmaps and pose proxies."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from test_sam.data import load_mask_tracking_sequence

from .backbones.sam3_wrapper import SAM3Wrapper
from .backbones.streamvggt_wrapper import StreamVGGTWrapper
from .bridge.gating import binary_iou
from .config import ExperimentConfig, load_config
from .geometry.export import save_aggregate_ply
from .geometry.gt_data import load_gt_geometry_sequence
from .geometry.registration import (
    align_world_to_camera,
    apply_rigid,
    apply_similarity,
    apply_world_correction_to_pose,
    nearest_neighbors,
    pose_errors,
    robust_icp,
    rotation_angle_degrees,
    symmetric_chamfer,
)
from .gt_mask_pose_experiment import (
    _camera_center,
    _finite_mean,
    _identity_icp,
    _plot_camera_trajectories,
    _write_csv,
    pointmap_errors,
    select_causal_reference,
)
from .pipeline import _mine_recovery, _resize_target_masks
from .sam3_mask_object_fusion_experiment import (
    _reference_similarity,
    _sam_masks_to_stream,
    _save_mask_report,
)


MASK_SOURCES = ("gt_oracle", "sam3_hard_memory")


def main() -> None:
    args = _parse_args()
    overrides = {
        key: value
        for key, value in {
            "manifest": args.manifest,
            "scene_id": args.scene_id,
            "instance_id": args.instance_id,
            "frame_indices": args.frame_indices,
            "sam3_device": args.sam3_device,
            "geometry_device": args.geometry_device,
            "output_dir": args.output_dir,
        }.items()
        if value is not None
    }
    run_experiment(
        load_config(args.config, overrides),
        geometry_source=args.geometry_source,
        alignment_confidence_threshold=args.alignment_confidence_threshold,
        icp_confidence_threshold=args.icp_confidence_threshold,
        alignment_trim_fraction=args.alignment_trim_fraction,
        icp_max_points=args.icp_max_points,
        icp_iterations=args.icp_iterations,
        icp_trim_fraction=args.icp_trim_fraction,
        icp_max_correspondence=args.icp_max_correspondence,
        icp_min_inliers=args.icp_min_inliers,
        icp_min_fitness=args.icp_min_fitness,
        icp_max_rmse=args.icp_max_rmse,
        pose_refinement_mode=args.pose_refinement_mode,
        reference_sequence_index=args.reference_sequence_index,
        delta_scales=args.delta_scales,
        object_map_modes=args.object_map_modes,
        object_map_voxel_size=args.object_map_voxel_size,
        object_map_max_points=args.object_map_max_points,
        scene_consistency=args.scene_consistency,
        scene_candidate_scales=args.scene_candidate_scales,
        scene_confidence_threshold=args.scene_confidence_threshold,
        scene_rmse_tolerance=args.scene_rmse_tolerance,
        scene_fitness_drop_tolerance=args.scene_fitness_drop_tolerance,
        scene_min_inliers=args.scene_min_inliers,
        scene_map_voxel_size=args.scene_map_voxel_size,
        scene_map_max_points=args.scene_map_max_points,
    )


def run_experiment(
    config: ExperimentConfig,
    *,
    geometry_source: str,
    alignment_confidence_threshold: float,
    icp_confidence_threshold: float,
    alignment_trim_fraction: float,
    icp_max_points: int,
    icp_iterations: int,
    icp_trim_fraction: float,
    icp_max_correspondence: float,
    icp_min_inliers: int,
    icp_min_fitness: float,
    icp_max_rmse: float,
    pose_refinement_mode: str,
    reference_sequence_index: int | None,
    delta_scales: list[float],
    object_map_modes: list[str],
    object_map_voxel_size: float,
    object_map_max_points: int,
    scene_consistency: str,
    scene_candidate_scales: list[float],
    scene_confidence_threshold: float,
    scene_rmse_tolerance: float,
    scene_fitness_drop_tolerance: float,
    scene_min_inliers: int,
    scene_map_voxel_size: float,
    scene_map_max_points: int,
) -> None:
    delta_scales = _validate_delta_scales(delta_scales, pose_refinement_mode)
    object_map_modes = _validate_object_map_modes(object_map_modes)
    scene_candidate_scales = _validate_scene_candidate_scales(
        scene_candidate_scales
    )
    if scene_consistency == "guard" and pose_refinement_mode != "translation_only":
        raise ValueError("Scene consistency guard currently requires translation_only.")
    if float(object_map_voxel_size) < 0.0:
        raise ValueError("object_map_voxel_size must be non-negative.")
    if int(object_map_max_points) < 1:
        raise ValueError("object_map_max_points must be positive.")
    if float(scene_map_voxel_size) < 0.0:
        raise ValueError("scene_map_voxel_size must be non-negative.")
    if int(scene_map_max_points) < 1 or int(scene_min_inliers) < 1:
        raise ValueError("Scene map sizes and inlier counts must be positive.")
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
    if "causal" in object_map_modes and reference != 0:
        raise ValueError(
            "The causal object-map ablation requires sequence index 0 as the "
            "reference to avoid using a future observation."
        )
    sequence = replace(sequence, reference_frame_idx=reference)
    target_output_masks = _resize_target_masks(
        sequence.target_masks,
        config.output_size,
    )
    print(
        f"target scene={sequence.scene_id} frames={sequence.frame_indices} "
        f"instance={sequence.instance_id} label={sequence.label!r} "
        f"reference={reference} pose_refinement={pose_refinement_mode}"
    )

    print("running frozen SAM3 original tracker...")
    sam3 = SAM3Wrapper(
        repo_path=config.sam3_repo,
        checkpoint_path=config.sam3_checkpoint,
        device=config.sam3_device,
        output_threshold=config.sam3_output_threshold,
        prompt_with_box=config.prompt_with_box,
    ).load()
    original_tracking = sam3.track(
        sequence.image_paths,
        prompt=sequence.label,
        output_size=config.output_size,
        reference_frame_idx=reference,
        reference_mask=target_output_masks[reference],
    )

    print("running frozen StreamVGGT once with causal caches...")
    geometry = StreamVGGTWrapper(
        repo_path=config.streamvggt_repo,
        checkpoint_path=config.streamvggt_checkpoint,
        device=config.geometry_device,
        image_mode=config.image_mode,
        streaming_cache=config.streaming_cache,
    ).load().extract(sequence.image_paths)
    geometry_points, geometry_confidence = _select_geometry_source(
        geometry,
        geometry_source,
    )

    gt = load_gt_geometry_sequence(
        config.manifest,
        scene_id=sequence.scene_id,
        frame_indices=sequence.frame_indices,
        instance_id=sequence.instance_id,
        processed_size=geometry.processed_size,
        image_mode=config.image_mode,
    )
    if tuple(geometry_points.shape) != tuple(gt.pointmaps.shape):
        raise RuntimeError(
            f"{geometry_source} pointmap shape {tuple(geometry_points.shape)} "
            f"does not match GT pointmap shape {tuple(gt.pointmaps.shape)}."
        )
    if tuple(geometry_confidence.shape) != tuple(geometry_points.shape[:-1]):
        raise RuntimeError(
            f"{geometry_source} confidence shape "
            f"{tuple(geometry_confidence.shape)} does not match pointmap grid "
            f"{tuple(geometry_points.shape[:-1])}."
        )
    hard_tracking, recovery = _run_hard_recovery(
        config,
        sequence=sequence,
        target_output_masks=target_output_masks,
        original_tracking=original_tracking,
        geometry=geometry,
        sam3=sam3,
    )
    del sam3
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    similarity = _reference_similarity(
        geometry_points[reference],
        geometry_confidence[reference],
        gt.pointmaps[reference],
        confidence_threshold=alignment_confidence_threshold,
        trim_fraction=alignment_trim_fraction,
    )
    aligned_points = apply_similarity(
        geometry_points,
        similarity.scale,
        similarity.rotation,
        similarity.translation,
    )
    raw_poses = torch.stack(
        [align_world_to_camera(pose, similarity) for pose in geometry.world_to_camera]
    )
    print(
        f"{geometry_source} reference Sim3 scale={similarity.scale:.6f} "
        f"inliers={similarity.inliers} rmse={similarity.rmse:.6f}"
    )

    masks = {
        "gt_oracle": gt.instance_masks.clone(),
        "sam3_hard_memory": _sam_masks_to_stream(
            hard_tracking.masks,
            geometry=geometry,
            image_mode=config.image_mode,
        ),
    }
    for branch_masks in masks.values():
        branch_masks[reference] = gt.instance_masks[reference]

    reference_valid = (
        gt.instance_masks[reference]
        & torch.isfinite(aligned_points[reference]).all(dim=-1)
        & (
            geometry_confidence[reference]
            >= float(icp_confidence_threshold)
        )
    )
    reference_points = aligned_points[reference][reference_valid]
    if reference_points.shape[0] < int(icp_min_inliers):
        raise RuntimeError(
            f"Reference instance contains too few confident {geometry_source} points: "
            f"{reference_points.shape[0]}."
        )

    all_rows: list[dict] = []
    summary_rows: list[dict] = []
    branch_outputs = {}
    branch_sources = {}
    transforms = {}
    for source in MASK_SOURCES:
        for object_map_mode in object_map_modes:
            for delta_scale in delta_scales:
                branch_name = (
                    f"{source}_{object_map_mode}_{scene_consistency}_alpha_"
                    f"{_format_scale(delta_scale)}"
                )
                rows, refined_points, refined_poses, corrections = _run_pose_branch(
                    source,
                    delta_scale=delta_scale,
                    object_map_mode=object_map_mode,
                    object_map_voxel_size=object_map_voxel_size,
                    object_map_max_points=object_map_max_points,
                    scene_consistency=scene_consistency,
                    scene_candidate_scales=scene_candidate_scales,
                    scene_confidence_threshold=scene_confidence_threshold,
                    scene_rmse_tolerance=scene_rmse_tolerance,
                    scene_fitness_drop_tolerance=scene_fitness_drop_tolerance,
                    scene_min_inliers=scene_min_inliers,
                    scene_map_voxel_size=scene_map_voxel_size,
                    scene_map_max_points=scene_map_max_points,
                    masks=masks[source],
                    reference=reference,
                    reference_points=reference_points,
                    raw_points=aligned_points,
                    raw_poses=raw_poses,
                    confidence=geometry_confidence,
                    gt_masks=gt.instance_masks,
                    gt_points=gt.pointmaps,
                    gt_poses=gt.world_to_camera,
                    frame_indices=sequence.frame_indices,
                    icp_confidence_threshold=icp_confidence_threshold,
                    icp_max_points=icp_max_points,
                    icp_iterations=icp_iterations,
                    icp_trim_fraction=icp_trim_fraction,
                    icp_max_correspondence=icp_max_correspondence,
                    icp_min_inliers=icp_min_inliers,
                    icp_min_fitness=icp_min_fitness,
                    icp_max_rmse=icp_max_rmse,
                    translation_only=pose_refinement_mode == "translation_only",
                )
                all_rows.extend(rows)
                transforms[branch_name] = corrections
                branch_outputs[branch_name] = (refined_points, refined_poses)
                branch_sources[branch_name] = source
                branch_metrics = _summarize_branch(
                    source,
                    rows,
                    reference=reference,
                    recovery_triggered=recovery["triggered"],
                )
                summary = {
                    "experiment_name": config.output_dir.name,
                    "scene_id": sequence.scene_id,
                    "instance_id": sequence.instance_id,
                    "instance_label": sequence.label,
                    "frame_indices": " ".join(
                        str(frame_index) for frame_index in sequence.frame_indices
                    ),
                    "mask_source": source,
                    "delta_scale": delta_scale,
                    "object_map_mode": object_map_mode,
                    "object_map_voxel_size": object_map_voxel_size,
                    "object_map_max_points": object_map_max_points,
                    "scene_consistency": scene_consistency,
                    "scene_candidate_scales": " ".join(
                        str(value) for value in scene_candidate_scales
                    ),
                    "scene_confidence_threshold": scene_confidence_threshold,
                    "scene_rmse_tolerance": scene_rmse_tolerance,
                    "scene_fitness_drop_tolerance": scene_fitness_drop_tolerance,
                    "scene_min_inliers": scene_min_inliers,
                    "scene_map_voxel_size": scene_map_voxel_size,
                    "scene_map_max_points": scene_map_max_points,
                    "pose_refinement_mode": pose_refinement_mode,
                    "alignment_confidence_threshold": alignment_confidence_threshold,
                    "icp_confidence_threshold": icp_confidence_threshold,
                    "alignment_trim_fraction": alignment_trim_fraction,
                    "icp_max_points": icp_max_points,
                    "icp_iterations": icp_iterations,
                    "icp_trim_fraction": icp_trim_fraction,
                    "icp_max_correspondence": icp_max_correspondence,
                    "icp_min_inliers": icp_min_inliers,
                    "icp_min_fitness": icp_min_fitness,
                    "icp_max_rmse": icp_max_rmse,
                    "reference_sequence_index": reference,
                    "reference_frame_index": sequence.frame_indices[reference],
                    "geometry_source": geometry_source,
                    "pose_delta_interpretation": (
                        "pointmap_registration_proxy"
                        if geometry_source == "point_head"
                        else "camera_consistent"
                    ),
                    "reference_selected_points": int(reference_points.shape[0]),
                    "reference_sim3_scale": similarity.scale,
                    "reference_sim3_inliers": similarity.inliers,
                    "reference_sim3_rmse": similarity.rmse,
                    "recovery_sequence_index": recovery["sequence_index"],
                    "recovery_frame_index": recovery["frame_index"],
                    **{
                        key: value
                        for key, value in branch_metrics.items()
                        if key not in {"mask_source", "reference_sequence_index"}
                    },
                }
                summary_rows.append(summary)
                ate_name = (
                    "ATE_PROXY" if geometry_source == "point_head" else "ATE_RMSE"
                )
                print(
                    f"trajectory source={source:<18} map={object_map_mode:<14} "
                    f"scene={scene_consistency:<5} "
                    f"alpha={delta_scale:.2f} "
                    f"{ate_name}={summary['raw_ate_rmse']:.4f}->"
                    f"{summary['refined_ate_rmse']:.4f} "
                    f"improvement={summary['ate_rmse_improvement']:.4f}"
                )
                if source == "sam3_hard_memory":
                    _plot_camera_trajectories(
                        config.output_dir / f"camera_trajectories_{branch_name}.png",
                        frame_indices=sequence.frame_indices,
                        gt_world_to_camera=gt.world_to_camera,
                        raw_world_to_camera=raw_poses,
                        refined_world_to_camera=refined_poses,
                        rows=rows,
                        title=(
                            f"{geometry_source}: {source}, {object_map_mode}, "
                            f"{scene_consistency}, alpha={delta_scale:g}"
                        ),
                        raw_translation_key="raw_pose_translation",
                        refined_translation_key="refined_pose_translation",
                        raw_rotation_key="raw_pose_rotation_degrees",
                        refined_rotation_key="refined_pose_rotation_degrees",
                    )

    _export_pointmaps(
        config.output_dir,
        raw_points=aligned_points,
        gt_points=gt.pointmaps,
        gt_masks=gt.instance_masks,
        colors=gt.colors,
        confidence=geometry_confidence,
        confidence_threshold=icp_confidence_threshold,
        masks=masks,
        branch_sources=branch_sources,
        branch_outputs=branch_outputs,
    )
    _save_mask_report(
        config.output_dir / "mask_sources.png",
        image_paths=sequence.image_paths,
        frame_indices=sequence.frame_indices,
        gt_masks=target_output_masks,
        original_masks=original_tracking.masks,
        memory_masks=hard_tracking.masks,
        recovery_index=recovery["sequence_index"],
        output_size=config.output_size,
        include_original=True,
    )
    _write_csv(config.output_dir / "frame_metrics.csv", all_rows)
    _write_csv(config.output_dir / "summary.csv", summary_rows)
    with (config.output_dir / "transforms.json").open("w", encoding="utf8") as handle:
        json.dump(
            {
                "settings": {
                    "mask_sources": MASK_SOURCES,
                    "geometry_source": geometry_source,
                    "pose_delta_interpretation": (
                        "pointmap_registration_proxy"
                        if geometry_source == "point_head"
                        else "camera_consistent"
                    ),
                    "camera_delta_applied_to_full_frame": True,
                    "pose_refinement_mode": pose_refinement_mode,
                    "delta_scales": delta_scales,
                    "object_map_modes": object_map_modes,
                    "object_map_voxel_size": object_map_voxel_size,
                    "object_map_max_points": object_map_max_points,
                    "scene_consistency": scene_consistency,
                    "scene_candidate_scales": scene_candidate_scales,
                    "scene_confidence_threshold": scene_confidence_threshold,
                    "scene_rmse_tolerance": scene_rmse_tolerance,
                    "scene_fitness_drop_tolerance": scene_fitness_drop_tolerance,
                    "scene_min_inliers": scene_min_inliers,
                    "scene_map_voxel_size": scene_map_voxel_size,
                    "scene_map_max_points": scene_map_max_points,
                    "reference_sequence_index": reference,
                    "alignment_confidence_threshold": alignment_confidence_threshold,
                    "icp_confidence_threshold": icp_confidence_threshold,
                    "alignment_trim_fraction": alignment_trim_fraction,
                    "icp_max_points": icp_max_points,
                    "icp_iterations": icp_iterations,
                    "icp_trim_fraction": icp_trim_fraction,
                    "icp_max_correspondence": icp_max_correspondence,
                    "icp_min_inliers": icp_min_inliers,
                    "icp_min_fitness": icp_min_fitness,
                    "icp_max_rmse": icp_max_rmse,
                    "gt_usage": "GT oracle branch, reference Sim3, and metrics only",
                },
                "similarity": {
                    "scale": similarity.scale,
                    "rotation": similarity.rotation.tolist(),
                    "translation": similarity.translation.tolist(),
                    "inliers": similarity.inliers,
                    "rmse": similarity.rmse,
                },
                "hard_recovery": recovery,
                "corrections": transforms,
                "raw_world_to_camera": raw_poses.tolist(),
                "refined_world_to_camera": {
                    source: output[1].tolist()
                    for source, output in branch_outputs.items()
                },
                "gt_world_to_camera": gt.world_to_camera.tolist(),
            },
            handle,
            indent=2,
        )
    print(f"summary: {config.output_dir / 'summary.csv'}")
    print(f"frame metrics: {config.output_dir / 'frame_metrics.csv'}")
    print(f"camera trajectories: {config.output_dir / 'camera_trajectories_*.png'}")
    print(f"pointmaps: {config.output_dir / 'pointmaps'}")


def _select_geometry_source(geometry, geometry_source):
    if geometry_source == "point_head":
        if geometry.world_points is None or geometry.confidence is None:
            raise RuntimeError("StreamVGGT point-head pointmap is unavailable.")
        return geometry.world_points, geometry.confidence
    if geometry_source == "depth_camera":
        if geometry.camera_world_points is None or geometry.depth_confidence is None:
            raise RuntimeError("StreamVGGT depth-camera pointmap is unavailable.")
        return geometry.camera_world_points, geometry.depth_confidence
    raise ValueError(f"Unsupported geometry_source={geometry_source!r}.")


def _run_hard_recovery(
    config,
    *,
    sequence,
    target_output_masks,
    original_tracking,
    geometry,
    sam3,
):
    result = _mine_recovery(
        config,
        sequence=sequence,
        target_masks=target_output_masks,
        original_masks=original_tracking.masks,
        original_scores=original_tracking.scores,
        geometry=geometry,
    )
    recovery_index = next(
        (
            index
            for index, row in enumerate(result["rows"])
            if row["use_correction"]
            and result["candidates"][index].supported_mask.any()
        ),
        None,
    )
    if recovery_index is None:
        print("hard recovery not triggered; hard branch equals SAM3 original")
        return original_tracking, {
            "triggered": False,
            "sequence_index": None,
            "frame_index": None,
            "persistent_obj_id": original_tracking.selected_obj_id,
        }
    candidate = result["candidates"][recovery_index]
    recovery_mask, recovery_score = sam3.recover_mask_with_text_geometry(
        sequence.image_paths[recovery_index],
        prompt=sequence.label,
        output_size=config.output_size,
        candidate_mask=candidate.mask,
        supported_mask=candidate.supported_mask,
    )
    if not recovery_mask.any():
        raise RuntimeError("Hard geometry-guided recovery returned an empty mask.")
    tracking = sam3.track_with_recovery_mask_memory(
        sequence.image_paths,
        prompt=sequence.label,
        output_size=config.output_size,
        reference_frame_idx=sequence.reference_frame_idx,
        reference_mask=target_output_masks[sequence.reference_frame_idx],
        recovery_frame_idx=recovery_index,
        recovery_mask=recovery_mask,
    )
    if tracking.selected_obj_id != original_tracking.selected_obj_id:
        raise RuntimeError(
            "Hard recovery changed the persistent SAM3 obj_id: "
            f"{original_tracking.selected_obj_id} -> {tracking.selected_obj_id}."
        )
    for index in range(recovery_index):
        if not torch.equal(tracking.masks[index], original_tracking.masks[index]):
            raise RuntimeError(
                "Original and hard-recovery tracks diverged before recovery at "
                f"sequence index {index}."
            )
    print(
        f"hard recovery frame={sequence.frame_indices[recovery_index]} "
        f"score={recovery_score:.4f} obj_id={tracking.selected_obj_id}"
    )
    return tracking, {
        "triggered": True,
        "sequence_index": recovery_index,
        "frame_index": sequence.frame_indices[recovery_index],
        "mask_score": float(recovery_score),
        "persistent_obj_id": tracking.selected_obj_id,
    }


def _run_pose_branch(
    source,
    *,
    delta_scale,
    object_map_mode,
    object_map_voxel_size,
    object_map_max_points,
    scene_consistency,
    scene_candidate_scales,
    scene_confidence_threshold,
    scene_rmse_tolerance,
    scene_fitness_drop_tolerance,
    scene_min_inliers,
    scene_map_voxel_size,
    scene_map_max_points,
    masks,
    reference,
    reference_points,
    raw_points,
    raw_poses,
    confidence,
    gt_masks,
    gt_points,
    gt_poses,
    frame_indices,
    icp_confidence_threshold,
    icp_max_points,
    icp_iterations,
    icp_trim_fraction,
    icp_max_correspondence,
    icp_min_inliers,
    icp_min_fitness,
    icp_max_rmse,
    translation_only,
):
    refined_points = raw_points.clone()
    refined_poses = raw_poses.clone()
    object_map = reference_points.clone()
    reference_scene_valid = (
        torch.isfinite(raw_points[reference]).all(dim=-1)
        & (confidence[reference] >= float(scene_confidence_threshold))
    )
    scene_map = raw_points[reference][reference_scene_valid].clone()
    rows = []
    corrections = []
    for sequence_index, frame_index in enumerate(frame_indices):
        selected = (
            masks[sequence_index]
            & torch.isfinite(raw_points[sequence_index]).all(dim=-1)
            & (
                confidence[sequence_index]
                >= float(icp_confidence_threshold)
            )
        )
        moving = raw_points[sequence_index][selected]
        object_map_points_before = int(object_map.shape[0])
        object_map_updated = False
        applied_delta_scale = 0.0
        scene_guard = {
            "reason": "disabled",
            "raw_rmse": float("nan"),
            "raw_fitness": float("nan"),
            "refined_rmse": float("nan"),
            "refined_fitness": float("nan"),
            "inliers": 0,
        }
        if sequence_index == reference:
            icp = _identity_icp(raw_points.dtype, reason="reference frame")
        elif moving.shape[0] < int(icp_min_inliers):
            icp = _identity_icp(raw_points.dtype, reason="too few mask-selected points")
        else:
            icp = robust_icp(
                moving,
                (
                    object_map
                    if object_map_mode == "causal"
                    else reference_points
                ),
                moving_weights=confidence[sequence_index][selected],
                max_points=icp_max_points,
                iterations=icp_iterations,
                trim_fraction=icp_trim_fraction,
                max_correspondence_distance=icp_max_correspondence,
                min_inliers=icp_min_inliers,
                min_fitness=icp_min_fitness,
                max_rmse=icp_max_rmse,
                translation_only=translation_only,
            )
            if icp.accepted:
                applied_delta_scale = float(delta_scale)
                if scene_consistency == "guard" and applied_delta_scale > 0.0:
                    scene_support = (
                        torch.isfinite(raw_points[sequence_index]).all(dim=-1)
                        & (
                            confidence[sequence_index]
                            >= float(scene_confidence_threshold)
                        )
                        & ~masks[sequence_index]
                    )
                    applied_delta_scale, scene_guard = _select_scene_guard_scale(
                        raw_points[sequence_index][scene_support],
                        scene_map,
                        translation=icp.translation,
                        max_scale=float(delta_scale),
                        candidate_scales=scene_candidate_scales,
                        max_points=icp_max_points,
                        trim_fraction=icp_trim_fraction,
                        max_correspondence=icp_max_correspondence,
                        min_inliers=scene_min_inliers,
                        rmse_tolerance=scene_rmse_tolerance,
                        fitness_drop_tolerance=scene_fitness_drop_tolerance,
                    )
                applied_rotation = icp.rotation
                applied_translation = icp.translation * applied_delta_scale
                if applied_delta_scale > 0.0:
                    refined_points[sequence_index] = apply_rigid(
                        raw_points[sequence_index],
                        applied_rotation,
                        applied_translation,
                    )
                    refined_poses[sequence_index] = apply_world_correction_to_pose(
                        raw_poses[sequence_index],
                        applied_rotation,
                        applied_translation,
                    )
                if object_map_mode == "causal":
                    map_observation = apply_rigid(
                        moving,
                        icp.rotation,
                        icp.translation,
                    )
                    object_map = _merge_object_map(
                        object_map,
                        map_observation,
                        voxel_size=object_map_voxel_size,
                        max_points=object_map_max_points,
                    )
                    object_map_updated = True

        if sequence_index != reference and scene_consistency == "guard":
            scene_map_valid = (
                torch.isfinite(refined_points[sequence_index]).all(dim=-1)
                & (
                    confidence[sequence_index]
                    >= float(scene_confidence_threshold)
                )
            )
            scene_map = _merge_object_map(
                scene_map,
                refined_points[sequence_index][scene_map_valid],
                voxel_size=scene_map_voxel_size,
                max_points=scene_map_max_points,
            )

        raw_rotation, raw_translation = pose_errors(
            raw_poses[sequence_index], gt_poses[sequence_index]
        )
        refined_rotation, refined_translation = pose_errors(
            refined_poses[sequence_index], gt_poses[sequence_index]
        )
        raw_center = _camera_center(raw_poses[sequence_index])
        refined_center = _camera_center(refined_poses[sequence_index])
        camera_delta = refined_center - raw_center
        raw_full = pointmap_errors(raw_points[sequence_index], gt_points[sequence_index])
        refined_full = pointmap_errors(
            refined_points[sequence_index], gt_points[sequence_index]
        )
        raw_object = pointmap_errors(
            raw_points[sequence_index],
            gt_points[sequence_index],
            mask=gt_masks[sequence_index],
        )
        refined_object = pointmap_errors(
            refined_points[sequence_index],
            gt_points[sequence_index],
            mask=gt_masks[sequence_index],
        )
        target_object = gt_points[sequence_index][gt_masks[sequence_index]]
        raw_object_cloud = raw_points[sequence_index][gt_masks[sequence_index]]
        refined_object_cloud = refined_points[sequence_index][gt_masks[sequence_index]]
        row = {
            "mask_source": source,
            "delta_scale": float(delta_scale),
            "applied_delta_scale": applied_delta_scale,
            "object_map_mode": object_map_mode,
            "scene_consistency": scene_consistency,
            "sequence_index": sequence_index,
            "frame_index": frame_index,
            "is_reference": int(sequence_index == reference),
            "gt_visible": int(gt_masks[sequence_index].any()),
            "mask_iou": binary_iou(masks[sequence_index], gt_masks[sequence_index]),
            "mask_pixels": int(masks[sequence_index].sum()),
            "selected_points": int(selected.sum()),
            "selected_contamination_ratio": float(
                (selected & ~gt_masks[sequence_index]).sum()
            )
            / max(int(selected.sum()), 1),
            "icp_accepted": int(icp.accepted),
            "icp_reason": icp.reason,
            "icp_inliers": icp.inliers,
            "icp_fitness": icp.fitness,
            "icp_rmse": icp.rmse,
            "icp_rotation_degrees": rotation_angle_degrees(icp.rotation),
            "icp_translation": float(torch.linalg.vector_norm(icp.translation)),
            "applied_icp_translation": float(
                torch.linalg.vector_norm(icp.translation) * applied_delta_scale
                if icp.accepted
                else 0.0
            ),
            "correction_applied": int(
                icp.accepted and applied_delta_scale > 0.0
            ),
            "scene_guard_reason": scene_guard["reason"],
            "scene_raw_rmse": scene_guard["raw_rmse"],
            "scene_refined_rmse": scene_guard["refined_rmse"],
            "scene_raw_fitness": scene_guard["raw_fitness"],
            "scene_refined_fitness": scene_guard["refined_fitness"],
            "scene_guard_inliers": scene_guard["inliers"],
            "scene_map_points": int(scene_map.shape[0]),
            "object_map_points_before": object_map_points_before,
            "object_map_points_after": int(object_map.shape[0]),
            "object_map_new_points": int(object_map.shape[0])
            - object_map_points_before,
            "object_map_updated": int(object_map_updated),
            "object_map_update_uses_full_icp": int(
                object_map_mode == "causal" and icp.accepted
            ),
            "raw_pose_rotation_degrees": raw_rotation,
            "refined_pose_rotation_degrees": refined_rotation,
            "raw_pose_translation": raw_translation,
            "refined_pose_translation": refined_translation,
            "pose_translation_improvement": raw_translation - refined_translation,
            "camera_delta_x": float(camera_delta[0]),
            "camera_delta_y": float(camera_delta[1]),
            "camera_delta_z": float(camera_delta[2]),
            "camera_delta_norm": float(torch.linalg.vector_norm(camera_delta)),
            "raw_full_point_rmse": raw_full["rmse"],
            "refined_full_point_rmse": refined_full["rmse"],
            "full_point_rmse_improvement": raw_full["rmse"]
            - refined_full["rmse"],
            "raw_object_point_rmse": raw_object["rmse"],
            "refined_object_point_rmse": refined_object["rmse"],
            "raw_object_chamfer": symmetric_chamfer(
                raw_object_cloud, target_object
            ),
            "refined_object_chamfer": symmetric_chamfer(
                refined_object_cloud, target_object
            ),
        }
        rows.append(row)
        correction = torch.eye(4, dtype=raw_points.dtype)
        if icp.accepted and applied_delta_scale > 0.0:
            correction[:3, :3] = icp.rotation
            correction[:3, 3] = icp.translation * applied_delta_scale
        corrections.append(
            {
                "sequence_index": sequence_index,
                "frame_index": frame_index,
                "accepted": icp.accepted,
                "delta_scale": float(delta_scale),
                "applied_delta_scale": applied_delta_scale,
                "object_map_mode": object_map_mode,
                "scene_consistency": scene_consistency,
                "scene_guard": scene_guard,
                "object_map_points_before": object_map_points_before,
                "object_map_points_after": int(object_map.shape[0]),
                "object_map_updated": object_map_updated,
                "reason": icp.reason,
                "estimated_translation": icp.translation.tolist(),
                "applied": correction.tolist(),
            }
        )
        print(
            f"source={source:<18} alpha={float(delta_scale):.2f} "
            f"applied={applied_delta_scale:.2f} "
            f"map={object_map_mode:<14} "
            f"frame={frame_index} "
            f"visible={row['gt_visible']} mask_iou={row['mask_iou']:.4f} "
            f"icp={icp.accepted} pose_t={raw_translation:.4f}->"
            f"{refined_translation:.4f} full_rmse={raw_full['rmse']:.4f}->"
            f"{refined_full['rmse']:.4f}"
        )
    return rows, refined_points, refined_poses, corrections


def _select_scene_guard_scale(
    moving_scene,
    scene_map,
    *,
    translation,
    max_scale,
    candidate_scales,
    max_points,
    trim_fraction,
    max_correspondence,
    min_inliers,
    rmse_tolerance,
    fitness_drop_tolerance,
):
    raw = _scene_nn_score(
        moving_scene,
        scene_map,
        max_points=max_points,
        trim_fraction=trim_fraction,
        max_correspondence=max_correspondence,
        min_inliers=min_inliers,
    )
    if not raw["valid"]:
        return float(max_scale), {
            "reason": "insufficient scene overlap; keep object ICP",
            "raw_rmse": raw["rmse"],
            "raw_fitness": raw["fitness"],
            "refined_rmse": raw["rmse"],
            "refined_fitness": raw["fitness"],
            "inliers": raw["inliers"],
        }

    best_scale = 0.0
    best = raw
    for fraction in sorted(float(value) for value in candidate_scales):
        scale = float(max_scale) * fraction
        candidate = _scene_nn_score(
            moving_scene + translation * scale,
            scene_map,
            max_points=max_points,
            trim_fraction=trim_fraction,
            max_correspondence=max_correspondence,
            min_inliers=min_inliers,
        )
        if not candidate["valid"]:
            continue
        rmse_ok = candidate["rmse"] <= raw["rmse"] * (
            1.0 + float(rmse_tolerance)
        )
        fitness_ok = candidate["fitness"] >= raw["fitness"] - float(
            fitness_drop_tolerance
        )
        if rmse_ok and fitness_ok and scale >= best_scale:
            best_scale = scale
            best = candidate
    if best_scale >= float(max_scale) - 1e-8:
        reason = "full object ICP passes scene guard"
    elif best_scale > 0.0:
        reason = "object ICP damped by scene guard"
    else:
        reason = "object ICP blocked by scene guard"
    return best_scale, {
        "reason": reason,
        "raw_rmse": raw["rmse"],
        "raw_fitness": raw["fitness"],
        "refined_rmse": best["rmse"],
        "refined_fitness": best["fitness"],
        "inliers": best["inliers"],
    }


def _scene_nn_score(
    moving,
    fixed,
    *,
    max_points,
    trim_fraction,
    max_correspondence,
    min_inliers,
):
    moving = moving[torch.isfinite(moving).all(dim=-1)].float()
    fixed = fixed[torch.isfinite(fixed).all(dim=-1)].float()
    moving = _deterministic_point_subsample(moving, max_points)
    fixed = _deterministic_point_subsample(fixed, max_points)
    if moving.shape[0] < int(min_inliers) or fixed.shape[0] < int(min_inliers):
        return {
            "valid": False,
            "rmse": float("nan"),
            "fitness": 0.0,
            "inliers": 0,
        }
    distances, _ = nearest_neighbors(moving, fixed)
    supported = torch.isfinite(distances) & (
        distances <= float(max_correspondence)
    )
    support_count = int(supported.sum())
    fitness = support_count / max(int(moving.shape[0]), 1)
    if support_count < int(min_inliers):
        return {
            "valid": False,
            "rmse": float("nan"),
            "fitness": fitness,
            "inliers": support_count,
        }
    supported_distances = distances[supported]
    threshold = torch.quantile(
        supported_distances,
        min(1.0, max(0.1, float(trim_fraction))),
    )
    inliers = supported & (distances <= threshold)
    inlier_count = int(inliers.sum())
    if inlier_count < int(min_inliers):
        return {
            "valid": False,
            "rmse": float("nan"),
            "fitness": fitness,
            "inliers": inlier_count,
        }
    rmse = float(torch.sqrt((distances[inliers] ** 2).mean()))
    return {
        "valid": True,
        "rmse": rmse,
        "fitness": fitness,
        "inliers": inlier_count,
    }


def _deterministic_point_subsample(points, max_points):
    if points.shape[0] <= int(max_points):
        return points
    indices = torch.linspace(
        0,
        points.shape[0] - 1,
        steps=int(max_points),
        device=points.device,
    ).long()
    return points[indices]


def _merge_object_map(
    existing: torch.Tensor,
    observation: torch.Tensor,
    *,
    voxel_size: float,
    max_points: int,
) -> torch.Tensor:
    existing = existing[torch.isfinite(existing).all(dim=-1)]
    observation = observation[torch.isfinite(observation).all(dim=-1)]
    if existing.shape[0] >= int(max_points):
        indices = torch.linspace(
            0,
            existing.shape[0] - 1,
            steps=int(max_points),
            device=existing.device,
        ).long()
        return existing[indices]
    if observation.numel() == 0:
        return existing
    if float(voxel_size) > 0.0:
        existing_keys = torch.floor(existing / float(voxel_size)).to(torch.int64)
        observation_keys = torch.floor(observation / float(voxel_size)).to(torch.int64)
        occupied = {tuple(key) for key in existing_keys.cpu().tolist()}
        new_indices = []
        for index, key in enumerate(observation_keys.cpu().tolist()):
            voxel = tuple(key)
            if voxel not in occupied:
                occupied.add(voxel)
                new_indices.append(index)
        if not new_indices:
            return existing
        observation = observation[
            torch.tensor(new_indices, dtype=torch.long, device=observation.device)
        ]
    remaining = int(max_points) - existing.shape[0]
    if observation.shape[0] > remaining:
        indices = torch.linspace(
            0,
            observation.shape[0] - 1,
            steps=remaining,
            device=observation.device,
        ).long()
        observation = observation[indices]
    return torch.cat((existing, observation), dim=0)


def _summarize_branch(source, rows, *, reference, recovery_triggered):
    visible = [
        row for row in rows if row["gt_visible"] and row["sequence_index"] != reference
    ]
    absent = [row for row in rows if not row["gt_visible"]]
    accepted_visible = [row for row in visible if row["icp_accepted"]]
    rejected_visible = [row for row in visible if not row["icp_accepted"]]
    summary = {
        "mask_source": source,
        "reference_sequence_index": reference,
        "visible_evaluation_frames": len(visible),
        "accepted_visible_icp_frames": len(accepted_visible),
        "accepted_absent_icp_frames": sum(row["icp_accepted"] for row in absent),
        "visible_icp_acceptance_rate": len(accepted_visible) / max(len(visible), 1),
        "accepted_visible_frame_indices": " ".join(
            str(row["frame_index"]) for row in accepted_visible
        ),
        "rejected_visible_frame_indices": " ".join(
            str(row["frame_index"]) for row in rejected_visible
        ),
        "rejected_visible_icp_reasons": " | ".join(
            f"{row['frame_index']}:{row['icp_reason']}" for row in rejected_visible
        ),
        "mean_cross_view_mask_iou": _finite_mean(
            row["mask_iou"] for row in visible
        ),
        "mean_selected_contamination_ratio": _finite_mean(
            row["selected_contamination_ratio"] for row in visible
        ),
        "mean_visible_selected_points": _finite_mean(
            row["selected_points"] for row in visible
        ),
        "mean_accepted_visible_icp_inliers": _finite_mean(
            row["icp_inliers"] for row in accepted_visible
        ),
        "mean_accepted_visible_icp_fitness": _finite_mean(
            row["icp_fitness"] for row in accepted_visible
        ),
        "mean_accepted_visible_icp_rmse": _finite_mean(
            row["icp_rmse"] for row in accepted_visible
        ),
        "mean_accepted_visible_icp_rotation_degrees": _finite_mean(
            row["icp_rotation_degrees"] for row in accepted_visible
        ),
        "mean_accepted_visible_icp_translation": _finite_mean(
            row["icp_translation"] for row in accepted_visible
        ),
        "mean_accepted_visible_applied_icp_translation": _finite_mean(
            row["applied_icp_translation"] for row in accepted_visible
        ),
        "mean_visible_camera_delta_norm": _finite_mean(
            row["camera_delta_norm"] for row in visible
        ),
        "mean_visible_applied_delta_scale": _finite_mean(
            row["applied_delta_scale"] for row in visible
        ),
        "scene_guard_damped_visible_frames": sum(
            row["icp_accepted"]
            and row["applied_delta_scale"] < row["delta_scale"]
            for row in visible
        ),
        "mean_visible_scene_raw_rmse": _finite_mean(
            row["scene_raw_rmse"] for row in visible
        ),
        "mean_visible_scene_refined_rmse": _finite_mean(
            row["scene_refined_rmse"] for row in visible
        ),
        "final_scene_map_points": rows[-1]["scene_map_points"],
        "object_map_updates": sum(row["object_map_updated"] for row in rows),
        "final_object_map_points": rows[-1]["object_map_points_after"],
        "hard_recovery_triggered": int(recovery_triggered),
    }
    for key in (
        "raw_pose_rotation_degrees",
        "refined_pose_rotation_degrees",
        "raw_pose_translation",
        "refined_pose_translation",
        "raw_full_point_rmse",
        "refined_full_point_rmse",
        "raw_object_point_rmse",
        "refined_object_point_rmse",
        "raw_object_chamfer",
        "refined_object_chamfer",
    ):
        summary[f"mean_{key}"] = _finite_mean(row[key] for row in visible)
    for metric in (
        "pose_rotation_degrees",
        "pose_translation",
        "full_point_rmse",
        "object_point_rmse",
        "object_chamfer",
    ):
        summary[f"mean_{metric}_improvement"] = (
            summary[f"mean_raw_{metric}"] - summary[f"mean_refined_{metric}"]
        )
    summary["raw_ate_rmse"] = _finite_rmse(
        row["raw_pose_translation"] for row in rows
    )
    summary["refined_ate_rmse"] = _finite_rmse(
        row["refined_pose_translation"] for row in rows
    )
    summary["ate_rmse_improvement"] = (
        summary["raw_ate_rmse"] - summary["refined_ate_rmse"]
    )
    summary["ate_alignment"] = "fixed_reference_sim3"
    return summary


def _finite_rmse(values) -> float:
    array = np.asarray(
        [float(value) for value in values if np.isfinite(float(value))],
        dtype=np.float64,
    )
    return float(np.sqrt(np.mean(array**2))) if array.size else float("nan")


def _export_pointmaps(
    output_dir,
    *,
    raw_points,
    gt_points,
    gt_masks,
    colors,
    confidence,
    confidence_threshold,
    masks,
    branch_sources,
    branch_outputs,
):
    root = output_dir / "pointmaps"
    root.mkdir(parents=True, exist_ok=True)
    for stale_ply in root.glob("*.ply"):
        stale_ply.unlink()
    save_aggregate_ply(
        root / "scene_raw.ply",
        raw_points,
        colors,
        confidence=confidence,
        confidence_threshold=confidence_threshold,
    )
    save_aggregate_ply(root / "scene_gt.ply", gt_points, colors)
    save_aggregate_ply(
        root / "object_gt.ply",
        gt_points,
        colors,
        masks=gt_masks,
    )
    save_aggregate_ply(
        root / "object_raw.ply",
        raw_points,
        colors,
        masks=masks["sam3_hard_memory"],
        confidence=confidence,
        confidence_threshold=confidence_threshold,
    )
    for branch_name, (refined_points, _) in branch_outputs.items():
        source = branch_sources[branch_name]
        if source != "sam3_hard_memory":
            continue
        save_aggregate_ply(
            root / f"scene_{branch_name}.ply",
            refined_points,
            colors,
            confidence=confidence,
            confidence_threshold=confidence_threshold,
        )
        save_aggregate_ply(
            root / f"object_{branch_name}.ply",
            refined_points,
            colors,
            masks=masks[source],
            confidence=confidence,
            confidence_threshold=confidence_threshold,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--scene-id")
    parser.add_argument("--instance-id", type=int)
    parser.add_argument("--frame-indices", type=int, nargs="+")
    parser.add_argument("--sam3-device")
    parser.add_argument("--geometry-device")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--geometry-source",
        choices=("point_head", "depth_camera"),
        default="point_head",
        help="StreamVGGT pointmap used for Sim3 alignment and instance ICP.",
    )
    parser.add_argument(
        "--alignment-confidence-threshold",
        type=float,
        default=0.30,
        help="StreamVGGT confidence threshold used only for reference Sim3.",
    )
    parser.add_argument(
        "--icp-confidence-threshold",
        type=float,
        default=0.30,
        help="StreamVGGT confidence threshold used for reference/current ICP points.",
    )
    parser.add_argument("--alignment-trim-fraction", type=float, default=0.70)
    parser.add_argument("--icp-max-points", type=int, default=2048)
    parser.add_argument("--icp-iterations", type=int, default=30)
    parser.add_argument("--icp-trim-fraction", type=float, default=0.70)
    parser.add_argument("--icp-max-correspondence", type=float, default=0.20)
    parser.add_argument("--icp-min-inliers", type=int, default=64)
    parser.add_argument("--icp-min-fitness", type=float, default=0.10)
    parser.add_argument("--icp-max-rmse", type=float, default=0.15)
    parser.add_argument(
        "--pose-refinement-mode",
        choices=("translation_only", "full_se3"),
        default="translation_only",
    )
    parser.add_argument("--reference-sequence-index", type=int)
    parser.add_argument(
        "--delta-scales",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 1.0],
        help="Fractions of the accepted translation correction to apply.",
    )
    parser.add_argument(
        "--object-map-modes",
        choices=("reference_only", "causal"),
        nargs="+",
        default=["reference_only", "causal"],
        help="Compare a fixed reference cloud with a causally updated object map.",
    )
    parser.add_argument("--object-map-voxel-size", type=float, default=0.02)
    parser.add_argument("--object-map-max-points", type=int, default=16384)
    parser.add_argument(
        "--scene-consistency",
        choices=("off", "guard"),
        default="off",
        help="Damp object ICP when it degrades causal non-object scene overlap.",
    )
    parser.add_argument(
        "--scene-candidate-scales",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    parser.add_argument("--scene-confidence-threshold", type=float, default=0.30)
    parser.add_argument("--scene-rmse-tolerance", type=float, default=0.05)
    parser.add_argument(
        "--scene-fitness-drop-tolerance",
        type=float,
        default=0.05,
    )
    parser.add_argument("--scene-min-inliers", type=int, default=128)
    parser.add_argument("--scene-map-voxel-size", type=float, default=0.05)
    parser.add_argument("--scene-map-max-points", type=int, default=32768)
    return parser.parse_args()


def _validate_delta_scales(values, pose_refinement_mode):
    scales = []
    for value in values:
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError("Every delta scale must be in [0, 1].")
        if value not in scales:
            scales.append(value)
    if not scales:
        raise ValueError("At least one delta scale is required.")
    if pose_refinement_mode != "translation_only" and scales != [1.0]:
        raise ValueError(
            "Delta damping currently supports translation_only. "
            "Use --delta-scales 1 with full_se3."
        )
    return scales


def _validate_object_map_modes(values):
    modes = []
    for value in values:
        if value not in {"reference_only", "causal"}:
            raise ValueError(f"Unsupported object map mode {value!r}.")
        if value not in modes:
            modes.append(value)
    if not modes:
        raise ValueError("At least one object map mode is required.")
    return modes


def _validate_scene_candidate_scales(values):
    scales = []
    for value in values:
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError("Every scene candidate scale must be in [0, 1].")
        if value not in scales:
            scales.append(value)
    for required in (0.0, 1.0):
        if required not in scales:
            scales.append(required)
    return sorted(scales)


def _format_scale(value):
    return f"{float(value):.2f}".replace(".", "p")


if __name__ == "__main__":
    main()
