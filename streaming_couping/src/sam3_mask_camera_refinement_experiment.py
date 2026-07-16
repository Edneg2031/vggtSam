"""Use SAM3 instance masks to refine StreamVGGT camera poses."""

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
from .geometry.export import save_aggregate_ply, save_pointmap_ply
from .geometry.gt_data import load_gt_geometry_sequence
from .geometry.registration import (
    align_world_to_camera,
    apply_rigid,
    apply_similarity,
    apply_world_correction_to_pose,
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


MASK_SOURCES = ("gt_oracle", "sam3_original", "sam3_hard_memory")


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
    )


def run_experiment(
    config: ExperimentConfig,
    *,
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
) -> None:
    delta_scales = _validate_delta_scales(delta_scales, pose_refinement_mode)
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
    if geometry.camera_world_points is None or geometry.depth_confidence is None:
        raise RuntimeError("StreamVGGT depth-camera pointmap is unavailable.")

    gt = load_gt_geometry_sequence(
        config.manifest,
        scene_id=sequence.scene_id,
        frame_indices=sequence.frame_indices,
        instance_id=sequence.instance_id,
        processed_size=geometry.processed_size,
        image_mode=config.image_mode,
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
        geometry.camera_world_points[reference],
        geometry.depth_confidence[reference],
        gt.pointmaps[reference],
        confidence_threshold=alignment_confidence_threshold,
        trim_fraction=alignment_trim_fraction,
    )
    aligned_points = apply_similarity(
        geometry.camera_world_points,
        similarity.scale,
        similarity.rotation,
        similarity.translation,
    )
    raw_poses = torch.stack(
        [align_world_to_camera(pose, similarity) for pose in geometry.world_to_camera]
    )
    print(
        f"depth-camera reference Sim3 scale={similarity.scale:.6f} "
        f"inliers={similarity.inliers} rmse={similarity.rmse:.6f}"
    )

    masks = {
        "gt_oracle": gt.instance_masks.clone(),
        "sam3_original": _sam_masks_to_stream(
            original_tracking.masks,
            geometry=geometry,
            image_mode=config.image_mode,
        ),
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
            geometry.depth_confidence[reference]
            >= float(icp_confidence_threshold)
        )
    )
    reference_points = aligned_points[reference][reference_valid]
    if reference_points.shape[0] < int(icp_min_inliers):
        raise RuntimeError(
            "Reference instance contains too few confident depth-camera points: "
            f"{reference_points.shape[0]}."
        )

    all_rows: list[dict] = []
    summary_rows: list[dict] = []
    branch_outputs = {}
    branch_sources = {}
    transforms = {}
    for source in MASK_SOURCES:
        for delta_scale in delta_scales:
            branch_name = f"{source}_alpha_{_format_scale(delta_scale)}"
            rows, refined_points, refined_poses, corrections = _run_pose_branch(
                source,
                delta_scale=delta_scale,
                masks=masks[source],
                reference=reference,
                reference_points=reference_points,
                raw_points=aligned_points,
                raw_poses=raw_poses,
                confidence=geometry.depth_confidence,
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
                "geometry_source": "depth_head_plus_camera_head",
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
            print(
                f"trajectory source={source:<18} alpha={delta_scale:.2f} "
                f"ATE_RMSE={summary['raw_ate_rmse']:.4f}->"
                f"{summary['refined_ate_rmse']:.4f} "
                f"improvement={summary['ate_rmse_improvement']:.4f}"
            )
            _plot_camera_trajectories(
                config.output_dir / f"camera_trajectories_{branch_name}.png",
                frame_indices=sequence.frame_indices,
                gt_world_to_camera=gt.world_to_camera,
                raw_world_to_camera=raw_poses,
                refined_world_to_camera=refined_poses,
                rows=rows,
                title=f"Depth-camera pose: {source}, alpha={delta_scale:g}",
                raw_translation_key="raw_pose_translation",
                refined_translation_key="refined_pose_translation",
                raw_rotation_key="raw_pose_rotation_degrees",
                refined_rotation_key="refined_pose_rotation_degrees",
            )

    _export_pointmaps(
        config.output_dir,
        frame_indices=sequence.frame_indices,
        raw_points=aligned_points,
        gt_points=gt.pointmaps,
        gt_masks=gt.instance_masks,
        colors=gt.colors,
        confidence=geometry.depth_confidence,
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
    )
    _write_csv(config.output_dir / "frame_metrics.csv", all_rows)
    _write_csv(config.output_dir / "summary.csv", summary_rows)
    with (config.output_dir / "transforms.json").open("w", encoding="utf8") as handle:
        json.dump(
            {
                "settings": {
                    "mask_sources": MASK_SOURCES,
                    "geometry_source": "depth_head_plus_camera_head",
                    "camera_delta_applied_to_full_frame": True,
                    "pose_refinement_mode": pose_refinement_mode,
                    "delta_scales": delta_scales,
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
        if sequence_index == reference:
            icp = _identity_icp(raw_points.dtype, reason="reference frame")
        elif moving.shape[0] < int(icp_min_inliers):
            icp = _identity_icp(raw_points.dtype, reason="too few mask-selected points")
        else:
            icp = robust_icp(
                moving,
                reference_points,
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
            if icp.accepted and float(delta_scale) > 0.0:
                applied_rotation = icp.rotation
                applied_translation = icp.translation * float(delta_scale)
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
                torch.linalg.vector_norm(icp.translation) * float(delta_scale)
                if icp.accepted
                else 0.0
            ),
            "correction_applied": int(
                icp.accepted and float(delta_scale) > 0.0
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
        if icp.accepted and float(delta_scale) > 0.0:
            correction[:3, :3] = icp.rotation
            correction[:3, 3] = icp.translation * float(delta_scale)
        corrections.append(
            {
                "sequence_index": sequence_index,
                "frame_index": frame_index,
                "accepted": icp.accepted,
                "delta_scale": float(delta_scale),
                "reason": icp.reason,
                "estimated_translation": icp.translation.tolist(),
                "applied": correction.tolist(),
            }
        )
        print(
            f"source={source:<18} alpha={float(delta_scale):.2f} "
            f"frame={frame_index} "
            f"visible={row['gt_visible']} mask_iou={row['mask_iou']:.4f} "
            f"icp={icp.accepted} pose_t={raw_translation:.4f}->"
            f"{refined_translation:.4f} full_rmse={raw_full['rmse']:.4f}->"
            f"{refined_full['rmse']:.4f}"
        )
    return rows, refined_points, refined_poses, corrections


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
    frame_indices,
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
    save_aggregate_ply(
        root / "sequence_depth_camera_raw.ply",
        raw_points,
        colors,
        confidence=confidence,
        confidence_threshold=confidence_threshold,
    )
    save_aggregate_ply(root / "sequence_gt.ply", gt_points, colors)
    for sequence_index, frame_index in enumerate(frame_indices):
        prefix = root / f"frame_{sequence_index:02d}_{frame_index}"
        save_pointmap_ply(
            prefix.with_name(prefix.name + "_depth_camera_raw.ply"),
            raw_points[sequence_index],
            colors[sequence_index],
            confidence=confidence[sequence_index],
            confidence_threshold=confidence_threshold,
        )
        save_pointmap_ply(
            prefix.with_name(prefix.name + "_gt.ply"),
            gt_points[sequence_index],
            colors[sequence_index],
        )
    for branch_name, (refined_points, _) in branch_outputs.items():
        source = branch_sources[branch_name]
        save_aggregate_ply(
            root / f"sequence_{branch_name}_refined.ply",
            refined_points,
            colors,
            confidence=confidence,
            confidence_threshold=confidence_threshold,
        )
        save_aggregate_ply(
            root / f"object_{branch_name}_selected_raw.ply",
            raw_points,
            colors,
            masks=masks[source],
            confidence=confidence,
            confidence_threshold=confidence_threshold,
        )
        save_aggregate_ply(
            root / f"object_{branch_name}_selected_refined.ply",
            refined_points,
            colors,
            masks=masks[source],
            confidence=confidence,
            confidence_threshold=confidence_threshold,
        )
        save_aggregate_ply(
            root / f"object_{branch_name}_gt_region_refined.ply",
            refined_points,
            colors,
            masks=gt_masks,
            confidence=confidence,
            confidence_threshold=confidence_threshold,
        )
        for sequence_index, frame_index in enumerate(frame_indices):
            prefix = root / f"frame_{sequence_index:02d}_{frame_index}_{branch_name}"
            save_pointmap_ply(
                prefix.with_name(prefix.name + "_refined.ply"),
                refined_points[sequence_index],
                colors[sequence_index],
                confidence=confidence[sequence_index],
                confidence_threshold=confidence_threshold,
            )
            save_pointmap_ply(
                prefix.with_name(prefix.name + "_selected_object.ply"),
                refined_points[sequence_index],
                colors[sequence_index],
                mask=masks[source][sequence_index],
                confidence=confidence[sequence_index],
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
        "--alignment-confidence-threshold",
        type=float,
        default=0.30,
        help="StreamVGGT confidence threshold used only for reference Sim3.",
    )
    parser.add_argument(
        "--icp-confidence-threshold",
        type=float,
        default=0.15,
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


def _format_scale(value):
    return f"{float(value):.2f}".replace(".", "p")


if __name__ == "__main__":
    main()
