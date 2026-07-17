"""Jointly refine one StreamVGGT frame from several persistent SAM3 instances."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from test_sam.coordinates import streamvggt_label_to_grid
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
    symmetric_chamfer,
)
from .gt_mask_pose_experiment import (
    _finite_mean,
    _plot_camera_trajectories,
    _write_csv,
    pointmap_errors,
    select_causal_reference,
)
from .pipeline import _resize_target_masks
from .sam3_mask_camera_refinement_experiment import (
    _format_scale,
    _merge_object_map,
    _run_hard_recovery,
    _select_geometry_source,
)
from .sam3_mask_object_fusion_experiment import (
    _reference_similarity,
    _sam_masks_to_stream,
)


INSTANCE_COLORS = (
    (220, 50, 47),
    (38, 139, 210),
    (133, 153, 0),
    (181, 137, 0),
    (108, 113, 196),
)


@dataclass(frozen=True)
class JointICPResult:
    translation: torch.Tensor
    accepted: bool
    reason: str
    participants: tuple[int, ...]
    inliers: int
    fitness: float
    rmse: float
    iterations: int
    per_instance: dict[int, dict]


def main() -> None:
    args = _parse_args()
    overrides = {
        key: value
        for key, value in {
            "manifest": args.manifest,
            "scene_id": args.scene_id,
            "instance_id": args.instance_ids[0],
            "frame_indices": args.frame_indices,
            "sam3_device": args.sam3_device,
            "geometry_device": args.geometry_device,
            "output_dir": args.output_dir,
        }.items()
        if value is not None
    }
    run_experiment(
        load_config(args.config, overrides),
        instance_ids=args.instance_ids,
        geometry_source=args.geometry_source,
        alignment_reference_sequence_index=(
            args.alignment_reference_sequence_index
        ),
        alignment_confidence_threshold=args.alignment_confidence_threshold,
        alignment_trim_fraction=args.alignment_trim_fraction,
        icp_confidence_threshold=args.icp_confidence_threshold,
        icp_max_points_per_instance=args.icp_max_points_per_instance,
        icp_iterations=args.icp_iterations,
        icp_trim_fraction=args.icp_trim_fraction,
        icp_max_correspondence=args.icp_max_correspondence,
        icp_min_inliers=args.icp_min_inliers,
        icp_min_fitness=args.icp_min_fitness,
        icp_max_rmse=args.icp_max_rmse,
        joint_min_instances=args.joint_min_instances,
        joint_max_instance_disagreement=args.joint_max_instance_disagreement,
        joint_max_translation=args.joint_max_translation,
        delta_scales=args.delta_scales,
        object_map_voxel_size=args.object_map_voxel_size,
        object_map_max_points=args.object_map_max_points,
    )


def run_experiment(
    config: ExperimentConfig,
    *,
    instance_ids: list[int],
    geometry_source: str,
    alignment_reference_sequence_index: int,
    alignment_confidence_threshold: float,
    alignment_trim_fraction: float,
    icp_confidence_threshold: float,
    icp_max_points_per_instance: int,
    icp_iterations: int,
    icp_trim_fraction: float,
    icp_max_correspondence: float,
    icp_min_inliers: int,
    icp_min_fitness: float,
    icp_max_rmse: float,
    joint_min_instances: int,
    joint_max_instance_disagreement: float,
    joint_max_translation: float,
    delta_scales: list[float],
    object_map_voxel_size: float,
    object_map_max_points: int,
) -> None:
    instance_ids = _unique_instance_ids(instance_ids)
    delta_scales = _validate_delta_scales(delta_scales)
    if not 1 <= int(joint_min_instances) <= len(instance_ids):
        raise ValueError("joint_min_instances must be in [1, num_instances].")
    torch.manual_seed(0)
    np.random.seed(0)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    sequences = {}
    target_output_masks = {}
    for instance_id in instance_ids:
        sequence = load_mask_tracking_sequence(
            config.manifest,
            scene_id=config.scene_id,
            frame_indices=config.frame_indices,
            sequence_length=len(config.frame_indices),
            frame_stride=1,
            window_index=0,
            instance_id=instance_id,
            min_pixels=config.min_pixels,
            max_area_ratio=config.max_area_ratio,
            min_visible_frames=1,
            excluded_labels=config.excluded_labels,
            seed=0,
        )
        reference = select_causal_reference(sequence, None)
        sequence = replace(sequence, reference_frame_idx=reference)
        sequences[instance_id] = sequence
        target_output_masks[instance_id] = _resize_target_masks(
            sequence.target_masks,
            config.output_size,
        )
    _validate_shared_sequence(sequences)
    shared = sequences[instance_ids[0]]
    if not 0 <= int(alignment_reference_sequence_index) < len(shared.frame_indices):
        raise ValueError("alignment_reference_sequence_index is outside the sequence.")
    print(
        f"joint target scene={shared.scene_id} frames={shared.frame_indices} "
        f"instances={[(iid, sequences[iid].label) for iid in instance_ids]} "
        f"references={[(iid, sequences[iid].reference_frame_idx) for iid in instance_ids]}"
    )

    print("loading frozen SAM3 once and tracking each persistent instance...")
    sam3 = SAM3Wrapper(
        repo_path=config.sam3_repo,
        checkpoint_path=config.sam3_checkpoint,
        device=config.sam3_device,
        output_threshold=config.sam3_output_threshold,
        prompt_with_box=config.prompt_with_box,
    ).load()
    original_tracking = {}
    for instance_id in instance_ids:
        sequence = sequences[instance_id]
        original_tracking[instance_id] = sam3.track(
            sequence.image_paths,
            prompt=sequence.label,
            output_size=config.output_size,
            reference_frame_idx=sequence.reference_frame_idx,
            reference_mask=target_output_masks[instance_id][
                sequence.reference_frame_idx
            ],
        )

    print("running frozen StreamVGGT once with causal caches...")
    geometry = StreamVGGTWrapper(
        repo_path=config.streamvggt_repo,
        checkpoint_path=config.streamvggt_checkpoint,
        device=config.geometry_device,
        image_mode=config.image_mode,
        streaming_cache=config.streaming_cache,
    ).load().extract(shared.image_paths)
    geometry_points, geometry_confidence = _select_geometry_source(
        geometry, geometry_source
    )

    gt = load_gt_geometry_sequence(
        config.manifest,
        scene_id=shared.scene_id,
        frame_indices=shared.frame_indices,
        instance_id=instance_ids[0],
        processed_size=geometry.processed_size,
        image_mode=config.image_mode,
    )
    gt_masks = {
        instance_id: torch.from_numpy(
            np.stack(
                [
                    streamvggt_label_to_grid(
                        instance_labels,
                        geometry.processed_size,
                        mode=config.image_mode,
                    )
                    == int(instance_id)
                    for instance_labels in sequences[instance_id].instance_masks
                ]
            )
        )
        for instance_id in instance_ids
    }
    if not torch.equal(gt_masks[instance_ids[0]], gt.instance_masks):
        raise RuntimeError("GT mask conversion differs from GT geometry loading.")

    hard_tracking = {}
    recovery = {}
    for instance_id in instance_ids:
        hard_tracking[instance_id], recovery[instance_id] = _run_hard_recovery(
            replace(config, instance_id=instance_id),
            sequence=sequences[instance_id],
            target_output_masks=target_output_masks[instance_id],
            original_tracking=original_tracking[instance_id],
            geometry=geometry,
            sam3=sam3,
        )
    del sam3
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    alignment_index = int(alignment_reference_sequence_index)
    similarity = _reference_similarity(
        geometry_points[alignment_index],
        geometry_confidence[alignment_index],
        gt.pointmaps[alignment_index],
        confidence_threshold=alignment_confidence_threshold,
        trim_fraction=alignment_trim_fraction,
    )
    raw_points = apply_similarity(
        geometry_points,
        similarity.scale,
        similarity.rotation,
        similarity.translation,
    )
    raw_poses = torch.stack(
        [align_world_to_camera(pose, similarity) for pose in geometry.world_to_camera]
    )
    print(
        f"{geometry_source} reference Sim3 frame={shared.frame_indices[alignment_index]} "
        f"scale={similarity.scale:.6f} inliers={similarity.inliers} "
        f"rmse={similarity.rmse:.6f}"
    )

    original_masks = {}
    hard_masks = {}
    for instance_id in instance_ids:
        original_masks[instance_id] = _sam_masks_to_stream(
            original_tracking[instance_id].masks,
            geometry=geometry,
            image_mode=config.image_mode,
        )
        hard_masks[instance_id] = _sam_masks_to_stream(
            hard_tracking[instance_id].masks,
            geometry=geometry,
            image_mode=config.image_mode,
        )
        reference = sequences[instance_id].reference_frame_idx
        hard_masks[instance_id][reference] = gt_masks[instance_id][reference]

    summary_rows = []
    frame_rows = []
    instance_rows = []
    outputs = {}
    final_object_maps = None
    transforms = {}
    for delta_scale in delta_scales:
        branch_name = f"joint_alpha_{_format_scale(delta_scale)}"
        result = _run_joint_branch(
            instance_ids=instance_ids,
            references={
                instance_id: sequences[instance_id].reference_frame_idx
                for instance_id in instance_ids
            },
            masks=hard_masks,
            gt_masks=gt_masks,
            raw_points=raw_points,
            raw_poses=raw_poses,
            confidence=geometry_confidence,
            gt_points=gt.pointmaps,
            gt_poses=gt.world_to_camera,
            frame_indices=shared.frame_indices,
            delta_scale=delta_scale,
            object_map_voxel_size=object_map_voxel_size,
            object_map_max_points=object_map_max_points,
            icp_confidence_threshold=icp_confidence_threshold,
            icp_max_points_per_instance=icp_max_points_per_instance,
            icp_iterations=icp_iterations,
            icp_trim_fraction=icp_trim_fraction,
            icp_max_correspondence=icp_max_correspondence,
            icp_min_inliers=icp_min_inliers,
            icp_min_fitness=icp_min_fitness,
            icp_max_rmse=icp_max_rmse,
            joint_min_instances=joint_min_instances,
            joint_max_instance_disagreement=joint_max_instance_disagreement,
            joint_max_translation=joint_max_translation,
        )
        frame_rows.extend(result["frame_rows"])
        instance_rows.extend(result["instance_rows"])
        outputs[branch_name] = (result["refined_points"], result["refined_poses"])
        transforms[branch_name] = result["corrections"]
        if final_object_maps is None:
            final_object_maps = result["object_maps"]
        summary_rows.append(
            _summarize_joint_branch(
                result,
                config=config,
                instance_ids=instance_ids,
                labels={iid: sequences[iid].label for iid in instance_ids},
                geometry_source=geometry_source,
                delta_scale=delta_scale,
                similarity=similarity,
                joint_min_instances=joint_min_instances,
                joint_max_instance_disagreement=joint_max_instance_disagreement,
            )
        )
        _plot_camera_trajectories(
            config.output_dir / f"camera_trajectories_{branch_name}.png",
            frame_indices=shared.frame_indices,
            gt_world_to_camera=gt.world_to_camera,
            raw_world_to_camera=raw_poses,
            refined_world_to_camera=result["refined_poses"],
            rows=result["frame_rows"],
            title=f"joint instances={instance_ids}, alpha={delta_scale:g}",
            raw_translation_key="raw_pose_translation",
            refined_translation_key="refined_pose_translation",
            raw_rotation_key="raw_pose_rotation_degrees",
            refined_rotation_key="refined_pose_rotation_degrees",
        )

    _export_results(
        config.output_dir,
        instance_ids=instance_ids,
        labels={iid: sequences[iid].label for iid in instance_ids},
        raw_points=raw_points,
        gt_points=gt.pointmaps,
        colors=gt.colors,
        confidence=geometry_confidence,
        confidence_threshold=icp_confidence_threshold,
        hard_masks=hard_masks,
        gt_masks=gt_masks,
        outputs=outputs,
        final_object_maps=final_object_maps or {},
        references={iid: sequences[iid].reference_frame_idx for iid in instance_ids},
        frame_indices=shared.frame_indices,
        hard_tracking=hard_tracking,
        recovery=recovery,
    )
    _save_multi_mask_report(
        config.output_dir / "mask_sources.png",
        image_paths=shared.image_paths,
        frame_indices=shared.frame_indices,
        instance_ids=instance_ids,
        labels={iid: sequences[iid].label for iid in instance_ids},
        gt_masks=target_output_masks,
        original_masks={iid: original_tracking[iid].masks for iid in instance_ids},
        hard_masks={iid: hard_tracking[iid].masks for iid in instance_ids},
        output_size=config.output_size,
    )
    _write_csv(config.output_dir / "summary.csv", summary_rows)
    _write_csv(config.output_dir / "frame_metrics.csv", frame_rows)
    _write_csv(config.output_dir / "instance_metrics.csv", instance_rows)
    _write_csv(
        config.output_dir / "instance_summary.csv",
        _summarize_instances(instance_rows, instance_ids, sequences),
    )
    with (config.output_dir / "transforms.json").open("w", encoding="utf8") as handle:
        json.dump(
            {
                "settings": {
                    "instance_ids": instance_ids,
                    "labels": {str(iid): sequences[iid].label for iid in instance_ids},
                    "instance_reference_indices": {
                        str(iid): sequences[iid].reference_frame_idx
                        for iid in instance_ids
                    },
                    "shared_pose_delta": True,
                    "instance_balanced_correspondences": True,
                    "joint_min_instances": joint_min_instances,
                    "joint_max_instance_disagreement": (
                        joint_max_instance_disagreement
                    ),
                    "delta_scales": delta_scales,
                    "geometry_source": geometry_source,
                    "gt_usage": (
                        "reference prompts, evaluation Sim3, and metrics only"
                    ),
                },
                "similarity": {
                    "scale": similarity.scale,
                    "rotation": similarity.rotation.tolist(),
                    "translation": similarity.translation.tolist(),
                    "inliers": similarity.inliers,
                    "rmse": similarity.rmse,
                },
                "hard_recovery": {str(key): value for key, value in recovery.items()},
                "corrections": transforms,
            },
            handle,
            indent=2,
        )
    print(f"summary: {config.output_dir / 'summary.csv'}")
    print(f"instance summary: {config.output_dir / 'instance_summary.csv'}")
    print(f"semantic map: {config.output_dir / 'semantic_map'}")


def _run_joint_branch(
    *,
    instance_ids,
    references,
    masks,
    gt_masks,
    raw_points,
    raw_poses,
    confidence,
    gt_points,
    gt_poses,
    frame_indices,
    delta_scale,
    object_map_voxel_size,
    object_map_max_points,
    icp_confidence_threshold,
    icp_max_points_per_instance,
    icp_iterations,
    icp_trim_fraction,
    icp_max_correspondence,
    icp_min_inliers,
    icp_min_fitness,
    icp_max_rmse,
    joint_min_instances,
    joint_max_instance_disagreement,
    joint_max_translation,
):
    refined_points = raw_points.clone()
    refined_poses = raw_poses.clone()
    object_maps: dict[int, torch.Tensor] = {}
    frame_rows = []
    instance_rows = []
    corrections = []

    for sequence_index, frame_index in enumerate(frame_indices):
        observations = {}
        selected_masks = {}
        map_points_before = {
            iid: int(object_maps[iid].shape[0]) if iid in object_maps else 0
            for iid in instance_ids
        }
        for instance_id in instance_ids:
            selected = (
                masks[instance_id][sequence_index]
                & torch.isfinite(raw_points[sequence_index]).all(dim=-1)
                & (
                    confidence[sequence_index]
                    >= float(icp_confidence_threshold)
                )
            )
            selected_masks[instance_id] = selected
            if sequence_index <= references[instance_id]:
                continue
            if instance_id not in object_maps:
                continue
            moving = raw_points[sequence_index][selected]
            if moving.shape[0] < int(icp_min_inliers):
                continue
            observations[instance_id] = {
                "moving": moving,
                "fixed": object_maps[instance_id],
                "weights": confidence[sequence_index][selected],
            }

        joint = _joint_translation_icp(
            observations,
            max_points_per_instance=icp_max_points_per_instance,
            iterations=icp_iterations,
            trim_fraction=icp_trim_fraction,
            max_correspondence=icp_max_correspondence,
            min_inliers=icp_min_inliers,
            min_fitness=icp_min_fitness,
            max_rmse=icp_max_rmse,
            min_instances=joint_min_instances,
            max_instance_disagreement=joint_max_instance_disagreement,
            max_translation=joint_max_translation,
            dtype=raw_points.dtype,
            device=raw_points.device,
        )
        applied_translation = (
            joint.translation * float(delta_scale)
            if joint.accepted
            else torch.zeros_like(joint.translation)
        )
        if joint.accepted and float(delta_scale) > 0.0:
            identity = torch.eye(3, dtype=raw_points.dtype, device=raw_points.device)
            refined_points[sequence_index] = apply_rigid(
                raw_points[sequence_index], identity, applied_translation
            )
            refined_poses[sequence_index] = apply_world_correction_to_pose(
                raw_poses[sequence_index], identity, applied_translation
            )

        full_map_translation = (
            joint.translation if joint.accepted else torch.zeros_like(joint.translation)
        )
        for instance_id in joint.participants:
            observation = observations[instance_id]["moving"] + full_map_translation
            object_maps[instance_id] = _merge_object_map(
                object_maps[instance_id],
                observation,
                voxel_size=object_map_voxel_size,
                max_points=object_map_max_points,
            )

        # An object is born only on its first GT-prompt frame. If other objects
        # already estimate a shared delta on this frame, initialize it in that
        # fully corrected internal map frame, independent of output alpha.
        for instance_id in instance_ids:
            if sequence_index != references[instance_id]:
                continue
            reference_valid = (
                gt_masks[instance_id][sequence_index]
                & torch.isfinite(raw_points[sequence_index]).all(dim=-1)
                & (
                    confidence[sequence_index]
                    >= float(icp_confidence_threshold)
                )
            )
            reference_points = (
                raw_points[sequence_index][reference_valid] + full_map_translation
            )
            if reference_points.shape[0] < int(icp_min_inliers):
                raise RuntimeError(
                    f"Instance {instance_id} reference frame {frame_index} has only "
                    f"{reference_points.shape[0]} confident points."
                )
            object_maps[instance_id] = reference_points.clone()

        raw_rotation, raw_translation = pose_errors(
            raw_poses[sequence_index], gt_poses[sequence_index]
        )
        refined_rotation, refined_translation = pose_errors(
            refined_poses[sequence_index], gt_poses[sequence_index]
        )
        raw_full = pointmap_errors(raw_points[sequence_index], gt_points[sequence_index])
        refined_full = pointmap_errors(
            refined_points[sequence_index], gt_points[sequence_index]
        )
        frame_rows.append(
            {
                "delta_scale": float(delta_scale),
                "sequence_index": sequence_index,
                "frame_index": frame_index,
                "candidate_instances": " ".join(map(str, observations)),
                "candidate_instance_count": len(observations),
                "participating_instances": " ".join(map(str, joint.participants)),
                "participating_instance_count": len(joint.participants),
                "joint_icp_accepted": int(joint.accepted),
                "joint_icp_reason": joint.reason,
                "joint_icp_inliers": joint.inliers,
                "joint_icp_fitness": joint.fitness,
                "joint_icp_rmse": joint.rmse,
                "joint_icp_iterations": joint.iterations,
                "joint_translation_x": float(joint.translation[0]),
                "joint_translation_y": float(joint.translation[1]),
                "joint_translation_z": float(joint.translation[2]),
                "joint_translation_norm": float(
                    torch.linalg.vector_norm(joint.translation)
                ),
                "applied_translation_norm": float(
                    torch.linalg.vector_norm(applied_translation)
                ),
                "raw_pose_rotation_degrees": raw_rotation,
                "refined_pose_rotation_degrees": refined_rotation,
                "raw_pose_translation": raw_translation,
                "refined_pose_translation": refined_translation,
                "raw_full_point_rmse": raw_full["rmse"],
                "refined_full_point_rmse": refined_full["rmse"],
                "full_point_rmse_improvement": (
                    raw_full["rmse"] - refined_full["rmse"]
                ),
            }
        )

        for instance_id in instance_ids:
            gt_mask = gt_masks[instance_id][sequence_index]
            selected = selected_masks[instance_id]
            raw_object = pointmap_errors(
                raw_points[sequence_index], gt_points[sequence_index], mask=gt_mask
            )
            refined_object = pointmap_errors(
                refined_points[sequence_index], gt_points[sequence_index], mask=gt_mask
            )
            target_cloud = gt_points[sequence_index][gt_mask]
            raw_cloud = raw_points[sequence_index][gt_mask]
            refined_cloud = refined_points[sequence_index][gt_mask]
            detail = joint.per_instance.get(instance_id, {})
            if sequence_index < references[instance_id]:
                reason = "before instance reference"
            elif sequence_index == references[instance_id]:
                reason = "instance reference initialized"
            elif instance_id not in observations:
                reason = "too few mask-selected points"
            else:
                reason = detail.get("reason", "not selected by joint consensus")
            instance_rows.append(
                {
                    "delta_scale": float(delta_scale),
                    "sequence_index": sequence_index,
                    "frame_index": frame_index,
                    "instance_id": instance_id,
                    "gt_visible": int(gt_mask.any()),
                    "is_instance_reference": int(
                        sequence_index == references[instance_id]
                    ),
                    "mask_iou": binary_iou(masks[instance_id][sequence_index], gt_mask),
                    "mask_pixels": int(masks[instance_id][sequence_index].sum()),
                    "selected_points": int(selected.sum()),
                    "selected_contamination_ratio": float(
                        (selected & ~gt_mask).sum()
                    )
                    / max(int(selected.sum()), 1),
                    "participated_in_joint_icp": int(
                        instance_id in joint.participants
                    ),
                    "instance_gate_reason": reason,
                    "instance_icp_inliers": detail.get("inliers", 0),
                    "instance_icp_fitness": detail.get("fitness", 0.0),
                    "instance_icp_rmse": detail.get("rmse", float("nan")),
                    "instance_delta_disagreement": detail.get(
                        "delta_disagreement", float("nan")
                    ),
                    "object_map_points_before": map_points_before[instance_id],
                    "object_map_points_after": (
                        int(object_maps[instance_id].shape[0])
                        if instance_id in object_maps
                        else 0
                    ),
                    "raw_object_point_rmse": raw_object["rmse"],
                    "refined_object_point_rmse": refined_object["rmse"],
                    "raw_object_chamfer": symmetric_chamfer(raw_cloud, target_cloud),
                    "refined_object_chamfer": symmetric_chamfer(
                        refined_cloud, target_cloud
                    ),
                }
            )

        corrections.append(
            {
                "sequence_index": sequence_index,
                "frame_index": frame_index,
                "accepted": joint.accepted,
                "reason": joint.reason,
                "participants": list(joint.participants),
                "estimated_translation": joint.translation.tolist(),
                "delta_scale": float(delta_scale),
                "applied_translation": applied_translation.tolist(),
            }
        )
        print(
            f"joint alpha={float(delta_scale):.2f} frame={frame_index} "
            f"candidates={list(observations)} participants={list(joint.participants)} "
            f"accepted={joint.accepted} full_rmse={raw_full['rmse']:.4f}->"
            f"{refined_full['rmse']:.4f}"
        )

    return {
        "frame_rows": frame_rows,
        "instance_rows": instance_rows,
        "refined_points": refined_points,
        "refined_poses": refined_poses,
        "object_maps": object_maps,
        "corrections": corrections,
    }


def _joint_translation_icp(
    observations,
    *,
    max_points_per_instance,
    iterations,
    trim_fraction,
    max_correspondence,
    min_inliers,
    min_fitness,
    max_rmse,
    min_instances,
    max_instance_disagreement,
    max_translation,
    dtype,
    device,
):
    zero = torch.zeros(3, dtype=dtype, device=device)
    prepared = {
        instance_id: _prepare_observation(
            observation, max_points=max_points_per_instance
        )
        for instance_id, observation in observations.items()
    }
    if len(prepared) < int(min_instances):
        return JointICPResult(
            zero, False, "too few candidate instances", tuple(), 0, 0.0,
            float("nan"), 0, {}
        )

    translation = zero.clone()
    participants: tuple[int, ...] = tuple()
    completed_iterations = 0
    details = {}
    for iteration in range(max(1, int(iterations))):
        completed_iterations = iteration + 1
        supports = {
            instance_id: _instance_support(
                observation,
                translation=translation,
                trim_fraction=trim_fraction,
                max_correspondence=max_correspondence,
                min_inliers=min_inliers,
                min_fitness=min_fitness,
                max_rmse=max_rmse,
            )
            for instance_id, observation in prepared.items()
        }
        passing = {key: value for key, value in supports.items() if value["passed"]}
        participants, disagreement = _translation_consensus(
            passing, max_disagreement=max_instance_disagreement
        )
        for instance_id, support in supports.items():
            support["delta_disagreement"] = disagreement.get(
                instance_id, float("nan")
            )
        details = supports
        if len(participants) < int(min_instances):
            break
        object_deltas = torch.stack(
            [passing[instance_id]["suggested_delta"] for instance_id in participants]
        )
        delta = object_deltas.mean(dim=0)
        translation = translation + delta
        if float(torch.linalg.vector_norm(delta)) < 1e-5:
            break

    final_supports = {
        instance_id: _instance_support(
            observation,
            translation=translation,
            trim_fraction=trim_fraction,
            max_correspondence=max_correspondence,
            min_inliers=min_inliers,
            min_fitness=min_fitness,
            max_rmse=max_rmse,
        )
        for instance_id, observation in prepared.items()
    }
    final_passing = {
        key: value for key, value in final_supports.items() if value["passed"]
    }
    participants, disagreement = _translation_consensus(
        final_passing, max_disagreement=max_instance_disagreement
    )
    for instance_id, support in final_supports.items():
        support["delta_disagreement"] = disagreement.get(
            instance_id, float("nan")
        )
    details = final_supports
    translation_norm = float(torch.linalg.vector_norm(translation))
    checks = (
        (len(participants) >= int(min_instances), "too few consistent instances"),
        (translation_norm <= float(max_translation), "joint translation too large"),
    )
    reason = next((message for passed, message in checks if not passed), "accepted")
    accepted = reason == "accepted"
    if not accepted:
        participants = tuple()
    selected = [details[instance_id] for instance_id in participants]
    compact_details = {
        instance_id: {
            **{
                key: value
                for key, value in support.items()
                if key not in {"suggested_delta"}
            },
            "reason": (
                "accepted"
                if instance_id in participants
                else (
                    "instance translation disagrees with joint consensus"
                    if support["passed"]
                    else support["reason"]
                )
            ),
        }
        for instance_id, support in details.items()
    }
    return JointICPResult(
        translation=translation.detach(),
        accepted=accepted,
        reason=reason,
        participants=participants,
        inliers=sum(int(item["inliers"]) for item in selected),
        fitness=_finite_mean(item["fitness"] for item in selected),
        rmse=_finite_mean(item["rmse"] for item in selected),
        iterations=completed_iterations,
        per_instance=compact_details,
    )


def _prepare_observation(observation, *, max_points):
    moving = observation["moving"]
    fixed = observation["fixed"]
    weights = observation["weights"]
    moving, weights = _paired_subsample(moving, weights, max_points)
    fixed = _point_subsample(fixed, max_points)
    return {"moving": moving, "fixed": fixed, "weights": weights}


def _instance_support(
    observation,
    *,
    translation,
    trim_fraction,
    max_correspondence,
    min_inliers,
    min_fitness,
    max_rmse,
):
    moving = observation["moving"]
    fixed = observation["fixed"]
    transformed = moving + translation
    distances, indices = nearest_neighbors(transformed, fixed)
    supported = torch.isfinite(distances) & (
        distances <= float(max_correspondence)
    )
    support_count = int(supported.sum())
    fitness = support_count / max(int(moving.shape[0]), 1)
    rmse = (
        float(torch.sqrt((distances[supported] ** 2).mean()))
        if support_count
        else float("inf")
    )
    if support_count:
        threshold = torch.quantile(
            distances[supported], min(1.0, max(0.1, float(trim_fraction)))
        )
        inliers = supported & (distances <= threshold)
    else:
        inliers = supported
    trimmed_count = int(inliers.sum())
    checks = (
        (support_count >= int(min_inliers), "too few ICP inliers"),
        (trimmed_count >= int(min_inliers), "too few trimmed ICP inliers"),
        (fitness >= float(min_fitness), "ICP fitness below threshold"),
        (rmse <= float(max_rmse), "ICP RMSE above threshold"),
    )
    reason = next((message for passed, message in checks if not passed), "accepted")
    if trimmed_count:
        weights = observation["weights"][inliers].clamp_min(1e-6)
        weights = weights / weights.sum()
        residual = fixed[indices[inliers]] - transformed[inliers]
        suggested_delta = (residual * weights[:, None]).sum(dim=0)
    else:
        suggested_delta = torch.zeros_like(translation)
    return {
        "passed": reason == "accepted",
        "reason": reason,
        "inliers": support_count,
        "trimmed_inliers": trimmed_count,
        "fitness": float(fitness),
        "rmse": float(rmse),
        "suggested_delta": suggested_delta,
    }


def _translation_consensus(supports, *, max_disagreement):
    if not supports:
        return tuple(), {}
    ids = list(supports)
    deltas = torch.stack([supports[instance_id]["suggested_delta"] for instance_id in ids])
    center = deltas.mean(dim=0) if len(ids) == 2 else deltas.median(dim=0).values
    distances = torch.linalg.vector_norm(deltas - center, dim=-1)
    disagreement = {
        instance_id: float(distances[index])
        for index, instance_id in enumerate(ids)
    }
    participants = tuple(
        instance_id
        for instance_id in ids
        if disagreement[instance_id] <= float(max_disagreement)
    )
    return participants, disagreement


def _paired_subsample(points, weights, max_points):
    if points.shape[0] <= int(max_points):
        return points, weights
    indices = torch.linspace(
        0, points.shape[0] - 1, int(max_points), device=points.device
    ).long()
    return points[indices], weights[indices]


def _point_subsample(points, max_points):
    if points.shape[0] <= int(max_points):
        return points
    indices = torch.linspace(
        0, points.shape[0] - 1, int(max_points), device=points.device
    ).long()
    return points[indices]


def _summarize_joint_branch(
    result,
    *,
    config,
    instance_ids,
    labels,
    geometry_source,
    delta_scale,
    similarity,
    joint_min_instances,
    joint_max_instance_disagreement,
):
    rows = result["frame_rows"]
    accepted = [row for row in rows if row["joint_icp_accepted"]]
    raw_ate = _rmse(row["raw_pose_translation"] for row in rows)
    refined_ate = _rmse(row["refined_pose_translation"] for row in rows)
    summary = {
        "experiment_name": config.output_dir.name,
        "scene_id": config.scene_id,
        "instance_ids": " ".join(map(str, instance_ids)),
        "instance_labels": " | ".join(labels[iid] for iid in instance_ids),
        "frame_indices": " ".join(map(str, config.frame_indices)),
        "geometry_source": geometry_source,
        "delta_scale": float(delta_scale),
        "joint_min_instances": int(joint_min_instances),
        "joint_max_instance_disagreement": float(joint_max_instance_disagreement),
        "reference_sim3_scale": similarity.scale,
        "reference_sim3_inliers": similarity.inliers,
        "reference_sim3_rmse": similarity.rmse,
        "evaluated_frames": len(rows),
        "accepted_joint_frames": len(accepted),
        "joint_acceptance_rate": len(accepted) / max(len(rows), 1),
        "accepted_frame_indices": " ".join(
            str(row["frame_index"]) for row in accepted
        ),
        "mean_participating_instances": _finite_mean(
            row["participating_instance_count"] for row in accepted
        ),
        "mean_joint_icp_fitness": _finite_mean(
            row["joint_icp_fitness"] for row in accepted
        ),
        "mean_joint_icp_rmse": _finite_mean(
            row["joint_icp_rmse"] for row in accepted
        ),
        "mean_raw_pose_translation": _finite_mean(
            row["raw_pose_translation"] for row in rows
        ),
        "mean_refined_pose_translation": _finite_mean(
            row["refined_pose_translation"] for row in rows
        ),
        "mean_raw_full_point_rmse": _finite_mean(
            row["raw_full_point_rmse"] for row in rows
        ),
        "mean_refined_full_point_rmse": _finite_mean(
            row["refined_full_point_rmse"] for row in rows
        ),
        "mean_full_point_rmse_improvement": _finite_mean(
            row["full_point_rmse_improvement"] for row in rows
        ),
        "raw_ate_rmse": raw_ate,
        "refined_ate_rmse": refined_ate,
        "ate_rmse_improvement": raw_ate - refined_ate,
        "ate_alignment": "fixed_reference_sim3",
    }
    summary["mean_pose_translation_improvement"] = (
        summary["mean_raw_pose_translation"]
        - summary["mean_refined_pose_translation"]
    )
    return summary


def _summarize_instances(rows, instance_ids, sequences):
    summaries = []
    scales = sorted({float(row["delta_scale"]) for row in rows})
    for delta_scale in scales:
        for instance_id in instance_ids:
            reference = sequences[instance_id].reference_frame_idx
            selected = [
                row
                for row in rows
                if row["instance_id"] == instance_id
                and float(row["delta_scale"]) == delta_scale
                and row["gt_visible"]
                and row["sequence_index"] != reference
            ]
            raw_rmse = _finite_mean(
                row["raw_object_point_rmse"] for row in selected
            )
            refined_rmse = _finite_mean(
                row["refined_object_point_rmse"] for row in selected
            )
            raw_chamfer = _finite_mean(
                row["raw_object_chamfer"] for row in selected
            )
            refined_chamfer = _finite_mean(
                row["refined_object_chamfer"] for row in selected
            )
            summaries.append(
                {
                    "delta_scale": delta_scale,
                    "instance_id": instance_id,
                    "instance_label": sequences[instance_id].label,
                    "reference_sequence_index": reference,
                    "reference_frame_index": sequences[instance_id].frame_indices[
                        reference
                    ],
                    "visible_evaluation_frames": len(selected),
                    "participating_visible_frames": sum(
                        row["participated_in_joint_icp"] for row in selected
                    ),
                    "mean_mask_iou": _finite_mean(
                        row["mask_iou"] for row in selected
                    ),
                    "mean_contamination_ratio": _finite_mean(
                        row["selected_contamination_ratio"] for row in selected
                    ),
                    "mean_raw_object_point_rmse": raw_rmse,
                    "mean_refined_object_point_rmse": refined_rmse,
                    "mean_object_point_rmse_improvement": raw_rmse - refined_rmse,
                    "mean_raw_object_chamfer": raw_chamfer,
                    "mean_refined_object_chamfer": refined_chamfer,
                    "mean_object_chamfer_improvement": (
                        raw_chamfer - refined_chamfer
                    ),
                }
            )
    return summaries


def _export_results(
    output_dir,
    *,
    instance_ids,
    labels,
    raw_points,
    gt_points,
    colors,
    confidence,
    confidence_threshold,
    hard_masks,
    gt_masks,
    outputs,
    final_object_maps,
    references,
    frame_indices,
    hard_tracking,
    recovery,
):
    point_root = output_dir / "pointmaps"
    point_root.mkdir(parents=True, exist_ok=True)
    for path in point_root.glob("*.ply"):
        path.unlink()
    save_aggregate_ply(
        point_root / "scene_raw.ply",
        raw_points,
        colors,
        confidence=confidence,
        confidence_threshold=confidence_threshold,
    )
    save_aggregate_ply(point_root / "scene_gt.ply", gt_points, colors)
    for branch_name, (refined_points, _) in outputs.items():
        save_aggregate_ply(
            point_root / f"scene_{branch_name}.ply",
            refined_points,
            colors,
            confidence=confidence,
            confidence_threshold=confidence_threshold,
        )
        for instance_id in instance_ids:
            save_aggregate_ply(
                point_root / f"object_{instance_id}_{branch_name}.ply",
                refined_points,
                colors,
                masks=hard_masks[instance_id],
                confidence=confidence,
                confidence_threshold=confidence_threshold,
            )
    for instance_id in instance_ids:
        save_aggregate_ply(
            point_root / f"object_{instance_id}_raw.ply",
            raw_points,
            colors,
            masks=hard_masks[instance_id],
            confidence=confidence,
            confidence_threshold=confidence_threshold,
        )
        save_aggregate_ply(
            point_root / f"object_{instance_id}_gt.ply",
            gt_points,
            colors,
            masks=gt_masks[instance_id],
        )

    semantic_root = output_dir / "semantic_map"
    semantic_root.mkdir(parents=True, exist_ok=True)
    for path in semantic_root.glob("object_*.ply"):
        path.unlink()
    registry = []
    for color_index, instance_id in enumerate(instance_ids):
        points = final_object_maps.get(instance_id, torch.empty((0, 3)))
        color = np.asarray(
            INSTANCE_COLORS[color_index % len(INSTANCE_COLORS)], dtype=np.uint8
        )
        point_colors = np.broadcast_to(color, (points.shape[0], 3)).copy()
        save_aggregate_ply(
            semantic_root / f"object_{instance_id}.ply", points, point_colors
        )
        registry.append(
            {
                "instance_id": instance_id,
                "semantic_label": labels[instance_id],
                "point_count": int(points.shape[0]),
                "reference_sequence_index": references[instance_id],
                "reference_frame_index": frame_indices[references[instance_id]],
                "sam3_persistent_obj_id": hard_tracking[
                    instance_id
                ].selected_obj_id,
                "hard_recovery": recovery[instance_id],
                "static": True,
            }
        )
    with (semantic_root / "object_registry.json").open("w", encoding="utf8") as handle:
        json.dump({"objects": registry}, handle, indent=2)


def _save_multi_mask_report(
    path,
    *,
    image_paths,
    frame_indices,
    instance_ids,
    labels,
    gt_masks,
    original_masks,
    hard_masks,
    output_size,
):
    height, width = output_size
    header = 34
    columns = 4
    canvas = Image.new(
        "RGB", (columns * width, len(image_paths) * (height + header)), "white"
    )
    draw = ImageDraw.Draw(canvas)
    column_names = ("RGB", "GT instances", "SAM3 original", "hard memory")
    for row, image_path in enumerate(image_paths):
        with Image.open(image_path) as image:
            rgb = image.convert("RGB").resize((width, height))
        panels = [rgb]
        for source in (gt_masks, original_masks, hard_masks):
            panel = rgb.copy()
            for color_index, instance_id in enumerate(instance_ids):
                panel = _overlay_mask(
                    panel,
                    source[instance_id][row],
                    INSTANCE_COLORS[color_index % len(INSTANCE_COLORS)],
                )
            panels.append(panel)
        top = row * (height + header)
        for column, panel in enumerate(panels):
            canvas.paste(panel, (column * width, top + header))
            draw.text(
                (column * width + 5, top + 5),
                f"{column_names[column]} | frame {frame_indices[row]}",
                fill="black",
            )
    legend = " | ".join(
        f"{instance_id}:{labels[instance_id]}" for instance_id in instance_ids
    )
    draw.text((5, 18), legend, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _overlay_mask(image, mask, color):
    mask_image = Image.fromarray(
        (mask.detach().cpu().numpy().astype(np.uint8) * 150), mode="L"
    ).resize(image.size, Image.Resampling.NEAREST)
    overlay = Image.new("RGB", image.size, color)
    return Image.composite(overlay, image, mask_image)


def _validate_shared_sequence(sequences):
    values = list(sequences.values())
    first = values[0]
    for sequence in values[1:]:
        if sequence.frame_indices != first.frame_indices:
            raise RuntimeError("All instances must use the same frame indices.")
        if [str(path) for path in sequence.image_paths] != [
            str(path) for path in first.image_paths
        ]:
            raise RuntimeError("All instances must use the same RGB sequence.")


def _unique_instance_ids(values):
    result = []
    for value in values:
        value = int(value)
        if value not in result:
            result.append(value)
    if len(result) < 2:
        raise ValueError("Joint refinement requires at least two instance IDs.")
    return result


def _validate_delta_scales(values):
    result = []
    for value in values:
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError("Every delta scale must be in [0, 1].")
        if value not in result:
            result.append(value)
    if not result:
        raise ValueError("At least one delta scale is required.")
    return result


def _rmse(values):
    array = np.asarray(
        [float(value) for value in values if np.isfinite(float(value))],
        dtype=np.float64,
    )
    return float(np.sqrt(np.mean(array**2))) if array.size else float("nan")


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--scene-id")
    parser.add_argument("--instance-ids", type=int, nargs="+", required=True)
    parser.add_argument("--frame-indices", type=int, nargs="+")
    parser.add_argument("--sam3-device")
    parser.add_argument("--geometry-device")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--geometry-source", choices=("point_head", "depth_camera"),
        default="point_head"
    )
    parser.add_argument("--alignment-reference-sequence-index", type=int, default=0)
    parser.add_argument("--alignment-confidence-threshold", type=float, default=0.30)
    parser.add_argument("--alignment-trim-fraction", type=float, default=0.70)
    parser.add_argument("--icp-confidence-threshold", type=float, default=0.30)
    parser.add_argument("--icp-max-points-per-instance", type=int, default=2048)
    parser.add_argument("--icp-iterations", type=int, default=30)
    parser.add_argument("--icp-trim-fraction", type=float, default=0.70)
    parser.add_argument("--icp-max-correspondence", type=float, default=0.20)
    parser.add_argument("--icp-min-inliers", type=int, default=64)
    parser.add_argument("--icp-min-fitness", type=float, default=0.10)
    parser.add_argument("--icp-max-rmse", type=float, default=0.15)
    parser.add_argument("--joint-min-instances", type=int, default=2)
    parser.add_argument(
        "--joint-max-instance-disagreement", type=float, default=0.15
    )
    parser.add_argument("--joint-max-translation", type=float, default=1.0)
    parser.add_argument("--delta-scales", type=float, nargs="+", default=[1.0])
    parser.add_argument("--object-map-voxel-size", type=float, default=0.02)
    parser.add_argument("--object-map-max-points", type=int, default=16384)
    return parser.parse_args()


if __name__ == "__main__":
    main()
