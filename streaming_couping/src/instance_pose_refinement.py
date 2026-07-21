"""V3 causal static-instance pointmap and ray-center pose refinement.

The deployable branch validates instance proposals before object-map writeback.
Camera and pointmap correction always use exactly one frame-wide translation,
obtained from multi-instance consensus or bounded temporal conflict filtering.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from test_sam.data import load_mask_tracking_sequence

from .backbones.sam3_wrapper import SAM3Wrapper
from .backbones.streamvggt_wrapper import StreamVGGTWrapper
from .config import ExperimentConfig, load_config
from .pointmap_alignment import prepare_map_evaluation
from .instance_point_cloud import (
    export_instance_point_clouds,
    load_processed_colors,
)
from .pose_evaluation import (
    PoseSequence,
    RayFitConfig,
    SimilarityAlignment,
    _all_pair_pose_metrics,
    _evaluate_pose_alignment,
    _fit_ray_center,
    _load_ground_truth_sequence,
    _pointmap_frame_metrics,
    _pose_sequence_from_centers,
    _prepare_pose_sequence,
    _prepare_ray_inputs,
    _reference_pose_alignment,
    _summarize_pointmap_rows,
    _summarize_pose_pairs,
)
from .recovery import output_mask_to_stream, resize_target_masks
from .tracking_recovery import run_natural_recovery_tracking
from .types import GeometrySequence, TrackingSequence


@dataclass(frozen=True)
class InstanceRefinementConfig:
    min_instance_points: int = 128
    icp_max_points: int = 1024
    map_max_points: int = 4096
    icp_iterations: int = 4
    icp_trim_quantile: float = 0.70
    min_icp_fitness: float = 0.25
    max_icp_rmse: float = 0.03
    correspondence_min_distance: float = 0.02
    correspondence_object_ratio: float = 0.05
    max_proposal_translation: float = 0.15
    min_participating_instances: int = 2
    consensus_distance: float = 0.02
    temporal_max_frame_gap: int = 15
    temporal_proposal_distance: float = 0.02
    temporal_current_weight: float = 0.50
    temporal_max_carry_frames: int = 2
    compute_device: str = "cpu"
    ray_max_points: int = 65536
    ray_min_points: int = 1024
    ray_max_condition_number: float = 1e8


@dataclass(frozen=True)
class RefinementMode:
    name: str
    role: str
    mask_source: str
    map_policy: str
    correction_scale: float
    proposal_profile: str = "strict"
    map_update_policy: str = "none"
    temporal_policy: str = "none"


@dataclass(frozen=True)
class TranslationProposal:
    instance_id: int
    translation: torch.Tensor
    accepted: bool
    reason: str
    current_points: int
    map_points: int
    correspondences: int
    fitness: float
    rmse: float
    correspondence_distance: float
    object_scale: float
    iterations: int
    initialization: str


def main() -> None:
    args = _parse_args()
    overrides = {
        key: value
        for key, value in {
            "manifest": args.manifest,
            "scene_id": args.scene_id,
            "frame_indices": args.frame_indices,
            "sam3_device": args.sam3_device,
            "geometry_device": args.geometry_device,
            "output_dir": args.output_dir,
        }.items()
        if value is not None
    }
    config = load_config(args.config, overrides)
    refinement = InstanceRefinementConfig(
        min_instance_points=args.instance_min_points,
        icp_max_points=args.instance_icp_max_points,
        map_max_points=args.instance_map_max_points,
        icp_iterations=args.instance_icp_iterations,
        icp_trim_quantile=args.instance_icp_trim_quantile,
        min_icp_fitness=args.instance_min_icp_fitness,
        max_icp_rmse=args.instance_max_icp_rmse,
        correspondence_min_distance=args.instance_correspondence_min_distance,
        correspondence_object_ratio=args.instance_correspondence_object_ratio,
        max_proposal_translation=args.instance_max_translation,
        min_participating_instances=args.instance_min_participants,
        consensus_distance=args.instance_consensus_distance,
        temporal_max_frame_gap=args.instance_temporal_max_frame_gap,
        temporal_proposal_distance=args.instance_temporal_distance,
        temporal_current_weight=args.instance_temporal_current_weight,
        temporal_max_carry_frames=args.instance_temporal_max_carry_frames,
        compute_device=(
            args.instance_icp_device
            if args.instance_icp_device is not None
            else config.geometry_device
        ),
        ray_max_points=args.ray_max_points,
        ray_min_points=args.ray_min_points,
        ray_max_condition_number=args.ray_max_condition_number,
    )
    run_experiment(
        config,
        instance_ids=_unique_ids(args.instance_ids),
        reference_sequence_index=args.reference_sequence_index,
        refinement=refinement,
        tracking_cache_path=args.tracking_cache,
    )


def run_experiment(
    config: ExperimentConfig,
    *,
    instance_ids: Sequence[int],
    reference_sequence_index: int,
    refinement: InstanceRefinementConfig,
    tracking_cache_path: Path | None = None,
) -> None:
    """Run cached tracking and the final V3 instance-pose refinement."""

    _validate_refinement_config(refinement)
    if int(reference_sequence_index) != 0:
        raise ValueError(
            "Causal instance refinement requires the first selected frame "
            "to be the reference (reference_sequence_index=0)."
        )
    if refinement.min_participating_instances > len(instance_ids):
        raise ValueError(
            "instance-min-participants cannot exceed the number of instances."
        )
    if any(
        int(second) <= int(first)
        for first, second in zip(
            config.frame_indices,
            config.frame_indices[1:],
        )
    ):
        raise ValueError(
            "Causal temporal refinement requires strictly increasing "
            "frame indices."
        )
    torch.manual_seed(0)
    np.random.seed(0)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    sequences, target_masks = _load_instance_sequences(
        config,
        instance_ids=instance_ids,
        reference_sequence_index=reference_sequence_index,
    )
    shared = sequences[int(instance_ids[0])]
    print(
        "instance pose refinement "
        f"scene={shared.scene_id} frames={shared.frame_indices} "
        f"instances={list(instance_ids)}"
    )

    print("extracting frozen StreamVGGT geometry once...")
    geometry = StreamVGGTWrapper(
        repo_path=config.streamvggt_repo,
        checkpoint_path=config.streamvggt_checkpoint,
        device=config.geometry_device,
        image_mode=config.image_mode,
        streaming_cache=config.streaming_cache,
    ).load().extract(shared.image_paths)
    map_context = prepare_map_evaluation(
        config,
        scene_id=shared.scene_id,
        frame_indices=shared.frame_indices,
        geometry=geometry,
        reference_frame_idx=reference_sequence_index,
    )

    cache_path = (
        Path(tracking_cache_path)
        if tracking_cache_path is not None
        else config.output_dir / "tracking_cache.npz"
    )
    cached = _load_tracking_cache(
        cache_path,
        config=config,
        instance_ids=instance_ids,
        frame_indices=shared.frame_indices,
    )
    if cached is None:
        print("tracking cache missing; running minimal natural SAM3 recovery...")
        sam3 = SAM3Wrapper(
            repo_path=config.sam3_repo,
            checkpoint_path=config.sam3_checkpoint,
            device=config.sam3_device,
            output_threshold=config.sam3_output_threshold,
            prompt_with_box=config.prompt_with_box,
        ).load()
        original_tracking: dict[int, TrackingSequence] = {}
        recovered_tracking: dict[int, TrackingSequence] = {}
        tracking_rows = []
        for instance_id in instance_ids:
            result = run_natural_recovery_tracking(
                config,
                sequence=sequences[int(instance_id)],
                target_masks=target_masks[int(instance_id)],
                geometry=geometry,
                sam3=sam3,
            )
            original_tracking[int(instance_id)] = result["original"]
            recovered_tracking[int(instance_id)] = result["recovered"]
            tracking_rows.append(
                {
                    "scene_id": config.scene_id,
                    "instance_id": int(instance_id),
                    "instance_label": sequences[int(instance_id)].label,
                    "recovery_applied": int(result["recovery_applied"]),
                    "recovery_sequence_index": result[
                        "recovery_sequence_index"
                    ],
                    "recovery_frame_index": result["recovery_frame_index"],
                    "recovery_reason": result["recovery_reason"],
                    "selected_support_coverage": result[
                        "selected_support_coverage"
                    ],
                    "selected_candidate_gt_iou": result[
                        "selected_candidate_gt_iou"
                    ],
                }
            )
        _save_tracking_cache(
            cache_path,
            config=config,
            instance_ids=instance_ids,
            frame_indices=shared.frame_indices,
            original=original_tracking,
            recovered=recovered_tracking,
            tracking_rows=tracking_rows,
        )
    else:
        print(f"reusing tracking cache: {cache_path}")
        original_tracking, recovered_tracking, tracking_rows = cached
    _write_csv(config.output_dir / "tracking_summary.csv", tracking_rows)

    ground_truth = _load_ground_truth_sequence(
        config.manifest,
        scene_id=config.scene_id,
        frame_indices=config.frame_indices,
    )
    predicted_pose = _prepare_pose_sequence(
        geometry.world_to_camera,
        frame_indices=config.frame_indices,
        source="streamvggt",
    )
    target_pose = _prepare_pose_sequence(
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
    reference_grid_masks = {
        int(instance_id): output_mask_to_stream(
            target_masks[int(instance_id)][reference_sequence_index],
            source_size=geometry.source_sizes[reference_sequence_index],
            processed_size=geometry.processed_size,
            image_mode=config.image_mode,
        )
        for instance_id in instance_ids
    }
    grid_masks = {
        "recovered": _tracking_masks_to_geometry_grid(
            recovered_tracking,
            geometry=geometry,
            image_mode=config.image_mode,
        ),
    }
    for instance_id in instance_ids:
        grid_masks["recovered"][int(instance_id)][
            reference_sequence_index
        ] = reference_grid_masks[int(instance_id)]
    tracking_scores = {
        "recovered": {
            int(key): value.scores.double().cpu()
            for key, value in recovered_tracking.items()
        },
    }
    modes = _refinement_modes()
    mode_specs = {mode.name: mode for mode in modes}
    corrected_points: dict[str, torch.Tensor] = {}
    correction_rows = []
    proposal_rows = []
    for mode in modes:
        print(
            f"running refinement mode={mode.name} "
            f"profile={mode.proposal_profile} "
            f"map_update={mode.map_update_policy} "
            f"temporal={mode.temporal_policy}"
        )
        if mode.mask_source == "none":
            corrections = torch.zeros(
                (len(config.frame_indices), 3),
                dtype=torch.float64,
            )
            events = _zero_correction_rows(
                mode,
                frame_indices=config.frame_indices,
                reference_index=reference_sequence_index,
                reason="ray-only baseline; no instance correction",
            )
            proposals = []
        else:
            corrections, events, proposals = _estimate_v3_corrections(
                mode,
                instance_ids=instance_ids,
                frame_indices=config.frame_indices,
                reference_index=reference_sequence_index,
                world_points=geometry.world_points,
                confidence=geometry.confidence,
                masks=grid_masks[mode.mask_source],
                scores=tracking_scores[mode.mask_source],
                confidence_threshold=config.point_cloud_confidence_threshold,
                map_update_min_score=config.map_update_min_score,
                config=refinement,
                metric_scale=point_alignment.scale,
            )
        corrected_points[mode.name] = (
            geometry.world_points.double().cpu()
            + corrections[:, None, None, :]
        )
        correction_rows.extend(events)
        proposal_rows.extend(proposals)

    ray_config = RayFitConfig(
        trim_quantile=0.80,
        max_iterations=1,
        min_points=refinement.ray_min_points,
        max_points=refinement.ray_max_points,
        max_condition_number=refinement.ray_max_condition_number,
    )
    pose_sequences: dict[str, PoseSequence] = {
        "raw_camera_head": predicted_pose,
    }
    ray_rows = []
    for mode in modes:
        pose, rows = _fit_all_point_ray_pose(
            corrected_points[mode.name],
            confidence=geometry.confidence,
            intrinsics=geometry.intrinsics,
            predicted=predicted_pose,
            frame_indices=config.frame_indices,
            mode=mode,
            correction_rows=correction_rows,
            confidence_threshold=config.point_cloud_confidence_threshold,
            ray_config=ray_config,
            metric_scale=point_alignment.scale,
        )
        pose_sequences[mode.name] = pose
        ray_rows.extend(rows)

    (
        pose_summary_rows,
        pose_frame_rows,
        pose_rpe_rows,
        pose_pair_rows,
        pose_pair_summary_rows,
    ) = _evaluate_pose_modes(
        pose_sequences,
        mode_specs=mode_specs,
        target=target_pose,
        point_alignment=point_alignment,
        frame_indices=config.frame_indices,
        reference_index=reference_sequence_index,
    )
    pointmap_frame_rows = []
    pointmap_summary_rows = []
    for mode_name, points in corrected_points.items():
        aligned = float(point_alignment.scale) * (
            points @ point_alignment.rotation.T
        ) + point_alignment.translation
        frames = _pointmap_frame_metrics(
            aligned_points=aligned,
            gt_points=map_context.gt_pointmaps,
            confidence=geometry.confidence,
            confidence_threshold=config.point_cloud_confidence_threshold,
            frame_indices=config.frame_indices,
            reference_index=reference_sequence_index,
        )
        pointmap_frame_rows.extend(
            {"mode": mode_name, **row} for row in frames
        )
        pointmap_summary_rows.extend(
            {"mode": mode_name, **row}
            for row in _summarize_pointmap_rows(frames)
        )

    rows_by_name = {
        "instance_correction_events.csv": correction_rows,
        "instance_icp_diagnostics.csv": proposal_rows,
        "instance_ray_fit.csv": ray_rows,
        "instance_pose_summary.csv": pose_summary_rows,
        "instance_pose_frame_metrics.csv": pose_frame_rows,
        "instance_pose_rpe.csv": pose_rpe_rows,
        "instance_pose_pair_metrics.csv": pose_pair_rows,
        "instance_pose_pair_summary.csv": pose_pair_summary_rows,
        "instance_pointmap_frame_metrics.csv": pointmap_frame_rows,
        "instance_pointmap_summary.csv": pointmap_summary_rows,
    }
    for filename, rows in rows_by_name.items():
        _write_csv(
            config.output_dir / filename,
            _with_scene(config.scene_id, rows),
        )

    point_cloud_rows = _export_selected_instance_clouds(
        config,
        sequences=sequences,
        recovered_tracking=recovered_tracking,
        geometry=geometry,
        corrected_points=corrected_points,
        modes=(
            "ray_only",
            "v3_temporal_validated_a050",
        ),
    )
    _write_csv(
        config.output_dir / "corrected_pointcloud_summary.csv",
        point_cloud_rows,
    )
    metadata = {
        "experiment": "causal_multi_instance_pointmap_ray_pose_refinement_v3",
        "scene_id": config.scene_id,
        "frame_indices": list(config.frame_indices),
        "instance_ids": [int(value) for value in instance_ids],
        "reference_sequence_index": int(reference_sequence_index),
        "tracking_cache": str(cache_path),
        "tracking_cache_version": 2,
        "sam3_role": (
            "generate original and natural-recovery same-ID masks once; cache "
            "is reused on later runs"
        ),
        "reference_mask_role": (
            "GT prompt and object-map initialization only; all deployable "
            "post-reference decisions use predictions"
        ),
        "deployable_mode": "v3_temporal_validated_a050",
        "v2_result": (
            "V2 improved adjacent RPE slightly but degraded official-style "
            "all-pairs translation direction; its runtime branch was removed"
        ),
        "correction_order": [
            "tracking masks and persistent IDs",
            "strict per-instance translation-only trimmed alignment",
            "temporal conflict filtering against the last shared translation",
            "validated consensus/carry participants update only their own maps",
            "at most two consecutive bounded single-instance carries",
            "translate the whole current pointmap",
            "all-point predicted-K/R ray-center pose repair",
        ],
        "correction_scale_semantics": (
            "alpha=0.5 scales only the whole-frame pointmap correction; "
            "validated object maps use unscaled per-instance proposals"
        ),
        "gt_role": "pose and pointmap evaluation only",
        "refinement_config": {
            key: (
                list(value)
                if isinstance(value, tuple)
                else value
            )
            for key, value in refinement.__dict__.items()
        },
        "modes": {
            mode.name: {
                "role": mode.role,
                "mask_source": mode.mask_source,
                "map_policy": mode.map_policy,
                "correction_scale": mode.correction_scale,
                "proposal_profile": mode.proposal_profile,
                "map_update_policy": mode.map_update_policy,
                "temporal_policy": mode.temporal_policy,
            }
            for mode in modes
        },
        "fixed_reference_sim3": {
            "scale": float(map_context.sim3_scale),
            "rotation": map_context.sim3_rotation.tolist(),
            "translation": map_context.sim3_translation.tolist(),
            "gt_role": "evaluation_only",
        },
        "outputs": [
            "tracking_summary.csv",
            str(cache_path),
            *rows_by_name.keys(),
            "corrected_pointcloud_summary.csv",
            "instance_<id>/pointclouds/<mode>/",
        ],
    }
    with (config.output_dir / "metadata.json").open(
        "w",
        encoding="utf8",
    ) as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    print(
        "instance pose summary: "
        f"{config.output_dir / 'instance_pose_summary.csv'}"
    )
    print(
        "instance pointmap summary: "
        f"{config.output_dir / 'instance_pointmap_summary.csv'}"
    )


def _load_instance_sequences(
    config: ExperimentConfig,
    *,
    instance_ids: Sequence[int],
    reference_sequence_index: int,
) -> tuple[dict[int, object], dict[int, torch.Tensor]]:
    sequences = {}
    target_masks = {}
    for instance_id in instance_ids:
        sequence = load_mask_tracking_sequence(
            config.manifest,
            scene_id=config.scene_id,
            frame_indices=config.frame_indices,
            sequence_length=len(config.frame_indices),
            frame_stride=1,
            window_index=0,
            instance_id=int(instance_id),
            min_pixels=config.min_pixels,
            max_area_ratio=config.max_area_ratio,
            min_visible_frames=1,
            excluded_labels=config.excluded_labels,
            seed=0,
        )
        resized = resize_target_masks(
            sequence.target_masks,
            config.output_size,
        )
        reference_index = int(reference_sequence_index)
        if not 0 <= reference_index < len(sequence.frame_indices):
            raise ValueError("reference_sequence_index is outside the sequence.")
        if not resized[reference_index].any():
            raise ValueError(
                f"Instance {instance_id} is absent at reference frame "
                f"{sequence.frame_indices[reference_index]}."
            )
        sequence = replace(
            sequence,
            reference_frame_idx=reference_index,
        )
        sequences[int(instance_id)] = sequence
        target_masks[int(instance_id)] = resized
    first = sequences[int(instance_ids[0])]
    for instance_id in instance_ids[1:]:
        current = sequences[int(instance_id)]
        if (
            tuple(current.frame_indices) != tuple(first.frame_indices)
            or [str(path) for path in current.image_paths]
            != [str(path) for path in first.image_paths]
        ):
            raise RuntimeError("All instances must share the same RGB sequence.")
    return sequences, target_masks


def _refinement_modes() -> tuple[RefinementMode, ...]:
    """Return the final method and its minimal ray-center baseline."""

    modes = (
        RefinementMode(
            name="ray_only",
            role="ray_center_baseline",
            mask_source="none",
            map_policy="none",
            correction_scale=0.0,
        ),
        RefinementMode(
            name="v3_temporal_validated_a050",
            role="final_v3_method",
            mask_source="recovered",
            map_policy="causal",
            correction_scale=0.5,
            map_update_policy="validated",
            temporal_policy="validated_carry",
        ),
    )
    names = [mode.name for mode in modes]
    if len(names) != len(set(names)):
        raise RuntimeError("Refinement modes must have unique names.")
    return modes


def _select_bounded_temporal_inlier(
    distances: Mapping[int, float],
    *,
    frame_gap: int | None,
    carry_count: int,
    max_frame_gap: int,
    max_distance: float,
    max_carry_frames: int,
) -> tuple[int | None, tuple[int, ...], bool, bool]:
    """Select one temporally consistent proposal under a bounded carry."""

    inlier_ids = tuple(
        sorted(
            int(instance_id)
            for instance_id, distance in distances.items()
            if math.isfinite(float(distance))
            and float(distance) <= float(max_distance)
        )
    )
    gap_ok = (
        frame_gap is not None
        and 0 < int(frame_gap) <= int(max_frame_gap)
    )
    carry_budget_ok = int(carry_count) < int(max_carry_frames)
    candidate_id = (
        inlier_ids[0]
        if gap_ok and carry_budget_ok and len(inlier_ids) == 1
        else None
    )
    return candidate_id, inlier_ids, gap_ok, carry_budget_ok


def _estimate_v3_corrections(
    mode: RefinementMode,
    *,
    instance_ids: Sequence[int],
    frame_indices: Sequence[int],
    reference_index: int,
    world_points: torch.Tensor,
    confidence: torch.Tensor,
    masks: Mapping[int, torch.Tensor],
    scores: Mapping[int, torch.Tensor],
    confidence_threshold: float,
    map_update_min_score: float,
    config: InstanceRefinementConfig,
    metric_scale: float,
) -> tuple[torch.Tensor, list[dict], list[dict]]:
    """Estimate V3 corrections while maintaining validated instance maps.

    V3 lets temporal evidence resolve a conflict only when exactly one strict
    proposal remains close to the last shared correction. Only proposals that
    participated in an accepted consensus/carry may update their maps.
    """
    world_points = world_points.double().cpu()
    confidence = confidence.double().cpu()
    corrections = torch.zeros(
        (len(frame_indices), 3),
        dtype=torch.float64,
    )
    object_maps = {}
    for instance_id in instance_ids:
        points = _masked_points(
            world_points[reference_index],
            confidence[reference_index],
            masks[int(instance_id)][reference_index],
            confidence_threshold=confidence_threshold,
            max_points=config.map_max_points,
        )
        if points.shape[0] < config.min_instance_points:
            raise RuntimeError(
                f"Reference object map {instance_id} has only "
                f"{points.shape[0]} points."
            )
        object_maps[int(instance_id)] = points

    event_rows = _zero_correction_rows(
        mode,
        frame_indices=frame_indices,
        reference_index=reference_index,
        reason="reference object maps initialized",
    )
    event_rows[reference_index]["map_total_points"] = sum(
        int(value.shape[0]) for value in object_maps.values()
    )
    event_rows[reference_index]["consensus_distance_native"] = (
        config.consensus_distance
    )
    event_rows[reference_index]["temporal_max_carry_frames"] = (
        config.temporal_max_carry_frames
    )
    proposal_rows = []
    last_temporal_translation: torch.Tensor | None = None
    last_temporal_frame: int | None = None
    temporal_carry_count = 0
    for sequence_index in range(reference_index + 1, len(frame_indices)):
        proposals = []
        current_by_instance = {}
        for instance_id in instance_ids:
            instance_id = int(instance_id)
            current = _masked_points(
                world_points[sequence_index],
                confidence[sequence_index],
                masks[instance_id][sequence_index],
                confidence_threshold=confidence_threshold,
                max_points=config.icp_max_points,
            )
            current_by_instance[instance_id] = current
            proposal = _translation_icp(
                current,
                object_maps[instance_id],
                instance_id=instance_id,
                config=config,
            )
            proposals.append(proposal)

        eligible_proposals = [
            proposal
            for proposal in proposals
            if proposal.accepted
            and float(scores[proposal.instance_id][sequence_index])
            >= float(map_update_min_score)
        ]
        shared, participating, disagreement = _proposal_consensus(
            eligible_proposals,
            min_instances=config.min_participating_instances,
            max_distance=config.consensus_distance,
        )
        accepted = shared is not None
        correction_source = "none"
        temporal_candidate_id: int | None = None
        temporal_reference_frame = last_temporal_frame
        temporal_frame_gap = (
            int(frame_indices[sequence_index]) - int(last_temporal_frame)
            if last_temporal_frame is not None
            else None
        )
        temporal_proposal_distance = float("nan")
        temporal_distances: dict[int, float] = {}
        temporal_inlier_ids: tuple[int, ...] = ()
        carry_count_before = temporal_carry_count
        if accepted:
            reason = "accepted multi-instance shared translation"
            correction_source = "multi_instance_consensus"
            last_temporal_translation = shared.clone()
            last_temporal_frame = int(frame_indices[sequence_index])
            temporal_carry_count = 0
        else:
            reason = "rejected: insufficient cross-instance consensus"
            accepted_proposals = list(eligible_proposals)
            if last_temporal_translation is not None:
                temporal_distances = {
                    int(proposal.instance_id): float(
                        torch.linalg.vector_norm(
                            proposal.translation - last_temporal_translation
                        )
                    )
                    for proposal in accepted_proposals
                }
            (
                temporal_candidate_id,
                temporal_inlier_ids,
                gap_ok,
                carry_budget_ok,
            ) = _select_bounded_temporal_inlier(
                temporal_distances,
                frame_gap=temporal_frame_gap,
                carry_count=temporal_carry_count,
                max_frame_gap=config.temporal_max_frame_gap,
                max_distance=config.temporal_proposal_distance,
                max_carry_frames=config.temporal_max_carry_frames,
            )
            if temporal_candidate_id is not None:
                candidate = next(
                    proposal
                    for proposal in accepted_proposals
                    if proposal.instance_id == temporal_candidate_id
                )
                temporal_proposal_distance = temporal_distances[
                    temporal_candidate_id
                ]
                weight = float(config.temporal_current_weight)
                shared = (
                    (1.0 - weight) * last_temporal_translation
                    + weight * candidate.translation
                )
                participating = (temporal_candidate_id,)
                accepted = True
                correction_source = "single_instance_validated_carry"
                reason = (
                    "accepted unique temporal inlier after proposal "
                    "conflict filtering"
                )
                last_temporal_translation = shared.clone()
                last_temporal_frame = int(frame_indices[sequence_index])
                temporal_carry_count += 1
            else:
                reason = (
                    "rejected bounded temporal filtering: "
                    f"eligible={len(accepted_proposals)} "
                    f"inliers={len(temporal_inlier_ids)} "
                    f"gap_ok={int(gap_ok)} "
                    f"carry_budget_ok={int(carry_budget_ok)}"
                )
                last_temporal_translation = None
                last_temporal_frame = None
                temporal_carry_count = 0

        if accepted:
            applied = float(mode.correction_scale) * shared
            corrections[sequence_index] = applied
        else:
            applied = torch.zeros(3, dtype=torch.float64)

        map_updates = 0
        map_update_ids = []
        map_update_translations: dict[int, torch.Tensor] = {}
        if accepted:
            update_proposals = [
                proposal
                for proposal in proposals
                if proposal.accepted
                and proposal.instance_id in participating
            ]
        else:
            update_proposals = []
        for proposal in update_proposals:
            instance_id = int(proposal.instance_id)
            if (
                float(scores[instance_id][sequence_index])
                < float(map_update_min_score)
            ):
                continue
            update_translation = proposal.translation
            corrected = current_by_instance[instance_id] + update_translation
            object_maps[instance_id] = _merge_map_points(
                object_maps[instance_id],
                corrected,
                max_points=config.map_max_points,
            )
            map_updates += 1
            map_update_ids.append(instance_id)
            map_update_translations[instance_id] = update_translation

        for proposal in proposals:
            instance_id = int(proposal.instance_id)
            map_translation = map_update_translations.get(
                instance_id,
                torch.zeros(3, dtype=torch.float64),
            )
            proposal_row = {
                "mode": mode.name,
                "mode_role": mode.role,
                "mask_source": mode.mask_source,
                "map_policy": mode.map_policy,
                "correction_scale": mode.correction_scale,
                "proposal_profile": mode.proposal_profile,
                "map_update_policy": mode.map_update_policy,
                "temporal_policy": mode.temporal_policy,
                "sequence_index": sequence_index,
                "frame_index": int(frame_indices[sequence_index]),
                "instance_id": instance_id,
                "tracker_score": float(
                    scores[instance_id][sequence_index]
                ),
                "proposal_accepted": int(proposal.accepted),
                "proposal_reason": proposal.reason,
                "proposal_score_eligible": int(
                    proposal.accepted
                    and float(scores[instance_id][sequence_index])
                    >= float(map_update_min_score)
                ),
                "consensus_participant": int(
                    correction_source == "multi_instance_consensus"
                    and instance_id in participating
                ),
                "temporal_participant": int(
                    correction_source == "single_instance_validated_carry"
                    and instance_id in participating
                ),
                "temporal_distance_native": temporal_distances.get(
                    instance_id,
                    float("nan"),
                ),
                "temporal_inlier": int(
                    instance_id in temporal_inlier_ids
                ),
                "proposal_validated": int(
                    accepted and instance_id in participating
                ),
                "map_updated": int(instance_id in map_update_ids),
                "current_points": proposal.current_points,
                "map_points": proposal.map_points,
                "correspondences": proposal.correspondences,
                "icp_fitness": proposal.fitness,
                "icp_rmse_native": proposal.rmse,
                "icp_rmse_aligned_meters": (
                    float(metric_scale) * proposal.rmse
                ),
                "max_icp_rmse_native": config.max_icp_rmse,
                "correspondence_object_ratio": (
                    config.correspondence_object_ratio
                ),
                "consensus_distance_native": (
                    config.consensus_distance
                ),
                "correspondence_distance_native": (
                    proposal.correspondence_distance
                ),
                "object_scale_native": proposal.object_scale,
                "icp_iterations": proposal.iterations,
                "icp_initialization": proposal.initialization,
            }
            _add_vector(
                proposal_row,
                "proposal_translation_native",
                proposal.translation,
            )
            _add_vector(
                proposal_row,
                "map_update_translation_native",
                map_translation,
            )
            proposal_rows.append(proposal_row)

        if map_update_translations:
            representative_map_translation = torch.quantile(
                torch.stack(list(map_update_translations.values())),
                0.50,
                dim=0,
            )
        else:
            representative_map_translation = torch.zeros(
                3,
                dtype=torch.float64,
            )
        row = event_rows[sequence_index]
        row.update(
            {
                "correction_accepted": int(accepted),
                "correction_reason": reason,
                "correction_source": correction_source,
                "accepted_instance_proposals": sum(
                    int(proposal.accepted) for proposal in proposals
                ),
                "eligible_instance_proposals": len(eligible_proposals),
                "participating_instances": len(participating),
                "participating_instance_ids": " ".join(
                    str(value) for value in participating
                ),
                "max_consensus_disagreement_native": disagreement,
                "consensus_distance_native": (
                    config.consensus_distance
                ),
                "max_consensus_disagreement_aligned_meters": (
                    float(metric_scale) * disagreement
                ),
                "temporal_candidate_instance_id": temporal_candidate_id,
                "temporal_reference_frame_index": temporal_reference_frame,
                "temporal_frame_gap": temporal_frame_gap,
                "temporal_proposal_distance_native": (
                    temporal_proposal_distance
                ),
                "temporal_inlier_instance_ids": " ".join(
                    str(value) for value in temporal_inlier_ids
                ),
                "temporal_carry_count_before": carry_count_before,
                "temporal_carry_count_after": temporal_carry_count,
                "temporal_max_carry_frames": (
                    config.temporal_max_carry_frames
                ),
                "map_updates": map_updates,
                "map_update_instance_ids": " ".join(
                    str(value) for value in sorted(map_update_ids)
                ),
                "map_total_points": sum(
                    int(value.shape[0]) for value in object_maps.values()
                ),
            }
        )
        _add_vector(
            row,
            "shared_translation_native",
            shared if shared is not None else applied,
        )
        _add_vector(
            row,
            "map_update_translation_native",
            representative_map_translation,
        )
        _add_vector(row, "applied_translation_native", applied)
        row["applied_translation_norm_native"] = float(
            torch.linalg.vector_norm(applied)
        )
        row["applied_translation_norm_aligned_meters"] = (
            float(metric_scale) * row["applied_translation_norm_native"]
        )
    return corrections, event_rows, proposal_rows


def _translation_icp(
    current: torch.Tensor,
    object_map: torch.Tensor,
    *,
    instance_id: int,
    config: InstanceRefinementConfig,
) -> TranslationProposal:
    device = torch.device(config.compute_device)
    current = _deterministic_limit(
        current.float(),
        config.icp_max_points,
    ).to(device)
    object_map = _deterministic_limit(
        object_map.float(),
        config.map_max_points,
    ).to(device)
    empty = torch.zeros(3, dtype=torch.float64)
    if current.shape[0] < config.min_instance_points:
        return TranslationProposal(
            instance_id=instance_id,
            translation=empty,
            accepted=False,
            reason="insufficient current instance points",
            current_points=int(current.shape[0]),
            map_points=int(object_map.shape[0]),
            correspondences=0,
            fitness=0.0,
            rmse=float("nan"),
            correspondence_distance=float("nan"),
            object_scale=float("nan"),
            iterations=0,
            initialization="none",
        )
    if object_map.shape[0] < config.min_instance_points:
        return TranslationProposal(
            instance_id=instance_id,
            translation=empty,
            accepted=False,
            reason="insufficient persistent map points",
            current_points=int(current.shape[0]),
            map_points=int(object_map.shape[0]),
            correspondences=0,
            fitness=0.0,
            rmse=float("nan"),
            correspondence_distance=float("nan"),
            object_scale=float("nan"),
            iterations=0,
            initialization="none",
        )
    low = torch.quantile(object_map, 0.05, dim=0)
    high = torch.quantile(object_map, 0.95, dim=0)
    object_scale = float(torch.linalg.vector_norm(high - low))
    max_distance = max(
        float(config.correspondence_min_distance),
        float(config.correspondence_object_ratio) * object_scale,
    )
    zero = torch.zeros(3, dtype=torch.float32, device=device)
    initializations = [("zero", zero)]
    centroid = (
        torch.quantile(object_map, 0.50, dim=0)
        - torch.quantile(current, 0.50, dim=0)
    )
    if (
        float(torch.linalg.vector_norm(centroid))
        <= float(config.max_proposal_translation)
    ):
        initializations.append(("robust_centroid", centroid))
    fits = [
        _translation_icp_from_start(
            current,
            object_map,
            initialization=name,
            initial_translation=initial,
            max_distance=max_distance,
            config=config,
        )
        for name, initial in initializations
    ]
    bounded_fits = [
        value
        for value in fits
        if float(
            torch.linalg.vector_norm(value["translation"])
        )
        <= float(config.max_proposal_translation)
    ]
    best = max(
        bounded_fits if bounded_fits else fits,
        key=lambda value: (
            value["fitness"],
            (
                -value["rmse"]
                if math.isfinite(value["rmse"])
                else float("-inf")
            ),
        ),
    )
    translation = best["translation"]
    iterations = best["iterations"]
    correspondences = best["correspondences"]
    fitness = best["fitness"]
    rmse = best["rmse"]
    translation_norm = float(torch.linalg.vector_norm(translation))
    accepted = (
        correspondences >= config.min_instance_points
        and fitness >= config.min_icp_fitness
        and rmse <= config.max_icp_rmse
        and translation_norm <= config.max_proposal_translation
        and math.isfinite(rmse)
    )
    reasons = []
    if correspondences < config.min_instance_points:
        reasons.append("too few correspondences")
    if fitness < config.min_icp_fitness:
        reasons.append("low fitness")
    if rmse > config.max_icp_rmse:
        reasons.append("high rmse")
    if translation_norm > config.max_proposal_translation:
        reasons.append("translation too large")
    if not math.isfinite(rmse):
        reasons.append("non-finite rmse")
    return TranslationProposal(
        instance_id=instance_id,
        translation=translation.double().cpu(),
        accepted=accepted,
        reason="accepted" if accepted else "; ".join(reasons),
        current_points=int(current.shape[0]),
        map_points=int(object_map.shape[0]),
        correspondences=correspondences,
        fitness=fitness,
        rmse=rmse,
        correspondence_distance=max_distance,
        object_scale=object_scale,
        iterations=iterations,
        initialization=str(best["initialization"]),
    )


def _translation_icp_from_start(
    current: torch.Tensor,
    object_map: torch.Tensor,
    *,
    initialization: str,
    initial_translation: torch.Tensor,
    max_distance: float,
    config: InstanceRefinementConfig,
) -> dict:
    translation = initial_translation.clone()
    iterations = 0
    for _ in range(config.icp_iterations):
        shifted = current + translation
        distances = torch.cdist(shifted, object_map)
        nearest_distance, nearest_index = distances.min(dim=1)
        cutoff = min(
            max_distance,
            float(
                torch.quantile(
                    nearest_distance,
                    config.icp_trim_quantile,
                )
            ),
        )
        keep = nearest_distance <= cutoff
        if int(keep.sum()) < config.min_instance_points:
            break
        matched = object_map.index_select(0, nearest_index[keep])
        residual = matched - shifted[keep]
        step = torch.quantile(residual, 0.50, dim=0)
        translation += step
        iterations += 1
        if float(torch.linalg.vector_norm(step)) <= 1e-4:
            break

    shifted = current + translation
    distances = torch.cdist(shifted, object_map)
    nearest_distance = distances.min(dim=1).values
    inliers = nearest_distance <= max_distance
    correspondences = int(inliers.sum())
    fitness = float(inliers.float().mean())
    rmse = (
        float(torch.sqrt(nearest_distance[inliers].square().mean()))
        if correspondences
        else float("nan")
    )
    return {
        "initialization": initialization,
        "translation": translation,
        "iterations": iterations,
        "correspondences": correspondences,
        "fitness": fitness,
        "rmse": rmse,
    }


def _proposal_consensus(
    proposals: Sequence[TranslationProposal],
    *,
    min_instances: int,
    max_distance: float,
) -> tuple[torch.Tensor | None, tuple[int, ...], float]:
    """Robust-center consensus with a gate on its final maximum residual."""

    accepted = [proposal for proposal in proposals if proposal.accepted]
    if len(accepted) < int(min_instances):
        return None, (), float("nan")
    cluster = list(accepted)
    disagreement = float("nan")
    while len(cluster) >= int(min_instances):
        translations = torch.stack(
            [proposal.translation for proposal in cluster]
        )
        shared = torch.quantile(translations, 0.50, dim=0)
        residuals = torch.linalg.vector_norm(
            translations - shared,
            dim=1,
        )
        disagreement = float(residuals.max())
        if disagreement <= float(max_distance):
            participating = tuple(
                sorted(proposal.instance_id for proposal in cluster)
            )
            return shared, participating, disagreement
        worst = int(torch.argmax(residuals))
        cluster.pop(worst)
    return None, (), disagreement


def _fit_all_point_ray_pose(
    world_points: torch.Tensor,
    *,
    confidence: torch.Tensor,
    intrinsics: torch.Tensor,
    predicted: PoseSequence,
    frame_indices: Sequence[int],
    mode: RefinementMode,
    correction_rows: Sequence[dict],
    confidence_threshold: float,
    ray_config: RayFitConfig,
    metric_scale: float,
) -> tuple[PoseSequence, list[dict]]:
    mode_corrections = {
        int(row["sequence_index"]): row
        for row in correction_rows
        if row["mode"] == mode.name
    }
    centers = []
    rows = []
    for index, frame_index in enumerate(frame_indices):
        (
            sampled_points,
            directions,
            weights,
            candidate_points,
        ) = _prepare_ray_inputs(
            world_points[index],
            confidence[index].double().cpu(),
            intrinsics[index].double().cpu(),
            predicted.camera_to_world_rotation[index],
            confidence_threshold=confidence_threshold,
            max_points=ray_config.max_points,
        )
        correction = mode_corrections[index]
        applied = torch.tensor(
            [
                correction["applied_translation_native_x"],
                correction["applied_translation_native_y"],
                correction["applied_translation_native_z"],
            ],
            dtype=torch.float64,
        )
        fit = _fit_ray_center(
            sampled_points,
            directions,
            weights,
            candidate_points=candidate_points,
            fallback_center=predicted.camera_centers[index] + applied,
            robust_trim=False,
            config=ray_config,
        )
        centers.append(fit.center)
        rows.append(
            {
                "mode": mode.name,
                "mode_role": mode.role,
                "sequence_index": index,
                "frame_index": int(frame_index),
                "fit_accepted": int(fit.fit_accepted),
                "fit_status": fit.status,
                "candidate_points": fit.candidate_points,
                "sampled_points": fit.sampled_points,
                "condition_number": fit.condition_number,
                "all_ray_residual_rmse_native": fit.all_residual_rmse,
                "all_ray_residual_rmse_aligned_meters": (
                    float(metric_scale) * fit.all_residual_rmse
                ),
            }
        )
    return (
        _pose_sequence_from_centers(
            predicted.camera_to_world_rotation,
            torch.stack(centers),
        ),
        rows,
    )


def _evaluate_pose_modes(
    sequences: Mapping[str, PoseSequence],
    *,
    mode_specs: Mapping[str, RefinementMode],
    target: PoseSequence,
    point_alignment: SimilarityAlignment,
    frame_indices: Sequence[int],
    reference_index: int,
) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict]]:
    summary_rows = []
    frame_rows = []
    rpe_rows = []
    pair_rows = []
    pair_summary_rows = []
    for mode_name, sequence in sequences.items():
        spec = mode_specs.get(mode_name)
        common = {
            "mode": mode_name,
            "mode_role": spec.role if spec is not None else "raw_baseline",
            "mask_source": spec.mask_source if spec is not None else "none",
            "map_policy": spec.map_policy if spec is not None else "none",
            "proposal_profile": (
                spec.proposal_profile if spec is not None else "none"
            ),
            "map_update_policy": (
                spec.map_update_policy if spec is not None else "none"
            ),
            "temporal_policy": (
                spec.temporal_policy if spec is not None else "none"
            ),
            "correction_scale": (
                spec.correction_scale if spec is not None else 0.0
            ),
        }
        alignments = (
            point_alignment,
            _reference_pose_alignment(
                sequence,
                target,
                reference_index=reference_index,
                scale=point_alignment.scale,
            ),
        )
        for alignment in alignments:
            summary, frames, rpe = _evaluate_pose_alignment(
                alignment,
                predicted=sequence,
                target=target,
                frame_indices=frame_indices,
                reference_index=reference_index,
            )
            summary_rows.append({**common, **summary})
            frame_rows.extend({**common, **row} for row in frames)
            rpe_rows.extend({**common, **row} for row in rpe)
        pairs = _all_pair_pose_metrics(
            sequence,
            target,
            frame_indices=frame_indices,
        )
        pair_rows.extend({**common, **row} for row in pairs)
        pair_summary_rows.extend(
            {**common, **row}
            for row in _summarize_pose_pairs(pairs)
        )
    return (
        summary_rows,
        frame_rows,
        rpe_rows,
        pair_rows,
        pair_summary_rows,
    )


def _zero_correction_rows(
    mode: RefinementMode,
    *,
    frame_indices: Sequence[int],
    reference_index: int,
    reason: str,
) -> list[dict]:
    rows = []
    zero = torch.zeros(3, dtype=torch.float64)
    for index, frame_index in enumerate(frame_indices):
        row = {
            "mode": mode.name,
            "mode_role": mode.role,
            "mask_source": mode.mask_source,
            "map_policy": mode.map_policy,
            "correction_scale": mode.correction_scale,
            "proposal_profile": mode.proposal_profile,
            "map_update_policy": mode.map_update_policy,
            "temporal_policy": mode.temporal_policy,
            "sequence_index": index,
            "frame_index": int(frame_index),
            "is_reference": int(index == int(reference_index)),
            "correction_accepted": 0,
            "correction_reason": reason,
            "correction_source": "none",
            "accepted_instance_proposals": 0,
            "eligible_instance_proposals": 0,
            "participating_instances": 0,
            "participating_instance_ids": "",
            "max_consensus_disagreement_native": float("nan"),
            "consensus_distance_native": float("nan"),
            "max_consensus_disagreement_aligned_meters": float("nan"),
            "map_updates": 0,
            "map_update_instance_ids": "",
            "map_total_points": 0,
            "temporal_candidate_instance_id": None,
            "temporal_reference_frame_index": None,
            "temporal_frame_gap": None,
            "temporal_proposal_distance_native": float("nan"),
            "temporal_inlier_instance_ids": "",
            "temporal_carry_count_before": 0,
            "temporal_carry_count_after": 0,
            "temporal_max_carry_frames": 0,
            "applied_translation_norm_native": 0.0,
            "applied_translation_norm_aligned_meters": 0.0,
        }
        _add_vector(row, "shared_translation_native", zero)
        _add_vector(row, "map_update_translation_native", zero)
        _add_vector(row, "applied_translation_native", zero)
        rows.append(row)
    return rows


def _masked_points(
    world_points: torch.Tensor,
    confidence: torch.Tensor,
    mask: torch.Tensor,
    *,
    confidence_threshold: float,
    max_points: int,
) -> torch.Tensor:
    valid = (
        mask.bool()
        & torch.isfinite(world_points).all(dim=-1)
        & torch.isfinite(confidence)
        & (confidence >= float(confidence_threshold))
    )
    return _deterministic_limit(
        world_points[valid].double().cpu(),
        max_points,
    )


def _merge_map_points(
    previous: torch.Tensor,
    current: torch.Tensor,
    *,
    max_points: int,
) -> torch.Tensor:
    return _deterministic_limit(
        torch.cat([previous, current], dim=0),
        max_points,
    )


def _deterministic_limit(
    values: torch.Tensor,
    limit: int,
) -> torch.Tensor:
    if values.shape[0] <= int(limit):
        return values
    positions = torch.linspace(
        0,
        values.shape[0] - 1,
        steps=int(limit),
        dtype=torch.float64,
    ).round().long()
    return values.index_select(0, positions)


def _tracking_masks_to_geometry_grid(
    trackings: Mapping[int, TrackingSequence],
    *,
    geometry: GeometrySequence,
    image_mode: str,
) -> dict[int, torch.Tensor]:
    return {
        int(instance_id): _masks_to_geometry_grid(
            tracking.masks,
            geometry=geometry,
            image_mode=image_mode,
        )
        for instance_id, tracking in trackings.items()
    }


def _masks_to_geometry_grid(
    masks: torch.Tensor,
    *,
    geometry: GeometrySequence,
    image_mode: str,
) -> torch.Tensor:
    return torch.stack(
        [
            output_mask_to_stream(
                mask,
                source_size=geometry.source_sizes[index],
                processed_size=geometry.processed_size,
                image_mode=image_mode,
            )
            for index, mask in enumerate(masks)
        ]
    ).bool()


def _export_selected_instance_clouds(
    config: ExperimentConfig,
    *,
    sequences: Mapping[int, object],
    recovered_tracking: Mapping[int, TrackingSequence],
    geometry: GeometrySequence,
    corrected_points: Mapping[str, torch.Tensor],
    modes: Sequence[str],
) -> list[dict]:
    colors = load_processed_colors(
        next(iter(sequences.values())).image_paths,
        processed_size=geometry.processed_size,
        image_mode=config.image_mode,
    )
    rows = []
    for mode in modes:
        if mode not in corrected_points:
            continue
        corrected_geometry = replace(
            geometry,
            world_points=corrected_points[mode].float(),
        )
        for instance_id, sequence in sequences.items():
            result = export_instance_point_clouds(
                config.output_dir
                / f"instance_{instance_id}"
                / "pointclouds"
                / mode,
                frame_indices=sequence.frame_indices,
                geometry=corrected_geometry,
                colors=colors,
                predictions={
                    "recovered_tracking": recovered_tracking[int(instance_id)]
                },
                reference_frame_idx=sequence.reference_frame_idx,
                reference_mask=torch.from_numpy(
                    np.asarray(
                        sequence.target_masks[sequence.reference_frame_idx]
                    )
                ).bool(),
                image_mode=config.image_mode,
                confidence_threshold=config.point_cloud_confidence_threshold,
                max_points=config.point_cloud_max_points,
            )
            rows.extend(
                {
                    "scene_id": config.scene_id,
                    "instance_id": int(instance_id),
                    "instance_label": sequence.label,
                    "pointmap_mode": mode,
                    **row,
                }
                for row in result["summary_rows"]
            )
    return rows


def _save_tracking_cache(
    path: Path,
    *,
    config: ExperimentConfig,
    instance_ids: Sequence[int],
    frame_indices: Sequence[int],
    original: Mapping[int, TrackingSequence],
    recovered: Mapping[int, TrackingSequence],
    tracking_rows: Sequence[dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "cache_version": np.asarray([2], dtype=np.int64),
        "instance_ids": np.asarray(instance_ids, dtype=np.int64),
        "frame_indices": np.asarray(frame_indices, dtype=np.int64),
        "output_size": np.asarray(config.output_size, dtype=np.int64),
        "tracking_signature": np.asarray(
            [_tracking_cache_signature(config)]
        ),
        "tracking_rows_json": np.asarray(
            [
                json.dumps(
                    list(tracking_rows),
                    ensure_ascii=False,
                    allow_nan=True,
                )
            ]
        ),
    }
    for instance_id in instance_ids:
        for name, values in (
            ("original", original[int(instance_id)]),
            ("recovered", recovered[int(instance_id)]),
        ):
            _validate_cached_tracking_shape(
                values,
                frame_count=len(frame_indices),
                output_size=config.output_size,
                name=f"{name}_{int(instance_id)}",
            )
            prefix = f"{name}_{int(instance_id)}"
            arrays[f"{prefix}_masks"] = values.masks.cpu().numpy().astype(
                np.uint8
            )
            arrays[f"{prefix}_scores"] = (
                values.scores.reshape(-1).cpu().numpy().astype(np.float32)
            )
            arrays[f"{prefix}_obj_id"] = np.asarray(
                [
                    -1
                    if values.selected_obj_id is None
                    else int(values.selected_obj_id)
                ],
                dtype=np.int64,
            )
    np.savez_compressed(path, **arrays)


def _load_tracking_cache(
    path: Path,
    *,
    config: ExperimentConfig,
    instance_ids: Sequence[int],
    frame_indices: Sequence[int],
) -> tuple[
    dict[int, TrackingSequence],
    dict[int, TrackingSequence],
    list[dict],
] | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as values:
            if (
                values["cache_version"].tolist() != [2]
                or values["instance_ids"].tolist()
                != [int(value) for value in instance_ids]
                or values["frame_indices"].tolist()
                != [int(value) for value in frame_indices]
                or values["output_size"].tolist()
                != [int(value) for value in config.output_size]
                or str(values["tracking_signature"][0])
                != _tracking_cache_signature(config)
            ):
                return None
            output = {}
            for name in ("original", "recovered"):
                trackings = {}
                for instance_id in instance_ids:
                    prefix = f"{name}_{int(instance_id)}"
                    selected = int(values[f"{prefix}_obj_id"][0])
                    tracking = TrackingSequence(
                        masks=torch.from_numpy(
                            values[f"{prefix}_masks"].copy()
                        ).bool(),
                        scores=torch.from_numpy(
                            values[f"{prefix}_scores"].copy()
                        ).float(),
                        selected_obj_id=(
                            None if selected < 0 else selected
                        ),
                    )
                    _validate_cached_tracking_shape(
                        tracking,
                        frame_count=len(frame_indices),
                        output_size=config.output_size,
                        name=prefix,
                    )
                    trackings[int(instance_id)] = tracking
                output[name] = trackings
            tracking_rows = json.loads(
                str(values["tracking_rows_json"][0])
            )
            if (
                not isinstance(tracking_rows, list)
                or len(tracking_rows) != len(instance_ids)
                or not all(isinstance(row, dict) for row in tracking_rows)
                or [
                    int(row.get("instance_id", -1))
                    for row in tracking_rows
                ]
                != [int(value) for value in instance_ids]
            ):
                return None
            return (
                output["original"],
                output["recovered"],
                tracking_rows,
            )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _tracking_cache_signature(config: ExperimentConfig) -> str:
    excluded = {"output_dir", "sam3_device", "geometry_device"}
    values = {
        key: _json_cache_value(value)
        for key, value in config.__dict__.items()
        if key not in excluded
    }
    return json.dumps(
        values,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _json_cache_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_cache_value(item) for item in value]
    return value


def _validate_cached_tracking_shape(
    tracking: TrackingSequence,
    *,
    frame_count: int,
    output_size: Sequence[int],
    name: str,
) -> None:
    expected_masks = (
        int(frame_count),
        int(output_size[0]),
        int(output_size[1]),
    )
    if tuple(tracking.masks.shape) != expected_masks:
        raise ValueError(
            f"Tracking cache {name} masks have shape "
            f"{tuple(tracking.masks.shape)}, expected {expected_masks}."
        )
    if tracking.scores.numel() != int(frame_count):
        raise ValueError(
            f"Tracking cache {name} scores contain "
            f"{tracking.scores.numel()} values, expected {frame_count}."
        )


def _validate_refinement_config(config: InstanceRefinementConfig) -> None:
    if config.min_instance_points < 3:
        raise ValueError("instance min points must be at least 3.")
    if config.icp_max_points < config.min_instance_points:
        raise ValueError("instance ICP max points is below min points.")
    if config.map_max_points < config.min_instance_points:
        raise ValueError("instance map max points is below min points.")
    if config.icp_iterations < 1:
        raise ValueError("instance ICP iterations must be positive.")
    if not 0.0 < config.icp_trim_quantile <= 1.0:
        raise ValueError("instance ICP trim quantile must be in (0,1].")
    if not 0.0 <= config.min_icp_fitness <= 1.0:
        raise ValueError("instance ICP fitness must be in [0,1].")
    for value in (
        config.max_icp_rmse,
        config.correspondence_min_distance,
        config.correspondence_object_ratio,
        config.max_proposal_translation,
        config.consensus_distance,
        config.temporal_proposal_distance,
    ):
        if value <= 0.0:
            raise ValueError("instance distance thresholds must be positive.")
    if config.min_participating_instances < 2:
        raise ValueError("instance consensus requires at least two instances.")
    if config.temporal_max_frame_gap < 1:
        raise ValueError("temporal max frame gap must be positive.")
    if not 0.0 < config.temporal_current_weight <= 1.0:
        raise ValueError("temporal current weight must be in (0,1].")
    if config.temporal_max_carry_frames < 1:
        raise ValueError("temporal max carry frames must be positive.")


def _unique_ids(values: Sequence[int]) -> tuple[int, ...]:
    result = tuple(dict.fromkeys(int(value) for value in values))
    if len(result) < 2:
        raise ValueError("At least two unique static instances are required.")
    return result


def _add_vector(row: dict, prefix: str, vector: torch.Tensor) -> None:
    row[f"{prefix}_x"] = float(vector[0])
    row[f"{prefix}_y"] = float(vector[1])
    row[f"{prefix}_z"] = float(vector[2])


def _with_scene(scene_id: str, rows: Sequence[dict]) -> list[dict]:
    return [{"scene_id": str(scene_id), **row} for row in rows]


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf8")
        return
    fieldnames = list(
        dict.fromkeys(
            key
            for row in rows
            for key in row.keys()
        )
    )
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
    parser.add_argument("--instance-ids", type=int, nargs="+", required=True)
    parser.add_argument("--frame-indices", type=int, nargs="+")
    parser.add_argument(
        "--reference-sequence-index",
        type=int,
        default=0,
    )
    parser.add_argument("--sam3-device")
    parser.add_argument("--geometry-device")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--tracking-cache",
        type=Path,
        help=(
            "Optional reusable tracking_cache.npz from a previous run. "
            "If it is absent, SAM3 runs once and creates it here."
        ),
    )
    parser.add_argument("--instance-min-points", type=int, default=128)
    parser.add_argument("--instance-icp-max-points", type=int, default=1024)
    parser.add_argument("--instance-map-max-points", type=int, default=4096)
    parser.add_argument("--instance-icp-iterations", type=int, default=4)
    parser.add_argument("--instance-icp-device")
    parser.add_argument(
        "--instance-icp-trim-quantile",
        type=float,
        default=0.70,
    )
    parser.add_argument(
        "--instance-min-icp-fitness",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--instance-max-icp-rmse",
        type=float,
        default=0.03,
    )
    parser.add_argument(
        "--instance-correspondence-min-distance",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--instance-correspondence-object-ratio",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--instance-max-translation",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--instance-min-participants",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--instance-consensus-distance",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--instance-temporal-max-frame-gap",
        type=int,
        default=15,
    )
    parser.add_argument(
        "--instance-temporal-distance",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--instance-temporal-current-weight",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--instance-temporal-max-carry-frames",
        type=int,
        default=2,
    )
    parser.add_argument("--ray-min-points", type=int, default=1024)
    parser.add_argument("--ray-max-points", type=int, default=65536)
    parser.add_argument(
        "--ray-max-condition-number",
        type=float,
        default=1e8,
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
