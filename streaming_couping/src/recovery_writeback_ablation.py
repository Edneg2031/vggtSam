"""Causal ablation for bidirectional instance memory and SAM3 writeback.

One invocation runs both a deployable natural joint gate and a scheduled
intervention probe. The probe guarantees that memory writeback can be tested
even when the frozen tracker is too strong to trigger recovery naturally.
GT after the reference is metrics-only except in explicitly named oracle
upper-bound branches.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw

from test_sam.data import load_mask_tracking_sequence

from .backbones.sam3_wrapper import SAM3Wrapper
from .backbones.streamvggt_wrapper import StreamVGGTWrapper
from .bridge.gating import binary_iou
from .config import ExperimentConfig, load_config
from .instance_point_cloud import (
    export_instance_point_clouds,
    load_processed_colors,
)
from .instance_map_evaluation import (
    evaluate_instance_maps,
    prepare_map_evaluation,
)
from .recovery import (
    mine_recovery,
    resize_target_masks,
    summarize_masks,
    summarize_visible_after,
)
from .types import GeometrySequence, SAM3MaskCandidate, TrackingSequence


MODES = (
    "original",
    "geometry_recovery_no_memory",
    "reference_geometry_same_id_memory",
    "geometry_recovery_same_id_memory",
    "shuffled_geometry_same_id_memory",
    "oracle_candidate_same_id_memory",
    "oracle_mask_same_id_memory",
)
EVENT_POLICIES = (
    "natural_joint_gate",
    "scheduled_probe",
)


def main() -> None:
    args = _parse_args()
    instance_ids = _unique_ids(args.instance_ids)
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
    run_experiment(
        config,
        instance_ids=instance_ids,
        reference_sequence_index=args.reference_sequence_index,
        event_policies=args.event_policies,
        probe_sequence_index=args.probe_sequence_index,
    )


def run_experiment(
    config: ExperimentConfig,
    *,
    instance_ids: Sequence[int],
    reference_sequence_index: int | None,
    event_policies: Sequence[str] = EVENT_POLICIES,
    probe_sequence_index: int = 4,
) -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    config.output_dir.mkdir(parents=True, exist_ok=True)

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
        resized = resize_target_masks(sequence.target_masks, config.output_size)
        if reference_sequence_index is None:
            visible_indices = resized.flatten(1).any(dim=1).nonzero(
                as_tuple=False
            ).flatten()
            if not len(visible_indices):
                raise ValueError(
                    f"Instance {instance_id} is not visible in the selected sequence."
                )
            reference_index = int(visible_indices[0])
        else:
            reference_index = int(reference_sequence_index)
            if not 0 <= reference_index < len(sequence.frame_indices):
                raise ValueError(
                    f"Reference sequence index {reference_index} is outside the "
                    f"{len(sequence.frame_indices)} selected frames."
                )
            if not resized[reference_index].any():
                raise ValueError(
                    f"Instance {instance_id} is absent at requested reference "
                    f"frame {sequence.frame_indices[reference_index]}."
                )
        sequence = replace(sequence, reference_frame_idx=reference_index)
        sequences[int(instance_id)] = sequence
        target_masks[int(instance_id)] = resized

    _validate_shared_sequence(sequences)
    shared = sequences[int(instance_ids[0])]
    event_policies = tuple(dict.fromkeys(str(value) for value in event_policies))
    invalid_policies = [
        value for value in event_policies if value not in EVENT_POLICIES
    ]
    if invalid_policies:
        raise ValueError(f"Unknown event policies: {invalid_policies}.")
    probe_sequence_index = int(probe_sequence_index)
    if "scheduled_probe" in event_policies and not (
        shared.reference_frame_idx
        < probe_sequence_index
        < len(shared.frame_indices) - 1
    ):
        raise ValueError(
            "The scheduled probe must be after the reference and before the "
            "last selected frame so that future propagation can be measured."
        )
    print(
        f"causal recovery ablation scene={shared.scene_id} "
        f"frames={shared.frame_indices} instances={list(instance_ids)} "
        f"policies={list(event_policies)} probe_index={probe_sequence_index}"
    )

    print("extracting frozen StreamVGGT geometry once...")
    geometry = StreamVGGTWrapper(
        repo_path=config.streamvggt_repo,
        checkpoint_path=config.streamvggt_checkpoint,
        device=config.geometry_device,
        image_mode=config.image_mode,
        streaming_cache=config.streaming_cache,
    ).load().extract(shared.image_paths)
    point_cloud_colors = load_processed_colors(
        shared.image_paths,
        processed_size=geometry.processed_size,
        image_mode=config.image_mode,
    )
    map_evaluation_context = None
    map_evaluation_errors = []
    try:
        map_evaluation_context = prepare_map_evaluation(
            config,
            scene_id=shared.scene_id,
            frame_indices=shared.frame_indices,
            geometry=geometry,
            reference_frame_idx=shared.reference_frame_idx,
        )
        print(
            "prepared evaluation-only GT map alignment: "
            f"scale={map_evaluation_context.sim3_scale:.4f} "
            f"rmse={map_evaluation_context.sim3_rmse:.4f}"
        )
    except Exception as error:
        message = (
            "map-quality evaluation disabled because GT preparation failed: "
            f"{type(error).__name__}: {error}"
        )
        print(f"WARNING: {message}")
        map_evaluation_errors.append(message)

    print("loading frozen SAM3...")
    sam3 = SAM3Wrapper(
        repo_path=config.sam3_repo,
        checkpoint_path=config.sam3_checkpoint,
        device=config.sam3_device,
        output_threshold=config.sam3_output_threshold,
        prompt_with_box=config.prompt_with_box,
    ).load()

    all_summary_rows = []
    all_frame_rows = []
    all_gate_rows = []
    all_candidate_rows = []
    all_candidate_screening_rows = []
    all_point_cloud_summary_rows = []
    all_point_cloud_frame_rows = []
    all_map_quality_rows = []
    instance_metadata = {}
    for instance_id in instance_ids:
        sequence = sequences[int(instance_id)]
        print(
            f"running instance={instance_id} label={sequence.label!r} "
            f"reference={sequence.frame_indices[sequence.reference_frame_idx]}"
        )
        instance_dir = config.output_dir / f"instance_{instance_id}"
        instance_dir.mkdir(parents=True, exist_ok=True)
        original = sam3.track(
            sequence.image_paths,
            prompt=sequence.label,
            output_size=config.output_size,
            reference_frame_idx=sequence.reference_frame_idx,
            reference_mask=target_masks[int(instance_id)][
                sequence.reference_frame_idx
            ],
        )
        candidate_cache: dict[int, list[SAM3MaskCandidate]] = {}
        screening = _screen_candidate_frames(
            config,
            sequence=sequence,
            target_masks=target_masks[int(instance_id)],
            geometry=geometry,
            sam3=sam3,
            original=original,
            candidate_cache=candidate_cache,
        )
        all_candidate_rows.extend(screening["candidate_rows"])
        all_candidate_screening_rows.extend(screening["summary_rows"])
        _write_csv(
            instance_dir / "candidate_screening.csv",
            screening["summary_rows"],
        )
        policy_metadata = {}
        for event_policy in event_policies:
            forced_index = (
                probe_sequence_index
                if event_policy == "scheduled_probe"
                else None
            )
            print(
                f"  policy={event_policy} "
                f"forced_sequence_index={forced_index}"
            )
            result = _run_instance(
                config,
                sequence=sequence,
                target_masks=target_masks[int(instance_id)],
                geometry=geometry,
                sam3=sam3,
                original=original,
                candidate_cache=candidate_cache,
                event_policy=event_policy,
                forced_recovery_index=forced_index,
            )
            policy_dir = instance_dir / event_policy
            policy_dir.mkdir(parents=True, exist_ok=True)
            point_cloud_result = export_instance_point_clouds(
                policy_dir / "pointclouds",
                frame_indices=sequence.frame_indices,
                geometry=geometry,
                colors=point_cloud_colors,
                predictions=result["predictions"],
                reference_frame_idx=sequence.reference_frame_idx,
                reference_mask=torch.from_numpy(
                    np.asarray(
                        sequence.target_masks[sequence.reference_frame_idx]
                    )
                ).bool(),
                image_mode=config.image_mode,
                confidence_threshold=(
                    config.point_cloud_confidence_threshold
                ),
                max_points=config.point_cloud_max_points,
            )
            point_cloud_summary_rows = [
                {
                    "scene_id": sequence.scene_id,
                    "instance_id": int(instance_id),
                    "instance_label": sequence.label,
                    "event_policy": event_policy,
                    **row,
                }
                for row in point_cloud_result["summary_rows"]
            ]
            point_cloud_frame_rows = [
                {
                    "scene_id": sequence.scene_id,
                    "instance_id": int(instance_id),
                    "instance_label": sequence.label,
                    "event_policy": event_policy,
                    **row,
                }
                for row in point_cloud_result["frame_rows"]
            ]
            map_quality_rows = []
            if map_evaluation_context is not None:
                try:
                    map_quality_rows = evaluate_instance_maps(
                        config,
                        context=map_evaluation_context,
                        sequence=sequence,
                        target_masks=target_masks[int(instance_id)],
                        geometry=geometry,
                        predictions=result["predictions"],
                        event_policy=event_policy,
                    )
                except Exception as error:
                    message = (
                        f"instance={instance_id} policy={event_policy}: "
                        f"{type(error).__name__}: {error}"
                    )
                    print(f"WARNING: map-quality evaluation failed: {message}")
                    map_evaluation_errors.append(message)

            all_summary_rows.extend(result["summary_rows"])
            all_frame_rows.extend(result["frame_rows"])
            all_gate_rows.extend(result["gate_rows"])
            all_candidate_rows.extend(result["candidate_rows"])
            all_point_cloud_summary_rows.extend(point_cloud_summary_rows)
            all_point_cloud_frame_rows.extend(point_cloud_frame_rows)
            all_map_quality_rows.extend(map_quality_rows)
            policy_metadata[event_policy] = result["metadata"]
            _write_csv(policy_dir / "summary.csv", result["summary_rows"])
            _write_csv(policy_dir / "frame_metrics.csv", result["frame_rows"])
            _write_csv(
                policy_dir / "geometry_gate_diagnostics.csv",
                result["gate_rows"],
            )
            _write_csv(
                policy_dir / "candidate_diagnostics.csv",
                result["candidate_rows"],
            )
            _write_csv(
                policy_dir / "pointcloud_summary.csv",
                point_cloud_summary_rows,
            )
            _write_csv(
                policy_dir / "pointcloud_frame_metrics.csv",
                point_cloud_frame_rows,
            )
            _write_csv(
                policy_dir / "map_quality.csv",
                map_quality_rows,
            )
            _save_report(
                policy_dir / "recovery_writeback_report.png",
                image_paths=sequence.image_paths,
                frame_indices=sequence.frame_indices,
                target_masks=target_masks[int(instance_id)],
                predictions=result["predictions"],
                aligned_recovery_index=result["aligned_recovery_index"],
                shuffled_recovery_index=result["shuffled_recovery_index"],
                output_size=config.output_size,
            )
        instance_metadata[str(instance_id)] = {
            "label": sequence.label,
            "policies": policy_metadata,
        }

    _write_csv(config.output_dir / "summary.csv", all_summary_rows)
    _write_csv(config.output_dir / "frame_metrics.csv", all_frame_rows)
    _write_csv(
        config.output_dir / "geometry_gate_diagnostics.csv",
        all_gate_rows,
    )
    _write_csv(
        config.output_dir / "candidate_diagnostics.csv",
        all_candidate_rows,
    )
    _write_csv(
        config.output_dir / "candidate_screening.csv",
        all_candidate_screening_rows,
    )
    threshold_sweep_rows = _summarize_threshold_sweep(
        all_candidate_screening_rows
    )
    _write_csv(
        config.output_dir / "threshold_sweep.csv",
        threshold_sweep_rows,
    )
    _write_csv(
        config.output_dir / "pointcloud_summary.csv",
        all_point_cloud_summary_rows,
    )
    _write_csv(
        config.output_dir / "pointcloud_frame_metrics.csv",
        all_point_cloud_frame_rows,
    )
    _write_csv(
        config.output_dir / "map_quality.csv",
        all_map_quality_rows,
    )
    metadata = {
        "experiment": "causal_geometry_recovery_memory_writeback_ablation",
        "scene_id": config.scene_id,
        "frame_indices": list(config.frame_indices),
        "instance_ids": [int(value) for value in instance_ids],
        "modes": list(MODES),
        "event_policies": list(event_policies),
        "scheduled_probe_sequence_index": probe_sequence_index,
        "scheduled_probe_frame_index": shared.frame_indices[
            probe_sequence_index
        ],
        "reference_policy": (
            "explicit_sequence_index"
            if reference_sequence_index is not None
            else "earliest_visible_frame_per_instance"
        ),
        "aligned_pair_shares_exact_recovery_mask": True,
        "later_gt_used_for_selection": (
            "only oracle_candidate and oracle_mask upper bounds"
        ),
        "shuffled_control": (
            "reference geometry fixed; non-reference StreamVGGT outputs are "
            "cyclically permuted while RGB and SAM3 candidates stay fixed"
        ),
        "point_cloud": {
            "coordinate_system": "streamvggt_point_head_native",
            "reference_mask": "initialization_gt",
            "later_masks": "branch_tracking_predictions",
            "confidence_threshold": (
                config.point_cloud_confidence_threshold
            ),
            "max_points_per_export": config.point_cloud_max_points,
            "icp_used": False,
            "gt_geometry_used": False,
        },
        "map_quality_evaluation": {
            "enabled": map_evaluation_context is not None,
            "alignment": "fixed_reference_frame_sim3",
            "gt_geometry_role": "evaluation_only",
            "metric_max_points": config.map_metric_max_points,
            "metric_thresholds": list(config.map_metric_thresholds),
            "errors": map_evaluation_errors,
        },
        "threshold_sweep": {
            "role": "posthoc_diagnostic_only",
            "tracker_score_thresholds": [0.30, 0.50, 0.70],
            "tracker_geometry_coverage_thresholds": [0.10, 0.25, 0.50],
            "candidate_support_coverage_thresholds": [0.25, 0.50, 0.75],
        },
        "instances": instance_metadata,
    }
    with (config.output_dir / "metadata.json").open("w", encoding="utf8") as handle:
        json.dump(metadata, handle, indent=2, allow_nan=True)
    print(f"summary: {config.output_dir / 'summary.csv'}")
    print(f"frames: {config.output_dir / 'frame_metrics.csv'}")
    print(
        "gates: "
        f"{config.output_dir / 'geometry_gate_diagnostics.csv'}"
    )
    print(f"candidates: {config.output_dir / 'candidate_diagnostics.csv'}")
    print(
        "candidate screening: "
        f"{config.output_dir / 'candidate_screening.csv'}"
    )
    print(f"threshold sweep: {config.output_dir / 'threshold_sweep.csv'}")
    print(f"map quality: {config.output_dir / 'map_quality.csv'}")
    print(
        "point clouds: "
        f"{config.output_dir / 'instance_<id>' / '<event_policy>' / 'pointclouds'}"
    )


def _run_instance(
    config: ExperimentConfig,
    *,
    sequence,
    target_masks: torch.Tensor,
    geometry: GeometrySequence,
    sam3: SAM3Wrapper,
    original: TrackingSequence,
    candidate_cache: dict[int, list[SAM3MaskCandidate]],
    event_policy: str,
    forced_recovery_index: int | None,
) -> dict:
    aligned = _prepare_recovery(
        config,
        sequence=sequence,
        target_masks=target_masks,
        original=original,
        geometry=geometry,
        geometry_alignment="aligned",
        geometry_permutation=tuple(range(len(sequence.frame_indices))),
        sam3=sam3,
        candidate_cache=candidate_cache,
        map_update_policy="joint_reliable",
        event_policy=event_policy,
        forced_recovery_index=forced_recovery_index,
    )
    if (
        event_policy == "scheduled_probe"
        and aligned["recovery_mask"] is None
    ):
        requested_index = forced_recovery_index
        for alternative_index in range(
            int(sequence.reference_frame_idx) + 1,
            len(sequence.frame_indices) - 1,
        ):
            if alternative_index == requested_index:
                continue
            alternative = _prepare_recovery(
                config,
                sequence=sequence,
                target_masks=target_masks,
                original=original,
                geometry=geometry,
                geometry_alignment="aligned",
                geometry_permutation=tuple(
                    range(len(sequence.frame_indices))
                ),
                sam3=sam3,
                candidate_cache=candidate_cache,
                map_update_policy="joint_reliable",
                event_policy=event_policy,
                forced_recovery_index=alternative_index,
                allow_natural_trigger=False,
            )
            if alternative["recovery_mask"] is not None:
                alternative["scheduled_probe_requested_index"] = (
                    requested_index
                )
                alternative["scheduled_probe_used_fallback"] = True
                alternative["scheduled_probe_fallback_reason"] = (
                    aligned["reason"]
                )
                aligned = alternative
                break

    # Every geometry control is evaluated at the primary aligned event frame.
    # This prevents different trigger times from becoming a hidden variable.
    control_index = aligned["recovery_index"]
    reference_geometry = _prepare_recovery(
        config,
        sequence=sequence,
        target_masks=target_masks,
        original=original,
        geometry=geometry,
        geometry_alignment="aligned_reference_only",
        geometry_permutation=tuple(range(len(sequence.frame_indices))),
        sam3=sam3,
        candidate_cache=candidate_cache,
        map_update_policy="reference_only",
        event_policy=event_policy,
        forced_recovery_index=control_index,
        allow_natural_trigger=False,
    )

    permutation = _shuffled_permutation(
        len(sequence.frame_indices),
        reference_index=sequence.reference_frame_idx,
    )
    shuffled_geometry = _permute_geometry(geometry, permutation)
    shuffled = _prepare_recovery(
        config,
        sequence=sequence,
        target_masks=target_masks,
        original=original,
        geometry=shuffled_geometry,
        geometry_alignment="shuffled",
        geometry_permutation=permutation,
        sam3=sam3,
        candidate_cache=candidate_cache,
        map_update_policy="joint_reliable",
        event_policy=event_policy,
        forced_recovery_index=control_index,
        allow_natural_trigger=False,
    )

    predictions: dict[str, TrackingSequence] = {"original": original}
    applied = {mode: False for mode in MODES}
    predictions["geometry_recovery_no_memory"] = _current_only_tracking(
        original,
        recovery_index=aligned["recovery_index"],
        recovery_mask=aligned["recovery_mask"],
    )
    applied["geometry_recovery_no_memory"] = (
        aligned["recovery_mask"] is not None
    )
    predictions["reference_geometry_same_id_memory"], applied[
        "reference_geometry_same_id_memory"
    ] = _writeback_or_original(
        config,
        sequence=sequence,
        target_masks=target_masks,
        sam3=sam3,
        original=original,
        recovery_index=reference_geometry["recovery_index"],
        recovery_mask=reference_geometry["recovery_mask"],
    )
    predictions["geometry_recovery_same_id_memory"], applied[
        "geometry_recovery_same_id_memory"
    ] = _writeback_or_original(
        config,
        sequence=sequence,
        target_masks=target_masks,
        sam3=sam3,
        original=original,
        recovery_index=aligned["recovery_index"],
        recovery_mask=aligned["recovery_mask"],
    )
    predictions["shuffled_geometry_same_id_memory"], applied[
        "shuffled_geometry_same_id_memory"
    ] = _writeback_or_original(
        config,
        sequence=sequence,
        target_masks=target_masks,
        sam3=sam3,
        original=original,
        recovery_index=shuffled["recovery_index"],
        recovery_mask=shuffled["recovery_mask"],
    )
    predictions["oracle_candidate_same_id_memory"], applied[
        "oracle_candidate_same_id_memory"
    ] = _writeback_or_original(
        config,
        sequence=sequence,
        target_masks=target_masks,
        sam3=sam3,
        original=original,
        recovery_index=aligned["recovery_index"],
        recovery_mask=aligned["oracle_candidate_mask"],
    )
    oracle_mask = (
        target_masks[int(aligned["recovery_index"])]
        if aligned["recovery_index"] is not None
        and target_masks[int(aligned["recovery_index"])].any()
        else None
    )
    predictions["oracle_mask_same_id_memory"], applied[
        "oracle_mask_same_id_memory"
    ] = _writeback_or_original(
        config,
        sequence=sequence,
        target_masks=target_masks,
        sam3=sam3,
        original=original,
        recovery_index=aligned["recovery_index"],
        recovery_mask=oracle_mask,
    )

    aligned_memory = predictions["geometry_recovery_same_id_memory"]
    aligned_no_memory = predictions["geometry_recovery_no_memory"]
    if applied["geometry_recovery_same_id_memory"]:
        recovery_index = int(aligned["recovery_index"])
        if not torch.equal(
            aligned_no_memory.masks[recovery_index],
            aligned_memory.masks[recovery_index],
        ):
            raise RuntimeError(
                "The aligned no-memory and memory branches did not use the "
                "same intervention mask."
            )

    original_metrics = summarize_masks(
        original.masks,
        target_masks,
        reference_frame_idx=sequence.reference_frame_idx,
    )
    aligned_index = aligned["recovery_index"]
    mode_events = {
        "original": aligned,
        "geometry_recovery_no_memory": aligned,
        "reference_geometry_same_id_memory": reference_geometry,
        "geometry_recovery_same_id_memory": aligned,
        "shuffled_geometry_same_id_memory": shuffled,
        "oracle_candidate_same_id_memory": aligned,
        "oracle_mask_same_id_memory": aligned,
    }
    selectors = {
        "original": "none",
        "geometry_recovery_no_memory": "geometry",
        "reference_geometry_same_id_memory": "reference_geometry",
        "geometry_recovery_same_id_memory": "geometry",
        "shuffled_geometry_same_id_memory": "shuffled_geometry",
        "oracle_candidate_same_id_memory": "gt_oracle_text_candidate",
        "oracle_mask_same_id_memory": "gt_mask_upper_bound",
    }
    summary_rows = []
    for mode in MODES:
        event = mode_events[mode]
        tracking = predictions[mode]
        metrics = summarize_masks(
            tracking.masks,
            target_masks,
            reference_frame_idx=sequence.reference_frame_idx,
        )
        post = summarize_visible_after(
            tracking.masks,
            target_masks,
            recovery_frame_idx=event["recovery_index"],
        )
        summary_rows.append(
            {
                "scene_id": sequence.scene_id,
                "instance_id": int(sequence.instance_id),
                "instance_label": sequence.label,
                "event_policy": event_policy,
                "scheduled_probe_requested_sequence_index": aligned[
                    "scheduled_probe_requested_index"
                ],
                "scheduled_probe_used_fallback": int(
                    aligned["scheduled_probe_used_fallback"]
                ),
                "frame_indices": " ".join(
                    str(value) for value in sequence.frame_indices
                ),
                "mode": mode,
                "candidate_selector": selectors[mode],
                "map_update_policy": event["map_update_policy"],
                "geometry_alignment": event["geometry_alignment"],
                "geometry_permutation": " ".join(
                    str(value) for value in event["geometry_permutation"]
                ),
                "reference_sequence_index": sequence.reference_frame_idx,
                "reference_frame_index": sequence.frame_indices[
                    sequence.reference_frame_idx
                ],
                **metrics,
                "cross_iou_gain_over_original": (
                    metrics["cross_view_iou"]
                    - original_metrics["cross_view_iou"]
                ),
                "visible_miss_rate": _visible_miss_rate(
                    tracking.masks, target_masks
                ),
                "recovery_requested": int(event["recovery_requested"]),
                "natural_gate_request_count": event[
                    "natural_gate_request_count"
                ],
                "recovery_applied": int(applied[mode]),
                "recovery_sequence_index": event["recovery_index"],
                "recovery_frame_index": event["recovery_frame_index"],
                "recovery_reason": event["reason"],
                "candidate_count": event["candidate_count"],
                "selected_support_coverage": event[
                    "selected_support_coverage"
                ],
                "selected_candidate_gt_iou": event[
                    "selected_candidate_gt_iou"
                ],
                "oracle_candidate_gt_iou": event[
                    "oracle_candidate_gt_iou"
                ],
                "geometry_selected_oracle": event[
                    "geometry_selected_oracle"
                ],
                "recovery_mask_iou": (
                    (
                        1.0
                        if mode == "oracle_mask_same_id_memory"
                        else (
                            event["oracle_candidate_gt_iou"]
                            if mode == "oracle_candidate_same_id_memory"
                            else event["selected_candidate_gt_iou"]
                        )
                    )
                    if applied[mode]
                    else float("nan")
                ),
                "post_recovery_iou": post["iou"],
                "post_recovery_recall": post["recall"],
                "post_recovery_visible_frames": post["visible_frames"],
                "post_recovery_miss_rate": _visible_miss_rate(
                    tracking.masks,
                    target_masks,
                    after_index=event["recovery_index"],
                ),
                "persistent_obj_id": tracking.selected_obj_id,
                "same_obj_id_as_original": int(
                    tracking.selected_obj_id == original.selected_obj_id
                ),
                "aligned_pair_exact_same_recovery_mask": int(
                    mode
                    in {
                        "geometry_recovery_no_memory",
                        "geometry_recovery_same_id_memory",
                    }
                    and applied[mode]
                ),
            }
        )

    no_memory_row = next(
        row
        for row in summary_rows
        if row["mode"] == "geometry_recovery_no_memory"
    )
    for row in summary_rows:
        row["memory_cross_iou_gain_over_no_memory"] = (
            row["cross_view_iou"] - no_memory_row["cross_view_iou"]
            if row["mode"] == "geometry_recovery_same_id_memory"
            else 0.0
        )
        row["memory_post_iou_gain_over_no_memory"] = (
            row["post_recovery_iou"] - no_memory_row["post_recovery_iou"]
            if row["mode"] == "geometry_recovery_same_id_memory"
            else 0.0
        )
        row["aligned_gain_over_reference_map"] = 0.0
        row["aligned_gain_over_shuffled_geometry"] = 0.0
        row["oracle_candidate_headroom"] = 0.0
    by_mode = {row["mode"]: row for row in summary_rows}
    by_mode["geometry_recovery_same_id_memory"][
        "aligned_gain_over_reference_map"
    ] = (
        by_mode["geometry_recovery_same_id_memory"]["post_recovery_iou"]
        - by_mode["reference_geometry_same_id_memory"]["post_recovery_iou"]
    )
    by_mode["geometry_recovery_same_id_memory"][
        "aligned_gain_over_shuffled_geometry"
    ] = (
        by_mode["geometry_recovery_same_id_memory"]["post_recovery_iou"]
        - by_mode["shuffled_geometry_same_id_memory"]["post_recovery_iou"]
    )
    by_mode["geometry_recovery_same_id_memory"][
        "oracle_candidate_headroom"
    ] = (
        by_mode["oracle_candidate_same_id_memory"]["post_recovery_iou"]
        - by_mode["geometry_recovery_same_id_memory"]["post_recovery_iou"]
    )

    frame_rows = []
    for mode, tracking in predictions.items():
        event = mode_events[mode]
        for index, frame_index in enumerate(sequence.frame_indices):
            frame_rows.append(
                {
                    "scene_id": sequence.scene_id,
                    "instance_id": int(sequence.instance_id),
                    "instance_label": sequence.label,
                    "event_policy": event_policy,
                    "mode": mode,
                    "sequence_index": index,
                    "frame_index": frame_index,
                    "gt_visible": int(target_masks[index].any()),
                    "prediction_pixels": int(tracking.masks[index].sum()),
                    "sam3_score": float(tracking.scores[index]),
                    "target_pixels": int(target_masks[index].sum()),
                    "iou": binary_iou(
                        tracking.masks[index], target_masks[index]
                    ),
                    "missed_visible_instance": int(
                        target_masks[index].any()
                        and not tracking.masks[index].any()
                    ),
                    "is_recovery_frame": int(
                        event["recovery_index"] == index
                    ),
                    "after_recovery": int(
                        event["recovery_index"] is not None
                        and index > int(event["recovery_index"])
                    ),
                    "persistent_obj_id": tracking.selected_obj_id,
                }
            )

    return {
        "summary_rows": summary_rows,
        "frame_rows": frame_rows,
        "gate_rows": [
            *aligned["gate_rows"],
            *reference_geometry["gate_rows"],
            *shuffled["gate_rows"],
        ],
        "candidate_rows": [
            *aligned["candidate_rows"],
            *reference_geometry["candidate_rows"],
            *shuffled["candidate_rows"],
        ],
        "predictions": predictions,
        "aligned_recovery_index": aligned_index,
        "shuffled_recovery_index": shuffled["recovery_index"],
        "metadata": {
            "event_policy": event_policy,
            "reference_sequence_index": sequence.reference_frame_idx,
            "reference_frame_index": sequence.frame_indices[
                sequence.reference_frame_idx
            ],
            "original_obj_id": original.selected_obj_id,
            "aligned_joint_reliable": _event_metadata(aligned),
            "aligned_reference_only": _event_metadata(reference_geometry),
            "shuffled": _event_metadata(shuffled),
        },
    }


def _screen_candidate_frames(
    config: ExperimentConfig,
    *,
    sequence,
    target_masks: torch.Tensor,
    geometry: GeometrySequence,
    sam3: SAM3Wrapper,
    original: TrackingSequence,
    candidate_cache: dict[int, list[SAM3MaskCandidate]],
) -> dict[str, list[dict]]:
    """Cache and diagnose global-text candidates on every post-reference frame."""

    summary_rows = []
    candidate_rows = []
    identity = tuple(range(len(sequence.frame_indices)))
    for sequence_index in range(
        int(sequence.reference_frame_idx) + 1,
        len(sequence.frame_indices),
    ):
        event = _prepare_recovery(
            config,
            sequence=sequence,
            target_masks=target_masks,
            original=original,
            geometry=geometry,
            geometry_alignment="aligned",
            geometry_permutation=identity,
            sam3=sam3,
            candidate_cache=candidate_cache,
            map_update_policy="joint_reliable",
            event_policy="candidate_screening",
            forced_recovery_index=sequence_index,
            allow_natural_trigger=False,
        )
        candidate_rows.extend(event["candidate_rows"])
        summary_rows.append(
            {
                "scene_id": sequence.scene_id,
                "instance_id": int(sequence.instance_id),
                "instance_label": sequence.label,
                "sequence_index": sequence_index,
                "frame_index": sequence.frame_indices[sequence_index],
                "sam3_original_score": float(
                    original.scores[sequence_index]
                ),
                "sam3_original_iou": binary_iou(
                    original.masks[sequence_index],
                    target_masks[sequence_index],
                ),
                "sam3_original_pixels": int(
                    original.masks[sequence_index].sum()
                ),
                "geometry_candidate_accepted": event[
                    "geometry_candidate_accepted"
                ],
                "geometry_candidate_reason": event[
                    "geometry_candidate_reason"
                ],
                "tracker_geometry_coverage": event[
                    "tracker_geometry_coverage"
                ],
                "natural_gate_would_trigger": event[
                    "natural_gate_would_trigger"
                ],
                "map_updates_before_event": event[
                    "map_updates_before_event"
                ],
                "candidate_count": event["candidate_count"],
                "selected_support_coverage": event[
                    "selected_support_coverage"
                ],
                "selected_candidate_gt_iou": event[
                    "selected_candidate_gt_iou"
                ],
                "oracle_candidate_gt_iou": event[
                    "oracle_candidate_gt_iou"
                ],
                "geometry_selected_oracle": event[
                    "geometry_selected_oracle"
                ],
                "geometry_recovery_mask_accepted": int(
                    event["recovery_mask"] is not None
                ),
                "reason": event["reason"],
            }
        )
    return {
        "summary_rows": summary_rows,
        "candidate_rows": candidate_rows,
    }


def _summarize_threshold_sweep(
    screening_rows: list[dict],
) -> list[dict]:
    """Evaluate gate thresholds posthoc without rerunning either backbone."""

    groups: dict[tuple[str, object], list[dict]] = {}
    for row in screening_rows:
        key = (str(row["scene_id"]), int(row["instance_id"]))
        groups.setdefault(key, []).append(row)
    if screening_rows:
        groups[(str(screening_rows[0]["scene_id"]), "all")] = screening_rows

    output = []
    for (scene_id, instance_id), rows in groups.items():
        for score_threshold in (0.30, 0.50, 0.70):
            for geometry_threshold in (0.10, 0.25, 0.50):
                for candidate_threshold in (0.25, 0.50, 0.75):
                    failures = [
                        row
                        for row in rows
                        if float(row["sam3_original_iou"]) < 0.50
                    ]
                    triggered = []
                    applied = []
                    for row in rows:
                        geometry_accepted = bool(
                            row["geometry_candidate_accepted"]
                        )
                        tracker_weak = (
                            int(row["sam3_original_pixels"]) == 0
                            or float(row["sam3_original_score"])
                            < score_threshold
                            or (
                                geometry_accepted
                                and float(row["tracker_geometry_coverage"])
                                < geometry_threshold
                            )
                        )
                        if tracker_weak:
                            triggered.append(row)
                        support = float(row["selected_support_coverage"])
                        candidate_available = (
                            int(row["candidate_count"]) > 0
                            and np.isfinite(support)
                        )
                        if (
                            tracker_weak
                            and geometry_accepted
                            and candidate_available
                            and support >= candidate_threshold
                        ):
                            applied.append(row)
                    recovered_good = sum(
                        float(row["selected_candidate_gt_iou"]) >= 0.50
                        for row in applied
                    )
                    beneficial = sum(
                        float(row["selected_candidate_gt_iou"])
                        > float(row["sam3_original_iou"])
                        for row in applied
                    )
                    false_interventions = sum(
                        float(row["sam3_original_iou"]) >= 0.50
                        for row in applied
                    )
                    applied_failures = sum(
                        float(row["sam3_original_iou"]) < 0.50
                        for row in applied
                    )
                    mean_gain = (
                        float(
                            np.mean(
                                [
                                    float(row["selected_candidate_gt_iou"])
                                    - float(row["sam3_original_iou"])
                                    for row in applied
                                ]
                            )
                        )
                        if applied
                        else float("nan")
                    )
                    output.append(
                        {
                            "scene_id": scene_id,
                            "instance_id": instance_id,
                            "diagnostic_frames": len(rows),
                            "tracker_failure_frames_iou_lt_0_5": len(failures),
                            "tracker_score_threshold": score_threshold,
                            "tracker_geometry_coverage_threshold": (
                                geometry_threshold
                            ),
                            "candidate_support_coverage_threshold": (
                                candidate_threshold
                            ),
                            "gate_triggered_frames": len(triggered),
                            "recovery_applied_frames": len(applied),
                            "applied_tracker_failure_frames": applied_failures,
                            "good_recovery_frames_iou_ge_0_5": recovered_good,
                            "beneficial_recovery_frames": beneficial,
                            "false_intervention_frames": false_interventions,
                            "failure_trigger_recall": (
                                applied_failures / len(failures)
                                if failures
                                else 0.0
                            ),
                            "applied_trigger_precision": (
                                applied_failures / len(applied)
                                if applied
                                else 0.0
                            ),
                            "mean_selected_iou_gain_when_applied": mean_gain,
                            "gt_role": "posthoc_threshold_diagnostic_only",
                        }
                    )
    return output


def _current_only_tracking(
    original: TrackingSequence,
    *,
    recovery_index: int | None,
    recovery_mask: torch.Tensor | None,
) -> TrackingSequence:
    if recovery_index is None or recovery_mask is None:
        return original
    masks = original.masks.clone()
    scores = original.scores.clone()
    masks[int(recovery_index)] = recovery_mask.detach().cpu().bool()
    scores[int(recovery_index)] = 1.0
    return TrackingSequence(
        masks=masks,
        scores=scores,
        selected_obj_id=original.selected_obj_id,
    )


def _writeback_or_original(
    config: ExperimentConfig,
    *,
    sequence,
    target_masks: torch.Tensor,
    sam3: SAM3Wrapper,
    original: TrackingSequence,
    recovery_index: int | None,
    recovery_mask: torch.Tensor | None,
) -> tuple[TrackingSequence, bool]:
    if recovery_index is None or recovery_mask is None or not recovery_mask.any():
        return original, False
    tracking = sam3.track_with_recovery_mask_memory(
        sequence.image_paths,
        prompt=sequence.label,
        output_size=config.output_size,
        reference_frame_idx=sequence.reference_frame_idx,
        reference_mask=target_masks[sequence.reference_frame_idx],
        recovery_frame_idx=int(recovery_index),
        recovery_mask=recovery_mask,
    )
    if tracking.selected_obj_id != original.selected_obj_id:
        raise RuntimeError(
            "Same-ID writeback changed the persistent object ID: "
            f"{original.selected_obj_id} -> {tracking.selected_obj_id}."
        )
    return tracking, True


def _prepare_recovery(
    config: ExperimentConfig,
    *,
    sequence,
    target_masks: torch.Tensor,
    original: TrackingSequence,
    geometry: GeometrySequence,
    geometry_alignment: str,
    geometry_permutation: tuple[int, ...],
    sam3: SAM3Wrapper,
    candidate_cache: dict[int, list[SAM3MaskCandidate]],
    map_update_policy: str,
    event_policy: str,
    forced_recovery_index: int | None,
    allow_natural_trigger: bool = True,
) -> dict:
    mined = mine_recovery(
        config,
        sequence=sequence,
        target_masks=target_masks,
        original_masks=original.masks,
        original_scores=original.scores,
        geometry=geometry,
        map_update_policy=map_update_policy,
    )
    if forced_recovery_index is not None:
        recovery_index = int(forced_recovery_index)
        if not (
            int(sequence.reference_frame_idx)
            < recovery_index
            < len(sequence.frame_indices)
        ):
            raise ValueError(
                "Forced recovery index must be after the reference and inside "
                "the selected sequence."
            )
    elif allow_natural_trigger:
        recovery_index = _first_viable_natural_event(
            config,
            sequence=sequence,
            mined=mined,
            sam3=sam3,
            candidate_cache=candidate_cache,
        )
    else:
        recovery_index = None
    natural_gate_request_count = sum(
        int(row["use_correction"])
        for index, row in enumerate(mined["rows"])
        if index > int(sequence.reference_frame_idx)
        and index < len(sequence.frame_indices) - 1
    )
    base = {
        "event_policy": event_policy,
        "map_update_policy": map_update_policy,
        "scheduled_probe_requested_index": (
            int(forced_recovery_index)
            if event_policy == "scheduled_probe"
            and forced_recovery_index is not None
            else None
        ),
        "scheduled_probe_used_fallback": False,
        "geometry_alignment": geometry_alignment,
        "geometry_permutation": geometry_permutation,
        "recovery_requested": (
            forced_recovery_index is not None
            or natural_gate_request_count > 0
        ),
        "natural_gate_request_count": natural_gate_request_count,
        "recovery_index": recovery_index,
        "recovery_frame_index": (
            sequence.frame_indices[recovery_index]
            if recovery_index is not None
            else None
        ),
        "recovery_mask": None,
        "oracle_candidate_mask": None,
        "candidate_count": 0,
        "selected_support_coverage": float("nan"),
        "selected_candidate_gt_iou": float("nan"),
        "oracle_candidate_gt_iou": float("nan"),
        "geometry_selected_oracle": 0,
        "geometry_candidate_accepted": (
            int(mined["candidates"][recovery_index].accepted)
            if recovery_index is not None
            else 0
        ),
        "geometry_candidate_reason": (
            mined["candidates"][recovery_index].reason
            if recovery_index is not None
            else "no event"
        ),
        "tracker_geometry_coverage": (
            mined["rows"][recovery_index]["tracker_geometry_coverage"]
            if recovery_index is not None
            else float("nan")
        ),
        "natural_gate_would_trigger": (
            int(mined["rows"][recovery_index]["use_correction"])
            if recovery_index is not None
            else 0
        ),
        "candidate_rows": [],
        "gate_rows": [
            {
                "scene_id": sequence.scene_id,
                "instance_id": int(sequence.instance_id),
                "instance_label": sequence.label,
                "event_policy": event_policy,
                "geometry_alignment": geometry_alignment,
                "geometry_permutation": " ".join(
                    str(value) for value in geometry_permutation
                ),
                **row,
            }
            for row in mined["rows"]
        ],
        "map_updates_before_event": (
            sum(
                int(row["map_updated"])
                for row in mined["rows"][:recovery_index]
            )
            if recovery_index is not None
            else sum(int(row["map_updated"]) for row in mined["rows"])
        ),
        "reason": (
            (
                "joint gate requested recovery but no full candidate passed"
                if natural_gate_request_count > 0
                else "no joint-gate recovery event"
            )
            if allow_natural_trigger
            else "primary aligned policy produced no event"
        ),
    }
    if recovery_index is None:
        return base

    candidates = candidate_cache.get(recovery_index)
    if candidates is None:
        candidates = sam3.propose_text_masks(
            sequence.image_paths[recovery_index],
            prompt=sequence.label,
            output_size=config.output_size,
        )
        candidate_cache[recovery_index] = candidates
    base["candidate_count"] = len(candidates)
    if not candidates:
        base["reason"] = "SAM3 global-text query produced no candidate mask"
        return base

    geometry_candidate = mined["candidates"][recovery_index]
    mining_row = mined["rows"][recovery_index]
    supported = geometry_candidate.supported_mask.bool()
    projected = geometry_candidate.projected_mask.bool()
    coarse_box = geometry_candidate.mask.bool()
    rows = []
    ranking = []
    gt_ious = []
    for candidate_index, candidate in enumerate(candidates):
        mask = candidate.mask.bool()
        support_coverage = _coverage(supported, mask)
        projected_coverage = _coverage(projected, mask)
        box_iou = binary_iou(mask, coarse_box)
        gt_iou = binary_iou(mask, target_masks[recovery_index])
        gt_ious.append(gt_iou)
        ranking.append(
            (
                support_coverage,
                projected_coverage,
                box_iou,
                float(candidate.score),
                -int(candidate.obj_id),
                candidate_index,
            )
        )
        rows.append(
            {
                "scene_id": sequence.scene_id,
                "instance_id": int(sequence.instance_id),
                "instance_label": sequence.label,
                "event_policy": event_policy,
                "map_update_policy": map_update_policy,
                "geometry_alignment": geometry_alignment,
                "geometry_permutation": " ".join(
                    str(value) for value in geometry_permutation
                ),
                "sequence_index": recovery_index,
                "frame_index": sequence.frame_indices[recovery_index],
                "candidate_index": candidate_index,
                "temporary_obj_id": candidate.obj_id,
                "sam3_score": candidate.score,
                "candidate_pixels": int(mask.sum()),
                "support_coverage": support_coverage,
                "projected_coverage": projected_coverage,
                "coarse_box_iou": box_iou,
                "geometry_candidate_accepted": int(
                    geometry_candidate.accepted
                ),
                "geometry_candidate_reason": geometry_candidate.reason,
                "tracker_geometry_coverage": mining_row[
                    "tracker_geometry_coverage"
                ],
                "natural_gate_would_trigger": int(
                    mining_row["use_correction"]
                ),
                "candidate_gt_iou": gt_iou,
                "selected": 0,
                "oracle": 0,
            }
        )
    selected_index = int(max(ranking)[-1])
    oracle_iou = max(gt_ious)
    oracle_indices = {
        index
        for index, value in enumerate(gt_ious)
        if abs(value - oracle_iou) <= 1e-8
    }
    rows[selected_index]["selected"] = 1
    for index in oracle_indices:
        rows[index]["oracle"] = 1
    base["candidate_rows"] = rows
    selected = candidates[selected_index]
    oracle_index = min(oracle_indices)
    base["oracle_candidate_mask"] = (
        candidates[oracle_index].mask.detach().cpu().bool()
    )
    support_coverage = float(rows[selected_index]["support_coverage"])
    base.update(
        {
            "selected_support_coverage": support_coverage,
            "selected_candidate_gt_iou": gt_ious[selected_index],
            "oracle_candidate_gt_iou": oracle_iou,
            "geometry_selected_oracle": int(selected_index in oracle_indices),
        }
    )
    if not selected.mask.any():
        base["reason"] = "selected SAM3 candidate mask is empty"
        return base
    if not geometry_candidate.accepted or not supported.any():
        base["reason"] = (
            "scheduled candidate diagnostic only; geometry rejected: "
            f"{geometry_candidate.reason}"
        )
        return base
    if support_coverage < float(
        config.recovery_min_support_coverage
    ):
        base["reason"] = (
            "selected full mask failed geometry support coverage: "
            f"{support_coverage:.4f} < "
            f"{config.recovery_min_support_coverage:.4f}"
        )
        return base
    base["recovery_mask"] = selected.mask.detach().cpu().bool()
    base["reason"] = (
        f"accepted full SAM3 candidate selected by {geometry_alignment}"
    )
    return base


def _first_viable_natural_event(
    config: ExperimentConfig,
    *,
    sequence,
    mined: dict,
    sam3: SAM3Wrapper,
    candidate_cache: dict[int, list[SAM3MaskCandidate]],
) -> int | None:
    """Return the first gate event with a deployable full-mask candidate."""

    for index, row in enumerate(mined["rows"]):
        geometry_candidate = mined["candidates"][index]
        if not (
            index > int(sequence.reference_frame_idx)
            and index < len(sequence.frame_indices) - 1
            and row["use_correction"]
            and geometry_candidate.accepted
            and geometry_candidate.supported_mask.any()
        ):
            continue
        candidates = candidate_cache.get(index)
        if candidates is None:
            candidates = sam3.propose_text_masks(
                sequence.image_paths[index],
                prompt=sequence.label,
                output_size=config.output_size,
            )
            candidate_cache[index] = candidates
        if not candidates:
            continue
        supported = geometry_candidate.supported_mask.bool()
        projected = geometry_candidate.projected_mask.bool()
        coarse_box = geometry_candidate.mask.bool()
        ranking = [
            (
                _coverage(supported, candidate.mask),
                _coverage(projected, candidate.mask),
                binary_iou(candidate.mask, coarse_box),
                float(candidate.score),
                -int(candidate.obj_id),
                candidate_index,
            )
            for candidate_index, candidate in enumerate(candidates)
        ]
        selected = candidates[int(max(ranking)[-1])]
        if (
            selected.mask.any()
            and _coverage(supported, selected.mask)
            >= config.recovery_min_support_coverage
        ):
            return index
    return None


def _shuffled_permutation(
    count: int,
    *,
    reference_index: int,
) -> tuple[int, ...]:
    if count < 3:
        raise ValueError("Shuffled geometry control requires at least three frames.")
    movable = [index for index in range(count) if index != int(reference_index)]
    rotated = movable[1:] + movable[:1]
    permutation = list(range(count))
    for destination, source in zip(movable, rotated):
        permutation[destination] = source
    return tuple(permutation)


def _permute_geometry(
    geometry: GeometrySequence,
    permutation: tuple[int, ...],
) -> GeometrySequence:
    def permute_tensor(value):
        if value is None:
            return None
        indices = torch.tensor(permutation, dtype=torch.long, device=value.device)
        return value.index_select(0, indices)

    return GeometrySequence(
        world_points=permute_tensor(geometry.world_points),
        confidence=permute_tensor(geometry.confidence),
        world_to_camera=permute_tensor(geometry.world_to_camera),
        intrinsics=permute_tensor(geometry.intrinsics),
        processed_size=geometry.processed_size,
        source_sizes=tuple(geometry.source_sizes[index] for index in permutation),
    )


def _coverage(evidence: torch.Tensor, mask: torch.Tensor) -> float:
    evidence = evidence.detach().cpu().bool()
    mask = mask.detach().cpu().bool()
    denominator = int(evidence.sum())
    if denominator == 0:
        return 0.0
    return float((evidence & mask).sum()) / denominator


def _visible_miss_rate(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    after_index: int | None = None,
) -> float:
    visible = target.flatten(1).any(dim=1)
    if after_index is not None:
        visible &= torch.arange(len(target)) > int(after_index)
    if not visible.any():
        return 0.0
    predicted = prediction.flatten(1).any(dim=1)
    return float((~predicted[visible]).float().mean())


def _event_metadata(event: dict) -> dict:
    return {
        key: value
        for key, value in event.items()
        if key not in {
            "recovery_mask",
            "oracle_candidate_mask",
            "candidate_rows",
            "gate_rows",
        }
    }


def _save_report(
    path: Path,
    *,
    image_paths,
    frame_indices,
    target_masks: torch.Tensor,
    predictions: dict[str, TrackingSequence],
    aligned_recovery_index: int | None,
    shuffled_recovery_index: int | None,
    output_size: tuple[int, int],
) -> None:
    height, width = output_size
    header = 34
    columns = 2 + len(MODES)
    canvas = Image.new(
        "RGB",
        (columns * width, len(image_paths) * (height + header)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    labels = ("RGB", "GT", *MODES)
    colors = {
        "GT": (20, 220, 70),
        "original": (220, 60, 60),
        "geometry_recovery_no_memory": (245, 170, 30),
        "reference_geometry_same_id_memory": (20, 155, 150),
        "geometry_recovery_same_id_memory": (45, 110, 255),
        "shuffled_geometry_same_id_memory": (170, 70, 210),
        "oracle_candidate_same_id_memory": (20, 190, 230),
        "oracle_mask_same_id_memory": (30, 180, 70),
    }
    for row, image_path in enumerate(image_paths):
        with Image.open(image_path) as source:
            rgb = source.convert("RGB").resize(
                (width, height), Image.Resampling.BILINEAR
            )
        panels = [rgb, _overlay(rgb, target_masks[row], colors["GT"])]
        panels.extend(
            _overlay(rgb, predictions[mode].masks[row], colors[mode])
            for mode in MODES
        )
        top = row * (height + header)
        for column, (label, panel) in enumerate(zip(labels, panels)):
            canvas.paste(panel, (column * width, top + header))
            suffix = f" frame={frame_indices[row]}"
            if label in predictions:
                suffix += (
                    f" IoU={binary_iou(predictions[label].masks[row], target_masks[row]):.3f}"
                )
            if row == aligned_recovery_index and label in {
                "geometry_recovery_no_memory",
                "geometry_recovery_same_id_memory",
            }:
                suffix += " RECOVERY"
            if (
                row == shuffled_recovery_index
                and label == "shuffled_geometry_same_id_memory"
            ):
                suffix += " SHUFFLED-RECOVERY"
            draw.text(
                (column * width + 5, top + 7),
                label + suffix,
                fill=colors.get(label, (0, 0, 0)),
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _overlay(
    image: Image.Image,
    mask: torch.Tensor,
    color: tuple[int, int, int],
) -> Image.Image:
    array = np.asarray(image).copy()
    selected = mask.detach().cpu().numpy().astype(bool)
    if selected.any():
        array[selected] = (
            0.45 * array[selected] + 0.55 * np.asarray(color)
        ).astype(np.uint8)
    return Image.fromarray(array)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def _validate_shared_sequence(sequences: dict[int, object]) -> None:
    values = list(sequences.values())
    first = values[0]
    for sequence in values[1:]:
        if sequence.frame_indices != first.frame_indices:
            raise RuntimeError("All instances must use the same frame indices.")
        if [str(path) for path in sequence.image_paths] != [
            str(path) for path in first.image_paths
        ]:
            raise RuntimeError("All instances must use the same RGB sequence.")


def _unique_ids(values: Sequence[int]) -> tuple[int, ...]:
    result = []
    for value in values:
        value = int(value)
        if value not in result:
            result.append(value)
    if not result:
        raise ValueError("At least one instance ID is required.")
    return tuple(result)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="streaming_couping/configs/default.yaml",
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--scene-id")
    parser.add_argument("--instance-ids", type=int, nargs="+", required=True)
    parser.add_argument("--frame-indices", type=int, nargs="+")
    parser.add_argument(
        "--reference-sequence-index",
        type=int,
        help=(
            "Use one explicit reference position for every instance. By default "
            "each instance uses its earliest visible selected frame."
        ),
    )
    parser.add_argument(
        "--event-policies",
        nargs="+",
        choices=EVENT_POLICIES,
        default=list(EVENT_POLICIES),
        help=(
            "natural_joint_gate tests the deployable trigger; "
            "scheduled_probe forces candidate evaluation at one fixed frame."
        ),
    )
    parser.add_argument(
        "--probe-sequence-index",
        type=int,
        default=4,
        help=(
            "Selected-sequence position used by scheduled_probe. It must have "
            "at least one future frame."
        ),
    )
    parser.add_argument("--sam3-device")
    parser.add_argument("--geometry-device")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    main()
