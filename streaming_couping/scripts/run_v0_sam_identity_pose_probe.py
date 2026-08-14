#!/usr/bin/env python3
"""Probe whether persistent SAM identity supplies a useful pose direction."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from typing import Any, Sequence

import torch
import yaml

from streaming_couping.src.external_repos import maybe_add_repo_to_path
from streaming_couping.src.config import load_config
from streaming_couping.src.learned_pose.baseline_runtime import (
    camera_centers,
    load_baseline_run_config,
    pose_metrics,
)
from streaming_couping.src.learned_pose.cache import (
    cache_path,
    load_feature_cache,
)
from streaming_couping.src.learned_pose.config import (
    ClipConfig,
    LearnedPoseConfig,
    load_learned_pose_config,
)
from streaming_couping.src.sam_identity_pose_probe import (
    PointBank,
    ProjectionGroup,
    erode_mask,
    evaluate_projection_groups,
    fixed_pose_candidates,
    masked_weighted_points,
    shift_mask,
)
from streaming_couping.src.semantic_map import normalize_confidence


REVISION = "v0_causal_sam_identity_fixed_direction_pose_probe_r1"
BRANCHES = (
    "global_geometry",
    "foreground_union",
    "correct_persistent_id",
    "shuffled_persistent_id",
    "shifted_mask_control",
)


@dataclass(frozen=True)
class ProbeRun:
    source_path: Path
    output_dir: Path
    confidence_threshold: float
    track_score_threshold: float
    mask_erosion_radius: int
    min_mask_pixels: int
    min_history_observations: int
    min_history_points: int
    points_per_instance: int
    points_per_frame_write: int
    instance_max_points: int
    global_max_points: int
    relative_depth_cap: float
    mask_miss_weight: float
    min_loss_decrease: float
    rotation_step_degrees: float
    translation_step_scene_fraction: float
    shifted_mask_x_fraction: float
    shifted_mask_y_fraction: float


def main() -> None:
    args = _parse_args()
    data = load_learned_pose_config(args.config)
    baseline = load_baseline_run_config(args.config)
    run = _load_run(args.config)
    if args.output_dir:
        run = replace(
            run,
            output_dir=Path(args.output_dir).expanduser().resolve(),
        )
    clip = _find_clip(data, baseline.clip_name)
    source_cache = cache_path(data, clip)
    payload = load_feature_cache(source_cache)
    qk_path = baseline.qk_pose_output
    qk = torch.load(qk_path, map_location="cpu", weights_only=False)
    _validate_inputs(payload, qk=qk, clip=clip)

    recovery = load_config(data.recovery_config)
    maybe_add_repo_to_path(recovery.streamvggt_repo)
    from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri

    image_size = tuple(int(value) for value in payload["image_size"])
    raw_pose_batch, raw_intrinsics_batch = pose_encoding_to_extri_intri(
        payload["baseline_pose_encoding"].unsqueeze(0).float(),
        image_size_hw=image_size,
    )
    artifact_raw_pose = qk["raw_world_to_camera"].detach().float().cpu()
    decoded_raw_pose = raw_pose_batch.detach().float().cpu()
    if not torch.allclose(
        artifact_raw_pose,
        decoded_raw_pose,
        atol=2e-5,
        rtol=1e-5,
    ):
        difference = float((artifact_raw_pose - decoded_raw_pose).abs().max())
        raise RuntimeError(
            "QK artifact was produced from a different raw V0 pose; "
            f"maximum absolute difference={difference}."
        )
    generation = _candidate_generation_payload(
        payload,
        qk=qk,
        intrinsics=raw_intrinsics_batch[0].detach().float().cpu(),
    )
    candidates, diagnostic_rows = _generate_candidates(generation, run=run)
    # GT is decoded only after every branch has frozen its candidate sequence.
    target_pose, _ = pose_encoding_to_extri_intri(
        payload["target_pose_encoding"].unsqueeze(0).float(),
        image_size_hw=image_size,
    )
    result = _score_and_write(
        payload=payload,
        qk=qk,
        candidates=candidates,
        diagnostic_rows=diagnostic_rows,
        target_pose=target_pose.detach().float().cpu(),
        cache_path_value=source_cache,
        qk_path=qk_path,
        run=run,
    )
    print(f"V0 SAM identity pose probe result={result}")


def _candidate_generation_payload(
    payload: dict[str, Any],
    *,
    qk: dict[str, Any],
    intrinsics: torch.Tensor,
) -> dict[str, Any]:
    """Expose only causal deployment fields before candidates are frozen."""

    output = {
        "frame_indices": tuple(int(value) for value in payload["frame_indices"]),
        "qk_world_to_camera": qk["selected_world_to_camera"].detach().float().cpu()[0],
        "intrinsics": intrinsics.detach().float().cpu(),
        "world_points": payload["baseline_world_points"].detach().float().cpu(),
        "world_confidence": payload["baseline_world_confidence"].detach().float().cpu(),
        "depth": payload["baseline_depth"].detach().float().cpu(),
        "depth_confidence": payload["baseline_depth_confidence"].detach().float().cpu(),
        "tracking_masks": payload["tracking_masks_stream"].detach().bool().cpu(),
        "tracking_scores": payload["tracking_scores"].detach().float().cpu(),
        "scene_scale": float(payload["scene_scale"]),
        "instance_ids": tuple(int(value) for value in payload["instance_ids"]),
        "sam_track_ids": tuple(int(value) for value in payload["sam_track_ids"]),
        "sam_track_prompts": tuple(str(value) for value in payload["sam_track_prompts"]),
    }
    forbidden = {
        "target_pose_encoding",
        "target_world_to_camera",
        "target_world_points",
        "target_depth",
    }
    if forbidden.intersection(output):
        raise RuntimeError("SAM identity candidate generation contains GT.")
    return output


def _generate_candidates(
    data: dict[str, Any],
    *,
    run: ProbeRun,
) -> tuple[dict[str, torch.Tensor], list[dict[str, object]]]:
    """Generate every branch before target pose is introduced."""

    frames = data["frame_indices"]
    qk_pose = data["qk_world_to_camera"]
    intrinsics = data["intrinsics"]
    points = data["world_points"]
    point_confidence = normalize_confidence(_scalar(data["world_confidence"]))
    depth = _scalar(data["depth"])
    depth_confidence = normalize_confidence(_scalar(data["depth_confidence"]))
    masks = data["tracking_masks"]
    scores = data["tracking_scores"]
    sequence, instances, height, width = masks.shape
    if qk_pose.shape != (sequence, 3, 4):
        raise ValueError("QK pose shape differs from SAM stream.")
    if points.shape != (sequence, height, width, 3):
        raise ValueError("World pointmap shape differs from SAM output masks.")
    if depth.shape != (sequence, height, width):
        raise ValueError("Raw depth shape differs from SAM output masks.")
    if scores.shape != (sequence, instances):
        raise ValueError("Tracking score shape differs from SAM masks.")

    selected = {
        branch: qk_pose.detach().clone()
        for branch in BRANCHES
    }
    instance_banks = [PointBank.empty() for _ in range(instances)]
    global_bank = PointBank.empty()
    rows: list[dict[str, object]] = []
    translation_step = max(
        float(data["scene_scale"]) * run.translation_step_scene_fraction,
        1e-6,
    )
    shift_x = max(1, round(width * run.shifted_mask_x_fraction))
    shift_y = max(1, round(height * run.shifted_mask_y_fraction))

    for frame in range(sequence):
        current_masks = [
            erode_mask(masks[frame, slot], run.mask_erosion_radius)
            for slot in range(instances)
        ]
        eligible = [
            slot
            for slot in range(instances)
            if float(scores[frame, slot]) >= run.track_score_threshold
            and int(current_masks[slot].sum()) >= run.min_mask_pixels
            and instance_banks[slot].observations >= run.min_history_observations
            and instance_banks[slot].points.shape[0] >= run.min_history_points
        ]
        frame_groups, reason = _build_equal_count_groups(
            eligible=eligible,
            current_masks=current_masks,
            instance_banks=instance_banks,
            global_bank=global_bank,
            points_per_instance=run.points_per_instance,
            min_history_points=run.min_history_points,
            shift_y=shift_y,
            shift_x=shift_x,
        )
        if frame_groups is None:
            for branch in BRANCHES:
                rows.append(
                    _inactive_row(
                        branch=branch,
                        sequence_index=frame,
                        frame_index=frames[frame],
                        eligible_slots=eligible,
                        reason=reason,
                    )
                )
        else:
            pose_candidates = fixed_pose_candidates(
                qk_pose[frame],
                rotation_step_degrees=run.rotation_step_degrees,
                translation_step=translation_step,
            )
            for branch in BRANCHES:
                evaluations = []
                for candidate_name, candidate_pose in pose_candidates:
                    metrics = evaluate_projection_groups(
                        frame_groups[branch],
                        world_to_camera=candidate_pose,
                        intrinsics=intrinsics[frame],
                        depth=depth[frame],
                        normalized_depth_confidence=depth_confidence[frame],
                        confidence_threshold=run.confidence_threshold,
                        relative_depth_cap=run.relative_depth_cap,
                        mask_miss_weight=run.mask_miss_weight,
                    )
                    evaluations.append((candidate_name, candidate_pose, metrics))
                identity = evaluations[0]
                best = identity
                for candidate in evaluations[1:]:
                    if float(candidate[2]["loss"]) < float(best[2]["loss"]) - 1e-12:
                        best = candidate
                if (
                    float(identity[2]["loss"]) - float(best[2]["loss"])
                    < run.min_loss_decrease
                ):
                    best = identity
                selected[branch][frame] = best[1]
                rows.append(
                    {
                        "branch": branch,
                        "sequence_index": frame,
                        "frame_index": int(frames[frame]),
                        "active": 1,
                        "reason": "active_equal_count_causal_history",
                        "eligible_slots": _space(eligible),
                        "eligible_sam_track_ids": _space(
                            data["sam_track_ids"][slot] for slot in eligible
                        ),
                        "eligible_prompts": " ".join(
                            data["sam_track_prompts"][slot] for slot in eligible
                        ),
                        "source_points": int(best[2]["source_points"]),
                        "candidate_count": len(evaluations),
                        "selected_candidate": best[0],
                        "pose_modified": int(best[0] != "identity"),
                        "identity_loss": float(identity[2]["loss"]),
                        "selected_loss": float(best[2]["loss"]),
                        "loss_decrease": float(identity[2]["loss"])
                        - float(best[2]["loss"]),
                        "identity_depth_support_rate": float(
                            identity[2]["depth_support_rate"]
                        ),
                        "selected_depth_support_rate": float(
                            best[2]["depth_support_rate"]
                        ),
                        "identity_ownership_hit_rate": float(
                            identity[2]["ownership_hit_rate"]
                        ),
                        "selected_ownership_hit_rate": float(
                            best[2]["ownership_hit_rate"]
                        ),
                    }
                )
        _write_current_frame_to_memory(
            frame=frame,
            points=points,
            normalized_confidence=point_confidence,
            masks=current_masks,
            scores=scores,
            global_bank=global_bank,
            instance_banks=instance_banks,
            run=run,
        )
    return selected, rows


def _build_equal_count_groups(
    *,
    eligible: Sequence[int],
    current_masks: Sequence[torch.Tensor],
    instance_banks: Sequence[PointBank],
    global_bank: PointBank,
    points_per_instance: int,
    min_history_points: int,
    shift_y: int,
    shift_x: int,
) -> tuple[dict[str, list[ProjectionGroup]] | None, str]:
    slots = tuple(int(value) for value in eligible)
    if len(slots) < 2:
        return None, "fewer_than_two_mature_visible_instances"
    correct_groups: list[ProjectionGroup] = []
    shuffled_groups: list[ProjectionGroup] = []
    shifted_groups: list[ProjectionGroup] = []
    correct_points: list[torch.Tensor] = []
    for position, slot in enumerate(slots):
        wrong = slots[(position + 1) % len(slots)]
        count = min(
            int(points_per_instance),
            int(instance_banks[slot].points.shape[0]),
            int(instance_banks[wrong].points.shape[0]),
        )
        if count < int(min_history_points):
            return None, "equal_count_instance_support_below_minimum"
        same_points = instance_banks[slot].sample(count)
        wrong_points = instance_banks[wrong].sample(count)
        correct_points.append(same_points)
        correct_groups.append(
            ProjectionGroup(same_points, current_masks[slot])
        )
        shuffled_groups.append(
            ProjectionGroup(wrong_points, current_masks[slot])
        )
        shifted_groups.append(
            ProjectionGroup(
                same_points,
                shift_mask(
                    current_masks[slot],
                    shift_y=shift_y,
                    shift_x=shift_x,
                ),
            )
        )
    union_points = torch.cat(correct_points, dim=0)
    total = int(union_points.shape[0])
    if global_bank.points.shape[0] < total:
        return None, "global_history_below_equal_count_requirement"
    union_mask = torch.stack(
        [current_masks[slot] for slot in slots], dim=0
    ).any(dim=0)
    groups = {
        "global_geometry": [
            ProjectionGroup(global_bank.sample(total), None)
        ],
        "foreground_union": [ProjectionGroup(union_points, union_mask)],
        "correct_persistent_id": correct_groups,
        "shuffled_persistent_id": shuffled_groups,
        "shifted_mask_control": shifted_groups,
    }
    counts = {
        branch: sum(group.points.shape[0] for group in branch_groups)
        for branch, branch_groups in groups.items()
    }
    if len(set(counts.values())) != 1:
        raise RuntimeError(f"SAM pose controls have unequal point counts: {counts}.")
    return groups, "active"


def _write_current_frame_to_memory(
    *,
    frame: int,
    points: torch.Tensor,
    normalized_confidence: torch.Tensor,
    masks: Sequence[torch.Tensor],
    scores: torch.Tensor,
    global_bank: PointBank,
    instance_banks: Sequence[PointBank],
    run: ProbeRun,
) -> None:
    height, width = points.shape[1:3]
    all_pixels = torch.ones(height, width, dtype=torch.bool)
    global_points, global_weights = masked_weighted_points(
        points[frame],
        normalized_confidence[frame],
        all_pixels,
        confidence_threshold=run.confidence_threshold,
        max_points=run.points_per_frame_write,
    )
    global_bank.update(
        global_points,
        global_weights,
        max_points=run.global_max_points,
    )
    for slot, mask in enumerate(masks):
        if (
            float(scores[frame, slot]) < run.track_score_threshold
            or int(mask.sum()) < run.min_mask_pixels
        ):
            continue
        slot_points, slot_weights = masked_weighted_points(
            points[frame],
            normalized_confidence[frame],
            mask,
            confidence_threshold=run.confidence_threshold,
            max_points=run.points_per_frame_write,
        )
        instance_banks[slot].update(
            slot_points,
            slot_weights,
            max_points=run.instance_max_points,
        )


def _score_and_write(
    *,
    payload: dict[str, Any],
    qk: dict[str, Any],
    candidates: dict[str, torch.Tensor],
    diagnostic_rows: list[dict[str, object]],
    target_pose: torch.Tensor,
    cache_path_value: Path,
    qk_path: Path,
    run: ProbeRun,
) -> Path:
    """Introduce GT only after all branch poses have been selected."""

    frames = tuple(int(value) for value in payload["frame_indices"])
    reference = int(payload["reference_sequence_index"])
    evaluation_indices = [index for index in range(len(frames)) if index != reference]
    qk_pose = qk["selected_world_to_camera"].detach().float().cpu()
    qk_metrics = pose_metrics(
        qk_pose,
        target_pose,
        reference_index=reference,
        evaluation_indices=evaluation_indices,
    )
    qk_rotation, qk_center = _pose_frame_errors(qk_pose, target_pose)
    diagnostic_by_branch = {
        branch: [row for row in diagnostic_rows if row["branch"] == branch]
        for branch in BRANCHES
    }
    branch_rows: list[dict[str, object]] = []
    frame_rows: list[dict[str, object]] = []
    for branch in BRANCHES:
        pose = candidates[branch].unsqueeze(0)
        metrics = pose_metrics(
            pose,
            target_pose,
            reference_index=reference,
            evaluation_indices=evaluation_indices,
        )
        rotation, center = _pose_frame_errors(pose, target_pose)
        active_indices = [
            int(row["sequence_index"])
            for row in diagnostic_by_branch[branch]
            if int(row["active"])
        ]
        modified_indices = [
            int(row["sequence_index"])
            for row in diagnostic_by_branch[branch]
            if int(row["active"]) and int(row["pose_modified"])
        ]
        branch_rows.append(
            {
                "branch": branch,
                "active_frames": len(active_indices),
                "modified_frames": len(modified_indices),
                "center_error_native": float(metrics["center_error_native"]),
                "center_gain_vs_qk_percent": _gain(
                    qk_metrics["center_error_native"],
                    metrics["center_error_native"],
                ),
                "center_worse_frames": _worse_count(
                    qk_center,
                    center,
                    evaluation_indices,
                ),
                "rotation_degrees": float(metrics["rotation_degrees"]),
                "rotation_gain_vs_qk_percent": _gain(
                    qk_metrics["rotation_degrees"],
                    metrics["rotation_degrees"],
                ),
                "rotation_worse_frames": _worse_count(
                    qk_rotation,
                    rotation,
                    evaluation_indices,
                ),
                "branch_pose_pass": int(
                    float(metrics["center_error_native"])
                    < float(qk_metrics["center_error_native"])
                    and float(metrics["rotation_degrees"])
                    < float(qk_metrics["rotation_degrees"])
                ),
            }
        )
        diagnostics = {
            int(row["sequence_index"]): row
            for row in diagnostic_by_branch[branch]
        }
        for index, frame in enumerate(frames):
            diag = diagnostics[index]
            frame_rows.append(
                {
                    "branch": branch,
                    "sequence_index": index,
                    "frame_index": frame,
                    "active": int(diag["active"]),
                    "selected_candidate": diag["selected_candidate"],
                    "qk_center_error_native": float(qk_center[index]),
                    "candidate_center_error_native": float(center[index]),
                    "qk_rotation_degrees": float(qk_rotation[index]),
                    "candidate_rotation_degrees": float(rotation[index]),
                }
            )
    by_branch = {str(row["branch"]): row for row in branch_rows}
    correct = by_branch["correct_persistent_id"]
    controls = [
        by_branch[name]
        for name in BRANCHES
        if name != "correct_persistent_id"
    ]
    unique_pass = int(
        int(correct["branch_pose_pass"])
        and all(
            float(correct["center_error_native"])
            < float(control["center_error_native"])
            and float(correct["rotation_degrees"])
            < float(control["rotation_degrees"])
            for control in controls
        )
    )
    active_counts = {int(row["active_frames"]) for row in branch_rows}
    equal_active_frames = int(len(active_counts) == 1)
    active_frames = next(iter(active_counts)) if equal_active_frames else 0
    decisions = {
        "equal_source_point_protocol": 1,
        "equal_active_frames": equal_active_frames,
        "probe_feasible": int(active_frames > 0),
        "correct_id_pose_pass": int(correct["branch_pose_pass"]),
        "correct_id_unique_vs_all_controls_pass": unique_pass,
        "sam_identity_pose_direction_evidence": unique_pass,
        "candidate_generation_gt_fields": 0,
        "formal_v0_pose_modified": 0,
        "formal_v0_pointmap_modified": 0,
        "next_gate": (
            "implement_bounded_iterative_sam_indexed_pose_residual"
            if unique_pass
            else (
                "inspect_visibility_only_probe_has_no_common_active_frames"
                if active_frames == 0
                else "stop_sam_pose_optimization_keep_semantic_lifting_only"
            )
        ),
    }
    summary = {
        "schema": 1,
        "revision": REVISION,
        "baseline_version": "v0",
        "clip": payload["clip_name"],
        "config": str(run.source_path),
        "cache": str(cache_path_value),
        "qk_artifact": str(qk_path),
        "frames": frames,
        "reference_frame": frames[reference],
        "evaluated_pose_frames": len(evaluation_indices),
        "branches": BRANCHES,
        "initial_pose": "retrieve_qk",
        "geometry_source": "causal_history_raw_full_history_world_pointmap",
        "current_measurement": "raw_depth_plus_eroded_sam_ownership_mask",
        "mask_source": "tracking_masks_stream",
        "mask_semantics": "deployed_sam31_online_geometry_compete_output",
        "sam_hidden_features_used": 0,
        "model_trained": 0,
        "candidate_generation_gt_fields": 0,
        "gt_role": "scoring_only_after_fixed_direction_selection",
        "rotation_step_degrees": run.rotation_step_degrees,
        "translation_step_scene_fraction": run.translation_step_scene_fraction,
        "minimum_loss_decrease": run.min_loss_decrease,
        "candidate_count_per_active_branch_frame": 13,
        "qk_metrics": qk_metrics,
        "branch_results": branch_rows,
        "decisions": decisions,
    }
    run.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(run.output_dir / "candidate_diagnostics.csv", diagnostic_rows)
    _write_csv(run.output_dir / "branch_summary.csv", branch_rows)
    _write_csv(run.output_dir / "frame_pose_scoring.csv", frame_rows)
    torch.save(
        {
            "revision": REVISION,
            "frame_indices": frames,
            "initial_qk_world_to_camera": qk_pose[0],
            "candidate_world_to_camera": candidates,
            "selected_for_v0": False,
        },
        run.output_dir / "candidate_poses.pt",
    )
    result = run.output_dir / "sam_identity_pose_probe_summary.json"
    result.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf8",
    )
    _write_copyable(
        run.output_dir / "copyable_result.txt",
        summary=summary,
        output_dir=run.output_dir,
    )
    print("V0 CAUSAL SAM PERSISTENT-ID FIXED-DIRECTION POSE PROBE")
    for row in branch_rows:
        print(
            f"  branch={row['branch']} active={row['active_frames']} "
            f"modified={row['modified_frames']} "
            f"center_gain={row['center_gain_vs_qk_percent']:.4f}% "
            f"R_gain={row['rotation_gain_vs_qk_percent']:.4f}% "
            f"pass={row['branch_pose_pass']}"
        )
    print(f"  decision={json.dumps(decisions, sort_keys=True)}")
    print(f"  copyable_report={run.output_dir / 'copyable_result.txt'}")
    return result


def _pose_frame_errors(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    relative = predicted[..., :3, :3] @ target[..., :3, :3].transpose(-1, -2)
    cosine = (
        torch.diagonal(relative, dim1=-2, dim2=-1).sum(dim=-1) - 1.0
    ) * 0.5
    rotation = torch.rad2deg(torch.acos(cosine.clamp(-1, 1)))[0]
    center = torch.linalg.vector_norm(
        camera_centers(predicted) - camera_centers(target), dim=-1
    )[0]
    return rotation.detach().float().cpu(), center.detach().float().cpu()


def _worse_count(
    baseline: torch.Tensor,
    candidate: torch.Tensor,
    indices: Sequence[int],
) -> int:
    index = torch.tensor(tuple(int(value) for value in indices), dtype=torch.long)
    return int(
        (
            candidate.index_select(0, index)
            > baseline.index_select(0, index)
        ).sum()
    )


def _gain(baseline: float, candidate: float) -> float:
    if not math.isfinite(float(baseline)) or float(baseline) <= 0.0:
        raise ValueError("Baseline metric must be finite and positive.")
    return 100.0 * (float(baseline) - float(candidate)) / float(baseline)


def _inactive_row(
    *,
    branch: str,
    sequence_index: int,
    frame_index: int,
    eligible_slots: Sequence[int],
    reason: str,
) -> dict[str, object]:
    return {
        "branch": branch,
        "sequence_index": int(sequence_index),
        "frame_index": int(frame_index),
        "active": 0,
        "reason": str(reason),
        "eligible_slots": _space(eligible_slots),
        "eligible_sam_track_ids": "",
        "eligible_prompts": "",
        "source_points": 0,
        "candidate_count": 0,
        "selected_candidate": "identity",
        "pose_modified": 0,
        "identity_loss": float("nan"),
        "selected_loss": float("nan"),
        "loss_decrease": 0.0,
        "identity_depth_support_rate": float("nan"),
        "selected_depth_support_rate": float("nan"),
        "identity_ownership_hit_rate": float("nan"),
        "selected_ownership_hit_rate": float("nan"),
    }


def _write_copyable(
    path: Path,
    *,
    summary: dict[str, object],
    output_dir: Path,
) -> None:
    lines = [
        "===== COPYABLE_V0_SAM_IDENTITY_POSE_PROBE_BEGIN =====",
        f"revision={REVISION}",
        f"clip={summary['clip']}",
        f"reference_frame={summary['reference_frame']}",
        f"evaluated_pose_frames={summary['evaluated_pose_frames']}",
        f"branches={','.join(summary['branches'])}",
        f"initial_pose={summary['initial_pose']}",
        f"geometry_source={summary['geometry_source']}",
        f"current_measurement={summary['current_measurement']}",
        f"mask_source={summary['mask_source']}",
        f"mask_semantics={summary['mask_semantics']}",
        "sam_hidden_features_used=0",
        "model_trained=0",
        "candidate_generation_gt_fields=0",
        "gt_role=scoring_only_after_fixed_direction_selection",
        f"rotation_step_degrees={summary['rotation_step_degrees']}",
        "translation_step_scene_fraction="
        f"{summary['translation_step_scene_fraction']}",
        f"minimum_loss_decrease={summary['minimum_loss_decrease']}",
        "candidate_count_per_active_branch_frame=13",
        "",
        "branch,active_frames,modified_frames,center_error_native,center_gain_vs_qk_percent,center_worse_frames,rotation_degrees,rotation_gain_vs_qk_percent,rotation_worse_frames,branch_pose_pass",
    ]
    columns = (
        "branch",
        "active_frames",
        "modified_frames",
        "center_error_native",
        "center_gain_vs_qk_percent",
        "center_worse_frames",
        "rotation_degrees",
        "rotation_gain_vs_qk_percent",
        "rotation_worse_frames",
        "branch_pose_pass",
    )
    for row in summary["branch_results"]:
        lines.append(",".join(str(row[column]) for column in columns))
    lines.extend(
        [
            "",
            "decision=" + json.dumps(summary["decisions"], sort_keys=True),
            "",
            "outputs:",
            f"summary={output_dir / 'sam_identity_pose_probe_summary.json'}",
            f"branch_csv={output_dir / 'branch_summary.csv'}",
            f"candidate_csv={output_dir / 'candidate_diagnostics.csv'}",
            f"frame_csv={output_dir / 'frame_pose_scoring.csv'}",
            f"poses={output_dir / 'candidate_poses.pt'}",
            f"copyable_report={path}",
            "===== COPYABLE_V0_SAM_IDENTITY_POSE_PROBE_END =====",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}.")
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _scalar(value: torch.Tensor) -> torch.Tensor:
    output = value.detach().float().cpu()
    if output.ndim == 4 and output.shape[-1] == 1:
        output = output[..., 0]
    if output.ndim != 3:
        raise ValueError("Expected scalar map [S,H,W] or [S,H,W,1].")
    return output


def _space(values: Sequence[int]) -> str:
    return " ".join(str(int(value)) for value in values)


def _load_run(path: str | Path) -> ProbeRun:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf8") as handle:
        raw = yaml.safe_load(handle) or {}
    section = raw.get("sam_identity_pose_probe", {})
    run = ProbeRun(
        source_path=source,
        output_dir=Path(
            section.get(
                "output_dir",
                "outputs/streaming_couping_v0/sam_identity_pose_probe",
            )
        ).expanduser().resolve(),
        confidence_threshold=float(section.get("confidence_threshold", 0.30)),
        track_score_threshold=float(section.get("track_score_threshold", 0.50)),
        mask_erosion_radius=int(section.get("mask_erosion_radius", 2)),
        min_mask_pixels=int(section.get("min_mask_pixels", 128)),
        min_history_observations=int(section.get("min_history_observations", 2)),
        min_history_points=int(section.get("min_history_points", 128)),
        points_per_instance=int(section.get("points_per_instance", 1024)),
        points_per_frame_write=int(section.get("points_per_frame_write", 4096)),
        instance_max_points=int(section.get("instance_max_points", 8192)),
        global_max_points=int(section.get("global_max_points", 32768)),
        relative_depth_cap=float(section.get("relative_depth_cap", 0.25)),
        mask_miss_weight=float(section.get("mask_miss_weight", 0.05)),
        min_loss_decrease=float(section.get("min_loss_decrease", 1e-5)),
        rotation_step_degrees=float(section.get("rotation_step_degrees", 0.25)),
        translation_step_scene_fraction=float(
            section.get("translation_step_scene_fraction", 0.0025)
        ),
        shifted_mask_x_fraction=float(section.get("shifted_mask_x_fraction", 0.10)),
        shifted_mask_y_fraction=float(section.get("shifted_mask_y_fraction", 0.07)),
    )
    _validate_run(run)
    return run


def _validate_run(run: ProbeRun) -> None:
    for name, value in (
        ("confidence_threshold", run.confidence_threshold),
        ("track_score_threshold", run.track_score_threshold),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"sam_identity_pose_probe.{name} must be in [0,1].")
    positive_ints = (
        run.min_mask_pixels,
        run.min_history_observations,
        run.min_history_points,
        run.points_per_instance,
        run.points_per_frame_write,
        run.instance_max_points,
        run.global_max_points,
    )
    if any(value <= 0 for value in positive_ints):
        raise ValueError("SAM identity pose probe counts must be positive.")
    if run.points_per_instance < run.min_history_points:
        raise ValueError("points_per_instance must cover min_history_points.")
    positive_floats = (
        run.relative_depth_cap,
        run.rotation_step_degrees,
        run.translation_step_scene_fraction,
        run.shifted_mask_x_fraction,
        run.shifted_mask_y_fraction,
        run.min_loss_decrease,
    )
    if any(value <= 0.0 for value in positive_floats) or run.mask_miss_weight < 0.0:
        raise ValueError("SAM identity pose probe step/loss settings are invalid.")


def _validate_inputs(
    payload: dict[str, Any],
    *,
    qk: dict[str, Any],
    clip: ClipConfig,
) -> None:
    required = (
        "frame_indices",
        "image_size",
        "reference_sequence_index",
        "baseline_pose_encoding",
        "target_pose_encoding",
        "baseline_world_points",
        "baseline_world_confidence",
        "baseline_depth",
        "baseline_depth_confidence",
        "tracking_masks_stream",
        "tracking_scores",
        "scene_scale",
        "instance_ids",
        "sam_track_ids",
        "sam_track_prompts",
    )
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError(f"V0 cache lacks SAM identity probe fields: {missing}.")
    frames = tuple(int(value) for value in payload["frame_indices"])
    if frames != clip.frame_indices:
        raise ValueError("V0 cache frame order differs from config.")
    if tuple(int(value) for value in qk.get("frame_indices", ())) != frames:
        raise ValueError("QK artifact frame order differs from V0 cache.")
    sequence = len(frames)
    if qk.get("selected_pose_branch") != "retrieve_qk":
        raise ValueError("SAM identity probe requires the locked QK pose artifact.")
    pose = qk.get("selected_world_to_camera")
    if not torch.is_tensor(pose) or pose.shape != (1, sequence, 3, 4):
        raise ValueError("QK selected pose shape is invalid.")
    raw_pose = qk.get("raw_world_to_camera")
    if not torch.is_tensor(raw_pose) or raw_pose.shape != (1, sequence, 3, 4):
        raise ValueError("QK raw pose shape is invalid.")
    if not bool(torch.isfinite(pose).all()):
        raise ValueError("QK selected pose contains non-finite values.")


def _find_clip(config: LearnedPoseConfig, name: str) -> ClipConfig:
    selected = [clip for clip in config.clips if clip.name == name]
    if len(selected) != 1:
        raise ValueError(f"Clip {name!r} was not found exactly once.")
    return selected[0]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="streaming_couping/configs/v0_baseline.yaml",
    )
    parser.add_argument("--output-dir")
    return parser.parse_args()


if __name__ == "__main__":
    main()
