#!/usr/bin/env python3
"""Audit and test causal SAM-indexed local StreamVGGT point adjustment."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch
import yaml

from streaming_couping.src.instance_geometry import (
    SurfelQueryResult,
    apply_sparse_deltas,
    bounded_surfel_query,
    erode_instance_masks,
    merge_sparse_deltas,
    select_mask_points,
    shift_instance_masks,
)
from streaming_couping.src.learned_pose.baseline_runtime import (
    load_baseline_run_config,
)
from streaming_couping.src.learned_pose.cache import cache_path, load_feature_cache
from streaming_couping.src.learned_pose.config import (
    ClipConfig,
    load_learned_pose_config,
)
from streaming_couping.src.semantic_map import normalize_confidence, semantic_slot_map


REVISION = "v1_causal_instance_surfel_audit_r1"
BASELINE_REVISION = "v0_frozen_semantic_mapping_pipeline_r1"
I0_BRANCHES = ("correct_id", "shuffled_id", "shifted_mask")
I1_BRANCHES = (
    "raw",
    "global_surfel",
    "foreground_union",
    "correct_id",
    "shuffled_id",
    "shifted_mask",
)


@dataclass(frozen=True)
class V1Run:
    source_path: Path
    baseline_config: Path
    output_dir: Path
    device: str
    max_active_instances: int
    min_history_visible_frames: int
    min_track_score: float
    min_mask_area_ratio: float
    max_mask_area_ratio: float
    min_current_points: int
    point_confidence_threshold: float
    erosion_radius: int
    neighbors: int
    min_support_frames: int
    max_current_points_per_instance: int
    max_history_points_per_frame: int
    max_history_points_total: int
    match_radius_scene_fraction: float
    normal_variance_max: float
    alpha: float
    max_displacement_scene_fraction: float
    chunk_size: int
    shifted_mask_y_fraction: float
    shifted_mask_x_fraction: float
    symmetric_metric_points: int


@dataclass
class CandidateChunks:
    indices: list[torch.Tensor]
    deltas: list[torch.Tensor]
    selected_points: int = 0
    valid_surfel_points: int = 0
    residual_before_sum: float = 0.0
    residual_after_sum: float = 0.0

    def add(
        self,
        *,
        global_indices: torch.Tensor,
        query: SurfelQueryResult,
    ) -> None:
        self.selected_points += int(global_indices.numel())
        valid = query.valid
        count = int(valid.sum())
        self.valid_surfel_points += count
        if count:
            self.indices.append(global_indices[valid].long().cpu())
            self.deltas.append(query.delta[valid].float().cpu())
            before = query.normal_residual[valid]
            displacement = torch.linalg.vector_norm(query.delta[valid], dim=-1)
            self.residual_before_sum += float(before.sum())
            self.residual_after_sum += float((before - displacement).abs().sum())


def main() -> None:
    args = _parse_args()
    run = _load_run(args.config)
    if args.device:
        run = _replace_device(run, args.device)
    data = load_learned_pose_config(run.baseline_config)
    baseline = load_baseline_run_config(run.baseline_config)
    clip = _find_clip(data.clips, baseline.clip_name)
    cache_path_value = cache_path(data, clip)
    payload = load_feature_cache(cache_path_value)
    baseline_summary = _validate_inputs(
        payload=payload,
        clip=clip,
        baseline_output_dir=baseline.output_dir,
    )

    points = payload["baseline_world_points"].detach().float().cpu()
    confidence = normalize_confidence(payload["baseline_world_confidence"])
    masks = payload["tracking_masks_stream"].detach().bool().cpu()
    scores = payload["tracking_scores"].detach().float().cpu()
    slot_map = semantic_slot_map(
        masks,
        scores,
        score_threshold=run.min_track_score,
    )
    owned_masks = torch.stack(
        [slot_map == slot for slot in range(masks.shape[1])], dim=1
    )
    interior_masks = erode_instance_masks(owned_masks, radius=run.erosion_radius)
    shift_y = max(1, round(points.shape[1] * run.shifted_mask_y_fraction))
    shift_x = max(1, round(points.shape[2] * run.shifted_mask_x_fraction))
    shifted_masks = shift_instance_masks(
        interior_masks,
        shift_y=shift_y,
        shift_x=shift_x,
    )
    valid_points = (
        torch.isfinite(points).all(dim=-1)
        & torch.isfinite(confidence)
        & (confidence >= run.point_confidence_threshold)
    )
    scene_diagonal = _scene_diagonal(points, valid_points)
    match_radius = scene_diagonal * run.match_radius_scene_fraction
    max_displacement = scene_diagonal * run.max_displacement_scene_fraction
    discovered_slots = _discovered_slots(payload)
    active_pairs, track_rows = _active_instance_pairs(
        payload=payload,
        scores=scores,
        interior_masks=interior_masks,
        valid_points=valid_points,
        discovered_slots=discovered_slots,
        run=run,
    )

    candidate_chunks = {branch: CandidateChunks([], []) for branch in I0_BRANCHES}
    audit_rows = _run_i0(
        points=points,
        confidence=confidence,
        valid_points=valid_points,
        interior_masks=interior_masks,
        shifted_masks=shifted_masks,
        active_pairs=active_pairs,
        discovered_slots=discovered_slots,
        prompts=tuple(str(value) for value in payload["sam_track_prompts"]),
        frame_indices=tuple(int(value) for value in payload["frame_indices"]),
        run=run,
        match_radius=match_radius,
        max_displacement=max_displacement,
        candidate_chunks=candidate_chunks,
    )
    i0_branch_rows = _summarize_i0(audit_rows)
    eligible_slots = tuple(
        int(row["slot"]) for row in track_rows if int(row["eligible_track"])
    )
    i0_decision = _i0_decision(
        audit_rows=audit_rows,
        branch_rows=i0_branch_rows,
        eligible_slots=eligible_slots,
    )

    sparse_candidates: dict[str, tuple[torch.Tensor, torch.Tensor]] = {
        "raw": (torch.empty(0, dtype=torch.long), torch.empty(0, 3))
    }
    branch_rows: list[dict[str, object]] = []
    per_track_rows: list[dict[str, object]] = []
    i1_run = int(i0_decision["i0_pass"])
    if i1_run:
        union_chunks = _run_group_candidate(
            branch="foreground_union",
            points=points,
            confidence=confidence,
            valid_points=valid_points,
            masks=interior_masks,
            active_pairs=active_pairs,
            discovered_slots=discovered_slots,
            run=run,
            match_radius=match_radius,
            max_displacement=max_displacement,
        )
        global_chunks = _run_group_candidate(
            branch="global_surfel",
            points=points,
            confidence=confidence,
            valid_points=valid_points,
            masks=interior_masks,
            active_pairs=active_pairs,
            discovered_slots=discovered_slots,
            run=run,
            match_radius=match_radius,
            max_displacement=max_displacement,
        )
        candidate_chunks["foreground_union"] = union_chunks
        candidate_chunks["global_surfel"] = global_chunks
        for branch in I1_BRANCHES[1:]:
            chunks = candidate_chunks[branch]
            sparse_candidates[branch] = merge_sparse_deltas(
                chunks.indices, chunks.deltas
            )

        # GT fields are first read here, after every candidate is frozen.
        branch_rows, per_track_rows = _score_candidates(
            points=points,
            confidence=confidence,
            target_points=payload["target_world_points"],
            alignment_scale=float(payload["point_alignment_scale"]),
            alignment_rotation=payload["point_alignment_rotation"],
            alignment_translation=payload["point_alignment_translation"],
            valid_points=valid_points,
            interior_masks=interior_masks,
            eligible_slots=eligible_slots,
            frame_indices=tuple(int(value) for value in payload["frame_indices"]),
            prompts=tuple(str(value) for value in payload["sam_track_prompts"]),
            candidates=sparse_candidates,
            chunks=candidate_chunks,
            run=run,
            max_displacement=max_displacement,
        )
    decision = _final_decision(
        i0_decision=i0_decision,
        branch_rows=branch_rows,
        per_track_rows=per_track_rows,
        max_displacement=max_displacement,
    )
    result = _write_outputs(
        run=run,
        payload=payload,
        baseline_summary=baseline_summary,
        cache_path_value=cache_path_value,
        scene_diagonal=scene_diagonal,
        match_radius=match_radius,
        max_displacement=max_displacement,
        shift_y=shift_y,
        shift_x=shift_x,
        track_rows=track_rows,
        audit_rows=audit_rows,
        i0_branch_rows=i0_branch_rows,
        branch_rows=branch_rows,
        per_track_rows=per_track_rows,
        sparse_candidates=sparse_candidates if i1_run else {},
        decision=decision,
    )
    print(f"V1 instance geometry result={result}")


def _run_i0(
    *,
    points: torch.Tensor,
    confidence: torch.Tensor,
    valid_points: torch.Tensor,
    interior_masks: torch.Tensor,
    shifted_masks: torch.Tensor,
    active_pairs: dict[int, tuple[int, ...]],
    discovered_slots: tuple[int, ...],
    prompts: tuple[str, ...],
    frame_indices: tuple[int, ...],
    run: V1Run,
    match_radius: float,
    max_displacement: float,
    candidate_chunks: dict[str, CandidateChunks],
) -> list[dict[str, object]]:
    rows = []
    shuffled = {
        slot: discovered_slots[(index + 1) % len(discovered_slots)]
        for index, slot in enumerate(discovered_slots)
    }
    pixels_per_frame = points.shape[1] * points.shape[2]
    for sequence_index, active_slots in active_pairs.items():
        for slot in active_slots:
            specifications = {
                "correct_id": (interior_masks, slot, slot),
                "shuffled_id": (interior_masks, slot, shuffled[slot]),
                "shifted_mask": (shifted_masks, slot, slot),
            }
            for branch, (branch_masks, current_slot, history_slot) in specifications.items():
                current_mask = (
                    branch_masks[sequence_index, current_slot]
                    & valid_points[sequence_index]
                )
                current_points, _, current_indices = select_mask_points(
                    points[sequence_index],
                    confidence[sequence_index],
                    current_mask,
                    limit=run.max_current_points_per_instance,
                )
                history_points, history_weights, history_frames = _collect_history(
                    points=points,
                    confidence=confidence,
                    valid_points=valid_points,
                    masks=branch_masks[:, history_slot],
                    before=sequence_index,
                    per_frame_limit=run.max_history_points_per_frame,
                    total_limit=run.max_history_points_total,
                )
                query = _query(
                    current_points=current_points,
                    history_points=history_points,
                    history_weights=history_weights,
                    history_frames=history_frames,
                    run=run,
                    match_radius=match_radius,
                    max_displacement=max_displacement,
                )
                global_indices = current_indices + sequence_index * pixels_per_frame
                candidate_chunks[branch].add(
                    global_indices=global_indices,
                    query=query,
                )
                rows.append(
                    {
                        "branch": branch,
                        "sequence_index": sequence_index,
                        "frame_index": frame_indices[sequence_index],
                        "slot": slot,
                        "history_slot": history_slot,
                        "prompt": prompts[slot],
                        "current_points": int(current_points.shape[0]),
                        "history_points": int(history_points.shape[0]),
                        "valid_surfel_points": int(query.valid.sum()),
                        "valid_surfel_rate": _ratio(
                            int(query.valid.sum()), int(current_points.shape[0])
                        ),
                        "nearest_median": _finite_quantile(
                            query.nearest_distance, 0.50
                        ),
                        "nearest_p90": _finite_quantile(
                            query.nearest_distance, 0.90
                        ),
                        "normal_residual_median": _finite_quantile(
                            query.normal_residual, 0.50
                        ),
                        "surface_thickness_median": _finite_quantile(
                            query.surface_thickness, 0.50
                        ),
                        "support_frames_median": _finite_quantile(
                            query.support_frames.float(), 0.50
                        ),
                    }
                )
    return rows


def _run_group_candidate(
    *,
    branch: str,
    points: torch.Tensor,
    confidence: torch.Tensor,
    valid_points: torch.Tensor,
    masks: torch.Tensor,
    active_pairs: dict[int, tuple[int, ...]],
    discovered_slots: tuple[int, ...],
    run: V1Run,
    match_radius: float,
    max_displacement: float,
) -> CandidateChunks:
    chunks = CandidateChunks([], [])
    pixels_per_frame = points.shape[1] * points.shape[2]
    if branch not in {"foreground_union", "global_surfel"}:
        raise ValueError(f"Unsupported grouped branch={branch!r}.")
    union_masks = masks[:, list(discovered_slots)].any(dim=1)
    for sequence_index, active_slots in active_pairs.items():
        if not active_slots:
            continue
        limit = run.max_current_points_per_instance * len(active_slots)
        if branch == "foreground_union":
            current_mask = masks[sequence_index, list(active_slots)].any(dim=0)
            history_masks = union_masks
        else:
            current_mask = valid_points[sequence_index]
            history_masks = valid_points
        current_mask = current_mask & valid_points[sequence_index]
        current_points, _, current_indices = select_mask_points(
            points[sequence_index],
            confidence[sequence_index],
            current_mask,
            limit=limit,
        )
        history_points, history_weights, history_frames = _collect_history(
            points=points,
            confidence=confidence,
            valid_points=valid_points,
            masks=history_masks,
            before=sequence_index,
            per_frame_limit=run.max_history_points_per_frame,
            total_limit=run.max_history_points_total,
        )
        query = _query(
            current_points=current_points,
            history_points=history_points,
            history_weights=history_weights,
            history_frames=history_frames,
            run=run,
            match_radius=match_radius,
            max_displacement=max_displacement,
        )
        chunks.add(
            global_indices=current_indices + sequence_index * pixels_per_frame,
            query=query,
        )
    return chunks


def _query(
    *,
    current_points: torch.Tensor,
    history_points: torch.Tensor,
    history_weights: torch.Tensor,
    history_frames: torch.Tensor,
    run: V1Run,
    match_radius: float,
    max_displacement: float,
) -> SurfelQueryResult:
    return bounded_surfel_query(
        current_points=current_points,
        history_points=history_points,
        history_weights=history_weights,
        history_frame_ids=history_frames,
        device=run.device,
        neighbors=run.neighbors,
        min_support_frames=run.min_support_frames,
        match_radius=match_radius,
        normal_variance_max=run.normal_variance_max,
        alpha=run.alpha,
        max_displacement=max_displacement,
        chunk_size=run.chunk_size,
    )


def _collect_history(
    *,
    points: torch.Tensor,
    confidence: torch.Tensor,
    valid_points: torch.Tensor,
    masks: torch.Tensor,
    before: int,
    per_frame_limit: int,
    total_limit: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    point_parts = []
    weight_parts = []
    frame_parts = []
    for frame in range(int(before)):
        selected, weights, _ = select_mask_points(
            points[frame],
            confidence[frame],
            masks[frame] & valid_points[frame],
            limit=per_frame_limit,
        )
        if not selected.numel():
            continue
        point_parts.append(selected)
        weight_parts.append(weights)
        frame_parts.append(torch.full((selected.shape[0],), frame, dtype=torch.long))
    if not point_parts:
        return torch.empty(0, 3), torch.empty(0), torch.empty(0, dtype=torch.long)
    history = torch.cat(point_parts)
    weights = torch.cat(weight_parts)
    frames = torch.cat(frame_parts)
    if history.shape[0] > int(total_limit):
        indices = torch.linspace(
            0, history.shape[0] - 1, steps=int(total_limit)
        ).round().long()
        history = history.index_select(0, indices)
        weights = weights.index_select(0, indices)
        frames = frames.index_select(0, indices)
    return history, weights, frames


def _active_instance_pairs(
    *,
    payload: dict[str, Any],
    scores: torch.Tensor,
    interior_masks: torch.Tensor,
    valid_points: torch.Tensor,
    discovered_slots: tuple[int, ...],
    run: V1Run,
) -> tuple[dict[int, tuple[int, ...]], list[dict[str, object]]]:
    sequence, _, height, width = interior_masks.shape
    qualifying = torch.zeros(sequence, interior_masks.shape[1], dtype=torch.bool)
    valid_counts = torch.zeros_like(qualifying, dtype=torch.long)
    area_ratios = torch.zeros_like(qualifying, dtype=torch.float32)
    for frame in range(sequence):
        for slot in discovered_slots:
            mask = interior_masks[frame, slot]
            area = float(mask.float().mean())
            count = int((mask & valid_points[frame]).sum())
            area_ratios[frame, slot] = area
            valid_counts[frame, slot] = count
            qualifying[frame, slot] = bool(
                float(scores[frame, slot]) >= run.min_track_score
                and run.min_mask_area_ratio <= area <= run.max_mask_area_ratio
                and count >= run.min_current_points
            )
    history_count = {slot: 0 for slot in discovered_slots}
    active_pairs: dict[int, tuple[int, ...]] = {}
    active_count = {slot: 0 for slot in discovered_slots}
    for frame in range(sequence):
        candidates = [
            slot
            for slot in discovered_slots
            if bool(qualifying[frame, slot])
            and history_count[slot] >= run.min_history_visible_frames
        ]
        candidates.sort(
            key=lambda slot: (
                -history_count[slot],
                -float(scores[frame, slot]),
                slot,
            )
        )
        selected = tuple(candidates[: run.max_active_instances])
        active_pairs[frame] = selected
        for slot in selected:
            active_count[slot] += 1
        for slot in discovered_slots:
            if bool(qualifying[frame, slot]):
                history_count[slot] += 1
    prompts = tuple(str(value) for value in payload["sam_track_prompts"])
    track_ids = tuple(int(value) for value in payload["sam_track_ids"])
    births = tuple(int(value) for value in payload["sam_birth_indices"])
    frame_indices = tuple(int(value) for value in payload["frame_indices"])
    rows = []
    for slot in discovered_slots:
        visible = interior_masks[:, slot].flatten(1).any(dim=1)
        rows.append(
            {
                "slot": slot,
                "sam_track_id": track_ids[slot],
                "prompt": prompts[slot],
                "birth_sequence_index": births[slot],
                "birth_frame": frame_indices[births[slot]],
                "visible_frames": int(visible.sum()),
                "qualifying_frames": int(qualifying[:, slot].sum()),
                "active_frames": active_count[slot],
                "dense_valid_interior_points": int(valid_counts[:, slot].sum()),
                "median_interior_area_ratio": _finite_quantile(
                    area_ratios[visible, slot], 0.50
                ),
                "median_track_score": _finite_quantile(scores[visible, slot], 0.50),
                "eligible_track": int(active_count[slot] > 0),
            }
        )
    return active_pairs, rows


def _summarize_i0(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    for branch in I0_BRANCHES:
        selected = [row for row in rows if row["branch"] == branch]
        output.append(
            {
                "branch": branch,
                "selected_track_frames": len(selected),
                "active_tracks": len(
                    {int(row["slot"]) for row in selected if int(row["current_points"]) > 0}
                ),
                "selected_points": sum(int(row["current_points"]) for row in selected),
                "valid_surfel_points": sum(
                    int(row["valid_surfel_points"]) for row in selected
                ),
                "valid_surfel_rate": _ratio(
                    sum(int(row["valid_surfel_points"]) for row in selected),
                    sum(int(row["current_points"]) for row in selected),
                ),
                "mean_nearest_median": _weighted_row_mean(
                    selected, "nearest_median", "current_points"
                ),
                "mean_normal_residual_median": _weighted_row_mean(
                    selected, "normal_residual_median", "current_points"
                ),
                "mean_surface_thickness_median": _weighted_row_mean(
                    selected, "surface_thickness_median", "current_points"
                ),
            }
        )
    return output


def _i0_decision(
    *,
    audit_rows: list[dict[str, object]],
    branch_rows: list[dict[str, object]],
    eligible_slots: tuple[int, ...],
) -> dict[str, object]:
    by_branch = {str(row["branch"]): row for row in branch_rows}
    correct = by_branch["correct_id"]
    controls = [by_branch["shuffled_id"], by_branch["shifted_mask"]]
    global_order = all(
        float(correct["mean_nearest_median"]) < float(row["mean_nearest_median"])
        and float(correct["mean_normal_residual_median"])
        < float(row["mean_normal_residual_median"])
        for row in controls
    )
    track_wins = 0
    for slot in eligible_slots:
        values = {}
        for branch in I0_BRANCHES:
            selected = [
                row
                for row in audit_rows
                if row["branch"] == branch and int(row["slot"]) == slot
            ]
            values[branch] = (
                _weighted_row_mean(selected, "nearest_median", "current_points"),
                _weighted_row_mean(
                    selected, "normal_residual_median", "current_points"
                ),
            )
        if all(
            values["correct_id"][0] < values[branch][0]
            and values["correct_id"][1] < values[branch][1]
            for branch in ("shuffled_id", "shifted_mask")
        ):
            track_wins += 1
    finite = all(
        math.isfinite(float(row["mean_nearest_median"]))
        and math.isfinite(float(row["mean_normal_residual_median"]))
        for row in branch_rows
    )
    passed = int(
        len(eligible_slots) >= 2
        and int(correct["active_tracks"]) >= 2
        and finite
        and global_order
        and track_wins >= 2
    )
    return {
        "eligible_track_count": len(eligible_slots),
        "correct_active_tracks": int(correct["active_tracks"]),
        "finite_equal_protocol": int(finite),
        "correct_global_order_pass": int(global_order),
        "correct_per_track_win_count": track_wins,
        "i0_pass": passed,
        "i0_claim": (
            "persistent_id_geometric_coherence_supported"
            if passed
            else "instance_geometry_support_not_established"
        ),
    }


def _score_candidates(
    *,
    points: torch.Tensor,
    confidence: torch.Tensor,
    target_points: torch.Tensor,
    alignment_scale: float,
    alignment_rotation: torch.Tensor,
    alignment_translation: torch.Tensor,
    valid_points: torch.Tensor,
    interior_masks: torch.Tensor,
    eligible_slots: tuple[int, ...],
    frame_indices: tuple[int, ...],
    prompts: tuple[str, ...],
    candidates: dict[str, tuple[torch.Tensor, torch.Tensor]],
    chunks: dict[str, CandidateChunks],
    run: V1Run,
    max_displacement: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    target = target_points.detach().float().cpu()
    rotation = alignment_rotation.detach().float().cpu()
    translation = alignment_translation.detach().float().cpu()
    target_valid = torch.isfinite(target).all(dim=-1)
    global_valid = valid_points & target_valid
    instance_masks = (
        interior_masks[:, list(eligible_slots)] if eligible_slots else None
    )
    instance_union = (
        instance_masks.any(dim=1)
        if instance_masks is not None
        else torch.zeros_like(global_valid)
    )
    global_valid = global_valid & torch.isfinite(confidence)
    instance_valid = global_valid & instance_union
    raw_aligned = float(alignment_scale) * (points @ rotation.T) + translation
    raw_global_rmse = _weighted_rmse(
        raw_aligned, target, confidence, global_valid
    )
    raw_instance_rmse = _weighted_rmse(
        raw_aligned, target, confidence, instance_valid
    )
    raw_frame_rmse = _frame_rmse(
        raw_aligned, target, confidence, global_valid
    )
    branch_rows = []
    per_track_rows = []
    for branch in I1_BRANCHES:
        indices, deltas = candidates[branch]
        candidate = apply_sparse_deltas(points, indices, deltas)
        aligned = float(alignment_scale) * (candidate @ rotation.T) + translation
        global_rmse = _weighted_rmse(aligned, target, confidence, global_valid)
        instance_rmse = _weighted_rmse(aligned, target, confidence, instance_valid)
        frame_rmse = _frame_rmse(aligned, target, confidence, global_valid)
        worse_frames = sum(
            int(math.isfinite(after) and after > before)
            for before, after in zip(raw_frame_rmse, frame_rmse)
            if math.isfinite(before)
        )
        modified_background = (
            int((~instance_union.reshape(-1).index_select(0, indices)).sum())
            if indices.numel()
            else 0
        )
        displacement = (
            torch.linalg.vector_norm(deltas, dim=-1)
            if deltas.numel()
            else torch.empty(0)
        )
        chunk = chunks.get(branch, CandidateChunks([], []))
        branch_rows.append(
            {
                "branch": branch,
                "selected_points": int(chunk.selected_points),
                "modified_points": int(indices.numel()),
                "valid_surfel_rate": _ratio(
                    chunk.valid_surfel_points, chunk.selected_points
                ),
                "global_weighted_rmse": global_rmse,
                "global_gain_vs_raw_percent": _gain(raw_global_rmse, global_rmse),
                "instance_weighted_rmse": instance_rmse,
                "instance_gain_vs_raw_percent": _gain(
                    raw_instance_rmse, instance_rmse
                ),
                "symmetric_mean": _symmetric_mean(
                    aligned,
                    target,
                    global_valid,
                    limit=run.symmetric_metric_points,
                    device=run.device,
                ),
                "global_worse_frames": worse_frames,
                "modified_background_points": modified_background,
                "background_exact_raw": int(modified_background == 0),
                "displacement_median": _finite_quantile(displacement, 0.50),
                "displacement_p95": _finite_quantile(displacement, 0.95),
                "displacement_max": (
                    float(displacement.max()) if displacement.numel() else 0.0
                ),
                "bounded_displacement_pass": int(
                    not displacement.numel()
                    or float(displacement.max()) <= max_displacement * 1.0001
                ),
                "mean_normal_residual_before": _ratio_float(
                    chunk.residual_before_sum, chunk.valid_surfel_points
                ),
                "mean_normal_residual_after": _ratio_float(
                    chunk.residual_after_sum, chunk.valid_surfel_points
                ),
            }
        )
        for slot in eligible_slots:
            mask = global_valid & interior_masks[:, slot]
            raw_value = _weighted_rmse(raw_aligned, target, confidence, mask)
            value = _weighted_rmse(aligned, target, confidence, mask)
            per_track_rows.append(
                {
                    "branch": branch,
                    "slot": slot,
                    "prompt": prompts[slot],
                    "paired_points": int(mask.sum()),
                    "raw_weighted_rmse": raw_value,
                    "candidate_weighted_rmse": value,
                    "gain_vs_raw_percent": _gain(raw_value, value),
                    "track_improved": int(value < raw_value),
                }
            )
    return branch_rows, per_track_rows


def _final_decision(
    *,
    i0_decision: dict[str, object],
    branch_rows: list[dict[str, object]],
    per_track_rows: list[dict[str, object]],
    max_displacement: float,
) -> dict[str, object]:
    if not int(i0_decision["i0_pass"]):
        return {
            **i0_decision,
            "i1_run": 0,
            "i1_pass": 0,
            "selected_pointmap_modified": 0,
            "claim": "stop_after_instance_feasibility_audit",
            "next_gate": "add_valid_local_instances_or_keep_sam_semantic_only",
        }
    by_branch = {str(row["branch"]): row for row in branch_rows}
    correct = by_branch["correct_id"]
    controls = [by_branch[name] for name in I1_BRANCHES if name not in {"raw", "correct_id"}]
    correct_best = all(
        float(correct["instance_weighted_rmse"])
        < float(row["instance_weighted_rmse"])
        for row in controls
    )
    improved_tracks = sum(
        int(row["track_improved"])
        for row in per_track_rows
        if row["branch"] == "correct_id"
    )
    passed = int(
        float(correct["global_gain_vs_raw_percent"]) >= 0.0
        and float(correct["instance_gain_vs_raw_percent"]) > 0.0
        and improved_tracks >= 2
        and correct_best
        and int(correct["background_exact_raw"]) == 1
        and int(correct["bounded_displacement_pass"]) == 1
    )
    return {
        **i0_decision,
        "i1_run": 1,
        "correct_global_non_degradation_pass": int(
            float(correct["global_gain_vs_raw_percent"]) >= 0.0
        ),
        "correct_instance_gain_pass": int(
            float(correct["instance_gain_vs_raw_percent"]) > 0.0
        ),
        "correct_improved_track_count": improved_tracks,
        "correct_unique_vs_controls_pass": int(correct_best),
        "correct_background_exact_raw_pass": int(correct["background_exact_raw"]),
        "correct_bounded_displacement_pass": int(
            float(correct["displacement_max"]) <= max_displacement * 1.0001
        ),
        "i1_pass": passed,
        "selected_pointmap_modified": 0,
        "claim": (
            "sam_indexed_local_geometry_candidate_supported_not_deployed"
            if passed
            else "sam_indexed_local_geometry_improvement_not_established"
        ),
        "next_gate": (
            "review_candidate_then_consider_raw_safe_deployment"
            if passed
            else "keep_v0_raw_pointmap"
        ),
    }


def _write_outputs(
    *,
    run: V1Run,
    payload: dict[str, Any],
    baseline_summary: dict[str, Any],
    cache_path_value: Path,
    scene_diagonal: float,
    match_radius: float,
    max_displacement: float,
    shift_y: int,
    shift_x: int,
    track_rows: list[dict[str, object]],
    audit_rows: list[dict[str, object]],
    i0_branch_rows: list[dict[str, object]],
    branch_rows: list[dict[str, object]],
    per_track_rows: list[dict[str, object]],
    sparse_candidates: dict[str, tuple[torch.Tensor, torch.Tensor]],
    decision: dict[str, object],
) -> Path:
    run.output_dir.mkdir(parents=True, exist_ok=True)
    _clear_stale_outputs(run.output_dir)
    _write_csv(run.output_dir / "track_eligibility.csv", track_rows)
    _write_csv(run.output_dir / "i0_frame_audit.csv", audit_rows)
    _write_csv(run.output_dir / "i0_branch_summary.csv", i0_branch_rows)
    if branch_rows:
        _write_csv(run.output_dir / "i1_branch_summary.csv", branch_rows)
        _write_csv(run.output_dir / "i1_track_summary.csv", per_track_rows)
        torch.save(
            {
                "schema": 1,
                "revision": REVISION,
                "clip": payload["clip_name"],
                "coordinate_frame": "streamvggt_raw_first_frame_reference_world",
                "candidate_generation_gt_fields": 0,
                "selected_pointmap_modified": 0,
                "candidates": {
                    branch: {"flat_indices": value[0], "raw_deltas": value[1]}
                    for branch, value in sparse_candidates.items()
                },
            },
            run.output_dir / "candidate_deltas.pt",
        )
    output = {
        "schema": 1,
        "revision": REVISION,
        "clip": payload["clip_name"],
        "config": str(run.source_path),
        "baseline_config": str(run.baseline_config),
        "baseline_revision": BASELINE_REVISION,
        "baseline_pose_branch": baseline_summary["selected_pose_branch"],
        "baseline_pointmap": "raw_full_history_world_pointmap",
        "prompts": tuple(str(value) for value in payload["instance_prompts"]),
        "frames": tuple(int(value) for value in payload["frame_indices"]),
        "model_trained": 0,
        "sam_hidden_features_used": 0,
        "pose_modified": 0,
        "history_candidate_writeback": 0,
        "candidate_generation_gt_fields": 0,
        "gt_role": "scoring_only_after_all_sparse_candidates_are_frozen",
        "cache": str(cache_path_value),
        "device": run.device,
        "scene_diagonal_raw": scene_diagonal,
        "match_radius_raw": match_radius,
        "max_displacement_raw": max_displacement,
        "shifted_mask_pixels": (shift_y, shift_x),
        "eligibility": {
            "max_active_instances": run.max_active_instances,
            "min_history_visible_frames": run.min_history_visible_frames,
            "min_track_score": run.min_track_score,
            "min_mask_area_ratio": run.min_mask_area_ratio,
            "max_mask_area_ratio": run.max_mask_area_ratio,
            "min_current_points": run.min_current_points,
            "point_confidence_threshold": run.point_confidence_threshold,
            "erosion_radius": run.erosion_radius,
        },
        "track_eligibility": track_rows,
        "i0_branches": i0_branch_rows,
        "i1_branches": branch_rows,
        "i1_tracks": per_track_rows,
        "decision": decision,
        "outputs": {
            "track_csv": str(run.output_dir / "track_eligibility.csv"),
            "i0_frame_csv": str(run.output_dir / "i0_frame_audit.csv"),
            "i0_branch_csv": str(run.output_dir / "i0_branch_summary.csv"),
            "i1_branch_csv": (
                str(run.output_dir / "i1_branch_summary.csv") if branch_rows else ""
            ),
            "i1_track_csv": (
                str(run.output_dir / "i1_track_summary.csv") if branch_rows else ""
            ),
            "candidates": (
                str(run.output_dir / "candidate_deltas.pt") if branch_rows else ""
            ),
        },
    }
    result = run.output_dir / "instance_geometry_summary.json"
    result.write_text(
        json.dumps(output, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf8",
    )
    _write_copyable(run.output_dir / "copyable_result.txt", output)
    print("V1 CAUSAL SAM-INDEXED INSTANCE GEOMETRY")
    print(
        f"  tracks={len(track_rows)} eligible={decision['eligible_track_count']} "
        f"I0={decision['i0_pass']} I1_run={decision['i1_run']} "
        f"I1_pass={decision['i1_pass']}"
    )
    print(f"  decision={json.dumps(decision, sort_keys=True)}")
    print(f"  copyable_report={run.output_dir / 'copyable_result.txt'}")
    return result


def _clear_stale_outputs(output_dir: Path) -> None:
    for name in (
        "track_eligibility.csv",
        "i0_frame_audit.csv",
        "i0_branch_summary.csv",
        "i1_branch_summary.csv",
        "i1_track_summary.csv",
        "candidate_deltas.pt",
        "instance_geometry_summary.json",
        "copyable_result.txt",
    ):
        path = output_dir / name
        if path.is_file():
            path.unlink()


def _write_copyable(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "===== COPYABLE_V1_INSTANCE_GEOMETRY_BEGIN =====",
        f"revision={summary['revision']}",
        f"clip={summary['clip']}",
        f"prompts={' '.join(summary['prompts'])}",
        f"frames={len(summary['frames'])}",
        "pose_modified=0",
        "history_candidate_writeback=0",
        "candidate_generation_gt_fields=0",
        f"scene_diagonal_raw={summary['scene_diagonal_raw']}",
        f"match_radius_raw={summary['match_radius_raw']}",
        f"max_displacement_raw={summary['max_displacement_raw']}",
        "",
        "slot,prompt,visible_frames,qualifying_frames,active_frames,dense_valid_interior_points,median_interior_area_ratio,eligible_track",
    ]
    for row in summary["track_eligibility"]:
        lines.append(
            ",".join(
                str(row[name])
                for name in (
                    "slot", "prompt", "visible_frames", "qualifying_frames",
                    "active_frames", "dense_valid_interior_points",
                    "median_interior_area_ratio", "eligible_track",
                )
            )
        )
    lines.extend(
        [
            "",
            "branch,selected_track_frames,active_tracks,selected_points,valid_surfel_rate,mean_nearest_median,mean_normal_residual_median,mean_surface_thickness_median",
        ]
    )
    for row in summary["i0_branches"]:
        lines.append(
            ",".join(
                str(row[name])
                for name in (
                    "branch", "selected_track_frames", "active_tracks",
                    "selected_points", "valid_surfel_rate", "mean_nearest_median",
                    "mean_normal_residual_median", "mean_surface_thickness_median",
                )
            )
        )
    if summary["i1_branches"]:
        lines.extend(
            [
                "",
                "branch,selected_points,modified_points,global_gain_vs_raw_percent,instance_gain_vs_raw_percent,global_worse_frames,modified_background_points,displacement_p95",
            ]
        )
        for row in summary["i1_branches"]:
            lines.append(
                ",".join(
                    str(row[name])
                    for name in (
                        "branch", "selected_points", "modified_points",
                        "global_gain_vs_raw_percent", "instance_gain_vs_raw_percent",
                        "global_worse_frames", "modified_background_points",
                        "displacement_p95",
                    )
                )
            )
    lines.extend(
        [
            "",
            f"decision={json.dumps(summary['decision'], sort_keys=True)}",
            "",
            f"summary={path.with_name('instance_geometry_summary.json')}",
            f"track_csv={summary['outputs']['track_csv']}",
            f"i0_frame_csv={summary['outputs']['i0_frame_csv']}",
            f"i0_branch_csv={summary['outputs']['i0_branch_csv']}",
            f"i1_branch_csv={summary['outputs']['i1_branch_csv']}",
            f"i1_track_csv={summary['outputs']['i1_track_csv']}",
            f"candidates={summary['outputs']['candidates']}",
            "===== COPYABLE_V1_INSTANCE_GEOMETRY_END =====",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def _validate_inputs(
    *,
    payload: dict[str, Any],
    clip: ClipConfig,
    baseline_output_dir: Path,
) -> dict[str, Any]:
    required = (
        "baseline_world_points", "baseline_world_confidence",
        "tracking_masks_stream", "tracking_scores", "sam_track_ids",
        "sam_track_prompts", "sam_birth_indices", "instance_prompts",
        "target_world_points", "point_alignment_scale",
        "point_alignment_rotation", "point_alignment_translation",
    )
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError(f"V0 cache lacks V1 fields={missing}.")
    if tuple(str(value) for value in payload["instance_prompts"]) != tuple(
        clip.instance_prompts
    ):
        raise ValueError("V0 cache prompt signature differs from the config.")
    if "bed" in {str(value).strip().lower() for value in payload["instance_prompts"]}:
        raise ValueError("V1 local-instance audit excludes the scene-dominant bed prompt.")
    summary_path = baseline_output_dir / "baseline_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError("Run commands_v0_baseline.txt before V1.")
    summary = json.loads(summary_path.read_text(encoding="utf8"))
    expected = {
        "schema": 6,
        "implementation_revision": BASELINE_REVISION,
        "selected_pose_branch": "retrieve_qk",
        "formal_pointmap_output": "raw_full_history_world_pointmap",
        "tracking_baseline_acceptance_pass": 1,
    }
    for name, value in expected.items():
        if summary.get(name) != value:
            raise ValueError(
                f"Baseline summary {name}={summary.get(name)!r}; expected {value!r}."
            )
    if tuple(summary.get("configured_instance_prompts", ())) != tuple(
        clip.instance_prompts
    ):
        raise ValueError("Baseline summary prompt signature is stale.")
    return summary


def _discovered_slots(payload: dict[str, Any]) -> tuple[int, ...]:
    return tuple(
        slot
        for slot, (track_id, prompt, birth) in enumerate(
            zip(
                payload["sam_track_ids"],
                payload["sam_track_prompts"],
                payload["sam_birth_indices"],
            )
        )
        if int(track_id) >= 0 and int(birth) >= 0 and str(prompt).strip()
    )


def _scene_diagonal(points: torch.Tensor, valid: torch.Tensor) -> float:
    selected = points[valid]
    if selected.shape[0] < 256:
        raise ValueError("Scene scale needs at least 256 valid raw points.")
    if selected.shape[0] > 100_000:
        indices = torch.linspace(0, selected.shape[0] - 1, steps=100_000).long()
        selected = selected.index_select(0, indices)
    low = torch.quantile(selected, 0.05, dim=0)
    high = torch.quantile(selected, 0.95, dim=0)
    diagonal = float(torch.linalg.vector_norm(high - low))
    if not math.isfinite(diagonal) or diagonal <= 0.0:
        raise ValueError("Raw scene diagonal is invalid.")
    return diagonal


def _weighted_rmse(
    predicted: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    if not bool(mask.any()):
        return float("nan")
    distance2 = (predicted[mask] - target[mask]).square().sum(dim=-1)
    selected_weights = weights[mask].clamp_min(0.0)
    return float(
        torch.sqrt(
            (selected_weights * distance2).sum()
            / selected_weights.sum().clamp_min(1e-8)
        )
    )


def _frame_rmse(
    predicted: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    mask: torch.Tensor,
) -> list[float]:
    return [
        _weighted_rmse(predicted[index], target[index], weights[index], mask[index])
        for index in range(predicted.shape[0])
    ]


def _symmetric_mean(
    predicted: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    *,
    limit: int,
    device: str,
) -> float:
    source = predicted[valid]
    destination = target[valid]
    if not source.numel():
        return float("nan")
    if source.shape[0] > int(limit):
        indices = torch.linspace(0, source.shape[0] - 1, steps=int(limit)).long()
        source = source.index_select(0, indices)
        destination = destination.index_select(0, indices)
    distances = torch.cdist(source.to(device), destination.to(device))
    return float(0.5 * (distances.min(dim=1).values.mean() + distances.min(dim=0).values.mean()))


def _weighted_row_mean(
    rows: Iterable[dict[str, object]],
    value_name: str,
    weight_name: str,
) -> float:
    weighted = []
    weights = []
    for row in rows:
        value = float(row[value_name])
        weight = int(row[weight_name])
        if math.isfinite(value) and weight > 0:
            weighted.append(value * weight)
            weights.append(weight)
    return sum(weighted) / sum(weights) if weights else float("nan")


def _finite_quantile(value: torch.Tensor, quantile: float) -> float:
    value = value.detach().float().cpu().reshape(-1)
    value = value[torch.isfinite(value)]
    return float(torch.quantile(value, float(quantile))) if value.numel() else float("nan")


def _gain(raw: float, candidate: float) -> float:
    if not math.isfinite(raw) or not math.isfinite(candidate) or raw <= 0.0:
        return float("nan")
    return 100.0 * (raw - candidate) / raw


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _ratio_float(numerator: float, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else float("nan")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_run(path: str | Path) -> V1Run:
    source = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf8")) or {}
    eligibility = raw.get("eligibility", {})
    surfel = raw.get("surfel", {})
    controls = raw.get("controls", {})
    evaluation = raw.get("evaluation", {})
    run = V1Run(
        source_path=source,
        baseline_config=Path(raw["baseline_config"]).expanduser().resolve(),
        output_dir=Path(raw["output_dir"]).expanduser().resolve(),
        device=str(raw.get("device", "cuda:0")),
        max_active_instances=int(eligibility.get("max_active_instances", 5)),
        min_history_visible_frames=int(
            eligibility.get("min_history_visible_frames", 4)
        ),
        min_track_score=float(eligibility.get("min_track_score", 0.5)),
        min_mask_area_ratio=float(eligibility.get("min_mask_area_ratio", 0.005)),
        max_mask_area_ratio=float(eligibility.get("max_mask_area_ratio", 0.25)),
        min_current_points=int(eligibility.get("min_current_points", 256)),
        point_confidence_threshold=float(
            eligibility.get("point_confidence_threshold", 0.30)
        ),
        erosion_radius=int(eligibility.get("erosion_radius", 2)),
        neighbors=int(surfel.get("neighbors", 16)),
        min_support_frames=int(surfel.get("min_support_frames", 3)),
        max_current_points_per_instance=int(
            surfel.get("max_current_points_per_instance", 1024)
        ),
        max_history_points_per_frame=int(
            surfel.get("max_history_points_per_frame", 512)
        ),
        max_history_points_total=int(surfel.get("max_history_points_total", 8192)),
        match_radius_scene_fraction=float(
            surfel.get("match_radius_scene_fraction", 0.01)
        ),
        normal_variance_max=float(surfel.get("normal_variance_max", 0.20)),
        alpha=float(surfel.get("alpha", 0.50)),
        max_displacement_scene_fraction=float(
            surfel.get("max_displacement_scene_fraction", 0.0025)
        ),
        chunk_size=int(surfel.get("chunk_size", 256)),
        shifted_mask_y_fraction=float(
            controls.get("shifted_mask_y_fraction", 0.0625)
        ),
        shifted_mask_x_fraction=float(
            controls.get("shifted_mask_x_fraction", 0.125)
        ),
        symmetric_metric_points=int(evaluation.get("symmetric_metric_points", 2048)),
    )
    _validate_run(run)
    return run


def _validate_run(run: V1Run) -> None:
    if run.max_active_instances < 1 or run.min_history_visible_frames < 1:
        raise ValueError("Instance count/history thresholds must be positive.")
    if not 0.0 <= run.min_track_score <= 1.0:
        raise ValueError("min_track_score must be in [0,1].")
    if not 0.0 <= run.min_mask_area_ratio < run.max_mask_area_ratio <= 1.0:
        raise ValueError("mask area ratio bounds are invalid.")
    positive = (
        run.min_current_points,
        run.neighbors,
        run.min_support_frames,
        run.max_current_points_per_instance,
        run.max_history_points_per_frame,
        run.max_history_points_total,
        run.chunk_size,
        run.symmetric_metric_points,
    )
    if any(value < 1 for value in positive):
        raise ValueError("All point/surfel counts must be positive.")
    if not 0.0 < run.match_radius_scene_fraction:
        raise ValueError("match radius fraction must be positive.")
    if not 0.0 < run.max_displacement_scene_fraction:
        raise ValueError("max displacement fraction must be positive.")
    if not 0.0 <= run.normal_variance_max <= 1.0:
        raise ValueError("normal_variance_max must be in [0,1].")
    if not 0.0 < run.alpha <= 1.0:
        raise ValueError("surfel alpha must be in (0,1].")


def _replace_device(run: V1Run, device: str) -> V1Run:
    return V1Run(**{**run.__dict__, "device": str(device)})


def _find_clip(clips: tuple[ClipConfig, ...], name: str) -> ClipConfig:
    selected = [clip for clip in clips if clip.name == name]
    if len(selected) != 1:
        raise ValueError(f"Clip {name!r} was not found exactly once.")
    return selected[0]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="streaming_couping/configs/v1_instance_geometry.yaml",
    )
    parser.add_argument("--device")
    return parser.parse_args()


if __name__ == "__main__":
    main()
