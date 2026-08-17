#!/usr/bin/env python3
"""Generate and score one fixed-radius SAM-indexed pointmap candidate."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Sequence

import torch
import yaml

from streaming_couping.src.instance_geometry import (
    SurfelQueryResult,
    apply_sparse_deltas,
    bounded_surfel_query,
    erode_instance_masks,
    merge_sparse_deltas,
    shift_instance_masks,
)
from streaming_couping.src.learned_pose.baseline_runtime import (
    load_baseline_run_config,
)
from streaming_couping.src.learned_pose.cache import cache_path, load_feature_cache
from streaming_couping.src.learned_pose.config import load_learned_pose_config
from streaming_couping.src.semantic_map import normalize_confidence, semantic_slot_map
from streaming_couping.scripts.run_v1_correspondence_sweep import (
    REVISION as SWEEP_REVISION,
    EqualSupportUnit,
    SweepRun,
    _build_equal_support_units,
    _load_sweep,
    _raw_input_projection,
    _validate_raw_inputs,
)
from streaming_couping.scripts.run_v1_instance_geometry import (
    I0_BRANCHES,
    V1Run,
    _active_instance_pairs,
    _discovered_slots,
    _find_clip,
    _finite_quantile,
    _frame_rmse,
    _gain,
    _ratio,
    _ratio_float,
    _scene_diagonal,
    _symmetric_mean,
    _weighted_rmse,
    geometry_excluded_slots,
)


REVISION = "v1_fixed_equal_support_instance_surfel_i1_r1"
BRANCHES = ("raw",) + I0_BRANCHES
GT_FIELDS = (
    "target_world_points",
    "point_alignment_scale",
    "point_alignment_rotation",
    "point_alignment_translation",
)


@dataclass(frozen=True)
class I1Run:
    source_path: Path
    sweep: SweepRun
    source_sweep_summary: Path
    output_dir: Path
    fixed_radius_scene_fraction: float
    minimum_improved_tracks: int


@dataclass
class CandidateAccumulator:
    indices: list[torch.Tensor] = field(default_factory=list)
    deltas: list[torch.Tensor] = field(default_factory=list)
    selected_points: int = 0
    valid_surfel_points: int = 0
    residual_before_sum: float = 0.0
    residual_after_sum: float = 0.0
    cap_hits: int = 0

    def add(
        self,
        *,
        global_indices: torch.Tensor,
        query: SurfelQueryResult,
        max_displacement: float,
    ) -> None:
        self.selected_points += int(global_indices.numel())
        valid = query.valid
        count = int(valid.sum())
        self.valid_surfel_points += count
        if not count:
            return
        valid_indices = global_indices[valid].detach().long().cpu()
        valid_deltas = query.delta[valid].detach().float().cpu()
        displacement = torch.linalg.vector_norm(valid_deltas, dim=-1)
        before = query.normal_residual[valid].detach().float().cpu()
        self.indices.append(valid_indices)
        self.deltas.append(valid_deltas)
        self.residual_before_sum += float(before.sum())
        self.residual_after_sum += float((before - displacement).abs().sum())
        self.cap_hits += int(
            (displacement >= float(max_displacement) * 0.9999).sum()
        )


def main() -> None:
    args = _parse_args()
    experiment = _load_i1(args.config)
    run = experiment.sweep.base
    if args.device:
        run = V1Run(**{**run.__dict__, "device": str(args.device)})
        sweep = SweepRun(**{**experiment.sweep.__dict__, "base": run})
        experiment = I1Run(**{**experiment.__dict__, "sweep": sweep})

    source_sweep = _validate_source_sweep(experiment)
    data = load_learned_pose_config(run.baseline_config)
    baseline = load_baseline_run_config(run.baseline_config)
    clip = _find_clip(data.clips, baseline.clip_name)
    cache_path_value = cache_path(data, clip)
    if Path(source_sweep["cache"]).resolve() != cache_path_value.resolve():
        raise ValueError("Passed sweep and fixed I1 do not reference the same cache.")

    # Candidate generation only sees this explicit raw/SAM allowlist.
    payload = _raw_input_projection(load_feature_cache(cache_path_value))
    baseline_summary = _validate_raw_inputs(
        payload=payload,
        clip=clip,
        baseline_output_dir=baseline.output_dir,
    )
    prepared = _prepare_units(payload=payload, run=run)
    if len(prepared.units) != int(
        source_sweep["equal_support"]["accepted_units"]
    ):
        raise ValueError(
            "Replayed equal-support unit count differs from the passed sweep."
        )

    radius_raw = (
        prepared.scene_diagonal * experiment.fixed_radius_scene_fraction
    )
    max_displacement = (
        prepared.scene_diagonal * run.max_displacement_scene_fraction
    )
    candidates, accumulators, diagnostic_rows = _generate_candidates(
        units=prepared.units,
        image_pixels=prepared.points.shape[1] * prepared.points.shape[2],
        run=run,
        radius_raw=radius_raw,
        max_displacement=max_displacement,
    )
    _validate_candidate_replay(
        source_sweep=source_sweep,
        fixed_radius=experiment.fixed_radius_scene_fraction,
        accumulators=accumulators,
    )

    experiment.output_dir.mkdir(parents=True, exist_ok=True)
    _clear_outputs(experiment.output_dir)
    candidate_path = experiment.output_dir / "candidate_deltas.pt"
    torch.save(
        {
            "schema": 1,
            "revision": REVISION,
            "clip": payload["clip_name"],
            "source_sweep_revision": SWEEP_REVISION,
            "fixed_radius_scene_fraction": (
                experiment.fixed_radius_scene_fraction
            ),
            "radius_raw": radius_raw,
            "max_displacement_raw": max_displacement,
            "coordinate_frame": "streamvggt_raw_first_frame_reference_world",
            "candidate_generation_fields": (
                "raw_world_pointmap",
                "raw_confidence",
                "sam_persistent_masks_and_ids",
            ),
            "candidate_generation_gt_fields": 0,
            "target_geometry_fields_accessed_before_freeze": 0,
            "history_candidate_writeback": 0,
            "selected_pointmap_modified": 0,
            "candidates": {
                branch: {
                    "flat_indices": value[0],
                    "raw_deltas": value[1],
                }
                for branch, value in candidates.items()
            },
        },
        candidate_path,
    )
    if not candidate_path.is_file() or candidate_path.stat().st_size == 0:
        raise RuntimeError("Candidate artifact was not frozen before GT scoring.")

    # GT fields are first indexed here, after candidate_deltas.pt exists.
    scoring_payload = load_feature_cache(cache_path_value)
    missing_gt = [name for name in GT_FIELDS if name not in scoring_payload]
    if missing_gt:
        raise ValueError(f"V0 cache lacks post-freeze scoring fields={missing_gt}.")
    branch_rows, track_rows = _score_candidates(
        points=prepared.points,
        confidence=prepared.confidence,
        valid_points=prepared.valid_points,
        interior_masks=prepared.interior_masks,
        eligible_slots=prepared.eligible_slots,
        prompts=prepared.prompts,
        target_points=scoring_payload["target_world_points"],
        alignment_scale=float(scoring_payload["point_alignment_scale"]),
        alignment_rotation=scoring_payload["point_alignment_rotation"],
        alignment_translation=scoring_payload["point_alignment_translation"],
        candidates=candidates,
        accumulators=accumulators,
        run=run,
        max_displacement=max_displacement,
    )
    decision = _i1_decision(
        branch_rows=branch_rows,
        track_rows=track_rows,
        minimum_improved_tracks=experiment.minimum_improved_tracks,
    )
    result = _write_outputs(
        experiment=experiment,
        payload=payload,
        baseline_summary=baseline_summary,
        cache_path_value=cache_path_value,
        source_sweep=source_sweep,
        prepared=prepared,
        radius_raw=radius_raw,
        max_displacement=max_displacement,
        diagnostic_rows=diagnostic_rows,
        branch_rows=branch_rows,
        track_rows=track_rows,
        decision=decision,
        candidate_path=candidate_path,
    )
    print(f"V1 fixed correspondence I1 result={result}")


@dataclass(frozen=True)
class PreparedInputs:
    points: torch.Tensor
    confidence: torch.Tensor
    valid_points: torch.Tensor
    interior_masks: torch.Tensor
    prompts: tuple[str, ...]
    eligible_slots: tuple[int, ...]
    units: tuple[EqualSupportUnit, ...]
    skipped_units: dict[str, int]
    eligibility_rows: list[dict[str, object]]
    scene_diagonal: float
    shift_y: int
    shift_x: int


def _prepare_units(*, payload: dict[str, Any], run: V1Run) -> PreparedInputs:
    points = payload["baseline_world_points"].detach().float().cpu()
    confidence = normalize_confidence(payload["baseline_world_confidence"])
    masks = payload["tracking_masks_stream"].detach().bool().cpu()
    scores = payload["tracking_scores"].detach().float().cpu()
    prompts = tuple(str(value) for value in payload["sam_track_prompts"])
    geometry_masks = masks.clone()
    excluded_slots = geometry_excluded_slots(prompts, run.geometry_excluded_prompts)
    if excluded_slots:
        geometry_masks[:, list(excluded_slots)] = False
    slot_map = semantic_slot_map(
        geometry_masks,
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
    discovered_slots = _discovered_slots(payload)
    active_pairs, eligibility_rows = _active_instance_pairs(
        payload=payload,
        scores=scores,
        interior_masks=interior_masks,
        valid_points=valid_points,
        discovered_slots=discovered_slots,
        run=run,
    )
    eligible_slots = tuple(
        int(row["slot"])
        for row in eligibility_rows
        if int(row["eligible_track"])
    )
    units, skipped_units = _build_equal_support_units(
        points=points,
        confidence=confidence,
        valid_points=valid_points,
        interior_masks=interior_masks,
        shifted_masks=shifted_masks,
        active_pairs=active_pairs,
        eligible_slots=eligible_slots,
        prompts=prompts,
        frame_indices=tuple(int(value) for value in payload["frame_indices"]),
        run=run,
    )
    return PreparedInputs(
        points=points,
        confidence=confidence,
        valid_points=valid_points,
        interior_masks=interior_masks,
        prompts=prompts,
        eligible_slots=eligible_slots,
        units=tuple(units),
        skipped_units=skipped_units,
        eligibility_rows=eligibility_rows,
        scene_diagonal=scene_diagonal,
        shift_y=shift_y,
        shift_x=shift_x,
    )


def _generate_candidates(
    *,
    units: Sequence[EqualSupportUnit],
    image_pixels: int,
    run: V1Run,
    radius_raw: float,
    max_displacement: float,
) -> tuple[
    dict[str, tuple[torch.Tensor, torch.Tensor]],
    dict[str, CandidateAccumulator],
    list[dict[str, object]],
]:
    accumulators = {
        branch: CandidateAccumulator() for branch in I0_BRANCHES
    }
    rows: list[dict[str, object]] = []
    for unit in units:
        for branch in I0_BRANCHES:
            if branch == "shifted_mask":
                current = unit.shifted_current
                local_indices = unit.shifted_indices
            else:
                current = unit.correct_current
                local_indices = unit.correct_indices
            history, weights, history_frames = unit.histories[branch]
            query = bounded_surfel_query(
                current_points=current,
                history_points=history,
                history_weights=weights,
                history_frame_ids=history_frames,
                device=run.device,
                neighbors=run.neighbors,
                min_support_frames=run.min_support_frames,
                match_radius=radius_raw,
                normal_variance_max=run.normal_variance_max,
                alpha=run.alpha,
                max_displacement=max_displacement,
                chunk_size=run.chunk_size,
            )
            global_indices = local_indices + unit.sequence_index * image_pixels
            accumulators[branch].add(
                global_indices=global_indices,
                query=query,
                max_displacement=max_displacement,
            )
            displacement = torch.linalg.vector_norm(query.delta, dim=-1)
            rows.append(
                {
                    "branch": branch,
                    "sequence_index": unit.sequence_index,
                    "frame_index": unit.frame_index,
                    "slot": unit.slot,
                    "history_slot": (
                        unit.shuffled_history_slot
                        if branch == "shuffled_id"
                        else unit.slot
                    ),
                    "prompt": unit.prompt,
                    "selected_points": unit.current_points,
                    "history_points": unit.history_points,
                    "valid_surfel_points": int(query.valid.sum()),
                    "valid_surfel_rate": _ratio(
                        int(query.valid.sum()), unit.current_points
                    ),
                    "nonzero_delta_points": int(
                        (displacement > 0.0).sum()
                    ),
                    "cap_hit_points": int(
                        (
                            query.valid
                            & (displacement >= max_displacement * 0.9999)
                        ).sum()
                    ),
                }
            )
    candidates = {
        "raw": (torch.empty(0, dtype=torch.long), torch.empty(0, 3))
    }
    for branch, accumulator in accumulators.items():
        candidates[branch] = merge_sparse_deltas(
            accumulator.indices, accumulator.deltas
        )
    return candidates, accumulators, rows


def _score_candidates(
    *,
    points: torch.Tensor,
    confidence: torch.Tensor,
    valid_points: torch.Tensor,
    interior_masks: torch.Tensor,
    eligible_slots: tuple[int, ...],
    prompts: tuple[str, ...],
    target_points: torch.Tensor,
    alignment_scale: float,
    alignment_rotation: torch.Tensor,
    alignment_translation: torch.Tensor,
    candidates: dict[str, tuple[torch.Tensor, torch.Tensor]],
    accumulators: dict[str, CandidateAccumulator],
    run: V1Run,
    max_displacement: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    target = target_points.detach().float().cpu()
    rotation = alignment_rotation.detach().float().cpu()
    translation = alignment_translation.detach().float().cpu()
    target_valid = torch.isfinite(target).all(dim=-1)
    global_valid = valid_points & target_valid & torch.isfinite(confidence)
    instance_union = (
        interior_masks[:, list(eligible_slots)].any(dim=1)
        if eligible_slots
        else torch.zeros_like(global_valid)
    )
    instance_valid = global_valid & instance_union
    correct_support = torch.zeros_like(global_valid).reshape(-1)
    correct_support_indices = candidates["correct_id"][0]
    correct_support[correct_support_indices] = True
    correct_support = correct_support.reshape_as(global_valid) & global_valid

    raw_aligned = float(alignment_scale) * (points @ rotation.T) + translation
    raw_global = _weighted_rmse(raw_aligned, target, confidence, global_valid)
    raw_instance = _weighted_rmse(raw_aligned, target, confidence, instance_valid)
    raw_support = _weighted_rmse(raw_aligned, target, confidence, correct_support)
    raw_frame = _frame_rmse(raw_aligned, target, confidence, global_valid)
    raw_instance_frame = _frame_rmse(
        raw_aligned, target, confidence, instance_valid
    )

    branch_rows: list[dict[str, object]] = []
    track_rows: list[dict[str, object]] = []
    for branch in BRANCHES:
        indices, deltas = candidates[branch]
        candidate = apply_sparse_deltas(points, indices, deltas)
        aligned = float(alignment_scale) * (candidate @ rotation.T) + translation
        global_rmse = _weighted_rmse(aligned, target, confidence, global_valid)
        instance_rmse = _weighted_rmse(
            aligned, target, confidence, instance_valid
        )
        support_rmse = _weighted_rmse(
            aligned, target, confidence, correct_support
        )
        frame_rmse = _frame_rmse(aligned, target, confidence, global_valid)
        instance_frame_rmse = _frame_rmse(
            aligned, target, confidence, instance_valid
        )
        displacement = (
            torch.linalg.vector_norm(deltas, dim=-1)
            if deltas.numel()
            else torch.empty(0)
        )
        nonzero = int((displacement > 0.0).sum())
        accumulator = accumulators.get(branch, CandidateAccumulator())
        modified_background = (
            int((~instance_union.reshape(-1).index_select(0, indices)).sum())
            if indices.numel()
            else 0
        )
        branch_rows.append(
            {
                "branch": branch,
                "selected_points": accumulator.selected_points,
                "candidate_support_points": int(indices.numel()),
                "nonzero_modified_points": nonzero,
                "valid_surfel_rate": _ratio(
                    accumulator.valid_surfel_points,
                    accumulator.selected_points,
                ),
                "global_weighted_rmse": global_rmse,
                "global_gain_vs_raw_percent": _gain(raw_global, global_rmse),
                "instance_weighted_rmse": instance_rmse,
                "instance_gain_vs_raw_percent": _gain(
                    raw_instance, instance_rmse
                ),
                "correct_support_weighted_rmse": support_rmse,
                "correct_support_gain_vs_raw_percent": _gain(
                    raw_support, support_rmse
                ),
                "symmetric_mean": _symmetric_mean(
                    aligned,
                    target,
                    global_valid,
                    limit=run.symmetric_metric_points,
                    device=run.device,
                ),
                "global_worse_frames": _worse_frames(raw_frame, frame_rmse),
                "instance_worse_frames": _worse_frames(
                    raw_instance_frame, instance_frame_rmse
                ),
                "modified_background_points": modified_background,
                "background_exact_raw": int(modified_background == 0),
                "displacement_median": _finite_quantile(displacement, 0.50),
                "displacement_p95": _finite_quantile(displacement, 0.95),
                "displacement_max": (
                    float(displacement.max()) if displacement.numel() else 0.0
                ),
                "cap_hit_points": accumulator.cap_hits,
                "cap_hit_rate": _ratio(
                    accumulator.cap_hits, accumulator.valid_surfel_points
                ),
                "bounded_displacement_pass": int(
                    not displacement.numel()
                    or float(displacement.max()) <= max_displacement * 1.0001
                ),
                "mean_normal_residual_before": _ratio_float(
                    accumulator.residual_before_sum,
                    accumulator.valid_surfel_points,
                ),
                "mean_normal_residual_after": _ratio_float(
                    accumulator.residual_after_sum,
                    accumulator.valid_surfel_points,
                ),
            }
        )
        for slot in eligible_slots:
            mask = global_valid & interior_masks[:, slot]
            raw_value = _weighted_rmse(raw_aligned, target, confidence, mask)
            value = _weighted_rmse(aligned, target, confidence, mask)
            track_rows.append(
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
    return branch_rows, track_rows


def _validate_candidate_replay(
    *,
    source_sweep: dict[str, Any],
    fixed_radius: float,
    accumulators: dict[str, CandidateAccumulator],
) -> None:
    source_rows = {
        str(row["branch"]): row
        for row in source_sweep["radius_branches"]
        if math.isclose(
            float(row["radius_scene_fraction"]),
            float(fixed_radius),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    }
    if set(source_rows) != set(I0_BRANCHES):
        raise ValueError("Passed sweep lacks the fixed-radius branch summaries.")
    for branch, accumulator in accumulators.items():
        source = source_rows[branch]
        expected = (
            int(source["selected_points"]),
            int(source["valid_surfel_points"]),
        )
        replayed = (
            accumulator.selected_points,
            accumulator.valid_surfel_points,
        )
        if replayed != expected:
            raise ValueError(
                f"Fixed candidate replay differs for {branch}: "
                f"replayed={replayed} expected={expected}."
            )


def _i1_decision(
    *,
    branch_rows: Sequence[dict[str, object]],
    track_rows: Sequence[dict[str, object]],
    minimum_improved_tracks: int,
) -> dict[str, object]:
    by_branch = {str(row["branch"]): row for row in branch_rows}
    correct = by_branch["correct_id"]
    controls = [by_branch["shuffled_id"], by_branch["shifted_mask"]]
    improved_tracks = sum(
        int(row["track_improved"])
        for row in track_rows
        if row["branch"] == "correct_id"
    )
    unique = all(
        float(correct["instance_weighted_rmse"])
        < float(row["instance_weighted_rmse"])
        for row in controls
    )
    passed = int(
        float(correct["global_gain_vs_raw_percent"]) >= 0.0
        and float(correct["instance_gain_vs_raw_percent"]) > 0.0
        and improved_tracks >= int(minimum_improved_tracks)
        and unique
        and int(correct["background_exact_raw"]) == 1
        and int(correct["bounded_displacement_pass"]) == 1
    )
    return {
        "fixed_candidate_generated": 1,
        "candidate_frozen_before_gt_access": 1,
        "candidate_generation_gt_fields": 0,
        "correct_global_non_degradation_pass": int(
            float(correct["global_gain_vs_raw_percent"]) >= 0.0
        ),
        "correct_instance_gain_pass": int(
            float(correct["instance_gain_vs_raw_percent"]) > 0.0
        ),
        "correct_improved_track_count": improved_tracks,
        "minimum_improved_tracks": int(minimum_improved_tracks),
        "correct_minimum_improved_tracks_pass": int(
            improved_tracks >= int(minimum_improved_tracks)
        ),
        "correct_unique_vs_controls_pass": int(unique),
        "correct_background_exact_raw_pass": int(
            correct["background_exact_raw"]
        ),
        "correct_bounded_displacement_pass": int(
            correct["bounded_displacement_pass"]
        ),
        "i1_pass": passed,
        "selected_pointmap_modified": 0,
        "claim": (
            "fixed_sam_indexed_pointmap_candidate_supported_not_deployed"
            if passed
            else "fixed_pointwise_surfel_update_did_not_improve_pointmap"
        ),
        "next_gate": (
            "cross_sequence_validation_before_any_deployment"
            if passed
            else "stop_pointwise_update_consider_shared_rigid_se3_factor"
        ),
    }


def _write_outputs(
    *,
    experiment: I1Run,
    payload: dict[str, Any],
    baseline_summary: dict[str, Any],
    cache_path_value: Path,
    source_sweep: dict[str, Any],
    prepared: PreparedInputs,
    radius_raw: float,
    max_displacement: float,
    diagnostic_rows: list[dict[str, object]],
    branch_rows: list[dict[str, object]],
    track_rows: list[dict[str, object]],
    decision: dict[str, object],
    candidate_path: Path,
) -> Path:
    _write_csv(experiment.output_dir / "candidate_diagnostics.csv", diagnostic_rows)
    _write_csv(experiment.output_dir / "branch_summary.csv", branch_rows)
    _write_csv(experiment.output_dir / "track_summary.csv", track_rows)
    _write_csv(
        experiment.output_dir / "track_eligibility.csv",
        prepared.eligibility_rows,
    )
    output = {
        "schema": 1,
        "revision": REVISION,
        "clip": payload["clip_name"],
        "config": str(experiment.source_path),
        "source_correspondence_config": str(
            experiment.sweep.base.source_path
        ),
        "source_sweep_summary": str(experiment.source_sweep_summary),
        "source_sweep_revision": source_sweep["revision"],
        "source_sweep_pass": int(
            source_sweep["decision"]["correspondence_sweep_pass"]
        ),
        "baseline_revision": baseline_summary["implementation_revision"],
        "baseline_pose_branch": baseline_summary["selected_pose_branch"],
        "baseline_pointmap": "raw_full_history_world_pointmap",
        "cache": str(cache_path_value),
        "frames": tuple(int(value) for value in payload["frame_indices"]),
        "branches": BRANCHES,
        "fixed_radius_scene_fraction": (
            experiment.fixed_radius_scene_fraction
        ),
        "radius_raw": radius_raw,
        "max_displacement_scene_fraction": (
            experiment.sweep.base.max_displacement_scene_fraction
        ),
        "max_displacement_raw": max_displacement,
        "alpha": experiment.sweep.base.alpha,
        "accepted_equal_support_units": len(prepared.units),
        "eligible_track_count": len(prepared.eligible_slots),
        "eligible_slots": prepared.eligible_slots,
        "skipped_units": prepared.skipped_units,
        "model_loaded_or_run": 0,
        "model_trained": 0,
        "sam_rerun": 0,
        "streamvggt_rerun": 0,
        "pose_modified": 0,
        "history_candidate_writeback": 0,
        "pointmap_candidate_generated": 1,
        "candidate_frozen_before_gt_access": 1,
        "candidate_generation_gt_fields": 0,
        "target_geometry_fields_accessed_before_freeze": 0,
        "target_geometry_fields_accessed_after_freeze": len(GT_FIELDS),
        "gt_role": "scoring_only_after_candidate_artifact_is_frozen",
        "selected_pointmap_modified": 0,
        "eligibility": source_sweep["eligibility"],
        "branch_results": branch_rows,
        "track_results": track_rows,
        "decision": decision,
        "outputs": {
            "candidate": str(candidate_path),
            "diagnostics_csv": str(
                experiment.output_dir / "candidate_diagnostics.csv"
            ),
            "branch_csv": str(experiment.output_dir / "branch_summary.csv"),
            "track_csv": str(experiment.output_dir / "track_summary.csv"),
            "eligibility_csv": str(
                experiment.output_dir / "track_eligibility.csv"
            ),
            "copyable_report": str(
                experiment.output_dir / "copyable_result.txt"
            ),
        },
    }
    result = experiment.output_dir / "fixed_correspondence_i1_summary.json"
    result.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf8"
    )
    _write_copyable(experiment.output_dir / "copyable_result.txt", output)
    correct = next(row for row in branch_rows if row["branch"] == "correct_id")
    print("V1 FIXED CORRESPONDENCE POINTMAP I1")
    print(
        f"  radius={experiment.fixed_radius_scene_fraction:.3f} "
        f"units={len(prepared.units)} tracks={len(prepared.eligible_slots)} "
        f"valid={float(correct['valid_surfel_rate']):.4f}"
    )
    print(
        f"  correct global_gain={float(correct['global_gain_vs_raw_percent']):.6f}% "
        f"instance_gain={float(correct['instance_gain_vs_raw_percent']):.6f}% "
        f"support_gain={float(correct['correct_support_gain_vs_raw_percent']):.6f}%"
    )
    print(f"  decision={json.dumps(decision, sort_keys=True)}")
    print(f"  copyable_report={experiment.output_dir / 'copyable_result.txt'}")
    return result


def _write_copyable(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "===== COPYABLE_V1_FIXED_CORRESPONDENCE_I1_BEGIN =====",
        f"revision={summary['revision']}",
        f"source_sweep_revision={summary['source_sweep_revision']}",
        f"clip={summary['clip']}",
        f"frames={len(summary['frames'])}",
        f"branches={','.join(summary['branches'])}",
        f"fixed_radius_scene_fraction={summary['fixed_radius_scene_fraction']}",
        f"radius_raw={summary['radius_raw']}",
        f"alpha={summary['alpha']}",
        f"max_displacement_raw={summary['max_displacement_raw']}",
        f"accepted_equal_support_units={summary['accepted_equal_support_units']}",
        f"eligible_track_count={summary['eligible_track_count']}",
        "model_loaded_or_run=0",
        "pose_modified=0",
        "history_candidate_writeback=0",
        "pointmap_candidate_generated=1",
        "candidate_frozen_before_gt_access=1",
        "candidate_generation_gt_fields=0",
        "target_geometry_fields_accessed_before_freeze=0",
        f"target_geometry_fields_accessed_after_freeze={len(GT_FIELDS)}",
        "gt_role=scoring_only_after_candidate_artifact_is_frozen",
        "selected_pointmap_modified=0",
        "",
        (
            "branch,selected_points,candidate_support_points,"
            "nonzero_modified_points,valid_surfel_rate,global_weighted_rmse,"
            "global_gain_vs_raw_percent,instance_weighted_rmse,"
            "instance_gain_vs_raw_percent,correct_support_weighted_rmse,"
            "correct_support_gain_vs_raw_percent,global_worse_frames,"
            "instance_worse_frames,modified_background_points,"
            "displacement_p95,cap_hit_rate"
        ),
    ]
    for row in summary["branch_results"]:
        lines.append(
            ",".join(
                str(row[name])
                for name in (
                    "branch",
                    "selected_points",
                    "candidate_support_points",
                    "nonzero_modified_points",
                    "valid_surfel_rate",
                    "global_weighted_rmse",
                    "global_gain_vs_raw_percent",
                    "instance_weighted_rmse",
                    "instance_gain_vs_raw_percent",
                    "correct_support_weighted_rmse",
                    "correct_support_gain_vs_raw_percent",
                    "global_worse_frames",
                    "instance_worse_frames",
                    "modified_background_points",
                    "displacement_p95",
                    "cap_hit_rate",
                )
            )
        )
    lines.extend(
        [
            "",
            (
                "slot,prompt,paired_points,raw_weighted_rmse,"
                "candidate_weighted_rmse,gain_vs_raw_percent,track_improved"
            ),
        ]
    )
    for row in summary["track_results"]:
        if row["branch"] != "correct_id":
            continue
        lines.append(
            ",".join(
                str(row[name])
                for name in (
                    "slot",
                    "prompt",
                    "paired_points",
                    "raw_weighted_rmse",
                    "candidate_weighted_rmse",
                    "gain_vs_raw_percent",
                    "track_improved",
                )
            )
        )
    lines.extend(
        [
            "",
            f"decision={json.dumps(summary['decision'], sort_keys=True)}",
            "",
            f"summary={path.with_name('fixed_correspondence_i1_summary.json')}",
            f"candidate={summary['outputs']['candidate']}",
            f"diagnostics_csv={summary['outputs']['diagnostics_csv']}",
            f"branch_csv={summary['outputs']['branch_csv']}",
            f"track_csv={summary['outputs']['track_csv']}",
            "===== COPYABLE_V1_FIXED_CORRESPONDENCE_I1_END =====",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def _validate_source_sweep(experiment: I1Run) -> dict[str, Any]:
    if not experiment.source_sweep_summary.is_file():
        raise FileNotFoundError(
            "Run commands_v1_correspondence_sweep.txt before fixed I1."
        )
    summary = json.loads(
        experiment.source_sweep_summary.read_text(encoding="utf8")
    )
    expected = {
        "schema": 1,
        "revision": SWEEP_REVISION,
        "pointmap_candidate_generated": 0,
        "candidate_generation_gt_fields": 0,
        "target_geometry_fields_read": 0,
    }
    for name, value in expected.items():
        if summary.get(name) != value:
            raise ValueError(
                f"Source sweep {name}={summary.get(name)!r}; expected {value!r}."
            )
    decision = summary.get("decision", {})
    if int(decision.get("correspondence_sweep_pass", 0)) != 1:
        raise ValueError("Source correspondence sweep did not pass.")
    selected = float(decision["selected_radius_scene_fraction"])
    if not math.isclose(
        selected,
        experiment.fixed_radius_scene_fraction,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "Fixed I1 radius differs from the pre-registered selected radius."
        )
    return summary


def _load_i1(path: str | Path) -> I1Run:
    source = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf8")) or {}
    correspondence_config = Path(raw["correspondence_config"]).expanduser().resolve()
    sweep = _load_sweep(correspondence_config)
    experiment = I1Run(
        source_path=source,
        sweep=sweep,
        source_sweep_summary=Path(raw["source_sweep_summary"])
        .expanduser()
        .resolve(),
        output_dir=Path(raw["output_dir"]).expanduser().resolve(),
        fixed_radius_scene_fraction=float(raw["fixed_radius_scene_fraction"]),
        minimum_improved_tracks=int(
            raw.get("acceptance", {}).get("minimum_improved_tracks", 2)
        ),
    )
    if experiment.fixed_radius_scene_fraction <= 0.0:
        raise ValueError("Fixed match radius must be positive.")
    if experiment.minimum_improved_tracks < 1:
        raise ValueError("minimum_improved_tracks must be positive.")
    if experiment.output_dir == sweep.base.output_dir:
        raise ValueError("Fixed I1 output must not overwrite the source sweep.")
    return experiment


def _worse_frames(before: Sequence[float], after: Sequence[float]) -> int:
    return sum(
        int(math.isfinite(candidate) and candidate > raw)
        for raw, candidate in zip(before, after)
        if math.isfinite(raw)
    )


def _clear_outputs(output_dir: Path) -> None:
    for name in (
        "candidate_deltas.pt",
        "candidate_diagnostics.csv",
        "branch_summary.csv",
        "track_summary.csv",
        "track_eligibility.csv",
        "fixed_correspondence_i1_summary.json",
        "copyable_result.txt",
    ):
        path = output_dir / name
        if path.is_file():
            path.unlink()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="streaming_couping/configs/v1_fixed_correspondence_i1.yaml",
    )
    parser.add_argument("--device")
    return parser.parse_args()


if __name__ == "__main__":
    main()
