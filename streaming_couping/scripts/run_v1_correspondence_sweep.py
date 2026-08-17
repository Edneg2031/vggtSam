#!/usr/bin/env python3
"""Sweep causal SAM-indexed surfel correspondence radius without GT or I1."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Sequence

import torch
import yaml

from streaming_couping.src.instance_geometry import (
    SurfelQueryResult,
    bounded_surfel_query,
    erode_instance_masks,
    select_mask_points,
    shift_instance_masks,
)
from streaming_couping.src.learned_pose.baseline_runtime import (
    load_baseline_run_config,
)
from streaming_couping.src.learned_pose.cache import cache_path, load_feature_cache
from streaming_couping.src.learned_pose.config import ClipConfig, load_learned_pose_config
from streaming_couping.src.semantic_map import normalize_confidence, semantic_slot_map
from streaming_couping.scripts.run_v1_instance_geometry import (
    BASELINE_REVISION,
    I0_BRANCHES,
    V1Run,
    _active_instance_pairs,
    _collect_history,
    _discovered_slots,
    _find_clip,
    _finite_quantile,
    _load_run,
    _normalize_prompt,
    _ratio,
    _scene_diagonal,
    _weighted_row_mean,
    geometry_excluded_slots,
)


REVISION = "v1_causal_correspondence_radius_sweep_r1"
RAW_INPUT_FIELDS = (
    "clip_name",
    "frame_indices",
    "baseline_world_points",
    "baseline_world_confidence",
    "tracking_masks_stream",
    "tracking_scores",
    "sam_track_ids",
    "sam_track_prompts",
    "sam_birth_indices",
    "instance_prompts",
)


@dataclass(frozen=True)
class SweepRun:
    base: V1Run
    radius_fractions: tuple[float, ...]
    minimum_correct_active_tracks: int
    minimum_correct_valid_surfel_rate: float
    minimum_valid_rate_ratio_vs_max_control: float


@dataclass(frozen=True)
class EqualSupportUnit:
    sequence_index: int
    frame_index: int
    slot: int
    shuffled_history_slot: int
    prompt: str
    correct_current: torch.Tensor
    shifted_current: torch.Tensor
    histories: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]

    @property
    def current_points(self) -> int:
        return int(self.correct_current.shape[0])

    @property
    def history_points(self) -> int:
        return int(self.histories["correct_id"][0].shape[0])


def main() -> None:
    args = _parse_args()
    sweep = _load_sweep(args.config)
    run = sweep.base
    if args.device:
        run = V1Run(**{**run.__dict__, "device": str(args.device)})
        sweep = SweepRun(**{**sweep.__dict__, "base": run})

    data = load_learned_pose_config(run.baseline_config)
    baseline = load_baseline_run_config(run.baseline_config)
    clip = _find_clip(data.clips, baseline.clip_name)
    cache_path_value = cache_path(data, clip)
    # Project onto an explicit allowlist immediately. No downstream helper can
    # see the GT pointmap/alignment entries that coexist in the legacy cache.
    payload = _raw_input_projection(load_feature_cache(cache_path_value))
    baseline_summary = _validate_raw_inputs(
        payload=payload,
        clip=clip,
        baseline_output_dir=baseline.output_dir,
    )

    configured_prompts = {
        _normalize_prompt(value) for value in payload["instance_prompts"]
    }
    missing_exclusions = set(run.geometry_excluded_prompts) - configured_prompts
    if missing_exclusions:
        raise ValueError(
            "Geometry exclusions are absent from V0 prompts: "
            f"{sorted(missing_exclusions)}"
        )

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
    active_pairs, track_rows = _active_instance_pairs(
        payload=payload,
        scores=scores,
        interior_masks=interior_masks,
        valid_points=valid_points,
        discovered_slots=discovered_slots,
        run=run,
    )
    eligible_slots = tuple(
        int(row["slot"]) for row in track_rows if int(row["eligible_track"])
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

    audit_rows: list[dict[str, object]] = []
    radius_rows: list[dict[str, object]] = []
    gate_rows: list[dict[str, object]] = []
    for fraction in sweep.radius_fractions:
        radius_raw = scene_diagonal * fraction
        rows = _run_radius(
            units=units,
            run=run,
            radius_fraction=fraction,
            radius_raw=radius_raw,
            max_displacement=scene_diagonal
            * run.max_displacement_scene_fraction,
        )
        audit_rows.extend(rows)
        summaries = _summarize_radius(rows, fraction, radius_raw)
        radius_rows.extend(summaries)
        gate_rows.append(_evaluate_radius(summaries, sweep))

    decision = _select_radius(gate_rows)
    result = _write_outputs(
        sweep=sweep,
        payload=payload,
        baseline_summary=baseline_summary,
        cache_path_value=cache_path_value,
        scene_diagonal=scene_diagonal,
        shift_y=shift_y,
        shift_x=shift_x,
        track_rows=track_rows,
        unit_count=len(units),
        skipped_units=skipped_units,
        audit_rows=audit_rows,
        radius_rows=radius_rows,
        gate_rows=gate_rows,
        decision=decision,
    )
    print(f"V1 correspondence sweep result={result}")


def _build_equal_support_units(
    *,
    points: torch.Tensor,
    confidence: torch.Tensor,
    valid_points: torch.Tensor,
    interior_masks: torch.Tensor,
    shifted_masks: torch.Tensor,
    active_pairs: dict[int, tuple[int, ...]],
    eligible_slots: tuple[int, ...],
    prompts: tuple[str, ...],
    frame_indices: tuple[int, ...],
    run: V1Run,
) -> tuple[list[EqualSupportUnit], dict[str, int]]:
    skipped = {
        "fewer_than_two_eligible_tracks": 0,
        "insufficient_equal_current_points": 0,
        "insufficient_equal_history_points": 0,
        "insufficient_history_frames": 0,
    }
    if len(eligible_slots) < 2:
        skipped["fewer_than_two_eligible_tracks"] = sum(
            len(slots) for slots in active_pairs.values()
        )
        return [], skipped

    shuffled = {
        slot: eligible_slots[(index + 1) % len(eligible_slots)]
        for index, slot in enumerate(eligible_slots)
    }
    units: list[EqualSupportUnit] = []
    for sequence_index, active_slots in active_pairs.items():
        for slot in active_slots:
            if slot not in shuffled:
                continue
            correct_current, _, _ = select_mask_points(
                points[sequence_index],
                confidence[sequence_index],
                interior_masks[sequence_index, slot] & valid_points[sequence_index],
                limit=run.max_current_points_per_instance,
            )
            shifted_current, _, _ = select_mask_points(
                points[sequence_index],
                confidence[sequence_index],
                shifted_masks[sequence_index, slot] & valid_points[sequence_index],
                limit=run.max_current_points_per_instance,
            )
            equal_current_count = min(
                int(correct_current.shape[0]), int(shifted_current.shape[0])
            )
            if equal_current_count < run.min_current_points:
                skipped["insufficient_equal_current_points"] += 1
                continue
            correct_current = _even_subsample(correct_current, equal_current_count)
            shifted_current = _even_subsample(shifted_current, equal_current_count)

            history_specs = {
                "correct_id": interior_masks[:, slot],
                "shuffled_id": interior_masks[:, shuffled[slot]],
                "shifted_mask": shifted_masks[:, slot],
            }
            histories = {
                branch: _collect_history(
                    points=points,
                    confidence=confidence,
                    valid_points=valid_points,
                    masks=branch_masks,
                    before=sequence_index,
                    per_frame_limit=run.max_history_points_per_frame,
                    total_limit=run.max_history_points_total,
                )
                for branch, branch_masks in history_specs.items()
            }
            equal_history_count = min(
                int(value[0].shape[0]) for value in histories.values()
            )
            if equal_history_count < run.neighbors:
                skipped["insufficient_equal_history_points"] += 1
                continue
            histories = {
                branch: _subsample_history(value, equal_history_count)
                for branch, value in histories.items()
            }
            if any(
                int(torch.unique(value[2]).numel()) < run.min_support_frames
                for value in histories.values()
            ):
                skipped["insufficient_history_frames"] += 1
                continue
            units.append(
                EqualSupportUnit(
                    sequence_index=sequence_index,
                    frame_index=frame_indices[sequence_index],
                    slot=slot,
                    shuffled_history_slot=shuffled[slot],
                    prompt=prompts[slot],
                    correct_current=correct_current,
                    shifted_current=shifted_current,
                    histories=histories,
                )
            )
    return units, skipped


def _run_radius(
    *,
    units: Sequence[EqualSupportUnit],
    run: V1Run,
    radius_fraction: float,
    radius_raw: float,
    max_displacement: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for unit in units:
        for branch in I0_BRANCHES:
            current = (
                unit.shifted_current
                if branch == "shifted_mask"
                else unit.correct_current
            )
            history, weights, frames = unit.histories[branch]
            query = bounded_surfel_query(
                current_points=current,
                history_points=history,
                history_weights=weights,
                history_frame_ids=frames,
                device=run.device,
                neighbors=run.neighbors,
                min_support_frames=run.min_support_frames,
                match_radius=radius_raw,
                normal_variance_max=run.normal_variance_max,
                alpha=run.alpha,
                max_displacement=max_displacement,
                chunk_size=run.chunk_size,
            )
            rows.append(
                _query_row(
                    query=query,
                    branch=branch,
                    unit=unit,
                    radius_fraction=radius_fraction,
                    radius_raw=radius_raw,
                )
            )
    return rows


def _query_row(
    *,
    query: SurfelQueryResult,
    branch: str,
    unit: EqualSupportUnit,
    radius_fraction: float,
    radius_raw: float,
) -> dict[str, object]:
    valid_count = int(query.valid.sum())
    history_frames = unit.histories[branch][2]
    return {
        "radius_scene_fraction": radius_fraction,
        "radius_raw": radius_raw,
        "branch": branch,
        "sequence_index": unit.sequence_index,
        "frame_index": unit.frame_index,
        "slot": unit.slot,
        "history_slot": (
            unit.shuffled_history_slot if branch == "shuffled_id" else unit.slot
        ),
        "prompt": unit.prompt,
        "current_points": unit.current_points,
        "history_points": unit.history_points,
        "history_frames": int(torch.unique(history_frames).numel()),
        "valid_surfel_points": valid_count,
        "valid_surfel_rate": _ratio(valid_count, unit.current_points),
        "nearest_median": _finite_quantile(query.nearest_distance, 0.50),
        "nearest_p90": _finite_quantile(query.nearest_distance, 0.90),
        "normal_residual_median": _finite_quantile(query.normal_residual, 0.50),
        "surface_thickness_median": _finite_quantile(
            query.surface_thickness, 0.50
        ),
        "support_frames_median": _finite_quantile(
            query.support_frames.float(), 0.50
        ),
    }


def _summarize_radius(
    rows: list[dict[str, object]],
    radius_fraction: float,
    radius_raw: float,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for branch in I0_BRANCHES:
        selected = [row for row in rows if row["branch"] == branch]
        selected_points = sum(int(row["current_points"]) for row in selected)
        valid_points = sum(int(row["valid_surfel_points"]) for row in selected)
        output.append(
            {
                "radius_scene_fraction": radius_fraction,
                "radius_raw": radius_raw,
                "branch": branch,
                "selected_track_frames": len(selected),
                "active_tracks": len({int(row["slot"]) for row in selected}),
                "selected_points": selected_points,
                "history_points": sum(int(row["history_points"]) for row in selected),
                "valid_surfel_points": valid_points,
                "valid_surfel_rate": _ratio(valid_points, selected_points),
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


def _evaluate_radius(
    summaries: list[dict[str, object]], sweep: SweepRun
) -> dict[str, object]:
    by_branch = {str(row["branch"]): row for row in summaries}
    correct = by_branch["correct_id"]
    controls = [by_branch["shuffled_id"], by_branch["shifted_mask"]]
    finite = all(
        math.isfinite(float(row[name]))
        for row in summaries
        for name in (
            "valid_surfel_rate",
            "mean_nearest_median",
            "mean_normal_residual_median",
        )
    )
    equal_support = (
        len({int(row["selected_track_frames"]) for row in summaries}) == 1
        and len({int(row["selected_points"]) for row in summaries}) == 1
        and len({int(row["history_points"]) for row in summaries}) == 1
    )
    active_pass = (
        int(correct["active_tracks"]) >= sweep.minimum_correct_active_tracks
    )
    valid_rate_pass = (
        float(correct["valid_surfel_rate"])
        >= sweep.minimum_correct_valid_surfel_rate
    )
    maximum_control_rate = max(float(row["valid_surfel_rate"]) for row in controls)
    control_ratio_pass = (
        float(correct["valid_surfel_rate"])
        >= sweep.minimum_valid_rate_ratio_vs_max_control * maximum_control_rate
    )
    nearest_order_pass = all(
        float(correct["mean_nearest_median"]) < float(row["mean_nearest_median"])
        for row in controls
    )
    normal_order_pass = all(
        float(correct["mean_normal_residual_median"])
        < float(row["mean_normal_residual_median"])
        for row in controls
    )
    passed = int(
        finite
        and equal_support
        and active_pass
        and valid_rate_pass
        and control_ratio_pass
        and nearest_order_pass
        and normal_order_pass
    )
    return {
        "radius_scene_fraction": float(correct["radius_scene_fraction"]),
        "radius_raw": float(correct["radius_raw"]),
        "finite_equal_support_pass": int(finite and equal_support),
        "correct_active_tracks": int(correct["active_tracks"]),
        "correct_active_tracks_pass": int(active_pass),
        "correct_valid_surfel_rate": float(correct["valid_surfel_rate"]),
        "maximum_control_valid_surfel_rate": maximum_control_rate,
        "correct_minimum_valid_rate_pass": int(valid_rate_pass),
        "correct_control_valid_rate_ratio_pass": int(control_ratio_pass),
        "correct_nearest_order_pass": int(nearest_order_pass),
        "correct_normal_order_pass": int(normal_order_pass),
        "radius_gate_pass": passed,
    }


def _select_radius(gate_rows: Sequence[dict[str, object]]) -> dict[str, object]:
    passing = [row for row in gate_rows if int(row["radius_gate_pass"])]
    selected = min(
        passing,
        key=lambda row: float(row["radius_scene_fraction"]),
        default=None,
    )
    return {
        "correspondence_sweep_pass": int(selected is not None),
        "selected_radius_scene_fraction": (
            float(selected["radius_scene_fraction"]) if selected else None
        ),
        "selected_radius_raw": float(selected["radius_raw"]) if selected else None,
        "selection_rule": "smallest_fixed_radius_passing_all_non_gt_gates",
        "selected_pointmap_modified": 0,
        "candidate_generation_gt_fields": 0,
        "target_geometry_fields_read": 0,
        "i1_run": 0,
        "claim": (
            "sam_identity_correspondence_radius_supported_not_pointmap_improvement"
            if selected
            else "no_correspondence_radius_passed_fixed_non_gt_gate"
        ),
        "next_gate": (
            "run_separate_fixed_radius_i1_candidate_audit"
            if selected
            else "keep_v0_raw_pointmap_and_stop_instance_geometry_adjustment"
        ),
    }


def _write_outputs(
    *,
    sweep: SweepRun,
    payload: dict[str, Any],
    baseline_summary: dict[str, Any],
    cache_path_value: Path,
    scene_diagonal: float,
    shift_y: int,
    shift_x: int,
    track_rows: list[dict[str, object]],
    unit_count: int,
    skipped_units: dict[str, int],
    audit_rows: list[dict[str, object]],
    radius_rows: list[dict[str, object]],
    gate_rows: list[dict[str, object]],
    decision: dict[str, object],
) -> Path:
    run = sweep.base
    run.output_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "track_eligibility.csv",
        "correspondence_diagnostics.csv",
        "radius_branch_summary.csv",
        "radius_gate_summary.csv",
        "correspondence_sweep_summary.json",
        "copyable_result.txt",
    ):
        path = run.output_dir / name
        if path.is_file():
            path.unlink()
    _write_csv(run.output_dir / "track_eligibility.csv", track_rows)
    _write_csv(run.output_dir / "correspondence_diagnostics.csv", audit_rows)
    _write_csv(run.output_dir / "radius_branch_summary.csv", radius_rows)
    _write_csv(run.output_dir / "radius_gate_summary.csv", gate_rows)

    output = {
        "schema": 1,
        "revision": REVISION,
        "clip": payload["clip_name"],
        "config": str(run.source_path),
        "baseline_config": str(run.baseline_config),
        "baseline_revision": BASELINE_REVISION,
        "baseline_pose_branch": baseline_summary["selected_pose_branch"],
        "baseline_pointmap": "raw_full_history_world_pointmap",
        "cache": str(cache_path_value),
        "frames": tuple(int(value) for value in payload["frame_indices"]),
        "prompts": tuple(str(value) for value in payload["instance_prompts"]),
        "branches": I0_BRANCHES,
        "radius_scene_fractions": sweep.radius_fractions,
        "scene_diagonal_raw": scene_diagonal,
        "shifted_mask_pixels": (shift_y, shift_x),
        "model_loaded_or_run": 0,
        "model_trained": 0,
        "sam_rerun": 0,
        "streamvggt_rerun": 0,
        "pose_modified": 0,
        "selected_pointmap_modified": 0,
        "pointmap_candidate_generated": 0,
        "candidate_generation_gt_fields": 0,
        "target_geometry_fields_read": 0,
        "gt_role": "not_read_or_scored",
        "history_candidate_writeback": 0,
        "prompt_selection_annotation_gt_used": int(
            baseline_summary["prompt_selection_annotation_gt_used"]
        ),
        "runtime_prompt_selection_gt_fields": int(
            baseline_summary["runtime_prompt_selection_gt_fields"]
        ),
        "eligibility": {
            "max_active_instances": run.max_active_instances,
            "geometry_excluded_prompts": run.geometry_excluded_prompts,
            "min_history_visible_frames": run.min_history_visible_frames,
            "min_track_score": run.min_track_score,
            "min_mask_area_ratio": run.min_mask_area_ratio,
            "max_mask_area_ratio": run.max_mask_area_ratio,
            "min_current_points": run.min_current_points,
            "point_confidence_threshold": run.point_confidence_threshold,
            "erosion_radius": run.erosion_radius,
        },
        "fixed_gates": {
            "minimum_correct_active_tracks": sweep.minimum_correct_active_tracks,
            "minimum_correct_valid_surfel_rate": (
                sweep.minimum_correct_valid_surfel_rate
            ),
            "minimum_valid_rate_ratio_vs_max_control": (
                sweep.minimum_valid_rate_ratio_vs_max_control
            ),
            "correct_nearest_lower_than_both_controls": 1,
            "correct_normal_residual_lower_than_both_controls": 1,
            "finite_equal_support_required": 1,
        },
        "equal_support": {
            "protocol": (
                "per_frame_instance_equal_current_and_history_point_counts_"
                "across_correct_shuffled_shifted"
            ),
            "accepted_units": unit_count,
            "skipped_units": skipped_units,
        },
        "track_eligibility": track_rows,
        "radius_branches": radius_rows,
        "radius_gates": gate_rows,
        "decision": decision,
        "outputs": {
            "track_csv": str(run.output_dir / "track_eligibility.csv"),
            "diagnostics_csv": str(
                run.output_dir / "correspondence_diagnostics.csv"
            ),
            "radius_csv": str(run.output_dir / "radius_branch_summary.csv"),
            "gate_csv": str(run.output_dir / "radius_gate_summary.csv"),
            "copyable_report": str(run.output_dir / "copyable_result.txt"),
        },
    }
    result = run.output_dir / "correspondence_sweep_summary.json"
    result.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf8"
    )
    _write_copyable(run.output_dir / "copyable_result.txt", output)
    print("V1 CAUSAL SAM-INDEXED CORRESPONDENCE RADIUS SWEEP")
    for row in gate_rows:
        print(
            f"  radius={float(row['radius_scene_fraction']):.3f} "
            f"valid={float(row['correct_valid_surfel_rate']):.4f} "
            f"control_max={float(row['maximum_control_valid_surfel_rate']):.4f} "
            f"pass={int(row['radius_gate_pass'])}"
        )
    print(f"  decision={json.dumps(decision, sort_keys=True)}")
    print(f"  copyable_report={run.output_dir / 'copyable_result.txt'}")
    return result


def _write_copyable(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "===== COPYABLE_V1_CORRESPONDENCE_SWEEP_BEGIN =====",
        f"revision={summary['revision']}",
        f"clip={summary['clip']}",
        f"frames={len(summary['frames'])}",
        f"branches={','.join(summary['branches'])}",
        "radius_scene_fractions="
        + " ".join(str(value) for value in summary["radius_scene_fractions"]),
        f"min_mask_area_ratio={summary['eligibility']['min_mask_area_ratio']}",
        f"accepted_equal_support_units={summary['equal_support']['accepted_units']}",
        "model_loaded_or_run=0",
        "pointmap_candidate_generated=0",
        "selected_pointmap_modified=0",
        "candidate_generation_gt_fields=0",
        "target_geometry_fields_read=0",
        "gt_role=not_read_or_scored",
        "i1_run=0",
        "",
        (
            "radius_scene_fraction,radius_raw,branch,selected_track_frames,"
            "active_tracks,selected_points,history_points,valid_surfel_points,"
            "valid_surfel_rate,mean_nearest_median,"
            "mean_normal_residual_median,mean_surface_thickness_median"
        ),
    ]
    for row in summary["radius_branches"]:
        lines.append(
            ",".join(
                str(row[name])
                for name in (
                    "radius_scene_fraction",
                    "radius_raw",
                    "branch",
                    "selected_track_frames",
                    "active_tracks",
                    "selected_points",
                    "history_points",
                    "valid_surfel_points",
                    "valid_surfel_rate",
                    "mean_nearest_median",
                    "mean_normal_residual_median",
                    "mean_surface_thickness_median",
                )
            )
        )
    lines.extend(
        [
            "",
            (
                "radius_scene_fraction,finite_equal_support_pass,"
                "correct_active_tracks,correct_valid_surfel_rate,"
                "maximum_control_valid_surfel_rate,"
                "correct_minimum_valid_rate_pass,"
                "correct_control_valid_rate_ratio_pass,"
                "correct_nearest_order_pass,correct_normal_order_pass,"
                "radius_gate_pass"
            ),
        ]
    )
    for row in summary["radius_gates"]:
        lines.append(
            ",".join(
                str(row[name])
                for name in (
                    "radius_scene_fraction",
                    "finite_equal_support_pass",
                    "correct_active_tracks",
                    "correct_valid_surfel_rate",
                    "maximum_control_valid_surfel_rate",
                    "correct_minimum_valid_rate_pass",
                    "correct_control_valid_rate_ratio_pass",
                    "correct_nearest_order_pass",
                    "correct_normal_order_pass",
                    "radius_gate_pass",
                )
            )
        )
    lines.extend(
        [
            "",
            f"decision={json.dumps(summary['decision'], sort_keys=True)}",
            "",
            f"summary={path.with_name('correspondence_sweep_summary.json')}",
            f"track_csv={summary['outputs']['track_csv']}",
            f"diagnostics_csv={summary['outputs']['diagnostics_csv']}",
            f"radius_csv={summary['outputs']['radius_csv']}",
            f"gate_csv={summary['outputs']['gate_csv']}",
            "===== COPYABLE_V1_CORRESPONDENCE_SWEEP_END =====",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def _validate_raw_inputs(
    *,
    payload: dict[str, Any],
    clip: ClipConfig,
    baseline_output_dir: Path,
) -> dict[str, Any]:
    missing = [name for name in RAW_INPUT_FIELDS if name not in payload]
    if missing:
        raise ValueError(f"V0 cache lacks correspondence fields={missing}.")
    if tuple(str(value) for value in payload["instance_prompts"]) != tuple(
        clip.instance_prompts
    ):
        raise ValueError("V0 cache prompt signature differs from the config.")
    if "bed" in {_normalize_prompt(value) for value in payload["instance_prompts"]}:
        raise ValueError("Local correspondence audit excludes the dominant bed prompt.")
    if str(payload["clip_name"]) != clip.name:
        raise ValueError("V0 cache clip signature differs from the config.")

    points = payload["baseline_world_points"]
    confidence = payload["baseline_world_confidence"]
    masks = payload["tracking_masks_stream"]
    scores = payload["tracking_scores"]
    if points.ndim != 4 or points.shape[-1] != 3:
        raise ValueError("baseline_world_points must have shape [S,H,W,3].")
    sequence, height, width, _ = points.shape
    if tuple(confidence.shape) != (sequence, height, width):
        raise ValueError("baseline_world_confidence is not aligned with points.")
    if masks.ndim != 4 or tuple(masks.shape[:1] + masks.shape[2:]) != (
        sequence,
        height,
        width,
    ):
        raise ValueError("tracking_masks_stream is not aligned with points.")
    instances = int(masks.shape[1])
    if tuple(scores.shape) != (sequence, instances):
        raise ValueError("tracking_scores is not aligned with masks.")
    for name in ("sam_track_ids", "sam_track_prompts", "sam_birth_indices"):
        if len(payload[name]) != instances:
            raise ValueError(f"{name} is not aligned with mask slots.")
    if len(payload["frame_indices"]) != sequence:
        raise ValueError("frame_indices is not aligned with points.")

    summary_path = baseline_output_dir / "baseline_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError("Run commands_v0_baseline.txt before the sweep.")
    summary = json.loads(summary_path.read_text(encoding="utf8"))
    expected = {
        "schema": 6,
        "implementation_revision": BASELINE_REVISION,
        "selected_pose_branch": "retrieve_qk",
        "formal_pointmap_output": "raw_full_history_world_pointmap",
        "tracking_baseline_acceptance_pass": 1,
        "prompt_selection_annotation_gt_used": 1,
        "runtime_prompt_selection_gt_fields": 0,
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


def _raw_input_projection(payload: dict[str, Any]) -> dict[str, Any]:
    missing = [name for name in RAW_INPUT_FIELDS if name not in payload]
    if missing:
        raise ValueError(f"V0 cache lacks correspondence fields={missing}.")
    return {name: payload[name] for name in RAW_INPUT_FIELDS}


def _load_sweep(path: str | Path) -> SweepRun:
    source = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf8")) or {}
    base = _load_run(source)
    values = raw.get("sweep", {})
    radius_fractions = tuple(
        sorted({float(value) for value in values["match_radius_scene_fractions"]})
    )
    sweep = SweepRun(
        base=base,
        radius_fractions=radius_fractions,
        minimum_correct_active_tracks=int(
            values.get("minimum_correct_active_tracks", 2)
        ),
        minimum_correct_valid_surfel_rate=float(
            values.get("minimum_correct_valid_surfel_rate", 0.10)
        ),
        minimum_valid_rate_ratio_vs_max_control=float(
            values.get("minimum_valid_rate_ratio_vs_max_control", 2.0)
        ),
    )
    if not sweep.radius_fractions or any(value <= 0.0 for value in radius_fractions):
        raise ValueError("Sweep radius fractions must be positive.")
    if sweep.minimum_correct_active_tracks < 2:
        raise ValueError("The correspondence gate requires at least two tracks.")
    if not 0.0 < sweep.minimum_correct_valid_surfel_rate <= 1.0:
        raise ValueError("minimum_correct_valid_surfel_rate must be in (0,1].")
    if sweep.minimum_valid_rate_ratio_vs_max_control <= 1.0:
        raise ValueError("Control valid-rate ratio must be greater than one.")
    return sweep


def _even_subsample(value: torch.Tensor, count: int) -> torch.Tensor:
    if int(value.shape[0]) == int(count):
        return value
    indices = torch.linspace(0, value.shape[0] - 1, steps=int(count)).round().long()
    return value.index_select(0, indices)


def _subsample_history(
    value: tuple[torch.Tensor, torch.Tensor, torch.Tensor], count: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    points, weights, frames = value
    if int(points.shape[0]) == int(count):
        return value
    indices = torch.linspace(0, points.shape[0] - 1, steps=int(count)).round().long()
    return (
        points.index_select(0, indices),
        weights.index_select(0, indices),
        frames.index_select(0, indices),
    )


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
        default="streaming_couping/configs/v1_correspondence_sweep.yaml",
    )
    parser.add_argument("--device")
    return parser.parse_args()


if __name__ == "__main__":
    main()
