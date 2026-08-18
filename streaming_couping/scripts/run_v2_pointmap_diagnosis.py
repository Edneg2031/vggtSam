#!/usr/bin/env python3
"""Run D0 -> D1 -> T0 against frozen V0 artifacts without map write-back."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from streaming_couping.src.external_repos import maybe_add_repo_to_path
from streaming_couping.src.learned_pose.cache import cache_path, load_feature_cache
from streaming_couping.src.learned_pose.config import load_learned_pose_config
from streaming_couping.src.pointmap_diagnosis import (
    CoordinateBundle,
    binary_erode,
    common_support,
    confidence_diagnostics,
    d0_branches,
    d0_predicted_k_supplement,
    d0_uncalibrated_depth_supplement,
    distribution_row,
    align_native_points,
    as_homogeneous_world_to_camera,
    intrinsics_from_pose_encoding,
    ownership_map,
    point_errors,
    pointmap_metrics,
    prepare_coordinate_bundle,
    region_masks,
)
from streaming_couping.src.triangulation_probe import (
    PairMatches,
    equalize_query_support,
    match_patch_descriptors,
    project_oracle_correspondence,
    sample_mask_grid,
    triangulate_two_view,
    triangulation_gate,
)


REVISION = "v2_pointmap_diagnosis_d0_d1_t0_r2"


@dataclass(frozen=True)
class V2Run:
    source_path: Path
    v0_config: Path
    output_dir: Path
    raw: dict[str, Any]


@dataclass(frozen=True)
class PairUnit:
    current_index: int
    history_index: int
    slot: int
    control_slot: int
    current_queries: torch.Tensor


def main() -> None:
    args = _parse_args()
    run = _load_run(args.config)
    data = load_learned_pose_config(run.v0_config)
    clip = data.clips[0]
    cache_path_value = cache_path(data, clip)
    payload = load_feature_cache(cache_path_value)
    qk_path = _resolve(
        _read_yaml(run.v0_config)["baseline"]["pose"]["qk_pose_output"]
    )
    qk_artifact = torch.load(qk_path, map_location="cpu", weights_only=False)
    _validate_frozen_inputs(payload, qk_artifact, clip.name)
    print("V2 READ-ONLY POINTMAP DIAGNOSIS")
    print("  phase 0: freeze RGB/SAM/native-QK real T0 candidates before GT scoring")
    # Freeze the deployable real correspondence/QK-triangulation branch before
    # target depth, target pose or target world points are read below. Dataset
    # intrinsics are treated as sensor calibration, not target geometry.
    prefrozen_t0 = (
        _freeze_real_t0_candidates(run, payload, qk_artifact)
        if _section(run, "t0").get("enabled", True)
        else None
    )
    if prefrozen_t0 is not None:
        print(
            "  prefrozen T0 "
            f"units={len(prefrozen_t0['units'])} "
            f"matched_units={len(prefrozen_t0['real_candidates'])}"
        )
    bundle = prepare_coordinate_bundle(payload, qk_artifact)
    run.output_dir.mkdir(parents=True, exist_ok=True)

    print("  formal V0 pose/pointmap/semantic outputs are never modified")
    d0 = _run_d0(run, payload, bundle)
    if not d0["sanity_pass"]:
        final = _write_final_summary(
            run,
            payload,
            cache_path_value,
            qk_path,
            d0=d0,
            d1=None,
            t0=None,
            stopped_after="D0",
        )
        print("  D0 coordinate sanity failed; D1/T0 intentionally not run")
        print(f"  result={final}")
        return

    d1 = _run_d1(run, payload, bundle)
    t0 = (
        _run_t0(run, payload, bundle, prefrozen_t0)
        if prefrozen_t0 is not None
        else None
    )
    final = _write_final_summary(
        run,
        payload,
        cache_path_value,
        qk_path,
        d0=d0,
        d1=d1,
        t0=t0,
        stopped_after=None,
    )
    print(f"  D0 sanity_pass={d0['sanity_pass']}")
    print(
        "  D1 boundary_signal="
        f"{d1['boundary_signal']} high_confidence_wrong="
        f"{d1['high_confidence_wrong']['high_confidence_wrong_percent']:.3f}%"
    )
    if t0 is not None:
        primary = t0["branch_lookup"].get("real_2d__qk_pose__correct_id__calibrated_k", {})
        print(
            "  T0 real/QK/correct valid="
            f"{primary.get('valid_anchors', 0)} improved="
            f"{primary.get('improved_anchor_percent', float('nan')):.3f}%"
        )
    print(f"  result={final}")


def _run_d0(run: V2Run, payload: dict, bundle: CoordinateBundle) -> dict[str, Any]:
    config = _section(run, "d0")
    output = run.output_dir / "d0_depth_pose_oracle"
    output.mkdir(parents=True, exist_ok=True)
    primary = d0_branches(bundle)
    support = common_support(primary.values(), bundle.target_world_points_metric)
    branch_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    for role, branches in (
        ("primary_calibrated_k", primary),
        ("report_only_predicted_k", d0_predicted_k_supplement(bundle)),
        (
            "report_only_invalid_pointmap_scale_assumption",
            d0_uncalibrated_depth_supplement(bundle),
        ),
    ):
        for name, points in branches.items():
            metrics, frames = pointmap_metrics(
                points,
                bundle.target_world_points_metric,
                support,
                paired_max_points=int(config["paired_max_points_per_frame"]),
                symmetric_max_points=int(config["symmetric_max_points_per_frame"]),
            )
            branch_rows.append({"branch": name, "table_role": role, **metrics})
            frame_rows.extend(
                {"branch": name, "table_role": role, **row} for row in frames
            )
    sanity_points = primary["gt_depth__gt_pose__calibrated_k"]
    sanity_support = common_support(
        (sanity_points,), bundle.target_world_points_metric
    )
    sanity, _ = pointmap_metrics(
        sanity_points,
        bundle.target_world_points_metric,
        sanity_support,
        paired_max_points=int(config["paired_max_points_per_frame"]),
        symmetric_max_points=int(config["symmetric_max_points_per_frame"]),
    )
    sanity_pass = int(
        sanity["paired_rmse"] <= float(config["sanity_max_paired_rmse_metric"])
        and sanity["paired_p90"] <= float(config["sanity_max_p90_metric"])
        and sanity["valid_ratio"] >= float(config["sanity_min_valid_ratio"])
    )
    summary = {
        "schema": 1,
        "revision": REVISION,
        "stage": "D0_depth_pose_oracle",
        "clip": payload["clip_name"],
        "coordinate_rule": "single_v0_reference_sim3_native_to_metric_gt_world",
        "primary_intrinsics": "processed_dataset_calibration",
        "predicted_intrinsics_role": "report_only",
        "depth_representation": "camera_space_z_depth",
        "raw_depth_scale_semantics": "independent_scale_shift_invariant_head",
        "raw_depth_alignment": "single_reference_frame_robust_affine_frozen_for_sequence",
        "raw_depth_reference_affine": {
            "scale": bundle.depth_reference_affine_scale,
            "shift": bundle.depth_reference_affine_shift,
            "fit_rmse": bundle.depth_reference_affine_fit_rmse,
            "reference_sequence_index": int(payload.get("reference_sequence_index", 0)),
            "reference_frame": int(
                payload["frame_indices"][
                    int(payload.get("reference_sequence_index", 0))
                ]
            ),
            "gt_role": "d0_oracle_gauge_calibration_only",
        },
        "pose_convention": "world_to_camera_opencv",
        "gt_role": "oracle_generation_and_scoring",
        "candidate_generation_gt_fields": "oracle_stage_not_applicable",
        "formal_v0_pose_modified": 0,
        "formal_v0_pointmap_modified": 0,
        "common_supported_points": int(support.sum()),
        "branches": branch_rows,
        "sanity_branch": "gt_depth__gt_pose__calibrated_k",
        "sanity_metrics_on_independent_gt_support": sanity,
        "sanity_thresholds": {
            "paired_rmse_metric": float(config["sanity_max_paired_rmse_metric"]),
            "p90_metric": float(config["sanity_max_p90_metric"]),
            "valid_ratio": float(config["sanity_min_valid_ratio"]),
        },
        "sanity_pass": sanity_pass,
        "next_gate": "D1_then_T0" if sanity_pass else "stop_fix_coordinates_k_or_evaluator",
    }
    _write_csv(output / "summary.csv", branch_rows)
    _write_csv(output / "per_frame.csv", frame_rows)
    (output / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf8"
    )
    _write_json(output / "summary.json", summary)
    return summary


def _run_d1(run: V2Run, payload: dict, bundle: CoordinateBundle) -> dict[str, Any]:
    config = _section(run, "d1")
    output = run.output_dir / "d1_pointmap_error_decomposition"
    output.mkdir(parents=True, exist_ok=True)
    ownership = ownership_map(
        payload["tracking_masks_stream"],
        payload["tracking_scores"],
        score_threshold=float(config["track_score_threshold"]),
    )
    errors, valid = point_errors(bundle)
    region_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    width_lookup: dict[int, dict[str, dict[str, Any]]] = {}
    for width in tuple(int(value) for value in config["boundary_widths"]):
        masks = region_masks(ownership, boundary_width=width)
        width_lookup[width] = {}
        for name, mask in masks.items():
            row = distribution_row(
                errors, valid & mask, boundary_width=width, region=name
            )
            region_rows.append(row)
            width_lookup[width][name] = row
            for frame in range(errors.shape[0]):
                frame_rows.append(
                    distribution_row(
                        errors[frame],
                        valid[frame] & mask[frame],
                        sequence_index=frame,
                        frame_index=int(payload["frame_indices"][frame]),
                        boundary_width=width,
                        region=name,
                    )
                )
    instance_rows = _instance_error_rows(
        payload=payload,
        ownership=ownership,
        errors=errors,
        valid=valid,
        width=3,
        score_threshold=float(config["track_score_threshold"]),
    )
    confidence_rows, risk_rows, high_wrong = confidence_diagnostics(
        errors, bundle.confidence, valid
    )
    boundary_signal_by_width: dict[str, int] = {}
    boundary_ratios: dict[str, float] = {}
    for width, rows in width_lookup.items():
        interior = float(rows["instance_interior"]["median"])
        boundary = float(rows["instance_inner_boundary"]["median"])
        ratio = boundary / max(interior, 1e-12)
        boundary_ratios[str(width)] = ratio
        boundary_signal_by_width[str(width)] = int(
            ratio >= float(config["boundary_signal_min_median_ratio"])
        )
    boundary_signal = int(
        bool(boundary_signal_by_width)
        and all(boundary_signal_by_width.values())
    )
    summary = {
        "schema": 1,
        "revision": REVISION,
        "stage": "D1_raw_pointmap_error_decomposition",
        "clip": payload["clip_name"],
        "alignment": "same_fixed_raw_reference_sim3_as_d0",
        "region_semantics": (
            "instance_interior,instance_inner_boundary,"
            "instance_outer_boundary,unprompted_complement"
        ),
        "background_claimed": 0,
        "confidence_filter_applied_before_quantiles": 0,
        "gt_role": "per_pixel_error_scoring_only",
        "boundary_median_ratio_by_width": boundary_ratios,
        "boundary_signal_by_width": boundary_signal_by_width,
        "boundary_signal": boundary_signal,
        "regions": region_rows,
        "instances": instance_rows,
        "confidence_bins": confidence_rows,
        "risk_coverage": risk_rows,
        "high_confidence_wrong": high_wrong,
        "formal_v0_pointmap_modified": 0,
        "outputs": {
            "region_summary": str(output / "region_summary.csv"),
            "instance_summary": str(output / "instance_summary.csv"),
            "confidence_summary": str(output / "confidence_summary.csv"),
            "risk_coverage": str(output / "risk_coverage.csv"),
            "per_frame": str(output / "per_frame.csv"),
        },
    }
    _write_csv(output / "region_summary.csv", region_rows)
    _write_csv(output / "instance_summary.csv", instance_rows)
    _write_csv(output / "confidence_summary.csv", confidence_rows)
    _write_csv(output / "risk_coverage.csv", risk_rows)
    _write_csv(output / "per_frame.csv", frame_rows)
    _write_json(output / "summary.json", summary)
    return summary


def _run_t0(
    run: V2Run,
    payload: dict,
    bundle: CoordinateBundle,
    prefrozen: dict[str, Any],
) -> dict[str, Any]:
    config = _section(run, "t0")
    output = run.output_dir / "t0_triangulation_probe"
    output.mkdir(parents=True, exist_ok=True)
    ownership = prefrozen["ownership"]
    units = prefrozen["units"]
    real_candidates = prefrozen["real_candidates"]
    if not torch.allclose(
        prefrozen["calibrated_intrinsics"],
        bundle.calibrated_intrinsics,
        atol=0.0,
        rtol=0.0,
    ):
        raise RuntimeError("Prefrozen T0 calibration changed before scoring.")
    if int(prefrozen["candidate_generation_target_geometry_fields"]) != 0:
        raise RuntimeError("Real T0 candidate accessed target geometry.")
    candidate_frozen_before_gt_access = 1
    trackhead = _trackhead_preflight(run, payload, ownership, units)
    anchor_rows: list[dict[str, Any]] = []
    tensor_rows: list[dict[str, Any]] = []
    for candidate in real_candidates:
        for identity in ("correct_id", "shuffled_id"):
            matches = candidate[identity]
            _score_match_branch(
                anchor_rows,
                tensor_rows,
                source="real_2d",
                identity=identity,
                calibration="calibrated_k",
                unit=candidate["unit"],
                matches=matches,
                bundle=bundle,
                config=config,
                qk_precomputed=candidate[f"{identity}_qk"],
            )
        # Predicted K is report-only and only the deployable correct/QK cell is
        # needed to expose calibration sensitivity.
        _score_match_branch(
            anchor_rows,
            tensor_rows,
            source="real_2d",
            identity="correct_id",
            calibration="predicted_k_report_only",
            unit=candidate["unit"],
            matches=candidate["correct_id"],
            bundle=bundle,
            config=config,
            qk_precomputed=None,
            pose_modes=("qk_pose",),
        )
    oracle_candidates = _generate_oracle_candidates(
        units=units,
        ownership=ownership,
        bundle=bundle,
        config=config,
    )
    for candidate in oracle_candidates:
        _score_match_branch(
            anchor_rows,
            tensor_rows,
            source="oracle_2d",
            identity="correct_id",
            calibration="calibrated_k",
            unit=candidate["unit"],
            matches=candidate["correct_id"],
            bundle=bundle,
            config=config,
        )
    branch_rows = _summarize_anchors(anchor_rows, group_fields=("branch",))
    instance_rows = _summarize_anchors(anchor_rows, group_fields=("branch", "slot"))
    prompts = tuple(str(value) for value in payload.get("sam_track_prompts", ()))
    track_ids = tuple(int(value) for value in payload.get("sam_track_ids", ()))
    for row in instance_rows:
        slot = int(row["slot"])
        row["prompt"] = prompts[slot] if slot < len(prompts) else ""
        row["sam_track_id"] = track_ids[slot] if slot < len(track_ids) else -1
    condition_rows = _condition_summary(anchor_rows)
    lookup = {row["branch"]: row for row in branch_rows}
    correct = lookup.get("real_2d__qk_pose__correct_id__calibrated_k", {})
    shuffled = lookup.get("real_2d__qk_pose__shuffled_id__calibrated_k", {})
    equal_support = int(
        correct.get("candidate_matches", 0) == shuffled.get("candidate_matches", -1)
    )
    correct_valid = int(correct.get("valid_anchors", 0))
    shuffled_valid = int(shuffled.get("valid_anchors", 0))
    identity_validity_advantage = int(
        correct_valid > 0
        and float(correct.get("valid_rate", 0.0))
        > float(shuffled.get("valid_rate", 0.0))
    )
    identity_error_comparison_available = int(
        correct_valid > 0 and shuffled_valid > 0
    )
    identity_error_advantage = int(
        identity_error_comparison_available
        and float(correct.get("triangulation_rmse", float("inf")))
        < float(shuffled.get("triangulation_rmse", float("inf")))
    )
    correct_beats_shuffled = int(
        identity_validity_advantage
        and (shuffled_valid == 0 or identity_error_advantage)
    )
    high_quality_gain = float(correct.get("high_quality_mean_delta", float("nan")))
    enough_high_quality = int(
        int(correct.get("high_quality_anchors", 0))
        >= int(config["min_high_quality_anchors_for_go"])
    )
    go_t1 = int(
        equal_support
        and correct_beats_shuffled
        and enough_high_quality
        and high_quality_gain > 0.0
    )
    summary = {
        "schema": 1,
        "revision": REVISION,
        "stage": "T0_sam_indexed_multiview_triangulation",
        "clip": payload["clip_name"],
        "pair_units": len(units),
        "correspondence_pose_matrix": (
            "oracle_2d_or_real_rgb_patch_x_gt_pose_or_qk_pose"
        ),
        "primary_intrinsics": "processed_dataset_calibration",
        "predicted_intrinsics_role": "report_only",
        "real_matcher": "frozen_rgb_patch_mutual_nearest_ratio",
        "model_trained": 0,
        "candidate_frozen_before_gt_access": candidate_frozen_before_gt_access,
        "candidate_generation_gt_fields": 0,
        "candidate_generation_target_geometry_fields": 0,
        "real_candidate_generation_fields": (
            "stream_images,tracking_masks_stream,tracking_scores,"
            "sam_track_prompts,qk_world_to_camera_native,calibrated_intrinsics"
        ),
        "real_candidate_forbidden_fields": (
            "target_depth,target_world_points,target_world_to_camera"
        ),
        "dataset_calibration_intrinsics_used": 1,
        "calibration_source_field": "target_pose_encoding_fov_only",
        "oracle_gt_generation_explicit": 1,
        "oracle_gt_role": "2d_correspondence_and_gt_pose_upper_bounds",
        "equal_current_query_support_correct_vs_shuffled": equal_support,
        "formal_v0_pose_modified": 0,
        "formal_v0_pointmap_modified": 0,
        "trackhead_preflight": trackhead,
        "branches": branch_rows,
        "branch_lookup": lookup,
        "correct_id_beats_shuffled": correct_beats_shuffled,
        "correct_id_validity_advantage": identity_validity_advantage,
        "correct_id_error_comparison_available": identity_error_comparison_available,
        "correct_id_error_advantage": identity_error_advantage,
        "enough_high_quality_anchors": enough_high_quality,
        "go_t1": go_t1,
        "claim": (
            "independent_sparse_anchor_candidate_supported"
            if go_t1
            else "independent_sparse_anchor_improvement_not_established"
        ),
    }
    _write_csv(output / "anchor_summary.csv", anchor_rows)
    _write_csv(output / "branch_summary.csv", branch_rows)
    _write_csv(output / "instance_summary.csv", instance_rows)
    _write_csv(output / "condition_summary.csv", condition_rows)
    torch.save(
        {
            "schema": 1,
            "revision": REVISION,
            "artifact_role": "diagnostic_sparse_anchors_not_v0_map",
            "anchors": tensor_rows,
        },
        output / "anchors.pt",
    )
    _write_json(output / "summary.json", summary)
    return summary


def _freeze_real_t0_candidates(
    run: V2Run, payload: dict, qk_artifact: dict
) -> dict[str, Any]:
    """Freeze the deployable T0 branch before target geometry is accessed."""

    config = _section(run, "t0")
    ownership = ownership_map(
        payload["tracking_masks_stream"],
        payload["tracking_scores"],
        score_threshold=float(config["track_score_threshold"]),
    )
    qk_native = as_homogeneous_world_to_camera(
        qk_artifact["selected_world_to_camera"]
    )
    calibrated_intrinsics = intrinsics_from_pose_encoding(
        payload["target_pose_encoding"],
        tuple(int(value) for value in payload["image_size"]),
    )
    units = _build_pair_units(payload, qk_native, ownership, config)
    real_candidates = _generate_real_candidates(
        images=payload["stream_images"],
        ownership=ownership,
        units=units,
        calibrated_intrinsics=calibrated_intrinsics,
        qk_world_to_camera_native=qk_native,
        config=config,
    )
    return {
        "ownership": ownership,
        "units": units,
        "real_candidates": real_candidates,
        "calibrated_intrinsics": calibrated_intrinsics,
        "candidate_generation_target_geometry_fields": 0,
    }


def _instance_error_rows(
    *, payload: dict, ownership: torch.Tensor, errors: torch.Tensor,
    valid: torch.Tensor, width: int, score_threshold: float,
) -> list[dict[str, Any]]:
    rows = []
    prompts = tuple(str(value) for value in payload.get("sam_track_prompts", ()))
    track_ids = tuple(int(value) for value in payload.get("sam_track_ids", ()))
    births = tuple(int(value) for value in payload.get("sam_birth_indices", ()))
    scores = payload["tracking_scores"].detach().float().cpu()
    for slot in range(scores.shape[1]):
        owned = ownership == slot
        if not bool(owned.any()):
            continue
        interior = binary_erode(owned, width)
        boundary = owned & ~interior
        whole_row = distribution_row(errors, valid & owned)
        interior_row = distribution_row(errors, valid & interior)
        boundary_row = distribution_row(errors, valid & boundary)
        rows.append(
            {
                "slot": slot,
                "sam_track_id": track_ids[slot] if slot < len(track_ids) else -1,
                "prompt": prompts[slot] if slot < len(prompts) else "",
                "birth_sequence_index": births[slot] if slot < len(births) else -1,
                "visible_frames": int((scores[:, slot] >= score_threshold).sum()),
                "point_count": whole_row["count"],
                "raw_error_mean": whole_row["mean"],
                "raw_error_median": whole_row["median"],
                "raw_error_p90": whole_row["p90"],
                "interior_count": interior_row["count"],
                "interior_error_median": interior_row["median"],
                "boundary_count": boundary_row["count"],
                "boundary_error_median": boundary_row["median"],
                "boundary_over_interior_median_ratio": (
                    float(boundary_row["median"])
                    / max(float(interior_row["median"]), 1e-12)
                ),
            }
        )
    return rows


def _build_pair_units(
    payload: dict,
    qk_world_to_camera_native: torch.Tensor,
    ownership: torch.Tensor,
    config: dict,
) -> list[PairUnit]:
    scores = payload["tracking_scores"].detach().float().cpu()
    threshold = float(config["track_score_threshold"])
    prompts = tuple(str(value).strip().lower() for value in payload.get("sam_track_prompts", ()))
    excluded = {str(value).strip().lower() for value in config.get("excluded_prompts", ())}
    slots = [
        slot for slot in range(scores.shape[1])
        if bool((scores[:, slot] >= threshold).any())
        and (slot >= len(prompts) or prompts[slot] not in excluded)
    ]
    centers = _camera_centers(qk_world_to_camera_native)
    units: list[PairUnit] = []
    for current in range(1, ownership.shape[0]):
        for slot in slots:
            current_mask = ownership[current] == slot
            if int(current_mask.sum()) < int(config["min_mask_pixels"]):
                continue
            history_by_control: list[tuple[int, list[int]]] = []
            for control in slots:
                if control == slot:
                    continue
                histories = [
                    history for history in range(current)
                    if int((ownership[history] == slot).sum()) >= int(config["min_mask_pixels"])
                    and int((ownership[history] == control).sum()) >= int(config["min_mask_pixels"])
                ]
                history_by_control.append((control, histories))
            history_by_control.sort(key=lambda value: (-len(value[1]), value[0]))
            if not history_by_control or not history_by_control[0][1]:
                continue
            control, histories = history_by_control[0]
            histories.sort(
                key=lambda history: float(
                    torch.linalg.vector_norm(centers[current] - centers[history])
                ),
                reverse=True,
            )
            queries = sample_mask_grid(
                binary_erode(current_mask[None], int(config["patch_radius"]))[0],
                stride=int(config["query_stride"]),
                max_points=int(config["max_queries_per_pair"]),
                margin=int(config["patch_radius"]),
            )
            if queries.numel() == 0:
                continue
            for history in histories[: int(config["history_pairs_per_instance_frame"])]:
                units.append(PairUnit(current, history, slot, control, queries))
    return units


def _generate_real_candidates(
    *, images: torch.Tensor, ownership: torch.Tensor, units: list[PairUnit],
    calibrated_intrinsics: torch.Tensor,
    qk_world_to_camera_native: torch.Tensor,
    config: dict,
) -> list[dict[str, Any]]:
    output = []
    for unit in units:
        kwargs = {
            "candidate_stride": int(config["history_candidate_stride"]),
            "max_history_points": int(config["max_history_candidates"]),
            "patch_radius": int(config["patch_radius"]),
            "ratio_threshold": float(config["descriptor_ratio_threshold"]),
        }
        correct = match_patch_descriptors(
            images[unit.current_index], images[unit.history_index],
            unit.current_queries, ownership[unit.history_index] == unit.slot, **kwargs,
        )
        shuffled = match_patch_descriptors(
            images[unit.current_index], images[unit.history_index],
            unit.current_queries, ownership[unit.history_index] == unit.control_slot, **kwargs,
        )
        correct, shuffled = equalize_query_support(correct, shuffled)
        if correct.current_xy.numel() == 0:
            continue
        row: dict[str, Any] = {"unit": unit, "correct_id": correct, "shuffled_id": shuffled}
        for identity, matches in (("correct_id", correct), ("shuffled_id", shuffled)):
            row[f"{identity}_qk"] = triangulate_two_view(
                matches.current_xy, matches.history_xy,
                calibrated_intrinsics[unit.current_index],
                calibrated_intrinsics[unit.history_index],
                qk_world_to_camera_native[unit.current_index],
                qk_world_to_camera_native[unit.history_index],
            )
        output.append(row)
    return output


def _generate_oracle_candidates(
    *, units: list[PairUnit], ownership: torch.Tensor,
    bundle: CoordinateBundle, config: dict,
) -> list[dict[str, Any]]:
    output = []
    for unit in units:
        common = {
            "current_xy": unit.current_queries,
            "current_target_world": bundle.target_world_points_metric[unit.current_index],
            "history_target_world": bundle.target_world_points_metric[unit.history_index],
            "history_intrinsics": bundle.calibrated_intrinsics[unit.history_index],
            "history_world_to_camera": bundle.target_world_to_camera_metric[unit.history_index],
            "current_intrinsics": bundle.calibrated_intrinsics[unit.current_index],
            "current_world_to_camera": bundle.target_world_to_camera_metric[unit.current_index],
            "depth_tolerance": float(config["oracle_depth_tolerance_metric"]),
        }
        correct = project_oracle_correspondence(
            history_mask=ownership[unit.history_index] == unit.slot, **common
        )
        if correct.current_xy.numel() > 0:
            output.append({"unit": unit, "correct_id": correct})
    return output


def _score_match_branch(
    anchor_rows: list[dict[str, Any]], tensor_rows: list[dict[str, Any]], *,
    source: str, identity: str, calibration: str, unit: PairUnit,
    matches: PairMatches, bundle: CoordinateBundle, config: dict,
    qk_precomputed=None, pose_modes: tuple[str, ...] = ("gt_pose", "qk_pose"),
) -> None:
    if matches.current_xy.numel() == 0:
        return
    intrinsics = (
        bundle.predicted_intrinsics
        if calibration == "predicted_k_report_only"
        else bundle.calibrated_intrinsics
    )
    for pose_mode in pose_modes:
        poses = (
            bundle.target_world_to_camera_metric
            if pose_mode == "gt_pose"
            else bundle.qk_world_to_camera_native
        )
        result = (
            qk_precomputed
            if pose_mode == "qk_pose" and qk_precomputed is not None
            else triangulate_two_view(
                matches.current_xy, matches.history_xy,
                intrinsics[unit.current_index], intrinsics[unit.history_index],
                poses[unit.current_index], poses[unit.history_index],
            )
        )
        evaluated_points = (
            result.points
            if pose_mode == "gt_pose"
            else align_native_points(
                result.points,
                scale=bundle.native_to_metric_scale,
                rotation=bundle.native_to_metric_rotation,
                translation=bundle.native_to_metric_translation,
            )
        )
        valid = triangulation_gate(
            result,
            min_ray_angle_degrees=float(config["min_ray_angle_degrees"]),
            max_ray_angle_degrees=float(config["max_ray_angle_degrees"]),
            max_condition=float(config["max_condition"]),
            max_reprojection_px=float(config["max_reprojection_px"]),
        )
        high_quality = valid & (
            (result.ray_angle_degrees >= float(config["high_quality_min_ray_angle_degrees"]))
            & (result.condition <= float(config["high_quality_max_condition"]))
            & (result.reprojection_max_px <= float(config["high_quality_max_reprojection_px"]))
        )
        evaluation_xy = (
            matches.current_xy
            if matches.evaluation_xy is None
            else matches.evaluation_xy
        )
        integer = evaluation_xy.round().long()
        target = bundle.target_world_points_metric[
            unit.current_index, integer[:, 1], integer[:, 0]
        ]
        raw = bundle.raw_world_points_metric[
            unit.current_index, integer[:, 1], integer[:, 0]
        ]
        finite_eval = torch.isfinite(target).all(dim=-1) & torch.isfinite(raw).all(dim=-1)
        valid &= finite_eval
        high_quality &= finite_eval
        tri_error = torch.linalg.vector_norm(evaluated_points - target, dim=-1)
        raw_error = torch.linalg.vector_norm(raw - target, dim=-1)
        delta = raw_error - tri_error
        centers = _camera_centers(poses)
        camera_baseline = float(
            torch.linalg.vector_norm(
                centers[unit.current_index] - centers[unit.history_index]
            )
        )
        if pose_mode == "qk_pose":
            camera_baseline *= bundle.native_to_metric_scale
        branch = f"{source}__{pose_mode}__{identity}__{calibration}"
        for index in range(matches.current_xy.shape[0]):
            anchor_rows.append(
                {
                    "branch": branch,
                    "source": source,
                    "pose": pose_mode,
                    "identity": identity,
                    "calibration": calibration,
                    "current_sequence_index": unit.current_index,
                    "history_sequence_index": unit.history_index,
                    "slot": unit.slot,
                    "control_slot": unit.control_slot,
                    "num_views": 2,
                    "history_temporal_distance": unit.current_index - unit.history_index,
                    "camera_baseline_metric": camera_baseline,
                    "same_persistent_slot": int(identity == "correct_id"),
                    "query_index": int(matches.query_indices[index]),
                    "match_confidence": float(matches.confidence[index]),
                    "raw_point_confidence": float(
                        bundle.confidence[
                            unit.current_index, integer[index, 1], integer[index, 0]
                        ]
                    ),
                    "ray_angle_degrees": float(result.ray_angle_degrees[index]),
                    "condition": float(result.condition[index]),
                    "reprojection_mean_px": float(result.reprojection_mean_px[index]),
                    "reprojection_max_px": float(result.reprojection_max_px[index]),
                    "valid": int(valid[index]),
                    "high_quality": int(high_quality[index]),
                    "triangulation_error": float(tri_error[index]),
                    "raw_error": float(raw_error[index]),
                    "delta_raw_minus_tri": float(delta[index]),
                    "improved": int(valid[index] and delta[index] > 0),
                }
            )
        tensor_rows.append(
            {
                "branch": branch,
                "current_sequence_index": unit.current_index,
                "history_sequence_index": unit.history_index,
                "slot": unit.slot,
                "current_xy": matches.current_xy,
                "evaluation_xy": evaluation_xy,
                "history_xy": matches.history_xy,
                "triangulated_points_metric": evaluated_points,
                "triangulated_points_native": (
                    result.points if pose_mode == "qk_pose" else None
                ),
                "valid": valid,
                "high_quality": high_quality,
            }
        )


def _summarize_anchors(
    rows: list[dict[str, Any]], *, group_fields: tuple[str, ...]
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[field] for field in group_fields), []).append(row)
    output = []
    for key, group in sorted(groups.items(), key=lambda item: str(item[0])):
        valid = [row for row in group if row["valid"]]
        high = [row for row in group if row["high_quality"]]
        summary = {field: value for field, value in zip(group_fields, key)}
        summary.update(
            {
                "candidate_matches": len(group),
                "valid_anchors": len(valid),
                "valid_rate": len(valid) / max(len(group), 1),
                "triangulation_rmse": _rmse(valid, "triangulation_error"),
                "triangulation_median": _median(valid, "triangulation_error"),
                "triangulation_p90": _quantile(valid, "triangulation_error", 0.9),
                "raw_rmse": _rmse(valid, "raw_error"),
                "raw_median": _median(valid, "raw_error"),
                "raw_p90": _quantile(valid, "raw_error", 0.9),
                "improved_anchor_percent": 100.0 * _mean(valid, "improved"),
                "mean_delta_raw_minus_tri": _mean(valid, "delta_raw_minus_tri"),
                "median_delta_raw_minus_tri": _median(valid, "delta_raw_minus_tri"),
                "high_quality_anchors": len(high),
                "high_quality_mean_delta": _mean(high, "delta_raw_minus_tri"),
                "high_quality_improved_percent": 100.0 * _mean(high, "improved"),
            }
        )
        output.append(summary)
    return output


def _condition_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bins = ((0.0, 2.0), (2.0, 5.0), (5.0, 10.0), (10.0, 60.0))
    output = []
    for branch in sorted({row["branch"] for row in rows}):
        branch_rows = [row for row in rows if row["branch"] == branch and row["valid"]]
        for low, high in bins:
            selected = [row for row in branch_rows if low <= row["ray_angle_degrees"] < high]
            output.append(
                {
                    "branch": branch,
                    "ray_angle_bin": f"[{low},{high})",
                    "valid_anchors": len(selected),
                    "triangulation_rmse": _rmse(selected, "triangulation_error"),
                    "raw_rmse": _rmse(selected, "raw_error"),
                    "mean_delta_raw_minus_tri": _mean(selected, "delta_raw_minus_tri"),
                    "improved_anchor_percent": 100.0 * _mean(selected, "improved"),
                }
            )
    return output


def _trackhead_preflight(
    run: V2Run, payload: dict, ownership: torch.Tensor, units: list[PairUnit]
) -> dict[str, Any]:
    config = _section(run, "t0").get("trackhead_preflight", {})
    if not config.get("enabled", True):
        return {"status": "DISABLED", "usable": 0}
    if not units:
        return {"status": "NO_EQUAL_SUPPORT_UNIT", "usable": 0}
    try:
        maybe_add_repo_to_path(_resolve(config["streamvggt_repo"]))
        from streamvggt.heads.track_head import TrackHead

        device = torch.device(str(config["device"]))
        token_levels = payload["token_levels"]
        layer_indices = tuple(int(value) for value in payload["dpt_layer_indices"])
        dim = int(token_levels.shape[-1])
        head = TrackHead(dim_in=dim).to(device).eval()
        checkpoint = torch.load(
            _resolve(config["checkpoint"]), map_location="cpu", weights_only=False
        )
        state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        head_state = {}
        for key, value in state.items():
            normalized = str(key)
            for prefix in ("module.", "model."):
                if normalized.startswith(prefix):
                    normalized = normalized[len(prefix):]
            if normalized.startswith("track_head."):
                head_state[normalized[len("track_head."):]] = value
        missing, unexpected = head.load_state_dict(head_state, strict=False)
        if missing or unexpected or not head_state:
            return {
                "status": "TRACKHEAD_WEIGHTS_INCOMPATIBLE",
                "usable": 0,
                "missing_keys": len(missing),
                "unexpected_keys": len(unexpected),
                "loaded_keys": len(head_state),
            }
        unit = units[0]
        window_count = min(
            int(config["window_frames"]),
            unit.current_index - unit.history_index + 1,
        )
        indices = torch.linspace(
            unit.history_index,
            unit.current_index,
            steps=max(2, window_count),
        ).round().long().unique(sorted=True).tolist()
        queries = unit.current_queries[: int(config["query_count"])]
        # TrackHead takes queries in the first frame. Use a mask-derived query
        # set from that first frame rather than mislabelling current-frame points.
        first_mask = ownership[indices[0]] == unit.slot
        queries = sample_mask_grid(
            first_mask, stride=8, max_points=int(config["query_count"]), margin=3
        )
        if queries.shape[0] == 0:
            return {"status": "NO_PREFLIGHT_QUERIES", "usable": 0}
        selected_layers: list[Any] = [None] * (max(layer_indices) + 1)
        for cached_index, layer_index in enumerate(layer_indices):
            selected_layers[layer_index] = token_levels[
                cached_index, indices
            ].unsqueeze(0).to(device)
        images = payload["stream_images"][indices].unsqueeze(0).to(device)
        with torch.inference_mode():
            coordinates, visibility, confidence = head(
                selected_layers,
                images,
                int(payload["patch_start_idx"]),
                query_points=queries.unsqueeze(0).to(device),
                iters=int(config["iterations"]),
            )
        coordinates = coordinates[-1].detach().float().cpu()
        visibility = visibility.detach().float().cpu()
        confidence = confidence.detach().float().cpu()
        passing = (
            (visibility >= float(config["visibility_threshold"]))
            & (confidence >= float(config["confidence_threshold"]))
            & torch.isfinite(coordinates).all(dim=-1)
        )
        usable = int(bool(passing[:, 1:].any())) if passing.shape[1] > 1 else 0
        return {
            "status": "TRACKHEAD_USABLE" if usable else "TRACKHEAD_UNUSABLE",
            "usable": usable,
            "window_sequence_indices": indices,
            "queries": int(queries.shape[0]),
            "observations": int(passing.numel()),
            "passing_observations": int(passing.sum()),
            "visibility_mean": float(visibility.mean()),
            "confidence_mean": float(confidence.mean()),
            "thresholds_lowered_after_run": 0,
            "sigmoid_applied_by_base_tracker": 1,
        }
    except Exception as error:
        return {
            "status": "TRACKHEAD_PREFLIGHT_ERROR",
            "usable": 0,
            "error_type": type(error).__name__,
            "error": str(error),
            "thresholds_lowered_after_run": 0,
        }


def _write_final_summary(
    run: V2Run, payload: dict, cache_path_value: Path, qk_path: Path, *,
    d0: dict, d1: dict | None, t0: dict | None, stopped_after: str | None,
) -> Path:
    if stopped_after:
        decision = "STOP_FIX_D0_COORDINATES"
    elif t0 and t0["go_t1"]:
        decision = "GO_T1"
    elif d1 and d1["boundary_signal"]:
        decision = "GO_BOUNDARY"
    else:
        decision = "NO_GEOMETRY_KEEP_V0"
    summary = {
        "schema": 1,
        "revision": REVISION,
        "baseline_version": "v0",
        "baseline_status": "frozen_unchanged",
        "clip": payload["clip_name"],
        "frames": payload["frame_indices"],
        "cache": str(cache_path_value),
        "qk_pose_artifact": str(qk_path),
        "model_trained": 0,
        "formal_v0_pose_modified": 0,
        "formal_v0_pointmap_modified": 0,
        "formal_v0_semantic_map_modified": 0,
        "stopped_after": stopped_after,
        "d0": d0,
        "d1": d1,
        "t0": t0,
        "decision": decision,
    }
    result = run.output_dir / "summary.json"
    _write_json(result, summary)
    _write_copyable(run.output_dir / "copyable_result.txt", summary)
    _write_results_markdown(run.output_dir / "results.md", summary)
    return result


def _write_copyable(path: Path, summary: dict[str, Any]) -> None:
    d0 = summary["d0"]
    d1 = summary.get("d1")
    t0 = summary.get("t0")
    lines = [
        "===== COPYABLE_V2_POINTMAP_DIAGNOSIS_BEGIN =====",
        f"revision={REVISION}",
        f"clip={summary['clip']}",
        "baseline_v0_modified=0",
        f"d0_sanity_pass={d0['sanity_pass']}",
        "d0_sanity_metrics=" + json.dumps(
            d0["sanity_metrics_on_independent_gt_support"], sort_keys=True
        ),
        "d0_raw_depth_reference_affine=" + json.dumps(
            d0["raw_depth_reference_affine"], sort_keys=True
        ),
        "",
        "D0_BRANCHES",
        "branch,table_role,supported_points,valid_ratio,paired_rmse,paired_median,paired_p90,symmetric_mean",
    ]
    for row in d0["branches"]:
        lines.append(
            ",".join(str(row.get(name, "")) for name in (
                "branch", "table_role", "supported_points", "valid_ratio",
                "paired_rmse", "paired_median", "paired_p90", "symmetric_mean",
            ))
        )
    if d1:
        lines.extend(
            (
                "",
                f"d1_boundary_signal={d1['boundary_signal']}",
                "d1_boundary_median_ratio_by_width=" + json.dumps(d1["boundary_median_ratio_by_width"], sort_keys=True),
                f"d1_high_confidence_wrong_percent={d1['high_confidence_wrong']['high_confidence_wrong_percent']}",
                "",
                "D1_REGIONS",
                "boundary_width,region,count,mean,median,p90,rmse",
            )
        )
        for row in d1["regions"]:
            lines.append(
                ",".join(str(row.get(name, "")) for name in (
                    "boundary_width", "region", "count", "mean", "median", "p90", "rmse",
                ))
            )
        lines.extend(("", "D1_INSTANCES", "slot,prompt,sam_track_id,visible_frames,point_count,raw_error_median,raw_error_p90,interior_error_median,boundary_error_median,boundary_over_interior_median_ratio"))
        for row in d1["instances"]:
            lines.append(
                ",".join(str(row.get(name, "")) for name in (
                    "slot", "prompt", "sam_track_id", "visible_frames", "point_count",
                    "raw_error_median", "raw_error_p90", "interior_error_median",
                    "boundary_error_median", "boundary_over_interior_median_ratio",
                ))
            )
        lines.extend(("", "D1_CONFIDENCE", "confidence_bin,count,confidence_min,confidence_max,mean,median,p90,rmse"))
        for row in d1["confidence_bins"]:
            lines.append(
                ",".join(str(row.get(name, "")) for name in (
                    "confidence_bin", "count", "confidence_min", "confidence_max",
                    "mean", "median", "p90", "rmse",
                ))
            )
    if t0:
        lines.extend(
            (
                f"t0_pair_units={t0['pair_units']}",
                f"t0_equal_support={t0['equal_current_query_support_correct_vs_shuffled']}",
                f"t0_correct_id_beats_shuffled={t0['correct_id_beats_shuffled']}",
                f"t0_correct_id_validity_advantage={t0['correct_id_validity_advantage']}",
                f"t0_correct_id_error_comparison_available={t0['correct_id_error_comparison_available']}",
                f"t0_correct_id_error_advantage={t0['correct_id_error_advantage']}",
                f"t0_go_t1={t0['go_t1']}",
                f"trackhead_status={t0['trackhead_preflight']['status']}",
                "",
                "branch,candidate_matches,valid_anchors,triangulation_rmse,raw_rmse,improved_anchor_percent,mean_delta_raw_minus_tri,high_quality_anchors,high_quality_mean_delta",
            )
        )
        for row in t0["branches"]:
            lines.append(
                ",".join(str(row.get(name, "")) for name in (
                    "branch", "candidate_matches", "valid_anchors", "triangulation_rmse",
                    "raw_rmse", "improved_anchor_percent", "mean_delta_raw_minus_tri",
                    "high_quality_anchors", "high_quality_mean_delta",
                ))
            )
    lines.extend(("", f"decision={summary['decision']}", "===== COPYABLE_V2_POINTMAP_DIAGNOSIS_END ====="))
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def _write_results_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Pointmap Diagnosis V2 Results", "", "## A. D0: Depth x Pose Oracle", "",
        f"- GT depth + GT pose sanity pass: {summary['d0']['sanity_pass']}",
        "- See `d0_depth_pose_oracle/summary.csv` for the fixed-K six-cell table.", "",
        "## B. D1: Raw Pointmap Error", "",
    ]
    if summary.get("d1"):
        lines.extend((
            f"- Boundary signal: {summary['d1']['boundary_signal']}",
            f"- Boundary/interior median ratios: `{summary['d1']['boundary_median_ratio_by_width']}`",
            f"- High-confidence-wrong: {summary['d1']['high_confidence_wrong']['high_confidence_wrong_percent']:.3f}%", "",
        ))
    else:
        lines.extend(("- Not run because D0 failed.", ""))
    lines.extend(("## C. T0: Independent Triangulation", ""))
    if summary.get("t0"):
        lines.extend((
            f"- Pair units: {summary['t0']['pair_units']}",
            f"- Correct ID validity advantage: {summary['t0']['correct_id_validity_advantage']}",
            f"- Correct/shuffled error comparison available: {summary['t0']['correct_id_error_comparison_available']}",
            f"- Correct ID error advantage: {summary['t0']['correct_id_error_advantage']}",
            f"- TrackHead preflight: `{summary['t0']['trackhead_preflight']['status']}`", "",
        ))
    else:
        lines.extend(("- Not run.", ""))
    lines.extend(("## D. Next Step", "", f"**{summary['decision']}**", ""))
    path.write_text("\n".join(lines), encoding="utf8")


def _validate_frozen_inputs(payload: dict, qk: dict, clip_name: str) -> None:
    if payload.get("clip_name") != clip_name or qk.get("selected_pose_branch") != "retrieve_qk":
        raise ValueError("V0 cache/QK artifacts do not match the frozen clip.")
    if tuple(qk.get("frame_indices", ())) != tuple(payload.get("frame_indices", ())):
        raise ValueError("QK pose artifact frame order differs from the V0 cache.")


def _camera_centers(world_to_camera: torch.Tensor) -> torch.Tensor:
    rotation = world_to_camera[:, :3, :3]
    translation = world_to_camera[:, :3, 3]
    return -(rotation.transpose(-1, -2) @ translation[..., None]).squeeze(-1)


def _mean(rows: list[dict[str, Any]], field: str) -> float:
    return float("nan") if not rows else sum(float(row[field]) for row in rows) / len(rows)


def _median(rows: list[dict[str, Any]], field: str) -> float:
    return float("nan") if not rows else float(torch.tensor([row[field] for row in rows]).median())


def _quantile(rows: list[dict[str, Any]], field: str, q: float) -> float:
    return float("nan") if not rows else float(torch.quantile(torch.tensor([row[field] for row in rows]).float(), q))


def _rmse(rows: list[dict[str, Any]], field: str) -> float:
    return float("nan") if not rows else float(torch.tensor([row[field] for row in rows]).float().square().mean().sqrt())


def _load_run(path: str | Path) -> V2Run:
    source = Path(path).expanduser().resolve()
    raw = _read_yaml(source)
    run = V2Run(
        source_path=source,
        v0_config=_resolve(raw["v0_config"]),
        output_dir=_resolve(raw["output_dir"]),
        raw=raw,
    )
    _validate_run(run)
    return run


def _validate_run(run: V2Run) -> None:
    if run.v0_config.name != "v0_baseline.yaml":
        raise ValueError("V2 diagnosis must read the frozen v0_baseline.yaml.")
    if run.output_dir.name != "pointmap_diagnosis_v2":
        raise ValueError("V2 output_dir must remain isolated as pointmap_diagnosis_v2.")
    d0, d1, t0 = (_section(run, name) for name in ("d0", "d1", "t0"))
    if int(d0.get("paired_max_points_per_frame", 0)) < 1 or int(
        d0.get("symmetric_max_points_per_frame", 0)
    ) < 1:
        raise ValueError("D0 sample budgets must be positive.")
    if tuple(int(value) for value in d1.get("boundary_widths", ())) != (3, 5):
        raise ValueError("D1 primary/sensitivity boundary widths are fixed at 3 and 5 px.")
    if float(d1.get("track_score_threshold", -1.0)) != float(
        t0.get("track_score_threshold", -2.0)
    ):
        raise ValueError("D1 and T0 must use the same fixed SAM score threshold.")
    ratio = float(t0.get("descriptor_ratio_threshold", -1.0))
    if not 0.0 < ratio < 1.0:
        raise ValueError("T0 descriptor ratio threshold must be in (0,1).")
    if int(t0.get("min_high_quality_anchors_for_go", 0)) < 1:
        raise ValueError("T0 GO gate requires a positive fixed anchor count.")


def _section(run: V2Run, name: str) -> dict[str, Any]:
    value = run.raw.get(name, {})
    if not isinstance(value, dict):
        raise TypeError(f"{name} config must be a mapping.")
    return value


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return value


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (Path.cwd() / value).resolve()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=True, default=str) + "\n",
        encoding="utf8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf8")
        return
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="streaming_couping/configs/v2_pointmap_diagnosis.yaml"
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
