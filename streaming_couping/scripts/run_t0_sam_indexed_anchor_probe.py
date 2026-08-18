#!/usr/bin/env python3
"""Probe independent 3D anchors from SAM-indexed 2D correspondences."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import torch
import yaml

from streaming_couping.src.config import load_config
from streaming_couping.src.external_repos import maybe_add_repo_to_path
from streaming_couping.src.independent_triangulation import (
    evenly_limited_indices,
    extract_frame_patch_descriptors,
    masks_to_patch_support,
    mutual_nearest_matches,
    patch_centers,
    shift_patch_mask_exact,
    triangulate_camera_rays,
)
from streaming_couping.src.learned_pose.baseline_runtime import (
    load_baseline_run_config,
)
from streaming_couping.src.learned_pose.cache import cache_path, load_feature_cache
from streaming_couping.src.learned_pose.config import (
    ClipConfig,
    load_learned_pose_config,
)
from streaming_couping.src.pointmap_alignment import _robust_similarity
from streaming_couping.src.semantic_map import normalize_confidence


REVISION = "t0_sam_indexed_independent_3d_anchor_probe_r1"
BRANCHES = (
    "foreground_union",
    "correct_persistent_id",
    "shuffled_persistent_id",
    "shifted_mask_control",
)


@dataclass(frozen=True)
class T0Run:
    source_path: Path
    v0_config: Path
    output_dir: Path
    descriptor_level_position: int
    descriptor_token_half: str
    track_score_threshold: float
    mask_erosion_pixels: int
    minimum_patch_coverage: float
    excluded_prompts: tuple[str, ...]
    qk_history_pool_size: int
    maximum_queries_per_instance_frame: int
    maximum_history_views: int
    minimum_match_similarity: float
    minimum_match_margin: float
    minimum_views: int
    minimum_ray_angle_degrees: float
    maximum_condition_number: float
    maximum_mean_reprojection_patch_fraction: float
    maximum_reprojection_patch_fraction: float
    shifted_mask_fraction_y: float
    shifted_mask_fraction_x: float
    minimum_equal_count: int


def main() -> None:
    args = _parse_args()
    run = _load_run(args.config)
    learned = load_learned_pose_config(run.v0_config)
    baseline = load_baseline_run_config(run.v0_config)
    clip = _find_clip(learned.clips, baseline.clip_name)
    payload_path = cache_path(learned, clip)
    payload = load_feature_cache(payload_path)
    qk_path = _qk_artifact_path(run.v0_config)
    qk = torch.load(qk_path, map_location="cpu", weights_only=False)
    ranking_path = qk_path.with_name("retrieval_diagnostics.csv")
    _validate_inputs(payload, qk, clip, ranking_path)

    recovery = load_config(learned.recovery_config)
    maybe_add_repo_to_path(recovery.streamvggt_repo)
    qk_pose, qk_intrinsics = _decode_pose(qk, payload)
    rankings = _load_qk_rankings(ranking_path, payload["frame_indices"])
    candidate_payload = _candidate_payload(
        payload,
        qk_pose=qk_pose,
        qk_intrinsics=qk_intrinsics,
        rankings=rankings,
        run=run,
    )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("T0 requested CUDA but torch.cuda.is_available() is false.")
    run.output_dir.mkdir(parents=True, exist_ok=True)
    print("T0 SAM-INDEXED INDEPENDENT 3D ANCHOR PROBE")
    print("  candidate=2D patch matches + frozen QK camera rays")
    print("  raw_pointmap=0 GT=0 model_loaded=0 model_trained=0")
    candidates, diagnostic_rows, unit_rows = _generate_candidates(
        candidate_payload,
        run=run,
        device=device,
    )
    artifact_path = run.output_dir / "anchors_candidate.pt"
    torch.save(
        {
            "schema": 1,
            "revision": REVISION,
            "artifact_role": "frozen_independent_3d_anchor_candidates",
            "clip_name": str(payload["clip_name"]),
            "scene_id": str(payload["scene_id"]),
            "frame_indices": tuple(int(value) for value in payload["frame_indices"]),
            "candidate_generation_fields": (
                "cached_streamvggt_patch_tokens",
                "sam_tracking_masks_and_scores",
                "sam_persistent_slots",
                "frozen_qk_pose_and_intrinsics",
                "frozen_qk_history_ranking",
            ),
            "candidate_generation_raw_pointmap_fields": 0,
            "candidate_generation_gt_fields": 0,
            "branches": candidates,
        },
        artifact_path,
    )
    print("  independent anchors frozen; raw pointmap and GT enter scoring now")
    summary, score_rows, instance_rows = _score_candidates(
        payload=payload,
        qk=qk,
        candidates=candidates,
        diagnostic_rows=diagnostic_rows,
        run=run,
        payload_path=payload_path,
        qk_path=qk_path,
        artifact_path=artifact_path,
    )
    _write_outputs(
        run,
        summary=summary,
        diagnostic_rows=diagnostic_rows,
        unit_rows=unit_rows,
        score_rows=score_rows,
        instance_rows=instance_rows,
    )
    primary = summary["branch_lookup"]["correct_persistent_id"]
    print(
        "  correct_id "
        f"anchors={primary['evaluated_anchor_count']} "
        f"tri_gain={primary['tri_gain_vs_raw_percent']:.4f}% "
        f"improved_ratio={primary['improved_anchor_ratio']:.4f}"
    )
    print(
        f"  decision={summary['decision']['t0_decision']} "
        f"sam_unique={summary['decision']['sam_identity_unique_evidence']}"
    )
    print(f"  result={run.output_dir / 'summary.json'}")


def _candidate_payload(
    payload: dict[str, Any],
    *,
    qk_pose: torch.Tensor,
    qk_intrinsics: torch.Tensor,
    rankings: tuple[tuple[int, ...], ...],
    run: T0Run,
) -> dict[str, Any]:
    allowed = {
        "token_levels": payload["token_levels"],
        "patch_start_idx": int(payload["patch_start_idx"]),
        "patch_shape": tuple(int(value) for value in payload["patch_shape"]),
        "image_size": tuple(int(value) for value in payload["image_size"]),
        "frame_indices": tuple(int(value) for value in payload["frame_indices"]),
        "tracking_masks_stream": payload["tracking_masks_stream"],
        "tracking_scores": payload["tracking_scores"],
        "sam_track_ids": tuple(int(value) for value in payload["sam_track_ids"]),
        "sam_track_prompts": tuple(str(value) for value in payload["sam_track_prompts"]),
        "qk_world_to_camera": qk_pose,
        "qk_intrinsics": qk_intrinsics,
        "qk_rankings": rankings,
    }
    forbidden = {
        "baseline_world_points",
        "baseline_world_confidence",
        "target_world_points",
        "target_world_to_camera",
        "target_depth",
    }
    if forbidden.intersection(allowed):
        raise RuntimeError("T0 candidate payload contains raw/GT geometry.")
    descriptors = extract_frame_patch_descriptors(
        allowed["token_levels"],
        level_position=run.descriptor_level_position,
        patch_start_idx=allowed["patch_start_idx"],
        patch_shape=allowed["patch_shape"],
        token_half=run.descriptor_token_half,
    )
    patch_masks = masks_to_patch_support(
        allowed["tracking_masks_stream"],
        allowed["tracking_scores"],
        patch_shape=allowed["patch_shape"],
        score_threshold=run.track_score_threshold,
        minimum_patch_coverage=run.minimum_patch_coverage,
        erosion_pixels=run.mask_erosion_pixels,
    )
    return {
        **{key: value for key, value in allowed.items() if key != "token_levels"},
        "patch_descriptors": descriptors,
        "patch_masks": patch_masks,
        "patch_centers_xy": patch_centers(
            allowed["patch_shape"], allowed["image_size"]
        ),
    }


@torch.inference_mode()
def _generate_candidates(
    payload: dict[str, Any],
    *,
    run: T0Run,
    device: torch.device,
) -> tuple[dict[str, dict[str, torch.Tensor]], list[dict[str, Any]], list[dict[str, Any]]]:
    descriptors = payload["patch_descriptors"].to(device)
    masks = payload["patch_masks"].bool().cpu()
    frames = payload["frame_indices"]
    prompts = payload["sam_track_prompts"]
    track_ids = payload["sam_track_ids"]
    rankings = payload["qk_rankings"]
    poses = payload["qk_world_to_camera"].float().cpu()
    intrinsics = payload["qk_intrinsics"].float().cpu()
    centers_xy = payload["patch_centers_xy"].float().cpu()
    sequence, slots, grid_height, grid_width = masks.shape
    if descriptors.shape[:2] != (sequence, grid_height * grid_width):
        raise ValueError("Patch descriptor/mask grids are not aligned.")
    if not (
        len(frames)
        == len(rankings)
        == poses.shape[0]
        == intrinsics.shape[0]
        == sequence
    ):
        raise ValueError("T0 frame-index, descriptor, ranking and camera lengths differ.")
    if len(prompts) != slots or len(track_ids) != slots:
        raise ValueError("T0 persistent-slot metadata does not match mask slots.")
    eligible_slots = tuple(
        slot
        for slot in range(slots)
        if int(track_ids[slot]) >= 0
        and prompts[slot].strip().lower() not in run.excluded_prompts
        and bool(masks[:, slot].any())
    )
    if len(eligible_slots) < 2:
        raise ValueError("T0 needs at least two eligible persistent instances.")
    shuffled = {
        slot: eligible_slots[(position + 1) % len(eligible_slots)]
        for position, slot in enumerate(eligible_slots)
    }
    ownership = _patch_ownership(
        masks,
        payload["tracking_scores"],
        eligible_slots=eligible_slots,
    )
    eligible_index = torch.tensor(eligible_slots, dtype=torch.long)
    foreground = masks.index_select(1, eligible_index).any(dim=1)
    shift_y = max(1, round(grid_height * run.shifted_mask_fraction_y))
    shift_x = max(1, round(grid_width * run.shifted_mask_fraction_x))
    pose_centers = _camera_centers(poses)
    maximum_mean_reprojection = run.maximum_mean_reprojection_patch_fraction * max(
        payload["image_size"][0] / grid_height,
        payload["image_size"][1] / grid_width,
    )
    maximum_reprojection = run.maximum_reprojection_patch_fraction * max(
        payload["image_size"][0] / grid_height,
        payload["image_size"][1] / grid_width,
    )

    branch_anchors: dict[str, list[dict[str, Any]]] = {
        branch: [] for branch in BRANCHES
    }
    diagnostics: list[dict[str, Any]] = []
    unit_rows: list[dict[str, Any]] = []
    for current in range(sequence):
        for slot in eligible_slots:
            current_mask = masks[current, slot] & (ownership[current] == slot)
            query_indices = evenly_limited_indices(
                current_mask,
                limit=run.maximum_queries_per_instance_frame,
            )
            if query_indices.numel() == 0:
                continue
            candidate_histories = tuple(
                history
                for history in rankings[current]
                if bool(masks[history, slot].any())
            )[: run.qk_history_pool_size]
            if not candidate_histories:
                continue
            query_device = query_indices.to(device)
            query_descriptors = descriptors[current].index_select(0, query_device)
            for branch in BRANCHES:
                tracks: dict[int, list[dict[str, Any]]] = {
                    int(index): [] for index in query_indices
                }
                match_count = 0
                for history in candidate_histories:
                    search_mask = _history_search_mask(
                        branch,
                        masks=masks,
                        foreground=foreground,
                        history=history,
                        slot=slot,
                        shuffled_slot=shuffled[slot],
                        shift_y=shift_y,
                        shift_x=shift_x,
                    )
                    search_indices = torch.nonzero(
                        search_mask.reshape(-1), as_tuple=False
                    )[:, 0].long()
                    if search_indices.numel() == 0:
                        continue
                    search_device = search_indices.to(device)
                    matches = mutual_nearest_matches(
                        query_descriptors,
                        descriptors[history].index_select(0, search_device),
                        minimum_similarity=run.minimum_match_similarity,
                        minimum_margin=run.minimum_match_margin,
                    )
                    match_values = zip(
                        matches.query_positions.detach().cpu().tolist(),
                        matches.search_positions.detach().cpu().tolist(),
                        matches.similarities.detach().cpu().tolist(),
                        matches.margins.detach().cpu().tolist(),
                    )
                    for query_position, search_position, similarity, margin in match_values:
                        query_flat = int(query_indices[query_position])
                        history_flat = int(search_indices[search_position])
                        tracks[query_flat].append(
                            {
                                "history": int(history),
                                "history_flat": history_flat,
                                "similarity": float(similarity),
                                "margin": float(margin),
                                "baseline": float(
                                    torch.linalg.vector_norm(
                                        pose_centers[history] - pose_centers[current]
                                    )
                                ),
                            }
                        )
                        match_count += 1

                accepted_in_unit = 0
                triangulated_in_unit = 0
                for query_flat in query_indices.tolist():
                    matches_for_query = sorted(
                        tracks[int(query_flat)],
                        key=lambda item: (
                            -float(item["baseline"]),
                            -float(item["similarity"]),
                            int(item["history"]),
                        ),
                    )[: run.maximum_history_views]
                    row = {
                        "branch": branch,
                        "current_sequence_index": current,
                        "current_frame_index": frames[current],
                        "slot": slot,
                        "sam_track_id": track_ids[slot],
                        "prompt": prompts[slot],
                        "control_history_slot": (
                            shuffled[slot]
                            if branch == "shuffled_persistent_id"
                            else slot
                        ),
                        "query_patch_flat_index": int(query_flat),
                        "candidate_history_sequence_indices": _space(candidate_histories),
                        "candidate_history_frame_indices": _space(
                            frames[index] for index in candidate_histories
                        ),
                        "matched_history_sequence_indices": _space(
                            item["history"] for item in matches_for_query
                        ),
                        "num_views": 1 + len(matches_for_query),
                    }
                    if len(matches_for_query) + 1 < run.minimum_views:
                        diagnostics.append(
                            {
                                **row,
                                "mean_match_similarity": float("nan"),
                                "mean_match_margin": float("nan"),
                                "condition_number": float("nan"),
                                "minimum_ray_angle_degrees": float("nan"),
                                "maximum_ray_angle_degrees": float("nan"),
                                "mean_reprojection_error_px": float("nan"),
                                "maximum_reprojection_error_px": float("nan"),
                                "positive_depth_rate": float("nan"),
                                "accepted": 0,
                                "rejection_reason": "insufficient_matched_views",
                            }
                        )
                        continue
                    view_indices = [current] + [
                        int(item["history"]) for item in matches_for_query
                    ]
                    pixel_indices = [int(query_flat)] + [
                        int(item["history_flat"]) for item in matches_for_query
                    ]
                    pixels = centers_xy.index_select(
                        0, torch.tensor(pixel_indices, dtype=torch.long)
                    )
                    triangulation = triangulate_camera_rays(
                        pixels,
                        torch.tensor(view_indices, dtype=torch.long),
                        poses,
                        intrinsics,
                    )
                    triangulated_in_unit += 1
                    mean_similarity = sum(
                        float(item["similarity"]) for item in matches_for_query
                    ) / len(matches_for_query)
                    mean_margin = sum(
                        float(item["margin"]) for item in matches_for_query
                    ) / len(matches_for_query)
                    rejection = _triangulation_rejection(
                        triangulation,
                        run=run,
                        maximum_mean_reprojection_px=maximum_mean_reprojection,
                        maximum_reprojection_px=maximum_reprojection,
                    )
                    accepted = int(rejection == "")
                    diagnostics.append(
                        {
                            **row,
                            "mean_match_similarity": mean_similarity,
                            "mean_match_margin": mean_margin,
                            "condition_number": triangulation.condition_number,
                            "minimum_ray_angle_degrees": triangulation.minimum_ray_angle_degrees,
                            "maximum_ray_angle_degrees": triangulation.maximum_ray_angle_degrees,
                            "mean_reprojection_error_px": triangulation.mean_reprojection_error_px,
                            "maximum_reprojection_error_px": triangulation.maximum_reprojection_error_px,
                            "positive_depth_rate": triangulation.positive_depth_rate,
                            "accepted": accepted,
                            "rejection_reason": rejection,
                        }
                    )
                    if not accepted:
                        continue
                    accepted_in_unit += 1
                    branch_anchors[branch].append(
                        {
                            "point_world": triangulation.point_world,
                            "current_sequence_index": current,
                            "current_frame_index": frames[current],
                            "slot": slot,
                            "sam_track_id": track_ids[slot],
                            "query_patch_flat_index": int(query_flat),
                            "query_pixel_xy": centers_xy[int(query_flat)],
                            "num_views": 1 + len(matches_for_query),
                            "mean_match_similarity": mean_similarity,
                            "mean_match_margin": mean_margin,
                            "condition_number": triangulation.condition_number,
                            "minimum_ray_angle_degrees": triangulation.minimum_ray_angle_degrees,
                            "maximum_ray_angle_degrees": triangulation.maximum_ray_angle_degrees,
                            "mean_reprojection_error_px": triangulation.mean_reprojection_error_px,
                            "maximum_reprojection_error_px": triangulation.maximum_reprojection_error_px,
                        }
                    )
                unit_rows.append(
                    {
                        "branch": branch,
                        "current_sequence_index": current,
                        "current_frame_index": frames[current],
                        "slot": slot,
                        "sam_track_id": track_ids[slot],
                        "prompt": prompts[slot],
                        "query_count": int(query_indices.numel()),
                        "candidate_history_count": len(candidate_histories),
                        "pairwise_match_count": match_count,
                        "triangulated_track_count": triangulated_in_unit,
                        "accepted_anchor_count": accepted_in_unit,
                    }
                )
    output = {
        branch: _pack_anchors(branch_anchors[branch]) for branch in BRANCHES
    }
    del descriptors
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output, diagnostics, unit_rows


def _score_candidates(
    *,
    payload: dict[str, Any],
    qk: dict[str, Any],
    candidates: dict[str, dict[str, torch.Tensor]],
    diagnostic_rows: list[dict[str, Any]],
    run: T0Run,
    payload_path: Path,
    qk_path: Path,
    artifact_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    raw_points = payload["baseline_world_points"].detach().float().cpu()
    target_points = payload["target_world_points"].detach().float().cpu()
    raw_confidence = normalize_confidence(payload["baseline_world_confidence"])
    target_pose = payload["target_world_to_camera"].detach().float().cpu()
    qk_pose = qk["selected_world_to_camera"].detach().float().cpu()
    raw_pose = qk["raw_world_to_camera"].detach().float().cpu()
    qk_alignment = _pose_sim3(qk_pose, target_pose)
    raw_alignment = _pose_sim3(raw_pose, target_pose)
    attempted = {
        branch: sum(1 for row in diagnostic_rows if row["branch"] == branch)
        for branch in BRANCHES
    }
    candidate_valid = {
        branch: sum(
            int(row["accepted"])
            for row in diagnostic_rows
            if row["branch"] == branch
        )
        for branch in BRANCHES
    }
    score_rows: list[dict[str, Any]] = []
    branch_rows: list[dict[str, Any]] = []
    rows_by_branch: dict[str, list[dict[str, Any]]] = {}
    for branch in BRANCHES:
        anchor = candidates[branch]
        branch_score_rows = []
        for index in range(anchor["point_world"].shape[0]):
            frame = int(anchor["current_sequence_index"][index])
            x = int(round(float(anchor["query_pixel_xy"][index, 0])))
            y = int(round(float(anchor["query_pixel_xy"][index, 1])))
            x = min(max(x, 0), raw_points.shape[2] - 1)
            y = min(max(y, 0), raw_points.shape[1] - 1)
            raw_native = raw_points[frame, y, x]
            truth = target_points[frame, y, x]
            triangulated = _apply_sim3(
                anchor["point_world"][index], qk_alignment
            )
            raw = _apply_sim3(raw_native, raw_alignment)
            if not (
                torch.isfinite(triangulated).all()
                and torch.isfinite(raw).all()
                and torch.isfinite(truth).all()
            ):
                continue
            tri_error = float(torch.linalg.vector_norm(triangulated - truth))
            raw_error = float(torch.linalg.vector_norm(raw - truth))
            row = {
                "branch": branch,
                "anchor_index": index,
                "current_sequence_index": frame,
                "current_frame_index": int(anchor["current_frame_index"][index]),
                "slot": int(anchor["slot"][index]),
                "sam_track_id": int(anchor["sam_track_id"][index]),
                "query_patch_flat_index": int(anchor["query_patch_flat_index"][index]),
                "query_pixel_x": float(anchor["query_pixel_xy"][index, 0]),
                "query_pixel_y": float(anchor["query_pixel_xy"][index, 1]),
                "num_views": int(anchor["num_views"][index]),
                "mean_match_similarity": float(anchor["mean_match_similarity"][index]),
                "mean_match_margin": float(anchor["mean_match_margin"][index]),
                "condition_number": float(anchor["condition_number"][index]),
                "maximum_ray_angle_degrees": float(
                    anchor["maximum_ray_angle_degrees"][index]
                ),
                "mean_reprojection_error_px": float(
                    anchor["mean_reprojection_error_px"][index]
                ),
                "raw_confidence": float(raw_confidence[frame, y, x]),
                "tri_gt_error": tri_error,
                "raw_gt_error": raw_error,
                "delta_raw_minus_tri": raw_error - tri_error,
                "tri_improved": int(tri_error < raw_error),
            }
            branch_score_rows.append(row)
            score_rows.append(row)
        rows_by_branch[branch] = branch_score_rows
        branch_rows.append(
            _branch_metrics(
                branch,
                branch_score_rows,
                attempted_queries=attempted[branch],
                candidate_valid_anchors=candidate_valid[branch],
            )
        )

    equal_branches = (
        "correct_persistent_id",
        "shuffled_persistent_id",
        "shifted_mask_control",
    )
    equal_count = min(len(rows_by_branch[name]) for name in equal_branches)
    equal_metrics = {}
    for branch in equal_branches:
        ordered = sorted(
            rows_by_branch[branch],
            key=lambda row: (
                float(row["mean_reprojection_error_px"]),
                float(row["condition_number"]),
                -float(row["mean_match_similarity"]),
                int(row["current_sequence_index"]),
                int(row["slot"]),
                int(row["query_patch_flat_index"]),
            ),
        )[:equal_count]
        equal_metrics[branch] = {
            **_paired_error_metrics(ordered),
            "mean_reprojection_error_px": _mean_or_nan(
                [float(row["mean_reprojection_error_px"]) for row in ordered]
            ),
        }
    lookup = {row["branch"]: row for row in branch_rows}
    for branch in equal_branches:
        for key, value in equal_metrics[branch].items():
            lookup[branch][f"equal_count_{key}"] = value
        lookup[branch]["equal_count"] = equal_count

    correct = lookup["correct_persistent_id"]
    shuffled = lookup["shuffled_persistent_id"]
    shifted = lookup["shifted_mask_control"]
    correct_help = _branch_anchor_help(correct)
    foreground_help = _branch_anchor_help(lookup["foreground_union"])
    equal_count_ready = int(equal_count >= run.minimum_equal_count)
    sam_unique = int(
        equal_count_ready
        and float(correct["equal_count_tri_rmse"])
        < min(
            float(shuffled["equal_count_tri_rmse"]),
            float(shifted["equal_count_tri_rmse"]),
        )
        and float(correct["candidate_valid_anchor_rate"])
        > max(
            float(shuffled["candidate_valid_anchor_rate"]),
            float(shifted["candidate_valid_anchor_rate"]),
        )
        and float(correct["equal_count_mean_reprojection_error_px"])
        < min(
            float(shuffled["equal_count_mean_reprojection_error_px"]),
            float(shifted["equal_count_mean_reprojection_error_px"]),
        )
    )
    if correct_help and sam_unique:
        t0_decision = "GO"
        claim = "sam_indexed_triangulation_provides_independent_better_3d_anchors"
        next_gate = "local_raw_plus_sparse_anchor_pointmap_refinement"
    elif correct_help or foreground_help:
        t0_decision = "PARTIAL_GO"
        claim = "triangulation_helps_but_sam_identity_unique_contribution_not_established"
        next_gate = "pure_geometry_sparse_anchor_refinement_sam_semantics_only"
    else:
        t0_decision = "NO_GO"
        claim = "frozen_correspondence_and_qk_camera_do_not_beat_raw_pointmap"
        next_gate = "stop_training_free_pointmap_refinement_keep_v0"
    decision = {
        "t0_decision": t0_decision,
        "correct_id_anchor_help": int(correct_help),
        "foreground_anchor_help": int(foreground_help),
        "sam_identity_unique_evidence": sam_unique,
        "equal_count_control_ready": equal_count_ready,
        "equal_count": equal_count,
        "equal_attempted_query_protocol": int(len(set(attempted.values())) == 1),
        "formal_v0_pose_modified": 0,
        "formal_v0_pointmap_modified": 0,
        "formal_v0_semantic_map_modified": 0,
        "next_gate": next_gate,
    }
    instance_rows = _instance_metrics(score_rows, payload)
    summary = {
        "schema": 1,
        "revision": REVISION,
        "experiment": "sam_indexed_independent_3d_anchor_probe",
        "baseline_version": "v0",
        "baseline_status": "frozen_unchanged",
        "clip": payload["clip_name"],
        "scene_id": str(payload["scene_id"]),
        "frames": tuple(int(value) for value in payload["frame_indices"]),
        "branches": branch_rows,
        "branch_lookup": lookup,
        "descriptor": {
            "source": "cached_streamvggt_dpt_patch_tokens",
            "level_position": run.descriptor_level_position,
            "token_half": run.descriptor_token_half,
            "matching": "cosine_mutual_nearest_neighbor_with_margin",
        },
        "matching_policy": {
            "track_score_threshold": run.track_score_threshold,
            "mask_erosion_pixels": run.mask_erosion_pixels,
            "minimum_patch_coverage": run.minimum_patch_coverage,
            "excluded_prompts": run.excluded_prompts,
            "maximum_queries_per_instance_frame": run.maximum_queries_per_instance_frame,
            "minimum_match_similarity": run.minimum_match_similarity,
            "minimum_match_margin": run.minimum_match_margin,
        },
        "history": {
            "source": "frozen_native_qk_ranking",
            "maximum_covisible_frames": run.qk_history_pool_size,
            "maximum_history_views_per_anchor": run.maximum_history_views,
            "causal_only": 1,
        },
        "control_protocol": {
            "same_current_queries": 1,
            "same_candidate_histories": 1,
            "only_historical_search_mask_changes": 1,
            "shifted_mask_area_exact": 1,
            "shuffled_id_rule": "fixed_cyclic_permutation_of_eligible_slots",
            "shifted_mask_fraction_y": run.shifted_mask_fraction_y,
            "shifted_mask_fraction_x": run.shifted_mask_fraction_x,
        },
        "triangulation": {
            "method": "linear_multi_ray_least_squares",
            "minimum_views": run.minimum_views,
            "minimum_ray_angle_degrees": run.minimum_ray_angle_degrees,
            "maximum_condition_number": run.maximum_condition_number,
            "maximum_mean_reprojection_patch_fraction": run.maximum_mean_reprojection_patch_fraction,
            "maximum_reprojection_patch_fraction": run.maximum_reprojection_patch_fraction,
            "thresholds_selected_with_gt": 0,
        },
        "candidate_generation_raw_pointmap_fields": 0,
        "candidate_generation_gt_fields": 0,
        "prompt_selection_annotation_gt_used": 1,
        "runtime_prompt_selection_gt_fields": 0,
        "raw_pointmap_role": "paired_same_query_pixel_scoring_only_after_anchor_freeze",
        "gt_role": "fixed_whole_trajectory_alignment_and_scoring_after_anchor_freeze",
        "model_loaded_or_run": 0,
        "model_trained": 0,
        "sam_hidden_features_used": 0,
        "pose_modified": 0,
        "pointmap_modified": 0,
        "qk_pose_alignment": _alignment_for_json(qk_alignment),
        "raw_pose_alignment": _alignment_for_json(raw_alignment),
        "cache": str(payload_path),
        "qk_pose_artifact": str(qk_path),
        "candidate_artifact": str(artifact_path),
        "decision": decision,
        "claim": claim,
    }
    return summary, score_rows, instance_rows


def _history_search_mask(
    branch: str,
    *,
    masks: torch.Tensor,
    foreground: torch.Tensor,
    history: int,
    slot: int,
    shuffled_slot: int,
    shift_y: int,
    shift_x: int,
) -> torch.Tensor:
    if branch == "foreground_union":
        return foreground[history]
    if branch == "correct_persistent_id":
        return masks[history, slot]
    if branch == "shuffled_persistent_id":
        return masks[history, shuffled_slot]
    if branch == "shifted_mask_control":
        return shift_patch_mask_exact(
            masks[history, slot], shift_y=shift_y, shift_x=shift_x
        )
    raise ValueError(f"Unknown T0 branch {branch!r}.")


def _triangulation_rejection(
    value,
    *,
    run: T0Run,
    maximum_mean_reprojection_px: float,
    maximum_reprojection_px: float,
) -> str:
    if not value.finite:
        return "nonfinite_triangulation"
    if value.positive_depth_rate < 1.0:
        return "negative_depth"
    if value.maximum_ray_angle_degrees < run.minimum_ray_angle_degrees:
        return "insufficient_ray_angle"
    if value.condition_number > run.maximum_condition_number:
        return "ill_conditioned"
    if value.mean_reprojection_error_px > maximum_mean_reprojection_px:
        return "mean_reprojection_too_large"
    if value.maximum_reprojection_error_px > maximum_reprojection_px:
        return "maximum_reprojection_too_large"
    return ""


def _pack_anchors(rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    if not rows:
        return {
            "point_world": torch.empty(0, 3),
            "current_sequence_index": torch.empty(0, dtype=torch.long),
            "current_frame_index": torch.empty(0, dtype=torch.long),
            "slot": torch.empty(0, dtype=torch.long),
            "sam_track_id": torch.empty(0, dtype=torch.long),
            "query_patch_flat_index": torch.empty(0, dtype=torch.long),
            "query_pixel_xy": torch.empty(0, 2),
            "num_views": torch.empty(0, dtype=torch.long),
            "mean_match_similarity": torch.empty(0),
            "mean_match_margin": torch.empty(0),
            "condition_number": torch.empty(0),
            "minimum_ray_angle_degrees": torch.empty(0),
            "maximum_ray_angle_degrees": torch.empty(0),
            "mean_reprojection_error_px": torch.empty(0),
            "maximum_reprojection_error_px": torch.empty(0),
        }
    long_fields = {
        "current_sequence_index",
        "current_frame_index",
        "slot",
        "sam_track_id",
        "query_patch_flat_index",
        "num_views",
    }
    output = {}
    for field in rows[0]:
        values = [row[field] for row in rows]
        if field in {"point_world", "query_pixel_xy"}:
            output[field] = torch.stack([torch.as_tensor(value).float() for value in values])
        elif field in long_fields:
            output[field] = torch.tensor(values, dtype=torch.long)
        else:
            output[field] = torch.tensor(values, dtype=torch.float32)
    return output


def _patch_ownership(
    masks: torch.Tensor,
    scores: torch.Tensor,
    *,
    eligible_slots: tuple[int, ...],
) -> torch.Tensor:
    index = torch.tensor(eligible_slots, dtype=torch.long)
    value = masks.index_select(1, index)
    confidence = scores.detach().float().cpu().index_select(1, index)[:, :, None, None]
    weighted = torch.where(value, confidence.expand_as(value), torch.full_like(value, -torch.inf, dtype=torch.float32))
    owner_position = weighted.argmax(dim=1)
    owner_slots = torch.tensor(eligible_slots, dtype=torch.long).index_select(
        0, owner_position.reshape(-1)
    ).reshape_as(owner_position)
    return torch.where(value.any(dim=1), owner_slots, torch.full_like(owner_slots, -1))


def _camera_centers(world_to_camera: torch.Tensor) -> torch.Tensor:
    pose = _homogeneous(world_to_camera)
    centers = []
    for matrix in pose:
        rotation = _project_rotation(matrix[:3, :3])
        centers.append(-(rotation.T @ matrix[:3, 3]))
    return torch.stack(centers).float()


def _pose_sim3(source_pose: torch.Tensor, target_pose: torch.Tensor) -> dict[str, Any]:
    source = _camera_centers(source_pose)
    target = _camera_centers(target_pose)
    if source.shape != target.shape or source.shape[0] < 3:
        raise ValueError("Pose Sim(3) needs aligned sequences with at least 3 frames.")
    scale, rotation, translation, inliers, rmse = _robust_similarity(
        source,
        target,
        min_points=3,
        trim_fraction=0.8,
        iterations=4,
    )
    return {
        "scale": float(scale),
        "rotation": rotation.float(),
        "translation": translation.float(),
        "inliers": int(inliers),
        "camera_center_fit_rmse": float(rmse),
        "fit_source": "whole_trajectory_camera_centers_scoring_only",
    }


def _apply_sim3(point: torch.Tensor, alignment: dict[str, Any]) -> torch.Tensor:
    return (
        float(alignment["scale"])
        * (point.detach().float().cpu() @ alignment["rotation"].T)
        + alignment["translation"]
    )


def _alignment_for_json(alignment: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.tolist() if torch.is_tensor(value) else value
        for key, value in alignment.items()
    }


def _branch_metrics(
    branch: str,
    rows: list[dict[str, Any]],
    *,
    attempted_queries: int,
    candidate_valid_anchors: int,
) -> dict[str, Any]:
    paired = _paired_error_metrics(rows)
    reprojection = [float(row["mean_reprojection_error_px"]) for row in rows]
    return {
        "branch": branch,
        "attempted_query_count": attempted_queries,
        "candidate_valid_anchor_count": candidate_valid_anchors,
        "candidate_valid_anchor_rate": candidate_valid_anchors / max(attempted_queries, 1),
        "evaluated_anchor_count": len(rows),
        **paired,
        "mean_reprojection_error_px": _mean_or_nan(reprojection),
        "median_reprojection_error_px": _median_or_nan(reprojection),
    }


def _paired_error_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "tri_rmse": float("nan"),
            "tri_median": float("nan"),
            "tri_p90": float("nan"),
            "raw_rmse": float("nan"),
            "raw_median": float("nan"),
            "raw_p90": float("nan"),
            "tri_gain_vs_raw_percent": float("nan"),
            "improved_anchor_ratio": float("nan"),
            "mean_delta_raw_minus_tri": float("nan"),
            "median_delta_raw_minus_tri": float("nan"),
        }
    tri = torch.tensor([float(row["tri_gt_error"]) for row in rows])
    raw = torch.tensor([float(row["raw_gt_error"]) for row in rows])
    delta = raw - tri
    tri_rmse = float(torch.sqrt(tri.square().mean()))
    raw_rmse = float(torch.sqrt(raw.square().mean()))
    return {
        "tri_rmse": tri_rmse,
        "tri_median": float(tri.median()),
        "tri_p90": float(torch.quantile(tri, 0.90)),
        "raw_rmse": raw_rmse,
        "raw_median": float(raw.median()),
        "raw_p90": float(torch.quantile(raw, 0.90)),
        "tri_gain_vs_raw_percent": 100.0 * (raw_rmse - tri_rmse) / max(raw_rmse, 1e-12),
        "improved_anchor_ratio": float((tri < raw).float().mean()),
        "mean_delta_raw_minus_tri": float(delta.mean()),
        "median_delta_raw_minus_tri": float(delta.median()),
    }


def _branch_anchor_help(row: dict[str, Any]) -> bool:
    return bool(
        int(row["evaluated_anchor_count"]) > 0
        and math.isfinite(float(row["tri_rmse"]))
        and float(row["tri_rmse"]) < float(row["raw_rmse"])
        and float(row["improved_anchor_ratio"]) > 0.5
    )


def _instance_metrics(
    score_rows: list[dict[str, Any]], payload: dict[str, Any]
) -> list[dict[str, Any]]:
    output = []
    prompts = tuple(str(value) for value in payload["sam_track_prompts"])
    for branch in BRANCHES:
        slots = sorted(
            {int(row["slot"]) for row in score_rows if row["branch"] == branch}
        )
        for slot in slots:
            selected = [
                row
                for row in score_rows
                if row["branch"] == branch and int(row["slot"]) == slot
            ]
            output.append(
                {
                    "branch": branch,
                    "slot": slot,
                    "prompt": prompts[slot],
                    "anchor_count": len(selected),
                    **_paired_error_metrics(selected),
                }
            )
    return output


def _decode_pose(
    qk: dict[str, Any], payload: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor]:
    from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri

    pose, intrinsics = pose_encoding_to_extri_intri(
        qk["pose_encoding"].detach().float().cpu().unsqueeze(0),
        image_size_hw=tuple(int(value) for value in payload["image_size"]),
    )
    decoded = pose[0].detach().float().cpu()
    stored = qk["selected_world_to_camera"].detach().float().cpu()
    if stored.ndim == 4 and stored.shape[0] == 1:
        stored = stored[0]
    if stored.shape != decoded.shape or not torch.allclose(
        stored, decoded, rtol=1e-5, atol=1e-5
    ):
        raise ValueError("Frozen QK pose encoding/artifact mismatch.")
    return decoded, intrinsics[0].detach().float().cpu()


def _load_qk_rankings(
    path: Path, frames: list[int] | tuple[int, ...]
) -> tuple[tuple[int, ...], ...]:
    with path.open("r", encoding="utf8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(frames):
        raise ValueError("QK retrieval diagnostics row count differs from V0 frames.")
    output = []
    for sequence_index, (row, frame) in enumerate(zip(rows, frames)):
        if int(row["sequence_index"]) != sequence_index or int(row["frame_index"]) != int(frame):
            raise ValueError("QK retrieval diagnostics frame order mismatch.")
        ranked = tuple(
            int(value)
            for value in str(row["qk_ranked_sequence_indices"]).split()
        )
        if set(ranked) != set(range(sequence_index)):
            raise ValueError("QK ranking is not a permutation of causal history.")
        output.append(ranked)
    return tuple(output)


def _validate_inputs(
    payload: dict[str, Any],
    qk: dict[str, Any],
    clip: ClipConfig,
    ranking_path: Path,
) -> None:
    required = (
        "token_levels",
        "patch_start_idx",
        "patch_shape",
        "image_size",
        "scene_id",
        "frame_indices",
        "tracking_masks_stream",
        "tracking_scores",
        "sam_track_ids",
        "sam_track_prompts",
        "baseline_world_points",
        "baseline_world_confidence",
        "target_world_points",
        "target_world_to_camera",
    )
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError(f"V0 cache lacks T0 fields: {missing}.")
    frames = tuple(int(value) for value in payload["frame_indices"])
    if frames != clip.frame_indices or payload.get("clip_name") != clip.name:
        raise ValueError("V0 cache does not match the configured clip.")
    if str(payload.get("scene_id", "")) != clip.scene_id:
        raise ValueError("V0 cache scene does not match the configured clip.")
    if qk.get("selected_pose_branch") != "retrieve_qk":
        raise ValueError("T0 requires the frozen retrieve_qk pose artifact.")
    if tuple(int(value) for value in qk.get("frame_indices", ())) != frames:
        raise ValueError("QK artifact frame order differs from the V0 cache.")
    for name in ("pose_encoding", "selected_world_to_camera", "raw_world_to_camera"):
        if not torch.is_tensor(qk.get(name)):
            raise ValueError(f"QK artifact lacks tensor {name!r}.")
    if not ranking_path.is_file():
        raise FileNotFoundError(f"Missing QK retrieval diagnostics: {ranking_path}")


def _write_outputs(
    run: T0Run,
    *,
    summary: dict[str, Any],
    diagnostic_rows: list[dict[str, Any]],
    unit_rows: list[dict[str, Any]],
    score_rows: list[dict[str, Any]],
    instance_rows: list[dict[str, Any]],
) -> None:
    _write_json(run.output_dir / "summary.json", summary)
    _write_csv(run.output_dir / "branch_summary.csv", summary["branches"])
    _write_csv(run.output_dir / "candidate_diagnostics.csv", diagnostic_rows)
    _write_csv(run.output_dir / "track_unit_summary.csv", unit_rows)
    _write_csv(run.output_dir / "anchor_scores.csv", score_rows)
    _write_csv(run.output_dir / "instance_summary.csv", instance_rows)
    _write_copyable(run.output_dir / "copyable_result.txt", summary)


def _write_copyable(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "===== COPYABLE_T0_SAM_INDEXED_ANCHOR_PROBE_BEGIN =====",
        f"revision={summary['revision']}",
        f"clip={summary['clip']}",
        f"scene_id={summary['scene_id']}",
        f"frames={len(summary['frames'])}",
        "branches=" + ",".join(row["branch"] for row in summary["branches"]),
        f"descriptor_source={summary['descriptor']['source']}",
        f"descriptor_level_position={summary['descriptor']['level_position']}",
        f"descriptor_token_half={summary['descriptor']['token_half']}",
        f"qk_history_pool_size={summary['history']['maximum_covisible_frames']}",
        f"maximum_history_views_per_anchor={summary['history']['maximum_history_views_per_anchor']}",
        "model_loaded_or_run=0",
        "model_trained=0",
        "candidate_generation_raw_pointmap_fields=0",
        "candidate_generation_gt_fields=0",
        "formal_v0_modified=0",
        "",
        "branch,attempted_query_count,candidate_valid_anchor_count,candidate_valid_anchor_rate,evaluated_anchor_count,tri_rmse,raw_rmse,tri_gain_vs_raw_percent,tri_median,raw_median,tri_p90,raw_p90,improved_anchor_ratio,mean_delta_raw_minus_tri,mean_reprojection_error_px,equal_count,equal_count_tri_rmse",
    ]
    fields = (
        "branch",
        "attempted_query_count",
        "candidate_valid_anchor_count",
        "candidate_valid_anchor_rate",
        "evaluated_anchor_count",
        "tri_rmse",
        "raw_rmse",
        "tri_gain_vs_raw_percent",
        "tri_median",
        "raw_median",
        "tri_p90",
        "raw_p90",
        "improved_anchor_ratio",
        "mean_delta_raw_minus_tri",
        "mean_reprojection_error_px",
        "equal_count",
        "equal_count_tri_rmse",
    )
    for row in summary["branches"]:
        lines.append(",".join(str(row.get(field, "")) for field in fields))
    lines.extend(
        (
            "",
            "decision=" + json.dumps(summary["decision"], sort_keys=True),
            f"claim={summary['claim']}",
            "outputs:",
            f"summary={path.with_name('summary.json')}",
            f"branch_csv={path.with_name('branch_summary.csv')}",
            f"candidate_csv={path.with_name('candidate_diagnostics.csv')}",
            f"anchor_csv={path.with_name('anchor_scores.csv')}",
            f"instance_csv={path.with_name('instance_summary.csv')}",
            f"candidate_artifact={summary['candidate_artifact']}",
            "===== COPYABLE_T0_SAM_INDEXED_ANCHOR_PROBE_END =====",
        )
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def _load_run(path: str | Path) -> T0Run:
    source = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf8")) or {}
    matching = raw.get("matching", {})
    triangulation = raw.get("triangulation", {})
    controls = raw.get("controls", {})
    evaluation = raw.get("evaluation", {})
    run = T0Run(
        source_path=source,
        v0_config=_resolve(raw.get("v0_config", "streaming_couping/configs/v0_baseline.yaml")),
        output_dir=_resolve(raw.get("output_dir", "outputs/streaming_couping_t0_anchor_probe")),
        descriptor_level_position=int(matching.get("descriptor_level_position", 0)),
        descriptor_token_half=str(matching.get("descriptor_token_half", "frame")),
        track_score_threshold=float(matching.get("track_score_threshold", 0.50)),
        mask_erosion_pixels=int(matching.get("mask_erosion_pixels", 3)),
        minimum_patch_coverage=float(matching.get("minimum_patch_coverage", 0.25)),
        excluded_prompts=tuple(str(value).strip().lower() for value in matching.get("excluded_prompts", ("wardrobe",))),
        qk_history_pool_size=int(matching.get("qk_history_pool_size", 8)),
        maximum_queries_per_instance_frame=int(matching.get("maximum_queries_per_instance_frame", 64)),
        maximum_history_views=int(matching.get("maximum_history_views", 4)),
        minimum_match_similarity=float(matching.get("minimum_match_similarity", 0.60)),
        minimum_match_margin=float(matching.get("minimum_match_margin", 0.01)),
        minimum_views=int(triangulation.get("minimum_views", 2)),
        minimum_ray_angle_degrees=float(triangulation.get("minimum_ray_angle_degrees", 1.5)),
        maximum_condition_number=float(triangulation.get("maximum_condition_number", 10000.0)),
        maximum_mean_reprojection_patch_fraction=float(triangulation.get("maximum_mean_reprojection_patch_fraction", 0.75)),
        maximum_reprojection_patch_fraction=float(triangulation.get("maximum_reprojection_patch_fraction", 1.50)),
        shifted_mask_fraction_y=float(controls.get("shifted_mask_fraction_y", 0.25)),
        shifted_mask_fraction_x=float(controls.get("shifted_mask_fraction_x", 0.20)),
        minimum_equal_count=int(evaluation.get("minimum_equal_count", 32)),
    )
    _validate_run(run)
    return run


def _validate_run(run: T0Run) -> None:
    source = yaml.safe_load(run.v0_config.read_text(encoding="utf8")) or {}
    baseline = source.get("baseline", {})
    if str(baseline.get("version", "")).strip().lower() != "v0":
        raise ValueError("T0 must consume a baseline.version=v0 configuration.")
    if str(baseline.get("status", "")).strip().lower() != "frozen":
        raise ValueError("T0 must consume a frozen V0 configuration.")
    selected = str(
        baseline.get("pose", {}).get("selected_branch", "")
    ).strip().lower()
    if selected != "retrieve_qk":
        raise ValueError("T0 requires the frozen retrieve_qk V0 pose branch.")
    if run.descriptor_token_half not in {"frame", "global"}:
        raise ValueError("descriptor_token_half must be frame/global.")
    for value in (
        run.track_score_threshold,
        run.minimum_patch_coverage,
        run.minimum_match_similarity,
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError("Matching probabilities/coverage must be in [0,1].")
    if run.minimum_match_margin < 0.0:
        raise ValueError("minimum_match_margin cannot be negative.")
    if run.qk_history_pool_size < 1 or run.maximum_history_views < 1:
        raise ValueError("History/view budgets must be positive.")
    if run.maximum_queries_per_instance_frame < 1 or run.minimum_views < 2:
        raise ValueError("Query budget/minimum views are invalid.")
    if run.maximum_condition_number <= 1.0 or run.minimum_equal_count < 1:
        raise ValueError("Condition/equal-count gates are invalid.")
    if run.mask_erosion_pixels < 0 or run.minimum_ray_angle_degrees <= 0.0:
        raise ValueError("Mask erosion/ray-angle gates are invalid.")
    for value in (
        run.maximum_mean_reprojection_patch_fraction,
        run.maximum_reprojection_patch_fraction,
        run.shifted_mask_fraction_y,
        run.shifted_mask_fraction_x,
    ):
        if value <= 0.0:
            raise ValueError("Reprojection/shift fractions must be positive.")


def _qk_artifact_path(config: Path) -> Path:
    raw = yaml.safe_load(config.read_text(encoding="utf8")) or {}
    return _resolve(raw["baseline"]["pose"]["qk_pose_output"])


def _find_clip(clips: tuple[ClipConfig, ...], name: str) -> ClipConfig:
    selected = [clip for clip in clips if clip.name == name]
    if len(selected) != 1:
        raise ValueError(f"Expected exactly one clip {name!r}.")
    return selected[0]


def _homogeneous(value: torch.Tensor) -> torch.Tensor:
    poses = value.detach().float().cpu()
    if poses.ndim == 4 and poses.shape[0] == 1:
        poses = poses[0]
    if poses.ndim != 3 or tuple(poses.shape[-2:]) not in {(3, 4), (4, 4)}:
        raise ValueError("Pose sequence must have shape [S,3,4] or [S,4,4].")
    if tuple(poses.shape[-2:]) == (4, 4):
        return poses.clone()
    output = torch.eye(4).expand(poses.shape[0], 4, 4).clone()
    output[:, :3] = poses
    return output


def _project_rotation(rotation: torch.Tensor) -> torch.Tensor:
    left, _, right_t = torch.linalg.svd(rotation.float())
    projected = left @ right_t
    if torch.det(projected) < 0:
        left = left.clone()
        left[:, -1] *= -1
        projected = left @ right_t
    return projected


def _mean_or_nan(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _median_or_nan(values: list[float]) -> float:
    return float(torch.tensor(values).median()) if values else float("nan")


def _space(values) -> str:
    return " ".join(str(int(value)) for value in values)


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf8")
        return
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="streaming_couping/configs/t0_sam_indexed_anchor_probe.yaml",
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


if __name__ == "__main__":
    main()
