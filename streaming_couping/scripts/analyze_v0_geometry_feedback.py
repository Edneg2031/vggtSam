#!/usr/bin/env python3
"""Run a frozen V0 joint historical-depth and scale-shift diagnostic.

This command is intentionally diagnostic-only.  It reuses the completed V0
cache, does not run StreamVGGT or SAM, and never writes a corrected pointmap
into the deployed pipeline.  Candidate mask Vetos and causal depth pairs are
frozen before ground truth is opened.

The primary Veto reference is a causal historical same-slot point cloud
transformed into the current camera.  The current frame's depth is used only
as the value being tested.  A current-mask-depth reference is retained as an
explicit circularity control, not as a proposed deployment method.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

from streaming_couping.scripts.run_semantic_map_evaluation import (
    _build_oracle_variant,
    _load_run as _load_evaluation_run,
)
from streaming_couping.scripts.run_v0_bidirectional_feedback import (
    V0Arrays,
    _find_clip,
    _load_arrays,
    _resize_masks,
    _validate_v0_inputs,
)
from streaming_couping.src.learned_pose.baseline_runtime import (
    load_baseline_run_config,
)
from streaming_couping.src.learned_pose.cache import cache_path, load_feature_cache
from streaming_couping.src.learned_pose.config import load_learned_pose_config
from streaming_couping.src.robust_depth_feedback import (
    HistoricalDepthVetoConfig,
    affine_error_metrics,
    apply_depth_veto,
    build_causal_history_cache,
    fit_robust_affine,
    transform_world_points,
)
from streaming_couping.src.semantic_map_metrics import (
    apply_similarity,
    evaluate_semantic_object_map,
    load_ground_truth_stream_masks,
)
from streaming_couping.src.semantic_tracking_metrics import (
    GroundTruthInstances,
    evaluate_tracking_variants,
    load_ground_truth_instances,
)
from streaming_couping.src.storage import expand_storage_path
from streaming_couping.src.temporal_prompt_matrix import (
    project_world_points,
    sample_depth_nearest,
)


REVISION = "v0_geometry_feedback_joint_diagnostic_r1"
DEFAULT_CONFIG = "streaming_couping/configs/v0_baseline.yaml"
DEFAULT_OUTPUT = (
    "${VGGT_SAM_STORAGE_ROOT}/outputs/"
    "streaming_couping_v0_geometry_feedback"
)
EXPECTED_CLIP = "00a231a370_90_525_step15_37_68_54"
VETO_METHODS = (
    "current_depth_reference",
    "history_median_mad",
    "history_quantile_interval",
)


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    config_path = Path(args.config).expanduser().resolve()
    data = load_learned_pose_config(config_path)
    baseline = load_baseline_run_config(config_path)
    if baseline.version != "v0":
        raise ValueError("This diagnostic requires baseline.version=v0.")
    if len(data.clips) != 1:
        raise ValueError("This diagnostic is intentionally sealed to one V0 clip.")
    clip = _find_clip(data, baseline.clip_name)
    if clip.name != EXPECTED_CLIP:
        raise ValueError(
            "The retained diagnostic is sealed to clip "
            f"{EXPECTED_CLIP!r}; got {clip.name!r}."
        )

    output_dir = expand_storage_path(
        args.output_dir or DEFAULT_OUTPUT,
        base=config_path.parent,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_value = cache_path(data, clip)
    if not cache_value.is_file():
        raise FileNotFoundError(
            f"Missing frozen V0 cache {cache_value}; run commands_v0_baseline.txt first."
        )
    payload = load_feature_cache(cache_value)
    baseline_summary, poses = _validate_v0_inputs(
        payload=payload,
        summary_path=baseline.output_dir / "baseline_summary.json",
        poses_path=baseline.output_dir / "poses.pt",
        clip=clip,
    )
    arrays = _load_arrays(payload, poses)
    frames = tuple(int(value) for value in payload["frame_indices"])
    track_ids = tuple(int(value) for value in payload["sam_track_ids"])
    track_prompts = tuple(str(value) for value in payload["sam_track_prompts"])

    print("V0 JOINT HISTORICAL-DEPTH / SCALE-SHIFT DIAGNOSTIC")
    print(f"config={config_path}")
    print(f"cache={cache_value}")
    print(f"output={output_dir}")
    print(
        "scope=one sealed V0 scene; raw masks, world pointmap, depth, and pose "
        "are frozen"
    )
    print(
        "candidate_generation=causal historical same-slot geometry only; "
        "GT is opened after candidates and depth pairs are frozen"
    )
    print(
        f"frames={len(frames)} slots={len(track_ids)} map_size={arrays.map_size} "
        f"depth_size={tuple(arrays.depth.shape[-2:])} image_size={arrays.image_size}"
    )
    print(
        "veto=history median+MAD and q05/q95 intervals; current-depth reference "
        "is a circularity control; padding is in native StreamVGGT depth units"
    )
    print(
        f"history_min_points={args.min_history_points} "
        f"confidence_threshold={args.confidence_threshold:.2f} "
        f"track_score_threshold={args.track_score_threshold:.2f} "
        f"calibration_prefix={args.calibration_fraction:.2f}"
    )

    veto_config = HistoricalDepthVetoConfig(
        confidence_threshold=float(args.confidence_threshold),
        track_score_threshold=float(args.track_score_threshold),
        min_history_points=int(args.min_history_points),
        max_history_points=int(args.max_history_points),
        max_points_per_history_frame=int(args.max_points_per_history_frame),
        mad_multiplier=float(args.mad_multiplier),
        absolute_padding_m=float(args.absolute_padding_m),
        lower_quantile=float(args.lower_quantile),
        upper_quantile=float(args.upper_quantile),
    )
    pose_sources = _pose_sources(arrays, args.pose_source)
    variant_masks: dict[str, torch.Tensor] = {
        "raw_v0": arrays.raw_masks.detach().cpu().bool().clone()
    }
    variant_scores: dict[str, torch.Tensor] = {
        "raw_v0": arrays.scores.detach().cpu().float().clone()
    }
    veto_causal_rows: list[dict[str, object]] = []
    pair_summary_rows: list[dict[str, object]] = []
    pair_data: dict[str, dict[str, list[torch.Tensor]]] = {}
    depth_size = tuple(int(value) for value in arrays.depth.shape[-2:])
    raw_masks_depth = _resize_masks(arrays.raw_masks, depth_size)
    confidence_depth = _resize_scalar_map(arrays.confidence, depth_size)
    history_cache = build_causal_history_cache(
        points=arrays.points,
        confidence=arrays.confidence,
        raw_masks=arrays.raw_masks,
        scores=arrays.scores,
        config=veto_config,
    )

    # Candidate generation phase.  This block deliberately never reads a GT
    # field from the cache.  ``target_world_points`` is not touched here.
    for pose_name, world_to_camera in pose_sources.items():
        for method in VETO_METHODS:
            variant = f"{method}_{pose_name}"
            variant_masks[variant] = torch.zeros_like(arrays.raw_masks)
            variant_scores[variant] = arrays.scores.detach().cpu().float().clone()
            for frame in range(len(frames)):
                for slot in range(len(track_ids)):
                    raw_depth_mask = raw_masks_depth[frame, slot]
                    current_depth = arrays.depth[frame]
                    current_conf = confidence_depth[frame]
                    if method == "current_depth_reference":
                        reference = current_depth[raw_depth_mask]
                    else:
                        history_camera = transform_world_points(
                            history_cache[frame][slot],
                            world_to_camera[frame],
                        )
                        reference = history_camera[:, 2]
                    veto = apply_depth_veto(
                        raw_depth_mask,
                        current_depth,
                        reference,
                        reference_kind=method,
                        config=veto_config,
                        current_confidence=current_conf,
                    )
                    result_mask = _resize_masks(veto.mask, arrays.map_size)
                    variant_masks[variant][frame, slot] = result_mask
                    row = veto.to_dict()
                    row.update(
                        {
                            "sequence_index": int(frame),
                            "frame_index": int(frames[frame]),
                            "slot": int(slot),
                            "sam_track_id": int(track_ids[slot]),
                            "track_prompt": str(track_prompts[slot]),
                            "pose_source": str(pose_name),
                            "method": str(method),
                            "variant": variant,
                            "depth_height": int(depth_size[0]),
                            "depth_width": int(depth_size[1]),
                            "map_height": int(arrays.map_size[0]),
                            "map_width": int(arrays.map_size[1]),
                            "gt_fields": 0,
                        }
                    )
                    veto_causal_rows.append(row)

        pair_source, pair_target, pair_frames, pair_rows = _build_self_pairs(
            raw_masks=arrays.raw_masks,
            depth=arrays.depth,
            history_cache=history_cache,
            raw_world_to_camera=world_to_camera,
            intrinsics=arrays.intrinsics,
            image_size=arrays.image_size,
            depth_size=depth_size,
            max_pair_points=int(args.max_pair_points_per_query),
            frames=frames,
            track_ids=track_ids,
            pose_source=pose_name,
        )
        pair_data[pose_name] = {
            "source": pair_source,
            "target": pair_target,
            "frame": pair_frames,
        }
        pair_summary_rows.extend(pair_rows)

    _write_csv(output_dir / "depth_veto_causal.csv", veto_causal_rows)
    _write_csv(output_dir / "scale_shift_pairs_causal.csv", pair_summary_rows)
    _write_json(
        output_dir / "candidate_metadata.json",
        {
            "schema": 1,
            "revision": REVISION,
            "candidate_generation_gt_fields": 0,
            "pose_sources": list(pose_sources),
            "veto_methods": list(VETO_METHODS),
            "veto_config": _config_dict(veto_config),
            "pair_count": int(_pair_sample_count(pair_data)),
            "frame_count": len(frames),
            "slot_count": len(track_ids),
        },
    )
    print(
        f"candidates frozen veto_rows={len(veto_causal_rows)} "
        f"scale_pair_rows={len(pair_summary_rows)} "
        f"scale_pair_samples={_pair_sample_count(pair_data)}"
    )

    # No-GT scale consistency readout.  The prefix/suffix split is fixed before
    # the run and is never selected from the evaluation output.
    calibration_count = max(
        1,
        min(len(frames) - 1, int(round(len(frames) * float(args.calibration_fraction)))),
    ) if len(frames) > 1 else 1
    scale_rows = _fit_self_scale_rows(
        pair_data,
        calibration_count=calibration_count,
        min_samples=int(args.min_affine_samples),
        max_samples=int(args.max_affine_samples),
        trim_quantile=float(args.affine_trim_quantile),
        max_iterations=int(args.affine_max_iterations),
    )

    # ------------------------------
    # Sealed evaluation phase.  GT is first opened here.
    # ------------------------------
    print("candidate artifacts frozen; opening sealed GT evaluation")
    evaluation_run = _load_evaluation_run(config_path)
    ground_truth = load_ground_truth_instances(
        data.manifest,
        scene_id=str(payload["scene_id"]),
        frame_indices=frames,
        output_size=arrays.map_size,
        prompts=tuple(str(value) for value in payload["instance_prompts"]),
        prompt_label_aliases=evaluation_run.prompt_label_aliases,
    )
    gt_masks = load_ground_truth_stream_masks(
        data.manifest,
        scene_id=str(payload["scene_id"]),
        frame_indices=frames,
        instance_ids=ground_truth.instance_ids,
        processed_size=arrays.map_size,
        image_mode=str(payload["image_mode"]),
    )
    gt = GroundTruthInstances(
        masks=gt_masks,
        instance_ids=ground_truth.instance_ids,
        labels=ground_truth.labels,
        all_visible_instance_ids=ground_truth.all_visible_instance_ids,
    )
    # Establish the raw assignment once.  All branches, including the GT
    # oracle, are evaluated under this same assignment so a branch cannot
    # improve its score by rematching after GT is opened.
    assignment_pass = evaluate_tracking_variants(
        scene_id=str(payload["scene_id"]),
        clip_name=str(payload["clip_name"]),
        frame_indices=frames,
        variant_masks=variant_masks,
        variant_scores=variant_scores,
        raw_variant="raw_v0",
        track_ids=track_ids,
        track_prompts=track_prompts,
        ground_truth=gt,
        config=evaluation_run.tracking,
        prompt_label_aliases=evaluation_run.prompt_label_aliases,
    )
    oracle_masks, oracle_scores = _build_oracle_variant(
        gt.masks,
        assignments=assignment_pass["assignments"],
        sequence_shape=tuple(arrays.raw_masks.shape),
    )
    variant_masks["gt_mask_oracle"] = oracle_masks
    variant_scores["gt_mask_oracle"] = oracle_scores
    tracking = evaluate_tracking_variants(
        scene_id=str(payload["scene_id"]),
        clip_name=str(payload["clip_name"]),
        frame_indices=frames,
        variant_masks=variant_masks,
        variant_scores=variant_scores,
        raw_variant="raw_v0",
        track_ids=track_ids,
        track_prompts=track_prompts,
        ground_truth=gt,
        config=evaluation_run.tracking,
        prompt_label_aliases=evaluation_run.prompt_label_aliases,
    )

    if "target_depth" not in payload:
        raise ValueError(
            "The frozen V0 cache lacks target_depth; rebuild the cache before "
            "running the scale-shift diagnostic."
        )
    target_points = _tensor(payload["target_world_points"], "target_world_points").float()
    target_depth = _scalar_sequence_map(
        _tensor(payload["target_depth"], "target_depth"),
        "target_depth",
    )
    aligned_points = apply_similarity(
        arrays.points,
        scale=float(payload["point_alignment_scale"]),
        rotation=_tensor(payload["point_alignment_rotation"], "point_alignment_rotation"),
        translation=_tensor(
            payload["point_alignment_translation"],
            "point_alignment_translation",
        ),
    )
    map_results: list[dict[str, object]] = []
    map_object_rows: list[dict[str, object]] = []
    for variant, masks in variant_masks.items():
        result = evaluate_semantic_object_map(
            scene_id=str(payload["scene_id"]),
            clip_name=str(payload["clip_name"]),
            variant=variant,
            map_policy="all_visible_observations",
            aligned_world_points=aligned_points,
            target_world_points=target_points,
            confidence=arrays.confidence,
            predicted_masks=masks,
            track_scores=variant_scores[variant],
            gt_masks=gt.masks,
            gt_instance_ids=gt.instance_ids,
            gt_labels=gt.labels,
            assignments=tracking["assignments"],
            config=evaluation_run.map_metrics,
        )
        map_results.append(result["summary"])
        map_object_rows.extend(result["object_rows"])

    evaluated_veto_rows = _add_veto_gt_metrics(
        veto_causal_rows,
        variant_masks=variant_masks,
        ground_truth=gt,
        assignments=tracking["assignments"],
    )
    _write_csv(output_dir / "depth_veto_metrics.csv", evaluated_veto_rows)
    _write_csv(output_dir / "per_pixel_metrics.csv", evaluated_veto_rows)
    _write_csv(output_dir / "tracking_metrics.csv", tracking["summary_rows"])
    _write_csv(output_dir / "tracking_frame_metrics.csv", tracking["frame_rows"])
    _write_csv(output_dir / "tracking_object_metrics.csv", tracking["object_rows"])
    _write_csv(output_dir / "map_metrics.csv", map_results)
    _write_csv(output_dir / "map_object_metrics.csv", map_object_rows)
    _write_csv(
        output_dir / "per_object_metrics.csv",
        [
            {"metric_group": "map", **dict(row)}
            for row in map_object_rows
        ]
        + [
            {"metric_group": "tracking", **dict(row)}
            for row in tracking["object_rows"]
        ],
    )

    branch_rows = _build_branch_rows(
        tracking["summary_rows"],
        tracking["frame_rows"],
        map_results,
    )
    _write_csv(output_dir / "per_scene_metrics.csv", branch_rows)
    scale_rows.extend(
        _fit_gt_scale_rows(
            arrays=arrays,
            target_depth=target_depth,
            raw_world_to_camera=arrays.raw_world_to_camera,
            calibration_count=calibration_count,
            confidence_threshold=float(args.confidence_threshold),
            min_samples=int(args.min_affine_samples),
            max_samples=int(args.max_affine_samples),
            trim_quantile=float(args.affine_trim_quantile),
            max_iterations=int(args.affine_max_iterations),
        )
    )
    _write_csv(output_dir / "scale_shift_metrics.csv", scale_rows)

    summary = _build_summary(
        payload=payload,
        baseline_summary=baseline_summary,
        cache_value=cache_value,
        output_dir=output_dir,
        arrays=arrays,
        pose_sources=pose_sources,
        veto_config=veto_config,
        calibration_count=calibration_count,
        causal_rows=veto_causal_rows,
        evaluated_veto_rows=evaluated_veto_rows,
        scale_rows=scale_rows,
        branch_rows=branch_rows,
        tracking=tracking,
        map_results=map_results,
        args=args,
    )
    summary_path = output_dir / "summary.json"
    summary["outputs"] = {
        "summary": str(summary_path),
        "copyable_result": str(output_dir / "copyable_result.txt"),
        "candidate_metadata": str(output_dir / "candidate_metadata.json"),
        "depth_veto_causal": str(output_dir / "depth_veto_causal.csv"),
        "depth_veto_metrics": str(output_dir / "depth_veto_metrics.csv"),
        "per_pixel_metrics": str(output_dir / "per_pixel_metrics.csv"),
        "scale_shift_pairs_causal": str(output_dir / "scale_shift_pairs_causal.csv"),
        "scale_shift_metrics": str(output_dir / "scale_shift_metrics.csv"),
        "tracking_metrics": str(output_dir / "tracking_metrics.csv"),
        "tracking_frame_metrics": str(output_dir / "tracking_frame_metrics.csv"),
        "tracking_object_metrics": str(output_dir / "tracking_object_metrics.csv"),
        "map_metrics": str(output_dir / "map_metrics.csv"),
        "map_object_metrics": str(output_dir / "map_object_metrics.csv"),
        "per_object_metrics": str(output_dir / "per_object_metrics.csv"),
        "per_scene_metrics": str(output_dir / "per_scene_metrics.csv"),
    }
    _write_json(summary_path, summary)
    copyable_path = output_dir / "copyable_result.txt"
    _write_copyable(copyable_path, summary)
    _print_summary(summary, branch_rows, scale_rows, evaluated_veto_rows)
    print(f"summary={summary_path}")
    print(f"copyable_result={copyable_path}")
    print("decision=DIAGNOSTIC_ONLY")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--pose-source",
        choices=("raw_streamvggt", "selected_v0", "both"),
        default="both",
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.30)
    parser.add_argument("--track-score-threshold", type=float, default=0.50)
    parser.add_argument("--min-history-points", type=int, default=16)
    parser.add_argument("--max-history-points", type=int, default=4096)
    parser.add_argument("--max-points-per-history-frame", type=int, default=512)
    parser.add_argument("--mad-multiplier", type=float, default=3.0)
    parser.add_argument(
        "--absolute-padding-m",
        type=float,
        default=0.05,
        help=(
            "Absolute interval padding in native StreamVGGT depth units. "
            "The _m name is retained for command compatibility; no GT Sim(3) "
            "scale is used during candidate generation."
        ),
    )
    parser.add_argument("--lower-quantile", type=float, default=0.05)
    parser.add_argument("--upper-quantile", type=float, default=0.95)
    parser.add_argument("--max-pair-points-per-query", type=int, default=2048)
    parser.add_argument("--min-affine-samples", type=int, default=64)
    parser.add_argument("--max-affine-samples", type=int, default=200000)
    parser.add_argument("--affine-trim-quantile", type=float, default=0.90)
    parser.add_argument("--affine-max-iterations", type=int, default=4)
    parser.add_argument("--calibration-fraction", type=float, default=0.70)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if not 0.0 <= float(args.confidence_threshold) <= 1.0:
        raise ValueError("--confidence-threshold must be in [0,1].")
    if not 0.0 <= float(args.track_score_threshold) <= 1.0:
        raise ValueError("--track-score-threshold must be in [0,1].")
    if int(args.min_history_points) < 1:
        raise ValueError("--min-history-points must be positive.")
    if int(args.max_history_points) < int(args.min_history_points):
        raise ValueError("--max-history-points must be >= min-history-points.")
    if int(args.max_points_per_history_frame) < 1:
        raise ValueError("--max-points-per-history-frame must be positive.")
    if int(args.max_pair_points_per_query) < 1:
        raise ValueError("--max-pair-points-per-query must be positive.")
    if int(args.min_affine_samples) < 2:
        raise ValueError("--min-affine-samples must be >= 2.")
    if int(args.max_affine_samples) < int(args.min_affine_samples):
        raise ValueError("--max-affine-samples must be >= min-affine-samples.")
    if not 0.5 <= float(args.affine_trim_quantile) < 1.0:
        raise ValueError("--affine-trim-quantile must be in [0.5,1).")
    if int(args.affine_max_iterations) < 1:
        raise ValueError("--affine-max-iterations must be positive.")
    if not 0.0 < float(args.calibration_fraction) < 1.0:
        raise ValueError("--calibration-fraction must be in (0,1).")


def _pose_sources(arrays: V0Arrays, selection: str) -> dict[str, torch.Tensor]:
    if selection == "raw_streamvggt":
        return {"raw_streamvggt": arrays.raw_world_to_camera}
    if selection == "selected_v0":
        return {"selected_v0": arrays.selected_world_to_camera}
    return {
        "raw_streamvggt": arrays.raw_world_to_camera,
        "selected_v0": arrays.selected_world_to_camera,
    }


def _build_self_pairs(
    *,
    raw_masks: torch.Tensor,
    depth: torch.Tensor,
    history_cache: Sequence[Sequence[torch.Tensor]],
    raw_world_to_camera: torch.Tensor,
    intrinsics: torch.Tensor,
    image_size: tuple[int, int],
    depth_size: tuple[int, int],
    max_pair_points: int,
    frames: Sequence[int],
    track_ids: Sequence[int],
    pose_source: str,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[dict[str, object]]]:
    """Build no-GT pairs ``current_depth ~= projected_history_depth``."""

    raw_depth_masks = _resize_masks(raw_masks, depth_size)
    sources: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    frame_values: list[torch.Tensor] = []
    rows: list[dict[str, object]] = []
    sequence = int(raw_masks.shape[0])
    if len(history_cache) != sequence or any(
        len(row) != int(raw_masks.shape[1]) for row in history_cache
    ):
        raise ValueError("history_cache does not match the raw mask sequence.")
    for frame in range(sequence):
        for slot in range(raw_masks.shape[1]):
            history = history_cache[frame][slot]
            source = target = torch.empty(0, dtype=torch.float32)
            if history.numel():
                projected = project_world_points(
                    history,
                    raw_world_to_camera[frame],
                    intrinsics[frame],
                    image_size,
                )
                depth_uv = _resize_uv(projected.uv, image_size, depth_size)
                sampled, sampled_valid = sample_depth_nearest(
                    depth[frame],
                    depth_uv,
                )
                mask_hit, mask_valid = _sample_bool_mask(
                    raw_depth_masks[frame, slot],
                    depth_uv,
                )
                valid = projected.valid_mask & sampled_valid & mask_valid & mask_hit
                valid &= torch.isfinite(projected.depth) & (projected.depth > 0)
                if bool(valid.any()):
                    source = sampled[valid].float().cpu()
                    target = projected.depth[valid].float().cpu()
                    if source.numel() > int(max_pair_points):
                        positions = _deterministic_positions(
                            source.numel(), int(max_pair_points)
                        )
                        source = source.index_select(0, positions)
                        target = target.index_select(0, positions)
            if source.numel():
                sources.append(source)
                targets.append(target)
                frame_values.append(
                    torch.full((source.numel(),), frame, dtype=torch.long)
                )
            rows.append(
                {
                    "metric_group": "causal_self_consistency",
                    "pose_source": str(pose_source),
                    "sequence_index": int(frame),
                    "frame_index": int(frames[frame]),
                    "slot": int(slot),
                    "sam_track_id": int(track_ids[slot]),
                    "history_point_count": int(history.shape[0]),
                    "pair_count": int(source.numel()),
                    "current_depth_mean": _mean_tensor(source),
                    "projected_history_depth_mean": _mean_tensor(target),
                    "gt_fields": 0,
                }
            )
    return sources, targets, frame_values, rows


def _fit_self_scale_rows(
    pair_data: Mapping[str, Mapping[str, list[torch.Tensor]]],
    *,
    calibration_count: int,
    min_samples: int,
    max_samples: int,
    trim_quantile: float,
    max_iterations: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for pose_source, data in pair_data.items():
        if not data["source"]:
            rows.append(
                {
                    "metric_group": "self_consistency",
                    "source_kind": "causal_history_depth",
                    "pose_source": str(pose_source),
                    "split": "calibration_and_holdout",
                    "status": "no_pairs",
                }
            )
            continue
        source = torch.cat(data["source"]).float()
        target = torch.cat(data["target"]).float()
        frame = torch.cat(data["frame"]).long()
        calibration = frame < int(calibration_count)
        fit = fit_robust_affine(
            source[calibration],
            target[calibration],
            min_samples=int(min_samples),
            max_samples=int(max_samples),
            trim_quantile=float(trim_quantile),
            max_iterations=int(max_iterations),
        )
        row = {
            "metric_group": "self_consistency",
            "source_kind": "causal_history_depth",
            "pose_source": str(pose_source),
            "split": "calibration_prefix",
            "calibration_frame_count": int(calibration_count),
            "holdout_frame_count": int(frame.max().item() + 1 - calibration_count)
            if frame.numel()
            else 0,
            **fit.to_dict(),
            "gt_fields": 0,
        }
        rows.append(row)
        if bool((~calibration).any()) and fit.accepted:
            before = affine_error_metrics(
                source[~calibration],
                target[~calibration],
                scale=1.0,
                shift=0.0,
            )
            holdout = affine_error_metrics(
                source[~calibration],
                target[~calibration],
                scale=fit.scale,
                shift=fit.shift,
            )
            rows.append(
                {
                    "metric_group": "self_consistency",
                    "source_kind": "causal_history_depth",
                    "pose_source": str(pose_source),
                    "split": "holdout_suffix",
                    "calibration_frame_count": int(calibration_count),
                    "holdout_frame_count": int(frame[~calibration].max().item() + 1 - calibration_count),
                    "fit_scale": float(fit.scale),
                    "fit_shift": float(fit.shift),
                    "sample_count": int(holdout["count"]),
                    "rmse_before": float(before["rmse"]),
                    "rmse_after": float(holdout["rmse"]),
                    "median_before": float(before["median"]),
                    "median_after": float(holdout["median"]),
                    "p90_before": float(before["p90"]),
                    "p90_after": float(holdout["p90"]),
                    "holdout_rmse": float(holdout["rmse"]),
                    "holdout_median": float(holdout["median"]),
                    "holdout_p90": float(holdout["p90"]),
                    "accepted": 1,
                    "status": "evaluated",
                    "gt_fields": 0,
                }
            )
    return rows


def _fit_gt_scale_rows(
    *,
    arrays: V0Arrays,
    target_depth: torch.Tensor,
    raw_world_to_camera: torch.Tensor,
    calibration_count: int,
    confidence_threshold: float,
    min_samples: int,
    max_samples: int,
    trim_quantile: float,
    max_iterations: int,
) -> list[dict[str, object]]:
    """Measure affine depth mismatch against the same-pixel GT depth.

    ``target_depth`` is produced from the sealed GT world pointmap and GT
    camera pose.  It is deliberately read only in the evaluation phase.  The
    predicted pointmap is transformed with the frozen native StreamVGGT pose,
    so its Z values remain in the native prediction gauge and the fitted
    affine parameters describe ``metric_depth ~= a * native_depth + b``.

    Using the GT depth at the same image/pointmap pixel is important here.  A
    GT point transformed by the *predicted* pose would mix depth-scale error
    with camera-pose error and would not be a valid pixelwise depth target.
    """

    pred_camera = transform_world_points(arrays.points, raw_world_to_camera)
    confidence = arrays.confidence.float().cpu()
    target = _resize_scalar_map(
        target_depth,
        tuple(int(value) for value in arrays.points.shape[1:3]),
    )
    if target.shape[0] != pred_camera.shape[0]:
        raise ValueError("target_depth and predicted pointmap disagree on S.")
    frame_ids = torch.arange(
        pred_camera.shape[0], dtype=torch.long
    ).reshape(-1, 1, 1).expand(pred_camera.shape[:3])
    common_valid = (
        torch.isfinite(target)
        & (target > 1e-6)
        & torch.isfinite(confidence)
        & (confidence >= float(confidence_threshold))
    )
    pointmap_source = pred_camera[..., 2]
    depth_head_source = _resize_scalar_map(
        arrays.depth.float().cpu(),
        tuple(int(value) for value in arrays.points.shape[1:3]),
    )

    rows: list[dict[str, object]] = []
    for source_kind, source_map in (
        ("gt_oracle_pointmap", pointmap_source),
        ("gt_oracle_baseline_depth", depth_head_source),
    ):
        valid = (
            common_valid
            & torch.isfinite(source_map)
            & (source_map > 1e-6)
        )
        source = source_map[valid].float()
        target_values = target[valid].float()
        frame = frame_ids[valid].long()
        calibration = frame < int(calibration_count)
        fit = fit_robust_affine(
            source[calibration],
            target_values[calibration],
            min_samples=int(min_samples),
            max_samples=int(max_samples),
            trim_quantile=float(trim_quantile),
            max_iterations=int(max_iterations),
        )
        rows.append(
            {
                "metric_group": "gt_oracle_scale_shift",
                "source_kind": source_kind,
                "pose_source": "raw_streamvggt",
                "target_source": "cached_gt_target_depth_same_pixel",
                "split": "calibration_prefix",
                "calibration_frame_count": int(calibration_count),
                "holdout_frame_count": int(frame.max().item() + 1 - calibration_count)
                if frame.numel()
                else 0,
                **fit.to_dict(),
                "gt_fields": 1,
            }
        )
        if bool((~calibration).any()) and fit.accepted:
            before = affine_error_metrics(
                source[~calibration],
                target_values[~calibration],
                scale=1.0,
                shift=0.0,
            )
            holdout = affine_error_metrics(
                source[~calibration],
                target_values[~calibration],
                scale=fit.scale,
                shift=fit.shift,
            )
            rows.append(
                {
                    "metric_group": "gt_oracle_scale_shift",
                    "source_kind": source_kind,
                    "pose_source": "raw_streamvggt",
                    "target_source": "cached_gt_target_depth_same_pixel",
                    "split": "holdout_suffix",
                    "calibration_frame_count": int(calibration_count),
                    "holdout_frame_count": int(frame[~calibration].max().item() + 1 - calibration_count),
                    "fit_scale": float(fit.scale),
                    "fit_shift": float(fit.shift),
                    "sample_count": int(holdout["count"]),
                    "rmse_before": float(before["rmse"]),
                    "rmse_after": float(holdout["rmse"]),
                    "median_before": float(before["median"]),
                    "median_after": float(holdout["median"]),
                    "p90_before": float(before["p90"]),
                    "p90_after": float(holdout["p90"]),
                    "holdout_rmse": float(holdout["rmse"]),
                    "holdout_median": float(holdout["median"]),
                    "holdout_p90": float(holdout["p90"]),
                    "accepted": 1,
                    "status": "evaluated",
                    "gt_fields": 1,
                }
            )
    return rows


def _add_veto_gt_metrics(
    rows: Sequence[Mapping[str, object]],
    *,
    variant_masks: Mapping[str, torch.Tensor],
    ground_truth: GroundTruthInstances,
    assignments: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    slot_to_target = {
        int(row["slot"]): int(row["gt_index"])
        for row in assignments
    }
    output: list[dict[str, object]] = []
    gt_masks = ground_truth.masks.detach().cpu().bool()
    for source in rows:
        row = dict(source)
        if "method" not in row:
            output.append(row)
            continue
        variant = str(row["variant"])
        frame = int(row["sequence_index"])
        slot = int(row["slot"])
        raw = variant_masks["raw_v0"][frame, slot]
        refined = variant_masks[variant][frame, slot]
        removed = raw & ~refined
        target_index = slot_to_target.get(slot, -1)
        target = (
            gt_masks[frame, target_index]
            if target_index >= 0
            else torch.zeros_like(raw)
        )
        raw_intersection = int((raw & target).sum())
        raw_union = int((raw | target).sum())
        refined_intersection = int((refined & target).sum())
        refined_union = int((refined | target).sum())
        removed_foreground = int((removed & target).sum())
        removed_background = int((removed & ~target).sum())
        row.update(
            {
                "metric_level": "frame_slot_pixel_aggregate",
                "gt_fields": 1,
                "gt_assignment_available": int(target_index >= 0),
                "gt_instance_id": (
                    int(ground_truth.instance_ids[target_index])
                    if target_index >= 0
                    else -1
                ),
                "removed_foreground_pixels": removed_foreground,
                "removed_background_pixels": removed_background,
                "removed_foreground_fraction": (
                    float(removed_foreground) / float(removed.sum())
                    if int(removed.sum())
                    else 0.0
                ),
                "removed_background_fraction": (
                    float(removed_background) / float(removed.sum())
                    if int(removed.sum())
                    else 0.0
                ),
                "raw_mask_iou": (
                    float(raw_intersection) / float(raw_union)
                    if raw_union
                    else 0.0
                ),
                "veto_mask_iou": (
                    float(refined_intersection) / float(refined_union)
                    if refined_union
                    else 0.0
                ),
                "raw_mask_pixels": int(raw.sum()),
                "veto_mask_pixels": int(refined.sum()),
                "removed_pixels_evaluated": int(removed.sum()),
            }
        )
        output.append(row)
    return output


def _build_branch_rows(
    tracking_rows: Sequence[Mapping[str, object]],
    frame_rows: Sequence[Mapping[str, object]],
    map_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    tracking = {str(row["variant"]): dict(row) for row in tracking_rows}
    maps = {str(row["variant"]): dict(row) for row in map_rows}
    variants = tuple(tracking)
    output: list[dict[str, object]] = []
    for variant in variants:
        row: dict[str, object] = {
            "variant": variant,
            "metric_group": "branch_summary",
        }
        row.update(tracking.get(variant, {}))
        row.update(maps.get(variant, {}))
        safety = _worsening_summary(frame_rows, variant)
        row.update({f"safety_{key}": value for key, value in safety.items()})
        output.append(row)
    return output


def _worsening_summary(
    frame_rows: Sequence[Mapping[str, object]],
    variant: str,
    *,
    margin: float = 0.05,
) -> dict[str, object]:
    eligible = [
        row
        for row in frame_rows
        if str(row.get("variant", "")) == str(variant)
        and int(row.get("slot", -1)) >= 0
        and int(row.get("gt_visible", 0)) == 1
        and _finite(row.get("iou"))
        and _finite(row.get("raw_iou"))
    ]
    worse = [
        row
        for row in eligible
        if float(row["iou"]) < float(row["raw_iou"]) - float(margin)
    ]
    raw_correct = [row for row in eligible if float(row["raw_iou"]) >= 0.50]
    raw_correct_worse = [
        row
        for row in raw_correct
        if float(row["iou"]) < float(row["raw_iou"]) - float(margin)
    ]
    scene_values: dict[int, list[Mapping[str, object]]] = {}
    for row in eligible:
        scene_values.setdefault(int(row["sequence_index"]), []).append(row)
    scene_deltas = []
    for values in scene_values.values():
        raw_mean = sum(float(row["raw_iou"]) for row in values) / len(values)
        current_mean = sum(float(row["iou"]) for row in values) / len(values)
        scene_deltas.append(current_mean - raw_mean)
    scene_worse = sum(value < -float(margin) for value in scene_deltas)
    return {
        "worsened_frame_ratio": _ratio(scene_worse, len(scene_deltas)),
        "worsened_frame_count": int(scene_worse),
        "scene_frame_count": int(len(scene_deltas)),
        "worsened_raw_correct_ratio": _ratio(len(raw_correct_worse), len(raw_correct)),
        "worsened_raw_correct_count": int(len(raw_correct_worse)),
        "raw_correct_object_frame_count": int(len(raw_correct)),
        "worsened_object_frame_ratio": _ratio(len(worse), len(eligible)),
        "worsened_object_frame_count": int(len(worse)),
        "object_frame_count": int(len(eligible)),
    }


def _build_summary(
    *,
    payload: Mapping[str, object],
    baseline_summary: Mapping[str, object],
    cache_value: Path,
    output_dir: Path,
    arrays: V0Arrays,
    pose_sources: Mapping[str, torch.Tensor],
    veto_config: HistoricalDepthVetoConfig,
    calibration_count: int,
    causal_rows: Sequence[Mapping[str, object]],
    evaluated_veto_rows: Sequence[Mapping[str, object]],
    scale_rows: Sequence[Mapping[str, object]],
    branch_rows: Sequence[Mapping[str, object]],
    tracking: Mapping[str, object],
    map_results: Sequence[Mapping[str, object]],
    args: argparse.Namespace,
) -> dict[str, object]:
    primary = {
        str(row["variant"]): dict(row)
        for row in branch_rows
        if str(row.get("variant", "")).endswith("_raw_streamvggt")
        or str(row.get("variant", "")) == "raw_v0"
    }
    removal_count = sum(
        int(row.get("removed_pixels_evaluated", 0))
        for row in evaluated_veto_rows
        if "method" in row
    )
    removal_fg = sum(
        int(row.get("removed_foreground_pixels", 0))
        for row in evaluated_veto_rows
        if "method" in row
    )
    return {
        "schema": 1,
        "revision": REVISION,
        "decision": "DIAGNOSTIC_ONLY",
        "diagnostic_only": 1,
        "clip": str(payload["clip_name"]),
        "scene_id": str(payload["scene_id"]),
        "cache": str(cache_value),
        "output_dir": str(output_dir),
        "baseline_revision": str(baseline_summary.get("implementation_revision", "")),
        "candidate_generation_gt_fields": 0,
        "evaluation_gt_fields": 1,
        "models_rerun": 0,
        "streamvggt_rerun": 0,
        "sam_rerun": 0,
        "pose_modified": 0,
        "pointmap_modified": 0,
        "pose_sources": list(pose_sources),
        "parameters": {
            "confidence_threshold": float(args.confidence_threshold),
            "track_score_threshold": float(args.track_score_threshold),
            "calibration_fraction": float(args.calibration_fraction),
            "calibration_frame_count": int(calibration_count),
            "min_affine_samples": int(args.min_affine_samples),
            "veto_config": _config_dict(veto_config),
        },
        "candidate_counts": {
            "causal_rows": int(len(causal_rows)),
            "evaluated_veto_rows": int(len(evaluated_veto_rows)),
            "removed_pixels": int(removal_count),
            "removed_foreground_pixels": int(removal_fg),
            "removed_foreground_fraction": _ratio(removal_fg, removal_count),
        },
        "branch_summary": [dict(row) for row in branch_rows],
        "primary_raw_pose_summary": primary,
        "scale_shift_summary": [dict(row) for row in scale_rows],
        "tracking_summary": [dict(row) for row in tracking["summary_rows"]],
        "map_summary": [dict(row) for row in map_results],
        "v0_shapes": {
            "world_points": list(arrays.points.shape),
            "raw_masks": list(arrays.raw_masks.shape),
            "depth": list(arrays.depth.shape),
            "image_size": list(arrays.image_size),
            "map_size": list(arrays.map_size),
        },
        "interpretation": {
            "history_reference": (
                "historical same-slot world points transformed with the selected "
                "frozen pose; current depth is only the tested value"
            ),
            "current_depth_control": (
                "current-mask depth reference is circularity control and is not a "
                "deployment proposal"
            ),
            "scale_shift": (
                "GT affine rows are evaluation-only upper bounds; no affine "
                "correction is written to the pointmap"
            ),
            "scope_caveat": (
                "single frozen V0 scene; a positive diagnostic requires a new "
                "scene-disjoint validation run before implementation"
            ),
        },
        "next_gate": (
            "A Veto is promotable only if both map metrics improve, completeness "
            "does not fall by more than 5 percent, ghost does not rise by more "
            "than 5 percentage points, and both worsening ratios stay <= 15 percent."
        ),
        "outputs": {},
    }


def _write_copyable(path: Path, summary: Mapping[str, object]) -> None:
    lines = [
        "===== V0_GEOMETRY_FEEDBACK_JOINT_BEGIN =====",
        f"revision={summary['revision']}",
        f"decision={summary['decision']}",
        f"clip={summary['clip']}",
        f"scene_id={summary['scene_id']}",
        "candidate_generation_gt_fields=0",
        "evaluation_gt_fields=1",
        "streamvggt_rerun=0",
        "sam_rerun=0",
        "pose_modified=0",
        "pointmap_modified=0",
        f"candidate_counts={json.dumps(summary['candidate_counts'], sort_keys=True, allow_nan=True)}",
        "",
        "branch_summary_json="
        + json.dumps(summary["branch_summary"], sort_keys=True, allow_nan=True, default=str),
        "",
        "scale_shift_summary_json="
        + json.dumps(summary["scale_shift_summary"], sort_keys=True, allow_nan=True, default=str),
        "",
        "interpretation="
        + json.dumps(summary["interpretation"], sort_keys=True, allow_nan=True),
        "===== V0_GEOMETRY_FEEDBACK_JOINT_END =====",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def _print_summary(
    summary: Mapping[str, object],
    branch_rows: Sequence[Mapping[str, object]],
    scale_rows: Sequence[Mapping[str, object]],
    veto_rows: Sequence[Mapping[str, object]],
) -> None:
    print("===== V0 JOINT GEOMETRY FEEDBACK SUMMARY =====")
    print(
        "columns=variant mean_frame_iou frame_IDF1 pixel_IDF1 "
        "map_voxelIoU5cm map_F5cm completeness_m ghost safety_worsened_frame_ratio "
        "safety_worsened_raw_correct_ratio"
    )
    for row in branch_rows:
        print(
            f"  variant={row.get('variant')} "
            f"mean_frame_iou={_display(row.get('mean_frame_iou'))} "
            f"frame_IDF1={_display(row.get('frame_idf1'))} "
            f"pixel_IDF1={_display(row.get('pixel_idf1'))} "
            f"map_voxelIoU5cm={_display(row.get('voxel_iou_5cm'))} "
            f"map_F5cm={_display(row.get('fscore_5cm'))} "
            f"completeness_m={_display(row.get('object_completeness_m'))} "
            f"ghost={_display(row.get('ghost_point_ratio'))} "
            f"safety_worsened_frame_ratio={_display(row.get('safety_worsened_frame_ratio'))} "
            f"safety_worsened_raw_correct_ratio={_display(row.get('safety_worsened_raw_correct_ratio'))}"
        )
    print("depth_veto_reference_summary")
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for row in veto_rows:
        if "method" in row:
            grouped.setdefault(
                (str(row.get("pose_source")), str(row.get("method"))),
                [],
            ).append(row)
    for (pose, method), rows in sorted(grouped.items()):
        input_pixels = sum(int(row.get("input_mask_pixels", 0)) for row in rows)
        removed = sum(int(row.get("removed_pixels", 0)) for row in rows)
        removed_fg = sum(int(row.get("removed_foreground_pixels", 0)) for row in rows)
        print(
            f"  pose={pose} method={method} rows={len(rows)} "
            f"removed_ratio={_display(_ratio(removed, input_pixels))} "
            f"removed_foreground_fraction={_display(_ratio(removed_fg, removed))}"
        )
    print("scale_shift")
    for row in scale_rows:
        print(
            f"  group={row.get('metric_group')} source={row.get('source_kind')} "
            f"pose={row.get('pose_source')} split={row.get('split')} "
            f"status={row.get('status', '')} scale={_display(row.get('scale', row.get('fit_scale')))} "
            f"shift={_display(row.get('shift', row.get('fit_shift')))} "
            f"rmse_before={_display(row.get('rmse_before'))} "
            f"rmse_after={_display(row.get('rmse_after'))} "
            f"holdout_rmse={_display(row.get('holdout_rmse'))}"
        )
    print(
        "gate=diagnostic_only; no Veto or affine correction is promoted automatically"
    )


def _pair_sample_count(
    pair_data: Mapping[str, Mapping[str, list[torch.Tensor]]],
) -> int:
    return int(
        sum(
            int(tensor.numel())
            for data in pair_data.values()
            for tensor in data.get("source", [])
        )
    )


def _config_dict(config: HistoricalDepthVetoConfig) -> dict[str, object]:
    return {
        "confidence_threshold": float(config.confidence_threshold),
        "track_score_threshold": float(config.track_score_threshold),
        "min_history_points": int(config.min_history_points),
        "max_history_points": int(config.max_history_points),
        "max_points_per_history_frame": int(config.max_points_per_history_frame),
        "mad_multiplier": float(config.mad_multiplier),
        "absolute_padding_m": float(config.absolute_padding_m),
        "absolute_padding_units": "native_streamvggt_depth",
        "lower_quantile": float(config.lower_quantile),
        "upper_quantile": float(config.upper_quantile),
    }


def _resize_scalar_map(value: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    tensor = value.detach().cpu().float()
    if tensor.ndim != 3:
        raise ValueError("Expected a scalar sequence map [S,H,W].")
    if tuple(tensor.shape[-2:]) == tuple(size):
        return tensor
    return F.interpolate(
        tensor[:, None],
        size=tuple(int(item) for item in size),
        mode="bilinear",
        align_corners=False,
    )[:, 0]


def _scalar_sequence_map(value: torch.Tensor, name: str) -> torch.Tensor:
    """Normalize cached scalar maps to ``[S,H,W]`` without resizing."""

    tensor = value.detach().cpu().float()
    if tensor.ndim == 4 and tensor.shape[-1] == 1:
        tensor = tensor[..., 0]
    elif tensor.ndim == 4 and tensor.shape[1] == 1:
        tensor = tensor[:, 0]
    if tensor.ndim != 3:
        raise ValueError(
            f"{name} must have shape [S,H,W] or a singleton-channel map, "
            f"got {tuple(tensor.shape)}."
        )
    return tensor


def _resize_uv(
    uv: torch.Tensor,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> torch.Tensor:
    source_h, source_w = (int(value) for value in source_size)
    target_h, target_w = (int(value) for value in target_size)
    del source_h, target_h
    output = uv.detach().float().clone()
    output[:, 0] = (output[:, 0] + 0.5) * target_w / float(source_w) - 0.5
    output[:, 1] = (output[:, 1] + 0.5) * target_h / float(source_size[0]) - 0.5
    return output


def _sample_bool_mask(
    mask: torch.Tensor,
    uv: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = mask.detach().cpu().bool()
    uv = uv.detach().float().cpu()
    height, width = mask.shape
    finite = torch.isfinite(uv).all(dim=1)
    x = torch.round(uv[:, 0]).long()
    y = torch.round(uv[:, 1]).long()
    valid = finite & (x >= 0) & (x < width) & (y >= 0) & (y < height)
    safe_x = x.clamp(0, width - 1)
    safe_y = y.clamp(0, height - 1)
    return mask[safe_y, safe_x], valid


def _deterministic_positions(count: int, limit: int) -> torch.Tensor:
    if count <= int(limit):
        return torch.arange(count, dtype=torch.long)
    return torch.linspace(
        0,
        count - 1,
        steps=int(limit),
        dtype=torch.float64,
    ).round().long()


def _tensor(value: object, name: str) -> torch.Tensor:
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    if not value.is_floating_point():
        value = value.float()
    return value.detach().cpu()


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=True, default=str)
        + "\n",
        encoding="utf8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            key = str(key)
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fields})


def _csv_value(value: object) -> object:
    if torch.is_tensor(value):
        return value.item() if value.ndim == 0 else str(value.tolist())
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True, allow_nan=True, default=str)
    return value


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _mean_tensor(value: torch.Tensor) -> float:
    return float(value.float().mean()) if value.numel() else float("nan")


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if int(denominator) else 0.0


def _display(value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "nan"
    return f"{numeric:.6f}" if math.isfinite(numeric) else "nan"


if __name__ == "__main__":
    main()
