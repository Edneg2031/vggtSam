#!/usr/bin/env python3
"""Run a frozen, single-scene V0 diagnostic for explicit feedback loops.

This command deliberately stops short of a closed-loop model implementation.
It reuses the completed V0 cache and evaluates two actionable diagnostics:

* a deterministic depth-cluster mask proposal, scored against the unchanged
  raw V0 masks and raw world pointmap;
* causal projection of a previous 3-D object center into the current image,
  scored as a spatial prompt prior without calling SAM again.

The static-background pose component is exercised only with a synthetic
gradient smoke because the retained V0 cache does not contain a genuine
per-pixel StreamVGGT tracking-loss map.  The distinction is recorded in every
output artifact so this experiment cannot silently become a pose-training
claim.

GT is not accessed during candidate generation.  It is opened only after the
depth proposals and all temporal projections have been frozen in memory.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from streaming_couping.src.bidirectional_feedback import (
    DepthGuidedMaskRefiner,
    StaticBackgroundPoseOptimizer,
    TemporalPromptProjector,
)
from streaming_couping.src.learned_pose.baseline_runtime import (
    load_baseline_run_config,
)
from streaming_couping.src.learned_pose.cache import cache_path, load_feature_cache
from streaming_couping.src.learned_pose.config import (
    ClipConfig,
    LearnedPoseConfig,
    load_learned_pose_config,
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

from streaming_couping.scripts.run_semantic_map_evaluation import (
    _build_oracle_variant,
    _load_run as _load_evaluation_run,
)
from streaming_couping.scripts.run_v0_semantic_map import (
    _validate_inputs as _validate_v0_inputs,
)


REVISION = "v0_bidirectional_feedback_frozen_diagnostic_r2"
DEFAULT_CONFIG = "streaming_couping/configs/v0_baseline.yaml"
DEFAULT_OUTPUT = (
    "${VGGT_SAM_STORAGE_ROOT}/outputs/"
    "streaming_couping_v0_bidirectional_feedback"
)


@dataclass(frozen=True)
class V0Arrays:
    """Shape-normalized tensors from one completed V0 cache."""

    points: torch.Tensor
    confidence: torch.Tensor
    raw_masks: torch.Tensor
    scores: torch.Tensor
    depth: torch.Tensor
    intrinsics: torch.Tensor
    raw_world_to_camera: torch.Tensor
    selected_world_to_camera: torch.Tensor
    image_size: tuple[int, int]
    map_size: tuple[int, int]


def main() -> None:
    args = _parse_args()
    if args.min_center_points < 1:
        raise ValueError("--min-center-points must be positive.")
    if not 0.0 <= args.confidence_threshold <= 1.0:
        raise ValueError("--confidence-threshold must be in [0,1].")
    if not 0.0 <= args.track_score_threshold <= 1.0:
        raise ValueError("--track-score-threshold must be in [0,1].")
    config_path = Path(args.config).expanduser().resolve()
    data = load_learned_pose_config(config_path)
    baseline = load_baseline_run_config(config_path)
    if baseline.version != "v0":
        raise ValueError("This diagnostic requires baseline.version=v0.")
    clip = _find_clip(data, baseline.clip_name)
    if len(data.clips) != 1 or clip.name != "00a231a370_90_525_step15_37_68_54":
        raise ValueError(
            "The V0 feedback diagnostic is intentionally sealed to the "
            "single baseline clip 00a231a370_90_525_step15_37_68_54."
        )
    output_dir = expand_storage_path(
        args.output_dir or DEFAULT_OUTPUT,
        base=config_path.parent,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    path = cache_path(data, clip)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing frozen V0 cache {path}; run commands_v0_baseline.txt first."
        )
    payload = load_feature_cache(path)
    baseline_summary, poses = _validate_v0_inputs(
        payload=payload,
        summary_path=baseline.output_dir / "baseline_summary.json",
        poses_path=baseline.output_dir / "poses.pt",
        clip=clip,
    )
    arrays = _load_arrays(payload, poses)
    track_ids = tuple(int(value) for value in payload["sam_track_ids"])
    track_prompts = tuple(str(value) for value in payload["sam_track_prompts"])
    frames = tuple(int(value) for value in payload["frame_indices"])

    print("V0 EXPLICIT BIDIRECTIONAL FEEDBACK FROZEN DIAGNOSTIC")
    print(f"config={config_path}")
    print(f"cache={path}")
    print(f"output={output_dir}")
    print(
        "scope=one sealed V0 scene; existing SAM prompts/tracks and "
        "StreamVGGT pointmap are reused"
    )
    print(
        "candidate_generation=raw V0 cache only; GT is opened after "
        "depth/projection candidates are frozen"
    )
    print(
        f"frames={len(frames)} slots={arrays.raw_masks.shape[1]} "
        f"map_size={arrays.map_size} depth_size={tuple(arrays.depth.shape[-2:])} "
        f"image_size={arrays.image_size}"
    )

    # Module 1 has no real loss map in the frozen V0 artifact.  Run an
    # independent differentiability smoke and retain the result as provenance.
    static_smoke = _static_background_gradient_smoke()

    # Candidate generation phase: no GT fields are read below this marker.
    refiner = DepthGuidedMaskRefiner(
        absolute_gap=float(args.absolute_gap),
        relative_gap=float(args.relative_gap),
    )
    refined_masks, depth_rows = _build_depth_refined_masks(
        arrays.raw_masks,
        arrays.depth,
        arrays.map_size,
        refiner,
        frames=frames,
        track_ids=track_ids,
        track_prompts=track_prompts,
    )
    pose_sources = _pose_source_tensors(arrays, args.pose_source)
    projection_rows, projection_meta = _build_temporal_projection_rows(
        points=arrays.points,
        confidence=arrays.confidence,
        raw_masks=arrays.raw_masks,
        scores=arrays.scores,
        intrinsics=arrays.intrinsics,
        pose_sources=pose_sources,
        image_size=arrays.image_size,
        frames=frames,
        track_ids=track_ids,
        min_center_points=int(args.min_center_points),
        confidence_threshold=float(args.confidence_threshold),
        track_score_threshold=float(args.track_score_threshold),
    )
    _write_csv(output_dir / "depth_refinement.csv", depth_rows)
    _write_csv(output_dir / "temporal_projection_causal.csv", projection_rows)
    print(
        f"candidates frozen depth_rows={len(depth_rows)} "
        f"projection_rows={len(projection_rows)} "
        f"history_centers={projection_meta['history_center_count']}"
    )

    # Sealed evaluation phase.  Everything involving GT—including alignment,
    # assignment, and mask-hit evaluation—is intentionally below this line.
    print("candidate artifacts frozen; opening sealed GT evaluation")
    evaluation_run = _load_evaluation_run(config_path)
    # First resolve the prompt-scope IDs and labels.  The masks used for all
    # stream/pointmap scoring are then rebuilt with the repository's canonical
    # StreamVGGT resize/crop/pad transform.  A plain resize would silently
    # disagree on portrait frames (and on any frame that is padded/cropped).
    ground_truth = load_ground_truth_instances(
        data.manifest,
        scene_id=str(payload["scene_id"]),
        frame_indices=frames,
        output_size=arrays.map_size,
        prompts=tuple(str(value) for value in payload["instance_prompts"]),
        prompt_label_aliases=evaluation_run.prompt_label_aliases,
    )
    gt_map_masks = load_ground_truth_stream_masks(
        data.manifest,
        scene_id=str(payload["scene_id"]),
        frame_indices=frames,
        instance_ids=ground_truth.instance_ids,
        processed_size=arrays.map_size,
        image_mode=str(payload["image_mode"]),
    )
    gt_map = GroundTruthInstances(
        masks=gt_map_masks,
        instance_ids=ground_truth.instance_ids,
        labels=ground_truth.labels,
        all_visible_instance_ids=ground_truth.all_visible_instance_ids,
    )
    gt_projection_masks = load_ground_truth_stream_masks(
        data.manifest,
        scene_id=str(payload["scene_id"]),
        frame_indices=frames,
        instance_ids=ground_truth.instance_ids,
        processed_size=arrays.image_size,
        image_mode=str(payload["image_mode"]),
    )
    gt_projection = GroundTruthInstances(
        masks=gt_projection_masks,
        instance_ids=ground_truth.instance_ids,
        labels=ground_truth.labels,
        all_visible_instance_ids=ground_truth.all_visible_instance_ids,
    )
    # The first pass establishes the frozen raw assignment.  The oracle is
    # then rebuilt with that assignment, so no branch can rematch itself.
    assignment_pass = evaluate_tracking_variants(
        scene_id=str(payload["scene_id"]),
        clip_name=str(payload["clip_name"]),
        frame_indices=frames,
        variant_masks={
            "raw_v0": arrays.raw_masks,
            "depth_refined_v0": refined_masks,
        },
        variant_scores={
            "raw_v0": arrays.scores,
            "depth_refined_v0": arrays.scores,
        },
        raw_variant="raw_v0",
        track_ids=track_ids,
        track_prompts=track_prompts,
        ground_truth=gt_map,
        config=evaluation_run.tracking,
        prompt_label_aliases=evaluation_run.prompt_label_aliases,
    )
    oracle_masks, oracle_scores = _build_oracle_variant(
        gt_map.masks,
        assignments=assignment_pass["assignments"],
        sequence_shape=tuple(arrays.raw_masks.shape),
    )
    tracking = evaluate_tracking_variants(
        scene_id=str(payload["scene_id"]),
        clip_name=str(payload["clip_name"]),
        frame_indices=frames,
        variant_masks={
            "raw_v0": arrays.raw_masks,
            "depth_refined_v0": refined_masks,
            "gt_mask_oracle": oracle_masks,
        },
        variant_scores={
            "raw_v0": arrays.scores,
            "depth_refined_v0": arrays.scores,
            "gt_mask_oracle": oracle_scores,
        },
        raw_variant="raw_v0",
        track_ids=track_ids,
        track_prompts=track_prompts,
        ground_truth=gt_map,
        config=evaluation_run.tracking,
        prompt_label_aliases=evaluation_run.prompt_label_aliases,
    )

    target_points = _tensor(payload["target_world_points"], "target_world_points").float()
    if tuple(target_points.shape) != tuple(arrays.points.shape):
        raise ValueError(
            "V0 target pointmap and baseline pointmap have different shapes: "
            f"{tuple(target_points.shape)} vs {tuple(arrays.points.shape)}"
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
    map_confidence = _map_confidence(payload["baseline_world_confidence"])
    map_results: list[dict[str, object]] = []
    map_object_rows: list[dict[str, object]] = []
    for variant, masks, scores in (
        ("raw_v0", arrays.raw_masks, arrays.scores),
        ("depth_refined_v0", refined_masks, arrays.scores),
        ("gt_mask_oracle", oracle_masks, oracle_scores),
    ):
        result = evaluate_semantic_object_map(
            scene_id=str(payload["scene_id"]),
            clip_name=str(payload["clip_name"]),
            variant=variant,
            map_policy="all_visible_observations",
            aligned_world_points=aligned_points,
            target_world_points=target_points,
            confidence=map_confidence,
            predicted_masks=masks,
            track_scores=scores,
            gt_masks=gt_map.masks,
            gt_instance_ids=gt_map.instance_ids,
            gt_labels=gt_map.labels,
            assignments=tracking["assignments"],
            config=evaluation_run.map_metrics,
        )
        map_results.append(result["summary"])
        map_object_rows.extend(result["object_rows"])

    projection_rows = _add_gt_projection_hits(
        projection_rows,
        ground_truth=gt_projection,
        assignments=tracking["assignments"],
        image_size=arrays.image_size,
    )
    projection_summary = _summarize_projection_rows(projection_rows)
    _write_csv(output_dir / "temporal_projection.csv", projection_rows)
    _write_csv(output_dir / "temporal_projection_summary.csv", projection_summary)
    _write_csv(output_dir / "tracking_metrics.csv", tracking["summary_rows"])
    _write_csv(output_dir / "tracking_object_metrics.csv", tracking["object_rows"])
    _write_csv(output_dir / "map_metrics.csv", map_results)
    _write_csv(output_dir / "map_object_metrics.csv", map_object_rows)

    summary = _build_summary(
        payload=payload,
        baseline_summary=baseline_summary,
        cache_path_value=path,
        output_dir=output_dir,
        arrays=arrays,
        static_smoke=static_smoke,
        depth_rows=depth_rows,
        map_results=map_results,
        tracking_rows=tracking["summary_rows"],
        projection_meta=projection_meta,
        projection_summary=projection_summary,
        args=args,
    )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=True, default=str)
        + "\n",
        encoding="utf8",
    )
    copyable_path = output_dir / "copyable_result.txt"
    _write_copyable(copyable_path, summary)
    summary["outputs"] = {
        "summary": str(summary_path),
        "copyable_result": str(copyable_path),
        "depth_refinement": str(output_dir / "depth_refinement.csv"),
        "temporal_projection": str(output_dir / "temporal_projection.csv"),
        "temporal_projection_causal": str(
            output_dir / "temporal_projection_causal.csv"
        ),
        "temporal_projection_summary": str(
            output_dir / "temporal_projection_summary.csv"
        ),
        "tracking_metrics": str(output_dir / "tracking_metrics.csv"),
        "tracking_object_metrics": str(
            output_dir / "tracking_object_metrics.csv"
        ),
        "map_metrics": str(output_dir / "map_metrics.csv"),
        "map_object_metrics": str(output_dir / "map_object_metrics.csv"),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=True, default=str)
        + "\n",
        encoding="utf8",
    )
    _write_copyable(copyable_path, summary)
    _print_diagnostic_summary(
        summary=summary,
        depth_rows=depth_rows,
        tracking_rows=tracking["summary_rows"],
        map_rows=map_results,
        projection_summary=projection_summary,
    )
    print(f"summary={summary_path}")
    print(f"copyable_result={copyable_path}")
    print("decision=DIAGNOSTIC_ONLY")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--pose-source",
        choices=("selected_v0", "raw_streamvggt", "both"),
        default="both",
        help="Pose used for projection diagnostics; both is the default control.",
    )
    parser.add_argument(
        "--absolute-gap",
        type=float,
        default=0.05,
        help="Fixed depth gap threshold for the diagnostic heuristic.",
    )
    parser.add_argument(
        "--relative-gap",
        type=float,
        default=0.05,
        help="Depth gap threshold as a fraction of the masked median depth.",
    )
    parser.add_argument("--min-center-points", type=int, default=8)
    parser.add_argument("--confidence-threshold", type=float, default=0.30)
    parser.add_argument("--track-score-threshold", type=float, default=0.50)
    return parser.parse_args()


def _find_clip(data: LearnedPoseConfig, name: str) -> ClipConfig:
    selected = [clip for clip in data.clips if clip.name == name]
    if len(selected) != 1:
        raise ValueError(f"V0 clip {name!r} was not found exactly once.")
    return selected[0]


def _load_arrays(payload: Mapping[str, object], poses: Mapping[str, object]) -> V0Arrays:
    points = _tensor(payload["baseline_world_points"], "baseline_world_points").float()
    confidence = _map_confidence(payload["baseline_world_confidence"])
    raw_masks = _tensor(payload["tracking_masks_stream"], "tracking_masks_stream").bool()
    scores = _tensor(payload["tracking_scores"], "tracking_scores").float()
    depth = _sequence_scalar_map(payload["baseline_depth"], "baseline_depth")
    intrinsics = _tensor(payload["baseline_intrinsics"], "baseline_intrinsics").float()
    raw_w2c = _pose_sequence(
        _tensor(payload["baseline_world_to_camera"], "baseline_world_to_camera")
    )
    selected_w2c = _pose_sequence(
        _tensor(poses["selected_world_to_camera"], "selected_world_to_camera")
    )
    if points.ndim != 4 or points.shape[-1] != 3:
        raise ValueError(f"V0 world points must be [S,H,W,3], got {tuple(points.shape)}")
    sequence, height, width = points.shape[:3]
    if tuple(confidence.shape) != (sequence, height, width):
        raise ValueError("V0 confidence does not match the world pointmap.")
    if raw_masks.ndim != 4 or raw_masks.shape[0] != sequence:
        raise ValueError("V0 tracking_masks_stream must be [S,K,H,W].")
    if tuple(raw_masks.shape[-2:]) != (height, width):
        raise ValueError(
            "V0 tracking_masks_stream must already use the pointmap grid; "
            f"got {tuple(raw_masks.shape[-2:])} vs {(height, width)}."
        )
    if scores.shape != raw_masks.shape[:2]:
        raise ValueError("V0 tracking scores do not match mask slots.")
    if depth.shape[0] != sequence:
        raise ValueError("V0 depth/frame count mismatch.")
    if intrinsics.shape != (sequence, 3, 3):
        raise ValueError(
            f"V0 intrinsics must be [S,3,3], got {tuple(intrinsics.shape)}"
        )
    if raw_w2c.shape != (sequence, 3, 4) or selected_w2c.shape != (sequence, 3, 4):
        raise ValueError("V0 raw/selected poses must be [S,3,4].")
    image_size = tuple(int(value) for value in payload["image_size"])
    if len(image_size) != 2 or min(image_size) <= 0:
        raise ValueError("V0 image_size is invalid.")
    return V0Arrays(
        points=points,
        confidence=confidence,
        raw_masks=raw_masks,
        scores=scores,
        depth=depth,
        intrinsics=intrinsics,
        raw_world_to_camera=raw_w2c,
        selected_world_to_camera=selected_w2c,
        image_size=(image_size[0], image_size[1]),
        map_size=(height, width),
    )


def _tensor(value: object, name: str) -> torch.Tensor:
    if not torch.is_tensor(value):
        return torch.as_tensor(value)
    return value.detach().cpu()


def _map_confidence(value: object) -> torch.Tensor:
    confidence = _tensor(value, "baseline_world_confidence").float()
    if confidence.ndim == 4 and confidence.shape[-1] == 1:
        confidence = confidence[..., 0]
    if confidence.ndim != 3:
        raise ValueError(
            "baseline_world_confidence must be [S,H,W] or [S,H,W,1]."
        )
    return confidence


def _sequence_scalar_map(value: object, name: str) -> torch.Tensor:
    tensor = _tensor(value, name).float()
    if tensor.ndim == 4 and tensor.shape[-1] == 1:
        tensor = tensor[..., 0]
    elif tensor.ndim == 4 and tensor.shape[1] == 1:
        tensor = tensor[:, 0]
    if tensor.ndim != 3:
        raise ValueError(f"{name} must be [S,H,W] or a singleton-channel map.")
    return tensor


def _pose_sequence(value: torch.Tensor) -> torch.Tensor:
    pose = value.float().cpu()
    if pose.ndim == 4 and pose.shape[0] == 1:
        pose = pose[0]
    if pose.ndim != 3 or tuple(pose.shape[1:]) != (3, 4):
        raise ValueError(f"Expected pose sequence [S,3,4], got {tuple(pose.shape)}")
    return pose


def _resize_masks(masks: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    masks = masks.detach().cpu().bool()
    if tuple(masks.shape[-2:]) == tuple(size):
        return masks
    if masks.numel() == 0:
        return torch.zeros(
            *masks.shape[:-2],
            int(size[0]),
            int(size[1]),
            dtype=torch.bool,
        )
    flat = masks.float().reshape(-1, 1, masks.shape[-2], masks.shape[-1])
    resized = F.interpolate(flat, size=size, mode="nearest")
    return resized.reshape(*masks.shape[:-2], size[0], size[1]).bool()


def _build_depth_refined_masks(
    raw_masks: torch.Tensor,
    depth: torch.Tensor,
    map_size: tuple[int, int],
    refiner: DepthGuidedMaskRefiner,
    *,
    frames: Sequence[int],
    track_ids: Sequence[int],
    track_prompts: Sequence[str],
) -> tuple[torch.Tensor, list[dict[str, object]]]:
    sequence, slots = raw_masks.shape[:2]
    depth_size = (int(depth.shape[1]), int(depth.shape[2]))
    masks_at_depth = _resize_masks(raw_masks, depth_size)
    refined_at_depth = torch.zeros_like(masks_at_depth)
    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for frame in range(sequence):
            for slot in range(slots):
                refined, stats = refiner.refine(
                    masks_at_depth[frame, slot],
                    depth[frame],
                    return_stats=True,
                )
                refined_at_depth[frame, slot] = refined
                row = stats.to_dict()
                row.update(
                    {
                        "sequence_index": int(frame),
                        "frame_index": int(frames[frame]),
                        "slot": int(slot),
                        "sam_track_id": int(track_ids[slot]),
                        "prompt": str(track_prompts[slot]),
                        "depth_height": int(depth_size[0]),
                        "depth_width": int(depth_size[1]),
                    }
                )
                rows.append(row)
    refined_map = _resize_masks(refined_at_depth, map_size)
    return refined_map, rows


def _pose_source_tensors(
    arrays: V0Arrays,
    selection: str,
) -> dict[str, torch.Tensor]:
    if selection == "selected_v0":
        return {"selected_v0": arrays.selected_world_to_camera}
    if selection == "raw_streamvggt":
        return {"raw_streamvggt": arrays.raw_world_to_camera}
    return {
        "selected_v0": arrays.selected_world_to_camera,
        "raw_streamvggt": arrays.raw_world_to_camera,
    }


def _build_temporal_projection_rows(
    *,
    points: torch.Tensor,
    confidence: torch.Tensor,
    raw_masks: torch.Tensor,
    scores: torch.Tensor,
    intrinsics: torch.Tensor,
    pose_sources: Mapping[str, torch.Tensor],
    image_size: tuple[int, int],
    frames: Sequence[int],
    track_ids: Sequence[int],
    min_center_points: int,
    confidence_threshold: float,
    track_score_threshold: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    sequence, height, width = points.shape[:3]
    slots = raw_masks.shape[1]
    if len(track_ids) != slots:
        raise ValueError("track_ids do not match V0 mask slots.")
    centers = torch.full((sequence, slots, 3), float("nan"))
    center_valid = torch.zeros(sequence, slots, dtype=torch.bool)
    center_counts = torch.zeros(sequence, slots, dtype=torch.long)
    for frame in range(sequence):
        for slot in range(slots):
            support = (
                raw_masks[frame, slot]
                & (scores[frame, slot] >= float(track_score_threshold))
                & (confidence[frame] >= float(confidence_threshold))
                & torch.isfinite(points[frame]).all(dim=-1)
            )
            count = int(support.sum())
            center_counts[frame, slot] = count
            if count >= int(min_center_points):
                centers[frame, slot] = torch.median(
                    points[frame][support].float(),
                    dim=0,
                ).values
                center_valid[frame, slot] = True

    history = torch.full((sequence, slots), -1, dtype=torch.long)
    last_seen = [-1] * slots
    for frame in range(sequence):
        history[frame] = torch.tensor(last_seen, dtype=torch.long)
        for slot in range(slots):
            if bool(center_valid[frame, slot]):
                last_seen[slot] = frame

    raw_image_masks = _resize_masks(raw_masks, image_size)
    projector = TemporalPromptProjector()
    rows: list[dict[str, object]] = []
    for frame in range(sequence):
        for slot in range(slots):
            source_frame = int(history[frame, slot])
            if source_frame < 0:
                continue
            center = centers[source_frame, slot].reshape(1, 3)
            for pose_name, world_to_camera in pose_sources.items():
                c2w = _w2c_to_c2w(world_to_camera[frame])
                result = projector.project(
                    center,
                    c2w,
                    intrinsics[frame],
                    image_size,
                    object_ids=(int(track_ids[slot]),),
                )
                valid = bool(result.valid_mask[0])
                uv = result.all_points_uv[0]
                depth_value = float(result.all_depth[0])
                raw_hit = (
                    int(_sample_mask(raw_image_masks[frame, slot], uv))
                    if valid
                    else 0
                )
                rows.append(
                    {
                        "pose_source": str(pose_name),
                        "sequence_index": int(frame),
                        "frame_index": int(frames[frame]),
                        "slot": int(slot),
                        "sam_track_id": int(track_ids[slot]),
                        "history_sequence_index": source_frame,
                        "history_frame_index": int(frames[source_frame]),
                        "history_gap": int(frame - source_frame),
                        "history_center_points": int(center_counts[source_frame, slot]),
                        "projected_u": _finite_float(uv[0]),
                        "projected_v": _finite_float(uv[1]),
                        "projected_depth": _finite_float(depth_value),
                        "in_bounds": int(valid),
                        "current_raw_mask_hit": int(raw_hit),
                        "gt_assignment_available": -1,
                        "gt_mask_hit": -1,
                        "gt_instance_id": -1,
                    }
                )
    meta = {
        "history_center_count": int(center_valid.sum()),
        "projection_candidate_count": len(rows),
        "center_valid_count": int(center_valid.sum()),
        "center_point_count_median": (
            float(center_counts[center_valid].float().median())
            if bool(center_valid.any())
            else 0.0
        ),
    }
    return rows, meta


def _w2c_to_c2w(world_to_camera: torch.Tensor) -> torch.Tensor:
    w2c = world_to_camera.float().cpu()
    if tuple(w2c.shape) != (3, 4):
        raise ValueError(f"Expected [3,4] pose, got {tuple(w2c.shape)}")
    homogeneous = torch.eye(4, dtype=w2c.dtype)
    homogeneous[:3] = w2c
    return torch.linalg.inv(homogeneous)


def _sample_mask(mask: torch.Tensor, uv: torch.Tensor) -> bool:
    if not bool(torch.isfinite(uv).all()):
        return False
    height, width = mask.shape[-2:]
    x = int(torch.round(uv[0]).item())
    y = int(torch.round(uv[1]).item())
    if x < 0 or x >= width or y < 0 or y >= height:
        return False
    return bool(mask[y, x])


def _add_gt_projection_hits(
    rows: list[dict[str, object]],
    *,
    ground_truth: GroundTruthInstances,
    assignments: Sequence[Mapping[str, object]],
    image_size: tuple[int, int],
) -> list[dict[str, object]]:
    slot_to_target = {
        int(row["slot"]): int(row["gt_index"])
        for row in assignments
    }
    gt_map = ground_truth.masks
    gt_image = _resize_masks(gt_map, image_size)
    for row in rows:
        slot = int(row["slot"])
        target = slot_to_target.get(slot, -1)
        row["gt_assignment_available"] = int(target >= 0)
        row["gt_instance_id"] = (
            int(ground_truth.instance_ids[target]) if target >= 0 else -1
        )
        if target < 0 or not int(row["in_bounds"]):
            row["gt_mask_hit"] = 0 if target >= 0 else -1
            continue
        # Projection coordinates are in processed image space.  The GT mask
        # is transformed to that same grid before sampling, exactly as for
        # the raw-mask hit test.
        projected = torch.tensor(
            [float(row["projected_u"]), float(row["projected_v"])],
            dtype=torch.float32,
        )
        source_height, source_width = image_size
        x = int(round(float(projected[0])))
        y = int(round(float(projected[1])))
        row["gt_mask_hit"] = int(
            0 <= x < source_width
            and 0 <= y < source_height
            and bool(gt_image[int(row["sequence_index"]), target, y, x])
        )
    return rows


def _summarize_projection_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    sources = sorted({str(row["pose_source"]) for row in rows})
    for source in sources:
        selected = [row for row in rows if str(row["pose_source"]) == source]
        valid = [row for row in selected if int(row["in_bounds"])]
        assigned = [row for row in valid if int(row["gt_assignment_available"]) == 1]
        output.append(
            {
                "pose_source": source,
                "candidate_rows": len(selected),
                "in_bounds_rows": len(valid),
                "in_bounds_rate": _ratio(len(valid), len(selected)),
                "raw_mask_hits": sum(int(row["current_raw_mask_hit"]) for row in valid),
                "raw_mask_hit_rate": _ratio(
                    sum(int(row["current_raw_mask_hit"]) for row in valid),
                    len(valid),
                ),
                "assigned_in_bounds_rows": len(assigned),
                "gt_mask_hits": sum(int(row["gt_mask_hit"]) for row in assigned),
                "gt_mask_hit_rate": _ratio(
                    sum(int(row["gt_mask_hit"]) for row in assigned),
                    len(assigned),
                ),
                "mean_history_gap": _mean(
                    [float(row["history_gap"]) for row in selected]
                ),
            }
        )
    return output


def _static_background_gradient_smoke() -> dict[str, object]:
    loss = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4).clone()
    loss.requires_grad_(True)
    masks = torch.zeros(1, 1, 3, 4, dtype=torch.bool)
    masks[0, 0, 1, 1] = True
    masks[0, 0, 2, 2] = True
    optimizer = StaticBackgroundPoseOptimizer()
    background = optimizer.static_background_mask(masks)
    value = optimizer(loss, masks)
    value.backward()
    foreground_grad_zero = bool((loss.grad[~background] == 0).all())
    background_grad_positive = bool((loss.grad[background] > 0).all())
    return {
        "evaluated_with_real_v0_loss_map": 0,
        "synthetic_gradient_smoke_pass": int(
            foreground_grad_zero and background_grad_positive
        ),
        "foreground_pixels": int((~background).sum()),
        "background_pixels": int(background.sum()),
        "pose_update_performed": 0,
    }


def _build_summary(
    *,
    payload: Mapping[str, object],
    baseline_summary: Mapping[str, object],
    cache_path_value: Path,
    output_dir: Path,
    arrays: V0Arrays,
    static_smoke: Mapping[str, object],
    depth_rows: Sequence[Mapping[str, object]],
    map_results: Sequence[Mapping[str, object]],
    tracking_rows: Sequence[Mapping[str, object]],
    projection_meta: Mapping[str, object],
    projection_summary: Sequence[Mapping[str, object]],
    args: argparse.Namespace,
) -> dict[str, object]:
    map_by_variant = {str(row["variant"]): dict(row) for row in map_results}
    tracking_by_variant = {
        str(row["variant"]): dict(row) for row in tracking_rows
    }
    raw_map = map_by_variant.get("raw_v0", {})
    refined_map = map_by_variant.get("depth_refined_v0", {})
    raw_tracking = tracking_by_variant.get("raw_v0", {})
    refined_tracking = tracking_by_variant.get("depth_refined_v0", {})
    depth_input = sum(int(row["input_mask_pixels"]) for row in depth_rows)
    depth_removed = sum(int(row["removed_pixels"]) for row in depth_rows)
    return {
        "schema": 1,
        "revision": REVISION,
        "diagnostic_only": 1,
        "decision": "DIAGNOSTIC_ONLY",
        "clip": str(payload["clip_name"]),
        "scene_id": str(payload["scene_id"]),
        "frames": list(int(value) for value in payload["frame_indices"]),
        "cache": str(cache_path_value),
        "output_dir": str(output_dir),
        "baseline_revision": str(baseline_summary.get("implementation_revision", "")),
        "baseline_selected_pose_branch": str(
            baseline_summary.get("selected_pose_branch", "")
        ),
        "prompt_source": "preconfigured_v0_sam31_online_prompts",
        "candidate_generation_gt_fields": 0,
        "evaluation_gt_fields": 1,
        "models_rerun": 0,
        "parameters_updated": 0,
        "pointmap_modified": 0,
        "pose_modified": 0,
        "sam_rerun_with_temporal_prompts": 0,
        "static_background_pose": dict(static_smoke),
        "depth_refinement": {
            "heuristic": "largest_sorted_depth_cluster",
            "absolute_gap": float(args.absolute_gap),
            "relative_gap": float(args.relative_gap),
            "rows": len(depth_rows),
            "input_mask_pixels": depth_input,
            "removed_pixels": depth_removed,
            "removed_pixel_ratio": _ratio(depth_removed, depth_input),
            "fallback_rows": sum(int(row["fallback_used"]) for row in depth_rows),
            "map_metrics_raw": raw_map,
            "map_metrics_refined": refined_map,
            "tracking_metrics_raw": raw_tracking,
            "tracking_metrics_refined": refined_tracking,
        },
        "temporal_prompt_projection": {
            "projector": "TemporalPromptProjector",
            "pose_sources": sorted(
                str(row["pose_source"]) for row in projection_summary
            ),
            "candidate_generation": dict(projection_meta),
            "summary": list(projection_summary),
            "sam_consumed_prompts": 0,
        },
        "v0_shapes": {
            "world_points": list(arrays.points.shape),
            "raw_masks": list(arrays.raw_masks.shape),
            "depth": list(arrays.depth.shape),
            "image_size": list(arrays.image_size),
            "map_size": list(arrays.map_size),
        },
        "caveat": (
            "Depth refinement is a heuristic diagnostic and temporal projection "
            "is evaluated as a spatial prior only; neither result proves a SAM "
            "closed loop or a pose improvement."
        ),
        "outputs": {},
    }


def _write_copyable(path: Path, summary: Mapping[str, object]) -> None:
    depth = summary["depth_refinement"]
    temporal = summary["temporal_prompt_projection"]
    lines = [
        "===== V0_BIDIRECTIONAL_FEEDBACK_BEGIN =====",
        f"revision={summary['revision']}",
        f"decision={summary['decision']}",
        f"clip={summary['clip']}",
        "scope=single frozen V0 scene",
        "prompt_source=preconfigured_v0_sam31_online_prompts",
        "candidate_generation_gt_fields=0",
        "models_rerun=0",
        "parameters_updated=0",
        "pointmap_modified=0",
        "pose_modified=0",
        "sam_rerun_with_temporal_prompts=0",
        "static_background_pose=interface_smoke_only_no_real_tracking_loss_map",
        f"static_background_smoke_pass={summary['static_background_pose']['synthetic_gradient_smoke_pass']}",
        f"depth_refined_removed_pixel_ratio={depth['removed_pixel_ratio']}",
        f"depth_refined_raw_map_voxel_iou={depth['map_metrics_raw'].get('voxel_iou_5cm')}",
        f"depth_refined_map_voxel_iou={depth['map_metrics_refined'].get('voxel_iou_5cm')}",
        f"depth_refined_raw_map_fscore_5cm={depth['map_metrics_raw'].get('fscore_5cm')}",
        f"depth_refined_map_fscore_5cm={depth['map_metrics_refined'].get('fscore_5cm')}",
        f"depth_refined_raw_tracking_iou={depth['tracking_metrics_raw'].get('mean_frame_iou')}",
        f"depth_refined_tracking_iou={depth['tracking_metrics_refined'].get('mean_frame_iou')}",
        "temporal_projection_summary="
        + json.dumps(list(temporal["summary"]), sort_keys=True, allow_nan=True),
        "temporal_projection_is_spatial_prior_only=1",
        "interpretation=diagnostic_only; temporal points were not fed back into SAM",
        "caveat=" + str(summary["caveat"]),
        "===== V0_BIDIRECTIONAL_FEEDBACK_END =====",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def _print_diagnostic_summary(
    *,
    summary: Mapping[str, object],
    depth_rows: Sequence[Mapping[str, object]],
    tracking_rows: Sequence[Mapping[str, object]],
    map_rows: Sequence[Mapping[str, object]],
    projection_summary: Sequence[Mapping[str, object]],
) -> None:
    """Print the small report needed for terminal-only experiment review.

    The detailed CSV/JSON artifacts remain the source of record.  This report
    deliberately prints aggregate rows only, so a 30-frame/16-slot run does
    not produce hundreds of terminal lines while still exposing every metric
    needed for the first GO/NO-GO decision.
    """

    depth = summary["depth_refinement"]
    static = summary["static_background_pose"]
    print("===== V0 BIDIRECTIONAL FEEDBACK DIAGNOSTIC SUMMARY =====")
    print(
        "static_background "
        f"synthetic_smoke={static.get('synthetic_gradient_smoke_pass')} "
        f"real_loss_map={static.get('evaluated_with_real_v0_loss_map')}"
    )
    input_pixels = int(depth.get("input_mask_pixels", 0))
    removed_pixels = int(depth.get("removed_pixels", 0))
    print(
        "depth_refinement "
        f"rows={len(depth_rows)} "
        f"input_pixels={input_pixels} "
        f"removed_pixels={removed_pixels} "
        f"removed_ratio={_display(depth.get('removed_pixel_ratio'))} "
        f"fallback_rows={depth.get('fallback_rows', 0)}"
    )

    temporal = summary["temporal_prompt_projection"]
    temporal_meta = temporal.get("candidate_generation", {})
    print(
        "temporal_support "
        f"history_centers={temporal_meta.get('history_center_count', 0)} "
        f"center_point_median={_display(temporal_meta.get('center_point_count_median'))} "
        f"projection_rows={temporal_meta.get('projection_candidate_count', 0)}"
    )

    print("tracking")
    for row in tracking_rows:
        print(
            "  "
            f"variant={row.get('variant')} "
            f"IoU={_display(row.get('mean_frame_iou'))} "
            f"frame_IDF1={_display(row.get('frame_idf1'))} "
            f"pixel_IDF1={_display(row.get('pixel_idf1'))} "
            f"reentry={row.get('reentry_successes', 0)}/"
            f"{row.get('reentry_events', 0)} "
            f"fragmentation={row.get('fragmentation_count', 0)} "
            f"merge_errors={row.get('merge_error_count', 0)}"
        )

    print("map")
    for row in map_rows:
        print(
            "  "
            f"variant={row.get('variant')} "
            f"voxelIoU5cm={_display(row.get('voxel_iou_5cm'))} "
            f"F5cm={_display(row.get('fscore_5cm'))} "
            f"ghost={_display(row.get('ghost_point_ratio'))} "
            f"accuracy_m={_display(row.get('object_accuracy_m'))} "
            f"completeness_m={_display(row.get('object_completeness_m'))}"
        )

    raw_map = _row_by_key(map_rows, "variant", "raw_v0")
    refined_map = _row_by_key(map_rows, "variant", "depth_refined_v0")
    if raw_map and refined_map:
        print(
            "map_delta refined_minus_raw "
            f"voxelIoU5cm={_display(_difference(refined_map.get('voxel_iou_5cm'), raw_map.get('voxel_iou_5cm')))} "
            f"F5cm={_display(_difference(refined_map.get('fscore_5cm'), raw_map.get('fscore_5cm')))} "
            f"ghost={_display(_difference(refined_map.get('ghost_point_ratio'), raw_map.get('ghost_point_ratio')))}"
        )

    raw_tracking = _row_by_key(tracking_rows, "variant", "raw_v0")
    refined_tracking = _row_by_key(
        tracking_rows,
        "variant",
        "depth_refined_v0",
    )
    if raw_tracking and refined_tracking:
        print(
            "tracking_delta refined_minus_raw "
            f"IoU={_display(_difference(refined_tracking.get('mean_frame_iou'), raw_tracking.get('mean_frame_iou')))} "
            f"frame_IDF1={_display(_difference(refined_tracking.get('frame_idf1'), raw_tracking.get('frame_idf1')))} "
            f"pixel_IDF1={_display(_difference(refined_tracking.get('pixel_idf1'), raw_tracking.get('pixel_idf1')))}"
        )

    print("temporal_projection")
    for row in projection_summary:
        print(
            "  "
            f"pose={row.get('pose_source')} "
            f"candidates={row.get('candidate_rows', 0)} "
            f"in_bounds={_display(row.get('in_bounds_rate'))} "
            f"raw_mask_hit={_display(row.get('raw_mask_hit_rate'))} "
            f"gt_mask_hit={_display(row.get('gt_mask_hit_rate'))} "
            f"mean_history_gap={_display(row.get('mean_history_gap'))}"
        )
    print(
        "interpretation=diagnostic_only; temporal points were not fed back into SAM; "
        "static pose used synthetic loss smoke only"
    )


def _row_by_key(
    rows: Sequence[Mapping[str, object]],
    key: str,
    value: object,
) -> Mapping[str, object] | None:
    for row in rows:
        if row.get(key) == value:
            return row
    return None


def _difference(first: object, second: object) -> float:
    try:
        return float(first) - float(second)
    except (TypeError, ValueError):
        return float("nan")


def _display(value: object) -> str:
    """Format numeric report values without turning missing fields into errors."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "nan"
    return f"{number:.6f}"


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [dict(row) for row in rows]
    if not rows:
        path.write_text("\n", encoding="utf8")
        return
    fields = sorted({str(key) for row in rows for key in row})
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: _csv_value(row.get(key, ""))
                    for key in fields
                }
            )


def _csv_value(value: object) -> object:
    if torch.is_tensor(value):
        if value.ndim == 0:
            return value.item()
        return json.dumps(value.detach().cpu().tolist(), allow_nan=True)
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True, allow_nan=True, default=str)
    return value


def _finite_float(value: torch.Tensor | float) -> float:
    number = float(value.item()) if torch.is_tensor(value) else float(value)
    return number if math.isfinite(number) else float("nan")


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if float(denominator) else float("nan")


def _mean(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return sum(finite) / len(finite) if finite else float("nan")


if __name__ == "__main__":
    main()
