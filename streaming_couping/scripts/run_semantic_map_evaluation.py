#!/usr/bin/env python3
"""Evaluate frozen SAM/StreamVGGT semantic-map branches offline.

The runner consumes completed V0/V6 feature caches.  It never reruns either
model and never changes the cached pointmap.  GT masks and pointmaps are read
only for this evaluation stage, after candidate generation has finished.

The retained Stage 0 branches are:

``raw_sam``
    Raw causal SAM3.1 masks on the raw StreamVGGT pointmap.
``v6_geometry``
    Frozen geometry-guided SAM3.1 masks on the same raw pointmap.
``gt_mask_oracle``
    Evaluation-only GT masks placed into the raw assignment slots.
``v6_memory_gate``
    V6 masks with the explicit short/long object-map write policy.

The raw branch fixes the track-to-GT assignment for every branch.  This keeps
the result diagnostic: a variant cannot improve its score by rematching after
seeing annotations.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml

from streaming_couping.src.learned_pose.cache import (
    cache_path,
    load_feature_cache,
)
from streaming_couping.src.learned_pose.config import (
    ClipConfig,
    LearnedPoseConfig,
    load_learned_pose_config,
)
from streaming_couping.src.object_memory import (
    ObjectMemoryConfig,
    PersistentObjectMemory,
    PersistentObjectMemoryConfig,
    build_object_memory_write_policy,
    collapse_persistent_tracks,
)
from streaming_couping.src.instance_association import InstanceAssociationConfig
from streaming_couping.src.semantic_map_metrics import (
    SemanticMapMetricConfig,
    apply_similarity,
    evaluate_semantic_object_map,
    load_ground_truth_stream_masks,
)
from streaming_couping.src.semantic_map import normalize_confidence
from streaming_couping.src.semantic_tracking_metrics import (
    TrackingMetricConfig,
    evaluate_tracking_variants,
    load_ground_truth_instances,
)
from streaming_couping.src.storage import expand_storage_path


REVISION = "semantic_map_stage0_offline_evaluation_r1"
RAW_CACHE_VARIANT = "sam31_online_forward"
V6_CACHE_VARIANT = "sam31_online_geometry_compete"


@dataclass(frozen=True)
class EvaluationRunConfig:
    source_path: Path
    output_dir: Path
    raw_cache_variant: str
    v6_cache_variant: str
    tracking: TrackingMetricConfig
    map_metrics: SemanticMapMetricConfig
    memory: ObjectMemoryConfig
    persistent_memory: PersistentObjectMemoryConfig
    use_cached_appearance: bool
    include_v1_memory: bool
    prompt_label_aliases: dict[str, tuple[str, ...]]


def main() -> None:
    args = _parse_args()
    data = load_learned_pose_config(args.config)
    run = _load_run(args.config)
    run = replace(
        run,
        output_dir=(
            Path(args.output_dir).expanduser().resolve()
            if args.output_dir
            else run.output_dir
        ),
        include_v1_memory=bool(args.include_v1_memory),
    )

    clips = _select_clips(data.clips, args.clip)
    run.output_dir.mkdir(parents=True, exist_ok=True)
    clip_results = []
    tracking_rows: list[dict[str, object]] = []
    tracking_object_rows: list[dict[str, object]] = []
    tracking_frame_rows: list[dict[str, object]] = []
    map_rows: list[dict[str, object]] = []
    map_object_rows: list[dict[str, object]] = []
    map_duplicate_rows: list[dict[str, object]] = []
    memory_rows: list[dict[str, object]] = []
    persistent_memory_rows: list[dict[str, object]] = []
    persistent_object_rows: list[dict[str, object]] = []
    include_v1 = bool(run.include_v1_memory)

    print(
        "STAGE 1 V1 + STAGE 0 SEMANTIC-MAP OFFLINE EVALUATION"
        if include_v1
        else "STAGE 0 SEMANTIC-MAP OFFLINE EVALUATION"
    )
    print(f"config={run.source_path}")
    print(f"output={run.output_dir}")
    print(
        "GT is opened only for evaluation; Stage 0 raw assignment is frozen; "
        "V1 receives a separate evaluation-only object assignment"
        if include_v1
        else "GT is opened only for evaluation; raw assignment is frozen for all variants"
    )
    for clip in clips:
        result = _evaluate_clip(data, clip, run)
        clip_results.append(result["clip"])
        tracking_rows.extend(result["tracking"]["summary_rows"])
        tracking_object_rows.extend(result["tracking"]["object_rows"])
        tracking_frame_rows.extend(result["tracking"]["frame_rows"])
        map_rows.extend(result["map"]["summary_rows"])
        map_object_rows.extend(result["map"]["object_rows"])
        map_duplicate_rows.extend(result["map"]["duplicate_rows"])
        memory_rows.extend(result["memory"]["rows"])
        if include_v1:
            persistent_memory_rows.extend(result["persistent_memory"]["events"])
            persistent_object_rows.extend(result["persistent_memory"]["objects"])
            tracking_rows.extend(
                result["persistent_memory"]["tracking"]["summary_rows"]
            )
            tracking_object_rows.extend(
                result["persistent_memory"]["tracking"]["object_rows"]
            )
            tracking_frame_rows.extend(
                result["persistent_memory"]["tracking"]["frame_rows"]
            )
            map_rows.extend(result["persistent_memory"]["map"]["summary_rows"])
            map_object_rows.extend(
                result["persistent_memory"]["map"]["object_rows"]
            )
            map_duplicate_rows.extend(
                result["persistent_memory"]["map"]["duplicate_rows"]
            )
        _print_clip_result(result)

    tracking_aggregate = _aggregate_rows(
        tracking_rows,
        kind="tracking",
    )
    map_aggregate = _aggregate_rows(
        map_rows,
        kind="map",
    )
    summary = {
        "schema": 1,
        "revision": (
            REVISION + "_with_v1_memory" if include_v1 else REVISION
        ),
        "config": str(run.source_path),
        "output_dir": str(run.output_dir),
        "clips": clip_results,
        "clip_count": len(clip_results),
        "candidate_generation_gt_fields": 0,
        "evaluation_gt_fields": 1,
        "raw_assignment_frozen": 1,
        "v1_assignment_evaluation_only": int(include_v1),
        "pointmap_modified": 0,
        "streamvggt_pose_modified": 0,
        "streamvggt_pointmap_source": "raw_full_history_world_pointmap",
        "tracking": {
            "raw_variant": "raw_sam",
            "summary_rows": tracking_rows,
            "aggregate": tracking_aggregate,
        },
        "semantic_map": {
            "summary_rows": map_rows,
            "aggregate": map_aggregate,
        },
        "memory": {
            "scope": "post_tracking_object_map_write_only",
            "sam_hidden_memory_modified": 0,
            "event_count": len(memory_rows),
        },
        "persistent_memory": {
            "enabled": int(include_v1),
            "scope": "causal_sam_observation_to_persistent_object_id",
            "sam_hidden_memory_modified": 0,
            "event_count": len(persistent_memory_rows),
            "object_count_rows": len(persistent_object_rows),
        },
        "decision": _decision_summary(
            tracking_rows=tracking_rows,
            map_rows=map_rows,
        ),
        "outputs": {},
    }
    summary_path = run.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf8",
    )
    output_paths = {
        "summary": summary_path,
        "tracking_summary": _write_csv(
            run.output_dir / "tracking_summary.csv",
            tracking_rows,
        ),
        "tracking_objects": _write_csv(
            run.output_dir / "tracking_objects.csv",
            tracking_object_rows,
        ),
        "tracking_frames": _write_csv(
            run.output_dir / "tracking_frames.csv",
            tracking_frame_rows,
        ),
        "map_summary": _write_csv(
            run.output_dir / "map_summary.csv",
            map_rows,
        ),
        "map_objects": _write_csv(
            run.output_dir / "map_objects.csv",
            map_object_rows,
        ),
        "map_duplicates": _write_csv(
            run.output_dir / "map_duplicates.csv",
            map_duplicate_rows,
        ),
        "memory_events": _write_csv(
            run.output_dir / "memory_events.csv",
            memory_rows,
        ),
    }
    if include_v1:
        output_paths["persistent_memory_events"] = _write_csv(
            run.output_dir / "persistent_memory_events.csv",
            persistent_memory_rows,
        )
        output_paths["persistent_objects"] = _write_csv(
            run.output_dir / "persistent_objects.csv",
            persistent_object_rows,
        )
    summary["outputs"] = {name: str(path) for name, path in output_paths.items()}
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf8",
    )
    copyable = run.output_dir / "copyable_result.txt"
    _write_copyable(copyable, summary)
    print(f"summary={summary_path}")
    print(f"copyable_result={copyable}")


def _evaluate_clip(
    data: LearnedPoseConfig,
    clip: ClipConfig,
    run: EvaluationRunConfig,
) -> dict[str, object]:
    path = cache_path(data, clip)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing completed feature cache for {clip.name}: {path}. "
            "Run commands_v0_baseline.txt first."
        )
    payload = load_feature_cache(path)
    _validate_payload(payload, clip, run)
    frames = tuple(int(value) for value in payload["frame_indices"])
    track_ids = tuple(int(value) for value in payload["sam_track_ids"])
    track_prompts = tuple(str(value) for value in payload["sam_track_prompts"])

    output_variants = payload["tracking_variant_masks_output"]
    output_scores = payload["tracking_variant_scores"]
    raw_output = _tensor(output_variants[run.raw_cache_variant]).bool()
    v6_output = _tensor(output_variants[run.v6_cache_variant]).bool()
    raw_scores = _tensor(output_scores[run.raw_cache_variant]).float()
    v6_scores = _tensor(output_scores[run.v6_cache_variant]).float()
    ground_truth = load_ground_truth_instances(
        data.manifest,
        scene_id=str(payload["scene_id"]),
        frame_indices=frames,
        output_size=(int(raw_output.shape[-2]), int(raw_output.shape[-1])),
        prompts=tuple(str(value) for value in payload["instance_prompts"]),
        prompt_label_aliases=run.prompt_label_aliases,
    )
    base_tracking = evaluate_tracking_variants(
        scene_id=str(payload["scene_id"]),
        clip_name=str(payload["clip_name"]),
        frame_indices=frames,
        variant_masks={
            "raw_sam": raw_output,
            "v6_geometry": v6_output,
        },
        variant_scores={
            "raw_sam": raw_scores,
            "v6_geometry": v6_scores,
        },
        raw_variant="raw_sam",
        track_ids=track_ids,
        track_prompts=track_prompts,
        ground_truth=ground_truth,
        config=run.tracking,
        prompt_label_aliases=run.prompt_label_aliases,
    )
    oracle_output, oracle_scores = _build_oracle_variant(
        ground_truth.masks,
        assignments=base_tracking["assignments"],
        sequence_shape=tuple(raw_output.shape),
    )
    tracking = evaluate_tracking_variants(
        scene_id=str(payload["scene_id"]),
        clip_name=str(payload["clip_name"]),
        frame_indices=frames,
        variant_masks={
            "raw_sam": raw_output,
            "v6_geometry": v6_output,
            "gt_mask_oracle": oracle_output,
        },
        variant_scores={
            "raw_sam": raw_scores,
            "v6_geometry": v6_scores,
            "gt_mask_oracle": oracle_scores,
        },
        raw_variant="raw_sam",
        track_ids=track_ids,
        track_prompts=track_prompts,
        ground_truth=ground_truth,
        config=run.tracking,
        prompt_label_aliases=run.prompt_label_aliases,
    )

    stream_variants = payload["tracking_variant_masks_stream"]
    stream_scores = payload["tracking_variant_scores"]
    raw_stream = _tensor(stream_variants[run.raw_cache_variant]).bool()
    v6_stream = _tensor(stream_variants[run.v6_cache_variant]).bool()
    raw_stream_scores = _tensor(stream_scores[run.raw_cache_variant]).float()
    v6_stream_scores = _tensor(stream_scores[run.v6_cache_variant]).float()
    processed_size = tuple(int(value) for value in payload["image_size"])
    gt_stream = load_ground_truth_stream_masks(
        data.manifest,
        scene_id=str(payload["scene_id"]),
        frame_indices=frames,
        instance_ids=ground_truth.instance_ids,
        processed_size=processed_size,
        image_mode=str(payload["image_mode"]),
    )
    oracle_stream, _ = _build_oracle_variant(
        gt_stream,
        assignments=tracking["assignments"],
        sequence_shape=tuple(raw_stream.shape),
    )
    confidence = _map_confidence(payload["baseline_world_confidence"])
    aligned_points = apply_similarity(
        payload["baseline_world_points"],
        scale=float(payload["point_alignment_scale"]),
        rotation=payload["point_alignment_rotation"],
        translation=payload["point_alignment_translation"],
    )
    target_points = _tensor(payload["target_world_points"]).float()
    memory = build_object_memory_write_policy(
        masks=v6_stream,
        scores=v6_stream_scores,
        geometry_confidence=confidence,
        frame_indices=frames,
        track_ids=track_ids,
        variant="v6_memory_gate",
        config=run.memory,
    )
    map_inputs = (
        ("raw_sam", raw_stream, raw_stream_scores, None),
        ("v6_geometry", v6_stream, v6_stream_scores, None),
        ("gt_mask_oracle", oracle_stream, oracle_stream.new_ones(oracle_stream.shape[:2], dtype=torch.float32), None),
        (
            "v6_memory_gate",
            v6_stream,
            v6_stream_scores,
            memory["map_write_mask"],
        ),
    )
    map_summary_rows: list[dict[str, object]] = []
    map_object_rows: list[dict[str, object]] = []
    map_duplicate_rows: list[dict[str, object]] = []
    for variant, masks, scores, write_mask in map_inputs:
        result = evaluate_semantic_object_map(
            scene_id=str(payload["scene_id"]),
            clip_name=str(payload["clip_name"]),
            variant=variant,
            map_policy=(
                "object_memory_short_long_gate"
                if write_mask is not None
                else "all_visible_observations"
            ),
            aligned_world_points=aligned_points,
            target_world_points=target_points,
            confidence=confidence,
            predicted_masks=masks,
            track_scores=scores,
            gt_masks=gt_stream,
            gt_instance_ids=ground_truth.instance_ids,
            gt_labels=ground_truth.labels,
            assignments=tracking["assignments"],
            map_write_mask=write_mask,
            track_ids=track_ids,
            config=run.map_metrics,
        )
        map_summary_rows.append(result["summary"])
        map_object_rows.extend(result["object_rows"])
        map_duplicate_rows.extend(result["duplicate_rows"])

    persistent_memory_result: dict[str, object] = {
        "events": [],
        "objects": [],
        "tracking": {
            "summary_rows": [],
            "object_rows": [],
            "frame_rows": [],
        },
        "map": {
            "summary_rows": [],
            "object_rows": [],
            "duplicate_rows": [],
        },
    }
    if run.include_v1_memory:
        persistent_confidence = normalize_confidence(confidence)
        appearance = None
        if run.use_cached_appearance and "appearance" in payload:
            candidate = payload["appearance"]
            if torch.is_tensor(candidate):
                candidate = candidate.detach().float().cpu()
                if candidate.ndim == 3 and tuple(candidate.shape[:2]) == tuple(
                    raw_stream.shape[:2]
                ):
                    appearance = candidate
        persistent_memory_model = PersistentObjectMemory(run.persistent_memory)
        persistent_result = persistent_memory_model.process_sequence(
            world_points=payload["baseline_world_points"],
            confidence=persistent_confidence,
            masks=raw_stream,
            track_scores=raw_stream_scores,
            frame_indices=frames,
            track_ids=track_ids,
            track_prompts=track_prompts,
            appearance=appearance,
        )
        object_ids = tuple(int(value) for value in persistent_result["object_ids"])
        object_prompts = {
            int(row["object_id"]): str(row["category"])
            for row in persistent_result["objects"]
        }
        persistent_output = collapse_persistent_tracks(
            masks=raw_output,
            scores=raw_scores,
            persistent_object_ids=persistent_result["persistent_object_ids"],
            object_ids=object_ids,
            object_prompts=object_prompts,
        )
        persistent_stream = collapse_persistent_tracks(
            masks=raw_stream,
            scores=raw_stream_scores,
            persistent_object_ids=persistent_result["persistent_object_ids"],
            object_ids=object_ids,
            object_prompts=object_prompts,
        )
        persistent_tracking = evaluate_tracking_variants(
            scene_id=str(payload["scene_id"]),
            clip_name=str(payload["clip_name"]),
            frame_indices=frames,
            variant_masks={"v1_object_memory": persistent_output["masks"]},
            variant_scores={"v1_object_memory": persistent_output["scores"]},
            raw_variant="v1_object_memory",
            track_ids=object_ids,
            track_prompts=tuple(str(value) for value in persistent_output["prompts"]),
            ground_truth=ground_truth,
            config=run.tracking,
            prompt_label_aliases=run.prompt_label_aliases,
        )
        persistent_write_mask = _collapse_object_write_mask(
            persistent_result["map_write_mask"],
            persistent_result["persistent_object_ids"],
            object_ids,
        )
        persistent_map = evaluate_semantic_object_map(
            scene_id=str(payload["scene_id"]),
            clip_name=str(payload["clip_name"]),
            variant="v1_object_memory",
            map_policy="persistent_object_memory",
            aligned_world_points=aligned_points,
            target_world_points=target_points,
            confidence=confidence,
            predicted_masks=persistent_stream["masks"],
            track_scores=persistent_stream["scores"],
            gt_masks=gt_stream,
            gt_instance_ids=ground_truth.instance_ids,
            gt_labels=ground_truth.labels,
            assignments=persistent_tracking["assignments"],
            map_write_mask=persistent_write_mask,
            track_ids=object_ids,
            config=run.map_metrics,
        )
        persistent_objects = []
        for row in persistent_result["objects"]:
            current = dict(row)
            current.update(
                {
                    "scene_id": str(payload["scene_id"]),
                    "clip": str(payload["clip_name"]),
                    "variant": "v1_object_memory",
                }
            )
            persistent_objects.append(current)
        persistent_events = []
        for row in persistent_result["events"]:
            current = dict(row)
            current.update(
                {
                    "scene_id": str(payload["scene_id"]),
                    "clip": str(payload["clip_name"]),
                    "variant": "v1_object_memory",
                }
            )
            persistent_events.append(current)
        persistent_memory_result = {
            "events": persistent_events,
            "objects": persistent_objects,
            "tracking": persistent_tracking,
            "map": {
                "summary_rows": [persistent_map["summary"]],
                "object_rows": persistent_map["object_rows"],
                "duplicate_rows": persistent_map["duplicate_rows"],
            },
            "object_count": int(persistent_result["persistent_object_count"]),
            "track_to_object": persistent_result["track_to_object"],
        }

    return {
        "clip": {
            "scene_id": str(payload["scene_id"]),
            "clip_name": str(payload["clip_name"]),
            "frames": frames,
            "cache": str(path),
            "raw_cache_variant": run.raw_cache_variant,
            "v6_cache_variant": run.v6_cache_variant,
            "eligible_gt_objects": len(ground_truth.instance_ids),
            "assignment_count": len(tracking["assignments"]),
        },
        "tracking": tracking,
        "map": {
            "summary_rows": map_summary_rows,
            "object_rows": map_object_rows,
            "duplicate_rows": map_duplicate_rows,
        },
        "memory": memory,
        "persistent_memory": persistent_memory_result,
    }


def _collapse_object_write_mask(
    write_mask: torch.Tensor,
    persistent_object_ids: torch.Tensor,
    object_ids: Sequence[int],
) -> torch.Tensor:
    writes = write_mask.detach().bool().cpu()
    ids = persistent_object_ids.detach().long().cpu()
    if tuple(writes.shape) != tuple(ids.shape):
        raise ValueError("Persistent write mask and IDs have different shapes.")
    output = torch.zeros(writes.shape[0], len(object_ids), dtype=torch.bool)
    for index, object_id in enumerate(object_ids):
        output[:, index] = (writes & (ids == int(object_id))).any(dim=1)
    return output


def _build_oracle_variant(
    target_masks: torch.Tensor,
    *,
    assignments: Sequence[Mapping[str, object]],
    sequence_shape: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(sequence_shape) != 4:
        raise ValueError("Oracle sequence shape must be [S,K,H,W].")
    sequence, tracks, height, width = sequence_shape
    target_masks = target_masks.detach().cpu().bool()
    if target_masks.ndim != 4 or tuple(target_masks.shape[0:1]) != (sequence,):
        raise ValueError("Oracle target masks have an incompatible sequence.")
    output = torch.zeros(sequence, tracks, height, width, dtype=torch.bool)
    scores = torch.zeros(sequence, tracks, dtype=torch.float32)
    for row in assignments:
        slot = int(row["slot"])
        target = int(row["gt_index"])
        if not 0 <= slot < tracks or not 0 <= target < target_masks.shape[1]:
            raise ValueError("Raw assignment points outside oracle dimensions.")
        current = target_masks[:, target]
        if tuple(current.shape[-2:]) != (height, width):
            raise ValueError("Oracle masks do not match prediction resolution.")
        output[:, slot] = current
        scores[:, slot] = current.flatten(1).any(dim=1).float()
    return output, scores


def _validate_payload(
    payload: Mapping[str, object],
    clip: ClipConfig,
    run: EvaluationRunConfig,
) -> None:
    required = (
        "clip_name",
        "scene_id",
        "frame_indices",
        "instance_prompts",
        "sam_track_ids",
        "sam_track_prompts",
        "baseline_world_points",
        "baseline_world_confidence",
        "target_world_points",
        "point_alignment_scale",
        "point_alignment_rotation",
        "point_alignment_translation",
        "tracking_variant_masks_output",
        "tracking_variant_masks_stream",
        "tracking_variant_scores",
        "image_size",
        "image_mode",
    )
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError(f"Cache {clip.name} lacks semantic-evaluation fields: {missing}")
    if str(payload["clip_name"]) != clip.name or str(payload["scene_id"]) != clip.scene_id:
        raise ValueError(f"Cache identity differs from config clip {clip.name}.")
    frames = tuple(int(value) for value in payload["frame_indices"])
    if frames != clip.frame_indices:
        raise ValueError(f"Cache frame order differs from config clip {clip.name}.")
    for field in (
        "tracking_variant_masks_output",
        "tracking_variant_masks_stream",
        "tracking_variant_scores",
    ):
        variants = payload[field]
        if not isinstance(variants, Mapping):
            raise ValueError(f"Cache field {field} must be a mapping.")
        for name in (run.raw_cache_variant, run.v6_cache_variant):
            if name not in variants:
                raise ValueError(
                    f"Cache {clip.name} lacks variant {name!r} in {field}. "
                    "Rebuild the V0 cache with geometry prompt variants enabled."
                )
    slot_count = len(payload["sam_track_ids"])
    if len(payload["sam_track_prompts"]) != slot_count:
        raise ValueError("SAM track IDs/prompts have different slot counts.")


def _map_confidence(value: object) -> torch.Tensor:
    confidence = _tensor(value).float().cpu()
    if confidence.ndim == 4 and confidence.shape[-1] == 1:
        confidence = confidence[..., 0]
    if confidence.ndim != 3:
        raise ValueError(
            "baseline_world_confidence must have shape [S,H,W] or [S,H,W,1]."
        )
    return confidence


def _tensor(value: object) -> torch.Tensor:
    if not torch.is_tensor(value):
        return torch.as_tensor(value)
    return value.detach().cpu()


def _load_run(path: str | Path) -> EvaluationRunConfig:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf8") as handle:
        raw = yaml.safe_load(handle) or {}
    section = raw.get("semantic_evaluation", {})
    tracking_raw = section.get("tracking", {}) or {}
    map_raw = section.get("map", {}) or {}
    memory_raw = section.get("object_memory", {}) or {}
    persistent_raw = section.get("persistent_object_memory", {}) or {}
    persistent_association_raw = persistent_raw.get("association", {}) or {}
    aliases = {
        str(key): tuple(str(value) for value in values)
        for key, values in (section.get("prompt_label_aliases", {}) or {}).items()
    }
    tracking = TrackingMetricConfig(
        assignment_min_iou=float(tracking_raw.get("assignment_min_iou", 0.01)),
        identity_iou_threshold=float(
            tracking_raw.get("identity_iou_threshold", 0.50)
        ),
        reentry_iou_threshold=float(
            tracking_raw.get("reentry_iou_threshold", 0.25)
        ),
        reentry_min_gap=int(tracking_raw.get("reentry_min_gap", 2)),
        reentry_window=int(tracking_raw.get("reentry_window", 3)),
        improvement_epsilon=float(
            tracking_raw.get("improvement_epsilon", 1e-6)
        ),
        fragmentation_iou_threshold=float(
            tracking_raw.get("fragmentation_iou_threshold", 0.05)
        ),
    )
    map_metrics = SemanticMapMetricConfig(
        confidence_threshold=float(map_raw.get("confidence_threshold", 0.30)),
        track_score_threshold=float(map_raw.get("track_score_threshold", 0.50)),
        max_points_per_object=int(map_raw.get("max_points_per_object", 4096)),
        distance_chunk_size=int(map_raw.get("distance_chunk_size", 512)),
        fscore_thresholds_m=tuple(
            float(value) for value in map_raw.get("fscore_thresholds_m", [0.05, 0.10])
        ),
        voxel_size_m=float(map_raw.get("voxel_size_m", 0.05)),
        ghost_distance_m=float(map_raw.get("ghost_distance_m", 0.10)),
        duplicate_voxel_iou=float(map_raw.get("duplicate_voxel_iou", 0.25)),
    )
    memory = ObjectMemoryConfig(
        short_term_capacity=int(memory_raw.get("short_term_capacity", 5)),
        long_term_capacity=int(memory_raw.get("long_term_capacity", 4)),
        min_track_score=float(memory_raw.get("min_track_score", 0.50)),
        min_geometry_confidence=float(
            memory_raw.get("min_geometry_confidence", 0.30)
        ),
        min_mask_pixels=int(memory_raw.get("min_mask_pixels", 32)),
        min_area_ratio=float(memory_raw.get("min_area_ratio", 0.35)),
        max_area_ratio=float(memory_raw.get("max_area_ratio", 2.85)),
        long_term_min_track_score=float(
            memory_raw.get("long_term_min_track_score", 0.75)
        ),
        long_term_min_geometry_confidence=float(
            memory_raw.get("long_term_min_geometry_confidence", 0.50)
        ),
        reentry_gap=int(memory_raw.get("reentry_gap", 3)),
    )
    persistent_association = InstanceAssociationConfig(
        center_weight=float(persistent_association_raw.get("center_weight", 0.40)),
        voxel_weight=float(persistent_association_raw.get("voxel_weight", 0.25)),
        chamfer_weight=float(
            persistent_association_raw.get("chamfer_weight", 0.20)
        ),
        appearance_weight=float(
            persistent_association_raw.get("appearance_weight", 0.10)
        ),
        category_weight=float(
            persistent_association_raw.get("category_weight", 0.05)
        ),
        min_match_score=float(
            persistent_association_raw.get("min_match_score", 0.50)
        ),
        max_center_distance_ratio=float(
            persistent_association_raw.get("max_center_distance_ratio", 0.35)
        ),
        center_scale_ratio=float(
            persistent_association_raw.get("center_scale_ratio", 0.12)
        ),
        chamfer_scale_ratio=float(
            persistent_association_raw.get("chamfer_scale_ratio", 0.12)
        ),
        voxel_size_ratio=float(
            persistent_association_raw.get("voxel_size_ratio", 0.02)
        ),
        absolute_voxel_size=float(
            persistent_association_raw.get("absolute_voxel_size", 0.02)
        ),
        max_points_per_comparison=int(
            persistent_association_raw.get("max_points_per_comparison", 256)
        ),
        category_hard_gate=bool(
            persistent_association_raw.get("category_hard_gate", True)
        ),
    )
    persistent_memory = PersistentObjectMemoryConfig(
        max_points_per_object=int(
            persistent_raw.get("max_points_per_object", 4096)
        ),
        min_observation_points=int(
            persistent_raw.get("min_observation_points", 16)
        ),
        min_mask_pixels=int(persistent_raw.get("min_mask_pixels", 32)),
        min_track_score=float(persistent_raw.get("min_track_score", 0.50)),
        min_geometry_confidence=float(
            persistent_raw.get("min_geometry_confidence", 0.30)
        ),
        center_ema_alpha=float(persistent_raw.get("center_ema_alpha", 0.25)),
        same_frame_merge_score=float(
            persistent_raw.get("same_frame_merge_score", 0.78)
        ),
        association=persistent_association,
    )
    run = EvaluationRunConfig(
        source_path=source,
        output_dir=expand_storage_path(
            section.get(
                "output_dir",
                "${VGGT_SAM_STORAGE_ROOT}/outputs/streaming_couping_semantic_map_stage0",
            ),
            base=source.parent,
        ),
        raw_cache_variant=str(
            section.get("raw_cache_variant", RAW_CACHE_VARIANT)
        ),
        v6_cache_variant=str(
            section.get("v6_cache_variant", V6_CACHE_VARIANT)
        ),
        tracking=tracking,
        map_metrics=map_metrics,
        memory=memory,
        persistent_memory=persistent_memory,
        use_cached_appearance=bool(
            persistent_raw.get("use_cached_appearance", False)
        ),
        include_v1_memory=False,
        prompt_label_aliases=aliases,
    )
    _validate_run(run)
    return run


def _validate_run(run: EvaluationRunConfig) -> None:
    if not run.raw_cache_variant or not run.v6_cache_variant:
        raise ValueError("Raw and V6 cache variant names must be non-empty.")
    if run.raw_cache_variant == run.v6_cache_variant:
        raise ValueError("Raw and V6 cache variants must differ.")
    if not 0.0 <= run.tracking.assignment_min_iou <= 1.0:
        raise ValueError("tracking.assignment_min_iou must be in [0,1].")
    if not 0.0 <= run.tracking.identity_iou_threshold <= 1.0:
        raise ValueError("tracking.identity_iou_threshold must be in [0,1].")
    if not 0.0 <= run.tracking.reentry_iou_threshold <= 1.0:
        raise ValueError("tracking.reentry_iou_threshold must be in [0,1].")
    if not 0.0 <= run.tracking.fragmentation_iou_threshold <= 1.0:
        raise ValueError("tracking.fragmentation_iou_threshold must be in [0,1].")


def _select_clips(
    clips: Sequence[ClipConfig],
    requested: str | None,
) -> tuple[ClipConfig, ...]:
    if requested is None:
        return tuple(clips)
    selected = tuple(clip for clip in clips if clip.name == requested)
    if len(selected) != 1:
        names = ", ".join(clip.name for clip in clips)
        raise ValueError(f"Unknown --clip {requested!r}; available: {names}")
    return selected


def _print_clip_result(result: Mapping[str, object]) -> None:
    clip = result["clip"]
    print(
        f"clip={clip['clip_name']} scene={clip['scene_id']} "
        f"frames={len(clip['frames'])} gt_objects={clip['eligible_gt_objects']}"
    )
    for row in result["tracking"]["summary_rows"]:
        print(
            f"  tracking variant={row['variant']} "
                f"IoU={_format_metric(row['mean_frame_iou'])} "
                f"frame_IDF1={_format_metric(row['frame_idf1'])} "
                f"pixel_IDF1={_format_metric(row['pixel_idf1'])} "
                f"reentry={row['reentry_successes']}/{row['reentry_events']} "
                f"fragmentation={row.get('fragmentation_count', 0)} "
                f"merge_errors={row.get('merge_error_count', 0)}"
            )
    for row in result["map"]["summary_rows"]:
        print(
            f"  map variant={row['variant']} "
            f"voxelIoU5cm={_format_metric(row['voxel_iou_5cm'])} "
            f"F5cm={_format_metric(row['fscore_5cm'])} "
            f"ghost={_format_metric(row['ghost_point_ratio'])} "
            f"complete={_format_metric(row['object_completeness_m'])}"
        )
    memory = result["memory"]["summary"]
    print(
        f"  memory writes={memory['accepted_map_writes']}/"
        f"{memory['visible_observations']} "
        f"ratio={_format_metric(memory['write_ratio_of_visible'])}"
    )
    persistent = result.get("persistent_memory", {})
    if persistent.get("tracking", {}).get("summary_rows"):
        for row in persistent["tracking"]["summary_rows"]:
            print(
                f"  tracking variant={row['variant']} "
                f"IoU={_format_metric(row['mean_frame_iou'])} "
                f"frame_IDF1={_format_metric(row['frame_idf1'])} "
                f"pixel_IDF1={_format_metric(row['pixel_idf1'])} "
                f"fragmentation={row.get('fragmentation_count', 0)} "
                f"merge_errors={row.get('merge_error_count', 0)}"
            )
        for row in persistent["map"]["summary_rows"]:
            print(
                f"  map variant={row['variant']} "
                f"voxelIoU5cm={_format_metric(row['voxel_iou_5cm'])} "
                f"F5cm={_format_metric(row['fscore_5cm'])} "
                f"ghost={_format_metric(row['ghost_point_ratio'])}"
            )
        print(
            f"  persistent_objects={persistent.get('object_count', 0)} "
            f"events={len(persistent.get('events', ())) }"
        )


def _aggregate_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    kind: str,
) -> list[dict[str, object]]:
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("variant", "")), []).append(row)
    output = []
    count_fields = {
        "frame_occurrences",
        "eligible_gt_objects",
        "all_visible_gt_objects",
        "matched_tracks",
        "unmatched_tracks",
        "unmatched_gt_objects",
        "frame_idtp",
        "frame_idfp",
        "frame_idfn",
        "pixel_idtp",
        "pixel_idfp",
        "pixel_idfn",
        "id_switches",
        "fragmentation_count",
        "merge_error_count",
        "reentry_events",
        "reentry_successes",
        "duplicate_objects",
        "unmatched_nonempty_tracks",
    }
    for variant, group in grouped.items():
        aggregate: dict[str, object] = {
            "variant": variant,
            "clip_count": len(group),
            "metric_kind": kind,
        }
        keys = sorted({key for row in group for key in row})
        for key in keys:
            if key in {"variant", "scene_id", "clip", "map_policy", "metric_kind"}:
                continue
            values = []
            for row in group:
                value = row.get(key)
                if isinstance(value, bool):
                    values.append(float(value))
                elif isinstance(value, (int, float)) and math.isfinite(float(value)):
                    values.append(float(value))
            if not values:
                continue
            if key in count_fields:
                aggregate[key] = int(round(sum(values)))
            else:
                aggregate[key] = sum(values) / len(values)
        if "frame_idtp" in aggregate:
            aggregate["frame_idf1"] = _f1(
                int(aggregate.get("frame_idtp", 0)),
                int(aggregate.get("frame_idfp", 0)),
                int(aggregate.get("frame_idfn", 0)),
            )
        if "pixel_idtp" in aggregate:
            aggregate["pixel_idf1"] = _f1(
                int(aggregate.get("pixel_idtp", 0)),
                int(aggregate.get("pixel_idfp", 0)),
                int(aggregate.get("pixel_idfn", 0)),
            )
        if "reentry_events" in aggregate:
            events = int(aggregate["reentry_events"])
            aggregate["reentry_success_rate"] = (
                int(aggregate.get("reentry_successes", 0)) / events
                if events
                else float("nan")
            )
        output.append(aggregate)
    return output


def _decision_summary(
    *,
    tracking_rows: Sequence[Mapping[str, object]],
    map_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    tracking = {
        str(row["variant"]): row
        for row in _aggregate_rows(tracking_rows, kind="tracking")
    }
    maps = {
        str(row["variant"]): row
        for row in _aggregate_rows(map_rows, kind="map")
    }
    raw = tracking.get("raw_sam", {})
    v6 = tracking.get("v6_geometry", {})
    raw_map = maps.get("raw_sam", {})
    v6_map = maps.get("v6_geometry", {})
    gate_map = maps.get("v6_memory_gate", {})
    v1 = tracking.get("v1_object_memory", {})
    v1_map = maps.get("v1_object_memory", {})
    return {
        "tracking_v6_mean_iou_delta_vs_raw": _delta(
            v6.get("mean_frame_iou"), raw.get("mean_frame_iou")
        ),
        "tracking_v6_frame_idf1_delta_vs_raw": _delta(
            v6.get("frame_idf1"), raw.get("frame_idf1")
        ),
        "map_v6_voxel_iou_delta_vs_raw": _delta(
            v6_map.get("voxel_iou_5cm"), raw_map.get("voxel_iou_5cm")
        ),
        "map_v6_memory_ghost_delta_vs_v6": _delta(
            gate_map.get("ghost_point_ratio"), v6_map.get("ghost_point_ratio")
        ),
        "tracking_v1_frame_idf1_delta_vs_raw": _delta(
            v1.get("frame_idf1"), raw.get("frame_idf1")
        ),
        "tracking_v1_pixel_idf1_delta_vs_raw": _delta(
            v1.get("pixel_idf1"), raw.get("pixel_idf1")
        ),
        "tracking_v1_fragmentation_count": v1.get(
            "fragmentation_count", float("nan")
        ),
        "tracking_v1_merge_error_count": v1.get(
            "merge_error_count", float("nan")
        ),
        "map_v1_voxel_iou_delta_vs_raw": _delta(
            v1_map.get("voxel_iou_5cm"), raw_map.get("voxel_iou_5cm")
        ),
        "map_v1_ghost_delta_vs_raw": _delta(
            v1_map.get("ghost_point_ratio"), raw_map.get("ghost_point_ratio")
        ),
        "interpretation": (
            "Use GT oracle to separate geometry from tracking; use V6 memory "
            "gate only as a map-write ablation, not as a pose improvement claim."
        ),
    }


def _delta(current: object, baseline: object) -> float:
    if not isinstance(current, (int, float)) or not isinstance(baseline, (int, float)):
        return float("nan")
    if not math.isfinite(float(current)) or not math.isfinite(float(baseline)):
        return float("nan")
    return float(current) - float(baseline)


def _f1(true_positive: int, false_positive: int, false_negative: int) -> float:
    denominator = 2 * true_positive + false_positive + false_negative
    return 2.0 * true_positive / denominator if denominator else float("nan")


def _format_metric(value: object) -> str:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return f"{float(value):.4f}"
    return "nan"


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
) -> Path:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(str(key))
    if not keys:
        keys = ["empty"]
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in keys})
    return path


def _csv_value(value: object) -> object:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True, default=str)
    return value


def _write_copyable(path: Path, summary: Mapping[str, object]) -> None:
    decision = summary["decision"]
    include_v1 = bool(summary.get("persistent_memory", {}).get("enabled", 0))
    begin = (
        "===== SEMANTIC_MAP_STAGE0_WITH_V1_BEGIN ====="
        if include_v1
        else "===== SEMANTIC_MAP_STAGE0_BEGIN ====="
    )
    end = (
        "===== SEMANTIC_MAP_STAGE0_WITH_V1_END ====="
        if include_v1
        else "===== SEMANTIC_MAP_STAGE0_END ====="
    )
    lines = [
        begin,
        f"revision={summary['revision']}",
        f"clips={summary['clip_count']}",
        "pointmap=raw_full_history_world_pointmap",
        "pose_modified=0",
        "raw_assignment_frozen=1",
        "candidate_generation_gt_fields=0",
        "evaluation_gt_fields=1",
        f"tracking_v6_mean_iou_delta_vs_raw={decision['tracking_v6_mean_iou_delta_vs_raw']}",
        f"tracking_v6_frame_idf1_delta_vs_raw={decision['tracking_v6_frame_idf1_delta_vs_raw']}",
        f"map_v6_voxel_iou_delta_vs_raw={decision['map_v6_voxel_iou_delta_vs_raw']}",
        f"map_v6_memory_ghost_delta_vs_v6={decision['map_v6_memory_ghost_delta_vs_v6']}",
        f"tracking_v1_frame_idf1_delta_vs_raw={decision['tracking_v1_frame_idf1_delta_vs_raw']}",
        f"tracking_v1_fragmentation_count={decision['tracking_v1_fragmentation_count']}",
        f"tracking_v1_merge_error_count={decision['tracking_v1_merge_error_count']}",
        f"map_v1_voxel_iou_delta_vs_raw={decision['map_v1_voxel_iou_delta_vs_raw']}",
        f"map_v1_ghost_delta_vs_raw={decision['map_v1_ghost_delta_vs_raw']}",
        f"summary={summary['outputs'].get('summary', '')}",
        end,
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="streaming_couping/configs/v0_baseline.yaml",
    )
    parser.add_argument("--clip", help="Evaluate one configured clip only.")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--include-v1-memory",
        action="store_true",
        help="Also evaluate raw SAM through V1 persistent object memory.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
