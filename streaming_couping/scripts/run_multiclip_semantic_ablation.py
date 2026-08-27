#!/usr/bin/env python3
"""Evaluate frozen semantic-map branches over multiple clips/scenes.

This is an offline evaluator only.  It never reruns StreamVGGT or SAM3 and it
opens ScanNet++ annotations only after all cache/artifact inputs are frozen.

The evaluator deliberately separates tracking from map writing:

``raw``
    Raw SAM cache masks and all visible observations.
``v21`` / ``v22``
    Frozen recovery artifacts, including their serialized map-write branch.
``v23_all_visible``
    V2.3 tracking masks with every visible observation written.  This isolates
    the effect of confidence-aware memory from the V2.2/V2.3 mask branch.
``v23``
    Frozen V2.3 tracking masks and its confidence-aware map-write branch.
``oracle``
    GT masks placed into the frozen raw-SAM slots, used only as a geometry
    upper bound during evaluation.

For more than one clip, the artifact root may either contain a flat
``semantic_map.pt`` (single-clip compatibility) or one directory per clip.
Both ``<root>/<clip_name>/semantic_map.pt`` and the exporter layout
``<root>/<clip_name>/semantic_map/semantic_map.pt`` are accepted.  A
scene/clip nested layout is also accepted.  The command wrapper supplies the
roots used by the retained server outputs.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from streaming_couping.src.learned_pose.cache import cache_path, load_feature_cache
from streaming_couping.src.learned_pose.config import ClipConfig, load_learned_pose_config
from streaming_couping.src.semantic_map import normalize_confidence
from streaming_couping.src.semantic_map_metrics import (
    apply_similarity,
    evaluate_semantic_object_map,
    load_ground_truth_stream_masks,
)
from streaming_couping.src.semantic_tracking_metrics import (
    evaluate_tracking_variants,
    load_ground_truth_instances,
)
from streaming_couping.src.storage import expand_storage_path

from streaming_couping.scripts.run_semantic_map_evaluation import (
    _build_oracle_variant,
    _load_run as _load_evaluation_run,
)


REVISION = "semantic_map_multiclip_ablation_offline_r1"
RAW_VARIANT = "raw_sam"
V21_VARIANT = "v2_1_failure_only_candidate_recovery"
V22_VARIANT = "v2_2_failure_only_geometry_consistency_recovery"
V23_ALL_VISIBLE_VARIANT = "v2_3_tracking_all_visible_map"
V23_VARIANT = "v2_3_failure_only_confidence_aware_voxel_memory"
ORACLE_VARIANT = "gt_mask_oracle"

SUPPORTED_VARIANTS = {
    "raw",
    "oracle",
    "v21",
    "v22",
    "v23_all_visible",
    "v23",
}

TRACKING_METRICS = (
    "mean_frame_iou",
    "positive_iou",
    "tracking_recall_025",
    "tracking_recall_050",
    "frame_idf1",
    "pixel_idf1",
    "id_switches",
    "fragmentation_count",
    "merge_error_count",
    "reentry_events",
    "reentry_successes",
    "reentry_success_rate",
    "improved_frame_ratio_vs_raw",
    "worsened_frame_ratio_vs_raw",
)

MAP_METRICS = (
    "object_accuracy_m",
    "object_completeness_m",
    "symmetric_distance_m",
    "paired_rmse_m",
    "fscore_5cm",
    "fscore_10cm",
    "ghost_point_ratio",
    "voxel_iou_5cm",
    "object_recall_iou25",
    "object_recall_iou50",
    "duplicate_objects",
    "duplicate_object_rate",
)


@dataclass(frozen=True)
class Artifact:
    variant: str
    path: Path
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class Branch:
    key: str
    label: str
    tracking_output: torch.Tensor
    tracking_stream: torch.Tensor
    tracking_scores: torch.Tensor
    map_stream: torch.Tensor
    map_write: torch.Tensor
    artifact_path: Path | None
    artifact_stats: Mapping[str, Any]
    map_policy: str


def main() -> None:
    args = _parse_args()
    data = load_learned_pose_config(args.config)
    evaluation_run = _load_evaluation_run(args.config)
    clips = _select_clips(data.clips, args.clip)
    variants = _parse_variants(args.variants)
    roots = {
        "v21": _optional_path(args.v21_root),
        "v22": _optional_path(args.v22_root),
        "v23": _optional_path(args.v23_root),
    }
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else expand_storage_path(
            "${VGGT_SAM_STORAGE_ROOT}/outputs/"
            "streaming_couping_semantic_map_multiclip_ablation",
            base=Path(args.config).expanduser().resolve().parent,
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("MULTI-CLIP SEMANTIC-MAP OFFLINE ABLATION")
    print(f"config={Path(args.config).expanduser().resolve()}")
    print(f"clips={len(clips)} variants={','.join(variants)}")
    print("models are not rerun; GT is opened only for evaluation")
    print("assignment=raw_sam_frozen_per_clip")

    clip_results: list[dict[str, Any]] = []
    tracking_rows: list[dict[str, Any]] = []
    map_rows: list[dict[str, Any]] = []
    for clip in clips:
        result = _evaluate_clip(
            data=data,
            clip=clip,
            evaluation_run=evaluation_run,
            variants=variants,
            roots=roots,
            raw_cache_variant=args.raw_cache_variant,
        )
        clip_results.append(result["clip"])
        tracking_rows.extend(result["tracking_rows"])
        map_rows.extend(result["map_rows"])
        _print_clip_result(result, variants)

    aggregate = _aggregate_results(
        clip_results=clip_results,
        tracking_rows=tracking_rows,
        map_rows=map_rows,
        primary_variant=(
            V23_VARIANT
            if "v23" in variants
            else _last_candidate_variant(variants)
        ),
    )
    summary: dict[str, Any] = {
        "schema": 1,
        "revision": REVISION,
        "evaluation_gt_fields": 1,
        "candidate_generation_gt_fields": 0,
        "assignment_policy": "raw_sam_frozen_per_clip",
        "clip_count": len(clip_results),
        "scene_count": len({str(row["scene_id"]) for row in clip_results}),
        "clips": clip_results,
        "requested_variants": variants,
        "tracking_summary": tracking_rows,
        "map_summary": map_rows,
        "aggregate": aggregate,
        "outputs": {},
    }
    summary_path = output_dir / "summary.json"
    _write_json(summary_path, summary)
    paths = {
        "summary": summary_path,
        "clip_variant_summary": _write_csv(
            output_dir / "clip_variant_summary.csv",
            _clip_variant_rows(clip_results),
        ),
        "tracking_summary": _write_csv(
            output_dir / "tracking_summary.csv", tracking_rows
        ),
        "map_summary": _write_csv(output_dir / "map_summary.csv", map_rows),
        "aggregate_summary": _write_csv(
            output_dir / "aggregate_summary.csv",
            aggregate["variant_rows"],
        ),
    }
    summary["outputs"] = {name: str(path) for name, path in paths.items()}
    named_summary = output_dir / "semantic_map_multiclip_ablation_summary.json"
    summary["outputs"]["named_summary"] = str(named_summary)
    _write_json(summary_path, summary)
    _write_json(named_summary, summary)
    copyable = output_dir / "copyable_result.txt"
    _write_copyable(copyable, summary)
    print(f"summary={summary_path}")
    print(f"copyable_result={copyable}")
    print(
        f"aggregate_decision={aggregate['decision']} "
        f"primary={aggregate['primary_variant']} "
        f"clips={len(clip_results)} scenes={summary['scene_count']}"
    )


def _evaluate_clip(
    *,
    data,
    clip: ClipConfig,
    evaluation_run,
    variants: Sequence[str],
    roots: Mapping[str, Path | None],
    raw_cache_variant: str,
) -> dict[str, Any]:
    cache_value = cache_path(data, clip)
    payload = load_feature_cache(cache_value)
    _validate_cache_for_evaluation(payload, clip)
    raw_output, raw_stream, raw_scores = _load_raw_branch(
        payload, raw_cache_variant
    )
    frames = tuple(int(value) for value in payload["frame_indices"])
    branches: dict[str, Branch] = {
        "raw": Branch(
            key="raw",
            label=RAW_VARIANT,
            tracking_output=raw_output,
            tracking_stream=raw_stream,
            tracking_scores=raw_scores,
            map_stream=raw_stream,
            map_write=raw_stream.flatten(2).any(dim=2),
            artifact_path=None,
            artifact_stats={},
            map_policy="all_visible_observations",
        )
    }
    for key in variants:
        if key in {"raw", "oracle"}:
            continue
        root_key = "v23" if key == "v23_all_visible" else key
        if root_key not in roots or roots[root_key] is None:
            raise ValueError(
                f"Variant {key!r} needs an artifact root. "
                f"Pass --{root_key}-root."
            )
        artifact_path = _resolve_artifact_path(roots[root_key], clip)
        artifact = _load_artifact(artifact_path, clip)
        branches[key] = _artifact_branch(
            key=key,
            artifact=artifact,
            artifact_path=artifact_path,
            raw_shape=tuple(raw_output.shape),
            raw_stream_shape=tuple(raw_stream.shape),
        )

    output_size = (int(raw_output.shape[-2]), int(raw_output.shape[-1]))
    ground_truth = load_ground_truth_instances(
        data.manifest,
        scene_id=str(payload["scene_id"]),
        frame_indices=frames,
        output_size=output_size,
        prompts=tuple(str(value) for value in payload["instance_prompts"]),
        prompt_label_aliases=evaluation_run.prompt_label_aliases,
    )
    processed_size = tuple(int(value) for value in payload["image_size"])
    gt_stream = load_ground_truth_stream_masks(
        data.manifest,
        scene_id=str(payload["scene_id"]),
        frame_indices=frames,
        instance_ids=ground_truth.instance_ids,
        processed_size=processed_size,
        image_mode=str(payload["image_mode"]),
    )

    # The oracle is an evaluation-only upper bound.  Its slot assignment is
    # inherited from raw SAM, so it does not give the evaluator a new
    # annotation-backed matching advantage.  It is deliberately constructed
    # only after the frozen cache/artifacts have been loaded.
    if "oracle" in variants:
        raw_assignment = evaluate_tracking_variants(
            scene_id=str(payload["scene_id"]),
            clip_name=str(payload["clip_name"]),
            frame_indices=frames,
            variant_masks={RAW_VARIANT: raw_output},
            variant_scores={RAW_VARIANT: raw_scores},
            raw_variant=RAW_VARIANT,
            track_ids=tuple(int(value) for value in payload["sam_track_ids"]),
            track_prompts=tuple(
                str(value) for value in payload["sam_track_prompts"]
            ),
            ground_truth=ground_truth,
            config=evaluation_run.tracking,
            prompt_label_aliases=evaluation_run.prompt_label_aliases,
        )
        oracle_output, oracle_scores = _build_oracle_variant(
            ground_truth.masks,
            assignments=raw_assignment["assignments"],
            sequence_shape=tuple(raw_output.shape),
        )
        oracle_stream, _ = _build_oracle_variant(
            gt_stream,
            assignments=raw_assignment["assignments"],
            sequence_shape=tuple(raw_stream.shape),
        )
        branches["oracle"] = Branch(
            key="oracle",
            label=ORACLE_VARIANT,
            tracking_output=oracle_output,
            tracking_stream=oracle_stream,
            tracking_scores=oracle_scores,
            map_stream=oracle_stream,
            map_write=oracle_stream.flatten(2).any(dim=2),
            artifact_path=None,
            artifact_stats={
                "gt_mask_oracle": 1,
                "candidate_generation_gt_fields": 0,
            },
            map_policy="gt_mask_oracle_all_visible_observations",
        )

    variant_masks = {
        branch.label: branch.tracking_output for branch in branches.values()
    }
    variant_scores = {
        branch.label: branch.tracking_scores for branch in branches.values()
    }
    tracking = evaluate_tracking_variants(
        scene_id=str(payload["scene_id"]),
        clip_name=str(payload["clip_name"]),
        frame_indices=frames,
        variant_masks=variant_masks,
        variant_scores=variant_scores,
        raw_variant=RAW_VARIANT,
        track_ids=tuple(int(value) for value in payload["sam_track_ids"]),
        track_prompts=tuple(str(value) for value in payload["sam_track_prompts"]),
        ground_truth=ground_truth,
        config=evaluation_run.tracking,
        prompt_label_aliases=evaluation_run.prompt_label_aliases,
    )
    aligned_points = apply_similarity(
        payload["baseline_world_points"],
        scale=float(payload["point_alignment_scale"]),
        rotation=payload["point_alignment_rotation"],
        translation=payload["point_alignment_translation"],
    )
    confidence = normalize_confidence(payload["baseline_world_confidence"])
    common_map = dict(
        scene_id=str(payload["scene_id"]),
        clip_name=str(payload["clip_name"]),
        aligned_world_points=aligned_points,
        target_world_points=_tensor(payload["target_world_points"]).float(),
        confidence=confidence,
        gt_masks=gt_stream,
        gt_instance_ids=ground_truth.instance_ids,
        gt_labels=ground_truth.labels,
        assignments=tracking["assignments"],
        track_ids=tuple(int(value) for value in payload["sam_track_ids"]),
        config=evaluation_run.map_metrics,
    )
    evaluated_map_rows: list[dict[str, Any]] = []
    for branch in branches.values():
        evaluated_map_rows.append(
            dict(
                evaluate_semantic_object_map(
                    variant=branch.label,
                    map_policy=branch.map_policy,
                    predicted_masks=branch.map_stream,
                    track_scores=branch.tracking_scores,
                    map_write_mask=branch.map_write,
                    **common_map,
                )["summary"]
            )
        )

    tracking_rows = [dict(row) for row in tracking["summary_rows"]]
    tracking_by_label = {str(row["variant"]): row for row in tracking_rows}
    map_by_label = {str(row["variant"]): row for row in evaluated_map_rows}
    clip_variants: dict[str, Any] = {}
    for key, branch in branches.items():
        tracking_row = tracking_by_label[branch.label]
        map_row = map_by_label[branch.label]
        clip_variants[key] = {
            "tracking": tracking_row,
            "map": map_row,
            "artifact": str(branch.artifact_path) if branch.artifact_path else None,
            "artifact_stats": _json_safe(dict(branch.artifact_stats)),
        }
    clip_result = {
        "scene_id": str(payload["scene_id"]),
        "clip_name": str(payload["clip_name"]),
        "split": str(clip.split),
        "frames": frames,
        "frame_count": len(frames),
        "cache": str(cache_value),
        "variants": clip_variants,
    }
    return {
        "clip": clip_result,
        "tracking_rows": tracking_rows,
        "map_rows": evaluated_map_rows,
    }


def _artifact_branch(
    *,
    key: str,
    artifact: Artifact,
    artifact_path: Path,
    raw_shape: tuple[int, ...],
    raw_stream_shape: tuple[int, ...],
) -> Branch:
    value = artifact.payload
    expected_identity = {
        "v21": V21_VARIANT,
        "v22": V22_VARIANT,
        "v23_all_visible": V23_VARIANT,
        "v23": V23_VARIANT,
    }[key]
    actual_identity = str(value.get("identity_mode", ""))
    if actual_identity != expected_identity:
        raise ValueError(
            f"{key} expects identity_mode={expected_identity!r}, got "
            f"{actual_identity!r}: {artifact_path}"
        )
    output = _tensor(value["tracking_masks_output"]).bool()
    stream = _tensor(value["tracking_masks_stream"]).bool()
    scores = _tensor(value["tracking_scores"]).float()
    if tuple(output.shape) != raw_shape:
        raise ValueError(
            f"{key} artifact output shape {tuple(output.shape)} differs "
            f"from raw {raw_shape}: {artifact_path}"
        )
    if tuple(stream.shape) != raw_stream_shape:
        raise ValueError(
            f"{key} artifact stream shape {tuple(stream.shape)} differs "
            f"from raw {raw_stream_shape}: {artifact_path}"
        )
    if tuple(scores.shape) != raw_shape[:2]:
        raise ValueError(
            f"{key} artifact score shape {tuple(scores.shape)} is invalid: "
            f"{artifact_path}"
        )
    if key == "v23_all_visible":
        map_stream = stream
        map_write = stream.flatten(2).any(dim=2)
        label = V23_ALL_VISIBLE_VARIANT
        policy = "v23_tracking_masks_all_visible_observations"
    else:
        map_stream = _tensor(value.get("map_masks_stream", stream)).bool()
        map_write = _tensor(
            value.get("map_write_mask", map_stream.flatten(2).any(dim=2))
        ).bool()
        if tuple(map_stream.shape) != raw_stream_shape:
            raise ValueError(
                f"{key} artifact map stream shape {tuple(map_stream.shape)} "
                f"differs from raw {raw_stream_shape}: {artifact_path}"
            )
        if tuple(map_write.shape) != raw_stream_shape[:2]:
            raise ValueError(
                f"{key} artifact map-write shape {tuple(map_write.shape)} "
                f"is invalid: {artifact_path}"
            )
        if key == "v21":
            label = V21_VARIANT
            policy = "v21_failure_only_candidate_recovery"
        elif key == "v22":
            label = V22_VARIANT
            policy = "v22_failure_only_world_geometry_validation"
        elif key == "v23":
            label = V23_VARIANT
            policy = "v23_confidence_aware_voxel_memory_map_writes"
        else:
            raise ValueError(f"Unhandled ablation variant {key!r}.")
    return Branch(
        key=key,
        label=label,
        tracking_output=output,
        tracking_stream=stream,
        tracking_scores=scores,
        map_stream=map_stream,
        map_write=map_write,
        artifact_path=artifact_path,
        artifact_stats=value.get("recovery_stats", {}) or {},
        map_policy=policy,
    )


def _load_artifact(path: Path, clip: ClipConfig) -> Artifact:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Semantic-map artifact is not a mapping: {path}")
    artifact_clip = str(payload.get("clip", ""))
    if artifact_clip and artifact_clip != clip.name:
        raise ValueError(
            f"Artifact clip={artifact_clip!r} does not match configured "
            f"clip={clip.name!r}: {path}"
        )
    required = ("tracking_masks_output", "tracking_masks_stream", "tracking_scores")
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError(f"Artifact {path} lacks fields {missing}.")
    return Artifact(variant=str(payload.get("identity_mode", "unknown")), path=path, payload=payload)


def _load_raw_branch(
    payload: Mapping[str, Any], raw_cache_variant: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    output_variants = payload.get("tracking_variant_masks_output")
    stream_variants = payload.get("tracking_variant_masks_stream")
    score_variants = payload.get("tracking_variant_scores")
    if not isinstance(output_variants, Mapping):
        raise ValueError("Cache lacks tracking_variant_masks_output.")
    if not isinstance(stream_variants, Mapping):
        raise ValueError("Cache lacks tracking_variant_masks_stream.")
    if not isinstance(score_variants, Mapping):
        raise ValueError("Cache lacks tracking_variant_scores.")
    for fields in (output_variants, stream_variants, score_variants):
        if raw_cache_variant not in fields:
            raise ValueError(
                f"Cache lacks raw SAM variant {raw_cache_variant!r}."
            )
    return (
        _tensor(output_variants[raw_cache_variant]).bool(),
        _tensor(stream_variants[raw_cache_variant]).bool(),
        _tensor(score_variants[raw_cache_variant]).float(),
    )


def _validate_cache_for_evaluation(payload: Mapping[str, Any], clip: ClipConfig) -> None:
    if str(payload.get("clip_name", "")) != clip.name:
        raise ValueError(
            f"Cache clip={payload.get('clip_name')!r} differs from {clip.name!r}."
        )
    frames = tuple(int(value) for value in payload.get("frame_indices", ()))
    if frames != tuple(int(value) for value in clip.frame_indices):
        raise ValueError(
            f"Cache frame order differs from configured clip {clip.name!r}."
        )
    required = (
        "scene_id",
        "image_mode",
        "image_size",
        "instance_prompts",
        "sam_track_ids",
        "sam_track_prompts",
        "baseline_world_points",
        "baseline_world_confidence",
        "target_world_points",
        "point_alignment_scale",
        "point_alignment_rotation",
        "point_alignment_translation",
        "stream_images",
    )
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError(f"Cache {clip.name!r} lacks evaluation fields {missing}.")


def _resolve_artifact_path(root: Path, clip: ClipConfig) -> Path:
    root = root.expanduser().resolve()
    candidates = [
        root if root.name == "semantic_map.pt" else root / "semantic_map.pt",
        root / clip.name / "semantic_map.pt",
        root / clip.name / "semantic_map" / "semantic_map.pt",
        root / clip.scene_id / clip.name / "semantic_map.pt",
        root / clip.scene_id / clip.name / "semantic_map" / "semantic_map.pt",
        root / clip.scene_id / "semantic_map.pt",
        root / clip.scene_id / "semantic_map" / "semantic_map.pt",
    ]
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return candidate
    attempted = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        f"No semantic_map.pt found for clip {clip.name!r} under {root}.\n{attempted}"
    )


def _select_clips(clips: Sequence[ClipConfig], names: Sequence[str] | None) -> tuple[ClipConfig, ...]:
    if not names:
        return tuple(clips)
    wanted = tuple(dict.fromkeys(str(name) for name in names))
    selected = tuple(clip for clip in clips if clip.name in wanted)
    missing = [name for name in wanted if name not in {clip.name for clip in selected}]
    if missing:
        raise ValueError(
            f"Requested clips are absent from config: {missing}. "
            "Add them to dataset.clips before running this evaluator."
        )
    return selected


def _parse_variants(value: str) -> tuple[str, ...]:
    variants = tuple(dict.fromkeys(part.strip() for part in value.split(",") if part.strip()))
    if not variants:
        raise ValueError("--variants must contain at least raw.")
    unknown = sorted(set(variants) - SUPPORTED_VARIANTS)
    if unknown:
        raise ValueError(f"Unknown variants {unknown}; supported={sorted(SUPPORTED_VARIANTS)}")
    if "raw" not in variants:
        raise ValueError("The raw branch is mandatory for frozen assignment comparison.")
    return variants


def _optional_path(value: str | None) -> Path | None:
    if value is None or not str(value).strip():
        return None
    return expand_storage_path(str(value))


def _aggregate_results(
    *,
    clip_results: Sequence[Mapping[str, Any]],
    tracking_rows: Sequence[Mapping[str, Any]],
    map_rows: Sequence[Mapping[str, Any]],
    primary_variant: str,
) -> dict[str, Any]:
    variant_rows: list[dict[str, Any]] = []
    all_variant_keys = sorted(
        {
            str(key)
            for clip in clip_results
            for key in (clip.get("variants", {}) or {})
        }
    )
    tracking_by_variant = _group_rows(tracking_rows)
    map_by_variant = _group_rows(map_rows)
    for key in all_variant_keys:
        label = _variant_label(key)
        tracking = tracking_by_variant.get(label, ())
        maps = map_by_variant.get(label, ())
        row: dict[str, Any] = {"variant": key, "label": label}
        for metric in TRACKING_METRICS:
            _add_aggregate_fields(row, f"tracking_{metric}", tracking, metric)
        for metric in MAP_METRICS:
            _add_aggregate_fields(row, f"map_{metric}", maps, metric)
        row["clip_count"] = int(
            sum(1 for clip in clip_results if key in (clip.get("variants", {}) or {}))
        )
        row["scene_count"] = int(
            len(
                {
                    str(clip["scene_id"])
                    for clip in clip_results
                    if key in (clip.get("variants", {}) or {})
                }
            )
        )
        variant_rows.append(row)

    comparisons = _compare_to_raw(clip_results, primary_variant)
    by_key = {str(row["variant"]): row for row in variant_rows}
    decision = _aggregate_decision(by_key, primary_variant)
    return {
        "primary_variant": primary_variant,
        "variant_rows": variant_rows,
        "comparisons_vs_raw": comparisons,
        "decision": decision,
        "decision_definition": (
            "GO only when the primary branch is no worse than raw in macro "
            "tracking IoU/IDF1, voxel IoU, F5cm and ghost rate."
        ),
    }


def _add_aggregate_fields(
    row: dict[str, Any], prefix: str, rows: Sequence[Mapping[str, Any]], metric: str
) -> None:
    values = [_finite_float(item.get(metric)) for item in rows]
    values = [value for value in values if value is not None]
    row[f"{prefix}_mean"] = _mean(values)
    row[f"{prefix}_std"] = _std(values)
    row[f"{prefix}_min"] = min(values) if values else float("nan")
    row[f"{prefix}_max"] = max(values) if values else float("nan")
    row[f"{prefix}_count"] = len(values)


def _compare_to_raw(
    clip_results: Sequence[Mapping[str, Any]], primary_variant: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for clip in clip_results:
        variants = clip.get("variants", {}) or {}
        raw = variants.get("raw")
        candidate = variants.get(primary_variant)
        if not raw or not candidate:
            continue
        raw_tracking = raw["tracking"]
        cand_tracking = candidate["tracking"]
        raw_map = raw["map"]
        cand_map = candidate["map"]
        rows.append(
            {
                "scene_id": clip["scene_id"],
                "clip_name": clip["clip_name"],
                "primary_variant": primary_variant,
                "delta_mean_frame_iou": _delta(cand_tracking, raw_tracking, "mean_frame_iou"),
                "delta_frame_idf1": _delta(cand_tracking, raw_tracking, "frame_idf1"),
                "delta_pixel_idf1": _delta(cand_tracking, raw_tracking, "pixel_idf1"),
                "delta_reentry_success_rate": _delta(cand_tracking, raw_tracking, "reentry_success_rate"),
                "delta_voxel_iou_5cm": _delta(cand_map, raw_map, "voxel_iou_5cm"),
                "delta_fscore_5cm": _delta(cand_map, raw_map, "fscore_5cm"),
                "delta_ghost_rate": _delta(cand_map, raw_map, "ghost_point_ratio"),
                "tracking_nonregression": int(
                    _ge(cand_tracking.get("mean_frame_iou"), raw_tracking.get("mean_frame_iou"))
                    and _ge(cand_tracking.get("frame_idf1"), raw_tracking.get("frame_idf1"))
                ),
                "map_nonregression": int(
                    _ge(cand_map.get("voxel_iou_5cm"), raw_map.get("voxel_iou_5cm"))
                    and _ge(cand_map.get("fscore_5cm"), raw_map.get("fscore_5cm"))
                    and _le(cand_map.get("ghost_point_ratio"), raw_map.get("ghost_point_ratio"))
                ),
            }
        )
    return rows


def _aggregate_decision(rows: Mapping[str, Mapping[str, Any]], primary: str) -> str:
    raw = rows.get("raw")
    candidate = rows.get(primary)
    if raw is None or candidate is None:
        return "NO_GO"
    checks = (
        _ge(candidate.get("tracking_mean_frame_iou_mean"), raw.get("tracking_mean_frame_iou_mean")),
        _ge(candidate.get("tracking_frame_idf1_mean"), raw.get("tracking_frame_idf1_mean")),
        _ge(candidate.get("map_voxel_iou_5cm_mean"), raw.get("map_voxel_iou_5cm_mean")),
        _ge(candidate.get("map_fscore_5cm_mean"), raw.get("map_fscore_5cm_mean")),
        _le(candidate.get("map_ghost_point_ratio_mean"), raw.get("map_ghost_point_ratio_mean")),
    )
    return "GO" if all(checks) else "NO_GO"


def _clip_variant_rows(clip_results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for clip in clip_results:
        for key, value in (clip.get("variants", {}) or {}).items():
            row: dict[str, Any] = {
                "scene_id": clip["scene_id"],
                "clip_name": clip["clip_name"],
                "split": clip.get("split", ""),
                "variant": key,
                "artifact": value.get("artifact"),
            }
            row.update(
                {
                    f"tracking_{metric}": value["tracking"].get(metric)
                    for metric in TRACKING_METRICS
                }
            )
            row.update(
                {
                    f"map_{metric}": value["map"].get(metric)
                    for metric in MAP_METRICS
                }
            )
            stats = value.get("artifact_stats", {}) or {}
            for name in (
                "recovery_trigger_count",
                "accepted_recovery_count",
                "geometry_validation_accept_count",
                "geometry_validation_reject_count",
                "memory_update_count",
                "low_confidence_reject_count",
            ):
                if name in stats:
                    row[name] = stats[name]
            rows.append(row)
    return rows


def _group_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["variant"]), []).append(row)
    return grouped


def _variant_label(key: str) -> str:
    return {
        "raw": RAW_VARIANT,
        "oracle": ORACLE_VARIANT,
        "v21": V21_VARIANT,
        "v22": V22_VARIANT,
        "v23_all_visible": V23_ALL_VISIBLE_VARIANT,
        "v23": V23_VARIANT,
    }[key]


def _last_candidate_variant(variants: Sequence[str]) -> str:
    for key in reversed(tuple(variants)):
        if key != "raw":
            return key
    return "raw"


def _print_clip_result(result: Mapping[str, Any], variants: Sequence[str]) -> None:
    clip = result["clip"]
    print(f"clip={clip['clip_name']} scene={clip['scene_id']} split={clip['split']}")
    for key in variants:
        value = clip["variants"][key]
        tracking = value["tracking"]
        maps = value["map"]
        print(
            f"  {key} tracking IoU={float(tracking['mean_frame_iou']):.4f} "
            f"frame_IDF1={float(tracking['frame_idf1']):.4f} "
            f"reentry={tracking['reentry_successes']}/{tracking['reentry_events']}"
        )
        print(
            f"    map voxelIoU5cm={float(maps['voxel_iou_5cm']):.4f} "
            f"F5cm={float(maps['fscore_5cm']):.4f} "
            f"ghost={float(maps['ghost_point_ratio']):.4f}"
        )
    raw = clip["variants"]["raw"]
    for key in variants:
        if key == "raw":
            continue
        value = clip["variants"][key]
        print(
            f"    delta[{key}] IoU={_delta(value['tracking'], raw['tracking'], 'mean_frame_iou'):+.4f} "
            f"IDF1={_delta(value['tracking'], raw['tracking'], 'frame_idf1'):+.4f} "
            f"voxelIoU={_delta(value['map'], raw['map'], 'voxel_iou_5cm'):+.4f} "
            f"ghost={_delta(value['map'], raw['map'], 'ghost_point_ratio'):+.4f}"
        )


def _write_copyable(path: Path, summary: Mapping[str, Any]) -> None:
    aggregate = summary["aggregate"]
    lines = [
        "===== MULTICLIP_SEMANTIC_MAP_ABLATION_BEGIN =====",
        f"revision={summary['revision']}",
        f"clip_count={summary['clip_count']}",
        f"scene_count={summary['scene_count']}",
        f"variants={','.join(summary['requested_variants'])}",
        "assignment=raw_sam_frozen_per_clip",
        "gt_role=evaluation_only",
        f"primary_variant={aggregate['primary_variant']}",
        f"decision={aggregate['decision']}",
    ]
    for row in aggregate["variant_rows"]:
        lines.append(
            "variant={} clips={} "
            "tracking_iou_mean={} tracking_idf1_mean={} "
            "map_voxel_iou_5cm_mean={} map_fscore_5cm_mean={} "
            "map_ghost_rate_mean={}".format(
                row["variant"],
                row["clip_count"],
                row.get("tracking_mean_frame_iou_mean"),
                row.get("tracking_frame_idf1_mean"),
                row.get("map_voxel_iou_5cm_mean"),
                row.get("map_fscore_5cm_mean"),
                row.get("map_ghost_point_ratio_mean"),
            )
        )
    lines.append("===== MULTICLIP_SEMANTIC_MAP_ABLATION_END =====")
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if str(key) not in keys:
                keys.append(str(key))
    if not keys:
        keys = ["empty"]
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in keys})
    return path


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(_json_safe(value), sort_keys=True)
    return value


def _json_safe(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _tensor(value: Any) -> torch.Tensor:
    if not torch.is_tensor(value):
        return torch.as_tensor(value)
    return value.detach().cpu()


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0 if values else float("nan")
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _delta(candidate: Mapping[str, Any], raw: Mapping[str, Any], key: str) -> float:
    left = _finite_float(candidate.get(key))
    right = _finite_float(raw.get(key))
    return float("nan") if left is None or right is None else left - right


def _ge(candidate: Any, baseline: Any, epsilon: float = 1e-6) -> bool:
    left, right = _finite_float(candidate), _finite_float(baseline)
    return left is not None and right is not None and left >= right - epsilon


def _le(candidate: Any, baseline: Any, epsilon: float = 1e-6) -> bool:
    left, right = _finite_float(candidate), _finite_float(baseline)
    return left is not None and right is not None and left <= right + epsilon


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="streaming_couping/configs/v0_baseline.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--clip", action="append")
    parser.add_argument(
        "--variants",
        default="raw,v21,v22,v23_all_visible,v23",
        help=(
            "Comma-separated branches: raw,oracle,v21,v22,v23_all_visible,v23. "
            "oracle is evaluation-only and needs no artifact root."
        ),
    )
    parser.add_argument("--v21-root")
    parser.add_argument("--v22-root")
    parser.add_argument("--v23-root")
    parser.add_argument("--raw-cache-variant", default="sam31_online_forward")
    return parser.parse_args()


if __name__ == "__main__":
    main()
