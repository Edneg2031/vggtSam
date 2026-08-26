#!/usr/bin/env python3
"""Test causal DINOv3 patch-to-patch retrieval as a source of geometry.

This is deliberately a feasibility test, not a learned fusion method.  It
uses only frozen StreamVGGT pointmaps, frozen SAM masks/track slots, and
frozen DINOv3 dense patch features to generate matches.  History is strictly
limited to earlier frames.  GT is opened after matching and is used only to
measure the distance of the current raw point and the retrieved historical
point to the current object's aggregated GT surface cloud.

The evaluator compares four retrieval controls:

``correct_track_dino``
    DINO nearest neighbours inside the same SAM slot/track.
``random_same_object``
    Random historical patches inside the same SAM slot.
``shuffled_track_dino``
    DINO nearest neighbours inside a deterministic wrong slot.
``unrestricted_dino``
    DINO nearest neighbours over every earlier object patch.

No XYZ correction is written and no model parameter is updated.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch.nn import functional as F

from streaming_couping.src.dinov3_object_geometry import apply_similarity
from streaming_couping.src.learned_pose.cache import load_feature_cache
from streaming_couping.src.learned_pose.config import load_learned_pose_config
from streaming_couping.scripts.run_dinov3_object_geometry import (
    _load_clip_data,
    _load_ground_truth_bundle,
    _validate_model_compatibility,
)


REVISION = "dinov3_patch_geometry_retrieval_diagnostic_r2"
DEFAULT_STORAGE_ROOT = "/data184/open_source/vggtSam"
DEFAULT_PROTOCOL = "outputs/streaming_couping_multiclip/protocol.yaml"
DEFAULT_FEATURE_DIR = "outputs/streaming_couping_dinov3_object_features"
DEFAULT_OUTPUT_DIR = "outputs/streaming_couping_dinov3_patch_geometry_retrieval"

MODES = (
    "correct_track_dino",
    "random_same_object",
    "shuffled_track_dino",
    "unrestricted_dino",
)
PAIR_CONTROLS = (
    "random_same_object",
    "shuffled_track_dino",
    "unrestricted_dino",
)
TOP_K_VALUES = (1, 3, 5)
SIMILARITY_BINS: tuple[tuple[float, float, str], ...] = (
    (float("-inf"), 0.50, "<0.50"),
    (0.50, 0.60, "0.50-0.60"),
    (0.60, 0.70, "0.60-0.70"),
    (0.70, 0.80, "0.70-0.80"),
    (0.80, 0.90, "0.80-0.90"),
    (0.90, 1.01, "0.90+")
)


@dataclass(frozen=True)
class PatchRecord:
    """One DINO patch with the raw StreamVGGT point at its patch centre."""

    frame: int
    slot: int
    track_id: int
    identity_key: tuple[int, int]
    patch_y: int
    patch_x: int
    pixel_y: int
    pixel_x: int
    feature: torch.Tensor
    point: torch.Tensor
    confidence: float


@dataclass(frozen=True)
class PatchMatch:
    query: PatchRecord
    source: PatchRecord
    cosine: float
    rank: int


def main() -> None:
    args = _parse_args()
    storage_root = Path(
        os.environ.get("VGGT_SAM_STORAGE_ROOT", DEFAULT_STORAGE_ROOT)
    ).expanduser().resolve()
    protocol = _resolve_path(args.protocol, storage_root, DEFAULT_PROTOCOL)
    feature_dir = _resolve_path(args.feature_dir, storage_root, DEFAULT_FEATURE_DIR)
    output_dir = _resolve_path(args.output_dir, storage_root, DEFAULT_OUTPUT_DIR)
    if not protocol.is_file():
        raise FileNotFoundError(f"Multi-scene protocol is missing: {protocol}")
    if not feature_dir.is_dir():
        raise FileNotFoundError(f"DINOv3 feature cache directory is missing: {feature_dir}")

    config = load_learned_pose_config(protocol)
    clips_by_split = {
        split: tuple(clip for clip in config.clips if clip.split == split)
        for split in ("validation", "test")
    }
    if not clips_by_split["validation"] or not clips_by_split["test"]:
        raise ValueError(
            "Patch retrieval requires validation and test clips; got "
            f"validation={len(clips_by_split['validation'])} "
            f"test={len(clips_by_split['test'])}."
        )

    device = _resolve_device(args.device)
    print("DINOv3 PATCH GEOMETRY RETRIEVAL DIAGNOSTIC")
    print(f"protocol={protocol}")
    print(f"feature_dir={feature_dir}")
    print(
        "candidate_generation=causal earlier frames only; same-track and "
        "control restrictions are applied before GT is opened"
    )
    print(
        "GT=offline nearest distance to aggregated object surface cloud; "
        "no XYZ correction, training, or parameter update"
    )
    print(
        f"device={device} max_patches_per_object_frame={args.max_patches_per_object_frame} "
        f"top_k={','.join(str(value) for value in TOP_K_VALUES)} "
        f"mutual_nearest={int(args.mutual_nearest)}"
    )

    condition_rows: list[dict[str, Any]] = []
    similarity_rows: list[dict[str, Any]] = []
    object_rows: list[dict[str, Any]] = []
    scene_rows: list[dict[str, Any]] = []
    clip_summaries: list[dict[str, Any]] = []
    all_match_rows: list[dict[str, Any]] = []
    all_topk_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []

    for split, clips in clips_by_split.items():
        data_values = [
            _load_clip_data(config, clip, feature_dir, confidence_threshold=0.30)
            for clip in clips
        ]
        _validate_model_compatibility(data_values)
        for value in data_values:
            _hide_evaluation_targets(value)
        for data in data_values:
            dino_path = feature_dir / f"{data.name}.pt"
            dino_payload = torch.load(dino_path, map_location="cpu", weights_only=False)
            dense, dense_meta = _load_dense_features(dino_payload, dino_path)
            records, frame_records = _build_patch_records(
                data,
                dense=dense,
                dense_meta=dense_meta,
                max_patches_per_object_frame=int(args.max_patches_per_object_frame),
            )
            # Candidate generation is intentionally completed before GT is
            # opened.  This makes it impossible for an evaluation annotation
            # to affect the retrieval candidates by accident.
            generated_matches = _generate_clip_matches(
                data,
                records=records,
                frame_records=frame_records,
                device=device,
                seed=int(args.seed),
                mutual_nearest=bool(args.mutual_nearest),
            )
            _restore_evaluation_target(data)
            ground_truth = _load_ground_truth_bundle(data)
            result = _analyze_clip(
                data,
                split=split,
                ground_truth=ground_truth,
                records=records,
                generated_matches=generated_matches,
                max_gt_surface_points=int(args.max_gt_surface_points),
            )
            condition_rows.extend(result["condition_rows"])
            similarity_rows.extend(result["similarity_rows"])
            object_rows.extend(result["object_rows"])
            scene_rows.extend(result["scene_rows"])
            paired_rows.extend(result["paired_rows"])
            all_match_rows.extend(result["match_rows"])
            all_topk_rows.extend(result["topk_rows"])
            clip_summaries.append(result["summary"])
            _print_clip_summary(result["summary"])
            _print_paired_summary(result["paired_rows"])

    aggregate_scene_rows = _aggregate_scene_rows(scene_rows)
    aggregate_object_rows = _aggregate_object_rows(object_rows)
    aggregate_paired_rows = _aggregate_paired_rows(paired_rows)
    interpretation = _interpret(
        aggregate_scene_rows,
        aggregate_object_rows,
        aggregate_paired_rows,
    )
    summary = {
        "schema": 1,
        "revision": REVISION,
        "protocol": str(protocol),
        "feature_dir": str(feature_dir),
        "branches": ["raw", *MODES],
        "top_k_values": list(TOP_K_VALUES),
        "validation_scenes": [clip.scene_id for clip in clips_by_split["validation"]],
        "test_scenes": [clip.scene_id for clip in clips_by_split["test"]],
        "candidate_generation": {
            "uses_gt": 0,
            "uses_future_frames": 0,
            "history_policy": "strictly earlier frames in the same clip",
            "same_track_policy": "SAM persistent track-id restriction",
            "unrestricted_policy": "all earlier valid object patches",
            "mutual_nearest_neighbor": int(args.mutual_nearest),
        },
        "evaluation": {
            "uses_gt": 1,
            "metrics": {
                "object_surface": (
                    "nearest distance to aggregated GT object surface cloud"
                ),
                "query_surface": (
                    "distance to the GT point at the queried current pixel"
                ),
            },
            "pointmap_source": "frozen raw StreamVGGT world pointmap",
        },
        "diagnostic_only": 1,
        "clip_summaries": clip_summaries,
        "scene_metrics": scene_rows,
        "aggregate_scene_metrics": aggregate_scene_rows,
        "aggregate_object_metrics": aggregate_object_rows,
        "paired_comparison_metrics": paired_rows,
        "aggregate_paired_comparison_metrics": aggregate_paired_rows,
        "interpretation": interpretation,
        "outputs": {},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output_dir / "summary.json",
        "match_metrics": output_dir / "match_metrics.csv",
        "similarity_bins": output_dir / "similarity_bins.csv",
        "per_object_metrics": output_dir / "per_object_metrics.csv",
        "per_scene_metrics": output_dir / "per_scene_metrics.csv",
        "paired_comparison": output_dir / "paired_comparison.csv",
        "condition_summary": output_dir / "condition_summary.csv",
        "copyable": output_dir / "copyable_result.txt",
    }
    _write_json(paths["summary"], summary)
    _write_csv(paths["match_metrics"], all_match_rows + all_topk_rows)
    _write_csv(paths["similarity_bins"], similarity_rows)
    _write_csv(paths["per_object_metrics"], object_rows + aggregate_object_rows)
    _write_csv(paths["per_scene_metrics"], scene_rows + aggregate_scene_rows)
    _write_csv(paths["paired_comparison"], paired_rows + aggregate_paired_rows)
    _write_csv(paths["condition_summary"], condition_rows)
    summary["outputs"] = {key: str(value) for key, value in paths.items()}
    _write_json(paths["summary"], summary)
    _write_copyable(paths["copyable"], summary)
    print(f"summary={paths['summary']}")
    print(f"match_metrics={paths['match_metrics']}")
    print(f"similarity_bins={paths['similarity_bins']}")
    print(f"per_object_metrics={paths['per_object_metrics']}")
    print(f"per_scene_metrics={paths['per_scene_metrics']}")
    print(f"paired_comparison={paths['paired_comparison']}")
    print(f"condition_summary={paths['condition_summary']}")
    print("decision=DIAGNOSTIC_ONLY")


def _generate_clip_matches(
    data: Any,
    *,
    records: Sequence[PatchRecord],
    frame_records: Mapping[int, Sequence[PatchRecord]],
    device: torch.device,
    seed: int,
    mutual_nearest: bool,
) -> dict[str, list[PatchMatch]]:
    """Generate all retrieval candidates without touching GT.

    The cache uses fixed mask slots, but a slot index is an implementation
    detail rather than an identity.  This function therefore indexes history
    by the persistent SAM track ID carried by each patch record.  The
    shuffled control deliberately redirects each query to the next track's
    history while keeping the same frame chronology and feature budget.
    """

    slot_count = int(data.masks.shape[1])
    track_ids = tuple(int(value) for value in data.track_ids)
    if len(track_ids) != slot_count:
        raise ValueError(
            f"Track-id count {len(track_ids)} does not match mask slots {slot_count}."
        )
    identity_keys = _persistent_identity_keys(track_ids)

    generated_matches: dict[str, list[PatchMatch]] = {
        mode: [] for mode in MODES
    }
    history_by_identity: dict[tuple[int, int], list[PatchRecord]] = {
        identity_key: [] for identity_key in identity_keys
    }
    history_all: list[PatchRecord] = []
    shuffled_identity_for_identity = {
        identity_keys[index]: identity_keys[(index + 1) % slot_count]
        for index in range(slot_count)
    }
    rng = torch.Generator(device="cpu").manual_seed(int(seed))

    for frame in range(int(data.raw_native.shape[0])):
        current = list(frame_records.get(frame, ()))
        current_by_identity: dict[tuple[int, int], list[PatchRecord]] = defaultdict(list)
        for query in current:
            current_by_identity[query.identity_key].append(query)

        for identity_key, queries in sorted(current_by_identity.items()):
            correct = _dino_matches(
                queries,
                history_by_identity.get(identity_key, ()),
                device=device,
                top_k=max(TOP_K_VALUES),
                mutual_nearest=mutual_nearest,
            )
            _extend_match_dict(generated_matches["correct_track_dino"], correct)

            shuffled_identity = shuffled_identity_for_identity[identity_key]
            shuffled = _dino_matches(
                queries,
                history_by_identity.get(shuffled_identity, ()),
                device=device,
                top_k=1,
                mutual_nearest=mutual_nearest,
            )
            _extend_match_dict(generated_matches["shuffled_track_dino"], shuffled)

            random_matches = _random_matches(
                queries,
                history_by_identity.get(identity_key, ()),
                generator=rng,
                top_k=max(TOP_K_VALUES),
            )
            _extend_match_dict(generated_matches["random_same_object"], random_matches)

        unrestricted = _dino_matches(
            current,
            history_all,
            device=device,
            top_k=1,
            mutual_nearest=mutual_nearest,
        )
        _extend_match_dict(generated_matches["unrestricted_dino"], unrestricted)

        # The update happens only after all current queries have been matched,
        # so no mode can retrieve another patch from the current frame.
        for query in current:
            history_by_identity.setdefault(query.identity_key, []).append(query)
            history_all.append(query)

    return generated_matches


def _persistent_identity_keys(
    track_ids: Sequence[int],
) -> tuple[tuple[int, int], ...]:
    """Make safe history keys while preserving valid SAM persistent IDs.

    ``-1`` denotes an empty SAM slot, and malformed caches can contain a
    duplicate ID.  Neither case is allowed to merge unrelated histories.  A
    unique non-negative ID gets key ``(0, track_id)``; invalid or duplicate
    IDs get a slot-local fallback key ``(1, slot)``.
    """

    counts: dict[int, int] = defaultdict(int)
    for track_id in track_ids:
        if int(track_id) >= 0:
            counts[int(track_id)] += 1
    keys: list[tuple[int, int]] = []
    for slot, track_id in enumerate(track_ids):
        track_id = int(track_id)
        if track_id >= 0 and counts[track_id] == 1:
            keys.append((0, track_id))
        else:
            keys.append((1, int(slot)))
    return tuple(keys)


def _hide_evaluation_targets(data: Any) -> None:
    """Make accidental GT access fail closed during candidate generation."""

    hidden = torch.full_like(data.raw_native, float("nan"))
    data.target_metric = hidden.clone()
    data.target_native = hidden.clone()
    data.support = torch.zeros_like(data.confidence, dtype=torch.bool)


def _restore_evaluation_target(data: Any) -> None:
    """Load the frozen target pointmap only after all matches are generated."""

    payload = load_feature_cache(data.cache_path)
    value = torch.as_tensor(payload["target_world_points"]).float().cpu()
    if tuple(value.shape) != tuple(data.raw_native.shape):
        raise ValueError(
            "Frozen target pointmap shape disagrees with the raw pointmap: "
            f"target={tuple(value.shape)} raw={tuple(data.raw_native.shape)}"
        )
    data.target_metric = value


def _analyze_clip(
    data: Any,
    *,
    split: str,
    ground_truth: Any,
    records: Sequence[PatchRecord],
    generated_matches: Mapping[str, Sequence[PatchMatch]],
    max_gt_surface_points: int,
) -> dict[str, Any]:
    gt_masks = ground_truth.stream_masks.detach().bool().cpu()
    gt_clouds = _build_gt_clouds(
        data,
        gt_masks=gt_masks,
        max_points=max_gt_surface_points,
    )
    gt_labels = tuple(str(value) for value in ground_truth.instances.labels)
    print(
        f"  clip={data.name} records={len(records)} "
        f"generated={','.join(f'{key}:{len(value)}' for key, value in generated_matches.items())}"
    )

    raw_by_query: dict[int, float] = {}
    raw_query_by_query: dict[int, float] = {}
    query_target: dict[int, torch.Tensor] = {}
    query_gt: dict[int, int] = {}
    for query in records:
        query_id = id(query)
        gt_index = _query_gt_object(query, gt_masks)
        query_gt[query_id] = gt_index
        if gt_index < 0:
            continue
        target = data.target_metric[
            query.frame, query.pixel_y, query.pixel_x
        ].detach().float().cpu()
        if not bool(torch.isfinite(target).all()):
            continue
        if not gt_clouds[gt_index].numel():
            continue
        raw_by_query[query_id] = float(
            _nearest_distance(query.point[None], gt_clouds[gt_index])[0]
        )
        raw_query_by_query[query_id] = float(
            torch.linalg.vector_norm(query.point.float() - target)
        )
        query_target[query_id] = target

    match_rows: list[dict[str, Any]] = []
    topk_rows: list[dict[str, Any]] = []
    for mode, matches in generated_matches.items():
        grouped: dict[int, list[PatchMatch]] = defaultdict(list)
        for match in matches:
            grouped[id(match.query)].append(match)
        for query_id, query_matches in grouped.items():
            query_matches.sort(key=lambda value: value.rank)
            if query_id not in raw_by_query:
                continue
            for match in query_matches[:1]:
                gt_index = query_gt[query_id]
                match_rows.append(
                    _match_row(
                        data=data,
                        split=split,
                        mode=mode,
                        aggregation="top1",
                        k=1,
                        match=match,
                        gt_index=gt_index,
                        gt_label=gt_labels[gt_index],
                        raw_error=raw_by_query[query_id],
                        raw_query_error=raw_query_by_query[query_id],
                        query_target=query_target[query_id],
                        retrieved_error=None,
                        retrieved_query_error=None,
                    )
                )
            if mode in {"correct_track_dino", "random_same_object"}:
                for k in TOP_K_VALUES:
                    if len(query_matches) < k:
                        continue
                    selected = query_matches[:k]
                    gt_index = query_gt[query_id]
                    aggregate_point = _aggregate_points(
                        [value.source.point for value in selected]
                    )
                    target = query_target[query_id]
                    topk_rows.append(
                        _topk_row(
                            data=data,
                            split=split,
                            mode=mode,
                            k=k,
                            matches=selected,
                            gt_index=gt_index,
                            gt_label=gt_labels[gt_index],
                            raw_error=raw_by_query[query_id],
                            raw_query_error=raw_query_by_query[query_id],
                            query_target=target,
                            retrieved_error=_nearest_distance(
                                aggregate_point[None], gt_clouds[gt_index]
                            )[0],
                            retrieved_query_error=torch.linalg.vector_norm(
                                aggregate_point.float() - target
                            ),
                        )
                    )

    # Fill source errors after all causal matches are frozen.  This keeps the
    # distinction between candidate generation and GT evaluation explicit.
    _fill_retrieved_errors(match_rows, gt_clouds)
    _fill_retrieved_errors(topk_rows, gt_clouds)
    condition_rows = _condition_rows(
        match_rows,
        topk_rows,
        split=split,
        clip=data.name,
    )
    similarity_rows = _similarity_rows(
        match_rows,
        split=split,
        clip=data.name,
    )
    object_rows = _object_metric_rows(
        match_rows,
        topk_rows,
        split=split,
        clip=data.name,
    )
    scene_rows = _scene_metric_rows(
        match_rows,
        topk_rows,
        split=split,
        clip=data.name,
        eligible_queries=len(raw_by_query),
    )
    paired_rows = _paired_comparison_rows(
        match_rows,
        split=split,
        clip=data.name,
        scene_id=data.scene_id,
    )
    summary = {
        "split": split,
        "clip": data.name,
        "scene_id": data.scene_id,
        "records": len(records),
        "eligible_queries": int(len(raw_by_query)),
        "generated_matches": {
            mode: len(values) for mode, values in generated_matches.items()
        },
        "scene_metrics": scene_rows,
    }
    return {
        "summary": summary,
        "match_rows": match_rows,
        "topk_rows": topk_rows,
        "condition_rows": condition_rows,
        "similarity_rows": similarity_rows,
        "object_rows": object_rows,
        "scene_rows": scene_rows,
        "paired_rows": paired_rows,
    }


def _build_patch_records(
    data: Any,
    *,
    dense: torch.Tensor,
    dense_meta: Mapping[str, Any],
    max_patches_per_object_frame: int,
) -> tuple[list[PatchRecord], dict[int, list[PatchRecord]]]:
    if dense.ndim != 4:
        raise ValueError(f"DINO dense features must be [S,h,w,C], got {dense.shape}")
    sequence, patch_height, patch_width, _ = map(int, dense.shape)
    if sequence != int(data.masks.shape[0]):
        raise ValueError("DINO dense feature frame count differs from frozen cache.")
    input_height, input_width = _dino_input_size(
        dense_meta,
        patch_height=patch_height,
        patch_width=patch_width,
        fallback_size=(int(data.masks.shape[-2]), int(data.masks.shape[-1])),
    )
    if input_height < patch_height or input_width < patch_width:
        raise ValueError(
            "DINO metadata input_size must be no smaller than its patch grid: "
            f"input={(input_height, input_width)} grid={(patch_height, patch_width)}"
        )
    patch_masks = F.interpolate(
        F.interpolate(
            data.masks.float().reshape(-1, 1, *data.masks.shape[-2:]),
            size=(input_height, input_width),
            mode="nearest",
        ),
        size=(patch_height, patch_width),
        mode="nearest",
    ).reshape(sequence, int(data.masks.shape[1]), patch_height, patch_width) > 0.5
    feature_values = F.normalize(dense.float(), dim=-1)
    raw = apply_similarity(
        data.raw_native,
        scale=data.scale,
        rotation=data.rotation,
        translation=data.translation,
    ).float().cpu()
    confidence = data.confidence.float().cpu()
    runtime_valid = (
        torch.isfinite(raw).all(dim=-1)
        & torch.isfinite(confidence)
        & (confidence >= 0.30)
    )
    scores = data.scores.float().cpu()
    height, width = (int(value) for value in raw.shape[1:3])
    pixel_y = _patch_centres(patch_height, height, resized_count=input_height)
    pixel_x = _patch_centres(patch_width, width, resized_count=input_width)
    track_ids = tuple(int(value) for value in data.track_ids)
    if len(track_ids) != int(data.masks.shape[1]):
        raise ValueError("DINO patch records cannot resolve SAM track IDs.")
    identity_keys = _persistent_identity_keys(track_ids)
    all_records: list[PatchRecord] = []
    frame_records: dict[int, list[PatchRecord]] = defaultdict(list)
    for frame in range(sequence):
        for slot in range(int(data.masks.shape[1])):
            if not math.isfinite(float(scores[frame, slot])) or float(scores[frame, slot]) < 0.50:
                continue
            coordinates = torch.nonzero(
                patch_masks[frame, slot],
                as_tuple=False,
            )
            coordinates = _subsample_coordinates(
                coordinates,
                limit=max_patches_per_object_frame,
            )
            for coordinate in coordinates:
                patch_y, patch_x = (int(value) for value in coordinate.tolist())
                y = int(pixel_y[patch_y])
                x = int(pixel_x[patch_x])
                if not bool(runtime_valid[frame, y, x]):
                    continue
                feature = feature_values[frame, patch_y, patch_x].detach().cpu()
                point = raw[frame, y, x].detach().cpu()
                if not bool(torch.isfinite(feature).all() and torch.isfinite(point).all()):
                    continue
                record = PatchRecord(
                    frame=frame,
                    slot=slot,
                    track_id=track_ids[slot],
                    identity_key=identity_keys[slot],
                    patch_y=patch_y,
                    patch_x=patch_x,
                    pixel_y=y,
                    pixel_x=x,
                    feature=feature,
                    point=point,
                    confidence=float(confidence[frame, y, x]),
                )
                all_records.append(record)
                frame_records[frame].append(record)
    return all_records, frame_records


def _load_dense_features(
    payload: Mapping[str, Any],
    path: Path,
) -> tuple[torch.Tensor, Mapping[str, Any]]:
    value = payload.get("dense_features")
    if value is None:
        raise ValueError(
            f"DINO cache lacks dense_features: {path}; "
            "rerun commands_analyze_dinov3_patch_geometry_retrieval.txt"
        )
    dense = torch.as_tensor(value).float().cpu()
    if dense.ndim != 4:
        raise ValueError(f"dense_features in {path} must be [S,h,w,C], got {dense.shape}")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        metadata = {}
    return dense, metadata


def _dino_input_size(
    metadata: Mapping[str, Any],
    *,
    patch_height: int,
    patch_width: int,
    fallback_size: tuple[int, int],
) -> tuple[int, int]:
    value = metadata.get("input_size")
    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 2
    ):
        try:
            return int(value[0]), int(value[1])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid DINO metadata input_size={value!r}") from exc
    # Dense caches produced before schema 2 did not retain preprocessing
    # metadata.  Preserve their direct stream-grid mapping in that case.
    if fallback_size[0] >= patch_height and fallback_size[1] >= patch_width:
        return fallback_size
    return patch_height, patch_width


def _dino_matches(
    queries: Sequence[PatchRecord],
    candidates: Sequence[PatchRecord],
    *,
    device: torch.device,
    top_k: int,
    mutual_nearest: bool,
) -> dict[int, list[PatchMatch]]:
    if not queries or not candidates:
        return {}
    if int(top_k) <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    query_values = F.normalize(
        torch.stack([value.feature for value in queries]).float(), dim=-1
    ).to(device)
    candidate_values = F.normalize(
        torch.stack([value.feature for value in candidates]).float(), dim=-1
    ).to(device)
    with torch.inference_mode():
        similarity = query_values @ candidate_values.T
        count = min(int(top_k), int(candidate_values.shape[0]))
        if mutual_nearest:
            candidate_nearest_query = similarity.argmax(dim=0)
            query_indices = torch.arange(
                int(query_values.shape[0]),
                device=similarity.device,
            )[:, None]
            mutual = candidate_nearest_query[None, :] == query_indices
            retrieval_scores = similarity.masked_fill(~mutual, -torch.inf)
        else:
            retrieval_scores = similarity
        scores, indices = torch.topk(retrieval_scores, k=count, dim=1)
    output: dict[int, list[PatchMatch]] = {}
    for query_index, query in enumerate(queries):
        rows: list[PatchMatch] = []
        for rank in range(count):
            candidate_index = int(indices[query_index, rank].item())
            if not math.isfinite(float(scores[query_index, rank].item())):
                continue
            source = candidates[candidate_index]
            accepted_rank = len(rows) + 1
            rows.append(
                PatchMatch(
                    query=query,
                    source=source,
                    cosine=float(scores[query_index, rank].item()),
                    rank=accepted_rank,
                )
            )
        if rows:
            output[id(query)] = rows
    return output


def _random_matches(
    queries: Sequence[PatchRecord],
    candidates: Sequence[PatchRecord],
    *,
    generator: torch.Generator,
    top_k: int,
) -> dict[int, list[PatchMatch]]:
    if not queries or not candidates:
        return {}
    output: dict[int, list[PatchMatch]] = {}
    for query in queries:
        count = min(int(top_k), len(candidates))
        permutation = torch.randperm(len(candidates), generator=generator)[:count]
        rows: list[PatchMatch] = []
        for rank, index in enumerate(permutation.tolist(), start=1):
            source = candidates[int(index)]
            cosine = F.cosine_similarity(
                query.feature[None].float(), source.feature[None].float()
            )[0]
            rows.append(
                PatchMatch(
                    query=query,
                    source=source,
                    cosine=float(cosine),
                    rank=rank,
                )
            )
        output[id(query)] = rows
    return output


def _extend_match_dict(target: list[PatchMatch], values: Mapping[int, Sequence[PatchMatch]]) -> None:
    for matches in values.values():
        target.extend(matches)


def _match_row(
    *,
    data: Any,
    split: str,
    mode: str,
    aggregation: str,
    k: int,
    match: PatchMatch,
    gt_index: int,
    gt_label: str,
    raw_error: float,
    raw_query_error: float,
    query_target: torch.Tensor,
    retrieved_error: float | None,
    retrieved_query_error: float | None,
) -> dict[str, Any]:
    return {
        "split": split,
        "clip": data.name,
        "scene_id": data.scene_id,
        "mode": mode,
        "aggregation": aggregation,
        "k": int(k),
        "query_frame": int(match.query.frame),
        "query_frame_id": int(data.frame_indices[match.query.frame]),
        "query_slot": int(match.query.slot),
        "query_track_id": int(match.query.track_id),
        "query_patch_y": int(match.query.patch_y),
        "query_patch_x": int(match.query.patch_x),
        "query_pixel_y": int(match.query.pixel_y),
        "query_pixel_x": int(match.query.pixel_x),
        "gt_object": int(gt_index),
        "gt_label": gt_label,
        "history_frame": int(match.source.frame),
        "history_frame_id": int(data.frame_indices[match.source.frame]),
        "history_slot": int(match.source.slot),
        "history_track_id": int(match.source.track_id),
        "history_patch_y": int(match.source.patch_y),
        "history_patch_x": int(match.source.patch_x),
        "history_pixel_y": int(match.source.pixel_y),
        "history_pixel_x": int(match.source.pixel_x),
        "dino_cosine": float(match.cosine),
        "query_confidence": float(match.query.confidence),
        "history_confidence": float(match.source.confidence),
        "raw_error_m": float(raw_error),
        "raw_query_error_m": float(raw_query_error),
        "retrieved_error_m": retrieved_error,
        "retrieved_query_error_m": retrieved_query_error,
        "gain_m": (
            float(raw_error - retrieved_error)
            if retrieved_error is not None
            else float("nan")
        ),
        "improved": (
            int(float(retrieved_error) < float(raw_error))
            if retrieved_error is not None
            else 0
        ),
        "query_gain_m": (
            float(raw_query_error - retrieved_query_error)
            if retrieved_query_error is not None
            else float("nan")
        ),
        "query_improved": (
            int(float(retrieved_query_error) < float(raw_query_error))
            if retrieved_query_error is not None
            else 0
        ),
        "_query_id": id(match.query),
        "_source_point": match.source.point,
        "_query_target": query_target,
    }


def _topk_row(
    *,
    data: Any,
    split: str,
    mode: str,
    k: int,
    matches: Sequence[PatchMatch],
    gt_index: int,
    gt_label: str,
    raw_error: float,
    raw_query_error: float,
    query_target: torch.Tensor,
    retrieved_error: float,
    retrieved_query_error: float,
) -> dict[str, Any]:
    first = matches[0]
    return {
        "split": split,
        "clip": data.name,
        "scene_id": data.scene_id,
        "mode": f"{mode}_topk",
        "aggregation": "coordinate_median",
        "k": int(k),
        "query_frame": int(first.query.frame),
        "query_frame_id": int(data.frame_indices[first.query.frame]),
        "query_slot": int(first.query.slot),
        "query_track_id": int(first.query.track_id),
        "query_patch_y": int(first.query.patch_y),
        "query_patch_x": int(first.query.patch_x),
        "query_pixel_y": int(first.query.pixel_y),
        "query_pixel_x": int(first.query.pixel_x),
        "gt_object": int(gt_index),
        "gt_label": gt_label,
        "history_frame": "|".join(str(value.source.frame) for value in matches),
        "history_frame_id": "|".join(
            str(data.frame_indices[value.source.frame]) for value in matches
        ),
        "history_slot": "|".join(str(value.source.slot) for value in matches),
        "history_track_id": "|".join(
            str(value.source.track_id) for value in matches
        ),
        "history_patch_y": "|".join(str(value.source.patch_y) for value in matches),
        "history_patch_x": "|".join(str(value.source.patch_x) for value in matches),
        "history_pixel_y": "|".join(str(value.source.pixel_y) for value in matches),
        "history_pixel_x": "|".join(str(value.source.pixel_x) for value in matches),
        "dino_cosine": float(sum(value.cosine for value in matches) / len(matches)),
        "query_confidence": float(first.query.confidence),
        "history_confidence": float(
            sum(value.source.confidence for value in matches) / len(matches)
        ),
        "raw_error_m": float(raw_error),
        "raw_query_error_m": float(raw_query_error),
        "retrieved_error_m": float(retrieved_error),
        "retrieved_query_error_m": float(retrieved_query_error),
        "gain_m": float(raw_error - retrieved_error),
        "improved": int(float(retrieved_error) < float(raw_error)),
        "query_gain_m": float(raw_query_error - retrieved_query_error),
        "query_improved": int(float(retrieved_query_error) < float(raw_query_error)),
        "_query_id": id(first.query),
        "_query_target": query_target,
    }


def _fill_retrieved_errors(
    rows: list[dict[str, Any]],
    gt_clouds: Sequence[torch.Tensor],
) -> None:
    for row in rows:
        if row.get("retrieved_error_m") is not None:
            continue
        source = row.pop("_source_point", None)
        if source is None:
            continue
        distance = _nearest_distance(
            source[None],
            gt_clouds[int(row["gt_object"])],
        )[0]
        value = float(distance)
        row["retrieved_error_m"] = value
        row["gain_m"] = float(row["raw_error_m"] - value)
        row["improved"] = int(value < float(row["raw_error_m"]))
        query_target = row.pop("_query_target", None)
        if query_target is not None:
            query_distance = torch.linalg.vector_norm(
                source.detach().float().cpu() - query_target.detach().float().cpu()
            )
            query_value = float(query_distance)
            row["retrieved_query_error_m"] = query_value
            row["query_gain_m"] = float(row["raw_query_error_m"] - query_value)
            row["query_improved"] = int(
                query_value < float(row["raw_query_error_m"])
            )
    # The top-k rows already contain their evaluated distance.  Remove any
    # internal tensor/object keys before they are written to CSV.
    for row in rows:
        for key in tuple(row):
            if key.startswith("_"):
                row.pop(key, None)


def _condition_rows(
    match_rows: Sequence[Mapping[str, Any]],
    topk_rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    clip: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    values_by_condition: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in match_rows:
        values_by_condition[(str(row["mode"]), "top1")].append(row)
    for row in topk_rows:
        values_by_condition[(str(row["mode"]), f"k{row['k']}")].append(row)
    for (mode, condition), values in sorted(values_by_condition.items()):
        rows.append(
            {
                "split": split,
                "clip": clip,
                "mode": mode,
                "condition": condition,
                **_metric_fields(values),
            }
        )
    return rows


def _similarity_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    clip: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        cosine = float(row["dino_cosine"])
        for low, high, label in SIMILARITY_BINS:
            if low <= cosine < high:
                grouped[(str(row["mode"]), label)].append(row)
                break
    for (mode, label), values in sorted(grouped.items()):
        bounds = next(value for value in SIMILARITY_BINS if value[2] == label)
        output.append(
            {
                "split": split,
                "clip": clip,
                "mode": mode,
                "similarity_bin": label,
                "lower": bounds[0],
                "upper": bounds[1] if math.isfinite(bounds[1]) else "inf",
                **_metric_fields(values),
            }
        )
    return output


def _object_metric_rows(
    match_rows: Sequence[Mapping[str, Any]],
    topk_rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    clip: str,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in (*match_rows, *topk_rows):
        grouped[(str(row["mode"]), int(row["gt_object"]))].append(row)
    output: list[dict[str, Any]] = []
    for (mode, object_index), values in sorted(grouped.items()):
        output.append(
            {
                "split": split,
                "clip": clip,
                "mode": mode,
                "gt_object": object_index,
                "gt_label": str(values[0]["gt_label"]),
                **_metric_fields(values),
            }
        )
    return output


def _scene_metric_rows(
    match_rows: Sequence[Mapping[str, Any]],
    topk_rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    clip: str,
    eligible_queries: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in (*match_rows, *topk_rows):
        grouped[str(row["mode"])].append(row)
    output: list[dict[str, Any]] = []
    for mode, values in sorted(grouped.items()):
        unique_queries = len(
            {
                (
                    int(row["query_frame"]),
                    int(row["query_slot"]),
                    int(row["query_patch_y"]),
                    int(row["query_patch_x"]),
                )
                for row in values
            }
        )
        output.append(
            {
                "split": split,
                "clip": clip,
                "scene_id": str(values[0]["scene_id"]),
                "mode": mode,
                "eligible_queries": int(eligible_queries),
                "matched_queries": int(unique_queries),
                "coverage": float(unique_queries / max(1, eligible_queries)),
                **_metric_fields(values),
            }
        )
    return output


def _paired_comparison_rows(
    match_rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    clip: str,
    scene_id: str,
) -> list[dict[str, Any]]:
    """Compare controls on the exact query intersection, not different samples."""

    by_mode: dict[str, dict[tuple[int, int, int, int], Mapping[str, Any]]] = {
        mode: {} for mode in MODES
    }
    for row in match_rows:
        mode = str(row["mode"])
        if mode not in by_mode or str(row.get("aggregation")) != "top1":
            continue
        by_mode[mode][_query_row_key(row)] = row

    correct = by_mode["correct_track_dino"]
    output: list[dict[str, Any]] = []
    for control in PAIR_CONTROLS:
        control_rows = by_mode[control]
        common = sorted(set(correct).intersection(control_rows))
        for mode, source in (
            ("correct_track_dino", correct),
            (control, control_rows),
        ):
            values = [source[key] for key in common]
            output.append(
                {
                    "split": split,
                    "clip": clip,
                    "scene_id": str(scene_id),
                    "comparison": f"correct_vs_{control}",
                    "mode": mode,
                    "correct_matched_queries": len(correct),
                    "control_matched_queries": len(control_rows),
                    "paired_queries": len(common),
                    "paired_fraction_of_correct": len(common) / max(1, len(correct)),
                    "paired_fraction_of_control": len(common) / max(1, len(control_rows)),
                    **_metric_fields(values),
                }
            )
    return output


def _query_row_key(row: Mapping[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(row["query_frame"]),
        int(row["query_slot"]),
        int(row["query_patch_y"]),
        int(row["query_patch_x"]),
    )


def _metric_fields(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    surface = _finite_metric_rows(rows, "raw_error_m", "retrieved_error_m")
    query = _finite_metric_rows(
        rows,
        "raw_query_error_m",
        "retrieved_query_error_m",
    )
    return {
        **_metric_fields_for_pair(
            surface,
            raw_key="raw_error_m",
            retrieved_key="retrieved_error_m",
        ),
        **_metric_fields_for_pair(
            query,
            raw_key="raw_query_error_m",
            retrieved_key="retrieved_query_error_m",
            prefix="query_",
        ),
    }


def _finite_metric_rows(
    rows: Sequence[Mapping[str, Any]],
    raw_key: str,
    retrieved_key: str,
) -> list[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if math.isfinite(float(row.get(raw_key, float("nan"))))
        and math.isfinite(float(row.get(retrieved_key, float("nan"))))
    ]


def _metric_fields_for_pair(
    rows: Sequence[Mapping[str, Any]],
    *,
    raw_key: str,
    retrieved_key: str,
    prefix: str = "",
) -> dict[str, Any]:
    def key(name: str) -> str:
        return f"{prefix}{name}"

    if not rows:
        return {
            key("count"): 0,
            key("raw_rmse_m"): float("nan"),
            key("retrieved_rmse_m"): float("nan"),
            key("raw_mean_m"): float("nan"),
            key("retrieved_mean_m"): float("nan"),
            key("raw_median_m"): float("nan"),
            key("retrieved_median_m"): float("nan"),
            key("retrieved_p90_m"): float("nan"),
            key("mean_gain_m"): float("nan"),
            key("gain_percent"): float("nan"),
            key("improved_fraction"): float("nan"),
        }
    raw = torch.tensor([float(row[raw_key]) for row in rows])
    retrieved = torch.tensor([float(row[retrieved_key]) for row in rows])
    gain = raw - retrieved
    return {
        key("count"): len(rows),
        key("raw_rmse_m"): _rmse(raw),
        key("retrieved_rmse_m"): _rmse(retrieved),
        key("raw_mean_m"): float(raw.mean()),
        key("retrieved_mean_m"): float(retrieved.mean()),
        key("raw_median_m"): float(raw.median()),
        key("retrieved_median_m"): float(retrieved.median()),
        key("retrieved_p90_m"): float(torch.quantile(retrieved, 0.90)),
        key("mean_gain_m"): float(gain.mean()),
        key("gain_percent"): 100.0 * float(gain.mean()) / max(float(raw.mean()), 1e-12),
        key("improved_fraction"): float((gain > 0.0).float().mean()),
    }


def _aggregate_scene_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["split"]), str(row["mode"]))].append(row)
    output: list[dict[str, Any]] = []
    for (split, mode), values in sorted(grouped.items()):
        # Reconstruct point-level sufficient statistics from scene metrics. A
        # scene-weighted aggregate is intentionally reported separately from
        # the per-scene rows so a large clip cannot hide a failing scene.
        matched = sum(int(row["matched_queries"]) for row in values)
        eligible = sum(int(row["eligible_queries"]) for row in values)
        output.append(
            {
                "split": split,
                "clip": "ALL",
                "scene_count": len(values),
                "mode": mode,
                "eligible_queries": eligible,
                "matched_queries": matched,
                "coverage": matched / max(1, eligible),
                **_aggregate_scene_pair(values),
                **_aggregate_scene_pair(values, prefix="query_"),
            }
        )
    return output


def _aggregate_scene_pair(
    rows: Sequence[Mapping[str, Any]],
    *,
    prefix: str = "",
) -> dict[str, Any]:
    """Aggregate sufficient point statistics from per-scene summaries."""

    count_key = f"{prefix}count"
    raw_rmse_key = f"{prefix}raw_rmse_m"
    retrieved_rmse_key = f"{prefix}retrieved_rmse_m"
    raw_mean_key = f"{prefix}raw_mean_m"
    retrieved_mean_key = f"{prefix}retrieved_mean_m"
    gain_key = f"{prefix}mean_gain_m"
    improved_key = f"{prefix}improved_fraction"
    valid = [
        row
        for row in rows
        if int(row.get(count_key, 0)) > 0
        and math.isfinite(float(row.get(raw_rmse_key, float("nan"))))
        and math.isfinite(float(row.get(retrieved_rmse_key, float("nan"))))
    ]
    count = sum(int(row[count_key]) for row in valid)
    if count <= 0:
        return _metric_fields_for_pair(
            (),
            raw_key="raw_error_m",
            retrieved_key="retrieved_error_m",
            prefix=prefix,
        )
    raw_sq = sum(float(row[raw_rmse_key]) ** 2 * int(row[count_key]) for row in valid)
    retrieved_sq = sum(
        float(row[retrieved_rmse_key]) ** 2 * int(row[count_key]) for row in valid
    )
    raw_sum = sum(
        float(row[raw_mean_key]) * int(row[count_key])
        for row in valid
        if math.isfinite(float(row.get(raw_mean_key, float("nan"))))
    )
    retrieved_sum = sum(
        float(row[retrieved_mean_key]) * int(row[count_key])
        for row in valid
        if math.isfinite(float(row.get(retrieved_mean_key, float("nan"))))
    )
    gain_sum = sum(
        float(row[gain_key]) * int(row[count_key])
        for row in valid
        if math.isfinite(float(row.get(gain_key, float("nan"))))
    )
    improved_sum = sum(
        float(row[improved_key]) * int(row[count_key])
        for row in valid
        if math.isfinite(float(row.get(improved_key, float("nan"))))
    )
    raw_mean = raw_sum / max(1, count)
    retrieved_mean = retrieved_sum / max(1, count)
    mean_gain = gain_sum / max(1, count)
    return {
        f"{prefix}count": count,
        f"{prefix}raw_rmse_m": math.sqrt(raw_sq / count),
        f"{prefix}retrieved_rmse_m": math.sqrt(retrieved_sq / count),
        f"{prefix}raw_mean_m": raw_mean,
        f"{prefix}retrieved_mean_m": retrieved_mean,
        f"{prefix}raw_median_m": float("nan"),
        f"{prefix}retrieved_median_m": float("nan"),
        f"{prefix}retrieved_p90_m": float("nan"),
        f"{prefix}mean_gain_m": mean_gain,
        f"{prefix}gain_percent": 100.0 * mean_gain / max(raw_mean, 1e-12),
        f"{prefix}improved_fraction": improved_sum / max(1, count),
    }


def _aggregate_paired_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (str(row["split"]), str(row["comparison"]), str(row["mode"]))
        ].append(row)
    output: list[dict[str, Any]] = []
    for (split, comparison, mode), values in sorted(grouped.items()):
        output.append(
            {
                "split": split,
                "clip": "ALL",
                "scene_count": len(values),
                "comparison": comparison,
                "mode": mode,
                "correct_matched_queries": sum(
                    int(row["correct_matched_queries"]) for row in values
                ),
                "control_matched_queries": sum(
                    int(row["control_matched_queries"]) for row in values
                ),
                "paired_queries": sum(int(row["paired_queries"]) for row in values),
                "paired_fraction_of_correct": _weighted_fraction(
                    values,
                    numerator_key="paired_queries",
                    denominator_key="correct_matched_queries",
                ),
                "paired_fraction_of_control": _weighted_fraction(
                    values,
                    numerator_key="paired_queries",
                    denominator_key="control_matched_queries",
                ),
                **_aggregate_scene_pair(values),
                **_aggregate_scene_pair(values, prefix="query_"),
            }
        )
    return output


def _weighted_fraction(
    rows: Sequence[Mapping[str, Any]],
    *,
    numerator_key: str,
    denominator_key: str,
) -> float:
    numerator = sum(int(row[numerator_key]) for row in rows)
    denominator = sum(int(row[denominator_key]) for row in rows)
    return numerator / max(1, denominator)


def _aggregate_object_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["split"]), str(row["mode"]))].append(row)
    output: list[dict[str, Any]] = []
    for (split, mode), values in sorted(grouped.items()):
        gains = [
            float(row["mean_gain_m"])
            for row in values
            if math.isfinite(float(row["mean_gain_m"]))
        ]
        query_gains = [
            float(row["query_mean_gain_m"])
            for row in values
            if math.isfinite(float(row.get("query_mean_gain_m", float("nan"))))
        ]
        output.append(
            {
                "split": split,
                "clip": "ALL_OBJECTS",
                "mode": mode,
                "gt_object": "ALL",
                "gt_label": "",
                "objects": len(gains),
                "positive_gain_objects": sum(value > 0.0 for value in gains),
                "positive_gain_fraction": (
                    sum(value > 0.0 for value in gains) / len(gains)
                    if gains
                    else float("nan")
                ),
                "mean_object_gain_m": sum(gains) / len(gains) if gains else float("nan"),
                "query_objects": len(query_gains),
                "query_positive_gain_objects": sum(value > 0.0 for value in query_gains),
                "query_positive_gain_fraction": (
                    sum(value > 0.0 for value in query_gains) / len(query_gains)
                    if query_gains
                    else float("nan")
                ),
                "mean_object_query_gain_m": (
                    sum(query_gains) / len(query_gains)
                    if query_gains
                    else float("nan")
                ),
            }
        )
    return output


def _interpret(
    scene_rows: Sequence[Mapping[str, Any]],
    object_rows: Sequence[Mapping[str, Any]],
    paired_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    def scene(split: str, mode: str) -> Mapping[str, Any] | None:
        for row in scene_rows:
            if row.get("clip") == "ALL" and row.get("split") == split and row.get("mode") == mode:
                return row
        return None

    def objects(split: str, mode: str) -> Mapping[str, Any] | None:
        for row in object_rows:
            if row.get("clip") == "ALL_OBJECTS" and row.get("split") == split and row.get("mode") == mode:
                return row
        return None

    def paired(comparison: str, mode: str) -> Mapping[str, Any] | None:
        for row in paired_rows:
            if (
                row.get("clip") == "ALL"
                and row.get("comparison") == comparison
                and row.get("mode") == mode
            ):
                return row
        return None

    correct = scene("test", "correct_track_dino")
    correct_object = objects("test", "correct_track_dino")
    correct_random = paired("correct_vs_random_same_object", "correct_track_dino")
    random_pair = paired("correct_vs_random_same_object", "random_same_object")
    correct_shuffled = paired("correct_vs_shuffled_track_dino", "correct_track_dino")
    shuffled_pair = paired("correct_vs_shuffled_track_dino", "shuffled_track_dino")
    correct_unrestricted = paired(
        "correct_vs_unrestricted_dino", "correct_track_dino"
    )
    unrestricted_pair = paired(
        "correct_vs_unrestricted_dino", "unrestricted_dino"
    )

    def beats(
        first: Mapping[str, Any] | None,
        second: Mapping[str, Any] | None,
        first_key: str,
        second_key: str,
    ) -> int:
        return int(
            bool(first)
            and bool(second)
            and math.isfinite(float(first.get(first_key, float("nan"))))
            and math.isfinite(float(second.get(second_key, float("nan"))))
            and float(first[first_key]) < float(second[second_key])
        )

    return {
        "correct_track_beats_raw": beats(
            correct, correct, "retrieved_rmse_m", "raw_rmse_m"
        ),
        "correct_track_beats_random_same_object": beats(
            correct_random,
            random_pair,
            "retrieved_rmse_m",
            "retrieved_rmse_m",
        ),
        "correct_track_beats_shuffled_track": beats(
            correct_shuffled,
            shuffled_pair,
            "retrieved_rmse_m",
            "retrieved_rmse_m",
        ),
        "correct_track_beats_unrestricted": beats(
            correct_unrestricted,
            unrestricted_pair,
            "retrieved_rmse_m",
            "retrieved_rmse_m",
        ),
        "correct_track_query_surface_beats_raw": beats(
            correct, correct, "query_retrieved_rmse_m", "query_raw_rmse_m"
        ),
        "correct_track_query_surface_beats_random_same_object": beats(
            correct_random,
            random_pair,
            "query_retrieved_rmse_m",
            "query_retrieved_rmse_m",
        ),
        "correct_track_query_surface_beats_shuffled_track": beats(
            correct_shuffled,
            shuffled_pair,
            "query_retrieved_rmse_m",
            "query_retrieved_rmse_m",
        ),
        "correct_track_query_surface_beats_unrestricted": beats(
            correct_unrestricted,
            unrestricted_pair,
            "query_retrieved_rmse_m",
            "query_retrieved_rmse_m",
        ),
        "correct_track_positive_gain_object_fraction": (
            correct_object.get("positive_gain_fraction", float("nan"))
            if correct_object
            else float("nan")
        ),
        "correct_track_positive_query_gain_object_fraction": (
            correct_object.get("query_positive_gain_fraction", float("nan"))
            if correct_object
            else float("nan")
        ),
        "interpretation": (
            "Surface gain measures object-level support: the retrieved point is "
            "closer to the queried object's aggregated GT surface. Query-surface "
            "gain is stricter: it measures distance to the GT point at the "
            "current query pixel and is the better test of local correspondence. "
            "Neither metric changes XYZ or feeds GT to candidate generation. "
            "Build geometry memory only if correct-track retrieval beats the "
            "controls on the strict metric with adequate coverage and without "
            "a P90 regression."
        ),
    }


def _build_gt_clouds(
    data: Any,
    *,
    gt_masks: torch.Tensor,
    max_points: int,
) -> list[torch.Tensor]:
    points = data.target_metric.detach().float().cpu()
    finite = torch.isfinite(points).all(dim=-1)
    clouds: list[torch.Tensor] = []
    for object_index in range(int(gt_masks.shape[1])):
        selected = points[gt_masks[:, object_index] & finite]
        clouds.append(_subsample_points(selected, max_points))
    return clouds


def _query_gt_object(query: PatchRecord, gt_masks: torch.Tensor) -> int:
    if query.frame >= int(gt_masks.shape[0]):
        return -1
    hits = torch.nonzero(
        gt_masks[query.frame, :, query.pixel_y, query.pixel_x],
        as_tuple=False,
    ).flatten()
    return int(hits[0]) if hits.numel() else -1


def _nearest_distance(points: torch.Tensor, cloud: torch.Tensor) -> torch.Tensor:
    values = points.detach().float().cpu().reshape(-1, 3)
    target = cloud.detach().float().cpu().reshape(-1, 3)
    if not values.numel() or not target.numel():
        return torch.full((values.shape[0],), float("nan"))
    chunks: list[torch.Tensor] = []
    for start in range(0, values.shape[0], 1024):
        chunks.append(torch.cdist(values[start : start + 1024], target).min(dim=1).values)
    return torch.cat(chunks)


def _aggregate_points(points: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.stack([value.detach().float().cpu() for value in points]).median(dim=0).values


def _patch_centres(
    patch_count: int,
    pixel_count: int,
    *,
    resized_count: int | None = None,
) -> torch.Tensor:
    """Map DINO patch centers back to the StreamVGGT pixel grid.

    DINO rounds each image dimension to a multiple of its patch size before
    encoding.  Using that rounded dimension avoids a one-pixel drift when the
    StreamVGGT grid is not itself divisible by the DINO patch size.
    """

    if int(patch_count) <= 0 or int(pixel_count) <= 0:
        raise ValueError(
            f"patch_count and pixel_count must be positive, got {patch_count}, {pixel_count}"
        )
    resized = int(resized_count) if resized_count is not None else int(pixel_count)
    if resized <= 0:
        raise ValueError(f"resized_count must be positive, got {resized_count}")
    return (
        (
            (torch.arange(patch_count).float() + 0.5)
            * float(resized)
            / float(patch_count)
            * float(pixel_count)
            / float(resized)
            - 0.5
        )
        .round()
        .long()
        .clamp(0, pixel_count - 1)
    )


def _subsample_coordinates(coordinates: torch.Tensor, limit: int) -> torch.Tensor:
    if coordinates.shape[0] <= int(limit):
        return coordinates
    indices = torch.linspace(
        0,
        coordinates.shape[0] - 1,
        steps=int(limit),
    ).round().long().unique()
    return coordinates.index_select(0, indices)


def _subsample_points(points: torch.Tensor, limit: int) -> torch.Tensor:
    if points.shape[0] <= int(limit):
        return points
    indices = torch.linspace(0, points.shape[0] - 1, steps=int(limit)).round().long().unique()
    return points.index_select(0, indices)


def _shuffled_slots(slot_count: int) -> tuple[int, ...]:
    if slot_count <= 1:
        return tuple(range(slot_count))
    return tuple((index + 1) % slot_count for index in range(slot_count))


def _rmse(values: torch.Tensor) -> float:
    finite = values[torch.isfinite(values)]
    return float(torch.sqrt(finite.square().mean())) if finite.numel() else float("nan")


def _resolve_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {device} requested but CUDA is unavailable.")
    return device


def _resolve_path(value: str | None, root: Path, default: str) -> Path:
    path = Path(value).expanduser() if value else root / default
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _print_clip_summary(summary: Mapping[str, Any]) -> None:
    print(
        f"  {summary['split']} clip={summary['clip']} "
        f"records={summary['records']} eligible_queries={summary['eligible_queries']}"
    )
    for row in summary["scene_metrics"]:
        print(
            f"    {row['mode']} matched={row['matched_queries']}/{row['eligible_queries']} "
            f"raw_rmse={row['raw_rmse_m']:.5f} "
            f"retrieved_rmse={row['retrieved_rmse_m']:.5f} "
            f"gain={row['mean_gain_m']:.5f} "
            f"improved={row['improved_fraction']:.3f} "
            f"query_raw_rmse={row['query_raw_rmse_m']:.5f} "
            f"query_retrieved_rmse={row['query_retrieved_rmse_m']:.5f} "
            f"query_gain={row['query_mean_gain_m']:.5f} "
            f"query_improved={row['query_improved_fraction']:.3f}"
        )


def _print_paired_summary(rows: Sequence[Mapping[str, Any]]) -> None:
    for row in rows:
        if str(row.get("split")) not in {"validation", "test"}:
            continue
        print(
            f"    paired[{row['comparison']}] mode={row['mode']} "
            f"common={row['paired_queries']} "
            f"surface_retrieved_rmse={row['retrieved_rmse_m']:.5f} "
            f"query_retrieved_rmse={row['query_retrieved_rmse_m']:.5f} "
            f"surface_gain={row['mean_gain_m']:.5f} "
            f"query_gain={row['query_mean_gain_m']:.5f}"
        )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf8")
        return
    clean_rows = [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in rows
    ]
    fields: list[str] = []
    for row in clean_rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(clean_rows)


def _write_copyable(path: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "===== DINOV3_PATCH_GEOMETRY_RETRIEVAL_BEGIN =====",
        f"revision={summary['revision']}",
        "diagnostic_only=1",
        "candidate_generation_uses_gt=0",
        "candidate_generation_uses_future_frames=0",
        "metrics=object_surface_and_query_surface",
        "",
        "split,mode,scene_count,eligible_queries,matched_queries,coverage,count,raw_rmse_m,retrieved_rmse_m,mean_gain_m,gain_percent,improved_fraction,query_count,query_raw_rmse_m,query_retrieved_rmse_m,query_mean_gain_m,query_gain_percent,query_improved_fraction",
    ]
    for row in summary["aggregate_scene_metrics"]:
        lines.append(
            ",".join(
                str(row.get(key, ""))
                for key in (
                    "split",
                    "mode",
                    "scene_count",
                    "eligible_queries",
                    "matched_queries",
                    "coverage",
                    "count",
                    "raw_rmse_m",
                    "retrieved_rmse_m",
                    "mean_gain_m",
                    "gain_percent",
                    "improved_fraction",
                    "query_count",
                    "query_raw_rmse_m",
                    "query_retrieved_rmse_m",
                    "query_mean_gain_m",
                    "query_gain_percent",
                    "query_improved_fraction",
                )
            )
        )
    lines.extend(
        (
            "",
            "interpretation=" + json.dumps(summary["interpretation"], sort_keys=True),
            f"correct_track_beats_random_same_object={summary['interpretation']['correct_track_beats_random_same_object']}",
            f"correct_track_beats_shuffled_track={summary['interpretation']['correct_track_beats_shuffled_track']}",
            f"correct_track_beats_unrestricted={summary['interpretation']['correct_track_beats_unrestricted']}",
            f"correct_track_query_surface_beats_raw={summary['interpretation']['correct_track_query_surface_beats_raw']}",
            f"correct_track_query_surface_beats_random_same_object={summary['interpretation']['correct_track_query_surface_beats_random_same_object']}",
            f"correct_track_query_surface_beats_shuffled_track={summary['interpretation']['correct_track_query_surface_beats_shuffled_track']}",
            f"correct_track_query_surface_beats_unrestricted={summary['interpretation']['correct_track_query_surface_beats_unrestricted']}",
            f"correct_track_positive_gain_object_fraction={summary['interpretation']['correct_track_positive_gain_object_fraction']}",
            f"correct_track_positive_query_gain_object_fraction={summary['interpretation']['correct_track_positive_query_gain_object_fraction']}",
            f"summary={path.with_name('summary.json')}",
            f"match_metrics={path.with_name('match_metrics.csv')}",
            f"similarity_bins={path.with_name('similarity_bins.csv')}",
            f"per_object_metrics={path.with_name('per_object_metrics.csv')}",
            f"per_scene_metrics={path.with_name('per_scene_metrics.csv')}",
            f"paired_comparison={path.with_name('paired_comparison.csv')}",
            f"condition_summary={path.with_name('condition_summary.csv')}",
            "===== DINOV3_PATCH_GEOMETRY_RETRIEVAL_END =====",
        )
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        return value.tolist()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol")
    parser.add_argument("--feature-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-patches-per-object-frame", type=int, default=32)
    parser.add_argument("--max-gt-surface-points", type=int, default=4096)
    parser.add_argument(
        "--mutual-nearest",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep DINO matches only when the source patch selects the query back.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
