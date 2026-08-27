#!/usr/bin/env python3
"""Evaluate frozen prompt-SAM, class-agnostic auto-SAM, and GT oracle masks."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from streaming_couping.scripts.generate_sam31_auto_proposals import (
    ARTIFACT_NAME,
)
from streaming_couping.scripts.run_semantic_map_evaluation import (
    _build_oracle_variant,
    _load_run as _load_evaluation_run,
)
from streaming_couping.src.learned_pose.cache import cache_path, load_feature_cache
from streaming_couping.src.learned_pose.config import (
    ClipConfig,
    load_learned_pose_config,
)
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


REVISION = "sam31_auto_proposal_multiclip_evaluation_r2_all_instance_scope"
PROMPT_LABEL = "prompt_sam31_online_forward"
AUTO_LABEL = "sam31_auto_visual_points"
ORACLE_LABEL = "gt_mask_oracle"
PROMPT_SCOPE = "prompt_scope"
ALL_INSTANCE_SCOPE = "all_instance_scope"


def main() -> None:
    args = _parse_args()
    config_path = Path(args.config).expanduser().resolve()
    data = load_learned_pose_config(config_path)
    evaluation_run = _load_evaluation_run(config_path)
    clips = _select_clips(data.clips, args.clip)
    artifact_root = Path(args.artifact_root).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else artifact_root / "evaluation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("SAM3.1 AUTO-PROPOSAL MULTI-CLIP EVALUATION")
    print(f"config={config_path}")
    print(f"artifact_root={artifact_root}")
    print(
        "branches=prompt_sam,auto_visual_points,gt_mask_oracle; "
        "prompt and auto use independent frozen GT assignments"
    )
    print(
        "auto candidate artifacts are already frozen; GT is opened only in "
        "this evaluation process"
    )
    print(
        "gt_scopes=prompt_scope (historical comparison) + "
        "all_instance_scope (open-world diagnostic)"
    )

    clip_rows: list[dict[str, object]] = []
    tracking_rows: list[dict[str, object]] = []
    map_rows: list[dict[str, object]] = []
    all_tracking_rows: list[dict[str, object]] = []
    all_map_rows: list[dict[str, object]] = []
    object_rows: list[dict[str, object]] = []
    all_object_rows: list[dict[str, object]] = []
    proposal_rows: list[dict[str, object]] = []
    for clip in clips:
        payload = load_feature_cache(cache_path(data, clip))
        artifact_path = artifact_root / clip.name / ARTIFACT_NAME
        auto = _load_auto_artifact(artifact_path, clip)
        evaluated = _evaluate_clip(
            data=data,
            clip=clip,
            payload=payload,
            auto=auto,
            tracking_config=evaluation_run.tracking,
            map_config=evaluation_run.map_metrics,
            prompt_label_aliases=evaluation_run.prompt_label_aliases,
            raw_cache_variant=args.raw_cache_variant,
        )
        clip_rows.append(evaluated["clip"])
        tracking_rows.extend(evaluated["tracking_rows"])
        map_rows.extend(evaluated["map_rows"])
        all_tracking_rows.extend(evaluated["all_tracking_rows"])
        all_map_rows.extend(evaluated["all_map_rows"])
        object_rows.extend(evaluated["object_rows"])
        all_object_rows.extend(evaluated["all_object_rows"])
        proposal_rows.extend(
            dict(row) for row in auto.get("proposal_diagnostics", ())
        )
        _print_clip(evaluated["clip"])

    aggregate = _aggregate(
        tracking_rows,
        map_rows,
        clip_rows,
        scope=PROMPT_SCOPE,
    )
    all_instance_aggregate = _aggregate(
        all_tracking_rows,
        all_map_rows,
        clip_rows,
        scope=ALL_INSTANCE_SCOPE,
    )
    decision = _decision(aggregate, clip_rows, scope=PROMPT_SCOPE)
    summary: dict[str, object] = {
        "schema": 2,
        "revision": REVISION,
        "candidate_generation_gt_fields": 0,
        "evaluation_gt_fields": 1,
        "semantic_text_prompt_used_by_auto": 0,
        "auto_semantic_label_policy": "generic_object_only",
        "assignment_policy": {
            "prompt": "prompt_branch_own_frozen_hungarian_assignment",
            "auto": "auto_branch_own_frozen_hungarian_assignment",
            "oracle_prompt_scope": "prompt_assignment_with_gt_masks",
            "oracle_all_instance_scope": "auto_assignment_with_gt_masks",
        },
        "clips": clip_rows,
        "tracking_summary": tracking_rows,
        "map_summary": map_rows,
        "all_instance_scope": {
            "tracking_summary": all_tracking_rows,
            "map_summary": all_map_rows,
            "aggregate": all_instance_aggregate,
        },
        "aggregate": aggregate,
        "decision": decision,
        "prompt_scope_decision": decision,
        "all_instance_scope_decision": {
            "status": "DIAGNOSTIC_ONLY",
            "reason": (
                "all-instance scope measures open-world recall; it is not a "
                "prompt-vs-auto non-regression gate"
            ),
        },
        "outputs": {},
    }
    summary_path = output_dir / "summary.json"
    outputs = {
        "summary": summary_path,
        "clip_variant_summary": _write_csv(
            output_dir / "clip_variant_summary.csv",
            _clip_variant_rows(clip_rows),
        ),
        "all_instance_clip_variant_summary": _write_csv(
            output_dir / "all_instance_clip_variant_summary.csv",
            _clip_variant_rows(clip_rows, scope=ALL_INSTANCE_SCOPE),
        ),
        "tracking_summary": _write_csv(
            output_dir / "tracking_summary.csv",
            tracking_rows,
        ),
        "map_summary": _write_csv(
            output_dir / "map_summary.csv",
            map_rows,
        ),
        "all_instance_tracking_summary": _write_csv(
            output_dir / "all_instance_tracking_summary.csv",
            all_tracking_rows,
        ),
        "all_instance_map_summary": _write_csv(
            output_dir / "all_instance_map_summary.csv",
            all_map_rows,
        ),
        "per_object_tracking": _write_csv(
            output_dir / "per_object_tracking.csv",
            object_rows,
        ),
        "all_instance_per_object_tracking": _write_csv(
            output_dir / "all_instance_per_object_tracking.csv",
            all_object_rows,
        ),
        "proposal_diagnostics": _write_csv(
            output_dir / "proposal_diagnostics.csv",
            proposal_rows,
        ),
    }
    summary["outputs"] = {name: str(path) for name, path in outputs.items()}
    _write_json(summary_path, summary)
    copyable = output_dir / "copyable_result.txt"
    _write_copyable(copyable, summary)
    summary["outputs"]["copyable_result"] = str(copyable)
    _write_json(summary_path, summary)
    print(f"summary={summary_path}")
    print(f"copyable_result={copyable}")
    print(f"decision={decision['status']}")


def _evaluate_clip(
    *,
    data,
    clip: ClipConfig,
    payload: Mapping[str, Any],
    auto: Mapping[str, Any],
    tracking_config,
    map_config,
    prompt_label_aliases,
    raw_cache_variant: str,
) -> dict[str, object]:
    prompt_output, prompt_stream, prompt_scores = _load_raw_branch(
        payload,
        raw_cache_variant,
    )
    auto_output = torch.as_tensor(auto["tracking_masks_output"]).bool()
    auto_stream = torch.as_tensor(auto["tracking_masks_stream"]).bool()
    auto_scores = torch.as_tensor(auto["tracking_scores"]).float()
    if tuple(auto_output.shape) != tuple(prompt_output.shape):
        raise ValueError(f"Auto/output shape mismatch for {clip.name}.")
    if tuple(auto_stream.shape) != tuple(prompt_stream.shape):
        raise ValueError(f"Auto/stream shape mismatch for {clip.name}.")
    if tuple(auto_scores.shape) != tuple(prompt_scores.shape):
        raise ValueError(f"Auto/score shape mismatch for {clip.name}.")

    frames = tuple(int(value) for value in payload["frame_indices"])
    output_size = (
        int(prompt_output.shape[-2]),
        int(prompt_output.shape[-1]),
    )
    prompt_ground_truth = load_ground_truth_instances(
        data.manifest,
        scene_id=clip.scene_id,
        frame_indices=frames,
        output_size=output_size,
        prompts=tuple(str(value) for value in payload["instance_prompts"]),
        prompt_label_aliases=prompt_label_aliases,
    )
    all_ground_truth = load_ground_truth_instances(
        data.manifest,
        scene_id=clip.scene_id,
        frame_indices=frames,
        output_size=output_size,
        prompts=(),
        prompt_label_aliases=prompt_label_aliases,
        include_all_instances=True,
    )
    processed_size = tuple(int(value) for value in payload["image_size"])
    prompt_gt_stream = load_ground_truth_stream_masks(
        data.manifest,
        scene_id=clip.scene_id,
        frame_indices=frames,
        instance_ids=prompt_ground_truth.instance_ids,
        processed_size=processed_size,
        image_mode=str(payload["image_mode"]),
    )
    all_gt_stream = load_ground_truth_stream_masks(
        data.manifest,
        scene_id=clip.scene_id,
        frame_indices=frames,
        instance_ids=all_ground_truth.instance_ids,
        processed_size=processed_size,
        image_mode=str(payload["image_mode"]),
    )
    auto_track_ids = tuple(int(value) for value in auto["track_ids"])
    auto_track_prompts = tuple(str(value) for value in auto["track_prompts"])
    aligned_points = apply_similarity(
        payload["baseline_world_points"],
        scale=float(payload["point_alignment_scale"]),
        rotation=payload["point_alignment_rotation"],
        translation=payload["point_alignment_translation"],
    )
    confidence = normalize_confidence(payload["baseline_world_confidence"])
    map_common = {
        "scene_id": clip.scene_id,
        "clip_name": clip.name,
        "aligned_world_points": aligned_points,
        "target_world_points": torch.as_tensor(
            payload["target_world_points"]
        ).float(),
        "confidence": confidence,
        "config": map_config,
    }
    prompt_scope = _evaluate_scope(
        scope=PROMPT_SCOPE,
        frame_indices=frames,
        ground_truth=prompt_ground_truth,
        gt_stream=prompt_gt_stream,
        prompt_output=prompt_output,
        prompt_stream=prompt_stream,
        prompt_scores=prompt_scores,
        auto_output=auto_output,
        auto_stream=auto_stream,
        auto_scores=auto_scores,
        prompt_track_ids=tuple(
            int(value) for value in payload["sam_track_ids"]
        ),
        prompt_track_prompts=tuple(
            str(value) for value in payload["sam_track_prompts"]
        ),
        auto_track_ids=auto_track_ids,
        auto_track_prompts=auto_track_prompts,
        tracking_config=tracking_config,
        map_config_common=map_common,
        prompt_label_aliases=prompt_label_aliases,
        oracle_source=PROMPT_LABEL,
    )
    all_scope = _evaluate_scope(
        scope=ALL_INSTANCE_SCOPE,
        frame_indices=frames,
        ground_truth=all_ground_truth,
        gt_stream=all_gt_stream,
        prompt_output=prompt_output,
        prompt_stream=prompt_stream,
        prompt_scores=prompt_scores,
        auto_output=auto_output,
        auto_stream=auto_stream,
        auto_scores=auto_scores,
        prompt_track_ids=tuple(
            int(value) for value in payload["sam_track_ids"]
        ),
        prompt_track_prompts=tuple(
            str(value) for value in payload["sam_track_prompts"]
        ),
        auto_track_ids=auto_track_ids,
        auto_track_prompts=auto_track_prompts,
        tracking_config=tracking_config,
        map_config_common=map_common,
        prompt_label_aliases=prompt_label_aliases,
        oracle_source=AUTO_LABEL,
    )
    clip_result = {
        "clip": clip.name,
        "scene_id": clip.scene_id,
        "split": clip.split,
        "eligible_gt_objects": len(prompt_ground_truth.instance_ids),
        "all_visible_gt_objects": len(
            all_ground_truth.all_visible_instance_ids
        ),
        "auto_retained_tracks": int(auto.get("retained_tracks", 0)),
        "variants": prompt_scope["variants"],
        "all_instance_scope": {
            "eligible_gt_objects": len(all_ground_truth.instance_ids),
            "variants": all_scope["variants"],
            "oracle_assignment": "auto_branch_independent_assignment",
        },
    }
    return {
        "clip": clip_result,
        "tracking_rows": prompt_scope["tracking_rows"],
        "map_rows": prompt_scope["map_rows"],
        "all_tracking_rows": all_scope["tracking_rows"],
        "all_map_rows": all_scope["map_rows"],
        "object_rows": prompt_scope["object_rows"],
        "all_object_rows": all_scope["object_rows"],
    }


def _evaluate_scope(
    *,
    scope: str,
    frame_indices: Sequence[int],
    ground_truth,
    gt_stream: torch.Tensor,
    prompt_output: torch.Tensor,
    prompt_stream: torch.Tensor,
    prompt_scores: torch.Tensor,
    auto_output: torch.Tensor,
    auto_stream: torch.Tensor,
    auto_scores: torch.Tensor,
    prompt_track_ids: Sequence[int],
    prompt_track_prompts: Sequence[str],
    auto_track_ids: Sequence[int],
    auto_track_prompts: Sequence[str],
    tracking_config,
    map_config_common: Mapping[str, Any],
    prompt_label_aliases,
    oracle_source: str,
) -> dict[str, object]:
    """Evaluate one GT scope with independent prompt and auto assignments."""

    prompt_tracking = evaluate_tracking_variants(
        scene_id=str(map_config_common["scene_id"]),
        clip_name=str(map_config_common["clip_name"]),
        frame_indices=frame_indices,
        variant_masks={PROMPT_LABEL: prompt_output},
        variant_scores={PROMPT_LABEL: prompt_scores},
        raw_variant=PROMPT_LABEL,
        track_ids=prompt_track_ids,
        track_prompts=prompt_track_prompts,
        ground_truth=ground_truth,
        config=tracking_config,
        prompt_label_aliases=prompt_label_aliases,
    )
    auto_tracking = evaluate_tracking_variants(
        scene_id=str(map_config_common["scene_id"]),
        clip_name=str(map_config_common["clip_name"]),
        frame_indices=frame_indices,
        variant_masks={AUTO_LABEL: auto_output},
        variant_scores={AUTO_LABEL: auto_scores},
        raw_variant=AUTO_LABEL,
        track_ids=auto_track_ids,
        track_prompts=auto_track_prompts,
        ground_truth=ground_truth,
        config=tracking_config,
        prompt_label_aliases=prompt_label_aliases,
    )
    # The oracle is deliberately tied to the assignment of one detector
    # branch.  Prompt scope uses the prompt branch; all-instance scope uses the
    # generic auto branch so its ceiling reflects auto-discovered slots.
    if oracle_source == PROMPT_LABEL:
        oracle_gt = ground_truth.masks
        oracle_assignments = prompt_tracking["assignments"]
        oracle_stream_source = gt_stream
        oracle_track_ids = prompt_track_ids
        oracle_track_prompts = prompt_track_prompts
        oracle_raw = PROMPT_LABEL
        oracle_output, oracle_scores = _build_oracle_variant(
            oracle_gt,
            assignments=oracle_assignments,
            sequence_shape=tuple(prompt_output.shape),
        )
        oracle_stream, _ = _build_oracle_variant(
            oracle_stream_source,
            assignments=oracle_assignments,
            sequence_shape=tuple(prompt_stream.shape),
        )
        oracle_tracking = evaluate_tracking_variants(
            scene_id=str(map_config_common["scene_id"]),
            clip_name=str(map_config_common["clip_name"]),
            frame_indices=frame_indices,
            variant_masks={
                PROMPT_LABEL: prompt_output,
                ORACLE_LABEL: oracle_output,
            },
            variant_scores={
                PROMPT_LABEL: prompt_scores,
                ORACLE_LABEL: oracle_scores,
            },
            raw_variant=oracle_raw,
            track_ids=oracle_track_ids,
            track_prompts=oracle_track_prompts,
            ground_truth=ground_truth,
            config=tracking_config,
            prompt_label_aliases=prompt_label_aliases,
        )
        prompt_row, oracle_row = (
            dict(row) for row in oracle_tracking["summary_rows"]
        )
        auto_row = dict(auto_tracking["summary_rows"][0])
        oracle_assignments_for_map = oracle_assignments
        oracle_ids_for_map = oracle_track_ids
    else:
        oracle_gt = ground_truth.masks
        oracle_assignments = auto_tracking["assignments"]
        oracle_stream_source = gt_stream
        oracle_track_ids = auto_track_ids
        oracle_track_prompts = auto_track_prompts
        oracle_raw = AUTO_LABEL
        oracle_output, oracle_scores = _build_oracle_variant(
            oracle_gt,
            assignments=oracle_assignments,
            sequence_shape=tuple(auto_output.shape),
        )
        oracle_stream, _ = _build_oracle_variant(
            oracle_stream_source,
            assignments=oracle_assignments,
            sequence_shape=tuple(auto_stream.shape),
        )
        oracle_tracking = evaluate_tracking_variants(
            scene_id=str(map_config_common["scene_id"]),
            clip_name=str(map_config_common["clip_name"]),
            frame_indices=frame_indices,
            variant_masks={
                AUTO_LABEL: auto_output,
                ORACLE_LABEL: oracle_output,
            },
            variant_scores={
                AUTO_LABEL: auto_scores,
                ORACLE_LABEL: oracle_scores,
            },
            raw_variant=oracle_raw,
            track_ids=oracle_track_ids,
            track_prompts=oracle_track_prompts,
            ground_truth=ground_truth,
            config=tracking_config,
            prompt_label_aliases=prompt_label_aliases,
        )
        auto_row, oracle_row = (
            dict(row) for row in oracle_tracking["summary_rows"]
        )
        prompt_row = dict(prompt_tracking["summary_rows"][0])
        oracle_assignments_for_map = oracle_assignments
        oracle_ids_for_map = oracle_track_ids

    rows = [prompt_row, auto_row, oracle_row]
    for row in rows:
        row["evaluation_scope"] = str(scope)
    prompt_map = _evaluate_map(
        variant=PROMPT_LABEL,
        policy=f"{scope}_prompt_sam_all_visible",
        masks=prompt_stream,
        scores=prompt_scores,
        assignments=prompt_tracking["assignments"],
        track_ids=prompt_track_ids,
        common={
            **map_config_common,
            "gt_masks": gt_stream,
            "gt_instance_ids": ground_truth.instance_ids,
            "gt_labels": ground_truth.labels,
        },
    )
    auto_map = _evaluate_map(
        variant=AUTO_LABEL,
        policy=f"{scope}_auto_visual_points_all_visible",
        masks=auto_stream,
        scores=auto_scores,
        assignments=auto_tracking["assignments"],
        track_ids=auto_track_ids,
        common={
            **map_config_common,
            "gt_masks": gt_stream,
            "gt_instance_ids": ground_truth.instance_ids,
            "gt_labels": ground_truth.labels,
        },
    )
    oracle_map = _evaluate_map(
        variant=ORACLE_LABEL,
        policy=f"{scope}_{oracle_source}_assignment_gt_mask_oracle",
        masks=oracle_stream,
        scores=oracle_scores,
        assignments=oracle_assignments_for_map,
        track_ids=oracle_ids_for_map,
        common={
            **map_config_common,
            "gt_masks": gt_stream,
            "gt_instance_ids": ground_truth.instance_ids,
            "gt_labels": ground_truth.labels,
        },
    )
    for row in (prompt_map, auto_map, oracle_map):
        row["evaluation_scope"] = str(scope)
    by_variant = {
        PROMPT_LABEL: {"tracking": prompt_row, "map": prompt_map},
        AUTO_LABEL: {"tracking": auto_row, "map": auto_map},
        ORACLE_LABEL: {
            "tracking": oracle_row,
            "map": oracle_map,
        },
    }
    object_rows = []
    object_rows.extend(
        dict(row) for row in prompt_tracking["object_rows"]
    )
    object_rows.extend(dict(row) for row in auto_tracking["object_rows"])
    for row in object_rows:
        row["evaluation_scope"] = str(scope)
    object_rows.extend(
        dict(row)
        for row in oracle_tracking["object_rows"]
        if str(row.get("variant")) == ORACLE_LABEL
    )
    for row in object_rows:
        row["evaluation_scope"] = str(scope)
    return {
        "scope": str(scope),
        "variants": by_variant,
        "tracking_rows": rows,
        "map_rows": [prompt_map, auto_map, oracle_map],
        "object_rows": object_rows,
    }


def _evaluate_map(
    *,
    variant: str,
    policy: str,
    masks: torch.Tensor,
    scores: torch.Tensor,
    assignments,
    track_ids,
    common: Mapping[str, Any],
) -> dict[str, object]:
    result = evaluate_semantic_object_map(
        variant=variant,
        map_policy=policy,
        predicted_masks=masks,
        track_scores=scores,
        assignments=assignments,
        map_write_mask=masks.flatten(2).any(dim=2),
        track_ids=track_ids,
        **common,
    )
    return dict(result["summary"])


def _load_raw_branch(
    payload: Mapping[str, Any],
    raw_cache_variant: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fields = (
        "tracking_variant_masks_output",
        "tracking_variant_masks_stream",
        "tracking_variant_scores",
    )
    values = []
    for field in fields:
        variants = payload.get(field)
        if not isinstance(variants, Mapping) or raw_cache_variant not in variants:
            raise ValueError(
                f"Cache lacks {field}[{raw_cache_variant!r}]."
            )
        values.append(torch.as_tensor(variants[raw_cache_variant]))
    return values[0].bool(), values[1].bool(), values[2].float()


def _load_auto_artifact(
    path: Path,
    clip: ClipConfig,
) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing auto-proposal artifact for {clip.name}: {path}"
        )
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, Mapping):
        raise ValueError(f"Invalid auto-proposal artifact: {path}")
    if str(value.get("clip_name", "")) != clip.name:
        raise ValueError(f"Auto artifact identity differs for {clip.name}.")
    if str(value.get("scene_id", "")) != clip.scene_id:
        raise ValueError(f"Auto artifact scene differs for {clip.name}.")
    if int(value.get("candidate_generation_gt_fields", -1)) != 0:
        raise ValueError("Auto artifact does not prove GT-free generation.")
    required = (
        "tracking_masks_output",
        "tracking_masks_stream",
        "tracking_scores",
        "track_ids",
        "track_prompts",
    )
    missing = [name for name in required if name not in value]
    if missing:
        raise ValueError(f"Auto artifact lacks fields {missing}: {path}")
    expected_frames = tuple(int(value) for value in clip.frame_indices)
    artifact_frames = tuple(
        int(item) for item in value.get("frame_indices", expected_frames)
    )
    if artifact_frames != expected_frames:
        raise ValueError(
            f"Auto artifact frame order differs for {clip.name}: {path}"
        )
    masks = torch.as_tensor(value["tracking_masks_output"])
    stream_masks = torch.as_tensor(value["tracking_masks_stream"])
    scores = torch.as_tensor(value["tracking_scores"])
    if masks.ndim != 4 or stream_masks.ndim != 4 or scores.ndim != 2:
        raise ValueError(f"Auto artifact tensors have invalid ranks: {path}")
    if masks.shape[:2] != scores.shape or masks.shape[0] != len(expected_frames):
        raise ValueError(f"Auto artifact output/score shapes disagree: {path}")
    if stream_masks.shape[:2] != masks.shape[:2]:
        raise ValueError(f"Auto artifact stream shape disagrees: {path}")
    if len(value["track_ids"]) != masks.shape[1] or len(
        value["track_prompts"]
    ) != masks.shape[1]:
        raise ValueError(f"Auto artifact track metadata disagrees: {path}")
    return value


def _aggregate(
    tracking_rows: Sequence[Mapping[str, object]],
    map_rows: Sequence[Mapping[str, object]],
    clip_rows: Sequence[Mapping[str, object]],
    *,
    scope: str = PROMPT_SCOPE,
) -> dict[str, object]:
    tracking_metrics = (
        "mean_frame_iou",
        "positive_iou",
        "frame_idf1",
        "pixel_idf1",
        "tracking_recall_025",
        "tracking_recall_050",
        "matched_tracks",
        "unmatched_gt_objects",
        "fragmentation_count",
        "merge_error_count",
    )
    map_metrics = (
        "voxel_iou_5cm",
        "fscore_5cm",
        "ghost_point_ratio",
        "object_accuracy_m",
        "object_completeness_m",
    )
    tracking = _group_means(tracking_rows, tracking_metrics)
    maps = _group_means(map_rows, map_metrics)
    by_tracking = {str(row["variant"]): row for row in tracking}
    by_map = {str(row["variant"]): row for row in maps}
    prompt = by_tracking[PROMPT_LABEL]
    auto = by_tracking[AUTO_LABEL]
    prompt_map = by_map[PROMPT_LABEL]
    auto_map = by_map[AUTO_LABEL]
    comparison = {
        "delta_tracking_iou": _difference(
            auto.get("mean_frame_iou_mean"),
            prompt.get("mean_frame_iou_mean"),
        ),
        "delta_frame_idf1": _difference(
            auto.get("frame_idf1_mean"),
            prompt.get("frame_idf1_mean"),
        ),
        "delta_matched_tracks": _difference(
            auto.get("matched_tracks_mean"),
            prompt.get("matched_tracks_mean"),
        ),
        "delta_map_voxel_iou_5cm": _difference(
            auto_map.get("voxel_iou_5cm_mean"),
            prompt_map.get("voxel_iou_5cm_mean"),
        ),
        "delta_map_fscore_5cm": _difference(
            auto_map.get("fscore_5cm_mean"),
            prompt_map.get("fscore_5cm_mean"),
        ),
        "clips_tracking_iou_improved": sum(
            _clip_delta(row, "mean_frame_iou", scope=scope) > 0.0
            for row in clip_rows
        ),
        "clips_frame_idf1_improved": sum(
            _clip_delta(row, "frame_idf1", scope=scope) > 0.0
            for row in clip_rows
        ),
        "clip_count": len(clip_rows),
        "evaluation_scope": str(scope),
    }
    return {
        "tracking": tracking,
        "map": maps,
        "comparison": comparison,
    }


def _decision(
    aggregate: Mapping[str, Any],
    clip_rows: Sequence[Mapping[str, Any]],
    *,
    scope: str = PROMPT_SCOPE,
) -> dict[str, object]:
    comparison = aggregate["comparison"]
    validation = [
        row
        for row in clip_rows
        if str(row.get("split", "")).lower() == "validation"
    ]
    selection_rows = validation or list(clip_rows)
    validation_iou_nonregression = all(
        _clip_delta(row, "mean_frame_iou", scope=scope) >= 0.0
        for row in selection_rows
    )
    validation_idf1_nonregression = all(
        _clip_delta(row, "frame_idf1", scope=scope) >= 0.0
        for row in selection_rows
    )
    aggregate_iou_improved = float(
        comparison["delta_tracking_iou"]
    ) > 0.0
    aggregate_idf1_improved = float(
        comparison["delta_frame_idf1"]
    ) > 0.0
    status = (
        "GO"
        if validation_iou_nonregression
        and validation_idf1_nonregression
        and aggregate_iou_improved
        and aggregate_idf1_improved
        else "NO_GO"
    )
    return {
        "status": status,
        "scope": f"mask_discovery_and_tracking_only:{scope}",
        "validation_or_all_iou_nonregression": int(
            validation_iou_nonregression
        ),
        "validation_or_all_idf1_nonregression": int(
            validation_idf1_nonregression
        ),
        "aggregate_iou_improved": int(aggregate_iou_improved),
        "aggregate_idf1_improved": int(aggregate_idf1_improved),
        "map_metrics_are_reported_but_not_a_detection_gate": 1,
        "semantic_labels_resolved": 0,
    }


def _group_means(
    rows: Sequence[Mapping[str, object]],
    metrics: Sequence[str],
) -> list[dict[str, object]]:
    variants = sorted({str(row["variant"]) for row in rows})
    output = []
    for variant in variants:
        current = [row for row in rows if str(row["variant"]) == variant]
        summary: dict[str, object] = {
            "variant": variant,
            "clips": len(current),
        }
        for metric in metrics:
            values = [
                value
                for row in current
                if (value := _finite(row.get(metric))) is not None
            ]
            summary[f"{metric}_mean"] = (
                sum(values) / len(values) if values else float("nan")
            )
        output.append(summary)
    return output


def _clip_delta(
    row: Mapping[str, Any],
    metric: str,
    *,
    scope: str = PROMPT_SCOPE,
) -> float:
    variants = _scope_variants(row, scope)
    return _difference(
        variants[AUTO_LABEL]["tracking"].get(metric),
        variants[PROMPT_LABEL]["tracking"].get(metric),
    )


def _scope_variants(
    row: Mapping[str, Any],
    scope: str,
) -> Mapping[str, Any]:
    if str(scope) == PROMPT_SCOPE:
        return row["variants"]
    nested = row.get(str(scope))
    if not isinstance(nested, Mapping) or "variants" not in nested:
        raise ValueError(
            f"Clip row lacks evaluation scope {scope!r}."
        )
    return nested["variants"]


def _difference(left: Any, right: Any) -> float:
    left_value = _finite(left)
    right_value = _finite(right)
    if left_value is None or right_value is None:
        return float("nan")
    return left_value - right_value


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _clip_variant_rows(
    clips: Sequence[Mapping[str, Any]],
    *,
    scope: str = PROMPT_SCOPE,
) -> list[dict[str, object]]:
    rows = []
    for clip in clips:
        variants = _scope_variants(clip, scope)
        for variant, values in variants.items():
            rows.append(
                {
                    "clip": clip["clip"],
                    "scene_id": clip["scene_id"],
                    "split": clip["split"],
                    "evaluation_scope": str(scope),
                    "variant": variant,
                    **{
                        f"tracking_{key}": value
                        for key, value in values["tracking"].items()
                        if key not in {"clip", "scene_id", "variant"}
                    },
                    **{
                        f"map_{key}": value
                        for key, value in values["map"].items()
                        if key not in {"clip", "scene_id", "variant"}
                    },
                }
            )
    return rows


def _print_clip(clip: Mapping[str, Any]) -> None:
    print(
        f"  clip={clip['clip']} split={clip['split']} "
        f"eligible_gt={clip['eligible_gt_objects']} "
        f"auto_tracks={clip['auto_retained_tracks']}"
    )
    for variant in (PROMPT_LABEL, AUTO_LABEL, ORACLE_LABEL):
        value = clip["variants"][variant]
        tracking = value["tracking"]
        maps = value["map"]
        print(
            f"    {variant} IoU={float(tracking['mean_frame_iou']):.5f} "
            f"IDF1={float(tracking['frame_idf1']):.5f} "
            f"matched={int(tracking['matched_tracks'])} "
            f"voxelIoU5={float(maps['voxel_iou_5cm']):.5f} "
            f"F5cm={float(maps['fscore_5cm']):.5f}"
        )
    all_scope = clip.get("all_instance_scope")
    if isinstance(all_scope, Mapping):
        print(
            f"    all-instance GT objects={int(all_scope['eligible_gt_objects'])}"
        )
        variants = all_scope["variants"]
        for variant in (PROMPT_LABEL, AUTO_LABEL, ORACLE_LABEL):
            value = variants[variant]
            tracking = value["tracking"]
            maps = value["map"]
            print(
                f"    all[{variant}] IoU={float(tracking['mean_frame_iou']):.5f} "
                f"IDF1={float(tracking['frame_idf1']):.5f} "
                f"matched={int(tracking['matched_tracks'])} "
                f"voxelIoU5={float(maps['voxel_iou_5cm']):.5f} "
                f"F5cm={float(maps['fscore_5cm']):.5f}"
            )


def _write_copyable(path: Path, summary: Mapping[str, Any]) -> None:
    comparison = summary["aggregate"]["comparison"]
    tracking = {
        str(row["variant"]): row
        for row in summary["aggregate"]["tracking"]
    }
    maps = {
        str(row["variant"]): row
        for row in summary["aggregate"]["map"]
    }
    all_aggregate = summary["all_instance_scope"]["aggregate"]
    all_tracking = {
        str(row["variant"]): row
        for row in all_aggregate["tracking"]
    }
    all_maps = {
        str(row["variant"]): row
        for row in all_aggregate["map"]
    }
    lines = [
        "===== SAM31_AUTO_PROPOSAL_BEGIN =====",
        f"revision={REVISION}",
        "candidate_generation_gt_fields=0",
        "auto_semantic_text_prompt_used=0",
        "auto_proposal=deterministic visual-point grid",
        "assignment=prompt and auto evaluated with independent frozen assignments",
        f"decision={summary['decision']['status']}",
        f"prompt_tracking={json.dumps(_json_safe(tracking[PROMPT_LABEL]), sort_keys=True)}",
        f"auto_tracking={json.dumps(_json_safe(tracking[AUTO_LABEL]), sort_keys=True)}",
        f"oracle_tracking={json.dumps(_json_safe(tracking[ORACLE_LABEL]), sort_keys=True)}",
        f"prompt_map={json.dumps(_json_safe(maps[PROMPT_LABEL]), sort_keys=True)}",
        f"auto_map={json.dumps(_json_safe(maps[AUTO_LABEL]), sort_keys=True)}",
        f"comparison={json.dumps(_json_safe(comparison), sort_keys=True)}",
        "all_instance_scope=all positive instance IDs in the selected clip",
        f"all_instance_prompt_tracking={json.dumps(_json_safe(all_tracking[PROMPT_LABEL]), sort_keys=True)}",
        f"all_instance_auto_tracking={json.dumps(_json_safe(all_tracking[AUTO_LABEL]), sort_keys=True)}",
        f"all_instance_oracle_tracking={json.dumps(_json_safe(all_tracking[ORACLE_LABEL]), sort_keys=True)}",
        f"all_instance_prompt_map={json.dumps(_json_safe(all_maps[PROMPT_LABEL]), sort_keys=True)}",
        f"all_instance_auto_map={json.dumps(_json_safe(all_maps[AUTO_LABEL]), sort_keys=True)}",
        f"all_instance_comparison={json.dumps(_json_safe(all_aggregate['comparison']), sort_keys=True)}",
        "semantic_caveat=auto tracks carry generic object identity only; category classification is not solved in this branch",
        "interpretation=prompt_scope decision is retained for historical comparability; all_instance_scope is diagnostic open-world recall and is not a GO gate",
        "===== SAM31_AUTO_PROPOSAL_END =====",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True) + "\n",
        encoding="utf8",
    )
    return path


def _json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, torch.Tensor):
        return value.tolist()
    return value


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = [dict(row) for row in rows]
    fieldnames = sorted({key for row in values for key in row})
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in values:
            writer.writerow(
                {
                    key: _csv_value(row.get(key))
                    for key in fieldnames
                }
            )
    return path


def _csv_value(value: object) -> object:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_json_safe(value), sort_keys=True)
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def _select_clips(
    clips: Sequence[ClipConfig],
    requested: Sequence[str] | None,
) -> tuple[ClipConfig, ...]:
    if not requested:
        return tuple(clips)
    names = {str(value) for value in requested}
    selected = tuple(
        clip
        for clip in clips
        if clip.name in names or clip.scene_id in names
    )
    if not selected:
        raise ValueError(f"No requested clip/scene found: {sorted(names)}")
    return selected


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="streaming_couping/configs/v0_baseline.yaml",
    )
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--clip", action="append")
    parser.add_argument("--raw-cache-variant", default="sam31_online_forward")
    return parser.parse_args()


if __name__ == "__main__":
    main()
