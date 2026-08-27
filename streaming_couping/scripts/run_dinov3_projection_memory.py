#!/usr/bin/env python3
"""Run the DINOv3-assisted projection-memory semantic-map ablation.

This experiment gives each component one job:

* StreamVGGT supplies the frozen world pointmap and camera geometry.
* SAM3 supplies masks and short-term source-track IDs.
* DINOv3 is used only as an appearance cue when a source track re-enters and
  must be associated with an existing world-memory object.

No point coordinates are regressed or corrected.  The only changed operation
is the score used for causal object re-association.  All branch artifacts are
written before ScanNet++ GT is opened for evaluation.
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

from streaming_couping.src.learned_pose.cache import cache_path, load_feature_cache
from streaming_couping.src.learned_pose.config import ClipConfig, load_learned_pose_config
from streaming_couping.src.object_memory import collapse_persistent_tracks
from streaming_couping.src.projection_object_memory import (
    ProjectionAssociationConfig,
    ProjectionObjectMemory,
    ProjectionObjectMemoryConfig,
)
from streaming_couping.src.semantic_map import normalize_confidence
from streaming_couping.src.semantic_map_metrics import (
    apply_similarity,
    evaluate_semantic_object_map,
    load_ground_truth_stream_masks,
    object_point_metrics,
)
from streaming_couping.src.semantic_tracking_metrics import (
    evaluate_tracking_variants,
    load_ground_truth_instances,
)
from streaming_couping.src.storage import expand_storage_path

from streaming_couping.scripts.run_semantic_map_evaluation import (
    _load_run as _load_evaluation_run,
)


REVISION = "dinov3_projection_temporal_memory_ablation_r1"
RAW_CACHE_VARIANT = "sam31_online_forward"

# These are deliberately constants rather than command-line knobs.  The
# controls must be fixed before the sealed test split is opened.
APPEARANCE_WEIGHT = 0.20
REASSOCIATION_GAP = 3
CONFIRMATION_FRAMES = 2
CONFIRMATION_WINDOW = 4
SHUFFLE_SEED = 2026

BRANCH_FEATURES: dict[str, tuple[str, str] | None] = {
    "world_memory_only": None,
    "single_view_dino": ("single_features", "single_valid"),
    "persistent_dino": ("persistent_features", "persistent_valid"),
    "shuffled_persistent_dino": (
        "shuffled_persistent_features",
        "shuffled_persistent_valid",
    ),
}


@dataclass
class FrozenBranch:
    """A branch after causal generation and before GT evaluation."""

    key: str
    output_masks: torch.Tensor
    output_scores: torch.Tensor
    output_track_ids: tuple[int, ...]
    output_prompts: tuple[str, ...]
    stream_masks: torch.Tensor
    stream_scores: torch.Tensor
    stream_track_ids: tuple[int, ...]
    stream_prompts: tuple[str, ...]
    persistent_ids_by_sam_slot: torch.Tensor
    map_write_by_sam_slot: torch.Tensor
    map_write: torch.Tensor
    fused_map: dict[str, torch.Tensor]
    object_metadata: list[dict[str, object]]
    association_summary: dict[str, object]
    artifact_path: Path | None = None
    memory: ProjectionObjectMemory | None = None


def main() -> None:
    args = _parse_args()
    config_path = Path(args.config).expanduser().resolve()
    data = load_learned_pose_config(config_path)
    evaluation_run = _load_evaluation_run(config_path)
    clips = _select_clips(data.clips, args.clip)
    feature_dir = (
        Path(args.feature_dir).expanduser().resolve()
        if args.feature_dir
        else expand_storage_path(
            "${VGGT_SAM_STORAGE_ROOT}/outputs/"
            "streaming_couping_dinov3_object_features",
            base=config_path.parent,
        )
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else expand_storage_path(
            "${VGGT_SAM_STORAGE_ROOT}/outputs/"
            "streaming_couping_dinov3_projection_memory",
            base=config_path.parent,
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("DINOv3 PROJECTION-MEMORY SEMANTIC-MAP ABLATION")
    print(f"protocol={config_path}")
    print(f"feature_dir={feature_dir}")
    print(
        "frozen=StreamVGGT pointmap/camera + SAM3 masks/IDs + DINOv3; "
        "no XYZ correction or model update"
    )
    print(
        f"appearance_weight={APPEARANCE_WEIGHT} "
        f"reassociation_gap={REASSOCIATION_GAP} "
        f"confirmation={CONFIRMATION_FRAMES}/{CONFIRMATION_WINDOW} "
        f"shuffle_seed={SHUFFLE_SEED}"
    )
    print(
        "branches=raw_all_visible,world_memory_only,single_view_dino,"
        "persistent_dino,shuffled_persistent_dino"
    )
    print("GT is not opened during causal branch generation")

    clip_states: list[dict[str, object]] = []
    all_tracking_rows: list[dict[str, object]] = []
    all_map_rows: list[dict[str, object]] = []
    all_association_rows: list[dict[str, object]] = []
    all_object_rows: list[dict[str, object]] = []

    # Phase 1: load frozen model outputs, run all causal branches, and freeze
    # branch artifacts.  No manifest/annotation reader is called in this loop.
    frozen_by_clip: list[tuple[ClipConfig, Mapping[str, Any], dict[str, FrozenBranch]]] = []
    for clip in clips:
        payload = _load_frozen_payload(data, clip, args.raw_cache_variant)
        dino_cache_path = feature_dir / f"{clip.name}.pt"
        dino = _load_dino_cache(
            dino_cache_path,
            payload,
            raw_cache_variant=args.raw_cache_variant,
        )
        if isinstance(payload, dict):
            payload["dino_cache_path"] = str(dino_cache_path)
        branches = _build_branches(
            payload,
            dino=dino,
            raw_cache_variant=args.raw_cache_variant,
        )
        clip_dir = output_dir / clip.name
        for branch in branches.values():
            branch_dir = clip_dir / branch.key
            branch_dir.mkdir(parents=True, exist_ok=True)
            branch.artifact_path = _write_artifact(
                branch_dir / "semantic_map.pt",
                payload=payload,
                branch=branch,
                dino_cache_path=feature_dir / f"{clip.name}.pt",
            )
            _write_json(
                branch_dir / "association_summary.json",
                branch.association_summary,
            )
            _write_csv(
                branch_dir / "association_events.csv",
                _events_from_branch(branch),
            )
            if branch.memory is not None:
                branch.memory.save_json(branch_dir / "object_memory.json")
                branch.memory.save_csv(branch_dir / "object_memory.csv")
        # Re-open the just-written artifacts for evaluation. This makes the
        # phase boundary real: evaluation cannot accidentally consume a
        # mutable in-memory branch that differs from the serialized result.
        frozen_branches = {
            key: _load_branch_artifact(value.artifact_path)
            for key, value in branches.items()
        }
        frozen_by_clip.append((clip, payload, frozen_branches))
        print(
            f"  frozen clip={clip.name} "
            + " ".join(
                f"{key}:objects={int(value.association_summary['persistent_object_count'])}"
                f",candidates={int(value.association_summary['candidate_event_count'])}"
                f",appearance={int(value.association_summary['appearance_compared_count'])}"
                for key, value in branches.items()
            )
        )

    print("all causal artifacts frozen; opening sealed GT evaluation")

    # Phase 2: evaluate only the frozen branch tensors.
    for clip, payload, branches in frozen_by_clip:
        evaluated = _evaluate_clip(
            data=data,
            clip=clip,
            payload=payload,
            branches=branches,
            tracking_config=evaluation_run.tracking,
            map_config=evaluation_run.map_metrics,
            prompt_label_aliases=evaluation_run.prompt_label_aliases,
            raw_cache_variant=args.raw_cache_variant,
        )
        clip_states.append(evaluated["clip"])
        all_tracking_rows.extend(evaluated["tracking_rows"])
        all_map_rows.extend(evaluated["map_rows"])
        all_association_rows.extend(evaluated["association_rows"])
        all_object_rows.extend(evaluated["object_rows"])
        _print_clip_result(evaluated)

    aggregate = _aggregate(
        tracking_rows=all_tracking_rows,
        map_rows=all_map_rows,
        association_rows=all_association_rows,
    )
    decision = _decision(aggregate, all_association_rows)
    summary: dict[str, object] = {
        "schema": 1,
        "revision": REVISION,
        "protocol": str(config_path),
        "feature_dir": str(feature_dir),
        "clip_count": len(clip_states),
        "candidate_generation_gt_fields": 0,
        "evaluation_gt_fields": 1,
        "pointmap_modified": 0,
        "streamvggt_pose_modified": 0,
        "dino_used_for_xyz_regression": 0,
        "fixed_policy": {
            "appearance_weight": APPEARANCE_WEIGHT,
            "reassociation_gap": REASSOCIATION_GAP,
            "confirmation_frames": CONFIRMATION_FRAMES,
            "confirmation_window": CONFIRMATION_WINDOW,
            "shuffle_seed": SHUFFLE_SEED,
            "raw_cache_variant": args.raw_cache_variant,
        },
        "branches": list(_branch_names()),
        "clips": clip_states,
        "tracking_summary": all_tracking_rows,
        "map_summary": all_map_rows,
        "association_summary": all_association_rows,
        "object_summary": all_object_rows,
        "aggregate": aggregate,
        "decision": decision,
        "outputs": {},
    }
    summary_path = output_dir / "summary.json"
    _write_json(summary_path, summary)
    outputs = {
        "summary": summary_path,
        "clip_variant_summary": _write_csv(
            output_dir / "clip_variant_summary.csv",
            _clip_variant_rows(clip_states),
        ),
        "tracking_summary": _write_csv(
            output_dir / "tracking_summary.csv", all_tracking_rows
        ),
        "map_summary": _write_csv(output_dir / "map_summary.csv", all_map_rows),
        "association_summary": _write_csv(
            output_dir / "association_summary.csv", all_association_rows
        ),
        "object_summary": _write_csv(
            output_dir / "object_summary.csv", all_object_rows
        ),
    }
    summary["outputs"] = {key: str(value) for key, value in outputs.items()}
    _write_json(summary_path, summary)
    copyable = output_dir / "copyable_result.txt"
    _write_copyable(copyable, summary)
    print(f"summary={summary_path}")
    print(f"copyable_result={copyable}")
    print(f"decision={decision['status']}")


def _build_branches(
    payload: Mapping[str, Any],
    *,
    dino: Mapping[str, torch.Tensor],
    raw_cache_variant: str,
) -> dict[str, FrozenBranch]:
    output_masks, stream_masks, scores = _load_raw_masks(
        payload, raw_cache_variant
    )
    sequence, tracks = (int(value) for value in scores.shape)
    raw_track_ids = tuple(int(value) for value in payload["sam_track_ids"])
    raw_prompts = tuple(str(value) for value in payload["sam_track_prompts"])
    raw_persistent_ids = torch.arange(tracks, dtype=torch.long).view(1, -1).expand(
        sequence, -1
    ).clone()
    raw_write_by_slot = stream_masks.flatten(2).any(dim=2)
    raw_branch = FrozenBranch(
        key="raw_all_visible",
        output_masks=output_masks,
        output_scores=scores,
        output_track_ids=raw_track_ids,
        output_prompts=raw_prompts,
        stream_masks=stream_masks,
        stream_scores=scores,
        stream_track_ids=raw_track_ids,
        stream_prompts=raw_prompts,
        persistent_ids_by_sam_slot=raw_persistent_ids,
        map_write_by_sam_slot=raw_write_by_slot,
        map_write=raw_write_by_slot,
        fused_map=_empty_fused_map(),
        object_metadata=[],
        association_summary={
            "branch": "raw_all_visible",
            "appearance_source": "none",
            "appearance_weight": 0.0,
            "reassociation_gap": 0,
            "persistent_object_count": tracks,
            "pending_track_count": 0,
            "confirmed_observation_count": int(raw_write_by_slot.sum()),
            "event_count": int(sequence * tracks),
            "candidate_event_count": 0,
            "candidate_count_total": 0,
            "appearance_compared_count": 0,
            "forced_reassociation_count": 0,
            "appearance_valid_observation_count": 0,
            "accepted_existing_association_count": 0,
            "candidate_generation_exercised": 0,
            "appearance_exercised": 0,
        },
    )
    branches: dict[str, FrozenBranch] = {raw_branch.key: raw_branch}
    for key, feature_keys in BRANCH_FEATURES.items():
        appearance = None
        appearance_valid = None
        if feature_keys is not None:
            appearance = dino[feature_keys[0]]
            appearance_valid = dino[feature_keys[1]]
        branches[key] = _build_memory_branch(
            key=key,
            payload=payload,
            output_masks=output_masks,
            stream_masks=stream_masks,
            scores=scores,
            appearance=appearance,
            appearance_valid=appearance_valid,
            raw_track_ids=raw_track_ids,
            raw_prompts=raw_prompts,
        )
    return branches


def _build_memory_branch(
    *,
    key: str,
    payload: Mapping[str, Any],
    output_masks: torch.Tensor,
    stream_masks: torch.Tensor,
    scores: torch.Tensor,
    appearance: torch.Tensor | None,
    appearance_valid: torch.Tensor | None,
    raw_track_ids: tuple[int, ...],
    raw_prompts: tuple[str, ...],
) -> FrozenBranch:
    config = _memory_config(appearance_weight=APPEARANCE_WEIGHT if appearance is not None else 0.0)
    memory = ProjectionObjectMemory(config)
    confidence = normalize_confidence(payload["baseline_world_confidence"])
    result = memory.process_sequence(
        world_points=payload["baseline_world_points"],
        confidence=confidence,
        masks=stream_masks,
        track_scores=scores,
        frame_indices=tuple(int(value) for value in payload["frame_indices"]),
        track_ids=raw_track_ids,
        track_prompts=raw_prompts,
        world_to_camera=payload["baseline_world_to_camera"],
        intrinsics=payload["baseline_intrinsics"],
        image_size=tuple(int(value) for value in payload["image_size"]),
        images=payload.get("stream_images"),
        appearance=appearance,
        appearance_valid=appearance_valid,
    )
    persistent_ids = result["persistent_object_ids"].detach().long().cpu()
    object_rows = [dict(row) for row in result["objects"]]
    object_ids = tuple(int(row["object_id"]) for row in object_rows)
    object_prompts = {
        int(row["object_id"]): str(row.get("category", "object"))
        for row in object_rows
    }
    collapsed_output = collapse_persistent_tracks(
        masks=output_masks,
        scores=scores,
        persistent_object_ids=persistent_ids,
        object_ids=object_ids,
        object_prompts=object_prompts,
    )
    collapsed_stream = collapse_persistent_tracks(
        masks=stream_masks,
        scores=scores,
        persistent_object_ids=persistent_ids,
        object_ids=object_ids,
        object_prompts=object_prompts,
    )
    write_by_slot = result["map_write_mask"].detach().bool().cpu()
    write = _collapse_write_mask(write_by_slot, persistent_ids, object_ids)
    stats = _association_summary(
        key=key,
        result=result,
        appearance_source=("none" if appearance is None else key),
        appearance_weight=config.association.appearance_weight,
    )
    return FrozenBranch(
        key=key,
        output_masks=collapsed_output["masks"],
        output_scores=collapsed_output["scores"],
        output_track_ids=tuple(int(value) for value in object_ids),
        output_prompts=tuple(str(value) for value in collapsed_output["prompts"]),
        stream_masks=collapsed_stream["masks"],
        stream_scores=collapsed_stream["scores"],
        stream_track_ids=tuple(int(value) for value in object_ids),
        stream_prompts=tuple(str(value) for value in collapsed_stream["prompts"]),
        persistent_ids_by_sam_slot=persistent_ids,
        map_write_by_sam_slot=write_by_slot,
        map_write=write,
        fused_map={
            str(name): value.detach().cpu()
            for name, value in result["fused_map"].items()
        },
        object_metadata=object_rows,
        association_summary=stats,
        memory=memory,
    )


def _memory_config(*, appearance_weight: float) -> ProjectionObjectMemoryConfig:
    return ProjectionObjectMemoryConfig(
        max_points_per_object=4096,
        max_fused_voxels_per_object=20000,
        min_observation_points=16,
        min_mask_pixels=32,
        min_track_score=0.50,
        min_geometry_confidence=0.30,
        center_ema_alpha=0.25,
        confirmation_frames=CONFIRMATION_FRAMES,
        confirmation_window=CONFIRMATION_WINDOW,
        max_pending_gap=4,
        reassociation_gap=REASSOCIATION_GAP,
        voxel_size_ratio=0.02,
        absolute_voxel_size=0.02,
        association=ProjectionAssociationConfig(
            projection_iou_weight=0.70,
            projection_recall_weight=0.12,
            projection_precision_weight=0.08,
            center_weight=0.05,
            category_weight=0.05,
            appearance_weight=float(appearance_weight),
            min_projection_iou=0.025,
            min_match_score=0.28,
            min_projected_pixels=8,
            top_k=5,
            projection_dilation_radius=3,
            max_projection_points=4096,
            center_scale_ratio=0.12,
            absolute_center_scale=0.02,
            category_hard_gate=True,
        ),
    )


def _association_summary(
    *,
    key: str,
    result: Mapping[str, object],
    appearance_source: str,
    appearance_weight: float,
) -> dict[str, object]:
    events = [dict(row) for row in result["events"]]
    candidate_events = [
        row for row in events if int(row.get("association_candidate_count", 0)) > 0
    ]
    appearance_events = [
        row
        for row in events
        if int(row.get("association_appearance_compared", 0))
    ]
    accepted_actions = {
        "confirm_associate_existing_object",
        "tentative_backfilled",
    }
    return {
        "branch": key,
        "appearance_source": appearance_source,
        "appearance_weight": float(appearance_weight),
        "reassociation_gap": REASSOCIATION_GAP,
        "persistent_object_count": int(result["persistent_object_count"]),
        "pending_track_count": int(result["pending_track_count"]),
        "confirmed_observation_count": int(result["confirmed_observation_count"]),
        "event_count": len(events),
        "candidate_event_count": len(candidate_events),
        "candidate_count_total": int(
            sum(int(row.get("association_candidate_count", 0)) for row in events)
        ),
        "appearance_compared_count": len(appearance_events),
        "forced_reassociation_count": int(
            sum(int(row.get("reassociation_forced", 0)) for row in events)
        ),
        "appearance_valid_observation_count": int(
            sum(int(row.get("observation_appearance_valid", 0)) for row in events)
        ),
        "accepted_existing_association_count": int(
            sum(str(row.get("action", "")) in accepted_actions for row in events)
        ),
        "candidate_generation_exercised": int(bool(candidate_events)),
        "appearance_exercised": int(bool(appearance_events)),
        "event_action_counts": _action_counts(events),
    }


def _action_counts(events: Sequence[Mapping[str, object]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in events:
        action = str(row.get("action", ""))
        counts[action] = counts.get(action, 0) + 1
    return dict(sorted(counts.items()))


def _evaluate_clip(
    *,
    data,
    clip: ClipConfig,
    payload: Mapping[str, Any],
    branches: Mapping[str, FrozenBranch],
    tracking_config,
    map_config,
    prompt_label_aliases,
    raw_cache_variant: str,
) -> dict[str, object]:
    # This is the first function in the runner that reads annotation-backed
    # GT. All branch inputs below came from the frozen artifacts above.
    raw = branches["raw_all_visible"]
    frames = tuple(int(value) for value in payload["frame_indices"])
    ground_truth_output = load_ground_truth_instances(
        data.manifest,
        scene_id=str(payload["scene_id"]),
        frame_indices=frames,
        output_size=(int(raw.output_masks.shape[-2]), int(raw.output_masks.shape[-1])),
        prompts=tuple(str(value) for value in payload["instance_prompts"]),
        prompt_label_aliases=prompt_label_aliases,
    )
    processed_size = tuple(int(value) for value in payload["image_size"])
    gt_stream = load_ground_truth_stream_masks(
        data.manifest,
        scene_id=str(payload["scene_id"]),
        frame_indices=frames,
        instance_ids=ground_truth_output.instance_ids,
        processed_size=processed_size,
        image_mode=str(payload["image_mode"]),
    )
    aligned_points = apply_similarity(
        payload["baseline_world_points"],
        scale=float(payload["point_alignment_scale"]),
        rotation=payload["point_alignment_rotation"],
        translation=payload["point_alignment_translation"],
    )
    target_points = torch.as_tensor(payload["target_world_points"]).float().cpu()
    confidence = normalize_confidence(payload["baseline_world_confidence"])

    tracking_rows: list[dict[str, object]] = []
    map_rows: list[dict[str, object]] = []
    association_rows: list[dict[str, object]] = []
    object_rows: list[dict[str, object]] = []
    clip_variants: dict[str, object] = {}
    for branch in branches.values():
        tracking = evaluate_tracking_variants(
            scene_id=str(payload["scene_id"]),
            clip_name=str(payload["clip_name"]),
            frame_indices=frames,
            variant_masks={branch.key: branch.output_masks},
            variant_scores={branch.key: branch.output_scores},
            raw_variant=branch.key,
            track_ids=branch.output_track_ids,
            track_prompts=branch.output_prompts,
            ground_truth=ground_truth_output,
            config=tracking_config,
            prompt_label_aliases=prompt_label_aliases,
        )
        tracking_row = dict(tracking["summary_rows"][0])
        tracking_row["branch"] = branch.key
        tracking_row["assignment_scope"] = (
            "raw_self" if branch.key == "raw_all_visible" else "branch_self"
        )
        tracking_rows.append(tracking_row)
        map_result = evaluate_semantic_object_map(
            scene_id=str(payload["scene_id"]),
            clip_name=str(payload["clip_name"]),
            variant=branch.key,
            map_policy=(
                "all_visible_observations"
                if branch.key == "raw_all_visible"
                else "confirmed_projection_memory_observations"
            ),
            aligned_world_points=aligned_points,
            target_world_points=target_points,
            confidence=confidence,
            predicted_masks=branch.stream_masks,
            track_scores=branch.stream_scores,
            gt_masks=gt_stream,
            gt_instance_ids=ground_truth_output.instance_ids,
            gt_labels=ground_truth_output.labels,
            assignments=tracking["assignments"],
            map_write_mask=branch.map_write,
            track_ids=branch.stream_track_ids,
            config=map_config,
        )
        map_row = dict(map_result["summary"])
        map_row.update({"branch": branch.key, "map_type": "observation"})
        map_rows.append(map_row)
        for row in map_result["object_rows"]:
            current = dict(row)
            current.update({"branch": branch.key, "map_type": "observation"})
            object_rows.append(current)
        association_row = dict(branch.association_summary)
        association_row.update(
            {
                "scene_id": str(payload["scene_id"]),
                "clip": str(payload["clip_name"]),
                "split": str(clip.split),
                "artifact": str(branch.artifact_path or ""),
            }
        )
        association_rows.append(association_row)

        branch_maps: dict[str, object] = {"observation": map_row}
        fused_result = None
        if branch.key != "raw_all_visible":
            fused_result = _evaluate_fused_map(
                branch=branch,
                payload=payload,
                ground_truth=ground_truth_output,
                gt_stream=gt_stream,
                target_points=target_points,
                assignments=tracking["assignments"],
                map_config=map_config,
                scene_id=str(payload["scene_id"]),
                clip_name=str(payload["clip_name"]),
            )
            fused_row = dict(fused_result["summary"])
            fused_row.update({"branch": branch.key, "map_type": "fused"})
            map_rows.append(fused_row)
            branch_maps["fused"] = fused_row
            for row in fused_result["object_rows"]:
                current = dict(row)
                current.update({"branch": branch.key, "map_type": "fused"})
                object_rows.append(current)
        clip_variants[branch.key] = {
            "artifact": str(branch.artifact_path or ""),
            "association": association_row,
            "tracking": tracking_row,
            "maps": branch_maps,
        }

    return {
        "clip": {
            "scene_id": str(payload["scene_id"]),
            "clip_name": str(payload["clip_name"]),
            "split": str(clip.split),
            "frames": frames,
            "cache": str(cache_path(data, clip)),
            "dino_cache": str(payload.get("dino_cache_path", "")),
            "variants": clip_variants,
        },
        "tracking_rows": tracking_rows,
        "map_rows": map_rows,
        "association_rows": association_rows,
        "object_rows": object_rows,
    }


def _evaluate_fused_map(
    *,
    branch: FrozenBranch,
    payload: Mapping[str, Any],
    ground_truth,
    gt_stream: torch.Tensor,
    target_points: torch.Tensor,
    assignments: Sequence[Mapping[str, object]],
    map_config,
    scene_id: str,
    clip_name: str,
) -> dict[str, object]:
    fused_points = branch.fused_map["world_points"].detach().float().cpu()
    fused_ids = branch.fused_map["object_ids"].detach().long().cpu()
    if fused_points.ndim != 2 or tuple(fused_points.shape[-1:]) != (3,):
        raise ValueError(f"Malformed fused points for branch {branch.key}.")
    if tuple(fused_ids.shape) != (fused_points.shape[0],):
        raise ValueError(f"Malformed fused object IDs for branch {branch.key}.")
    aligned = apply_similarity(
        fused_points,
        scale=float(payload["point_alignment_scale"]),
        rotation=payload["point_alignment_rotation"],
        translation=payload["point_alignment_translation"],
    )
    object_metric_rows: list[dict[str, object]] = []
    for assignment in assignments:
        object_id = int(assignment["track_id"])
        target_index = int(assignment["gt_index"])
        predicted = aligned[fused_ids == object_id]
        target = target_points[gt_stream[:, target_index]]
        metrics = object_point_metrics(
            predicted,
            target,
            fscore_thresholds=map_config.fscore_thresholds_m,
            voxel_size=map_config.voxel_size_m,
            ghost_distance=map_config.ghost_distance_m,
            chunk_size=map_config.distance_chunk_size,
        )
        object_metric_rows.append(
            {
                "scene_id": scene_id,
                "clip": clip_name,
                "variant": f"{branch.key}_fused",
                "branch": branch.key,
                "map_type": "fused",
                "persistent_object_id": object_id,
                "gt_instance_id": int(assignment["gt_instance_id"]),
                "gt_label": str(assignment["gt_label"]),
                "predicted_points": int(predicted.shape[0]),
                "target_points": int(target.shape[0]),
                **metrics,
            }
        )
    summary = {
        "scene_id": scene_id,
        "clip": clip_name,
        "variant": f"{branch.key}_fused",
        "branch": branch.key,
        "map_type": "fused",
        "map_policy": "object_level_voxel_fusion",
        "eligible_gt_objects": int(len(ground_truth.instance_ids)),
        "matched_objects": int(len(object_metric_rows)),
        "object_accuracy_m": _mean_metric(object_metric_rows, "object_accuracy_m"),
        "object_completeness_m": _mean_metric(
            object_metric_rows, "object_completeness_m"
        ),
        "symmetric_distance_m": _mean_metric(
            object_metric_rows, "symmetric_distance_m"
        ),
        "fscore_5cm": _mean_metric(object_metric_rows, "fscore_5cm"),
        "fscore_10cm": _mean_metric(object_metric_rows, "fscore_10cm"),
        "ghost_point_ratio": _mean_metric(object_metric_rows, "ghost_point_ratio"),
        "voxel_iou_5cm": _mean_metric(object_metric_rows, "voxel_iou"),
        "fused_points": int(fused_points.shape[0]),
        "unmatched_nonempty_tracks": 0,
        "duplicate_objects": 0,
    }
    return {"summary": summary, "object_rows": object_metric_rows}


def _load_frozen_payload(
    data,
    clip: ClipConfig,
    raw_cache_variant: str,
) -> dict[str, Any]:
    path = cache_path(data, clip)
    payload = load_feature_cache(path)
    required = (
        "clip_name",
        "scene_id",
        "frame_indices",
        "image_size",
        "image_mode",
        "instance_prompts",
        "sam_track_ids",
        "sam_track_prompts",
        "baseline_world_points",
        "baseline_world_confidence",
        "baseline_world_to_camera",
        "baseline_intrinsics",
        "stream_images",
        "tracking_variant_masks_output",
        "tracking_variant_masks_stream",
        "tracking_variant_scores",
    )
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError(f"Cache {clip.name!r} lacks fields {missing}.")
    if str(payload["clip_name"]) != clip.name or str(payload["scene_id"]) != clip.scene_id:
        raise ValueError(f"Cache identity differs from configured clip {clip.name!r}.")
    frames = tuple(int(value) for value in payload["frame_indices"])
    if frames != tuple(int(value) for value in clip.frame_indices):
        raise ValueError(f"Cache frame order differs from clip {clip.name!r}.")
    masks, stream, scores = _load_raw_masks(payload, raw_cache_variant)
    points = torch.as_tensor(payload["baseline_world_points"])
    confidence = torch.as_tensor(payload["baseline_world_confidence"])
    images = torch.as_tensor(payload["stream_images"])
    if points.ndim != 4 or tuple(points.shape[-1:]) != (3,):
        raise ValueError(f"Cache {clip.name!r} has malformed world points.")
    if tuple(stream.shape[0:1]) != tuple(points.shape[0:1]) or tuple(stream.shape[2:]) != tuple(points.shape[1:3]):
        raise ValueError(f"Cache {clip.name!r} stream masks do not align with points.")
    if tuple(masks.shape[:2]) != tuple(scores.shape) or tuple(stream.shape[:2]) != tuple(scores.shape):
        raise ValueError(f"Cache {clip.name!r} mask/score shapes disagree.")
    if confidence.ndim == 4 and confidence.shape[-1] == 1:
        confidence = confidence[..., 0]
    if tuple(confidence.shape) != tuple(points.shape[:3]):
        raise ValueError(f"Cache {clip.name!r} confidence does not align with points.")
    if tuple(images.shape) != (points.shape[0], 3, points.shape[1], points.shape[2]):
        raise ValueError(f"Cache {clip.name!r} stream_images do not align with points.")
    if len(payload["sam_track_ids"]) != scores.shape[1] or len(payload["sam_track_prompts"]) != scores.shape[1]:
        raise ValueError(f"Cache {clip.name!r} track metadata does not align with masks.")
    return payload


def _load_raw_masks(
    payload: Mapping[str, Any], raw_cache_variant: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    variants = payload["tracking_variant_masks_output"]
    stream_variants = payload["tracking_variant_masks_stream"]
    score_variants = payload["tracking_variant_scores"]
    for name, values in (
        ("tracking_variant_masks_output", variants),
        ("tracking_variant_masks_stream", stream_variants),
        ("tracking_variant_scores", score_variants),
    ):
        if not isinstance(values, Mapping) or raw_cache_variant not in values:
            raise ValueError(f"Cache lacks {name}[{raw_cache_variant!r}].")
    return (
        torch.as_tensor(variants[raw_cache_variant]).bool().cpu(),
        torch.as_tensor(stream_variants[raw_cache_variant]).bool().cpu(),
        torch.as_tensor(score_variants[raw_cache_variant]).float().cpu(),
    )


def _load_dino_cache(
    path: Path,
    payload: Mapping[str, Any],
    *,
    raw_cache_variant: str,
) -> dict[str, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing DINOv3 cache for {payload['clip_name']}: {path}. "
            "Run commands_cache_dinov3_object_features.txt first."
        )
    cache_payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(cache_payload, Mapping):
        raise ValueError(f"DINO cache is not a mapping: {path}")
    _, _, scores = _load_raw_masks(payload, raw_cache_variant)
    if str(cache_payload.get("clip", payload["clip_name"])) != str(
        payload["clip_name"]
    ):
        raise ValueError(f"DINO cache clip differs from frozen cache: {path}")
    cached_variant = str(cache_payload.get("raw_cache_variant", ""))
    if cached_variant and cached_variant != raw_cache_variant:
        raise ValueError(
            f"DINO cache raw variant={cached_variant!r} differs from "
            f"runner variant={raw_cache_variant!r}: {path}"
        )
    sequence, tracks = (int(value) for value in scores.shape)
    output: dict[str, torch.Tensor] = {}
    for feature_name, valid_name in (
        ("single_features", "single_valid"),
        ("persistent_features", "persistent_valid"),
        ("shuffled_persistent_features", "shuffled_persistent_valid"),
    ):
        if feature_name not in cache_payload or valid_name not in cache_payload:
            raise ValueError(f"DINO cache {path} lacks {feature_name}/{valid_name}.")
        features = torch.as_tensor(cache_payload[feature_name]).float().cpu()
        valid = torch.as_tensor(cache_payload[valid_name]).bool().cpu()
        if features.ndim != 3 or tuple(features.shape[:2]) != (sequence, tracks):
            raise ValueError(f"DINO cache {path} has invalid {feature_name} shape {tuple(features.shape)}.")
        if tuple(valid.shape) != (sequence, tracks):
            raise ValueError(f"DINO cache {path} has invalid {valid_name} shape {tuple(valid.shape)}.")
        output[feature_name] = features
        output[valid_name] = valid
    track_ids = tuple(int(value) for value in payload["sam_track_ids"])
    cached_ids = tuple(
        int(item)
        for item in torch.as_tensor(cache_payload.get("track_ids", ()))
        .reshape(-1)
        .tolist()
    )
    if cached_ids and cached_ids != track_ids:
        raise ValueError(f"DINO cache track IDs differ from frozen SAM cache: {path}")
    return output


def _write_artifact(
    path: Path,
    *,
    payload: Mapping[str, Any],
    branch: FrozenBranch,
    dino_cache_path: Path,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": 1,
            "revision": REVISION,
            "branch": branch.key,
            "clip": str(payload["clip_name"]),
            "scene_id": str(payload["scene_id"]),
            "frame_indices": tuple(int(value) for value in payload["frame_indices"]),
            "identity_mode": "dinov3_projection_temporal_voxel_memory",
            "pointmap_source": "frozen_streamvggt_world_pointmap",
            "association_source": (
                "projection_iou_plus_fixed_dino_appearance_on_reentry"
                if branch.key not in {"raw_all_visible", "world_memory_only"}
                else "projection_iou_temporal_memory"
            ),
            "dino_cache": str(dino_cache_path),
            "persistent_object_ids_by_sam_slot": branch.persistent_ids_by_sam_slot,
            "map_write_mask_by_sam_slot": branch.map_write_by_sam_slot,
            "map_write_mask": branch.map_write,
            "tracking_masks_output": branch.output_masks,
            "tracking_scores_output": branch.output_scores,
            "tracking_track_ids_output": branch.output_track_ids,
            "tracking_prompts_output": branch.output_prompts,
            "tracking_masks_stream": branch.stream_masks,
            "tracking_scores_stream": branch.stream_scores,
            "tracking_track_ids_stream": branch.stream_track_ids,
            "tracking_prompts_stream": branch.stream_prompts,
            "object_metadata": branch.object_metadata,
            "object_memory": (
                branch.memory.to_dict() if branch.memory is not None else None
            ),
            "association_summary": branch.association_summary,
            "association_events": _events_from_branch(branch),
            **{
                f"fused_{name}": value
                for name, value in branch.fused_map.items()
            },
        },
        path,
    )
    return path


def _events_from_branch(branch: FrozenBranch) -> list[dict[str, object]]:
    if branch.memory is None:
        return []
    return [dict(row) for row in branch.memory.events]


def _load_branch_artifact(path: Path | None) -> FrozenBranch:
    if path is None:
        raise ValueError("Cannot reload a branch without an artifact path.")
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, Mapping):
        raise ValueError(f"Branch artifact is not a mapping: {path}")
    required = (
        "branch",
        "tracking_masks_output",
        "tracking_scores_output",
        "tracking_masks_stream",
        "tracking_scores_stream",
        "tracking_track_ids_output",
        "tracking_prompts_output",
        "tracking_track_ids_stream",
        "tracking_prompts_stream",
        "persistent_object_ids_by_sam_slot",
        "map_write_mask_by_sam_slot",
        "map_write_mask",
        "object_metadata",
        "association_summary",
    )
    missing = [name for name in required if name not in value]
    if missing:
        raise ValueError(f"Branch artifact {path} lacks fields {missing}.")
    fused_map = {
        name: torch.as_tensor(value[f"fused_{name}"]).cpu()
        for name in (
            "world_points",
            "rgb",
            "confidence",
            "object_ids",
            "observation_counts",
        )
        if f"fused_{name}" in value
    }
    if set(fused_map) != {
        "world_points",
        "rgb",
        "confidence",
        "object_ids",
        "observation_counts",
    }:
        raise ValueError(f"Branch artifact {path} has incomplete fused map fields.")
    return FrozenBranch(
        key=str(value["branch"]),
        output_masks=torch.as_tensor(value["tracking_masks_output"]).bool().cpu(),
        output_scores=torch.as_tensor(value["tracking_scores_output"]).float().cpu(),
        output_track_ids=tuple(int(item) for item in value["tracking_track_ids_output"]),
        output_prompts=tuple(str(item) for item in value["tracking_prompts_output"]),
        stream_masks=torch.as_tensor(value["tracking_masks_stream"]).bool().cpu(),
        stream_scores=torch.as_tensor(value["tracking_scores_stream"]).float().cpu(),
        stream_track_ids=tuple(int(item) for item in value["tracking_track_ids_stream"]),
        stream_prompts=tuple(str(item) for item in value["tracking_prompts_stream"]),
        persistent_ids_by_sam_slot=torch.as_tensor(
            value["persistent_object_ids_by_sam_slot"]
        ).long().cpu(),
        map_write_by_sam_slot=torch.as_tensor(
            value["map_write_mask_by_sam_slot"]
        ).bool().cpu(),
        map_write=torch.as_tensor(value["map_write_mask"]).bool().cpu(),
        fused_map=fused_map,
        object_metadata=[dict(item) for item in value["object_metadata"]],
        association_summary=dict(value["association_summary"]),
        artifact_path=path,
        memory=None,
    )


def _collapse_write_mask(
    write_by_slot: torch.Tensor,
    persistent_ids: torch.Tensor,
    object_ids: Sequence[int],
) -> torch.Tensor:
    writes = write_by_slot.detach().bool().cpu()
    ids = persistent_ids.detach().long().cpu()
    if tuple(writes.shape) != tuple(ids.shape):
        raise ValueError("Write mask and persistent IDs have different shapes.")
    output = torch.zeros(writes.shape[0], len(object_ids), dtype=torch.bool)
    for index, object_id in enumerate(object_ids):
        output[:, index] = (writes & (ids == int(object_id))).any(dim=1)
    return output


def _empty_fused_map() -> dict[str, torch.Tensor]:
    return {
        "world_points": torch.empty(0, 3),
        "rgb": torch.empty(0, 3),
        "confidence": torch.empty(0),
        "object_ids": torch.empty(0, dtype=torch.long),
        "observation_counts": torch.empty(0, dtype=torch.long),
    }


def _aggregate(
    *,
    tracking_rows: Sequence[Mapping[str, object]],
    map_rows: Sequence[Mapping[str, object]],
    association_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    tracking_metrics = (
        "mean_frame_iou",
        "frame_idf1",
        "pixel_idf1",
        "reentry_success_rate",
        "fragmentation_count",
        "merge_error_count",
    )
    map_metrics = (
        "object_accuracy_m",
        "object_completeness_m",
        "symmetric_distance_m",
        "fscore_5cm",
        "fscore_10cm",
        "ghost_point_ratio",
        "voxel_iou_5cm",
    )
    tracking_aggregate = _group_aggregate(
        tracking_rows, group_keys=("branch",), metrics=tracking_metrics
    )
    map_aggregate = _group_aggregate(
        map_rows, group_keys=("branch", "map_type"), metrics=map_metrics
    )
    association_aggregate = _group_aggregate(
        association_rows,
        group_keys=("branch",),
        metrics=(
            "candidate_event_count",
            "candidate_count_total",
            "appearance_compared_count",
            "forced_reassociation_count",
            "confirmed_observation_count",
            "persistent_object_count",
        ),
    )
    return {
        "tracking": tracking_aggregate,
        "map": map_aggregate,
        "association": association_aggregate,
        "comparisons": _comparisons(map_aggregate),
    }


def _group_aggregate(
    rows: Sequence[Mapping[str, object]],
    *,
    group_keys: Sequence[str],
    metrics: Sequence[str],
) -> list[dict[str, object]]:
    groups: dict[tuple[str, ...], list[Mapping[str, object]]] = {}
    for row in rows:
        key = tuple(str(row.get(name, "")) for name in group_keys)
        groups.setdefault(key, []).append(row)
    output: list[dict[str, object]] = []
    for key, members in sorted(groups.items()):
        row = {name: value for name, value in zip(group_keys, key)}
        row["count"] = len(members)
        for metric in metrics:
            values = [
                float(member[metric])
                for member in members
                if _finite(member.get(metric))
            ]
            row[f"{metric}_mean"] = sum(values) / len(values) if values else float("nan")
            row[f"{metric}_count"] = len(values)
        output.append(row)
    return output


def _comparisons(map_aggregate: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    lookup = {
        (str(row.get("branch")), str(row.get("map_type"))): row
        for row in map_aggregate
    }
    persistent = lookup.get(("persistent_dino", "fused"), {})
    rows: list[dict[str, object]] = []
    for control in ("world_memory_only", "single_view_dino", "shuffled_persistent_dino"):
        candidate = lookup.get((control, "fused"), {})
        row: dict[str, object] = {
            "candidate": "persistent_dino",
            "control": control,
            "map_type": "fused",
        }
        for metric in ("fscore_5cm", "voxel_iou_5cm", "ghost_point_ratio"):
            current = persistent.get(f"{metric}_mean")
            baseline = candidate.get(f"{metric}_mean")
            row[f"delta_{metric}"] = _delta(current, baseline)
        row["nonregression"] = int(
            _ge(persistent.get("fscore_5cm_mean"), candidate.get("fscore_5cm_mean"))
            and _ge(
                persistent.get("voxel_iou_5cm_mean"),
                candidate.get("voxel_iou_5cm_mean"),
            )
            and _le(
                persistent.get("ghost_point_ratio_mean"),
                candidate.get("ghost_point_ratio_mean"),
            )
        )
        rows.append(row)
    return rows


def _decision(
    aggregate: Mapping[str, object],
    association_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    dino_rows = [
        row
        for row in association_rows
        if str(row.get("branch")) in {
            "single_view_dino",
            "persistent_dino",
            "shuffled_persistent_dino",
        }
    ]
    candidate_count = sum(int(row.get("candidate_event_count", 0)) for row in dino_rows)
    appearance_count = sum(
        int(row.get("appearance_compared_count", 0)) for row in dino_rows
    )
    comparisons = list(aggregate.get("comparisons", ()))
    exercised = candidate_count > 0 and appearance_count > 0
    if not exercised:
        status = "NOT_EXERCISED"
    else:
        status = "GO" if comparisons and all(
            int(row.get("nonregression", 0)) for row in comparisons
        ) else "NO_GO"
    return {
        "status": status,
        "candidate_generation_exercised": int(candidate_count > 0),
        "appearance_exercised": int(appearance_count > 0),
        "candidate_event_count": candidate_count,
        "appearance_compared_count": appearance_count,
        "comparisons": comparisons,
        "definition": (
            "GO requires persistent_dino fused F5cm and voxelIoU to be no worse "
            "than world_memory_only, single_view_dino, and shuffled_persistent_dino, "
            "with no higher fused ghost rate. NOT_EXERCISED means no DINO candidate "
            "comparison occurred and is not evidence that DINO is ineffective."
        ),
    }


def _print_clip_result(result: Mapping[str, object]) -> None:
    clip = result["clip"]
    print(f"  evaluated clip={clip['clip_name']} scene={clip['scene_id']}")
    for row in result["association_rows"]:
        print(
            f"    association branch={row['branch']} "
            f"candidates={row['candidate_event_count']} "
            f"appearance_compared={row['appearance_compared_count']} "
            f"forced={row['forced_reassociation_count']} "
            f"objects={row['persistent_object_count']}"
        )
    for row in result["map_rows"]:
        print(
            f"    map branch={row['branch']} type={row['map_type']} "
            f"voxelIoU5cm={_format(row.get('voxel_iou_5cm'))} "
            f"F5cm={_format(row.get('fscore_5cm'))} "
            f"ghost={_format(row.get('ghost_point_ratio'))}"
        )


def _clip_variant_rows(clips: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for clip in clips:
        for branch, value in (clip.get("variants", {}) or {}).items():
            row: dict[str, object] = {
                "scene_id": clip.get("scene_id", ""),
                "clip": clip.get("clip_name", ""),
                "split": clip.get("split", ""),
                "branch": branch,
                "artifact": value.get("artifact", ""),
            }
            association = value.get("association", {}) or {}
            tracking = value.get("tracking", {}) or {}
            row.update(
                {
                    f"association_{key}": item
                    for key, item in association.items()
                    if key not in {"branch", "scene_id", "clip", "split", "artifact"}
                }
            )
            row.update(
                {
                    f"tracking_{key}": item
                    for key, item in tracking.items()
                    if key not in {"branch"}
                }
            )
            for map_type, map_row in (value.get("maps", {}) or {}).items():
                for key, item in map_row.items():
                    if key not in {"branch", "map_type"}:
                        row[f"map_{map_type}_{key}"] = item
            rows.append(row)
    return rows


def _branch_names() -> tuple[str, ...]:
    return (
        "raw_all_visible",
        "world_memory_only",
        "single_view_dino",
        "persistent_dino",
        "shuffled_persistent_dino",
    )


def _select_clips(
    clips: Sequence[ClipConfig], requested: str | None
) -> tuple[ClipConfig, ...]:
    if requested is None:
        return tuple(clips)
    selected = tuple(clip for clip in clips if clip.name == requested)
    if len(selected) != 1:
        raise ValueError(f"Clip {requested!r} was not found exactly once.")
    return selected


def _mean_metric(rows: Sequence[Mapping[str, object]], key: str) -> float:
    values = [float(row[key]) for row in rows if _finite(row.get(key))]
    return sum(values) / len(values) if values else float("nan")


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _delta(current: object, baseline: object) -> float:
    return float(current) - float(baseline) if _finite(current) and _finite(baseline) else float("nan")


def _ge(current: object, baseline: object) -> bool:
    return _finite(current) and _finite(baseline) and float(current) >= float(baseline)


def _le(current: object, baseline: object) -> bool:
    return _finite(current) and _finite(baseline) and float(current) <= float(baseline)


def _format(value: object) -> str:
    return f"{float(value):.5f}" if _finite(value) else "nan"


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True) + "\n",
        encoding="utf8",
    )
    return path


def _json_safe(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if str(key) not in fields:
                fields.append(str(key))
    if not fields:
        fields = ["empty"]
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fields})
    return path


def _csv_value(value: object) -> object:
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(_json_safe(value), sort_keys=True)
    if isinstance(value, torch.Tensor):
        return json.dumps(_json_safe(value), sort_keys=True)
    return value


def _write_copyable(path: Path, summary: Mapping[str, object]) -> None:
    aggregate = summary["aggregate"]
    decision = summary["decision"]
    lines = [
        "===== DINOV3_PROJECTION_MEMORY_ABLATION_BEGIN =====",
        f"revision={summary['revision']}",
        f"clips={summary['clip_count']}",
        "candidate_generation_gt_fields=0",
        "evaluation_gt_fields=1",
        "pointmap_modified=0",
        "dino_used_for_xyz_regression=0",
        f"appearance_weight={APPEARANCE_WEIGHT}",
        f"reassociation_gap={REASSOCIATION_GAP}",
        f"decision={decision['status']}",
        f"candidate_generation_exercised={decision['candidate_generation_exercised']}",
        f"appearance_exercised={decision['appearance_exercised']}",
        f"candidate_event_count={decision['candidate_event_count']}",
        f"appearance_compared_count={decision['appearance_compared_count']}",
        f"comparisons={decision['comparisons']}",
        f"aggregate_association={aggregate['association']}",
        f"aggregate_map={aggregate['map']}",
        f"summary={summary['outputs'].get('summary', '')}",
        "===== DINOV3_PROJECTION_MEMORY_ABLATION_END =====",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="streaming_couping/configs/v0_baseline.yaml",
    )
    parser.add_argument("--feature-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--clip")
    parser.add_argument("--raw-cache-variant", default=RAW_CACHE_VARIANT)
    return parser.parse_args()


if __name__ == "__main__":
    main()
