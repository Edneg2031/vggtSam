#!/usr/bin/env python3
"""Run the V2.1 failure-only geometry-guided SAM recovery map.

The V0 cache and sealed QK pose artifact are reused.  SAM3.1 is loaded only
for empty/missing/re-entry recovery candidates; normal SAM masks are copied
unchanged.  No GT is opened by this exporter.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

from streaming_couping.src.aggregation.mine_revisit_segments import (
    mine_revisit_candidate,
)
from streaming_couping.src.backbones.sam3_wrapper import SAM3Wrapper
from streaming_couping.src.config import load_config
from streaming_couping.src.learned_pose.baseline_runtime import (
    load_baseline_run_config,
)
from streaming_couping.src.learned_pose.cache import cache_path, load_feature_cache
from streaming_couping.src.learned_pose.config import (
    ClipConfig,
    load_learned_pose_config,
)
from streaming_couping.src.semantic_map import (
    BACKGROUND_SEMANTIC_RGB8,
    INSTANCE_PALETTE_RGB8,
    SemanticPointMap,
    build_semantic_pointmap,
    normalize_confidence,
)
from streaming_couping.src.storage import expand_storage_path
from streaming_couping.src.types import RevisitCandidate
from streaming_couping.src.v2_geometry_recovery import V2ObjectMemoryState
from streaming_couping.src.v21_geometry_recovery import (
    V21RecoveryConfig,
    process_v21_sequence,
)

from streaming_couping.scripts.run_v0_semantic_map import _validate_inputs
from streaming_couping.scripts.run_v2_semantic_map import (
    DEFAULT_RAW_VARIANT,
    _find_clip,
    _load_raw_variants,
    _pose_sequence,
    _track_metadata,
    _write_binary_ply,
    _write_csv,
)


REVISION = "v2_1_failure_only_candidate_recovery_r1"
BASELINE_REVISION = "v0_frozen_semantic_mapping_pipeline_r1"


@dataclass(frozen=True)
class V21Run:
    source_path: Path
    output_dir: Path
    confidence_threshold: float
    track_score_threshold: float
    max_map_points: int
    raw_cache_variant: str
    recovery: V21RecoveryConfig


def main() -> None:
    args = _parse_args()
    data = load_learned_pose_config(args.config)
    baseline = load_baseline_run_config(args.config)
    run = _load_run(args.config)
    if args.output_dir:
        run = replace(run, output_dir=Path(args.output_dir).expanduser().resolve())
    recovery = (
        load_config(data.recovery_config, {"sam3_device": str(args.sam3_device)})
        if args.sam3_device
        else load_config(data.recovery_config)
    )

    clip = _find_clip(data.clips, baseline.clip_name)
    cache_value = cache_path(data, clip)
    payload = load_feature_cache(cache_value)
    baseline_summary, poses = _validate_inputs(
        payload=payload,
        summary_path=baseline.output_dir / "baseline_summary.json",
        poses_path=baseline.output_dir / "poses.pt",
        clip=clip,
    )
    raw_output, raw_stream, raw_scores = _load_raw_variants(payload, run)
    output_size = tuple(int(value) for value in raw_output.shape[-2:])
    processed_size = tuple(int(value) for value in payload["image_size"])
    source_sizes = tuple(
        tuple(int(value) for value in row) for row in payload["source_sizes"]
    )
    selected_pose = _pose_sequence(poses["selected_world_to_camera"])
    intrinsics = payload["baseline_intrinsics"].detach().float().cpu()
    confidence = normalize_confidence(payload["baseline_world_confidence"])

    if recovery.sam3_version != "sam3.1":
        raise ValueError("V2.1 failure-only recovery requires sam3.version=sam3.1.")
    print("V2.1 FAILURE-ONLY CANDIDATE GEOMETRY RECOVERY")
    print(f"cache={cache_value}")
    print(f"raw_cache_variant={run.raw_cache_variant}")
    print(
        "policy=only empty/missing/re-entry triggers; raw SAM is preserved "
        "unless recovery score beats raw by a margin"
    )
    print(
        f"frames={raw_output.shape[0]} slots={raw_output.shape[1]} "
        f"output_size={output_size} stream_size={processed_size} "
        f"sam_device={recovery.sam3_device}"
    )

    sam3 = SAM3Wrapper(
        repo_path=recovery.sam3_repo,
        checkpoint_path=recovery.sam3_checkpoint,
        device=recovery.sam3_device,
        output_threshold=recovery.sam3_output_threshold,
        prompt_with_box=recovery.prompt_with_box,
        version=recovery.sam3_version,
        use_fa3=recovery.sam3_use_fa3,
        max_num_objects=recovery.sam3_max_num_objects,
        multiplex_count=recovery.sam3_multiplex_count,
    ).load()

    world_points = payload["baseline_world_points"].detach().float().cpu()
    frames = tuple(int(value) for value in payload["frame_indices"])
    track_ids = tuple(int(value) for value in payload["sam_track_ids"])
    track_prompts = tuple(str(value) for value in payload["sam_track_prompts"])

    def proposal_builder(
        frame: int,
        slot: int,
        memory: V2ObjectMemoryState,
    ) -> RevisitCandidate:
        return mine_revisit_candidate(
            memory.world_points,
            current_world_points=world_points[frame],
            world_to_camera=selected_pose[frame],
            intrinsics=intrinsics[frame],
            source_size=source_sizes[frame],
            processed_size=processed_size,
            output_size=output_size,
            image_mode=recovery.image_mode,
            box_quantile=run.recovery.proposal_box_quantile,
            box_padding_ratio=run.recovery.proposal_box_padding_ratio,
            min_projected_points=run.recovery.min_projected_points,
            min_projected_fraction=run.recovery.min_projected_fraction,
            min_supported_points=run.recovery.min_supported_points,
            min_support_ratio=run.recovery.min_support_ratio,
            support_abs_distance=run.recovery.support_abs_distance,
            support_relative_distance=run.recovery.support_relative_distance,
        )

    def refine_callback(
        frame: int,
        slot: int,
        prompt: str,
        proposal: RevisitCandidate,
    ):
        positive = proposal.supported_mask.detach().cpu().bool()
        if not bool(positive.any()):
            positive = proposal.projected_mask.detach().cpu().bool()
        if not bool(positive.any()):
            return []
        return sam3.propose_geometry_prompted_masks(
            payload["image_paths"][frame],
            prompt=prompt or "object",
            output_size=output_size,
            geometry_prompt=proposal.mask.detach().cpu().bool(),
            positive_prompt=positive,
            max_positive_points=run.recovery.max_positive_points,
            use_box=True,
            use_points=True,
        )

    result = process_v21_sequence(
        world_points=world_points,
        confidence=confidence,
        raw_masks_output=raw_output,
        raw_scores=raw_scores,
        raw_masks_stream=raw_stream,
        frame_indices=frames,
        track_ids=track_ids,
        track_prompts=track_prompts,
        world_to_camera=selected_pose,
        intrinsics=intrinsics,
        source_sizes=source_sizes,
        processed_size=processed_size,
        image_mode=recovery.image_mode,
        config=run.recovery,
        proposal_builder=proposal_builder,
        refine_callback=refine_callback,
    )
    semantic = build_semantic_pointmap(
        world_points=world_points,
        confidence=payload["baseline_world_confidence"],
        masks=result["masks_stream"],
        track_scores=result["scores"],
        images=payload["stream_images"],
        confidence_threshold=run.confidence_threshold,
        track_score_threshold=run.track_score_threshold,
        max_map_points=run.max_map_points,
    )
    output_path = _write_outputs(
        payload=payload,
        baseline_summary=baseline_summary,
        poses=poses,
        cache_path_value=cache_value,
        run=run,
        result=result,
        semantic=semantic,
        raw_output=raw_output,
        recovery_config_path=data.recovery_config,
    )
    stats = result["stats"]
    print(
        "  recovery_trigger_count="
        f"{stats['recovery_trigger_count']} "
        f"accepted_recovery_count={stats['accepted_recovery_count']} "
        f"recovery_reject_count={stats['recovery_reject_count']}"
    )
    print(
        f"  trigger_reasons={stats['trigger_reason_counts']} "
        f"raw_fallback_unchanged={stats['raw_fallback_unchanged_ratio']:.4f} "
        f"normal_unchanged={stats['normal_frame_unchanged_ratio']:.4f}"
    )
    print(
        f"  points={semantic.world_points.shape[0]} "
        f"semantic={int((semantic.semantic_slots >= 0).sum())}"
    )
    print(f"  ply={run.output_dir / 'semantic_map.ply'}")
    print(f"V2.1 semantic map result={output_path}")


def _write_outputs(
    *,
    payload: Mapping[str, Any],
    baseline_summary: Mapping[str, Any],
    poses: Mapping[str, Any],
    cache_path_value: Path,
    run: V21Run,
    result: Mapping[str, Any],
    semantic: SemanticPointMap,
    raw_output: torch.Tensor,
    recovery_config_path: Path,
    revision: str = REVISION,
    identity_mode: str = "v2_1_failure_only_candidate_recovery",
    semantic_source: str = "sam3_raw_tracking_with_failure_only_geometry_recovery",
    claim: str = "frozen_v2_1_failure_only_candidate_recovery",
    identity_policy: str = "one_v21_memory_entry_per_existing_v0_sam_slot",
    recovery_description: str = "only_empty_missing_or_reentry_plus_candidate_score_margin",
    copyable_writer=None,
) -> Path:
    frames = tuple(int(value) for value in payload["frame_indices"])
    track_ids = torch.as_tensor(payload["sam_track_ids"], dtype=torch.long)
    safe_slots = semantic.semantic_slots.clamp_min(0)
    semantic_track_ids = torch.where(
        semantic.semantic_slots >= 0,
        track_ids.index_select(0, safe_slots),
        torch.full_like(semantic.semantic_slots, -1),
    )
    semantic_points = int((semantic.semantic_slots >= 0).sum())
    tracks = _track_metadata(payload, semantic, result["masks_stream"])
    object_rows: list[dict[str, Any]] = []
    for row in result["objects"]:
        current = dict(row)
        slot = int(current["object_id"])
        tensor_row = result["object_tensors"][slot]
        current["world_point_count"] = int(tensor_row["world_points"].shape[0])
        current["recovery_trigger_count"] = int(
            result["stats"]["per_object_trigger_count"][str(slot)]
        )
        current["recovery_count"] = int(
            result["stats"]["per_object_recovery_count"][str(slot)]
        )
        current["reentry_count"] = int(
            result["stats"]["per_object_reentry_count"][str(slot)]
        )
        current["reentry_recovery_count"] = int(
            result["stats"]["per_object_reentry_recovery_count"][str(slot)]
        )
        current["color_red"], current["color_green"], current["color_blue"] = (
            INSTANCE_PALETTE_RGB8[slot % len(INSTANCE_PALETTE_RGB8)]
        )
        object_rows.append(current)

    run.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = run.output_dir / "semantic_map.pt"
    torch.save(
        {
            "schema": 1,
            "revision": revision,
            "baseline_revision": BASELINE_REVISION,
            "identity_mode": identity_mode,
            "raw_cache_variant": run.raw_cache_variant,
            "clip": payload["clip_name"],
            "frame_indices": frames,
            "reference_sequence_index": int(payload["reference_sequence_index"]),
            "coordinate_frame": "streamvggt_first_frame_reference_world",
            "world_points": semantic.world_points,
            "rgb": semantic.rgb,
            "semantic_rgb": semantic.semantic_rgb,
            "semantic_slots": semantic.semantic_slots,
            "semantic_track_ids": semantic_track_ids,
            "confidence": semantic.confidence,
            "sequence_indices": semantic.sequence_indices,
            "flat_indices": semantic.flat_indices,
            "tracking_masks_output": result["masks_output"],
            "tracking_masks_stream": result["masks_stream"],
            "tracking_scores": result["scores"],
            "history_masks_output": result["history_masks_output"],
            # V2.3 keeps tracking masks intact and may provide a separate,
            # confidence-filtered map-write branch.  Older V2.1/V2.2 result
            # dictionaries do not contain these optional fields, so their
            # serialized artifacts remain backward compatible.
            "map_masks_stream": result.get(
                "map_masks_stream", result["masks_stream"]
            ),
            "map_write_mask": result.get(
                "map_write_mask",
                torch.ones(
                    result["masks_stream"].shape[:2], dtype=torch.bool
                ),
            ),
            "map_observation_confidence": result.get(
                "map_observation_confidence",
                torch.ones(result["masks_stream"].shape[:2], dtype=torch.float32),
            ),
            "object_memory_confidence": result.get(
                "object_memory_confidence", {}
            ),
            "object_metadata": object_rows,
            "object_memory": result["objects"],
            "object_memory_world_points": {
                int(slot): values["world_points"]
                for slot, values in result["object_tensors"].items()
            },
            "object_memory_world_weights": {
                int(slot): values["world_weights"]
                for slot, values in result["object_tensors"].items()
            },
            "recovery_events": result["events"],
            "recovery_stats": result["stats"],
            "slot_palette_rgb8": INSTANCE_PALETTE_RGB8,
            "unlabeled_background_rgb8": BACKGROUND_SEMANTIC_RGB8,
            "raw_world_to_camera": poses["raw_world_to_camera"],
            "selected_world_to_camera": poses["selected_world_to_camera"],
            "selected_pose_branch": poses["selected_pose_branch"],
            "baseline_world_to_camera": payload["baseline_world_to_camera"],
            "baseline_intrinsics": payload["baseline_intrinsics"],
            "pointmap_source": "raw_full_history_streamvggt_world_pointmap",
            "semantic_source": semantic_source,
            "recovery_config": str(recovery_config_path),
        },
        artifact_path,
    )

    semantic_ply = run.output_dir / "semantic_map.ply"
    _write_binary_ply(
        semantic_ply,
        points=semantic.world_points,
        rgb=semantic.semantic_rgb,
        slots=semantic.semantic_slots,
        track_ids=semantic_track_ids,
        confidence=semantic.confidence,
    )
    rgb_ply = run.output_dir / "rgb_map.ply"
    _write_binary_ply(
        rgb_ply,
        points=semantic.world_points,
        rgb=semantic.rgb,
        slots=semantic.semantic_slots,
        track_ids=semantic_track_ids,
        confidence=semantic.confidence,
    )
    event_path = _write_csv(run.output_dir / "recovery_events.csv", result["events"])
    memory_json = run.output_dir / "object_memory.json"
    memory_json.write_text(
        json.dumps(
            {
                "schema": 1,
                "revision": revision,
                "identity_policy": identity_policy,
                "objects": object_rows,
                "tensor_artifact": str(artifact_path),
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf8",
    )
    memory_csv = _write_csv(
        run.output_dir / "object_memory.csv",
        [
            {
                key: row.get(key, "")
                for key in (
                    "object_id",
                    "prompt",
                    "sam_track_id",
                    "observation_count",
                    "last_seen_sequence_index",
                    "last_seen_frame_index",
                    "world_point_count",
                    "confidence_ema",
                    "recovery_trigger_count",
                    "recovery_count",
                    "reentry_count",
                    "reentry_recovery_count",
                )
            }
            for row in object_rows
        ],
    )
    summary: dict[str, Any] = {
        "schema": 1,
        "revision": revision,
        "baseline_version": "v0",
        "baseline_status": "frozen",
        "baseline_revision": BASELINE_REVISION,
        "identity_mode": identity_mode,
        "raw_cache_variant": run.raw_cache_variant,
        "clip": payload["clip_name"],
        "config": str(run.source_path),
        "cache": str(cache_path_value),
        "frames": frames,
        "reference_frame": frames[int(payload["reference_sequence_index"])],
        "map_generation_gt_fields": 0,
        "model_trained": 0,
        "streamvggt_frozen": 1,
        "sam3_frozen": 1,
        "sam_pose_inputs": 0,
        "pose_source": "streamvggt_native_qk_history_retrieval",
        "pointmap_source": "raw_full_history_streamvggt_world_pointmap",
        "semantic_source": semantic_source,
        "recovery_description": recovery_description,
        "identity_policy": identity_policy,
        "prompt_selection_annotation_gt_used": int(
            baseline_summary["prompt_selection_annotation_gt_used"]
        ),
        "runtime_prompt_selection_gt_fields": int(
            baseline_summary["runtime_prompt_selection_gt_fields"]
        ),
        "selected_pose_branch": poses["selected_pose_branch"],
        "pose_improvement_claim": int(baseline_summary["pose_improvement_claim"]),
        "pointmap_modified": 0,
        "source_sam_track_count": int(raw_output.shape[1]),
        "persistent_object_count": int(len(object_rows)),
        "saved_map_points": int(semantic.world_points.shape[0]),
        "saved_semantic_points": semantic_points,
        "semantic_coverage_percent": 100.0
        * semantic_points
        / max(1, int(semantic.world_points.shape[0])),
        "recovery_stats": result["stats"],
        "object_count_reduction_goal": 0,
        "normal_frame_identity_policy": "raw_sam_mask_exact_copy_when_not_triggered",
        "failure_only_trigger_policy": [
            "sam_track_lost",
            "unobserved_gap",
            "reentry_after_gap",
        ],
        "tracks": tracks,
        "objects": object_rows,
        "semantic_map_pipeline_ready": int(
            semantic.world_points.shape[0] > 0
            and semantic_points > 0
            and object_rows
            and poses["selected_pose_branch"] == "retrieve_qk"
        ),
        "claim": claim,
        "outputs": {
            "artifact": str(artifact_path),
            "semantic_ply": str(semantic_ply),
            "rgb_ply": str(rgb_ply),
            "recovery_events_csv": str(event_path),
            "object_memory_json": str(memory_json),
            "object_memory_csv": str(memory_csv),
        },
    }
    summary_path = run.output_dir / "semantic_map_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf8",
    )
    if copyable_writer is None:
        _write_copyable(run.output_dir / "copyable_result.txt", summary)
    else:
        copyable_writer(run.output_dir / "copyable_result.txt", summary)
    return summary_path


def _write_copyable(path: Path, summary: Mapping[str, Any]) -> None:
    stats = summary["recovery_stats"]
    outputs = summary["outputs"]
    lines = [
        "===== COPYABLE_V2_1_FAILURE_ONLY_RECOVERY_BEGIN =====",
        f"revision={summary['revision']}",
        f"clip={summary['clip']}",
        f"frames={len(summary['frames'])}",
        "identity_mode=v2_1_failure_only_candidate_recovery",
        "pose=retrieve_qk",
        "pointmap=raw_full_history_streamvggt_world_pointmap",
        "normal_frames=raw_sam_mask_exact_copy",
        "recovery=only_empty_missing_or_reentry_plus_candidate_score_margin",
        "map_generation_gt_fields=0",
        f"source_sam_track_count={summary['source_sam_track_count']}",
        f"persistent_object_count={summary['persistent_object_count']}",
        f"saved_map_points={summary['saved_map_points']}",
        f"saved_semantic_points={summary['saved_semantic_points']}",
        f"semantic_coverage_percent={summary['semantic_coverage_percent']}",
        f"recovery_trigger_count={stats['recovery_trigger_count']}",
        f"accepted_recovery_count={stats['accepted_recovery_count']}",
        f"recovery_reject_count={stats['recovery_reject_count']}",
        f"trigger_reason_counts={stats['trigger_reason_counts']}",
        f"raw_fallback_unchanged_ratio={stats['raw_fallback_unchanged_ratio']}",
        f"normal_frame_unchanged_ratio={stats['normal_frame_unchanged_ratio']}",
        f"semantic_map_pipeline_ready={summary['semantic_map_pipeline_ready']}",
        f"artifact={outputs['artifact']}",
        f"semantic_ply={outputs['semantic_ply']}",
        f"rgb_ply={outputs['rgb_ply']}",
        f"recovery_events_csv={outputs['recovery_events_csv']}",
        f"object_memory_json={outputs['object_memory_json']}",
        f"object_memory_csv={outputs['object_memory_csv']}",
        "===== COPYABLE_V2_1_FAILURE_ONLY_RECOVERY_END =====",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def _load_run(path: str | Path) -> V21Run:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf8") as handle:
        raw = yaml.safe_load(handle) or {}
    section = raw.get("semantic_map", {}) or {}
    v21 = section.get("v2_1_geometry_recovery", {}) or {}
    recovery = v21.get("recovery", {}) or {}
    return V21Run(
        source_path=source,
        output_dir=expand_storage_path(
            v21.get(
                "output_dir",
                "${VGGT_SAM_STORAGE_ROOT}/outputs/streaming_couping_v21/semantic_map",
            ),
            base=source.parent,
        ),
        confidence_threshold=float(section.get("confidence_threshold", 0.30)),
        track_score_threshold=float(section.get("track_score_threshold", 0.50)),
        max_map_points=int(section.get("max_map_points", 400000)),
        raw_cache_variant=str(v21.get("raw_cache_variant", DEFAULT_RAW_VARIANT)),
        recovery=V21RecoveryConfig(
            unobserved_gap=int(recovery.get("unobserved_gap", 2)),
            reentry_gap=int(recovery.get("reentry_gap", 3)),
            low_score_threshold=float(recovery.get("low_score_threshold", 0.50)),
            min_mask_pixels=int(recovery.get("min_mask_pixels", 32)),
            min_area_ratio=float(recovery.get("min_area_ratio", 0.35)),
            max_area_ratio=float(recovery.get("max_area_ratio", 2.85)),
            min_history_points=int(recovery.get("min_history_points", 16)),
            max_points_per_object=int(recovery.get("max_points_per_object", 4096)),
            min_memory_update_score=float(
                recovery.get("min_memory_update_score", 0.50)
            ),
            min_geometry_confidence=float(
                recovery.get("min_geometry_confidence", 0.30)
            ),
            max_positive_points=int(recovery.get("max_positive_points", 6)),
            min_candidate_score=float(recovery.get("min_candidate_score", 0.50)),
            min_candidate_support_recall=float(
                recovery.get("min_candidate_support_recall", 0.15)
            ),
            min_candidate_iou=float(recovery.get("min_candidate_iou", 0.02)),
            min_candidate_area_ratio=float(
                recovery.get("min_candidate_area_ratio", 0.20)
            ),
            max_candidate_area_ratio=float(
                recovery.get("max_candidate_area_ratio", 5.00)
            ),
            min_geometry_consistency=float(
                recovery.get("min_geometry_consistency", 0.12)
            ),
            min_recovery_score=float(recovery.get("min_recovery_score", 0.50)),
            replacement_margin=float(recovery.get("replacement_margin", 0.12)),
            sam_score_weight=float(recovery.get("sam_score_weight", 0.35)),
            geometry_score_weight=float(
                recovery.get("geometry_score_weight", 0.40)
            ),
            temporal_score_weight=float(
                recovery.get("temporal_score_weight", 0.25)
            ),
            proposal_box_quantile=float(
                recovery.get("proposal_box_quantile", 0.02)
            ),
            proposal_box_padding_ratio=float(
                recovery.get("proposal_box_padding_ratio", 0.12)
            ),
            min_projected_points=int(recovery.get("min_projected_points", 8)),
            min_projected_fraction=float(
                recovery.get("min_projected_fraction", 0.005)
            ),
            min_supported_points=int(recovery.get("min_supported_points", 8)),
            min_support_ratio=float(recovery.get("min_support_ratio", 0.02)),
            support_abs_distance=float(
                recovery.get("support_abs_distance", 0.15)
            ),
            support_relative_distance=float(
                recovery.get("support_relative_distance", 0.10)
            ),
        ),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="streaming_couping/configs/v0_baseline.yaml"
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--sam3-device")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
