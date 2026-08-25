#!/usr/bin/env python3
"""Run V2.3 confidence-aware voxel memory on frozen V2.2 recovery masks."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

from streaming_couping.src.learned_pose.baseline_runtime import (
    load_baseline_run_config,
)
from streaming_couping.src.learned_pose.cache import cache_path, load_feature_cache
from streaming_couping.src.learned_pose.config import load_learned_pose_config
from streaming_couping.src.semantic_map import build_semantic_pointmap, normalize_confidence
from streaming_couping.src.storage import expand_storage_path
from streaming_couping.src.v21_geometry_recovery import (
    V21RecoveryConfig,
)
from streaming_couping.src.v22_geometry_validation import (
    V22GeometryValidationConfig,
)
from streaming_couping.src.v23_confidence_memory import (
    V23ConfidenceMemoryConfig,
    process_v23_memory_sequence,
)

from streaming_couping.scripts.run_v0_semantic_map import _validate_inputs
from streaming_couping.scripts.run_v2_semantic_map import (
    DEFAULT_RAW_VARIANT,
    _find_clip,
    _load_raw_variants,
)
from streaming_couping.scripts.run_v21_semantic_map import (
    _write_outputs,
)
from streaming_couping.scripts.run_v22_semantic_map import (
    _add_validation_stats,
    _annotate_object_geometry,
)
REVISION = "v2_3_failure_only_confidence_aware_voxel_memory_r1"
IDENTITY_MODE = "v2_3_failure_only_confidence_aware_voxel_memory"
SEMANTIC_SOURCE = "sam3_v2_2_recovery_with_confidence_aware_voxel_memory"
CLAIM = "frozen_v2_3_confidence_aware_object_memory"


@dataclass(frozen=True)
class V23Run:
    source_path: Path
    output_dir: Path
    v22_map_dir: Path
    confidence_threshold: float
    track_score_threshold: float
    max_map_points: int
    raw_cache_variant: str
    recovery: V21RecoveryConfig
    validation: V22GeometryValidationConfig
    memory: V23ConfidenceMemoryConfig


def main() -> None:
    args = _parse_args()
    data = load_learned_pose_config(args.config)
    baseline = load_baseline_run_config(args.config)
    run = _load_run(args.config)
    if args.output_dir:
        run = replace(run, output_dir=Path(args.output_dir).expanduser().resolve())
    if args.v22_map_dir:
        run = replace(run, v22_map_dir=Path(args.v22_map_dir).expanduser().resolve())

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
    confidence = normalize_confidence(payload["baseline_world_confidence"])
    world_points = payload["baseline_world_points"].detach().float().cpu()
    frames = tuple(int(value) for value in payload["frame_indices"])
    track_ids = tuple(int(value) for value in payload["sam_track_ids"])
    track_prompts = tuple(str(value) for value in payload["sam_track_prompts"])

    v22_artifact_path = run.v22_map_dir / "semantic_map.pt"
    if not v22_artifact_path.is_file():
        raise FileNotFoundError(
            f"Missing frozen V2.2 artifact {v22_artifact_path}; "
            "run commands_semantic_mapping_v22.txt first."
        )
    v22_artifact = torch.load(
        v22_artifact_path,
        map_location="cpu",
        weights_only=False,
    )
    if v22_artifact.get("identity_mode") != "v2_2_failure_only_geometry_consistency_recovery":
        raise ValueError(
            "V2.3 requires the frozen V2.2 geometry-consistency artifact; "
            f"got {v22_artifact.get('identity_mode')!r}."
        )
    v22_output = _cpu_tensor(v22_artifact["tracking_masks_output"]).bool()
    v22_stream = _cpu_tensor(v22_artifact["tracking_masks_stream"]).bool()
    v22_scores = _cpu_tensor(v22_artifact["tracking_scores"]).float()
    if tuple(v22_output.shape) != tuple(raw_output.shape):
        raise ValueError("V2.2 and raw cache output mask shapes disagree.")
    if tuple(v22_stream.shape) != tuple(raw_stream.shape):
        raise ValueError("V2.2 and raw cache stream mask shapes disagree.")
    print("V2.3 CONFIDENCE-AWARE OBJECT MEMORY")
    print("models are not rerun; frozen V2.2 masks and QK pose artifact are reused")
    print(f"cache={cache_value}")
    print(f"v22_artifact={v22_artifact_path}")
    print(f"raw_cache_variant={run.raw_cache_variant}")
    print(
        "policy=V2.2 failure-only recovery plus confidence-aware voxel fusion; "
        "tracking masks remain unchanged"
    )
    print(
        f"frames={raw_output.shape[0]} slots={raw_output.shape[1]} "
        f"output_size={output_size} stream_size={processed_size} "
        f"memory_voxel_size_m={run.memory.voxel_size_m}"
    )

    # The V2.2 artifact is the sealed input to this ablation.  V2.3 does not
    # call SAM3 or StreamVGGT again, so any tracking difference cannot be
    # attributed to a second model invocation.
    v22_result = {
        "masks_output": v22_output,
        "masks_stream": v22_stream,
        "scores": v22_scores,
        "history_masks_output": _cpu_tensor(
            v22_artifact.get("history_masks_output", v22_output)
        ).bool(),
        "events": [dict(row) for row in v22_artifact.get("recovery_events", ())],
        "stats": dict(v22_artifact.get("recovery_stats", {})),
    }
    _add_validation_stats(v22_result["stats"], v22_result["events"])

    memory_result = process_v23_memory_sequence(
        world_points=world_points,
        confidence=confidence,
        masks_stream=v22_result["masks_stream"],
        scores=v22_result["scores"],
        events=v22_result["events"],
        frame_indices=frames,
        track_ids=track_ids,
        track_prompts=track_prompts,
        config=run.memory,
    )
    result = dict(v22_result)
    result["events"] = memory_result["events"]
    result["objects"] = memory_result["objects"]
    result["object_tensors"] = memory_result["object_tensors"]
    result["map_masks_stream"] = memory_result["map_masks_stream"]
    result["map_write_mask"] = memory_result["map_write_mask"]
    result["map_observation_confidence"] = memory_result[
        "map_observation_confidence"
    ]
    result["object_memory_confidence"] = memory_result[
        "object_memory_confidence"
    ]
    merged_stats = dict(v22_result["stats"])
    merged_stats.update(memory_result["stats"])
    result["stats"] = merged_stats
    _annotate_object_geometry(result)

    semantic = build_semantic_pointmap(
        world_points=world_points,
        confidence=payload["baseline_world_confidence"],
        masks=result["map_masks_stream"],
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
        revision=REVISION,
        identity_mode=IDENTITY_MODE,
        semantic_source=SEMANTIC_SOURCE,
        claim=CLAIM,
        identity_policy="one_confidence_aware_voxel_memory_entry_per_v0_sam_slot",
        recovery_description=(
            "V2.2_failure_only_world_validation_plus_confidence_aware_voxel_fusion"
        ),
        copyable_writer=_write_v23_copyable,
    )

    confidence_path = run.output_dir / "object_memory_confidence.json"
    confidence_payload = dict(memory_result["object_memory_confidence"])
    confidence_payload.update(
        {
            "revision": REVISION,
            "identity_mode": IDENTITY_MODE,
            "source_confidence": {
                "raw_sam": run.memory.raw_observation_confidence,
                "validated_recovery": run.memory.validated_recovery_confidence,
                "recovery_candidate": run.memory.recovery_candidate_confidence,
            },
        }
    )
    confidence_path.write_text(
        json.dumps(confidence_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf8",
    )
    summary = json.loads(Path(output_path).read_text(encoding="utf8"))
    summary["recovery_stats"] = result["stats"]
    summary["memory_policy"] = {
        "voxel_size_m": run.memory.voxel_size_m,
        "raw_observation_confidence": run.memory.raw_observation_confidence,
        "validated_recovery_confidence": run.memory.validated_recovery_confidence,
        "recovery_candidate_confidence": run.memory.recovery_candidate_confidence,
        "recovery_support_distance_m": run.memory.recovery_support_distance_m,
        "min_recovery_support_ratio": run.memory.min_recovery_support_ratio,
    }
    summary["object_memory_confidence"] = confidence_payload
    summary["outputs"]["object_memory_confidence"] = str(confidence_path)
    summary["map_mask_policy"] = (
        "tracking masks are frozen V2.2 outputs; map masks contain raw observations "
        "or supported validated-recovery voxels only"
    )
    Path(output_path).write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf8",
    )
    named_summary = run.output_dir / "semantic_map_v23_summary.json"
    named_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf8",
    )
    _write_v23_copyable(run.output_dir / "copyable_result.txt", summary)

    stats = result["stats"]
    print(
        "  memory_update_count="
        f"{stats['memory_update_count']} "
        f"raw_observation_update={stats['raw_observation_update']} "
        f"validated_recovery_update={stats['validated_recovery_update']}"
    )
    print(
        f"  low_confidence_reject_count={stats['low_confidence_reject_count']} "
        f"voxel_conflict_reject_count={stats['voxel_conflict_reject_count']}"
    )
    print(
        f"  average_object_confidence={stats['average_object_confidence']:.4f} "
        f"recovery_point_ratio={stats['recovery_point_ratio']:.4f} "
        f"low_confidence_point_ratio={stats['low_confidence_point_ratio']:.4f}"
    )
    print(
        f"  V2.2 tracking recovery accepted={stats['accepted_recovery_count']} "
        f"geometry_reject_reasons={stats['geometry_validation_reject_reasons']} "
        f"normal_unchanged={stats['normal_frame_unchanged_ratio']:.4f}"
    )
    print(
        f"  tracking_points={int(result['masks_stream'].sum())} "
        f"map_points={int(result['map_masks_stream'].sum())} "
        f"semantic={int((semantic.semantic_slots >= 0).sum())}"
    )
    print(f"  ply={run.output_dir / 'semantic_map.ply'}")
    print(f"V2.3 semantic map result={output_path}")
    print(f"V2.3 semantic map summary={named_summary}")


def _write_v23_copyable(path: Path, summary: Mapping[str, Any]) -> None:
    stats = summary["recovery_stats"]
    outputs = summary["outputs"]
    lines = [
        "===== COPYABLE_V2_3_CONFIDENCE_AWARE_MEMORY_BEGIN =====",
        f"revision={summary['revision']}",
        f"clip={summary['clip']}",
        f"frames={len(summary['frames'])}",
        f"identity_mode={IDENTITY_MODE}",
        "pose=retrieve_qk",
        "pointmap=raw_full_history_streamvggt_world_pointmap",
        "normal_frames=V2.2_tracking_masks_exact_copy",
        "recovery=V2.2_failure_only_world_validation",
        "memory=confidence_aware_voxel_fusion",
        "map_generation_gt_fields=0",
        f"source_sam_track_count={summary['source_sam_track_count']}",
        f"persistent_object_count={summary['persistent_object_count']}",
        f"saved_map_points={summary['saved_map_points']}",
        f"saved_semantic_points={summary['saved_semantic_points']}",
        f"semantic_coverage_percent={summary['semantic_coverage_percent']}",
        f"recovery_trigger_count={stats['recovery_trigger_count']}",
        f"accepted_recovery_count={stats['accepted_recovery_count']}",
        f"geometry_validation_accept_count={stats['geometry_validation_accept_count']}",
        f"geometry_validation_reject_count={stats['geometry_validation_reject_count']}",
        f"memory_update_count={stats['memory_update_count']}",
        f"raw_observation_update={stats['raw_observation_update']}",
        f"validated_recovery_update={stats['validated_recovery_update']}",
        f"low_confidence_reject_count={stats['low_confidence_reject_count']}",
        f"voxel_conflict_reject_count={stats['voxel_conflict_reject_count']}",
        f"average_object_confidence={stats['average_object_confidence']}",
        f"recovery_point_ratio={stats['recovery_point_ratio']}",
        f"low_confidence_point_ratio={stats['low_confidence_point_ratio']}",
        f"raw_fallback_unchanged_ratio={stats['raw_fallback_unchanged_ratio']}",
        f"normal_frame_unchanged_ratio={stats['normal_frame_unchanged_ratio']}",
        f"semantic_map_pipeline_ready={summary['semantic_map_pipeline_ready']}",
        f"artifact={outputs['artifact']}",
        f"semantic_ply={outputs['semantic_ply']}",
        f"rgb_ply={outputs['rgb_ply']}",
        f"recovery_events_csv={outputs['recovery_events_csv']}",
        f"object_memory_json={outputs['object_memory_json']}",
        f"object_memory_csv={outputs['object_memory_csv']}",
        f"object_memory_confidence={outputs['object_memory_confidence']}",
        "===== COPYABLE_V2_3_CONFIDENCE_AWARE_MEMORY_END =====",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def _load_run(path: str | Path) -> V23Run:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf8") as handle:
        raw = yaml.safe_load(handle) or {}
    section = raw.get("semantic_map", {}) or {}
    v23 = section.get("v2_3_geometry_recovery", {}) or {}
    recovery = v23.get("recovery", {}) or {}
    validation = v23.get("validation", {}) or {}
    memory = v23.get("memory", {}) or {}
    return V23Run(
        source_path=source,
        output_dir=expand_storage_path(
            v23.get(
                "output_dir",
                "${VGGT_SAM_STORAGE_ROOT}/outputs/streaming_couping_v23/semantic_map",
            ),
            base=source.parent,
        ),
        v22_map_dir=expand_storage_path(
            v23.get(
                "v22_map_dir",
                "${VGGT_SAM_STORAGE_ROOT}/outputs/streaming_couping_v22/semantic_map",
            ),
            base=source.parent,
        ),
        confidence_threshold=float(section.get("confidence_threshold", 0.30)),
        track_score_threshold=float(section.get("track_score_threshold", 0.50)),
        max_map_points=int(section.get("max_map_points", 400000)),
        raw_cache_variant=str(v23.get("raw_cache_variant", DEFAULT_RAW_VARIANT)),
        recovery=_load_recovery_config(recovery),
        validation=V22GeometryValidationConfig(
            max_candidate_points=int(validation.get("max_candidate_points", 2048)),
            max_historical_points=int(validation.get("max_historical_points", 2048)),
            min_candidate_points=int(validation.get("min_candidate_points", 8)),
            min_historical_points=int(validation.get("min_historical_points", 16)),
            centroid_distance_threshold_m=float(validation.get("centroid_distance_threshold_m", 0.35)),
            centroid_distance_scale=float(validation.get("centroid_distance_scale", 0.75)),
            point_overlap_distance_m=float(validation.get("point_overlap_distance_m", 0.10)),
            min_point_overlap_ratio=float(validation.get("min_point_overlap_ratio", 0.20)),
            min_shape_score=float(validation.get("min_shape_score", 0.25)),
            alpha_2d_score=float(validation.get("alpha_2d_score", 0.55)),
            beta_3d_score=float(validation.get("beta_3d_score", 0.45)),
            final_accept_threshold=float(validation.get("final_accept_threshold", 0.58)),
            distance_chunk_size=int(validation.get("distance_chunk_size", 512)),
            covariance_epsilon=float(validation.get("covariance_epsilon", 1.0e-6)),
        ),
        memory=V23ConfidenceMemoryConfig(
            voxel_size_m=float(memory.get("voxel_size_m", 0.05)),
            max_voxels_per_object=int(memory.get("max_voxels_per_object", 4096)),
            max_points_per_observation=int(memory.get("max_points_per_observation", 2048)),
            point_confidence_floor=float(memory.get("point_confidence_floor", 0.30)),
            min_observation_score=float(memory.get("min_observation_score", 0.50)),
            raw_observation_confidence=float(memory.get("raw_observation_confidence", 1.0)),
            validated_recovery_confidence=float(memory.get("validated_recovery_confidence", 0.70)),
            recovery_candidate_confidence=float(memory.get("recovery_candidate_confidence", 0.30)),
            confidence_decay=float(memory.get("confidence_decay", 0.98)),
            recovery_support_distance_m=float(memory.get("recovery_support_distance_m", 0.10)),
            min_recovery_support_ratio=float(memory.get("min_recovery_support_ratio", 0.20)),
            min_recovery_supported_points=int(memory.get("min_recovery_supported_points", 8)),
            distance_chunk_size=int(memory.get("distance_chunk_size", 512)),
            covariance_epsilon=float(memory.get("covariance_epsilon", 1.0e-6)),
        ),
    )


def _load_recovery_config(raw: Mapping[str, Any]) -> V21RecoveryConfig:
    return V21RecoveryConfig(
        unobserved_gap=int(raw.get("unobserved_gap", 2)),
        reentry_gap=int(raw.get("reentry_gap", 3)),
        low_score_threshold=float(raw.get("low_score_threshold", 0.50)),
        min_mask_pixels=int(raw.get("min_mask_pixels", 32)),
        min_area_ratio=float(raw.get("min_area_ratio", 0.35)),
        max_area_ratio=float(raw.get("max_area_ratio", 2.85)),
        min_history_points=int(raw.get("min_history_points", 16)),
        max_points_per_object=int(raw.get("max_points_per_object", 4096)),
        min_memory_update_score=float(raw.get("min_memory_update_score", 0.50)),
        min_geometry_confidence=float(raw.get("min_geometry_confidence", 0.30)),
        max_positive_points=int(raw.get("max_positive_points", 6)),
        min_candidate_score=float(raw.get("min_candidate_score", 0.50)),
        min_candidate_support_recall=float(raw.get("min_candidate_support_recall", 0.15)),
        min_candidate_iou=float(raw.get("min_candidate_iou", 0.02)),
        min_candidate_area_ratio=float(raw.get("min_candidate_area_ratio", 0.20)),
        max_candidate_area_ratio=float(raw.get("max_candidate_area_ratio", 5.00)),
        min_geometry_consistency=float(raw.get("min_geometry_consistency", 0.12)),
        min_recovery_score=float(raw.get("min_recovery_score", 0.50)),
        replacement_margin=float(raw.get("replacement_margin", 0.12)),
        sam_score_weight=float(raw.get("sam_score_weight", 0.35)),
        geometry_score_weight=float(raw.get("geometry_score_weight", 0.40)),
        temporal_score_weight=float(raw.get("temporal_score_weight", 0.25)),
        proposal_box_quantile=float(raw.get("proposal_box_quantile", 0.02)),
        proposal_box_padding_ratio=float(raw.get("proposal_box_padding_ratio", 0.12)),
        min_projected_points=int(raw.get("min_projected_points", 8)),
        min_projected_fraction=float(raw.get("min_projected_fraction", 0.005)),
        min_supported_points=int(raw.get("min_supported_points", 8)),
        min_support_ratio=float(raw.get("min_support_ratio", 0.02)),
        support_abs_distance=float(raw.get("support_abs_distance", 0.15)),
        support_relative_distance=float(raw.get("support_relative_distance", 0.10)),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="streaming_couping/configs/v0_baseline.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--v22-map-dir")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _cpu_tensor(value: Any) -> torch.Tensor:
    if not torch.is_tensor(value):
        return torch.as_tensor(value)
    return value.detach().cpu()


if __name__ == "__main__":
    main()
