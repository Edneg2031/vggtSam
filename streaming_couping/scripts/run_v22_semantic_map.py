#!/usr/bin/env python3
"""Run V2.2 failure-only recovery with world-space validation."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any, Mapping

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
from streaming_couping.src.learned_pose.config import load_learned_pose_config
from streaming_couping.src.semantic_map import (
    SemanticPointMap,
    build_semantic_pointmap,
    normalize_confidence,
)
from streaming_couping.src.storage import expand_storage_path
from streaming_couping.src.types import RevisitCandidate
from streaming_couping.src.v21_geometry_recovery import (
    V21RecoveryConfig,
    process_v21_sequence,
)
from streaming_couping.src.v22_geometry_validation import (
    V22GeometryValidationConfig,
    validate_v22_candidate,
)

from streaming_couping.scripts.run_v0_semantic_map import _validate_inputs
from streaming_couping.scripts.run_v2_semantic_map import (
    DEFAULT_RAW_VARIANT,
    _find_clip,
    _load_raw_variants,
    _pose_sequence,
)
from streaming_couping.scripts.run_v21_semantic_map import (
    _track_metadata,
    _write_binary_ply,
    _write_csv,
    _write_outputs,
)
from streaming_couping.src.v2_geometry_recovery import V2ObjectMemoryState


REVISION = "v2_2_failure_only_geometry_consistency_recovery_r1"
IDENTITY_MODE = "v2_2_failure_only_geometry_consistency_recovery"
SEMANTIC_SOURCE = "sam3_raw_tracking_with_failure_only_2d_and_world_geometry_recovery"
CLAIM = "frozen_v2_2_failure_only_geometry_consistency_recovery"


@dataclass(frozen=True)
class V22Run:
    source_path: Path
    output_dir: Path
    confidence_threshold: float
    track_score_threshold: float
    max_map_points: int
    raw_cache_variant: str
    recovery: V21RecoveryConfig
    validation: V22GeometryValidationConfig


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
        raise ValueError("V2.2 recovery requires sam3.version=sam3.1.")
    print("V2.2 FAILURE-ONLY RECOVERY WITH WORLD-SPACE VALIDATION")
    print(f"cache={cache_value}")
    print(f"raw_cache_variant={run.raw_cache_variant}")
    print(
        "policy=V2.1 failure-only 2D candidate gate plus world-space "
        "centroid/overlap/shape validation"
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

    def proposal_builder(frame: int, slot: int, memory: V2ObjectMemoryState):
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

    def refine_callback(frame: int, slot: int, prompt: str, proposal: RevisitCandidate):
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

    def candidate_validator(**kwargs: Any) -> Mapping[str, Any]:
        return validate_v22_candidate(config=run.validation, **kwargs)

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
        candidate_validator=candidate_validator,
    )
    _add_validation_stats(result["stats"], result["events"])
    _annotate_object_geometry(result)
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
        revision=REVISION,
        identity_mode=IDENTITY_MODE,
        semantic_source=SEMANTIC_SOURCE,
        claim=CLAIM,
        identity_policy="one_v22_memory_entry_per_existing_v0_sam_slot",
        recovery_description=(
            "V2.1_failure_only_plus_world_centroid_overlap_covariance_validation"
        ),
        copyable_writer=_write_copyable,
    )
    summary_path = run.output_dir / "semantic_map_v22_summary.json"
    summary_path.write_text(Path(output_path).read_text(encoding="utf8"), encoding="utf8")
    stats = result["stats"]
    print(
        "  recovery_trigger_count="
        f"{stats['recovery_trigger_count']} "
        f"accepted_recovery_count={stats['accepted_recovery_count']} "
        f"geometry_validation_accept_count={stats['geometry_validation_accept_count']} "
        f"geometry_validation_reject_count={stats['geometry_validation_reject_count']}"
    )
    print(
        f"  geometry_validation_reject_reasons={stats['geometry_validation_reject_reasons']} "
        f"raw_fallback_unchanged={stats['raw_fallback_unchanged_ratio']:.4f} "
        f"normal_unchanged={stats['normal_frame_unchanged_ratio']:.4f}"
    )
    print(
        f"  points={semantic.world_points.shape[0]} "
        f"semantic={int((semantic.semantic_slots >= 0).sum())}"
    )
    print(f"  ply={run.output_dir / 'semantic_map.ply'}")
    print(f"V2.2 semantic map result={output_path}")
    print(f"V2.2 semantic map summary={summary_path}")


def _add_validation_stats(stats: dict[str, Any], events: list[dict[str, Any]]) -> None:
    attempted = [row for row in events if int(row.get("geometry_validation_attempted", 0))]
    accepted = [row for row in attempted if int(row.get("geometry_validation_accepted", 0))]
    rejected = [row for row in attempted if not int(row.get("geometry_validation_accepted", 0))]
    reasons = Counter(str(row.get("geometry_validation_reason", "unknown")) for row in rejected)
    stats["geometry_validation_attempt_count"] = int(len(attempted))
    stats["geometry_validation_accept_count"] = int(len(accepted))
    stats["geometry_validation_reject_count"] = int(len(rejected))
    stats["geometry_validation_accept_rate"] = (
        float(len(accepted)) / float(len(attempted)) if attempted else 1.0
    )
    stats["geometry_validation_reject_reasons"] = dict(sorted(reasons.items()))


def _annotate_object_geometry(result: dict[str, Any]) -> None:
    """Persist compact world-space memory statistics beside point tensors."""

    tensors = result["object_tensors"]
    rows = result["objects"]
    for row in rows:
        slot = int(row["object_id"])
        points = tensors[slot]["world_points"].detach().float().cpu().reshape(-1, 3)
        points = points[torch.isfinite(points).all(dim=-1)]
        if not bool(points.numel()):
            row["world_centroid"] = []
            row["world_bbox_min"] = []
            row["world_bbox_max"] = []
            row["world_covariance"] = []
            continue
        center = points.mean(dim=0)
        if int(points.shape[0]) > 1:
            centered = points - center[None]
            covariance = centered.T @ centered / max(1, int(points.shape[0]) - 1)
        else:
            covariance = torch.zeros(3, 3)
        row["world_centroid"] = [float(value) for value in center.tolist()]
        row["world_bbox_min"] = [float(value) for value in points.min(dim=0).values.tolist()]
        row["world_bbox_max"] = [float(value) for value in points.max(dim=0).values.tolist()]
        row["world_covariance"] = [
            [float(value) for value in values] for values in covariance.tolist()
        ]


def _write_copyable(path: Path, summary: Mapping[str, Any]) -> None:
    stats = summary["recovery_stats"]
    outputs = summary["outputs"]
    lines = [
        "===== COPYABLE_V2_2_FAILURE_ONLY_GEOMETRY_CONSISTENCY_BEGIN =====",
        f"revision={summary['revision']}",
        f"clip={summary['clip']}",
        f"frames={len(summary['frames'])}",
        f"identity_mode={IDENTITY_MODE}",
        "pose=retrieve_qk",
        "pointmap=raw_full_history_streamvggt_world_pointmap",
        "normal_frames=raw_sam_mask_exact_copy",
        "recovery=V2.1_failure_only_plus_world_centroid_overlap_covariance_validation",
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
        f"geometry_validation_accept_rate={stats['geometry_validation_accept_rate']}",
        f"geometry_validation_reject_reasons={stats['geometry_validation_reject_reasons']}",
        f"raw_fallback_unchanged_ratio={stats['raw_fallback_unchanged_ratio']}",
        f"normal_frame_unchanged_ratio={stats['normal_frame_unchanged_ratio']}",
        f"semantic_map_pipeline_ready={summary['semantic_map_pipeline_ready']}",
        f"artifact={outputs['artifact']}",
        f"semantic_ply={outputs['semantic_ply']}",
        f"rgb_ply={outputs['rgb_ply']}",
        f"recovery_events_csv={outputs['recovery_events_csv']}",
        f"object_memory_json={outputs['object_memory_json']}",
        f"object_memory_csv={outputs['object_memory_csv']}",
        "===== COPYABLE_V2_2_FAILURE_ONLY_GEOMETRY_CONSISTENCY_END =====",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def _load_run(path: str | Path) -> V22Run:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf8") as handle:
        raw = yaml.safe_load(handle) or {}
    section = raw.get("semantic_map", {}) or {}
    v22 = section.get("v2_2_geometry_recovery", {}) or {}
    recovery = v22.get("recovery", {}) or {}
    validation = v22.get("validation", {}) or {}
    return V22Run(
        source_path=source,
        output_dir=expand_storage_path(
            v22.get(
                "output_dir",
                "${VGGT_SAM_STORAGE_ROOT}/outputs/streaming_couping_v22/semantic_map",
            ),
            base=source.parent,
        ),
        confidence_threshold=float(section.get("confidence_threshold", 0.30)),
        track_score_threshold=float(section.get("track_score_threshold", 0.50)),
        max_map_points=int(section.get("max_map_points", 400000)),
        raw_cache_variant=str(v22.get("raw_cache_variant", DEFAULT_RAW_VARIANT)),
        recovery=V21RecoveryConfig(
            unobserved_gap=int(recovery.get("unobserved_gap", 2)),
            reentry_gap=int(recovery.get("reentry_gap", 3)),
            low_score_threshold=float(recovery.get("low_score_threshold", 0.50)),
            min_mask_pixels=int(recovery.get("min_mask_pixels", 32)),
            min_area_ratio=float(recovery.get("min_area_ratio", 0.35)),
            max_area_ratio=float(recovery.get("max_area_ratio", 2.85)),
            min_history_points=int(recovery.get("min_history_points", 16)),
            max_points_per_object=int(recovery.get("max_points_per_object", 4096)),
            min_memory_update_score=float(recovery.get("min_memory_update_score", 0.50)),
            min_geometry_confidence=float(recovery.get("min_geometry_confidence", 0.30)),
            max_positive_points=int(recovery.get("max_positive_points", 6)),
            min_candidate_score=float(recovery.get("min_candidate_score", 0.50)),
            min_candidate_support_recall=float(recovery.get("min_candidate_support_recall", 0.15)),
            min_candidate_iou=float(recovery.get("min_candidate_iou", 0.02)),
            min_candidate_area_ratio=float(recovery.get("min_candidate_area_ratio", 0.20)),
            max_candidate_area_ratio=float(recovery.get("max_candidate_area_ratio", 5.00)),
            min_geometry_consistency=float(recovery.get("min_geometry_consistency", 0.12)),
            min_recovery_score=float(recovery.get("min_recovery_score", 0.50)),
            replacement_margin=float(recovery.get("replacement_margin", 0.12)),
            sam_score_weight=float(recovery.get("sam_score_weight", 0.35)),
            geometry_score_weight=float(recovery.get("geometry_score_weight", 0.40)),
            temporal_score_weight=float(recovery.get("temporal_score_weight", 0.25)),
            proposal_box_quantile=float(recovery.get("proposal_box_quantile", 0.02)),
            proposal_box_padding_ratio=float(recovery.get("proposal_box_padding_ratio", 0.12)),
            min_projected_points=int(recovery.get("min_projected_points", 8)),
            min_projected_fraction=float(recovery.get("min_projected_fraction", 0.005)),
            min_supported_points=int(recovery.get("min_supported_points", 8)),
            min_support_ratio=float(recovery.get("min_support_ratio", 0.02)),
            support_abs_distance=float(recovery.get("support_abs_distance", 0.15)),
            support_relative_distance=float(recovery.get("support_relative_distance", 0.10)),
        ),
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
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="streaming_couping/configs/v0_baseline.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--sam3-device")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
