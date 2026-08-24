#!/usr/bin/env python3
"""Export the frozen V1.1 projection-memory semantic map.

Pipeline:

    cached RGB/SAM observations + frozen StreamVGGT pointmap/camera
        -> projected historical object masks
        -> top-k temporal confirmation
        -> persistent object IDs
        -> observation map and voxel-fused object map

No model is rerun and no GT is read by this exporter.  Evaluation-only GT is
handled by ``run_v11_semantic_map_evaluation.py``.
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

from streaming_couping.src.learned_pose.baseline_runtime import (
    load_baseline_run_config,
)
from streaming_couping.src.learned_pose.cache import cache_path, load_feature_cache
from streaming_couping.src.learned_pose.config import ClipConfig, load_learned_pose_config
from streaming_couping.src.projection_object_memory import (
    ProjectionAssociationConfig,
    ProjectionObjectMemory,
    ProjectionObjectMemoryConfig,
)
from streaming_couping.src.semantic_map import (
    BACKGROUND_SEMANTIC_RGB8,
    INSTANCE_PALETTE_RGB8,
    SemanticPointMap,
    build_persistent_semantic_pointmap,
    normalize_confidence,
)
from streaming_couping.src.storage import expand_storage_path

from streaming_couping.scripts.run_v0_semantic_map import _validate_inputs


REVISION = "v1_1_projection_temporal_voxel_semantic_mapping_r1"
BASELINE_REVISION = "v0_frozen_semantic_mapping_pipeline_r1"
RAW_CACHE_VARIANT = "sam31_online_forward"


@dataclass(frozen=True)
class V11Run:
    source_path: Path
    output_dir: Path
    confidence_threshold: float
    track_score_threshold: float
    max_map_points: int
    raw_cache_variant: str
    memory: ProjectionObjectMemoryConfig


def main() -> None:
    args = _parse_args()
    data = load_learned_pose_config(args.config)
    baseline = load_baseline_run_config(args.config)
    run = _load_run(args.config)
    if args.output_dir:
        run = replace(run, output_dir=Path(args.output_dir).expanduser().resolve())
    clip = _find_clip(data.clips, baseline.clip_name)
    cache_value = cache_path(data, clip)
    payload = load_feature_cache(cache_value)
    baseline_summary, poses = _validate_inputs(
        payload=payload,
        summary_path=baseline.output_dir / "baseline_summary.json",
        poses_path=baseline.output_dir / "poses.pt",
        clip=clip,
    )

    variant_masks = payload.get("tracking_variant_masks_stream")
    variant_scores = payload.get("tracking_variant_scores")
    if not isinstance(variant_masks, Mapping) or not isinstance(variant_scores, Mapping):
        raise ValueError("V1.1 cache lacks tracking variant stream fields.")
    if run.raw_cache_variant not in variant_masks or run.raw_cache_variant not in variant_scores:
        raise ValueError(
            f"Cache lacks raw SAM variant {run.raw_cache_variant!r}; rebuild V0 cache."
        )
    masks = variant_masks[run.raw_cache_variant].detach().bool().cpu()
    scores = variant_scores[run.raw_cache_variant].detach().float().cpu()
    confidence = normalize_confidence(payload["baseline_world_confidence"])
    appearance = None
    if "appearance" in payload and torch.is_tensor(payload["appearance"]):
        candidate = payload["appearance"].detach().float().cpu()
        if candidate.ndim == 3 and tuple(candidate.shape[:2]) == tuple(scores.shape):
            appearance = candidate

    memory = ProjectionObjectMemory(run.memory)
    image_size = tuple(int(value) for value in payload["image_size"])
    memory_result = memory.process_sequence(
        world_points=payload["baseline_world_points"],
        confidence=confidence,
        masks=masks,
        track_scores=scores,
        frame_indices=tuple(int(value) for value in payload["frame_indices"]),
        track_ids=tuple(int(value) for value in payload["sam_track_ids"]),
        track_prompts=tuple(str(value) for value in payload["sam_track_prompts"]),
        world_to_camera=payload["baseline_world_to_camera"],
        intrinsics=payload["baseline_intrinsics"],
        image_size=image_size,
        images=payload.get("stream_images"),
        appearance=appearance,
    )
    observation_map = build_persistent_semantic_pointmap(
        world_points=payload["baseline_world_points"],
        confidence=confidence,
        masks=masks,
        track_scores=scores,
        persistent_object_ids=memory_result["persistent_object_ids"],
        sam_track_ids=tuple(int(value) for value in payload["sam_track_ids"]),
        images=payload["stream_images"],
        confidence_threshold=run.confidence_threshold,
        track_score_threshold=run.track_score_threshold,
        max_map_points=run.max_map_points,
        map_write_mask=memory_result["map_write_mask"],
    )
    fused_map = memory.fused_map(max_points=run.max_map_points)
    object_prompts = {
        int(row["object_id"]): str(row["category"])
        for row in memory_result["objects"]
    }
    result = _write_outputs(
        payload=payload,
        baseline_summary=baseline_summary,
        poses=poses,
        observation_map=observation_map,
        fused_map=fused_map,
        memory=memory,
        memory_result=memory_result,
        object_prompts=object_prompts,
        cache_path_value=cache_value,
        poses_path=baseline.output_dir / "poses.pt",
        run=run,
    )
    print(f"V1.1 projection-memory semantic map result={result}")


def _write_outputs(
    *,
    payload: Mapping[str, Any],
    baseline_summary: Mapping[str, Any],
    poses: Mapping[str, Any],
    observation_map: SemanticPointMap,
    fused_map: Mapping[str, torch.Tensor],
    memory: ProjectionObjectMemory,
    memory_result: Mapping[str, object],
    object_prompts: Mapping[int, str],
    cache_path_value: Path,
    poses_path: Path,
    run: V11Run,
) -> Path:
    frames = tuple(int(value) for value in payload["frame_indices"])
    dense_ids = observation_map.dense_semantic_object_ids
    if dense_ids is None:
        dense_ids = observation_map.dense_semantic_slots
    object_rows: list[dict[str, object]] = []
    for row in memory_result["objects"]:
        current = dict(row)
        object_id = int(current["object_id"])
        owned = (dense_ids == object_id) & observation_map.dense_valid
        color = INSTANCE_PALETTE_RGB8[object_id % len(INSTANCE_PALETTE_RGB8)]
        current.update(
            {
                "dense_owned_valid_points": int(owned.sum()),
                "observation_saved_points": int(
                    (observation_map.semantic_slots == object_id).sum()
                ),
                "fused_saved_points": int(
                    (fused_map["object_ids"] == object_id).sum()
                ),
                "color_red": int(color[0]),
                "color_green": int(color[1]),
                "color_blue": int(color[2]),
                "prompt": str(object_prompts.get(object_id, current["category"])),
            }
        )
        object_rows.append(current)
    object_rows.sort(key=lambda row: int(row["object_id"]))

    observation_object_ids = observation_map.semantic_object_ids
    observation_track_ids = observation_map.semantic_track_ids
    if observation_object_ids is None or observation_track_ids is None:
        raise RuntimeError("V1.1 observation map lost identity provenance.")
    fused_object_ids = fused_map["object_ids"].detach().long().cpu()
    fused_track_ids = torch.full_like(fused_object_ids, -1)
    observation_points = int((observation_object_ids >= 0).sum())
    fused_points = int(fused_map["world_points"].shape[0])
    pipeline_ready = int(
        observation_map.world_points.shape[0] > 0
        and fused_points > 0
        and object_rows
        and poses["selected_pose_branch"] == "retrieve_qk"
    )
    run.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = run.output_dir / "semantic_map.pt"
    torch.save(
        {
            "schema": 4,
            "revision": REVISION,
            "baseline_revision": BASELINE_REVISION,
            "identity_mode": "v1_1_projection_temporal_voxel_memory",
            "raw_cache_variant": run.raw_cache_variant,
            "clip": payload["clip_name"],
            "frame_indices": frames,
            "reference_sequence_index": int(payload["reference_sequence_index"]),
            "coordinate_frame": "streamvggt_first_frame_reference_world",
            "world_points": observation_map.world_points,
            "rgb": observation_map.rgb,
            "semantic_rgb": observation_map.semantic_rgb,
            "semantic_slots": observation_map.semantic_slots,
            "semantic_object_ids": observation_object_ids,
            "semantic_track_ids": observation_track_ids,
            "persistent_object_ids_dense": dense_ids,
            "persistent_object_ids_by_sam_slot": memory_result[
                "persistent_object_ids"
            ],
            "map_write_mask": memory_result["map_write_mask"],
            "confidence": observation_map.confidence,
            "sequence_indices": observation_map.sequence_indices,
            "flat_indices": observation_map.flat_indices,
            "object_metadata": object_rows,
            "object_memory": memory.to_dict(),
            "association_events": memory_result["events"],
            "fused_world_points": fused_map["world_points"],
            "fused_rgb": fused_map["rgb"],
            "fused_confidence": fused_map["confidence"],
            "fused_object_ids": fused_object_ids,
            "fused_observation_counts": fused_map["observation_counts"],
            "slot_palette_rgb8": INSTANCE_PALETTE_RGB8,
            "unlabeled_background_rgb8": BACKGROUND_SEMANTIC_RGB8,
            "raw_world_to_camera": poses["raw_world_to_camera"],
            "selected_world_to_camera": poses["selected_world_to_camera"],
            "selected_pose_branch": poses["selected_pose_branch"],
            "baseline_world_to_camera": payload["baseline_world_to_camera"],
            "baseline_intrinsics": payload["baseline_intrinsics"],
            "pointmap_source": "raw_full_history_streamvggt_world_pointmap",
            "semantic_source": "sam_projection_iou_temporal_confirmation_voxel_fusion",
            "pose_source": "streamvggt_native_qk_history_retrieval",
        },
        artifact_path,
    )

    observation_ply = run.output_dir / "observation_map.ply"
    _write_binary_ply(
        observation_ply,
        points=observation_map.world_points,
        rgb=observation_map.semantic_rgb,
        object_ids=observation_object_ids,
        sam_track_ids=observation_track_ids,
        confidence=observation_map.confidence,
        observation_counts=torch.ones_like(observation_map.confidence, dtype=torch.long),
    )
    fused_ply = run.output_dir / "semantic_map.ply"
    _write_binary_ply(
        fused_ply,
        points=fused_map["world_points"],
        rgb=_semantic_colors(fused_object_ids),
        object_ids=fused_object_ids,
        sam_track_ids=fused_track_ids,
        confidence=fused_map["confidence"],
        observation_counts=fused_map["observation_counts"],
    )
    rgb_ply = run.output_dir / "rgb_map.ply"
    _write_binary_ply(
        rgb_ply,
        points=observation_map.world_points,
        rgb=observation_map.rgb,
        object_ids=observation_object_ids,
        sam_track_ids=observation_track_ids,
        confidence=observation_map.confidence,
        observation_counts=torch.ones_like(observation_map.confidence, dtype=torch.long),
    )
    fused_rgb_ply = run.output_dir / "fused_rgb_map.ply"
    _write_binary_ply(
        fused_rgb_ply,
        points=fused_map["world_points"],
        rgb=fused_map["rgb"],
        object_ids=fused_object_ids,
        sam_track_ids=fused_track_ids,
        confidence=fused_map["confidence"],
        observation_counts=fused_map["observation_counts"],
    )
    memory_json = memory.save_json(run.output_dir / "object_memory.json")
    memory_csv = memory.save_csv(run.output_dir / "object_memory.csv")
    events_csv = _write_csv(run.output_dir / "association_events.csv", memory_result["events"])
    prompt_csv = _write_csv(
        run.output_dir / "prompt_discovery.csv",
        [dict(row) for row in payload.get("sam_prompt_diagnostics", ())],
    )
    summary: dict[str, object] = {
        "schema": 4,
        "revision": REVISION,
        "baseline_version": "v0",
        "baseline_status": "frozen",
        "identity_mode": "v1_1_projection_temporal_voxel_memory",
        "raw_cache_variant": run.raw_cache_variant,
        "clip": payload["clip_name"],
        "config": str(run.source_path),
        "cache": str(cache_path_value),
        "poses": str(poses_path),
        "frames": frames,
        "reference_frame": frames[int(payload["reference_sequence_index"])],
        "map_generation_gt_fields": 0,
        "model_trained": 0,
        "pose_source": "streamvggt_native_qk_history_retrieval",
        "pointmap_source": "raw_full_history_streamvggt_world_pointmap",
        "semantic_source": "sam_projection_iou_temporal_confirmation_voxel_fusion",
        "prompt_selection_annotation_gt_used": int(
            baseline_summary["prompt_selection_annotation_gt_used"]
        ),
        "runtime_prompt_selection_gt_fields": 0,
        "source_sam_track_count": int(len(payload["sam_track_ids"])),
        "persistent_object_count": int(len(object_rows)),
        "persistent_identity_compression": len(object_rows)
        / max(1, len(payload["sam_track_ids"])),
        "pending_track_count": int(memory_result["pending_track_count"]),
        "confirmed_observation_count": int(memory_result["confirmed_observation_count"]),
        "observation_map_points": int(observation_map.world_points.shape[0]),
        "observation_semantic_points": observation_points,
        "observation_semantic_coverage_percent": (
            100.0 * observation_points / max(1, int(observation_map.world_points.shape[0]))
        ),
        "fused_map_points": fused_points,
        "fused_voxel_observation_count_sum": int(
            fused_map["observation_counts"].sum()
        ),
        "fused_map_compression_vs_observation": fused_points
        / max(1, observation_points),
        "scene_scale": float(memory_result["scene_scale"]),
        "association_policy": {
            "primary": "historical_world_points_projected_to_current_mask_iou",
            "top_k": int(run.memory.association.top_k),
            "confirmation_frames": int(run.memory.confirmation_frames),
            "confirmation_window": int(run.memory.confirmation_window),
            "voxel_fusion": 1,
        },
        "selected_pose_branch": poses["selected_pose_branch"],
        "pose_improvement_claim": int(baseline_summary["pose_improvement_claim"]),
        "pointmap_modified": 0,
        "sam_hidden_features_used": 0,
        "sam_pose_inputs": 0,
        "semantic_map_pipeline_ready": pipeline_ready,
        "claim": "frozen_v1_1_projection_temporal_voxel_semantic_mapping",
        "outputs": {
            "artifact": str(artifact_path),
            "semantic_ply": str(fused_ply),
            "observation_map_ply": str(observation_ply),
            "rgb_ply": str(rgb_ply),
            "fused_rgb_ply": str(fused_rgb_ply),
            "object_memory_json": str(memory_json),
            "object_memory_csv": str(memory_csv),
            "association_events_csv": str(events_csv),
            "prompt_discovery_csv": str(prompt_csv),
        },
        "objects": object_rows,
    }
    summary_path = run.output_dir / "semantic_map_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf8",
    )
    _write_copyable(run.output_dir / "copyable_result.txt", summary)
    print("V1.1 PROJECTION-FIRST TEMPORAL VOXEL SEMANTIC MAP")
    print(
        f"  frames={len(frames)} sam_tracks={len(payload['sam_track_ids'])} "
        f"objects={len(object_rows)} pending={summary['pending_track_count']} "
        f"observation_points={summary['observation_map_points']} "
        f"fused_points={summary['fused_map_points']} ready={pipeline_ready}"
    )
    print(f"  semantic_ply={fused_ply}")
    print(f"  observation_ply={observation_ply}")
    print(f"  memory={memory_json}")
    return summary_path


def _write_binary_ply(
    path: Path,
    *,
    points: torch.Tensor,
    rgb: torch.Tensor,
    object_ids: torch.Tensor,
    sam_track_ids: torch.Tensor,
    confidence: torch.Tensor,
    observation_counts: torch.Tensor,
) -> None:
    points = points.detach().float().cpu().reshape(-1, 3)
    rgb = rgb.detach().float().cpu().reshape(-1, 3).clamp(0.0, 1.0)
    object_ids = object_ids.detach().long().cpu().reshape(-1)
    sam_track_ids = sam_track_ids.detach().long().cpu().reshape(-1)
    confidence = confidence.detach().float().cpu().reshape(-1)
    observation_counts = observation_counts.detach().long().cpu().reshape(-1)
    count = int(points.shape[0])
    if any(
        int(value.shape[0]) != count
        for value in (rgb, object_ids, sam_track_ids, confidence, observation_counts)
    ):
        raise ValueError("V1.1 PLY fields have different point counts.")
    array = np.empty(
        count,
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("semantic_slot", "<i4"),
            ("persistent_object_id", "<i4"),
            # Keep the legacy viewer schema.  Full-width SAM IDs remain in
            # semantic_map.pt and association_events.csv.
            ("sam_track_id", "<i4"),
            ("confidence", "<f4"),
            ("observation_count", "<i4"),
        ],
    )
    array["x"], array["y"], array["z"] = points.numpy().T
    array["red"], array["green"], array["blue"] = (
        rgb.mul(255).round().byte().numpy().T
    )
    array["semantic_slot"] = object_ids.numpy().astype(np.int32)
    array["persistent_object_id"] = object_ids.numpy().astype(np.int32)
    array["sam_track_id"] = sam_track_ids.numpy().astype(np.int32)
    array["confidence"] = confidence.numpy()
    array["observation_count"] = observation_counts.numpy().astype(np.int32)
    header = "\n".join(
        (
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {count}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "property int semantic_slot",
            "property int persistent_object_id",
            "property int sam_track_id",
            "property float confidence",
            "property int observation_count",
            "end_header",
            "",
        )
    ).encode("ascii")
    with path.open("wb") as handle:
        handle.write(header)
        array.tofile(handle)


def _semantic_colors(object_ids: torch.Tensor) -> torch.Tensor:
    palette = torch.tensor(INSTANCE_PALETTE_RGB8, dtype=torch.float32) / 255.0
    ids = object_ids.detach().long().cpu()
    if not ids.numel():
        return torch.empty(0, 3)
    return palette.index_select(0, ids.remainder(palette.shape[0]))


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    outputs = summary["outputs"]
    lines = [
        "===== COPYABLE_V1_1_PROJECTION_MEMORY_SEMANTIC_MAP_BEGIN =====",
        f"revision={summary['revision']}",
        f"clip={summary['clip']}",
        "identity_mode=v1_1_projection_temporal_voxel_memory",
        f"raw_cache_variant={summary['raw_cache_variant']}",
        "pose=retrieve_qk",
        "pointmap=raw_full_history_streamvggt_world_pointmap",
        "association=projected_historical_world_points_vs_current_sam_mask_iou",
        f"source_sam_track_count={summary['source_sam_track_count']}",
        f"persistent_object_count={summary['persistent_object_count']}",
        f"pending_track_count={summary['pending_track_count']}",
        f"confirmed_observation_count={summary['confirmed_observation_count']}",
        f"observation_map_points={summary['observation_map_points']}",
        f"observation_semantic_points={summary['observation_semantic_points']}",
        f"fused_map_points={summary['fused_map_points']}",
        f"fused_map_compression_vs_observation={summary['fused_map_compression_vs_observation']}",
        f"semantic_map_pipeline_ready={summary['semantic_map_pipeline_ready']}",
        f"artifact={outputs['artifact']}",
        f"semantic_ply={outputs['semantic_ply']}",
        f"observation_map_ply={outputs['observation_map_ply']}",
        f"object_memory_json={outputs['object_memory_json']}",
        f"object_memory_csv={outputs['object_memory_csv']}",
        f"association_events_csv={outputs['association_events_csv']}",
        "===== COPYABLE_V1_1_PROJECTION_MEMORY_SEMANTIC_MAP_END =====",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def _load_run(path: str | Path) -> V11Run:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf8") as handle:
        raw = yaml.safe_load(handle) or {}
    section = raw.get("semantic_map", {}) or {}
    v11 = section.get("v1_1_projection_memory", {}) or {}
    association = v11.get("association", {}) or {}
    association_config = ProjectionAssociationConfig(
        projection_iou_weight=float(association.get("projection_iou_weight", 0.70)),
        projection_recall_weight=float(association.get("projection_recall_weight", 0.12)),
        projection_precision_weight=float(association.get("projection_precision_weight", 0.08)),
        center_weight=float(association.get("center_weight", 0.05)),
        category_weight=float(association.get("category_weight", 0.05)),
        appearance_weight=float(association.get("appearance_weight", 0.0)),
        min_projection_iou=float(association.get("min_projection_iou", 0.025)),
        min_match_score=float(association.get("min_match_score", 0.28)),
        min_projected_pixels=int(association.get("min_projected_pixels", 8)),
        top_k=int(association.get("top_k", 5)),
        projection_dilation_radius=int(association.get("projection_dilation_radius", 3)),
        max_projection_points=int(association.get("max_projection_points", 4096)),
        center_scale_ratio=float(association.get("center_scale_ratio", 0.12)),
        absolute_center_scale=float(association.get("absolute_center_scale", 0.02)),
        category_hard_gate=bool(association.get("category_hard_gate", True)),
    )
    memory = ProjectionObjectMemoryConfig(
        max_points_per_object=int(v11.get("max_points_per_object", 4096)),
        max_fused_voxels_per_object=int(v11.get("max_fused_voxels_per_object", 20000)),
        min_observation_points=int(v11.get("min_observation_points", 16)),
        min_mask_pixels=int(v11.get("min_mask_pixels", 32)),
        min_track_score=float(v11.get("min_track_score", 0.50)),
        min_geometry_confidence=float(v11.get("min_geometry_confidence", 0.30)),
        center_ema_alpha=float(v11.get("center_ema_alpha", 0.25)),
        confirmation_frames=int(v11.get("confirmation_frames", 2)),
        confirmation_window=int(v11.get("confirmation_window", 4)),
        max_pending_gap=int(v11.get("max_pending_gap", 4)),
        voxel_size_ratio=float(v11.get("voxel_size_ratio", 0.02)),
        absolute_voxel_size=float(v11.get("absolute_voxel_size", 0.02)),
        association=association_config,
    )
    run = V11Run(
        source_path=source,
        output_dir=expand_storage_path(
            v11.get(
                "output_dir",
                "${VGGT_SAM_STORAGE_ROOT}/outputs/streaming_couping_v11/semantic_map",
            ),
            base=source.parent,
        ),
        confidence_threshold=float(section.get("confidence_threshold", 0.30)),
        track_score_threshold=float(section.get("track_score_threshold", 0.50)),
        max_map_points=int(section.get("max_map_points", 400000)),
        raw_cache_variant=str(v11.get("raw_cache_variant", RAW_CACHE_VARIANT)),
        memory=memory,
    )
    if not 0.0 <= run.confidence_threshold <= 1.0:
        raise ValueError("semantic_map.confidence_threshold must be in [0,1].")
    if not 0.0 <= run.track_score_threshold <= 1.0:
        raise ValueError("semantic_map.track_score_threshold must be in [0,1].")
    if run.max_map_points < 1:
        raise ValueError("semantic_map.max_map_points must be positive.")
    return run


def _find_clip(clips: tuple[ClipConfig, ...], name: str) -> ClipConfig:
    selected = [clip for clip in clips if clip.name == name]
    if len(selected) != 1:
        raise ValueError(f"Clip {name!r} was not found exactly once.")
    return selected[0]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="streaming_couping/configs/v0_baseline.yaml"
    )
    parser.add_argument("--output-dir")
    return parser.parse_args()


if __name__ == "__main__":
    main()
