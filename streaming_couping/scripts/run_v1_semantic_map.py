#!/usr/bin/env python3
"""Export a frozen V1 persistent-3-D-instance semantic map.

This exporter consumes the cache and the already accepted V0 QK pose
artifact.  It does not rerun StreamVGGT or SAM3 and it never reads GT.  The
only change from V0 is the identity layer:

    SAM observation -> persistent object ID -> accumulated world points
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

from streaming_couping.src.learned_pose.baseline_runtime import (
    load_baseline_run_config,
)
from streaming_couping.src.learned_pose.cache import cache_path, load_feature_cache
from streaming_couping.src.learned_pose.config import ClipConfig, load_learned_pose_config
from streaming_couping.src.object_memory import (
    PersistentObjectMemory,
    PersistentObjectMemoryConfig,
)
from streaming_couping.src.instance_association import InstanceAssociationConfig
from streaming_couping.src.semantic_map import (
    BACKGROUND_SEMANTIC_RGB8,
    INSTANCE_PALETTE_RGB8,
    SemanticPointMap,
    build_persistent_semantic_pointmap,
    normalize_confidence,
)
from streaming_couping.src.storage import expand_storage_path

from streaming_couping.scripts.run_v0_semantic_map import _validate_inputs


REVISION = "v1_persistent_3d_instance_semantic_mapping_r1"
BASELINE_REVISION = "v0_frozen_semantic_mapping_pipeline_r1"


@dataclass(frozen=True)
class V1Run:
    source_path: Path
    output_dir: Path
    identity_mode: str
    confidence_threshold: float
    track_score_threshold: float
    max_map_points: int
    memory: PersistentObjectMemoryConfig
    use_cached_appearance: bool


def main() -> None:
    args = _parse_args()
    data = load_learned_pose_config(args.config)
    baseline = load_baseline_run_config(args.config)
    run = _load_run(args.config, identity_mode=args.identity_mode)
    if args.output_dir:
        run = replace(run, output_dir=Path(args.output_dir).expanduser().resolve())
    if run.identity_mode != "v1_object_memory":
        raise ValueError(
            "V1 exporter requires semantic_map.identity_mode=v1_object_memory "
            "or --identity-mode v1_object_memory."
        )
    clip = _find_clip(data.clips, baseline.clip_name)
    cache_value = cache_path(data, clip)
    payload = load_feature_cache(cache_value)
    summary, poses = _validate_inputs(
        payload=payload,
        summary_path=baseline.output_dir / "baseline_summary.json",
        poses_path=baseline.output_dir / "poses.pt",
        clip=clip,
    )
    masks = payload["tracking_masks_stream"].detach().bool().cpu()
    scores = payload["tracking_scores"].detach().float().cpu()
    confidence = normalize_confidence(payload["baseline_world_confidence"])
    appearance = None
    if run.use_cached_appearance and "appearance" in payload:
        candidate = payload["appearance"]
        if torch.is_tensor(candidate):
            appearance = candidate.detach().float().cpu()
    memory = PersistentObjectMemory(run.memory)
    memory_result = memory.process_sequence(
        world_points=payload["baseline_world_points"],
        confidence=confidence,
        masks=masks,
        track_scores=scores,
        frame_indices=tuple(int(value) for value in payload["frame_indices"]),
        track_ids=tuple(int(value) for value in payload["sam_track_ids"]),
        track_prompts=tuple(str(value) for value in payload["sam_track_prompts"]),
        appearance=appearance,
    )
    object_prompts = {
        int(row["object_id"]): str(row["category"])
        for row in memory_result["objects"]
    }
    semantic = build_persistent_semantic_pointmap(
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
    result = _write_outputs(
        payload=payload,
        baseline_summary=summary,
        poses=poses,
        semantic=semantic,
        memory=memory,
        memory_result=memory_result,
        object_prompts=object_prompts,
        cache_path_value=cache_value,
        poses_path=baseline.output_dir / "poses.pt",
        run=run,
    )
    print(f"V1 persistent semantic map result={result}")


def _write_outputs(
    *,
    payload: Mapping[str, Any],
    baseline_summary: Mapping[str, Any],
    poses: Mapping[str, Any],
    semantic: SemanticPointMap,
    memory: PersistentObjectMemory,
    memory_result: Mapping[str, object],
    object_prompts: Mapping[int, str],
    cache_path_value: Path,
    poses_path: Path,
    run: V1Run,
) -> Path:
    frames = tuple(int(value) for value in payload["frame_indices"])
    object_rows = []
    dense_ids = semantic.dense_semantic_object_ids
    if dense_ids is None:
        dense_ids = semantic.dense_semantic_slots
    for row in memory_result["objects"]:
        current = dict(row)
        object_id = int(current["object_id"])
        owned = (dense_ids == object_id) & semantic.dense_valid
        current["dense_owned_valid_points"] = int(owned.sum())
        current["saved_points"] = int((semantic.semantic_slots == object_id).sum())
        color = INSTANCE_PALETTE_RGB8[object_id % len(INSTANCE_PALETTE_RGB8)]
        current.update(
            {
                "color_red": int(color[0]),
                "color_green": int(color[1]),
                "color_blue": int(color[2]),
                "prompt": str(object_prompts.get(object_id, current["category"])),
            }
        )
        object_rows.append(current)
    object_rows.sort(key=lambda row: int(row["object_id"]))
    semantic_object_ids = semantic.semantic_object_ids
    semantic_track_ids = semantic.semantic_track_ids
    if semantic_object_ids is None or semantic_track_ids is None:
        raise RuntimeError("V1 semantic map did not retain identity provenance.")
    semantic_points = int((semantic_object_ids >= 0).sum())
    pipeline_ready = int(
        semantic.world_points.shape[0] > 0
        and semantic_points > 0
        and object_rows
        and poses["selected_pose_branch"] == "retrieve_qk"
    )
    run.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = run.output_dir / "semantic_map.pt"
    torch.save(
        {
            "schema": 3,
            "revision": REVISION,
            "baseline_revision": BASELINE_REVISION,
            "identity_mode": "v1_object_memory",
            "clip": payload["clip_name"],
            "frame_indices": frames,
            "reference_sequence_index": int(payload["reference_sequence_index"]),
            "coordinate_frame": "streamvggt_first_frame_reference_world",
            "world_points": semantic.world_points,
            "rgb": semantic.rgb,
            "semantic_rgb": semantic.semantic_rgb,
            "semantic_slots": semantic.semantic_slots,
            "semantic_object_ids": semantic_object_ids,
            "semantic_track_ids": semantic_track_ids,
            "persistent_object_ids_dense": dense_ids,
            "persistent_object_ids_by_sam_slot": memory_result[
                "persistent_object_ids"
            ],
            "map_write_mask": memory_result["map_write_mask"],
            "confidence": semantic.confidence,
            "sequence_indices": semantic.sequence_indices,
            "flat_indices": semantic.flat_indices,
            "object_metadata": object_rows,
            "object_memory": memory.to_dict(),
            "association_events": memory_result["events"],
            "slot_palette_rgb8": INSTANCE_PALETTE_RGB8,
            "unlabeled_background_rgb8": BACKGROUND_SEMANTIC_RGB8,
            "raw_world_to_camera": poses["raw_world_to_camera"],
            "selected_world_to_camera": poses["selected_world_to_camera"],
            "selected_pose_branch": poses["selected_pose_branch"],
            "pointmap_source": "raw_full_history_streamvggt_world_pointmap",
            "semantic_source": "sam_observation_to_persistent_3d_object_memory",
            "pose_source": "streamvggt_native_qk_history_retrieval",
        },
        artifact_path,
    )
    ply_path = run.output_dir / "semantic_map.ply"
    _write_binary_ply(
        ply_path,
        points=semantic.world_points,
        rgb=semantic.semantic_rgb,
        object_ids=semantic_object_ids,
        sam_track_ids=semantic_track_ids,
        confidence=semantic.confidence,
    )
    rgb_ply_path = run.output_dir / "rgb_map.ply"
    _write_binary_ply(
        rgb_ply_path,
        points=semantic.world_points,
        rgb=semantic.rgb,
        object_ids=semantic_object_ids,
        sam_track_ids=semantic_track_ids,
        confidence=semantic.confidence,
    )
    object_json = memory.save_json(run.output_dir / "object_memory.json")
    object_csv = memory.save_csv(run.output_dir / "object_memory.csv")
    events_csv = _write_csv(run.output_dir / "association_events.csv", memory_result["events"])
    prompt_path = run.output_dir / "prompt_discovery.csv"
    _write_csv(prompt_path, [dict(row) for row in payload.get("sam_prompt_diagnostics", ())])
    output = {
        "schema": 3,
        "revision": REVISION,
        "baseline_version": "v0",
        "baseline_status": "frozen",
        "identity_mode": "v1_object_memory",
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
        "semantic_source": "sam_observation_to_persistent_3d_object_memory",
        "prompt_selection_annotation_gt_used": int(
            baseline_summary["prompt_selection_annotation_gt_used"]
        ),
        "runtime_prompt_selection_gt_fields": 0,
        "persistent_object_count": int(len(object_rows)),
        "source_sam_track_count": int(len(payload["sam_track_ids"])),
        "persistent_identity_compression": (
            len(object_rows) / max(1, len(payload["sam_track_ids"]))
        ),
        "saved_map_points": int(semantic.world_points.shape[0]),
        "saved_semantic_points": semantic_points,
        "semantic_coverage_percent": (
            100.0 * semantic_points / int(semantic.world_points.shape[0])
        ),
        "map_write_observations": int(memory_result["map_write_mask"].sum()),
        "scene_scale": float(memory_result["scene_scale"]),
        "semantic_color_policy": "fixed_color_per_persistent_object_id",
        "selected_pose_branch": poses["selected_pose_branch"],
        "pose_improvement_claim": int(baseline_summary["pose_improvement_claim"]),
        "pointmap_modified": 0,
        "sam_hidden_features_used": 0,
        "sam_pose_inputs": 0,
        "semantic_map_pipeline_ready": pipeline_ready,
        "claim": "frozen_v1_persistent_3d_instance_memory_semantic_mapping",
        "outputs": {
            "artifact": str(artifact_path),
            "semantic_ply": str(ply_path),
            "rgb_ply": str(rgb_ply_path),
            "object_memory_json": str(object_json),
            "object_memory_csv": str(object_csv),
            "association_events_csv": str(events_csv),
            "prompt_discovery_csv": str(prompt_path),
        },
        "objects": object_rows,
    }
    summary_path = run.output_dir / "semantic_map_summary.json"
    summary_path.write_text(
        json.dumps(output, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf8",
    )
    _write_copyable(run.output_dir / "copyable_result.txt", output)
    print("V1 PERSISTENT 3-D INSTANCE MEMORY SEMANTIC MAP")
    print(
        f"  frames={len(frames)} sam_tracks={len(payload['sam_track_ids'])} "
        f"objects={len(object_rows)} points={output['saved_map_points']} "
        f"semantic={semantic_points} ready={pipeline_ready}"
    )
    print(f"  ply={ply_path}")
    print(f"  memory={object_json}")
    return summary_path


def _write_binary_ply(
    path: Path,
    *,
    points: torch.Tensor,
    rgb: torch.Tensor,
    object_ids: torch.Tensor,
    sam_track_ids: torch.Tensor,
    confidence: torch.Tensor,
) -> None:
    count = int(points.shape[0])
    array = np.empty(
        count,
        dtype=[
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
            ("semantic_slot", "<i4"),
            ("persistent_object_id", "<i4"),
            # Keep the V0 PLY schema for viewer compatibility.  Full-width
            # SAM IDs remain available in semantic_map.pt and CSV provenance.
            ("sam_track_id", "<i4"),
            ("confidence", "<f4"),
        ],
    )
    xyz = points.detach().float().cpu().numpy()
    colors = rgb.detach().clamp(0.0, 1.0).mul(255).round().byte().cpu().numpy()
    array["x"], array["y"], array["z"] = xyz.T
    array["red"], array["green"], array["blue"] = colors.T
    array["semantic_slot"] = object_ids.detach().int().cpu().numpy()
    array["persistent_object_id"] = object_ids.detach().int().cpu().numpy()
    array["sam_track_id"] = sam_track_ids.detach().int().cpu().numpy()
    array["confidence"] = confidence.detach().float().cpu().numpy()
    header = "\n".join(
        (
            "ply", "format binary_little_endian 1.0", f"element vertex {count}",
            "property float x", "property float y", "property float z",
            "property uchar red", "property uchar green", "property uchar blue",
            "property int semantic_slot",
            "property int persistent_object_id",
            "property int sam_track_id",
            "property float confidence", "end_header", "",
        )
    ).encode("ascii")
    with path.open("wb") as handle:
        handle.write(header)
        array.tofile(handle)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf8")
        return path
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(str(key))
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_copyable(path: Path, summary: Mapping[str, object]) -> None:
    lines = [
        "===== COPYABLE_V1_PERSISTENT_SEMANTIC_MAP_BEGIN =====",
        f"revision={summary['revision']}",
        f"clip={summary['clip']}",
        "identity_mode=v1_object_memory",
        "pose=retrieve_qk",
        "pointmap=raw_full_history_streamvggt_world_pointmap",
        "semantics=sam_observation_to_persistent_3d_object_memory",
        "map_generation_gt_fields=0",
        f"source_sam_track_count={summary['source_sam_track_count']}",
        f"persistent_object_count={summary['persistent_object_count']}",
        f"saved_map_points={summary['saved_map_points']}",
        f"saved_semantic_points={summary['saved_semantic_points']}",
        f"semantic_coverage_percent={summary['semantic_coverage_percent']}",
        f"semantic_map_pipeline_ready={summary['semantic_map_pipeline_ready']}",
        f"artifact={summary['outputs']['artifact']}",
        f"semantic_ply={summary['outputs']['semantic_ply']}",
        f"object_memory_json={summary['outputs']['object_memory_json']}",
        f"object_memory_csv={summary['outputs']['object_memory_csv']}",
        f"association_events_csv={summary['outputs']['association_events_csv']}",
        "===== COPYABLE_V1_PERSISTENT_SEMANTIC_MAP_END =====",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def _load_run(path: str | Path, *, identity_mode: str | None) -> V1Run:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf8") as handle:
        raw = yaml.safe_load(handle) or {}
    section = raw.get("semantic_map", {}) or {}
    mode = str(identity_mode or section.get("identity_mode", "v0_slot"))
    v1 = section.get("v1_object_memory", {}) or {}
    association = v1.get("association", {}) or {}
    association_config = InstanceAssociationConfig(
        center_weight=float(association.get("center_weight", 0.40)),
        voxel_weight=float(association.get("voxel_weight", 0.25)),
        chamfer_weight=float(association.get("chamfer_weight", 0.20)),
        appearance_weight=float(association.get("appearance_weight", 0.10)),
        category_weight=float(association.get("category_weight", 0.05)),
        min_match_score=float(association.get("min_match_score", 0.50)),
        max_center_distance_ratio=float(
            association.get("max_center_distance_ratio", 0.35)
        ),
        center_scale_ratio=float(association.get("center_scale_ratio", 0.12)),
        chamfer_scale_ratio=float(association.get("chamfer_scale_ratio", 0.12)),
        voxel_size_ratio=float(association.get("voxel_size_ratio", 0.02)),
        absolute_voxel_size=float(association.get("absolute_voxel_size", 0.02)),
        max_points_per_comparison=int(
            association.get("max_points_per_comparison", 256)
        ),
        category_hard_gate=bool(association.get("category_hard_gate", True)),
    )
    memory = PersistentObjectMemoryConfig(
        max_points_per_object=int(v1.get("max_points_per_object", 4096)),
        min_observation_points=int(v1.get("min_observation_points", 16)),
        min_mask_pixels=int(v1.get("min_mask_pixels", 32)),
        min_track_score=float(v1.get("min_track_score", 0.50)),
        min_geometry_confidence=float(v1.get("min_geometry_confidence", 0.30)),
        center_ema_alpha=float(v1.get("center_ema_alpha", 0.25)),
        same_frame_merge_score=float(v1.get("same_frame_merge_score", 0.78)),
        association=association_config,
    )
    run = V1Run(
        source_path=source,
        output_dir=expand_storage_path(
            v1.get(
                "output_dir",
                "${VGGT_SAM_STORAGE_ROOT}/outputs/streaming_couping_v1/semantic_map",
            ),
            base=source.parent,
        ),
        identity_mode=mode,
        confidence_threshold=float(section.get("confidence_threshold", 0.30)),
        track_score_threshold=float(section.get("track_score_threshold", 0.50)),
        max_map_points=int(section.get("max_map_points", 400000)),
        memory=memory,
        use_cached_appearance=bool(v1.get("use_cached_appearance", False)),
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
    parser.add_argument("--identity-mode")
    parser.add_argument("--output-dir")
    return parser.parse_args()


if __name__ == "__main__":
    main()
