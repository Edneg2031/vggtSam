#!/usr/bin/env python3
"""Export the frozen V0 QK-pose/raw-pointmap/SAM semantic map."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

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
)
from streaming_couping.src.storage import expand_storage_path


REVISION = "v0_frozen_semantic_map_export_r2"
BASELINE_REVISION = "v0_frozen_semantic_mapping_pipeline_r1"


@dataclass(frozen=True)
class SemanticMapRun:
    source_path: Path
    output_dir: Path
    identity_mode: str
    confidence_threshold: float
    track_score_threshold: float
    max_map_points: int


def main() -> None:
    args = _parse_args()
    data = load_learned_pose_config(args.config)
    baseline = load_baseline_run_config(args.config)
    run = _load_run(args.config)
    if args.output_dir:
        run = replace(run, output_dir=Path(args.output_dir).expanduser().resolve())
    if run.identity_mode != "v0_slot":
        raise ValueError(
            "V0 exporter requires semantic_map.identity_mode=v0_slot; "
            "use commands_semantic_mapping_v1.txt for V1."
        )
    clip = _find_clip(data.clips, baseline.clip_name)
    cache_path_value = cache_path(data, clip)
    payload = load_feature_cache(cache_path_value)
    summary_path = baseline.output_dir / "baseline_summary.json"
    poses_path = baseline.output_dir / "poses.pt"
    summary, poses = _validate_inputs(
        payload=payload,
        summary_path=summary_path,
        poses_path=poses_path,
        clip=clip,
    )
    semantic = build_semantic_pointmap(
        world_points=payload["baseline_world_points"],
        confidence=payload["baseline_world_confidence"],
        masks=payload["tracking_masks_stream"],
        track_scores=payload["tracking_scores"],
        images=payload["stream_images"],
        confidence_threshold=run.confidence_threshold,
        track_score_threshold=run.track_score_threshold,
        max_map_points=run.max_map_points,
    )
    result = _write_outputs(
        payload=payload,
        baseline_summary=summary,
        poses=poses,
        semantic=semantic,
        cache_path_value=cache_path_value,
        poses_path=poses_path,
        run=run,
    )
    print(f"V0 semantic map result={result}")


def _write_outputs(
    *,
    payload: dict[str, Any],
    baseline_summary: dict[str, Any],
    poses: dict[str, Any],
    semantic: SemanticPointMap,
    cache_path_value: Path,
    poses_path: Path,
    run: SemanticMapRun,
) -> Path:
    frames = tuple(int(value) for value in payload["frame_indices"])
    tracks = _track_metadata(payload, semantic)
    slot_to_track = torch.tensor(payload["sam_track_ids"], dtype=torch.long)
    safe_slots = semantic.semantic_slots.clamp_min(0)
    semantic_track_ids = torch.where(
        semantic.semantic_slots >= 0,
        slot_to_track.index_select(0, safe_slots),
        torch.full_like(semantic.semantic_slots, -1),
    )
    semantic_points = int((semantic.semantic_slots >= 0).sum())
    pipeline_ready = int(
        semantic.world_points.shape[0] > 0
        and semantic_points > 0
        and len(tracks) > 0
        and poses["selected_pose_branch"] == "retrieve_qk"
    )
    run.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = run.output_dir / "semantic_map.pt"
    torch.save(
        {
            "schema": 2,
            "revision": REVISION,
            "baseline_revision": BASELINE_REVISION,
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
            "track_metadata": tracks,
            "slot_palette_rgb8": INSTANCE_PALETTE_RGB8,
            "unlabeled_background_rgb8": BACKGROUND_SEMANTIC_RGB8,
            "raw_world_to_camera": poses["raw_world_to_camera"],
            "selected_world_to_camera": poses["selected_world_to_camera"],
            "selected_pose_branch": poses["selected_pose_branch"],
            "pointmap_source": "raw_full_history_streamvggt_world_pointmap",
            "semantic_source": "sam31_persistent_mask_and_id",
            "pose_source": "streamvggt_native_qk_history_retrieval",
        },
        artifact_path,
    )
    ply_path = run.output_dir / "semantic_map.ply"
    _write_binary_ply(
        ply_path,
        points=semantic.world_points,
        rgb=semantic.semantic_rgb,
        slots=semantic.semantic_slots,
        track_ids=semantic_track_ids,
        confidence=semantic.confidence,
    )
    rgb_ply_path = run.output_dir / "rgb_map.ply"
    _write_binary_ply(
        rgb_ply_path,
        points=semantic.world_points,
        rgb=semantic.rgb,
        slots=semantic.semantic_slots,
        track_ids=semantic_track_ids,
        confidence=semantic.confidence,
    )
    tracks_path = run.output_dir / "tracks.csv"
    _write_tracks_csv(tracks_path, tracks)
    prompt_rows = [
        dict(row) for row in payload.get("sam_prompt_diagnostics", ())
    ]
    prompt_path = run.output_dir / "prompt_discovery.csv"
    _write_prompt_csv(prompt_path, prompt_rows)
    output = {
        "schema": 2,
        "revision": REVISION,
        "baseline_version": "v0",
        "baseline_status": "frozen",
        "baseline_revision": BASELINE_REVISION,
        "clip": payload["clip_name"],
        "config": str(run.source_path),
        "cache": str(cache_path_value),
        "poses": str(poses_path),
        "frames": frames,
        "reference_frame": frames[int(payload["reference_sequence_index"])],
        "map_generation_fields": (
            "baseline_world_points",
            "baseline_world_confidence",
            "tracking_masks_stream",
            "tracking_scores",
            "stream_images",
            "sam_track_ids",
            "sam_track_prompts",
            "sam_birth_indices",
        ),
        "map_generation_gt_fields": 0,
        "model_trained": 0,
        "pose_source": "streamvggt_native_qk_history_retrieval",
        "pointmap_source": "raw_full_history_streamvggt_world_pointmap",
        "semantic_source": "sam31_persistent_mask_and_id",
        "prompt_selection_scope": baseline_summary["prompt_selection_scope"],
        "prompt_selection_annotation_gt_used": int(
            baseline_summary["prompt_selection_annotation_gt_used"]
        ),
        "runtime_prompt_selection_gt_fields": int(
            baseline_summary["runtime_prompt_selection_gt_fields"]
        ),
        "semantic_color_policy": "fixed_color_per_persistent_slot_gray_unlabeled",
        "unlabeled_background_rgb8": BACKGROUND_SEMANTIC_RGB8,
        "sam_hidden_features_used": 0,
        "sam_pose_inputs": 0,
        "selected_pose_branch": poses["selected_pose_branch"],
        "pose_improvement_claim": int(baseline_summary["pose_improvement_claim"]),
        "pointmap_modified": 0,
        "geometry_improvement_claim": 0,
        "coordinate_frame": "streamvggt_first_frame_reference_world",
        "confidence_normalization": "per_frame_5_95_quantile",
        "confidence_threshold": run.confidence_threshold,
        "track_score_threshold": run.track_score_threshold,
        "dense_valid_points": int(semantic.dense_valid.sum()),
        "saved_map_points": int(semantic.world_points.shape[0]),
        "saved_semantic_points": semantic_points,
        "semantic_coverage_percent": (
            100.0 * semantic_points / int(semantic.world_points.shape[0])
        ),
        "discovered_track_count": len(tracks),
        "configured_instance_prompts": tuple(payload["instance_prompts"]),
        "prompt_discovery_diagnostics": prompt_rows,
        "tracks": tracks,
        "semantic_map_pipeline_ready": pipeline_ready,
        "claim": "frozen_v0_semantic_mapping_pipeline",
        "outputs": {
            "artifact": str(artifact_path),
            "semantic_ply": str(ply_path),
            "rgb_ply": str(rgb_ply_path),
            "tracks_csv": str(tracks_path),
            "prompt_discovery_csv": str(prompt_path),
        },
    }
    result = run.output_dir / "semantic_map_summary.json"
    result.write_text(
        json.dumps(output, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf8",
    )
    report = run.output_dir / "copyable_result.txt"
    _write_copyable(report, output)
    print("V0 FROZEN SEMANTIC MAPPING PIPELINE")
    print(
        f"  frames={len(frames)} tracks={len(tracks)} "
        f"points={output['saved_map_points']} semantic={semantic_points} "
        f"ready={pipeline_ready}"
    )
    print(f"  ply={ply_path}")
    return result


def _track_metadata(
    payload: dict[str, Any],
    semantic: SemanticPointMap,
) -> list[dict[str, object]]:
    frames = tuple(int(value) for value in payload["frame_indices"])
    output = []
    for slot, (track_id, prompt, birth) in enumerate(
        zip(
            payload["sam_track_ids"],
            payload["sam_track_prompts"],
            payload["sam_birth_indices"],
        )
    ):
        if int(track_id) < 0 or int(birth) < 0:
            continue
        dense = (semantic.dense_semantic_slots == slot) & semantic.dense_valid
        visible = dense.flatten(1).any(dim=1)
        raw_mask = payload["tracking_masks_stream"][:, slot].detach().bool()
        raw_frame_pixels = raw_mask.flatten(1).sum(dim=1)
        raw_visible_pixels = raw_frame_pixels[raw_frame_pixels > 0]
        output.append(
            {
                "slot": slot,
                "sam_track_id": int(track_id),
                "prompt": str(prompt),
                "birth_sequence_index": int(birth),
                "birth_frame": frames[int(birth)],
                "visible_frames": int(visible.sum()),
                "raw_mask_visible_frames": int(
                    (raw_frame_pixels > 0).sum()
                ),
                "raw_mask_pixels_total": int(raw_frame_pixels.sum()),
                "raw_mask_pixels_median_visible": int(
                    raw_visible_pixels.median()
                    if raw_visible_pixels.numel()
                    else 0
                ),
                "raw_mask_pixels_max": int(raw_frame_pixels.max()),
                "dense_owned_valid_points": int(dense.sum()),
                "saved_points": int((semantic.semantic_slots == slot).sum()),
                "color_red": INSTANCE_PALETTE_RGB8[
                    slot % len(INSTANCE_PALETTE_RGB8)
                ][0],
                "color_green": INSTANCE_PALETTE_RGB8[
                    slot % len(INSTANCE_PALETTE_RGB8)
                ][1],
                "color_blue": INSTANCE_PALETTE_RGB8[
                    slot % len(INSTANCE_PALETTE_RGB8)
                ][2],
            }
        )
    return output


def _validate_inputs(
    *,
    payload: dict[str, Any],
    summary_path: Path,
    poses_path: Path,
    clip: ClipConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not summary_path.is_file() or not poses_path.is_file():
        raise FileNotFoundError("Run commands_v0_baseline.txt before map export.")
    summary = json.loads(summary_path.read_text(encoding="utf8"))
    expected = {
        "schema": 6,
        "baseline_status": "frozen",
        "implementation_revision": BASELINE_REVISION,
        "selected_pose_branch": "retrieve_qk",
        "selected_pose_exact_raw": 0,
        "pose_selection_fallback_used": 0,
        "pose_improvement_claim": True,
        "formal_pose_output": "retrieve_qk",
        "formal_pointmap_output": "raw_full_history_world_pointmap",
        "formal_semantic_output": "sam_persistent_id_lifted_raw_world_pointmap",
        "candidate_geometry_available": False,
        "prompt_selection_scope": (
            "this_clip_annotation_assisted_manual_prompt_planning"
        ),
        "prompt_selection_annotation_gt_used": 1,
        "runtime_prompt_selection_gt_fields": 0,
    }
    for name, value in expected.items():
        if summary.get(name) != value:
            raise ValueError(
                f"Baseline summary {name}={summary.get(name)!r}; expected {value!r}."
            )
    required = (
        "baseline_world_points",
        "baseline_world_confidence",
        "tracking_masks_stream",
        "tracking_scores",
        "stream_images",
        "sam_track_ids",
        "sam_track_prompts",
        "sam_birth_indices",
        "instance_prompts",
        "sam_prompt_diagnostics",
        "frame_indices",
        "reference_sequence_index",
        "clip_name",
    )
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError(f"V0 cache lacks semantic-map fields: {missing}.")
    prompt_rows = payload["sam_prompt_diagnostics"]
    expected_prompts = tuple(clip.instance_prompts) or (clip.instance_prompt,)
    if (
        not isinstance(prompt_rows, list)
        or len(prompt_rows) != len(expected_prompts)
        or tuple(str(row.get("prompt", "")) for row in prompt_rows)
        != expected_prompts
    ):
        raise ValueError(
            "V0 cache prompt discovery audit differs from the config; "
            "rebuild it."
        )
    slot_count = int(payload["tracking_masks_stream"].shape[1])
    if any(
        len(payload[name]) != slot_count
        for name in ("sam_track_ids", "sam_track_prompts", "sam_birth_indices")
    ):
        raise ValueError("SAM registry length differs from the mask slot count.")
    frames = tuple(int(value) for value in payload["frame_indices"])
    if frames != clip.frame_indices or payload["clip_name"] != clip.name:
        raise ValueError("V0 semantic-map cache differs from the configured clip.")
    poses = torch.load(poses_path, map_location="cpu", weights_only=False)
    expected_pose_fields = {
        "schema": 1,
        "baseline_revision": BASELINE_REVISION,
        "selected_pose_branch": "retrieve_qk",
        "pointmap_source": "raw_full_history_streamvggt_world_pointmap",
        "candidate_pointmap_used": False,
    }
    for name, value in expected_pose_fields.items():
        if poses.get(name) != value:
            raise ValueError(
                f"V0 pose artifact {name}={poses.get(name)!r}; expected {value!r}."
            )
    if tuple(int(value) for value in poses.get("frame_indices", ())) != frames:
        raise ValueError("V0 pose artifact frame order differs from the cache.")
    raw = poses.get("raw_world_to_camera")
    selected = poses.get("selected_world_to_camera")
    if (
        not torch.is_tensor(raw)
        or not torch.is_tensor(selected)
        or raw.shape != selected.shape
        or not bool(torch.isfinite(raw).all())
        or not bool(torch.isfinite(selected).all())
    ):
        raise ValueError("V0 raw/selected pose tensors are invalid.")
    return summary, poses


def _write_binary_ply(
    path: Path,
    *,
    points: torch.Tensor,
    rgb: torch.Tensor,
    slots: torch.Tensor,
    track_ids: torch.Tensor,
    confidence: torch.Tensor,
) -> None:
    count = int(points.shape[0])
    array = np.empty(
        count,
        dtype=[
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
            ("semantic_slot", "<i4"), ("sam_track_id", "<i4"),
            ("confidence", "<f4"),
        ],
    )
    xyz = points.detach().float().cpu().numpy()
    colors = rgb.detach().clamp(0.0, 1.0).mul(255).round().byte().cpu().numpy()
    array["x"], array["y"], array["z"] = xyz.T
    array["red"], array["green"], array["blue"] = colors.T
    array["semantic_slot"] = slots.detach().int().cpu().numpy()
    array["sam_track_id"] = track_ids.detach().int().cpu().numpy()
    array["confidence"] = confidence.detach().float().cpu().numpy()
    header = "\n".join(
        (
            "ply", "format binary_little_endian 1.0", f"element vertex {count}",
            "property float x", "property float y", "property float z",
            "property uchar red", "property uchar green", "property uchar blue",
            "property int semantic_slot", "property int sam_track_id",
            "property float confidence", "end_header", "",
        )
    ).encode("ascii")
    with path.open("wb") as handle:
        handle.write(header)
        array.tofile(handle)


def _write_tracks_csv(path: Path, tracks: list[dict[str, object]]) -> None:
    if not tracks:
        raise ValueError("Cannot export a semantic map without discovered tracks.")
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(tracks[0]))
        writer.writeheader()
        writer.writerows(tracks)


def _write_prompt_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("Cannot audit semantic discovery without prompt rows.")
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_copyable(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "===== COPYABLE_V0_SEMANTIC_MAP_BEGIN =====",
        f"revision={summary['revision']}",
        f"clip={summary['clip']}",
        f"frames={len(summary['frames'])}",
        "pose=retrieve_qk",
        "pointmap=raw_full_history_streamvggt_world_pointmap",
        "semantics=sam31_persistent_mask_and_id",
        "semantic_colors=fixed_color_per_persistent_slot_gray_unlabeled",
        "model_trained=0",
        "sam_pose_inputs=0",
        "sam_hidden_features_used=0",
        "map_generation_gt_fields=0",
        "prompt_selection_annotation_gt_used=1",
        "runtime_prompt_selection_gt_fields=0",
        f"saved_map_points={summary['saved_map_points']}",
        f"saved_semantic_points={summary['saved_semantic_points']}",
        f"semantic_coverage_percent={summary['semantic_coverage_percent']}",
        f"discovered_track_count={summary['discovered_track_count']}",
        f"semantic_map_pipeline_ready={summary['semantic_map_pipeline_ready']}",
        "",
        "prompt,raw_detections,birth_eligible_tracks,retained_tracks,raw_visible_track_frames,raw_mask_pixels",
    ]
    for row in summary["prompt_discovery_diagnostics"]:
        lines.append(
            ",".join(
                str(row[name])
                for name in (
                    "prompt", "raw_detections", "birth_eligible_tracks",
                    "retained_tracks", "raw_visible_track_frames",
                    "raw_mask_pixels",
                )
            )
        )
    lines.extend(
        [
            "",
            "slot,sam_track_id,prompt,birth_frame,visible_frames,raw_mask_visible_frames,raw_mask_pixels_total,dense_owned_valid_points,saved_points,color_rgb8",
        ]
    )
    for track in summary["tracks"]:
        values = [
            str(track[name])
            for name in (
                "slot", "sam_track_id", "prompt", "birth_frame",
                "visible_frames", "raw_mask_visible_frames",
                "raw_mask_pixels_total", "dense_owned_valid_points",
                "saved_points",
            )
        ]
        values.append(
            f"{track['color_red']} {track['color_green']} {track['color_blue']}"
        )
        lines.append(",".join(values))
    lines.extend(
        [
            "",
            f"artifact={summary['outputs']['artifact']}",
            f"semantic_ply={summary['outputs']['semantic_ply']}",
            f"rgb_ply={summary['outputs']['rgb_ply']}",
            f"tracks_csv={summary['outputs']['tracks_csv']}",
            f"prompt_discovery_csv={summary['outputs']['prompt_discovery_csv']}",
            f"summary={path.with_name('semantic_map_summary.json')}",
            "===== COPYABLE_V0_SEMANTIC_MAP_END =====",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def _load_run(path: str | Path) -> SemanticMapRun:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf8") as handle:
        raw = yaml.safe_load(handle) or {}
    section = raw.get("semantic_map", {})
    run = SemanticMapRun(
        source_path=source,
        output_dir=expand_storage_path(
            section.get("output_dir", "outputs/streaming_couping_v0/semantic_map")
        ),
        identity_mode=str(section.get("identity_mode", "v0_slot")),
        confidence_threshold=float(section.get("confidence_threshold", 0.30)),
        track_score_threshold=float(section.get("track_score_threshold", 0.50)),
        max_map_points=int(section.get("max_map_points", 400000)),
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
