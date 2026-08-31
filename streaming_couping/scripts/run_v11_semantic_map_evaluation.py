#!/usr/bin/env python3
"""Offline evaluation for the V1.1 projection-memory semantic map.

This script opens GT only after the frozen V0 cache and V1.1 artifact exist.
It reports the raw SAM baseline, V1.1 observation-map metrics, and V1.1
voxel-fused object-map metrics separately.  The split is intentional: an
identity improvement and a point-fusion improvement must not be conflated.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import torch

from streaming_couping.src.learned_pose.cache import cache_path, load_feature_cache
from streaming_couping.src.learned_pose.config import ClipConfig, load_learned_pose_config
from streaming_couping.src.learned_pose.baseline_runtime import load_baseline_run_config
from streaming_couping.src.object_memory import collapse_persistent_tracks
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
from streaming_couping.scripts.run_v0_semantic_map import _validate_inputs
from streaming_couping.scripts.run_v11_semantic_map import (
    _find_clip,
    _load_run as _load_v11_run,
)


REVISION = "semantic_map_v11_projection_memory_offline_evaluation_r1"


def main() -> None:
    args = _parse_args()
    data = load_learned_pose_config(args.config)
    baseline = load_baseline_run_config(args.config)
    evaluation_run = _load_evaluation_run(args.config)
    v11_run = _load_v11_run(args.config)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else expand_storage_path(
            "${VGGT_SAM_STORAGE_ROOT}/outputs/streaming_couping_semantic_map_v11",
            base=Path(args.config).expanduser().resolve().parent,
        )
    )
    map_dir = (
        Path(args.map_dir).expanduser().resolve()
        if args.map_dir
        else v11_run.output_dir
    )
    clips = (
        _select_clips(data.clips, args.clip)
        if args.clip
        else (_find_clip(data.clips, baseline.clip_name),)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print("STAGE 1.1 V1.1 PROJECTION-MEMORY OFFLINE EVALUATION")
    print(f"config={Path(args.config).expanduser().resolve()}")
    print(f"map_dir={map_dir}")
    print("GT is opened only for evaluation; frozen cache and model outputs are reused")

    clip_results: list[dict[str, object]] = []
    tracking_rows: list[dict[str, object]] = []
    map_rows: list[dict[str, object]] = []
    fused_object_rows: list[dict[str, object]] = []
    for clip in clips:
        result = _evaluate_clip(
            data=data,
            clip=clip,
            baseline=baseline,
            evaluation_run=evaluation_run,
            map_dir=map_dir,
        )
        clip_results.append(result["clip"])
        tracking_rows.extend(result["tracking_rows"])
        map_rows.extend(result["map_rows"])
        fused_object_rows.extend(result["fused_object_rows"])
        _print_clip_result(result)

    summary: dict[str, object] = {
        "schema": 1,
        "revision": REVISION,
        "candidate_generation_gt_fields": 0,
        "evaluation_gt_fields": 1,
        "identity_mode": "v1_1_projection_temporal_voxel_memory",
        "clip_count": len(clip_results),
        "clips": clip_results,
        "tracking_summary": tracking_rows,
        "map_summary": map_rows,
        "fused_object_summary": fused_object_rows,
        "decision": _decision(tracking_rows, map_rows),
        "outputs": {},
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf8",
    )
    paths = {
        "summary": summary_path,
        "tracking_summary": _write_csv(output_dir / "tracking_summary.csv", tracking_rows),
        "map_summary": _write_csv(output_dir / "map_summary.csv", map_rows),
        "fused_object_summary": _write_csv(
            output_dir / "fused_object_summary.csv", fused_object_rows
        ),
    }
    summary["outputs"] = {name: str(path) for name, path in paths.items()}
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf8",
    )
    copyable = output_dir / "copyable_result.txt"
    _write_copyable(copyable, summary)
    print(f"summary={summary_path}")
    print(f"copyable_result={copyable}")


def _evaluate_clip(
    *,
    data,
    clip: ClipConfig,
    baseline,
    evaluation_run,
    map_dir: Path,
) -> dict[str, object]:
    path = cache_path(data, clip)
    payload = load_feature_cache(path)
    _validate_inputs(
        payload=payload,
        summary_path=baseline.output_dir / "baseline_summary.json",
        poses_path=baseline.output_dir / "poses.pt",
        clip=clip,
    )
    artifact_path = map_dir / "semantic_map.pt"
    if not artifact_path.is_file():
        raise FileNotFoundError(
            f"Missing V1.1 artifact {artifact_path}; run commands_semantic_mapping_v11.txt."
        )
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    if artifact.get("identity_mode") != "v1_1_projection_temporal_voxel_memory":
        raise ValueError(f"Unexpected V1.1 identity mode: {artifact.get('identity_mode')!r}")
    frames = tuple(int(value) for value in payload["frame_indices"])
    if tuple(int(value) for value in artifact.get("frame_indices", ())) != frames:
        raise ValueError("V1.1 artifact frame order differs from the frozen cache.")
    stream_variants = payload["tracking_variant_masks_stream"]
    stream_scores = payload["tracking_variant_scores"]
    raw_masks = _tensor(stream_variants[ evaluation_run.raw_cache_variant ]).bool()
    raw_scores = _tensor(stream_scores[evaluation_run.raw_cache_variant]).float()
    output_variants = payload["tracking_variant_masks_output"]
    raw_output = _tensor(output_variants[evaluation_run.raw_cache_variant]).bool()
    output_scores = payload["tracking_variant_scores"]
    raw_output_scores = _tensor(output_scores[evaluation_run.raw_cache_variant]).float()
    processed_size = tuple(int(value) for value in payload["image_size"])
    ground_truth = load_ground_truth_instances(
        data.manifest,
        scene_id=str(payload["scene_id"]),
        frame_indices=frames,
        output_size=processed_size,
        prompts=tuple(str(value) for value in payload["instance_prompts"]),
        prompt_label_aliases=evaluation_run.prompt_label_aliases,
    )
    ground_truth_output = load_ground_truth_instances(
        data.manifest,
        scene_id=str(payload["scene_id"]),
        frame_indices=frames,
        output_size=(int(raw_output.shape[-2]), int(raw_output.shape[-1])),
        prompts=tuple(str(value) for value in payload["instance_prompts"]),
        prompt_label_aliases=evaluation_run.prompt_label_aliases,
    )
    gt_stream = load_ground_truth_stream_masks(
        data.manifest,
        scene_id=str(payload["scene_id"]),
        frame_indices=frames,
        instance_ids=ground_truth.instance_ids,
        processed_size=processed_size,
        image_mode=str(payload["image_mode"]),
    )

    raw_tracking = evaluate_tracking_variants(
        scene_id=str(payload["scene_id"]),
        clip_name=str(payload["clip_name"]),
        frame_indices=frames,
        variant_masks={"raw_sam": raw_output},
        variant_scores={"raw_sam": raw_output_scores},
        raw_variant="raw_sam",
        track_ids=tuple(int(value) for value in payload["sam_track_ids"]),
        track_prompts=tuple(str(value) for value in payload["sam_track_prompts"]),
        ground_truth=ground_truth_output,
        config=evaluation_run.tracking,
        prompt_label_aliases=evaluation_run.prompt_label_aliases,
    )

    persistent_ids = _tensor(artifact["persistent_object_ids_by_sam_slot"]).long()
    map_write = _tensor(artifact["map_write_mask"]).bool()
    object_rows = artifact.get("object_metadata", ())
    object_ids = tuple(sorted(int(row["object_id"]) for row in object_rows))
    object_prompts = {
        int(row["object_id"]): str(row.get("prompt", row.get("category", "object")))
        for row in object_rows
    }
    persistent = collapse_persistent_tracks(
        masks=raw_output,
        scores=raw_output_scores,
        persistent_object_ids=persistent_ids,
        object_ids=object_ids,
        object_prompts=object_prompts,
    )
    persistent_stream = collapse_persistent_tracks(
        masks=raw_masks,
        scores=raw_scores,
        persistent_object_ids=persistent_ids,
        object_ids=object_ids,
        object_prompts=object_prompts,
    )
    persistent_tracking = evaluate_tracking_variants(
        scene_id=str(payload["scene_id"]),
        clip_name=str(payload["clip_name"]),
        frame_indices=frames,
        variant_masks={"v1_1_projection_memory": persistent["masks"]},
        variant_scores={"v1_1_projection_memory": persistent["scores"]},
        raw_variant="v1_1_projection_memory",
        track_ids=tuple(int(value) for value in persistent["object_ids"]),
        track_prompts=tuple(str(value) for value in persistent["prompts"]),
        ground_truth=ground_truth_output,
        config=evaluation_run.tracking,
        prompt_label_aliases=evaluation_run.prompt_label_aliases,
    )

    aligned_points = apply_similarity(
        payload["baseline_world_points"],
        scale=float(payload["point_alignment_scale"]),
        rotation=payload["point_alignment_rotation"],
        translation=payload["point_alignment_translation"],
    )
    raw_map = evaluate_semantic_object_map(
        scene_id=str(payload["scene_id"]),
        clip_name=str(payload["clip_name"]),
        variant="raw_sam",
        map_policy="all_visible_observations",
        aligned_world_points=aligned_points,
        target_world_points=_tensor(payload["target_world_points"]).float(),
        confidence=_map_confidence(payload["baseline_world_confidence"]),
        predicted_masks=raw_masks,
        track_scores=raw_scores,
        gt_masks=gt_stream,
        gt_instance_ids=ground_truth.instance_ids,
        gt_labels=ground_truth.labels,
        assignments=raw_tracking["assignments"],
        track_ids=tuple(int(value) for value in payload["sam_track_ids"]),
        config=evaluation_run.map_metrics,
    )
    persistent_write = _collapse_object_write_mask(
        map_write,
        persistent_ids,
        tuple(int(value) for value in persistent["object_ids"]),
    )
    observation_map = evaluate_semantic_object_map(
        scene_id=str(payload["scene_id"]),
        clip_name=str(payload["clip_name"]),
        variant="v1_1_projection_memory_observation",
        map_policy="confirmed_observations",
        aligned_world_points=aligned_points,
        target_world_points=_tensor(payload["target_world_points"]).float(),
        confidence=_map_confidence(payload["baseline_world_confidence"]),
        predicted_masks=persistent_stream["masks"],
        track_scores=persistent_stream["scores"],
        gt_masks=gt_stream,
        gt_instance_ids=ground_truth.instance_ids,
        gt_labels=ground_truth.labels,
        assignments=persistent_tracking["assignments"],
        map_write_mask=persistent_write,
        track_ids=tuple(int(value) for value in persistent_stream["object_ids"]),
        config=evaluation_run.map_metrics,
    )
    fused = _evaluate_fused_map(
        artifact=artifact,
        payload=payload,
        ground_truth=ground_truth,
        gt_stream=gt_stream,
        assignments=persistent_tracking["assignments"],
        config=evaluation_run.map_metrics,
        scene_id=str(payload["scene_id"]),
        clip_name=str(payload["clip_name"]),
    )
    tracking_rows = [
        dict(row) for row in raw_tracking["summary_rows"]
    ] + [dict(row) for row in persistent_tracking["summary_rows"]]
    map_rows = [dict(raw_map["summary"]), dict(observation_map["summary"]), dict(fused["summary"])]
    fused_object_rows = [dict(row) for row in fused["object_rows"]]
    return {
        "clip": {
            "scene_id": str(payload["scene_id"]),
            "clip_name": str(payload["clip_name"]),
            "frames": frames,
            "cache": str(path),
            "artifact": str(artifact_path),
            "persistent_object_count": int(len(object_ids)),
            "pending_track_count": int(artifact.get("pending_track_count", -1)),
        },
        "tracking_rows": tracking_rows,
        "map_rows": map_rows,
        "fused_object_rows": fused_object_rows,
    }


def _evaluate_fused_map(
    *,
    artifact: Mapping[str, object],
    payload: Mapping[str, object],
    ground_truth,
    gt_stream: torch.Tensor,
    assignments: Sequence[Mapping[str, object]],
    config,
    scene_id: str,
    clip_name: str,
) -> dict[str, object]:
    fused_points = _tensor(artifact["fused_world_points"]).float()
    fused_ids = _tensor(artifact["fused_object_ids"]).long()
    if tuple(fused_points.shape[-1:]) != (3,) or fused_ids.shape != (fused_points.shape[0],):
        raise ValueError("V1.1 fused map tensors are malformed.")
    aligned_fused = apply_similarity(
        fused_points,
        scale=float(payload["point_alignment_scale"]),
        rotation=payload["point_alignment_rotation"],
        translation=payload["point_alignment_translation"],
    )
    # Unlike the prediction-side fused points, this tensor is the metric GT
    # pointmap already stored by the frozen cache.  Using the aligned
    # prediction pointmap here would make the fused score self-referential.
    target_points = _tensor(payload["target_world_points"]).float()
    object_rows: list[dict[str, object]] = []
    for assignment in assignments:
        object_id = int(assignment["track_id"])
        target_index = int(assignment["gt_index"])
        predicted = aligned_fused[fused_ids == object_id]
        target = target_points[gt_stream[:, target_index]]
        metrics = object_point_metrics(
            predicted,
            target,
            fscore_thresholds=config.fscore_thresholds_m,
            voxel_size=config.voxel_size_m,
            ghost_distance=config.ghost_distance_m,
            chunk_size=config.distance_chunk_size,
        )
        object_rows.append(
            {
                "scene_id": scene_id,
                "clip": clip_name,
                "variant": "v1_1_projection_memory_fused",
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
        "variant": "v1_1_projection_memory_fused",
        "map_policy": "object_level_voxel_fusion",
        "eligible_gt_objects": int(len(ground_truth.instance_ids)),
        "matched_objects": int(len(object_rows)),
        "object_accuracy_m": _mean(object_rows, "object_accuracy_m"),
        "object_completeness_m": _mean(object_rows, "object_completeness_m"),
        "symmetric_distance_m": _mean(object_rows, "symmetric_distance_m"),
        "fscore_5cm": _mean(object_rows, "fscore_5cm"),
        "fscore_10cm": _mean(object_rows, "fscore_10cm"),
        "ghost_point_ratio": _mean(object_rows, "ghost_point_ratio"),
        "voxel_iou_5cm": _mean(object_rows, "voxel_iou"),
        "fused_points": int(fused_points.shape[0]),
        "unmatched_nonempty_tracks": 0,
        "duplicate_objects": 0,
    }
    return {"summary": summary, "object_rows": object_rows}


def _collapse_object_write_mask(
    write_mask: torch.Tensor,
    persistent_ids: torch.Tensor,
    object_ids: Sequence[int],
) -> torch.Tensor:
    writes = write_mask.detach().bool().cpu()
    ids = persistent_ids.detach().long().cpu()
    if tuple(writes.shape) != tuple(ids.shape):
        raise ValueError("V1.1 write mask and persistent IDs have different shapes.")
    output = torch.zeros(writes.shape[0], len(object_ids), dtype=torch.bool)
    for index, object_id in enumerate(object_ids):
        output[:, index] = (writes & (ids == int(object_id))).any(dim=1)
    return output


def _map_confidence(value: object) -> torch.Tensor:
    confidence = _tensor(value).float().cpu()
    if confidence.ndim == 4 and confidence.shape[-1] == 1:
        confidence = confidence[..., 0]
    return confidence


def _tensor(value: object) -> torch.Tensor:
    return value.detach().cpu() if torch.is_tensor(value) else torch.as_tensor(value)


def _select_clips(clips: Sequence[ClipConfig], name: str | None) -> tuple[ClipConfig, ...]:
    if name is None:
        return tuple(clips)
    selected = tuple(clip for clip in clips if clip.name == name)
    if len(selected) != 1:
        raise ValueError(f"Clip {name!r} was not found exactly once.")
    return selected


def _print_clip_result(result: Mapping[str, object]) -> None:
    clip = result["clip"]
    print(
        f"clip={clip['clip_name']} scene={clip['scene_id']} "
        f"persistent_objects={clip['persistent_object_count']} "
        f"pending={clip['pending_track_count']}"
    )
    for row in result["tracking_rows"]:
        print(
            f"  tracking variant={row['variant']} "
            f"IoU={_format(row.get('mean_frame_iou'))} "
            f"frame_IDF1={_format(row.get('frame_idf1'))} "
            f"pixel_IDF1={_format(row.get('pixel_idf1'))} "
            f"fragmentation={row.get('fragmentation_count', 0)} "
            f"merge_errors={row.get('merge_error_count', 0)}"
        )
    for row in result["map_rows"]:
        print(
            f"  map variant={row['variant']} policy={row.get('map_policy', '')} "
            f"voxelIoU5cm={_format(row.get('voxel_iou_5cm'))} "
            f"F5cm={_format(row.get('fscore_5cm'))} "
            f"ghost={_format(row.get('ghost_point_ratio'))}"
        )


def _decision(tracking_rows: Sequence[Mapping[str, object]], map_rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    raw_tracking = _first(tracking_rows, "raw_sam")
    v11_tracking = _first(tracking_rows, "v1_1_projection_memory")
    raw_map = _first(map_rows, "raw_sam")
    observation_map = _first(map_rows, "v1_1_projection_memory_observation")
    fused_map = _first(map_rows, "v1_1_projection_memory_fused")
    return {
        "tracking_v11_frame_idf1_delta_vs_raw": _delta(
            v11_tracking.get("frame_idf1"), raw_tracking.get("frame_idf1")
        ),
        "tracking_v11_mean_iou_delta_vs_raw": _delta(
            v11_tracking.get("mean_frame_iou"), raw_tracking.get("mean_frame_iou")
        ),
        "tracking_v11_fragmentation_count": v11_tracking.get("fragmentation_count", float("nan")),
        "tracking_v11_merge_error_count": v11_tracking.get("merge_error_count", float("nan")),
        "map_v11_observation_voxel_iou_delta_vs_raw": _delta(
            observation_map.get("voxel_iou_5cm"), raw_map.get("voxel_iou_5cm")
        ),
        "map_v11_fused_voxel_iou_delta_vs_raw": _delta(
            fused_map.get("voxel_iou_5cm"), raw_map.get("voxel_iou_5cm")
        ),
        "map_v11_fused_fscore_delta_vs_raw": _delta(
            fused_map.get("fscore_5cm"), raw_map.get("fscore_5cm")
        ),
        "interpretation": (
            "Read observation-map metrics as identity/lifting evidence and "
            "fused-map metrics as the separate voxel-fusion evidence."
        ),
    }


def _first(rows: Sequence[Mapping[str, object]], variant: str) -> Mapping[str, object]:
    for row in rows:
        if row.get("variant") == variant:
            return row
    return {}


def _mean(rows: Sequence[Mapping[str, object]], key: str) -> float:
    values = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return sum(values) / len(values) if values else float("nan")


def _delta(current: object, baseline: object) -> float:
    if not isinstance(current, (int, float)) or not isinstance(baseline, (int, float)):
        return float("nan")
    if not math.isfinite(float(current)) or not math.isfinite(float(baseline)):
        return float("nan")
    return float(current) - float(baseline)


def _format(value: object) -> str:
    return f"{float(value):.4f}" if isinstance(value, (int, float)) and math.isfinite(float(value)) else "nan"


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
    decision = summary["decision"]
    lines = [
        "===== COPYABLE_V1_1_PROJECTION_MEMORY_EVALUATION_BEGIN =====",
        f"revision={summary['revision']}",
        f"clips={summary['clip_count']}",
        "candidate_generation_gt_fields=0",
        "evaluation_gt_fields=1",
        f"tracking_v11_frame_idf1_delta_vs_raw={decision['tracking_v11_frame_idf1_delta_vs_raw']}",
        f"tracking_v11_mean_iou_delta_vs_raw={decision['tracking_v11_mean_iou_delta_vs_raw']}",
        f"tracking_v11_fragmentation_count={decision['tracking_v11_fragmentation_count']}",
        f"tracking_v11_merge_error_count={decision['tracking_v11_merge_error_count']}",
        f"map_v11_observation_voxel_iou_delta_vs_raw={decision['map_v11_observation_voxel_iou_delta_vs_raw']}",
        f"map_v11_fused_voxel_iou_delta_vs_raw={decision['map_v11_fused_voxel_iou_delta_vs_raw']}",
        f"map_v11_fused_fscore_delta_vs_raw={decision['map_v11_fused_fscore_delta_vs_raw']}",
        f"summary={summary['outputs'].get('summary', '')}",
        "===== COPYABLE_V1_1_PROJECTION_MEMORY_EVALUATION_END =====",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="streaming_couping/configs/v0_baseline.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--map-dir")
    parser.add_argument("--clip")
    return parser.parse_args()


if __name__ == "__main__":
    main()
