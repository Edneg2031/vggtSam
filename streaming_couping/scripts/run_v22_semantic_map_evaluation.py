#!/usr/bin/env python3
"""GT-only evaluation for the V2.2 world-space validated recovery map."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from streaming_couping.src.learned_pose.baseline_runtime import (
    load_baseline_run_config,
)
from streaming_couping.src.learned_pose.cache import cache_path, load_feature_cache
from streaming_couping.src.learned_pose.config import ClipConfig, load_learned_pose_config
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

from streaming_couping.scripts.run_semantic_map_evaluation import (
    _load_run as _load_evaluation_run,
)
from streaming_couping.scripts.run_v0_semantic_map import _validate_inputs
from streaming_couping.scripts.run_v2_semantic_map import _find_clip
from streaming_couping.scripts.run_v22_semantic_map import _load_run as _load_v22_run
from streaming_couping.scripts.run_v2_semantic_map_evaluation import (
    _evaluate_recovery_events,
)


REVISION = "semantic_map_v2_2_failure_only_geometry_consistency_offline_evaluation_r1"
V22_VARIANT = "v2_2_failure_only_geometry_consistency_recovery"
V22_IDENTITY = "v2_2_failure_only_geometry_consistency_recovery"


def main() -> None:
    args = _parse_args()
    data = load_learned_pose_config(args.config)
    baseline = load_baseline_run_config(args.config)
    evaluation_run = _load_evaluation_run(args.config)
    v22_run = _load_v22_run(args.config)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else expand_storage_path(
            "${VGGT_SAM_STORAGE_ROOT}/outputs/streaming_couping_semantic_map_v22",
            base=Path(args.config).expanduser().resolve().parent,
        )
    )
    map_dir = (
        Path(args.map_dir).expanduser().resolve()
        if args.map_dir
        else v22_run.output_dir
    )
    clips = (
        (_find_clip(data.clips, args.clip),)
        if args.clip
        else (_find_clip(data.clips, baseline.clip_name),)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print("STAGE V2.2 WORLD-SPACE GEOMETRY-CONSISTENCY OFFLINE EVALUATION")
    print(f"config={Path(args.config).expanduser().resolve()}")
    print(f"map_dir={map_dir}")
    print("GT is opened only for evaluation; V0 cache and V2.2 masks are frozen")

    clip_results: list[dict[str, Any]] = []
    tracking_rows: list[dict[str, Any]] = []
    map_rows: list[dict[str, Any]] = []
    recovery_rows: list[dict[str, Any]] = []
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
        recovery_rows.extend(result["recovery_rows"])
        _print_clip_result(result)

    summary: dict[str, Any] = {
        "schema": 1,
        "revision": REVISION,
        "candidate_generation_gt_fields": 0,
        "evaluation_gt_fields": 1,
        "identity_mode": V22_IDENTITY,
        "clip_count": len(clip_results),
        "clips": clip_results,
        "tracking_summary": tracking_rows,
        "map_summary": map_rows,
        "recovery_evaluation": recovery_rows,
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
        "recovery_evaluation": _write_csv(
            output_dir / "recovery_evaluation.csv", recovery_rows
        ),
    }
    summary["outputs"] = {name: str(path) for name, path in paths.items()}
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf8",
    )
    # Keep the requested V2.2-named evaluation summary alongside the generic
    # summary.json used by the command wrapper.
    named_summary_path = output_dir / "semantic_map_v22_summary.json"
    named_summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf8",
    )
    summary["outputs"]["semantic_map_v22_summary"] = str(named_summary_path)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf8",
    )
    named_summary_path.write_text(
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
) -> dict[str, Any]:
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
            f"Missing V2.2 artifact {artifact_path}; run commands_semantic_mapping_v22.txt."
        )
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    if artifact.get("identity_mode") != V22_IDENTITY:
        raise ValueError(
            f"Unexpected V2.2 identity mode: {artifact.get('identity_mode')!r}"
        )
    frames = tuple(int(value) for value in payload["frame_indices"])
    if tuple(int(value) for value in artifact.get("frame_indices", ())) != frames:
        raise ValueError("V2.2 artifact frame order differs from the frozen cache.")

    raw_variant = str(artifact.get("raw_cache_variant", "sam31_online_forward"))
    raw_output = _tensor(payload["tracking_variant_masks_output"][raw_variant]).bool()
    raw_stream = _tensor(payload["tracking_variant_masks_stream"][raw_variant]).bool()
    raw_scores = _tensor(payload["tracking_variant_scores"][raw_variant]).float()
    v22_output = _tensor(artifact["tracking_masks_output"]).bool()
    v22_stream = _tensor(artifact["tracking_masks_stream"]).bool()
    v22_scores = _tensor(artifact["tracking_scores"]).float()
    if tuple(v22_output.shape) != tuple(raw_output.shape):
        raise ValueError("V2.2 output masks do not match raw cache shape.")
    if tuple(v22_stream.shape) != tuple(raw_stream.shape):
        raise ValueError("V2.2 stream masks do not match raw cache shape.")

    processed_size = tuple(int(value) for value in payload["image_size"])
    ground_truth = load_ground_truth_instances(
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
    tracking = evaluate_tracking_variants(
        scene_id=str(payload["scene_id"]),
        clip_name=str(payload["clip_name"]),
        frame_indices=frames,
        variant_masks={"raw_sam": raw_output, V22_VARIANT: v22_output},
        variant_scores={"raw_sam": raw_scores, V22_VARIANT: v22_scores},
        raw_variant="raw_sam",
        track_ids=tuple(int(value) for value in payload["sam_track_ids"]),
        track_prompts=tuple(str(value) for value in payload["sam_track_prompts"]),
        ground_truth=ground_truth,
        config=evaluation_run.tracking,
        prompt_label_aliases=evaluation_run.prompt_label_aliases,
    )
    aligned_points = apply_similarity(
        payload["baseline_world_points"],
        scale=float(payload["point_alignment_scale"]),
        rotation=payload["point_alignment_rotation"],
        translation=payload["point_alignment_translation"],
    )
    confidence = normalize_confidence(payload["baseline_world_confidence"])
    common_map = dict(
        scene_id=str(payload["scene_id"]),
        clip_name=str(payload["clip_name"]),
        aligned_world_points=aligned_points,
        target_world_points=_tensor(payload["target_world_points"]).float(),
        confidence=confidence,
        gt_masks=gt_stream,
        gt_instance_ids=ground_truth.instance_ids,
        gt_labels=ground_truth.labels,
        assignments=tracking["assignments"],
        track_ids=tuple(int(value) for value in payload["sam_track_ids"]),
        config=evaluation_run.map_metrics,
    )
    raw_map = evaluate_semantic_object_map(
        variant="raw_sam",
        map_policy="all_visible_observations",
        predicted_masks=raw_stream,
        track_scores=raw_scores,
        **common_map,
    )
    v22_map = evaluate_semantic_object_map(
        variant=V22_VARIANT,
        map_policy="all_visible_observations_after_world_geometry_validation",
        predicted_masks=v22_stream,
        track_scores=v22_scores,
        **common_map,
    )
    recovery_rows, recovery_summary = _evaluate_recovery_events(
        events=artifact.get("recovery_events", ()),
        raw_masks=raw_output,
        v2_masks=v22_output,
        ground_truth=ground_truth,
        assignments=tracking["assignments"],
        threshold=float(evaluation_run.tracking.reentry_iou_threshold),
    )
    stats = artifact.get("recovery_stats", {})
    tracking_rows = [dict(row) for row in tracking["summary_rows"]]
    map_rows = [dict(raw_map["summary"]), dict(v22_map["summary"])]
    raw_tracking = _row(tracking_rows, "raw_sam")
    v22_tracking = _row(tracking_rows, V22_VARIANT)
    raw_map_row = map_rows[0]
    v22_map_row = map_rows[1]
    clip_result = {
        "scene_id": str(payload["scene_id"]),
        "clip_name": str(payload["clip_name"]),
        "frames": frames,
        "cache": str(path),
        "artifact": str(artifact_path),
        "geometry_validation_accept_count": int(
            stats.get("geometry_validation_accept_count", 0)
        ),
        "geometry_validation_reject_count": int(
            stats.get("geometry_validation_reject_count", 0)
        ),
        "v22_geometry_consistency_accept_rate": float(
            stats.get("geometry_validation_accept_rate", 0.0)
        ),
        "geometry_validation_reject_reasons": stats.get(
            "geometry_validation_reject_reasons", {}
        ),
        "recovery_trigger_count": int(stats.get("recovery_trigger_count", 0)),
        "accepted_recovery_count": int(stats.get("accepted_recovery_count", 0)),
        "raw_fallback_unchanged_ratio": float(
            stats.get("raw_fallback_unchanged_ratio", 1.0)
        ),
        "normal_frame_unchanged_ratio": float(
            stats.get("normal_frame_unchanged_ratio", 1.0)
        ),
        "recovery_pre_IoU": recovery_summary["recovery_pre_iou_mean"],
        "recovery_post_IoU": recovery_summary["recovery_post_iou_mean"],
        "recovery_iou_improved_ratio": recovery_summary[
            "recovery_iou_improved_ratio"
        ],
        "reentry_events": recovery_summary["reentry_events"],
        "reentry_successes": recovery_summary["reentry_successes"],
        "reentry_success_rate": recovery_summary["reentry_success_rate"],
        "raw_mean_frame_iou": raw_tracking["mean_frame_iou"],
        "v22_mean_frame_iou": v22_tracking["mean_frame_iou"],
        "raw_frame_idf1": raw_tracking["frame_idf1"],
        "v22_frame_idf1": v22_tracking["frame_idf1"],
        "raw_voxel_iou_5cm": raw_map_row["voxel_iou_5cm"],
        "v22_voxel_iou_5cm": v22_map_row["voxel_iou_5cm"],
        # Preserve the spelling used in the experiment protocol in addition
        # to the snake_case field used by the metric implementation.
        "v22_voxelIoU_5cm": v22_map_row["voxel_iou_5cm"],
        "v22_ghost_rate": v22_map_row["ghost_point_ratio"],
        "raw_ghost_rate": raw_map_row["ghost_point_ratio"],
        "raw_fscore_5cm": raw_map_row["fscore_5cm"],
        "v22_fscore_5cm": v22_map_row["fscore_5cm"],
    }
    return {
        "clip": clip_result,
        "tracking_rows": tracking_rows,
        "map_rows": map_rows,
        "recovery_rows": recovery_rows,
    }


def _print_clip_result(result: Mapping[str, Any]) -> None:
    clip = result["clip"]
    tracking = {row["variant"]: row for row in result["tracking_rows"]}
    maps = {row["variant"]: row for row in result["map_rows"]}
    raw = tracking["raw_sam"]
    v22 = tracking[V22_VARIANT]
    raw_map = maps["raw_sam"]
    v22_map = maps[V22_VARIANT]
    print(
        f"clip={clip['clip_name']} scene={clip['scene_id']} "
        f"geometry_accept={clip['geometry_validation_accept_count']} "
        f"geometry_reject={clip['geometry_validation_reject_count']} "
        f"accept_rate={clip['v22_geometry_consistency_accept_rate']:.4f}"
    )
    print(f"  reject_reasons={clip['geometry_validation_reject_reasons']}")
    print(
        f"  tracking raw_sam IoU={raw['mean_frame_iou']:.4f} "
        f"frame_IDF1={raw['frame_idf1']:.4f} "
        f"reentry={raw['reentry_successes']}/{raw['reentry_events']}"
    )
    print(
        f"  tracking {V22_VARIANT} IoU={v22['mean_frame_iou']:.4f} "
        f"frame_IDF1={v22['frame_idf1']:.4f} "
        f"reentry={v22['reentry_successes']}/{v22['reentry_events']}"
    )
    print(
        f"  map raw_sam voxelIoU5cm={raw_map['voxel_iou_5cm']:.4f} "
        f"F5cm={raw_map['fscore_5cm']:.4f} ghost={raw_map['ghost_point_ratio']:.4f}"
    )
    print(
        f"  map {V22_VARIANT} voxelIoU5cm={v22_map['voxel_iou_5cm']:.4f} "
        f"F5cm={v22_map['fscore_5cm']:.4f} ghost={v22_map['ghost_point_ratio']:.4f}"
    )
    print(
        f"  recovery pre_IoU={clip['recovery_pre_IoU']:.4f} "
        f"post_IoU={clip['recovery_post_IoU']:.4f} "
        f"improved_ratio={clip['recovery_iou_improved_ratio']:.4f}"
    )
    print(
        f"  raw_test_equivalent=sealed_clip; decision="
        f"{_decision([raw, v22], [raw_map, v22_map])}"
    )


def _decision(
    tracking_rows: Sequence[Mapping[str, Any]],
    map_rows: Sequence[Mapping[str, Any]],
) -> str:
    by_tracking = {str(row["variant"]): row for row in tracking_rows}
    by_map = {str(row["variant"]): row for row in map_rows}
    if "raw_sam" not in by_tracking or V22_VARIANT not in by_tracking:
        return "NO_GO"
    raw = by_tracking["raw_sam"]
    v22 = by_tracking[V22_VARIANT]
    raw_map = by_map.get("raw_sam", {})
    v22_map = by_map.get(V22_VARIANT, {})
    checks = (
        _finite_ge(v22.get("mean_frame_iou"), raw.get("mean_frame_iou")),
        _finite_ge(v22.get("frame_idf1"), raw.get("frame_idf1")),
        _finite_ge(v22_map.get("voxel_iou_5cm"), raw_map.get("voxel_iou_5cm")),
        _finite_ge(v22_map.get("fscore_5cm"), raw_map.get("fscore_5cm")),
    )
    return "GO" if all(checks) else "NO_GO"


def _finite_ge(candidate: Any, baseline: Any) -> bool:
    try:
        return (
            math.isfinite(float(candidate))
            and math.isfinite(float(baseline))
            and float(candidate) >= float(baseline) - 1e-6
        )
    except (TypeError, ValueError):
        return False


def _row(rows: Sequence[Mapping[str, Any]], variant: str) -> Mapping[str, Any]:
    for row in rows:
        if str(row.get("variant")) == variant:
            return row
    raise KeyError(f"Missing evaluation row {variant!r}.")


def _tensor(value: Any) -> torch.Tensor:
    if not torch.is_tensor(value):
        return torch.as_tensor(value)
    return value.detach().cpu()


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if str(key) not in keys:
                keys.append(str(key))
    if not keys:
        keys = ["empty"]
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in keys})
    return path


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True, default=str)
    return value


def _write_copyable(path: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "===== COPYABLE_V2_2_FAILURE_ONLY_GEOMETRY_CONSISTENCY_EVALUATION_BEGIN =====",
        f"revision={summary['revision']}",
        f"clip_count={summary['clip_count']}",
        f"identity_mode={V22_IDENTITY}",
        "assignment=raw_sam_frozen",
        "gt_role=evaluation_only",
    ]
    for clip in summary["clips"]:
        lines.extend(
            [
                f"clip={clip['clip_name']}",
                f"geometry_validation_accept_count={clip['geometry_validation_accept_count']}",
                f"geometry_validation_reject_count={clip['geometry_validation_reject_count']}",
                f"v22_geometry_consistency_accept_rate={clip['v22_geometry_consistency_accept_rate']}",
                f"geometry_validation_reject_reasons={clip['geometry_validation_reject_reasons']}",
                f"recovery_pre_IoU={clip['recovery_pre_IoU']}",
                f"recovery_post_IoU={clip['recovery_post_IoU']}",
                f"raw_mean_frame_iou={clip['raw_mean_frame_iou']}",
                f"v22_mean_frame_iou={clip['v22_mean_frame_iou']}",
                f"raw_frame_idf1={clip['raw_frame_idf1']}",
                f"v22_frame_idf1={clip['v22_frame_idf1']}",
                f"raw_voxel_iou_5cm={clip['raw_voxel_iou_5cm']}",
                f"v22_voxel_iou_5cm={clip['v22_voxel_iou_5cm']}",
                f"v22_voxelIoU_5cm={clip['v22_voxelIoU_5cm']}",
                f"v22_ghost_rate={clip['v22_ghost_rate']}",
                f"v22_fscore_5cm={clip['v22_fscore_5cm']}",
                f"reentry_success_rate={clip['reentry_success_rate']}",
            ]
        )
    lines.extend(
        [
            f"decision={summary['decision']}",
            "===== COPYABLE_V2_2_FAILURE_ONLY_GEOMETRY_CONSISTENCY_EVALUATION_END =====",
        ]
    )
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
