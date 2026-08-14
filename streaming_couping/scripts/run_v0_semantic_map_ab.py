#!/usr/bin/env python3
"""Build and score V0 raw-pose versus QK-pose semantic point maps."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import yaml

from streaming_couping.src.config import load_config
from streaming_couping.src.external_repos import maybe_add_repo_to_path
from streaming_couping.src.learned_pose.baseline_runtime import (
    load_baseline_run_config,
)
from streaming_couping.src.learned_pose.cache import (
    cache_path,
    load_feature_cache,
)
from streaming_couping.src.learned_pose.config import (
    ClipConfig,
    LearnedPoseConfig,
    load_learned_pose_config,
)
from streaming_couping.src.semantic_map import (
    SemanticMapInputs,
    SemanticMapPair,
    apply_similarity,
    build_shared_semantic_maps,
)


REVISION = "v0_semantic_map_shared_raw_geometry_ab_r1"


@dataclass(frozen=True)
class MapRun:
    source_path: Path
    output_dir: Path
    confidence_threshold: float
    track_score_threshold: float
    max_map_points: int
    metric_max_points: int
    metric_thresholds: tuple[float, ...]


def main() -> None:
    args = _parse_args()
    data = load_learned_pose_config(args.config)
    baseline = load_baseline_run_config(args.config)
    run = _load_run(args.config, data=data)
    if args.output_dir:
        run = replace(
            run,
            output_dir=Path(args.output_dir).expanduser().resolve(),
        )
    clip = _find_clip(data, baseline.clip_name)
    path = cache_path(data, clip)
    payload = load_feature_cache(path)
    poses_path = baseline.output_dir / "poses.pt"
    summary_path = baseline.output_dir / "baseline_summary.json"
    _validate_baseline_artifacts(
        payload=payload,
        poses_path=poses_path,
        summary_path=summary_path,
        clip=clip,
    )
    poses = torch.load(poses_path, map_location="cpu", weights_only=False)
    recovery = load_config(data.recovery_config)
    maybe_add_repo_to_path(recovery.streamvggt_repo)
    from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri

    decoded_raw_pose, raw_intrinsics = pose_encoding_to_extri_intri(
        payload["baseline_pose_encoding"].unsqueeze(0).float(),
        image_size_hw=tuple(int(value) for value in payload["image_size"]),
    )
    artifact_raw_pose = poses["raw_world_to_camera"].detach().float().cpu()
    raw_pose_max_abs_difference = float(
        (artifact_raw_pose - decoded_raw_pose.detach().float().cpu()).abs().max()
    )
    if not torch.allclose(
        artifact_raw_pose,
        decoded_raw_pose.detach().float().cpu(),
        atol=2e-5,
        rtol=1e-5,
    ):
        raise RuntimeError(
            "V0 poses.pt raw pose differs from the cache; "
            f"maximum absolute difference={raw_pose_max_abs_difference}."
        )
    generation = _generation_payload(payload, poses, raw_intrinsics[0])
    pair = build_shared_semantic_maps(
        SemanticMapInputs(**generation),
        confidence_threshold=run.confidence_threshold,
        track_score_threshold=run.track_score_threshold,
        max_map_points=run.max_map_points,
    )
    result = _score_and_write(
        payload=payload,
        poses=poses,
        pair=pair,
        cache_path_value=path,
        poses_path=poses_path,
        run=run,
        raw_pose_max_abs_difference=raw_pose_max_abs_difference,
    )
    print(f"V0 semantic map A/B result={result}")


def _generation_payload(
    payload: dict[str, Any],
    poses: dict[str, Any],
    raw_intrinsics: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return only deployable frozen geometry, semantics and selected pose."""

    allowed_cache = {
        "depth": payload["baseline_depth"],
        "confidence": payload["baseline_depth_confidence"],
        "masks": payload["tracking_masks_stream"],
        "track_scores": payload["tracking_scores"],
        "images": payload["stream_images"],
    }
    output = {
        **allowed_cache,
        "intrinsics": raw_intrinsics.detach().float().cpu(),
        "raw_world_to_camera": poses["raw_world_to_camera"].detach().float().cpu(),
        "selected_world_to_camera": poses[
            "selected_world_to_camera"
        ].detach().float().cpu(),
    }
    forbidden = {
        "target_pose_encoding",
        "target_world_to_camera",
        "target_world_points",
        "target_depth",
    }
    if forbidden.intersection(output):
        raise RuntimeError("Semantic map generation contains a GT field.")
    return output


def _score_and_write(
    *,
    payload: dict[str, Any],
    poses: dict[str, Any],
    pair: SemanticMapPair,
    cache_path_value: Path,
    poses_path: Path,
    run: MapRun,
    raw_pose_max_abs_difference: float,
) -> Path:
    """Introduce fixed evaluation Sim(3) and GT only after map generation."""

    scale = float(payload["point_alignment_scale"])
    rotation = payload["point_alignment_rotation"].detach().float().cpu()
    translation = payload["point_alignment_translation"].detach().float().cpu()
    raw_aligned = apply_similarity(
        pair.raw_dense_points,
        scale=scale,
        rotation=rotation,
        translation=translation,
    )
    selected_aligned = apply_similarity(
        pair.selected_dense_points,
        scale=scale,
        rotation=rotation,
        translation=translation,
    )
    target = payload["target_world_points"].detach().float().cpu()
    frames = tuple(int(value) for value in payload["frame_indices"])
    reference = int(payload["reference_sequence_index"])
    per_frame_rows = _per_frame_rows(
        frames=frames,
        reference=reference,
        raw=raw_aligned,
        selected=selected_aligned,
        target=target,
        valid=pair.valid,
        semantic_slots=pair.semantic_slots,
    )
    nonreference = [row for row in per_frame_rows if not row["is_reference"]]
    raw_metrics = _aggregate_rows(nonreference, prefix="raw")
    selected_metrics = _aggregate_rows(nonreference, prefix="selected")
    raw_semantic_metrics = _aggregate_rows(nonreference, prefix="raw_semantic")
    selected_semantic_metrics = _aggregate_rows(
        nonreference, prefix="selected_semantic"
    )
    paired_gain = _gain(
        raw_metrics["point_weighted_rmse"],
        selected_metrics["point_weighted_rmse"],
    )
    semantic_paired_gain = _gain_or_nan(
        raw_semantic_metrics["point_weighted_rmse"],
        selected_semantic_metrics["point_weighted_rmse"],
    )

    target_flat = target.reshape(-1, 3).index_select(
        0, pair.map_flat_indices
    )
    target_valid = torch.isfinite(target_flat).all(dim=-1)
    fused_raw = pair.raw_map_points[target_valid]
    fused_selected = pair.selected_map_points[target_valid]
    fused_target = target_flat[target_valid]
    fused_slots = pair.map_semantic_slots[target_valid]
    raw_fused = _bidirectional_metrics(
        apply_similarity(
            fused_raw,
            scale=scale,
            rotation=rotation,
            translation=translation,
        ),
        fused_target,
        max_points=run.metric_max_points,
        thresholds=run.metric_thresholds,
    )
    selected_fused = _bidirectional_metrics(
        apply_similarity(
            fused_selected,
            scale=scale,
            rotation=rotation,
            translation=translation,
        ),
        fused_target,
        max_points=run.metric_max_points,
        thresholds=run.metric_thresholds,
    )
    semantic_keep = fused_slots >= 0
    raw_semantic_fused = _bidirectional_metrics_or_empty(
        apply_similarity(
            fused_raw[semantic_keep],
            scale=scale,
            rotation=rotation,
            translation=translation,
        ),
        fused_target[semantic_keep],
        max_points=run.metric_max_points,
        thresholds=run.metric_thresholds,
    )
    selected_semantic_fused = _bidirectional_metrics_or_empty(
        apply_similarity(
            fused_selected[semantic_keep],
            scale=scale,
            rotation=rotation,
            translation=translation,
        ),
        fused_target[semantic_keep],
        max_points=run.metric_max_points,
        thresholds=run.metric_thresholds,
    )
    fused_gain = _gain(
        raw_fused["symmetric_mean"],
        selected_fused["symmetric_mean"],
    )
    semantic_fused_gain = _gain_or_nan(
        raw_semantic_fused.get("symmetric_mean", float("nan")),
        selected_semantic_fused.get("symmetric_mean", float("nan")),
    )

    pointmap_aligned = apply_similarity(
        payload["baseline_world_points"].detach().float().cpu(),
        scale=scale,
        rotation=rotation,
        translation=translation,
    )
    depth_pointmap_rmse = _dense_rmse(
        raw_aligned,
        pointmap_aligned,
        pair.valid,
    )
    geometry_pass = int(paired_gain > 0.0 and fused_gain > 0.0)
    semantic_geometry_pass = int(
        semantic_paired_gain > 0.0 and semantic_fused_gain > 0.0
    )
    discovered = _track_metadata(payload)

    run.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(run.output_dir / "frame_metrics.csv", per_frame_rows)
    map_path = run.output_dir / "semantic_maps.pt"
    torch.save(
        {
            "revision": REVISION,
            "clip": payload["clip_name"],
            "frame_indices": frames,
            "coordinate_frame": "streamvggt_native_reference",
            "raw_points": pair.raw_map_points,
            "selected_points": pair.selected_map_points,
            "rgb": pair.map_rgb,
            "semantic_slots": pair.map_semantic_slots,
            "confidence": pair.map_confidence,
            "sequence_indices": pair.map_sequence_indices,
            "track_metadata": discovered,
            "shared_support_exact": True,
            "depth_source": "raw_streamvggt",
            "intrinsics_source": "raw_streamvggt",
            "mask_source": "sam31_tracking_masks_stream",
            "selected_pose_branch": poses["selected_pose_branch"],
        },
        map_path,
    )
    _write_binary_ply(
        run.output_dir / "raw_semantic_map.ply",
        points=pair.raw_map_points,
        rgb=pair.map_rgb,
        slots=pair.map_semantic_slots,
        confidence=pair.map_confidence,
    )
    _write_binary_ply(
        run.output_dir / "selected_semantic_map.ply",
        points=pair.selected_map_points,
        rgb=pair.map_rgb,
        slots=pair.map_semantic_slots,
        confidence=pair.map_confidence,
    )
    summary = {
        "schema": 1,
        "revision": REVISION,
        "baseline_version": "v0",
        "clip": payload["clip_name"],
        "frames": frames,
        "reference_frame": frames[reference],
        "evaluated_frames": tuple(
            frame for index, frame in enumerate(frames) if index != reference
        ),
        "cache": str(cache_path_value),
        "poses": str(poses_path),
        "config": str(run.source_path),
        "map_generation_gt_fields": 0,
        "gt_role": "scoring_only_after_both_maps_are_generated",
        "raw_depth_shared": 1,
        "raw_intrinsics_shared": 1,
        "sam_masks_shared": 1,
        "rgb_shared": 1,
        "confidence_support_shared": 1,
        "shared_support_exact": 1,
        "candidate_pointmap_used": 0,
        "pose_branches": ("raw_streamvggt", "retrieve_qk"),
        "selected_pose_branch": poses["selected_pose_branch"],
        "raw_pose_cache_max_abs_difference": raw_pose_max_abs_difference,
        "map_coordinate_frame": "streamvggt_native_reference",
        "evaluation_alignment": "fixed_raw_reference_sim3",
        "confidence_normalization": "per_frame_5_95_quantile",
        "confidence_threshold": run.confidence_threshold,
        "track_score_threshold": run.track_score_threshold,
        "valid_dense_points": int(pair.valid.sum()),
        "saved_map_points": int(pair.raw_map_points.shape[0]),
        "saved_semantic_points": int((pair.map_semantic_slots >= 0).sum()),
        "discovered_tracks": discovered,
        "raw_depth_backprojection_vs_raw_pointmap_rmse_metric": (
            depth_pointmap_rmse
        ),
        "raw_paired_metrics": raw_metrics,
        "selected_paired_metrics": selected_metrics,
        "paired_rmse_gain_percent": paired_gain,
        "raw_semantic_paired_metrics": raw_semantic_metrics,
        "selected_semantic_paired_metrics": selected_semantic_metrics,
        "semantic_paired_rmse_gain_percent": semantic_paired_gain,
        "raw_fused_metrics": raw_fused,
        "selected_fused_metrics": selected_fused,
        "fused_symmetric_gain_percent": fused_gain,
        "raw_semantic_fused_metrics": raw_semantic_fused,
        "selected_semantic_fused_metrics": selected_semantic_fused,
        "semantic_fused_symmetric_gain_percent": semantic_fused_gain,
        "geometry_map_pass": geometry_pass,
        "semantic_region_geometry_pass": semantic_geometry_pass,
        "claim": (
            "shared_raw_geometry_qk_pose_improves_map_on_single_sequence"
            if geometry_pass
            else "qk_pose_map_improvement_not_established"
        ),
    }
    result = run.output_dir / "semantic_map_summary.json"
    result.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf8",
    )
    _write_copyable(run.output_dir / "copyable_result.txt", summary, run)
    print("V0 SHARED-RAW-GEOMETRY SEMANTIC MAP A/B")
    print(
        f"  points={summary['saved_map_points']} "
        f"semantic={summary['saved_semantic_points']} "
        f"paired_gain={paired_gain:.4f}% fused_gain={fused_gain:.4f}% "
        f"pass={geometry_pass}"
    )
    print(f"  copyable_report={run.output_dir / 'copyable_result.txt'}")
    return result


def _per_frame_rows(
    *,
    frames: tuple[int, ...],
    reference: int,
    raw: torch.Tensor,
    selected: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    semantic_slots: torch.Tensor,
) -> list[dict[str, object]]:
    rows = []
    for index, frame in enumerate(frames):
        support = valid[index] & torch.isfinite(target[index]).all(dim=-1)
        semantic = support & (semantic_slots[index] >= 0)
        raw_rmse, count = _frame_rmse(raw[index], target[index], support)
        selected_rmse, _ = _frame_rmse(
            selected[index], target[index], support
        )
        raw_semantic_rmse, semantic_count = _frame_rmse(
            raw[index], target[index], semantic
        )
        selected_semantic_rmse, _ = _frame_rmse(
            selected[index], target[index], semantic
        )
        rows.append(
            {
                "sequence_index": index,
                "frame_index": frame,
                "is_reference": int(index == reference),
                "support_points": count,
                "semantic_points": semantic_count,
                "raw_rmse": raw_rmse,
                "selected_rmse": selected_rmse,
                "raw_semantic_rmse": raw_semantic_rmse,
                "selected_semantic_rmse": selected_semantic_rmse,
            }
        )
    return rows


def _aggregate_rows(
    rows: Sequence[dict[str, object]], *, prefix: str
) -> dict[str, float | int]:
    key = f"{prefix}_rmse"
    count_key = "semantic_points" if "semantic" in prefix else "support_points"
    valid = [
        row
        for row in rows
        if np.isfinite(float(row[key])) and int(row[count_key]) > 0
    ]
    if not valid:
        return {
            "frames": 0,
            "points": 0,
            "mean_frame_rmse": float("nan"),
            "point_weighted_rmse": float("nan"),
        }
    weights = np.asarray([int(row[count_key]) for row in valid], dtype=np.float64)
    values = np.asarray([float(row[key]) for row in valid], dtype=np.float64)
    return {
        "frames": len(valid),
        "points": int(weights.sum()),
        "mean_frame_rmse": float(values.mean()),
        "point_weighted_rmse": float(
            np.sqrt(np.sum(weights * values**2) / np.sum(weights))
        ),
    }


def _frame_rmse(
    predicted: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[float, int]:
    count = int(valid.sum())
    if count == 0:
        return float("nan"), 0
    distance = torch.linalg.vector_norm(
        predicted[valid] - target[valid], dim=-1
    )
    return float(torch.sqrt(distance.square().mean())), count


def _dense_rmse(
    left: torch.Tensor,
    right: torch.Tensor,
    valid: torch.Tensor,
) -> float:
    support = valid & torch.isfinite(left).all(dim=-1) & torch.isfinite(right).all(dim=-1)
    if not bool(support.any()):
        return float("nan")
    distance = torch.linalg.vector_norm(left[support] - right[support], dim=-1)
    return float(torch.sqrt(distance.square().mean()))


def _bidirectional_metrics_or_empty(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    max_points: int,
    thresholds: tuple[float, ...],
) -> dict[str, float | int]:
    if predicted.shape[0] < 2 or target.shape[0] < 2:
        return {"points": 0, "symmetric_mean": float("nan")}
    return _bidirectional_metrics(
        predicted,
        target,
        max_points=max_points,
        thresholds=thresholds,
    )


def _bidirectional_metrics(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    max_points: int,
    thresholds: tuple[float, ...],
) -> dict[str, float | int]:
    predicted = _deterministic_limit(predicted.float().cpu(), max_points)
    target = _deterministic_limit(target.float().cpu(), max_points)
    accuracy = _nearest_distances(predicted, target)
    completeness = _nearest_distances(target, predicted)
    result: dict[str, float | int] = {
        "predicted_points": int(predicted.shape[0]),
        "target_points": int(target.shape[0]),
        "accuracy_mean": float(accuracy.mean()),
        "completeness_mean": float(completeness.mean()),
        "symmetric_mean": float(0.5 * (accuracy.mean() + completeness.mean())),
    }
    for threshold in thresholds:
        precision = float((accuracy <= float(threshold)).float().mean())
        recall = float((completeness <= float(threshold)).float().mean())
        fscore = 2.0 * precision * recall / max(precision + recall, 1e-12)
        suffix = str(threshold).replace(".", "p")
        result[f"precision_at_{suffix}"] = precision
        result[f"recall_at_{suffix}"] = recall
        result[f"fscore_at_{suffix}"] = fscore
    return result


def _nearest_distances(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    output = []
    for start in range(0, source.shape[0], 512):
        distance = torch.cdist(source[start : start + 512], target)
        output.append(distance.min(dim=1).values)
    return torch.cat(output)


def _deterministic_limit(points: torch.Tensor, limit: int) -> torch.Tensor:
    if points.shape[0] <= int(limit):
        return points
    indices = torch.linspace(
        0, points.shape[0] - 1, steps=int(limit)
    ).round().long()
    return points.index_select(0, indices)


def _gain(raw: float, selected: float) -> float:
    if not np.isfinite(raw) or not np.isfinite(selected) or raw <= 0.0:
        raise ValueError("Map gain requires finite positive raw/selected metrics.")
    return 100.0 * (float(raw) - float(selected)) / float(raw)


def _gain_or_nan(raw: float, selected: float) -> float:
    if not np.isfinite(raw) or not np.isfinite(selected) or raw <= 0.0:
        return float("nan")
    return _gain(raw, selected)


def _track_metadata(payload: dict[str, Any]) -> list[dict[str, object]]:
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
        output.append(
            {
                "slot": slot,
                "sam_track_id": int(track_id),
                "prompt": str(prompt),
                "birth_sequence_index": int(birth),
                "birth_frame": int(payload["frame_indices"][int(birth)]),
            }
        )
    return output


def _write_binary_ply(
    path: Path,
    *,
    points: torch.Tensor,
    rgb: torch.Tensor,
    slots: torch.Tensor,
    confidence: torch.Tensor,
) -> None:
    count = int(points.shape[0])
    array = np.empty(
        count,
        dtype=[
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
            ("semantic_slot", "<i4"), ("confidence", "<f4"),
        ],
    )
    xyz = points.detach().float().cpu().numpy()
    colors = (
        rgb.detach().float().cpu().clamp(0.0, 1.0).mul(255).round().byte().numpy()
    )
    array["x"], array["y"], array["z"] = xyz.T
    array["red"], array["green"], array["blue"] = colors.T
    array["semantic_slot"] = slots.detach().int().cpu().numpy()
    array["confidence"] = confidence.detach().float().cpu().numpy()
    header = "\n".join(
        (
            "ply", "format binary_little_endian 1.0",
            f"element vertex {count}",
            "property float x", "property float y", "property float z",
            "property uchar red", "property uchar green", "property uchar blue",
            "property int semantic_slot", "property float confidence",
            "end_header", "",
        )
    ).encode("ascii")
    with path.open("wb") as handle:
        handle.write(header)
        array.tofile(handle)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty semantic-map CSV.")
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_copyable(path: Path, summary: dict[str, Any], run: MapRun) -> None:
    lines = [
        "===== COPYABLE_V0_SEMANTIC_MAP_AB_BEGIN =====",
        f"revision={REVISION}",
        f"clip={summary['clip']}",
        "branches=raw_streamvggt_pose,retrieve_qk_pose",
        "shared_inputs=raw_depth,raw_intrinsics,sam_masks,rgb,confidence_support",
        "candidate_pointmap_used=0",
        "map_generation_gt_fields=0",
        "gt_role=scoring_only_after_both_maps_are_generated",
        f"saved_map_points={summary['saved_map_points']}",
        f"saved_semantic_points={summary['saved_semantic_points']}",
        "",
        "metric,raw,selected,gain_percent",
        ",".join(
            str(value) for value in (
                "paired_point_weighted_rmse",
                summary["raw_paired_metrics"]["point_weighted_rmse"],
                summary["selected_paired_metrics"]["point_weighted_rmse"],
                summary["paired_rmse_gain_percent"],
            )
        ),
        ",".join(
            str(value) for value in (
                "fused_symmetric_mean",
                summary["raw_fused_metrics"]["symmetric_mean"],
                summary["selected_fused_metrics"]["symmetric_mean"],
                summary["fused_symmetric_gain_percent"],
            )
        ),
        "",
        f"geometry_map_pass={summary['geometry_map_pass']}",
        f"semantic_region_geometry_pass={summary['semantic_region_geometry_pass']}",
        f"claim={summary['claim']}",
        "",
        "outputs:",
        f"summary={run.output_dir / 'semantic_map_summary.json'}",
        f"maps={run.output_dir / 'semantic_maps.pt'}",
        f"raw_ply={run.output_dir / 'raw_semantic_map.ply'}",
        f"selected_ply={run.output_dir / 'selected_semantic_map.ply'}",
        f"frame_csv={run.output_dir / 'frame_metrics.csv'}",
        f"copyable_report={path}",
        "===== COPYABLE_V0_SEMANTIC_MAP_AB_END =====",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def _validate_baseline_artifacts(
    *,
    payload: dict[str, Any],
    poses_path: Path,
    summary_path: Path,
    clip: ClipConfig,
) -> None:
    if not poses_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError("Run commands_v0_baseline.txt before semantic map A/B.")
    summary = json.loads(summary_path.read_text(encoding="utf8"))
    expected = {
        "schema": 5,
        "implementation_revision": "qk_retrieved_pose_semantic_tracking_r5",
        "selected_pose_branch": "retrieve_qk",
        "selected_pose_exact_raw": 0,
        "pose_selection_fallback_used": 0,
        "candidate_pointmap_used": False,
        "depth_source": "raw_streamvggt",
        "intrinsics_source": "raw_streamvggt",
    }
    for name, value in expected.items():
        if summary.get(name) != value:
            raise ValueError(
                f"Baseline summary {name}={summary.get(name)!r}; expected {value!r}."
            )
    required = (
        "baseline_depth", "baseline_depth_confidence", "baseline_pose_encoding",
        "baseline_world_points", "tracking_masks_stream", "tracking_scores",
        "stream_images", "target_world_points", "point_alignment_scale",
        "point_alignment_rotation", "point_alignment_translation",
    )
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError(f"V0 cache lacks semantic-map fields: {missing}.")
    if tuple(int(value) for value in payload["frame_indices"]) != clip.frame_indices:
        raise ValueError("V0 semantic-map cache frame order differs from config.")
    poses = torch.load(poses_path, map_location="cpu", weights_only=False)
    if poses.get("selected_pose_branch") != "retrieve_qk":
        raise ValueError("V0 poses.pt does not select retrieve_qk.")
    if bool(poses.get("selected_pose_exact_raw", True)):
        raise ValueError("V0 selected semantic-map pose is exact raw.")
    if tuple(int(value) for value in poses.get("frame_indices", ())) != clip.frame_indices:
        raise ValueError("V0 poses.pt frame order differs from config.")
    raw = poses.get("raw_world_to_camera")
    selected = poses.get("selected_world_to_camera")
    if not torch.is_tensor(raw) or not torch.is_tensor(selected) or raw.shape != selected.shape:
        raise ValueError("V0 raw/selected pose tensors are invalid.")


def _load_run(path: str | Path, *, data: LearnedPoseConfig) -> MapRun:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf8") as handle:
        raw = yaml.safe_load(handle) or {}
    section = raw.get("semantic_map_ab", {})
    recovery = load_config(data.recovery_config)
    thresholds = tuple(
        float(value)
        for value in section.get(
            "metric_thresholds", recovery.map_metric_thresholds
        )
    )
    run = MapRun(
        source_path=source,
        output_dir=Path(
            section.get(
                "output_dir",
                "outputs/streaming_couping_v0/semantic_map_ab",
            )
        ).expanduser().resolve(),
        confidence_threshold=float(
            section.get(
                "confidence_threshold",
                recovery.point_cloud_confidence_threshold,
            )
        ),
        track_score_threshold=float(section.get("track_score_threshold", 0.5)),
        max_map_points=int(
            section.get("max_map_points", recovery.point_cloud_max_points)
        ),
        metric_max_points=int(
            section.get("metric_max_points", recovery.map_metric_max_points)
        ),
        metric_thresholds=thresholds,
    )
    if not 0.0 <= run.confidence_threshold <= 1.0:
        raise ValueError("semantic_map_ab.confidence_threshold must be in [0,1].")
    if not 0.0 <= run.track_score_threshold <= 1.0:
        raise ValueError("semantic_map_ab.track_score_threshold must be in [0,1].")
    if run.max_map_points < 1 or run.metric_max_points < 2:
        raise ValueError("Semantic-map point limits are invalid.")
    if not run.metric_thresholds or any(value <= 0 for value in run.metric_thresholds):
        raise ValueError("Semantic-map metric thresholds must be positive.")
    return run


def _find_clip(config: LearnedPoseConfig, name: str) -> ClipConfig:
    selected = [clip for clip in config.clips if clip.name == name]
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
