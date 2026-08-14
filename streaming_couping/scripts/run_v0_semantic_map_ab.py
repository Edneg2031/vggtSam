#!/usr/bin/env python3
"""Score raw, pose-only QK and joint-QK semantic point maps."""

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
from streaming_couping.src.pointmap_alignment import _robust_similarity
from streaming_couping.src.semantic_map import (
    SemanticMapInputs,
    SemanticMapPair,
    apply_similarity,
    backproject_depth,
    build_shared_semantic_maps,
    camera_to_world,
)


REVISION = "v0_semantic_map_qk_joint_geometry_ab_r3"


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
    joint_path = baseline.qk_pose_output
    _validate_baseline_artifacts(
        payload=payload,
        poses_path=poses_path,
        summary_path=summary_path,
        joint_path=joint_path,
        clip=clip,
    )
    poses = torch.load(poses_path, map_location="cpu", weights_only=False)
    joint = torch.load(joint_path, map_location="cpu", weights_only=False)
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
        joint=joint,
        pair=pair,
        cache_path_value=path,
        poses_path=poses_path,
        joint_path=joint_path,
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
    joint: dict[str, Any],
    pair: SemanticMapPair,
    cache_path_value: Path,
    poses_path: Path,
    joint_path: Path,
    run: MapRun,
    raw_pose_max_abs_difference: float,
) -> Path:
    """Introduce two fixed reference alignments and GT after map generation."""

    target = payload["target_world_points"].detach().float().cpu()
    frames = tuple(int(value) for value in payload["frame_indices"])
    reference = int(payload["reference_sequence_index"])
    joint_depth = joint["selected_depth"].detach().float().cpu()[..., 0]
    joint_intrinsics = joint["selected_intrinsics"].detach().float().cpu()
    joint_pose = joint["selected_world_to_camera"].detach().float().cpu()
    joint_depth_points = camera_to_world(
        backproject_depth(joint_depth, joint_intrinsics),
        joint_pose,
    )
    raw_pointmap = payload["baseline_world_points"].detach().float().cpu()
    joint_pointmap = joint["selected_pointmap"].detach().float().cpu()
    _validate_dense_branch_shapes(
        target,
        pair.raw_dense_points,
        pair.selected_dense_points,
        joint_depth_points,
        raw_pointmap,
        joint_pointmap,
    )

    native_families = {
        "depth_backprojection": {
            "raw_depth_pose": pair.raw_dense_points,
            "qk_pose_raw_depth": pair.selected_dense_points,
            "qk_joint_depth_pose": joint_depth_points,
        },
        "point_head": {
            "raw_pointmap": raw_pointmap,
            "qk_joint_pointmap": joint_pointmap,
        },
    }
    family_raw = {
        "depth_backprojection": "raw_depth_pose",
        "point_head": "raw_pointmap",
    }
    aligned_families: dict[str, dict[str, torch.Tensor]] = {}
    alignments: dict[str, dict[str, object]] = {}
    for family, branches in native_families.items():
        raw_branch = family_raw[family]
        scale, rotation, translation, inliers, rmse = (
            _fit_raw_depth_reference_similarity(
                branches[raw_branch][reference],
                target[reference],
                pair.valid[reference],
            )
        )
        aligned_families[family] = {
            name: apply_similarity(
                points,
                scale=scale,
                rotation=rotation,
                translation=translation,
            )
            for name, points in branches.items()
        }
        alignments[family] = {
            "source_branch": raw_branch,
            "reference_frame": frames[reference],
            "inliers": inliers,
            "fit_rmse_metric": rmse,
            "fixed_for_all_family_branches": 1,
        }

    branch_metrics: dict[str, dict[str, object]] = {}
    frame_rows: list[dict[str, object]] = []
    native_maps: dict[str, torch.Tensor] = {}
    for family, aligned in aligned_families.items():
        metrics, rows, _ = _score_family(
            family=family,
            branches=aligned,
            raw_branch=family_raw[family],
            frames=frames,
            reference=reference,
            target=target,
            valid=pair.valid,
            semantic_slots=pair.semantic_slots,
            map_flat_indices=pair.map_flat_indices,
            map_semantic_slots=pair.map_semantic_slots,
            max_points=run.metric_max_points,
            thresholds=run.metric_thresholds,
        )
        branch_metrics.update(metrics)
        frame_rows.extend(rows)
        native_maps.update(
            {
                name: native_families[family][name].reshape(-1, 3).index_select(
                    0, pair.map_flat_indices
                )
                for name in native_families[family]
            }
        )

    discovered = _track_metadata(payload)
    instance_rows = _per_instance_rows(
        families=aligned_families,
        family_raw=family_raw,
        tracks=discovered,
        frames=frames,
        reference=reference,
        target=target,
        valid=pair.valid,
        semantic_slots=pair.semantic_slots,
        map_flat_indices=pair.map_flat_indices,
        map_semantic_slots=pair.map_semantic_slots,
        max_points=run.metric_max_points,
        thresholds=run.metric_thresholds,
    )

    pose_only = branch_metrics["qk_pose_raw_depth"]
    joint_depth_metrics = branch_metrics["qk_joint_depth_pose"]
    joint_point_metrics = branch_metrics["qk_joint_pointmap"]
    joint_depth_pass = int(
        joint_depth_metrics["overall_pass"]
        and joint_depth_metrics["semantic_pass"]
    )
    joint_point_pass = int(
        joint_point_metrics["overall_pass"]
        and joint_point_metrics["semantic_pass"]
    )
    joint_any_pass = int(joint_depth_pass or joint_point_pass)

    run.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(run.output_dir / "frame_metrics.csv", frame_rows)
    _write_csv(run.output_dir / "instance_metrics.csv", instance_rows)
    map_path = run.output_dir / "semantic_maps.pt"
    torch.save(
        {
            "revision": REVISION,
            "clip": payload["clip_name"],
            "frame_indices": frames,
            "coordinate_frame": "streamvggt_native_reference",
            "native_branch_points": native_maps,
            "rgb": pair.map_rgb,
            "semantic_slots": pair.map_semantic_slots,
            "confidence": pair.map_confidence,
            "sequence_indices": pair.map_sequence_indices,
            "track_metadata": discovered,
            "shared_pixel_support_exact": True,
            "mask_source": "sam31_tracking_masks_stream",
            "qk_geometry_source": str(joint_path),
        },
        map_path,
    )
    ply_names = {
        "raw_depth_pose": "raw_semantic_map.ply",
        "qk_pose_raw_depth": "qk_pose_raw_depth_semantic_map.ply",
        "qk_joint_depth_pose": "qk_joint_depth_semantic_map.ply",
        "raw_pointmap": "raw_pointmap_semantic_map.ply",
        "qk_joint_pointmap": "qk_joint_pointmap_semantic_map.ply",
    }
    for branch, filename in ply_names.items():
        _write_binary_ply(
            run.output_dir / filename,
            points=native_maps[branch],
            rgb=pair.map_rgb,
            slots=pair.map_semantic_slots,
            confidence=pair.map_confidence,
        )

    summary = {
        "schema": 2,
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
        "qk_joint_geometry": str(joint_path),
        "config": str(run.source_path),
        "map_generation_gt_fields": 0,
        "gt_role": "scoring_only_after_all_native_maps_are_generated",
        "sam_masks_shared": 1,
        "rgb_shared": 1,
        "pixel_support_source": "raw_streamvggt_confidence",
        "shared_pixel_support_exact": 1,
        "candidate_depth_used": 1,
        "candidate_pointmap_used": 1,
        "selected_pose_branch": poses["selected_pose_branch"],
        "raw_pose_cache_max_abs_difference": raw_pose_max_abs_difference,
        "map_coordinate_frame": "streamvggt_native_reference",
        "evaluation_alignment": "fixed_family_raw_reference_sim3",
        "alignments": alignments,
        "confidence_normalization": "raw_per_frame_5_95_quantile",
        "confidence_threshold": run.confidence_threshold,
        "track_score_threshold": run.track_score_threshold,
        "valid_dense_points": int(pair.valid.sum()),
        "saved_map_points": int(pair.raw_map_points.shape[0]),
        "saved_semantic_points": int((pair.map_semantic_slots >= 0).sum()),
        "discovered_tracks": discovered,
        "branch_metrics": branch_metrics,
        "instance_metric_rows": len(instance_rows),
        "instance_metrics": instance_rows,
        "pose_only_raw_depth_pass": int(
            pose_only["overall_pass"] and pose_only["semantic_pass"]
        ),
        "qk_joint_depth_map_pass": joint_depth_pass,
        "qk_joint_pointmap_pass": joint_point_pass,
        "qk_joint_any_semantic_map_pass": joint_any_pass,
        "claim": (
            "qk_joint_geometry_improves_semantic_map_on_single_sequence"
            if joint_any_pass
            else "qk_joint_geometry_map_improvement_not_established"
        ),
    }
    result = run.output_dir / "semantic_map_summary.json"
    result.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf8",
    )
    _write_copyable(run.output_dir / "copyable_result.txt", summary, run)
    print("V0 QK JOINT-GEOMETRY SEMANTIC MAP A/B")
    print(
        f"  points={summary['saved_map_points']} "
        f"semantic={summary['saved_semantic_points']} "
        f"joint_depth_pass={joint_depth_pass} "
        f"joint_pointmap_pass={joint_point_pass}"
    )
    print(f"  copyable_report={run.output_dir / 'copyable_result.txt'}")
    return result


def _validate_dense_branch_shapes(
    target: torch.Tensor,
    *branches: torch.Tensor,
) -> None:
    expected = tuple(target.shape)
    for branch in branches:
        if tuple(branch.shape) != expected:
            raise ValueError(
                f"Geometry branch shape {tuple(branch.shape)} differs from "
                f"target shape {expected}."
            )
        if not bool(torch.isfinite(branch).all()):
            raise ValueError("Geometry branch contains non-finite points.")


def _score_family(
    *,
    family: str,
    branches: dict[str, torch.Tensor],
    raw_branch: str,
    frames: tuple[int, ...],
    reference: int,
    target: torch.Tensor,
    valid: torch.Tensor,
    semantic_slots: torch.Tensor,
    map_flat_indices: torch.Tensor,
    map_semantic_slots: torch.Tensor,
    max_points: int,
    thresholds: tuple[float, ...],
) -> tuple[
    dict[str, dict[str, object]],
    list[dict[str, object]],
    dict[str, torch.Tensor],
]:
    target_flat = target.reshape(-1, 3).index_select(0, map_flat_indices)
    target_valid = torch.isfinite(target_flat).all(dim=-1)
    fused_target = target_flat[target_valid]
    fused_slots = map_semantic_slots[target_valid]
    rows: list[dict[str, object]] = []
    maps: dict[str, torch.Tensor] = {}
    metrics: dict[str, dict[str, object]] = {}
    for branch, points in branches.items():
        branch_rows = []
        for index, frame in enumerate(frames):
            support = valid[index] & torch.isfinite(target[index]).all(dim=-1)
            semantic = support & (semantic_slots[index] >= 0)
            rmse, count = _frame_rmse(points[index], target[index], support)
            semantic_rmse, semantic_count = _frame_rmse(
                points[index], target[index], semantic
            )
            row = {
                "family": family,
                "branch": branch,
                "sequence_index": index,
                "frame_index": frame,
                "is_reference": int(index == reference),
                "support_points": count,
                "semantic_points": semantic_count,
                "paired_rmse": rmse,
                "semantic_paired_rmse": semantic_rmse,
            }
            branch_rows.append(row)
            rows.append(row)
        nonreference = [row for row in branch_rows if not row["is_reference"]]
        paired = _aggregate_metric_rows(
            nonreference,
            value_key="paired_rmse",
            count_key="support_points",
        )
        semantic_paired = _aggregate_metric_rows(
            nonreference,
            value_key="semantic_paired_rmse",
            count_key="semantic_points",
        )
        branch_map = points.reshape(-1, 3).index_select(0, map_flat_indices)
        branch_map = branch_map[target_valid]
        maps[branch] = branch_map
        fused = _bidirectional_metrics(
            branch_map,
            fused_target,
            max_points=max_points,
            thresholds=thresholds,
        )
        semantic_keep = fused_slots >= 0
        semantic_fused = _bidirectional_metrics_or_empty(
            branch_map[semantic_keep],
            fused_target[semantic_keep],
            max_points=max_points,
            thresholds=thresholds,
        )
        metrics[branch] = {
            "family": family,
            "raw_reference_branch": raw_branch,
            "paired": paired,
            "semantic_paired": semantic_paired,
            "fused": fused,
            "semantic_fused": semantic_fused,
        }

    raw = metrics[raw_branch]
    for branch, current in metrics.items():
        if branch == raw_branch:
            paired_gain = fused_gain = 0.0
            semantic_paired_gain = semantic_fused_gain = 0.0
        else:
            paired_gain = _gain_or_nan(
                raw["paired"]["point_weighted_rmse"],
                current["paired"]["point_weighted_rmse"],
            )
            fused_gain = _gain_or_nan(
                raw["fused"]["symmetric_mean"],
                current["fused"]["symmetric_mean"],
            )
            semantic_paired_gain = _gain_or_nan(
                raw["semantic_paired"]["point_weighted_rmse"],
                current["semantic_paired"]["point_weighted_rmse"],
            )
            semantic_fused_gain = _gain_or_nan(
                raw["semantic_fused"].get("symmetric_mean", float("nan")),
                current["semantic_fused"].get("symmetric_mean", float("nan")),
            )
        current["paired_gain_percent"] = paired_gain
        current["fused_gain_percent"] = fused_gain
        current["semantic_paired_gain_percent"] = semantic_paired_gain
        current["semantic_fused_gain_percent"] = semantic_fused_gain
        current["overall_pass"] = int(
            branch != raw_branch and paired_gain > 0.0 and fused_gain > 0.0
        )
        current["semantic_pass"] = int(
            branch != raw_branch
            and semantic_paired_gain > 0.0
            and semantic_fused_gain > 0.0
        )
    return metrics, rows, maps


def _per_instance_rows(
    *,
    families: dict[str, dict[str, torch.Tensor]],
    family_raw: dict[str, str],
    tracks: list[dict[str, object]],
    frames: tuple[int, ...],
    reference: int,
    target: torch.Tensor,
    valid: torch.Tensor,
    semantic_slots: torch.Tensor,
    map_flat_indices: torch.Tensor,
    map_semantic_slots: torch.Tensor,
    max_points: int,
    thresholds: tuple[float, ...],
) -> list[dict[str, object]]:
    target_flat = target.reshape(-1, 3).index_select(0, map_flat_indices)
    target_valid = torch.isfinite(target_flat).all(dim=-1)
    target_map = target_flat[target_valid]
    map_slots = map_semantic_slots[target_valid]
    output: list[dict[str, object]] = []
    for family, branches in families.items():
        raw_branch = family_raw[family]
        family_rows: list[dict[str, object]] = []
        for track in tracks:
            slot = int(track["slot"])
            slot_map = map_slots == slot
            for branch, points in branches.items():
                paired_rows = []
                visible_frames = 0
                for index in range(len(frames)):
                    if index == reference:
                        continue
                    support = (
                        valid[index]
                        & torch.isfinite(target[index]).all(dim=-1)
                        & (semantic_slots[index] == slot)
                    )
                    rmse, count = _frame_rmse(
                        points[index], target[index], support
                    )
                    if count:
                        visible_frames += 1
                    paired_rows.append({"rmse": rmse, "points": count})
                paired = _aggregate_metric_rows(
                    paired_rows,
                    value_key="rmse",
                    count_key="points",
                )
                branch_map = points.reshape(-1, 3).index_select(
                    0, map_flat_indices
                )[target_valid]
                fused = _bidirectional_metrics_or_empty(
                    branch_map[slot_map],
                    target_map[slot_map],
                    max_points=max_points,
                    thresholds=thresholds,
                )
                family_rows.append(
                    {
                        "family": family,
                        "branch": branch,
                        **track,
                        "visible_evaluation_frames": visible_frames,
                        "paired_points": paired["points"],
                        "paired_point_weighted_rmse": paired[
                            "point_weighted_rmse"
                        ],
                        "fused_points": fused.get("predicted_points", 0),
                        "fused_symmetric_mean": fused.get(
                            "symmetric_mean", float("nan")
                        ),
                    }
                )
        by_slot_branch = {
            (int(row["slot"]), str(row["branch"])): row
            for row in family_rows
        }
        for row in family_rows:
            raw = by_slot_branch[(int(row["slot"]), raw_branch)]
            if row["branch"] == raw_branch:
                paired_gain = fused_gain = 0.0
            else:
                paired_gain = _gain_or_nan(
                    raw["paired_point_weighted_rmse"],
                    row["paired_point_weighted_rmse"],
                )
                fused_gain = _gain_or_nan(
                    raw["fused_symmetric_mean"],
                    row["fused_symmetric_mean"],
                )
            row["paired_gain_percent"] = paired_gain
            row["fused_gain_percent"] = fused_gain
            row["instance_pass"] = int(
                row["branch"] != raw_branch
                and paired_gain > 0.0
                and fused_gain > 0.0
            )
        output.extend(family_rows)
    return output


def _fit_raw_depth_reference_similarity(
    raw_points: torch.Tensor,
    target_points: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[float, torch.Tensor, torch.Tensor, int, float]:
    """Fit one scoring-only Sim(3) from the geometry actually being mapped."""

    support = (
        valid.bool()
        & torch.isfinite(raw_points).all(dim=-1)
        & torch.isfinite(target_points).all(dim=-1)
    )
    source = raw_points[support].float().cpu()
    target = target_points[support].float().cpu()
    if source.shape[0] > 30_000:
        indices = torch.linspace(
            0,
            source.shape[0] - 1,
            steps=30_000,
        ).round().long()
        source = source.index_select(0, indices)
        target = target.index_select(0, indices)
    return _robust_similarity(source, target, min_points=128)


def _aggregate_metric_rows(
    rows: Sequence[dict[str, object]],
    *,
    value_key: str,
    count_key: str,
) -> dict[str, float | int]:
    valid = [
        row
        for row in rows
        if np.isfinite(float(row[value_key])) and int(row[count_key]) > 0
    ]
    if not valid:
        return {
            "frames": 0,
            "points": 0,
            "mean_frame_rmse": float("nan"),
            "point_weighted_rmse": float("nan"),
        }
    weights = np.asarray([int(row[count_key]) for row in valid], dtype=np.float64)
    values = np.asarray(
        [float(row[value_key]) for row in valid], dtype=np.float64
    )
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
        "branches=raw_depth_pose,qk_pose_raw_depth,qk_joint_depth_pose,raw_pointmap,qk_joint_pointmap",
        "candidate_generation=rgb_native_qk_only_then_frozen_camera_depth_point_heads",
        "shared_inputs=sam_masks,rgb,raw_confidence_pixel_support",
        "candidate_depth_used=1",
        "candidate_pointmap_used=1",
        "map_generation_gt_fields=0",
        "gt_role=scoring_only_after_all_native_maps_are_generated",
        "evaluation_alignment=fixed_family_raw_reference_sim3",
        f"depth_alignment_fit_rmse_metric={summary['alignments']['depth_backprojection']['fit_rmse_metric']}",
        f"point_head_alignment_fit_rmse_metric={summary['alignments']['point_head']['fit_rmse_metric']}",
        f"saved_map_points={summary['saved_map_points']}",
        f"saved_semantic_points={summary['saved_semantic_points']}",
        "",
        "family,branch,paired_rmse,paired_gain_percent,fused_symmetric,fused_gain_percent,semantic_paired_rmse,semantic_paired_gain_percent,semantic_fused_symmetric,semantic_fused_gain_percent,overall_pass,semantic_pass",
    ]
    branch_order = (
        "raw_depth_pose",
        "qk_pose_raw_depth",
        "qk_joint_depth_pose",
        "raw_pointmap",
        "qk_joint_pointmap",
    )
    for branch in branch_order:
        metric = summary["branch_metrics"][branch]
        lines.append(
            ",".join(
                str(value)
                for value in (
                    metric["family"],
                    branch,
                    metric["paired"]["point_weighted_rmse"],
                    metric["paired_gain_percent"],
                    metric["fused"]["symmetric_mean"],
                    metric["fused_gain_percent"],
                    metric["semantic_paired"]["point_weighted_rmse"],
                    metric["semantic_paired_gain_percent"],
                    metric["semantic_fused"].get("symmetric_mean", float("nan")),
                    metric["semantic_fused_gain_percent"],
                    metric["overall_pass"],
                    metric["semantic_pass"],
                )
            )
        )
    lines.extend(
        [
        "",
        "family,branch,slot,track_id,prompt,visible_frames,paired_points,paired_rmse,paired_gain_percent,fused_points,fused_symmetric,fused_gain_percent,instance_pass",
        ]
    )
    for row in summary["instance_metrics"]:
        lines.append(
            ",".join(
                str(value)
                for value in (
                    row["family"],
                    row["branch"],
                    row["slot"],
                    row["sam_track_id"],
                    row["prompt"],
                    row["visible_evaluation_frames"],
                    row["paired_points"],
                    row["paired_point_weighted_rmse"],
                    row["paired_gain_percent"],
                    row["fused_points"],
                    row["fused_symmetric_mean"],
                    row["fused_gain_percent"],
                    row["instance_pass"],
                )
            )
        )
    lines.extend(
        [
        "",
        f"pose_only_raw_depth_pass={summary['pose_only_raw_depth_pass']}",
        f"qk_joint_depth_map_pass={summary['qk_joint_depth_map_pass']}",
        f"qk_joint_pointmap_pass={summary['qk_joint_pointmap_pass']}",
        f"qk_joint_any_semantic_map_pass={summary['qk_joint_any_semantic_map_pass']}",
        f"claim={summary['claim']}",
        "",
        "outputs:",
        f"summary={run.output_dir / 'semantic_map_summary.json'}",
        f"maps={run.output_dir / 'semantic_maps.pt'}",
        f"raw_ply={run.output_dir / 'raw_semantic_map.ply'}",
        f"pose_only_ply={run.output_dir / 'qk_pose_raw_depth_semantic_map.ply'}",
        f"joint_depth_ply={run.output_dir / 'qk_joint_depth_semantic_map.ply'}",
        f"raw_pointmap_ply={run.output_dir / 'raw_pointmap_semantic_map.ply'}",
        f"joint_pointmap_ply={run.output_dir / 'qk_joint_pointmap_semantic_map.ply'}",
        f"frame_csv={run.output_dir / 'frame_metrics.csv'}",
        f"instance_csv={run.output_dir / 'instance_metrics.csv'}",
        f"copyable_report={path}",
        "===== COPYABLE_V0_SEMANTIC_MAP_AB_END =====",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def _validate_baseline_artifacts(
    *,
    payload: dict[str, Any],
    poses_path: Path,
    summary_path: Path,
    joint_path: Path,
    clip: ClipConfig,
) -> None:
    if (
        not poses_path.is_file()
        or not summary_path.is_file()
        or not joint_path.is_file()
    ):
        raise FileNotFoundError("Run commands_v0_baseline.txt before semantic map A/B.")
    summary = json.loads(summary_path.read_text(encoding="utf8"))
    expected = {
        "schema": 5,
        "baseline_status": "frozen",
        "implementation_revision": "v0_frozen_qk_pose_raw_pointmap_semantic_tracking_r7",
        "selected_pose_branch": "retrieve_qk",
        "selected_pose_exact_raw": 0,
        "pose_selection_fallback_used": 0,
        "candidate_pointmap_used": False,
        "candidate_geometry_available": True,
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
        "baseline_world_points",
        "tracking_masks_stream", "tracking_scores", "stream_images",
        "target_world_points",
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
    joint = torch.load(joint_path, map_location="cpu", weights_only=False)
    expected_joint = {
        "revision": "v0_streamvggt_qk_joint_geometry_replay_r2",
        "selected_pose_branch": "retrieve_qk",
        "geometry_branch": "retrieve_qk_joint_heads",
    }
    for name, value in expected_joint.items():
        if joint.get(name) != value:
            raise ValueError(
                f"QK joint artifact {name}={joint.get(name)!r}; "
                f"expected {value!r}."
            )
    if tuple(int(value) for value in joint.get("frame_indices", ())) != (
        clip.frame_indices
    ):
        raise ValueError("QK joint artifact frame order differs from config.")
    joint_pose = joint.get("selected_world_to_camera")
    if not torch.is_tensor(joint_pose) or joint_pose.shape != selected.shape:
        raise ValueError("QK joint pose shape differs from baseline selected pose.")
    if not torch.allclose(joint_pose, selected, atol=2e-5, rtol=1e-5):
        raise ValueError("QK joint pose differs from baseline selected pose.")
    sequence = len(clip.frame_indices)
    geometry_shapes = {
        "selected_intrinsics": (sequence, 3, 3),
        "selected_depth": None,
        "selected_depth_confidence": None,
        "selected_pointmap": None,
        "selected_pointmap_confidence": None,
    }
    for name, expected_shape in geometry_shapes.items():
        value = joint.get(name)
        if not torch.is_tensor(value) or not bool(torch.isfinite(value).all()):
            raise ValueError(f"QK joint artifact lacks finite tensor {name}.")
        if expected_shape is not None and tuple(value.shape) != expected_shape:
            raise ValueError(
                f"QK joint {name} shape={tuple(value.shape)}; "
                f"expected {expected_shape}."
            )
    depth = joint["selected_depth"]
    if depth.ndim != 4 or depth.shape[0] != sequence or depth.shape[-1] != 1:
        raise ValueError("QK joint selected_depth must have shape [S,H,W,1].")
    if joint["selected_depth_confidence"].shape != depth.shape:
        raise ValueError("QK joint depth confidence shape mismatch.")
    if joint["selected_pointmap"].shape != (*depth.shape[:3], 3):
        raise ValueError("QK joint selected_pointmap shape mismatch.")
    if joint["selected_pointmap_confidence"].shape != depth.shape:
        raise ValueError("QK joint pointmap confidence shape mismatch.")


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
