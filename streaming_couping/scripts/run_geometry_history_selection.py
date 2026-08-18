#!/usr/bin/env python3
"""Replay frozen StreamVGGT point heads with head-specific history policies."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from streaming_couping.src.backbones.streamvggt_latent import (
    ensure_thwc,
    load_streamvggt_latent_model,
)
from streaming_couping.src.backbones.streamvggt_parallel import (
    LayerShardedStreamVGGT,
    assert_frame_repository_cache_equivalence,
    assert_processed_key_cache_equivalence,
)
from streaming_couping.src.config import load_config
from streaming_couping.src.external_repos import maybe_add_repo_to_path
from streaming_couping.src.geometry_history_selection import (
    GeometryHistoryPolicy,
    select_geometry_history,
)
from streaming_couping.src.learned_pose.baseline_runtime import (
    load_baseline_run_config,
)
from streaming_couping.src.learned_pose.cache import cache_path, load_feature_cache
from streaming_couping.src.learned_pose.config import (
    ClipConfig,
    load_learned_pose_config,
)
from streaming_couping.src.pointmap_alignment import (
    _paired_limit,
    _robust_similarity,
)
from streaming_couping.src.qk_pose_retrieval import (
    QKRetrievalPolicy,
    rank_qk_history,
    select_qk_history,
)
from streaming_couping.src.semantic_map import normalize_confidence


REVISION = "geometry_specific_history_selection_r1"


@dataclass(frozen=True)
class ExperimentRun:
    source_path: Path
    v0_config: Path
    output_dir: Path
    geometry_policy: GeometryHistoryPolicy
    confidence_threshold: float
    alignment_max_points: int
    paired_max_points_per_frame: int
    symmetric_max_points_per_frame: int
    compatibility_max_points_per_frame: int
    reference_equivalence_max_rmse_native: float


def main() -> None:
    args = _parse_args()
    run = _load_run(args.config)
    data = load_learned_pose_config(run.v0_config)
    if args.streamvggt_devices:
        data = replace(
            data,
            streamvggt_devices=tuple(
                value.strip()
                for value in args.streamvggt_devices.split(",")
                if value.strip()
            ),
        )
    baseline = load_baseline_run_config(run.v0_config)
    clip = _find_clip(data.clips, baseline.clip_name)
    payload_path = cache_path(data, clip)
    payload = load_feature_cache(payload_path)
    qk_path = _qk_artifact_path(run.v0_config)
    qk_artifact = torch.load(qk_path, map_location="cpu", weights_only=False)
    _validate_frozen_inputs(payload, qk_artifact, clip)

    recovery = load_config(data.recovery_config)
    if len(data.streamvggt_devices) < 2 or not recovery.streaming_cache:
        raise ValueError("Geometry history replay requires two-device streaming.")
    maybe_add_repo_to_path(recovery.streamvggt_repo)
    assert_processed_key_cache_equivalence()
    assert_frame_repository_cache_equivalence()
    print("Geometry history processed-key/repository equivalence passed")
    model = load_streamvggt_latent_model(
        repo_path=recovery.streamvggt_repo,
        checkpoint_path=recovery.streamvggt_checkpoint,
        device="cpu",
        strict=True,
    )
    runner = LayerShardedStreamVGGT(
        model,
        data.streamvggt_devices,
        selected_layer_indices=data.fusion.dpt_layer_indices,
        amp_dtype=data.streamvggt_amp_dtype,
    )
    for line in runner.layout_summary():
        print(f"  {line}")

    run.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_payload = _candidate_generation_payload(payload, qk_artifact)
    print("GEOMETRY-SPECIFIC HISTORY POINT-HEAD REPLAY")
    print("  candidate fields=RGB,frame_indices,frozen_QK_pose")
    print("  SAM=0 GT=0 training=0 camera_head=0 depth_head=0")
    candidates = _generate_candidates(
        runner,
        candidate_payload,
        run=run,
    )
    runner.reset()
    artifact_path = run.output_dir / "candidate_pointmaps_native.pt"
    torch.save(
        {
            "schema": 1,
            "revision": REVISION,
            "artifact_role": "read_only_point_head_candidates_not_v0",
            "frame_indices": tuple(int(value) for value in payload["frame_indices"]),
            "candidate_generation_fields": tuple(candidate_payload),
            "candidate_generation_gt_fields": 0,
            "branches": {
                name: {
                    "world_points": value["world_points"],
                    "confidence": value["confidence"],
                }
                for name, value in candidates.items()
            },
        },
        artifact_path,
    )
    print("  candidates frozen; target pointmap is introduced for scoring now")
    summary, frame_rows = _score_candidates(
        run=run,
        payload=payload,
        qk_artifact=qk_artifact,
        candidates=candidates,
        cache_path_value=payload_path,
        qk_path=qk_path,
        artifact_path=artifact_path,
    )
    _write_outputs(run, summary, frame_rows, candidates)
    print(
        "  geometry_top4 "
        f"paired_gain={summary['branch_lookup']['geometry_top4']['paired_gain_vs_full_percent']:.4f}% "
        f"symmetric_gain={summary['branch_lookup']['geometry_top4']['symmetric_gain_vs_full_percent']:.4f}% "
        f"pass={summary['decision']['geometry_history_pass']}"
    )
    print(f"  result={run.output_dir / 'summary.json'}")


@torch.inference_mode()
def _generate_candidates(
    runner: LayerShardedStreamVGGT,
    payload: dict[str, Any],
    *,
    run: ExperimentRun,
) -> dict[str, dict[str, Any]]:
    images = payload["stream_images"].detach().float().cpu()
    frames = tuple(int(value) for value in payload["frame_indices"])
    qk_poses = payload["qk_world_to_camera"].detach().float().cpu()
    qk_policy = QKRetrievalPolicy(
        total_frame_budget=run.geometry_policy.total_frame_budget,
        anchor_frames=run.geometry_policy.anchor_frames,
    )
    output = {}
    for branch in ("qk_top4", "geometry_top4"):
        print(f"  replay branch={branch} frames={len(frames)}")
        runner.reset()
        points = []
        confidences = []
        history_rows: list[dict[str, Any]] = []
        pool_rows: list[dict[str, Any]] = []

        def selector(frame_index: int, qk_scores: torch.Tensor) -> tuple[int, ...]:
            ranked = rank_qk_history(qk_scores)
            if branch == "qk_top4":
                selected = select_qk_history(
                    frame_index,
                    qk_scores,
                    policy=qk_policy,
                )
                pool = tuple(ranked)
            else:
                selection = select_geometry_history(
                    frame_index,
                    qk_scores,
                    qk_poses,
                    policy=run.geometry_policy,
                )
                selected = selection.selected_indices
                pool = selection.qk_pool_indices
                for row in selection.diagnostics:
                    pool_rows.append(
                        {
                            "branch": branch,
                            "current_sequence_index": int(frame_index),
                            "current_frame_index": frames[int(frame_index)],
                            **row,
                            "history_frame_index": frames[
                                int(row["history_sequence_index"])
                            ],
                        }
                    )
            history_rows.append(
                {
                    "branch": branch,
                    "current_sequence_index": int(frame_index),
                    "current_frame_index": frames[int(frame_index)],
                    "available_history_frames": int(frame_index),
                    "selected_sequence_indices": _space(selected),
                    "selected_frame_indices": _space(frames[index] for index in selected),
                    "qk_pool_sequence_indices": _space(pool),
                    "qk_ranked_sequence_indices": _space(ranked),
                }
            )
            return selected

        for frame_index in range(images.shape[0]):
            batch_frame = images[frame_index : frame_index + 1].unsqueeze(0)
            selected_tokens = runner.aggregate_frame(
                batch_frame,
                frame_index,
                history_selector=selector,
            )
            point_output, confidence_output = runner.points(
                selected_tokens,
                batch_frame,
            )
            point_frame = ensure_thwc(point_output[0]).detach().float().cpu()
            confidence_frame = ensure_thwc(
                confidence_output[0]
            ).detach().float().cpu()
            if point_frame.shape[0] != 1 or point_frame.shape[-1] != 3:
                raise ValueError(
                    f"Unexpected point-head output {tuple(point_frame.shape)}."
                )
            if confidence_frame.shape[0] != 1 or confidence_frame.shape[-1] != 1:
                raise ValueError(
                    "Unexpected point confidence output "
                    f"{tuple(confidence_frame.shape)}."
                )
            points.append(point_frame[0])
            confidences.append(confidence_frame[0, ..., 0])
            del selected_tokens, point_output, confidence_output
        output[branch] = {
            "world_points": torch.stack(points),
            "confidence": torch.stack(confidences),
            "history_rows": history_rows,
            "pool_rows": pool_rows,
        }
        runner.reset()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return output


def _score_candidates(
    *,
    run: ExperimentRun,
    payload: dict[str, Any],
    qk_artifact: dict[str, Any],
    candidates: dict[str, dict[str, Any]],
    cache_path_value: Path,
    qk_path: Path,
    artifact_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    frames = tuple(int(value) for value in payload["frame_indices"])
    reference = int(payload["reference_sequence_index"])
    target = payload["target_world_points"].detach().float().cpu()
    branches = {
        "full_history": {
            "world_points": payload["baseline_world_points"].detach().float().cpu(),
            "confidence": payload["baseline_world_confidence"].detach().float().cpu(),
        },
        **candidates,
    }
    # The frozen V0 cache already stores per-frame normalized confidence.
    # Replay branches contain raw head confidence and receive the same one-time
    # normalization here.
    for name, value in branches.items():
        if name != "full_history":
            value["confidence"] = normalize_confidence(value["confidence"])
    expected_shape = target.shape
    for name, value in branches.items():
        if value["world_points"].shape != expected_shape:
            raise ValueError(
                f"Branch {name} pointmap {tuple(value['world_points'].shape)} "
                f"does not match target {tuple(expected_shape)}."
            )
        if value["confidence"].shape != expected_shape[:-1]:
            raise ValueError(f"Branch {name} confidence shape is inconsistent.")

    aligned = {}
    alignments = {}
    for name, value in branches.items():
        aligned[name], alignments[name] = _reference_align(
            value["world_points"],
            value["confidence"],
            target,
            reference_index=reference,
            confidence_threshold=run.confidence_threshold,
            max_points=run.alignment_max_points,
        )
    support = torch.isfinite(target).all(dim=-1)
    candidate_support = torch.ones_like(support)
    for value in branches.values():
        finite = torch.isfinite(value["world_points"]).all(dim=-1)
        confident = value["confidence"] >= run.confidence_threshold
        support &= finite & confident
        candidate_support &= finite & confident

    branch_rows = []
    frame_rows: list[dict[str, Any]] = []
    metric_lookup = {}
    for name in ("full_history", "qk_top4", "geometry_top4"):
        metrics, per_frame = _pointmap_metrics(
            aligned[name],
            target,
            support,
            paired_max_points=run.paired_max_points_per_frame,
            symmetric_max_points=run.symmetric_max_points_per_frame,
        )
        metric_lookup[name] = metrics
        frame_rows.extend(
            {
                "branch": name,
                "frame_index": frames[int(row["sequence_index"])],
                **row,
            }
            for row in per_frame
        )
        branch_rows.append(
            {
                "branch": name,
                **metrics,
                **{f"alignment_{key}": value for key, value in alignments[name].items()},
            }
        )

    qk_pose, qk_intrinsics = _decode_qk_pose(qk_artifact, payload)
    compatibility = {
        name: _pose_pointmap_compatibility(
            value["world_points"],
            candidate_support,
            qk_pose,
            qk_intrinsics,
            max_points_per_frame=run.compatibility_max_points_per_frame,
        )
        for name, value in branches.items()
    }
    reference_equivalence = {
        name: _reference_equivalence(
            branches["full_history"]["world_points"][reference],
            value["world_points"][reference],
        )
        for name, value in branches.items()
    }
    full_metrics = metric_lookup["full_history"]
    full_frame = {
        int(row["sequence_index"]): row
        for row in frame_rows
        if row["branch"] == "full_history"
    }
    for row in branch_rows:
        name = row["branch"]
        row["paired_gain_vs_full_percent"] = _gain(
            full_metrics["paired_rmse"], row["paired_rmse"]
        )
        row["symmetric_gain_vs_full_percent"] = _gain(
            full_metrics["symmetric_mean"], row["symmetric_mean"]
        )
        row["worse_nonreference_frames_vs_full"] = sum(
            1
            for item in frame_rows
            if item["branch"] == name
            and int(item["sequence_index"]) != reference
            and float(item["paired_rmse"])
            > float(full_frame[int(item["sequence_index"])]["paired_rmse"])
        )
        row.update(
            {
                f"compatibility_{key}": value
                for key, value in compatibility[name].items()
            }
        )
        row.update(
            {
                f"reference_{key}": value
                for key, value in reference_equivalence[name].items()
            }
        )

    lookup = {row["branch"]: row for row in branch_rows}
    geometry = lookup["geometry_top4"]
    qk = lookup["qk_top4"]
    full = lookup["full_history"]
    reference_pass = int(
        float(geometry["reference_rmse_native"])
        <= run.reference_equivalence_max_rmse_native
    )
    compatibility_pass = int(
        float(geometry["compatibility_reprojection_median_px"])
        <= float(full["compatibility_reprojection_median_px"])
    )
    frame_majority_pass = int(
        int(geometry["worse_nonreference_frames_vs_full"])
        < (len(frames) - 1) / 2.0
    )
    geometry_pass = int(
        reference_pass
        and compatibility_pass
        and frame_majority_pass
        and float(geometry["paired_gain_vs_full_percent"]) > 0.0
        and float(geometry["symmetric_gain_vs_full_percent"]) > 0.0
    )
    decision = {
        "geometry_history_pass": geometry_pass,
        "reference_gauge_pass": reference_pass,
        "pose_pointmap_compatibility_pass": compatibility_pass,
        "nonreference_frame_majority_pass": frame_majority_pass,
        "geometry_beats_qk_paired": int(
            float(geometry["paired_rmse"]) < float(qk["paired_rmse"])
        ),
        "geometry_beats_qk_symmetric": int(
            float(geometry["symmetric_mean"]) < float(qk["symmetric_mean"])
        ),
        "formal_v0_pose_modified": 0,
        "formal_v0_pointmap_modified": 0,
        "formal_v0_semantic_map_modified": 0,
        "next_gate": (
            "sam_covisibility_overlap_pool_probe"
            if geometry_pass
            else "keep_full_history_pointmap"
        ),
    }
    summary = {
        "schema": 1,
        "revision": REVISION,
        "experiment": "geometry_specific_history_selection",
        "baseline_version": "v0",
        "baseline_status": "frozen_unchanged",
        "clip": payload["clip_name"],
        "frames": frames,
        "reference_sequence_index": reference,
        "reference_frame": frames[reference],
        "branches": branch_rows,
        "branch_lookup": lookup,
        "history_policy": {
            "total_frame_budget": run.geometry_policy.total_frame_budget,
            "anchor_frames": run.geometry_policy.anchor_frames,
            "qk_overlap_pool_size": run.geometry_policy.overlap_pool_size,
            "geometry_rank": "equal_rank_sum_camera_center_baseline_and_relative_rotation",
        },
        "pose_branch": "frozen_v0_retrieve_qk_unchanged",
        "point_head_intermediate_layers": tuple(payload["dpt_layer_indices"]),
        "candidate_generation_fields": (
            "stream_images",
            "frame_indices",
            "frozen_qk_world_to_camera",
            "native_first_layer_qk_scores",
        ),
        "candidate_generation_gt_fields": 0,
        "sam_candidate_inputs": 0,
        "model_trained": 0,
        "camera_head_run": 0,
        "depth_head_run": 0,
        "point_head_run": 1,
        "gt_role": "reference_alignment_and_scoring_after_candidates_are_frozen",
        "common_supported_points": int(support.sum()),
        "cache": str(cache_path_value),
        "qk_pose_artifact": str(qk_path),
        "candidate_artifact": str(artifact_path),
        "decision": decision,
        "claim": (
            "geometry_specific_history_improves_pointmap_on_single_sequence"
            if geometry_pass
            else "geometry_specific_history_improvement_not_established"
        ),
    }
    return summary, frame_rows


def _reference_align(
    points: torch.Tensor,
    confidence: torch.Tensor,
    target: torch.Tensor,
    *,
    reference_index: int,
    confidence_threshold: float,
    max_points: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    source = points[int(reference_index)].reshape(-1, 3)
    truth = target[int(reference_index)].reshape(-1, 3)
    weights = confidence[int(reference_index)].reshape(-1)
    valid = (
        torch.isfinite(source).all(dim=-1)
        & torch.isfinite(truth).all(dim=-1)
        & torch.isfinite(weights)
        & (weights >= float(confidence_threshold))
    )
    source_fit, truth_fit = _paired_limit(
        source[valid],
        truth[valid],
        max_points=int(max_points),
    )
    scale, rotation, translation, inliers, fit_rmse = _robust_similarity(
        source_fit,
        truth_fit,
        min_points=128,
    )
    aligned = float(scale) * (points.float() @ rotation.T) + translation
    return aligned, {
        "scale": float(scale),
        "inliers": int(inliers),
        "fit_rmse": float(fit_rmse),
    }


def _pointmap_metrics(
    predicted: torch.Tensor,
    target: torch.Tensor,
    support: torch.Tensor,
    *,
    paired_max_points: int,
    symmetric_max_points: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    all_errors = []
    symmetric_values = []
    frame_rows = []
    for frame in range(predicted.shape[0]):
        selected = torch.nonzero(support[frame].reshape(-1), as_tuple=False)[:, 0]
        if selected.numel() == 0:
            raise ValueError(
                "Common pointmap evaluation support is empty for "
                f"sequence index {frame}."
            )
        paired_index = _even_limit(selected, paired_max_points)
        pred = predicted[frame].reshape(-1, 3).index_select(0, paired_index)
        truth = target[frame].reshape(-1, 3).index_select(0, paired_index)
        error = torch.linalg.vector_norm(pred - truth, dim=-1)
        all_errors.append(error)
        symmetric_index = _even_limit(selected, symmetric_max_points)
        symmetric = _symmetric_mean(
            predicted[frame].reshape(-1, 3).index_select(0, symmetric_index),
            target[frame].reshape(-1, 3).index_select(0, symmetric_index),
        )
        symmetric_values.append(symmetric)
        frame_rows.append(
            {
                "sequence_index": frame,
                "supported_points": int(selected.numel()),
                "paired_points": int(error.numel()),
                "paired_rmse": _rmse(error),
                "paired_median": float(error.median()),
                "paired_p90": float(torch.quantile(error, 0.90)),
                "symmetric_mean": symmetric,
            }
        )
    joined = torch.cat(all_errors)
    return {
        "supported_points": int(support.sum()),
        "paired_rmse": _rmse(joined),
        "paired_median": float(joined.median()),
        "paired_p90": float(torch.quantile(joined, 0.90)),
        "symmetric_mean": float(torch.tensor(symmetric_values).mean()),
    }, frame_rows


def _symmetric_mean(first: torch.Tensor, second: torch.Tensor) -> float:
    if first.numel() == 0 or second.numel() == 0:
        return float("nan")
    distance = torch.cdist(first.float(), second.float())
    return float(0.5 * (distance.min(dim=1).values.mean() + distance.min(dim=0).values.mean()))


def _pose_pointmap_compatibility(
    points: torch.Tensor,
    support: torch.Tensor,
    world_to_camera: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    max_points_per_frame: int,
) -> dict[str, Any]:
    errors = []
    positive = 0
    in_bounds = 0
    count = 0
    height, width = support.shape[1:]
    for frame in range(points.shape[0]):
        selected = torch.nonzero(support[frame].reshape(-1), as_tuple=False)[:, 0]
        selected = _even_limit(selected, max_points_per_frame)
        if selected.numel() == 0:
            continue
        world = points[frame].reshape(-1, 3).index_select(0, selected)
        pose = world_to_camera[frame]
        camera = world @ pose[:3, :3].T + pose[:3, 3]
        pixel_h = camera @ intrinsics[frame].T
        depth = pixel_h[:, 2:]
        signed_epsilon = torch.where(
            depth >= 0.0,
            torch.full_like(depth, 1e-12),
            torch.full_like(depth, -1e-12),
        )
        safe_depth = torch.where(depth.abs() < 1e-12, signed_epsilon, depth)
        pixel = pixel_h[:, :2] / safe_depth
        y = torch.div(selected, width, rounding_mode="floor").float()
        x = selected.remainder(width).float()
        expected = torch.stack((x, y), dim=-1)
        finite = torch.isfinite(pixel).all(dim=-1)
        errors.append(torch.linalg.vector_norm(pixel[finite] - expected[finite], dim=-1))
        positive += int((camera[:, 2] > 0.0).sum())
        in_bounds += int(
            (
                finite
                & (camera[:, 2] > 0.0)
                & (pixel[:, 0] >= 0.0)
                & (pixel[:, 0] <= float(width - 1))
                & (pixel[:, 1] >= 0.0)
                & (pixel[:, 1] <= float(height - 1))
            ).sum()
        )
        count += int(selected.numel())
    if count == 0:
        raise ValueError("Pose/pointmap compatibility support is empty.")
    joined = torch.cat(errors)
    if joined.numel() == 0:
        raise ValueError("Pose/pointmap compatibility has no finite projections.")
    return {
        "samples": count,
        "reprojection_median_px": float(joined.median()),
        "reprojection_p90_px": float(torch.quantile(joined, 0.90)),
        "positive_z_rate": positive / max(count, 1),
        "in_bounds_rate": in_bounds / max(count, 1),
    }


def _decode_qk_pose(
    qk_artifact: dict[str, Any], payload: dict[str, Any]
) -> tuple[torch.Tensor, torch.Tensor]:
    from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri

    encoding = qk_artifact["pose_encoding"].detach().float().cpu()
    pose, intrinsics = pose_encoding_to_extri_intri(
        encoding.unsqueeze(0),
        image_size_hw=tuple(int(value) for value in payload["image_size"]),
    )
    decoded = pose[0].detach().float().cpu()
    decoded_intrinsics = intrinsics[0].detach().float().cpu()
    stored = qk_artifact["selected_world_to_camera"].detach().float().cpu()
    if stored.ndim == 4 and stored.shape[0] == 1:
        stored = stored[0]
    if stored.shape != decoded.shape or not torch.allclose(
        stored,
        decoded,
        rtol=1e-5,
        atol=1e-5,
    ):
        raise ValueError(
            "Decoded QK pose does not match selected_world_to_camera in the "
            "frozen V0 artifact."
        )
    if not torch.isfinite(decoded).all() or not torch.isfinite(decoded_intrinsics).all():
        raise ValueError("Decoded QK camera parameters contain non-finite values.")
    return decoded, decoded_intrinsics


def _reference_equivalence(
    full_reference: torch.Tensor, candidate_reference: torch.Tensor
) -> dict[str, Any]:
    valid = (
        torch.isfinite(full_reference).all(dim=-1)
        & torch.isfinite(candidate_reference).all(dim=-1)
    )
    difference = torch.linalg.vector_norm(
        full_reference[valid] - candidate_reference[valid], dim=-1
    )
    if difference.numel() == 0:
        raise ValueError("Reference-frame equivalence has no finite point pairs.")
    return {
        "compared_points": int(difference.numel()),
        "rmse_native": _rmse(difference),
        "maximum_native": float(difference.max()),
    }


def _write_outputs(
    run: ExperimentRun,
    summary: dict[str, Any],
    frame_rows: list[dict[str, Any]],
    candidates: dict[str, dict[str, Any]],
) -> None:
    _write_json(run.output_dir / "summary.json", summary)
    _write_csv(run.output_dir / "branch_summary.csv", summary["branches"])
    _write_csv(run.output_dir / "frame_metrics.csv", frame_rows)
    history_rows = [
        row
        for value in candidates.values()
        for row in value["history_rows"]
    ]
    pool_rows = [
        row
        for value in candidates.values()
        for row in value["pool_rows"]
    ]
    _write_csv(run.output_dir / "history_selection.csv", history_rows)
    _write_csv(run.output_dir / "geometry_pool_diagnostics.csv", pool_rows)
    _write_copyable(run.output_dir / "copyable_result.txt", summary)


def _write_copyable(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "===== COPYABLE_GEOMETRY_HISTORY_SELECTION_BEGIN =====",
        f"revision={summary['revision']}",
        f"clip={summary['clip']}",
        f"frames={len(summary['frames'])}",
        "branches=full_history,qk_top4,geometry_top4",
        f"pose_branch={summary['pose_branch']}",
        f"point_head_intermediate_layers={_space(summary['point_head_intermediate_layers'])}",
        f"total_frame_budget={summary['history_policy']['total_frame_budget']}",
        f"anchor_frames={summary['history_policy']['anchor_frames']}",
        f"qk_overlap_pool_size={summary['history_policy']['qk_overlap_pool_size']}",
        "model_trained=0",
        "sam_candidate_inputs=0",
        "candidate_generation_gt_fields=0",
        "formal_v0_modified=0",
        "",
        "branch,supported_points,paired_rmse,paired_median,paired_p90,symmetric_mean,paired_gain_vs_full_percent,symmetric_gain_vs_full_percent,worse_nonreference_frames_vs_full,compatibility_reprojection_median_px,compatibility_reprojection_p90_px,compatibility_positive_z_rate,compatibility_in_bounds_rate,reference_rmse_native",
    ]
    fields = (
        "branch",
        "supported_points",
        "paired_rmse",
        "paired_median",
        "paired_p90",
        "symmetric_mean",
        "paired_gain_vs_full_percent",
        "symmetric_gain_vs_full_percent",
        "worse_nonreference_frames_vs_full",
        "compatibility_reprojection_median_px",
        "compatibility_reprojection_p90_px",
        "compatibility_positive_z_rate",
        "compatibility_in_bounds_rate",
        "reference_rmse_native",
    )
    for row in summary["branches"]:
        lines.append(",".join(str(row[field]) for field in fields))
    lines.extend(
        (
            "",
            "decision=" + json.dumps(summary["decision"], sort_keys=True),
            f"claim={summary['claim']}",
            "outputs:",
            f"summary={path.with_name('summary.json')}",
            f"branch_csv={path.with_name('branch_summary.csv')}",
            f"frame_csv={path.with_name('frame_metrics.csv')}",
            f"history_csv={path.with_name('history_selection.csv')}",
            f"geometry_pool_csv={path.with_name('geometry_pool_diagnostics.csv')}",
            f"candidate_artifact={summary['candidate_artifact']}",
            "===== COPYABLE_GEOMETRY_HISTORY_SELECTION_END =====",
        )
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def _candidate_generation_payload(
    payload: dict[str, Any], qk_artifact: dict[str, Any]
) -> dict[str, Any]:
    output = {
        "stream_images": payload["stream_images"],
        "frame_indices": tuple(int(value) for value in payload["frame_indices"]),
        "qk_world_to_camera": qk_artifact["selected_world_to_camera"],
    }
    forbidden = (
        "target_world_points",
        "target_depth",
        "target_pose_encoding",
        "tracking_masks_stream",
        "sam_track_ids",
    )
    if any(name in output for name in forbidden):
        raise RuntimeError("Geometry candidate payload contains a forbidden field.")
    return output


def _validate_frozen_inputs(
    payload: dict[str, Any], qk: dict[str, Any], clip: ClipConfig
) -> None:
    required = (
        "stream_images",
        "frame_indices",
        "reference_sequence_index",
        "baseline_world_points",
        "baseline_world_confidence",
        "target_world_points",
        "dpt_layer_indices",
        "image_size",
    )
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError(f"Frozen V0 cache lacks fields: {missing}.")
    frames = tuple(int(value) for value in payload["frame_indices"])
    if frames != clip.frame_indices or payload.get("clip_name") != clip.name:
        raise ValueError("Frozen V0 cache does not match the configured clip.")
    if qk.get("selected_pose_branch") != "retrieve_qk":
        raise ValueError("QK artifact is not the frozen retrieve_qk branch.")
    if tuple(int(value) for value in qk.get("frame_indices", ())) != frames:
        raise ValueError("QK artifact frame order differs from the V0 cache.")
    for name in ("selected_world_to_camera", "pose_encoding"):
        if not torch.is_tensor(qk.get(name)):
            raise ValueError(f"QK artifact lacks tensor {name!r}.")


def _load_run(path: str | Path) -> ExperimentRun:
    source = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf8")) or {}
    history = raw.get("history", {})
    evaluation = raw.get("evaluation", {})
    run = ExperimentRun(
        source_path=source,
        v0_config=_resolve(raw.get("v0_config", "streaming_couping/configs/v0_baseline.yaml")),
        output_dir=_resolve(
            raw.get(
                "output_dir",
                "outputs/streaming_couping_geometry_history_selection",
            )
        ),
        geometry_policy=GeometryHistoryPolicy(
            total_frame_budget=int(history.get("total_frame_budget", 5)),
            anchor_frames=int(history.get("anchor_frames", 1)),
            overlap_pool_size=int(history.get("qk_overlap_pool_size", 8)),
        ),
        confidence_threshold=float(evaluation.get("confidence_threshold", 0.30)),
        alignment_max_points=int(evaluation.get("alignment_max_points", 30000)),
        paired_max_points_per_frame=int(
            evaluation.get("paired_max_points_per_frame", 8192)
        ),
        symmetric_max_points_per_frame=int(
            evaluation.get("symmetric_max_points_per_frame", 512)
        ),
        compatibility_max_points_per_frame=int(
            evaluation.get("compatibility_max_points_per_frame", 4096)
        ),
        reference_equivalence_max_rmse_native=float(
            evaluation.get("reference_equivalence_max_rmse_native", 1e-4)
        ),
    )
    run.geometry_policy.validate()
    if run.v0_config.name != "v0_baseline.yaml":
        raise ValueError("Experiment must consume the frozen V0 configuration.")
    if not 0.0 <= run.confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be in [0,1].")
    for value in (
        run.alignment_max_points,
        run.paired_max_points_per_frame,
        run.symmetric_max_points_per_frame,
        run.compatibility_max_points_per_frame,
    ):
        if int(value) < 128:
            raise ValueError("Evaluation sample budgets must be at least 128.")
    return run


def _qk_artifact_path(config: Path) -> Path:
    raw = yaml.safe_load(config.read_text(encoding="utf8")) or {}
    return _resolve(raw["baseline"]["pose"]["qk_pose_output"])


def _find_clip(clips: tuple[ClipConfig, ...], name: str) -> ClipConfig:
    selected = [clip for clip in clips if clip.name == name]
    if len(selected) != 1:
        raise ValueError(f"Expected exactly one clip {name!r}.")
    return selected[0]


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def _even_limit(indices: torch.Tensor, limit: int) -> torch.Tensor:
    if indices.numel() <= int(limit):
        return indices
    position = torch.linspace(0, indices.numel() - 1, steps=int(limit)).long()
    return indices.index_select(0, position)


def _rmse(values: torch.Tensor) -> float:
    if values.numel() == 0:
        raise ValueError("RMSE requires at least one value.")
    return float(torch.sqrt(values.float().square().mean()))


def _gain(baseline: float, candidate: float) -> float:
    return 100.0 * (float(baseline) - float(candidate)) / max(abs(float(baseline)), 1e-12)


def _space(values) -> str:
    return " ".join(str(int(value)) for value in values)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf8")
        return
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="streaming_couping/configs/geometry_history_selection.yaml",
    )
    parser.add_argument("--streamvggt-devices", default="")
    return parser.parse_args()


if __name__ == "__main__":
    main()
