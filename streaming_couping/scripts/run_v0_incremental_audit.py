#!/usr/bin/env python3
"""Run the two gated V0 diagnostics in one frozen-model invocation."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import gc
import json
import math
from pathlib import Path
from typing import Any, Sequence

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
from streaming_couping.src.context_policy_audit import (
    fixed_alignment_pointmap_rows,
    pose_pointmap_consistency_rows,
    select_random_history,
    select_recent_history,
    summarize_consistency_rows,
    summarize_pointmap_rows,
)
from streaming_couping.src.external_repos import maybe_add_repo_to_path
from streaming_couping.src.learned_pose.baseline_runtime import (
    load_baseline_run_config,
    pose_metrics,
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
from streaming_couping.src.qk_pose_retrieval import rank_qk_history


REVISION = "v0_pose_pointmap_and_context_policy_audit_r1"


@dataclass(frozen=True)
class AuditRun:
    source_path: Path
    output_dir: Path
    total_history_budget: int
    anchor_frames: int
    random_seeds: tuple[int, ...]
    confidence_threshold: float


def main() -> None:
    args = _parse_args()
    data = load_learned_pose_config(args.config)
    baseline = load_baseline_run_config(args.config)
    run = _load_run(args.config)
    if args.output_dir:
        run = replace(
            run,
            output_dir=Path(args.output_dir).expanduser().resolve(),
        )
    if args.streamvggt_devices:
        data = replace(
            data,
            streamvggt_devices=tuple(
                value.strip()
                for value in args.streamvggt_devices.split(",")
                if value.strip()
            ),
        )
    clip = _find_clip(data, baseline.clip_name)
    source_cache = cache_path(data, clip)
    payload = load_feature_cache(source_cache)
    qk_path = baseline.qk_pose_output
    qk = torch.load(qk_path, map_location="cpu", weights_only=False)
    _validate_inputs(payload, qk=qk, clip=clip)

    recovery = load_config(data.recovery_config)
    if not data.streamvggt_devices or not recovery.streaming_cache:
        raise ValueError("V0 context audit requires layer-sharded streaming.")
    maybe_add_repo_to_path(recovery.streamvggt_repo)
    from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri

    image_size = tuple(int(value) for value in payload["image_size"])
    raw_pose_batch, raw_intrinsics_batch = pose_encoding_to_extri_intri(
        payload["baseline_pose_encoding"].unsqueeze(0).float(),
        image_size_hw=image_size,
    )
    target_pose_batch, _ = pose_encoding_to_extri_intri(
        payload["target_pose_encoding"].unsqueeze(0).float(),
        image_size_hw=image_size,
    )
    raw_pose = raw_pose_batch.detach().float().cpu()
    target_pose = target_pose_batch.detach().float().cpu()
    raw_intrinsics = raw_intrinsics_batch[0].detach().float().cpu()
    qk_pose = qk["selected_world_to_camera"].detach().float().cpu()
    artifact_raw_pose = qk["raw_world_to_camera"].detach().float().cpu()
    if not torch.allclose(
        artifact_raw_pose,
        raw_pose,
        atol=2e-5,
        rtol=1e-5,
    ):
        difference = float((artifact_raw_pose - raw_pose).abs().max())
        raise RuntimeError(
            "QK artifact was produced from a different raw V0 pose; "
            f"maximum absolute difference={difference}."
        )
    frames = tuple(int(value) for value in payload["frame_indices"])
    reference = int(payload["reference_sequence_index"])
    evaluation_indices = [index for index in range(len(frames)) if index != reference]

    run.output_dir.mkdir(parents=True, exist_ok=True)
    consistency_rows, consistency_summaries = _run_consistency_audit(
        payload=payload,
        frames=frames,
        raw_intrinsics=raw_intrinsics,
        raw_pose=raw_pose,
        qk_pose=qk_pose,
        confidence_threshold=run.confidence_threshold,
    )
    _write_csv(run.output_dir / "pose_pointmap_consistency.csv", consistency_rows)
    print("V0 gate 1/2: raw-pointmap pose consistency audit completed")

    context_rows: list[dict[str, object]] = []
    pointmap_rows: list[dict[str, object]] = []
    retrieval_rows = _load_qk_retrieval_rows(qk_path)
    pose_outputs: dict[str, torch.Tensor] = {}
    full_candidate = {
        "pose": raw_pose,
        "depth": payload["baseline_depth"].detach().float().cpu(),
        "pointmap": payload["baseline_world_points"].detach().float().cpu(),
    }
    full_row, full_point_rows = _score_context_candidate(
        method="full_history",
        policy="native_full_history",
        seed=None,
        payload=payload,
        candidate=full_candidate,
        raw_pose=raw_pose,
        target_pose=target_pose,
        frames=frames,
        reference=reference,
        evaluation_indices=evaluation_indices,
        confidence_threshold=run.confidence_threshold,
        raw_context=None,
    )
    context_rows.append(full_row)
    pointmap_rows.extend(full_point_rows)
    pose_outputs["full_history"] = raw_pose[0].clone()

    qk_candidate = {
        "pose": qk_pose,
        "depth": qk["selected_depth"].detach().float().cpu(),
        "pointmap": qk["selected_pointmap"].detach().float().cpu(),
    }
    qk_row, qk_point_rows = _score_context_candidate(
        method="qk_topk",
        policy="first_anchor_plus_native_qk_topk",
        seed=None,
        payload=payload,
        candidate=qk_candidate,
        raw_pose=raw_pose,
        target_pose=target_pose,
        frames=frames,
        reference=reference,
        evaluation_indices=evaluation_indices,
        confidence_threshold=run.confidence_threshold,
        raw_context=full_row,
    )
    context_rows.append(qk_row)
    pointmap_rows.extend(qk_point_rows)
    pose_outputs["qk_topk"] = qk_pose[0].clone()
    del qk_candidate

    assert_processed_key_cache_equivalence()
    assert_frame_repository_cache_equivalence()
    print("V0 gate 2/2: processed-key/repository equivalence passed")
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

    methods: list[tuple[str, str, int | None]] = [
        ("recent_topk", "first_anchor_plus_recent_topk", None),
        *[
            (
                f"random_topk_seed_{seed}",
                "first_anchor_plus_random_topk",
                int(seed),
            )
            for seed in run.random_seeds
        ],
    ]
    generation = {
        "stream_images": payload["stream_images"].detach().float().cpu(),
        "frame_indices": frames,
    }
    for method, policy, seed in methods:
        print(f"V0 context replay method={method}")
        replay = _generate_context_candidate(
            runner,
            generation,
            method=method,
            total_history_budget=run.total_history_budget,
            anchor_frames=run.anchor_frames,
            seed=seed,
        )
        candidate_pose, _ = pose_encoding_to_extri_intri(
            replay["pose_encoding"].unsqueeze(0).float(),
            image_size_hw=image_size,
        )
        candidate = {
            "pose": candidate_pose.detach().float().cpu(),
            "depth": replay["depth"],
            "pointmap": replay["pointmap"],
        }
        row, candidate_point_rows = _score_context_candidate(
            method=method,
            policy=policy,
            seed=seed,
            payload=payload,
            candidate=candidate,
            raw_pose=raw_pose,
            target_pose=target_pose,
            frames=frames,
            reference=reference,
            evaluation_indices=evaluation_indices,
            confidence_threshold=run.confidence_threshold,
            raw_context=full_row,
        )
        context_rows.append(row)
        pointmap_rows.extend(candidate_point_rows)
        retrieval_rows.extend(replay["retrieval_rows"])
        pose_outputs[method] = candidate_pose[0].detach().float().cpu()
        del replay, candidate, candidate_pose
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    runner.reset()

    random_summary = _random_control_summary(context_rows)
    decisions = _decisions(
        context_rows=context_rows,
        consistency_summaries=consistency_summaries,
        random_summary=random_summary,
    )
    summary = {
        "schema": 1,
        "revision": REVISION,
        "baseline_version": "v0",
        "clip": payload["clip_name"],
        "config": str(run.source_path),
        "cache": str(source_cache),
        "qk_artifact": str(qk_path),
        "frames": frames,
        "stream_frames": len(frames),
        "reference_sequence_index": reference,
        "reference_frame": frames[reference],
        "evaluated_pose_frames": len(evaluation_indices),
        "model_trained": 0,
        "sam_used_by_context_candidates": 0,
        "gt_role": "scoring_only_after_each_rgb_only_candidate_is_generated",
        "candidate_generation_fields": ("stream_images", "frame_indices"),
        "total_history_budget": run.total_history_budget,
        "anchor_frames": run.anchor_frames,
        "random_seeds": run.random_seeds,
        "confidence_threshold": run.confidence_threshold,
        "pose_pointmap_consistency": consistency_summaries,
        "context_policy_results": context_rows,
        "random_control_summary": random_summary,
        "decisions": decisions,
    }
    _write_csv(run.output_dir / "context_policy_summary.csv", context_rows)
    _write_csv(run.output_dir / "context_pointmap_frames.csv", pointmap_rows)
    _write_csv(run.output_dir / "context_retrieval_frames.csv", retrieval_rows)
    torch.save(
        {
            "revision": REVISION,
            "frame_indices": frames,
            "coordinate_frame": "streamvggt_native_first_frame_gauge",
            "world_to_camera": pose_outputs,
            "geometry_saved": False,
        },
        run.output_dir / "context_pose_outputs.pt",
    )
    result = run.output_dir / "incremental_audit_summary.json"
    result.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf8",
    )
    _write_copyable(
        run.output_dir / "copyable_result.txt",
        summary=summary,
        output_dir=run.output_dir,
    )
    print("V0 INCREMENTAL POSE/POINTMAP + CONTEXT AUDIT")
    print(
        "  context_control_pass="
        f"{decisions['qk_beats_recent_and_random_pose_controls']} "
        f"headwise_conflict={decisions['headwise_context_conflict_observed']}"
    )
    print(f"  copyable_report={run.output_dir / 'copyable_result.txt'}")
    print(f"V0 incremental audit result={result}")


@torch.inference_mode()
def _generate_context_candidate(
    runner: LayerShardedStreamVGGT,
    payload: dict[str, Any],
    *,
    method: str,
    total_history_budget: int,
    anchor_frames: int,
    seed: int | None,
) -> dict[str, Any]:
    images = payload["stream_images"]
    frames = tuple(int(value) for value in payload["frame_indices"])
    pose_encodings: list[torch.Tensor] = []
    depths: list[torch.Tensor] = []
    pointmaps: list[torch.Tensor] = []
    retrieval_rows: list[dict[str, object]] = []
    runner.reset()

    def selector(
        frame_index: int,
        qk_scores: torch.Tensor,
    ) -> tuple[int, ...]:
        if method == "recent_topk":
            selected = select_recent_history(
                frame_index,
                total_history_budget=total_history_budget,
                anchor_frames=anchor_frames,
            )
        elif method.startswith("random_topk_seed_") and seed is not None:
            selected = select_random_history(
                frame_index,
                total_history_budget=total_history_budget,
                anchor_frames=anchor_frames,
                seed=seed,
            )
        else:
            raise ValueError(f"Unsupported context replay method={method!r}.")
        ranked = rank_qk_history(qk_scores)
        retrieval_rows.append(
            {
                "method": method,
                "random_seed": "" if seed is None else int(seed),
                "sequence_index": int(frame_index),
                "frame_index": frames[int(frame_index)],
                "history_frames": int(frame_index),
                "selected_history_count": len(selected),
                "selected_sequence_indices": _space(selected),
                "selected_frame_indices": _space(frames[index] for index in selected),
                "native_qk_ranked_sequence_indices": _space(ranked),
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
        pose_encoding = runner.camera(selected_tokens)
        depth, depth_confidence = runner.depth(selected_tokens, batch_frame)
        pointmap, pointmap_confidence = runner.points(
            selected_tokens,
            batch_frame,
        )
        pose_encodings.append(pose_encoding[0, 0].detach().float().cpu())
        depths.append(ensure_thwc(depth[0]).detach().float().cpu())
        pointmaps.append(ensure_thwc(pointmap[0]).detach().float().cpu())
        del (
            selected_tokens,
            pose_encoding,
            depth,
            depth_confidence,
            pointmap,
            pointmap_confidence,
        )
    runner.reset()
    return {
        "pose_encoding": torch.stack(pose_encodings),
        "depth": torch.cat(depths, dim=0),
        "pointmap": torch.cat(pointmaps, dim=0),
        "retrieval_rows": retrieval_rows,
    }


def _run_consistency_audit(
    *,
    payload: dict[str, Any],
    frames: tuple[int, ...],
    raw_intrinsics: torch.Tensor,
    raw_pose: torch.Tensor,
    qk_pose: torch.Tensor,
    confidence_threshold: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for branch, pose in (
        ("raw_pose", raw_pose),
        ("qk_pose", qk_pose),
    ):
        branch_rows = pose_pointmap_consistency_rows(
            branch=branch,
            frame_indices=frames,
            world_points=payload["baseline_world_points"],
            world_confidence=payload["baseline_world_confidence"],
            depth=payload["baseline_depth"],
            depth_confidence=payload["baseline_depth_confidence"],
            intrinsics=raw_intrinsics,
            world_to_camera=pose,
            confidence_threshold=confidence_threshold,
        )
        rows.extend(branch_rows)
        summaries.append(summarize_consistency_rows(branch_rows))
    return rows, summaries


def _score_context_candidate(
    *,
    method: str,
    policy: str,
    seed: int | None,
    payload: dict[str, Any],
    candidate: dict[str, torch.Tensor],
    raw_pose: torch.Tensor,
    target_pose: torch.Tensor,
    frames: tuple[int, ...],
    reference: int,
    evaluation_indices: list[int],
    confidence_threshold: float,
    raw_context: dict[str, object] | None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    pose = candidate["pose"].detach().float().cpu()
    raw_metrics = pose_metrics(
        raw_pose,
        target_pose,
        reference_index=reference,
        evaluation_indices=evaluation_indices,
    )
    pose_result = pose_metrics(
        pose,
        target_pose,
        reference_index=reference,
        evaluation_indices=evaluation_indices,
    )
    point_rows = fixed_alignment_pointmap_rows(
        method=method,
        frame_indices=frames,
        reference_index=reference,
        pointmap=candidate["pointmap"],
        raw_confidence=payload["baseline_world_confidence"],
        raw_world_points=payload["baseline_world_points"],
        target_world_points=payload["target_world_points"],
        scale=float(payload["point_alignment_scale"]),
        rotation=payload["point_alignment_rotation"],
        translation=payload["point_alignment_translation"],
        confidence_threshold=confidence_threshold,
    )
    point_summary = summarize_pointmap_rows(point_rows)
    depth_abs_rel = _depth_abs_rel(
        candidate["depth"],
        target=payload["target_depth"],
        raw_depth=payload["baseline_depth"],
        raw_confidence=payload["baseline_depth_confidence"],
        scale=float(payload["point_alignment_scale"]),
        confidence_threshold=confidence_threshold,
        reference_index=reference,
    )
    row: dict[str, object] = {
        "method": method,
        "history_policy": policy,
        "random_seed": "" if seed is None else int(seed),
        "center_error_native": float(pose_result["center_error_native"]),
        "rotation_degrees": float(pose_result["rotation_degrees"]),
        "center_gain_vs_raw_percent": _gain(
            raw_metrics["center_error_native"],
            pose_result["center_error_native"],
        ),
        "rotation_gain_vs_raw_percent": _gain(
            raw_metrics["rotation_degrees"],
            pose_result["rotation_degrees"],
        ),
        "pointmap_paired_weighted_rmse": float(
            point_summary["mean_frame_paired_weighted_rmse"]
        ),
        "pointmap_gain_vs_full_percent": 0.0,
        "depth_abs_rel": depth_abs_rel,
        "depth_gain_vs_full_percent": 0.0,
    }
    if raw_context is not None:
        row["pointmap_gain_vs_full_percent"] = _gain(
            float(raw_context["pointmap_paired_weighted_rmse"]),
            float(row["pointmap_paired_weighted_rmse"]),
        )
        row["depth_gain_vs_full_percent"] = _gain(
            float(raw_context["depth_abs_rel"]),
            float(row["depth_abs_rel"]),
        )
    return row, point_rows


def _depth_abs_rel(
    candidate: torch.Tensor,
    *,
    target: torch.Tensor,
    raw_depth: torch.Tensor,
    raw_confidence: torch.Tensor,
    scale: float,
    confidence_threshold: float,
    reference_index: int,
) -> float:
    predicted = _scalar(candidate)
    target_value = _scalar(target)
    raw_value = _scalar(raw_depth)
    confidence = _normalized(raw_confidence)
    if predicted.shape != target_value.shape or predicted.shape != raw_value.shape:
        raise ValueError("Candidate/raw/target depth shapes differ.")
    valid = (
        torch.isfinite(raw_value)
        & (raw_value > 1e-6)
        & torch.isfinite(target_value)
        & (target_value > 1e-6)
        & torch.isfinite(confidence)
        & (confidence >= float(confidence_threshold))
        & torch.isfinite(predicted)
        & (predicted > 1e-6)
    )
    valid[int(reference_index)] = False
    values = (
        (float(scale) * predicted - target_value).abs()
        / target_value.clamp_min(1e-6)
    )[valid]
    if not values.numel():
        raise ValueError("Depth context audit has no valid points.")
    return float(values.mean())


def _normalized(value: torch.Tensor) -> torch.Tensor:
    from streaming_couping.src.semantic_map import normalize_confidence

    return normalize_confidence(_scalar(value))


def _scalar(value: torch.Tensor) -> torch.Tensor:
    output = value.detach().float().cpu()
    if output.ndim == 4 and output.shape[-1] == 1:
        output = output[..., 0]
    if output.ndim != 3:
        raise ValueError("Expected scalar map [S,H,W] or [S,H,W,1].")
    return output


def _random_control_summary(
    rows: Sequence[dict[str, object]],
) -> dict[str, object]:
    random_rows = [
        row for row in rows if str(row["method"]).startswith("random_topk_seed_")
    ]
    if not random_rows:
        raise ValueError("Context audit requires at least one random seed.")
    keys = (
        "center_error_native",
        "rotation_degrees",
        "pointmap_paired_weighted_rmse",
        "depth_abs_rel",
    )
    output: dict[str, object] = {
        "seeds": [int(row["random_seed"]) for row in random_rows],
        "runs": len(random_rows),
    }
    for key in keys:
        values = torch.tensor([float(row[key]) for row in random_rows])
        output[f"mean_{key}"] = float(values.mean())
        output[f"std_{key}"] = float(values.std(unbiased=False))
    return output


def _decisions(
    *,
    context_rows: Sequence[dict[str, object]],
    consistency_summaries: Sequence[dict[str, object]],
    random_summary: dict[str, object],
) -> dict[str, object]:
    by_method = {str(row["method"]): row for row in context_rows}
    consistency = {str(row["branch"]): row for row in consistency_summaries}
    qk = by_method["qk_topk"]
    recent = by_method["recent_topk"]
    qk_beats_recent = (
        float(qk["center_error_native"]) < float(recent["center_error_native"])
        and float(qk["rotation_degrees"]) < float(recent["rotation_degrees"])
    )
    qk_beats_random = (
        float(qk["center_error_native"])
        < float(random_summary["mean_center_error_native"])
        and float(qk["rotation_degrees"])
        < float(random_summary["mean_rotation_degrees"])
    )
    qk_pose_improved = (
        float(qk["center_gain_vs_raw_percent"]) > 0.0
        and float(qk["rotation_gain_vs_raw_percent"]) > 0.0
    )
    qk_geometry_worse = (
        float(qk["pointmap_gain_vs_full_percent"]) < 0.0
        and float(qk["depth_gain_vs_full_percent"]) < 0.0
    )
    return {
        "qk_pose_improves_over_full": int(qk_pose_improved),
        "qk_pose_beats_recent": int(qk_beats_recent),
        "qk_pose_beats_random_mean": int(qk_beats_random),
        "qk_beats_recent_and_random_pose_controls": int(
            qk_pose_improved and qk_beats_recent and qk_beats_random
        ),
        "headwise_context_conflict_observed": int(
            qk_pose_improved and qk_geometry_worse
        ),
        "qk_reprojection_median_delta_px": float(
            consistency["qk_pose"]["mean_frame_reprojection_median_px"]
        )
        - float(consistency["raw_pose"]["mean_frame_reprojection_median_px"]),
        "qk_relative_depth_median_delta": float(
            consistency["qk_pose"]["mean_frame_relative_depth_median"]
        )
        - float(consistency["raw_pose"]["mean_frame_relative_depth_median"]),
        "pose_pointmap_compatibility_claim": 0,
        "compatibility_role": "continuous_diagnostic_no_posthoc_threshold",
        "formal_v0_pose_modified": 0,
        "formal_v0_pointmap_modified": 0,
        "sam_pose_probe_run": 0,
        "next_gate": (
            "set_pose_pointmap_compatibility_threshold_then_sam_identity_probe"
            if qk_pose_improved and qk_beats_recent and qk_beats_random
            else "do_not_add_sam_optimizer_reassess_qk_retrieval"
        ),
    }


def _write_copyable(
    path: Path,
    *,
    summary: dict[str, object],
    output_dir: Path,
) -> None:
    consistency = summary["pose_pointmap_consistency"]
    context = summary["context_policy_results"]
    lines = [
        "===== COPYABLE_V0_INCREMENTAL_AUDIT_BEGIN =====",
        f"revision={REVISION}",
        f"clip={summary['clip']}",
        f"stream_frames={summary['stream_frames']}",
        f"reference_frame={summary['reference_frame']}",
        f"evaluated_pose_frames={summary['evaluated_pose_frames']}",
        f"total_history_budget={summary['total_history_budget']}",
        f"anchor_frames={summary['anchor_frames']}",
        "random_seeds=" + " ".join(str(value) for value in summary["random_seeds"]),
        "model_trained=0",
        "sam_used_by_context_candidates=0",
        "gt_role=scoring_only_after_each_rgb_only_candidate_is_generated",
        "",
        "branch,frames,reprojection_median_px,reprojection_p90_px,relative_depth_median,relative_depth_p90,positive_z_rate,in_bounds_rate",
    ]
    for row in consistency:
        lines.append(
            ",".join(
                str(value)
                for value in (
                    row["branch"],
                    row["valid_frames"],
                    row["mean_frame_reprojection_median_px"],
                    row["mean_frame_reprojection_p90_px"],
                    row["mean_frame_relative_depth_median"],
                    row["mean_frame_relative_depth_p90"],
                    row["mean_positive_z_rate"],
                    row["mean_in_bounds_rate"],
                )
            )
        )
    lines.extend(
        [
            "",
            "method,history_policy,random_seed,center_error_native,rotation_degrees,center_gain_vs_raw_percent,rotation_gain_vs_raw_percent,pointmap_paired_weighted_rmse,pointmap_gain_vs_full_percent,depth_abs_rel,depth_gain_vs_full_percent",
        ]
    )
    columns = (
        "method",
        "history_policy",
        "random_seed",
        "center_error_native",
        "rotation_degrees",
        "center_gain_vs_raw_percent",
        "rotation_gain_vs_raw_percent",
        "pointmap_paired_weighted_rmse",
        "pointmap_gain_vs_full_percent",
        "depth_abs_rel",
        "depth_gain_vs_full_percent",
    )
    for row in context:
        lines.append(",".join(str(row[column]) for column in columns))
    lines.extend(
        [
            "",
            "decision=" + json.dumps(summary["decisions"], sort_keys=True),
            "",
            "outputs:",
            f"summary={output_dir / 'incremental_audit_summary.json'}",
            f"consistency_csv={output_dir / 'pose_pointmap_consistency.csv'}",
            f"context_csv={output_dir / 'context_policy_summary.csv'}",
            f"pointmap_csv={output_dir / 'context_pointmap_frames.csv'}",
            f"retrieval_csv={output_dir / 'context_retrieval_frames.csv'}",
            f"pose_outputs={output_dir / 'context_pose_outputs.pt'}",
            f"copyable_report={path}",
            "===== COPYABLE_V0_INCREMENTAL_AUDIT_END =====",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}.")
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_qk_retrieval_rows(qk_path: Path) -> list[dict[str, object]]:
    """Normalize the locked QK artifact's selection log for this audit."""

    path = qk_path.parent / "retrieval_diagnostics.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing locked QK retrieval diagnostics: {path}."
        )
    output: list[dict[str, object]] = []
    with path.open("r", encoding="utf8", newline="") as handle:
        for row in csv.DictReader(handle):
            selected = str(row["selected_sequence_indices"]).strip()
            output.append(
                {
                    "method": "qk_topk",
                    "random_seed": "",
                    "sequence_index": int(row["sequence_index"]),
                    "frame_index": int(row["frame_index"]),
                    "history_frames": int(row["history_frames"]),
                    "selected_history_count": (
                        len(selected.split()) if selected else 0
                    ),
                    "selected_sequence_indices": selected,
                    "selected_frame_indices": str(
                        row["selected_frame_indices"]
                    ).strip(),
                    "native_qk_ranked_sequence_indices": str(
                        row["qk_ranked_sequence_indices"]
                    ).strip(),
                }
            )
    if not output:
        raise ValueError(f"Locked QK retrieval diagnostics are empty: {path}.")
    return output


def _gain(raw: float, candidate: float) -> float:
    if not math.isfinite(float(raw)) or float(raw) <= 0.0:
        raise ValueError("Raw metric must be finite and positive.")
    return 100.0 * (float(raw) - float(candidate)) / float(raw)


def _space(values: Sequence[int]) -> str:
    return " ".join(str(int(value)) for value in values)


def _load_run(path: str | Path) -> AuditRun:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf8") as handle:
        raw = yaml.safe_load(handle) or {}
    section = raw.get("incremental_audit", {})
    seeds = tuple(int(value) for value in section.get("random_seeds", (0, 1, 2)))
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("incremental_audit requires at least three unique random seeds.")
    budget = int(section.get("total_history_budget", 5))
    anchors = int(section.get("anchor_frames", 1))
    if budget < 2 or not 1 <= anchors < budget:
        raise ValueError("Invalid incremental_audit history budget/anchors.")
    threshold = float(section.get("confidence_threshold", 0.30))
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("incremental_audit.confidence_threshold must be in [0,1].")
    return AuditRun(
        source_path=source,
        output_dir=Path(
            section.get(
                "output_dir",
                "outputs/streaming_couping_v0/incremental_audit",
            )
        ).expanduser().resolve(),
        total_history_budget=budget,
        anchor_frames=anchors,
        random_seeds=seeds,
        confidence_threshold=threshold,
    )


def _validate_inputs(
    payload: dict[str, Any],
    *,
    qk: dict[str, Any],
    clip: ClipConfig,
) -> None:
    required = (
        "stream_images",
        "frame_indices",
        "image_size",
        "reference_sequence_index",
        "baseline_pose_encoding",
        "target_pose_encoding",
        "baseline_depth",
        "baseline_depth_confidence",
        "baseline_world_points",
        "baseline_world_confidence",
        "target_depth",
        "target_world_points",
        "point_alignment_scale",
        "point_alignment_rotation",
        "point_alignment_translation",
    )
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError(f"V0 cache lacks incremental-audit fields: {missing}.")
    frames = tuple(int(value) for value in payload["frame_indices"])
    if frames != clip.frame_indices:
        raise ValueError("V0 cache frame order differs from config.")
    qk_frames = tuple(int(value) for value in qk.get("frame_indices", ()))
    if qk_frames != frames:
        raise ValueError("QK artifact frame order differs from V0 cache.")
    sequence = len(frames)
    expected_dense = tuple(payload["baseline_world_points"].shape)
    for name in (
        "selected_world_to_camera",
        "raw_world_to_camera",
        "selected_depth",
        "selected_pointmap",
    ):
        if name not in qk or not torch.is_tensor(qk[name]):
            raise ValueError(f"QK artifact lacks tensor field {name!r}.")
    if qk["selected_world_to_camera"].shape != (1, sequence, 3, 4):
        raise ValueError("QK selected_world_to_camera shape is invalid.")
    if qk["raw_world_to_camera"].shape != (1, sequence, 3, 4):
        raise ValueError("QK raw_world_to_camera shape is invalid.")
    if tuple(qk["selected_pointmap"].shape) != expected_dense:
        raise ValueError("QK selected_pointmap shape differs from raw pointmap.")
    if not bool(torch.isfinite(qk["selected_world_to_camera"]).all()):
        raise ValueError("QK pose contains non-finite values.")


def _find_clip(config: LearnedPoseConfig, name: str) -> ClipConfig:
    selected = [clip for clip in config.clips if clip.name == name]
    if len(selected) != 1:
        raise ValueError(f"Clip {name!r} was not found exactly once.")
    return selected[0]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="streaming_couping/configs/v0_baseline.yaml",
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--streamvggt-devices")
    return parser.parse_args()


if __name__ == "__main__":
    main()
