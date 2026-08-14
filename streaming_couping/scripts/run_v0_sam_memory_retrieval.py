#!/usr/bin/env python3
"""Probe SAM persistent-instance guidance for frozen StreamVGGT KV retrieval."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml

from streaming_couping.src.backbones.streamvggt_parallel import (
    LayerShardedStreamVGGT,
    assert_frame_repository_cache_equivalence,
    assert_processed_key_cache_equivalence,
)
from streaming_couping.src.backbones.streamvggt_latent import (
    load_streamvggt_latent_model,
)
from streaming_couping.src.config import load_config
from streaming_couping.src.external_repos import maybe_add_repo_to_path
from streaming_couping.src.learned_pose.baseline_runtime import (
    camera_centers,
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
from streaming_couping.src.sam_memory_retrieval import (
    RETRIEVAL_METHODS,
    RetrievalPolicy,
    persistent_visibility,
    qk_rank_diagnostics,
    same_instance_history_frames,
    select_retrieval_history,
)


REVISION = "v0_sam_memory_masked_qk_retrievevggt_probe_r2"
FOLDS = ("short", "medium", "long")


@dataclass(frozen=True)
class RetrievalRun:
    source_path: Path
    output_dir: Path
    methods: tuple[str, ...]
    primary_method: str
    policy: RetrievalPolicy
    minimum_track_score: float
    point_confidence_threshold: float
    raw_equivalence_atol: float


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
    path = cache_path(data, clip)
    payload = load_feature_cache(path)
    _validate_payload(payload, clip=clip)
    recovery = load_config(data.recovery_config)
    if not data.streamvggt_devices:
        raise ValueError("V0 retrieval probe requires layer-sharded StreamVGGT.")
    if not recovery.streaming_cache:
        raise ValueError("V0 retrieval probe requires streaming_cache=true.")

    maybe_add_repo_to_path(recovery.streamvggt_repo)
    assert_processed_key_cache_equivalence()
    assert_frame_repository_cache_equivalence()
    print("V0 retrieval processed-key and frame-repository equivalence passed")
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

    candidate_payload = _candidate_generation_payload(payload)
    visibility = persistent_visibility(
        candidate_payload["tracking_masks_output"],
        candidate_payload["tracking_scores"],
        candidate_payload["sam_track_ids"],
        minimum_score=run.minimum_track_score,
    )
    region_patch_masks = _region_patch_masks(
        candidate_payload["tracking_masks_stream"],
        visibility,
        patch_shape=tuple(
            int(value) for value in candidate_payload["patch_shape"]
        ),
    )
    generation = _generate_candidates(
        runner=runner,
        payload=candidate_payload,
        visibility=visibility,
        region_patch_masks=region_patch_masks,
        methods=run.methods,
        policy=run.policy,
    )
    raw_audit = _raw_equivalence_audit(
        generation["raw_full_history"],
        payload,
        atol=run.raw_equivalence_atol,
    )
    if not int(raw_audit["pass"]):
        raise RuntimeError(f"Raw retrieval replay is not equivalent: {raw_audit}")

    result = _score_and_write(
        payload=payload,
        cache_path_value=path,
        baseline_evaluation_frames=baseline.evaluation_frames,
        run=run,
        generation=generation,
        raw_audit=raw_audit,
    )
    print(f"V0 SAM-memory retrieval result={result}")


@torch.inference_mode()
def _generate_candidates(
    *,
    runner: LayerShardedStreamVGGT,
    payload: dict,
    visibility: torch.Tensor,
    region_patch_masks: torch.Tensor,
    methods: tuple[str, ...],
    policy: RetrievalPolicy,
) -> dict[str, dict[str, Any]]:
    """Generate frozen-model branches without exposing any GT field."""

    images = payload["stream_images"].detach().float().cpu()
    frame_numbers = tuple(int(value) for value in payload["frame_indices"])
    output: dict[str, dict[str, Any]] = {}
    for method in methods:
        print(f"V0 retrieval branch={method}")
        runner.reset()
        pose_encodings = []
        world_points = []
        confidences = []
        retrieval_rows: list[dict[str, object]] = []

        def selector(
            frame_index: int,
            scores: torch.Tensor,
            sam_region_scores: torch.Tensor,
            shuffled_region_scores: torch.Tensor,
        ) -> tuple[int, ...]:
            sam_candidates = same_instance_history_frames(
                visibility,
                frame_index,
            )
            shuffled_candidates = same_instance_history_frames(
                visibility,
                frame_index,
                shuffled_identity=True,
            )
            selected = select_retrieval_history(
                method=method,
                frame_index=frame_index,
                qk_scores=scores,
                sam_region_scores=sam_region_scores,
                shuffled_region_scores=shuffled_region_scores,
                sam_candidates=sam_candidates,
                shuffled_candidates=shuffled_candidates,
                policy=policy,
            )
            diagnostic_candidates = (
                shuffled_candidates
                if method == "shuffled_instance_memory"
                else sam_candidates
            )
            rank = qk_rank_diagnostics(
                scores,
                diagnostic_candidates,
                selected,
            )
            qk_rank = tuple(int(value) for value in rank["qk_ranked_history"])
            sam_region_rank = _finite_score_rank(sam_region_scores)
            shuffled_region_rank = _finite_score_rank(
                shuffled_region_scores
            )
            current_slots = tuple(
                int(value)
                for value in torch.nonzero(
                    visibility[int(frame_index)],
                    as_tuple=False,
                ).flatten().tolist()
            )
            retrieval_rows.append(
                {
                    "method": method,
                    "sequence_index": int(frame_index),
                    "frame_index": frame_numbers[int(frame_index)],
                    "history_frames": int(frame_index),
                    "selected_history_budget": (
                        int(frame_index)
                        if method == "raw_full_history"
                        else min(
                            int(policy.total_frame_budget),
                            int(frame_index),
                        )
                    ),
                    "selected_sequence_indices": _space(selected),
                    "selected_frame_indices": _space(
                        frame_numbers[index] for index in selected
                    ),
                    "qk_ranked_sequence_indices": _space(qk_rank),
                    "qk_ranked_frame_indices": _space(
                        frame_numbers[index] for index in qk_rank
                    ),
                    "qk_scores": " ".join(
                        f"{float(value):.8g}" for value in scores
                    ),
                    "sam_region_scores": _score_text(sam_region_scores),
                    "shuffled_region_scores": _score_text(
                        shuffled_region_scores
                    ),
                    "sam_region_ranked_sequence_indices": _space(
                        sam_region_rank
                    ),
                    "sam_region_ranked_frame_indices": _space(
                        frame_numbers[index] for index in sam_region_rank
                    ),
                    "shuffled_region_ranked_sequence_indices": _space(
                        shuffled_region_rank
                    ),
                    "shuffled_region_ranked_frame_indices": _space(
                        frame_numbers[index]
                        for index in shuffled_region_rank
                    ),
                    "current_visible_slots": _space(current_slots),
                    "current_visible_sam_track_ids": _space(
                        payload["sam_track_ids"][slot]
                        for slot in current_slots
                    ),
                    "current_visible_prompts": " ".join(
                        str(payload.get("sam_track_prompts", ())[slot])
                        for slot in current_slots
                    ),
                    "sam_candidate_sequence_indices": _space(sam_candidates),
                    "sam_candidate_frame_indices": _space(
                        frame_numbers[index] for index in sam_candidates
                    ),
                    "shuffled_candidate_sequence_indices": _space(
                        shuffled_candidates
                    ),
                    "shuffled_candidate_frame_indices": _space(
                        frame_numbers[index] for index in shuffled_candidates
                    ),
                    "diagnostic_candidate_source": (
                        "shuffled_identity"
                        if method == "shuffled_instance_memory"
                        else "persistent_instance"
                    ),
                    "sam_candidate_count": int(rank["sam_candidate_count"]),
                    "sam_candidate_best_qk_rank": int(
                        rank["sam_candidate_best_qk_rank"]
                    ),
                    "sam_candidate_mean_qk_rank": float(
                        rank["sam_candidate_mean_qk_rank"]
                    ),
                    "selected_sam_candidate_count": int(
                        rank["selected_sam_candidate_count"]
                    ),
                }
            )
            return selected

        for frame_index in range(images.shape[0]):
            batch_frame = images[frame_index : frame_index + 1].unsqueeze(0)
            selected_tokens = runner.aggregate_frame(
                batch_frame,
                frame_index,
                history_selector=selector,
                region_patch_masks=region_patch_masks,
            )
            pose_encoding = runner.camera(selected_tokens)
            points, confidence = runner.points(selected_tokens, batch_frame)
            pose_encodings.append(pose_encoding[0, 0].detach().float().cpu())
            world_points.append(points[0, 0].detach().float().cpu())
            confidences.append(confidence[0, 0].detach().float().cpu())
            del selected_tokens, pose_encoding, points, confidence
        output[method] = {
            "pose_encoding": torch.stack(pose_encodings),
            "world_points": torch.stack(world_points),
            "world_confidence": _normalize_confidence(
                torch.stack(confidences)
            ),
            "retrieval_rows": retrieval_rows,
        }
        print(f"V0 retrieval completed method={method}")
    runner.reset()
    return output


def _raw_equivalence_audit(
    raw: dict[str, Any],
    payload: dict,
    *,
    atol: float,
) -> dict[str, object]:
    comparisons = {
        "pose_encoding": (
            raw["pose_encoding"],
            payload["baseline_pose_encoding"],
        ),
        "world_points": (
            raw["world_points"],
            payload["baseline_world_points"],
        ),
        "world_confidence": (
            raw["world_confidence"],
            payload["baseline_world_confidence"],
        ),
    }
    differences = {
        name: float((left.float() - right.float()).abs().max())
        for name, (left, right) in comparisons.items()
    }
    return {
        "atol": float(atol),
        "maximum_absolute_difference": differences,
        "pass": int(all(value <= float(atol) for value in differences.values())),
    }


def _candidate_generation_payload(payload: dict) -> dict[str, Any]:
    """Expose only causal RGB/tracking fields to frozen candidate generation."""

    allowed = (
        "stream_images",
        "tracking_masks_output",
        "tracking_masks_stream",
        "tracking_scores",
        "sam_track_ids",
        "sam_track_prompts",
        "frame_indices",
        "patch_shape",
    )
    deployable = {name: payload[name] for name in allowed}
    if any(name.startswith("target_") for name in deployable):
        raise RuntimeError("Candidate generation payload unexpectedly contains GT.")
    return deployable


def _region_patch_masks(
    masks: torch.Tensor,
    visibility: torch.Tensor,
    *,
    patch_shape: tuple[int, int],
) -> torch.Tensor:
    """Convert causal SAM regions to fractional StreamVGGT patch occupancy."""

    if masks.ndim != 4 or visibility.shape != masks.shape[:2]:
        raise ValueError(
            "Region masks/visibility must be [T,N,H,W] and [T,N]."
        )
    height, width = (int(patch_shape[0]), int(patch_shape[1]))
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid StreamVGGT patch shape {patch_shape}.")
    frames, instances = masks.shape[:2]
    active = masks.detach().float().cpu()
    active = active * visibility.detach().float().cpu()[..., None, None]
    pooled = F.adaptive_avg_pool2d(
        active.reshape(-1, 1, *active.shape[-2:]),
        output_size=(height, width),
    )
    return pooled.reshape(frames, instances, height, width).clamp(0, 1)


def _score_and_write(
    *,
    payload: dict,
    cache_path_value: Path,
    baseline_evaluation_frames: tuple[int, ...],
    run: RetrievalRun,
    generation: dict[str, dict[str, Any]],
    raw_audit: dict[str, object],
) -> Path:
    """Introduce GT only after every frozen candidate has been generated."""

    from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri

    frames = tuple(int(value) for value in payload["frame_indices"])
    positions = {frame: index for index, frame in enumerate(frames)}
    evaluation_indices = [positions[frame] for frame in baseline_evaluation_frames]
    image_size = tuple(int(value) for value in payload["image_size"])
    target_pose, _ = pose_encoding_to_extri_intri(
        payload["target_pose_encoding"].unsqueeze(0).float(),
        image_size_hw=image_size,
    )
    target_points = payload["target_world_points"].float()
    fixed_point_confidence = payload["baseline_world_confidence"].float()
    scale = float(payload["point_alignment_scale"])
    rotation = payload["point_alignment_rotation"].float()
    translation = payload["point_alignment_translation"].float()

    frame_rows: list[dict[str, object]] = []
    method_state: dict[str, dict[str, Any]] = {}
    for method, branch in generation.items():
        predicted_pose, _ = pose_encoding_to_extri_intri(
            branch["pose_encoding"].unsqueeze(0).float(),
            image_size_hw=image_size,
        )
        aligned_points = scale * (
            branch["world_points"].float() @ rotation.T
        ) + translation
        point_rmse = _pointmap_frame_rmse(
            aligned_points,
            target_points,
            fixed_point_confidence,
            confidence_threshold=run.point_confidence_threshold,
        )
        rotation_error, center_error = _pose_frame_errors(
            predicted_pose,
            target_pose,
        )
        method_state[method] = {
            "pose": predicted_pose,
            "point_rmse": point_rmse,
            "rotation_error": rotation_error,
            "center_error": center_error,
        }
        for index, frame in enumerate(frames):
            frame_rows.append(
                {
                    "method": method,
                    "sequence_index": index,
                    "frame_index": frame,
                    "rotation_error_degrees": float(rotation_error[index]),
                    "center_error_native": float(center_error[index]),
                    "pointmap_paired_rmse": float(point_rmse[index]),
                }
            )

    raw = method_state["raw_full_history"]
    fold_rows: list[dict[str, object]] = []
    for method in run.methods:
        state = method_state[method]
        for fold_index, fold in enumerate(FOLDS):
            indices = evaluation_indices[4 * fold_index : 4 * (fold_index + 1)]
            raw_metrics = pose_metrics(
                raw["pose"],
                target_pose,
                reference_index=int(payload["reference_sequence_index"]),
                evaluation_indices=indices,
            )
            metrics = pose_metrics(
                state["pose"],
                target_pose,
                reference_index=int(payload["reference_sequence_index"]),
                evaluation_indices=indices,
            )
            raw_point = _finite_mean(raw["point_rmse"], indices)
            predicted_point = _finite_mean(state["point_rmse"], indices)
            center_gain = _gain(
                raw_metrics["center_error_native"],
                metrics["center_error_native"],
            )
            rotation_gain = _gain(
                raw_metrics["rotation_degrees"],
                metrics["rotation_degrees"],
            )
            point_gain = _gain(raw_point, predicted_point)
            center_worse = _worse_count(
                raw["center_error"],
                state["center_error"],
                indices,
            )
            rotation_worse = _worse_count(
                raw["rotation_error"],
                state["rotation_error"],
                indices,
            )
            point_worse = _worse_count(
                raw["point_rmse"],
                state["point_rmse"],
                indices,
            )
            fold_pass = int(
                method != "raw_full_history"
                and center_gain > 0.0
                and rotation_gain > 0.0
                and point_gain > 0.0
                and center_worse == 0
                and rotation_worse == 0
                and point_worse == 0
            )
            fold_rows.append(
                {
                    "fold": fold,
                    "method": method,
                    "test_frames": _space(frames[index] for index in indices),
                    "raw_center_error_native": raw_metrics[
                        "center_error_native"
                    ],
                    "candidate_center_error_native": metrics[
                        "center_error_native"
                    ],
                    "center_gain_percent": center_gain,
                    "center_worse_frames": center_worse,
                    "raw_rotation_degrees": raw_metrics["rotation_degrees"],
                    "candidate_rotation_degrees": metrics[
                        "rotation_degrees"
                    ],
                    "rotation_gain_percent": rotation_gain,
                    "rotation_worse_frames": rotation_worse,
                    "raw_pointmap_rmse": raw_point,
                    "candidate_pointmap_rmse": predicted_point,
                    "pointmap_gain_percent": point_gain,
                    "pointmap_worse_frames": point_worse,
                    "fold_pass": fold_pass,
                }
            )

    branch_all_fold = {
        method: int(
            all(
                int(row["fold_pass"])
                for row in fold_rows
                if row["method"] == method
            )
        )
        for method in run.methods
        if method != "raw_full_history"
    }
    selection_audit = _selection_intervention_audit(generation)
    hybrid_vs_qk = _all_fold_better(
        method_state["sam_hybrid_qk"],
        method_state["retrieve_qk"],
        evaluation_indices,
    )
    shuffled_damage = _all_fold_better(
        method_state["sam_hybrid_qk"],
        method_state["shuffled_instance_memory"],
        evaluation_indices,
    )
    causal_pass = int(
        branch_all_fold.get(run.primary_method, 0)
        and int(selection_audit["pass"])
        and hybrid_vs_qk
        and shuffled_damage
    )
    decision = {
        "primary_method": run.primary_method,
        "branch_all_fold_pass": branch_all_fold,
        "retrieve_qk_all_fold_pass": branch_all_fold.get("retrieve_qk", 0),
        "sam_hybrid_better_than_qk_all_folds": int(hybrid_vs_qk),
        "shuffled_instance_memory_destroys_gain_all_folds": int(
            shuffled_damage
        ),
        "sam_memory_retrieval_causal_pass": causal_pass,
        "selection_intervention_audit": selection_audit,
        "r1_whole_frame_candidate_gate_identifiable": 0,
        "r1_result": "all_non_raw_branches_identical",
        "selected_pose_modified": 0,
        "candidate_generation_gt_fields": 0,
        "gt_role": "scoring_only_after_all_frozen_branches",
        "claim": (
            "sam_persistent_identity_improves_streamvggt_context"
            if causal_pass
            else "retrieval_probe_only_no_sam_pose_or_pointmap_claim"
        ),
    }

    run.output_dir.mkdir(parents=True, exist_ok=True)
    retrieval_rows = [
        row
        for method in run.methods
        for row in generation[method]["retrieval_rows"]
    ]
    _write_csv(run.output_dir / "retrieval_diagnostics.csv", retrieval_rows)
    _write_csv(run.output_dir / "frame_metrics.csv", frame_rows)
    _write_csv(run.output_dir / "fold_summary.csv", fold_rows)
    summary = {
        "schema": 1,
        "revision": REVISION,
        "baseline_version": "v0",
        "clip": payload["clip_name"],
        "cache": str(cache_path_value),
        "config": str(run.source_path),
        "frames": frames,
        "evaluation_frames": baseline_evaluation_frames,
        "methods": run.methods,
        "primary_method": run.primary_method,
        "total_frame_budget": run.policy.total_frame_budget,
        "anchor_frames": run.policy.anchor_frames,
        "sam_frame_quota": run.policy.sam_frame_quota,
        "sam_role": "persistent_instance_masks_and_identity_for_qk_pooling",
        "streamvggt_role": "native_mask_pooled_first_global_qk_and_whole_frame_kv",
        "pointmap_scoring_support": "fixed_raw_streamvggt_confidence",
        "sam_hidden_features_used": 0,
        "sam_appearance_tokens_used": 0,
        "model_trained": 0,
        "pose_loss_used": 0,
        "raw_replay_equivalence": raw_audit,
        "selection_intervention_audit": selection_audit,
        "decision": decision,
        "fold_results": fold_rows,
    }
    result = run.output_dir / "retrieval_summary.json"
    result.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf8",
    )
    torch.save(
        {
            "revision": REVISION,
            "frame_indices": frames,
            "pose_encoding": {
                method: generation[method]["pose_encoding"]
                for method in run.methods
            },
            "world_to_camera": {
                method: method_state[method]["pose"].detach().cpu()
                for method in run.methods
            },
            "selected_history_indices": {
                method: tuple(
                    row["selected_sequence_indices"]
                    for row in generation[method]["retrieval_rows"]
                )
                for method in run.methods
            },
            "selected_pose_modified": False,
        },
        run.output_dir / "retrieval_outputs.pt",
    )
    _write_copyable_report(
        run.output_dir / "copyable_result.txt",
        payload=payload,
        run=run,
        raw_audit=raw_audit,
        fold_rows=fold_rows,
        decision=decision,
    )
    print("V0 SAM-MEMORY RETRIEVEVGGT PROBE")
    for row in fold_rows:
        if row["method"] == "raw_full_history":
            continue
        print(
            f"  fold={row['fold']} method={row['method']} "
            f"center_gain={float(row['center_gain_percent']):.4f}% "
            f"R_gain={float(row['rotation_gain_percent']):.4f}% "
            f"point_gain={float(row['pointmap_gain_percent']):.4f}% "
            f"pass={row['fold_pass']}"
        )
    print(f"  decision={json.dumps(decision, sort_keys=True)}")
    print(f"  copyable_report={run.output_dir / 'copyable_result.txt'}")
    return result


def _pose_frame_errors(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    relative = (
        predicted[..., :3, :3]
        @ target[..., :3, :3].transpose(-1, -2)
    )
    cosine = (
        torch.diagonal(relative, dim1=-2, dim2=-1).sum(dim=-1) - 1.0
    ) * 0.5
    rotation = torch.rad2deg(torch.acos(cosine.clamp(-1, 1)))[0]
    center = torch.linalg.vector_norm(
        camera_centers(predicted) - camera_centers(target),
        dim=-1,
    )[0]
    return rotation.detach().float().cpu(), center.detach().float().cpu()


def _pointmap_frame_rmse(
    predicted: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
    *,
    confidence_threshold: float,
) -> torch.Tensor:
    output = []
    for index in range(predicted.shape[0]):
        left = predicted[index].reshape(-1, 3)
        right = target[index].reshape(-1, 3)
        weight = confidence[index].reshape(-1)
        valid = (
            torch.isfinite(left).all(dim=-1)
            & torch.isfinite(right).all(dim=-1)
            & torch.isfinite(weight)
            & (weight >= float(confidence_threshold))
        )
        if not bool(valid.any()):
            output.append(torch.tensor(float("nan")))
        else:
            distance = torch.linalg.vector_norm(left[valid] - right[valid], dim=-1)
            output.append(torch.sqrt(distance.square().mean()))
    return torch.stack(output).float().cpu()


def _normalize_confidence(confidence: torch.Tensor) -> torch.Tensor:
    value = torch.nan_to_num(confidence.float(), nan=0.0, posinf=0.0, neginf=0.0)
    if value.ndim == 4 and value.shape[-1] == 1:
        value = value[..., 0]
    flat = value.flatten(1)
    low = torch.quantile(flat, 0.05, dim=1, keepdim=True)
    high = torch.quantile(flat, 0.95, dim=1, keepdim=True)
    return ((flat - low) / (high - low).clamp_min(1e-6)).clamp(0, 1).reshape_as(
        value
    )


def _all_fold_better(
    left: dict[str, Any],
    right: dict[str, Any],
    evaluation_indices: list[int],
) -> bool:
    for fold_index in range(3):
        indices = evaluation_indices[4 * fold_index : 4 * (fold_index + 1)]
        if not (
            _finite_mean(left["center_error"], indices)
            < _finite_mean(right["center_error"], indices)
            and _finite_mean(left["rotation_error"], indices)
            < _finite_mean(right["rotation_error"], indices)
            and _finite_mean(left["point_rmse"], indices)
            < _finite_mean(right["point_rmse"], indices)
        ):
            return False
    return True


def _selection_intervention_audit(
    generation: dict[str, dict[str, Any]],
) -> dict[str, int]:
    selected = {
        method: tuple(
            str(row["selected_sequence_indices"])
            for row in branch["retrieval_rows"]
        )
        for method, branch in generation.items()
    }
    lengths = {len(values) for values in selected.values()}
    if len(lengths) != 1:
        raise ValueError("Retrieval branches produced different frame counts.")

    def difference(left: str, right: str) -> int:
        return sum(
            first != second
            for first, second in zip(selected[left], selected[right])
        )

    hybrid_qk = difference("sam_hybrid_qk", "retrieve_qk")
    hybrid_shuffled = difference(
        "sam_hybrid_qk",
        "shuffled_instance_memory",
    )
    gated_qk = difference("sam_gated_qk", "retrieve_qk")
    return {
        "sam_hybrid_vs_retrieve_qk_different_frames": hybrid_qk,
        "sam_hybrid_vs_shuffled_different_frames": hybrid_shuffled,
        "sam_gated_vs_retrieve_qk_different_frames": gated_qk,
        "pass": int(hybrid_qk > 0 and hybrid_shuffled > 0),
    }


def _finite_mean(values: torch.Tensor, indices: list[int]) -> float:
    selected = values.index_select(0, torch.tensor(indices, dtype=torch.long))
    selected = selected[torch.isfinite(selected)]
    return float(selected.mean()) if selected.numel() else float("nan")


def _gain(raw: float, candidate: float) -> float:
    if not math.isfinite(raw) or not math.isfinite(candidate) or raw <= 0:
        return float("nan")
    return 100.0 * (float(raw) - float(candidate)) / float(raw)


def _worse_count(
    raw: torch.Tensor,
    candidate: torch.Tensor,
    indices: list[int],
) -> int:
    index = torch.tensor(indices, dtype=torch.long)
    left = raw.index_select(0, index)
    right = candidate.index_select(0, index)
    valid = torch.isfinite(left) & torch.isfinite(right)
    return int((right[valid] > left[valid]).sum())


def _write_copyable_report(
    path: Path,
    *,
    payload: dict,
    run: RetrievalRun,
    raw_audit: dict[str, object],
    fold_rows: list[dict[str, object]],
    decision: dict[str, object],
) -> None:
    columns = tuple(fold_rows[0])
    lines = [
        "===== COPYABLE_V0_SAM_MEMORY_RETRIEVAL_BEGIN =====",
        f"revision={REVISION}",
        f"clip={payload['clip_name']}",
        f"methods={','.join(run.methods)}",
        f"primary_method={run.primary_method}",
        f"total_frame_budget={run.policy.total_frame_budget}",
        f"anchor_frames={run.policy.anchor_frames}",
        f"sam_frame_quota={run.policy.sam_frame_quota}",
        "sam_role=persistent_instance_masks_and_identity_for_qk_pooling",
        "sam_hidden_features_used=0",
        "retrieval_score=native_streamvggt_qk_pooled_inside_same_id_masks",
        "streamvggt_context=whole_frame_native_kv",
        "pointmap_scoring_support=fixed_raw_streamvggt_confidence",
        "model_trained=0",
        "candidate_generation_gt_fields=0",
        f"raw_replay_equivalence={json.dumps(raw_audit, sort_keys=True)}",
        "",
        ",".join(columns),
    ]
    for row in fold_rows:
        lines.append(
            ",".join(_csv_text(row.get(column, "")) for column in columns)
        )
    lines.extend(
        (
            "",
            f"decision={json.dumps(decision, sort_keys=True)}",
            "",
            "outputs:",
            f"summary={path.parent / 'retrieval_summary.json'}",
            f"fold_csv={path.parent / 'fold_summary.csv'}",
            f"frame_csv={path.parent / 'frame_metrics.csv'}",
            f"retrieval_csv={path.parent / 'retrieval_diagnostics.csv'}",
            f"copyable_report={path}",
            "===== COPYABLE_V0_SAM_MEMORY_RETRIEVAL_END =====",
        )
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def _load_run(path: str | Path) -> RetrievalRun:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf8") as handle:
        raw = yaml.safe_load(handle) or {}
    section = raw.get("sam_memory_retrieval", {})
    policy = RetrievalPolicy(
        total_frame_budget=int(section.get("total_frame_budget", 5)),
        anchor_frames=int(section.get("anchor_frames", 1)),
        sam_frame_quota=int(section.get("sam_frame_quota", 2)),
    )
    policy.validate()
    methods = tuple(str(value) for value in section.get("methods", ()))
    if methods != RETRIEVAL_METHODS:
        raise ValueError(
            "sam_memory_retrieval.methods must contain the locked five methods "
            f"in order: {RETRIEVAL_METHODS}."
        )
    primary = str(section.get("primary_method", "sam_hybrid_qk"))
    if primary != "sam_hybrid_qk":
        raise ValueError("The locked primary method is sam_hybrid_qk.")
    return RetrievalRun(
        source_path=source,
        output_dir=Path(
            section.get(
                "output_dir",
                "outputs/streaming_couping_v0/sam_memory_retrieval",
            )
        ).expanduser().resolve(),
        methods=methods,
        primary_method=primary,
        policy=policy,
        minimum_track_score=float(section.get("minimum_track_score", 0.5)),
        point_confidence_threshold=float(
            section.get("point_confidence_threshold", 0.3)
        ),
        raw_equivalence_atol=float(section.get("raw_equivalence_atol", 2e-4)),
    )


def _validate_payload(payload: dict, *, clip: ClipConfig) -> None:
    required = (
        "stream_images",
        "tracking_masks_output",
        "tracking_masks_stream",
        "tracking_scores",
        "sam_track_ids",
        "sam_track_prompts",
        "baseline_pose_encoding",
        "baseline_world_points",
        "baseline_world_confidence",
        "target_pose_encoding",
        "target_world_points",
        "point_alignment_scale",
        "point_alignment_rotation",
        "point_alignment_translation",
        "patch_shape",
    )
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError(f"V0 cache lacks retrieval fields: {missing}.")
    if str(payload.get("instance_source")) != "sam31_online":
        raise ValueError("V0 SAM-memory retrieval requires sam31_online tracks.")
    if tuple(int(value) for value in payload["frame_indices"]) != clip.frame_indices:
        raise ValueError("V0 retrieval cache frames differ from config.")
    if bool(payload.get("cache_sam_appearance", True)):
        raise ValueError("V0 retrieval must not read cached SAM appearance tokens.")


def _find_clip(config: LearnedPoseConfig, name: str) -> ClipConfig:
    selected = [clip for clip in config.clips if clip.name == name]
    if len(selected) != 1:
        raise ValueError(f"Clip {name!r} was not found exactly once.")
    return selected[0]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = tuple(rows[0])
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(
            {column: row.get(column, "") for column in columns}
            for row in rows
        )


def _space(values) -> str:
    return " ".join(str(int(value)) for value in values)


def _finite_score_rank(scores: torch.Tensor) -> tuple[int, ...]:
    return tuple(
        sorted(
            (
                index
                for index in range(scores.numel())
                if bool(torch.isfinite(scores[index]))
            ),
            key=lambda index: (-float(scores[index]), index),
        )
    )


def _score_text(scores: torch.Tensor) -> str:
    return " ".join(
        "nan" if not bool(torch.isfinite(value)) else f"{float(value):.8g}"
        for value in scores
    )


def _csv_text(value: object) -> str:
    text = str(value)
    if any(character in text for character in (",", '"', "\n")):
        return '"' + text.replace('"', '""') + '"'
    return text


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
