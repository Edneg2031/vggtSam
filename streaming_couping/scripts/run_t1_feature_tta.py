#!/usr/bin/env python3
"""Full-history teacher to masked-history LoRA student TTA.

The three stages are intentionally separate processes:

``prepare`` opens V0 once and exports a strict non-GT artifact;
``adapt`` can open only that artifact while updating LoRA and freezing candidates;
``score`` hashes the candidates before target geometry is introduced.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

from streaming_couping.src.backbones.streamvggt_latent import (
    ensure_thwc,
    load_streamvggt_latent_model,
)
from streaming_couping.src.backbones.streamvggt_parallel import (
    LayerShardedStreamVGGT,
    assert_processed_key_cache_equivalence,
)
from streaming_couping.src.config import load_config
from streaming_couping.src.external_repos import maybe_add_repo_to_path
from streaming_couping.src.feature_tta import (
    LoRAConfig,
    assert_only_lora_trainable,
    clip_gradients,
    cosine_feature_loss,
    deterministic_history_keep,
    global_patch_tokens,
    gradient_norm,
    history_for_replay_frame,
    inject_global_attention_lora,
    lora_parameters,
    lora_state_dict,
    reset_lora_modules,
    shuffled_teacher_index,
)
from streaming_couping.src.learned_pose.baseline_runtime import (
    load_baseline_run_config,
)
from streaming_couping.src.learned_pose.cache import cache_path, load_feature_cache
from streaming_couping.src.learned_pose.config import ClipConfig, load_learned_pose_config
from streaming_couping.src.pointmap_alignment import _paired_limit, _robust_similarity
from streaming_couping.src.semantic_map import normalize_confidence


REVISION = "t1_full_history_teacher_masked_history_lora_tta_r1"
CLEAN_ROLE = "strict_non_gt_feature_tta_input"
CANDIDATE_ROLE = "frozen_non_gt_lora_pointmap_candidates"
TRAINED_BRANCHES = ("correct_teacher_lora", "shuffled_teacher_lora")
CANDIDATE_BRANCHES = (
    "zero_update_lora",
    "correct_teacher_lora",
    "shuffled_teacher_lora",
)
SCORE_BRANCHES = (
    "raw_full_history",
    "correct_teacher_lora",
    "shuffled_teacher_lora",
    "zero_update_lora",
)


@dataclass(frozen=True)
class T1Run:
    source_path: Path
    v0_config: Path
    output_dir: Path
    adaptation_frame_count: int
    history_drop_probability: float
    seed: int
    epochs: int
    learning_rate: float
    weight_decay: float
    grad_clip_norm: float
    lora: LoRAConfig
    confidence_threshold: float
    alignment_max_points: int
    paired_max_points_per_frame: int
    symmetric_max_points_per_frame: int
    minimum_improved_frames: int
    zero_update_point_rmse_tolerance: float
    zero_update_confidence_rmse_tolerance: float

    @property
    def clean_path(self) -> Path:
        return self.output_dir / "clean_tta_input.pt"

    @property
    def candidate_path(self) -> Path:
        return self.output_dir / "candidate_pointmaps.pt"

    @property
    def lora_path(self) -> Path:
        return self.output_dir / "lora_states.pt"

    @property
    def adaptation_path(self) -> Path:
        return self.output_dir / "adaptation_summary.json"


def main() -> None:
    args = _parse_args()
    run = _load_run(args.config)
    if args.stage == "prepare":
        _prepare(run)
    elif args.stage == "adapt":
        _adapt(run, streamvggt_devices=args.streamvggt_devices)
    elif args.stage == "score":
        _score(run)
    else:
        raise ValueError(f"Unsupported T1 stage {args.stage!r}.")


def _prepare(run: T1Run) -> None:
    data = load_learned_pose_config(run.v0_config)
    baseline = load_baseline_run_config(run.v0_config)
    clip = _find_clip(data.clips, baseline.clip_name)
    payload_path = cache_path(data, clip)
    payload = load_feature_cache(payload_path)
    qk_path = _qk_artifact_path(run.v0_config)
    qk = torch.load(qk_path, map_location="cpu", weights_only=False)
    _validate_v0_prepare_inputs(payload, qk, clip, run)

    frames = tuple(int(value) for value in payload["frame_indices"])
    evaluation_frames = tuple(int(value) for value in baseline.evaluation_frames)
    evaluation_indices = tuple(frames.index(frame) for frame in evaluation_frames)
    adaptation_indices = tuple(range(run.adaptation_frame_count))
    expected_evaluation = tuple(range(run.adaptation_frame_count, len(frames)))
    if evaluation_indices != expected_evaluation:
        raise ValueError(
            "T1 requires the frozen V0 temporal split: first 18 adaptation, "
            "last 12 evaluation frames."
        )

    levels = payload["token_levels"].detach().float().cpu()
    if levels.ndim != 4:
        raise ValueError(f"token_levels must be [L,S,T,2C], got {levels.shape}.")
    layer_indices = tuple(int(value) for value in payload["dpt_layer_indices"])
    if layer_indices != run.lora.layer_indices:
        raise ValueError("T1 LoRA layers must match the frozen DPT feature layers.")
    if levels.shape[0] != len(layer_indices) or levels.shape[1] != len(frames):
        raise ValueError("Frozen token levels do not match layer/frame metadata.")
    patch_start = int(payload["patch_start_idx"])
    channels = int(levels.shape[-1])
    if channels % 2 or not 0 <= patch_start < int(levels.shape[-2]):
        raise ValueError("Frozen token layout is not frame/global concatenation.")
    teacher = levels[
        :, : run.adaptation_frame_count, patch_start:, channels // 2 :
    ].half().contiguous()

    recovery = load_config(data.recovery_config)
    clean = {
        "schema": 1,
        "revision": REVISION,
        "artifact_role": CLEAN_ROLE,
        "clip": payload["clip_name"],
        "frame_indices": frames,
        "reference_sequence_index": int(payload["reference_sequence_index"]),
        "adaptation_sequence_indices": adaptation_indices,
        "evaluation_sequence_indices": evaluation_indices,
        "image_size": tuple(int(value) for value in payload["image_size"]),
        "patch_start_idx": patch_start,
        "patch_shape": tuple(int(value) for value in payload["patch_shape"]),
        "dpt_layer_indices": layer_indices,
        "stream_images": payload["stream_images"].detach().float().cpu().contiguous(),
        "teacher_global_patch_tokens": teacher,
        "teacher_source": "frozen_v0_full_causal_history_streamvggt_features",
        "teacher_storage_dtype": "float16",
        "model_spec": {
            "repo_path": str(recovery.streamvggt_repo),
            "checkpoint_path": str(recovery.streamvggt_checkpoint),
            "amp_dtype": str(data.streamvggt_amp_dtype),
        },
        "qk_pose_artifact": str(qk_path),
        "qk_pose_sha256": _sha256_file(qk_path),
        "candidate_generation_gt_fields": 0,
        "candidate_generation_raw_pointmap_fields": 0,
        "sam_candidate_inputs": 0,
    }
    _validate_clean_artifact(clean, run)
    run.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(clean, run.clean_path)
    print("T1 CLEAN NON-GT FEATURE ARTIFACT")
    print(
        f"  frames={len(frames)} adaptation={len(adaptation_indices)} "
        f"evaluation={len(evaluation_indices)}"
    )
    print(
        "  teacher="
        f"{tuple(teacher.shape)} dtype=float16 images=float32 GT=0 raw_pointmap=0"
    )
    print(f"  qk_sha256={clean['qk_pose_sha256'][:12]} unchanged")
    print(f"  result={run.clean_path}")


def _adapt(run: T1Run, *, streamvggt_devices: str) -> None:
    clean = torch.load(run.clean_path, map_location="cpu", weights_only=False)
    _validate_clean_artifact(clean, run)
    clean_sha = _sha256_file(run.clean_path)
    devices = tuple(
        value.strip() for value in streamvggt_devices.split(",") if value.strip()
    )
    if len(devices) < 2:
        raise ValueError("T1 adaptation requires two logical CUDA devices.")

    spec = clean["model_spec"]
    maybe_add_repo_to_path(Path(spec["repo_path"]))
    assert_processed_key_cache_equivalence()
    model = load_streamvggt_latent_model(
        repo_path=spec["repo_path"],
        checkpoint_path=spec["checkpoint_path"],
        device="cpu",
        strict=True,
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    runner = LayerShardedStreamVGGT(
        model,
        devices,
        selected_layer_indices=run.lora.layer_indices,
        amp_dtype=str(spec["amp_dtype"]),
    )
    modules = inject_global_attention_lora(
        runner.aggregator,
        run.lora,
        seed=run.seed,
    )
    model.eval()
    assert_only_lora_trainable(model)
    parameters = lora_parameters(modules)
    trainable_count = sum(int(value.numel()) for value in parameters)
    print("T1 MASKED-HISTORY LoRA TEST-TIME ADAPTATION")
    print("  clean artifact only: RGB + teacher features; GT=0 raw_pointmap=0 SAM=0")
    print(
        f"  LoRA layers={run.lora.layer_indices} rank={run.lora.rank} "
        f"trainable={trainable_count}"
    )
    for line in runner.layout_summary():
        print(f"  {line}")

    # Branch D: a freshly inserted, never-updated LoRA must be functionally
    # identical to the frozen model. Its strict numeric check happens in score.
    zero_state = lora_state_dict(modules)
    candidates: dict[str, dict[str, torch.Tensor]] = {
        "zero_update_lora": _replay_full_history_candidate(runner, clean)
    }
    states: dict[str, dict[str, torch.Tensor]] = {
        "zero_update_lora": zero_state
    }
    diagnostics: dict[str, Any] = {}
    history_rows: list[dict[str, Any]] = []

    for branch in TRAINED_BRANCHES:
        reset_lora_modules(modules, seed=run.seed)
        runner.reset()
        branch_diagnostics, branch_rows = _adapt_branch(
            runner,
            clean,
            run=run,
            parameters=parameters,
            branch=branch,
        )
        diagnostics[branch] = branch_diagnostics
        history_rows.extend(branch_rows)
        states[branch] = lora_state_dict(modules)
        candidates[branch] = _replay_full_history_candidate(runner, clean)
        print(
            f"  {branch} loss "
            f"{branch_diagnostics['initial_epoch_mean_loss']:.6f} -> "
            f"{branch_diagnostics['final_epoch_mean_loss']:.6f} "
            f"updates={branch_diagnostics['optimizer_steps']}"
        )

    torch.save(
        {
            "schema": 1,
            "revision": REVISION,
            "artifact_role": "frozen_lora_states_not_formal_v0",
            "clean_artifact_sha256": clean_sha,
            "branches": states,
        },
        run.lora_path,
    )
    lora_sha = _sha256_file(run.lora_path)
    candidate = {
        "schema": 1,
        "revision": REVISION,
        "artifact_role": CANDIDATE_ROLE,
        "clip": clean["clip"],
        "frame_indices": clean["frame_indices"],
        "saved_sequence_indices": (
            int(clean["reference_sequence_index"]),
            *tuple(int(value) for value in clean["evaluation_sequence_indices"]),
        ),
        "clean_artifact_sha256": clean_sha,
        "qk_pose_sha256_before_adaptation": clean["qk_pose_sha256"],
        "lora_states": str(run.lora_path),
        "lora_states_sha256": lora_sha,
        "candidate_generation_fields": (
            "stream_images",
            "full_history_teacher_features",
            "deterministic_history_dropout",
        ),
        "candidate_generation_gt_fields": 0,
        "candidate_generation_raw_pointmap_fields": 0,
        "sam_candidate_inputs": 0,
        "camera_head_run": 0,
        "depth_head_run": 0,
        "point_head_run": 1,
        "branches": candidates,
    }
    _validate_candidate_artifact(candidate, clean)
    torch.save(candidate, run.candidate_path)
    adaptation_summary = {
        "schema": 1,
        "revision": REVISION,
        "clip": clean["clip"],
        "gt_fields_read": 0,
        "raw_pointmap_fields_read": 0,
        "sam_inputs": 0,
        "teacher": "frozen_full_causal_history",
        "student": "masked_causal_history",
        "student_history_context_gradient": "stop_gradient_truncated_bptt",
        "final_candidate_history": "full_causal_history",
        "loss": "mean_cosine_distance_current_global_patch_tokens",
        "lora": {
            "layers": run.lora.layer_indices,
            "projections": ("qkv", "proj"),
            "rank": run.lora.rank,
            "alpha": run.lora.alpha,
            "dropout": run.lora.dropout,
            "trainable_parameters": trainable_count,
        },
        "optimization": {
            "epochs": run.epochs,
            "prefixes_per_epoch": run.adaptation_frame_count - 1,
            "expected_steps": run.epochs * (run.adaptation_frame_count - 1),
            "learning_rate": run.learning_rate,
            "weight_decay": run.weight_decay,
            "grad_clip_norm": run.grad_clip_norm,
            "history_drop_probability": run.history_drop_probability,
            "seed": run.seed,
        },
        "branches": diagnostics,
        "clean_artifact": str(run.clean_path),
        "clean_artifact_sha256": clean_sha,
        "lora_states": str(run.lora_path),
        "lora_states_sha256": lora_sha,
        "candidate_artifact": str(run.candidate_path),
        "candidate_artifact_sha256": _sha256_file(run.candidate_path),
    }
    _write_json(run.adaptation_path, adaptation_summary)
    _write_csv(run.output_dir / "history_dropout.csv", history_rows)
    runner.reset()
    print("  candidates and LoRA states frozen; scoring must run separately")
    print(f"  result={run.candidate_path}")


def _adapt_branch(
    runner: LayerShardedStreamVGGT,
    clean: Mapping[str, Any],
    *,
    run: T1Run,
    parameters: list[torch.nn.Parameter],
    branch: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if branch not in TRAINED_BRANCHES:
        raise ValueError(f"Unknown trained branch {branch!r}.")
    images = clean["stream_images"]
    teacher = clean["teacher_global_patch_tokens"]
    layer_indices = tuple(int(value) for value in clean["dpt_layer_indices"])
    optimizer = torch.optim.AdamW(
        parameters,
        lr=run.learning_rate,
        weight_decay=run.weight_decay,
        foreach=False,
    )
    initial_parameters = [value.detach().float().cpu().clone() for value in parameters]
    epoch_losses: list[float] = []
    all_grad_norms: list[float] = []
    rows: list[dict[str, Any]] = []
    first_nonzero_gradient = 0
    steps = 0
    for epoch in range(run.epochs):
        losses = []
        for prefix in range(1, run.adaptation_frame_count):
            selection_seed = run.seed + epoch * 100003 + prefix * 1009
            retained = deterministic_history_keep(
                prefix,
                drop_probability=run.history_drop_probability,
                seed=selection_seed,
            )
            runner.reset()

            def selector(frame_index: int, _scores: torch.Tensor) -> tuple[int, ...]:
                return history_for_replay_frame(frame_index, retained)

            selected = None
            for frame in range(prefix + 1):
                batch = images[frame : frame + 1].unsqueeze(0)
                if frame < prefix:
                    with torch.no_grad():
                        historical = runner.aggregate_frame(
                            batch,
                            frame,
                            history_selector=selector,
                        )
                    del historical
                else:
                    selected = runner.aggregate_frame(
                        batch,
                        frame,
                        history_selector=selector,
                    )
            if selected is None:
                raise RuntimeError("Masked-history replay did not emit current tokens.")
            teacher_prefix = (
                prefix
                if branch == "correct_teacher_lora"
                else shuffled_teacher_index(
                    prefix, run.adaptation_frame_count - 1
                )
            )
            layer_losses = []
            for level_position, layer_index in enumerate(layer_indices):
                student_tokens = global_patch_tokens(
                    selected,
                    layer_index=layer_index,
                    patch_start_idx=int(clean["patch_start_idx"]),
                )
                target_tokens = teacher[level_position, teacher_prefix]
                layer_loss = cosine_feature_loss(student_tokens, target_tokens)
                layer_losses.append(layer_loss.to(runner.first_device))
            loss = torch.stack(layer_losses).mean()
            if not bool(torch.isfinite(loss).detach().cpu()):
                raise RuntimeError(
                    f"Non-finite T1 feature loss at {branch} epoch={epoch} "
                    f"prefix={prefix}."
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            raw_norm = gradient_norm(parameters)
            if not math.isfinite(raw_norm):
                raise RuntimeError(
                    f"Non-finite T1 LoRA gradient at {branch} epoch={epoch} "
                    f"prefix={prefix}."
                )
            if steps == 0:
                first_nonzero_gradient = int(raw_norm > 0.0)
                if not first_nonzero_gradient:
                    raise RuntimeError("LoRA gradient smoke failed on the first update.")
            clip_gradients(parameters, run.grad_clip_norm)
            optimizer.step()
            value = float(loss.detach().cpu())
            losses.append(value)
            all_grad_norms.append(raw_norm)
            rows.append(
                {
                    "branch": branch,
                    "epoch": epoch,
                    "prefix_sequence_index": prefix,
                    "prefix_frame_index": int(clean["frame_indices"][prefix]),
                    "teacher_sequence_index": teacher_prefix,
                    "teacher_frame_index": int(
                        clean["frame_indices"][teacher_prefix]
                    ),
                    "retained_history_sequence_indices": _space(retained),
                    "retained_history_frame_indices": _space(
                        clean["frame_indices"][index] for index in retained
                    ),
                    "retained_history_count": len(retained),
                    "available_history_count": prefix,
                    "loss": value,
                    "gradient_norm_before_clip": raw_norm,
                }
            )
            steps += 1
            runner.reset()
            del selected, loss, layer_losses
        epoch_losses.append(sum(losses) / len(losses))
    expected = run.epochs * (run.adaptation_frame_count - 1)
    if steps != expected:
        raise RuntimeError(f"Expected {expected} optimizer steps, got {steps}.")
    update_norm = math.sqrt(
        sum(
            float(
                (
                    parameter.detach().float().cpu() - initial
                ).square().sum()
            )
            for parameter, initial in zip(parameters, initial_parameters)
        )
    )
    if not math.isfinite(update_norm) or update_norm <= 0.0:
        raise RuntimeError(f"T1 branch {branch} did not update its LoRA parameters.")
    return {
        "optimizer_steps": steps,
        "first_update_nonzero_lora_gradient": first_nonzero_gradient,
        "lora_parameter_update_norm": update_norm,
        "initial_epoch_mean_loss": epoch_losses[0],
        "final_epoch_mean_loss": epoch_losses[-1],
        "feature_loss_decreased": int(epoch_losses[-1] < epoch_losses[0]),
        "epoch_mean_losses": epoch_losses,
        "maximum_gradient_norm_before_clip": max(all_grad_norms),
    }, rows


@torch.inference_mode()
def _replay_full_history_candidate(
    runner: LayerShardedStreamVGGT,
    clean: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    runner.reset()
    images = clean["stream_images"]
    saved = (
        int(clean["reference_sequence_index"]),
        *tuple(int(value) for value in clean["evaluation_sequence_indices"]),
    )
    saved_set = set(saved)
    points: dict[int, torch.Tensor] = {}
    confidences: dict[int, torch.Tensor] = {}
    for frame in range(int(images.shape[0])):
        batch = images[frame : frame + 1].unsqueeze(0)
        selected = runner.aggregate_frame(batch, frame)
        if frame in saved_set:
            point_output, confidence_output = runner.points(selected, batch)
            point = ensure_thwc(point_output[0]).detach().float().cpu()
            confidence = ensure_thwc(confidence_output[0]).detach().float().cpu()
            if point.shape[0] != 1 or point.shape[-1] != 3:
                raise ValueError(f"Unexpected point output {tuple(point.shape)}.")
            if confidence.shape[0] != 1 or confidence.shape[-1] != 1:
                raise ValueError(
                    f"Unexpected confidence output {tuple(confidence.shape)}."
                )
            points[frame] = point[0]
            confidences[frame] = confidence[0, ..., 0]
            del point_output, confidence_output
        del selected
    runner.reset()
    if tuple(points) != saved or tuple(confidences) != saved:
        raise RuntimeError("Full-history replay did not save the requested frames.")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "world_points": torch.stack([points[index] for index in saved]),
        "confidence": torch.stack([confidences[index] for index in saved]),
    }


def _score(run: T1Run) -> None:
    # Freeze/integrity checks happen before V0 target geometry is opened.
    candidate_sha = _sha256_file(run.candidate_path)
    candidate = torch.load(run.candidate_path, map_location="cpu", weights_only=False)
    clean = torch.load(run.clean_path, map_location="cpu", weights_only=False)
    _validate_clean_artifact(clean, run)
    _validate_candidate_artifact(candidate, clean)
    if candidate["clean_artifact_sha256"] != _sha256_file(run.clean_path):
        raise ValueError("Clean TTA artifact changed after candidate generation.")
    if candidate["lora_states_sha256"] != _sha256_file(run.lora_path):
        raise ValueError("Frozen LoRA state artifact changed before scoring.")

    data = load_learned_pose_config(run.v0_config)
    baseline = load_baseline_run_config(run.v0_config)
    clip = _find_clip(data.clips, baseline.clip_name)
    qk_path = _qk_artifact_path(run.v0_config)
    qk_sha_after = _sha256_file(qk_path)
    qk_unchanged = int(qk_sha_after == clean["qk_pose_sha256"])
    if not qk_unchanged:
        raise RuntimeError("Formal V0 QK pose artifact changed during T1.")

    payload_path = cache_path(data, clip)
    payload = load_feature_cache(payload_path)
    _validate_score_inputs(payload, candidate, clean)
    print("T1 candidates frozen and hashed; target geometry enters scoring now")
    summary, frame_rows = _score_candidates(
        run,
        payload=payload,
        clean=clean,
        candidate=candidate,
        candidate_sha=candidate_sha,
        cache_path_value=payload_path,
        qk_path=qk_path,
        qk_sha_after=qk_sha_after,
    )
    _write_outputs(run, summary, frame_rows)
    correct = summary["branch_lookup"]["correct_teacher_lora"]
    print("T1 FULL-HISTORY TEACHER / MASKED-HISTORY STUDENT RESULT")
    print(
        f"  correct RMSE gain={correct['rmse_gain_vs_raw_percent']:.4f}% "
        f"P90 gain={correct['p90_gain_vs_raw_percent']:.4f}% "
        f"frames={correct['improved_frames_vs_raw']}/12"
    )
    print(
        f"  zero_equivalence={summary['decision']['zero_update_equivalence_pass']} "
        f"decision={summary['decision']['t1_decision']}"
    )
    print(f"  result={run.output_dir / 'summary.json'}")


def _score_candidates(
    run: T1Run,
    *,
    payload: Mapping[str, Any],
    clean: Mapping[str, Any],
    candidate: Mapping[str, Any],
    candidate_sha: str,
    cache_path_value: Path,
    qk_path: Path,
    qk_sha_after: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    saved = tuple(int(value) for value in candidate["saved_sequence_indices"])
    evaluation_indices = tuple(int(value) for value in clean["evaluation_sequence_indices"])
    if saved != (int(clean["reference_sequence_index"]), *evaluation_indices):
        raise ValueError("Candidate saved-frame order is invalid.")
    target_all = payload["target_world_points"].detach().float().cpu()
    raw_points_all = payload["baseline_world_points"].detach().float().cpu()
    raw_confidence_all = payload["baseline_world_confidence"].detach().float().cpu()
    target = target_all[list(evaluation_indices)]
    branches: dict[str, dict[str, torch.Tensor]] = {
        "raw_full_history": {
            "reference": raw_points_all[int(clean["reference_sequence_index"])],
            "world_points": raw_points_all[list(evaluation_indices)],
            "reference_confidence": raw_confidence_all[
                int(clean["reference_sequence_index"])
            ],
            "confidence": raw_confidence_all[list(evaluation_indices)],
        }
    }
    for name in CANDIDATE_BRANCHES:
        value = candidate["branches"][name]
        confidence = normalize_confidence(value["confidence"].detach().float().cpu())
        points = value["world_points"].detach().float().cpu()
        branches[name] = {
            "reference": points[0],
            "world_points": points[1:],
            "reference_confidence": confidence[0],
            "confidence": confidence[1:],
        }

    target_reference = target_all[int(clean["reference_sequence_index"])]
    aligned = {}
    alignments = {}
    for name, value in branches.items():
        aligned[name], alignments[name] = _align_from_reference(
            value["reference"],
            value["reference_confidence"],
            value["world_points"],
            target_reference,
            confidence_threshold=run.confidence_threshold,
            max_points=run.alignment_max_points,
        )
    support = torch.isfinite(target).all(dim=-1)
    for value in branches.values():
        support &= (
            torch.isfinite(value["world_points"]).all(dim=-1)
            & torch.isfinite(value["confidence"])
            & (value["confidence"] >= run.confidence_threshold)
        )
    if any(int(support[index].sum()) == 0 for index in range(len(evaluation_indices))):
        raise ValueError("T1 common evaluation support is empty for a frame.")

    branch_rows = []
    frame_rows: list[dict[str, Any]] = []
    per_frame_lookup: dict[str, dict[int, dict[str, Any]]] = {}
    for name in SCORE_BRANCHES:
        metrics, per_frame = _pointmap_metrics(
            aligned[name],
            target,
            support,
            paired_max_points=run.paired_max_points_per_frame,
            symmetric_max_points=run.symmetric_max_points_per_frame,
        )
        lookup = {}
        for row in per_frame:
            sequence_index = evaluation_indices[int(row["evaluation_position"])]
            complete = {
                "branch": name,
                "sequence_index": sequence_index,
                "frame_index": int(clean["frame_indices"][sequence_index]),
                **row,
            }
            lookup[sequence_index] = complete
            frame_rows.append(complete)
        per_frame_lookup[name] = lookup
        branch_rows.append(
            {
                "branch": name,
                **metrics,
                **{f"alignment_{key}": value for key, value in alignments[name].items()},
            }
        )

    raw = next(row for row in branch_rows if row["branch"] == "raw_full_history")
    for row in branch_rows:
        name = row["branch"]
        row["rmse_gain_vs_raw_percent"] = _gain(raw["paired_rmse"], row["paired_rmse"])
        row["median_gain_vs_raw_percent"] = _gain(
            raw["paired_median"], row["paired_median"]
        )
        row["p90_gain_vs_raw_percent"] = _gain(raw["paired_p90"], row["paired_p90"])
        row["improved_frames_vs_raw"] = sum(
            int(per_frame_lookup[name][index]["paired_rmse"]
                < per_frame_lookup["raw_full_history"][index]["paired_rmse"])
            for index in evaluation_indices
        )
        row["improved_frame_ratio_vs_raw"] = row["improved_frames_vs_raw"] / len(
            evaluation_indices
        )
    lookup = {row["branch"]: row for row in branch_rows}

    zero_points = candidate["branches"]["zero_update_lora"]["world_points"].float()
    raw_saved = raw_points_all[list(saved)]
    point_valid = torch.isfinite(zero_points).all(dim=-1) & torch.isfinite(raw_saved).all(dim=-1)
    point_difference = torch.linalg.vector_norm(
        zero_points[point_valid] - raw_saved[point_valid], dim=-1
    )
    zero_confidence = normalize_confidence(
        candidate["branches"]["zero_update_lora"]["confidence"].float()
    )
    raw_confidence_saved = raw_confidence_all[list(saved)]
    confidence_valid = torch.isfinite(zero_confidence) & torch.isfinite(raw_confidence_saved)
    confidence_difference = (
        zero_confidence[confidence_valid] - raw_confidence_saved[confidence_valid]
    )
    zero_point_rmse = _rmse(point_difference)
    zero_confidence_rmse = _rmse(confidence_difference)
    zero_pass = int(
        zero_point_rmse <= run.zero_update_point_rmse_tolerance
        and zero_confidence_rmse <= run.zero_update_confidence_rmse_tolerance
    )
    correct = lookup["correct_teacher_lora"]
    shuffled = lookup["shuffled_teacher_lora"]
    go = int(
        zero_pass
        and float(correct["paired_rmse"]) < float(raw["paired_rmse"])
        and float(correct["paired_p90"]) <= float(raw["paired_p90"])
        and int(correct["improved_frames_vs_raw"]) >= run.minimum_improved_frames
        and float(correct["paired_rmse"]) < float(shuffled["paired_rmse"])
    )
    decision = {
        "t1_decision": "GO" if go else "NO_GO",
        "zero_update_equivalence_pass": zero_pass,
        "formal_qk_artifact_sha256_unchanged": 1,
        "correct_rmse_beats_raw": int(correct["paired_rmse"] < raw["paired_rmse"]),
        "correct_p90_not_worse_than_raw": int(correct["paired_p90"] <= raw["paired_p90"]),
        "correct_frame_majority_pass": int(
            correct["improved_frames_vs_raw"] >= run.minimum_improved_frames
        ),
        "correct_rmse_beats_shuffled": int(
            correct["paired_rmse"] < shuffled["paired_rmse"]
        ),
        "formal_v0_pose_modified": 0,
        "formal_v0_pointmap_modified": 0,
        "formal_v0_semantic_map_modified": 0,
        "next_gate": (
            "optimize_lora_placement_without_gt"
            if go
            else "stop_feature_consistency_tta_keep_formal_v0"
        ),
    }
    adaptation = json.loads(run.adaptation_path.read_text(encoding="utf8"))
    summary = {
        "schema": 1,
        "revision": REVISION,
        "experiment": "full_history_teacher_masked_history_student_lora_tta",
        "baseline_version": "v0",
        "baseline_status": "frozen_unchanged",
        "clip": clean["clip"],
        "adaptation_frames": tuple(
            int(clean["frame_indices"][index])
            for index in clean["adaptation_sequence_indices"]
        ),
        "evaluation_frames": tuple(
            int(clean["frame_indices"][index]) for index in evaluation_indices
        ),
        "branches": branch_rows,
        "branch_lookup": lookup,
        "teacher": "frozen_full_causal_history_intermediate_features",
        "student_adaptation_history": "anchor_plus_deterministic_random_history_dropout",
        "student_evaluation_history": "full_causal_history",
        "loss": "current_global_patch_token_cosine_consistency",
        "candidate_generation_gt_fields": 0,
        "candidate_generation_raw_pointmap_fields": 0,
        "gt_role": "reference_alignment_and_scoring_after_candidates_frozen_and_hashed",
        "sam_candidate_inputs": 0,
        "base_model_parameters_updated": 0,
        "lora_parameters_updated": 1,
        "camera_head_run": 0,
        "depth_head_run": 0,
        "point_head_run": 1,
        "common_supported_points": int(support.sum()),
        "zero_update": {
            "pointmap_native_rmse": zero_point_rmse,
            "pointmap_native_maximum": float(point_difference.max()),
            "confidence_native_rmse": zero_confidence_rmse,
            "pointmap_rmse_tolerance": run.zero_update_point_rmse_tolerance,
            "confidence_rmse_tolerance": run.zero_update_confidence_rmse_tolerance,
        },
        "adaptation_diagnostics": adaptation["branches"],
        "clean_artifact": str(run.clean_path),
        "candidate_artifact": str(run.candidate_path),
        "candidate_artifact_sha256_before_gt": candidate_sha,
        "lora_states": str(run.lora_path),
        "v0_cache": str(cache_path_value),
        "qk_pose_artifact": str(qk_path),
        "qk_pose_sha256_before": clean["qk_pose_sha256"],
        "qk_pose_sha256_after": qk_sha_after,
        "decision": decision,
        "claim": (
            "feature_consistency_tta_improves_heldout_dense_pointmap"
            if go
            else "feature_consistency_tta_pointmap_improvement_not_established"
        ),
    }
    return summary, frame_rows


def _align_from_reference(
    reference: torch.Tensor,
    reference_confidence: torch.Tensor,
    points: torch.Tensor,
    target_reference: torch.Tensor,
    *,
    confidence_threshold: float,
    max_points: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    source = reference.reshape(-1, 3)
    truth = target_reference.reshape(-1, 3)
    weights = reference_confidence.reshape(-1)
    valid = (
        torch.isfinite(source).all(dim=-1)
        & torch.isfinite(truth).all(dim=-1)
        & torch.isfinite(weights)
        & (weights >= float(confidence_threshold))
    )
    source_fit, truth_fit = _paired_limit(
        source[valid], truth[valid], max_points=int(max_points)
    )
    scale, rotation, translation, inliers, fit_rmse = _robust_similarity(
        source_fit, truth_fit, min_points=128
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
    errors = []
    symmetric_values = []
    rows = []
    for position in range(int(predicted.shape[0])):
        selected = torch.nonzero(support[position].reshape(-1), as_tuple=False)[:, 0]
        paired = _even_limit(selected, paired_max_points)
        pred = predicted[position].reshape(-1, 3).index_select(0, paired)
        truth = target[position].reshape(-1, 3).index_select(0, paired)
        error = torch.linalg.vector_norm(pred - truth, dim=-1)
        errors.append(error)
        symmetric_index = _even_limit(selected, symmetric_max_points)
        symmetric = _symmetric_mean(
            predicted[position].reshape(-1, 3).index_select(0, symmetric_index),
            target[position].reshape(-1, 3).index_select(0, symmetric_index),
        )
        symmetric_values.append(symmetric)
        rows.append(
            {
                "evaluation_position": position,
                "supported_points": int(selected.numel()),
                "paired_points": int(error.numel()),
                "paired_rmse": _rmse(error),
                "paired_median": float(error.median()),
                "paired_p90": float(torch.quantile(error, 0.90)),
                "symmetric_mean": symmetric,
            }
        )
    joined = torch.cat(errors)
    return {
        "supported_points": int(support.sum()),
        "paired_rmse": _rmse(joined),
        "paired_median": float(joined.median()),
        "paired_p90": float(torch.quantile(joined, 0.90)),
        "symmetric_mean": float(torch.tensor(symmetric_values).mean()),
    }, rows


def _symmetric_mean(first: torch.Tensor, second: torch.Tensor) -> float:
    distance = torch.cdist(first.float(), second.float())
    return float(
        0.5
        * (
            distance.min(dim=1).values.mean()
            + distance.min(dim=0).values.mean()
        )
    )


def _validate_v0_prepare_inputs(
    payload: Mapping[str, Any],
    qk: Mapping[str, Any],
    clip: ClipConfig,
    run: T1Run,
) -> None:
    required = {
        "stream_images",
        "token_levels",
        "dpt_layer_indices",
        "frame_indices",
        "clip_name",
        "reference_sequence_index",
        "image_size",
        "patch_start_idx",
        "patch_shape",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"V0 cache lacks T1 prepare fields: {sorted(missing)}.")
    if tuple(int(value) for value in payload["frame_indices"]) != clip.frame_indices:
        raise ValueError("V0 cache frame order differs from the frozen clip.")
    if payload["clip_name"] != clip.name:
        raise ValueError("V0 cache clip identity is inconsistent.")
    if qk.get("selected_pose_branch") != "retrieve_qk":
        raise ValueError("T1 requires the frozen retrieve_qk pose artifact.")
    if tuple(int(value) for value in qk.get("frame_indices", ())) != clip.frame_indices:
        raise ValueError("QK pose frame order differs from the frozen clip.")
    if len(clip.frame_indices) != 30 or run.adaptation_frame_count != 18:
        raise ValueError("T1 r1 protocol is fixed to an 18/12 split of 30 frames.")


def _validate_clean_artifact(clean: Mapping[str, Any], run: T1Run) -> None:
    required = {
        "schema",
        "revision",
        "artifact_role",
        "clip",
        "frame_indices",
        "reference_sequence_index",
        "adaptation_sequence_indices",
        "evaluation_sequence_indices",
        "image_size",
        "patch_start_idx",
        "patch_shape",
        "dpt_layer_indices",
        "stream_images",
        "teacher_global_patch_tokens",
        "teacher_source",
        "teacher_storage_dtype",
        "model_spec",
        "qk_pose_artifact",
        "qk_pose_sha256",
        "candidate_generation_gt_fields",
        "candidate_generation_raw_pointmap_fields",
        "sam_candidate_inputs",
    }
    if set(clean) != required:
        raise ValueError(
            "Clean T1 artifact field whitelist mismatch: "
            f"missing={sorted(required.difference(clean))} "
            f"extra={sorted(set(clean).difference(required))}."
        )
    if clean["schema"] != 1 or clean["revision"] != REVISION:
        raise ValueError("Unsupported clean T1 artifact schema/revision.")
    if clean["artifact_role"] != CLEAN_ROLE:
        raise ValueError("Input is not the strict clean T1 artifact.")
    if any(
        int(clean[name]) != 0
        for name in (
            "candidate_generation_gt_fields",
            "candidate_generation_raw_pointmap_fields",
            "sam_candidate_inputs",
        )
    ):
        raise ValueError("Clean T1 artifact contains a forbidden input class.")
    frames = tuple(int(value) for value in clean["frame_indices"])
    adaptation = tuple(int(value) for value in clean["adaptation_sequence_indices"])
    evaluation = tuple(int(value) for value in clean["evaluation_sequence_indices"])
    if adaptation != tuple(range(run.adaptation_frame_count)):
        raise ValueError("Clean artifact adaptation split is not fixed first-18.")
    if evaluation != tuple(range(run.adaptation_frame_count, len(frames))):
        raise ValueError("Clean artifact evaluation split is not fixed last-12.")
    images = torch.as_tensor(clean["stream_images"])
    teacher = torch.as_tensor(clean["teacher_global_patch_tokens"])
    if images.ndim != 4 or images.shape[0] != len(frames) or images.shape[1] != 3:
        raise ValueError("Clean stream_images must have shape [S,3,H,W].")
    expected_patches = int(clean["patch_shape"][0]) * int(clean["patch_shape"][1])
    if teacher.shape[:3] != (
        len(run.lora.layer_indices),
        run.adaptation_frame_count,
        expected_patches,
    ):
        raise ValueError("Clean teacher feature shape is inconsistent.")
    if tuple(int(value) for value in clean["dpt_layer_indices"]) != run.lora.layer_indices:
        raise ValueError("Clean teacher layers differ from the T1 LoRA layers.")


def _validate_candidate_artifact(
    candidate: Mapping[str, Any], clean: Mapping[str, Any]
) -> None:
    if candidate.get("schema") != 1 or candidate.get("revision") != REVISION:
        raise ValueError("Unsupported T1 candidate schema/revision.")
    if candidate.get("artifact_role") != CANDIDATE_ROLE:
        raise ValueError("Input is not a frozen T1 candidate artifact.")
    for name in (
        "candidate_generation_gt_fields",
        "candidate_generation_raw_pointmap_fields",
        "sam_candidate_inputs",
        "camera_head_run",
        "depth_head_run",
    ):
        if int(candidate.get(name, -1)) != 0:
            raise ValueError(f"T1 candidate violates {name}=0.")
    if int(candidate.get("point_head_run", 0)) != 1:
        raise ValueError("T1 candidate must be produced by the point head.")
    if candidate.get("clip") != clean["clip"]:
        raise ValueError("T1 candidate clip differs from clean input.")
    if tuple(candidate.get("frame_indices", ())) != tuple(clean["frame_indices"]):
        raise ValueError("T1 candidate frame order differs from clean input.")
    if candidate.get("qk_pose_sha256_before_adaptation") != clean["qk_pose_sha256"]:
        raise ValueError("T1 candidate does not preserve the clean QK hash.")
    branches = candidate.get("branches", {})
    if tuple(branches) != CANDIDATE_BRANCHES:
        raise ValueError("T1 candidate branches/order are not fixed.")
    saved_count = len(candidate.get("saved_sequence_indices", ()))
    for name in CANDIDATE_BRANCHES:
        if set(branches[name]) != {"world_points", "confidence"}:
            raise ValueError(f"T1 candidate branch {name} has extra fields.")
        points = torch.as_tensor(branches[name]["world_points"])
        confidence = torch.as_tensor(branches[name]["confidence"])
        if points.ndim != 4 or points.shape[0] != saved_count or points.shape[-1] != 3:
            raise ValueError(f"T1 candidate point shape is invalid for {name}.")
        if confidence.shape != points.shape[:-1]:
            raise ValueError(f"T1 candidate confidence shape is invalid for {name}.")


def _validate_score_inputs(
    payload: Mapping[str, Any],
    candidate: Mapping[str, Any],
    clean: Mapping[str, Any],
) -> None:
    for name in (
        "target_world_points",
        "baseline_world_points",
        "baseline_world_confidence",
        "frame_indices",
        "clip_name",
    ):
        if name not in payload:
            raise ValueError(f"V0 scoring cache lacks {name!r}.")
    if payload["clip_name"] != candidate["clip"]:
        raise ValueError("V0 scoring cache clip differs from frozen candidates.")
    if tuple(int(value) for value in payload["frame_indices"]) != tuple(
        clean["frame_indices"]
    ):
        raise ValueError("V0 scoring frame order differs from frozen candidates.")


def _write_outputs(
    run: T1Run, summary: dict[str, Any], frame_rows: list[dict[str, Any]]
) -> None:
    _write_json(run.output_dir / "summary.json", summary)
    _write_csv(run.output_dir / "branch_summary.csv", summary["branches"])
    _write_csv(run.output_dir / "frame_metrics.csv", frame_rows)
    _write_copyable(run.output_dir / "copyable_result.txt", summary)


def _write_copyable(path: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "===== COPYABLE_T1_FEATURE_TTA_BEGIN =====",
        f"revision={summary['revision']}",
        f"clip={summary['clip']}",
        f"adaptation_frames={len(summary['adaptation_frames'])}",
        f"evaluation_frames={len(summary['evaluation_frames'])}",
        "branches=" + ",".join(row["branch"] for row in summary["branches"]),
        f"teacher={summary['teacher']}",
        f"student_adaptation_history={summary['student_adaptation_history']}",
        f"student_evaluation_history={summary['student_evaluation_history']}",
        f"loss={summary['loss']}",
        "candidate_generation_gt_fields=0",
        "candidate_generation_raw_pointmap_fields=0",
        "sam_candidate_inputs=0",
        "base_model_parameters_updated=0",
        "lora_parameters_updated=1",
        "camera_head_run=0",
        "formal_v0_modified=0",
        "",
        "branch,supported_points,paired_rmse,paired_median,paired_p90,symmetric_mean,rmse_gain_vs_raw_percent,median_gain_vs_raw_percent,p90_gain_vs_raw_percent,improved_frames_vs_raw",
    ]
    fields = (
        "branch",
        "supported_points",
        "paired_rmse",
        "paired_median",
        "paired_p90",
        "symmetric_mean",
        "rmse_gain_vs_raw_percent",
        "median_gain_vs_raw_percent",
        "p90_gain_vs_raw_percent",
        "improved_frames_vs_raw",
    )
    for row in summary["branches"]:
        lines.append(",".join(str(row[field]) for field in fields))
    zero = summary["zero_update"]
    lines.extend(
        (
            "",
            f"zero_update_pointmap_native_rmse={zero['pointmap_native_rmse']}",
            f"zero_update_confidence_native_rmse={zero['confidence_native_rmse']}",
            "decision=" + json.dumps(summary["decision"], sort_keys=True),
            f"claim={summary['claim']}",
            "outputs:",
            f"summary={path.with_name('summary.json')}",
            f"branch_csv={path.with_name('branch_summary.csv')}",
            f"frame_csv={path.with_name('frame_metrics.csv')}",
            f"history_csv={path.with_name('history_dropout.csv')}",
            f"adaptation_summary={path.with_name('adaptation_summary.json')}",
            f"candidate_artifact={summary['candidate_artifact']}",
            f"lora_states={summary['lora_states']}",
            "===== COPYABLE_T1_FEATURE_TTA_END =====",
        )
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def _load_run(path: str | Path) -> T1Run:
    source = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf8")) or {}
    adaptation = raw.get("adaptation", {})
    evaluation = raw.get("evaluation", {})
    run = T1Run(
        source_path=source,
        v0_config=_resolve(raw.get("v0_config", "streaming_couping/configs/v0_baseline.yaml")),
        output_dir=_resolve(raw.get("output_dir", "outputs/streaming_couping_t1_feature_tta")),
        adaptation_frame_count=int(adaptation.get("adaptation_frame_count", 18)),
        history_drop_probability=float(adaptation.get("history_drop_probability", 0.5)),
        seed=int(adaptation.get("seed", 0)),
        epochs=int(adaptation.get("epochs", 5)),
        learning_rate=float(adaptation.get("learning_rate", 1e-4)),
        weight_decay=float(adaptation.get("weight_decay", 0.0)),
        grad_clip_norm=float(adaptation.get("grad_clip_norm", 1.0)),
        lora=LoRAConfig(
            layer_indices=tuple(int(value) for value in adaptation.get("lora_layers", (4, 11, 17, 23))),
            rank=int(adaptation.get("lora_rank", 4)),
            alpha=float(adaptation.get("lora_alpha", 4.0)),
            dropout=float(adaptation.get("lora_dropout", 0.0)),
        ),
        confidence_threshold=float(evaluation.get("confidence_threshold", 0.30)),
        alignment_max_points=int(evaluation.get("alignment_max_points", 30000)),
        paired_max_points_per_frame=int(evaluation.get("paired_max_points_per_frame", 8192)),
        symmetric_max_points_per_frame=int(evaluation.get("symmetric_max_points_per_frame", 512)),
        minimum_improved_frames=int(evaluation.get("minimum_improved_frames", 7)),
        zero_update_point_rmse_tolerance=float(evaluation.get("zero_update_point_rmse_tolerance", 1e-4)),
        zero_update_confidence_rmse_tolerance=float(evaluation.get("zero_update_confidence_rmse_tolerance", 1e-4)),
    )
    run.lora.validate(depth=24)
    if run.v0_config.name != "v0_baseline.yaml":
        raise ValueError("T1 must consume the frozen V0 configuration.")
    fixed_protocol = {
        "adaptation_frame_count": (run.adaptation_frame_count, 18),
        "history_drop_probability": (run.history_drop_probability, 0.5),
        "seed": (run.seed, 0),
        "epochs": (run.epochs, 5),
        "learning_rate": (run.learning_rate, 1e-4),
        "weight_decay": (run.weight_decay, 0.0),
        "grad_clip_norm": (run.grad_clip_norm, 1.0),
        "lora_layers": (run.lora.layer_indices, (4, 11, 17, 23)),
        "lora_rank": (run.lora.rank, 4),
        "lora_alpha": (run.lora.alpha, 4.0),
        "lora_dropout": (run.lora.dropout, 0.0),
        "confidence_threshold": (run.confidence_threshold, 0.30),
        "minimum_improved_frames": (run.minimum_improved_frames, 7),
        "zero_update_point_rmse_tolerance": (
            run.zero_update_point_rmse_tolerance,
            1e-4,
        ),
        "zero_update_confidence_rmse_tolerance": (
            run.zero_update_confidence_rmse_tolerance,
            1e-4,
        ),
    }
    changed = {
        name: actual
        for name, (actual, expected) in fixed_protocol.items()
        if actual != expected
    }
    if changed:
        raise ValueError(f"T1 r1 fixed protocol was changed: {changed}.")
    if min(
        run.alignment_max_points,
        run.paired_max_points_per_frame,
        run.symmetric_max_points_per_frame,
    ) < 128:
        raise ValueError("T1 evaluation sample budgets must be at least 128.")
    return run


def _qk_artifact_path(config: Path) -> Path:
    raw = yaml.safe_load(config.read_text(encoding="utf8")) or {}
    return _resolve(raw["baseline"]["pose"]["qk_pose_output"])


def _find_clip(clips: tuple[ClipConfig, ...], name: str) -> ClipConfig:
    selected = [clip for clip in clips if clip.name == name]
    if len(selected) != 1:
        raise ValueError(f"Expected exactly one clip {name!r}.")
    return selected[0]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    return 100.0 * (float(baseline) - float(candidate)) / max(
        abs(float(baseline)), 1e-12
    )


def _space(values) -> str:
    return " ".join(str(int(value)) for value in values)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
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


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="streaming_couping/configs/t1_feature_tta.yaml",
    )
    parser.add_argument("--stage", choices=("prepare", "adapt", "score"), required=True)
    parser.add_argument("--streamvggt-devices", default="cuda:0,cuda:1")
    return parser.parse_args()


if __name__ == "__main__":
    main()
