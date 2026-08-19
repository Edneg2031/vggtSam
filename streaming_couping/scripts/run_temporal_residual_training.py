#!/usr/bin/env python3
"""Single-scene development test for trust-aware pointmap residual learning.

This is deliberately a supervised temporal split, not a cross-scene claim.
The frozen V0 cache is read-only; only three small external heads are trained.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import gc
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import torch
import yaml

from streaming_couping.src.learned_pose.baseline_runtime import (
    load_baseline_run_config,
)
from streaming_couping.src.learned_pose.cache import cache_path, load_feature_cache
from streaming_couping.src.learned_pose.config import ClipConfig, load_learned_pose_config
from streaming_couping.src.semantic_map import normalize_confidence
from streaming_couping.src.storage import expand_storage_path
from streaming_couping.src.trust_aware_residual import (
    TrustAwareResidualHead,
    apply_similarity,
    heteroscedastic_point_loss,
    invert_similarity,
    point_head_patch_features,
    robust_point_loss,
    validation_checkpoint_is_better,
)


REVISION = "phase1_temporal_trust_aware_pointmap_residual_r2"
LEARNED_BRANCHES = (
    "residual_no_gate",
    "gated_residual",
    "gated_residual_uncertainty",
)
SCORE_BRANCHES = ("raw_full_history", *LEARNED_BRANCHES)


@dataclass(frozen=True)
class BranchSpec:
    use_gate: bool
    use_uncertainty: bool


BRANCH_SPECS = {
    "residual_no_gate": BranchSpec(False, False),
    "gated_residual": BranchSpec(True, False),
    "gated_residual_uncertainty": BranchSpec(True, True),
}


@dataclass(frozen=True)
class RunConfig:
    source_path: Path
    v0_config: Path
    output_dir: Path
    device: str
    seed: int
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    projection_channels: int
    hidden_channels: int
    gate_bias: float
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    grad_clip_norm: float
    point_beta: float
    correction_regularization_weight: float
    uncertainty_weight: float
    confidence_threshold: float
    maximum_points_per_frame: int
    minimum_improved_test_frames: int
    uncertainty_minimum_spearman: float


@dataclass
class ExperimentData:
    clip: str
    frame_indices: tuple[int, ...]
    reference_sequence_index: int
    layer_indices: tuple[int, ...]
    patch_shape: tuple[int, int]
    image_size: tuple[int, int]
    features: torch.Tensor
    raw_native: torch.Tensor
    target_native: torch.Tensor
    target_metric: torch.Tensor
    confidence: torch.Tensor
    support: torch.Tensor
    scale: float
    rotation: torch.Tensor
    translation: torch.Tensor
    cache_path: Path


def main() -> None:
    args = _parse_args()
    run = _load_run(args.config, device_override=args.device)
    _seed_everything(run.seed)
    run.output_dir.mkdir(parents=True, exist_ok=True)
    data, qk_path, qk_sha_before = _load_experiment_data(run)
    _validate_split(run, len(data.frame_indices))
    device = _resolve_device(run.device)
    print("PHASE 1 R2 BEST-VALIDATION POINTMAP RESIDUAL TRAINING")
    print(
        f"  clip={data.clip} frames={len(data.frame_indices)} "
        f"split={len(run.train_indices)}/{len(run.validation_indices)}/{len(run.test_indices)}"
    )
    print(
        f"  device={device} features={tuple(data.features.shape)} "
        f"patch={data.patch_shape} dense={data.image_size}"
    )
    print("  frozen=StreamVGGT backbone, PointHead, CameraHead; SAM=0")
    print("  GT role=supervised train labels, validation monitoring, sealed test scoring")

    training_rows: list[dict[str, Any]] = []
    risk_rows: list[dict[str, Any]] = []
    states: dict[str, Any] = {}
    selection_summaries: dict[str, dict[str, Any]] = {}

    for branch in LEARNED_BRANCHES:
        spec = BRANCH_SPECS[branch]
        _seed_everything(run.seed)
        model = TrustAwareResidualHead(
            feature_channels=int(data.features.shape[-1]),
            level_count=int(data.features.shape[1]),
            patch_shape=data.patch_shape,
            projection_channels=run.projection_channels,
            hidden_channels=run.hidden_channels,
            use_gate=spec.use_gate,
            use_uncertainty=spec.use_uncertainty,
            gate_bias=run.gate_bias,
        ).to(device)
        zero_error = _zero_update_error(model, data, run, device)
        if zero_error != 0.0:
            raise RuntimeError(
                f"Protected raw fallback is not exact for {branch}: {zero_error}."
            )
        print(
            f"  training {branch} gate={int(spec.use_gate)} "
            f"uncertainty={int(spec.use_uncertainty)}"
        )
        curve, best_state, selection = _train_branch(
            model, branch, data, run, device
        )
        training_rows.extend(curve)
        selection_summaries[branch] = selection
        states[branch] = {
            "spec": asdict(spec),
            "state_dict": best_state,
            "zero_update_native_maximum": zero_error,
            "checkpoint_selection": selection,
        }
        print(
            f"    selected best-validation epoch={selection['best_epoch'] + 1} "
            f"RMSE={selection['best_validation_rmse']:.6f} "
            f"P90={selection['best_validation_p90']:.6f}"
        )
        model.to("cpu")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("  all best-validation checkpoints frozen; opening one-pass test scoring")
    raw_rows, raw_frames = _score_raw(data, run)
    branch_rows = list(raw_rows)
    frame_rows = list(raw_frames)
    branch_lookup: dict[str, dict[str, dict[str, Any]]] = {
        "validation": {
            row["branch"]: row for row in branch_rows if row["split"] == "validation"
        },
        "test": {row["branch"]: row for row in branch_rows if row["split"] == "test"},
    }
    for branch in LEARNED_BRANCHES:
        spec = BRANCH_SPECS[branch]
        selection = selection_summaries[branch]
        model = TrustAwareResidualHead(
            feature_channels=int(data.features.shape[-1]),
            level_count=int(data.features.shape[1]),
            patch_shape=data.patch_shape,
            projection_channels=run.projection_channels,
            hidden_channels=run.hidden_channels,
            use_gate=spec.use_gate,
            use_uncertainty=spec.use_uncertainty,
            gate_bias=run.gate_bias,
        ).to(device)
        model.load_state_dict(states[branch]["state_dict"], strict=True)
        for split, indices in (
            ("validation", run.validation_indices),
            ("test", run.test_indices),
        ):
            raw = branch_lookup[split]["raw_full_history"]
            row, rows, uncertainty, candidate_errors, raw_errors = _score_model(
                model,
                branch=branch,
                split=split,
                indices=indices,
                data=data,
                run=run,
                device=device,
                raw_frame_rows=raw_frames,
            )
            row["rmse_gain_vs_raw_percent"] = _gain(raw["rmse"], row["rmse"])
            row["median_gain_vs_raw_percent"] = _gain(
                raw["median"], row["median"]
            )
            row["p90_gain_vs_raw_percent"] = _gain(raw["p90"], row["p90"])
            row["selected_checkpoint_epoch"] = selection["best_epoch"]
            row["selected_checkpoint_epoch_one_based"] = selection["best_epoch"] + 1
            if split == "validation":
                equivalent = int(
                    math.isclose(
                        row["rmse"],
                        selection["best_validation_rmse"],
                        rel_tol=0.0,
                        abs_tol=1e-6,
                    )
                    and math.isclose(
                        row["median"],
                        selection["best_validation_median"],
                        rel_tol=0.0,
                        abs_tol=1e-6,
                    )
                    and math.isclose(
                        row["p90"],
                        selection["best_validation_p90"],
                        rel_tol=0.0,
                        abs_tol=1e-6,
                    )
                )
                if not equivalent:
                    raise RuntimeError(
                        f"Reloaded best-validation metrics changed for {branch}."
                    )
                selection["reloaded_validation_equivalence_pass"] = equivalent
            branch_rows.append(row)
            frame_rows.extend(rows)
            branch_lookup[split][branch] = row
            if uncertainty is not None:
                correlation = _spearman(uncertainty, candidate_errors)
                row["uncertainty_error_spearman"] = correlation
                coverage_rows = _risk_coverage(
                    uncertainty,
                    candidate_errors,
                    raw_errors,
                    branch=branch,
                    split=split,
                )
                risk_rows.extend(coverage_rows)
                half = next(item for item in coverage_rows if item["coverage"] == 0.5)
                row["half_coverage_rmse"] = half["candidate_rmse"]
                row["half_coverage_risk_below_full"] = int(
                    half["candidate_rmse"] < row["rmse"]
                )
        test = branch_lookup["test"][branch]
        print(
            f"  {branch} best-epoch={selection['best_epoch'] + 1} "
            f"test RMSE gain={test['rmse_gain_vs_raw_percent']:.4f}% "
            f"P90 gain={test['p90_gain_vs_raw_percent']:.4f}% "
            f"frames={test['improved_frames_vs_raw']}/{len(run.test_indices)}"
        )
        model.to("cpu")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    qk_sha_after = _sha256_file(qk_path)
    if qk_sha_after != qk_sha_before:
        raise RuntimeError("Formal V0 QK pose artifact changed during residual training.")
    test_raw = branch_lookup["test"]["raw_full_history"]
    test_final = branch_lookup["test"]["gated_residual_uncertainty"]
    uncertainty_pass = int(
        float(test_final.get("uncertainty_error_spearman", -1.0))
        >= run.uncertainty_minimum_spearman
        and int(test_final.get("half_coverage_risk_below_full", 0)) == 1
    )
    branch_test_decisions = {
        branch: {
            "test_rmse_beats_raw": int(
                branch_lookup["test"][branch]["rmse"] < test_raw["rmse"]
            ),
            "test_p90_not_worse_than_raw": int(
                branch_lookup["test"][branch]["p90"] <= test_raw["p90"]
            ),
            "test_go": int(
                branch_lookup["test"][branch]["rmse"] < test_raw["rmse"]
                and branch_lookup["test"][branch]["p90"] <= test_raw["p90"]
            ),
        }
        for branch in LEARNED_BRANCHES
    }
    passing_branches = tuple(
        branch for branch in LEARNED_BRANCHES if branch_test_decisions[branch]["test_go"]
    )
    temporal_go = int(bool(passing_branches))
    decision = {
        "scope": "single_scene_temporal_development_only",
        "temporal_development_decision": "GO" if temporal_go else "NO_GO",
        "cross_scene_generalization_claim": 0,
        "go_rule": "any_predeclared_learned_branch_test_rmse_below_raw_and_p90_not_above_raw",
        "branch_test_decisions": branch_test_decisions,
        "passing_branches": passing_branches,
        "uncertainty_reliability_diagnostic": uncertainty_pass,
        "formal_v0_pose_modified": 0,
        "formal_v0_pointmap_modified": 0,
        "formal_v0_semantic_map_modified": 0,
        "next_gate": (
            "repeat_fixed_protocol_on_heldout_scenes_when_storage_returns"
            if temporal_go
            else "stop_geometry_only_residual_then_reassess_joint_training_as_new_hypothesis"
        ),
    }
    summary = {
        "schema": 1,
        "revision": REVISION,
        "experiment": "geometry_only_trust_aware_pointmap_residual",
        "scope": "single_scene_temporal_split_pipeline_and_proof_of_learning",
        "clip": data.clip,
        "frame_indices": data.frame_indices,
        "split": {
            "train_sequence_indices": run.train_indices,
            "validation_sequence_indices": run.validation_indices,
            "test_sequence_indices": run.test_indices,
            "train_frame_indices": tuple(data.frame_indices[i] for i in run.train_indices),
            "validation_frame_indices": tuple(
                data.frame_indices[i] for i in run.validation_indices
            ),
            "test_frame_indices": tuple(data.frame_indices[i] for i in run.test_indices),
        },
        "branches": branch_rows,
        "branch_lookup": branch_lookup,
        "pointmap_formula": "X_new_native = X_raw_native + gate * delta_native",
        "supervision_gauge": (
            "GT inverse-mapped by frozen reference Sim3 into StreamVGGT native gauge"
        ),
        "evaluation_gauge": "frozen reference-frame Sim3 metric gauge",
        "feature_source": "cached frozen StreamVGGT DPT point-head patch tokens (frame+global)",
        "dpt_layer_indices": data.layer_indices,
        "sam_inputs": 0,
        "backbone_parameters_updated": 0,
        "point_head_parameters_updated": 0,
        "camera_head_parameters_updated": 0,
        "external_residual_head_parameters_updated": 1,
        "test_frames_used_for_optimization": 0,
        "checkpoint_selection": (
            "independent_per_branch_minimum_validation_rmse_then_minimum_validation_p90"
        ),
        "selected_checkpoints": selection_summaries,
        "test_evaluation_policy": "one_pass_after_best_validation_checkpoint_loaded",
        "pose_branch": "formal_v0_retrieve_qk_frozen_unchanged",
        "qk_pose_artifact": str(qk_path),
        "qk_pose_sha256_before": qk_sha_before,
        "qk_pose_sha256_after": qk_sha_after,
        "v0_cache": str(data.cache_path),
        "formal_v0_modified": 0,
        "decision": decision,
        "claim": (
            "temporal_development_signal_only_requires_cross_scene_confirmation"
            if temporal_go
            else "trust_aware_residual_temporal_improvement_not_established"
        ),
    }
    model_payload = {
        "schema": 1,
        "revision": REVISION,
        "formal_v0_artifact": 0,
        "clip": data.clip,
        "model": {
            "feature_channels": int(data.features.shape[-1]),
            "level_count": int(data.features.shape[1]),
            "patch_shape": data.patch_shape,
            "projection_channels": run.projection_channels,
            "hidden_channels": run.hidden_channels,
            "gate_bias": run.gate_bias,
        },
        "training_config": _serializable_run(run),
        "branches": states,
    }
    torch.save(model_payload, run.output_dir / "models.pt")
    _write_json(run.output_dir / "summary.json", summary)
    _write_csv(run.output_dir / "branch_summary.csv", branch_rows)
    _write_csv(run.output_dir / "frame_metrics.csv", frame_rows)
    _write_csv(run.output_dir / "training_curve.csv", training_rows)
    _write_csv(run.output_dir / "risk_coverage.csv", risk_rows)
    _write_copyable(run.output_dir / "copyable_result.txt", summary)
    print(
        "  decision="
        f"{decision['temporal_development_decision']} "
        "scope=single-scene-temporal-only"
    )
    print(f"  result={run.output_dir / 'summary.json'}")


def _load_experiment_data(
    run: RunConfig,
) -> tuple[ExperimentData, Path, str]:
    config = load_learned_pose_config(run.v0_config)
    baseline = load_baseline_run_config(run.v0_config)
    clip = _find_clip(config.clips, baseline.clip_name)
    source = cache_path(config, clip)
    if not source.is_file():
        raise FileNotFoundError(
            f"Frozen V0 cache is missing: {source}. Run commands_v0_baseline.txt first."
        )
    payload = load_feature_cache(source)
    required = {
        "clip_name",
        "frame_indices",
        "reference_sequence_index",
        "token_levels",
        "dpt_layer_indices",
        "patch_start_idx",
        "patch_shape",
        "image_size",
        "baseline_world_points",
        "baseline_world_confidence",
        "target_world_points",
        "point_alignment_scale",
        "point_alignment_rotation",
        "point_alignment_translation",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"V0 cache lacks Phase 1 fields: {sorted(missing)}.")
    frames = tuple(int(value) for value in payload["frame_indices"])
    if frames != clip.frame_indices or payload["clip_name"] != clip.name:
        raise ValueError("Frozen V0 cache identity/frame order mismatch.")
    patch_shape = tuple(int(value) for value in payload["patch_shape"])
    features = point_head_patch_features(
        payload.pop("token_levels"),
        patch_start_idx=int(payload["patch_start_idx"]),
        patch_shape=patch_shape,
    ).half()
    raw = payload.pop("baseline_world_points").detach().float().cpu()
    confidence = normalize_confidence(
        payload.pop("baseline_world_confidence").detach().float().cpu()
    )
    target_metric = payload.pop("target_world_points").detach().float().cpu()
    scale = float(payload["point_alignment_scale"])
    rotation = payload["point_alignment_rotation"].detach().float().cpu()
    translation = payload["point_alignment_translation"].detach().float().cpu()
    target_native = invert_similarity(
        target_metric,
        scale=scale,
        rotation=rotation,
        translation=translation,
    )
    image_size = tuple(int(value) for value in raw.shape[1:3])
    if tuple(int(value) for value in payload["image_size"]) != image_size:
        raise ValueError(
            f"Cache image size {payload['image_size']} differs from pointmap {image_size}."
        )
    if target_metric.shape != raw.shape or confidence.shape != raw.shape[:-1]:
        raise ValueError("V0 raw/target/confidence dense shapes are inconsistent.")
    support = (
        torch.isfinite(raw).all(dim=-1)
        & torch.isfinite(target_native).all(dim=-1)
        & torch.isfinite(confidence)
        & (confidence >= run.confidence_threshold)
    )
    if any(int(support[index].sum()) < 128 for index in range(len(frames))):
        raise ValueError("At least one temporal frame has fewer than 128 valid points.")
    layer_indices = tuple(int(value) for value in payload["dpt_layer_indices"])
    if layer_indices != (4, 11, 17, 23) or int(features.shape[1]) != 4:
        raise ValueError(
            f"Phase 1 requires the frozen PointHead levels (4,11,17,23), got "
            f"{layer_indices}."
        )
    if int(clip.reference_sequence_index) != 0:
        raise ValueError("Phase 1 r2 requires the formal first-frame reference gauge.")
    del payload
    gc.collect()
    qk_path = _qk_artifact_path(run.v0_config)
    if not qk_path.is_file():
        raise FileNotFoundError(f"Formal V0 QK pose artifact is missing: {qk_path}.")
    qk = torch.load(qk_path, map_location="cpu", weights_only=False)
    if qk.get("selected_pose_branch") != "retrieve_qk":
        raise ValueError("Phase 1 requires the formal V0 retrieve_qk pose branch.")
    if tuple(int(value) for value in qk.get("frame_indices", ())) != frames:
        raise ValueError("Formal V0 QK artifact frame order differs from the cache.")
    del qk
    return (
        ExperimentData(
            clip=clip.name,
            frame_indices=frames,
            reference_sequence_index=int(clip.reference_sequence_index),
            layer_indices=layer_indices,
            patch_shape=patch_shape,
            image_size=image_size,
            features=features,
            raw_native=raw,
            target_native=target_native,
            target_metric=target_metric,
            confidence=confidence,
            support=support,
            scale=scale,
            rotation=rotation,
            translation=translation,
            cache_path=source,
        ),
        qk_path,
        _sha256_file(qk_path),
    )


def _train_branch(
    model: TrustAwareResidualHead,
    branch: str,
    data: ExperimentData,
    run: RunConfig,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor], dict[str, Any]]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=run.learning_rate,
        weight_decay=run.weight_decay,
        foreach=False,
    )
    rows = []
    best_state: dict[str, torch.Tensor] | None = None
    best_score: tuple[float, float] | None = None
    best_epoch = -1
    generator = torch.Generator(device="cpu")
    for epoch in range(run.epochs):
        generator.manual_seed(run.seed + epoch * 1009)
        order = torch.tensor(run.train_indices)[
            torch.randperm(len(run.train_indices), generator=generator)
        ].tolist()
        point_total = 0.0
        regularization_total = 0.0
        uncertainty_total = 0.0
        total = 0.0
        gradient_maximum = 0.0
        batches = 0
        model.train()
        for start in range(0, len(order), run.batch_size):
            indices = order[start : start + run.batch_size]
            features, raw, target, valid = _device_batch(data, indices, device)
            output = model(features, output_size=data.image_size)
            predicted = raw + output.correction
            point_loss = robust_point_loss(
                predicted, target, valid, beta=run.point_beta
            )
            regularization = output.correction[valid].abs().sum(dim=-1).mean()
            uncertainty_loss = torch.zeros((), device=device)
            if output.log_variance is not None:
                uncertainty_loss = heteroscedastic_point_loss(
                    predicted,
                    target,
                    output.log_variance,
                    valid,
                )
            loss = (
                point_loss
                + run.correction_regularization_weight * regularization
                + run.uncertainty_weight * uncertainty_loss
            )
            if not bool(torch.isfinite(loss).detach().cpu()):
                raise RuntimeError(f"Non-finite loss for {branch} epoch={epoch}.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(
                model.parameters(), run.grad_clip_norm
            )
            gradient_value = float(gradient.detach().cpu())
            if not math.isfinite(gradient_value):
                raise RuntimeError(f"Non-finite gradient for {branch} epoch={epoch}.")
            optimizer.step()
            point_total += float(point_loss.detach().cpu())
            regularization_total += float(regularization.detach().cpu())
            uncertainty_total += float(uncertainty_loss.detach().cpu())
            total += float(loss.detach().cpu())
            gradient_maximum = max(gradient_maximum, gradient_value)
            batches += 1
        validation_rmse, validation_median, validation_p90 = _quick_metrics(
            model, run.validation_indices, data, run, device
        )
        score = (validation_rmse, validation_p90)
        selected_as_best = int(
            validation_checkpoint_is_better(
                validation_rmse, validation_p90, best_score
            )
        )
        if selected_as_best:
            best_score = score
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        row = {
            "branch": branch,
            "epoch": epoch,
            "batches": batches,
            "mean_loss": total / batches,
            "mean_point_loss": point_total / batches,
            "mean_correction_regularization": regularization_total / batches,
            "mean_uncertainty_loss": uncertainty_total / batches,
            "maximum_gradient_norm_before_clip": gradient_maximum,
            "validation_rmse": validation_rmse,
            "validation_median": validation_median,
            "validation_p90": validation_p90,
            "selected_as_best_validation_checkpoint": selected_as_best,
        }
        rows.append(row)
        if epoch in {0, run.epochs - 1} or (epoch + 1) % 10 == 0:
            print(
                f"    epoch={epoch + 1}/{run.epochs} "
                f"loss={row['mean_loss']:.6f} val_rmse={validation_rmse:.6f}"
            )
    if best_state is None or best_score is None or best_epoch < 0:
        raise RuntimeError(f"No validation checkpoint was selected for {branch}.")
    selected_rows = [
        row for row in rows if row["selected_as_best_validation_checkpoint"] == 1
    ]
    if not selected_rows or int(selected_rows[-1]["epoch"]) != best_epoch:
        raise RuntimeError(f"Validation checkpoint audit failed for {branch}.")
    for row in rows:
        row["is_final_selected_checkpoint"] = int(int(row["epoch"]) == best_epoch)
    return (
        rows,
        best_state,
        {
            "rule": "minimum_validation_rmse_then_minimum_validation_p90",
            "best_epoch": best_epoch,
            "best_validation_rmse": best_score[0],
            "best_validation_median": float(selected_rows[-1]["validation_median"]),
            "best_validation_p90": best_score[1],
            "epochs_evaluated": run.epochs,
            "test_metrics_read_during_selection": 0,
        },
    )


@torch.inference_mode()
def _quick_metrics(
    model: TrustAwareResidualHead,
    indices: Sequence[int],
    data: ExperimentData,
    run: RunConfig,
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()
    errors = []
    for index in indices:
        features, raw, _, _ = _device_batch(data, [int(index)], device)
        output = model(features, output_size=data.image_size)
        metric = apply_similarity(
            raw + output.correction,
            scale=data.scale,
            rotation=data.rotation,
            translation=data.translation,
        ).cpu()[0]
        selected = _limited_indices(data.support[int(index)], run.maximum_points_per_frame)
        errors.append(
            torch.linalg.vector_norm(
                metric.reshape(-1, 3).index_select(0, selected)
                - data.target_metric[int(index)].reshape(-1, 3).index_select(0, selected),
                dim=-1,
            )
        )
    joined = torch.cat(errors)
    return (
        _rmse(joined),
        float(joined.median()),
        float(torch.quantile(joined, 0.90)),
    )


def _score_raw(
    data: ExperimentData, run: RunConfig
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metric = apply_similarity(
        data.raw_native,
        scale=data.scale,
        rotation=data.rotation,
        translation=data.translation,
    )
    summaries = []
    frame_rows = []
    for split, indices in (
        ("validation", run.validation_indices),
        ("test", run.test_indices),
    ):
        errors = []
        for index in indices:
            selected = _limited_indices(data.support[index], run.maximum_points_per_frame)
            error = torch.linalg.vector_norm(
                metric[index].reshape(-1, 3).index_select(0, selected)
                - data.target_metric[index].reshape(-1, 3).index_select(0, selected),
                dim=-1,
            )
            errors.append(error)
            frame_rows.append(
                _frame_row(
                    branch="raw_full_history",
                    split=split,
                    sequence_index=index,
                    frame_index=data.frame_indices[index],
                    supported_points=int(data.support[index].sum()),
                    error=error,
                    raw_rmse=_rmse(error),
                    mean_gate=0.0,
                    mean_correction=0.0,
                )
            )
        joined = torch.cat(errors)
        summaries.append(
            {
                "split": split,
                "branch": "raw_full_history",
                "supported_points": sum(int(data.support[i].sum()) for i in indices),
                "evaluated_points": int(joined.numel()),
                "rmse": _rmse(joined),
                "median": float(joined.median()),
                "p90": float(torch.quantile(joined, 0.90)),
                "rmse_gain_vs_raw_percent": 0.0,
                "median_gain_vs_raw_percent": 0.0,
                "p90_gain_vs_raw_percent": 0.0,
                "improved_frames_vs_raw": 0,
                "improved_frame_ratio_vs_raw": 0.0,
                "mean_gate": 0.0,
                "mean_correction_norm_native": 0.0,
            }
        )
    return summaries, frame_rows


@torch.inference_mode()
def _score_model(
    model: TrustAwareResidualHead,
    *,
    branch: str,
    split: str,
    indices: Sequence[int],
    data: ExperimentData,
    run: RunConfig,
    device: torch.device,
    raw_frame_rows: Sequence[Mapping[str, Any]],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor,
]:
    model.eval()
    errors = []
    raw_errors = []
    uncertainty_values = []
    rows = []
    gate_sum = 0.0
    correction_sum = 0.0
    point_count = 0
    raw_lookup = {
        int(row["sequence_index"]): float(row["rmse"])
        for row in raw_frame_rows
        if row["split"] == split and row["branch"] == "raw_full_history"
    }
    improved = 0
    for index in indices:
        features, raw, _, valid = _device_batch(data, [int(index)], device)
        output = model(features, output_size=data.image_size)
        predicted_native = raw + output.correction
        predicted_metric = apply_similarity(
            predicted_native,
            scale=data.scale,
            rotation=data.rotation,
            translation=data.translation,
        ).cpu()[0]
        raw_metric = apply_similarity(
            raw,
            scale=data.scale,
            rotation=data.rotation,
            translation=data.translation,
        ).cpu()[0]
        selected = _limited_indices(data.support[index], run.maximum_points_per_frame)
        target = data.target_metric[index].reshape(-1, 3).index_select(0, selected)
        error = torch.linalg.vector_norm(
            predicted_metric.reshape(-1, 3).index_select(0, selected) - target,
            dim=-1,
        )
        raw_error = torch.linalg.vector_norm(
            raw_metric.reshape(-1, 3).index_select(0, selected) - target,
            dim=-1,
        )
        gate = output.gate.cpu()[0, ..., 0]
        correction = torch.linalg.vector_norm(output.correction.cpu()[0], dim=-1)
        selected_valid = torch.nonzero(valid.cpu()[0].reshape(-1), as_tuple=False)[:, 0]
        mean_gate = float(gate.reshape(-1).index_select(0, selected_valid).mean())
        mean_correction = float(
            correction.reshape(-1).index_select(0, selected_valid).mean()
        )
        candidate_rmse = _rmse(error)
        improved += int(candidate_rmse < raw_lookup[index])
        rows.append(
            _frame_row(
                branch=branch,
                split=split,
                sequence_index=index,
                frame_index=data.frame_indices[index],
                supported_points=int(data.support[index].sum()),
                error=error,
                raw_rmse=raw_lookup[index],
                mean_gate=mean_gate,
                mean_correction=mean_correction,
            )
        )
        errors.append(error)
        raw_errors.append(raw_error)
        gate_sum += mean_gate * int(selected_valid.numel())
        correction_sum += mean_correction * int(selected_valid.numel())
        point_count += int(selected_valid.numel())
        if output.log_variance is not None:
            uncertainty = torch.exp(0.5 * output.log_variance.cpu()[0, ..., 0])
            uncertainty_values.append(
                uncertainty.reshape(-1).index_select(0, selected)
            )
    joined = torch.cat(errors)
    joined_raw = torch.cat(raw_errors)
    row = {
        "split": split,
        "branch": branch,
        "supported_points": sum(int(data.support[i].sum()) for i in indices),
        "evaluated_points": int(joined.numel()),
        "rmse": _rmse(joined),
        "median": float(joined.median()),
        "p90": float(torch.quantile(joined, 0.90)),
        "improved_frames_vs_raw": improved,
        "improved_frame_ratio_vs_raw": improved / len(indices),
        "mean_gate": gate_sum / point_count,
        "mean_correction_norm_native": correction_sum / point_count,
    }
    return (
        row,
        rows,
        torch.cat(uncertainty_values) if uncertainty_values else None,
        joined,
        joined_raw,
    )


def _frame_row(
    *,
    branch: str,
    split: str,
    sequence_index: int,
    frame_index: int,
    supported_points: int,
    error: torch.Tensor,
    raw_rmse: float,
    mean_gate: float,
    mean_correction: float,
) -> dict[str, Any]:
    rmse = _rmse(error)
    return {
        "split": split,
        "branch": branch,
        "sequence_index": int(sequence_index),
        "frame_index": int(frame_index),
        "supported_points": int(supported_points),
        "evaluated_points": int(error.numel()),
        "rmse": rmse,
        "median": float(error.median()),
        "p90": float(torch.quantile(error, 0.90)),
        "raw_rmse": float(raw_rmse),
        "rmse_gain_vs_raw_percent": _gain(raw_rmse, rmse),
        "improved_vs_raw": int(rmse < raw_rmse),
        "mean_gate": float(mean_gate),
        "mean_correction_norm_native": float(mean_correction),
    }


def _risk_coverage(
    uncertainty: torch.Tensor,
    candidate_error: torch.Tensor,
    raw_error: torch.Tensor,
    *,
    branch: str,
    split: str,
) -> list[dict[str, Any]]:
    order = torch.argsort(uncertainty)
    rows = []
    for coverage in (0.1, 0.25, 0.5, 0.75, 1.0):
        count = max(1, int(math.floor(order.numel() * coverage)))
        selected = order[:count]
        candidate = candidate_error.index_select(0, selected)
        raw = raw_error.index_select(0, selected)
        rows.append(
            {
                "split": split,
                "branch": branch,
                "coverage": coverage,
                "retained_points": count,
                "candidate_rmse": _rmse(candidate),
                "candidate_median": float(candidate.median()),
                "candidate_p90": float(torch.quantile(candidate, 0.90)),
                "raw_rmse_on_same_points": _rmse(raw),
                "improved_point_ratio": float((candidate < raw).float().mean()),
                "mean_uncertainty": float(
                    uncertainty.index_select(0, selected).mean()
                ),
            }
        )
    return rows


def _spearman(first: torch.Tensor, second: torch.Tensor) -> float:
    if first.numel() != second.numel() or first.numel() < 2:
        raise ValueError("Spearman correlation needs two equal nontrivial vectors.")
    first_rank = torch.argsort(torch.argsort(first)).float()
    second_rank = torch.argsort(torch.argsort(second)).float()
    first_rank -= first_rank.mean()
    second_rank -= second_rank.mean()
    denominator = torch.sqrt(
        first_rank.square().sum() * second_rank.square().sum()
    ).clamp_min(1e-12)
    return float((first_rank * second_rank).sum() / denominator)


@torch.inference_mode()
def _zero_update_error(
    model: TrustAwareResidualHead,
    data: ExperimentData,
    run: RunConfig,
    device: torch.device,
) -> float:
    model.eval()
    features, _, _, _ = _device_batch(data, [run.train_indices[0]], device)
    correction = model(features, output_size=data.image_size).correction
    return float(correction.abs().max().cpu())


def _device_batch(
    data: ExperimentData, indices: Sequence[int], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    selected = list(int(value) for value in indices)
    return (
        data.features[selected].to(device=device, dtype=torch.float32),
        data.raw_native[selected].to(device=device),
        data.target_native[selected].to(device=device),
        data.support[selected].to(device=device),
    )


def _limited_indices(mask: torch.Tensor, maximum: int) -> torch.Tensor:
    selected = torch.nonzero(mask.reshape(-1), as_tuple=False)[:, 0]
    if selected.numel() <= int(maximum):
        return selected
    positions = torch.linspace(0, selected.numel() - 1, steps=int(maximum)).long()
    return selected.index_select(0, positions)


def _validate_split(run: RunConfig, frame_count: int) -> None:
    expected = {
        "train": tuple(range(0, 18)),
        "validation": tuple(range(18, 24)),
        "test": tuple(range(24, 30)),
    }
    actual = {
        "train": run.train_indices,
        "validation": run.validation_indices,
        "test": run.test_indices,
    }
    if frame_count != 30 or actual != expected:
        raise ValueError(
            f"Phase 1 r2 requires the fixed 30-frame 18/6/6 split; got "
            f"frames={frame_count}, split={actual}."
        )
    joined = (*run.train_indices, *run.validation_indices, *run.test_indices)
    if tuple(joined) != tuple(range(frame_count)):
        raise ValueError("Temporal split must cover every frame exactly once.")


def _load_run(path: str | Path, *, device_override: str | None) -> RunConfig:
    source = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf8")) or {}
    split = raw.get("split", {})
    model = raw.get("model", {})
    training = raw.get("training", {})
    evaluation = raw.get("evaluation", {})
    run = RunConfig(
        source_path=source,
        v0_config=expand_storage_path(
            raw.get("v0_config", "streaming_couping/configs/v0_baseline.yaml")
        ),
        output_dir=expand_storage_path(
            raw.get(
                "output_dir",
                "${VGGT_SAM_STORAGE_ROOT}/outputs/streaming_couping_phase1_temporal_residual_r2",
            )
        ),
        device=str(device_override or raw.get("device", "cuda:0")),
        seed=int(raw.get("seed", 2026)),
        train_indices=tuple(int(v) for v in split.get("train_sequence_indices", range(18))),
        validation_indices=tuple(
            int(v) for v in split.get("validation_sequence_indices", range(18, 24))
        ),
        test_indices=tuple(int(v) for v in split.get("test_sequence_indices", range(24, 30))),
        projection_channels=int(model.get("projection_channels", 64)),
        hidden_channels=int(model.get("hidden_channels", 128)),
        gate_bias=float(model.get("gate_bias", -2.0)),
        epochs=int(training.get("epochs", 30)),
        batch_size=int(training.get("batch_size", 2)),
        learning_rate=float(training.get("learning_rate", 2e-4)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
        grad_clip_norm=float(training.get("grad_clip_norm", 1.0)),
        point_beta=float(training.get("point_beta", 0.05)),
        correction_regularization_weight=float(
            training.get("correction_regularization_weight", 0.01)
        ),
        uncertainty_weight=float(training.get("uncertainty_weight", 0.1)),
        confidence_threshold=float(evaluation.get("confidence_threshold", 0.30)),
        maximum_points_per_frame=int(evaluation.get("maximum_points_per_frame", 8192)),
        minimum_improved_test_frames=int(
            evaluation.get("minimum_improved_test_frames", 4)
        ),
        uncertainty_minimum_spearman=float(
            evaluation.get("uncertainty_minimum_spearman", 0.20)
        ),
    )
    if run.v0_config.name != "v0_baseline.yaml":
        raise ValueError("Phase 1 must consume the formal frozen V0 configuration.")
    fixed = {
        "seed": (run.seed, 2026),
        "projection_channels": (run.projection_channels, 64),
        "hidden_channels": (run.hidden_channels, 128),
        "gate_bias": (run.gate_bias, -2.0),
        "epochs": (run.epochs, 30),
        "batch_size": (run.batch_size, 2),
        "learning_rate": (run.learning_rate, 2e-4),
        "weight_decay": (run.weight_decay, 1e-4),
        "grad_clip_norm": (run.grad_clip_norm, 1.0),
        "point_beta": (run.point_beta, 0.05),
        "correction_regularization_weight": (
            run.correction_regularization_weight,
            0.01,
        ),
        "uncertainty_weight": (run.uncertainty_weight, 0.1),
        "confidence_threshold": (run.confidence_threshold, 0.30),
        "maximum_points_per_frame": (run.maximum_points_per_frame, 8192),
        "minimum_improved_test_frames": (run.minimum_improved_test_frames, 4),
        "uncertainty_minimum_spearman": (
            run.uncertainty_minimum_spearman,
            0.20,
        ),
    }
    changed = {name: value for name, (value, expected) in fixed.items() if value != expected}
    if changed:
        raise ValueError(f"Phase 1 r2 fixed protocol was changed: {changed}.")
    return run


def _find_clip(clips: tuple[ClipConfig, ...], name: str) -> ClipConfig:
    selected = [clip for clip in clips if clip.name == name]
    if len(selected) != 1:
        raise ValueError(f"Expected exactly one V0 clip {name!r}.")
    return selected[0]


def _qk_artifact_path(config: Path) -> Path:
    raw = yaml.safe_load(config.read_text(encoding="utf8")) or {}
    return expand_storage_path(raw["baseline"]["pose"]["qk_pose_output"])


def _resolve_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA training requested on {device}, but CUDA is unavailable.")
    return device


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _gain(baseline: float, candidate: float) -> float:
    return 100.0 * (float(baseline) - float(candidate)) / max(
        abs(float(baseline)), 1e-12
    )


def _rmse(values: torch.Tensor) -> float:
    if values.numel() == 0:
        raise ValueError("RMSE requires non-empty values.")
    return float(torch.sqrt(values.float().square().mean()))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _serializable_run(run: RunConfig) -> dict[str, Any]:
    value = asdict(run)
    for key in ("source_path", "v0_config", "output_dir"):
        value[key] = str(value[key])
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
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


def _write_copyable(path: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "===== COPYABLE_PHASE1_TEMPORAL_RESIDUAL_BEGIN =====",
        f"revision={summary['revision']}",
        f"clip={summary['clip']}",
        f"scope={summary['scope']}",
        "split=train18,val6,test6",
        "branches=" + ",".join(SCORE_BRANCHES),
        f"feature_source={summary['feature_source']}",
        f"pointmap_formula={summary['pointmap_formula']}",
        "sam_inputs=0",
        "backbone_parameters_updated=0",
        "point_head_parameters_updated=0",
        "camera_head_parameters_updated=0",
        "test_frames_used_for_optimization=0",
        "checkpoint_selection=minimum_validation_rmse_then_minimum_validation_p90",
        "test_evaluation_policy=one_pass_after_best_validation_checkpoint_loaded",
        "formal_v0_modified=0",
        "cross_scene_generalization_claim=0",
        "",
        (
            "branch,best_epoch_one_based,best_validation_rmse,"
            "best_validation_median,best_validation_p90"
        ),
    ]
    for branch in LEARNED_BRANCHES:
        selected = summary["selected_checkpoints"][branch]
        lines.append(
            ",".join(
                str(value)
                for value in (
                    branch,
                    int(selected["best_epoch"]) + 1,
                    selected["best_validation_rmse"],
                    selected["best_validation_median"],
                    selected["best_validation_p90"],
                )
            )
        )
    lines.extend(
        (
            "",
            (
                "split,branch,evaluated_points,rmse,median,p90,"
                "rmse_gain_vs_raw_percent,p90_gain_vs_raw_percent,"
                "improved_frames_vs_raw,improved_frame_ratio_vs_raw,"
                "uncertainty_error_spearman"
            ),
        )
    )
    fields = (
        "split",
        "branch",
        "evaluated_points",
        "rmse",
        "median",
        "p90",
        "rmse_gain_vs_raw_percent",
        "p90_gain_vs_raw_percent",
        "improved_frames_vs_raw",
        "improved_frame_ratio_vs_raw",
        "uncertainty_error_spearman",
    )
    for row in summary["branches"]:
        lines.append(",".join(str(row.get(field, "")) for field in fields))
    lines.extend(
        (
            "",
            "decision=" + json.dumps(summary["decision"], sort_keys=True),
            f"claim={summary['claim']}",
            "outputs:",
            f"summary={path.with_name('summary.json')}",
            f"branch_csv={path.with_name('branch_summary.csv')}",
            f"frame_csv={path.with_name('frame_metrics.csv')}",
            f"training_csv={path.with_name('training_curve.csv')}",
            f"risk_coverage_csv={path.with_name('risk_coverage.csv')}",
            f"models={path.with_name('models.pt')}",
            "===== COPYABLE_PHASE1_TEMPORAL_RESIDUAL_END =====",
        )
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="streaming_couping/configs/temporal_residual_training.yaml",
    )
    parser.add_argument("--device", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
