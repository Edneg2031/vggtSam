#!/usr/bin/env python3
"""V9.8 causal SAM3.1 video-memory correspondence experiment.

No pose model is trained.  A parameter-matched dense matcher is supervised by
GT 2D correspondence labels only, then frozen before future-frame evaluation.
Relative pose always comes from the locked V9.3 O-R1 solver.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable, Sequence

import torch
import yaml

from streaming_couping.scripts.run_v90_local_token_matcher import (
    StageAConfig,
    load_stage_a_config,
)
from streaming_couping.scripts.run_v92_support_factorization import (
    _build_support_records,
    _load_support_payload,
    _prepare_support_data,
    load_support_config,
)
from streaming_couping.scripts.run_v93_quantization_tolerance import (
    load_diagnostic_config,
)
from streaming_couping.scripts.run_v95_spatial_support_scope import (
    _frame_edge_map,
    _locked_edge_indices,
    load_spatial_scope_config,
)
from streaming_couping.scripts.run_v96_dense_grid_upper_bound import (
    load_dense_grid_config,
)
from streaming_couping.scripts.run_v97_dense_descriptor_causality import (
    DenseData,
    DenseRecord,
    DenseTrainingConfig,
    TrainState,
    _build_evaluation_records,
    _build_training_records,
    _checkpoint_signature as _v97_checkpoint_signature,
    _prepare_dense_data,
    load_v97_config,
)
from streaming_couping.src.v74_temporal_protocol import EXPECTED_FRAMES, FOLDS
from streaming_couping.src.v80_pose_geometry import invert_rigid, rotation_error_degrees
from streaming_couping.src.v90_epipolar_geometry import (
    estimate_relative_epipolar_pose,
    relative_translation_direction_error_degrees,
)
from streaming_couping.src.v95_spatial_support import take_surface, uv_hull_coverage
from streaming_couping.src.v97_dense_descriptor_matcher import (
    DenseMatchTarget,
    DenseMatcherConfig,
    DenseSubgridMatcher,
    build_dense_match_target,
    decode_dense_matches,
    dense_match_loss,
    deterministic_channel_permutation,
)
from streaming_couping.src.v98_temporal_memory import (
    TEMPORAL_ARCHITECTURES,
    TEMPORAL_VARIANTS,
)


SUMMARY_COLUMNS = (
    "fold", "architecture", "variant", "descriptor_source", "train_frames",
    "test_frames", "parameters", "trainable_parameters", "steps",
    "initial_train_loss", "final_train_loss", "train_loss_drop_percent",
    "frames", "active_frames", "inactive_frames", "queries",
    "accepted_correspondences", "pck_threshold_pixels", "pck_accuracy",
    "mean_epe_pixels", "raw_rotation_error_deg", "refined_rotation_error_deg",
    "rotation_gain_percent", "raw_translation_direction_error_deg",
    "refined_translation_direction_error_deg", "translation_direction_gain_percent",
    "raw_relative_aggregate_deg", "refined_relative_aggregate_deg",
    "relative_aggregate_gain_percent", "relative_aggregate_worse_frames",
    "parameter_matched", "initialization_matched", "control_support_exact", "matcher_frozen_exact",
    "perturbed_pairs", "fold_correspondence_pass", "fold_pose_pass",
    "fold_sam_memory_causal_pass", "all_folds_sam_memory_causal_pass",
)

FRAME_COLUMNS = (
    "fold", "architecture", "variant", "sequence_index", "frame_index",
    "history_sequence_index", "history_frame_index", "queries",
    "accepted_correspondences", "active", "solver_reason", "pck_correct",
    "epe_sum_pixels", "current_hull_coverage",
    "predicted_history_hull_coverage", "raw_rotation_error_deg",
    "refined_rotation_error_deg", "raw_translation_direction_error_deg",
    "refined_translation_direction_error_deg", "relative_aggregate_worse",
)

TRAIN_COLUMNS = (
    "fold", "architecture", "seed", "step", "loss", "classification_loss",
    "offset_loss", "supervised_queries", "visible_queries",
    "query_projection_grad_norm", "key_projection_grad_norm",
    "offset_head_grad_norm",
)

AUDIT_COLUMNS = (
    "fold", "split", "descriptor_source", "pairs", "queries", "pck_threshold_pixels",
    "pck_accuracy", "mean_epe_pixels", "checkpoint", "checkpoint_uses_pose_loss",
)


@dataclass(frozen=True)
class V98Config:
    source_path: Path
    v97_config: Path
    data_config: Path
    output_dir: Path
    temporal_cache_path: Path
    v97_dense_cache_path: Path
    v97_output_dir: Path
    clip_name: str
    grid_size: tuple[int, int]
    cache_device: str
    matcher: DenseMatcherConfig
    training: DenseTrainingConfig

    @property
    def expected_frames(self) -> tuple[int, ...]:
        return EXPECTED_FRAMES


@dataclass(frozen=True)
class TemporalData:
    memory: torch.Tensor
    raw: torch.Tensor
    valid: torch.Tensor
    metadata: dict[str, Any]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="streaming_couping/configs/v98_temporal_memory_causality.yaml",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--temporal-cache-path", default=None)
    parser.add_argument("--v97-output-dir", default=None)
    parser.add_argument("--v97-dense-cache-path", default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    config = load_v98_config(args.config)
    updates = {}
    for name in (
        "output_dir", "temporal_cache_path", "v97_output_dir", "v97_dense_cache_path"
    ):
        value = getattr(args, name)
        if value:
            updates[name] = Path(value).expanduser().resolve()
    if updates:
        config = replace(config, **updates)
    result = run_v98(config, resume=not bool(args.no_resume))
    print(f"V9.8 temporal memory result={result}")


def run_v98(config: V98Config, *, resume: bool = True) -> Path:
    _seed_everything(config.training.seed)
    v97 = replace(
        load_v97_config(config.v97_config),
        data_config=config.data_config,
        dense_cache_path=config.v97_dense_cache_path,
        output_dir=config.v97_output_dir,
    )
    if asdict(config.matcher) != asdict(v97.matcher):
        raise ValueError("V9.8 matcher is not exactly parameter-matched to V9.7.")
    initialization_fields = (
        "seed", "steps", "batch_pairs", "learning_rate", "weight_decay"
    )
    if any(
        getattr(config.training, name) != getattr(v97.training, name)
        for name in initialization_fields
    ):
        raise ValueError(
            "V9.8 must reuse V9.7 seed/steps/batch/optimizer for exact "
            "initialization and training-order controls."
        )
    v96 = load_dense_grid_config(v97.v96_config)
    v95 = load_spatial_scope_config(v96.v95_config)
    v93 = load_diagnostic_config(v95.v93_config)
    support_config = load_support_config(v93.support_config)
    stage_config = replace(
        load_stage_a_config(support_config.stage_a_config),
        data_config=config.data_config,
        output_dir=config.output_dir,
    )
    payload, source_cache = _load_support_payload(support_config)
    support = _prepare_support_data(
        payload, config=support_config, stage_config=stage_config
    )
    del payload
    dense_payload = _torch_load(config.v97_dense_cache_path, "V9.7 dense cache")
    dense = _prepare_dense_data(dense_payload, support=support, config=v97)
    del dense_payload
    temporal = _prepare_temporal_data(
        _torch_load(config.temporal_cache_path, "V9.8 temporal cache"),
        dense=dense,
        config=config,
    )

    positions = {frame: index for index, frame in enumerate(support.frames)}
    instance_records = {
        fold.name: _build_support_records(
            support,
            current_indices=[positions[value] for value in fold.test_frames],
            query_token_count=support_config.query_token_count,
            stage_config=stage_config,
        )
        for fold in FOLDS
    }
    fixed_edges = _locked_edge_indices(v95, support)
    evaluation_records = {
        fold.name: _build_evaluation_records(
            fold_name=fold.name,
            fixed_edges=fixed_edges[fold.name],
            instance_records=instance_records[fold.name],
            data=dense,
            stage_config=stage_config,
            query_count=None,
        )
        for fold in FOLDS
    }
    training_records = {
        fold.name: _build_training_records(
            [positions[value] for value in fold.train_frames],
            data=dense,
            stage_config=stage_config,
            config=v97,
        )
        for fold in FOLDS
    }
    _require_temporal_support(temporal, training_records, evaluation_records, support.frames)

    v97_rows = _read_v97_summary(config.v97_output_dir)
    audit_rows = _audit_v97_train_future(
        config=config,
        v97=v97,
        dense=dense,
        training_records=training_records,
        evaluation_records=evaluation_records,
        source_cache=source_cache,
    )

    states: dict[tuple[str, str], TrainState] = {}
    train_rows: list[dict[str, object]] = []
    for fold in FOLDS:
        for architecture in TEMPORAL_ARCHITECTURES:
            if architecture == "sam_memory_train_off":
                state, logs = _reuse_v97_train_off(
                    fold_name=fold.name,
                    config=config,
                    v97=v97,
                    source_cache=source_cache,
                )
            else:
                state, logs = _train_matcher(
                    fold_name=fold.name,
                    architecture=architecture,
                    records=training_records[fold.name],
                    temporal=temporal,
                    dense=dense,
                    config=config,
                    source_cache=source_cache,
                    resume=resume,
                )
            states[(fold.name, architecture)] = state
            train_rows.extend(logs)

    summaries: list[dict[str, object]] = []
    frames: list[dict[str, object]] = []
    for fold in FOLDS:
        for architecture in TEMPORAL_ARCHITECTURES:
            variants = TEMPORAL_VARIANTS if architecture == "sam_memory" else ("normal",)
            for variant in variants:
                summary, rows = _evaluate_model(
                    fold_name=fold.name,
                    train_frames=fold.train_frames,
                    test_frames=fold.test_frames,
                    architecture=architecture,
                    variant=variant,
                    state=states[(fold.name, architecture)],
                    records=evaluation_records[fold.name],
                    temporal=temporal,
                    dense=dense,
                    stage_config=stage_config,
                    config=config,
                )
                summaries.append(summary)
                frames.extend(rows)
    _annotate_decisions(summaries, v97_rows)
    _validate_outputs(summaries, frames, audit_rows)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = config.output_dir / "v98_temporal_memory_summary.csv"
    frame_path = config.output_dir / "v98_temporal_memory_frames.csv"
    training_path = config.output_dir / "v98_temporal_memory_training.csv"
    audit_path = config.output_dir / "v98_v97_train_future_audit.csv"
    decision_path = config.output_dir / "v98_temporal_memory_decision.md"
    metadata_path = config.output_dir / "v98_temporal_memory_metadata.json"
    _write_csv(summary_path, summaries, SUMMARY_COLUMNS)
    _write_csv(frame_path, frames, FRAME_COLUMNS)
    _write_csv(training_path, train_rows, TRAIN_COLUMNS)
    _write_csv(audit_path, audit_rows, AUDIT_COLUMNS)
    decision_path.write_text(
        _decision_markdown(summaries, v97_rows, audit_rows), encoding="utf8"
    )
    metadata_path.write_text(
        json.dumps(
            {
                "experiment": "V9.8 SAM3.1 causal video-memory correspondence",
                "config": _jsonable_config(config),
                "source_cache": _file_provenance(source_cache),
                "v97_dense_cache": _file_provenance(config.v97_dense_cache_path),
                "temporal_cache": _file_provenance(config.temporal_cache_path),
                "fixed_history_edges": _frame_edge_map(fixed_edges, support),
                "feature_boundary": temporal.metadata,
                "query_support": "identical V9.7 full 72x72 grid and current-only FPS",
                "trains_pose_model": False,
                "uses_pose_loss": False,
                "matcher_supervision": "GT continuous 2D correspondence only",
                "reused_control": "V9.7 sam_train_off checkpoint (identical zero-input training)",
                "pose_solver": "frozen V9.3 O-R1 calibrated relative epipolar solver",
                "claim_scope": "one-scene temporal future relative rotation/translation direction",
                "outputs": {
                    "summary": str(summary_path), "frames": str(frame_path),
                    "training": str(training_path), "v97_audit": str(audit_path),
                    "decision": str(decision_path),
                },
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf8",
    )
    print(f"V9.8 wrote summary={summary_path}")
    print(f"V9.8 wrote decision={decision_path}")
    return summary_path


def _prepare_temporal_data(payload: dict[str, Any], *, dense: DenseData, config: V98Config) -> TemporalData:
    required = {
        "complete", "frame_indices", "grid_size", "memory_conditioned_features",
        "memory_off_raw_features", "memory_read_valid", "past_memory_read_valid", "feature_source",
        "memory_off_source", "prompt_combination", "propagation_direction",
        "causal_confirmation", "uses_future_frames",
        "strictly_earlier_memory_observed",
        "same_frame_memory_excluded",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"V9.8 temporal cache lacks fields={sorted(missing)}.")
    memory = torch.as_tensor(payload["memory_conditioned_features"]).float().cpu()
    raw = torch.as_tensor(payload["memory_off_raw_features"]).float().cpu()
    valid = torch.as_tensor(payload["memory_read_valid"]).bool().cpu()
    past_valid = torch.as_tensor(payload["past_memory_read_valid"]).bool().cpu()
    expected = (len(dense.support.frames), len(dense.grid_uv_normalized), config.matcher.canonical_dim)
    if not payload["complete"] or memory.shape != expected or raw.shape != expected:
        raise ValueError("V9.8 temporal cache tensor shape/completeness failed.")
    if tuple(int(value) for value in payload["frame_indices"]) != dense.support.frames:
        raise ValueError("V9.8 temporal/source frame order differs.")
    if tuple(int(value) for value in payload["grid_size"]) != config.grid_size:
        raise ValueError("V9.8 temporal grid differs from V9.7.")
    if valid.shape != (len(dense.support.frames),) or past_valid.shape != valid.shape:
        raise ValueError("V9.8 memory-read validity has the wrong shape.")
    if str(payload["propagation_direction"]) != "forward" or not payload["causal_confirmation"]:
        raise ValueError("V9.8 temporal cache is not forward-only causal tracking.")
    if bool(payload["uses_future_frames"]):
        raise ValueError("V9.8 refuses a temporal cache that uses future frames.")
    if not bool(payload["strictly_earlier_memory_observed"]):
        raise ValueError("V9.8 cache never read strictly earlier SAM memory.")
    if not bool(payload["same_frame_memory_excluded"]):
        raise ValueError("V9.8 cache did not exclude same-frame memory conditioning.")
    if not bool(torch.isfinite(memory).all() and torch.isfinite(raw).all()):
        raise ValueError("V9.8 temporal cache contains NaN/Inf.")
    metadata = {
        key: payload[key]
        for key in (
            "feature_source", "memory_off_source", "prompt_combination",
            "propagation_direction", "causal_confirmation", "uses_future_frames",
            "strictly_earlier_memory_observed",
            "same_frame_memory_excluded",
        )
    }
    return TemporalData(memory, raw, valid & past_valid, metadata)


def _train_matcher(*, fold_name, architecture, records, temporal, dense, config, source_cache, resume):
    if architecture not in TEMPORAL_ARCHITECTURES:
        raise ValueError(f"Unknown V9.8 architecture={architecture!r}.")
    device = _training_device(config.training.device)
    fold_seed = config.training.seed + 1009 * [fold.name for fold in FOLDS].index(fold_name)
    _seed_everything(fold_seed)
    model = DenseSubgridMatcher(config.matcher).to(device)
    checkpoint_dir = config.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / f"{fold_name}_{architecture}.pt"
    signature = _checkpoint_signature(config, fold_name, architecture, source_cache)
    if resume and checkpoint.is_file():
        saved = _torch_load(checkpoint, "V9.8 checkpoint")
        if saved.get("signature") == signature:
            model.load_state_dict(saved["model"])
            print(f"V9.8 resumed fold={fold_name} architecture={architecture}")
            return TrainState(
                model=model,
                parameters=sum(value.numel() for value in model.parameters()),
                trainable_parameters=sum(value.numel() for value in model.parameters() if value.requires_grad),
                steps=int(saved["steps"]),
                initial_loss=float(saved["initial_loss"]),
                final_loss=float(saved["final_loss"]),
            ), list(saved.get("logs", []))
        print(f"V9.8 ignored stale checkpoint={checkpoint}")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    features = _architecture_features(architecture, temporal)
    initial = _dataset_loss(model, records, features, architecture, dense, config, device)
    logs = []
    for step in range(config.training.steps):
        indices = _cyclic_batch_indices(len(records), config.training.batch_pairs, step, fold_seed)
        batch = [records[index] for index in indices]
        query, key, query_valid, key_valid, target = _training_batch(
            batch, features, architecture, dense, config, device
        )
        result = dense_match_loss(model, model(query, key, query_valid, key_valid), target)
        optimizer.zero_grad(set_to_none=True)
        result.loss.backward()
        q_grad = _grad_norm(model.query_projection.weight)
        k_grad = _grad_norm(model.key_projection.weight)
        offset_grad = _module_grad_norm(model.offset_head)
        optimizer.step()
        if step == 0 or (step + 1) % config.training.log_every == 0:
            value = float(result.loss.detach().cpu())
            print(
                f"V9.8 fold={fold_name} architecture={architecture} "
                f"step={step + 1}/{config.training.steps} loss={value:.6g}"
            )
            logs.append({
                "fold": fold_name, "architecture": architecture, "seed": fold_seed,
                "step": step + 1, "loss": value,
                "classification_loss": float(result.classification.detach().cpu()),
                "offset_loss": float(result.offset.detach().cpu()),
                "supervised_queries": result.supervised_queries,
                "visible_queries": result.visible_queries,
                "query_projection_grad_norm": q_grad,
                "key_projection_grad_norm": k_grad,
                "offset_head_grad_norm": offset_grad,
            })
    final = _dataset_loss(model, records, features, architecture, dense, config, device)
    saved = {
        "signature": signature,
        "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "steps": config.training.steps, "initial_loss": initial, "final_loss": final,
        "logs": logs, "uses_pose_loss": False, "contains_pose_model": False,
    }
    torch.save(saved, checkpoint)
    return TrainState(
        model=model,
        parameters=sum(value.numel() for value in model.parameters()),
        trainable_parameters=sum(value.numel() for value in model.parameters() if value.requires_grad),
        steps=config.training.steps,
        initial_loss=initial,
        final_loss=final,
    ), logs


def _reuse_v97_train_off(*, fold_name, config, v97, source_cache):
    """Reuse the exact zero-input control; retraining it would be duplication."""

    checkpoint = config.v97_output_dir / "checkpoints" / f"{fold_name}_sam_train_off.pt"
    saved = _torch_load(checkpoint, "completed V9.7 train-off checkpoint")
    expected = _v97_checkpoint_signature(
        v97,
        fold=fold_name,
        architecture="sam_train_off",
        source_cache=source_cache,
    )
    if saved.get("signature") != expected or bool(saved.get("uses_pose_loss", False)):
        raise ValueError(f"V9.8 found incompatible V9.7 train-off checkpoint={checkpoint}.")
    model = DenseSubgridMatcher(config.matcher).to(_training_device(config.training.device))
    model.load_state_dict(saved["model"])
    logs = []
    for row in saved.get("logs", []):
        current = dict(row)
        current["architecture"] = "sam_memory_train_off"
        logs.append(current)
    print(f"V9.8 reused V9.7 zero-input control fold={fold_name}")
    return TrainState(
        model=model,
        parameters=sum(value.numel() for value in model.parameters()),
        trainable_parameters=sum(
            value.numel() for value in model.parameters() if value.requires_grad
        ),
        steps=int(saved["steps"]),
        initial_loss=float(saved["initial_loss"]),
        final_loss=float(saved["final_loss"]),
    ), logs


def _training_batch(records, features, architecture, dense, config, device):
    maximum = max(record.correspondences.count for record in records)
    channels = int(features.shape[-1])
    keys = len(dense.grid_uv_normalized)
    query = torch.zeros(len(records), maximum, channels)
    key = torch.zeros(len(records), keys, channels)
    query_valid = torch.zeros(len(records), maximum, dtype=torch.bool)
    key_valid = torch.ones(len(records), keys, dtype=torch.bool)
    target_uv = torch.zeros(len(records), maximum, 2, dtype=torch.float64)
    visible = torch.zeros(len(records), maximum, dtype=torch.bool)
    for batch_index, record in enumerate(records):
        count = record.correspondences.count
        query[batch_index, :count] = features[record.current].index_select(0, record.query_grid_indices)
        key[batch_index] = features[record.history]
        query_valid[batch_index, :count] = True
        visible[batch_index, :count] = True
        target_uv[batch_index, :count] = record.correspondences.history_uv
    if architecture == "sam_memory_train_off":
        query.zero_()
        key.zero_()
    target = build_dense_match_target(
        target_uv.to(device), query_valid=query_valid.to(device), visible=visible.to(device),
        grid_size=config.grid_size, image_size=dense.support.image_size,
    )
    return query.to(device), key.to(device), query_valid.to(device), key_valid.to(device), target


def _dataset_loss(model, records, features, architecture, dense, config, device):
    model.eval()
    weighted = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, len(records), config.training.batch_pairs):
            batch = records[start:start + config.training.batch_pairs]
            query, key, query_valid, key_valid, target = _training_batch(
                batch, features, architecture, dense, config, device
            )
            result = dense_match_loss(model, model(query, key, query_valid, key_valid), target)
            weight = max(result.supervised_queries, 1)
            weighted += float(result.loss.cpu()) * weight
            count += weight
    model.train()
    return weighted / max(count, 1)


def _evaluate_model(*, fold_name, train_frames, test_frames, architecture, variant, state,
                    records, temporal, dense, stage_config, config):
    device = next(state.model.parameters()).device
    state.model.eval()
    before = {name: value.detach().cpu().clone() for name, value in state.model.state_dict().items()}
    base_features = _architecture_features(architecture, temporal)
    rows = []
    perturbed = 0
    support_exact = True
    diagonal = math.hypot(*dense.support.image_size)
    with torch.no_grad():
        for record in records:
            current = base_features[record.current].clone()
            history = base_features[record.history].clone()
            if architecture == "sam_memory":
                current, history, applied = _perturb_features(
                    variant, record, temporal, current, history
                )
                perturbed += applied
            query = current.index_select(0, record.query_grid_indices)
            key = history
            query_valid = torch.ones(1, len(query), dtype=torch.bool, device=device)
            key_valid = torch.ones(1, len(key), dtype=torch.bool, device=device)
            output = state.model(query[None].to(device), key[None].to(device), query_valid, key_valid)
            decoded = decode_dense_matches(
                state.model, output, grid_size=config.grid_size,
                image_size=dense.support.image_size, query_valid=query_valid,
            )
            accepted = decoded.accepted[0].cpu()
            predicted_uv = decoded.history_uv_pixels[0].cpu()
            indices = torch.nonzero(accepted, as_tuple=False).flatten()
            selected = take_surface(record.correspondences, indices)
            selected.history_uv = predicted_uv.index_select(0, indices)
            errors = torch.linalg.vector_norm(predicted_uv - record.correspondences.history_uv, dim=-1)
            epe_sum = float(torch.where(accepted, errors, torch.full_like(errors, diagonal)).sum())
            pck = int((accepted & errors.le(config.matcher.pck_threshold_pixels)).sum())
            support_exact = support_exact and len(query) == record.correspondences.count and len(key) == 72 * 72
            l0_relative = dense.support.baseline[record.history] @ invert_rigid(dense.support.baseline[record.current])
            target_relative = dense.support.target[record.history] @ invert_rigid(dense.support.target[record.current])
            raw_r = float(rotation_error_degrees(l0_relative, target_relative))
            raw_t = relative_translation_direction_error_degrees(
                l0_relative[:3, :3], l0_relative[:3, 3], target_relative
            )
            estimate = estimate_relative_epipolar_pose(
                selected.current_uv, selected.history_uv, selected.weights,
                dense.support.intrinsics[record.current], dense.support.intrinsics[record.history],
                l0_relative, config=stage_config.epipolar,
            )
            refined_r, refined_t = raw_r, raw_t
            if estimate.success:
                candidate = torch.eye(4, dtype=torch.float64)
                candidate[:3, :3] = estimate.rotation_current_to_history
                refined_r = float(rotation_error_degrees(candidate, target_relative))
                refined_t = relative_translation_direction_error_degrees(
                    estimate.rotation_current_to_history,
                    estimate.translation_current_origin_in_history, target_relative,
                )
            rows.append({
                "fold": fold_name, "architecture": architecture, "variant": variant,
                "sequence_index": record.current,
                "frame_index": dense.support.frames[record.current],
                "history_sequence_index": record.history,
                "history_frame_index": dense.support.frames[record.history],
                "queries": record.correspondences.count,
                "accepted_correspondences": selected.count,
                "active": int(estimate.success), "solver_reason": estimate.reason,
                "pck_correct": pck, "epe_sum_pixels": epe_sum,
                "current_hull_coverage": uv_hull_coverage(selected.current_uv, dense.support.image_size),
                "predicted_history_hull_coverage": uv_hull_coverage(selected.history_uv, dense.support.image_size),
                "raw_rotation_error_deg": raw_r, "refined_rotation_error_deg": refined_r,
                "raw_translation_direction_error_deg": raw_t,
                "refined_translation_direction_error_deg": refined_t,
                "relative_aggregate_worse": int(refined_r + refined_t > raw_r + raw_t + 1e-12),
            })
    frozen = int(all(torch.equal(value, state.model.state_dict()[name].detach().cpu()) for name, value in before.items()))
    return _summary_row(
        fold_name, architecture, variant, train_frames, test_frames, state, rows,
        int(support_exact), frozen, perturbed, config
    ), rows


def _summary_row(fold_name, architecture, variant, train_frames, test_frames, state,
                 rows, support_exact, frozen, perturbed, config):
    queries = sum(int(row["queries"]) for row in rows)
    accepted = sum(int(row["accepted_correspondences"]) for row in rows)
    raw_r = _finite_mean(row["raw_rotation_error_deg"] for row in rows)
    refined_r = _finite_mean(row["refined_rotation_error_deg"] for row in rows)
    raw_t = _finite_mean(row["raw_translation_direction_error_deg"] for row in rows)
    refined_t = _finite_mean(row["refined_translation_direction_error_deg"] for row in rows)
    descriptor = {
        "sam_memory": "sam31_tracker_history_read",
        "memory_off_raw": "same_tracker_current_propagation_before_memory",
        "sam_memory_train_off": "temporal_eval_zero_during_training",
    }[architecture]
    return {
        "fold": fold_name, "architecture": architecture, "variant": variant,
        "descriptor_source": descriptor,
        "train_frames": " ".join(str(value) for value in train_frames),
        "test_frames": " ".join(str(value) for value in test_frames),
        "parameters": state.parameters, "trainable_parameters": state.trainable_parameters,
        "steps": state.steps, "initial_train_loss": state.initial_loss,
        "final_train_loss": state.final_loss,
        "train_loss_drop_percent": _gain(state.initial_loss, state.final_loss),
        "frames": len(rows), "active_frames": sum(int(row["active"]) for row in rows),
        "inactive_frames": sum(not int(row["active"]) for row in rows),
        "queries": queries, "accepted_correspondences": accepted,
        "pck_threshold_pixels": config.matcher.pck_threshold_pixels,
        "pck_accuracy": _ratio(sum(row["pck_correct"] for row in rows), queries),
        "mean_epe_pixels": _ratio(sum(row["epe_sum_pixels"] for row in rows), queries),
        "raw_rotation_error_deg": raw_r, "refined_rotation_error_deg": refined_r,
        "rotation_gain_percent": _gain(raw_r, refined_r),
        "raw_translation_direction_error_deg": raw_t,
        "refined_translation_direction_error_deg": refined_t,
        "translation_direction_gain_percent": _gain(raw_t, refined_t),
        "raw_relative_aggregate_deg": raw_r + raw_t,
        "refined_relative_aggregate_deg": refined_r + refined_t,
        "relative_aggregate_gain_percent": _gain(raw_r + raw_t, refined_r + refined_t),
        "relative_aggregate_worse_frames": sum(int(row["relative_aggregate_worse"]) for row in rows),
        "parameter_matched": 0, "initialization_matched": 0,
        "control_support_exact": support_exact,
        "matcher_frozen_exact": frozen, "perturbed_pairs": perturbed,
        "fold_correspondence_pass": 0, "fold_pose_pass": 0,
        "fold_sam_memory_causal_pass": 0, "all_folds_sam_memory_causal_pass": 0,
    }


def _annotate_decisions(rows, v97_rows):
    passes = []
    for fold in FOLDS:
        group = {(row["architecture"], row["variant"]): row for row in rows if row["fold"] == fold.name}
        required = {("sam_memory", value) for value in TEMPORAL_VARIANTS} | {
            ("memory_off_raw", "normal"), ("sam_memory_train_off", "normal")
        }
        if required - set(group):
            raise ValueError(f"V9.8 fold={fold.name} lacks controls={sorted(required - set(group))}.")
        v97_group = {
            row["architecture"]: row for row in v97_rows
            if row["fold"] == fold.name and row["variant"] == "normal"
        }
        if not {"sam_dense", "stream_dense"}.issubset(v97_group):
            raise ValueError(f"V9.8 lacks V9.7 detector/Stream controls for fold={fold.name}.")
        candidate = group[("sam_memory", "normal")]
        controls = [group[("memory_off_raw", "normal")], group[("sam_memory_train_off", "normal")],
                    v97_group["sam_dense"], v97_group["stream_dense"]]
        perturbations = [group[("sam_memory", value)] for value in TEMPORAL_VARIANTS[1:]]
        parameter_matched = int(len({int(group[key]["parameters"]) for key in required}) == 1)
        for row in group.values():
            row["parameter_matched"] = parameter_matched
            row["initialization_matched"] = 1
        correspondence = int(
            all(float(candidate["pck_accuracy"]) > float(control["pck_accuracy"])
                and float(candidate["mean_epe_pixels"]) < float(control["mean_epe_pixels"])
                for control in controls)
            and all(int(row["perturbed_pairs"]) > 0
                    and float(row["pck_accuracy"]) < float(candidate["pck_accuracy"])
                    and float(row["mean_epe_pixels"]) > float(candidate["mean_epe_pixels"])
                    for row in perturbations)
        )
        pose = int(
            int(candidate["active_frames"]) == int(candidate["frames"])
            and float(candidate["refined_rotation_error_deg"]) < float(candidate["raw_rotation_error_deg"])
            and float(candidate["refined_translation_direction_error_deg"]) < float(candidate["raw_translation_direction_error_deg"])
            and int(candidate["relative_aggregate_worse_frames"]) == 0
            and all(float(candidate["refined_relative_aggregate_deg"]) < float(control["refined_relative_aggregate_deg"])
                    for control in controls)
            and all(float(row["refined_relative_aggregate_deg"]) > float(candidate["refined_relative_aggregate_deg"])
                    for row in perturbations)
        )
        passed = int(parameter_matched and candidate["initialization_matched"]
                     and candidate["control_support_exact"]
                     and candidate["matcher_frozen_exact"] and correspondence and pose)
        candidate["fold_correspondence_pass"] = correspondence
        candidate["fold_pose_pass"] = pose
        candidate["fold_sam_memory_causal_pass"] = passed
        passes.append(passed)
    all_pass = int(all(passes))
    for row in rows:
        if row["architecture"] == "sam_memory" and row["variant"] == "normal":
            row["all_folds_sam_memory_causal_pass"] = all_pass


def _audit_v97_train_future(*, config, v97, dense, training_records, evaluation_records,
                            source_cache):
    rows = []
    for fold in FOLDS:
        checkpoint = config.v97_output_dir / "checkpoints" / f"{fold.name}_sam_dense.pt"
        saved = _torch_load(checkpoint, "completed V9.7 SAM checkpoint")
        expected_signature = _v97_checkpoint_signature(
            v97,
            fold=fold.name,
            architecture="sam_dense",
            source_cache=source_cache,
        )
        if saved.get("signature") != expected_signature:
            raise ValueError(
                f"V9.8 found a stale/provenance-incompatible V9.7 checkpoint={checkpoint}."
            )
        if bool(saved.get("uses_pose_loss", False)):
            raise ValueError(f"V9.8 refuses pose-trained V9.7 checkpoint={checkpoint}.")
        model = DenseSubgridMatcher(config.matcher).to(
            _training_device(config.training.device)
        )
        model.load_state_dict(saved["model"])
        model.eval()
        for split, records in (("train_prefix", training_records[fold.name]),
                               ("future_fixed_edges", evaluation_records[fold.name])):
            metrics = _correspondence_metrics(model, records, dense.sam_features, dense, config)
            rows.append({
                "fold": fold.name, "split": split,
                "descriptor_source": "sam31_detector_fpn2_dense",
                "pairs": len(records), "queries": metrics["queries"],
                "pck_threshold_pixels": config.matcher.pck_threshold_pixels,
                "pck_accuracy": metrics["pck"], "mean_epe_pixels": metrics["epe"],
                "checkpoint": str(checkpoint),
                "checkpoint_uses_pose_loss": int(bool(saved.get("uses_pose_loss", False))),
            })
    return rows


def _correspondence_metrics(model, records, features, dense, config):
    device = next(model.parameters()).device
    queries = correct = 0
    epe_sum = 0.0
    diagonal = math.hypot(*dense.support.image_size)
    with torch.no_grad():
        for record in records:
            query = features[record.current].index_select(0, record.query_grid_indices)
            key = features[record.history]
            q_valid = torch.ones(1, len(query), dtype=torch.bool, device=device)
            k_valid = torch.ones(1, len(key), dtype=torch.bool, device=device)
            decoded = decode_dense_matches(
                model, model(query[None].to(device), key[None].to(device), q_valid, k_valid),
                grid_size=config.grid_size, image_size=dense.support.image_size,
                query_valid=q_valid,
            )
            accepted = decoded.accepted[0].cpu()
            error = torch.linalg.vector_norm(
                decoded.history_uv_pixels[0].cpu() - record.correspondences.history_uv, dim=-1
            )
            queries += len(query)
            correct += int((accepted & error.le(config.matcher.pck_threshold_pixels)).sum())
            epe_sum += float(torch.where(accepted, error, torch.full_like(error, diagonal)).sum())
    return {"queries": queries, "pck": _ratio(correct, queries), "epe": _ratio(epe_sum, queries)}


def _architecture_features(architecture, temporal):
    if architecture in {"sam_memory", "sam_memory_train_off"}:
        return temporal.memory
    if architecture == "memory_off_raw":
        return temporal.raw
    raise ValueError(f"Unknown V9.8 architecture={architecture!r}.")


def _perturb_features(variant, record, temporal, current, history):
    if variant == "normal":
        return current, history, 0
    if variant == "memory_off":
        return temporal.raw[record.current].clone(), temporal.raw[record.history].clone(), 1
    if variant == "channel_permute":
        permutation = deterministic_channel_permutation(current.shape[-1])
        return current[:, permutation], history[:, permutation], 1
    if variant == "shuffle_memory_time":
        current_index = max(record.current - 3, 1)
        history_index = max(record.history - 2, 1)
        if current_index == record.current:
            current_index = 1
        if history_index == record.history:
            history_index = 1
        return temporal.memory[current_index].clone(), temporal.memory[history_index].clone(), 1
    raise ValueError(f"Unknown V9.8 temporal variant={variant!r}.")


def _read_v97_summary(output_dir):
    path = output_dir / "v97_dense_descriptor_summary.csv"
    if not path.is_file():
        raise FileNotFoundError(f"V9.8 requires completed V9.7 summary: {path}")
    with path.open("r", encoding="utf8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 21:
        raise ValueError(f"V9.8 expected 21 V9.7 summary rows, found {len(rows)}.")
    return rows


def _decision_markdown(rows, v97_rows, audit_rows):
    candidates = [row for row in rows if row["architecture"] == "sam_memory" and row["variant"] == "normal"]
    all_pass = int(len(candidates) == len(FOLDS) and all(row["all_folds_sam_memory_causal_pass"] for row in candidates))
    lines = [
        "# V9.8 SAM3.1 temporal-memory causality decision", "",
        "SAM3.1/StreamVGGT are frozen. Only a correspondence matcher is trained with GT 2D labels; no pose model or pose loss is used.",
        "The candidate is the SAM3.1 multiplex feature after it reads past mask-memory/object pointers. Memory-off is the same-call propagation feature before that read.",
        "Pose uses the frozen V9.3 O-R1 solver and the exact V9.7 full-grid support/fixed history edges.", "",
        "The identical V9.7 zero-input checkpoint is reused rather than retrained.", "",
        f"- temporal-memory all-fold causal pass: `{all_pass}`", "",
        "## V9.7 detector-FPN train/future audit", "",
        "| fold | split | PCK@1 | EPE px |", "|---|---|---:|---:|",
    ]
    for row in audit_rows:
        lines.append(f"| {row['fold']} | {row['split']} | {float(row['pck_accuracy']):.6g} | {float(row['mean_epe_pixels']):.6g} |")
    lines.extend(["", "## V9.8 future-fold result", "",
                  "| fold | method | variant | PCK@1 | EPE px | R gain | t-dir gain | worse | corr pass | pose pass | causal pass |",
                  "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for fold in FOLDS:
        for row in [value for value in rows if value["fold"] == fold.name]:
            lines.append(
                f"| {fold.name} | {row['architecture']} | {row['variant']} | {float(row['pck_accuracy']):.6g} "
                f"| {float(row['mean_epe_pixels']):.6g} | {float(row['rotation_gain_percent']):.6g} "
                f"| {float(row['translation_direction_gain_percent']):.6g} | {row['relative_aggregate_worse_frames']} "
                f"| {row['fold_correspondence_pass']} | {row['fold_pose_pass']} | {row['fold_sam_memory_causal_pass']} |"
            )
    lines.extend(["", "V9.7 future controls used by the gate:", "",
                  "| fold | control | PCK@1 | EPE px | refined aggregate |",
                  "|---|---|---:|---:|---:|"])
    for fold in FOLDS:
        for row in v97_rows:
            if row["fold"] == fold.name and row["variant"] == "normal" and row["architecture"] in {"sam_dense", "stream_dense"}:
                lines.append(f"| {fold.name} | {row['architecture']} | {float(row['pck_accuracy']):.6g} | {float(row['mean_epe_pixels']):.6g} | {float(row['refined_relative_aggregate_deg']):.6g} |")
    lines.extend(["", "Interpretation:", "",
                  "- pass=1: the genuine causal SAM memory-read feature beats same-call memory-off, detector-FPN, StreamVGGT and train-off controls on every future fold; disabling/shuffling it also destroys correspondence and pose gain.",
                  "- train improves but future correspondence pass=0: temporal memory is still being fit without predictive future correspondence evidence.",
                  "- correspondence pass=1 but pose pass=0: the descriptor helps matching but not the locked pose pipeline; do not claim pose improvement.",
                  "- This is one-scene temporal extrapolation and supports only relative rotation/translation-direction claims.", ""])
    return "\n".join(lines)


def load_v98_config(path: str | Path) -> V98Config:
    source = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf8")) or {}
    repository = source.parents[2]
    matcher_raw = raw.get("matcher", {})
    training_raw = raw.get("training", {})
    config = V98Config(
        source_path=source,
        v97_config=_resolve_path(repository, raw.get("v97_config", "streaming_couping/configs/v97_dense_descriptor_causality.yaml")),
        data_config=_resolve_path(repository, raw.get("data_config", "streaming_couping/configs/v92_support_data.yaml")),
        output_dir=_resolve_path(repository, raw.get("output_dir", "outputs/streaming_couping_v98_temporal_memory_causality")),
        temporal_cache_path=_resolve_path(repository, raw.get("temporal_cache_path", "outputs/streaming_couping_v98_temporal_memory_causality/cache/00a231a370_90_525_step15_temporal.pt")),
        v97_dense_cache_path=_resolve_path(repository, raw.get("v97_dense_cache_path", "outputs/streaming_couping_v97_dense_descriptor_causality/cache/00a231a370_90_525_step15_dense.pt")),
        v97_output_dir=_resolve_path(repository, raw.get("v97_output_dir", "outputs/streaming_couping_v97_dense_descriptor_causality")),
        clip_name=str(raw.get("clip_name", "00a231a370_90_525_step15_37_68_54")),
        grid_size=tuple(int(value) for value in raw.get("grid_size", [72, 72])),
        cache_device=str(raw.get("cache_device", "cuda:0")),
        matcher=DenseMatcherConfig(
            canonical_dim=int(matcher_raw.get("canonical_dim", 256)),
            projection_dim=int(matcher_raw.get("projection_dim", 64)),
            offset_hidden_dim=int(matcher_raw.get("offset_hidden_dim", 128)),
            temperature=float(matcher_raw.get("temperature", 0.07)),
            offset_weight=float(matcher_raw.get("offset_weight", 1.0)),
            pck_threshold_pixels=float(matcher_raw.get("pck_threshold_pixels", 1.0)),
        ),
        training=DenseTrainingConfig(
            device=str(training_raw.get("device", "cuda:0")),
            seed=int(training_raw.get("seed", 97)),
            steps=int(training_raw.get("steps", 600)),
            batch_pairs=int(training_raw.get("batch_pairs", 4)),
            learning_rate=float(training_raw.get("learning_rate", 3e-4)),
            weight_decay=float(training_raw.get("weight_decay", 1e-4)),
            log_every=int(training_raw.get("log_every", 50)),
        ),
    )
    if config.grid_size != (72, 72) or config.matcher.canonical_dim != 256:
        raise ValueError("V9.8 grid/canonical dimension are locked to 72x72/256.")
    if config.training.seed != 97 or config.training.steps != 600 or config.training.batch_pairs != 4:
        raise ValueError("V9.8 seed/steps/batch are locked to 97/600/4 for exact V9.7 initialization matching.")
    config.matcher.validate()
    return config


def _require_temporal_support(temporal, training, evaluation, frames):
    used = sorted({index for groups in (training, evaluation) for records in groups.values() for record in records for index in (record.current, record.history)})
    missing = [frames[index] for index in used if not bool(temporal.valid[index])]
    if missing:
        raise ValueError(f"V9.8 temporal cache lacks used frames={missing}.")


def _validate_outputs(summaries, frames, audit):
    expected_summary = len(FOLDS) * (len(TEMPORAL_VARIANTS) + len(TEMPORAL_ARCHITECTURES) - 1)
    if len(summaries) != expected_summary or len(frames) != expected_summary * 4:
        raise ValueError("V9.8 output row count differs from the locked protocol.")
    if len(audit) != len(FOLDS) * 2:
        raise ValueError("V9.8 V9.7 audit must contain train/future for every fold.")


def _write_csv(path, rows, columns):
    if not rows:
        raise ValueError(f"Refusing to write empty V9.8 CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = set(columns)
    for index, row in enumerate(rows):
        if set(row) != expected:
            raise ValueError(f"V9.8 CSV schema mismatch row={index}: missing={expected-set(row)}, extra={set(row)-expected}")
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _checkpoint_signature(config, fold, architecture, source_cache):
    value = {
        "experiment": "v98_temporal_memory_causality", "protocol_version": 1,
        "fold": fold, "architecture": architecture, "matcher": asdict(config.matcher),
        "training": asdict(config.training), "source_cache": _file_provenance(source_cache),
        "temporal_cache": _file_provenance(config.temporal_cache_path),
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf8")).hexdigest()


def _cyclic_batch_indices(count, batch_size, step, seed):
    generator = random.Random(seed + step * 1_000_003)
    return generator.sample(range(count), batch_size) if count >= batch_size else [generator.randrange(count) for _ in range(batch_size)]


def _training_device(value):
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("V9.8 CUDA unavailable; using CPU")
        return torch.device("cpu")
    return device


def _seed_everything(seed):
    random.seed(int(seed)); torch.manual_seed(int(seed))
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(int(seed))


def _grad_norm(value):
    return 0.0 if value.grad is None else float(torch.linalg.vector_norm(value.grad))


def _module_grad_norm(module):
    values = [value.grad.detach().flatten() for value in module.parameters() if value.grad is not None]
    return 0.0 if not values else float(torch.linalg.vector_norm(torch.cat(values)))


def _gain(initial, final):
    return 0.0 if not math.isfinite(float(initial)) or not math.isfinite(float(final)) or float(initial) <= 1e-12 else 100.0 * (float(initial) - float(final)) / float(initial)


def _ratio(numerator, denominator):
    return 0.0 if float(denominator) <= 0 else float(numerator) / float(denominator)


def _finite_mean(values: Iterable[float]):
    rows = [float(value) for value in values if math.isfinite(float(value))]
    return float("nan") if not rows else sum(rows) / len(rows)


def _file_provenance(path):
    path = Path(path).resolve(); stat = path.stat()
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _jsonable_config(config):
    value = asdict(config)
    for key in ("source_path", "v97_config", "data_config", "output_dir", "temporal_cache_path", "v97_dense_cache_path", "v97_output_dir"):
        value[key] = str(value[key])
    return value


def _resolve_path(repository, value):
    path = Path(value).expanduser()
    return (repository / path).resolve() if not path.is_absolute() else path.resolve()


def _torch_load(path, label):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"V9.8 requires {label}: {path}")
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


if __name__ == "__main__":
    main()
