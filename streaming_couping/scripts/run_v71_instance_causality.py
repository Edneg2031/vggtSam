#!/usr/bin/env python3
"""Measure causal instance gains over a frozen camera-only V7 baseline."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import platform
import socket
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

from streaming_couping.src.config import load_config
from streaming_couping.src.external_repos import maybe_add_repo_to_path
from streaming_couping.src.learned_pose.config import load_learned_pose_config
from streaming_couping.src.learned_pose.v7_fusion import (
    V7_INPUT_VARIANTS,
    V7PoseFusion,
    perturb_v7_inputs,
)
from streaming_couping.src.learned_pose.v71_causal_fusion import (
    V71_DESCRIPTIONS,
    V71_RESIDUAL_ARCHITECTURES,
    V71FrozenResidualFusion,
    build_common_instance_state,
)
from streaming_couping.scripts.run_v7_fusion_ablation import (
    V7ExperimentConfig,
    _camera_batch,
    _evaluate,
    _evaluation_indices,
    _find_clip,
    _load_cache,
    _pose_metrics,
    _predict,
    _seed_everything,
    _train_model,
    load_v7_config,
)


@dataclass(frozen=True)
class V71Config:
    source_path: Path
    v7_config: Path
    output_dir: Path
    device: str
    long_clip_name: str
    validation_clip_name: str
    external_clip_name: str
    base_train_frames: tuple[int, ...]
    residual_train_frames: tuple[int, ...]
    development_frames: tuple[int, ...]
    future_frames: tuple[int, ...]
    architectures: tuple[str, ...]
    base_steps: int
    residual_steps: int
    seed: int


@dataclass
class EvaluationSpec:
    name: str
    frames: tuple[int, ...]
    batch: dict[str, torch.Tensor]
    baseline_w2c: torch.Tensor
    target_w2c: torch.Tensor
    reference_index: int
    evaluation_indices: list[int] | None
    raw_metrics: dict[str, float]


def main() -> None:
    args = _parse_args()
    config = load_v71_config(args.config)
    if args.device:
        config = replace(config, device=args.device)
    if args.output_dir:
        config = replace(
            config,
            output_dir=Path(args.output_dir).expanduser().resolve(),
        )
    if args.seed is not None:
        config = replace(config, seed=int(args.seed))
    if args.base_steps is not None:
        config = replace(config, base_steps=int(args.base_steps))
    if args.residual_steps is not None:
        config = replace(config, residual_steps=int(args.residual_steps))
    if args.architectures:
        requested = tuple(
            value.strip()
            for value in args.architectures.split(",")
            if value.strip()
        )
        unknown = set(requested) - set(V71_RESIDUAL_ARCHITECTURES)
        if not requested or unknown:
            raise ValueError(
                f"Invalid --architectures selection; unknown={sorted(unknown)}"
            )
        controls = (
            "camera_extra_all",
            "camera_extra_common_gate",
        )
        selected = controls + tuple(
            value for value in requested if value not in controls
        )
        config = replace(config, architectures=selected)
    if config.base_steps < 1 or config.residual_steps < 1:
        raise ValueError("V7.1 training steps must be positive.")
    result = run_v71(config, resume=args.resume)
    print(f"V7.1 one-table result={result}")


def run_v71(config: V71Config, *, resume: bool = False) -> Path:
    base_v7 = load_v7_config(config.v7_config)
    base_v7 = replace(
        base_v7,
        device=config.device,
        training=replace(base_v7.training, seed=config.seed),
    )
    data = load_learned_pose_config(base_v7.data_config)
    long_clip = _find_clip(data, config.long_clip_name)
    validation_clip = _find_clip(data, config.validation_clip_name)
    external_clip = _find_clip(data, config.external_clip_name)
    long_payload = _load_cache(data, long_clip)
    validation_payload = _load_cache(data, validation_clip)
    external_payload = _load_cache(data, external_clip)
    for name, payload in (
        ("long", long_payload),
        ("validation", validation_payload),
        ("cross", external_payload),
    ):
        if str(payload.get("sam_version")) != "sam3.1":
            raise ValueError(
                f"V7.1 {name} cache is not SAM3.1: "
                f"{payload.get('sam_version')!r}."
            )

    recovery = load_config(data.recovery_config)
    maybe_add_repo_to_path(recovery.streamvggt_repo)
    from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri

    long_batch, long_baseline, long_target = _prepare_payload(
        long_payload,
        config=base_v7,
        pose_decoder=pose_encoding_to_extri_intri,
    )
    validation_batch, validation_baseline, validation_target = (
        _prepare_payload(
            validation_payload,
            config=base_v7,
            pose_decoder=pose_encoding_to_extri_intri,
        )
    )
    external_batch, external_baseline, external_target = _prepare_payload(
        external_payload,
        config=base_v7,
        pose_decoder=pose_encoding_to_extri_intri,
    )
    long_positions = {
        int(frame): index
        for index, frame in enumerate(long_payload["frame_indices"])
    }
    frame_groups = {
        "base_train": config.base_train_frames,
        "residual_train": config.residual_train_frames,
        "development": config.development_frames,
        "future": config.future_frames,
    }
    _validate_temporal_partition(
        frame_groups,
        frame_indices=tuple(
            int(value) for value in long_payload["frame_indices"]
        ),
        reference_index=int(long_payload["reference_sequence_index"]),
    )
    temporal_indices = {
        name: [long_positions[int(frame)] for frame in frames]
        for name, frames in frame_groups.items()
    }
    base_prefix_length = long_positions[max(config.base_train_frames)] + 1
    residual_prefix_length = (
        long_positions[max(config.residual_train_frames)] + 1
    )
    base_training_batch = _slice_batch_prefix(
        long_batch,
        length=base_prefix_length,
    )
    residual_training_batch = _slice_batch_prefix(
        long_batch,
        length=residual_prefix_length,
    )
    specs = [
        _make_spec(
            name=name,
            frames=frames,
            batch=long_batch,
            baseline=long_baseline,
            target=long_target,
            reference_index=int(long_payload["reference_sequence_index"]),
            evaluation_indices=temporal_indices[name],
            translation_weight=base_v7.training.translation_weight,
        )
        for name, frames in frame_groups.items()
    ]
    specs.extend(
        [
            _make_spec(
                name="validation",
                frames=tuple(
                    int(value)
                    for index, value in enumerate(
                        validation_payload["frame_indices"]
                    )
                    if index
                    != int(validation_payload["reference_sequence_index"])
                ),
                batch=validation_batch,
                baseline=validation_baseline,
                target=validation_target,
                reference_index=int(
                    validation_payload["reference_sequence_index"]
                ),
                evaluation_indices=None,
                translation_weight=base_v7.training.translation_weight,
            ),
            _make_spec(
                name="cross",
                frames=tuple(
                    int(value)
                    for index, value in enumerate(
                        external_payload["frame_indices"]
                    )
                    if index
                    != int(external_payload["reference_sequence_index"])
                ),
                batch=external_batch,
                baseline=external_baseline,
                target=external_target,
                reference_index=int(
                    external_payload["reference_sequence_index"]
                ),
                evaluation_indices=None,
                translation_weight=base_v7.training.translation_weight,
            ),
        ]
    )

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    _seed_everything(config.seed)
    base_model = V7PoseFusion(
        architecture="l0_camera_only",
        camera_dim=int(long_batch["camera_hidden"].shape[-1]),
        appearance_dim=int(long_batch["appearance"].shape[-1]),
        geometry_dim=int(long_batch["pose_geometry"].shape[-1]),
        local_feature_dim=int(long_batch["local_features"].shape[-1]),
        config=base_v7.fusion,
    ).to(config.device)
    base_training_config = replace(
        base_v7,
        training=replace(base_v7.training, steps=config.base_steps),
    )
    base_checkpoint_path = output_dir / "frozen_l0.pt"
    base_signature = _checkpoint_signature(
        config,
        base_v7,
        architecture="frozen_l0",
    )
    if resume and base_checkpoint_path.is_file():
        checkpoint = _load_checkpoint(
            base_checkpoint_path,
            expected_signature=base_signature,
        )
        base_model.load_state_dict(checkpoint["model"])
        base_seconds = float(checkpoint.get("training_seconds", 0.0))
        base_training = dict(checkpoint.get("training_result", {}))
        print(f"V7.1 resumed frozen L0: {base_checkpoint_path}")
    else:
        print(
            "V7.1 stage A: train camera-only on frames="
            + " ".join(str(value) for value in config.base_train_frames)
        )
        base_started = time.perf_counter()
        base_training = _train_model(
            base_model,
            batch=base_training_batch,
            baseline_w2c=long_baseline[:, :base_prefix_length],
            target_w2c=long_target[:, :base_prefix_length],
            reference_index=int(long_payload["reference_sequence_index"]),
            config=base_training_config,
            training_indices=temporal_indices["base_train"],
        )
        base_seconds = time.perf_counter() - base_started
    for parameter in base_model.parameters():
        parameter.requires_grad_(False)
    base_model.eval()
    if not (resume and base_checkpoint_path.is_file()):
        _save_checkpoint_atomic(
            {
                "signature": base_signature,
                "architecture": "frozen_l0",
                "model": {
                    name: value.detach().cpu()
                    for name, value in base_model.state_dict().items()
                },
                "training_frames": config.base_train_frames,
                "training_seconds": base_seconds,
                "training_result": base_training,
            },
            base_checkpoint_path,
        )

    base_metrics = {
        spec.name: _evaluate_normal(base_model, spec, base_v7)
        for spec in specs
    }
    base_outputs = {
        spec.name: _predict_normal(base_model, spec)
        for spec in specs
    }
    gate_diagnostics = {
        spec.name: _gate_diagnostics(spec, base_v7)
        for spec in specs
    }
    diagnostic_rows: list[dict[str, object]] = []
    for spec in specs:
        _append_frame_diagnostics(
            diagnostic_rows,
            architecture="raw_streamvggt",
            spec=spec,
            predicted=spec.baseline_w2c,
            base_predicted=base_outputs[spec.name]["world_to_camera"],
            active=None,
            gate=gate_diagnostics[spec.name],
            config=base_v7,
        )
        _append_frame_diagnostics(
            diagnostic_rows,
            architecture="frozen_l0",
            spec=spec,
            predicted=base_outputs[spec.name]["world_to_camera"],
            base_predicted=base_outputs[spec.name]["world_to_camera"],
            active=base_outputs[spec.name]["active_frames"],
            gate=gate_diagnostics[spec.name],
            config=base_v7,
        )
    rows = [
        _control_row(
            architecture="raw_streamvggt",
            evidence="raw_control",
            specs=specs,
            metrics={spec.name: spec.raw_metrics for spec in specs},
            base_metrics=base_metrics,
            parameters=0,
            trainable_parameters=0,
            training_seconds=0.0,
        ),
        _control_row(
            architecture="frozen_l0",
            evidence="camera_baseline_trained_on_early_frames",
            specs=specs,
            metrics=base_metrics,
            base_metrics=base_metrics,
            parameters=sum(value.numel() for value in base_model.parameters()),
            trainable_parameters=0,
            training_seconds=base_seconds,
        ),
    ]

    residual_training_config = replace(
        base_v7,
        training=replace(base_v7.training, steps=config.residual_steps),
    )
    for architecture in config.architectures:
        _seed_everything(config.seed)
        model = V71FrozenResidualFusion(
            base_model=copy.deepcopy(base_model),
            architecture=architecture,
            appearance_dim=int(long_batch["appearance"].shape[-1]),
            geometry_dim=int(long_batch["pose_geometry"].shape[-1]),
            local_feature_dim=int(long_batch["local_features"].shape[-1]),
            config=base_v7.fusion,
        ).to(config.device)
        checkpoint_path = output_dir / f"{architecture}.pt"
        signature = _checkpoint_signature(
            config,
            base_v7,
            architecture=architecture,
        )
        if resume and checkpoint_path.is_file():
            checkpoint = _load_checkpoint(
                checkpoint_path,
                expected_signature=signature,
            )
            model.load_state_dict(checkpoint["model"])
            training_seconds = float(
                checkpoint.get("training_seconds", 0.0)
            )
            training_result = dict(
                checkpoint.get("training_result", {})
            )
            print(f"V7.1 resumed architecture={architecture}")
        else:
            print(
                f"V7.1 stage B: architecture={architecture} frames="
                + " ".join(
                    str(value) for value in config.residual_train_frames
                )
            )
            started = time.perf_counter()
            training_result = _train_model(
                model,
                batch=residual_training_batch,
                baseline_w2c=long_baseline[:, :residual_prefix_length],
                target_w2c=long_target[:, :residual_prefix_length],
                reference_index=int(
                    long_payload["reference_sequence_index"]
                ),
                config=residual_training_config,
                training_indices=temporal_indices["residual_train"],
            )
            training_seconds = time.perf_counter() - started
        metrics_by_split = {
            spec.name: {
                variant: _evaluate_variant(
                    model,
                    spec,
                    base_v7,
                    variant=variant,
                )
                for variant in V7_INPUT_VARIANTS
            }
            for spec in specs
        }
        total_parameters = sum(
            value.numel() for value in model.parameters()
        )
        trainable_parameters = sum(
            value.numel()
            for value in model.parameters()
            if value.requires_grad
        )
        if not (resume and checkpoint_path.is_file()):
            _save_checkpoint_atomic(
                {
                    "signature": signature,
                    "architecture": architecture,
                    "model": {
                        name: value.detach().cpu()
                        for name, value in model.state_dict().items()
                    },
                    "base_training_frames": config.base_train_frames,
                    "residual_training_frames": (
                        config.residual_train_frames
                    ),
                    "fusion": asdict(base_v7.fusion),
                    "total_parameters": total_parameters,
                    "trainable_parameters": trainable_parameters,
                    "training_seconds": training_seconds,
                    "training_result": training_result,
                },
                checkpoint_path,
            )
        rows.append(
            _residual_row(
                architecture=architecture,
                specs=specs,
                metrics_by_split=metrics_by_split,
                base_metrics=base_metrics,
                parameters=total_parameters,
                trainable_parameters=trainable_parameters,
                training_seconds=training_seconds,
            )
        )
        for spec in specs:
            normal_output = _predict_normal(model, spec)
            _append_frame_diagnostics(
                diagnostic_rows,
                architecture=architecture,
                spec=spec,
                predicted=normal_output["world_to_camera"],
                base_predicted=base_outputs[spec.name]["world_to_camera"],
                active=normal_output["active_frames"],
                gate=gate_diagnostics[spec.name],
                config=base_v7,
            )

    _label_rows(rows)
    path = output_dir / "v71_instance_causality.csv"
    _write_csv(path, rows)
    diagnostic_path = output_dir / "v71_frame_diagnostics.csv"
    _write_csv(diagnostic_path, diagnostic_rows)
    _write_run_metadata(
        output_dir / "run_metadata.json",
        config=config,
        base_v7=base_v7,
        resume=resume,
        result_path=path,
        diagnostic_path=diagnostic_path,
    )
    print("V7.1 INSTANCE CAUSALITY (single CSV)")
    print(path.read_text(encoding="utf8").rstrip())
    return path


def _prepare_payload(
    payload: dict,
    *,
    config: V7ExperimentConfig,
    pose_decoder,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    batch = _camera_batch(
        payload,
        device=config.device,
        local_token_count=config.local_token_count,
    )
    image_size = tuple(int(value) for value in payload["image_size"])
    baseline, _ = pose_decoder(
        batch["baseline_pose_encoding"],
        image_size_hw=image_size,
    )
    target, _ = pose_decoder(
        batch["target_pose_encoding"],
        image_size_hw=image_size,
    )
    return batch, baseline, target


def _slice_batch_prefix(
    batch: dict[str, torch.Tensor],
    *,
    length: int,
) -> dict[str, torch.Tensor]:
    sequence = int(batch["camera_hidden"].shape[1])
    if length < 2 or length > sequence:
        raise ValueError(
            f"V7.1 prefix length {length} is outside [2,{sequence}]."
        )
    output = {}
    for name, value in batch.items():
        if value.ndim < 2 or value.shape[1] != sequence:
            raise ValueError(
                f"V7.1 batch field {name!r} lacks sequence dimension 1."
            )
        output[name] = value[:, :length]
    return output


def _make_spec(
    *,
    name: str,
    frames: tuple[int, ...],
    batch: dict[str, torch.Tensor],
    baseline: torch.Tensor,
    target: torch.Tensor,
    reference_index: int,
    evaluation_indices: list[int] | None,
    translation_weight: float,
) -> EvaluationSpec:
    return EvaluationSpec(
        name=name,
        frames=frames,
        batch=batch,
        baseline_w2c=baseline,
        target_w2c=target,
        reference_index=reference_index,
        evaluation_indices=evaluation_indices,
        raw_metrics=_pose_metrics(
            baseline,
            target,
            reference_index=reference_index,
            translation_weight=translation_weight,
            evaluation_indices=evaluation_indices,
        ),
    )


def _evaluate_normal(
    model,
    spec: EvaluationSpec,
    config: V7ExperimentConfig,
) -> dict[str, float | int]:
    return _evaluate_variant(model, spec, config, variant="normal")


def _predict_normal(model, spec: EvaluationSpec) -> dict[str, torch.Tensor]:
    return _predict(
        model,
        batch=spec.batch,
        baseline_w2c=spec.baseline_w2c,
        reference_index=spec.reference_index,
        variant="normal",
    )


def _gate_diagnostics(
    spec: EvaluationSpec,
    config: V7ExperimentConfig,
) -> dict[str, torch.Tensor]:
    inputs = perturb_v7_inputs(spec.batch, "normal")
    state = build_common_instance_state(
        appearance=inputs["appearance"],
        geometry=inputs["pose_geometry"],
        quality=inputs["quality"],
        observed=inputs["observed"],
        identity_valid=inputs["identity_valid"],
        identity_unknown=inputs["identity_unknown"],
        config=config.fusion,
    )
    return {
        "valid": state.valid.detach(),
        "reliability": state.reliability.detach(),
    }


def _append_frame_diagnostics(
    rows: list[dict[str, object]],
    *,
    architecture: str,
    spec: EvaluationSpec,
    predicted: torch.Tensor,
    base_predicted: torch.Tensor,
    active: torch.Tensor | None,
    gate: dict[str, torch.Tensor],
    config: V7ExperimentConfig,
) -> None:
    indices = _evaluation_indices(
        predicted.shape[1],
        reference_index=spec.reference_index,
        evaluation_indices=spec.evaluation_indices,
    )
    if len(indices) != len(spec.frames):
        raise ValueError(
            f"V7.1 frame diagnostics mismatch for {spec.name}: "
            f"{len(indices)} indices vs {len(spec.frames)} frames."
        )
    for frame, index in zip(spec.frames, indices):
        raw = _pose_metrics(
            spec.baseline_w2c,
            spec.target_w2c,
            reference_index=spec.reference_index,
            translation_weight=config.training.translation_weight,
            evaluation_indices=[index],
        )
        base = _pose_metrics(
            base_predicted,
            spec.target_w2c,
            reference_index=spec.reference_index,
            translation_weight=config.training.translation_weight,
            evaluation_indices=[index],
        )
        current = _pose_metrics(
            predicted,
            spec.target_w2c,
            reference_index=spec.reference_index,
            translation_weight=config.training.translation_weight,
            evaluation_indices=[index],
        )
        correction = _pose_metrics(
            predicted,
            base_predicted,
            reference_index=spec.reference_index,
            translation_weight=config.training.translation_weight,
            evaluation_indices=[index],
        )
        valid = gate["valid"][0, index].bool()
        reliability = gate["reliability"][0, index]
        usable = int(valid.sum().cpu())
        mean_reliability = (
            float(reliability[valid].mean().cpu()) if usable else 0.0
        )
        rows.append(
            {
                "architecture": architecture,
                "split": spec.name,
                "frame": int(frame),
                "usable_instances": usable,
                "mean_instance_reliability": _short(mean_reliability),
                "active": int(
                    bool(active[0, index].cpu())
                    if active is not None
                    else False
                ),
                "raw_rotation_deg": _short(raw["rotation_degrees"]),
                "raw_translation": _short(raw["translation_native"]),
                "raw_loss": _short(raw["loss"]),
                "frozen_l0_rotation_deg": _short(
                    base["rotation_degrees"]
                ),
                "frozen_l0_translation": _short(
                    base["translation_native"]
                ),
                "frozen_l0_loss": _short(base["loss"]),
                "rotation_deg": _short(current["rotation_degrees"]),
                "translation": _short(current["translation_native"]),
                "loss": _short(current["loss"]),
                "gain_vs_frozen_l0_percent": _short(
                    _gain(float(base["loss"]), float(current["loss"]))
                ),
                "correction_from_l0_rotation_deg": _short(
                    correction["rotation_degrees"]
                ),
                "correction_from_l0_center": _short(
                    correction["translation_native"]
                ),
            }
        )


def _evaluate_variant(
    model,
    spec: EvaluationSpec,
    config: V7ExperimentConfig,
    *,
    variant: str,
) -> dict[str, float | int]:
    return _evaluate(
        model,
        batch=spec.batch,
        baseline_w2c=spec.baseline_w2c,
        target_w2c=spec.target_w2c,
        reference_index=spec.reference_index,
        evaluation_indices=spec.evaluation_indices,
        variant=variant,
        translation_weight=config.training.translation_weight,
    )


def _control_row(
    *,
    architecture: str,
    evidence: str,
    specs: list[EvaluationSpec],
    metrics: dict[str, dict[str, float | int]],
    base_metrics: dict[str, dict[str, float | int]],
    parameters: int,
    trainable_parameters: int,
    training_seconds: float,
) -> dict[str, object]:
    row: dict[str, object] = {
        "architecture": architecture,
        "evidence": evidence,
        "instance_content": 0,
        "spatial_tokens": 0,
        "parameters": parameters,
        "trainable_parameters": trainable_parameters,
        "training_seconds": _short(training_seconds),
        "development_score": "",
        "development_best": 0,
        "causal_instance_pass": 0,
    }
    for spec in specs:
        current = metrics[spec.name]
        base = base_metrics[spec.name]
        _add_normal_metrics(row, spec.name, current, spec, base)
        _add_empty_perturbations(row, spec.name)
    return row


def _residual_row(
    *,
    architecture: str,
    specs: list[EvaluationSpec],
    metrics_by_split: dict[
        str,
        dict[str, dict[str, float | int]],
    ],
    base_metrics: dict[str, dict[str, float | int]],
    parameters: int,
    trainable_parameters: int,
    training_seconds: float,
) -> dict[str, object]:
    description = V71_DESCRIPTIONS[architecture]
    row: dict[str, object] = {
        "architecture": architecture,
        "evidence": description.evidence,
        "instance_content": int(description.instance_content),
        "spatial_tokens": description.spatial_tokens,
        "parameters": parameters,
        "trainable_parameters": trainable_parameters,
        "training_seconds": _short(training_seconds),
        "development_score": "",
        "development_best": 0,
        "causal_instance_pass": 0,
    }
    for spec in specs:
        variants = metrics_by_split[spec.name]
        normal = variants["normal"]
        base = base_metrics[spec.name]
        _add_normal_metrics(row, spec.name, normal, spec, base)
        for variant in V7_INPUT_VARIANTS[1:]:
            row[f"{spec.name}_{variant}_loss"] = _short(
                float(variants[variant]["loss"])
            )
        wrong = float(variants["wrong_geometry"]["loss"])
        shuffle = float(variants["shuffle_time"]["loss"])
        normal_loss = float(normal["loss"])
        row[f"{spec.name}_wrong_geometry_damage_percent"] = _short(
            _relative_damage(normal_loss, wrong)
        )
        row[f"{spec.name}_shuffle_time_damage_percent"] = _short(
            _relative_damage(normal_loss, shuffle)
        )
    return row


def _add_normal_metrics(
    row: dict[str, object],
    prefix: str,
    metrics: dict[str, float | int],
    spec: EvaluationSpec,
    base: dict[str, float | int],
) -> None:
    loss = float(metrics["loss"])
    row[f"{prefix}_frames"] = " ".join(str(value) for value in spec.frames)
    row[f"{prefix}_rotation_deg"] = _short(
        float(metrics["rotation_degrees"])
    )
    row[f"{prefix}_translation"] = _short(
        float(metrics["translation_native"])
    )
    row[f"{prefix}_loss"] = _short(loss)
    row[f"{prefix}_gain_vs_raw_percent"] = _short(
        _gain(float(spec.raw_metrics["loss"]), loss)
    )
    row[f"{prefix}_gain_vs_frozen_l0_percent"] = _short(
        _gain(float(base["loss"]), loss)
    )
    row[f"{prefix}_active_frames"] = int(metrics.get("active_frames", 0))
    row[f"{prefix}_reference_exact"] = int(
        metrics.get("reference_exact", 1)
    )


def _add_empty_perturbations(row: dict[str, object], prefix: str) -> None:
    for variant in V7_INPUT_VARIANTS[1:]:
        row[f"{prefix}_{variant}_loss"] = ""
    row[f"{prefix}_wrong_geometry_damage_percent"] = ""
    row[f"{prefix}_shuffle_time_damage_percent"] = ""


def _label_rows(rows: list[dict[str, object]]) -> None:
    base = next(row for row in rows if row["architecture"] == "frozen_l0")
    development_splits = ("development", "validation")
    for row in rows:
        ratios = [
            float(row[f"{split}_loss"]) / float(base[f"{split}_loss"])
            for split in development_splits
        ]
        row["development_score"] = _short(sum(ratios) / len(ratios))
    best_score = min(float(row["development_score"]) for row in rows)
    camera_controls = [
        next(row for row in rows if row["architecture"] == architecture)
        for architecture in (
            "camera_extra_all",
            "camera_extra_common_gate",
        )
    ]
    for row in rows:
        row["development_best"] = int(
            abs(float(row["development_score"]) - best_score) <= 1e-10
        )
        architecture = str(row["architecture"])
        is_content = int(row["instance_content"]) == 1
        beats_base = all(
            float(row[f"{split}_loss"]) < float(base[f"{split}_loss"])
            for split in ("development", "validation", "future", "cross")
        )
        beats_capacity = all(
            float(row[f"{split}_loss"]) < float(control[f"{split}_loss"])
            for control in camera_controls
            for split in ("future", "cross")
        )
        instance_off_returns_base = all(
            abs(
                float(row[f"{split}_instance_off_loss"])
                - float(base[f"{split}_loss"])
            )
            <= 1e-6
            for split in ("development", "validation", "future", "cross")
        ) if architecture in V71_RESIDUAL_ARCHITECTURES else False
        row["causal_instance_pass"] = int(
            is_content
            and beats_base
            and beats_capacity
            and instance_off_returns_base
        )


def _validate_temporal_partition(
    groups: dict[str, tuple[int, ...]],
    *,
    frame_indices: tuple[int, ...],
    reference_index: int,
) -> None:
    expected = {
        int(frame)
        for index, frame in enumerate(frame_indices)
        if index != int(reference_index)
    }
    flattened = [int(frame) for frames in groups.values() for frame in frames]
    if len(flattened) != len(set(flattened)):
        raise ValueError("V7.1 temporal frame groups overlap.")
    if set(flattened) != expected:
        missing = sorted(expected - set(flattened))
        extra = sorted(set(flattened) - expected)
        raise ValueError(
            f"V7.1 temporal partition mismatch: missing={missing} extra={extra}."
        )
    ordered = [frame for frames in groups.values() for frame in frames]
    if ordered != sorted(ordered):
        raise ValueError("V7.1 temporal groups must be strictly chronological.")


def _checkpoint_signature(
    config: V71Config,
    base_v7: V7ExperimentConfig,
    *,
    architecture: str,
) -> str:
    payload = {
        "schema": 1,
        "architecture": architecture,
        "seed": config.seed,
        "long_clip_name": config.long_clip_name,
        "base_train_frames": config.base_train_frames,
        "residual_train_frames": config.residual_train_frames,
        "base_steps": config.base_steps,
        "residual_steps": config.residual_steps,
        "fusion": asdict(base_v7.fusion),
        "training": asdict(base_v7.training),
        "local_token_count": base_v7.local_token_count,
    }
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=list,
    ).encode("utf8")
    return hashlib.sha256(serialized).hexdigest()


def _load_checkpoint(
    path: Path,
    *,
    expected_signature: str,
) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"Invalid V7.1 checkpoint: {path}")
    actual = str(payload.get("signature", ""))
    if actual != expected_signature:
        raise ValueError(
            f"V7.1 checkpoint signature mismatch: {path}. "
            "Use V71_FRESH=1 with the command to retrain and overwrite it."
        )
    return payload


def _save_checkpoint_atomic(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_v71_config(path: str | Path) -> V71Config:
    source = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf8")) or {}
    frames = raw.get("frames", {})
    training = raw.get("training", {})
    config = V71Config(
        source_path=source,
        v7_config=_path(raw.get("v7_config")),
        output_dir=_path(raw.get("output_dir")),
        device=str(raw.get("device", "cuda:0")),
        long_clip_name=str(raw.get("long_clip_name", "")),
        validation_clip_name=str(raw.get("validation_clip_name", "")),
        external_clip_name=str(raw.get("external_clip_name", "")),
        base_train_frames=_frame_tuple(frames.get("base_train")),
        residual_train_frames=_frame_tuple(frames.get("residual_train")),
        development_frames=_frame_tuple(frames.get("development")),
        future_frames=_frame_tuple(frames.get("future")),
        architectures=tuple(
            str(value)
            for value in raw.get(
                "architectures",
                V71_RESIDUAL_ARCHITECTURES,
            )
        ),
        base_steps=int(training.get("base_steps", 1200)),
        residual_steps=int(training.get("residual_steps", 1200)),
        seed=int(training.get("seed", 0)),
    )
    unknown = set(config.architectures) - set(V71_RESIDUAL_ARCHITECTURES)
    if unknown:
        raise ValueError(
            "Unknown V7.1 architecture(s): " + ", ".join(sorted(unknown))
        )
    if len(config.architectures) != len(set(config.architectures)):
        raise ValueError("V7.1 architecture list contains duplicates.")
    missing_controls = {
        "camera_extra_all",
        "camera_extra_common_gate",
    } - set(config.architectures)
    if missing_controls:
        raise ValueError(
            "V7.1 is missing camera controls: "
            + ", ".join(sorted(missing_controls))
        )
    if config.base_steps < 1 or config.residual_steps < 1:
        raise ValueError("V7.1 training steps must be positive.")
    return config


def _frame_tuple(value: object) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("V7.1 frame groups must be non-empty lists.")
    return tuple(int(item) for item in value)


def _path(value: object) -> Path:
    if value is None or not str(value).strip():
        raise ValueError("V7.1 configuration contains an empty path.")
    return Path(str(value)).expanduser().resolve()


def _gain(baseline: float, current: float) -> float:
    return 0.0 if baseline <= 1e-12 else 100.0 * (baseline - current) / baseline


def _relative_damage(normal: float, perturbed: float) -> float:
    return 0.0 if normal <= 1e-12 else 100.0 * (perturbed - normal) / normal


def _short(value: float) -> str:
    return f"{float(value):.8g}"


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_run_metadata(
    path: Path,
    *,
    config: V71Config,
    base_v7: V7ExperimentConfig,
    resume: bool,
    result_path: Path,
    diagnostic_path: Path,
) -> None:
    cuda_devices = []
    if torch.cuda.is_available():
        cuda_devices = [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]
    metadata = {
        "schema": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_devices": cuda_devices,
        "peak_gpu_memory_gib": (
            torch.cuda.max_memory_allocated() / (1024**3)
            if torch.cuda.is_available()
            else 0.0
        ),
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_dirty": bool(_git_output("status", "--porcelain")),
        "resume": bool(resume),
        "experiment": asdict(config),
        "fusion": asdict(base_v7.fusion),
        "training": asdict(base_v7.training),
        "result_csv": str(result_path),
        "frame_diagnostics_csv": str(diagnostic_path),
    }
    path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf8",
    )


def _git_output(*args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return completed.stdout.strip()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="streaming_couping/configs/v71_instance_causality.yaml",
    )
    parser.add_argument("--device")
    parser.add_argument("--output-dir")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--base-steps", type=int)
    parser.add_argument("--residual-steps", type=int)
    parser.add_argument(
        "--architectures",
        help=(
            "Comma-separated residual subset. The two camera controls are "
            "automatically retained so causal labeling remains valid."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse matching completed L0/residual checkpoints.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
