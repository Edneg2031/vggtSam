#!/usr/bin/env python3
"""Train and compare V7 fusion architectures using V7-owned frozen caches."""

from __future__ import annotations

import argparse
import csv
import gc
import random
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from streaming_couping.src.config import load_config
from streaming_couping.src.external_repos import maybe_add_repo_to_path
from streaming_couping.src.learned_pose.cache import (
    cache_path,
    load_feature_cache,
)
from streaming_couping.src.learned_pose.config import (
    ClipConfig,
    LearnedPoseConfig,
    load_learned_pose_config,
)
from streaming_couping.src.learned_pose.pipeline import (
    _slice_training_payload,
)
from streaming_couping.src.learned_pose.v6_camera_fusion import (
    V6FusionConfig,
)
from streaming_couping.src.learned_pose.v7_fusion import (
    V7_ARCHITECTURES,
    V7_DESCRIPTIONS,
    V7_INPUT_VARIANTS,
    V7PoseFusion,
    perturb_v7_inputs,
)


@dataclass(frozen=True)
class V7TrainingConfig:
    steps: int = 1200
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    grad_clip_norm: float = 5.0
    translation_weight: float = 10.0
    seed: int = 0
    log_every: int = 200


@dataclass(frozen=True)
class V7ExperimentConfig:
    source_path: Path
    data_config: Path
    output_dir: Path
    training_clip_name: str
    validation_clip_name: str
    long_clip_name: str
    external_clip_name: str
    device: str
    architectures: tuple[str, ...]
    local_token_count: int
    fusion: V6FusionConfig
    training: V7TrainingConfig


@dataclass
class EvaluationSplit:
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
    config = load_v7_config(args.config)
    if args.device:
        config = replace(config, device=args.device)
    result = run_v7_ablation(config)
    print(f"V7 one-table result={result}")


def run_v7_ablation(config: V7ExperimentConfig) -> Path:
    _seed_everything(config.training.seed)
    data = load_learned_pose_config(config.data_config)
    train_clip = _find_clip(data, config.training_clip_name)
    validation_clip = _find_clip(data, config.validation_clip_name)
    long_clip = _find_clip(data, config.long_clip_name)
    external_clip = _find_clip(data, config.external_clip_name)

    train_full_payload = _load_cache(data, train_clip)
    validation_payload = _load_cache(data, validation_clip)
    long_payload = _load_cache(data, long_clip)
    external_payload = _load_cache(data, external_clip)
    for name, payload in (
        ("training", train_full_payload),
        ("validation", validation_payload),
        ("long30", long_payload),
        ("external", external_payload),
    ):
        if str(payload.get("sam_version")) != "sam3.1":
            raise ValueError(
                f"V7 {name} cache is not SAM3.1: "
                f"{payload.get('sam_version')!r}."
            )

    recovery = load_config(data.recovery_config)
    maybe_add_repo_to_path(recovery.streamvggt_repo)
    from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri

    training_payload = _slice_training_payload(
        train_full_payload,
        train_clip,
    )
    training_batch = _camera_batch(
        training_payload,
        device=config.device,
        local_token_count=config.local_token_count,
    )
    training_baseline, _ = pose_encoding_to_extri_intri(
        training_batch["baseline_pose_encoding"],
        image_size_hw=tuple(
            int(value) for value in training_payload["image_size"]
        ),
    )
    training_target, _ = pose_encoding_to_extri_intri(
        training_batch["target_pose_encoding"],
        image_size_hw=tuple(
            int(value) for value in training_payload["image_size"]
        ),
    )
    training_reference_index = int(
        training_payload["reference_sequence_index"]
    )
    splits = _build_splits(
        config,
        train_clip=train_clip,
        training_payload=training_payload,
        train_full_payload=train_full_payload,
        validation_payload=validation_payload,
        long_payload=long_payload,
        external_payload=external_payload,
        pose_decoder=pose_encoding_to_extri_intri,
    )
    # The cache also contains dense DPT/pointmap tensors that V7 never trains.
    # Release those CPU payloads after constructing the compact V7 batches.
    del (
        training_payload,
        train_full_payload,
        validation_payload,
        long_payload,
        external_payload,
    )
    gc.collect()
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        _raw_row(split, training=config.training)
        for split in splits
    ]
    for architecture in config.architectures:
        _seed_everything(config.training.seed)
        model = V7PoseFusion(
            architecture=architecture,
            camera_dim=int(training_batch["camera_hidden"].shape[-1]),
            appearance_dim=int(training_batch["appearance"].shape[-1]),
            geometry_dim=int(training_batch["pose_geometry"].shape[-1]),
            local_feature_dim=int(
                training_batch["local_features"].shape[-1]
            ),
            config=config.fusion,
        ).to(config.device)
        print(f"V7 training architecture={architecture}")
        started = time.perf_counter()
        training_result = _train_model(
            model,
            batch=training_batch,
            baseline_w2c=training_baseline,
            target_w2c=training_target,
            reference_index=training_reference_index,
            config=config,
        )
        training_seconds = time.perf_counter() - started
        parameters = sum(
            value.numel() for value in model.parameters()
        )
        train_metrics = _evaluate(
            model,
            batch=training_batch,
            baseline_w2c=training_baseline,
            target_w2c=training_target,
            reference_index=training_reference_index,
            evaluation_indices=None,
            variant="normal",
            translation_weight=config.training.translation_weight,
        )
        raw_train_loss = next(
            split.raw_metrics["loss"]
            for split in splits
            if split.name == "train_capacity"
        )
        overfit_pass = int(
            _loss_drop(raw_train_loss, float(train_metrics["loss"]))
            >= 95.0
            and int(train_metrics["reference_exact"]) == 1
        )
        torch.save(
            {
                "architecture": architecture,
                "model": {
                    name: value.detach().cpu()
                    for name, value in model.state_dict().items()
                },
                "fusion": asdict(config.fusion),
                "training": asdict(config.training),
                "local_token_count": config.local_token_count,
                "parameters": parameters,
                **training_result,
            },
            output_dir / f"{architecture}.pt",
        )

        for split in splits:
            metrics_by_variant = {
                variant: _evaluate(
                    model,
                    batch=split.batch,
                    baseline_w2c=split.baseline_w2c,
                    target_w2c=split.target_w2c,
                    reference_index=split.reference_index,
                    evaluation_indices=split.evaluation_indices,
                    variant=variant,
                    translation_weight=(
                        config.training.translation_weight
                    ),
                )
                for variant in V7_INPUT_VARIANTS
            }
            rows.append(
                _architecture_row(
                    split,
                    architecture=architecture,
                    metrics_by_variant=metrics_by_variant,
                    parameters=parameters,
                    training_seconds=training_seconds,
                    overfit_pass=overfit_pass,
                    training=config.training,
                )
            )

    _label_rows(rows, architectures=config.architectures)
    compact_rows = _compact_rows(
        rows,
        architectures=config.architectures,
    )
    path = output_dir / "v7_ablation.csv"
    _write_csv(path, compact_rows)
    print("V7 ABLATION (single CSV)")
    with path.open("r", encoding="utf8") as handle:
        print(handle.read().rstrip())
    return path


def _build_splits(
    config: V7ExperimentConfig,
    *,
    train_clip: ClipConfig,
    training_payload: dict,
    train_full_payload: dict,
    validation_payload: dict,
    long_payload: dict,
    external_payload: dict,
    pose_decoder,
) -> list[EvaluationSplit]:
    specs: list[tuple[str, dict, list[int] | None]] = [
        ("train_capacity", training_payload, None),
    ]
    heldout_frames = tuple(train_clip.evaluation_frame_indices or ())
    if not heldout_frames:
        raise ValueError("V7 training clip requires evaluation_frame_indices.")
    positions = {
        int(frame): index
        for index, frame in enumerate(train_full_payload["frame_indices"])
    }
    specs.extend(
        [
            (
                "temporal_holdout",
                train_full_payload,
                [positions[int(frame)] for frame in heldout_frames],
            ),
            ("validation", validation_payload, None),
            ("cross_clip", external_payload, None),
            ("long30", long_payload, None),
        ]
    )
    splits = []
    for name, payload, evaluation_indices in specs:
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
        reference = int(payload["reference_sequence_index"])
        raw = _pose_metrics(
            baseline,
            target,
            reference_index=reference,
            translation_weight=config.training.translation_weight,
            evaluation_indices=evaluation_indices,
        )
        evaluated = _evaluation_indices(
            len(payload["frame_indices"]),
            reference_index=reference,
            evaluation_indices=evaluation_indices,
        )
        splits.append(
            EvaluationSplit(
                name=name,
                frames=tuple(
                    int(payload["frame_indices"][index])
                    for index in evaluated
                ),
                batch=batch,
                baseline_w2c=baseline,
                target_w2c=target,
                reference_index=reference,
                evaluation_indices=evaluation_indices,
                raw_metrics=raw,
            )
        )
    return splits


def _train_model(
    model: V7PoseFusion,
    *,
    batch: dict[str, torch.Tensor],
    baseline_w2c: torch.Tensor,
    target_w2c: torch.Tensor,
    reference_index: int,
    config: V7ExperimentConfig,
) -> dict[str, float | int]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    initial = _predict(
        model,
        batch=batch,
        baseline_w2c=baseline_w2c,
        reference_index=reference_index,
        variant="normal",
    )
    initial_loss = float(
        _pose_loss(
            initial["world_to_camera"],
            target_w2c,
            reference_index=reference_index,
            translation_weight=config.training.translation_weight,
        ).cpu()
    )
    if not bool(initial["active_frames"].any()):
        raise RuntimeError(
            f"V7 {model.architecture} has no usable training frame."
        )
    best_loss = float("inf")
    best_step = 0
    best_state = None
    model.train()
    for step in range(1, config.training.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        output = _forward(
            model,
            batch=batch,
            baseline_w2c=baseline_w2c,
            reference_index=reference_index,
            variant="normal",
        )
        loss = _pose_loss(
            output["world_to_camera"],
            target_w2c,
            reference_index=reference_index,
            translation_weight=config.training.translation_weight,
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(
                f"Non-finite V7 {model.architecture} loss at step {step}."
            )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.training.grad_clip_norm,
        )
        if not bool(torch.isfinite(grad_norm)):
            raise RuntimeError(
                f"Non-finite V7 {model.architecture} gradient at step {step}."
            )
        optimizer.step()
        if (
            step % config.training.log_every == 0
            or step == config.training.steps
        ):
            checked = _predict(
                model,
                batch=batch,
                baseline_w2c=baseline_w2c,
                reference_index=reference_index,
                variant="normal",
            )
            current = float(
                _pose_loss(
                    checked["world_to_camera"],
                    target_w2c,
                    reference_index=reference_index,
                    translation_weight=(
                        config.training.translation_weight
                    ),
                ).cpu()
            )
            print(
                f"V7 {model.architecture:34s} "
                f"{step:4d}/{config.training.steps}: loss={current:.6g}"
            )
            if current < best_loss:
                best_loss = current
                best_step = step
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
            model.train()
    if best_state is None:
        raise RuntimeError(f"V7 {model.architecture} made no checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    return {
        "initial_loss": initial_loss,
        "best_logged_loss": best_loss,
        "best_step": best_step,
    }


def _evaluate(
    model: V7PoseFusion,
    *,
    batch: dict[str, torch.Tensor],
    baseline_w2c: torch.Tensor,
    target_w2c: torch.Tensor,
    reference_index: int,
    evaluation_indices: list[int] | None,
    variant: str,
    translation_weight: float,
) -> dict[str, float | int]:
    output = _predict(
        model,
        batch=batch,
        baseline_w2c=baseline_w2c,
        reference_index=reference_index,
        variant=variant,
    )
    metrics = _pose_metrics(
        output["world_to_camera"],
        target_w2c,
        reference_index=reference_index,
        translation_weight=translation_weight,
        evaluation_indices=evaluation_indices,
    )
    metrics["reference_exact"] = int(
        torch.equal(
            output["world_to_camera"][:, reference_index].cpu(),
            baseline_w2c[:, reference_index].cpu(),
        )
    )
    indices = _evaluation_indices(
        output["active_frames"].shape[1],
        reference_index=reference_index,
        evaluation_indices=evaluation_indices,
    )
    index = torch.tensor(
        indices,
        dtype=torch.long,
        device=output["active_frames"].device,
    )
    metrics["active_frames"] = int(
        output["active_frames"].index_select(1, index).sum().cpu()
    )
    return metrics


def _predict(
    model: V7PoseFusion,
    *,
    batch: dict[str, torch.Tensor],
    baseline_w2c: torch.Tensor,
    reference_index: int,
    variant: str,
) -> dict[str, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        return _forward(
            model,
            batch=batch,
            baseline_w2c=baseline_w2c,
            reference_index=reference_index,
            variant=variant,
        )


def _forward(
    model: V7PoseFusion,
    *,
    batch: dict[str, torch.Tensor],
    baseline_w2c: torch.Tensor,
    reference_index: int,
    variant: str,
) -> dict[str, torch.Tensor]:
    inputs = perturb_v7_inputs(batch, variant)
    return model(
        camera_hidden=inputs["camera_hidden"],
        baseline_world_to_camera=baseline_w2c,
        appearance=inputs["appearance"],
        geometry=inputs["pose_geometry"],
        quality=inputs["quality"],
        observed=inputs["observed"],
        identity_valid=inputs["identity_valid"],
        identity_unknown=inputs["identity_unknown"],
        local_features=inputs["local_features"],
        local_valid=inputs["local_valid"],
        reference_index=reference_index,
    )


def _camera_batch(
    payload: dict,
    *,
    device: str,
    local_token_count: int,
) -> dict[str, torch.Tensor]:
    output = {}
    for field in (
        "camera_hidden",
        "baseline_pose_encoding",
        "target_pose_encoding",
        "appearance",
        "pose_geometry",
        "quality",
        "observed",
        "identity_valid",
        "identity_unknown",
    ):
        value = payload.get(field)
        if not torch.is_tensor(value):
            raise ValueError(f"V7 cache is missing tensor field {field!r}.")
        output[field] = value.unsqueeze(0).to(device)
    local_features, local_valid = _local_geometry_features(
        payload,
        max_points=local_token_count,
    )
    output["local_features"] = local_features.unsqueeze(0).to(device)
    output["local_valid"] = local_valid.unsqueeze(0).to(device)
    return output


def _local_geometry_features(
    payload: dict,
    *,
    max_points: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create 32 backend-neutral local geometry tokens from cached samples."""

    world = torch.as_tensor(
        payload["baseline_world_points"]
    ).detach().float().cpu()
    confidence = torch.as_tensor(
        payload["baseline_world_confidence"]
    ).detach().float().cpu()
    if confidence.ndim == 4 and confidence.shape[-1] == 1:
        confidence = confidence[..., 0]
    uvd = torch.as_tensor(payload["instance_uvd"]).detach().float().cpu()
    valid = torch.as_tensor(
        payload["instance_uvd_valid"]
    ).detach().bool().cpu()
    if world.ndim != 4 or world.shape[-1] != 3:
        raise ValueError("V7 baseline_world_points must be [S,H,W,3].")
    if confidence.shape != world.shape[:3]:
        raise ValueError("V7 point confidence/world point shapes differ.")
    if uvd.ndim != 4 or uvd.shape[-1] != 3 or valid.shape != uvd.shape[:-1]:
        raise ValueError("V7 cached UVD samples have invalid shapes.")
    sequence, instances = uvd.shape[:2]
    features = torch.zeros(
        sequence,
        instances,
        int(max_points),
        7,
        dtype=torch.float32,
    )
    output_valid = torch.zeros(
        sequence,
        instances,
        int(max_points),
        dtype=torch.bool,
    )
    height, width = world.shape[1:3]
    origin = torch.as_tensor(
        payload["scene_origin"]
    ).detach().float().cpu()
    scale = max(float(payload["scene_scale"]), 1e-6)
    for frame in range(sequence):
        for slot in range(instances):
            available = torch.nonzero(
                valid[frame, slot],
                as_tuple=False,
            ).flatten()
            if not available.numel():
                continue
            if available.numel() > int(max_points):
                positions = torch.linspace(
                    0,
                    available.numel() - 1,
                    steps=int(max_points),
                ).round().long()
                available = available.index_select(0, positions)
            samples = uvd[frame, slot].index_select(0, available)
            x = samples[:, 0].round().long().clamp(0, width - 1)
            y = samples[:, 1].round().long().clamp(0, height - 1)
            points = world[frame, y, x]
            point_confidence = confidence[frame, y, x]
            finite = (
                torch.isfinite(points).all(dim=-1)
                & torch.isfinite(samples).all(dim=-1)
                & torch.isfinite(point_confidence)
                & samples[:, 2].gt(1e-6)
            )
            if not bool(finite.any()):
                continue
            points = points[finite]
            samples = samples[finite]
            point_confidence = point_confidence[finite]
            count = min(int(points.shape[0]), int(max_points))
            normalized_world = (points[:count] - origin) / scale
            normalized_u = (
                2.0 * samples[:count, 0] / max(width - 1, 1) - 1.0
            )
            normalized_v = (
                2.0 * samples[:count, 1] / max(height - 1, 1) - 1.0
            )
            log_depth = torch.log(
                (samples[:count, 2] / scale).clamp_min(1e-6)
            ).clamp(-8.0, 8.0)
            normalized_confidence = (
                torch.log1p(point_confidence[:count].clamp_min(0.0))
                / 4.0
            ).clamp(0.0, 2.0)
            features[frame, slot, :count] = torch.cat(
                [
                    normalized_world,
                    normalized_u[:, None],
                    normalized_v[:, None],
                    log_depth[:, None],
                    normalized_confidence[:, None],
                ],
                dim=-1,
            )
            output_valid[frame, slot, :count] = True
    return features, output_valid


def _raw_row(
    split: EvaluationSplit,
    *,
    training: V7TrainingConfig,
) -> dict[str, object]:
    return {
        "architecture_level": -1,
        "architecture": "raw_streamvggt",
        "mechanism": "raw_control",
        "key_source": "none",
        "value_source": "none",
        "split": split.name,
        "frames": " ".join(str(value) for value in split.frames),
        "parameters": 0,
        "spatial_tokens": 0,
        "rotation_deg": _short(split.raw_metrics["rotation_degrees"]),
        "translation_native": _short(
            split.raw_metrics["translation_native"]
        ),
        "loss": _short(split.raw_metrics["loss"]),
        "loss_drop_percent": 0,
        "active_frames": 0,
        "reference_exact": 1,
        "model_overfit_pass": 0,
        "instance_off_loss": "",
        "wrong_geometry_loss": "",
        "shuffle_time_loss": "",
        "appearance_only_loss": "",
        "geometry_only_loss": "",
        "wrong_geometry_delta_percent": "",
        "shuffle_time_delta_percent": "",
        "training_seconds": 0,
        "split_best": 0,
        "development_score": "",
        "development_best": 0,
        "report_only_split": int(split.name in {"cross_clip", "long30"}),
    }


def _architecture_row(
    split: EvaluationSplit,
    *,
    architecture: str,
    metrics_by_variant: dict[str, dict[str, float | int]],
    parameters: int,
    training_seconds: float,
    overfit_pass: int,
    training: V7TrainingConfig,
) -> dict[str, object]:
    normal = metrics_by_variant["normal"]
    normal_loss = float(normal["loss"])
    wrong_loss = float(metrics_by_variant["wrong_geometry"]["loss"])
    shuffle_loss = float(metrics_by_variant["shuffle_time"]["loss"])
    description = V7_DESCRIPTIONS[architecture]
    return {
        "architecture_level": description.level,
        "architecture": architecture,
        "mechanism": description.mechanism,
        "key_source": description.key_source,
        "value_source": description.value_source,
        "split": split.name,
        "frames": " ".join(str(value) for value in split.frames),
        "parameters": parameters,
        "spatial_tokens": description.spatial_tokens,
        "rotation_deg": _short(float(normal["rotation_degrees"])),
        "translation_native": _short(
            float(normal["translation_native"])
        ),
        "loss": _short(normal_loss),
        "loss_drop_percent": _short(
            _loss_drop(float(split.raw_metrics["loss"]), normal_loss)
        ),
        "active_frames": int(normal["active_frames"]),
        "reference_exact": int(normal["reference_exact"]),
        "model_overfit_pass": overfit_pass,
        "instance_off_loss": _short(
            float(metrics_by_variant["instance_off"]["loss"])
        ),
        "wrong_geometry_loss": _short(wrong_loss),
        "shuffle_time_loss": _short(shuffle_loss),
        "appearance_only_loss": _short(
            float(metrics_by_variant["appearance_only"]["loss"])
        ),
        "geometry_only_loss": _short(
            float(metrics_by_variant["geometry_only"]["loss"])
        ),
        "wrong_geometry_delta_percent": _short(
            _relative_damage(normal_loss, wrong_loss)
        ),
        "shuffle_time_delta_percent": _short(
            _relative_damage(normal_loss, shuffle_loss)
        ),
        "training_seconds": _short(training_seconds),
        "split_best": 0,
        "development_score": "",
        "development_best": 0,
        "report_only_split": int(split.name in {"cross_clip", "long30"}),
    }


def _label_rows(
    rows: list[dict[str, object]],
    *,
    architectures: tuple[str, ...],
) -> None:
    for split in {
        str(row["split"])
        for row in rows
    }:
        candidates = [
            row
            for row in rows
            if row["split"] == split
        ]
        best = min(float(row["loss"]) for row in candidates)
        for row in candidates:
            row["split_best"] = int(
                abs(float(row["loss"]) - best) <= 1e-10
            )
    development_splits = ("temporal_holdout", "validation")
    raw = {
        str(row["split"]): float(row["loss"])
        for row in rows
        if row["architecture"] == "raw_streamvggt"
    }
    scores = {"raw_streamvggt": 1.0}
    for architecture in architectures:
        ratios = [
            float(
                next(
                    row["loss"]
                    for row in rows
                    if row["architecture"] == architecture
                    and row["split"] == split
                )
            )
            / raw[split]
            for split in development_splits
        ]
        scores[architecture] = sum(ratios) / len(ratios)
    best_architecture = min(scores, key=scores.get)
    for row in rows:
        architecture = str(row["architecture"])
        if architecture in scores:
            row["development_score"] = _short(scores[architecture])
            row["development_best"] = int(
                architecture == best_architecture
            )


def _compact_rows(
    rows: list[dict[str, object]],
    *,
    architectures: tuple[str, ...],
) -> list[dict[str, object]]:
    """Pivot split rows into six upload-friendly architecture rows."""

    split_order = (
        "train_capacity",
        "temporal_holdout",
        "validation",
        "cross_clip",
        "long30",
    )
    architecture_order = ("raw_streamvggt", *architectures)
    output = []
    for architecture in architecture_order:
        current = {
            str(row["split"]): row
            for row in rows
            if row["architecture"] == architecture
        }
        if set(current) != set(split_order):
            raise ValueError(
                f"V7 compact output for {architecture} has splits "
                f"{sorted(current)}, expected {list(split_order)}."
            )
        first = current["train_capacity"]
        compact: dict[str, object] = {
            "architecture_level": first["architecture_level"],
            "architecture": architecture,
            "mechanism": first["mechanism"],
            "key_source": first["key_source"],
            "value_source": first["value_source"],
            "parameters": first["parameters"],
            "spatial_tokens": first["spatial_tokens"],
            "training_seconds": first["training_seconds"],
            "model_overfit_pass": first["model_overfit_pass"],
            "development_score": first["development_score"],
            "development_best": first["development_best"],
        }
        for split in split_order:
            row = current[split]
            prefix = {
                "train_capacity": "train",
                "temporal_holdout": "temporal",
                "validation": "validation",
                "cross_clip": "cross",
                "long30": "long30",
            }[split]
            compact.update(
                {
                    f"{prefix}_rotation_deg": row["rotation_deg"],
                    f"{prefix}_translation": row["translation_native"],
                    f"{prefix}_loss": row["loss"],
                    f"{prefix}_drop_percent": row["loss_drop_percent"],
                    f"{prefix}_active_frames": row["active_frames"],
                    f"{prefix}_best": row["split_best"],
                }
            )
            if split != "train_capacity":
                compact.update(
                    {
                        f"{prefix}_instance_off_loss": row[
                            "instance_off_loss"
                        ],
                        f"{prefix}_wrong_geometry_loss": row[
                            "wrong_geometry_loss"
                        ],
                        f"{prefix}_shuffle_time_loss": row[
                            "shuffle_time_loss"
                        ],
                        f"{prefix}_appearance_only_loss": row[
                            "appearance_only_loss"
                        ],
                        f"{prefix}_geometry_only_loss": row[
                            "geometry_only_loss"
                        ],
                        f"{prefix}_wrong_geometry_delta_percent": row[
                            "wrong_geometry_delta_percent"
                        ],
                        f"{prefix}_shuffle_time_delta_percent": row[
                            "shuffle_time_delta_percent"
                        ],
                    }
                )
        compact["reference_exact_all_splits"] = int(
            all(int(current[split]["reference_exact"]) for split in split_order)
        )
        output.append(compact)
    return output


def _pose_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    reference_index: int,
    translation_weight: float,
    evaluation_indices: list[int] | None = None,
) -> torch.Tensor:
    indices = _evaluation_indices(
        predicted.shape[1],
        reference_index=reference_index,
        evaluation_indices=evaluation_indices,
    )
    index = torch.tensor(
        indices,
        dtype=torch.long,
        device=predicted.device,
    )
    predicted = predicted.index_select(1, index)
    target = target.index_select(1, index)
    rotation = F.mse_loss(
        predicted[..., :3, :3],
        target[..., :3, :3],
    )
    translation = F.smooth_l1_loss(
        _camera_centers(predicted),
        _camera_centers(target),
        beta=0.01,
    )
    return rotation + float(translation_weight) * translation


def _pose_metrics(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    reference_index: int,
    translation_weight: float,
    evaluation_indices: list[int] | None,
) -> dict[str, float]:
    indices = _evaluation_indices(
        predicted.shape[1],
        reference_index=reference_index,
        evaluation_indices=evaluation_indices,
    )
    index = torch.tensor(
        indices,
        dtype=torch.long,
        device=predicted.device,
    )
    predicted_eval = predicted.index_select(1, index)
    target_eval = target.index_select(1, index)
    relative = (
        predicted_eval[..., :3, :3]
        @ target_eval[..., :3, :3].transpose(-1, -2)
    )
    cosine = (
        torch.diagonal(relative, dim1=-2, dim2=-1).sum(dim=-1) - 1.0
    ) * 0.5
    rotation = torch.rad2deg(
        torch.acos(cosine.clamp(-1.0, 1.0))
    ).mean()
    translation = torch.linalg.vector_norm(
        _camera_centers(predicted_eval) - _camera_centers(target_eval),
        dim=-1,
    ).mean()
    loss = _pose_loss(
        predicted,
        target,
        reference_index=reference_index,
        translation_weight=translation_weight,
        evaluation_indices=evaluation_indices,
    )
    return {
        "rotation_degrees": float(rotation.cpu()),
        "translation_native": float(translation.cpu()),
        "loss": float(loss.cpu()),
    }


def _evaluation_indices(
    sequence_length: int,
    *,
    reference_index: int,
    evaluation_indices: list[int] | None,
) -> list[int]:
    indices = (
        [
            index
            for index in range(sequence_length)
            if index != int(reference_index)
        ]
        if evaluation_indices is None
        else [int(value) for value in evaluation_indices]
    )
    if not indices or int(reference_index) in indices:
        raise ValueError("V7 evaluation indices are empty or include reference.")
    if len(indices) != len(set(indices)):
        raise ValueError("V7 evaluation indices contain duplicates.")
    if any(index < 0 or index >= sequence_length for index in indices):
        raise ValueError("V7 evaluation index is out of range.")
    return indices


def _camera_centers(world_to_camera: torch.Tensor) -> torch.Tensor:
    rotation = world_to_camera[..., :3, :3]
    translation = world_to_camera[..., :3, 3]
    return -(
        rotation.transpose(-1, -2) @ translation[..., None]
    ).squeeze(-1)


def load_v7_config(path: str | Path) -> V7ExperimentConfig:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf8") as handle:
        raw = yaml.safe_load(handle) or {}
    model = raw.get("model", {})
    training = raw.get("training", {})
    config = V7ExperimentConfig(
        source_path=source,
        data_config=_path(raw.get("data_config"), source.parent),
        output_dir=_path(raw.get("output_dir"), source.parent),
        training_clip_name=str(raw.get("training_clip_name", "")),
        validation_clip_name=str(raw.get("validation_clip_name", "")),
        long_clip_name=str(raw.get("long_clip_name", "")),
        external_clip_name=str(raw.get("external_clip_name", "")),
        device=str(raw.get("device", "cuda:0")),
        architectures=tuple(
            str(value)
            for value in raw.get("architectures", V7_ARCHITECTURES)
        ),
        local_token_count=int(model.get("local_token_count", 32)),
        fusion=V6FusionConfig(
            hidden_dim=int(model.get("hidden_dim", 256)),
            num_heads=int(model.get("num_heads", 8)),
            dropout=float(model.get("dropout", 0.0)),
            memory_momentum=float(model.get("memory_momentum", 0.90)),
            min_track_confidence=float(
                model.get("min_track_confidence", 0.25)
            ),
            unknown_reliability=float(
                model.get("unknown_reliability", 0.50)
            ),
            softened_mismatch_reliability=float(
                model.get("softened_mismatch_reliability", 0.25)
            ),
            identity_gate_policy=str(
                model.get(
                    "identity_gate_policy",
                    "soft_unknown_strict_memory",
                )
            ),
            max_rotation_degrees=float(
                model.get("max_rotation_degrees", 15.0)
            ),
            max_translation_native=float(
                model.get("max_translation_native", 0.50)
            ),
        ),
        training=V7TrainingConfig(
            steps=int(training.get("steps", 1200)),
            learning_rate=float(training.get("learning_rate", 1e-3)),
            weight_decay=float(training.get("weight_decay", 0.0)),
            grad_clip_norm=float(training.get("grad_clip_norm", 5.0)),
            translation_weight=float(
                training.get("translation_weight", 10.0)
            ),
            seed=int(training.get("seed", 0)),
            log_every=int(training.get("log_every", 200)),
        ),
    )
    _validate_config(config)
    return config


def _validate_config(config: V7ExperimentConfig) -> None:
    for name, value in (
        ("training_clip_name", config.training_clip_name),
        ("validation_clip_name", config.validation_clip_name),
        ("long_clip_name", config.long_clip_name),
        ("external_clip_name", config.external_clip_name),
    ):
        if not value:
            raise ValueError(f"V7 {name} is required.")
    if not config.architectures:
        raise ValueError("V7 requires at least one architecture.")
    unknown = set(config.architectures) - set(V7_ARCHITECTURES)
    if unknown:
        raise ValueError(
            "Unknown V7 architecture(s): " + ", ".join(sorted(unknown))
        )
    if len(set(config.architectures)) != len(config.architectures):
        raise ValueError("V7 architecture list contains duplicates.")
    if config.local_token_count < 2:
        raise ValueError("V7 local_token_count must be at least two.")
    if config.fusion.hidden_dim % config.fusion.num_heads:
        raise ValueError("V7 hidden_dim must be divisible by num_heads.")
    if config.training.steps < 1 or config.training.log_every < 1:
        raise ValueError("V7 training steps/log_every must be positive.")


def _find_clip(config: LearnedPoseConfig, name: str) -> ClipConfig:
    selected = [clip for clip in config.clips if clip.name == name]
    if len(selected) != 1:
        raise ValueError(
            f"V7 clip {name!r} was not found exactly once in "
            f"{config.source_path}."
        )
    return selected[0]


def _load_cache(config: LearnedPoseConfig, clip: ClipConfig) -> dict:
    path = cache_path(config, clip)
    if not path.is_file():
        raise FileNotFoundError(
            f"V7 requires its frozen cache {path}. Run "
            "streaming_couping/commands_v7_fusion_ablation.txt so the V7 "
            "data stage builds any missing SAM3.1/StreamVGGT observations."
        )
    print(f"V7 reusing cache: {path}")
    return load_feature_cache(path)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _loss_drop(initial: float, final: float) -> float:
    return 0.0 if initial <= 1e-12 else 100.0 * (initial - final) / initial


def _relative_damage(normal: float, perturbed: float) -> float:
    if normal <= 1e-12:
        return 0.0
    return 100.0 * (perturbed - normal) / normal


def _short(value: float) -> str:
    return f"{float(value):.8g}"


def _path(value: object, base: Path) -> Path:
    if value is None or not str(value).strip():
        raise ValueError("V7 configuration contains an empty path.")
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    cwd = (Path.cwd() / path).resolve()
    if cwd.exists() or not (base / path).exists():
        return cwd
    return (base / path).resolve()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="streaming_couping/configs/v7_fusion_ablation.yaml",
    )
    parser.add_argument("--device", help="Override V7 training device.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
