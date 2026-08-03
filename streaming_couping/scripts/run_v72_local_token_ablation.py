#!/usr/bin/env python3
"""Run a causal SAM3.1 mask-local token count/modality ablation."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch
import yaml

from streaming_couping.src.config import load_config
from streaming_couping.src.external_repos import maybe_add_repo_to_path
from streaming_couping.src.learned_pose.config import load_learned_pose_config
from streaming_couping.src.learned_pose.v7_fusion import (
    V7PoseFusion,
    perturb_v7_inputs,
)
from streaming_couping.src.learned_pose.v71_causal_fusion import (
    V71FrozenResidualFusion,
)
from streaming_couping.src.learned_pose.v72_local_fusion import (
    V72_ARCHITECTURES,
    V72_DESCRIPTIONS,
    V72_INPUT_VARIANTS,
    V72FrozenLocalResidual,
    perturb_v72_inputs,
)
from streaming_couping.scripts.run_v7_fusion_ablation import (
    V7ExperimentConfig,
    _camera_batch,
    _evaluation_indices,
    _find_clip,
    _load_cache,
    _pose_loss,
    _pose_metrics,
    _seed_everything,
    load_v7_config,
)
from streaming_couping.scripts.run_v71_instance_causality import (
    EvaluationSpec,
    _checkpoint_signature as _v71_checkpoint_signature,
    _load_checkpoint as _load_v71_checkpoint,
    _make_spec,
    _slice_batch_prefix,
    _validate_temporal_partition,
    load_v71_config,
)


CONTROL_ARCHITECTURES = (
    "camera_extra_all",
    "camera_extra_common_gate",
)


class V72CheckpointSignatureMismatch(ValueError):
    """A valid V7.2 checkpoint belongs to another experiment provenance."""


@dataclass(frozen=True)
class V72Config:
    source_path: Path
    v71_config: Path
    data_config: Path
    output_dir: Path
    device: str
    token_counts: tuple[int, ...]
    architectures: tuple[str, ...]
    base_steps: int
    residual_steps: int
    seed: int


def main() -> None:
    args = _parse_args()
    config = load_v72_config(args.config)
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
        selected = tuple(
            value.strip()
            for value in args.architectures.split(",")
            if value.strip()
        )
        unknown = set(selected) - set(V72_ARCHITECTURES)
        if not selected or unknown:
            raise ValueError(
                f"Invalid --architectures selection; unknown={sorted(unknown)}"
            )
        config = replace(config, architectures=selected)
    if args.token_counts:
        selected_counts = tuple(
            sorted(
                {
                    int(value.strip())
                    for value in args.token_counts.split(",")
                    if value.strip()
                }
            )
        )
        if not selected_counts or any(value < 2 for value in selected_counts):
            raise ValueError("--token-counts requires comma-separated values >= 2.")
        config = replace(config, token_counts=selected_counts)
    result = run_v72(config, resume=bool(args.resume))
    print(f"V7.2 one-table result={result}")


def run_v72(config: V72Config, *, resume: bool) -> Path:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    v71 = load_v71_config(config.v71_config)
    source_v7 = load_v7_config(v71.v7_config)
    v71_base_v7 = replace(
        source_v7,
        device=config.device,
        training=replace(source_v7.training, seed=v71.seed),
    )
    base_v7 = replace(
        source_v7,
        device=config.device,
        training=replace(source_v7.training, seed=config.seed),
    )
    if config.base_steps != v71.base_steps:
        raise ValueError(
            "V7.2 must reuse the exact V7.1 frozen L0: "
            f"V7.2 base_steps={config.base_steps}, "
            f"V7.1 base_steps={v71.base_steps}."
        )
    data = load_learned_pose_config(config.data_config)
    long_clip = _find_clip(data, v71.long_clip_name)
    validation_clip = _find_clip(data, v71.validation_clip_name)
    external_clip = _find_clip(data, v71.external_clip_name)
    long_payload = _load_cache(data, long_clip)
    validation_payload = _load_cache(data, validation_clip)
    external_payload = _load_cache(data, external_clip)
    max_tokens = max(config.token_counts)
    for name, payload in (
        ("long", long_payload),
        ("validation", validation_payload),
        ("cross", external_payload),
    ):
        _validate_local_payload(payload, name=name, minimum_tokens=max_tokens)

    recovery = load_config(data.recovery_config)
    maybe_add_repo_to_path(recovery.streamvggt_repo)
    from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri

    long_batch, long_baseline, long_target = _prepare_payload(
        long_payload,
        config=base_v7,
        pose_decoder=pose_encoding_to_extri_intri,
        device=config.device,
    )
    validation_batch, validation_baseline, validation_target = _prepare_payload(
        validation_payload,
        config=base_v7,
        pose_decoder=pose_encoding_to_extri_intri,
        device=config.device,
    )
    external_batch, external_baseline, external_target = _prepare_payload(
        external_payload,
        config=base_v7,
        pose_decoder=pose_encoding_to_extri_intri,
        device=config.device,
    )

    positions = {
        int(frame): index
        for index, frame in enumerate(long_payload["frame_indices"])
    }
    groups = {
        "base_train": v71.base_train_frames,
        "residual_train": v71.residual_train_frames,
        "development": v71.development_frames,
        "future": v71.future_frames,
    }
    _validate_temporal_partition(
        groups,
        frame_indices=tuple(int(v) for v in long_payload["frame_indices"]),
        reference_index=int(long_payload["reference_sequence_index"]),
    )
    temporal_indices = {
        name: [positions[int(frame)] for frame in frames]
        for name, frames in groups.items()
    }
    reference_index = int(long_payload["reference_sequence_index"])
    specs = [
        _make_spec(
            name=name,
            frames=frames,
            batch=long_batch,
            baseline=long_baseline,
            target=long_target,
            reference_index=reference_index,
            evaluation_indices=temporal_indices[name],
            translation_weight=base_v7.training.translation_weight,
        )
        for name, frames in groups.items()
    ]
    specs.extend(
        [
            _make_spec(
                name="validation",
                frames=_nonreference_frames(validation_payload),
                batch=validation_batch,
                baseline=validation_baseline,
                target=validation_target,
                reference_index=int(validation_payload["reference_sequence_index"]),
                evaluation_indices=None,
                translation_weight=base_v7.training.translation_weight,
            ),
            _make_spec(
                name="cross",
                frames=_nonreference_frames(external_payload),
                batch=external_batch,
                baseline=external_baseline,
                target=external_target,
                reference_index=int(external_payload["reference_sequence_index"]),
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
    residual_prefix = positions[max(v71.residual_train_frames)] + 1
    v71_base_path = v71.output_dir / "frozen_l0.pt"
    if not v71_base_path.is_file():
        raise FileNotFoundError(
            "V7.2 requires the exact V7.1 frozen L0 checkpoint: "
            f"{v71_base_path}. Run "
            "`zsh streaming_couping/commands_v71_instance_causality.txt` "
            "before V7.2."
        )
    v71_base_signature = _v71_checkpoint_signature(
        v71,
        v71_base_v7,
        architecture="frozen_l0",
    )
    v71_base_checkpoint = _load_v71_checkpoint(
        v71_base_path,
        expected_signature=v71_base_signature,
    )
    base_model.load_state_dict(v71_base_checkpoint["model"], strict=True)
    base_seconds = float(v71_base_checkpoint.get("training_seconds", 0.0))
    v71_base_sha256 = _sha256_file(v71_base_path)
    base_signature = _signature(
        config,
        base_v7,
        "frozen_l0",
        token_count=0,
        frozen_l0_signature=v71_base_signature,
    )
    base_path = output_dir / "frozen_l0.pt"
    _save_checkpoint(
        base_path,
        base_signature,
        base_model,
        training_seconds=base_seconds,
        metadata={
            "source": str(v71_base_path),
            "source_signature": v71_base_signature,
            "source_sha256": v71_base_sha256,
        },
    )
    print(
        "V7.2 loaded exact V7.1 frozen L0: "
        f"{v71_base_path} sha256={v71_base_sha256}"
    )
    for parameter in base_model.parameters():
        parameter.requires_grad_(False)
    base_model.eval()

    raw_metrics = {spec.name: spec.raw_metrics for spec in specs}
    base_metrics = {
        spec.name: _evaluate_model(base_model, spec, base_v7, variant="normal")
        for spec in specs
    }
    _assert_v71_baseline_metrics(v71.output_dir, base_metrics)
    rows = [
        _result_row(
            architecture="raw_streamvggt",
            token_count=0,
            mechanism="raw_control",
            identity_source="none",
            value_source="none",
            metrics={name: {"normal": value} for name, value in raw_metrics.items()},
            specs=specs,
            base_metrics=base_metrics,
            parameters=0,
            trainable_parameters=0,
            training_seconds=0.0,
            instance_content=False,
        ),
        _result_row(
            architecture="frozen_l0",
            token_count=0,
            mechanism="early_frame_camera_baseline",
            identity_source="none",
            value_source="camera_hidden",
            metrics={name: {"normal": value} for name, value in base_metrics.items()},
            specs=specs,
            base_metrics=base_metrics,
            parameters=sum(p.numel() for p in base_model.parameters()),
            trainable_parameters=0,
            training_seconds=base_seconds,
            instance_content=False,
        ),
    ]
    frame_rows: list[dict[str, Any]] = []
    for spec in specs:
        _append_frame_rows(
            frame_rows,
            architecture="frozen_l0",
            token_count=0,
            model=base_model,
            spec=spec,
            experiment=base_v7,
            base_metrics=base_metrics,
        )

    # Train camera controls once. They share the exact same Stage-A L0 and
    # Stage-B frames as every local model.
    controls: dict[str, dict[str, dict[str, float | int]]] = {}
    residual_training_batch = _slice_batch_prefix(
        long_batch, length=residual_prefix
    )
    for architecture in CONTROL_ARCHITECTURES:
        _seed_everything(config.seed)
        model = V71FrozenResidualFusion(
            base_model=copy.deepcopy(base_model),
            architecture=architecture,
            appearance_dim=int(long_batch["appearance"].shape[-1]),
            geometry_dim=int(long_batch["pose_geometry"].shape[-1]),
            local_feature_dim=int(long_batch["local_features"].shape[-1]),
            config=base_v7.fusion,
        ).to(config.device)
        training_seconds = _train_or_resume(
            model,
            architecture=architecture,
            token_count=0,
            frozen_l0_signature=v71_base_signature,
            config=config,
            experiment=base_v7,
            output_dir=output_dir,
            resume=resume,
            batch=residual_training_batch,
            baseline=long_baseline[:, :residual_prefix],
            target=long_target[:, :residual_prefix],
            reference_index=reference_index,
            training_indices=temporal_indices["residual_train"],
        )
        metrics = {
            spec.name: {
                "normal": _evaluate_model(model, spec, base_v7, variant="normal"),
                "instance_off": _evaluate_model(
                    model, spec, base_v7, variant="instance_off"
                ),
            }
            for spec in specs
        }
        controls[architecture] = metrics
        if (
            config.seed == v71.seed
            and config.residual_steps == v71.residual_steps
        ):
            _assert_v71_metrics(
                v71.output_dir,
                architecture,
                {
                    name: values["normal"]
                    for name, values in metrics.items()
                },
            )
        rows.append(
            _result_row(
                architecture=architecture,
                token_count=0,
                mechanism="camera_capacity_control",
                identity_source="none",
                value_source="camera_hidden",
                metrics=metrics,
                specs=specs,
                base_metrics=base_metrics,
                parameters=sum(p.numel() for p in model.parameters()),
                trainable_parameters=sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
                training_seconds=training_seconds,
                instance_content=False,
            )
        )

    for token_count in config.token_counts:
        limited_specs = [
            replace(spec, batch=_limit_local_tokens(spec.batch, token_count))
            for spec in specs
        ]
        limited_training = _limit_local_tokens(
            residual_training_batch, token_count
        )
        for architecture in config.architectures:
            _seed_everything(config.seed)
            model = V72FrozenLocalResidual(
                base_model=copy.deepcopy(base_model),
                architecture=architecture,
                sam_local_dim=int(long_batch["sam_local_features"].shape[-1]),
                geometry_dim=int(long_batch["pose_geometry"].shape[-1]),
                geometry_local_dim=int(long_batch["local_features"].shape[-1]),
                config=base_v7.fusion,
            ).to(config.device)
            label = f"{architecture}_k{token_count:02d}"
            training_seconds = _train_or_resume(
                model,
                architecture=architecture,
                token_count=token_count,
                frozen_l0_signature=v71_base_signature,
                config=config,
                experiment=base_v7,
                output_dir=output_dir,
                resume=resume,
                batch=limited_training,
                baseline=long_baseline[:, :residual_prefix],
                target=long_target[:, :residual_prefix],
                reference_index=reference_index,
                training_indices=temporal_indices["residual_train"],
            )
            metrics = {
                spec.name: {
                    variant: _evaluate_model(
                        model, spec, base_v7, variant=variant
                    )
                    for variant in V72_INPUT_VARIANTS
                }
                for spec in limited_specs
            }
            description = V72_DESCRIPTIONS[architecture]
            rows.append(
                _result_row(
                    architecture=label,
                    token_count=token_count,
                    mechanism=description.mechanism,
                    identity_source=description.identity_source,
                    value_source=description.value_source,
                    metrics=metrics,
                    specs=limited_specs,
                    base_metrics=base_metrics,
                    parameters=sum(p.numel() for p in model.parameters()),
                    trainable_parameters=sum(
                        p.numel() for p in model.parameters() if p.requires_grad
                    ),
                    training_seconds=training_seconds,
                    instance_content=True,
                )
            )
            for spec in limited_specs:
                _append_frame_rows(
                    frame_rows,
                    architecture=label,
                    token_count=token_count,
                    model=model,
                    spec=spec,
                    experiment=base_v7,
                    base_metrics=base_metrics,
                )

    _label_rows(rows, controls=controls)
    peak_gpu_memory_gib = (
        torch.cuda.max_memory_allocated() / (1024**3)
        if torch.cuda.is_available()
        else 0.0
    )
    for row in rows:
        row["peak_gpu_memory_gib"] = _short(peak_gpu_memory_gib)
    result_path = output_dir / "v72_local_token_ablation.csv"
    frame_path = output_dir / "v72_frame_diagnostics.csv"
    _write_csv(result_path, rows)
    _write_csv(frame_path, frame_rows)
    (output_dir / "v72_run_metadata.json").write_text(
        json.dumps(
            {
                "schema": 3,
                "config": asdict(config),
                "v71": asdict(v71),
                "v7_fusion": asdict(base_v7.fusion),
                "frozen_l0_source": str(v71_base_path),
                "frozen_l0_source_signature": v71_base_signature,
                "frozen_l0_source_sha256": v71_base_sha256,
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device": (
                    torch.cuda.get_device_name()
                    if torch.cuda.is_available()
                    else ""
                ),
                "peak_gpu_memory_gib": peak_gpu_memory_gib,
                "result_csv": str(result_path),
                "frame_diagnostics_csv": str(frame_path),
            },
            indent=2,
            default=str,
            sort_keys=True,
        )
        + "\n",
        encoding="utf8",
    )
    print("V7.2 LOCAL TOKEN ABLATION (single CSV)")
    print(result_path.read_text(encoding="utf8").rstrip())
    return result_path


def _prepare_payload(
    payload: dict,
    *,
    config: V7ExperimentConfig,
    pose_decoder,
    device: str,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    batch = _camera_batch(
        payload,
        device=device,
        local_token_count=max(2, config.local_token_count),
    )
    for field in ("sam_local_features", "sam_local_uv", "sam_local_valid"):
        batch[field] = torch.as_tensor(payload[field]).unsqueeze(0).to(device)
    batch["sam_local_features"] = batch["sam_local_features"].float()
    batch["sam_local_uv"] = batch["sam_local_uv"].float()
    batch["sam_local_valid"] = batch["sam_local_valid"].bool()
    image_size = tuple(int(value) for value in payload["image_size"])
    baseline, _ = pose_decoder(
        batch["baseline_pose_encoding"], image_size_hw=image_size
    )
    target, _ = pose_decoder(
        batch["target_pose_encoding"], image_size_hw=image_size
    )
    return batch, baseline, target


def _validate_local_payload(payload: dict, *, name: str, minimum_tokens: int) -> None:
    if str(payload.get("sam_version", "")) != "sam3.1":
        raise ValueError(f"V7.2 {name} cache is not SAM3.1.")
    for field in ("sam_local_features", "sam_local_uv", "sam_local_valid"):
        if not torch.is_tensor(payload.get(field)):
            raise ValueError(
                f"V7.2 {name} cache lacks {field}. Run the V7.2 cache stage."
            )
    features = payload["sam_local_features"]
    uv = payload["sam_local_uv"]
    valid = payload["sam_local_valid"]
    if features.ndim != 4 or int(features.shape[2]) < int(minimum_tokens):
        raise ValueError(
            f"V7.2 {name} local feature shape {tuple(features.shape)} cannot "
            f"support K={minimum_tokens}."
        )
    if uv.shape != (*features.shape[:3], 2) or valid.shape != features.shape[:3]:
        raise ValueError(f"V7.2 {name} local cache shapes disagree.")
    if not bool(valid.any()):
        raise ValueError(f"V7.2 {name} has no valid SAM local token.")


def _limit_local_tokens(
    batch: dict[str, torch.Tensor], token_count: int
) -> dict[str, torch.Tensor]:
    output = {name: value for name, value in batch.items()}
    for name in ("sam_local_features", "sam_local_uv"):
        output[name] = batch[name][..., : int(token_count), :]
    output["sam_local_valid"] = batch["sam_local_valid"][
        ..., : int(token_count)
    ]
    output["local_features"] = batch["local_features"][..., : int(token_count), :]
    output["local_valid"] = batch["local_valid"][..., : int(token_count)]
    return output


def _train_or_resume(
    model,
    *,
    architecture: str,
    token_count: int,
    frozen_l0_signature: str,
    config: V72Config,
    experiment: V7ExperimentConfig,
    output_dir: Path,
    resume: bool,
    batch: dict[str, torch.Tensor],
    baseline: torch.Tensor,
    target: torch.Tensor,
    reference_index: int,
    training_indices: list[int],
) -> float:
    label = architecture if not token_count else f"{architecture}_k{token_count:02d}"
    signature = _signature(
        config,
        experiment,
        architecture,
        token_count=token_count,
        frozen_l0_signature=frozen_l0_signature,
    )
    path = output_dir / f"{label}.pt"
    if resume and path.is_file():
        try:
            checkpoint = _load_checkpoint(path, signature)
        except V72CheckpointSignatureMismatch:
            print(
                "V7.2 invalidating stale Stage-B checkpoint tied to a "
                f"different frozen L0: {path}"
            )
        else:
            model.load_state_dict(checkpoint["model"])
            print(f"V7.2 resumed architecture={label}")
            return float(checkpoint.get("training_seconds", 0.0))
    print(f"V7.2 stage B architecture={label}")
    started = time.perf_counter()
    training_result = _train(
        model,
        batch=batch,
        baseline=baseline,
        target=target,
        reference_index=reference_index,
        training_indices=training_indices,
        steps=config.residual_steps,
        experiment=experiment,
    )
    elapsed = time.perf_counter() - started
    _save_checkpoint(
        path,
        signature,
        model,
        training_seconds=elapsed,
        metadata={
            "frozen_l0_signature": frozen_l0_signature,
            "training_result": training_result,
        },
    )
    return elapsed


def _train(
    model,
    *,
    batch: dict[str, torch.Tensor],
    baseline: torch.Tensor,
    target: torch.Tensor,
    reference_index: int,
    training_indices: list[int],
    steps: int,
    experiment: V7ExperimentConfig,
) -> dict[str, float | int]:
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("V7.2 model contains no trainable parameter.")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(experiment.training.learning_rate),
        weight_decay=float(experiment.training.weight_decay),
    )
    model.eval()
    with torch.no_grad():
        initial_output = _forward_model(
            model,
            batch=batch,
            baseline=baseline,
            reference_index=reference_index,
            variant="normal",
        )
        index = torch.tensor(training_indices, device=baseline.device)
        if not bool(
            initial_output["active_frames"].index_select(1, index).any()
        ):
            raise RuntimeError("V7.2 has no active residual-training frame.")
        initial_loss = float(
            _pose_loss(
                initial_output["world_to_camera"],
                target,
                reference_index=reference_index,
                translation_weight=experiment.training.translation_weight,
                evaluation_indices=training_indices,
            ).cpu()
        )
    best_loss = float("inf")
    best_step = 0
    best_state = None
    model.train()
    for step in range(1, int(steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        output = _forward_model(
            model,
            batch=batch,
            baseline=baseline,
            reference_index=reference_index,
            variant="normal",
        )
        loss = _pose_loss(
            output["world_to_camera"],
            target,
            reference_index=reference_index,
            translation_weight=experiment.training.translation_weight,
            evaluation_indices=training_indices,
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"Non-finite V7.2 loss at step {step}.")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters, float(experiment.training.grad_clip_norm)
        )
        if not bool(torch.isfinite(grad_norm)):
            raise RuntimeError(f"Non-finite V7.2 gradient at step {step}.")
        optimizer.step()
        if (
            step % int(experiment.training.log_every) == 0
            or step == int(steps)
        ):
            model.eval()
            with torch.no_grad():
                checked = _forward_model(
                    model,
                    batch=batch,
                    baseline=baseline,
                    reference_index=reference_index,
                    variant="normal",
                )
                current = float(
                    _pose_loss(
                        checked["world_to_camera"],
                        target,
                        reference_index=reference_index,
                        translation_weight=(
                            experiment.training.translation_weight
                        ),
                        evaluation_indices=training_indices,
                    ).cpu()
                )
            print(f"  step={step}/{steps} loss={current:.8f}")
            if current < best_loss:
                best_loss = current
                best_step = step
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
            model.train()
    if best_state is None:
        raise RuntimeError("V7.2 training produced no checkpoint candidate.")
    model.load_state_dict(best_state)
    model.eval()
    return {
        "initial_loss": initial_loss,
        "best_logged_loss": best_loss,
        "best_step": best_step,
    }


def _evaluate_model(
    model,
    spec: EvaluationSpec,
    experiment: V7ExperimentConfig,
    *,
    variant: str,
) -> dict[str, float | int]:
    model.eval()
    with torch.no_grad():
        output = _forward_model(
            model,
            batch=spec.batch,
            baseline=spec.baseline_w2c,
            reference_index=spec.reference_index,
            variant=variant,
        )
    metrics = _pose_metrics(
        output["world_to_camera"],
        spec.target_w2c,
        reference_index=spec.reference_index,
        translation_weight=experiment.training.translation_weight,
        evaluation_indices=spec.evaluation_indices,
    )
    indices = _evaluation_indices(
        output["world_to_camera"].shape[1],
        reference_index=spec.reference_index,
        evaluation_indices=spec.evaluation_indices,
    )
    index = torch.tensor(indices, device=spec.baseline_w2c.device)
    metrics["active_frames"] = int(
        output["active_frames"].index_select(1, index).sum().cpu()
    )
    metrics["reference_exact"] = int(
        torch.equal(
            output["world_to_camera"][:, spec.reference_index],
            spec.baseline_w2c[:, spec.reference_index],
        )
    )
    return metrics


def _forward_model(
    model,
    *,
    batch: dict[str, torch.Tensor],
    baseline: torch.Tensor,
    reference_index: int,
    variant: str,
) -> dict[str, torch.Tensor]:
    if isinstance(model, V72FrozenLocalResidual):
        inputs = perturb_v72_inputs(batch, variant)
        return model(
            camera_hidden=inputs["camera_hidden"],
            baseline_world_to_camera=baseline,
            appearance=inputs["appearance"],
            geometry=inputs["pose_geometry"],
            quality=inputs["quality"],
            observed=inputs["observed"],
            identity_valid=inputs["identity_valid"],
            identity_unknown=inputs["identity_unknown"],
            local_features=inputs["local_features"],
            local_valid=inputs["local_valid"],
            sam_local_features=inputs["sam_local_features"],
            sam_local_uv=inputs["sam_local_uv"],
            sam_local_valid=inputs["sam_local_valid"],
            reference_index=reference_index,
        )
    mapped_variant = variant if variant in {"normal", "instance_off", "wrong_geometry"} else "normal"
    inputs = perturb_v7_inputs(batch, mapped_variant)
    return model(
        camera_hidden=inputs["camera_hidden"],
        baseline_world_to_camera=baseline,
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


def _result_row(
    *,
    architecture: str,
    token_count: int,
    mechanism: str,
    identity_source: str,
    value_source: str,
    metrics: dict[str, dict[str, dict[str, float | int]]],
    specs: list[EvaluationSpec],
    base_metrics: dict[str, dict[str, float | int]],
    parameters: int,
    trainable_parameters: int,
    training_seconds: float,
    instance_content: bool,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "architecture": architecture,
        "token_count": token_count,
        "mechanism": mechanism,
        "identity_source": identity_source,
        "value_source": value_source,
        "instance_content": int(instance_content),
        "parameters": parameters,
        "trainable_parameters": trainable_parameters,
        "training_seconds": _short(training_seconds),
        "peak_gpu_memory_gib": "",
        "development_score": "",
        "development_best": 0,
        "beats_camera_controls_report_only": 0,
        "instance_off_exact": 0,
        "causal_local_pass": 0,
    }
    for spec in specs:
        variants = metrics[spec.name]
        normal = variants["normal"]
        prefix = spec.name
        loss = float(normal["loss"])
        row[f"{prefix}_frames"] = " ".join(str(v) for v in spec.frames)
        row[f"{prefix}_rotation_deg"] = _short(float(normal["rotation_degrees"]))
        row[f"{prefix}_translation"] = _short(float(normal["translation_native"]))
        row[f"{prefix}_loss"] = _short(loss)
        row[f"{prefix}_gain_vs_l0_percent"] = _short(
            _gain(float(base_metrics[prefix]["loss"]), loss)
        )
        row[f"{prefix}_active_frames"] = int(normal.get("active_frames", 0))
        for variant in V72_INPUT_VARIANTS[1:]:
            row[f"{prefix}_{variant}_loss"] = (
                _short(float(variants[variant]["loss"]))
                if variant in variants
                else ""
            )
    return row


def _label_rows(
    rows: list[dict[str, Any]],
    *,
    controls: dict[str, dict[str, dict[str, float | int]]],
) -> None:
    base = next(row for row in rows if row["architecture"] == "frozen_l0")
    for row in rows:
        row["development_score"] = _short(
            0.5
            * (
                float(row["development_loss"]) / float(base["development_loss"])
                + float(row["validation_loss"]) / float(base["validation_loss"])
            )
        )
    best = min(float(row["development_score"]) for row in rows)
    for row in rows:
        row["development_best"] = int(
            abs(float(row["development_score"]) - best) <= 1e-10
        )
        if not int(row["instance_content"]):
            continue
        beats_controls = all(
            float(row[f"{split}_loss"])
            < float(controls[control][split]["normal"]["loss"])
            for control in CONTROL_ARCHITECTURES
            for split in ("future", "cross")
        )
        instance_off_exact = all(
            abs(
                float(row[f"{split}_instance_off_loss"])
                - float(base[f"{split}_loss"])
            )
            <= 1e-6
            for split in ("development", "validation", "future", "cross")
        )
        beats_l0 = all(
            float(row[f"{split}_loss"]) < float(base[f"{split}_loss"])
            for split in ("development", "validation", "future", "cross")
        )
        row["beats_camera_controls_report_only"] = int(beats_controls)
        row["instance_off_exact"] = int(instance_off_exact)
        row["causal_local_pass"] = int(
            beats_controls and instance_off_exact and beats_l0
        )


def _append_frame_rows(
    rows: list[dict[str, Any]],
    *,
    architecture: str,
    token_count: int,
    model,
    spec: EvaluationSpec,
    experiment: V7ExperimentConfig,
    base_metrics: dict[str, dict[str, float | int]],
) -> None:
    model.eval()
    with torch.no_grad():
        output = _forward_model(
            model,
            batch=spec.batch,
            baseline=spec.baseline_w2c,
            reference_index=spec.reference_index,
            variant="normal",
        )
    indices = _evaluation_indices(
        output["world_to_camera"].shape[1],
        reference_index=spec.reference_index,
        evaluation_indices=spec.evaluation_indices,
    )
    for frame, index in zip(spec.frames, indices):
        current = _pose_metrics(
            output["world_to_camera"],
            spec.target_w2c,
            reference_index=spec.reference_index,
            translation_weight=experiment.training.translation_weight,
            evaluation_indices=[index],
        )
        base_frame = _pose_metrics(
            _forward_model(
                model.base_model if hasattr(model, "base_model") else model,
                batch=spec.batch,
                baseline=spec.baseline_w2c,
                reference_index=spec.reference_index,
                variant="normal",
            )["world_to_camera"],
            spec.target_w2c,
            reference_index=spec.reference_index,
            translation_weight=experiment.training.translation_weight,
            evaluation_indices=[index],
        )
        local_counts = spec.batch["sam_local_valid"][0, index].sum(dim=-1)
        attention_available = output.get("sam_local_available")
        attention_entropy = output.get("sam_attention_entropy")
        attention_max = output.get("sam_attention_max")
        if (
            attention_available is not None
            and attention_entropy is not None
            and attention_max is not None
        ):
            selected_attention = attention_available[0, index].bool()
            mean_attention_entropy = (
                float(attention_entropy[0, index][selected_attention].mean().cpu())
                if bool(selected_attention.any())
                else 0.0
            )
            mean_attention_max = (
                float(attention_max[0, index][selected_attention].mean().cpu())
                if bool(selected_attention.any())
                else 0.0
            )
        else:
            mean_attention_entropy = 0.0
            mean_attention_max = 0.0
        rows.append(
            {
                "architecture": architecture,
                "token_count": token_count,
                "split": spec.name,
                "frame": int(frame),
                "sam_local_instances": int(local_counts.gt(0).sum().cpu()),
                "sam_local_tokens": int(local_counts.sum().cpu()),
                "sam_attention_entropy_normalized": _short(
                    mean_attention_entropy
                ),
                "sam_attention_max_probability": _short(
                    mean_attention_max
                ),
                "active": int(bool(output["active_frames"][0, index].cpu())),
                "frozen_l0_rotation_deg": _short(base_frame["rotation_degrees"]),
                "frozen_l0_translation": _short(base_frame["translation_native"]),
                "frozen_l0_loss": _short(base_frame["loss"]),
                "rotation_deg": _short(current["rotation_degrees"]),
                "translation": _short(current["translation_native"]),
                "loss": _short(current["loss"]),
                "gain_vs_l0_percent": _short(
                    _gain(float(base_frame["loss"]), float(current["loss"]))
                ),
            }
        )


def _nonreference_frames(payload: dict) -> tuple[int, ...]:
    reference = int(payload["reference_sequence_index"])
    return tuple(
        int(frame)
        for index, frame in enumerate(payload["frame_indices"])
        if index != reference
    )


def _signature(
    config: V72Config,
    experiment: V7ExperimentConfig,
    architecture: str,
    *,
    token_count: int,
    frozen_l0_signature: str,
) -> str:
    value = {
        "schema": 3,
        "architecture": architecture,
        "token_count": token_count,
        "frozen_l0_signature": frozen_l0_signature,
        "seed": config.seed,
        "base_steps": config.base_steps,
        "residual_steps": config.residual_steps,
        "fusion": asdict(experiment.fusion),
        "training": asdict(experiment.training),
        "data_config": str(config.data_config),
    }
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode("utf8")
    ).hexdigest()


def _save_checkpoint(
    path: Path,
    signature: str,
    model,
    *,
    training_seconds: float,
    metadata: dict[str, Any] | None = None,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "signature": signature,
        "model": {
            name: value.detach().cpu()
            for name, value in model.state_dict().items()
        },
        "training_seconds": training_seconds,
    }
    if metadata is not None:
        payload["metadata"] = metadata
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_checkpoint(path: Path, signature: str) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"Invalid V7.2 checkpoint: {path}")
    if payload.get("signature") != signature:
        raise V72CheckpointSignatureMismatch(
            f"V7.2 checkpoint provenance mismatch: {path}"
        )
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_v71_baseline_metrics(
    v71_output_dir: Path,
    base_metrics: dict[str, dict[str, float | int]],
) -> None:
    _assert_v71_metrics(v71_output_dir, "frozen_l0", base_metrics)


def _assert_v71_metrics(
    v71_output_dir: Path,
    architecture: str,
    metrics: dict[str, dict[str, float | int]],
) -> None:
    csv_path = v71_output_dir / "v71_instance_causality.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(
            "V7.2 requires the V7.1 result CSV to verify shared-model "
            f"equivalence: {csv_path}"
        )
    with csv_path.open("r", encoding="utf8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    matches = [
        row for row in rows if row.get("architecture") == architecture
    ]
    if len(matches) != 1:
        raise ValueError(
            "V7.1 result must contain exactly one row for "
            f"{architecture}: {csv_path}"
        )
    expected = matches[0]
    metric_fields = {
        "rotation_deg": "rotation_degrees",
        "translation": "translation_native",
        "loss": "loss",
    }
    mismatches = []
    for split, current in metrics.items():
        for csv_suffix, metric_name in metric_fields.items():
            expected_value = float(expected[f"{split}_{csv_suffix}"])
            current_value = float(current[metric_name])
            tolerance = 5e-7 * max(1.0, abs(expected_value))
            if abs(current_value - expected_value) > tolerance:
                mismatches.append(
                    f"{split}.{csv_suffix}: "
                    f"V7.1={expected_value:.10g} V7.2={current_value:.10g}"
                )
    if mismatches:
        raise RuntimeError(
            f"V7.2 {architecture} does not reproduce V7.1 metrics:\n  "
            + "\n  ".join(mismatches)
        )
    print(f"V7.2 {architecture} metrics exactly reproduce V7.1: {csv_path}")


def load_v72_config(path: str | Path) -> V72Config:
    source = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf8")) or {}
    training = raw.get("training", {})
    config = V72Config(
        source_path=source,
        v71_config=_path(raw.get("v71_config")),
        data_config=_path(raw.get("data_config")),
        output_dir=_path(raw.get("output_dir")),
        device=str(raw.get("device", "cuda:0")),
        token_counts=tuple(int(v) for v in raw.get("token_counts", (8, 16, 32))),
        architectures=tuple(str(v) for v in raw.get("architectures", V72_ARCHITECTURES)),
        base_steps=int(training.get("base_steps", 1200)),
        residual_steps=int(training.get("residual_steps", 1200)),
        seed=int(training.get("seed", 0)),
    )
    if not config.token_counts or any(v < 2 for v in config.token_counts):
        raise ValueError("V7.2 token_counts must contain values >= 2.")
    if tuple(sorted(set(config.token_counts))) != config.token_counts:
        raise ValueError("V7.2 token_counts must be unique and increasing.")
    unknown = set(config.architectures) - set(V72_ARCHITECTURES)
    if unknown:
        raise ValueError(f"Unknown V7.2 architectures: {sorted(unknown)}")
    if len(config.architectures) != len(set(config.architectures)):
        raise ValueError("V7.2 architectures contain duplicates.")
    if config.base_steps < 1 or config.residual_steps < 1:
        raise ValueError("V7.2 training steps must be positive.")
    return config


def _path(value: Any) -> Path:
    if value is None or not str(value).strip():
        raise ValueError("V7.2 configuration contains an empty path.")
    return Path(str(value)).expanduser().resolve()


def _gain(baseline: float, current: float) -> float:
    return 0.0 if baseline <= 1e-12 else 100.0 * (baseline - current) / baseline


def _short(value: float) -> str:
    return f"{float(value):.8g}"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="streaming_couping/configs/v72_local_token_ablation.yaml",
    )
    parser.add_argument("--device")
    parser.add_argument("--output-dir")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--base-steps", type=int)
    parser.add_argument("--residual-steps", type=int)
    parser.add_argument(
        "--architectures",
        help="Comma-separated V7.2 subset; camera controls remain automatic.",
    )
    parser.add_argument(
        "--token-counts",
        help="Comma-separated local token counts, for example 8,16,32.",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
