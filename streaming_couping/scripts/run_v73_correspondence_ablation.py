#!/usr/bin/env python3
"""Run V7.3 SAM-weighted StreamVGGT correspondence ablations."""

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
    V72FrozenLocalResidual,
    perturb_v72_inputs,
)
from streaming_couping.src.learned_pose.v73_correspondence_fusion import (
    V73_ARCHITECTURES,
    V73_DESCRIPTIONS,
    V73_INPUT_VARIANTS,
    V73FrozenCorrespondenceResidual,
    perturb_v73_inputs,
)
from streaming_couping.scripts.run_v7_fusion_ablation import (
    _evaluation_indices,
    _find_clip,
    _load_cache,
    _pose_loss,
    _pose_metrics,
    _seed_everything,
    load_v7_config,
)
from streaming_couping.scripts.run_v71_instance_causality import (
    _checkpoint_signature as _v71_checkpoint_signature,
    _load_checkpoint as _load_v71_checkpoint,
    _make_spec,
    _slice_batch_prefix,
    _validate_temporal_partition,
    load_v71_config,
)
from streaming_couping.scripts.run_v72_local_token_ablation import (
    CONTROL_ARCHITECTURES,
    _assert_v71_baseline_metrics,
    _limit_local_tokens,
    _load_checkpoint as _load_v72_checkpoint,
    _nonreference_frames,
    _prepare_payload,
    _sha256_file,
    _signature as _v72_signature,
    _validate_local_payload,
    load_v72_config,
)


RETAINED_V72_ARCHITECTURE = "geometry_local_match"
CAUSAL_SPLITS = ("development", "validation", "future", "cross")
SAM_PERTURBATIONS = (
    "sam_off",
    "uniform_sam",
    "wrong_sam_identity",
    "shuffle_sam_time",
)


class V73CheckpointSignatureMismatch(ValueError):
    """A checkpoint was created by a different V7.3 experiment."""


@dataclass(frozen=True)
class V73Config:
    source_path: Path
    v72_config: Path
    output_dir: Path
    device: str
    token_counts: tuple[int, ...]
    architectures: tuple[str, ...]
    residual_steps: int
    seed: int


def main() -> None:
    args = _parse_args()
    config = load_v73_config(args.config)
    if args.device:
        config = replace(config, device=args.device)
    if args.output_dir:
        config = replace(
            config, output_dir=Path(args.output_dir).expanduser().resolve()
        )
    if args.seed is not None:
        config = replace(config, seed=int(args.seed))
    if args.residual_steps is not None:
        config = replace(config, residual_steps=int(args.residual_steps))
    if args.architectures:
        selected = tuple(v.strip() for v in args.architectures.split(",") if v.strip())
        unknown = set(selected) - set(V73_ARCHITECTURES)
        if not selected or unknown:
            raise ValueError(f"Unknown V7.3 architectures: {sorted(unknown)}")
        config = replace(config, architectures=selected)
    if args.token_counts:
        selected_counts = tuple(
            sorted({int(v.strip()) for v in args.token_counts.split(",") if v.strip()})
        )
        if not selected_counts or any(v < 2 for v in selected_counts):
            raise ValueError("--token-counts requires comma-separated values >= 2.")
        config = replace(config, token_counts=selected_counts)
    result = run_v73(config, resume=bool(args.resume))
    print(f"V7.3 one-table result={result}")


def run_v73(config: V73Config, *, resume: bool) -> Path:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    v72 = load_v72_config(config.v72_config)
    v71 = load_v71_config(v72.v71_config)
    source_v7 = load_v7_config(v71.v7_config)
    experiment = replace(
        source_v7,
        device=config.device,
        training=replace(source_v7.training, seed=config.seed),
    )
    v71_experiment = replace(
        source_v7,
        device=config.device,
        training=replace(source_v7.training, seed=v71.seed),
    )
    v72_experiment = replace(
        source_v7,
        device=config.device,
        training=replace(source_v7.training, seed=v72.seed),
    )
    _validate_upstream_protocol(config, v72, v71)

    data = load_learned_pose_config(v72.data_config)
    long_clip = _find_clip(data, v71.long_clip_name)
    validation_clip = _find_clip(data, v71.validation_clip_name)
    external_clip = _find_clip(data, v71.external_clip_name)
    long_payload = _load_cache(data, long_clip)
    validation_payload = _load_cache(data, validation_clip)
    external_payload = _load_cache(data, external_clip)
    maximum_tokens = max(config.token_counts)
    for name, payload in (
        ("long", long_payload),
        ("validation", validation_payload),
        ("cross", external_payload),
    ):
        _validate_local_payload(payload, name=name, minimum_tokens=maximum_tokens)

    recovery = load_config(data.recovery_config)
    maybe_add_repo_to_path(recovery.streamvggt_repo)
    from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri

    long_batch, long_baseline, long_target = _prepare_payload(
        long_payload,
        config=experiment,
        pose_decoder=pose_encoding_to_extri_intri,
        device=config.device,
    )
    validation_batch, validation_baseline, validation_target = _prepare_payload(
        validation_payload,
        config=experiment,
        pose_decoder=pose_encoding_to_extri_intri,
        device=config.device,
    )
    external_batch, external_baseline, external_target = _prepare_payload(
        external_payload,
        config=experiment,
        pose_decoder=pose_encoding_to_extri_intri,
        device=config.device,
    )

    positions = {int(frame): index for index, frame in enumerate(long_payload["frame_indices"])}
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
            translation_weight=experiment.training.translation_weight,
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
                translation_weight=experiment.training.translation_weight,
            ),
            _make_spec(
                name="cross",
                frames=_nonreference_frames(external_payload),
                batch=external_batch,
                baseline=external_baseline,
                target=external_target,
                reference_index=int(external_payload["reference_sequence_index"]),
                evaluation_indices=None,
                translation_weight=experiment.training.translation_weight,
            ),
        ]
    )

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    base_model, base_signature, base_source, base_sha256, base_seconds = _load_exact_l0(
        config=config,
        v71=v71,
        experiment=v71_experiment,
        batch=long_batch,
        output_dir=output_dir,
    )
    raw_metrics = {spec.name: spec.raw_metrics for spec in specs}
    base_metrics = {
        spec.name: _evaluate_model(base_model, spec, experiment, variant="normal")
        for spec in specs
    }
    _assert_v71_baseline_metrics(v71.output_dir, base_metrics)

    rows = [
        _result_row(
            architecture="raw_streamvggt",
            architecture_family="control",
            token_count=0,
            mechanism="raw_control",
            correspondence_source="none",
            metrics={name: {"normal": value} for name, value in raw_metrics.items()},
            specs=specs,
            base_metrics=base_metrics,
            parameters=0,
            trainable_parameters=0,
            training_seconds=0.0,
            uses_sam=False,
        ),
        _result_row(
            architecture="frozen_l0",
            architecture_family="control",
            token_count=0,
            mechanism="early_frame_camera_baseline",
            correspondence_source="none",
            metrics={name: {"normal": value} for name, value in base_metrics.items()},
            specs=specs,
            base_metrics=base_metrics,
            parameters=sum(p.numel() for p in base_model.parameters()),
            trainable_parameters=0,
            training_seconds=base_seconds,
            uses_sam=False,
        ),
    ]
    frame_rows: list[dict[str, Any]] = []
    controls: dict[str, dict[str, dict[str, float | int]]] = {}

    # Load, rather than retrain, V7.2 controls. Their checkpoint signatures
    # bind them to the same exact V7.1 L0 used above.
    for architecture in CONTROL_ARCHITECTURES:
        model = V71FrozenResidualFusion(
            base_model=copy.deepcopy(base_model),
            architecture=architecture,
            appearance_dim=int(long_batch["appearance"].shape[-1]),
            geometry_dim=int(long_batch["pose_geometry"].shape[-1]),
            local_feature_dim=int(long_batch["local_features"].shape[-1]),
            config=experiment.fusion,
        ).to(config.device)
        retained_trainable = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        retained_checkpoint = _load_retained_v72(
            model,
            path=v72.output_dir / f"{architecture}.pt",
            signature=_v72_signature(
                v72,
                v72_experiment,
                architecture,
                token_count=0,
                frozen_l0_signature=base_signature,
            ),
        )
        metrics = {
            spec.name: {
                variant: _evaluate_model(model, spec, experiment, variant=variant)
                for variant in ("normal", "instance_off")
            }
            for spec in specs
        }
        controls[architecture] = metrics
        rows.append(
            _result_row(
                architecture=architecture,
                architecture_family="retained_v72_camera_control",
                token_count=0,
                mechanism="camera_capacity_control",
                correspondence_source="none",
                metrics=metrics,
                specs=specs,
                base_metrics=base_metrics,
                parameters=sum(p.numel() for p in model.parameters()),
                trainable_parameters=retained_trainable,
                training_seconds=float(
                    retained_checkpoint.get("training_seconds", 0.0)
                ),
                uses_sam=False,
            )
        )

    for token_count in config.token_counts:
        model = V72FrozenLocalResidual(
            base_model=copy.deepcopy(base_model),
            architecture=RETAINED_V72_ARCHITECTURE,
            sam_local_dim=int(long_batch["sam_local_features"].shape[-1]),
            geometry_dim=int(long_batch["pose_geometry"].shape[-1]),
            geometry_local_dim=int(long_batch["local_features"].shape[-1]),
            config=experiment.fusion,
        ).to(config.device)
        label = f"retained_v72_geometry_local_match_k{token_count:02d}"
        retained_trainable = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        retained_checkpoint = _load_retained_v72(
            model,
            path=v72.output_dir / f"{RETAINED_V72_ARCHITECTURE}_k{token_count:02d}.pt",
            signature=_v72_signature(
                v72,
                v72_experiment,
                RETAINED_V72_ARCHITECTURE,
                token_count=token_count,
                frozen_l0_signature=base_signature,
            ),
        )
        limited_specs = [
            replace(spec, batch=_limit_local_tokens(spec.batch, token_count))
            for spec in specs
        ]
        metrics = {
            spec.name: {
                variant: _evaluate_model(model, spec, experiment, variant=variant)
                for variant in ("normal", "instance_off")
            }
            for spec in limited_specs
        }
        rows.append(
            _result_row(
                architecture=label,
                architecture_family="retained_v72_geometry_control",
                token_count=token_count,
                mechanism="v72_geometry_local_attention",
                correspondence_source="streamvggt_local_geometry",
                metrics=metrics,
                specs=limited_specs,
                base_metrics=base_metrics,
                parameters=sum(p.numel() for p in model.parameters()),
                trainable_parameters=retained_trainable,
                training_seconds=float(
                    retained_checkpoint.get("training_seconds", 0.0)
                ),
                uses_sam=False,
            )
        )

    residual_prefix = positions[max(v71.residual_train_frames)] + 1
    training_batch = _slice_batch_prefix(long_batch, length=residual_prefix)
    for token_count in config.token_counts:
        limited_training = _limit_local_tokens(training_batch, token_count)
        limited_specs = [
            replace(spec, batch=_limit_local_tokens(spec.batch, token_count))
            for spec in specs
        ]
        for architecture in config.architectures:
            _seed_everything(config.seed)
            model = V73FrozenCorrespondenceResidual(
                base_model=copy.deepcopy(base_model),
                architecture=architecture,
                sam_local_dim=int(long_batch["sam_local_features"].shape[-1]),
                geometry_local_dim=int(long_batch["local_features"].shape[-1]),
                config=experiment.fusion,
            ).to(config.device)
            label = f"{architecture}_k{token_count:02d}"
            training_seconds = _train_or_resume(
                model,
                architecture=architecture,
                token_count=token_count,
                frozen_l0_signature=base_signature,
                config=config,
                experiment=experiment,
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
                    variant: _evaluate_model(model, spec, experiment, variant=variant)
                    for variant in V73_INPUT_VARIANTS
                }
                for spec in limited_specs
            }
            description = V73_DESCRIPTIONS[architecture]
            rows.append(
                _result_row(
                    architecture=label,
                    architecture_family="v73_correspondence",
                    token_count=token_count,
                    mechanism=description.mechanism,
                    correspondence_source=description.correspondence_source,
                    metrics=metrics,
                    specs=limited_specs,
                    base_metrics=base_metrics,
                    parameters=sum(p.numel() for p in model.parameters()),
                    trainable_parameters=sum(
                        p.numel() for p in model.parameters() if p.requires_grad
                    ),
                    training_seconds=training_seconds,
                    uses_sam=description.uses_sam,
                )
            )
            for spec in limited_specs:
                _append_frame_rows(
                    frame_rows,
                    architecture=label,
                    token_count=token_count,
                    model=model,
                    spec=spec,
                    experiment=experiment,
                )

    _label_rows(rows, controls=controls)
    peak_gpu_memory_gib = (
        torch.cuda.max_memory_allocated() / (1024**3)
        if torch.cuda.is_available()
        else 0.0
    )
    for row in rows:
        row["peak_gpu_memory_gib"] = _short(peak_gpu_memory_gib)
    result_path = output_dir / "v73_correspondence_ablation.csv"
    frame_path = output_dir / "v73_frame_diagnostics.csv"
    _write_csv(result_path, rows)
    _write_csv(frame_path, frame_rows)
    (output_dir / "v73_run_metadata.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "config": asdict(config),
                "v72_config": asdict(v72),
                "frozen_l0_source": str(base_source),
                "frozen_l0_signature": base_signature,
                "frozen_l0_sha256": base_sha256,
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device": torch.cuda.get_device_name() if torch.cuda.is_available() else "",
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
    print("V7.3 CORRESPONDENCE ABLATION (COPY THIS ONE CSV)")
    print(result_path.read_text(encoding="utf8").rstrip())
    return result_path


def _load_exact_l0(*, config, v71, experiment, batch, output_dir):
    _seed_everything(config.seed)
    model = V7PoseFusion(
        architecture="l0_camera_only",
        camera_dim=int(batch["camera_hidden"].shape[-1]),
        appearance_dim=int(batch["appearance"].shape[-1]),
        geometry_dim=int(batch["pose_geometry"].shape[-1]),
        local_feature_dim=int(batch["local_features"].shape[-1]),
        config=experiment.fusion,
    ).to(config.device)
    source = v71.output_dir / "frozen_l0.pt"
    if not source.is_file():
        raise FileNotFoundError(
            f"V7.3 requires {source}; run commands_v71_instance_causality.txt first."
        )
    signature = _v71_checkpoint_signature(
        v71, experiment, architecture="frozen_l0"
    )
    checkpoint = _load_v71_checkpoint(source, expected_signature=signature)
    model.load_state_dict(checkpoint["model"], strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    sha256 = _sha256_file(source)
    copied = output_dir / "frozen_l0.pt"
    torch.save(
        {
            "signature": signature,
            "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "training_seconds": float(checkpoint.get("training_seconds", 0.0)),
            "metadata": {"source": str(source), "source_sha256": sha256},
        },
        copied,
    )
    print(f"V7.3 loaded exact V7.1 L0: {source} sha256={sha256}")
    return model, signature, source, sha256, float(checkpoint.get("training_seconds", 0.0))


def _load_retained_v72(model, *, path: Path, signature: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"V7.3 retained control is missing: {path}. Complete the fixed V7.2 formal run first."
        )
    checkpoint = _load_v72_checkpoint(path, signature)
    model.load_state_dict(checkpoint["model"], strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    print(f"V7.3 retained V7.2 checkpoint={path}")
    return checkpoint


def _train_or_resume(
    model,
    *,
    architecture: str,
    token_count: int,
    frozen_l0_signature: str,
    config: V73Config,
    experiment,
    output_dir: Path,
    resume: bool,
    batch,
    baseline,
    target,
    reference_index: int,
    training_indices: list[int],
) -> float:
    label = f"{architecture}_k{token_count:02d}"
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
        except V73CheckpointSignatureMismatch:
            print(f"V7.3 invalidating stale checkpoint={path}")
        else:
            model.load_state_dict(checkpoint["model"], strict=True)
            model.eval()
            print(f"V7.3 resumed architecture={label}")
            return float(checkpoint.get("training_seconds", 0.0))
    print(f"V7.3 training architecture={label}")
    started = time.perf_counter()
    result = _train(
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
        metadata={"frozen_l0_signature": frozen_l0_signature, "training_result": result},
    )
    return elapsed


def _train(
    model,
    *,
    batch,
    baseline,
    target,
    reference_index: int,
    training_indices: list[int],
    steps: int,
    experiment,
) -> dict[str, float | int]:
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters:
        raise ValueError("V7.3 model contains no trainable parameter.")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(experiment.training.learning_rate),
        weight_decay=float(experiment.training.weight_decay),
    )
    model.eval()
    with torch.no_grad():
        initial = _forward_model(
            model,
            batch=batch,
            baseline=baseline,
            reference_index=reference_index,
            variant="normal",
        )
        index = torch.tensor(training_indices, device=baseline.device)
        if not bool(initial["active_frames"].index_select(1, index).any()):
            raise RuntimeError("V7.3 has no active residual-training frame.")
        initial_loss = float(
            _pose_loss(
                initial["world_to_camera"],
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
            raise RuntimeError(f"Non-finite V7.3 loss at step {step}.")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters, float(experiment.training.grad_clip_norm)
        )
        if not bool(torch.isfinite(grad_norm)):
            raise RuntimeError(f"Non-finite V7.3 gradient at step {step}.")
        optimizer.step()
        if step % int(experiment.training.log_every) == 0 or step == int(steps):
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
                        translation_weight=experiment.training.translation_weight,
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
        raise RuntimeError("V7.3 training produced no logged checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    return {
        "initial_loss": initial_loss,
        "best_logged_loss": best_loss,
        "best_step": best_step,
    }


def _evaluate_model(model, spec, experiment, *, variant: str):
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
    metrics["active_frames"] = int(output["active_frames"].index_select(1, index).sum().cpu())
    metrics["reference_exact"] = int(
        torch.equal(
            output["world_to_camera"][:, spec.reference_index],
            spec.baseline_w2c[:, spec.reference_index],
        )
    )
    return metrics


def _forward_model(model, *, batch, baseline, reference_index: int, variant: str):
    if isinstance(model, V73FrozenCorrespondenceResidual):
        inputs, uniform_sam = perturb_v73_inputs(batch, variant)
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
            uniform_sam=uniform_sam,
        )
    if isinstance(model, V72FrozenLocalResidual):
        inputs = perturb_v72_inputs(
            batch, variant if variant in {"normal", "instance_off"} else "normal"
        )
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
    mapped = variant if variant in {"normal", "instance_off", "wrong_geometry"} else "normal"
    inputs = perturb_v7_inputs(batch, mapped)
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
    *, architecture, architecture_family, token_count, mechanism,
    correspondence_source, metrics, specs, base_metrics, parameters,
    trainable_parameters, training_seconds, uses_sam,
):
    row: dict[str, Any] = {
        "architecture": architecture,
        "architecture_family": architecture_family,
        "token_count": token_count,
        "mechanism": mechanism,
        "correspondence_source": correspondence_source,
        "geometry_value_source": "streamvggt_local_geometry" if token_count else "none",
        "uses_sam": int(uses_sam),
        "parameters": parameters,
        "trainable_parameters": trainable_parameters,
        "training_seconds": _short(training_seconds),
        "peak_gpu_memory_gib": "",
        "development_score": "",
        "development_best": 0,
        "v73_development_best": 0,
        "same_k_geometry_control": "",
        "beats_same_k_geometry_all_splits": 0,
        "retained_v72_geometry_control": "",
        "beats_retained_v72_geometry_all_splits": 0,
        "beats_camera_controls_report_only": 0,
        "instance_off_exact": 0,
        "sam_perturbation_hurt_count": 0,
        "sam_perturbation_total": 0,
        "causal_sam_pass": 0,
    }
    for spec in specs:
        variants = metrics[spec.name]
        normal = variants["normal"]
        prefix = spec.name
        loss = float(normal["loss"])
        row[f"{prefix}_frames"] = " ".join(str(v) for v in spec.frames)
        row[f"{prefix}_rotation_deg"] = _short(normal["rotation_degrees"])
        row[f"{prefix}_translation"] = _short(normal["translation_native"])
        row[f"{prefix}_loss"] = _short(loss)
        row[f"{prefix}_gain_vs_l0_percent"] = _short(
            _gain(float(base_metrics[prefix]["loss"]), loss)
        )
        row[f"{prefix}_active_frames"] = int(normal.get("active_frames", 0))
        for variant in V73_INPUT_VARIANTS[1:]:
            row[f"{prefix}_{variant}_loss"] = (
                _short(variants[variant]["loss"]) if variant in variants else ""
            )
    return row


def _label_rows(rows, *, controls):
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
    v73_best = min(
        float(row["development_score"])
        for row in rows
        if row["architecture_family"] == "v73_correspondence"
    )
    by_name = {row["architecture"]: row for row in rows}
    for row in rows:
        row["development_best"] = int(
            abs(float(row["development_score"]) - best) <= 1e-10
        )
        row["v73_development_best"] = int(
            row["architecture_family"] == "v73_correspondence"
            and abs(float(row["development_score"]) - v73_best) <= 1e-10
        )
        if row["architecture_family"] != "v73_correspondence":
            continue
        instance_off_exact = all(
            abs(
                float(row[f"{split}_instance_off_loss"])
                - float(base[f"{split}_loss"])
            )
            <= 1e-6
            for split in CAUSAL_SPLITS
        )
        row["instance_off_exact"] = int(instance_off_exact)
        retained_name = (
            "retained_v72_geometry_local_match_"
            f"k{int(row['token_count']):02d}"
        )
        retained_geometry = by_name.get(retained_name)
        if retained_geometry is not None:
            row["retained_v72_geometry_control"] = retained_name
            row["beats_retained_v72_geometry_all_splits"] = int(
                all(
                    float(row[f"{split}_loss"])
                    < float(retained_geometry[f"{split}_loss"])
                    for split in CAUSAL_SPLITS
                )
            )
        beats_controls = all(
            float(row[f"{split}_loss"])
            < float(controls[control][split]["normal"]["loss"])
            for control in CONTROL_ARCHITECTURES
            for split in ("future", "cross")
        )
        row["beats_camera_controls_report_only"] = int(beats_controls)
        if not int(row["uses_sam"]):
            continue
        geometry_name = f"geometry_transport_k{int(row['token_count']):02d}"
        geometry_control = by_name.get(geometry_name)
        if geometry_control is None:
            continue
        row["same_k_geometry_control"] = geometry_name
        beats_geometry = all(
            float(row[f"{split}_loss"]) < float(geometry_control[f"{split}_loss"])
            for split in CAUSAL_SPLITS
        )
        row["beats_same_k_geometry_all_splits"] = int(beats_geometry)
        hurt = sum(
            float(row[f"{split}_{variant}_loss"])
            > float(row[f"{split}_loss"]) + 1e-8
            for split in CAUSAL_SPLITS
            for variant in SAM_PERTURBATIONS
        )
        total = len(CAUSAL_SPLITS) * len(SAM_PERTURBATIONS)
        row["sam_perturbation_hurt_count"] = hurt
        row["sam_perturbation_total"] = total
        row["causal_sam_pass"] = int(
            instance_off_exact and beats_controls and beats_geometry and hurt == total
        )


def _append_frame_rows(rows, *, architecture, token_count, model, spec, experiment):
    model.eval()
    with torch.no_grad():
        output = _forward_model(
            model,
            batch=spec.batch,
            baseline=spec.baseline_w2c,
            reference_index=spec.reference_index,
            variant="normal",
        )
        base_output = _forward_model(
            model.base_model,
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
        base = _pose_metrics(
            base_output["world_to_camera"],
            spec.target_w2c,
            reference_index=spec.reference_index,
            translation_weight=experiment.training.translation_weight,
            evaluation_indices=[index],
        )
        available = output["usable_instance_mask"][0, index].bool()
        sam_used = output["sam_used"][0, index].bool()

        def selected_mean(name, mask):
            return (
                float(output[name][0, index][mask].mean().cpu())
                if bool(mask.any())
                else 0.0
            )

        rows.append(
            {
                "architecture": architecture,
                "token_count": token_count,
                "split": spec.name,
                "frame": int(frame),
                "usable_instances": int(available.sum().cpu()),
                "sam_used_instances": int((sam_used & available).sum().cpu()),
                "transport_entropy_normalized": _short(
                    selected_mean("transport_entropy", available)
                ),
                "transport_max_probability": _short(
                    selected_mean("transport_max", available)
                ),
                "sam_affinity_delta": _short(
                    selected_mean("sam_affinity_delta", sam_used & available)
                ),
                "active": int(bool(output["active_frames"][0, index].cpu())),
                "frozen_l0_rotation_deg": _short(base["rotation_degrees"]),
                "frozen_l0_translation": _short(base["translation_native"]),
                "frozen_l0_loss": _short(base["loss"]),
                "rotation_deg": _short(current["rotation_degrees"]),
                "translation": _short(current["translation_native"]),
                "loss": _short(current["loss"]),
                "gain_vs_l0_percent": _short(
                    _gain(float(base["loss"]), float(current["loss"]))
                ),
            }
        )


def _signature(
    config,
    experiment,
    architecture,
    *,
    token_count,
    frozen_l0_signature,
):
    value = {
        "schema": 1,
        "architecture": architecture,
        "token_count": token_count,
        "frozen_l0_signature": frozen_l0_signature,
        "seed": config.seed,
        "residual_steps": config.residual_steps,
        "fusion": asdict(experiment.fusion),
        "training": asdict(experiment.training),
        "v72_config": str(config.v72_config),
    }
    serialized = json.dumps(value, sort_keys=True, default=str).encode("utf8")
    return hashlib.sha256(serialized).hexdigest()


def _save_checkpoint(path, signature, model, *, training_seconds, metadata=None):
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "signature": signature,
        "model": {
            name: value.detach().cpu()
            for name, value in model.state_dict().items()
        },
        "training_seconds": training_seconds,
        "metadata": metadata or {},
    }
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_checkpoint(path, signature):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"Invalid V7.3 checkpoint: {path}")
    if payload.get("signature") != signature:
        raise V73CheckpointSignatureMismatch(f"V7.3 checkpoint provenance mismatch: {path}")
    return payload


def _validate_upstream_protocol(config, v72, v71):
    missing = [v for v in config.token_counts if v not in v72.token_counts]
    if missing:
        raise ValueError(f"V7.3 retained V7.2 controls lack token counts: {missing}")
    if RETAINED_V72_ARCHITECTURE not in v72.architectures:
        raise ValueError("V7.2 config did not train geometry_local_match.")
    if v72.base_steps != v71.base_steps:
        raise ValueError("V7.2/V7.1 frozen-L0 protocols disagree.")


def load_v73_config(path: str | Path) -> V73Config:
    source = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf8")) or {}
    training = raw.get("training", {})
    config = V73Config(
        source_path=source,
        v72_config=_path(raw.get("v72_config")),
        output_dir=_path(raw.get("output_dir")),
        device=str(raw.get("device", "cuda:0")),
        token_counts=tuple(int(v) for v in raw.get("token_counts", (8, 16))),
        architectures=tuple(str(v) for v in raw.get("architectures", V73_ARCHITECTURES)),
        residual_steps=int(training.get("residual_steps", 1200)),
        seed=int(training.get("seed", 0)),
    )
    if not config.token_counts or any(v < 2 for v in config.token_counts):
        raise ValueError("V7.3 token_counts must contain values >= 2.")
    if tuple(sorted(set(config.token_counts))) != config.token_counts:
        raise ValueError("V7.3 token_counts must be unique and increasing.")
    unknown = set(config.architectures) - set(V73_ARCHITECTURES)
    if unknown or len(config.architectures) != len(set(config.architectures)):
        raise ValueError(f"Invalid V7.3 architectures: {sorted(unknown)}")
    if config.residual_steps < 1:
        raise ValueError("V7.3 residual_steps must be positive.")
    return config


def _path(value) -> Path:
    if value is None or not str(value).strip():
        raise ValueError("V7.3 configuration contains an empty path.")
    return Path(str(value)).expanduser().resolve()


def _gain(baseline: float, current: float) -> float:
    return 0.0 if baseline <= 1e-12 else 100.0 * (baseline - current) / baseline


def _short(value: float) -> str:
    return f"{float(value):.8g}"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty V7.3 CSV: {path}")
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="streaming_couping/configs/v73_correspondence_ablation.yaml",
    )
    parser.add_argument("--device")
    parser.add_argument("--output-dir")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--residual-steps", type=int)
    parser.add_argument("--architectures", help="Comma-separated V7.3 subset.")
    parser.add_argument("--token-counts", help="Comma-separated values, e.g. 8,16.")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
