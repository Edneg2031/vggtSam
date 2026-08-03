#!/usr/bin/env python3
"""Test whether V7 SAM correspondence transfers to later temporal folds."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch

from streaming_couping.src.config import load_config
from streaming_couping.src.external_repos import maybe_add_repo_to_path
from streaming_couping.src.learned_pose.config import load_learned_pose_config
from streaming_couping.src.learned_pose.v73_correspondence_fusion import (
    V73FrozenCorrespondenceResidual,
)
from streaming_couping.src.v74_temporal_protocol import (
    EXPECTED_FRAMES,
    FOLDS,
    GEOMETRY_CONTROL,
    MINIMUM_CONTROL_GAIN_PERCENT,
    MINIMUM_PERTURBATION_DAMAGE_PERCENT,
    PRIMARY_SAM_BRANCH,
    SAM_OFF_CONTROL,
    V74_COLUMNS,
    annotate_controls,
    validate_folds,
    validate_rows,
)
from streaming_couping.scripts.run_v7_fusion_ablation import (
    _find_clip,
    _load_cache,
    _pose_metrics,
    _seed_everything,
    load_v7_config,
)
from streaming_couping.scripts.run_v71_instance_causality import (
    _checkpoint_signature as _v71_checkpoint_signature,
    _slice_batch_prefix,
    load_v71_config,
)
from streaming_couping.scripts.run_v72_local_token_ablation import (
    _limit_local_tokens,
    _prepare_payload,
    _sha256_file,
    _validate_local_payload,
    load_v72_config,
)
from streaming_couping.scripts.run_v73_correspondence_ablation import (
    _forward_model,
    load_v73_config,
)
from streaming_couping.scripts.run_v73_long_capacity import (
    BRANCHES,
    EVALUATION_VARIANTS,
    OVERFIT_LOSS_THRESHOLD,
    _gain,
    _make_l0,
    _short,
    _signature as _v73_long_signature,
    _train,
)

def main() -> None:
    args = _parse_args()
    source = load_v73_config(args.config)
    device = args.device or source.device
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else Path("outputs/streaming_couping_v74_temporal_scaling").resolve()
    )
    selected = tuple(
        value.strip() for value in args.branches.split(",") if value.strip()
    )
    unknown = set(selected) - set(BRANCHES)
    required = {PRIMARY_SAM_BRANCH, SAM_OFF_CONTROL, GEOMETRY_CONTROL}
    if not selected or unknown:
        raise ValueError(f"Unknown V7.4 branches: {sorted(unknown)}")
    if not required.issubset(selected):
        raise ValueError(
            "V7.4 requires geometry, SAM+geometry and the matched SAM-off "
            f"control; missing={sorted(required - set(selected))}."
        )
    result = run_temporal_scaling(
        v73_config=source,
        output_dir=output_dir,
        device=device,
        token_count=int(args.token_count),
        steps=int(args.steps),
        seed=int(args.seed),
        branches=selected,
        resume=bool(args.resume),
        frozen_l0_path=(
            Path(args.frozen_l0).expanduser().resolve()
            if args.frozen_l0
            else None
        ),
    )
    print(f"V7.4 temporal-scaling result={result}")


def run_temporal_scaling(
    *,
    v73_config,
    output_dir: Path,
    device: str,
    token_count: int,
    steps: int,
    seed: int,
    branches: tuple[str, ...],
    resume: bool,
    frozen_l0_path: Path | None,
) -> Path:
    if token_count < 2 or steps < 1:
        raise ValueError("V7.4 requires token_count >= 2 and steps >= 1.")
    v72 = load_v72_config(v73_config.v72_config)
    v71 = load_v71_config(v72.v71_config)
    source_v7 = load_v7_config(v71.v7_config)
    experiment = replace(
        source_v7,
        device=device,
        training=replace(source_v7.training, seed=seed),
    )
    v71_experiment = replace(
        source_v7,
        device=device,
        training=replace(source_v7.training, seed=v71.seed),
    )
    data = load_learned_pose_config(v72.data_config)
    long_clip = _find_clip(data, v71.long_clip_name)
    payload = _load_cache(data, long_clip)
    _validate_local_payload(
        payload,
        name="v74_temporal_scaling",
        minimum_tokens=token_count,
    )
    recovery = load_config(data.recovery_config)
    maybe_add_repo_to_path(recovery.streamvggt_repo)
    from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri

    batch, baseline, target = _prepare_payload(
        payload,
        config=experiment,
        pose_decoder=pose_encoding_to_extri_intri,
        device=device,
    )
    batch = _limit_local_tokens(batch, token_count)
    frames = tuple(int(value) for value in payload["frame_indices"])
    reference_index = int(payload["reference_sequence_index"])
    if frames != EXPECTED_FRAMES or reference_index != 0:
        raise ValueError(
            "V7.4 requires the frame-90 reference and frames 90:15:525; "
            f"cache contains reference={reference_index}, frames={frames}."
        )
    positions = {frame: index for index, frame in enumerate(frames)}
    validate_folds(FOLDS, available_frames=set(frames))
    output_dir.mkdir(parents=True, exist_ok=True)
    (
        base_model,
        frozen_signature,
        frozen_sha256,
        frozen_source,
    ) = _load_available_l0(
        explicit=frozen_l0_path,
        v71_output=v71.output_dir,
        v72_output=v72.output_dir,
        v73_output=v73_config.output_dir,
        v74_output=output_dir,
        v73_long_output=Path(
            "outputs/streaming_couping_v73_long_capacity"
        ).resolve(),
        v71=v71,
        experiment=v71_experiment,
        batch=batch,
        device=device,
    )
    _retain_clean_l0(
        model=base_model,
        signature=frozen_signature,
        source=frozen_source,
        source_sha256=frozen_sha256,
        destination=output_dir / "frozen_l0.pt",
    )
    with torch.no_grad():
        base_output = _forward_model(
            base_model,
            batch=batch,
            baseline=baseline,
            reference_index=reference_index,
            variant="normal",
        )

    rows: list[dict[str, Any]] = []
    for fold in FOLDS:
        train_indices = [positions[frame] for frame in fold.train_frames]
        test_indices = [positions[frame] for frame in fold.test_frames]
        raw_test = _metrics(
            baseline,
            target,
            reference_index,
            test_indices,
            experiment,
        )
        l0_test = _metrics(
            base_output["world_to_camera"],
            target,
            reference_index,
            test_indices,
            experiment,
        )
        rows.extend(
            [
                _control_row(
                    fold=fold,
                    architecture="raw_streamvggt",
                    metrics=raw_test,
                    parameters=0,
                    token_count=0,
                    seed=seed,
                ),
                _control_row(
                    fold=fold,
                    architecture="frozen_l0",
                    metrics=l0_test,
                    parameters=sum(
                        parameter.numel()
                        for parameter in base_model.parameters()
                    ),
                    token_count=0,
                    seed=seed,
                ),
            ]
        )
        prefix_length = positions[max(fold.train_frames)] + 1
        training_batch = _slice_batch_prefix(batch, length=prefix_length)
        training_baseline = baseline[:, :prefix_length]
        training_target = target[:, :prefix_length]
        for label in branches:
            architecture, train_variant = BRANCHES[label]
            _seed_everything(seed)
            model = V73FrozenCorrespondenceResidual(
                base_model=copy.deepcopy(base_model),
                architecture=architecture,
                sam_local_dim=int(batch["sam_local_features"].shape[-1]),
                geometry_local_dim=int(batch["local_features"].shape[-1]),
                config=experiment.fusion,
            ).to(device)
            model.eval()
            with torch.no_grad():
                initial = _forward_model(
                    model,
                    batch=training_batch,
                    baseline=training_baseline,
                    reference_index=reference_index,
                    variant=train_variant,
                )
            active_train_indices = [
                index
                for index in train_indices
                if bool(initial["active_frames"][0, index].cpu())
            ]
            if not active_train_indices:
                raise RuntimeError(
                    f"V7.4 fold={fold.name} branch={label} has no active "
                    "training frame."
                )
            training = _train_or_resume(
                model=model,
                fold=fold,
                label=label,
                architecture=architecture,
                train_variant=train_variant,
                output_dir=output_dir,
                resume=resume,
                batch=training_batch,
                baseline=training_baseline,
                target=training_target,
                reference_index=reference_index,
                active_indices=active_train_indices,
                steps=steps,
                experiment=experiment,
                token_count=token_count,
                seed=seed,
                frozen_l0_signature=frozen_signature,
            )
            with torch.no_grad():
                primary = _forward_model(
                    model,
                    batch=batch,
                    baseline=baseline,
                    reference_index=reference_index,
                    variant=train_variant,
                )
            current_active_train = [
                index
                for index in train_indices
                if bool(primary["active_frames"][0, index].cpu())
            ]
            if current_active_train != active_train_indices:
                raise RuntimeError(
                    f"V7.4 fold={fold.name} branch={label} changed its "
                    "training active-frame set."
                )
            rows.append(
                _model_row(
                    fold=fold,
                    label=label,
                    architecture=architecture,
                    train_variant=train_variant,
                    model=model,
                    primary=primary,
                    base_output=base_output,
                    batch=batch,
                    baseline=baseline,
                    target=target,
                    reference_index=reference_index,
                    train_indices=train_indices,
                    active_train_indices=active_train_indices,
                    test_indices=test_indices,
                    experiment=experiment,
                    training=training,
                    token_count=token_count,
                    steps=steps,
                    seed=seed,
                )
            )
    annotate_controls(rows)
    result = output_dir / "v74_temporal_scaling.csv"
    _write_csv(result, rows)
    metadata = {
        "schema": 1,
        "purpose": "same_scene_temporal_prefix_generalization",
        "cross_scene_generalization": False,
        "reference_frame": 90,
        "folds": [asdict(fold) for fold in FOLDS],
        "token_count": token_count,
        "steps": steps,
        "seed": seed,
        "branches": branches,
        "minimum_perturbation_damage_percent": (
            MINIMUM_PERTURBATION_DAMAGE_PERCENT
        ),
        "minimum_control_gain_percent": MINIMUM_CONTROL_GAIN_PERCENT,
        "frozen_l0_source": str(frozen_source),
        "frozen_l0_signature": frozen_signature,
        "frozen_l0_sha256": frozen_sha256,
        "result_csv": str(result),
    }
    (output_dir / "v74_temporal_scaling_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf8",
    )
    print("V7.4 TEMPORAL SCALING (COPY THIS CSV)")
    print(result.read_text(encoding="utf8").rstrip())
    return result


def _train_or_resume(
    *,
    model,
    fold,
    label,
    architecture,
    train_variant,
    output_dir,
    resume,
    batch,
    baseline,
    target,
    reference_index,
    active_indices,
    steps,
    experiment,
    token_count,
    seed,
    frozen_l0_signature,
) -> dict[str, float | int]:
    signature = _signature(
        fold=fold,
        label=label,
        architecture=architecture,
        train_variant=train_variant,
        token_count=token_count,
        seed=seed,
        steps=steps,
        frozen_l0_signature=frozen_l0_signature,
        experiment=experiment,
    )
    checkpoint_dir = output_dir / "checkpoints" / fold.name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"{label}_k{token_count:02d}_seed{seed}.pt"
    if resume and path.is_file():
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("signature") == signature and "model" in checkpoint:
            model.load_state_dict(checkpoint["model"], strict=True)
            model.eval()
            print(f"V7.4 resumed fold={fold.name} branch={label}")
            return dict(checkpoint["training"])
        print(f"V7.4 invalidating stale checkpoint={path}")
    print(
        f"V7.4 training fold={fold.name} branch={label} "
        f"active={len(active_indices)}/{len(fold.train_frames)}"
    )
    started = time.perf_counter()
    training = _train(
        model=model,
        batch=batch,
        baseline=baseline,
        target=target,
        reference_index=reference_index,
        active_indices=active_indices,
        steps=steps,
        experiment=experiment,
        variant=train_variant,
    )
    training["training_seconds"] = time.perf_counter() - started
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "signature": signature,
            "model": {
                name: value.detach().cpu()
                for name, value in model.state_dict().items()
            },
            "training": training,
        },
        temporary,
    )
    temporary.replace(path)
    return training


def _load_available_l0(
    *,
    explicit: Path | None,
    v71_output: Path,
    v72_output: Path,
    v73_output: Path,
    v74_output: Path,
    v73_long_output: Path,
    v71,
    experiment,
    batch,
    device,
):
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(
                f"Explicit V7.4 frozen L0 does not exist: {explicit}"
            )
        candidates = (explicit,)
    else:
        candidates = (
            v71_output / "frozen_l0.pt",
            v74_output / "frozen_l0.pt",
            v73_output / "frozen_l0.pt",
            v72_output / "frozen_l0.pt",
        )
    failures = []
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            model, signature, sha256 = _load_compatible_l0(
                v71=v71,
                experiment=experiment,
                batch=batch,
                device=device,
                source=candidate,
            )
        except (RuntimeError, ValueError) as error:
            failures.append(f"{candidate}: {error}")
            continue
        return model, signature, sha256, candidate
    recovered = _recover_l0_from_v73_long(
        output_dir=v73_long_output,
        v71=v71,
        experiment=experiment,
        batch=batch,
        device=device,
    )
    if recovered is not None:
        return recovered
    if failures:
        details = "\n  ".join(failures)
        raise ValueError(
            "V7.4 found frozen L0 files, but none has compatible V7.1 "
            f"provenance:\n  {details}"
        )
    searched = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "V7.4 could not find a compatible frozen L0. Searched:\n  "
        f"{searched}\nRun commands_v71_instance_causality.txt only if none "
        "of these retained artifacts exists."
    )


def _recover_l0_from_v73_long(
    *, output_dir, v71, experiment, batch, device
):
    metadata_path = output_dir / "v73_long_capacity_metadata.json"
    if not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf8"))
    expected_l0 = _v71_checkpoint_signature(
        v71,
        experiment,
        architecture="frozen_l0",
    )
    if str(metadata.get("frozen_l0_signature", "")) != expected_l0:
        raise ValueError(
            "V7.3 long-capacity metadata does not originate from the "
            f"required V7.1 L0: {metadata_path}"
        )
    token_count = int(metadata["token_count"])
    steps = int(metadata["steps"])
    seed = int(metadata["seed"])
    frames = tuple(int(value) for value in metadata["supervised_frames"])
    branches = tuple(str(value) for value in metadata["branches"])
    long_experiment = replace(
        experiment,
        training=replace(experiment.training, seed=seed),
    )
    failures = []
    for label in branches:
        if label not in BRANCHES:
            continue
        architecture, train_variant = BRANCHES[label]
        path = output_dir / f"{label}_k{token_count:02d}.pt"
        if not path.is_file():
            continue
        expected_checkpoint = _v73_long_signature(
            label=label,
            architecture=architecture,
            train_variant=train_variant,
            token_count=token_count,
            seed=seed,
            steps=steps,
            frozen_l0_signature=expected_l0,
            frames=frames,
            experiment=long_experiment,
        )
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if (
            not isinstance(checkpoint, dict)
            or checkpoint.get("signature") != expected_checkpoint
            or not isinstance(checkpoint.get("model"), dict)
        ):
            failures.append(str(path))
            continue
        prefix = "base_model."
        base_state = {
            name[len(prefix) :]: value
            for name, value in checkpoint["model"].items()
            if name.startswith(prefix)
        }
        model = _make_l0(batch=batch, experiment=experiment, device=device)
        try:
            model.load_state_dict(base_state, strict=True)
        except RuntimeError:
            failures.append(str(path))
            continue
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval()
        sha256 = _sha256_file(path)
        print(
            "V7.4 recovered frozen L0 from signed V7.3 long checkpoint: "
            f"{path} sha256={sha256}"
        )
        return model, expected_l0, sha256, path
    if failures:
        raise ValueError(
            "V7.4 found V7.3 long checkpoints but their signatures or "
            f"embedded L0 states are invalid: {failures}"
        )
    return None


def _retain_clean_l0(
    *, model, signature, source, source_sha256, destination
) -> None:
    if source.resolve() == destination.resolve() and destination.is_file():
        return
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(
        {
            "signature": signature,
            "model": {
                name: value.detach().cpu()
                for name, value in model.state_dict().items()
            },
            "metadata": {
                "source": str(source),
                "source_signature": signature,
                "source_sha256": source_sha256,
            },
        },
        temporary,
    )
    temporary.replace(destination)
    print(f"V7.4 retained clean frozen L0: {destination}")


def _load_compatible_l0(*, v71, experiment, batch, device, source):
    model = _make_l0(batch=batch, experiment=experiment, device=device)
    expected = _v71_checkpoint_signature(
        v71,
        experiment,
        architecture="frozen_l0",
    )
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"Invalid V7.4 frozen L0 checkpoint: {source}")
    metadata = checkpoint.get("metadata", {})
    source_signature = (
        metadata.get("source_signature", "")
        if isinstance(metadata, dict)
        else ""
    )
    signatures = {
        str(checkpoint.get("signature", "")),
        str(source_signature),
    }
    if expected not in signatures:
        raise ValueError(
            "V7.4 frozen L0 provenance mismatch: "
            f"{source}. Expected V7.1 signature={expected}."
        )
    model.load_state_dict(checkpoint["model"], strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    sha256 = _sha256_file(source)
    print(f"V7.4 loaded compatible frozen L0: {source} sha256={sha256}")
    return model, expected, sha256


def _model_row(
    *,
    fold,
    label,
    architecture,
    train_variant,
    model,
    primary,
    base_output,
    batch,
    baseline,
    target,
    reference_index,
    train_indices,
    active_train_indices,
    test_indices,
    experiment,
    training,
    token_count,
    steps,
    seed,
) -> dict[str, Any]:
    active_test_indices = [
        index
        for index in test_indices
        if bool(primary["active_frames"][0, index].cpu())
    ]
    inactive_test_indices = [
        index for index in test_indices if index not in active_test_indices
    ]
    train_metrics = _metrics(
        primary["world_to_camera"],
        target,
        reference_index,
        active_train_indices,
        experiment,
    )
    test_metrics = _metrics(
        primary["world_to_camera"],
        target,
        reference_index,
        test_indices,
        experiment,
    )
    active_test_metrics = (
        _metrics(
            primary["world_to_camera"],
            target,
            reference_index,
            active_test_indices,
            experiment,
        )
        if active_test_indices
        else None
    )
    l0_test = _metrics(
        base_output["world_to_camera"],
        target,
        reference_index,
        test_indices,
        experiment,
    )
    fallback_exact = all(
        torch.equal(
            primary["world_to_camera"][:, index],
            base_output["world_to_camera"][:, index],
        )
        for index in inactive_test_indices
    )
    variant_losses = {}
    with torch.no_grad():
        for variant in EVALUATION_VARIANTS:
            output = _forward_model(
                model,
                batch=batch,
                baseline=baseline,
                reference_index=reference_index,
                variant=variant,
            )
            variant_losses[variant] = _metrics(
                output["world_to_camera"],
                target,
                reference_index,
                test_indices,
                experiment,
            )["loss"]
    train_frame_losses = _per_frame_losses(
        primary["world_to_camera"],
        target,
        reference_index,
        active_train_indices,
        experiment,
    )
    test_frame_losses = _per_frame_losses(
        primary["world_to_camera"],
        target,
        reference_index,
        test_indices,
        experiment,
    )
    worst_train_index, maximum_train_loss = max(
        train_frame_losses.items(), key=lambda item: item[1]
    )
    worst_test_index, maximum_test_loss = max(
        test_frame_losses.items(), key=lambda item: item[1]
    )
    initial_loss = float(training["initial_active_loss"])
    train_loss = float(train_metrics["loss"])
    reference_exact = torch.equal(
        primary["world_to_camera"][:, reference_index],
        baseline[:, reference_index],
    )
    peak = (
        torch.cuda.max_memory_allocated() / (1024**3)
        if torch.cuda.is_available()
        else 0.0
    )
    return {
        "fold": fold.name,
        "architecture": label,
        "underlying_architecture": architecture,
        "train_input": train_variant,
        "token_count": token_count,
        "seed": seed,
        "train_frames": _frame_text(fold.train_frames),
        "test_frames": _frame_text(fold.test_frames),
        "train_active_frames": len(active_train_indices),
        "train_active_frame_indices": _indices_to_frames(
            active_train_indices
        ),
        "test_active_frames": len(active_test_indices),
        "test_active_frame_indices": _indices_to_frames(
            active_test_indices
        ),
        "test_inactive_frames": len(inactive_test_indices),
        "test_inactive_frame_indices": _indices_to_frames(
            inactive_test_indices
        ),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "steps": steps,
        "best_step": int(training["best_step"]),
        "training_seconds": _short(training["training_seconds"]),
        "initial_train_active_loss": _short(initial_loss),
        "final_train_active_loss": _short(train_loss),
        "max_train_frame_loss": _short(maximum_train_loss),
        "worst_train_frame": _sequence_frame(worst_train_index),
        "train_loss_drop_percent": _short(_gain(initial_loss, train_loss)),
        "train_overfit_pass": int(
            maximum_train_loss <= OVERFIT_LOSS_THRESHOLD
            and _gain(initial_loss, train_loss) >= 99.0
        ),
        "test_rotation_deg": _short(test_metrics["rotation_degrees"]),
        "test_translation": _short(test_metrics["translation_native"]),
        "test_loss": _short(test_metrics["loss"]),
        "test_active_loss": (
            _short(active_test_metrics["loss"])
            if active_test_metrics is not None
            else ""
        ),
        "max_test_frame_loss": _short(maximum_test_loss),
        "worst_test_frame": _sequence_frame(worst_test_index),
        "gain_vs_frozen_l0_percent": _short(
            _gain(float(l0_test["loss"]), float(test_metrics["loss"]))
        ),
        "geometry_control_test_loss": "",
        "sam_off_trained_control_test_loss": "",
        "gain_vs_geometry_percent": "",
        "gain_vs_sam_off_trained_percent": "",
        "normal_test_loss": _short(variant_losses["normal"]),
        "sam_off_test_loss": _short(variant_losses["sam_off"]),
        "uniform_sam_test_loss": _short(variant_losses["uniform_sam"]),
        "wrong_sam_identity_test_loss": _short(
            variant_losses["wrong_sam_identity"]
        ),
        "shuffle_sam_time_test_loss": _short(
            variant_losses["shuffle_sam_time"]
        ),
        "wrong_local_geometry_test_loss": _short(
            variant_losses["wrong_local_geometry"]
        ),
        "sam_off_damage_percent": _short(
            _damage(variant_losses["normal"], variant_losses["sam_off"])
        ),
        "uniform_sam_damage_percent": _short(
            _damage(variant_losses["normal"], variant_losses["uniform_sam"])
        ),
        "wrong_sam_identity_damage_percent": _short(
            _damage(
                variant_losses["normal"],
                variant_losses["wrong_sam_identity"],
            )
        ),
        "shuffle_sam_time_damage_percent": _short(
            _damage(
                variant_losses["normal"],
                variant_losses["shuffle_sam_time"],
            )
        ),
        "reference_exact": int(reference_exact),
        "inactive_fallback_exact": int(fallback_exact),
        "control_support_exact": 0,
        "beats_geometry_control": 0,
        "beats_sam_off_trained_control": 0,
        "fold_sam_causal_pass": 0,
        "all_folds_sam_causal_pass": 0,
        "peak_gpu_memory_gib": _short(peak),
    }


def _control_row(
    *, fold, architecture, metrics, parameters, token_count, seed
) -> dict[str, Any]:
    return {
        "fold": fold.name,
        "architecture": architecture,
        "underlying_architecture": "control",
        "train_input": "none",
        "token_count": token_count,
        "seed": seed,
        "train_frames": _frame_text(fold.train_frames),
        "test_frames": _frame_text(fold.test_frames),
        "train_active_frames": 0,
        "train_active_frame_indices": "",
        "test_active_frames": 0,
        "test_active_frame_indices": "",
        "test_inactive_frames": 0,
        "test_inactive_frame_indices": "",
        "parameters": parameters,
        "trainable_parameters": 0,
        "steps": 0,
        "best_step": 0,
        "training_seconds": 0,
        "initial_train_active_loss": "",
        "final_train_active_loss": "",
        "max_train_frame_loss": "",
        "worst_train_frame": "",
        "train_loss_drop_percent": "",
        "train_overfit_pass": 0,
        "test_rotation_deg": _short(metrics["rotation_degrees"]),
        "test_translation": _short(metrics["translation_native"]),
        "test_loss": _short(metrics["loss"]),
        "test_active_loss": "",
        "max_test_frame_loss": "",
        "worst_test_frame": "",
        "gain_vs_frozen_l0_percent": "",
        "geometry_control_test_loss": "",
        "sam_off_trained_control_test_loss": "",
        "gain_vs_geometry_percent": "",
        "gain_vs_sam_off_trained_percent": "",
        "normal_test_loss": "",
        "sam_off_test_loss": "",
        "uniform_sam_test_loss": "",
        "wrong_sam_identity_test_loss": "",
        "shuffle_sam_time_test_loss": "",
        "wrong_local_geometry_test_loss": "",
        "sam_off_damage_percent": "",
        "uniform_sam_damage_percent": "",
        "wrong_sam_identity_damage_percent": "",
        "shuffle_sam_time_damage_percent": "",
        "reference_exact": 1,
        "inactive_fallback_exact": 1,
        "control_support_exact": 0,
        "beats_geometry_control": 0,
        "beats_sam_off_trained_control": 0,
        "fold_sam_causal_pass": 0,
        "all_folds_sam_causal_pass": 0,
        "peak_gpu_memory_gib": "",
    }


def _metrics(pose, target, reference_index, indices, experiment):
    return _pose_metrics(
        pose,
        target,
        reference_index=reference_index,
        translation_weight=experiment.training.translation_weight,
        evaluation_indices=indices,
    )


def _per_frame_losses(
    pose, target, reference_index, indices, experiment
) -> dict[int, float]:
    return {
        index: float(
            _metrics(
                pose,
                target,
                reference_index,
                [index],
                experiment,
            )["loss"]
        )
        for index in indices
    }


def _indices_to_frames(indices) -> str:
    return " ".join(str(_sequence_frame(index)) for index in indices)


def _sequence_frame(index: int) -> int:
    return EXPECTED_FRAMES[int(index)]


def _frame_text(frames) -> str:
    return " ".join(str(value) for value in frames)


def _damage(normal: float, perturbed: float) -> float:
    denominator = max(abs(float(normal)), 1e-12)
    return 100.0 * (float(perturbed) - float(normal)) / denominator


def _signature(
    *,
    fold,
    label,
    architecture,
    train_variant,
    token_count,
    seed,
    steps,
    frozen_l0_signature,
    experiment,
) -> str:
    payload = {
        "schema": 1,
        "purpose": "v74_temporal_prefix_generalization",
        "fold": asdict(fold),
        "label": label,
        "architecture": architecture,
        "train_variant": train_variant,
        "token_count": token_count,
        "seed": seed,
        "steps": steps,
        "frozen_l0_signature": frozen_l0_signature,
        "fusion": asdict(experiment.fusion),
        "training": asdict(experiment.training),
    }
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf8")
    return hashlib.sha256(serialized).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    validate_rows(rows)
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=V74_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="streaming_couping/configs/v73_correspondence_ablation.yaml",
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--device")
    parser.add_argument("--frozen-l0")
    parser.add_argument("--token-count", type=int, default=8)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--branches",
        default=",".join(BRANCHES),
        help="Comma-separated V7.4 branch subset.",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
