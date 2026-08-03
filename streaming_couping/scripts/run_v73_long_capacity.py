#!/usr/bin/env python3
"""Fit all available frames in the V7.3 long sequence as a capacity test."""

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
from streaming_couping.scripts.run_v7_fusion_ablation import (
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


BRANCHES = {
    "uniform_transport": ("uniform_transport", "normal"),
    "geometry_transport": ("geometry_transport", "normal"),
    "sam_transport": ("sam_transport", "normal"),
    "sam_geometry_transport": ("sam_geometry_transport", "normal"),
    # Same modules and parameter count as the combined branch, but SAM logits
    # are disabled throughout training and primary evaluation.
    "sam_geometry_train_sam_off": ("sam_geometry_transport", "sam_off"),
}

EVALUATION_VARIANTS = (
    "normal",
    "sam_off",
    "uniform_sam",
    "wrong_sam_identity",
    "shuffle_sam_time",
    "wrong_local_geometry",
)

OVERFIT_LOSS_THRESHOLD = 1e-4


def main() -> None:
    args = _parse_args()
    source = load_v73_config(args.config)
    device = args.device or source.device
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else Path("outputs/streaming_couping_v73_long_capacity").resolve()
    )
    selected = tuple(
        value.strip()
        for value in args.branches.split(",")
        if value.strip()
    )
    unknown = set(selected) - set(BRANCHES)
    if not selected or unknown:
        raise ValueError(f"Unknown V7.3 capacity branches: {sorted(unknown)}")
    result = run_capacity(
        v73_config=source,
        output_dir=output_dir,
        device=device,
        token_count=int(args.token_count),
        steps=int(args.steps),
        seed=int(args.seed),
        branches=selected,
        resume=bool(args.resume),
    )
    print(f"V7.3 long-sequence capacity result={result}")


def run_capacity(
    *,
    v73_config,
    output_dir: Path,
    device: str,
    token_count: int,
    steps: int,
    seed: int,
    branches: tuple[str, ...],
    resume: bool,
) -> Path:
    if token_count < 2 or steps < 1:
        raise ValueError("V7.3 capacity requires token_count >= 2 and steps >= 1.")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
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
        payload, name="long_capacity", minimum_tokens=token_count
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
    reference_index = int(payload["reference_sequence_index"])
    if reference_index != 0:
        raise ValueError("V7.3 long capacity expects frame 90 at sequence index 0.")
    frames = tuple(int(value) for value in payload["frame_indices"])
    expected_frames = tuple(range(90, 526, 15))
    if frames != expected_frames:
        raise ValueError(
            "V7.3 long capacity requires frames 90:15:525; "
            f"cache contains {frames}."
        )
    all_indices = [index for index in range(len(frames)) if index != reference_index]
    supervised_frames = tuple(frames[index] for index in all_indices)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_model, frozen_signature, frozen_source, frozen_sha256 = _load_l0(
        v71=v71,
        experiment=v71_experiment,
        batch=batch,
        device=device,
    )
    with torch.no_grad():
        base_output = _forward_model(
            base_model,
            batch=batch,
            baseline=baseline,
            reference_index=reference_index,
            variant="normal",
        )
    raw_metrics = _pose_metrics(
        baseline,
        target,
        reference_index=reference_index,
        translation_weight=experiment.training.translation_weight,
        evaluation_indices=all_indices,
    )
    base_metrics = _pose_metrics(
        base_output["world_to_camera"],
        target,
        reference_index=reference_index,
        translation_weight=experiment.training.translation_weight,
        evaluation_indices=all_indices,
    )
    rows = [
        _control_row(
            architecture="raw_streamvggt",
            frames=supervised_frames,
            metrics=raw_metrics,
            parameters=0,
        ),
        _control_row(
            architecture="frozen_l0",
            frames=supervised_frames,
            metrics=base_metrics,
            parameters=sum(
                parameter.numel() for parameter in base_model.parameters()
            ),
        ),
    ]
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
                batch=batch,
                baseline=baseline,
                reference_index=reference_index,
                variant=train_variant,
            )
        active_indices = [
            index
            for index in all_indices
            if bool(initial["active_frames"][0, index].cpu())
        ]
        if not active_indices:
            raise RuntimeError(f"V7.3 capacity branch {label} has no active frame.")
        training = _train_or_resume(
            model=model,
            label=label,
            architecture=architecture,
            train_variant=train_variant,
            output_dir=output_dir,
            resume=resume,
            batch=batch,
            baseline=baseline,
            target=target,
            reference_index=reference_index,
            active_indices=active_indices,
            steps=steps,
            experiment=experiment,
            token_count=token_count,
            seed=seed,
            frozen_l0_signature=frozen_signature,
            frames=supervised_frames,
        )
        with torch.no_grad():
            primary = _forward_model(
                model,
                batch=batch,
                baseline=baseline,
                reference_index=reference_index,
                variant=train_variant,
            )
        trained_active_indices = [
            index
            for index in all_indices
            if bool(primary["active_frames"][0, index].cpu())
        ]
        if trained_active_indices != active_indices:
            raise RuntimeError(
                f"V7.3 capacity branch {label} changed its active-frame set "
                "during training."
            )
        rows.append(
            _capacity_row(
                label=label,
                architecture=architecture,
                train_variant=train_variant,
                model=model,
                primary=primary,
                batch=batch,
                baseline=baseline,
                target=target,
                reference_index=reference_index,
                all_indices=all_indices,
                active_indices=active_indices,
                frames=supervised_frames,
                experiment=experiment,
                training=training,
                token_count=token_count,
                steps=steps,
            )
        )
    peak = (
        torch.cuda.max_memory_allocated() / (1024**3)
        if torch.cuda.is_available()
        else 0.0
    )
    for row in rows:
        row["peak_gpu_memory_gib"] = _short(peak)
    result = output_dir / "v73_long_capacity.csv"
    _write_csv(result, rows)
    (output_dir / "v73_long_capacity_metadata.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "purpose": "single_scene_full_long_sequence_capacity_only",
                "not_a_generalization_result": True,
                "frames": frames,
                "supervised_frames": supervised_frames,
                "token_count": token_count,
                "steps": steps,
                "seed": seed,
                "branches": branches,
                "frozen_l0_source": str(frozen_source),
                "frozen_l0_signature": frozen_signature,
                "frozen_l0_sha256": frozen_sha256,
                "result_csv": str(result),
            },
            indent=2,
            default=str,
            sort_keys=True,
        )
        + "\n",
        encoding="utf8",
    )
    print("V7.3 LONG-SEQUENCE CAPACITY ONLY (COPY THIS CSV)")
    print(result.read_text(encoding="utf8").rstrip())
    return result


def _load_l0(*, v71, experiment, batch, device):
    model = _make_l0(batch=batch, experiment=experiment, device=device)
    source = v71.output_dir / "frozen_l0.pt"
    if not source.is_file():
        raise FileNotFoundError(
            f"V7.3 long capacity requires {source}; complete the V7.1 "
            "experiment first."
        )
    signature = _v71_checkpoint_signature(
        v71, experiment, architecture="frozen_l0"
    )
    checkpoint = _load_v71_checkpoint(source, expected_signature=signature)
    model.load_state_dict(checkpoint["model"], strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return model, signature, source, _sha256_file(source)


def _make_l0(*, batch, experiment, device):
    from streaming_couping.src.learned_pose.v7_fusion import V7PoseFusion

    return V7PoseFusion(
        architecture="l0_camera_only",
        camera_dim=int(batch["camera_hidden"].shape[-1]),
        appearance_dim=int(batch["appearance"].shape[-1]),
        geometry_dim=int(batch["pose_geometry"].shape[-1]),
        local_feature_dim=int(batch["local_features"].shape[-1]),
        config=experiment.fusion,
    ).to(device)


def _train_or_resume(
    *, model, label, architecture, train_variant, output_dir, resume, batch,
    baseline, target, reference_index, active_indices, steps, experiment,
    token_count, seed, frozen_l0_signature, frames,
) -> dict[str, float | int]:
    signature = _signature(
        label=label,
        architecture=architecture,
        train_variant=train_variant,
        token_count=token_count,
        seed=seed,
        steps=steps,
        frozen_l0_signature=frozen_l0_signature,
        frames=frames,
        experiment=experiment,
    )
    path = output_dir / f"{label}_k{token_count:02d}.pt"
    if resume and path.is_file():
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("signature") == signature and "model" in checkpoint:
            model.load_state_dict(checkpoint["model"], strict=True)
            model.eval()
            print(f"V7.3 long capacity resumed branch={label}")
            return dict(checkpoint["training"])
        print(f"V7.3 long capacity invalidating stale checkpoint={path}")
    print(
        f"V7.3 long capacity training branch={label} "
        f"active={len(active_indices)}/{len(frames)}"
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


def _train(
    *, model, batch, baseline, target, reference_index, active_indices,
    steps, experiment, variant,
) -> dict[str, float | int]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
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
            variant=variant,
        )
        initial_loss = float(
            _pose_loss(
                initial["world_to_camera"],
                target,
                reference_index=reference_index,
                translation_weight=experiment.training.translation_weight,
                evaluation_indices=active_indices,
            ).cpu()
        )
    best_loss = float("inf")
    best_step = 0
    best_state = None
    model.train()
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        output = _forward_model(
            model,
            batch=batch,
            baseline=baseline,
            reference_index=reference_index,
            variant=variant,
        )
        loss = _pose_loss(
            output["world_to_camera"],
            target,
            reference_index=reference_index,
            translation_weight=experiment.training.translation_weight,
            evaluation_indices=active_indices,
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"Non-finite V7.3 long capacity loss at step {step}.")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            parameters, float(experiment.training.grad_clip_norm)
        )
        if not bool(torch.isfinite(norm)):
            raise RuntimeError(f"Non-finite V7.3 long capacity gradient at {step}.")
        optimizer.step()
        if step % int(experiment.training.log_every) == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                checked = _forward_model(
                    model,
                    batch=batch,
                    baseline=baseline,
                    reference_index=reference_index,
                    variant=variant,
                )
                current = float(
                    _pose_loss(
                        checked["world_to_camera"],
                        target,
                        reference_index=reference_index,
                        translation_weight=experiment.training.translation_weight,
                        evaluation_indices=active_indices,
                    ).cpu()
                )
            print(f"  step={step}/{steps} active_loss={current:.8f}")
            if current < best_loss:
                best_loss = current
                best_step = step
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
            model.train()
    if best_state is None:
        raise RuntimeError("V7.3 long capacity produced no checkpoint candidate.")
    model.load_state_dict(best_state)
    model.eval()
    return {
        "initial_active_loss": initial_loss,
        "best_active_loss": best_loss,
        "best_step": best_step,
    }


def _capacity_row(
    *, label, architecture, train_variant, model, primary, batch, baseline,
    target, reference_index, all_indices, active_indices, frames, experiment,
    training, token_count, steps,
) -> dict[str, Any]:
    inactive_indices = [index for index in all_indices if index not in active_indices]
    active = _metrics(
        primary["world_to_camera"], target, reference_index, active_indices, experiment
    )
    overall = _metrics(
        primary["world_to_camera"], target, reference_index, all_indices, experiment
    )
    inactive = (
        _metrics(
            primary["world_to_camera"],
            target,
            reference_index,
            inactive_indices,
            experiment,
        )
        if inactive_indices
        else None
    )
    with torch.no_grad():
        base = _forward_model(
            model.base_model,
            batch=batch,
            baseline=baseline,
            reference_index=reference_index,
            variant="normal",
        )
    fallback_exact = all(
        torch.equal(
            primary["world_to_camera"][:, index],
            base["world_to_camera"][:, index],
        )
        for index in inactive_indices
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
                active_indices,
                experiment,
            )["loss"]
    initial_loss = float(training["initial_active_loss"])
    final_loss = float(active["loss"])
    per_frame_losses = {
        int(frames[index - 1]): float(
            _metrics(
                primary["world_to_camera"],
                target,
                reference_index,
                [index],
                experiment,
            )["loss"]
        )
        for index in active_indices
    }
    worst_frame, maximum_frame_loss = max(
        per_frame_losses.items(), key=lambda item: item[1]
    )
    fitted_frames = sum(
        loss <= OVERFIT_LOSS_THRESHOLD
        for loss in per_frame_losses.values()
    )
    return {
        "architecture": label,
        "underlying_architecture": architecture,
        "train_input": train_variant,
        "token_count": token_count,
        "frames": " ".join(str(value) for value in frames),
        "total_nonreference_frames": len(all_indices),
        "active_frames": len(active_indices),
        "active_frame_indices": " ".join(
            str(frames[index - 1]) for index in active_indices
        ),
        "inactive_fallback_frames": len(inactive_indices),
        "inactive_frame_indices": " ".join(
            str(frames[index - 1]) for index in inactive_indices
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
        "initial_active_loss": _short(initial_loss),
        "final_active_rotation_deg": _short(active["rotation_degrees"]),
        "final_active_translation": _short(active["translation_native"]),
        "final_active_loss": _short(final_loss),
        "max_active_frame_loss": _short(maximum_frame_loss),
        "worst_active_frame": worst_frame,
        "fitted_active_frames": fitted_frames,
        "active_loss_drop_percent": _short(_gain(initial_loss, final_loss)),
        "all_frames_rotation_deg": _short(overall["rotation_degrees"]),
        "all_frames_translation": _short(overall["translation_native"]),
        "all_frames_loss": _short(overall["loss"]),
        "inactive_frames_loss": _short(inactive["loss"]) if inactive else "",
        "reference_exact": int(
            torch.equal(
                primary["world_to_camera"][:, reference_index],
                baseline[:, reference_index],
            )
        ),
        "inactive_fallback_exact": int(fallback_exact),
        "long_active_overfit_pass": int(
            maximum_frame_loss <= OVERFIT_LOSS_THRESHOLD
            and _gain(initial_loss, final_loss) >= 99.0
            and fallback_exact
        ),
        "normal_active_loss": _short(variant_losses["normal"]),
        "sam_off_active_loss": _short(variant_losses["sam_off"]),
        "uniform_sam_active_loss": _short(variant_losses["uniform_sam"]),
        "wrong_sam_identity_active_loss": _short(
            variant_losses["wrong_sam_identity"]
        ),
        "shuffle_sam_time_active_loss": _short(
            variant_losses["shuffle_sam_time"]
        ),
        "wrong_local_geometry_active_loss": _short(
            variant_losses["wrong_local_geometry"]
        ),
        "peak_gpu_memory_gib": "",
    }


def _control_row(*, architecture, frames, metrics, parameters) -> dict[str, Any]:
    return {
        "architecture": architecture,
        "underlying_architecture": "control",
        "train_input": "none",
        "token_count": 0,
        "frames": " ".join(str(value) for value in frames),
        "total_nonreference_frames": len(frames),
        "active_frames": 0,
        "active_frame_indices": "",
        "inactive_fallback_frames": 0,
        "inactive_frame_indices": "",
        "parameters": parameters,
        "trainable_parameters": 0,
        "steps": 0,
        "best_step": 0,
        "training_seconds": 0,
        "initial_active_loss": "",
        "final_active_rotation_deg": "",
        "final_active_translation": "",
        "final_active_loss": "",
        "max_active_frame_loss": "",
        "worst_active_frame": "",
        "fitted_active_frames": "",
        "active_loss_drop_percent": "",
        "all_frames_rotation_deg": _short(metrics["rotation_degrees"]),
        "all_frames_translation": _short(metrics["translation_native"]),
        "all_frames_loss": _short(metrics["loss"]),
        "inactive_frames_loss": "",
        "reference_exact": 1,
        "inactive_fallback_exact": 1,
        "long_active_overfit_pass": 0,
        "normal_active_loss": "",
        "sam_off_active_loss": "",
        "uniform_sam_active_loss": "",
        "wrong_sam_identity_active_loss": "",
        "shuffle_sam_time_active_loss": "",
        "wrong_local_geometry_active_loss": "",
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


def _signature(
    *, label, architecture, train_variant, token_count, seed, steps,
    frozen_l0_signature, frames, experiment,
) -> str:
    payload = {
        "schema": 1,
        "purpose": "v73_full_long_capacity",
        "label": label,
        "architecture": architecture,
        "train_variant": train_variant,
        "token_count": token_count,
        "seed": seed,
        "steps": steps,
        "frozen_l0_signature": frozen_l0_signature,
        "frames": frames,
        "fusion": asdict(experiment.fusion),
        "training": asdict(experiment.training),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf8")
    ).hexdigest()


def _gain(initial: float, final: float) -> float:
    return 0.0 if initial <= 1e-12 else 100.0 * (initial - final) / initial


def _short(value: float) -> str:
    return f"{float(value):.8g}"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
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
    parser.add_argument("--token-count", type=int, default=8)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--branches",
        default=",".join(BRANCHES),
        help="Comma-separated capacity branches.",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
