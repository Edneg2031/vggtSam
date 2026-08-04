#!/usr/bin/env python3
"""Run V8 matcher-first SAM3.1/StreamVGGT causal ablations."""

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
from streaming_couping.src.learned_pose.v73_correspondence_fusion import (
    perturb_v73_inputs,
)
from streaming_couping.src.learned_pose.v80_supervised_correspondence import (
    V80MatchingConfig,
    V80SupervisedCorrespondenceResidual,
    affinity_health,
    compute_v80_matching_loss,
    configure_v80_training_stage,
    projection_gradient_norms,
    sample_gt_world_at_local_tokens,
    v80_memory_write,
)
from streaming_couping.src.v74_temporal_protocol import EXPECTED_FRAMES, FOLDS
from streaming_couping.scripts.run_v7_fusion_ablation import (
    _find_clip,
    _load_cache,
    _pose_loss,
    _pose_metrics,
    _seed_everything,
    load_v7_config,
)
from streaming_couping.scripts.run_v71_instance_causality import (
    _slice_batch_prefix,
    load_v71_config,
)
from streaming_couping.scripts.run_v72_local_token_ablation import (
    _limit_local_tokens,
    _prepare_payload,
    _validate_local_payload,
    load_v72_config,
)
from streaming_couping.scripts.run_v73_correspondence_ablation import (
    load_v73_config,
)
from streaming_couping.scripts.run_v74_temporal_scaling import (
    _train_or_resume_v74_l0,
)


@dataclass(frozen=True)
class V80Branch:
    architecture: str
    value_mode: str
    train_variant: str = "normal"


BRANCHES = {
    "geometry_match": V80Branch("geometry_transport", "geometry_only"),
    "sam_match": V80Branch("sam_transport", "geometry_only"),
    "sam_geometry_match": V80Branch(
        "sam_geometry_transport", "geometry_only"
    ),
    "sam_geometry_train_sam_off": V80Branch(
        "sam_geometry_transport", "geometry_only", "sam_off"
    ),
    # V8.1 capacity check. It is report-only and never defines V8.0 success.
    "sam_geometry_dual_value": V80Branch(
        "sam_geometry_transport", "geometry_sam_dual"
    ),
}

PRIMARY = "sam_geometry_match"
GEOMETRY_CONTROL = "geometry_match"
SAM_OFF_CONTROL = "sam_geometry_train_sam_off"
EVALUATION_VARIANTS = ("normal", "sam_off", "wrong_sam_identity", "shuffle_sam_time")


@dataclass(frozen=True)
class V80Config:
    v73_config: Path
    data_config: Path
    output_dir: Path
    device: str
    token_count: int
    match_steps: int
    pose_steps: int
    base_steps: int
    seed: int
    branches: tuple[str, ...]
    matching: V80MatchingConfig


def main() -> None:
    args = _parse_args()
    config = load_v80_config(args.config)
    if args.output_dir:
        config = replace(config, output_dir=Path(args.output_dir).expanduser().resolve())
    if args.device:
        config = replace(config, device=args.device)
    if args.match_steps is not None:
        config = replace(config, match_steps=int(args.match_steps))
    if args.pose_steps is not None:
        config = replace(config, pose_steps=int(args.pose_steps))
    result = run_v80(config, resume=bool(args.resume))
    print(f"V8 supervised-correspondence result={result}")


def run_v80(config: V80Config, *, resume: bool) -> Path:
    _validate_config(config)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    v73 = load_v73_config(config.v73_config)
    v72 = load_v72_config(v73.v72_config)
    v71 = load_v71_config(v72.v71_config)
    source_v7 = load_v7_config(v71.v7_config)
    experiment = replace(
        source_v7,
        device=config.device,
        training=replace(source_v7.training, seed=config.seed),
    )
    data = load_learned_pose_config(config.data_config)
    clip = _find_clip(data, v71.long_clip_name)
    payload = _load_cache(data, clip)
    _validate_local_payload(payload, name="v80", minimum_tokens=config.token_count)
    if str(payload.get("instance_source", "")) != "sam31_online":
        raise ValueError("V8 requires the dynamic SAM3.1 online-instance cache.")
    for name in ("target_world_points", "target_world_to_camera"):
        if not torch.is_tensor(payload.get(name)):
            raise ValueError(f"V8 cache lacks training-only GT field {name!r}.")
    recovery = load_config(data.recovery_config)
    maybe_add_repo_to_path(recovery.streamvggt_repo)
    from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri

    batch, baseline, target = _prepare_payload(
        payload,
        config=experiment,
        pose_decoder=pose_encoding_to_extri_intri,
        device=config.device,
    )
    batch = _limit_local_tokens(batch, config.token_count)
    frames = tuple(int(value) for value in payload["frame_indices"])
    reference_index = int(payload["reference_sequence_index"])
    if frames != EXPECTED_FRAMES or reference_index != 0:
        raise ValueError(
            "V8 requires frame 90 as gauge and frames 90:15:525; "
            f"got reference={reference_index}, frames={frames}."
        )
    positions = {frame: index for index, frame in enumerate(frames)}
    config.output_dir.mkdir(parents=True, exist_ok=True)

    base_model, l0_signature, l0_source, l0_sha256 = _train_or_resume_v74_l0(
        output_dir=config.output_dir,
        resume=resume,
        batch=batch,
        baseline=baseline,
        target=target,
        reference_index=reference_index,
        positions=positions,
        base_frames=v71.base_train_frames,
        experiment=experiment,
        base_steps=config.base_steps,
        seed=config.seed,
        data_config=config.data_config,
        device=config.device,
    )

    dense_gt = torch.as_tensor(payload["target_world_points"]).to(config.device)
    local_gt, local_gt_valid = sample_gt_world_at_local_tokens(
        local_features=batch["local_features"],
        local_valid=batch["local_valid"],
        target_world_points=dense_gt,
    )
    del dense_gt
    batch["local_gt_world"] = local_gt
    batch["local_gt_valid"] = local_gt_valid

    with torch.no_grad():
        base_output = _forward_base(
            base_model, batch=batch, baseline=baseline, reference_index=reference_index
        )
    rows: list[dict[str, Any]] = []
    for fold in FOLDS:
        train_indices = [positions[frame] for frame in fold.train_frames]
        test_indices = [positions[frame] for frame in fold.test_frames]
        prefix_length = positions[max(fold.train_frames)] + 1
        training_batch = _slice_batch_prefix(batch, length=prefix_length)
        training_baseline = baseline[:, :prefix_length]
        training_target = target[:, :prefix_length]
        for label in config.branches:
            spec = BRANCHES[label]
            _seed_everything(config.seed)
            model = V80SupervisedCorrespondenceResidual(
                base_model=copy.deepcopy(base_model),
                architecture=spec.architecture,
                sam_local_dim=int(batch["sam_local_features"].shape[-1]),
                geometry_local_dim=int(batch["local_features"].shape[-1]),
                config=experiment.fusion,
                value_mode=spec.value_mode,
                pose_input_mode="evidence_only",
            ).to(config.device)
            match_training = _train_or_resume_matching(
                model=model,
                label=label,
                fold=fold.name,
                output_dir=config.output_dir,
                resume=resume,
                batch=training_batch,
                baseline=training_baseline,
                reference_index=reference_index,
                indices=train_indices,
                steps=config.match_steps,
                experiment=experiment,
                matching=config.matching,
                variant=spec.train_variant,
                signature_context=l0_signature,
                seed=config.seed,
            )
            match_train = _evaluate_matching(
                model,
                batch=training_batch,
                baseline=training_baseline,
                reference_index=reference_index,
                indices=train_indices,
                matching=config.matching,
                variant=spec.train_variant,
            )
            match_test = _evaluate_matching(
                model,
                batch=batch,
                baseline=baseline,
                reference_index=reference_index,
                indices=test_indices,
                matching=config.matching,
                variant=spec.train_variant,
            )
            pose_train_indices = [
                index
                for index in train_indices
                if bool(match_train["result"].active_frames[0, index].cpu())
            ]
            if not pose_train_indices:
                raise RuntimeError(
                    f"V8 fold={fold.name} branch={label} has no GT-matched pose frame."
                )
            pose_training = _train_or_resume_pose(
                model=model,
                label=label,
                fold=fold.name,
                output_dir=config.output_dir,
                resume=resume,
                batch=training_batch,
                baseline=training_baseline,
                target=training_target,
                reference_index=reference_index,
                indices=pose_train_indices,
                steps=config.pose_steps,
                experiment=experiment,
                signature_context=match_training["checkpoint_signature"],
                seed=config.seed,
            )
            row = _result_row(
                fold=fold,
                label=label,
                spec=spec,
                model=model,
                batch=batch,
                baseline=baseline,
                target=target,
                base_output=base_output,
                reference_index=reference_index,
                train_indices=train_indices,
                pose_train_indices=pose_train_indices,
                test_indices=test_indices,
                experiment=experiment,
                matching=config.matching,
                match_training=match_training,
                pose_training=pose_training,
                token_count=config.token_count,
                seed=config.seed,
            )
            rows.append(row)
    _annotate_decisions(rows)
    result = config.output_dir / "v80_supervised_correspondence.csv"
    _write_csv(result, rows)
    metadata = {
        "schema": 1,
        "purpose": "matcher_first_sam31_streamvggt_causality",
        "config": _jsonable_config(config),
        "l0_source": str(l0_source),
        "l0_sha256": l0_sha256,
        "pose_input_mode": "evidence_only",
        "matching_gt": "mesh_rasterized_world_pointmap_sampled_at_local_token_uv",
        "unmatched_query_policy": "excluded",
        "dual_value_is_report_only": True,
        "result_csv": str(result),
    }
    (config.output_dir / "v80_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf8",
    )
    print("V8 SUPERVISED CORRESPONDENCE (COPY THIS CSV)")
    print(result.read_text(encoding="utf8").rstrip())
    return result


def _forward_base(model, *, batch, baseline, reference_index):
    return model(
        camera_hidden=batch["camera_hidden"],
        baseline_world_to_camera=baseline,
        appearance=batch["appearance"],
        geometry=batch["pose_geometry"],
        quality=batch["quality"],
        observed=batch["observed"],
        identity_valid=batch["identity_valid"],
        identity_unknown=batch["identity_unknown"],
        local_features=batch["local_features"],
        local_valid=batch["local_valid"],
        reference_index=reference_index,
    )


def _forward_v80(model, *, batch, baseline, reference_index, variant):
    inputs, uniform_sam = perturb_v73_inputs(batch, variant)
    output = model(
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
    return output, inputs


def _matching_result(
    model,
    *,
    batch,
    baseline,
    reference_index,
    indices,
    matching,
    variant,
):
    output, inputs = _forward_v80(
        model,
        batch=batch,
        baseline=baseline,
        reference_index=reference_index,
        variant=variant,
    )
    result = compute_v80_matching_loss(
        output,
        current_gt_world=batch["local_gt_world"],
        current_gt_valid=batch["local_gt_valid"],
        source_local_valid=inputs["local_valid"],
        memory_write=v80_memory_write(
            inputs, min_confidence=model.config.min_track_confidence
        ),
        config=matching,
        sequence_indices=indices,
    )
    return output, result


def _evaluate_matching(model, **kwargs):
    model.eval()
    with torch.no_grad():
        output, result = _matching_result(model, **kwargs)
    return {
        "output": output,
        "result": result,
        "metrics": result.detached_metrics(),
        "health": affinity_health(output),
    }


def _train_or_resume_matching(
    *, model, label, fold, output_dir, resume, batch, baseline,
    reference_index, indices, steps, experiment, matching, variant,
    signature_context, seed,
):
    signature = _stage_signature(
        stage="matching", label=label, fold=fold, steps=steps, seed=seed,
        context=signature_context, matching=matching,
    )
    path = output_dir / "checkpoints" / fold / f"{label}_matching.pt"
    resumed = _maybe_resume(model, path=path, signature=signature, resume=resume)
    if resumed is not None:
        return resumed
    parameters = configure_v80_training_stage(model, "matching")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(experiment.training.learning_rate),
        weight_decay=float(experiment.training.weight_decay),
    )
    model.eval()
    with torch.no_grad():
        _, initial_result = _matching_result(
            model, batch=batch, baseline=baseline,
            reference_index=reference_index, indices=indices,
            matching=matching, variant=variant,
        )
    if int(initial_result.supervised_queries) == 0:
        raise RuntimeError(f"V8 fold={fold} branch={label} has no GT matches.")
    initial_loss = float(initial_result.loss.cpu())
    best_loss = float("inf")
    best_step = 0
    best_state = None
    step20_loss = initial_loss
    maximum_grads = {
        "geometry_projection_grad_norm": 0.0,
        "sam_projection_grad_norm": 0.0,
        "sam_to_geometry_grad_ratio": 0.0,
    }
    started = time.perf_counter()
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        _, result = _matching_result(
            model, batch=batch, baseline=baseline,
            reference_index=reference_index, indices=indices,
            matching=matching, variant=variant,
        )
        if not bool(torch.isfinite(result.loss)):
            raise RuntimeError(f"Non-finite V8 match loss at step={step}.")
        result.loss.backward()
        grads = projection_gradient_norms(model)
        for name, value in grads.items():
            maximum_grads[name] = max(maximum_grads[name], float(value))
        norm = torch.nn.utils.clip_grad_norm_(
            parameters, float(experiment.training.grad_clip_norm)
        )
        if not bool(torch.isfinite(norm)):
            raise RuntimeError(f"Non-finite V8 match gradient at step={step}.")
        optimizer.step()
        should_check = step == min(20, steps) or step % int(
            experiment.training.log_every
        ) == 0 or step == steps
        if should_check:
            with torch.no_grad():
                _, checked = _matching_result(
                    model, batch=batch, baseline=baseline,
                    reference_index=reference_index, indices=indices,
                    matching=matching, variant=variant,
                )
            current = float(checked.loss.cpu())
            if step == min(20, steps):
                step20_loss = current
            print(
                f"V8 match fold={fold} branch={label} "
                f"step={step}/{steps} loss={current:.8f}"
            )
            if current < best_loss:
                best_loss = current
                best_step = step
                best_state = _cpu_state(model)
    if best_state is None:
        raise RuntimeError("V8 matching produced no checkpoint.")
    model.load_state_dict(best_state, strict=True)
    uses_sam = BRANCHES[label].architecture in {
        "sam_transport", "sam_geometry_transport"
    } and variant != "sam_off"
    if uses_sam and maximum_grads["sam_projection_grad_norm"] <= 0.0:
        raise RuntimeError(f"V8 fold={fold} branch={label} SAM gradient is zero.")
    training = {
        "checkpoint_signature": signature,
        "match_steps": steps,
        "initial_match_loss": initial_loss,
        "step20_match_loss": step20_loss,
        "best_match_loss": best_loss,
        "best_match_step": best_step,
        "matching_seconds": time.perf_counter() - started,
        **maximum_grads,
    }
    _save_checkpoint(model, path=path, signature=signature, training=training)
    return training


def _train_or_resume_pose(
    *, model, label, fold, output_dir, resume, batch, baseline, target,
    reference_index, indices, steps, experiment, signature_context, seed,
):
    signature = _stage_signature(
        stage="pose", label=label, fold=fold, steps=steps, seed=seed,
        context=signature_context, matching=None,
    )
    path = output_dir / "checkpoints" / fold / f"{label}_pose.pt"
    resumed = _maybe_resume(model, path=path, signature=signature, resume=resume)
    if resumed is not None:
        return resumed
    parameters = configure_v80_training_stage(model, "pose")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(experiment.training.learning_rate),
        weight_decay=float(experiment.training.weight_decay),
    )
    with torch.no_grad():
        initial_output, _ = _forward_v80(
            model, batch=batch, baseline=baseline,
            reference_index=reference_index, variant=BRANCHES[label].train_variant,
        )
        initial_loss = float(
            _pose_loss(
                initial_output["world_to_camera"], target,
                reference_index=reference_index,
                translation_weight=experiment.training.translation_weight,
                evaluation_indices=indices,
            ).cpu()
        )
    best_loss = float("inf")
    best_step = 0
    best_state = None
    started = time.perf_counter()
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        output, _ = _forward_v80(
            model, batch=batch, baseline=baseline,
            reference_index=reference_index, variant=BRANCHES[label].train_variant,
        )
        loss = _pose_loss(
            output["world_to_camera"], target,
            reference_index=reference_index,
            translation_weight=experiment.training.translation_weight,
            evaluation_indices=indices,
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"Non-finite V8 pose loss at step={step}.")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            parameters, float(experiment.training.grad_clip_norm)
        )
        if not bool(torch.isfinite(norm)):
            raise RuntimeError(f"Non-finite V8 pose gradient at step={step}.")
        optimizer.step()
        if step % int(experiment.training.log_every) == 0 or step == steps:
            with torch.no_grad():
                checked, _ = _forward_v80(
                    model, batch=batch, baseline=baseline,
                    reference_index=reference_index,
                    variant=BRANCHES[label].train_variant,
                )
                current = float(
                    _pose_loss(
                        checked["world_to_camera"], target,
                        reference_index=reference_index,
                        translation_weight=experiment.training.translation_weight,
                        evaluation_indices=indices,
                    ).cpu()
                )
            print(
                f"V8 pose fold={fold} branch={label} "
                f"step={step}/{steps} loss={current:.8f}"
            )
            if current < best_loss:
                best_loss = current
                best_step = step
                best_state = _cpu_state(model)
    if best_state is None:
        raise RuntimeError("V8 pose training produced no checkpoint.")
    model.load_state_dict(best_state, strict=True)
    training = {
        "checkpoint_signature": signature,
        "pose_steps": steps,
        "initial_pose_loss": initial_loss,
        "best_pose_loss": best_loss,
        "best_pose_step": best_step,
        "pose_seconds": time.perf_counter() - started,
    }
    _save_checkpoint(model, path=path, signature=signature, training=training)
    return training


def _result_row(
    *, fold, label, spec, model, batch, baseline, target, base_output,
    reference_index, train_indices, pose_train_indices, test_indices,
    experiment, matching, match_training, pose_training, token_count, seed,
):
    train_match = _evaluate_matching(
        model, batch=batch, baseline=baseline, reference_index=reference_index,
        indices=train_indices, matching=matching, variant=spec.train_variant,
    )
    test_match = _evaluate_matching(
        model, batch=batch, baseline=baseline, reference_index=reference_index,
        indices=test_indices, matching=matching, variant=spec.train_variant,
    )
    with torch.no_grad():
        normal, _ = _forward_v80(
            model, batch=batch, baseline=baseline,
            reference_index=reference_index, variant=spec.train_variant,
        )
    train_pose = _pose_metrics(
        normal["world_to_camera"], target, reference_index=reference_index,
        translation_weight=experiment.training.translation_weight,
        evaluation_indices=pose_train_indices,
    )
    test_pose = _pose_metrics(
        normal["world_to_camera"], target, reference_index=reference_index,
        translation_weight=experiment.training.translation_weight,
        evaluation_indices=test_indices,
    )
    l0_pose = _pose_metrics(
        base_output["world_to_camera"], target, reference_index=reference_index,
        translation_weight=experiment.training.translation_weight,
        evaluation_indices=test_indices,
    )
    variant_metrics = {}
    for variant in EVALUATION_VARIANTS:
        current_match = _evaluate_matching(
            model, batch=batch, baseline=baseline,
            reference_index=reference_index, indices=test_indices,
            matching=matching, variant=variant,
        )["metrics"]
        with torch.no_grad():
            current_pose, _ = _forward_v80(
                model, batch=batch, baseline=baseline,
                reference_index=reference_index, variant=variant,
            )
        pose = _pose_metrics(
            current_pose["world_to_camera"], target,
            reference_index=reference_index,
            translation_weight=experiment.training.translation_weight,
            evaluation_indices=test_indices,
        )
        variant_metrics[variant] = (current_match, pose)
    inactive = ~normal["active_frames"]
    fallback_exact = torch.equal(
        normal["world_to_camera"][inactive],
        base_output["world_to_camera"][inactive],
    )
    row = {
        "fold": fold.name,
        "architecture": label,
        "underlying_architecture": spec.architecture,
        "value_mode": spec.value_mode,
        "pose_input_mode": "evidence_only",
        "train_input": spec.train_variant,
        "token_count": token_count,
        "seed": seed,
        "train_frames": _frame_text(fold.train_frames),
        "test_frames": _frame_text(fold.test_frames),
        "match_steps": match_training.get("match_steps", ""),
        "best_match_step": match_training["best_match_step"],
        "initial_match_loss": _short(match_training["initial_match_loss"]),
        "step20_match_loss": _short(match_training["step20_match_loss"]),
        "best_train_match_loss": _short(match_training["best_match_loss"]),
        "train_match_loss": _short(train_match["metrics"]["match_loss"]),
        "train_match_coverage": _short(train_match["metrics"]["match_coverage"]),
        "train_top1_within_radius": _short(train_match["metrics"]["top1_within_radius"]),
        "test_match_loss": _short(test_match["metrics"]["match_loss"]),
        "test_match_coverage": _short(test_match["metrics"]["match_coverage"]),
        "test_supervised_queries": int(test_match["metrics"]["supervised_queries"]),
        "test_top1_exact_accuracy": _short(test_match["metrics"]["top1_exact_accuracy"]),
        "test_top1_within_radius": _short(test_match["metrics"]["top1_within_radius"]),
        "test_expected_gt_distance": _short(test_match["metrics"]["expected_gt_distance"]),
        "test_valid_key_probability_mass": _short(test_match["metrics"]["valid_key_probability_mass"]),
        "geometry_logit_rms": _short(test_match["health"]["geometry_logit_rms"]),
        "sam_logit_rms": _short(test_match["health"]["sam_logit_rms"]),
        "sam_to_geometry_logit_ratio": _short(test_match["health"]["sam_to_geometry_logit_ratio"]),
        "transport_entropy": _short(test_match["health"]["transport_entropy"]),
        "sam_affinity_delta": _short(test_match["health"]["sam_affinity_delta"]),
        "geometry_projection_grad_norm": _short(match_training["geometry_projection_grad_norm"]),
        "sam_projection_grad_norm": _short(match_training["sam_projection_grad_norm"]),
        "sam_to_geometry_grad_ratio": _short(match_training["sam_to_geometry_grad_ratio"]),
        "pose_steps": pose_training.get("pose_steps", ""),
        "pose_train_frames": _indices_to_frames(pose_train_indices),
        "initial_pose_loss": _short(pose_training["initial_pose_loss"]),
        "best_train_pose_loss": _short(pose_training["best_pose_loss"]),
        "train_pose_loss": _short(train_pose["loss"]),
        "test_rotation_deg": _short(test_pose["rotation_degrees"]),
        "test_translation": _short(test_pose["translation_native"]),
        "test_pose_loss": _short(test_pose["loss"]),
        "frozen_l0_test_pose_loss": _short(l0_pose["loss"]),
        "pose_gain_vs_frozen_l0_percent": _short(_gain(l0_pose["loss"], test_pose["loss"])),
        "sam_off_test_match_loss": _short(variant_metrics["sam_off"][0]["match_loss"]),
        "wrong_id_test_match_loss": _short(variant_metrics["wrong_sam_identity"][0]["match_loss"]),
        "shuffle_time_test_match_loss": _short(variant_metrics["shuffle_sam_time"][0]["match_loss"]),
        "sam_off_test_top1": _short(variant_metrics["sam_off"][0]["top1_within_radius"]),
        "wrong_id_test_top1": _short(variant_metrics["wrong_sam_identity"][0]["top1_within_radius"]),
        "shuffle_time_test_top1": _short(variant_metrics["shuffle_sam_time"][0]["top1_within_radius"]),
        "sam_off_test_pose_loss": _short(variant_metrics["sam_off"][1]["loss"]),
        "wrong_id_test_pose_loss": _short(variant_metrics["wrong_sam_identity"][1]["loss"]),
        "shuffle_time_test_pose_loss": _short(variant_metrics["shuffle_sam_time"][1]["loss"]),
        "reference_exact": int(torch.equal(normal["world_to_camera"][:, reference_index], base_output["world_to_camera"][:, reference_index])),
        "inactive_fallback_exact": int(fallback_exact),
        "parameters": sum(p.numel() for p in model.parameters()),
        "matching_seconds": _short(match_training["matching_seconds"]),
        "pose_seconds": _short(pose_training["pose_seconds"]),
        "geometry_control_match_loss": "",
        "sam_off_control_match_loss": "",
        "geometry_control_pose_loss": "",
        "sam_off_control_pose_loss": "",
        "match_gain_vs_geometry_percent": "",
        "pose_gain_vs_geometry_percent": "",
        "match_beats_geometry": 0,
        "pose_beats_geometry": 0,
        "sam_perturbations_hurt_matching": 0,
        "fold_sam_causal_pass": 0,
        "all_folds_sam_causal_pass": 0,
        "dual_value_gain_vs_geometry_value_percent": "",
        "peak_gpu_memory_gib": _short(torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0),
    }
    return row


def _annotate_decisions(rows):
    all_passes = []
    for fold in FOLDS:
        group = {row["architecture"]: row for row in rows if row["fold"] == fold.name}
        required = {PRIMARY, GEOMETRY_CONTROL, SAM_OFF_CONTROL}
        missing = required - set(group)
        if missing:
            raise ValueError(f"V8 fold={fold.name} lacks controls={sorted(missing)}.")
        geometry_match = float(group[GEOMETRY_CONTROL]["test_match_loss"])
        geometry_pose = float(group[GEOMETRY_CONTROL]["test_pose_loss"])
        off_match = float(group[SAM_OFF_CONTROL]["test_match_loss"])
        off_pose = float(group[SAM_OFF_CONTROL]["test_pose_loss"])
        for row in group.values():
            row["geometry_control_match_loss"] = _short(geometry_match)
            row["sam_off_control_match_loss"] = _short(off_match)
            row["geometry_control_pose_loss"] = _short(geometry_pose)
            row["sam_off_control_pose_loss"] = _short(off_pose)
            row["match_gain_vs_geometry_percent"] = _short(_gain(geometry_match, float(row["test_match_loss"])))
            row["pose_gain_vs_geometry_percent"] = _short(_gain(geometry_pose, float(row["test_pose_loss"])))
            row["match_beats_geometry"] = int(float(row["test_match_loss"]) < geometry_match)
            row["pose_beats_geometry"] = int(float(row["test_pose_loss"]) < geometry_pose)
        primary = group[PRIMARY]
        perturb = all(
            float(primary[name]) < float(primary["test_top1_within_radius"])
            for name in ("sam_off_test_top1", "wrong_id_test_top1", "shuffle_time_test_top1")
        )
        primary["sam_perturbations_hurt_matching"] = int(perturb)
        passed = int(
            int(primary["reference_exact"])
            and int(primary["inactive_fallback_exact"])
            and float(primary["test_match_loss"]) < min(geometry_match, off_match)
            and float(primary["test_pose_loss"]) < min(geometry_pose, off_pose)
            and perturb
        )
        primary["fold_sam_causal_pass"] = passed
        all_passes.append(passed)
        dual = group.get("sam_geometry_dual_value")
        if dual is not None:
            dual["dual_value_gain_vs_geometry_value_percent"] = _short(
                _gain(float(primary["test_pose_loss"]), float(dual["test_pose_loss"]))
            )
    all_passed = int(all(all_passes))
    for row in rows:
        if row["architecture"] == PRIMARY:
            row["all_folds_sam_causal_pass"] = all_passed


def _stage_signature(*, stage, label, fold, steps, seed, context, matching):
    payload = {
        "schema": 1, "stage": stage, "label": label, "fold": fold,
        "steps": steps, "seed": seed, "context": context,
        "matching": asdict(matching) if matching is not None else None,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _maybe_resume(model, *, path, signature, resume):
    if not resume or not path.is_file():
        return None
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("signature") != signature:
        print(f"V8 invalidating stale checkpoint={path}")
        return None
    model.load_state_dict(checkpoint["model"], strict=True)
    print(f"V8 resumed checkpoint={path}")
    return dict(checkpoint["training"])


def _save_checkpoint(model, *, path, signature, training):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"signature": signature, "model": _cpu_state(model), "training": training}, temporary)
    temporary.replace(path)


def _cpu_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _gain(initial, final):
    return 0.0 if abs(float(initial)) <= 1e-12 else 100.0 * (float(initial) - float(final)) / abs(float(initial))


def _short(value):
    return f"{float(value):.8g}"


def _frame_text(frames):
    return " ".join(str(value) for value in frames)


def _indices_to_frames(indices):
    return " ".join(str(EXPECTED_FRAMES[int(index)]) for index in indices)


def _write_csv(path, rows):
    if not rows:
        raise ValueError("Refusing to write an empty V8 CSV.")
    columns = list(rows[0])
    for row in rows:
        if list(row) != columns:
            raise ValueError("V8 CSV row schemas differ.")
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def load_v80_config(path) -> V80Config:
    path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf8")) or {}
    matching = raw.get("matching", {})
    training = raw.get("training", {})
    branches = tuple(raw.get("branches", BRANCHES))
    return V80Config(
        v73_config=Path(raw.get("v73_config", "streaming_couping/configs/v73_correspondence_ablation.yaml")).expanduser().resolve(),
        data_config=Path(raw.get("data_config", "streaming_couping/configs/v74_temporal_data.yaml")).expanduser().resolve(),
        output_dir=Path(raw.get("output_dir", "outputs/streaming_couping_v80_supervised_correspondence")).expanduser().resolve(),
        device=str(raw.get("device", "cuda:0")),
        token_count=int(raw.get("token_count", 8)),
        match_steps=int(training.get("match_steps", 1200)),
        pose_steps=int(training.get("pose_steps", 1200)),
        base_steps=int(training.get("base_steps", 1200)),
        seed=int(training.get("seed", 0)),
        branches=branches,
        matching=V80MatchingConfig(
            max_distance=float(matching.get("max_distance", 0.10)),
            temperature=float(matching.get("temperature", 0.025)),
            require_mutual_nearest=bool(matching.get("require_mutual_nearest", True)),
        ),
    )


def _validate_config(config):
    unknown = set(config.branches) - set(BRANCHES)
    required = {PRIMARY, GEOMETRY_CONTROL, SAM_OFF_CONTROL}
    if unknown or not required.issubset(config.branches):
        raise ValueError(
            f"V8 branches invalid: unknown={sorted(unknown)}, "
            f"missing={sorted(required - set(config.branches))}."
        )
    if min(config.token_count, config.match_steps, config.pose_steps, config.base_steps) < 1:
        raise ValueError("V8 token count and training steps must be positive.")
    config.matching.validate()


def _jsonable_config(config):
    value = asdict(config)
    return {key: str(item) if isinstance(item, Path) else item for key, item in value.items()}


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="streaming_couping/configs/v80_supervised_correspondence.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--device")
    parser.add_argument("--match-steps", type=int)
    parser.add_argument("--pose-steps", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
