#!/usr/bin/env python3
"""Build, fit and evaluate the retained dynamic-instance geometry baseline."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time

import torch

from streaming_couping.src.config import load_config
from streaming_couping.src.external_repos import maybe_add_repo_to_path
from streaming_couping.src.learned_pose.baseline_runtime import (
    BaselineRunConfig,
    camera_centers,
    forward_model,
    load_baseline_run_config,
    pose_metrics,
    prepare_cached_batch,
    seed_everything,
    slice_batch_prefix,
    train_pose_model,
)
from streaming_couping.src.learned_pose.cache import (
    build_feature_caches,
    cache_path,
    load_feature_cache,
)
from streaming_couping.src.learned_pose.config import (
    ClipConfig,
    LearnedPoseConfig,
    load_learned_pose_config,
)
from streaming_couping.src.learned_pose.dynamic_instance_baseline import (
    CameraPoseBaseline,
    DynamicInstanceGeometryRefiner,
)


FRAME_COLUMNS = (
    "sequence_index",
    "frame_index",
    "sam_tracks_discovered",
    "observed_instances",
    "eligible_instances",
    "mature_instances",
    "pose_active",
    "exact_l0_fallback",
    "raw_rotation_error_deg",
    "l0_rotation_error_deg",
    "refined_rotation_error_deg",
    "raw_center_error_native",
    "l0_center_error_native",
    "refined_center_error_native",
)


def main() -> None:
    args = _parse_args()
    data = load_learned_pose_config(args.config)
    run = load_baseline_run_config(args.config)
    if args.output_dir:
        run = replace(
            run, output_dir=Path(args.output_dir).expanduser().resolve()
        )
    if args.training_device:
        run = replace(run, training_device=args.training_device)
    if args.base_steps is not None or args.refiner_steps is not None:
        run = replace(
            run,
            optimizer=replace(
                run.optimizer,
                base_steps=(
                    int(args.base_steps)
                    if args.base_steps is not None
                    else run.optimizer.base_steps
                ),
                refiner_steps=(
                    int(args.refiner_steps)
                    if args.refiner_steps is not None
                    else run.optimizer.refiner_steps
                ),
            ),
        )
    if args.sam3_device or args.geometry_device or args.streamvggt_devices:
        data = replace(
            data,
            sam3_device=args.sam3_device or data.sam3_device,
            geometry_device=args.geometry_device or data.geometry_device,
            streamvggt_devices=(
                tuple(
                    value.strip()
                    for value in args.streamvggt_devices.split(",")
                    if value.strip()
                )
                if args.streamvggt_devices
                else data.streamvggt_devices
            ),
        )
    if args.rebuild_cache:
        data = replace(data, features=replace(data.features, rebuild=True))
    if args.stage in {"all", "cache"}:
        build_feature_caches(data)
    if args.stage in {"all", "fit"}:
        result = run_baseline(data, run, resume=not args.no_resume)
        print(f"dynamic-instance baseline result={result}")


def run_baseline(
    data: LearnedPoseConfig,
    run: BaselineRunConfig,
    *,
    resume: bool,
) -> Path:
    clip = _find_clip(data, run.clip_name)
    path = cache_path(data, clip)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing baseline cache {path}; run --stage cache or --stage all."
        )
    payload = load_feature_cache(path)
    _validate_payload(payload, clip=clip, run=run)
    recovery = load_config(data.recovery_config)
    maybe_add_repo_to_path(recovery.streamvggt_repo)
    from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri

    batch, raw_pose, target_pose = prepare_cached_batch(
        payload,
        pose_decoder=pose_encoding_to_extri_intri,
        device=run.training_device,
        local_point_count=run.local_point_count,
    )
    frames = tuple(int(value) for value in payload["frame_indices"])
    positions = {frame: index for index, frame in enumerate(frames)}
    reference = int(payload["reference_sequence_index"])
    if reference != 0:
        raise ValueError("The retained baseline currently requires reference index 0.")
    all_requested = (
        run.base_train_frames
        + run.geometry_train_frames
        + run.evaluation_frames
    )
    missing = set(all_requested) - set(frames)
    if missing:
        raise ValueError(f"Baseline cache lacks requested frames={sorted(missing)}.")

    run.output_dir.mkdir(parents=True, exist_ok=True)
    signature = _signature(run=run, payload=payload, cache_path_value=path)
    seed_everything(run.optimizer.seed)
    base = CameraPoseBaseline(
        camera_dim=int(batch["camera_hidden"].shape[-1]),
        config=run.model,
    ).to(run.training_device)
    base_path = run.output_dir / "camera_baseline.pt"
    base_training = _load_or_train_base(
        base,
        checkpoint=base_path,
        signature=signature,
        resume=resume,
        batch=batch,
        raw_pose=raw_pose,
        target_pose=target_pose,
        reference=reference,
        positions=positions,
        run=run,
    )
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    base.eval()
    with torch.no_grad():
        l0 = forward_model(
            base,
            batch=batch,
            baseline=raw_pose,
            reference_index=reference,
        )

    refiner = DynamicInstanceGeometryRefiner(
        base_model=base,
        geometry_dim=int(batch["local_features"].shape[-1]),
        config=run.model,
    ).to(run.training_device)
    geometry_last = positions[max(run.geometry_train_frames)]
    training_batch = slice_batch_prefix(batch, length=geometry_last + 1)
    training_raw = raw_pose[:, : geometry_last + 1]
    training_target = target_pose[:, : geometry_last + 1]
    with torch.no_grad():
        initial = forward_model(
            refiner,
            batch=training_batch,
            baseline=training_raw,
            reference_index=reference,
        )
    requested_indices = [positions[frame] for frame in run.geometry_train_frames]
    active_indices = [
        index
        for index in requested_indices
        if bool(initial["active_frames"][0, index].cpu())
    ]
    if not active_indices:
        raise RuntimeError(
            "No mature, static, geometry-valid instance exists on refiner "
            "training frames. Inspect dynamic_instance_diagnostics.csv."
        )
    refiner_path = run.output_dir / "geometry_pose_refiner.pt"
    refiner_training = _load_or_train_refiner(
        refiner,
        checkpoint=refiner_path,
        signature=signature,
        active_indices=active_indices,
        resume=resume,
        batch=training_batch,
        raw_pose=training_raw,
        target_pose=training_target,
        reference=reference,
        run=run,
    )
    with torch.no_grad():
        refined = forward_model(
            refiner,
            batch=batch,
            baseline=raw_pose,
            reference_index=reference,
        )
    _assert_causal_prefix_equivalence(
        refiner,
        full=refined,
        batch=batch,
        raw_pose=raw_pose,
        reference=reference,
    )
    result = _write_outputs(
        run=run,
        payload=payload,
        cache_path_value=path,
        signature=signature,
        raw_pose=raw_pose,
        l0_pose=l0["world_to_camera"],
        refined=refined,
        target_pose=target_pose,
        reference=reference,
        evaluation_indices=[positions[frame] for frame in run.evaluation_frames],
        base_training=base_training,
        refiner_training=refiner_training,
    )
    return result


def _load_or_train_base(
    model,
    *,
    checkpoint,
    signature,
    resume,
    batch,
    raw_pose,
    target_pose,
    reference,
    positions,
    run,
):
    expected = f"{signature}:camera"
    loaded = _load_checkpoint(checkpoint, expected, model) if resume else None
    if loaded is not None:
        print(f"resumed camera baseline: {checkpoint}")
        return loaded
    last = positions[max(run.base_train_frames)]
    training_batch = slice_batch_prefix(batch, length=last + 1)
    started = time.perf_counter()
    training = train_pose_model(
        model,
        batch=training_batch,
        baseline=raw_pose[:, : last + 1],
        target=target_pose[:, : last + 1],
        reference_index=reference,
        training_indices=[positions[frame] for frame in run.base_train_frames],
        steps=run.optimizer.base_steps,
        config=run.optimizer,
    )
    training["seconds"] = time.perf_counter() - started
    _save_checkpoint(checkpoint, expected, model, training)
    return training


def _load_or_train_refiner(
    model,
    *,
    checkpoint,
    signature,
    active_indices,
    resume,
    batch,
    raw_pose,
    target_pose,
    reference,
    run,
):
    expected = f"{signature}:geometry:{','.join(str(v) for v in active_indices)}"
    loaded = _load_checkpoint(checkpoint, expected, model) if resume else None
    if loaded is not None:
        print(f"resumed geometry pose refiner: {checkpoint}")
        return loaded
    started = time.perf_counter()
    training = train_pose_model(
        model,
        batch=batch,
        baseline=raw_pose,
        target=target_pose,
        reference_index=reference,
        training_indices=active_indices,
        steps=run.optimizer.refiner_steps,
        config=run.optimizer,
    )
    training["seconds"] = time.perf_counter() - started
    training["active_training_indices"] = tuple(active_indices)
    _save_checkpoint(checkpoint, expected, model, training)
    return training


def _load_checkpoint(path: Path, signature: str, model):
    if not path.is_file():
        return None
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("signature") != signature or "model" not in checkpoint:
        print(f"invalidating stale checkpoint: {path}")
        return None
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return dict(checkpoint["training"])


def _save_checkpoint(path: Path, signature: str, model, training: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "signature": signature,
            "model": {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            },
            "training": training,
        },
        temporary,
    )
    temporary.replace(path)


def _assert_causal_prefix_equivalence(
    model,
    *,
    full,
    batch,
    raw_pose,
    reference,
) -> None:
    model.eval()
    sequence = int(batch["camera_hidden"].shape[1])
    with torch.no_grad():
        for length in range(2, sequence + 1):
            prefix = forward_model(
                model,
                batch=slice_batch_prefix(batch, length=length),
                baseline=raw_pose[:, :length],
                reference_index=reference,
            )
            for field in ("world_to_camera", "active_frames", "usable_instance_mask"):
                expected = full[field][:, :length]
                equal = (
                    torch.equal(prefix[field], expected)
                    if expected.dtype == torch.bool
                    else torch.allclose(
                        prefix[field], expected, rtol=0.0, atol=2e-6
                    )
                )
                if not equal:
                    raise RuntimeError(
                        f"Causal prefix equivalence failed field={field} length={length}."
                    )


def _write_outputs(
    *,
    run,
    payload,
    cache_path_value,
    signature,
    raw_pose,
    l0_pose,
    refined,
    target_pose,
    reference,
    evaluation_indices,
    base_training,
    refiner_training,
) -> Path:
    frames = tuple(int(value) for value in payload["frame_indices"])
    dynamic = {
        int(row["sequence_index"]): row
        for row in payload.get("dynamic_instance_diagnostics", ())
    }
    rows = []
    for index, frame in enumerate(frames):
        raw_rotation, raw_center = _frame_errors(raw_pose, target_pose, index)
        l0_rotation, l0_center = _frame_errors(l0_pose, target_pose, index)
        refined_rotation, refined_center = _frame_errors(
            refined["world_to_camera"], target_pose, index
        )
        active = bool(refined["active_frames"][0, index].cpu())
        fallback = torch.equal(
            refined["world_to_camera"][:, index], l0_pose[:, index]
        )
        diag = dynamic.get(index, {})
        rows.append(
            {
                "sequence_index": index,
                "frame_index": frame,
                "sam_tracks_discovered": int(diag.get("discovered_tracks", 0)),
                "observed_instances": int(payload["observed"][index].sum()),
                "eligible_instances": int(
                    refined["eligible_instances"][0, index].sum().cpu()
                ),
                "mature_instances": int(
                    refined["usable_instance_mask"][0, index].sum().cpu()
                ),
                "pose_active": int(active),
                "exact_l0_fallback": int((not active) and fallback),
                "raw_rotation_error_deg": raw_rotation,
                "l0_rotation_error_deg": l0_rotation,
                "refined_rotation_error_deg": refined_rotation,
                "raw_center_error_native": raw_center,
                "l0_center_error_native": l0_center,
                "refined_center_error_native": refined_center,
            }
        )
    frame_path = run.output_dir / "frame_diagnostics.csv"
    _write_csv(frame_path, rows, FRAME_COLUMNS)
    dynamic_path = run.output_dir / "dynamic_instance_diagnostics.csv"
    dynamic_rows = list(payload.get("dynamic_instance_diagnostics", ()))
    if dynamic_rows:
        _write_csv(dynamic_path, dynamic_rows, tuple(dynamic_rows[0]))

    raw_metrics = pose_metrics(
        raw_pose,
        target_pose,
        reference_index=reference,
        translation_weight=run.optimizer.translation_weight,
        evaluation_indices=evaluation_indices,
    )
    l0_metrics = pose_metrics(
        l0_pose,
        target_pose,
        reference_index=reference,
        translation_weight=run.optimizer.translation_weight,
        evaluation_indices=evaluation_indices,
    )
    refined_metrics = pose_metrics(
        refined["world_to_camera"],
        target_pose,
        reference_index=reference,
        translation_weight=run.optimizer.translation_weight,
        evaluation_indices=evaluation_indices,
    )
    births = tuple(int(value) for value in payload.get("sam_birth_indices", ()))
    late_births = tuple(value for value in births if value > reference)
    evaluated_active = sum(
        bool(refined["active_frames"][0, index].cpu())
        for index in evaluation_indices
    )
    inactive_exact = all(
        torch.equal(refined["world_to_camera"][:, index], l0_pose[:, index])
        for index in evaluation_indices
        if not bool(refined["active_frames"][0, index].cpu())
    )
    correction_rows = [
        row
        for row in payload.get("segmentation_diagnostics", ())
        if str(row.get("selected_variant", "")) == "sam31_online_geometry_compete"
    ]
    summary = {
        "schema": 1,
        "baseline_version": run.version,
        "method": "v0_dynamic_instance_geometry_baseline",
        "claim_level": "empirical_baseline_not_sam_token_causality",
        "config": str(run.source_path),
        "cache": str(cache_path_value),
        "signature": signature,
        "clip": payload["clip_name"],
        "frames": frames,
        "evaluation_frames": tuple(frames[index] for index in evaluation_indices),
        "sam_role": "prompted_dynamic_discovery_mask_persistent_id",
        "pose_role": "streamvggt_geometry_transport_without_sam_appearance_tokens",
        "segmentation_variant": payload.get("sam_segmentation_variant"),
        "geometry_corrections_applied": sum(
            int(row.get("correction_applied", 0)) for row in correction_rows
        ),
        "geometry_correction_memory_writeback": False,
        "sam_appearance_cached": bool(payload.get("cache_sam_appearance", True)),
        "late_birth_count": len(late_births),
        "late_birth_sequence_indices": late_births,
        "evaluation_active_frames": evaluated_active,
        "evaluation_inactive_exact_l0": int(inactive_exact),
        "causal_prefix_check": "passed_atol_2e-6",
        "raw_metrics": raw_metrics,
        "camera_baseline_metrics": l0_metrics,
        "geometry_refined_metrics": refined_metrics,
        "camera_training": base_training,
        "geometry_training": refiner_training,
        "model": asdict(run.model),
        "optimizer": asdict(run.optimizer),
    }
    result = run.output_dir / "baseline_summary.json"
    result.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf8",
    )
    torch.save(
        {
            "frame_indices": frames,
            "raw_world_to_camera": raw_pose.detach().cpu(),
            "camera_baseline_world_to_camera": l0_pose.detach().cpu(),
            "refined_world_to_camera": refined["world_to_camera"].detach().cpu(),
            "active_frames": refined["active_frames"].detach().cpu(),
            "usable_instance_mask": refined["usable_instance_mask"].detach().cpu(),
        },
        run.output_dir / "poses.pt",
    )
    print("V0 DYNAMIC INSTANCE GEOMETRY BASELINE SUMMARY")
    print(result.read_text(encoding="utf8").rstrip())
    return result


def _frame_errors(predicted, target, index: int) -> tuple[float, float]:
    left = predicted[:, index]
    right = target[:, index]
    relative = left[..., :3, :3] @ right[..., :3, :3].transpose(-1, -2)
    cosine = (torch.diagonal(relative, dim1=-2, dim2=-1).sum(dim=-1) - 1) * 0.5
    rotation = torch.rad2deg(torch.acos(cosine.clamp(-1, 1))).mean()
    center = torch.linalg.vector_norm(
        camera_centers(left) - camera_centers(right), dim=-1
    ).mean()
    return float(rotation.cpu()), float(center.cpu())


def _signature(*, run, payload, cache_path_value: Path) -> str:
    identity = {
        "schema": 1,
        "purpose": "v0_dynamic_instance_geometry_baseline",
        "config": asdict(run),
        "cache": str(cache_path_value),
        "clip": payload.get("clip_name"),
        "frames": payload.get("frame_indices"),
        "sam_checkpoint": payload.get("sam_checkpoint"),
        "sam_track_ids": payload.get("sam_track_ids"),
        "sam_birth_indices": payload.get("sam_birth_indices"),
        "instance_birth_indices": payload.get("instance_birth_indices"),
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, default=str).encode("utf8")
    ).hexdigest()


def _validate_payload(payload: dict, *, clip: ClipConfig, run: BaselineRunConfig) -> None:
    if str(payload.get("instance_source")) != "sam31_online":
        raise ValueError("Baseline requires instance_source=sam31_online.")
    if str(payload.get("sam_version")) != "sam3.1":
        raise ValueError("Baseline cache is not SAM3.1.")
    if bool(payload.get("cache_sam_appearance", True)):
        raise ValueError(
            "Retained baseline cache must disable SAM appearance extraction."
        )
    if payload.get("appearance") is not None:
        raise ValueError("Geometry-only baseline cache unexpectedly contains appearance.")
    if str(payload.get("sam_segmentation_variant")) != "sam31_online_geometry_compete":
        raise ValueError("Baseline requires causal online geometry mask competition.")
    if tuple(int(value) for value in payload.get("instance_ids", ())) != clip.instance_ids:
        raise ValueError("Baseline cache slot layout differs from config.")
    if not payload.get("dynamic_instance_diagnostics"):
        raise ValueError("Baseline cache lacks dynamic instance diagnostics.")
    if not any(int(value) > 0 for value in payload.get("sam_birth_indices", ())):
        raise ValueError("This audit requires at least one late-born SAM track.")
    if run.local_point_count > int(payload["instance_uvd"].shape[2]):
        raise ValueError("Configured local point count exceeds cached support.")


def _find_clip(config: LearnedPoseConfig, name: str) -> ClipConfig:
    selected = [clip for clip in config.clips if clip.name == name]
    if len(selected) != 1:
        raise ValueError(f"Clip {name!r} was not found exactly once.")
    return selected[0]


def _write_csv(path: Path, rows: list[dict], columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in columns})


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="streaming_couping/configs/v0_baseline.yaml",
    )
    parser.add_argument("--stage", choices=("all", "cache", "fit"), default="all")
    parser.add_argument("--output-dir")
    parser.add_argument("--sam3-device")
    parser.add_argument("--geometry-device")
    parser.add_argument("--streamvggt-devices")
    parser.add_argument("--training-device")
    parser.add_argument("--base-steps", type=int)
    parser.add_argument("--refiner-steps", type=int)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
