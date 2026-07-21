"""Training and evaluation pipeline for persistent-instance pose adapters."""

from __future__ import annotations

from contextlib import nullcontext
import csv
import json
from pathlib import Path
import random
from typing import Iterable, Mapping

import numpy as np
import torch

from vggtsam.adapters.streamvggt_latent import load_streamvggt_latent_model

from ..config import load_config
from ..pose_evaluation import (
    _all_pair_pose_metrics,
    _evaluate_pose_alignment,
    _prepare_pose_sequence,
    _reference_pose_alignment,
    _summarize_pose_pairs,
)
from .cache import cache_path, load_feature_cache
from .config import LearnedPoseConfig
from .losses import compute_training_loss, instance_rigid_losses
from .model import InstancePoseAdapter


def train_all_modes(config: LearnedPoseConfig) -> None:
    train_paths = [
        cache_path(config, clip)
        for clip in config.clips
        if clip.split.lower() == "train"
    ]
    validation_paths = [
        cache_path(config, clip)
        for clip in config.clips
        if clip.split.lower() in {"val", "validation"}
    ]
    if not train_paths:
        raise ValueError("At least one dataset clip must use split: train.")
    first = load_feature_cache(train_paths[0])
    frozen = _load_frozen_streamvggt(config, device=config.training.device)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    for mode in config.training.modes:
        if mode == "all_token_fusion" and "token_levels" not in first:
            raise RuntimeError(
                "all_token_fusion requires features.cache_all_token_levels=true."
            )
        print(f"training learned pose mode={mode}")
        _seed_everything(config.training.seed)
        adapter = _new_adapter(first, config).to(config.training.device)
        mode_dir = config.output_dir / "checkpoints" / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        equivalence = _assert_zero_initialization(
            adapter,
            frozen,
            first,
            config,
            mode=mode,
        )
        _write_csv(mode_dir / "zero_init_equivalence.csv", [equivalence])
        optimizer = torch.optim.AdamW(
            adapter.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        history = []
        best_loss = float("inf")
        payload_cache = {
            path: _payload_for_mode(load_feature_cache(path), mode)
            for path in set(train_paths + validation_paths)
        }
        for epoch in range(config.training.epochs):
            adapter.train()
            epoch_rows = []
            schedule = list(train_paths) * config.training.repeats_per_epoch
            random.Random(config.training.seed + epoch).shuffle(schedule)
            for step, path in enumerate(schedule):
                batch = _batch_to_device(
                    payload_cache[path],
                    config.training.device,
                )
                optimizer.zero_grad(set_to_none=True)
                with _autocast_context(config):
                    outputs = _forward_mode(
                        frozen,
                        adapter,
                        batch,
                        mode=mode,
                        perturbation="aligned",
                    )
                    total, terms = compute_training_loss(
                        outputs,
                        batch,
                        config.loss,
                        mode=mode,
                    )
                if not bool(torch.isfinite(total)):
                    raise RuntimeError(
                        f"Non-finite training loss mode={mode} epoch={epoch} step={step}."
                    )
                total.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    adapter.parameters(),
                    config.training.grad_clip_norm,
                )
                if not bool(torch.isfinite(grad_norm)):
                    raise RuntimeError(
                        f"Non-finite gradient mode={mode} epoch={epoch} step={step}."
                    )
                optimizer.step()
                row = {
                    "mode": mode,
                    "epoch": epoch,
                    "step": step,
                    "clip": batch["clip_name"],
                    "gradient_norm": float(grad_norm),
                    **{name: float(value.detach().cpu()) for name, value in terms.items()},
                    **_scalar_logs(outputs.get("diagnostics", {})),
                }
                epoch_rows.append(row)
                history.append(row)
                print(
                    f"mode={mode} epoch={epoch:03d} step={step:03d} "
                    f"loss={row['total']:.6f} pose={row['camera']:.6f} "
                    f"rigid={row['rigid']:.6f} grad={row['gradient_norm']:.4f}"
                )
            validation_loss = _validation_loss(
                frozen,
                adapter,
                validation_paths or train_paths,
                config,
                mode=mode,
                payload_cache=payload_cache,
            )
            train_loss = float(np.mean([row["total"] for row in epoch_rows]))
            checkpoint = {
                "mode": mode,
                "epoch": epoch,
                "adapter": adapter.state_dict(),
                "adapter_metadata": adapter.metadata(),
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "config_path": str(config.source_path),
            }
            torch.save(checkpoint, mode_dir / "checkpoint_last.pt")
            if validation_loss < best_loss:
                best_loss = validation_loss
                torch.save(checkpoint, mode_dir / "checkpoint_best.pt")
            _write_csv(mode_dir / "training_history.csv", history)
        print(f"finished mode={mode} best_validation_loss={best_loss:.6f}")


@torch.no_grad()
def evaluate_all_modes(config: LearnedPoseConfig) -> None:
    evaluation_clips = [
        clip
        for clip in config.clips
        if clip.split.lower() in {"val", "validation", "test"}
    ]
    if not evaluation_clips:
        evaluation_clips = list(config.clips)
        print(
            "warning: no val/test clips configured; evaluating training clips only. "
            "These numbers are diagnostics, not generalization evidence."
        )
    first = load_feature_cache(cache_path(config, evaluation_clips[0]))
    frozen = _load_frozen_streamvggt(config, device=config.training.device)
    summary_rows = []
    frame_rows = []
    rpe_rows = []
    pair_rows = []
    pair_summary_rows = []
    equivalence_rows = []
    diagnostic_rows = []

    # Raw StreamVGGT is evaluated once per clip through the exact same batch
    # CameraHead path used by every adapter.
    for clip in evaluation_clips:
        payload = load_feature_cache(cache_path(config, clip))
        batch = _batch_to_device(payload, config.training.device)
        baseline_outputs = _forward_baseline(frozen, batch)
        cached_pose_difference = float(
            (
                baseline_outputs["pose_encoding"]
                - batch["baseline_pose_encoding"]
            )
            .abs()
            .max()
            .cpu()
        )
        cached_pose_equal = cached_pose_difference == 0.0
        equivalence_rows.append(
            {
                "clip": clip.name,
                "mode": "baseline_streaming_redecode",
                "token_max_abs_diff": 0.0,
                "pose_max_abs_diff": cached_pose_difference,
                "depth_max_abs_diff": 0.0,
                "pointmap_max_abs_diff": 0.0,
                "strict_equal": int(cached_pose_equal),
            }
        )
        if config.evaluation.strict_equivalence and not cached_pose_equal:
            raise RuntimeError(
                "Cached camera hidden tokens do not reproduce the original "
                f"streaming StreamVGGT pose for clip={clip.name}: "
                f"max_abs_diff={cached_pose_difference}."
            )
        _append_pose_metrics(
            summary_rows,
            frame_rows,
            rpe_rows,
            pair_rows,
            pair_summary_rows,
            payload=payload,
            pose_encoding=baseline_outputs["pose_encoding"],
            mode="baseline",
            perturbation="module_off",
        )

    for mode in config.training.modes:
        checkpoint_path = config.output_dir / "checkpoints" / mode / "checkpoint_best.pt"
        if not checkpoint_path.exists():
            print(f"skipping missing checkpoint: {checkpoint_path}")
            continue
        checkpoint = _torch_load(checkpoint_path)
        adapter = _new_adapter(first, config).to(config.training.device)
        adapter.load_state_dict(checkpoint["adapter"], strict=True)
        adapter.eval()
        for clip in evaluation_clips:
            payload = load_feature_cache(cache_path(config, clip))
            batch = _batch_to_device(payload, config.training.device)
            baseline_outputs = _forward_baseline(frozen, batch)
            perturbations = config.evaluation.perturbations
            for perturbation in perturbations:
                outputs = _forward_mode(
                    frozen,
                    adapter,
                    batch,
                    mode=mode,
                    perturbation=perturbation,
                )
                _append_pose_metrics(
                    summary_rows,
                    frame_rows,
                    rpe_rows,
                    pair_rows,
                    pair_summary_rows,
                    payload=payload,
                    pose_encoding=outputs["pose_encoding"],
                    mode=mode,
                    perturbation=perturbation,
                )
                rigid, centroid = instance_rigid_losses(
                    outputs["pose_encoding"].float(),
                    batch["instance_uvd"].float(),
                    batch["instance_uvd_valid"].bool(),
                    batch["instance_rigid_weight"].float(),
                    image_size=tuple(int(v) for v in batch["image_size"]),
                    scene_scale=float(batch["scene_scale"]),
                    trim_quantile=config.loss.rigid_trim_quantile,
                )
                rigid_meters, centroid_meters = instance_rigid_losses(
                    outputs["pose_encoding"].float(),
                    batch["instance_uvd"].float(),
                    batch["instance_uvd_valid"].bool(),
                    batch["instance_rigid_weight"].float(),
                    image_size=tuple(int(v) for v in batch["image_size"]),
                    scene_scale=(
                        1.0 / max(float(batch["point_alignment_scale"]), 1e-8)
                    ),
                    trim_quantile=config.loss.rigid_trim_quantile,
                )
                diagnostic_rows.append(
                    {
                        "clip": clip.name,
                        "scene_id": clip.scene_id,
                        "mode": mode,
                        "perturbation": perturbation,
                        "instance_chamfer_normalized": float(rigid.cpu()),
                        "matched_centroid_normalized": float(centroid.cpu()),
                        "instance_chamfer_aligned_meters": float(rigid_meters.cpu()),
                        "matched_centroid_aligned_meters": float(centroid_meters.cpu()),
                        **_scalar_logs(outputs.get("diagnostics", {})),
                    }
                )
                if perturbation == "module_off":
                    token_difference = float(
                        (outputs["camera_hidden"] - batch["camera_hidden"]).abs().max().cpu()
                    )
                    pose_difference = float(
                        (outputs["pose_encoding"] - baseline_outputs["pose_encoding"]).abs().max().cpu()
                    )
                    depth_difference = 0.0
                    pointmap_difference = 0.0
                    if mode == "all_token_fusion":
                        depth_difference = float(
                            (outputs["depth"] - batch["baseline_depth"]).abs().max().cpu()
                        )
                        pointmap_difference = float(
                            (
                                outputs["world_points"]
                                - batch["baseline_world_points"]
                            )
                            .abs()
                            .max()
                            .cpu()
                        )
                    passed = (
                        token_difference == 0.0
                        and pose_difference == 0.0
                        and depth_difference == 0.0
                        and pointmap_difference == 0.0
                    )
                    equivalence_rows.append(
                        {
                            "clip": clip.name,
                            "mode": mode,
                            "token_max_abs_diff": token_difference,
                            "pose_max_abs_diff": pose_difference,
                            "depth_max_abs_diff": depth_difference,
                            "pointmap_max_abs_diff": pointmap_difference,
                            "strict_equal": int(passed),
                        }
                    )
                    if config.evaluation.strict_equivalence and not passed:
                        raise RuntimeError(
                            f"Module-off equivalence failed mode={mode} clip={clip.name}: "
                            f"token={token_difference} pose={pose_difference} "
                            f"depth={depth_difference} pointmap={pointmap_difference}."
                        )

    output = config.output_dir / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "pose_summary.csv", summary_rows)
    _write_csv(output / "pose_frame_metrics.csv", frame_rows)
    _write_csv(output / "pose_rpe.csv", rpe_rows)
    _write_csv(output / "pose_pair_metrics.csv", pair_rows)
    _write_csv(output / "pose_pair_summary.csv", pair_summary_rows)
    _write_csv(output / "instance_diagnostics.csv", diagnostic_rows)
    _write_csv(output / "baseline_equivalence.csv", equivalence_rows)
    metadata = {
        "config": str(config.source_path),
        "causal_control": (
            "All zero/shuffle perturbations are inference-time tests of one aligned checkpoint; "
            "they are not separately trained shuffled models."
        ),
        "sam_role": (
            "SAM3 supplies recovered masks, persistent IDs, scores, and frozen pooled appearance; "
            "no fused token is written into SAM3."
        ),
        "gt_role": "pose/depth/pointmap training supervision and evaluation only",
    }
    with (output / "metadata.json").open("w", encoding="utf8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    print(f"learned-pose evaluation written to {output}")


def _load_frozen_streamvggt(config: LearnedPoseConfig, *, device: str):
    recovery = load_config(config.recovery_config)
    model = load_streamvggt_latent_model(
        repo_path=recovery.streamvggt_repo,
        checkpoint_path=recovery.streamvggt_checkpoint,
        device=device,
        strict=True,
    )
    model.requires_grad_(False)
    model.eval()
    return model


def _new_adapter(payload: dict, config: LearnedPoseConfig) -> InstancePoseAdapter:
    return InstancePoseAdapter(
        appearance_dim=int(payload["appearance_dim"]),
        geometry_dim=int(payload["geometry_dim"]),
        token_dim=int(payload["camera_hidden"].shape[-1]),
        config=config.fusion,
    )


def _forward_mode(
    frozen,
    adapter: InstancePoseAdapter,
    batch: dict,
    *,
    mode: str,
    perturbation: str,
) -> dict[str, torch.Tensor | dict]:
    if mode == "all_token_fusion":
        layer_indices = [int(v) for v in batch["dpt_layer_indices"]]
        level_tensor = batch["token_levels"]
        token_levels = {
            layer: level_tensor[:, index]
            for index, layer in enumerate(layer_indices)
        }
        updated, diagnostics = adapter.forward_all_tokens(
            token_levels,
            appearance=batch["appearance"],
            geometry=batch["geometry"],
            quality=batch["quality"],
            observed=batch["observed"],
            perturbation=perturbation,
        )
        final_layer = max(layer_indices)
        camera_hidden = updated[final_layer][:, :, 0]
        pose_encoding = _decode_streaming_camera_head(
            frozen.camera_head,
            updated[final_layer][:, :, 0],
        )
        aggregated = [None] * (final_layer + 1)
        for layer, value in updated.items():
            aggregated[int(layer)] = value
        depth, depth_confidence = _decode_streaming_dpt_head(
            frozen.depth_head,
            aggregated,
            batch["stream_images"],
            patch_start_idx=int(batch["patch_start_idx"]),
        )
        world_points, world_confidence = _decode_streaming_dpt_head(
            frozen.point_head,
            aggregated,
            batch["stream_images"],
            patch_start_idx=int(batch["patch_start_idx"]),
        )
        residual_square_values = [
            value
            for key, value in diagnostics.items()
            if key.endswith("residual_mean_square")
        ]
        residual_mean_square = (
            torch.stack(residual_square_values).mean()
            if residual_square_values
            else pose_encoding.new_zeros(())
        )
        return {
            "pose_encoding": pose_encoding,
            "camera_hidden": camera_hidden,
            "depth": depth,
            "depth_confidence": depth_confidence,
            "world_points": world_points,
            "world_confidence": world_confidence,
            "residual_mean_square": residual_mean_square,
            "diagnostics": diagnostics,
        }
    camera_hidden, diagnostics = adapter.forward_camera(
        batch["camera_hidden"],
        appearance=batch["appearance"],
        geometry=batch["geometry"],
        quality=batch["quality"],
        observed=batch["observed"],
        mode=mode,
        perturbation=perturbation,
    )
    pose_encoding = _decode_streaming_camera_head(
        frozen.camera_head,
        camera_hidden,
    )
    return {
        "pose_encoding": pose_encoding,
        "camera_hidden": camera_hidden,
        "residual_mean_square": diagnostics["residual_mean_square"],
        "diagnostics": diagnostics,
    }


def _forward_baseline(frozen, batch: dict) -> dict[str, torch.Tensor]:
    pose = _decode_streaming_camera_head(
        frozen.camera_head,
        batch["camera_hidden"],
    )
    return {"pose_encoding": pose, "camera_hidden": batch["camera_hidden"]}


def _decode_streaming_camera_head(
    camera_head,
    camera_hidden: torch.Tensor,
) -> torch.Tensor:
    """Replay the CameraHead path used by ``StreamVGGTLatentAdapter``.

    The active recovery configuration runs StreamVGGT frame by frame with a
    CameraHead KV cache.  Decoding all cached camera tokens in one non-cached
    call is not the same computation, so every train/eval branch must replay
    this causal path.
    """

    if camera_hidden.ndim != 3:
        raise ValueError("camera_hidden must have shape [B,S,D].")
    past_key_values = [None] * int(camera_head.trunk_depth)
    pose_rows = []
    for frame_index in range(camera_hidden.shape[1]):
        frame_tokens = camera_hidden[:, frame_index : frame_index + 1].unsqueeze(2)
        pose_encodings, past_key_values = camera_head(
            [frame_tokens],
            past_key_values_camera=past_key_values,
            use_cache=True,
        )
        pose_rows.append(pose_encodings[-1])
    return torch.cat(pose_rows, dim=1)


def _decode_streaming_dpt_head(
    head,
    token_levels: list[torch.Tensor | None],
    images: torch.Tensor,
    *,
    patch_start_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replay the frame-wise DPT calls used by streaming extraction."""

    if images.ndim != 5:
        raise ValueError("stream_images must have shape [B,S,3,H,W].")
    predictions = []
    confidences = []
    for frame_index in range(images.shape[1]):
        current_levels = [
            (
                value[:, frame_index : frame_index + 1]
                if torch.is_tensor(value)
                else None
            )
            for value in token_levels
        ]
        prediction, confidence = head(
            current_levels,
            images=images[:, frame_index : frame_index + 1],
            patch_start_idx=int(patch_start_idx),
        )
        predictions.append(prediction)
        confidences.append(confidence)
    return torch.cat(predictions, dim=1), torch.cat(confidences, dim=1)


@torch.no_grad()
def _validation_loss(
    frozen,
    adapter,
    paths: Iterable[Path],
    config: LearnedPoseConfig,
    *,
    mode: str,
    payload_cache: Mapping[Path, dict],
) -> float:
    adapter.eval()
    values = []
    for path in paths:
        batch = _batch_to_device(payload_cache[path], config.training.device)
        outputs = _forward_mode(
            frozen,
            adapter,
            batch,
            mode=mode,
            perturbation="aligned",
        )
        total, _ = compute_training_loss(outputs, batch, config.loss, mode=mode)
        if not bool(torch.isfinite(total)):
            raise RuntimeError(
                f"Non-finite validation loss mode={mode} path={path}."
            )
        values.append(float(total.cpu()))
    return float(np.mean(values))


@torch.no_grad()
def _assert_zero_initialization(
    adapter,
    frozen,
    payload: dict,
    config: LearnedPoseConfig,
    *,
    mode: str,
) -> dict:
    batch = _batch_to_device(payload, config.training.device)
    baseline = _forward_baseline(frozen, batch)
    output = _forward_mode(
        frozen,
        adapter,
        batch,
        mode=mode,
        perturbation="aligned",
    )
    token_difference = float((output["camera_hidden"] - baseline["camera_hidden"]).abs().max().cpu())
    pose_difference = float((output["pose_encoding"] - baseline["pose_encoding"]).abs().max().cpu())
    cached_pose_difference = float(
        (baseline["pose_encoding"] - batch["baseline_pose_encoding"])
        .abs()
        .max()
        .cpu()
    )
    if (
        token_difference != 0.0
        or pose_difference != 0.0
        or cached_pose_difference != 0.0
    ):
        raise RuntimeError(
            "Zero-initialized adapter does not exactly reproduce baseline: "
            f"token={token_difference}, pose={pose_difference}, "
            f"cached_pose={cached_pose_difference}."
        )
    print("zero-init baseline equivalence: exact")
    return {
        "mode": mode,
        "clip": payload["clip_name"],
        "token_max_abs_diff": token_difference,
        "pose_max_abs_diff": pose_difference,
        "cached_pose_max_abs_diff": cached_pose_difference,
        "strict_equal": 1,
    }


def _append_pose_metrics(
    summary_rows: list[dict],
    frame_rows: list[dict],
    rpe_rows: list[dict],
    pair_rows: list[dict],
    pair_summary_rows: list[dict],
    *,
    payload: dict,
    pose_encoding: torch.Tensor,
    mode: str,
    perturbation: str,
) -> None:
    from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri

    predicted_w2c, _ = pose_encoding_to_extri_intri(
        pose_encoding.float(),
        image_size_hw=tuple(payload["image_size"]),
    )
    predicted = _prepare_pose_sequence(
        predicted_w2c[0].detach().double().cpu(),
        frame_indices=payload["frame_indices"],
        source=f"{mode}:{perturbation}",
    )
    target = _prepare_pose_sequence(
        payload["target_world_to_camera"].double().cpu(),
        frame_indices=payload["frame_indices"],
        source="scannetpp_colmap",
    )
    alignment = _reference_pose_alignment(
        predicted,
        target,
        reference_index=int(payload["reference_sequence_index"]),
        scale=float(payload["point_alignment_scale"]),
    )
    summary, current_frames, current_rpe = _evaluate_pose_alignment(
        alignment,
        predicted=predicted,
        target=target,
        frame_indices=payload["frame_indices"],
        reference_index=int(payload["reference_sequence_index"]),
    )
    current_pairs = _all_pair_pose_metrics(
        predicted,
        target,
        frame_indices=payload["frame_indices"],
    )
    current_pair_summary = _summarize_pose_pairs(current_pairs)
    prefix = {
        "clip": payload["clip_name"],
        "scene_id": payload["scene_id"],
        "mode": mode,
        "perturbation": perturbation,
    }
    summary_rows.append({**prefix, **summary})
    frame_rows.extend({**prefix, **row} for row in current_frames)
    rpe_rows.extend({**prefix, **row} for row in current_rpe)
    pair_rows.extend({**prefix, **row} for row in current_pairs)
    pair_summary_rows.extend({**prefix, **row} for row in current_pair_summary)


def _batch_to_device(payload: dict, device: str) -> dict:
    batched_fields = {
        "camera_hidden",
        "appearance",
        "geometry",
        "quality",
        "observed",
        "target_pose_encoding",
        "instance_uvd",
        "instance_uvd_valid",
        "instance_rigid_weight",
        "target_world_points",
        "target_depth",
        "baseline_pose_encoding",
        "baseline_depth",
        "baseline_world_points",
        "stream_images",
    }
    output = dict(payload)
    for name in batched_fields:
        value = payload.get(name)
        if torch.is_tensor(value):
            output[name] = value.unsqueeze(0).to(device=device)
    if torch.is_tensor(payload.get("token_levels")):
        output["token_levels"] = payload["token_levels"].unsqueeze(0).to(device=device)
    for name in (
        "point_alignment_rotation",
        "point_alignment_translation",
    ):
        if torch.is_tensor(payload.get(name)):
            output[name] = payload[name].to(device=device)
    # Frozen head inputs are cached and replayed in fp32 so module-off can be
    # checked against the original StreamVGGT outputs exactly.
    for name in ("camera_hidden", "appearance", "token_levels", "stream_images"):
        if torch.is_tensor(output.get(name)):
            output[name] = output[name].float()
    return output


def _payload_for_mode(payload: dict, mode: str) -> dict:
    if mode == "all_token_fusion":
        return payload
    # Camera-only modes never rerun DPT heads. Keep the common cache on disk,
    # but do not retain its large four-level token tensor in host memory.
    return {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "token_levels",
            "stream_images",
            "target_world_points",
            "target_depth",
            "baseline_depth",
            "baseline_world_points",
        }
    }


def _autocast_context(config: LearnedPoseConfig):
    if config.training.amp and str(config.training.device).startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _scalar_logs(values: Mapping[str, torch.Tensor]) -> dict[str, float]:
    output = {}
    for name, value in values.items():
        if torch.is_tensor(value):
            output[name] = float(value.detach().float().mean().cpu())
    return output


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _torch_load(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf8")
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
