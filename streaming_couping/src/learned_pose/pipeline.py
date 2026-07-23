"""Training and evaluation pipeline for persistent-instance pose adapters."""

from __future__ import annotations

from contextlib import nullcontext
import csv
import gc
import json
from pathlib import Path
import random
from typing import Iterable, Mapping

import numpy as np
import torch

from vggtsam.adapters.streamvggt_latent import load_streamvggt_latent_model

from ..config import load_config
from ..pose_evaluation import (
    PoseSequence,
    _all_pair_pose_metrics,
    _evaluate_pose_alignment,
    _prepare_pose_sequence,
    _reference_pose_alignment,
    _summarize_pose_pairs,
)
from .cache import cache_path, load_feature_cache
from .config import (
    GEOMETRY_MODES,
    PATCH_MODES,
    V2_MODES,
    ClipConfig,
    LearnedPoseConfig,
)
from .geometry_metrics import (
    append_geometry_metrics,
    summarize_depth_metrics,
    summarize_pointmap_metrics,
)
from .losses import compute_training_loss, instance_rigid_losses
from .model import InstancePoseAdapter
from .ray_pose import recover_ray_pose_variants


def train_all_modes(config: LearnedPoseConfig) -> None:
    train_clips = [
        clip for clip in config.clips if clip.split.lower() == "train"
    ]
    validation_clips = [
        clip
        for clip in config.clips
        if clip.split.lower() in {"val", "validation"}
    ]
    if not train_clips:
        raise ValueError("At least one dataset clip must use split: train.")
    train_paths = [cache_path(config, clip) for clip in train_clips]
    validation_paths = [cache_path(config, clip) for clip in validation_clips]
    first = _slice_training_payload(
        load_feature_cache(train_paths[0]),
        train_clips[0],
    )
    frozen = _load_frozen_streamvggt(config, device=config.training.device)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    for mode in config.training.modes:
        if mode in GEOMETRY_MODES and "token_levels" not in first:
            raise RuntimeError(
                f"{mode} requires features.cache_all_token_levels=true."
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
        payload_cache = {}
        for clip, path in zip(train_clips, train_paths):
            payload = load_feature_cache(path)
            payload = _slice_training_payload(payload, clip)
            payload_cache[path] = _payload_for_mode(payload, mode)
        for clip, path in zip(validation_clips, validation_paths):
            payload_cache[path] = _payload_for_mode(
                load_feature_cache(path),
                mode,
            )
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
                    "supervision_frame_indices": " ".join(
                        str(value) for value in batch["frame_indices"]
                    ),
                    "gradient_norm": float(grad_norm),
                    **{name: float(value.detach().cpu()) for name, value in terms.items()},
                    **_scalar_logs(outputs.get("diagnostics", {})),
                }
                epoch_rows.append(row)
                history.append(row)
                print(
                    f"mode={mode} epoch={epoch:03d} step={step:03d} "
                    f"loss={row['total']:.6f} pose={row['camera']:.6f} "
                    f"pointmap={row['pointmap']:.6f} depth={row['depth']:.6f} "
                    f"rigid={row['rigid']:.6f} grad={row['gradient_norm']:.4f}"
                )
                del batch, outputs, total, terms, grad_norm
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
                "checkpoint_selection": (
                    "validation_clips"
                    if validation_paths
                    else "training_supervision_frames"
                ),
                "training_frame_indices": {
                    clip.name: list(_training_frame_indices(clip))
                    for clip in train_clips
                },
                "config_path": str(config.source_path),
            }
            torch.save(checkpoint, mode_dir / "checkpoint_last.pt")
            if validation_loss < best_loss:
                best_loss = validation_loss
                torch.save(checkpoint, mode_dir / "checkpoint_best.pt")
            _write_csv(mode_dir / "training_history.csv", history)
        print(f"finished mode={mode} best_validation_loss={best_loss:.6f}")
        del adapter, optimizer, payload_cache, checkpoint
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@torch.no_grad()
def evaluate_all_modes(
    config: LearnedPoseConfig,
    *,
    ray_pose_only: bool = False,
) -> None:
    if ray_pose_only and not config.evaluation.ray_pose.enabled:
        raise ValueError(
            "--stage ray requires evaluation.ray_pose.enabled=true."
        )
    evaluation_clips = [
        clip
        for clip in config.clips
        if clip.split.lower() in {"val", "validation", "test"}
        or clip.evaluation_frame_indices is not None
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
    pointmap_frame_rows = []
    depth_frame_rows = []
    ray_summary_rows = []
    ray_frame_rows = []
    ray_rpe_rows = []
    ray_pair_rows = []
    ray_pair_summary_rows = []
    ray_fit_rows = []
    ray_completed_clips: set[str] = set()
    ray_pose_predictions: dict[str, dict] = {}

    # Raw StreamVGGT is evaluated once per clip through the exact same batch
    # CameraHead path used by every adapter.
    for clip in evaluation_clips:
        payload = load_feature_cache(cache_path(config, clip))
        _validate_cached_payload(payload, clip)
        batch = _batch_to_device(payload, config.training.device)
        baseline_outputs = _forward_baseline(batch)
        evaluation_indices = _evaluation_sequence_indices(clip)
        evaluation_metadata = _evaluation_metadata(clip)
        equivalence_rows.append(
            {
                "clip": clip.name,
                "mode": "baseline_cached_output",
                "token_max_abs_diff": 0.0,
                "pose_max_abs_diff": 0.0,
                "depth_max_abs_diff": 0.0,
                "pointmap_max_abs_diff": 0.0,
                "strict_equal": 1,
                **evaluation_metadata,
            }
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
            sequence_indices=evaluation_indices,
            evaluation_metadata=evaluation_metadata,
        )
        append_geometry_metrics(
            pointmap_frame_rows,
            depth_frame_rows,
            batch=batch,
            outputs=baseline_outputs,
            mode="baseline",
            perturbation="module_off",
            sequence_indices=evaluation_indices,
            evaluation_metadata=evaluation_metadata,
        )

    evaluation_modes = (
        (config.evaluation.ray_pose.source_mode,)
        if ray_pose_only
        else config.training.modes
    )
    for mode in evaluation_modes:
        checkpoint_path = config.output_dir / "checkpoints" / mode / "checkpoint_best.pt"
        if not checkpoint_path.exists():
            print(f"skipping missing checkpoint: {checkpoint_path}")
            continue
        checkpoint = _torch_load(checkpoint_path)
        adapter = _new_adapter(first, config).to(config.training.device)
        _load_adapter_checkpoint(adapter, checkpoint, mode=mode)
        adapter.eval()
        for clip in evaluation_clips:
            payload = load_feature_cache(cache_path(config, clip))
            _validate_cached_payload(payload, clip)
            batch = _batch_to_device(payload, config.training.device)
            baseline_outputs = _forward_baseline(batch)
            evaluation_indices = _evaluation_sequence_indices(clip)
            evaluation_metadata = _evaluation_metadata(clip)
            perturbations = (
                (config.evaluation.ray_pose.source_perturbation,)
                if ray_pose_only
                else config.evaluation.perturbations
            )
            for perturbation in perturbations:
                if (
                    perturbation in {"pose_branch_off", "geometry_branch_off"}
                    and mode != "decoupled_dual_branch"
                ):
                    continue
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
                    sequence_indices=evaluation_indices,
                    evaluation_metadata=evaluation_metadata,
                )
                append_geometry_metrics(
                    pointmap_frame_rows,
                    depth_frame_rows,
                    batch=batch,
                    outputs=outputs,
                    mode=mode,
                    perturbation=perturbation,
                    sequence_indices=evaluation_indices,
                    evaluation_metadata=evaluation_metadata,
                )
                ray_config = config.evaluation.ray_pose
                if (
                    ray_config.enabled
                    and mode == ray_config.source_mode
                    and perturbation == ray_config.source_perturbation
                ):
                    baseline_world_confidence = _decode_baseline_world_confidence(
                        frozen,
                        batch,
                    )
                    control_metadata = {
                        **evaluation_metadata,
                        "ray_pose_role": "raw_baseline_control",
                    }
                    _append_pose_metrics(
                        ray_summary_rows,
                        ray_frame_rows,
                        ray_rpe_rows,
                        ray_pair_rows,
                        ray_pair_summary_rows,
                        payload=payload,
                        pose_encoding=baseline_outputs["pose_encoding"],
                        mode="instance_ray_pose_v3",
                        perturbation="raw_baseline_control",
                        sequence_indices=evaluation_indices,
                        evaluation_metadata=control_metadata,
                    )
                    _append_pose_metrics(
                        ray_summary_rows,
                        ray_frame_rows,
                        ray_rpe_rows,
                        ray_pair_rows,
                        ray_pair_summary_rows,
                        payload=payload,
                        pose_encoding=outputs["pose_encoding"],
                        mode="instance_ray_pose_v3",
                        perturbation="v2_learned_pose_control",
                        sequence_indices=evaluation_indices,
                        evaluation_metadata={
                            **evaluation_metadata,
                            "ray_pose_role": "learned_pose_control",
                        },
                    )
                    ray_results = recover_ray_pose_variants(
                        batch=batch,
                        baseline_outputs=baseline_outputs,
                        refined_outputs=outputs,
                        baseline_world_confidence=baseline_world_confidence,
                        config=ray_config,
                    )
                    clip_predictions = {
                        "raw_baseline_control": baseline_outputs[
                            "pose_encoding"
                        ].detach().float().cpu(),
                        "v2_learned_pose_control": outputs[
                            "pose_encoding"
                        ].detach().float().cpu(),
                    }
                    for result in ray_results:
                        clip_predictions[result.name] = (
                            result.pose_encoding.detach().float().cpu()
                        )
                        result_metadata = {
                            **evaluation_metadata,
                            "ray_pose_role": result.role,
                        }
                        _append_pose_metrics(
                            ray_summary_rows,
                            ray_frame_rows,
                            ray_rpe_rows,
                            ray_pair_rows,
                            ray_pair_summary_rows,
                            payload=payload,
                            pose_encoding=result.pose_encoding,
                            mode="instance_ray_pose_v3",
                            perturbation=result.name,
                            sequence_indices=evaluation_indices,
                            evaluation_metadata=result_metadata,
                        )
                        ray_fit_rows.extend(
                            {
                                "clip": clip.name,
                                "scene_id": clip.scene_id,
                                "mode": "instance_ray_pose_v3",
                                "perturbation": result.name,
                                **result_metadata,
                                **row,
                            }
                            for row in result.diagnostics
                        )
                    ray_pose_predictions[clip.name] = {
                        "scene_id": clip.scene_id,
                        "frame_indices": list(clip.frame_indices),
                        "instance_ids": list(clip.instance_ids),
                        "image_paths": list(payload["image_paths"]),
                        "pose_encodings": clip_predictions,
                        "refined_world_points": outputs[
                            "world_points"
                        ].detach().float().cpu(),
                        "refined_world_confidence": outputs[
                            "world_confidence"
                        ].detach().float().cpu(),
                        "tracking_masks_stream": batch[
                            "tracking_masks_stream"
                        ].detach().bool().cpu(),
                    }
                    ray_completed_clips.add(clip.name)
                rigid, centroid = instance_rigid_losses(
                    outputs["pose_encoding"].float(),
                    batch["instance_uvd"].float(),
                    batch["instance_uvd_valid"].bool(),
                    batch["instance_rigid_weight"].float(),
                    image_size=tuple(int(v) for v in batch["image_size"]),
                    scene_scale=float(batch["scene_scale"]),
                    trim_quantile=config.loss.rigid_trim_quantile,
                    sequence_indices=evaluation_indices,
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
                    sequence_indices=evaluation_indices,
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
                        **evaluation_metadata,
                        "instance_metric_scope": "evaluated_frames_only",
                        "adapter_log_scope": "full_causal_context",
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
                    if mode in GEOMETRY_MODES:
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
                            **evaluation_metadata,
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
    if not ray_pose_only:
        _write_csv(output / "pose_summary.csv", summary_rows)
        _write_csv(output / "pose_frame_metrics.csv", frame_rows)
        _write_csv(output / "pose_rpe.csv", rpe_rows)
        _write_csv(output / "pose_pair_metrics.csv", pair_rows)
        _write_csv(output / "pose_pair_summary.csv", pair_summary_rows)
        _write_csv(output / "instance_diagnostics.csv", diagnostic_rows)
        _write_csv(output / "baseline_equivalence.csv", equivalence_rows)
        _write_csv(output / "pointmap_frame_metrics.csv", pointmap_frame_rows)
        _write_csv(
            output / "pointmap_summary.csv",
            summarize_pointmap_metrics(pointmap_frame_rows),
        )
        _write_csv(output / "depth_frame_metrics.csv", depth_frame_rows)
        _write_csv(
            output / "depth_summary.csv",
            summarize_depth_metrics(depth_frame_rows),
        )
    if config.evaluation.ray_pose.enabled:
        expected_clips = {clip.name for clip in evaluation_clips}
        missing_clips = sorted(expected_clips - ray_completed_clips)
        if missing_clips:
            raise RuntimeError(
                "Ray-pose evaluation did not run for clips: "
                f"{missing_clips}. Check the configured source checkpoint/mode."
            )
        _write_csv(output / "ray_pose_summary.csv", ray_summary_rows)
        _write_csv(output / "ray_pose_frame_metrics.csv", ray_frame_rows)
        _write_csv(output / "ray_pose_rpe.csv", ray_rpe_rows)
        _write_csv(output / "ray_pose_pair_metrics.csv", ray_pair_rows)
        _write_csv(output / "ray_pose_pair_summary.csv", ray_pair_summary_rows)
        _write_csv(output / "ray_pose_fit_diagnostics.csv", ray_fit_rows)
        _write_csv(
            output / "ray_pose_compact_summary.csv",
            _compact_ray_pose_summary(
                ray_summary_rows,
                ray_pair_summary_rows,
                ray_fit_rows,
            ),
        )
        torch.save(
            {
                "config": str(config.source_path),
                "source_mode": config.evaluation.ray_pose.source_mode,
                "source_perturbation": (
                    config.evaluation.ray_pose.source_perturbation
                ),
                "final_variant": config.evaluation.ray_pose.final_variant,
                "predictions": ray_pose_predictions,
            },
            output / "ray_pose_predictions.pt",
        )
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
        "gt_role": (
            "pose/depth/pointmap training supervision and evaluation only; target "
            "intrinsics are additionally read by the explicitly named GT-K oracle row"
        ),
        "geometry_evaluation": (
            "All pointmap modes reuse the baseline point-head reference-frame Sim(3); "
            "it is never refit after refinement. point_head measures direct DPT point "
            "output, while baseline_point_head_refined_pose isolates the pose update by "
            "reprojecting frozen baseline geometry. Depth uses its own baseline-reference "
            "median scale because the StreamVGGT depth and point heads have different scales."
        ),
        "decoupled_v2": (
            "Patch modes preserve every camera/register prefix token. The dual branch uses "
            "independent pose and geometry tokenizers, attentions, gates, and projections."
        ),
        "instance_ray_pose_v3": {
            "enabled": config.evaluation.ray_pose.enabled,
            "source_mode": config.evaluation.ray_pose.source_mode,
            "source_perturbation": config.evaluation.ray_pose.source_perturbation,
            "final_variant": config.evaluation.ray_pose.final_variant,
            "variants": list(config.evaluation.ray_pose.variants),
            "causal_constraint": (
                "Each requested camera center is solved from its current causal pointmap "
                "and rays. Reference-frame predicted intrinsics are the only cross-frame "
                "camera calibration used; no evaluation GT enters a deployable variant."
            ),
            "oracle_constraint": (
                "Only the explicitly named gt_k_oracle variant reads target intrinsics."
            ),
        },
        "evaluation_splits": [
            {
                "clip": clip.name,
                **_evaluation_metadata(clip),
                "checkpoint_selection": (
                    "validation_clips"
                    if any(
                        item.split.lower() in {"val", "validation"}
                        for item in config.clips
                    )
                    else "training_supervision_frames"
                ),
            }
            for clip in evaluation_clips
        ],
    }
    metadata_name = "ray_pose_metadata.json" if ray_pose_only else "metadata.json"
    with (output / metadata_name).open("w", encoding="utf8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    label = "ray-pose" if ray_pose_only else "learned-pose"
    print(f"{label} evaluation written to {output}")


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


def _load_adapter_checkpoint(
    adapter: InstancePoseAdapter,
    checkpoint: dict,
    *,
    mode: str,
) -> None:
    checkpoint_mode = str(checkpoint.get("mode", mode))
    if checkpoint_mode in V2_MODES:
        adapter.load_state_dict(checkpoint["adapter"], strict=True)
        return
    incompatible = adapter.load_state_dict(checkpoint["adapter"], strict=False)
    allowed_missing = ("geometry_tokenizer.", "patch_token_fusions.")
    bad_missing = [
        key
        for key in incompatible.missing_keys
        if not key.startswith(allowed_missing)
    ]
    if bad_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint is incompatible with the instance adapter: "
            f"missing={bad_missing}, unexpected={incompatible.unexpected_keys}."
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
        token_levels = _token_level_mapping(batch)
        updated, diagnostics = adapter.forward_all_tokens(
            token_levels,
            appearance=batch["appearance"],
            geometry=batch["geometry"],
            quality=batch["quality"],
            observed=batch["observed"],
            perturbation=perturbation,
        )
        final_layer = max(updated)
        camera_hidden = updated[final_layer][:, :, 0]
        pose_delta = _decode_streaming_camera_delta(
            frozen.camera_head,
            batch["camera_hidden"],
            updated[final_layer][:, :, 0],
        )
        pose_encoding = batch["baseline_pose_encoding"] + pose_delta
        return _geometry_head_outputs(
            frozen,
            batch,
            updated,
            pose_encoding=pose_encoding,
            camera_hidden=camera_hidden,
            diagnostics=diagnostics,
        )
    if mode in PATCH_MODES:
        token_levels = _token_level_mapping(batch)
        updated, diagnostics = adapter.forward_patch_tokens(
            token_levels,
            patch_start_idx=int(batch["patch_start_idx"]),
            appearance=batch["appearance"],
            geometry=batch["geometry"],
            quality=batch["quality"],
            observed=batch["observed"],
            mode=mode,
            perturbation=perturbation,
        )
        return _geometry_head_outputs(
            frozen,
            batch,
            updated,
            pose_encoding=batch["baseline_pose_encoding"],
            camera_hidden=batch["camera_hidden"],
            diagnostics=diagnostics,
        )
    if mode == "decoupled_dual_branch":
        pose_perturbation = perturbation
        geometry_perturbation = perturbation
        if perturbation == "pose_branch_off":
            pose_perturbation = "module_off"
            geometry_perturbation = "aligned"
        elif perturbation == "geometry_branch_off":
            pose_perturbation = "aligned"
            geometry_perturbation = "module_off"
        camera_hidden, pose_diagnostics = adapter.forward_camera(
            batch["camera_hidden"],
            appearance=batch["appearance"],
            geometry=batch["geometry"],
            quality=batch["quality"],
            observed=batch["observed"],
            mode="camera_sam_only",
            perturbation=pose_perturbation,
        )
        pose_delta = _decode_streaming_camera_delta(
            frozen.camera_head,
            batch["camera_hidden"],
            camera_hidden,
        )
        pose_encoding = batch["baseline_pose_encoding"] + pose_delta
        updated, geometry_diagnostics = adapter.forward_patch_tokens(
            _token_level_mapping(batch),
            patch_start_idx=int(batch["patch_start_idx"]),
            appearance=batch["appearance"],
            geometry=batch["geometry"],
            quality=batch["quality"],
            observed=batch["observed"],
            mode="patch_sam_geometry_tracker_gate",
            perturbation=geometry_perturbation,
        )
        diagnostics = {
            **{f"pose_{key}": value for key, value in pose_diagnostics.items()},
            **{
                f"geometry_{key}": value
                for key, value in geometry_diagnostics.items()
            },
        }
        return _geometry_head_outputs(
            frozen,
            batch,
            updated,
            pose_encoding=pose_encoding,
            camera_hidden=camera_hidden,
            diagnostics=diagnostics,
        )
    camera_hidden, diagnostics = adapter.forward_camera(
        batch["camera_hidden"],
        appearance=batch["appearance"],
        geometry=batch["geometry"],
        quality=batch["quality"],
        observed=batch["observed"],
        mode=mode,
        perturbation=perturbation,
    )
    pose_delta = _decode_streaming_camera_delta(
        frozen.camera_head,
        batch["camera_hidden"],
        camera_hidden,
    )
    pose_encoding = batch["baseline_pose_encoding"] + pose_delta
    return {
        "pose_encoding": pose_encoding,
        "camera_hidden": camera_hidden,
        "residual_mean_square": diagnostics["residual_mean_square"],
        "diagnostics": diagnostics,
    }


def _token_level_mapping(batch: dict) -> dict[int, torch.Tensor]:
    layers = [int(value) for value in batch["dpt_layer_indices"]]
    values = batch["token_levels"]
    return {
        layer: values[:, index]
        for index, layer in enumerate(layers)
    }


def _geometry_head_outputs(
    frozen,
    batch: dict,
    updated: Mapping[int, torch.Tensor],
    *,
    pose_encoding: torch.Tensor,
    camera_hidden: torch.Tensor,
    diagnostics: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor | dict]:
    final_layer = max(int(layer) for layer in updated)
    aggregated: list[torch.Tensor | None] = [None] * (final_layer + 1)
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
    residual_values = [
        value
        for key, value in diagnostics.items()
        if key.endswith("residual_mean_square")
    ]
    residual_mean_square = (
        torch.stack(residual_values).mean()
        if residual_values
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
        "diagnostics": dict(diagnostics),
    }


def _decode_baseline_world_confidence(frozen, batch: dict) -> torch.Tensor:
    """Replay only the frozen point head to recover its uncached confidence."""

    token_levels = _token_level_mapping(batch)
    final_layer = max(int(layer) for layer in token_levels)
    aggregated: list[torch.Tensor | None] = [None] * (final_layer + 1)
    for layer, value in token_levels.items():
        aggregated[int(layer)] = value
    decoded_points, confidence = _decode_streaming_dpt_head(
        frozen.point_head,
        aggregated,
        batch["stream_images"],
        patch_start_idx=int(batch["patch_start_idx"]),
    )
    difference = float(
        (decoded_points - batch["baseline_world_points"]).abs().max().cpu()
    )
    if difference != 0.0:
        raise RuntimeError(
            "Frozen point-head replay does not reproduce the cached baseline: "
            f"max_abs_diff={difference}."
        )
    return confidence


def _forward_baseline(batch: dict) -> dict[str, torch.Tensor]:
    return {
        "pose_encoding": batch["baseline_pose_encoding"],
        "camera_hidden": batch["camera_hidden"],
        "depth": batch["baseline_depth"],
        "world_points": batch["baseline_world_points"],
    }


def _decode_streaming_camera_delta(
    camera_head,
    raw_camera_hidden: torch.Tensor,
    refined_camera_hidden: torch.Tensor,
) -> torch.Tensor:
    """Decode a frozen-head delta while anchoring the exact cached raw pose."""

    if raw_camera_hidden.shape != refined_camera_hidden.shape:
        raise ValueError("Raw and refined camera hidden shapes disagree.")
    batch = raw_camera_hidden.shape[0]
    combined = torch.cat(
        [raw_camera_hidden.detach(), refined_camera_hidden],
        dim=0,
    )
    decoded = _decode_streaming_camera_head(camera_head, combined)
    raw_decoded = decoded[:batch]
    refined_decoded = decoded[batch:]
    delta = refined_decoded - raw_decoded.detach()
    if torch.equal(raw_camera_hidden, refined_camera_hidden):
        # Enforce a numerically exact zero while preserving the CameraHead
        # Jacobian needed to train the initially-zero projection.
        return delta - delta.detach()
    return delta


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
    baseline = _forward_baseline(batch)
    output = _forward_mode(
        frozen,
        adapter,
        batch,
        mode=mode,
        perturbation="aligned",
    )
    token_difference = float((output["camera_hidden"] - baseline["camera_hidden"]).abs().max().cpu())
    pose_difference = float((output["pose_encoding"] - baseline["pose_encoding"]).abs().max().cpu())
    depth_difference = 0.0
    pointmap_difference = 0.0
    if mode in GEOMETRY_MODES:
        depth_difference = float(
            (output["depth"] - baseline["depth"]).abs().max().cpu()
        )
        pointmap_difference = float(
            (output["world_points"] - baseline["world_points"]).abs().max().cpu()
        )
    if (
        token_difference != 0.0
        or pose_difference != 0.0
        or depth_difference != 0.0
        or pointmap_difference != 0.0
    ):
        raise RuntimeError(
            "Zero-initialized adapter does not exactly reproduce baseline: "
            f"token={token_difference}, pose={pose_difference}, "
            f"depth={depth_difference}, pointmap={pointmap_difference}."
        )
    print("zero-init baseline equivalence: exact")
    return {
        "mode": mode,
        "clip": payload["clip_name"],
        "token_max_abs_diff": token_difference,
        "pose_max_abs_diff": pose_difference,
        "depth_max_abs_diff": depth_difference,
        "pointmap_max_abs_diff": pointmap_difference,
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
    sequence_indices: Iterable[int] | None = None,
    evaluation_metadata: Mapping[str, object] | None = None,
) -> None:
    from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri

    predicted_w2c, _ = pose_encoding_to_extri_intri(
        pose_encoding.float(),
        image_size_hw=tuple(payload["image_size"]),
    )
    predicted_full = _prepare_pose_sequence(
        predicted_w2c[0].detach().double().cpu(),
        frame_indices=payload["frame_indices"],
        source=f"{mode}:{perturbation}",
    )
    target_full = _prepare_pose_sequence(
        payload["target_world_to_camera"].double().cpu(),
        frame_indices=payload["frame_indices"],
        source="scannetpp_colmap",
    )
    alignment = _reference_pose_alignment(
        predicted_full,
        target_full,
        reference_index=int(payload["reference_sequence_index"]),
        scale=float(payload["point_alignment_scale"]),
    )
    indices = _normalize_sequence_indices(
        sequence_indices,
        sequence_length=len(payload["frame_indices"]),
    )
    predicted = _slice_pose_sequence(predicted_full, indices)
    target = _slice_pose_sequence(target_full, indices)
    evaluated_frames = [int(payload["frame_indices"][index]) for index in indices]
    original_reference = int(payload["reference_sequence_index"])
    evaluation_reference = (
        indices.index(original_reference)
        if original_reference in indices
        else -1
    )
    summary, current_frames, current_rpe = _evaluate_pose_alignment(
        alignment,
        predicted=predicted,
        target=target,
        frame_indices=evaluated_frames,
        reference_index=evaluation_reference,
    )
    current_pairs = _all_pair_pose_metrics(
        predicted,
        target,
        frame_indices=evaluated_frames,
    )
    _restore_source_sequence_indices(
        current_frames,
        current_rpe,
        current_pairs,
        source_indices=indices,
    )
    current_pair_summary = _summarize_pose_pairs(current_pairs)
    prefix = {
        "clip": payload["clip_name"],
        "scene_id": payload["scene_id"],
        "mode": mode,
        "perturbation": perturbation,
        **dict(evaluation_metadata or {}),
    }
    summary_rows.append({**prefix, **summary})
    frame_rows.extend({**prefix, **row} for row in current_frames)
    rpe_rows.extend({**prefix, **row} for row in current_rpe)
    pair_rows.extend({**prefix, **row} for row in current_pairs)
    pair_summary_rows.extend({**prefix, **row} for row in current_pair_summary)


def _training_frame_indices(clip: ClipConfig) -> tuple[int, ...]:
    return clip.training_frame_indices or clip.frame_indices


def _sequence_indices_for_frames(
    clip: ClipConfig,
    frame_indices: Iterable[int],
) -> list[int]:
    lookup = {int(frame): index for index, frame in enumerate(clip.frame_indices)}
    return [lookup[int(frame)] for frame in frame_indices]


def _evaluation_sequence_indices(clip: ClipConfig) -> list[int]:
    frames = clip.evaluation_frame_indices or clip.frame_indices
    return _sequence_indices_for_frames(clip, frames)


def _evaluation_metadata(clip: ClipConfig) -> dict[str, object]:
    evaluated = clip.evaluation_frame_indices or clip.frame_indices
    training = (
        _training_frame_indices(clip)
        if clip.split.lower() == "train"
        else ()
    )
    if clip.evaluation_frame_indices is not None:
        protocol = "causal_temporal_holdout"
    elif clip.split.lower() in {"val", "validation", "test"}:
        protocol = "held_out_clip"
    else:
        protocol = "in_sample_all_frames"
    return {
        "evaluation_protocol": protocol,
        "context_frames": len(clip.frame_indices),
        "context_frame_indices": " ".join(str(value) for value in clip.frame_indices),
        "training_frame_indices": " ".join(str(value) for value in training),
        "evaluated_frame_indices": " ".join(str(value) for value in evaluated),
        "alignment_reference_frame_index": clip.frame_indices[
            clip.reference_sequence_index
        ],
    }


def _slice_training_payload(payload: dict, clip: ClipConfig) -> dict:
    _validate_cached_payload(payload, clip)
    frames = _training_frame_indices(clip)
    indices = _sequence_indices_for_frames(clip, frames)
    if len(indices) == len(clip.frame_indices):
        output = dict(payload)
        output["cache_context_frame_indices"] = list(clip.frame_indices)
        output["supervision_frame_indices"] = list(frames)
        return output

    sequence_length = len(clip.frame_indices)
    tensor_fields = {
        "camera_hidden",
        "appearance",
        "geometry",
        "quality",
        "observed",
        "target_pose_encoding",
        "target_world_to_camera",
        "instance_uvd",
        "instance_uvd_valid",
        "instance_rigid_weight",
        "target_world_points",
        "target_depth",
        "baseline_pose_encoding",
        "baseline_depth",
        "baseline_world_points",
        "stream_images",
        "tracking_masks_output",
        "tracking_masks_stream",
        "tracking_scores",
    }
    index = torch.tensor(indices, dtype=torch.long)
    output = dict(payload)
    for field in tensor_fields:
        value = payload.get(field)
        if not torch.is_tensor(value):
            continue
        if value.shape[0] != sequence_length:
            raise ValueError(
                f"Cannot temporal-slice {field}: expected leading sequence "
                f"dimension {sequence_length}, got {tuple(value.shape)}."
            )
        output[field] = value.index_select(0, index)
    token_levels = payload.get("token_levels")
    if torch.is_tensor(token_levels):
        if token_levels.ndim < 2 or token_levels.shape[1] != sequence_length:
            raise ValueError(
                "Cannot temporal-slice token_levels: expected [L,S,...] with "
                f"S={sequence_length}, got {tuple(token_levels.shape)}."
            )
        output["token_levels"] = token_levels.index_select(1, index)
    image_paths = payload.get("image_paths")
    if image_paths is not None:
        output["image_paths"] = [image_paths[item] for item in indices]
    original_reference = int(payload["reference_sequence_index"])
    if original_reference not in indices:
        raise ValueError(
            f"Training frames for {clip.name!r} do not contain reference index "
            f"{original_reference}."
        )
    output["cache_context_frame_indices"] = list(payload["frame_indices"])
    output["supervision_frame_indices"] = list(frames)
    output["frame_indices"] = list(frames)
    output["reference_sequence_index"] = indices.index(original_reference)
    return output


def _validate_cached_payload(payload: dict, clip: ClipConfig) -> None:
    if str(payload.get("clip_name")) != clip.name:
        raise ValueError(
            f"Cached clip name is {payload.get('clip_name')!r}, expected "
            f"{clip.name!r}."
        )
    cached_frames = tuple(int(value) for value in payload.get("frame_indices", ()))
    if cached_frames != clip.frame_indices:
        raise ValueError(
            f"Cached frames for {clip.name!r} are {cached_frames}, expected "
            f"{clip.frame_indices}. Rebuild or select the correct cache."
        )
    if str(payload.get("scene_id")) != clip.scene_id:
        raise ValueError(
            f"Cached scene for {clip.name!r} is {payload.get('scene_id')!r}, "
            f"expected {clip.scene_id!r}."
        )


def _normalize_sequence_indices(
    sequence_indices: Iterable[int] | None,
    *,
    sequence_length: int,
) -> list[int]:
    if sequence_indices is None:
        return list(range(sequence_length))
    indices = [int(value) for value in sequence_indices]
    if not indices:
        raise ValueError("Evaluation sequence_indices must not be empty.")
    if len(set(indices)) != len(indices):
        raise ValueError("Evaluation sequence_indices contain duplicates.")
    if any(value < 0 or value >= sequence_length for value in indices):
        raise ValueError(
            f"Evaluation sequence_indices must be in [0,{sequence_length})."
        )
    if indices != sorted(indices):
        raise ValueError("Evaluation sequence_indices must be increasing.")
    return indices


def _slice_pose_sequence(
    sequence: PoseSequence,
    indices: list[int],
) -> PoseSequence:
    index = torch.tensor(indices, dtype=torch.long)
    return PoseSequence(
        world_to_camera=sequence.world_to_camera.index_select(0, index),
        camera_to_world_rotation=sequence.camera_to_world_rotation.index_select(
            0,
            index,
        ),
        camera_centers=sequence.camera_centers.index_select(0, index),
        rotation_quality_rows=tuple(
            sequence.rotation_quality_rows[value] for value in indices
        ),
    )


def _restore_source_sequence_indices(
    frame_rows: list[dict],
    rpe_rows: list[dict],
    pair_rows: list[dict],
    *,
    source_indices: list[int],
) -> None:
    for row in frame_rows:
        row["sequence_index"] = source_indices[int(row["sequence_index"])]
    for row in [*rpe_rows, *pair_rows]:
        row["first_sequence_index"] = source_indices[
            int(row["first_sequence_index"])
        ]
        row["second_sequence_index"] = source_indices[
            int(row["second_sequence_index"])
        ]


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
        "tracking_masks_stream",
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
    if mode in GEOMETRY_MODES:
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


def _compact_ray_pose_summary(
    pose_rows: list[dict],
    pair_rows: list[dict],
    fit_rows: list[dict],
) -> list[dict]:
    """Join primary pose/pair/fit values into one copy-friendly CSV."""

    pair_by_variant = {
        (str(row["clip"]), str(row["perturbation"])): row
        for row in pair_rows
    }
    fits_by_variant: dict[tuple[str, str], list[dict]] = {}
    for row in fit_rows:
        evaluated = {
            int(value)
            for value in str(row.get("evaluated_frame_indices", "")).split()
        }
        if evaluated and int(row["frame_index"]) not in evaluated:
            continue
        key = (str(row["clip"]), str(row["perturbation"]))
        fits_by_variant.setdefault(key, []).append(row)
    output = []
    for pose in pose_rows:
        variant = str(pose["perturbation"])
        key = (str(pose["clip"]), variant)
        pair = pair_by_variant.get(key, {})
        fits = fits_by_variant.get(key, [])
        output.append(
            {
                "clip": pose["clip"],
                "variant": variant,
                "variant_role": pose.get("ray_pose_role", ""),
                "evaluation_protocol": pose.get("evaluation_protocol", ""),
                "training_frame_indices": pose.get("training_frame_indices", ""),
                "evaluated_frame_indices": pose.get("evaluated_frame_indices", ""),
                "evaluated_frames": pose["evaluated_frames"],
                "ate_rmse": pose["ate_rmse"],
                "translation_error_mean": pose["translation_error_mean"],
                "rotation_error_mean_degrees": pose[
                    "rotation_error_mean_degrees"
                ],
                "rpe_translation_rmse": pose["rpe_translation_rmse"],
                "rpe_rotation_mean_degrees": pose[
                    "rpe_rotation_mean_degrees"
                ],
                "pair_rotation_error_mean_degrees": pair.get(
                    "rotation_error_mean_degrees",
                    float("nan"),
                ),
                "pair_translation_direction_error_mean_degrees": pair.get(
                    "translation_direction_error_mean_degrees",
                    float("nan"),
                ),
                "fit_frames": len(fits),
                "fit_accepted_frames": sum(
                    int(row["fit_accepted"]) for row in fits
                ),
                "mean_initial_point_ray_rmse_native": _finite_row_mean(
                    fits,
                    "initial_point_ray_rmse_native",
                ),
                "mean_fitted_point_ray_rmse_native": _finite_row_mean(
                    fits,
                    "fitted_point_ray_rmse_native",
                ),
                "mean_initial_angular_rmse": _finite_row_mean(
                    fits,
                    "initial_angular_rmse",
                ),
                "mean_fitted_angular_rmse": _finite_row_mean(
                    fits,
                    "fitted_angular_rmse",
                ),
                "mean_applied_center_shift_native": _finite_row_mean(
                    fits,
                    "applied_center_shift_native",
                ),
                "max_condition_number": _finite_row_max(
                    fits,
                    "condition_number",
                ),
            }
        )
    return output


def _finite_row_mean(rows: list[dict], field: str) -> float:
    values = [
        float(row[field])
        for row in rows
        if np.isfinite(float(row[field]))
    ]
    return float(np.mean(values)) if values else float("nan")


def _finite_row_max(rows: list[dict], field: str) -> float:
    values = [
        float(row[field])
        for row in rows
        if np.isfinite(float(row[field]))
    ]
    return float(np.max(values)) if values else float("nan")


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
