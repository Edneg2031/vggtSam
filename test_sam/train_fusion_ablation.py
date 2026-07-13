#!/usr/bin/env python3
"""Train one SAM3/StreamVGGT mask-fusion ablation.

This experiment deliberately excludes pointmap prediction and camera tokens.
The only learned output is an FPN residual consumed by SAM3's original tracker
memory, object-presence gate, and mask decoder.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
import yaml

from test_sam.data import MaskTrackingSequence, load_mask_tracking_sequence
from test_sam.fusion import (
    FUSION_METHODS,
    SAM3GeometryFusion,
    stream_tokens_to_maps,
)
from vggtsam.adapters.sam3_intermediate import (
    SAM3IntermediateAdapter,
    load_sam3_image_model,
)
from vggtsam.adapters.sam3_video import (
    SAM3VideoTrackerAdapter,
    load_sam3_video_predictor,
)
from vggtsam.adapters.streamvggt_latent import (
    StreamVGGTLatentAdapter,
    load_streamvggt_latent_model,
)
from vggtsam.training.dense_fusion import (
    extract_sam3_sequence,
    run_sam3_source_tracker_flow,
)


@dataclass(frozen=True)
class ExperimentConfig:
    manifest: Path
    scene_id: str
    frame_indices: list[int] | None
    sequence_length: int
    frame_stride: int
    window_index: int
    instance_id: int | None
    min_pixels: int
    max_area_ratio: float
    min_visible_frames: int
    excluded_labels: list[str]
    sam3_repo: Path
    sam3_checkpoint: Path
    sam3_feature_device: str
    tracker_device: str
    direct_device: str
    sam3_resolution: int
    sam3_frame_chunk_size: int
    reference_prompt_mode: str
    compare_direct: bool
    streamvggt_repo: Path
    streamvggt_checkpoint: Path
    geometry_device: str
    geometry_streaming_cache: bool
    geometry_image_mode: str
    context_grid: tuple[int, int]
    layer_indices: tuple[int, ...]
    fusion_method: str
    hidden_channels: int
    num_heads: int
    dropout: float
    residual_scale: float
    residual_init_std: float
    inject_levels: tuple[str, ...]
    zero_geometry: bool
    shuffle_geometry: bool
    iterations: int
    lr: float
    tracker_lr: float
    seed: int
    amp: bool
    train_tracker: bool
    focal_weight: float
    dice_weight: float
    presence_weight: float
    gradient_clip: float
    log_every: int
    visualize_every: int
    save_every: int
    output_size: tuple[int, int]
    output_dir: Path


def main() -> None:
    args = parse_args()
    raw = load_yaml(args.config)
    apply_cli_overrides(raw, args)
    config = build_config(raw)
    run_experiment(config)


def run_experiment(config: ExperimentConfig) -> None:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    write_resolved_config(config)

    sequence = load_mask_tracking_sequence(
        config.manifest,
        scene_id=config.scene_id,
        frame_indices=config.frame_indices,
        sequence_length=config.sequence_length,
        frame_stride=config.frame_stride,
        window_index=config.window_index,
        instance_id=config.instance_id,
        min_pixels=config.min_pixels,
        max_area_ratio=config.max_area_ratio,
        min_visible_frames=config.min_visible_frames,
        excluded_labels=config.excluded_labels,
        seed=config.seed,
    )
    target_masks = resize_target_masks(sequence, config.output_size)
    visible = target_masks.flatten(1).any(dim=1)
    prompt_pixels = target_masks.flatten(1).sum(dim=1).tolist()
    print(
        "target "
        f"scene={sequence.scene_id} frames={sequence.frame_indices} "
        f"instance={sequence.instance_id} label={sequence.label!r} "
        f"reference_frame={sequence.reference_frame_idx} "
        f"prompt_pixels={[int(value) for value in prompt_pixels]}"
    )

    sam_features, sam_input_images, sam_tracker = extract_sam_features(
        config,
        sequence,
    )
    sam_fpn2 = sam_features["fpn2"].float()
    geometry_levels = extract_geometry_features(config, sequence)

    direct_masks = None
    direct_obj_id = None
    if config.compare_direct:
        direct_masks, direct_obj_id = run_direct_sam3(
            config,
            sequence,
            target_masks,
        )
        direct_metrics = tracking_metrics(
            direct_masks.float(),
            target_masks.float(),
            threshold=None,
            reference_frame_idx=sequence.reference_frame_idx,
        )
        print(
            "original_sam3 "
            f"obj_id={direct_obj_id} mean_iou={direct_metrics['mean_iou']:.4f} "
            f"positive_iou={direct_metrics['positive_iou']:.4f} "
            f"cross_iou={direct_metrics['cross_view_iou']:.4f} "
            f"cross_recall={direct_metrics['cross_view_recall']:.4f} "
            f"absent_fp={direct_metrics['absent_fp_ratio']:.6f}"
        )

    model = SAM3GeometryFusion(
        method=config.fusion_method,
        sam_channels=int(sam_fpn2.shape[1]),
        geometry_channels=(
            int(geometry_levels[-1].shape[1])
            if geometry_levels is not None
            else 2048
        ),
        hidden_channels=config.hidden_channels,
        num_heads=config.num_heads,
        dropout=config.dropout,
        inject_levels=config.inject_levels,
        residual_init_std=config.residual_init_std,
    ).to(config.tracker_device)
    fusion_trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    tracker_trainable = []
    sam_tracker.requires_grad_(False)
    if config.train_tracker:
        tracker_trainable = enable_tracker_training(sam_tracker)
    trainable = fusion_trainable + tracker_trainable
    parameter_groups = [{"params": fusion_trainable, "lr": config.lr}]
    if tracker_trainable:
        parameter_groups.append(
            {"params": tracker_trainable, "lr": config.tracker_lr}
        )
    optimizer = torch.optim.AdamW(parameter_groups)
    print(
        f"fusion={config.fusion_method} zero_geometry={config.zero_geometry} "
        f"shuffle_geometry={config.shuffle_geometry} "
        f"inject_levels={config.inject_levels} "
        f"residual_init_std={config.residual_init_std:g} "
        f"train_tracker={config.train_tracker} "
        f"fusion_lr={config.lr:g} tracker_lr={config.tracker_lr:g} "
        f"trainable_parameters={sum(parameter.numel() for parameter in trainable)}"
    )

    target_masks = target_masks.to(config.tracker_device)
    visible = visible.to(config.tracker_device)
    sam_input_images = sam_input_images.to(config.tracker_device)
    sam_fpn2 = sam_fpn2.to(config.tracker_device)
    if geometry_levels is not None:
        geometry_levels = [
            level.to(config.tracker_device) for level in geometry_levels
        ]

    metrics_path = config.output_dir / "training_history.csv"
    initialize_metrics_csv(metrics_path)
    for step in range(0, config.iterations + 1):
        training_step = step > 0
        train_loss_value = 0.0
        train_focal_value = float("nan")
        train_dice_value = float("nan")
        train_presence_value = float("nan")
        gradient_norm = 0.0
        fusion_gradient_norm = 0.0
        tracker_gradient_norm = 0.0
        residual_gradient_norm = 0.0
        if training_step:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(config):
                train_residuals = model(sam_fpn2, geometry_levels)
                for residual in train_residuals:
                    if residual.requires_grad:
                        residual.retain_grad()
                train_logits, train_aux = run_sam3_source_tracker_flow(
                    sam_tracker=sam_tracker,
                    sam_tracker_features=sam_features,
                    sam_input_images=sam_input_images,
                    tracker_residuals=train_residuals,
                    reference_mask=target_masks[sequence.reference_frame_idx],
                    reference_frame_idx=sequence.reference_frame_idx,
                    output_size=config.output_size,
                    residual_scale=config.residual_scale,
                    device=config.tracker_device,
                    reference_prompt_mode=config.reference_prompt_mode,
                    training_target_masks=target_masks,
                )
                train_focal = sigmoid_focal_loss(
                    train_logits,
                    target_masks.float(),
                )
                train_dice = dice_loss(train_logits, target_masks.float())
                train_object_scores = train_aux["object_score_logits"].reshape(-1)
                train_presence = F.binary_cross_entropy_with_logits(
                    train_object_scores,
                    visible.to(dtype=train_object_scores.dtype),
                )
                train_loss = (
                    config.focal_weight * train_focal
                    + config.dice_weight * train_dice
                    + config.presence_weight * train_presence
                )
            train_loss.backward()
            residual_gradient_norm = retained_gradient_norm(train_residuals)
            fusion_gradient_norm = total_gradient_norm(fusion_trainable)
            tracker_gradient_norm = total_gradient_norm(tracker_trainable)
            if config.gradient_clip > 0:
                norm = torch.nn.utils.clip_grad_norm_(
                    trainable,
                    config.gradient_clip,
                )
                gradient_norm = float(norm.detach().float().cpu())
            else:
                gradient_norm = total_gradient_norm(trainable)
            if step == 1 and (
                not np.isfinite(fusion_gradient_norm)
                or fusion_gradient_norm <= 1e-12
                or residual_gradient_norm <= 1e-12
            ):
                raise RuntimeError(
                    "Fusion adapter received no usable gradient at step 1. "
                    f"fusion_gradient_norm={fusion_gradient_norm}, "
                    f"tracker_gradient_norm={tracker_gradient_norm}, "
                    f"residual_gradient_norm={residual_gradient_norm}, "
                    f"train_logits_requires_grad={train_logits.requires_grad}. "
                    "If residual_gradient_norm is zero, the gradient is blocked "
                    "inside the SAM3 source flow; otherwise inspect the residual head."
                )
            optimizer.step()
            train_loss_value = float(train_loss.detach().float().cpu())
            train_focal_value = float(train_focal.detach().float().cpu())
            train_dice_value = float(train_dice.detach().float().cpu())
            train_presence_value = float(train_presence.detach().float().cpu())

        should_evaluate = (
            step == 0
            or step == 1
            or step == config.iterations
            or step % config.log_every == 0
            or step % config.visualize_every == 0
            or (training_step and step % config.save_every == 0)
        )
        if not should_evaluate:
            continue

        model.eval()
        with torch.no_grad(), autocast_context(config):
            residuals = model(sam_fpn2, geometry_levels)
            logits, source_aux = run_sam3_source_tracker_flow(
                sam_tracker=sam_tracker,
                sam_tracker_features=sam_features,
                sam_input_images=sam_input_images,
                tracker_residuals=residuals,
                reference_mask=target_masks[sequence.reference_frame_idx],
                reference_frame_idx=sequence.reference_frame_idx,
                output_size=config.output_size,
                residual_scale=config.residual_scale,
                device=config.tracker_device,
                reference_prompt_mode=config.reference_prompt_mode,
            )
            focal = sigmoid_focal_loss(logits, target_masks.float())
            dice = dice_loss(logits, target_masks.float())
            object_scores = source_aux["object_score_logits"].reshape(-1)
            presence = F.binary_cross_entropy_with_logits(
                object_scores,
                visible.to(dtype=object_scores.dtype),
            )
            loss = (
                config.focal_weight * focal
                + config.dice_weight * dice
                + config.presence_weight * presence
            )
            current_metrics = tracking_metrics(
                logits,
                target_masks.float(),
                threshold=0.5,
                reference_frame_idx=sequence.reference_frame_idx,
            )
            present_count = int((object_scores > 0).sum().item())
            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "train_loss": (
                    train_loss_value
                    if training_step
                    else float(loss.detach().cpu())
                ),
                "train_focal": train_focal_value,
                "train_dice": train_dice_value,
                "train_presence": train_presence_value,
                "focal": float(focal.detach().cpu()),
                "dice": float(dice.detach().cpu()),
                "presence": float(presence.detach().cpu()),
                "mean_iou": current_metrics["mean_iou"],
                "positive_iou": current_metrics["positive_iou"],
                "cross_view_iou": current_metrics["cross_view_iou"],
                "tracking_recall": current_metrics["tracking_recall"],
                "cross_view_recall": current_metrics["cross_view_recall"],
                "absent_fp_ratio": current_metrics["absent_fp_ratio"],
                "present_frames": present_count,
                "object_score_mean": float(object_scores.mean().detach().cpu()),
                "residual_rms": residual_rms(residuals),
                "gradient_norm": gradient_norm,
                "fusion_gradient_norm": fusion_gradient_norm,
                "tracker_gradient_norm": tracker_gradient_norm,
                "residual_gradient_norm": residual_gradient_norm,
            }
            append_metrics(metrics_path, row)

        if step % config.log_every == 0 or step == 1:
            print(
                f"step={step} eval_loss={row['loss']:.4f} "
                f"train_loss={row['train_loss']:.4f} "
                f"train_mask={row['train_focal']:.4f}/{row['train_dice']:.4f} "
                f"train_presence={row['train_presence']:.4f} "
                f"eval_mask={row['focal']:.4f}/{row['dice']:.4f} "
                f"eval_presence={row['presence']:.4f} "
                f"iou={row['mean_iou']:.4f} pos_iou={row['positive_iou']:.4f} "
                f"cross_iou={row['cross_view_iou']:.4f} "
                f"cross_recall={row['cross_view_recall']:.4f} "
                f"present={present_count}/{len(sequence.frame_indices)} "
                f"residual_rms={row['residual_rms']:.6f} "
                f"fusion_grad={fusion_gradient_norm:.6f} "
                f"tracker_grad={tracker_gradient_norm:.6f} "
                f"residual_grad={residual_gradient_norm:.6f}"
            )
        if step % config.visualize_every == 0 or step == config.iterations:
            save_visualization(
                config.output_dir / "visualizations" / f"step_{step:04d}.jpg",
                sequence=sequence,
                target_masks=target_masks,
                pred_logits=logits,
                direct_masks=direct_masks,
                object_scores=object_scores,
                output_size=config.output_size,
                fusion_method=config.fusion_method,
                zero_geometry=config.zero_geometry,
                shuffle_geometry=config.shuffle_geometry,
            )
            write_frame_metrics(
                config.output_dir / "frame_metrics.csv",
                sequence=sequence,
                target_masks=target_masks,
                pred_logits=logits,
                direct_masks=direct_masks,
                object_scores=object_scores,
            )
        if training_step and (
            step % config.save_every == 0 or step == config.iterations
        ):
            save_checkpoint(
                config.output_dir / "checkpoints" / f"step_{step:04d}.pt",
                step=step,
                model=model,
                sam_tracker=sam_tracker,
                train_tracker=config.train_tracker,
                config=config,
            )

    print(f"training history: {metrics_path}")
    plot_training_history(
        metrics_path,
        config.output_dir / "training_curves.png",
    )
    print(f"visualizations: {config.output_dir / 'visualizations'}")


def extract_sam_features(
    config: ExperimentConfig,
    sequence: MaskTrackingSequence,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.nn.Module]:
    sam_model = load_sam3_image_model(
        repo_path=config.sam3_repo,
        checkpoint_path=config.sam3_checkpoint,
        device=config.sam3_feature_device,
        enable_inst_interactivity=True,
    )
    sam_model.requires_grad_(False)
    inst_predictor = getattr(sam_model, "inst_interactive_predictor", None)
    if inst_predictor is None:
        raise RuntimeError("SAM3 did not expose inst_interactive_predictor.")
    sam_tracker = inst_predictor.model.to(config.tracker_device).eval()
    sam_tracker.requires_grad_(False)
    adapter = SAM3IntermediateAdapter(
        sam_model,
        device=config.sam3_feature_device,
        resolution=config.sam3_resolution,
        source="tracker_fpn2",
        text_conditioning="none",
        token_grid=(72, 72),
    )
    with torch.no_grad():
        output = extract_sam3_sequence(
            adapter,
            sequence.image_paths,
            prompt=sequence.label,
            chunk_size=config.sam3_frame_chunk_size,
            sam_tracker_for_features=sam_tracker,
        )
        sam_input_images = adapter._load_images(sequence.image_paths).detach().cpu()
    features = output.sam_tracker_features
    if features is None:
        raise RuntimeError("Failed to extract SAM3 tracker FPN features.")
    print(
        "SAM3 features "
        + " ".join(f"{key}={tuple(value.shape)}" for key, value in features.items())
    )
    del output
    del adapter
    del sam_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return features, sam_input_images, sam_tracker


def extract_geometry_features(
    config: ExperimentConfig,
    sequence: MaskTrackingSequence,
) -> list[torch.Tensor] | None:
    if config.fusion_method == "sam_only":
        print("StreamVGGT skipped for sam_only ablation.")
        return None
    model = load_streamvggt_latent_model(
        repo_path=config.streamvggt_repo,
        checkpoint_path=config.streamvggt_checkpoint,
        device=config.geometry_device,
        strict=True,
    )
    model.requires_grad_(False)
    adapter = StreamVGGTLatentAdapter(
        model,
        device=config.geometry_device,
        token_grid=(72, 72),
        context_grid=config.context_grid,
        layer_index=-1,
        dpt_layer_indices=config.layer_indices,
        image_mode=config.geometry_image_mode,
    )
    with torch.no_grad():
        output = adapter.extract_from_paths(
            sequence.image_paths,
            return_pointmap=False,
            streaming_cache=config.geometry_streaming_cache,
        )
    raw_levels = output.geometry.aux.get("stream_dpt_tokens")
    patch_start_idx = output.geometry.aux.get("patch_start_idx")
    patch_shape = output.aux.get("patch_shape")
    if raw_levels is None or patch_start_idx is None or patch_shape is None:
        raise RuntimeError("StreamVGGT did not return the requested aggregator layers.")
    levels = stream_tokens_to_maps(
        raw_levels,
        patch_start_idx=int(patch_start_idx),
        patch_shape=tuple(int(value) for value in patch_shape),
        output_grid=config.context_grid,
        zero_geometry=config.zero_geometry,
    )
    if config.shuffle_geometry:
        if levels[0].shape[0] < 2:
            raise ValueError("Geometry shuffling requires at least two frames.")
        permutation = torch.roll(
            torch.arange(levels[0].shape[0], device=levels[0].device),
            shifts=1,
        )
        levels = [level.index_select(0, permutation) for level in levels]
        print(f"StreamVGGT frame permutation={permutation.cpu().tolist()}")
    if config.fusion_method != "multilevel_cross_attention":
        levels = [levels[-1]]
    levels = [level.detach().cpu() for level in levels]
    print(
        f"StreamVGGT layers={config.layer_indices} streaming_cache="
        f"{config.geometry_streaming_cache} maps="
        f"{[tuple(level.shape) for level in levels]}"
    )
    del output
    del adapter
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return levels


def run_direct_sam3(
    config: ExperimentConfig,
    sequence: MaskTrackingSequence,
    target_masks: torch.Tensor,
) -> tuple[torch.Tensor, int | None]:
    predictor = load_sam3_video_predictor(
        repo_path=config.sam3_repo,
        checkpoint_path=config.sam3_checkpoint,
        device=config.direct_device,
        async_loading_frames=False,
    )
    tracker = SAM3VideoTrackerAdapter(
        predictor,
        output_prob_thresh=0.5,
        prompt_with_box=True,
    )
    output = tracker.track_from_paths(
        sequence.image_paths,
        prompt=sequence.label,
        output_size=config.output_size,
        prompt_frame_idx=sequence.reference_frame_idx,
        reference_mask=target_masks[sequence.reference_frame_idx],
    )
    masks = output.masks.detach().cpu().bool()
    selected_obj_id = output.selected_obj_id
    del output
    del tracker
    del predictor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return masks, selected_obj_id


def enable_tracker_training(sam_tracker: torch.nn.Module) -> list[torch.nn.Parameter]:
    memory_name = next(
        (
            name
            for name in ("memory_attention", "transformer")
            if isinstance(getattr(sam_tracker, name, None), torch.nn.Module)
        ),
        None,
    )
    if memory_name is None:
        raise RuntimeError(
            "SAM tracker exposes neither 'memory_attention' nor 'transformer'. "
            f"Available child modules: {list(dict(sam_tracker.named_children()))}"
        )
    decoder_name = "sam_mask_decoder"
    decoder = getattr(sam_tracker, decoder_name, None)
    if not isinstance(decoder, torch.nn.Module):
        raise RuntimeError(
            f"SAM tracker is missing trainable module {decoder_name!r}."
        )

    selected = [
        (memory_name, getattr(sam_tracker, memory_name)),
        (decoder_name, decoder),
    ]
    parameters = []
    parameter_ids = set()
    for _, module in selected:
        module.requires_grad_(True)
        for parameter in module.parameters():
            if parameter.requires_grad and id(parameter) not in parameter_ids:
                parameters.append(parameter)
                parameter_ids.add(id(parameter))
    print(
        "train_tracker enables "
        + " + ".join(f"SAM3 {name}" for name, _ in selected)
        + "."
    )
    return parameters


def resize_target_masks(
    sequence: MaskTrackingSequence,
    output_size: tuple[int, int],
) -> torch.Tensor:
    masks = torch.from_numpy(
        np.stack(sequence.target_masks, axis=0).astype(np.float32)
    )[:, None]
    return F.interpolate(masks, size=output_size, mode="nearest")[:, 0].bool()


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    cross_entropy = F.binary_cross_entropy_with_logits(
        logits.float(),
        targets.float(),
        reduction="none",
    )
    probability = logits.float().sigmoid()
    p_t = probability * targets + (1.0 - probability) * (1.0 - targets)
    loss = cross_entropy * (1.0 - p_t).pow(gamma)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return (alpha_t * loss).mean()


def dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probability = logits.float().sigmoid()
    probability = probability.flatten(1)
    targets = targets.float().flatten(1)
    numerator = 2.0 * (probability * targets).sum(dim=1) + 1.0
    denominator = probability.sum(dim=1) + targets.sum(dim=1) + 1.0
    return (1.0 - numerator / denominator).mean()


def tracking_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    threshold: float | None,
    reference_frame_idx: int | None = None,
) -> dict[str, float]:
    if threshold is not None:
        prediction = prediction.float().sigmoid() >= float(threshold)
    else:
        prediction = prediction.bool()
    target = target.bool().to(prediction.device)
    intersection = (prediction & target).flatten(1).sum(dim=1).float()
    union = (prediction | target).flatten(1).sum(dim=1).float()
    iou = torch.where(
        union > 0,
        intersection / union.clamp_min(1.0),
        torch.ones_like(union),
    )
    visible = target.flatten(1).any(dim=1)
    positive_iou = iou[visible].mean() if visible.any() else iou.new_tensor(0.0)
    recall = (
        (iou[visible] >= 0.5).float().mean()
        if visible.any()
        else iou.new_tensor(0.0)
    )
    cross_view = visible.clone()
    if reference_frame_idx is not None:
        cross_view[int(reference_frame_idx)] = False
    cross_view_iou = (
        iou[cross_view].mean() if cross_view.any() else iou.new_tensor(0.0)
    )
    cross_view_recall = (
        (iou[cross_view] >= 0.5).float().mean()
        if cross_view.any()
        else iou.new_tensor(0.0)
    )
    absent = ~visible
    absent_fp = (
        prediction[absent].float().mean()
        if absent.any()
        else iou.new_tensor(0.0)
    )
    return {
        "mean_iou": float(iou.mean().detach().cpu()),
        "positive_iou": float(positive_iou.detach().cpu()),
        "cross_view_iou": float(cross_view_iou.detach().cpu()),
        "tracking_recall": float(recall.detach().cpu()),
        "cross_view_recall": float(cross_view_recall.detach().cpu()),
        "absent_fp_ratio": float(absent_fp.detach().cpu()),
    }


def residual_rms(residuals: list[torch.Tensor]) -> float:
    values = torch.cat([residual.float().flatten() for residual in residuals])
    return float(values.square().mean().sqrt().detach().cpu())


def total_gradient_norm(parameters: list[torch.nn.Parameter]) -> float:
    squared = [
        parameter.grad.detach().float().square().sum()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not squared:
        return 0.0
    return float(torch.stack(squared).sum().sqrt().cpu())


def retained_gradient_norm(
    tensors: list[torch.Tensor],
) -> float:
    squared = [
        tensor.grad.detach().float().square().sum()
        for tensor in tensors
        if tensor.grad is not None
    ]
    if not squared:
        return 0.0
    return float(torch.stack(squared).sum().sqrt().cpu())


def save_visualization(
    path: Path,
    *,
    sequence: MaskTrackingSequence,
    target_masks: torch.Tensor,
    pred_logits: torch.Tensor,
    direct_masks: torch.Tensor | None,
    object_scores: torch.Tensor,
    output_size: tuple[int, int],
    fusion_method: str,
    zero_geometry: bool,
    shuffle_geometry: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    target = target_masks.detach().cpu().numpy().astype(bool)
    pred = (pred_logits.detach().float().sigmoid().cpu().numpy() >= 0.5)
    direct = (
        direct_masks.detach().cpu().numpy().astype(bool)
        if direct_masks is not None
        else None
    )
    scores = object_scores.detach().float().cpu().tolist()
    rows = []
    if fusion_method == "sam_only":
        geometry_label = "no StreamVGGT"
    elif zero_geometry:
        geometry_label = "StreamVGGT zeroed"
    elif shuffle_geometry:
        geometry_label = "StreamVGGT frame-shuffled"
    else:
        geometry_label = "StreamVGGT aligned"
    for index, image_path in enumerate(sequence.image_paths):
        with Image.open(image_path) as source:
            rgb = source.convert("RGB").resize(
                (output_size[1], output_size[0]),
                Image.Resampling.BILINEAR,
            )
        frame_state = frame_status(
            index=index,
            reference_frame_idx=sequence.reference_frame_idx,
            gt_visible=bool(target[index].any()),
        )
        gt_pixels = int(target[index].sum())
        pred_pixels = int(pred[index].sum())
        direct_pixels = int(direct[index].sum()) if direct is not None else 0
        frame_iou = mask_iou(pred[index], target[index])
        direct_iou = mask_iou(direct[index], target[index]) if direct is not None else None

        gt_panel = mark_mask(
            overlay_mask(rgb, target[index], (0, 210, 0)),
            target[index],
            (0, 255, 80),
            "GT",
        )
        pred_panel = mark_mask(
            overlay_mask(rgb, pred[index], (235, 50, 50)),
            pred[index],
            (255, 80, 80),
            "PRED",
        )
        direct_panel = (
            mark_mask(
                overlay_mask(rgb, direct[index], (30, 110, 255)),
                direct[index],
                (80, 170, 255),
                "SAM3",
            )
            if direct is not None
            else rgb.copy()
        )
        direct_text = (
            f"Original SAM3\nIoU={direct_iou:.3f} pix={direct_pixels}"
            if direct_iou is not None
            else "Original SAM3\nnot run"
        )
        panels = [
            annotate(
                rgb,
                f"RGB frame={sequence.frame_indices[index]} idx={index} {frame_state}\n"
                f"{fusion_method} | {geometry_label} | target={sequence.label}",
            ),
            annotate(gt_panel, f"GT instance={sequence.instance_id}\npix={gt_pixels}"),
            annotate(direct_panel, direct_text),
            annotate(
                pred_panel,
                f"Fused/source SAM3\nIoU={frame_iou:.3f} pix={pred_pixels} score={scores[index]:.2f}",
            ),
        ]
        rows.append(concatenate_horizontal(panels))
    concatenate_vertical(rows).save(path, quality=92)


def frame_status(
    *,
    index: int,
    reference_frame_idx: int,
    gt_visible: bool,
) -> str:
    if index == reference_frame_idx:
        return "ref"
    if gt_visible:
        return "cross-visible"
    return "absent"


def write_frame_metrics(
    path: Path,
    *,
    sequence: MaskTrackingSequence,
    target_masks: torch.Tensor,
    pred_logits: torch.Tensor,
    direct_masks: torch.Tensor | None,
    object_scores: torch.Tensor,
) -> None:
    target = target_masks.detach().cpu().numpy().astype(bool)
    pred = (pred_logits.detach().float().sigmoid().cpu().numpy() >= 0.5)
    direct = (
        direct_masks.detach().cpu().numpy().astype(bool)
        if direct_masks is not None
        else None
    )
    scores = object_scores.detach().float().cpu().tolist()
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sequence_index",
                "dataset_frame_index",
                "gt_visible",
                "gt_pixels",
                "fused_iou",
                "original_iou",
                "object_score",
            ],
        )
        writer.writeheader()
        for index in range(len(sequence.frame_indices)):
            writer.writerow(
                {
                    "sequence_index": index,
                    "dataset_frame_index": sequence.frame_indices[index],
                    "gt_visible": int(target[index].any()),
                    "gt_pixels": int(target[index].sum()),
                    "fused_iou": mask_iou(pred[index], target[index]),
                    "original_iou": (
                        mask_iou(direct[index], target[index])
                        if direct is not None
                        else ""
                    ),
                    "object_score": scores[index],
                }
            )


def mask_iou(prediction: np.ndarray, target: np.ndarray) -> float:
    union = np.logical_or(prediction, target).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(prediction, target).sum() / union)


def overlay_mask(
    image: Image.Image,
    mask: np.ndarray,
    color: tuple[int, int, int],
) -> Image.Image:
    base = np.asarray(image).copy()
    overlay = np.empty_like(base)
    overlay[...] = np.asarray(color, dtype=np.uint8)
    base[mask] = (0.45 * base[mask] + 0.55 * overlay[mask]).astype(np.uint8)
    return Image.fromarray(base)


def mark_mask(
    image: Image.Image,
    mask: np.ndarray,
    color: tuple[int, int, int],
    label: str,
) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    bbox = mask_bbox(mask)
    if bbox is None:
        draw.rectangle((6, 32, 90, 52), fill=(0, 0, 0))
        draw.text((10, 36), f"{label}: empty", fill=color)
        return output
    x0, y0, x1, y1 = bbox
    draw.rectangle((x0, y0, x1, y1), outline=color, width=3)
    text_y = max(30, y0 - 18)
    draw.rectangle((x0, text_y, min(output.width - 1, x0 + 118), text_y + 17), fill=(0, 0, 0))
    draw.text((x0 + 4, text_y + 3), f"{label}: {int(mask.sum())}", fill=color)
    return draw_mask_boundary(output, mask, color)


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def draw_mask_boundary(
    image: Image.Image,
    mask: np.ndarray,
    color: tuple[int, int, int],
) -> Image.Image:
    pixels = np.asarray(image).copy()
    if not mask.any():
        return Image.fromarray(pixels)
    padded = np.pad(mask.astype(bool), 1, mode="constant", constant_values=False)
    center = padded[1:-1, 1:-1]
    interior = (
        padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    boundary = center & ~interior
    pixels[boundary] = np.asarray(color, dtype=np.uint8)
    return Image.fromarray(pixels)


def annotate(image: Image.Image, text: str) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    lines = text.splitlines() or [text]
    bar_height = 20 * len(lines) + 6
    draw.rectangle((0, 0, output.width, bar_height), fill=(0, 0, 0))
    for line_index, line in enumerate(lines):
        draw.text((6, 5 + 18 * line_index), line, fill=(255, 255, 255))
    return output


def concatenate_horizontal(images: list[Image.Image]) -> Image.Image:
    output = Image.new(
        "RGB",
        (sum(image.width for image in images), max(image.height for image in images)),
    )
    x = 0
    for image in images:
        output.paste(image, (x, 0))
        x += image.width
    return output


def concatenate_vertical(images: list[Image.Image]) -> Image.Image:
    output = Image.new(
        "RGB",
        (max(image.width for image in images), sum(image.height for image in images)),
    )
    y = 0
    for image in images:
        output.paste(image, (0, y))
        y += image.height
    return output


def autocast_context(config: ExperimentConfig):
    if config.amp and str(config.tracker_device).startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


METRIC_FIELDS = [
    "step",
    "loss",
    "train_loss",
    "train_focal",
    "train_dice",
    "train_presence",
    "focal",
    "dice",
    "presence",
    "mean_iou",
    "positive_iou",
    "cross_view_iou",
    "tracking_recall",
    "cross_view_recall",
    "absent_fp_ratio",
    "present_frames",
    "object_score_mean",
    "residual_rms",
    "gradient_norm",
    "fusion_gradient_norm",
    "tracker_gradient_norm",
    "residual_gradient_norm",
]


def initialize_metrics_csv(path: Path) -> None:
    with path.open("w", newline="", encoding="utf8") as handle:
        csv.DictWriter(handle, fieldnames=METRIC_FIELDS).writeheader()


def append_metrics(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", newline="", encoding="utf8") as handle:
        csv.DictWriter(handle, fieldnames=METRIC_FIELDS).writerow(row)


def plot_training_history(csv_path: Path, output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib is unavailable; skipping training_curves.png")
        return
    with csv_path.open("r", encoding="utf8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return
    steps = [int(row["step"]) for row in rows]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(steps, [float(row["loss"]) for row in rows], label="eval total")
    axes[0].plot(
        steps,
        [float(row["train_loss"]) for row in rows],
        label="train total",
    )
    axes[0].plot(
        steps,
        [float(row["train_focal"]) for row in rows],
        label="train focal",
        linestyle="--",
    )
    axes[0].plot(
        steps,
        [float(row["train_dice"]) for row in rows],
        label="train dice",
        linestyle="--",
    )
    axes[0].plot(
        steps,
        [float(row["train_presence"]) for row in rows],
        label="train presence",
        linestyle="--",
    )
    axes[0].plot(steps, [float(row["focal"]) for row in rows], label="focal")
    axes[0].plot(steps, [float(row["dice"]) for row in rows], label="dice")
    axes[0].plot(steps, [float(row["presence"]) for row in rows], label="presence")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(
        steps,
        [float(row["cross_view_iou"]) for row in rows],
        label="cross-view IoU",
    )
    axes[1].plot(
        steps,
        [float(row["cross_view_recall"]) for row in rows],
        label="cross-view recall@0.5",
    )
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Cross-view tracking")
    axes[1].legend()
    axes[2].plot(
        steps,
        [float(row["absent_fp_ratio"]) for row in rows],
        label="absent false-positive ratio",
    )
    axes[2].plot(
        steps,
        [float(row["residual_rms"]) for row in rows],
        label="residual RMS",
    )
    axes[2].set_title("Failure / intervention")
    axes[2].legend()
    for axis in axes:
        axis.set_xlabel("step")
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    print(f"training curves: {output_path}")


def save_checkpoint(
    path: Path,
    *,
    step: int,
    model: torch.nn.Module,
    sam_tracker: torch.nn.Module,
    train_tracker: bool,
    config: ExperimentConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tracker_state = None
    if train_tracker:
        tracker_state = {
            name: value.detach().cpu()
            for name, value in sam_tracker.state_dict().items()
            if name.startswith("memory_attention.")
            or name.startswith("sam_mask_decoder.")
        }
    torch.save(
        {
            "step": step,
            "fusion": model.state_dict(),
            "tracker_trainable": tracker_state,
            "config": config_to_json(config),
        },
        path,
    )


def write_resolved_config(config: ExperimentConfig) -> None:
    with (config.output_dir / "resolved_config.json").open(
        "w",
        encoding="utf8",
    ) as handle:
        json.dump(config_to_json(config), handle, indent=2)


def config_to_json(config: ExperimentConfig) -> dict[str, Any]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in config.__dict__.items()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("test_sam/config.yaml"))
    parser.add_argument("--fusion-method", choices=FUSION_METHODS)
    parser.add_argument("--scene-id")
    parser.add_argument("--instance-id", type=int)
    parser.add_argument("--frame-indices", type=int, nargs="+")
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--tracker-device")
    parser.add_argument("--sam3-feature-device")
    parser.add_argument("--geometry-device")
    parser.add_argument("--direct-device")
    parser.add_argument("--zero-geometry", action="store_true")
    parser.add_argument("--shuffle-geometry", action="store_true")
    parser.add_argument(
        "--train-tracker",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--tracker-lr", type=float)
    parser.add_argument("--no-compare-direct", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf8") as handle:
        return yaml.safe_load(handle)


def apply_cli_overrides(raw: dict[str, Any], args: argparse.Namespace) -> None:
    mappings = [
        ("fusion_method", "fusion", "method"),
        ("scene_id", "dataset", "scene_id"),
        ("instance_id", "dataset", "instance_id"),
        ("frame_indices", "dataset", "frame_indices"),
        ("iterations", "training", "iterations"),
        ("lr", "training", "lr"),
        ("tracker_lr", "training", "tracker_lr"),
        ("output_dir", "training", "output_dir"),
        ("tracker_device", "sam3", "tracker_device"),
        ("sam3_feature_device", "sam3", "feature_device"),
        ("geometry_device", "streamvggt", "device"),
        ("direct_device", "sam3", "direct_device"),
    ]
    for argument, section, key in mappings:
        value = getattr(args, argument)
        if value is not None:
            raw.setdefault(section, {})[key] = value
    if args.zero_geometry:
        raw.setdefault("fusion", {})["zero_geometry"] = True
    if args.shuffle_geometry:
        raw.setdefault("fusion", {})["shuffle_geometry"] = True
    if args.train_tracker is not None:
        raw.setdefault("training", {})["train_tracker"] = args.train_tracker
    if args.no_compare_direct:
        raw.setdefault("sam3", {})["compare_direct"] = False


def build_config(raw: dict[str, Any]) -> ExperimentConfig:
    dataset = raw["dataset"]
    sam3 = raw["sam3"]
    stream = raw["streamvggt"]
    fusion = raw["fusion"]
    training = raw["training"]
    return ExperimentConfig(
        manifest=Path(dataset["manifest"]),
        scene_id=str(dataset["scene_id"]),
        frame_indices=(
            [int(value) for value in dataset["frame_indices"]]
            if dataset.get("frame_indices")
            else None
        ),
        sequence_length=int(dataset.get("sequence_length", 4)),
        frame_stride=int(dataset.get("frame_stride", 1)),
        window_index=int(dataset.get("window_index", 0)),
        instance_id=(
            int(dataset["instance_id"])
            if dataset.get("instance_id") is not None
            else None
        ),
        min_pixels=int(dataset.get("min_pixels", 128)),
        max_area_ratio=float(dataset.get("max_area_ratio", 0.25)),
        min_visible_frames=int(dataset.get("min_visible_frames", 2)),
        excluded_labels=list(dataset.get("excluded_labels", [])),
        sam3_repo=Path(sam3["repo"]),
        sam3_checkpoint=Path(sam3["checkpoint"]),
        sam3_feature_device=str(sam3["feature_device"]),
        tracker_device=str(sam3["tracker_device"]),
        direct_device=str(sam3["direct_device"]),
        sam3_resolution=int(sam3.get("resolution", 1008)),
        sam3_frame_chunk_size=int(sam3.get("frame_chunk_size", 1)),
        reference_prompt_mode=str(sam3.get("reference_prompt_mode", "mask")),
        compare_direct=bool(sam3.get("compare_direct", True)),
        streamvggt_repo=Path(stream["repo"]),
        streamvggt_checkpoint=Path(stream["checkpoint"]),
        geometry_device=str(stream["device"]),
        geometry_streaming_cache=bool(stream.get("streaming_cache", True)),
        geometry_image_mode=str(stream.get("image_mode", "crop")),
        context_grid=tuple(int(value) for value in stream.get("context_grid", [12, 12])),
        layer_indices=tuple(int(value) for value in stream["layer_indices"]),
        fusion_method=str(fusion["method"]),
        hidden_channels=int(fusion.get("hidden_channels", 256)),
        num_heads=int(fusion.get("num_heads", 8)),
        dropout=float(fusion.get("dropout", 0.0)),
        residual_scale=float(fusion.get("residual_scale", 1.0)),
        residual_init_std=float(fusion.get("residual_init_std", 1e-4)),
        inject_levels=tuple(
            str(value) for value in fusion.get("inject_levels", ["fpn2"])
        ),
        zero_geometry=bool(fusion.get("zero_geometry", False)),
        shuffle_geometry=bool(fusion.get("shuffle_geometry", False)),
        iterations=int(training.get("iterations", 700)),
        lr=float(training.get("lr", 3e-4)),
        tracker_lr=float(training.get("tracker_lr", 5e-5)),
        seed=int(training.get("seed", 0)),
        amp=bool(training.get("amp", True)),
        train_tracker=bool(training.get("train_tracker", False)),
        focal_weight=float(training.get("focal_weight", 20.0)),
        dice_weight=float(training.get("dice_weight", 1.0)),
        presence_weight=float(training.get("presence_weight", 1.0)),
        gradient_clip=float(training.get("gradient_clip", 1.0)),
        log_every=int(training.get("log_every", 10)),
        visualize_every=int(training.get("visualize_every", 20)),
        save_every=int(training.get("save_every", 100)),
        output_size=tuple(int(value) for value in training.get("output_size", [256, 384])),
        output_dir=Path(training.get("output_dir", "outputs/test_sam")),
    )


if __name__ == "__main__":
    main()
