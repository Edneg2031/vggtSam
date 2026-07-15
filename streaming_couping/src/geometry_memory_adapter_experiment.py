"""Train the learned §3.1 geometry-aware SAM3 memory adapter."""

from __future__ import annotations

import argparse
import csv
import json
from contextlib import nullcontext
from pathlib import Path
import random
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

from test_sam.data import load_mask_tracking_sequence
from vggtsam.adapters.sam3_intermediate import (
    SAM3IntermediateAdapter,
    load_sam3_image_model,
)
from vggtsam.training.dense_fusion import (
    extract_sam3_sequence,
    run_sam3_source_tracker_flow,
)

from .backbones.streamvggt_wrapper import StreamVGGTWrapper
from .bridge.geometry_feature_merger import GeometryAwareSAM3Adapter
from .bridge.memory_warp import (
    GeometryMemoryPositionWarper,
    install_memory_position_warp,
)
from .config import ExperimentConfig, load_config
from .pipeline import _resize_target_masks, summarize_masks


def main() -> None:
    args = _parse_args()
    overrides = {
        key: value
        for key, value in {
            "manifest": args.manifest,
            "scene_id": args.scene_id,
            "instance_id": args.instance_id,
            "frame_indices": args.frame_indices,
            "sam3_device": args.sam3_device,
            "geometry_device": args.geometry_device,
            "output_dir": args.output_dir,
        }.items()
        if value is not None
    }
    config = load_config(args.config, overrides)
    run_experiment(
        config,
        sam3_feature_device=args.sam3_feature_device or config.sam3_device,
        layer_indices=tuple(args.geometry_layers),
        context_grid=tuple(args.context_grid),
        point_source=args.geometry_point_source,
        min_geometry_confidence=args.min_geometry_confidence,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        hidden_channels=args.hidden_channels,
        num_heads=args.num_heads,
        residual_init_std=args.residual_init_std,
        residual_scale=args.residual_scale,
        focal_weight=args.focal_weight,
        dice_weight=args.dice_weight,
        presence_weight=args.presence_weight,
        rank_weight=args.geometry_rank_weight,
        rank_margin=args.geometry_rank_margin,
        rank_every=args.geometry_rank_every,
        residual_weight=args.residual_weight,
        reference_prompt_mode=args.reference_prompt_mode,
        amp=args.amp,
        gradient_clip=args.gradient_clip,
        log_every=args.log_every,
        visualize_every=args.visualize_every,
        save_every=args.save_every,
        seed=args.seed,
    )


def run_experiment(
    config: ExperimentConfig,
    *,
    sam3_feature_device: str,
    layer_indices: tuple[int, ...],
    context_grid: tuple[int, int],
    point_source: str,
    min_geometry_confidence: float,
    iterations: int,
    learning_rate: float,
    hidden_channels: int,
    num_heads: int,
    residual_init_std: float,
    residual_scale: float,
    focal_weight: float,
    dice_weight: float,
    presence_weight: float,
    rank_weight: float,
    rank_margin: float,
    rank_every: int,
    residual_weight: float,
    reference_prompt_mode: str,
    amp: bool,
    gradient_clip: float,
    log_every: int,
    visualize_every: int,
    save_every: int,
    seed: int,
) -> None:
    _set_seed(seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    sequence = load_mask_tracking_sequence(
        config.manifest,
        scene_id=config.scene_id,
        frame_indices=config.frame_indices,
        sequence_length=len(config.frame_indices),
        frame_stride=1,
        window_index=0,
        instance_id=config.instance_id,
        min_pixels=config.min_pixels,
        max_area_ratio=config.max_area_ratio,
        min_visible_frames=1,
        excluded_labels=config.excluded_labels,
        seed=seed,
    )
    target_masks = _resize_target_masks(sequence.target_masks, config.output_size)
    visible = target_masks.flatten(1).any(dim=1)
    print(
        f"target scene={sequence.scene_id} frames={sequence.frame_indices} "
        f"instance={sequence.instance_id} label={sequence.label!r} "
        f"reference={sequence.reference_frame_idx} "
        f"visible={visible.tolist()}"
    )

    print("extracting frozen StreamVGGT geometry and multi-level features once...")
    stream = StreamVGGTWrapper(
        repo_path=config.streamvggt_repo,
        checkpoint_path=config.streamvggt_checkpoint,
        device=config.geometry_device,
        image_mode=config.image_mode,
        streaming_cache=config.streaming_cache,
    ).load()
    geometry, geometry_levels = stream.extract_with_latents(
        sequence.image_paths,
        layer_indices=layer_indices,
        context_grid=context_grid,
    )
    del stream
    _empty_cuda_cache()
    confidence = (
        geometry.depth_confidence
        if point_source == "depth_camera"
        else geometry.confidence
    )
    if confidence is None:
        raise RuntimeError(f"Geometry confidence is unavailable for {point_source!r}.")
    print(
        f"StreamVGGT layers={layer_indices} maps="
        f"{[tuple(level.shape) for level in geometry_levels]} "
        f"point_source={point_source}"
    )

    print("extracting frozen SAM3 tracker FPN features once...")
    sam_features, sam_input_images, sam_tracker = _extract_sam3_features(
        config,
        sequence.image_paths,
        prompt=sequence.label,
        feature_device=sam3_feature_device,
        tracker_device=config.sam3_device,
    )
    sam_tracker.requires_grad_(False)
    sam_tracker.eval()

    tracker_device = torch.device(config.sam3_device)
    target_masks = target_masks.to(tracker_device)
    visible = visible.to(tracker_device)
    sam_input_images = sam_input_images.to(tracker_device)
    sam_fpn2 = sam_features["fpn2"].float().to(tracker_device)
    geometry_levels = tuple(level.to(tracker_device) for level in geometry_levels)
    confidence = confidence.float().to(tracker_device)
    permutation = torch.roll(
        torch.arange(len(sequence.frame_indices), device=tracker_device),
        shifts=1,
    )
    shuffled_levels = tuple(
        level.index_select(0, permutation) for level in geometry_levels
    )
    shuffled_confidence = confidence.index_select(0, permutation)

    adapter = GeometryAwareSAM3Adapter(
        num_geometry_levels=len(layer_indices),
        sam_channels=int(sam_fpn2.shape[1]),
        geometry_channels=int(geometry_levels[0].shape[1]),
        hidden_channels=hidden_channels,
        num_heads=num_heads,
        residual_init_std=residual_init_std,
    ).to(tracker_device)
    trainable = [parameter for parameter in adapter.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate)
    print(
        f"trainable=geometry_feature_merger parameters="
        f"{sum(parameter.numel() for parameter in trainable)} "
        f"SAM3=frozen StreamVGGT=frozen default_gate=object_score_logits>0"
    )

    aligned_train_warper = _make_warper(
        geometry,
        mode="aligned",
        image_mode=config.image_mode,
        permutation=permutation.cpu().tolist(),
        point_source=point_source,
        min_geometry_confidence=min_geometry_confidence,
        record_observations=False,
    )
    shuffled_train_warper = _make_warper(
        geometry,
        mode="shuffled",
        image_mode=config.image_mode,
        permutation=permutation.cpu().tolist(),
        point_source=point_source,
        min_geometry_confidence=min_geometry_confidence,
        record_observations=False,
    )
    history_path = config.output_dir / "training_history.csv"
    _initialize_history(history_path)

    for step in range(iterations + 1):
        train_values = _empty_train_values()
        if step > 0:
            adapter.train()
            optimizer.zero_grad(set_to_none=True)
            shuffled_scores = None
            if rank_weight > 0 and step % max(rank_every, 1) == 0:
                with torch.no_grad(), _autocast(config.sam3_device, amp):
                    shuffled_residuals, _ = adapter(
                        sam_fpn2,
                        shuffled_levels,
                        shuffled_confidence,
                    )
                    _, shuffled_aux = _run_source_flow(
                        sam_tracker=sam_tracker,
                        sam_features=sam_features,
                        sam_input_images=sam_input_images,
                        residuals=shuffled_residuals,
                        target_masks=target_masks,
                        reference_frame_idx=sequence.reference_frame_idx,
                        output_size=config.output_size,
                        residual_scale=residual_scale,
                        device=config.sam3_device,
                        reference_prompt_mode=reference_prompt_mode,
                        training=True,
                        warper=shuffled_train_warper,
                    )
                    shuffled_scores = shuffled_aux["object_score_logits"].detach()

            with _autocast(config.sam3_device, amp):
                residuals, adapter_aux = adapter(
                    sam_fpn2,
                    geometry_levels,
                    confidence,
                )
                residuals[-1].retain_grad()
                train_logits, train_aux = _run_source_flow(
                    sam_tracker=sam_tracker,
                    sam_features=sam_features,
                    sam_input_images=sam_input_images,
                    residuals=residuals,
                    target_masks=target_masks,
                    reference_frame_idx=sequence.reference_frame_idx,
                    output_size=config.output_size,
                    residual_scale=residual_scale,
                    device=config.sam3_device,
                    reference_prompt_mode=reference_prompt_mode,
                    training=True,
                    warper=aligned_train_warper,
                )
                focal = _sigmoid_focal_loss(train_logits, target_masks.float())
                dice = _visible_dice_loss(train_logits, target_masks.float(), visible)
                object_scores = train_aux["object_score_logits"].reshape(-1)
                presence = F.binary_cross_entropy_with_logits(
                    object_scores.float(),
                    visible.to(dtype=object_scores.dtype).float(),
                )
                rank = _geometry_rank_loss(
                    object_scores,
                    shuffled_scores,
                    visible=visible,
                    reference_frame_idx=sequence.reference_frame_idx,
                    margin=rank_margin,
                )
                residual_regularization = residuals[-1].float().square().mean()
                loss = (
                    focal_weight * focal
                    + dice_weight * dice
                    + presence_weight * presence
                    + rank_weight * rank
                    + residual_weight * residual_regularization
                )
            loss.backward()
            residual_gradient = _tensor_gradient_norm(residuals[-1])
            adapter_gradient = _parameter_gradient_norm(trainable)
            if step == 1 and (adapter_gradient <= 1e-12 or residual_gradient <= 1e-12):
                raise RuntimeError(
                    "The §3.1 adapter received no gradient through SAM3: "
                    f"adapter_grad={adapter_gradient}, residual_grad={residual_gradient}."
                )
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable, gradient_clip)
            optimizer.step()
            train_values = {
                "train_loss": float(loss.detach().cpu()),
                "train_focal": float(focal.detach().cpu()),
                "train_dice": float(dice.detach().cpu()),
                "train_presence": float(presence.detach().cpu()),
                "train_rank": float(rank.detach().cpu()),
                "adapter_gradient_norm": adapter_gradient,
                "residual_gradient_norm": residual_gradient,
                "geometry_gate_mean": float(
                    adapter_aux["geometry_gate_mean"].detach().cpu()
                ),
            }

        should_log = (
            step == 0
            or step == 1
            or step == iterations
            or step % max(log_every, 1) == 0
        )
        if should_log:
            complete = _evaluate_one(
                adapter=adapter,
                mode="complete",
                sam_tracker=sam_tracker,
                sam_features=sam_features,
                sam_input_images=sam_input_images,
                sam_fpn2=sam_fpn2,
                geometry_levels=geometry_levels,
                confidence=confidence,
                geometry=geometry,
                target_masks=target_masks,
                reference_frame_idx=sequence.reference_frame_idx,
                output_size=config.output_size,
                residual_scale=residual_scale,
                device=config.sam3_device,
                reference_prompt_mode=reference_prompt_mode,
                image_mode=config.image_mode,
                point_source=point_source,
                min_geometry_confidence=min_geometry_confidence,
                permutation=permutation.cpu().tolist(),
                amp=amp,
            )
            row = {"step": step, **train_values, **complete["metrics"]}
            _append_history(history_path, row)
            print(
                f"step={step} train={row['train_loss']:.4f} "
                f"focal={row['train_focal']:.4f} dice={row['train_dice']:.4f} "
                f"presence={row['train_presence']:.4f} rank={row['train_rank']:.4f} "
                f"cross_iou={row['cross_view_iou']:.4f} "
                f"cross_recall={row['cross_view_recall']:.4f} "
                f"absent_fp={row['absent_fp_ratio']:.6f} "
                f"score={row['cross_object_score_mean']:.3f}/"
                f"{row['absent_object_score_mean']:.3f} "
                f"gate={row['geometry_gate_mean']:.4f}"
            )
            if step % max(visualize_every, 1) == 0 or step == iterations:
                _save_single_visualization(
                    config.output_dir / "visualizations" / f"step_{step:04d}.png",
                    image_paths=sequence.image_paths,
                    frame_indices=sequence.frame_indices,
                    target_masks=target_masks,
                    prediction=complete["prediction"],
                    object_scores=complete["object_scores"],
                    output_size=config.output_size,
                )
        if step > 0 and (step % max(save_every, 1) == 0 or step == iterations):
            _save_checkpoint(
                config.output_dir / "checkpoints" / f"step_{step:04d}.pt",
                step=step,
                adapter=adapter,
                layer_indices=layer_indices,
                context_grid=context_grid,
            )

    final_results = _evaluate_controls(
        adapter=adapter,
        sam_tracker=sam_tracker,
        sam_features=sam_features,
        sam_input_images=sam_input_images,
        sam_fpn2=sam_fpn2,
        geometry_levels=geometry_levels,
        confidence=confidence,
        shuffled_levels=shuffled_levels,
        shuffled_confidence=shuffled_confidence,
        geometry=geometry,
        target_masks=target_masks,
        reference_frame_idx=sequence.reference_frame_idx,
        output_size=config.output_size,
        residual_scale=residual_scale,
        device=config.sam3_device,
        reference_prompt_mode=reference_prompt_mode,
        image_mode=config.image_mode,
        point_source=point_source,
        min_geometry_confidence=min_geometry_confidence,
        permutation=permutation.cpu().tolist(),
        amp=amp,
    )
    _write_control_outputs(
        config.output_dir,
        results=final_results,
        image_paths=sequence.image_paths,
        frame_indices=sequence.frame_indices,
        target_masks=target_masks,
        output_size=config.output_size,
    )
    _write_resolved_experiment(
        config.output_dir / "resolved_experiment.json",
        config=config,
        sequence=sequence,
        layer_indices=layer_indices,
        context_grid=context_grid,
        point_source=point_source,
        min_geometry_confidence=min_geometry_confidence,
        iterations=iterations,
        learning_rate=learning_rate,
        reference_prompt_mode=reference_prompt_mode,
        rank_weight=rank_weight,
        rank_margin=rank_margin,
        seed=seed,
    )
    _plot_training_history(history_path, config.output_dir / "training_curves.png")
    print(f"training history: {history_path}")
    print(f"ablation summary: {config.output_dir / 'ablation_summary.csv'}")


def _extract_sam3_features(
    config: ExperimentConfig,
    image_paths: Sequence[Path],
    *,
    prompt: str,
    feature_device: str,
    tracker_device: str,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.nn.Module]:
    model = load_sam3_image_model(
        repo_path=config.sam3_repo,
        checkpoint_path=config.sam3_checkpoint,
        device=feature_device,
        enable_inst_interactivity=True,
    )
    model.requires_grad_(False)
    predictor = getattr(model, "inst_interactive_predictor", None)
    if predictor is None:
        raise RuntimeError("SAM3 did not expose inst_interactive_predictor.")
    tracker = predictor.model.to(tracker_device).eval()
    tracker.requires_grad_(False)
    adapter = SAM3IntermediateAdapter(
        model,
        device=feature_device,
        resolution=1008,
        source="tracker_fpn2",
        text_conditioning="none",
        token_grid=(72, 72),
    )
    with torch.no_grad():
        output = extract_sam3_sequence(
            adapter,
            image_paths,
            prompt=prompt,
            chunk_size=1,
            sam_tracker_for_features=tracker,
        )
        images = adapter._load_images(image_paths).detach().cpu()
    features = output.sam_tracker_features
    if features is None:
        raise RuntimeError("SAM3 tracker FPN extraction failed.")
    print(
        "SAM3 features "
        + " ".join(f"{key}={tuple(value.shape)}" for key, value in features.items())
    )
    del output, adapter, model
    _empty_cuda_cache()
    return features, images, tracker


def _run_source_flow(
    *,
    sam_tracker,
    sam_features,
    sam_input_images,
    residuals,
    target_masks,
    reference_frame_idx,
    output_size,
    residual_scale,
    device,
    reference_prompt_mode,
    training,
    warper,
):
    context = (
        install_memory_position_warp(sam_tracker, warper)
        if warper is not None
        else nullcontext()
    )
    with context:
        return run_sam3_source_tracker_flow(
            sam_tracker=sam_tracker,
            sam_tracker_features=sam_features,
            sam_input_images=sam_input_images,
            tracker_residuals=residuals,
            reference_mask=target_masks[reference_frame_idx],
            reference_frame_idx=reference_frame_idx,
            output_size=output_size,
            residual_scale=residual_scale,
            device=device,
            reference_prompt_mode=reference_prompt_mode,
            training_target_masks=target_masks if training else None,
        )


def _make_warper(
    geometry,
    *,
    mode: str,
    image_mode: str,
    permutation: Sequence[int],
    point_source: str,
    min_geometry_confidence: float,
    record_observations: bool,
) -> GeometryMemoryPositionWarper:
    return GeometryMemoryPositionWarper(
        geometry,
        mode=mode,
        image_mode=image_mode,
        frame_permutation=tuple(int(index) for index in permutation),
        min_geometry_confidence=min_geometry_confidence,
        point_source=point_source,
        record_observations=record_observations,
    )


def _evaluate_one(
    *,
    adapter,
    mode,
    sam_tracker,
    sam_features,
    sam_input_images,
    sam_fpn2,
    geometry_levels,
    confidence,
    geometry,
    target_masks,
    reference_frame_idx,
    output_size,
    residual_scale,
    device,
    reference_prompt_mode,
    image_mode,
    point_source,
    min_geometry_confidence,
    permutation,
    amp,
):
    adapter.eval()
    with torch.no_grad(), _autocast(device, amp):
        if mode in {"original", "warp_only"}:
            residuals = _zero_residuals(sam_fpn2)
            adapter_aux = {
                "geometry_gate_mean": sam_fpn2.new_zeros(()),
                "fpn2_residual_rms": sam_fpn2.new_zeros(()),
            }
        else:
            residuals, adapter_aux = adapter(
                sam_fpn2,
                geometry_levels,
                confidence,
            )
        warper = None
        if mode in {"warp_only", "complete", "shuffled"}:
            warper = _make_warper(
                geometry,
                mode="shuffled" if mode == "shuffled" else "aligned",
                image_mode=image_mode,
                permutation=permutation,
                point_source=point_source,
                min_geometry_confidence=min_geometry_confidence,
                record_observations=True,
            )
        logits, aux = _run_source_flow(
            sam_tracker=sam_tracker,
            sam_features=sam_features,
            sam_input_images=sam_input_images,
            residuals=residuals,
            target_masks=target_masks,
            reference_frame_idx=reference_frame_idx,
            output_size=output_size,
            residual_scale=residual_scale,
            device=device,
            reference_prompt_mode=reference_prompt_mode,
            training=False,
            warper=warper,
        )
    prediction = logits.float().sigmoid() >= 0.5
    object_scores = aux["object_score_logits"].float().reshape(-1)
    metrics = summarize_masks(
        prediction,
        target_masks,
        reference_frame_idx=reference_frame_idx,
    )
    visible = target_masks.flatten(1).any(dim=1)
    cross = visible.clone()
    cross[reference_frame_idx] = False
    absent = ~visible
    warp_summary = warper.summary() if warper is not None else {}
    metrics.update(
        {
            "cross_object_score_mean": _masked_mean(object_scores, cross),
            "absent_object_score_mean": _masked_mean(object_scores, absent),
            "present_frames": int((object_scores > 0).sum().item()),
            "geometry_gate_mean": float(
                adapter_aux["geometry_gate_mean"].detach().float().cpu()
            ),
            "fpn2_residual_rms": float(
                adapter_aux["fpn2_residual_rms"].detach().float().cpu()
            ),
            "valid_warp_ratio": float(warp_summary.get("valid_warp_ratio", 0.0)),
        }
    )
    return {
        "mode": mode,
        "logits": logits.detach().float().cpu(),
        "prediction": prediction.detach().cpu(),
        "object_scores": object_scores.detach().cpu(),
        "metrics": metrics,
    }


def _evaluate_controls(
    *,
    adapter,
    sam_tracker,
    sam_features,
    sam_input_images,
    sam_fpn2,
    geometry_levels,
    confidence,
    shuffled_levels,
    shuffled_confidence,
    geometry,
    target_masks,
    reference_frame_idx,
    output_size,
    residual_scale,
    device,
    reference_prompt_mode,
    image_mode,
    point_source,
    min_geometry_confidence,
    permutation,
    amp,
):
    results = {}
    for mode in ("original", "warp_only", "merger_only", "complete", "shuffled"):
        mode_levels = shuffled_levels if mode == "shuffled" else geometry_levels
        mode_confidence = (
            shuffled_confidence if mode == "shuffled" else confidence
        )
        result = _evaluate_one(
            adapter=adapter,
            mode=mode,
            sam_tracker=sam_tracker,
            sam_features=sam_features,
            sam_input_images=sam_input_images,
            sam_fpn2=sam_fpn2,
            geometry_levels=mode_levels,
            confidence=mode_confidence,
            geometry=geometry,
            target_masks=target_masks,
            reference_frame_idx=reference_frame_idx,
            output_size=output_size,
            residual_scale=residual_scale,
            device=device,
            reference_prompt_mode=reference_prompt_mode,
            image_mode=image_mode,
            point_source=point_source,
            min_geometry_confidence=min_geometry_confidence,
            permutation=permutation,
            amp=amp,
        )
        results[mode] = result
        row = result["metrics"]
        print(
            f"mode={mode:<11} cross_iou={row['cross_view_iou']:.4f} "
            f"recall={row['cross_view_recall']:.4f} "
            f"absent_fp={row['absent_fp_ratio']:.6f} "
            f"score={row['cross_object_score_mean']:.3f}/"
            f"{row['absent_object_score_mean']:.3f}"
        )
    return results


def _zero_residuals(sam_fpn2: torch.Tensor) -> list[torch.Tensor]:
    return [
        sam_fpn2.new_zeros(sam_fpn2.shape[0], 32, 288, 288),
        sam_fpn2.new_zeros(sam_fpn2.shape[0], 64, 144, 144),
        torch.zeros_like(sam_fpn2),
    ]


def _geometry_rank_loss(
    aligned_scores: torch.Tensor,
    shuffled_scores: torch.Tensor | None,
    *,
    visible: torch.Tensor,
    reference_frame_idx: int,
    margin: float,
) -> torch.Tensor:
    if shuffled_scores is None:
        return aligned_scores.new_zeros(())
    cross = visible.bool().clone()
    cross[int(reference_frame_idx)] = False
    if not cross.any():
        return aligned_scores.new_zeros(())
    aligned = aligned_scores.reshape(-1)[cross]
    shuffled = shuffled_scores.to(aligned.device).reshape(-1)[cross]
    return F.relu(float(margin) - (aligned - shuffled)).mean()


def _sigmoid_focal_loss(
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
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return (alpha_t * cross_entropy * (1.0 - p_t).pow(gamma)).mean()


def _visible_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    visible: torch.Tensor,
) -> torch.Tensor:
    if not visible.any():
        return logits.new_zeros(())
    probability = logits.float().sigmoid()[visible].flatten(1)
    selected_targets = targets.float()[visible].flatten(1)
    numerator = 2.0 * (probability * selected_targets).sum(dim=1) + 1.0
    denominator = probability.sum(dim=1) + selected_targets.sum(dim=1) + 1.0
    return (1.0 - numerator / denominator).mean()


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    if not mask.any():
        return 0.0
    return float(values[mask].mean().detach().cpu())


def _parameter_gradient_norm(parameters: Sequence[torch.nn.Parameter]) -> float:
    terms = [
        parameter.grad.detach().float().square().sum()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not terms:
        return 0.0
    return float(torch.stack(terms).sum().sqrt().cpu())


def _tensor_gradient_norm(tensor: torch.Tensor) -> float:
    if tensor.grad is None:
        return 0.0
    return float(tensor.grad.detach().float().norm().cpu())


def _autocast(device: str, enabled: bool):
    if enabled and str(device).startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _empty_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


HISTORY_FIELDS = [
    "step",
    "train_loss",
    "train_focal",
    "train_dice",
    "train_presence",
    "train_rank",
    "mean_iou",
    "positive_iou",
    "cross_view_iou",
    "cross_view_recall",
    "absent_fp_ratio",
    "cross_object_score_mean",
    "absent_object_score_mean",
    "present_frames",
    "geometry_gate_mean",
    "fpn2_residual_rms",
    "valid_warp_ratio",
    "adapter_gradient_norm",
    "residual_gradient_norm",
]


def _empty_train_values() -> dict[str, float]:
    return {
        "train_loss": 0.0,
        "train_focal": 0.0,
        "train_dice": 0.0,
        "train_presence": 0.0,
        "train_rank": 0.0,
        "adapter_gradient_norm": 0.0,
        "residual_gradient_norm": 0.0,
        "geometry_gate_mean": 0.0,
    }


def _initialize_history(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf8") as handle:
        csv.DictWriter(handle, fieldnames=HISTORY_FIELDS).writeheader()


def _append_history(path: Path, row: dict[str, Any]) -> None:
    normalized = {field: row.get(field, 0.0) for field in HISTORY_FIELDS}
    with path.open("a", newline="", encoding="utf8") as handle:
        csv.DictWriter(handle, fieldnames=HISTORY_FIELDS).writerow(normalized)


def _save_checkpoint(
    path: Path,
    *,
    step: int,
    adapter: GeometryAwareSAM3Adapter,
    layer_indices: tuple[int, ...],
    context_grid: tuple[int, int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": int(step),
            "adapter": adapter.state_dict(),
            "layer_indices": tuple(int(value) for value in layer_indices),
            "context_grid": tuple(int(value) for value in context_grid),
        },
        path,
    )


def _write_control_outputs(
    output_dir: Path,
    *,
    results,
    image_paths,
    frame_indices,
    target_masks,
    output_size,
) -> None:
    summary_rows = [
        {"mode": mode, **result["metrics"]}
        for mode, result in results.items()
    ]
    _write_csv(output_dir / "ablation_summary.csv", summary_rows)
    frame_rows = []
    targets = target_masks.detach().cpu().bool()
    for mode, result in results.items():
        for sequence_index, (prediction, target, score) in enumerate(
            zip(result["prediction"], targets, result["object_scores"])
        ):
            frame_rows.append(
                {
                    "mode": mode,
                    "sequence_index": sequence_index,
                    "frame_index": frame_indices[sequence_index],
                    "gt_visible": int(target.any()),
                    "object_score_logit": float(score),
                    "presence_gate_pass": int(float(score) > 0.0),
                    "iou": _binary_iou(prediction, target),
                    "prediction_pixels": int(prediction.sum()),
                    "target_pixels": int(target.sum()),
                }
            )
    _write_csv(output_dir / "ablation_frame_metrics.csv", frame_rows)
    _save_control_visualization(
        output_dir / "ablation_report.png",
        image_paths=image_paths,
        frame_indices=frame_indices,
        target_masks=targets,
        results=results,
        output_size=output_size,
    )


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _binary_iou(left: torch.Tensor, right: torch.Tensor) -> float:
    union = (left.bool() | right.bool()).sum()
    if int(union) == 0:
        return 1.0
    return float(
        ((left.bool() & right.bool()).sum().float() / union.float()).item()
    )


def _save_single_visualization(
    path: Path,
    *,
    image_paths,
    frame_indices,
    target_masks,
    prediction,
    object_scores,
    output_size,
) -> None:
    result = {
        "complete": {
            "prediction": prediction.detach().cpu(),
            "object_scores": object_scores.detach().cpu(),
        }
    }
    _save_control_visualization(
        path,
        image_paths=image_paths,
        frame_indices=frame_indices,
        target_masks=target_masks.detach().cpu(),
        results=result,
        output_size=output_size,
    )


def _save_control_visualization(
    path: Path,
    *,
    image_paths,
    frame_indices,
    target_masks,
    results,
    output_size,
) -> None:
    rows = []
    for frame_position, image_path in enumerate(image_paths):
        rgb = Image.open(image_path).convert("RGB").resize(
            (output_size[1], output_size[0]),
            Image.Resampling.BILINEAR,
        )
        cells = [
            _annotate(rgb, f"RGB frame={frame_indices[frame_position]}"),
            _annotate(
                _overlay(rgb, target_masks[frame_position], (0, 220, 80)),
                f"GT pixels={int(target_masks[frame_position].sum())}",
            ),
        ]
        for mode, result in results.items():
            mask = result["prediction"][frame_position]
            score = float(result["object_scores"][frame_position])
            cells.append(
                _annotate(
                    _overlay(rgb, mask, (30, 120, 255)),
                    f"{mode} score={score:.3f} pixels={int(mask.sum())}",
                )
            )
        rows.append(_concat_horizontal(cells))
    path.parent.mkdir(parents=True, exist_ok=True)
    _concat_vertical(rows).save(path)


def _overlay(image: Image.Image, mask: torch.Tensor, color) -> Image.Image:
    pixels = np.asarray(image).copy()
    selected = mask.detach().cpu().numpy().astype(bool)
    if selected.any():
        tint = np.asarray(color, dtype=np.float32)
        pixels[selected] = (
            0.55 * pixels[selected].astype(np.float32) + 0.45 * tint
        ).astype(np.uint8)
    return Image.fromarray(pixels)


def _annotate(image: Image.Image, text: str) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    draw.rectangle((0, 0, output.width, 24), fill=(0, 0, 0))
    draw.text((5, 5), text, fill=(255, 255, 255))
    return output


def _concat_horizontal(images: Sequence[Image.Image]) -> Image.Image:
    output = Image.new(
        "RGB",
        (sum(image.width for image in images), max(image.height for image in images)),
    )
    offset = 0
    for image in images:
        output.paste(image, (offset, 0))
        offset += image.width
    return output


def _concat_vertical(images: Sequence[Image.Image]) -> Image.Image:
    output = Image.new(
        "RGB",
        (max(image.width for image in images), sum(image.height for image in images)),
    )
    offset = 0
    for image in images:
        output.paste(image, (0, offset))
        offset += image.height
    return output


def _plot_training_history(csv_path: Path, output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib unavailable; skipping training_curves.png")
        return
    with csv_path.open("r", encoding="utf8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return
    steps = [int(row["step"]) for row in rows]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    for key in ("train_loss", "train_focal", "train_dice", "train_presence"):
        axes[0].plot(steps, [float(row[key]) for row in rows], label=key)
    axes[0].set_title("Training loss")
    axes[0].legend()
    axes[1].plot(
        steps,
        [float(row["cross_view_iou"]) for row in rows],
        label="cross IoU",
    )
    axes[1].plot(
        steps,
        [float(row["cross_view_recall"]) for row in rows],
        label="cross recall",
    )
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Default-gate tracking")
    axes[1].legend()
    axes[2].plot(
        steps,
        [float(row["cross_object_score_mean"]) for row in rows],
        label="visible score",
    )
    axes[2].plot(
        steps,
        [float(row["absent_object_score_mean"]) for row in rows],
        label="absent score",
    )
    axes[2].axhline(0.0, color="black", linewidth=1, linestyle="--")
    axes[2].set_title("SAM3 object-score logits")
    axes[2].legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _write_resolved_experiment(
    path: Path,
    *,
    config,
    sequence,
    layer_indices,
    context_grid,
    point_source,
    min_geometry_confidence,
    iterations,
    learning_rate,
    reference_prompt_mode,
    rank_weight,
    rank_margin,
    seed,
) -> None:
    payload = {
        "experiment": "sam3_geometry_memory_adapter",
        "scene_id": sequence.scene_id,
        "frame_indices": list(sequence.frame_indices),
        "instance_id": int(sequence.instance_id),
        "label": sequence.label,
        "reference_sequence_index": int(sequence.reference_frame_idx),
        "geometry_layers": list(layer_indices),
        "context_grid": list(context_grid),
        "geometry_point_source": point_source,
        "min_geometry_confidence": float(min_geometry_confidence),
        "iterations": int(iterations),
        "learning_rate": float(learning_rate),
        "reference_prompt_mode": reference_prompt_mode,
        "geometry_rank_weight": float(rank_weight),
        "geometry_rank_margin": float(rank_margin),
        "seed": int(seed),
        "default_presence_gate": "object_score_logits > 0",
        "trainable": ["geometry_feature_merger", "fpn2_residual_adapter"],
        "frozen": ["SAM3", "StreamVGGT"],
        "training_supervision": {
            "mask": "GT instance mask focal + Dice",
            "presence": "GT visibility BCE on SAM3 object_score_logits",
            "geometry_control": "aligned score must exceed shuffled score",
            "teacher_forcing": (
                "GT visibility keeps the raw SAM3 mask branch and memory update "
                "active during training; it is disabled for every reported "
                "inference metric"
            ),
        },
        "scientific_controls": {
            "reference_gt_prompt_only": True,
            "later_gt_used_for_training_loss_and_teacher_forcing": True,
            "later_gt_used_during_reported_inference": False,
            "hard_fallback_disabled": True,
            "redetection_disabled": True,
            "memory_writeback_disabled": True,
            "presence_threshold_lowering_disabled": True,
            "sam3_original_gate_preserved": True,
            "modes": [
                "original",
                "warp_only",
                "merger_only",
                "complete",
                "shuffled",
            ],
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the learned §3.1 StreamVGGT-to-SAM3 adapter."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--scene-id")
    parser.add_argument("--instance-id", type=int)
    parser.add_argument("--frame-indices", type=int, nargs="+")
    parser.add_argument("--sam3-device")
    parser.add_argument("--sam3-feature-device")
    parser.add_argument("--geometry-device")
    parser.add_argument("--geometry-layers", type=int, nargs="+", default=[4, 11, 17])
    parser.add_argument("--context-grid", type=int, nargs=2, default=[24, 24])
    parser.add_argument(
        "--geometry-point-source",
        choices=("depth_camera", "point_head"),
        default="depth_camera",
    )
    parser.add_argument("--min-geometry-confidence", type=float, default=0.20)
    parser.add_argument("--iterations", type=int, default=700)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--hidden-channels", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--residual-init-std", type=float, default=1e-4)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--focal-weight", type=float, default=20.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--presence-weight", type=float, default=1.0)
    parser.add_argument("--geometry-rank-weight", type=float, default=0.1)
    parser.add_argument("--geometry-rank-margin", type=float, default=0.25)
    parser.add_argument("--geometry-rank-every", type=int, default=5)
    parser.add_argument("--residual-weight", type=float, default=1e-4)
    parser.add_argument(
        "--reference-prompt-mode",
        choices=("mask", "box_points"),
        default="mask",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--visualize-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    main()
