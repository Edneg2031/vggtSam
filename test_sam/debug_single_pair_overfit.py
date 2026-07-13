#!/usr/bin/env python3
"""Diagnose and overfit one fixed query/target pair through SAM3 full flow."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
import yaml

from test_sam.coordinates import (
    compose,
    output_mask_transform,
    processed_to_grid_transform,
    sam3_resize_transform,
    streamvggt_image_transform,
)
from test_sam.data import load_mask_tracking_sequence
from test_sam.fusion import (
    ConstantPromptFusion,
    SAM3GeometryFusion,
    stream_tokens_to_maps,
)
from test_sam.train_fusion_ablation import extract_sam_features, resize_target_masks
from vggtsam.adapters.streamvggt_latent import (
    StreamVGGTLatentAdapter,
    load_streamvggt_latent_model,
)
from vggtsam.training.dense_fusion import run_sam3_source_tracker_flow


MODES = ("sam_only", "constant_prompt", "random_geometry", "real_geometry")


@dataclass(frozen=True)
class DebugConfig:
    manifest: Path
    scene_id: str
    query_frame: int
    target_frame: int
    instance_id: int
    sam3_repo: Path
    sam3_checkpoint: Path
    sam3_feature_device: str
    tracker_device: str
    sam3_resolution: int
    sam3_frame_chunk_size: int
    reference_prompt_mode: str
    streamvggt_repo: Path
    streamvggt_checkpoint: Path
    geometry_device: str
    geometry_streaming_cache: bool
    geometry_image_mode: str
    context_grid: tuple[int, int]
    layer_indices: tuple[int, ...]
    mode: str
    iterations: int
    lr: float
    seed: int
    presence_weight: float
    gradient_clip: float
    log_every: int
    visualize_every: int
    save_every: int
    output_size: tuple[int, int]
    residual_init_std: float
    output_dir: Path

    @property
    def fusion_method(self) -> str:
        return "sam_only" if self.mode in {"sam_only", "constant_prompt"} else "cross_attention"

    @property
    def zero_geometry(self) -> bool:
        return False

    @property
    def shuffle_geometry(self) -> bool:
        return False

    @property
    def compare_direct(self) -> bool:
        return False


def main() -> None:
    args = parse_args()
    raw = load_yaml(args.config)
    apply_overrides(raw, args)
    run(build_config(raw))


def run(config: DebugConfig) -> None:
    if config.mode not in MODES:
        raise ValueError(f"Unknown mode {config.mode!r}; choose from {MODES}")
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(config.output_dir / "resolved_config.json", config_to_dict(config))

    sequence = load_mask_tracking_sequence(
        config.manifest,
        scene_id=config.scene_id,
        frame_indices=[config.query_frame, config.target_frame],
        sequence_length=2,
        frame_stride=1,
        window_index=0,
        instance_id=config.instance_id,
        min_pixels=1,
        max_area_ratio=1.0,
        min_visible_frames=1,
        excluded_labels=[],
        seed=config.seed,
    )
    sequence = replace(sequence, reference_frame_idx=0)
    target_masks = resize_target_masks(sequence, config.output_size)
    if not bool(target_masks[0].any()) or not bool(target_masks[1].any()):
        raise RuntimeError(
            "Single-pair overfit requires the target instance in both frames; "
            f"pixels={[int(mask.sum()) for mask in target_masks]}."
        )

    sam_features, sam_input_images, sam_tracker = extract_sam_features(config, sequence)
    sam_fpn2 = sam_features["fpn2"].float()
    assert_shape(sam_fpn2, (2, 256, 72, 72), "sam_fpn2")
    geometry_levels, geometry_meta = build_geometry(config, sequence.image_paths)
    model = build_model(config, sam_fpn2, geometry_levels).to(config.tracker_device)
    sam_tracker.requires_grad_(False)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("Debug adapter has no trainable parameters.")
    optimizer = torch.optim.AdamW(parameters, lr=config.lr)
    audit_parameters(model, optimizer, config.output_dir / "parameter_audit.csv")

    device = torch.device(config.tracker_device)
    target_masks = target_masks.to(device)
    sam_input_images = sam_input_images.to(device)
    sam_fpn2 = sam_fpn2.to(device)
    if geometry_levels is not None:
        geometry_levels = [level.to(device) for level in geometry_levels]

    history_path = config.output_dir / "training_history.csv"
    module_path = config.output_dir / "module_diagnostics.csv"
    initialize_csv(history_path, HISTORY_FIELDS)
    initialize_csv(module_path, MODULE_FIELDS)
    tensor_audit_written = False

    for step in range(1, config.iterations + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        before = snapshot_parameters(model)
        residuals = forward_adapter(model, config.mode, sam_fpn2, geometry_levels)
        assert_residual_shapes(residuals, frames=2)
        for residual in residuals:
            if residual.requires_grad:
                residual.retain_grad()

        logits, aux = run_sam3_source_tracker_flow(
            sam_tracker=sam_tracker,
            sam_tracker_features=sam_features,
            sam_input_images=sam_input_images,
            tracker_residuals=residuals,
            reference_mask=target_masks[0],
            reference_frame_idx=0,
            output_size=config.output_size,
            residual_scale=1.0,
            device=config.tracker_device,
            reference_prompt_mode=config.reference_prompt_mode,
            training_target_masks=target_masks,
        )
        assert_shape(logits, (2, *config.output_size), "train_logits")
        if not logits.requires_grad:
            raise RuntimeError("SAM3 training logits are detached from the adapter.")
        target_logits = logits[1]
        target_gt = target_masks[1].float()
        assert_shape(target_logits, config.output_size, "target_logits")
        assert_shape(target_gt, config.output_size, "target_gt")
        bce = F.binary_cross_entropy_with_logits(target_logits.float(), target_gt)
        dice = binary_dice_loss(target_logits, target_gt)
        object_score = aux["object_score_logits"].reshape(-1)[1]
        presence = F.binary_cross_entropy_with_logits(
            object_score.float().reshape(()),
            object_score.new_ones(()).float(),
        )
        loss = bce + dice + config.presence_weight * presence
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {float(loss.detach())}")
        loss.backward()

        adapter_grad_norm = gradient_norm(model.parameters())
        residual_grad_norm = gradient_norm(residuals)
        if step == 1 and not tensor_audit_written:
            write_tensor_audit(
                config.output_dir / "tensor_audit.json",
                config=config,
                sequence=sequence,
                sam_features=sam_features,
                sam_fpn2=sam_fpn2,
                geometry_levels=geometry_levels,
                geometry_meta=geometry_meta,
                target_masks=target_masks,
                train_logits=logits,
                residuals=residuals,
            )
            tensor_audit_written = True
        if step == 1 and adapter_grad_norm <= 1e-12:
            append_module_diagnostics(module_path, step, model, before)
            raise RuntimeError(
                "No adapter gradient at step 1: "
                f"adapter_grad={adapter_grad_norm}, residual_grad={residual_grad_norm}."
            )
        if config.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(parameters, config.gradient_clip)
        optimizer.step()
        append_module_diagnostics(module_path, step, model, before)

        should_log = step == 1 or step % config.log_every == 0 or step == config.iterations
        if not should_log:
            continue
        model.eval()
        with torch.no_grad():
            eval_residuals = forward_adapter(model, config.mode, sam_fpn2, geometry_levels)
            eval_logits, eval_aux = run_sam3_source_tracker_flow(
                sam_tracker=sam_tracker,
                sam_tracker_features=sam_features,
                sam_input_images=sam_input_images,
                tracker_residuals=eval_residuals,
                reference_mask=target_masks[0],
                reference_frame_idx=0,
                output_size=config.output_size,
                residual_scale=1.0,
                device=config.tracker_device,
                reference_prompt_mode=config.reference_prompt_mode,
            )
        train_iou = mask_iou(target_logits.detach(), target_gt, logits=True)
        eval_iou = mask_iou(eval_logits[1], target_gt, logits=True)
        train_pixels = int((target_logits.detach().sigmoid() >= 0.5).sum().cpu())
        eval_pixels = int((eval_logits[1].sigmoid() >= 0.5).sum().cpu())
        gt_pixels = int(target_gt.sum().cpu())
        adapter_output_norm = tensor_norm(eval_residuals[-1])
        sam_feature_norm = tensor_norm(sam_fpn2)
        geometry_feature_norm = (
            tensor_norm(geometry_levels[-1]) if geometry_levels is not None else 0.0
        )
        fusion_residual_ratio = adapter_output_norm / (sam_feature_norm + 1e-12)
        row = {
            "step": step,
            "total_loss": float(loss.detach().cpu()),
            "bce_loss": float(bce.detach().cpu()),
            "dice_loss": float(dice.detach().cpu()),
            "presence_loss": float(presence.detach().cpu()),
            "train_iou": train_iou,
            "eval_iou": eval_iou,
            "predicted_foreground_pixels": train_pixels,
            "eval_foreground_pixels": eval_pixels,
            "gt_foreground_pixels": gt_pixels,
            "adapter_gradient_norm": adapter_grad_norm,
            "residual_gradient_norm": residual_grad_norm,
            "adapter_output_norm": adapter_output_norm,
            "sam_feature_norm": sam_feature_norm,
            "geometry_feature_norm": geometry_feature_norm,
            "fusion_residual_ratio": fusion_residual_ratio,
            "train_object_score": float(object_score.detach().cpu()),
            "eval_object_score": float(eval_aux["object_score_logits"].reshape(-1)[1].cpu()),
        }
        append_csv(history_path, row, HISTORY_FIELDS)
        print(
            f"step={step} loss={row['total_loss']:.6f} bce={row['bce_loss']:.6f} "
            f"dice={row['dice_loss']:.6f} presence={row['presence_loss']:.6f} "
            f"train_iou={train_iou:.4f} eval_iou={eval_iou:.4f} "
            f"pixels={train_pixels}/{gt_pixels} grad={adapter_grad_norm:.6f} "
            f"residual_grad={residual_grad_norm:.6f} "
            f"residual_ratio={fusion_residual_ratio:.6f}"
        )

        if step % config.visualize_every == 0 or step == config.iterations:
            save_visualization(
                config.output_dir / "visualizations" / f"step_{step:04d}.jpg",
                sequence.image_paths,
                target_masks,
                target_logits.detach(),
                eval_logits[1],
                config.output_size,
                step,
                train_iou,
                eval_iou,
            )
        if step % config.save_every == 0 or step == config.iterations:
            checkpoint_dir = config.output_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"step": step, "model": model.state_dict(), "config": config_to_dict(config)},
                checkpoint_dir / f"step_{step:04d}.pt",
            )

    print(f"history: {history_path}")
    print(f"diagnostics: {module_path}")


def build_geometry(config: DebugConfig, image_paths) -> tuple[list[torch.Tensor] | None, dict]:
    if config.mode in {"sam_only", "constant_prompt"}:
        return None, {"loaded": False}
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
            image_paths,
            return_pointmap=False,
            streaming_cache=config.geometry_streaming_cache,
        )
    raw_levels = output.geometry.aux["stream_dpt_tokens"]
    levels = stream_tokens_to_maps(
        raw_levels,
        patch_start_idx=int(output.geometry.aux["patch_start_idx"]),
        patch_shape=tuple(output.aux["patch_shape"]),
        output_grid=config.context_grid,
    )
    real = levels[-1].float().cpu()
    if config.mode == "random_geometry":
        generator = torch.Generator(device="cpu").manual_seed(config.seed)
        random_feature = torch.randn(real.shape, generator=generator)
        real_std = real.std().clamp_min(1e-6)
        real = random_feature * real_std + real.mean()
    meta = {
        "loaded": True,
        "image_shape": list(output.aux["image_shape"]),
        "patch_shape": list(output.aux["patch_shape"]),
        "patch_start_idx": int(output.geometry.aux["patch_start_idx"]),
        "source": config.mode,
    }
    del output, adapter, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return [real], meta


def build_model(config, sam_fpn2, geometry_levels):
    if config.mode == "constant_prompt":
        return ConstantPromptFusion(
            sam_channels=int(sam_fpn2.shape[1]),
            residual_init_std=config.residual_init_std,
        )
    return SAM3GeometryFusion(
        method="sam_only" if config.mode == "sam_only" else "cross_attention",
        sam_channels=int(sam_fpn2.shape[1]),
        geometry_channels=(2048 if geometry_levels is None else int(geometry_levels[-1].shape[1])),
        inject_levels=("fpn2",),
        residual_init_std=config.residual_init_std,
    )


def forward_adapter(model, mode, sam_fpn2, geometry_levels):
    if mode == "constant_prompt":
        return model(sam_fpn2)
    return model(sam_fpn2, geometry_levels)


def audit_parameters(model, optimizer, path: Path) -> None:
    optimizer_membership = {
        id(parameter): (group_index, float(group["lr"]))
        for group_index, group in enumerate(optimizer.param_groups)
        for parameter in group["params"]
    }
    optimizer_ids = set(optimizer_membership)
    trainable_ids = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if optimizer_ids != trainable_ids:
        raise RuntimeError(
            f"Optimizer mismatch: missing={len(trainable_ids - optimizer_ids)} "
            f"unexpected={len(optimizer_ids - trainable_ids)}"
        )
    fields = [
        "name", "shape", "numel", "requires_grad", "in_optimizer",
        "optimizer_group", "learning_rate", "dtype",
    ]
    initialize_csv(path, fields)
    for name, parameter in model.named_parameters():
        append_csv(
            path,
            {
                "name": name,
                "shape": list(parameter.shape),
                "numel": parameter.numel(),
                "requires_grad": parameter.requires_grad,
                "in_optimizer": id(parameter) in optimizer_ids,
                "optimizer_group": (
                    optimizer_membership[id(parameter)][0]
                    if id(parameter) in optimizer_membership else ""
                ),
                "learning_rate": (
                    optimizer_membership[id(parameter)][1]
                    if id(parameter) in optimizer_membership else ""
                ),
                "dtype": str(parameter.dtype),
            },
            fields,
        )
    print(
        f"trainable_parameters={sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)}"
    )


def append_module_diagnostics(path, step, model, before) -> None:
    groups: dict[str, list[tuple[str, torch.nn.Parameter]]] = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            groups.setdefault(name.split(".", 1)[0], []).append((name, parameter))
    for module, items in groups.items():
        parameters = [parameter for _, parameter in items]
        updates = [parameter.detach() - before[name] for name, parameter in items]
        invalid = any(not torch.isfinite(parameter).all() for parameter in parameters)
        invalid = invalid or any(
            parameter.grad is not None and not torch.isfinite(parameter.grad).all()
            for parameter in parameters
        )
        append_csv(
            path,
            {
                "step": step,
                "module": module,
                "parameter_norm": tensor_list_norm(parameters),
                "gradient_norm": gradient_norm(parameters),
                "update_norm": tensor_list_norm(updates),
                "has_nan_or_inf": invalid,
            },
            MODULE_FIELDS,
        )


def write_tensor_audit(path, *, config, sequence, sam_features, sam_fpn2, geometry_levels, geometry_meta, target_masks, train_logits, residuals):
    transforms = []
    for image_path in sequence.image_paths:
        with Image.open(image_path) as image:
            source_size = (image.height, image.width)
        sam_image = sam3_resize_transform(source_size, resolution=config.sam3_resolution)
        sam_grid = processed_to_grid_transform(
            sam_image.target_size,
            (72, 72),
            description="SAM3 FPN2 grid",
        )
        item = {
            "image": str(image_path),
            "source_size": source_size,
            "sam3_image": sam_image.to_dict(),
            "source_to_sam_fpn2": compose(sam_image, sam_grid).to_dict(),
            "source_to_output_mask": output_mask_transform(source_size, config.output_size).to_dict(),
        }
        if geometry_meta.get("loaded"):
            stream_image = streamvggt_image_transform(source_size, mode=config.geometry_image_mode)
            actual_shape = tuple(int(value) for value in geometry_meta["image_shape"])
            if stream_image.target_size != actual_shape:
                raise RuntimeError(
                    f"StreamVGGT transform mismatch: computed={stream_image.target_size}, actual={actual_shape}"
                )
            stream_grid = processed_to_grid_transform(
                actual_shape,
                tuple(geometry_meta["patch_shape"]),
                description="StreamVGGT native patch grid",
            )
            item["streamvggt_image"] = stream_image.to_dict()
            item["source_to_stream_patch"] = compose(stream_image, stream_grid).to_dict()
        transforms.append(item)
    write_json(
        path,
        {
            "mode": config.mode,
            "sam_features": {key: tensor_summary(value) for key, value in sam_features.items()},
            "sam_fpn2": tensor_summary(sam_fpn2),
            "geometry": [tensor_summary(value) for value in geometry_levels] if geometry_levels else None,
            "geometry_meta": geometry_meta,
            "target_masks": tensor_summary(target_masks),
            "train_logits": tensor_summary(train_logits),
            "residuals": [tensor_summary(value) for value in residuals],
            "coordinate_transforms": transforms,
            "preprocessing": {
                "sam3": "RGB -> stretch 1008x1008 -> float [0,1] -> normalize mean=std=0.5",
                "streamvggt": (
                    f"RGB -> width/long-edge 518 with aspect preservation, "
                    f"14-multiple rounding, mode={config.geometry_image_mode}, float [0,1]"
                ),
                "gt_mask": "original instance-id PNG -> nearest resize to output_size",
            },
            "loss_contract": "binary_cross_entropy_with_logits receives raw SAM3 logits exactly once",
        },
    )


def binary_dice_loss(logits, target):
    probability = logits.float().sigmoid().flatten()
    target = target.float().flatten()
    return 1.0 - (2.0 * (probability * target).sum() + 1.0) / (
        probability.sum() + target.sum() + 1.0
    )


def mask_iou(prediction, target, *, logits):
    pred = prediction.float().sigmoid() >= 0.5 if logits else prediction.bool()
    target = target.bool()
    intersection = (pred & target).sum().float()
    union = (pred | target).sum().float()
    return float((intersection / union.clamp_min(1)).cpu())


def snapshot_parameters(model):
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def gradient_norm(values):
    tensors = []
    for value in values:
        gradient = value.grad if isinstance(value, torch.Tensor) else None
        if gradient is not None:
            tensors.append(gradient)
    return tensor_list_norm(tensors)


def tensor_list_norm(values):
    squared = [value.detach().float().square().sum() for value in values]
    return float(torch.stack(squared).sum().sqrt().cpu()) if squared else 0.0


def tensor_norm(value):
    return float(value.detach().float().norm().cpu())


def tensor_summary(value):
    tensor = value.detach().float()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "requires_grad": value.requires_grad,
        "min": float(tensor.min().cpu()),
        "max": float(tensor.max().cpu()),
        "mean": float(tensor.mean().cpu()),
        "std": float(tensor.std().cpu()),
        "finite": bool(torch.isfinite(tensor).all()),
    }


def assert_residual_shapes(residuals, *, frames):
    expected = [(frames, 32, 288, 288), (frames, 64, 144, 144), (frames, 256, 72, 72)]
    if len(residuals) != 3:
        raise RuntimeError(f"Expected three FPN residuals, got {len(residuals)}")
    for index, (value, shape) in enumerate(zip(residuals, expected)):
        assert_shape(value, shape, f"residual_fpn{index}")


def assert_shape(value, expected, name):
    if tuple(value.shape) != tuple(expected):
        raise RuntimeError(f"{name} shape mismatch: expected={expected}, actual={tuple(value.shape)}")


def save_visualization(path, image_paths, target_masks, train_logits, eval_logits, output_size, step, train_iou, eval_iou):
    path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_paths[0]) as image:
        query = image.convert("RGB").resize((output_size[1], output_size[0]))
    with Image.open(image_paths[1]) as image:
        target = image.convert("RGB").resize((output_size[1], output_size[0]))
    gt_query = target_masks[0].detach().cpu().numpy().astype(bool)
    gt_target = target_masks[1].detach().cpu().numpy().astype(bool)
    train_pred = (train_logits.sigmoid() >= 0.5).cpu().numpy()
    eval_pred = (eval_logits.sigmoid() >= 0.5).cpu().numpy()
    panels = [
        annotate(overlay(query, gt_query, (0, 220, 0)), "Query + GT prompt"),
        annotate(overlay(target, gt_target, (0, 220, 0)), "Target GT"),
        annotate(overlay(target, train_pred, (230, 60, 60)), f"Teacher-forced IoU={train_iou:.3f}"),
        annotate(overlay(target, eval_pred, (40, 120, 255)), f"Full-flow IoU={eval_iou:.3f}"),
    ]
    canvas = Image.new("RGB", (sum(panel.width for panel in panels), panels[0].height))
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width
    canvas.save(path, quality=92)


def overlay(image, mask, color):
    array = np.asarray(image).copy()
    array[mask] = (0.5 * array[mask] + 0.5 * np.asarray(color)).astype(np.uint8)
    return Image.fromarray(array)


def annotate(image, text):
    output = image.copy()
    draw = ImageDraw.Draw(output)
    draw.rectangle((0, 0, output.width, 26), fill=(0, 0, 0))
    draw.text((6, 6), text, fill=(255, 255, 255))
    return output


def initialize_csv(path, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf8") as handle:
        csv.DictWriter(handle, fieldnames=fields).writeheader()


def append_csv(path, row, fields):
    with path.open("a", newline="", encoding="utf8") as handle:
        csv.DictWriter(handle, fieldnames=fields).writerow(row)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as handle:
        json.dump(value, handle, indent=2)


def config_to_dict(config):
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in config.__dict__.items()
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("test_sam/debug_single_pair.yaml"))
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--query-frame", type=int)
    parser.add_argument("--target-frame", type=int)
    parser.add_argument("--tracker-device")
    parser.add_argument("--sam3-feature-device")
    parser.add_argument("--geometry-device")
    return parser.parse_args()


def apply_overrides(raw, args):
    mappings = [
        ("mode", "training", "mode"),
        ("iterations", "training", "iterations"),
        ("output_dir", "training", "output_dir"),
        ("query_frame", "dataset", "query_frame"),
        ("target_frame", "dataset", "target_frame"),
        ("tracker_device", "sam3", "tracker_device"),
        ("sam3_feature_device", "sam3", "feature_device"),
        ("geometry_device", "streamvggt", "device"),
    ]
    for argument, section, key in mappings:
        value = getattr(args, argument)
        if value is not None:
            raw.setdefault(section, {})[key] = value


def load_yaml(path):
    with path.open("r", encoding="utf8") as handle:
        return yaml.safe_load(handle)


def build_config(raw):
    dataset, sam, stream, training = raw["dataset"], raw["sam3"], raw["streamvggt"], raw["training"]
    return DebugConfig(
        manifest=Path(dataset["manifest"]), scene_id=str(dataset["scene_id"]),
        query_frame=int(dataset["query_frame"]), target_frame=int(dataset["target_frame"]),
        instance_id=int(dataset["instance_id"]), sam3_repo=Path(sam["repo"]),
        sam3_checkpoint=Path(sam["checkpoint"]), sam3_feature_device=str(sam["feature_device"]),
        tracker_device=str(sam["tracker_device"]), sam3_resolution=int(sam.get("resolution", 1008)),
        sam3_frame_chunk_size=1, reference_prompt_mode=str(sam.get("reference_prompt_mode", "mask")),
        streamvggt_repo=Path(stream["repo"]), streamvggt_checkpoint=Path(stream["checkpoint"]),
        geometry_device=str(stream["device"]), geometry_streaming_cache=bool(stream.get("streaming_cache", True)),
        geometry_image_mode=str(stream.get("image_mode", "crop")),
        context_grid=tuple(int(v) for v in stream.get("context_grid", [12, 12])),
        layer_indices=tuple(int(v) for v in stream.get("layer_indices", [4, 11, 17, 23])),
        mode=str(training.get("mode", "sam_only")), iterations=int(training.get("iterations", 1000)),
        lr=float(training.get("lr", 3e-4)), seed=int(training.get("seed", 0)),
        presence_weight=float(training.get("presence_weight", 1.0)),
        gradient_clip=float(training.get("gradient_clip", 1.0)), log_every=int(training.get("log_every", 20)),
        visualize_every=int(training.get("visualize_every", 100)), save_every=int(training.get("save_every", 200)),
        output_size=tuple(int(v) for v in training.get("output_size", [256, 384])),
        residual_init_std=float(training.get("residual_init_std", 1e-4)),
        output_dir=Path(training.get("output_dir", "outputs/debug_single_pair")),
    )


HISTORY_FIELDS = [
    "step", "total_loss", "bce_loss", "dice_loss", "presence_loss", "train_iou", "eval_iou",
    "predicted_foreground_pixels", "eval_foreground_pixels", "gt_foreground_pixels",
    "adapter_gradient_norm", "residual_gradient_norm", "adapter_output_norm", "sam_feature_norm",
    "geometry_feature_norm", "fusion_residual_ratio", "train_object_score", "eval_object_score",
]
MODULE_FIELDS = ["step", "module", "parameter_norm", "gradient_norm", "update_norm", "has_nan_or_inf"]


if __name__ == "__main__":
    main()
