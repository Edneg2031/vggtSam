#!/usr/bin/env python3
"""Train and evaluate DINOv3-conditioned object geometry residuals.

The experiment is intentionally cross-scene and test-blind:

* StreamVGGT, its point/camera heads, and SAM3 masks are read from frozen
  caches;
* DINOv3 object features are read from the previously generated cache;
* only this small residual head is optimized on configured train scenes;
* checkpoint selection uses the validation scene only;
* the test scene is loaded and scored after all checkpoints are frozen.

Branches are the required causal comparison:

``raw``
    Original frozen world pointmap.
``geometry_only``
    Residual head with the SAM object write gate but no DINO condition.
``single_view_dino``
    Current-frame masked DINOv3 object feature.
``persistent_dino``
    Causal EMA feature aggregated by SAM track identity.
``shuffled_persistent_dino``
    Same feature budget with deterministic wrong-slot identity control.

The residual is applied only on the SAM object-mask union.  Background pixels
are therefore an exact raw fallback for every learned branch.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import gc
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence

import torch

from streaming_couping.src.dinov3_object_geometry import (
    ObjectConditionedResidualHead,
    apply_similarity,
    invert_similarity,
    point_head_patch_features,
    robust_point_loss,
)
from streaming_couping.src.learned_pose.cache import cache_path, load_feature_cache
from streaming_couping.src.learned_pose.config import ClipConfig, LearnedPoseConfig, load_learned_pose_config
from streaming_couping.src.semantic_map import normalize_confidence
from streaming_couping.src.semantic_map_metrics import (
    deterministic_limit_pairs,
    deterministic_limit_points,
    load_ground_truth_stream_masks,
    object_point_metrics,
)
from streaming_couping.src.semantic_tracking_metrics import load_ground_truth_instances


REVISION = "dinov3_object_conditioned_geometry_cross_scene_r1"
BRANCHES = (
    "raw",
    "geometry_only",
    "single_view_dino",
    "persistent_dino",
    "shuffled_persistent_dino",
)
LEARNED_BRANCHES = BRANCHES[1:]


@dataclass
class ClipData:
    name: str
    scene_id: str
    split: str
    frame_indices: tuple[int, ...]
    prompts: tuple[str, ...]
    image_mode: str
    patch_shape: tuple[int, int]
    image_size: tuple[int, int]
    features: torch.Tensor
    raw_native: torch.Tensor
    target_native: torch.Tensor
    target_metric: torch.Tensor
    support: torch.Tensor
    masks: torch.Tensor
    single_features: torch.Tensor
    single_valid: torch.Tensor
    persistent_features: torch.Tensor
    persistent_valid: torch.Tensor
    shuffled_features: torch.Tensor
    shuffled_valid: torch.Tensor
    scale: float
    rotation: torch.Tensor
    translation: torch.Tensor
    manifest: Path
    cache_path: Path
    dino_path: Path


@dataclass(frozen=True)
class Example:
    clip_index: int
    frame_index: int


def main() -> None:
    args = _parse_args()
    _seed_everything(int(args.seed))
    protocol = Path(args.protocol).expanduser().resolve()
    feature_dir = Path(args.feature_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not protocol.is_file():
        raise FileNotFoundError(f"Multi-scene protocol is missing: {protocol}")
    if not feature_dir.is_dir():
        raise FileNotFoundError(f"DINOv3 feature cache directory is missing: {feature_dir}")

    config = load_learned_pose_config(protocol)
    train_clips = tuple(clip for clip in config.clips if clip.split == "train")
    validation_clips = tuple(
        clip for clip in config.clips if clip.split == "validation"
    )
    test_clips = tuple(clip for clip in config.clips if clip.split == "test")
    if not train_clips or not validation_clips or not test_clips:
        raise ValueError(
            "DINOv3 object geometry requires non-empty train/validation/test "
            f"scene splits; got train={len(train_clips)} "
            f"validation={len(validation_clips)} test={len(test_clips)}."
        )

    device = _resolve_device(args.device)
    print("DINOv3 OBJECT-CONDITIONED GEOMETRY")
    print(f"protocol={protocol}")
    print(f"feature_dir={feature_dir}")
    print(
        "frozen=StreamVGGT backbone/PointHead/CameraHead + SAM3 + DINOv3; "
        "trainable=small residual head"
    )
    print(
        f"device={device} epochs={args.epochs} lr={args.learning_rate} "
        f"object_mask_only=1 ema_feature_cache=frozen"
    )
    print(
        "splits="
        f"train:{','.join(clip.name for clip in train_clips)} "
        f"validation:{','.join(clip.name for clip in validation_clips)} "
        f"test:{','.join(clip.name for clip in test_clips)}"
    )

    train_data = [
        _load_clip_data(config, clip, feature_dir, confidence_threshold=args.confidence_threshold)
        for clip in train_clips
    ]
    validation_data = [
        _load_clip_data(config, clip, feature_dir, confidence_threshold=args.confidence_threshold)
        for clip in validation_clips
    ]
    all_loaded = (*train_data, *validation_data)
    _validate_model_compatibility(all_loaded)
    train_examples = _build_examples(train_data, minimum_points=args.minimum_object_points)
    validation_examples = _build_examples(
        validation_data, minimum_points=args.minimum_object_points
    )
    if not train_examples or not validation_examples:
        raise ValueError(
            "No train/validation frames have enough valid SAM-object points: "
            f"train={len(train_examples)} validation={len(validation_examples)}."
        )
    print(
        f"loaded train_frames={len(train_examples)} validation_frames={len(validation_examples)} "
        f"feature_channels={all_loaded[0].features.shape[-1]} "
        f"dino_dim={all_loaded[0].single_features.shape[-1]}"
    )

    model_spec = _model_spec(all_loaded[0], args)
    states: dict[str, dict[str, Any]] = {}
    training_rows: list[dict[str, Any]] = []
    selections: dict[str, dict[str, Any]] = {}
    for branch in LEARNED_BRANCHES:
        _seed_everything(int(args.seed))
        model = ObjectConditionedResidualHead(**model_spec).to(device)
        zero_error = _zero_update_error(model, train_data[0], 0, branch, device)
        if zero_error != 0.0:
            raise RuntimeError(
                f"Raw fallback is not exact at initialization for {branch}: {zero_error}."
            )
        print(f"  training branch={branch}")
        rows, state, selection = _train_branch(
            model,
            branch=branch,
            train_data=train_data,
            train_examples=train_examples,
            validation_data=validation_data,
            validation_examples=validation_examples,
            device=device,
            args=args,
        )
        training_rows.extend(rows)
        states[branch] = {
            "state_dict": state,
            "checkpoint_selection": selection,
            "zero_update_dense_maximum": zero_error,
        }
        selections[branch] = selection
        print(
            f"    selected epoch={selection['best_epoch'] + 1} "
            f"val_object_rmse={selection['best_validation_rmse']:.6f} "
            f"val_object_p90={selection['best_validation_p90']:.6f}"
        )
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    model_payload = {
        "schema": 1,
        "revision": REVISION,
        "protocol": str(protocol),
        "feature_dir": str(feature_dir),
        "model": model_spec,
        "branches": states,
        "checkpoint_selection": "validation_object_support_rmse_then_p90",
        "test_metrics_read_during_selection": 0,
        "backbone_parameters_updated": 0,
        "point_head_parameters_updated": 0,
        "camera_head_parameters_updated": 0,
        "sam_parameters_updated": 0,
        "dinov3_parameters_updated": 0,
    }
    torch.save(model_payload, output_dir / "models.pt")

    # The test cache and its GT-bearing fields are opened only after every
    # learned branch has a frozen validation-selected state.
    print("  validation checkpoints frozen; opening sealed test split")
    test_data = [
        _load_clip_data(config, clip, feature_dir, confidence_threshold=args.confidence_threshold)
        for clip in test_clips
    ]
    _validate_model_compatibility((*validation_data, *test_data))

    metric_rows: list[dict[str, Any]] = []
    object_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    for split, split_data in (
        ("validation", validation_data),
        ("test", test_data),
    ):
        for data in split_data:
            result = _evaluate_clip(
                data,
                states=states,
                model_spec=model_spec,
                device=device,
                args=args,
                split=split,
                output_dir=output_dir,
            )
            metric_rows.extend(result["metrics"])
            object_rows.extend(result["objects"])
            frame_rows.extend(result["frames"])
            _print_clip_metrics(result, split)

    summary = _build_summary(
        protocol=protocol,
        feature_dir=feature_dir,
        model_spec=model_spec,
        selections=selections,
        metric_rows=metric_rows,
        object_rows=object_rows,
        frame_rows=frame_rows,
        train_clips=train_clips,
        validation_clips=validation_clips,
        test_clips=test_clips,
    )
    _write_json(output_dir / "summary.json", summary)
    _write_csv(output_dir / "training_curve.csv", training_rows)
    _write_csv(output_dir / "branch_summary.csv", metric_rows)
    _write_csv(output_dir / "object_metrics.csv", object_rows)
    _write_csv(output_dir / "frame_metrics.csv", frame_rows)
    _write_copyable(output_dir / "copyable_result.txt", summary)
    print(
        f"DINOv3 object geometry result={output_dir / 'summary.json'} "
        f"decision={summary['decision']['overall']}"
    )


def _load_clip_data(
    config: LearnedPoseConfig,
    clip: ClipConfig,
    feature_dir: Path,
    *,
    confidence_threshold: float,
) -> ClipData:
    source = cache_path(config, clip)
    dino_path = feature_dir / f"{clip.name}.pt"
    if not source.is_file():
        raise FileNotFoundError(f"Frozen StreamVGGT/SAM cache is missing: {source}")
    if not dino_path.is_file():
        raise FileNotFoundError(f"DINOv3 object feature cache is missing: {dino_path}")
    payload = load_feature_cache(source)
    dino = torch.load(dino_path, map_location="cpu", weights_only=False)
    if not isinstance(dino, Mapping):
        raise ValueError(f"DINO cache is not a mapping: {dino_path}")
    if str(dino.get("clip", "")) != clip.name:
        raise ValueError(f"DINO cache clip mismatch: {dino_path}")

    token_levels = payload["token_levels"]
    patch_shape = tuple(int(value) for value in payload["patch_shape"])
    features = point_head_patch_features(
        token_levels,
        patch_start_idx=int(payload["patch_start_idx"]),
        patch_shape=patch_shape,
    ).half().contiguous()
    raw_native = torch.as_tensor(payload["baseline_world_points"]).float().cpu()
    target_metric = torch.as_tensor(payload["target_world_points"]).float().cpu()
    if raw_native.ndim != 4 or raw_native.shape[-1] != 3:
        raise ValueError(f"Invalid raw pointmap shape in {source}: {raw_native.shape}")
    if tuple(target_metric.shape) != tuple(raw_native.shape):
        raise ValueError(f"Raw/target pointmap shapes disagree in {source}.")
    scale = float(payload["point_alignment_scale"])
    rotation = torch.as_tensor(payload["point_alignment_rotation"]).float().cpu()
    translation = torch.as_tensor(payload["point_alignment_translation"]).float().cpu()
    target_native = invert_similarity(
        target_metric,
        scale=scale,
        rotation=rotation,
        translation=translation,
    ).float().cpu()
    confidence = normalize_confidence(
        torch.as_tensor(payload["baseline_world_confidence"]).float().cpu()
    )
    if confidence.ndim == 4 and confidence.shape[-1] == 1:
        confidence = confidence[..., 0]
    if tuple(confidence.shape) != tuple(raw_native.shape[:-1]):
        raise ValueError(f"Point confidence shape disagrees in {source}: {confidence.shape}")
    masks = _load_raw_masks(payload, str(dino.get("raw_cache_variant", "sam31_online_forward")))
    if tuple(masks.shape[0:1] + masks.shape[2:]) != tuple(raw_native.shape[:-1]):
        raise ValueError(
            f"SAM mask/pointmap shape mismatch in {source}: masks={masks.shape} raw={raw_native.shape}"
        )
    single = _tensor_3d(dino, "single_features", dino_path)
    persistent = _tensor_3d(dino, "persistent_features", dino_path)
    shuffled = _tensor_3d(dino, "shuffled_persistent_features", dino_path)
    single_valid = _tensor_2d(dino, "single_valid", dino_path)
    persistent_valid = _tensor_2d(dino, "persistent_valid", dino_path)
    shuffled_valid = _tensor_2d(dino, "shuffled_persistent_valid", dino_path)
    expected_feature_shape = (int(raw_native.shape[0]), int(masks.shape[1]))
    for name, value in (
        ("single_features", single),
        ("persistent_features", persistent),
        ("shuffled_persistent_features", shuffled),
    ):
        if tuple(value.shape[:2]) != expected_feature_shape:
            raise ValueError(f"{name} shape does not match cache slots: {value.shape}")
    for name, value in (
        ("single_valid", single_valid),
        ("persistent_valid", persistent_valid),
        ("shuffled_persistent_valid", shuffled_valid),
    ):
        if tuple(value.shape) != expected_feature_shape:
            raise ValueError(f"{name} shape does not match cache slots: {value.shape}")
    support = (
        torch.isfinite(raw_native).all(dim=-1)
        & torch.isfinite(target_native).all(dim=-1)
        & torch.isfinite(target_metric).all(dim=-1)
        & torch.isfinite(confidence)
        & (confidence >= float(confidence_threshold))
    )
    return ClipData(
        name=clip.name,
        scene_id=str(payload["scene_id"]),
        split=str(clip.split),
        frame_indices=tuple(int(value) for value in payload["frame_indices"]),
        prompts=tuple(str(value) for value in payload.get("instance_prompts", clip.instance_prompts)),
        image_mode=str(payload.get("image_mode", "crop")),
        patch_shape=patch_shape,
        image_size=(int(raw_native.shape[1]), int(raw_native.shape[2])),
        features=features,
        raw_native=raw_native,
        target_native=target_native,
        target_metric=target_metric,
        support=support,
        masks=masks,
        single_features=single,
        single_valid=single_valid,
        persistent_features=persistent,
        persistent_valid=persistent_valid,
        shuffled_features=shuffled,
        shuffled_valid=shuffled_valid,
        scale=scale,
        rotation=rotation,
        translation=translation,
        manifest=config.manifest,
        cache_path=source,
        dino_path=dino_path,
    )


def _load_raw_masks(payload: Mapping[str, Any], variant: str) -> torch.Tensor:
    variants = payload.get("tracking_variant_masks_stream")
    if isinstance(variants, Mapping) and variant in variants:
        return torch.as_tensor(variants[variant]).bool().cpu()
    fallback = payload.get("tracking_masks_stream")
    if fallback is None:
        raise ValueError("Frozen cache lacks tracking masks for DINO geometry.")
    return torch.as_tensor(fallback).bool().cpu()


def _tensor_3d(payload: Mapping[str, Any], key: str, path: Path) -> torch.Tensor:
    value = torch.as_tensor(payload[key]).float().cpu()
    if value.ndim != 3:
        raise ValueError(f"{key} in {path} must be [S,K,C], got {value.shape}")
    return value


def _tensor_2d(payload: Mapping[str, Any], key: str, path: Path) -> torch.Tensor:
    value = torch.as_tensor(payload[key]).bool().cpu()
    if value.ndim != 2:
        raise ValueError(f"{key} in {path} must be [S,K], got {value.shape}")
    return value


def _validate_model_compatibility(data: Sequence[ClipData]) -> None:
    if not data:
        return
    first = data[0]
    expected = (
        int(first.features.shape[1]),
        int(first.features.shape[-1]),
        int(first.single_features.shape[-1]),
    )
    for value in data[1:]:
        actual = (
            int(value.features.shape[1]),
            int(value.features.shape[-1]),
            int(value.single_features.shape[-1]),
        )
        if actual != expected:
            raise ValueError(
                "Cross-scene feature dimensions disagree: "
                f"first={expected} clip={value.name} actual={actual}."
            )


def _model_spec(data: ClipData, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "feature_channels": int(data.features.shape[-1]),
        "level_count": int(data.features.shape[1]),
        "object_feature_channels": int(data.single_features.shape[-1]),
        "projection_channels": int(args.projection_channels),
        "object_projection_channels": int(args.object_projection_channels),
        "hidden_channels": int(args.hidden_channels),
    }


def _build_examples(
    data: Sequence[ClipData], *, minimum_points: int
) -> list[Example]:
    output: list[Example] = []
    for clip_index, clip in enumerate(data):
        object_union = clip.masks.any(dim=1)
        for frame_index in range(len(clip.frame_indices)):
            count = int((clip.support[frame_index] & object_union[frame_index]).sum())
            if count >= int(minimum_points):
                output.append(Example(clip_index, frame_index))
    return output


def _train_branch(
    model: ObjectConditionedResidualHead,
    *,
    branch: str,
    train_data: Sequence[ClipData],
    train_examples: Sequence[Example],
    validation_data: Sequence[ClipData],
    validation_examples: Sequence[Example],
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor], dict[str, Any]]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        foreach=False,
    )
    generator = torch.Generator(device="cpu")
    rows: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_score: tuple[float, float] | None = None
    best_epoch = -1
    for epoch in range(int(args.epochs)):
        generator.manual_seed(int(args.seed) + epoch * 1009)
        order = torch.randperm(len(train_examples), generator=generator).tolist()
        model.train()
        losses: list[float] = []
        point_losses: list[float] = []
        updates = 0
        max_gradient = 0.0
        for position in order:
            example = train_examples[position]
            data = train_data[example.clip_index]
            output, raw, target, support = _forward_example(
                model, data, example.frame_index, branch, device
            )
            valid = support[0] & output.object_union[0]
            if int(valid.sum()) < int(args.minimum_object_points):
                continue
            point_loss = robust_point_loss(
                raw + output.correction,
                target,
                valid,
                beta=float(args.point_beta),
            )
            correction_norm = output.correction[0][valid].norm(dim=-1).mean()
            loss = point_loss + float(args.correction_regularization) * correction_norm
            if not bool(torch.isfinite(loss).detach().cpu()):
                raise RuntimeError(f"Non-finite {branch} loss at epoch={epoch}.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(args.grad_clip_norm)
            )
            gradient_value = float(gradient.detach().cpu())
            if not math.isfinite(gradient_value):
                raise RuntimeError(f"Non-finite {branch} gradient at epoch={epoch}.")
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            point_losses.append(float(point_loss.detach().cpu()))
            max_gradient = max(max_gradient, gradient_value)
            updates += 1

        validation = _quick_metrics(
            model,
            branch=branch,
            data=validation_data,
            examples=validation_examples,
            device=device,
            maximum_points=int(args.maximum_points_per_frame),
        )
        score = (validation["object_rmse"], validation["object_p90"])
        selected = int(best_score is None or score < best_score)
        if selected:
            best_score = score
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        rows.append(
            {
                "branch": branch,
                "epoch": epoch,
                "updates": updates,
                "mean_loss": _mean_or_nan(losses),
                "mean_point_loss": _mean_or_nan(point_losses),
                "max_gradient_norm": max_gradient,
                "validation_object_rmse": validation["object_rmse"],
                "validation_object_median": validation["object_median"],
                "validation_object_p90": validation["object_p90"],
                "selected_as_best": selected,
            }
        )
        if epoch == 0 or epoch == int(args.epochs) - 1 or (epoch + 1) % 5 == 0:
            print(
                f"    epoch={epoch + 1}/{args.epochs} updates={updates} "
                f"loss={rows[-1]['mean_loss']:.6f} "
                f"val_rmse={validation['object_rmse']:.6f}"
            )
    if best_state is None or best_score is None:
        raise RuntimeError(f"No validation checkpoint selected for {branch}.")
    return (
        rows,
        best_state,
        {
            "rule": "minimum_validation_object_support_rmse_then_p90",
            "best_epoch": best_epoch,
            "best_validation_rmse": best_score[0],
            "best_validation_p90": best_score[1],
            "epochs_evaluated": int(args.epochs),
            "test_metrics_read_during_selection": 0,
        },
    )


@torch.inference_mode()
def _quick_metrics(
    model: ObjectConditionedResidualHead,
    *,
    branch: str,
    data: Sequence[ClipData],
    examples: Sequence[Example],
    device: torch.device,
    maximum_points: int,
) -> dict[str, float]:
    model.eval()
    errors: list[torch.Tensor] = []
    for example in examples:
        clip = data[example.clip_index]
        output, raw, _, support = _forward_example(
            model, clip, example.frame_index, branch, device
        )
        valid = support[0] & output.object_union[0]
        if not bool(valid.any()):
            continue
        predicted_metric = apply_similarity(
            raw + output.correction,
            scale=clip.scale,
            rotation=clip.rotation,
            translation=clip.translation,
        )[0].cpu()
        target_metric = clip.target_metric[example.frame_index]
        selected = _limited_indices(valid.cpu(), maximum_points)
        error = torch.linalg.vector_norm(
            predicted_metric.reshape(-1, 3).index_select(0, selected)
            - target_metric.reshape(-1, 3).index_select(0, selected),
            dim=-1,
        )
        errors.append(error)
    if not errors:
        raise ValueError(f"Validation has no object support for branch={branch}.")
    joined = torch.cat(errors)
    return {
        "object_rmse": _rmse(joined),
        "object_median": float(joined.median()),
        "object_p90": float(torch.quantile(joined, 0.90)),
    }


@torch.inference_mode()
def _zero_update_error(
    model: ObjectConditionedResidualHead,
    data: ClipData,
    frame_index: int,
    branch: str,
    device: torch.device,
) -> float:
    model.eval()
    output, _, _, _ = _forward_example(model, data, frame_index, branch, device)
    return float(output.correction.abs().max().cpu())


def _forward_example(
    model: ObjectConditionedResidualHead,
    data: ClipData,
    frame_index: int,
    branch: str,
    device: torch.device,
):
    index = int(frame_index)
    if branch == "geometry_only":
        object_features = None
        object_valid = None
    elif branch == "single_view_dino":
        object_features = data.single_features[index : index + 1]
        object_valid = data.single_valid[index : index + 1]
    elif branch == "persistent_dino":
        object_features = data.persistent_features[index : index + 1]
        object_valid = data.persistent_valid[index : index + 1]
    elif branch == "shuffled_persistent_dino":
        object_features = data.shuffled_features[index : index + 1]
        object_valid = data.shuffled_valid[index : index + 1]
    else:
        raise ValueError(f"A learned branch is required, got {branch!r}.")
    output = model(
        data.features[index : index + 1].to(device=device, dtype=torch.float32),
        output_size=data.image_size,
        patch_shape=data.patch_shape,
        object_masks=data.masks[index : index + 1].to(device),
        object_features=(object_features.to(device) if object_features is not None else None),
        object_valid=(object_valid.to(device) if object_valid is not None else None),
    )
    raw = data.raw_native[index : index + 1].to(device)
    target = data.target_native[index : index + 1].to(device)
    support = data.support[index : index + 1].to(device)
    return output, raw, target, support


def _evaluate_clip(
    data: ClipData,
    *,
    states: Mapping[str, Mapping[str, Any]],
    model_spec: Mapping[str, Any],
    device: torch.device,
    args: argparse.Namespace,
    split: str,
    output_dir: Path,
) -> dict[str, Any]:
    predictions: dict[str, torch.Tensor] = {
        "raw": data.raw_native.clone()
    }
    for branch in LEARNED_BRANCHES:
        model = ObjectConditionedResidualHead(**dict(model_spec)).to(device)
        model.load_state_dict(states[branch]["state_dict"], strict=True)
        model.eval()
        branch_predictions: list[torch.Tensor] = []
        for index in range(len(data.frame_indices)):
            output, raw, _, _ = _forward_example(model, data, index, branch, device)
            branch_predictions.append((raw + output.correction).cpu()[0])
        predictions[branch] = torch.stack(branch_predictions)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    aligned_predictions = {
        branch: apply_similarity(
            points,
            scale=data.scale,
            rotation=data.rotation,
            translation=data.translation,
        ).float()
        for branch, points in predictions.items()
    }
    gt_instances = load_ground_truth_instances(
        data.manifest,
        scene_id=data.scene_id,
        frame_indices=data.frame_indices,
        output_size=data.image_size,
        prompts=data.prompts,
        prompt_label_aliases=None,
    )
    gt_masks = load_ground_truth_stream_masks(
        data.manifest,
        scene_id=data.scene_id,
        frame_indices=data.frame_indices,
        instance_ids=gt_instances.instance_ids,
        processed_size=data.image_size,
        image_mode=data.image_mode,
    )
    metrics: list[dict[str, Any]] = []
    objects: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []
    raw_object_lookup: dict[int, float] = {}
    for branch in BRANCHES:
        result = _score_predictions(
            data,
            branch=branch,
            points=aligned_predictions[branch],
            gt_masks=gt_masks,
            gt_instances=gt_instances,
            args=args,
            split=split,
        )
        metrics.append(result["summary"])
        objects.extend(result["objects"])
        frames.extend(result["frames"])
        if branch == "raw":
            raw_object_lookup = {
                int(row["object_index"]): float(row["paired_rmse_m"])
                for row in result["objects"]
            }
    for row in objects:
        if row["branch"] != "raw":
            raw_value = raw_object_lookup.get(int(row["object_index"]), float("nan"))
            row["paired_rmse_gain_vs_raw_percent"] = _gain(raw_value, float(row["paired_rmse_m"]))
            row["improved_vs_raw"] = int(
                math.isfinite(raw_value)
                and math.isfinite(float(row["paired_rmse_m"]))
                and float(row["paired_rmse_m"]) < raw_value
            )
    for row in metrics:
        if row["branch"] == "raw":
            continue
        branch_objects = [
            value
            for value in objects
            if value["branch"] == row["branch"]
        ]
        row["improved_objects_vs_raw"] = int(
            sum(int(value["improved_vs_raw"]) for value in branch_objects)
        )
        row["improved_object_ratio_vs_raw"] = (
            float(row["improved_objects_vs_raw"] / len(branch_objects))
            if branch_objects
            else 0.0
        )
    return {
        "clip": data.name,
        "scene_id": data.scene_id,
        "split": split,
        "metrics": metrics,
        "objects": objects,
        "frames": frames,
    }


def _score_predictions(
    data: ClipData,
    *,
    branch: str,
    points: torch.Tensor,
    gt_masks: torch.Tensor,
    gt_instances: Any,
    args: argparse.Namespace,
    split: str,
) -> dict[str, Any]:
    support = data.support
    frame_errors: list[torch.Tensor] = []
    frame_rows: list[dict[str, Any]] = []
    for index in range(points.shape[0]):
        selected = _limited_indices(support[index], int(args.maximum_points_per_frame))
        if not selected.numel():
            frame_rows.append(
                {
                    "clip": data.name,
                    "scene_id": data.scene_id,
                    "split": split,
                    "branch": branch,
                    "sequence_index": index,
                    "frame_index": data.frame_indices[index],
                    "evaluated_points": 0,
                    "rmse_m": float("nan"),
                    "median_m": float("nan"),
                    "p90_m": float("nan"),
                }
            )
            continue
        error = torch.linalg.vector_norm(
            points[index].reshape(-1, 3).index_select(0, selected)
            - data.target_metric[index].reshape(-1, 3).index_select(0, selected),
            dim=-1,
        )
        frame_errors.append(error)
        frame_rows.append(
            {
                "clip": data.name,
                "scene_id": data.scene_id,
                "split": split,
                "branch": branch,
                "sequence_index": index,
                "frame_index": data.frame_indices[index],
                "evaluated_points": int(error.numel()),
                "rmse_m": _rmse(error),
                "median_m": float(error.median()),
                "p90_m": float(torch.quantile(error, 0.90)),
            }
        )
    joined = torch.cat(frame_errors) if frame_errors else torch.empty(0)
    object_rows: list[dict[str, Any]] = []
    for object_index in range(int(gt_masks.shape[1])):
        object_support = gt_masks[:, object_index] & support
        predicted = deterministic_limit_points(
            points[object_support], int(args.maximum_object_points)
        )
        target = deterministic_limit_points(
            data.target_metric[object_support], int(args.maximum_object_points)
        )
        paired_pred, paired_target = deterministic_limit_pairs(
            points[object_support],
            data.target_metric[object_support],
            int(args.maximum_object_points),
        )
        if paired_pred.numel():
            paired_error = torch.linalg.vector_norm(
                paired_pred - paired_target, dim=-1
            )
            paired_rmse = _rmse(paired_error)
            paired_median = float(paired_error.median())
            paired_p90 = float(torch.quantile(paired_error, 0.90))
        else:
            paired_rmse = paired_median = paired_p90 = float("nan")
        if predicted.numel() and target.numel():
            shape_metrics = object_point_metrics(
                predicted,
                target,
                fscore_thresholds=(0.05, 0.10),
                voxel_size=0.05,
                ghost_distance=0.10,
                chunk_size=256,
            )
        else:
            shape_metrics = {
                "object_accuracy_m": float("nan"),
                "object_completeness_m": float("nan"),
                "symmetric_distance_m": float("nan"),
                "ghost_point_ratio": 1.0 if predicted.numel() else 0.0,
                "voxel_iou": 0.0,
                "fscore_5cm": 0.0,
                "fscore_10cm": 0.0,
            }
        object_rows.append(
            {
                "clip": data.name,
                "scene_id": data.scene_id,
                "split": split,
                "branch": branch,
                "object_index": object_index,
                "gt_instance_id": int(gt_instances.instance_ids[object_index]),
                "gt_label": str(gt_instances.labels[object_index]),
                "support_points": int(object_support.sum()),
                "predicted_points": int(predicted.shape[0]),
                "target_points": int(target.shape[0]),
                "paired_rmse_m": paired_rmse,
                "paired_median_m": paired_median,
                "paired_p90_m": paired_p90,
                "object_accuracy_m": shape_metrics.get("object_accuracy_m", float("nan")),
                "object_completeness_m": shape_metrics.get("object_completeness_m", float("nan")),
                "symmetric_distance_m": shape_metrics.get("symmetric_distance_m", float("nan")),
                "fscore_5cm": shape_metrics.get("fscore_5cm", 0.0),
                "fscore_10cm": shape_metrics.get("fscore_10cm", 0.0),
                "voxel_iou_5cm": shape_metrics.get("voxel_iou", 0.0),
                "ghost_point_ratio": shape_metrics.get("ghost_point_ratio", 0.0),
                "improved_vs_raw": 0,
            }
        )
    summary = {
        "clip": data.name,
        "scene_id": data.scene_id,
        "split": split,
        "branch": branch,
        "evaluated_points": int(joined.numel()),
        "global_rmse_m": _rmse(joined),
        "global_median_m": float(joined.median()) if joined.numel() else float("nan"),
        "global_p90_m": float(torch.quantile(joined, 0.90)) if joined.numel() else float("nan"),
        "object_count": len(object_rows),
        "object_paired_rmse_m": _finite_mean(object_rows, "paired_rmse_m"),
        "object_fscore_5cm": _finite_mean(object_rows, "fscore_5cm"),
        "object_voxel_iou_5cm": _finite_mean(object_rows, "voxel_iou_5cm"),
        "object_ghost_rate": _finite_mean(object_rows, "ghost_point_ratio"),
        "improved_objects_vs_raw": int(sum(int(row["improved_vs_raw"]) for row in object_rows)),
        "improved_object_ratio_vs_raw": 0.0,
    }
    if object_rows:
        summary["improved_object_ratio_vs_raw"] = float(
            summary["improved_objects_vs_raw"] / len(object_rows)
        )
    return {"summary": summary, "objects": object_rows, "frames": frame_rows}


def _build_summary(
    *,
    protocol: Path,
    feature_dir: Path,
    model_spec: Mapping[str, Any],
    selections: Mapping[str, Mapping[str, Any]],
    metric_rows: Sequence[Mapping[str, Any]],
    object_rows: Sequence[Mapping[str, Any]],
    frame_rows: Sequence[Mapping[str, Any]],
    train_clips: Sequence[ClipConfig],
    validation_clips: Sequence[ClipConfig],
    test_clips: Sequence[ClipConfig],
) -> dict[str, Any]:
    aggregate: dict[str, dict[str, Any]] = {}
    for branch in BRANCHES:
        rows = [row for row in metric_rows if row["branch"] == branch]
        test_rows = [row for row in rows if row["split"] == "test"]
        aggregate[branch] = {
            "test_clips": len(test_rows),
            "test_global_rmse_m": _mean_field(test_rows, "global_rmse_m"),
            "test_global_p90_m": _mean_field(test_rows, "global_p90_m"),
            "test_object_paired_rmse_m": _mean_field(test_rows, "object_paired_rmse_m"),
            "test_object_fscore_5cm": _mean_field(test_rows, "object_fscore_5cm"),
            "test_object_voxel_iou_5cm": _mean_field(test_rows, "object_voxel_iou_5cm"),
            "test_object_ghost_rate": _mean_field(test_rows, "object_ghost_rate"),
            "test_improved_objects": sum(int(row["improved_objects_vs_raw"]) for row in test_rows),
        }
    persistent = aggregate["persistent_dino"]
    single = aggregate["single_view_dino"]
    shuffled = aggregate["shuffled_persistent_dino"]
    persistent_better_single = (
        persistent["test_object_paired_rmse_m"] < single["test_object_paired_rmse_m"]
        and persistent["test_object_fscore_5cm"] >= single["test_object_fscore_5cm"]
    )
    persistent_better_shuffled = (
        persistent["test_object_paired_rmse_m"] < shuffled["test_object_paired_rmse_m"]
        and persistent["test_object_fscore_5cm"] >= shuffled["test_object_fscore_5cm"]
    )
    decision = {
        "persistent_better_than_single": int(persistent_better_single),
        "persistent_better_than_shuffled": int(persistent_better_shuffled),
        "overall": "GO" if persistent_better_single and persistent_better_shuffled else "NO_GO",
        "rule": "persistent object paired RMSE lower and object F-score not lower than both single-view and shuffled controls",
        "test_metrics_used_after_selection": 1,
    }
    return {
        "schema": 1,
        "revision": REVISION,
        "protocol": str(protocol),
        "feature_dir": str(feature_dir),
        "branches": list(BRANCHES),
        "model": dict(model_spec),
        "train_scenes": [clip.scene_id for clip in train_clips],
        "validation_scenes": [clip.scene_id for clip in validation_clips],
        "test_scenes": [clip.scene_id for clip in test_clips],
        "checkpoint_selection": dict(selections),
        "metrics": list(metric_rows),
        "aggregate": aggregate,
        "decision": decision,
        "data_policy": {
            "streamvggt_rerun": 0,
            "sam_rerun": 0,
            "dinov3_rerun": 0,
            "backbone_parameters_updated": 0,
            "point_head_parameters_updated": 0,
            "camera_head_parameters_updated": 0,
            "dinov3_parameters_updated": 0,
            "test_gt_read_during_checkpoint_selection": 0,
            "residual_application": "SAM raw object-mask union only; background exact raw fallback",
        },
    }


def _print_clip_metrics(result: Mapping[str, Any], split: str) -> None:
    print(f"  {split} clip={result['clip']} scene={result['scene_id']}")
    for row in result["metrics"]:
        print(
            f"    {row['branch']} global_rmse={row['global_rmse_m']:.5f} "
            f"object_rmse={row['object_paired_rmse_m']:.5f} "
            f"object_fscore5={row['object_fscore_5cm']:.5f} "
            f"object_voxelIoU5={row['object_voxel_iou_5cm']:.5f}"
        )


def _limited_indices(mask: torch.Tensor, maximum: int) -> torch.Tensor:
    selected = torch.nonzero(mask.reshape(-1), as_tuple=False)[:, 0]
    if selected.numel() <= int(maximum):
        return selected
    positions = torch.linspace(
        0, selected.numel() - 1, steps=int(maximum)
    ).round().long()
    return selected.index_select(0, positions)


def _finite_mean(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    values = [float(row[field]) for row in rows if math.isfinite(float(row[field]))]
    return sum(values) / len(values) if values else float("nan")


def _mean_field(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    values = [float(row[field]) for row in rows if math.isfinite(float(row[field]))]
    return sum(values) / len(values) if values else float("nan")


def _mean_or_nan(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _rmse(values: torch.Tensor) -> float:
    if not values.numel():
        return float("nan")
    return float(torch.sqrt(values.float().square().mean()))


def _gain(raw: float, candidate: float) -> float:
    if not math.isfinite(float(raw)) or not math.isfinite(float(candidate)):
        return float("nan")
    return 100.0 * (float(raw) - float(candidate)) / max(abs(float(raw)), 1e-12)


def _resolve_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {device} requested but CUDA is unavailable.")
    return device


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf8")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_copyable(path: Path, summary: Mapping[str, Any]) -> None:
    aggregate = summary["aggregate"]
    lines = [
        "===== COPYABLE_DINOV3_OBJECT_CONDITIONED_GEOMETRY_BEGIN =====",
        f"revision={summary['revision']}",
        "branches=" + ",".join(summary["branches"]),
        "train_scenes=" + ",".join(summary["train_scenes"]),
        "validation_scenes=" + ",".join(summary["validation_scenes"]),
        "test_scenes=" + ",".join(summary["test_scenes"]),
        "feature_condition=persistent_DINOv3_masked_mean_EMA_by_SAM_track_ID",
        "residual_application=SAM_object_mask_union_only",
        "backbone_parameters_updated=0",
        "dinov3_parameters_updated=0",
        "test_gt_read_during_checkpoint_selection=0",
        "",
        "branch,test_object_paired_rmse_m,test_object_fscore_5cm,test_object_voxel_iou_5cm,test_object_ghost_rate",
    ]
    for branch in summary["branches"]:
        value = aggregate[branch]
        lines.append(
            ",".join(
                [
                    branch,
                    str(value["test_object_paired_rmse_m"]),
                    str(value["test_object_fscore_5cm"]),
                    str(value["test_object_voxel_iou_5cm"]),
                    str(value["test_object_ghost_rate"]),
                ]
            )
        )
    lines.extend(
        (
            "",
            "decision=" + json.dumps(summary["decision"], sort_keys=True),
            f"summary={path.with_name('summary.json')}",
            f"models={path.with_name('models.pt')}",
            "===== COPYABLE_DINOV3_OBJECT_CONDITIONED_GEOMETRY_END =====",
        )
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        return value.tolist()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--point-beta", type=float, default=0.05)
    parser.add_argument("--correction-regularization", type=float, default=0.01)
    parser.add_argument("--projection-channels", type=int, default=64)
    parser.add_argument("--object-projection-channels", type=int, default=64)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--confidence-threshold", type=float, default=0.30)
    parser.add_argument("--minimum-object-points", type=int, default=128)
    parser.add_argument("--maximum-points-per-frame", type=int, default=8192)
    parser.add_argument("--maximum-object-points", type=int, default=1024)
    return parser.parse_args()


if __name__ == "__main__":
    main()
