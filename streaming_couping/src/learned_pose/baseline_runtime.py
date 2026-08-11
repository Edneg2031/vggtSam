"""Runtime helpers shared by the retained dynamic-instance baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn import functional as F
import yaml

from .dynamic_instance_baseline import (
    BaselineModelConfig,
    CameraPoseBaseline,
    DynamicInstanceGeometryRefiner,
)


@dataclass(frozen=True)
class OptimizerConfig:
    base_steps: int = 1200
    refiner_steps: int = 3000
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    grad_clip_norm: float = 5.0
    translation_weight: float = 10.0
    log_every: int = 200
    seed: int = 0
    min_train_loss_drop_percent: float = 1.0
    min_evaluation_gain_percent: float = 0.0


@dataclass(frozen=True)
class BaselineRunConfig:
    source_path: Path
    version: str
    output_dir: Path
    clip_name: str
    training_device: str
    local_point_count: int
    base_train_frames: tuple[int, ...]
    geometry_train_frames: tuple[int, ...]
    evaluation_frames: tuple[int, ...]
    model: BaselineModelConfig
    optimizer: OptimizerConfig


def load_baseline_run_config(path: str | Path) -> BaselineRunConfig:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf8") as handle:
        raw = yaml.safe_load(handle) or {}
    section = raw.get("baseline", {})
    model = section.get("model", {})
    optimization = section.get("optimization", {})
    frames = section.get("frames", {})
    config = BaselineRunConfig(
        source_path=source,
        version=str(section.get("version", "")).strip().lower(),
        output_dir=_path(section.get("output_dir", "outputs/streaming_couping_v0")),
        clip_name=str(section.get("clip_name", "")),
        training_device=str(section.get("training_device", "cuda:0")),
        local_point_count=int(section.get("local_point_count", 32)),
        base_train_frames=_int_tuple(frames.get("camera_base", ())),
        geometry_train_frames=_int_tuple(frames.get("geometry_refiner", ())),
        evaluation_frames=_int_tuple(frames.get("evaluation", ())),
        model=BaselineModelConfig(
            hidden_dim=int(model.get("hidden_dim", 256)),
            min_track_confidence=float(model.get("min_track_confidence", 0.50)),
            min_geometry_confidence=float(model.get("min_geometry_confidence", 0.20)),
            min_static_score=float(model.get("min_static_score", 0.20)),
            max_rotation_degrees=float(model.get("max_rotation_degrees", 15.0)),
            max_translation_native=float(model.get("max_translation_native", 0.50)),
            affinity_temperature=float(model.get("affinity_temperature", 0.10)),
        ),
        optimizer=OptimizerConfig(
            base_steps=int(optimization.get("base_steps", 1200)),
            refiner_steps=int(optimization.get("refiner_steps", 3000)),
            learning_rate=float(optimization.get("learning_rate", 1e-3)),
            weight_decay=float(optimization.get("weight_decay", 0.0)),
            grad_clip_norm=float(optimization.get("grad_clip_norm", 5.0)),
            translation_weight=float(optimization.get("translation_weight", 10.0)),
            log_every=int(optimization.get("log_every", 200)),
            seed=int(optimization.get("seed", 0)),
            min_train_loss_drop_percent=float(
                optimization.get("min_train_loss_drop_percent", 1.0)
            ),
            min_evaluation_gain_percent=float(
                optimization.get("min_evaluation_gain_percent", 0.0)
            ),
        ),
    )
    _validate_config(config)
    return config


def prepare_cached_batch(
    payload: dict,
    *,
    pose_decoder,
    device: str,
    local_point_count: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    fields = (
        "camera_hidden",
        "baseline_pose_encoding",
        "target_pose_encoding",
        "quality",
        "observed",
        "identity_valid",
    )
    batch: dict[str, torch.Tensor] = {}
    for field in fields:
        value = payload.get(field)
        if not torch.is_tensor(value):
            raise ValueError(f"Baseline cache lacks tensor field {field!r}.")
        batch[field] = value.unsqueeze(0).to(device)
    local, valid = local_geometry_features(
        payload, max_points=int(local_point_count)
    )
    batch["local_features"] = local.unsqueeze(0).to(device)
    batch["local_valid"] = valid.unsqueeze(0).to(device)
    image_size = tuple(int(value) for value in payload["image_size"])
    baseline, _ = pose_decoder(
        batch["baseline_pose_encoding"], image_size_hw=image_size
    )
    target, _ = pose_decoder(
        batch["target_pose_encoding"], image_size_hw=image_size
    )
    return batch, baseline, target


def local_geometry_features(
    payload: dict,
    *,
    max_points: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create backend-neutral local geometry tokens selected by SAM masks."""

    world = torch.as_tensor(payload["baseline_world_points"]).detach().float().cpu()
    confidence = torch.as_tensor(
        payload["baseline_world_confidence"]
    ).detach().float().cpu()
    if confidence.ndim == 4 and confidence.shape[-1] == 1:
        confidence = confidence[..., 0]
    uvd = torch.as_tensor(payload["instance_uvd"]).detach().float().cpu()
    valid = torch.as_tensor(payload["instance_uvd_valid"]).detach().bool().cpu()
    if world.ndim != 4 or world.shape[-1] != 3:
        raise ValueError("baseline_world_points must be [S,H,W,3].")
    if confidence.shape != world.shape[:3]:
        raise ValueError("point confidence/world point shapes differ.")
    if uvd.ndim != 4 or uvd.shape[-1] != 3 or valid.shape != uvd.shape[:-1]:
        raise ValueError("cached instance UVD shapes are invalid.")

    sequence, instances = uvd.shape[:2]
    output = torch.zeros(sequence, instances, int(max_points), 7)
    output_valid = torch.zeros(
        sequence, instances, int(max_points), dtype=torch.bool
    )
    height, width = world.shape[1:3]
    origin = torch.as_tensor(payload["scene_origin"]).detach().float().cpu()
    scale = max(float(payload["scene_scale"]), 1e-6)
    for frame in range(sequence):
        for slot in range(instances):
            available = torch.nonzero(valid[frame, slot], as_tuple=False).flatten()
            if not available.numel():
                continue
            if available.numel() > int(max_points):
                positions = torch.linspace(
                    0, available.numel() - 1, steps=int(max_points)
                ).round().long()
                available = available.index_select(0, positions)
            samples = uvd[frame, slot].index_select(0, available)
            x = samples[:, 0].round().long().clamp(0, width - 1)
            y = samples[:, 1].round().long().clamp(0, height - 1)
            points = world[frame, y, x]
            point_confidence = confidence[frame, y, x]
            finite = (
                torch.isfinite(points).all(dim=-1)
                & torch.isfinite(samples).all(dim=-1)
                & torch.isfinite(point_confidence)
                & samples[:, 2].gt(1e-6)
            )
            if not bool(finite.any()):
                continue
            points = points[finite]
            samples = samples[finite]
            point_confidence = point_confidence[finite]
            count = min(int(points.shape[0]), int(max_points))
            normalized_world = (points[:count] - origin) / scale
            normalized_u = 2.0 * samples[:count, 0] / max(width - 1, 1) - 1.0
            normalized_v = 2.0 * samples[:count, 1] / max(height - 1, 1) - 1.0
            log_depth = torch.log(
                (samples[:count, 2] / scale).clamp_min(1e-6)
            ).clamp(-8.0, 8.0)
            normalized_confidence = (
                torch.log1p(point_confidence[:count].clamp_min(0.0)) / 4.0
            ).clamp(0.0, 2.0)
            output[frame, slot, :count] = torch.cat(
                [
                    normalized_world,
                    normalized_u[:, None],
                    normalized_v[:, None],
                    log_depth[:, None],
                    normalized_confidence[:, None],
                ],
                dim=-1,
            )
            output_valid[frame, slot, :count] = True
    return output, output_valid


def forward_model(
    model: CameraPoseBaseline | DynamicInstanceGeometryRefiner,
    *,
    batch: dict[str, torch.Tensor],
    baseline: torch.Tensor,
    reference_index: int,
) -> dict[str, torch.Tensor]:
    common = {
        "camera_hidden": batch["camera_hidden"],
        "baseline_world_to_camera": baseline,
        "reference_index": int(reference_index),
    }
    if isinstance(model, CameraPoseBaseline):
        return model(**common)
    return model(
        **common,
        quality=batch["quality"],
        observed=batch["observed"],
        identity_valid=batch["identity_valid"],
        local_features=batch["local_features"],
        local_valid=batch["local_valid"],
    )


def train_pose_model(
    model: CameraPoseBaseline | DynamicInstanceGeometryRefiner,
    *,
    batch: dict[str, torch.Tensor],
    baseline: torch.Tensor,
    target: torch.Tensor,
    reference_index: int,
    training_indices: list[int],
    steps: int,
    config: OptimizerConfig,
) -> dict[str, float | int | str]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("Pose model has no trainable parameters.")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    initial_parameters = {
        name: value.detach().cpu().clone()
        for name, value in model.named_parameters()
        if value.requires_grad
    }
    model.eval()
    with torch.no_grad():
        initial = forward_model(
            model,
            batch=batch,
            baseline=baseline,
            reference_index=reference_index,
        )
        initial_loss = float(
            pose_loss(
                initial["world_to_camera"],
                target,
                reference_index=reference_index,
                translation_weight=config.translation_weight,
                evaluation_indices=training_indices,
            ).cpu()
        )
    best_loss = initial_loss
    best_step = 0
    best_state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    maximum_gradient_norm = 0.0
    first_gradient_norm = 0.0
    model.train()
    for step in range(1, int(steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        output = forward_model(
            model,
            batch=batch,
            baseline=baseline,
            reference_index=reference_index,
        )
        loss = pose_loss(
            output["world_to_camera"],
            target,
            reference_index=reference_index,
            translation_weight=config.translation_weight,
            evaluation_indices=training_indices,
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"Non-finite baseline loss at step {step}.")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, config.grad_clip_norm)
        if not bool(torch.isfinite(norm)):
            raise RuntimeError(f"Non-finite baseline gradient at step {step}.")
        maximum_gradient_norm = max(
            maximum_gradient_norm, float(norm.detach().cpu())
        )
        if step == 1:
            first_gradient_norm = float(norm.detach().cpu())
            if first_gradient_norm <= 0.0:
                raise RuntimeError(
                    "V0 training is a no-op at step 1: every gradient is zero."
                )
        optimizer.step()
        if step % int(config.log_every) == 0 or step == int(steps):
            model.eval()
            with torch.no_grad():
                checked = forward_model(
                    model,
                    batch=batch,
                    baseline=baseline,
                    reference_index=reference_index,
                )
                current = float(
                    pose_loss(
                        checked["world_to_camera"],
                        target,
                        reference_index=reference_index,
                        translation_weight=config.translation_weight,
                        evaluation_indices=training_indices,
                    ).cpu()
                )
            print(f"  step={step}/{steps} loss={current:.8f}")
            if current < best_loss:
                best_loss = current
                best_step = step
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
            model.train()
    model.load_state_dict(best_state)
    model.eval()
    parameter_update_norm = 0.0
    for name, value in model.named_parameters():
        if name not in initial_parameters:
            continue
        difference = value.detach().cpu() - initial_parameters[name]
        parameter_update_norm += float(difference.square().sum())
    parameter_update_norm = parameter_update_norm**0.5
    loss_drop_percent = 100.0 * (initial_loss - best_loss) / max(
        abs(initial_loss), 1e-12
    )
    if maximum_gradient_norm <= 0.0:
        raise RuntimeError("V0 training is a no-op: every gradient norm is zero.")
    if parameter_update_norm <= 1e-10:
        raise RuntimeError("V0 training is a no-op: parameters did not change.")
    if loss_drop_percent < float(config.min_train_loss_drop_percent):
        raise RuntimeError(
            "V0 training loss did not pass the no-op gate: "
            f"drop={loss_drop_percent:.6g}% required>="
            f"{float(config.min_train_loss_drop_percent):.6g}%."
        )
    return {
        "initial_loss": initial_loss,
        "best_loss": best_loss,
        "best_step": best_step,
        "loss_drop_percent": loss_drop_percent,
        "maximum_gradient_norm": maximum_gradient_norm,
        "first_gradient_norm": first_gradient_norm,
        "parameter_update_norm": parameter_update_norm,
        "no_op_check": "passed",
    }


def pose_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    reference_index: int,
    translation_weight: float,
    evaluation_indices: list[int] | None,
) -> torch.Tensor:
    indices = evaluation_indices_or_default(
        predicted.shape[1],
        reference_index=reference_index,
        evaluation_indices=evaluation_indices,
    )
    index = torch.tensor(indices, dtype=torch.long, device=predicted.device)
    predicted = predicted.index_select(1, index)
    target = target.index_select(1, index)
    rotation = F.mse_loss(predicted[..., :3, :3], target[..., :3, :3])
    translation = F.smooth_l1_loss(
        camera_centers(predicted), camera_centers(target), beta=0.01
    )
    return rotation + float(translation_weight) * translation


def pose_metrics(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    reference_index: int,
    translation_weight: float,
    evaluation_indices: list[int] | None,
) -> dict[str, float]:
    indices = evaluation_indices_or_default(
        predicted.shape[1],
        reference_index=reference_index,
        evaluation_indices=evaluation_indices,
    )
    index = torch.tensor(indices, dtype=torch.long, device=predicted.device)
    predicted_eval = predicted.index_select(1, index)
    target_eval = target.index_select(1, index)
    relative = predicted_eval[..., :3, :3] @ target_eval[..., :3, :3].transpose(-1, -2)
    cosine = (torch.diagonal(relative, dim1=-2, dim2=-1).sum(dim=-1) - 1.0) * 0.5
    rotation = torch.rad2deg(torch.acos(cosine.clamp(-1, 1))).mean()
    translation = torch.linalg.vector_norm(
        camera_centers(predicted_eval) - camera_centers(target_eval), dim=-1
    ).mean()
    loss = pose_loss(
        predicted,
        target,
        reference_index=reference_index,
        translation_weight=translation_weight,
        evaluation_indices=evaluation_indices,
    )
    return {
        "rotation_degrees": float(rotation.cpu()),
        "center_error_native": float(translation.cpu()),
        "loss": float(loss.cpu()),
    }


def evaluation_indices_or_default(
    sequence_length: int,
    *,
    reference_index: int,
    evaluation_indices: list[int] | None,
) -> list[int]:
    indices = (
        [index for index in range(sequence_length) if index != int(reference_index)]
        if evaluation_indices is None
        else [int(value) for value in evaluation_indices]
    )
    if not indices or int(reference_index) in indices:
        raise ValueError("Evaluation indices are empty or include the reference.")
    if len(indices) != len(set(indices)):
        raise ValueError("Evaluation indices contain duplicates.")
    if any(index < 0 or index >= sequence_length for index in indices):
        raise ValueError("Evaluation index is outside the sequence.")
    return indices


def camera_centers(world_to_camera: torch.Tensor) -> torch.Tensor:
    rotation = world_to_camera[..., :3, :3]
    translation = world_to_camera[..., :3, 3]
    return -(rotation.transpose(-1, -2) @ translation[..., None]).squeeze(-1)


def slice_batch_prefix(
    batch: dict[str, torch.Tensor], *, length: int
) -> dict[str, torch.Tensor]:
    sequence = int(batch["camera_hidden"].shape[1])
    if length < 2 or length > sequence:
        raise ValueError(f"Prefix length {length} is outside [2,{sequence}].")
    output = {}
    for name, value in batch.items():
        if value.ndim < 2 or int(value.shape[1]) != sequence:
            raise ValueError(f"Batch field {name!r} lacks sequence dimension 1.")
        output[name] = value[:, :length]
    return output


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(True, warn_only=True)


def _int_tuple(value) -> tuple[int, ...]:
    return tuple(int(item) for item in (value or ()))


def _path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _validate_config(config: BaselineRunConfig) -> None:
    if config.version != "v0":
        raise ValueError("The retained baseline config must declare baseline.version=v0.")
    if not config.clip_name:
        raise ValueError("baseline.clip_name is required.")
    groups = (
        config.base_train_frames,
        config.geometry_train_frames,
        config.evaluation_frames,
    )
    if any(not group for group in groups):
        raise ValueError("All baseline frame groups must be non-empty.")
    if set(config.base_train_frames) & set(config.geometry_train_frames):
        raise ValueError("Camera and geometry training frames must be disjoint.")
    trained = config.base_train_frames + config.geometry_train_frames
    if max(trained) >= min(config.evaluation_frames):
        raise ValueError("Evaluation frames must be strictly after training.")
    if config.local_point_count < 2:
        raise ValueError("baseline.local_point_count must be at least two.")
    if config.model.hidden_dim < 1:
        raise ValueError("baseline.model.hidden_dim must be positive.")
    if config.optimizer.base_steps < 1 or config.optimizer.refiner_steps < 1:
        raise ValueError("Baseline training steps must be positive.")
    if config.optimizer.log_every < 1:
        raise ValueError("baseline.optimization.log_every must be positive.")
    if config.optimizer.min_train_loss_drop_percent <= 0:
        raise ValueError(
            "baseline.optimization.min_train_loss_drop_percent must be positive."
        )
