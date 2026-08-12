"""Runtime helpers for the retained raw-pose dynamic-tracking baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import yaml


@dataclass(frozen=True)
class BaselineRunConfig:
    source_path: Path
    version: str
    output_dir: Path
    clip_name: str
    audit_device: str
    evaluation_frames: tuple[int, ...]


def load_baseline_run_config(path: str | Path) -> BaselineRunConfig:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf8") as handle:
        raw = yaml.safe_load(handle) or {}
    section = raw.get("baseline", {})
    frames = section.get("frames", {})
    config = BaselineRunConfig(
        source_path=source,
        version=str(section.get("version", "")).strip().lower(),
        output_dir=_path(section.get("output_dir", "outputs/streaming_couping_v0")),
        clip_name=str(section.get("clip_name", "")),
        audit_device=str(
            section.get(
                "audit_device",
                section.get("training_device", "cuda:0"),
            )
        ),
        evaluation_frames=_int_tuple(frames.get("evaluation", ())),
    )
    _validate_config(config)
    return config


def decode_cached_poses(
    payload: dict,
    *,
    pose_decoder,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    baseline_encoding = payload.get("baseline_pose_encoding")
    target_encoding = payload.get("target_pose_encoding")
    if not torch.is_tensor(baseline_encoding):
        raise ValueError("Baseline cache lacks tensor field 'baseline_pose_encoding'.")
    if not torch.is_tensor(target_encoding):
        raise ValueError("Baseline cache lacks tensor field 'target_pose_encoding'.")
    image_size = tuple(int(value) for value in payload["image_size"])
    baseline, _ = pose_decoder(
        baseline_encoding.unsqueeze(0).to(device),
        image_size_hw=image_size,
    )
    target, _ = pose_decoder(
        target_encoding.unsqueeze(0).to(device),
        image_size_hw=image_size,
    )
    return baseline, target


def pose_metrics(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    reference_index: int,
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
    relative = (
        predicted_eval[..., :3, :3]
        @ target_eval[..., :3, :3].transpose(-1, -2)
    )
    cosine = (
        torch.diagonal(relative, dim1=-2, dim2=-1).sum(dim=-1) - 1.0
    ) * 0.5
    rotation = torch.rad2deg(torch.acos(cosine.clamp(-1, 1))).mean()
    center = torch.linalg.vector_norm(
        camera_centers(predicted_eval) - camera_centers(target_eval),
        dim=-1,
    ).mean()
    return {
        "rotation_degrees": float(rotation.cpu()),
        "center_error_native": float(center.cpu()),
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


def _int_tuple(value) -> tuple[int, ...]:
    return tuple(int(item) for item in (value or ()))


def _path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _validate_config(config: BaselineRunConfig) -> None:
    if config.version != "v0":
        raise ValueError("The retained baseline config must declare baseline.version=v0.")
    if not config.clip_name:
        raise ValueError("baseline.clip_name is required.")
    if len(config.evaluation_frames) != 12:
        raise ValueError("V0 requires twelve future evaluation frames.")
    if len(set(config.evaluation_frames)) != len(config.evaluation_frames):
        raise ValueError("V0 evaluation frames must be unique.")
    if tuple(sorted(config.evaluation_frames)) != config.evaluation_frames:
        raise ValueError("V0 evaluation frames must be strictly time ordered.")
