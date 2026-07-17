"""Shared data structures for the explicit bridge."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class GeometrySequence:
    world_points: torch.Tensor
    confidence: torch.Tensor
    world_to_camera: torch.Tensor
    intrinsics: torch.Tensor
    processed_size: tuple[int, int]
    source_sizes: tuple[tuple[int, int], ...]
    depth: torch.Tensor | None = None
    depth_confidence: torch.Tensor | None = None
    camera_world_points: torch.Tensor | None = None


@dataclass(frozen=True)
class TrackingSequence:
    masks: torch.Tensor
    scores: torch.Tensor
    selected_obj_id: int | None


@dataclass(frozen=True)
class SAM3MaskCandidate:
    obj_id: int
    mask: torch.Tensor
    score: float


@dataclass(frozen=True)
class SAM3SoftSequence:
    probabilities: torch.Tensor
    presence_logits: torch.Tensor
    captures_per_frame: torch.Tensor


@dataclass(frozen=True)
class RevisitCandidate:
    mask: torch.Tensor
    projected_mask: torch.Tensor
    supported_mask: torch.Tensor
    box_xyxy: tuple[int, int, int, int] | None
    projected_points: int
    supported_points: int
    projected_fraction: float
    support_ratio: float
    accepted: bool
    reason: str


@dataclass(frozen=True)
class SequenceInput:
    scene_id: str
    frame_indices: tuple[int, ...]
    image_paths: tuple[Path, ...]
    target_masks: torch.Tensor
    instance_id: int
    label: str
    reference_frame_idx: int
