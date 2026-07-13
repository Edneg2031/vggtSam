"""Shared data structures for the explicit dual-framework bridge."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class GeometrySequence:
    """Frozen StreamVGGT outputs in one shared world coordinate system."""

    world_points: torch.Tensor  # [T, H, W, 3]
    confidence: torch.Tensor  # [T, H, W], normalized to [0, 1]
    world_to_camera: torch.Tensor  # [T, 3, 4]
    intrinsics: torch.Tensor  # [T, 3, 3], in the processed image coordinates
    processed_size: tuple[int, int]
    source_sizes: list[tuple[int, int]]


@dataclass
class ObjectMapEntry:
    instance_id: int
    label: str
    points: torch.Tensor
    weights: torch.Tensor
    observations: int = 0
    last_seen: int = -1
    centroid_history: list[torch.Tensor] = field(default_factory=list)


@dataclass(frozen=True)
class GateDecision:
    update_map: bool
    use_fallback: bool
    track_confidence: float
    geometry_confidence: float
    persistence: int
    reason: str

