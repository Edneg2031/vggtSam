"""Causal geometry guidance for the semantic-mapping SAM adapter.

The baseline SAM tracker remains the source of persistent identities.  This
module only proposes an image-space box and a few positive points from an
object's *previous* world points, asks SAM for a competing mask, and keeps the
raw mask when the candidate does not pass a conservative gate.  It therefore
does not alter HorizonStream/StreamVGGT predictions or SAM's hidden video
memory.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from ..temporal_prompt_matrix import (
    project_world_points,
    projected_bbox,
    select_surface_indices,
)
from ..types import SAM3MaskCandidate
from .contracts import GeometryFrame
from .geometry import geometry_confidence_for_frame, resize_bool_mask, world_points_for_frame


@dataclass(frozen=True)
class GeometryGuidanceConfig:
    """Causal prompt and replacement policy.

    The class is intentionally independent of a particular geometry model.
    The defaults are conservative; the CLI keeps this feature disabled unless
    the caller explicitly enables it.
    """

    min_history_points: int = 16
    max_history_points: int = 4096
    max_history_frames: int = 8
    max_history_gap: int = 30
    min_memory_score: float = 0.50
    min_memory_geometry_confidence: float = 0.30
    prompt_mode: str = "box_points"
    max_positive_points: int = 6
    front_fraction: float = 0.35
    box_quantile: float = 0.02
    box_padding_ratio: float = 0.08
    box_padding_pixels: int = 2
    min_candidate_support_recall: float = 0.25
    support_dilation: int = 5
    raw_support_recall: float = 0.70
    raw_box_precision: float = 0.10
    unreliable_support_margin: float = 0.05
    unreliable_score_margin: float = 0.05
    reliable_support_margin: float = 0.10
    reliable_score_margin: float = 0.10
    min_area_ratio: float = 0.50
    max_area_ratio: float = 4.00

    def validate(self) -> "GeometryGuidanceConfig":
        for name, value in (
            ("min_history_points", self.min_history_points),
            ("max_history_points", self.max_history_points),
            ("max_history_frames", self.max_history_frames),
            ("max_positive_points", self.max_positive_points),
            ("box_padding_pixels", self.box_padding_pixels),
        ):
            if int(value) < 1:
                raise ValueError(f"{name} must be positive.")
        if int(self.max_history_gap) < 0:
            raise ValueError("max_history_gap must be non-negative.")
        for name, value in (
            ("min_memory_score", self.min_memory_score),
            ("min_memory_geometry_confidence", self.min_memory_geometry_confidence),
            ("min_candidate_support_recall", self.min_candidate_support_recall),
            ("raw_support_recall", self.raw_support_recall),
            ("raw_box_precision", self.raw_box_precision),
            ("unreliable_support_margin", self.unreliable_support_margin),
            ("unreliable_score_margin", self.unreliable_score_margin),
            ("reliable_support_margin", self.reliable_support_margin),
            ("reliable_score_margin", self.reliable_score_margin),
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0,1].")
        if not 0.0 < float(self.front_fraction) <= 1.0:
            raise ValueError("front_fraction must be in (0,1].")
        if not 0.0 <= float(self.box_quantile) < 0.5:
            raise ValueError("box_quantile must be in [0,0.5).")
        if float(self.box_padding_ratio) < 0.0:
            raise ValueError("box_padding_ratio must be non-negative.")
        if int(self.support_dilation) < 0:
            raise ValueError("support_dilation must be non-negative.")
        if not 0.0 < float(self.min_area_ratio) <= 1.0:
            raise ValueError("min_area_ratio must be in (0,1].")
        if float(self.max_area_ratio) < 1.0:
            raise ValueError("max_area_ratio must be at least one.")
        if str(self.prompt_mode).strip().lower() not in {
            "box_points",
            "box_only",
            "points_only",
        }:
            raise ValueError(
                "prompt_mode must be one of box_points, box_only, points_only."
            )
        return self

    def to_dict(self) -> dict[str, object]:
        return {
            "min_history_points": int(self.min_history_points),
            "max_history_points": int(self.max_history_points),
            "max_history_frames": int(self.max_history_frames),
            "max_history_gap": int(self.max_history_gap),
            "min_memory_score": float(self.min_memory_score),
            "min_memory_geometry_confidence": float(
                self.min_memory_geometry_confidence
            ),
            "prompt_mode": str(self.prompt_mode),
            "max_positive_points": int(self.max_positive_points),
            "front_fraction": float(self.front_fraction),
            "box_quantile": float(self.box_quantile),
            "box_padding_ratio": float(self.box_padding_ratio),
            "box_padding_pixels": int(self.box_padding_pixels),
            "min_candidate_support_recall": float(
                self.min_candidate_support_recall
            ),
            "support_dilation": int(self.support_dilation),
            "raw_support_recall": float(self.raw_support_recall),
            "raw_box_precision": float(self.raw_box_precision),
            "unreliable_support_margin": float(self.unreliable_support_margin),
            "unreliable_score_margin": float(self.unreliable_score_margin),
            "reliable_support_margin": float(self.reliable_support_margin),
            "reliable_score_margin": float(self.reliable_score_margin),
            "min_area_ratio": float(self.min_area_ratio),
            "max_area_ratio": float(self.max_area_ratio),
        }


@dataclass(frozen=True)
class GeometrySegmentationPrompt:
    """A box and positive-pixel support in the current image grid."""

    box_mask: torch.Tensor
    positive_mask: torch.Tensor
    projected_points: int
    source_point_count: int
    source_frame_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.box_mask.ndim != 2 or self.positive_mask.ndim != 2:
            raise ValueError("Geometry prompt masks must be two-dimensional.")
        if tuple(self.box_mask.shape) != tuple(self.positive_mask.shape):
            raise ValueError("Geometry prompt masks must have the same shape.")
        if int(self.projected_points) < 0 or int(self.source_point_count) < 0:
            raise ValueError("Geometry prompt point counts must be non-negative.")

    @property
    def image_size(self) -> tuple[int, int]:
        return tuple(int(value) for value in self.box_mask.shape)


@dataclass(frozen=True)
class GeometryHistory:
    """Bounded world-point memory for one persistent semantic instance."""

    points: torch.Tensor
    source_frame_ids: tuple[int, ...]
    last_frame_id: int
    observation_count: int


@dataclass
class _HistoryState:
    entries: list[tuple[int, torch.Tensor]] = field(default_factory=list)
    observation_count: int = 0


class CausalObjectGeometryMemory:
    """Store only earlier high-confidence object points for prompt generation."""

    def __init__(self, config: GeometryGuidanceConfig) -> None:
        self.config = config.validate()
        self._states: dict[int, _HistoryState] = {}

    def get(
        self,
        instance_id: int,
        *,
        current_frame_id: int,
    ) -> GeometryHistory | None:
        state = self._states.get(int(instance_id))
        if state is None or not state.entries:
            return None
        last_frame_id = int(state.entries[-1][0])
        gap = int(current_frame_id) - last_frame_id - 1
        if gap < 0:
            raise ValueError(
                "Causal geometry memory was queried with a non-increasing frame."
            )
        if gap > int(self.config.max_history_gap):
            return None
        points = torch.cat([entry[1] for entry in state.entries], dim=0)
        if points.shape[0] > int(self.config.max_history_points):
            points = _evenly_sample(points, int(self.config.max_history_points))
        return GeometryHistory(
            points=points,
            source_frame_ids=tuple(int(entry[0]) for entry in state.entries),
            last_frame_id=last_frame_id,
            observation_count=int(state.observation_count),
        )

    def update(
        self,
        instance_id: int,
        geometry: GeometryFrame,
        mask: torch.Tensor,
        *,
        frame_id: int,
        score: float,
    ) -> dict[str, object]:
        """Add one accepted mask's world points and return update diagnostics."""

        frame_id = int(frame_id)
        if float(score) < float(self.config.min_memory_score):
            return {
                "updated": 0,
                "reason": "skip:low_track_score",
                "point_count": 0,
            }
        points, point_valid = world_points_for_frame(geometry)
        confidence = geometry_confidence_for_frame(geometry)
        resized_mask = resize_bool_mask(mask, geometry.image_size)
        selected = (
            point_valid
            & resized_mask
            & (confidence >= float(self.config.min_memory_geometry_confidence))
        )
        if not bool(selected.any()):
            return {
                "updated": 0,
                "reason": "skip:no_valid_geometry_points",
                "point_count": 0,
            }
        values = points[selected]
        values = _evenly_sample(values, int(self.config.max_history_points))
        state = self._states.setdefault(int(instance_id), _HistoryState())
        if state.entries and frame_id <= int(state.entries[-1][0]):
            raise ValueError(
                "Causal geometry memory updates must use increasing frame IDs."
            )
        state.entries.append((frame_id, values.detach().float().cpu()))
        state.entries = state.entries[-int(self.config.max_history_frames) :]
        state.observation_count += 1
        total_points = sum(int(entry[1].shape[0]) for entry in state.entries)
        return {
            "updated": 1,
            "reason": "update:high_confidence_object_points",
            "point_count": int(values.shape[0]),
            "stored_point_count": int(total_points),
        }

    def summary(self) -> dict[str, object]:
        return {
            "instance_count": int(len(self._states)),
            "stored_point_count": int(
                sum(
                    int(sum(entry[1].shape[0] for entry in state.entries))
                    for state in self._states.values()
                )
            ),
        }


def build_geometry_prompt(
    history: GeometryHistory | None,
    current_geometry: GeometryFrame,
    *,
    config: GeometryGuidanceConfig,
    output_size: Sequence[int] | None = None,
) -> GeometrySegmentationPrompt | None:
    """Project previous world points into the current camera and make prompts."""

    config.validate()
    if history is None:
        return None
    if int(history.points.shape[0]) < int(config.min_history_points):
        return None
    if current_geometry.camera_to_world is None or current_geometry.intrinsics is None:
        return None
    current_geometry.validate()
    world_to_camera = _world_to_camera(current_geometry.camera_to_world)
    projected = project_world_points(
        history.points,
        world_to_camera,
        current_geometry.intrinsics,
        current_geometry.image_size,
    )
    valid_count = int(projected.valid_mask.sum())
    if valid_count < 1:
        return None
    box = projected_bbox(
        projected.uv,
        projected.valid_mask,
        current_geometry.image_size,
        quantile=float(config.box_quantile),
        min_points=1,
    )
    if box is None:
        return None
    prompt_size = tuple(int(value) for value in current_geometry.image_size)
    box_mask = _box_mask(
        box,
        prompt_size,
        padding_ratio=float(config.box_padding_ratio),
        padding_pixels=int(config.box_padding_pixels),
    )
    selected = select_surface_indices(
        projected.uv,
        projected.depth,
        projected.valid_mask,
        int(config.max_positive_points),
        front_fraction=float(config.front_fraction),
    )
    positive_mask = torch.zeros(prompt_size, dtype=torch.bool)
    if selected.numel():
        uv = projected.uv.index_select(0, selected)
        x = uv[:, 0].round().long().clamp(0, prompt_size[1] - 1)
        y = uv[:, 1].round().long().clamp(0, prompt_size[0] - 1)
        positive_mask[y, x] = True
    if not bool(positive_mask.any()):
        return None
    prompt = GeometrySegmentationPrompt(
        box_mask=box_mask,
        positive_mask=positive_mask,
        projected_points=valid_count,
        source_point_count=int(history.points.shape[0]),
        source_frame_ids=tuple(int(value) for value in history.source_frame_ids),
    )
    if output_size is None or tuple(int(value) for value in output_size) == prompt_size:
        return prompt
    return resize_geometry_prompt(prompt, output_size)


def resize_geometry_prompt(
    prompt: GeometrySegmentationPrompt,
    output_size: Sequence[int],
) -> GeometrySegmentationPrompt:
    """Resize prompt masks while retaining their causal provenance."""

    size = tuple(int(value) for value in output_size)
    return GeometrySegmentationPrompt(
        box_mask=resize_bool_mask(prompt.box_mask, size),
        positive_mask=resize_bool_mask(prompt.positive_mask, size),
        projected_points=int(prompt.projected_points),
        source_point_count=int(prompt.source_point_count),
        source_frame_ids=tuple(prompt.source_frame_ids),
    )


def evaluate_geometry_mask(
    mask: torch.Tensor,
    *,
    score: float,
    prompt: GeometrySegmentationPrompt,
    support_dilation: int,
) -> dict[str, object]:
    """Score a raw or prompted mask against causal image-space support."""

    value = mask.detach().cpu().bool()
    if tuple(value.shape) != prompt.image_size:
        raise ValueError(
            f"Mask shape {tuple(value.shape)} differs from prompt {prompt.image_size}."
        )
    positive = prompt.positive_mask
    box = prompt.box_mask
    positive_pixels = int(positive.sum())
    mask_pixels = int(value.sum())
    support_recall = (
        float((value & positive).sum()) / positive_pixels
        if positive_pixels
        else 0.0
    )
    box_precision = float((value & box).sum()) / mask_pixels if mask_pixels else 0.0
    support_region = _dilate(positive, int(support_dilation))
    support_precision = (
        float((value & support_region).sum()) / mask_pixels if mask_pixels else 0.0
    )
    geometry_score = (
        0.50 * support_recall
        + 0.25 * box_precision
        + 0.15 * support_precision
        + 0.10 * float(torch.as_tensor(score).clamp(0.0, 1.0))
    )
    return {
        "mask": value,
        "score": float(score),
        "support_recall": float(support_recall),
        "box_precision": float(box_precision),
        "support_precision": float(support_precision),
        "geometry_score": float(geometry_score),
        "mask_pixels": int(mask_pixels),
        "box_pixels": int(box.sum()),
    }


def choose_geometry_candidate(
    candidates: Sequence[SAM3MaskCandidate],
    *,
    prompt: GeometrySegmentationPrompt,
    config: GeometryGuidanceConfig,
) -> dict[str, object] | None:
    """Select the strongest candidate that covers causal positive support."""

    evaluated: list[dict[str, object]] = []
    for candidate in candidates:
        try:
            row = evaluate_geometry_mask(
                candidate.mask,
                score=float(candidate.score),
                prompt=prompt,
                support_dilation=int(config.support_dilation),
            )
        except (TypeError, ValueError):
            continue
        row["candidate"] = candidate
        if float(row["support_recall"]) >= float(config.min_candidate_support_recall):
            evaluated.append(row)
    if not evaluated:
        return None
    return max(
        evaluated,
        key=lambda row: (
            float(row["geometry_score"]),
            float(row["support_precision"]),
            float(row["score"]),
            -int(row["candidate"].obj_id),
        ),
    )


def gate_geometry_replacement(
    *,
    raw_row: Mapping[str, object],
    candidate_row: Mapping[str, object] | None,
    config: GeometryGuidanceConfig,
) -> tuple[Mapping[str, object] | None, str]:
    """Compare a prompted candidate with raw SAM and keep a safe fallback."""

    if candidate_row is None:
        return None, "keep_raw:no_geometry_candidate"
    raw_pixels = int(raw_row["mask_pixels"])
    candidate_pixels = int(candidate_row["mask_pixels"])
    if raw_pixels:
        area_ratio = candidate_pixels / raw_pixels
        if area_ratio < float(config.min_area_ratio):
            return None, "keep_raw:candidate_too_small"
        if area_ratio > float(config.max_area_ratio):
            return None, "keep_raw:candidate_too_large"
    raw_unreliable = (
        raw_pixels == 0
        or float(raw_row["support_recall"]) < float(config.raw_support_recall)
        or float(raw_row["box_precision"]) < float(config.raw_box_precision)
    )
    support_gain = float(candidate_row["support_recall"]) - float(
        raw_row["support_recall"]
    )
    score_gain = float(candidate_row["geometry_score"]) - float(
        raw_row["geometry_score"]
    )
    support_margin = (
        float(config.unreliable_support_margin)
        if raw_unreliable
        else float(config.reliable_support_margin)
    )
    score_margin = (
        float(config.unreliable_score_margin)
        if raw_unreliable
        else float(config.reliable_score_margin)
    )
    if support_gain < support_margin:
        return None, "keep_raw:insufficient_support_gain"
    if score_gain < score_margin:
        return None, "keep_raw:insufficient_geometry_gain"
    return candidate_row, (
        "apply_geometry:raw_unreliable"
        if raw_unreliable
        else "apply_geometry:competitive_gain"
    )


def _world_to_camera(camera_to_world: torch.Tensor) -> torch.Tensor:
    pose = camera_to_world.detach().float().cpu()
    if tuple(pose.shape) == (3, 4):
        homogeneous = torch.eye(4, dtype=pose.dtype)
        homogeneous[:3] = pose
        pose = homogeneous
    if tuple(pose.shape) != (4, 4):
        raise ValueError("camera_to_world must have shape [3,4] or [4,4].")
    return torch.linalg.inv(pose)[:3]


def _box_mask(
    box: torch.Tensor,
    image_size: tuple[int, int],
    *,
    padding_ratio: float,
    padding_pixels: int,
) -> torch.Tensor:
    height, width = image_size
    x0, y0, x1, y1 = (float(value) for value in box.tolist())
    pad_x = max(float(padding_pixels), (x1 - x0 + 1.0) * float(padding_ratio))
    pad_y = max(float(padding_pixels), (y1 - y0 + 1.0) * float(padding_ratio))
    left = max(0, int(math.floor(x0 - pad_x)))
    top = max(0, int(math.floor(y0 - pad_y)))
    right = min(width, int(math.ceil(x1 + pad_x)) + 1)
    bottom = min(height, int(math.ceil(y1 + pad_y)) + 1)
    output = torch.zeros(height, width, dtype=torch.bool)
    if right > left and bottom > top:
        output[top:bottom, left:right] = True
    return output


def _dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if int(radius) <= 0:
        return mask.detach().cpu().bool()
    radius = int(radius)
    return F.max_pool2d(
        mask.detach().float().cpu()[None, None],
        kernel_size=2 * radius + 1,
        stride=1,
        padding=radius,
    )[0, 0].bool()


def _evenly_sample(points: torch.Tensor, limit: int) -> torch.Tensor:
    values = points.detach().float().cpu().reshape(-1, 3)
    values = values[torch.isfinite(values).all(dim=1)]
    if values.shape[0] <= int(limit):
        return values
    indices = torch.linspace(
        0,
        values.shape[0] - 1,
        steps=int(limit),
    ).round().long()
    return values.index_select(0, indices)


def summarize_diagnostics(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Produce compact JSON-safe counts for the pipeline result metadata."""

    reasons = Counter(str(row.get("reason", "unknown")) for row in rows)
    return {
        "observation_count": int(len(rows)),
        "prompt_available_count": int(
            sum(int(bool(row.get("prompt_available", 0))) for row in rows)
        ),
        "attempted_count": int(
            sum(int(bool(row.get("prompt_attempted", 0))) for row in rows)
        ),
        "applied_count": int(
            sum(int(bool(row.get("correction_applied", 0))) for row in rows)
        ),
        "fallback_count": int(
            sum(int(not bool(row.get("correction_applied", 0))) for row in rows)
        ),
        "reasons": dict(sorted(reasons.items())),
    }
