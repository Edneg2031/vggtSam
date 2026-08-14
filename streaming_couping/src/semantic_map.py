"""Lift persistent SAM instance labels onto a frozen StreamVGGT world pointmap."""

from __future__ import annotations

from dataclasses import dataclass

import torch


# Fixed by permanent slot, so one tracked instance keeps the same color in
# every frame and every export. Colors are distinct at the configured 16 slots.
INSTANCE_PALETTE_RGB8: tuple[tuple[int, int, int], ...] = (
    (230, 25, 75),
    (60, 180, 75),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 212),
    (0, 128, 128),
    (220, 190, 255),
    (170, 110, 40),
    (255, 250, 200),
    (128, 0, 0),
    (170, 255, 195),
    (128, 128, 0),
)
BACKGROUND_SEMANTIC_RGB8 = (96, 96, 96)


@dataclass(frozen=True)
class SemanticPointMap:
    world_points: torch.Tensor
    rgb: torch.Tensor
    semantic_rgb: torch.Tensor
    semantic_slots: torch.Tensor
    confidence: torch.Tensor
    sequence_indices: torch.Tensor
    flat_indices: torch.Tensor
    dense_semantic_slots: torch.Tensor
    dense_valid: torch.Tensor


def build_semantic_pointmap(
    *,
    world_points: torch.Tensor,
    confidence: torch.Tensor,
    masks: torch.Tensor,
    track_scores: torch.Tensor,
    images: torch.Tensor,
    confidence_threshold: float,
    track_score_threshold: float,
    max_map_points: int,
) -> SemanticPointMap:
    """Attach SAM slots to the raw full-history world pointmap.

    Point positions are never transformed by the selected camera pose. StreamVGGT's
    point head already emits them in its first-frame reference coordinate system.
    """

    points = world_points.detach().float().cpu()
    confidence = normalize_confidence(_squeeze_scalar_map(confidence, "confidence"))
    masks = masks.detach().bool().cpu()
    scores = track_scores.detach().float().cpu()
    images = images.detach().float().cpu()
    if points.ndim != 4 or points.shape[-1] != 3:
        raise ValueError("world_points must have shape [S,H,W,3].")
    sequence, height, width, _ = points.shape
    if confidence.shape != (sequence, height, width):
        raise ValueError("confidence is not aligned to world_points.")
    if masks.ndim != 4 or masks.shape[0] != sequence or masks.shape[2:] != (
        height,
        width,
    ):
        raise ValueError("masks must have shape [S,I,H,W].")
    if masks.shape[1] < 1 or scores.shape != masks.shape[:2]:
        raise ValueError("track_scores must have shape [S,I] with I >= 1.")
    if images.shape != (sequence, 3, height, width):
        raise ValueError("images must have shape [S,3,H,W].")
    if not 0.0 <= float(confidence_threshold) <= 1.0:
        raise ValueError("confidence_threshold must be in [0,1].")
    if not 0.0 <= float(track_score_threshold) <= 1.0:
        raise ValueError("track_score_threshold must be in [0,1].")
    if int(max_map_points) < 1:
        raise ValueError("max_map_points must be positive.")

    slots = semantic_slot_map(
        masks,
        scores,
        score_threshold=float(track_score_threshold),
    )
    valid = (
        torch.isfinite(points).all(dim=-1)
        & torch.isfinite(confidence)
        & (confidence >= float(confidence_threshold))
    )
    flat_valid = torch.nonzero(valid.reshape(-1), as_tuple=False)[:, 0]
    if not flat_valid.numel():
        raise ValueError("Semantic map has no finite high-confidence points.")
    retained = _top_confidence_indices(
        confidence.reshape(-1).index_select(0, flat_valid),
        limit=int(max_map_points),
    )
    flat_indices = flat_valid.index_select(0, retained)
    rgb = images.clamp(0.0, 1.0).permute(0, 2, 3, 1).reshape(-1, 3)
    frame_grid = (
        torch.arange(sequence, dtype=torch.long)[:, None, None]
        .expand(sequence, height, width)
        .reshape(-1)
    )
    selected_slots = slots.reshape(-1).index_select(0, flat_indices)
    return SemanticPointMap(
        world_points=points.reshape(-1, 3).index_select(0, flat_indices),
        rgb=rgb.index_select(0, flat_indices),
        semantic_rgb=semantic_instance_colors(selected_slots),
        semantic_slots=selected_slots,
        confidence=confidence.reshape(-1).index_select(0, flat_indices),
        sequence_indices=frame_grid.index_select(0, flat_indices),
        flat_indices=flat_indices,
        dense_semantic_slots=slots,
        dense_valid=valid,
    )


def normalize_confidence(confidence: torch.Tensor) -> torch.Tensor:
    """Normalize StreamVGGT point confidence independently in each frame."""

    value = torch.nan_to_num(
        confidence.detach().float().cpu(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    flat = value.flatten(1)
    low = torch.quantile(flat, 0.05, dim=1, keepdim=True)
    high = torch.quantile(flat, 0.95, dim=1, keepdim=True)
    return ((flat - low) / (high - low).clamp_min(1e-6)).clamp(0.0, 1.0).reshape_as(
        value
    )


def semantic_slot_map(
    masks: torch.Tensor,
    scores: torch.Tensor,
    *,
    score_threshold: float,
) -> torch.Tensor:
    """Resolve overlapping masks by the highest persistent-track score."""

    if masks.ndim != 4 or scores.shape != masks.shape[:2]:
        raise ValueError("masks/scores must have shapes [S,I,H,W] and [S,I].")
    eligible = masks.bool() & (scores[:, :, None, None] >= float(score_threshold))
    weighted = torch.where(
        eligible,
        scores[:, :, None, None].expand_as(masks),
        torch.full(
            masks.shape,
            -torch.inf,
            dtype=torch.float32,
            device=masks.device,
        ),
    )
    best = weighted.argmax(dim=1).long()
    return torch.where(eligible.any(dim=1), best, torch.full_like(best, -1))


def semantic_instance_colors(slots: torch.Tensor) -> torch.Tensor:
    """Return categorical RGB in [0,1], with unlabeled points in gray."""

    slots = slots.detach().long().cpu()
    palette = torch.tensor(INSTANCE_PALETTE_RGB8, dtype=torch.float32) / 255.0
    background = torch.tensor(BACKGROUND_SEMANTIC_RGB8, dtype=torch.float32) / 255.0
    colors = background.expand(*slots.shape, 3).clone()
    labeled = slots >= 0
    if bool(labeled.any()):
        colors[labeled] = palette.index_select(
            0,
            slots[labeled].remainder(palette.shape[0]),
        )
    return colors


def _squeeze_scalar_map(value: torch.Tensor, name: str) -> torch.Tensor:
    output = value.detach().float().cpu()
    if output.ndim == 4 and output.shape[-1] == 1:
        output = output[..., 0]
    if output.ndim != 3:
        raise ValueError(f"{name} must have shape [S,H,W] or [S,H,W,1].")
    return output


def _top_confidence_indices(confidence: torch.Tensor, *, limit: int) -> torch.Tensor:
    if confidence.numel() <= limit:
        return torch.arange(confidence.numel(), dtype=torch.long)
    return torch.argsort(confidence, descending=True, stable=True)[:limit]
