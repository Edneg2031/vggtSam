"""Lift persistent SAM instance labels onto a frozen StreamVGGT world pointmap."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

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
    # V1 fields are optional so the frozen V0 exporter and its serialized
    # schema remain unchanged.  In V1 ``semantic_slots`` is the persistent
    # object ID for backward-compatible coloring, while these fields preserve
    # explicit provenance.
    semantic_object_ids: torch.Tensor | None = None
    semantic_track_ids: torch.Tensor | None = None
    dense_semantic_object_ids: torch.Tensor | None = None
    dense_semantic_track_ids: torch.Tensor | None = None


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


def build_persistent_semantic_pointmap(
    *,
    world_points: torch.Tensor,
    confidence: torch.Tensor,
    masks: torch.Tensor,
    track_scores: torch.Tensor,
    persistent_object_ids: torch.Tensor,
    sam_track_ids: Sequence[int],
    images: torch.Tensor,
    confidence_threshold: float,
    track_score_threshold: float,
    max_map_points: int,
    map_write_mask: torch.Tensor | None = None,
) -> SemanticPointMap:
    """Lift SAM observations through V1 persistent object identities.

    Multiple short-term SAM slots may map to one object.  At a pixel where
    object masks overlap, the highest score wins, exactly as in V0.  The
    returned ``semantic_slots`` intentionally contains persistent IDs so old
    PLY viewers can still color the map, while explicit object/track tensors
    make the identity conversion inspectable.
    """

    points = world_points.detach().float().cpu()
    confidence = normalize_confidence(_squeeze_scalar_map(confidence, "confidence"))
    masks = masks.detach().bool().cpu()
    scores = track_scores.detach().float().cpu()
    object_ids = persistent_object_ids.detach().long().cpu()
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
    tracks = masks.shape[1]
    if tuple(scores.shape) != (sequence, tracks):
        raise ValueError("track_scores must have shape [S,I].")
    if tuple(object_ids.shape) != (sequence, tracks):
        raise ValueError("persistent_object_ids must have shape [S,I].")
    if len(sam_track_ids) != tracks:
        raise ValueError("sam_track_ids must match the mask slot count.")
    if images.shape != (sequence, 3, height, width):
        raise ValueError("images must have shape [S,3,H,W].")
    if map_write_mask is None:
        writes = torch.ones(sequence, tracks, dtype=torch.bool)
    else:
        writes = map_write_mask.detach().bool().cpu()
        if tuple(writes.shape) != (sequence, tracks):
            raise ValueError("map_write_mask must have shape [S,I].")

    dense_objects = persistent_object_id_map(
        masks,
        scores,
        object_ids,
        writes,
        score_threshold=float(track_score_threshold),
    )
    dense_tracks = persistent_track_id_map(
        masks,
        scores,
        object_ids,
        sam_track_ids,
        writes,
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
    selected_objects = dense_objects.reshape(-1).index_select(0, flat_indices)
    selected_tracks = dense_tracks.reshape(-1).index_select(0, flat_indices)
    return SemanticPointMap(
        world_points=points.reshape(-1, 3).index_select(0, flat_indices),
        rgb=rgb.index_select(0, flat_indices),
        semantic_rgb=semantic_instance_colors(selected_objects),
        semantic_slots=selected_objects,
        confidence=confidence.reshape(-1).index_select(0, flat_indices),
        sequence_indices=frame_grid.index_select(0, flat_indices),
        flat_indices=flat_indices,
        dense_semantic_slots=dense_objects,
        dense_valid=valid,
        semantic_object_ids=selected_objects,
        semantic_track_ids=selected_tracks,
        dense_semantic_object_ids=dense_objects,
        dense_semantic_track_ids=dense_tracks,
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


def persistent_object_id_map(
    masks: torch.Tensor,
    scores: torch.Tensor,
    persistent_object_ids: torch.Tensor,
    map_write_mask: torch.Tensor,
    *,
    score_threshold: float,
) -> torch.Tensor:
    """Resolve overlapping observations into persistent object IDs."""

    if masks.ndim != 4:
        raise ValueError("masks must have shape [S,I,H,W].")
    if tuple(scores.shape) != masks.shape[:2]:
        raise ValueError("scores do not match masks.")
    if tuple(persistent_object_ids.shape) != masks.shape[:2]:
        raise ValueError("persistent IDs do not match masks.")
    if tuple(map_write_mask.shape) != masks.shape[:2]:
        raise ValueError("map_write_mask does not match masks.")
    eligible = (
        masks.bool()
        & map_write_mask[:, :, None, None].bool()
        & (scores[:, :, None, None] >= float(score_threshold))
        & (persistent_object_ids[:, :, None, None] >= 0)
    )
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
    best = weighted.argmax(dim=1)
    object_grid = persistent_object_ids[:, :, None, None].expand_as(masks)
    best_ids = torch.gather(
        object_grid,
        1,
        best[:, None].expand(-1, 1, masks.shape[-2], masks.shape[-1]),
    )[:, 0]
    return torch.where(eligible.any(dim=1), best_ids.long(), torch.full_like(best_ids, -1))


def persistent_track_id_map(
    masks: torch.Tensor,
    scores: torch.Tensor,
    persistent_object_ids: torch.Tensor,
    sam_track_ids: Sequence[int],
    map_write_mask: torch.Tensor,
    *,
    score_threshold: float,
) -> torch.Tensor:
    """Return the source SAM ID that owns each selected persistent point."""

    track_ids = torch.as_tensor(tuple(int(value) for value in sam_track_ids))
    if track_ids.numel() != masks.shape[1]:
        raise ValueError("sam_track_ids do not match masks.")
    eligible = (
        masks.bool()
        & map_write_mask[:, :, None, None].bool()
        & (scores[:, :, None, None] >= float(score_threshold))
        & (persistent_object_ids[:, :, None, None] >= 0)
    )
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
    best = weighted.argmax(dim=1)
    track_grid = track_ids[None, :, None, None].expand_as(masks)
    best_track_ids = torch.gather(
        track_grid,
        1,
        best[:, None].expand(-1, 1, masks.shape[-2], masks.shape[-1]),
    )[:, 0]
    return torch.where(
        eligible.any(dim=1),
        best_track_ids.long(),
        torch.full_like(best_track_ids, -1),
    )


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
