"""Shared-support semantic point maps from frozen depth, K, masks and pose."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SemanticMapInputs:
    depth: torch.Tensor
    confidence: torch.Tensor
    intrinsics: torch.Tensor
    raw_world_to_camera: torch.Tensor
    selected_world_to_camera: torch.Tensor
    masks: torch.Tensor
    track_scores: torch.Tensor
    images: torch.Tensor


@dataclass(frozen=True)
class SemanticMapPair:
    raw_dense_points: torch.Tensor
    selected_dense_points: torch.Tensor
    valid: torch.Tensor
    semantic_slots: torch.Tensor
    normalized_confidence: torch.Tensor
    raw_map_points: torch.Tensor
    selected_map_points: torch.Tensor
    map_rgb: torch.Tensor
    map_semantic_slots: torch.Tensor
    map_confidence: torch.Tensor
    map_sequence_indices: torch.Tensor
    map_flat_indices: torch.Tensor


def build_shared_semantic_maps(
    inputs: SemanticMapInputs,
    *,
    confidence_threshold: float,
    track_score_threshold: float,
    max_map_points: int,
) -> SemanticMapPair:
    """Backproject one depth/K/mask support under raw and selected poses."""

    depth = _squeeze_scalar_map(inputs.depth, "depth")
    confidence = normalize_confidence(
        _squeeze_scalar_map(inputs.confidence, "confidence")
    )
    sequence, height, width = depth.shape
    _validate_inputs(inputs, sequence=sequence, height=height, width=width)
    slots = semantic_slot_map(
        inputs.masks.bool(),
        inputs.track_scores.float(),
        score_threshold=float(track_score_threshold),
    )
    camera_points = backproject_depth(depth, inputs.intrinsics.float())
    raw_native = camera_to_world(
        camera_points,
        inputs.raw_world_to_camera.float(),
    )
    selected_native = camera_to_world(
        camera_points,
        inputs.selected_world_to_camera.float(),
    )
    raw = raw_native
    selected = selected_native
    valid = (
        torch.isfinite(depth)
        & (depth > 0.0)
        & torch.isfinite(confidence)
        & (confidence >= float(confidence_threshold))
        & torch.isfinite(raw).all(dim=-1)
        & torch.isfinite(selected).all(dim=-1)
    )
    flat_valid = torch.nonzero(valid.reshape(-1), as_tuple=False)[:, 0]
    if not flat_valid.numel():
        raise ValueError("Semantic map has no valid raw-depth points.")
    flat_confidence = confidence.reshape(-1).index_select(0, flat_valid)
    retained = _top_confidence_indices(
        flat_confidence,
        limit=int(max_map_points),
    )
    map_flat_indices = flat_valid.index_select(0, retained)
    raw_flat = raw.reshape(-1, 3)
    selected_flat = selected.reshape(-1, 3)
    rgb_flat = (
        inputs.images.float()
        .clamp(0.0, 1.0)
        .permute(0, 2, 3, 1)
        .reshape(-1, 3)
    )
    frame_grid = (
        torch.arange(sequence, dtype=torch.long)[:, None, None]
        .expand(sequence, height, width)
        .reshape(-1)
    )
    return SemanticMapPair(
        raw_dense_points=raw,
        selected_dense_points=selected,
        valid=valid,
        semantic_slots=slots,
        normalized_confidence=confidence,
        raw_map_points=raw_flat.index_select(0, map_flat_indices).cpu(),
        selected_map_points=selected_flat.index_select(0, map_flat_indices).cpu(),
        map_rgb=rgb_flat.index_select(0, map_flat_indices).cpu(),
        map_semantic_slots=slots.reshape(-1).index_select(
            0, map_flat_indices
        ).cpu(),
        map_confidence=confidence.reshape(-1).index_select(
            0, map_flat_indices
        ).cpu(),
        map_sequence_indices=frame_grid.index_select(
            0, map_flat_indices
        ).cpu(),
        map_flat_indices=map_flat_indices.cpu(),
    )


def normalize_confidence(confidence: torch.Tensor) -> torch.Tensor:
    """Normalize expp1 head confidence per frame using fixed 5/95% quantiles."""

    value = torch.nan_to_num(
        confidence.float(), nan=0.0, posinf=0.0, neginf=0.0
    )
    flat = value.flatten(1)
    low = torch.quantile(flat, 0.05, dim=1, keepdim=True)
    high = torch.quantile(flat, 0.95, dim=1, keepdim=True)
    normalized = (flat - low) / (high - low).clamp_min(1e-6)
    return normalized.clamp(0.0, 1.0).reshape_as(value)


def semantic_slot_map(
    masks: torch.Tensor,
    scores: torch.Tensor,
    *,
    score_threshold: float,
) -> torch.Tensor:
    """Resolve mask overlaps by the highest persistent-track score."""

    if masks.ndim != 4:
        raise ValueError("masks must have shape [S,I,H,W].")
    if scores.shape != masks.shape[:2]:
        raise ValueError("track_scores must have shape [S,I].")
    eligible = masks.bool() & (
        scores[:, :, None, None] >= float(score_threshold)
    )
    weighted = torch.where(
        eligible,
        scores[:, :, None, None].expand_as(masks),
        torch.full_like(masks, -torch.inf, dtype=torch.float32),
    )
    best = weighted.argmax(dim=1).long()
    return torch.where(
        eligible.any(dim=1),
        best,
        torch.full_like(best, -1),
    )


def backproject_depth(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
) -> torch.Tensor:
    if depth.ndim != 3:
        raise ValueError("depth must have shape [S,H,W].")
    sequence, height, width = depth.shape
    if intrinsics.shape != (sequence, 3, 3):
        raise ValueError("intrinsics must have shape [S,3,3].")
    y, x = torch.meshgrid(
        torch.arange(height, dtype=depth.dtype, device=depth.device),
        torch.arange(width, dtype=depth.dtype, device=depth.device),
        indexing="ij",
    )
    pixels = torch.stack((x, y, torch.ones_like(x)), dim=-1)
    rays = torch.einsum(
        "hwj,sij->shwi",
        pixels,
        torch.linalg.inv(intrinsics),
    )
    return rays * depth[..., None]


def camera_to_world(
    camera_points: torch.Tensor,
    world_to_camera: torch.Tensor,
) -> torch.Tensor:
    if world_to_camera.ndim == 4 and world_to_camera.shape[0] == 1:
        world_to_camera = world_to_camera[0]
    if world_to_camera.shape != (camera_points.shape[0], 3, 4):
        raise ValueError("world_to_camera must have shape [S,3,4].")
    rotation = world_to_camera[:, :3, :3]
    translation = world_to_camera[:, :3, 3]
    return torch.einsum(
        "sij,shwj->shwi",
        rotation.transpose(-1, -2),
        camera_points - translation[:, None, None, :],
    )


def apply_similarity(
    points: torch.Tensor,
    *,
    scale: float,
    rotation: torch.Tensor,
    translation: torch.Tensor,
) -> torch.Tensor:
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("Similarity rotation/translation shapes are invalid.")
    return float(scale) * (points @ rotation.T) + translation


def _squeeze_scalar_map(value: torch.Tensor, name: str) -> torch.Tensor:
    value = value.detach().float().cpu()
    if value.ndim == 4 and value.shape[-1] == 1:
        value = value[..., 0]
    if value.ndim != 3:
        raise ValueError(f"{name} must have shape [S,H,W] or [S,H,W,1].")
    return value


def _top_confidence_indices(confidence: torch.Tensor, *, limit: int) -> torch.Tensor:
    if limit < 1:
        raise ValueError("max_map_points must be positive.")
    if confidence.numel() <= limit:
        return torch.arange(confidence.numel(), dtype=torch.long)
    # Stable tie-breaking keeps shared support exactly reproducible.
    order = torch.argsort(confidence, descending=True, stable=True)
    return order[:limit]


def _validate_inputs(
    inputs: SemanticMapInputs,
    *,
    sequence: int,
    height: int,
    width: int,
) -> None:
    if inputs.intrinsics.shape != (sequence, 3, 3):
        raise ValueError("Raw intrinsics shape mismatch.")
    if inputs.masks.shape[0] != sequence or inputs.masks.shape[2:] != (
        height,
        width,
    ):
        raise ValueError("Tracking masks are not aligned to raw depth.")
    if inputs.images.shape != (sequence, 3, height, width):
        raise ValueError("Stream images are not aligned to raw depth.")
    if inputs.track_scores.shape != inputs.masks.shape[:2]:
        raise ValueError("Track scores are not aligned to tracking masks.")
    if inputs.raw_world_to_camera.shape != inputs.selected_world_to_camera.shape:
        raise ValueError("Raw and selected pose shapes differ.")
