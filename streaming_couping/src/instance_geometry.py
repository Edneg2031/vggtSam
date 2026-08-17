"""Causal, bounded point adjustments indexed by persistent SAM instances."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SurfelQueryResult:
    """Per-query local surface diagnostics and bounded displacement."""

    valid: torch.Tensor
    delta: torch.Tensor
    nearest_distance: torch.Tensor
    normal_residual: torch.Tensor
    surface_thickness: torch.Tensor
    support_frames: torch.Tensor


def erode_instance_masks(masks: torch.Tensor, *, radius: int) -> torch.Tensor:
    """Erode independent boolean masks without mixing their identities."""

    value = masks.detach().bool().cpu()
    if value.ndim != 4:
        raise ValueError("masks must have shape [S,I,H,W].")
    radius = int(radius)
    if radius < 0:
        raise ValueError("erosion radius must be non-negative.")
    if radius == 0:
        return value.clone()
    sequence, instances, height, width = value.shape
    inverse = (~value).reshape(sequence * instances, 1, height, width).float()
    inverse = F.pad(
        inverse,
        (radius, radius, radius, radius),
        mode="constant",
        value=1.0,
    )
    dilated_inverse = F.max_pool2d(
        inverse,
        kernel_size=2 * radius + 1,
        stride=1,
        padding=0,
    )
    return (~dilated_inverse.bool()).reshape_as(value)


def shift_instance_masks(
    masks: torch.Tensor,
    *,
    shift_y: int,
    shift_x: int,
) -> torch.Tensor:
    """Area-preserving spatial-ownership control."""

    value = masks.detach().bool().cpu()
    if value.ndim != 4:
        raise ValueError("masks must have shape [S,I,H,W].")
    return torch.roll(value, shifts=(int(shift_y), int(shift_x)), dims=(-2, -1))


def select_mask_points(
    points: torch.Tensor,
    confidence: torch.Tensor,
    mask: torch.Tensor,
    *,
    limit: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select deterministic high-confidence points and their flat pixel indices."""

    points = points.detach().float().cpu()
    confidence = confidence.detach().float().cpu()
    mask = mask.detach().bool().cpu()
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("points must have shape [H,W,3].")
    if confidence.shape != points.shape[:2] or mask.shape != points.shape[:2]:
        raise ValueError("confidence/mask must align with points.")
    limit = int(limit)
    if limit < 1:
        raise ValueError("point selection limit must be positive.")
    valid = (
        mask
        & torch.isfinite(points).all(dim=-1)
        & torch.isfinite(confidence)
    )
    indices = torch.nonzero(valid.reshape(-1), as_tuple=False)[:, 0]
    if not indices.numel():
        return (
            torch.empty(0, 3, dtype=torch.float32),
            torch.empty(0, dtype=torch.float32),
            torch.empty(0, dtype=torch.long),
        )
    weights = confidence.reshape(-1).index_select(0, indices)
    if indices.numel() > limit:
        order = torch.argsort(weights, descending=True, stable=True)[:limit]
        indices = indices.index_select(0, order)
        weights = weights.index_select(0, order)
    selected = points.reshape(-1, 3).index_select(0, indices)
    return selected, weights, indices


def bounded_surfel_query(
    *,
    current_points: torch.Tensor,
    history_points: torch.Tensor,
    history_weights: torch.Tensor,
    history_frame_ids: torch.Tensor,
    device: str,
    neighbors: int,
    min_support_frames: int,
    match_radius: float,
    normal_variance_max: float,
    alpha: float,
    max_displacement: float,
    chunk_size: int,
) -> SurfelQueryResult:
    """Fit local historical surfels and return normal-only bounded residuals.

    The function never receives a target/GT pointmap. All history is supplied by
    the caller, which is responsible for enforcing frames strictly earlier than
    the current observation.
    """

    current = current_points.detach().float().cpu()
    history = history_points.detach().float().cpu()
    weights = history_weights.detach().float().cpu().clamp_min(1e-6)
    frame_ids = history_frame_ids.detach().long().cpu()
    if current.ndim != 2 or current.shape[-1] != 3:
        raise ValueError("current_points must have shape [N,3].")
    if history.ndim != 2 or history.shape[-1] != 3:
        raise ValueError("history_points must have shape [M,3].")
    if weights.shape != history.shape[:1] or frame_ids.shape != history.shape[:1]:
        raise ValueError("history weights/frame IDs must align with history points.")
    count = int(current.shape[0])
    empty = SurfelQueryResult(
        valid=torch.zeros(count, dtype=torch.bool),
        delta=torch.zeros(count, 3, dtype=torch.float32),
        nearest_distance=torch.full((count,), float("nan")),
        normal_residual=torch.full((count,), float("nan")),
        surface_thickness=torch.full((count,), float("nan")),
        support_frames=torch.zeros(count, dtype=torch.long),
    )
    neighbors = min(int(neighbors), int(history.shape[0]))
    if count == 0 or neighbors < 3:
        return empty
    if int(min_support_frames) < 1:
        raise ValueError("min_support_frames must be positive.")
    if not 0.0 < float(alpha) <= 1.0:
        raise ValueError("alpha must be in (0,1].")
    if float(match_radius) <= 0.0 or float(max_displacement) <= 0.0:
        raise ValueError("distance bounds must be positive.")

    compute = torch.device(device)
    history_device = history.to(compute)
    weights_device = weights.to(compute)
    frames_device = frame_ids.to(compute)
    valid_parts = []
    delta_parts = []
    nearest_parts = []
    residual_parts = []
    thickness_parts = []
    support_parts = []
    for start in range(0, count, int(chunk_size)):
        current_chunk = current[start : start + int(chunk_size)].to(compute)
        distances = torch.cdist(current_chunk, history_device)
        nearest, indices = torch.topk(
            distances,
            k=neighbors,
            dim=1,
            largest=False,
            sorted=True,
        )
        local = history_device[indices]
        local_weights = weights_device[indices]
        local_weights = local_weights / local_weights.sum(dim=1, keepdim=True).clamp_min(
            1e-8
        )
        center = (local * local_weights[..., None]).sum(dim=1)
        centered = local - center[:, None, :]
        covariance = torch.einsum(
            "bki,bkj,bk->bij",
            centered,
            centered,
            local_weights,
        )
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance.double())
        eigenvalues = eigenvalues.float().clamp_min(0.0)
        normal = eigenvectors[..., 0].float()
        residual = ((current_chunk - center) * normal).sum(dim=-1)
        thickness = eigenvalues[:, 0].sqrt()
        normal_ratio = eigenvalues[:, 0] / eigenvalues.sum(dim=-1).clamp_min(1e-8)
        local_frames = torch.sort(frames_device[indices], dim=1).values
        support_frames = torch.ones(
            local_frames.shape[0], dtype=torch.long, device=compute
        )
        if local_frames.shape[1] > 1:
            support_frames = support_frames + (
                local_frames[:, 1:] != local_frames[:, :-1]
            ).sum(dim=1)
        median_neighbor = nearest[:, neighbors // 2]
        valid = (
            torch.isfinite(residual)
            & torch.isfinite(thickness)
            & (support_frames >= int(min_support_frames))
            & (nearest[:, 0] <= float(match_radius))
            & (median_neighbor <= float(match_radius))
            & (residual.abs() <= float(match_radius))
            & (normal_ratio <= float(normal_variance_max))
            & (eigenvalues[:, 1] >= (0.02 * float(match_radius)) ** 2)
        )
        bounded = residual.mul(float(alpha)).clamp(
            -float(max_displacement), float(max_displacement)
        )
        delta = -bounded[:, None] * normal
        delta = torch.where(valid[:, None], delta, torch.zeros_like(delta))
        valid_parts.append(valid.cpu())
        delta_parts.append(delta.cpu())
        nearest_parts.append(nearest[:, 0].cpu())
        residual_parts.append(residual.abs().cpu())
        thickness_parts.append(thickness.cpu())
        support_parts.append(support_frames.cpu())
    return SurfelQueryResult(
        valid=torch.cat(valid_parts),
        delta=torch.cat(delta_parts),
        nearest_distance=torch.cat(nearest_parts),
        normal_residual=torch.cat(residual_parts),
        surface_thickness=torch.cat(thickness_parts),
        support_frames=torch.cat(support_parts),
    )


def merge_sparse_deltas(
    indices: list[torch.Tensor],
    deltas: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge sparse point displacements and average accidental duplicates."""

    if len(indices) != len(deltas):
        raise ValueError("indices and deltas must contain the same number of chunks.")
    if not indices:
        return torch.empty(0, dtype=torch.long), torch.empty(0, 3)
    flat_indices = torch.cat([value.detach().long().cpu() for value in indices])
    flat_deltas = torch.cat([value.detach().float().cpu() for value in deltas])
    if flat_deltas.shape != (flat_indices.numel(), 3):
        raise ValueError("sparse deltas must have shape [N,3].")
    unique, inverse = torch.unique(flat_indices, sorted=True, return_inverse=True)
    merged = torch.zeros(unique.numel(), 3, dtype=torch.float32)
    counts = torch.zeros(unique.numel(), dtype=torch.float32)
    merged.index_add_(0, inverse, flat_deltas)
    counts.index_add_(0, inverse, torch.ones_like(inverse, dtype=torch.float32))
    return unique, merged / counts[:, None].clamp_min(1.0)


def apply_sparse_deltas(
    points: torch.Tensor,
    indices: torch.Tensor,
    deltas: torch.Tensor,
) -> torch.Tensor:
    """Return a copy of dense points with sparse raw-coordinate displacements."""

    output = points.detach().float().cpu().clone()
    flat = output.reshape(-1, 3)
    indices = indices.detach().long().cpu()
    deltas = deltas.detach().float().cpu()
    if deltas.shape != (indices.numel(), 3):
        raise ValueError("indices/deltas are not aligned.")
    if indices.numel():
        flat.index_add_(0, indices, deltas)
    return output
