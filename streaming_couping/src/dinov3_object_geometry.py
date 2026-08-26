"""Object-conditioned residual geometry on top of frozen StreamVGGT outputs.

This module deliberately keeps the geometry model frozen.  It consumes the
cached DPT inputs of the original StreamVGGT point head and a per-slot visual
condition (single-frame, persistent, or shuffled DINOv3 features).  The
learned head is only allowed to change pixels covered by the supplied SAM
object masks; all background pixels remain an exact raw-pointmap fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch.nn import functional as F
from torch import nn


@dataclass(frozen=True)
class ObjectResidualOutput:
    """Outputs needed for training and diagnostics."""

    correction: torch.Tensor
    object_union: torch.Tensor
    object_context: torch.Tensor


@dataclass(frozen=True)
class WorldSpaceGateConfig:
    """Causal, model-free safety policy for residual point corrections.

    The gate compares a corrected object observation with the raw observation
    and a memory built only from earlier accepted observations.  It therefore
    does not require GT at inference time and cannot use future frames.
    """

    confidence_threshold: float = 0.30
    track_score_threshold: float = 0.50
    min_points: int = 32
    max_correction_m: float = 0.08
    consistency_margin_m: float = 0.02
    shape_weight: float = 0.25
    memory_momentum: float = 0.80
    shape_scale_m: float = 0.05


@dataclass(frozen=True)
class WorldSpaceGateOutput:
    """Pointmap and diagnostics returned by the causal world-space gate."""

    points: torch.Tensor
    object_gate: torch.Tensor
    pixel_gate: torch.Tensor
    stats: Mapping[str, Any]


def point_head_patch_features(
    token_levels: torch.Tensor,
    *,
    patch_start_idx: int,
    patch_shape: Sequence[int],
) -> torch.Tensor:
    """Convert cached DPT levels to ``[S,L,N,C]`` point-head features."""

    levels = torch.as_tensor(token_levels)
    if levels.ndim != 4:
        raise ValueError(
            f"token_levels must have shape [L,S,T,C], got {tuple(levels.shape)}."
        )
    height, width = (int(value) for value in patch_shape)
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid patch_shape={tuple(patch_shape)}.")
    start = int(patch_start_idx)
    if not 0 <= start < int(levels.shape[-2]):
        raise ValueError(
            f"Invalid patch_start_idx={start} for token count {levels.shape[-2]}."
        )
    patches = levels[:, :, start:, :]
    expected = height * width
    if int(patches.shape[-2]) != expected:
        raise ValueError(
            f"patch_shape={(height, width)} expects {expected} patches, "
            f"got {patches.shape[-2]}."
        )
    return patches.permute(1, 0, 2, 3).contiguous()


class ObjectConditionedResidualHead(nn.Module):
    """Small fully-convolutional residual head with optional object context.

    ``features`` are the frozen StreamVGGT DPT point-head inputs.  Object
    features are projected once per slot and averaged at each patch location
    according to the resized SAM masks.  The same head therefore supports:

    * geometry-only: ``object_features=None``;
    * single-view DINO: ``single_features``;
    * persistent DINO: ``persistent_features``;
    * shuffled persistent control: ``shuffled_features``.

    The output layer is zero initialized, making every branch exactly equal to
    the raw pointmap before optimization.  The output is masked at dense
    resolution, so this guarantee also holds for non-object/background pixels.
    """

    def __init__(
        self,
        *,
        feature_channels: int,
        level_count: int,
        object_feature_channels: int,
        projection_channels: int = 64,
        object_projection_channels: int = 64,
        hidden_channels: int = 128,
    ) -> None:
        super().__init__()
        self.feature_channels = int(feature_channels)
        self.level_count = int(level_count)
        self.object_feature_channels = int(object_feature_channels)
        self.projection_channels = int(projection_channels)
        self.object_projection_channels = int(object_projection_channels)
        self.hidden_channels = int(hidden_channels)
        if min(
            self.feature_channels,
            self.level_count,
            self.object_feature_channels,
            self.projection_channels,
            self.object_projection_channels,
            self.hidden_channels,
        ) <= 0:
            raise ValueError("Residual head dimensions must be positive.")

        self.level_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(self.feature_channels),
                    nn.Linear(self.feature_channels, self.projection_channels),
                    nn.GELU(),
                )
                for _ in range(self.level_count)
            ]
        )
        self.object_projection = nn.Sequential(
            nn.LayerNorm(self.object_feature_channels),
            nn.Linear(
                self.object_feature_channels,
                self.object_projection_channels,
            ),
            nn.GELU(),
        )
        fused_channels = (
            self.level_count * self.projection_channels
            + self.object_projection_channels
        )
        self.trunk = nn.Sequential(
            nn.Conv2d(fused_channels, self.hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.hidden_channels, self.hidden_channels, 3, padding=1),
            nn.GELU(),
        )
        self.residual_head = nn.Conv2d(self.hidden_channels, 3, 1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def forward(
        self,
        features: torch.Tensor,
        *,
        output_size: Sequence[int],
        object_masks: torch.Tensor | None = None,
        object_features: torch.Tensor | None = None,
        object_valid: torch.Tensor | None = None,
        patch_shape: Sequence[int] | None = None,
    ) -> ObjectResidualOutput:
        if features.ndim != 4:
            raise ValueError(
                f"features must have shape [B,L,N,C], got {tuple(features.shape)}."
            )
        batch, levels, patches, channels = map(int, features.shape)
        if (levels, channels) != (self.level_count, self.feature_channels):
            raise ValueError(
                "Geometry feature layout mismatch: "
                f"got levels/channels={(levels, channels)}, expected "
                f"{(self.level_count, self.feature_channels)}."
            )
        if patch_shape is None:
            side = int(patches**0.5)
            if side * side != patches:
                raise ValueError("patch_shape is required for a non-square grid.")
            height, width = side, side
        else:
            height, width = (int(value) for value in patch_shape)
        if height <= 0 or width <= 0 or height * width != patches:
            raise ValueError(
                f"patch_shape={(height, width)} is incompatible with {patches} patches."
            )

        projected_levels = [
            projection(features[:, level])
            for level, projection in enumerate(self.level_projections)
        ]
        geometry_context = torch.cat(projected_levels, dim=-1)
        geometry_context = geometry_context.transpose(1, 2).reshape(
            batch,
            self.level_count * self.projection_channels,
            height,
            width,
        )

        patch_union, object_context = self._object_context(
            object_masks=object_masks,
            object_features=object_features,
            object_valid=object_valid,
            patch_shape=(height, width),
            batch=batch,
            device=features.device,
            dtype=geometry_context.dtype,
        )
        fused = torch.cat(
            [geometry_context, object_context.permute(0, 3, 1, 2)],
            dim=1,
        )
        hidden = self.trunk(fused)
        dense_size = tuple(int(value) for value in output_size)
        correction = F.interpolate(
            self.residual_head(hidden),
            size=dense_size,
            mode="bilinear",
            align_corners=False,
        ).permute(0, 2, 3, 1)
        if object_masks is None:
            object_union = torch.ones(
                batch,
                dense_size[0],
                dense_size[1],
                dtype=torch.bool,
                device=features.device,
            )
        else:
            dense_masks = torch.as_tensor(object_masks, device=features.device).float()
            if tuple(int(value) for value in dense_masks.shape[-2:]) != dense_size:
                dense_masks = F.interpolate(
                    dense_masks, size=dense_size, mode="nearest"
                )
            object_union = dense_masks.bool().any(dim=1)
        correction = correction * object_union.unsqueeze(-1).to(correction)
        return ObjectResidualOutput(
            correction=correction,
            object_union=object_union,
            object_context=object_context,
        )

    def _object_context(
        self,
        *,
        object_masks: torch.Tensor | None,
        object_features: torch.Tensor | None,
        object_valid: torch.Tensor | None,
        patch_shape: tuple[int, int],
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if object_masks is None:
            object_union = torch.ones(
                batch,
                patch_shape[0],
                patch_shape[1],
                dtype=torch.bool,
                device=device,
            )
            return object_union, torch.zeros(
                batch,
                patch_shape[0],
                patch_shape[1],
                self.object_projection_channels,
                dtype=dtype,
                device=device,
            )
        masks = torch.as_tensor(object_masks, device=device)
        if masks.ndim != 4 or int(masks.shape[0]) != batch:
            raise ValueError(
                "object_masks must have shape [B,K,H,W] and match geometry batch."
            )
        patch_masks = F.interpolate(
            masks.float(), size=patch_shape, mode="nearest"
        ) > 0.5
        if object_features is None:
            object_union = patch_masks.any(dim=1)
            return object_union, torch.zeros(
                batch,
                patch_shape[0],
                patch_shape[1],
                self.object_projection_channels,
                dtype=dtype,
                device=device,
            )
        values = torch.as_tensor(object_features, device=device).float()
        if values.ndim != 3 or int(values.shape[0]) != batch:
            raise ValueError(
                "object_features must have shape [B,K,C] and match geometry batch."
            )
        if int(values.shape[1]) != int(masks.shape[1]):
            raise ValueError("Object feature slots and object mask slots disagree.")
        if int(values.shape[2]) != self.object_feature_channels:
            raise ValueError(
                f"Object feature channels={values.shape[2]} do not match "
                f"head={self.object_feature_channels}."
            )
        if object_valid is None:
            valid = torch.ones(
                values.shape[:2], dtype=torch.bool, device=device
            )
        else:
            valid = torch.as_tensor(object_valid, device=device).bool()
            if tuple(valid.shape) != tuple(values.shape[:2]):
                raise ValueError("object_valid must have shape [B,K].")
        weights = patch_masks.float() * valid[:, :, None, None].float()
        projected = self.object_projection(values).to(dtype)
        numerator = torch.einsum("bkhw,bkc->bhwc", weights, projected)
        denominator = weights.sum(dim=1).unsqueeze(-1)
        context = numerator / denominator.clamp_min(1.0)
        context = torch.where(
            denominator > 0.0,
            context,
            torch.zeros_like(context),
        )
        return patch_masks.any(dim=1), context


def apply_similarity(
    points: torch.Tensor,
    *,
    scale: float,
    rotation: torch.Tensor,
    translation: torch.Tensor,
) -> torch.Tensor:
    """Map native StreamVGGT points into the metric reference gauge."""

    return float(scale) * (points @ rotation.to(points).T) + translation.to(points)


def invert_similarity(
    points: torch.Tensor,
    *,
    scale: float,
    rotation: torch.Tensor,
    translation: torch.Tensor,
) -> torch.Tensor:
    """Map metric supervision back into the frozen native pointmap gauge."""

    if float(scale) <= 0.0:
        raise ValueError(f"Similarity scale must be positive, got {scale}.")
    return ((points - translation.to(points)) @ rotation.to(points)) / float(scale)


def apply_world_space_consistency_gate(
    raw_points: torch.Tensor,
    candidate_points: torch.Tensor,
    *,
    object_masks: torch.Tensor,
    point_confidence: torch.Tensor,
    track_scores: torch.Tensor | None = None,
    config: WorldSpaceGateConfig = WorldSpaceGateConfig(),
) -> WorldSpaceGateOutput:
    """Safely accept residual corrections using causal object geometry memory.

    For each tracked object in temporal order, the current candidate is
    compared with the raw observation and with a robust centroid/extent state
    accumulated from earlier accepted observations.  A correction is applied
    only to the current object's valid pixels when it is small enough and does
    not make the world-space consistency cost worse than the raw observation
    by more than ``consistency_margin_m``.  Rejected observations fall back to
    the raw pointmap and still update the memory with raw geometry.

    This is deliberately an inference-time safeguard: no GT, future frame, or
    trainable parameter is used here.
    """

    _validate_world_space_gate_config(config)
    raw = raw_points.detach().float()
    candidate = candidate_points.detach().float()
    masks = object_masks.detach().bool().to(raw.device)
    confidence = point_confidence.detach().float().to(raw.device)
    if track_scores is None:
        scores = torch.ones(
            masks.shape[:2],
            dtype=torch.float32,
            device=raw.device,
        )
    else:
        scores = track_scores.detach().float().to(raw.device)
    if raw.ndim != 4 or raw.shape[-1] != 3:
        raise ValueError("raw_points must have shape [S,H,W,3].")
    if tuple(candidate.shape) != tuple(raw.shape):
        raise ValueError("raw_points and candidate_points must have the same shape.")
    sequence, height, width = (int(value) for value in raw.shape[:3])
    if masks.ndim != 4 or tuple(masks.shape[0:1] + masks.shape[2:]) != (
        sequence,
        height,
        width,
    ):
        raise ValueError("object_masks must have shape [S,K,H,W] matching points.")
    tracks = int(masks.shape[1])
    if tuple(confidence.shape) != (sequence, height, width):
        raise ValueError("point_confidence must have shape [S,H,W].")
    if tuple(scores.shape) != (sequence, tracks):
        raise ValueError("track_scores must have shape [S,K].")

    gated = raw.clone()
    object_gate = torch.zeros(
        sequence,
        tracks,
        dtype=torch.bool,
        device=raw.device,
    )
    pixel_gate = torch.zeros(
        sequence,
        height,
        width,
        dtype=torch.bool,
        device=raw.device,
    )
    memory_centroids: list[torch.Tensor | None] = [None] * tracks
    memory_extents: list[torch.Tensor | None] = [None] * tracks
    memory_counts = [0] * tracks
    reason_counts: dict[str, int] = {}
    valid_points = 0
    accepted_points = 0
    observed_object_observations = 0

    for frame in range(sequence):
        finite_raw = torch.isfinite(raw[frame]).all(dim=-1)
        finite_candidate = torch.isfinite(candidate[frame]).all(dim=-1)
        finite_confidence = torch.isfinite(confidence[frame])
        for slot in range(tracks):
            valid = (
                masks[frame, slot]
                & finite_raw
                & finite_candidate
                & finite_confidence
                & (confidence[frame] >= float(config.confidence_threshold))
                & (scores[frame, slot] >= float(config.track_score_threshold))
            )
            count = int(valid.sum().item())
            valid_points += count
            if count < int(config.min_points):
                _increment_reason(reason_counts, "insufficient_points")
                continue
            observed_object_observations += 1

            raw_object = raw[frame][valid]
            candidate_object = candidate[frame][valid]
            correction = torch.linalg.vector_norm(
                candidate_object - raw_object,
                dim=-1,
            )
            correction_q90 = float(torch.quantile(correction, 0.90).item())
            raw_centroid, raw_extent = _robust_object_summary(raw_object)
            candidate_centroid, candidate_extent = _robust_object_summary(
                candidate_object
            )
            accepted = correction_q90 <= float(config.max_correction_m)
            if accepted and memory_centroids[slot] is not None:
                raw_cost = _world_space_consistency_cost(
                    raw_centroid,
                    raw_extent,
                    memory_centroids[slot],
                    memory_extents[slot],
                    config,
                )
                candidate_cost = _world_space_consistency_cost(
                    candidate_centroid,
                    candidate_extent,
                    memory_centroids[slot],
                    memory_extents[slot],
                    config,
                )
                accepted = candidate_cost <= (
                    raw_cost + float(config.consistency_margin_m)
                )
                if not accepted:
                    _increment_reason(reason_counts, "world_space_inconsistent")
            elif not accepted:
                _increment_reason(reason_counts, "correction_too_large")

            if accepted:
                object_gate[frame, slot] = True
                pixel_gate[frame] |= valid
                gated[frame][valid] = candidate[frame][valid]
                accepted_points += count
                update_centroid, update_extent = candidate_centroid, candidate_extent
                _increment_reason(reason_counts, "accepted")
            else:
                update_centroid, update_extent = raw_centroid, raw_extent

            memory_centroids[slot], memory_extents[slot] = _update_world_memory(
                memory_centroids[slot],
                memory_extents[slot],
                update_centroid,
                update_extent,
                momentum=float(config.memory_momentum),
            )
            memory_counts[slot] += 1

    stats = {
        "sequence": sequence,
        "tracks": tracks,
        "accepted_object_observations": int(object_gate.sum().item()),
        "total_object_observations": int(sequence * tracks),
        "observed_object_observations": int(observed_object_observations),
        "accepted_valid_points": int(accepted_points),
        "valid_points": int(valid_points),
        "accepted_point_ratio": accepted_points / max(1, valid_points),
        "accepted_object_ratio": accepted_object_observations_ratio(
            accepted=int(object_gate.sum().item()),
            observed=observed_object_observations,
        ),
        "memory_observations": [int(value) for value in memory_counts],
        "reject_reasons": reason_counts,
    }
    return WorldSpaceGateOutput(
        points=gated,
        object_gate=object_gate,
        pixel_gate=pixel_gate,
        stats=stats,
    )


def accepted_object_observations_ratio(*, accepted: int, observed: int) -> float:
    """Return an object-level gate acceptance ratio with safe empty handling."""

    return float(accepted) / max(1, int(observed))


def _validate_world_space_gate_config(config: WorldSpaceGateConfig) -> None:
    if not 0.0 <= float(config.confidence_threshold) <= 1.0:
        raise ValueError("World-space gate confidence threshold must be in [0,1].")
    if not 0.0 <= float(config.track_score_threshold) <= 1.0:
        raise ValueError("World-space gate track threshold must be in [0,1].")
    if int(config.min_points) < 1:
        raise ValueError("World-space gate min_points must be positive.")
    if float(config.max_correction_m) <= 0.0:
        raise ValueError("World-space gate max_correction_m must be positive.")
    if float(config.consistency_margin_m) < 0.0:
        raise ValueError("World-space gate consistency margin must be non-negative.")
    if float(config.shape_weight) < 0.0 or float(config.shape_scale_m) <= 0.0:
        raise ValueError("World-space gate shape parameters are invalid.")
    if not 0.0 <= float(config.memory_momentum) < 1.0:
        raise ValueError("World-space gate memory_momentum must be in [0,1).")


def _increment_reason(counter: dict[str, int], name: str) -> None:
    counter[name] = int(counter.get(name, 0)) + 1


def _robust_object_summary(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    values = points.reshape(-1, 3)
    centroid = values.median(dim=0).values
    lower = torch.quantile(values, 0.10, dim=0)
    upper = torch.quantile(values, 0.90, dim=0)
    return centroid, (upper - lower).clamp_min(0.0)


def _world_space_consistency_cost(
    centroid: torch.Tensor,
    extent: torch.Tensor,
    memory_centroid: torch.Tensor | None,
    memory_extent: torch.Tensor | None,
    config: WorldSpaceGateConfig,
) -> torch.Tensor:
    if memory_centroid is None or memory_extent is None:
        return torch.zeros((), dtype=centroid.dtype, device=centroid.device)
    centroid_cost = torch.linalg.vector_norm(centroid - memory_centroid)
    shape_cost = torch.linalg.vector_norm(extent - memory_extent) / (
        torch.linalg.vector_norm(memory_extent).clamp_min(
            float(config.shape_scale_m)
        )
    )
    return centroid_cost + float(config.shape_weight) * float(
        config.shape_scale_m
    ) * shape_cost


def _update_world_memory(
    previous_centroid: torch.Tensor | None,
    previous_extent: torch.Tensor | None,
    centroid: torch.Tensor,
    extent: torch.Tensor,
    *,
    momentum: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if previous_centroid is None or previous_extent is None:
        return centroid.detach().clone(), extent.detach().clone()
    weight = 1.0 - float(momentum)
    return (
        float(momentum) * previous_centroid + weight * centroid,
        float(momentum) * previous_extent + weight * extent,
    )


def robust_point_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    *,
    beta: float = 0.05,
) -> torch.Tensor:
    """Smooth-L1 radial point loss on a selected dense support."""

    if predicted.shape != target.shape or valid.shape != predicted.shape[:-1]:
        raise ValueError("Point loss tensors have incompatible shapes.")
    if not bool(valid.any()):
        raise ValueError("Point loss received an empty support mask.")
    component = F.smooth_l1_loss(
        predicted[valid], target[valid], beta=float(beta), reduction="none"
    )
    return component.sum(dim=-1).mean()


__all__ = [
    "ObjectConditionedResidualHead",
    "ObjectResidualOutput",
    "WorldSpaceGateConfig",
    "WorldSpaceGateOutput",
    "apply_similarity",
    "apply_world_space_consistency_gate",
    "invert_similarity",
    "point_head_patch_features",
    "robust_point_loss",
]
