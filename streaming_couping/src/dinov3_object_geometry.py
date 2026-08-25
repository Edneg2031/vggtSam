"""Object-conditioned residual geometry on top of frozen StreamVGGT outputs.

This module deliberately keeps the geometry model frozen.  It consumes the
cached DPT inputs of the original StreamVGGT point head and a per-slot visual
condition (single-frame, persistent, or shuffled DINOv3 features).  The
learned head is only allowed to change pixels covered by the supplied SAM
object masks; all background pixels remain an exact raw-pointmap fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch.nn import functional as F
from torch import nn


@dataclass(frozen=True)
class ObjectResidualOutput:
    """Outputs needed for training and diagnostics."""

    correction: torch.Tensor
    object_union: torch.Tensor
    object_context: torch.Tensor


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
    "apply_similarity",
    "invert_similarity",
    "point_head_patch_features",
    "robust_point_loss",
]
