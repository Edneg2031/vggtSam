"""Small protected-fallback heads for dense pointmap residual learning.

The frozen StreamVGGT pointmap remains the base prediction.  This module only
predicts a local residual, an optional trust gate, and an optional uncertainty
map from the four cached DPT feature levels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ResidualOutput:
    correction: torch.Tensor
    gate: torch.Tensor
    log_variance: torch.Tensor | None


def point_head_patch_features(
    token_levels: torch.Tensor,
    *,
    patch_start_idx: int,
    patch_shape: Sequence[int],
) -> torch.Tensor:
    """Return the exact cached DPT patch inputs as ``[S,L,N,2C]``.

    StreamVGGT caches every DPT level as ``[L,S,T,2C]``.  The first channel
    half is the frame-attention stream and the second half is the global
    history stream.  The frozen DPT point head consumes both halves.
    """

    levels = torch.as_tensor(token_levels)
    if levels.ndim != 4:
        raise ValueError(f"token_levels must be [L,S,T,2C], got {levels.shape}.")
    channels = int(levels.shape[-1])
    start = int(patch_start_idx)
    height, width = (int(value) for value in patch_shape)
    if channels % 2:
        raise ValueError("The cached StreamVGGT token channel count must be even.")
    if not 0 <= start < int(levels.shape[-2]):
        raise ValueError(f"Invalid patch_start_idx={start} for {levels.shape[-2]} tokens.")
    patches = levels[:, :, start:, :]
    if int(patches.shape[-2]) != height * width:
        raise ValueError(
            f"patch_shape={(height, width)} expects {height * width} tokens, "
            f"got {patches.shape[-2]}."
        )
    return patches.permute(1, 0, 2, 3).contiguous()


def apply_similarity(
    points: torch.Tensor,
    *,
    scale: float,
    rotation: torch.Tensor,
    translation: torch.Tensor,
) -> torch.Tensor:
    """Apply the cache's row-vector native-to-metric Sim(3)."""

    return float(scale) * (points @ rotation.to(points).T) + translation.to(points)


def invert_similarity(
    points: torch.Tensor,
    *,
    scale: float,
    rotation: torch.Tensor,
    translation: torch.Tensor,
) -> torch.Tensor:
    """Map metric supervision into the frozen StreamVGGT native gauge."""

    if not float(scale) > 0.0:
        raise ValueError(f"Similarity scale must be positive, got {scale}.")
    return ((points - translation.to(points)) @ rotation.to(points)) / float(scale)


class TrustAwareResidualHead(nn.Module):
    """Predict ``X_raw + gate * delta`` without changing frozen features."""

    def __init__(
        self,
        *,
        feature_channels: int,
        level_count: int,
        patch_shape: Sequence[int],
        projection_channels: int = 64,
        hidden_channels: int = 128,
        use_gate: bool,
        use_uncertainty: bool,
        gate_bias: float = -2.0,
    ) -> None:
        super().__init__()
        self.feature_channels = int(feature_channels)
        self.level_count = int(level_count)
        self.patch_shape = tuple(int(value) for value in patch_shape)
        self.use_gate = bool(use_gate)
        self.use_uncertainty = bool(use_uncertainty)
        if self.feature_channels <= 0 or self.level_count <= 0:
            raise ValueError("Residual-head feature dimensions must be positive.")
        if len(self.patch_shape) != 2 or min(self.patch_shape) <= 0:
            raise ValueError(f"Invalid patch shape {self.patch_shape}.")

        self.level_projections = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(self.feature_channels),
                nn.Linear(self.feature_channels, int(projection_channels)),
                nn.GELU(),
            )
            for _ in range(self.level_count)
        )
        fused_channels = self.level_count * int(projection_channels)
        self.trunk = nn.Sequential(
            nn.Conv2d(fused_channels, int(hidden_channels), 3, padding=1),
            nn.GELU(),
            nn.Conv2d(int(hidden_channels), int(hidden_channels), 3, padding=1),
            nn.GELU(),
        )
        self.residual_head = nn.Conv2d(int(hidden_channels), 3, 1)
        self.gate_head = (
            nn.Conv2d(int(hidden_channels), 1, 1) if self.use_gate else None
        )
        self.uncertainty_head = (
            nn.Conv2d(int(hidden_channels), 1, 1)
            if self.use_uncertainty
            else None
        )
        self._initialize_output_heads(float(gate_bias))

    def _initialize_output_heads(self, gate_bias: float) -> None:
        # Every learned branch is exactly the raw fallback before optimization.
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)
        if self.gate_head is not None:
            nn.init.zeros_(self.gate_head.weight)
            nn.init.constant_(self.gate_head.bias, gate_bias)
        if self.uncertainty_head is not None:
            nn.init.zeros_(self.uncertainty_head.weight)
            nn.init.zeros_(self.uncertainty_head.bias)

    def forward(
        self,
        features: torch.Tensor,
        *,
        output_size: Sequence[int],
    ) -> ResidualOutput:
        if features.ndim != 4:
            raise ValueError(f"features must be [B,L,N,C], got {features.shape}.")
        batch, levels, patches, channels = map(int, features.shape)
        height, width = self.patch_shape
        if (levels, patches, channels) != (
            self.level_count,
            height * width,
            self.feature_channels,
        ):
            raise ValueError(
                "Residual feature layout mismatch: "
                f"got {(levels, patches, channels)}, expected "
                f"{(self.level_count, height * width, self.feature_channels)}."
            )
        projected = [
            projection(features[:, position])
            for position, projection in enumerate(self.level_projections)
        ]
        fused = torch.cat(projected, dim=-1)
        fused = fused.transpose(1, 2).reshape(batch, -1, height, width)
        hidden = self.trunk(fused)
        dense_size = tuple(int(value) for value in output_size)
        residual = F.interpolate(
            self.residual_head(hidden),
            size=dense_size,
            mode="bilinear",
            align_corners=False,
        ).permute(0, 2, 3, 1)
        if self.gate_head is None:
            gate = torch.ones(
                (*residual.shape[:-1], 1),
                dtype=residual.dtype,
                device=residual.device,
            )
        else:
            gate = torch.sigmoid(
                F.interpolate(
                    self.gate_head(hidden),
                    size=dense_size,
                    mode="bilinear",
                    align_corners=False,
                ).permute(0, 2, 3, 1)
            )
        log_variance = None
        if self.uncertainty_head is not None:
            log_variance = F.interpolate(
                self.uncertainty_head(hidden),
                size=dense_size,
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1).clamp(-8.0, 8.0)
        return ResidualOutput(
            correction=gate * residual,
            gate=gate,
            log_variance=log_variance,
        )


def robust_point_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    *,
    beta: float,
) -> torch.Tensor:
    if not bool(valid.any()):
        raise ValueError("Point loss received an empty support mask.")
    component = F.smooth_l1_loss(
        predicted[valid], target[valid], beta=float(beta), reduction="none"
    )
    return component.sum(dim=-1).mean()


def heteroscedastic_point_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    log_variance: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Gaussian radial NLL used only by the uncertainty ablation."""

    # Select first so NaNs outside the GT raster support never enter autograd.
    selected_error = (predicted[valid] - target[valid]).square().sum(dim=-1)
    selected_log_var = log_variance[..., 0][valid]
    return 0.5 * (
        selected_error * torch.exp(-selected_log_var) + selected_log_var
    ).mean()
