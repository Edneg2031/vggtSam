"""3AM-style StreamVGGT feature merger for the SAM3 tracker FPN."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
import torch.nn.functional as F


class GeometryAwareSAM3Adapter(nn.Module):
    """Fuse multi-level geometry, then emit one confidence-gated FPN2 residual.

    StreamVGGT levels are merged from shallow to deep with self/cross attention.
    The result is convolutionally fused with the original SAM3 tracker FPN2.
    FPN0/FPN1 remain untouched so the pretrained high-resolution mask boundary
    path is preserved.
    """

    def __init__(
        self,
        *,
        num_geometry_levels: int = 3,
        sam_channels: int = 256,
        geometry_channels: int = 2048,
        hidden_channels: int = 256,
        num_heads: int = 8,
        dropout: float = 0.0,
        residual_init_std: float = 1e-4,
    ) -> None:
        super().__init__()
        if num_geometry_levels < 2:
            raise ValueError("At least two StreamVGGT levels are required.")
        if hidden_channels % num_heads != 0:
            raise ValueError("hidden_channels must be divisible by num_heads.")
        self.num_geometry_levels = int(num_geometry_levels)
        self.sam_channels = int(sam_channels)
        self.geometry_channels = int(geometry_channels)
        self.hidden_channels = int(hidden_channels)

        self.geometry_projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(self.geometry_channels),
                    nn.Linear(self.geometry_channels, self.hidden_channels),
                )
                for _ in range(self.num_geometry_levels)
            ]
        )
        self.geometry_self_attention = nn.MultiheadAttention(
            self.hidden_channels,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.geometry_cross_attention = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    self.hidden_channels,
                    num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(self.num_geometry_levels - 1)
            ]
        )
        self.geometry_norms = nn.ModuleList(
            [nn.LayerNorm(self.hidden_channels) for _ in range(self.num_geometry_levels)]
        )

        fusion_channels = self.sam_channels + self.hidden_channels + 1
        self.spatial_refine = nn.Sequential(
            nn.Conv2d(fusion_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, self.hidden_channels),
            nn.GELU(),
            nn.Conv2d(
                self.hidden_channels,
                self.hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.GroupNorm(8, self.hidden_channels),
            nn.GELU(),
        )
        self.learned_geometry_gate = nn.Sequential(
            nn.Conv2d(fusion_channels, self.hidden_channels // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(self.hidden_channels // 4, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.fpn2_residual = nn.Conv2d(
            self.hidden_channels,
            self.sam_channels,
            kernel_size=1,
        )
        if residual_init_std == 0:
            nn.init.zeros_(self.fpn2_residual.weight)
        else:
            nn.init.normal_(
                self.fpn2_residual.weight,
                mean=0.0,
                std=float(residual_init_std),
            )
        nn.init.zeros_(self.fpn2_residual.bias)

    def forward(
        self,
        sam_fpn2: torch.Tensor,
        geometry_levels: Sequence[torch.Tensor],
        geometry_confidence: torch.Tensor,
    ) -> tuple[list[torch.Tensor], dict[str, torch.Tensor]]:
        if sam_fpn2.ndim != 4 or sam_fpn2.shape[1] != self.sam_channels:
            raise ValueError(
                f"SAM3 FPN2 must be [T,{self.sam_channels},H,W], got "
                f"{tuple(sam_fpn2.shape)}."
            )
        levels = self._validate_levels(geometry_levels, sam_fpn2.shape[0])
        projected = [
            projection(level.flatten(2).transpose(1, 2))
            for projection, level in zip(self.geometry_projections, levels)
        ]
        merged_update, _ = self.geometry_self_attention(
            projected[0],
            projected[0],
            projected[0],
            need_weights=False,
        )
        merged = self.geometry_norms[0](projected[0] + merged_update)
        for index, (attention, level) in enumerate(
            zip(self.geometry_cross_attention, projected[1:]),
            start=1,
        ):
            update, _ = attention(merged, level, level, need_weights=False)
            merged = self.geometry_norms[index](merged + update)

        geometry_height, geometry_width = levels[0].shape[-2:]
        geometry_map = merged.transpose(1, 2).reshape(
            sam_fpn2.shape[0],
            self.hidden_channels,
            geometry_height,
            geometry_width,
        )
        geometry_map = F.interpolate(
            geometry_map,
            size=sam_fpn2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        geometry_map = geometry_map.to(
            device=sam_fpn2.device,
            dtype=sam_fpn2.dtype,
        )
        confidence = self._resize_confidence(
            geometry_confidence,
            size=sam_fpn2.shape[-2:],
            device=sam_fpn2.device,
            dtype=sam_fpn2.dtype,
        )
        fusion_input = torch.cat((sam_fpn2, geometry_map, confidence), dim=1)
        learned_gate = self.learned_geometry_gate(fusion_input)
        effective_gate = confidence * learned_gate
        refined = self.spatial_refine(fusion_input)
        fpn2 = self.fpn2_residual(refined) * effective_gate
        zeros_fpn0 = fpn2.new_zeros(fpn2.shape[0], 32, 288, 288)
        zeros_fpn1 = fpn2.new_zeros(fpn2.shape[0], 64, 144, 144)
        diagnostics = {
            "geometry_gate_mean": effective_gate.mean(),
            "geometry_gate_max": effective_gate.amax(),
            "learned_gate_mean": learned_gate.mean(),
            "geometry_confidence_mean": confidence.mean(),
            "fpn2_residual_rms": fpn2.float().square().mean().sqrt(),
        }
        return [zeros_fpn0, zeros_fpn1, fpn2], diagnostics

    def _validate_levels(
        self,
        geometry_levels: Sequence[torch.Tensor],
        num_frames: int,
    ) -> list[torch.Tensor]:
        levels = list(geometry_levels)
        if len(levels) != self.num_geometry_levels:
            raise ValueError(
                f"Expected {self.num_geometry_levels} geometry levels, got "
                f"{len(levels)}."
            )
        spatial_shape = None
        for level in levels:
            if level.ndim != 4:
                raise ValueError(
                    f"Geometry levels must be [T,C,H,W], got {tuple(level.shape)}."
                )
            if level.shape[0] != num_frames:
                raise ValueError(
                    f"SAM/geometry frame mismatch: {num_frames} vs {level.shape[0]}."
                )
            if level.shape[1] != self.geometry_channels:
                raise ValueError(
                    f"Expected {self.geometry_channels} geometry channels, got "
                    f"{level.shape[1]}."
                )
            if spatial_shape is None:
                spatial_shape = tuple(level.shape[-2:])
            elif tuple(level.shape[-2:]) != spatial_shape:
                raise ValueError("All geometry levels must share one spatial grid.")
        return levels

    @staticmethod
    def _resize_confidence(
        confidence: torch.Tensor,
        *,
        size: tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        confidence = confidence.detach().float()
        if confidence.ndim == 4 and confidence.shape[-1] == 1:
            confidence = confidence[..., 0]
        if confidence.ndim == 3:
            confidence = confidence[:, None]
        if confidence.ndim != 4 or confidence.shape[1] != 1:
            raise ValueError(
                "Geometry confidence must be [T,H,W] or [T,H,W,1], got "
                f"{tuple(confidence.shape)}."
            )
        confidence = torch.nan_to_num(confidence, nan=0.0, posinf=0.0, neginf=0.0)
        confidence = F.interpolate(
            confidence,
            size=size,
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)
        return confidence.to(device=device, dtype=dtype)
