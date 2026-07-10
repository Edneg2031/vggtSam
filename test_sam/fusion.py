"""Ablatable StreamVGGT-to-SAM3 feature mergers.

Every merger returns residuals for SAM3 tracker FPN0/FPN1/FPN2. The original
SAM3 features remain the main branch. A tiny residual initialization keeps all
methods near the same pretrained tracker behavior without creating a dead gate.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
import torch.nn.functional as F


FUSION_METHODS = (
    "sam_only",
    "add",
    "concat_conv",
    "film",
    "cross_attention",
    "gated_cross_attention",
    "multilevel_cross_attention",
)


class SAM3GeometryFusion(nn.Module):
    """Fuse SAM3 FPN2 with StreamVGGT latent geometry and emit FPN residuals."""

    def __init__(
        self,
        *,
        method: str,
        sam_channels: int = 256,
        geometry_channels: int = 2048,
        hidden_channels: int = 256,
        num_heads: int = 8,
        dropout: float = 0.0,
        inject_levels: Sequence[str] = ("fpn2",),
        residual_init_std: float = 1e-4,
    ) -> None:
        super().__init__()
        method = method.strip().lower()
        if method not in FUSION_METHODS:
            raise ValueError(f"Unknown fusion method {method!r}; choose from {FUSION_METHODS}")
        self.method = method
        self.sam_channels = int(sam_channels)
        self.geometry_channels = int(geometry_channels)
        self.hidden_channels = int(hidden_channels)
        self.residual_init_std = float(residual_init_std)
        if self.residual_init_std < 0:
            raise ValueError("residual_init_std must be non-negative.")
        self.inject_levels = {
            str(level).strip().lower() for level in inject_levels
        }
        invalid_levels = self.inject_levels - {"fpn0", "fpn1", "fpn2"}
        if invalid_levels:
            raise ValueError(f"Unknown SAM3 FPN injection levels: {invalid_levels}")

        self.sam_proj = nn.Sequential(
            nn.Conv2d(self.sam_channels, self.hidden_channels, kernel_size=1),
            nn.GroupNorm(8, self.hidden_channels),
            nn.GELU(),
        )
        self.geometry_proj = nn.Conv2d(
            self.geometry_channels,
            self.hidden_channels,
            kernel_size=1,
        )
        self.concat_refine = ConvRefine(self.hidden_channels * 2, self.hidden_channels)
        self.add_refine = ConvRefine(self.hidden_channels, self.hidden_channels)
        self.film = nn.Sequential(
            nn.Linear(self.geometry_channels, self.hidden_channels * 2),
            nn.GELU(),
            nn.Linear(self.hidden_channels * 2, self.hidden_channels * 2),
        )
        self.sam_norm = nn.LayerNorm(self.hidden_channels)
        self.geometry_norm = nn.LayerNorm(self.geometry_channels)
        self.query_proj = nn.Linear(self.hidden_channels, self.hidden_channels)
        self.key_proj = nn.Linear(self.geometry_channels, self.hidden_channels)
        self.value_proj = nn.Linear(self.geometry_channels, self.hidden_channels)
        self.cross_attention = nn.MultiheadAttention(
            self.hidden_channels,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_out = nn.Linear(self.hidden_channels, self.hidden_channels)
        self.attention_norm = nn.LayerNorm(self.hidden_channels)
        self.attention_gate = nn.Sequential(
            nn.Linear(self.hidden_channels * 2, self.hidden_channels),
            nn.GELU(),
            nn.Linear(self.hidden_channels, self.hidden_channels),
            nn.Sigmoid(),
        )

        self.multilevel_proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(self.geometry_channels),
                    nn.Linear(self.geometry_channels, self.hidden_channels),
                )
                for _ in range(4)
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
                for _ in range(3)
            ]
        )
        self.geometry_level_norms = nn.ModuleList(
            [nn.LayerNorm(self.hidden_channels) for _ in range(4)]
        )
        self.multilevel_refine = ConvRefine(
            self.hidden_channels * 2,
            self.hidden_channels,
        )

        self.fpn2_head = residual_head(self.hidden_channels, 256)
        self.fpn1_head = residual_head(self.hidden_channels, 64)
        self.fpn0_head = residual_head(self.hidden_channels, 32)
        self._initialize_residual_outputs()
        self._freeze_unused_branches()

    def forward(
        self,
        sam_fpn2: torch.Tensor,
        geometry_levels: Sequence[torch.Tensor] | None,
    ) -> list[torch.Tensor]:
        """Return [FPN0, FPN1, FPN2] residuals.

        Args:
            sam_fpn2: [T, 256, 72, 72].
            geometry_levels: StreamVGGT maps [T, 2048, Hg, Wg]. The last level
                is used by single-level methods; all four are used by the
                3AM-like multilevel merger.
        """
        if sam_fpn2.ndim != 4:
            raise ValueError(f"sam_fpn2 must be [T,C,H,W], got {sam_fpn2.shape}")
        sam = self.sam_proj(sam_fpn2)

        if self.method == "sam_only":
            fused = self.add_refine(sam)
        else:
            levels = self._validate_geometry(geometry_levels, sam.shape[0])
            if self.method == "add":
                geometry = self._geometry_map(levels[-1], sam.shape[-2:])
                fused = self.add_refine(sam + geometry)
            elif self.method == "concat_conv":
                geometry = self._geometry_map(levels[-1], sam.shape[-2:])
                fused = self.concat_refine(torch.cat([sam, geometry], dim=1))
            elif self.method == "film":
                pooled = levels[-1].mean(dim=(-2, -1))
                gamma, beta = self.film(pooled).chunk(2, dim=-1)
                fused = self.add_refine(
                    sam * (1.0 + gamma[:, :, None, None])
                    + beta[:, :, None, None]
                )
            elif self.method in {"cross_attention", "gated_cross_attention"}:
                fused = self._cross_attention_fusion(
                    sam,
                    levels[-1],
                    gated=self.method == "gated_cross_attention",
                )
            elif self.method == "multilevel_cross_attention":
                fused = self._multilevel_fusion(sam, levels)
            else:
                raise AssertionError(self.method)

        fpn2 = (
            self.fpn2_head(fused)
            if "fpn2" in self.inject_levels
            else fused.new_zeros(fused.shape[0], 256, 72, 72)
        )
        fpn1 = (
            self.fpn1_head(
                F.interpolate(
                    fused,
                    size=(144, 144),
                    mode="bilinear",
                    align_corners=False,
                )
            )
            if "fpn1" in self.inject_levels
            else fused.new_zeros(fused.shape[0], 64, 144, 144)
        )
        fpn0 = (
            self.fpn0_head(
                F.interpolate(
                    fused,
                    size=(288, 288),
                    mode="bilinear",
                    align_corners=False,
                )
            )
            if "fpn0" in self.inject_levels
            else fused.new_zeros(fused.shape[0], 32, 288, 288)
        )
        return [fpn0, fpn1, fpn2]

    def _geometry_map(
        self,
        geometry: torch.Tensor,
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        projected = self.geometry_proj(geometry)
        return F.interpolate(
            projected,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )

    def _cross_attention_fusion(
        self,
        sam: torch.Tensor,
        geometry: torch.Tensor,
        *,
        gated: bool,
    ) -> torch.Tensor:
        batch, channels, height, width = sam.shape
        sam_tokens = sam.flatten(2).transpose(1, 2)
        geometry_tokens = geometry.flatten(2).transpose(1, 2)
        query = self.query_proj(self.sam_norm(sam_tokens))
        geometry_tokens = self.geometry_norm(geometry_tokens)
        key = self.key_proj(geometry_tokens)
        value = self.value_proj(geometry_tokens)
        attended, _ = self.cross_attention(
            query=query,
            key=key,
            value=value,
            need_weights=False,
        )
        attended = self.attention_out(attended)
        if gated:
            gate = self.attention_gate(torch.cat([sam_tokens, attended], dim=-1))
            attended = attended * gate
        tokens = self.attention_norm(sam_tokens + attended)
        return tokens.transpose(1, 2).reshape(batch, channels, height, width)

    def _multilevel_fusion(
        self,
        sam: torch.Tensor,
        geometry_levels: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        if len(geometry_levels) != 4:
            raise ValueError(
                "multilevel_cross_attention requires exactly four StreamVGGT "
                f"levels, got {len(geometry_levels)}"
            )
        projected = [
            projection(level.flatten(2).transpose(1, 2))
            for projection, level in zip(self.multilevel_proj, geometry_levels)
        ]
        merged, _ = self.geometry_self_attention(
            projected[0],
            projected[0],
            projected[0],
            need_weights=False,
        )
        merged = self.geometry_level_norms[0](projected[0] + merged)
        for index, (attention, level) in enumerate(
            zip(self.geometry_cross_attention, projected[1:]),
            start=1,
        ):
            update, _ = attention(merged, level, level, need_weights=False)
            merged = self.geometry_level_norms[index](merged + update)

        batch, _, geometry_height, geometry_width = geometry_levels[0].shape
        merged_map = merged.transpose(1, 2).reshape(
            batch,
            self.hidden_channels,
            geometry_height,
            geometry_width,
        )
        merged_map = F.interpolate(
            merged_map,
            size=sam.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return self.multilevel_refine(torch.cat([sam, merged_map], dim=1))

    def _validate_geometry(
        self,
        geometry_levels: Sequence[torch.Tensor] | None,
        batch_size: int,
    ) -> list[torch.Tensor]:
        if not geometry_levels:
            raise ValueError(f"Fusion method {self.method!r} requires geometry features.")
        levels = list(geometry_levels)
        for level in levels:
            if level.ndim != 4:
                raise ValueError(f"Geometry level must be [T,C,H,W], got {level.shape}")
            if level.shape[0] != batch_size:
                raise ValueError(
                    f"SAM/geometry frame mismatch: {batch_size} vs {level.shape[0]}"
                )
            if level.shape[1] != self.geometry_channels:
                raise ValueError(
                    f"Expected {self.geometry_channels} geometry channels, "
                    f"got {level.shape[1]}"
                )
        return levels

    def _initialize_residual_outputs(self) -> None:
        for head in (self.fpn0_head, self.fpn1_head, self.fpn2_head):
            output = head[-1]
            if self.residual_init_std == 0:
                nn.init.zeros_(output.weight)
            else:
                nn.init.normal_(output.weight, mean=0.0, std=self.residual_init_std)
            nn.init.zeros_(output.bias)

    def _freeze_unused_branches(self) -> None:
        active = {
            "sam_proj",
        }
        active.update(f"{level}_head" for level in self.inject_levels)
        if self.method == "sam_only":
            active.add("add_refine")
        elif self.method == "add":
            active.update({"geometry_proj", "add_refine"})
        elif self.method == "concat_conv":
            active.update({"geometry_proj", "concat_refine"})
        elif self.method == "film":
            active.update({"film", "add_refine"})
        elif self.method in {"cross_attention", "gated_cross_attention"}:
            active.update(
                {
                    "sam_norm",
                    "geometry_norm",
                    "query_proj",
                    "key_proj",
                    "value_proj",
                    "cross_attention",
                    "attention_out",
                    "attention_norm",
                }
            )
            if self.method == "gated_cross_attention":
                active.add("attention_gate")
        elif self.method == "multilevel_cross_attention":
            active.update(
                {
                    "multilevel_proj",
                    "geometry_self_attention",
                    "geometry_cross_attention",
                    "geometry_level_norms",
                    "multilevel_refine",
                }
            )
        for name, parameter in self.named_parameters():
            root = name.split(".", 1)[0]
            parameter.requires_grad_(root in active)


class ConvRefine(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, output_channels),
            nn.GELU(),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, output_channels),
            nn.GELU(),
        )


def residual_head(input_channels: int, output_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1),
        nn.GELU(),
        nn.Conv2d(input_channels, output_channels, kernel_size=1),
    )


def stream_tokens_to_maps(
    layer_tokens: Sequence[torch.Tensor],
    *,
    patch_start_idx: int,
    patch_shape: tuple[int, int],
    output_grid: tuple[int, int],
    zero_geometry: bool = False,
) -> list[torch.Tensor]:
    """Convert cached StreamVGGT layers to [T,C,H,W] geometry maps."""
    maps: list[torch.Tensor] = []
    patch_height, patch_width = patch_shape
    expected = patch_height * patch_width
    for tokens in layer_tokens:
        if tokens.ndim != 4 or tokens.shape[0] != 1:
            raise ValueError(
                "StreamVGGT layer tokens must be [1,T,N,C], "
                f"got {tuple(tokens.shape)}"
            )
        patches = tokens[0, :, int(patch_start_idx) :, :]
        if patches.shape[1] != expected:
            raise ValueError(
                f"Expected {expected} patch tokens, got {patches.shape[1]}"
            )
        feature = patches.reshape(
            patches.shape[0],
            patch_height,
            patch_width,
            patches.shape[-1],
        ).permute(0, 3, 1, 2)
        feature = F.interpolate(
            feature.float(),
            size=output_grid,
            mode="bilinear",
            align_corners=False,
        )
        if zero_geometry:
            feature = torch.zeros_like(feature)
        maps.append(feature)
    return maps
