"""Dense latent SAM3/StreamVGGT fusion model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .fusion import LatentGeometrySemanticFusion
from .tokens import GeometryTokens, SemanticTokens


@dataclass
class LatentFusionOutput:
    fused_tokens: torch.Tensor
    logits: torch.Tensor
    pointmap: torch.Tensor
    embeddings: torch.Tensor


class LatentSAMVGGTModel(nn.Module):
    """Fuse SAM3 intermediate tokens with StreamVGGT latent geometry tokens."""

    def __init__(
        self,
        *,
        sam_dim: int,
        geometry_dim: int,
        camera_dim: int = 9,
        d_fuse: int = 256,
        num_heads: int = 8,
        num_classes: int = 1024,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.fusion = LatentGeometrySemanticFusion(
            geometry_dim=geometry_dim,
            semantic_dim=sam_dim,
            camera_dim=camera_dim,
            d_fuse=d_fuse,
            num_heads=num_heads,
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(
        self,
        *,
        sam_tokens: torch.Tensor,
        geometry_tokens: torch.Tensor,
        camera_tokens: torch.Tensor | None = None,
    ) -> LatentFusionOutput:
        output = self.fusion(
            GeometryTokens(tokens=geometry_tokens, camera_tokens=camera_tokens),
            SemanticTokens(tokens=sam_tokens),
        )
        return LatentFusionOutput(
            fused_tokens=output.fused_tokens,
            logits=output.pred_logits,
            pointmap=output.pred_pointmap,
            embeddings=output.match_embeddings,
        )
