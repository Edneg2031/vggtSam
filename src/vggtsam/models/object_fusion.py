"""Object-level model built on latent geometry-semantic fusion."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .fusion import LatentGeometrySemanticFusion
from .tokens import GeometryTokens, SemanticTokens


@dataclass
class ObjectFusionOutput:
    logits: torch.Tensor
    centroids_3d: torch.Tensor
    embeddings: torch.Tensor


class ObjectFusionModel(nn.Module):
    """Fuse dense geometry maps with object queries from masks."""

    def __init__(
        self,
        *,
        geometry_dim: int,
        object_dim: int,
        camera_dim: int = 9,
        d_fuse: int = 256,
        num_heads: int = 8,
        num_classes: int = 1024,
    ) -> None:
        super().__init__()
        self.fusion = LatentGeometrySemanticFusion(
            geometry_dim=geometry_dim,
            semantic_dim=object_dim,
            camera_dim=camera_dim,
            d_fuse=d_fuse,
            num_heads=num_heads,
            num_classes=num_classes,
        )

    def forward(
        self,
        *,
        geometry_tokens: torch.Tensor,
        object_tokens: torch.Tensor,
        camera_tokens: torch.Tensor,
    ) -> ObjectFusionOutput:
        output = self.fusion(
            GeometryTokens(tokens=geometry_tokens, camera_tokens=camera_tokens),
            SemanticTokens(tokens=object_tokens),
        )
        return ObjectFusionOutput(
            logits=output.pred_logits,
            centroids_3d=output.pred_pointmap,
            embeddings=output.match_embeddings,
        )
