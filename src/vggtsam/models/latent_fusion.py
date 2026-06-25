"""Dense latent SAM3/StreamVGGT fusion model."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from .fusion import LatentGeometrySemanticFusion
from .tokens import GeometryTokens, SemanticTokens


@dataclass
class LatentFusionOutput:
    fused_tokens: torch.Tensor
    logits: torch.Tensor
    pointmap: torch.Tensor
    embeddings: torch.Tensor
    object_logits: Optional[torch.Tensor] = None
    object_points: Optional[torch.Tensor] = None
    object_embeddings: Optional[torch.Tensor] = None
    mask_logits: Optional[torch.Tensor] = None


class LatentSAMVGGTModel(nn.Module):
    """Fuse SAM3 intermediate tokens with StreamVGGT latent geometry tokens."""

    def __init__(
        self,
        *,
        sam_dim: int,
        geometry_dim: int,
        camera_dim: int | None = 9,
        d_fuse: int = 256,
        num_heads: int = 8,
        num_classes: int = 1024,
        dropout: float = 0.0,
        token_grid: tuple[int, int] = (72, 72),
        mask_grid: tuple[int, int] = (144, 144),
        num_queries: int = 32,
    ) -> None:
        super().__init__()
        self.token_grid = tuple(int(v) for v in token_grid)
        self.mask_grid = tuple(int(v) for v in mask_grid)
        self.num_queries = int(num_queries)
        self.d_fuse = int(d_fuse)
        self.fusion = LatentGeometrySemanticFusion(
            geometry_dim=geometry_dim,
            semantic_dim=sam_dim,
            camera_dim=camera_dim,
            d_fuse=d_fuse,
            num_heads=num_heads,
            num_classes=num_classes,
            dropout=dropout,
        )
        self.object_queries = nn.Parameter(torch.randn(self.num_queries, d_fuse) * 0.02)
        self.object_attention = nn.MultiheadAttention(
            embed_dim=d_fuse,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.object_norm = nn.LayerNorm(d_fuse)
        self.object_semantic_head = nn.Sequential(
            nn.Linear(d_fuse, d_fuse),
            nn.GELU(),
            nn.Linear(d_fuse, num_classes),
        )
        self.object_point_head = nn.Sequential(
            nn.Linear(d_fuse, d_fuse),
            nn.GELU(),
            nn.Linear(d_fuse, 3),
        )
        self.object_match_head = nn.Sequential(
            nn.Linear(d_fuse, d_fuse),
            nn.GELU(),
            nn.Linear(d_fuse, d_fuse),
        )
        self.mask_feature_head = nn.Sequential(
            nn.Linear(d_fuse, d_fuse),
            nn.GELU(),
            nn.Linear(d_fuse, d_fuse),
        )
        self.mask_embed_head = nn.Sequential(
            nn.Linear(d_fuse, d_fuse),
            nn.GELU(),
            nn.Linear(d_fuse, d_fuse),
        )

    def forward(
        self,
        *,
        sam_tokens: torch.Tensor,
        geometry_tokens: torch.Tensor,
        camera_tokens: torch.Tensor | None = None,
        num_frames: int | None = None,
    ) -> LatentFusionOutput:
        output = self.fusion(
            GeometryTokens(tokens=geometry_tokens, camera_tokens=camera_tokens),
            SemanticTokens(tokens=sam_tokens),
        )
        object_logits = None
        object_points = None
        object_embeddings = None
        mask_logits = None
        if num_frames is not None:
            object_logits, object_points, object_embeddings, mask_logits = (
                self.decode_objects(output.fused_tokens, num_frames=num_frames)
            )
        return LatentFusionOutput(
            fused_tokens=output.fused_tokens,
            logits=output.pred_logits,
            pointmap=output.pred_pointmap,
            embeddings=output.match_embeddings,
            object_logits=object_logits,
            object_points=object_points,
            object_embeddings=object_embeddings,
            mask_logits=mask_logits,
        )

    def decode_objects(
        self,
        fused_tokens: torch.Tensor,
        *,
        num_frames: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, total_tokens, channels = fused_tokens.shape
        token_h, token_w = self.token_grid
        tokens_per_frame = token_h * token_w
        expected = int(num_frames) * tokens_per_frame
        if total_tokens != expected:
            raise ValueError(
                f"Expected {expected} fused tokens for num_frames={num_frames} "
                f"and token_grid={self.token_grid}, got {total_tokens}"
            )

        frame_tokens = fused_tokens.reshape(
            batch, int(num_frames), tokens_per_frame, channels
        )
        flat_tokens = frame_tokens.reshape(
            batch * int(num_frames), tokens_per_frame, channels
        )
        query = self.object_queries[None].expand(flat_tokens.shape[0], -1, -1)
        objects, _ = self.object_attention(
            query=query,
            key=flat_tokens,
            value=flat_tokens,
            need_weights=False,
        )
        objects = self.object_norm(objects + query)
        objects = objects.reshape(batch, int(num_frames), self.num_queries, channels)

        object_logits = self.object_semantic_head(objects)
        object_points = self.object_point_head(objects)
        object_embeddings = F.normalize(self.object_match_head(objects), dim=-1)

        mask_features = self.mask_feature_head(frame_tokens)
        mask_features = mask_features.reshape(
            batch * int(num_frames), token_h, token_w, channels
        ).permute(0, 3, 1, 2)
        mask_features = F.interpolate(
            mask_features,
            size=self.mask_grid,
            mode="bilinear",
            align_corners=False,
        )
        mask_features = mask_features.reshape(
            batch, int(num_frames), channels, self.mask_grid[0], self.mask_grid[1]
        )
        mask_embeddings = self.mask_embed_head(objects)
        mask_logits = torch.einsum(
            "btqc,btchw->btqhw",
            mask_embeddings,
            mask_features,
        ) / math.sqrt(float(channels))
        return object_logits, object_points, object_embeddings, mask_logits
