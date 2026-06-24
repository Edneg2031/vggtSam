"""Latent geometry-semantic fusion model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from .tokens import GeometryTokens, SemanticTokens


@dataclass
class FusionOutput:
    fused_tokens: torch.Tensor
    pred_pointmap: torch.Tensor
    pred_logits: torch.Tensor
    match_embeddings: torch.Tensor
    attention_weights: Optional[torch.Tensor] = None


class LatentGeometrySemanticFusion(nn.Module):
    """Fuse SAM-style semantic tokens with VGGT-style geometry tokens.

    The module intentionally consumes generic token containers. Adapters can
    choose which backbone layer to expose after we inspect the real server-side
    outputs.
    """

    def __init__(
        self,
        geometry_dim: int,
        semantic_dim: int,
        *,
        camera_dim: Optional[int] = None,
        d_fuse: int = 512,
        num_classes: int = 256,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.geometry_dim = geometry_dim
        self.semantic_dim = semantic_dim
        self.camera_dim = camera_dim
        self.d_fuse = d_fuse

        self.proj_geometry = nn.Linear(geometry_dim, d_fuse)
        self.proj_semantic = nn.Linear(semantic_dim, d_fuse)
        self.proj_camera = (
            nn.Linear(camera_dim, d_fuse) if camera_dim is not None else None
        )

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_fuse,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.fusion_norm = nn.LayerNorm(d_fuse)
        self.context_norm = nn.LayerNorm(d_fuse)

        self.point_head = nn.Sequential(
            nn.Linear(d_fuse, d_fuse),
            nn.GELU(),
            nn.Linear(d_fuse, 3),
        )
        self.semantic_head = nn.Sequential(
            nn.Linear(d_fuse, d_fuse),
            nn.GELU(),
            nn.Linear(d_fuse, num_classes),
        )
        self.match_head = nn.Sequential(
            nn.Linear(d_fuse, d_fuse),
            nn.GELU(),
            nn.Linear(d_fuse, d_fuse),
        )

    def forward(
        self,
        geometry: GeometryTokens,
        semantics: SemanticTokens,
        *,
        return_attention: bool = False,
    ) -> FusionOutput:
        geometry_tokens = self._ensure_batched_tokens(geometry.tokens, "geometry")
        semantic_tokens = self._ensure_batched_tokens(semantics.tokens, "semantic")

        geometry_context = [self.proj_geometry(geometry_tokens)]
        if geometry.camera_tokens is not None:
            if self.proj_camera is None:
                raise ValueError("camera_tokens were provided but camera_dim is None")
            camera_tokens = self._ensure_batched_tokens(
                geometry.camera_tokens, "camera"
            )
            geometry_context.append(self.proj_camera(camera_tokens))

        context = self.context_norm(torch.cat(geometry_context, dim=1))
        query = self.proj_semantic(semantic_tokens)
        fused, attention = self.cross_attention(
            query=query,
            key=context,
            value=context,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        fused = self.fusion_norm(fused + query)

        pred_pointmap = self.point_head(fused)
        pred_logits = self.semantic_head(fused)
        match_embeddings = F.normalize(self.match_head(fused), dim=-1)
        return FusionOutput(
            fused_tokens=fused,
            pred_pointmap=pred_pointmap,
            pred_logits=pred_logits,
            match_embeddings=match_embeddings,
            attention_weights=attention if return_attention else None,
        )

    def correspondence_logits(
        self,
        current_embeddings: torch.Tensor,
        history_embeddings: torch.Tensor,
        *,
        temperature: float = 0.07,
    ) -> torch.Tensor:
        current_embeddings = F.normalize(current_embeddings, dim=-1)
        history_embeddings = F.normalize(history_embeddings, dim=-1)
        return torch.matmul(
            current_embeddings, history_embeddings.transpose(-1, -2)
        ) / temperature

    @staticmethod
    def _ensure_batched_tokens(tokens: torch.Tensor, name: str) -> torch.Tensor:
        if tokens.ndim == 2:
            return tokens.unsqueeze(0)
        if tokens.ndim != 3:
            raise ValueError(
                f"{name} tokens must have shape [B, N, D] or [N, D], got {tuple(tokens.shape)}"
            )
        return tokens
