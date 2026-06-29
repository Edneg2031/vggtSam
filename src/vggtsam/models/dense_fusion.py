"""Dense prompt-conditioned SAM3/StreamVGGT fusion model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .tokens import GeometryTokens, SemanticTokens


@dataclass
class DenseFusionOutput:
    fused_tokens: torch.Tensor
    mask_logits: torch.Tensor
    pointmap: torch.Tensor
    semantic_embedding: torch.Tensor
    prompt_score: torch.Tensor
    instance_embedding: torch.Tensor
    aux_logits: torch.Tensor | None = None


class DenseSAMVGGTModel(nn.Module):
    """Fuse latent SAM3/StreamVGGT tokens and decode dense image-grid outputs."""

    def __init__(
        self,
        *,
        sam_dim: int,
        geometry_dim: int,
        text_dim: int,
        camera_dim: int | None = None,
        d_fuse: int = 256,
        num_heads: int = 8,
        output_size: tuple[int, int] = (256, 384),
        feature_grid: tuple[int, int] = (72, 72),
        embedding_dim: int = 256,
        num_classes: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.feature_grid = tuple(int(v) for v in feature_grid)
        self.output_size = tuple(int(v) for v in output_size)
        self.text_dim = int(text_dim)
        self.embedding_dim = int(embedding_dim)

        self.proj_sam = nn.Linear(sam_dim, d_fuse)
        self.proj_geometry = nn.Linear(geometry_dim, d_fuse)
        self.proj_camera = (
            nn.Linear(camera_dim, d_fuse) if camera_dim is not None else None
        )
        self.context_norm = nn.LayerNorm(d_fuse)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_fuse,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.fusion_norm = nn.LayerNorm(d_fuse)

        self.decoder = nn.Sequential(
            nn.Conv2d(d_fuse, d_fuse, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=d_fuse),
            nn.GELU(),
            nn.Conv2d(d_fuse, d_fuse, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=d_fuse),
            nn.GELU(),
        )
        self.mask_head = nn.Conv2d(d_fuse, 1, kernel_size=1)
        self.point_head = nn.Sequential(
            nn.Conv2d(d_fuse, d_fuse, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(d_fuse, 3, kernel_size=1),
        )
        self.semantic_head = nn.Sequential(
            nn.Conv2d(d_fuse, d_fuse, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(d_fuse, self.text_dim, kernel_size=1),
        )
        self.instance_head = nn.Sequential(
            nn.Conv2d(d_fuse, d_fuse, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(d_fuse, self.embedding_dim, kernel_size=1),
        )
        self.aux_head = (
            nn.Conv2d(d_fuse, int(num_classes), kernel_size=1)
            if num_classes is not None and int(num_classes) > 0
            else None
        )
        self.prompt_logit_scale = nn.Parameter(torch.tensor(10.0))

    def forward(
        self,
        *,
        sam_tokens: torch.Tensor,
        geometry_tokens: torch.Tensor,
        text_embedding: torch.Tensor,
        camera_tokens: torch.Tensor | None = None,
    ) -> DenseFusionOutput:
        fused_tokens = self.fuse_tokens(
            sam_tokens=sam_tokens,
            geometry_tokens=geometry_tokens,
            camera_tokens=camera_tokens,
        )
        dense = self.tokens_to_dense(fused_tokens)
        mask_logits = self.mask_head(dense).squeeze(1)
        pointmap = self.point_head(dense).permute(0, 2, 3, 1).contiguous()
        semantic_embedding = self.semantic_head(dense)
        instance_embedding = F.normalize(self.instance_head(dense), dim=1)
        prompt_score = self.compute_prompt_score(
            semantic_embedding,
            text_embedding,
        )
        aux_logits = self.aux_head(dense) if self.aux_head is not None else None
        return DenseFusionOutput(
            fused_tokens=fused_tokens,
            mask_logits=mask_logits,
            pointmap=pointmap,
            semantic_embedding=semantic_embedding.permute(0, 2, 3, 1).contiguous(),
            prompt_score=prompt_score,
            instance_embedding=instance_embedding.permute(0, 2, 3, 1).contiguous(),
            aux_logits=aux_logits,
        )

    def fuse_tokens(
        self,
        *,
        sam_tokens: torch.Tensor,
        geometry_tokens: torch.Tensor,
        camera_tokens: torch.Tensor | None,
    ) -> torch.Tensor:
        sam_tokens = self._ensure_batched(sam_tokens)
        geometry_tokens = self._ensure_batched(geometry_tokens)
        context = [self.proj_geometry(geometry_tokens)]
        if camera_tokens is not None:
            if self.proj_camera is None:
                raise ValueError("camera_tokens were provided but camera_dim is None")
            context.append(self.proj_camera(self._ensure_batched(camera_tokens)))
        context_tokens = self.context_norm(torch.cat(context, dim=1))
        query = self.proj_sam(sam_tokens)
        fused, _ = self.cross_attention(
            query=query,
            key=context_tokens,
            value=context_tokens,
            need_weights=False,
        )
        return self.fusion_norm(fused + query)

    def tokens_to_dense(self, fused_tokens: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = fused_tokens.shape
        grid_h, grid_w = self.feature_grid
        expected = grid_h * grid_w
        if tokens != expected:
            raise ValueError(
                f"Expected {expected} fused tokens from feature_grid={self.feature_grid}, "
                f"got {tokens}"
            )
        dense = fused_tokens.transpose(1, 2).reshape(batch, channels, grid_h, grid_w)
        dense = F.interpolate(
            dense,
            size=self.output_size,
            mode="bilinear",
            align_corners=False,
        )
        return self.decoder(dense)

    def compute_prompt_score(
        self,
        semantic_embedding: torch.Tensor,
        text_embedding: torch.Tensor,
    ) -> torch.Tensor:
        if text_embedding.ndim == 1:
            text_embedding = text_embedding[None]
        if text_embedding.ndim != 2:
            raise ValueError(
                f"text_embedding must have shape [B, C] or [C], got {tuple(text_embedding.shape)}"
            )
        if text_embedding.shape[0] == 1 and semantic_embedding.shape[0] > 1:
            text_embedding = text_embedding.expand(semantic_embedding.shape[0], -1)
        if semantic_embedding.shape[1] != text_embedding.shape[-1]:
            raise ValueError(
                "semantic embedding channel and text embedding dimension differ: "
                f"{semantic_embedding.shape[1]} vs {text_embedding.shape[-1]}"
            )
        semantic = F.normalize(semantic_embedding, dim=1)
        text = F.normalize(text_embedding, dim=-1)
        scale = self.prompt_logit_scale.clamp(1.0, 100.0)
        return torch.einsum("bchw,bc->bhw", semantic, text) * scale

    @staticmethod
    def _ensure_batched(tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim == 2:
            return tokens.unsqueeze(0)
        if tokens.ndim != 3:
            raise ValueError(f"Expected tokens [B, N, C] or [N, C], got {tuple(tokens.shape)}")
        return tokens
