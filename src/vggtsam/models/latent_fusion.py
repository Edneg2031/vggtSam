"""Dense latent SAM3/StreamVGGT fusion model."""

from __future__ import annotations

from dataclasses import dataclass

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
    ) -> None:
        super().__init__()
        self.token_grid = tuple(int(v) for v in token_grid)
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
        self.mask_feature_head = nn.Sequential(
            nn.Linear(d_fuse, d_fuse),
            nn.GELU(),
            nn.Linear(d_fuse, d_fuse),
        )
        self.mask_query_head = nn.Sequential(
            nn.Linear(d_fuse, d_fuse),
            nn.GELU(),
            nn.Linear(d_fuse, d_fuse),
        )
        self.mask_logit_scale = nn.Parameter(torch.tensor(10.0))

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

    def compute_mask_correspondence(
        self,
        current_embeddings: torch.Tensor,
        history_embeddings: torch.Tensor,
        *,
        temperature: float = 0.07,
    ) -> torch.Tensor:
        """Return current-to-history token correspondence logits."""
        current_embeddings = F.normalize(current_embeddings, dim=-1)
        history_embeddings = F.normalize(history_embeddings, dim=-1)
        return torch.matmul(
            current_embeddings,
            history_embeddings.transpose(-1, -2),
        ) / max(float(temperature), 1e-6)

    def build_mask_prototype(
        self,
        ref_fused_tokens: torch.Tensor,
        ref_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Build an instance prototype from fused tokens inside a reference mask."""
        if ref_fused_tokens.ndim != 3:
            raise ValueError(
                "ref_fused_tokens must have shape [B, N, C], "
                f"got {tuple(ref_fused_tokens.shape)}"
            )
        if ref_mask.ndim == 1:
            ref_mask = ref_mask[None]
        if ref_mask.shape != ref_fused_tokens.shape[:2]:
            raise ValueError(
                f"ref_mask must have shape {tuple(ref_fused_tokens.shape[:2])}, "
                f"got {tuple(ref_mask.shape)}"
            )
        weights = ref_mask.to(ref_fused_tokens.dtype).unsqueeze(-1)
        query_features = self.mask_query_head(ref_fused_tokens)
        prototype = (query_features * weights).sum(dim=1)
        prototype = prototype / weights.sum(dim=1).clamp_min(1.0)
        return F.normalize(prototype, dim=-1)

    def decode_mask_from_prototype(
        self,
        fused_tokens: torch.Tensor,
        prototype: torch.Tensor,
    ) -> torch.Tensor:
        """Decode token-level instance mask logits from a reference prototype."""
        if fused_tokens.ndim != 3:
            raise ValueError(
                f"fused_tokens must have shape [B, N, C], got {tuple(fused_tokens.shape)}"
            )
        if prototype.ndim == 1:
            prototype = prototype[None]
        mask_features = F.normalize(self.mask_feature_head(fused_tokens), dim=-1)
        prototype = F.normalize(prototype, dim=-1)
        scale = self.mask_logit_scale.clamp(1.0, 100.0)
        return torch.einsum("bnc,bc->bn", mask_features, prototype) * scale
