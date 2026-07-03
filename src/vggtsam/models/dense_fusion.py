"""Dense prompt-conditioned SAM3/StreamVGGT fusion model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .tokens import GeometryTokens, SemanticTokens


class CameraGuidedTokenFusion(nn.Module):
    """Camera-guided geometry-to-visual fusion.

    This keeps the visual token shape unchanged: SAM3 tokens remain the query
    stream, while StreamVGGT spatial and camera tokens provide a learned
    geometry residual.
    """

    def __init__(
        self,
        *,
        visual_dim: int,
        spatial_dim: int,
        camera_dim: int | None,
        attention_dim: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.visual_dim = int(visual_dim)
        self.spatial_dim = int(spatial_dim)
        self.camera_dim = int(camera_dim) if camera_dim is not None else None
        self.attention_dim = int(attention_dim)

        self.visual_norm = nn.LayerNorm(self.visual_dim)
        self.spatial_norm = nn.LayerNorm(self.spatial_dim)
        self.query_proj = nn.Linear(self.visual_dim, self.attention_dim)
        self.key_proj = nn.Linear(self.spatial_dim, self.attention_dim)
        self.value_proj = nn.Linear(self.spatial_dim, self.attention_dim)
        self.camera_proj = (
            nn.Linear(self.camera_dim, self.attention_dim)
            if self.camera_dim is not None
            else None
        )
        self.null_camera = nn.Parameter(torch.zeros(1, 1, self.attention_dim))

        self.geom_mlp = nn.Sequential(
            nn.Linear(self.spatial_dim + self.attention_dim, self.attention_dim),
            nn.GELU(),
            nn.Linear(self.attention_dim, self.attention_dim),
            nn.LayerNorm(self.attention_dim),
        )
        self.token_weight_proj = nn.Linear(self.spatial_dim, self.attention_dim)
        self.token_weight_attn = nn.MultiheadAttention(
            embed_dim=self.attention_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.token_weight_mlp = nn.Sequential(
            nn.Linear(self.attention_dim, max(1, self.attention_dim // 4)),
            nn.GELU(),
            nn.Linear(max(1, self.attention_dim // 4), 1),
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.attention_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.out_proj = nn.Linear(self.attention_dim, self.visual_dim)
        self.camera_gate = nn.Linear(self.attention_dim, self.visual_dim)
        self.out_norm = nn.LayerNorm(self.visual_dim)
        self.dropout = nn.Dropout(dropout)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        *,
        visual_tokens: torch.Tensor,
        spatial_tokens: torch.Tensor,
        camera_tokens: torch.Tensor | None,
    ) -> torch.Tensor:
        visual_tokens = ensure_batched(visual_tokens)
        spatial_tokens = ensure_batched(spatial_tokens)
        if spatial_tokens.shape[0] == 1 and visual_tokens.shape[0] > 1:
            spatial_tokens = spatial_tokens.expand(visual_tokens.shape[0], -1, -1)
        if spatial_tokens.shape[0] != visual_tokens.shape[0]:
            raise ValueError(
                "Camera-guided fusion batch mismatch: "
                f"visual={visual_tokens.shape[0]} spatial={spatial_tokens.shape[0]}"
            )

        visual_norm = self.visual_norm(visual_tokens)
        spatial_norm = self.spatial_norm(spatial_tokens)
        query = self.query_proj(visual_norm)
        key = self.key_proj(spatial_norm)
        value = self.value_proj(spatial_norm)

        camera = self._project_camera(
            camera_tokens,
            batch_size=visual_tokens.shape[0],
            device=visual_tokens.device,
            dtype=visual_tokens.dtype,
        )
        camera_summary = camera.mean(dim=1, keepdim=True)
        camera_for_spatial = camera_summary.expand(-1, spatial_norm.shape[1], -1)
        geom_bias = self.geom_mlp(
            torch.cat([spatial_norm, camera_for_spatial], dim=-1)
        ).to(dtype=key.dtype)
        key = key + geom_bias
        value = value + geom_bias

        token_input = self.token_weight_proj(spatial_norm)
        token_context, _ = self.token_weight_attn(
            token_input,
            token_input,
            token_input,
            need_weights=False,
        )
        token_weight = torch.sigmoid(self.token_weight_mlp(token_input + token_context))
        value = value * token_weight

        key = torch.cat([camera, key], dim=1)
        value = torch.cat([camera, value], dim=1)
        fused, _ = self.cross_attention(
            query=query,
            key=key,
            value=value,
            need_weights=False,
        )
        fused = self.out_proj(fused)
        gate = torch.sigmoid(self.camera_gate(camera_summary.squeeze(1)))
        fused = self.dropout(fused * gate[:, None])
        return self.out_norm(visual_tokens + self.residual_scale * fused)

    def _project_camera(
        self,
        camera_tokens: torch.Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if camera_tokens is None:
            return self.null_camera.to(device=device, dtype=dtype).expand(
                batch_size,
                -1,
                -1,
            )
        camera_tokens = ensure_batched(camera_tokens).to(device=device, dtype=dtype)
        if camera_tokens.shape[0] == 1 and batch_size > 1:
            camera_tokens = camera_tokens.expand(batch_size, -1, -1)
        if camera_tokens.shape[0] != batch_size:
            raise ValueError(
                "Camera token batch mismatch: "
                f"camera={camera_tokens.shape[0]} visual={batch_size}"
            )
        if self.camera_proj is None:
            if camera_tokens.shape[-1] != self.attention_dim:
                raise ValueError(
                    "Camera tokens were provided but camera_dim=None and token width "
                    f"{camera_tokens.shape[-1]} != attention_dim={self.attention_dim}."
                )
            return camera_tokens
        return self.camera_proj(camera_tokens)


@dataclass
class DenseFusionOutput:
    fused_tokens: torch.Tensor
    mask_logits: torch.Tensor
    fused_sam_mask_logits: torch.Tensor | None
    sam3_direct_mask: torch.Tensor | None
    pointmap: torch.Tensor
    streamvggt_pointmap: torch.Tensor | None
    point_conf: torch.Tensor | None
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
        point_decoder: str = "simple",
        point_mask_condition: str = "none",
        stream_dpt_freeze: bool = False,
        enable_fused_sam_decoder: bool = False,
        fusion_type: str = "simple_cross_attn",
    ) -> None:
        super().__init__()
        self.feature_grid = tuple(int(v) for v in feature_grid)
        self.output_size = tuple(int(v) for v in output_size)
        self.text_dim = int(text_dim)
        self.embedding_dim = int(embedding_dim)
        self.enable_fused_sam_decoder = bool(enable_fused_sam_decoder)
        self.point_decoder = point_decoder.strip().lower()
        self.point_mask_condition = point_mask_condition.strip().lower()
        self.fusion_type = fusion_type.strip().lower()
        if self.point_decoder not in {"simple", "stream_dpt"}:
            raise ValueError(
                "point_decoder must be 'simple' or 'stream_dpt', "
                f"got {point_decoder!r}"
            )
        if self.fusion_type not in {"simple_cross_attn", "camera_guided"}:
            raise ValueError(
                "fusion_type must be 'simple_cross_attn' or 'camera_guided', "
                f"got {fusion_type!r}"
            )
        if self.point_mask_condition not in {"none", "gt_soft"}:
            raise ValueError(
                "point_mask_condition must be 'none' or 'gt_soft', "
                f"got {point_mask_condition!r}"
            )

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
        self.camera_guided_fusion = (
            CameraGuidedTokenFusion(
                visual_dim=d_fuse,
                spatial_dim=geometry_dim,
                camera_dim=camera_dim,
                attention_dim=d_fuse,
                num_heads=num_heads,
                dropout=dropout,
            )
            if self.fusion_type == "camera_guided"
            else None
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(d_fuse, d_fuse, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=d_fuse),
            nn.GELU(),
            nn.Conv2d(d_fuse, d_fuse, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=d_fuse),
            nn.GELU(),
        )
        self.mask_head = nn.Conv2d(d_fuse, 1, kernel_size=1)
        if self.point_decoder == "simple":
            self.point_head = nn.Sequential(
                nn.Conv2d(d_fuse, d_fuse, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(d_fuse, 3, kernel_size=1),
            )
            self.stream_point_decoder = None
            self.stream_condition_proj = None
            self.stream_condition_norm = None
            self.stream_condition_scale = None
            self.stream_mask_condition_proj = None
            self.stream_mask_condition_norm = None
            self.stream_mask_condition_scale = None
        else:
            from streamvggt.heads.dpt_head import DPTHead

            self.point_head = None
            self.stream_point_decoder = DPTHead(
                dim_in=geometry_dim,
                output_dim=4,
                activation="inv_log",
                conf_activation="expp1",
                intermediate_layer_idx=[0, 1, 2, 3],
            )
            self.stream_condition_proj = nn.Linear(d_fuse, geometry_dim)
            self.stream_condition_norm = nn.LayerNorm(geometry_dim)
            self.stream_condition_scale = nn.Parameter(torch.tensor(0.1))
            self.stream_mask_condition_proj = nn.Linear(1, geometry_dim)
            self.stream_mask_condition_norm = nn.LayerNorm(geometry_dim)
            self.stream_mask_condition_scale = nn.Parameter(torch.tensor(0.01))
            if stream_dpt_freeze:
                for param in self.stream_point_decoder.parameters():
                    param.requires_grad = False
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
        self.object_query_proj = nn.Linear(d_fuse, d_fuse)
        self.object_query_to_embedding = nn.Linear(d_fuse, self.embedding_dim)
        self.object_memory_update = nn.GRUCell(d_fuse, d_fuse)
        self.object_memory_norm = nn.LayerNorm(d_fuse)
        if self.enable_fused_sam_decoder:
            self.fused_sam_image_proj = nn.Sequential(
                nn.Conv2d(d_fuse, d_fuse, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(d_fuse, d_fuse, kernel_size=1),
            )
            self.fused_sam_high_s1 = nn.Sequential(
                nn.Conv2d(d_fuse, d_fuse, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(d_fuse, d_fuse // 4, kernel_size=1),
            )
            self.fused_sam_high_s0 = nn.Sequential(
                nn.Conv2d(d_fuse, d_fuse, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(d_fuse, d_fuse // 8, kernel_size=1),
            )
            self.fused_sam_object_proj = nn.Linear(d_fuse, d_fuse)
            self.fused_sam_residual_scale = nn.Parameter(torch.tensor(0.1))
        else:
            self.fused_sam_image_proj = None
            self.fused_sam_high_s1 = None
            self.fused_sam_high_s0 = None
            self.fused_sam_object_proj = None
            self.fused_sam_residual_scale = None
        self.prompt_logit_scale = nn.Parameter(torch.tensor(10.0))
        self.instance_logit_scale = nn.Parameter(torch.tensor(10.0))

    def forward(
        self,
        *,
        sam_tokens: torch.Tensor,
        geometry_tokens: torch.Tensor,
        text_embedding: torch.Tensor,
        camera_tokens: torch.Tensor | None = None,
        object_query: torch.Tensor | None = None,
        stream_tokens: Sequence[torch.Tensor] | None = None,
        stream_images: torch.Tensor | None = None,
        stream_patch_start_idx: int | None = None,
        point_mask_condition: torch.Tensor | None = None,
    ) -> DenseFusionOutput:
        fused_tokens = self.fuse_tokens(
            sam_tokens=sam_tokens,
            geometry_tokens=geometry_tokens,
            camera_tokens=camera_tokens,
        )
        return self.decode(
            fused_tokens=fused_tokens,
            text_embedding=text_embedding,
            object_query=object_query,
            stream_tokens=stream_tokens,
            stream_images=stream_images,
            stream_patch_start_idx=stream_patch_start_idx,
            point_mask_condition=point_mask_condition,
        )

    def decode(
        self,
        *,
        fused_tokens: torch.Tensor,
        text_embedding: torch.Tensor,
        object_query: torch.Tensor | None = None,
        stream_tokens: Sequence[torch.Tensor] | None = None,
        stream_images: torch.Tensor | None = None,
        stream_patch_start_idx: int | None = None,
        point_mask_condition: torch.Tensor | None = None,
    ) -> DenseFusionOutput:
        dense = self.tokens_to_dense(fused_tokens)
        object_query = self._prepare_object_query(
            object_query,
            batch_size=dense.shape[0],
            device=dense.device,
        )
        if object_query is not None:
            dense = dense + self.object_query_proj(object_query)[:, :, None, None]
        mask_logits = self.mask_head(dense).squeeze(1)
        if self.point_decoder == "simple":
            assert self.point_head is not None
            pointmap = self.point_head(dense).permute(0, 2, 3, 1).contiguous()
            point_conf = None
        else:
            pointmap, point_conf = self.decode_stream_pointmap(
                fused_tokens=fused_tokens,
                object_query=object_query,
                stream_tokens=stream_tokens,
                stream_images=stream_images,
                stream_patch_start_idx=stream_patch_start_idx,
                point_mask_condition=point_mask_condition,
            )
        semantic_embedding = self.semantic_head(dense)
        instance_embedding_chw = F.normalize(self.instance_head(dense), dim=1)
        if object_query is not None:
            object_embedding = F.normalize(
                self.object_query_to_embedding(object_query),
                dim=-1,
            )
            instance_score = torch.einsum(
                "bchw,bc->bhw",
                instance_embedding_chw,
                object_embedding,
            )
            scale = self.instance_logit_scale.clamp(1.0, 100.0)
            mask_logits = mask_logits + instance_score * scale
        prompt_score = self.compute_prompt_score(
            semantic_embedding,
            text_embedding,
        )
        aux_logits = self.aux_head(dense) if self.aux_head is not None else None
        return DenseFusionOutput(
            fused_tokens=fused_tokens,
            mask_logits=mask_logits,
            fused_sam_mask_logits=None,
            sam3_direct_mask=None,
            pointmap=pointmap,
            streamvggt_pointmap=None,
            point_conf=point_conf,
            semantic_embedding=semantic_embedding.permute(0, 2, 3, 1).contiguous(),
            prompt_score=prompt_score,
            instance_embedding=instance_embedding_chw.permute(0, 2, 3, 1).contiguous(),
            aux_logits=aux_logits,
        )

    def decode_stream_pointmap(
        self,
        *,
        fused_tokens: torch.Tensor,
        object_query: torch.Tensor | None,
        stream_tokens: Sequence[torch.Tensor] | None,
        stream_images: torch.Tensor | None,
        stream_patch_start_idx: int | None,
        point_mask_condition: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            stream_tokens is None
            or stream_images is None
            or stream_patch_start_idx is None
        ):
            raise ValueError(
                "stream_dpt point decoder requires stream_tokens, stream_images, "
                "and stream_patch_start_idx."
            )
        assert self.stream_point_decoder is not None
        conditioned_tokens = self.condition_stream_tokens(
            stream_tokens,
            fused_tokens=fused_tokens,
            object_query=object_query,
            stream_images=stream_images,
            patch_start_idx=int(stream_patch_start_idx),
            point_mask_condition=point_mask_condition,
        )
        points, confidence = self.stream_point_decoder(
            list(conditioned_tokens),
            images=stream_images,
            patch_start_idx=int(stream_patch_start_idx),
            frames_chunk_size=None,
        )
        points = points[:, 0]
        confidence = confidence[:, 0]
        if points.shape[1:3] != self.output_size:
            points = resize_bhwc(points, self.output_size)
            confidence = resize_bhw(confidence, self.output_size)
        return points.contiguous(), confidence.contiguous()

    def condition_stream_tokens(
        self,
        stream_tokens: Sequence[torch.Tensor],
        *,
        fused_tokens: torch.Tensor,
        object_query: torch.Tensor | None,
        stream_images: torch.Tensor,
        patch_start_idx: int,
        point_mask_condition: torch.Tensor | None,
    ) -> list[torch.Tensor]:
        assert self.stream_condition_proj is not None
        assert self.stream_condition_norm is not None
        assert self.stream_condition_scale is not None
        assert self.stream_mask_condition_proj is not None
        assert self.stream_mask_condition_norm is not None
        assert self.stream_mask_condition_scale is not None
        fused_tokens = self._ensure_batched(fused_tokens)
        object_query = self._prepare_object_query(
            object_query,
            batch_size=fused_tokens.shape[0],
            device=fused_tokens.device,
        )
        if object_query is not None:
            fused_tokens = fused_tokens + self.object_query_proj(object_query)[:, None]

        _, _, _, image_h, image_w = stream_images.shape
        patch_size = int(self.stream_point_decoder.patch_size)
        patch_shape = (int(image_h // patch_size), int(image_w // patch_size))
        condition = self.fused_tokens_to_patch_condition(fused_tokens, patch_shape)
        fused_scale = self.stream_condition_scale
        mask_condition = self.mask_to_patch_condition(
            point_mask_condition,
            patch_shape=patch_shape,
            batch_size=fused_tokens.shape[0],
            device=fused_tokens.device,
            dtype=fused_tokens.dtype,
        )
        mask_scale = self.stream_mask_condition_scale

        conditioned = []
        for tokens in stream_tokens:
            tokens = tokens.to(device=fused_tokens.device, dtype=fused_tokens.dtype)
            patch_tokens = tokens[:, :, patch_start_idx:, :]
            if patch_tokens.shape[2] != condition.shape[1]:
                raise ValueError(
                    "Stream DPT patch token count does not match fused condition: "
                    f"{patch_tokens.shape[2]} vs {condition.shape[1]}"
            )
            updated = tokens.clone()
            residual = fused_scale * condition[:, None]
            if mask_condition is not None:
                residual = residual + mask_scale * mask_condition[:, None]
            updated[:, :, patch_start_idx:, :] = patch_tokens + residual
            conditioned.append(updated)
        return conditioned

    def fused_tokens_to_patch_condition(
        self,
        fused_tokens: torch.Tensor,
        patch_shape: tuple[int, int],
    ) -> torch.Tensor:
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
            size=patch_shape,
            mode="bilinear",
            align_corners=False,
        )
        condition = dense.flatten(2).transpose(1, 2)
        condition = self.stream_condition_proj(condition)
        return self.stream_condition_norm(condition)

    def mask_to_patch_condition(
        self,
        mask: torch.Tensor | None,
        *,
        patch_shape: tuple[int, int],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if self.point_mask_condition == "none":
            return None
        if mask is None:
            raise ValueError(
                f"point_mask_condition={self.point_mask_condition!r} requires a mask."
            )
        if mask.ndim == 2:
            mask = mask[None, None]
        elif mask.ndim == 3:
            mask = mask[:, None]
        elif mask.ndim != 4:
            raise ValueError(
                "point_mask_condition mask must have shape [H, W], [B, H, W], "
                f"or [B, 1, H, W], got {tuple(mask.shape)}"
            )
        mask = mask.to(device=device, dtype=dtype)
        if mask.shape[0] == 1 and batch_size > 1:
            mask = mask.expand(batch_size, -1, -1, -1)
        if mask.shape[0] != batch_size:
            raise ValueError(
                f"Mask batch size {mask.shape[0]} does not match {batch_size}."
            )
        mask = F.interpolate(
            mask,
            size=patch_shape,
            mode="bilinear",
            align_corners=False,
        )
        condition = mask.flatten(2).transpose(1, 2)
        assert self.stream_mask_condition_proj is not None
        assert self.stream_mask_condition_norm is not None
        condition = self.stream_mask_condition_proj(condition)
        return self.stream_mask_condition_norm(condition)

    def load_stream_point_decoder_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        *,
        strict: bool = False,
    ):
        if self.stream_point_decoder is None:
            return [], []
        return self.stream_point_decoder.load_state_dict(state_dict, strict=strict)

    def fuse_tokens(
        self,
        *,
        sam_tokens: torch.Tensor,
        geometry_tokens: torch.Tensor,
        camera_tokens: torch.Tensor | None,
    ) -> torch.Tensor:
        sam_tokens = self._ensure_batched(sam_tokens)
        geometry_tokens = self._ensure_batched(geometry_tokens)
        query = self.proj_sam(sam_tokens)
        if self.fusion_type == "camera_guided":
            assert self.camera_guided_fusion is not None
            return self.camera_guided_fusion(
                visual_tokens=query,
                spatial_tokens=geometry_tokens,
                camera_tokens=camera_tokens,
            )
        context = [self.proj_geometry(geometry_tokens)]
        if camera_tokens is not None:
            if self.proj_camera is None:
                raise ValueError("camera_tokens were provided but camera_dim is None")
            context.append(self.proj_camera(self._ensure_batched(camera_tokens)))
        context_tokens = self.context_norm(torch.cat(context, dim=1))
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

    def tokens_to_feature_grid(self, fused_tokens: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = fused_tokens.shape
        grid_h, grid_w = self.feature_grid
        expected = grid_h * grid_w
        if tokens != expected:
            raise ValueError(
                f"Expected {expected} fused tokens from feature_grid={self.feature_grid}, "
                f"got {tokens}"
            )
        return fused_tokens.transpose(1, 2).reshape(batch, channels, grid_h, grid_w)

    def decode_fused_sam_mask(
        self,
        *,
        fused_tokens: torch.Tensor,
        sam_tracker,
        mask_prompt: torch.Tensor | None,
        object_query: torch.Tensor | None = None,
        sam_features: dict[str, torch.Tensor] | None = None,
        feature_mode: str = "replace",
    ) -> torch.Tensor:
        """Run SAM3's prompt encoder + mask decoder on adapted fused tokens."""
        if not self.enable_fused_sam_decoder:
            raise RuntimeError("Fused SAM decoder is disabled for this model.")
        assert self.fused_sam_image_proj is not None
        assert self.fused_sam_high_s1 is not None
        assert self.fused_sam_high_s0 is not None
        assert self.fused_sam_object_proj is not None
        assert self.fused_sam_residual_scale is not None

        feature_mode = feature_mode.strip().lower()
        if feature_mode not in {"replace", "residual"}:
            raise ValueError(
                "feature_mode must be 'replace' or 'residual', "
                f"got {feature_mode!r}"
            )
        fused_tokens = self._ensure_batched(fused_tokens)
        feature = self.tokens_to_feature_grid(fused_tokens)
        object_query = self._prepare_object_query(
            object_query,
            batch_size=feature.shape[0],
            device=feature.device,
        )
        if object_query is not None:
            feature = feature + self.fused_sam_object_proj(object_query)[
                :, :, None, None
            ]
        residual_image = self.fused_sam_image_proj(feature)
        residual_s1 = self.fused_sam_high_s1(
            F.interpolate(feature, size=(144, 144), mode="bilinear", align_corners=False)
        )
        residual_s0 = self.fused_sam_high_s0(
            F.interpolate(feature, size=(288, 288), mode="bilinear", align_corners=False)
        )
        if feature_mode == "replace":
            image_embed = residual_image
            high_s1 = residual_s1
            high_s0 = residual_s0
        else:
            if sam_features is None:
                raise ValueError(
                    "feature_mode='residual' requires original SAM3 tracker features."
                )
            image_embed = self._sam_feature(
                sam_features,
                key="image_embed",
                reference=residual_image,
            )
            high_s1 = self._sam_feature(
                sam_features,
                key="high_s1",
                reference=residual_s1,
            )
            high_s0 = self._sam_feature(
                sam_features,
                key="high_s0",
                reference=residual_s0,
            )
            scale = self.fused_sam_residual_scale
            image_embed = image_embed + scale * residual_image
            high_s1 = high_s1 + scale * residual_s1
            high_s0 = high_s0 + scale * residual_s0

        if mask_prompt is not None:
            if mask_prompt.ndim == 2:
                mask_prompt = mask_prompt[None, None]
            elif mask_prompt.ndim == 3:
                mask_prompt = mask_prompt[:, None]
            elif mask_prompt.ndim != 4:
                raise ValueError(
                    "mask_prompt must have shape [H, W], [B, H, W], or [B, 1, H, W], "
                    f"got {tuple(mask_prompt.shape)}"
                )
            mask_prompt = mask_prompt.to(device=feature.device, dtype=feature.dtype)

        batch = image_embed.shape[0]
        device = image_embed.device
        point_coords = torch.zeros(batch, 1, 2, device=device, dtype=image_embed.dtype)
        point_labels = -torch.ones(batch, 1, device=device, dtype=torch.int32)
        if mask_prompt is not None:
            mask_size = sam_tracker.sam_prompt_encoder.mask_input_size
            if tuple(mask_prompt.shape[-2:]) != tuple(mask_size):
                mask_prompt = F.interpolate(
                    mask_prompt.float(),
                    size=mask_size,
                    mode="bilinear",
                    align_corners=False,
                ).to(dtype=image_embed.dtype)
        sparse_embeddings, dense_embeddings = sam_tracker.sam_prompt_encoder(
            points=(point_coords, point_labels),
            boxes=None,
            masks=mask_prompt,
        )
        mask_logits, _, _, _ = sam_tracker.sam_mask_decoder(
            image_embeddings=image_embed,
            image_pe=sam_tracker.sam_prompt_encoder.get_dense_pe().to(
                device=device,
                dtype=image_embed.dtype,
            ),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            repeat_image=False,
            high_res_features=[high_s0, high_s1],
        )
        if mask_logits.shape[-2:] != self.output_size:
            mask_logits = F.interpolate(
                mask_logits.float(),
                size=self.output_size,
                mode="bilinear",
                align_corners=False,
            )
        return mask_logits[:, 0].contiguous()

    def build_sam3_tracker_fpn_residuals(
        self,
        *,
        fused_tokens: torch.Tensor,
        object_query: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """Project fused tokens into SAM3 tracker FPN residuals.

        Returned levels are [fpn0, fpn1, fpn2]. Injecting these into SAM3's
        tracker backbone lets the original video memory and propagation path run
        on the 3D-aware features.
        """
        if not self.enable_fused_sam_decoder:
            raise RuntimeError("Fused SAM adapter layers are disabled for this model.")
        assert self.fused_sam_image_proj is not None
        assert self.fused_sam_high_s1 is not None
        assert self.fused_sam_high_s0 is not None
        assert self.fused_sam_object_proj is not None
        assert self.fused_sam_residual_scale is not None

        fused_tokens = self._ensure_batched(fused_tokens)
        feature = self.tokens_to_feature_grid(fused_tokens)
        object_query = self._prepare_object_query(
            object_query,
            batch_size=feature.shape[0],
            device=feature.device,
        )
        if object_query is not None:
            feature = feature + self.fused_sam_object_proj(object_query)[
                :, :, None, None
            ]

        scale = self.fused_sam_residual_scale
        residual_fpn2 = scale * self.fused_sam_image_proj(feature)
        residual_fpn1 = scale * self.fused_sam_high_s1(
            F.interpolate(
                feature,
                size=(144, 144),
                mode="bilinear",
                align_corners=False,
            )
        )
        residual_fpn0 = scale * self.fused_sam_high_s0(
            F.interpolate(
                feature,
                size=(288, 288),
                mode="bilinear",
                align_corners=False,
            )
        )
        return [residual_fpn0, residual_fpn1, residual_fpn2]

    @staticmethod
    def _sam_feature(
        sam_features: dict[str, torch.Tensor],
        *,
        key: str,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if key not in sam_features:
            raise KeyError(f"sam_features is missing {key!r}")
        value = sam_features[key]
        if value.ndim == 3:
            value = value[None]
        if value.ndim != 4:
            raise ValueError(
                f"sam_features[{key!r}] must have shape [B, C, H, W], "
                f"got {tuple(value.shape)}"
            )
        value = value.to(device=reference.device, dtype=reference.dtype)
        if value.shape[0] == 1 and reference.shape[0] > 1:
            value = value.expand(reference.shape[0], -1, -1, -1)
        if value.shape[0] != reference.shape[0]:
            raise ValueError(
                f"sam_features[{key!r}] batch {value.shape[0]} does not match "
                f"reference batch {reference.shape[0]}"
            )
        if value.shape[1] != reference.shape[1]:
            raise ValueError(
                f"sam_features[{key!r}] channels {value.shape[1]} do not match "
                f"adapter channels {reference.shape[1]}"
            )
        if value.shape[-2:] != reference.shape[-2:]:
            value = F.interpolate(
                value,
                size=reference.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return value

    def pool_object_query(
        self,
        fused_tokens: torch.Tensor,
        reference_mask: torch.Tensor,
    ) -> torch.Tensor:
        fused_tokens = self._ensure_batched(fused_tokens)
        if reference_mask.ndim == 2:
            reference_mask = reference_mask[None]
        if reference_mask.ndim != 3:
            raise ValueError(
                "reference_mask must have shape [H, W] or [B, H, W], "
                f"got {tuple(reference_mask.shape)}"
            )
        reference_mask = reference_mask.to(
            device=fused_tokens.device,
            dtype=fused_tokens.dtype,
        )
        if reference_mask.shape[0] == 1 and fused_tokens.shape[0] > 1:
            reference_mask = reference_mask.expand(fused_tokens.shape[0], -1, -1)
        mask_grid = F.interpolate(
            reference_mask[:, None],
            size=self.feature_grid,
            mode="nearest",
        )
        weights = mask_grid.flatten(2).transpose(1, 2)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (fused_tokens * weights).sum(dim=1) / denom

    def update_object_query(
        self,
        object_query: torch.Tensor,
        fused_tokens: torch.Tensor,
        update_mask: torch.Tensor,
    ) -> torch.Tensor:
        object_query = self._prepare_object_query(
            object_query,
            batch_size=self._ensure_batched(fused_tokens).shape[0],
            device=fused_tokens.device,
        )
        if update_mask.ndim == 2:
            has_update = bool(update_mask.any().detach().cpu().item())
        else:
            has_update = bool(update_mask.flatten(1).any(dim=1).all().detach().cpu().item())
        if not has_update:
            return object_query
        candidate = self.pool_object_query(fused_tokens, update_mask)
        updated = self.object_memory_update(candidate, object_query)
        return self.object_memory_norm(updated + object_query)

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

    @staticmethod
    def _prepare_object_query(
        object_query: torch.Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if object_query is None:
            return None
        if object_query.ndim == 1:
            object_query = object_query[None]
        if object_query.ndim != 2:
            raise ValueError(
                f"object_query must have shape [C] or [B, C], got {tuple(object_query.shape)}"
            )
        object_query = object_query.to(device=device)
        if object_query.shape[0] == 1 and batch_size > 1:
            object_query = object_query.expand(batch_size, -1)
        if object_query.shape[0] != batch_size:
            raise ValueError(
                f"object_query batch size {object_query.shape[0]} does not match {batch_size}"
            )
        return object_query


def resize_bhwc(values: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Resize a dense map with channels in the last dimension."""
    x = values.permute(0, 3, 1, 2)
    x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
    return x.permute(0, 2, 3, 1).contiguous()


def resize_bhw(values: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Resize a dense scalar map."""
    x = values[:, None]
    x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
    return x[:, 0].contiguous()


def ensure_batched(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.ndim == 2:
        return tokens.unsqueeze(0)
    if tokens.ndim != 3:
        raise ValueError(
            f"Expected token tensor [B, N, C] or [N, C], got {tuple(tokens.shape)}"
        )
    return tokens
