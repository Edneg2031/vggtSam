"""Dense prompt-conditioned SAM3/StreamVGGT fusion model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .tokens import GeometryTokens, SemanticTokens


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
    ) -> None:
        super().__init__()
        self.feature_grid = tuple(int(v) for v in feature_grid)
        self.output_size = tuple(int(v) for v in output_size)
        self.text_dim = int(text_dim)
        self.embedding_dim = int(embedding_dim)
        self.enable_fused_sam_decoder = bool(enable_fused_sam_decoder)
        self.point_decoder = point_decoder.strip().lower()
        self.point_mask_condition = point_mask_condition.strip().lower()
        if self.point_decoder not in {"simple", "stream_dpt"}:
            raise ValueError(
                "point_decoder must be 'simple' or 'stream_dpt', "
                f"got {point_decoder!r}"
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
        else:
            self.fused_sam_image_proj = None
            self.fused_sam_high_s1 = None
            self.fused_sam_high_s0 = None
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
    ) -> torch.Tensor:
        """Run SAM3's prompt encoder + mask decoder on adapted fused tokens."""
        if not self.enable_fused_sam_decoder:
            raise RuntimeError("Fused SAM decoder is disabled for this model.")
        assert self.fused_sam_image_proj is not None
        assert self.fused_sam_high_s1 is not None
        assert self.fused_sam_high_s0 is not None

        fused_tokens = self._ensure_batched(fused_tokens)
        feature = self.tokens_to_feature_grid(fused_tokens)
        object_query = self._prepare_object_query(
            object_query,
            batch_size=feature.shape[0],
            device=feature.device,
        )
        if object_query is not None:
            feature = feature + self.object_query_proj(object_query)[:, :, None, None]
        image_embed = self.fused_sam_image_proj(feature)
        high_s1 = self.fused_sam_high_s1(
            F.interpolate(feature, size=(144, 144), mode="bilinear", align_corners=False)
        )
        high_s0 = self.fused_sam_high_s0(
            F.interpolate(feature, size=(288, 288), mode="bilinear", align_corners=False)
        )

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
