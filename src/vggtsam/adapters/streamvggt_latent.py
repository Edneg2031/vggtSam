"""StreamVGGT latent geometry adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from vggtsam.adapters.vggt import load_streamvggt_model
from vggtsam.models.tokens import GeometryTokens


@dataclass
class StreamVGGTLatentOutput:
    geometry: GeometryTokens
    pointmap_grid: torch.Tensor
    confidence_grid: Optional[torch.Tensor] = None
    raw_output: Any = None
    aux: Dict[str, Any] = field(default_factory=dict)


def load_streamvggt_latent_model(
    *,
    repo_path: Optional[str | Path],
    checkpoint_path: str | Path,
    device: str,
    strict: bool = True,
):
    return load_streamvggt_model(
        repo_path=repo_path,
        checkpoint_path=checkpoint_path,
        device=device,
        strict=strict,
    )


class StreamVGGTLatentAdapter:
    """Expose StreamVGGT aggregator patch tokens for fusion."""

    def __init__(
        self,
        model,
        *,
        device: str,
        token_grid: Tuple[int, int] = (72, 72),
        context_grid: Tuple[int, int] = (24, 24),
        layer_index: int = -1,
        image_mode: str = "crop",
    ) -> None:
        self.model = model.eval()
        self.device = device
        self.token_grid = token_grid
        self.context_grid = context_grid
        self.layer_index = int(layer_index)
        self.image_mode = image_mode

    @torch.no_grad()
    def extract_from_paths(
        self,
        image_paths: Sequence[str | Path],
    ) -> StreamVGGTLatentOutput:
        from streamvggt.utils.load_fn import load_and_preprocess_images

        images = load_and_preprocess_images(
            [str(path) for path in image_paths],
            mode=self.image_mode,
        ).to(self.device)
        return self.extract(images)

    @torch.no_grad()
    def extract(self, images: torch.Tensor) -> StreamVGGTLatentOutput:
        if images.ndim != 4:
            raise ValueError(f"Expected images [T, 3, H, W], got {tuple(images.shape)}")
        images = images.to(self.device)
        batch_images = images.unsqueeze(0)  # [1, T, 3, H, W]

        aggregated_tokens_list, patch_start_idx = self.model.aggregator(batch_images)
        tokens = aggregated_tokens_list[self.layer_index].float()
        patch_tokens = tokens[:, :, patch_start_idx:, :]
        patch_shape = patch_grid_from_images(images, patch_size=self.model.aggregator.patch_size)

        spatial_tokens = reshape_patch_tokens(patch_tokens, patch_shape)
        context_tokens = resize_token_map(spatial_tokens, self.context_grid)
        dense_tokens = resize_token_map(spatial_tokens, self.token_grid)

        camera_tokens = None
        if getattr(self.model, "camera_head", None) is not None:
            with torch.cuda.amp.autocast(enabled=False):
                pose_enc_list = self.model.camera_head(aggregated_tokens_list)
            camera_tokens = pose_enc_list[-1].float()

        pointmap_grid = None
        confidence_grid = None
        raw_output = None
        if getattr(self.model, "point_head", None) is not None:
            with torch.cuda.amp.autocast(enabled=False):
                pts3d, pts3d_conf = self.model.point_head(
                    aggregated_tokens_list,
                    images=batch_images,
                    patch_start_idx=patch_start_idx,
                )
            pointmap_grid = resize_dense_map(ensure_thwc(pts3d[0]).float(), self.token_grid)
            confidence_grid = resize_dense_map(
                ensure_thwc(pts3d_conf[0]).float(),
                self.token_grid,
            )

        geometry = GeometryTokens(
            tokens=context_tokens.reshape(1, -1, context_tokens.shape[-1]),
            camera_tokens=camera_tokens,
            pointmap=(
                pointmap_grid.reshape(1, -1, pointmap_grid.shape[-1])
                if pointmap_grid is not None
                else None
            ),
            spatial_shape=self.context_grid,
            aux={
                "token_grid": self.token_grid,
                "context_grid": self.context_grid,
                "patch_shape": patch_shape,
                "layer_index": self.layer_index,
                "dense_tokens": dense_tokens.reshape(1, -1, dense_tokens.shape[-1]),
            },
        )
        return StreamVGGTLatentOutput(
            geometry=geometry,
            pointmap_grid=pointmap_grid,
            confidence_grid=confidence_grid,
            raw_output=raw_output,
            aux={
                "patch_start_idx": patch_start_idx,
                "patch_shape": patch_shape,
                "image_shape": tuple(int(v) for v in images.shape[-2:]),
            },
        )


def patch_grid_from_images(images: torch.Tensor, *, patch_size: int) -> Tuple[int, int]:
    height, width = images.shape[-2:]
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(
            f"StreamVGGT image shape {(height, width)} is not divisible by patch_size={patch_size}"
        )
    return int(height // patch_size), int(width // patch_size)


def reshape_patch_tokens(
    patch_tokens: torch.Tensor,
    patch_shape: Tuple[int, int],
) -> torch.Tensor:
    batch, frames, num_patches, channels = patch_tokens.shape
    grid_h, grid_w = patch_shape
    expected = grid_h * grid_w
    if num_patches != expected:
        raise ValueError(
            f"Expected {expected} patch tokens from patch_shape={patch_shape}, got {num_patches}"
        )
    return patch_tokens.reshape(batch, frames, grid_h, grid_w, channels)


def resize_token_map(
    token_map: torch.Tensor,
    size: Tuple[int, int],
) -> torch.Tensor:
    batch, frames, height, width, channels = token_map.shape
    x = token_map.reshape(batch * frames, height, width, channels).permute(0, 3, 1, 2)
    x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
    x = x.permute(0, 2, 3, 1).reshape(batch, frames, size[0], size[1], channels)
    return x


def resize_dense_map(
    dense_map: torch.Tensor,
    size: Tuple[int, int],
) -> torch.Tensor:
    # dense_map: [T, H, W, C]
    frames, height, width, channels = dense_map.shape
    x = dense_map.permute(0, 3, 1, 2)
    x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
    return x.permute(0, 2, 3, 1).reshape(frames, size[0], size[1], channels)


def ensure_thwc(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 3:
        return tensor.unsqueeze(-1)
    if tensor.ndim == 4:
        return tensor
    raise ValueError(f"Expected dense map [T, H, W] or [T, H, W, C], got {tuple(tensor.shape)}")
