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
    pointmap_grid: Optional[torch.Tensor]
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
        dpt_layer_indices: Sequence[int] = (4, 11, 17, 23),
        image_mode: str = "crop",
    ) -> None:
        self.model = model.eval()
        self.device = device
        self.token_grid = token_grid
        self.context_grid = context_grid
        self.layer_index = int(layer_index)
        self.dpt_layer_indices = tuple(int(index) for index in dpt_layer_indices)
        self.image_mode = image_mode

    @torch.no_grad()
    def extract_from_paths(
        self,
        image_paths: Sequence[str | Path],
        *,
        return_pointmap: bool = True,
        streaming_cache: bool = False,
    ) -> StreamVGGTLatentOutput:
        from streamvggt.utils.load_fn import load_and_preprocess_images

        images = load_and_preprocess_images(
            [str(path) for path in image_paths],
            mode=self.image_mode,
        ).to(self.device)
        if streaming_cache:
            return self.extract_streaming(images, return_pointmap=return_pointmap)
        return self.extract(images, return_pointmap=return_pointmap)

    @torch.no_grad()
    def extract(
        self,
        images: torch.Tensor,
        *,
        return_pointmap: bool = True,
    ) -> StreamVGGTLatentOutput:
        if images.ndim != 4:
            raise ValueError(f"Expected images [T, 3, H, W], got {tuple(images.shape)}")
        images = images.to(self.device)
        batch_images = images.unsqueeze(0)  # [1, T, 3, H, W]

        aggregated_tokens_list, patch_start_idx = self.model.aggregator(batch_images)
        dpt_tokens = [
            aggregated_tokens_list[index].float() for index in self.dpt_layer_indices
        ]
        camera_hidden = aggregated_tokens_list[-1][:, :, 0].float()
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
        pointmap_dense = None
        confidence_grid = None
        confidence_dense = None
        depth_dense = None
        depth_confidence_dense = None
        raw_output = None
        if return_pointmap and getattr(self.model, "point_head", None) is not None:
            with torch.cuda.amp.autocast(enabled=False):
                pts3d, pts3d_conf = self.model.point_head(
                    aggregated_tokens_list,
                    images=batch_images,
                    patch_start_idx=patch_start_idx,
                )
            pointmap_dense = ensure_thwc(pts3d[0]).float()
            confidence_dense = ensure_thwc(pts3d_conf[0]).float()
            pointmap_grid = resize_dense_map(pointmap_dense, self.token_grid)
            confidence_grid = resize_dense_map(
                confidence_dense,
                self.token_grid,
            )
        if return_pointmap and getattr(self.model, "depth_head", None) is not None:
            with torch.cuda.amp.autocast(enabled=False):
                depth, depth_confidence = self.model.depth_head(
                    aggregated_tokens_list,
                    images=batch_images,
                    patch_start_idx=patch_start_idx,
                )
            depth_dense = ensure_thwc(depth[0]).float()
            depth_confidence_dense = ensure_thwc(depth_confidence[0]).float()

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
                "dpt_layer_indices": self.dpt_layer_indices,
                "dense_tokens": dense_tokens.reshape(1, -1, dense_tokens.shape[-1]),
                "stream_dpt_tokens": dpt_tokens,
                "stream_camera_hidden": camera_hidden,
                "stream_images": images,
                "patch_start_idx": patch_start_idx,
                "pointmap_dense": pointmap_dense,
                "confidence_dense": confidence_dense,
                "depth_dense": depth_dense,
                "depth_confidence_dense": depth_confidence_dense,
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

    @torch.no_grad()
    def extract_streaming(
        self,
        images: torch.Tensor,
        *,
        return_pointmap: bool = True,
    ) -> StreamVGGTLatentOutput:
        if images.ndim != 4:
            raise ValueError(f"Expected images [T, 3, H, W], got {tuple(images.shape)}")
        images = images.to(self.device)
        past_key_values = [None] * self.model.aggregator.depth
        context_chunks = []
        dense_chunks = []
        dpt_layer_chunks = [[] for _ in self.dpt_layer_indices]
        camera_hidden_chunks = []
        camera_chunks = []
        pointmap_chunks = []
        pointmap_dense_chunks = []
        confidence_chunks = []
        confidence_dense_chunks = []
        depth_dense_chunks = []
        depth_confidence_dense_chunks = []
        patch_start_idx = None
        past_key_values_camera = (
            [None] * self.model.camera_head.trunk_depth
            if getattr(self.model, "camera_head", None) is not None
            else None
        )
        patch_shape = patch_grid_from_images(
            images[:1],
            patch_size=self.model.aggregator.patch_size,
        )

        for frame_idx in range(images.shape[0]):
            frame = images[frame_idx : frame_idx + 1]
            batch_frame = frame.unsqueeze(0)  # [1, 1, 3, H, W]
            aggregator_output = self.model.aggregator(
                batch_frame,
                past_key_values=past_key_values,
                use_cache=True,
                past_frame_idx=frame_idx,
            )
            if isinstance(aggregator_output, tuple) and len(aggregator_output) == 3:
                aggregated_tokens_list, patch_start_idx, past_key_values = aggregator_output
            else:
                aggregated_tokens_list, patch_start_idx = aggregator_output

            for layer_out, layer_index in zip(dpt_layer_chunks, self.dpt_layer_indices):
                layer_out.append(aggregated_tokens_list[layer_index].float())
            # Cache the exact tensor consumed by CameraHead instead of
            # inferring it later from one of the DPT levels.
            camera_hidden_chunks.append(
                aggregated_tokens_list[-1][:, :, 0].float()
            )

            tokens = aggregated_tokens_list[self.layer_index].float()
            patch_tokens = tokens[:, :, patch_start_idx:, :]
            spatial_tokens = reshape_patch_tokens(patch_tokens, patch_shape)
            context_chunks.append(resize_token_map(spatial_tokens, self.context_grid))
            dense_chunks.append(resize_token_map(spatial_tokens, self.token_grid))

            if getattr(self.model, "camera_head", None) is not None:
                with torch.cuda.amp.autocast(enabled=False):
                    pose_enc_list, past_key_values_camera = self.model.camera_head(
                        aggregated_tokens_list,
                        past_key_values_camera=past_key_values_camera,
                        use_cache=True,
                    )
                camera_chunks.append(pose_enc_list[-1].float())

            if return_pointmap and getattr(self.model, "point_head", None) is not None:
                with torch.cuda.amp.autocast(enabled=False):
                    pts3d, pts3d_conf = self.model.point_head(
                        aggregated_tokens_list,
                        images=batch_frame,
                        patch_start_idx=patch_start_idx,
                    )
                pointmap_dense = ensure_thwc(pts3d[0]).float()
                confidence_dense = ensure_thwc(pts3d_conf[0]).float()
                pointmap_dense_chunks.append(pointmap_dense)
                confidence_dense_chunks.append(confidence_dense)
                pointmap_chunks.append(
                    resize_dense_map(pointmap_dense, self.token_grid)
                )
                confidence_chunks.append(
                    resize_dense_map(
                        confidence_dense,
                        self.token_grid,
                    )
                )

            if return_pointmap and getattr(self.model, "depth_head", None) is not None:
                with torch.cuda.amp.autocast(enabled=False):
                    depth, depth_confidence = self.model.depth_head(
                        aggregated_tokens_list,
                        images=batch_frame,
                        patch_start_idx=patch_start_idx,
                    )
                depth_dense_chunks.append(ensure_thwc(depth[0]).float())
                depth_confidence_dense_chunks.append(
                    ensure_thwc(depth_confidence[0]).float()
                )

        context_tokens = torch.cat(context_chunks, dim=1)
        dense_tokens = torch.cat(dense_chunks, dim=1)
        dpt_tokens = [torch.cat(chunks, dim=1) for chunks in dpt_layer_chunks]
        camera_hidden = torch.cat(camera_hidden_chunks, dim=1)
        camera_tokens = torch.cat(camera_chunks, dim=1) if camera_chunks else None
        pointmap_grid = torch.cat(pointmap_chunks, dim=0) if pointmap_chunks else None
        pointmap_dense = (
            torch.cat(pointmap_dense_chunks, dim=0)
            if pointmap_dense_chunks
            else None
        )
        confidence_grid = torch.cat(confidence_chunks, dim=0) if confidence_chunks else None
        confidence_dense = (
            torch.cat(confidence_dense_chunks, dim=0)
            if confidence_dense_chunks
            else None
        )
        depth_dense = torch.cat(depth_dense_chunks, dim=0) if depth_dense_chunks else None
        depth_confidence_dense = (
            torch.cat(depth_confidence_dense_chunks, dim=0)
            if depth_confidence_dense_chunks
            else None
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
                "dpt_layer_indices": self.dpt_layer_indices,
                "dense_tokens": dense_tokens.reshape(1, -1, dense_tokens.shape[-1]),
                "stream_dpt_tokens": dpt_tokens,
                "stream_camera_hidden": camera_hidden,
                "stream_images": images,
                "patch_start_idx": patch_start_idx,
                "pointmap_dense": pointmap_dense,
                "confidence_dense": confidence_dense,
                "depth_dense": depth_dense,
                "depth_confidence_dense": depth_confidence_dense,
                "streaming_cache": True,
            },
        )
        return StreamVGGTLatentOutput(
            geometry=geometry,
            pointmap_grid=pointmap_grid,
            confidence_grid=confidence_grid,
            raw_output=None,
            aux={
                "patch_start_idx": patch_start_idx,
                "patch_shape": patch_shape,
                "image_shape": tuple(int(v) for v in images.shape[-2:]),
                "streaming_cache": True,
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
