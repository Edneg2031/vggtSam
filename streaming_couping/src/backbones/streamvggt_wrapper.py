"""StreamVGGT wrapper backed by the repository's tested latent adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from PIL import Image

from vggtsam.adapters.streamvggt_latent import (
    StreamVGGTLatentAdapter,
    load_streamvggt_latent_model,
)

from ..types import GeometrySequence


class StreamVGGTWrapper:
    def __init__(
        self,
        *,
        repo_path: str | Path,
        checkpoint_path: str | Path,
        device: str,
        image_mode: str,
        streaming_cache: bool,
    ) -> None:
        self.repo_path = Path(repo_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.device = str(device)
        self.image_mode = str(image_mode)
        self.streaming_cache = bool(streaming_cache)
        self.model = None
        self.adapter = None

    def load(self) -> "StreamVGGTWrapper":
        self.model = load_streamvggt_latent_model(
            repo_path=self.repo_path,
            checkpoint_path=self.checkpoint_path,
            device=self.device,
            strict=True,
        )
        self.adapter = StreamVGGTLatentAdapter(
            self.model,
            device=self.device,
            image_mode=self.image_mode,
        )
        return self

    @torch.no_grad()
    def extract(self, image_paths: Sequence[str | Path]) -> GeometrySequence:
        if self.adapter is None:
            raise RuntimeError("Call StreamVGGTWrapper.load() before inference.")
        output = self.adapter.extract_from_paths(
            image_paths,
            return_pointmap=True,
            streaming_cache=self.streaming_cache,
        )
        return self._geometry_from_output(output, image_paths)

    @torch.no_grad()
    def extract_with_latents(
        self,
        image_paths: Sequence[str | Path],
        *,
        layer_indices: Sequence[int] = (4, 11, 17),
        context_grid: tuple[int, int] = (24, 24),
    ) -> tuple[GeometrySequence, tuple[torch.Tensor, ...]]:
        """Run StreamVGGT once and return explicit geometry plus latent maps."""

        if self.model is None:
            raise RuntimeError("Call StreamVGGTWrapper.load() before inference.")
        layer_indices = tuple(int(index) for index in layer_indices)
        if len(layer_indices) < 2:
            raise ValueError("Feature merging requires at least two geometry layers.")
        adapter = StreamVGGTLatentAdapter(
            self.model,
            device=self.device,
            context_grid=tuple(int(value) for value in context_grid),
            dpt_layer_indices=layer_indices,
            image_mode=self.image_mode,
        )
        output = adapter.extract_from_paths(
            image_paths,
            return_pointmap=True,
            streaming_cache=self.streaming_cache,
        )
        raw_levels = output.geometry.aux.get("stream_dpt_tokens")
        patch_start_idx = output.geometry.aux.get("patch_start_idx")
        patch_shape = output.aux.get("patch_shape")
        if raw_levels is None or patch_start_idx is None or patch_shape is None:
            raise RuntimeError("StreamVGGT did not expose requested aggregator layers.")
        levels = _stream_tokens_to_maps(
            raw_levels,
            patch_start_idx=int(patch_start_idx),
            patch_shape=tuple(int(value) for value in patch_shape),
            output_grid=tuple(int(value) for value in context_grid),
        )
        geometry = self._geometry_from_output(output, image_paths)
        return geometry, tuple(level.detach().float().cpu() for level in levels)

    @staticmethod
    def _geometry_from_output(output, image_paths) -> GeometrySequence:
        points = output.geometry.aux.get("pointmap_dense")
        confidence = output.geometry.aux.get("confidence_dense")
        depth = output.geometry.aux.get("depth_dense")
        depth_confidence = output.geometry.aux.get("depth_confidence_dense")
        pose_encoding = output.geometry.camera_tokens
        if (
            points is None
            or confidence is None
            or depth is None
            or depth_confidence is None
            or pose_encoding is None
        ):
            raise RuntimeError(
                "StreamVGGT did not expose pointmap, depth, confidence, and camera outputs."
            )

        from streamvggt.utils.geometry import unproject_depth_map_to_point_map
        from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri

        processed_size = tuple(int(value) for value in output.aux["image_shape"])
        world_to_camera, intrinsics = pose_encoding_to_extri_intri(
            pose_encoding.float(),
            image_size_hw=processed_size,
        )
        camera_world_points = torch.from_numpy(
            unproject_depth_map_to_point_map(
                depth.detach().float().cpu(),
                world_to_camera[0].detach().float().cpu(),
                intrinsics[0].detach().float().cpu(),
            )
        ).float()
        valid_depth = torch.isfinite(depth).all(dim=-1) & (depth[..., 0] > 0.0)
        camera_world_points[~valid_depth.cpu()] = float("nan")
        source_sizes = []
        for path in image_paths:
            with Image.open(path) as image:
                source_sizes.append((image.height, image.width))
        return GeometrySequence(
            world_points=points.detach().float().cpu(),
            confidence=_normalize_confidence(confidence.detach().float().cpu()),
            world_to_camera=world_to_camera[0].detach().float().cpu(),
            intrinsics=intrinsics[0].detach().float().cpu(),
            processed_size=(processed_size[0], processed_size[1]),
            source_sizes=tuple(source_sizes),
            depth=depth.detach().float().cpu(),
            depth_confidence=_normalize_confidence(
                depth_confidence.detach().float().cpu()
            ),
            camera_world_points=camera_world_points,
        )


def _normalize_confidence(confidence: torch.Tensor) -> torch.Tensor:
    confidence = torch.nan_to_num(confidence, nan=0.0, posinf=0.0, neginf=0.0)
    if confidence.ndim == 4 and confidence.shape[-1] == 1:
        confidence = confidence[..., 0]
    flat = confidence.flatten(1)
    low = torch.quantile(flat, 0.05, dim=1, keepdim=True)
    high = torch.quantile(flat, 0.95, dim=1, keepdim=True)
    return ((flat - low) / (high - low).clamp_min(1e-6)).clamp(0.0, 1.0).reshape_as(
        confidence
    )


def _stream_tokens_to_maps(
    layer_tokens: Sequence[torch.Tensor],
    *,
    patch_start_idx: int,
    patch_shape: tuple[int, int],
    output_grid: tuple[int, int],
) -> list[torch.Tensor]:
    """Convert StreamVGGT cached layers to frame-major feature maps."""

    patch_height, patch_width = patch_shape
    expected = patch_height * patch_width
    maps = []
    for tokens in layer_tokens:
        if tokens.ndim != 4 or tokens.shape[0] != 1:
            raise ValueError(
                "StreamVGGT layer tokens must be [1,T,N,C], got "
                f"{tuple(tokens.shape)}"
            )
        patches = tokens[0, :, int(patch_start_idx) :, :]
        if patches.shape[1] != expected:
            raise ValueError(
                f"Expected {expected} patch tokens, got {patches.shape[1]}."
            )
        feature = patches.reshape(
            patches.shape[0],
            patch_height,
            patch_width,
            patches.shape[-1],
        ).permute(0, 3, 1, 2)
        maps.append(
            F.interpolate(
                feature.float(),
                size=output_grid,
                mode="bilinear",
                align_corners=False,
            )
        )
    return maps
