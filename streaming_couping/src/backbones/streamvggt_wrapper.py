"""StreamVGGT wrapper backed by the repository's tested latent adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
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
        points = output.geometry.aux.get("pointmap_dense")
        confidence = output.geometry.aux.get("confidence_dense")
        pose_encoding = output.geometry.camera_tokens
        if points is None or confidence is None or pose_encoding is None:
            raise RuntimeError(
                "StreamVGGT did not expose pointmap_dense, confidence_dense, and camera_tokens."
            )

        from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri

        processed_size = tuple(int(value) for value in output.aux["image_shape"])
        world_to_camera, intrinsics = pose_encoding_to_extri_intri(
            pose_encoding.float(),
            image_size_hw=processed_size,
        )
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
