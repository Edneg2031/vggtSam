"""StreamVGGT wrapper backed by the repository's tested latent adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from PIL import Image

from .streamvggt_latent import (
    StreamVGGTLatentAdapter,
    load_streamvggt_latent_model,
)

from ..types import GeometrySequence


def materialize_streamvggt_rgb(
    image_paths: Sequence[str | Path],
    output_dir: str | Path,
    *,
    image_mode: str = "crop",
) -> tuple[tuple[Path, ...], tuple[int, int]]:
    """Write StreamVGGT's processed RGB frames for another path-based backend.

    StreamVGGT and SAM3 both consume image paths, but their model-space grids
    must refer to the same pixels before a SAM mask is fused with a pointmap.
    Reusing the upstream StreamVGGT preprocessing here makes the temporary
    files the single source of truth for both providers.  JPEG is used to keep
    compatibility with SAM3's video loader; the same files are then read by
    StreamVGGT and SAM3.
    """

    paths = tuple(Path(path).expanduser().resolve() for path in image_paths)
    if not paths:
        raise ValueError("At least one image path is required.")
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    output.chmod(0o755)

    from streamvggt.utils.load_fn import load_and_preprocess_images

    images = load_and_preprocess_images(
        [str(path) for path in paths],
        mode=str(image_mode),
    )
    if not torch.is_tensor(images) or images.ndim != 4:
        raise ValueError(
            "StreamVGGT preprocessing must return [T,3,H,W], got "
            f"{type(images)!r} with shape {getattr(images, 'shape', None)}."
        )
    if int(images.shape[0]) != len(paths) or int(images.shape[1]) != 3:
        raise ValueError(
            "StreamVGGT preprocessing changed the frame list or channel count: "
            f"images={tuple(images.shape)} paths={len(paths)}."
        )
    height, width = int(images.shape[-2]), int(images.shape[-1])
    if height < 1 or width < 1:
        raise ValueError(f"Invalid processed image size: {(height, width)}.")

    aligned_paths: list[Path] = []
    for index, frame in enumerate(images):
        array = (
            frame.detach()
            .float()
            .cpu()
            .clamp(0.0, 1.0)
            .mul(255.0)
            .round()
            .to(torch.uint8)
            .permute(1, 2, 0)
            .contiguous()
            .numpy()
        )
        path = output / f"{index:06d}.jpg"
        Image.fromarray(array, mode="RGB").save(
            path,
            format="JPEG",
            quality=95,
            subsampling=0,
        )
        path.chmod(0o644)
        aligned_paths.append(path)
    return tuple(aligned_paths), (height, width)


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

    @staticmethod
    def _geometry_from_output(output, image_paths) -> GeometrySequence:
        points = output.geometry.aux.get("pointmap_dense")
        confidence = output.geometry.aux.get("confidence_dense")
        pose_encoding = output.geometry.camera_tokens
        if points is None or confidence is None or pose_encoding is None:
            raise RuntimeError(
                "StreamVGGT did not expose pointmap, confidence, and camera outputs."
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
