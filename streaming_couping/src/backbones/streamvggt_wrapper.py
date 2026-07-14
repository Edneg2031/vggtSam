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

from ..types import GeometrySequence, ReferencePointTracks


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

        from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri

        processed_size = tuple(int(value) for value in output.aux["image_shape"])
        world_to_camera, intrinsics = pose_encoding_to_extri_intri(
            pose_encoding.float(),
            image_size_hw=processed_size,
        )
        camera_world_points = unproject_depth_to_world(
            depth,
            world_to_camera[0],
            intrinsics[0],
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
            depth=depth.detach().float().cpu(),
            depth_confidence=_normalize_confidence(
                depth_confidence.detach().float().cpu()
            ),
            camera_world_points=camera_world_points,
        )

    @torch.no_grad()
    def track_reference_points(
        self,
        image_paths: Sequence[str | Path],
        *,
        reference_index: int,
        query_points: torch.Tensor,
        iterations: int = 4,
    ) -> ReferencePointTracks:
        """Track reference-frame pixels pairwise without future-frame leakage."""

        if self.model is None:
            raise RuntimeError("Call StreamVGGTWrapper.load() before inference.")
        if query_points.ndim != 2 or query_points.shape[-1] != 2:
            raise ValueError(
                f"Expected query_points [N,2] in xy order, got {tuple(query_points.shape)}"
            )
        if not 0 <= int(reference_index) < len(image_paths):
            raise ValueError(f"Invalid reference_index={reference_index}.")

        from streamvggt.utils.load_fn import load_and_preprocess_images

        images = load_and_preprocess_images(
            [str(path) for path in image_paths],
            mode=self.image_mode,
        ).to(self.device)
        query = query_points.detach().float().to(self.device)
        frame_count = images.shape[0]
        point_count = query.shape[0]
        coordinates = torch.empty(
            frame_count,
            point_count,
            2,
            dtype=torch.float32,
        )
        visibility = torch.ones(frame_count, point_count, dtype=torch.float32)
        confidence = torch.ones(frame_count, point_count, dtype=torch.float32)
        coordinates[int(reference_index)] = query.cpu()

        for frame_index in range(frame_count):
            if frame_index == int(reference_index):
                continue
            pair = torch.stack(
                [images[int(reference_index)], images[frame_index]],
                dim=0,
            ).unsqueeze(0)
            with _autocast_for(self.device):
                aggregated_tokens, patch_start_idx = self.model.aggregator(pair)
            with torch.amp.autocast(
                device_type=_device_type(self.device),
                enabled=False,
            ):
                track_list, visible, track_confidence = self.model.track_head(
                    aggregated_tokens,
                    images=pair,
                    patch_start_idx=patch_start_idx,
                    query_points=query.unsqueeze(0),
                    iters=int(iterations),
                )
            coordinates[frame_index] = track_list[-1][0, 1].float().cpu()
            visibility[frame_index] = visible[0, 1].float().cpu()
            confidence[frame_index] = track_confidence[0, 1].float().cpu()
            del pair, aggregated_tokens, track_list, visible, track_confidence
            if _device_type(self.device) == "cuda":
                torch.cuda.empty_cache()

        return ReferencePointTracks(
            query_points=query.cpu(),
            coordinates=coordinates,
            visibility=visibility,
            confidence=confidence,
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


def unproject_depth_to_world(
    depth: torch.Tensor,
    world_to_camera: torch.Tensor,
    intrinsics: torch.Tensor,
) -> torch.Tensor:
    from streamvggt.utils.geometry import unproject_depth_map_to_point_map

    points = torch.from_numpy(
        unproject_depth_map_to_point_map(
            depth.detach().float().cpu(),
            world_to_camera.detach().float().cpu(),
            intrinsics.detach().float().cpu(),
        )
    ).float()
    valid = torch.isfinite(depth).all(dim=-1) & (depth[..., 0] > 0.0)
    points[~valid.cpu()] = float("nan")
    return points


def _device_type(device: str) -> str:
    return "cuda" if str(device).startswith("cuda") else "cpu"


def _autocast_for(device: str):
    device_type = _device_type(device)
    if device_type == "cuda":
        dtype = (
            torch.bfloat16
            if torch.cuda.get_device_capability(torch.device(device))[0] >= 8
            else torch.float16
        )
        return torch.amp.autocast(device_type="cuda", dtype=dtype)
    return torch.amp.autocast(device_type="cpu", enabled=False)
