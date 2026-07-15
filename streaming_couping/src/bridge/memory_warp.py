"""Geometry-aware spatial position encoding for SAM3 video memory.

This module changes only the positional encoding attached to historical SAM3
memory features. The memory features, object pointers, decoder, and persistent
object ID remain owned by the original SAM3 tracker.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from types import MethodType
from typing import Iterator, Sequence

import torch
import torch.nn.functional as F

from test_sam.coordinates import sam3_resize_transform, streamvggt_image_transform

from ..types import GeometrySequence


@dataclass(frozen=True)
class MemoryWarpObservation:
    current_frame: int
    memory_frame: int
    geometry_current_frame: int
    geometry_memory_frame: int
    total_tokens: int
    valid_tokens: int
    valid_ratio: float
    mean_displacement_pixels: float


class GeometryMemoryPositionWarper:
    """Reproject historical memory positions into the current SAM3 view."""

    MODES = {"identity", "aligned", "shuffled"}

    def __init__(
        self,
        geometry: GeometrySequence,
        *,
        mode: str,
        image_mode: str,
        frame_permutation: Sequence[int] | None = None,
        min_geometry_confidence: float = 0.0,
        point_source: str = "depth_camera",
        sam_resolution: int = 1008,
    ) -> None:
        mode = str(mode).strip().lower()
        if mode not in self.MODES:
            raise ValueError(f"Unsupported memory warp mode {mode!r}.")
        point_source = str(point_source).strip().lower()
        if point_source not in {"depth_camera", "point_head"}:
            raise ValueError(f"Unsupported geometry point source {point_source!r}.")
        if point_source == "depth_camera" and geometry.camera_world_points is None:
            raise ValueError(
                "depth_camera memory warping requires StreamVGGT depth+camera points."
            )
        num_frames = _num_geometry_frames(
            geometry.camera_world_points
            if point_source == "depth_camera"
            else geometry.world_points
        )
        if frame_permutation is None:
            frame_permutation = tuple(range(num_frames))
        permutation = tuple(int(index) for index in frame_permutation)
        if sorted(permutation) != list(range(num_frames)):
            raise ValueError(
                "frame_permutation must contain every geometry frame exactly once."
            )
        self.geometry = geometry
        self.mode = mode
        self.image_mode = str(image_mode)
        self.frame_permutation = permutation
        self.min_geometry_confidence = float(min_geometry_confidence)
        self.point_source = point_source
        self.sam_resolution = int(sam_resolution)
        self.observations: list[MemoryWarpObservation] = []
        self.hook_calls = 0

    def warp(
        self,
        position_encoding: torch.Tensor,
        *,
        memory_frame: int,
        current_frame: int,
    ) -> torch.Tensor:
        """Return a position map indexed by historical memory-token location."""

        self.hook_calls += 1
        if position_encoding.ndim != 4:
            raise ValueError(
                "SAM3 memory position encoding must be [B,C,H,W], got "
                f"{tuple(position_encoding.shape)}."
            )
        batch, _, height, width = position_encoding.shape
        if self.mode == "identity":
            self.observations.append(
                MemoryWarpObservation(
                    current_frame=int(current_frame),
                    memory_frame=int(memory_frame),
                    geometry_current_frame=int(current_frame),
                    geometry_memory_frame=int(memory_frame),
                    total_tokens=height * width,
                    valid_tokens=height * width,
                    valid_ratio=1.0,
                    mean_displacement_pixels=0.0,
                )
            )
            return position_encoding

        geometry_memory = self._geometry_index(memory_frame)
        geometry_current = self._geometry_index(current_frame)
        sample_grid, valid, displacement = self._build_grid(
            memory_frame=int(memory_frame),
            current_frame=int(current_frame),
            geometry_memory=geometry_memory,
            geometry_current=geometry_current,
            memory_size=(height, width),
        )
        work = position_encoding.detach().float()
        grid = sample_grid.to(device=work.device, dtype=work.dtype)
        if batch != 1:
            grid = grid.expand(batch, -1, -1, -1)
        sampled = F.grid_sample(
            work,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        valid_map = valid.to(device=work.device)[None, None]
        if batch != 1:
            valid_map = valid_map.expand(batch, -1, -1, -1)
        warped = torch.where(valid_map, sampled, work)
        warped = warped.to(
            device=position_encoding.device,
            dtype=position_encoding.dtype,
        )

        valid_tokens = int(valid.sum().item())
        total_tokens = int(valid.numel())
        self.observations.append(
            MemoryWarpObservation(
                current_frame=int(current_frame),
                memory_frame=int(memory_frame),
                geometry_current_frame=geometry_current,
                geometry_memory_frame=geometry_memory,
                total_tokens=total_tokens,
                valid_tokens=valid_tokens,
                valid_ratio=valid_tokens / max(total_tokens, 1),
                mean_displacement_pixels=(
                    float(displacement[valid].mean().item()) if valid_tokens else 0.0
                ),
            )
        )
        return warped

    def summary(self) -> dict[str, float | int | str | list[int]]:
        total = sum(item.total_tokens for item in self.observations)
        valid = sum(item.valid_tokens for item in self.observations)
        weighted_displacement = sum(
            item.mean_displacement_pixels * item.valid_tokens
            for item in self.observations
        )
        return {
            "mode": self.mode,
            "point_source": self.point_source,
            "hook_calls": int(self.hook_calls),
            "memory_pairs": len(self.observations),
            "warped_tokens": int(total),
            "valid_warped_tokens": int(valid),
            "valid_warp_ratio": float(valid / max(total, 1)),
            "mean_warp_displacement_pixels": float(
                weighted_displacement / max(valid, 1)
            ),
            "frame_permutation": list(self.frame_permutation),
        }

    def observation_rows(self) -> list[dict[str, float | int]]:
        return [asdict(item) for item in self.observations]

    def project_reference_mask(
        self,
        source_mask: torch.Tensor,
        *,
        memory_frame: int,
        current_frame: int,
        output_size: tuple[int, int],
        memory_size: tuple[int, int] = (72, 72),
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int]]:
        """Project reference object tokens for metrics-only geometry diagnosis."""

        geometry_memory = self._geometry_index(memory_frame)
        geometry_current = self._geometry_index(current_frame)
        sample_grid, valid, displacement = self._build_grid(
            memory_frame=int(memory_frame),
            current_frame=int(current_frame),
            geometry_memory=geometry_memory,
            geometry_current=geometry_current,
            memory_size=memory_size,
        )
        source_tokens = F.interpolate(
            source_mask.detach().float()[None, None],
            size=memory_size,
            mode="nearest",
        )[0, 0].bool()
        selected = source_tokens & valid
        output_height, output_width = (int(value) for value in output_size)
        point_mask = torch.zeros(output_height, output_width, dtype=torch.bool)
        normalized_x = sample_grid[0, ..., 0]
        normalized_y = sample_grid[0, ..., 1]
        output_x = ((normalized_x + 1.0) * output_width / 2.0 - 0.5).round().long()
        output_y = ((normalized_y + 1.0) * output_height / 2.0 - 0.5).round().long()
        selected &= (
            (output_x >= 0)
            & (output_x < output_width)
            & (output_y >= 0)
            & (output_y < output_height)
        )
        if selected.any():
            point_mask[output_y[selected], output_x[selected]] = True
        token_height, token_width = memory_size
        radius_y = max(1, int(round(output_height / token_height / 2.0)))
        radius_x = max(1, int(round(output_width / token_width / 2.0)))
        projected_mask = F.max_pool2d(
            point_mask.float()[None, None],
            kernel_size=(2 * radius_y + 1, 2 * radius_x + 1),
            stride=1,
            padding=(radius_y, radius_x),
        )[0, 0].bool()
        selected_count = int(source_tokens.sum().item())
        projected_count = int(selected.sum().item())
        stats = {
            "source_object_tokens": selected_count,
            "valid_projected_object_tokens": projected_count,
            "object_token_valid_ratio": float(
                projected_count / max(selected_count, 1)
            ),
            "unique_projected_pixels": int(point_mask.sum().item()),
            "mean_object_displacement_pixels": (
                float(displacement[selected].mean().item())
                if projected_count
                else 0.0
            ),
        }
        return projected_mask, point_mask, stats

    def _geometry_index(self, sequence_index: int) -> int:
        sequence_index = int(sequence_index)
        if sequence_index < 0 or sequence_index >= len(self.frame_permutation):
            raise IndexError(
                f"SAM3 frame {sequence_index} is outside the geometry sequence."
            )
        if self.mode == "shuffled":
            return self.frame_permutation[sequence_index]
        return sequence_index

    def _build_grid(
        self,
        *,
        memory_frame: int,
        current_frame: int,
        geometry_memory: int,
        geometry_current: int,
        memory_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        height, width = memory_size
        if self.point_source == "depth_camera":
            point_sequence = self.geometry.camera_world_points
            confidence_sequence = self.geometry.depth_confidence
            if point_sequence is None or confidence_sequence is None:
                raise RuntimeError("StreamVGGT depth-camera geometry is unavailable.")
        else:
            point_sequence = self.geometry.world_points
            confidence_sequence = self.geometry.confidence
        points = _frame(point_sequence, geometry_memory).float().cpu()
        confidence = _frame(confidence_sequence, geometry_memory).float().cpu()
        if confidence.ndim == 3 and confidence.shape[-1] == 1:
            confidence = confidence[..., 0]
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError(
                "StreamVGGT world pointmap must be [H,W,3], got "
                f"{tuple(points.shape)}."
            )
        if confidence.shape != points.shape[:2]:
            raise ValueError(
                "StreamVGGT confidence and pointmap disagree: "
                f"{tuple(confidence.shape)} vs {tuple(points.shape[:2])}."
            )

        memory_source_size = self.geometry.source_sizes[geometry_memory]
        current_geometry_source_size = self.geometry.source_sizes[geometry_current]
        current_sam_source_size = self.geometry.source_sizes[current_frame]
        sam_memory_transform = sam3_resize_transform(
            memory_source_size,
            resolution=self.sam_resolution,
        )
        stream_memory_transform = streamvggt_image_transform(
            memory_source_size,
            mode=self.image_mode,
        )
        stream_current_transform = streamvggt_image_transform(
            current_geometry_source_size,
            mode=self.image_mode,
        )
        sam_current_transform = sam3_resize_transform(
            current_sam_source_size,
            resolution=self.sam_resolution,
        )

        yy, xx = torch.meshgrid(
            torch.arange(height, dtype=torch.float32),
            torch.arange(width, dtype=torch.float32),
            indexing="ij",
        )
        sam_x = (xx + 0.5) * self.sam_resolution / width - 0.5
        sam_y = (yy + 0.5) * self.sam_resolution / height - 0.5
        original_x, original_y = _inverse_map(
            sam_x,
            sam_y,
            sam_memory_transform.scale_xy,
            sam_memory_transform.offset_xy,
        )
        processed_x, processed_y = _map(
            original_x,
            original_y,
            stream_memory_transform.scale_xy,
            stream_memory_transform.offset_xy,
        )
        point_height, point_width = points.shape[:2]
        processed_height, processed_width = self.geometry.processed_size
        point_x = (processed_x + 0.5) * point_width / processed_width - 0.5
        point_y = (processed_y + 0.5) * point_height / processed_height - 0.5
        source_valid = (
            (point_x >= -0.5)
            & (point_x < point_width - 0.5)
            & (point_y >= -0.5)
            & (point_y < point_height - 0.5)
        )
        point_ix = point_x.round().long().clamp(0, point_width - 1)
        point_iy = point_y.round().long().clamp(0, point_height - 1)
        world = points[point_iy, point_ix]
        sampled_confidence = confidence[point_iy, point_ix]
        source_valid &= torch.isfinite(world).all(dim=-1)
        source_valid &= sampled_confidence >= self.min_geometry_confidence

        extrinsic = _frame(self.geometry.world_to_camera, geometry_current).float().cpu()
        intrinsic = _frame(self.geometry.intrinsics, geometry_current).float().cpu()
        if extrinsic.shape == (4, 4):
            extrinsic = extrinsic[:3]
        if extrinsic.shape != (3, 4) or intrinsic.shape != (3, 3):
            raise ValueError(
                "Expected StreamVGGT extrinsic [3,4] and intrinsic [3,3], got "
                f"{tuple(extrinsic.shape)} and {tuple(intrinsic.shape)}."
            )
        camera = world @ extrinsic[:, :3].T + extrinsic[:, 3]
        depth = camera[..., 2]
        projected = camera @ intrinsic.T
        current_processed_x = projected[..., 0] / depth.clamp_min(1e-8)
        current_processed_y = projected[..., 1] / depth.clamp_min(1e-8)
        current_original_x, current_original_y = _inverse_map(
            current_processed_x,
            current_processed_y,
            stream_current_transform.scale_xy,
            stream_current_transform.offset_xy,
        )

        # A shuffled geometry frame can have a different source resolution. Move
        # through normalized original-image coordinates before entering SAM3.
        geometry_height, geometry_width = current_geometry_source_size
        sam_height, sam_width = current_sam_source_size
        normalized_x = (current_original_x + 0.5) / geometry_width
        normalized_y = (current_original_y + 0.5) / geometry_height
        current_sam_original_x = normalized_x * sam_width - 0.5
        current_sam_original_y = normalized_y * sam_height - 0.5
        current_sam_x, current_sam_y = _map(
            current_sam_original_x,
            current_sam_original_y,
            sam_current_transform.scale_xy,
            sam_current_transform.offset_xy,
        )
        current_grid_x = (current_sam_x + 0.5) * width / self.sam_resolution - 0.5
        current_grid_y = (current_sam_y + 0.5) * height / self.sam_resolution - 0.5
        valid = source_valid & torch.isfinite(camera).all(dim=-1) & (depth > 1e-6)
        valid &= (
            (current_grid_x >= -0.5)
            & (current_grid_x < width - 0.5)
            & (current_grid_y >= -0.5)
            & (current_grid_y < height - 0.5)
        )
        normalized_grid_x = 2.0 * (current_grid_x + 0.5) / width - 1.0
        normalized_grid_y = 2.0 * (current_grid_y + 0.5) / height - 1.0
        sample_grid = torch.stack((normalized_grid_x, normalized_grid_y), dim=-1)[None]
        displacement = torch.sqrt(
            (current_grid_x - xx).square() + (current_grid_y - yy).square()
        )
        sample_grid = torch.nan_to_num(sample_grid, nan=2.0, posinf=2.0, neginf=-2.0)
        return sample_grid, valid, displacement


@contextmanager
def install_memory_position_warp(
    tracker,
    warper: GeometryMemoryPositionWarper,
) -> Iterator[None]:
    """Temporarily replace stored SAM3 memory position encodings at read time."""

    original_method = tracker._prepare_memory_conditioned_features

    def wrapped(
        tracker_self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        output_dict,
        num_frames,
        track_in_reverse=False,
        use_prev_mem_frame=True,
    ):
        touched: list[tuple[dict, object]] = []
        seen: set[int] = set()
        if not is_init_cond_frame and use_prev_mem_frame:
            for store_name in ("cond_frame_outputs", "non_cond_frame_outputs"):
                for memory_frame, previous in output_dict.get(store_name, {}).items():
                    if previous is None or id(previous) in seen:
                        continue
                    position_encodings = previous.get("maskmem_pos_enc")
                    if not position_encodings:
                        continue
                    seen.add(id(previous))
                    replacement = list(position_encodings)
                    replacement[-1] = warper.warp(
                        position_encodings[-1],
                        memory_frame=int(memory_frame),
                        current_frame=int(frame_idx),
                    )
                    touched.append((previous, position_encodings))
                    previous["maskmem_pos_enc"] = replacement
        try:
            return original_method(
                frame_idx,
                is_init_cond_frame,
                current_vision_feats,
                current_vision_pos_embeds,
                feat_sizes,
                output_dict,
                num_frames,
                track_in_reverse=track_in_reverse,
                use_prev_mem_frame=use_prev_mem_frame,
            )
        finally:
            for previous, original_position_encodings in touched:
                previous["maskmem_pos_enc"] = original_position_encodings

    tracker._prepare_memory_conditioned_features = MethodType(wrapped, tracker)
    try:
        yield
    finally:
        tracker._prepare_memory_conditioned_features = original_method


def _num_geometry_frames(value: torch.Tensor) -> int:
    if value.ndim != 4:
        raise ValueError(
            f"Geometry world_points must be [T,H,W,3], got {tuple(value.shape)}."
        )
    return int(value.shape[0])


def _frame(value: torch.Tensor, index: int) -> torch.Tensor:
    return value[int(index)]


def _map(
    x: torch.Tensor,
    y: torch.Tensor,
    scale_xy: tuple[float, float],
    offset_xy: tuple[float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    sx, sy = scale_xy
    ox, oy = offset_xy
    return (x + 0.5) * sx - 0.5 + ox, (y + 0.5) * sy - 0.5 + oy


def _inverse_map(
    x: torch.Tensor,
    y: torch.Tensor,
    scale_xy: tuple[float, float],
    offset_xy: tuple[float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    sx, sy = scale_xy
    ox, oy = offset_xy
    return (x - ox + 0.5) / sx - 0.5, (y - oy + 0.5) / sy - 0.5
