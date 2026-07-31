"""Adapt StreamVGGT outputs to the lightweight V6 segmentation contract."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from .aggregation.mine_revisit_segments import mine_revisit_candidate
from .aggregation.point_map_fusion import sample_masked_observation
from .config import ExperimentConfig
from .recovery import output_mask_to_stream
from .v6_geometry_segmentation import GeometrySegmentationPrompt


@dataclass(frozen=True)
class StreamVGGTGeometryPromptBatch:
    """Image-space prompts plus backend-only diagnostics."""

    prompts: tuple[GeometrySegmentationPrompt | None, ...]
    diagnostics: tuple[dict[str, object], ...]


@torch.no_grad()
def build_streamvggt_geometry_prompts(
    *,
    recovery: ExperimentConfig,
    sequence,
    reference_mask: torch.Tensor,
    world_points: torch.Tensor,
    confidence: torch.Tensor,
    world_to_camera: torch.Tensor,
    intrinsics: torch.Tensor,
    source_sizes: Sequence[tuple[int, int]],
    processed_size: tuple[int, int],
    point_confidence_threshold: float,
) -> StreamVGGTGeometryPromptBatch:
    """Project one reference object into each frame using StreamVGGT outputs."""

    sequence_length = len(sequence.frame_indices)
    _validate_inputs(
        sequence_length=sequence_length,
        world_points=world_points,
        confidence=confidence,
        world_to_camera=world_to_camera,
        intrinsics=intrinsics,
        source_sizes=source_sizes,
    )
    reference = int(sequence.reference_frame_idx)
    reference_grid = output_mask_to_stream(
        reference_mask,
        source_size=source_sizes[reference],
        processed_size=processed_size,
        image_mode=recovery.image_mode,
    )
    reference_grid = reference_grid & (
        confidence[reference] >= float(point_confidence_threshold)
    )
    points, _ = sample_masked_observation(
        world_points[reference],
        confidence[reference],
        reference_grid,
        max_points=recovery.max_points_per_object,
    )
    reference_available = bool(points.numel())

    prompts: list[GeometrySegmentationPrompt | None] = []
    diagnostics: list[dict[str, object]] = []
    for frame, frame_index in enumerate(sequence.frame_indices):
        if frame == reference:
            prompts.append(None)
            diagnostics.append(
                _diagnostic(
                    frame=frame,
                    frame_index=int(frame_index),
                    status=(
                        "reference_mask"
                        if reference_available
                        else "reference_unavailable"
                    ),
                    accepted=reference_available,
                    projected_points=int(points.shape[0]),
                    supported_points=int(points.shape[0]),
                )
            )
            continue
        if not reference_available:
            prompts.append(None)
            diagnostics.append(
                _diagnostic(
                    frame=frame,
                    frame_index=int(frame_index),
                    status="reference_geometry_unavailable",
                )
            )
            continue

        geometry = mine_revisit_candidate(
            points,
            current_world_points=world_points[frame],
            world_to_camera=world_to_camera[frame],
            intrinsics=intrinsics[frame],
            source_size=source_sizes[frame],
            processed_size=processed_size,
            output_size=tuple(int(value) for value in recovery.output_size),
            image_mode=recovery.image_mode,
            box_quantile=recovery.box_quantile,
            box_padding_ratio=recovery.box_padding_ratio,
            min_projected_points=recovery.min_projected_points,
            min_projected_fraction=recovery.min_projected_fraction,
            min_supported_points=recovery.min_supported_points,
            min_support_ratio=recovery.min_support_ratio,
            support_abs_distance=recovery.support_abs_distance,
            support_relative_distance=recovery.support_relative_distance,
        )
        accepted = bool(geometry.accepted and geometry.supported_mask.any())
        prompts.append(_geometry_to_prompt(geometry) if accepted else None)
        diagnostics.append(
            _diagnostic(
                frame=frame,
                frame_index=int(frame_index),
                status=str(geometry.reason),
                accepted=accepted,
                projected_points=int(geometry.projected_points),
                supported_points=int(geometry.supported_points),
            )
        )

    return StreamVGGTGeometryPromptBatch(
        prompts=tuple(prompts),
        diagnostics=tuple(diagnostics),
    )


def _geometry_to_prompt(geometry) -> GeometrySegmentationPrompt:
    return GeometrySegmentationPrompt(
        box_mask=geometry.mask.detach().cpu().bool(),
        positive_mask=geometry.supported_mask.detach().cpu().bool(),
    )


def _diagnostic(
    *,
    frame: int,
    frame_index: int,
    status: str,
    accepted: bool = False,
    projected_points: int = 0,
    supported_points: int = 0,
) -> dict[str, object]:
    return {
        "sequence_index": frame,
        "frame_index": frame_index,
        "geometry_backend": "streamvggt_reference_projection",
        "geometry_accepted": int(accepted),
        "geometry_reason": status,
        "projected_points": projected_points,
        "supported_points": supported_points,
    }


def _validate_inputs(
    *,
    sequence_length: int,
    world_points: torch.Tensor,
    confidence: torch.Tensor,
    world_to_camera: torch.Tensor,
    intrinsics: torch.Tensor,
    source_sizes: Sequence[tuple[int, int]],
) -> None:
    if world_points.shape[:3] != confidence.shape:
        raise ValueError("StreamVGGT pointmap/confidence shape mismatch.")
    if world_points.shape[0] != sequence_length:
        raise ValueError("StreamVGGT pointmap/frame count mismatch.")
    if world_to_camera.shape != (sequence_length, 3, 4):
        raise ValueError("StreamVGGT world_to_camera shape mismatch.")
    if intrinsics.shape != (sequence_length, 3, 3):
        raise ValueError("StreamVGGT intrinsics shape mismatch.")
    if len(source_sizes) != sequence_length:
        raise ValueError("StreamVGGT source-size/frame count mismatch.")
