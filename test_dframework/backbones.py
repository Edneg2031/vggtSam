"""Frozen SAM3 and StreamVGGT extraction for the explicit bridge."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from test_sam.data import MaskTrackingSequence
from vggtsam.adapters.sam3_video import (
    SAM3TrackOutput,
    SAM3VideoTrackerAdapter,
    load_sam3_video_predictor,
)
from vggtsam.adapters.streamvggt_latent import (
    StreamVGGTLatentAdapter,
    load_streamvggt_latent_model,
)

from .config import ExperimentConfig
from .geometry import normalize_confidence
from .types import GeometrySequence


def run_frozen_sam3(
    config: ExperimentConfig,
    sequence: MaskTrackingSequence,
    target_masks: torch.Tensor,
) -> SAM3TrackOutput:
    predictor = load_sam3_video_predictor(
        repo_path=config.sam3_repo,
        checkpoint_path=config.sam3_checkpoint,
        device=config.sam3_device,
        quiet=True,
    )
    tracker = SAM3VideoTrackerAdapter(
        predictor,
        output_prob_thresh=config.sam3_output_threshold,
        prompt_with_box=config.prompt_with_box,
    )
    return tracker.track_from_paths(
        sequence.image_paths,
        prompt=sequence.label,
        output_size=config.output_size,
        prompt_frame_idx=sequence.reference_frame_idx,
        reference_mask=target_masks[sequence.reference_frame_idx],
        quiet=True,
    )


def run_frozen_streamvggt(
    config: ExperimentConfig,
    sequence: MaskTrackingSequence,
) -> GeometrySequence:
    model = load_streamvggt_latent_model(
        repo_path=config.streamvggt_repo,
        checkpoint_path=config.streamvggt_checkpoint,
        device=config.geometry_device,
        strict=True,
    )
    adapter = StreamVGGTLatentAdapter(
        model,
        device=config.geometry_device,
        image_mode=config.image_mode,
    )
    output = adapter.extract_from_paths(
        sequence.image_paths,
        return_pointmap=True,
        streaming_cache=config.streaming_cache,
    )
    points = output.geometry.aux.get("pointmap_dense")
    confidence = output.geometry.aux.get("confidence_dense")
    pose_encoding = output.geometry.camera_tokens
    if points is None or confidence is None or pose_encoding is None:
        raise RuntimeError("StreamVGGT must expose pointmap, confidence, and camera outputs.")
    from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri

    processed_size = tuple(int(value) for value in output.aux["image_shape"])
    world_to_camera, intrinsics = pose_encoding_to_extri_intri(
        pose_encoding.float(),
        image_size_hw=processed_size,
    )
    source_sizes = []
    for path in sequence.image_paths:
        with Image.open(path) as image:
            source_sizes.append((image.height, image.width))
    return GeometrySequence(
        world_points=points.detach().float().cpu(),
        confidence=normalize_confidence(confidence.detach().float().cpu()),
        world_to_camera=world_to_camera[0].detach().float().cpu(),
        intrinsics=intrinsics[0].detach().float().cpu(),
        processed_size=processed_size,
        source_sizes=source_sizes,
    )

