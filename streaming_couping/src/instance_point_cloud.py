"""Export colored instance point clouds from tracking masks and pointmaps."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from PIL import Image

from test_sam.coordinates import streamvggt_image_transform

from .recovery import output_mask_to_stream
from .types import GeometrySequence, TrackingSequence


def load_processed_colors(
    image_paths: Sequence[str | Path],
    *,
    processed_size: tuple[int, int],
    image_mode: str,
) -> torch.Tensor:
    """Load RGB on the same spatial grid as StreamVGGT point-head output."""

    colors = [
        _process_rgb(
            Path(image_path),
            processed_size=processed_size,
            image_mode=image_mode,
        )
        for image_path in image_paths
    ]
    return torch.from_numpy(np.stack(colors)).to(torch.uint8)


def export_instance_point_clouds(
    output_dir: str | Path,
    *,
    frame_indices: Sequence[int],
    geometry: GeometrySequence,
    colors: torch.Tensor,
    predictions: Mapping[str, TrackingSequence],
    reference_frame_idx: int,
    reference_mask: torch.Tensor,
    image_mode: str,
    confidence_threshold: float,
    max_points: int,
) -> dict[str, list[dict]]:
    """Aggregate one native-coordinate instance PLY for each tracking mode.

    The reference observation always uses the initialization GT mask. All
    later observations use the corresponding branch prediction.
    """

    confidence_threshold = float(confidence_threshold)
    max_points = int(max_points)
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("Point-cloud confidence threshold must be in [0, 1].")
    if max_points < 1:
        raise ValueError("Point-cloud max_points must be positive.")

    frame_count = len(frame_indices)
    if geometry.world_points.shape[0] != frame_count:
        raise ValueError("Geometry length does not match the selected frames.")
    if tuple(colors.shape) != (
        frame_count,
        *geometry.processed_size,
        3,
    ):
        raise ValueError(
            "Processed RGB shape does not match the StreamVGGT pointmap: "
            f"{tuple(colors.shape)} versus "
            f"{(frame_count, *geometry.processed_size, 3)}."
        )
    reference_frame_idx = int(reference_frame_idx)
    if not 0 <= reference_frame_idx < frame_count:
        raise ValueError("Reference frame is outside the selected sequence.")

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    reference_grid_mask = output_mask_to_stream(
        reference_mask,
        source_size=geometry.source_sizes[reference_frame_idx],
        processed_size=geometry.processed_size,
        image_mode=image_mode,
    )

    summary_rows = []
    frame_rows = []
    for mode, tracking in predictions.items():
        if tracking.masks.shape[0] != frame_count:
            raise ValueError(
                f"Tracking mode {mode!r} does not cover every selected frame."
            )
        grid_masks = _tracking_masks_to_stream_grid(
            tracking.masks,
            geometry=geometry,
            image_mode=image_mode,
        )
        grid_masks[reference_frame_idx] = reference_grid_mask

        point_chunks = []
        color_chunks = []
        confidence_chunks = []
        for sequence_index, frame_index in enumerate(frame_indices):
            valid = (
                grid_masks[sequence_index]
                & torch.isfinite(
                    geometry.world_points[sequence_index]
                ).all(dim=-1)
                & torch.isfinite(geometry.confidence[sequence_index])
                & (
                    geometry.confidence[sequence_index]
                    >= confidence_threshold
                )
            )
            points = geometry.world_points[sequence_index][valid]
            point_colors = colors[sequence_index][valid]
            point_confidence = geometry.confidence[sequence_index][valid]
            point_chunks.append(points)
            color_chunks.append(point_colors)
            confidence_chunks.append(point_confidence)
            frame_rows.append(
                {
                    "mode": mode,
                    "sequence_index": sequence_index,
                    "frame_index": int(frame_index),
                    "is_reference": int(
                        sequence_index == reference_frame_idx
                    ),
                    "mask_source": (
                        "reference_gt"
                        if sequence_index == reference_frame_idx
                        else "tracking_prediction"
                    ),
                    "mask_pixels_on_geometry_grid": int(
                        grid_masks[sequence_index].sum()
                    ),
                    "selected_points": int(points.shape[0]),
                    "mean_selected_confidence": (
                        float(point_confidence.mean())
                        if point_confidence.numel()
                        else float("nan")
                    ),
                }
            )

        points = _concatenate_points(point_chunks)
        point_colors = _concatenate_colors(color_chunks)
        point_confidence = _concatenate_confidence(confidence_chunks)
        selected_points = int(points.shape[0])
        points, point_colors = _limit_points(
            points,
            point_colors,
            max_points=max_points,
        )
        path = root / f"{mode}.ply"
        _write_binary_ply(path, points, point_colors)
        summary_rows.append(
            {
                "mode": mode,
                "coordinate_system": "streamvggt_point_head_native",
                "confidence_threshold": confidence_threshold,
                "selected_points_before_limit": selected_points,
                "exported_points": int(points.shape[0]),
                "observation_frames": sum(
                    int(chunk.shape[0] > 0) for chunk in point_chunks
                ),
                "mean_selected_confidence": (
                    float(point_confidence.mean())
                    if point_confidence.numel()
                    else float("nan")
                ),
                "ply_path": str(path),
            }
        )
    return {
        "summary_rows": summary_rows,
        "frame_rows": frame_rows,
    }


def _tracking_masks_to_stream_grid(
    masks: torch.Tensor,
    *,
    geometry: GeometrySequence,
    image_mode: str,
) -> torch.Tensor:
    return torch.stack(
        [
            output_mask_to_stream(
                mask,
                source_size=geometry.source_sizes[index],
                processed_size=geometry.processed_size,
                image_mode=image_mode,
            )
            for index, mask in enumerate(masks)
        ]
    ).bool()


def _process_rgb(
    image_path: Path,
    *,
    processed_size: tuple[int, int],
    image_mode: str,
) -> np.ndarray:
    mode = image_mode.strip().lower()
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        transform = streamvggt_image_transform(
            (image.height, image.width),
            mode=mode,
        )
        if tuple(transform.target_size) != tuple(processed_size):
            raise ValueError(
                "StreamVGGT RGB transform does not match the pointmap grid: "
                f"{transform.target_size} versus {processed_size}."
            )
        resized_width = int(round(image.width * transform.scale_xy[0]))
        resized_height = int(round(image.height * transform.scale_xy[1]))
        image = image.resize(
            (resized_width, resized_height),
            resample=Image.Resampling.BICUBIC,
        )
        if mode == "crop":
            crop_top = max(0, (resized_height - 518) // 2)
            image = image.crop(
                (
                    0,
                    crop_top,
                    resized_width,
                    crop_top + min(518, resized_height),
                )
            )
        else:
            canvas = Image.new("RGB", (518, 518), "white")
            canvas.paste(
                image,
                ((518 - resized_width) // 2, (518 - resized_height) // 2),
            )
            image = canvas
        return np.asarray(image, dtype=np.uint8).copy()


def _concatenate_points(chunks: list[torch.Tensor]) -> torch.Tensor:
    nonempty = [chunk.detach().float().cpu() for chunk in chunks if chunk.numel()]
    return (
        torch.cat(nonempty, dim=0)
        if nonempty
        else torch.empty((0, 3), dtype=torch.float32)
    )


def _concatenate_colors(chunks: list[torch.Tensor]) -> torch.Tensor:
    nonempty = [
        chunk.detach().to(torch.uint8).cpu()
        for chunk in chunks
        if chunk.numel()
    ]
    return (
        torch.cat(nonempty, dim=0)
        if nonempty
        else torch.empty((0, 3), dtype=torch.uint8)
    )


def _concatenate_confidence(chunks: list[torch.Tensor]) -> torch.Tensor:
    nonempty = [chunk.detach().float().cpu() for chunk in chunks if chunk.numel()]
    return (
        torch.cat(nonempty, dim=0)
        if nonempty
        else torch.empty((0,), dtype=torch.float32)
    )


def _limit_points(
    points: torch.Tensor,
    colors: torch.Tensor,
    *,
    max_points: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if points.shape[0] <= int(max_points):
        return points, colors
    indices = torch.linspace(
        0,
        points.shape[0] - 1,
        steps=int(max_points),
    ).long()
    return points.index_select(0, indices), colors.index_select(0, indices)


def _write_binary_ply(
    path: Path,
    points: torch.Tensor,
    colors: torch.Tensor,
) -> None:
    point_array = points.detach().float().cpu().numpy().astype(np.float32)
    color_array = colors.detach().cpu().numpy().astype(np.uint8)
    if (
        point_array.shape != color_array.shape
        or point_array.ndim != 2
        or point_array.shape[1:] != (3,)
    ):
        raise ValueError(
            "PLY points and colors must both have shape [N, 3], got "
            f"{point_array.shape} and {color_array.shape}."
        )
    vertices = np.empty(
        point_array.shape[0],
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    if len(vertices):
        vertices["x"], vertices["y"], vertices["z"] = point_array.T
        vertices["red"], vertices["green"], vertices["blue"] = (
            color_array.T
        )
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        vertices.tofile(handle)
