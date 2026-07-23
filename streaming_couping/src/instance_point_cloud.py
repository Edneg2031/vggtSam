"""Export colored instance point clouds from tracking masks and pointmaps."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image

from test_sam.coordinates import streamvggt_image_transform


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
