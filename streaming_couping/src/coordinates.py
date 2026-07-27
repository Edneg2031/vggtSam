"""Explicit image/grid coordinate transforms used by SAM3 and StreamVGGT."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class SpatialTransform:
    """Affine pixel-center transform from an original image to a processed image."""

    source_size: tuple[int, int]
    target_size: tuple[int, int]
    scale_xy: tuple[float, float]
    offset_xy: tuple[float, float]
    description: str

    def map_xy(self, x: float, y: float) -> tuple[float, float]:
        sx, sy = self.scale_xy
        ox, oy = self.offset_xy
        return (x + 0.5) * sx - 0.5 + ox, (y + 0.5) * sy - 0.5 + oy

    def inverse_xy(self, x: float, y: float) -> tuple[float, float]:
        sx, sy = self.scale_xy
        ox, oy = self.offset_xy
        return (x - ox + 0.5) / sx - 0.5, (y - oy + 0.5) / sy - 0.5

    def to_dict(self) -> dict:
        return asdict(self)


def sam3_resize_transform(
    source_size: Sequence[int],
    *,
    resolution: int = 1008,
) -> SpatialTransform:
    height, width = _size(source_size)
    return SpatialTransform(
        source_size=(height, width),
        target_size=(int(resolution), int(resolution)),
        scale_xy=(float(resolution) / width, float(resolution) / height),
        offset_xy=(0.0, 0.0),
        description="SAM3 direct stretch resize",
    )


def output_mask_transform(
    source_size: Sequence[int],
    output_size: Sequence[int],
) -> SpatialTransform:
    height, width = _size(source_size)
    output_height, output_width = _size(output_size)
    return SpatialTransform(
        source_size=(height, width),
        target_size=(output_height, output_width),
        scale_xy=(output_width / width, output_height / height),
        offset_xy=(0.0, 0.0),
        description="nearest-neighbor GT mask resize",
    )


def streamvggt_image_transform(
    source_size: Sequence[int],
    *,
    mode: str = "crop",
    target_size: int = 518,
    patch_size: int = 14,
) -> SpatialTransform:
    """Mirror StreamVGGT's load_and_preprocess_images geometry for one image."""

    height, width = _size(source_size)
    mode = mode.strip().lower()
    if mode not in {"crop", "pad"}:
        raise ValueError(f"mode must be 'crop' or 'pad', got {mode!r}")
    if mode == "crop":
        resized_width = int(target_size)
        resized_height = round(height * (resized_width / width) / patch_size) * patch_size
    elif width >= height:
        resized_width = int(target_size)
        resized_height = round(height * (resized_width / width) / patch_size) * patch_size
    else:
        resized_height = int(target_size)
        resized_width = round(width * (resized_height / height) / patch_size) * patch_size
    if resized_height <= 0 or resized_width <= 0:
        raise ValueError("StreamVGGT preprocessing produced a non-positive size.")

    crop_top = max(0, (resized_height - target_size) // 2) if mode == "crop" else 0
    cropped_height = min(resized_height, target_size) if mode == "crop" else resized_height
    if mode == "pad":
        pad_top = max(0, (target_size - resized_height) // 2)
        pad_left = max(0, (target_size - resized_width) // 2)
        target_height = target_width = int(target_size)
    else:
        pad_top = pad_left = 0
        target_height, target_width = cropped_height, resized_width

    return SpatialTransform(
        source_size=(height, width),
        target_size=(target_height, target_width),
        scale_xy=(resized_width / width, resized_height / height),
        offset_xy=(float(pad_left), float(pad_top - crop_top)),
        description=(
            f"StreamVGGT {mode}: resize={(resized_height, resized_width)}, "
            f"crop_top={crop_top}, pad_top={pad_top}, pad_left={pad_left}"
        ),
    )


def processed_to_grid_transform(
    processed_size: Sequence[int],
    grid_size: Sequence[int],
    *,
    description: str,
) -> SpatialTransform:
    height, width = _size(processed_size)
    grid_height, grid_width = _size(grid_size)
    return SpatialTransform(
        source_size=(height, width),
        target_size=(grid_height, grid_width),
        scale_xy=(grid_width / width, grid_height / height),
        offset_xy=(0.0, 0.0),
        description=description,
    )


def resize_label_map(
    labels: np.ndarray,
    output_size: Sequence[int],
) -> np.ndarray:
    """Nearest-neighbor resize that preserves integer instance/semantic IDs."""

    if labels.ndim != 2:
        raise ValueError(f"Expected label map [H,W], got {labels.shape}")
    output_height, output_width = _size(output_size)
    image = Image.fromarray(labels.astype(np.int32), mode="I")
    resized = image.resize(
        (output_width, output_height),
        resample=Image.Resampling.NEAREST,
    )
    return np.asarray(resized, dtype=labels.dtype).copy()


def streamvggt_label_to_grid(
    labels: np.ndarray,
    grid_size: Sequence[int],
    *,
    mode: str = "crop",
    target_size: int = 518,
    patch_size: int = 14,
    padding_value: int = 0,
) -> np.ndarray:
    """Apply StreamVGGT resize/crop/pad to labels, then map them to a token grid."""

    if labels.ndim != 2:
        raise ValueError(f"Expected label map [H,W], got {labels.shape}")
    transform = streamvggt_image_transform(
        labels.shape,
        mode=mode,
        target_size=target_size,
        patch_size=patch_size,
    )
    height, width = labels.shape
    resized_width = int(round(width * transform.scale_xy[0]))
    resized_height = int(round(height * transform.scale_xy[1]))
    image = Image.fromarray(labels.astype(np.int32), mode="I").resize(
        (resized_width, resized_height),
        resample=Image.Resampling.NEAREST,
    )
    array = np.asarray(image, dtype=np.int32)
    mode = mode.strip().lower()
    if mode == "crop":
        crop_top = max(0, (resized_height - target_size) // 2)
        array = array[crop_top : crop_top + min(resized_height, target_size), :]
    else:
        pad_top = max(0, (target_size - resized_height) // 2)
        pad_bottom = max(0, target_size - resized_height - pad_top)
        pad_left = max(0, (target_size - resized_width) // 2)
        pad_right = max(0, target_size - resized_width - pad_left)
        array = np.pad(
            array,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=int(padding_value),
        )
    if tuple(array.shape) != transform.target_size:
        raise RuntimeError(
            "StreamVGGT label transform disagrees with image transform: "
            f"labels={array.shape}, expected={transform.target_size}"
        )
    return resize_label_map(array.astype(labels.dtype), grid_size)


def compose(first: SpatialTransform, second: SpatialTransform) -> SpatialTransform:
    if first.target_size != second.source_size:
        raise ValueError(
            f"Cannot compose transform sizes {first.target_size} and {second.source_size}."
        )
    sx1, sy1 = first.scale_xy
    ox1, oy1 = first.offset_xy
    sx2, sy2 = second.scale_xy
    ox2, oy2 = second.offset_xy
    return SpatialTransform(
        source_size=first.source_size,
        target_size=second.target_size,
        scale_xy=(sx1 * sx2, sy1 * sy2),
        offset_xy=(ox1 * sx2 + ox2, oy1 * sy2 + oy2),
        description=f"{first.description} -> {second.description}",
    )


def _size(value: Sequence[int]) -> tuple[int, int]:
    if len(value) != 2:
        raise ValueError(f"Expected (height, width), got {value!r}")
    height, width = int(value[0]), int(value[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"Spatial dimensions must be positive, got {(height, width)}")
    return height, width
