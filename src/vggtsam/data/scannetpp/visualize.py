"""Visualization helpers for label masks."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


def colorize_ids(ids: np.ndarray, ignore_values: Iterable[int]) -> np.ndarray:
    ignore = np.zeros(ids.shape, dtype=bool)
    for value in ignore_values:
        ignore |= ids == int(value)

    out = np.zeros((*ids.shape, 3), dtype=np.uint8)
    unique_ids = np.unique(ids[~ignore])
    for value in unique_ids:
        out[ids == value] = _id_to_color(int(value))
    return out


def overlay_labels(
    image_rgb: np.ndarray, labels: np.ndarray, color: np.ndarray, alpha: float
) -> np.ndarray:
    valid = color.sum(axis=2) > 0
    out = image_rgb.copy()
    blended = (
        (1.0 - alpha) * out[valid].astype(np.float32)
        + alpha * color[valid].astype(np.float32)
    )
    out[valid] = np.clip(blended, 0, 255).astype(np.uint8)
    return out


def save_rgb(path: Path, image_rgb: np.ndarray) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))


def _id_to_color(value: int) -> np.ndarray:
    # Deterministic integer hash to an RGB color with decent saturation.
    x = np.uint32(value)
    x ^= x >> np.uint32(16)
    x *= np.uint32(0x7FEB352D)
    x ^= x >> np.uint32(15)
    x *= np.uint32(0x846CA68B)
    x ^= x >> np.uint32(16)
    return np.array(
        [
            64 + int(x & np.uint32(127)),
            64 + int((x >> np.uint32(8)) & np.uint32(127)),
            64 + int((x >> np.uint32(16)) & np.uint32(127)),
        ],
        dtype=np.uint8,
    )
