"""Visualization helpers for label masks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

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


def save_labeled_summary(
    path: Path,
    image_rgb: np.ndarray,
    semantic: np.ndarray,
    instance: np.ndarray,
    *,
    objects: Dict[int, Any],
    semantic_ignore_label: int,
    alpha: float = 0.55,
    max_labels: int = 24,
) -> None:
    """Save RGB/semantic/instance panels with instance id labels."""
    import cv2

    sem_color = colorize_ids(semantic, ignore_values=(semantic_ignore_label,))
    inst_color = colorize_ids(instance, ignore_values=(0,))
    sem_overlay = overlay_labels(image_rgb, semantic, sem_color, alpha=alpha)
    inst_overlay = overlay_labels(image_rgb, instance, inst_color, alpha=alpha)
    inst_overlay = draw_instance_labels(
        inst_overlay,
        instance,
        objects=objects,
        max_labels=max_labels,
    )

    panels = [
        add_panel_title(image_rgb, "RGB"),
        add_panel_title(sem_overlay, "Semantic projection"),
        add_panel_title(inst_overlay, "Instance projection"),
    ]
    canvas = np.concatenate(panels, axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def add_panel_title(image_rgb: np.ndarray, title: str) -> np.ndarray:
    import cv2

    out = image_rgb.copy()
    h, w = out.shape[:2]
    title_h = max(26, min(44, h // 16))
    cv2.rectangle(out, (0, 0), (w, title_h), (0, 0, 0), thickness=-1)
    cv2.putText(
        out,
        title,
        (8, max(18, title_h - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        lineType=cv2.LINE_AA,
    )
    return out


def draw_instance_labels(
    image_rgb: np.ndarray,
    instance: np.ndarray,
    *,
    objects: Dict[int, Any],
    max_labels: int,
) -> np.ndarray:
    import cv2

    out = image_rgb.copy()
    ids, counts = np.unique(instance[instance > 0], return_counts=True)
    if ids.size == 0:
        return out
    order = np.argsort(counts)[::-1]
    for idx in order[:max_labels]:
        instance_id = int(ids[idx])
        mask = instance == instance_id
        ys, xs = np.where(mask)
        if xs.size == 0:
            continue
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        x = int(np.median(xs))
        y = int(np.median(ys))
        label = object_label(objects.get(instance_id))
        text = f"{instance_id}:{label}" if label else str(instance_id)
        cv2.rectangle(out, (x0, y0), (x1, y1), (255, 255, 255), thickness=1)
        cv2.putText(
            out,
            text[:48],
            (max(2, x), max(18, y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            3,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            out,
            text[:48],
            (max(2, x), max(18, y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            lineType=cv2.LINE_AA,
        )
    return out


def object_label(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in (
            "label",
            "label_name",
            "class",
            "class_name",
            "category",
            "category_name",
            "rawLabel",
        ):
            item = value.get(key)
            if isinstance(item, str) and item:
                return item
        semantic = value.get("semantic_id")
        if semantic is not None:
            return f"semantic_{semantic}"
    return ""


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
