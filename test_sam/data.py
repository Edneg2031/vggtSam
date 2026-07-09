"""Minimal ScanNet++ sequence loader for mask-tracking ablations."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image

from vggtsam.data.scannetpp.object_sequence import extract_object_labels


@dataclass(frozen=True)
class MaskTrackingSequence:
    scene_id: str
    frame_indices: list[int]
    image_paths: list[Path]
    instance_masks: list[np.ndarray]
    instance_id: int
    label: str
    reference_frame_idx: int

    @property
    def target_masks(self) -> list[np.ndarray]:
        return [mask == self.instance_id for mask in self.instance_masks]


def load_mask_tracking_sequence(
    manifest_path: str | Path,
    *,
    scene_id: str,
    frame_indices: Sequence[int] | None,
    sequence_length: int,
    frame_stride: int,
    window_index: int,
    instance_id: int | None,
    min_pixels: int,
    max_area_ratio: float,
    min_visible_frames: int,
    excluded_labels: Iterable[str],
    seed: int,
) -> MaskTrackingSequence:
    manifest_path = Path(manifest_path)
    with manifest_path.open("r", encoding="utf8") as handle:
        manifest = json.load(handle)

    scene = next(
        (item for item in manifest.get("scenes", []) if item.get("scene_id") == scene_id),
        None,
    )
    if scene is None:
        available = [item.get("scene_id") for item in manifest.get("scenes", [])]
        raise ValueError(
            f"Scene {scene_id!r} is not present in {manifest_path}. "
            f"Available scenes: {available[:20]}"
        )

    frames = scene.get("frames", [])
    selected_indices = resolve_frame_indices(
        num_frames=len(frames),
        frame_indices=frame_indices,
        sequence_length=sequence_length,
        frame_stride=frame_stride,
        window_index=window_index,
    )
    selected_frames = [frames[index] for index in selected_indices]
    image_paths = [
        resolve_manifest_path(frame["image_path"], manifest_path)
        for frame in selected_frames
    ]
    instance_masks = [
        read_mask(resolve_manifest_path(frame["instance_mask"], manifest_path))
        for frame in selected_frames
    ]
    labels = extract_object_labels(scene.get("objects", {}))

    if instance_id is None:
        instance_id = choose_instance(
            instance_masks,
            labels=labels,
            min_pixels=min_pixels,
            max_area_ratio=max_area_ratio,
            min_visible_frames=min_visible_frames,
            excluded_labels=excluded_labels,
            seed=seed,
        )
    instance_id = int(instance_id)
    label = labels.get(instance_id, "object")
    counts = [int((mask == instance_id).sum()) for mask in instance_masks]
    if max(counts, default=0) == 0:
        raise ValueError(
            f"Instance {instance_id} is absent from frames {selected_indices}."
        )
    reference_frame_idx = int(np.argmax(counts))

    return MaskTrackingSequence(
        scene_id=scene_id,
        frame_indices=selected_indices,
        image_paths=image_paths,
        instance_masks=instance_masks,
        instance_id=instance_id,
        label=label,
        reference_frame_idx=reference_frame_idx,
    )


def resolve_frame_indices(
    *,
    num_frames: int,
    frame_indices: Sequence[int] | None,
    sequence_length: int,
    frame_stride: int,
    window_index: int,
) -> list[int]:
    if frame_indices:
        indices = [int(index) for index in frame_indices]
    else:
        start = int(window_index)
        indices = [
            start + offset * int(frame_stride)
            for offset in range(int(sequence_length))
        ]
    invalid = [index for index in indices if index < 0 or index >= num_frames]
    if invalid:
        raise ValueError(
            f"Frame indices {invalid} are outside [0, {num_frames - 1}]."
        )
    return indices


def choose_instance(
    masks: Sequence[np.ndarray],
    *,
    labels: dict[int, str],
    min_pixels: int,
    max_area_ratio: float,
    min_visible_frames: int,
    excluded_labels: Iterable[str],
    seed: int,
) -> int:
    excluded = {label.strip().lower() for label in excluded_labels}
    visibility: dict[int, int] = {}
    total_pixels = float(masks[0].shape[0] * masks[0].shape[1])
    for mask in masks:
        ids, counts = np.unique(mask, return_counts=True)
        for raw_id, count in zip(ids, counts):
            current_id = int(raw_id)
            if current_id == 0:
                continue
            if int(count) < int(min_pixels):
                continue
            if float(count) / total_pixels > float(max_area_ratio):
                continue
            if labels.get(current_id, "object").strip().lower() in excluded:
                continue
            visibility[current_id] = visibility.get(current_id, 0) + 1
    candidates = sorted(
        instance
        for instance, count in visibility.items()
        if count >= int(min_visible_frames)
    )
    if not candidates:
        raise RuntimeError(
            "No valid instance remains after size, visibility, and label filtering."
        )
    return random.Random(seed).choice(candidates)


def resolve_manifest_path(value: str | Path, manifest_path: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (manifest_path.parent / path).resolve()


def read_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image).copy()

