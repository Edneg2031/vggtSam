"""Minimal ScanNet++ manifest, mask, and pointmap readers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class RGBSequence:
    scene_id: str
    frame_indices: list[int]
    image_paths: list[Path]


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


def load_rgb_sequence(
    manifest_path: str | Path,
    *,
    scene_id: str,
    frame_indices: Sequence[int],
) -> RGBSequence:
    """Load only RGB paths, without requiring a GT instance to be visible."""

    manifest_path = Path(manifest_path).expanduser().resolve()
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
        sequence_length=len(frame_indices),
        frame_stride=1,
        window_index=0,
    )
    return RGBSequence(
        scene_id=scene_id,
        frame_indices=selected_indices,
        image_paths=[
            resolve_manifest_path(frames[index]["image_path"], manifest_path)
            for index in selected_indices
        ],
    )


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
    manifest_path = Path(manifest_path).expanduser().resolve()
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

    # Preprocessing manifests may store either repository-relative paths such
    # as data/processed/... or paths relative to the manifest directory.
    candidates = [
        (Path.cwd() / path).resolve(),
        (manifest_path.parent / path).resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    attempted = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        f"Could not resolve manifest path {str(value)!r}. Tried:\n{attempted}"
    )


def read_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image).copy()


def read_pointmap(path: str | Path) -> np.ndarray:
    payload = np.load(path)
    if isinstance(payload, np.lib.npyio.NpzFile):
        try:
            key = "pointmap" if "pointmap" in payload.files else payload.files[0]
            pointmap = payload[key]
        finally:
            payload.close()
    else:
        pointmap = payload
    pointmap = np.asarray(pointmap, dtype=np.float32)
    if pointmap.ndim != 3 or pointmap.shape[-1] != 3:
        raise ValueError(
            f"Pointmap must have shape [H, W, 3], got {pointmap.shape}: {path}"
        )
    return pointmap


def extract_object_labels(objects: dict[str, Any]) -> dict[int, str]:
    labels: dict[int, str] = {}
    for object_id, metadata in objects.items():
        try:
            instance_id = int(object_id)
        except (TypeError, ValueError):
            continue
        label = _find_label_string(metadata)
        if label:
            labels[instance_id] = label
    return labels


def _find_label_string(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        preferred = (
            "label",
            "label_name",
            "labelName",
            "class",
            "class_name",
            "className",
            "category",
            "category_name",
            "nyuClass",
            "rawLabel",
        )
        for key in preferred:
            item = value.get(key)
            if isinstance(item, str) and item:
                return item
        for item in value.values():
            found = _find_label_string(item)
            if found:
                return found
    if isinstance(value, list):
        for item in value:
            found = _find_label_string(item)
            if found:
                return found
    return None
