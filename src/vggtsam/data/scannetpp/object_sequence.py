"""Object-centric ScanNet++ sequence sampling."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from .io import read_json


@dataclass(frozen=True)
class ObjectSamplingConfig:
    min_pixels: int = 128
    max_area_ratio: float = 0.25
    min_visible_frames: int = 2
    max_objects_per_frame: int = 32
    ignore_instance_id: int = 0
    semantic_ignore_label: int = 65535


@dataclass
class ObjectSequence:
    scene_id: str
    frame_indices: List[int]
    image_paths: List[Path]
    instance_masks: List[np.ndarray]
    semantic_masks: List[np.ndarray]
    visible_instance_ids: List[List[int]]


class ScanNetPPObjectSequenceDataset:
    """Sample continuous clips with cross-frame-consistent instance IDs."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        scene_id: Optional[str] = None,
        sequence_length: int = 4,
        frame_stride: int = 1,
        object_config: Optional[ObjectSamplingConfig] = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.manifest = read_json(self.manifest_path)
        self.sequence_length = int(sequence_length)
        self.frame_stride = int(frame_stride)
        self.object_config = object_config or ObjectSamplingConfig()

        scenes = self.manifest.get("scenes", [])
        if scene_id is not None:
            scenes = [scene for scene in scenes if scene.get("scene_id") == scene_id]
        if not scenes:
            raise ValueError(f"No scenes found in {self.manifest_path} for scene_id={scene_id!r}")

        self.scenes = scenes
        self.windows: List[Dict[str, Any]] = []
        for scene in scenes:
            frames = scene.get("frames", [])
            window_size = (self.sequence_length - 1) * self.frame_stride + 1
            if len(frames) < window_size:
                continue
            for start in range(0, len(frames) - window_size + 1):
                indices = [start + i * self.frame_stride for i in range(self.sequence_length)]
                self.windows.append({"scene": scene, "indices": indices})

        if not self.windows:
            raise ValueError(
                f"No valid windows in {self.manifest_path}; sequence_length={sequence_length}, "
                f"frame_stride={frame_stride}"
            )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> ObjectSequence:
        window = self.windows[index % len(self.windows)]
        scene = window["scene"]
        frames = scene["frames"]
        frame_indices = list(window["indices"])
        selected = [frames[i] for i in frame_indices]

        image_paths = [Path(frame["image_path"]) for frame in selected]
        instance_masks = [read_mask(frame["instance_mask"]) for frame in selected]
        semantic_masks = [read_mask(frame["semantic_mask"]) for frame in selected]
        visible_instance_ids = [
            filter_visible_instances(
                inst,
                object_config=self.object_config,
            )
            for inst in instance_masks
        ]
        return ObjectSequence(
            scene_id=scene["scene_id"],
            frame_indices=frame_indices,
            image_paths=image_paths,
            instance_masks=instance_masks,
            semantic_masks=semantic_masks,
            visible_instance_ids=visible_instance_ids,
        )

    def sample(self, rng: random.Random) -> ObjectSequence:
        return self[rng.randrange(len(self.windows))]


def read_mask(path: str | Path) -> np.ndarray:
    image = Image.open(path)
    return np.asarray(image)


def filter_visible_instances(
    instance_mask: np.ndarray,
    *,
    object_config: ObjectSamplingConfig,
) -> List[int]:
    height, width = instance_mask.shape[:2]
    total = float(height * width)
    ids, counts = np.unique(instance_mask, return_counts=True)
    out: List[int] = []
    for instance_id, count in zip(ids, counts):
        instance_id = int(instance_id)
        if instance_id == object_config.ignore_instance_id:
            continue
        if count < object_config.min_pixels:
            continue
        if count / total > object_config.max_area_ratio:
            continue
        out.append(instance_id)
    return out[: object_config.max_objects_per_frame]


def keep_instances_visible_in_multiple_frames(
    per_frame_ids: List[List[int]],
    *,
    min_visible_frames: int,
) -> List[set[int]]:
    counts: Dict[int, int] = {}
    for ids in per_frame_ids:
        for instance_id in set(ids):
            counts[instance_id] = counts.get(instance_id, 0) + 1
    keep = {
        instance_id
        for instance_id, count in counts.items()
        if count >= min_visible_frames
    }
    return [set(ids).intersection(keep) for ids in per_frame_ids]
