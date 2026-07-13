"""Configuration loading for explicit dual-framework experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .gates import GateConfig


@dataclass(frozen=True)
class ExperimentConfig:
    manifest: Path
    scene_id: str
    frame_indices: list[int]
    instance_id: int
    min_pixels: int
    max_area_ratio: float
    excluded_labels: list[str]
    sam3_repo: Path
    sam3_checkpoint: Path
    sam3_device: str
    sam3_output_threshold: float
    prompt_with_box: bool
    streamvggt_repo: Path
    streamvggt_checkpoint: Path
    geometry_device: str
    streaming_cache: bool
    image_mode: str
    output_size: tuple[int, int]
    geometry_modes: tuple[str, ...]
    max_points_per_observation: int
    max_points_per_object: int
    splat_radius: int
    occlusion_depth_tolerance: float
    occlusion_relative_tolerance: float
    map_update_min_prior_iou: float
    map_update_source: str
    gates: GateConfig
    output_dir: Path


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> ExperimentConfig:
    path = Path(path)
    with path.open("r", encoding="utf8") as handle:
        raw = yaml.safe_load(handle)
    overrides = overrides or {}
    dataset = raw["dataset"]
    sam3 = raw["sam3"]
    stream = raw["streamvggt"]
    bridge = raw["bridge"]
    output = raw["output"]
    frame_indices = overrides.get("frame_indices", dataset["frame_indices"])
    modes = overrides.get("geometry_modes", bridge.get("geometry_modes", ["aligned"]))
    valid_modes = {"aligned", "zero", "shuffled"}
    invalid = set(modes) - valid_modes
    if invalid:
        raise ValueError(f"Unknown geometry modes: {sorted(invalid)}")
    map_update_source = str(bridge.get("map_update_source", "sam3"))
    if map_update_source not in {"sam3", "oracle"}:
        raise ValueError("bridge.map_update_source must be 'sam3' or 'oracle'.")
    return ExperimentConfig(
        manifest=Path(dataset["manifest"]),
        scene_id=str(overrides.get("scene_id", dataset["scene_id"])),
        frame_indices=[int(value) for value in frame_indices],
        instance_id=int(overrides.get("instance_id", dataset["instance_id"])),
        min_pixels=int(dataset.get("min_pixels", 128)),
        max_area_ratio=float(dataset.get("max_area_ratio", 0.25)),
        excluded_labels=[str(value) for value in dataset.get("excluded_labels", [])],
        sam3_repo=Path(sam3["repo"]),
        sam3_checkpoint=Path(sam3["checkpoint"]),
        sam3_device=str(overrides.get("sam3_device", sam3["device"])),
        sam3_output_threshold=float(sam3.get("output_threshold", 0.5)),
        prompt_with_box=bool(sam3.get("prompt_with_box", True)),
        streamvggt_repo=Path(stream["repo"]),
        streamvggt_checkpoint=Path(stream["checkpoint"]),
        geometry_device=str(overrides.get("geometry_device", stream["device"])),
        streaming_cache=bool(stream.get("streaming_cache", True)),
        image_mode=str(stream.get("image_mode", "crop")),
        output_size=tuple(int(value) for value in bridge.get("output_size", [256, 384])),
        geometry_modes=tuple(str(value) for value in modes),
        max_points_per_observation=int(bridge.get("max_points_per_observation", 8000)),
        max_points_per_object=int(bridge.get("max_points_per_object", 20000)),
        splat_radius=int(bridge.get("splat_radius", 3)),
        occlusion_depth_tolerance=float(bridge.get("occlusion_depth_tolerance", 0.02)),
        occlusion_relative_tolerance=float(bridge.get("occlusion_relative_tolerance", 0.05)),
        map_update_min_prior_iou=float(bridge.get("map_update_min_prior_iou", 0.1)),
        map_update_source=map_update_source,
        gates=GateConfig(
            track_update_threshold=float(bridge.get("track_update_threshold", 0.7)),
            track_fallback_threshold=float(bridge.get("track_fallback_threshold", 0.5)),
            geometry_threshold=float(bridge.get("geometry_threshold", 0.45)),
            min_persistence=int(bridge.get("min_persistence", 1)),
        ),
        output_dir=Path(overrides.get("output_dir", output["dir"])),
    )
