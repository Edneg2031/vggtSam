"""Configuration loading for the streaming coupling experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    manifest: Path
    scene_id: str
    frame_indices: tuple[int, ...]
    min_pixels: int
    max_area_ratio: float
    excluded_labels: tuple[str, ...]

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
    max_points_per_object: int
    max_points_per_observation: int

    box_quantile: float
    box_padding_ratio: float
    min_projected_points: int
    min_projected_fraction: float
    min_supported_points: int
    min_support_ratio: float
    support_abs_distance: float
    support_relative_distance: float

    tracker_low_score: float
    fallback_on_missing_mask: bool
    fallback_on_geometry_disagreement: bool
    tracker_min_geometry_coverage: float
    recovery_min_support_coverage: float
    map_update_min_score: float
    map_update_min_geometry_coverage: float

    point_cloud_confidence_threshold: float
    point_cloud_max_points: int
    map_metric_max_points: int
    map_metric_thresholds: tuple[float, ...]
    output_dir: Path


def load_config(
    path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> ExperimentConfig:
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf8") as handle:
        raw = yaml.safe_load(handle) or {}
    overrides = overrides or {}
    dataset = raw.get("dataset", {})
    sam3 = raw.get("sam3", {})
    stream = raw.get("streamvggt", {})
    bridge = raw.get("bridge", {})
    candidate = raw.get("candidate", {})
    point_cloud = raw.get("point_cloud", {})

    frame_indices = overrides.get(
        "frame_indices",
        dataset.get("frame_indices", []),
    )
    if not frame_indices:
        raise ValueError("dataset.frame_indices must contain at least one frame index.")
    output_size = tuple(int(value) for value in bridge.get("output_size", [256, 384]))
    if len(output_size) != 2:
        raise ValueError("bridge.output_size must be [height, width].")

    config = ExperimentConfig(
        manifest=_path(overrides.get("manifest", dataset.get("manifest"))),
        scene_id=str(overrides.get("scene_id", dataset.get("scene_id"))),
        frame_indices=tuple(int(value) for value in frame_indices),
        min_pixels=int(dataset.get("min_pixels", 128)),
        max_area_ratio=float(dataset.get("max_area_ratio", 0.25)),
        excluded_labels=tuple(
            str(value) for value in dataset.get("excluded_labels", [])
        ),
        sam3_repo=_path(sam3.get("repo", "externals/sam3")),
        sam3_checkpoint=_path(sam3.get("checkpoint")),
        sam3_device=str(overrides.get("sam3_device", sam3.get("device", "cuda:0"))),
        sam3_output_threshold=float(sam3.get("output_threshold", 0.5)),
        prompt_with_box=bool(sam3.get("prompt_with_box", True)),
        streamvggt_repo=_path(stream.get("repo", "externals/streamvggt")),
        streamvggt_checkpoint=_path(stream.get("checkpoint")),
        geometry_device=str(
            overrides.get("geometry_device", stream.get("device", "cuda:1"))
        ),
        streaming_cache=bool(stream.get("streaming_cache", True)),
        image_mode=str(stream.get("image_mode", "crop")),
        output_size=(output_size[0], output_size[1]),
        max_points_per_object=int(bridge.get("max_points_per_object", 20000)),
        max_points_per_observation=int(
            bridge.get("max_points_per_observation", 8000)
        ),
        box_quantile=float(candidate.get("box_quantile", 0.02)),
        box_padding_ratio=float(candidate.get("box_padding_ratio", 0.12)),
        min_projected_points=int(candidate.get("min_projected_points", 24)),
        min_projected_fraction=float(candidate.get("min_projected_fraction", 0.005)),
        min_supported_points=int(candidate.get("min_supported_points", 8)),
        min_support_ratio=float(candidate.get("min_support_ratio", 0.02)),
        support_abs_distance=float(candidate.get("support_abs_distance", 0.15)),
        support_relative_distance=float(
            candidate.get("support_relative_distance", 0.10)
        ),
        tracker_low_score=float(bridge.get("tracker_low_score", 0.5)),
        fallback_on_missing_mask=bool(
            bridge.get("fallback_on_missing_mask", True)
        ),
        fallback_on_geometry_disagreement=bool(
            bridge.get("fallback_on_geometry_disagreement", True)
        ),
        tracker_min_geometry_coverage=float(
            bridge.get("tracker_min_geometry_coverage", 0.25)
        ),
        recovery_min_support_coverage=float(
            bridge.get("recovery_min_support_coverage", 0.50)
        ),
        map_update_min_score=float(
            bridge.get("map_update_min_score", 0.50)
        ),
        map_update_min_geometry_coverage=float(
            bridge.get("map_update_min_geometry_coverage", 0.50)
        ),
        point_cloud_confidence_threshold=float(
            point_cloud.get("confidence_threshold", 0.30)
        ),
        point_cloud_max_points=int(
            point_cloud.get("max_points", 400_000)
        ),
        map_metric_max_points=int(
            point_cloud.get("metric_max_points", 4096)
        ),
        map_metric_thresholds=tuple(
            float(value)
            for value in point_cloud.get(
                "metric_thresholds",
                [0.05, 0.10],
            )
        ),
        output_dir=_path(
            overrides.get("output_dir", raw.get("output", {}).get("dir"))
        ),
    )
    for name, value in (
        (
            "bridge.tracker_min_geometry_coverage",
            config.tracker_min_geometry_coverage,
        ),
        (
            "bridge.recovery_min_support_coverage",
            config.recovery_min_support_coverage,
        ),
        ("bridge.map_update_min_score", config.map_update_min_score),
        (
            "bridge.map_update_min_geometry_coverage",
            config.map_update_min_geometry_coverage,
        ),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1].")
    if not 0.0 <= config.point_cloud_confidence_threshold <= 1.0:
        raise ValueError(
            "point_cloud.confidence_threshold must be in [0, 1]."
        )
    if config.point_cloud_max_points < 1:
        raise ValueError("point_cloud.max_points must be positive.")
    if config.map_metric_max_points < 1:
        raise ValueError("point_cloud.metric_max_points must be positive.")
    if not config.map_metric_thresholds or any(
        value <= 0.0 for value in config.map_metric_thresholds
    ):
        raise ValueError(
            "point_cloud.metric_thresholds must contain positive distances."
        )
    return config


def _path(value: Any) -> Path:
    if value is None or str(value).strip() == "":
        raise ValueError("A required path is missing from the configuration.")
    return Path(value).expanduser()
