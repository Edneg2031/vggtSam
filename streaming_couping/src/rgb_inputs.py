"""Deterministic RGB input expansion shared by model backends."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .data import resolve_manifest_path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class RGBInputSelection:
    """Resolved image paths plus their positions in the original source."""

    image_paths: tuple[Path, ...]
    source_positions: tuple[int, ...]
    source_count: int
    metadata: dict[str, Any]


def resolve_rgb_inputs(
    *,
    frames: Sequence[Path] | None = None,
    manifest: Path | None = None,
    scene_id: str | None = None,
    dataset_frame_indices: Sequence[int] | None = None,
    start: int = 0,
    stride: int = 1,
    count: int = 0,
) -> RGBInputSelection:
    """Expand one RGB source and apply deterministic positional sampling."""

    if bool(frames) == bool(manifest):
        raise ValueError("Provide exactly one of frames or manifest.")

    if manifest is not None:
        paths, source_positions, source_count = expand_manifest_paths(
            manifest,
            scene_id=scene_id,
            frame_indices=dataset_frame_indices,
        )
        metadata: dict[str, Any] = {
            "manifest": str(manifest.expanduser().resolve()),
            "scene_id": str(scene_id),
            "dataset_frame_indices": (
                None
                if dataset_frame_indices is None
                else [int(value) for value in dataset_frame_indices]
            ),
        }
    else:
        paths = expand_image_paths(frames)
        source_positions = tuple(range(len(paths)))
        source_count = len(paths)
        metadata = {"frame_paths": [str(path) for path in paths]}

    positions = selected_positions(len(paths), start, stride, count)
    selected_paths = tuple(paths[index] for index in positions)
    selected_source_positions = tuple(source_positions[index] for index in positions)
    return RGBInputSelection(
        image_paths=selected_paths,
        source_positions=selected_source_positions,
        source_count=int(source_count),
        metadata=metadata,
    )


def expand_image_paths(values: Sequence[Path] | None) -> tuple[Path, ...]:
    if not values:
        return ()
    output: list[Path] = []
    for value in values:
        path = Path(value).expanduser()
        if path.is_dir():
            output.extend(
                sorted(
                    candidate
                    for candidate in path.iterdir()
                    if candidate.is_file()
                    and candidate.suffix.lower() in IMAGE_SUFFIXES
                )
            )
        elif path.is_file():
            output.append(path)
        else:
            raise FileNotFoundError(f"RGB path does not exist: {path}")
    return tuple(path.resolve() for path in output)


def expand_manifest_paths(
    manifest_path: Path,
    *,
    scene_id: str | None,
    frame_indices: Sequence[int] | None,
) -> tuple[tuple[Path, ...], tuple[int, ...], int]:
    """Resolve RGB paths and original scene positions from a manifest."""

    manifest_path = manifest_path.expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"RGB manifest does not exist: {manifest_path}")
    if not str(scene_id or "").strip():
        raise ValueError("--scene-id is required when --manifest is used.")
    with manifest_path.open("r", encoding="utf8") as handle:
        manifest = json.load(handle)
    scene = next(
        (
            item
            for item in manifest.get("scenes", [])
            if str(item.get("scene_id")) == str(scene_id)
        ),
        None,
    )
    if scene is None:
        available = [item.get("scene_id") for item in manifest.get("scenes", [])]
        raise ValueError(
            f"Scene {scene_id!r} is not present in {manifest_path}. "
            f"Available scenes: {available[:20]}"
        )
    frames = scene.get("frames", [])
    indices = (
        tuple(range(len(frames)))
        if frame_indices is None
        else tuple(int(value) for value in frame_indices)
    )
    invalid = [index for index in indices if index < 0 or index >= len(frames)]
    if invalid:
        raise ValueError(
            f"Manifest frame positions {invalid} are outside [0, {len(frames) - 1}]."
        )
    paths = tuple(
        resolve_manifest_path(frames[index]["image_path"], manifest_path).resolve()
        for index in indices
    )
    return paths, indices, len(frames)


def select_image_paths(
    paths: Sequence[Path],
    *,
    start: int,
    stride: int,
    count: int,
) -> tuple[Path, ...]:
    positions = selected_positions(len(paths), start, stride, count)
    return tuple(Path(paths[index]) for index in positions)


def selected_positions(total: int, start: int, stride: int, count: int) -> list[int]:
    start = int(start)
    stride = int(stride)
    count = int(count)
    if start < 0:
        raise ValueError("--frame-start must be non-negative.")
    if stride < 1:
        raise ValueError("--frame-stride must be positive.")
    if count < 0:
        raise ValueError("--frame-count must be non-negative; use 0 for all.")
    positions = list(range(start, int(total), stride))
    return positions[:count] if count else positions
