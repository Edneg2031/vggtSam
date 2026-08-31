#!/usr/bin/env python3
"""Build a semantic-instance map from RGB prompts or a frozen V0 cache.

The model path uses the current StreamVGGT and SAM3.1 wrappers.  The cache path
is useful for replaying a V0 run without rerunning either model.  Both paths
converge on the same backend-neutral ``SemanticMapPipeline``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from streaming_couping.src.config import load_config
from streaming_couping.src.data import resolve_manifest_path
from streaming_couping.src.semantic_mapping.adapters import (
    SAM31SegmentationAdapter,
    StreamVGGTGeometryAdapter,
    V0CacheGeometryAdapter,
    V0CacheSegmentationAdapter,
)
from streaming_couping.src.semantic_mapping.export import export_semantic_map
from streaming_couping.src.semantic_mapping.mapping import (
    SemanticMapBuilder,
    SemanticMapConfig,
)
from streaming_couping.src.semantic_mapping.pipeline import SemanticMapPipeline


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    args = _parse_args()
    prompts = _normalize_prompts(args.prompts)
    mapper = SemanticMapBuilder(
        SemanticMapConfig(
            voxel_size_m=args.voxel_size_m,
            min_geometry_confidence=args.min_geometry_confidence,
            min_track_score=args.min_track_score,
            static_score_threshold=args.static_score_threshold,
            require_static_score=args.require_static_score,
            include_dynamic_tracks=not args.exclude_dynamic_tracks,
            max_points_per_observation=args.max_points_per_observation,
            max_points_per_track=args.max_points_per_track,
            max_voxels=args.max_voxels,
        )
    )
    if args.cache:
        result = _run_from_cache(args, prompts, mapper)
    else:
        result = _run_from_rgb(args, prompts, mapper)
    summary = export_semantic_map(result, args.output_dir)
    print(
        "semantic map completed "
        f"frames={summary['metadata'].get('frame_count', 0)} "
        f"voxels={summary['voxel_count']} "
        f"objects={summary['instance_count']}"
    )
    print(f"semantic_map={summary['outputs']['artifact']}")
    print(f"semantic_ply={summary['outputs']['semantic_ply']}")


def _run_from_cache(
    args: argparse.Namespace,
    prompts: tuple[str, ...],
    mapper: SemanticMapBuilder,
):
    payload = _load_payload(args.cache)
    payload = _select_cache_frames(
        payload,
        start=args.frame_start,
        stride=args.frame_stride,
        count=args.frame_count,
    )
    geometry = V0CacheGeometryAdapter(payload, scale_type=args.scale_type)
    segmentation = V0CacheSegmentationAdapter(
        payload,
        mask_field=args.mask_field,
    )
    image_paths = tuple(
        Path(path)
        for path in payload.get(
            "image_paths",
            [str(index) for index in range(len(geometry.frame_ids))],
        )
    )
    return SemanticMapPipeline(
        geometry=geometry,
        segmentation=segmentation,
        mapper=mapper,
    ).run(
        image_paths,
        prompts=prompts or None,
        metadata={
            "input_mode": "frozen_v0_cache",
            "cache": str(Path(args.cache).expanduser().resolve()),
            "scale_type": args.scale_type,
            "frame_selection": {
                "start": int(args.frame_start),
                "stride": int(args.frame_stride),
                "count": int(args.frame_count),
            },
        },
    )


def _run_from_rgb(
    args: argparse.Namespace,
    prompts: tuple[str, ...],
    mapper: SemanticMapBuilder,
):
    if not prompts:
        raise ValueError("--prompts is required when --cache is not used.")
    if args.manifest:
        image_paths = _expand_manifest_paths(
            args.manifest,
            scene_id=args.scene_id,
            frame_indices=args.dataset_frame_indices,
        )
        input_metadata = {
            "manifest": str(args.manifest.expanduser().resolve()),
            "scene_id": str(args.scene_id),
            "dataset_frame_indices": (
                None
                if args.dataset_frame_indices is None
                else [int(value) for value in args.dataset_frame_indices]
            ),
        }
    else:
        image_paths = _expand_image_paths(args.frames)
        input_metadata = {
            "frame_paths": [str(path) for path in image_paths],
        }
    image_paths = _select_image_paths(
        image_paths,
        start=args.frame_start,
        stride=args.frame_stride,
        count=args.frame_count,
    )
    if not image_paths:
        raise ValueError("No RGB images were selected from the requested input.")
    overrides = {}
    if args.sam_device:
        overrides["sam3_device"] = args.sam_device
    if args.geometry_device:
        overrides["geometry_device"] = args.geometry_device
    recovery = load_config(args.recovery_config, overrides)
    from streaming_couping.src.backbones.sam3_wrapper import SAM3Wrapper
    from streaming_couping.src.backbones.streamvggt_wrapper import StreamVGGTWrapper

    stream = StreamVGGTWrapper(
        repo_path=recovery.streamvggt_repo,
        checkpoint_path=recovery.streamvggt_checkpoint,
        device=recovery.geometry_device,
        image_mode=recovery.image_mode,
        streaming_cache=recovery.streaming_cache,
    ).load()
    sam = SAM3Wrapper(
        repo_path=recovery.sam3_repo,
        checkpoint_path=recovery.sam3_checkpoint,
        device=recovery.sam3_device,
        output_threshold=recovery.sam3_output_threshold,
        prompt_with_box=recovery.prompt_with_box,
        version=recovery.sam3_version,
        use_fa3=recovery.sam3_use_fa3,
        max_num_objects=recovery.sam3_max_num_objects,
        multiplex_count=recovery.sam3_multiplex_count,
    ).load()
    geometry = StreamVGGTGeometryAdapter(
        stream,
        scale_type=args.scale_type,
    )
    segmentation = SAM31SegmentationAdapter(
        sam,
        output_size=tuple(args.output_size or recovery.output_size),
        max_objects_per_prompt=recovery.sam3_max_num_objects,
        max_total_objects=args.max_objects,
        min_birth_pixels=recovery.min_pixels,
    )
    return SemanticMapPipeline(
        geometry=geometry,
        segmentation=segmentation,
        mapper=mapper,
    ).run(
        image_paths,
        prompts=prompts,
        metadata={
            "input_mode": "rgb_and_text_prompts",
            "scale_type": args.scale_type,
            "recovery_config": str(Path(args.recovery_config).resolve()),
            "frame_selection": {
                "start": int(args.frame_start),
                "stride": int(args.frame_stride),
                "count": int(args.frame_count),
            },
            **input_metadata,
        },
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--cache",
        type=Path,
        help="Frozen V0 feature cache; skips StreamVGGT and SAM inference.",
    )
    source.add_argument(
        "--frames",
        nargs="+",
        type=Path,
        help="RGB files and/or directories, sorted within each directory.",
    )
    source.add_argument(
        "--manifest",
        type=Path,
        help="Processed ScanNet++ manifest containing scene RGB paths.",
    )
    parser.add_argument(
        "--scene-id",
        default=None,
        help="Scene ID used with --manifest.",
    )
    parser.add_argument(
        "--dataset-frame-indices",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Frame positions in the manifest scene. Omit to use every scene "
            "frame; selection is then applied in manifest order."
        ),
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        help="Text prompts, either as separate arguments or comma-separated.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for semantic_map.pt, PLY, and JSON artifacts.",
    )
    parser.add_argument(
        "--recovery-config",
        type=Path,
        default=Path("streaming_couping/configs/recovery_dynamic_instance.yaml"),
    )
    parser.add_argument("--geometry-device", default=None)
    parser.add_argument("--sam-device", default=None)
    parser.add_argument("--output-size", type=_parse_size, default=None)
    parser.add_argument(
        "--frame-start",
        type=int,
        default=0,
        help="0-based position in the sorted RGB inputs at which to start.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Take every Nth RGB input after frame-start.",
    )
    parser.add_argument(
        "--frame-count",
        type=int,
        default=0,
        help="Number of selected RGB inputs; 0 means all remaining inputs.",
    )
    parser.add_argument("--max-objects", type=int, default=16)
    parser.add_argument("--mask-field", default="tracking_masks_stream")
    parser.add_argument(
        "--scale-type",
        choices=("metric", "relative", "unknown"),
        default="unknown",
    )
    parser.add_argument("--voxel-size-m", type=float, default=0.05)
    parser.add_argument("--min-geometry-confidence", type=float, default=0.30)
    parser.add_argument("--min-track-score", type=float, default=0.50)
    parser.add_argument("--static-score-threshold", type=float, default=0.20)
    parser.add_argument("--require-static-score", action="store_true")
    parser.add_argument("--exclude-dynamic-tracks", action="store_true")
    parser.add_argument("--max-points-per-observation", type=int, default=8_000)
    parser.add_argument("--max-points-per-track", type=int, default=250_000)
    parser.add_argument("--max-voxels", type=int, default=1_000_000)
    return parser.parse_args()


def _load_payload(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"V0 cache does not exist: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a dictionary V0 cache, got {type(payload)!r}.")
    return payload


def _expand_image_paths(values: list[Path] | None) -> tuple[Path, ...]:
    if not values:
        return ()
    output = []
    for value in values:
        path = value.expanduser()
        if path.is_dir():
            output.extend(
                sorted(
                    candidate
                    for candidate in path.iterdir()
                    if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES
                )
            )
        elif path.is_file():
            output.append(path)
        else:
            raise FileNotFoundError(f"RGB path does not exist: {path}")
    return tuple(output)


def _expand_manifest_paths(
    manifest_path: Path,
    *,
    scene_id: str | None,
    frame_indices: list[int] | None,
) -> tuple[Path, ...]:
    """Resolve RGB paths from the processed dataset manifest."""

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
        list(range(len(frames)))
        if frame_indices is None
        else [int(value) for value in frame_indices]
    )
    invalid = [index for index in indices if index < 0 or index >= len(frames)]
    if invalid:
        raise ValueError(
            f"Manifest frame positions {invalid} are outside [0, {len(frames) - 1}]."
        )
    return tuple(
        resolve_manifest_path(frames[index]["image_path"], manifest_path)
        for index in indices
    )


def _select_cache_frames(
    payload: dict[str, Any],
    *,
    start: int,
    stride: int,
    count: int,
) -> dict[str, Any]:
    """Apply the same deterministic frame selection to a V0 cache replay."""

    frame_indices = payload.get("frame_indices")
    if frame_indices is None:
        total = _cache_frame_count(payload)
        frame_indices = list(range(total))
    total = len(frame_indices)
    selected_positions = _selected_positions(total, start, stride, count)
    if not selected_positions:
        raise ValueError("Frame selection produced no cache frames.")
    if len(selected_positions) == total and selected_positions == list(range(total)):
        return payload

    selected = dict(payload)
    frame_fields = (
        "frame_indices",
        "image_paths",
        "stream_images",
        "images",
        "baseline_world_points",
        "baseline_world_confidence",
        "world_points",
        "confidence",
        "tracking_masks_stream",
        "tracking_scores",
        "quality",
    )
    for name in frame_fields:
        if name not in selected:
            continue
        value = selected[name]
        try:
            if len(value) != total:
                continue
            if isinstance(value, (list, tuple)):
                selected[name] = type(value)(
                    value[index] for index in selected_positions
                )
            elif hasattr(value, "__getitem__"):
                selected[name] = value[selected_positions]
        except (TypeError, IndexError, KeyError):
            continue
    return selected


def _cache_frame_count(payload: dict[str, Any]) -> int:
    for name in (
        "baseline_world_points",
        "world_points",
        "tracking_masks_stream",
        "stream_images",
        "images",
    ):
        value = payload.get(name)
        if value is not None:
            return int(len(value))
    raise ValueError("V0 cache does not expose a frame-indexed field.")


def _select_image_paths(
    paths: tuple[Path, ...],
    *,
    start: int,
    stride: int,
    count: int,
) -> tuple[Path, ...]:
    """Select a deterministic subsequence from already sorted RGB inputs."""

    start = int(start)
    stride = int(stride)
    count = int(count)
    positions = _selected_positions(len(paths), start, stride, count)
    return tuple(paths[index] for index in positions)


def _selected_positions(
    total: int,
    start: int,
    stride: int,
    count: int,
) -> list[int]:
    start = int(start)
    stride = int(stride)
    count = int(count)
    if start < 0:
        raise ValueError("--frame-start must be non-negative.")
    if stride < 1:
        raise ValueError("--frame-stride must be positive.")
    if count < 0:
        raise ValueError("--frame-count must be non-negative; use 0 for all.")
    positions = list(range(start, total, stride))
    return positions[:count] if count else positions


def _normalize_prompts(values: list[str] | None) -> tuple[str, ...]:
    if not values:
        return ()
    output = []
    for value in values:
        output.extend(piece.strip() for piece in value.split(",") if piece.strip())
    return tuple(output)


def _parse_size(value: str) -> tuple[int, int]:
    pieces = value.lower().replace("x", ",").split(",")
    if len(pieces) != 2:
        raise argparse.ArgumentTypeError("Expected SIZE as H,W or HxW.")
    try:
        height, width = (int(piece) for piece in pieces)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("SIZE values must be integers.") from exc
    if height <= 0 or width <= 0:
        raise argparse.ArgumentTypeError("SIZE values must be positive.")
    return height, width


if __name__ == "__main__":
    main()
