#!/usr/bin/env python3
"""Build a semantic-instance map from RGB prompts or a frozen V0 cache.

The model path uses the current StreamVGGT and SAM3.1 wrappers.  The cache path
is useful for replaying a V0 run without rerunning either model.  Both paths
converge on the same backend-neutral ``SemanticMapPipeline``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from streaming_couping.src.config import load_config
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
        },
    )


def _run_from_rgb(
    args: argparse.Namespace,
    prompts: tuple[str, ...],
    mapper: SemanticMapBuilder,
):
    if not prompts:
        raise ValueError("--prompts is required when --cache is not used.")
    image_paths = _expand_image_paths(args.frames)
    image_paths = _select_image_paths(
        image_paths,
        start=args.frame_start,
        stride=args.frame_stride,
        count=args.frame_count,
    )
    if not image_paths:
        raise ValueError("No RGB images were found in --frames.")
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
    if start < 0:
        raise ValueError("--frame-start must be non-negative.")
    if stride < 1:
        raise ValueError("--frame-stride must be positive.")
    if count < 0:
        raise ValueError("--frame-count must be non-negative; use 0 for all.")
    selected = paths[start::stride]
    if count:
        selected = selected[:count]
    return tuple(selected)


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
