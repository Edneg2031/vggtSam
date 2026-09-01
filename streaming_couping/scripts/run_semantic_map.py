#!/usr/bin/env python3
"""Build a semantic-instance map from RGB prompts or a frozen V0 cache.

The RGB path accepts either an isolated HorizonStream geometry cache or the
legacy in-process StreamVGGT wrapper, then runs SAM3.1.  The V0 cache path
replays both frozen providers.  Every path converges on the same
backend-neutral ``SemanticMapPipeline``.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from pathlib import Path
import tempfile
from typing import Any

import torch

from streaming_couping.src.config import load_config
from streaming_couping.src.horizonstream_cache import (
    load_horizonstream_cache,
    materialize_horizonstream_rgb,
    validate_horizonstream_cache,
)
from streaming_couping.src.rgb_inputs import (
    IMAGE_SUFFIXES,
    expand_image_paths,
    expand_manifest_paths,
    resolve_rgb_inputs,
    select_image_paths,
    selected_positions,
)
from streaming_couping.src.semantic_mapping.adapters import (
    HorizonStreamGeometryCacheAdapter,
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
            max_scene_voxels=args.max_scene_voxels,
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
        f"scene_voxels={summary['scene_voxel_count']} "
        f"voxels={summary['voxel_count']} "
        f"objects={summary['instance_count']}"
    )
    print(f"semantic_map={summary['outputs']['artifact']}")
    print(f"scene_rgb_ply={summary['outputs']['scene_rgb_ply']}")
    print(f"scene_semantic_ply={summary['outputs']['scene_semantic_ply']}")
    print(f"semantic_ply={summary['outputs']['semantic_ply']}")
    print(f"object_plys={summary['outputs']['objects_dir']}")


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
    selection = resolve_rgb_inputs(
        frames=args.frames,
        manifest=args.manifest,
        scene_id=args.scene_id,
        dataset_frame_indices=args.dataset_frame_indices,
        start=args.frame_start,
        stride=args.frame_stride,
        count=args.frame_count,
    )
    image_paths = selection.image_paths
    if not image_paths:
        raise ValueError("No RGB images were selected from the requested input.")
    overrides = {}
    if args.sam_device:
        overrides["sam3_device"] = args.sam_device
    if args.geometry_device:
        overrides["geometry_device"] = args.geometry_device
    recovery = load_config(args.recovery_config, overrides)

    geometry_payload = None
    pipeline_image_paths = image_paths
    if args.geometry_cache is not None:
        geometry_payload = load_horizonstream_cache(args.geometry_cache)
        validate_horizonstream_cache(
            geometry_payload,
            expected_image_paths=image_paths,
        )
        geometry = HorizonStreamGeometryCacheAdapter(geometry_payload)
        geometry_backend = geometry.backend_name
        geometry_scale_type = "metric"
        segmentation_output_size = geometry.image_size
    else:
        from streaming_couping.src.backbones.streamvggt_wrapper import (
            StreamVGGTWrapper,
            materialize_streamvggt_rgb,
        )

        stream = StreamVGGTWrapper(
            repo_path=recovery.streamvggt_repo,
            checkpoint_path=recovery.streamvggt_checkpoint,
            device=recovery.geometry_device,
            image_mode=recovery.image_mode,
            streaming_cache=recovery.streaming_cache,
        ).load()
        geometry = StreamVGGTGeometryAdapter(
            stream,
            scale_type=args.scale_type,
        )
        geometry_backend = geometry.backend_name
        geometry_scale_type = args.scale_type

    with ExitStack() as stack:
        if geometry_payload is None:
            # StreamVGGT's pointmap is indexed on its processed image grid.
            # SAM must consume those same processed pixels; resizing a
            # 256x384 mask onto the pointmap grid is not a valid alignment.
            directory = stack.enter_context(
                tempfile.TemporaryDirectory(prefix="streamvggt_sam_rgb_")
            )
            pipeline_image_paths, processed_size = materialize_streamvggt_rgb(
                image_paths,
                directory,
                image_mode=recovery.image_mode,
            )
            segmentation_output_size = tuple(args.output_size or processed_size)
        else:
            # SAM must see the exact center-cropped pixels used for cached
            # HorizonStream geometry.
            directory = stack.enter_context(
                tempfile.TemporaryDirectory(prefix="horizonstream_sam_rgb_")
            )
            pipeline_image_paths = materialize_horizonstream_rgb(
                geometry_payload,
                directory,
            )
            segmentation_output_size = geometry.image_size

        print(
            "semantic map runtime "
            f"frames={len(image_paths)} "
            f"geometry_backend={geometry_backend} "
            f"model_input_size={segmentation_output_size} "
            f"sam_device={recovery.sam3_device} "
            f"sam_grounding_batch={args.sam_grounding_batch_size} "
            f"sam_video_cpu_offload={int(args.sam_offload_video_to_cpu)}"
        )
        from streaming_couping.src.backbones.sam3_wrapper import SAM3Wrapper

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
            grounding_batch_size=args.sam_grounding_batch_size,
            offload_video_to_cpu=args.sam_offload_video_to_cpu,
        ).load()
        segmentation = SAM31SegmentationAdapter(
            sam,
            output_size=segmentation_output_size,
            max_objects_per_prompt=recovery.sam3_max_num_objects,
            max_total_objects=args.max_objects,
            min_birth_pixels=recovery.min_pixels,
        )
        pipeline = SemanticMapPipeline(
            geometry=geometry,
            segmentation=segmentation,
            mapper=mapper,
        )
        metadata = {
            "input_mode": "rgb_and_text_prompts",
            "scale_type": geometry_scale_type,
            "recovery_config": str(Path(args.recovery_config).resolve()),
            "geometry_cache": (
                None
                if args.geometry_cache is None
                else str(args.geometry_cache.expanduser().resolve())
            ),
            "source_image_paths": [str(path) for path in image_paths],
            "source_positions": list(selection.source_positions),
            "source_count": int(selection.source_count),
            "frame_selection": {
                "start": int(args.frame_start),
                "stride": int(args.frame_stride),
                "count": int(args.frame_count),
            },
            "model_input_size": list(segmentation_output_size),
            "shared_geometry_segmentation_pixels": True,
            **selection.metadata,
        }
        return pipeline.run(
            pipeline_image_paths,
            prompts=prompts,
            metadata=metadata,
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
        "--geometry-cache",
        type=Path,
        default=None,
        help=(
            "HorizonStream geometry cache for RGB mode. Geometry is not rerun "
            "and StreamVGGT is not loaded."
        ),
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
    parser.add_argument(
        "--sam-grounding-batch-size",
        type=int,
        default=4,
        help="Frames per SAM3.1 grounding batch; lower values reduce peak VRAM.",
    )
    parser.add_argument(
        "--sam-offload-video-to-cpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep decoded SAM video frames in CPU memory until they are needed.",
    )
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
    parser.add_argument(
        "--max-scene-voxels",
        type=int,
        default=1_000_000,
        help="Maximum full-scene voxels retained in exported artifacts.",
    )
    args = parser.parse_args()
    if args.cache is not None and args.geometry_cache is not None:
        parser.error("--geometry-cache cannot be combined with the frozen V0 --cache.")
    return args


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
    return expand_image_paths(values)


def _expand_manifest_paths(
    manifest_path: Path,
    *,
    scene_id: str | None,
    frame_indices: list[int] | None,
) -> tuple[Path, ...]:
    paths, _, _ = expand_manifest_paths(
        manifest_path,
        scene_id=scene_id,
        frame_indices=frame_indices,
    )
    return paths


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
    return select_image_paths(paths, start=start, stride=stride, count=count)


def _selected_positions(
    total: int,
    start: int,
    stride: int,
    count: int,
) -> list[int]:
    return selected_positions(total, start, stride, count)


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
