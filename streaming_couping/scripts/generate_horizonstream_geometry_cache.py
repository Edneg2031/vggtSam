#!/usr/bin/env python3
"""Run HorizonStream once and save backend-neutral metric geometry."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import copy
import gc
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch
from PIL import Image
import yaml

from streaming_couping.src.horizonstream_cache import (
    HORIZONSTREAM_CACHE_SCHEMA,
    HORIZONSTREAM_CACHE_VERSION,
    horizonstream_cache_matches,
    image_file_signatures,
    load_horizonstream_cache,
    normalize_horizonstream_confidence,
    validate_horizonstream_cache,
)
from streaming_couping.src.rgb_inputs import resolve_rgb_inputs


GENERATOR_REVISION = "horizonstream_semantic_geometry_generator_r1"


def main() -> None:
    args = _parse_args()
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
        raise ValueError("Frame selection produced no RGB images.")

    repo_path = args.repo_path.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    config_path = (
        args.config.expanduser().resolve()
        if args.config is not None
        else repo_path / "configs" / "horizonstream_infer.yaml"
    )
    _require_file(
        repo_path / "horizonstream" / "core" / "model.py",
        "HorizonStream source",
    )
    _require_file(checkpoint, "HorizonStream checkpoint")
    _require_file(config_path, "HorizonStream config")

    settings = _request_settings(args, repo_path, checkpoint, config_path)
    output_cache = args.output_cache.expanduser().resolve()
    if args.reuse_if_valid and output_cache.is_file():
        try:
            existing = load_horizonstream_cache(output_cache)
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            print(f"HorizonStream cache is stale or invalid; regenerating: {exc}")
        else:
            if horizonstream_cache_matches(
                existing,
                image_paths=image_paths,
                checkpoint=checkpoint,
                settings=settings,
            ):
                print(
                    "HorizonStream geometry cache reused "
                    f"frames={len(image_paths)} path={output_cache}"
                )
                return
            print("HorizonStream cache request changed; regenerating.")

    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))
    from horizonstream.core.model import HorizonStreamModel
    from horizonstream.data.dataloader import HorizonStreamDataLoader
    from horizonstream.runtime.motion_averaging import (
        compute_motion_averaged_camera_maps,
    )
    from horizonstream.utils.vendor.models.components.utils.pose_enc import (
        pose_encoding_to_extri_intri,
    )

    with config_path.open("r", encoding="utf-8") as handle:
        upstream_config = yaml.safe_load(handle) or {}
    model_config = copy.deepcopy(upstream_config.get("model", {}))
    model_config["checkpoint"] = str(checkpoint)
    model_config["strict_load"] = bool(args.strict_load)
    horizon_config = model_config.setdefault("horizonstream_cfg", {})
    horizon_config["enable_metric_readout_token"] = True

    data_config = copy.deepcopy(upstream_config.get("data", {}))
    data_config.update(
        {
            "format": "image_list",
            "image_paths": [str(path) for path in image_paths],
            "image_scene_name": str(args.scene_id or "semantic_map"),
            "size": int(args.image_size),
            "crop": bool(args.crop),
            "patch_size": int(args.patch_size),
            "camera_preprocess": False,
            "max_frames": None,
        }
    )

    print(
        "HorizonStream geometry inference "
        f"frames={len(image_paths)} device={args.device} "
        f"window={args.window_size} sliding={args.sliding_size} "
        f"precision={args.precision}"
    )
    model = HorizonStreamModel(model_config).to(args.device).eval()
    print("HorizonStream model ready")
    loader = HorizonStreamDataLoader(data_config)
    sequences = iter(loader)
    sequence = next(sequences)
    try:
        next(sequences)
    except StopIteration:
        pass
    else:
        raise ValueError("Geometry cache generation expects exactly one image sequence.")

    images = sequence.images.detach().cpu()
    if images.ndim != 5 or images.shape[0] != 1:
        raise ValueError(
            "HorizonStream image loader must return shape [1,S,3,H,W], got "
            f"{tuple(images.shape)}."
        )
    _, frame_count, channels, height, width = images.shape
    if channels != 3 or frame_count != len(image_paths):
        raise ValueError(
            "HorizonStream preprocessed images do not match selected RGB inputs."
        )

    chunks = _chunk_schedule(frame_count, args.window_size, args.sliding_size)
    state = model.build_sequence_state()
    depth_chunks: list[torch.Tensor] = []
    confidence_chunks: list[torch.Tensor] = []
    camera_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for chunk_index, (start, end) in enumerate(chunks):
            print(
                "HorizonStream chunk "
                f"{chunk_index + 1}/{len(chunks)} frames=[{start},{end})"
            )
            chunk_images = images[:, start:end].to(args.device, non_blocking=True)
            current_window_size = end - start if chunk_index == 0 else args.window_size
            with _autocast_context(args.device, args.precision):
                outputs = model.forward_chunk(
                    chunk_images,
                    window_size=current_window_size,
                    chunk_idx=chunk_index,
                    state=state,
                )
            depth_chunks.append(outputs["depth"][0].detach().float().cpu())
            confidence_chunks.append(
                outputs["depth_conf"][0].detach().float().cpu()
            )
            camera_chunks.append(
                outputs["chunk_cam_map"].detach().float().cpu()
            )
            model.advance_sequence_state(
                state,
                is_last_chunk=chunk_index == len(chunks) - 1,
            )
            del outputs, chunk_images
            if str(args.device).startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()

    depth = torch.cat(depth_chunks, dim=0)
    raw_confidence = torch.cat(confidence_chunks, dim=0)
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if raw_confidence.ndim == 4 and raw_confidence.shape[-1] == 1:
        raw_confidence = raw_confidence[..., 0]
    if tuple(depth.shape) != (frame_count, height, width):
        raise ValueError(
            "HorizonStream depth output does not match preprocessed images: "
            f"depth={tuple(depth.shape)} images={(frame_count, height, width)}."
        )
    confidence, confidence_q05, confidence_q95 = (
        normalize_horizonstream_confidence(raw_confidence)
    )

    motion_maps = compute_motion_averaged_camera_maps(
        camera_chunks,
        frames_num=frame_count,
        window_size=int(args.window_size),
        dtype=torch.float32,
        enable_offline=False,
    )
    online_camera_map = motion_maps["online_cam_map"].detach().float().cpu()
    world_to_camera, intrinsics = pose_encoding_to_extri_intri(
        online_camera_map,
        image_size_hw=(height, width),
    )
    world_to_camera = world_to_camera[0].detach().float().cpu()
    intrinsics = intrinsics[0].detach().float().cpu()

    processed_rgb = (
        images[0]
        .permute(0, 2, 3, 1)
        .mul(255.0)
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
        .contiguous()
    )
    source_sizes = []
    for path in image_paths:
        with Image.open(path) as image:
            source_sizes.append((int(image.height), int(image.width)))

    payload: dict[str, Any] = {
        "schema": HORIZONSTREAM_CACHE_SCHEMA,
        "schema_version": HORIZONSTREAM_CACHE_VERSION,
        "backend": "horizonstream",
        "image_paths": [str(path.resolve()) for path in image_paths],
        "frame_ids": list(range(frame_count)),
        "source_positions": list(selection.source_positions),
        "source_count": int(selection.source_count),
        "depth": depth.contiguous(),
        "confidence": confidence.contiguous(),
        "world_to_camera": world_to_camera.contiguous(),
        "intrinsics": intrinsics.contiguous(),
        "processed_rgb": processed_rgb,
        "processed_size": (int(height), int(width)),
        "source_sizes": source_sizes,
        "scale_type": "metric",
        "pose_source": "online_motion_averaged",
        "pose_convention": "opencv_world_to_camera",
        "confidence_normalization": "per_frame_q05_q95",
        "confidence_raw_q05": confidence_q05,
        "confidence_raw_q95": confidence_q95,
        "window_size": int(args.window_size),
        "sliding_size": int(args.sliding_size),
        "checkpoint": str(checkpoint),
        "preprocess": {
            "image_size": int(args.image_size),
            "crop": bool(args.crop),
            "patch_size": int(args.patch_size),
        },
        "input": {
            **selection.metadata,
            "frame_selection": {
                "start": int(args.frame_start),
                "stride": int(args.frame_stride),
                "count": int(args.frame_count),
            },
        },
        "request": {
            "checkpoint": str(checkpoint),
            "settings": settings,
            "image_signatures": image_file_signatures(image_paths),
        },
    }
    validate_horizonstream_cache(payload, expected_image_paths=image_paths)
    output_cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_cache.with_name(f".{output_cache.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        temporary.replace(output_cache)
    finally:
        if temporary.exists():
            temporary.unlink()

    del model, state, images
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(
        "HorizonStream geometry cache completed "
        f"frames={frame_count} size={(height, width)} path={output_cache}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--frames", nargs="+", type=Path)
    source.add_argument("--manifest", type=Path)
    parser.add_argument("--scene-id", default=None)
    parser.add_argument("--dataset-frame-indices", nargs="+", type=int, default=None)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--frame-count", type=int, default=0)
    parser.add_argument("--repo-path", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-cache", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--sliding-size", type=int, default=21)
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument(
        "--crop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the upstream center crop and cache the aligned processed RGB.",
    )
    parser.add_argument(
        "--precision",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    parser.add_argument(
        "--strict-load",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--reuse-if-valid",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()
    for name in ("window_size", "sliding_size", "image_size", "patch_size"):
        if int(getattr(args, name)) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    return args


def _chunk_schedule(
    frame_count: int,
    window_size: int,
    sliding_size: int,
) -> list[tuple[int, int]]:
    if frame_count <= 0:
        return []
    if frame_count <= window_size:
        return [(0, frame_count)]
    chunks = [(0, window_size)]
    start = window_size
    while start < frame_count:
        end = min(start + sliding_size, frame_count)
        chunks.append((start, end))
        start = end
    return chunks


def _autocast_context(device: str, precision: str):
    if not str(device).startswith("cuda") or precision == "float32":
        return nullcontext()
    dtype = torch.float16 if precision == "float16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _request_settings(
    args: argparse.Namespace,
    repo_path: Path,
    checkpoint: Path,
    config_path: Path,
) -> dict[str, Any]:
    stat = checkpoint.stat()
    return {
        "window_size": int(args.window_size),
        "sliding_size": int(args.sliding_size),
        "image_size": int(args.image_size),
        "patch_size": int(args.patch_size),
        "crop": bool(args.crop),
        "precision": str(args.precision),
        "strict_load": bool(args.strict_load),
        "metric_readout": True,
        "pose_source": "online_motion_averaged",
        "checkpoint_size": int(stat.st_size),
        "checkpoint_mtime_ns": int(stat.st_mtime_ns),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "repo_revision": _repo_revision(repo_path),
        "generator_revision": GENERATOR_REVISION,
    }


def _repo_revision(repo_path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} does not exist: {path}")


if __name__ == "__main__":
    main()
