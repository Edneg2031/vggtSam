#!/usr/bin/env python3
"""Cache one full SAM3.1 72x72 descriptor map per V9.7 frame."""

from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import hashlib
from pathlib import Path
from typing import Any

import torch

from streaming_couping.scripts.run_v97_dense_descriptor_causality import (
    load_v97_config,
)
from streaming_couping.src.backbones.sam3_intermediate import (
    SAM3IntermediateAdapter,
    load_sam3_image_model,
)
from streaming_couping.src.config import load_config
from streaming_couping.src.learned_pose.cache import (
    _extract_sam_tokens_batched,
    cache_path,
    load_feature_cache,
)
from streaming_couping.src.learned_pose.config import load_learned_pose_config
from streaming_couping.src.v90_explicit_matcher import (
    canonicalize_descriptor_channels,
    sample_stream_patch_descriptors,
)
from streaming_couping.src.v96_dense_grid_decoder import dense_grid_normalized


DENSE_CACHE_VERSION = 2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="streaming_couping/configs/v97_dense_descriptor_causality.yaml",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--dense-cache-path", default=None)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    config = load_v97_config(args.config)
    if args.dense_cache_path:
        config = replace(
            config,
            dense_cache_path=Path(args.dense_cache_path).expanduser().resolve(),
        )
    path = build_dense_descriptor_cache(
        config,
        device=args.device or config.cache_device,
        rebuild=bool(args.rebuild),
    )
    print(f"V9.7 dense descriptor cache={path}")


def build_dense_descriptor_cache(config, *, device: str, rebuild: bool) -> Path:
    learned = load_learned_pose_config(config.data_config)
    clip = next(
        (value for value in learned.clips if value.name == config.clip_name), None
    )
    if clip is None:
        raise ValueError(f"V9.7 clip={config.clip_name!r} is not configured.")
    source_path = cache_path(learned, clip)
    if not source_path.is_file():
        raise FileNotFoundError(
            "V9.7 requires the retained V9.2 observation cache: "
            f"{source_path}"
        )
    expected_provenance = _source_provenance(source_path)
    output = config.dense_cache_path
    if output.is_file() and not rebuild:
        cached = _torch_load(output)
        if _cache_valid(cached, config=config, provenance=expected_provenance):
            print("V9.7 reusing provenance-compatible dense descriptor cache")
            return output
        print(f"V9.7 rebuilding stale dense descriptor cache={output}")

    payload = load_feature_cache(source_path)
    required = {
        "frame_indices",
        "image_paths",
        "image_size",
        "token_levels",
        "patch_start_idx",
        "patch_shape",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"V9.7 source cache lacks fields={sorted(missing)}.")
    frames = tuple(int(value) for value in payload["frame_indices"])
    if frames != config.expected_frames:
        raise ValueError("V9.7 source cache does not contain the locked 30 frames.")
    grid = dense_grid_normalized(config.grid_size).float()
    image_size = tuple(int(value) for value in payload["image_size"])
    grid_for_sampler = grid.reshape(1, 1, -1, 2).expand(len(frames), -1, -1, -1)
    stream = sample_stream_patch_descriptors(
        torch.as_tensor(payload["token_levels"]).cpu(),
        patch_start_idx=int(payload["patch_start_idx"]),
        patch_shape=tuple(int(value) for value in payload["patch_shape"]),
        local_uv_normalized=grid_for_sampler,
    )[:, 0]
    image_paths = [str(value) for value in payload["image_paths"]]
    del payload

    recovery = load_config(learned.recovery_config)
    model = load_sam3_image_model(
        repo_path=recovery.sam3_repo,
        checkpoint_path=recovery.sam3_checkpoint,
        device=device,
        version=recovery.sam3_version,
        enable_segmentation=False,
        enable_inst_interactivity=False,
    )
    adapter = SAM3IntermediateAdapter(
        model,
        device=device,
        resolution=learned.features.sam_resolution,
        source=learned.features.sam_source,
        text_conditioning="none",
        token_grid=config.grid_size,
    )
    tokens = _extract_sam_tokens_batched(
        adapter,
        image_paths,
        token_grid=config.grid_size,
        batch_size=config.cache_batch_size,
    )
    sam = tokens.reshape(len(frames), config.grid_size[0] * config.grid_size[1], -1)
    if sam.shape[:2] != stream.shape[:2]:
        raise ValueError(
            "V9.7 SAM/Stream dense grids disagree: "
            f"{tuple(sam.shape)} vs {tuple(stream.shape)}."
        )
    raw_sam_dim = int(sam.shape[-1])
    raw_stream_dim = int(stream.shape[-1])
    # Keep the cache and training memory bounded while preserving the exact
    # parameter-free channel canonicalization used by every matcher control.
    sam = canonicalize_descriptor_channels(
        sam.float(), int(config.matcher.canonical_dim)
    ).contiguous()
    stream = canonicalize_descriptor_channels(
        stream.float(), int(config.matcher.canonical_dim)
    ).contiguous()
    cached: dict[str, Any] = {
        "cache_version": DENSE_CACHE_VERSION,
        "complete": True,
        "source_provenance": expected_provenance,
        "frame_indices": list(frames),
        "image_size": list(image_size),
        "grid_size": list(config.grid_size),
        "grid_uv_normalized": grid,
        "sam_dense_features": sam.to(torch.float16),
        "stream_dense_features": stream.to(torch.float16),
        "sam_source": learned.features.sam_source,
        "sam_version": recovery.sam3_version,
        "sam_text_conditioning": "none",
        "sam_feature_dim": int(sam.shape[-1]),
        "stream_feature_dim": int(stream.shape[-1]),
        "sam_raw_feature_dim": raw_sam_dim,
        "stream_raw_feature_dim": raw_stream_dim,
        "channel_canonicalization": "parameter_free_adaptive_pool_or_pad",
        "image_paths_sha256": _string_digest(image_paths),
        "uses_pose_loss": False,
        "contains_pose_model": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cached, output)
    print(
        "V9.7 cached dense maps "
        f"frames={len(frames)} grid={config.grid_size[0]}x{config.grid_size[1]} "
        f"sam_dim={raw_sam_dim}->{sam.shape[-1]} "
        f"stream_dim={raw_stream_dim}->{stream.shape[-1]}"
    )
    del adapter, model, tokens, sam, stream, cached
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output


def _cache_valid(value: Any, *, config, provenance: dict[str, Any]) -> bool:
    if not isinstance(value, dict):
        return False
    required = {
        "cache_version",
        "complete",
        "source_provenance",
        "frame_indices",
        "grid_size",
        "grid_uv_normalized",
        "sam_dense_features",
        "stream_dense_features",
    }
    if required - set(value):
        return False
    grid_tokens = config.grid_size[0] * config.grid_size[1]
    sam = torch.as_tensor(value["sam_dense_features"])
    stream = torch.as_tensor(value["stream_dense_features"])
    return bool(
        int(value["cache_version"]) == DENSE_CACHE_VERSION
        and value["complete"]
        and value["source_provenance"] == provenance
        and tuple(int(item) for item in value["frame_indices"])
        == config.expected_frames
        and tuple(int(item) for item in value["grid_size"]) == config.grid_size
        and tuple(sam.shape[:2]) == (len(config.expected_frames), grid_tokens)
        and tuple(stream.shape[:2]) == (len(config.expected_frames), grid_tokens)
        and int(sam.shape[-1]) == int(config.matcher.canonical_dim)
        and int(stream.shape[-1]) == int(config.matcher.canonical_dim)
        and bool(torch.isfinite(sam.float()).all())
        and bool(torch.isfinite(stream.float()).all())
    )


def _source_provenance(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _string_digest(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


if __name__ == "__main__":
    main()
