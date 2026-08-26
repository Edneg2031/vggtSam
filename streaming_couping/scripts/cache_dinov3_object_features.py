#!/usr/bin/env python3
"""Cache frozen DINOv3 object features for the existing StreamVGGT/SAM cache."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from streaming_couping.src.dinov3_object_features import (
    DinoV3DenseEncoder,
    DinoV3FeatureConfig,
    aggregate_persistent_features,
    masked_mean_pool,
    resolve_dinov3_checkpoint,
    shuffled_persistent_features,
)
from streaming_couping.src.learned_pose.cache import cache_path, load_feature_cache
from streaming_couping.src.learned_pose.config import load_learned_pose_config
from streaming_couping.src.storage import expand_storage_path


RAW_CACHE_VARIANT = "sam31_online_forward"


def main() -> None:
    args = _parse_args()
    data = load_learned_pose_config(args.config)
    checkpoint = resolve_dinov3_checkpoint(
        args.checkpoint,
        root=args.dinov3_root,
        preferred_variant=args.preferred_variant,
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else expand_storage_path(
            "${VGGT_SAM_STORAGE_ROOT}/outputs/streaming_couping_dinov3_object_features",
            base=Path(args.config).expanduser().resolve().parent,
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    device = str(args.device)
    print("DINOv3 MULTI-LAYER DENSE OBJECT FEATURE CACHE")
    print(f"checkpoint={checkpoint}")
    print(f"device={device} dtype={args.dtype} ema_beta={args.ema_beta}")
    print("models=StreamVGGT and SAM3 caches are reused; no GT is read")

    encoder = DinoV3DenseEncoder(
        DinoV3FeatureConfig(
            checkpoint=checkpoint,
            root=Path(args.dinov3_root),
            preferred_variant=args.preferred_variant,
            device=device,
            dtype=args.dtype,
            ema_beta=float(args.ema_beta),
        )
    )
    for clip in data.clips:
        cache_value = cache_path(data, clip)
        payload = load_feature_cache(cache_value)
        output_path = output_dir / f"{clip.name}.pt"
        if output_path.is_file() and not args.overwrite:
            print(f"  skip clip={clip.name} cache={output_path}")
            continue
        images = payload.get("stream_images")
        if images is None:
            raise ValueError(f"Frozen cache lacks stream_images: {cache_value}")
        masks = _load_masks(payload, args.raw_cache_variant)
        if int(images.shape[0]) != int(masks.shape[0]):
            raise ValueError(f"Image/mask frame count mismatch for {clip.name}.")
        single_rows: list[torch.Tensor] = []
        valid_rows: list[torch.Tensor] = []
        dense_rows_by_layer: dict[int, list[torch.Tensor]] = {}
        dense_layer_ids: tuple[int, ...] | None = None
        metadata: dict[str, Any] | None = None
        for start in range(0, int(images.shape[0]), max(1, int(args.batch_size))):
            stop = min(int(images.shape[0]), start + max(1, int(args.batch_size)))
            dense_by_layer, current_meta = encoder.encode_dense_layers(
                images[start:stop]
            )
            current_layer_ids = tuple(
                int(value) for value in current_meta.get("layer_ids", ())
            )
            if not current_layer_ids:
                raise RuntimeError(
                    "DINOv3 multi-layer encoder returned no layer IDs."
                )
            if dense_layer_ids is None:
                dense_layer_ids = current_layer_ids
            elif dense_layer_ids != current_layer_ids:
                raise RuntimeError(
                    "DINOv3 layer IDs changed between cache batches: "
                    f"first={dense_layer_ids} current={current_layer_ids}"
                )
            for layer_id in dense_layer_ids:
                if layer_id not in dense_by_layer:
                    raise RuntimeError(
                        f"DINOv3 layer {layer_id} is missing from encoder output."
                    )
                dense_rows_by_layer.setdefault(layer_id, []).append(
                    dense_by_layer[layer_id].to(torch.float16).cpu()
                )
            final_layer = dense_by_layer[dense_layer_ids[-1]]
            pooled, valid = masked_mean_pool(
                final_layer,
                masks[start:stop],
                normalize=True,
            )
            single_rows.append(pooled.cpu())
            valid_rows.append(valid.cpu())
            metadata = current_meta
        single = torch.cat(single_rows, dim=0)
        valid = torch.cat(valid_rows, dim=0)
        if dense_layer_ids is None or not dense_rows_by_layer:
            raise RuntimeError(f"No DINOv3 dense layers were cached for {clip.name}.")
        dense_features_by_layer = {
            str(layer_id): torch.cat(rows, dim=0)
            for layer_id, rows in sorted(dense_rows_by_layer.items())
        }
        # ``dense_features`` remains the final-layer compatibility field used
        # by the earlier patch-retrieval diagnostic and existing consumers.
        dense_features = dense_features_by_layer[str(dense_layer_ids[-1])]
        track_ids = torch.as_tensor(payload["sam_track_ids"], dtype=torch.long)
        persistent, persistent_valid = aggregate_persistent_features(
            single,
            valid,
            beta=float(args.ema_beta),
            track_ids=track_ids,
        )
        shuffled, shuffled_valid, permutation = shuffled_persistent_features(
            single,
            valid,
            beta=float(args.ema_beta),
            seed=int(args.shuffle_seed),
        )
        result = {
            "schema": 3,
            "revision": "dinov3_persistent_object_feature_cache_r3_multilayer_dense",
            "clip": clip.name,
            "scene_id": str(payload["scene_id"]),
            "frame_indices": list(payload["frame_indices"]),
            "cache_path": str(cache_value),
            "raw_cache_variant": str(args.raw_cache_variant),
            "checkpoint": str(checkpoint),
            "feature_dim": int(single.shape[-1]),
            "dense_features": dense_features,
            "dense_feature_shape": list(dense_features.shape[1:]),
            "dense_feature_dtype": "float16",
            "dense_features_by_layer": dense_features_by_layer,
            "dense_layer_ids": [int(value) for value in dense_layer_ids],
            "dense_layer_metadata": {
                str(layer_id): {
                    "shape": list(dense_features_by_layer[str(layer_id)].shape[1:]),
                    "dtype": "float16",
                }
                for layer_id in dense_layer_ids
            },
            "ema_beta": float(args.ema_beta),
            "track_ids": track_ids,
            "track_prompts": list(payload["sam_track_prompts"]),
            "single_features": single,
            "single_valid": valid,
            "persistent_features": persistent,
            "persistent_valid": persistent_valid,
            "shuffled_persistent_features": shuffled,
            "shuffled_persistent_valid": shuffled_valid,
            "shuffle_permutation": permutation,
            "mask_shape": list(masks.shape),
            "metadata": metadata or {},
        }
        torch.save(result, output_path)
        print(
            f"  clip={clip.name} frames={single.shape[0]} slots={single.shape[1]} "
            f"dim={single.shape[2]} layers={','.join(str(value) for value in dense_layer_ids)} "
            f"valid={int(valid.sum())} output={output_path}"
        )
    print(f"DINOv3 feature cache completed output_dir={output_dir}")


def _load_masks(payload: dict[str, Any], variant: str) -> torch.Tensor:
    values = payload.get("tracking_variant_masks_stream", {})
    if isinstance(values, dict) and variant in values:
        return torch.as_tensor(values[variant]).bool().cpu()
    fallback = payload.get("tracking_masks_stream")
    if fallback is None:
        raise ValueError(f"Cache lacks tracking mask variant {variant!r}.")
    return torch.as_tensor(fallback).bool().cpu()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="streaming_couping/configs/v0_baseline.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--dinov3-root",
        default="/home/bod/86Nas/95_data_bak/FoundationModels/dinov3",
    )
    parser.add_argument("--checkpoint")
    parser.add_argument("--preferred-variant", default="dinov3-vitl16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--ema-beta", type=float, default=0.90)
    parser.add_argument("--shuffle-seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--raw-cache-variant", default=RAW_CACHE_VARIANT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
