#!/usr/bin/env python3
"""Prepare frozen StreamVGGT geometry features for a scene-disjoint pilot.

The script performs inference-only candidate generation.  GT pointmaps are
opened only after the frozen StreamVGGT features and raw pointmaps have been
materialized in memory.  Each episode gets its own reference-frame Sim(3), so
episodes from different scenes can be trained together without mixing gauges.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any, Sequence

import torch
import yaml

from streaming_couping.src.backbones.streamvggt_latent import (
    StreamVGGTLatentAdapter,
    load_streamvggt_latent_model,
)
from streaming_couping.src.backbones.streamvggt_parallel import (
    LayerShardedStreamVGGT,
    assert_processed_key_cache_equivalence,
)
from streaming_couping.src.backbones.streamvggt_wrapper import StreamVGGTWrapper
from streaming_couping.src.external_repos import maybe_add_repo_to_path
from streaming_couping.src.pointmap_alignment import (
    _load_gt_pointmaps,
    _paired_limit,
    _robust_similarity,
)
from streaming_couping.src.semantic_map import normalize_confidence
from streaming_couping.src.storage import expand_storage_path
from streaming_couping.src.trust_aware_residual import (
    invert_similarity,
    point_head_patch_features,
)


REVISION = "multiscene_geometry_cache_r1"
EXPECTED_DPT_LAYERS = (4, 11, 17, 23)


def main() -> None:
    args = _parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = _load_config(config_path)
    manifest_path = expand_storage_path(config["manifest"], base=config_path.parent)
    output_dir = expand_storage_path(config["output_dir"], base=config_path.parent)
    manifest = _load_manifest(manifest_path)
    splits = _load_splits(config)
    episodes = _build_episode_specs(manifest, splits, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "episodes").mkdir(parents=True, exist_ok=True)

    print("MULTI-SCENE FROZEN GEOMETRY CACHE PREPARATION")
    print(f"manifest={manifest_path}")
    print(f"output={output_dir}")
    print(
        f"scenes=train={','.join(splits['train'])} "
        f"validation={','.join(splits['validation'])} "
        f"test={','.join(splits['test'])}"
    )
    print(
        f"episodes={len(episodes)} length={int(config['episode_length'])} "
        f"sam=0 training=0 gt_for_candidate_generation=0"
    )

    runtime = _load_runtime(config["v0_config"], config_path.parent)
    devices = tuple(str(value) for value in config.get("devices", ("cuda:0", "cuda:1")))
    maybe_add_repo_to_path(runtime["streamvggt_repo"])
    assert_processed_key_cache_equivalence()
    model = load_streamvggt_latent_model(
        repo_path=runtime["streamvggt_repo"],
        checkpoint_path=runtime["streamvggt_checkpoint"],
        device="cpu",
        strict=True,
    )
    runner = LayerShardedStreamVGGT(
        model,
        devices,
        selected_layer_indices=EXPECTED_DPT_LAYERS,
        amp_dtype=runtime["amp_dtype"],
    )
    image_mode = str(config.get("image_mode", runtime["image_mode"]))
    if image_mode not in {"crop", "pad"}:
        raise ValueError(
            f"Multi-scene geometry image_mode must be 'crop' or 'pad', got {image_mode!r}."
        )
    print(f"image_mode={image_mode}")
    for line in runner.layout_summary():
        print(f"  {line}")
    adapter = StreamVGGTLatentAdapter(
        model,
        device=devices[0],
        image_mode=image_mode,
        dpt_layer_indices=EXPECTED_DPT_LAYERS,
        parallel_runner=runner,
    )

    records = []
    for spec in episodes:
        cache_path = output_dir / "episodes" / f"{spec['episode_id']}.pt"
        if cache_path.is_file() and not args.overwrite:
            print(f"skip existing episode={spec['episode_id']} path={cache_path}")
            records.append({**spec, "cache_path": str(cache_path)})
            continue
        record = _prepare_episode(
            spec,
            cache_path=cache_path,
            manifest_path=manifest_path,
            adapter=adapter,
            runner=runner,
            image_mode=image_mode,
            confidence_threshold=float(config.get("confidence_threshold", 0.30)),
            max_alignment_points=int(config.get("max_alignment_points", 30000)),
        )
        records.append(record)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    index = {
        "schema": 1,
        "revision": REVISION,
        "manifest": str(manifest_path),
        "output_dir": str(output_dir),
        "dpt_layer_indices": list(EXPECTED_DPT_LAYERS),
        "image_mode": image_mode,
        "confidence_threshold": float(config.get("confidence_threshold", 0.30)),
        "episode_length": int(config["episode_length"]),
        "splits": splits,
        "episodes": records,
        "candidate_generation": {
            "streamvggt_frozen": 1,
            "sam_inputs": 0,
            "training": 0,
            "gt_values_used": 0,
        },
    }
    _write_json(output_dir / "index.json", index)
    print(f"cache_index={output_dir / 'index.json'}")
    print(f"cache_result=READY episodes={len(records)}")


def _prepare_episode(
    spec: dict[str, Any],
    *,
    cache_path: Path,
    manifest_path: Path,
    adapter: StreamVGGTLatentAdapter,
    runner: LayerShardedStreamVGGT,
    image_mode: str,
    confidence_threshold: float,
    max_alignment_points: int,
) -> dict[str, Any]:
    image_paths = [Path(value) for value in spec["image_paths"]]
    if not all(path.is_file() for path in image_paths):
        missing = [str(path) for path in image_paths if not path.is_file()]
        raise FileNotFoundError(f"Missing episode images: {missing[:3]}")

    print(
        f"episode={spec['episode_id']} scene={spec['scene_id']} "
        f"frames={len(image_paths)} candidate_generation=RGB+frozen_StreamVGGT"
    )
    with torch.inference_mode():
        output = adapter.extract_from_paths(
            image_paths,
            return_pointmap=True,
            streaming_cache=True,
        )

    aux = output.geometry.aux
    dpt_tokens = aux.get("stream_dpt_tokens")
    if not isinstance(dpt_tokens, list) or len(dpt_tokens) != len(EXPECTED_DPT_LAYERS):
        raise ValueError("StreamVGGT output lacks the expected DPT token levels.")
    levels = torch.stack(
        [value.detach().float().cpu()[0] for value in dpt_tokens],
        dim=0,
    )
    patch_shape = tuple(int(value) for value in aux["patch_shape"])
    features = point_head_patch_features(
        levels,
        patch_start_idx=int(aux["patch_start_idx"]),
        patch_shape=patch_shape,
    ).contiguous()
    raw_native = aux.get("pointmap_dense")
    raw_confidence = aux.get("confidence_dense")
    if raw_native is None or raw_confidence is None:
        raise ValueError("StreamVGGT output lacks dense pointmap/confidence.")
    raw_native = raw_native.detach().float().cpu()
    confidence = normalize_confidence(raw_confidence.detach().float().cpu())
    if confidence.ndim == 4 and confidence.shape[-1] == 1:
        confidence = confidence[..., 0]
    image_size = tuple(int(value) for value in output.aux["image_shape"])
    if tuple(int(value) for value in raw_native.shape[1:3]) != image_size:
        raise ValueError(
            f"Raw pointmap shape {tuple(raw_native.shape)} does not match "
            f"processed image size {image_size}."
        )

    # Candidate generation is complete before GT pointmaps are opened.
    del output, levels, dpt_tokens
    target_metric = _load_gt_pointmaps(
        manifest_path,
        scene_id=spec["scene_id"],
        frame_indices=spec["frame_indices"],
        processed_size=image_size,
        image_mode=image_mode,
    )
    reference_valid = (
        torch.isfinite(raw_native[0]).all(dim=-1)
        & torch.isfinite(target_metric[0]).all(dim=-1)
        & torch.isfinite(confidence[0])
        & (confidence[0] >= float(confidence_threshold))
    )
    source, target = _paired_limit(
        raw_native[0].reshape(-1, 3)[reference_valid.reshape(-1)],
        target_metric[0].reshape(-1, 3)[reference_valid.reshape(-1)],
        max_points=max_alignment_points,
    )
    scale, rotation, translation, inliers, fit_rmse = _robust_similarity(
        source,
        target,
        min_points=128,
    )
    target_native = invert_similarity(
        target_metric,
        scale=scale,
        rotation=rotation,
        translation=translation,
    )
    support = (
        torch.isfinite(raw_native).all(dim=-1)
        & torch.isfinite(target_native).all(dim=-1)
        & torch.isfinite(target_metric).all(dim=-1)
        & torch.isfinite(confidence)
        & (confidence >= float(confidence_threshold))
    )
    if any(int(support[index].sum()) < 128 for index in range(len(image_paths))):
        raise ValueError(f"Episode {spec['episode_id']} has a frame with <128 support points.")

    payload = {
        "schema": 1,
        "revision": REVISION,
        "scene_id": spec["scene_id"],
        "episode_id": spec["episode_id"],
        "split": spec["split"],
        "frame_indices": tuple(int(value) for value in spec["frame_indices"]),
        "image_names": tuple(spec["image_names"]),
        "features": features.half(),
        "raw_native": raw_native,
        "target_native": target_native,
        "target_metric": target_metric,
        "confidence": confidence.float(),
        "support": support,
        "image_size": image_size,
        "patch_shape": patch_shape,
        "dpt_layer_indices": EXPECTED_DPT_LAYERS,
        "sim3_scale": float(scale),
        "sim3_rotation": rotation.float(),
        "sim3_translation": translation.float(),
        "sim3_inliers": int(inliers),
        "sim3_fit_rmse": float(fit_rmse),
        "gt_opened_after_candidate_generation": 1,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    print(
        f"  saved={cache_path} feature_shape={tuple(features.shape)} "
        f"image_size={image_size} sim3_inliers={inliers}"
    )
    return {**spec, "cache_path": str(cache_path)}


def _build_episode_specs(
    manifest: dict[str, Any],
    splits: dict[str, list[str]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    scenes = {
        str(scene.get("scene_id")): scene
        for scene in manifest.get("scenes", ())
        if isinstance(scene, dict) and scene.get("scene_id")
    }
    length = int(config["episode_length"])
    count = int(config.get("episodes_per_scene", 2))
    if length < 2 or count < 1:
        raise ValueError("episode_length must be >=2 and episodes_per_scene >=1")
    specs = []
    for split, scene_ids in splits.items():
        for scene_id in scene_ids:
            scene = scenes.get(scene_id)
            if scene is None:
                raise ValueError(f"Split scene {scene_id} is absent from manifest.")
            frames = list(scene.get("frames", ()))
            if len(frames) < length:
                raise ValueError(f"Scene {scene_id} has only {len(frames)} frames.")
            max_start = len(frames) - length
            starts = torch.linspace(0, max_start, steps=count).round().long().tolist()
            starts = list(dict.fromkeys(int(value) for value in starts))
            for episode_index, start in enumerate(starts):
                selected = frames[start : start + length]
                specs.append(
                    {
                        "episode_id": f"{scene_id}__ep{episode_index:02d}",
                        "scene_id": scene_id,
                        "split": split,
                        "episode_index": episode_index,
                        "frame_indices": tuple(range(start, start + length)),
                        "image_names": tuple(str(row["image_name"]) for row in selected),
                        "image_paths": tuple(
                            str(_resolve_manifest_path(row["image_path"], manifest))
                            for row in selected
                        ),
                    }
                )
    return specs


def _load_splits(config: dict[str, Any]) -> dict[str, list[str]]:
    raw = config.get("scene_splits", {})
    splits = {
        "train": [str(value) for value in raw.get("train", ())],
        "validation": [str(value) for value in raw.get("validation", ())],
        "test": [str(value) for value in raw.get("test", ())],
    }
    if not all(splits.values()):
        raise ValueError(f"Each scene split must be non-empty: {splits}")
    flat = sum(splits.values(), [])
    if len(flat) != len(set(flat)):
        raise ValueError(f"Scene splits overlap: {splits}")
    return splits


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Manifest must be a JSON object: {path}")
    return payload


def _resolve_manifest_path(value: str, manifest: dict[str, Any]) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    output_root = Path(str(manifest.get("output_root", "."))).expanduser()
    return output_root / path


def _load_runtime(value: str, base: Path) -> dict[str, str]:
    config_path = expand_storage_path(value, base=base)
    raw = yaml.safe_load(config_path.read_text(encoding="utf8")) or {}
    recovery_path = expand_storage_path(raw["recovery_config"], base=config_path.parent)
    recovery = yaml.safe_load(recovery_path.read_text(encoding="utf8")) or {}
    stream = recovery.get("streamvggt", {})
    return {
        "streamvggt_repo": str(
            expand_storage_path(stream.get("repo", "externals/streamvggt"), base=recovery_path.parent)
        ),
        "streamvggt_checkpoint": str(
            expand_storage_path(stream["checkpoint"], base=recovery_path.parent)
        ),
        "image_mode": str(stream.get("image_mode", "crop")),
        "amp_dtype": str(recovery.get("runtime", {}).get("streamvggt_amp_dtype", "bfloat16")),
    }


def _load_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    required = ("manifest", "output_dir", "v0_config", "episode_length", "scene_splits")
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"Config missing keys {missing}: {path}")
    return raw


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf8",
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        return value.tolist()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="streaming_couping/configs/multiscene_geometry.yaml")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
