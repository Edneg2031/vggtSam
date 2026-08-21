#!/usr/bin/env python3
"""Run the pre-registered multi-scene Direct Geometry Decoder baseline.

This is Stage 1 branch C for the scene-disjoint pilot.  The StreamVGGT
aggregator/backbone remains frozen and the released original PointHead is
initialized from the checkpoint, then optimized on the cached DPT patch
features.  The cache already contains the per-episode native-gauge target and
the fixed reference-frame Sim(3); no new candidate generation or SAM input is
used here.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import random
from pathlib import Path
from typing import Any, Sequence

import torch
import yaml

from streaming_couping.scripts.run_multiscene_geometry_training import (
    _annotate_comparison,
    _beats_raw,
    _frame_coverage,
    _load_episodes,
    _limited_indices,
    _print_frame_coverage,
    _score_raw,
    _validate_splits,
    _write_csv,
    _write_json,
)
from streaming_couping.src.backbones.streamvggt_latent import (
    load_streamvggt_latent_model,
)
from streaming_couping.src.external_repos import maybe_add_repo_to_path
from streaming_couping.src.storage import expand_storage_path
from streaming_couping.src.trust_aware_residual import (
    apply_similarity,
    robust_point_loss,
)


REVISION = "multiscene_geometry_direct_point_head_r1"
EXPECTED_DPT_LAYERS = (4, 11, 17, 23)
FIXED_PROTOCOL = {
    "epochs": 30,
    "batch_size": 2,
    "learning_rate": 2e-4,
    "weight_decay": 1e-4,
    "grad_clip_norm": 1.0,
    "point_beta": 0.05,
}


def main() -> None:
    args = _parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = _load_config(config_path)
    _validate_protocol(config)

    index_path = expand_storage_path(config["cache_index"], base=config_path.parent)
    index = json.loads(index_path.read_text(encoding="utf8"))
    output_dir = expand_storage_path(config["output_dir"], base=config_path.parent)
    output_dir.mkdir(parents=True, exist_ok=True)

    episodes = _load_episodes(index, index_path)
    splits = index.get("splits", {})
    _validate_splits(episodes, splits)
    train = [item for item in episodes if item["split"] == "train"]
    validation = [item for item in episodes if item["split"] == "validation"]
    test = [item for item in episodes if item["split"] == "test"]
    if not train or not validation or not test:
        raise ValueError("Train, validation and test episodes must all be non-empty.")

    sample = train[0]
    feature_shape = tuple(int(value) for value in sample["features"].shape)
    if len(feature_shape) != 4:
        raise ValueError(f"Expected episode features [T,L,N,C], got {feature_shape}")
    _, levels, patches, channels = feature_shape
    patch_shape = tuple(int(value) for value in sample["patch_shape"])
    image_size = tuple(int(value) for value in sample["image_size"])
    for item in episodes:
        shape = tuple(int(value) for value in item["features"].shape)
        if shape[1:] != (levels, patches, channels):
            raise ValueError(f"Feature layout mismatch in {item['episode_id']}: {shape}")
        if tuple(item["patch_shape"]) != patch_shape or tuple(item["image_size"]) != image_size:
            raise ValueError(f"Spatial layout mismatch in {item['episode_id']}")

    coverage = _frame_coverage(episodes)
    maximum_points = int(config.get("maximum_points_per_frame", 8192))
    raw_val, raw_val_frames = _score_raw(validation, "validation", maximum_points)

    device = _resolve_device(str(config.get("device", "cuda:0")))
    _seed_everything(int(config.get("seed", 2026)))
    print("MULTI-SCENE DIRECT GEOMETRY DECODER TRAINING")
    print(f"cache_index={index_path}")
    print(
        f"scenes=train={','.join(splits['train'])} "
        f"validation={','.join(splits['validation'])} "
        f"test={','.join(splits['test'])}"
    )
    print(
        f"episodes=train={len(train)} validation={len(validation)} test={len(test)} "
        f"total_frame_occurrences={sum(item['features'].shape[0] for item in episodes)} "
        f"train_frame_occurrences={sum(item['features'].shape[0] for item in train)}"
    )
    print(
        "frozen=StreamVGGT aggregator/backbone + CameraHead; "
        "trainable=original pretrained PointHead; SAM=0"
    )
    print("checkpoint selection=validation scene only; test=one-pass sealed scoring")
    _print_frame_coverage(coverage)
    print(
        f"  raw_val_rmse={raw_val['rmse']:.6f} "
        f"raw_val_median={raw_val['median']:.6f} "
        f"raw_val_p90={raw_val['p90']:.6f}"
    )

    point_head, layer_indices = _load_pretrained_point_head(config, config_path.parent)
    if layer_indices != EXPECTED_DPT_LAYERS:
        raise ValueError(
            f"Direct decoder requires PointHead layers {EXPECTED_DPT_LAYERS}, "
            f"got {layer_indices}"
        )
    point_head = point_head.to(device)
    initial_error = _zero_update_error(point_head, train[0], device, image_size)
    print(f"  pretrained_raw_fallback_max_abs_error={initial_error:.8f}")
    tolerance = float(config.get("raw_fallback_tolerance", 1e-2))
    if not math.isfinite(initial_error) or initial_error > tolerance:
        raise RuntimeError(
            "Pretrained PointHead does not reproduce the cached raw pointmap "
            f"within tolerance={tolerance}: max_abs_error={initial_error}"
        )

    optimizer = torch.optim.AdamW(
        point_head.parameters(),
        lr=FIXED_PROTOCOL["learning_rate"],
        weight_decay=FIXED_PROTOCOL["weight_decay"],
        foreach=False,
    )
    train_items = [
        (episode_index, frame_index)
        for episode_index, item in enumerate(episodes)
        if item["split"] == "train"
        for frame_index in range(item["features"].shape[0])
    ]
    seed = int(config.get("seed", 2026))
    best_state: dict[str, torch.Tensor] | None = None
    best_score: tuple[float, float] | None = None
    best_epoch = -1
    curve: list[dict[str, Any]] = []

    for epoch in range(FIXED_PROTOCOL["epochs"]):
        point_head.train()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + epoch * 1009)
        order = torch.randperm(len(train_items), generator=generator).tolist()
        losses: list[float] = []
        for start in range(0, len(order), FIXED_PROTOCOL["batch_size"]):
            selected = [
                train_items[order[index]]
                for index in range(
                    start,
                    min(start + FIXED_PROTOCOL["batch_size"], len(order)),
                )
            ]
            features, target, valid = _batch(episodes, selected, device)
            prediction = _forward_point_head(
                point_head,
                features,
                image_size=image_size,
                device=device,
            )
            loss = robust_point_loss(
                prediction,
                target,
                valid,
                beta=FIXED_PROTOCOL["point_beta"],
            )
            if not bool(torch.isfinite(loss).detach().cpu()):
                raise RuntimeError(f"Non-finite direct decoder loss at epoch {epoch + 1}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(
                point_head.parameters(),
                FIXED_PROTOCOL["grad_clip_norm"],
            )
            if not math.isfinite(float(gradient.detach().cpu())):
                raise RuntimeError(f"Non-finite direct decoder gradient at epoch {epoch + 1}")
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        validation_row, _ = _score_direct(
            point_head,
            validation,
            "validation",
            device,
            maximum_points,
            image_size,
        )
        validation_row = _annotate_comparison(validation_row, raw_val)
        score = (validation_row["rmse"], validation_row["p90"])
        selected_as_best = int(best_score is None or score < best_score)
        if selected_as_best:
            best_score = score
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in point_head.state_dict().items()
            }
        curve.append(
            {
                "epoch": epoch,
                "train_loss": float(sum(losses) / max(len(losses), 1)),
                "validation_rmse": validation_row["rmse"],
                "validation_median": validation_row["median"],
                "validation_p90": validation_row["p90"],
                "raw_validation_rmse": raw_val["rmse"],
                "raw_validation_median": raw_val["median"],
                "raw_validation_p90": raw_val["p90"],
                "validation_rmse_gain_vs_raw_percent": validation_row[
                    "rmse_gain_vs_raw_percent"
                ],
                "validation_improved_frame_ratio_vs_raw": validation_row[
                    "improved_frame_ratio_vs_raw"
                ],
                "selected_as_best": selected_as_best,
            }
        )
        if epoch == 0 or epoch == FIXED_PROTOCOL["epochs"] - 1 or (epoch + 1) % 5 == 0:
            print(
                f"  epoch={epoch + 1}/{FIXED_PROTOCOL['epochs']} "
                f"loss={curve[-1]['train_loss']:.6f} "
                f"val_rmse={validation_row['rmse']:.6f} "
                f"val_gain={validation_row['rmse_gain_vs_raw_percent']:.4f}%"
            )

    if best_state is None or best_epoch < 0:
        raise RuntimeError("Direct decoder training did not produce a validation checkpoint")
    point_head.load_state_dict(best_state, strict=True)
    point_head.eval()

    val_row, val_frames = _score_direct(
        point_head,
        validation,
        "validation",
        device,
        maximum_points,
        image_size,
    )
    val_row = _annotate_comparison(val_row, raw_val)
    test_row, test_frames = _score_direct(
        point_head,
        test,
        "test",
        device,
        maximum_points,
        image_size,
    )
    raw_test, raw_test_frames = _score_raw(test, "test", maximum_points)
    test_row = _annotate_comparison(test_row, raw_test)
    validation_beats_raw = _beats_raw(val_row, raw_val)
    test_pass = _beats_raw(test_row, raw_test)
    decision = {
        "direct_geometry_decoder_decision": "GO" if test_pass else "NO_GO",
        "rule": "test_rmse_below_raw_and_p90_not_above_raw_and_strict_majority_frames_improved",
        "validation_residual_comparison": "not_applicable",
        "validation_direct_beats_raw": int(validation_beats_raw),
        "interpretation_case": (
            "C_direct_beats_raw_on_validation_and_test"
            if validation_beats_raw and test_pass
            else "direct_validation_beats_raw_but_test_fails"
            if validation_beats_raw
            else "direct_does_not_beat_validation_raw"
        ),
        "test_scene_ids": splits["test"],
        "test_metrics_read_during_selection": 0,
    }
    summary = {
        "schema": 1,
        "revision": REVISION,
        "experiment": "frozen_streamvggt_multiscene_direct_original_point_head",
        "cache_index": str(index_path),
        "scene_splits": splits,
        "scene_disjoint_protocol": {
            "name": "fold1_2_train_1_validation_1_test",
            "train_scene_count": len(splits["train"]),
            "validation_scene_count": len(splits["validation"]),
            "test_scene_count": len(splits["test"]),
        },
        "episodes": {
            split: [
                item["episode_id"]
                for item in episodes
                if item["split"] == split
            ]
            for split in ("train", "validation", "test")
        },
        "data_coverage": coverage,
        "model": {
            "streamvggt_aggregator_backbone_frozen": 1,
            "camera_head_frozen": 1,
            "original_point_head_initialized_from_checkpoint": 1,
            "original_point_head_trainable": 1,
            "sam_inputs": 0,
            "feature_shape": (levels, patches, channels),
            "patch_shape": patch_shape,
            "image_size": image_size,
            "cached_patch_start_idx_used_for_direct_head": 0,
            "pretrained_raw_fallback_max_abs_error": initial_error,
        },
        "protocol": FIXED_PROTOCOL,
        "checkpoint_selection": {
            "rule": "minimum_validation_rmse_then_minimum_validation_p90",
            "best_epoch": best_epoch,
            "test_metrics_read_during_selection": 0,
        },
        "validation": {
            "raw": raw_val,
            "direct_point_head": val_row,
            "direct_beats_raw": int(validation_beats_raw),
        },
        "test": {"raw": raw_test, "direct_point_head": test_row},
        "frame_metrics": raw_val_frames + val_frames + raw_test_frames + test_frames,
        "decision": decision,
        "formal_v0_modified": 0,
    }
    torch.save(
        {
            "schema": 1,
            "revision": REVISION,
            "state_dict": best_state,
            "feature_shape": (levels, patches, channels),
            "patch_shape": patch_shape,
            "image_size": image_size,
            "patch_start_idx": 0,
            "dpt_layer_indices": EXPECTED_DPT_LAYERS,
            "best_epoch": best_epoch,
            "scene_splits": splits,
        },
        output_dir / "direct_point_head.pt",
    )
    _write_json(output_dir / "summary.json", summary)
    _write_csv(output_dir / "training_curve.csv", curve)
    _write_csv(output_dir / "frame_metrics.csv", summary["frame_metrics"])

    print(
        f"  val_raw=(rmse={raw_val['rmse']:.6f}, median={raw_val['median']:.6f}, "
        f"p90={raw_val['p90']:.6f}) "
        f"val_direct=(rmse={val_row['rmse']:.6f}, median={val_row['median']:.6f}, "
        f"p90={val_row['p90']:.6f}) "
        f"gain={val_row['rmse_gain_vs_raw_percent']:.4f}% "
        f"improved_frame_ratio={val_row['improved_frame_ratio_vs_raw']:.4f}"
    )
    print(
        f"  raw_test_rmse={raw_test['rmse']:.6f} "
        f"direct_test_rmse={test_row['rmse']:.6f} "
        f"gain={test_row['rmse_gain_vs_raw_percent']:.4f}% "
        f"decision={decision['direct_geometry_decoder_decision']}"
    )
    print(f"  result={output_dir / 'summary.json'}")

    point_head.to("cpu")
    del point_head
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    required = ("cache_index", "output_dir", "v0_config")
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"Direct decoder config missing {missing}: {path}")
    return raw


def _validate_protocol(config: dict[str, Any]) -> None:
    for key, expected in FIXED_PROTOCOL.items():
        actual = float(config.get(key, expected))
        if actual != float(expected):
            raise ValueError(
                f"Direct geometry fixed protocol changed: {key}={actual}, "
                f"expected {expected}"
            )


def _load_pretrained_point_head(
    config: dict[str, Any],
    config_base: Path,
) -> tuple[torch.nn.Module, tuple[int, ...]]:
    v0_config = expand_storage_path(config["v0_config"], base=config_base)
    raw = yaml.safe_load(v0_config.read_text(encoding="utf8")) or {}
    recovery_path = expand_storage_path(raw["recovery_config"], base=v0_config.parent)
    recovery = yaml.safe_load(recovery_path.read_text(encoding="utf8")) or {}
    stream = recovery.get("streamvggt", {})
    repo = expand_storage_path(
        stream.get("repo", "externals/streamvggt"),
        base=recovery_path.parent,
    )
    checkpoint = expand_storage_path(
        stream["checkpoint"],
        base=recovery_path.parent,
    )
    maybe_add_repo_to_path(repo)
    model = load_streamvggt_latent_model(
        repo_path=repo,
        checkpoint_path=checkpoint,
        device="cpu",
        strict=True,
    )
    point_head = copy.deepcopy(model.point_head).float()
    layer_indices = tuple(int(value) for value in point_head.intermediate_layer_idx)
    del model
    gc.collect()
    return point_head, layer_indices


def _batch(
    episodes: list[dict[str, Any]],
    selected: list[tuple[int, int]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features = torch.stack(
        [episodes[episode]["features"][frame].float() for episode, frame in selected]
    ).to(device)
    target = torch.stack(
        [episodes[episode]["target_native"][frame] for episode, frame in selected]
    ).to(device)
    valid = torch.stack(
        [episodes[episode]["support"][frame] for episode, frame in selected]
    ).to(device)
    return features, target, valid


def _forward_point_head(
    point_head: torch.nn.Module,
    features: torch.Tensor,
    *,
    image_size: Sequence[int],
    device: torch.device,
) -> torch.Tensor:
    """Run the original PointHead from patch-only cached DPT features."""

    if features.ndim != 4:
        raise ValueError(f"Expected features [S,L,N,C], got {features.shape}")
    frame_count, level_count, _, _ = map(int, features.shape)
    if level_count != len(EXPECTED_DPT_LAYERS):
        raise ValueError(f"Expected four DPT levels, got {level_count}")
    token_list: list[torch.Tensor | None] = [None] * (max(EXPECTED_DPT_LAYERS) + 1)
    for level, layer_index in enumerate(EXPECTED_DPT_LAYERS):
        # The multi-scene cache stores only the tokens after the original
        # patch_start_idx.  Passing them at index zero is exactly what DPTHead
        # consumes internally and avoids fabricating special-token values.
        token_list[layer_index] = features[:, level].unsqueeze(0)
    height, width = (int(value) for value in image_size)
    images = torch.zeros(
        (1, frame_count, 3, height, width),
        dtype=features.dtype,
        device=device,
    )
    prediction, _ = point_head(
        token_list,
        images=images,
        patch_start_idx=0,
        frames_chunk_size=max(1, frame_count),
    )
    prediction = prediction[0]
    if prediction.ndim != 4 or int(prediction.shape[-1]) != 3:
        raise ValueError(f"Direct PointHead returned unexpected shape: {prediction.shape}")
    if tuple(int(value) for value in prediction.shape[1:3]) != (height, width):
        raise ValueError(
            f"Direct PointHead output {prediction.shape} does not match image_size={image_size}"
        )
    return prediction


@torch.inference_mode()
def _zero_update_error(
    point_head: torch.nn.Module,
    episode: dict[str, Any],
    device: torch.device,
    image_size: Sequence[int],
) -> float:
    prediction = _forward_point_head(
        point_head,
        episode["features"][0:1].float().to(device),
        image_size=image_size,
        device=device,
    ).cpu()[0]
    return float((prediction - episode["raw_native"][0]).abs().max())


@torch.inference_mode()
def _score_direct(
    point_head: torch.nn.Module,
    episodes: list[dict[str, Any]],
    split: str,
    device: torch.device,
    maximum_points: int,
    image_size: Sequence[int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    point_head.eval()
    errors: list[torch.Tensor] = []
    raw_errors: list[torch.Tensor] = []
    frame_rows: list[dict[str, Any]] = []
    for episode in episodes:
        for frame_index in range(episode["features"].shape[0]):
            prediction = _forward_point_head(
                point_head,
                episode["features"][frame_index : frame_index + 1].float().to(device),
                image_size=image_size,
                device=device,
            ).cpu()[0]
            selected = _limited_indices(episode["support"][frame_index], maximum_points)
            metric = apply_similarity(
                prediction,
                scale=float(episode["sim3_scale"]),
                rotation=episode["sim3_rotation"],
                translation=episode["sim3_translation"],
            )
            raw_metric = apply_similarity(
                episode["raw_native"][frame_index],
                scale=float(episode["sim3_scale"]),
                rotation=episode["sim3_rotation"],
                translation=episode["sim3_translation"],
            )
            target = episode["target_metric"][frame_index]
            error = torch.linalg.vector_norm(
                metric.reshape(-1, 3).index_select(0, selected)
                - target.reshape(-1, 3).index_select(0, selected),
                dim=-1,
            )
            raw_error = torch.linalg.vector_norm(
                raw_metric.reshape(-1, 3).index_select(0, selected)
                - target.reshape(-1, 3).index_select(0, selected),
                dim=-1,
            )
            errors.append(error)
            raw_errors.append(raw_error)
            frame_rows.append(
                {
                    "split": split,
                    "scene_id": episode["scene_id"],
                    "episode_id": episode["episode_id"],
                    "frame_index": int(episode["frame_indices"][frame_index]),
                    "branch": "direct_point_head",
                    "rmse": _rmse(error),
                    "raw_rmse": _rmse(raw_error),
                    "improved_vs_raw": int(_rmse(error) < _rmse(raw_error)),
                }
            )
    joined = torch.cat(errors)
    raw_joined = torch.cat(raw_errors)
    return (
        {
            "split": split,
            "branch": "direct_point_head",
            "rmse": _rmse(joined),
            "median": float(joined.median()),
            "p90": float(torch.quantile(joined, 0.90)),
            "improved_frames_vs_raw": sum(
                row["improved_vs_raw"] for row in frame_rows
            ),
            "frame_count": len(frame_rows),
            "raw_rmse_reference": _rmse(raw_joined),
        },
        frame_rows,
    )


def _rmse(values: torch.Tensor) -> float:
    return float(torch.sqrt(values.float().square().mean()))


def _resolve_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested on {device}, but CUDA is unavailable")
    return device


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="streaming_couping/configs/multiscene_geometry_direct_train.yaml",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
