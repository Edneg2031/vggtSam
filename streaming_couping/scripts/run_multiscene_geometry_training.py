#!/usr/bin/env python3
"""Train and evaluate the first scene-disjoint geometry-only pilot."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch
import yaml

from streaming_couping.src.storage import expand_storage_path
from streaming_couping.src.trust_aware_residual import (
    TrustAwareResidualHead,
    apply_similarity,
    robust_point_loss,
    validation_checkpoint_is_better,
)


REVISION = "multiscene_geometry_only_training_r1"


def main() -> None:
    args = _parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = _load_config(config_path)
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
    device = _resolve_device(str(args.device or config.get("device", "cuda:0")))
    seed = int(config.get("seed", 2026))
    _seed_everything(seed)
    epochs = int(config.get("epochs", 20))
    batch_size = int(config.get("batch_size", 2))
    learning_rate = float(config.get("learning_rate", 2e-4))
    weight_decay = float(config.get("weight_decay", 1e-4))
    grad_clip = float(config.get("grad_clip_norm", 1.0))
    point_beta = float(config.get("point_beta", 0.05))
    correction_reg = float(config.get("correction_regularization_weight", 0.01))
    maximum_points = int(config.get("maximum_points_per_frame", 8192))

    for item in episodes:
        shape = tuple(int(value) for value in item["features"].shape)
        if shape[1:] != (levels, patches, channels):
            raise ValueError(f"Feature layout mismatch in {item['episode_id']}: {shape}")
        if tuple(item["patch_shape"]) != patch_shape or tuple(item["image_size"]) != image_size:
            raise ValueError(f"Spatial layout mismatch in {item['episode_id']}")

    print("MULTI-SCENE GEOMETRY-ONLY TRAINING")
    print(f"cache_index={index_path}")
    print(
        f"scenes=train={','.join(splits['train'])} "
        f"validation={','.join(splits['validation'])} "
        f"test={','.join(splits['test'])}"
    )
    print(
        f"episodes=train={len(train)} validation={len(validation)} test={len(test)} "
        f"frames={sum(item['features'].shape[0] for item in episodes)}"
    )
    print("frozen=StreamVGGT backbone and original PointHead; SAM=0")
    print("checkpoint selection=validation scene only; test=one-pass sealed scoring")

    model = TrustAwareResidualHead(
        feature_channels=channels,
        level_count=levels,
        patch_shape=patch_shape,
        projection_channels=int(config.get("projection_channels", 64)),
        hidden_channels=int(config.get("hidden_channels", 128)),
        use_gate=False,
        use_uncertainty=False,
        gate_bias=float(config.get("gate_bias", -2.0)),
    ).to(device)
    zero = _zero_update_error(model, train[0], device, image_size)
    if zero != 0.0:
        raise RuntimeError(f"Raw fallback is not exact at initialization: {zero}")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        foreach=False,
    )
    items = [(episode_index, frame_index) for episode_index, item in enumerate(episodes) if item["split"] == "train" for frame_index in range(item["features"].shape[0])]
    best_state = None
    best_score = None
    best_epoch = -1
    curve = []
    for epoch in range(epochs):
        model.train()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + epoch * 1009)
        order = torch.randperm(len(items), generator=generator).tolist()
        total_loss = 0.0
        point_loss_total = 0.0
        batches = 0
        for start in range(0, len(order), batch_size):
            selected = [items[order[index]] for index in range(start, min(start + batch_size, len(order)))]
            features, raw, target, valid = _batch(episodes, selected, device)
            output = model(features, output_size=image_size)
            predicted = raw + output.correction
            point_loss = robust_point_loss(predicted, target, valid, beta=point_beta)
            regularization = output.correction[valid].abs().sum(dim=-1).mean()
            loss = point_loss + correction_reg * regularization
            if not bool(torch.isfinite(loss).detach().cpu()):
                raise RuntimeError(f"Non-finite training loss at epoch {epoch + 1}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if not math.isfinite(float(gradient.detach().cpu())):
                raise RuntimeError(f"Non-finite gradient at epoch {epoch + 1}")
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            point_loss_total += float(point_loss.detach().cpu())
            batches += 1

        validation_row, _ = _score_split(model, validation, "validation", device, maximum_points, image_size)
        score = (validation_row["rmse"], validation_row["p90"])
        selected = int(validation_checkpoint_is_better(*score, best_score))
        if selected:
            best_score = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        curve.append(
            {
                "epoch": epoch,
                "mean_loss": total_loss / max(batches, 1),
                "mean_point_loss": point_loss_total / max(batches, 1),
                "validation_rmse": validation_row["rmse"],
                "validation_p90": validation_row["p90"],
                "selected_as_best": selected,
            }
        )
        if epoch == 0 or epoch == epochs - 1 or (epoch + 1) % 5 == 0:
            print(
                f"  epoch={epoch + 1}/{epochs} loss={curve[-1]['mean_loss']:.6f} "
                f"val_rmse={validation_row['rmse']:.6f}"
            )

    if best_state is None or best_epoch < 0:
        raise RuntimeError("No validation checkpoint was selected.")
    model.load_state_dict(best_state, strict=True)
    val_row, val_frames = _score_split(model, validation, "validation", device, maximum_points, image_size)
    raw_val, raw_val_frames = _score_raw(validation, "validation", maximum_points)
    test_row, test_frames = _score_split(model, test, "test", device, maximum_points, image_size)
    raw_test, raw_test_frames = _score_raw(test, "test", maximum_points)
    test_row["rmse_gain_vs_raw_percent"] = _gain(raw_test["rmse"], test_row["rmse"])
    test_row["p90_gain_vs_raw_percent"] = _gain(raw_test["p90"], test_row["p90"])
    test_row["improved_frame_ratio_vs_raw"] = test_row["improved_frames_vs_raw"] / max(len(test_frames), 1)
    decision = {
        "cross_scene_pilot_decision": "GO"
        if (
            test_row["rmse"] < raw_test["rmse"]
            and test_row["p90"] <= raw_test["p90"]
            and test_row["improved_frame_ratio_vs_raw"] > 0.5
        )
        else "NO_GO",
        "rule": "test_rmse_below_raw_and_p90_not_above_raw_and_strict_majority_frames_improved",
        "test_scene_ids": splits["test"],
        "cross_scene_generalization_claim": 0,
    }
    summary = {
        "schema": 1,
        "revision": REVISION,
        "experiment": "frozen_streamvggt_multiscene_residual_no_gate",
        "cache_index": str(index_path),
        "scene_splits": splits,
        "episodes": {split: [item["episode_id"] for item in episodes if item["split"] == split] for split in ("train", "validation", "test")},
        "model": {
            "streamvggt_frozen": 1,
            "original_point_head_frozen": 1,
            "sam_inputs": 0,
            "trainable_branch": "residual_no_gate",
            "feature_shape": (levels, patches, channels),
            "patch_shape": patch_shape,
            "image_size": image_size,
        },
        "checkpoint_selection": {
            "rule": "minimum_validation_rmse_then_minimum_validation_p90",
            "best_epoch": best_epoch,
            "test_metrics_read_during_selection": 0,
        },
        "validation": {"raw": raw_val, "residual_no_gate": val_row},
        "test": {"raw": raw_test, "residual_no_gate": test_row},
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
            "best_epoch": best_epoch,
            "scene_splits": splits,
        },
        output_dir / "residual_no_gate.pt",
    )
    _write_json(output_dir / "summary.json", summary)
    _write_csv(output_dir / "training_curve.csv", curve)
    _write_csv(output_dir / "frame_metrics.csv", summary["frame_metrics"])
    print(
        f"  raw_test_rmse={raw_test['rmse']:.6f} "
        f"residual_test_rmse={test_row['rmse']:.6f} "
        f"gain={test_row['rmse_gain_vs_raw_percent']:.4f}% "
        f"decision={decision['cross_scene_pilot_decision']}"
    )
    print(f"  result={output_dir / 'summary.json'}")


def _load_episodes(index: dict[str, Any], index_path: Path) -> list[dict[str, Any]]:
    episodes = []
    for record in index.get("episodes", ()):
        path = Path(record["cache_path"]).expanduser()
        if not path.is_absolute():
            path = index_path.parent / path
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload["episode_id"] = str(record["episode_id"])
        payload["scene_id"] = str(record["scene_id"])
        payload["split"] = str(record["split"])
        payload["cache_path"] = str(path)
        episodes.append(payload)
    if not episodes:
        raise ValueError("Cache index contains no episodes.")
    return episodes


def _validate_splits(episodes: list[dict[str, Any]], splits: dict[str, Any]) -> None:
    for split in ("train", "validation", "test"):
        expected = set(str(value) for value in splits.get(split, ()))
        actual = {item["scene_id"] for item in episodes if item["split"] == split}
        if expected != actual:
            raise ValueError(f"Split mismatch for {split}: expected={expected} actual={actual}")


def _batch(episodes: list[dict[str, Any]], selected: list[tuple[int, int]], device: torch.device):
    features = torch.stack([episodes[e]["features"][f].float() for e, f in selected]).to(device)
    raw = torch.stack([episodes[e]["raw_native"][f] for e, f in selected]).to(device)
    target = torch.stack([episodes[e]["target_native"][f] for e, f in selected]).to(device)
    valid = torch.stack([episodes[e]["support"][f] for e, f in selected]).to(device)
    return features, raw, target, valid


@torch.inference_mode()
def _score_split(model, episodes, split, device, maximum_points, image_size):
    rows = []
    frame_rows = []
    all_errors = []
    raw_frame_lookup = {}
    for episode in episodes:
        for frame_index in range(episode["features"].shape[0]):
            features = episode["features"][frame_index : frame_index + 1].float().to(device)
            output = model(features, output_size=image_size)
            predicted = output.correction.cpu()[0] + episode["raw_native"][frame_index]
            metric = apply_similarity(
                predicted,
                scale=float(episode["sim3_scale"]),
                rotation=episode["sim3_rotation"],
                translation=episode["sim3_translation"],
            )
            target = episode["target_metric"][frame_index]
            selected = _limited_indices(episode["support"][frame_index], maximum_points)
            error = torch.linalg.vector_norm(
                metric.reshape(-1, 3).index_select(0, selected)
                - target.reshape(-1, 3).index_select(0, selected),
                dim=-1,
            )
            raw_metric = apply_similarity(
                episode["raw_native"][frame_index],
                scale=float(episode["sim3_scale"]),
                rotation=episode["sim3_rotation"],
                translation=episode["sim3_translation"],
            )
            raw_error = torch.linalg.vector_norm(
                raw_metric.reshape(-1, 3).index_select(0, selected)
                - target.reshape(-1, 3).index_select(0, selected),
                dim=-1,
            )
            frame_rows.append({
                "split": split,
                "scene_id": episode["scene_id"],
                "episode_id": episode["episode_id"],
                "frame_index": int(episode["frame_indices"][frame_index]),
                "branch": "residual_no_gate",
                "rmse": _rmse(error),
                "raw_rmse": _rmse(raw_error),
                "improved_vs_raw": int(_rmse(error) < _rmse(raw_error)),
            })
            all_errors.append(error)
            raw_frame_lookup[(episode["episode_id"], frame_index)] = raw_error
    joined = torch.cat(all_errors)
    raw_joined = torch.cat(list(raw_frame_lookup.values()))
    return (
        {
            "split": split,
            "branch": "residual_no_gate",
            "rmse": _rmse(joined),
            "median": float(joined.median()),
            "p90": float(torch.quantile(joined, 0.90)),
            "improved_frames_vs_raw": sum(row["improved_vs_raw"] for row in frame_rows),
            "frame_count": len(frame_rows),
            "raw_rmse_reference": _rmse(raw_joined),
        },
        frame_rows,
    )


def _score_raw(episodes, split, maximum_points):
    rows = []
    frame_rows = []
    errors = []
    for episode in episodes:
        for frame_index in range(episode["features"].shape[0]):
            selected = _limited_indices(episode["support"][frame_index], maximum_points)
            metric = apply_similarity(
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
            errors.append(error)
            frame_rows.append({
                "split": split,
                "scene_id": episode["scene_id"],
                "episode_id": episode["episode_id"],
                "frame_index": int(episode["frame_indices"][frame_index]),
                "branch": "raw_full_history",
                "rmse": _rmse(error),
                "raw_rmse": _rmse(error),
                "improved_vs_raw": 0,
            })
    joined = torch.cat(errors)
    return (
        {
            "split": split,
            "branch": "raw_full_history",
            "rmse": _rmse(joined),
            "median": float(joined.median()),
            "p90": float(torch.quantile(joined, 0.90)),
            "frame_count": len(frame_rows),
        },
        frame_rows,
    )


def _zero_update_error(model, episode, device, image_size):
    model.eval()
    output = model(episode["features"][0:1].float().to(device), output_size=image_size)
    return float(output.correction.abs().max().detach().cpu())


def _limited_indices(mask, maximum):
    selected = torch.nonzero(mask.reshape(-1), as_tuple=False)[:, 0]
    if selected.numel() <= int(maximum):
        return selected
    return selected.index_select(0, torch.linspace(0, selected.numel() - 1, int(maximum)).long())


def _rmse(values):
    return float(torch.sqrt(values.float().square().mean()))


def _gain(raw, candidate):
    return 100.0 * (float(raw) - float(candidate)) / max(abs(float(raw)), 1e-12)


def _resolve_device(value):
    device = torch.device(str(value))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for multi-scene geometry training.")
    return device


def _seed_everything(seed):
    import random

    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _load_config(path):
    raw = yaml.safe_load(path.read_text(encoding="utf8")) or {}
    required = ("cache_index", "output_dir")
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"Training config missing {missing}: {path}")
    return raw


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf8")


def _write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _json_default(value):
    if torch.is_tensor(value):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not serializable: {type(value).__name__}")


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="streaming_couping/configs/multiscene_geometry_train.yaml")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
