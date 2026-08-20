#!/usr/bin/env python3
"""Train and evaluate the geometry-only trainable PointHead baseline.

Stage 2 keeps the StreamVGGT aggregator/backbone and CameraHead frozen.  The
pretrained StreamVGGT PointHead is initialized from the released checkpoint,
then trained only on the development-scene V0 feature cache.  Validation
RMSE/P90 select one checkpoint; that checkpoint is evaluated once on the two
held-out R3 scenes.  No SAM, QK pose update, external residual head, or
held-out GT enters candidate generation.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
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
from streaming_couping.src.learned_pose.baseline_runtime import (
    load_baseline_run_config,
)
from streaming_couping.src.learned_pose.cache import cache_path, load_feature_cache
from streaming_couping.src.learned_pose.config import load_learned_pose_config
from streaming_couping.src.pointmap_alignment import (
    _load_gt_pointmaps,
    _paired_limit,
    _robust_similarity,
)
from streaming_couping.src.storage import expand_storage_path
from streaming_couping.src.trust_aware_residual import (
    apply_similarity,
    invert_similarity,
)


REVISION = "stage2_trainable_point_head_geometry_only_r1"
EXPECTED_DPT_LAYERS = (4, 11, 17, 23)


@dataclass(frozen=True)
class RunConfig:
    source_path: Path
    v0_config: Path
    output_dir: Path
    device: str
    devices: tuple[str, ...]
    seed: int
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    grad_clip_norm: float
    point_beta: float
    confidence_threshold: float
    maximum_points: int
    heldout_scenes: tuple[tuple[str, Path], ...]


@dataclass
class DevelopmentData:
    clip: str
    levels: torch.Tensor  # [L,S,T,C]
    patch_start_idx: int
    patch_shape: tuple[int, int]
    image_size: tuple[int, int]
    raw_native: torch.Tensor  # [S,H,W,3]
    target_native: torch.Tensor
    target_metric: torch.Tensor
    confidence: torch.Tensor
    support: torch.Tensor
    scale: float
    rotation: torch.Tensor
    translation: torch.Tensor
    frame_indices: tuple[int, ...]
    cache_path: Path


@dataclass
class HeldoutFeatures:
    scene_id: str
    r3_manifest: Path
    frames: list[dict[str, Any]]
    levels: torch.Tensor  # [L,S,T,C]
    patch_start_idx: int
    patch_shape: tuple[int, int]
    image_size: tuple[int, int]
    raw: torch.Tensor
    confidence: torch.Tensor


def main() -> None:
    args = _parse_args()
    run = _load_run(args.config, device_override=args.device)
    _seed_everything(run.seed)
    run.output_dir.mkdir(parents=True, exist_ok=True)
    development = _load_development_data(run)
    _validate_split(run, len(development.frame_indices))
    device = _resolve_device(run.device)

    print("STAGE 2 GEOMETRY-ONLY TRAINABLE POINTHEAD")
    print(
        f"  development_clip={development.clip} frames={len(development.frame_indices)} "
        f"split={len(run.train_indices)}/{len(run.validation_indices)}"
    )
    print(
        f"  features={tuple(development.levels.shape)} "
        f"patch={development.patch_shape} dense={development.image_size} device={device}"
    )
    print("  frozen=StreamVGGT aggregator/backbone + CameraHead; SAM=0; QK=unchanged")
    print("  trainable=pretrained original PointHead only")

    point_head, layer_indices = _load_pretrained_point_head(run)
    if layer_indices != EXPECTED_DPT_LAYERS:
        raise ValueError(
            f"Stage 2 requires PointHead layers {EXPECTED_DPT_LAYERS}, got {layer_indices}"
        )
    point_head = point_head.to(device)
    curve, best_state, selection = _train_point_head(
        point_head,
        development,
        run,
        device,
    )
    point_head.load_state_dict(best_state, strict=True)
    point_head.eval()
    dev_rows = _development_evaluation(point_head, development, run, device)
    print(
        f"  best-validation epoch={selection['best_epoch'] + 1} "
        f"rmse={selection['best_validation_rmse']:.6f} "
        f"p90={selection['best_validation_p90']:.6f}"
    )

    # Extract held-out features and raw predictions before opening held-out GT.
    heldout = _extract_heldout_features(run)
    print("  held-out candidate features frozen; opening GT only for scoring")
    heldout_summaries: list[dict[str, Any]] = []
    heldout_frames: list[dict[str, Any]] = []
    for item in heldout:
        summary, rows = _evaluate_heldout(point_head, item, run, device)
        heldout_summaries.append(summary)
        heldout_frames.extend(rows)
        print(
            f"  scene={item.scene_id} raw_rmse={summary['raw_rmse']:.6f} "
            f"point_head_rmse={summary['point_head_rmse']:.6f} "
            f"raw_p90={summary['raw_p90']:.6f} "
            f"point_head_p90={summary['point_head_p90']:.6f} "
            f"decision={summary['decision']}"
        )

    checkpoint = {
        "schema": 1,
        "revision": REVISION,
        "formal_v0_artifact": 0,
        "model": {
            "point_head_layer_indices": list(layer_indices),
            "patch_shape": list(development.patch_shape),
            "patch_start_idx": development.patch_start_idx,
            "image_size": list(development.image_size),
        },
        "training_config": _serializable_run(run),
        "checkpoint_selection": selection,
        "state_dict": best_state,
    }
    torch.save(checkpoint, run.output_dir / "point_head_only.pt")

    summary = {
        "schema": 1,
        "revision": REVISION,
        "experiment": "geometry_only_trainable_original_point_head",
        "development_clip": development.clip,
        "development_cache": str(development.cache_path),
        "development_split": {
            "train_sequence_indices": list(run.train_indices),
            "validation_sequence_indices": list(run.validation_indices),
        },
        "branches": [
            {
                "branch": "raw_full_history",
                **dev_rows["raw_full_history"],
            },
            {
                "branch": "point_head_only",
                **dev_rows["point_head_only"],
                "selected_checkpoint_epoch_one_based": selection["best_epoch"] + 1,
            },
        ],
        "heldout_scenes": heldout_summaries,
        "training_curve": curve,
        "checkpoint_selection": selection,
        "dpt_layer_indices": list(layer_indices),
        "sam_inputs": 0,
        "qk_pose_modified": 0,
        "backbone_parameters_updated": 0,
        "camera_head_parameters_updated": 0,
        "point_head_parameters_updated": 1,
        "heldout_gt_used_for_candidate_generation": 0,
        "formal_v0_modified": 0,
        "decision": {
            "heldout_scene_count": len(heldout_summaries),
            "all_heldout_rmse_below_raw": int(
                all(row["point_head_rmse"] < row["raw_rmse"] for row in heldout_summaries)
            ),
            "all_heldout_p90_not_above_raw": int(
                all(row["point_head_p90"] <= row["raw_p90"] for row in heldout_summaries)
            ),
            "stage2_point_head_decision": (
                "GO"
                if heldout_summaries
                and all(row["decision"] == "GO" for row in heldout_summaries)
                else "NO_GO"
            ),
            "next_gate": "compare_external_residual_and_point_head_plus_residual"
            if heldout_summaries
            and all(row["decision"] == "GO" for row in heldout_summaries)
            else "do_not_add_sam_or_finetune_backbone",
        },
    }
    _write_json(run.output_dir / "summary.json", summary)
    _write_csv(run.output_dir / "training_curve.csv", curve)
    _write_csv(run.output_dir / "heldout_frame_metrics.csv", heldout_frames)
    _write_copyable(run.output_dir / "copyable_result.txt", summary)
    print(f"  result={run.output_dir / 'summary.json'}")

    point_head.to("cpu")
    del point_head
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_development_data(run: RunConfig) -> DevelopmentData:
    config = load_learned_pose_config(run.v0_config)
    baseline = load_baseline_run_config(run.v0_config)
    clips = [clip for clip in config.clips if clip.name == baseline.clip_name]
    if len(clips) != 1:
        raise ValueError(f"Expected one development clip {baseline.clip_name!r}")
    clip = clips[0]
    source = cache_path(config, clip)
    if not source.is_file():
        raise FileNotFoundError(f"Missing frozen V0 cache: {source}")
    payload = load_feature_cache(source)
    required = {
        "clip_name",
        "frame_indices",
        "token_levels",
        "patch_start_idx",
        "patch_shape",
        "image_size",
        "baseline_world_points",
        "baseline_world_confidence",
        "target_world_points",
        "point_alignment_scale",
        "point_alignment_rotation",
        "point_alignment_translation",
        "dpt_layer_indices",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"V0 cache lacks Stage 2 fields: {sorted(missing)}")
    frame_indices = tuple(int(value) for value in payload["frame_indices"])
    if frame_indices != clip.frame_indices:
        raise ValueError("V0 cache frame order differs from the development clip")
    levels = torch.as_tensor(payload["token_levels"]).detach().float().cpu()
    if levels.ndim != 4 or int(levels.shape[0]) != 4:
        raise ValueError(f"Expected cached token_levels [4,S,T,C], got {levels.shape}")
    layer_indices = tuple(int(value) for value in payload["dpt_layer_indices"])
    if layer_indices != EXPECTED_DPT_LAYERS:
        raise ValueError(f"Unexpected cached DPT levels: {layer_indices}")
    patch_start_idx = int(payload["patch_start_idx"])
    patch_shape = tuple(int(value) for value in payload["patch_shape"])
    if int(levels.shape[-2]) - patch_start_idx != patch_shape[0] * patch_shape[1]:
        raise ValueError("Cached patch_start_idx/patch_shape do not match token levels")
    raw_native = torch.as_tensor(payload["baseline_world_points"]).detach().float().cpu()
    confidence = torch.as_tensor(
        payload["baseline_world_confidence"]
    ).detach().float().cpu()
    target_metric = torch.as_tensor(payload["target_world_points"]).detach().float().cpu()
    scale = float(payload["point_alignment_scale"])
    rotation = torch.as_tensor(payload["point_alignment_rotation"]).detach().float().cpu()
    translation = torch.as_tensor(payload["point_alignment_translation"]).detach().float().cpu()
    target_native = invert_similarity(
        target_metric,
        scale=scale,
        rotation=rotation,
        translation=translation,
    )
    image_size = tuple(int(value) for value in payload["image_size"])
    if tuple(int(value) for value in raw_native.shape[1:3]) != image_size:
        raise ValueError(
            f"Cache image_size={image_size} differs from raw={tuple(raw_native.shape)}"
        )
    if raw_native.shape != target_native.shape:
        raise ValueError("Development raw/target pointmap shapes differ")
    confidence = _align_confidence(confidence, raw_native)
    support = (
        torch.isfinite(raw_native).all(dim=-1)
        & torch.isfinite(target_native).all(dim=-1)
        & torch.isfinite(confidence)
        & (confidence >= run.confidence_threshold)
    )
    if any(int(support[index].sum()) < 128 for index in range(len(frame_indices))):
        raise ValueError("A development frame has fewer than 128 valid support pixels")
    return DevelopmentData(
        clip=clip.name,
        levels=levels,
        patch_start_idx=patch_start_idx,
        patch_shape=patch_shape,
        image_size=image_size,
        raw_native=raw_native,
        target_native=target_native,
        target_metric=target_metric,
        confidence=confidence,
        support=support,
        scale=scale,
        rotation=rotation,
        translation=translation,
        frame_indices=frame_indices,
        cache_path=source,
    )


def _load_pretrained_point_head(run: RunConfig) -> tuple[torch.nn.Module, tuple[int, ...]]:
    runtime = _load_v0_runtime(run.v0_config)
    maybe_add_repo_to_path(runtime["streamvggt_repo"])
    model = load_streamvggt_latent_model(
        repo_path=runtime["streamvggt_repo"],
        checkpoint_path=runtime["streamvggt_checkpoint"],
        device="cpu",
        strict=True,
    )
    head = copy.deepcopy(model.point_head).float()
    layers = tuple(int(value) for value in head.intermediate_layer_idx)
    del model
    gc.collect()
    return head, layers


def _train_point_head(
    head: torch.nn.Module,
    data: DevelopmentData,
    run: RunConfig,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, torch.Tensor], dict[str, Any]]:
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=run.learning_rate,
        weight_decay=run.weight_decay,
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_score: tuple[float, float] | None = None
    selection: dict[str, Any] = {
        "best_epoch": None,
        "best_validation_rmse": None,
        "best_validation_median": None,
        "best_validation_p90": None,
        "test_metrics_read_during_selection": 0,
    }
    curve: list[dict[str, Any]] = []
    for epoch in range(run.epochs):
        head.train()
        order = list(run.train_indices)
        random.shuffle(order)
        losses = []
        for start in range(0, len(order), run.batch_size):
            indices = order[start : start + run.batch_size]
            prediction = _forward_head(
                head,
                data.levels[:, indices],
                patch_start_idx=data.patch_start_idx,
                patch_shape=data.patch_shape,
                image_size=data.image_size,
                device=device,
            )
            target = data.target_native[indices].to(device)
            valid = data.support[indices].to(device)
            if not bool(valid.any()):
                continue
            loss = F.smooth_l1_loss(
                prediction[valid],
                target[valid],
                beta=run.point_beta,
                reduction="none",
            ).sum(dim=-1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), run.grad_clip_norm)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        head.eval()
        with torch.inference_mode():
            validation = _evaluate_native_split(
                head,
                data,
                run.validation_indices,
                run,
                device,
                metric_space=True,
            )
        row = {
            "epoch": epoch,
            "train_loss": float(sum(losses) / max(len(losses), 1)),
            "validation_rmse": validation["rmse"],
            "validation_median": validation["median"],
            "validation_p90": validation["p90"],
        }
        curve.append(row)
        score = (float(validation["rmse"]), float(validation["p90"]))
        if best_score is None or score < best_score:
            best_score = score
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in head.state_dict().items()
            }
            selection.update(
                {
                    "best_epoch": epoch,
                    "best_validation_rmse": validation["rmse"],
                    "best_validation_median": validation["median"],
                    "best_validation_p90": validation["p90"],
                }
            )
    if best_state is None:
        raise RuntimeError("PointHead training did not produce a validation checkpoint")
    return curve, best_state, selection


def _development_evaluation(
    head: torch.nn.Module,
    data: DevelopmentData,
    run: RunConfig,
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    point_head = _evaluate_native_split(
        head,
        data,
        run.validation_indices,
        run,
        device,
        metric_space=True,
    )
    raw = _metric_values(
        apply_similarity(
            data.raw_native[run.validation_indices],
            scale=data.scale,
            rotation=data.rotation,
            translation=data.translation,
        ),
        data.target_metric[run.validation_indices],
        data.support[run.validation_indices],
        run.maximum_points,
    )
    return {"raw_full_history": raw, "point_head_only": point_head}


def _evaluate_native_split(
    head: torch.nn.Module,
    data: DevelopmentData,
    indices: Sequence[int],
    run: RunConfig,
    device: torch.device,
    *,
    metric_space: bool,
) -> dict[str, Any]:
    with torch.inference_mode():
        prediction = _forward_head(
            head,
            data.levels[:, list(indices)],
            patch_start_idx=data.patch_start_idx,
            patch_shape=data.patch_shape,
            image_size=data.image_size,
            device=device,
        ).cpu()
    target = data.target_native[list(indices)]
    if metric_space:
        prediction = apply_similarity(
            prediction,
            scale=data.scale,
            rotation=data.rotation,
            translation=data.translation,
        )
        target = data.target_metric[list(indices)]
    return _metric_values(
        prediction,
        target,
        data.support[list(indices)],
        run.maximum_points,
    )


def _extract_heldout_features(run: RunConfig) -> list[HeldoutFeatures]:
    runtime = _load_v0_runtime(run.v0_config)
    maybe_add_repo_to_path(runtime["streamvggt_repo"])
    assert_processed_key_cache_equivalence()
    model = load_streamvggt_latent_model(
        repo_path=runtime["streamvggt_repo"],
        checkpoint_path=runtime["streamvggt_checkpoint"],
        device="cpu" if run.devices else run.device,
        strict=True,
    )
    runner = None
    if run.devices:
        runner = LayerShardedStreamVGGT(
            model,
            run.devices,
            selected_layer_indices=EXPECTED_DPT_LAYERS,
            amp_dtype=runtime["amp_dtype"],
        )
        for line in runner.layout_summary():
            print(f"  {line}")
    adapter = StreamVGGTLatentAdapter(
        model,
        device=run.devices[0] if run.devices else run.device,
        image_mode=runtime["image_mode"],
        dpt_layer_indices=EXPECTED_DPT_LAYERS,
        parallel_runner=runner,
    )
    results: list[HeldoutFeatures] = []
    try:
        for scene_id, manifest_path in run.heldout_scenes:
            actual, frames = _load_r3_manifest(manifest_path, scene_id)
            image_paths = [
                Path(str(row["image_path"])).expanduser().resolve() for row in frames
            ]
            if not all(path.is_file() for path in image_paths):
                raise FileNotFoundError(f"Held-out scene {scene_id} has missing RGB")
            with torch.inference_mode():
                output = adapter.extract_from_paths(
                    image_paths,
                    return_pointmap=True,
                    streaming_cache=True,
                )
            geometry = StreamVGGTWrapper._geometry_from_output(output, image_paths)
            aux = output.geometry.aux
            tokens = aux.get("stream_dpt_tokens")
            if not isinstance(tokens, list) or len(tokens) != 4:
                raise ValueError(f"Held-out scene {scene_id} lacks four DPT levels")
            levels = torch.stack(
                [value.detach().float().cpu()[0] for value in tokens], dim=0
            )
            patch_shape = tuple(int(value) for value in aux["patch_shape"])
            if int(levels.shape[-2]) - int(aux["patch_start_idx"]) != patch_shape[0] * patch_shape[1]:
                raise ValueError(f"Held-out scene {scene_id} DPT patch layout mismatch")
            results.append(
                HeldoutFeatures(
                    scene_id=actual,
                    r3_manifest=manifest_path,
                    frames=frames,
                    levels=levels,
                    patch_start_idx=int(aux["patch_start_idx"]),
                    patch_shape=patch_shape,
                    image_size=tuple(int(value) for value in geometry.processed_size),
                    raw=geometry.world_points.detach().float().cpu(),
                    confidence=geometry.confidence.detach().float().cpu(),
                )
            )
            del output
    finally:
        del adapter, runner, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results


def _evaluate_heldout(
    head: torch.nn.Module,
    item: HeldoutFeatures,
    run: RunConfig,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with torch.inference_mode():
        prediction = _forward_head(
            head,
            item.levels,
            patch_start_idx=item.patch_start_idx,
            patch_shape=item.patch_shape,
            image_size=item.image_size,
            device=device,
            batch_size=run.batch_size,
        ).cpu()
    raw = _normalize_dense(item.raw, item.image_size, channels=3)
    prediction = _normalize_dense(prediction, item.image_size, channels=3)
    confidence = _align_confidence(item.confidence, raw)
    target = _load_gt_pointmaps(
        item.r3_manifest,
        scene_id=item.scene_id,
        frame_indices=tuple(range(len(item.frames))),
        processed_size=item.image_size,
        image_mode="crop",
    )
    target = _normalize_dense(target, item.image_size, channels=3)
    valid = (
        torch.isfinite(raw).all(dim=-1)
        & torch.isfinite(prediction).all(dim=-1)
        & torch.isfinite(target).all(dim=-1)
        & torch.isfinite(confidence)
        & (confidence >= run.confidence_threshold)
    )
    if any(int(valid[index].sum()) < 128 for index in range(len(item.frames))):
        raise ValueError(f"Held-out scene {item.scene_id} has insufficient valid pixels")
    source, truth = _paired_limit(
        raw[0].reshape(-1, 3)[valid[0].reshape(-1)],
        target[0].reshape(-1, 3)[valid[0].reshape(-1)],
        max_points=30000,
    )
    scale, rotation, translation, inliers, fit_rmse = _robust_similarity(
        source, truth, min_points=128
    )
    aligned_raw = apply_similarity(raw, scale=scale, rotation=rotation, translation=translation)
    aligned_prediction = apply_similarity(
        prediction, scale=scale, rotation=rotation, translation=translation
    )
    raw_metrics, raw_frames = _metric_values_with_frames(
        aligned_raw, target, valid, run.maximum_points, branch="raw_full_history"
    )
    point_metrics, point_frames = _metric_values_with_frames(
        aligned_prediction, target, valid, run.maximum_points, branch="point_head_only"
    )
    raw_metrics.update({"alignment_scale": float(scale), "alignment_inliers": int(inliers), "alignment_fit_rmse": float(fit_rmse)})
    point_metrics.update({"alignment_scale": float(scale), "alignment_inliers": int(inliers), "alignment_fit_rmse": float(fit_rmse)})
    raw_by_frame = {row["sequence_index"]: row for row in raw_frames}
    point_by_frame = {row["sequence_index"]: row for row in point_frames}
    improved = sum(
        point_by_frame[index]["rmse"] < raw_by_frame[index]["rmse"]
        for index in raw_by_frame
    )
    summary = {
        "scene_id": item.scene_id,
        "frame_count": len(item.frames),
        "raw_rmse": raw_metrics["rmse"],
        "point_head_rmse": point_metrics["rmse"],
        "raw_median": raw_metrics["median"],
        "point_head_median": point_metrics["median"],
        "raw_p90": raw_metrics["p90"],
        "point_head_p90": point_metrics["p90"],
        "improved_frames": int(improved),
        "improved_frame_ratio": float(improved / len(raw_by_frame)),
        "decision": "GO"
        if point_metrics["rmse"] < raw_metrics["rmse"]
        and point_metrics["p90"] <= raw_metrics["p90"]
        else "NO_GO",
    }
    return summary, raw_frames + point_frames


def _forward_head(
    head: torch.nn.Module,
    levels: torch.Tensor,
    *,
    patch_start_idx: int,
    patch_shape: tuple[int, int],
    image_size: tuple[int, int],
    device: torch.device,
    batch_size: int | None = None,
) -> torch.Tensor:
    if levels.ndim != 4:
        raise ValueError(f"Expected levels [L,S,T,C], got {levels.shape}")
    frame_count = int(levels.shape[1])
    batch_size = int(batch_size or frame_count)
    output: list[torch.Tensor] = []
    layer_indices = tuple(int(value) for value in head.intermediate_layer_idx)
    for start in range(0, frame_count, batch_size):
        end = min(start + batch_size, frame_count)
        selected = levels[:, start:end].to(device)
        token_list: list[torch.Tensor | None] = [None] * (max(layer_indices) + 1)
        for level, layer_index in enumerate(layer_indices):
            token_list[layer_index] = selected[level].unsqueeze(0)
        images = torch.zeros(
            (1, end - start, 3, image_size[0], image_size[1]),
            dtype=torch.float32,
            device=device,
        )
        preds, _ = head(
            token_list,
            images=images,
            patch_start_idx=int(patch_start_idx),
            frames_chunk_size=max(1, end - start),
        )
        # Keep the graph during development training; inference callers wrap
        # this helper in torch.inference_mode() and therefore receive a
        # detached tensor naturally.
        output.append(preds[0])
    return torch.cat(output, dim=0)


def _metric_values(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    maximum_points: int,
) -> dict[str, Any]:
    errors = []
    for index in range(int(prediction.shape[0])):
        selected = _limited_indices(valid[index], maximum_points)
        errors.append(
            torch.linalg.vector_norm(
                prediction[index].reshape(-1, 3).index_select(0, selected)
                - target[index].reshape(-1, 3).index_select(0, selected),
                dim=-1,
            )
        )
    joined = torch.cat(errors)
    return {
        "evaluated_points": int(joined.numel()),
        "rmse": _rmse(joined),
        "median": float(joined.median()),
        "p90": float(torch.quantile(joined, 0.90)),
    }


def _metric_values_with_frames(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    maximum_points: int,
    *,
    branch: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    frame_rows = []
    errors = []
    for index in range(int(prediction.shape[0])):
        selected = _limited_indices(valid[index], maximum_points)
        error = torch.linalg.vector_norm(
            prediction[index].reshape(-1, 3).index_select(0, selected)
            - target[index].reshape(-1, 3).index_select(0, selected),
            dim=-1,
        )
        errors.append(error)
        frame_rows.append(
            {
                "branch": branch,
                "sequence_index": index,
                "evaluated_points": int(error.numel()),
                "rmse": _rmse(error),
                "median": float(error.median()),
                "p90": float(torch.quantile(error, 0.90)),
            }
        )
    joined = torch.cat(errors)
    return (
        {
            "evaluated_points": int(joined.numel()),
            "rmse": _rmse(joined),
            "median": float(joined.median()),
            "p90": float(torch.quantile(joined, 0.90)),
        },
        frame_rows,
    )


def _load_r3_manifest(path: Path, scene_id: str) -> tuple[str, list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf8"))
    scenes = payload.get("scenes", [])
    if len(scenes) != 1:
        raise ValueError(f"R3 manifest must contain one scene: {path}")
    scene = scenes[0]
    actual = str(scene.get("scene_id", ""))
    if actual != scene_id:
        raise ValueError(f"R3 scene mismatch: requested={scene_id}, manifest={actual}")
    frames = list(scene.get("frames", []))
    if len(frames) != 30:
        raise ValueError(f"R3 protocol requires 30 frames, got {len(frames)}")
    if any("semantic_mask" in row or "instance_mask" in row for row in frames):
        raise ValueError("Stage 2 held-out candidate generation must not consume masks")
    return actual, frames


def _load_v0_runtime(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf8")) or {}
    recovery_path = expand_storage_path(raw["recovery_config"], base=path.parent)
    recovery = yaml.safe_load(recovery_path.read_text(encoding="utf8")) or {}
    stream = recovery.get("streamvggt", {})
    return {
        "streamvggt_repo": expand_storage_path(
            stream.get("repo", "externals/streamvggt"), base=path.parent
        ),
        "streamvggt_checkpoint": expand_storage_path(
            stream["checkpoint"], base=path.parent
        ),
        "image_mode": str(stream.get("image_mode", "crop")),
        "amp_dtype": "bfloat16",
    }


def _load_run(path: str | Path, *, device_override: str | None) -> RunConfig:
    source = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf8")) or {}
    split = raw.get("split", {})
    training = raw.get("training", {})
    evaluation = raw.get("evaluation", {})
    heldout = tuple(
        (
            str(item["scene_id"]),
            expand_storage_path(item["r3_manifest"], base=source.parent),
        )
        for item in raw.get("heldout_scenes", ())
    )
    if not heldout:
        raise ValueError("Stage 2 requires at least one held-out R3 scene")
    return RunConfig(
        source_path=source,
        v0_config=expand_storage_path(
            raw.get("v0_config", "streaming_couping/configs/v0_baseline.yaml"),
            base=source.parent,
        ),
        output_dir=expand_storage_path(
            raw.get(
                "output_dir",
                "${VGGT_SAM_STORAGE_ROOT}/outputs/streaming_couping_stage2_point_head",
            ),
            base=source.parent,
        ),
        device=str(device_override or raw.get("device", "cuda:0")),
        devices=tuple(str(value) for value in raw.get("devices", ("cuda:0", "cuda:1"))),
        seed=int(raw.get("seed", 2026)),
        train_indices=tuple(int(value) for value in split.get("train_sequence_indices", range(18))),
        validation_indices=tuple(int(value) for value in split.get("validation_sequence_indices", range(18, 24))),
        epochs=int(training.get("epochs", 30)),
        batch_size=int(training.get("batch_size", 2)),
        learning_rate=float(training.get("learning_rate", 2e-4)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
        grad_clip_norm=float(training.get("grad_clip_norm", 1.0)),
        point_beta=float(training.get("point_beta", 0.05)),
        confidence_threshold=float(evaluation.get("confidence_threshold", 0.30)),
        maximum_points=int(evaluation.get("maximum_points_per_frame", 8192)),
        heldout_scenes=heldout,
    )


def _validate_split(run: RunConfig, frame_count: int) -> None:
    if frame_count != 30:
        raise ValueError(f"Stage 2 requires 30 development frames, got {frame_count}")
    if run.train_indices != tuple(range(18)) or run.validation_indices != tuple(range(18, 24)):
        raise ValueError("Stage 2 fixed split must be train=0..17 and validation=18..23")
    if run.epochs != 30 or run.batch_size != 2 or run.learning_rate != 2e-4:
        raise ValueError("Stage 2 fixed protocol changed; keep 30 epochs, batch 2, lr 2e-4")


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


def _align_confidence(confidence: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    if confidence.ndim == 4 and confidence.shape[-1] == 1:
        confidence = confidence[..., 0]
    expected = tuple(int(value) for value in points.shape[:-1])
    if tuple(int(value) for value in confidence.shape) == expected:
        return confidence.contiguous()
    if confidence.ndim == 3 and tuple(int(value) for value in confidence.shape) == (
        expected[0], expected[2], expected[1]
    ):
        return confidence.transpose(1, 2).contiguous()
    raise ValueError(f"Confidence/pointmap layout mismatch: {confidence.shape} vs {points.shape}")


def _normalize_dense(values: torch.Tensor, expected_hw: tuple[int, int], *, channels: int) -> torch.Tensor:
    tensor = values[0] if values.ndim == 5 and values.shape[0] == 1 else values
    if tensor.ndim != 4 or int(tensor.shape[-1]) != channels:
        raise ValueError(f"Expected dense [S,H,W,{channels}], got {tensor.shape}")
    height, width = expected_hw
    if tuple(int(value) for value in tensor.shape[1:3]) == (height, width):
        return tensor.contiguous()
    if tuple(int(value) for value in tensor.shape[1:3]) == (width, height):
        return tensor.transpose(1, 2).contiguous()
    raise ValueError(f"Dense output shape {tensor.shape} does not match {expected_hw}")


def _limited_indices(mask: torch.Tensor, maximum: int) -> torch.Tensor:
    selected = torch.nonzero(mask.reshape(-1), as_tuple=False)[:, 0]
    if selected.numel() <= int(maximum):
        return selected
    positions = torch.linspace(0, selected.numel() - 1, int(maximum)).long()
    return selected.index_select(0, positions)


def _rmse(values: torch.Tensor) -> float:
    return float(torch.sqrt(values.float().square().mean()))


def _serializable_run(run: RunConfig) -> dict[str, Any]:
    value = asdict(run)
    for key in ("source_path", "v0_config", "output_dir"):
        value[key] = str(value[key])
    value["heldout_scenes"] = [
        {"scene_id": scene, "r3_manifest": str(path)}
        for scene, path in run.heldout_scenes
    ]
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf8")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_copyable(path: Path, summary: Mapping[str, Any]) -> None:
    decision = summary["decision"]
    lines = [
        "===== COPYABLE_STAGE2_POINT_HEAD_BEGIN =====",
        f"revision={summary['revision']}",
        f"experiment={summary['experiment']}",
        f"development_clip={summary['development_clip']}",
        "trainable=original_pretrained_point_head_only",
        "frozen=streamvggt_aggregator_backbone_camera_head",
        "sam_inputs=0",
        "qk_pose_modified=0",
        "formal_v0_modified=0",
        f"selected_checkpoint_epoch_one_based={summary['checkpoint_selection']['best_epoch'] + 1}",
        "",
        "scene,raw_rmse,point_head_rmse,raw_p90,point_head_p90,improved_frame_ratio,decision",
    ]
    for row in summary["heldout_scenes"]:
        lines.append(
            ",".join(
                str(row[field])
                for field in (
                    "scene_id",
                    "raw_rmse",
                    "point_head_rmse",
                    "raw_p90",
                    "point_head_p90",
                    "improved_frame_ratio",
                    "decision",
                )
            )
        )
    lines.extend(
        (
            "",
            "decision=" + json.dumps(decision, sort_keys=True),
            f"summary={path.with_name('summary.json')}",
            f"checkpoint={path.with_name('point_head_only.pt')}",
            f"training_curve={path.with_name('training_curve.csv')}",
            f"heldout_frame_metrics={path.with_name('heldout_frame_metrics.csv')}",
            "===== COPYABLE_STAGE2_POINT_HEAD_END =====",
        )
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        return value.tolist()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="streaming_couping/configs/stage2_point_head_training.yaml",
    )
    parser.add_argument("--device", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
