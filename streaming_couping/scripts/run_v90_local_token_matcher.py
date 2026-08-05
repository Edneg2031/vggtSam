#!/usr/bin/env python3
"""Run V9 Stage A0/A1 local-token epipolar causality experiments.

A0 keeps exactly the cached 32 SAM local-token query locations and supplies
their continuous GT-visible historical UV.  A1 then trains only a small Q/K
correspondence projector.  StreamVGGT, SAM3.1 and the O-R1 epipolar solver are
frozen; no pose loss is back-propagated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import yaml

from streaming_couping.src.config import load_config
from streaming_couping.src.external_repos import maybe_add_repo_to_path
from streaming_couping.src.learned_pose.cache import cache_path, load_feature_cache
from streaming_couping.src.learned_pose.config import load_learned_pose_config
from streaming_couping.src.v74_temporal_protocol import (
    EXPECTED_FRAMES,
    FOLDS,
    validate_folds,
)
from streaming_couping.src.v80_pose_geometry import (
    homogeneous,
    invert_rigid,
    rotation_error_degrees,
)
from streaming_couping.src.v90_epipolar_geometry import (
    EpipolarConfig,
    LocalTokenReprojection,
    SurfaceCorrespondences,
    VisibilityConfig,
    causal_mask_history_indices,
    concatenate_correspondences,
    estimate_relative_epipolar_pose,
    local_token_reprojection_labels,
    relative_translation_direction_error_degrees,
)
from streaming_couping.src.v90_explicit_matcher import (
    MATCHER_ARCHITECTURES,
    MATCHER_EVAL_VARIANTS,
    ExplicitLocalMatcher,
    MatcherConfig,
    SoftMatchTarget,
    build_soft_match_target,
    correspondence_loss,
    probability_to_correspondences,
    sample_stream_patch_descriptors,
    uniform_match_probability,
)


SUMMARY_COLUMNS = (
    "stage",
    "fold",
    "architecture",
    "variant",
    "descriptor_source",
    "train_frames",
    "test_frames",
    "token_count",
    "parameters",
    "trainable_parameters",
    "steps",
    "best_step",
    "initial_train_loss",
    "final_train_loss",
    "train_loss_drop_percent",
    "frames",
    "active_frames",
    "inactive_frames",
    "history_edges",
    "solved_edges",
    "supervised_queries",
    "visible_queries",
    "visible_key_supported_queries",
    "accepted_correspondences",
    "perturbed_pairs",
    "pck_threshold_pixels",
    "pck_accuracy",
    "mean_epe_pixels",
    "dustbin_accuracy",
    "mean_sampson_rmse",
    "raw_edge_rotation_error_deg",
    "refined_edge_rotation_error_deg",
    "edge_rotation_gain_percent",
    "raw_edge_translation_direction_error_deg",
    "refined_edge_translation_direction_error_deg",
    "edge_translation_direction_gain_percent",
    "raw_relative_aggregate_deg",
    "refined_relative_aggregate_deg",
    "relative_aggregate_gain_percent",
    "relative_aggregate_worse_frames",
    "parameter_matched_qk",
    "control_support_exact",
    "matcher_frozen_exact",
    "fold_a0_pass",
    "all_folds_a0_pass",
    "fold_sam_causal_pass",
    "all_folds_sam_causal_pass",
)

FRAME_COLUMNS = (
    "stage",
    "fold",
    "architecture",
    "variant",
    "sequence_index",
    "frame_index",
    "history_edges",
    "solved_edges",
    "active",
    "supervised_queries",
    "visible_queries",
    "visible_key_supported_queries",
    "accepted_correspondences",
    "pck_correct",
    "epe_sum_pixels",
    "dustbin_correct",
    "dustbin_queries",
    "mean_sampson_rmse",
    "raw_edge_rotation_error_deg",
    "refined_edge_rotation_error_deg",
    "raw_edge_translation_direction_error_deg",
    "refined_edge_translation_direction_error_deg",
    "raw_relative_aggregate_deg",
    "refined_relative_aggregate_deg",
    "relative_aggregate_worse",
)

TRAIN_COLUMNS = (
    "fold",
    "architecture",
    "seed",
    "step",
    "loss",
    "cross_entropy",
    "cycle_loss",
    "supervised_queries",
    "query_projection_grad_norm",
    "key_projection_grad_norm",
)

PAIR_COLUMNS = (
    "fold",
    "split",
    "current_sequence_index",
    "current_frame_index",
    "history_sequence_index",
    "history_frame_index",
    "slot",
    "query_valid_tokens",
    "target_visible_tokens",
    "history_valid_keys",
    "visible_key_supported_tokens",
    "dustbin_target_tokens",
    "mean_nearest_key_distance_pixels",
)

EDGE_COLUMNS = (
    "stage",
    "fold",
    "architecture",
    "variant",
    "sequence_index",
    "frame_index",
    "history_sequence_index",
    "history_frame_index",
    "participating_slots",
    "slot_correspondence_counts",
    "correspondences",
    "effective_correspondences",
    "current_uv_bbox_coverage_fraction",
    "history_uv_bbox_coverage_fraction",
    "current_uv_hull_coverage_fraction",
    "history_uv_hull_coverage_fraction",
    "current_uv_covariance_ratio",
    "history_uv_covariance_ratio",
    "design_rank_ratio",
    "design_condition",
    "sampson_rmse",
    "eight_point_sampson_rmse",
    "l0_local_sampson_rmse",
    "inlier_ratio",
    "cheirality_fraction",
    "refinement_iterations",
    "initialization",
    "edge_success",
    "edge_reason",
    "raw_edge_rotation_error_deg",
    "refined_edge_rotation_error_deg",
    "rotation_improvement_deg",
    "raw_edge_translation_direction_error_deg",
    "refined_edge_translation_direction_error_deg",
    "translation_direction_improvement_deg",
    "raw_relative_aggregate_deg",
    "refined_relative_aggregate_deg",
    "relative_aggregate_improvement_deg",
    "relative_aggregate_worse",
)


@dataclass(frozen=True)
class TrainingConfig:
    device: str = "cuda:0"
    seed: int = 90
    steps: int = 800
    batch_pairs: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    log_every: int = 50


@dataclass(frozen=True)
class StageAConfig:
    source_path: Path
    data_config: Path
    output_dir: Path
    clip_name: str
    max_history: int
    visibility: VisibilityConfig
    epipolar: EpipolarConfig
    matcher: MatcherConfig
    training: TrainingConfig


@dataclass
class StageAData:
    frames: tuple[int, ...]
    image_size: tuple[int, int]
    masks: torch.Tensor
    local_features_sam: torch.Tensor
    local_features_stream: torch.Tensor
    local_uv: torch.Tensor
    local_valid: torch.Tensor
    history_bank: torch.Tensor
    world_points: torch.Tensor
    depth: torch.Tensor
    global_w2c: torch.Tensor
    intrinsics: torch.Tensor
    baseline: torch.Tensor
    target: torch.Tensor


@dataclass
class PairRecord:
    current: int
    history: int
    slot: int
    labels: LocalTokenReprojection
    target: SoftMatchTarget


@dataclass
class TrainState:
    model: ExplicitLocalMatcher
    parameters: int
    trainable_parameters: int
    steps: int
    best_step: int
    initial_loss: float
    final_loss: float


def main() -> None:
    args = _parse_args()
    config = load_stage_a_config(args.config)
    if args.output_dir:
        config = replace(
            config, output_dir=Path(args.output_dir).expanduser().resolve()
        )
    result = run_stage_a(config, stage=args.stage, resume=not args.no_resume)
    print(f"V9 Stage A result={result}")


def run_stage_a(
    config: StageAConfig, *, stage: str = "all", resume: bool = True
) -> Path:
    if stage not in {"all", "a0", "a1"}:
        raise ValueError(f"Unknown V9 Stage A stage={stage!r}.")
    _seed_everything(config.training.seed)
    payload, cache_file = _load_payload(config)
    data = _prepare_data(payload, config)
    # token_levels can be much larger than the sampled local control.  Drop
    # the cache payload before matcher training so Stage A remains lightweight.
    del payload
    positions = {frame: index for index, frame in enumerate(data.frames)}
    records_by_fold: dict[str, dict[str, list[PairRecord]]] = {}
    for fold in FOLDS:
        records_by_fold[fold.name] = {
            "train": _build_records(
                data,
                current_indices=[positions[value] for value in fold.train_frames],
                config=config,
            ),
            "test": _build_records(
                data,
                current_indices=[positions[value] for value in fold.test_frames],
                config=config,
            ),
        }

    config.output_dir.mkdir(parents=True, exist_ok=True)
    pair_rows = _pair_diagnostic_rows(records_by_fold, data=data)
    summary_rows: list[dict[str, object]] = []
    frame_rows: list[dict[str, object]] = []
    edge_rows: list[dict[str, object]] = []
    train_rows: list[dict[str, object]] = []
    for fold in FOLDS:
        rows, frames, edges = _evaluate_a0(
            fold_name=fold.name,
            test_frames=fold.test_frames,
            records=records_by_fold[fold.name]["test"],
            data=data,
            config=config,
        )
        summary_rows.append(rows)
        frame_rows.extend(frames)
        edge_rows.extend(edges)
    all_a0_pass = int(
        len(summary_rows) == len(FOLDS)
        and all(int(row["fold_a0_pass"]) for row in summary_rows)
    )
    for row in summary_rows:
        row["all_folds_a0_pass"] = all_a0_pass

    # A0 is a hard support gate.  A failed gate is a valid completed run, not
    # a shell error and not permission to train a matcher on inadequate data.
    if stage == "a0" or (stage == "all" and not all_a0_pass):
        result = _write_outputs(
            config,
            summary_rows=summary_rows,
            frame_rows=frame_rows,
            train_rows=train_rows,
            pair_rows=pair_rows,
            edge_rows=edge_rows,
            cache_file=cache_file,
            stopped_after_a0=not bool(all_a0_pass),
        )
        if not all_a0_pass:
            print("V9 A0 did not pass all folds; A1 was intentionally not trained.")
        return result
    if stage == "a1" and not all_a0_pass:
        raise RuntimeError("V9 A1 is locked because local32 A0 did not pass all folds.")

    trained: dict[tuple[str, str], TrainState] = {}
    for fold in FOLDS:
        training_records = records_by_fold[fold.name]["train"]
        for architecture in (
            "sam_match",
            "stream_patch_match",
            "sam_train_off",
        ):
            state, logs = _train_matcher(
                fold_name=fold.name,
                architecture=architecture,
                records=training_records,
                data=data,
                config=config,
                cache_file=cache_file,
                resume=resume,
            )
            trained[(fold.name, architecture)] = state
            train_rows.extend(logs)

    for fold in FOLDS:
        test_records = records_by_fold[fold.name]["test"]
        for architecture in MATCHER_ARCHITECTURES:
            variants = (
                MATCHER_EVAL_VARIANTS
                if architecture == "sam_match"
                else ("normal",)
            )
            state = trained.get((fold.name, architecture))
            for variant in variants:
                row, frames, edges = _evaluate_matcher(
                    fold_name=fold.name,
                    train_frames=fold.train_frames,
                    test_frames=fold.test_frames,
                    architecture=architecture,
                    variant=variant,
                    records=test_records,
                    data=data,
                    config=config,
                    state=state,
                )
                summary_rows.append(row)
                frame_rows.extend(frames)
                edge_rows.extend(edges)
    _annotate_a1_decisions(summary_rows)
    return _write_outputs(
        config,
        summary_rows=summary_rows,
        frame_rows=frame_rows,
        train_rows=train_rows,
        pair_rows=pair_rows,
        edge_rows=edge_rows,
        cache_file=cache_file,
        stopped_after_a0=False,
    )


def _load_payload(config: StageAConfig) -> tuple[dict[str, Any], Path]:
    learned = load_learned_pose_config(config.data_config)
    clip = next((item for item in learned.clips if item.name == config.clip_name), None)
    if clip is None:
        raise ValueError(f"V9 Stage A clip={config.clip_name!r} is not configured.")
    path = cache_path(learned, clip)
    if not path.is_file():
        raise FileNotFoundError(
            "V9 Stage A requires the retained V7.4 observation cache. "
            f"Missing: {path}"
        )
    payload = load_feature_cache(path)
    required = {
        "frame_indices",
        "image_size",
        "tracking_masks_stream",
        "sam_local_features",
        "sam_local_uv",
        "sam_local_valid",
        "token_levels",
        "patch_start_idx",
        "patch_shape",
        "target_world_points",
        "target_depth",
        "target_world_to_camera",
        "baseline_pose_encoding",
        "target_pose_encoding",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"V9 Stage A cache lacks fields={sorted(missing)}.")
    if str(payload.get("sam_version", "")) != "sam3.1":
        raise ValueError("V9 Stage A requires cached SAM3.1 local tokens.")
    if str(payload.get("instance_source", "")) != "sam31_online":
        raise ValueError("V9 Stage A requires dynamic online SAM3.1 slots.")
    return payload, path


def _prepare_data(payload: dict[str, Any], config: StageAConfig) -> StageAData:
    frames = tuple(int(value) for value in payload["frame_indices"])
    if frames != EXPECTED_FRAMES:
        raise ValueError(f"V9 Stage A requires frames 90:15:525, got {frames}.")
    validate_folds(FOLDS, available_frames=set(frames))
    learned = load_learned_pose_config(config.data_config)
    recovery = load_config(learned.recovery_config)
    maybe_add_repo_to_path(recovery.streamvggt_repo)
    from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri

    image_size = tuple(int(value) for value in payload["image_size"])
    baseline, _ = pose_encoding_to_extri_intri(
        torch.as_tensor(payload["baseline_pose_encoding"])[None].float(),
        image_size_hw=image_size,
    )
    target, intrinsics = pose_encoding_to_extri_intri(
        torch.as_tensor(payload["target_pose_encoding"])[None].float(),
        image_size_hw=image_size,
    )
    local_uv = torch.as_tensor(payload["sam_local_uv"]).float().cpu()
    local_valid = torch.as_tensor(payload["sam_local_valid"]).bool().cpu()
    sam = torch.as_tensor(payload["sam_local_features"]).float().cpu()
    if sam.ndim != 4 or local_uv.shape != (*sam.shape[:3], 2):
        raise ValueError("V9 Stage A SAM local cache shapes disagree.")
    if int(sam.shape[2]) != 32:
        raise ValueError(
            f"V9 A0 locks the actual local support to 32 tokens, got {sam.shape[2]}."
        )
    stream = sample_stream_patch_descriptors(
        torch.as_tensor(payload["token_levels"]).cpu(),
        patch_start_idx=int(payload["patch_start_idx"]),
        patch_shape=tuple(int(value) for value in payload["patch_shape"]),
        local_uv_normalized=local_uv,
    )
    masks = torch.as_tensor(payload["tracking_masks_stream"]).bool().cpu()
    world = torch.as_tensor(payload["target_world_points"]).double().cpu()
    depth = torch.as_tensor(payload["target_depth"]).double().cpu()
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    global_w2c = homogeneous(
        torch.as_tensor(payload["target_world_to_camera"]).double().cpu()
    )
    baseline = homogeneous(baseline[0].double().cpu())
    target = homogeneous(target[0].double().cpu())
    intrinsics = intrinsics[0].double().cpu()
    sequence, slots, height, width = masks.shape
    expected = {
        "sam": (sequence, slots, 32),
        "stream": (sequence, slots, 32),
        "world": (sequence, height, width, 3),
        "depth": (sequence, height, width),
        "global_w2c": (sequence, 4, 4),
        "intrinsics": (sequence, 3, 3),
        "baseline": (sequence, 4, 4),
        "target": (sequence, 4, 4),
    }
    actual = {
        "sam": sam.shape[:3],
        "stream": stream.shape[:3],
        "world": world.shape,
        "depth": depth.shape,
        "global_w2c": global_w2c.shape,
        "intrinsics": intrinsics.shape,
        "baseline": baseline.shape,
        "target": target.shape,
    }
    for name, shape in expected.items():
        if tuple(actual[name]) != tuple(shape):
            raise ValueError(
                f"V9 Stage A {name} shape={tuple(actual[name])}, expected={shape}."
            )
    return StageAData(
        frames=frames,
        image_size=image_size,
        masks=masks,
        local_features_sam=sam,
        local_features_stream=stream,
        local_uv=local_uv,
        local_valid=local_valid,
        history_bank=causal_mask_history_indices(
            masks, max_history=config.max_history
        ),
        world_points=world,
        depth=depth,
        global_w2c=global_w2c,
        intrinsics=intrinsics,
        baseline=baseline,
        target=target,
    )


def _build_records(
    data: StageAData,
    *,
    current_indices: Sequence[int],
    config: StageAConfig,
) -> list[PairRecord]:
    records = []
    for current in current_indices:
        observed = data.masks[current].flatten(1).any(dim=-1)
        for slot in range(data.masks.shape[1]):
            if not bool(observed[slot]):
                continue
            histories = [
                int(value)
                for value in data.history_bank[current, slot].tolist()
                if int(value) >= 0
            ]
            for history in histories:
                labels = local_token_reprojection_labels(
                    current_frame=current,
                    history_frame=history,
                    slot=slot,
                    local_uv_normalized=data.local_uv[current, slot],
                    local_valid=data.local_valid[current, slot],
                    masks=data.masks,
                    world_points_metric=data.world_points,
                    depth_metric=data.depth,
                    global_world_to_camera=data.global_w2c,
                    intrinsics=data.intrinsics,
                    config=config.visibility,
                )
                target = build_soft_match_target(
                    labels,
                    history_uv_normalized=data.local_uv[history, slot],
                    history_valid=data.local_valid[history, slot],
                    image_size=data.image_size,
                    config=config.matcher,
                )
                records.append(
                    PairRecord(
                        current=current,
                        history=history,
                        slot=slot,
                        labels=labels,
                        target=target,
                    )
                )
    return records


def _evaluate_a0(
    *,
    fold_name: str,
    test_frames: Sequence[int],
    records: Sequence[PairRecord],
    data: StageAData,
    config: StageAConfig,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    predictions = {
        id(record): record.labels.visible_correspondences() for record in records
    }
    frame_rows, edge_rows = _pose_frame_rows(
        stage="A0",
        fold_name=fold_name,
        architecture="local32_same_support_oracle",
        variant="continuous_gt_uv",
        test_frames=test_frames,
        records=records,
        predictions=predictions,
        match_stats=None,
        data=data,
        config=config,
    )
    summary = _summary_row(
        stage="A0",
        fold_name=fold_name,
        architecture="local32_same_support_oracle",
        variant="continuous_gt_uv",
        descriptor_source="none_gt_label_only",
        train_frames=(),
        test_frames=test_frames,
        state=None,
        frames=frame_rows,
        pck_threshold=config.matcher.pck_threshold_pixels,
        control_support_exact=1,
        matcher_frozen_exact=1,
        perturbed_pairs=0,
    )
    passed = int(
        int(summary["active_frames"]) == int(summary["frames"])
        and float(summary["refined_edge_rotation_error_deg"])
        < float(summary["raw_edge_rotation_error_deg"])
        and float(summary["refined_edge_translation_direction_error_deg"])
        < float(summary["raw_edge_translation_direction_error_deg"])
        and int(summary["relative_aggregate_worse_frames"]) == 0
    )
    summary["fold_a0_pass"] = passed
    return summary, frame_rows, edge_rows


def _train_matcher(
    *,
    fold_name: str,
    architecture: str,
    records: Sequence[PairRecord],
    data: StageAData,
    config: StageAConfig,
    cache_file: Path,
    resume: bool,
) -> tuple[TrainState, list[dict[str, object]]]:
    if architecture not in {"sam_match", "stream_patch_match", "sam_train_off"}:
        raise ValueError(f"V9 architecture={architecture!r} is not trainable.")
    if not records:
        raise ValueError(f"V9 fold={fold_name} has no causal training pairs.")
    device = _training_device(config.training.device)
    seed = int(config.training.seed) + 1009 * [fold.name for fold in FOLDS].index(fold_name)
    _seed_everything(seed)
    model = ExplicitLocalMatcher(config.matcher).to(device)
    signature = _checkpoint_signature(
        config, cache_file=cache_file, fold=fold_name, architecture=architecture
    )
    checkpoint_dir = config.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / f"{fold_name}_{architecture}.pt"
    if resume and checkpoint.is_file():
        saved = torch.load(checkpoint, map_location="cpu")
        if saved.get("signature") == signature:
            model.load_state_dict(saved["model"])
            state = TrainState(
                model=model,
                parameters=sum(parameter.numel() for parameter in model.parameters()),
                trainable_parameters=sum(
                    parameter.numel() for parameter in model.parameters() if parameter.requires_grad
                ),
                steps=int(saved["steps"]),
                best_step=int(saved["best_step"]),
                initial_loss=float(saved["initial_loss"]),
                final_loss=float(saved["final_loss"]),
            )
            print(f"V9 resumed fold={fold_name} architecture={architecture}")
            return state, []
        print(f"V9 ignored stale checkpoint={checkpoint}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.training.learning_rate),
        weight_decay=float(config.training.weight_decay),
    )
    initial_loss = _dataset_match_loss(
        model,
        records=records,
        architecture=architecture,
        data=data,
        device=device,
        batch_pairs=int(config.training.batch_pairs),
        cycle_weight=float(config.matcher.cycle_weight),
    )
    best_loss = initial_loss
    best_step = 0
    logs: list[dict[str, object]] = []
    for step in range(int(config.training.steps)):
        indices = _cyclic_batch_indices(
            count=len(records),
            batch_size=int(config.training.batch_pairs),
            step=step,
            seed=seed,
        )
        batch_records = [records[index] for index in indices]
        query, key, query_valid, key_valid, target = _training_batch(
            batch_records,
            architecture=architecture,
            data=data,
            device=device,
        )
        output = model(query, key, query_valid, key_valid)
        reverse = model(key, query, key_valid, query_valid)
        result = correspondence_loss(
            output["probability"],
            target,
            reverse_probability=reverse["probability"],
            cycle_weight=float(config.matcher.cycle_weight),
        )
        optimizer.zero_grad(set_to_none=True)
        result.loss.backward()
        query_grad = _grad_norm(model.query_projection.weight)
        key_grad = _grad_norm(model.key_projection.weight)
        optimizer.step()
        value = float(result.loss.detach().cpu())
        if value < best_loss:
            best_loss = value
            best_step = step + 1
        if step == 0 or (step + 1) % int(config.training.log_every) == 0:
            logs.append(
                {
                    "fold": fold_name,
                    "architecture": architecture,
                    "seed": seed,
                    "step": step + 1,
                    "loss": value,
                    "cross_entropy": float(result.cross_entropy.detach().cpu()),
                    "cycle_loss": float(result.cycle.detach().cpu()),
                    "supervised_queries": result.supervised_queries,
                    "query_projection_grad_norm": query_grad,
                    "key_projection_grad_norm": key_grad,
                }
            )
    # The checkpoint is the fixed final training step.  best_step is report
    # only and never selects a model using held-out data (or noisy minibatches).
    final_loss = _dataset_match_loss(
        model,
        records=records,
        architecture=architecture,
        data=data,
        device=device,
        batch_pairs=int(config.training.batch_pairs),
        cycle_weight=float(config.matcher.cycle_weight),
    )
    final_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    torch.save(
        {
            "signature": signature,
            "model": final_state,
            "steps": int(config.training.steps),
            "best_step": best_step,
            "initial_loss": initial_loss,
            "final_loss": final_loss,
        },
        checkpoint,
    )
    state = TrainState(
        model=model,
        parameters=sum(parameter.numel() for parameter in model.parameters()),
        trainable_parameters=sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        steps=int(config.training.steps),
        best_step=best_step,
        initial_loss=initial_loss,
        final_loss=final_loss,
    )
    return state, logs


def _dataset_match_loss(
    model: ExplicitLocalMatcher,
    *,
    records: Sequence[PairRecord],
    architecture: str,
    data: StageAData,
    device: torch.device,
    batch_pairs: int,
    cycle_weight: float,
) -> float:
    values = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(records), int(batch_pairs)):
            batch_records = records[start : start + int(batch_pairs)]
            query, key, query_valid, key_valid, target = _training_batch(
                batch_records,
                architecture=architecture,
                data=data,
                device=device,
            )
            forward = model(query, key, query_valid, key_valid)
            reverse = model(key, query, key_valid, query_valid)
            result = correspondence_loss(
                forward["probability"],
                target,
                reverse_probability=reverse["probability"],
                cycle_weight=float(cycle_weight),
            )
            values.append(float(result.loss.cpu()))
    model.train()
    return _finite_mean(values)


def _training_batch(
    records: Sequence[PairRecord],
    *,
    architecture: str,
    data: StageAData,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, SoftMatchTarget]:
    source = (
        data.local_features_stream
        if architecture == "stream_patch_match"
        else data.local_features_sam
    )
    query = torch.stack(
        [source[row.current, row.slot] for row in records]
    ).to(device)
    key = torch.stack(
        [source[row.history, row.slot] for row in records]
    ).to(device)
    if architecture == "sam_train_off":
        query = torch.zeros_like(query)
        key = torch.zeros_like(key)
    query_valid = torch.stack(
        [row.labels.query_valid for row in records]
    ).to(device)
    key_valid = torch.stack(
        [data.local_valid[row.history, row.slot] for row in records]
    ).to(device)
    target = SoftMatchTarget(
        probability=torch.stack([row.target.probability for row in records]).to(device),
        supervised=torch.stack([row.target.supervised for row in records]).to(device),
        visible_with_key_support=torch.stack(
            [row.target.visible_with_key_support for row in records]
        ).to(device),
        target_uv=torch.stack([row.target.target_uv for row in records]).to(device),
        nearest_key_distance_pixels=torch.stack(
            [row.target.nearest_key_distance_pixels for row in records]
        ).to(device),
    )
    return query, key, query_valid, key_valid, target


def _evaluate_matcher(
    *,
    fold_name: str,
    train_frames: Sequence[int],
    test_frames: Sequence[int],
    architecture: str,
    variant: str,
    records: Sequence[PairRecord],
    data: StageAData,
    config: StageAConfig,
    state: TrainState | None,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    if architecture == "mask_uv_uniform":
        if state is not None:
            raise ValueError("V9 uniform control must not have a trained state.")
        device = torch.device("cpu")
    else:
        if state is None:
            raise ValueError(f"V9 architecture={architecture} lacks a trained state.")
        device = next(state.model.parameters()).device
        state.model.eval()
    predictions: dict[int, SurfaceCorrespondences] = {}
    stats: dict[int, dict[str, float]] = {}
    frozen_before = (
        None
        if state is None
        else {
            name: value.detach().cpu().clone()
            for name, value in state.model.state_dict().items()
        }
    )
    support_exact = True
    with torch.no_grad():
        for record in records:
            query, key, query_valid, key_valid, history_uv = _evaluation_inputs(
                record,
                architecture=architecture,
                variant=variant,
                data=data,
            )
            support_exact = support_exact and bool(
                torch.equal(query_valid, record.labels.query_valid)
                and torch.equal(
                    key_valid, data.local_valid[record.history, record.slot]
                )
                and torch.equal(
                    history_uv, data.local_uv[record.history, record.slot]
                )
            )
            if architecture == "mask_uv_uniform":
                probability = uniform_match_probability(
                    query_valid[None], key_valid[None]
                )[0]
            else:
                output = state.model(
                    query[None].to(device),
                    key[None].to(device),
                    query_valid[None].to(device),
                    key_valid[None].to(device),
                )
                probability = output["probability"][0].cpu()
            current_uv, predicted_uv, weights, accepted = probability_to_correspondences(
                probability,
                current_uv=record.labels.current_uv,
                history_uv_normalized=history_uv,
                query_valid=query_valid,
                key_valid=key_valid,
                image_size=data.image_size,
            )
            predictions[id(record)] = SurfaceCorrespondences(
                current_frame=record.current,
                history_frame=record.history,
                slot=record.slot,
                current_uv=current_uv,
                history_uv=predicted_uv,
                weights=weights,
                depth_residual_metric=torch.empty(0, dtype=torch.float64),
                sampled_queries=int(record.labels.query_valid.sum()),
                projected_in_bounds=int(record.labels.query_valid.sum()),
                visible_queries=int(accepted.sum()),
            )
            stats[id(record)] = _match_statistics(
                record, probability=probability, accepted=accepted, config=config
            )
    frame_rows, edge_rows = _pose_frame_rows(
        stage="A1",
        fold_name=fold_name,
        architecture=architecture,
        variant=variant,
        test_frames=test_frames,
        records=records,
        predictions=predictions,
        match_stats=stats,
        data=data,
        config=config,
    )
    descriptor = {
        "sam_match": "sam31_local_descriptor",
        "stream_patch_match": "frozen_streamvggt_patch_descriptor",
        "mask_uv_uniform": "none_uniform_valid_uv",
        "sam_train_off": "sam31_eval_but_zero_during_training",
    }[architecture]
    summary = _summary_row(
        stage="A1",
        fold_name=fold_name,
        architecture=architecture,
        variant=variant,
        descriptor_source=descriptor,
        train_frames=train_frames,
        test_frames=test_frames,
        state=state,
        frames=frame_rows,
        pck_threshold=config.matcher.pck_threshold_pixels,
        control_support_exact=int(support_exact),
        matcher_frozen_exact=int(
            frozen_before is None
            or all(
                torch.equal(value, state.model.state_dict()[name].detach().cpu())
                for name, value in frozen_before.items()
            )
        ),
        perturbed_pairs=sum(
            _perturbation_applied(record, variant=variant, data=data)
            for record in records
        ),
    )
    return summary, frame_rows, edge_rows


def _evaluation_inputs(
    record: PairRecord,
    *,
    architecture: str,
    variant: str,
    data: StageAData,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    source = (
        data.local_features_stream
        if architecture == "stream_patch_match"
        else data.local_features_sam
    )
    current, history, slot = record.current, record.history, record.slot
    query_history, query_slot = current, slot
    if architecture == "sam_match" and variant == "wrong_identity":
        candidates = [
            index
            for index in range(data.local_valid.shape[1])
            if index != slot and bool(data.local_valid[current, index].any())
        ]
        if candidates:
            query_slot = candidates[0]
    elif architecture == "sam_match" and variant == "shuffle_time":
        candidates = [
            index
            for index in range(current)
            if index != history and bool(data.local_valid[index, slot].any())
        ]
        if candidates:
            query_history = candidates[0]
    # Perturb descriptor content only.  True current/history UV and valid
    # support remain fixed across normal/off/wrong-ID/shuffle/channel controls.
    query = source[query_history, query_slot].clone()
    key = source[history, slot].clone()
    query_valid = record.labels.query_valid.clone()
    key_valid = data.local_valid[history, slot].clone()
    history_uv = data.local_uv[history, slot].clone()
    if architecture == "sam_match" and variant == "sam_off":
        query.zero_()
        key.zero_()
    elif architecture == "sam_match" and variant == "channel_permute":
        query = torch.roll(query, shifts=1, dims=-1)
        key = torch.roll(key, shifts=1, dims=-1)
    return query, key, query_valid, key_valid, history_uv


def _perturbation_applied(
    record: PairRecord, *, variant: str, data: StageAData
) -> int:
    if variant in {"sam_off", "channel_permute"}:
        return 1
    if variant == "wrong_identity":
        return int(
            any(
                index != record.slot
                and bool(data.local_valid[record.current, index].any())
                for index in range(data.local_valid.shape[1])
            )
        )
    if variant == "shuffle_time":
        return int(
            any(
                index != record.history
                and bool(data.local_valid[index, record.slot].any())
                for index in range(record.current)
            )
        )
    return 0


def _match_statistics(
    record: PairRecord,
    *,
    probability: torch.Tensor,
    accepted: torch.Tensor,
    config: StageAConfig,
) -> dict[str, float]:
    target = record.target
    visible = record.labels.target_visible & record.labels.query_valid
    target_class = target.probability.argmax(dim=-1)
    predicted_class = probability.argmax(dim=-1)
    dustbin_index = probability.shape[-1] - 1
    dustbin_queries = target.supervised & target_class.eq(dustbin_index)
    dustbin_correct = dustbin_queries & predicted_class.eq(dustbin_index)
    del config
    # EPE/PCK need the predicted continuous UV retained in the Surface row and
    # are filled in _pose_frame_rows.  Class/dustbin counts are available here.
    return {
        "supervised_queries": float(target.supervised.sum()),
        "visible_queries": float(visible.sum()),
        "visible_key_supported_queries": float(
            target.visible_with_key_support.sum()
        ),
        "accepted_correspondences": float(accepted.sum()),
        "dustbin_correct": float(dustbin_correct.sum()),
        "dustbin_queries": float(dustbin_queries.sum()),
        "pck_correct": 0.0,
        "epe_sum_pixels": 0.0,
    }


def _pose_frame_rows(
    *,
    stage: str,
    fold_name: str,
    architecture: str,
    variant: str,
    test_frames: Sequence[int],
    records: Sequence[PairRecord],
    predictions: dict[int, SurfaceCorrespondences],
    match_stats: dict[int, dict[str, float]] | None,
    data: StageAData,
    config: StageAConfig,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    by_current: dict[int, list[PairRecord]] = {}
    for record in records:
        by_current.setdefault(record.current, []).append(record)
    rows = []
    edge_rows = []
    diagonal = math.hypot(*data.image_size)
    for frame_value in test_frames:
        current = data.frames.index(int(frame_value))
        current_records = by_current.get(current, [])
        by_history: dict[int, list[PairRecord]] = {}
        for record in current_records:
            by_history.setdefault(record.history, []).append(record)
        raw_rotation_rows = []
        refined_rotation_rows = []
        raw_direction_rows = []
        refined_direction_rows = []
        sampson_rows = []
        solved = 0
        for history, edge_records in sorted(by_history.items()):
            pairs = [predictions[id(record)] for record in edge_records]
            current_uv, history_uv, weights = concatenate_correspondences(pairs)
            l0_relative = data.baseline[history] @ invert_rigid(data.baseline[current])
            target_relative = data.target[history] @ invert_rigid(data.target[current])
            estimate = estimate_relative_epipolar_pose(
                current_uv,
                history_uv,
                weights,
                data.intrinsics[current],
                data.intrinsics[history],
                l0_relative,
                config=config.epipolar,
            )
            raw_r = float(rotation_error_degrees(l0_relative, target_relative))
            raw_t = relative_translation_direction_error_degrees(
                l0_relative[:3, :3], l0_relative[:3, 3], target_relative
            )
            refined_r, refined_t = raw_r, raw_t
            if estimate.success:
                solved += 1
                candidate = torch.eye(4, dtype=torch.float64)
                candidate[:3, :3] = estimate.rotation_current_to_history
                refined_r = float(rotation_error_degrees(candidate, target_relative))
                refined_t = relative_translation_direction_error_degrees(
                    estimate.rotation_current_to_history,
                    estimate.translation_current_origin_in_history,
                    target_relative,
                )
                sampson_rows.append(float(estimate.sampson_rmse))
            edge_rows.append(
                _edge_diagnostic_row(
                    stage=stage,
                    fold_name=fold_name,
                    architecture=architecture,
                    variant=variant,
                    current=current,
                    history=history,
                    edge_records=edge_records,
                    pairs=pairs,
                    current_uv=current_uv,
                    history_uv=history_uv,
                    weights=weights,
                    estimate=estimate,
                    raw_rotation=raw_r,
                    refined_rotation=refined_r,
                    raw_direction=raw_t,
                    refined_direction=refined_t,
                    data=data,
                )
            )
            raw_rotation_rows.append(raw_r)
            refined_rotation_rows.append(refined_r)
            raw_direction_rows.append(raw_t)
            refined_direction_rows.append(refined_t)

        aggregate_stats = {
            name: sum(float(item[name]) for item in (
                match_stats[id(record)] for record in current_records
            ))
            for name in (
                "supervised_queries",
                "visible_queries",
                "visible_key_supported_queries",
                "accepted_correspondences",
                "dustbin_correct",
                "dustbin_queries",
            )
        } if match_stats is not None else {
            "supervised_queries": sum(record.labels.query_count for record in current_records),
            "visible_queries": sum(record.labels.visible_count for record in current_records),
            "visible_key_supported_queries": sum(record.labels.visible_count for record in current_records),
            "accepted_correspondences": sum(predictions[id(record)].count for record in current_records),
            "dustbin_correct": 0.0,
            "dustbin_queries": 0.0,
        }
        pck_correct = (
            float(aggregate_stats["visible_queries"])
            if match_stats is None
            else 0.0
        )
        epe_sum = 0.0
        if match_stats is not None:
            for record in current_records:
                predicted = predictions[id(record)]
                # Recover per-query predictions again only for metrics; this
                # never changes the fixed solver input or its accepted rows.
                query, key, query_valid, key_valid, history_uv = _evaluation_inputs(
                    record,
                    architecture=architecture,
                    variant=variant,
                    data=data,
                )
                del query, key
                # Match accepted current UV back to fixed token rows.
                visible = record.labels.target_visible & query_valid
                for index in torch.nonzero(visible, as_tuple=False).flatten().tolist():
                    uv = record.labels.current_uv[index]
                    if predicted.count:
                        distance_to_row = torch.linalg.vector_norm(
                            predicted.current_uv - uv[None, :], dim=-1
                        )
                        row_index = int(distance_to_row.argmin())
                        found = float(distance_to_row[row_index]) < 1e-8
                    else:
                        found = False
                        row_index = 0
                    if found:
                        epe = float(
                            torch.linalg.vector_norm(
                                predicted.history_uv[row_index]
                                - record.labels.history_target_uv[index]
                            )
                        )
                        pck_correct += int(epe <= config.matcher.pck_threshold_pixels)
                        epe_sum += epe
                    else:
                        epe_sum += diagonal
        raw_r = _finite_mean(raw_rotation_rows)
        refined_r = _finite_mean(refined_rotation_rows)
        raw_t = _finite_mean(raw_direction_rows)
        refined_t = _finite_mean(refined_direction_rows)
        raw_aggregate = raw_r + raw_t
        refined_aggregate = refined_r + refined_t
        rows.append(
            {
                "stage": stage,
                "fold": fold_name,
                "architecture": architecture,
                "variant": variant,
                "sequence_index": current,
                "frame_index": int(frame_value),
                "history_edges": len(by_history),
                "solved_edges": solved,
                "active": int(solved > 0),
                **aggregate_stats,
                "pck_correct": pck_correct,
                "epe_sum_pixels": epe_sum,
                "mean_sampson_rmse": _finite_mean(sampson_rows),
                "raw_edge_rotation_error_deg": raw_r,
                "refined_edge_rotation_error_deg": refined_r,
                "raw_edge_translation_direction_error_deg": raw_t,
                "refined_edge_translation_direction_error_deg": refined_t,
                "raw_relative_aggregate_deg": raw_aggregate,
                "refined_relative_aggregate_deg": refined_aggregate,
                "relative_aggregate_worse": int(
                    refined_aggregate > raw_aggregate + 1e-12
                ),
            }
        )
    return rows, edge_rows


def _edge_diagnostic_row(
    *,
    stage: str,
    fold_name: str,
    architecture: str,
    variant: str,
    current: int,
    history: int,
    edge_records: Sequence[PairRecord],
    pairs: Sequence[SurfaceCorrespondences],
    current_uv: torch.Tensor,
    history_uv: torch.Tensor,
    weights: torch.Tensor,
    estimate,
    raw_rotation: float,
    refined_rotation: float,
    raw_direction: float,
    refined_direction: float,
    data: StageAData,
) -> dict[str, object]:
    del weights
    slot_counts = [
        (record.slot, pair.count)
        for record, pair in zip(edge_records, pairs)
        if pair.count > 0
    ]
    raw_aggregate = float(raw_rotation) + float(raw_direction)
    refined_aggregate = float(refined_rotation) + float(refined_direction)
    return {
        "stage": stage,
        "fold": fold_name,
        "architecture": architecture,
        "variant": variant,
        "sequence_index": current,
        "frame_index": data.frames[current],
        "history_sequence_index": history,
        "history_frame_index": data.frames[history],
        "participating_slots": " ".join(str(slot) for slot, _ in slot_counts),
        "slot_correspondence_counts": " ".join(
            f"{slot}:{count}" for slot, count in slot_counts
        ),
        "correspondences": int(current_uv.shape[0]),
        "effective_correspondences": float(estimate.effective_correspondences),
        "current_uv_bbox_coverage_fraction": _uv_bbox_coverage(
            current_uv, image_size=data.image_size
        ),
        "history_uv_bbox_coverage_fraction": _uv_bbox_coverage(
            history_uv, image_size=data.image_size
        ),
        "current_uv_hull_coverage_fraction": _uv_hull_coverage(
            current_uv, image_size=data.image_size
        ),
        "history_uv_hull_coverage_fraction": _uv_hull_coverage(
            history_uv, image_size=data.image_size
        ),
        "current_uv_covariance_ratio": _uv_covariance_ratio(current_uv),
        "history_uv_covariance_ratio": _uv_covariance_ratio(history_uv),
        "design_rank_ratio": float(estimate.design_rank_ratio),
        "design_condition": float(estimate.design_condition),
        "sampson_rmse": float(estimate.sampson_rmse),
        "eight_point_sampson_rmse": float(estimate.eight_point_sampson_rmse),
        "l0_local_sampson_rmse": float(estimate.l0_local_sampson_rmse),
        "inlier_ratio": float(estimate.inlier_ratio),
        "cheirality_fraction": float(estimate.cheirality_fraction),
        "refinement_iterations": int(estimate.refinement_iterations),
        "initialization": str(estimate.initialization),
        "edge_success": int(estimate.success),
        "edge_reason": str(estimate.reason),
        "raw_edge_rotation_error_deg": raw_rotation,
        "refined_edge_rotation_error_deg": refined_rotation,
        "rotation_improvement_deg": raw_rotation - refined_rotation,
        "raw_edge_translation_direction_error_deg": raw_direction,
        "refined_edge_translation_direction_error_deg": refined_direction,
        "translation_direction_improvement_deg": raw_direction - refined_direction,
        "raw_relative_aggregate_deg": raw_aggregate,
        "refined_relative_aggregate_deg": refined_aggregate,
        "relative_aggregate_improvement_deg": raw_aggregate - refined_aggregate,
        "relative_aggregate_worse": int(
            refined_aggregate > raw_aggregate + 1e-12
        ),
    }


def _uv_bbox_coverage(
    uv: torch.Tensor, *, image_size: tuple[int, int]
) -> float:
    finite = uv[torch.isfinite(uv).all(dim=-1)].double()
    if int(finite.shape[0]) < 2:
        return 0.0
    height, width = (int(value) for value in image_size)
    extent = finite.max(dim=0).values - finite.min(dim=0).values
    denominator = float(max(width - 1, 1) * max(height - 1, 1))
    return float((extent[0] * extent[1]).clamp_min(0.0)) / denominator


def _uv_hull_coverage(
    uv: torch.Tensor, *, image_size: tuple[int, int]
) -> float:
    finite = uv[torch.isfinite(uv).all(dim=-1)].double().cpu().tolist()
    points = sorted({(float(row[0]), float(row[1])) for row in finite})
    if len(points) < 3:
        return 0.0

    def cross(origin, first, second) -> float:
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (
            first[1] - origin[1]
        ) * (second[0] - origin[0])

    lower = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    hull = lower[:-1] + upper[:-1]
    area = 0.5 * abs(
        sum(
            first[0] * second[1] - second[0] * first[1]
            for first, second in zip(hull, hull[1:] + hull[:1])
        )
    )
    height, width = (int(value) for value in image_size)
    return area / float(max(width - 1, 1) * max(height - 1, 1))


def _uv_covariance_ratio(uv: torch.Tensor) -> float:
    finite = uv[torch.isfinite(uv).all(dim=-1)].double()
    if int(finite.shape[0]) < 3:
        return 0.0
    centered = finite - finite.mean(dim=0, keepdim=True)
    covariance = centered.transpose(0, 1) @ centered
    covariance = covariance / max(int(finite.shape[0]) - 1, 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
    maximum = float(eigenvalues[-1])
    return 0.0 if maximum <= 1e-12 else float(eigenvalues[0]) / maximum


def _summary_row(
    *,
    stage: str,
    fold_name: str,
    architecture: str,
    variant: str,
    descriptor_source: str,
    train_frames: Sequence[int],
    test_frames: Sequence[int],
    state: TrainState | None,
    frames: Sequence[dict[str, object]],
    pck_threshold: float,
    control_support_exact: int,
    matcher_frozen_exact: int,
    perturbed_pairs: int,
) -> dict[str, object]:
    visible = sum(float(row["visible_queries"]) for row in frames)
    dustbin_queries = sum(float(row["dustbin_queries"]) for row in frames)
    raw_r = _finite_mean(float(row["raw_edge_rotation_error_deg"]) for row in frames)
    refined_r = _finite_mean(
        float(row["refined_edge_rotation_error_deg"]) for row in frames
    )
    raw_t = _finite_mean(
        float(row["raw_edge_translation_direction_error_deg"]) for row in frames
    )
    refined_t = _finite_mean(
        float(row["refined_edge_translation_direction_error_deg"]) for row in frames
    )
    raw_aggregate = raw_r + raw_t
    refined_aggregate = refined_r + refined_t
    initial = float("nan") if state is None else state.initial_loss
    final = float("nan") if state is None else state.final_loss
    return {
        "stage": stage,
        "fold": fold_name,
        "architecture": architecture,
        "variant": variant,
        "descriptor_source": descriptor_source,
        "train_frames": " ".join(str(value) for value in train_frames),
        "test_frames": " ".join(str(value) for value in test_frames),
        "token_count": 32,
        "parameters": 0 if state is None else state.parameters,
        "trainable_parameters": 0 if state is None else state.trainable_parameters,
        "steps": 0 if state is None else state.steps,
        "best_step": 0 if state is None else state.best_step,
        "initial_train_loss": initial,
        "final_train_loss": final,
        "train_loss_drop_percent": _gain(initial, final),
        "frames": len(frames),
        "active_frames": sum(int(row["active"]) for row in frames),
        "inactive_frames": sum(not int(row["active"]) for row in frames),
        "history_edges": sum(int(row["history_edges"]) for row in frames),
        "solved_edges": sum(int(row["solved_edges"]) for row in frames),
        "supervised_queries": sum(float(row["supervised_queries"]) for row in frames),
        "visible_queries": visible,
        "visible_key_supported_queries": sum(
            float(row["visible_key_supported_queries"]) for row in frames
        ),
        "accepted_correspondences": sum(
            float(row["accepted_correspondences"]) for row in frames
        ),
        "perturbed_pairs": int(perturbed_pairs),
        "pck_threshold_pixels": pck_threshold,
        "pck_accuracy": (
            float("nan")
            if visible <= 0.0
            else sum(float(row["pck_correct"]) for row in frames) / visible
        ),
        "mean_epe_pixels": (
            float("nan")
            if visible <= 0.0
            else sum(float(row["epe_sum_pixels"]) for row in frames) / visible
        ),
        "dustbin_accuracy": (
            float("nan")
            if dustbin_queries <= 0.0
            else sum(float(row["dustbin_correct"]) for row in frames) / dustbin_queries
        ),
        "mean_sampson_rmse": _finite_mean(
            float(row["mean_sampson_rmse"]) for row in frames
        ),
        "raw_edge_rotation_error_deg": raw_r,
        "refined_edge_rotation_error_deg": refined_r,
        "edge_rotation_gain_percent": _gain(raw_r, refined_r),
        "raw_edge_translation_direction_error_deg": raw_t,
        "refined_edge_translation_direction_error_deg": refined_t,
        "edge_translation_direction_gain_percent": _gain(raw_t, refined_t),
        "raw_relative_aggregate_deg": raw_aggregate,
        "refined_relative_aggregate_deg": refined_aggregate,
        "relative_aggregate_gain_percent": _gain(raw_aggregate, refined_aggregate),
        "relative_aggregate_worse_frames": sum(
            int(row["relative_aggregate_worse"]) for row in frames
        ),
        "parameter_matched_qk": 1,
        "control_support_exact": int(control_support_exact),
        "matcher_frozen_exact": int(matcher_frozen_exact),
        "fold_a0_pass": 0,
        "all_folds_a0_pass": 0,
        "fold_sam_causal_pass": 0,
        "all_folds_sam_causal_pass": 0,
    }


def _annotate_a1_decisions(rows: list[dict[str, object]]) -> None:
    a0_pass = int(
        all(
            int(row["fold_a0_pass"])
            for row in rows
            if row["stage"] == "A0"
        )
    )
    fold_passes = []
    for fold in FOLDS:
        group = {
            (str(row["architecture"]), str(row["variant"])): row
            for row in rows
            if row["stage"] == "A1" and row["fold"] == fold.name
        }
        required = {
            ("sam_match", "normal"),
            ("stream_patch_match", "normal"),
            ("mask_uv_uniform", "normal"),
            ("sam_train_off", "normal"),
            *(("sam_match", value) for value in MATCHER_EVAL_VARIANTS[1:]),
        }
        missing = required - set(group)
        if missing:
            raise ValueError(f"V9 fold={fold.name} lacks controls={sorted(missing)}.")
        candidate = group[("sam_match", "normal")]
        controls = [
            group[("stream_patch_match", "normal")],
            group[("mask_uv_uniform", "normal")],
            group[("sam_train_off", "normal")],
        ]
        perturbations = [
            group[("sam_match", value)] for value in MATCHER_EVAL_VARIANTS[1:]
        ]
        perturbations_applied = all(
            int(row["perturbed_pairs"]) > 0 for row in perturbations
        )
        parameter_matched = int(
            int(candidate["parameters"])
            == int(group[("stream_patch_match", "normal")]["parameters"])
            == int(group[("sam_train_off", "normal")]["parameters"])
        )
        for row in group.values():
            row["parameter_matched_qk"] = parameter_matched
        candidate_pose = float(candidate["refined_relative_aggregate_deg"])
        candidate_epe = float(candidate["mean_epe_pixels"])
        candidate_pck = float(candidate["pck_accuracy"])
        candidate_rotation = float(candidate["refined_edge_rotation_error_deg"])
        candidate_direction = float(
            candidate["refined_edge_translation_direction_error_deg"]
        )
        passed = int(
            a0_pass
            and parameter_matched
            and all(int(row["control_support_exact"]) for row in group.values())
            and all(int(row["matcher_frozen_exact"]) for row in group.values())
            and perturbations_applied
            and float(candidate["final_train_loss"])
            < float(candidate["initial_train_loss"])
            and candidate_rotation < float(candidate["raw_edge_rotation_error_deg"])
            and candidate_direction
            < float(candidate["raw_edge_translation_direction_error_deg"])
            and candidate_pose < float(candidate["raw_relative_aggregate_deg"])
            and all(
                candidate_rotation < float(row["refined_edge_rotation_error_deg"])
                and candidate_direction
                < float(row["refined_edge_translation_direction_error_deg"])
                for row in controls
            )
            and all(candidate_pose < float(row["refined_relative_aggregate_deg"]) for row in controls)
            and all(candidate_epe < float(row["mean_epe_pixels"]) for row in controls)
            and all(candidate_pck > float(row["pck_accuracy"]) for row in controls)
            and all(candidate_pose < float(row["refined_relative_aggregate_deg"]) for row in perturbations)
            and all(candidate_epe < float(row["mean_epe_pixels"]) for row in perturbations)
            and all(candidate_pck > float(row["pck_accuracy"]) for row in perturbations)
            and int(candidate["relative_aggregate_worse_frames"]) == 0
        )
        candidate["fold_sam_causal_pass"] = passed
        fold_passes.append(passed)
    all_pass = int(len(fold_passes) == len(FOLDS) and all(fold_passes))
    for row in rows:
        row["all_folds_a0_pass"] = a0_pass
        row["all_folds_sam_causal_pass"] = all_pass


def _write_outputs(
    config: StageAConfig,
    *,
    summary_rows: Sequence[dict[str, object]],
    frame_rows: Sequence[dict[str, object]],
    train_rows: Sequence[dict[str, object]],
    pair_rows: Sequence[dict[str, object]],
    edge_rows: Sequence[dict[str, object]],
    cache_file: Path,
    stopped_after_a0: bool,
) -> Path:
    summary = config.output_dir / "v90_stage_a_summary.csv"
    frames = config.output_dir / "v90_stage_a_frames.csv"
    training = config.output_dir / "v90_stage_a_training.csv"
    pairs = config.output_dir / "v90_stage_a_pairs.csv"
    edges = config.output_dir / "v90_stage_a_edges.csv"
    problem_edges = config.output_dir / "v90_stage_a_problem_edges.csv"
    _write_csv(summary, summary_rows, SUMMARY_COLUMNS)
    _write_csv(frames, frame_rows, FRAME_COLUMNS)
    _write_csv(training, train_rows, TRAIN_COLUMNS, allow_empty=True)
    _write_csv(pairs, pair_rows, PAIR_COLUMNS)
    _write_csv(edges, edge_rows, EDGE_COLUMNS)
    problem_frame_keys = {
        (
            row["stage"],
            row["fold"],
            row["architecture"],
            row["variant"],
            row["frame_index"],
        )
        for row in frame_rows
        if int(row["relative_aggregate_worse"])
    }
    selected_problem_edges = [
        row
        for row in edge_rows
        if (
            row["stage"],
            row["fold"],
            row["architecture"],
            row["variant"],
            row["frame_index"],
        )
        in problem_frame_keys
    ]
    _write_csv(
        problem_edges,
        selected_problem_edges,
        EDGE_COLUMNS,
        allow_empty=True,
    )
    (config.output_dir / "v90_stage_a_decision.md").write_text(
        _decision_markdown(summary_rows, stopped_after_a0=stopped_after_a0),
        encoding="utf8",
    )
    metadata = {
        "experiment": "V9 Stage A local32 oracle then explicit correspondence matcher",
        "config": _jsonable_config(config),
        "cache": {
            "path": str(cache_file),
            "size_bytes": cache_file.stat().st_size,
            "mtime_ns": cache_file.stat().st_mtime_ns,
        },
        "a0_hard_stop": bool(stopped_after_a0),
        "trained_pose_model": False,
        "pose_loss_used": False,
        "forbidden_inputs_confirmed_absent": [
            "predicted depth",
            "predicted pointmap",
            "camera hidden",
            "frame index feature",
            "learned pose head",
            "GT-error fallback",
        ],
        "outputs": {
            "summary": str(summary),
            "frames": str(frames),
            "training": str(training),
            "pairs": str(pairs),
            "edges": str(edges),
            "problem_edges": str(problem_edges),
        },
    }
    (config.output_dir / "v90_stage_a_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf8",
    )
    print("V9 STAGE A COPY THIS CSV")
    print(summary.read_text(encoding="utf8").rstrip())
    return summary


def _pair_diagnostic_rows(
    records_by_fold: dict[str, dict[str, list[PairRecord]]],
    *,
    data: StageAData,
) -> list[dict[str, object]]:
    rows = []
    for fold in FOLDS:
        for split in ("train", "test"):
            for record in records_by_fold[fold.name][split]:
                finite_distance = record.target.nearest_key_distance_pixels[
                    torch.isfinite(record.target.nearest_key_distance_pixels)
                    & record.labels.target_visible
                ]
                target_class = record.target.probability.argmax(dim=-1)
                rows.append(
                    {
                        "fold": fold.name,
                        "split": split,
                        "current_sequence_index": record.current,
                        "current_frame_index": data.frames[record.current],
                        "history_sequence_index": record.history,
                        "history_frame_index": data.frames[record.history],
                        "slot": record.slot,
                        "query_valid_tokens": record.labels.query_count,
                        "target_visible_tokens": record.labels.visible_count,
                        "history_valid_keys": int(
                            data.local_valid[record.history, record.slot].sum()
                        ),
                        "visible_key_supported_tokens": int(
                            record.target.visible_with_key_support.sum()
                        ),
                        "dustbin_target_tokens": int(
                            (
                                record.target.supervised
                                & target_class.eq(record.target.probability.shape[-1] - 1)
                            ).sum()
                        ),
                        "mean_nearest_key_distance_pixels": (
                            float("nan")
                            if not finite_distance.numel()
                            else float(finite_distance.mean())
                        ),
                    }
                )
    return rows


def _decision_markdown(
    rows: Sequence[dict[str, object]], *, stopped_after_a0: bool
) -> str:
    a0_rows = [row for row in rows if row["stage"] == "A0"]
    all_a0 = int(bool(a0_rows) and all(int(row["fold_a0_pass"]) for row in a0_rows))
    a1_rows = [
        row
        for row in rows
        if row["stage"] == "A1"
        and row["architecture"] == "sam_match"
        and row["variant"] == "normal"
    ]
    all_sam = int(
        bool(a1_rows) and all(int(row["fold_sam_causal_pass"]) for row in a1_rows)
    )
    lines = [
        "# V9 Stage A local-token correspondence decision",
        "",
        "No pose model is trained. SAM3.1 and StreamVGGT are frozen.",
        "All pose metrics are relative edge rotation/translation direction from the fixed O-R1 solver.",
        "",
        f"- local32 same-support A0 all-fold pass: `{all_a0}`",
        f"- explicit SAM matcher all-fold causal pass: `{all_sam}`",
        f"- stopped before matcher training: `{int(stopped_after_a0)}`",
        "",
        "| stage | fold | architecture | variant | active | PCK | EPE px | R gain | t-dir gain | worse | pass |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        pass_value = (
            row["fold_a0_pass"]
            if row["stage"] == "A0"
            else row["fold_sam_causal_pass"]
        )
        lines.append(
            f"| {row['stage']} | {row['fold']} | {row['architecture']} | {row['variant']} "
            f"| {row['active_frames']}/{row['frames']} | {float(row['pck_accuracy']):.6g} "
            f"| {float(row['mean_epe_pixels']):.6g} "
            f"| {float(row['edge_rotation_gain_percent']):.6g} "
            f"| {float(row['edge_translation_direction_gain_percent']):.6g} "
            f"| {row['relative_aggregate_worse_frames']} | {pass_value} |"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- A0=0: fixed 32-token support is insufficient; A1 is intentionally not trained.",
            "- A0=1, SAM pass=0: solver/support is viable but cached SAM descriptors do not establish causal pose benefit.",
            "- SAM pass=1: SAM local descriptors beat patch/uniform/trained-off and all perturbations on every temporal fold.",
            "- No result here supports metric center or absolute trajectory claims.",
            "",
        ]
    )
    return "\n".join(lines)


def load_stage_a_config(path: str | Path) -> StageAConfig:
    source = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf8")) or {}
    repository = source.parents[2]
    data_config = _resolve_path(
        repository,
        raw.get("data_config", "streaming_couping/configs/v74_temporal_data.yaml"),
    )
    output_dir = _resolve_path(
        repository,
        raw.get(
            "output_dir",
            "outputs/streaming_couping_v90_epipolar_token_stage_a",
        ),
    )
    visibility = raw.get("visibility", {})
    solver = raw.get("solver", {})
    matcher = raw.get("matcher", {})
    training = raw.get("training", {})
    config = StageAConfig(
        source_path=source,
        data_config=data_config,
        output_dir=output_dir,
        clip_name=str(raw.get("clip_name", "00a231a370_90_525_step15_37_68_54")),
        max_history=int(raw.get("max_history", 2)),
        visibility=VisibilityConfig(
            max_queries_per_instance=32,
            depth_tolerance_metric=float(visibility.get("depth_tolerance_metric", 0.03)),
            relative_depth_tolerance=float(visibility.get("relative_depth_tolerance", 0.01)),
        ),
        epipolar=EpipolarConfig(
            min_correspondences=int(solver.get("min_correspondences", 8)),
            min_design_rank_ratio=float(solver.get("min_design_rank_ratio", 1e-7)),
            min_cheirality_fraction=float(solver.get("min_cheirality_fraction", 0.50)),
            refinement_iterations=int(solver.get("refinement_iterations", 5)),
            refinement_huber_delta=float(solver.get("refinement_huber_delta", 0.002)),
            refinement_damping=float(solver.get("refinement_damping", 1e-6)),
            refinement_step_epsilon=float(solver.get("refinement_step_epsilon", 1e-6)),
            max_refinement_step=float(solver.get("max_refinement_step", 0.10)),
            cheirality_max_points=int(solver.get("cheirality_max_points", 128)),
        ),
        matcher=MatcherConfig(
            canonical_dim=int(matcher.get("canonical_dim", 256)),
            projection_dim=int(matcher.get("projection_dim", 64)),
            temperature=float(matcher.get("temperature", 0.07)),
            target_sigma_pixels=float(matcher.get("target_sigma_pixels", 6.0)),
            target_radius_pixels=float(matcher.get("target_radius_pixels", 12.0)),
            pck_threshold_pixels=float(matcher.get("pck_threshold_pixels", 8.0)),
            cycle_weight=float(matcher.get("cycle_weight", 0.10)),
        ),
        training=TrainingConfig(
            device=str(training.get("device", "cuda:0")),
            seed=int(training.get("seed", 90)),
            steps=int(training.get("steps", 800)),
            batch_pairs=int(training.get("batch_pairs", 16)),
            learning_rate=float(training.get("learning_rate", 3e-4)),
            weight_decay=float(training.get("weight_decay", 1e-4)),
            log_every=int(training.get("log_every", 50)),
        ),
    )
    _validate_config(config)
    return config


def _validate_config(config: StageAConfig) -> None:
    if config.max_history != 2:
        raise ValueError("V9 Stage A locks max_history=2.")
    if config.epipolar.min_correspondences != 8:
        raise ValueError("V9 Stage A locks calibrated eight-point estimation.")
    config.matcher.validate()
    training = config.training
    if training.steps < 1 or training.batch_pairs < 1 or training.log_every < 1:
        raise ValueError("V9 training counts must be positive.")
    if training.learning_rate <= 0.0 or training.weight_decay < 0.0:
        raise ValueError("V9 optimizer settings are invalid.")


def _write_csv(
    path: Path,
    rows: Sequence[dict[str, object]],
    columns: Sequence[str],
    *,
    allow_empty: bool = False,
) -> None:
    if not rows and not allow_empty:
        raise ValueError(f"Refusing to write empty V9 Stage A CSV: {path.name}")
    expected = set(columns)
    for index, row in enumerate(rows):
        if set(row) != expected:
            raise ValueError(
                f"V9 Stage A CSV {path.name} row={index} mismatch: "
                f"missing={sorted(expected - set(row))}, extra={sorted(set(row) - expected)}"
            )
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _cyclic_batch_indices(
    *, count: int, batch_size: int, step: int, seed: int
) -> list[int]:
    generator = random.Random(int(seed) + int(step))
    if count >= batch_size:
        return generator.sample(range(count), batch_size)
    return [generator.randrange(count) for _ in range(batch_size)]


def _checkpoint_signature(
    config: StageAConfig,
    *,
    cache_file: Path,
    fold: str,
    architecture: str,
) -> str:
    value = {
        "schema": 1,
        "config_sha256": hashlib.sha256(config.source_path.read_bytes()).hexdigest(),
        "cache_size": cache_file.stat().st_size,
        "cache_mtime_ns": cache_file.stat().st_mtime_ns,
        "fold": fold,
        "architecture": architecture,
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _training_device(value: str) -> torch.device:
    requested = torch.device(value)
    if requested.type == "cuda" and not torch.cuda.is_available():
        print("V9 CUDA is unavailable; matcher training falls back to CPU")
        return torch.device("cpu")
    return requested


def _grad_norm(parameter: torch.Tensor) -> float:
    if parameter.grad is None:
        return 0.0
    return float(torch.linalg.vector_norm(parameter.grad.detach()).cpu())


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _finite_mean(values: Iterable[float]) -> float:
    rows = [float(value) for value in values if math.isfinite(float(value))]
    return float("nan") if not rows else sum(rows) / len(rows)


def _gain(initial: float, final: float) -> float:
    if not math.isfinite(initial) or not math.isfinite(final) or initial <= 1e-12:
        return 0.0
    return 100.0 * (initial - final) / initial


def _resolve_path(repository: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repository / path).resolve()


def _jsonable_config(config: StageAConfig) -> dict[str, object]:
    return {
        "source_path": str(config.source_path),
        "data_config": str(config.data_config),
        "output_dir": str(config.output_dir),
        "clip_name": config.clip_name,
        "max_history": config.max_history,
        "visibility": asdict(config.visibility),
        "epipolar": asdict(config.epipolar),
        "matcher": asdict(config.matcher),
        "training": asdict(config.training),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="streaming_couping/configs/v90_local_token_matcher.yaml",
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--stage", choices=("all", "a0", "a1"), default="all")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
