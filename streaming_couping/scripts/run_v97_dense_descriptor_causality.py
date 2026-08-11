#!/usr/bin/env python3
"""Run V9.7 dense SAM descriptor correspondence causality experiment.

SAM3.1 and StreamVGGT are frozen.  A small parameter-matched matcher is
trained only from 2D correspondence labels.  Its predicted continuous matches
are passed to the frozen O-R1 relative-pose solver; no pose model or pose loss
exists in this experiment.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable, Sequence

import torch
import yaml

from streaming_couping.scripts.run_v90_local_token_matcher import (
    StageAConfig,
    load_stage_a_config,
)
from streaming_couping.scripts.run_v92_support_factorization import (
    SupportData,
    _build_support_records,
    _load_support_payload,
    _prepare_support_data,
    load_support_config,
)
from streaming_couping.scripts.run_v93_quantization_tolerance import (
    load_diagnostic_config,
)
from streaming_couping.scripts.run_v95_spatial_support_scope import (
    _frame_edge_map,
    _locked_edge_indices,
    load_spatial_scope_config,
)
from streaming_couping.scripts.run_v96_dense_grid_upper_bound import (
    load_dense_grid_config,
)
from streaming_couping.src.v74_temporal_protocol import (
    EXPECTED_FRAMES,
    FOLDS,
)
from streaming_couping.src.v80_pose_geometry import (
    invert_rigid,
    rotation_error_degrees,
)
from streaming_couping.src.v90_epipolar_geometry import (
    SurfaceCorrespondences,
    estimate_relative_epipolar_pose,
    local_token_reprojection_labels,
    relative_translation_direction_error_degrees,
)
from streaming_couping.src.v95_spatial_support import (
    concatenate_surface_rows,
    take_surface,
    uv_hull_coverage,
)
from streaming_couping.src.v96_dense_grid_decoder import (
    current_farthest_indices,
    dense_grid_normalized,
)
from streaming_couping.src.v97_dense_descriptor_matcher import (
    DENSE_ARCHITECTURES,
    SAM_EVAL_VARIANTS,
    DenseMatchTarget,
    DenseMatcherConfig,
    DenseSubgridMatcher,
    build_dense_match_target,
    coordinate_descriptors,
    decode_dense_matches,
    dense_match_loss,
    deterministic_channel_permutation,
)


SUMMARY_COLUMNS = (
    "fold",
    "architecture",
    "variant",
    "descriptor_source",
    "train_frames",
    "test_frames",
    "grid_size",
    "query_selection",
    "history_key_count",
    "parameters",
    "trainable_parameters",
    "steps",
    "initial_train_loss",
    "final_train_loss",
    "train_loss_drop_percent",
    "frames",
    "active_frames",
    "inactive_frames",
    "mean_queries",
    "accepted_correspondences",
    "acceptance_fraction",
    "pck_threshold_pixels",
    "pck_accuracy",
    "mean_epe_pixels",
    "raw_rotation_error_deg",
    "refined_rotation_error_deg",
    "rotation_gain_percent",
    "raw_translation_direction_error_deg",
    "refined_translation_direction_error_deg",
    "translation_direction_gain_percent",
    "raw_relative_aggregate_deg",
    "refined_relative_aggregate_deg",
    "relative_aggregate_gain_percent",
    "relative_aggregate_worse_frames",
    "parameter_matched",
    "control_support_exact",
    "matcher_frozen_exact",
    "perturbed_pairs",
    "fold_sam_causal_pass",
    "all_folds_sam_causal_pass",
)

FRAME_COLUMNS = (
    "fold",
    "architecture",
    "variant",
    "sequence_index",
    "frame_index",
    "history_sequence_index",
    "history_frame_index",
    "queries",
    "accepted_correspondences",
    "active",
    "solver_reason",
    "pck_correct",
    "epe_sum_pixels",
    "current_hull_coverage",
    "predicted_history_hull_coverage",
    "raw_rotation_error_deg",
    "refined_rotation_error_deg",
    "raw_translation_direction_error_deg",
    "refined_translation_direction_error_deg",
    "relative_aggregate_worse",
)

TRAIN_COLUMNS = (
    "fold",
    "architecture",
    "seed",
    "step",
    "loss",
    "classification_loss",
    "offset_loss",
    "supervised_queries",
    "visible_queries",
    "query_projection_grad_norm",
    "key_projection_grad_norm",
    "offset_head_grad_norm",
)


@dataclass(frozen=True)
class DenseTrainingConfig:
    device: str
    seed: int
    steps: int
    batch_pairs: int
    learning_rate: float
    weight_decay: float
    log_every: int


@dataclass(frozen=True)
class V97Config:
    source_path: Path
    v96_config: Path
    data_config: Path
    output_dir: Path
    dense_cache_path: Path
    clip_name: str
    grid_size: tuple[int, int]
    cache_device: str
    cache_batch_size: int
    training_query_count: int
    training_history_offsets: tuple[int, ...]
    matcher: DenseMatcherConfig
    training: DenseTrainingConfig

    @property
    def expected_frames(self) -> tuple[int, ...]:
        return EXPECTED_FRAMES


@dataclass
class DenseData:
    support: SupportData
    grid_uv_normalized: torch.Tensor
    grid_uv_pixels: torch.Tensor
    sam_features: torch.Tensor
    stream_features: torch.Tensor
    coordinate_features: torch.Tensor


@dataclass
class DenseRecord:
    current: int
    history: int
    query_grid_indices: torch.Tensor
    correspondences: SurfaceCorrespondences


@dataclass
class TrainState:
    model: DenseSubgridMatcher
    parameters: int
    trainable_parameters: int
    steps: int
    initial_loss: float
    final_loss: float


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="streaming_couping/configs/v97_dense_descriptor_causality.yaml",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dense-cache-path", default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    config = load_v97_config(args.config)
    if args.output_dir:
        config = replace(
            config, output_dir=Path(args.output_dir).expanduser().resolve()
        )
    if args.dense_cache_path:
        config = replace(
            config,
            dense_cache_path=Path(args.dense_cache_path).expanduser().resolve(),
        )
    result = run_v97(config, resume=not bool(args.no_resume))
    print(f"V9.7 dense descriptor result={result}")


def run_v97(config: V97Config, *, resume: bool = True) -> Path:
    _seed_everything(config.training.seed)
    v96 = load_dense_grid_config(config.v96_config)
    v95 = load_spatial_scope_config(v96.v95_config)
    v93 = load_diagnostic_config(v95.v93_config)
    support_config = load_support_config(v93.support_config)
    stage_config = load_stage_a_config(support_config.stage_a_config)
    stage_config = replace(
        stage_config,
        data_config=config.data_config,
        output_dir=config.output_dir,
    )
    payload, source_cache = _load_support_payload(support_config)
    support = _prepare_support_data(
        payload, config=support_config, stage_config=stage_config
    )
    del payload
    dense_payload = _torch_load(config.dense_cache_path)
    data = _prepare_dense_data(dense_payload, support=support, config=config)
    del dense_payload

    positions = {frame: index for index, frame in enumerate(support.frames)}
    instance_records = {
        fold.name: _build_support_records(
            support,
            current_indices=[positions[value] for value in fold.test_frames],
            query_token_count=support_config.query_token_count,
            stage_config=stage_config,
        )
        for fold in FOLDS
    }
    fixed_edges = _locked_edge_indices(v95, support)
    evaluation_records = {
        fold.name: _build_evaluation_records(
            fold_name=fold.name,
            fixed_edges=fixed_edges[fold.name],
            instance_records=instance_records[fold.name],
            data=data,
            stage_config=stage_config,
            query_count=None,
        )
        for fold in FOLDS
    }
    training_records = {
        fold.name: _build_training_records(
            [positions[value] for value in fold.train_frames],
            data=data,
            stage_config=stage_config,
            config=config,
        )
        for fold in FOLDS
    }

    states: dict[tuple[str, str], TrainState] = {}
    train_rows: list[dict[str, object]] = []
    for fold in FOLDS:
        for architecture in DENSE_ARCHITECTURES:
            state, logs = _train_matcher(
                fold_name=fold.name,
                architecture=architecture,
                records=training_records[fold.name],
                data=data,
                config=config,
                source_cache=source_cache,
                resume=resume,
            )
            states[(fold.name, architecture)] = state
            train_rows.extend(logs)

    summaries: list[dict[str, object]] = []
    frames: list[dict[str, object]] = []
    for fold in FOLDS:
        for architecture in DENSE_ARCHITECTURES:
            variants = SAM_EVAL_VARIANTS if architecture == "sam_dense" else ("normal",)
            for variant in variants:
                summary, rows = _evaluate_model(
                    fold_name=fold.name,
                    train_frames=fold.train_frames,
                    test_frames=fold.test_frames,
                    architecture=architecture,
                    variant=variant,
                    state=states[(fold.name, architecture)],
                    records=evaluation_records[fold.name],
                    data=data,
                    stage_config=stage_config,
                    config=config,
                )
                summaries.append(summary)
                frames.extend(rows)
    _annotate_decisions(summaries)
    _validate_outputs(summaries, frames)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = config.output_dir / "v97_dense_descriptor_summary.csv"
    frames_path = config.output_dir / "v97_dense_descriptor_frames.csv"
    train_path = config.output_dir / "v97_dense_descriptor_training.csv"
    decision_path = config.output_dir / "v97_dense_descriptor_decision.md"
    metadata_path = config.output_dir / "v97_dense_descriptor_metadata.json"
    _write_csv(summary_path, summaries, SUMMARY_COLUMNS)
    _write_csv(frames_path, frames, FRAME_COLUMNS)
    _write_csv(train_path, train_rows, TRAIN_COLUMNS)
    decision_path.write_text(_decision_markdown(summaries), encoding="utf8")
    metadata_path.write_text(
        json.dumps(
            {
                "experiment": "V9.7 dense descriptor correspondence causality",
                "config": _jsonable_config(config),
                "source_cache": _file_provenance(source_cache),
                "dense_cache": _file_provenance(config.dense_cache_path),
                "fixed_history_edges": _frame_edge_map(fixed_edges, support),
                "training_history_policy": "strictly causal previous 1/2 sampled frames",
                "query_support": "current-only FPS on GT-visible full 72x72 grid",
                "history_support": "all 5184 actual cached grid descriptors",
                "dynamic_instance_usage": "inherits only the V9.6 equal-count budget; masks/slots do not gate dense matching",
                "trains_pose_model": False,
                "uses_pose_loss": False,
                "matcher_supervision": "GT continuous 2D correspondence only",
                "pose_solver": "frozen O-R1 calibrated relative epipolar solver",
                "descriptor_claim_scope": "same-scene temporal future relative rotation and translation direction",
                "metric_center_or_absolute_trajectory_claim": False,
                "gt_visibility_required_at_evaluation": True,
                "outputs": {
                    "summary": str(summary_path),
                    "frames": str(frames_path),
                    "training": str(train_path),
                    "decision": str(decision_path),
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf8",
    )
    print(f"V9.7 wrote summary={summary_path}")
    print(f"V9.7 wrote decision={decision_path}")
    return summary_path


def _prepare_dense_data(
    payload: dict[str, Any], *, support: SupportData, config: V97Config
) -> DenseData:
    required = {
        "complete",
        "frame_indices",
        "image_size",
        "grid_size",
        "grid_uv_normalized",
        "sam_dense_features",
        "stream_dense_features",
        "sam_source",
        "sam_version",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"V9.7 dense cache lacks fields={sorted(missing)}.")
    if not payload["complete"]:
        raise ValueError("V9.7 dense cache is incomplete.")
    if tuple(int(value) for value in payload["frame_indices"]) != support.frames:
        raise ValueError("V9.7 dense/source cache frame order differs.")
    if tuple(int(value) for value in payload["image_size"]) != support.image_size:
        raise ValueError("V9.7 dense/source image sizes differ.")
    if tuple(int(value) for value in payload["grid_size"]) != config.grid_size:
        raise ValueError("V9.7 dense cache grid size differs from protocol.")
    if str(payload["sam_source"]) != "detector_fpn2":
        raise ValueError("V9.7 requires dense detector_fpn2 descriptors.")
    if str(payload["sam_version"]) != "sam3.1":
        raise ValueError("V9.7 requires SAM3.1 descriptors.")
    grid = torch.as_tensor(payload["grid_uv_normalized"]).float().cpu()
    sam = torch.as_tensor(payload["sam_dense_features"]).float().cpu()
    stream = torch.as_tensor(payload["stream_dense_features"]).float().cpu()
    tokens = config.grid_size[0] * config.grid_size[1]
    expected = (len(support.frames), tokens)
    if tuple(sam.shape[:2]) != expected or tuple(stream.shape[:2]) != expected:
        raise ValueError("V9.7 dense descriptor tensors have the wrong shape.")
    if sam.shape[-1] != config.matcher.canonical_dim or stream.shape[-1] != config.matcher.canonical_dim:
        raise ValueError(
            "V9.7 dense cache was not parameter-free canonicalized to "
            f"{config.matcher.canonical_dim} channels."
        )
    if grid.shape != (tokens, 2):
        raise ValueError("V9.7 dense grid UV has the wrong shape.")
    image_h, image_w = support.image_size
    grid_pixels = torch.stack(
        [
            (grid[:, 0].double() + 1.0) * 0.5 * max(image_w - 1, 1),
            (grid[:, 1].double() + 1.0) * 0.5 * max(image_h - 1, 1),
        ],
        dim=-1,
    )
    coordinate = coordinate_descriptors(
        grid, channels=config.matcher.canonical_dim
    )
    return DenseData(support, grid, grid_pixels, sam, stream, coordinate)


def _build_evaluation_records(
    *,
    fold_name: str,
    fixed_edges: dict[int, int],
    instance_records,
    data: DenseData,
    stage_config: StageAConfig,
    query_count: int | None,
) -> list[DenseRecord]:
    records = []
    full_masks = torch.ones_like(data.support.masks[:, :1])
    grid_valid = torch.ones(len(data.grid_uv_normalized), dtype=torch.bool)
    for current, history in sorted(fixed_edges.items()):
        local_rows = [
            record.labels.visible_correspondences()
            for record in instance_records
            if record.current == current and record.history == history
        ]
        instance_support = concatenate_surface_rows(
            local_rows, current_frame=current, history_frame=history
        )
        target_count = instance_support.count if query_count is None else int(query_count)
        record = _full_grid_record(
            current=current,
            history=history,
            target_count=target_count,
            full_masks=full_masks,
            data=data,
            stage_config=stage_config,
        )
        if record.correspondences.count != target_count:
            raise ValueError(
                f"V9.7 fold={fold_name} frame={data.support.frames[current]} "
                f"queries={record.correspondences.count}, required={target_count}."
            )
        records.append(record)
    return records


def _build_training_records(
    current_indices: Sequence[int],
    *,
    data: DenseData,
    stage_config: StageAConfig,
    config: V97Config,
) -> list[DenseRecord]:
    output = []
    full_masks = torch.ones_like(data.support.masks[:, :1])
    for current in current_indices:
        for offset in config.training_history_offsets:
            history = current - int(offset)
            if history < 0:
                continue
            record = _full_grid_record(
                current=current,
                history=history,
                target_count=config.training_query_count,
                full_masks=full_masks,
                data=data,
                stage_config=stage_config,
            )
            if record.correspondences.count >= stage_config.epipolar.min_correspondences:
                output.append(record)
    if not output:
        raise ValueError("V9.7 found no causal dense training records.")
    return output


def _full_grid_record(
    *,
    current: int,
    history: int,
    target_count: int,
    full_masks: torch.Tensor,
    data: DenseData,
    stage_config: StageAConfig,
) -> DenseRecord:
    labels = local_token_reprojection_labels(
        current_frame=current,
        history_frame=history,
        slot=0,
        local_uv_normalized=data.grid_uv_normalized,
        local_valid=torch.ones(len(data.grid_uv_normalized), dtype=torch.bool),
        masks=full_masks,
        world_points_metric=data.support.world_points,
        depth_metric=data.support.depth,
        global_world_to_camera=data.support.global_w2c,
        intrinsics=data.support.intrinsics,
        config=stage_config.visibility,
    )
    full = labels.visible_correspondences()
    indices = current_farthest_indices(
        full, int(target_count), data.support.image_size
    )
    selected = take_surface(full, indices)
    query_indices = _current_uv_to_grid_indices(
        selected.current_uv, data.grid_uv_pixels
    )
    return DenseRecord(current, history, query_indices, selected)


def _current_uv_to_grid_indices(
    current_uv: torch.Tensor, grid_uv_pixels: torch.Tensor
) -> torch.Tensor:
    if not int(current_uv.shape[0]):
        return torch.empty(0, dtype=torch.long)
    distance = torch.cdist(current_uv.double(), grid_uv_pixels.double())
    values, indices = distance.min(dim=-1)
    # local_token_reprojection_labels receives float32 normalized grid UV and
    # converts it back to pixel coordinates.  That round trip can accumulate
    # roughly 1e-5 px error on the real image width.  A 1e-3 px tolerance is
    # still thousands of times smaller than one 72x72 grid-cell spacing, so it
    # accepts only the same actual grid key and cannot hide off-grid support.
    if float(values.max()) > 1e-3:
        raise ValueError(
            f"V9.7 current queries are not actual grid cells; max={float(values.max())}."
        )
    return indices.long().cpu()


def _train_matcher(
    *,
    fold_name: str,
    architecture: str,
    records: Sequence[DenseRecord],
    data: DenseData,
    config: V97Config,
    source_cache: Path,
    resume: bool,
) -> tuple[TrainState, list[dict[str, object]]]:
    if architecture not in DENSE_ARCHITECTURES:
        raise ValueError(f"Unknown V9.7 architecture={architecture!r}.")
    device = _training_device(config.training.device)
    fold_seed = int(config.training.seed) + 1009 * [
        fold.name for fold in FOLDS
    ].index(fold_name)
    # All descriptor controls intentionally share the same initialization.
    _seed_everything(fold_seed)
    model = DenseSubgridMatcher(config.matcher).to(device)
    checkpoint_dir = config.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / f"{fold_name}_{architecture}.pt"
    signature = _checkpoint_signature(
        config,
        fold=fold_name,
        architecture=architecture,
        source_cache=source_cache,
    )
    if resume and checkpoint.is_file():
        saved = _torch_load(checkpoint)
        if saved.get("signature") == signature:
            model.load_state_dict(saved["model"])
            print(f"V9.7 resumed fold={fold_name} architecture={architecture}")
            return (
                TrainState(
                    model=model,
                    parameters=sum(value.numel() for value in model.parameters()),
                    trainable_parameters=sum(
                        value.numel() for value in model.parameters()
                        if value.requires_grad
                    ),
                    steps=int(saved["steps"]),
                    initial_loss=float(saved["initial_loss"]),
                    final_loss=float(saved["final_loss"]),
                ),
                list(saved.get("logs", [])),
            )
        print(f"V9.7 ignored stale checkpoint={checkpoint}")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    initial = _dataset_loss(
        model,
        records=records,
        architecture=architecture,
        data=data,
        config=config,
        device=device,
    )
    logs = []
    for step in range(config.training.steps):
        indices = _cyclic_batch_indices(
            len(records), config.training.batch_pairs, step, fold_seed
        )
        batch = [records[index] for index in indices]
        query, key, query_valid, key_valid, target = _training_batch(
            batch,
            architecture=architecture,
            data=data,
            config=config,
            device=device,
        )
        output = model(query, key, query_valid, key_valid)
        result = dense_match_loss(model, output, target)
        optimizer.zero_grad(set_to_none=True)
        result.loss.backward()
        q_grad = _grad_norm(model.query_projection.weight)
        k_grad = _grad_norm(model.key_projection.weight)
        offset_grad = _module_grad_norm(model.offset_head)
        optimizer.step()
        if step == 0 or (step + 1) % config.training.log_every == 0:
            value = float(result.loss.detach().cpu())
            print(
                f"V9.7 fold={fold_name} architecture={architecture} "
                f"step={step + 1}/{config.training.steps} loss={value:.6g}"
            )
            logs.append(
                {
                    "fold": fold_name,
                    "architecture": architecture,
                    "seed": fold_seed,
                    "step": step + 1,
                    "loss": value,
                    "classification_loss": float(result.classification.detach().cpu()),
                    "offset_loss": float(result.offset.detach().cpu()),
                    "supervised_queries": result.supervised_queries,
                    "visible_queries": result.visible_queries,
                    "query_projection_grad_norm": q_grad,
                    "key_projection_grad_norm": k_grad,
                    "offset_head_grad_norm": offset_grad,
                }
            )
    final = _dataset_loss(
        model,
        records=records,
        architecture=architecture,
        data=data,
        config=config,
        device=device,
    )
    saved = {
        "signature": signature,
        "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "steps": config.training.steps,
        "initial_loss": initial,
        "final_loss": final,
        "logs": logs,
        "uses_pose_loss": False,
    }
    torch.save(saved, checkpoint)
    return (
        TrainState(
            model=model,
            parameters=sum(value.numel() for value in model.parameters()),
            trainable_parameters=sum(
                value.numel() for value in model.parameters() if value.requires_grad
            ),
            steps=config.training.steps,
            initial_loss=initial,
            final_loss=final,
        ),
        logs,
    )


def _dataset_loss(
    model,
    *,
    records,
    architecture,
    data,
    config,
    device,
) -> float:
    model.eval()
    weighted = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, len(records), config.training.batch_pairs):
            batch = records[start : start + config.training.batch_pairs]
            query, key, query_valid, key_valid, target = _training_batch(
                batch,
                architecture=architecture,
                data=data,
                config=config,
                device=device,
            )
            result = dense_match_loss(
                model, model(query, key, query_valid, key_valid), target
            )
            weight = max(result.supervised_queries, 1)
            weighted += float(result.loss.cpu()) * weight
            count += weight
    model.train()
    return weighted / max(count, 1)


def _training_batch(
    records: Sequence[DenseRecord],
    *,
    architecture: str,
    data: DenseData,
    config: V97Config,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, DenseMatchTarget]:
    maximum = max(record.correspondences.count for record in records)
    source = _feature_source(data, architecture)
    channels = int(source.shape[-1])
    keys = int(data.grid_uv_normalized.shape[0])
    query = torch.zeros(len(records), maximum, channels)
    key = torch.zeros(len(records), keys, channels)
    query_valid = torch.zeros(len(records), maximum, dtype=torch.bool)
    key_valid = torch.ones(len(records), keys, dtype=torch.bool)
    target_uv = torch.zeros(len(records), maximum, 2, dtype=torch.float64)
    visible = torch.zeros(len(records), maximum, dtype=torch.bool)
    for batch_index, record in enumerate(records):
        count = record.correspondences.count
        current_source = _source_frame(source, record.current)
        history_source = _source_frame(source, record.history)
        query[batch_index, :count] = current_source.index_select(
            0, record.query_grid_indices
        )
        key[batch_index] = history_source
        query_valid[batch_index, :count] = True
        visible[batch_index, :count] = True
        target_uv[batch_index, :count] = record.correspondences.history_uv
    if architecture == "sam_train_off":
        query.zero_()
        key.zero_()
    target = build_dense_match_target(
        target_uv.to(device),
        query_valid=query_valid.to(device),
        visible=visible.to(device),
        grid_size=config.grid_size,
        image_size=data.support.image_size,
    )
    return (
        query.to(device),
        key.to(device),
        query_valid.to(device),
        key_valid.to(device),
        target,
    )


def _evaluate_model(
    *,
    fold_name: str,
    train_frames: Sequence[int],
    test_frames: Sequence[int],
    architecture: str,
    variant: str,
    state: TrainState,
    records: Sequence[DenseRecord],
    data: DenseData,
    stage_config: StageAConfig,
    config: V97Config,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    device = next(state.model.parameters()).device
    state.model.eval()
    before = {
        name: value.detach().cpu().clone()
        for name, value in state.model.state_dict().items()
    }
    source = _feature_source(data, architecture)
    rows = []
    support_exact = True
    perturbed = 0
    diagonal = math.hypot(*data.support.image_size)
    with torch.no_grad():
        for record in records:
            current_source = _source_frame(source, record.current).clone()
            history_source = _source_frame(source, record.history).clone()
            if architecture == "sam_dense":
                current_source, history_source, applied = _perturb_sam_features(
                    variant,
                    record=record,
                    data=data,
                    current=current_source,
                    history=history_source,
                )
                perturbed += int(applied)
            query = current_source.index_select(0, record.query_grid_indices)
            key = history_source
            query_valid = torch.ones(1, len(query), dtype=torch.bool, device=device)
            key_valid = torch.ones(1, len(key), dtype=torch.bool, device=device)
            output = state.model(
                query[None].to(device), key[None].to(device), query_valid, key_valid
            )
            decoded = decode_dense_matches(
                state.model,
                output,
                grid_size=config.grid_size,
                image_size=data.support.image_size,
                query_valid=query_valid,
            )
            accepted = decoded.accepted[0].cpu()
            predicted_uv = decoded.history_uv_pixels[0].cpu()
            accepted_indices = torch.nonzero(accepted, as_tuple=False).flatten()
            selected = take_surface(record.correspondences, accepted_indices)
            selected.history_uv = predicted_uv.index_select(0, accepted_indices)
            errors = torch.linalg.vector_norm(
                predicted_uv - record.correspondences.history_uv, dim=-1
            )
            epe_sum = float(
                torch.where(
                    accepted,
                    errors,
                    torch.full_like(errors, diagonal),
                ).sum()
            )
            pck = int(
                (accepted & errors.le(config.matcher.pck_threshold_pixels)).sum()
            )
            support_exact = support_exact and bool(
                len(query) == record.correspondences.count
                and len(key) == config.grid_size[0] * config.grid_size[1]
            )
            l0_relative = data.support.baseline[record.history] @ invert_rigid(
                data.support.baseline[record.current]
            )
            target_relative = data.support.target[record.history] @ invert_rigid(
                data.support.target[record.current]
            )
            raw_r = float(rotation_error_degrees(l0_relative, target_relative))
            raw_t = relative_translation_direction_error_degrees(
                l0_relative[:3, :3], l0_relative[:3, 3], target_relative
            )
            estimate = estimate_relative_epipolar_pose(
                selected.current_uv,
                selected.history_uv,
                selected.weights,
                data.support.intrinsics[record.current],
                data.support.intrinsics[record.history],
                l0_relative,
                config=stage_config.epipolar,
            )
            refined_r, refined_t = raw_r, raw_t
            if estimate.success:
                candidate = torch.eye(4, dtype=torch.float64)
                candidate[:3, :3] = estimate.rotation_current_to_history
                refined_r = float(rotation_error_degrees(candidate, target_relative))
                refined_t = relative_translation_direction_error_degrees(
                    estimate.rotation_current_to_history,
                    estimate.translation_current_origin_in_history,
                    target_relative,
                )
            rows.append(
                {
                    "fold": fold_name,
                    "architecture": architecture,
                    "variant": variant,
                    "sequence_index": record.current,
                    "frame_index": data.support.frames[record.current],
                    "history_sequence_index": record.history,
                    "history_frame_index": data.support.frames[record.history],
                    "queries": record.correspondences.count,
                    "accepted_correspondences": selected.count,
                    "active": int(estimate.success),
                    "solver_reason": estimate.reason,
                    "pck_correct": pck,
                    "epe_sum_pixels": epe_sum,
                    "current_hull_coverage": uv_hull_coverage(
                        selected.current_uv, data.support.image_size
                    ),
                    "predicted_history_hull_coverage": uv_hull_coverage(
                        selected.history_uv, data.support.image_size
                    ),
                    "raw_rotation_error_deg": raw_r,
                    "refined_rotation_error_deg": refined_r,
                    "raw_translation_direction_error_deg": raw_t,
                    "refined_translation_direction_error_deg": refined_t,
                    "relative_aggregate_worse": int(
                        refined_r + refined_t > raw_r + raw_t + 1e-12
                    ),
                }
            )
    frozen = int(
        all(
            torch.equal(value, state.model.state_dict()[name].detach().cpu())
            for name, value in before.items()
        )
    )
    return (
        _summary_row(
            fold_name=fold_name,
            architecture=architecture,
            variant=variant,
            train_frames=train_frames,
            test_frames=test_frames,
            state=state,
            rows=rows,
            support_exact=int(support_exact),
            frozen=frozen,
            perturbed_pairs=perturbed,
            config=config,
        ),
        rows,
    )


def _summary_row(
    *,
    fold_name,
    architecture,
    variant,
    train_frames,
    test_frames,
    state,
    rows,
    support_exact,
    frozen,
    perturbed_pairs,
    config,
) -> dict[str, object]:
    queries = sum(int(row["queries"]) for row in rows)
    accepted = sum(int(row["accepted_correspondences"]) for row in rows)
    raw_r = _finite_mean(float(row["raw_rotation_error_deg"]) for row in rows)
    refined_r = _finite_mean(
        float(row["refined_rotation_error_deg"]) for row in rows
    )
    raw_t = _finite_mean(
        float(row["raw_translation_direction_error_deg"]) for row in rows
    )
    refined_t = _finite_mean(
        float(row["refined_translation_direction_error_deg"]) for row in rows
    )
    descriptor = {
        "sam_dense": "sam31_detector_fpn2_dense",
        "stream_dense": "frozen_streamvggt_patch_dense",
        "coordinate_only": "deterministic_grid_position_only",
        "sam_train_off": "sam31_eval_but_zero_during_training",
    }[architecture]
    return {
        "fold": fold_name,
        "architecture": architecture,
        "variant": variant,
        "descriptor_source": descriptor,
        "train_frames": " ".join(str(value) for value in train_frames),
        "test_frames": " ".join(str(value) for value in test_frames),
        "grid_size": f"{config.grid_size[0]}x{config.grid_size[1]}",
        "query_selection": "current_only_fps_gt_visible",
        "history_key_count": config.grid_size[0] * config.grid_size[1],
        "parameters": state.parameters,
        "trainable_parameters": state.trainable_parameters,
        "steps": state.steps,
        "initial_train_loss": state.initial_loss,
        "final_train_loss": state.final_loss,
        "train_loss_drop_percent": _gain(state.initial_loss, state.final_loss),
        "frames": len(rows),
        "active_frames": sum(int(row["active"]) for row in rows),
        "inactive_frames": sum(not int(row["active"]) for row in rows),
        "mean_queries": _ratio(queries, len(rows)),
        "accepted_correspondences": accepted,
        "acceptance_fraction": _ratio(accepted, queries),
        "pck_threshold_pixels": config.matcher.pck_threshold_pixels,
        "pck_accuracy": _ratio(
            sum(float(row["pck_correct"]) for row in rows), queries
        ),
        "mean_epe_pixels": _ratio(
            sum(float(row["epe_sum_pixels"]) for row in rows), queries
        ),
        "raw_rotation_error_deg": raw_r,
        "refined_rotation_error_deg": refined_r,
        "rotation_gain_percent": _gain(raw_r, refined_r),
        "raw_translation_direction_error_deg": raw_t,
        "refined_translation_direction_error_deg": refined_t,
        "translation_direction_gain_percent": _gain(raw_t, refined_t),
        "raw_relative_aggregate_deg": raw_r + raw_t,
        "refined_relative_aggregate_deg": refined_r + refined_t,
        "relative_aggregate_gain_percent": _gain(raw_r + raw_t, refined_r + refined_t),
        "relative_aggregate_worse_frames": sum(
            int(row["relative_aggregate_worse"]) for row in rows
        ),
        "parameter_matched": 0,
        "control_support_exact": int(support_exact),
        "matcher_frozen_exact": int(frozen),
        "perturbed_pairs": int(perturbed_pairs),
        "fold_sam_causal_pass": 0,
        "all_folds_sam_causal_pass": 0,
    }


def _annotate_decisions(rows: list[dict[str, object]]) -> None:
    fold_passes = []
    for fold in FOLDS:
        group = {
            (str(row["architecture"]), str(row["variant"])): row
            for row in rows
            if row["fold"] == fold.name
        }
        required = {
            ("sam_dense", value) for value in SAM_EVAL_VARIANTS
        } | {
            ("stream_dense", "normal"),
            ("coordinate_only", "normal"),
            ("sam_train_off", "normal"),
        }
        missing = required - set(group)
        if missing:
            raise ValueError(f"V9.7 fold={fold.name} lacks controls={sorted(missing)}.")
        candidate = group[("sam_dense", "normal")]
        controls = [
            group[("stream_dense", "normal")],
            group[("coordinate_only", "normal")],
            group[("sam_train_off", "normal")],
        ]
        perturbations = [
            group[("sam_dense", value)] for value in SAM_EVAL_VARIANTS[1:]
        ]
        parameter_matched = int(
            len({int(row["parameters"]) for row in group.values()}) == 1
        )
        for row in group.values():
            row["parameter_matched"] = parameter_matched
        descriptor_better = all(
            float(candidate["pck_accuracy"]) > float(control["pck_accuracy"])
            and float(candidate["mean_epe_pixels"]) < float(control["mean_epe_pixels"])
            for control in controls
        )
        pose_better = all(
            float(candidate["refined_rotation_error_deg"])
            < float(control["refined_rotation_error_deg"])
            and float(candidate["refined_translation_direction_error_deg"])
            < float(control["refined_translation_direction_error_deg"])
            for control in controls
        )
        perturbations_hurt = all(
            int(row["perturbed_pairs"]) > 0
            and float(row["pck_accuracy"]) < float(candidate["pck_accuracy"])
            and float(row["mean_epe_pixels"]) > float(candidate["mean_epe_pixels"])
            and float(row["refined_relative_aggregate_deg"])
            > float(candidate["refined_relative_aggregate_deg"])
            for row in perturbations
        )
        candidate_improves_l0 = bool(
            int(candidate["active_frames"]) == int(candidate["frames"])
            and float(candidate["refined_rotation_error_deg"])
            < float(candidate["raw_rotation_error_deg"])
            and float(candidate["refined_translation_direction_error_deg"])
            < float(candidate["raw_translation_direction_error_deg"])
            and int(candidate["relative_aggregate_worse_frames"]) == 0
        )
        passed = int(
            parameter_matched
            and int(candidate["control_support_exact"])
            and int(candidate["matcher_frozen_exact"])
            and descriptor_better
            and pose_better
            and perturbations_hurt
            and candidate_improves_l0
        )
        candidate["fold_sam_causal_pass"] = passed
        fold_passes.append(passed)
    all_pass = int(all(fold_passes))
    for row in rows:
        if row["architecture"] == "sam_dense" and row["variant"] == "normal":
            row["all_folds_sam_causal_pass"] = all_pass


def _perturb_sam_features(
    variant: str,
    *,
    record: DenseRecord,
    data: DenseData,
    current: torch.Tensor,
    history: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if variant == "normal":
        return current, history, 0
    if variant == "sam_off":
        return torch.zeros_like(current), torch.zeros_like(history), 1
    if variant == "channel_permute":
        permutation = deterministic_channel_permutation(current.shape[-1])
        return current[:, permutation], history[:, permutation], 1
    if variant == "shuffle_time":
        current_index = max(record.current - 3, 0)
        history_index = max(record.history - 2, 0)
        if current_index == record.current:
            current_index = 0
        if history_index == record.history:
            history_index = 0
        return (
            data.sam_features[current_index].clone(),
            data.sam_features[history_index].clone(),
            1,
        )
    raise ValueError(f"Unknown V9.7 SAM variant={variant!r}.")


def _feature_source(data: DenseData, architecture: str) -> torch.Tensor:
    if architecture == "stream_dense":
        return data.stream_features
    if architecture == "coordinate_only":
        return data.coordinate_features
    if architecture in {"sam_dense", "sam_train_off"}:
        return data.sam_features
    raise ValueError(f"Unknown V9.7 architecture={architecture!r}.")


def _source_frame(source: torch.Tensor, index: int) -> torch.Tensor:
    return source if source.ndim == 2 else source[index]


def _decision_markdown(rows: Sequence[dict[str, object]]) -> str:
    candidates = [
        row
        for row in rows
        if row["architecture"] == "sam_dense" and row["variant"] == "normal"
    ]
    all_pass = int(
        len(candidates) == len(FOLDS)
        and all(int(row["all_folds_sam_causal_pass"]) for row in candidates)
    )
    lines = [
        "# V9.7 dense SAM descriptor causality decision",
        "",
        "SAM3.1 and StreamVGGT are frozen. The matcher is trained only with GT 2D correspondence labels; no pose model or pose loss is trained.",
        "Pose always uses the frozen O-R1 solver on the fixed V9.3 history edges.",
        "",
        f"- dense SAM descriptor all-fold causal pass: `{all_pass}`",
        "",
        "| fold | method | variant | PCK@1 | EPE px | R gain | t-dir gain | worse | causal pass |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for fold in FOLDS:
        for row in [value for value in rows if value["fold"] == fold.name]:
            lines.append(
                f"| {fold.name} | {row['architecture']} | {row['variant']} "
                f"| {float(row['pck_accuracy']):.6g} | {float(row['mean_epe_pixels']):.6g} "
                f"| {float(row['rotation_gain_percent']):.6g} "
                f"| {float(row['translation_direction_gain_percent']):.6g} "
                f"| {row['relative_aggregate_worse_frames']} "
                f"| {row['fold_sam_causal_pass']} |"
            )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- pass=1: dense SAM descriptors predict future correspondences and improve relative pose beyond parameter/support-matched controls; SAM perturbations also destroy the gain.",
            "- train loss down but pass=0: matcher fitting is not evidence that SAM helps pose.",
            "- SAM below StreamVGGT/coordinate/train-off: SAM detector-FPN descriptors provide no causal benefit in this protocol.",
            "- Evaluation still uses GT visibility and one scene, so no metric-center, absolute-trajectory or cross-scene claim is allowed.",
            "",
        ]
    )
    return "\n".join(lines)


def load_v97_config(path: str | Path) -> V97Config:
    source = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf8")) or {}
    repository = source.parents[2]
    matcher_raw = raw.get("matcher", {})
    training_raw = raw.get("training", {})
    config = V97Config(
        source_path=source,
        v96_config=_resolve_path(
            repository,
            raw.get(
                "v96_config",
                "streaming_couping/configs/v96_dense_grid_upper_bound.yaml",
            ),
        ),
        data_config=_resolve_path(
            repository,
            raw.get(
                "data_config", "streaming_couping/configs/v92_support_data.yaml"
            ),
        ),
        output_dir=_resolve_path(
            repository,
            raw.get(
                "output_dir",
                "outputs/streaming_couping_v97_dense_descriptor_causality",
            ),
        ),
        dense_cache_path=_resolve_path(
            repository,
            raw.get(
                "dense_cache_path",
                "outputs/streaming_couping_v97_dense_descriptor_causality/cache/00a231a370_90_525_step15_dense.pt",
            ),
        ),
        clip_name=str(raw.get("clip_name", "00a231a370_90_525_step15_37_68_54")),
        grid_size=tuple(int(value) for value in raw.get("grid_size", [72, 72])),
        cache_device=str(raw.get("cache_device", "cuda:0")),
        cache_batch_size=int(raw.get("cache_batch_size", 2)),
        training_query_count=int(raw.get("training_query_count", 32)),
        training_history_offsets=tuple(
            int(value) for value in raw.get("training_history_offsets", [1, 2])
        ),
        matcher=DenseMatcherConfig(
            canonical_dim=int(matcher_raw.get("canonical_dim", 256)),
            projection_dim=int(matcher_raw.get("projection_dim", 64)),
            offset_hidden_dim=int(matcher_raw.get("offset_hidden_dim", 128)),
            temperature=float(matcher_raw.get("temperature", 0.07)),
            offset_weight=float(matcher_raw.get("offset_weight", 1.0)),
            pck_threshold_pixels=float(
                matcher_raw.get("pck_threshold_pixels", 1.0)
            ),
        ),
        training=DenseTrainingConfig(
            device=str(training_raw.get("device", "cuda:0")),
            seed=int(training_raw.get("seed", 97)),
            steps=int(training_raw.get("steps", 600)),
            batch_pairs=int(training_raw.get("batch_pairs", 4)),
            learning_rate=float(training_raw.get("learning_rate", 3e-4)),
            weight_decay=float(training_raw.get("weight_decay", 1e-4)),
            log_every=int(training_raw.get("log_every", 50)),
        ),
    )
    _validate_config(config)
    return config


def _validate_config(config: V97Config) -> None:
    if config.grid_size != (72, 72):
        raise ValueError("V9.7 dense grid is protocol-locked to 72x72.")
    if config.training_query_count != 32:
        raise ValueError("V9.7 training query count is protocol-locked to 32.")
    if config.training_history_offsets != (1, 2):
        raise ValueError("V9.7 causal training histories are protocol-locked to 1/2.")
    if config.cache_batch_size < 1:
        raise ValueError("V9.7 cache batch size must be positive.")
    config.matcher.validate()
    training = config.training
    if min(training.steps, training.batch_pairs, training.log_every) < 1:
        raise ValueError("V9.7 training counts must be positive.")
    if training.seed != 97:
        raise ValueError("V9.7 seed is protocol-locked to 97.")
    if training.steps != 600 or training.batch_pairs != 4:
        raise ValueError("V9.7 steps/batch are protocol-locked to 600/4.")


def _validate_outputs(summaries, frames) -> None:
    expected_summary = len(FOLDS) * (
        len(SAM_EVAL_VARIANTS) + len(DENSE_ARCHITECTURES) - 1
    )
    expected_frames = expected_summary * len(FOLDS[0].test_frames)
    if len(summaries) != expected_summary:
        raise ValueError(
            f"V9.7 summary rows={len(summaries)}, expected={expected_summary}."
        )
    if len(frames) != expected_frames:
        raise ValueError(f"V9.7 frame rows={len(frames)}, expected={expected_frames}.")


def _write_csv(path: Path, rows, columns) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty V9.7 CSV: {path.name}")
    expected = set(columns)
    for index, row in enumerate(rows):
        if set(row) != expected:
            raise ValueError(
                f"V9.7 CSV {path.name} row={index} mismatch: "
                f"missing={sorted(expected - set(row))}, "
                f"extra={sorted(set(row) - expected)}"
            )
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _checkpoint_signature(config, *, fold, architecture, source_cache) -> str:
    payload = {
        "experiment": "v97_dense_descriptor_causality",
        "protocol_version": 2,
        "fold": fold,
        "architecture": architecture,
        "matcher": asdict(config.matcher),
        "training": asdict(config.training),
        "source_cache": _file_provenance(source_cache),
        "dense_cache": _file_provenance(config.dense_cache_path),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf8")
    ).hexdigest()


def _cyclic_batch_indices(count: int, batch_size: int, step: int, seed: int) -> list[int]:
    generator = random.Random(int(seed) + int(step) * 1_000_003)
    if count >= batch_size:
        return generator.sample(range(count), batch_size)
    return [generator.randrange(count) for _ in range(batch_size)]


def _training_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("V9.7 CUDA unavailable; using CPU")
        return torch.device("cpu")
    return device


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _grad_norm(value: torch.Tensor) -> float:
    return 0.0 if value.grad is None else float(torch.linalg.vector_norm(value.grad))


def _module_grad_norm(module: torch.nn.Module) -> float:
    values = [
        value.grad.detach().flatten()
        for value in module.parameters()
        if value.grad is not None
    ]
    return 0.0 if not values else float(torch.linalg.vector_norm(torch.cat(values)))


def _gain(initial: float, final: float) -> float:
    if not math.isfinite(initial) or not math.isfinite(final) or initial <= 1e-12:
        return 0.0
    return 100.0 * (initial - final) / initial


def _ratio(numerator: float, denominator: float) -> float:
    return 0.0 if float(denominator) <= 0.0 else float(numerator) / float(denominator)


def _finite_mean(values: Iterable[float]) -> float:
    rows = [float(value) for value in values if math.isfinite(float(value))]
    return float("nan") if not rows else sum(rows) / len(rows)


def _file_provenance(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _jsonable_config(config: V97Config) -> dict[str, object]:
    output = asdict(config)
    for key in ("source_path", "v96_config", "data_config", "output_dir", "dense_cache_path"):
        output[key] = str(output[key])
    return output


def _resolve_path(repository: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (repository / path).resolve() if not path.is_absolute() else path.resolve()


def _torch_load(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"V9.7 dense cache is missing: {path}")
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


if __name__ == "__main__":
    main()
