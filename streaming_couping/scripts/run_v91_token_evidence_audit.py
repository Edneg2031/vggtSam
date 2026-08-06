#!/usr/bin/env python3
"""Run V9.1 A0-Q and train/future local-token matcher audits.

No model is trained here.  The runner reads the retained V7.4 observation
cache and the completed V9 Stage-A Q/K checkpoints.  A0-Q restricts the GT
history projection to the actual cached 32 keys.  The audit then compares raw
descriptor cosine and trained Q/K on identical train/future support using both
Top-1 and soft-expectation decoding.  Every pose row is produced by the frozen
O-R1 epipolar solver.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import yaml

from streaming_couping.scripts.run_v90_local_token_matcher import (
    A1_EDGE_SELECTION_POLICY,
    EDGE_COLUMNS as V90_EDGE_COLUMNS,
    FRAME_COLUMNS as V90_FRAME_COLUMNS,
    PairRecord,
    StageAConfig,
    StageAData,
    _build_records,
    _checkpoint_signature,
    _load_payload,
    _pose_frame_rows,
    _prepare_data,
    load_stage_a_config,
)
from streaming_couping.src.learned_pose.cache import cache_path
from streaming_couping.src.learned_pose.config import load_learned_pose_config
from streaming_couping.src.v74_temporal_protocol import FOLDS
from streaming_couping.src.v90_epipolar_geometry import SurfaceCorrespondences
from streaming_couping.src.v90_explicit_matcher import (
    ExplicitLocalMatcher,
    uniform_match_probability,
)
from streaming_couping.src.v91_token_evidence import (
    PREDICTION_MODES,
    TokenAuditMetrics,
    audit_token_probability,
    decode_token_probability,
    hard_discrete_oracle_probability,
    raw_cosine_probability,
)


SUMMARY_COLUMNS = (
    "stage",
    "fold",
    "split",
    "method",
    "descriptor_source",
    "projection",
    "prediction_mode",
    "checkpoint",
    "checkpoint_signature_exact",
    "train_frames",
    "eval_frames",
    "token_count",
    "pairs",
    "frames",
    "active_frames",
    "inactive_frames",
    "supervised_queries",
    "visible_queries",
    "visible_key_supported_queries",
    "visible_key_coverage_fraction",
    "coverage_at_8px",
    "coverage_at_12px",
    "coverage_at_16px",
    "accepted_correspondences",
    "pck_threshold_pixels",
    "visible_pck_accuracy",
    "visible_mean_epe_pixels",
    "supported_pck_accuracy",
    "supported_mean_epe_pixels",
    "dustbin_accuracy",
    "visible_cross_entropy",
    "dustbin_cross_entropy",
    "mean_entropy_normalized",
    "mean_real_key_mass",
    "mean_max_probability",
    "initial_train_loss",
    "final_train_loss",
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
    "fold_a0q_pass",
    "all_folds_a0q_pass",
    "trained_pose_model",
)

PAIR_COLUMNS = (
    "stage",
    "fold",
    "split",
    "method",
    "prediction_mode",
    "current_sequence_index",
    "current_frame_index",
    "history_sequence_index",
    "history_frame_index",
    "slot",
    "query_valid_tokens",
    "target_visible_tokens",
    "history_valid_keys",
    "visible_key_supported_tokens",
    "coverage_at_8px_tokens",
    "coverage_at_12px_tokens",
    "coverage_at_16px_tokens",
    "mean_nearest_key_distance_pixels",
    "accepted_correspondences",
    "visible_pck_correct",
    "visible_epe_sum_pixels",
    "supported_pck_correct",
    "supported_epe_sum_pixels",
    "dustbin_queries",
    "dustbin_correct",
    "visible_ce_sum",
    "dustbin_ce_sum",
    "entropy_sum",
    "real_key_mass_sum",
    "max_probability_sum",
)

FRAME_COLUMNS = ("split", "method", "prediction_mode", *V90_FRAME_COLUMNS)
EDGE_COLUMNS = ("split", "method", "prediction_mode", *V90_EDGE_COLUMNS)


@dataclass(frozen=True)
class AuditConfig:
    source_path: Path
    stage_a_config: Path
    stage_a_output_dir: Path
    output_dir: Path
    raw_temperature: float
    raw_dustbin_logit: float
    coverage_thresholds_pixels: tuple[float, float, float]


@dataclass(frozen=True)
class MethodSpec:
    name: str
    descriptor_source: str
    projection: str
    source: str
    checkpoint_architecture: str | None = None
    oracle: bool = False
    uniform: bool = False


@dataclass
class LoadedCheckpoint:
    model: ExplicitLocalMatcher
    path: Path
    signature_exact: int
    initial_loss: float
    final_loss: float


METHODS = (
    MethodSpec(
        name="discrete_key_oracle",
        descriptor_source="gt_projection_to_actual_history_key",
        projection="none",
        source="sam",
        oracle=True,
    ),
    MethodSpec(
        name="sam_raw_cosine",
        descriptor_source="sam31_detector_fpn2",
        projection="raw_cosine",
        source="sam",
    ),
    MethodSpec(
        name="sam_trained_qk",
        descriptor_source="sam31_detector_fpn2",
        projection="trained_linear_qk",
        source="sam",
        checkpoint_architecture="sam_match",
    ),
    MethodSpec(
        name="stream_raw_cosine",
        descriptor_source="frozen_streamvggt_patch",
        projection="raw_cosine",
        source="stream",
    ),
    MethodSpec(
        name="stream_trained_qk",
        descriptor_source="frozen_streamvggt_patch",
        projection="trained_linear_qk",
        source="stream",
        checkpoint_architecture="stream_patch_match",
    ),
    MethodSpec(
        name="sam_train_off_qk",
        descriptor_source="sam31_eval_zero_descriptor_during_training",
        projection="trained_linear_qk",
        source="sam",
        checkpoint_architecture="sam_train_off",
    ),
    MethodSpec(
        name="uniform_uv",
        descriptor_source="none_uniform_valid_keys",
        projection="none",
        source="sam",
        uniform=True,
    ),
)


def main() -> None:
    args = _parse_args()
    config = load_audit_config(args.config)
    if args.stage_a_output_dir:
        config = replace(
            config,
            stage_a_output_dir=Path(args.stage_a_output_dir).expanduser().resolve(),
        )
    if args.output_dir:
        config = replace(
            config, output_dir=Path(args.output_dir).expanduser().resolve()
        )
    if args.checkpoint_preflight:
        try:
            validate_checkpoint_provenance(config)
        except (FileNotFoundError, KeyError, RuntimeError, ValueError) as error:
            print(f"V9.1 checkpoint preflight failed: {error}")
            raise SystemExit(2) from None
        print("V9.1 checkpoint provenance preflight passed")
        return
    result = run_audit(config)
    print(f"V9.1 token evidence audit result={result}")


def run_audit(config: AuditConfig) -> Path:
    stage_config = load_stage_a_config(config.stage_a_config)
    stage_config = replace(stage_config, output_dir=config.stage_a_output_dir)
    payload, cache_file = _load_payload(stage_config)
    data = _prepare_data(payload, stage_config)
    del payload
    positions = {frame: index for index, frame in enumerate(data.frames)}
    records_by_fold: dict[str, dict[str, list[PairRecord]]] = {}
    for fold in FOLDS:
        records_by_fold[fold.name] = {
            "train": _build_records(
                data,
                current_indices=[positions[value] for value in fold.train_frames],
                config=stage_config,
            ),
            "test": _build_records(
                data,
                current_indices=[positions[value] for value in fold.test_frames],
                config=stage_config,
            ),
        }

    summary_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    frame_rows: list[dict[str, object]] = []
    edge_rows: list[dict[str, object]] = []
    checkpoint_metadata: dict[str, dict[str, object]] = {}
    for fold in FOLDS:
        checkpoints = _load_fold_checkpoints(
            fold_name=fold.name,
            config=config,
            stage_config=stage_config,
            cache_file=cache_file,
        )
        for architecture, checkpoint in checkpoints.items():
            checkpoint_metadata[f"{fold.name}:{architecture}"] = {
                "path": str(checkpoint.path),
                "signature_exact": checkpoint.signature_exact,
                "initial_loss": checkpoint.initial_loss,
                "final_loss": checkpoint.final_loss,
            }
        for split, eval_frames in (
            ("train", fold.train_frames),
            ("test", fold.test_frames),
        ):
            records = records_by_fold[fold.name][split]
            for method in METHODS:
                modes = ("top1",) if method.oracle else PREDICTION_MODES
                for mode in modes:
                    checkpoint = (
                        None
                        if method.checkpoint_architecture is None
                        else checkpoints[method.checkpoint_architecture]
                    )
                    result = _evaluate_method(
                        fold_name=fold.name,
                        split=split,
                        train_frames=fold.train_frames,
                        eval_frames=eval_frames,
                        records=records,
                        method=method,
                        mode=mode,
                        checkpoint=checkpoint,
                        data=data,
                        stage_config=stage_config,
                        config=config,
                    )
                    summary_rows.append(result[0])
                    pair_rows.extend(result[1])
                    frame_rows.extend(result[2])
                    edge_rows.extend(result[3])

    _validate_output_rows(
        summary_rows=summary_rows,
        pair_rows=pair_rows,
        frame_rows=frame_rows,
        records_by_fold=records_by_fold,
    )
    _annotate_a0q(summary_rows)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = config.output_dir / "v91_token_evidence_summary.csv"
    pairs_path = config.output_dir / "v91_token_evidence_pairs.csv"
    frames_path = config.output_dir / "v91_token_evidence_frames.csv"
    edges_path = config.output_dir / "v91_token_evidence_edges.csv"
    _write_csv(summary_path, summary_rows, SUMMARY_COLUMNS)
    _write_csv(pairs_path, pair_rows, PAIR_COLUMNS)
    _write_csv(frames_path, frame_rows, FRAME_COLUMNS)
    _write_csv(edges_path, edge_rows, EDGE_COLUMNS)
    decision_path = config.output_dir / "v91_token_evidence_decision.md"
    decision_path.write_text(_decision_markdown(summary_rows), encoding="utf8")
    metadata = {
        "experiment": "V9.1 discrete-key oracle and train/future matcher audit",
        "config": _jsonable_config(config),
        "stage_a_config": str(stage_config.source_path),
        "cache": {
            "path": str(cache_file),
            "size_bytes": cache_file.stat().st_size,
            "mtime_ns": cache_file.stat().st_mtime_ns,
        },
        "checkpoints": checkpoint_metadata,
        "audit_trains_model": False,
        "evaluates_trained_qk_checkpoints": True,
        "trained_pose_model": False,
        "pose_loss_used": False,
        "edge_selection_policy": A1_EDGE_SELECTION_POLICY,
        "edge_selection_reads_gt_pose_error": False,
        "a0q_gt_usage": (
            "visibility and continuous history projection only; prediction is "
            "restricted to the actual cached 32 history keys"
        ),
        "outputs": {
            "summary": str(summary_path),
            "pairs": str(pairs_path),
            "frames": str(frames_path),
            "edges": str(edges_path),
            "decision": str(decision_path),
        },
    }
    (config.output_dir / "v91_token_evidence_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf8"
    )
    print(f"V9.1 wrote summary={summary_path}")
    print(f"V9.1 wrote decision={decision_path}")
    return summary_path


def validate_checkpoint_provenance(config: AuditConfig) -> None:
    """Fail fast when retained Q/K checkpoints do not match the cache/config."""

    stage_config = load_stage_a_config(config.stage_a_config)
    stage_config = replace(stage_config, output_dir=config.stage_a_output_dir)
    learned = load_learned_pose_config(stage_config.data_config)
    clip = next(
        (item for item in learned.clips if item.name == stage_config.clip_name),
        None,
    )
    if clip is None:
        raise ValueError(
            f"V9.1 clip={stage_config.clip_name!r} is absent from the data config."
        )
    cache_file = cache_path(learned, clip)
    if not cache_file.is_file():
        raise FileNotFoundError(f"V9.1 cache is missing: {cache_file}")
    for fold in FOLDS:
        _load_fold_checkpoints(
            fold_name=fold.name,
            config=config,
            stage_config=stage_config,
            cache_file=cache_file,
        )


def _load_fold_checkpoints(
    *,
    fold_name: str,
    config: AuditConfig,
    stage_config: StageAConfig,
    cache_file: Path,
) -> dict[str, LoadedCheckpoint]:
    output = {}
    for architecture in ("sam_match", "stream_patch_match", "sam_train_off"):
        path = config.stage_a_output_dir / "checkpoints" / (
            f"{fold_name}_{architecture}.pt"
        )
        if not path.is_file():
            raise FileNotFoundError(
                "V9.1 requires the completed V9 Stage-A checkpoints. "
                f"Missing: {path}"
            )
        saved = torch.load(path, map_location="cpu", weights_only=False)
        expected = _checkpoint_signature(
            stage_config,
            cache_file=cache_file,
            fold=fold_name,
            architecture=architecture,
        )
        signature_exact = int(str(saved.get("signature", "")) == expected)
        if not signature_exact:
            raise ValueError(
                f"V9.1 checkpoint provenance mismatch: {path}. "
                "Rerun commands_v90_stage_a_correspondence.txt."
            )
        model = ExplicitLocalMatcher(stage_config.matcher).cpu()
        model.load_state_dict(saved["model"])
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        output[architecture] = LoadedCheckpoint(
            model=model,
            path=path,
            signature_exact=signature_exact,
            initial_loss=float(saved["initial_loss"]),
            final_loss=float(saved["final_loss"]),
        )
    return output


def _evaluate_method(
    *,
    fold_name: str,
    split: str,
    train_frames: Sequence[int],
    eval_frames: Sequence[int],
    records: Sequence[PairRecord],
    method: MethodSpec,
    mode: str,
    checkpoint: LoadedCheckpoint | None,
    data: StageAData,
    stage_config: StageAConfig,
    config: AuditConfig,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    predictions: dict[int, SurfaceCorrespondences] = {}
    match_stats: dict[int, dict[str, float]] = {}
    metrics_rows: list[TokenAuditMetrics] = []
    pair_rows: list[dict[str, object]] = []
    with torch.no_grad():
        for record in records:
            source = (
                data.local_features_stream
                if method.source == "stream"
                else data.local_features_sam
            )
            query = source[record.current, record.slot]
            key = source[record.history, record.slot]
            query_valid = record.labels.query_valid
            key_valid = data.local_valid[record.history, record.slot]
            if method.oracle:
                probability = hard_discrete_oracle_probability(record.target)
            elif method.uniform:
                probability = uniform_match_probability(
                    query_valid[None], key_valid[None]
                )[0]
            elif checkpoint is None:
                probability = raw_cosine_probability(
                    query,
                    key,
                    query_valid,
                    key_valid,
                    canonical_dim=stage_config.matcher.canonical_dim,
                    temperature=config.raw_temperature,
                    dustbin_logit=config.raw_dustbin_logit,
                )
            else:
                probability = checkpoint.model(
                    query[None], key[None], query_valid[None], key_valid[None]
                )["probability"][0].cpu()
            decoded = decode_token_probability(
                probability,
                history_uv_normalized=data.local_uv[
                    record.history, record.slot
                ],
                query_valid=query_valid,
                key_valid=key_valid,
                image_size=data.image_size,
                mode=mode,
            )
            metrics = audit_token_probability(
                probability,
                labels=record.labels,
                target=record.target,
                prediction=decoded,
                pck_threshold_pixels=stage_config.matcher.pck_threshold_pixels,
                image_size=data.image_size,
            )
            metrics_rows.append(metrics)
            accepted = decoded.accepted
            weights = decoded.weights[accepted].double()
            if method.oracle:
                weights = weights * record.labels.weights[accepted].double()
            predictions[id(record)] = SurfaceCorrespondences(
                current_frame=record.current,
                history_frame=record.history,
                slot=record.slot,
                current_uv=record.labels.current_uv[accepted].double(),
                history_uv=decoded.predicted_uv[accepted].double(),
                weights=weights,
                depth_residual_metric=record.labels.depth_residual_metric[
                    accepted
                ].double(),
                sampled_queries=record.labels.query_count,
                projected_in_bounds=record.labels.query_count,
                visible_queries=int(accepted.sum()),
            )
            match_stats[id(record)] = metrics.as_match_stats()
            pair_rows.append(
                _pair_row(
                    fold_name=fold_name,
                    split=split,
                    method=method,
                    mode=mode,
                    record=record,
                    metrics=metrics,
                    data=data,
                    thresholds=config.coverage_thresholds_pixels,
                )
            )
    frames, edges = _pose_frame_rows(
        stage="A0-Q" if method.oracle else "AUDIT",
        fold_name=fold_name,
        architecture=method.name,
        variant=mode,
        edge_selection_policy=A1_EDGE_SELECTION_POLICY,
        test_frames=eval_frames,
        records=records,
        predictions=predictions,
        match_stats=match_stats,
        data=data,
        config=stage_config,
    )
    audit_frames = [
        {"split": split, "method": method.name, "prediction_mode": mode, **row}
        for row in frames
    ]
    audit_edges = [
        {"split": split, "method": method.name, "prediction_mode": mode, **row}
        for row in edges
    ]
    summary = _summary_row(
        fold_name=fold_name,
        split=split,
        train_frames=train_frames,
        eval_frames=eval_frames,
        method=method,
        mode=mode,
        checkpoint=checkpoint,
        records=records,
        metrics=metrics_rows,
        frames=frames,
        thresholds=config.coverage_thresholds_pixels,
        pck_threshold=stage_config.matcher.pck_threshold_pixels,
    )
    return summary, pair_rows, audit_frames, audit_edges


def _pair_row(
    *,
    fold_name: str,
    split: str,
    method: MethodSpec,
    mode: str,
    record: PairRecord,
    metrics: TokenAuditMetrics,
    data: StageAData,
    thresholds: tuple[float, float, float],
) -> dict[str, object]:
    visible_distance = record.target.nearest_key_distance_pixels[
        record.labels.target_visible
        & record.labels.query_valid
        & torch.isfinite(record.target.nearest_key_distance_pixels)
    ]
    coverage = [
        int(visible_distance.le(float(threshold)).sum()) for threshold in thresholds
    ]
    return {
        "stage": "A0-Q" if method.oracle else "AUDIT",
        "fold": fold_name,
        "split": split,
        "method": method.name,
        "prediction_mode": mode,
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
        "visible_key_supported_tokens": metrics.visible_supported_queries,
        "coverage_at_8px_tokens": coverage[0],
        "coverage_at_12px_tokens": coverage[1],
        "coverage_at_16px_tokens": coverage[2],
        "mean_nearest_key_distance_pixels": (
            float("nan")
            if not visible_distance.numel()
            else float(visible_distance.mean())
        ),
        "accepted_correspondences": metrics.accepted_correspondences,
        "visible_pck_correct": metrics.visible_pck_correct,
        "visible_epe_sum_pixels": metrics.visible_epe_sum_pixels,
        "supported_pck_correct": metrics.supported_pck_correct,
        "supported_epe_sum_pixels": metrics.supported_epe_sum_pixels,
        "dustbin_queries": metrics.dustbin_queries,
        "dustbin_correct": metrics.dustbin_correct,
        "visible_ce_sum": metrics.visible_ce_sum,
        "dustbin_ce_sum": metrics.dustbin_ce_sum,
        "entropy_sum": metrics.entropy_sum,
        "real_key_mass_sum": metrics.real_key_mass_sum,
        "max_probability_sum": metrics.max_probability_sum,
    }


def _summary_row(
    *,
    fold_name: str,
    split: str,
    train_frames: Sequence[int],
    eval_frames: Sequence[int],
    method: MethodSpec,
    mode: str,
    checkpoint: LoadedCheckpoint | None,
    records: Sequence[PairRecord],
    metrics: Sequence[TokenAuditMetrics],
    frames: Sequence[dict[str, object]],
    thresholds: tuple[float, float, float],
    pck_threshold: float,
) -> dict[str, object]:
    totals = _sum_metrics(metrics)
    visible_distances = [
        record.target.nearest_key_distance_pixels[
            record.labels.target_visible
            & record.labels.query_valid
            & torch.isfinite(record.target.nearest_key_distance_pixels)
        ]
        for record in records
    ]
    visible_distance = (
        torch.cat(visible_distances)
        if any(value.numel() for value in visible_distances)
        else torch.empty(0)
    )
    coverage = [
        _safe_ratio(
            int(visible_distance.le(float(threshold)).sum()),
            totals.visible_queries,
        )
        for threshold in thresholds
    ]
    raw_r = _finite_mean(
        float(row["raw_edge_rotation_error_deg"]) for row in frames
    )
    refined_r = _finite_mean(
        float(row["refined_edge_rotation_error_deg"]) for row in frames
    )
    raw_t = _finite_mean(
        float(row["raw_edge_translation_direction_error_deg"]) for row in frames
    )
    refined_t = _finite_mean(
        float(row["refined_edge_translation_direction_error_deg"])
        for row in frames
    )
    raw_pose = raw_r + raw_t
    refined_pose = refined_r + refined_t
    return {
        "stage": "A0-Q" if method.oracle else "AUDIT",
        "fold": fold_name,
        "split": split,
        "method": method.name,
        "descriptor_source": method.descriptor_source,
        "projection": method.projection,
        "prediction_mode": mode,
        "checkpoint": "" if checkpoint is None else str(checkpoint.path),
        "checkpoint_signature_exact": (
            "" if checkpoint is None else checkpoint.signature_exact
        ),
        "train_frames": " ".join(str(value) for value in train_frames),
        "eval_frames": " ".join(str(value) for value in eval_frames),
        "token_count": 32,
        "pairs": len(records),
        "frames": len(frames),
        "active_frames": sum(int(row["active"]) for row in frames),
        "inactive_frames": sum(not int(row["active"]) for row in frames),
        "supervised_queries": totals.supervised_queries,
        "visible_queries": totals.visible_queries,
        "visible_key_supported_queries": totals.visible_supported_queries,
        "visible_key_coverage_fraction": _safe_ratio(
            totals.visible_supported_queries, totals.visible_queries
        ),
        "coverage_at_8px": coverage[0],
        "coverage_at_12px": coverage[1],
        "coverage_at_16px": coverage[2],
        "accepted_correspondences": totals.accepted_correspondences,
        "pck_threshold_pixels": pck_threshold,
        "visible_pck_accuracy": _safe_ratio(
            totals.visible_pck_correct, totals.visible_queries
        ),
        "visible_mean_epe_pixels": _safe_ratio(
            totals.visible_epe_sum_pixels, totals.visible_queries
        ),
        "supported_pck_accuracy": _safe_ratio(
            totals.supported_pck_correct, totals.visible_supported_queries
        ),
        "supported_mean_epe_pixels": _safe_ratio(
            totals.supported_epe_sum_pixels, totals.visible_supported_queries
        ),
        "dustbin_accuracy": _safe_ratio(
            totals.dustbin_correct, totals.dustbin_queries
        ),
        "visible_cross_entropy": _safe_ratio(
            totals.visible_ce_sum, totals.visible_ce_queries
        ),
        "dustbin_cross_entropy": _safe_ratio(
            totals.dustbin_ce_sum, totals.dustbin_queries
        ),
        "mean_entropy_normalized": _safe_ratio(
            totals.entropy_sum, totals.supervised_queries
        ),
        "mean_real_key_mass": _safe_ratio(
            totals.real_key_mass_sum, totals.supervised_queries
        ),
        "mean_max_probability": _safe_ratio(
            totals.max_probability_sum, totals.supervised_queries
        ),
        "initial_train_loss": (
            float("nan") if checkpoint is None else checkpoint.initial_loss
        ),
        "final_train_loss": (
            float("nan") if checkpoint is None else checkpoint.final_loss
        ),
        "raw_edge_rotation_error_deg": raw_r,
        "refined_edge_rotation_error_deg": refined_r,
        "edge_rotation_gain_percent": _gain(raw_r, refined_r),
        "raw_edge_translation_direction_error_deg": raw_t,
        "refined_edge_translation_direction_error_deg": refined_t,
        "edge_translation_direction_gain_percent": _gain(raw_t, refined_t),
        "raw_relative_aggregate_deg": raw_pose,
        "refined_relative_aggregate_deg": refined_pose,
        "relative_aggregate_gain_percent": _gain(raw_pose, refined_pose),
        "relative_aggregate_worse_frames": sum(
            int(row["relative_aggregate_worse"]) for row in frames
        ),
        "fold_a0q_pass": 0,
        "all_folds_a0q_pass": 0,
        "trained_pose_model": 0,
    }


def _sum_metrics(rows: Sequence[TokenAuditMetrics]) -> TokenAuditMetrics:
    return TokenAuditMetrics(
        **{
            name: sum(getattr(row, name) for row in rows)
            for name in TokenAuditMetrics.__dataclass_fields__
        }
    )


def _validate_output_rows(
    *,
    summary_rows: Sequence[dict[str, object]],
    pair_rows: Sequence[dict[str, object]],
    frame_rows: Sequence[dict[str, object]],
    records_by_fold: dict[str, dict[str, list[PairRecord]]],
) -> None:
    """Lock experiment cardinality before any CSV or decision is written."""

    methods_per_split = 1 + (len(METHODS) - 1) * len(PREDICTION_MODES)
    expected_summary = len(FOLDS) * 2 * methods_per_split
    if len(summary_rows) != expected_summary:
        raise ValueError(
            f"V9.1 summary rows={len(summary_rows)}, expected={expected_summary}."
        )
    summary_keys = {
        (
            row["fold"],
            row["split"],
            row["method"],
            row["prediction_mode"],
        )
        for row in summary_rows
    }
    if len(summary_keys) != expected_summary:
        raise ValueError("V9.1 summary contains duplicate experiment keys.")
    expected_pairs = methods_per_split * sum(
        len(records_by_fold[fold.name][split])
        for fold in FOLDS
        for split in ("train", "test")
    )
    if len(pair_rows) != expected_pairs:
        raise ValueError(
            f"V9.1 pair rows={len(pair_rows)}, expected={expected_pairs}."
        )
    expected_frames = methods_per_split * sum(
        len(fold.train_frames) + len(fold.test_frames) for fold in FOLDS
    )
    if len(frame_rows) != expected_frames:
        raise ValueError(
            f"V9.1 frame rows={len(frame_rows)}, expected={expected_frames}."
        )


def _annotate_a0q(rows: list[dict[str, object]]) -> None:
    fold_passes = []
    for fold in FOLDS:
        candidates = [
            row
            for row in rows
            if row["stage"] == "A0-Q"
            and row["fold"] == fold.name
            and row["split"] == "test"
            and row["method"] == "discrete_key_oracle"
            and row["prediction_mode"] == "top1"
        ]
        if len(candidates) != 1:
            raise ValueError(f"V9.1 fold={fold.name} lacks one A0-Q test row.")
        row = candidates[0]
        passed = int(_strict_pose_pass(row))
        row["fold_a0q_pass"] = passed
        fold_passes.append(passed)
    all_pass = int(len(fold_passes) == len(FOLDS) and all(fold_passes))
    for row in rows:
        row["all_folds_a0q_pass"] = all_pass


def _decision_markdown(rows: Sequence[dict[str, object]]) -> str:
    a0q_rows = [
        row
        for row in rows
        if row["stage"] == "A0-Q" and row["split"] == "test"
    ]
    all_a0q = int(
        len(a0q_rows) == len(FOLDS)
        and all(int(row["fold_a0q_pass"]) for row in a0q_rows)
    )
    sam_pose_passes = {}
    for mode in PREDICTION_MODES:
        mode_rows = [
            row
            for row in rows
            if row["split"] == "test"
            and row["method"] == "sam_trained_qk"
            and row["prediction_mode"] == mode
        ]
        sam_pose_passes[mode] = int(
            len(mode_rows) == len(FOLDS)
            and all(_strict_pose_pass(row) for row in mode_rows)
        )
    lines = [
        "# V9.1 discrete-token evidence decision",
        "",
        "This audit performs no training; learned Q/K rows reuse the completed V9 Stage-A checkpoints.",
        "No pose model is trained. Pose always uses the frozen O-R1 solver.",
        "A0-Q uses GT only for visibility/projection and must select an actual cached history key.",
        "The Q/K matcher checkpoints were trained with GT correspondence labels, never pose loss.",
        "",
        f"- A0-Q discrete local32 all-fold pass: `{all_a0q}`",
        f"- trained SAM Q/K Top-1 future pose pass: `{sam_pose_passes['top1']}`",
        f"- trained SAM Q/K soft-expectation future pose pass: `{sam_pose_passes['soft_expectation']}`",
        "",
        "## A0-Q future-fold result",
        "",
        "| fold | key coverage | PCK | EPE px | R gain | t-dir gain | worse | pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in a0q_rows:
        lines.append(
            f"| {row['fold']} | {float(row['visible_key_coverage_fraction']):.6g} "
            f"| {float(row['visible_pck_accuracy']):.6g} "
            f"| {float(row['visible_mean_epe_pixels']):.6g} "
            f"| {float(row['edge_rotation_gain_percent']):.6g} "
            f"| {float(row['edge_translation_direction_gain_percent']):.6g} "
            f"| {row['relative_aggregate_worse_frames']} "
            f"| {row['fold_a0q_pass']} |"
        )
    lines.extend(
        [
            "",
            "## SAM detector-FPN train/future audit",
            "",
            "| fold | mode | train PCK | future PCK | train EPE | future EPE | future pose gain |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for fold in FOLDS:
        for mode in PREDICTION_MODES:
            group = {
                str(row["split"]): row
                for row in rows
                if row["fold"] == fold.name
                and row["method"] == "sam_trained_qk"
                and row["prediction_mode"] == mode
            }
            if set(group) != {"train", "test"}:
                raise ValueError(
                    f"V9.1 fold={fold.name} mode={mode} lacks train/test SAM rows."
                )
            train, test = group["train"], group["test"]
            lines.append(
                f"| {fold.name} | {mode} "
                f"| {float(train['visible_pck_accuracy']):.6g} "
                f"| {float(test['visible_pck_accuracy']):.6g} "
                f"| {float(train['visible_mean_epe_pixels']):.6g} "
                f"| {float(test['visible_mean_epe_pixels']):.6g} "
                f"| {float(test['relative_aggregate_gain_percent']):.6g} |"
            )
    lines.extend(
        [
            "",
            "## Future Top-1 controls",
            "",
            "| fold | method | PCK | EPE px | pose gain | worse |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    control_methods = (
        "sam_raw_cosine",
        "sam_trained_qk",
        "stream_raw_cosine",
        "stream_trained_qk",
        "sam_train_off_qk",
        "uniform_uv",
    )
    for fold in FOLDS:
        for method in control_methods:
            candidates = [
                row
                for row in rows
                if row["fold"] == fold.name
                and row["split"] == "test"
                and row["method"] == method
                and row["prediction_mode"] == "top1"
            ]
            if len(candidates) != 1:
                raise ValueError(
                    f"V9.1 fold={fold.name} method={method} lacks one Top-1 row."
                )
            row = candidates[0]
            lines.append(
                f"| {fold.name} | {method} "
                f"| {float(row['visible_pck_accuracy']):.6g} "
                f"| {float(row['visible_mean_epe_pixels']):.6g} "
                f"| {float(row['relative_aggregate_gain_percent']):.6g} "
                f"| {row['relative_aggregate_worse_frames']} |"
            )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- A0-Q=0: actual local32 history-key sampling is already an upper-bound bottleneck; do not tune the matcher.",
            "- A0-Q=1 with low train PCK: the objective/dustbin/decoder or descriptor separability fails even in-sample.",
            "- A0-Q=1 with high train but low future PCK: the learned Q/K temporally overfits.",
            "- Raw and trained SAM rows below StreamVGGT/uniform controls: stop detector_fpn2 and test a genuinely temporal SAM memory feature only.",
            "- No result here supports metric center, absolute trajectory, or cross-scene generalization.",
            "",
        ]
    )
    return "\n".join(lines)


def _strict_pose_pass(row: dict[str, object]) -> bool:
    return bool(
        int(row["active_frames"]) == int(row["frames"])
        and float(row["refined_edge_rotation_error_deg"])
        < float(row["raw_edge_rotation_error_deg"])
        and float(row["refined_edge_translation_direction_error_deg"])
        < float(row["raw_edge_translation_direction_error_deg"])
        and int(row["relative_aggregate_worse_frames"]) == 0
    )


def load_audit_config(path: str | Path) -> AuditConfig:
    source = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf8")) or {}
    repository = source.parents[2]
    thresholds = tuple(
        float(value)
        for value in raw.get("coverage_thresholds_pixels", (8.0, 12.0, 16.0))
    )
    if thresholds != (8.0, 12.0, 16.0):
        raise ValueError("V9.1 locks coverage thresholds to 8/12/16 pixels.")
    config = AuditConfig(
        source_path=source,
        stage_a_config=_resolve_path(
            repository,
            raw.get(
                "stage_a_config",
                "streaming_couping/configs/v90_local_token_matcher.yaml",
            ),
        ),
        stage_a_output_dir=_resolve_path(
            repository,
            raw.get(
                "stage_a_output_dir",
                "outputs/streaming_couping_v90_epipolar_token_stage_a",
            ),
        ),
        output_dir=_resolve_path(
            repository,
            raw.get(
                "output_dir",
                "outputs/streaming_couping_v91_token_evidence_audit",
            ),
        ),
        raw_temperature=float(raw.get("raw_temperature", 0.07)),
        raw_dustbin_logit=float(raw.get("raw_dustbin_logit", 0.0)),
        coverage_thresholds_pixels=thresholds,
    )
    if config.raw_temperature <= 0.0 or not math.isfinite(config.raw_temperature):
        raise ValueError("V9.1 raw temperature must be finite and positive.")
    if not math.isfinite(config.raw_dustbin_logit):
        raise ValueError("V9.1 raw dustbin logit must be finite.")
    return config


def _write_csv(
    path: Path, rows: Sequence[dict[str, object]], columns: Sequence[str]
) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty V9.1 CSV: {path.name}")
    expected = set(columns)
    for index, row in enumerate(rows):
        if set(row) != expected:
            raise ValueError(
                f"V9.1 {path.name} row={index} mismatch: "
                f"missing={sorted(expected - set(row))}, "
                f"extra={sorted(set(row) - expected)}"
            )
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if float(denominator) <= 0.0:
        return float("nan")
    return float(numerator) / float(denominator)


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


def _jsonable_config(config: AuditConfig) -> dict[str, Any]:
    value = asdict(config)
    for key in ("source_path", "stage_a_config", "stage_a_output_dir", "output_dir"):
        value[key] = str(value[key])
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="streaming_couping/configs/v91_token_evidence_audit.yaml",
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--stage-a-output-dir")
    parser.add_argument(
        "--checkpoint-preflight",
        action="store_true",
        help="Validate cache/config/checkpoint provenance, then exit.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
