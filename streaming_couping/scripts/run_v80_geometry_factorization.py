#!/usr/bin/env python3
"""V8 O2.6: isolate depth, intrinsics and scale in explicit pose recovery.

No model is trained.  GT-world pseudo-correspondences and GT history poses are
held fixed while the camera-space point construction is changed one factor at
a time.  Per-frame GT scale/affine depth calibration is an oracle diagnostic,
never a deployable inference path.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch
import yaml

from streaming_couping.scripts.run_v80_geometry_support_ablation import (
    SupportSpec,
    _effective_count,
    _pairs_for,
    _prepare_tokens,
    _unique_rows,
)
from streaming_couping.scripts.run_v80_theory_validation import (
    SolverSpec,
    TheoryConfig,
    _memory_write,
    _validate_payload,
    load_theory_config,
)
from streaming_couping.src.config import load_config
from streaming_couping.src.external_repos import maybe_add_repo_to_path
from streaming_couping.src.learned_pose.cache import cache_path, load_feature_cache
from streaming_couping.src.learned_pose.config import load_learned_pose_config
from streaming_couping.src.solvers.weighted_kabsch import weighted_kabsch
from streaming_couping.src.solvers.weighted_umeyama import weighted_umeyama
from streaming_couping.src.v74_temporal_protocol import EXPECTED_FRAMES, FOLDS
from streaming_couping.src.v80_pose_geometry import (
    CausalPairIndices,
    backproject_depth_at_local_tokens,
    camera_centers,
    gather_pair_points,
    invert_rigid,
    rotation_error_degrees,
    transform_points,
)


@dataclass(frozen=True)
class GeometryVariant:
    name: str
    depth_source: str
    intrinsics_source: str
    calibration: str
    oracle_level: str


VARIANTS = (
    GeometryVariant(
        "o1_direct_gt_camera",
        "direct_gt_camera",
        "none",
        "none",
        "O1",
    ),
    GeometryVariant(
        "o26_gt_depth_gt_intrinsics",
        "gt_depth",
        "gt_intrinsics",
        "none",
        "O2.6-RP",
    ),
    GeometryVariant(
        "o26_gt_depth_pred_intrinsics",
        "gt_depth",
        "pred_intrinsics",
        "none",
        "O2.6-K",
    ),
    GeometryVariant(
        "o26_pred_depth_gt_intrinsics",
        "pred_depth",
        "gt_intrinsics",
        "none",
        "O2.6-Z",
    ),
    GeometryVariant(
        "o26_pred_depth_pred_intrinsics",
        "pred_depth",
        "pred_intrinsics",
        "none",
        "O2",
    ),
    GeometryVariant(
        "o26_scale_calibrated_depth_gt_intrinsics",
        "pred_depth",
        "gt_intrinsics",
        "scale",
        "O2.6-ZS",
    ),
    GeometryVariant(
        "o26_scale_calibrated_depth_pred_intrinsics",
        "pred_depth",
        "pred_intrinsics",
        "scale",
        "O2.6-ZS+K",
    ),
    GeometryVariant(
        "o26_affine_calibrated_depth_gt_intrinsics",
        "pred_depth",
        "gt_intrinsics",
        "affine",
        "O2.6-ZA",
    ),
    GeometryVariant(
        "o26_affine_calibrated_depth_pred_intrinsics",
        "pred_depth",
        "pred_intrinsics",
        "affine",
        "O2.6-ZA+K",
    ),
)


@dataclass(frozen=True)
class SolverFamily:
    name: str
    transform_family: str
    base: SolverSpec


@dataclass(frozen=True)
class CalibrationConfig:
    trim_quantile: float
    trim_iterations: int
    min_pixels: int
    max_pixels: int


@dataclass(frozen=True)
class O26GateConfig:
    min_effective_correspondences: float
    min_unique_current_correspondences: int
    min_active_fraction: float


@dataclass(frozen=True)
class O26Config:
    source_path: Path
    theory: TheoryConfig
    output_dir: Path
    support: SupportSpec
    calibration: CalibrationConfig
    gates: O26GateConfig


@dataclass
class DepthCalibration:
    scale: torch.Tensor
    offset: torch.Tensor
    valid_pixels: torch.Tensor
    before_rmse_metric: torch.Tensor
    after_rmse_metric: torch.Tensor
    after_median_metric: torch.Tensor
    after_p90_metric: torch.Tensor


@dataclass
class Candidate:
    candidate_w2c: torch.Tensor
    accepted_by_solver: bool
    bounded_accept: bool
    fit_scale: float
    fit_rmse_metric: float
    fit_p90_metric: float
    inlier_ratio: float
    retained_count: int
    source_extent_metric: float
    correction_rotation_deg: float
    correction_center_native: float
    reasons: list[str]


def main() -> None:
    args = _parse_args()
    config = load_o26_config(args.config)
    if args.output_dir:
        config = replace(
            config,
            output_dir=Path(args.output_dir).expanduser().resolve(),
        )
    result = run_o26_factorization(config)
    print(f"V8 O2.6 geometry-factorization result={result}")


def run_o26_factorization(config: O26Config) -> Path:
    data = load_learned_pose_config(config.theory.data_config)
    clip = next(
        (item for item in data.clips if item.name == config.theory.clip_name),
        None,
    )
    if clip is None:
        raise ValueError(f"O2.6 clip={config.theory.clip_name!r} is not configured.")
    path = cache_path(data, clip)
    if not path.is_file():
        raise FileNotFoundError(f"O2.6 requires retained/rebuilt V7.4 cache: {path}")
    payload = load_feature_cache(path)
    _validate_payload(payload)
    required = {"target_depth", "target_pose_encoding"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"O2.6 cache lacks fields={sorted(missing)}; rebuild it.")
    frames = tuple(int(value) for value in payload["frame_indices"])
    if frames != EXPECTED_FRAMES:
        raise ValueError(f"O2.6 requires frames 90:15:525, got {frames}.")
    if int(payload.get("reference_sequence_index", -1)) != 0:
        raise ValueError("O2.6 requires frame 90 as the fixed coordinate gauge.")

    recovery = load_config(data.recovery_config)
    maybe_add_repo_to_path(recovery.streamvggt_repo)
    from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri

    image_size = tuple(int(value) for value in payload["image_size"])
    baseline, predicted_intrinsics = pose_encoding_to_extri_intri(
        torch.as_tensor(payload["baseline_pose_encoding"])[None].float(),
        image_size_hw=image_size,
    )
    target, gt_intrinsics = pose_encoding_to_extri_intri(
        torch.as_tensor(payload["target_pose_encoding"])[None].float(),
        image_size_hw=image_size,
    )
    baseline = baseline[0].double().cpu()
    predicted_intrinsics = predicted_intrinsics[0].double().cpu()
    target = target[0].double().cpu()
    gt_intrinsics = gt_intrinsics[0].double().cpu()
    native_to_metric = float(payload["point_alignment_scale"])
    if not math.isfinite(native_to_metric) or native_to_metric <= 0.0:
        raise ValueError("O2.6 point alignment scale must be finite and positive.")

    memory_write = _memory_write(
        payload,
        min_track=data.fusion.min_track_confidence,
    )
    prepared = _prepare_tokens(
        payload=payload,
        token_count=config.support.token_count,
        max_history=config.support.history_count,
        baseline=baseline,
        intrinsics=predicted_intrinsics,
        gt_global_w2c=torch.as_tensor(payload["target_world_to_camera"]).double().cpu(),
        memory_write=memory_write,
        scale=native_to_metric,
    )
    predicted_depth = torch.as_tensor(payload["baseline_depth"]).float().cpu()
    gt_depth_native = (
        torch.as_tensor(payload["target_depth"]).float().cpu() / native_to_metric
    )
    scale_calibration = _calibrate_depth(
        predicted_depth,
        gt_depth_native,
        mode="scale",
        native_to_metric=native_to_metric,
        config=config.calibration,
    )
    affine_calibration = _calibrate_depth(
        predicted_depth,
        gt_depth_native,
        mode="affine",
        native_to_metric=native_to_metric,
        config=config.calibration,
    )
    dense_depths = {
        "gt_depth": gt_depth_native,
        "pred_depth": predicted_depth,
        "scale": _apply_calibration(predicted_depth, scale_calibration),
        "affine": _apply_calibration(predicted_depth, affine_calibration),
    }
    intrinsics = {
        "gt_intrinsics": gt_intrinsics,
        "pred_intrinsics": predicted_intrinsics,
    }
    local_features = _local_features_from_prepared(
        payload,
        config.support.token_count,
        expected_valid=prepared.local_valid,
    )
    geometry: dict[str, tuple[torch.Tensor, torch.Tensor]] = {
        "o1_direct_gt_camera": (prepared.gt_camera, prepared.gt_camera_valid),
    }
    for variant in VARIANTS[1:]:
        depth_key = (
            variant.calibration
            if variant.calibration != "none"
            else variant.depth_source
        )
        geometry[variant.name] = backproject_depth_at_local_tokens(
            dense_depths[depth_key],
            intrinsics[variant.intrinsics_source],
            local_features=local_features,
            local_valid=prepared.local_valid,
        )

    calibration_rows = _calibration_rows(
        frames=frames,
        scale_fit=scale_calibration,
        affine_fit=affine_calibration,
        predicted_intrinsics=predicted_intrinsics,
        gt_intrinsics=gt_intrinsics,
    )
    calibration_by_frame = {
        (str(row["calibration"]), int(row["sequence_index"])): row
        for row in calibration_rows
    }
    solver_families = tuple(
        SolverFamily(
            f"se3_{solver.name}",
            "SE3",
            solver,
        )
        for solver in config.theory.solvers
    ) + tuple(
        SolverFamily(
            f"sim3_{solver.name.replace('kabsch', 'umeyama')}",
            "Sim3",
            solver,
        )
        for solver in config.theory.solvers
    )
    positions = {frame: index for index, frame in enumerate(frames)}
    pair_cache: dict[tuple, CausalPairIndices] = {}
    frame_rows: list[dict[str, object]] = []
    for fold in FOLDS:
        for frame in fold.test_frames:
            sequence_index = positions[int(frame)]
            pairs = _pairs_for(
                pair_cache,
                prepared=prepared,
                current=sequence_index,
                spec=config.support,
            )
            for variant in VARIANTS:
                for solver in solver_families:
                    frame_rows.append(
                        _frame_row(
                            fold=fold.name,
                            frame_index=int(frame),
                            sequence_index=sequence_index,
                            variant=variant,
                            solver=solver,
                            pairs=pairs,
                            geometry=geometry[variant.name],
                            baseline=baseline,
                            target=target,
                            native_to_metric=native_to_metric,
                            config=config,
                            calibration_by_frame=calibration_by_frame,
                            predicted_intrinsics=predicted_intrinsics,
                            gt_intrinsics=gt_intrinsics,
                        )
                    )
    summary = _summarize(frame_rows, config=config)
    compact = _compact_decision(summary)
    medium_diagnosis = _medium_diagnosis(
        frame_rows,
        compact=compact,
        calibration_rows=calibration_rows,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = config.output_dir / "v80_o26_geometry_factorization.csv"
    compact_path = config.output_dir / "v80_o26_decision.csv"
    medium_path = config.output_dir / "v80_o26_medium_diagnosis.csv"
    frame_path = config.output_dir / "v80_o26_frame_diagnostics.csv"
    calibration_path = config.output_dir / "v80_o26_depth_intrinsics_diagnostics.csv"
    decision_path = config.output_dir / "v80_o26_decision.md"
    _write_csv(summary_path, summary)
    _write_csv(compact_path, compact)
    _write_csv(medium_path, medium_diagnosis)
    _write_csv(frame_path, frame_rows)
    _write_csv(calibration_path, calibration_rows)
    decision_path.write_text(_decision_markdown(summary), encoding="utf8")
    metadata = {
        "purpose": "V8_O2.6_no_training_depth_intrinsics_scale_factorization",
        "config": _jsonable(config),
        "cache": str(path),
        "cache_complete": bool(payload.get("complete", False)),
        "frames": list(frames),
        "correspondence": "GT-world pseudo-match; unchanged from O2.5",
        "history_pose": "GT diagnostic history for every branch",
        "oracle_calibration": "per-frame same-frame GT depth; diagnosis only",
        "outputs": {
            "summary": str(summary_path),
            "compact_decision": str(compact_path),
            "medium_diagnosis": str(medium_path),
            "frames": str(frame_path),
            "calibration": str(calibration_path),
            "decision": str(decision_path),
        },
    }
    (config.output_dir / "v80_o26_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf8",
    )
    print("V8 O2.6 COMPACT GEOMETRY DECISION (COPY THIS CSV)")
    print(compact_path.read_text(encoding="utf8").rstrip())
    print("V8 O2.6 MEDIUM FOUR-FRAME DIAGNOSIS (COPY THIS CSV)")
    print(medium_path.read_text(encoding="utf8").rstrip())
    print(f"V8 O2.6 full fold summary: {summary_path}")
    print(f"V8 O2.6 frame diagnostics: {frame_path}")
    print(f"V8 O2.6 depth/intrinsics diagnostics: {calibration_path}")
    print(f"V8 O2.6 decision: {decision_path}")
    return summary_path


def _local_features_from_prepared(
    payload: dict[str, Any],
    token_count: int,
    *,
    expected_valid: torch.Tensor,
) -> torch.Tensor:
    # Use the same deterministic local-token selector as O2.5.  Keeping this
    # call local prevents a dense-depth variant from changing support.
    from streaming_couping.scripts.run_v7_fusion_ablation import (
        _local_geometry_features,
    )

    local_features, local_valid = _local_geometry_features(
        payload,
        max_points=token_count,
    )
    if not torch.equal(local_valid.bool().cpu(), expected_valid.bool().cpu()):
        raise RuntimeError("O2.6 local-token selection changed between preparations.")
    return local_features.float().cpu()


def _calibrate_depth(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    mode: str,
    native_to_metric: float,
    config: CalibrationConfig,
) -> DepthCalibration:
    if predicted.shape != target.shape:
        raise ValueError("O2.6 predicted/GT depth shapes disagree.")
    if predicted.ndim != 4 or predicted.shape[-1] != 1:
        raise ValueError("O2.6 depth tensors must be [S,H,W,1].")
    if mode not in {"scale", "affine"}:
        raise ValueError(f"Unknown depth calibration mode={mode!r}.")
    rows = predicted.shape[0]
    scale = torch.ones(rows, dtype=torch.double)
    offset = torch.zeros(rows, dtype=torch.double)
    valid_pixels = torch.zeros(rows, dtype=torch.long)
    before_rmse = torch.full((rows,), float("nan"), dtype=torch.double)
    after_rmse = torch.full((rows,), float("nan"), dtype=torch.double)
    after_median = torch.full((rows,), float("nan"), dtype=torch.double)
    after_p90 = torch.full((rows,), float("nan"), dtype=torch.double)
    for frame in range(rows):
        x = predicted[frame, ..., 0].reshape(-1).double()
        y = target[frame, ..., 0].reshape(-1).double()
        valid = torch.isfinite(x) & torch.isfinite(y) & x.gt(1e-6) & y.gt(1e-6)
        indices = valid.nonzero(as_tuple=False)[:, 0]
        if indices.numel() > int(config.max_pixels):
            select = (
                torch.linspace(
                    0,
                    indices.numel() - 1,
                    int(config.max_pixels),
                )
                .round()
                .long()
            )
            indices = indices.index_select(0, select)
        valid_pixels[frame] = indices.numel()
        if indices.numel() < int(config.min_pixels):
            continue
        x = x.index_select(0, indices)
        y = y.index_select(0, indices)
        keep = torch.ones_like(x, dtype=torch.bool)
        a = x.new_tensor(1.0)
        b = x.new_tensor(0.0)
        for iteration in range(int(config.trim_iterations)):
            a, b = _fit_depth_line(x[keep], y[keep], affine=mode == "affine")
            residual = (a * x + b - y).abs()
            if iteration + 1 == int(config.trim_iterations):
                break
            cutoff = torch.quantile(residual[keep], float(config.trim_quantile))
            next_keep = residual.le(cutoff)
            if int(next_keep.sum()) < int(config.min_pixels) or torch.equal(
                next_keep, keep
            ):
                break
            keep = next_keep
        calibrated = a * x + b
        residual = (calibrated - y).abs() * native_to_metric
        scale[frame] = a
        offset[frame] = b
        before_rmse[frame] = torch.sqrt((x - y).square().mean()) * native_to_metric
        after_rmse[frame] = (
            torch.sqrt((calibrated - y).square().mean()) * native_to_metric
        )
        after_median[frame] = torch.quantile(residual, 0.50)
        after_p90[frame] = torch.quantile(residual, 0.90)
    return DepthCalibration(
        scale=scale,
        offset=offset,
        valid_pixels=valid_pixels,
        before_rmse_metric=before_rmse,
        after_rmse_metric=after_rmse,
        after_median_metric=after_median,
        after_p90_metric=after_p90,
    )


def _fit_depth_line(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    affine: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not affine:
        scale = (predicted * target).sum() / predicted.square().sum().clamp_min(1e-12)
        return scale.clamp_min(1e-8), predicted.new_tensor(0.0)
    design = torch.stack([predicted, torch.ones_like(predicted)], dim=-1)
    solution = torch.linalg.lstsq(design, target[:, None]).solution[:, 0]
    return solution[0].clamp_min(1e-8), solution[1]


def _apply_calibration(
    depth: torch.Tensor,
    fit: DepthCalibration,
) -> torch.Tensor:
    result = fit.scale[:, None, None, None].to(depth) * depth + fit.offset[
        :, None, None, None
    ].to(depth)
    valid = torch.isfinite(depth) & torch.isfinite(result) & result.gt(1e-6)
    return torch.where(valid, result, torch.full_like(result, float("nan")))


def _frame_row(
    *,
    fold: str,
    frame_index: int,
    sequence_index: int,
    variant: GeometryVariant,
    solver: SolverFamily,
    pairs: CausalPairIndices,
    geometry: tuple[torch.Tensor, torch.Tensor],
    baseline: torch.Tensor,
    target: torch.Tensor,
    native_to_metric: float,
    config: O26Config,
    calibration_by_frame: dict[tuple[str, int], dict[str, object]],
    predicted_intrinsics: torch.Tensor,
    gt_intrinsics: torch.Tensor,
) -> dict[str, object]:
    points, point_valid = geometry
    current, history, pair_valid = gather_pair_points(
        points,
        point_valid,
        current_frame=sequence_index,
        pairs=pairs,
    )
    current = current[pair_valid].double()
    history = history[pair_valid].double()
    history_frames = pairs.history_frames[pair_valid]
    current_slots = pairs.current_slots[pair_valid]
    current_ids = pairs.current_points[pair_valid]
    distances = pairs.gt_distances_metric[pair_valid].double()
    previous_world = transform_points(
        invert_rigid(target).index_select(0, history_frames),
        history,
    )
    weights = torch.exp(
        -distances / max(float(config.theory.match_weight_temperature_metric), 1e-12)
    )
    native_solver = replace(
        solver.base.config,
        inlier_distance=float(config.support.radius_metric)
        / max(native_to_metric, 1e-12),
    )
    candidate = _solve(
        current=current,
        target=previous_world,
        weights=weights,
        solver=solver,
        solver_config=native_solver,
        raw_pose=baseline[sequence_index],
        native_to_metric=native_to_metric,
        theory=config.theory,
    )
    effective = _effective_count(weights)
    unique_current = _unique_rows(current_slots, current_ids)
    support_pass = (
        effective >= config.gates.min_effective_correspondences
        and unique_current >= config.gates.min_unique_current_correspondences
    )
    reasons = list(candidate.reasons)
    if effective < config.gates.min_effective_correspondences:
        reasons.append("effective_correspondences_below_limit")
    if unique_current < config.gates.min_unique_current_correspondences:
        reasons.append("unique_current_points_below_limit")
    reliable = candidate.bounded_accept and support_pass
    raw_metrics = _pose_metrics(
        baseline[sequence_index],
        target[sequence_index],
        config.theory.translation_weight,
    )
    proposed_metrics = (
        _pose_metrics(
            candidate.candidate_w2c,
            target[sequence_index],
            config.theory.translation_weight,
        )
        if candidate.accepted_by_solver
        else (float("nan"), float("nan"), float("nan"))
    )
    refined = candidate.candidate_w2c if reliable else baseline[sequence_index].clone()
    reliable_metrics = _pose_metrics(
        refined,
        target[sequence_index],
        config.theory.translation_weight,
    )
    calibration = (
        calibration_by_frame.get((variant.calibration, sequence_index))
        if variant.calibration in {"scale", "affine"}
        else None
    )
    k_error = _intrinsics_error(
        predicted_intrinsics[sequence_index],
        gt_intrinsics[sequence_index],
    )
    return {
        "phase": "geometry_factorization",
        "fold": fold,
        "branch": variant.name,
        "oracle_level": variant.oracle_level,
        "depth_source": variant.depth_source,
        "intrinsics_source": variant.intrinsics_source,
        "depth_calibration": variant.calibration,
        "history_pose_source": "gt_history",
        "correspondence_source": "gt_world_pseudo_match",
        "solver": solver.name,
        "transform_family": solver.transform_family,
        "token_count": config.support.token_count,
        "history_count": config.support.history_count,
        "match_mode": config.support.match_mode,
        "radius_metric": config.support.radius_metric,
        "sequence_index": sequence_index,
        "frame_index": frame_index,
        "history_frame_indices": " ".join(
            str(int(EXPECTED_FRAMES[index]))
            for index in torch.unique(history_frames).tolist()
        ),
        "instance_slots": " ".join(
            str(int(slot)) for slot in torch.unique(current_slots).tolist()
        ),
        "participating_instances": int(torch.unique(current_slots).numel()),
        "correspondences": int(current.shape[0]),
        "effective_correspondences": effective,
        "unique_current_correspondences": unique_current,
        "retained_correspondences": candidate.retained_count,
        "fit_scale": candidate.fit_scale,
        "fit_abs_log_scale": (
            abs(math.log(candidate.fit_scale))
            if math.isfinite(candidate.fit_scale) and candidate.fit_scale > 0.0
            else float("nan")
        ),
        "fit_rmse_metric": candidate.fit_rmse_metric,
        "fit_p90_metric": candidate.fit_p90_metric,
        "fit_inlier_ratio": candidate.inlier_ratio,
        "source_extent_metric": candidate.source_extent_metric,
        "support_pass": int(support_pass),
        "bounded_accept": int(candidate.bounded_accept),
        "reliable_accept": int(reliable),
        "fallback_exact": int(
            reliable or torch.equal(refined, baseline[sequence_index])
        ),
        "reason": "accepted" if reliable else ";".join(reasons),
        "raw_rotation_error_deg": raw_metrics[0],
        "raw_center_error_native": raw_metrics[1],
        "raw_pose_loss": raw_metrics[2],
        "proposed_rotation_error_deg": proposed_metrics[0],
        "proposed_center_error_native": proposed_metrics[1],
        "proposed_pose_loss": proposed_metrics[2],
        "reliable_rotation_error_deg": reliable_metrics[0],
        "reliable_center_error_native": reliable_metrics[1],
        "reliable_pose_loss": reliable_metrics[2],
        "pose_improved": int(reliable_metrics[2] < raw_metrics[2] - 1e-12),
        "pose_worse": int(reliable_metrics[2] > raw_metrics[2] + 1e-12),
        "oracle_depth_scale": (
            float(calibration["depth_scale"]) if calibration else float("nan")
        ),
        "oracle_depth_offset_native": (
            float(calibration["depth_offset_native"]) if calibration else float("nan")
        ),
        "depth_rmse_before_metric": (
            float(calibration["before_rmse_metric"]) if calibration else float("nan")
        ),
        "depth_rmse_after_metric": (
            float(calibration["after_rmse_metric"]) if calibration else float("nan")
        ),
        "pred_fx_relative_error": k_error[0],
        "pred_fy_relative_error": k_error[1],
        "pred_cx_error_pixels": k_error[2],
        "pred_cy_error_pixels": k_error[3],
    }


def _solve(
    *,
    current: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    solver: SolverFamily,
    solver_config,
    raw_pose: torch.Tensor,
    native_to_metric: float,
    theory: TheoryConfig,
) -> Candidate:
    valid = torch.ones(current.shape[0], dtype=torch.bool)
    if solver.transform_family == "SE3":
        fit = weighted_kabsch(
            current,
            target,
            weights=weights,
            valid=valid,
            config=solver_config,
        )
        fit_scale = 1.0
    elif solver.transform_family == "Sim3":
        fit = weighted_umeyama(
            current,
            target,
            weights=weights,
            valid=valid,
            config=solver_config,
        )
        fit_scale = float(fit.scale)
    else:
        raise ValueError(f"Unknown transform family={solver.transform_family!r}.")
    candidate_w2c = invert_rigid(fit.transform)
    accepted_by_solver = bool(fit.accepted)
    rmse_metric = float(fit.rmse) * native_to_metric
    p90_metric = float(fit.p90_residual) * native_to_metric
    extent_metric = (
        float(
            torch.linalg.vector_norm(
                current.max(dim=0).values - current.min(dim=0).values
            )
        )
        * native_to_metric
        if current.shape[0]
        else 0.0
    )
    correction_rotation = (
        float(rotation_error_degrees(candidate_w2c, raw_pose))
        if accepted_by_solver
        else float("nan")
    )
    correction_center = (
        float(
            torch.linalg.vector_norm(
                camera_centers(candidate_w2c) - camera_centers(raw_pose)
            )
        )
        if accepted_by_solver
        else float("nan")
    )
    reasons: list[str] = []
    if int(fit.point_count) < int(solver_config.min_points):
        reasons.append("insufficient_correspondences")
    if bool(fit.degenerate):
        reasons.append("degenerate_geometry")
    if not math.isfinite(rmse_metric):
        reasons.append("non_finite_fit")
    elif rmse_metric > theory.max_fit_rmse_metric:
        reasons.append("fit_rmse_above_limit")
    if extent_metric < theory.min_source_extent_metric:
        reasons.append("source_extent_below_limit")
    if (
        math.isfinite(correction_rotation)
        and correction_rotation > theory.max_rotation_correction_degrees
    ):
        reasons.append("rotation_correction_above_limit")
    if (
        math.isfinite(correction_center)
        and correction_center > theory.max_center_shift_native
    ):
        reasons.append("center_correction_above_limit")
    bounded = (
        accepted_by_solver
        and math.isfinite(rmse_metric)
        and rmse_metric <= theory.max_fit_rmse_metric
        and extent_metric >= theory.min_source_extent_metric
        and correction_rotation <= theory.max_rotation_correction_degrees
        and correction_center <= theory.max_center_shift_native
    )
    return Candidate(
        candidate_w2c=candidate_w2c,
        accepted_by_solver=accepted_by_solver,
        bounded_accept=bounded,
        fit_scale=fit_scale,
        fit_rmse_metric=rmse_metric,
        fit_p90_metric=p90_metric,
        inlier_ratio=float(fit.inlier_ratio),
        retained_count=int(fit.retained_count),
        source_extent_metric=extent_metric,
        correction_rotation_deg=correction_rotation,
        correction_center_native=correction_center,
        reasons=reasons,
    )


def _summarize(
    rows: Sequence[dict[str, object]],
    *,
    config: O26Config,
) -> list[dict[str, object]]:
    names = (
        "phase",
        "fold",
        "branch",
        "oracle_level",
        "depth_source",
        "intrinsics_source",
        "depth_calibration",
        "history_pose_source",
        "correspondence_source",
        "solver",
        "transform_family",
        "token_count",
        "history_count",
        "match_mode",
        "radius_metric",
    )
    groups: dict[tuple, list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[name] for name in names), []).append(row)
    output = []
    for key, current in groups.items():
        raw_loss = _mean(float(row["raw_pose_loss"]) for row in current)
        reliable_loss = _mean(float(row["reliable_pose_loss"]) for row in current)
        active = sum(int(row["reliable_accept"]) for row in current)
        improved = sum(int(row["pose_improved"]) for row in current)
        worse = sum(int(row["pose_worse"]) for row in current)
        required = math.ceil(len(current) * config.gates.min_active_fraction)
        gain = _gain(raw_loss, reliable_loss)
        output.append(
            {
                **dict(zip(names, key)),
                "test_frames": " ".join(
                    str(int(row["frame_index"])) for row in current
                ),
                "frames": len(current),
                "mean_correspondences": _mean(
                    float(row["correspondences"]) for row in current
                ),
                "min_correspondences": min(
                    int(row["correspondences"]) for row in current
                ),
                "mean_effective_correspondences": _mean(
                    float(row["effective_correspondences"]) for row in current
                ),
                "mean_fit_scale": _finite_mean(
                    float(row["fit_scale"]) for row in current
                ),
                "std_fit_scale": _finite_std(
                    float(row["fit_scale"]) for row in current
                ),
                "mean_fit_abs_log_scale": _finite_mean(
                    float(row["fit_abs_log_scale"]) for row in current
                ),
                "mean_fit_rmse_metric": _finite_mean(
                    float(row["fit_rmse_metric"]) for row in current
                ),
                "mean_fit_p90_metric": _finite_mean(
                    float(row["fit_p90_metric"]) for row in current
                ),
                "mean_oracle_depth_scale": _finite_mean(
                    float(row["oracle_depth_scale"]) for row in current
                ),
                "std_oracle_depth_scale": _finite_std(
                    float(row["oracle_depth_scale"]) for row in current
                ),
                "mean_oracle_depth_offset_native": _finite_mean(
                    float(row["oracle_depth_offset_native"]) for row in current
                ),
                "mean_depth_rmse_before_metric": _finite_mean(
                    float(row["depth_rmse_before_metric"]) for row in current
                ),
                "mean_depth_rmse_after_metric": _finite_mean(
                    float(row["depth_rmse_after_metric"]) for row in current
                ),
                "mean_pred_fx_relative_error": _mean(
                    float(row["pred_fx_relative_error"]) for row in current
                ),
                "mean_pred_fy_relative_error": _mean(
                    float(row["pred_fy_relative_error"]) for row in current
                ),
                "mean_pred_cx_error_pixels": _mean(
                    float(row["pred_cx_error_pixels"]) for row in current
                ),
                "mean_pred_cy_error_pixels": _mean(
                    float(row["pred_cy_error_pixels"]) for row in current
                ),
                "raw_pose_loss": raw_loss,
                "proposed_pose_loss_valid_only": _finite_mean(
                    float(row["proposed_pose_loss"]) for row in current
                ),
                "reliable_active_frames": active,
                "reliable_pose_loss": reliable_loss,
                "reliable_gain_percent": gain,
                "reliable_improved_frames": improved,
                "reliable_worse_frames": worse,
                "reliable_fallback_exact": int(
                    all(int(row["fallback_exact"]) for row in current)
                ),
                "fold_robust_pass": int(
                    active >= required
                    and gain >= config.theory.minimum_gain_percent
                    and worse == 0
                    and all(int(row["fallback_exact"]) for row in current)
                ),
            }
        )
    return output


def _compact_decision(
    rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["branch"]), str(row["transform_family"]))
        groups.setdefault(key, []).append(row)
    output = []
    fold_names = tuple(fold.name for fold in FOLDS)
    for (branch, family), current in groups.items():
        by_solver: dict[str, list[dict[str, object]]] = {}
        for row in current:
            by_solver.setdefault(str(row["solver"]), []).append(row)

        def rank(item: tuple[str, list[dict[str, object]]]) -> tuple[int, int, float]:
            name, solver_rows = item
            passes = sum(int(row["fold_robust_pass"]) for row in solver_rows)
            gain = _finite_mean(
                float(row["reliable_gain_percent"]) for row in solver_rows
            )
            weighted = int("weighted" in name)
            return passes, weighted, gain

        solver_name, selected = max(by_solver.items(), key=rank)
        by_fold = {str(row["fold"]): row for row in selected}
        template = selected[0]
        result: dict[str, object] = {
            "branch": branch,
            "oracle_level": template["oracle_level"],
            "depth_source": template["depth_source"],
            "intrinsics_source": template["intrinsics_source"],
            "depth_calibration": template["depth_calibration"],
            "transform_family": family,
            "selected_solver": solver_name,
        }
        for fold in fold_names:
            row = by_fold[fold]
            result[f"{fold}_active_frames"] = row["reliable_active_frames"]
            result[f"{fold}_gain_percent"] = row["reliable_gain_percent"]
            result[f"{fold}_worse_frames"] = row["reliable_worse_frames"]
            result[f"{fold}_pass"] = row["fold_robust_pass"]
            result[f"{fold}_fit_scale"] = row["mean_fit_scale"]
            result[f"{fold}_fit_rmse_metric"] = row["mean_fit_rmse_metric"]
        result["all_fold_pass"] = int(
            all(int(by_fold[fold]["fold_robust_pass"]) for fold in fold_names)
        )
        output.append(result)
    return output


def _medium_diagnosis(
    frame_rows: Sequence[dict[str, object]],
    *,
    compact: Sequence[dict[str, object]],
    calibration_rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    branches = {
        "o26_gt_depth_pred_intrinsics",
        "o26_pred_depth_gt_intrinsics",
        "o26_pred_depth_pred_intrinsics",
        "o26_affine_calibrated_depth_gt_intrinsics",
    }
    selected_solver = {
        str(row["branch"]): str(row["selected_solver"])
        for row in compact
        if row["transform_family"] == "SE3" and row["branch"] in branches
    }
    calibration = {
        (str(row["calibration"]), int(row["sequence_index"])): row
        for row in calibration_rows
    }
    output = []
    for row in frame_rows:
        branch = str(row["branch"])
        if (
            row["fold"] != "medium"
            or branch not in branches
            or row["solver"] != selected_solver.get(branch)
        ):
            continue
        sequence_index = int(row["sequence_index"])
        scale_fit = calibration[("scale", sequence_index)]
        affine_fit = calibration[("affine", sequence_index)]
        raw_loss = float(row["raw_pose_loss"])
        reliable_loss = float(row["reliable_pose_loss"])
        output.append(
            {
                "fold": row["fold"],
                "frame_index": row["frame_index"],
                "branch": branch,
                "depth_source": row["depth_source"],
                "intrinsics_source": row["intrinsics_source"],
                "depth_calibration": row["depth_calibration"],
                "selected_solver": row["solver"],
                "history_frame_indices": row["history_frame_indices"],
                "instance_slots": row["instance_slots"],
                "participating_instances": row["participating_instances"],
                "correspondences": row["correspondences"],
                "effective_correspondences": row["effective_correspondences"],
                "unique_current_correspondences": row["unique_current_correspondences"],
                "fit_scale": row["fit_scale"],
                "fit_rmse_metric": row["fit_rmse_metric"],
                "fit_p90_metric": row["fit_p90_metric"],
                "reliable_accept": row["reliable_accept"],
                "reason": row["reason"],
                "raw_rotation_error_deg": row["raw_rotation_error_deg"],
                "raw_center_error_native": row["raw_center_error_native"],
                "raw_pose_loss": raw_loss,
                "proposed_rotation_error_deg": row["proposed_rotation_error_deg"],
                "proposed_center_error_native": row["proposed_center_error_native"],
                "proposed_pose_loss": row["proposed_pose_loss"],
                "reliable_pose_loss": reliable_loss,
                "frame_gain_percent": _gain(raw_loss, reliable_loss),
                "pose_improved": row["pose_improved"],
                "pose_worse": row["pose_worse"],
                "pred_fx_relative_error": row["pred_fx_relative_error"],
                "pred_fy_relative_error": row["pred_fy_relative_error"],
                "pred_cx_error_pixels": row["pred_cx_error_pixels"],
                "pred_cy_error_pixels": row["pred_cy_error_pixels"],
                "global_scale_depth_a": scale_fit["depth_scale"],
                "global_scale_depth_rmse_metric": scale_fit["after_rmse_metric"],
                "global_affine_depth_a": affine_fit["depth_scale"],
                "global_affine_depth_b_native": affine_fit["depth_offset_native"],
                "global_affine_depth_rmse_metric": affine_fit["after_rmse_metric"],
                "global_affine_depth_p90_metric": affine_fit["after_p90_metric"],
            }
        )
    expected = len(branches) * len(FOLDS[1].test_frames)
    if len(output) != expected:
        raise RuntimeError(
            "O2.6 medium diagnosis schema selection failed: "
            f"expected={expected}, got={len(output)}."
        )
    return output


def _calibration_rows(
    *,
    frames: Sequence[int],
    scale_fit: DepthCalibration,
    affine_fit: DepthCalibration,
    predicted_intrinsics: torch.Tensor,
    gt_intrinsics: torch.Tensor,
) -> list[dict[str, object]]:
    rows = []
    for mode, fit in (("scale", scale_fit), ("affine", affine_fit)):
        for index, frame in enumerate(frames):
            k_error = _intrinsics_error(
                predicted_intrinsics[index], gt_intrinsics[index]
            )
            rows.append(
                {
                    "sequence_index": index,
                    "frame_index": int(frame),
                    "calibration": mode,
                    "valid_pixels": int(fit.valid_pixels[index]),
                    "depth_scale": float(fit.scale[index]),
                    "depth_offset_native": float(fit.offset[index]),
                    "before_rmse_metric": float(fit.before_rmse_metric[index]),
                    "after_rmse_metric": float(fit.after_rmse_metric[index]),
                    "after_median_metric": float(fit.after_median_metric[index]),
                    "after_p90_metric": float(fit.after_p90_metric[index]),
                    "pred_fx_relative_error": k_error[0],
                    "pred_fy_relative_error": k_error[1],
                    "pred_cx_error_pixels": k_error[2],
                    "pred_cy_error_pixels": k_error[3],
                }
            )
    return rows


def _intrinsics_error(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> tuple[float, float, float, float]:
    fx = abs(float(predicted[0, 0] / target[0, 0].clamp_min(1e-12) - 1.0))
    fy = abs(float(predicted[1, 1] / target[1, 1].clamp_min(1e-12) - 1.0))
    cx = abs(float(predicted[0, 2] - target[0, 2]))
    cy = abs(float(predicted[1, 2] - target[1, 2]))
    return fx, fy, cx, cy


def _pose_metrics(
    pose: torch.Tensor,
    target: torch.Tensor,
    translation_weight: float,
) -> tuple[float, float, float]:
    rotation = float(rotation_error_degrees(pose, target))
    center = float(
        torch.linalg.vector_norm(camera_centers(pose) - camera_centers(target))
    )
    return rotation, center, rotation + translation_weight * center


def _decision_markdown(rows: Sequence[dict[str, object]]) -> str:
    def all_fold(branch: str, family: str) -> bool:
        selected = [
            row
            for row in rows
            if row["branch"] == branch and row["transform_family"] == family
        ]
        by_solver: dict[str, list[dict[str, object]]] = {}
        for row in selected:
            by_solver.setdefault(str(row["solver"]), []).append(row)
        return any(
            len({row["fold"] for row in solver_rows}) == len(FOLDS)
            and all(int(row["fold_robust_pass"]) for row in solver_rows)
            for solver_rows in by_solver.values()
        )

    flags = {
        "direct_o1": all_fold("o1_direct_gt_camera", "SE3"),
        "backprojection": all_fold("o26_gt_depth_gt_intrinsics", "SE3"),
        "gt_depth_pred_k": all_fold("o26_gt_depth_pred_intrinsics", "SE3"),
        "pred_depth_gt_k": all_fold("o26_pred_depth_gt_intrinsics", "SE3"),
        "current_o2": all_fold("o26_pred_depth_pred_intrinsics", "SE3"),
        "scale_gt_k": all_fold("o26_scale_calibrated_depth_gt_intrinsics", "SE3"),
        "scale_pred_k": all_fold("o26_scale_calibrated_depth_pred_intrinsics", "SE3"),
        "affine_gt_k": all_fold("o26_affine_calibrated_depth_gt_intrinsics", "SE3"),
        "affine_pred_k": all_fold("o26_affine_calibrated_depth_pred_intrinsics", "SE3"),
        "sim3_current": all_fold("o26_pred_depth_pred_intrinsics", "Sim3"),
    }
    lines = [
        "# V8 O2.6 geometry-factorization decision",
        "",
        "No model is trained. GT-world pseudo-correspondence and GT history are fixed.",
        "Scale/affine depth calibration reads same-frame GT and is oracle diagnosis only.",
        "",
    ]
    labels = {
        "direct_o1": "direct GT-camera O1 pass",
        "backprojection": "GT-depth + GT-intrinsics backprojection pass",
        "gt_depth_pred_k": "GT-depth + predicted-intrinsics pass",
        "pred_depth_gt_k": "predicted-depth + GT-intrinsics pass",
        "current_o2": "predicted-depth + predicted-intrinsics current O2 pass",
        "scale_gt_k": "oracle scale-depth + GT-intrinsics pass",
        "scale_pred_k": "oracle scale-depth + predicted-intrinsics pass",
        "affine_gt_k": "oracle affine-depth + GT-intrinsics pass",
        "affine_pred_k": "oracle affine-depth + predicted-intrinsics pass",
        "sim3_current": "uncalibrated predicted geometry Sim3 pass",
    }
    lines.extend(f"- {labels[key]}: `{int(value)}`" for key, value in flags.items())
    lines.extend(["", "Interpretation:", ""])
    if not flags["direct_o1"]:
        lines.append("- Direct O1 failed: stop and recheck support/solver regression.")
    elif not flags["backprojection"]:
        lines.append(
            "- Direct O1 passes but GT depth+K fails: backprojection/sampling is wrong."
        )
    else:
        lines.append(
            "- Predicted intrinsics are compatible with GT depth."
            if flags["gt_depth_pred_k"]
            else "- GT depth works only with GT K: predicted intrinsics materially damage pose recovery."
        )
        lines.append(
            "- Predicted depth is compatible when GT K is supplied."
            if flags["pred_depth_gt_k"]
            else "- Predicted depth fails even with GT K: depth materially damages pose recovery."
        )
        lines.append(
            "- Current O2 passes; geometry is sufficient for SAM correspondence O3."
            if flags["current_o2"]
            else "- Current predicted-depth + predicted-intrinsics O2 still fails."
        )
    if not flags["current_o2"]:
        if flags["scale_gt_k"] or flags["scale_pred_k"] or flags["sim3_current"]:
            lines.append(
                "- Scale-only correction forms an all-fold upper bound: per-frame depth scale drift is dominant."
            )
        elif flags["affine_gt_k"] or flags["affine_pred_k"]:
            lines.append(
                "- Affine but not scale-only correction passes: depth scale plus offset is dominant."
            )
        else:
            lines.append(
                "- Neither scale, affine nor Sim3 passes all folds: local depth shape/cross-frame consistency remains wrong."
            )
    lines.extend(
        [
            "- Do not start SAM O3 unless a predicted-geometry/calibrated upper bound passes all folds.",
            "",
        ]
    )
    return "\n".join(lines)


def load_o26_config(path: str | Path) -> O26Config:
    source = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf8")) or {}
    theory_path = Path(
        raw.get("theory_config", "streaming_couping/configs/v80_theory_validation.yaml")
    )
    if not theory_path.is_absolute():
        theory_path = (source.parents[2] / theory_path).resolve()
    theory = load_theory_config(theory_path)
    output = Path(
        raw.get("output_dir", "outputs/streaming_couping_v80_geometry_factorization")
    )
    if not output.is_absolute():
        output = (source.parents[2] / output).resolve()
    support = raw.get("fixed_support", {})
    calibration = raw.get("oracle_depth_calibration", {})
    gates = raw.get("gating", {})
    config = O26Config(
        source_path=source,
        theory=theory,
        output_dir=output,
        support=SupportSpec(
            token_count=int(support.get("token_count", 64)),
            history_count=int(support.get("history_count", 2)),
            match_mode=str(support.get("match_mode", "mutual")),
            radius_metric=float(support.get("radius_metric", 0.10)),
        ),
        calibration=CalibrationConfig(
            trim_quantile=float(calibration.get("trim_quantile", 0.90)),
            trim_iterations=int(calibration.get("trim_iterations", 3)),
            min_pixels=int(calibration.get("min_pixels", 128)),
            max_pixels=int(calibration.get("max_pixels", 65536)),
        ),
        gates=O26GateConfig(
            min_effective_correspondences=float(
                gates.get("min_effective_correspondences", 6.0)
            ),
            min_unique_current_correspondences=int(
                gates.get("min_unique_current_correspondences", 6)
            ),
            min_active_fraction=float(gates.get("min_active_fraction", 0.75)),
        ),
    )
    _validate_config(config)
    return config


def _validate_config(config: O26Config) -> None:
    if config.support != SupportSpec(64, 2, "mutual", 0.10):
        raise ValueError(
            "O2.6 locks the O2.5-selected support at K64/history2/mutual/0.10m."
        )
    if not 0.0 < config.calibration.trim_quantile <= 1.0:
        raise ValueError("O2.6 calibration trim quantile must be in (0,1].")
    if config.calibration.trim_iterations < 1:
        raise ValueError("O2.6 calibration trim iterations must be positive.")
    if (
        config.calibration.min_pixels < 3
        or config.calibration.max_pixels < config.calibration.min_pixels
    ):
        raise ValueError("O2.6 calibration pixel limits are invalid.")
    if config.gates.min_unique_current_correspondences < 3:
        raise ValueError("O2.6 needs at least three unique current points.")
    if not 0.0 < config.gates.min_active_fraction <= 1.0:
        raise ValueError("O2.6 active fraction must be in (0,1].")


def _mean(values) -> float:
    rows = list(values)
    return sum(rows) / len(rows) if rows else float("nan")


def _finite_mean(values) -> float:
    rows = [value for value in values if math.isfinite(value)]
    return _mean(rows)


def _finite_std(values) -> float:
    rows = [value for value in values if math.isfinite(value)]
    if not rows:
        return float("nan")
    mean = _mean(rows)
    return math.sqrt(_mean((value - mean) ** 2 for value in rows))


def _gain(initial: float, final: float) -> float:
    if not math.isfinite(initial) or not math.isfinite(final):
        return float("nan")
    return 0.0 if abs(initial) <= 1e-12 else 100.0 * (initial - final) / abs(initial)


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty O2.6 CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _jsonable(config: O26Config) -> dict[str, Any]:
    value = asdict(config)
    value["source_path"] = str(config.source_path)
    value["output_dir"] = str(config.output_dir)
    for key in ("source_path", "data_config", "output_dir"):
        value["theory"][key] = str(value["theory"][key])
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="streaming_couping/configs/v80_geometry_factorization.yaml",
    )
    parser.add_argument("--output-dir")
    return parser.parse_args()


if __name__ == "__main__":
    main()
