#!/usr/bin/env python3
"""Run coordinate audit and O1/O2 explicit-pose oracle validation."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
from typing import Any, Sequence

import torch
import yaml

from streaming_couping.src.config import load_config
from streaming_couping.src.external_repos import maybe_add_repo_to_path
from streaming_couping.src.learned_pose.cache import cache_path, load_feature_cache
from streaming_couping.src.learned_pose.config import load_learned_pose_config
from streaming_couping.src.solvers.weighted_kabsch import (
    KabschConfig,
    weighted_kabsch,
)
from streaming_couping.src.v74_temporal_protocol import EXPECTED_FRAMES, FOLDS
from streaming_couping.src.v80_pose_geometry import (
    backproject_depth_at_local_tokens,
    camera_centers,
    causal_gt_nearest_pairs,
    causal_history_indices,
    gather_pair_points,
    gt_camera_tokens_native,
    homogeneous,
    invert_rigid,
    rotation_error_degrees,
    sample_dense_at_local_tokens,
    transform_points,
)
from streaming_couping.scripts.run_v7_fusion_ablation import (
    _local_geometry_features,
)


@dataclass(frozen=True)
class OracleBranch:
    name: str
    oracle_level: str
    geometry_source: str
    history_pose_source: str


BRANCHES = (
    OracleBranch(
        "o1_gt_corr_gt_geometry_gt_history",
        "O1",
        "gt_camera_points",
        "gt_history",
    ),
    OracleBranch(
        "o2_gt_corr_stream_geometry_gt_history",
        "O2",
        "streamvggt_depth_camera_points",
        "gt_history",
    ),
    OracleBranch(
        "o2_gt_corr_stream_geometry_l0_history",
        "O2",
        "streamvggt_depth_camera_points",
        "l0_history",
    ),
)


@dataclass(frozen=True)
class SolverSpec:
    name: str
    config: KabschConfig


@dataclass(frozen=True)
class TheoryConfig:
    source_path: Path
    data_config: Path
    output_dir: Path
    clip_name: str
    token_count: int
    max_match_distance_metric: float
    match_weight_temperature_metric: float
    require_mutual_nearest: bool
    max_fit_rmse_metric: float
    min_source_extent_metric: float
    max_rotation_correction_degrees: float
    max_center_shift_native: float
    translation_weight: float
    minimum_gain_percent: float
    solvers: tuple[SolverSpec, ...]


def main() -> None:
    args = _parse_args()
    config = load_theory_config(args.config)
    if args.output_dir:
        config = replace(
            config, output_dir=Path(args.output_dir).expanduser().resolve()
        )
    result = run_theory_validation(config)
    print(f"V8 theory-validation result={result}")


def run_theory_validation(config: TheoryConfig) -> Path:
    data = load_learned_pose_config(config.data_config)
    clip = next((item for item in data.clips if item.name == config.clip_name), None)
    if clip is None:
        raise ValueError(f"V8 theory clip={config.clip_name!r} is not configured.")
    path = cache_path(data, clip)
    if not path.is_file():
        raise FileNotFoundError(
            "V8 theory validation reuses the V7.4 cache and never runs a "
            f"backbone. Missing cache: {path}"
        )
    payload = load_feature_cache(path)
    frames = tuple(int(value) for value in payload["frame_indices"])
    if frames != EXPECTED_FRAMES:
        raise ValueError(
            f"V8 theory validation requires frames 90:15:525, got {frames}."
        )
    if int(payload["reference_sequence_index"]) != 0:
        raise ValueError("V8 theory validation requires frame 90 as gauge.")
    _validate_payload(payload)

    recovery = load_config(data.recovery_config)
    maybe_add_repo_to_path(recovery.streamvggt_repo)
    from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri

    image_size = tuple(int(value) for value in payload["image_size"])
    baseline, baseline_intrinsics = pose_encoding_to_extri_intri(
        torch.as_tensor(payload["baseline_pose_encoding"])[None].float(),
        image_size_hw=image_size,
    )
    target, _ = pose_encoding_to_extri_intri(
        torch.as_tensor(payload["target_pose_encoding"])[None].float(),
        image_size_hw=image_size,
    )
    baseline = baseline[0].double().cpu()
    baseline_intrinsics = baseline_intrinsics[0].double().cpu()
    target = target[0].double().cpu()
    local_features, local_valid = _local_geometry_features(
        payload, max_points=config.token_count
    )
    local_features = local_features.float().cpu()
    local_valid = local_valid.bool().cpu()

    dense_gt = torch.as_tensor(payload["target_world_points"]).float().cpu()
    local_gt_world, local_gt_valid = sample_dense_at_local_tokens(
        dense_gt,
        local_features=local_features,
        local_valid=local_valid,
    )
    predicted_camera, predicted_camera_valid = backproject_depth_at_local_tokens(
        torch.as_tensor(payload["baseline_depth"]).float().cpu(),
        baseline_intrinsics,
        local_features=local_features,
        local_valid=local_valid,
    )
    scale = float(payload["point_alignment_scale"])
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("V8 theory cache point_alignment_scale must be positive.")
    gt_global_w2c = torch.as_tensor(payload["target_world_to_camera"]).double().cpu()
    gt_camera, gt_camera_valid = gt_camera_tokens_native(
        local_gt_world,
        local_gt_valid,
        gt_global_w2c,
        native_to_metric_scale=scale,
    )
    memory_write = _memory_write(payload, min_track=data.fusion.min_track_confidence)
    history = causal_history_indices(memory_write, local_valid)
    quality = torch.as_tensor(payload["quality"]).float().cpu()
    positions = {frame: index for index, frame in enumerate(frames)}

    config.output_dir.mkdir(parents=True, exist_ok=True)
    audit_rows = _coordinate_audit(
        frames=frames,
        payload=payload,
        baseline=baseline,
        target=target,
        local_features=local_features,
        local_valid=local_valid,
        local_gt_world=local_gt_world,
        local_gt_valid=local_gt_valid,
        gt_camera=gt_camera,
        gt_camera_valid=gt_camera_valid,
        predicted_camera=predicted_camera,
        predicted_camera_valid=predicted_camera_valid,
        gt_global_w2c=gt_global_w2c,
        scale=scale,
    )
    pair_rows: list[dict[str, object]] = []
    for fold in FOLDS:
        for frame in fold.test_frames:
            current = positions[int(frame)]
            pairs = causal_gt_nearest_pairs(
                current_frame=current,
                history_indices=history[current],
                gt_world_metric=local_gt_world,
                gt_valid=local_gt_valid,
                max_distance_metric=config.max_match_distance_metric,
                require_mutual_nearest=config.require_mutual_nearest,
            )
            for branch in BRANCHES:
                for solver in config.solvers:
                    pair_rows.append(
                        _oracle_row(
                            fold=fold.name,
                            frame_index=int(frame),
                            sequence_index=current,
                            frames=frames,
                            branch=branch,
                            solver=solver,
                            pairs=pairs,
                            baseline=baseline,
                            target=target,
                            gt_camera=gt_camera,
                            gt_camera_valid=gt_camera_valid,
                            predicted_camera=predicted_camera,
                            predicted_camera_valid=predicted_camera_valid,
                            quality=quality,
                            scale=scale,
                            config=config,
                        )
                    )
    summaries = _summary_rows(pair_rows, config=config)
    audit_path = config.output_dir / "v80_coordinate_audit.csv"
    pair_path = config.output_dir / "v80_oracle_pose_pairs.csv"
    summary_path = config.output_dir / "v80_oracle_pose_summary.csv"
    _write_csv(audit_path, audit_rows)
    _write_csv(pair_path, pair_rows)
    _write_csv(summary_path, summaries)
    decision_path = config.output_dir / "v80_theory_decision.md"
    decision_path.write_text(
        _decision_markdown(summaries, audit_rows), encoding="utf8"
    )
    metadata = {
        "purpose": "V8_coordinate_audit_and_O1_O2_explicit_pose_validation",
        "cache": str(path),
        "frames": list(frames),
        "coordinate_convention": {
            "pose": "world_to_camera",
            "kabsch": "current_camera_native_to_reference_gauge_native",
            "gt_camera": "metric_GT_camera_coordinates_divided_by_reference_Sim3_scale",
            "stream_geometry": "baseline_depth_backprojected_with_predicted_intrinsics",
            "history_target": "history_camera_points_placed_by_GT_or_L0_camera_to_world",
            "candidate": "inverse_of_solved_current_camera_to_world",
        },
        "gt_correspondence_label": "mutual_GT_world_nearest_neighbour_pseudo_correspondence",
        "config": _jsonable(config),
        "outputs": {
            "audit": str(audit_path),
            "pairs": str(pair_path),
            "summary": str(summary_path),
            "decision": str(decision_path),
        },
    }
    (config.output_dir / "v80_theory_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf8"
    )
    print("V8 THEORY VALIDATION SUMMARY (COPY THIS CSV)")
    print(summary_path.read_text(encoding="utf8").rstrip())
    print(f"V8 coordinate audit: {audit_path}")
    print(f"V8 pair diagnostics: {pair_path}")
    print(f"V8 decision: {decision_path}")
    return summary_path


def _oracle_row(
    *,
    fold: str,
    frame_index: int,
    sequence_index: int,
    frames: Sequence[int],
    branch: OracleBranch,
    solver: SolverSpec,
    pairs,
    baseline: torch.Tensor,
    target: torch.Tensor,
    gt_camera: torch.Tensor,
    gt_camera_valid: torch.Tensor,
    predicted_camera: torch.Tensor,
    predicted_camera_valid: torch.Tensor,
    quality: torch.Tensor,
    scale: float,
    config: TheoryConfig,
) -> dict[str, object]:
    if branch.geometry_source == "gt_camera_points":
        points, valid = gt_camera, gt_camera_valid
    else:
        points, valid = predicted_camera, predicted_camera_valid
    current, previous, pair_valid = gather_pair_points(
        points, valid, current_frame=sequence_index, pairs=pairs
    )
    current = current[pair_valid].double()
    previous = previous[pair_valid].double()
    history_frames = pairs.history_frames[pair_valid]
    history_slots = pairs.history_slots[pair_valid]
    current_slots = pairs.current_slots[pair_valid]
    gt_distance = pairs.gt_distances_metric[pair_valid].double()
    anchor = target if branch.history_pose_source == "gt_history" else baseline
    history_c2w = invert_rigid(anchor).index_select(0, history_frames)
    previous_world = transform_points(history_c2w, previous)
    weights = torch.exp(
        -gt_distance
        / max(float(config.match_weight_temperature_metric), 1e-12)
    )
    effective_correspondences = float(
        weights.sum().square() / weights.square().sum().clamp_min(1e-12)
    )
    native_solver = replace(
        solver.config,
        inlier_distance=float(solver.config.inlier_distance) / max(scale, 1e-12),
    )
    fit = weighted_kabsch(
        current,
        previous_world,
        weights=weights,
        valid=torch.ones(current.shape[0], dtype=torch.bool),
        config=native_solver,
    )
    candidate = invert_rigid(fit.transform)
    raw_pose = baseline[sequence_index]
    target_pose = target[sequence_index]
    fit_well_posed = bool(fit.accepted)
    correction_rotation = (
        float(rotation_error_degrees(candidate, raw_pose))
        if fit_well_posed
        else float("nan")
    )
    correction_center = (
        float(
            torch.linalg.vector_norm(
                camera_centers(candidate) - camera_centers(raw_pose)
            )
        )
        if fit_well_posed
        else float("nan")
    )
    fit_rmse_metric = float(fit.rmse) * scale
    source_extent = (
        float(
            torch.linalg.vector_norm(
                current.max(dim=0).values - current.min(dim=0).values
            )
        )
        * scale
        if current.shape[0]
        else 0.0
    )
    accepted = (
        fit_well_posed
        and math.isfinite(fit_rmse_metric)
        and fit_rmse_metric <= float(config.max_fit_rmse_metric)
        and source_extent >= float(config.min_source_extent_metric)
        and correction_rotation <= float(config.max_rotation_correction_degrees)
        and correction_center <= float(config.max_center_shift_native)
    )
    reasons = []
    if int(fit.point_count) < int(native_solver.min_points):
        reasons.append("insufficient_correspondences")
    if bool(fit.degenerate):
        reasons.append("degenerate_geometry")
    if not math.isfinite(fit_rmse_metric):
        reasons.append("non_finite_fit")
    elif fit_rmse_metric > float(config.max_fit_rmse_metric):
        reasons.append("fit_rmse_above_limit")
    if source_extent < float(config.min_source_extent_metric):
        reasons.append("source_extent_below_limit")
    if correction_rotation > float(config.max_rotation_correction_degrees):
        reasons.append("rotation_correction_above_limit")
    if correction_center > float(config.max_center_shift_native):
        reasons.append("center_correction_above_limit")
    refined = candidate if accepted else raw_pose.clone()
    fallback_exact = bool(torch.equal(refined, raw_pose)) if not accepted else True
    raw_rotation = float(rotation_error_degrees(raw_pose, target_pose))
    proposed_rotation = (
        float(rotation_error_degrees(candidate, target_pose))
        if fit_well_posed
        else float("nan")
    )
    refined_rotation = float(rotation_error_degrees(refined, target_pose))
    raw_center = float(
        torch.linalg.vector_norm(
            camera_centers(raw_pose) - camera_centers(target_pose)
        )
    )
    proposed_center = (
        float(
            torch.linalg.vector_norm(
                camera_centers(candidate) - camera_centers(target_pose)
            )
        )
        if fit_well_posed
        else float("nan")
    )
    refined_center = float(
        torch.linalg.vector_norm(
            camera_centers(refined) - camera_centers(target_pose)
        )
    )
    raw_loss = raw_rotation + float(config.translation_weight) * raw_center
    proposed_loss = (
        proposed_rotation + float(config.translation_weight) * proposed_center
    )
    refined_loss = (
        refined_rotation + float(config.translation_weight) * refined_center
    )
    unique_slots = torch.unique(history_slots).tolist()
    unique_history = torch.unique(history_frames).tolist()
    eigen = fit.source_eigenvalues.double() * (scale**2)
    return {
        "fold": fold,
        "branch": branch.name,
        "oracle_level": branch.oracle_level,
        "correspondence_source": "gt_world_mutual_nn_pseudo_match",
        "geometry_source": branch.geometry_source,
        "history_pose_source": branch.history_pose_source,
        "solver": solver.name,
        "sequence_index": sequence_index,
        "frame_index": frame_index,
        "history_sequence_indices": _int_text(unique_history),
        "history_frame_indices": _int_text([frames[index] for index in unique_history]),
        "instance_slots": _int_text(unique_slots),
        "participating_instances": len(unique_slots),
        "mean_current_track_confidence": _mean_tensor(
            quality[sequence_index, current_slots, 0]
        ),
        "mean_current_geometry_confidence": _mean_tensor(
            quality[sequence_index, current_slots, 1]
        ),
        "mean_current_static_score": _mean_tensor(
            quality[sequence_index, current_slots, 2]
        ),
        "mean_history_track_confidence": _mean_tensor(
            quality[history_frames, history_slots, 0]
        ),
        "mean_history_geometry_confidence": _mean_tensor(
            quality[history_frames, history_slots, 1]
        ),
        "mean_history_static_score": _mean_tensor(
            quality[history_frames, history_slots, 2]
        ),
        "correspondences": int(fit.point_count),
        "effective_correspondences": effective_correspondences,
        "retained_correspondences": int(fit.retained_count),
        "mean_gt_match_distance_metric": _mean_tensor(gt_distance),
        "median_gt_match_distance_metric": _quantile(gt_distance, 0.50),
        "p90_gt_match_distance_metric": _quantile(gt_distance, 0.90),
        "source_extent_metric": source_extent,
        "source_cov_eigenvalue_1_metric2": float(eigen[-1]),
        "source_cov_eigenvalue_2_metric2": float(eigen[-2]),
        "source_cov_eigenvalue_3_metric2": float(eigen[-3]),
        "secondary_eigenvalue_ratio": float(fit.secondary_eigenvalue_ratio),
        "condition_number": float(fit.condition_number),
        "degenerate": int(bool(fit.degenerate)),
        "fit_rmse_metric": fit_rmse_metric,
        "fit_median_metric": float(fit.median_residual) * scale,
        "fit_p90_metric": float(fit.p90_residual) * scale,
        "fit_inlier_ratio": float(fit.inlier_ratio),
        "reflection_corrected": int(bool(fit.reflection_corrected)),
        "proposed_rotation_correction_deg": correction_rotation,
        "proposed_center_correction_native": correction_center,
        "accepted": int(accepted),
        "fallback_reason": "accepted" if accepted else ";".join(reasons),
        "fallback_exact": int(fallback_exact),
        "raw_rotation_error_deg": raw_rotation,
        "proposed_rotation_error_deg": proposed_rotation,
        "refined_rotation_error_deg": refined_rotation,
        "rotation_improvement_deg": raw_rotation - refined_rotation,
        "raw_center_error_native": raw_center,
        "proposed_center_error_native": proposed_center,
        "refined_center_error_native": refined_center,
        "center_improvement_native": raw_center - refined_center,
        "raw_center_error_metric": raw_center * scale,
        "proposed_center_error_metric": proposed_center * scale,
        "refined_center_error_metric": refined_center * scale,
        "raw_pose_loss": raw_loss,
        "proposed_pose_loss": proposed_loss,
        "proposed_pose_gain_percent": _gain(raw_loss, proposed_loss),
        "refined_pose_loss": refined_loss,
        "pose_gain_percent": _gain(raw_loss, refined_loss),
        "pose_improved": int(refined_loss < raw_loss - 1e-12),
    }


def _coordinate_audit(
    *,
    frames,
    payload,
    baseline,
    target,
    local_features,
    local_valid,
    local_gt_world,
    local_gt_valid,
    gt_camera,
    gt_camera_valid,
    predicted_camera,
    predicted_camera_valid,
    gt_global_w2c,
    scale,
) -> list[dict[str, object]]:
    predicted_world, predicted_world_valid = sample_dense_at_local_tokens(
        torch.as_tensor(payload["baseline_world_points"]).float().cpu(),
        local_features=local_features,
        local_valid=local_valid,
    )
    baseline_c2w = invert_rigid(baseline)
    depth_world = transform_points(baseline_c2w[:, None], predicted_camera)
    reference_global = homogeneous(gt_global_w2c)[0]
    canonical_gt = transform_points(reference_global, local_gt_world) / max(scale, 1e-12)
    target_c2w = invert_rigid(target)
    gt_roundtrip = transform_points(target_c2w[:, None], gt_camera)
    rows = []
    for index, frame in enumerate(frames):
        pred_valid = predicted_camera_valid[index] & predicted_world_valid[index]
        gt_valid = gt_camera_valid[index] & local_gt_valid[index]
        pred_difference = torch.linalg.vector_norm(
            depth_world[index] - predicted_world[index], dim=-1
        )[pred_valid]
        gt_difference = torch.linalg.vector_norm(
            gt_roundtrip[index] - canonical_gt[index], dim=-1
        )[gt_valid]
        baseline_h = homogeneous(baseline[index])
        target_h = homogeneous(target[index])
        baseline_determinant = float(torch.det(baseline_h[:3, :3]))
        target_determinant = float(torch.det(target_h[:3, :3]))
        gt_roundtrip_p90 = _quantile(gt_difference, 0.90)
        gt_roundtrip_ok = (
            not gt_difference.numel() or gt_roundtrip_p90 < 1e-4
        )
        rows.append(
            {
                "sequence_index": index,
                "frame_index": int(frame),
                "point_alignment_scale": scale,
                "local_tokens": int(local_valid[index].sum()),
                "predicted_camera_valid": int(predicted_camera_valid[index].sum()),
                "gt_camera_valid": int(gt_camera_valid[index].sum()),
                "baseline_rotation_determinant": baseline_determinant,
                "target_rotation_determinant": target_determinant,
                "baseline_rotation_orthogonality_max": _orthogonality_error(baseline_h),
                "target_rotation_orthogonality_max": _orthogonality_error(target_h),
                "baseline_inverse_roundtrip_max": float(
                    (baseline_h @ invert_rigid(baseline_h) - torch.eye(4)).abs().max()
                ),
                "target_inverse_roundtrip_max": float(
                    (target_h @ invert_rigid(target_h) - torch.eye(4)).abs().max()
                ),
                "baseline_camera_center_norm_native": float(
                    torch.linalg.vector_norm(camera_centers(baseline_h))
                ),
                "target_camera_center_norm_native": float(
                    torch.linalg.vector_norm(camera_centers(target_h))
                ),
                "gt_pose_point_roundtrip_median_native": _quantile(gt_difference, 0.50),
                "gt_pose_point_roundtrip_p90_native": gt_roundtrip_p90,
                "depth_world_vs_point_head_median_native": _quantile(pred_difference, 0.50),
                "depth_world_vs_point_head_p90_native": _quantile(pred_difference, 0.90),
                "coordinate_audit_pass": int(
                    _orthogonality_error(baseline_h) < 1e-4
                    and _orthogonality_error(target_h) < 1e-4
                    and abs(baseline_determinant - 1.0) < 1e-4
                    and abs(target_determinant - 1.0) < 1e-4
                    and gt_roundtrip_ok
                ),
            }
        )
    return rows


def _summary_rows(
    pair_rows: Sequence[dict[str, object]], *, config: TheoryConfig
) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in pair_rows:
        key = (str(row["fold"]), str(row["branch"]), str(row["solver"]))
        groups.setdefault(key, []).append(row)
    output = []
    for (fold, branch, solver), rows in groups.items():
        first = rows[0]
        raw_rotation = _mean(float(row["raw_rotation_error_deg"]) for row in rows)
        proposed_rotation = _finite_mean(
            float(row["proposed_rotation_error_deg"]) for row in rows
        )
        refined_rotation = _mean(
            float(row["refined_rotation_error_deg"]) for row in rows
        )
        raw_center = _mean(float(row["raw_center_error_native"]) for row in rows)
        proposed_center = _finite_mean(
            float(row["proposed_center_error_native"]) for row in rows
        )
        refined_center = _mean(
            float(row["refined_center_error_native"]) for row in rows
        )
        raw_loss = _mean(float(row["raw_pose_loss"]) for row in rows)
        proposed_loss = _finite_mean(
            float(row["proposed_pose_loss"]) for row in rows
        )
        refined_loss = _mean(float(row["refined_pose_loss"]) for row in rows)
        active = sum(int(row["accepted"]) for row in rows)
        gain = _gain(raw_loss, refined_loss)
        output.append(
            {
                "fold": fold,
                "branch": branch,
                "oracle_level": first["oracle_level"],
                "correspondence_source": first["correspondence_source"],
                "geometry_source": first["geometry_source"],
                "history_pose_source": first["history_pose_source"],
                "solver": solver,
                "test_frames": " ".join(str(row["frame_index"]) for row in rows),
                "frames": len(rows),
                "active_frames": active,
                "fallback_frames": len(rows) - active,
                "mean_correspondences": _mean(
                    float(row["correspondences"]) for row in rows
                ),
                "mean_effective_correspondences": _mean(
                    float(row["effective_correspondences"]) for row in rows
                ),
                "mean_participating_instances": _mean(
                    float(row["participating_instances"]) for row in rows
                ),
                "mean_current_geometry_confidence": _finite_mean(
                    float(row["mean_current_geometry_confidence"]) for row in rows
                ),
                "mean_current_static_score": _finite_mean(
                    float(row["mean_current_static_score"]) for row in rows
                ),
                "mean_fit_rmse_metric": _finite_mean(
                    float(row["fit_rmse_metric"]) for row in rows
                ),
                "mean_inlier_ratio": _mean(
                    float(row["fit_inlier_ratio"]) for row in rows
                ),
                "degenerate_frames": sum(int(row["degenerate"]) for row in rows),
                "raw_rotation_error_deg": raw_rotation,
                "proposed_rotation_error_deg": proposed_rotation,
                "refined_rotation_error_deg": refined_rotation,
                "rotation_improvement_deg": raw_rotation - refined_rotation,
                "raw_center_error_native": raw_center,
                "proposed_center_error_native": proposed_center,
                "refined_center_error_native": refined_center,
                "center_improvement_native": raw_center - refined_center,
                "raw_center_error_metric": _mean(
                    float(row["raw_center_error_metric"]) for row in rows
                ),
                "proposed_center_error_metric": _mean(
                    float(row["proposed_center_error_metric"]) for row in rows
                ),
                "refined_center_error_metric": _mean(
                    float(row["refined_center_error_metric"]) for row in rows
                ),
                "raw_pose_loss": raw_loss,
                "proposed_pose_loss": proposed_loss,
                "proposed_pose_gain_percent": _gain(raw_loss, proposed_loss),
                "refined_pose_loss": refined_loss,
                "pose_gain_percent": gain,
                "improved_frames": sum(int(row["pose_improved"]) for row in rows),
                "worse_frames": sum(
                    int(float(row["refined_pose_loss"]) > float(row["raw_pose_loss"]) + 1e-12)
                    for row in rows
                ),
                "fallback_exact": int(all(int(row["fallback_exact"]) for row in rows)),
                "oracle_pass": int(
                    active > 0
                    and gain >= float(config.minimum_gain_percent)
                    and all(int(row["fallback_exact"]) for row in rows)
                ),
            }
        )
    return output


def _decision_markdown(
    summaries: Sequence[dict[str, object]], audit: Sequence[dict[str, object]]
) -> str:
    audit_pass = all(int(row["coordinate_audit_pass"]) for row in audit)
    lines = [
        "# V8 O1/O2 explicit-pose decision",
        "",
        "This experiment trains no network and reuses the frozen V7.4 cache.",
        "GT-world nearest neighbours are diagnostic pseudo-correspondences, not material-point labels.",
        "",
        f"- coordinate audit all-frame pass: `{int(audit_pass)}`",
    ]
    for level in ("O1", "O2"):
        rows = [row for row in summaries if row["oracle_level"] == level]
        lines.append(
            f"- {level} all-fold pass: `"
            f"{int(bool(rows) and all(int(row['oracle_pass']) for row in rows))}`"
        )
    lines.extend(
        [
            "",
            "| fold | oracle | geometry | history | solver | active | gain | pass |",
            "|---|---|---|---|---|---:|---:|---:|",
        ]
    )
    for row in summaries:
        lines.append(
            f"| {row['fold']} | {row['oracle_level']} | {row['geometry_source']} "
            f"| {row['history_pose_source']} | {row['solver']} "
            f"| {row['active_frames']}/{row['frames']} "
            f"| {float(row['pose_gain_percent']):.4f} | {row['oracle_pass']} |"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- O1 fails: fix solver/coordinate convention before studying SAM.",
            "- O1 passes and O2 GT-history fails: StreamVGGT depth geometry is the bottleneck.",
            "- O2 GT-history passes but L0-history fails: accumulated history pose is the bottleneck.",
            "- Both O2 variants pass: proceed to O3/O4 predicted-correspondence tests.",
            "",
        ]
    )
    return "\n".join(lines)


def load_theory_config(path: str | Path) -> TheoryConfig:
    source = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf8")) or {}
    matching = raw.get("matching", {})
    gating = raw.get("gating", {})
    solver_rows = raw.get("solvers", {})
    common = {
        "min_points": int(solver_rows.get("min_points", 6)),
        "min_secondary_eigenvalue_ratio": float(
            solver_rows.get("min_secondary_eigenvalue_ratio", 1e-4)
        ),
        "inlier_distance": float(matching.get("max_distance_metric", 0.10)),
    }
    solvers = (
        SolverSpec(
            "weighted_kabsch",
            KabschConfig(**common, trim_quantile=1.0, trim_iterations=1),
        ),
        SolverSpec(
            "trimmed_kabsch",
            KabschConfig(
                **common,
                trim_quantile=float(solver_rows.get("trim_quantile", 0.70)),
                trim_iterations=int(solver_rows.get("trim_iterations", 3)),
            ),
        ),
    )
    output = Path(
        raw.get(
            "output_dir",
            "outputs/streaming_couping_v80_theory_validation",
        )
    )
    if not output.is_absolute():
        output = (source.parents[2] / output).resolve()
    data_config = Path(raw.get("data_config", "v74_temporal_data.yaml"))
    if not data_config.is_absolute():
        candidate = (source.parent / data_config).resolve()
        data_config = (
            candidate
            if candidate.is_file()
            else (source.parents[2] / data_config).resolve()
        )
    config = TheoryConfig(
        source_path=source,
        data_config=data_config,
        output_dir=output,
        clip_name=str(
            raw.get(
                "clip_name",
                "00a231a370_90_525_step15_37_68_54",
            )
        ),
        token_count=int(raw.get("token_count", 32)),
        max_match_distance_metric=float(matching.get("max_distance_metric", 0.10)),
        match_weight_temperature_metric=float(
            matching.get("weight_temperature_metric", 0.025)
        ),
        require_mutual_nearest=bool(matching.get("require_mutual_nearest", True)),
        max_fit_rmse_metric=float(gating.get("max_fit_rmse_metric", 0.10)),
        min_source_extent_metric=float(
            gating.get("min_source_extent_metric", 0.02)
        ),
        max_rotation_correction_degrees=float(
            gating.get("max_rotation_correction_degrees", 15.0)
        ),
        max_center_shift_native=float(gating.get("max_center_shift_native", 0.50)),
        translation_weight=float(raw.get("translation_weight", 10.0)),
        minimum_gain_percent=float(raw.get("minimum_gain_percent", 1.0)),
        solvers=solvers,
    )
    if config.token_count < 6:
        raise ValueError("V8 theory validation needs at least six local tokens.")
    if (
        not math.isfinite(config.match_weight_temperature_metric)
        or config.match_weight_temperature_metric <= 0.0
    ):
        raise ValueError("V8 theory match weight temperature must be positive.")
    if (
        not math.isfinite(config.max_match_distance_metric)
        or config.max_match_distance_metric <= 0.0
    ):
        raise ValueError("V8 theory match distance must be finite and positive.")
    if (
        not math.isfinite(config.max_fit_rmse_metric)
        or config.max_fit_rmse_metric <= 0.0
    ):
        raise ValueError("V8 theory fit RMSE limit must be finite and positive.")
    if (
        not math.isfinite(config.min_source_extent_metric)
        or config.min_source_extent_metric < 0.0
    ):
        raise ValueError("V8 theory minimum source extent cannot be negative.")
    if (
        not math.isfinite(config.max_rotation_correction_degrees)
        or config.max_rotation_correction_degrees <= 0.0
    ):
        raise ValueError("V8 theory rotation correction limit must be positive.")
    if (
        not math.isfinite(config.max_center_shift_native)
        or config.max_center_shift_native <= 0.0
    ):
        raise ValueError("V8 theory center correction limit must be positive.")
    if not math.isfinite(config.translation_weight) or config.translation_weight < 0.0:
        raise ValueError("V8 theory translation weight must be finite and nonnegative.")
    if not math.isfinite(config.minimum_gain_percent):
        raise ValueError("V8 theory minimum gain must be finite.")
    for solver in config.solvers:
        solver.config.validate()
    return config


def _validate_payload(payload: dict[str, Any]) -> None:
    fields = (
        "baseline_pose_encoding",
        "target_pose_encoding",
        "baseline_depth",
        "baseline_world_points",
        "baseline_world_confidence",
        "target_world_points",
        "target_world_to_camera",
        "point_alignment_scale",
        "instance_uvd",
        "instance_uvd_valid",
        "observed",
        "identity_valid",
        "quality",
    )
    missing = [name for name in fields if name not in payload]
    if missing:
        raise ValueError(f"V8 theory cache lacks fields={missing}.")


def _memory_write(payload: dict[str, Any], *, min_track: float) -> torch.Tensor:
    return (
        torch.as_tensor(payload["observed"]).bool()
        & torch.as_tensor(payload["identity_valid"]).bool()
        & torch.as_tensor(payload["quality"])[..., 0].ge(float(min_track))
    ).cpu()


def _orthogonality_error(matrix: torch.Tensor) -> float:
    rotation = homogeneous(matrix)[:3, :3]
    return float((rotation @ rotation.T - torch.eye(3, dtype=rotation.dtype)).abs().max())


def _mean(values) -> float:
    rows = list(values)
    return sum(rows) / len(rows) if rows else float("nan")


def _finite_mean(values) -> float:
    rows = [value for value in values if math.isfinite(value)]
    return _mean(rows)


def _mean_tensor(value: torch.Tensor) -> float:
    return float(value.mean()) if value.numel() else float("nan")


def _quantile(value: torch.Tensor, q: float) -> float:
    return float(torch.quantile(value.double(), float(q))) if value.numel() else float("nan")


def _gain(initial: float, final: float) -> float:
    return 0.0 if abs(initial) <= 1e-12 else 100.0 * (initial - final) / abs(initial)


def _int_text(values: Sequence[int]) -> str:
    return " ".join(str(int(value)) for value in values)


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _jsonable(config: TheoryConfig) -> dict[str, Any]:
    value = asdict(config)
    for name in ("source_path", "data_config", "output_dir"):
        value[name] = str(value[name])
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="streaming_couping/configs/v80_theory_validation.yaml",
    )
    parser.add_argument("--output-dir")
    return parser.parse_args()


if __name__ == "__main__":
    main()
