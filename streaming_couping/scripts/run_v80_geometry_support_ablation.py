#!/usr/bin/env python3
"""V8 O2.5: no-training support and geometry-source pose ablation."""

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
    KabschResult,
    weighted_kabsch,
)
from streaming_couping.src.v74_temporal_protocol import EXPECTED_FRAMES, FOLDS
from streaming_couping.src.v80_pose_geometry import (
    CausalPairIndices,
    backproject_depth_at_local_tokens,
    camera_centers,
    causal_gt_nearest_pairs_multi_history,
    causal_history_bank_indices,
    gather_pair_points,
    gt_camera_tokens_native,
    invert_rigid,
    rotation_error_degrees,
    sample_dense_at_local_tokens,
    transform_points,
)
from streaming_couping.scripts.run_v7_fusion_ablation import (
    _local_geometry_features,
)
from streaming_couping.scripts.run_v80_theory_validation import (
    SolverSpec,
    TheoryConfig,
    _memory_write,
    _validate_payload,
    load_theory_config,
)


@dataclass(frozen=True)
class GeometryBranch:
    name: str
    oracle_level: str
    geometry_source: str
    history_pose_source: str
    pose_entangled: bool = False


BRANCHES = (
    GeometryBranch(
        "o1_gt_camera_gt_history",
        "O1",
        "gt_camera",
        "gt_history",
    ),
    GeometryBranch(
        "o2_depth_camera_gt_history",
        "O2",
        "depth_camera",
        "gt_history",
    ),
    GeometryBranch(
        "o2_depth_camera_l0_history",
        "O2",
        "depth_camera",
        "l0_history",
    ),
    GeometryBranch(
        "o2p_point_head_camera_gt_history",
        "O2P",
        "point_head_camera",
        "gt_history",
        True,
    ),
    GeometryBranch(
        "o2p_point_head_camera_l0_history",
        "O2P",
        "point_head_camera",
        "l0_history",
        True,
    ),
)


@dataclass(frozen=True)
class SupportSpec:
    token_count: int
    history_count: int
    match_mode: str
    radius_metric: float

    @property
    def mutual(self) -> bool:
        return self.match_mode == "mutual"


@dataclass(frozen=True)
class O25GateConfig:
    min_effective_correspondences: float
    min_unique_current_correspondences: int
    max_head_consistency_p90_metric: float
    min_instance_candidates: int
    max_instance_consensus_rotation_deg: float
    max_instance_consensus_center_native: float
    min_active_fraction: float


@dataclass(frozen=True)
class O25Config:
    source_path: Path
    theory: TheoryConfig
    output_dir: Path
    token_counts: tuple[int, ...]
    history_counts: tuple[int, ...]
    match_modes: tuple[str, ...]
    radii_metric: tuple[float, ...]
    primary: SupportSpec
    auto_select_primary: bool
    gates: O25GateConfig


@dataclass
class PreparedTokens:
    local_valid: torch.Tensor
    gt_world: torch.Tensor
    gt_world_valid: torch.Tensor
    gt_camera: torch.Tensor
    gt_camera_valid: torch.Tensor
    depth_camera: torch.Tensor
    depth_camera_valid: torch.Tensor
    point_head_camera: torch.Tensor
    point_head_camera_valid: torch.Tensor
    history_bank: torch.Tensor


@dataclass
class CandidateFit:
    fit: KabschResult
    candidate_w2c: torch.Tensor
    fit_well_posed: bool
    bounded_accepted: bool
    fit_rmse_metric: float
    source_extent_metric: float
    correction_rotation_deg: float
    correction_center_native: float
    reasons: list[str]


def main() -> None:
    args = _parse_args()
    config = load_o25_config(args.config)
    if args.output_dir:
        config = replace(
            config,
            output_dir=Path(args.output_dir).expanduser().resolve(),
        )
    result = run_o25_ablation(config)
    print(f"V8 O2.5 geometry-support result={result}")


def run_o25_ablation(config: O25Config) -> Path:
    data = load_learned_pose_config(config.theory.data_config)
    clip = next(
        (item for item in data.clips if item.name == config.theory.clip_name),
        None,
    )
    if clip is None:
        raise ValueError(f"O2.5 clip={config.theory.clip_name!r} is not configured.")
    path = cache_path(data, clip)
    if not path.is_file():
        raise FileNotFoundError(f"O2.5 requires retained V7.4 cache: {path}")
    payload = load_feature_cache(path)
    _validate_payload(payload)
    frames = tuple(int(value) for value in payload["frame_indices"])
    if frames != EXPECTED_FRAMES:
        raise ValueError(f"O2.5 requires frames 90:15:525, got {frames}.")
    if int(payload.get("reference_sequence_index", -1)) != 0:
        raise ValueError("O2.5 requires frame 90 as the fixed coordinate gauge.")

    recovery = load_config(data.recovery_config)
    maybe_add_repo_to_path(recovery.streamvggt_repo)
    from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri

    image_size = tuple(int(value) for value in payload["image_size"])
    baseline, intrinsics = pose_encoding_to_extri_intri(
        torch.as_tensor(payload["baseline_pose_encoding"])[None].float(),
        image_size_hw=image_size,
    )
    target, _ = pose_encoding_to_extri_intri(
        torch.as_tensor(payload["target_pose_encoding"])[None].float(),
        image_size_hw=image_size,
    )
    baseline = baseline[0].double().cpu()
    intrinsics = intrinsics[0].double().cpu()
    target = target[0].double().cpu()
    scale = float(payload["point_alignment_scale"])
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("O2.5 point alignment scale must be finite and positive.")
    gt_global_w2c = torch.as_tensor(
        payload["target_world_to_camera"]
    ).double().cpu()
    memory_write = _memory_write(
        payload,
        min_track=data.fusion.min_track_confidence,
    )
    quality = torch.as_tensor(payload["quality"]).float().cpu()
    prepared = {
        token_count: _prepare_tokens(
            payload=payload,
            token_count=token_count,
            max_history=max(config.history_counts),
            baseline=baseline,
            intrinsics=intrinsics,
            gt_global_w2c=gt_global_w2c,
            memory_write=memory_write,
            scale=scale,
        )
        for token_count in config.token_counts
    }
    positions = {frame: index for index, frame in enumerate(frames)}
    pair_cache: dict[tuple, CausalPairIndices] = {}
    frame_rows: list[dict[str, object]] = []

    support_specs = tuple(
        SupportSpec(token_count, history_count, mode, radius)
        for token_count in config.token_counts
        for history_count in config.history_counts
        for mode in config.match_modes
        for radius in config.radii_metric
    )
    o1 = BRANCHES[0]
    for spec in support_specs:
        for fold in FOLDS:
            for frame in fold.test_frames:
                current = positions[int(frame)]
                pairs = _pairs_for(
                    pair_cache,
                    prepared=prepared[spec.token_count],
                    current=current,
                    spec=spec,
                )
                for solver in config.theory.solvers:
                    frame_rows.append(
                        _frame_row(
                            phase="support_sweep",
                            fold=fold.name,
                            frame_index=int(frame),
                            sequence_index=current,
                            frames=frames,
                            spec=spec,
                            branch=o1,
                            solver=solver,
                            pairs=pairs,
                            tokens=prepared[spec.token_count],
                            baseline=baseline,
                            target=target,
                            quality=quality,
                            scale=scale,
                            theory=config.theory,
                            gates=config.gates,
                        )
                    )

    support_rows = _summarize(
        [row for row in frame_rows if row["phase"] == "support_sweep"],
        config=config,
    )
    spec = (
        _select_primary_support(support_rows, config=config)
        if config.auto_select_primary
        else config.primary
    )
    print(
        "V8 O2.5 selected primary support: "
        f"K={spec.token_count} history={spec.history_count} "
        f"mode={spec.match_mode} radius={spec.radius_metric:g}m"
    )
    for fold in FOLDS:
        for frame in fold.test_frames:
            current = positions[int(frame)]
            pairs = _pairs_for(
                pair_cache,
                prepared=prepared[spec.token_count],
                current=current,
                spec=spec,
            )
            for branch in BRANCHES:
                for solver in config.theory.solvers:
                    frame_rows.append(
                        _frame_row(
                            phase="geometry_ablation",
                            fold=fold.name,
                            frame_index=int(frame),
                            sequence_index=current,
                            frames=frames,
                            spec=spec,
                            branch=branch,
                            solver=solver,
                            pairs=pairs,
                            tokens=prepared[spec.token_count],
                            baseline=baseline,
                            target=target,
                            quality=quality,
                            scale=scale,
                            theory=config.theory,
                            gates=config.gates,
                        )
                    )

    geometry_rows = _summarize(
        [row for row in frame_rows if row["phase"] == "geometry_ablation"],
        config=config,
    )
    expected_support = len(support_specs) * len(FOLDS) * len(config.theory.solvers)
    expected_geometry = len(FOLDS) * len(BRANCHES) * len(config.theory.solvers)
    if len(support_rows) != expected_support:
        raise RuntimeError(
            f"O2.5 support summary rows={len(support_rows)}, "
            f"expected={expected_support}."
        )
    if len(geometry_rows) != expected_geometry:
        raise RuntimeError(
            f"O2.5 geometry summary rows={len(geometry_rows)}, "
            f"expected={expected_geometry}."
        )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    frame_path = config.output_dir / "v80_o25_frame_diagnostics.csv"
    support_path = config.output_dir / "v80_o25_support_ablation.csv"
    geometry_path = config.output_dir / "v80_o25_geometry_ablation.csv"
    _write_csv(frame_path, frame_rows)
    _write_csv(support_path, support_rows)
    _write_csv(geometry_path, geometry_rows)
    decision_path = config.output_dir / "v80_o25_decision.md"
    decision_path.write_text(
        _decision_markdown(support_rows, geometry_rows),
        encoding="utf8",
    )
    metadata = {
        "purpose": "V8_O2.5_no_training_geometry_support_ablation",
        "cache": str(path),
        "frames": list(frames),
        "config": _jsonable(config),
        "configured_primary_support": asdict(config.primary),
        "selected_primary_support": asdict(spec),
        "important_boundary": (
            "point_head_camera uses L0 W2C and is pose-entangled diagnostic only"
        ),
        "outputs": {
            "primary_geometry": str(geometry_path),
            "support_sweep": str(support_path),
            "frame_diagnostics": str(frame_path),
            "decision": str(decision_path),
        },
    }
    (config.output_dir / "v80_o25_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf8",
    )
    print("V8 O2.5 PRIMARY GEOMETRY ABLATION (COPY THIS CSV)")
    print(geometry_path.read_text(encoding="utf8").rstrip())
    print(f"V8 O2.5 support sweep: {support_path}")
    print(f"V8 O2.5 frame diagnostics: {frame_path}")
    print(f"V8 O2.5 decision: {decision_path}")
    return geometry_path


def _prepare_tokens(
    *,
    payload: dict[str, Any],
    token_count: int,
    max_history: int,
    baseline: torch.Tensor,
    intrinsics: torch.Tensor,
    gt_global_w2c: torch.Tensor,
    memory_write: torch.Tensor,
    scale: float,
) -> PreparedTokens:
    local_features, local_valid = _local_geometry_features(
        payload,
        max_points=token_count,
    )
    local_features = local_features.float().cpu()
    local_valid = local_valid.bool().cpu()
    gt_world, gt_world_valid = sample_dense_at_local_tokens(
        torch.as_tensor(payload["target_world_points"]).float().cpu(),
        local_features=local_features,
        local_valid=local_valid,
    )
    gt_camera, gt_camera_valid = gt_camera_tokens_native(
        gt_world,
        gt_world_valid,
        gt_global_w2c,
        native_to_metric_scale=scale,
    )
    depth_camera, depth_camera_valid = backproject_depth_at_local_tokens(
        torch.as_tensor(payload["baseline_depth"]).float().cpu(),
        intrinsics,
        local_features=local_features,
        local_valid=local_valid,
    )
    point_world, point_world_valid = sample_dense_at_local_tokens(
        torch.as_tensor(payload["baseline_world_points"]).float().cpu(),
        local_features=local_features,
        local_valid=local_valid,
    )
    point_camera = transform_points(baseline[:, None], point_world)
    point_camera_valid = point_world_valid & torch.isfinite(point_camera).all(dim=-1)
    point_camera = torch.where(
        point_camera_valid[..., None],
        point_camera,
        torch.zeros_like(point_camera),
    )
    history_bank = causal_history_bank_indices(
        memory_write,
        local_valid,
        max_history=max_history,
    )
    return PreparedTokens(
        local_valid=local_valid,
        gt_world=gt_world,
        gt_world_valid=gt_world_valid,
        gt_camera=gt_camera,
        gt_camera_valid=gt_camera_valid,
        depth_camera=depth_camera,
        depth_camera_valid=depth_camera_valid,
        point_head_camera=point_camera,
        point_head_camera_valid=point_camera_valid,
        history_bank=history_bank,
    )


def _pairs_for(
    cache: dict[tuple, CausalPairIndices],
    *,
    prepared: PreparedTokens,
    current: int,
    spec: SupportSpec,
) -> CausalPairIndices:
    key = (
        spec.token_count,
        spec.history_count,
        spec.match_mode,
        spec.radius_metric,
        current,
    )
    if key not in cache:
        cache[key] = causal_gt_nearest_pairs_multi_history(
            current_frame=current,
            history_indices=prepared.history_bank[
                current, :, : spec.history_count
            ],
            gt_world_metric=prepared.gt_world,
            gt_valid=prepared.gt_world_valid,
            max_distance_metric=spec.radius_metric,
            require_mutual_nearest=spec.mutual,
        )
    return cache[key]


def _geometry_tensors(
    tokens: PreparedTokens,
    branch: GeometryBranch,
) -> tuple[torch.Tensor, torch.Tensor]:
    if branch.geometry_source == "gt_camera":
        return tokens.gt_camera, tokens.gt_camera_valid
    if branch.geometry_source == "depth_camera":
        return tokens.depth_camera, tokens.depth_camera_valid
    if branch.geometry_source == "point_head_camera":
        return tokens.point_head_camera, tokens.point_head_camera_valid
    raise ValueError(f"Unknown O2.5 geometry source={branch.geometry_source!r}.")


def _frame_row(
    *,
    phase: str,
    fold: str,
    frame_index: int,
    sequence_index: int,
    frames: Sequence[int],
    spec: SupportSpec,
    branch: GeometryBranch,
    solver: SolverSpec,
    pairs: CausalPairIndices,
    tokens: PreparedTokens,
    baseline: torch.Tensor,
    target: torch.Tensor,
    quality: torch.Tensor,
    scale: float,
    theory: TheoryConfig,
    gates: O25GateConfig,
) -> dict[str, object]:
    points, valid = _geometry_tensors(tokens, branch)
    current, previous, pair_valid = gather_pair_points(
        points,
        valid,
        current_frame=sequence_index,
        pairs=pairs,
    )
    current = current[pair_valid].double()
    previous = previous[pair_valid].double()
    history_frames = pairs.history_frames[pair_valid]
    history_slots = pairs.history_slots[pair_valid]
    current_slots = pairs.current_slots[pair_valid]
    current_point_ids = pairs.current_points[pair_valid]
    history_point_ids = pairs.history_points[pair_valid]
    gt_distance = pairs.gt_distances_metric[pair_valid].double()
    anchor = target if branch.history_pose_source == "gt_history" else baseline
    history_c2w = invert_rigid(anchor).index_select(0, history_frames)
    previous_gauge = transform_points(history_c2w, previous)
    weights = torch.exp(
        -gt_distance
        / max(float(theory.match_weight_temperature_metric), 1e-12)
    )
    effective = _effective_count(weights)
    unique_current = _unique_rows(current_slots, current_point_ids)
    unique_history = _unique_rows(
        history_frames,
        history_slots,
        history_point_ids,
    )
    native_solver = replace(
        solver.config,
        inlier_distance=float(spec.radius_metric) / max(scale, 1e-12),
    )
    raw_pose = baseline[sequence_index]
    target_pose = target[sequence_index]
    combined = _solve_candidate(
        source=current,
        target=previous_gauge,
        weights=weights,
        solver=native_solver,
        raw_pose=raw_pose,
        scale=scale,
        theory=theory,
    )
    head_p90 = _head_consistency_p90_metric(
        tokens=tokens,
        pairs=pairs,
        pair_valid=pair_valid,
        sequence_index=sequence_index,
        scale=scale,
    )
    instance_count, consensus_rotation, consensus_center = _instance_consensus(
        current=current,
        previous_gauge=previous_gauge,
        weights=weights,
        slots=current_slots,
        current_point_ids=current_point_ids,
        solver=native_solver,
        raw_pose=raw_pose,
        scale=scale,
        theory=theory,
        min_effective_correspondences=gates.min_effective_correspondences,
        min_unique_current_correspondences=(
            gates.min_unique_current_correspondences
        ),
    )
    effective_pass = effective >= float(gates.min_effective_correspondences)
    unique_current_pass = unique_current >= int(
        gates.min_unique_current_correspondences
    )
    head_pass = (
        branch.geometry_source == "gt_camera"
        or (
            math.isfinite(head_p90)
            and head_p90 <= float(gates.max_head_consistency_p90_metric)
        )
    )
    consensus_pass = (
        instance_count >= int(gates.min_instance_candidates)
        and math.isfinite(consensus_rotation)
        and math.isfinite(consensus_center)
        and consensus_rotation
        <= float(gates.max_instance_consensus_rotation_deg)
        and consensus_center <= float(gates.max_instance_consensus_center_native)
    )
    bounded_accept = combined.bounded_accepted
    reliable_accept = (
        bounded_accept
        and effective_pass
        and unique_current_pass
        and head_pass
    )
    consensus_accept = reliable_accept and consensus_pass
    raw_metrics = _pose_metrics(raw_pose, target_pose, theory.translation_weight)
    proposal_metrics = (
        _pose_metrics(
            combined.candidate_w2c,
            target_pose,
            theory.translation_weight,
        )
        if combined.fit_well_posed
        else (float("nan"), float("nan"), float("nan"))
    )
    policy = {
        "bounded": _policy_metrics(
            accepted=bounded_accept,
            candidate=combined.candidate_w2c,
            raw=raw_pose,
            target=target_pose,
            translation_weight=theory.translation_weight,
        ),
        "reliable": _policy_metrics(
            accepted=reliable_accept,
            candidate=combined.candidate_w2c,
            raw=raw_pose,
            target=target_pose,
            translation_weight=theory.translation_weight,
        ),
        "consensus": _policy_metrics(
            accepted=consensus_accept,
            candidate=combined.candidate_w2c,
            raw=raw_pose,
            target=target_pose,
            translation_weight=theory.translation_weight,
        ),
    }
    reliable_reasons = list(combined.reasons)
    if not effective_pass:
        reliable_reasons.append("effective_correspondences_below_limit")
    if not unique_current_pass:
        reliable_reasons.append("unique_current_points_below_limit")
    if not head_pass:
        reliable_reasons.append("head_consistency_above_limit")
    consensus_reasons = list(reliable_reasons)
    if instance_count < int(gates.min_instance_candidates):
        consensus_reasons.append("insufficient_instance_candidates")
    elif not consensus_pass:
        consensus_reasons.append("instance_candidates_disagree")
    unique_slots = torch.unique(current_slots).tolist()
    unique_history = torch.unique(history_frames).tolist()
    row: dict[str, object] = {
        "phase": phase,
        "fold": fold,
        "branch": branch.name,
        "oracle_level": branch.oracle_level,
        "geometry_source": branch.geometry_source,
        "history_pose_source": branch.history_pose_source,
        "pose_entangled_diagnostic": int(branch.pose_entangled),
        "solver": solver.name,
        "token_count": spec.token_count,
        "history_count": spec.history_count,
        "match_mode": spec.match_mode,
        "radius_metric": spec.radius_metric,
        "sequence_index": sequence_index,
        "frame_index": frame_index,
        "history_frame_indices": _int_text(
            [frames[index] for index in unique_history]
        ),
        "instance_slots": _int_text(unique_slots),
        "participating_instances": len(unique_slots),
        "correspondences": int(combined.fit.point_count),
        "effective_correspondences": effective,
        "unique_current_correspondences": unique_current,
        "unique_history_correspondences": unique_history,
        "retained_correspondences": int(combined.fit.retained_count),
        "mean_gt_match_distance_metric": _mean_tensor(gt_distance),
        "p90_gt_match_distance_metric": _quantile(gt_distance, 0.90),
        "fit_well_posed": int(combined.fit_well_posed),
        "fit_rmse_metric": combined.fit_rmse_metric,
        "fit_inlier_ratio": float(combined.fit.inlier_ratio),
        "source_extent_metric": combined.source_extent_metric,
        "secondary_eigenvalue_ratio": float(
            combined.fit.secondary_eigenvalue_ratio
        ),
        "head_consistency_p90_metric": head_p90,
        "effective_correspondence_pass": int(effective_pass),
        "unique_current_correspondence_pass": int(unique_current_pass),
        "head_consistency_pass": int(head_pass),
        "instance_candidates": instance_count,
        "instance_consensus_rotation_deg": consensus_rotation,
        "instance_consensus_center_native": consensus_center,
        "instance_consensus_pass": int(consensus_pass),
        "proposed_rotation_correction_deg": (
            combined.correction_rotation_deg
        ),
        "proposed_center_correction_native": (
            combined.correction_center_native
        ),
        "bounded_accept": int(bounded_accept),
        "bounded_reason": (
            "accepted" if bounded_accept else ";".join(combined.reasons)
        ),
        "reliable_accept": int(reliable_accept),
        "reliable_reason": (
            "accepted" if reliable_accept else ";".join(reliable_reasons)
        ),
        "consensus_accept": int(consensus_accept),
        "consensus_reason": (
            "accepted" if consensus_accept else ";".join(consensus_reasons)
        ),
        "raw_rotation_error_deg": raw_metrics[0],
        "raw_center_error_native": raw_metrics[1],
        "raw_pose_loss": raw_metrics[2],
        "proposed_rotation_error_deg": proposal_metrics[0],
        "proposed_center_error_native": proposal_metrics[1],
        "proposed_pose_loss": proposal_metrics[2],
    }
    for name, values in policy.items():
        row[f"{name}_rotation_error_deg"] = values[0]
        row[f"{name}_center_error_native"] = values[1]
        row[f"{name}_pose_loss"] = values[2]
        row[f"{name}_fallback_exact"] = values[3]
        row[f"{name}_pose_improved"] = int(values[2] < raw_metrics[2] - 1e-12)
        row[f"{name}_pose_worse"] = int(values[2] > raw_metrics[2] + 1e-12)
    row["mean_current_geometry_confidence"] = _mean_tensor(
        quality[sequence_index, current_slots, 1]
    )
    row["mean_current_static_score"] = _mean_tensor(
        quality[sequence_index, current_slots, 2]
    )
    return row


def _solve_candidate(
    *,
    source: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    solver: KabschConfig,
    raw_pose: torch.Tensor,
    scale: float,
    theory: TheoryConfig,
) -> CandidateFit:
    fit = weighted_kabsch(
        source,
        target,
        weights=weights,
        valid=torch.ones(source.shape[0], dtype=torch.bool),
        config=solver,
    )
    candidate = invert_rigid(fit.transform)
    well_posed = bool(fit.accepted)
    fit_rmse_metric = float(fit.rmse) * scale
    extent = (
        float(
            torch.linalg.vector_norm(
                source.max(dim=0).values - source.min(dim=0).values
            )
        )
        * scale
        if source.shape[0]
        else 0.0
    )
    rotation = (
        float(rotation_error_degrees(candidate, raw_pose))
        if well_posed
        else float("nan")
    )
    center = (
        float(
            torch.linalg.vector_norm(
                camera_centers(candidate) - camera_centers(raw_pose)
            )
        )
        if well_posed
        else float("nan")
    )
    accepted = (
        well_posed
        and math.isfinite(fit_rmse_metric)
        and fit_rmse_metric <= float(theory.max_fit_rmse_metric)
        and extent >= float(theory.min_source_extent_metric)
        and rotation <= float(theory.max_rotation_correction_degrees)
        and center <= float(theory.max_center_shift_native)
    )
    reasons: list[str] = []
    if int(fit.point_count) < int(solver.min_points):
        reasons.append("insufficient_correspondences")
    if bool(fit.degenerate):
        reasons.append("degenerate_geometry")
    if not math.isfinite(fit_rmse_metric):
        reasons.append("non_finite_fit")
    elif fit_rmse_metric > float(theory.max_fit_rmse_metric):
        reasons.append("fit_rmse_above_limit")
    if extent < float(theory.min_source_extent_metric):
        reasons.append("source_extent_below_limit")
    if math.isfinite(rotation) and rotation > float(
        theory.max_rotation_correction_degrees
    ):
        reasons.append("rotation_correction_above_limit")
    if math.isfinite(center) and center > float(theory.max_center_shift_native):
        reasons.append("center_correction_above_limit")
    return CandidateFit(
        fit=fit,
        candidate_w2c=candidate,
        fit_well_posed=well_posed,
        bounded_accepted=accepted,
        fit_rmse_metric=fit_rmse_metric,
        source_extent_metric=extent,
        correction_rotation_deg=rotation,
        correction_center_native=center,
        reasons=reasons,
    )


def _instance_consensus(
    *,
    current: torch.Tensor,
    previous_gauge: torch.Tensor,
    weights: torch.Tensor,
    slots: torch.Tensor,
    current_point_ids: torch.Tensor,
    solver: KabschConfig,
    raw_pose: torch.Tensor,
    scale: float,
    theory: TheoryConfig,
    min_effective_correspondences: float,
    min_unique_current_correspondences: int,
) -> tuple[int, float, float]:
    candidates = []
    for slot in torch.unique(slots).tolist():
        selected = slots.eq(int(slot))
        result = _solve_candidate(
            source=current[selected],
            target=previous_gauge[selected],
            weights=weights[selected],
            solver=solver,
            raw_pose=raw_pose,
            scale=scale,
            theory=theory,
        )
        if (
            result.bounded_accepted
            and _effective_count(weights[selected])
            >= float(min_effective_correspondences)
            and int(torch.unique(current_point_ids[selected]).numel())
            >= int(min_unique_current_correspondences)
        ):
            candidates.append(result.candidate_w2c)
    if len(candidates) < 2:
        return len(candidates), float("nan"), float("nan")
    max_rotation = 0.0
    max_center = 0.0
    for left in range(len(candidates)):
        for right in range(left + 1, len(candidates)):
            max_rotation = max(
                max_rotation,
                float(
                    rotation_error_degrees(
                        candidates[left], candidates[right]
                    )
                ),
            )
            max_center = max(
                max_center,
                float(
                    torch.linalg.vector_norm(
                        camera_centers(candidates[left])
                        - camera_centers(candidates[right])
                    )
                ),
            )
    return len(candidates), max_rotation, max_center


def _head_consistency_p90_metric(
    *,
    tokens: PreparedTokens,
    pairs: CausalPairIndices,
    pair_valid: torch.Tensor,
    sequence_index: int,
    scale: float,
) -> float:
    current_depth, history_depth, depth_valid = gather_pair_points(
        tokens.depth_camera,
        tokens.depth_camera_valid,
        current_frame=sequence_index,
        pairs=pairs,
    )
    current_point, history_point, point_valid = gather_pair_points(
        tokens.point_head_camera,
        tokens.point_head_camera_valid,
        current_frame=sequence_index,
        pairs=pairs,
    )
    valid = pair_valid & depth_valid & point_valid
    if not bool(valid.any()):
        return float("nan")
    residual = torch.cat(
        [
            torch.linalg.vector_norm(
                current_depth[valid] - current_point[valid], dim=-1
            ),
            torch.linalg.vector_norm(
                history_depth[valid] - history_point[valid], dim=-1
            ),
        ]
    )
    return _quantile(residual.double() * scale, 0.90)


def _pose_metrics(
    pose: torch.Tensor,
    target: torch.Tensor,
    translation_weight: float,
) -> tuple[float, float, float]:
    rotation = float(rotation_error_degrees(pose, target))
    center = float(
        torch.linalg.vector_norm(camera_centers(pose) - camera_centers(target))
    )
    return rotation, center, rotation + float(translation_weight) * center


def _policy_metrics(
    *,
    accepted: bool,
    candidate: torch.Tensor,
    raw: torch.Tensor,
    target: torch.Tensor,
    translation_weight: float,
) -> tuple[float, float, float, int]:
    refined = candidate if accepted else raw.clone()
    metrics = _pose_metrics(refined, target, translation_weight)
    exact = int(True if accepted else torch.equal(refined, raw))
    return (*metrics, exact)


def _summarize(
    rows: Sequence[dict[str, object]],
    *,
    config: O25Config,
) -> list[dict[str, object]]:
    names = (
        "phase",
        "fold",
        "branch",
        "oracle_level",
        "geometry_source",
        "history_pose_source",
        "pose_entangled_diagnostic",
        "solver",
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
        base = dict(zip(names, key))
        raw_loss = _mean(float(row["raw_pose_loss"]) for row in current)
        result: dict[str, object] = {
            **base,
            "test_frames": _int_text([int(row["frame_index"]) for row in current]),
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
            "min_effective_correspondences": min(
                float(row["effective_correspondences"]) for row in current
            ),
            "mean_unique_current_correspondences": _mean(
                float(row["unique_current_correspondences"]) for row in current
            ),
            "min_unique_current_correspondences": min(
                int(row["unique_current_correspondences"]) for row in current
            ),
            "mean_participating_instances": _mean(
                float(row["participating_instances"]) for row in current
            ),
            "mean_fit_rmse_metric": _finite_mean(
                float(row["fit_rmse_metric"]) for row in current
            ),
            "mean_head_consistency_p90_metric": _finite_mean(
                float(row["head_consistency_p90_metric"]) for row in current
            ),
            "mean_instance_candidates": _mean(
                float(row["instance_candidates"]) for row in current
            ),
            "raw_rotation_error_deg": _mean(
                float(row["raw_rotation_error_deg"]) for row in current
            ),
            "raw_center_error_native": _mean(
                float(row["raw_center_error_native"]) for row in current
            ),
            "raw_pose_loss": raw_loss,
            "proposed_pose_loss_valid_only": _finite_mean(
                float(row["proposed_pose_loss"]) for row in current
            ),
        }
        required = math.ceil(len(current) * float(config.gates.min_active_fraction))
        for policy in ("bounded", "reliable", "consensus"):
            active = sum(int(row[f"{policy}_accept"]) for row in current)
            loss = _mean(float(row[f"{policy}_pose_loss"]) for row in current)
            improved = sum(
                int(row[f"{policy}_pose_improved"]) for row in current
            )
            worse = sum(int(row[f"{policy}_pose_worse"]) for row in current)
            exact = int(
                all(int(row[f"{policy}_fallback_exact"]) for row in current)
            )
            gain = _gain(raw_loss, loss)
            result[f"{policy}_active_frames"] = active
            result[f"{policy}_pose_loss"] = loss
            result[f"{policy}_gain_percent"] = gain
            result[f"{policy}_improved_frames"] = improved
            result[f"{policy}_worse_frames"] = worse
            result[f"{policy}_fallback_exact"] = exact
            result[f"{policy}_robust_pass"] = int(
                active >= required
                and gain >= float(config.theory.minimum_gain_percent)
                and worse == 0
                and bool(exact)
            )
        output.append(result)
    return output


def _select_primary_support(
    rows: Sequence[dict[str, object]],
    *,
    config: O25Config,
) -> SupportSpec:
    names = (
        "token_count",
        "history_count",
        "match_mode",
        "radius_metric",
    )
    groups: dict[tuple, list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[name] for name in names), []).append(row)
    expected = len(FOLDS) * len(config.theory.solvers)
    eligible = [
        key
        for key, current in groups.items()
        if len(current) == expected
        and all(int(row["reliable_robust_pass"]) for row in current)
    ]
    if not eligible:
        print(
            "V8 O2.5 warning: no support setting passed all folds/solvers; "
            "using configured fallback primary."
        )
        return config.primary
    selected = min(
        eligible,
        key=lambda key: (
            int(key[0]),
            int(key[1]),
            0 if str(key[2]) == "mutual" else 1,
            float(key[3]),
        ),
    )
    return SupportSpec(
        token_count=int(selected[0]),
        history_count=int(selected[1]),
        match_mode=str(selected[2]),
        radius_metric=float(selected[3]),
    )


def _decision_markdown(
    support: Sequence[dict[str, object]],
    geometry: Sequence[dict[str, object]],
) -> str:
    support_keys = (
        "token_count",
        "history_count",
        "match_mode",
        "radius_metric",
        "solver",
    )
    by_support: dict[tuple, list[dict[str, object]]] = {}
    for row in support:
        by_support.setdefault(tuple(row[key] for key in support_keys), []).append(row)
    all_fold_support = [
        key
        for key, rows in by_support.items()
        if len(rows) == len(FOLDS)
        and all(int(row["reliable_robust_pass"]) for row in rows)
    ]
    by_geometry: dict[tuple, list[dict[str, object]]] = {}
    for row in geometry:
        key = (row["branch"], row["solver"])
        by_geometry.setdefault(key, []).append(row)
    all_fold_geometry = [
        key
        for key, rows in by_geometry.items()
        if len(rows) == len(FOLDS)
        and all(int(row["reliable_robust_pass"]) for row in rows)
    ]
    branch_pass = {
        key: int(
            len(rows) == len(FOLDS)
            and all(int(row["reliable_robust_pass"]) for row in rows)
        )
        for key, rows in by_geometry.items()
    }
    selected_primary = (
        (
            geometry[0]["token_count"],
            geometry[0]["history_count"],
            geometry[0]["match_mode"],
            geometry[0]["radius_metric"],
        )
        if geometry
        else None
    )
    o1_pass = any(
        value
        for (branch, _), value in branch_pass.items()
        if str(branch).startswith("o1_")
    )
    depth_gt_pass = any(
        value
        for (branch, _), value in branch_pass.items()
        if branch == "o2_depth_camera_gt_history"
    )
    depth_l0_pass = any(
        value
        for (branch, _), value in branch_pass.items()
        if branch == "o2_depth_camera_l0_history"
    )
    lines = [
        "# V8 O2.5 geometry-support decision",
        "",
        "No model is trained. Correspondence remains GT-world pseudo-match.",
        "",
        f"- all-fold reliable support configurations: `{len(all_fold_support)}`",
        f"- all-fold reliable primary geometry branches: `{len(all_fold_geometry)}`",
        f"- automatically selected primary support: `{selected_primary}`",
        f"- selected-support O1 pass: `{int(bool(o1_pass))}`",
        f"- selected-support depth O2 GT-history pass: `{int(bool(depth_gt_pass))}`",
        f"- selected-support depth O2 L0-history pass: `{int(bool(depth_l0_pass))}`",
        "",
        "All-fold support configurations:",
        "",
    ]
    lines.extend(
        [f"- `{key}`" for key in all_fold_support]
        or ["- none"]
    )
    lines.extend(["", "All-fold primary geometry branches:", ""])
    lines.extend(
        [f"- `{key}`" for key in all_fold_geometry]
        or ["- none"]
    )
    lines.extend(["", "Interpretation:", ""])
    if not all_fold_support:
        lines.append(
            "- No O1 support setting passes all folds: instance overlap/history remains insufficient."
        )
    elif not o1_pass:
        lines.append(
            "- Support sweep has valid settings, but selected-support O1 failed; do not diagnose depth."
        )
    elif not depth_gt_pass:
        lines.append(
            "- O1 passes while depth O2 with GT history fails: predicted camera geometry is the bottleneck."
        )
    elif not depth_l0_pass:
        lines.append(
            "- Depth O2 passes with GT history but fails with L0 history: history pose drift is the bottleneck."
        )
    else:
        lines.append(
            "- Depth O2 passes with GT and L0 history: proceed to SAM correspondence O3."
        )
    lines.extend(
        [
            "- point-head O2P remains diagnostic only because its camera points use L0 W2C.",
            "",
        ]
    )
    return "\n".join(lines)


def load_o25_config(path: str | Path) -> O25Config:
    source = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf8")) or {}
    theory_path = Path(
        raw.get(
            "theory_config",
            "streaming_couping/configs/v80_theory_validation.yaml",
        )
    )
    if not theory_path.is_absolute():
        theory_path = (source.parents[2] / theory_path).resolve()
    theory = load_theory_config(theory_path)
    output = Path(
        raw.get(
            "output_dir",
            "outputs/streaming_couping_v80_geometry_support_ablation",
        )
    )
    if not output.is_absolute():
        output = (source.parents[2] / output).resolve()
    support = raw.get("support_sweep", {})
    primary = raw.get("primary_support", {})
    gating = raw.get("gating", {})
    token_counts = tuple(int(value) for value in support.get("token_counts", [32, 64]))
    history_counts = tuple(
        int(value) for value in support.get("history_counts", [1, 2, 4])
    )
    match_modes = tuple(str(value) for value in support.get("match_modes", ["mutual", "one_way"]))
    radii = tuple(float(value) for value in support.get("radii_metric", [0.10, 0.15]))
    primary_spec = SupportSpec(
        token_count=int(primary.get("token_count", 64)),
        history_count=int(primary.get("history_count", 4)),
        match_mode=str(primary.get("match_mode", "one_way")),
        radius_metric=float(primary.get("radius_metric", 0.15)),
    )
    gates = O25GateConfig(
        min_effective_correspondences=float(
            gating.get("min_effective_correspondences", 6.0)
        ),
        min_unique_current_correspondences=int(
            gating.get("min_unique_current_correspondences", 6)
        ),
        max_head_consistency_p90_metric=float(
            gating.get("max_head_consistency_p90_metric", 0.60)
        ),
        min_instance_candidates=int(gating.get("min_instance_candidates", 2)),
        max_instance_consensus_rotation_deg=float(
            gating.get("max_instance_consensus_rotation_deg", 5.0)
        ),
        max_instance_consensus_center_native=float(
            gating.get("max_instance_consensus_center_native", 0.25)
        ),
        min_active_fraction=float(gating.get("min_active_fraction", 0.75)),
    )
    config = O25Config(
        source_path=source,
        theory=theory,
        output_dir=output,
        token_counts=token_counts,
        history_counts=history_counts,
        match_modes=match_modes,
        radii_metric=radii,
        primary=primary_spec,
        auto_select_primary=bool(
            primary.get("auto_select_from_support", True)
        ),
        gates=gates,
    )
    _validate_o25_config(config)
    return config


def _validate_o25_config(config: O25Config) -> None:
    if not config.token_counts or min(config.token_counts) < 6:
        raise ValueError("O2.5 token counts must all be at least six.")
    if not config.history_counts or min(config.history_counts) < 1:
        raise ValueError("O2.5 history counts must be positive.")
    if set(config.match_modes) - {"mutual", "one_way"}:
        raise ValueError("O2.5 match modes must be mutual/one_way.")
    if not config.radii_metric or any(
        not math.isfinite(value) or value <= 0.0
        for value in config.radii_metric
    ):
        raise ValueError("O2.5 radii must be finite and positive.")
    if config.primary.token_count not in config.token_counts:
        raise ValueError("O2.5 primary token count is absent from sweep.")
    if config.primary.history_count not in config.history_counts:
        raise ValueError("O2.5 primary history count is absent from sweep.")
    if config.primary.match_mode not in config.match_modes:
        raise ValueError("O2.5 primary match mode is absent from sweep.")
    if config.primary.radius_metric not in config.radii_metric:
        raise ValueError("O2.5 primary radius is absent from sweep.")
    if not 0.0 < config.gates.min_active_fraction <= 1.0:
        raise ValueError("O2.5 minimum active fraction must be in (0,1].")
    positive = (
        config.gates.min_effective_correspondences,
        config.gates.max_head_consistency_p90_metric,
        config.gates.max_instance_consensus_rotation_deg,
        config.gates.max_instance_consensus_center_native,
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in positive):
        raise ValueError("O2.5 gate thresholds must be finite and positive.")
    if config.gates.min_instance_candidates < 2:
        raise ValueError("O2.5 consensus needs at least two instance candidates.")
    if config.gates.min_unique_current_correspondences < 3:
        raise ValueError("O2.5 needs at least three unique current points.")


def _effective_count(weights: torch.Tensor) -> float:
    if not weights.numel():
        return 0.0
    return float(
        weights.sum().square() / weights.square().sum().clamp_min(1e-12)
    )


def _unique_rows(*values: torch.Tensor) -> int:
    if not values or not values[0].numel():
        return 0
    if any(value.shape != values[0].shape for value in values):
        raise ValueError("O2.5 unique-index tensors must share shape.")
    return int(torch.unique(torch.stack(values, dim=-1), dim=0).shape[0])


def _mean(values) -> float:
    rows = list(values)
    return sum(rows) / len(rows) if rows else float("nan")


def _finite_mean(values) -> float:
    rows = [value for value in values if math.isfinite(value)]
    return _mean(rows)


def _mean_tensor(value: torch.Tensor) -> float:
    return float(value.mean()) if value.numel() else float("nan")


def _quantile(value: torch.Tensor, q: float) -> float:
    return (
        float(torch.quantile(value.double(), float(q)))
        if value.numel()
        else float("nan")
    )


def _gain(initial: float, final: float) -> float:
    if not math.isfinite(initial) or not math.isfinite(final):
        return float("nan")
    return 0.0 if abs(initial) <= 1e-12 else 100.0 * (initial - final) / abs(initial)


def _int_text(values: Sequence[int]) -> str:
    return " ".join(str(int(value)) for value in values)


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty O2.5 CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _jsonable(config: O25Config) -> dict[str, Any]:
    value = asdict(config)
    value["source_path"] = str(config.source_path)
    value["output_dir"] = str(config.output_dir)
    for name in ("source_path", "data_config", "output_dir"):
        value["theory"][name] = str(value["theory"][name])
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="streaming_couping/configs/v80_geometry_support_ablation.yaml",
    )
    parser.add_argument("--output-dir")
    return parser.parse_args()


if __name__ == "__main__":
    main()
