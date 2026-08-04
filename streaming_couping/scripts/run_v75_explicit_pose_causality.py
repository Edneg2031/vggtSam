#!/usr/bin/env python3
"""Run the no-training V7.5 SAM-mask/history pose causality experiment."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
import yaml

from streaming_couping.src.config import load_config
from streaming_couping.src.data import load_mask_tracking_sequence
from streaming_couping.src.explicit_pose_causality import (
    ExplicitPoseConfig,
    ExplicitPoseResult,
    run_explicit_pose_variant,
)
from streaming_couping.src.external_repos import maybe_add_repo_to_path
from streaming_couping.src.instance_observations import InstanceRefinementConfig
from streaming_couping.src.recovery import output_mask_to_stream
from streaming_couping.src.learned_pose.cache import (
    cache_path,
    load_feature_cache,
)
from streaming_couping.src.learned_pose.config import (
    RayPoseConfig,
    load_learned_pose_config,
)
from streaming_couping.src.v75_explicit_protocol import (
    EXPECTED_FRAMES,
    FOLDS,
    PRIMARY,
    SUMMARY_COLUMNS,
    annotate_decisions,
    run_specs,
    validate_protocol,
    validate_summary_rows,
)


@dataclass(frozen=True)
class V75Settings:
    source_path: Path
    data_config: Path
    output_dir: Path
    oracle_instance_ids: tuple[int, ...]
    translation_weight: float
    solver: ExplicitPoseConfig


def main() -> None:
    args = _parse_args()
    settings = _load_settings(args.config)
    if args.compute_device:
        settings = replace(
            settings,
            solver=replace(
                settings.solver,
                refinement=replace(
                    settings.solver.refinement,
                    compute_device=str(args.compute_device),
                ),
            ),
        )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else settings.output_dir
    )
    result = run_v75(settings=settings, output_dir=output_dir)
    print(f"V7.5 explicit-pose result={result}")


def run_v75(*, settings: V75Settings, output_dir: Path) -> Path:
    data = load_learned_pose_config(settings.data_config)
    if len(data.clips) != 1:
        raise ValueError("V7.5 expects exactly one V7.4 long-sequence clip.")
    clip = data.clips[0]
    path = cache_path(data, clip)
    if not path.is_file():
        raise FileNotFoundError(
            "V7.5 requires the V7.4 dynamic-instance cache. Run the one-click "
            f"command so it can build {path}."
        )
    payload = load_feature_cache(path)
    frames = tuple(int(value) for value in payload["frame_indices"])
    if frames != EXPECTED_FRAMES:
        raise ValueError(
            f"V7.5 requires frames 90:15:525, cache contains {frames}."
        )
    validate_protocol(available_frames=set(frames))
    if str(payload.get("instance_source", "")) != "sam31_online":
        raise ValueError("V7.5 requires the V7.4 sam31_online cache.")
    if int(payload["reference_sequence_index"]) != 0:
        raise ValueError("V7.5 requires frame 90 as camera gauge index 0.")

    recovery = load_config(
        data.recovery_config,
        overrides={
            "scene_id": clip.scene_id,
            "frame_indices": clip.frame_indices,
        },
    )
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
    baseline = baseline[0].detach().double().cpu()
    intrinsics = intrinsics[0].detach().double().cpu()
    target = target[0].detach().double().cpu()
    world_points = torch.as_tensor(payload["baseline_world_points"]).float().cpu()
    confidence = torch.as_tensor(payload["baseline_world_confidence"]).float().cpu()
    sam_masks = torch.as_tensor(payload["tracking_masks_stream"]).bool().cpu()
    sam_scores = torch.as_tensor(payload["tracking_scores"]).float().cpu()
    gt_masks = _load_oracle_masks(
        manifest=data.manifest,
        scene_id=clip.scene_id,
        frame_indices=frames,
        instance_ids=settings.oracle_instance_ids,
        processed_size=image_size,
        image_mode=recovery.image_mode,
        min_pixels=recovery.min_pixels,
        max_area_ratio=recovery.max_area_ratio,
        excluded_labels=recovery.excluded_labels,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    positions = {frame: index for index, frame in enumerate(frames)}
    for fold in FOLDS:
        history_indices = [positions[frame] for frame in fold.history_frames]
        test_indices = [positions[frame] for frame in fold.test_frames]
        raw_errors = _per_frame_center_errors(
            baseline,
            target,
            test_indices,
        )
        for variant, memory_mode in run_specs():
            print(
                f"V7.5 fold={fold.name} variant={variant.name} "
                f"memory={memory_mode}"
            )
            result = run_explicit_pose_variant(
                variant=variant,
                memory_mode=memory_mode,
                frame_indices=frames,
                history_indices=history_indices,
                test_indices=test_indices,
                baseline_world_to_camera=baseline,
                intrinsics=intrinsics,
                world_points=world_points,
                confidence=confidence,
                sam_masks=sam_masks,
                sam_scores=sam_scores,
                gt_masks=gt_masks,
                config=settings.solver,
            )
            _validate_result(
                result=result,
                baseline=baseline,
                test_indices=test_indices,
            )
            row = _summary_row(
                fold=fold,
                variant=variant,
                memory_mode=memory_mode,
                result=result,
                baseline=baseline,
                target=target,
                test_indices=test_indices,
                raw_errors=raw_errors,
                translation_weight=settings.translation_weight,
            )
            rows.append(row)
            frame_rows.extend(
                _evaluation_frame_rows(
                    fold=fold.name,
                    result=result,
                    baseline=baseline,
                    target=target,
                    test_indices=test_indices,
                    translation_weight=settings.translation_weight,
                )
            )
    _annotate_raw_gains(rows)
    annotate_decisions(rows)
    validate_summary_rows(rows)
    expected_frame_rows = sum(len(fold.test_frames) for fold in FOLDS) * len(
        run_specs()
    )
    if len(frame_rows) != expected_frame_rows:
        raise RuntimeError(
            "V7.5 per-frame result is incomplete: "
            f"expected={expected_frame_rows}, got={len(frame_rows)}."
        )
    summary_path = output_dir / "v75_explicit_pose_causality.csv"
    frame_path = output_dir / "v75_explicit_pose_frames.csv"
    dynamic_path = output_dir / "v75_dynamic_instance_diagnostics.csv"
    _write_csv(summary_path, rows, fieldnames=SUMMARY_COLUMNS)
    _write_csv(frame_path, frame_rows)
    _write_csv(
        dynamic_path,
        list(payload.get("dynamic_instance_diagnostics", ())),
    )
    decision_path = output_dir / "v75_decision_summary.md"
    decision_path.write_text(_decision_summary(rows), encoding="utf8")
    metadata = {
        "schema": 1,
        "purpose": "no_training_sam_mask_and_history_pose_causality",
        "learned_pose_head": False,
        "sam31_frozen": True,
        "streamvggt_frozen": True,
        "rotation_update": False,
        "camera_center_update": True,
        "same_scene_temporal_prefix_generalization": True,
        "cross_scene_generalization": False,
        "decision_metric": "mean_camera_center_error",
        "data_config": str(settings.data_config),
        "cache": str(path),
        "cache_sha256": _sha256(path),
        "oracle_instance_ids": settings.oracle_instance_ids,
        "folds": [asdict(fold) for fold in FOLDS],
        "solver": _settings_json(settings.solver),
        "summary_csv": str(summary_path),
        "frame_csv": str(frame_path),
        "dynamic_instance_csv": str(dynamic_path),
        "decision_summary": str(decision_path),
    }
    (output_dir / "v75_run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf8",
    )
    print("V7.5 EXPLICIT POSE CAUSALITY (COPY THIS CSV)")
    print(summary_path.read_text(encoding="utf8").rstrip())
    print("V7.5 DECISION SUMMARY")
    print(decision_path.read_text(encoding="utf8").rstrip())
    return summary_path


def _summary_row(
    *,
    fold,
    variant,
    memory_mode: str,
    result: ExplicitPoseResult,
    baseline: torch.Tensor,
    target: torch.Tensor,
    test_indices: Sequence[int],
    raw_errors: torch.Tensor,
    translation_weight: float,
) -> dict[str, Any]:
    metrics = _pose_metrics(
        result.world_to_camera,
        target,
        test_indices,
        translation_weight=translation_weight,
    )
    active_indices = [
        index for index in test_indices if bool(result.active_frames[index])
    ]
    inactive_indices = [
        index for index in test_indices if index not in active_indices
    ]
    diagnostics = [
        row
        for row in result.diagnostics
        if int(row["sequence_index"]) in set(test_indices)
    ]
    shifts = [float(row["applied_center_shift"]) for row in diagnostics]
    center_errors = _per_frame_center_errors(
        result.world_to_camera,
        target,
        test_indices,
    )
    return {
        "fold": fold.name,
        "variant": variant.name,
        "region_source": variant.region,
        "history_source": variant.history,
        "memory_mode": memory_mode,
        "report_only": int(variant.report_only),
        "history_frames": _frame_text(fold.history_frames),
        "test_frames": _frame_text(fold.test_frames),
        "test_frame_count": len(test_indices),
        "active_frames": len(active_indices),
        "active_frame_indices": _indices_to_frames(active_indices, result.diagnostics),
        "inactive_frames": len(inactive_indices),
        "inactive_frame_indices": _indices_to_frames(inactive_indices, result.diagnostics),
        "observed_test_frames": sum(
            int(row["observed_instances"]) > 0 for row in diagnostics
        ),
        "mean_observed_instances": _short(
            _mean(float(row["observed_instances"]) for row in diagnostics)
        ),
        "mean_map_instances_before": _short(
            _mean(float(row["map_instances_before"]) for row in diagnostics)
        ),
        "ray_accepted_frames": sum(
            str(row["fit_status"]) == "accepted" for row in diagnostics
        ),
        "icp_proposal_frames": sum(
            int(row["participating_instances"]) > 0 for row in diagnostics
        ),
        "mean_participating_instances": _short(
            _mean(float(row["participating_instances"]) for row in diagnostics)
        ),
        "mean_candidate_points": _short(
            _mean(float(row["candidate_points"]) for row in diagnostics)
        ),
        "mean_sampled_points": _short(
            _mean(float(row["sampled_points"]) for row in diagnostics)
        ),
        "mean_applied_center_shift": _short(_mean(shifts)),
        "max_applied_center_shift": _short(max(shifts, default=0.0)),
        "rotation_deg": _short(metrics["rotation_deg"]),
        "center_error": _short(metrics["center_error"]),
        "center_rmse": _short(metrics["center_rmse"]),
        "pose_loss": _short(metrics["pose_loss"]),
        "rpe_translation": _short(metrics["rpe_translation"]),
        "decision_metric": "center_error",
        "improved_frames_vs_raw": int((center_errors < raw_errors - 1e-9).sum()),
        "worse_frames_vs_raw": int((center_errors > raw_errors + 1e-9).sum()),
        "gain_vs_raw_percent": "",
        "gain_vs_current_only_percent": "",
        "gain_vs_wrong_id_percent": "",
        "worst_region_control_center_error": "",
        "best_region_control_center_error": "",
        "gain_vs_best_region_control_percent": "",
        "oracle_gain_vs_raw_percent": "",
        "oracle_solver_pass": 0,
        "sam_region_pass": 0,
        "sam_history_pass": 0,
        "all_folds_frozen_region_pass": 0,
        "all_folds_online_region_pass": 0,
        "all_folds_region_pass": 0,
        "all_folds_frozen_history_pass": 0,
        "all_folds_online_history_pass": 0,
        "all_folds_history_pass": 0,
    }


def _evaluation_frame_rows(
    *,
    fold: str,
    result: ExplicitPoseResult,
    baseline: torch.Tensor,
    target: torch.Tensor,
    test_indices: Sequence[int],
    translation_weight: float,
) -> list[dict[str, Any]]:
    baseline_centers = _camera_centers(baseline)
    predicted_centers = _camera_centers(result.world_to_camera)
    target_centers = _camera_centers(target)
    by_index = {
        int(row["sequence_index"]): row for row in result.diagnostics
    }
    rows = []
    for index in test_indices:
        row = {
            "fold": fold,
            **dict(by_index[int(index)]),
            "active": int(result.active_frames[index]),
            "raw_center_error": float(
                torch.linalg.vector_norm(
                    baseline_centers[index] - target_centers[index]
                )
            ),
            "refined_center_error": float(
                torch.linalg.vector_norm(
                    predicted_centers[index] - target_centers[index]
                )
            ),
            "raw_pose_loss": _single_frame_loss(
                baseline[index],
                target[index],
                translation_weight=translation_weight,
            ),
            "refined_pose_loss": _single_frame_loss(
                result.world_to_camera[index],
                target[index],
                translation_weight=translation_weight,
            ),
        }
        rows.append(row)
    return rows


def _pose_metrics(
    predicted: torch.Tensor,
    target: torch.Tensor,
    indices: Sequence[int],
    *,
    translation_weight: float,
) -> dict[str, float]:
    index = torch.tensor(indices, dtype=torch.long)
    predicted = predicted.index_select(0, index)
    target = target.index_select(0, index)
    relative = predicted[:, :3, :3] @ target[:, :3, :3].transpose(-1, -2)
    cosine = (torch.diagonal(relative, dim1=-2, dim2=-1).sum(dim=-1) - 1.0) * 0.5
    rotation = torch.rad2deg(torch.acos(cosine.clamp(-1.0, 1.0))).mean()
    predicted_centers = _camera_centers(predicted)
    target_centers = _camera_centers(target)
    errors = torch.linalg.vector_norm(predicted_centers - target_centers, dim=-1)
    rotation_loss = F.mse_loss(predicted[:, :3, :3], target[:, :3, :3])
    center_loss = F.smooth_l1_loss(
        predicted_centers,
        target_centers,
        beta=0.01,
    )
    rpe_values = []
    for current in range(len(indices) - 1):
        predicted_delta = predicted_centers[current + 1] - predicted_centers[current]
        target_delta = target_centers[current + 1] - target_centers[current]
        rpe_values.append(torch.linalg.vector_norm(predicted_delta - target_delta))
    rpe = (
        torch.sqrt(torch.stack(rpe_values).square().mean())
        if rpe_values
        else torch.zeros((), dtype=torch.float64)
    )
    return {
        "rotation_deg": float(rotation),
        "center_error": float(errors.mean()),
        "center_rmse": float(torch.sqrt(errors.square().mean())),
        "pose_loss": float(rotation_loss + float(translation_weight) * center_loss),
        "rpe_translation": float(rpe),
    }


def _single_frame_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    translation_weight: float,
) -> float:
    rotation = F.mse_loss(predicted[:3, :3], target[:3, :3])
    center = F.smooth_l1_loss(
        _camera_centers(predicted[None]),
        _camera_centers(target[None]),
        beta=0.01,
    )
    return float(rotation + float(translation_weight) * center)


def _load_oracle_masks(
    *,
    manifest: Path,
    scene_id: str,
    frame_indices: Sequence[int],
    instance_ids: Sequence[int],
    processed_size: tuple[int, int],
    image_mode: str,
    min_pixels: int,
    max_area_ratio: float,
    excluded_labels: Sequence[str],
) -> torch.Tensor:
    if not instance_ids:
        raise ValueError("V7.5 oracle_instance_ids must be non-empty.")
    sequence = load_mask_tracking_sequence(
        manifest,
        scene_id=scene_id,
        frame_indices=frame_indices,
        sequence_length=len(frame_indices),
        frame_stride=1,
        window_index=0,
        instance_id=int(instance_ids[0]),
        min_pixels=min_pixels,
        max_area_ratio=max_area_ratio,
        min_visible_frames=1,
        excluded_labels=excluded_labels,
        seed=0,
        allow_absent=True,
    )
    output = []
    for labels in sequence.instance_masks:
        slots = []
        for instance_id in instance_ids:
            grid = output_mask_to_stream(
                torch.from_numpy(labels == int(instance_id)),
                source_size=(int(labels.shape[0]), int(labels.shape[1])),
                processed_size=processed_size,
                image_mode=image_mode,
            )
            slots.append(grid.bool())
        output.append(torch.stack(slots))
    return torch.stack(output).bool()


def _annotate_raw_gains(rows: list[dict[str, Any]]) -> None:
    raw = {
        str(row["fold"]): float(row["center_error"])
        for row in rows
        if row["variant"] == "raw_streamvggt"
    }
    for row in rows:
        row["gain_vs_raw_percent"] = _short(
            _gain(raw[str(row["fold"])], float(row["center_error"]))
        )


def _decision_summary(rows: list[dict[str, Any]]) -> str:
    primary = [row for row in rows if row["variant"] == PRIMARY]
    online = [row for row in primary if row["memory_mode"] == "online"]
    all_region = int(online[0]["all_folds_region_pass"]) if online else 0
    all_history = int(online[0]["all_folds_history_pass"]) if online else 0
    frozen_region = (
        int(online[0]["all_folds_frozen_region_pass"]) if online else 0
    )
    online_region = (
        int(online[0]["all_folds_online_region_pass"]) if online else 0
    )
    frozen_history = (
        int(online[0]["all_folds_frozen_history_pass"]) if online else 0
    )
    online_history = (
        int(online[0]["all_folds_online_history_pass"]) if online else 0
    )
    lines = [
        "# V7.5 explicit-pose causality decision",
        "",
        "This run trains no pose model. SAM3.1 and StreamVGGT are frozen; "
        "only bounded ICP and the analytic V5 ray-centre solver are used.",
        "All gain/pass fields use mean camera-center error; pose_loss is "
        "reported only as a secondary metric.",
        "",
        f"- frozen-prefix all-fold region pass: `{frozen_region}`",
        f"- online-memory all-fold region pass: `{online_region}`",
        f"- strict all-fold/all-memory region pass: `{all_region}`",
        f"- frozen-prefix all-fold history pass: `{frozen_history}`",
        f"- online-memory all-fold history pass: `{online_history}`",
        f"- strict all-fold/all-memory history pass: `{all_history}`",
        "",
        "| fold | memory | gain vs raw | gain vs current | gain vs wrong ID | "
        "gain vs best region | oracle pass | region pass | history pass |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in primary:
        lines.append(
            "| {fold} | {memory_mode} | {gain_vs_raw_percent} | "
            "{gain_vs_current_only_percent} | {gain_vs_wrong_id_percent} | "
            "{gain_vs_best_region_control_percent} | {oracle_solver_pass} | "
            "{sam_region_pass} | {sam_history_pass} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- oracle pass=0: diagnose the frozen pointmap/ray solver before SAM.",
            "- region pass=1: the true SAM region beats full/bbox/random/stale controls.",
            "- history pass=1: correct persistent identity beats raw, current-only and wrong-ID.",
            "",
        ]
    )
    return "\n".join(lines)


def _load_settings(path: str | Path) -> V75Settings:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf8") as handle:
        raw = yaml.safe_load(handle) or {}
    solver = raw.get("solver", {})
    history = raw.get("history", {})
    data_config = _path(raw.get("data_config"), source.parent)
    output_dir = _path(raw.get("output_dir"), source.parent)
    ray = RayPoseConfig(
        confidence_threshold=float(solver.get("confidence_threshold", 0.30)),
        min_points=int(solver.get("min_ray_points", 128)),
        max_points=int(solver.get("matched_point_budget", 1024)),
        max_iterations=int(solver.get("max_iterations", 6)),
        max_condition_number=float(solver.get("max_condition_number", 1e8)),
        max_center_shift=float(solver.get("max_center_shift", 0.15)),
        max_residual_rmse=float(solver.get("max_ray_rmse", 0.20)),
        blend=float(solver.get("center_blend", 1.0)),
        angular_huber_delta=float(solver.get("angular_huber_delta", 0.02)),
        angular_min_range=float(solver.get("angular_min_range", 0.05)),
        preserve_reference=True,
        solver_modes=("current_refined",),
    )
    refinement = InstanceRefinementConfig(
        min_instance_points=int(history.get("min_instance_points", 128)),
        icp_max_points=int(history.get("icp_max_points", 512)),
        map_max_points=int(history.get("map_max_points", 2048)),
        icp_iterations=int(history.get("icp_iterations", 4)),
        icp_trim_quantile=float(history.get("icp_trim_quantile", 0.70)),
        min_icp_fitness=float(history.get("min_icp_fitness", 0.25)),
        max_icp_rmse=float(history.get("max_icp_rmse", 0.03)),
        correspondence_min_distance=float(
            history.get("correspondence_min_distance", 0.02)
        ),
        correspondence_object_ratio=float(
            history.get("correspondence_object_ratio", 0.05)
        ),
        max_proposal_translation=float(
            history.get("max_proposal_translation", 0.15)
        ),
        min_participating_instances=1,
        consensus_distance=float(history.get("consensus_distance", 0.02)),
        temporal_max_frame_gap=15,
        compute_device=str(history.get("compute_device", "cuda:0")),
    )
    explicit = ExplicitPoseConfig(
        point_confidence_threshold=float(solver.get("confidence_threshold", 0.30)),
        track_confidence_threshold=float(history.get("track_confidence_threshold", 0.50)),
        matched_point_budget=int(solver.get("matched_point_budget", 1024)),
        max_center_shift=float(solver.get("max_center_shift", 0.15)),
        center_blend=float(solver.get("center_blend", 1.0)),
        ray=ray,
        refinement=refinement,
    )
    settings = V75Settings(
        source_path=source,
        data_config=data_config,
        output_dir=output_dir,
        oracle_instance_ids=tuple(
            int(value) for value in raw.get("oracle_instance_ids", (37, 68, 54))
        ),
        translation_weight=float(raw.get("translation_weight", 10.0)),
        solver=explicit,
    )
    _validate_settings(settings)
    return settings


def _validate_settings(settings: V75Settings) -> None:
    solver = settings.solver
    if solver.matched_point_budget < solver.ray.min_points:
        raise ValueError("matched_point_budget must be >= min_ray_points.")
    if not 0.0 < solver.center_blend <= 1.0:
        raise ValueError("center_blend must be in (0,1].")
    if solver.max_center_shift <= 0.0:
        raise ValueError("max_center_shift must be positive.")
    if settings.translation_weight <= 0.0:
        raise ValueError("translation_weight must be positive.")


def _validate_result(
    *,
    result: ExplicitPoseResult,
    baseline: torch.Tensor,
    test_indices: Sequence[int],
) -> None:
    """Enforce the V7.5 no-training/fallback contract before scoring."""

    if result.world_to_camera.shape != baseline.shape:
        raise RuntimeError("V7.5 result/baseline pose shapes disagree.")
    if result.active_frames.shape != (baseline.shape[0],):
        raise RuntimeError("V7.5 active-frame vector has the wrong shape.")
    test = torch.zeros(baseline.shape[0], dtype=torch.bool)
    test[torch.as_tensor(test_indices, dtype=torch.long)] = True
    if bool((result.active_frames & ~test).any()):
        raise RuntimeError("V7.5 altered a history or future-outside-fold frame.")
    if not torch.equal(
        result.world_to_camera[:, :3, :3],
        baseline[:, :3, :3],
    ):
        raise RuntimeError("V7.5 must preserve every raw StreamVGGT rotation.")
    inactive = ~result.active_frames
    if not torch.equal(
        result.world_to_camera[inactive],
        baseline[inactive],
    ):
        raise RuntimeError("V7.5 inactive fallback is not bit-exact to raw pose.")


def _settings_json(config: ExplicitPoseConfig) -> dict[str, Any]:
    return {
        "point_confidence_threshold": config.point_confidence_threshold,
        "track_confidence_threshold": config.track_confidence_threshold,
        "matched_point_budget": config.matched_point_budget,
        "max_center_shift": config.max_center_shift,
        "center_blend": config.center_blend,
        "ray": asdict(config.ray),
        "refinement": asdict(config.refinement),
    }


def _camera_centers(world_to_camera: torch.Tensor) -> torch.Tensor:
    rotation = world_to_camera[..., :3, :3]
    translation = world_to_camera[..., :3, 3]
    return -(rotation.transpose(-1, -2) @ translation[..., None]).squeeze(-1)


def _per_frame_center_errors(
    predicted: torch.Tensor,
    target: torch.Tensor,
    indices: Sequence[int],
) -> torch.Tensor:
    centers = _camera_centers(predicted)
    target_centers = _camera_centers(target)
    index = torch.tensor(indices, dtype=torch.long)
    return torch.linalg.vector_norm(
        centers.index_select(0, index) - target_centers.index_select(0, index),
        dim=-1,
    )


def _indices_to_frames(
    indices: Sequence[int], diagnostics: Sequence[dict[str, object]]
) -> str:
    mapping = {
        int(row["sequence_index"]): int(row["frame_index"])
        for row in diagnostics
    }
    return " ".join(str(mapping[index]) for index in indices)


def _frame_text(values: Sequence[int]) -> str:
    return " ".join(str(int(value)) for value in values)


def _gain(initial: float, final: float) -> float:
    return 0.0 if initial <= 1e-12 else 100.0 * (initial - final) / initial


def _short(value: float) -> str:
    return f"{float(value):.8g}"


def _mean(values) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return sum(finite) / len(finite) if finite else 0.0


def _write_csv(
    path: Path,
    rows: Sequence[dict[str, Any]],
    *,
    fieldnames: Sequence[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        if fieldnames is None:
            path.write_text("", encoding="utf8")
            return
        fields = list(fieldnames)
    else:
        fields = list(fieldnames or _field_union(rows))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _field_union(rows: Sequence[dict[str, Any]]) -> list[str]:
    output: list[str] = []
    for row in rows:
        for name in row:
            if name not in output:
                output.append(name)
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _path(value: Any, parent: Path) -> Path:
    if value is None or not str(value).strip():
        raise ValueError("V7.5 configuration path is missing.")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (parent / path).resolve()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="streaming_couping/configs/v75_explicit_pose_causality.yaml",
    )
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--compute-device",
        help="ICP cdist device; use cpu if no CUDA device is available.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
