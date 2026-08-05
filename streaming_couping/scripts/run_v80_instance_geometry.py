#!/usr/bin/env python3
"""V8 O2.7: fixed-camera K and SAM-instance affine-depth diagnosis."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import torch
import yaml

from streaming_couping.scripts.run_v80_geometry_factorization import (
    CalibrationConfig,
    GeometryVariant,
    O26Config,
    SolverFamily,
    _apply_calibration,
    _calibrate_depth,
    _calibration_rows,
    _compact_decision,
    _frame_row,
    _gain,
    _local_features_from_prepared,
    _summarize,
    _write_csv,
    load_o26_config,
)
from streaming_couping.scripts.run_v80_geometry_support_ablation import (
    _pairs_for,
    _prepare_tokens,
)
from streaming_couping.scripts.run_v80_theory_validation import (
    _memory_write,
    _validate_payload,
)
from streaming_couping.src.config import load_config
from streaming_couping.src.external_repos import maybe_add_repo_to_path
from streaming_couping.src.learned_pose.cache import cache_path, load_feature_cache
from streaming_couping.src.learned_pose.config import load_learned_pose_config
from streaming_couping.src.v74_temporal_protocol import EXPECTED_FRAMES, FOLDS
from streaming_couping.src.v80_pose_geometry import (
    CausalPairIndices,
    backproject_depth_at_local_tokens,
    sample_dense_at_local_tokens,
)

DEPTH_MODES = ("raw", "global_affine", "instance_affine")
INTRINSICS_MODES = (
    "current_predicted",
    "reference_predicted",
    "causal_median_predicted",
    "gt",
)


@dataclass(frozen=True)
class O27Config:
    source_path: Path
    base: O26Config
    output_dir: Path
    instance_affine: CalibrationConfig


@dataclass
class InstanceAffineCalibration:
    scale: torch.Tensor
    offset: torch.Tensor
    valid: torch.Tensor
    valid_pixels: torch.Tensor
    before_rmse_metric: torch.Tensor
    after_rmse_metric: torch.Tensor
    after_p90_metric: torch.Tensor


def main() -> None:
    args = _parse_args()
    config = load_o27_config(args.config)
    if args.output_dir:
        config = replace(
            config,
            output_dir=Path(args.output_dir).expanduser().resolve(),
        )
    result = run_o27(config)
    print(f"V8 O2.7 instance-geometry result={result}")


def run_o27(config: O27Config) -> Path:
    data = load_learned_pose_config(config.base.theory.data_config)
    clip = next(
        (item for item in data.clips if item.name == config.base.theory.clip_name),
        None,
    )
    if clip is None:
        raise ValueError(
            f"O2.7 clip={config.base.theory.clip_name!r} is not configured."
        )
    path = cache_path(data, clip)
    if not path.is_file():
        raise FileNotFoundError(f"O2.7 requires retained/rebuilt V7.4 cache: {path}")
    payload = load_feature_cache(path)
    _validate_payload(payload)
    required = {"target_depth", "tracking_masks_stream", "target_pose_encoding"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"O2.7 cache lacks fields={sorted(missing)}; rebuild it.")
    frames = tuple(int(value) for value in payload["frame_indices"])
    if frames != EXPECTED_FRAMES:
        raise ValueError(f"O2.7 requires frames 90:15:525, got {frames}.")
    if int(payload.get("reference_sequence_index", -1)) != 0:
        raise ValueError("O2.7 requires frame 90 as the fixed coordinate gauge.")

    recovery = load_config(data.recovery_config)
    maybe_add_repo_to_path(recovery.streamvggt_repo)
    from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri

    image_size = tuple(int(value) for value in payload["image_size"])
    baseline, predicted_k = pose_encoding_to_extri_intri(
        torch.as_tensor(payload["baseline_pose_encoding"])[None].float(),
        image_size_hw=image_size,
    )
    target, gt_k = pose_encoding_to_extri_intri(
        torch.as_tensor(payload["target_pose_encoding"])[None].float(),
        image_size_hw=image_size,
    )
    baseline = baseline[0].double().cpu()
    predicted_k = predicted_k[0].double().cpu()
    target = target[0].double().cpu()
    gt_k = gt_k[0].double().cpu()
    native_to_metric = float(payload["point_alignment_scale"])
    if not math.isfinite(native_to_metric) or native_to_metric <= 0.0:
        raise ValueError("O2.7 point alignment scale must be finite and positive.")

    memory_write = _memory_write(
        payload,
        min_track=data.fusion.min_track_confidence,
    )
    prepared = _prepare_tokens(
        payload=payload,
        token_count=config.base.support.token_count,
        max_history=config.base.support.history_count,
        baseline=baseline,
        intrinsics=predicted_k,
        gt_global_w2c=torch.as_tensor(payload["target_world_to_camera"]).double().cpu(),
        memory_write=memory_write,
        scale=native_to_metric,
    )
    local_features = _local_features_from_prepared(
        payload,
        config.base.support.token_count,
        expected_valid=prepared.local_valid,
    )
    predicted_depth = torch.as_tensor(payload["baseline_depth"]).float().cpu()
    target_depth_native = (
        torch.as_tensor(payload["target_depth"]).float().cpu() / native_to_metric
    )
    masks = torch.as_tensor(payload["tracking_masks_stream"]).bool().cpu()
    _validate_masks(masks, predicted_depth)

    global_affine = _calibrate_depth(
        predicted_depth,
        target_depth_native,
        mode="affine",
        native_to_metric=native_to_metric,
        config=config.base.calibration,
    )
    instance_affine = _calibrate_instance_affine(
        predicted_depth,
        target_depth_native,
        masks,
        native_to_metric=native_to_metric,
        config=config.instance_affine,
    )
    k_modes = _intrinsics_modes(predicted_k, gt_k)
    dense_global_affine = _apply_calibration(predicted_depth, global_affine)

    variants: list[GeometryVariant] = []
    geometry: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for depth_mode in DEPTH_MODES:
        for k_mode in INTRINSICS_MODES:
            name = f"o27_{depth_mode}_{k_mode}_k"
            variants.append(
                GeometryVariant(
                    name=name,
                    depth_source=depth_mode,
                    intrinsics_source=k_mode,
                    calibration=depth_mode,
                    oracle_level=(
                        "O2.7-causal-K-component"
                        if depth_mode == "raw" and k_mode != "gt"
                        else "O2.7-oracle-depth"
                        if depth_mode != "raw"
                        else "O2.7-oracle-K"
                    ),
                )
            )
            if depth_mode == "raw":
                geometry[name] = backproject_depth_at_local_tokens(
                    predicted_depth,
                    k_modes[k_mode],
                    local_features=local_features,
                    local_valid=prepared.local_valid,
                )
            elif depth_mode == "global_affine":
                geometry[name] = backproject_depth_at_local_tokens(
                    dense_global_affine,
                    k_modes[k_mode],
                    local_features=local_features,
                    local_valid=prepared.local_valid,
                )
            else:
                geometry[name] = _backproject_instance_affine(
                    predicted_depth,
                    k_modes[k_mode],
                    fit=instance_affine,
                    local_features=local_features,
                    local_valid=prepared.local_valid,
                )

    global_rows = _calibration_rows(
        frames=frames,
        scale_fit=global_affine,
        affine_fit=global_affine,
        predicted_intrinsics=predicted_k,
        gt_intrinsics=gt_k,
    )
    global_by_frame = {
        (str(row["calibration"]), int(row["sequence_index"])): row
        for row in global_rows
    }
    solvers = tuple(
        SolverFamily(f"se3_{solver.name}", "SE3", solver)
        for solver in config.base.theory.solvers
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
                spec=config.base.support,
            )
            for variant in variants:
                for solver in solvers:
                    row = _frame_row(
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
                        config=config.base,
                        calibration_by_frame=global_by_frame,
                        predicted_intrinsics=predicted_k,
                        gt_intrinsics=gt_k,
                    )
                    _annotate_o27_row(
                        row,
                        sequence_index=sequence_index,
                        selected_k=k_modes[variant.intrinsics_source][sequence_index],
                        gt_k=gt_k[sequence_index],
                        instance_fit=instance_affine,
                    )
                    frame_rows.append(row)

    summary = _summarize(frame_rows, config=config.base)
    compact = _compact_decision(summary)
    medium = _medium_diagnosis(frame_rows, compact=compact)
    instance_rows = _instance_calibration_rows(
        frames,
        instance_affine,
        memory_write=memory_write,
    )
    decision = _decision_markdown(compact, medium)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    compact_path = config.output_dir / "v80_o27_decision.csv"
    medium_path = config.output_dir / "v80_o27_medium_diagnosis.csv"
    summary_path = config.output_dir / "v80_o27_fold_summary.csv"
    frame_path = config.output_dir / "v80_o27_frame_diagnostics.csv"
    instance_path = config.output_dir / "v80_o27_instance_affine.csv"
    decision_path = config.output_dir / "v80_o27_decision.md"
    _write_csv(compact_path, compact)
    _write_csv(medium_path, medium)
    _write_csv(summary_path, summary)
    _write_csv(frame_path, frame_rows)
    _write_csv(instance_path, instance_rows)
    decision_path.write_text(decision, encoding="utf8")
    metadata = {
        "purpose": "V8_O2.7_fixed_camera_K_and_SAM_instance_affine_depth",
        "config": _jsonable(config),
        "cache": str(path),
        "frames": list(frames),
        "correspondence": "GT-world pseudo-match",
        "history_pose": "GT diagnostic history",
        "deployable_components_only": [
            "current_K",
            "reference_K",
            "causal_median_K",
        ],
        "oracle": ["GT_K", "global_affine_depth", "instance_affine_depth"],
        "outputs": {
            "decision_csv": str(compact_path),
            "medium_csv": str(medium_path),
            "fold_summary": str(summary_path),
            "frame_diagnostics": str(frame_path),
            "instance_affine": str(instance_path),
            "decision_md": str(decision_path),
        },
    }
    (config.output_dir / "v80_o27_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf8",
    )
    print("V8 O2.7 COMPACT DECISION (COPY THIS CSV)")
    print(compact_path.read_text(encoding="utf8").rstrip())
    print("V8 O2.7 MEDIUM DIAGNOSIS (COPY THIS CSV)")
    print(medium_path.read_text(encoding="utf8").rstrip())
    print(decision.rstrip())
    return compact_path


def _intrinsics_modes(
    predicted: torch.Tensor,
    gt: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if (
        predicted.shape != gt.shape
        or predicted.ndim != 3
        or predicted.shape[-2:] != (3, 3)
    ):
        raise ValueError("O2.7 intrinsics must share shape [S,3,3].")
    reference = predicted[0:1].expand_as(predicted).clone()
    causal = torch.stack(
        [
            predicted[: index + 1].median(dim=0).values
            for index in range(predicted.shape[0])
        ]
    )
    return {
        "current_predicted": predicted,
        "reference_predicted": reference,
        "causal_median_predicted": causal,
        "gt": gt,
    }


def _calibrate_instance_affine(
    predicted: torch.Tensor,
    target: torch.Tensor,
    masks: torch.Tensor,
    *,
    native_to_metric: float,
    config: CalibrationConfig,
) -> InstanceAffineCalibration:
    sequence, instances = masks.shape[:2]
    shape = (sequence, instances)
    scale = torch.ones(shape, dtype=torch.double)
    offset = torch.zeros(shape, dtype=torch.double)
    fit_valid = torch.zeros(shape, dtype=torch.bool)
    pixels = torch.zeros(shape, dtype=torch.long)
    before = torch.full(shape, float("nan"), dtype=torch.double)
    after = torch.full(shape, float("nan"), dtype=torch.double)
    p90 = torch.full(shape, float("nan"), dtype=torch.double)
    for frame in range(sequence):
        x_all = predicted[frame, ..., 0].reshape(-1).double()
        y_all = target[frame, ..., 0].reshape(-1).double()
        finite = (
            torch.isfinite(x_all)
            & torch.isfinite(y_all)
            & x_all.gt(1e-6)
            & y_all.gt(1e-6)
        )
        for slot in range(instances):
            selected = finite & masks[frame, slot].reshape(-1)
            indices = selected.nonzero(as_tuple=False)[:, 0]
            if indices.numel() > config.max_pixels:
                positions = (
                    torch.linspace(0, indices.numel() - 1, config.max_pixels)
                    .round()
                    .long()
                )
                indices = indices.index_select(0, positions)
            pixels[frame, slot] = indices.numel()
            if indices.numel() < config.min_pixels:
                continue
            x = x_all.index_select(0, indices)
            y = y_all.index_select(0, indices)
            keep = torch.ones_like(x, dtype=torch.bool)
            a = x.new_tensor(1.0)
            b = x.new_tensor(0.0)
            for iteration in range(config.trim_iterations):
                a, b = _fit_affine(x[keep], y[keep])
                residual = (a * x + b - y).abs()
                if iteration + 1 == config.trim_iterations:
                    break
                cutoff = torch.quantile(residual[keep], config.trim_quantile)
                next_keep = residual.le(cutoff)
                if int(next_keep.sum()) < config.min_pixels or torch.equal(
                    next_keep, keep
                ):
                    break
                keep = next_keep
            calibrated = a * x + b
            residual_metric = (calibrated - y).abs() * native_to_metric
            scale[frame, slot] = a
            offset[frame, slot] = b
            fit_valid[frame, slot] = bool(
                torch.isfinite(a) & torch.isfinite(b) & a.gt(0.0)
            )
            before[frame, slot] = torch.sqrt((x - y).square().mean()) * native_to_metric
            after[frame, slot] = (
                torch.sqrt((calibrated - y).square().mean()) * native_to_metric
            )
            p90[frame, slot] = torch.quantile(residual_metric, 0.90)
    return InstanceAffineCalibration(
        scale=scale,
        offset=offset,
        valid=fit_valid,
        valid_pixels=pixels,
        before_rmse_metric=before,
        after_rmse_metric=after,
        after_p90_metric=p90,
    )


def _fit_affine(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    design = torch.stack([x, torch.ones_like(x)], dim=-1)
    solution = torch.linalg.lstsq(design, y[:, None]).solution[:, 0]
    return solution[0].clamp_min(1e-8), solution[1]


def _backproject_instance_affine(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    fit: InstanceAffineCalibration,
    local_features: torch.Tensor,
    local_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    sampled, valid = sample_dense_at_local_tokens(
        depth,
        local_features=local_features,
        local_valid=local_valid,
    )
    z = sampled[..., 0].double()
    z = fit.scale[..., None] * z + fit.offset[..., None]
    valid = valid & fit.valid[..., None] & torch.isfinite(z) & z.gt(1e-6)
    sequence, instances, points = z.shape
    height, width = depth.shape[1:3]
    uv = local_features[..., 3:5].double()
    u = ((uv[..., 0] + 1.0) * 0.5) * max(width - 1, 1)
    v = ((uv[..., 1] + 1.0) * 0.5) * max(height - 1, 1)
    fx = intrinsics[:, 0, 0].reshape(sequence, 1, 1)
    fy = intrinsics[:, 1, 1].reshape(sequence, 1, 1)
    cx = intrinsics[:, 0, 2].reshape(sequence, 1, 1)
    cy = intrinsics[:, 1, 2].reshape(sequence, 1, 1)
    focal_valid = torch.isfinite(fx) & torch.isfinite(fy) & fx.gt(1e-8) & fy.gt(1e-8)
    camera = torch.stack(
        [(u - cx) * z / fx, (v - cy) * z / fy, z],
        dim=-1,
    )
    valid = valid & focal_valid & torch.isfinite(camera).all(dim=-1)
    camera = torch.where(valid[..., None], camera, torch.zeros_like(camera))
    if camera.shape != (sequence, instances, points, 3):
        raise RuntimeError("O2.7 instance camera point shape mismatch.")
    return camera, valid


def _annotate_o27_row(
    row: dict[str, object],
    *,
    sequence_index: int,
    selected_k: torch.Tensor,
    gt_k: torch.Tensor,
    instance_fit: InstanceAffineCalibration,
) -> None:
    signed = _signed_intrinsics_error(selected_k, gt_k)
    slots = [int(value) for value in str(row["instance_slots"]).split() if value]
    valid_slots = [
        slot for slot in slots if bool(instance_fit.valid[sequence_index, slot])
    ]
    row["selected_fx_relative_error_signed"] = signed[0]
    row["selected_fy_relative_error_signed"] = signed[1]
    row["selected_cx_error_pixels_signed"] = signed[2]
    row["selected_cy_error_pixels_signed"] = signed[3]
    row["instance_affine_valid_slots"] = " ".join(str(slot) for slot in valid_slots)
    row["mean_instance_affine_a"] = _finite_mean(
        float(instance_fit.scale[sequence_index, slot]) for slot in valid_slots
    )
    row["mean_instance_affine_b_native"] = _finite_mean(
        float(instance_fit.offset[sequence_index, slot]) for slot in valid_slots
    )
    row["mean_instance_affine_rmse_metric"] = _finite_mean(
        float(instance_fit.after_rmse_metric[sequence_index, slot])
        for slot in valid_slots
    )
    row["mean_instance_affine_p90_metric"] = _finite_mean(
        float(instance_fit.after_p90_metric[sequence_index, slot])
        for slot in valid_slots
    )


def _signed_intrinsics_error(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> tuple[float, float, float, float]:
    return (
        float(predicted[0, 0] / target[0, 0].clamp_min(1e-12) - 1.0),
        float(predicted[1, 1] / target[1, 1].clamp_min(1e-12) - 1.0),
        float(predicted[0, 2] - target[0, 2]),
        float(predicted[1, 2] - target[1, 2]),
    )


def _medium_diagnosis(
    rows: Sequence[dict[str, object]],
    *,
    compact: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    branches = {
        "o27_raw_current_predicted_k",
        "o27_raw_reference_predicted_k",
        "o27_raw_causal_median_predicted_k",
        "o27_raw_gt_k",
        "o27_global_affine_gt_k",
        "o27_instance_affine_gt_k",
    }
    selected = {
        str(row["branch"]): str(row["selected_solver"])
        for row in compact
        if row["branch"] in branches
    }
    output = []
    for row in rows:
        if (
            row["fold"] != "medium"
            or row["branch"] not in branches
            or row["solver"] != selected.get(str(row["branch"]))
        ):
            continue
        raw_loss = float(row["raw_pose_loss"])
        reliable_loss = float(row["reliable_pose_loss"])
        output.append(
            {
                "frame_index": row["frame_index"],
                "branch": row["branch"],
                "depth_mode": row["depth_source"],
                "intrinsics_mode": row["intrinsics_source"],
                "solver": row["solver"],
                "history_frame_indices": row["history_frame_indices"],
                "instance_slots": row["instance_slots"],
                "correspondences": row["correspondences"],
                "effective_correspondences": row["effective_correspondences"],
                "fit_rmse_metric": row["fit_rmse_metric"],
                "reliable_accept": row["reliable_accept"],
                "reason": row["reason"],
                "raw_pose_loss": raw_loss,
                "proposed_pose_loss": row["proposed_pose_loss"],
                "reliable_pose_loss": reliable_loss,
                "frame_gain_percent": _gain(raw_loss, reliable_loss),
                "pose_worse": row["pose_worse"],
                "selected_fx_relative_error_signed": row[
                    "selected_fx_relative_error_signed"
                ],
                "selected_fy_relative_error_signed": row[
                    "selected_fy_relative_error_signed"
                ],
                "instance_affine_valid_slots": row["instance_affine_valid_slots"],
                "mean_instance_affine_a": row["mean_instance_affine_a"],
                "mean_instance_affine_b_native": row["mean_instance_affine_b_native"],
                "mean_instance_affine_rmse_metric": row[
                    "mean_instance_affine_rmse_metric"
                ],
            }
        )
    expected = len(branches) * len(FOLDS[1].test_frames)
    if len(output) != expected:
        raise RuntimeError(
            f"O2.7 medium diagnosis expected={expected}, got={len(output)}."
        )
    return output


def _instance_calibration_rows(
    frames: Sequence[int],
    fit: InstanceAffineCalibration,
    *,
    memory_write: torch.Tensor,
) -> list[dict[str, object]]:
    rows = []
    for sequence_index, frame in enumerate(frames):
        for slot in range(fit.scale.shape[1]):
            rows.append(
                {
                    "sequence_index": sequence_index,
                    "frame_index": int(frame),
                    "instance_slot": slot,
                    "memory_write": int(memory_write[sequence_index, slot]),
                    "fit_valid": int(fit.valid[sequence_index, slot]),
                    "valid_pixels": int(fit.valid_pixels[sequence_index, slot]),
                    "affine_a": float(fit.scale[sequence_index, slot]),
                    "affine_b_native": float(fit.offset[sequence_index, slot]),
                    "before_rmse_metric": float(
                        fit.before_rmse_metric[sequence_index, slot]
                    ),
                    "after_rmse_metric": float(
                        fit.after_rmse_metric[sequence_index, slot]
                    ),
                    "after_p90_metric": float(
                        fit.after_p90_metric[sequence_index, slot]
                    ),
                }
            )
    return rows


def _decision_markdown(
    compact: Sequence[dict[str, object]],
    medium: Sequence[dict[str, object]],
) -> str:
    by_branch = {str(row["branch"]): row for row in compact}

    def passed(branch: str) -> int:
        return int(by_branch[branch]["all_fold_pass"])

    def gain_435(branch: str) -> float:
        row = next(
            row
            for row in medium
            if row["frame_index"] == 435 and row["branch"] == branch
        )
        return float(row["frame_gain_percent"])

    raw_current = "o27_raw_current_predicted_k"
    raw_reference = "o27_raw_reference_predicted_k"
    raw_causal = "o27_raw_causal_median_predicted_k"
    raw_gt = "o27_raw_gt_k"
    global_gt = "o27_global_affine_gt_k"
    instance_gt = "o27_instance_affine_gt_k"
    lines = [
        "# V8 O2.7 fixed-K and instance-affine decision",
        "",
        "No model is trained. Correspondence and GT history remain diagnostic oracles.",
        "Reference/causal-median predicted K are deployable; GT K and affine depth are not.",
        "",
        f"- raw current predicted K all-fold pass: `{passed(raw_current)}`",
        f"- raw reference predicted K all-fold pass: `{passed(raw_reference)}`",
        f"- raw causal-median predicted K all-fold pass: `{passed(raw_causal)}`",
        f"- raw GT K all-fold pass: `{passed(raw_gt)}`",
        f"- global affine depth + GT K all-fold pass: `{passed(global_gt)}`",
        f"- instance affine depth + GT K all-fold pass: `{passed(instance_gt)}`",
        f"- frame 435 raw current-K gain: `{gain_435(raw_current):.8g}`",
        f"- frame 435 raw reference-K gain: `{gain_435(raw_reference):.8g}`",
        f"- frame 435 raw causal-median-K gain: `{gain_435(raw_causal):.8g}`",
        f"- frame 435 global-affine GT-K gain: `{gain_435(global_gt):.8g}`",
        f"- frame 435 instance-affine GT-K gain: `{gain_435(instance_gt):.8g}`",
        "",
        "Interpretation:",
        "",
    ]
    if passed(raw_reference) or passed(raw_causal):
        lines.append(
            "- A causal fixed-camera intrinsics policy repairs raw predicted geometry across folds."
        )
    elif gain_435(raw_reference) > gain_435(raw_current) or gain_435(
        raw_causal
    ) > gain_435(raw_current):
        lines.append(
            "- Intrinsics stabilization helps frame 435 but is not sufficient across folds."
        )
    else:
        lines.append(
            "- Intrinsics stabilization does not repair the critical frame; K drift is secondary there."
        )
    if gain_435(instance_gt) > 0.0 and gain_435(global_gt) <= 0.0:
        lines.append(
            "- SAM-instance affine depth repairs frame 435: global calibration was mixing incompatible regions."
        )
    elif gain_435(instance_gt) > gain_435(global_gt):
        lines.append(
            "- SAM-instance affine depth is a better upper bound than full-image affine depth."
        )
    else:
        lines.append(
            "- Instance-region affine does not beat global affine: local depth shape/support remains limiting."
        )
    lines.append(
        "- Begin SAM correspondence O3 only if a predicted-K policy or instance-depth upper bound is safe across folds."
    )
    lines.append("")
    return "\n".join(lines)


def _validate_masks(masks: torch.Tensor, depth: torch.Tensor) -> None:
    if masks.ndim != 4:
        raise ValueError("O2.7 tracking masks must be [S,K,H,W].")
    if masks.shape[0] != depth.shape[0] or masks.shape[2:] != depth.shape[1:3]:
        raise ValueError(
            "O2.7 tracking mask/depth grid mismatch: "
            f"masks={tuple(masks.shape)}, depth={tuple(depth.shape)}."
        )


def load_o27_config(path: str | Path) -> O27Config:
    source = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf8")) or {}
    base_path = Path(
        raw.get(
            "o26_config",
            "streaming_couping/configs/v80_geometry_factorization.yaml",
        )
    )
    if not base_path.is_absolute():
        base_path = (source.parents[2] / base_path).resolve()
    base = load_o26_config(base_path)
    output = Path(
        raw.get(
            "output_dir",
            "outputs/streaming_couping_v80_instance_geometry",
        )
    )
    if not output.is_absolute():
        output = (source.parents[2] / output).resolve()
    affine = raw.get("instance_affine", {})
    config = O27Config(
        source_path=source,
        base=base,
        output_dir=output,
        instance_affine=CalibrationConfig(
            trim_quantile=float(affine.get("trim_quantile", 0.90)),
            trim_iterations=int(affine.get("trim_iterations", 3)),
            min_pixels=int(affine.get("min_pixels", 64)),
            max_pixels=int(affine.get("max_pixels", 32768)),
        ),
    )
    _validate_config(config)
    return config


def _validate_config(config: O27Config) -> None:
    value = config.instance_affine
    if not 0.0 < value.trim_quantile <= 1.0:
        raise ValueError("O2.7 trim quantile must be in (0,1].")
    if value.trim_iterations < 1:
        raise ValueError("O2.7 trim iterations must be positive.")
    if value.min_pixels < 3 or value.max_pixels < value.min_pixels:
        raise ValueError("O2.7 affine pixel limits are invalid.")


def _finite_mean(values) -> float:
    rows = [value for value in values if math.isfinite(value)]
    return sum(rows) / len(rows) if rows else float("nan")


def _jsonable(config: O27Config) -> dict[str, Any]:
    value = asdict(config)
    value["source_path"] = str(config.source_path)
    value["output_dir"] = str(config.output_dir)
    value["base"]["source_path"] = str(config.base.source_path)
    value["base"]["output_dir"] = str(config.base.output_dir)
    for key in ("source_path", "data_config", "output_dir"):
        value["base"]["theory"][key] = str(value["base"]["theory"][key])
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="streaming_couping/configs/v80_instance_geometry.yaml",
    )
    parser.add_argument("--output-dir")
    return parser.parse_args()


if __name__ == "__main__":
    main()
