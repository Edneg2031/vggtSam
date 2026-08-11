#!/usr/bin/env python3
"""Run the V9.2 discrete-support density/collision factorization.

No matcher or pose model is trained.  Current queries are always the first 32
deterministic farthest-UV SAM3.1 tokens.  Only the number of actual historical
keys (32/64/128/256) and the deterministic assignment rule change.  GT is used
for visibility and continuous reprojection labels; pose is recovered by the
frozen V9 O-R1 epipolar solver.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import yaml

from streaming_couping.scripts.run_v90_local_token_matcher import (
    A1_EDGE_SELECTION_POLICY,
    StageAConfig,
    _pose_frame_rows,
    load_stage_a_config,
)
from streaming_couping.src.config import load_config
from streaming_couping.src.external_repos import maybe_add_repo_to_path
from streaming_couping.src.learned_pose.cache import cache_path, load_feature_cache
from streaming_couping.src.learned_pose.config import load_learned_pose_config
from streaming_couping.src.v74_temporal_protocol import (
    EXPECTED_FRAMES,
    FOLDS,
    validate_folds,
)
from streaming_couping.src.v80_pose_geometry import homogeneous
from streaming_couping.src.v90_epipolar_geometry import (
    LocalTokenReprojection,
    causal_mask_history_indices,
    local_token_reprojection_labels,
)
from streaming_couping.src.v92_support_factorization import (
    MATCH_STRATEGIES,
    SupportDiagnostics,
    match_discrete_support,
)


SUMMARY_COLUMNS = (
    "fold",
    "history_key_count",
    "strategy",
    "query_token_count",
    "test_frames",
    "pairs",
    "frames",
    "active_frames",
    "inactive_frames",
    "visible_queries",
    "coverage_at_8px",
    "coverage_at_12px",
    "coverage_at_16px",
    "accepted_correspondences",
    "mean_accepted_per_pair",
    "unique_history_keys",
    "unique_key_fraction",
    "nearest_collision_fraction",
    "pck_threshold_pixels",
    "visible_pck_accuracy",
    "visible_mean_epe_pixels",
    "selected_mean_epe_pixels",
    "mean_selected_current_hull_coverage",
    "mean_selected_history_hull_coverage",
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
    "fold_support_pass",
    "all_folds_support_pass",
)

FRAME_COLUMNS = (
    "fold",
    "history_key_count",
    "strategy",
    "sequence_index",
    "frame_index",
    "observed_slots",
    "history_edges",
    "solved_edges",
    "active",
    "selected_history_frame_indices",
    "visible_queries",
    "coverage_at_12px",
    "accepted_correspondences",
    "unique_history_keys",
    "nearest_collision_fraction",
    "visible_pck_accuracy",
    "visible_mean_epe_pixels",
    "selected_current_hull_coverage",
    "selected_history_hull_coverage",
    "raw_rotation_error_deg",
    "refined_rotation_error_deg",
    "raw_translation_direction_error_deg",
    "refined_translation_direction_error_deg",
    "relative_aggregate_worse",
)


@dataclass(frozen=True)
class SupportConfig:
    source_path: Path
    data_config: Path
    stage_a_config: Path
    output_dir: Path
    clip_name: str
    query_token_count: int
    history_key_counts: tuple[int, ...]
    strategies: tuple[str, ...]
    coverage_thresholds_pixels: tuple[float, ...]


@dataclass
class SupportData:
    frames: tuple[int, ...]
    image_size: tuple[int, int]
    masks: torch.Tensor
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
class SupportRecord:
    current: int
    history: int
    slot: int
    labels: LocalTokenReprojection


def main() -> None:
    args = _parse_args()
    config = load_support_config(args.config)
    if args.output_dir:
        config = replace(
            config, output_dir=Path(args.output_dir).expanduser().resolve()
        )
    result = run_support_factorization(config)
    print(f"V9.2 support factorization result={result}")


def run_support_factorization(config: SupportConfig) -> Path:
    stage_config = load_stage_a_config(config.stage_a_config)
    stage_config = replace(
        stage_config,
        data_config=config.data_config,
        output_dir=config.output_dir,
    )
    payload, cache_file = _load_support_payload(config)
    data = _prepare_support_data(payload, config=config, stage_config=stage_config)
    del payload
    positions = {frame: index for index, frame in enumerate(data.frames)}
    records_by_fold = {
        fold.name: _build_support_records(
            data,
            current_indices=[positions[value] for value in fold.test_frames],
            query_token_count=config.query_token_count,
            stage_config=stage_config,
        )
        for fold in FOLDS
    }

    summaries: list[dict[str, object]] = []
    selected_frames: list[dict[str, object]] = []
    for key_count in config.history_key_counts:
        for strategy in config.strategies:
            for fold in FOLDS:
                summary, frames = _evaluate_configuration(
                    fold_name=fold.name,
                    test_frames=fold.test_frames,
                    records=records_by_fold[fold.name],
                    history_key_count=key_count,
                    strategy=strategy,
                    data=data,
                    config=config,
                    stage_config=stage_config,
                )
                summaries.append(summary)
                selected_frames.extend(frames)
    _annotate_passes(summaries)
    _validate_outputs(summaries, selected_frames)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = config.output_dir / "v92_support_summary.csv"
    frames_path = config.output_dir / "v92_selected_frames.csv"
    decision_path = config.output_dir / "v92_support_decision.md"
    metadata_path = config.output_dir / "v92_support_metadata.json"
    _write_csv(summary_path, summaries, SUMMARY_COLUMNS)
    _write_csv(frames_path, selected_frames, FRAME_COLUMNS)
    decision_path.write_text(_decision_markdown(summaries), encoding="utf8")
    metadata = {
        "experiment": "V9.2 discrete support density/collision factorization",
        "config": _jsonable_config(config),
        "stage_a_solver_config": str(stage_config.source_path),
        "cache": {
            "path": str(cache_file),
            "size_bytes": cache_file.stat().st_size,
            "mtime_ns": cache_file.stat().st_mtime_ns,
            "sam_local_token_count": int(data.local_uv.shape[2]),
        },
        "trains_matcher": False,
        "trains_pose_model": False,
        "uses_pose_loss": False,
        "current_query_support": "first 32 deterministic farthest-UV tokens",
        "history_support": "actual cached farthest-UV prefix",
        "gt_usage": "visibility, continuous history reprojection labels, and scoring only",
        "pose_solver": "frozen V9 O-R1 calibrated epipolar solver",
        "edge_selection_policy": A1_EDGE_SELECTION_POLICY,
        "edge_selection_reads_gt_pose_error": False,
        "dynamic_instance_birth_retained": True,
        "outputs": {
            "summary": str(summary_path),
            "selected_frames": str(frames_path),
            "decision": str(decision_path),
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf8"
    )
    print(f"V9.2 wrote summary={summary_path}")
    print(f"V9.2 wrote decision={decision_path}")
    return summary_path


def _load_support_payload(
    config: SupportConfig,
) -> tuple[dict[str, Any], Path]:
    learned = load_learned_pose_config(config.data_config)
    clip = next((item for item in learned.clips if item.name == config.clip_name), None)
    if clip is None:
        raise ValueError(f"V9.2 clip={config.clip_name!r} is not configured.")
    path = cache_path(learned, clip)
    if not path.is_file():
        raise FileNotFoundError(
            "V9.2 requires its standalone 256-token observation cache. "
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
        "target_world_points",
        "target_depth",
        "target_world_to_camera",
        "baseline_pose_encoding",
        "target_pose_encoding",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"V9.2 cache lacks fields={sorted(missing)}.")
    if str(payload.get("sam_version", "")) != "sam3.1":
        raise ValueError("V9.2 requires cached SAM3.1 observations.")
    if str(payload.get("instance_source", "")) != "sam31_online":
        raise ValueError("V9.2 requires forward-only dynamic SAM3.1 slots.")
    if str(payload.get("sam_local_sampling", "")) != "farthest_uv":
        raise ValueError("V9.2 requires deterministic farthest-UV token sampling.")
    required_tokens = max(config.history_key_counts)
    if int(payload.get("sam_local_token_count", -1)) != required_tokens:
        raise ValueError(
            "V9.2 cache token count mismatch: "
            f"got {payload.get('sam_local_token_count')}, expected {required_tokens}."
        )
    return payload, path


def _prepare_support_data(
    payload: dict[str, Any],
    *,
    config: SupportConfig,
    stage_config: StageAConfig,
) -> SupportData:
    frames = tuple(int(value) for value in payload["frame_indices"])
    if frames != EXPECTED_FRAMES:
        raise ValueError(f"V9.2 requires frames 90:15:525, got {frames}.")
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
    feature_shape = tuple(payload["sam_local_features"].shape)
    if local_uv.ndim != 4 or local_uv.shape[-1] != 2:
        raise ValueError("V9.2 local UV must be [S,K,P,2].")
    if local_valid.shape != local_uv.shape[:-1]:
        raise ValueError("V9.2 local UV/valid shapes disagree.")
    if feature_shape[:3] != tuple(local_valid.shape):
        raise ValueError("V9.2 local feature/valid shapes disagree.")
    if int(local_uv.shape[2]) != max(config.history_key_counts):
        raise ValueError("V9.2 cache does not expose the locked 256-token support.")
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
        "local_uv": (sequence, slots, max(config.history_key_counts), 2),
        "world": (sequence, height, width, 3),
        "depth": (sequence, height, width),
        "global_w2c": (sequence, 4, 4),
        "intrinsics": (sequence, 3, 3),
        "baseline": (sequence, 4, 4),
        "target": (sequence, 4, 4),
    }
    actual = {
        "local_uv": tuple(local_uv.shape),
        "world": tuple(world.shape),
        "depth": tuple(depth.shape),
        "global_w2c": tuple(global_w2c.shape),
        "intrinsics": tuple(intrinsics.shape),
        "baseline": tuple(baseline.shape),
        "target": tuple(target.shape),
    }
    for name, shape in expected.items():
        if actual[name] != shape:
            raise ValueError(
                f"V9.2 {name} shape={actual[name]}, expected={shape}."
            )
    return SupportData(
        frames=frames,
        image_size=image_size,
        masks=masks,
        local_uv=local_uv,
        local_valid=local_valid,
        history_bank=causal_mask_history_indices(
            masks, max_history=stage_config.max_history
        ),
        world_points=world,
        depth=depth,
        global_w2c=global_w2c,
        intrinsics=intrinsics,
        baseline=baseline,
        target=target,
    )


def _build_support_records(
    data: SupportData,
    *,
    current_indices: Sequence[int],
    query_token_count: int,
    stage_config: StageAConfig,
) -> list[SupportRecord]:
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
                    local_uv_normalized=data.local_uv[
                        current, slot, :query_token_count
                    ],
                    local_valid=data.local_valid[
                        current, slot, :query_token_count
                    ],
                    masks=data.masks,
                    world_points_metric=data.world_points,
                    depth_metric=data.depth,
                    global_world_to_camera=data.global_w2c,
                    intrinsics=data.intrinsics,
                    config=stage_config.visibility,
                )
                records.append(
                    SupportRecord(
                        current=current,
                        history=history,
                        slot=slot,
                        labels=labels,
                    )
                )
    return records


def _evaluate_configuration(
    *,
    fold_name: str,
    test_frames: Sequence[int],
    records: Sequence[SupportRecord],
    history_key_count: int,
    strategy: str,
    data: SupportData,
    config: SupportConfig,
    stage_config: StageAConfig,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    predictions = {}
    match_stats = {}
    diagnostics: dict[int, SupportDiagnostics] = {}
    for record in records:
        result = match_discrete_support(
            record.labels,
            history_uv_normalized=data.local_uv[
                record.history, record.slot, :history_key_count
            ],
            history_valid=data.local_valid[
                record.history, record.slot, :history_key_count
            ],
            image_size=data.image_size,
            strategy=strategy,
            match_radius_pixels=stage_config.matcher.target_radius_pixels,
            pck_threshold_pixels=stage_config.matcher.pck_threshold_pixels,
        )
        predictions[id(record)] = result.correspondences
        diagnostics[id(record)] = result.diagnostics
        match_stats[id(record)] = result.diagnostics.as_pose_match_stats()

    pose_frames, edge_rows = _pose_frame_rows(
        stage="V9.2-O",
        fold_name=fold_name,
        architecture="gt_discrete_history_support",
        variant=f"k{history_key_count:03d}_{strategy}",
        edge_selection_policy=A1_EDGE_SELECTION_POLICY,
        test_frames=test_frames,
        records=records,
        predictions=predictions,
        match_stats=match_stats,
        data=data,
        config=stage_config,
    )
    totals = _sum_diagnostics(diagnostics.values())
    selected_edges = [row for row in edge_rows if int(row["selected_by_policy"])]
    raw_r = _finite_mean(
        float(row["raw_edge_rotation_error_deg"]) for row in pose_frames
    )
    refined_r = _finite_mean(
        float(row["refined_edge_rotation_error_deg"]) for row in pose_frames
    )
    raw_t = _finite_mean(
        float(row["raw_edge_translation_direction_error_deg"])
        for row in pose_frames
    )
    refined_t = _finite_mean(
        float(row["refined_edge_translation_direction_error_deg"])
        for row in pose_frames
    )
    raw_aggregate = raw_r + raw_t
    refined_aggregate = refined_r + refined_t
    summary = {
        "fold": fold_name,
        "history_key_count": history_key_count,
        "strategy": strategy,
        "query_token_count": config.query_token_count,
        "test_frames": " ".join(str(value) for value in test_frames),
        "pairs": len(records),
        "frames": len(pose_frames),
        "active_frames": sum(int(row["active"]) for row in pose_frames),
        "inactive_frames": sum(not int(row["active"]) for row in pose_frames),
        "visible_queries": totals.visible_queries,
        "coverage_at_8px": _ratio(totals.coverage_at_8px, totals.visible_queries),
        "coverage_at_12px": _ratio(
            totals.coverage_at_12px, totals.visible_queries
        ),
        "coverage_at_16px": _ratio(
            totals.coverage_at_16px, totals.visible_queries
        ),
        "accepted_correspondences": totals.accepted_correspondences,
        "mean_accepted_per_pair": _ratio(
            totals.accepted_correspondences, len(records)
        ),
        "unique_history_keys": totals.unique_history_keys,
        "unique_key_fraction": _ratio(
            totals.unique_history_keys, totals.accepted_correspondences
        ),
        "nearest_collision_fraction": _ratio(
            totals.nearest_collisions, totals.nearest_supported_assignments
        ),
        "pck_threshold_pixels": stage_config.matcher.pck_threshold_pixels,
        "visible_pck_accuracy": _ratio(
            totals.pck_correct, totals.visible_queries
        ),
        "visible_mean_epe_pixels": _ratio(
            totals.epe_sum_pixels, totals.visible_queries
        ),
        "selected_mean_epe_pixels": _ratio(
            totals.selected_epe_sum_pixels, totals.accepted_correspondences
        ),
        "mean_selected_current_hull_coverage": _finite_mean(
            float(row["current_uv_hull_coverage_fraction"])
            for row in selected_edges
        ),
        "mean_selected_history_hull_coverage": _finite_mean(
            float(row["history_uv_hull_coverage_fraction"])
            for row in selected_edges
        ),
        "raw_rotation_error_deg": raw_r,
        "refined_rotation_error_deg": refined_r,
        "rotation_gain_percent": _gain(raw_r, refined_r),
        "raw_translation_direction_error_deg": raw_t,
        "refined_translation_direction_error_deg": refined_t,
        "translation_direction_gain_percent": _gain(raw_t, refined_t),
        "raw_relative_aggregate_deg": raw_aggregate,
        "refined_relative_aggregate_deg": refined_aggregate,
        "relative_aggregate_gain_percent": _gain(
            raw_aggregate, refined_aggregate
        ),
        "relative_aggregate_worse_frames": sum(
            int(row["relative_aggregate_worse"]) for row in pose_frames
        ),
        "fold_support_pass": 0,
        "all_folds_support_pass": 0,
    }
    frame_output = _compact_frame_rows(
        fold_name=fold_name,
        history_key_count=history_key_count,
        strategy=strategy,
        pose_frames=pose_frames,
        edge_rows=edge_rows,
        records=records,
        diagnostics=diagnostics,
    )
    return summary, frame_output


def _compact_frame_rows(
    *,
    fold_name: str,
    history_key_count: int,
    strategy: str,
    pose_frames: Sequence[dict[str, object]],
    edge_rows: Sequence[dict[str, object]],
    records: Sequence[SupportRecord],
    diagnostics: dict[int, SupportDiagnostics],
) -> list[dict[str, object]]:
    output = []
    for pose in pose_frames:
        current = int(pose["sequence_index"])
        current_records = [record for record in records if record.current == current]
        total = _sum_diagnostics(diagnostics[id(record)] for record in current_records)
        selected = [
            row
            for row in edge_rows
            if int(row["sequence_index"]) == current
            and int(row["selected_by_policy"])
        ]
        output.append(
            {
                "fold": fold_name,
                "history_key_count": history_key_count,
                "strategy": strategy,
                "sequence_index": current,
                "frame_index": int(pose["frame_index"]),
                "observed_slots": " ".join(
                    str(value) for value in sorted({r.slot for r in current_records})
                ),
                "history_edges": int(pose["history_edges"]),
                "solved_edges": int(pose["solved_edges"]),
                "active": int(pose["active"]),
                "selected_history_frame_indices": str(
                    pose["selected_history_frame_indices"]
                ),
                "visible_queries": total.visible_queries,
                "coverage_at_12px": _ratio(
                    total.coverage_at_12px, total.visible_queries
                ),
                "accepted_correspondences": total.accepted_correspondences,
                "unique_history_keys": total.unique_history_keys,
                "nearest_collision_fraction": _ratio(
                    total.nearest_collisions,
                    total.nearest_supported_assignments,
                ),
                "visible_pck_accuracy": _ratio(
                    total.pck_correct, total.visible_queries
                ),
                "visible_mean_epe_pixels": _ratio(
                    total.epe_sum_pixels, total.visible_queries
                ),
                "selected_current_hull_coverage": _finite_mean(
                    float(row["current_uv_hull_coverage_fraction"])
                    for row in selected
                ),
                "selected_history_hull_coverage": _finite_mean(
                    float(row["history_uv_hull_coverage_fraction"])
                    for row in selected
                ),
                "raw_rotation_error_deg": float(
                    pose["raw_edge_rotation_error_deg"]
                ),
                "refined_rotation_error_deg": float(
                    pose["refined_edge_rotation_error_deg"]
                ),
                "raw_translation_direction_error_deg": float(
                    pose["raw_edge_translation_direction_error_deg"]
                ),
                "refined_translation_direction_error_deg": float(
                    pose["refined_edge_translation_direction_error_deg"]
                ),
                "relative_aggregate_worse": int(
                    pose["relative_aggregate_worse"]
                ),
            }
        )
    return output


def _sum_diagnostics(rows: Iterable[SupportDiagnostics]) -> SupportDiagnostics:
    values = list(rows)
    return SupportDiagnostics(
        **{
            name: sum(getattr(row, name) for row in values)
            for name in SupportDiagnostics.__dataclass_fields__
        }
    )


def _annotate_passes(rows: list[dict[str, object]]) -> None:
    for row in rows:
        row["fold_support_pass"] = int(_strict_pass(row))
    for key_count in sorted({int(row["history_key_count"]) for row in rows}):
        for strategy in MATCH_STRATEGIES:
            group = [
                row
                for row in rows
                if int(row["history_key_count"]) == key_count
                and row["strategy"] == strategy
            ]
            if len(group) != len(FOLDS):
                raise ValueError(
                    f"V9.2 k={key_count} strategy={strategy} lacks folds."
                )
            passed = int(all(int(row["fold_support_pass"]) for row in group))
            for row in group:
                row["all_folds_support_pass"] = passed


def _strict_pass(row: dict[str, object]) -> bool:
    return bool(
        int(row["active_frames"]) == int(row["frames"])
        and float(row["refined_rotation_error_deg"])
        < float(row["raw_rotation_error_deg"])
        and float(row["refined_translation_direction_error_deg"])
        < float(row["raw_translation_direction_error_deg"])
        and int(row["relative_aggregate_worse_frames"]) == 0
    )


def _decision_markdown(rows: Sequence[dict[str, object]]) -> str:
    combinations = []
    for key_count in sorted({int(row["history_key_count"]) for row in rows}):
        for strategy in MATCH_STRATEGIES:
            group = [
                row
                for row in rows
                if int(row["history_key_count"]) == key_count
                and row["strategy"] == strategy
            ]
            combinations.append(
                {
                    "key_count": key_count,
                    "strategy": strategy,
                    "coverage": _finite_mean(
                        float(row["coverage_at_12px"]) for row in group
                    ),
                    "pck": _finite_mean(
                        float(row["visible_pck_accuracy"]) for row in group
                    ),
                    "collision": _finite_mean(
                        float(row["nearest_collision_fraction"]) for row in group
                    ),
                    "r_gain": _finite_mean(
                        float(row["rotation_gain_percent"]) for row in group
                    ),
                    "t_gain": _finite_mean(
                        float(row["translation_direction_gain_percent"])
                        for row in group
                    ),
                    "worse": sum(
                        int(row["relative_aggregate_worse_frames"])
                        for row in group
                    ),
                    "pass": int(
                        len(group) == len(FOLDS)
                        and all(int(row["all_folds_support_pass"]) for row in group)
                    ),
                }
            )
    passing = [row for row in combinations if int(row["pass"])]
    density_rescue = int(any(int(row["key_count"]) > 32 for row in passing))
    uniqueness_rescue = int(
        any(
            row["strategy"] in {"mutual", "greedy_unique"}
            and int(row["pass"])
            and not any(
                int(other["key_count"]) == int(row["key_count"])
                and other["strategy"] == "nearest"
                and int(other["pass"])
                for other in combinations
            )
            for row in combinations
        )
    )
    lines = [
        "# V9.2 discrete-support factorization decision",
        "",
        "No matcher or pose model is trained. Current support is fixed to the first 32 SAM3.1 farthest-UV tokens.",
        "GT provides visibility/continuous reprojection labels only; every predicted history point is an actual cached key.",
        "Pose always uses the frozen O-R1 solver and causal best-design-condition edge selection.",
        "",
        f"- all-fold passing configurations: `{len(passing)}`",
        f"- denser history support rescues local32: `{density_rescue}`",
        f"- one-to-one assignment rescues nearest matching: `{uniqueness_rescue}`",
        "",
        "| history keys | strategy | coverage@12 | PCK@8 | collision | mean R gain | mean t-dir gain | worse | all-fold pass |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in combinations:
        lines.append(
            f"| {row['key_count']} | {row['strategy']} "
            f"| {float(row['coverage']):.6g} | {float(row['pck']):.6g} "
            f"| {float(row['collision']):.6g} | {float(row['r_gain']):.6g} "
            f"| {float(row['t_gain']):.6g} | {row['worse']} | {row['pass']} |"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- A denser nearest row passing means history sampling density, not the learned matcher, was the local32 bottleneck.",
            "- A mutual/greedy-unique row passing while nearest fails means duplicate-key collisions were the bottleneck.",
            "- Coverage/PCK improve but pose still fails: spatial conditioning or the epipolar solver remains the bottleneck.",
            "- No 256-key strategy passes: do not train another detector-FPN matcher; fixed query support or descriptor choice must change next.",
            "- This remains one-scene temporal extrapolation and supports only relative rotation/translation-direction claims.",
            "",
        ]
    )
    return "\n".join(lines)


def load_support_config(path: str | Path) -> SupportConfig:
    source = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf8")) or {}
    repository = source.parents[2]
    config = SupportConfig(
        source_path=source,
        data_config=_resolve_path(
            repository,
            raw.get(
                "data_config",
                "streaming_couping/configs/v92_support_data.yaml",
            ),
        ),
        stage_a_config=_resolve_path(
            repository,
            raw.get(
                "stage_a_config",
                "streaming_couping/configs/v90_local_token_matcher.yaml",
            ),
        ),
        output_dir=_resolve_path(
            repository,
            raw.get(
                "output_dir",
                "outputs/streaming_couping_v92_support_factorization",
            ),
        ),
        clip_name=str(raw.get("clip_name", "00a231a370_90_525_step15_37_68_54")),
        query_token_count=int(raw.get("query_token_count", 32)),
        history_key_counts=tuple(
            int(value) for value in raw.get("history_key_counts", (32, 64, 128, 256))
        ),
        strategies=tuple(str(value) for value in raw.get("strategies", MATCH_STRATEGIES)),
        coverage_thresholds_pixels=tuple(
            float(value)
            for value in raw.get("coverage_thresholds_pixels", (8.0, 12.0, 16.0))
        ),
    )
    if config.query_token_count != 32:
        raise ValueError("V9.2 locks current query support to 32 tokens.")
    if config.history_key_counts != (32, 64, 128, 256):
        raise ValueError("V9.2 locks history keys to 32/64/128/256.")
    if config.strategies != MATCH_STRATEGIES:
        raise ValueError("V9.2 locks nearest/mutual/greedy_unique order.")
    if config.coverage_thresholds_pixels != (8.0, 12.0, 16.0):
        raise ValueError("V9.2 locks coverage thresholds to 8/12/16 pixels.")
    return config


def _validate_outputs(
    summaries: Sequence[dict[str, object]],
    frames: Sequence[dict[str, object]],
) -> None:
    expected_summary = len(FOLDS) * 4 * len(MATCH_STRATEGIES)
    expected_frames = expected_summary * len(FOLDS[0].test_frames)
    if len(summaries) != expected_summary:
        raise ValueError(
            f"V9.2 summary rows={len(summaries)}, expected={expected_summary}."
        )
    if len(frames) != expected_frames:
        raise ValueError(f"V9.2 frame rows={len(frames)}, expected={expected_frames}.")
    summary_keys = {
        (row["fold"], row["history_key_count"], row["strategy"])
        for row in summaries
    }
    if len(summary_keys) != expected_summary:
        raise ValueError("V9.2 summary has duplicate experiment keys.")


def _write_csv(
    path: Path,
    rows: Sequence[dict[str, object]],
    columns: Sequence[str],
) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty V9.2 CSV: {path.name}")
    expected = set(columns)
    for index, row in enumerate(rows):
        if set(row) != expected:
            raise ValueError(
                f"V9.2 CSV {path.name} row={index} mismatch: "
                f"missing={sorted(expected - set(row))}, "
                f"extra={sorted(set(row) - expected)}"
            )
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _ratio(numerator: float, denominator: float) -> float:
    return 0.0 if float(denominator) <= 0.0 else float(numerator) / float(denominator)


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


def _jsonable_config(config: SupportConfig) -> dict[str, object]:
    value = asdict(config)
    for name in ("source_path", "data_config", "stage_a_config", "output_dir"):
        value[name] = str(value[name])
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="streaming_couping/configs/v92_support_factorization.yaml",
    )
    parser.add_argument("--output-dir")
    return parser.parse_args()


if __name__ == "__main__":
    main()
