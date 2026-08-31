#!/usr/bin/env python3
"""Run the frozen V0 A--E temporal-prompt projection diagnostic.

This experiment is deliberately one step before a real SAM re-segmentation
ablation.  It reuses a completed V0 cache, projects historical object geometry
with the *raw* StreamVGGT pose, and freezes all candidate points/boxes before
opening ground-truth masks.  No SAM call, model update, pointmap edit, or
threshold selected from test annotations occurs here.

The five prompt families are:

``A_center``
    One projected 3-D center from the most recent earlier observation.
``B_surface_3`` / ``C_surface_5``
    Three/five front-depth points selected with deterministic 2-D farthest
    point sampling.
``D_bbox``
    A robust quantile box around the projected historical point cloud.
``E_depth_gate``
    The C points retained by a fixed current-depth consistency gate.  Several
    predeclared relative tolerances are reported as a curve.

The output intentionally distinguishes point precision from object-frame
coverage.  A branch that keeps one very safe point is not allowed to look good
merely because its conditional hit rate is high.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import torch

from streaming_couping.scripts.run_semantic_map_evaluation import (
    _load_run as _load_evaluation_run,
)
from streaming_couping.scripts.run_v0_bidirectional_feedback import (
    V0Arrays,
    _find_clip,
    _load_arrays,
    _resize_masks,
    _validate_v0_inputs,
)
from streaming_couping.src.learned_pose.baseline_runtime import (
    load_baseline_run_config,
)
from streaming_couping.src.learned_pose.cache import cache_path, load_feature_cache
from streaming_couping.src.learned_pose.config import load_learned_pose_config
from streaming_couping.src.semantic_map_metrics import (
    load_ground_truth_stream_masks,
)
from streaming_couping.src.semantic_tracking_metrics import (
    GroundTruthInstances,
    evaluate_tracking_variants,
    load_ground_truth_instances,
)
from streaming_couping.src.storage import expand_storage_path
from streaming_couping.src.temporal_prompt_matrix import (
    depth_consistency_gate,
    harmonic_mean,
    project_world_points,
    projected_bbox,
    sample_depth_nearest,
    select_surface_indices,
    spatial_dispersion,
)


REVISION = "v0_temporal_prompt_matrix_frozen_diagnostic_r1"
DEFAULT_CONFIG = "streaming_couping/configs/v0_baseline.yaml"
DEFAULT_OUTPUT = (
    "${VGGT_SAM_STORAGE_ROOT}/outputs/"
    "streaming_couping_v0_temporal_prompt_matrix"
)

POINT_BRANCHES = ("A_center", "B_surface_3", "C_surface_5")
BOX_BRANCH = "D_bbox"
DEPTH_BRANCH = "E_depth_gate"
DEFAULT_DEPTH_RELATIVE_TOLERANCES = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50)


@dataclass(frozen=True)
class CandidateBundle:
    """All causal candidates and event rows produced before GT is opened."""

    point_rows: list[dict[str, object]]
    box_rows: list[dict[str, object]]
    query_rows: list[dict[str, object]]
    metadata: dict[str, object]


def main() -> None:
    args = _parse_args()
    _validate_args(args)

    config_path = Path(args.config).expanduser().resolve()
    data = load_learned_pose_config(config_path)
    baseline = load_baseline_run_config(config_path)
    if baseline.version != "v0":
        raise ValueError("This diagnostic requires baseline.version=v0.")
    if len(data.clips) != 1:
        raise ValueError(
            "The temporal prompt matrix is intentionally sealed to one V0 clip."
        )
    clip = _find_clip(data, baseline.clip_name)
    if clip.name != "00a231a370_90_525_step15_37_68_54":
        raise ValueError(
            "This diagnostic is sealed to the retained V0 baseline clip "
            "00a231a370_90_525_step15_37_68_54."
        )

    output_dir = expand_storage_path(
        args.output_dir or DEFAULT_OUTPUT,
        base=config_path.parent,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_path_value = cache_path(data, clip)
    if not cache_path_value.is_file():
        raise FileNotFoundError(
            f"Missing frozen V0 cache {cache_path_value}; "
            "run commands_v0_baseline.txt first."
        )
    payload = load_feature_cache(cache_path_value)
    baseline_summary, poses = _validate_v0_inputs(
        payload=payload,
        summary_path=baseline.output_dir / "baseline_summary.json",
        poses_path=baseline.output_dir / "poses.pt",
        clip=clip,
    )
    arrays = _load_arrays(payload, poses)
    frames = tuple(int(value) for value in payload["frame_indices"])
    track_ids = tuple(int(value) for value in payload["sam_track_ids"])
    track_prompts = tuple(str(value) for value in payload["sam_track_prompts"])

    print("V0 TEMPORAL PROMPT A-E MATRIX DIAGNOSTIC")
    print(f"config={config_path}")
    print(f"cache={cache_path_value}")
    print(f"output={output_dir}")
    print(
        "scope=one sealed V0 scene; frozen SAM/StreamVGGT cache is reused; "
        "no SAM rerun"
    )
    print(
        "candidate_generation=earlier-frame geometry only; raw StreamVGGT "
        "pose; GT is opened after candidates are frozen"
    )
    print(
        f"frames={len(frames)} slots={arrays.raw_masks.shape[1]} "
        f"map_size={arrays.map_size} image_size={arrays.image_size} "
        f"depth_size={tuple(arrays.depth.shape[-2:])}"
    )
    print(
        "branches=A_center,B_surface_3,C_surface_5,D_bbox,E_depth_gate "
        f"depth_relative_tolerances={_format_float_list(args.depth_relative_tolerances)}"
    )

    # ------------------------------
    # Causal candidate-generation phase.  Nothing below this marker reads GT.
    # ------------------------------
    with torch.no_grad():
        candidates = _build_candidates(
            arrays=arrays,
            frames=frames,
            track_ids=track_ids,
            raw_world_to_camera=arrays.raw_world_to_camera,
            max_history_points=int(args.max_history_points),
            confidence_threshold=float(args.confidence_threshold),
            track_score_threshold=float(args.track_score_threshold),
            front_fraction=float(args.front_fraction),
            box_quantile=float(args.box_quantile),
            min_box_points=int(args.min_box_points),
            absolute_depth_tolerance=float(args.absolute_depth_tolerance),
            depth_relative_tolerances=args.depth_relative_tolerances,
        )
    _write_csv(output_dir / "point_candidates_causal.csv", candidates.point_rows)
    _write_csv(output_dir / "box_candidates_causal.csv", candidates.box_rows)
    _write_csv(output_dir / "query_events_causal.csv", candidates.query_rows)
    (output_dir / "candidate_metadata.json").write_text(
        json.dumps(candidates.metadata, indent=2, sort_keys=True, allow_nan=True)
        + "\n",
        encoding="utf8",
    )
    print(
        f"candidates frozen queries={candidates.metadata['query_count']} "
        f"point_rows={len(candidates.point_rows)} "
        f"box_rows={len(candidates.box_rows)} "
        f"history_observations={candidates.metadata['history_observation_count']}"
    )

    # ------------------------------
    # Sealed evaluation phase.  GT is first opened here.
    # ------------------------------
    print("candidate artifacts frozen; opening sealed GT evaluation")
    evaluation_run = _load_evaluation_run(config_path)
    ground_truth = load_ground_truth_instances(
        data.manifest,
        scene_id=str(payload["scene_id"]),
        frame_indices=frames,
        output_size=arrays.map_size,
        prompts=tuple(str(value) for value in payload["instance_prompts"]),
        prompt_label_aliases=evaluation_run.prompt_label_aliases,
    )
    gt_map_masks = load_ground_truth_stream_masks(
        data.manifest,
        scene_id=str(payload["scene_id"]),
        frame_indices=frames,
        instance_ids=ground_truth.instance_ids,
        processed_size=arrays.map_size,
        image_mode=str(payload["image_mode"]),
    )
    gt_map = GroundTruthInstances(
        masks=gt_map_masks,
        instance_ids=ground_truth.instance_ids,
        labels=ground_truth.labels,
        all_visible_instance_ids=ground_truth.all_visible_instance_ids,
    )
    gt_projection_masks = load_ground_truth_stream_masks(
        data.manifest,
        scene_id=str(payload["scene_id"]),
        frame_indices=frames,
        instance_ids=ground_truth.instance_ids,
        processed_size=arrays.image_size,
        image_mode=str(payload["image_mode"]),
    )
    gt_projection = GroundTruthInstances(
        masks=gt_projection_masks,
        instance_ids=ground_truth.instance_ids,
        labels=ground_truth.labels,
        all_visible_instance_ids=ground_truth.all_visible_instance_ids,
    )

    tracking = evaluate_tracking_variants(
        scene_id=str(payload["scene_id"]),
        clip_name=str(payload["clip_name"]),
        frame_indices=frames,
        variant_masks={"raw_v0": arrays.raw_masks},
        variant_scores={"raw_v0": arrays.scores},
        raw_variant="raw_v0",
        track_ids=track_ids,
        track_prompts=track_prompts,
        ground_truth=gt_map,
        config=evaluation_run.tracking,
        prompt_label_aliases=evaluation_run.prompt_label_aliases,
    )
    _annotate_with_ground_truth(
        candidates.point_rows,
        candidates.box_rows,
        candidates.query_rows,
        ground_truth=gt_projection,
        assignments=tracking["assignments"],
    )

    point_summary = _summarize_point_branches(
        candidates.point_rows,
        candidates.query_rows,
        scene_id=str(payload["scene_id"]),
        clip_name=str(payload["clip_name"]),
    )
    per_object_summary = _summarize_per_object(
        candidates.point_rows,
        candidates.query_rows,
        scene_id=str(payload["scene_id"]),
        clip_name=str(payload["clip_name"]),
    )
    box_summary = _summarize_box_branches(
        candidates.box_rows,
        candidates.query_rows,
        scene_id=str(payload["scene_id"]),
        clip_name=str(payload["clip_name"]),
    )
    branch_summary = point_summary + box_summary
    depth_curve = [
        row
        for row in point_summary
        if str(row.get("branch")) == DEPTH_BRANCH
    ]

    _write_csv(output_dir / "point_prompt_metrics.csv", candidates.point_rows)
    _write_csv(output_dir / "box_prompt_metrics.csv", candidates.box_rows)
    _write_csv(output_dir / "query_metrics.csv", candidates.query_rows)
    _write_csv(output_dir / "branch_summary.csv", branch_summary)
    _write_csv(output_dir / "per_scene_metrics.csv", branch_summary)
    _write_csv(output_dir / "per_object_metrics.csv", per_object_summary)
    _write_csv(output_dir / "depth_gate_curve.csv", depth_curve)

    summary = _build_summary(
        payload=payload,
        baseline_summary=baseline_summary,
        cache_path_value=cache_path_value,
        output_dir=output_dir,
        arrays=arrays,
        candidates=candidates,
        branch_summary=branch_summary,
        per_object_summary=per_object_summary,
        tracking=tracking,
        args=args,
    )
    summary_path = output_dir / "summary.json"
    summary["outputs"] = {
        "summary": str(summary_path),
        "copyable_result": str(output_dir / "copyable_result.txt"),
        "candidate_metadata": str(output_dir / "candidate_metadata.json"),
        "point_candidates_causal": str(output_dir / "point_candidates_causal.csv"),
        "box_candidates_causal": str(output_dir / "box_candidates_causal.csv"),
        "query_events_causal": str(output_dir / "query_events_causal.csv"),
        "point_prompt_metrics": str(output_dir / "point_prompt_metrics.csv"),
        "box_prompt_metrics": str(output_dir / "box_prompt_metrics.csv"),
        "query_metrics": str(output_dir / "query_metrics.csv"),
        "branch_summary": str(output_dir / "branch_summary.csv"),
        "per_scene_metrics": str(output_dir / "per_scene_metrics.csv"),
        "per_object_metrics": str(output_dir / "per_object_metrics.csv"),
        "depth_gate_curve": str(output_dir / "depth_gate_curve.csv"),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=True, default=str)
        + "\n",
        encoding="utf8",
    )
    copyable_path = output_dir / "copyable_result.txt"
    _write_copyable(copyable_path, summary)
    _print_summary(summary, branch_summary, candidates.metadata)
    print(f"summary={summary_path}")
    print(f"copyable_result={copyable_path}")
    print("decision=DIAGNOSTIC_ONLY")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-history-points", type=int, default=4096)
    parser.add_argument("--confidence-threshold", type=float, default=0.30)
    parser.add_argument("--track-score-threshold", type=float, default=0.50)
    parser.add_argument("--front-fraction", type=float, default=0.35)
    parser.add_argument("--box-quantile", type=float, default=0.02)
    parser.add_argument("--min-box-points", type=int, default=8)
    parser.add_argument("--absolute-depth-tolerance", type=float, default=0.05)
    parser.add_argument(
        "--depth-relative-tolerances",
        type=_parse_float_list,
        default=DEFAULT_DEPTH_RELATIVE_TOLERANCES,
        help="Comma-separated fixed relative depth tolerances.",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if int(args.max_history_points) < 1:
        raise ValueError("--max-history-points must be positive.")
    for name in ("confidence_threshold", "track_score_threshold"):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0,1].")
    if not 0.0 < float(args.front_fraction) <= 1.0:
        raise ValueError("--front-fraction must be in (0,1].")
    if not 0.0 <= float(args.box_quantile) < 0.5:
        raise ValueError("--box-quantile must be in [0,0.5).")
    if int(args.min_box_points) < 1:
        raise ValueError("--min-box-points must be positive.")
    if float(args.absolute_depth_tolerance) < 0.0:
        raise ValueError("--absolute-depth-tolerance must be non-negative.")
    values = tuple(float(value) for value in args.depth_relative_tolerances)
    if not values or any(value < 0.0 for value in values):
        raise ValueError("Depth relative tolerances must be non-empty/non-negative.")
    args.depth_relative_tolerances = tuple(sorted(set(values)))


def _parse_float_list(value: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in str(value).split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Expected comma-separated floating-point values."
        ) from error
    if not values or any(not math.isfinite(item) for item in values):
        raise argparse.ArgumentTypeError("Depth tolerances must be finite.")
    return values


def _build_candidates(
    *,
    arrays: V0Arrays,
    frames: Sequence[int],
    track_ids: Sequence[int],
    raw_world_to_camera: torch.Tensor,
    max_history_points: int,
    confidence_threshold: float,
    track_score_threshold: float,
    front_fraction: float,
    box_quantile: float,
    min_box_points: int,
    absolute_depth_tolerance: float,
    depth_relative_tolerances: Sequence[float],
) -> CandidateBundle:
    """Generate all A--E candidates without consulting annotation fields."""

    sequence, slots, height, width = arrays.raw_masks.shape
    if len(track_ids) != slots:
        raise ValueError("track_ids do not match V0 mask slots.")
    if tuple(raw_world_to_camera.shape) != (sequence, 3, 4):
        raise ValueError("raw_world_to_camera must have shape [S,3,4].")

    source_clouds: list[list[torch.Tensor | None]] = [
        [None for _ in range(slots)] for _ in range(sequence)
    ]
    source_counts = torch.zeros(sequence, slots, dtype=torch.long)
    for frame in range(sequence):
        for slot in range(slots):
            support = (
                arrays.raw_masks[frame, slot]
                & (arrays.scores[frame, slot] >= float(track_score_threshold))
                & (arrays.confidence[frame] >= float(confidence_threshold))
                & torch.isfinite(arrays.points[frame]).all(dim=-1)
            )
            points = arrays.points[frame][support].float()
            source_counts[frame, slot] = int(points.shape[0])
            if points.shape[0] == 0:
                continue
            if points.shape[0] > int(max_history_points):
                # Evenly spaced selection is deterministic and avoids making
                # the memory depend on a random subsampling seed.
                selection = torch.linspace(
                    0,
                    points.shape[0] - 1,
                    int(max_history_points),
                ).round().long()
                points = points[selection]
            source_clouds[frame][slot] = points.contiguous()

    # ``history[frame,slot]`` is the last valid source strictly before frame.
    history = torch.full((sequence, slots), -1, dtype=torch.long)
    last_seen = [-1 for _ in range(slots)]
    for frame in range(sequence):
        history[frame] = torch.tensor(last_seen, dtype=torch.long)
        for slot in range(slots):
            if source_clouds[frame][slot] is not None:
                last_seen[slot] = frame

    raw_image_masks = _resize_masks(arrays.raw_masks, arrays.image_size)
    depth_size = tuple(int(value) for value in arrays.depth.shape[-2:])
    point_rows: list[dict[str, object]] = []
    box_rows: list[dict[str, object]] = []
    query_rows: list[dict[str, object]] = []

    query_count = 0
    for frame in range(sequence):
        for slot in range(slots):
            source_frame = int(history[frame, slot])
            if source_frame < 0:
                continue
            source = source_clouds[source_frame][slot]
            if source is None or source.numel() == 0:
                continue
            query_count += 1
            query_base = {
                "sequence_index": int(frame),
                "frame_index": int(frames[frame]),
                "slot": int(slot),
                "sam_track_id": int(track_ids[slot]),
                "history_sequence_index": int(source_frame),
                "history_frame_index": int(frames[source_frame]),
                "history_gap": int(frame - source_frame),
                "history_point_count": int(source.shape[0]),
            }

            projected = project_world_points(
                source,
                raw_world_to_camera[frame],
                arrays.intrinsics[frame],
                arrays.image_size,
            )

            # A: one median world-space center.
            center = torch.median(source, dim=0).values.reshape(1, 3)
            center_projection = project_world_points(
                center,
                raw_world_to_camera[frame],
                arrays.intrinsics[frame],
                arrays.image_size,
            )
            center_rows = _append_point_rows(
                point_rows,
                branch="A_center",
                tolerance_label="",
                query_base=query_base,
                uv=center_projection.uv,
                projected_depth=center_projection.depth,
                valid_mask=center_projection.valid_mask,
                source_indices=(-1,),
                raw_mask=raw_image_masks[frame, slot],
                gate_accepted=None,
                gate_absolute_residual=None,
                gate_relative_residual=None,
            )
            _append_query_row(
                query_rows,
                branch="A_center",
                tolerance_label="",
                query_base=query_base,
                point_rows=center_rows,
                image_size=arrays.image_size,
            )

            selected_by_count: dict[int, tuple[torch.Tensor, ...]] = {}
            for count, branch in ((3, "B_surface_3"), (5, "C_surface_5")):
                selected = select_surface_indices(
                    projected.uv,
                    projected.depth,
                    projected.valid_mask,
                    count,
                    front_fraction=front_fraction,
                )
                selected_by_count[count] = tuple(
                    int(value) for value in selected.tolist()
                )
                if selected.numel():
                    selected_uv = projected.uv[selected]
                    selected_depth = projected.depth[selected]
                    selected_valid = projected.valid_mask[selected]
                    source_indices = tuple(int(value) for value in selected.tolist())
                else:
                    selected_uv = torch.empty(0, 2, dtype=projected.uv.dtype)
                    selected_depth = torch.empty(0, dtype=projected.depth.dtype)
                    selected_valid = torch.empty(0, dtype=torch.bool)
                    source_indices = ()
                rows = _append_point_rows(
                    point_rows,
                    branch=branch,
                    tolerance_label="",
                    query_base=query_base,
                    uv=selected_uv,
                    projected_depth=selected_depth,
                    valid_mask=selected_valid,
                    source_indices=source_indices,
                    raw_mask=raw_image_masks[frame, slot],
                    gate_accepted=None,
                    gate_absolute_residual=None,
                    gate_relative_residual=None,
                )
                _append_query_row(
                    query_rows,
                    branch=branch,
                    tolerance_label="",
                    query_base=query_base,
                    point_rows=rows,
                    image_size=arrays.image_size,
                )

            # E: use exactly C's candidate locations, then apply only the
            # current predicted depth gate.  Every tolerance is predeclared
            # before evaluation so no test annotation can pick one.
            c_indices = selected_by_count[5]
            if c_indices:
                c_indices_tensor = torch.tensor(c_indices, dtype=torch.long)
                c_uv = projected.uv[c_indices_tensor]
                c_depth = projected.depth[c_indices_tensor]
                depth_uv = _rescale_uv(c_uv, arrays.image_size, depth_size)
                sampled_depth, depth_valid = sample_depth_nearest(
                    arrays.depth[frame], depth_uv
                )
            else:
                c_indices_tensor = torch.empty(0, dtype=torch.long)
                c_uv = torch.empty(0, 2, dtype=projected.uv.dtype)
                c_depth = torch.empty(0, dtype=projected.depth.dtype)
                sampled_depth = torch.empty(0, dtype=projected.depth.dtype)
                depth_valid = torch.empty(0, dtype=torch.bool)

            for relative_tolerance in depth_relative_tolerances:
                accepted_gate, absolute_residual, relative_residual = (
                    depth_consistency_gate(
                        c_depth,
                        sampled_depth,
                        depth_valid,
                        absolute_tolerance=absolute_depth_tolerance,
                        relative_tolerance=float(relative_tolerance),
                    )
                )
                # The depth helper does not know whether the original
                # projection was in bounds.  Keep that validity separate and
                # let the row helper combine it into accepted_prompt.
                e_rows = _append_point_rows(
                    point_rows,
                    branch=DEPTH_BRANCH,
                    tolerance_label=_tolerance_label(relative_tolerance),
                    query_base=query_base,
                    uv=c_uv,
                    projected_depth=c_depth,
                    valid_mask=projected.valid_mask[c_indices_tensor]
                    if c_indices
                    else torch.empty(0, dtype=torch.bool),
                    source_indices=c_indices,
                    raw_mask=raw_image_masks[frame, slot],
                    gate_accepted=accepted_gate,
                    gate_absolute_residual=absolute_residual,
                    gate_relative_residual=relative_residual,
                )
                _append_query_row(
                    query_rows,
                    branch=DEPTH_BRANCH,
                    tolerance_label=_tolerance_label(relative_tolerance),
                    query_base=query_base,
                    point_rows=e_rows,
                    image_size=arrays.image_size,
                )

            # D: a box is an independent prompt type.  It is not converted to
            # a point precision number; evaluation reports purity/recall/IoU.
            box = projected_bbox(
                projected.uv,
                projected.valid_mask,
                arrays.image_size,
                quantile=box_quantile,
                min_points=min_box_points,
            )
            box_row = _make_box_row(
                branch=BOX_BRANCH,
                tolerance_label="",
                query_base=query_base,
                box=box,
                projected_valid_count=int(projected.valid_mask.sum()),
                image_size=arrays.image_size,
            )
            box_rows.append(box_row)
            _append_query_row(
                query_rows,
                branch=BOX_BRANCH,
                tolerance_label="",
                query_base=query_base,
                point_rows=(),
                box_row=box_row,
                image_size=arrays.image_size,
            )

    valid_source_counts = source_counts[source_counts > 0].float()
    metadata = {
        "revision": REVISION,
        "candidate_generation_gt_fields": 0,
        "sam_called": 0,
        "models_rerun": 0,
        "pose_source": "raw_streamvggt",
        "sequence_count": int(sequence),
        "slot_count": int(slots),
        "query_count": int(query_count),
        "history_observation_count": int(sum(
            source is not None
            for frame_sources in source_clouds
            for source in frame_sources
        )),
        "source_point_count_median": (
            float(valid_source_counts.median())
            if valid_source_counts.numel()
            else 0.0
        ),
        "max_history_points": int(max_history_points),
        "front_fraction": float(front_fraction),
        "box_quantile": float(box_quantile),
        "min_box_points": int(min_box_points),
        "absolute_depth_tolerance": float(absolute_depth_tolerance),
        "depth_relative_tolerances": [
            float(value) for value in depth_relative_tolerances
        ],
    }
    return CandidateBundle(
        point_rows=point_rows,
        box_rows=box_rows,
        query_rows=query_rows,
        metadata=metadata,
    )


def _append_point_rows(
    output: list[dict[str, object]],
    *,
    branch: str,
    tolerance_label: str,
    query_base: Mapping[str, object],
    uv: torch.Tensor,
    projected_depth: torch.Tensor,
    valid_mask: torch.Tensor,
    source_indices: Sequence[int],
    raw_mask: torch.Tensor,
    gate_accepted: torch.Tensor | None,
    gate_absolute_residual: torch.Tensor | None,
    gate_relative_residual: torch.Tensor | None,
) -> list[dict[str, object]]:
    uv = uv.detach().float().cpu()
    projected_depth = projected_depth.detach().float().cpu().reshape(-1)
    valid_mask = valid_mask.detach().cpu().bool().reshape(-1)
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError("Projected prompt UV must have shape [N,2].")
    if len(source_indices) != uv.shape[0]:
        raise ValueError("source_indices do not match projected prompt count.")
    if projected_depth.shape[0] != uv.shape[0] or valid_mask.shape[0] != uv.shape[0]:
        raise ValueError("Prompt projection vectors have inconsistent lengths.")
    gate = (
        torch.ones(uv.shape[0], dtype=torch.bool)
        if gate_accepted is None
        else gate_accepted.detach().cpu().bool().reshape(-1)
    )
    abs_residual = (
        torch.full((uv.shape[0],), float("nan"))
        if gate_absolute_residual is None
        else gate_absolute_residual.detach().float().cpu().reshape(-1)
    )
    rel_residual = (
        torch.full((uv.shape[0],), float("nan"))
        if gate_relative_residual is None
        else gate_relative_residual.detach().float().cpu().reshape(-1)
    )
    if gate.shape[0] != uv.shape[0] or abs_residual.shape[0] != uv.shape[0]:
        raise ValueError("Depth gate vectors do not match prompt count.")
    if rel_residual.shape[0] != uv.shape[0]:
        raise ValueError("Depth relative residuals do not match prompt count.")

    rows: list[dict[str, object]] = []
    for index in range(uv.shape[0]):
        point = uv[index]
        in_bounds = bool(valid_mask[index])
        raw_hit = int(_sample_mask(raw_mask, point)) if in_bounds else 0
        accepted = int(in_bounds and bool(gate[index]))
        row = dict(query_base)
        row.update(
            {
                "branch": str(branch),
                "tolerance": str(tolerance_label),
                "candidate_index": int(index),
                "source_point_index": int(source_indices[index]),
                "projected_u": _finite_float(point[0]),
                "projected_v": _finite_float(point[1]),
                "projected_depth": _finite_float(projected_depth[index]),
                "in_bounds": int(in_bounds),
                "depth_gate_accepted": int(bool(gate[index])),
                "accepted_prompt": int(accepted),
                "depth_absolute_residual": _finite_float(abs_residual[index]),
                "depth_relative_residual": _finite_float(rel_residual[index]),
                "current_raw_mask_hit": int(raw_hit),
                "gt_assignment_available": -1,
                "gt_visible": -1,
                "gt_mask_hit": -1,
                "gt_instance_id": -1,
            }
        )
        output.append(row)
        rows.append(row)
    return rows


def _append_query_row(
    output: list[dict[str, object]],
    *,
    branch: str,
    tolerance_label: str,
    query_base: Mapping[str, object],
    point_rows: Sequence[Mapping[str, object]],
    image_size: tuple[int, int],
    box_row: Mapping[str, object] | None = None,
) -> None:
    accepted = [row for row in point_rows if int(row["accepted_prompt"]) == 1]
    accepted_coordinates = [
        [float(row["projected_u"]), float(row["projected_v"])]
        for row in accepted
        if math.isfinite(float(row["projected_u"]))
        and math.isfinite(float(row["projected_v"]))
    ]
    accepted_uv = (
        torch.tensor(accepted_coordinates, dtype=torch.float32)
        if accepted_coordinates
        else torch.empty(0, 2, dtype=torch.float32)
    )
    # The query row is causal; GT columns are filled only in the evaluation
    # phase.  Keeping one row even for an empty candidate set makes coverage
    # denominators auditable.
    dispersion = spatial_dispersion(accepted_uv, image_size)
    row = dict(query_base)
    row.update(
        {
            "branch": str(branch),
            "tolerance": str(tolerance_label),
            "candidate_count": int(len(point_rows)),
            "in_bounds_count": sum(int(r["in_bounds"]) for r in point_rows),
            "accepted_count": int(len(accepted)),
            "raw_mask_hit_count": sum(
                int(r["current_raw_mask_hit"]) for r in point_rows
            ),
            "accepted_mean_pairwise_px": float(dispersion.mean_pairwise_px),
            "accepted_min_pairwise_px": float(dispersion.min_pairwise_px),
            "accepted_normalized_dispersion": float(
                dispersion.normalized_mean_pairwise
            ),
            "box_available": int(
                bool(box_row is not None and int(box_row["box_available"]) == 1)
            ),
            "gt_assignment_available": -1,
            "gt_visible": -1,
            "gt_correct_count": -1,
            "gt_area_pixels": -1,
        }
    )
    output.append(row)


def _make_box_row(
    *,
    branch: str,
    tolerance_label: str,
    query_base: Mapping[str, object],
    box: torch.Tensor | None,
    projected_valid_count: int,
    image_size: tuple[int, int],
) -> dict[str, object]:
    row = dict(query_base)
    row.update(
        {
            "branch": str(branch),
            "tolerance": str(tolerance_label),
            "projected_valid_count": int(projected_valid_count),
            "box_available": int(box is not None),
            "box_x0": float(box[0]) if box is not None else float("nan"),
            "box_y0": float(box[1]) if box is not None else float("nan"),
            "box_x1": float(box[2]) if box is not None else float("nan"),
            "box_y1": float(box[3]) if box is not None else float("nan"),
            "box_area_pixels": (
                _box_area_pixels(box, image_size) if box is not None else 0
            ),
            "gt_assignment_available": -1,
            "gt_visible": -1,
            "gt_instance_id": -1,
            "gt_area_pixels": -1,
            "box_intersection_pixels": -1,
            "box_union_pixels": -1,
            "box_purity": float("nan"),
            "box_recall": float("nan"),
            "box_iou": float("nan"),
            "box_overlap": -1,
            "box_hit_iou_025": -1,
            "box_hit_iou_050": -1,
            "box_area_ratio": float("nan"),
        }
    )
    return row


def _annotate_with_ground_truth(
    point_rows: list[dict[str, object]],
    box_rows: list[dict[str, object]],
    query_rows: list[dict[str, object]],
    *,
    ground_truth: GroundTruthInstances,
    assignments: Sequence[Mapping[str, object]],
) -> None:
    slot_to_target = {
        int(row["slot"]): int(row["gt_index"]) for row in assignments
    }
    masks = ground_truth.masks.detach().cpu().bool()
    point_groups: dict[tuple[str, str, int, int], list[dict[str, object]]] = {}
    for row in point_rows:
        key = _query_key(row)
        point_groups.setdefault(key, []).append(row)
        _annotate_point_row(row, masks, ground_truth, slot_to_target)

    for row in box_rows:
        slot = int(row["slot"])
        target = slot_to_target.get(slot, -1)
        row["gt_assignment_available"] = int(target >= 0)
        if target < 0:
            continue
        gt_mask = masks[int(row["sequence_index"]), target]
        gt_area = int(gt_mask.sum())
        row["gt_visible"] = int(gt_area > 0)
        row["gt_instance_id"] = int(ground_truth.instance_ids[target])
        row["gt_area_pixels"] = gt_area
        if int(row["box_available"]) != 1:
            row["box_overlap"] = 0
            row["box_hit_iou_025"] = 0
            row["box_hit_iou_050"] = 0
            continue
        box_mask = _box_to_mask(
            torch.tensor(
                [
                    float(row["box_x0"]),
                    float(row["box_y0"]),
                    float(row["box_x1"]),
                    float(row["box_y1"]),
                ]
            ),
            tuple(int(value) for value in masks.shape[-2:]),
        )
        intersection = int((box_mask & gt_mask).sum())
        union = int((box_mask | gt_mask).sum())
        purity = _safe_ratio(intersection, int(box_mask.sum()))
        recall = _safe_ratio(intersection, gt_area)
        iou = _safe_ratio(intersection, union)
        row["box_intersection_pixels"] = intersection
        row["box_union_pixels"] = union
        row["box_purity"] = purity
        row["box_recall"] = recall
        row["box_iou"] = iou
        row["box_overlap"] = int(intersection > 0)
        row["box_hit_iou_025"] = int(math.isfinite(iou) and iou >= 0.25)
        row["box_hit_iou_050"] = int(math.isfinite(iou) and iou >= 0.50)
        row["box_area_ratio"] = _safe_ratio(int(box_mask.sum()), gt_area)

    for row in query_rows:
        slot = int(row["slot"])
        target = slot_to_target.get(slot, -1)
        row["gt_assignment_available"] = int(target >= 0)
        if target < 0:
            row["gt_visible"] = 0
            row["gt_correct_count"] = -1
            row["gt_area_pixels"] = -1
            continue
        gt_mask = masks[int(row["sequence_index"]), target]
        row["gt_visible"] = int(bool(gt_mask.any()))
        row["gt_area_pixels"] = int(gt_mask.sum())
        if str(row["branch"]) == BOX_BRANCH:
            row["gt_correct_count"] = int(
                int(_find_box_row(box_rows, row)["box_overlap"]) == 1
            )
        else:
            rows = point_groups.get(_query_key(row), [])
            row["gt_correct_count"] = sum(
                int(item["accepted_prompt"]) == 1
                and int(item["gt_mask_hit"]) == 1
                for item in rows
            )


def _annotate_point_row(
    row: dict[str, object],
    masks: torch.Tensor,
    ground_truth: GroundTruthInstances,
    slot_to_target: Mapping[int, int],
) -> None:
    slot = int(row["slot"])
    target = int(slot_to_target.get(slot, -1))
    row["gt_assignment_available"] = int(target >= 0)
    if target < 0:
        row["gt_visible"] = 0
        row["gt_mask_hit"] = -1
        row["gt_instance_id"] = -1
        return
    gt_mask = masks[int(row["sequence_index"]), target]
    row["gt_visible"] = int(bool(gt_mask.any()))
    row["gt_instance_id"] = int(ground_truth.instance_ids[target])
    if int(row["in_bounds"]) != 1:
        row["gt_mask_hit"] = 0
        return
    x = int(round(float(row["projected_u"])))
    y = int(round(float(row["projected_v"])))
    height, width = gt_mask.shape
    row["gt_mask_hit"] = int(
        0 <= x < width and 0 <= y < height and bool(gt_mask[y, x])
    )


def _summarize_point_branches(
    point_rows: Sequence[Mapping[str, object]],
    query_rows: Sequence[Mapping[str, object]],
    *,
    scene_id: str,
    clip_name: str,
) -> list[dict[str, object]]:
    groups = _group_query_rows(query_rows, point_only=True)
    point_by_group: dict[tuple[str, str, int, int], list[Mapping[str, object]]] = {}
    for row in point_rows:
        if str(row["branch"]) == BOX_BRANCH:
            continue
        point_by_group.setdefault(_query_key(row), []).append(row)
    output: list[dict[str, object]] = []
    for group_key, events in sorted(groups.items()):
        branch, tolerance = group_key
        rows = [
            item
            for event in events
            for item in point_by_group.get(_query_key(event), [])
        ]
        assigned_events = [e for e in events if int(e["gt_assignment_available"]) == 1]
        visible_events = [
            e
            for e in assigned_events
            if int(e["gt_visible"]) == 1
        ]
        accepted_rows = [
            r
            for r in rows
            if int(r["gt_assignment_available"]) == 1
            and int(r["accepted_prompt"]) == 1
        ]
        correct_rows = [r for r in accepted_rows if int(r["gt_mask_hit"]) == 1]
        precision = _safe_ratio(len(correct_rows), len(accepted_rows))
        visible_accepted_rows = [
            r
            for r in accepted_rows
            if int(r["gt_visible"]) == 1
        ]
        visible_correct_rows = [
            r for r in visible_accepted_rows if int(r["gt_mask_hit"]) == 1
        ]
        visible_coverage = _safe_ratio(
            sum(int(e["gt_correct_count"]) > 0 for e in visible_events),
            len(visible_events),
        )
        strict_coverage = _safe_ratio(
            sum(int(e["gt_correct_count"]) > 0 for e in assigned_events),
            len(assigned_events),
        )
        availability = _safe_ratio(
            sum(int(e["accepted_count"]) > 0 for e in visible_events),
            len(visible_events),
        )
        all_correct = _safe_ratio(
            sum(
                int(e["accepted_count"]) > 0
                and int(e["gt_correct_count"]) == int(e["accepted_count"])
                for e in visible_events
            ),
            len(visible_events),
        )
        dispersions = [
            float(e["accepted_normalized_dispersion"])
            for e in events
            if int(e["accepted_count"]) >= 2
            and math.isfinite(float(e["accepted_normalized_dispersion"]))
        ]
        mean_pairwise = [
            float(e["accepted_mean_pairwise_px"])
            for e in events
            if int(e["accepted_count"]) >= 2
        ]
        row = {
            "scene_id": scene_id,
            "clip": clip_name,
            "prompt_type": "point",
            "branch": branch,
            "tolerance": tolerance,
            "query_count": len(events),
            "assigned_queries": len(assigned_events),
            "visible_assigned_queries": len(visible_events),
            "candidate_count": len(rows),
            "in_bounds_count": sum(int(r["in_bounds"]) for r in rows),
            "accepted_prompt_count": len(accepted_rows),
            "correct_prompt_count": len(correct_rows),
            "in_bounds_rate": _safe_ratio(
                sum(int(r["in_bounds"]) for r in rows), len(rows)
            ),
            "accepted_prompt_precision": precision,
            "visible_prompt_precision": _safe_ratio(
                len(visible_correct_rows), len(visible_accepted_rows)
            ),
            "object_frame_coverage": visible_coverage,
            "object_frame_coverage_strict": strict_coverage,
            "prompt_availability": availability,
            "precision_coverage_f1": harmonic_mean(precision, visible_coverage),
            "precision_strict_coverage_f1": harmonic_mean(
                precision, strict_coverage
            ),
            "all_points_correct_frame_rate": all_correct,
            "mean_candidates_per_query": _safe_ratio(len(rows), len(events)),
            "mean_accepted_per_query": _safe_ratio(
                sum(int(e["accepted_count"]) for e in events), len(events)
            ),
            "mean_history_gap": _mean(
                [float(e["history_gap"]) for e in events]
            ),
            "mean_pairwise_px_accepted": _mean(mean_pairwise),
            "mean_normalized_dispersion_accepted": _mean(dispersions),
            "raw_mask_hit_rate": _safe_ratio(
                sum(int(r["current_raw_mask_hit"]) for r in rows),
                sum(int(r["in_bounds"]) for r in rows),
            ),
        }
        output.append(row)
    return output


def _summarize_per_object(
    point_rows: Sequence[Mapping[str, object]],
    query_rows: Sequence[Mapping[str, object]],
    *,
    scene_id: str,
    clip_name: str,
) -> list[dict[str, object]]:
    """Return point metrics split by persistent V0 slot.

    The scene aggregate can be dominated by a few easy objects.  This table
    preserves the long-tail evidence needed before promoting a prompt branch
    to a real SAM A/B experiment.
    """

    event_groups: dict[tuple[str, str, int], list[Mapping[str, object]]] = {}
    for event in query_rows:
        if str(event["branch"]) == BOX_BRANCH:
            continue
        key = (
            str(event["branch"]),
            str(event.get("tolerance", "")),
            int(event["slot"]),
        )
        event_groups.setdefault(key, []).append(event)
    point_groups: dict[tuple[str, str, int, int], list[Mapping[str, object]]] = {}
    for point in point_rows:
        if str(point["branch"]) == BOX_BRANCH:
            continue
        point_groups.setdefault(_query_key(point), []).append(point)

    output: list[dict[str, object]] = []
    for (branch, tolerance, slot), events in sorted(event_groups.items()):
        rows = [
            point
            for event in events
            for point in point_groups.get(_query_key(event), [])
        ]
        assigned = [
            event for event in events
            if int(event["gt_assignment_available"]) == 1
        ]
        visible = [
            event for event in assigned
            if int(event["gt_visible"]) == 1
        ]
        accepted = [
            point
            for point in rows
            if int(point["gt_assignment_available"]) == 1
            and int(point["accepted_prompt"]) == 1
        ]
        correct = [point for point in accepted if int(point["gt_mask_hit"]) == 1]
        precision = _safe_ratio(len(correct), len(accepted))
        coverage = _safe_ratio(
            sum(int(event["gt_correct_count"]) > 0 for event in visible),
            len(visible),
        )
        dispersions = [
            float(event["accepted_normalized_dispersion"])
            for event in events
            if int(event["accepted_count"]) >= 2
            and math.isfinite(float(event["accepted_normalized_dispersion"]))
        ]
        output.append(
            {
                "scene_id": scene_id,
                "clip": clip_name,
                "prompt_type": "point",
                "branch": branch,
                "tolerance": tolerance,
                "slot": int(slot),
                "sam_track_id": int(events[0]["sam_track_id"]),
                "query_count": len(events),
                "assigned_queries": len(assigned),
                "visible_assigned_queries": len(visible),
                "accepted_prompt_count": len(accepted),
                "correct_prompt_count": len(correct),
                "accepted_prompt_precision": precision,
                "object_frame_coverage": coverage,
                "precision_coverage_f1": harmonic_mean(precision, coverage),
                "prompt_availability": _safe_ratio(
                    sum(int(event["accepted_count"]) > 0 for event in visible),
                    len(visible),
                ),
                "mean_normalized_dispersion_accepted": _mean(dispersions),
            }
        )
    return output


def _summarize_box_branches(
    box_rows: Sequence[Mapping[str, object]],
    query_rows: Sequence[Mapping[str, object]],
    *,
    scene_id: str,
    clip_name: str,
) -> list[dict[str, object]]:
    events = [row for row in query_rows if str(row["branch"]) == BOX_BRANCH]
    if not events:
        return []
    assigned = [e for e in events if int(e["gt_assignment_available"]) == 1]
    visible = [e for e in assigned if int(e["gt_visible"]) == 1]
    boxes = [
        row
        for row in box_rows
        if int(row["gt_assignment_available"]) == 1
    ]
    visible_boxes = [row for row in boxes if int(row["gt_visible"]) == 1]
    available_visible = [row for row in visible_boxes if int(row["box_available"]) == 1]
    return [
        {
            "scene_id": scene_id,
            "clip": clip_name,
            "prompt_type": "box",
            "branch": BOX_BRANCH,
            "tolerance": "",
            "query_count": len(events),
            "assigned_queries": len(assigned),
            "visible_assigned_queries": len(visible),
            "candidate_count": len(box_rows),
            "box_available_count": sum(int(r["box_available"]) for r in box_rows),
            "box_availability": _safe_ratio(
                sum(int(r["box_available"]) for r in visible), len(visible)
            ),
            "box_purity_mean": _mean(
                [float(r["box_purity"]) for r in available_visible]
            ),
            "box_purity_median": _median(
                [float(r["box_purity"]) for r in available_visible]
            ),
            "box_recall_mean": _mean(
                [float(r["box_recall"]) for r in available_visible]
            ),
            "box_iou_mean": _mean(
                [float(r["box_iou"]) for r in available_visible]
            ),
            "box_iou_median": _median(
                [float(r["box_iou"]) for r in available_visible]
            ),
            "box_overlap_rate": _safe_ratio(
                sum(int(r["box_overlap"]) for r in available_visible),
                len(available_visible),
            ),
            "box_hit_iou_025": _safe_ratio(
                sum(int(r["box_hit_iou_025"]) for r in available_visible),
                len(available_visible),
            ),
            "box_hit_iou_050": _safe_ratio(
                sum(int(r["box_hit_iou_050"]) for r in available_visible),
                len(available_visible),
            ),
            "box_object_frame_coverage_iou025": _safe_ratio(
                sum(int(r["box_hit_iou_025"]) == 1 for r in visible_boxes),
                len(visible),
            ),
            "box_object_frame_coverage_iou050": _safe_ratio(
                sum(int(r["box_hit_iou_050"]) == 1 for r in visible_boxes),
                len(visible),
            ),
            "box_area_ratio_mean": _mean(
                [float(r["box_area_ratio"]) for r in available_visible]
            ),
            "mean_history_gap": _mean(
                [float(e["history_gap"]) for e in events]
            ),
        }
    ]


def _group_query_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    point_only: bool = False,
) -> dict[tuple[str, str], list[Mapping[str, object]]]:
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for row in rows:
        branch = str(row["branch"])
        if point_only and branch == BOX_BRANCH:
            continue
        key = (branch, str(row.get("tolerance", "")))
        grouped.setdefault(key, []).append(row)
    return grouped


def _query_key(row: Mapping[str, object]) -> tuple[str, str, int, int]:
    return (
        str(row["branch"]),
        str(row.get("tolerance", "")),
        int(row["sequence_index"]),
        int(row["slot"]),
    )


def _find_box_row(
    box_rows: Sequence[Mapping[str, object]],
    query_row: Mapping[str, object],
) -> Mapping[str, object]:
    for row in box_rows:
        if (
            int(row["sequence_index"]) == int(query_row["sequence_index"])
            and int(row["slot"]) == int(query_row["slot"])
        ):
            return row
    raise KeyError("Box row for query was not found.")


def _build_summary(
    *,
    payload: Mapping[str, object],
    baseline_summary: Mapping[str, object],
    cache_path_value: Path,
    output_dir: Path,
    arrays: V0Arrays,
    candidates: CandidateBundle,
    branch_summary: Sequence[Mapping[str, object]],
    per_object_summary: Sequence[Mapping[str, object]],
    tracking: Mapping[str, object],
    args: argparse.Namespace,
) -> dict[str, object]:
    return {
        "schema": 1,
        "revision": REVISION,
        "decision": "DIAGNOSTIC_ONLY",
        "diagnostic_only": 1,
        "clip": str(payload["clip_name"]),
        "scene_id": str(payload["scene_id"]),
        "frames": [int(value) for value in payload["frame_indices"]],
        "cache": str(cache_path_value),
        "output_dir": str(output_dir),
        "baseline_revision": str(
            baseline_summary.get("implementation_revision", "")
        ),
        "pose_source": "raw_streamvggt",
        "prompt_source": "frozen_v0_history_geometry",
        "candidate_generation_gt_fields": 0,
        "evaluation_gt_fields": 1,
        "models_rerun": 0,
        "sam_called": 0,
        "parameters_updated": 0,
        "pointmap_modified": 0,
        "pose_modified": 0,
        "sam_rerun_with_temporal_prompts": 0,
        "tracking_assignment_count": len(tracking["assignments"]),
        "parameters": {
            "confidence_threshold": float(args.confidence_threshold),
            "track_score_threshold": float(args.track_score_threshold),
            "max_history_points": int(args.max_history_points),
            "front_fraction": float(args.front_fraction),
            "box_quantile": float(args.box_quantile),
            "min_box_points": int(args.min_box_points),
            "absolute_depth_tolerance": float(args.absolute_depth_tolerance),
            "depth_relative_tolerances": [
                float(value) for value in args.depth_relative_tolerances
            ],
        },
        "candidate_metadata": dict(candidates.metadata),
        "branch_summary": [dict(row) for row in branch_summary],
        "per_object_summary": [dict(row) for row in per_object_summary],
        "tracking_summary": [dict(row) for row in tracking["summary_rows"]],
        "interpretation": {
            "primary_point_precision": (
                "correct accepted in-bounds points / accepted in-bounds points "
                "for slots with frozen raw-to-GT assignment"
            ),
            "object_frame_coverage": (
                "visible assigned query frame-slots with at least one correct "
                "accepted point / visible assigned query frame-slots"
            ),
            "coverage_f1": (
                "harmonic mean of primary point precision and visible object-" 
                "frame coverage"
            ),
            "box_metrics": (
                "boxes are reported with purity, recall, IoU, and IoU-hit "
                "rates; they are not treated as point prompts"
            ),
            "gate_selection": (
                "relative depth tolerances are a predeclared curve; no test "
                "value is selected automatically"
            ),
        },
        "next_gate": (
            "Only a branch with a meaningful coverage-preserving improvement "
            "over A_center should proceed to a real SAM A/B rerun."
        ),
        "outputs": {},
    }


def _write_copyable(path: Path, summary: Mapping[str, object]) -> None:
    lines = [
        "===== V0_TEMPORAL_PROMPT_MATRIX_BEGIN =====",
        f"revision={summary['revision']}",
        f"decision={summary['decision']}",
        f"clip={summary['clip']}",
        f"scene_id={summary['scene_id']}",
        "pose_source=raw_streamvggt",
        "candidate_generation_gt_fields=0",
        "evaluation_gt_fields=1",
        "models_rerun=0",
        "sam_called=0",
        "pointmap_modified=0",
        "sam_rerun_with_temporal_prompts=0",
        "metrics=precision,visible_object_frame_coverage,coverage_f1,availability,spatial_dispersion",
        "",
        "branch_summary_json="
        + json.dumps(
            summary["branch_summary"],
            sort_keys=True,
            allow_nan=True,
            default=str,
        ),
        "",
        "interpretation=diagnostic_only; no branch was fed back into SAM",
        "next_gate=" + str(summary["next_gate"]),
        "===== V0_TEMPORAL_PROMPT_MATRIX_END =====",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def _print_summary(
    summary: Mapping[str, object],
    branch_summary: Sequence[Mapping[str, object]],
    metadata: Mapping[str, object],
) -> None:
    print("===== V0 TEMPORAL PROMPT MATRIX SUMMARY =====")
    print(
        "candidate_provenance "
        f"queries={metadata.get('query_count', 0)} "
        f"history_observations={metadata.get('history_observation_count', 0)} "
        f"source_points_median={_display(metadata.get('source_point_count_median'))}"
    )
    print(
        "columns=precision coverage coverage_f1 availability "
        "mean_dispersion_px normalized_dispersion"
    )
    for row in branch_summary:
        if row.get("prompt_type") == "point":
            print(
                "  "
                f"branch={row.get('branch')} "
                f"tol={row.get('tolerance', '') or '-'} "
                f"queries={row.get('query_count', 0)} "
                f"accepted={row.get('accepted_prompt_count', 0)} "
                f"precision={_display(row.get('accepted_prompt_precision'))} "
                f"coverage={_display(row.get('object_frame_coverage'))} "
                f"coverage_f1={_display(row.get('precision_coverage_f1'))} "
                f"availability={_display(row.get('prompt_availability'))} "
                f"disp_px={_display(row.get('mean_pairwise_px_accepted'))} "
                f"disp_norm={_display(row.get('mean_normalized_dispersion_accepted'))}"
            )
        else:
            print(
                "  "
                f"branch={row.get('branch')} box_availability="
                f"{_display(row.get('box_availability'))} "
                f"purity={_display(row.get('box_purity_mean'))} "
                f"recall={_display(row.get('box_recall_mean'))} "
                f"IoU={_display(row.get('box_iou_mean'))} "
                f"IoU025_coverage="
                f"{_display(row.get('box_object_frame_coverage_iou025'))} "
                f"IoU050_coverage="
                f"{_display(row.get('box_object_frame_coverage_iou050'))}"
            )
    print(
        "interpretation=diagnostic_only; compare coverage-preserving gains to "
        "A_center before any SAM A/B rerun"
    )


def _rescale_uv(
    uv: torch.Tensor,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> torch.Tensor:
    """Map pixel centers between image grids without using annotations."""

    source_h, source_w = (float(value) for value in source_size)
    target_h, target_w = (float(value) for value in target_size)
    output = uv.clone().float()
    output[:, 0] = (output[:, 0] + 0.5) * target_w / source_w - 0.5
    output[:, 1] = (output[:, 1] + 0.5) * target_h / source_h - 0.5
    return output


def _box_to_mask(box: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    height, width = image_size
    x0, y0, x1, y1 = (float(value) for value in box)
    left = max(0, int(math.ceil(x0)))
    top = max(0, int(math.ceil(y0)))
    right = min(width - 1, int(math.floor(x1)))
    bottom = min(height - 1, int(math.floor(y1)))
    mask = torch.zeros(height, width, dtype=torch.bool)
    if right >= left and bottom >= top:
        mask[top : bottom + 1, left : right + 1] = True
    return mask


def _box_area_pixels(box: torch.Tensor, image_size: tuple[int, int]) -> int:
    return int(_box_to_mask(box, image_size).sum())


def _sample_mask(mask: torch.Tensor, uv: torch.Tensor) -> bool:
    if not bool(torch.isfinite(uv).all()):
        return False
    height, width = mask.shape[-2:]
    x = int(torch.round(uv[0]).item())
    y = int(torch.round(uv[1]).item())
    if x < 0 or x >= width or y < 0 or y >= height:
        return False
    return bool(mask[y, x])


def _tolerance_label(value: float) -> str:
    return f"{float(value):.6f}"


def _format_float_list(values: Sequence[float]) -> str:
    return ",".join(f"{float(value):.2f}" for value in values)


def _finite_float(value: torch.Tensor | float) -> float:
    number = float(value.item()) if torch.is_tensor(value) else float(value)
    return number if math.isfinite(number) else float("nan")


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if float(denominator) else float("nan")


def _mean(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return sum(finite) / len(finite) if finite else float("nan")


def _median(values: Sequence[float]) -> float:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return float("nan")
    middle = len(finite) // 2
    if len(finite) % 2:
        return finite[middle]
    return 0.5 * (finite[middle - 1] + finite[middle])


def _display(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.6f}" if math.isfinite(number) else "nan"


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    materialized = [dict(row) for row in rows]
    if not materialized:
        path.write_text("\n", encoding="utf8")
        return
    fields = sorted({str(key) for row in materialized for key in row})
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in materialized:
            writer.writerow({field: _csv_value(row.get(field, "")) for field in fields})


def _csv_value(value: object) -> object:
    if torch.is_tensor(value):
        if value.ndim == 0:
            return value.item()
        return json.dumps(value.detach().cpu().tolist(), allow_nan=True)
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True, allow_nan=True, default=str)
    return value


if __name__ == "__main__":
    main()
