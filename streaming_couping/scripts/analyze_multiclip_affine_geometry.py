#!/usr/bin/env python3
"""Diagnose causal depth affine calibration and ray reconstruction.

This is an evaluation-only experiment over the frozen multi-scene V0 caches.
It does not rerun StreamVGGT or SAM and it never writes a corrected pointmap
into the deployed baseline.  Candidate generation uses only the raw
StreamVGGT pointmap/depth, raw SAM masks, confidence, intrinsics, and raw
StreamVGGT pose.  Ground truth fields are first accessed after every causal
pair, affine fit, and candidate geometry branch has been frozen.

The two no-GT pair sources are intentionally separate:

``depth_head_to_historical_pointmap``
    Current depth-head Z is paired with the Z of an earlier same-slot
    pointmap observation reprojected into the current frame.

``pointmap_z_to_historical_pointmap``
    Current raw pointmap Z is paired with the same historical pointmap target.

For each clip, a robust positive-slope affine map is fitted on the first
``calibration_fraction`` of frames and evaluated causally on the suffix.  The
geometry branches replace only the raw SAM object union; background points
remain the original raw pointmap.  The depth-head branches reconstruct world
points from the corrected depth along the current pixel rays.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from streaming_couping.scripts.run_semantic_map_evaluation import (
    _load_run as _load_evaluation_run,
)
from streaming_couping.src.learned_pose.cache import cache_path, load_feature_cache
from streaming_couping.src.learned_pose.config import (
    ClipConfig,
    LearnedPoseConfig,
    load_learned_pose_config,
)
from streaming_couping.src.robust_depth_feedback import (
    HistoricalDepthVetoConfig,
    RobustAffineFit,
    affine_error_metrics,
    apply_ray_depth_affine,
    build_causal_history_cache,
    fit_robust_affine,
    resize_intrinsics,
    transform_world_points,
)
from streaming_couping.src.semantic_map_metrics import (
    apply_similarity,
    evaluate_semantic_object_map,
    load_ground_truth_stream_masks,
)
from streaming_couping.src.semantic_tracking_metrics import (
    GroundTruthInstances,
    evaluate_tracking_variants,
    load_ground_truth_instances,
)
from streaming_couping.src.storage import expand_storage_path
from streaming_couping.src.temporal_prompt_matrix import (
    project_world_points,
    sample_depth_nearest,
)


REVISION = "multiclip_affine_geometry_diagnostic_r1"
DEFAULT_CONFIG = "streaming_couping/configs/v0_baseline.yaml"
DEFAULT_OUTPUT = (
    "${VGGT_SAM_STORAGE_ROOT}/outputs/"
    "streaming_couping_multiclip_affine_geometry"
)

RAW_VARIANT = "raw_v0"
POINTMAP_AFFINE_VARIANT = "pointmap_self_affine_holdout"
DEPTH_IDENTITY_VARIANT = "depth_head_ray_identity_holdout"
DEPTH_AFFINE_VARIANT = "depth_head_ray_self_affine_holdout"
DEPTH_AFFINE_ALL_VARIANT = "depth_head_ray_self_affine_all"
GEOMETRY_VARIANTS = (
    RAW_VARIANT,
    POINTMAP_AFFINE_VARIANT,
    DEPTH_IDENTITY_VARIANT,
    DEPTH_AFFINE_VARIANT,
    DEPTH_AFFINE_ALL_VARIANT,
)

DEPTH_HEAD_SOURCE = "depth_head_to_historical_pointmap"
POINTMAP_SOURCE = "pointmap_z_to_historical_pointmap"
PAIR_SOURCES = (DEPTH_HEAD_SOURCE, POINTMAP_SOURCE)


@dataclass(frozen=True)
class ClipArrays:
    """Whitelist of non-GT tensors used during candidate generation."""

    clip: ClipConfig
    payload: Mapping[str, Any]
    points: torch.Tensor
    confidence: torch.Tensor
    raw_masks: torch.Tensor
    scores: torch.Tensor
    depth: torch.Tensor
    intrinsics: torch.Tensor
    raw_world_to_camera: torch.Tensor
    image_size: tuple[int, int]
    map_size: tuple[int, int]


@dataclass(frozen=True)
class PairCollection:
    """Causal scalar pairs and their compact per-query audit rows."""

    source: torch.Tensor
    target: torch.Tensor
    frame: torch.Tensor
    rows: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class ClipCandidates:
    """Frozen geometry branches and no-GT affine metadata for one clip."""

    branches: Mapping[str, torch.Tensor]
    applied_masks: Mapping[str, torch.Tensor]
    pair_data: Mapping[str, PairCollection]
    affine_fits: Mapping[str, RobustAffineFit]
    calibration_count: int
    metadata: Mapping[str, object]


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    config_path = Path(args.config).expanduser().resolve()
    data = load_learned_pose_config(config_path)
    clips = _select_clips(data.clips, args.clip)
    evaluation_run = _load_evaluation_run(config_path)
    output_dir = expand_storage_path(
        args.output_dir or DEFAULT_OUTPUT,
        base=config_path.parent,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    pose_root = (
        expand_storage_path(args.pose_root, base=config_path.parent)
        if args.pose_root
        else None
    )
    print("MULTI-CLIP CAUSAL AFFINE / DEPTH-HEAD RAY DIAGNOSTIC")
    print(f"config={config_path}")
    print(f"output={output_dir}")
    print(
        "models are not rerun; raw V0 caches are reused; GT is opened only "
        "after causal candidates are frozen"
    )
    print(
        "branches=" + ",".join(GEOMETRY_VARIANTS)
    )
    print(
        "pair_sources=" + ",".join(PAIR_SOURCES)
        + "; pointmap branches use raw StreamVGGT pose only"
    )

    # ------------------------------------------------------------------
    # Candidate phase.  No target_* field and no manifest annotation is read
    # below this marker.  ``load_feature_cache`` loads a serialized mapping,
    # but all accesses in this phase are limited to the explicit non-GT
    # whitelist in ``_load_clip_arrays``.
    # ------------------------------------------------------------------
    all_clip_arrays: list[ClipArrays] = []
    all_candidates: list[ClipCandidates] = []
    selected_pose_status: list[dict[str, object]] = []
    for clip in clips:
        arrays = _load_clip_arrays(data, clip, args.raw_cache_variant)
        all_clip_arrays.append(arrays)
        selected_pose, selected_path, selected_status = _load_selected_pose(
            arrays,
            pose_root=pose_root,
        )
        selected_pose_status.append(
            {
                "clip": clip.name,
                "selected_pose_available": int(selected_pose is not None),
                "selected_pose_path": str(selected_path) if selected_path else "",
                "status": selected_status,
            }
        )
        pose_sources: dict[str, torch.Tensor] = {
            "raw_streamvggt": arrays.raw_world_to_camera,
        }
        if args.pose_source == "both":
            if selected_pose is None:
                raise FileNotFoundError(
                    f"--pose-source=both requires a per-clip poses.pt for "
                    f"{clip.name}; searched under {pose_root or '<default paths>'}."
                )
            pose_sources["selected_v0"] = selected_pose
        candidates = _build_clip_candidates(
            arrays,
            pose_sources=pose_sources,
            calibration_fraction=float(args.calibration_fraction),
            history_config=_history_config(args),
            max_pair_points=int(args.max_pair_points_per_query),
            min_affine_samples=int(args.min_affine_samples),
            max_affine_samples=int(args.max_affine_samples),
            affine_trim_quantile=float(args.affine_trim_quantile),
            affine_max_iterations=int(args.affine_max_iterations),
        )
        all_candidates.append(candidates)
        _save_candidate_artifact(output_dir, arrays, candidates)

    candidate_metadata = {
        "schema": 1,
        "revision": REVISION,
        "candidate_generation_gt_fields": 0,
        "models_rerun": 0,
        "streamvggt_rerun": 0,
        "sam_rerun": 0,
        "pose_modified": 0,
        "pointmap_modified": 0,
        "raw_cache_variant": str(args.raw_cache_variant),
        "pose_source": str(args.pose_source),
        "calibration_fraction": float(args.calibration_fraction),
        "clips": [
            {
                "clip": arrays.clip.name,
                "scene_id": arrays.clip.scene_id,
                "split": arrays.clip.split,
                "frame_count": len(arrays.payload["frame_indices"]),
                "calibration_count": int(candidate.calibration_count),
                "holdout_count": int(
                    len(arrays.payload["frame_indices"])
                    - candidate.calibration_count
                ),
                "pair_rows": int(
                    sum(len(value.rows) for value in candidate.pair_data.values())
                ),
                "pair_samples": int(
                    sum(
                        int(value.source.numel())
                        for value in candidate.pair_data.values()
                    )
                ),
                "affine_fits": {
                    key: fit.to_dict()
                    for key, fit in candidate.affine_fits.items()
                },
                "branches": _json_safe(dict(candidate.metadata)),
            }
            for arrays, candidate in zip(all_clip_arrays, all_candidates)
        ],
        "selected_pose_status": selected_pose_status,
    }
    _write_json(output_dir / "candidate_metadata.json", candidate_metadata)
    pair_rows = [
        row
        for candidate in all_candidates
        for pair in candidate.pair_data.values()
        for row in pair.rows
    ]
    _write_csv(output_dir / "scale_shift_pairs_causal.csv", pair_rows)
    scale_rows = _scale_shift_rows(all_clip_arrays, all_candidates)
    _write_csv(output_dir / "scale_shift_metrics.csv", scale_rows)
    print(
        "candidates frozen "
        f"clips={len(all_clip_arrays)} "
        f"total_frames={sum(len(item.payload['frame_indices']) for item in all_clip_arrays)} "
        f"pair_rows={len(pair_rows)} "
        f"pair_samples={sum(int(value.source.numel()) for candidate in all_candidates for value in candidate.pair_data.values())}"
    )

    # ------------------------------------------------------------------
    # Sealed evaluation phase.  GT and target geometry are first opened here.
    # ------------------------------------------------------------------
    print("candidate artifacts frozen; opening sealed GT evaluation")
    tracking_rows: list[dict[str, object]] = []
    map_rows: list[dict[str, object]] = []
    map_object_rows: list[dict[str, object]] = []
    depth_rows: list[dict[str, object]] = []
    per_clip_rows: list[dict[str, object]] = []
    for arrays, candidates in zip(all_clip_arrays, all_candidates):
        evaluation = _evaluate_clip(
            arrays,
            candidates,
            manifest=data.manifest,
            evaluation_run=evaluation_run,
            confidence_threshold=float(args.confidence_threshold),
        )
        tracking_rows.extend(evaluation["tracking_rows"])
        map_rows.extend(evaluation["map_rows"])
        map_object_rows.extend(evaluation["map_object_rows"])
        depth_rows.extend(evaluation["depth_rows"])
        per_clip_rows.extend(evaluation["per_clip_rows"])

    aggregate = _aggregate_map_rows(map_rows)
    affine_gate = _affine_evidence_gate(scale_rows, all_clip_arrays)
    geometry_gate = _geometry_gate(map_rows, all_clip_arrays)
    gate = "GO" if affine_gate["pass"] and geometry_gate["pass"] else "NO_GO"

    _write_csv(output_dir / "depth_source_metrics.csv", depth_rows)
    _write_csv(output_dir / "map_metrics.csv", map_rows)
    _write_csv(output_dir / "map_object_metrics.csv", map_object_rows)
    _write_csv(output_dir / "tracking_metrics.csv", tracking_rows)
    _write_csv(output_dir / "per_clip_metrics.csv", per_clip_rows)
    _write_csv(output_dir / "aggregate_metrics.csv", aggregate["rows"])

    summary = {
        "schema": 1,
        "revision": REVISION,
        "decision": "DIAGNOSTIC_ONLY",
        "diagnostic_only": 1,
        "gate": gate,
        "config": str(config_path),
        "output_dir": str(output_dir),
        "clip_count": len(all_clip_arrays),
        "scene_count": len({item.clip.scene_id for item in all_clip_arrays}),
        "total_frame_count": sum(
            len(item.payload["frame_indices"]) for item in all_clip_arrays
        ),
        "candidate_generation_gt_fields": 0,
        "evaluation_gt_fields": 1,
        "models_rerun": 0,
        "streamvggt_rerun": 0,
        "sam_rerun": 0,
        "pose_modified": 0,
        "pointmap_modified": 0,
        "calibration_fraction": float(args.calibration_fraction),
        "clips": [
            {
                "clip": arrays.clip.name,
                "scene_id": arrays.clip.scene_id,
                "split": arrays.clip.split,
                "frames": list(int(value) for value in arrays.payload["frame_indices"]),
                "frame_count": len(arrays.payload["frame_indices"]),
                "calibration_count": int(candidate.calibration_count),
                "holdout_count": int(
                    len(arrays.payload["frame_indices"])
                    - candidate.calibration_count
                ),
                "cache": str(cache_path(data, arrays.clip)),
                "affine_fits": {
                    key: fit.to_dict()
                    for key, fit in candidate.affine_fits.items()
                },
                "candidate_metadata": _json_safe(dict(candidate.metadata)),
            }
            for arrays, candidate in zip(all_clip_arrays, all_candidates)
        ],
        "affine_evidence_gate": affine_gate,
        "geometry_gate": geometry_gate,
        "aggregate_map_metrics": aggregate,
        "outputs": {},
        "interpretation": {
            "primary_pose": "raw_streamvggt",
            "selected_pose": (
                "self_consistency_control_only; never used to construct map branches"
            ),
            "affine": (
                "per-clip positive-slope fit on the calibration prefix; holdout "
                "is causal and never used to fit parameters"
            ),
            "ray_reconstruction": (
                "corrected depth is backprojected on the existing pixel ray and "
                "only raw SAM object-union pixels are replaced"
            ),
            "gt_role": (
                "GT is evaluation-only; the GT affine upper bound is not fit and "
                "no correction is promoted automatically"
            ),
            "scope_caveat": (
                "four pilot scene-disjoint clips are a validation diagnostic, not "
                "a new formal held-out test rotation"
            ),
        },
    }
    outputs = {
        "summary": output_dir / "summary.json",
        "copyable_result": output_dir / "copyable_result.txt",
        "candidate_metadata": output_dir / "candidate_metadata.json",
        "scale_shift_pairs_causal": output_dir / "scale_shift_pairs_causal.csv",
        "scale_shift_metrics": output_dir / "scale_shift_metrics.csv",
        "depth_source_metrics": output_dir / "depth_source_metrics.csv",
        "map_metrics": output_dir / "map_metrics.csv",
        "map_object_metrics": output_dir / "map_object_metrics.csv",
        "tracking_metrics": output_dir / "tracking_metrics.csv",
        "per_clip_metrics": output_dir / "per_clip_metrics.csv",
        "aggregate_metrics": output_dir / "aggregate_metrics.csv",
    }
    summary["outputs"] = {key: str(value) for key, value in outputs.items()}
    _write_json(outputs["summary"], summary)
    _write_copyable(outputs["copyable_result"], summary, aggregate, affine_gate, geometry_gate)
    _print_summary(
        all_clip_arrays,
        all_candidates,
        scale_rows,
        map_rows,
        depth_rows,
        affine_gate,
        geometry_gate,
        gate,
    )
    print(f"summary={outputs['summary']}")
    print(f"copyable_result={outputs['copyable_result']}")
    print("decision=DIAGNOSTIC_ONLY")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--clip", action="append", default=None)
    parser.add_argument(
        "--pose-source",
        choices=("raw", "both"),
        default="raw",
        help="Use raw pose only, or add selected V0 pose as a self-consistency control.",
    )
    parser.add_argument(
        "--pose-root",
        default=None,
        help="Multi-clip run root containing v0/<clip>/poses.pt; required for --pose-source=both.",
    )
    parser.add_argument("--raw-cache-variant", default="sam31_online_forward")
    parser.add_argument("--confidence-threshold", type=float, default=0.30)
    parser.add_argument("--track-score-threshold", type=float, default=0.50)
    parser.add_argument("--min-history-points", type=int, default=16)
    parser.add_argument("--max-history-points", type=int, default=4096)
    parser.add_argument("--max-points-per-history-frame", type=int, default=512)
    parser.add_argument("--max-pair-points-per-query", type=int, default=2048)
    parser.add_argument("--min-affine-samples", type=int, default=64)
    parser.add_argument("--max-affine-samples", type=int, default=200000)
    parser.add_argument("--affine-trim-quantile", type=float, default=0.90)
    parser.add_argument("--affine-max-iterations", type=int, default=4)
    parser.add_argument("--calibration-fraction", type=float, default=0.70)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if not 0.0 <= float(args.confidence_threshold) <= 1.0:
        raise ValueError("--confidence-threshold must be in [0,1].")
    if not 0.0 <= float(args.track_score_threshold) <= 1.0:
        raise ValueError("--track-score-threshold must be in [0,1].")
    if int(args.min_history_points) < 1:
        raise ValueError("--min-history-points must be positive.")
    if int(args.max_history_points) < int(args.min_history_points):
        raise ValueError("--max-history-points must be >= min-history-points.")
    if int(args.max_points_per_history_frame) < 1:
        raise ValueError("--max-points-per-history-frame must be positive.")
    if int(args.max_pair_points_per_query) < 1:
        raise ValueError("--max-pair-points-per-query must be positive.")
    if int(args.min_affine_samples) < 2:
        raise ValueError("--min-affine-samples must be >= 2.")
    if int(args.max_affine_samples) < int(args.min_affine_samples):
        raise ValueError("--max-affine-samples must be >= --min-affine-samples.")
    if not 0.5 <= float(args.affine_trim_quantile) < 1.0:
        raise ValueError("--affine-trim-quantile must be in [0.5,1).")
    if int(args.affine_max_iterations) < 1:
        raise ValueError("--affine-max-iterations must be positive.")
    if not 0.0 < float(args.calibration_fraction) < 1.0:
        raise ValueError("--calibration-fraction must be in (0,1).")


def _select_clips(
    clips: Sequence[ClipConfig], names: Sequence[str] | None
) -> tuple[ClipConfig, ...]:
    if not names:
        return tuple(clips)
    wanted = tuple(dict.fromkeys(str(value) for value in names))
    by_name = {clip.name: clip for clip in clips}
    missing = [name for name in wanted if name not in by_name]
    if missing:
        raise ValueError(f"Requested clips are absent from config: {missing}.")
    return tuple(by_name[name] for name in wanted)


def _load_clip_arrays(
    data: LearnedPoseConfig,
    clip: ClipConfig,
    raw_cache_variant: str,
) -> ClipArrays:
    path = cache_path(data, clip)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing frozen cache for {clip.name}: {path}; run the frozen "
            "multi-scene baseline first."
        )
    payload = load_feature_cache(path)
    _validate_cache_identity(payload, clip)

    # Keep this access list deliberately GT-free.  The target fields are not
    # even looked up until _evaluate_clip below.
    points = _tensor(payload["baseline_world_points"], "baseline_world_points").float()
    confidence = _scalar_map(
        payload["baseline_world_confidence"],
        "baseline_world_confidence",
    )
    masks_by_variant = payload.get("tracking_variant_masks_stream")
    scores_by_variant = payload.get("tracking_variant_scores")
    if not isinstance(masks_by_variant, Mapping) or not isinstance(scores_by_variant, Mapping):
        raise ValueError(f"Cache {clip.name!r} lacks tracking variant mappings.")
    if raw_cache_variant not in masks_by_variant or raw_cache_variant not in scores_by_variant:
        raise ValueError(
            f"Cache {clip.name!r} lacks raw cache variant {raw_cache_variant!r}."
        )
    raw_masks = _tensor(masks_by_variant[raw_cache_variant], "tracking_masks_stream").bool()
    scores = _tensor(scores_by_variant[raw_cache_variant], "tracking_scores").float()
    depth = _scalar_map(payload["baseline_depth"], "baseline_depth")
    intrinsics = _tensor(payload["baseline_intrinsics"], "baseline_intrinsics").float()
    raw_w2c = _pose_sequence(
        _tensor(payload["baseline_world_to_camera"], "baseline_world_to_camera")
    )
    frame_indices = tuple(int(value) for value in payload["frame_indices"])
    if frame_indices != tuple(int(value) for value in clip.frame_indices):
        raise ValueError(f"Cache frame order differs from clip {clip.name!r}.")
    if points.ndim != 4 or tuple(points.shape[-1:]) != (3,):
        raise ValueError(f"{clip.name}: baseline_world_points must be [S,H,W,3].")
    sequence, height, width = points.shape[:3]
    if confidence.shape != (sequence, height, width):
        raise ValueError(f"{clip.name}: confidence does not match pointmap.")
    if raw_masks.ndim != 4 or raw_masks.shape[0] != sequence or tuple(raw_masks.shape[-2:]) != (height, width):
        raise ValueError(f"{clip.name}: raw tracking masks are malformed.")
    if scores.shape != raw_masks.shape[:2]:
        raise ValueError(f"{clip.name}: tracking scores do not match masks.")
    if depth.shape[0] != sequence:
        raise ValueError(f"{clip.name}: depth/frame count mismatch.")
    if intrinsics.shape != (sequence, 3, 3):
        raise ValueError(f"{clip.name}: intrinsics must be [S,3,3].")
    if raw_w2c.shape != (sequence, 3, 4):
        raise ValueError(f"{clip.name}: raw pose must be [S,3,4].")
    if len(payload["sam_track_ids"]) != raw_masks.shape[1] or len(payload["sam_track_prompts"]) != raw_masks.shape[1]:
        raise ValueError(f"{clip.name}: track metadata does not match mask slots.")
    image_size = tuple(int(value) for value in payload["image_size"])
    if len(image_size) != 2 or min(image_size) <= 0:
        raise ValueError(f"{clip.name}: image_size is invalid.")
    return ClipArrays(
        clip=clip,
        payload=payload,
        points=points.cpu(),
        confidence=confidence.cpu(),
        raw_masks=raw_masks.cpu(),
        scores=scores.cpu(),
        depth=depth.cpu(),
        intrinsics=intrinsics.cpu(),
        raw_world_to_camera=raw_w2c.cpu(),
        image_size=(image_size[0], image_size[1]),
        map_size=(height, width),
    )


def _validate_cache_identity(payload: Mapping[str, Any], clip: ClipConfig) -> None:
    required = (
        "clip_name",
        "scene_id",
        "frame_indices",
        "image_size",
        "baseline_world_points",
        "baseline_world_confidence",
        "baseline_depth",
        "baseline_intrinsics",
        "baseline_world_to_camera",
        "tracking_variant_masks_stream",
        "tracking_variant_scores",
        "sam_track_ids",
        "sam_track_prompts",
        "instance_prompts",
        "image_mode",
    )
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError(f"Cache {clip.name!r} lacks fields {missing}.")
    if str(payload["clip_name"]) != clip.name or str(payload["scene_id"]) != clip.scene_id:
        raise ValueError(f"Cache identity differs from configured clip {clip.name!r}.")


def _load_selected_pose(
    arrays: ClipArrays,
    *,
    pose_root: Path | None,
) -> tuple[torch.Tensor | None, Path | None, str]:
    candidates: list[Path] = []
    if pose_root is not None:
        candidates.extend(
            (
                pose_root / "v0" / arrays.clip.name / "poses.pt",
                pose_root / arrays.clip.name / "poses.pt",
            )
        )
    # The cache does not encode the run's v0 output root, so the explicit
    # --pose-root path is preferred.  A pose artifact adjacent to a config is
    # also accepted when a caller supplied a per-clip config.
    baseline_output = arrays.payload.get("baseline_output_dir")
    if baseline_output:
        candidates.append(Path(str(baseline_output)).expanduser() / "poses.pt")
    for path in candidates:
        if not path.is_file():
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping) or "selected_world_to_camera" not in payload:
            return None, path, "malformed_pose_artifact"
        pose = _pose_sequence(_tensor(payload["selected_world_to_camera"], "selected_world_to_camera"))
        if tuple(pose.shape) != tuple(arrays.raw_world_to_camera.shape):
            return None, path, "pose_shape_mismatch"
        return pose, path, "ok"
    return None, None, "not_found"


def _history_config(args: argparse.Namespace) -> HistoricalDepthVetoConfig:
    return HistoricalDepthVetoConfig(
        confidence_threshold=float(args.confidence_threshold),
        track_score_threshold=float(args.track_score_threshold),
        min_history_points=int(args.min_history_points),
        max_history_points=int(args.max_history_points),
        max_points_per_history_frame=int(args.max_points_per_history_frame),
        mad_multiplier=3.0,
        absolute_padding_m=0.05,
    )


def _build_clip_candidates(
    arrays: ClipArrays,
    *,
    pose_sources: Mapping[str, torch.Tensor],
    calibration_fraction: float,
    history_config: HistoricalDepthVetoConfig,
    max_pair_points: int,
    min_affine_samples: int,
    max_affine_samples: int,
    affine_trim_quantile: float,
    affine_max_iterations: int,
) -> ClipCandidates:
    sequence = int(arrays.points.shape[0])
    calibration_count = _calibration_count(sequence, calibration_fraction)
    history_cache = build_causal_history_cache(
        points=arrays.points,
        confidence=arrays.confidence,
        raw_masks=arrays.raw_masks,
        scores=arrays.scores,
        config=history_config,
    )
    pair_data: dict[str, PairCollection] = {}
    affine_fits: dict[str, RobustAffineFit] = {}
    raw_pose = pose_sources["raw_streamvggt"]
    raw_pairs = _build_pair_sources(
        arrays,
        history_cache=history_cache,
        world_to_camera=raw_pose,
        pose_source="raw_streamvggt",
        max_pair_points=max_pair_points,
    )
    for source_kind, pair in raw_pairs.items():
        pair_data[f"raw_streamvggt/{source_kind}"] = pair
        fit = _fit_pair(
            pair,
            calibration_count=calibration_count,
            min_samples=min_affine_samples,
            max_samples=max_affine_samples,
            trim_quantile=affine_trim_quantile,
            max_iterations=affine_max_iterations,
        )
        affine_fits[f"raw_streamvggt/{source_kind}"] = fit
    if "selected_v0" in pose_sources:
        selected_pairs = _build_pair_sources(
            arrays,
            history_cache=history_cache,
            world_to_camera=pose_sources["selected_v0"],
            pose_source="selected_v0",
            max_pair_points=max_pair_points,
        )
        for source_kind, pair in selected_pairs.items():
            pair_data[f"selected_v0/{source_kind}"] = pair
            affine_fits[f"selected_v0/{source_kind}"] = _fit_pair(
                pair,
                calibration_count=calibration_count,
                min_samples=min_affine_samples,
                max_samples=max_affine_samples,
                trim_quantile=affine_trim_quantile,
                max_iterations=affine_max_iterations,
            )

    raw_object_union = arrays.raw_masks.any(dim=1)
    pointmap_fit = affine_fits[f"raw_streamvggt/{POINTMAP_SOURCE}"]
    depth_fit = affine_fits[f"raw_streamvggt/{DEPTH_HEAD_SOURCE}"]
    branches: dict[str, torch.Tensor] = {
        RAW_VARIANT: arrays.points.clone(),
    }
    applied_masks: dict[str, torch.Tensor] = {
        RAW_VARIANT: torch.zeros_like(raw_object_union),
    }
    branches[POINTMAP_AFFINE_VARIANT], applied_masks[POINTMAP_AFFINE_VARIANT] = (
        _reconstruct_branch(
            arrays,
            source_kind=POINTMAP_SOURCE,
            scale=pointmap_fit.scale if pointmap_fit.accepted else 1.0,
            shift=pointmap_fit.shift if pointmap_fit.accepted else 0.0,
            update_frames=range(calibration_count, sequence),
            update_mask=raw_object_union,
        )
    )
    branches[DEPTH_IDENTITY_VARIANT], applied_masks[DEPTH_IDENTITY_VARIANT] = (
        _reconstruct_branch(
            arrays,
            source_kind=DEPTH_HEAD_SOURCE,
            scale=1.0,
            shift=0.0,
            update_frames=range(calibration_count, sequence),
            update_mask=raw_object_union,
        )
    )
    branches[DEPTH_AFFINE_VARIANT], applied_masks[DEPTH_AFFINE_VARIANT] = (
        _reconstruct_branch(
            arrays,
            source_kind=DEPTH_HEAD_SOURCE,
            scale=depth_fit.scale if depth_fit.accepted else 1.0,
            shift=depth_fit.shift if depth_fit.accepted else 0.0,
            update_frames=range(calibration_count, sequence),
            update_mask=raw_object_union,
        )
    )
    branches[DEPTH_AFFINE_ALL_VARIANT], applied_masks[DEPTH_AFFINE_ALL_VARIANT] = (
        _reconstruct_branch(
            arrays,
            source_kind=DEPTH_HEAD_SOURCE,
            scale=depth_fit.scale if depth_fit.accepted else 1.0,
            shift=depth_fit.shift if depth_fit.accepted else 0.0,
            update_frames=range(sequence),
            update_mask=raw_object_union,
        )
    )
    metadata = {
        "calibration_count": int(calibration_count),
        "holdout_count": int(sequence - calibration_count),
        "history_observation_count": int(
            sum(
                int(value.shape[0])
                for frame in history_cache
                for value in frame
            )
        ),
        "branch_applied_pixels": {
            key: int(value.sum()) for key, value in applied_masks.items()
        },
        "branch_update_frames": {
            POINTMAP_AFFINE_VARIANT: "holdout_suffix",
            DEPTH_IDENTITY_VARIANT: "holdout_suffix",
            DEPTH_AFFINE_VARIANT: "holdout_suffix",
            DEPTH_AFFINE_ALL_VARIANT: "all_frames",
        },
        "candidate_generation_gt_fields": 0,
    }
    return ClipCandidates(
        branches=branches,
        applied_masks=applied_masks,
        pair_data=pair_data,
        affine_fits=affine_fits,
        calibration_count=calibration_count,
        metadata=metadata,
    )


def _build_pair_sources(
    arrays: ClipArrays,
    *,
    history_cache: Sequence[Sequence[torch.Tensor]],
    world_to_camera: torch.Tensor,
    pose_source: str,
    max_pair_points: int,
) -> dict[str, PairCollection]:
    depth_size = tuple(int(value) for value in arrays.depth.shape[-2:])
    map_size = arrays.map_size
    masks_depth = _resize_masks(arrays.raw_masks, depth_size)
    masks_map = arrays.raw_masks
    source_values: dict[str, list[torch.Tensor]] = {
        DEPTH_HEAD_SOURCE: [],
        POINTMAP_SOURCE: [],
    }
    target_values: dict[str, list[torch.Tensor]] = {
        DEPTH_HEAD_SOURCE: [],
        POINTMAP_SOURCE: [],
    }
    frame_values: dict[str, list[torch.Tensor]] = {
        DEPTH_HEAD_SOURCE: [],
        POINTMAP_SOURCE: [],
    }
    rows: dict[str, list[dict[str, object]]] = {
        DEPTH_HEAD_SOURCE: [],
        POINTMAP_SOURCE: [],
    }
    for frame in range(int(arrays.points.shape[0])):
        depth_intrinsics = resize_intrinsics(
            arrays.intrinsics[frame], arrays.image_size, depth_size
        )
        map_intrinsics = resize_intrinsics(
            arrays.intrinsics[frame], arrays.image_size, map_size
        )
        current_map_camera = transform_world_points(
            arrays.points[frame], world_to_camera[frame]
        )
        current_map_depth = current_map_camera[..., 2]
        for slot in range(int(arrays.raw_masks.shape[1])):
            history = history_cache[frame][slot]
            for source_kind, target_size, matrix, current_mask, current_source in (
                (
                    DEPTH_HEAD_SOURCE,
                    depth_size,
                    depth_intrinsics,
                    masks_depth[frame, slot],
                    arrays.depth[frame],
                ),
                (
                    POINTMAP_SOURCE,
                    map_size,
                    map_intrinsics,
                    masks_map[frame, slot],
                    current_map_depth,
                ),
            ):
                source = target = torch.empty(0, dtype=torch.float32)
                history_count = int(history.shape[0])
                if history_count:
                    projected = project_world_points(
                        history,
                        world_to_camera[frame],
                        matrix,
                        target_size,
                    )
                    sampled, sampled_valid = sample_depth_nearest(
                        current_source,
                        projected.uv,
                    )
                    mask_hit, mask_valid = _sample_bool_mask(
                        current_mask,
                        projected.uv,
                    )
                    valid = (
                        projected.valid_mask
                        & sampled_valid
                        & mask_valid
                        & mask_hit
                        & torch.isfinite(projected.depth)
                        & (projected.depth > 1e-6)
                    )
                    if bool(valid.any()):
                        source = sampled[valid].float().cpu()
                        target = projected.depth[valid].float().cpu()
                        if source.numel() > int(max_pair_points):
                            positions = _deterministic_positions(
                                source.numel(), int(max_pair_points)
                            )
                            source = source.index_select(0, positions)
                            target = target.index_select(0, positions)
                if source.numel():
                    source_values[source_kind].append(source)
                    target_values[source_kind].append(target)
                    frame_values[source_kind].append(
                        torch.full((source.numel(),), frame, dtype=torch.long)
                    )
                rows[source_kind].append(
                    {
                        "clip": arrays.clip.name,
                        "scene_id": arrays.clip.scene_id,
                        "split": arrays.clip.split,
                        "pose_source": pose_source,
                        "source_kind": source_kind,
                        "sequence_index": int(frame),
                        "frame_index": int(arrays.payload["frame_indices"][frame]),
                        "slot": int(slot),
                        "sam_track_id": int(arrays.payload["sam_track_ids"][slot]),
                        "history_point_count": history_count,
                        "pair_count": int(source.numel()),
                        "source_mean": _mean_tensor(source),
                        "target_mean": _mean_tensor(target),
                        "gt_fields": 0,
                    }
                )
    output: dict[str, PairCollection] = {}
    for source_kind in PAIR_SOURCES:
        output[source_kind] = PairCollection(
            source=_cat_or_empty(source_values[source_kind]),
            target=_cat_or_empty(target_values[source_kind]),
            frame=_cat_or_empty(frame_values[source_kind]).long(),
            rows=tuple(rows[source_kind]),
        )
    return output


def _fit_pair(
    pair: PairCollection,
    *,
    calibration_count: int,
    min_samples: int,
    max_samples: int,
    trim_quantile: float,
    max_iterations: int,
) -> RobustAffineFit:
    if not pair.source.numel():
        return fit_robust_affine(
            torch.empty(0),
            torch.empty(0),
            min_samples=min_samples,
            max_samples=max_samples,
            trim_quantile=trim_quantile,
            max_iterations=max_iterations,
        )
    calibration = pair.frame < int(calibration_count)
    return fit_robust_affine(
        pair.source[calibration],
        pair.target[calibration],
        min_samples=min_samples,
        max_samples=max_samples,
        trim_quantile=trim_quantile,
        max_iterations=max_iterations,
    )


def _reconstruct_branch(
    arrays: ClipArrays,
    *,
    source_kind: str,
    scale: float,
    shift: float,
    update_frames: Sequence[int] | range,
    update_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = arrays.points.clone()
    applied = torch.zeros_like(update_mask)
    map_size = arrays.map_size
    map_intrinsics = torch.stack(
        [
            resize_intrinsics(arrays.intrinsics[frame], arrays.image_size, map_size)
            for frame in range(int(arrays.points.shape[0]))
        ],
        dim=0,
    )
    depth_on_map = _resize_scalar_sequence(arrays.depth, map_size)
    for frame in update_frames:
        if source_kind == POINTMAP_SOURCE:
            current_camera = transform_world_points(
                arrays.points[frame], arrays.raw_world_to_camera[frame]
            )
            source_depth = current_camera[..., 2]
        elif source_kind == DEPTH_HEAD_SOURCE:
            source_depth = depth_on_map[frame]
        else:
            raise ValueError(f"Unknown reconstruction source_kind={source_kind!r}.")
        corrected, current_applied = apply_ray_depth_affine(
            arrays.points[frame],
            source_depth,
            map_intrinsics[frame],
            arrays.raw_world_to_camera[frame],
            scale=float(scale),
            shift=float(shift),
            update_mask=update_mask[frame],
        )
        output[frame] = corrected
        applied[frame] = current_applied
    return output, applied


def _scale_shift_rows(
    arrays_list: Sequence[ClipArrays],
    candidates_list: Sequence[ClipCandidates],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for arrays, candidates in zip(arrays_list, candidates_list):
        sequence = int(arrays.points.shape[0])
        calibration = candidates.calibration_count
        for key, pair in candidates.pair_data.items():
            pose_source, source_kind = key.split("/", 1)
            fit = candidates.affine_fits[key]
            base = {
                "clip": arrays.clip.name,
                "scene_id": arrays.clip.scene_id,
                "split_label": arrays.clip.split,
                "pose_source": pose_source,
                "source_kind": source_kind,
                "calibration_frame_count": calibration,
                "holdout_frame_count": sequence - calibration,
                "fit_scale": float(fit.scale),
                "fit_shift": float(fit.shift),
                "fit_status": fit.status,
                "fit_accepted": int(fit.accepted),
                "fit_sample_count": int(fit.sample_count),
                "fit_inlier_count": int(fit.inlier_count),
                "gt_fields": 0,
            }
            calibration_mask = pair.frame < calibration
            holdout_mask = ~calibration_mask
            for split_name, selected in (
                ("calibration_prefix", calibration_mask),
                ("holdout_suffix", holdout_mask),
            ):
                current = pair.source[selected]
                target = pair.target[selected]
                row = dict(base)
                row.update(
                    {
                        "split": split_name,
                        "sample_count": int(current.numel()),
                    }
                )
                if current.numel():
                    before = affine_error_metrics(
                        current, target, scale=1.0, shift=0.0
                    )
                    after = (
                        affine_error_metrics(
                            current,
                            target,
                            scale=fit.scale,
                            shift=fit.shift,
                        )
                        if fit.accepted
                        else before
                    )
                    row.update(
                        {
                            "rmse_before": before["rmse"],
                            "rmse_after": after["rmse"],
                            "median_before": before["median"],
                            "median_after": after["median"],
                            "p90_before": before["p90"],
                            "p90_after": after["p90"],
                            "holdout_evaluated": int(split_name == "holdout_suffix"),
                        }
                    )
                else:
                    row.update(_nan_metric_fields())
                    row["holdout_evaluated"] = int(split_name == "holdout_suffix")
                rows.append(row)
    return rows


def _evaluate_clip(
    arrays: ClipArrays,
    candidates: ClipCandidates,
    *,
    manifest: Path,
    evaluation_run: Any,
    confidence_threshold: float,
) -> dict[str, list[dict[str, object]]]:
    payload = arrays.payload
    frames = tuple(int(value) for value in payload["frame_indices"])
    ground_truth = load_ground_truth_instances(
        manifest,
        scene_id=str(payload["scene_id"]),
        frame_indices=frames,
        output_size=arrays.map_size,
        prompts=tuple(str(value) for value in payload["instance_prompts"]),
        prompt_label_aliases=evaluation_run.prompt_label_aliases,
    )
    gt_masks = load_ground_truth_stream_masks(
        manifest,
        scene_id=str(payload["scene_id"]),
        frame_indices=frames,
        instance_ids=ground_truth.instance_ids,
        processed_size=arrays.map_size,
        image_mode=str(payload["image_mode"]),
    )
    gt = GroundTruthInstances(
        masks=gt_masks,
        instance_ids=ground_truth.instance_ids,
        labels=ground_truth.labels,
        all_visible_instance_ids=ground_truth.all_visible_instance_ids,
    )
    raw_assignment = evaluate_tracking_variants(
        scene_id=str(payload["scene_id"]),
        clip_name=str(payload["clip_name"]),
        frame_indices=frames,
        variant_masks={RAW_VARIANT: arrays.raw_masks},
        variant_scores={RAW_VARIANT: arrays.scores},
        raw_variant=RAW_VARIANT,
        track_ids=tuple(int(value) for value in payload["sam_track_ids"]),
        track_prompts=tuple(str(value) for value in payload["sam_track_prompts"]),
        ground_truth=gt,
        config=evaluation_run.tracking,
        prompt_label_aliases=evaluation_run.prompt_label_aliases,
    )
    tracking = [dict(row) for row in raw_assignment["summary_rows"]]
    for row in tracking:
        row["clip"] = arrays.clip.name
        row["scene_id"] = arrays.clip.scene_id
        row["gt_fields"] = 1

    target_points = _tensor(payload["target_world_points"], "target_world_points").float()
    if tuple(target_points.shape) != tuple(arrays.points.shape):
        raise ValueError(
            f"{arrays.clip.name}: target_world_points shape {tuple(target_points.shape)} "
            f"does not match raw pointmap {tuple(arrays.points.shape)}."
        )
    target_pose = _pose_sequence(
        _tensor(payload["target_world_to_camera"], "target_world_to_camera")
    )
    if tuple(target_pose.shape) != tuple(arrays.raw_world_to_camera.shape):
        raise ValueError(
            f"{arrays.clip.name}: target pose shape {tuple(target_pose.shape)} "
            f"does not match frame count {tuple(arrays.raw_world_to_camera.shape)}."
        )
    alignment = {
        "scale": float(payload["point_alignment_scale"]),
        "rotation": _tensor(payload["point_alignment_rotation"], "point_alignment_rotation"),
        "translation": _tensor(payload["point_alignment_translation"], "point_alignment_translation"),
    }
    target_depth = _target_depth_for_evaluation(
        payload,
        target_points=target_points,
        target_pose=target_pose,
        map_size=arrays.map_size,
    )
    if tuple(target_depth.shape) != tuple(arrays.points.shape[:3]):
        raise ValueError(
            f"{arrays.clip.name}: target depth shape {tuple(target_depth.shape)} "
            f"does not match pointmap grid {tuple(arrays.points.shape[:3])}."
        )
    raw_union = arrays.raw_masks.any(dim=1)
    map_rows: list[dict[str, object]] = []
    object_rows: list[dict[str, object]] = []
    depth_rows: list[dict[str, object]] = []
    per_clip_rows: list[dict[str, object]] = []
    split_ranges = {
        "all": range(int(arrays.points.shape[0])),
        "calibration_prefix": range(candidates.calibration_count),
        "holdout_suffix": range(candidates.calibration_count, int(arrays.points.shape[0])),
    }
    for variant in GEOMETRY_VARIANTS:
        branch_points = candidates.branches[variant]
        aligned = apply_similarity(
            branch_points,
            scale=alignment["scale"],
            rotation=alignment["rotation"],
            translation=alignment["translation"],
        )
        for split_name, frame_range in split_ranges.items():
            indices = torch.tensor(list(frame_range), dtype=torch.long)
            map_result = evaluate_semantic_object_map(
                scene_id=str(payload["scene_id"]),
                clip_name=str(payload["clip_name"]),
                variant=variant,
                map_policy=f"{variant}_{split_name}",
                aligned_world_points=aligned.index_select(0, indices),
                target_world_points=target_points.index_select(0, indices),
                confidence=arrays.confidence.index_select(0, indices),
                predicted_masks=arrays.raw_masks.index_select(0, indices),
                track_scores=arrays.scores.index_select(0, indices),
                gt_masks=gt.masks.index_select(0, indices),
                gt_instance_ids=gt.instance_ids,
                gt_labels=gt.labels,
                assignments=raw_assignment["assignments"],
                config=evaluation_run.map_metrics,
            )
            map_summary = dict(map_result["summary"])
            map_summary.update(
                {
                    "scene_id": arrays.clip.scene_id,
                    "clip": arrays.clip.name,
                    "split": split_name,
                    "gt_fields": 1,
                }
            )
            map_rows.append(map_summary)
            for row in map_result["object_rows"]:
                current = dict(row)
                current.update({"split": split_name, "gt_fields": 1})
                object_rows.append(current)
            depth_rows.extend(
                _depth_metric_rows(
                    arrays,
                    variant=variant,
                    branch_points=branch_points,
                    target_depth=target_depth,
                    target_pose=target_pose,
                    alignment=alignment,
                    raw_union=raw_union,
                    split=split_name,
                    frame_indices=indices,
                    confidence_threshold=confidence_threshold,
                )
            )
            per_clip_rows.append(
                {
                    "scene_id": arrays.clip.scene_id,
                    "clip": arrays.clip.name,
                    "split_label": arrays.clip.split,
                    "variant": variant,
                    "split": split_name,
                    "tracking_variant": RAW_VARIANT,
                    "tracking_mean_frame_iou": tracking[0].get("mean_frame_iou")
                    if tracking
                    else float("nan"),
                    "tracking_frame_idf1": tracking[0].get("frame_idf1")
                    if tracking
                    else float("nan"),
                    **map_summary,
                    "gt_fields": 1,
                }
            )
    return {
        "tracking_rows": tracking,
        "map_rows": map_rows,
        "map_object_rows": object_rows,
        "depth_rows": depth_rows,
        "per_clip_rows": per_clip_rows,
    }


def _target_depth_for_evaluation(
    payload: Mapping[str, Any],
    *,
    target_points: torch.Tensor,
    target_pose: torch.Tensor,
    map_size: tuple[int, int],
) -> torch.Tensor:
    if "target_depth" in payload:
        target = _scalar_map(payload["target_depth"], "target_depth")
        if target.shape[-2:] != map_size:
            target = _resize_scalar_sequence(target, map_size)
        return target
    target_camera = transform_world_points(target_points, target_pose)
    return target_camera[..., 2]


def _depth_metric_rows(
    arrays: ClipArrays,
    *,
    variant: str,
    branch_points: torch.Tensor,
    target_depth: torch.Tensor,
    target_pose: torch.Tensor,
    alignment: Mapping[str, object],
    raw_union: torch.Tensor,
    split: str,
    frame_indices: torch.Tensor,
    confidence_threshold: float,
) -> list[dict[str, object]]:
    aligned = apply_similarity(
        branch_points,
        scale=float(alignment["scale"]),
        rotation=alignment["rotation"],
        translation=alignment["translation"],
    )
    metric_camera = transform_world_points(aligned, target_pose)
    source_depth = metric_camera[..., 2]
    target = target_depth
    if target.shape[0] != source_depth.shape[0]:
        raise ValueError("Target depth and branch geometry disagree on sequence length.")
    base_valid = (
        torch.isfinite(source_depth)
        & (source_depth > 1e-6)
        & torch.isfinite(target)
        & (target > 1e-6)
        & torch.isfinite(arrays.confidence)
        & (arrays.confidence >= float(confidence_threshold))
    )
    selected = torch.zeros_like(base_valid)
    selected[frame_indices] = True
    output: list[dict[str, object]] = []
    for scope, scope_mask in (
        ("all_confident", torch.ones_like(raw_union)),
        ("raw_object_union", raw_union),
    ):
        valid = base_valid & selected & scope_mask
        residual = (source_depth - target).abs()[valid]
        row: dict[str, object] = {
            "scene_id": arrays.clip.scene_id,
            "clip": arrays.clip.name,
            "split_label": arrays.clip.split,
            "variant": variant,
            "split": split,
            "scope": scope,
            "count": int(residual.numel()),
            "gt_fields": 1,
        }
        if residual.numel():
            row.update(
                {
                    "rmse_m": float(torch.sqrt(residual.square().mean())),
                    "median_m": float(torch.median(residual)),
                    "p90_m": float(torch.quantile(residual, 0.90)),
                    "mean_m": float(residual.mean()),
                }
            )
        else:
            row.update({"rmse_m": float("nan"), "median_m": float("nan"), "p90_m": float("nan"), "mean_m": float("nan")})
        output.append(row)
    return output


def _aggregate_map_rows(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    output_rows: list[dict[str, object]] = []
    metrics = (
        "voxel_iou_5cm",
        "fscore_5cm",
        "ghost_point_ratio",
        "object_accuracy_m",
        "object_completeness_m",
    )
    keys = sorted({(str(row["variant"]), str(row["split"])) for row in rows})
    for variant, split in keys:
        selected = [row for row in rows if str(row["variant"]) == variant and str(row["split"]) == split]
        result: dict[str, object] = {"variant": variant, "split": split, "clip_count": len(selected)}
        for metric in metrics:
            values = [_finite_float(row.get(metric)) for row in selected]
            values = [value for value in values if value is not None]
            result[f"{metric}_mean"] = _mean(values)
            result[f"{metric}_std"] = _std(values)
        output_rows.append(result)
    return {"rows": output_rows, "variant_count": len({row["variant"] for row in output_rows})}


def _affine_evidence_gate(
    scale_rows: Sequence[Mapping[str, object]],
    arrays_list: Sequence[ClipArrays],
) -> dict[str, object]:
    clips = {arrays.clip.name for arrays in arrays_list}
    eligible: dict[str, dict[str, object]] = {}
    for clip in clips:
        rows = [
            row
            for row in scale_rows
            if str(row.get("clip")) == clip
            and str(row.get("pose_source")) == "raw_streamvggt"
            and str(row.get("source_kind")) == DEPTH_HEAD_SOURCE
            and str(row.get("split")) == "holdout_suffix"
        ]
        if not rows:
            eligible[clip] = {"pass": False, "reason": "missing_holdout_row"}
            continue
        row = rows[0]
        before_median = _finite_float(row.get("median_before"))
        after_median = _finite_float(row.get("median_after"))
        before_p90 = _finite_float(row.get("p90_before"))
        after_p90 = _finite_float(row.get("p90_after"))
        scale = _finite_float(row.get("fit_scale"))
        passed = (
            int(row.get("fit_accepted", 0)) == 1
            and scale is not None
            and scale > 0.0
            and before_median is not None
            and after_median is not None
            and before_p90 is not None
            and after_p90 is not None
            and after_median < before_median
            and after_p90 < before_p90
        )
        eligible[clip] = {
            "pass": int(passed),
            "fit_scale": scale,
            "median_before": before_median,
            "median_after": after_median,
            "p90_before": before_p90,
            "p90_after": after_p90,
        }
    passed_count = sum(int(value.get("pass", 0)) for value in eligible.values())
    required = max(1, math.ceil(0.75 * len(clips)))
    return {
        "pass": int(passed_count >= required),
        "passed_clip_count": passed_count,
        "required_clip_count": required,
        "clip_results": eligible,
        "criterion": "holdout median and P90 both improve for depth-head self-affine pairs; scale positive",
    }


def _geometry_gate(
    map_rows: Sequence[Mapping[str, object]],
    arrays_list: Sequence[ClipArrays],
) -> dict[str, object]:
    primary = DEPTH_AFFINE_VARIANT
    clips = [arrays.clip.name for arrays in arrays_list]
    raw_rows = {
        (str(row["clip"]), str(row["split"])): row
        for row in map_rows
        if str(row["variant"]) == RAW_VARIANT
    }
    candidate_rows = {
        (str(row["clip"]), str(row["split"])): row
        for row in map_rows
        if str(row["variant"]) == primary
    }
    per_clip: dict[str, dict[str, object]] = {}
    for clip in clips:
        raw = raw_rows.get((clip, "holdout_suffix"))
        candidate = candidate_rows.get((clip, "holdout_suffix"))
        if raw is None or candidate is None:
            per_clip[clip] = {"pass": False, "reason": "missing_holdout_map_row"}
            continue
        voxel = _finite_float(candidate.get("voxel_iou_5cm"))
        raw_voxel = _finite_float(raw.get("voxel_iou_5cm"))
        fscore = _finite_float(candidate.get("fscore_5cm"))
        raw_fscore = _finite_float(raw.get("fscore_5cm"))
        completeness = _finite_float(candidate.get("object_completeness_m"))
        raw_completeness = _finite_float(raw.get("object_completeness_m"))
        ghost = _finite_float(candidate.get("ghost_point_ratio"))
        raw_ghost = _finite_float(raw.get("ghost_point_ratio"))
        passed = (
            voxel is not None
            and raw_voxel is not None
            and fscore is not None
            and raw_fscore is not None
            and completeness is not None
            and raw_completeness is not None
            and ghost is not None
            and raw_ghost is not None
            and voxel >= raw_voxel - 1e-6
            and fscore >= raw_fscore - 1e-6
            and completeness <= raw_completeness * 1.05 + 1e-6
            and ghost <= raw_ghost + 0.05 + 1e-6
        )
        per_clip[clip] = {
            "pass": int(passed),
            "delta_voxel_iou_5cm": _difference(voxel, raw_voxel),
            "delta_fscore_5cm": _difference(fscore, raw_fscore),
            "delta_completeness_m": _difference(completeness, raw_completeness),
            "delta_ghost_point_ratio": _difference(ghost, raw_ghost),
        }
    passed_count = sum(int(value.get("pass", 0)) for value in per_clip.values())
    required = max(1, math.ceil(0.75 * len(clips)))
    aggregate_raw = _aggregate_one(map_rows, RAW_VARIANT, "holdout_suffix")
    aggregate_candidate = _aggregate_one(map_rows, primary, "holdout_suffix")
    aggregate_pass = (
        _ge(aggregate_candidate.get("voxel_iou_5cm"), aggregate_raw.get("voxel_iou_5cm"))
        and _ge(aggregate_candidate.get("fscore_5cm"), aggregate_raw.get("fscore_5cm"))
        and _le(
            aggregate_candidate.get("object_completeness_m"),
            _scale_metric(aggregate_raw.get("object_completeness_m"), 1.05),
        )
        and _le(
            aggregate_candidate.get("ghost_point_ratio"),
            _sum_metric(aggregate_raw.get("ghost_point_ratio"), 0.05),
        )
    )
    return {
        "pass": int(passed_count >= required and aggregate_pass),
        "passed_clip_count": passed_count,
        "required_clip_count": required,
        "aggregate_pass": int(aggregate_pass),
        "primary_variant": primary,
        "clip_results": per_clip,
        "criterion": "holdout voxelIoU/F5cm non-regression, completeness error <=+5%, ghost <=+5pp",
    }


def _aggregate_one(rows: Sequence[Mapping[str, object]], variant: str, split: str) -> dict[str, object]:
    selected = [row for row in rows if str(row.get("variant")) == variant and str(row.get("split")) == split]
    output: dict[str, object] = {}
    for key in ("voxel_iou_5cm", "fscore_5cm", "object_completeness_m", "ghost_point_ratio"):
        values = [_finite_float(row.get(key)) for row in selected]
        values = [value for value in values if value is not None]
        output[key] = _mean(values)
    return output


def _save_candidate_artifact(
    output_dir: Path,
    arrays: ClipArrays,
    candidates: ClipCandidates,
) -> None:
    path = output_dir / f"candidate_geometry_{arrays.clip.name}.pt"
    torch.save(
        {
            "schema": 1,
            "revision": REVISION,
            "clip": arrays.clip.name,
            "scene_id": arrays.clip.scene_id,
            "frame_indices": tuple(int(value) for value in arrays.payload["frame_indices"]),
            "candidate_generation_gt_fields": 0,
            "branches": {key: value.float() for key, value in candidates.branches.items()},
            "applied_masks": {key: value.bool() for key, value in candidates.applied_masks.items()},
            "affine_fits": {key: fit.to_dict() for key, fit in candidates.affine_fits.items()},
            "metadata": _json_safe(dict(candidates.metadata)),
        },
        path,
    )


def _print_summary(
    arrays_list: Sequence[ClipArrays],
    candidates_list: Sequence[ClipCandidates],
    scale_rows: Sequence[Mapping[str, object]],
    map_rows: Sequence[Mapping[str, object]],
    depth_rows: Sequence[Mapping[str, object]],
    affine_gate: Mapping[str, object],
    geometry_gate: Mapping[str, object],
    gate: str,
) -> None:
    print("===== MULTI-CLIP AFFINE GEOMETRY SUMMARY =====")
    print(
        "clips="
        f"{len(arrays_list)} total_frames={sum(int(item.points.shape[0]) for item in arrays_list)} "
        "calibration/holdout="
        + ",".join(
            f"{item.clip.name}:{candidate.calibration_count}/{int(item.points.shape[0]) - candidate.calibration_count}"
            for item, candidate in zip(arrays_list, candidates_list)
        )
    )
    print("affine_holdout_raw_pose")
    for row in scale_rows:
        if (
            str(row.get("pose_source")) == "raw_streamvggt"
            and str(row.get("split")) == "holdout_suffix"
        ):
            print(
                f"  clip={row.get('clip')} source={row.get('source_kind')} "
                f"samples={row.get('sample_count', 0)} accepted={row.get('fit_accepted', 0)} "
                f"scale={_display(row.get('fit_scale'))} "
                f"median={_display(row.get('median_before'))}->{_display(row.get('median_after'))} "
                f"p90={_display(row.get('p90_before'))}->{_display(row.get('p90_after'))}"
            )
    print("map_holdout")
    for row in map_rows:
        if str(row.get("split")) != "holdout_suffix":
            continue
        print(
            f"  clip={row.get('clip')} variant={row.get('variant')} "
            f"voxelIoU5cm={_display(row.get('voxel_iou_5cm'))} "
            f"F5cm={_display(row.get('fscore_5cm'))} "
            f"completeness_m={_display(row.get('object_completeness_m'))} "
            f"ghost={_display(row.get('ghost_point_ratio'))}"
        )
    print("depth_holdout_object_union")
    for row in depth_rows:
        if (
            str(row.get("split")) == "holdout_suffix"
            and str(row.get("scope")) == "raw_object_union"
            and str(row.get("variant")) in {RAW_VARIANT, DEPTH_AFFINE_VARIANT}
        ):
            print(
                f"  clip={row.get('clip')} variant={row.get('variant')} "
                f"count={row.get('count', 0)} "
                f"rmse={_display(row.get('rmse_m'))} "
                f"median={_display(row.get('median_m'))} "
                f"p90={_display(row.get('p90_m'))}"
            )
    print(
        f"affine_evidence_gate={affine_gate.get('pass')} "
        f"passed={affine_gate.get('passed_clip_count')}/{affine_gate.get('required_clip_count')}"
    )
    print(
        f"geometry_gate={geometry_gate.get('pass')} "
        f"passed={geometry_gate.get('passed_clip_count')}/{geometry_gate.get('required_clip_count')} "
        f"aggregate_pass={geometry_gate.get('aggregate_pass')}"
    )
    print(f"gate={gate}; no affine correction is promoted automatically")
    print("depth_source_metrics=GT evaluation only; candidate generation remained GT-free")


def _write_copyable(
    path: Path,
    summary: Mapping[str, object],
    aggregate: Mapping[str, object],
    affine_gate: Mapping[str, object],
    geometry_gate: Mapping[str, object],
) -> None:
    lines = [
        "===== MULTICLIP_AFFINE_GEOMETRY_BEGIN =====",
        f"revision={summary['revision']}",
        f"decision={summary['decision']}",
        f"gate={summary['gate']}",
        f"clip_count={summary['clip_count']}",
        f"scene_count={summary['scene_count']}",
        f"total_frame_count={summary['total_frame_count']}",
        "candidate_generation_gt_fields=0",
        "evaluation_gt_fields=1",
        "streamvggt_rerun=0",
        "sam_rerun=0",
        "pose_modified=0",
        "pointmap_modified=0",
        "affine_evidence_gate=" + json.dumps(_json_safe(dict(affine_gate)), sort_keys=True),
        "geometry_gate=" + json.dumps(_json_safe(dict(geometry_gate)), sort_keys=True),
        "aggregate_map_metrics=" + json.dumps(_json_safe(dict(aggregate)), sort_keys=True),
        "interpretation=" + json.dumps(_json_safe(dict(summary["interpretation"])), sort_keys=True),
        "===== MULTICLIP_AFFINE_GEOMETRY_END =====",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def _sample_bool_mask(mask: torch.Tensor, uv: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    values = mask.detach().cpu().bool()
    points = uv.detach().float().cpu()
    height, width = values.shape
    finite = torch.isfinite(points).all(dim=1)
    x = torch.round(points[:, 0]).long()
    y = torch.round(points[:, 1]).long()
    valid = finite & (x >= 0) & (x < width) & (y >= 0) & (y < height)
    safe_x = x.clamp(0, width - 1)
    safe_y = y.clamp(0, height - 1)
    return values[safe_y, safe_x], valid


def _reconstruct_depth_smoke() -> dict[str, object]:
    """Small no-data smoke used by the test suite and command wrapper."""

    height, width = 4, 5
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    intrinsics = torch.tensor(
        [[10.0, 0.0, 2.0], [0.0, 10.0, 1.5], [0.0, 0.0, 1.0]]
    )
    depth = torch.full((height, width), 2.0)
    rays = torch.stack(((xx - 2.0) / 10.0, (yy - 1.5) / 10.0, torch.ones_like(xx)), dim=-1)
    points = rays * depth[..., None]
    pose = torch.eye(4, dtype=torch.float32)[:3]
    source_depth = (depth - 0.20) / 1.25
    update = torch.ones(height, width, dtype=torch.bool)
    corrected, applied = apply_ray_depth_affine(
        points,
        source_depth,
        intrinsics,
        pose,
        scale=1.25,
        shift=0.20,
        update_mask=update,
    )
    passed = bool(applied.all()) and bool(torch.allclose(corrected, points, atol=1e-5))
    if not passed:
        raise AssertionError("Ray-depth affine smoke failed.")
    return {"passed": 1, "applied_pixels": int(applied.sum())}


def _calibration_count(sequence: int, fraction: float) -> int:
    if int(sequence) < 2:
        raise ValueError("Each clip needs at least two frames for a holdout.")
    return max(1, min(int(sequence) - 1, int(round(int(sequence) * float(fraction)))))


def _resize_masks(masks: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    values = masks.detach().cpu().bool()
    if tuple(values.shape[-2:]) == tuple(size):
        return values
    if not values.numel():
        return torch.zeros(*values.shape[:-2], size[0], size[1], dtype=torch.bool)
    flat = values.float().reshape(-1, 1, values.shape[-2], values.shape[-1])
    output = F.interpolate(flat, size=size, mode="nearest")
    return output.reshape(*values.shape[:-2], size[0], size[1]).bool()


def _resize_scalar_sequence(value: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    tensor = _scalar_map(value, "scalar_sequence")
    if tuple(tensor.shape[-2:]) == tuple(size):
        return tensor
    return F.interpolate(
        tensor[:, None], size=size, mode="bilinear", align_corners=False
    )[:, 0]


def _scalar_map(value: object, name: str) -> torch.Tensor:
    tensor = _tensor(value, name).float()
    if tensor.ndim == 4 and tensor.shape[-1] == 1:
        tensor = tensor[..., 0]
    elif tensor.ndim == 4 and tensor.shape[1] == 1:
        tensor = tensor[:, 0]
    if tensor.ndim != 3:
        raise ValueError(f"{name} must be [S,H,W] or singleton-channel map.")
    return tensor


def _pose_sequence(value: torch.Tensor) -> torch.Tensor:
    pose = value.detach().float().cpu()
    if pose.ndim == 4 and pose.shape[0] == 1:
        pose = pose[0]
    if pose.ndim == 3 and tuple(pose.shape[-2:]) == (4, 4):
        pose = pose[:, :3]
    if pose.ndim != 3 or tuple(pose.shape[-2:]) != (3, 4):
        raise ValueError(f"Pose sequence must be [S,3,4] or [S,4,4], got {tuple(pose.shape)}.")
    return pose


def _tensor(value: object, name: str) -> torch.Tensor:
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    return value.detach().cpu()


def _cat_or_empty(values: Sequence[torch.Tensor]) -> torch.Tensor:
    if not values:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat(values).float().cpu()


def _deterministic_positions(count: int, limit: int) -> torch.Tensor:
    if count <= int(limit):
        return torch.arange(count, dtype=torch.long)
    return torch.linspace(0, count - 1, steps=int(limit), dtype=torch.float64).round().long()


def _mean_tensor(value: torch.Tensor) -> float:
    return float(value.float().mean()) if value.numel() else float("nan")


def _nan_metric_fields() -> dict[str, float]:
    return {
        "rmse_before": float("nan"),
        "rmse_after": float("nan"),
        "median_before": float("nan"),
        "median_after": float("nan"),
        "p90_before": float("nan"),
        "p90_after": float("nan"),
    }


def _finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _std(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _difference(first: float | None, second: float | None) -> float:
    if first is None or second is None:
        return float("nan")
    return float(first) - float(second)


def _scale_metric(value: object, multiplier: float) -> float:
    numeric = _finite_float(value)
    return float("nan") if numeric is None else numeric * float(multiplier)


def _sum_metric(value: object, increment: float) -> float:
    numeric = _finite_float(value)
    return float("nan") if numeric is None else numeric + float(increment)


def _ge(candidate: object, baseline: object, epsilon: float = 1e-6) -> bool:
    left, right = _finite_float(candidate), _finite_float(baseline)
    return left is not None and right is not None and left >= right - epsilon


def _le(candidate: object, baseline: object, epsilon: float = 1e-6) -> bool:
    left, right = _finite_float(candidate), _finite_float(baseline)
    return left is not None and right is not None and left <= right + epsilon


def _display(value: object) -> str:
    number = _finite_float(value)
    return "nan" if number is None else f"{number:.6f}"


def _json_safe(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True) + "\n",
        encoding="utf8",
    )


def _csv_value(value: object) -> object:
    if torch.is_tensor(value):
        return value.item() if value.ndim == 0 else json.dumps(_json_safe(value))
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(_json_safe(value), sort_keys=True)
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if str(key) not in fields:
                fields.append(str(key))
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fields})


if __name__ == "__main__":
    main()
