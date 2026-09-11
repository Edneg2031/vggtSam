#!/usr/bin/env python3
"""Stage-2b: SAM-object-consensus camera-pose feedback analysis (CPU only).

This script consumes three stage-2a artifacts:

* ``<run-dir>/object_pose_refinement/feedback_diagnostics.pt``  (per-instance
  observations, pairing snapshots, and accepted/rejected 6DoF proposals);
* the HorizonStream geometry cache (raw poses, depth provenance, frame ids);
* the chunk camera-map aux artifact written by
  ``generate_horizonstream_geometry_cache --save-chunk-cam-maps``.

It then, entirely on CPU:

1. replays the causal ``online_motion_averaging`` accumulator once without
   injections and asserts equivalence with the cached raw trajectory
   (branch A);
2. scores every per-object proposal with semantic/geometric reliability,
   runs the five consensus ablation variants, and gates each frame;
3. replays the accumulator once per variant with the accepted absolute
   target poses injected through the verified GT-feedback primitive
   (``_replace_internal_pose_for_public_target``);
4. only after all feedback decisions are frozen, loads GT poses and evaluates
   branches, proposals, consensus, future gains, and writes the GO/NO-GO
   decision.

Ground truth never enters reliability, consensus, or gating.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import fields as dataclass_fields
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from streaming_couping.scripts.run_scannet_horizonstream_gt_feedback_poc import (
    _homogeneous,
    _load_gt_c2w,
    _relative_to_first,
    _rotation_error_deg,
    _trajectory_metrics,
    _write_csv,
)
from streaming_couping.src.horizonstream_cache import load_horizonstream_cache
from streaming_couping.src.semantic_mapping.object_pose_feedback import (
    CONSENSUS_VARIANTS,
    MAIN_VARIANT_NAME,
    ObjectPoseFeedbackConfig,
    RPE_EXCLUDED_BOUNDARY_PAIR_COUNT_KEY,
    RPE_ROTATION_BOUNDARY_EXCLUDED_KEY,
    RPE_TRANSLATION_BOUNDARY_EXCLUDED_KEY,
    build_frame_contexts,
    build_proposals,
    check_refiner_settings,
    consensus_weight,
    decide_object_feedback,
    assess_bottleneck,
    gate_frame,
    gt_correction_error,
    robust_consensus,
    rotation_angle_deg,
    translation_norm,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HORIZON_REPO = REPO_ROOT / "externals" / "horizonstream"


def replay_trajectory_with_injections(
    chunk_cam_maps: Sequence[torch.Tensor],
    *,
    frame_count: int,
    window_size: int,
    injections: Mapping[int, np.ndarray],
    horizon_repo: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Replay the causal pose accumulator with absolute-target injections.

    Mirrors the verified GT-feedback POC path: chunk outputs are replayed
    through ``online_motion_averaging`` and each accepted frame's public c2w
    target is written into ``online_absolute_poses``/``last_absolute_poses``
    via ``_replace_internal_pose_for_public_target`` (frame-0 gauge, internal
    w2c).  Injections are absolute targets anchored to the raw-world
    reference clouds, so repeated corrections overwrite instead of
    accumulating.
    """

    repo = Path(horizon_repo).expanduser().resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from horizonstream.runtime.motion_averaging import online_motion_averaging

    from streaming_couping.scripts.run_scannet_horizonstream_gt_feedback_poc import (
        _normalize_internal_w2c_to_first,
        _replace_internal_pose_for_public_target,
    )

    if frame_count <= int(window_size):
        raise ValueError(
            "Replay needs more frames than the streaming window so that "
            "future chunks exist."
        )
    known_injections = {
        int(frame): np.asarray(target, dtype=np.float64)
        for frame, target in injections.items()
    }
    for frame in known_injections:
        if not 0 <= int(frame) < int(frame_count):
            raise ValueError(f"Injection frame {frame} is outside the sequence.")

    batch_size = 1
    valid_relative_poses = torch.empty(
        batch_size,
        frame_count - (int(window_size) - 1),
        int(window_size),
        3,
        4,
        dtype=torch.float32,
    )
    valid_focals_scales_shifts = torch.empty(
        batch_size,
        frame_count - (int(window_size) - 1),
        int(window_size),
        4,
        dtype=torch.float32,
    )
    online_absolute_poses = torch.empty(
        batch_size, frame_count, 3, 4, dtype=torch.float32
    )
    last_absolute_poses = torch.empty_like(online_absolute_poses)
    online_scales = torch.ones(
        batch_size,
        frame_count - (int(window_size) - 1),
        1,
        dtype=torch.float32,
    )

    online_ptr = 0
    valid_ptr = 0
    applied: list[int] = []
    for chunk_index, chunk_cam_map in enumerate(chunk_cam_maps):
        before_ptr = int(online_ptr)
        online_ptr, valid_ptr = online_motion_averaging(
            chunk_cam_map=chunk_cam_map,
            chunk_idx=chunk_index,
            B=batch_size,
            Win=int(window_size),
            online_absolute_poses=online_absolute_poses,
            online_S=online_scales,
            valid_relative_poses=valid_relative_poses,
            valid_focals_scales_shifts=valid_focals_scales_shifts,
            online_ptr=online_ptr,
            valid_ptr=valid_ptr,
            last_absolute_poses=last_absolute_poses,
            dtype=torch.float32,
        )
        after_ptr = int(online_ptr)
        for frame in sorted(
            frame for frame in known_injections if before_ptr <= frame < after_ptr
        ):
            target = known_injections[frame]
            _replace_internal_pose_for_public_target(
                online_absolute_poses,
                frame_index=int(frame),
                target_public_c2w=target,
            )
            _replace_internal_pose_for_public_target(
                last_absolute_poses,
                frame_index=int(frame),
                target_public_c2w=target,
            )
            applied.append(int(frame))

    missed = sorted(set(known_injections) - set(applied))
    if missed:
        raise RuntimeError(
            f"Injection frames were never reached by the chunk schedule: {missed}"
        )

    public_c2w = _normalize_internal_w2c_to_first(online_absolute_poses)
    payload = {
        "applied_frames": applied,
        "injection_count": int(len(applied)),
        "kv_cache_modified": False,
        "gla_cache_modified": False,
        "feedback_state": "online_absolute_poses_and_last_absolute_poses",
    }
    return public_c2w, payload


def trajectory_metrics_with_boundaries(
    predicted_c2w: np.ndarray,
    gt_c2w: np.ndarray,
    *,
    correction_frames: set[int],
) -> dict[str, Any]:
    """POC trajectory metrics plus RPE that excludes correction boundaries."""

    summary, _rows = _trajectory_metrics(predicted_c2w, gt_c2w)
    rpe_translation: list[float] = []
    rpe_rotation: list[float] = []
    excluded = 0
    for frame in range(1, len(predicted_c2w)):
        if frame in correction_frames or (frame - 1) in correction_frames:
            excluded += 1
            continue
        predicted_delta = (
            np.linalg.inv(predicted_c2w[frame - 1]) @ predicted_c2w[frame]
        )
        gt_delta = np.linalg.inv(gt_c2w[frame - 1]) @ gt_c2w[frame]
        error = np.linalg.inv(gt_delta) @ predicted_delta
        rpe_translation.append(float(np.linalg.norm(error[:3, 3])))
        rpe_rotation.append(
            _rotation_error_deg(error, np.eye(4, dtype=np.float64))
        )

    def rmse(values: Sequence[float]) -> float | None:
        finite = [float(value) for value in values if math.isfinite(float(value))]
        if not finite:
            return None
        return float(np.sqrt(np.mean(np.square(finite))))

    summary[RPE_TRANSLATION_BOUNDARY_EXCLUDED_KEY] = rmse(rpe_translation)
    summary[RPE_ROTATION_BOUNDARY_EXCLUDED_KEY] = rmse(rpe_rotation)
    summary[RPE_EXCLUDED_BOUNDARY_PAIR_COUNT_KEY] = int(excluded)
    return summary


def check_replay_equivalence(
    replay_c2w: np.ndarray,
    cache_world_to_camera: np.ndarray,
    diagnostics_raw_poses: torch.Tensor,
    *,
    translation_tolerance_m: float,
    rotation_tolerance_deg: float,
) -> dict[str, Any]:
    """Replay-without-injections must reproduce the cached raw trajectory."""

    cache_c2w = np.stack(
        [
            np.linalg.inv(_homogeneous(np.asarray(pose, dtype=np.float64)))
            for pose in np.asarray(cache_world_to_camera, dtype=np.float64)
        ],
        axis=0,
    )
    diagnostics_c2w = (
        torch.as_tensor(diagnostics_raw_poses).detach().float().cpu().numpy()
    )
    if replay_c2w.shape != cache_c2w.shape or diagnostics_c2w.shape != cache_c2w.shape:
        raise ValueError("Trajectory shapes disagree between replay/cache/diagnostics.")

    max_cache_translation = 0.0
    max_cache_rotation = 0.0
    max_diagnostics_translation = 0.0
    max_diagnostics_rotation = 0.0
    for index in range(cache_c2w.shape[0]):
        max_cache_translation = max(
            max_cache_translation,
            float(
                np.linalg.norm(
                    replay_c2w[index, :3, 3] - cache_c2w[index, :3, 3]
                )
            ),
        )
        max_cache_rotation = max(
            max_cache_rotation,
            _rotation_error_deg(replay_c2w[index], cache_c2w[index]),
        )
        max_diagnostics_translation = max(
            max_diagnostics_translation,
            float(
                np.linalg.norm(
                    replay_c2w[index, :3, 3] - diagnostics_c2w[index, :3, 3]
                )
            ),
        )
        max_diagnostics_rotation = max(
            max_diagnostics_rotation,
            _rotation_error_deg(replay_c2w[index], diagnostics_c2w[index]),
        )
    passed = (
        max_cache_translation <= float(translation_tolerance_m)
        and max_cache_rotation <= float(rotation_tolerance_deg)
        and max_diagnostics_translation <= float(translation_tolerance_m)
        and max_diagnostics_rotation <= float(rotation_tolerance_deg)
    )
    return {
        "passed": bool(passed),
        "translation_tolerance_m": float(translation_tolerance_m),
        "rotation_tolerance_deg": float(rotation_tolerance_deg),
        "max_replay_vs_cache_translation_m": max_cache_translation,
        "max_replay_vs_cache_rotation_deg": max_cache_rotation,
        "max_replay_vs_diagnostics_translation_m": max_diagnostics_translation,
        "max_replay_vs_diagnostics_rotation_deg": max_diagnostics_rotation,
    }


def _percentile(values: Sequence[float], q: float) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.percentile(finite, q)) if finite else None


def _median(values: Sequence[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.median(finite)) if finite else None


def _mean(values: Sequence[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else None


def _ratio(value: float | None) -> float | None:
    return None if value is None else float(value)


def _fmt(value: Any) -> str:
    return "None" if value is None else f"{float(value):.6f}"


def _load_diagnostics(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "object_pose_loss_feedback_diagnostics_r1":
        raise ValueError(f"Unsupported feedback diagnostics schema: {path}")
    return payload


def _load_chunk_cam_maps(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "horizonstream_chunk_cam_maps":
        raise ValueError(f"Unsupported chunk camera-map schema: {path}")
    return payload


def _build_config(args: argparse.Namespace) -> ObjectPoseFeedbackConfig:
    values = {}
    defaults = ObjectPoseFeedbackConfig()
    for field in dataclass_fields(ObjectPoseFeedbackConfig):
        values[field.name] = getattr(args, field.name)
    config = ObjectPoseFeedbackConfig(**values)
    del defaults
    return config.validate()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Stage-2a output directory containing object_pose_refinement/.",
    )
    parser.add_argument("--geometry-cache", type=Path, required=True)
    parser.add_argument(
        "--chunk-cam-maps",
        type=Path,
        default=None,
        help="Explicit chunk camera-map path; defaults to the sibling "
        "<cache stem>_chunk_cam_maps.pt.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <run-dir>/object_pose_feedback.",
    )
    parser.add_argument(
        "--horizon-repo",
        type=Path,
        default=DEFAULT_HORIZON_REPO,
    )
    parser.add_argument(
        "--equivalence-translation-tolerance-m",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--equivalence-rotation-tolerance-deg",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--allow-refiner-mismatch",
        action="store_true",
        help=(
            "Downgrade a stage-2a/stage-2b threshold mismatch from an error to "
            "a warning. The aggregate re-check and the correction clamps then "
            "no longer mirror the refiner that produced the proposals, so the "
            "run is recorded with refiner_config_match=false."
        ),
    )
    for field in dataclass_fields(ObjectPoseFeedbackConfig):
        default = getattr(ObjectPoseFeedbackConfig(), field.name)
        flag = "--" + field.name.replace("_", "-")
        if isinstance(default, bool):
            parser.add_argument(flag, action="store_true")
        elif isinstance(default, int):
            parser.add_argument(flag, type=int, default=default)
        else:
            parser.add_argument(flag, type=float, default=default)
    return parser.parse_args()


def pin_thread_determinism() -> None:
    """Force single-threaded torch for the accumulator replay.

    The replay re-derives the causal trajectory through
    ``online_motion_averaging``, whose medians and matrix products are
    reductions.  A multi-threaded reduction sums in a scheduling-dependent
    order, so two replays of the same cached chunk maps disagree in the last
    bits -- and the rotation metrics turn that into a percent of a 0.04 degree
    angle, which the determinism gate then reports as a material change.
    Pinning the thread count makes the replay bit-reproducible, which is what
    lets a branch difference be attributed to the branch.
    """

    torch.set_num_threads(1)


def main() -> None:
    args = _parse_args()
    config = _build_config(args)
    pin_thread_determinism()
    run_dir = args.run_dir.expanduser().resolve()
    geometry_cache = args.geometry_cache.expanduser().resolve()
    diagnostics_path = run_dir / "object_pose_refinement" / "feedback_diagnostics.pt"
    if not diagnostics_path.is_file():
        raise FileNotFoundError(
            f"Missing feedback diagnostics (was stage 2a run with "
            f"--object-pose-loss-export-feedback-diagnostics?): {diagnostics_path}"
        )
    chunk_cam_maps_path = (
        args.chunk_cam_maps.expanduser().resolve()
        if args.chunk_cam_maps is not None
        else geometry_cache.parent
        / f"{geometry_cache.stem}_chunk_cam_maps.pt"
    )
    if not chunk_cam_maps_path.is_file():
        raise FileNotFoundError(
            "Missing chunk camera-map artifact (regenerate the geometry cache "
            f"with --save-chunk-cam-maps): {chunk_cam_maps_path}"
        )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else run_dir / "object_pose_feedback"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    diagnostics = _load_diagnostics(diagnostics_path)
    refiner_settings = diagnostics.get("refiner_settings")
    if not refiner_settings:
        raise ValueError(
            "Feedback diagnostics carry no refiner_settings block; stage 2a "
            "predates the threshold-mirroring check. Re-run stage 2a with "
            "--object-pose-loss-export-feedback-diagnostics."
        )
    refiner_match = check_refiner_settings(refiner_settings, config=config)
    if not refiner_match["matches"]:
        message = (
            "Stage-2a refiner settings and stage-2b feedback thresholds "
            f"disagree: {refiner_match}. The aggregate alignment re-check and "
            "the correction clamps are documented to mirror the refiner; "
            "align the flags (or pass --allow-refiner-mismatch to record the "
            "run as intentionally non-mirroring)."
        )
        if not args.allow_refiner_mismatch:
            raise RuntimeError(message)
        print(f"WARNING: {message}", file=sys.stderr)
    cache_payload = load_horizonstream_cache(geometry_cache)
    aux = _load_chunk_cam_maps(chunk_cam_maps_path)

    frame_ids = [int(value) for value in diagnostics.get("frame_ids", ())]
    frame_count = len(frame_ids)
    if frame_ids != list(range(frame_count)):
        raise ValueError(
            "Feedback analysis expects contiguous frame ids 0..F-1 "
            "(HorizonStream cache convention)."
        )
    cache_frame_ids = [int(value) for value in cache_payload.get("frame_ids", ())]
    if cache_frame_ids != frame_ids:
        raise ValueError("Geometry cache frame ids do not match diagnostics.")
    if int(aux["frame_count"]) != frame_count:
        raise ValueError("Chunk camera-map frame count does not match diagnostics.")
    if int(aux["window_size"]) != int(cache_payload["window_size"]) or int(
        aux["sliding_size"]
    ) != int(cache_payload["sliding_size"]):
        raise ValueError(
            "Chunk camera-map window/sliding settings do not match the cache."
        )
    window_size = int(aux["window_size"])
    chunk_cam_maps = [
        torch.as_tensor(chunk).detach().float().cpu()
        for chunk in aux["chunk_cam_maps"]
    ]

    # ---- Branch A: raw replay + equivalence assertions ------------------
    raw_c2w, raw_payload = replay_trajectory_with_injections(
        chunk_cam_maps,
        frame_count=frame_count,
        window_size=window_size,
        injections={},
        horizon_repo=args.horizon_repo,
    )
    equivalence = check_replay_equivalence(
        raw_c2w,
        cache_payload["world_to_camera"],
        diagnostics["raw_camera_to_world"],
        translation_tolerance_m=float(args.equivalence_translation_tolerance_m),
        rotation_tolerance_deg=float(args.equivalence_rotation_tolerance_deg),
    )
    if not equivalence["passed"]:
        raise RuntimeError(
            "Raw replay does not reproduce the cached trajectory; the chunk "
            f"camera maps are stale or inconsistent: {equivalence}"
        )
    print(
        "Raw replay equivalence OK "
        f"(max_t={_fmt(equivalence['max_replay_vs_cache_translation_m'])} m, "
        f"max_r={_fmt(equivalence['max_replay_vs_cache_rotation_deg'])} deg)"
    )

    # ---- Proposals, reliability, consensus, gating (no GT) --------------
    proposals = build_proposals(diagnostics, config=config)
    contexts = build_frame_contexts(diagnostics, config=config)
    proposals_by_frame: dict[int, list[Any]] = {
        index: [] for index in range(frame_count)
    }
    for proposal in proposals:
        proposals_by_frame[proposal.sequence_index].append(proposal)

    gate_results: dict[str, list[Any]] = {}
    for variant in CONSENSUS_VARIANTS:
        gate_results[variant.name] = [
            gate_frame(
                proposals_by_frame[index],
                variant,
                context=contexts[index],
                config=config,
            )
            for index in range(frame_count)
        ]

    # ---- Per-variant replays (still no GT) ------------------------------
    trajectories: dict[str, np.ndarray] = {"raw": raw_c2w}
    replay_payloads: dict[str, dict[str, Any]] = {"raw": raw_payload}
    for variant in CONSENSUS_VARIANTS:
        injections = {
            result.sequence_index: result.target_c2w.numpy()
            for result in gate_results[variant.name]
            if result.accepted and result.target_c2w is not None
        }
        trajectory, payload = replay_trajectory_with_injections(
            chunk_cam_maps,
            frame_count=frame_count,
            window_size=window_size,
            injections=injections,
            horizon_repo=args.horizon_repo,
        )
        trajectories[variant.name] = trajectory
        replay_payloads[variant.name] = payload

    # ---- GT is loaded strictly after all feedback decisions --------------
    gt_c2w_absolute = _load_gt_c2w(
        args.manifest.expanduser().resolve(),
        scene_id=str(args.scene_id),
        source_positions=[int(value) for value in cache_payload["source_positions"]],
    )
    gt_c2w = _relative_to_first(gt_c2w_absolute)
    if gt_c2w.shape[0] != frame_count:
        raise ValueError("GT frame count does not match the replayed sequence.")
    gt_deltas = np.stack(
        [
            gt_c2w[index] @ np.linalg.inv(raw_c2w[index])
            for index in range(frame_count)
        ],
        axis=0,
    )
    gt_translation_saturated = np.stack(
        [
            float(np.linalg.norm(gt_deltas[index, :3, 3]))
            > float(config.max_feedback_translation_m)
            for index in range(frame_count)
        ]
    )
    gt_rotation_saturated = np.stack(
        [
            _rotation_error_deg(gt_deltas[index], np.eye(4))
            > float(config.max_feedback_rotation_deg)
            for index in range(frame_count)
        ]
    )

    # ---- Per-proposal GT evaluation -------------------------------------
    main_variant = next(
        variant for variant in CONSENSUS_VARIANTS if variant.name == MAIN_VARIANT_NAME
    )
    consensus_errors_by_frame: dict[int, dict[int, tuple[float, float, bool]]] = {}
    for index in range(frame_count):
        frame_proposals = [
            proposal
            for proposal in proposals_by_frame[index]
            if proposal.correction is not None
        ]
        semantic_ok = [
            proposal
            for proposal in frame_proposals
            if proposal.semantic_reject_reason is None
        ]
        reliable = [
            proposal
            for proposal in semantic_ok
            if proposal.accepted_by_refiner
            and proposal.geometry_reject_reason is None
        ]
        if len(reliable) >= int(config.min_consensus_objects):
            consensus = robust_consensus(reliable, main_variant, config=config)
            errors: dict[int, tuple[float, float, bool]] = {}
            for proposal, t_error, r_error, inlier in zip(
                reliable,
                consensus.translation_errors,
                consensus.rotation_errors,
                consensus.inlier_flags,
            ):
                errors[proposal.instance_id] = (t_error, r_error, inlier)
            consensus_errors_by_frame[index] = errors

    proposal_rows: list[dict[str, Any]] = []
    proposal_gt_translation: list[float] = []
    proposal_gt_rotation: list[float] = []
    proposal_semantic_scores: list[float | None] = []
    for proposal in proposals:
        gt_t_error: float | None = None
        gt_r_error: float | None = None
        delta_translation: float | None = None
        delta_rotation: float | None = None
        if proposal.correction is not None:
            delta_translation = translation_norm(proposal.correction)
            delta_rotation = rotation_angle_deg(proposal.correction[:3, :3])
            gt_t_error, gt_r_error = gt_correction_error(
                proposal.correction, gt_deltas[proposal.sequence_index]
            )
            proposal_gt_translation.append(gt_t_error)
            proposal_gt_rotation.append(gt_r_error)
            proposal_semantic_scores.append(proposal.semantic_confidence)
        consensus_entry = consensus_errors_by_frame.get(
            proposal.sequence_index, {}
        ).get(proposal.instance_id)
        reference_frames = ";".join(
            str(value) for value in proposal.reference_frames
        )
        reference_roles = ";".join(
            str(value) for value in proposal.reference_roles
        )
        proposal_rows.append(
            {
                "frame": int(proposal.sequence_index),
                "instance_id": int(proposal.instance_id),
                "category": proposal.category,
                "track_length": int(proposal.track_length),
                "visibility_ratio": float(proposal.visibility_ratio),
                "point_count": int(proposal.point_count),
                "overlap_count": int(proposal.overlap_count),
                "alignment_loss_before": _ratio(proposal.alignment_loss_before),
                "alignment_loss_after": _ratio(proposal.alignment_loss_after),
                "inlier_ratio": float(proposal.inlier_ratio),
                "eigenvalue_1": float(proposal.eigenvalue_1),
                "eigenvalue_2": float(proposal.eigenvalue_2),
                "eigenvalue_3": float(proposal.eigenvalue_3),
                "geometry_type": proposal.geometry_type,
                "degeneracy_factor": float(proposal.degeneracy_factor),
                "delta_translation_norm": _ratio(delta_translation),
                "delta_rotation_deg": _ratio(delta_rotation),
                "semantic_confidence": _ratio(proposal.semantic_confidence),
                "geometry_confidence": _ratio(proposal.geometry_confidence),
                "accepted_by_refiner": bool(proposal.accepted_by_refiner),
                "refiner_reason": proposal.refiner_reason,
                "object_reject_reason": proposal.object_reject_reason,
                "consensus_inlier": (
                    consensus_entry[2] if consensus_entry is not None else None
                ),
                "translation_consensus_error": (
                    consensus_entry[0] if consensus_entry is not None else None
                ),
                "rotation_consensus_error": (
                    consensus_entry[1] if consensus_entry is not None else None
                ),
                "gt_translation_correction_error": _ratio(gt_t_error),
                "gt_rotation_correction_error": _ratio(gt_r_error),
                "gt_translation_saturated": bool(
                    gt_translation_saturated[proposal.sequence_index]
                ),
                "gt_rotation_saturated": bool(
                    gt_rotation_saturated[proposal.sequence_index]
                ),
                "reference_frames": reference_frames,
                "reference_roles": reference_roles,
            }
        )
    _write_csv(
        output_dir / "object_proposals.csv",
        proposal_rows,
        (
            "frame",
            "instance_id",
            "category",
            "track_length",
            "visibility_ratio",
            "point_count",
            "overlap_count",
            "alignment_loss_before",
            "alignment_loss_after",
            "inlier_ratio",
            "eigenvalue_1",
            "eigenvalue_2",
            "eigenvalue_3",
            "geometry_type",
            "degeneracy_factor",
            "delta_translation_norm",
            "delta_rotation_deg",
            "semantic_confidence",
            "geometry_confidence",
            "accepted_by_refiner",
            "refiner_reason",
            "object_reject_reason",
            "consensus_inlier",
            "translation_consensus_error",
            "rotation_consensus_error",
            "gt_translation_correction_error",
            "gt_rotation_correction_error",
            "gt_translation_saturated",
            "gt_rotation_saturated",
            "reference_frames",
            "reference_roles",
        ),
    )

    # ---- Branch trajectories vs GT --------------------------------------
    branch_metrics: dict[str, dict[str, Any]] = {}
    for name, trajectory in trajectories.items():
        correction_frames = (
            set()
            if name == "raw"
            else {
                result.sequence_index
                for result in gate_results[name]
                if result.accepted
            }
        )
        branch_metrics[name] = trajectory_metrics_with_boundaries(
            trajectory, gt_c2w, correction_frames=correction_frames
        )

    # ---- frame_metrics.csv ------------------------------------------------
    frame_rows: list[dict[str, Any]] = []
    raw_frame_errors: list[tuple[float, float]] = []
    for index in range(frame_count):
        raw_t = float(np.linalg.norm(raw_c2w[index, :3, 3] - gt_c2w[index, :3, 3]))
        raw_r = _rotation_error_deg(raw_c2w[index], gt_c2w[index])
        raw_frame_errors.append((raw_t, raw_r))
        for name in ["raw"] + [variant.name for variant in CONSENSUS_VARIANTS]:
            gate = (
                None if name == "raw" else gate_results[name][index]
            )
            trajectory = trajectories[name]
            feedback_t = float(
                np.linalg.norm(
                    trajectory[index, :3, 3] - gt_c2w[index, :3, 3]
                )
            )
            feedback_r = _rotation_error_deg(trajectory[index], gt_c2w[index])
            frame_rows.append(
                {
                    "variant": name,
                    "frame": int(index),
                    "raw_translation_error": raw_t,
                    "feedback_translation_error": feedback_t,
                    "raw_rotation_error": raw_r,
                    "feedback_rotation_error": feedback_r,
                    "num_object_proposals": (
                        0 if gate is None else int(gate.num_proposals)
                    ),
                    "num_reliable_objects": (
                        0 if gate is None else int(gate.num_reliable)
                    ),
                    "accepted": True if gate is None else bool(gate.accepted),
                    "reject_reason": (
                        None if gate is None or gate.accepted else gate.reason
                    ),
                    "consensus_translation_norm": (
                        None
                        if gate is None
                        else _ratio(gate.consensus_translation_norm)
                    ),
                    "consensus_rotation_deg": (
                        None
                        if gate is None
                        else _ratio(gate.consensus_rotation_deg)
                    ),
                }
            )
    _write_csv(
        output_dir / "frame_metrics.csv",
        frame_rows,
        (
            "variant",
            "frame",
            "raw_translation_error",
            "feedback_translation_error",
            "raw_rotation_error",
            "feedback_rotation_error",
            "num_object_proposals",
            "num_reliable_objects",
            "accepted",
            "reject_reason",
            "consensus_translation_norm",
            "consensus_rotation_deg",
        ),
    )

    # ---- consensus_metrics.csv -------------------------------------------
    consensus_rows: list[dict[str, Any]] = []
    consensus_gt_translation_main: list[float] = []
    consensus_gt_rotation_main: list[float] = []
    gate_reason_counts: dict[str, dict[str, int]] = {}
    accepted_frames_by_variant: dict[str, list[int]] = {}
    eligible_frames = {
        index
        for index in range(frame_count)
        if index >= int(config.anchor_frame_count)
        and proposals_by_frame[index]
    }
    for variant in CONSENSUS_VARIANTS:
        reason_counts: dict[str, int] = {}
        accepted_frames: list[int] = []
        for index in range(frame_count):
            gate = gate_results[variant.name][index]
            if not gate.accepted and gate.reason is not None:
                reason_counts[gate.reason] = (
                    reason_counts.get(gate.reason, 0) + 1
                )
            if gate.accepted:
                accepted_frames.append(index)
            gt_consensus_t: float | None = None
            gt_consensus_r: float | None = None
            if gate.delta is not None:
                gt_consensus_t, gt_consensus_r = gt_correction_error(
                    gate.delta, gt_deltas[index]
                )
                if variant.name == MAIN_VARIANT_NAME:
                    consensus_gt_translation_main.append(gt_consensus_t)
                    consensus_gt_rotation_main.append(gt_consensus_r)
            consensus_rows.append(
                {
                    "variant": variant.name,
                    "frame": int(index),
                    "num_proposals": int(gate.num_proposals),
                    "num_semantic_reliable": int(gate.num_semantic_reliable),
                    "num_reliable": int(gate.num_reliable),
                    "inlier_count": int(gate.inlier_count),
                    "consensus_translation_norm": _ratio(
                        gate.consensus_translation_norm
                    ),
                    "consensus_rotation_deg": _ratio(gate.consensus_rotation_deg),
                    "aggregate_loss_before": _ratio(gate.aggregate_loss_before),
                    "aggregate_loss_after": _ratio(gate.aggregate_loss_after),
                    "accepted": bool(gate.accepted),
                    "reason": gate.reason,
                    "gt_consensus_translation_error": _ratio(gt_consensus_t),
                    "gt_consensus_rotation_error": _ratio(gt_consensus_r),
                }
            )
        gate_reason_counts[variant.name] = reason_counts
        accepted_frames_by_variant[variant.name] = accepted_frames
    _write_csv(
        output_dir / "consensus_metrics.csv",
        consensus_rows,
        (
            "variant",
            "frame",
            "num_proposals",
            "num_semantic_reliable",
            "num_reliable",
            "inlier_count",
            "consensus_translation_norm",
            "consensus_rotation_deg",
            "aggregate_loss_before",
            "aggregate_loss_after",
            "accepted",
            "reason",
            "gt_consensus_translation_error",
            "gt_consensus_rotation_error",
        ),
    )

    # ---- future_pose_gain.csv --------------------------------------------
    future_rows: list[dict[str, Any]] = []
    future_translation_gains: dict[str, list[float]] = {}
    future_rotation_gains: dict[str, list[float]] = {}
    window = int(config.future_gain_window)
    for variant in CONSENSUS_VARIANTS:
        gains_t: list[float] = []
        gains_r: list[float] = []
        trajectory = trajectories[variant.name]
        for frame in accepted_frames_by_variant[variant.name]:
            end = min(frame_count, frame + window + 1)
            for future in range(frame + 1, end):
                raw_t = raw_frame_errors[future][0]
                raw_r = raw_frame_errors[future][1]
                feedback_t = float(
                    np.linalg.norm(
                        trajectory[future, :3, 3] - gt_c2w[future, :3, 3]
                    )
                )
                feedback_r = _rotation_error_deg(trajectory[future], gt_c2w[future])
                gain_t = raw_t - feedback_t
                gain_r = raw_r - feedback_r
                gains_t.append(gain_t)
                gains_r.append(gain_r)
                future_rows.append(
                    {
                        "variant": variant.name,
                        "correction_frame": int(frame),
                        "k": int(future - frame),
                        "frame": int(future),
                        "raw_translation_error": raw_t,
                        "feedback_translation_error": feedback_t,
                        "raw_rotation_error": raw_r,
                        "feedback_rotation_error": feedback_r,
                        "future_translation_gain": gain_t,
                        "future_rotation_gain": gain_r,
                    }
                )
        future_translation_gains[variant.name] = gains_t
        future_rotation_gains[variant.name] = gains_r
    _write_csv(
        output_dir / "future_pose_gain.csv",
        future_rows,
        (
            "variant",
            "correction_frame",
            "k",
            "frame",
            "raw_translation_error",
            "feedback_translation_error",
            "raw_rotation_error",
            "feedback_rotation_error",
            "future_translation_gain",
            "future_rotation_gain",
        ),
    )

    # ---- Decisions and summary -------------------------------------------
    decisions: dict[str, Any] = {}
    accepted_ratios: dict[str, float | None] = {}
    for variant in CONSENSUS_VARIANTS:
        accepted_count = len(accepted_frames_by_variant[variant.name])
        accepted_ratio = (
            float(accepted_count) / len(eligible_frames)
            if eligible_frames
            else None
        )
        accepted_ratios[variant.name] = accepted_ratio
        decisions[variant.name] = decide_object_feedback(
            variant_name=variant.name,
            raw_metrics=branch_metrics["raw"],
            feedback_metrics=branch_metrics[variant.name],
            future_translation_gains=future_translation_gains[variant.name],
            future_rotation_gains=future_rotation_gains[variant.name],
            accepted_ratio=accepted_ratio,
            config=config,
        )

    main_accepted_ratio = accepted_ratios[MAIN_VARIANT_NAME]
    bottleneck = assess_bottleneck(
        proposal_gt_translation_errors=proposal_gt_translation,
        proposal_gt_rotation_errors=proposal_gt_rotation,
        proposal_semantic_confidences=proposal_semantic_scores,
        consensus_gt_translation_errors=consensus_gt_translation_main,
        consensus_gt_rotation_errors=consensus_gt_rotation_main,
        gate_reason_counts=gate_reason_counts[MAIN_VARIANT_NAME],
        accepted_ratio=main_accepted_ratio,
        feedback_ate_improvement_ratio=(
            (
                float(branch_metrics["raw"]["ate_rmse_m"])
                - float(branch_metrics[MAIN_VARIANT_NAME]["ate_rmse_m"])
            )
            / float(branch_metrics["raw"]["ate_rmse_m"])
            if branch_metrics["raw"].get("ate_rmse_m") is not None
            and branch_metrics[MAIN_VARIANT_NAME].get("ate_rmse_m") is not None
            and float(branch_metrics["raw"]["ate_rmse_m"]) > 0.0
            else None
        ),
    )

    by_category: dict[str, list[float]] = {}
    by_geometry: dict[str, list[float]] = {}
    for row in proposal_rows:
        if row["gt_translation_correction_error"] is None:
            continue
        by_category.setdefault(row["category"], []).append(
            float(row["gt_translation_correction_error"])
        )
        by_geometry.setdefault(row["geometry_type"], []).append(
            float(row["gt_translation_correction_error"])
        )

    proposal_within_5cm = (
        sum(1 for value in proposal_gt_translation if value <= 0.05)
        / len(proposal_gt_translation)
        if proposal_gt_translation
        else None
    )
    single_variant = "single"
    single_consensus_gt_t = [
        row["gt_consensus_translation_error"]
        for row in consensus_rows
        if row["variant"] == single_variant
        and row["gt_consensus_translation_error"] is not None
    ]
    answers = {
        "q1_same_instance_alignment_contains_camera_signal": {
            "fraction_proposals_within_5cm": proposal_within_5cm,
            "proposal_gt_translation_error_median_m": _median(
                proposal_gt_translation
            ),
            "proposal_gt_rotation_error_median_deg": _median(
                proposal_gt_rotation
            ),
            "gt_translation_saturated_frame_ratio": float(
                np.mean(gt_translation_saturated)
            ),
            "gt_rotation_saturated_frame_ratio": float(
                np.mean(gt_rotation_saturated)
            ),
        },
        "q2_single_proposal_error": {
            "translation_median_m": _median(proposal_gt_translation),
            "translation_p75_m": _percentile(proposal_gt_translation, 75),
            "translation_p90_m": _percentile(proposal_gt_translation, 90),
            "rotation_median_deg": _median(proposal_gt_rotation),
            "rotation_p90_deg": _percentile(proposal_gt_rotation, 90),
        },
        "q3_consensus_vs_single_object": {
            "per_proposal_translation_median_m": _median(proposal_gt_translation),
            "single_variant_consensus_translation_median_m": _median(
                single_consensus_gt_t
            ),
            "main_variant_consensus_translation_median_m": _median(
                consensus_gt_translation_main
            ),
            "main_variant_consensus_rotation_median_deg": _median(
                consensus_gt_rotation_main
            ),
        },
        "q4_error_prone_objects": {
            "mean_gt_translation_error_by_category_m": {
                key: _mean(values) for key, values in sorted(by_category.items())
            },
            "mean_gt_translation_error_by_geometry_type_m": {
                key: _mean(values) for key, values in sorted(by_geometry.items())
            },
        },
        "q5_future_gain_after_feedback": {
            variant.name: {
                "translation_gain_median_m": _median(
                    future_translation_gains[variant.name]
                ),
                "translation_gain_positive_ratio": (
                    (
                        sum(
                            1
                            for value in future_translation_gains[variant.name]
                            if value > 1e-9
                        )
                        / len(future_translation_gains[variant.name])
                    )
                    if future_translation_gains[variant.name]
                    else None
                ),
                "rotation_gain_median_deg": _median(
                    future_rotation_gains[variant.name]
                ),
            }
            for variant in CONSENSUS_VARIANTS
        },
        "q6_translation_vs_rotation_reliability": {
            "proposal_translation_median_m": _median(proposal_gt_translation),
            "proposal_rotation_median_deg": _median(proposal_gt_rotation),
            "future_translation_gain_median_m": _median(
                future_translation_gains[MAIN_VARIANT_NAME]
            ),
            "future_rotation_gain_median_deg": _median(
                future_rotation_gains[MAIN_VARIANT_NAME]
            ),
        },
        "q7_long_term_drift": {
            "raw_direct_ate_rmse_m": branch_metrics["raw"].get("ate_rmse_m"),
            "variant_direct_ate_rmse_m": {
                name: metrics.get("ate_rmse_m")
                for name, metrics in branch_metrics.items()
                if name != "raw"
            },
            "raw_sim3_ate_rmse_m": branch_metrics["raw"].get("ate_rmse_sim3_m"),
            "variant_sim3_ate_rmse_m": {
                name: metrics.get("ate_rmse_sim3_m")
                for name, metrics in branch_metrics.items()
                if name != "raw"
            },
        },
        "q8_bottleneck": bottleneck,
    }

    summary = {
        "schema": "horizonstream_object_pose_feedback_v1",
        "scene_id": str(args.scene_id),
        "frame_count": int(frame_count),
        "run_dir": str(run_dir),
        "geometry_cache": str(geometry_cache),
        "chunk_cam_maps": str(chunk_cam_maps_path),
        "variants": [variant.name for variant in CONSENSUS_VARIANTS],
        "main_variant": MAIN_VARIANT_NAME,
        "config": config.to_dict(),
        "refiner_config_match": bool(refiner_match["matches"]),
        "refiner_settings_audit": {
            "exported_by_stage_2a": {
                str(key): refiner_settings[key]
                for key in sorted(refiner_settings)
            },
            "mirror_check": refiner_match,
            "mismatch_enforced": not bool(args.allow_refiner_mismatch),
        },
        "pose_convention": {
            "branch_trajectories": "camera_to_world, frame-0 gauge",
            "proposal_correction": "delta_c2w = T_aligned @ inverse(T_raw), "
            "left-multiplied on raw-world object points",
            "feedback_target": "absolute public c2w injected into "
            "online_motion_averaging.online_absolute_poses via the verified "
            "GT-feedback primitive",
            "gt_source": "manifest.world_to_camera inverted to camera_to_world, "
            "normalized to the first selected frame",
        },
        "gt_audit": {
            "gt_used_for_proposals": False,
            "gt_used_for_reliability_consensus_or_gating": False,
            "gt_loaded_after_feedback_decisions": True,
            "gt_usage": "offline evaluation only",
        },
        "replay_equivalence": equivalence,
        "replay_payloads": {
            name: {"injection_count": payload["injection_count"]}
            for name, payload in replay_payloads.items()
        },
        "branches": branch_metrics,
        "gate_stats": {
            variant.name: {
                "accepted_frame_count": len(
                    accepted_frames_by_variant[variant.name]
                ),
                "accepted_frames": accepted_frames_by_variant[variant.name],
                "accepted_ratio_over_eligible": accepted_ratios[variant.name],
                "reject_reason_counts": gate_reason_counts[variant.name],
            }
            for variant in CONSENSUS_VARIANTS
        },
        "eligible_frame_count": len(eligible_frames),
        "proposal_count": len(proposal_rows),
        "decisions": decisions,
        "decision": decisions[MAIN_VARIANT_NAME]["decision"],
        "answers": answers,
        "outputs": {
            "object_proposals_csv": str(output_dir / "object_proposals.csv"),
            "frame_metrics_csv": str(output_dir / "frame_metrics.csv"),
            "consensus_metrics_csv": str(output_dir / "consensus_metrics.csv"),
            "future_pose_gain_csv": str(output_dir / "future_pose_gain.csv"),
            "poses_pt": str(output_dir / "poses.pt"),
            "summary_json": str(output_dir / "summary.json"),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    torch.save(
        {
            "raw_c2w": torch.from_numpy(raw_c2w).float(),
            "gt_c2w": torch.from_numpy(gt_c2w).float(),
            "gt_deltas": torch.from_numpy(gt_deltas).float(),
            "variant_c2w": {
                name: torch.from_numpy(trajectory).float()
                for name, trajectory in trajectories.items()
                if name != "raw"
            },
        },
        output_dir / "poses.pt",
    )

    print(
        f"Raw direct_ATE = {_fmt(branch_metrics['raw']['ate_rmse_m'])} m, "
        f"RPE_t = {_fmt(branch_metrics['raw']['rpe_translation_rmse_m'])} m"
    )
    for variant in CONSENSUS_VARIANTS:
        metrics = branch_metrics[variant.name]
        print(
            f"{variant.name}: direct_ATE = {_fmt(metrics['ate_rmse_m'])} m, "
            f"accepted = {len(accepted_frames_by_variant[variant.name])}, "
            f"decision = {decisions[variant.name]['decision']}"
        )
    print(f"Main variant decision: {summary['decision']}")
    print(f"Primary bottleneck: {bottleneck['primary_bottleneck']}")
    for key, value in summary["outputs"].items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
