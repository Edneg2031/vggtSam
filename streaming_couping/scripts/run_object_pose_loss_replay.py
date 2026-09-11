#!/usr/bin/env python3
"""Re-run the object-pose-loss refiner from cached observations (CPU only).

The refiner normally runs inside stage 2a, right after the segmentation model,
so changing any refiner knob means another segmentation run -- and the
segmentation model is not deterministic across runs on GPU.  The 100-frame
branch sweep showed the consequence: re-running the *unchanged* baseline moved
the direct-ATE gain from 0.053 to 0.079 and flipped the sim3 gain's sign, while
the branch effect being measured was 0.006.  The segmentation noise was four
times the effect.

``feedback_diagnostics.pt`` already carries everything the refiner consumes:
the per-observation camera-space clouds and weights, the frame ids, the raw
poses, the full refiner config, and the pre-filter collection counters.  This
script rebuilds the refiner from that file alone, so a refiner-parameter
comparison costs no GPU time and carries none of the segmentation model's
run-to-run variance.

    python -m streaming_couping.scripts.run_object_pose_loss_replay \
        --diagnostics <run>/object_pose_refinement/feedback_diagnostics.pt \
        --output-dir <branch-dir> \
        --object-pose-loss-max-reference-age-frames 15 \
        --object-pose-loss-proposal-mode rotation_then_translation

Writes ``<output-dir>/object_pose_refinement/feedback_diagnostics.pt``, which
is exactly where stage 2b looks, so the existing analysis runs unchanged:

    python -m streaming_couping.scripts.run_object_pose_feedback \
        --run-dir <branch-dir> --geometry-cache <shared> ...

``--check-equivalence`` replays with the source's own settings and asserts the
result matches, which is what makes the path trustworthy.
"""

from __future__ import annotations

import argparse
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import torch

from streaming_couping.src.semantic_mapping.object_pose_feedback import (
    rotation_angle_deg,
)
from streaming_couping.src.semantic_mapping.object_pose_loss_refinement import (
    PROPOSAL_MODES,
    ObjectCloudObservation,
    ObjectPoseLossRefinementConfig,
    ObjectPoseLossRefiner,
)

SCHEMA = "object_pose_loss_feedback_diagnostics_r1"

#: Config key -> CLI flag for the knobs a replay may override.
OVERRIDE_FLAGS: tuple[tuple[str, str], ...] = (
    ("max_reference_age_frames", "object-pose-loss-max-reference-age-frames"),
    (
        "anchor_refresh_interval_frames",
        "object-pose-loss-anchor-refresh-interval-frames",
    ),
    ("proposal_mode", "object-pose-loss-proposal-mode"),
)


def _flag_dest(flag: str) -> str:
    """argparse dest for a flag, so overrides read from the right attribute."""

    return flag.replace("-", "_")


def load_diagnostics(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"Unsupported feedback diagnostics schema: {path}")
    return payload


def observations_from_diagnostics(
    diagnostics: Mapping[str, Any],
) -> dict[int, tuple[ObjectCloudObservation, ...]]:
    """Rebuild the per-frame observations the refiner consumed."""

    by_frame: dict[int, list[ObjectCloudObservation]] = {}
    for row in diagnostics.get("observations", ()):
        frame_id = int(row["frame_id"])
        by_frame.setdefault(frame_id, []).append(
            ObjectCloudObservation(
                frame_id=frame_id,
                instance_id=int(row["instance_id"]),
                category=str(row["category"]),
                points_camera=torch.as_tensor(row["points_camera"])
                .detach()
                .float()
                .cpu(),
                weights=torch.as_tensor(row["weights"]).detach().float().cpu(),
                track_score=float(row["track_score"]),
                geometry_confidence=float(row["geometry_confidence"]),
                mask_pixels=int(row["mask_pixels"]),
                static=bool(row["static"]),
            )
        )
    return {frame_id: tuple(rows) for frame_id, rows in by_frame.items()}


def config_from_diagnostics(
    diagnostics: Mapping[str, Any],
    overrides: Mapping[str, Any],
) -> ObjectPoseLossRefinementConfig:
    stored = diagnostics.get("refiner_settings")
    if not isinstance(stored, Mapping) or not stored:
        raise ValueError(
            "Diagnostics carry no refiner_settings; they were written before "
            "the full config was exported. Re-run stage 2a."
        )
    settings = {str(key): value for key, value in stored.items()}
    settings.update({str(key): value for key, value in overrides.items()})
    # A replay is a stage-2a substitute: it must produce the diagnostics the
    # consumer expects, so force the export on regardless of the source config.
    settings["export_feedback_diagnostics"] = True
    settings["independent_instance_poses"] = True
    return ObjectPoseLossRefinementConfig(**settings).validate()


def _rotation_gap_deg(left: torch.Tensor, right: torch.Tensor) -> float:
    """Angle between two rotations.

    Uses the shared stable formulation: the trace-based acos reports ~0.03
    degrees for a rotation compared against *itself*, because float32 loses
    orthonormality at the 1e-7 level and acos amplifies it as sqrt(2 * eps).
    """

    relative = right[:3, :3].transpose(0, 1) @ left[:3, :3]
    return rotation_angle_deg(relative)


def compare_diagnostics(
    source: Mapping[str, Any],
    replay: Mapping[str, Any],
    *,
    translation_tolerance_m: float,
    rotation_tolerance_deg: float,
) -> dict[str, Any]:
    """Do a replay with the source settings reproduce the source proposals?"""

    def keyed(payload: Mapping[str, Any]) -> dict[tuple[int, int], Any]:
        return {
            (int(row["sequence_index"]), int(row["instance_id"])): row
            for row in payload.get("proposals", ())
        }

    source_proposals = keyed(source)
    replay_proposals = keyed(replay)
    if set(source_proposals) != set(replay_proposals):
        missing = sorted(set(source_proposals) - set(replay_proposals))[:5]
        extra = sorted(set(replay_proposals) - set(source_proposals))[:5]
        return {
            "passed": False,
            "reason": "proposal key sets differ",
            "missing_in_replay": missing,
            "extra_in_replay": extra,
        }

    max_translation = 0.0
    max_rotation = 0.0
    compared = 0
    for key, source_row in source_proposals.items():
        source_correction = source_row.get("correction")
        replay_correction = replay_proposals[key].get("correction")
        if source_correction is None or replay_correction is None:
            if (source_correction is None) != (replay_correction is None):
                return {
                    "passed": False,
                    "reason": f"correction presence differs at {key}",
                }
            continue
        left = torch.as_tensor(source_correction).detach().float().cpu()
        right = torch.as_tensor(replay_correction).detach().float().cpu()
        max_translation = max(
            max_translation,
            float(torch.linalg.vector_norm(left[:3, 3] - right[:3, 3])),
        )
        max_rotation = max(max_rotation, _rotation_gap_deg(left, right))
        compared += 1

    max_pose_translation = 0.0
    max_pose_rotation = 0.0
    source_poses = source.get("raw_camera_to_world")
    if source_poses is not None:
        # raw_camera_to_world is the source geometry, not a refiner output; it
        # is compared only to confirm the replay saw the same input poses.
        replay_poses = replay.get("raw_camera_to_world")
        if replay_poses is not None:
            left_stack = torch.as_tensor(source_poses).detach().float().cpu()
            right_stack = torch.as_tensor(replay_poses).detach().float().cpu()
            if left_stack.shape == right_stack.shape:
                max_pose_translation = float(
                    torch.linalg.vector_norm(
                        left_stack[:, :3, 3] - right_stack[:, :3, 3], dim=-1
                    ).max()
                )
                max_pose_rotation = max(
                    _rotation_gap_deg(left_stack[index], right_stack[index])
                    for index in range(int(left_stack.shape[0]))
                )
    passed = (
        max_translation <= float(translation_tolerance_m)
        and max_rotation <= float(rotation_tolerance_deg)
        and max_pose_translation <= float(translation_tolerance_m)
        and max_pose_rotation <= float(rotation_tolerance_deg)
    )
    return {
        "passed": bool(passed),
        "compared_proposal_count": int(compared),
        "max_correction_translation_gap_m": max_translation,
        "max_correction_rotation_gap_deg": max_rotation,
        "max_raw_pose_translation_gap_m": max_pose_translation,
        "max_raw_pose_rotation_gap_deg": max_pose_rotation,
        "translation_tolerance_m": float(translation_tolerance_m),
        "rotation_tolerance_deg": float(rotation_tolerance_deg),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Stage-2b-shaped run dir; the diagnostics land in its "
        "object_pose_refinement/ subdirectory.",
    )
    parser.add_argument(
        "--check-equivalence",
        action="store_true",
        help="Replay with the source settings and assert the proposals match.",
    )
    parser.add_argument("--equivalence-translation-tolerance-m", type=float, default=1e-3)
    parser.add_argument("--equivalence-rotation-tolerance-deg", type=float, default=0.01)
    for name, flag in OVERRIDE_FLAGS:
        if name == "proposal_mode":
            parser.add_argument(
                f"--{flag}", default=None, choices=sorted(PROPOSAL_MODES)
            )
        else:
            parser.add_argument(f"--{flag}", type=int, default=None)
    return parser.parse_args()


def collect_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Only the flags the caller actually passed become overrides."""

    overrides: dict[str, Any] = {}
    for name, flag in OVERRIDE_FLAGS:
        value = getattr(args, _flag_dest(flag))
        if value is not None:
            overrides[name] = value
    return overrides


def main() -> None:
    args = _parse_args()
    source_path = args.diagnostics.expanduser().resolve()
    source = load_diagnostics(source_path)
    output_dir = args.output_dir.expanduser().resolve()
    refinement_dir = output_dir / "object_pose_refinement"
    refinement_dir.mkdir(parents=True, exist_ok=True)

    overrides = collect_overrides(args)

    frame_ids = tuple(int(value) for value in source["frame_ids"])
    raw_poses = tuple(
        torch.as_tensor(pose).detach().float().cpu()
        for pose in source["raw_camera_to_world"]
    )
    if len(frame_ids) != len(raw_poses):
        raise ValueError("frame_ids and raw_camera_to_world disagree in length.")
    observations_by_frame = observations_from_diagnostics(source)

    collection = source.get("collection") or {}
    tracked_ids = {int(value) for value in collection.get("tracked_instance_ids", ())}
    filter_stats = Counter(
        {
            str(key): int(value)
            for key, value in (collection.get("filter_stats") or {}).items()
        }
    )
    raw_observation_count = int(
        collection.get("raw_observation_count")
        or sum(len(rows) for rows in observations_by_frame.values())
    )

    config = config_from_diagnostics(source, overrides)
    print(
        "replay config: "
        f"max_reference_age_frames={config.max_reference_age_frames} "
        f"anchor_refresh_interval_frames={config.anchor_refresh_interval_frames} "
        f"proposal_mode={config.proposal_mode}"
    )
    print(
        f"source={source_path} frames={len(frame_ids)} "
        f"observations={sum(len(rows) for rows in observations_by_frame.values())}"
    )

    refiner = ObjectPoseLossRefiner(config)
    result = refiner.refine_from_observations(
        frame_ids=frame_ids,
        raw_poses=raw_poses,
        observations_by_frame=observations_by_frame,
        tracked_ids=tracked_ids,
        filter_stats=filter_stats,
        raw_observation_count=raw_observation_count,
    )
    diagnostic_payload = result.feedback_diagnostics
    if diagnostic_payload is None:
        raise RuntimeError("Replay produced no feedback diagnostics.")
    payload = dict(diagnostic_payload)
    payload["replay_provenance"] = {
        "source_diagnostics": str(source_path),
        "overrides": {str(key): value for key, value in sorted(overrides.items())},
        "segmentation_reused": True,
        "gpu_used": False,
    }
    if args.check_equivalence and not overrides:
        comparison = compare_diagnostics(
            source,
            payload,
            translation_tolerance_m=float(args.equivalence_translation_tolerance_m),
            rotation_tolerance_deg=float(args.equivalence_rotation_tolerance_deg),
        )
        payload["replay_equivalence"] = comparison
        if not comparison["passed"]:
            raise RuntimeError(
                "Replay does not reproduce the source proposals; the cached "
                f"observations are not sufficient to rebuild this run: {comparison}"
            )
        print(
            "replay equivalence OK "
            f"(max_t={comparison['max_correction_translation_gap_m']:.2e} m, "
            f"max_r={comparison['max_correction_rotation_gap_deg']:.2e} deg)"
        )
    elif overrides:
        print("equivalence check skipped: overrides were supplied")

    target = refinement_dir / "feedback_diagnostics.pt"
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(f"proposals={len(payload.get('proposals', ()))}")
    print(f"feedback_diagnostics={target}")


if __name__ == "__main__":
    main()
