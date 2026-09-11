#!/usr/bin/env python3
"""ORACLE CONTROL: run the refiner on ground-truth instance masks.

This deliberately breaks the project rule that ground truth never enters
candidate generation.  That is the whole point: the measured proposal error is
0.086 m against a raw ATE of 0.117 m, and two proposal-side interventions
(bounded reference age, rotation-then-translation) failed to move it.  The
remaining question is *where* that error comes from, and only a mask oracle can
split it:

* GT masks + GT identity give a much smaller proposal error
      -> the bottleneck is the segmentation model: identity, mask boundary, or
         re-entry.
* GT masks leave the proposal error roughly unchanged
      -> the bottleneck is downstream of the mask: nearest-neighbour
         correspondence, or the object's degenerate 3D geometry.

Nothing produced here is a method result.  The outputs are diagnostic-only,
carry an explicit audit flag, and must be reported as an oracle upper bound --
never as a branch of the system.  Compare ``prop_err`` between this run and a
real one, nothing else.

It needs no GPU: the geometry comes from the cached HorizonStream run and the
masks from the manifest's ``instance_mask`` labels.

    python -m streaming_couping.scripts.run_object_pose_loss_oracle \
        --geometry-cache <run>/horizonstream_geometry.pt \
        --manifest <manifest.json> --scene-id 00a231a370 \
        --prompts bed wardrobe chair rug dustbin \
        --output-dir <base>.oracle_mask
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from streaming_couping.src.horizonstream_cache import load_horizonstream_cache
from streaming_couping.src.semantic_mapping.adapters import (
    HorizonStreamGeometryCacheAdapter,
)
from streaming_couping.src.semantic_mapping.contracts import (
    ObjectObservation,
    SegmentationFrame,
)
from streaming_couping.src.semantic_mapping.object_pose_loss_refinement import (
    PROPOSAL_MODES,
    ObjectPoseLossRefinementConfig,
    ObjectPoseLossRefiner,
)
from streaming_couping.src.semantic_tracking_metrics import (
    load_ground_truth_instances,
)

ORACLE_MASK_SOURCE = "ground_truth_instance_mask"


def build_config(args: argparse.Namespace) -> ObjectPoseLossRefinementConfig:
    values: dict[str, Any] = {
        "independent_instance_poses": True,
        "export_feedback_diagnostics": True,
        "device": args.device,
        "outer_iterations": args.outer_iterations,
        "optimizer_steps": args.optimizer_steps,
        "learning_rate": args.learning_rate,
        "min_mask_pixels": args.min_mask_pixels,
        "min_geometry_confidence": args.min_geometry_confidence,
        "max_match_distance_m": args.max_match_distance_m,
        "trim_ratio": args.trim_ratio,
        "min_matches_per_pair": args.min_matches_per_pair,
        "min_total_matches": args.min_total_matches,
        "max_correction_rotation_deg": args.max_correction_rotation_deg,
        "max_correction_translation_m": args.max_correction_translation_m,
        "min_relative_loss_improvement": args.min_relative_loss_improvement,
        "max_reference_age_frames": args.max_reference_age_frames,
        "anchor_refresh_interval_frames": args.anchor_refresh_interval_frames,
        "proposal_mode": args.proposal_mode,
    }
    return ObjectPoseLossRefinementConfig(**values).validate()


def build_segmentation_frames(
    *,
    ground_truth: Any,
    frame_ids: Sequence[int],
    image_size: tuple[int, int],
) -> tuple[SegmentationFrame, ...]:
    """One observation per GT instance that is visible in a frame.

    ``score`` and ``static_score`` are set to 1.0: the oracle has no tracker,
    so there is no score to propagate, and leaving them low would let the
    refiner's own thresholds reject the GT masks and make the control
    uninterpretable.
    """

    frames: list[SegmentationFrame] = []
    instance_count = len(ground_truth.instance_ids)
    for index, frame_id in enumerate(frame_ids):
        observations = []
        for target in range(instance_count):
            mask = ground_truth.masks[index, target]
            if not bool(mask.any()):
                continue
            observations.append(
                ObjectObservation(
                    category=str(ground_truth.labels[target]),
                    instance_id=int(ground_truth.instance_ids[target]),
                    mask=mask.detach().bool().cpu(),
                    score=1.0,
                    static_score=1.0,
                )
            )
        frames.append(
            SegmentationFrame(
                frame_id=int(frame_id),
                image_size=(int(image_size[0]), int(image_size[1])),
                observations=tuple(observations),
                backend=ORACLE_MASK_SOURCE,
            )
        )
    return tuple(frames)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry-cache", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--prompts", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--outer-iterations", type=int, default=4)
    parser.add_argument("--optimizer-steps", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--min-mask-pixels", type=int, default=32)
    parser.add_argument("--min-geometry-confidence", type=float, default=0.30)
    parser.add_argument("--max-match-distance-m", type=float, default=0.25)
    parser.add_argument("--trim-ratio", type=float, default=0.70)
    parser.add_argument("--min-matches-per-pair", type=int, default=8)
    parser.add_argument("--min-total-matches", type=int, default=16)
    parser.add_argument("--max-correction-rotation-deg", type=float, default=10.0)
    parser.add_argument("--max-correction-translation-m", type=float, default=0.25)
    parser.add_argument("--min-relative-loss-improvement", type=float, default=0.02)
    parser.add_argument("--max-reference-age-frames", type=int, default=0)
    parser.add_argument("--anchor-refresh-interval-frames", type=int, default=0)
    parser.add_argument(
        "--proposal-mode", default="joint", choices=sorted(PROPOSAL_MODES)
    )
    parser.add_argument(
        "--include-all-instances",
        action="store_true",
        help=(
            "Use every visible GT instance rather than only those matching the "
            "prompts.  Not comparable with a SAM run that used prompts."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cache_path = args.geometry_cache.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    refinement_dir = output_dir / "object_pose_refinement"
    refinement_dir.mkdir(parents=True, exist_ok=True)

    payload = load_horizonstream_cache(cache_path)
    adapter = HorizonStreamGeometryCacheAdapter(payload)
    geometry_frames = adapter.infer(adapter.image_paths)
    frame_ids = tuple(int(frame.frame_id) for frame in geometry_frames)
    print(
        f"geometry frames={len(frame_ids)} size={adapter.image_size} "
        f"cache={cache_path}"
    )

    source_positions = [int(value) for value in payload["source_positions"]]
    if len(source_positions) != len(frame_ids):
        raise ValueError(
            "Cache source_positions and frame count disagree; the GT masks "
            "cannot be aligned to the geometry."
        )
    ground_truth = load_ground_truth_instances(
        args.manifest.expanduser().resolve(),
        scene_id=str(args.scene_id),
        frame_indices=source_positions,
        output_size=adapter.image_size,
        prompts=tuple(args.prompts),
        include_all_instances=bool(args.include_all_instances),
    )
    print(
        f"gt instances={len(ground_truth.instance_ids)} "
        f"labels={sorted(set(ground_truth.labels))} "
        f"visible={len(ground_truth.all_visible_instance_ids)}"
    )
    if not ground_truth.instance_ids:
        raise RuntimeError(
            "No GT instance matched the prompts; the oracle would be empty. "
            "Check --prompts and the manifest's object labels."
        )

    segmentation_frames = build_segmentation_frames(
        ground_truth=ground_truth,
        frame_ids=frame_ids,
        image_size=adapter.image_size,
    )
    frames_with_gt = sum(1 for frame in segmentation_frames if frame.observations)
    print(f"frames carrying a GT observation={frames_with_gt}/{len(frame_ids)}")

    config = build_config(args)
    refiner = ObjectPoseLossRefiner(config)
    result = refiner.refine(geometry_frames, segmentation_frames, adapter.image_paths)
    diagnostics = result.feedback_diagnostics
    if diagnostics is None:
        raise RuntimeError("Oracle run produced no feedback diagnostics.")

    diagnostics = dict(diagnostics)
    diagnostics["oracle"] = {
        "mask_source": ORACLE_MASK_SOURCE,
        "gt_used_for_proposals": True,
        "purpose": "diagnostic_upper_bound_only",
        "not_a_method_result": True,
        "instance_count": int(len(ground_truth.instance_ids)),
        "instance_ids": [int(value) for value in ground_truth.instance_ids],
        "include_all_instances": bool(args.include_all_instances),
        "segmentation_model_bypassed": True,
        "gpu_used": False,
    }
    target = refinement_dir / "feedback_diagnostics.pt"
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    try:
        torch.save(diagnostics, temporary)
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()

    summary = result.summary
    print(f"proposals={len(diagnostics.get('proposals', ()))}")
    print(
        "accepted frames="
        f"{_dig(summary, 'accepted_frame_count')} "
        f"tracked instances={_dig(summary, 'tracked_instance_count')}"
    )
    print(f"feedback_diagnostics={target}")
    print(
        "NOTE: this is an oracle control. Its output is diagnostic-only and "
        "must not be reported as a branch of the system."
    )


def _dig(mapping: Mapping[str, Any] | None, key: str) -> Any:
    if not isinstance(mapping, Mapping):
        return None
    return mapping.get(key)


if __name__ == "__main__":
    main()
