"""Minimal geometry-gated SAM3 recovery used by the geometry pipeline.

This module intentionally contains only the deployable path that survived the
completed SAM ablation: original tracking, one natural geometry-disagreement
event, full-mask candidate selection, and same-ID memory writeback.
"""

from __future__ import annotations

import torch

from .backbones.sam3_wrapper import SAM3Wrapper
from .bridge.gating import binary_iou
from .config import ExperimentConfig
from .recovery import mine_recovery
from .types import GeometrySequence, SAM3MaskCandidate


def run_natural_recovery_tracking(
    config: ExperimentConfig,
    *,
    sequence,
    target_masks: torch.Tensor,
    geometry: GeometrySequence,
    sam3: SAM3Wrapper,
) -> dict:
    """Run the frozen original tracker and its one deployable recovery path."""

    original = sam3.track(
        sequence.image_paths,
        prompt=sequence.label,
        output_size=config.output_size,
        reference_frame_idx=sequence.reference_frame_idx,
        reference_mask=target_masks[sequence.reference_frame_idx],
    )
    mined = mine_recovery(
        config,
        sequence=sequence,
        reference_mask=target_masks[sequence.reference_frame_idx],
        original_masks=original.masks,
        original_scores=original.scores,
        geometry=geometry,
        map_update_policy="joint_reliable",
    )
    event = _select_first_viable_event(
        config,
        sequence=sequence,
        target_masks=target_masks,
        mined=mined,
        sam3=sam3,
    )
    if event["mask"] is None:
        return {
            "original": original,
            "recovered": original,
            "recovery_applied": False,
            "recovery_sequence_index": event["sequence_index"],
            "recovery_frame_index": event["frame_index"],
            "recovery_reason": event["reason"],
            "selected_support_coverage": event["support_coverage"],
            "selected_candidate_gt_iou": event["candidate_gt_iou"],
        }

    recovered = sam3.track_with_recovery_mask_memory(
        sequence.image_paths,
        prompt=sequence.label,
        output_size=config.output_size,
        reference_frame_idx=sequence.reference_frame_idx,
        reference_mask=target_masks[sequence.reference_frame_idx],
        recovery_frame_idx=int(event["sequence_index"]),
        recovery_mask=event["mask"],
    )
    if recovered.selected_obj_id != original.selected_obj_id:
        raise RuntimeError(
            "Same-ID writeback changed the persistent object ID: "
            f"{original.selected_obj_id} -> {recovered.selected_obj_id}."
        )
    return {
        "original": original,
        "recovered": recovered,
        "recovery_applied": True,
        "recovery_sequence_index": event["sequence_index"],
        "recovery_frame_index": event["frame_index"],
        "recovery_reason": event["reason"],
        "selected_support_coverage": event["support_coverage"],
        "selected_candidate_gt_iou": event["candidate_gt_iou"],
    }


def _select_first_viable_event(
    config: ExperimentConfig,
    *,
    sequence,
    target_masks: torch.Tensor,
    mined: dict,
    sam3: SAM3Wrapper,
) -> dict:
    gate_requested = False
    last_reason = "no joint-gate recovery event"
    for index, row in enumerate(mined["rows"]):
        geometry_candidate = mined["candidates"][index]
        if not (
            index > int(sequence.reference_frame_idx)
            and index < len(sequence.frame_indices) - 1
            and row["use_correction"]
        ):
            continue
        gate_requested = True
        if (
            not geometry_candidate.accepted
            or not geometry_candidate.supported_mask.any()
        ):
            last_reason = (
                "joint gate requested recovery but geometry support "
                f"was rejected: {geometry_candidate.reason}"
            )
            continue
        candidates = sam3.propose_text_masks(
            sequence.image_paths[index],
            prompt=sequence.label,
            output_size=config.output_size,
        )
        if not candidates:
            last_reason = "SAM3 global-text query produced no candidate mask"
            continue
        selected = _select_geometry_supported_candidate(
            geometry_candidate,
            candidates,
        )
        support_coverage = _coverage(
            geometry_candidate.supported_mask,
            selected.mask,
        )
        candidate_gt_iou = binary_iou(
            selected.mask,
            target_masks[index],
        )
        if not selected.mask.any():
            last_reason = "selected SAM3 candidate mask is empty"
            continue
        if support_coverage < float(
            config.recovery_min_support_coverage
        ):
            last_reason = (
                "selected full mask failed geometry support coverage: "
                f"{support_coverage:.4f} < "
                f"{config.recovery_min_support_coverage:.4f}"
            )
            continue
        return {
            "sequence_index": index,
            "frame_index": int(sequence.frame_indices[index]),
            "mask": selected.mask.detach().cpu().bool(),
            "support_coverage": support_coverage,
            "candidate_gt_iou": candidate_gt_iou,
            "reason": "accepted full SAM3 candidate selected by aligned geometry",
        }

    return {
        "sequence_index": None,
        "frame_index": None,
        "mask": None,
        "support_coverage": float("nan"),
        "candidate_gt_iou": float("nan"),
        "reason": (
            last_reason
            if gate_requested
            else "no joint-gate recovery event"
        ),
    }


def _select_geometry_supported_candidate(
    geometry_candidate,
    candidates: list[SAM3MaskCandidate],
) -> SAM3MaskCandidate:
    supported = geometry_candidate.supported_mask.bool()
    projected = geometry_candidate.projected_mask.bool()
    coarse = geometry_candidate.mask.bool()
    ranking = [
        (
            _coverage(supported, candidate.mask),
            _coverage(projected, candidate.mask),
            binary_iou(candidate.mask, coarse),
            float(candidate.score),
            -int(candidate.obj_id),
            index,
        )
        for index, candidate in enumerate(candidates)
    ]
    return candidates[int(max(ranking)[-1])]


def _coverage(evidence: torch.Tensor, mask: torch.Tensor) -> float:
    evidence = evidence.detach().cpu().bool()
    mask = mask.detach().cpu().bool()
    denominator = int(evidence.sum())
    if denominator == 0:
        return 0.0
    return float((evidence & mask).sum()) / denominator
