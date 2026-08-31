"""Failure-only, candidate-scored geometry recovery for V2.1.

V2.1 is intentionally narrower than the original V2 ablation:

* low SAM confidence, small masks, and area outliers are never triggers;
* an empty SAM observation, a persistent missing streak, or a re-entry after a
  gap can trigger a geometry proposal;
* the raw SAM mask remains the default candidate;
* a geometry-guided SAM candidate replaces it only when its combined SAM,
  projection, and temporal score is better by a configured margin.

The foundation models remain frozen.  The callbacks are kept model-agnostic
so the policy can be tested on CPU without loading SAM3.1 or StreamVGGT.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping, Sequence

import torch

from .aggregation.point_map_fusion import sample_masked_observation
from .recovery import output_mask_to_stream
from .types import RevisitCandidate, SAM3MaskCandidate
from .v2_geometry_recovery import (
    V2ObjectMemoryBank,
    V2ObjectMemoryState,
    _normalize_pose_tensor,
)


@dataclass(frozen=True)
class V21RecoveryConfig:
    """Trigger, proposal, and replacement thresholds for V2.1."""

    # Trigger policy.  ``low_score_threshold`` and area thresholds are kept
    # only for config-file compatibility; V2.1 does not use them as triggers.
    unobserved_gap: int = 2
    reentry_gap: int = 3
    low_score_threshold: float = 0.50
    min_mask_pixels: int = 32
    min_area_ratio: float = 0.35
    max_area_ratio: float = 2.85

    # Causal memory policy.
    min_history_points: int = 16
    max_points_per_object: int = 4096
    min_memory_update_score: float = 0.50
    min_geometry_confidence: float = 0.30

    # Frozen SAM3.1 proposal call.
    max_positive_points: int = 6

    # Candidate hard gates.
    min_candidate_score: float = 0.50
    min_candidate_support_recall: float = 0.15
    min_candidate_iou: float = 0.02
    min_candidate_area_ratio: float = 0.20
    max_candidate_area_ratio: float = 5.00
    min_geometry_consistency: float = 0.12
    min_recovery_score: float = 0.50
    replacement_margin: float = 0.12

    # Candidate score = weighted SAM confidence, geometry consistency, and
    # temporal/shape consistency.  The weights are normalized at runtime.
    sam_score_weight: float = 0.35
    geometry_score_weight: float = 0.40
    temporal_score_weight: float = 0.25

    # Projection proposal parameters.
    proposal_box_quantile: float = 0.02
    proposal_box_padding_ratio: float = 0.12
    min_projected_points: int = 8
    min_projected_fraction: float = 0.005
    min_supported_points: int = 8
    min_support_ratio: float = 0.02
    support_abs_distance: float = 0.15
    support_relative_distance: float = 0.10


def decide_v21_trigger(
    *,
    mask: torch.Tensor,
    score: float,
    sequence_index: int,
    memory: V2ObjectMemoryState,
    config: V21RecoveryConfig,
) -> dict[str, Any]:
    """Return a failure-only causal trigger decision.

    A non-empty mask is normal by default, regardless of its SAM score or
    area.  The score is logged but deliberately cannot trigger recovery.
    """

    pixels = int(mask.detach().bool().sum())
    gap = (
        int(sequence_index) - int(memory.last_seen_sequence_index) - 1
        if memory.last_seen_sequence_index >= 0
        else -1
    )
    missing_streak = max(0, gap + 1) if pixels == 0 else max(0, gap)
    base = {
        "triggered": 0,
        "reason": "no_history",
        "raw_mask_pixels": pixels,
        "raw_sam_score": float(score),
        # Keep raw_score as a compatibility alias for existing CSV tooling.
        "raw_score": float(score),
        "gap_since_last_seen": int(gap),
        "missing_streak": int(missing_streak),
        "raw_area_ratio_vs_memory": float("nan"),
    }
    if (
        not memory.has_history
        or int(memory.world_points.shape[0]) < int(config.min_history_points)
    ):
        return base

    if pixels == 0:
        reason = (
            "unobserved_gap"
            if missing_streak >= int(config.unobserved_gap)
            else "sam_track_lost"
        )
        base.update(triggered=1, reason=reason)
        return base

    # Re-entry is defined by a non-empty observation after a missing gap.
    # Ordinary low-confidence or area-outlier observations are not triggers.
    if gap >= int(config.reentry_gap):
        base.update(triggered=1, reason="reentry_after_gap")
        return base

    base["reason"] = "normal_sam_observation"
    return base


def _clamp01(value: float) -> float:
    if not math.isfinite(float(value)):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def _mask_iou(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.detach().cpu().bool()
    right = right.detach().cpu().bool()
    union = int((left | right).sum())
    return float((left & right).sum()) / union if union else 0.0


def _geometry_consistency(
    mask: torch.Tensor,
    proposal: RevisitCandidate,
) -> tuple[float, float, float, float]:
    """Return total geometry score, proposal IoU, support recall, precision."""

    mask = mask.detach().cpu().bool()
    projected = proposal.mask.detach().cpu().bool()
    support = proposal.supported_mask.detach().cpu().bool()
    if not bool(support.any()):
        support = proposal.projected_mask.detach().cpu().bool()
    proposal_iou = _mask_iou(mask, projected)
    support_pixels = int(support.sum())
    mask_pixels = int(mask.sum())
    overlap = int((mask & support).sum())
    recall = overlap / support_pixels if support_pixels else 0.0
    precision = overlap / mask_pixels if mask_pixels else 0.0
    # IoU and support recall identify the projected object; precision keeps a
    # broad, wrong SAM mask from winning merely by covering the proposal.
    score = 0.50 * proposal_iou + 0.30 * recall + 0.20 * precision
    return _clamp01(score), float(proposal_iou), float(recall), float(precision)


def _temporal_consistency(
    mask: torch.Tensor,
    memory: V2ObjectMemoryState,
) -> float:
    """Measure shape/area consistency with the causal object history."""

    pixels = int(mask.detach().bool().sum())
    expected = float(memory.mask_area_ema)
    if pixels <= 0 or expected <= 0.0:
        return 0.0 if pixels <= 0 else 0.5
    ratio = (float(pixels) + 1.0) / (expected + 1.0)
    # exp(-abs(log ratio)) is symmetric for over- and under-segmentation and
    # remains in [0,1].  It does not compare masks across moving cameras.
    return _clamp01(math.exp(-abs(math.log(max(ratio, 1e-6)))))


def _weighted_score(
    *,
    sam_confidence: float,
    geometry: float,
    temporal: float,
    config: V21RecoveryConfig,
) -> float:
    weights = (
        max(0.0, float(config.sam_score_weight)),
        max(0.0, float(config.geometry_score_weight)),
        max(0.0, float(config.temporal_score_weight)),
    )
    total = sum(weights)
    if total <= 0.0:
        raise ValueError("V2.1 candidate score weights must sum to > 0.")
    return _clamp01(
        (
            weights[0] * _clamp01(sam_confidence)
            + weights[1] * _clamp01(geometry)
            + weights[2] * _clamp01(temporal)
        )
        / total
    )


def score_v21_raw_mask(
    *,
    raw_mask: torch.Tensor,
    raw_sam_score: float,
    proposal: RevisitCandidate,
    memory: V2ObjectMemoryState,
    config: V21RecoveryConfig,
) -> dict[str, float]:
    """Score the unchanged raw SAM candidate against the recovery proposal."""

    geometry, proposal_iou, support_recall, support_precision = (
        _geometry_consistency(raw_mask, proposal)
    )
    temporal = _temporal_consistency(raw_mask, memory)
    score = _weighted_score(
        sam_confidence=raw_sam_score if bool(raw_mask.any()) else 0.0,
        geometry=geometry,
        temporal=temporal,
        config=config,
    )
    return {
        "selection_score": float(score),
        "sam_confidence": _clamp01(raw_sam_score)
        if bool(raw_mask.any())
        else 0.0,
        "geometry_consistency": float(geometry),
        "temporal_consistency": float(temporal),
        "proposal_iou": float(proposal_iou),
        "support_recall": float(support_recall),
        "support_precision": float(support_precision),
    }


def evaluate_v21_candidate(
    candidate: SAM3MaskCandidate,
    *,
    proposal: RevisitCandidate,
    raw_mask: torch.Tensor,
    raw_sam_score: float,
    memory: V2ObjectMemoryState,
    config: V21RecoveryConfig,
) -> dict[str, Any]:
    """Score one recovery candidate and apply the replacement margin gate."""

    mask = candidate.mask.detach().cpu().bool()
    candidate_pixels = int(mask.sum())
    proposal_pixels = int(proposal.mask.detach().cpu().bool().sum())
    geometry, proposal_iou, support_recall, support_precision = (
        _geometry_consistency(mask, proposal)
    )
    temporal = _temporal_consistency(mask, memory)
    recovery_score = _weighted_score(
        sam_confidence=float(candidate.score),
        geometry=geometry,
        temporal=temporal,
        config=config,
    )
    raw_metrics = score_v21_raw_mask(
        raw_mask=raw_mask,
        raw_sam_score=raw_sam_score,
        proposal=proposal,
        memory=memory,
        config=config,
    )
    raw_pixels = int(raw_mask.detach().bool().sum())
    area_ratio = candidate_pixels / max(proposal_pixels, 1)
    raw_area_ratio = (
        candidate_pixels / raw_pixels if raw_pixels else float("nan")
    )
    checks = {
        "nonempty": candidate_pixels > 0,
        "sam_confidence": float(candidate.score)
        >= float(config.min_candidate_score),
        "geometry_consistency": geometry
        >= float(config.min_geometry_consistency),
        "support_recall": support_recall
        >= float(config.min_candidate_support_recall),
        "proposal_iou": proposal_iou >= float(config.min_candidate_iou),
        "area_ratio": (
            float(config.min_candidate_area_ratio)
            <= area_ratio
            <= float(config.max_candidate_area_ratio)
        ),
        "minimum_recovery_score": recovery_score
        >= float(config.min_recovery_score),
        "replacement_margin": recovery_score
        >= raw_metrics["selection_score"] + float(config.replacement_margin),
    }
    accepted = all(checks.values())
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "candidate": candidate,
        "candidate_pixels": candidate_pixels,
        "candidate_score": float(candidate.score),
        "recovery_score": float(recovery_score),
        "raw_selection_score": float(raw_metrics["selection_score"]),
        "score_margin": float(recovery_score - raw_metrics["selection_score"]),
        "recovery_geometry_consistency": float(geometry),
        "recovery_temporal_consistency": float(temporal),
        "raw_geometry_consistency": float(raw_metrics["geometry_consistency"]),
        "raw_temporal_consistency": float(raw_metrics["temporal_consistency"]),
        "raw_sam_confidence": float(raw_metrics["sam_confidence"]),
        "proposal_pixels": proposal_pixels,
        "support_pixels": int(proposal.supported_mask.detach().cpu().bool().sum()),
        "proposal_iou": float(proposal_iou),
        "support_recall": float(support_recall),
        "support_precision": float(support_precision),
        "candidate_area_ratio": float(area_ratio),
        "candidate_area_ratio_vs_raw": float(raw_area_ratio),
        "accepted": int(accepted),
        "reason": "accepted" if accepted else "rejected:" + ",".join(failed),
    }


def select_v21_candidate(
    candidates: Sequence[SAM3MaskCandidate],
    *,
    proposal: RevisitCandidate,
    raw_mask: torch.Tensor,
    raw_sam_score: float,
    memory: V2ObjectMemoryState,
    config: V21RecoveryConfig,
) -> dict[str, Any]:
    """Select a candidate only when it clearly beats the raw mask."""

    evaluated = [
        evaluate_v21_candidate(
            candidate,
            proposal=proposal,
            raw_mask=raw_mask,
            raw_sam_score=raw_sam_score,
            memory=memory,
            config=config,
        )
        for candidate in candidates
    ]
    accepted = [row for row in evaluated if int(row["accepted"])]
    if not accepted:
        raw_metrics = score_v21_raw_mask(
            raw_mask=raw_mask,
            raw_sam_score=raw_sam_score,
            proposal=proposal,
            memory=memory,
            config=config,
        )
        return {
            "selected": None,
            "evaluated": evaluated,
            "reason": "no_candidate" if not evaluated else "raw_preserved",
            "raw_metrics": raw_metrics,
        }
    selected = max(
        accepted,
        key=lambda row: (
            float(row["recovery_score"]),
            float(row["recovery_geometry_consistency"]),
            float(row["candidate_score"]),
        ),
    )
    return {
        "selected": selected,
        "evaluated": evaluated,
        "reason": "accepted_recovery_candidate",
        "raw_metrics": {
            "selection_score": float(selected["raw_selection_score"]),
            "geometry_consistency": float(selected["raw_geometry_consistency"]),
            "temporal_consistency": float(selected["raw_temporal_consistency"]),
            "sam_confidence": float(selected["raw_sam_confidence"]),
        },
    }


def process_v21_sequence(
    *,
    world_points: torch.Tensor,
    confidence: torch.Tensor,
    raw_masks_output: torch.Tensor,
    raw_scores: torch.Tensor,
    raw_masks_stream: torch.Tensor,
    frame_indices: Sequence[int],
    track_ids: Sequence[int],
    track_prompts: Sequence[str],
    world_to_camera: torch.Tensor,
    intrinsics: torch.Tensor,
    source_sizes: Sequence[Sequence[int]],
    processed_size: Sequence[int],
    image_mode: str,
    config: V21RecoveryConfig,
    proposal_builder: Callable[
        [int, int, V2ObjectMemoryState], RevisitCandidate
    ],
    refine_callback: Callable[
        [int, int, str, RevisitCandidate], Sequence[SAM3MaskCandidate]
    ],
    candidate_validator: Callable[..., Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run V2.1 over a frozen V0 cache.

    ``candidate_validator`` is optional so V2.1 remains unchanged.  V2.2
    supplies a world-space validator that runs after the 2-D candidate margin
    gate and before the candidate is applied.
    """

    raw_masks_output = raw_masks_output.detach().cpu().bool()
    raw_scores = raw_scores.detach().cpu().float()
    raw_masks_stream = raw_masks_stream.detach().cpu().bool()
    world_points = world_points.detach().float().cpu()
    confidence = confidence.detach().float().cpu()
    world_to_camera = _normalize_pose_tensor(world_to_camera)
    intrinsics = intrinsics.detach().float().cpu()
    sequence, tracks = raw_masks_output.shape[:2]
    if raw_masks_output.ndim != 4:
        raise ValueError("V2.1 output masks must have shape [S,K,H,W].")
    if tuple(raw_scores.shape) != (sequence, tracks):
        raise ValueError("V2.1 raw scores do not match output masks.")
    if tuple(raw_masks_stream.shape[:2]) != (sequence, tracks):
        raise ValueError("V2.1 stream masks do not match output masks.")
    if world_points.ndim != 4 or world_points.shape[-1] != 3:
        raise ValueError("V2.1 world_points must have shape [S,H,W,3].")
    if confidence.shape != world_points.shape[:3]:
        raise ValueError("V2.1 confidence does not match world_points.")
    if world_points.shape[0] != sequence:
        raise ValueError("V2.1 pointmap/frame count mismatch.")
    if world_to_camera.shape != (sequence, 3, 4):
        raise ValueError("V2.1 world_to_camera must have shape [S,3,4].")
    if intrinsics.shape != (sequence, 3, 3):
        raise ValueError("V2.1 intrinsics must have shape [S,3,3].")
    if len(frame_indices) != sequence or len(source_sizes) != sequence:
        raise ValueError("V2.1 frame metadata length mismatch.")
    if len(track_ids) != tracks or len(track_prompts) != tracks:
        raise ValueError("V2.1 track metadata length mismatch.")
    if tuple(raw_masks_stream.shape[-2:]) != tuple(world_points.shape[1:3]):
        raise ValueError("V2.1 stream masks and pointmap grids disagree.")

    bank = V2ObjectMemoryBank(
        track_ids=track_ids,
        prompts=track_prompts,
        max_points_per_object=config.max_points_per_object,
    )
    final_output = raw_masks_output.clone()
    final_scores = raw_scores.clone()
    final_stream = raw_masks_stream.clone()
    history_masks_output = torch.zeros_like(final_output)
    events: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    applied_by_slot: Counter[int] = Counter()
    trigger_by_slot: Counter[int] = Counter()
    reentry_by_slot: Counter[int] = Counter()
    reentry_applied_by_slot: Counter[int] = Counter()

    for frame in range(sequence):
        for slot in range(tracks):
            memory = bank.get(slot)
            raw_mask = raw_masks_output[frame, slot]
            raw_sam_score = float(raw_scores[frame, slot])
            decision = decide_v21_trigger(
                mask=raw_mask,
                score=raw_sam_score,
                sequence_index=frame,
                memory=memory,
                config=config,
            )
            triggered = bool(decision["triggered"])
            trigger_reason = str(decision["reason"])
            if triggered:
                trigger_by_slot[slot] += 1
                reason_counts[trigger_reason] += 1
                if trigger_reason == "reentry_after_gap":
                    reentry_by_slot[slot] += 1

            default_raw_score = _weighted_score(
                sam_confidence=raw_sam_score if bool(raw_mask.any()) else 0.0,
                geometry=0.0,
                temporal=_temporal_consistency(raw_mask, memory),
                config=config,
            )
            event: dict[str, Any] = {
                "sequence_index": int(frame),
                "frame_index": int(frame_indices[frame]),
                "slot": int(slot),
                "object_id": int(slot),
                "sam_track_id": int(track_ids[slot]),
                "prompt": str(track_prompts[slot]),
                **decision,
                "trigger_reason": trigger_reason,
                "recovery_applied": 0,
                "recovery_rejected": 0,
                "raw_mask_preserved": 1,
                "proposal_available": 0,
                "proposal_reason": "not_triggered",
                "candidate_count": 0,
                "candidate_score": float("nan"),
                "raw_selection_score": float(default_raw_score),
                "recovery_selection_score": float("nan"),
                "score_margin": float("nan"),
                "raw_geometry_consistency": float("nan"),
                "raw_temporal_consistency": float(
                    _temporal_consistency(raw_mask, memory)
                ),
                "recovery_geometry_consistency": float("nan"),
                "recovery_temporal_consistency": float("nan"),
                "proposal_iou": float("nan"),
                "support_recall": float("nan"),
                "support_precision": float("nan"),
                "candidate_area_ratio": float("nan"),
                "candidate_area_ratio_vs_raw": float("nan"),
                "recovery_reason": trigger_reason,
                "normal_frame_unchanged": int(not triggered),
            }
            source = "raw_sam"
            if triggered:
                proposal: RevisitCandidate | None
                try:
                    proposal = proposal_builder(frame, slot, memory)
                    event["proposal_available"] = int(
                        bool(proposal.accepted and proposal.mask.any())
                    )
                    event["proposal_reason"] = str(proposal.reason)
                    event["projected_points"] = int(proposal.projected_points)
                    event["supported_points"] = int(proposal.supported_points)
                    event["projected_fraction"] = float(proposal.projected_fraction)
                    event["support_ratio"] = float(proposal.support_ratio)
                except Exception as exc:  # raw V0 remains the fallback
                    proposal = None
                    event["proposal_reason"] = f"proposal_error:{type(exc).__name__}"
                    event["recovery_reason"] = event["proposal_reason"]

                if proposal is not None and bool(proposal.accepted and proposal.mask.any()):
                    try:
                        prompt = str(track_prompts[slot]) or "object"
                        candidates = list(refine_callback(frame, slot, prompt, proposal))
                        event["candidate_count"] = int(len(candidates))
                        selected = select_v21_candidate(
                            candidates,
                            proposal=proposal,
                            raw_mask=raw_mask,
                            raw_sam_score=raw_sam_score,
                            memory=memory,
                            config=config,
                        )
                        raw_metrics = selected.get("raw_metrics", {})
                        event.update(
                            {
                                "raw_selection_score": float(
                                    raw_metrics.get(
                                        "selection_score",
                                        event["raw_selection_score"],
                                    )
                                ),
                                "raw_geometry_consistency": float(
                                    raw_metrics.get(
                                        "geometry_consistency",
                                        raw_metrics.get(
                                            "raw_geometry_consistency", float("nan")
                                        ),
                                    )
                                ),
                                "raw_temporal_consistency": float(
                                    raw_metrics.get(
                                        "temporal_consistency",
                                        raw_metrics.get(
                                            "raw_temporal_consistency",
                                            event["raw_temporal_consistency"],
                                        ),
                                    )
                                ),
                            }
                        )
                        if selected.get("selected") is not None:
                            row = selected["selected"]
                            candidate = row["candidate"]
                            candidate_mask = candidate.mask.detach().cpu().bool()
                            candidate_stream = output_mask_to_stream(
                                candidate.mask,
                                source_size=tuple(
                                    int(value) for value in source_sizes[frame]
                                ),
                                processed_size=tuple(
                                    int(value) for value in processed_size
                                ),
                                image_mode=str(image_mode),
                            )
                            event.update(
                                {
                                    "candidate_score": float(row["candidate_score"]),
                                    "recovery_selection_score": float(
                                        row["recovery_score"]
                                    ),
                                    "score_margin": float(row["score_margin"]),
                                    "recovery_geometry_consistency": float(
                                        row["recovery_geometry_consistency"]
                                    ),
                                    "recovery_temporal_consistency": float(
                                        row["recovery_temporal_consistency"]
                                    ),
                                    "proposal_iou": float(row["proposal_iou"]),
                                    "support_recall": float(row["support_recall"]),
                                    "support_precision": float(
                                        row["support_precision"]
                                    ),
                                    "candidate_area_ratio": float(
                                        row["candidate_area_ratio"]
                                    ),
                                    "candidate_area_ratio_vs_raw": float(
                                        row["candidate_area_ratio_vs_raw"]
                                    ),
                                    "selected_gate_reason": str(row["reason"]),
                                }
                            )
                            validation: dict[str, Any] = {
                                "accepted": 1,
                                "geometry_validation_reason": "not_requested",
                            }
                            if candidate_validator is not None:
                                try:
                                    validation = dict(
                                        candidate_validator(
                                            frame=frame,
                                            slot=slot,
                                            memory=memory,
                                            proposal=proposal,
                                            candidate_row=row,
                                            candidate_mask=candidate_mask,
                                            candidate_stream_mask=candidate_stream,
                                            current_world_points=world_points[frame],
                                            current_confidence=confidence[frame],
                                        )
                                    )
                                except Exception as exc:
                                    validation = {
                                        "accepted": 0,
                                        "geometry_validation_reason": (
                                            f"validation_error:{type(exc).__name__}"
                                        ),
                                    }
                            for key, value in validation.items():
                                if key != "accepted":
                                    event[str(key)] = value
                            if bool(validation.get("accepted", 0)):
                                final_output[frame, slot] = candidate_mask
                                final_scores[frame, slot] = float(candidate.score)
                                final_stream[frame, slot] = candidate_stream
                                source = "geometry_recovery"
                                applied_by_slot[slot] += 1
                                if trigger_reason == "reentry_after_gap":
                                    reentry_applied_by_slot[slot] += 1
                                event.update(
                                    {
                                        "recovery_applied": 1,
                                        "raw_mask_preserved": 0,
                                        "recovery_reason": "candidate_beats_raw",
                                        "normal_frame_unchanged": 0,
                                    }
                                )
                            else:
                                event["recovery_rejected"] = 1
                                event["recovery_reason"] = (
                                    "geometry_validation_rejected:"
                                    + str(
                                        validation.get(
                                            "geometry_validation_reason",
                                            "rejected",
                                        )
                                    )
                                )
                        else:
                            event["recovery_rejected"] = 1
                            event["recovery_reason"] = str(selected.get("reason", "raw_preserved"))
                            evaluated = selected.get("evaluated", [])
                            if evaluated:
                                best = max(
                                    evaluated,
                                    key=lambda row: float(row.get("recovery_score", 0.0)),
                                )
                                event.update(
                                    {
                                        "candidate_score": float(
                                            best.get("candidate_score", float("nan"))
                                        ),
                                        "recovery_selection_score": float(
                                            best.get("recovery_score", float("nan"))
                                        ),
                                        "score_margin": float(
                                            best.get("score_margin", float("nan"))
                                        ),
                                        "recovery_geometry_consistency": float(
                                            best.get(
                                                "recovery_geometry_consistency", float("nan")
                                            )
                                        ),
                                        "recovery_temporal_consistency": float(
                                            best.get(
                                                "recovery_temporal_consistency", float("nan")
                                            )
                                        ),
                                    }
                                )
                    except Exception as exc:  # raw V0 remains the fallback
                        event["recovery_rejected"] = 1
                        event["recovery_reason"] = (
                            f"sam_refinement_error:{type(exc).__name__}"
                        )
                else:
                    event["recovery_rejected"] = 1

            final_mask = final_output[frame, slot]
            final_score = float(final_scores[frame, slot])
            history_masks_output[frame, slot] = final_mask
            unsafe_empty_fallback = (
                triggered
                and source != "geometry_recovery"
                and trigger_reason in {"sam_track_lost", "unobserved_gap"}
            )
            can_update = bool(final_mask.any()) and final_score >= float(
                config.min_memory_update_score
            ) and not unsafe_empty_fallback
            memory_updated = False
            if can_update:
                points, weights = sample_masked_observation(
                    world_points[frame],
                    confidence[frame],
                    final_stream[frame, slot]
                    & (confidence[frame] >= float(config.min_geometry_confidence)),
                    max_points=int(config.max_points_per_object),
                )
                if points.numel():
                    memory.update(
                        points=points,
                        weights=weights,
                        mask=final_mask,
                        score=final_score,
                        sequence_index=frame,
                        frame_index=int(frame_indices[frame]),
                        camera=world_to_camera[frame],
                        source=source,
                        trigger_reason=trigger_reason,
                        recovery_applied=bool(event["recovery_applied"]),
                    )
                    memory_updated = True
            event["memory_updated"] = int(memory_updated)
            event["memory_point_count"] = int(memory.world_points.shape[0])
            event["memory_observation_count"] = int(memory.observation_count)
            event["last_seen_sequence_index"] = int(memory.last_seen_sequence_index)
            events.append(event)

    normal_events = [row for row in events if not bool(row["triggered"])]
    raw_fallback_events = [row for row in events if bool(row["triggered"]) and not bool(row["recovery_applied"])]
    stats = {
        "frame_count": int(sequence),
        "slot_count": int(tracks),
        "event_count": int(len(events)),
        "recovery_trigger_count": int(sum(trigger_by_slot.values())),
        "recovery_success_count": int(sum(applied_by_slot.values())),
        "accepted_recovery_count": int(sum(applied_by_slot.values())),
        "recovery_reject_count": int(
            sum(int(row["recovery_rejected"]) for row in events)
        ),
        "candidate_selection_attempt_count": int(
            sum(int(row["candidate_count"]) > 0 for row in events)
        ),
        "raw_fallback_count": int(len(raw_fallback_events)),
        "raw_fallback_unchanged_count": int(
            sum(int(row["raw_mask_preserved"]) for row in raw_fallback_events)
        ),
        "raw_fallback_unchanged_ratio": _ratio(
            sum(int(row["raw_mask_preserved"]) for row in raw_fallback_events),
            len(raw_fallback_events),
        ),
        "normal_frame_count": int(len(normal_events)),
        "normal_frame_unchanged_count": int(
            sum(int(row["normal_frame_unchanged"]) for row in normal_events)
        ),
        "normal_frame_unchanged_ratio": _ratio(
            sum(int(row["normal_frame_unchanged"]) for row in normal_events),
            len(normal_events),
        ),
        "trigger_reason_counts": dict(sorted(reason_counts.items())),
        "per_object_trigger_count": {
            str(slot): int(trigger_by_slot.get(slot, 0)) for slot in range(tracks)
        },
        "per_object_recovery_count": {
            str(slot): int(applied_by_slot.get(slot, 0)) for slot in range(tracks)
        },
        "per_object_reentry_count": {
            str(slot): int(reentry_by_slot.get(slot, 0)) for slot in range(tracks)
        },
        "per_object_reentry_recovery_count": {
            str(slot): int(reentry_applied_by_slot.get(slot, 0))
            for slot in range(tracks)
        },
    }
    return {
        "masks_output": final_output,
        "scores": final_scores,
        "masks_stream": final_stream,
        "history_masks_output": history_masks_output,
        "events": events,
        "objects": bank.metadata(),
        "object_tensors": bank.point_tensors(),
        "stats": stats,
    }


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if int(denominator) else 1.0


__all__ = [
    "V21RecoveryConfig",
    "decide_v21_trigger",
    "score_v21_raw_mask",
    "evaluate_v21_candidate",
    "select_v21_candidate",
    "process_v21_sequence",
]
