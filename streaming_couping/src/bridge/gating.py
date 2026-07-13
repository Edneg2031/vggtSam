"""Decision gates for geometry-assisted SAM3 re-segmentation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..types import RevisitCandidate


@dataclass(frozen=True)
class BridgeDecision:
    use_tracker: bool
    use_fallback: bool
    update_map: bool
    reason: str


def decide_bridge_action(
    *,
    tracker_mask: torch.Tensor,
    tracker_score: float,
    candidate: RevisitCandidate,
    tracker_low_score: float,
    fallback_on_missing_mask: bool,
    allow_map_update: bool,
    tracker_candidate_iou: float,
    map_update_min_iou: float,
) -> BridgeDecision:
    tracker_present = bool(tracker_mask.any())
    tracker_reliable = tracker_present and float(tracker_score) >= float(
        tracker_low_score
    )
    tracker_missing = not tracker_present
    tracker_weak = tracker_missing or not tracker_reliable

    use_fallback = (
        candidate.accepted
        and tracker_weak
        and (fallback_on_missing_mask or not tracker_missing)
    )
    update_map = (
        allow_map_update
        and tracker_reliable
        and candidate.accepted
        and tracker_candidate_iou >= float(map_update_min_iou)
    )
    if use_fallback:
        reason = "weak/missing tracker mask: refine accepted geometry candidate"
    elif tracker_reliable:
        reason = "reliable SAM3 mask"
    elif not candidate.accepted:
        reason = f"weak tracker; geometry rejected: {candidate.reason}"
    else:
        reason = "weak tracker; fallback disabled"
    return BridgeDecision(
        use_tracker=not use_fallback,
        use_fallback=use_fallback,
        update_map=update_map,
        reason=reason,
    )


def binary_iou(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.bool()
    right = right.bool()
    union = (left | right).sum()
    if int(union) == 0:
        return 1.0
    return float((left & right).sum().float() / union.float())
