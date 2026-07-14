"""Decision gates for geometry-assisted same-instance SAM3 correction."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..types import RevisitCandidate


@dataclass(frozen=True)
class CorrectionDecision:
    use_correction: bool
    reason: str


def decide_correction(
    *,
    tracker_mask: torch.Tensor,
    tracker_score: float,
    candidate: RevisitCandidate,
    tracker_low_score: float,
    fallback_on_missing_mask: bool,
) -> CorrectionDecision:
    tracker_present = bool(tracker_mask.any())
    tracker_reliable = tracker_present and float(tracker_score) >= float(
        tracker_low_score
    )
    tracker_missing = not tracker_present
    tracker_weak = tracker_missing or not tracker_reliable

    use_correction = (
        candidate.accepted
        and tracker_weak
        and (fallback_on_missing_mask or not tracker_missing)
    )
    if use_correction:
        reason = "weak/missing tracker mask: refine accepted geometry candidate"
    elif tracker_reliable:
        reason = "reliable SAM3 mask"
    elif not candidate.accepted:
        reason = f"weak tracker; geometry rejected: {candidate.reason}"
    else:
        reason = "weak tracker; correction disabled"
    return CorrectionDecision(
        use_correction=use_correction,
        reason=reason,
    )


def binary_iou(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.bool()
    right = right.bool()
    union = (left | right).sum()
    if int(union) == 0:
        return 1.0
    return float((left & right).sum().float() / union.float())
