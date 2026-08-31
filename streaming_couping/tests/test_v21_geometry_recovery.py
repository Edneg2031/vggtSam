from __future__ import annotations

import torch

from streaming_couping.src.types import RevisitCandidate, SAM3MaskCandidate
from streaming_couping.src.v2_geometry_recovery import V2ObjectMemoryState
from streaming_couping.src.v21_geometry_recovery import (
    V21RecoveryConfig,
    decide_v21_trigger,
    evaluate_v21_candidate,
)


def _state() -> V2ObjectMemoryState:
    state = V2ObjectMemoryState(
        object_id=0,
        prompt="chair",
        sam_track_id=11,
        max_points=32,
    )
    state.world_points = torch.ones(16, 3)
    state.world_weights = torch.ones(16)
    state.last_seen_sequence_index = 2
    state.mask_area_ema = 16.0
    return state


def _proposal(mask: torch.Tensor) -> RevisitCandidate:
    return RevisitCandidate(
        mask=mask,
        projected_mask=mask,
        supported_mask=mask,
        box_xyxy=(2, 2, 6, 6),
        projected_points=int(mask.sum()),
        supported_points=int(mask.sum()),
        projected_fraction=1.0,
        support_ratio=1.0,
        accepted=True,
        reason="test",
    )


def test_v21_does_not_trigger_low_score_or_area_outlier() -> None:
    state = _state()
    mask = torch.ones(4, 4, dtype=torch.bool)
    config = V21RecoveryConfig(min_history_points=8, reentry_gap=3)
    low = decide_v21_trigger(
        mask=mask,
        score=0.0,
        sequence_index=3,
        memory=state,
        config=config,
    )
    assert low["triggered"] == 0
    assert low["reason"] == "normal_sam_observation"

    state.mask_area_ema = 1.0
    outlier = decide_v21_trigger(
        mask=mask,
        score=1.0,
        sequence_index=3,
        memory=state,
        config=config,
    )
    assert outlier["triggered"] == 0


def test_v21_triggers_failure_and_reentry() -> None:
    state = _state()
    config = V21RecoveryConfig(min_history_points=8, unobserved_gap=2, reentry_gap=3)
    empty = decide_v21_trigger(
        mask=torch.zeros(4, 4, dtype=torch.bool),
        score=0.0,
        sequence_index=3,
        memory=state,
        config=config,
    )
    assert empty["triggered"] == 1
    assert empty["reason"] == "sam_track_lost"

    mask = torch.ones(4, 4, dtype=torch.bool)
    reentry = decide_v21_trigger(
        mask=mask,
        score=1.0,
        sequence_index=6,
        memory=state,
        config=config,
    )
    assert reentry["triggered"] == 1
    assert reentry["reason"] == "reentry_after_gap"


def test_v21_candidate_requires_margin_over_raw() -> None:
    state = _state()
    mask = torch.zeros(8, 8, dtype=torch.bool)
    mask[2:6, 2:6] = True
    proposal = _proposal(mask)
    candidate = SAM3MaskCandidate(obj_id=1, mask=mask, score=0.9)
    config = V21RecoveryConfig(
        min_candidate_score=0.5,
        min_candidate_support_recall=0.5,
        min_candidate_iou=0.2,
        min_geometry_consistency=0.1,
        min_recovery_score=0.5,
        replacement_margin=0.05,
    )
    row = evaluate_v21_candidate(
        candidate,
        proposal=proposal,
        raw_mask=torch.zeros_like(mask),
        raw_sam_score=0.0,
        memory=state,
        config=config,
    )
    assert row["accepted"] == 1
    assert row["recovery_score"] > row["raw_selection_score"]
