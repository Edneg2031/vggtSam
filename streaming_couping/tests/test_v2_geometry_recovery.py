from __future__ import annotations

import torch

from streaming_couping.src.types import RevisitCandidate, SAM3MaskCandidate
from streaming_couping.src.v2_geometry_recovery import (
    V2ObjectMemoryState,
    V2RecoveryConfig,
    decide_v2_trigger,
    evaluate_recovery_candidate,
    select_recovery_candidate,
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


def test_v2_trigger_is_conditional() -> None:
    config = V2RecoveryConfig(
        min_history_points=8,
        min_mask_pixels=1,
        reentry_gap=3,
    )
    state = _state()
    mask = torch.ones(4, 4, dtype=torch.bool)
    assert decide_v2_trigger(
        mask=mask,
        score=1.0,
        sequence_index=3,
        memory=state,
        config=config,
    )["triggered"] == 0
    assert decide_v2_trigger(
        mask=torch.zeros_like(mask),
        score=0.0,
        sequence_index=3,
        memory=state,
        config=config,
    )["reason"] == "sam_track_lost"
    assert decide_v2_trigger(
        mask=mask,
        score=1.0,
        sequence_index=6,
        memory=state,
        config=config,
    )["reason"] == "reentry_after_gap"


def test_v2_candidate_gate_and_selection() -> None:
    mask = torch.zeros(8, 8, dtype=torch.bool)
    mask[2:6, 2:6] = True
    proposal = RevisitCandidate(
        mask=mask,
        projected_mask=mask,
        supported_mask=mask,
        box_xyxy=(2, 2, 6, 6),
        projected_points=16,
        supported_points=16,
        projected_fraction=1.0,
        support_ratio=1.0,
        accepted=True,
        reason="test",
    )
    good = SAM3MaskCandidate(obj_id=1, mask=mask, score=0.9)
    bad = SAM3MaskCandidate(obj_id=2, mask=torch.zeros_like(mask), score=1.0)
    config = V2RecoveryConfig(
        min_candidate_score=0.5,
        min_candidate_support_recall=0.5,
        min_candidate_iou=0.2,
    )
    row = evaluate_recovery_candidate(
        good,
        proposal=proposal,
        raw_mask=torch.zeros_like(mask),
        config=config,
    )
    assert row["accepted"] == 1
    selected = select_recovery_candidate(
        [bad, good],
        proposal=proposal,
        raw_mask=torch.zeros_like(mask),
        config=config,
    )
    assert selected is not None
    assert selected["selected"]["candidate"].obj_id == 1
