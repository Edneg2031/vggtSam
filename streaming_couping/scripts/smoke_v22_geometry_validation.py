#!/usr/bin/env python3
"""CPU smoke test for V2.2 world-space candidate validation."""

from __future__ import annotations

import torch

from streaming_couping.src.types import RevisitCandidate
from streaming_couping.src.v2_geometry_recovery import V2ObjectMemoryState
from streaming_couping.src.v22_geometry_validation import (
    V22GeometryValidationConfig,
    validate_v22_candidate,
)


def _state(points: torch.Tensor) -> V2ObjectMemoryState:
    state = V2ObjectMemoryState(
        object_id=0,
        prompt="chair",
        sam_track_id=7,
        max_points=128,
    )
    state.world_points = points.clone().float()
    state.world_weights = torch.ones(points.shape[0])
    state.mask_area_ema = 16.0
    state.last_seen_sequence_index = 3
    return state


def _proposal(mask: torch.Tensor) -> RevisitCandidate:
    return RevisitCandidate(
        mask=mask,
        projected_mask=mask,
        supported_mask=mask,
        box_xyxy=(0, 0, 4, 4),
        projected_points=int(mask.sum()),
        supported_points=int(mask.sum()),
        projected_fraction=1.0,
        support_ratio=1.0,
        accepted=True,
        reason="smoke",
    )


def main() -> None:
    points = torch.tensor(
        [[0.00, 0.00, 1.00], [0.05, 0.00, 1.00],
         [0.00, 0.05, 1.00], [0.05, 0.05, 1.00]]
    ).repeat(4, 1)
    current = points + torch.tensor([0.01, 0.00, 0.00])
    mask = torch.ones(4, 4, dtype=torch.bool)
    row = validate_v22_candidate(
        candidate_row={"recovery_score": 0.90},
        candidate_stream_mask=mask,
        current_world_points=current.reshape(4, 4, 3),
        current_confidence=torch.ones(4, 4),
        memory=_state(points),
        config=V22GeometryValidationConfig(
            min_candidate_points=1,
            min_historical_points=1,
            min_point_overlap_ratio=0.1,
            min_shape_score=0.1,
            final_accept_threshold=0.5,
        ),
        proposal=_proposal(mask),
        candidate_mask=mask,
    )
    assert row["geometry_validation_attempted"] == 1, row
    assert row["accepted"] == 1, row
    assert row["geometry_validation_accepted"] == 1, row

    far = validate_v22_candidate(
        candidate_row={"recovery_score": 0.90},
        candidate_stream_mask=mask,
        current_world_points=(current + torch.tensor([2.0, 0.0, 0.0])).reshape(4, 4, 3),
        current_confidence=torch.ones(4, 4),
        memory=_state(points),
        config=V22GeometryValidationConfig(
            min_candidate_points=1,
            min_historical_points=1,
            centroid_distance_threshold_m=0.1,
            centroid_distance_scale=0.1,
        ),
        proposal=_proposal(mask),
        candidate_mask=mask,
    )
    assert far["accepted"] == 0, far
    assert far["geometry_validation_reason"] == "centroid_far", far
    print(
        "V2.2 world-space validation smoke passed "
        f"accepted={row['geometry_validation_accepted']} "
        f"far_reason={far['geometry_validation_reason']}"
    )


if __name__ == "__main__":
    main()
