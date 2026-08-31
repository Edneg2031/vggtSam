#!/usr/bin/env python3
"""Small CPU smoke test for the V2.1 failure-only recovery policy."""

from __future__ import annotations

import torch

from streaming_couping.src.recovery import output_mask_to_stream
from streaming_couping.src.types import RevisitCandidate, SAM3MaskCandidate
from streaming_couping.src.v21_geometry_recovery import (
    V21RecoveryConfig,
    process_v21_sequence,
)


def main() -> None:
    sequence = 4
    tracks = 1
    output_size = (4, 4)
    stream_size = (518, 518)
    output_masks = torch.zeros(sequence, tracks, *output_size, dtype=torch.bool)
    stable = torch.zeros(output_size, dtype=torch.bool)
    stable[1:3, 1:3] = True
    output_masks[0, 0] = stable
    output_masks[2, 0] = stable
    stream_masks = torch.zeros(sequence, tracks, *stream_size, dtype=torch.bool)
    stream_stable = output_mask_to_stream(
        stable,
        source_size=output_size,
        processed_size=stream_size,
        image_mode="crop",
    )
    stream_masks[0, 0] = stream_stable
    stream_masks[2, 0] = stream_stable
    scores = torch.zeros(sequence, tracks)
    scores[0, 0] = 1.0
    scores[2, 0] = 1.0
    world_points = torch.zeros(sequence, *stream_size, 3)
    world_points[..., 2] = 1.0
    confidence = torch.ones(sequence, *stream_size)
    pose = torch.zeros(sequence, 3, 4)
    pose[:, :, :3] = torch.eye(3)
    intrinsics = torch.eye(3).repeat(sequence, 1, 1)
    proposal = RevisitCandidate(
        mask=stable,
        projected_mask=stable,
        supported_mask=stable,
        box_xyxy=(1, 1, 3, 3),
        projected_points=16,
        supported_points=16,
        projected_fraction=1.0,
        support_ratio=1.0,
        accepted=True,
        reason="smoke",
    )

    def build_proposal(frame, slot, memory):
        assert int(memory.world_points.shape[0]) > 0
        return proposal

    def refine(frame, slot, prompt, current_proposal):
        return [SAM3MaskCandidate(obj_id=7, mask=stable, score=1.0)]

    result = process_v21_sequence(
        world_points=world_points,
        confidence=confidence,
        raw_masks_output=output_masks,
        raw_scores=scores,
        raw_masks_stream=stream_masks,
        frame_indices=(0, 1, 2, 3),
        track_ids=(7,),
        track_prompts=("chair",),
        world_to_camera=pose,
        intrinsics=intrinsics,
        source_sizes=(output_size,) * sequence,
        processed_size=stream_size,
        image_mode="crop",
        config=V21RecoveryConfig(
            min_history_points=1,
            min_candidate_support_recall=0.1,
            min_candidate_iou=0.1,
            min_geometry_consistency=0.1,
            replacement_margin=0.05,
        ),
        proposal_builder=build_proposal,
        refine_callback=refine,
    )
    stats = result["stats"]
    assert stats["recovery_trigger_count"] == 2, stats
    assert stats["accepted_recovery_count"] == 2, stats
    assert stats["normal_frame_unchanged_ratio"] == 1.0, stats
    assert stats["raw_fallback_unchanged_ratio"] == 1.0, stats
    print(
        "V2.1 failure-only recovery smoke passed "
        f"triggers={stats['recovery_trigger_count']} "
        f"accepted={stats['accepted_recovery_count']} "
        f"reasons={stats['trigger_reason_counts']}"
    )


if __name__ == "__main__":
    main()
