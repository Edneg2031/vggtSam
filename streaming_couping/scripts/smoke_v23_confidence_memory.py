#!/usr/bin/env python3
"""CPU smoke test for the V2.3 confidence-aware voxel memory."""

from __future__ import annotations

import torch

from streaming_couping.src.v23_confidence_memory import (
    V23ConfidenceMemoryConfig,
    process_v23_memory_sequence,
)


def main() -> None:
    world_points = torch.tensor(
        [
            [
                [[0.00, 0.00, 1.00], [0.04, 0.00, 1.00]],
                [[0.00, 0.04, 1.00], [0.04, 0.04, 1.00]],
            ],
            [
                [[0.01, 0.00, 1.00], [0.05, 0.00, 1.00]],
                [[0.01, 0.04, 1.00], [0.05, 0.04, 1.00]],
            ],
        ],
        dtype=torch.float32,
    )
    confidence = torch.ones(2, 2, 2)
    masks = torch.ones(2, 1, 2, 2, dtype=torch.bool)
    scores = torch.ones(2, 1)
    events = [
        {"recovery_applied": 0, "triggered": 0},
        {"recovery_applied": 1, "triggered": 1, "reason": "reentry_after_gap"},
    ]
    config = V23ConfidenceMemoryConfig(
        voxel_size_m=0.05,
        max_voxels_per_object=64,
        max_points_per_observation=16,
        point_confidence_floor=0.30,
        min_observation_score=0.50,
        recovery_support_distance_m=0.10,
        min_recovery_support_ratio=0.25,
        min_recovery_supported_points=2,
    )
    result = process_v23_memory_sequence(
        world_points=world_points,
        confidence=confidence,
        masks_stream=masks,
        scores=scores,
        events=events,
        frame_indices=(0, 1),
        track_ids=(7,),
        track_prompts=("chair",),
        config=config,
    )
    stats = result["stats"]
    assert stats["memory_update_count"] == 2, stats
    assert stats["raw_observation_update"] == 1, stats
    assert stats["validated_recovery_update"] == 1, stats
    assert int(result["map_write_mask"].sum()) == 2
    assert result["objects"][0]["recovery_observation_count"] == 1
    assert 0.0 <= float(stats["average_object_confidence"]) <= 1.0
    print(
        "V2.3 confidence-aware memory smoke passed "
        f"updates={stats['memory_update_count']} "
        f"voxels={stats['total_memory_voxels']} "
        f"recovery_updates={stats['validated_recovery_update']}"
    )


if __name__ == "__main__":
    main()
