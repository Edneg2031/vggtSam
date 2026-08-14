#!/usr/bin/env python3
"""CPU smoke for shared-support QK joint semantic-map scoring."""

from __future__ import annotations

import torch

from streaming_couping.src.semantic_map import (
    SemanticMapInputs,
    apply_similarity,
    build_shared_semantic_maps,
)
from streaming_couping.scripts.run_v0_semantic_map_ab import (
    _per_instance_rows,
    _score_family,
)


def main() -> None:
    depth = torch.ones(1, 2, 2, 1)
    confidence = torch.tensor([[[[1.0], [2.0]], [[3.0], [4.0]]]])
    intrinsics = torch.eye(3)[None]
    raw_pose = torch.eye(4)[None, :3]
    selected_pose = raw_pose.clone()
    selected_pose[0, 0, 3] = 1.0
    masks = torch.zeros(1, 2, 2, 2, dtype=torch.bool)
    masks[0, 0, 0, 0] = True
    masks[0, 1, 0, 0] = True
    masks[0, 1, 1, 1] = True
    scores = torch.tensor([[0.6, 0.9]])
    images = torch.full((1, 3, 2, 2), 0.5)
    pair = build_shared_semantic_maps(
        SemanticMapInputs(
            depth=depth,
            confidence=confidence,
            intrinsics=intrinsics,
            raw_world_to_camera=raw_pose,
            selected_world_to_camera=selected_pose,
            masks=masks,
            track_scores=scores,
            images=images,
        ),
        confidence_threshold=0.0,
        track_score_threshold=0.5,
        max_map_points=4,
    )
    assert pair.raw_map_points.shape == (4, 3)
    assert torch.equal(pair.map_semantic_slots, torch.tensor([1, -1, -1, 1]))
    expected_shift = torch.tensor([-1.0, 0.0, 0.0]).expand(4, 3)
    assert torch.allclose(
        pair.selected_map_points - pair.raw_map_points,
        expected_shift,
        atol=1e-6,
    )
    transformed = apply_similarity(
        pair.raw_map_points,
        scale=2.0,
        rotation=torch.eye(3),
        translation=torch.tensor([1.0, 2.0, 3.0]),
    )
    assert torch.allclose(
        transformed,
        2.0 * pair.raw_map_points + torch.tensor([1.0, 2.0, 3.0]),
    )
    assert torch.equal(pair.map_rgb, torch.full((4, 3), 0.5))

    target = torch.tensor(
        [
            [[[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
             [[0.0, 1.0, 1.0], [1.0, 1.0, 1.0]]],
            [[[0.0, 0.0, 2.0], [1.0, 0.0, 2.0]],
             [[0.0, 1.0, 2.0], [1.0, 1.0, 2.0]]],
        ]
    )
    valid = torch.ones(2, 2, 2, dtype=torch.bool)
    slots = torch.zeros(2, 2, 2, dtype=torch.long)
    flat_indices = torch.arange(8)
    metrics, rows, _ = _score_family(
        family="smoke",
        branches={
            "raw": target + 0.10,
            "candidate": target + 0.02,
        },
        raw_branch="raw",
        frames=(90, 105),
        reference=0,
        target=target,
        valid=valid,
        semantic_slots=slots,
        map_flat_indices=flat_indices,
        map_semantic_slots=slots.reshape(-1),
        max_points=8,
        thresholds=(0.05, 0.10),
    )
    assert len(rows) == 4
    assert metrics["candidate"]["overall_pass"] == 1
    assert metrics["candidate"]["semantic_pass"] == 1
    instance_rows = _per_instance_rows(
        families={
            "smoke": {
                "raw": target + 0.10,
                "candidate": target + 0.02,
            }
        },
        family_raw={"smoke": "raw"},
        tracks=[
            {
                "slot": 0,
                "sam_track_id": 7,
                "prompt": "chair",
                "birth_sequence_index": 0,
                "birth_frame": 90,
            }
        ],
        frames=(90, 105),
        reference=0,
        target=target,
        valid=valid,
        semantic_slots=slots,
        map_flat_indices=flat_indices,
        map_semantic_slots=slots.reshape(-1),
        max_points=8,
        thresholds=(0.05, 0.10),
    )
    assert len(instance_rows) == 2
    assert instance_rows[1]["instance_pass"] == 1
    print("V0 QK joint semantic map smoke passed")


if __name__ == "__main__":
    main()
