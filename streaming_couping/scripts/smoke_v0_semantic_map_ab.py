#!/usr/bin/env python3
"""CPU smoke for shared-support raw/QK semantic point-map construction."""

from __future__ import annotations

import torch

from streaming_couping.src.semantic_map import (
    SemanticMapInputs,
    apply_similarity,
    build_shared_semantic_maps,
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
    print("V0 shared-support semantic map smoke passed")


if __name__ == "__main__":
    main()
