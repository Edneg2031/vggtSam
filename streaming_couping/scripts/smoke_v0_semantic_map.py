#!/usr/bin/env python3
"""CPU smoke for frozen raw-pointmap semantic lifting."""

from __future__ import annotations

import torch

from streaming_couping.src.semantic_map import build_semantic_pointmap


def main() -> None:
    points = torch.tensor(
        [
            [
                [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
                [[0.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            ]
        ]
    )
    confidence = torch.tensor([[[[1.0], [2.0]], [[3.0], [4.0]]]])
    masks = torch.zeros(1, 2, 2, 2, dtype=torch.bool)
    masks[0, 0, 0, 0] = True
    masks[0, 1, 0, 0] = True
    masks[0, 1, 1, 1] = True
    scores = torch.tensor([[0.6, 0.9]])
    images = torch.full((1, 3, 2, 2), 0.5)
    result = build_semantic_pointmap(
        world_points=points,
        confidence=confidence,
        masks=masks,
        track_scores=scores,
        images=images,
        confidence_threshold=0.0,
        track_score_threshold=0.5,
        max_map_points=4,
    )
    assert torch.equal(result.world_points, points.reshape(-1, 3))
    assert torch.equal(result.semantic_slots, torch.tensor([1, -1, -1, 1]))
    assert torch.equal(result.sequence_indices, torch.zeros(4, dtype=torch.long))
    assert torch.equal(result.rgb, torch.full((4, 3), 0.5))
    assert bool(result.dense_valid.all())
    print("V0 raw-pointmap semantic lifting smoke passed")


if __name__ == "__main__":
    main()
