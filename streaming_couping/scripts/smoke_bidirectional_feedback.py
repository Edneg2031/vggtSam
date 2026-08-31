#!/usr/bin/env python3
"""Small CPU smoke tests for the explicit feedback primitives."""

from __future__ import annotations

import torch

from streaming_couping.src.bidirectional_feedback import (
    DepthGuidedMaskRefiner,
    StaticBackgroundPoseOptimizer,
    TemporalPromptProjector,
)


def main() -> None:
    # Module 1: foreground pixels must receive exactly zero gradient.
    loss = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4).clone()
    loss.requires_grad_(True)
    masks = torch.zeros(1, 2, 3, 4, dtype=torch.bool)
    masks[0, 0, 1, 1] = True
    masks[0, 1, 2, 2] = True
    optimizer = StaticBackgroundPoseOptimizer()
    background = optimizer.static_background_mask(masks)
    scalar = optimizer(loss, masks)
    scalar.backward()
    assert bool((loss.grad[~background] == 0).all())
    assert bool((loss.grad[background] > 0).all())

    # Module 2: the larger near-depth mode is retained and the separated
    # background mode is removed.  The test uses an explicit threshold so the
    # expected clustering is independent of a particular model scale.
    depth = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 3.0],
            [1.0, 1.0, 3.0, 3.0],
        ]
    )
    mask = torch.ones(3, 4, dtype=torch.bool)
    refined, stats = DepthGuidedMaskRefiner(
        absolute_gap=0.5,
        relative_gap=0.01,
    ).refine(mask, depth, return_stats=True)
    assert int(refined.sum()) == 9
    assert int(stats.removed_pixels) == 3
    assert int(stats.cluster_count) == 2

    # Module 3: standard pinhole projection uses (u,v), filters behind and
    # out-of-frame points, and preserves caller-supplied IDs.
    intrinsics = torch.tensor(
        [[10.0, 0.0, 5.0], [0.0, 10.0, 5.0], [0.0, 0.0, 1.0]]
    )
    centers = torch.tensor(
        [
            [0.0, 0.0, 1.0],  # (5,5), valid
            [1.0, 0.0, 1.0],  # (15,5), valid
            [0.0, 0.0, -1.0],  # behind camera
            [3.0, 0.0, 1.0],  # outside width
        ]
    )
    projection = TemporalPromptProjector().project(
        centers,
        torch.eye(4),
        intrinsics,
        (10, 20),
        object_ids=(10, 11, 12, 13),
    )
    assert projection.valid_mask.tolist() == [True, True, False, False]
    assert projection.object_ids.tolist() == [10, 11]
    assert torch.allclose(
        projection.points_uv,
        torch.tensor([[5.0, 5.0], [15.0, 5.0]]),
    )
    print(
        "bidirectional feedback smoke passed "
        f"background={int(background.sum())} "
        f"refined={int(refined.sum())} "
        f"projected={int(projection.valid_mask.sum())}"
    )


if __name__ == "__main__":
    main()

