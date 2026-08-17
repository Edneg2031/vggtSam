#!/usr/bin/env python3
"""CPU smoke for causal mask handling and bounded surfel displacement."""

from __future__ import annotations

import torch

from streaming_couping.src.instance_geometry import (
    apply_sparse_deltas,
    bounded_surfel_query,
    erode_instance_masks,
    merge_sparse_deltas,
    select_mask_points,
    shift_instance_masks,
)
from streaming_couping.scripts.run_v1_instance_geometry import (
    geometry_excluded_slots,
)


def main() -> None:
    assert geometry_excluded_slots(
        ("wardrobe", "chair", "Nightstand"),
        (" wardrobe ",),
    ) == (0,)

    masks = torch.zeros(2, 1, 7, 7, dtype=torch.bool)
    masks[:, :, 1:6, 1:6] = True
    eroded = erode_instance_masks(masks, radius=1)
    assert int(eroded.sum()) == 18
    shifted = shift_instance_masks(eroded, shift_y=1, shift_x=2)
    assert int(shifted.sum()) == int(eroded.sum())

    grid_y, grid_x = torch.meshgrid(
        torch.arange(7, dtype=torch.float32),
        torch.arange(7, dtype=torch.float32),
        indexing="ij",
    )
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1)
    confidence = torch.ones(7, 7)
    selected, weights, indices = select_mask_points(
        points,
        confidence,
        eroded[0, 0],
        limit=32,
    )
    assert selected.shape == (9, 3)
    assert weights.shape == (9,)
    assert indices.shape == (9,)

    history_xy = torch.tensor(
        [[x, y] for frame in range(3) for y in (2.0, 3.0, 4.0) for x in (2.0, 3.0, 4.0)]
    )
    history = torch.cat(
        [history_xy, torch.ones(history_xy.shape[0], 1)], dim=1
    )
    frame_ids = torch.arange(3).repeat_interleave(9)
    current = torch.tensor([[3.0, 3.0, 1.1], [20.0, 20.0, 1.0]])
    result = bounded_surfel_query(
        current_points=current,
        history_points=history,
        history_weights=torch.ones(history.shape[0]),
        history_frame_ids=frame_ids,
        device="cpu",
        neighbors=16,
        min_support_frames=3,
        match_radius=2.0,
        normal_variance_max=0.20,
        alpha=0.5,
        max_displacement=0.2,
        chunk_size=2,
    )
    assert result.valid.tolist() == [True, False]
    assert 0.0 < float(-result.delta[0, 2]) <= 0.2
    assert torch.equal(result.delta[1], torch.zeros(3))

    merged_indices, merged_deltas = merge_sparse_deltas(
        [torch.tensor([0, 0, 2])],
        [torch.tensor([[0.0, 0.0, 0.1], [0.0, 0.0, 0.3], [1.0, 0.0, 0.0]])],
    )
    assert merged_indices.tolist() == [0, 2]
    assert torch.allclose(merged_deltas[0], torch.tensor([0.0, 0.0, 0.2]))
    dense = apply_sparse_deltas(
        torch.zeros(1, 1, 3, 3), merged_indices, merged_deltas
    )
    assert torch.allclose(dense.reshape(-1, 3)[0], merged_deltas[0])
    assert torch.allclose(dense.reshape(-1, 3)[2], merged_deltas[1])
    print("V1 causal bounded instance-surface geometry smoke passed")


if __name__ == "__main__":
    main()
