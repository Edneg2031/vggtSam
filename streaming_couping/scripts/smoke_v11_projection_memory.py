#!/usr/bin/env python3
"""Deterministic smoke test for V1.1 projection and temporal fusion."""

from __future__ import annotations

import torch

from streaming_couping.src.projection_object_memory import (
    ProjectionAssociationConfig,
    ProjectionObjectMemory,
    ProjectionObjectMemoryConfig,
    mask_overlap_stats,
    project_world_points_to_mask,
)


def main() -> None:
    sequence, height, width, tracks = 4, 16, 16, 2
    fx = fy = 10.0
    cx = cy = 8.0
    grid_y, grid_x = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    world = torch.stack(
        ((grid_x - cx) / fx, (grid_y - cy) / fy, torch.ones_like(grid_x)),
        dim=-1,
    ).unsqueeze(0).repeat(sequence, 1, 1, 1)
    confidence = torch.ones(sequence, height, width)
    masks = torch.zeros(sequence, tracks, height, width, dtype=torch.bool)
    rectangle = torch.zeros(height, width, dtype=torch.bool)
    rectangle[6:11, 6:11] = True
    masks[0, 0] = rectangle
    masks[1, 0] = rectangle
    masks[2, 1] = rectangle
    masks[3, 1] = rectangle
    scores = torch.ones(sequence, tracks)
    w2c = torch.eye(4)[:3].unsqueeze(0).repeat(sequence, 1, 1)
    intrinsics = torch.tensor(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
    ).unsqueeze(0).repeat(sequence, 1, 1)
    projected = project_world_points_to_mask(
        world[0][rectangle],
        world_to_camera=w2c[0],
        intrinsics=intrinsics[0],
        image_size=(height, width),
        dilation_radius=0,
    )
    overlap = mask_overlap_stats(rectangle, projected)
    assert float(overlap["projection_iou"]) > 0.95

    config = ProjectionObjectMemoryConfig(
        min_observation_points=4,
        min_mask_pixels=4,
        confirmation_frames=2,
        confirmation_window=3,
        max_pending_gap=3,
        absolute_voxel_size=0.04,
        association=ProjectionAssociationConfig(
            projection_dilation_radius=0,
            min_projected_pixels=4,
            min_projection_iou=0.20,
            min_match_score=0.35,
        ),
    )
    memory = ProjectionObjectMemory(config)
    result = memory.process_sequence(
        world_points=world,
        confidence=confidence,
        masks=masks,
        track_scores=scores,
        frame_indices=(10, 11, 12, 13),
        track_ids=(100, 200),
        track_prompts=("chair", "chair"),
        world_to_camera=w2c,
        intrinsics=intrinsics,
        image_size=(height, width),
        images=torch.zeros(sequence, 3, height, width),
    )
    ids = result["persistent_object_ids"]
    assert int(result["persistent_object_count"]) == 1
    assert int(result["pending_track_count"]) == 0
    assert int(result["confirmed_observation_count"]) == 4
    assert ids[0, 0].item() == ids[1, 0].item()
    assert ids[1, 0].item() == ids[2, 1].item()
    assert ids[2, 1].item() == ids[3, 1].item()
    fused = result["fused_map"]
    assert int(fused["world_points"].shape[0]) > 0
    assert int(fused["world_points"].shape[0]) <= int(
        (masks & masks).sum()
    )
    print(
        "V1.1 projection-memory smoke passed "
        f"objects={result['persistent_object_count']} "
        f"confirmed={result['confirmed_observation_count']} "
        f"fused_voxels={fused['world_points'].shape[0]}"
    )


if __name__ == "__main__":
    main()

