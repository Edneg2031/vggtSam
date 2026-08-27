#!/usr/bin/env python3
"""Deterministic smoke test for the DINO-assisted projection memory route."""

from __future__ import annotations

import torch

from streaming_couping.src.projection_object_memory import (
    ProjectionAssociationConfig,
    ProjectionObjectMemory,
    ProjectionObjectMemoryConfig,
)


def main() -> None:
    # Two objects occupy different image regions.  Slot 0 disappears for two
    # frames and re-enters with a new appearance.  The appearance term must be
    # consulted at re-entry, while an invalid appearance must be ignored.
    sequence, height, width, tracks = 6, 20, 20, 2
    grid_y, grid_x = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    world = torch.stack(
        ((grid_x - 10.0) / 10.0, (grid_y - 10.0) / 10.0, torch.ones_like(grid_x)),
        dim=-1,
    ).unsqueeze(0).repeat(sequence, 1, 1, 1)
    confidence = torch.ones(sequence, height, width)
    masks = torch.zeros(sequence, tracks, height, width, dtype=torch.bool)
    left = torch.zeros(height, width, dtype=torch.bool)
    left[5:12, 3:10] = True
    right = torch.zeros(height, width, dtype=torch.bool)
    right[5:12, 12:19] = True
    masks[0, 0] = left
    masks[1, 0] = left
    masks[2, 1] = right
    masks[4, 0] = left
    masks[5, 0] = left
    masks[0, 1] = right
    masks[1, 1] = right
    masks[3, 1] = right
    masks[4, 1] = right
    masks[5, 1] = right
    scores = torch.ones(sequence, tracks)
    w2c = torch.eye(4)[:3].unsqueeze(0).repeat(sequence, 1, 1)
    intrinsics = torch.tensor(
        [[10.0, 0.0, 10.0], [0.0, 10.0, 10.0], [0.0, 0.0, 1.0]]
    ).unsqueeze(0).repeat(sequence, 1, 1)
    appearance = torch.zeros(sequence, tracks, 4)
    appearance[:, 0, 0] = 1.0
    appearance[:, 1, 1] = 1.0
    appearance_valid = torch.ones(sequence, tracks, dtype=torch.bool)
    appearance[4, 0] = 0.0
    appearance_valid[4, 0] = False

    config = ProjectionObjectMemoryConfig(
        min_observation_points=4,
        min_mask_pixels=4,
        confirmation_frames=2,
        confirmation_window=3,
        max_pending_gap=3,
        reassociation_gap=2,
        absolute_voxel_size=0.04,
        association=ProjectionAssociationConfig(
            projection_dilation_radius=0,
            min_projected_pixels=4,
            min_projection_iou=0.20,
            min_match_score=0.30,
            appearance_weight=0.20,
        ),
    )
    memory = ProjectionObjectMemory(config)
    result = memory.process_sequence(
        world_points=world,
        confidence=confidence,
        masks=masks,
        track_scores=scores,
        frame_indices=tuple(range(sequence)),
        track_ids=(100, 200),
        track_prompts=("chair", "chair"),
        world_to_camera=w2c,
        intrinsics=intrinsics,
        image_size=(height, width),
        appearance=appearance,
        appearance_valid=appearance_valid,
    )
    events = result["events"]
    forced = [row for row in events if int(row.get("reassociation_forced", 0))]
    compared = [
        row
        for row in events
        if int(row.get("association_appearance_compared", 0))
    ]
    assert int(result["persistent_object_count"]) == 2
    assert forced, "the configured re-entry gap did not force ranking"
    assert compared, "appearance was never compared at re-entry"
    assert all(
        int(row.get("observation_appearance_valid", 1)) == 1
        for row in compared
    )
    assert int(result["pending_track_count"]) == 0
    print(
        "DINOv3 projection-memory smoke passed "
        f"objects={result['persistent_object_count']} "
        f"forced_reassociations={len(forced)} "
        f"appearance_compared={len(compared)}"
    )


if __name__ == "__main__":
    main()
