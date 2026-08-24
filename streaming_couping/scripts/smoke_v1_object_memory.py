#!/usr/bin/env python3
"""Small deterministic smoke test for V1 persistent object identity."""

from __future__ import annotations

import torch

from streaming_couping.src.object_memory import (
    PersistentObjectMemory,
    PersistentObjectMemoryConfig,
    collapse_persistent_tracks,
)
from streaming_couping.src.semantic_map import (
    build_persistent_semantic_pointmap,
)


def main() -> None:
    sequence, height, width, tracks = 4, 4, 4, 2
    base = torch.tensor([0.2, 0.1, 1.0])
    world = torch.zeros(sequence, height, width, 3)
    for frame in range(sequence):
        # A static object has the same world support after SAM re-entry.
        world[frame] = base
    confidence = torch.linspace(
        0.2, 1.0, steps=height * width
    ).reshape(1, height, width).repeat(sequence, 1, 1)
    masks = torch.zeros(sequence, tracks, height, width, dtype=torch.bool)
    masks[0:2, 0] = True
    masks[2:4, 1] = True
    scores = torch.ones(sequence, tracks)
    config = PersistentObjectMemoryConfig(
        min_observation_points=4,
        min_mask_pixels=4,
    )
    memory = PersistentObjectMemory(config)
    result = memory.process_sequence(
        world_points=world,
        confidence=confidence,
        masks=masks,
        track_scores=scores,
        frame_indices=(10, 11, 12, 13),
        track_ids=(100, 200),
        track_prompts=("chair", "chair"),
    )
    ids = result["persistent_object_ids"]
    assert int(result["persistent_object_count"]) == 1
    assert ids[0, 0].item() == ids[2, 1].item()
    collapsed = collapse_persistent_tracks(
        masks=masks,
        scores=scores,
        persistent_object_ids=ids,
        object_ids=result["object_ids"],
    )
    assert collapsed["masks"].shape[1] == 1
    images = torch.zeros(sequence, 3, height, width)
    semantic = build_persistent_semantic_pointmap(
        world_points=world,
        confidence=confidence,
        masks=masks,
        track_scores=scores,
        persistent_object_ids=ids,
        sam_track_ids=(100, 200),
        images=images,
        confidence_threshold=0.3,
        track_score_threshold=0.5,
        max_map_points=128,
        map_write_mask=result["map_write_mask"],
    )
    assert semantic.semantic_object_ids is not None
    assert bool((semantic.semantic_object_ids == ids[0, 0]).any())
    print(
        "V1 persistent object-memory smoke passed "
        f"objects={result['persistent_object_count']} "
        f"events={len(result['events'])}"
    )


if __name__ == "__main__":
    main()
