#!/usr/bin/env python3
"""CPU-only checks for V7.4 prompts, causal dedup and cache audit."""

from __future__ import annotations

import torch

from streaming_couping.scripts.audit_v7_cache import (
    _uniform_sample_indices,
)
from streaming_couping.src.learned_pose.cache import (
    _online_tracks_duplicate_at_birth,
)
from streaming_couping.src.learned_pose.config import (
    load_learned_pose_config,
)


def main() -> None:
    config = load_learned_pose_config(
        "streaming_couping/configs/v74_temporal_data.yaml"
    )
    clip = config.clips[0]
    _require(
        clip.instance_prompts == ("bed", "wardrobe"),
        f"unexpected V7.4 concepts: {clip.instance_prompts!r}",
    )

    accepted_masks = torch.zeros(3, 6, 6, dtype=torch.bool)
    accepted_masks[0:, 1:4, 1:4] = True
    duplicate_masks = torch.zeros_like(accepted_masks)
    duplicate_masks[1:, 1:4, 1:4] = True
    accepted = {"birth": 0, "masks": accepted_masks}
    duplicate = {"birth": 1, "masks": duplicate_masks}
    _require(
        _online_tracks_duplicate_at_birth(duplicate, accepted),
        "same-object prompt tracks were not deduplicated at birth",
    )

    future_only_masks = torch.zeros_like(accepted_masks)
    future_only_masks[2, 1:4, 1:4] = True
    future_only = {"birth": 1, "masks": future_only_masks}
    _require(
        not _online_tracks_duplicate_at_birth(future_only, accepted),
        "future overlap retroactively changed birth-time deduplication",
    )

    elements = 30 * 8 * 256 * 384
    indices = _uniform_sample_indices(elements, 1_000_000)
    _require(int(indices[0]) == 0, "audit sample lost first element")
    _require(
        int(indices[-1]) == elements - 1,
        "audit sample rounded beyond the final mask element",
    )
    print("V7.4 dynamic prompt and cache-audit smoke passed")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"V7.4 smoke failed: {message}")


if __name__ == "__main__":
    main()
