#!/usr/bin/env python3
"""No-model smoke test for DINOv3 object feature memory."""

from __future__ import annotations

import torch

from streaming_couping.src.dinov3_object_features import (
    aggregate_persistent_features,
    masked_mean_pool,
    shuffled_persistent_features,
)


def main() -> None:
    dense = torch.zeros(2, 2, 3, 4)
    dense[0, :, :, 0] = 1.0
    dense[1, :, :, 1] = 1.0
    masks = torch.zeros(2, 2, 4, 6, dtype=torch.bool)
    masks[:, 0, :, :3] = True
    masks[:, 1, :, 3:] = True
    pooled, valid = masked_mean_pool(dense, masks)
    assert tuple(pooled.shape) == (2, 2, 4)
    assert bool(valid.all())
    assert float(pooled[0, 0, 0]) > 0.99
    assert float(pooled[0, 1, 0]) < 0.01

    single = torch.zeros(4, 2, 4)
    single[:, 0, 0] = 1.0
    single[:, 1, 1] = 1.0
    observed = torch.ones(4, 2, dtype=torch.bool)
    persistent, persistent_valid = aggregate_persistent_features(
        single,
        observed,
        beta=0.9,
    )
    shuffled, shuffled_valid, permutation = shuffled_persistent_features(
        single,
        observed,
        beta=0.9,
        seed=2026,
    )
    assert tuple(persistent.shape) == tuple(single.shape)
    assert bool(persistent_valid.all()) and bool(shuffled_valid.all())
    assert sorted(permutation.tolist()) == [0, 1]
    assert torch.isfinite(shuffled).all()
    print(
        "DINOv3 object feature smoke passed "
        f"single={tuple(single.shape)} permutation={permutation.tolist()}"
    )


if __name__ == "__main__":
    main()
