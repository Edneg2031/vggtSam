#!/usr/bin/env python3
"""CPU contract smoke for the V0 ALIKED-LightGlue correspondence adapter."""

from __future__ import annotations

import torch

from streaming_couping.src.learned_pose.baseline_runtime import (
    load_lightglue_pnp_candidate_config,
)
from streaming_couping.src.v0_lightglue_pnp import (
    LightGlueFeatureSet,
    build_lightglue_match_pool,
    match_aliked_features,
    spatial_hull_coverage,
)


class _IdentityMatcher:
    def __call__(self, data):
        left = data["image0"]["keypoints"]
        right = data["image1"]["keypoints"]
        count = min(left.shape[1], right.shape[1])
        indices = torch.arange(count, device=left.device)
        return {
            "matches": torch.stack((indices, indices), dim=-1)[None],
            "scores": torch.linspace(0.9, 0.6, count, device=left.device)[None],
        }


def main() -> None:
    config = load_lightglue_pnp_candidate_config(
        "streaming_couping/configs/v0_baseline.yaml"
    )
    assert config.primary_method == "sam_dynamic_excluded"
    assert config.methods == ("full_image", "sam_dynamic_excluded")
    assert config.min_correspondences == 32
    points = torch.tensor(
        [[0.0, 0.0], [9.0, 0.0], [9.0, 9.0], [0.0, 9.0]]
    )
    features = LightGlueFeatureSet(
        points=points,
        descriptors=torch.eye(4, 128),
        scores=torch.ones(4),
        image_size=torch.tensor([10.0, 10.0]),
    )
    matches, scores = match_aliked_features(
        features,
        features,
        matcher=_IdentityMatcher(),
        device="cpu",
    )
    torch.testing.assert_close(matches, torch.stack((torch.arange(4),) * 2, dim=-1))
    torch.testing.assert_close(scores, torch.linspace(0.9, 0.6, 4))
    coverage = spatial_hull_coverage(points, image_size=(10, 10))
    assert abs(coverage - 0.81) < 1e-6
    _test_causal_pool(config, features)
    print("V0 frozen ALIKED-LightGlue adapter smoke passed")


def _test_causal_pool(config, features: LightGlueFeatureSet) -> None:
    frames, height, width = 4, 10, 10
    y, x = torch.meshgrid(
        torch.arange(height),
        torch.arange(width),
        indexing="ij",
    )
    world = torch.stack((x, y, torch.ones_like(x)), dim=-1).float()
    payload = {
        "stream_images": torch.zeros(frames, 3, height, width),
        "baseline_world_points": world[None].repeat(frames, 1, 1, 1),
        "baseline_world_confidence": torch.ones(frames, height, width),
        "tracking_masks_stream": torch.zeros(
            frames, 1, height, width, dtype=torch.bool
        ),
        "trusted_tracking_masks_stream": torch.zeros(
            frames, 1, height, width, dtype=torch.bool
        ),
    }
    pool, pair_rows = build_lightglue_match_pool(
        payload=payload,
        current_index=3,
        features=[features] * frames,
        matcher=_IdentityMatcher(),
        device="cpu",
        config=config,
    )
    assert len(pair_rows) == 3
    assert all(row["raw_pair_matches"] == 4 for row in pair_rows)
    # Three history frames share the same current IDs; one-current-one-anchor
    # pooling must retain only the four most recent candidates.
    assert pool.current_points.shape == (4, 2)
    assert set(pool.anchor_indices.tolist()) == {2}
    torch.testing.assert_close(pool.world_points[:, 2], torch.ones(4))


if __name__ == "__main__":
    main()
