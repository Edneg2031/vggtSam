#!/usr/bin/env python3
"""Synthetic smoke tests for the V0 joint geometry-feedback diagnostic."""

from __future__ import annotations

import torch

from streaming_couping.src.robust_depth_feedback import (
    HistoricalDepthVetoConfig,
    affine_error_metrics,
    apply_depth_veto,
    build_causal_history_cache,
    fit_robust_affine,
    gather_historical_object_depths,
    gather_historical_object_points,
    transform_world_points,
)


def main() -> None:
    config = HistoricalDepthVetoConfig(
        min_history_points=4,
        max_history_points=64,
        max_points_per_history_frame=64,
        confidence_threshold=0.30,
        track_score_threshold=0.50,
        mad_multiplier=3.0,
        absolute_padding_m=0.05,
    )
    sequence, height, width = 3, 4, 4
    points = torch.zeros(sequence, height, width, 3)
    points[..., 2] = 1.0
    points[0, :2, :2, 0] = 0.1
    points[1, :2, :2, 0] = 0.2
    confidence = torch.ones(sequence, height, width)
    masks = torch.zeros(sequence, 1, height, width, dtype=torch.bool)
    masks[:, 0, :2, :2] = True
    scores = torch.ones(sequence, 1)
    pose = torch.eye(4, dtype=torch.float32)[:3]

    history_points = gather_historical_object_points(
        points=points,
        confidence=confidence,
        raw_masks=masks,
        scores=scores,
        current_frame=2,
        slot=0,
        config=config,
    )
    assert history_points.shape == (8, 3)
    history_cache = build_causal_history_cache(
        points=points,
        confidence=confidence,
        raw_masks=masks,
        scores=scores,
        config=config,
    )
    assert torch.equal(history_cache[2][0], history_points)
    history_depths = gather_historical_object_depths(
        points=points,
        confidence=confidence,
        raw_masks=masks,
        scores=scores,
        current_frame=2,
        slot=0,
        current_world_to_camera=pose,
        config=config,
    )
    assert history_depths.shape == (8,)
    transformed = transform_world_points(history_points[:2], pose)
    assert torch.allclose(transformed[:, 2], torch.ones(2))
    batched_pose = pose.unsqueeze(0).expand(sequence, -1, -1).clone()
    transformed_batched = transform_world_points(points, batched_pose)
    assert transformed_batched.shape == points.shape
    assert torch.allclose(transformed_batched[..., 2], points[..., 2])

    current_depth = torch.ones(height, width)
    current_depth[0, 0] = 2.0
    veto = apply_depth_veto(
        masks[2, 0],
        current_depth,
        history_depths,
        reference_kind="history_median_mad",
        config=config,
    )
    assert veto.removed_pixels == 1
    assert not bool(veto.mask[0, 0])
    assert int(veto.mask.sum()) == 3

    no_history = apply_depth_veto(
        masks[0, 0],
        current_depth,
        torch.empty(0),
        reference_kind="history_median_mad",
        config=config,
    )
    assert no_history.fallback_used
    assert torch.equal(no_history.mask, masks[0, 0])

    source = torch.arange(1.0, 101.0)
    target = 1.25 * source + 0.15
    fit = fit_robust_affine(
        source,
        target,
        min_samples=16,
        max_samples=256,
        trim_quantile=0.90,
        max_iterations=4,
    )
    assert fit.accepted
    assert abs(fit.scale - 1.25) < 1e-4
    assert abs(fit.shift - 0.15) < 1e-4
    heldout = affine_error_metrics(
        source[-20:],
        target[-20:],
        scale=fit.scale,
        shift=fit.shift,
    )
    assert heldout["rmse"] < 1e-4
    print(
        "V0 joint geometry-feedback smoke passed "
        f"history_points={history_points.shape[0]} "
        f"veto_removed={veto.removed_pixels} "
        f"affine_scale={fit.scale:.3f}"
    )


if __name__ == "__main__":
    main()
