#!/usr/bin/env python3
"""CPU geometry/sampling smoke for the V0 Track-BA pose candidate."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch

from streaming_couping.src.learned_pose.baseline_runtime import (
    TrackBACandidateConfig,
)
from streaming_couping.src.v0_track_ba import (
    TrackWindow,
    cache_tokens_for_track_head,
    compose_relative_candidate,
    optimize_track_window,
    project_points,
    sample_query_points,
    so3_exp,
    support_region,
    track_validity_and_diagnostics,
)


def main() -> None:
    config = _config()
    _cache_shape_smoke()
    _sampling_smoke(config)
    _validity_smoke(config)
    _optimizer_smoke(config)
    print("V0 causal TrackHead motion-only BA smoke passed")


def _cache_shape_smoke() -> None:
    levels = torch.zeros(4, 6, 9, 2048)
    payload = {
        "token_levels": levels,
        "dpt_layer_indices": [4, 11, 17, 23],
    }
    token_list = cache_tokens_for_track_head(
        payload,
        indices=[0, 3, 5],
        device="cpu",
    )
    assert len(token_list) == 24
    for index in (4, 11, 17, 23):
        assert token_list[index].shape == (1, 3, 9, 2048)


def _sampling_smoke(config: TrackBACandidateConfig) -> None:
    sequence, instances, height, width = 3, 2, 16, 24
    tracking = torch.zeros(sequence, instances, height, width, dtype=torch.bool)
    trusted = torch.zeros_like(tracking)
    tracking[0, 0, 2:8, 3:10] = True
    trusted[0, 0] = tracking[0, 0]
    payload = {
        "tracking_masks_stream": tracking,
        "trusted_tracking_masks_stream": trusted,
        "instance_rigid_weight": torch.tensor(
            [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
        ),
    }
    full = support_region(
        payload,
        frame_index=0,
        method="full_image",
        output_size=(height, width),
    )
    assert bool(full.all())
    region = support_region(
        payload,
        frame_index=0,
        method="sam_instance_background_stratified",
        output_size=(height, width),
    )
    points, feasible = sample_query_points(
        region,
        allowed=torch.ones_like(region),
        count=32,
        grid_shape=(12, 16),
        method="sam_instance_background_stratified",
        frame_index=0,
    )
    assert points.shape == (32, 2) and feasible
    pixel_x = points[:, 0].round().long()
    pixel_y = points[:, 1].round().long()
    assert int(region[pixel_y, pixel_x].sum()) == 16


def _optimizer_smoke(config: TrackBACandidateConfig) -> None:
    torch.manual_seed(0)
    frames, points = 3, 96
    world = torch.randn(points, 3, dtype=torch.double)
    world[:, :2] *= 0.6
    world[:, 2] = world[:, 2].abs() + 3.0
    rotation = torch.eye(3, dtype=torch.double).repeat(frames, 1, 1)
    true_translation = torch.zeros(frames, 3, dtype=torch.double)
    true_translation[1, 0] = 0.08
    true_translation[2, 0] = 0.16
    calibration = torch.eye(3, dtype=torch.double).repeat(frames, 1, 1)
    calibration[:, 0, 0] = 300.0
    calibration[:, 1, 1] = 300.0
    calibration[:, 0, 2] = 192.0
    calibration[:, 1, 2] = 128.0
    tracks, visible = project_points(
        world,
        rotation,
        true_translation,
        calibration,
    )
    raw = torch.cat(
        (rotation, torch.zeros(frames, 3, 1, dtype=torch.double)),
        dim=-1,
    )[None].float()
    window = TrackWindow(
        anchor_index=0,
        current_index=2,
        sequence_indices=(0, 1, 2),
        query_points=tracks[0].float(),
        world_points=world.float(),
        tracks=tracks.float(),
        valid=visible,
        visibility=torch.ones(frames, points),
        confidence=torch.ones(frames, points),
        region_mask=torch.ones(256, 384, dtype=torch.bool),
        equal_count_region_feasible=True,
    )
    before, _ = project_points(
        world,
        rotation,
        torch.zeros_like(true_translation),
        calibration,
    )
    initial_error = torch.linalg.vector_norm(before[-1] - tracks[-1], dim=-1).mean()
    candidate_relative, diagnostics = optimize_track_window(
        raw_world_to_camera=raw,
        intrinsics=calibration[None].float(),
        window=window,
        scene_scale=3.0,
        config=replace(
            config,
            optimizer_steps=180,
            rotation_prior_weight=0.01,
            translation_prior_weight=0.01,
            temporal_prior_weight=0.01,
        ),
    )
    candidate = compose_relative_candidate(
        raw_world_to_camera=raw,
        anchor_index=0,
        candidate_relative=candidate_relative,
    )
    refined, _ = project_points(
        world,
        candidate[None, :3, :3].double(),
        candidate[None, :3, 3].double(),
        calibration[-1:],
    )
    final_error = torch.linalg.vector_norm(refined[0] - tracks[-1], dim=-1).mean()
    assert diagnostics["optimized"] == 1
    assert float(final_error) < float(initial_error) * 0.25
    rotation_matrix = so3_exp(torch.zeros(4, 3, dtype=torch.double))
    torch.testing.assert_close(
        rotation_matrix,
        torch.eye(3, dtype=torch.double).repeat(4, 1, 1),
    )


def _validity_smoke(config: TrackBACandidateConfig) -> None:
    tracks = torch.tensor(
        [
            [[2.0, 2.0], [4.0, 4.0], [6.0, 6.0], [8.0, 8.0]],
            [[3.0, 3.0], [-1.0, 4.0], [float("nan"), 6.0], [20.0, 8.0]],
        ]
    )
    visibility = torch.tensor(
        [[1.0, 1.0, 1.0, 1.0], [0.9, 0.9, 0.9, 0.01]]
    )
    confidence = torch.tensor(
        [[1.0, 1.0, 1.0, 1.0], [0.9, 0.01, 0.9, 0.9]]
    )
    valid, diagnostics = track_validity_and_diagnostics(
        tracks=tracks,
        visibility=visibility,
        confidence=confidence,
        geometry_valid=torch.tensor([True, True, True, True]),
        width=16,
        height=12,
        visibility_threshold=config.visibility_threshold,
        confidence_threshold=config.track_confidence_threshold,
    )
    assert int(valid[-1].sum()) == 1
    assert diagnostics["current_track_finite_count"] == 3
    assert diagnostics["current_track_in_bounds_count"] == 1
    assert diagnostics["current_visibility_pass_count"] == 3
    assert diagnostics["current_confidence_pass_count"] == 3
    assert diagnostics["current_valid_after_geometry_count"] == 1


def _config() -> TrackBACandidateConfig:
    return TrackBACandidateConfig(
        enabled=True,
        output_dir=Path("/tmp/v0_track_ba_smoke"),
        device="cpu",
        primary_method="sam_dynamic_excluded",
        methods=("full_image", "sam_dynamic_excluded"),
        window_frames=3,
        query_count=32,
        query_grid=(12, 16),
        track_iterations=2,
        visibility_threshold=0.05,
        track_confidence_threshold=0.05,
        point_confidence_threshold=0.30,
        min_correspondences=16,
        optimizer_steps=80,
        learning_rate=0.03,
        robust_delta_pixels=4.0,
        rotation_prior_weight=0.1,
        translation_prior_weight=0.1,
        temporal_prior_weight=0.1,
        max_rotation_degrees=10.0,
        max_translation_scene_fraction=0.25,
    )


if __name__ == "__main__":
    main()
