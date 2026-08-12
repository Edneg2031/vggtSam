#!/usr/bin/env python3
"""CPU smoke for V0 SIFT mutual matching and calibrated PnP."""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np
import torch

from streaming_couping.src.v0_feature_pnp import (
    MatchPool,
    mutual_ratio_matches,
    select_landmark_source,
    select_method_correspondences,
    solve_feature_pnp,
)
from streaming_couping.src.learned_pose.baseline_runtime import (
    load_feature_pnp_candidate_config,
)


def main() -> None:
    config = load_feature_pnp_candidate_config(
        "streaming_couping/configs/v0_baseline.yaml"
    )
    assert config.primary_method == "sam_dynamic_excluded"
    assert config.primary_landmark_source == "native_world_pointmap"
    assert config.primary_history_scope == "all_causal"
    _test_mutual_ratio_matching()
    _test_equal_count_selection()
    _test_pnp_and_fallbacks(config)
    print("V0 causal SIFT-depth-PnP smoke passed")


def _test_mutual_ratio_matching() -> None:
    left = np.asarray(
        [[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]],
        dtype=np.float32,
    )
    right = left + np.asarray([0.01, -0.01], dtype=np.float32)
    matches = mutual_ratio_matches(left, right, ratio=0.75)
    assert [(left_id, right_id) for left_id, right_id, _ in matches] == [
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 3),
    ]
    assert mutual_ratio_matches(
        np.empty((0, 2), dtype=np.float32),
        right,
        ratio=0.75,
    ) == []


def _test_equal_count_selection() -> None:
    count = 12
    full = torch.ones(count, dtype=torch.bool)
    region = torch.zeros(count, dtype=torch.bool)
    region[:6] = True
    scarce_region = torch.zeros(count, dtype=torch.bool)
    scarce_region[:3] = True
    pool = MatchPool(
        current_points=torch.arange(count * 2).reshape(count, 2).float(),
        world_points=torch.ones(count, 3),
        distances=torch.arange(count).float(),
        anchor_indices=torch.zeros(count, dtype=torch.long),
        anchor_region={
            "full_image": full,
            "sam_dynamic_excluded": full,
            "sam_instance_background_stratified": region,
            "bbox_instance_background_stratified": scarce_region,
        },
        current_region={
            "full_image": full,
            "sam_dynamic_excluded": full,
            "sam_instance_background_stratified": region,
            "bbox_instance_background_stratified": scarce_region,
        },
        landmark_world_points={
            "raw_depth_unprojected": torch.ones(count, 3),
            "native_world_pointmap": torch.full((count, 3), 2.0),
        },
    )
    native = select_landmark_source(pool, "native_world_pointmap")
    torch.testing.assert_close(native.world_points, torch.full((count, 3), 2.0))
    torch.testing.assert_close(native.current_points, pool.current_points)
    selected, feasible, _ = select_method_correspondences(
        pool,
        method="full_image",
        count=8,
    )
    assert feasible and selected.shape[0] == 8
    selected, feasible, diagnostics = select_method_correspondences(
        pool,
        method="sam_instance_background_stratified",
        count=8,
    )
    assert feasible and selected.shape[0] == 8
    assert int((selected < 6).sum()) == 4
    assert diagnostics == {"available": 12, "region": 6, "background": 6}
    selected, feasible, diagnostics = select_method_correspondences(
        pool,
        method="bbox_instance_background_stratified",
        count=8,
    )
    assert not feasible and selected.shape[0] == 7
    assert diagnostics == {"available": 12, "region": 3, "background": 9}
    selected, feasible, _ = select_method_correspondences(
        pool,
        method="full_image",
        count=0,
    )
    assert not feasible and selected.numel() == 0


def _test_pnp_and_fallbacks(config) -> None:
    rng = np.random.default_rng(0)
    count = 96
    world = rng.normal(size=(count, 3))
    world[:, :2] *= 0.5
    world[:, 2] = np.abs(world[:, 2]) + 3.0
    calibration = np.array(
        [[300.0, 0.0, 192.0], [0.0, 300.0, 128.0], [0.0, 0.0, 1.0]]
    )
    true_rotation = np.eye(3)
    true_translation = np.array([0.12, -0.02, 0.01])
    rvec, _ = cv2.Rodrigues(true_rotation)
    image, _ = cv2.projectPoints(
        world,
        rvec,
        true_translation[:, None],
        calibration,
        None,
    )
    pool = MatchPool(
        current_points=torch.from_numpy(image[:, 0]).float(),
        world_points=torch.from_numpy(world).float(),
        distances=torch.arange(count).float(),
        anchor_indices=torch.zeros(count, dtype=torch.long),
        anchor_region={"full_image": torch.ones(count, dtype=torch.bool)},
        current_region={"full_image": torch.ones(count, dtype=torch.bool)},
    )
    raw = torch.eye(4)[:3].repeat(2, 1, 1)[None]
    intrinsics = torch.from_numpy(calibration).float().repeat(2, 1, 1)[None]
    arguments = dict(
        pool=pool,
        selected=torch.arange(count),
        raw_world_to_camera=raw,
        intrinsics=intrinsics,
        current_index=1,
        scene_scale=3.0,
        config=config,
    )
    pose, diagnostics = solve_feature_pnp(**arguments)
    assert diagnostics["optimized"] == 1
    assert diagnostics["inlier_positive_depth_fraction"] == 1.0
    torch.testing.assert_close(
        pose[:, 3],
        torch.from_numpy(true_translation).float(),
        atol=1e-3,
        rtol=1e-3,
    )
    repeated_pose, repeated_diagnostics = solve_feature_pnp(**arguments)
    torch.testing.assert_close(repeated_pose, pose, atol=1e-6, rtol=1e-6)
    assert repeated_diagnostics["inliers"] == diagnostics["inliers"]
    assert (
        repeated_diagnostics["reprojection_rmse_pixels"]
        == diagnostics["reprojection_rmse_pixels"]
    )

    bounded_arguments = dict(arguments)
    bounded_arguments["config"] = replace(
        config,
        max_translation_scene_fraction=1e-4,
    )
    bounded_pose, bounded_diagnostics = solve_feature_pnp(**bounded_arguments)
    assert bounded_diagnostics["optimized"] == 0
    assert bounded_diagnostics["reason"] == "pnp_update_outside_locked_bounds"
    torch.testing.assert_close(bounded_pose, raw[0, 1])

    fallback_pose, fallback_diagnostics = solve_feature_pnp(
        **arguments,
        force_fallback_reason="equal_count_infeasible",
    )
    assert fallback_diagnostics["optimized"] == 0
    assert fallback_diagnostics["reason"] == "equal_count_infeasible"
    torch.testing.assert_close(fallback_pose, raw[0, 1])


if __name__ == "__main__":
    main()
