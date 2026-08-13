#!/usr/bin/env python3
"""CPU smoke for SAM-region identity filtering and fixed PnP."""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np
import torch

from streaming_couping.src.sam_region_pose_candidate import (
    MatchPool,
    build_instance_label_map,
    deployable_cache_view,
    load_sam_region_pose_config,
    method_mask,
    mutual_ratio_matches,
    select_equal_count_correspondences,
    shuffled_labels,
    solve_pose_candidate,
)


def main() -> None:
    config = load_sam_region_pose_config(
        "streaming_couping/configs/v0_baseline.yaml"
    )
    test_gt_isolation_and_labels(config)
    test_mutual_matching()
    test_equal_count_selection(config)
    test_pnp(config)
    print("V0 MESA-style SAM-region pose smoke passed")


def test_gt_isolation_and_labels(config) -> None:
    masks = torch.zeros(2, 3, 32, 64, dtype=torch.bool)
    masks[1, 0, 6:24, 8:26] = True
    masks[1, 1, 6:24, 36:54] = True
    payload = {
        "frame_indices": [0, 1],
        "instance_ids": [0, 1, 2],
        "instance_prompts": ["chair"],
        "sam_track_ids": [10, 11, -1],
        "sam_birth_indices": [1, 1, -1],
        "stream_images": torch.zeros(2, 3, 32, 64),
        "baseline_world_points": torch.zeros(2, 32, 64, 3),
        "baseline_world_confidence": torch.ones(2, 32, 64),
        "tracking_masks_stream": masks,
        "tracking_scores": torch.ones(2, 3),
        "scene_scale": 1.0,
        "target_world_to_camera": torch.eye(4),
    }
    deployable = deployable_cache_view(payload)
    assert not any(name.startswith("target_") for name in deployable)
    labels, active = build_instance_label_map(
        deployable,
        frame_index=1,
        output_size=(32, 64),
        config=replace(config, mask_erosion_pixels=1, max_pose_instances=1),
    )
    assert active == (1,)
    assert int(labels[12, 14]) == 1
    assert int(labels[5, 14]) == -1
    assert int(labels[0, 0]) == 0
    assert int(labels[12, 42]) == -1
    shifted = shuffled_labels(labels, active_labels=active, registry_capacity=3)
    assert int(shifted[12, 14]) == 2


def test_mutual_matching() -> None:
    left = np.asarray([[0, 0], [10, 0], [0, 10], [10, 10]], dtype=np.float32)
    right = left + np.asarray([0.01, -0.01], dtype=np.float32)
    matches = mutual_ratio_matches(left, right, ratio=0.75)
    assert [(row[0], row[1]) for row in matches] == [(0, 0), (1, 1), (2, 2), (3, 3)]


def test_equal_count_selection(config) -> None:
    count = 48
    points = torch.stack(
        (torch.arange(count).remainder(12) * 10 + 3, torch.arange(count).div(12, rounding_mode="floor") * 10 + 3),
        dim=-1,
    ).float()
    anchor = torch.zeros(count, dtype=torch.long)
    current = torch.zeros(count, dtype=torch.long)
    anchor[:16] = 1
    current[:16] = 1
    shuffled = current.clone()
    shuffled[:16] = 2
    pool = MatchPool(
        current_points=points,
        world_points=torch.ones(count, 3),
        descriptor_distances=torch.arange(count).float(),
        raw_reprojection_errors=torch.zeros(count),
        anchor_indices=torch.zeros(count, dtype=torch.long),
        anchor_labels=anchor,
        current_labels=current,
        shuffled_current_labels=shuffled,
    )
    assert int(method_mask(pool, "sam_region_identity").sum()) == count
    assert int(method_mask(pool, "shuffled_instance_identity").sum()) == count - 16
    selected, diagnostics = select_equal_count_correspondences(
        pool,
        config=replace(config, max_correspondences=24, min_correspondences=8),
        image_size=(48, 128),
    )
    assert {int(value.shape[0]) for value in selected.values()} == {24}
    assert diagnostics["sam_region_identity"]["selected_instance_correspondences"] > 0


def test_pnp(config) -> None:
    rng = np.random.default_rng(0)
    count = 120
    world = rng.normal(size=(count, 3))
    world[:, :2] *= 0.5
    world[:, 2] = np.abs(world[:, 2]) + 3.0
    calibration = np.asarray(
        [[300.0, 0.0, 192.0], [0.0, 300.0, 128.0], [0.0, 0.0, 1.0]]
    )
    translation = np.asarray([0.05, -0.02, 0.01])
    image, _ = cv2.projectPoints(
        world,
        np.zeros((3, 1)),
        translation[:, None],
        calibration,
        None,
    )
    labels = torch.ones(count, dtype=torch.long)
    pool = MatchPool(
        current_points=torch.from_numpy(image[:, 0]).float(),
        world_points=torch.from_numpy(world).float(),
        descriptor_distances=torch.arange(count).float(),
        raw_reprojection_errors=torch.ones(count),
        anchor_indices=torch.zeros(count, dtype=torch.long),
        anchor_labels=labels,
        current_labels=labels,
        shuffled_current_labels=labels + 1,
    )
    relaxed = replace(
        config,
        min_correspondences=32,
        min_inliers=20,
        min_validation_correspondences=8,
        max_rotation_degrees=10.0,
        max_translation_scene_fraction=0.2,
    )
    pose, diagnostics = solve_pose_candidate(
        pool=pool,
        selected=torch.arange(count),
        raw_pose=torch.eye(4)[:3],
        intrinsics=torch.from_numpy(calibration).float(),
        scene_scale=3.0,
        config=relaxed,
        sam_evidence_active=True,
    )
    assert diagnostics["optimized"] == 1
    torch.testing.assert_close(
        pose[:, 3], torch.from_numpy(translation).float(), atol=1e-3, rtol=1e-3
    )


if __name__ == "__main__":
    main()
