#!/usr/bin/env python3
"""CPU smoke checks for independent feature/ray triangulation."""

from __future__ import annotations

import torch

from streaming_couping.src.independent_triangulation import (
    extract_frame_patch_descriptors,
    masks_to_patch_support,
    mutual_nearest_matches,
    patch_centers,
    shift_patch_mask_exact,
    triangulate_camera_rays,
)


def main() -> None:
    test_cached_frame_half_descriptor_extraction()
    test_mask_support_and_area_exact_shift()
    test_mutual_nearest_correspondence()
    test_two_view_ray_triangulation()
    print("T0 independent 2D correspondence/ray triangulation smoke passed")


def test_cached_frame_half_descriptor_extraction() -> None:
    tokens = torch.zeros(1, 1, 2, 6, 4)
    tokens[0, 0, :, 2:, :2] = torch.tensor([1.0, 0.0])
    tokens[0, 0, :, 2:, 2:] = torch.tensor([0.0, 1.0])
    frame = extract_frame_patch_descriptors(
        tokens,
        level_position=0,
        patch_start_idx=2,
        patch_shape=(2, 2),
        token_half="frame",
    )
    global_descriptor = extract_frame_patch_descriptors(
        tokens,
        level_position=0,
        patch_start_idx=2,
        patch_shape=(2, 2),
        token_half="global",
    )
    torch.testing.assert_close(frame[0, 0], torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(global_descriptor[0, 0], torch.tensor([0.0, 1.0]))


def test_mask_support_and_area_exact_shift() -> None:
    masks = torch.zeros(2, 2, 8, 8, dtype=torch.bool)
    masks[:, 0, :4, :4] = True
    scores = torch.tensor([[0.9, 0.2], [0.8, 0.1]])
    support = masks_to_patch_support(
        masks,
        scores,
        patch_shape=(2, 2),
        score_threshold=0.5,
        minimum_patch_coverage=0.5,
        erosion_pixels=0,
    )
    assert tuple(support.shape) == (2, 2, 2, 2)
    assert int(support[:, 0].sum()) == 2
    assert not bool(support[:, 1].any())
    shifted = shift_patch_mask_exact(support[0, 0], shift_y=1, shift_x=1)
    assert int(shifted.sum()) == int(support[0, 0].sum())


def test_mutual_nearest_correspondence() -> None:
    query = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    search = torch.tensor([[0.99, 0.01], [0.01, 0.99], [-1.0, 0.0]])
    query = torch.nn.functional.normalize(query, dim=-1)
    search = torch.nn.functional.normalize(search, dim=-1)
    matches = mutual_nearest_matches(
        query,
        search,
        minimum_similarity=0.9,
        minimum_margin=0.1,
    )
    assert matches.query_positions.tolist() == [0, 1]
    assert matches.search_positions.tolist() == [0, 1]


def test_two_view_ray_triangulation() -> None:
    poses = torch.eye(4).repeat(2, 1, 1)
    poses[1, 0, 3] = -1.0
    intrinsics = torch.tensor(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]
    ).repeat(2, 1, 1)
    pixels = torch.tensor([[50.0, 50.0], [30.0, 50.0]])
    result = triangulate_camera_rays(
        pixels,
        torch.tensor([0, 1]),
        poses,
        intrinsics,
    )
    torch.testing.assert_close(
        result.point_world,
        torch.tensor([0.0, 0.0, 5.0]),
        rtol=1e-5,
        atol=1e-5,
    )
    assert result.finite
    assert result.positive_depth_rate == 1.0
    assert result.maximum_ray_angle_degrees > 10.0
    assert result.mean_reprojection_error_px < 1e-5
    centers = patch_centers((2, 2), (20, 20))
    torch.testing.assert_close(centers[0], torch.tensor([4.5, 4.5], dtype=torch.float64))


if __name__ == "__main__":
    main()
