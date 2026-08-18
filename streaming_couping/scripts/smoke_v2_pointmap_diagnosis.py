#!/usr/bin/env python3
"""CPU-only smoke checks for the V2 diagnosis geometry."""

from __future__ import annotations

import torch

from streaming_couping.src.pointmap_diagnosis import (
    _robust_affine_depth_alignment,
    binary_dilate,
    binary_erode,
    native_pose_to_metric_world,
    region_masks,
    unproject_z_depth,
)
from streaming_couping.src.triangulation_probe import (
    PairMatches,
    equalize_query_support,
    project_oracle_correspondence,
    project_world_points,
    triangulate_two_view,
    triangulation_gate,
)


def main() -> None:
    test_native_pose_sim3_change_of_world()
    test_gt_depth_pose_k_closure()
    test_reference_depth_affine()
    test_region_partition()
    test_two_view_triangulation()
    test_oracle_two_endpoint_projection()
    test_equal_query_support()
    print("V2 pointmap diagnosis geometry smoke passed")


def test_native_pose_sim3_change_of_world() -> None:
    angle = torch.tensor(0.3)
    rotation = torch.tensor(
        [
            [torch.cos(angle), -torch.sin(angle), 0.0],
            [torch.sin(angle), torch.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translation = torch.tensor([1.0, -0.5, 2.0])
    native_pose = torch.eye(4)[None]
    native_pose[0, :3, 3] = torch.tensor([0.2, -0.1, 0.4])
    metric_pose = native_pose_to_metric_world(
        native_pose, scale=2.5, rotation=rotation, translation=translation
    )
    native_point = torch.tensor([[0.4, 0.7, 3.0]])
    metric_point = 2.5 * (native_point @ rotation.T) + translation
    camera_native = native_point @ native_pose[0, :3, :3].T + native_pose[0, :3, 3]
    camera_metric = metric_point @ metric_pose[0, :3, :3].T + metric_pose[0, :3, 3]
    assert torch.allclose(camera_metric, 2.5 * camera_native, atol=1e-5)


def test_gt_depth_pose_k_closure() -> None:
    depth = torch.full((2, 7, 9), 4.0)
    intrinsics = torch.tensor(
        [[[80.0, 0.0, 4.5], [0.0, 80.0, 3.5], [0.0, 0.0, 1.0]]]
    ).repeat(2, 1, 1)
    pose = torch.eye(4)[None].repeat(2, 1, 1)
    pose[1, 0, 3] = -0.75
    points = unproject_z_depth(depth, intrinsics, pose)
    for frame in range(2):
        projected, z = project_world_points(points[frame].reshape(-1, 3), intrinsics[frame], pose[frame])
        y, x = torch.meshgrid(torch.arange(7), torch.arange(9), indexing="ij")
        target = torch.stack((x.reshape(-1), y.reshape(-1)), dim=1).float()
        assert torch.allclose(projected, target, atol=1e-4)
        assert torch.allclose(z, depth[frame].reshape(-1), atol=1e-5)


def test_reference_depth_affine() -> None:
    source = torch.linspace(0.5, 5.0, 4096).reshape(64, 64)
    target = 2.25 * source - 0.35
    target = target.clone()
    target.reshape(-1)[::31] += 4.0
    scale, shift, rmse = _robust_affine_depth_alignment(source, target)
    assert abs(scale - 2.25) < 1e-4
    assert abs(shift + 0.35) < 1e-4
    assert rmse < 1e-4


def test_region_partition() -> None:
    ownership = torch.full((2, 20, 24), -1, dtype=torch.long)
    ownership[:, 5:15, 6:18] = 0
    masks = region_masks(ownership, boundary_width=2)
    stack = torch.stack(tuple(masks.values())).long()
    assert bool((stack.sum(dim=0) == 1).all())
    assert int(binary_erode(ownership >= 0, 2).sum()) < int((ownership >= 0).sum())
    assert int(binary_dilate(ownership >= 0, 2).sum()) > int((ownership >= 0).sum())


def test_two_view_triangulation() -> None:
    intrinsics = torch.tensor(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]]
    )
    first = torch.eye(4)
    second = torch.eye(4)
    second[0, 3] = -1.0
    truth = torch.tensor([[0.25, -0.1, 5.0], [-0.4, 0.3, 7.0]])
    first_xy, _ = project_world_points(truth, intrinsics, first)
    second_xy, _ = project_world_points(truth, intrinsics, second)
    result = triangulate_two_view(first_xy, second_xy, intrinsics, intrinsics, first, second)
    assert torch.allclose(result.points, truth, atol=1e-4)
    valid = triangulation_gate(
        result,
        min_ray_angle_degrees=0.1,
        max_ray_angle_degrees=60.0,
        max_condition=1e6,
        max_reprojection_px=0.01,
    )
    assert bool(valid.all())


def test_oracle_two_endpoint_projection() -> None:
    height, width = 21, 41
    intrinsics = torch.tensor(
        [[100.0, 0.0, 20.0], [0.0, 100.0, 10.0], [0.0, 0.0, 1.0]]
    )
    current_pose = torch.eye(4)
    history_pose = torch.eye(4)
    history_pose[0, 3] = -0.5
    depth = torch.full((2, height, width), 5.0)
    poses = torch.stack((current_pose, history_pose))
    intrinsics_sequence = intrinsics[None].repeat(2, 1, 1)
    target_world = unproject_z_depth(depth, intrinsics_sequence, poses)

    # Mimic a small processed-GT grid inconsistency.  Oracle DLT must use the
    # projection of this exact world point at both endpoints; the integer query
    # remains only the pixel at which raw/GT point errors are evaluated.
    query = torch.tensor([[25.0, 10.0]])
    target_world[0, 10, 25] += torch.tensor([0.01, 0.005, 0.0])
    matches = project_oracle_correspondence(
        current_xy=query,
        current_target_world=target_world[0],
        history_target_world=target_world[1],
        history_mask=torch.ones(height, width, dtype=torch.bool),
        history_intrinsics=intrinsics,
        history_world_to_camera=history_pose,
        current_intrinsics=intrinsics,
        current_world_to_camera=current_pose,
        depth_tolerance=0.01,
    )
    assert matches.current_xy.shape[0] == 1
    assert torch.equal(matches.evaluation_xy, query)
    assert not torch.equal(matches.current_xy, query)
    result = triangulate_two_view(
        matches.current_xy,
        matches.history_xy,
        intrinsics,
        intrinsics,
        current_pose,
        history_pose,
    )
    assert torch.allclose(result.points[0], target_world[0, 10, 25], atol=1e-4)


def test_equal_query_support() -> None:
    correct = PairMatches(
        current_xy=torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]),
        history_xy=torch.tensor([[4.0, 1.0], [5.0, 2.0], [6.0, 3.0]]),
        query_indices=torch.tensor([0, 1, 2]),
        confidence=torch.tensor([0.9, 0.8, 0.7]),
    )
    control = PairMatches(
        current_xy=torch.tensor([[2.0, 2.0], [3.0, 3.0], [8.0, 8.0]]),
        history_xy=torch.tensor([[9.0, 2.0], [9.0, 3.0], [9.0, 8.0]]),
        query_indices=torch.tensor([1, 2, 7]),
        confidence=torch.tensor([0.6, 0.5, 0.4]),
    )
    left, right = equalize_query_support(correct, control)
    assert left.query_indices.tolist() == [1, 2]
    assert torch.equal(left.query_indices, right.query_indices)
    assert torch.equal(left.current_xy, right.current_xy)


if __name__ == "__main__":
    main()
