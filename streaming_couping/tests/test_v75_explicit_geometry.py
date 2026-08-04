import torch

from streaming_couping.src.explicit_pose_causality import (
    ExplicitPoseConfig,
    _bbox_masks,
    _random_shift_masks,
    _solve_frame_center,
)
from streaming_couping.src.instance_observations import (
    InstanceRefinementConfig,
    translation_icp,
)
from streaming_couping.src.learned_pose.config import RayPoseConfig


def test_region_controls_preserve_expected_support() -> None:
    mask = torch.zeros(2, 10, 14, dtype=torch.bool)
    mask[0, 2:6, 3:8] = True
    mask[1, 5:9, 8:12] = True
    bbox = _bbox_masks(mask)
    random = _random_shift_masks(mask)
    assert torch.equal(bbox, mask)
    assert int(random.sum()) == int(mask.sum())
    assert not torch.equal(random, mask)


def test_v5_ray_center_recovers_synthetic_center() -> None:
    height, width = 18, 24
    intrinsics = torch.tensor(
        [[20.0, 0.0, 11.5], [0.0, 20.0, 8.5], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    rows, columns = torch.meshgrid(
        torch.arange(height, dtype=torch.float64),
        torch.arange(width, dtype=torch.float64),
        indexing="ij",
    )
    pixels = torch.stack([columns, rows, torch.ones_like(columns)], dim=-1)
    directions = pixels @ torch.linalg.inv(intrinsics).T
    directions /= torch.linalg.vector_norm(directions, dim=-1, keepdim=True)
    target_center = torch.tensor([0.04, -0.03, 0.02], dtype=torch.float64)
    points = target_center + 2.0 * directions
    ray = RayPoseConfig(
        min_points=64,
        max_points=height * width,
        max_center_shift=0.15,
        max_residual_rmse=0.01,
    )
    config = ExplicitPoseConfig(
        point_confidence_threshold=0.3,
        track_confidence_threshold=0.5,
        matched_point_budget=height * width,
        max_center_shift=0.15,
        center_blend=1.0,
        ray=ray,
        refinement=InstanceRefinementConfig(min_instance_points=16),
    )
    center, accepted, _ = _solve_frame_center(
        baseline_world_to_camera=torch.eye(4, dtype=torch.float64)[:3],
        intrinsics=intrinsics,
        world_points=points.float(),
        confidence=torch.ones(height, width),
        masks=torch.ones(1, height, width, dtype=torch.bool),
        translation=torch.zeros(3),
        matched_budget=height * width,
        config=config,
    )
    assert accepted
    assert torch.linalg.vector_norm(center - target_center) < 1e-6


def test_icp_translation_has_the_ray_solver_sign() -> None:
    height, width = 18, 24
    intrinsics = torch.tensor(
        [[20.0, 0.0, 11.5], [0.0, 20.0, 8.5], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    rows, columns = torch.meshgrid(
        torch.arange(height, dtype=torch.float64),
        torch.arange(width, dtype=torch.float64),
        indexing="ij",
    )
    pixels = torch.stack([columns, rows, torch.ones_like(columns)], dim=-1)
    directions = pixels @ torch.linalg.inv(intrinsics).T
    directions /= torch.linalg.vector_norm(directions, dim=-1, keepdim=True)
    target_center = torch.tensor([0.04, -0.03, 0.02], dtype=torch.float64)
    persistent = target_center + 2.0 * directions
    correction = torch.tensor([0.025, -0.015, 0.01])
    current = persistent.float() - correction[None]
    refinement = InstanceRefinementConfig(min_instance_points=16)
    proposal = translation_icp(
        current.reshape(-1, 3),
        persistent.float().reshape(-1, 3),
        instance_id=0,
        config=refinement,
    )
    assert proposal.accepted
    assert torch.linalg.vector_norm(proposal.translation - correction) < 1e-5
    config = ExplicitPoseConfig(
        point_confidence_threshold=0.3,
        track_confidence_threshold=0.5,
        matched_point_budget=height * width,
        max_center_shift=0.15,
        center_blend=1.0,
        ray=RayPoseConfig(
            min_points=64,
            max_points=height * width,
            max_center_shift=0.15,
            max_residual_rmse=0.01,
        ),
        refinement=refinement,
    )
    center, accepted, _ = _solve_frame_center(
        baseline_world_to_camera=torch.eye(4, dtype=torch.float64)[:3],
        intrinsics=intrinsics,
        world_points=current,
        confidence=torch.ones(height, width),
        masks=torch.ones(1, height, width, dtype=torch.bool),
        translation=proposal.translation,
        matched_budget=height * width,
        config=config,
    )
    assert accepted
    assert torch.linalg.vector_norm(center - target_center) < 1e-6
