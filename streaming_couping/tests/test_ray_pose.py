import torch

from streaming_couping.src.learned_pose.config import RayPoseConfig
from streaming_couping.src.learned_pose.ray_pose import (
    _fit_angular_huber_center,
)


def test_angular_huber_ray_center_recovers_known_center_with_outliers() -> None:
    torch.manual_seed(11)
    true_center = torch.tensor([0.20, -0.10, 0.30], dtype=torch.float64)
    directions = torch.randn(2048, 3, dtype=torch.float64)
    directions = directions / torch.linalg.vector_norm(
        directions,
        dim=-1,
        keepdim=True,
    )
    ranges = 1.0 + 2.0 * torch.rand(2048, dtype=torch.float64)
    points = true_center + ranges[:, None] * directions
    # Ten percent of the learned pointmap is deliberately inconsistent with
    # its pixel ray.  The angular Huber IRLS should retain the common center.
    points[:205] += 0.20 * torch.randn(205, 3, dtype=torch.float64)
    fit = _fit_angular_huber_center(
        points,
        directions,
        torch.ones(2048, dtype=torch.float64),
        candidate_points=2048,
        fallback_center=torch.zeros(3, dtype=torch.float64),
        config=RayPoseConfig(
            min_points=128,
            max_points=4096,
            max_iterations=8,
            angular_huber_delta=0.02,
            angular_min_range=0.05,
        ),
    )

    assert fit["solver_accepted"]
    assert float(torch.linalg.vector_norm(fit["center"] - true_center)) < 0.02
    assert fit["angular_residual_rmse"] < 0.10


def test_angular_huber_ray_center_falls_back_with_too_few_points() -> None:
    center = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
    points = torch.zeros(4, 3, dtype=torch.float64)
    directions = torch.eye(3, dtype=torch.float64)[[0, 1, 2, 0]]
    fit = _fit_angular_huber_center(
        points,
        directions,
        torch.ones(4, dtype=torch.float64),
        candidate_points=4,
        fallback_center=center,
        config=RayPoseConfig(min_points=8, max_points=32),
    )

    assert not fit["solver_accepted"]
    assert torch.equal(fit["center"], center)
    assert fit["status"].startswith("fallback_insufficient_points")
