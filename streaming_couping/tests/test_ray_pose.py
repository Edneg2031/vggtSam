import pytest
import torch

from streaming_couping.src.learned_pose import ray_pose
from streaming_couping.src.learned_pose.config import RayPoseConfig
from streaming_couping.src.learned_pose.ray_pose import (
    FINAL_RAY_POSE_NAME,
    REFERENCE_BLEND_ROLE,
    RayPoseResult,
    _accept_center_fit,
    _fit_angular_huber_center,
    _historical_correspondences,
    recover_final_ray_pose,
    reference_blend_pose_name,
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


def test_historical_correspondences_reject_unrelated_geometry() -> None:
    current = torch.randn(256, 3, dtype=torch.float64)
    historical = current + torch.tensor(
        [5.0, 0.0, 0.0],
        dtype=torch.float64,
    )
    pixels = torch.randn(256, 2, dtype=torch.float64)
    weights = torch.ones(256, dtype=torch.float64)

    matched, matched_pixels, matched_weights = _historical_correspondences(
        current,
        pixels,
        weights,
        historical,
        config=RayPoseConfig(
            historical_min_correspondences=32,
            historical_max_points_per_instance=512,
            max_center_shift=0.75,
            historical_min_distance=0.01,
            historical_object_ratio=0.01,
        ),
    )

    assert matched.shape == (0, 3)
    assert matched_pixels.shape == (0, 2)
    assert matched_weights.shape == (0,)


def test_center_fit_policy_rejects_excessive_shift() -> None:
    fit = {
        "solver_accepted": True,
        "status": "accepted_angular_huber",
        "point_residual_rmse": 0.01,
    }
    accepted, reasons = _accept_center_fit(
        fit,
        proposed_shift=1.0,
        config=RayPoseConfig(max_center_shift=0.2),
    )
    assert not accepted
    assert reasons == ["center_shift_above_limit"]


@pytest.mark.parametrize(
    ("blend", "expected"),
    [
        (0.25, "ray_current_refined_preserve_reference_blend_025"),
        (0.50, "ray_current_refined_preserve_reference_blend_050"),
        (0.75, "ray_current_refined_preserve_reference_blend_075"),
        (1.00, "ray_current_refined_preserve_reference_blend_100"),
    ],
)
def test_reference_blend_pose_name_is_stable(
    blend: float,
    expected: str,
) -> None:
    assert reference_blend_pose_name(blend) == expected


def test_reference_blend_sweep_has_distinct_anchored_results(
    monkeypatch,
) -> None:
    calls = []

    def fake_recover(**kwargs):
        calls.append(kwargs)
        return RayPoseResult(
            name=kwargs.get("name_override") or FINAL_RAY_POSE_NAME,
            role=kwargs.get("role_override") or "deployable",
            pose_encoding=torch.zeros(1),
            diagnostics=(),
        )

    monkeypatch.setattr(ray_pose, "_recover_current_pointmap_pose", fake_recover)
    results = recover_final_ray_pose(
        batch={},
        baseline_outputs={},
        refined_outputs={},
        config=RayPoseConfig(
            preserve_reference=False,
            blend=1.0,
            solver_modes=("current_refined",),
            reference_blend_values=(0.25, 0.5, 0.75, 1.0),
        ),
    )

    assert [result.name for result in results] == [
        FINAL_RAY_POSE_NAME,
        reference_blend_pose_name(0.25),
        reference_blend_pose_name(0.50),
        reference_blend_pose_name(0.75),
        reference_blend_pose_name(1.00),
    ]
    assert calls[0]["config"].preserve_reference is False
    assert all(call["config"].preserve_reference for call in calls[1:])
    assert all(
        result.role == REFERENCE_BLEND_ROLE for result in results[1:]
    )
