from __future__ import annotations

import math

import torch

from streaming_couping.src.v90_epipolar_geometry import (
    EpipolarConfig,
    relative_translation_direction_error_degrees,
)
from streaming_couping.src.v94_robust_epipolar import (
    ROBUST_SOLVERS,
    RobustEpipolarConfig,
    _sample_indices,
    estimate_robust_relative_pose,
)


def _rotation_y(angle: float) -> torch.Tensor:
    cosine, sine = math.cos(angle), math.sin(angle)
    return torch.tensor(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=torch.float64,
    )


def _w2c(rotation: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    output = torch.eye(4, dtype=torch.float64)
    output[:3, :3] = rotation
    output[:3, 3] = -(rotation @ center)
    return output


def _invert(transform: torch.Tensor) -> torch.Tensor:
    output = torch.eye(4, dtype=torch.float64)
    output[:3, :3] = transform[:3, :3].transpose(0, 1)
    output[:3, 3] = -(output[:3, :3] @ transform[:3, 3])
    return output


def _project(transform: torch.Tensor, points: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    camera = (transform[:3, :3] @ points.transpose(0, 1)).transpose(0, 1)
    camera = camera + transform[:3, 3]
    homogeneous = (k @ camera.transpose(0, 1)).transpose(0, 1)
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def _fixture():
    generator = torch.Generator().manual_seed(94)
    count = 80
    points = torch.randn((count, 3), generator=generator, dtype=torch.float64)
    points[:, :2] *= 0.5
    points[:, 2] = torch.linspace(3.0, 7.0, count)
    history = torch.eye(4, dtype=torch.float64)
    current = _w2c(
        _rotation_y(0.08), torch.tensor([0.4, 0.05, 0.02], dtype=torch.float64)
    )
    truth = history @ _invert(current)
    k = torch.tensor(
        [[420.0, 0.0, 320.0], [0.0, 415.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    current_uv = _project(current, points, k)
    history_uv = _project(history, points, k)
    l0 = _w2c(
        _rotation_y(0.10), torch.tensor([0.38, 0.08, 0.02], dtype=torch.float64)
    )
    return current_uv, history_uv, k, history @ _invert(l0), truth


def test_every_solver_recovers_exact_correspondence() -> None:
    current, history, k, l0, truth = _fixture()
    for solver in ROBUST_SOLVERS:
        result = estimate_robust_relative_pose(
            current,
            history,
            torch.ones(len(current), dtype=torch.float64),
            k,
            k,
            l0,
            solver=solver,
            epipolar_config=EpipolarConfig(),
            robust_config=RobustEpipolarConfig(),
            image_size=(480, 640),
        )
        assert result.estimate.success, (solver, result.estimate.reason)
        relative = result.estimate.rotation_current_to_history @ truth[:3, :3].T
        cosine = ((torch.trace(relative) - 1.0) * 0.5).clamp(-1.0, 1.0)
        assert float(torch.rad2deg(torch.acos(cosine))) < 0.1
        assert relative_translation_direction_error_degrees(
            result.estimate.rotation_current_to_history,
            result.estimate.translation_current_origin_in_history,
            truth,
        ) < 0.1


def test_ransac_is_deterministic_with_corrupted_rows() -> None:
    current, history, k, l0, _ = _fixture()
    history = history.clone()
    history[:12] = history[:12].flip(0)
    kwargs = {
        "current_uv": current,
        "history_uv": history,
        "weights": torch.ones(len(current), dtype=torch.float64),
        "current_intrinsics": k,
        "history_intrinsics": k,
        "l0_current_to_history": l0,
        "solver": "deterministic_ransac_inlier_refine",
        "epipolar_config": EpipolarConfig(),
        "robust_config": RobustEpipolarConfig(),
        "image_size": (480, 640),
    }
    first = estimate_robust_relative_pose(**kwargs)
    second = estimate_robust_relative_pose(**kwargs)
    assert first.consensus.hypotheses_tested == 128
    assert first.consensus.inliers == second.consensus.inliers
    assert first.consensus.msac_objective == second.consensus.msac_objective
    assert torch.equal(first.estimate.essential, second.estimate.essential)


def test_spatial_sample_is_unique_and_spread() -> None:
    coordinate = torch.arange(40, dtype=torch.float64)
    current = torch.stack([coordinate, coordinate.remainder(7)], dim=-1)
    history = current + torch.tensor([3.0, -2.0], dtype=torch.float64)
    sample = _sample_indices(
        current,
        history,
        generator=torch.Generator().manual_seed(94),
        sample_size=8,
        spatial=True,
        candidate_pool=32,
        image_size=(64, 128),
    )
    assert sample.shape == (8,)
    assert sample.unique().numel() == 8
