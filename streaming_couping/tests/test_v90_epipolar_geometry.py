from __future__ import annotations

import torch

from streaming_couping.scripts.smoke_v90_epipolar_oracle import (
    _smoke_epipolar_pose,
    _smoke_exact_fallback,
    _smoke_runner_frame,
    _smoke_surface_reprojection_and_causality,
)
from streaming_couping.src.v90_epipolar_geometry import (
    EpipolarConfig,
    causal_mask_history_indices,
    estimate_relative_epipolar_pose,
)


def test_visible_surface_reprojection_and_dynamic_birth() -> None:
    _smoke_surface_reprojection_and_causality()


def test_perfect_calibrated_correspondence_recovers_pose() -> None:
    _smoke_epipolar_pose()


def test_unsolvable_edge_is_bit_exact_fallback() -> None:
    _smoke_exact_fallback()


def test_runner_frame_and_csv_schema() -> None:
    _smoke_runner_frame()


def test_history_bank_is_strictly_causal() -> None:
    masks = torch.zeros(5, 2, 3, 3, dtype=torch.bool)
    masks[[0, 2, 4], 0, 1, 1] = True
    masks[[1, 3], 1, 1, 1] = True
    bank = causal_mask_history_indices(masks, max_history=2)
    assert bank[0, 0].tolist() == [-1, -1]
    assert bank[2, 0].tolist() == [0, -1]
    assert bank[4, 0].tolist() == [2, 0]
    assert bank[1, 1].tolist() == [-1, -1]
    assert bank[3, 1].tolist() == [1, -1]


def test_degenerate_collinear_pixels_are_rejected() -> None:
    coordinate = torch.linspace(0.0, 20.0, 16, dtype=torch.float64)
    uv = torch.stack([coordinate, coordinate], dim=-1)
    result = estimate_relative_epipolar_pose(
        uv,
        uv,
        torch.ones(16, dtype=torch.float64),
        torch.eye(3, dtype=torch.float64),
        torch.eye(3, dtype=torch.float64),
        torch.eye(4, dtype=torch.float64),
        config=EpipolarConfig(),
    )
    assert not result.success
    assert result.reason == "degenerate_design_matrix"
