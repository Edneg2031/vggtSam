from __future__ import annotations

import torch

from streaming_couping.src.v90_epipolar_geometry import LocalTokenReprojection
from streaming_couping.src.v93_quantization_tolerance import (
    continuous_prediction,
    filter_prediction_by_oracle_error,
    hard_nearest_prediction,
    noisy_continuous_prediction,
    project_to_convex_hull_2d,
    soft_knn_convex_prediction,
)


IMAGE_SIZE = (64, 128)


def _normalized(pixels: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [pixels[:, 0] / 127.0 * 2.0 - 1.0, pixels[:, 1] / 63.0 * 2.0 - 1.0],
        dim=-1,
    ).float()


def _labels(target: torch.Tensor) -> LocalTokenReprojection:
    valid = torch.ones(target.shape[0], dtype=torch.bool)
    return LocalTokenReprojection(
        current_frame=2,
        history_frame=1,
        slot=0,
        current_uv=target.double().clone(),
        history_target_uv=target.double(),
        query_valid=valid,
        target_visible=valid.clone(),
        weights=torch.ones(target.shape[0], dtype=torch.float64),
        depth_residual_metric=torch.zeros(target.shape[0], dtype=torch.float64),
    )


def test_convex_projection_is_exact_inside_triangle_and_clamped_outside() -> None:
    triangle = torch.tensor([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    inside = project_to_convex_hull_2d(torch.tensor([2.0, 3.0]), triangle)
    outside = project_to_convex_hull_2d(torch.tensor([8.0, 8.0]), triangle)
    assert torch.allclose(inside, torch.tensor([2.0, 3.0], dtype=torch.float64))
    assert torch.allclose(outside, torch.tensor([5.0, 5.0], dtype=torch.float64))


def test_soft_convex_can_remove_hard_nearest_quantization() -> None:
    target = torch.tensor([[5.0, 5.0], [2.0, 3.0]])
    labels = _labels(target)
    keys = _normalized(
        torch.tensor([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
    )
    hard = hard_nearest_prediction(
        labels,
        history_uv_normalized=keys,
        history_valid=torch.ones(4, dtype=torch.bool),
        image_size=IMAGE_SIZE,
    )
    soft = soft_knn_convex_prediction(
        labels,
        history_uv_normalized=keys,
        history_valid=torch.ones(4, dtype=torch.bool),
        image_size=IMAGE_SIZE,
        neighbors=4,
    )
    assert hard.diagnostics.selected_epe_sum_pixels > 0.0
    assert soft.diagnostics.selected_epe_sum_pixels < 1e-8
    assert soft.diagnostics.exact_interpolations == 2


def test_error_filter_retains_only_requested_hard_rows() -> None:
    target = torch.tensor([[0.5, 0.0], [3.0, 0.0], [7.0, 0.0]])
    labels = _labels(target)
    keys = _normalized(torch.tensor([[0.0, 0.0], [10.0, 0.0]]))
    hard = hard_nearest_prediction(
        labels,
        history_uv_normalized=keys,
        history_valid=torch.ones(2, dtype=torch.bool),
        image_size=IMAGE_SIZE,
    )
    filtered = filter_prediction_by_oracle_error(
        hard,
        labels,
        max_error_pixels=2.0,
        image_size=IMAGE_SIZE,
    )
    assert hard.diagnostics.accepted_correspondences == 3
    assert filtered.diagnostics.accepted_correspondences == 1
    assert filtered.selected_query_indices.tolist() == [0]


def test_continuous_and_noise_are_deterministic() -> None:
    labels = _labels(torch.tensor([[20.0, 20.0], [40.0, 30.0]]))
    exact = continuous_prediction(labels, image_size=IMAGE_SIZE)
    first = noisy_continuous_prediction(
        labels, sigma_pixels=2.0, seed=93, image_size=IMAGE_SIZE
    )
    second = noisy_continuous_prediction(
        labels, sigma_pixels=2.0, seed=93, image_size=IMAGE_SIZE
    )
    assert exact.diagnostics.selected_epe_sum_pixels == 0.0
    assert torch.equal(first.correspondences.history_uv, second.correspondences.history_uv)
    assert first.diagnostics.selected_epe_sum_pixels > 0.0
