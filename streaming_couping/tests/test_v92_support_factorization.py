from __future__ import annotations

import torch

from streaming_couping.src.learned_pose.observations import _farthest_uv_indices
from streaming_couping.src.v90_epipolar_geometry import LocalTokenReprojection
from streaming_couping.src.v92_support_factorization import match_discrete_support


def _normalized(pixels: torch.Tensor, image_size=(64, 128)) -> torch.Tensor:
    height, width = image_size
    return torch.stack(
        [
            pixels[:, 0] / (width - 1) * 2.0 - 1.0,
            pixels[:, 1] / (height - 1) * 2.0 - 1.0,
        ],
        dim=-1,
    ).float()


def _labels(targets: torch.Tensor) -> LocalTokenReprojection:
    count = int(targets.shape[0])
    valid = torch.ones(count, dtype=torch.bool)
    return LocalTokenReprojection(
        current_frame=2,
        history_frame=1,
        slot=0,
        current_uv=targets.double().clone(),
        history_target_uv=targets.double(),
        query_valid=valid,
        target_visible=valid.clone(),
        weights=torch.ones(count, dtype=torch.float64),
        depth_residual_metric=torch.zeros(count, dtype=torch.float64),
    )


def test_nested_farthest_uv_support_is_prefix_exact() -> None:
    generator = torch.Generator().manual_seed(92)
    uv = torch.rand((300, 2), generator=generator)
    selected_32 = _farthest_uv_indices(uv, count=32)
    selected_256 = _farthest_uv_indices(uv, count=256)
    assert torch.equal(selected_32, selected_256[:32])


def test_nearest_reports_collisions_and_unique_rules_remove_them() -> None:
    targets = torch.tensor([[10.0, 20.0], [11.0, 20.0], [40.0, 20.0]])
    keys = _normalized(torch.tensor([[10.0, 20.0], [40.0, 20.0]]))
    labels = _labels(targets)
    common = {
        "history_uv_normalized": keys,
        "history_valid": torch.ones(2, dtype=torch.bool),
        "image_size": (64, 128),
        "match_radius_pixels": 12.0,
        "pck_threshold_pixels": 8.0,
    }
    nearest = match_discrete_support(labels, strategy="nearest", **common)
    mutual = match_discrete_support(labels, strategy="mutual", **common)
    greedy = match_discrete_support(labels, strategy="greedy_unique", **common)
    assert nearest.diagnostics.accepted_correspondences == 3
    assert nearest.diagnostics.unique_history_keys == 2
    assert nearest.diagnostics.nearest_collisions == 1
    assert mutual.diagnostics.accepted_correspondences == 2
    assert greedy.diagnostics.accepted_correspondences == 2
    assert mutual.diagnostics.unique_history_keys == 2
    assert greedy.diagnostics.unique_history_keys == 2


def test_greedy_unique_can_recover_more_pairs_than_mutual() -> None:
    targets = torch.tensor([[0.0, 20.0], [3.0, 20.0]])
    keys = _normalized(torch.tensor([[2.0, 20.0], [100.0, 20.0]]))
    labels = _labels(targets)
    common = {
        "history_uv_normalized": keys,
        "history_valid": torch.ones(2, dtype=torch.bool),
        "image_size": (64, 128),
        "match_radius_pixels": 100.0,
        "pck_threshold_pixels": 8.0,
    }
    mutual = match_discrete_support(labels, strategy="mutual", **common)
    greedy = match_discrete_support(labels, strategy="greedy_unique", **common)
    assert mutual.diagnostics.accepted_correspondences == 1
    assert greedy.diagnostics.accepted_correspondences == 2
    assert greedy.selected_key_indices.unique().numel() == 2


def test_no_valid_history_key_returns_exact_empty_support() -> None:
    labels = _labels(torch.tensor([[10.0, 20.0], [30.0, 20.0]]))
    result = match_discrete_support(
        labels,
        history_uv_normalized=_normalized(
            torch.tensor([[10.0, 20.0], [30.0, 20.0]])
        ),
        history_valid=torch.zeros(2, dtype=torch.bool),
        image_size=(64, 128),
        strategy="nearest",
    )
    assert result.correspondences.count == 0
    assert result.diagnostics.visible_queries == 2
    assert result.diagnostics.accepted_correspondences == 0
