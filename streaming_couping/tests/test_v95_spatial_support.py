from __future__ import annotations

import torch

from streaming_couping.src.v90_epipolar_geometry import SurfaceCorrespondences
from streaming_couping.src.v95_spatial_support import (
    build_equal_count_supports,
    perturb_history_uv,
    region_mask_streams,
    spatial_farthest_indices,
    uv_hull_coverage,
)


def _surface(
    current_uv: torch.Tensor,
    *,
    current: int = 3,
    history: int = 1,
) -> SurfaceCorrespondences:
    uv = current_uv.double()
    count = len(uv)
    return SurfaceCorrespondences(
        current_frame=current,
        history_frame=history,
        slot=-1,
        current_uv=uv,
        history_uv=uv + torch.tensor([2.0, -1.0], dtype=torch.float64),
        weights=torch.ones(count, dtype=torch.float64),
        depth_residual_metric=torch.zeros(count, dtype=torch.float64),
        sampled_queries=count,
        projected_in_bounds=count,
        visible_queries=count,
    )


def test_equal_count_supports_balance_instance_and_background() -> None:
    instance_uv = torch.tensor(
        [[35 + 5 * (index % 4), 35 + 5 * (index // 4)] for index in range(12)]
    )
    full_uv = torch.tensor(
        [[5 + 15 * x, 5 + 15 * y] for y in range(6) for x in range(6)]
    )
    background_uv = torch.tensor(
        [
            [5 + 15 * x, 5 + 15 * y]
            for y in range(6)
            for x in range(6)
            if not (2 <= x <= 3 and 2 <= y <= 3)
        ]
    )
    mask = torch.zeros((100, 100), dtype=torch.bool)
    mask[30:61, 30:61] = True
    supports = build_equal_count_supports(
        instance=_surface(instance_uv),
        full_image_candidates=_surface(full_uv),
        background_candidates=_surface(background_uv),
        instance_union_mask=mask,
        image_size=(100, 100),
        background_fraction=0.5,
    )
    assert {value.count for value in supports.values()} == {12}
    hybrid = supports["instance_background_balanced"]
    assert hybrid.instance_rows == 6
    assert hybrid.background_rows == 6
    assert supports["instance_local32"].instance_rows == 12


def test_joint_view_farthest_sampling_and_noise_are_deterministic() -> None:
    uv = torch.tensor(
        [[float(index), float(index % 5)] for index in range(30)],
        dtype=torch.float64,
    )
    row = _surface(uv)
    first = spatial_farthest_indices(row, 8, (32, 40))
    second = spatial_farthest_indices(row, 8, (32, 40))
    assert torch.equal(first, second)
    assert first.unique().numel() == 8

    noisy_a, error_a = perturb_history_uv(row, sigma_pixels=0.5, seed=95)
    noisy_b, error_b = perturb_history_uv(row, sigma_pixels=0.5, seed=95)
    assert torch.equal(noisy_a.history_uv, noisy_b.history_uv)
    assert torch.equal(error_a, error_b)
    assert not torch.equal(noisy_a.history_uv, row.history_uv)


def test_region_masks_and_hull_coverage() -> None:
    masks = torch.zeros((2, 3, 20, 30), dtype=torch.bool)
    masks[:, 1, 5:10, 8:14] = True
    full, background = region_mask_streams(masks)
    assert full.shape == (2, 1, 20, 30)
    assert bool(full.all())
    assert not bool(background[:, :, 5:10, 8:14].any())

    corners = torch.tensor(
        [[0.0, 0.0], [29.0, 0.0], [29.0, 19.0], [0.0, 19.0]]
    )
    assert abs(uv_hull_coverage(corners, (20, 30)) - 1.0) < 1e-12
