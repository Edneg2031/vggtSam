from __future__ import annotations

import torch

from streaming_couping.src.v90_epipolar_geometry import SurfaceCorrespondences
from streaming_couping.src.v96_dense_grid_decoder import (
    bbox_mask_stream,
    build_equal_count_grid_supports,
    current_farthest_indices,
    decode_history_grid,
    dense_grid_normalized,
    random_shifted_mask_stream,
)
from streaming_couping.scripts.run_v96_dense_grid_upper_bound import (
    SUMMARY_COLUMNS,
    _annotate_passes,
    _decision_markdown,
)
from streaming_couping.src.v74_temporal_protocol import FOLDS


def _surface(uv: torch.Tensor) -> SurfaceCorrespondences:
    uv = uv.double()
    return SurfaceCorrespondences(
        current_frame=4,
        history_frame=2,
        slot=-1,
        current_uv=uv,
        history_uv=uv + torch.tensor([2.3, -1.7], dtype=torch.float64),
        weights=torch.ones(len(uv), dtype=torch.float64),
        depth_residual_metric=torch.zeros(len(uv), dtype=torch.float64),
        sampled_queries=len(uv),
        projected_in_bounds=len(uv),
        visible_queries=len(uv),
    )


def test_dense_grid_matches_align_corners_order() -> None:
    grid = dense_grid_normalized((3, 4))
    assert grid.shape == (12, 2)
    assert torch.equal(grid[0], torch.tensor([-1.0, -1.0], dtype=torch.float64))
    assert torch.equal(grid[3], torch.tensor([1.0, -1.0], dtype=torch.float64))
    assert torch.equal(grid[-1], torch.tensor([1.0, 1.0], dtype=torch.float64))


def test_soft_bilinear_uses_grid_keys_and_is_exact() -> None:
    row = _surface(
        torch.tensor(
            [[10.0, 12.0], [25.0, 30.0], [70.0, 50.0]], dtype=torch.float64
        )
    )
    soft = decode_history_grid(
        row, mode="soft_bilinear_k4", grid_size=(8, 10), image_size=(80, 100)
    )
    hard = decode_history_grid(
        row, mode="hard_nearest", grid_size=(8, 10), image_size=(80, 100)
    )
    assert soft.keys_per_query == 4
    assert float(soft.errors_pixels.max()) < 1e-10
    assert torch.allclose(soft.correspondences.history_uv, row.history_uv)
    assert hard.keys_per_query == 1
    assert float(hard.errors_pixels.max()) > 0.0


def test_mask_controls_preserve_area_and_bbox_is_superset() -> None:
    masks = torch.zeros((3, 2, 30, 40), dtype=torch.bool)
    masks[:, 0, 5:10, 7:12] = True
    masks[:, 1, 14:18, 20:24] = True
    union = masks.any(dim=1, keepdim=True)
    bbox = bbox_mask_stream(masks)
    shifted = random_shifted_mask_stream(masks, seed=96)
    assert bool((bbox | ~union).all())
    assert torch.equal(shifted.flatten(1).sum(dim=1), union.flatten(1).sum(dim=1))
    assert not torch.equal(shifted, union)


def test_equal_count_scopes_and_current_only_selection() -> None:
    full_uv = torch.tensor(
        [[5.0 + 10 * x, 5.0 + 10 * y] for y in range(7) for x in range(9)]
    )
    region_uv = full_uv[::2]
    complement_uv = full_uv[1::2]
    full = _surface(full_uv)
    region = _surface(region_uv)
    complement = _surface(complement_uv)
    regions = {
        "sam_mask_balanced": (region, complement),
        "bbox_balanced": (region, complement),
        "random_shifted_mask_balanced": (region, complement),
    }
    supports = build_equal_count_grid_supports(
        full=full,
        regions=regions,
        target_count=20,
        image_size=(80, 100),
    )
    assert {value.count for value in supports.values()} == {20}
    assert supports["sam_mask_balanced"].region_rows == 10
    assert supports["sam_mask_balanced"].complement_rows == 10

    changed_history = _surface(full_uv)
    changed_history.history_uv = torch.flip(changed_history.history_uv, dims=(0,))
    assert torch.equal(
        current_farthest_indices(full, 20, (80, 100)),
        current_farthest_indices(changed_history, 20, (80, 100)),
    )


def test_unavailable_balanced_control_is_reported_without_aborting() -> None:
    full_uv = torch.tensor(
        [[5.0 + 10 * x, 5.0 + 10 * y] for y in range(4) for x in range(5)]
    )
    full = _surface(full_uv)
    scarce_region = _surface(full_uv[:5])
    empty_complement = _surface(full_uv[:0])
    feasible = (_surface(full_uv[:10]), _surface(full_uv[10:]))
    supports = build_equal_count_grid_supports(
        full=full,
        regions={
            "sam_mask_balanced": feasible,
            "bbox_balanced": (scarce_region, empty_complement),
            "random_shifted_mask_balanced": feasible,
        },
        target_count=10,
        image_size=(50, 60),
    )
    assert supports["bbox_balanced"].count == 5
    assert supports["full_grid"].count == 10
    assert supports["sam_mask_balanced"].count == 10


def test_dense_stage_gate_depends_on_full_grid_not_balanced_feasibility() -> None:
    rows = []
    for scope in (
        "full_grid",
        "sam_mask_balanced",
        "bbox_balanced",
        "random_shifted_mask_balanced",
    ):
        for decoder in ("continuous_gt", "hard_nearest", "soft_bilinear_k4"):
            for fold in FOLDS:
                row = dict.fromkeys(SUMMARY_COLUMNS, 0)
                full_positive = scope == "full_grid" and decoder in {
                    "continuous_gt",
                    "soft_bilinear_k4",
                }
                row.update(
                    {
                        "fold": fold.name,
                        "support_scope": scope,
                        "decoder": decoder,
                        "frames": 4,
                        "active_frames": 4,
                        "equal_count_exact": int(full_positive),
                        "raw_rotation_error_deg": 2.0,
                        "refined_rotation_error_deg": 1.0 if full_positive else 3.0,
                        "raw_translation_direction_error_deg": 2.0,
                        "refined_translation_direction_error_deg": (
                            1.0 if full_positive else 3.0
                        ),
                    }
                )
                rows.append(row)
    _annotate_passes(rows)
    decision = _decision_markdown(rows)
    assert "SAM-mask balanced equal-count feasible: `0`" in decision
    assert "dense-descriptor stage allowed: `1`" in decision
