#!/usr/bin/env python3
"""Dependency-light V9 A0/A1 tensor, gradient and causality smoke."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from streaming_couping.scripts.run_v90_local_token_matcher import (
    FRAME_COLUMNS,
    SUMMARY_COLUMNS,
    _select_edge_outcome_indices,
    _summary_row,
)
from streaming_couping.src.v90_epipolar_geometry import (
    LocalTokenReprojection,
    VisibilityConfig,
    causal_mask_history_indices,
    local_token_reprojection_labels,
)
from streaming_couping.src.v90_explicit_matcher import (
    ExplicitLocalMatcher,
    MatcherConfig,
    build_soft_match_target,
    canonicalize_descriptor_channels,
    correspondence_loss,
    probability_to_correspondences,
    uniform_match_probability,
)


def main() -> None:
    torch.manual_seed(901)
    _smoke_local32_reprojection()
    _smoke_matcher_gradient_and_loss()
    _smoke_dustbin_and_conversion()
    _smoke_edge_selection()
    _smoke_output_schemas()
    print("V9 local-token matcher smoke passed")


def _smoke_local32_reprojection() -> None:
    sequence, slots, height, width, points = 3, 2, 24, 32, 32
    masks = torch.zeros(sequence, slots, height, width, dtype=torch.bool)
    masks[:, 0, 2:-2, 2:-2] = True
    masks[1:, 1, 6:-6, 8:-8] = True
    history = causal_mask_history_indices(masks, max_history=2)
    _require(int(history[1, 1, 0]) == -1, "late birth cannot read itself")
    _require(int(history[2, 1, 0]) == 1, "late birth is readable later")

    focal = 40.0
    intrinsics = torch.tensor(
        [[focal, 0.0, 15.5], [0.0, focal, 11.5], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    ).repeat(sequence, 1, 1)
    y, x = torch.meshgrid(
        torch.arange(height, dtype=torch.float64),
        torch.arange(width, dtype=torch.float64),
        indexing="ij",
    )
    z = torch.full_like(x, 3.0)
    world = torch.stack(
        [(x - 15.5) * z / focal, (y - 11.5) * z / focal, z], dim=-1
    ).repeat(sequence, 1, 1, 1)
    depth = z.repeat(sequence, 1, 1)
    pose = torch.eye(4, dtype=torch.float64).repeat(sequence, 1, 1)
    uv = torch.stack(
        [torch.linspace(-0.7, 0.7, points), torch.linspace(-0.6, 0.6, points)],
        dim=-1,
    )
    labels = local_token_reprojection_labels(
        current_frame=1,
        history_frame=0,
        slot=0,
        local_uv_normalized=uv,
        local_valid=torch.ones(points, dtype=torch.bool),
        masks=masks,
        world_points_metric=world,
        depth_metric=depth,
        global_world_to_camera=pose,
        intrinsics=intrinsics,
        config=VisibilityConfig(max_queries_per_instance=32),
    )
    _require(labels.query_count == points, "A0 keeps all fixed local32 queries")
    _require(labels.visible_count == points, "identity view keeps local32 visible")
    _require(
        float(
            (
                labels.current_uv[labels.target_visible]
                - labels.history_target_uv[labels.target_visible]
            ).abs().max()
        )
        < 1e-8,
        "A0 continuous reprojection preserves UV",
    )


def _smoke_matcher_gradient_and_loss() -> None:
    points, channels = 16, 24
    config = MatcherConfig(
        canonical_dim=24,
        projection_dim=16,
        target_sigma_pixels=2.0,
        target_radius_pixels=4.0,
        cycle_weight=0.05,
    )
    uv = torch.stack(
        [torch.linspace(-0.8, 0.8, points), torch.zeros(points)], dim=-1
    )
    pixel = torch.stack(
        [(uv[:, 0] + 1.0) * 31.5, (uv[:, 1] + 1.0) * 23.5], dim=-1
    ).double()
    labels = LocalTokenReprojection(
        current_frame=1,
        history_frame=0,
        slot=0,
        current_uv=pixel,
        history_target_uv=pixel.clone(),
        query_valid=torch.ones(points, dtype=torch.bool),
        target_visible=torch.ones(points, dtype=torch.bool),
        weights=torch.ones(points, dtype=torch.float64),
        depth_residual_metric=torch.zeros(points, dtype=torch.float64),
    )
    target = build_soft_match_target(
        labels,
        history_uv_normalized=uv,
        history_valid=torch.ones(points, dtype=torch.bool),
        image_size=(48, 64),
        config=config,
    )
    base = torch.randn(points, channels)
    query = base + 0.01 * torch.randn_like(base)
    key = base + 0.01 * torch.randn_like(base)
    valid = torch.ones(1, points, dtype=torch.bool)
    model = ExplicitLocalMatcher(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    losses = []
    for _ in range(25):
        forward = model(query[None], key[None], valid, valid)
        reverse = model(key[None], query[None], valid, valid)
        result = correspondence_loss(
            forward["probability"],
            SoftTargetBatch(target),
            reverse_probability=reverse["probability"],
            cycle_weight=config.cycle_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        result.loss.backward()
        if not losses:
            _require(
                model.query_projection.weight.grad is not None
                and float(model.query_projection.weight.grad.abs().sum()) > 0.0,
                "SAM Q projection receives gradient",
            )
            _require(
                model.key_projection.weight.grad is not None
                and float(model.key_projection.weight.grad.abs().sum()) > 0.0,
                "SAM K projection receives gradient",
            )
        optimizer.step()
        losses.append(float(result.loss.detach()))
    _require(losses[-1] < losses[0], "explicit matching loss decreases")
    _require(
        canonicalize_descriptor_channels(torch.randn(2, 31), 24).shape == (2, 24),
        "parameter-free descriptor canonicalization",
    )


def _smoke_dustbin_and_conversion() -> None:
    query_valid = torch.tensor([[True, True, False]])
    key_valid = torch.tensor([[True, True, False]])
    uniform = uniform_match_probability(query_valid, key_valid)
    _require(uniform.shape == (1, 3, 4), "uniform control includes dustbin")
    _require(int(uniform[0, 2].argmax()) == 3, "padding is exact dustbin")
    current_uv = torch.tensor([[2.0, 3.0], [4.0, 5.0], [0.0, 0.0]])
    history_uv = torch.tensor([[-0.5, 0.0], [0.5, 0.0], [0.0, 0.0]])
    rows = probability_to_correspondences(
        uniform[0],
        current_uv=current_uv,
        history_uv_normalized=history_uv,
        query_valid=query_valid[0],
        key_valid=key_valid[0],
        image_size=(20, 30),
    )
    _require(rows[0].shape[-1] == 2 and rows[2].ndim == 1, "solver UV conversion")


def _smoke_edge_selection() -> None:
    outcomes = [
        SimpleNamespace(
            history=23,
            estimate=SimpleNamespace(
                success=True, design_condition=1722.0, design_rank_ratio=0.00058
            ),
        ),
        SimpleNamespace(
            history=17,
            estimate=SimpleNamespace(
                success=True, design_condition=416.0, design_rank_ratio=0.00240
            ),
        ),
        SimpleNamespace(
            history=21,
            estimate=SimpleNamespace(
                success=False, design_condition=1.0, design_rank_ratio=1.0
            ),
        ),
    ]
    _require(
        _select_edge_outcome_indices(outcomes, policy="all_edges_mean") == [0, 1],
        "all-edges policy retains all successful causal edges",
    )
    _require(
        _select_edge_outcome_indices(
            outcomes, policy="best_design_condition_single"
        )
        == [1],
        "best-condition policy ignores failed edges and selects minimum condition",
    )
    tied = [
        SimpleNamespace(
            history=9,
            estimate=SimpleNamespace(
                success=True, design_condition=20.0, design_rank_ratio=0.1
            ),
        ),
        SimpleNamespace(
            history=7,
            estimate=SimpleNamespace(
                success=True, design_condition=20.0, design_rank_ratio=0.2
            ),
        ),
        SimpleNamespace(
            history=5,
            estimate=SimpleNamespace(
                success=True, design_condition=20.0, design_rank_ratio=0.2
            ),
        ),
    ]
    _require(
        _select_edge_outcome_indices(
            tied, policy="best_design_condition_single"
        )
        == [2],
        "best-condition tie-break is rank ratio then causal history index",
    )
    failed = [
        SimpleNamespace(
            history=0,
            estimate=SimpleNamespace(
                success=False, design_condition=1.0, design_rank_ratio=1.0
            ),
        )
    ]
    _require(
        not _select_edge_outcome_indices(
            failed, policy="best_design_condition_single"
        ),
        "best-condition policy has exact inactive fallback with no solved edge",
    )


def _smoke_output_schemas() -> None:
    frame = {
        "stage": "A0",
        "fold": "smoke",
        "architecture": "local32_same_support_oracle",
        "variant": "all_edges_mean",
        "edge_selection_policy": "all_edges_mean",
        "sequence_index": 2,
        "frame_index": 120,
        "history_edges": 1,
        "solved_edges": 1,
        "pose_edges_used": 1,
        "selected_history_sequence_indices": "1",
        "selected_history_frame_indices": "105",
        "active": 1,
        "supervised_queries": 32,
        "visible_queries": 24,
        "visible_key_supported_queries": 24,
        "accepted_correspondences": 24,
        "pck_correct": 24,
        "epe_sum_pixels": 0.0,
        "dustbin_correct": 8,
        "dustbin_queries": 8,
        "mean_sampson_rmse": 1e-4,
        "raw_edge_rotation_error_deg": 2.0,
        "refined_edge_rotation_error_deg": 1.0,
        "raw_edge_translation_direction_error_deg": 4.0,
        "refined_edge_translation_direction_error_deg": 2.0,
        "raw_relative_aggregate_deg": 6.0,
        "refined_relative_aggregate_deg": 3.0,
        "relative_aggregate_worse": 0,
    }
    _require(set(frame) == set(FRAME_COLUMNS), "frame CSV schema")
    summary = _summary_row(
        stage="A0",
        fold_name="smoke",
        architecture="local32_same_support_oracle",
        variant="all_edges_mean",
        edge_selection_policy="all_edges_mean",
        descriptor_source="none_gt_label_only",
        train_frames=(),
        test_frames=(120,),
        state=None,
        frames=[frame],
        pck_threshold=8.0,
        control_support_exact=1,
        matcher_frozen_exact=1,
        perturbed_pairs=0,
    )
    _require(set(summary) == set(SUMMARY_COLUMNS), "summary CSV schema")


def SoftTargetBatch(target):
    """Add the batch dimension while retaining the production dataclass."""

    return type(target)(
        probability=target.probability[None],
        supervised=target.supervised[None],
        visible_with_key_support=target.visible_with_key_support[None],
        target_uv=target.target_uv[None],
        nearest_key_distance_pixels=target.nearest_key_distance_pixels[None],
    )


def _require(condition: bool, label: str) -> None:
    if not condition:
        raise RuntimeError(f"V9 Stage A smoke failed: {label}")


if __name__ == "__main__":
    main()
