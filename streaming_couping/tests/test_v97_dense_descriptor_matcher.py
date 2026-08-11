from __future__ import annotations

import torch

from streaming_couping.scripts.run_v92_support_factorization import SupportData
from streaming_couping.scripts.run_v97_dense_descriptor_causality import (
    DenseData,
    DenseRecord,
    _training_batch,
    load_v97_config,
)
from streaming_couping.src.v90_epipolar_geometry import SurfaceCorrespondences
from streaming_couping.src.v96_dense_grid_decoder import dense_grid_normalized
from streaming_couping.src.v97_dense_descriptor_matcher import (
    DenseMatcherConfig,
    DenseSubgridMatcher,
    build_dense_match_target,
    coordinate_descriptors,
    decode_dense_matches,
    dense_match_loss,
    deterministic_channel_permutation,
)


def test_dense_target_coarse_cell_and_subcell_offset() -> None:
    target_uv = torch.tensor([[[5.5, 4.0], [9.0, 9.0]]], dtype=torch.float64)
    valid = torch.tensor([[True, True]])
    visible = torch.tensor([[True, False]])
    target = build_dense_match_target(
        target_uv,
        query_valid=valid,
        visible=visible,
        grid_size=(3, 3),
        image_size=(11, 11),
    )
    assert target.coarse_index.tolist() == [[4, 9]]
    assert torch.allclose(
        target.offset_normalized[0, 0], torch.tensor([0.2, -0.4])
    )


def test_dense_matcher_shapes_gradients_and_bounded_decode() -> None:
    config = DenseMatcherConfig(
        canonical_dim=8,
        projection_dim=4,
        offset_hidden_dim=6,
        temperature=0.2,
    )
    model = DenseSubgridMatcher(config)
    query = torch.randn(2, 3, 8)
    key = torch.randn(2, 9, 8)
    query_valid = torch.tensor([[True, True, True], [True, True, False]])
    key_valid = torch.ones(2, 9, dtype=torch.bool)
    target_uv = torch.tensor(
        [
            [[1.0, 1.0], [5.0, 4.0], [8.0, 7.0]],
            [[2.0, 3.0], [7.0, 8.0], [0.0, 0.0]],
        ],
        dtype=torch.float64,
    )
    target = build_dense_match_target(
        target_uv,
        query_valid=query_valid,
        visible=query_valid,
        grid_size=(3, 3),
        image_size=(10, 10),
    )
    output = model(query, key, query_valid, key_valid)
    assert output["logits"].shape == (2, 3, 10)
    result = dense_match_loss(model, output, target)
    result.loss.backward()
    assert float(model.query_projection.weight.grad.abs().sum()) > 0.0
    assert float(model.key_projection.weight.grad.abs().sum()) > 0.0
    assert any(
        value.grad is not None and float(value.grad.abs().sum()) > 0.0
        for value in model.offset_head.parameters()
    )
    decoded = decode_dense_matches(
        model,
        output,
        grid_size=(3, 3),
        image_size=(10, 10),
        query_valid=query_valid,
    )
    assert decoded.history_uv_pixels.shape == (2, 3, 2)
    assert bool((decoded.history_uv_pixels >= 0.0).all())
    assert bool((decoded.history_uv_pixels <= 9.0).all())


def test_coordinate_control_and_permutation_are_deterministic() -> None:
    grid = dense_grid_normalized((4, 5)).float()
    first = coordinate_descriptors(grid, channels=16)
    second = coordinate_descriptors(grid, channels=16)
    assert first.shape == (20, 16)
    assert torch.equal(first, second)
    permutation = deterministic_channel_permutation(16)
    assert torch.equal(permutation, deterministic_channel_permutation(16))
    assert sorted(permutation.tolist()) == list(range(16))


def test_runner_batches_full_5184_key_bank_without_pose_inputs() -> None:
    config = load_v97_config(
        "streaming_couping/configs/v97_dense_descriptor_causality.yaml"
    )
    tokens = 72 * 72
    support = SupportData(
        frames=(90, 105, 120),
        image_size=(72, 96),
        masks=torch.ones(3, 1, 72, 96, dtype=torch.bool),
        local_uv=torch.empty(3, 1, 0, 2),
        local_valid=torch.empty(3, 1, 0, dtype=torch.bool),
        history_bank=torch.empty(3, 1, 0, dtype=torch.long),
        world_points=torch.empty(3, 72, 96, 3),
        depth=torch.empty(3, 72, 96),
        global_w2c=torch.eye(4).repeat(3, 1, 1),
        intrinsics=torch.eye(3).repeat(3, 1, 1),
        baseline=torch.eye(4).repeat(3, 1, 1),
        target=torch.eye(4).repeat(3, 1, 1),
    )
    grid = dense_grid_normalized((72, 72)).float()
    data = DenseData(
        support=support,
        grid_uv_normalized=grid,
        grid_uv_pixels=torch.empty(tokens, 2),
        sam_features=torch.randn(3, tokens, 8),
        stream_features=torch.randn(3, tokens, 10),
        coordinate_features=coordinate_descriptors(grid, channels=256),
    )
    current_uv = torch.tensor(
        [[0.0, 0.0], [10.0, 10.0], [20.0, 20.0], [30.0, 30.0]],
        dtype=torch.float64,
    )
    surface = SurfaceCorrespondences(
        current_frame=2,
        history_frame=1,
        slot=-1,
        current_uv=current_uv,
        history_uv=current_uv + torch.tensor([1.2, -0.4]),
        weights=torch.ones(4, dtype=torch.float64),
        depth_residual_metric=torch.zeros(4, dtype=torch.float64),
        sampled_queries=4,
        projected_in_bounds=4,
        visible_queries=4,
    )
    record = DenseRecord(2, 1, torch.tensor([0, 1, 72, 73]), surface)
    query, key, query_valid, key_valid, target = _training_batch(
        [record],
        architecture="sam_dense",
        data=data,
        config=config,
        device=torch.device("cpu"),
    )
    assert query.shape == (1, 4, 8)
    assert key.shape == (1, tokens, 8)
    assert query_valid.all() and key_valid.all()
    assert target.coarse_index.shape == (1, 4)
