from __future__ import annotations

import torch

from streaming_couping.src.v90_epipolar_geometry import LocalTokenReprojection
from streaming_couping.src.v90_explicit_matcher import (
    ExplicitLocalMatcher,
    MatcherConfig,
    build_soft_match_target,
    canonicalize_descriptor_channels,
    correspondence_loss,
    probability_to_correspondences,
)


def _fixture(points: int = 12):
    uv = torch.stack(
        [torch.linspace(-0.8, 0.8, points), torch.linspace(-0.5, 0.5, points)],
        dim=-1,
    )
    pixel = torch.stack(
        [(uv[:, 0] + 1.0) * 31.5, (uv[:, 1] + 1.0) * 23.5], dim=-1
    ).double()
    valid = torch.ones(points, dtype=torch.bool)
    labels = LocalTokenReprojection(
        current_frame=2,
        history_frame=1,
        slot=0,
        current_uv=pixel,
        history_target_uv=pixel.clone(),
        query_valid=valid,
        target_visible=valid,
        weights=torch.ones(points, dtype=torch.float64),
        depth_residual_metric=torch.zeros(points, dtype=torch.float64),
    )
    config = MatcherConfig(
        canonical_dim=16,
        projection_dim=8,
        target_sigma_pixels=2.0,
        target_radius_pixels=5.0,
    )
    target = build_soft_match_target(
        labels,
        history_uv_normalized=uv,
        history_valid=valid,
        image_size=(48, 64),
        config=config,
    )
    return config, labels, uv, valid, target


def test_matcher_shape_gradient_and_parameter_free_channel_adapter() -> None:
    torch.manual_seed(9)
    config, _, _, valid, target = _fixture()
    query = torch.randn(1, valid.numel(), 21)
    key = query.clone() + 0.01 * torch.randn_like(query)
    model = ExplicitLocalMatcher(config)
    output = model(query, key, valid[None], valid[None])
    assert output["probability"].shape == (1, valid.numel(), valid.numel() + 1)
    batched_target = type(target)(
        probability=target.probability[None],
        supervised=target.supervised[None],
        visible_with_key_support=target.visible_with_key_support[None],
        target_uv=target.target_uv[None],
        nearest_key_distance_pixels=target.nearest_key_distance_pixels[None],
    )
    reverse = model(key, query, valid[None], valid[None])
    loss = correspondence_loss(
        output["probability"],
        batched_target,
        reverse_probability=reverse["probability"],
        cycle_weight=config.cycle_weight,
    )
    loss.loss.backward()
    assert model.query_projection.weight.grad is not None
    assert model.query_projection.weight.grad.abs().sum() > 0
    assert model.key_projection.weight.grad is not None
    assert model.key_projection.weight.grad.abs().sum() > 0
    assert canonicalize_descriptor_channels(query, config.canonical_dim).shape[-1] == 16


def test_visibility_target_uses_dustbin_and_solver_conversion_has_no_threshold() -> None:
    config, labels, uv, valid, target = _fixture()
    labels.target_visible[-1] = False
    target = build_soft_match_target(
        labels,
        history_uv_normalized=uv,
        history_valid=valid,
        image_size=(48, 64),
        config=config,
    )
    assert int(target.probability[-1].argmax()) == valid.numel()
    model = ExplicitLocalMatcher(config)
    features = torch.randn(1, valid.numel(), 16)
    probability = model(features, features, valid[None], valid[None])["probability"][0]
    current, history, weights, accepted = probability_to_correspondences(
        probability,
        current_uv=labels.current_uv,
        history_uv_normalized=uv,
        query_valid=valid,
        key_valid=valid,
        image_size=(48, 64),
    )
    assert current.shape == history.shape
    assert weights.shape == accepted[accepted].shape
    assert accepted.dtype == torch.bool
