from __future__ import annotations

import torch

from streaming_couping.src.v90_epipolar_geometry import LocalTokenReprojection
from streaming_couping.src.v90_explicit_matcher import (
    MatcherConfig,
    build_soft_match_target,
)
from streaming_couping.src.v91_token_evidence import (
    audit_token_probability,
    decode_token_probability,
    hard_discrete_oracle_probability,
    normalized_uv_to_pixels,
    raw_cosine_probability,
)


def _fixture():
    pixels = torch.tensor([[10.0, 12.0], [50.0, 30.0]], dtype=torch.float64)
    normalized = torch.stack(
        [pixels[:, 0] / 63.0 * 2.0 - 1.0, pixels[:, 1] / 47.0 * 2.0 - 1.0],
        dim=-1,
    ).float()
    target_uv = torch.tensor(
        [[11.0, 12.0], [49.0, 30.0], [30.0, 44.0]], dtype=torch.float64
    )
    valid = torch.ones(3, dtype=torch.bool)
    labels = LocalTokenReprojection(
        current_frame=2,
        history_frame=1,
        slot=0,
        current_uv=target_uv.clone(),
        history_target_uv=target_uv,
        query_valid=valid,
        target_visible=valid.clone(),
        weights=torch.ones(3, dtype=torch.float64),
        depth_residual_metric=torch.zeros(3, dtype=torch.float64),
    )
    target = build_soft_match_target(
        labels,
        history_uv_normalized=normalized,
        history_valid=torch.ones(2, dtype=torch.bool),
        image_size=(48, 64),
        config=MatcherConfig(
            canonical_dim=4,
            projection_dim=2,
            target_sigma_pixels=3.0,
            target_radius_pixels=12.0,
        ),
    )
    return labels, target, normalized, valid


def test_normalized_uv_round_trip() -> None:
    _, _, normalized, _ = _fixture()
    pixels = normalized_uv_to_pixels(normalized, (48, 64))
    assert torch.allclose(
        pixels, torch.tensor([[10.0, 12.0], [50.0, 30.0]], dtype=torch.float64)
    )


def test_discrete_oracle_uses_actual_keys_and_dustbin() -> None:
    labels, target, history_uv, valid = _fixture()
    probability = hard_discrete_oracle_probability(target)
    prediction = decode_token_probability(
        probability,
        history_uv_normalized=history_uv,
        query_valid=valid,
        key_valid=torch.ones(2, dtype=torch.bool),
        image_size=(48, 64),
        mode="top1",
    )
    metrics = audit_token_probability(
        probability,
        labels=labels,
        target=target,
        prediction=prediction,
        pck_threshold_pixels=8.0,
        image_size=(48, 64),
    )
    assert prediction.accepted.tolist() == [True, True, False]
    assert metrics.visible_supported_queries == 2
    assert metrics.supported_pck_correct == 2
    assert metrics.dustbin_queries == metrics.dustbin_correct == 1


def test_raw_cosine_recovers_identity_and_masks_invalid_rows() -> None:
    features = torch.eye(4, dtype=torch.float32)[:3]
    query_valid = torch.tensor([True, True, False])
    key_valid = torch.tensor([True, True, True])
    probability = raw_cosine_probability(
        features,
        features,
        query_valid,
        key_valid,
        canonical_dim=4,
        temperature=0.07,
    )
    assert probability.shape == (3, 4)
    assert probability[:2].argmax(dim=-1).tolist() == [0, 1]
    assert int(probability[2].argmax()) == 3


def test_top1_and_soft_expectation_decode_differently() -> None:
    _, _, history_uv, _ = _fixture()
    probability = torch.tensor([[0.75, 0.25, 0.0]])
    kwargs = {
        "history_uv_normalized": history_uv,
        "query_valid": torch.ones(1, dtype=torch.bool),
        "key_valid": torch.ones(2, dtype=torch.bool),
        "image_size": (48, 64),
    }
    top1 = decode_token_probability(probability, mode="top1", **kwargs)
    soft = decode_token_probability(probability, mode="soft_expectation", **kwargs)
    assert top1.accepted.item() and soft.accepted.item()
    assert not torch.allclose(top1.predicted_uv, soft.predicted_uv)
