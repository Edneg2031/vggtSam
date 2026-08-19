#!/usr/bin/env python3
"""CPU smoke checks for the trust-aware residual Phase 1 head."""

from __future__ import annotations

import torch

from streaming_couping.src.trust_aware_residual import (
    TrustAwareResidualHead,
    apply_similarity,
    heteroscedastic_point_loss,
    invert_similarity,
    point_head_patch_features,
    robust_point_loss,
    validation_checkpoint_is_better,
)


def main() -> None:
    _test_cached_feature_layout()
    _test_similarity_round_trip()
    _test_validation_checkpoint_selection()
    _test_protected_fallback_and_gradient()
    print("Phase 1 temporal trust-aware residual smoke passed")


def _test_cached_feature_layout() -> None:
    levels = torch.randn(4, 3, 8, 16)
    features = point_head_patch_features(
        levels,
        patch_start_idx=2,
        patch_shape=(2, 3),
    )
    assert features.shape == (3, 4, 6, 16)
    torch.testing.assert_close(features[1, 2], levels[2, 1, 2:])


def _test_similarity_round_trip() -> None:
    angle = torch.tensor(0.3)
    rotation = torch.tensor(
        [
            [torch.cos(angle), -torch.sin(angle), 0.0],
            [torch.sin(angle), torch.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translation = torch.tensor([0.2, -0.4, 1.1])
    native = torch.randn(2, 4, 5, 3)
    metric = apply_similarity(
        native, scale=1.7, rotation=rotation, translation=translation
    )
    recovered = invert_similarity(
        metric, scale=1.7, rotation=rotation, translation=translation
    )
    torch.testing.assert_close(recovered, native, rtol=1e-5, atol=1e-5)


def _test_validation_checkpoint_selection() -> None:
    assert validation_checkpoint_is_better(0.12, 0.20, None)
    assert validation_checkpoint_is_better(0.11, 0.30, (0.12, 0.20))
    assert validation_checkpoint_is_better(0.12, 0.19, (0.12, 0.20))
    assert not validation_checkpoint_is_better(0.12, 0.21, (0.12, 0.20))
    assert not validation_checkpoint_is_better(0.13, 0.10, (0.12, 0.20))


def _test_protected_fallback_and_gradient() -> None:
    torch.manual_seed(9)
    features = torch.randn(2, 4, 6, 8)
    raw = torch.randn(2, 8, 10, 3)
    target = raw + 0.1 * torch.randn_like(raw)
    valid = torch.ones(2, 8, 10, dtype=torch.bool)
    for use_gate, use_uncertainty in ((False, False), (True, False), (True, True)):
        model = TrustAwareResidualHead(
            feature_channels=8,
            level_count=4,
            patch_shape=(2, 3),
            projection_channels=4,
            hidden_channels=8,
            use_gate=use_gate,
            use_uncertainty=use_uncertainty,
        )
        initial = model(features, output_size=(8, 10))
        assert float(initial.correction.abs().max()) == 0.0
        torch.testing.assert_close(raw + initial.correction, raw, rtol=0, atol=0)
        point = robust_point_loss(
            raw + initial.correction,
            target,
            valid,
            beta=0.05,
        )
        loss = point + 0.01 * initial.correction[valid].abs().mean()
        if initial.log_variance is not None:
            loss = loss + 0.1 * heteroscedastic_point_loss(
                raw + initial.correction,
                target,
                initial.log_variance,
                valid,
            )
        loss.backward()
        assert any(
            parameter.grad is not None and float(parameter.grad.abs().sum()) > 0.0
            for parameter in model.parameters()
        )
        assert bool(((initial.gate >= 0.0) & (initial.gate <= 1.0)).all())


if __name__ == "__main__":
    main()
