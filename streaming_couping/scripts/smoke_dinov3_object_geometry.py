#!/usr/bin/env python3
"""No-model smoke test for the object-conditioned residual head."""

from __future__ import annotations

import torch

from streaming_couping.src.dinov3_object_geometry import (
    ObjectConditionedResidualHead,
)


def main() -> None:
    features = torch.randn(2, 2, 6, 8)
    masks = torch.zeros(2, 2, 4, 6, dtype=torch.bool)
    masks[:, 0, :, :2] = True
    masks[:, 1, :, 4:] = True
    object_features = torch.randn(2, 2, 5)
    valid = torch.ones(2, 2, dtype=torch.bool)
    head = ObjectConditionedResidualHead(
        feature_channels=8,
        level_count=2,
        object_feature_channels=5,
        projection_channels=4,
        object_projection_channels=3,
        hidden_channels=6,
    )

    # Zero initialization must preserve the raw pointmap exactly.
    output = head(
        features,
        output_size=(4, 6),
        patch_shape=(2, 3),
        object_masks=masks,
        object_features=object_features,
        object_valid=valid,
    )
    assert tuple(output.correction.shape) == (2, 4, 6, 3)
    assert tuple(output.object_union.shape) == (2, 4, 6)
    assert bool(output.object_union.any()) and not bool(output.object_union.all())
    assert float(output.correction.abs().max()) == 0.0

    # A constant output must be clipped to the object support, never to the
    # background.  This tests the deployment-time safety invariant directly.
    with torch.no_grad():
        head.residual_head.bias.fill_(1.0)
    masked = head(
        features[:1],
        output_size=(4, 6),
        patch_shape=(2, 3),
        object_masks=masks[:1],
        object_features=object_features[:1],
        object_valid=valid[:1],
    )
    object_values = masked.correction[0][masked.object_union[0]]
    background_values = masked.correction[0][~masked.object_union[0]]
    assert float(object_values.abs().min()) > 0.99
    assert float(background_values.abs().max()) == 0.0

    # Geometry-only B uses the same head and the same mask write gate.
    geometry_only = head(
        features[:1],
        output_size=(4, 6),
        patch_shape=(2, 3),
        object_masks=masks[:1],
        object_features=None,
    )
    assert tuple(geometry_only.correction.shape) == (1, 4, 6, 3)
    print(
        "DINOv3 object-conditioned geometry smoke passed "
        f"features={tuple(features.shape)} output={tuple(output.correction.shape)}"
    )


if __name__ == "__main__":
    main()
