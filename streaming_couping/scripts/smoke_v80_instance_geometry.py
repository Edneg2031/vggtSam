#!/usr/bin/env python3
"""Deterministic tensor checks for V8 O2.7."""

from __future__ import annotations

import torch

from streaming_couping.scripts.run_v80_geometry_factorization import (
    CalibrationConfig,
)
from streaming_couping.scripts.run_v80_instance_geometry import (
    _backproject_instance_affine,
    _calibrate_instance_affine,
    _intrinsics_modes,
)


def main() -> None:
    _check_intrinsics_modes()
    _check_instance_affine()
    print("V8 O2.7 instance-geometry smoke passed")


def _check_intrinsics_modes() -> None:
    predicted = torch.eye(3, dtype=torch.double).repeat(3, 1, 1)
    predicted[:, 0, 0] = torch.tensor([100.0, 120.0, 80.0])
    predicted[:, 1, 1] = torch.tensor([110.0, 130.0, 90.0])
    gt = torch.eye(3, dtype=torch.double).repeat(3, 1, 1)
    modes = _intrinsics_modes(predicted, gt)
    assert torch.equal(modes["current_predicted"], predicted)
    assert torch.equal(
        modes["reference_predicted"],
        predicted[0:1].expand_as(predicted),
    )
    assert float(modes["causal_median_predicted"][2, 0, 0]) == 100.0
    assert torch.equal(modes["gt"], gt)


def _check_instance_affine() -> None:
    height, width = 12, 14
    predicted = torch.linspace(0.5, 2.5, height * width).reshape(1, height, width, 1)
    masks = torch.zeros(1, 2, height, width, dtype=torch.bool)
    masks[0, 0, :, : width // 2] = True
    masks[0, 1, :, width // 2 :] = True
    target = predicted.clone()
    target[0, :, : width // 2] = 0.8 * predicted[0, :, : width // 2] + 0.1
    target[0, :, width // 2 :] = 1.2 * predicted[0, :, width // 2 :] - 0.05
    fit = _calibrate_instance_affine(
        predicted,
        target,
        masks,
        native_to_metric=1.0,
        config=CalibrationConfig(
            trim_quantile=1.0,
            trim_iterations=1,
            min_pixels=32,
            max_pixels=1024,
        ),
    )
    assert fit.valid.all()
    assert torch.allclose(
        fit.scale[0], torch.tensor([0.8, 1.2], dtype=torch.double), atol=1e-6
    )
    assert torch.allclose(
        fit.offset[0], torch.tensor([0.1, -0.05], dtype=torch.double), atol=1e-6
    )
    assert float(fit.after_rmse_metric.max()) < 1e-6

    local = torch.zeros(1, 2, 2, 5)
    local[:, 0, :, 3] = -0.75
    local[:, 1, :, 3] = 0.75
    local[..., 4] = 0.0
    intrinsics = torch.tensor(
        [[[10.0, 0.0, 6.5], [0.0, 10.0, 5.5], [0.0, 0.0, 1.0]]],
        dtype=torch.double,
    )
    camera, valid = _backproject_instance_affine(
        predicted,
        intrinsics,
        fit=fit,
        local_features=local,
        local_valid=torch.ones(1, 2, 2, dtype=torch.bool),
    )
    assert camera.shape == (1, 2, 2, 3)
    assert valid.all()
    assert torch.isfinite(camera).all()


if __name__ == "__main__":
    main()
