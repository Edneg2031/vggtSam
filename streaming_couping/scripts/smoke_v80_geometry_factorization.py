#!/usr/bin/env python3
"""Dependency-light deterministic checks for the V8 O2.6 experiment."""

from __future__ import annotations

import torch

from streaming_couping.scripts.run_v80_geometry_factorization import (
    CalibrationConfig,
    _apply_calibration,
    _calibrate_depth,
    _compact_decision,
    _decision_markdown,
)
from streaming_couping.src.solvers.weighted_kabsch import KabschConfig
from streaming_couping.src.solvers.weighted_umeyama import weighted_umeyama
from streaming_couping.src.v80_pose_geometry import backproject_depth_at_local_tokens


def main() -> None:
    _check_umeyama()
    _check_depth_calibration()
    _check_backprojection()
    _check_decision_reduction()
    print("V8 O2.6 geometry-factorization smoke passed")


def _check_umeyama() -> None:
    generator = torch.Generator().manual_seed(31)
    source = torch.randn(48, 3, generator=generator, dtype=torch.double)
    angle = torch.tensor(0.41, dtype=torch.double)
    rotation = torch.tensor(
        [
            [torch.cos(angle), 0.0, torch.sin(angle)],
            [0.0, 1.0, 0.0],
            [-torch.sin(angle), 0.0, torch.cos(angle)],
        ],
        dtype=torch.double,
    )
    expected_scale = 1.35
    translation = torch.tensor([0.3, -0.2, 0.7], dtype=torch.double)
    target = expected_scale * (source @ rotation.T) + translation
    result = weighted_umeyama(
        source,
        target,
        config=KabschConfig(min_points=6),
    )
    assert bool(result.accepted)
    assert abs(float(result.scale) - expected_scale) < 1e-9
    assert torch.allclose(result.rotation, rotation, atol=1e-9)
    assert torch.allclose(result.translation, translation, atol=1e-9)
    assert torch.allclose(
        result.transform[3], torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.double)
    )


def _check_depth_calibration() -> None:
    base = torch.linspace(0.5, 3.0, 400).reshape(1, 20, 20, 1)
    target = 1.2 * base + 0.15
    config = CalibrationConfig(
        trim_quantile=0.90,
        trim_iterations=3,
        min_pixels=32,
        max_pixels=4096,
    )
    scale = _calibrate_depth(
        base,
        1.2 * base,
        mode="scale",
        native_to_metric=2.0,
        config=config,
    )
    assert abs(float(scale.scale[0]) - 1.2) < 1e-6
    assert float(scale.after_rmse_metric[0]) < 1e-5
    affine = _calibrate_depth(
        base,
        target,
        mode="affine",
        native_to_metric=2.0,
        config=config,
    )
    assert abs(float(affine.scale[0]) - 1.2) < 1e-5
    assert abs(float(affine.offset[0]) - 0.15) < 1e-5
    assert torch.allclose(_apply_calibration(base, affine), target, atol=1e-5)


def _check_backprojection() -> None:
    depth = torch.full((1, 3, 4, 1), 2.0)
    intrinsics = torch.tensor([[[2.0, 0.0, 1.5], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]]])
    # Normalized UV=(1,1) maps to pixel (W-1,H-1)=(3,2).
    local = torch.zeros(1, 1, 1, 5)
    local[..., 3:5] = 1.0
    points, valid = backproject_depth_at_local_tokens(
        depth,
        intrinsics,
        local_features=local,
        local_valid=torch.ones(1, 1, 1, dtype=torch.bool),
    )
    assert bool(valid[0, 0, 0])
    assert torch.allclose(points[0, 0, 0], torch.tensor([1.5, 1.0, 2.0]))


def _check_decision_reduction() -> None:
    rows = []
    for solver in ("se3_weighted_kabsch", "se3_trimmed_kabsch"):
        for fold in ("short", "medium", "long"):
            rows.append(
                {
                    "branch": "o1_direct_gt_camera",
                    "oracle_level": "O1",
                    "depth_source": "direct_gt_camera",
                    "intrinsics_source": "none",
                    "depth_calibration": "none",
                    "transform_family": "SE3",
                    "solver": solver,
                    "fold": fold,
                    "fold_robust_pass": 1,
                    "reliable_gain_percent": 50.0,
                    "reliable_active_frames": 4,
                    "reliable_worse_frames": 0,
                    "mean_fit_scale": 1.0,
                    "mean_fit_rmse_metric": 0.01,
                }
            )
    compact = _compact_decision(rows)
    assert len(compact) == 1
    assert compact[0]["selected_solver"] == "se3_weighted_kabsch"
    assert compact[0]["all_fold_pass"] == 1
    decision = _decision_markdown(rows)
    assert "direct GT-camera O1 pass: `1`" in decision


if __name__ == "__main__":
    main()
