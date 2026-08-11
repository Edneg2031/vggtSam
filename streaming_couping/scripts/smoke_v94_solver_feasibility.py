#!/usr/bin/env python3
"""Dependency-light V9.4 robust-solver and output-schema smoke."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from streaming_couping.scripts.run_v94_solver_feasibility import (
    FRAME_COLUMNS,
    SUMMARY_COLUMNS,
    _annotate_passes,
    _decision_markdown,
    _evidence_branches,
    _validate_outputs,
    _write_csv,
    load_solver_config,
)
from streaming_couping.scripts.run_v93_quantization_tolerance import load_diagnostic_config
from streaming_couping.src.v74_temporal_protocol import FOLDS
from streaming_couping.src.v90_epipolar_geometry import EpipolarConfig
from streaming_couping.src.v94_robust_epipolar import (
    ROBUST_SOLVERS,
    estimate_robust_relative_pose,
)


def main() -> None:
    config = load_solver_config(
        "streaming_couping/configs/v94_solver_feasibility.yaml"
    )
    _solver_smoke(config)
    _schema_smoke(config)
    print("V9.4 solver feasibility smoke passed")


def _solver_smoke(config) -> None:
    generator = torch.Generator().manual_seed(94)
    points = torch.randn((48, 3), generator=generator, dtype=torch.float64)
    points[:, :2] *= 0.4
    points[:, 2] = torch.linspace(3.0, 6.0, 48)
    k = torch.tensor(
        [[220.0, 0.0, 96.0], [0.0, 215.0, 72.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    angle = 0.06
    rotation = torch.tensor(
        [
            [torch.cos(torch.tensor(angle)), 0.0, torch.sin(torch.tensor(angle))],
            [0.0, 1.0, 0.0],
            [-torch.sin(torch.tensor(angle)), 0.0, torch.cos(torch.tensor(angle))],
        ],
        dtype=torch.float64,
    )
    translation = torch.tensor([0.3, 0.03, 0.01], dtype=torch.float64)
    history_points = (rotation @ points.T).T + translation
    first = (k @ points.T).T
    second = (k @ history_points.T).T
    first = first[:, :2] / first[:, 2:3]
    second = second[:, :2] / second[:, 2:3]
    l0 = torch.eye(4, dtype=torch.float64)
    l0[:3, :3] = rotation
    l0[:3, 3] = translation
    for solver in ROBUST_SOLVERS:
        result = estimate_robust_relative_pose(
            first,
            second,
            torch.ones(48, dtype=torch.float64),
            k,
            k,
            l0,
            solver=solver,
            epipolar_config=EpipolarConfig(),
            robust_config=config.robust,
            image_size=(144, 192),
        )
        _require(result.estimate.success, f"exact solver {solver}")


def _schema_smoke(config) -> None:
    v93 = load_diagnostic_config(config.v93_config)
    branches = _evidence_branches(v93)
    summaries = []
    frames = []
    for branch in branches:
        for solver in ROBUST_SOLVERS:
            for fold in FOLDS:
                summary = dict.fromkeys(SUMMARY_COLUMNS, 0)
                summary.update(
                    {
                        "fold": fold.name,
                        "evidence_branch": branch.name,
                        "evidence_family": branch.family,
                        "evidence_parameter": branch.parameter,
                        "solver": solver,
                        "replicates": branch.replicates,
                        "evaluation_rows": len(fold.test_frames) * branch.replicates,
                        "active_rows": len(fold.test_frames) * branch.replicates,
                        "raw_rotation_error_deg": 2.0,
                        "refined_rotation_error_deg": 1.0,
                        "raw_translation_direction_error_deg": 2.0,
                        "refined_translation_direction_error_deg": 1.0,
                    }
                )
                summaries.append(summary)
                for replicate in range(branch.replicates):
                    for frame_index in fold.test_frames:
                        frame = dict.fromkeys(FRAME_COLUMNS, 0)
                        frame.update(
                            {
                                "fold": fold.name,
                                "evidence_branch": branch.name,
                                "solver": solver,
                                "replicate": replicate,
                                "frame_index": frame_index,
                            }
                        )
                        frames.append(frame)
    _annotate_passes(summaries)
    _validate_outputs(summaries, frames, branches=branches)
    decision = _decision_markdown(summaries)
    _require("instance-essential route solver-feasible: `1`" in decision, "decision")
    with TemporaryDirectory() as directory:
        root = Path(directory)
        _write_csv(root / "summary.csv", summaries, SUMMARY_COLUMNS)
        _write_csv(root / "frames.csv", frames, FRAME_COLUMNS)
        _require((root / "summary.csv").stat().st_size > 0, "summary CSV")


def _require(condition: bool, label: str) -> None:
    if not condition:
        raise RuntimeError(f"V9.4 smoke failed: {label}")


if __name__ == "__main__":
    main()
