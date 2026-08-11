#!/usr/bin/env python3
"""Dependency-light V9.3 convex/noise/filter and schema smoke."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from streaming_couping.scripts.run_v93_quantization_tolerance import (
    FRAME_COLUMNS,
    SUMMARY_COLUMNS,
    _annotate_passes,
    _branch_specs,
    _decision_markdown,
    _validate_outputs,
    _write_csv,
    load_diagnostic_config,
)
from streaming_couping.src.v74_temporal_protocol import FOLDS
from streaming_couping.src.v90_epipolar_geometry import LocalTokenReprojection
from streaming_couping.src.v93_quantization_tolerance import (
    continuous_prediction,
    filter_prediction_by_oracle_error,
    hard_nearest_prediction,
    noisy_continuous_prediction,
    soft_knn_convex_prediction,
)


def main() -> None:
    _prediction_smoke()
    config = load_diagnostic_config(
        "streaming_couping/configs/v93_quantization_tolerance.yaml"
    )
    _schema_smoke(config)
    print("V9.3 quantization tolerance smoke passed")


def _prediction_smoke() -> None:
    image_size = (64, 128)
    target = torch.tensor([[5.0, 5.0], [2.0, 3.0]], dtype=torch.float64)
    valid = torch.ones(2, dtype=torch.bool)
    labels = LocalTokenReprojection(
        current_frame=2,
        history_frame=1,
        slot=0,
        current_uv=target.clone(),
        history_target_uv=target.clone(),
        query_valid=valid,
        target_visible=valid.clone(),
        weights=torch.ones(2, dtype=torch.float64),
        depth_residual_metric=torch.zeros(2, dtype=torch.float64),
    )
    pixels = torch.tensor(
        [[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]]
    )
    keys = torch.stack(
        [pixels[:, 0] / 127.0 * 2.0 - 1.0, pixels[:, 1] / 63.0 * 2.0 - 1.0],
        dim=-1,
    )
    history = {
        "history_uv_normalized": keys,
        "history_valid": torch.ones(4, dtype=torch.bool),
        "image_size": image_size,
    }
    exact = continuous_prediction(labels, image_size=image_size)
    hard = hard_nearest_prediction(labels, **history)
    soft = soft_knn_convex_prediction(labels, neighbors=4, **history)
    filtered = filter_prediction_by_oracle_error(
        hard, labels, max_error_pixels=4.0, image_size=image_size
    )
    noise_a = noisy_continuous_prediction(
        labels, sigma_pixels=1.0, seed=93, image_size=image_size
    )
    noise_b = noisy_continuous_prediction(
        labels, sigma_pixels=1.0, seed=93, image_size=image_size
    )
    _require(exact.diagnostics.selected_epe_sum_pixels == 0.0, "continuous")
    _require(hard.diagnostics.selected_epe_sum_pixels > 0.0, "hard quantization")
    _require(soft.diagnostics.selected_epe_sum_pixels < 1e-8, "soft convex")
    _require(
        filtered.diagnostics.accepted_correspondences
        <= hard.diagnostics.accepted_correspondences,
        "oracle filter",
    )
    _require(
        torch.equal(noise_a.correspondences.history_uv, noise_b.correspondences.history_uv),
        "noise determinism",
    )


def _schema_smoke(config) -> None:
    branches = _branch_specs(config)
    summaries = []
    frames = []
    for branch in branches:
        for fold in FOLDS:
            summary = dict.fromkeys(SUMMARY_COLUMNS, 0)
            summary.update(
                {
                    "fold": fold.name,
                    "branch": branch.name,
                    "family": branch.family,
                    "parameter": branch.parameter,
                    "replicates": branch.replicates,
                    "fixed_edge_frames": len(fold.test_frames),
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
                            "branch": branch.name,
                            "family": branch.family,
                            "parameter": branch.parameter,
                            "replicate": replicate,
                            "frame_index": frame_index,
                        }
                    )
                    frames.append(frame)
    _annotate_passes(summaries)
    _validate_outputs(summaries, frames, branches=branches)
    decision = _decision_markdown(summaries)
    _require("continuous positive-control all-fold pass: `1`" in decision, "decision")
    with TemporaryDirectory() as directory:
        root = Path(directory)
        _write_csv(root / "summary.csv", summaries, SUMMARY_COLUMNS)
        _write_csv(root / "frames.csv", frames, FRAME_COLUMNS)
        _require((root / "summary.csv").stat().st_size > 0, "summary CSV")
        _require((root / "frames.csv").stat().st_size > 0, "frame CSV")


def _require(condition: bool, label: str) -> None:
    if not condition:
        raise RuntimeError(f"V9.3 smoke failed: {label}")


if __name__ == "__main__":
    main()
