#!/usr/bin/env python3
"""Dependency-light V9.6 actual-grid and output-schema smoke."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from streaming_couping.scripts.run_v96_dense_grid_upper_bound import (
    FRAME_COLUMNS,
    SUMMARY_COLUMNS,
    _annotate_passes,
    _decision_markdown,
    _validate_outputs,
    _write_csv,
    load_dense_grid_config,
)
from streaming_couping.src.v74_temporal_protocol import FOLDS
from streaming_couping.src.v90_epipolar_geometry import SurfaceCorrespondences
from streaming_couping.src.v96_dense_grid_decoder import (
    GRID_DECODERS,
    GRID_SUPPORT_SCOPES,
    decode_history_grid,
    dense_grid_normalized,
)


def main() -> None:
    config = load_dense_grid_config(
        "streaming_couping/configs/v96_dense_grid_upper_bound.yaml"
    )
    _grid_smoke(config)
    _schema_smoke(config)
    print("V9.6 dense-grid upper-bound smoke passed")


def _grid_smoke(config) -> None:
    normalized = dense_grid_normalized(config.grid_size)
    _require(normalized.shape == (72 * 72, 2), "grid shape")
    current = torch.tensor(
        [[10.0, 10.0], [30.0, 20.0], [55.0, 42.0], [80.0, 60.0]],
        dtype=torch.float64,
    )
    row = SurfaceCorrespondences(
        current_frame=3,
        history_frame=1,
        slot=-1,
        current_uv=current,
        history_uv=current + torch.tensor([1.3, -0.7]),
        weights=torch.ones(4, dtype=torch.float64),
        depth_residual_metric=torch.zeros(4, dtype=torch.float64),
        sampled_queries=4,
        projected_in_bounds=4,
        visible_queries=4,
    )
    soft = decode_history_grid(
        row,
        mode="soft_bilinear_k4",
        grid_size=config.grid_size,
        image_size=(72, 96),
    )
    _require(float(soft.errors_pixels.max()) < 1e-10, "soft K4 exact")
    _require(soft.keys_per_query == 4, "four actual keys")


def _schema_smoke(config) -> None:
    summaries = []
    frames = []
    for scope in GRID_SUPPORT_SCOPES:
        for decoder in GRID_DECODERS:
            for fold in FOLDS:
                summary = dict.fromkeys(SUMMARY_COLUMNS, 0)
                summary.update(
                    {
                        "fold": fold.name,
                        "support_scope": scope,
                        "decoder": decoder,
                        "frames": len(fold.test_frames),
                        "active_frames": len(fold.test_frames),
                        "equal_count_exact": 1,
                        "raw_rotation_error_deg": 2.0,
                        "refined_rotation_error_deg": 1.0,
                        "raw_translation_direction_error_deg": 2.0,
                        "refined_translation_direction_error_deg": 1.0,
                    }
                )
                summaries.append(summary)
                for frame_index in fold.test_frames:
                    frame = dict.fromkeys(FRAME_COLUMNS, 0)
                    frame.update(
                        {
                            "fold": fold.name,
                            "support_scope": scope,
                            "decoder": decoder,
                            "frame_index": frame_index,
                        }
                    )
                    frames.append(frame)
    _annotate_passes(summaries)
    _validate_outputs(summaries, frames, config=config)
    decision = _decision_markdown(summaries)
    _require("dense-descriptor stage allowed: `1`" in decision, "gate")
    with TemporaryDirectory() as directory:
        root = Path(directory)
        _write_csv(root / "summary.csv", summaries, SUMMARY_COLUMNS)
        _write_csv(root / "frames.csv", frames, FRAME_COLUMNS)
        _require((root / "summary.csv").stat().st_size > 0, "summary CSV")


def _require(condition: bool, label: str) -> None:
    if not condition:
        raise RuntimeError(f"V9.6 smoke failed: {label}")


if __name__ == "__main__":
    main()
