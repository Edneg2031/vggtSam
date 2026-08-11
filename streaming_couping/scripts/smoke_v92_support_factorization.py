#!/usr/bin/env python3
"""Dependency-light V9.2 support matching and output-schema smoke."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from streaming_couping.scripts.run_v92_support_factorization import (
    FRAME_COLUMNS,
    SUMMARY_COLUMNS,
    _annotate_passes,
    _decision_markdown,
    _validate_outputs,
    _write_csv,
    load_support_config,
)
from streaming_couping.src.learned_pose.observations import _farthest_uv_indices
from streaming_couping.src.v74_temporal_protocol import FOLDS
from streaming_couping.src.v90_epipolar_geometry import LocalTokenReprojection
from streaming_couping.src.v92_support_factorization import (
    MATCH_STRATEGIES,
    match_discrete_support,
)


def main() -> None:
    _matching_smoke()
    _prefix_smoke()
    _schema_smoke()
    config = load_support_config(
        "streaming_couping/configs/v92_support_factorization.yaml"
    )
    _require(config.query_token_count == 32, "current query count")
    _require(config.history_key_counts == (32, 64, 128, 256), "key counts")
    _require(config.strategies == MATCH_STRATEGIES, "strategy order")
    print("V9.2 support factorization smoke passed")


def _matching_smoke() -> None:
    image_size = (64, 128)
    target = torch.tensor(
        [[10.0, 20.0], [11.0, 20.0], [40.0, 20.0]], dtype=torch.float64
    )
    key_pixels = torch.tensor([[10.0, 20.0], [40.0, 20.0]])
    keys = torch.stack(
        [
            key_pixels[:, 0] / 127.0 * 2.0 - 1.0,
            key_pixels[:, 1] / 63.0 * 2.0 - 1.0,
        ],
        dim=-1,
    )
    valid = torch.ones(3, dtype=torch.bool)
    labels = LocalTokenReprojection(
        current_frame=2,
        history_frame=1,
        slot=0,
        current_uv=target.clone(),
        history_target_uv=target.clone(),
        query_valid=valid,
        target_visible=valid.clone(),
        weights=torch.ones(3, dtype=torch.float64),
        depth_residual_metric=torch.zeros(3, dtype=torch.float64),
    )
    common = {
        "history_uv_normalized": keys,
        "history_valid": torch.ones(2, dtype=torch.bool),
        "image_size": image_size,
    }
    nearest = match_discrete_support(labels, strategy="nearest", **common)
    mutual = match_discrete_support(labels, strategy="mutual", **common)
    greedy = match_discrete_support(labels, strategy="greedy_unique", **common)
    _require(nearest.correspondences.count == 3, "nearest accepts support")
    _require(nearest.diagnostics.nearest_collisions == 1, "collision count")
    _require(mutual.correspondences.count == 2, "mutual uniqueness")
    _require(greedy.correspondences.count == 2, "greedy uniqueness")
    _require(
        nearest.diagnostics.as_pose_match_stats()["metrics_precomputed"] == 1.0,
        "pose reporter receives precomputed matching metrics",
    )


def _prefix_smoke() -> None:
    uv = torch.rand((300, 2), generator=torch.Generator().manual_seed(92))
    short = _farthest_uv_indices(uv, count=32)
    long = _farthest_uv_indices(uv, count=256)
    _require(torch.equal(short, long[:32]), "farthest-UV prefix nesting")


def _schema_smoke() -> None:
    summaries = []
    frames = []
    for key_count in (32, 64, 128, 256):
        for strategy in MATCH_STRATEGIES:
            for fold in FOLDS:
                row = dict.fromkeys(SUMMARY_COLUMNS, 0)
                row.update(
                    {
                        "fold": fold.name,
                        "history_key_count": key_count,
                        "strategy": strategy,
                        "frames": len(fold.test_frames),
                        "active_frames": len(fold.test_frames),
                        "raw_rotation_error_deg": 2.0,
                        "refined_rotation_error_deg": 1.0,
                        "raw_translation_direction_error_deg": 2.0,
                        "refined_translation_direction_error_deg": 1.0,
                    }
                )
                summaries.append(row)
                for frame_index in fold.test_frames:
                    frame = dict.fromkeys(FRAME_COLUMNS, 0)
                    frame.update(
                        {
                            "fold": fold.name,
                            "history_key_count": key_count,
                            "strategy": strategy,
                            "frame_index": frame_index,
                        }
                    )
                    frames.append(frame)
    _annotate_passes(summaries)
    _validate_outputs(summaries, frames)
    decision = _decision_markdown(summaries)
    _require("all-fold passing configurations: `12`" in decision, "decision")
    with TemporaryDirectory() as directory:
        root = Path(directory)
        _write_csv(root / "summary.csv", summaries, SUMMARY_COLUMNS)
        _write_csv(root / "frames.csv", frames, FRAME_COLUMNS)
        _require((root / "summary.csv").stat().st_size > 0, "summary CSV")
        _require((root / "frames.csv").stat().st_size > 0, "frame CSV")


def _require(condition: bool, label: str) -> None:
    if not condition:
        raise RuntimeError(f"V9.2 smoke failed: {label}")


if __name__ == "__main__":
    main()
