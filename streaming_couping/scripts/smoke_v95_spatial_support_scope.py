#!/usr/bin/env python3
"""Dependency-light V9.5 equal-count support and schema smoke."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from streaming_couping.scripts.run_v95_spatial_support_scope import (
    FRAME_COLUMNS,
    SUMMARY_COLUMNS,
    _annotate_passes,
    _decision_markdown,
    _validate_outputs,
    _write_csv,
    load_spatial_scope_config,
)
from streaming_couping.src.v74_temporal_protocol import FOLDS
from streaming_couping.src.v90_epipolar_geometry import SurfaceCorrespondences
from streaming_couping.src.v95_spatial_support import (
    SUPPORT_SCOPES,
    build_equal_count_supports,
)


def main() -> None:
    config = load_spatial_scope_config(
        "streaming_couping/configs/v95_spatial_support_scope.yaml"
    )
    _support_smoke()
    _schema_smoke(config)
    print("V9.5 spatial support smoke passed")


def _surface(uv: torch.Tensor) -> SurfaceCorrespondences:
    uv = uv.double()
    return SurfaceCorrespondences(
        current_frame=4,
        history_frame=2,
        slot=-1,
        current_uv=uv,
        history_uv=uv + 1.0,
        weights=torch.ones(len(uv), dtype=torch.float64),
        depth_residual_metric=torch.zeros(len(uv), dtype=torch.float64),
        sampled_queries=len(uv),
        projected_in_bounds=len(uv),
        visible_queries=len(uv),
    )


def _support_smoke() -> None:
    instance = _surface(
        torch.tensor([[30.0 + x, 30.0 + y] for y in range(3) for x in range(4)])
    )
    full = _surface(
        torch.tensor([[5.0 + 10 * x, 5.0 + 10 * y] for y in range(6) for x in range(8)])
    )
    background = _surface(
        torch.tensor([[5.0 + 10 * x, 5.0 + 10 * y] for y in range(2) for x in range(8)])
    )
    mask = torch.zeros((80, 100), dtype=torch.bool)
    mask[25:45, 25:45] = True
    supports = build_equal_count_supports(
        instance=instance,
        full_image_candidates=full,
        background_candidates=background,
        instance_union_mask=mask,
        image_size=(80, 100),
    )
    _require(tuple(supports) == SUPPORT_SCOPES, "scope order")
    _require(all(value.count == instance.count for value in supports.values()), "equal count")
    hybrid = supports["instance_background_balanced"]
    _require(hybrid.instance_rows == hybrid.background_rows == 6, "balanced rows")


def _schema_smoke(config) -> None:
    summaries = []
    frames = []
    for scope in SUPPORT_SCOPES:
        for sigma in config.noise_sigmas_pixels:
            replicates = 1 if sigma == 0.0 else config.noise_replicates
            for fold in FOLDS:
                evaluation_rows = replicates * len(fold.test_frames)
                summary = dict.fromkeys(SUMMARY_COLUMNS, 0)
                summary.update(
                    {
                        "fold": fold.name,
                        "support_scope": scope,
                        "noise_sigma_pixels": sigma,
                        "replicates": replicates,
                        "evaluation_rows": evaluation_rows,
                        "active_rows": evaluation_rows,
                        "equal_count_exact": 1,
                        "raw_rotation_error_deg": 2.0,
                        "refined_rotation_error_deg": 1.0,
                        "raw_translation_direction_error_deg": 2.0,
                        "refined_translation_direction_error_deg": 1.0,
                    }
                )
                summaries.append(summary)
                for replicate in range(replicates):
                    for frame_index in fold.test_frames:
                        frame = dict.fromkeys(FRAME_COLUMNS, 0)
                        frame.update(
                            {
                                "fold": fold.name,
                                "support_scope": scope,
                                "noise_sigma_pixels": sigma,
                                "replicate": replicate,
                                "frame_index": frame_index,
                            }
                        )
                        frames.append(frame)
    _annotate_passes(summaries)
    _validate_outputs(summaries, frames, config=config)
    decision = _decision_markdown(summaries)
    _require("expanded spatial support noise-feasible: `1`" in decision, "decision")
    with TemporaryDirectory() as directory:
        root = Path(directory)
        _write_csv(root / "summary.csv", summaries, SUMMARY_COLUMNS)
        _write_csv(root / "frames.csv", frames, FRAME_COLUMNS)
        _require((root / "summary.csv").stat().st_size > 0, "summary CSV")


def _require(condition: bool, label: str) -> None:
    if not condition:
        raise RuntimeError(f"V9.5 smoke failed: {label}")


if __name__ == "__main__":
    main()
