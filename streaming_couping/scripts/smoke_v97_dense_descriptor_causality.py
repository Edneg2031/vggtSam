#!/usr/bin/env python3
"""Dependency-light V9.7 matcher/gradient/decision smoke."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from streaming_couping.scripts.run_v97_dense_descriptor_causality import (
    FRAME_COLUMNS,
    SUMMARY_COLUMNS,
    _annotate_decisions,
    _decision_markdown,
    _validate_outputs,
    _write_csv,
    load_v97_config,
)
from streaming_couping.src.v74_temporal_protocol import FOLDS
from streaming_couping.src.v97_dense_descriptor_matcher import (
    DENSE_ARCHITECTURES,
    SAM_EVAL_VARIANTS,
    DenseMatcherConfig,
    DenseSubgridMatcher,
    build_dense_match_target,
    decode_dense_matches,
    dense_match_loss,
)


def main() -> None:
    config = load_v97_config(
        "streaming_couping/configs/v97_dense_descriptor_causality.yaml"
    )
    _matcher_smoke()
    _schema_smoke(config)
    print("V9.7 dense descriptor causality smoke passed")


def _matcher_smoke() -> None:
    config = DenseMatcherConfig(
        canonical_dim=8,
        projection_dim=4,
        offset_hidden_dim=6,
        temperature=0.2,
    )
    model = DenseSubgridMatcher(config)
    query = torch.randn(1, 4, 8)
    key = torch.randn(1, 9, 8)
    valid = torch.ones(1, 4, dtype=torch.bool)
    target = build_dense_match_target(
        torch.tensor([[[1.0, 1.0], [4.0, 4.0], [7.0, 7.0], [3.0, 6.0]]]),
        query_valid=valid,
        visible=valid,
        grid_size=(3, 3),
        image_size=(9, 9),
    )
    output = model(query, key, valid, torch.ones(1, 9, dtype=torch.bool))
    result = dense_match_loss(model, output, target)
    result.loss.backward()
    _require(model.query_projection.weight.grad is not None, "query gradient")
    _require(model.key_projection.weight.grad is not None, "key gradient")
    decoded = decode_dense_matches(
        model,
        output,
        grid_size=(3, 3),
        image_size=(9, 9),
        query_valid=valid,
    )
    _require(decoded.history_uv_pixels.shape == (1, 4, 2), "decode shape")


def _schema_smoke(config) -> None:
    summaries = []
    frames = []
    for fold in FOLDS:
        methods = [
            ("sam_dense", value) for value in SAM_EVAL_VARIANTS
        ] + [
            (architecture, "normal")
            for architecture in DENSE_ARCHITECTURES
            if architecture != "sam_dense"
        ]
        for architecture, variant in methods:
            row = dict.fromkeys(SUMMARY_COLUMNS, 0)
            candidate = architecture == "sam_dense" and variant == "normal"
            perturbation = architecture == "sam_dense" and variant != "normal"
            row.update(
                {
                    "fold": fold.name,
                    "architecture": architecture,
                    "variant": variant,
                    "parameters": 100,
                    "frames": len(fold.test_frames),
                    "active_frames": len(fold.test_frames),
                    "control_support_exact": 1,
                    "matcher_frozen_exact": 1,
                    "perturbed_pairs": len(fold.test_frames) if perturbation else 0,
                    "pck_accuracy": 0.9 if candidate else (0.2 if perturbation else 0.4),
                    "mean_epe_pixels": 0.2 if candidate else (4.0 if perturbation else 2.0),
                    "raw_rotation_error_deg": 3.0,
                    "refined_rotation_error_deg": 0.5 if candidate else 2.0,
                    "raw_translation_direction_error_deg": 5.0,
                    "refined_translation_direction_error_deg": 0.7 if candidate else 3.0,
                    "refined_relative_aggregate_deg": 1.2 if candidate else 5.0,
                }
            )
            summaries.append(row)
            for frame_index in fold.test_frames:
                frame = dict.fromkeys(FRAME_COLUMNS, 0)
                frame.update(
                    {
                        "fold": fold.name,
                        "architecture": architecture,
                        "variant": variant,
                        "frame_index": frame_index,
                    }
                )
                frames.append(frame)
    _annotate_decisions(summaries)
    _validate_outputs(summaries, frames)
    decision = _decision_markdown(summaries)
    _require("all-fold causal pass: `1`" in decision, "causal gate")
    with TemporaryDirectory() as directory:
        root = Path(directory)
        _write_csv(root / "summary.csv", summaries, SUMMARY_COLUMNS)
        _write_csv(root / "frames.csv", frames, FRAME_COLUMNS)
        _require((root / "summary.csv").stat().st_size > 0, "summary CSV")


def _require(condition: bool, label: str) -> None:
    if not condition:
        raise RuntimeError(f"V9.7 smoke failed: {label}")


if __name__ == "__main__":
    main()

