#!/usr/bin/env python3
"""Dependency-light V9.8 memory capture, matcher and decision smoke."""

from __future__ import annotations

import torch

from streaming_couping.scripts.run_v98_temporal_memory_causality import (
    AUDIT_COLUMNS,
    FRAME_COLUMNS,
    SUMMARY_COLUMNS,
    _annotate_decisions,
    _decision_markdown,
    _validate_outputs,
    load_v98_config,
)
from streaming_couping.src.v74_temporal_protocol import FOLDS
from streaming_couping.src.v97_dense_descriptor_matcher import (
    DenseMatcherConfig,
    DenseSubgridMatcher,
    build_dense_match_target,
    dense_match_loss,
)
from streaming_couping.src.v98_temporal_memory import (
    SAM31MemoryReadCapture,
    TEMPORAL_ARCHITECTURES,
    TEMPORAL_VARIANTS,
    combine_prompt_feature_runs,
    resolve_sam31_memory_model,
)


class _DummyTracker:
    def _prepare_memory_conditioned_features(self, **kwargs):
        source = kwargs["current_vision_feats"][-1]
        height, width = kwargs["feat_sizes"][-1]
        raw = source.permute(1, 2, 0).reshape(1, source.shape[-1], height, width)
        return raw + 0.25


class _Proxy:
    def __init__(self, model):
        self.model = model


class _Orchestrator:
    def __init__(self, model):
        self.tracker = _Proxy(model)


def main() -> None:
    load_v98_config("streaming_couping/configs/v98_temporal_memory_causality.yaml")
    _capture_smoke()
    _matcher_smoke()
    _decision_smoke()
    print("V9.8 temporal memory causality smoke passed")


def _capture_smoke() -> None:
    tracker = _DummyTracker()
    _require(resolve_sam31_memory_model(_Orchestrator(tracker)) is tracker, "tracker resolution")
    source = torch.arange(4 * 6, dtype=torch.float32).reshape(4, 1, 6)
    with SAM31MemoryReadCapture(
        tracker, grid_size=(3, 3), canonical_dim=8
    ) as capture:
        tracker._prepare_memory_conditioned_features(
            frame_idx=1,
            current_vision_feats=[source],
            feat_sizes=[(2, 2)],
            output_dict={
                "cond_frame_outputs": {0: {"maskmem_features": torch.ones(1)}},
                "non_cond_frame_outputs": {},
            },
        )
    result = capture.finalized(3)
    _require(result["valid"].tolist() == [False, True, False], "capture validity")
    _require(result["memory"].shape == (3, 9, 8), "capture shape")
    _require(float(result["memory_delta_l2"][1]) > 0.0, "memory delta")
    memory, raw, valid = combine_prompt_feature_runs(
        torch.stack([result["memory"], result["memory"] + 2.0]),
        torch.stack([result["raw"], result["raw"] + 2.0]),
        torch.stack([result["valid"], result["valid"]]),
    )
    _require(valid[1] and torch.allclose(memory[1], result["memory"][1] + 1.0), "prompt mean")
    _require(torch.allclose(raw[1], result["raw"][1] + 1.0), "raw prompt mean")


def _matcher_smoke() -> None:
    matcher_config = DenseMatcherConfig(
        canonical_dim=8, projection_dim=4, offset_hidden_dim=6, temperature=0.2
    )
    model = DenseSubgridMatcher(matcher_config)
    query = torch.randn(1, 4, 8)
    key = torch.randn(1, 9, 8)
    valid = torch.ones(1, 4, dtype=torch.bool)
    output = model(query, key, valid, torch.ones(1, 9, dtype=torch.bool))
    target = build_dense_match_target(
        torch.tensor([[[1.0, 1.0], [4.0, 4.0], [7.0, 7.0], [3.0, 6.0]]]),
        query_valid=valid, visible=valid, grid_size=(3, 3), image_size=(9, 9),
    )
    result = dense_match_loss(model, output, target)
    result.loss.backward()
    _require(model.query_projection.weight.grad is not None, "query gradient")
    _require(model.key_projection.weight.grad is not None, "key gradient")


def _decision_smoke() -> None:
    summaries = []
    frames = []
    v97 = []
    audit = []
    for fold in FOLDS:
        methods = [("sam_memory", value) for value in TEMPORAL_VARIANTS] + [
            (architecture, "normal")
            for architecture in TEMPORAL_ARCHITECTURES
            if architecture != "sam_memory"
        ]
        for architecture, variant in methods:
            candidate = architecture == "sam_memory" and variant == "normal"
            perturb = architecture == "sam_memory" and variant != "normal"
            row = dict.fromkeys(SUMMARY_COLUMNS, 0)
            row.update({
                "fold": fold.name, "architecture": architecture, "variant": variant,
                "parameters": 100, "frames": 4, "active_frames": 4,
                "control_support_exact": 1, "matcher_frozen_exact": 1,
                "perturbed_pairs": 4 if perturb else 0,
                "pck_accuracy": 0.9 if candidate else (0.2 if perturb else 0.4),
                "mean_epe_pixels": 0.1 if candidate else (4.0 if perturb else 2.0),
                "raw_rotation_error_deg": 3.0,
                "refined_rotation_error_deg": 0.2 if candidate else (3.0 if perturb else 2.0),
                "raw_translation_direction_error_deg": 5.0,
                "refined_translation_direction_error_deg": 0.3 if candidate else (5.0 if perturb else 3.0),
                "refined_relative_aggregate_deg": 0.5 if candidate else (8.0 if perturb else 5.0),
                "relative_aggregate_worse_frames": 0,
            })
            summaries.append(row)
            for frame_index in fold.test_frames:
                frame = dict.fromkeys(FRAME_COLUMNS, 0)
                frame.update({"fold": fold.name, "architecture": architecture,
                              "variant": variant, "frame_index": frame_index})
                frames.append(frame)
        for architecture in ("sam_dense", "stream_dense"):
            v97.append({
                "fold": fold.name, "architecture": architecture, "variant": "normal",
                "pck_accuracy": "0.3", "mean_epe_pixels": "3.0",
                "refined_relative_aggregate_deg": "6.0",
            })
        for split in ("train_prefix", "future_fixed_edges"):
            row = dict.fromkeys(AUDIT_COLUMNS, 0)
            row.update({"fold": fold.name, "split": split,
                        "pck_accuracy": 0.8 if split == "train_prefix" else 0.0,
                        "mean_epe_pixels": 1.0 if split == "train_prefix" else 100.0})
            audit.append(row)
    _annotate_decisions(summaries, v97)
    _validate_outputs(summaries, frames, audit)
    decision = _decision_markdown(summaries, v97, audit)
    _require("all-fold causal pass: `1`" in decision, "causal gate")


def _require(condition, label: str) -> None:
    if not bool(condition):
        raise RuntimeError(f"V9.8 smoke failed: {label}")


if __name__ == "__main__":
    main()
