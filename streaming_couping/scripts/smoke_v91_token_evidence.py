#!/usr/bin/env python3
"""Dependency-light V9.1 discrete oracle and matcher-audit smoke."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from streaming_couping.scripts.run_v91_token_evidence_audit import (
    EDGE_COLUMNS,
    FRAME_COLUMNS,
    METHODS,
    PAIR_COLUMNS,
    SUMMARY_COLUMNS,
    _annotate_a0q,
    _decision_markdown,
    _validate_output_rows,
    _write_csv,
    load_audit_config,
)
from streaming_couping.src.v74_temporal_protocol import FOLDS
from streaming_couping.src.v90_epipolar_geometry import LocalTokenReprojection
from streaming_couping.src.v90_explicit_matcher import (
    MatcherConfig,
    build_soft_match_target,
)
from streaming_couping.src.v91_token_evidence import (
    PREDICTION_MODES,
    audit_token_probability,
    decode_token_probability,
    hard_discrete_oracle_probability,
    raw_cosine_probability,
)


def main() -> None:
    labels, target, history_uv, valid = _example()
    oracle = hard_discrete_oracle_probability(target)
    decoded = decode_token_probability(
        oracle,
        history_uv_normalized=history_uv,
        query_valid=valid,
        key_valid=torch.ones(2, dtype=torch.bool),
        image_size=(48, 64),
        mode="top1",
    )
    metrics = audit_token_probability(
        oracle,
        labels=labels,
        target=target,
        prediction=decoded,
        pck_threshold_pixels=8.0,
        image_size=(48, 64),
    )
    _require(int(decoded.accepted.sum()) == 2, "A0-Q accepts only supported rows")
    _require(metrics.visible_queries == 3, "visible denominator is retained")
    _require(metrics.visible_supported_queries == 2, "discrete support is explicit")
    _require(metrics.supported_pck_correct == 2, "nearest discrete keys are correct")
    _require(metrics.dustbin_correct == 1, "unsupported projection uses dustbin")

    features = torch.eye(3, dtype=torch.float32)[:2]
    raw = raw_cosine_probability(
        features,
        features,
        torch.ones(2, dtype=torch.bool),
        torch.ones(2, dtype=torch.bool),
        canonical_dim=3,
        temperature=0.07,
    )
    _require(tuple(raw.shape) == (2, 3), "raw cosine includes one dustbin")
    _require(raw[:, :-1].argmax(dim=-1).tolist() == [0, 1], "raw cosine Top-1")

    soft_probability = torch.tensor([[0.75, 0.25, 0.0]], dtype=torch.float32)
    top1 = decode_token_probability(
        soft_probability,
        history_uv_normalized=history_uv,
        query_valid=torch.ones(1, dtype=torch.bool),
        key_valid=torch.ones(2, dtype=torch.bool),
        image_size=(48, 64),
        mode="top1",
    )
    soft = decode_token_probability(
        soft_probability,
        history_uv_normalized=history_uv,
        query_valid=torch.ones(1, dtype=torch.bool),
        key_valid=torch.ones(2, dtype=torch.bool),
        image_size=(48, 64),
        mode="soft_expectation",
    )
    _require(
        not torch.allclose(top1.predicted_uv, soft.predicted_uv),
        "Top-1 and soft expectation are independently audited",
    )
    config = load_audit_config(
        "streaming_couping/configs/v91_token_evidence_audit.yaml"
    )
    _require(
        config.coverage_thresholds_pixels == (8.0, 12.0, 16.0),
        "coverage thresholds are locked",
    )
    _schema_smoke()
    print("V9.1 token evidence smoke passed")


def _schema_smoke() -> None:
    """Exercise runner cardinality, decision and exact CSV schemas synthetically."""

    summary_rows = []
    pair_rows = []
    frame_rows = []
    methods_per_split = 1 + (len(METHODS) - 1) * len(PREDICTION_MODES)
    records_by_fold = {
        fold.name: {"train": [object()], "test": [object()]} for fold in FOLDS
    }
    for fold in FOLDS:
        for split, frames in (("train", fold.train_frames), ("test", fold.test_frames)):
            for method in METHODS:
                modes = ("top1",) if method.oracle else PREDICTION_MODES
                for mode in modes:
                    row = dict.fromkeys(SUMMARY_COLUMNS, 0)
                    row.update(
                        {
                            "stage": "A0-Q" if method.oracle else "AUDIT",
                            "fold": fold.name,
                            "split": split,
                            "method": method.name,
                            "prediction_mode": mode,
                            "frames": len(frames),
                            "active_frames": len(frames),
                            "raw_edge_rotation_error_deg": 2.0,
                            "refined_edge_rotation_error_deg": 1.0,
                            "raw_edge_translation_direction_error_deg": 2.0,
                            "refined_edge_translation_direction_error_deg": 1.0,
                        }
                    )
                    summary_rows.append(row)
            pair_rows.extend(
                dict.fromkeys(PAIR_COLUMNS, 0) for _ in range(methods_per_split)
            )
            frame_rows.extend(
                dict.fromkeys(FRAME_COLUMNS, 0)
                for _ in range(methods_per_split * len(frames))
            )
    _validate_output_rows(
        summary_rows=summary_rows,
        pair_rows=pair_rows,
        frame_rows=frame_rows,
        records_by_fold=records_by_fold,
    )
    _annotate_a0q(summary_rows)
    decision = _decision_markdown(summary_rows)
    _require(
        "A0-Q discrete local32 all-fold pass: `1`" in decision,
        "synthetic decision is complete",
    )
    with TemporaryDirectory() as directory:
        root = Path(directory)
        for name, columns in (
            ("summary.csv", SUMMARY_COLUMNS),
            ("pairs.csv", PAIR_COLUMNS),
            ("frames.csv", FRAME_COLUMNS),
            ("edges.csv", EDGE_COLUMNS),
        ):
            path = root / name
            _write_csv(path, [dict.fromkeys(columns, 0)], columns)
            _require(path.read_text(encoding="utf8").startswith(columns[0]), name)


def _example():
    image_size = (48, 64)
    key_pixels = torch.tensor([[10.0, 20.0], [50.0, 20.0]], dtype=torch.float64)
    history_uv = torch.stack(
        [
            key_pixels[:, 0] / 63.0 * 2.0 - 1.0,
            key_pixels[:, 1] / 47.0 * 2.0 - 1.0,
        ],
        dim=-1,
    ).float()
    current_uv = torch.tensor(
        [[8.0, 18.0], [48.0, 18.0], [30.0, 40.0]], dtype=torch.float64
    )
    target_uv = torch.tensor(
        [[11.0, 20.0], [49.0, 20.0], [30.0, 40.0]], dtype=torch.float64
    )
    valid = torch.ones(3, dtype=torch.bool)
    labels = LocalTokenReprojection(
        current_frame=2,
        history_frame=1,
        slot=0,
        current_uv=current_uv,
        history_target_uv=target_uv,
        query_valid=valid,
        target_visible=valid.clone(),
        weights=torch.ones(3, dtype=torch.float64),
        depth_residual_metric=torch.zeros(3, dtype=torch.float64),
    )
    target = build_soft_match_target(
        labels,
        history_uv_normalized=history_uv,
        history_valid=torch.ones(2, dtype=torch.bool),
        image_size=image_size,
        config=MatcherConfig(
            canonical_dim=4,
            projection_dim=2,
            target_sigma_pixels=3.0,
            target_radius_pixels=12.0,
        ),
    )
    return labels, target, history_uv, valid


def _require(condition: bool, label: str) -> None:
    if not condition:
        raise RuntimeError(f"V9.1 smoke failed: {label}")


if __name__ == "__main__":
    main()
