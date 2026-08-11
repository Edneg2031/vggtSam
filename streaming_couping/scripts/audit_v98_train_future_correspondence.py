#!/usr/bin/env python3
"""Audit V9.8 train-prefix versus future correspondence without retraining."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import math
from pathlib import Path

import torch

from streaming_couping.scripts.run_v90_local_token_matcher import load_stage_a_config
from streaming_couping.scripts.run_v92_support_factorization import (
    _build_support_records,
    _load_support_payload,
    _prepare_support_data,
    load_support_config,
)
from streaming_couping.scripts.run_v93_quantization_tolerance import (
    load_diagnostic_config,
)
from streaming_couping.scripts.run_v95_spatial_support_scope import (
    _locked_edge_indices,
    load_spatial_scope_config,
)
from streaming_couping.scripts.run_v96_dense_grid_upper_bound import (
    load_dense_grid_config,
)
from streaming_couping.scripts.run_v97_dense_descriptor_causality import (
    _build_evaluation_records,
    _build_training_records,
    _prepare_dense_data,
    load_v97_config,
)
from streaming_couping.scripts.run_v98_temporal_memory_causality import (
    _architecture_features,
    _checkpoint_signature,
    _correspondence_metrics,
    _prepare_temporal_data,
    _require_temporal_support,
    _torch_load,
    _training_device,
    load_v98_config,
)
from streaming_couping.src.v74_temporal_protocol import FOLDS
from streaming_couping.src.v97_dense_descriptor_matcher import DenseSubgridMatcher


COLUMNS = (
    "fold",
    "architecture",
    "split",
    "pairs",
    "queries",
    "pck_threshold_pixels",
    "pck_accuracy",
    "mean_epe_pixels",
    "initial_train_loss",
    "final_train_loss",
    "train_loss_drop_percent",
    "checkpoint_signature_valid",
    "checkpoint_uses_pose_loss",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="streaming_couping/configs/v98_temporal_memory_causality.yaml",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--temporal-cache-path", default=None)
    parser.add_argument("--v97-dense-cache-path", default=None)
    args = parser.parse_args()
    config = load_v98_config(args.config)
    updates = {}
    for name in ("output_dir", "temporal_cache_path", "v97_dense_cache_path"):
        value = getattr(args, name)
        if value:
            updates[name] = Path(value).expanduser().resolve()
    if updates:
        config = replace(config, **updates)
    csv_path, decision_path = audit_v98_train_future(config)
    print(f"V9.8 train/future audit={csv_path}")
    print(f"V9.8 train/future decision={decision_path}")


def audit_v98_train_future(config) -> tuple[Path, Path]:
    v97 = replace(
        load_v97_config(config.v97_config),
        data_config=config.data_config,
        dense_cache_path=config.v97_dense_cache_path,
    )
    v96 = load_dense_grid_config(v97.v96_config)
    v95 = load_spatial_scope_config(v96.v95_config)
    v93 = load_diagnostic_config(v95.v93_config)
    support_config = load_support_config(v93.support_config)
    stage_config = replace(
        load_stage_a_config(support_config.stage_a_config),
        data_config=config.data_config,
        output_dir=config.output_dir,
    )
    payload, source_cache = _load_support_payload(support_config)
    support = _prepare_support_data(
        payload, config=support_config, stage_config=stage_config
    )
    del payload
    dense = _prepare_dense_data(
        _torch_load(config.v97_dense_cache_path, "V9.7 dense cache"),
        support=support,
        config=v97,
    )
    temporal = _prepare_temporal_data(
        _torch_load(config.temporal_cache_path, "V9.8 temporal cache"),
        dense=dense,
        config=config,
    )
    positions = {frame: index for index, frame in enumerate(support.frames)}
    instance_records = {
        fold.name: _build_support_records(
            support,
            current_indices=[positions[value] for value in fold.test_frames],
            query_token_count=support_config.query_token_count,
            stage_config=stage_config,
        )
        for fold in FOLDS
    }
    fixed_edges = _locked_edge_indices(v95, support)
    future_records = {
        fold.name: _build_evaluation_records(
            fold_name=fold.name,
            fixed_edges=fixed_edges[fold.name],
            instance_records=instance_records[fold.name],
            data=dense,
            stage_config=stage_config,
            query_count=None,
        )
        for fold in FOLDS
    }
    train_records = {
        fold.name: _build_training_records(
            [positions[value] for value in fold.train_frames],
            data=dense,
            stage_config=stage_config,
            config=v97,
        )
        for fold in FOLDS
    }
    _require_temporal_support(
        temporal, train_records, future_records, support.frames
    )

    rows = []
    for fold in FOLDS:
        for architecture in ("sam_memory", "memory_off_raw"):
            checkpoint = (
                config.output_dir / "checkpoints" / f"{fold.name}_{architecture}.pt"
            )
            saved = _torch_load(checkpoint, "completed V9.8 checkpoint")
            expected = _checkpoint_signature(
                config, fold.name, architecture, source_cache
            )
            signature_valid = int(saved.get("signature") == expected)
            uses_pose_loss = int(bool(saved.get("uses_pose_loss", False)))
            if not signature_valid or uses_pose_loss:
                raise ValueError(
                    f"V9.8 audit refuses incompatible checkpoint={checkpoint}."
                )
            model = DenseSubgridMatcher(config.matcher).to(
                _training_device(config.training.device)
            )
            model.load_state_dict(saved["model"])
            model.eval()
            features = _architecture_features(architecture, temporal)
            drop = _gain(saved["initial_loss"], saved["final_loss"])
            for split, records in (
                ("train_prefix", train_records[fold.name]),
                ("future_fixed_edges", future_records[fold.name]),
            ):
                metrics = _correspondence_metrics(
                    model, records, features, dense, config
                )
                rows.append(
                    {
                        "fold": fold.name,
                        "architecture": architecture,
                        "split": split,
                        "pairs": len(records),
                        "queries": metrics["queries"],
                        "pck_threshold_pixels": config.matcher.pck_threshold_pixels,
                        "pck_accuracy": metrics["pck"],
                        "mean_epe_pixels": metrics["epe"],
                        "initial_train_loss": float(saved["initial_loss"]),
                        "final_train_loss": float(saved["final_loss"]),
                        "train_loss_drop_percent": drop,
                        "checkpoint_signature_valid": signature_valid,
                        "checkpoint_uses_pose_loss": uses_pose_loss,
                    }
                )
            del model

    _validate(rows)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = config.output_dir / "v98_train_future_correspondence_audit.csv"
    decision_path = config.output_dir / "v98_train_future_correspondence_audit.md"
    _write_csv(csv_path, rows)
    decision_path.write_text(_decision_markdown(rows), encoding="utf8")
    return csv_path, decision_path


def _decision_markdown(rows) -> str:
    lines = [
        "# V9.8 train/future correspondence audit",
        "",
        "No model is trained. Completed provenance-compatible V9.8 checkpoints are evaluated on their original train-prefix pairs and locked future edges.",
        "",
        "| fold | feature | train loss | train PCK@1 | train EPE | future PCK@1 | future EPE | PCK gap |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for fold in FOLDS:
        for architecture in ("sam_memory", "memory_off_raw"):
            group = {
                row["split"]: row
                for row in rows
                if row["fold"] == fold.name
                and row["architecture"] == architecture
            }
            train = group["train_prefix"]
            future = group["future_fixed_edges"]
            lines.append(
                f"| {fold.name} | {architecture} "
                f"| {float(train['initial_train_loss']):.5g}→{float(train['final_train_loss']):.5g} "
                f"| {float(train['pck_accuracy']):.6g} "
                f"| {float(train['mean_epe_pixels']):.6g} "
                f"| {float(future['pck_accuracy']):.6g} "
                f"| {float(future['mean_epe_pixels']):.6g} "
                f"| {float(train['pck_accuracy']) - float(future['pck_accuracy']):.6g} |"
            )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- High train PCK with collapsed future PCK means temporal overfitting.",
            "- Low train PCK despite a large loss drop means the coarse-cell/offset objective is not producing pixel-accurate correspondence even in sample.",
            "- In either case, the completed V9.8 future causal conclusion is unchanged; this audit only localizes the failure.",
            "",
        ]
    )
    return "\n".join(lines)


def _validate(rows) -> None:
    if len(rows) != len(FOLDS) * 2 * 2:
        raise ValueError("V9.8 train/future audit row count is not 12.")
    expected = set(COLUMNS)
    keys = {
        (row["fold"], row["architecture"], row["split"])
        for row in rows
    }
    if len(keys) != len(rows):
        raise ValueError("V9.8 train/future audit has duplicate rows.")
    for row in rows:
        if set(row) != expected:
            raise ValueError("V9.8 train/future audit schema mismatch.")
        if not math.isfinite(float(row["pck_accuracy"])):
            raise ValueError("V9.8 train/future audit contains non-finite PCK.")
        if not math.isfinite(float(row["mean_epe_pixels"])):
            raise ValueError("V9.8 train/future audit contains non-finite EPE.")


def _write_csv(path: Path, rows) -> None:
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _gain(initial, final) -> float:
    initial = float(initial)
    final = float(final)
    return 0.0 if initial <= 1e-12 else 100.0 * (initial - final) / initial


if __name__ == "__main__":
    main()
