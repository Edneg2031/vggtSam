#!/usr/bin/env python3
"""Dependency-free V7.5 protocol and decision-logic smoke test."""

from __future__ import annotations

from streaming_couping.src.v75_explicit_protocol import (
    EXPECTED_FRAMES,
    FOLDS,
    MEMORY_MODES,
    PRIMARY,
    REGION_CONTROLS,
    annotate_decisions,
    run_specs,
    validate_protocol,
)


def main() -> None:
    validate_protocol(available_frames=set(EXPECTED_FRAMES))
    specs = run_specs()
    _require(
        sum(variant.name == PRIMARY for variant, _ in specs) == len(MEMORY_MODES),
        "primary must run once per memory mode",
    )
    rows = []
    for fold in FOLDS:
        losses = {
            "raw_streamvggt": 1.00,
            "sam_current_ray": 0.90,
            "full_image_history": 0.88,
            "sam_bbox_history": 0.87,
            "random_same_area_history": 0.89,
            "sam_stale_time_history": 0.86,
            "sam_history_correct": 0.70,
            "sam_history_wrong_id": 0.95,
            "gt_mask_oracle_history": 0.60,
        }
        for variant, mode in specs:
            rows.append(
                {
                    "fold": fold.name,
                    "variant": variant.name,
                    "memory_mode": mode,
                    # Deliberately constant so this smoke fails if the causal
                    # verdict ever regresses to the combined pose loss.
                    "pose_loss": 5.0,
                    "center_error": losses[variant.name],
                    "gain_vs_current_only_percent": "",
                    "gain_vs_wrong_id_percent": "",
                    "best_region_control_center_error": "",
                    "worst_region_control_center_error": "",
                    "gain_vs_best_region_control_percent": "",
                    "oracle_gain_vs_raw_percent": "",
                    "oracle_solver_pass": 0,
                    "sam_region_pass": 0,
                    "sam_history_pass": 0,
                    "all_folds_frozen_region_pass": 0,
                    "all_folds_online_region_pass": 0,
                    "all_folds_region_pass": 0,
                    "all_folds_frozen_history_pass": 0,
                    "all_folds_online_history_pass": 0,
                    "all_folds_history_pass": 0,
                }
            )
    annotate_decisions(rows)
    primary = [
        row
        for row in rows
        if row["variant"] == PRIMARY and row["memory_mode"] == "online"
    ]
    _require(len(primary) == len(FOLDS), "missing online primary folds")
    _require(
        all(int(row["sam_region_pass"]) == 1 for row in primary),
        f"region controls did not pass: {REGION_CONTROLS}",
    )
    _require(
        all(int(row["sam_history_pass"]) == 1 for row in primary),
        "history controls did not pass",
    )
    _require(
        all(int(row["all_folds_history_pass"]) == 1 for row in primary),
        "aggregate history verdict failed",
    )
    frozen_failure = [dict(row) for row in rows]
    failed = next(
        row
        for row in frozen_failure
        if row["fold"] == "short"
        and row["variant"] == PRIMARY
        and row["memory_mode"] == "frozen"
    )
    failed["center_error"] = 1.05
    annotate_decisions(frozen_failure)
    marker = next(
        row
        for row in frozen_failure
        if row["variant"] == PRIMARY and row["memory_mode"] == "online"
    )
    _require(
        int(marker["all_folds_frozen_history_pass"]) == 0
        and int(marker["all_folds_online_history_pass"]) == 1
        and int(marker["all_folds_history_pass"]) == 0,
        "strict verdict did not preserve frozen/online distinction",
    )
    print("V7.5 explicit-pose protocol smoke passed")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"V7.5 protocol smoke failed: {message}")


if __name__ == "__main__":
    main()
