#!/usr/bin/env python3
"""Dependency-free checks for the locked V7.4 temporal protocol."""

from __future__ import annotations

from streaming_couping.src.v74_temporal_protocol import (
    EXPECTED_FRAMES,
    FOLDS,
    GEOMETRY_CONTROL,
    PRIMARY_SAM_BRANCH,
    SAM_OFF_CONTROL,
    V74_COLUMNS,
    annotate_controls,
    validate_folds,
    validate_rows,
)


def main() -> None:
    _check_folds()
    _check_positive_decision()
    _check_failed_perturbation()
    _check_mismatched_support()
    _check_csv_schema()
    print("V7.4 dependency-free protocol smoke passed")


def _check_folds() -> None:
    validate_folds(FOLDS, available_frames=set(EXPECTED_FRAMES))
    expected = {
        "short": (
            (270, 285, 300, 315, 330, 345),
            (360, 375, 390, 405),
        ),
        "medium": (tuple(range(270, 406, 15)), (420, 435, 450, 465)),
        "long": (tuple(range(270, 466, 15)), (480, 495, 510, 525)),
    }
    actual = {
        fold.name: (fold.train_frames, fold.test_frames) for fold in FOLDS
    }
    _require(actual == expected, "locked temporal folds changed")


def _check_positive_decision() -> None:
    rows = _decision_rows()
    annotate_controls(rows)
    for row in rows:
        if row["architecture"] != PRIMARY_SAM_BRANCH:
            continue
        _require(row["fold_sam_causal_pass"] == 1, "valid fold did not pass")
        _require(row["all_folds_sam_causal_pass"] == 1, "all-fold pass lost")
        _require(row["control_support_exact"] == 1, "support match lost")
        _require(row["beats_geometry_control"] == 1, "geometry win lost")
        _require(row["beats_sam_off_trained_control"] == 1, "SAM-off win lost")


def _check_failed_perturbation() -> None:
    rows = _decision_rows()
    candidate = next(
        row
        for row in rows
        if row["fold"] == "short"
        and row["architecture"] == PRIMARY_SAM_BRANCH
    )
    candidate["wrong_sam_identity_damage_percent"] = "0.99"
    annotate_controls(rows)
    _require(candidate["fold_sam_causal_pass"] == 0, "weak SAM damage passed")
    for row in rows:
        if row["architecture"] == PRIMARY_SAM_BRANCH:
            _require(
                row["all_folds_sam_causal_pass"] == 0,
                "one failed fold did not clear the global decision",
            )


def _check_csv_schema() -> None:
    complete = {name: "" for name in V74_COLUMNS}
    validate_rows([complete])
    missing = dict(complete)
    missing.pop("test_loss")
    try:
        validate_rows([missing])
    except ValueError:
        pass
    else:
        raise RuntimeError("V7.4 smoke failed: missing CSV column accepted")


def _check_mismatched_support() -> None:
    rows = _decision_rows()
    candidate = next(
        row
        for row in rows
        if row["fold"] == "medium"
        and row["architecture"] == PRIMARY_SAM_BRANCH
    )
    candidate["test_active_frame_indices"] = "420"
    annotate_controls(rows)
    _require(candidate["control_support_exact"] == 0, "support drift hidden")
    _require(candidate["fold_sam_causal_pass"] == 0, "support drift passed")


def _decision_rows() -> list[dict[str, object]]:
    rows = []
    for fold in FOLDS:
        rows.extend(
            [
                _row(fold.name, GEOMETRY_CONTROL, 1.0),
                _row(fold.name, SAM_OFF_CONTROL, 0.9),
                _row(fold.name, PRIMARY_SAM_BRANCH, 0.8),
            ]
        )
    return rows


def _row(fold: str, architecture: str, loss: float) -> dict[str, object]:
    return {
        "fold": fold,
        "architecture": architecture,
        "underlying_architecture": architecture,
        "train_active_frame_indices": "270 285",
        "test_active_frame_indices": "360 375",
        "test_loss": str(loss),
        "geometry_control_test_loss": "",
        "sam_off_trained_control_test_loss": "",
        "gain_vs_geometry_percent": "",
        "gain_vs_sam_off_trained_percent": "",
        "sam_off_damage_percent": "2",
        "uniform_sam_damage_percent": "2",
        "wrong_sam_identity_damage_percent": "2",
        "shuffle_sam_time_damage_percent": "2",
        "train_overfit_pass": 1,
        "reference_exact": 1,
        "inactive_fallback_exact": 1,
        "control_support_exact": 0,
        "beats_geometry_control": 0,
        "beats_sam_off_trained_control": 0,
        "fold_sam_causal_pass": 0,
        "all_folds_sam_causal_pass": 0,
    }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"V7.4 smoke failed: {message}")


if __name__ == "__main__":
    main()
