"""Dependency-free protocol definitions for the V7.4 temporal experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TemporalFold:
    name: str
    train_frames: tuple[int, ...]
    test_frames: tuple[int, ...]


EXPECTED_FRAMES = tuple(range(90, 526, 15))

FOLDS = (
    TemporalFold(
        name="short",
        train_frames=tuple(range(270, 346, 15)),
        test_frames=tuple(range(360, 406, 15)),
    ),
    TemporalFold(
        name="medium",
        train_frames=tuple(range(270, 406, 15)),
        test_frames=tuple(range(420, 466, 15)),
    ),
    TemporalFold(
        name="long",
        train_frames=tuple(range(270, 466, 15)),
        test_frames=tuple(range(480, 526, 15)),
    ),
)

PRIMARY_SAM_BRANCH = "sam_geometry_transport"
SAM_OFF_CONTROL = "sam_geometry_train_sam_off"
GEOMETRY_CONTROL = "geometry_transport"
MINIMUM_PERTURBATION_DAMAGE_PERCENT = 1.0
MINIMUM_CONTROL_GAIN_PERCENT = 1.0

V74_COLUMNS = (
    "fold",
    "architecture",
    "underlying_architecture",
    "train_input",
    "token_count",
    "seed",
    "train_frames",
    "test_frames",
    "train_active_frames",
    "train_active_frame_indices",
    "test_active_frames",
    "test_active_frame_indices",
    "test_inactive_frames",
    "test_inactive_frame_indices",
    "parameters",
    "trainable_parameters",
    "steps",
    "best_step",
    "training_seconds",
    "initial_train_active_loss",
    "final_train_active_loss",
    "max_train_frame_loss",
    "worst_train_frame",
    "train_loss_drop_percent",
    "train_overfit_pass",
    "test_rotation_deg",
    "test_translation",
    "test_loss",
    "test_active_loss",
    "max_test_frame_loss",
    "worst_test_frame",
    "gain_vs_frozen_l0_percent",
    "geometry_control_test_loss",
    "sam_off_trained_control_test_loss",
    "gain_vs_geometry_percent",
    "gain_vs_sam_off_trained_percent",
    "normal_test_loss",
    "sam_off_test_loss",
    "uniform_sam_test_loss",
    "wrong_sam_identity_test_loss",
    "shuffle_sam_time_test_loss",
    "wrong_local_geometry_test_loss",
    "sam_off_damage_percent",
    "uniform_sam_damage_percent",
    "wrong_sam_identity_damage_percent",
    "shuffle_sam_time_damage_percent",
    "reference_exact",
    "inactive_fallback_exact",
    "control_support_exact",
    "beats_geometry_control",
    "beats_sam_off_trained_control",
    "fold_sam_causal_pass",
    "all_folds_sam_causal_pass",
    "peak_gpu_memory_gib",
)


def validate_folds(
    folds: tuple[TemporalFold, ...], *, available_frames: set[int]
) -> None:
    if tuple(fold.name for fold in folds) != ("short", "medium", "long"):
        raise ValueError("V7.4 fold names or order changed unexpectedly.")
    for fold in folds:
        train = fold.train_frames
        test = fold.test_frames
        if not train or not test or set(train) & set(test):
            raise ValueError(f"Invalid V7.4 fold={fold.name} partition.")
        if max(train) >= min(test):
            raise ValueError(f"V7.4 fold={fold.name} is not chronological.")
        missing = (set(train) | set(test)) - available_frames
        if missing:
            raise ValueError(
                f"V7.4 fold={fold.name} lacks frames={sorted(missing)}."
            )
        if tuple(range(train[0], train[-1] + 1, 15)) != train:
            raise ValueError(f"V7.4 fold={fold.name} train frames have gaps.")
        if tuple(range(test[0], test[-1] + 1, 15)) != test:
            raise ValueError(f"V7.4 fold={fold.name} test frames have gaps.")


def annotate_controls(rows: list[dict[str, Any]]) -> None:
    """Add locked control gains and the strict SAM-causality decision."""

    by_fold = {
        fold.name: {
            row["architecture"]: row
            for row in rows
            if row["fold"] == fold.name
        }
        for fold in FOLDS
    }
    required = {GEOMETRY_CONTROL, SAM_OFF_CONTROL, PRIMARY_SAM_BRANCH}
    fold_passes = []
    for fold in FOLDS:
        group = by_fold[fold.name]
        missing = required - set(group)
        if missing:
            raise ValueError(
                f"V7.4 fold={fold.name} lacks controls={sorted(missing)}."
            )
        geometry_loss = float(group[GEOMETRY_CONTROL]["test_loss"])
        sam_off_loss = float(group[SAM_OFF_CONTROL]["test_loss"])
        for row in group.values():
            if row["underlying_architecture"] == "control":
                continue
            current = float(row["test_loss"])
            row["geometry_control_test_loss"] = _short(geometry_loss)
            row["sam_off_trained_control_test_loss"] = _short(sam_off_loss)
            row["gain_vs_geometry_percent"] = _short(
                _gain(geometry_loss, current)
            )
            row["gain_vs_sam_off_trained_percent"] = _short(
                _gain(sam_off_loss, current)
            )
            row["beats_geometry_control"] = int(current < geometry_loss)
            row["beats_sam_off_trained_control"] = int(
                current < sam_off_loss
            )
        candidate = group[PRIMARY_SAM_BRANCH]
        support_fields = (
            "train_active_frame_indices",
            "test_active_frame_indices",
        )
        support_exact = all(
            candidate[field] == group[GEOMETRY_CONTROL][field]
            == group[SAM_OFF_CONTROL][field]
            for field in support_fields
        )
        candidate["control_support_exact"] = int(support_exact)
        perturbation_fields = (
            "sam_off_damage_percent",
            "uniform_sam_damage_percent",
            "wrong_sam_identity_damage_percent",
            "shuffle_sam_time_damage_percent",
        )
        perturbations_hurt = all(
            float(candidate[field]) >= MINIMUM_PERTURBATION_DAMAGE_PERCENT
            for field in perturbation_fields
        )
        controls_beaten = (
            float(candidate["gain_vs_geometry_percent"])
            >= MINIMUM_CONTROL_GAIN_PERCENT
            and float(candidate["gain_vs_sam_off_trained_percent"])
            >= MINIMUM_CONTROL_GAIN_PERCENT
        )
        passed = int(
            int(candidate["train_overfit_pass"]) == 1
            and int(candidate["reference_exact"]) == 1
            and int(candidate["inactive_fallback_exact"]) == 1
            and support_exact
            and controls_beaten
            and perturbations_hurt
        )
        candidate["fold_sam_causal_pass"] = passed
        fold_passes.append(passed)
    all_passed = int(all(fold_passes))
    for fold in FOLDS:
        by_fold[fold.name][PRIMARY_SAM_BRANCH][
            "all_folds_sam_causal_pass"
        ] = all_passed


def validate_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Refusing to write an empty V7.4 result CSV.")
    expected = set(V74_COLUMNS)
    for index, row in enumerate(rows):
        keys = set(row)
        if keys != expected:
            raise ValueError(
                f"V7.4 CSV row={index} schema mismatch: "
                f"missing={sorted(expected - keys)}, "
                f"extra={sorted(keys - expected)}."
            )


def _gain(initial: float, final: float) -> float:
    return 0.0 if initial <= 1e-12 else 100.0 * (initial - final) / initial


def _short(value: float) -> str:
    return f"{float(value):.8g}"
