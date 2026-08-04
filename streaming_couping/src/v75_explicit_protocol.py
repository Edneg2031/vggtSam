"""Dependency-free protocol for the V7.5 explicit-pose causality test.

V7.5 is deliberately not another learned fusion architecture.  It combines
the V7.4 forward-only dynamic-instance observations with the retained V5
analytic centre solver and the existing bounded instance ICP proposal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExplicitPoseFold:
    name: str
    history_frames: tuple[int, ...]
    test_frames: tuple[int, ...]


@dataclass(frozen=True)
class ExplicitVariant:
    name: str
    region: str
    history: str
    report_only: bool = False


EXPECTED_FRAMES = tuple(range(90, 526, 15))

# Each fold asks whether a prefix can constrain the immediately following
# frames.  Unlike V7.4, there is no residual training interval.
FOLDS = (
    ExplicitPoseFold(
        name="short",
        history_frames=tuple(range(90, 256, 15)),
        test_frames=tuple(range(270, 346, 15)),
    ),
    ExplicitPoseFold(
        name="medium",
        history_frames=tuple(range(90, 346, 15)),
        test_frames=tuple(range(360, 406, 15)),
    ),
    ExplicitPoseFold(
        name="long",
        history_frames=tuple(range(90, 406, 15)),
        test_frames=tuple(range(420, 526, 15)),
    ),
)

VARIANTS = (
    ExplicitVariant("raw_streamvggt", "none", "none"),
    ExplicitVariant("sam_current_ray", "sam", "none"),
    ExplicitVariant("full_image_history", "full", "correct"),
    ExplicitVariant("sam_bbox_history", "bbox", "correct"),
    ExplicitVariant("random_same_area_history", "random", "correct"),
    ExplicitVariant("sam_stale_time_history", "stale", "correct"),
    ExplicitVariant("sam_history_correct", "sam", "correct"),
    ExplicitVariant("sam_history_wrong_id", "sam", "wrong_id"),
    ExplicitVariant(
        "gt_mask_oracle_history",
        "gt",
        "correct",
        report_only=True,
    ),
)

MEMORY_MODES = ("frozen", "online")
PRIMARY = "sam_history_correct"
WRONG_ID = "sam_history_wrong_id"
CURRENT_ONLY = "sam_current_ray"
ORACLE = "gt_mask_oracle_history"
REGION_CONTROLS = (
    "full_image_history",
    "sam_bbox_history",
    "random_same_area_history",
    "sam_stale_time_history",
)
MINIMUM_GAIN_PERCENT = 1.0

SUMMARY_COLUMNS = (
    "fold",
    "variant",
    "region_source",
    "history_source",
    "memory_mode",
    "report_only",
    "history_frames",
    "test_frames",
    "test_frame_count",
    "active_frames",
    "active_frame_indices",
    "inactive_frames",
    "inactive_frame_indices",
    "observed_test_frames",
    "mean_observed_instances",
    "mean_map_instances_before",
    "ray_accepted_frames",
    "icp_proposal_frames",
    "mean_participating_instances",
    "mean_candidate_points",
    "mean_sampled_points",
    "mean_applied_center_shift",
    "max_applied_center_shift",
    "rotation_deg",
    "center_error",
    "center_rmse",
    "pose_loss",
    "rpe_translation",
    "decision_metric",
    "improved_frames_vs_raw",
    "worse_frames_vs_raw",
    "gain_vs_raw_percent",
    "gain_vs_current_only_percent",
    "gain_vs_wrong_id_percent",
    "worst_region_control_center_error",
    "best_region_control_center_error",
    "gain_vs_best_region_control_percent",
    "oracle_gain_vs_raw_percent",
    "oracle_solver_pass",
    "sam_region_pass",
    "sam_history_pass",
    "all_folds_frozen_region_pass",
    "all_folds_online_region_pass",
    "all_folds_region_pass",
    "all_folds_frozen_history_pass",
    "all_folds_online_history_pass",
    "all_folds_history_pass",
)


def validate_protocol(*, available_frames: set[int]) -> None:
    if tuple(fold.name for fold in FOLDS) != ("short", "medium", "long"):
        raise ValueError("V7.5 fold names or order changed unexpectedly.")
    if tuple(variant.name for variant in VARIANTS).count(PRIMARY) != 1:
        raise ValueError("V7.5 must contain exactly one primary variant.")
    for fold in FOLDS:
        history = fold.history_frames
        test = fold.test_frames
        if not history or not test or set(history) & set(test):
            raise ValueError(f"Invalid V7.5 fold={fold.name} partition.")
        if max(history) >= min(test):
            raise ValueError(f"V7.5 fold={fold.name} is not chronological.")
        if tuple(range(history[0], history[-1] + 1, 15)) != history:
            raise ValueError(f"V7.5 fold={fold.name} history has gaps.")
        if tuple(range(test[0], test[-1] + 1, 15)) != test:
            raise ValueError(f"V7.5 fold={fold.name} test has gaps.")
        missing = (set(history) | set(test)) - available_frames
        if missing:
            raise ValueError(
                f"V7.5 fold={fold.name} lacks frames={sorted(missing)}."
            )


def run_specs() -> tuple[tuple[ExplicitVariant, str], ...]:
    output: list[tuple[ExplicitVariant, str]] = []
    for variant in VARIANTS:
        if variant.history == "none":
            mode = "none" if variant.region == "none" else "current_only"
            output.append((variant, mode))
        else:
            output.extend((variant, mode) for mode in MEMORY_MODES)
    return tuple(output)


def annotate_decisions(rows: list[dict[str, Any]]) -> None:
    """Annotate matched controls using the camera-centre error.

    Rotation is deliberately frozen in V7.5.  Using the combined pose loss for
    the pass/fail percentages would therefore dilute a centre improvement by a
    constant rotation term.  The combined loss remains in the CSV as a
    secondary metric, while every causal verdict uses ``center_error``.
    """

    groups = {
        (str(row["fold"]), str(row["memory_mode"])): row
        for row in rows
        if row["variant"] == PRIMARY
    }
    fold_region = {mode: {} for mode in MEMORY_MODES}
    fold_history = {mode: {} for mode in MEMORY_MODES}
    for fold in FOLDS:
        raw = _one(rows, fold.name, "raw_streamvggt", "none")
        current = _one(rows, fold.name, CURRENT_ONLY, "current_only")
        raw_loss = float(raw["center_error"])
        current_loss = float(current["center_error"])
        for mode in MEMORY_MODES:
            primary = groups[(fold.name, mode)]
            wrong = _one(rows, fold.name, WRONG_ID, mode)
            oracle = _one(rows, fold.name, ORACLE, mode)
            controls = [
                _one(rows, fold.name, name, mode)
                for name in REGION_CONTROLS
            ]
            primary_loss = float(primary["center_error"])
            wrong_loss = float(wrong["center_error"])
            control_losses = [float(row["center_error"]) for row in controls]
            best_control = min(control_losses)
            worst_control = max(control_losses)
            oracle_loss = float(oracle["center_error"])
            primary["gain_vs_current_only_percent"] = _short(
                _gain(current_loss, primary_loss)
            )
            primary["gain_vs_wrong_id_percent"] = _short(
                _gain(wrong_loss, primary_loss)
            )
            primary["best_region_control_center_error"] = _short(best_control)
            primary["worst_region_control_center_error"] = _short(worst_control)
            primary["gain_vs_best_region_control_percent"] = _short(
                _gain(best_control, primary_loss)
            )
            primary["oracle_gain_vs_raw_percent"] = _short(
                _gain(raw_loss, oracle_loss)
            )
            primary["oracle_solver_pass"] = int(oracle_loss < raw_loss)
            region_pass = int(
                _gain(best_control, primary_loss) >= MINIMUM_GAIN_PERCENT
            )
            history_pass = int(
                _gain(raw_loss, primary_loss) >= MINIMUM_GAIN_PERCENT
                and _gain(current_loss, primary_loss)
                >= MINIMUM_GAIN_PERCENT
                and _gain(wrong_loss, primary_loss)
                >= MINIMUM_GAIN_PERCENT
            )
            primary["sam_region_pass"] = region_pass
            primary["sam_history_pass"] = history_pass
            fold_region[mode][fold.name] = region_pass
            fold_history[mode][fold.name] = history_pass
    all_region_by_mode = {
        mode: int(
            len(fold_region[mode]) == len(FOLDS)
            and all(fold_region[mode].values())
        )
        for mode in MEMORY_MODES
    }
    all_history_by_mode = {
        mode: int(
            len(fold_history[mode]) == len(FOLDS)
            and all(fold_history[mode].values())
        )
        for mode in MEMORY_MODES
    }
    all_region = int(all(all_region_by_mode.values()))
    all_history = int(all(all_history_by_mode.values()))
    for row in rows:
        if row["variant"] == PRIMARY:
            row["all_folds_frozen_region_pass"] = all_region_by_mode["frozen"]
            row["all_folds_online_region_pass"] = all_region_by_mode["online"]
            row["all_folds_region_pass"] = all_region
            row["all_folds_frozen_history_pass"] = all_history_by_mode["frozen"]
            row["all_folds_online_history_pass"] = all_history_by_mode["online"]
            row["all_folds_history_pass"] = all_history


def validate_summary_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Refusing to write an empty V7.5 CSV.")
    expected_rows = len(FOLDS) * len(run_specs())
    if len(rows) != expected_rows:
        raise ValueError(
            f"V7.5 summary requires {expected_rows} rows, got {len(rows)}."
        )
    identities = [
        (str(row["fold"]), str(row["variant"]), str(row["memory_mode"]))
        for row in rows
    ]
    if len(set(identities)) != len(identities):
        raise ValueError("V7.5 summary contains duplicate fold/variant/memory rows.")
    expected = set(SUMMARY_COLUMNS)
    for index, row in enumerate(rows):
        keys = set(row)
        if keys != expected:
            raise ValueError(
                f"V7.5 row={index} schema mismatch: "
                f"missing={sorted(expected - keys)}, "
                f"extra={sorted(keys - expected)}."
            )


def _one(
    rows: list[dict[str, Any]], fold: str, variant: str, mode: str
) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if row["fold"] == fold
        and row["variant"] == variant
        and row["memory_mode"] == mode
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one V7.5 row for {fold}/{variant}/{mode}, "
            f"found {len(matches)}."
        )
    return matches[0]


def _gain(initial: float, final: float) -> float:
    return 0.0 if initial <= 1e-12 else 100.0 * (initial - final) / initial


def _short(value: float) -> str:
    return f"{float(value):.8g}"
