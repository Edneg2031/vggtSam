from pathlib import Path

from streaming_couping.src.v75_explicit_protocol import (
    EXPECTED_FRAMES,
    FOLDS,
    MEMORY_MODES,
    PRIMARY,
    annotate_decisions,
    run_specs,
    validate_protocol,
)


ROOT = Path(__file__).resolve().parents[2]


def test_v75_protocol_is_chronological_and_complete() -> None:
    validate_protocol(available_frames=set(EXPECTED_FRAMES))
    assert tuple(fold.name for fold in FOLDS) == ("short", "medium", "long")
    for fold in FOLDS:
        assert max(fold.history_frames) < min(fold.test_frames)


def test_v75_primary_runs_frozen_and_online() -> None:
    primary_modes = tuple(
        mode for variant, mode in run_specs() if variant.name == PRIMARY
    )
    assert primary_modes == MEMORY_MODES


def test_v75_decision_requires_region_and_history_controls() -> None:
    losses = {
        "raw_streamvggt": 1.0,
        "sam_current_ray": 0.9,
        "full_image_history": 0.88,
        "sam_bbox_history": 0.87,
        "random_same_area_history": 0.89,
        "sam_stale_time_history": 0.86,
        "sam_history_correct": 0.7,
        "sam_history_wrong_id": 0.95,
        "gt_mask_oracle_history": 0.6,
    }
    rows = []
    for fold in FOLDS:
        for variant, mode in run_specs():
            rows.append(
                {
                    "fold": fold.name,
                    "variant": variant.name,
                    "memory_mode": mode,
                    # Keep this unrelated to the decision values: V7.5 only
                    # changes center and must decide on center_error.
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
    primary = [row for row in rows if row["variant"] == PRIMARY]
    assert all(int(row["sam_region_pass"]) == 1 for row in primary)
    assert all(int(row["sam_history_pass"]) == 1 for row in primary)
    assert all(
        int(row["all_folds_frozen_history_pass"]) == 1 for row in primary
    )
    assert all(
        int(row["all_folds_online_history_pass"]) == 1 for row in primary
    )
    assert all(int(row["all_folds_history_pass"]) == 1 for row in primary)


def test_v75_command_is_standalone_and_no_training() -> None:
    command = (
        ROOT / "streaming_couping/commands_v75_explicit_pose_causality.txt"
    ).read_text(encoding="utf8")
    assert "run_v75_explicit_pose_causality" in command
    assert "run_v74_temporal_scaling" not in command
    assert "--stage train" not in command
    assert "v75_explicit_pose_causality.csv" in command
    assert "v75_explicit_pose_frames.csv" in command


def test_v75_strict_verdict_requires_frozen_and_online() -> None:
    losses = {
        "raw_streamvggt": 1.0,
        "sam_current_ray": 0.9,
        "full_image_history": 0.88,
        "sam_bbox_history": 0.87,
        "random_same_area_history": 0.89,
        "sam_stale_time_history": 0.86,
        "sam_history_correct": 0.70,
        "sam_history_wrong_id": 0.95,
        "gt_mask_oracle_history": 0.60,
    }
    rows = []
    for fold in FOLDS:
        for variant, mode in run_specs():
            center_error = losses[variant.name]
            if (
                fold.name == "short"
                and variant.name == PRIMARY
                and mode == "frozen"
            ):
                center_error = 1.05
            rows.append(
                {
                    "fold": fold.name,
                    "variant": variant.name,
                    "memory_mode": mode,
                    "center_error": center_error,
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
    marker = next(row for row in rows if row["variant"] == PRIMARY)
    assert int(marker["all_folds_frozen_history_pass"]) == 0
    assert int(marker["all_folds_online_history_pass"]) == 1
    assert int(marker["all_folds_history_pass"]) == 0
