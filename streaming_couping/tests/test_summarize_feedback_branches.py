"""Tests for the cross-branch comparison table."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from streaming_couping.scripts.summarize_feedback_branches import (
    HEADERS,
    branch_row,
    main,
    noise_floor,
    render_noise_floor,
)


def _write_branch(
    root: Path,
    name: str,
    *,
    proposal_mode: str = "joint",
    max_age: int = 0,
    refresh: int = 0,
    proposals: int = 229,
    accepted: int = 13,
    raw_ate: float = 0.1166,
    variant_ate: float = 0.1081,
    raw_sim3: float = 0.0551,
    variant_sim3: float = 0.0554,
    decision: str = "OBJECT_FEEDBACK_GO",
    anchor_rho: float | None = 0.7848,
    anchor_age: float | None = 37.5,
    proposal_error: float = 0.0856,
    future_rotation: float = -0.0105,
    with_attribution: bool = True,
) -> Path:
    run_dir = root / name
    feedback = run_dir / "object_pose_feedback"
    feedback.mkdir(parents=True, exist_ok=True)
    summary = {
        "main_variant": "robust_semantic_geometric",
        "proposal_count": proposals,
        "refiner_settings_audit": {
            "exported_by_stage_2a": {
                "proposal_mode": proposal_mode,
                "max_reference_age_frames": max_age,
                "anchor_refresh_interval_frames": refresh,
            }
        },
        "branches": {
            "raw": {"ate_rmse_m": raw_ate, "ate_rmse_sim3_m": raw_sim3},
            "robust_semantic_geometric": {
                "ate_rmse_m": variant_ate,
                "ate_rmse_sim3_m": variant_sim3,
            },
        },
        "gate_stats": {
            "robust_semantic_geometric": {
                "accepted_frame_count": accepted,
                "reject_reason_counts": {"insufficient_objects": 45},
            }
        },
        "decisions": {
            "robust_semantic_geometric": {
                "decision": decision,
                "future_rotation_gain_median_deg": future_rotation,
            }
        },
        "answers": {
            "q2_single_proposal_error": {"translation_median_m": proposal_error},
            "q3_consensus_vs_single_object": {
                "main_variant_consensus_translation_median_m": 0.0424,
            },
        },
    }
    (feedback / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    if with_attribution:
        attribution = {
            "reference_staleness": {
                "anchor_reference_age": {
                    "spearman_rho_within_category": anchor_rho,
                    "permutation_p_within_category": 0.0002,
                    "median_age_frames": anchor_age,
                },
                "oldest_reference_age": {
                    "spearman_rho_within_category": 0.75,
                    "median_age_frames": 20.0,
                },
            }
        }
        (feedback / "attribution.json").write_text(
            json.dumps(attribution), encoding="utf-8"
        )
    return run_dir


def test_row_reports_the_knobs_and_the_improvements(tmp_path: Path) -> None:
    run_dir = _write_branch(
        tmp_path,
        "run.fresh",
        proposal_mode="joint",
        max_age=15,
        refresh=20,
        raw_ate=0.10,
        variant_ate=0.08,
    )
    row, payload = branch_row(run_dir)
    assert len(row) == len(HEADERS)
    assert payload["proposal_mode"] == "joint"
    assert payload["max_reference_age_frames"] == 15
    assert payload["anchor_refresh_interval_frames"] == 20
    assert payload["direct_ate_improvement_ratio"] == pytest.approx(0.20)
    assert payload["anchor_reference_age_rho_within_category"] == pytest.approx(0.7848)


def test_sim3_improvement_is_reported_separately(tmp_path: Path) -> None:
    """The direct/sim3 split is what caught the main variant being a gauge fix."""

    run_dir = _write_branch(
        tmp_path,
        "run.gauge",
        raw_ate=0.1166,
        variant_ate=0.1105,
        raw_sim3=0.0551,
        variant_sim3=0.0554,
    )
    _, payload = branch_row(run_dir)
    assert payload["direct_ate_improvement_ratio"] > 0
    assert payload["sim3_ate_improvement_ratio"] < 0


def test_freshness_column_is_blank_when_both_knobs_are_off(tmp_path: Path) -> None:
    off = _write_branch(tmp_path, "a")
    on = _write_branch(tmp_path, "b", max_age=15, refresh=20)
    assert branch_row(off)[0][HEADERS.index("fresh")] == "-"
    assert branch_row(on)[0][HEADERS.index("fresh")] == "15/20"


def test_proposal_error_and_rotation_gain_are_surfaced(tmp_path: Path) -> None:
    """These two are the targets of this round's change; both must be visible."""

    run_dir = _write_branch(
        tmp_path, "run.fresh", proposal_error=0.061, future_rotation=0.052
    )
    row, payload = branch_row(run_dir)
    assert payload["proposal_translation_error_median_m"] == pytest.approx(0.061)
    assert payload["future_rotation_gain_median_deg"] == pytest.approx(0.052)
    assert row[HEADERS.index("prop_err")] == pytest.approx(0.061)
    assert row[HEADERS.index("fut_rot")] == pytest.approx(0.052)


def test_anchor_age_is_reported_from_the_attribution(tmp_path: Path) -> None:
    run_dir = _write_branch(tmp_path, "run.fresh", anchor_age=7.5)
    row, payload = branch_row(run_dir)
    assert payload["anchor_reference_age_median_frames"] == pytest.approx(7.5)
    assert row[HEADERS.index("anchor_age")] == pytest.approx(7.5)


def test_missing_summary_is_reported_not_crashed(tmp_path: Path) -> None:
    row, payload = branch_row(tmp_path / "nope")
    assert payload["available"] is False
    assert row[1] == "no summary.json"
    assert len(row) == len(HEADERS)


def test_missing_attribution_leaves_the_staleness_cell_blank(tmp_path: Path) -> None:
    run_dir = _write_branch(tmp_path, "run.nostats", with_attribution=False)
    row, payload = branch_row(run_dir)
    assert payload["available"] is True
    assert payload["anchor_reference_age_rho_within_category"] is None
    assert row[HEADERS.index("anchor_rho")] is None


def test_decisions_are_shortened_for_the_table(tmp_path: Path) -> None:
    go = _write_branch(tmp_path, "a", decision="OBJECT_FEEDBACK_GO")
    no_go = _write_branch(tmp_path, "b", decision="OBJECT_FEEDBACK_NO_GO")
    assert branch_row(go)[0][HEADERS.index("decision")] == "GO"
    assert branch_row(no_go)[0][HEADERS.index("decision")] == "NO_GO"


def test_cli_prints_every_branch_in_order_and_writes_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    baseline = _write_branch(tmp_path, "run.baseline", proposals=229)
    fresh = _write_branch(
        tmp_path,
        "run.fresh",
        max_age=15,
        refresh=20,
        proposals=210,
        anchor_rho=0.21,
    )
    out = tmp_path / "branches.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "summarize",
            "--run-dir",
            str(baseline),
            "--run-dir",
            str(fresh),
            "--json-out",
            str(out),
        ],
    )
    main()
    printed = capsys.readouterr().out
    assert "run.baseline" in printed
    assert "run.fresh" in printed
    assert printed.index("run.baseline") < printed.index("run.fresh")
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert [entry["run_dir"].split("/")[-1] for entry in payload["branches"]] == [
        "run.baseline",
        "run.fresh",
    ]
    assert payload["branches"][1]["proposal_count"] == 210


# ---------------------------------------------------------------------------
# Noise floor.
# ---------------------------------------------------------------------------


def test_noise_floor_reports_the_spread_across_repeats(tmp_path: Path) -> None:
    first = _write_branch(
        tmp_path,
        "r1",
        proposals=229,
        variant_ate=0.1081,
        variant_sim3=0.0554,
    )
    second = _write_branch(
        tmp_path,
        "r2",
        proposals=241,
        variant_ate=0.1042,
        variant_sim3=0.0531,
    )
    floor = noise_floor([first, second])
    assert floor["run_count"] == 2
    assert floor["metrics"]["proposal_count"]["spread"] == 12
    # the ATE improvement moved too, and that is the point of measuring it
    assert floor["metrics"]["direct_ate_improvement_ratio"]["spread"] > 0
    rendered = render_noise_floor(floor)
    assert "noise floor from 2 runs" in rendered
    assert "direct_ate_improvement_ratio" in rendered


def test_noise_floor_says_so_when_there_is_only_one_run(tmp_path: Path) -> None:
    only = _write_branch(tmp_path, "r1")
    floor = noise_floor([only])
    assert floor["run_count"] == 1
    assert "metrics" not in floor or not floor["metrics"]
    assert "not measured" in render_noise_floor(floor)


def test_noise_floor_ignores_unavailable_runs(tmp_path: Path) -> None:
    good = _write_branch(tmp_path, "r1", proposals=229)
    floor = noise_floor([good, tmp_path / "missing"])
    assert floor["run_count"] == 1
    assert not floor.get("metrics")


def test_cli_prints_the_noise_floor_before_the_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    first = _write_branch(tmp_path, "run.baseline", proposals=229)
    repeat = _write_branch(tmp_path, "run.repeat", proposals=241)
    fresh = _write_branch(tmp_path, "run.fresh", max_age=15, proposals=210)
    monkeypatch.setattr(
        "sys.argv",
        [
            "summarize",
            "--run-dir",
            str(first),
            "--run-dir",
            str(fresh),
            "--noise-floor",
            str(first),
            "--noise-floor",
            str(repeat),
        ],
    )
    main()
    printed = capsys.readouterr().out
    assert "noise floor from 2 runs" in printed
    assert printed.index("noise floor") < printed.index("object pose feedback branches")
