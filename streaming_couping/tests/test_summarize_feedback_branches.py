"""Tests for the cross-branch comparison table."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from streaming_couping.scripts.summarize_feedback_branches import (
    HEADERS,
    branch_row,
    main,
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
            "robust_semantic_geometric": {"decision": decision},
        },
    }
    (feedback / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    if with_attribution:
        attribution = {
            "reference_staleness": {
                "anchor_reference_age": {
                    "spearman_rho_within_category": anchor_rho,
                    "permutation_p_within_category": 0.0002,
                    "median_age_frames": 37.5,
                }
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
    assert [entry["run_dir"].split("/")[-1] for entry in payload] == [
        "run.baseline",
        "run.fresh",
    ]
    assert payload[1]["proposal_count"] == 210
