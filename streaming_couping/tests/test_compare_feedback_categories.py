"""The category comparison: what a prompt change swapped, and whether it helped.

The branch table can say a generation got worse.  It cannot say that the
segmenter stopped tracking an object it used to track, which is the thing a
prompt-set change actually does -- so the comparison has to name the categories
that only one run has, and has to separate "produced proposals" from "entered
the consensus".
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from streaming_couping.scripts.compare_feedback_categories import (
    category_errors,
    consensus_frames,
    consensus_net_effect,
    consensus_participation,
    load_run,
    main,
)

PROPOSAL_HEADER = (
    "frame,instance_id,category,gt_translation_correction_error,"
    "consensus_inlier\n"
)
CONSENSUS_HEADER = "variant,frame,num_reliable,inlier_count\n"


def _write_run(
    root: Path,
    name: str,
    *,
    proposals: list[tuple[int, int, str, float | None, bool]],
    consensus: list[tuple[int, int, int]] | None = None,
    variant: str = "robust_semantic",
    proposal_error: float = 0.09,
    consensus_error: float = 0.07,
) -> Path:
    run_dir = root / name
    feedback = run_dir / "object_pose_feedback"
    feedback.mkdir(parents=True, exist_ok=True)
    lines = [PROPOSAL_HEADER]
    for frame, instance, category, error, inlier in proposals:
        lines.append(
            f"{frame},{instance},{category},"
            f"{'' if error is None else error},{inlier}\n"
        )
    (feedback / "object_proposals.csv").write_text("".join(lines), encoding="utf-8")

    rows = consensus if consensus is not None else [(0, 3, 3)]
    body = [CONSENSUS_HEADER]
    for frame, reliable, inliers in rows:
        body.append(f"{variant},{frame},{reliable},{inliers}\n")
    (feedback / "consensus_metrics.csv").write_text("".join(body), encoding="utf-8")

    (feedback / "summary.json").write_text(
        json.dumps(
            {
                "main_variant": variant,
                "answers": {
                    "q2_single_proposal_error": {
                        "translation_median_m": proposal_error
                    },
                    "q3_consensus_vs_single_object": {
                        "main_variant_consensus_translation_median_m": consensus_error
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return run_dir


# ---------------------------------------------------------------------------
# Units.
# ---------------------------------------------------------------------------


def test_category_errors_counts_proposals_even_when_unscored() -> None:
    rows = [
        {"category": "bed", "gt_translation_correction_error": "0.10"},
        {"category": "bed", "gt_translation_correction_error": "0.20"},
        {"category": "bed", "gt_translation_correction_error": ""},
    ]
    entry = category_errors(rows)["bed"]
    assert entry["proposals"] == 3
    assert entry["scored"] == 2
    assert entry["median_m"] == pytest.approx(0.15)


def test_participation_separates_proposals_from_inliers() -> None:
    """A category can propose on every frame and never once be an inlier."""

    rows = [
        {
            "category": "bed",
            "frame": "0",
            "consensus_inlier": "True",
        },
        {
            "category": "bed",
            "frame": "1",
            "consensus_inlier": "True",
        },
        {
            "category": "rug",
            "frame": "0",
            "consensus_inlier": "False",
        },
    ]
    participation = consensus_participation(rows)
    assert participation["bed"]["inlier_frames"] == 2
    assert participation["bed"]["vote_share"] == pytest.approx(1.0)
    assert "rug" not in participation


def test_consensus_frames_reads_only_the_named_variant() -> None:
    rows = [
        {"variant": "robust_semantic", "frame": "0", "num_reliable": "5", "inlier_count": "5"},
        {"variant": "robust_semantic", "frame": "1", "num_reliable": "3", "inlier_count": "1"},
        {"variant": "single", "frame": "0", "num_reliable": "9", "inlier_count": "9"},
    ]
    frames = consensus_frames(rows, "robust_semantic")
    assert frames["frames"] == 2
    assert frames["inlier_median"] == pytest.approx(3.0)
    assert frames["reliable_median"] == pytest.approx(4.0)


def test_net_effect_is_negative_when_the_consensus_loses_to_one_proposal() -> None:
    assert consensus_net_effect(0.0888, 0.0923) == pytest.approx(-0.0394, abs=1e-4)
    assert consensus_net_effect(0.0868, 0.0730) == pytest.approx(0.1589, abs=1e-4)
    # no baseline to improve on, and a zero baseline, are both unreadable
    assert consensus_net_effect(None, 0.07) is None
    assert consensus_net_effect(0.0, 0.07) is None


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def test_cli_names_categories_only_one_run_has(
    tmp_path: Path, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The v1-v2 shape: a category the wider prompt set lost, and one it gained."""

    old = _write_run(
        tmp_path,
        "base_v1.baseline",
        proposals=[
            (0, 0, "bed", 0.10, True),
            (0, 1, "wardrobe", 0.09, True),
        ],
    )
    new = _write_run(
        tmp_path,
        "base_v2.baseline",
        proposals=[
            (0, 0, "bed", 0.12, True),
            (0, 1, "cabinet", 0.08, True),
        ],
    )
    monkeypatch.setattr(
        "sys.argv",
        ["compare", "--run-dir", str(old), "--run-dir", str(new)],
    )
    main()
    printed = capsys.readouterr().out
    assert "v1/baseline" in printed
    assert "v2/baseline" in printed
    assert "'wardrobe': present in v1/baseline; absent from v2/baseline" in printed
    assert "'cabinet': present in v2/baseline; absent from v1/baseline" in printed


def test_load_run_reports_the_consensus_losing_to_a_single_proposal(
    tmp_path: Path,
) -> None:
    run_dir = _write_run(
        tmp_path,
        "base_v2.baseline",
        proposals=[(0, 0, "bed", 0.10, True)],
        proposal_error=0.0888,
        consensus_error=0.0923,
    )
    run = load_run(run_dir, None)
    assert run["label"] == "v2/baseline"
    assert run["variant"] == "robust_semantic"
    assert run["net_effect"] < 0.0


def test_load_run_rejects_a_renamed_column(tmp_path: Path) -> None:
    """A renamed writer column must fail loudly, not report an empty section."""

    run_dir = _write_run(
        tmp_path, "base_v2.baseline", proposals=[(0, 0, "bed", 0.10, True)]
    )
    feedback = run_dir / "object_pose_feedback"
    (feedback / "object_proposals.csv").write_text(
        "frame,instance_id,label,gt_translation_correction_error,consensus_inlier\n"
        "0,0,bed,0.10,True\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="category"):
        load_run(run_dir, None)


def test_load_run_rejects_a_missing_artifact(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="object_proposals.csv"):
        load_run(tmp_path / "nope", None)
