"""Reading back what each prompt contributed.

The trap this guards is reporting an empty result for a run that simply
predates the ledger: "no prompts were dropped" and "this run cannot say" are
opposite conclusions drawn from the same empty table.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from streaming_couping.scripts.show_sam3_candidate_ledger import (
    load_ledger,
    main,
    outcome_matrix,
)


def _write_run(root: Path, name: str, metadata: dict | None) -> Path:
    run_dir = root / name
    branch = run_dir / "raw_pose"
    branch.mkdir(parents=True, exist_ok=True)
    payload: dict = {"semantic_rgb": torch.zeros(1)}
    if metadata is not None:
        payload["metadata"] = metadata
    torch.save(payload, branch / "semantic_map.pt")
    return run_dir


LEDGER_SUMMARY = {
    "total": 4,
    "prompt_count": 3,
    "per_prompt": {
        "bed": {"accepted": 1},
        "table": {"birth_mask_rejected": 2},
        "rug": {},
    },
    "prompts_with_no_track": ["rug", "table"],
    "prompts_returning_nothing": ["rug"],
    "prompts_found_but_unusable": ["table"],
}


def test_load_ledger_returns_none_for_a_run_that_predates_it(tmp_path: Path) -> None:
    run_dir = _write_run(tmp_path, "old.baseline", {"prompts": ["bed"]})
    assert load_ledger(run_dir / "raw_pose" / "semantic_map.pt") is None


def test_load_ledger_reads_what_the_adapter_exported(tmp_path: Path) -> None:
    run_dir = _write_run(
        tmp_path,
        "new.baseline",
        {
            "prompts": ["bed", "table", "rug"],
            "sam3_candidate_ledger": [{"prompt": "bed", "outcome": "accepted"}],
            "sam3_candidate_ledger_summary": LEDGER_SUMMARY,
        },
    )
    loaded = load_ledger(run_dir / "raw_pose" / "semantic_map.pt")
    assert loaded is not None
    assert loaded["prompts"] == ["bed", "table", "rug"]
    assert loaded["summary"]["total"] == 4


def test_outcome_matrix_keeps_prompts_in_request_order() -> None:
    _, outcomes, rows = outcome_matrix(
        {**LEDGER_SUMMARY, "prompts": ["bed", "table", "rug"]}
    )
    assert [row[0] for row in rows] == ["bed", "table", "rug"]
    assert "accepted" in outcomes and "birth_mask_rejected" in outcomes
    # rug was requested and returned nothing, so it is a row of zeros -- not a
    # missing row, which would mean it was never asked for
    rug = rows[[row[0] for row in rows].index("rug")]
    assert sum(rug[1:]) == 0


def test_outcome_matrix_adds_prompts_seen_only_in_the_ledger() -> None:
    """A ledger naming a prompt absent from the request list must still show."""

    _, _, rows = outcome_matrix(
        {
            "per_prompt": {"bed": {"accepted": 1}, "ghost": {"accepted": 1}},
            "prompts": ["bed"],
        }
    )
    assert [row[0] for row in rows] == ["bed", "ghost"]


def test_cli_says_a_run_predates_the_ledger_instead_of_reporting_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = _write_run(tmp_path, "base_v1.baseline", {"prompts": ["bed"]})
    monkeypatch.setattr("sys.argv", ["show", "--run-dir", str(old)])
    main()
    printed = capsys.readouterr().out
    assert "predates the ledger" in printed
    # and it must not look like an empty result
    assert "0 track" not in printed


def test_cli_prints_the_outcome_table(
    tmp_path: Path, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _write_run(
        tmp_path,
        "base_v3.baseline",
        {
            "prompts": ["bed", "table", "rug"],
            "sam3_candidate_ledger": [{"prompt": "bed", "outcome": "accepted"}],
            "sam3_candidate_ledger_summary": {
                "total": 6,
                "prompt_count": 3,
                "per_prompt": {
                    "bed": {"accepted": 1, "duplicate": 1},
                    "table": {"birth_mask_rejected": 2},
                    "rug": {"over_object_cap": 2},
                },
                "prompts_with_no_track": ["table"],
                "prompts_returning_nothing": [],
                "prompts_found_but_unusable": ["table"],
            },
        },
    )
    monkeypatch.setattr("sys.argv", ["show", "--run-dir", str(run)])
    main()
    printed = capsys.readouterr().out
    # table has a birth_mask_rejected row, so it WAS found -- the distinction
    # from "returned nothing" is the point of the message
    assert "found but unusable" in printed
    assert "table" in printed
    # each outcome that actually occurred is explained, because the codes alone
    # do not say what to do about them
    assert "overlaps a track that was born earlier" in printed
    assert "dropped: behind the --max-objects cut" in printed
    assert "dropped: birth mask below the pixel floor" in printed


def test_digest_line_uses_the_shared_short_label() -> None:
    """The whole run-directory name would push the line off a terminal."""

    from streaming_couping.scripts.show_sam3_candidate_ledger import digest_line

    line = digest_line(
        "semantic_map_100frames_horizonstream_object_pose_feedback_90_189_v2.baseline",
        {
            "total": 3,
            "prompts": ["bed", "rug"],
            "per_prompt": {"bed": {"accepted": 1}, "rug": {}},
            "prompts_with_no_track": ["rug"],
        },
        None,
    )
    assert "v2/baseline" in line
    assert "semantic_map_100frames" not in line
    assert "no track: rug" in line


def test_digest_line_for_a_run_that_cannot_answer_says_so() -> None:
    from streaming_couping.scripts.show_sam3_candidate_ledger import digest_line

    line = digest_line("prefix_v1.baseline", None, "predates the ledger")
    assert "v1/baseline" in line
    assert "predates the ledger" in line
    # a run that cannot answer must not read as one that found nothing
    assert "0 tracks" not in line


def test_report_out_sends_the_tables_to_a_file(
    tmp_path: Path, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _write_run(
        tmp_path,
        "base_v3.baseline",
        {
            "prompts": ["bed", "table"],
            "sam3_candidate_ledger": [{"prompt": "bed", "outcome": "accepted"}],
            "sam3_candidate_ledger_summary": {
                "total": 2,
                "prompt_count": 2,
                "per_prompt": {"bed": {"accepted": 1}, "table": {"birth_mask_rejected": 1}},
                "prompts_with_no_track": ["table"],
                "prompts_returning_nothing": [],
                "prompts_found_but_unusable": ["table"],
            },
        },
    )
    report = tmp_path / "out" / "report.txt"
    monkeypatch.setattr("sys.argv", ["show", "--run-dir", str(run), "--report-out", str(report)])
    main()
    printed = capsys.readouterr().out
    body = report.read_text(encoding="utf-8")
    # the dashed rule only ever comes from a rendered table
    assert "--------" not in printed
    assert "--------" in body
    assert "v3/baseline" in printed
    assert "birth_mask_rejected" in body
    assert "dropped: birth mask below the pixel floor" in body
    # stdout stays a headline: one line per run plus the paths
    assert len([line for line in printed.splitlines() if line.strip()]) <= 4


def test_a_prompt_that_returned_nothing_is_not_called_unusable() -> None:
    """The two failures have opposite fixes and must not share a label.

    Absent from the ledger means the text grounding found no object -- replace
    the word.  Present with outcomes but none usable means the object WAS found
    and every track died on the way in -- look at the thresholds instead.
    """

    from streaming_couping.src.semantic_mapping.adapters import ledger_summary

    summary = ledger_summary(
        [{"prompt": "chair", "outcome": "birth_mask_rejected"}],
        ["chair", "table"],
    )
    assert summary["prompts_returning_nothing"] == ["table"]
    assert summary["prompts_found_but_unusable"] == ["chair"]
    # the merged list still exists for callers that only want "not usable"
    assert summary["prompts_with_no_track"] == ["chair", "table"]
