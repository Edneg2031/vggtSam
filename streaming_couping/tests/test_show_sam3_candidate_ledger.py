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
    assert "predates the candidate ledger" in printed


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
            },
        },
    )
    monkeypatch.setattr("sys.argv", ["show", "--run-dir", str(run)])
    main()
    printed = capsys.readouterr().out
    assert "returned no track at all: table" in printed
    # each outcome that actually occurred is explained, because the codes alone
    # do not say what to do about them
    assert "overlaps a track that was born earlier" in printed
    assert "dropped: behind the --max-objects cut" in printed
    assert "dropped: birth mask below the pixel floor" in printed
