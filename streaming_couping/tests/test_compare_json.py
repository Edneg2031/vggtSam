"""Tests for the JSON leaf differ used to audit stage-2b regeneration."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from streaming_couping.scripts.compare_json import _same, _walk
from streaming_couping.scripts.compare_json import main as compare_main


def _differences(before, after) -> list[tuple[str, object, object]]:
    found: list[tuple[str, object, object]] = []
    _walk(before, after, "", found)
    return found


def test_identical_trees_report_nothing() -> None:
    tree = {"a": 1, "b": {"c": [1, 2, 3]}}
    assert _differences(tree, json.loads(json.dumps(tree))) == []


def test_new_informational_field_is_reported_by_path() -> None:
    before = {"variant": "main", "criteria": {"ate": {"passed": True}}}
    after = {
        "variant": "main",
        "criteria": {"ate": {"passed": True}},
        "new_informational": 0.5,
    }
    found = _differences(before, after)
    assert found == [("new_informational", "<absent>", 0.5)]


def test_flipped_decision_is_reported_with_both_values() -> None:
    before = {"variant": "main", "decision": "OBJECT_FEEDBACK_GO"}
    after = {"variant": "main", "decision": "OBJECT_FEEDBACK_NO_GO"}
    assert _differences(before, after) == [
        ("decision", "OBJECT_FEEDBACK_GO", "OBJECT_FEEDBACK_NO_GO")
    ]


def test_nested_and_list_paths_are_qualified() -> None:
    before = {"criteria": {"ate": {"passed": True}, "ratio": [1.0, 2.0]}}
    after = {"criteria": {"ate": {"passed": False}, "ratio": [1.0, 2.5]}}
    paths = [path for path, _, _ in _differences(before, after)]
    assert paths == ["criteria.ate.passed", "criteria.ratio[1]"]


def test_length_change_reports_the_lengths() -> None:
    found = _differences({"a": [1, 2]}, {"a": [1, 2, 3]})
    assert found == [("a[]", "len=2", "len=3")]


def test_nan_compares_equal_to_nan() -> None:
    assert _same(float("nan"), float("nan"))
    assert not _same(float("nan"), 1.0)
    assert _same({"x": float("nan")}, {"x": float("nan")})


def test_prefix_selects_a_subtree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    before = {"decisions": {"a": 1}, "branches": {"ate": 0.1}}
    after = {"decisions": {"a": 1}, "branches": {"ate": 0.9}}
    left, right = tmp_path / "before.json", tmp_path / "after.json"
    left.write_text(json.dumps(before), encoding="utf-8")
    right.write_text(json.dumps(after), encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv",
        ["compare_json", str(left), str(right), "--prefix", "decisions"],
    )
    compare_main()  # equal subtree -> no SystemExit
    assert "identical" in capsys.readouterr().out

    monkeypatch.setattr(
        "sys.argv",
        ["compare_json", str(left), str(right), "--prefix", "branches"],
    )
    with pytest.raises(SystemExit) as exit_info:
        compare_main()
    assert exit_info.value.code == 1
    output = capsys.readouterr().out
    assert "branches" in output  # the header names the compared subtree
    assert "ate: 0.1 -> 0.9" in output


def test_missing_prefix_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "doc.json"
    path.write_text(json.dumps({"a": 1}), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv", ["compare_json", str(path), str(path), "--prefix", "nope"]
    )
    with pytest.raises(KeyError):
        compare_main()
