#!/usr/bin/env python3
"""CPU smoke for the fixed I1 acceptance decision."""

from __future__ import annotations

from streaming_couping.scripts.run_v1_fixed_correspondence_i1 import (
    _i1_decision,
)


def main() -> None:
    branch_rows = [
        _branch("raw", 1.0, 1.0),
        _branch("correct_id", 0.99, 0.95),
        _branch("shuffled_id", 1.0, 1.0),
        _branch("shifted_mask", 1.001, 1.002),
    ]
    track_rows = [
        {"branch": branch, "track_improved": int(branch == "correct_id")}
        for branch in ("raw", "shuffled_id", "shifted_mask")
        for _ in range(2)
    ]
    track_rows.extend(
        {"branch": "correct_id", "track_improved": value}
        for value in (1, 1, 0)
    )
    decision = _i1_decision(
        branch_rows=branch_rows,
        track_rows=track_rows,
        minimum_improved_tracks=2,
    )
    assert decision["i1_pass"] == 1
    assert decision["correct_improved_track_count"] == 2
    assert decision["selected_pointmap_modified"] == 0
    failed_rows = [dict(row) for row in branch_rows]
    failed_rows[1]["global_gain_vs_raw_percent"] = -0.01
    assert _i1_decision(
        branch_rows=failed_rows,
        track_rows=track_rows,
        minimum_improved_tracks=2,
    )["i1_pass"] == 0
    print("V1 fixed correspondence I1 decision smoke passed")


def _branch(branch: str, global_rmse: float, instance_rmse: float) -> dict[str, object]:
    return {
        "branch": branch,
        "global_weighted_rmse": global_rmse,
        "global_gain_vs_raw_percent": 100.0 * (1.0 - global_rmse),
        "instance_weighted_rmse": instance_rmse,
        "instance_gain_vs_raw_percent": 100.0 * (1.0 - instance_rmse),
        "background_exact_raw": 1,
        "bounded_displacement_pass": 1,
    }


if __name__ == "__main__":
    main()
