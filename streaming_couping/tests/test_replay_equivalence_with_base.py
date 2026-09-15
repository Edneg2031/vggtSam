"""The equivalence check when a base trajectory is supplied.

The check has two halves.  The first -- replay against the cached trajectory --
is an integrity check on stage 1 and is unaffected by which base is analysed.
The second asks whether what is being scored against is what the diagnostics
were solved from, and that question changes shape between rounds: in round one
both sides are the raw trajectory, and in round two both sides are the base.

Getting the operands wrong does not crash.  It reports the base difference as
corruption, which fails a run that is correct, or -- worse -- passes one whose
base silently disagrees with its own diagnostics.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from streaming_couping.scripts.run_object_pose_feedback import (
    check_replay_equivalence,
)

TOLERANCE = {"translation_tolerance_m": 1e-3, "rotation_tolerance_deg": 1e-2}


def _trajectory(count: int, drift: float) -> np.ndarray:
    poses = np.tile(np.eye(4), (count, 1, 1))
    for index in range(count):
        poses[index, :3, 3] = [0.1 * index, drift * index * 0.01, 0.0]
    return poses


def _cache_w2c(c2w: np.ndarray) -> np.ndarray:
    return np.stack([np.linalg.inv(pose) for pose in c2w])[:, :3]


def _check(replay, diagnostics, base=None):
    return check_replay_equivalence(
        replay,
        _cache_w2c(replay),
        torch.tensor(diagnostics, dtype=torch.float32),
        supplied_base_c2w=base,
        **TOLERANCE,
    )


def test_without_a_base_it_compares_the_replay_to_the_diagnostics() -> None:
    raw = _trajectory(8, 0.0)
    result = _check(raw, raw)
    assert result["passed"] is True
    assert result["recorded_reference"] == "diagnostics"


def test_with_a_base_it_compares_the_base_to_the_diagnostics() -> None:
    """Round two's replay is still raw; its diagnostics record the base."""

    raw = _trajectory(8, 0.0)
    corrected = _trajectory(8, 0.0)
    corrected[:, 1, 3] = 0.05
    result = _check(raw, corrected, base=corrected)
    assert result["passed"] is True
    assert result["recorded_reference"] == "supplied_base"


def test_a_base_the_diagnostics_were_not_solved_against_is_caught() -> None:
    """The failure this half exists for: scoring against the wrong base."""

    raw = _trajectory(8, 0.0)
    corrected = _trajectory(8, 0.0)
    corrected[:, 1, 3] = 0.05
    result = _check(raw, corrected, base=_trajectory(8, 0.3))
    assert result["passed"] is False
    assert result["max_replay_vs_diagnostics_translation_m"] > 1e-2


def test_stale_chunk_maps_still_fail_with_a_good_base() -> None:
    """The stage-1 half is not weakened by supplying a base."""

    raw = _trajectory(8, 0.0)
    corrupted_replay = _trajectory(8, 0.9)
    result = check_replay_equivalence(
        corrupted_replay,
        _cache_w2c(raw),
        torch.tensor(raw, dtype=torch.float32),
        supplied_base_c2w=raw,
        **TOLERANCE,
    )
    assert result["passed"] is False
    assert result["max_replay_vs_cache_translation_m"] > 1e-2
