"""The pose comparison figure.

Two things have to hold for the figure to mean anything: the error it draws
must be measured against ground truth (a curve against "raw" would show the
correction working even when it moves away from truth), and the rotation angle
must come from atan2 rather than acos of the trace, which is flat near identity
and turns float noise into tenths of a degree.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from streaming_couping.scripts.plot_pose_comparison import (
    load_poses,
    main,
    plot,
    rotation_error_deg,
    translation_error_m,
)


def _poses(count: int, offset: tuple[float, float, float]) -> np.ndarray:
    """A straight path along x, displaced by ``offset`` in every axis."""

    poses = np.tile(np.eye(4), (count, 1, 1))
    poses[:, 0, 3] = np.arange(count, dtype=np.float64) + offset[0]
    poses[:, 1, 3] = offset[1]
    poses[:, 2, 3] = offset[2]
    return poses


def _rotation(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    pose = np.eye(4)
    pose[0, 0] = math.cos(angle)
    pose[0, 2] = math.sin(angle)
    pose[2, 0] = -math.sin(angle)
    pose[2, 2] = math.cos(angle)
    return pose


def test_rotation_error_is_zero_for_an_identical_pose() -> None:
    """acos of the trace reports ~0.03 deg here; that is the bug this avoids."""

    pose = _rotation(0.0).reshape(1, 4, 4)
    assert rotation_error_deg(pose, pose)[0] == pytest.approx(0.0, abs=1e-9)


def test_rotation_error_recovers_a_known_angle() -> None:
    left = _rotation(7.0).reshape(1, 4, 4)
    right = np.tile(np.eye(4), (1, 1, 1))
    assert rotation_error_deg(left, right)[0] == pytest.approx(7.0, abs=1e-6)


def test_translation_error_is_measured_between_positions() -> None:
    """3-4-5, so a wrong axis or a missing component cannot pass by accident."""

    left = _poses(3, (0.0, 0.0, 0.0))
    right = _poses(3, (0.0, 3.0, 4.0))
    assert translation_error_m(left, right) == pytest.approx([5.0, 5.0, 5.0])


def test_load_poses_refuses_a_payload_with_no_ground_truth(tmp_path: Path) -> None:
    path = tmp_path / "poses.pt"
    torch.save({"raw_c2w": torch.eye(4).repeat(2, 1, 1)}, path)
    with pytest.raises(KeyError, match="gt_c2w"):
        load_poses(path)


def test_plot_scores_against_ground_truth_not_against_raw(tmp_path: Path) -> None:
    """A correction that moves AWAY from truth must not look like an improvement."""

    truth = _poses(6, (0.0, 0.0, 0.0))
    raw = _poses(6, (0.0, 0.0, 0.0))
    raw[:, :3, 3] += 0.10  # raw is off by 0.10 m in y
    worse = raw.copy()
    worse[:, :3, 3] += 0.20  # and the "correction" doubles the error

    stats = plot(
        {"gt": truth, "raw": raw, "robust_semantic": worse},
        variant="robust_semantic",
        title="t",
        out_path=tmp_path / "f.png",
    )
    assert stats["translation_improvement_ratio"] < 0.0
    assert (tmp_path / "f.png").is_file()


def test_plot_reports_an_improvement_when_there_is_one(tmp_path: Path) -> None:
    truth = _poses(6, (0.0, 0.0, 0.0))
    raw = _poses(6, (0.0, 0.0, 0.0))
    raw[:, :3, 3] += 0.10
    better = raw.copy()
    better[:, :3, 3] -= 0.05

    stats = plot(
        {"gt": truth, "raw": raw, "robust_semantic": better},
        variant="robust_semantic",
        title="t",
        out_path=tmp_path / "f.png",
    )
    assert stats["translation_improvement_ratio"] == pytest.approx(0.5, abs=1e-6)


def test_cli_names_the_available_variants_when_asked_for_a_missing_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run.baseline"
    feedback = run_dir / "object_pose_feedback"
    feedback.mkdir(parents=True)
    torch.save(
        {
            "raw_c2w": torch.eye(4).repeat(2, 1, 1),
            "gt_c2w": torch.eye(4).repeat(2, 1, 1),
            "variant_c2w": {"robust_semantic": torch.eye(4).repeat(2, 1, 1)},
        },
        feedback / "poses.pt",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "plot",
            "--run-dir", str(run_dir),
            "--variant", "nope",
            "--out", str(tmp_path / "f.png"),
        ],
    )
    with pytest.raises(KeyError, match="robust_semantic"):
        main()
