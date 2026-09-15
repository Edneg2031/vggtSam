"""Closing the estimation loop one step.

Two ways this experiment can lie, and both are about which trajectory is used
as the reference for "how wrong is this proposal":

* the gauge between the two ways a trajectory is written (the diagnostics'
  camera-to-world against the run's) must be measured on the RAW pair.  Measure
  it against the corrected trajectory and the correction is assumed to be zero,
  which is the thing under test;
* the ground truth a round is scored against must be relative to the base THAT
  round was solved from.  Score round two against the raw base and it is
  answering a different question.
"""

from __future__ import annotations

import math

import pytest
import torch

from streaming_couping.scripts.run_object_pose_loss_reestimate import (
    rebase,
    residual_proposal_error,
    resolve_gauge,
)


def _pose(x: float, y: float = 0.0, z: float = 0.0) -> torch.Tensor:
    pose = torch.eye(4)
    pose[0, 3] = x
    pose[1, 3] = y
    pose[2, 3] = z
    return pose


def test_gauge_is_identity_when_both_writes_agree() -> None:
    poses = torch.stack([_pose(0.0), _pose(1.0), _pose(2.0)])
    gauge = resolve_gauge(poses, poses.clone())
    assert torch.allclose(gauge, torch.eye(4), atol=1e-6)


def test_gauge_is_recovered_when_the_two_writes_differ_by_a_constant() -> None:
    diagnostics = torch.stack([_pose(0.0), _pose(1.0), _pose(2.0)])
    shift = _pose(0.0, 5.0, 0.0)  # a different world origin
    run = torch.stack([shift @ pose for pose in diagnostics])
    gauge = resolve_gauge(diagnostics, run)
    assert torch.allclose(gauge, shift, atol=1e-6)
    # and mapping back undoes it
    recovered = rebase(run, gauge=gauge)
    assert torch.allclose(recovered, diagnostics, atol=1e-5)


def test_gauge_refuses_when_the_pair_is_not_a_single_transform() -> None:
    """A frame-dependent mismatch means these are not two writings of one path."""

    diagnostics = torch.stack([_pose(0.0), _pose(1.0), _pose(2.0)])
    run = torch.stack([_pose(0.0), _pose(9.0), _pose(2.0)])
    with pytest.raises(ValueError, match="SAME trajectory"):
        resolve_gauge(diagnostics, run)


def test_proposal_error_is_scored_against_the_round_s_own_base() -> None:
    """An identity proposal is perfect against the base it was solved from."""

    base = torch.stack([_pose(0.0)])
    gt = torch.stack([_pose(1.0)])
    rows = [{"frame_id": 0, "correction": torch.eye(4)}]
    # solved from `base`, the GT residual is a 1 m move; an identity proposal is
    # 1 m off, not 0
    error = residual_proposal_error(
        rows, gt_c2w=gt, base_c2w=base, frame_ids=(0,)
    )
    assert error["translation_median_m"] == pytest.approx(1.0, abs=1e-5)

    # solved from a base that already IS ground truth, identity is perfect
    at_gt = residual_proposal_error(
        rows, gt_c2w=gt, base_c2w=gt, frame_ids=(0,)
    )
    assert at_gt["translation_median_m"] == pytest.approx(0.0, abs=1e-5)


def test_proposal_error_ignores_rows_that_never_solved() -> None:
    rows = [
        {"frame_id": 0, "correction": None},
        {"frame_id": 0, "correction": torch.eye(4)},
    ]
    error = residual_proposal_error(
        rows, gt_c2w=torch.stack([_pose(0.0)]), base_c2w=torch.stack([_pose(0.0)]), frame_ids=(0,)
    )
    assert error["proposal_count"] == 1


def test_proposal_error_reports_nothing_rather_than_nan_for_an_empty_round() -> None:
    error = residual_proposal_error(
        [], gt_c2w=torch.stack([_pose(0.0)]), base_c2w=torch.stack([_pose(0.0)]), frame_ids=(0,)
    )
    assert error == {"proposal_count": 0}


def test_proposal_rotation_error_uses_atan2_not_acos() -> None:
    """A 90 deg error must read as 90, not as whatever acos of the trace gives."""

    rotation = torch.eye(4)
    rotation[0, 0] = 0.0
    rotation[0, 2] = 1.0
    rotation[2, 0] = -1.0
    rotation[2, 2] = 0.0
    error = residual_proposal_error(
        [{"frame_id": 0, "correction": rotation}],
        gt_c2w=torch.stack([torch.eye(4)]),
        base_c2w=torch.stack([torch.eye(4)]),
        frame_ids=(0,),
    )
    assert error["rotation_median_deg"] == pytest.approx(90.0, abs=1e-4)
