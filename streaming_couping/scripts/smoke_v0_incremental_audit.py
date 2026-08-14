#!/usr/bin/env python3
"""CPU smoke for V0 context controls and pose/pointmap diagnostics."""

from __future__ import annotations

import torch

from streaming_couping.src.context_policy_audit import (
    fixed_alignment_pointmap_rows,
    pose_pointmap_consistency_rows,
    select_random_history,
    select_recent_history,
    summarize_consistency_rows,
    summarize_pointmap_rows,
)


def main() -> None:
    _test_history_controls()
    _test_pose_pointmap_consistency()
    _test_fixed_alignment_geometry_score()
    print("V0 incremental context/consistency audit smoke passed")


def _test_history_controls() -> None:
    assert select_recent_history(
        0, total_history_budget=5, anchor_frames=1
    ) == ()
    assert select_recent_history(
        4, total_history_budget=5, anchor_frames=1
    ) == (0, 1, 2, 3)
    assert select_recent_history(
        8, total_history_budget=5, anchor_frames=1
    ) == (0, 4, 5, 6, 7)
    first = select_random_history(
        12,
        total_history_budget=5,
        anchor_frames=1,
        seed=0,
    )
    repeated = select_random_history(
        12,
        total_history_budget=5,
        anchor_frames=1,
        seed=0,
    )
    other = select_random_history(
        12,
        total_history_budget=5,
        anchor_frames=1,
        seed=1,
    )
    assert first == repeated
    assert first != other
    assert len(first) == 5 and first[0] == 0
    assert all(0 <= index < 12 for index in first)


def _test_pose_pointmap_consistency() -> None:
    y, x = torch.meshgrid(torch.arange(3), torch.arange(4), indexing="ij")
    points = torch.stack(
        [x.float(), y.float(), torch.ones_like(x).float()], dim=-1
    )[None]
    confidence = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4, 1)
    depth = torch.ones(1, 3, 4, 1)
    intrinsics = torch.eye(3)[None]
    raw_pose = torch.eye(4)[None, None, :3]
    shifted_pose = raw_pose.clone()
    shifted_pose[0, 0, 0, 3] = 0.25
    raw_rows = pose_pointmap_consistency_rows(
        branch="raw_pose",
        frame_indices=(90,),
        world_points=points,
        world_confidence=confidence,
        depth=depth,
        depth_confidence=confidence,
        intrinsics=intrinsics,
        world_to_camera=raw_pose,
        confidence_threshold=0.0,
    )
    shifted_rows = pose_pointmap_consistency_rows(
        branch="qk_pose",
        frame_indices=(90,),
        world_points=points,
        world_confidence=confidence,
        depth=depth,
        depth_confidence=confidence,
        intrinsics=intrinsics,
        world_to_camera=shifted_pose,
        confidence_threshold=0.0,
    )
    raw = summarize_consistency_rows(raw_rows)
    shifted = summarize_consistency_rows(shifted_rows)
    assert raw["mean_frame_reprojection_median_px"] < 1e-6
    assert shifted["mean_frame_reprojection_median_px"] > 0.2
    assert raw["mean_frame_relative_depth_median"] < 1e-6


def _test_fixed_alignment_geometry_score() -> None:
    target = torch.tensor(
        [
            [
                [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
                [[0.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            ],
            [
                [[0.0, 0.0, 2.0], [1.0, 0.0, 2.0]],
                [[0.0, 1.0, 2.0], [1.0, 1.0, 2.0]],
            ],
        ]
    )
    raw = target + 0.10
    candidate = target + 0.02
    confidence = torch.arange(8, dtype=torch.float32).reshape(2, 2, 2, 1)
    rows = fixed_alignment_pointmap_rows(
        method="candidate",
        frame_indices=(90, 105),
        reference_index=0,
        pointmap=candidate,
        raw_confidence=confidence,
        raw_world_points=raw,
        target_world_points=target,
        scale=1.0,
        rotation=torch.eye(3),
        translation=torch.zeros(3),
        confidence_threshold=0.0,
    )
    summary = summarize_pointmap_rows(rows)
    assert summary["evaluated_frames"] == 1
    assert summary["mean_frame_paired_weighted_rmse"] < 0.04


if __name__ == "__main__":
    main()
