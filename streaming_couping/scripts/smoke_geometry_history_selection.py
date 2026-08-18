#!/usr/bin/env python3
"""CPU smoke checks for fixed geometry-specific history selection."""

from __future__ import annotations

import torch

from streaming_couping.src.geometry_history_selection import (
    GeometryHistoryPolicy,
    select_geometry_history,
    select_recent_history,
)
from streaming_couping.src.qk_pose_retrieval import (
    QKRetrievalPolicy,
    select_qk_history,
)


def main() -> None:
    test_overlap_then_greedy_view_diversity()
    test_overlap_pool_is_fixed_top12()
    test_early_history_and_native_scale_invariance()
    test_recent_control()
    print("Geometry-specific history selection smoke passed")


def test_overlap_then_greedy_view_diversity() -> None:
    poses = _pose_sequence(11, translation_scale=1.0)
    qk_scores = torch.arange(10, dtype=torch.float32)
    policy = GeometryHistoryPolicy(
        total_frame_budget=5,
        anchor_frames=1,
        overlap_pool_size=12,
    )
    geometry = select_geometry_history(
        10,
        qk_scores,
        poses,
        policy=policy,
    )
    qk = select_qk_history(
        10,
        qk_scores,
        policy=QKRetrievalPolicy(total_frame_budget=5, anchor_frames=1),
    )
    assert qk == (0, 6, 7, 8, 9)
    assert geometry.qk_pool_indices == (9, 8, 7, 6, 5, 4, 3, 2, 1)
    assert geometry.selected_indices == (0, 1, 2, 8, 9)
    assert geometry.selected_indices != qk
    assert len(geometry.selected_indices) == 5
    assert geometry.selected_indices[0] == 0
    assert all(index < 10 for index in geometry.selected_indices)
    assert sum(int(row["selected"]) for row in geometry.diagnostics) == 4
    selected_rows = sorted(
        (row for row in geometry.diagnostics if row["selected"]),
        key=lambda row: int(row["greedy_selection_step"]),
    )
    assert [int(row["history_sequence_index"]) for row in selected_rows] == [
        9,
        1,
        8,
        2,
    ]


def test_overlap_pool_is_fixed_top12() -> None:
    poses = _pose_sequence(16, translation_scale=1.0)
    selection = select_geometry_history(
        15,
        torch.arange(15, dtype=torch.float32),
        poses,
        policy=GeometryHistoryPolicy(5, 1, 12),
    )
    assert selection.qk_pool_indices == tuple(range(14, 2, -1))
    first_selected = min(
        (row for row in selection.diagnostics if row["selected"]),
        key=lambda row: int(row["greedy_selection_step"]),
    )
    assert int(first_selected["history_sequence_index"]) == 14


def test_early_history_and_native_scale_invariance() -> None:
    policy = GeometryHistoryPolicy(5, 1, 12)
    poses = _pose_sequence(11, translation_scale=1.0)
    scaled = _pose_sequence(11, translation_scale=17.0)
    early = select_geometry_history(
        3,
        torch.tensor([0.2, 0.8, 0.5]),
        poses,
        policy=policy,
    )
    assert early.selected_indices == (0, 1, 2)
    scores = torch.arange(10, dtype=torch.float32)
    first = select_geometry_history(10, scores, poses, policy=policy)
    second = select_geometry_history(10, scores, scaled, policy=policy)
    assert first.selected_indices == second.selected_indices


def test_recent_control() -> None:
    assert select_recent_history(
        3,
        total_frame_budget=5,
        anchor_frames=1,
    ) == (0, 1, 2)
    assert select_recent_history(
        10,
        total_frame_budget=5,
        anchor_frames=1,
    ) == (0, 6, 7, 8, 9)


def _pose_sequence(count: int, *, translation_scale: float) -> torch.Tensor:
    output = torch.eye(4).repeat(count, 1, 1)
    for index in range(count):
        angle = torch.deg2rad(torch.tensor(float(index)))
        rotation = torch.tensor(
            [
                [torch.cos(angle), 0.0, torch.sin(angle)],
                [0.0, 1.0, 0.0],
                [-torch.sin(angle), 0.0, torch.cos(angle)],
            ]
        )
        center = torch.tensor(
            [float(index) * float(translation_scale), 0.0, 0.0]
        )
        output[index, :3, :3] = rotation
        output[index, :3, 3] = -(rotation @ center)
    return output


if __name__ == "__main__":
    main()
