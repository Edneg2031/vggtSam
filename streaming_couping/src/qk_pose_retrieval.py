"""Minimal training-free native-QK history selection used by V0 pose."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class QKRetrievalPolicy:
    total_frame_budget: int = 5
    anchor_frames: int = 1

    def validate(self) -> None:
        if int(self.total_frame_budget) < 2:
            raise ValueError("QK retrieval needs at least two total frames.")
        if not 1 <= int(self.anchor_frames) < int(self.total_frame_budget):
            raise ValueError("anchor_frames must be in [1, total_frame_budget).")


def select_qk_history(
    frame_index: int,
    qk_scores: torch.Tensor,
    *,
    policy: QKRetrievalPolicy,
) -> tuple[int, ...]:
    """Keep fixed early anchors and fill the budget by native QK score."""

    policy.validate()
    frame = int(frame_index)
    if qk_scores.ndim != 1 or qk_scores.numel() != frame:
        raise ValueError(
            f"Expected {frame} QK history scores, got {tuple(qk_scores.shape)}."
        )
    budget = min(int(policy.total_frame_budget), frame)
    if frame <= budget:
        return tuple(range(frame))
    anchors = tuple(range(min(int(policy.anchor_frames), frame)))
    available = tuple(index for index in range(frame) if index not in anchors)
    remaining = budget - len(anchors)
    ranked = rank_qk_history(qk_scores, indices=available)
    return tuple(sorted((*anchors, *ranked[:remaining])))


def rank_qk_history(
    qk_scores: torch.Tensor,
    *,
    indices: Sequence[int] | None = None,
) -> tuple[int, ...]:
    if qk_scores.ndim != 1:
        raise ValueError("QK scores must be one-dimensional.")
    values = tuple(
        range(qk_scores.numel()) if indices is None else (int(v) for v in indices)
    )
    if any(index < 0 or index >= qk_scores.numel() for index in values):
        raise ValueError("A QK retrieval candidate is outside history.")
    return tuple(
        sorted(values, key=lambda index: (-float(qk_scores[index]), index))
    )
