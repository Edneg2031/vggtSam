"""Training-free SAM identity guidance for StreamVGGT frame retrieval.

SAM contributes only a causal persistent-instance index.  All descriptors and
all key/value tensors consumed by the geometry model remain native StreamVGGT
states.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


RETRIEVAL_METHODS = (
    "raw_full_history",
    "retrieve_qk",
    "sam_gated_qk",
    "sam_hybrid_qk",
    "shuffled_instance_memory",
)


@dataclass(frozen=True)
class RetrievalPolicy:
    total_frame_budget: int = 5
    anchor_frames: int = 1
    sam_frame_quota: int = 2

    def validate(self) -> None:
        if int(self.total_frame_budget) < 2:
            raise ValueError("Retrieval needs a total frame budget of at least two.")
        if not 1 <= int(self.anchor_frames) < int(self.total_frame_budget):
            raise ValueError("anchor_frames must be in [1, total_frame_budget).")
        remaining = int(self.total_frame_budget) - int(self.anchor_frames)
        if not 1 <= int(self.sam_frame_quota) <= remaining:
            raise ValueError("sam_frame_quota exceeds the non-anchor budget.")


def persistent_visibility(
    masks: torch.Tensor,
    scores: torch.Tensor,
    track_ids: Sequence[int],
    *,
    minimum_score: float,
) -> torch.Tensor:
    """Return causal per-frame persistent-track visibility [T,N]."""

    if masks.ndim != 4:
        raise ValueError(f"Expected masks [T,N,H,W], got {tuple(masks.shape)}.")
    if scores.shape != masks.shape[:2]:
        raise ValueError("Tracking score shape does not match masks.")
    if len(track_ids) != masks.shape[1]:
        raise ValueError("Track-ID count does not match mask slots.")
    registered = torch.tensor(
        [int(value) >= 0 for value in track_ids],
        dtype=torch.bool,
    )
    return (
        masks.bool().flatten(2).any(dim=-1)
        & torch.isfinite(scores)
        & (scores >= float(minimum_score))
        & registered[None]
    )


def same_instance_history_frames(
    visibility: torch.Tensor,
    frame_index: int,
    *,
    shuffled_identity: bool = False,
) -> tuple[int, ...]:
    """Find strictly past frames sharing a current persistent instance."""

    frame = int(frame_index)
    if visibility.ndim != 2 or not 0 <= frame < visibility.shape[0]:
        raise ValueError("Visibility/frame shape is invalid.")
    if frame == 0:
        return ()
    current = visibility[frame].bool()
    history = visibility[:frame].bool()
    if shuffled_identity:
        active_slots = torch.nonzero(
            visibility[: frame + 1].any(dim=0),
            as_tuple=False,
        ).flatten()
        if active_slots.numel() > 1:
            source_slots = torch.roll(active_slots, shifts=1)
            shifted = history.clone()
            shifted[:, active_slots] = history[:, source_slots]
            history = shifted
    shared = (history & current[None]).any(dim=1)
    return tuple(
        int(value)
        for value in torch.nonzero(shared, as_tuple=False).flatten().tolist()
    )


def select_retrieval_history(
    *,
    method: str,
    frame_index: int,
    qk_scores: torch.Tensor,
    sam_region_scores: torch.Tensor,
    shuffled_region_scores: torch.Tensor,
    sam_candidates: Sequence[int],
    shuffled_candidates: Sequence[int],
    policy: RetrievalPolicy,
) -> tuple[int, ...]:
    """Select a time-ordered, strictly causal, fixed-budget frame set."""

    policy.validate()
    method = str(method)
    if method not in RETRIEVAL_METHODS:
        raise ValueError(f"Unknown retrieval method {method!r}.")
    frame = int(frame_index)
    if qk_scores.ndim != 1 or qk_scores.numel() != frame:
        raise ValueError(
            f"Expected {frame} QK scores, got {tuple(qk_scores.shape)}."
        )
    for name, scores in (
        ("sam_region_scores", sam_region_scores),
        ("shuffled_region_scores", shuffled_region_scores),
    ):
        if scores.ndim != 1 or scores.numel() != frame:
            raise ValueError(
                f"Expected {frame} {name}, got {tuple(scores.shape)}."
            )
    if method == "raw_full_history":
        return tuple(range(frame))
    budget = min(int(policy.total_frame_budget), frame)
    if frame <= budget:
        return tuple(range(frame))

    anchors = tuple(range(min(int(policy.anchor_frames), frame)))
    available = tuple(index for index in range(frame) if index not in anchors)
    remaining = budget - len(anchors)
    global_rank = _ranked_indices(qk_scores, available)
    if method == "retrieve_qk":
        return tuple(sorted((*anchors, *global_rank[:remaining])))

    candidates = (
        shuffled_candidates
        if method == "shuffled_instance_memory"
        else sam_candidates
    )
    candidate_scores = (
        shuffled_region_scores
        if method == "shuffled_instance_memory"
        else sam_region_scores
    )
    candidate_pool = tuple(
        index
        for index in candidates
        if index in available and bool(torch.isfinite(candidate_scores[index]))
    )
    candidate_rank = _ranked_indices(candidate_scores, candidate_pool)
    if method == "sam_gated_qk":
        chosen = list(candidate_rank[:remaining])
    else:
        quota = min(int(policy.sam_frame_quota), remaining)
        chosen = list(candidate_rank[:quota])
    for index in global_rank:
        if len(chosen) >= remaining:
            break
        if index not in chosen:
            chosen.append(index)
    return tuple(sorted((*anchors, *chosen)))


def qk_rank_diagnostics(
    qk_scores: torch.Tensor,
    candidates: Sequence[int],
    selected: Sequence[int],
) -> dict[str, object]:
    """Summarize candidate ranks without using pose or pointmap ground truth."""

    ranked = _ranked_indices(qk_scores, range(qk_scores.numel()))
    rank_by_index = {index: rank + 1 for rank, index in enumerate(ranked)}
    candidate_ranks = [
        rank_by_index[int(index)]
        for index in candidates
        if int(index) in rank_by_index
    ]
    selected_candidates = set(int(value) for value in candidates).intersection(
        int(value) for value in selected
    )
    return {
        "sam_candidate_count": len(tuple(candidates)),
        "sam_candidate_best_qk_rank": (
            min(candidate_ranks) if candidate_ranks else -1
        ),
        "sam_candidate_mean_qk_rank": (
            float(sum(candidate_ranks) / len(candidate_ranks))
            if candidate_ranks
            else float("nan")
        ),
        "selected_sam_candidate_count": len(selected_candidates),
        "qk_ranked_history": tuple(ranked),
    }


def _ranked_indices(
    scores: torch.Tensor,
    indices: Sequence[int],
) -> tuple[int, ...]:
    values = tuple(int(index) for index in indices)
    if any(index < 0 or index >= scores.numel() for index in values):
        raise ValueError("A retrieval candidate is outside QK score history.")
    return tuple(
        sorted(
            values,
            key=lambda index: (-float(scores[index]), index),
        )
    )
