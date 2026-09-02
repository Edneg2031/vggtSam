"""Causal world-space support for semantic-map voxel writes.

This module is deliberately downstream of both model providers.  It does not
change a geometry prediction or a SAM mask/track; it only decides which points
from an already accepted object observation are trusted by the semantic voxel
map.  Every query is made before the current observation is inserted into the
history, so the policy cannot use a future frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import itertools
import math

import torch


@dataclass(frozen=True)
class TemporalConsensusConfig:
    """Configuration for causal object-point consensus.

    ``support_radius_m`` is an object-point distance in the common world
    frame.  A point outside the historical support is treated as a novel
    surface point.  Novel points are retained only in a small, confidence
    ordered budget and receive a lower map-write multiplier.
    """

    history_frames: int = 8
    max_history_points: int = 4096
    min_history_points: int = 16
    support_radius_m: float = 0.08
    min_support_points: int = 16
    min_support_ratio: float = 0.20
    max_novel_points: int = 256
    novel_weight: float = 0.20

    def validate(self) -> "TemporalConsensusConfig":
        for name, value in (
            ("history_frames", self.history_frames),
            ("max_history_points", self.max_history_points),
            ("min_history_points", self.min_history_points),
            ("min_support_points", self.min_support_points),
            ("max_novel_points", self.max_novel_points),
        ):
            if int(value) < 1:
                raise ValueError(f"temporal_consensus.{name} must be positive.")
        if int(self.min_history_points) > int(self.max_history_points):
            raise ValueError(
                "temporal_consensus.min_history_points cannot exceed "
                "max_history_points."
            )
        if float(self.support_radius_m) <= 0.0:
            raise ValueError(
                "temporal_consensus.support_radius_m must be positive."
            )
        if not 0.0 <= float(self.min_support_ratio) <= 1.0:
            raise ValueError(
                "temporal_consensus.min_support_ratio must be in [0,1]."
            )
        if not 0.0 < float(self.novel_weight) <= 1.0:
            raise ValueError(
                "temporal_consensus.novel_weight must be in (0,1]."
            )
        return self

    def to_dict(self) -> dict[str, object]:
        return {
            "history_frames": int(self.history_frames),
            "max_history_points": int(self.max_history_points),
            "min_history_points": int(self.min_history_points),
            "support_radius_m": float(self.support_radius_m),
            "min_support_points": int(self.min_support_points),
            "min_support_ratio": float(self.min_support_ratio),
            "max_novel_points": int(self.max_novel_points),
            "novel_weight": float(self.novel_weight),
        }


@dataclass(frozen=True)
class TemporalConsensusDecision:
    """Map-write decision for one already filtered object observation."""

    use_consensus: bool
    keep_mask: torch.Tensor
    weight_multipliers: torch.Tensor
    reason: str
    history_point_count: int
    history_frame_count: int
    supported_points: int
    novel_points: int
    output_points: int
    support_ratio: float
    filtered_point_ratio: float
    downweighted_point_ratio: float

    def __post_init__(self) -> None:
        if self.keep_mask.ndim != 1 or self.weight_multipliers.ndim != 1:
            raise ValueError("Consensus masks and weights must be one-dimensional.")
        if self.keep_mask.shape != self.weight_multipliers.shape:
            raise ValueError("Consensus masks and weights must have equal shapes.")

    def to_dict(self) -> dict[str, object]:
        return {
            "use_consensus": int(self.use_consensus),
            "reason": str(self.reason),
            "history_point_count": int(self.history_point_count),
            "history_frame_count": int(self.history_frame_count),
            "supported_points": int(self.supported_points),
            "novel_points": int(self.novel_points),
            "output_points": int(self.output_points),
            "support_ratio": float(self.support_ratio),
            "filtered_point_ratio": float(self.filtered_point_ratio),
            "downweighted_point_ratio": float(self.downweighted_point_ratio),
        }


@dataclass
class _ConsensusState:
    entries: list[tuple[int, torch.Tensor]] = field(default_factory=list)
    observation_count: int = 0


class TemporalConsensusMemory:
    """Bounded causal memory for one or more persistent object instances."""

    def __init__(self, config: TemporalConsensusConfig | None = None) -> None:
        self.config = (config or TemporalConsensusConfig()).validate()
        self._states: dict[int, _ConsensusState] = {}
        self._decision_count = 0
        self._consensus_count = 0
        self._fallback_count = 0
        self._filtered_points = 0
        self._downweighted_points = 0
        self._raw_points = 0
        self._output_points = 0

    def decide(
        self,
        instance_id: int,
        points: torch.Tensor,
        weights: torch.Tensor,
        *,
        frame_id: int,
    ) -> TemporalConsensusDecision:
        """Score current points against strictly earlier object points.

        ``points`` and ``weights`` describe the same already accepted object
        observation.  The returned tensors are aligned to that input.  The
        state is not modified by this method; call :meth:`update` after the
        map write decision has been made.
        """

        frame_id = int(frame_id)
        if frame_id < 0:
            raise ValueError("Temporal consensus frame IDs must be non-negative.")
        current = torch.as_tensor(points).detach().float().cpu().reshape(-1, 3)
        current_weights = (
            torch.as_tensor(weights).detach().float().cpu().reshape(-1)
        )
        if current.shape[0] != current_weights.shape[0]:
            raise ValueError("Consensus points and weights have different lengths.")
        finite = torch.isfinite(current).all(dim=1) & torch.isfinite(current_weights)
        if not bool(finite.all()):
            raise ValueError("Consensus input contains non-finite points or weights.")
        raw_count = int(current.shape[0])
        state = self._states.get(int(instance_id))
        if state is not None and state.entries and frame_id <= int(state.entries[-1][0]):
            raise ValueError(
                "Temporal consensus decisions require a frame later than the "
                "latest stored history frame."
            )
        history, history_frames = self._history(state)
        history_count = int(history.shape[0])
        self._decision_count += 1
        self._raw_points += raw_count

        all_mask = torch.ones(raw_count, dtype=torch.bool)
        all_weights = torch.ones(raw_count, dtype=torch.float32)
        if raw_count == 0:
            return TemporalConsensusDecision(
                use_consensus=False,
                keep_mask=all_mask,
                weight_multipliers=all_weights,
                reason="fallback:empty_observation",
                history_point_count=history_count,
                history_frame_count=history_frames,
                supported_points=0,
                novel_points=0,
                output_points=0,
                support_ratio=0.0,
                filtered_point_ratio=0.0,
                downweighted_point_ratio=0.0,
            )
        if history_count < int(self.config.min_history_points):
            self._fallback_count += 1
            self._output_points += raw_count
            return TemporalConsensusDecision(
                use_consensus=False,
                keep_mask=all_mask,
                weight_multipliers=all_weights,
                reason="fallback:history_insufficient",
                history_point_count=history_count,
                history_frame_count=history_frames,
                supported_points=0,
                novel_points=raw_count,
                output_points=raw_count,
                support_ratio=0.0,
                filtered_point_ratio=0.0,
                downweighted_point_ratio=0.0,
            )

        supported = _points_within_radius(
            current,
            history,
            radius=float(self.config.support_radius_m),
        )
        supported_count = int(supported.sum())
        support_ratio = supported_count / raw_count if raw_count else 0.0
        if (
            supported_count < int(self.config.min_support_points)
            or support_ratio < float(self.config.min_support_ratio)
        ):
            self._fallback_count += 1
            self._output_points += raw_count
            return TemporalConsensusDecision(
                use_consensus=False,
                keep_mask=all_mask,
                weight_multipliers=all_weights,
                reason="fallback:low_historical_support",
                history_point_count=history_count,
                history_frame_count=history_frames,
                supported_points=supported_count,
                novel_points=raw_count - supported_count,
                output_points=raw_count,
                support_ratio=support_ratio,
                filtered_point_ratio=0.0,
                downweighted_point_ratio=0.0,
            )

        novel = ~supported
        novel_indices = novel.nonzero(as_tuple=False).flatten()
        if novel_indices.numel() > int(self.config.max_novel_points):
            # The mapper has already sorted points by confidence.  Stable
            # ordering keeps this selection deterministic and favors the
            # highest-confidence novel points.
            novel_indices = novel_indices[: int(self.config.max_novel_points)]
        keep = supported.clone()
        keep[novel_indices] = True
        multipliers = torch.ones(raw_count, dtype=torch.float32)
        multipliers[novel_indices] = float(self.config.novel_weight)
        output_count = int(keep.sum())
        downweighted_count = int(novel_indices.numel())
        self._consensus_count += 1
        self._filtered_points += raw_count - output_count
        self._downweighted_points += downweighted_count
        self._output_points += output_count
        return TemporalConsensusDecision(
            use_consensus=True,
            keep_mask=keep,
            weight_multipliers=multipliers,
            reason="consensus:historical_support",
            history_point_count=history_count,
            history_frame_count=history_frames,
            supported_points=supported_count,
            novel_points=downweighted_count,
            output_points=output_count,
            support_ratio=support_ratio,
            filtered_point_ratio=(
                (raw_count - output_count) / raw_count if raw_count else 0.0
            ),
            downweighted_point_ratio=(
                downweighted_count / output_count if output_count else 0.0
            ),
        )

    def update(
        self,
        instance_id: int,
        points: torch.Tensor,
        weights: torch.Tensor,
        *,
        frame_id: int,
        decision: TemporalConsensusDecision,
    ) -> None:
        """Insert an accepted observation after its decision was consumed."""

        values = torch.as_tensor(points).detach().float().cpu().reshape(-1, 3)
        scores = torch.as_tensor(weights).detach().float().cpu().reshape(-1)
        if values.shape[0] != scores.shape[0] or values.shape[0] != decision.keep_mask.shape[0]:
            raise ValueError("Consensus update tensors do not have equal lengths.")
        if not values.numel():
            return
        state = self._states.setdefault(int(instance_id), _ConsensusState())
        if state.entries and int(frame_id) <= int(state.entries[-1][0]):
            raise ValueError("Temporal consensus updates require increasing frame IDs.")

        # Do not poison an established memory with a low-support fallback.
        # Bootstrap is the one intentional exception: the first observation
        # must seed a new instance so the next frame can be checked.
        should_insert = not state.entries or bool(decision.use_consensus)
        if not should_insert:
            state.observation_count += 1
            return
        selected = values[decision.keep_mask]
        selected_scores = scores[decision.keep_mask]
        if not selected.numel():
            state.observation_count += 1
            return
        per_entry_limit = max(
            1,
            math.ceil(
                int(self.config.max_history_points)
                / int(self.config.history_frames)
            ),
        )
        selected = _top_weighted_points(
            selected,
            selected_scores,
            limit=per_entry_limit,
        )
        state.entries.append((int(frame_id), selected))
        state.entries = state.entries[-int(self.config.history_frames) :]
        state.observation_count += 1

    def summary(self) -> dict[str, object]:
        history_points = sum(
            int(sum(int(entry[1].shape[0]) for entry in state.entries))
            for state in self._states.values()
        )
        history_frames = sum(
            int(len(state.entries)) for state in self._states.values()
        )
        return {
            "enabled": True,
            "config": self.config.to_dict(),
            "instance_count": int(len(self._states)),
            "stored_history_points": int(history_points),
            "stored_history_frames": int(history_frames),
            "decision_count": int(self._decision_count),
            "consensus_count": int(self._consensus_count),
            "fallback_count": int(self._fallback_count),
            "raw_points": int(self._raw_points),
            "output_points": int(self._output_points),
            "filtered_points": int(self._filtered_points),
            "downweighted_points": int(self._downweighted_points),
            "filtered_point_ratio": (
                self._filtered_points / self._raw_points
                if self._raw_points
                else 0.0
            ),
            "downweighted_point_ratio": (
                self._downweighted_points / self._output_points
                if self._output_points
                else 0.0
            ),
            "states": [
                {
                    "instance_id": int(instance_id),
                    "history_frames": int(len(self._states[instance_id].entries)),
                    "history_points": int(
                        sum(
                            int(entry[1].shape[0])
                            for entry in self._states[instance_id].entries
                        )
                    ),
                    "observation_count": int(
                        self._states[instance_id].observation_count
                    ),
                }
                for instance_id in sorted(self._states)
            ],
        }

    def _history(
        self,
        state: _ConsensusState | None,
    ) -> tuple[torch.Tensor, int]:
        if state is None or not state.entries:
            return torch.empty(0, 3, dtype=torch.float32), 0
        values = torch.cat([entry[1] for entry in state.entries], dim=0)
        if values.shape[0] > int(self.config.max_history_points):
            indices = torch.linspace(
                0,
                values.shape[0] - 1,
                steps=int(self.config.max_history_points),
            ).round().long()
            values = values.index_select(0, indices)
        return values, len(state.entries)


def _top_weighted_points(
    points: torch.Tensor,
    weights: torch.Tensor,
    *,
    limit: int,
) -> torch.Tensor:
    if points.shape[0] <= int(limit):
        return points.detach().float().cpu()
    order = torch.argsort(weights, descending=True, stable=True)[: int(limit)]
    return points.index_select(0, order).detach().float().cpu()


def _points_within_radius(
    points: torch.Tensor,
    history: torch.Tensor,
    *,
    radius: float,
) -> torch.Tensor:
    """Return exact nearest-neighbor support using a small spatial hash.

    A full ``N x M`` distance matrix is unnecessarily expensive for dense
    masks.  The hash limits each current cell to its 27 neighboring cells;
    ``cdist`` is still used inside each local group, so the final radius test
    remains exact for the searched neighborhood.
    """

    if not points.numel() or not history.numel():
        return torch.zeros(points.shape[0], dtype=torch.bool)
    cell_size = float(radius)
    history_cells = torch.floor(history / cell_size).long()
    point_cells = torch.floor(points / cell_size).long()
    buckets: dict[tuple[int, int, int], list[int]] = {}
    for index, cell in enumerate(history_cells.tolist()):
        buckets.setdefault(tuple(int(value) for value in cell), []).append(index)
    supported = torch.zeros(points.shape[0], dtype=torch.bool)
    unique_cells, inverse = torch.unique(
        point_cells,
        dim=0,
        sorted=True,
        return_inverse=True,
    )
    offsets = tuple(itertools.product((-1, 0, 1), repeat=3))
    for group_index, cell in enumerate(unique_cells.tolist()):
        current_indices = (inverse == int(group_index)).nonzero(
            as_tuple=False
        ).flatten()
        candidate_indices: list[int] = []
        base = tuple(int(value) for value in cell)
        for dx, dy, dz in offsets:
            candidate_indices.extend(
                buckets.get((base[0] + dx, base[1] + dy, base[2] + dz), ())
            )
        if not candidate_indices:
            continue
        history_indices = torch.tensor(candidate_indices, dtype=torch.long)
        distances = torch.cdist(
            points.index_select(0, current_indices),
            history.index_select(0, history_indices),
        )
        supported[current_indices] = distances.min(dim=1).values <= float(radius)
    return supported


__all__ = [
    "TemporalConsensusConfig",
    "TemporalConsensusDecision",
    "TemporalConsensusMemory",
]
