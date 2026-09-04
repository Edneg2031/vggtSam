"""Causal point-cloud consistency for persistent semantic instances.

This module is deliberately downstream of HorizonStream and SAM3.1.  It does
not change camera poses, depth, SAM masks, or persistent IDs.  For one
persistent instance it keeps points that agree with earlier world-space
observations, allows a bounded amount of new surface, and rejects points that
fall outside the historical object extent.  The raw per-frame points remain
available in ``object_tracks`` for an auditable comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch

from .temporal_consensus import _points_within_radius, _top_weighted_points


@dataclass(frozen=True)
class InstancePointConsistencyConfig:
    """Configuration for a causal object-level point support gate."""

    enabled: bool = False
    history_frames: int = 12
    max_history_points: int = 4096
    min_history_points: int = 16
    support_radius_m: float = 0.10
    min_support_points: int = 8
    min_support_ratio: float = 0.15
    bounds_margin_m: float = 0.20
    max_novel_points: int = 512
    novel_weight: float = 0.25

    def validate(self) -> "InstancePointConsistencyConfig":
        for name, value in (
            ("history_frames", self.history_frames),
            ("max_history_points", self.max_history_points),
            ("min_history_points", self.min_history_points),
            ("min_support_points", self.min_support_points),
            ("max_novel_points", self.max_novel_points),
        ):
            if int(value) < 1:
                raise ValueError(f"instance_point_consistency.{name} must be positive.")
        if int(self.min_history_points) > int(self.max_history_points):
            raise ValueError(
                "instance_point_consistency.min_history_points cannot exceed "
                "max_history_points."
            )
        if float(self.support_radius_m) <= 0.0:
            raise ValueError(
                "instance_point_consistency.support_radius_m must be positive."
            )
        if float(self.bounds_margin_m) < 0.0:
            raise ValueError(
                "instance_point_consistency.bounds_margin_m cannot be negative."
            )
        if not 0.0 <= float(self.min_support_ratio) <= 1.0:
            raise ValueError(
                "instance_point_consistency.min_support_ratio must be in [0,1]."
            )
        if not 0.0 < float(self.novel_weight) <= 1.0:
            raise ValueError(
                "instance_point_consistency.novel_weight must be in (0,1]."
            )
        return self

    def to_dict(self) -> dict[str, object]:
        return {
            "enabled": bool(self.enabled),
            "history_frames": int(self.history_frames),
            "max_history_points": int(self.max_history_points),
            "min_history_points": int(self.min_history_points),
            "support_radius_m": float(self.support_radius_m),
            "min_support_points": int(self.min_support_points),
            "min_support_ratio": float(self.min_support_ratio),
            "bounds_margin_m": float(self.bounds_margin_m),
            "max_novel_points": int(self.max_novel_points),
            "novel_weight": float(self.novel_weight),
        }


@dataclass(frozen=True)
class InstancePointConsistencyDecision:
    """Decision for one object observation before semantic map insertion."""

    keep_mask: torch.Tensor
    weight_multipliers: torch.Tensor
    reason: str
    history_point_count: int
    history_frame_count: int
    supported_points: int
    in_bounds_points: int
    novel_points: int
    output_points: int
    support_ratio: float
    filtered_point_ratio: float
    downweighted_point_ratio: float
    center_distance_m: float | None
    consistency_ready: bool

    def __post_init__(self) -> None:
        if self.keep_mask.ndim != 1 or self.weight_multipliers.ndim != 1:
            raise ValueError("Consistency masks and weights must be one-dimensional.")
        if self.keep_mask.shape != self.weight_multipliers.shape:
            raise ValueError("Consistency masks and weights must have equal shapes.")

    def to_dict(self) -> dict[str, object]:
        return {
            "reason": str(self.reason),
            "history_point_count": int(self.history_point_count),
            "history_frame_count": int(self.history_frame_count),
            "supported_points": int(self.supported_points),
            "in_bounds_points": int(self.in_bounds_points),
            "novel_points": int(self.novel_points),
            "output_points": int(self.output_points),
            "support_ratio": float(self.support_ratio),
            "filtered_point_ratio": float(self.filtered_point_ratio),
            "downweighted_point_ratio": float(self.downweighted_point_ratio),
            "center_distance_m": (
                None
                if self.center_distance_m is None
                else float(self.center_distance_m)
            ),
            "consistency_ready": int(self.consistency_ready),
        }


@dataclass
class _InstanceState:
    entries: list[tuple[int, torch.Tensor]] = field(default_factory=list)
    observation_count: int = 0


class InstancePointConsistencyMemory:
    """Bounded, causal memory keyed by SAM persistent instance ID."""

    def __init__(
        self,
        config: InstancePointConsistencyConfig | None = None,
    ) -> None:
        self.config = (config or InstancePointConsistencyConfig()).validate()
        self._states: dict[int, _InstanceState] = {}
        self._decision_count = 0
        self._bootstrap_count = 0
        self._consistency_count = 0
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
    ) -> InstancePointConsistencyDecision:
        """Compare points with strictly earlier observations of the instance."""

        frame_id = int(frame_id)
        if frame_id < 0:
            raise ValueError("Instance consistency frame IDs must be non-negative.")
        current = torch.as_tensor(points).detach().float().cpu().reshape(-1, 3)
        current_weights = (
            torch.as_tensor(weights).detach().float().cpu().reshape(-1)
        )
        if current.shape[0] != current_weights.shape[0]:
            raise ValueError("Consistency points and weights have different lengths.")
        finite = torch.isfinite(current).all(dim=1) & torch.isfinite(current_weights)
        if not bool(finite.all()):
            raise ValueError("Consistency input contains non-finite points or weights.")

        state = self._states.get(int(instance_id))
        if state is not None and state.entries and frame_id <= int(state.entries[-1][0]):
            raise ValueError(
                "Instance consistency decisions require a frame later than the "
                "latest stored history frame."
            )
        history, history_frames = self._history(state)
        history_count = int(history.shape[0])
        raw_count = int(current.shape[0])
        self._decision_count += 1
        self._raw_points += raw_count

        all_keep = torch.ones(raw_count, dtype=torch.bool)
        all_weights = torch.ones(raw_count, dtype=torch.float32)
        if raw_count == 0:
            return InstancePointConsistencyDecision(
                keep_mask=all_keep,
                weight_multipliers=all_weights,
                reason="fallback:empty_observation",
                history_point_count=history_count,
                history_frame_count=history_frames,
                supported_points=0,
                in_bounds_points=0,
                novel_points=0,
                output_points=0,
                support_ratio=0.0,
                filtered_point_ratio=0.0,
                downweighted_point_ratio=0.0,
                center_distance_m=None,
                consistency_ready=False,
            )
        if history_count < int(self.config.min_history_points):
            self._bootstrap_count += 1
            self._output_points += raw_count
            return InstancePointConsistencyDecision(
                keep_mask=all_keep,
                weight_multipliers=all_weights,
                reason="bootstrap:history_insufficient",
                history_point_count=history_count,
                history_frame_count=history_frames,
                supported_points=0,
                in_bounds_points=raw_count,
                novel_points=raw_count,
                output_points=raw_count,
                support_ratio=0.0,
                filtered_point_ratio=0.0,
                downweighted_point_ratio=0.0,
                center_distance_m=None,
                consistency_ready=False,
            )

        supported = _points_within_radius(
            current,
            history,
            radius=float(self.config.support_radius_m),
        )
        history_min = history.min(dim=0).values - float(self.config.bounds_margin_m)
        history_max = history.max(dim=0).values + float(self.config.bounds_margin_m)
        in_bounds = (current >= history_min).all(dim=1) & (
            current <= history_max
        ).all(dim=1)
        trusted = supported | in_bounds
        supported_count = int(supported.sum())
        in_bounds_count = int(in_bounds.sum())
        support_ratio = supported_count / raw_count if raw_count else 0.0
        consistency_ready = (
            supported_count >= int(self.config.min_support_points)
            and support_ratio >= float(self.config.min_support_ratio)
        )

        history_center = history.mean(dim=0)
        current_center = current.mean(dim=0)
        center_distance = float(torch.linalg.vector_norm(current_center - history_center))

        novel_indices = (trusted & ~supported).nonzero(as_tuple=False).flatten()
        if novel_indices.numel() > int(self.config.max_novel_points):
            novel_indices = novel_indices[: int(self.config.max_novel_points)]
        keep = supported.clone()
        keep[novel_indices] = True
        multipliers = torch.ones(raw_count, dtype=torch.float32)
        multipliers[novel_indices] = float(self.config.novel_weight)
        output_count = int(keep.sum())
        downweighted_count = int(novel_indices.numel())
        self._consistency_count += 1
        self._filtered_points += raw_count - output_count
        self._downweighted_points += downweighted_count
        self._output_points += output_count
        return InstancePointConsistencyDecision(
            keep_mask=keep,
            weight_multipliers=multipliers,
            reason=(
                "consistency:historical_support"
                if consistency_ready
                else "guard:bounded_historical_extent"
            ),
            history_point_count=history_count,
            history_frame_count=history_frames,
            supported_points=supported_count,
            in_bounds_points=in_bounds_count,
            novel_points=downweighted_count,
            output_points=output_count,
            support_ratio=support_ratio,
            filtered_point_ratio=(
                (raw_count - output_count) / raw_count if raw_count else 0.0
            ),
            downweighted_point_ratio=(
                downweighted_count / output_count if output_count else 0.0
            ),
            center_distance_m=center_distance,
            consistency_ready=consistency_ready,
        )

    def update(
        self,
        instance_id: int,
        points: torch.Tensor,
        weights: torch.Tensor,
        *,
        frame_id: int,
        decision: InstancePointConsistencyDecision,
    ) -> None:
        """Insert only accepted points after the current decision."""

        values = torch.as_tensor(points).detach().float().cpu().reshape(-1, 3)
        scores = torch.as_tensor(weights).detach().float().cpu().reshape(-1)
        if (
            values.shape[0] != scores.shape[0]
            or values.shape[0] != decision.keep_mask.shape[0]
        ):
            raise ValueError("Consistency update tensors have different lengths.")
        state = self._states.setdefault(int(instance_id), _InstanceState())
        if state.entries and int(frame_id) <= int(state.entries[-1][0]):
            raise ValueError("Instance consistency updates require increasing frame IDs.")
        if not values.numel():
            state.observation_count += 1
            return

        # Bootstrap seeds a new object.  Once a history exists, a low-support
        # observation is not allowed to move the historical extent by itself.
        should_insert = not state.entries or bool(decision.consistency_ready)
        if should_insert:
            selected = values[decision.keep_mask]
            selected_scores = scores[decision.keep_mask]
            if selected.numel():
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
        history_frames = sum(len(state.entries) for state in self._states.values())
        return {
            "enabled": True,
            "config": self.config.to_dict(),
            "instance_count": int(len(self._states)),
            "stored_history_points": int(history_points),
            "stored_history_frames": int(history_frames),
            "decision_count": int(self._decision_count),
            "bootstrap_count": int(self._bootstrap_count),
            "consistency_count": int(self._consistency_count),
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
        state: _InstanceState | None,
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


__all__ = [
    "InstancePointConsistencyConfig",
    "InstancePointConsistencyDecision",
    "InstancePointConsistencyMemory",
]
