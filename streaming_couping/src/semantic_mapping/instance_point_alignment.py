"""Causal local rigid alignment for persistent SAM object point clouds.

This module operates after HorizonStream has produced world-space points and
after SAM has supplied persistent instance masks.  It estimates a small
rigid transform for the current instance cloud against earlier aligned
observations of the same instance.  The transform is applied only to the
object points that enter the semantic map and object track; camera poses and
full-scene geometry are never modified.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch

from .object_pose_refinement import weighted_rigid_transform


@dataclass(frozen=True)
class InstancePointAlignmentConfig:
    """Configuration for local, object-only 3D point-cloud alignment."""

    enabled: bool = False
    history_frames: int = 8
    max_history_points: int = 2048
    min_history_points: int = 32
    max_registration_points: int = 512
    max_match_distance_m: float = 0.15
    min_matches: int = 16
    trim_ratio: float = 0.80
    max_iterations: int = 4
    min_relative_improvement: float = 0.05
    max_rotation_deg: float = 5.0
    max_translation_m: float = 0.10

    def validate(self) -> "InstancePointAlignmentConfig":
        for name, value in (
            ("history_frames", self.history_frames),
            ("max_history_points", self.max_history_points),
            ("min_history_points", self.min_history_points),
            ("max_registration_points", self.max_registration_points),
            ("min_matches", self.min_matches),
            ("max_iterations", self.max_iterations),
        ):
            if int(value) < 1:
                raise ValueError(f"instance_point_alignment.{name} must be positive.")
        if int(self.min_history_points) > int(self.max_history_points):
            raise ValueError(
                "instance_point_alignment.min_history_points cannot exceed "
                "max_history_points."
            )
        for name, value in (
            ("max_match_distance_m", self.max_match_distance_m),
            ("max_rotation_deg", self.max_rotation_deg),
            ("max_translation_m", self.max_translation_m),
        ):
            if float(value) <= 0.0:
                raise ValueError(
                    f"instance_point_alignment.{name} must be positive."
                )
        if not 0.0 < float(self.trim_ratio) <= 1.0:
            raise ValueError("instance_point_alignment.trim_ratio must be in (0,1].")
        if not 0.0 <= float(self.min_relative_improvement) < 1.0:
            raise ValueError(
                "instance_point_alignment.min_relative_improvement must be in [0,1)."
            )
        return self

    def to_dict(self) -> dict[str, object]:
        return {
            "enabled": bool(self.enabled),
            "history_frames": int(self.history_frames),
            "max_history_points": int(self.max_history_points),
            "min_history_points": int(self.min_history_points),
            "max_registration_points": int(self.max_registration_points),
            "max_match_distance_m": float(self.max_match_distance_m),
            "min_matches": int(self.min_matches),
            "trim_ratio": float(self.trim_ratio),
            "max_iterations": int(self.max_iterations),
            "min_relative_improvement": float(self.min_relative_improvement),
            "max_rotation_deg": float(self.max_rotation_deg),
            "max_translation_m": float(self.max_translation_m),
        }


@dataclass(frozen=True)
class InstancePointAlignmentDecision:
    """Auditable local alignment decision for one object observation."""

    transform: torch.Tensor
    accepted: bool
    bootstrap: bool
    update_history: bool
    reason: str
    history_point_count: int
    history_frame_count: int
    match_count: int
    initial_rmse_m: float | None
    final_rmse_m: float | None
    relative_improvement: float | None
    correction_rotation_deg: float
    correction_translation_m: float

    def __post_init__(self) -> None:
        if tuple(self.transform.shape) != (4, 4):
            raise ValueError("Alignment transform must have shape [4,4].")

    def to_dict(self) -> dict[str, object]:
        return {
            "accepted": int(self.accepted),
            "bootstrap": int(self.bootstrap),
            "update_history": int(self.update_history),
            "reason": str(self.reason),
            "history_point_count": int(self.history_point_count),
            "history_frame_count": int(self.history_frame_count),
            "match_count": int(self.match_count),
            "initial_rmse_m": _finite_or_none(self.initial_rmse_m),
            "final_rmse_m": _finite_or_none(self.final_rmse_m),
            "relative_improvement": _finite_or_none(self.relative_improvement),
            "correction_rotation_deg": float(self.correction_rotation_deg),
            "correction_translation_m": float(self.correction_translation_m),
        }


@dataclass
class _InstanceState:
    entries: list[tuple[int, torch.Tensor]] = field(default_factory=list)
    observation_count: int = 0


class InstancePointAlignmentMemory:
    """Bounded causal memory keyed by persistent SAM instance ID."""

    def __init__(
        self,
        config: InstancePointAlignmentConfig | None = None,
    ) -> None:
        self.config = (config or InstancePointAlignmentConfig()).validate()
        self._states: dict[int, _InstanceState] = {}
        self._decision_count = 0
        self._bootstrap_count = 0
        self._accepted_count = 0
        self._rejected_count = 0
        self._raw_points = 0
        self._aligned_points = 0
        self._events: list[dict[str, object]] = []

    def decide(
        self,
        instance_id: int,
        points: torch.Tensor,
        weights: torch.Tensor,
        *,
        frame_id: int,
    ) -> InstancePointAlignmentDecision:
        """Estimate a transform mapping current points into historical support."""

        frame_id = int(frame_id)
        if frame_id < 0:
            raise ValueError("Instance alignment frame IDs must be non-negative.")
        current = torch.as_tensor(points).detach().float().cpu().reshape(-1, 3)
        current_weights = torch.as_tensor(weights).detach().float().cpu().reshape(-1)
        if current.shape[0] != current_weights.shape[0]:
            raise ValueError("Alignment points and weights have different lengths.")
        finite = torch.isfinite(current).all(dim=1) & torch.isfinite(current_weights)
        if not bool(finite.all()):
            raise ValueError("Alignment input contains non-finite points or weights.")

        state = self._states.get(int(instance_id))
        if state is not None and state.entries and frame_id <= int(state.entries[-1][0]):
            raise ValueError(
                "Instance alignment decisions require a frame later than the "
                "latest stored history frame."
            )
        history, history_frames = self._history(state)
        history_count = int(history.shape[0])
        raw_count = int(current.shape[0])
        self._decision_count += 1
        self._raw_points += raw_count

        identity = torch.eye(4, dtype=torch.float32)
        if raw_count < 3:
            decision = self._decision(
                identity,
                accepted=False,
                bootstrap=False,
                update_history=not bool(state and state.entries),
                reason="reject:too_few_current_points",
                history_point_count=history_count,
                history_frame_count=history_frames,
                match_count=0,
                initial_rmse_m=None,
                final_rmse_m=None,
                relative_improvement=None,
            )
            self._record(instance_id, frame_id, decision, raw_count)
            return decision

        if history_count < int(self.config.min_history_points):
            decision = self._decision(
                identity,
                accepted=False,
                bootstrap=True,
                update_history=True,
                reason="bootstrap:history_insufficient",
                history_point_count=history_count,
                history_frame_count=history_frames,
                match_count=0,
                initial_rmse_m=None,
                final_rmse_m=None,
                relative_improvement=None,
            )
            self._bootstrap_count += 1
            self._record(instance_id, frame_id, decision, raw_count)
            return decision

        source, source_weights = _select_weighted(
            current,
            current_weights,
            int(self.config.max_registration_points),
        )
        target = _select_history(history, int(self.config.max_history_points))
        if int(source.shape[0]) < 3 or int(target.shape[0]) < 3:
            decision = self._decision(
                identity,
                accepted=False,
                bootstrap=False,
                update_history=False,
                reason="reject:insufficient_registration_points",
                history_point_count=history_count,
                history_frame_count=history_frames,
                match_count=0,
                initial_rmse_m=None,
                final_rmse_m=None,
                relative_improvement=None,
            )
            self._rejected_count += 1
            self._record(instance_id, frame_id, decision, raw_count)
            return decision

        transform = identity
        initial_rmse: float | None = None
        match_count = 0
        reason = "reject:no_valid_correspondences"
        for _ in range(int(self.config.max_iterations)):
            transformed = apply_point_alignment(transform, source)
            distances, nearest = _nearest_neighbors(transformed, target)
            valid = distances <= float(self.config.max_match_distance_m)
            valid_count = int(valid.sum())
            if valid_count < int(self.config.min_matches):
                break
            keep = _trim_indices(
                distances,
                valid,
                trim_ratio=float(self.config.trim_ratio),
                min_count=int(self.config.min_matches),
            )
            if int(keep.numel()) < int(self.config.min_matches):
                break
            match_count = int(keep.numel())
            if initial_rmse is None:
                initial_rmse = _weighted_rmse(
                    distances.index_select(0, keep),
                    source_weights.index_select(0, keep),
                )
            candidate = weighted_rigid_transform(
                source.index_select(0, keep),
                target.index_select(0, nearest.index_select(0, keep)),
                source_weights.index_select(0, keep),
            )
            delta = torch.linalg.vector_norm(
                (candidate[:3, :3] - transform[:3, :3]).reshape(-1)
            ) + torch.linalg.vector_norm(candidate[:3, 3] - transform[:3, 3])
            transform = candidate
            reason = "candidate"
            if float(delta) < 1e-5:
                break

        transformed = apply_point_alignment(transform, source)
        final_distances, _ = _nearest_neighbors(transformed, target)
        final_valid = final_distances <= float(self.config.max_match_distance_m)
        final_keep = _trim_indices(
            final_distances,
            final_valid,
            trim_ratio=float(self.config.trim_ratio),
            min_count=int(self.config.min_matches),
        )
        if int(final_keep.numel()) < int(self.config.min_matches):
            final_rmse = None
            relative_improvement = None
            transform = identity
            reason = "reject:insufficient_final_matches"
        else:
            match_count = int(final_keep.numel())
            final_rmse = _weighted_rmse(
                final_distances.index_select(0, final_keep),
                source_weights.index_select(0, final_keep),
            )
            relative_improvement = (
                None
                if initial_rmse is None
                else (initial_rmse - final_rmse) / max(initial_rmse, 1e-6)
            )

        rotation_deg, translation_m = _transform_magnitude(transform)
        accepted = bool(
            final_rmse is not None
            and relative_improvement is not None
            and initial_rmse is not None
            and float(relative_improvement)
            >= float(self.config.min_relative_improvement)
            and float(rotation_deg) <= float(self.config.max_rotation_deg)
            and float(translation_m) <= float(self.config.max_translation_m)
        )
        if not accepted:
            if reason == "candidate":
                if (
                    rotation_deg > float(self.config.max_rotation_deg)
                    or translation_m > float(self.config.max_translation_m)
                ):
                    reason = "reject:correction_gate"
                else:
                    reason = "reject:insufficient_relative_improvement"
            transform = identity
            rotation_deg = 0.0
            translation_m = 0.0
            self._rejected_count += 1
        else:
            reason = "accept:local_rigid_alignment"
            self._accepted_count += 1

        decision = self._decision(
            transform,
            accepted=accepted,
            bootstrap=False,
            update_history=accepted,
            reason=reason,
            history_point_count=history_count,
            history_frame_count=history_frames,
            match_count=match_count,
            initial_rmse_m=initial_rmse,
            final_rmse_m=final_rmse,
            relative_improvement=relative_improvement,
            correction_rotation_deg=rotation_deg,
            correction_translation_m=translation_m,
        )
        self._record(instance_id, frame_id, decision, raw_count)
        return decision

    def update(
        self,
        instance_id: int,
        points: torch.Tensor,
        weights: torch.Tensor,
        *,
        frame_id: int,
        decision: InstancePointAlignmentDecision,
    ) -> None:
        """Store only bootstrap or accepted aligned observations."""

        values = torch.as_tensor(points).detach().float().cpu().reshape(-1, 3)
        scores = torch.as_tensor(weights).detach().float().cpu().reshape(-1)
        if values.shape[0] != scores.shape[0]:
            raise ValueError("Alignment update points and weights differ in length.")
        state = self._states.setdefault(int(instance_id), _InstanceState())
        if state.entries and int(frame_id) <= int(state.entries[-1][0]):
            raise ValueError("Instance alignment updates require increasing frame IDs.")
        if decision.update_history and values.numel():
            selected, _ = _select_weighted(
                values,
                scores,
                max(
                    1,
                    math.ceil(
                        int(self.config.max_history_points)
                        / int(self.config.history_frames)
                    ),
                ),
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
        accepted_events = [
            event
            for event in self._events
            if bool(event.get("accepted", False))
        ]
        return {
            "enabled": True,
            "camera_pose_modified": False,
            "full_scene_geometry_modified": False,
            "object_points_modified": True,
            "config": self.config.to_dict(),
            "instance_count": int(len(self._states)),
            "stored_history_points": int(history_points),
            "stored_history_frames": int(history_frames),
            "decision_count": int(self._decision_count),
            "bootstrap_count": int(self._bootstrap_count),
            "accepted_count": int(self._accepted_count),
            "rejected_count": int(self._rejected_count),
            "raw_points": int(self._raw_points),
            "aligned_points": int(self._aligned_points),
            "mean_initial_rmse_m": _mean_or_none(
                event.get("initial_rmse_m") for event in accepted_events
            ),
            "mean_final_rmse_m": _mean_or_none(
                event.get("final_rmse_m") for event in accepted_events
            ),
            "mean_relative_improvement": _mean_or_none(
                event.get("relative_improvement") for event in accepted_events
            ),
            "mean_correction_rotation_deg": _mean_or_none(
                event.get("correction_rotation_deg") for event in accepted_events
            ),
            "mean_correction_translation_m": _mean_or_none(
                event.get("correction_translation_m") for event in accepted_events
            ),
            "events": list(self._events),
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
            values, _ = _select_weighted(
                values,
                torch.ones(values.shape[0]),
                int(self.config.max_history_points),
            )
        return values, len(state.entries)

    def _decision(
        self,
        transform: torch.Tensor,
        *,
        accepted: bool,
        bootstrap: bool,
        update_history: bool,
        reason: str,
        history_point_count: int,
        history_frame_count: int,
        match_count: int,
        initial_rmse_m: float | None,
        final_rmse_m: float | None,
        relative_improvement: float | None,
        correction_rotation_deg: float = 0.0,
        correction_translation_m: float = 0.0,
    ) -> InstancePointAlignmentDecision:
        return InstancePointAlignmentDecision(
            transform=transform.detach().float().cpu(),
            accepted=bool(accepted),
            bootstrap=bool(bootstrap),
            update_history=bool(update_history),
            reason=str(reason),
            history_point_count=int(history_point_count),
            history_frame_count=int(history_frame_count),
            match_count=int(match_count),
            initial_rmse_m=initial_rmse_m,
            final_rmse_m=final_rmse_m,
            relative_improvement=relative_improvement,
            correction_rotation_deg=float(correction_rotation_deg),
            correction_translation_m=float(correction_translation_m),
        )

    def _record(
        self,
        instance_id: int,
        frame_id: int,
        decision: InstancePointAlignmentDecision,
        raw_count: int,
    ) -> None:
        event = {
            "instance_id": int(instance_id),
            "frame_id": int(frame_id),
            "raw_points": int(raw_count),
            **decision.to_dict(),
        }
        self._events.append(event)
        if decision.accepted:
            self._aligned_points += int(raw_count)


def apply_point_alignment(transform: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Apply a homogeneous [4,4] transform to [N,3] row-vector points."""

    matrix = torch.as_tensor(transform).detach().float().cpu()
    values = torch.as_tensor(points).detach().float().cpu().reshape(-1, 3)
    if tuple(matrix.shape) != (4, 4):
        raise ValueError("Point alignment transform must have shape [4,4].")
    return values @ matrix[:3, :3].transpose(0, 1) + matrix[:3, 3]


def _select_weighted(
    points: torch.Tensor,
    weights: torch.Tensor,
    limit: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.as_tensor(points).detach().float().cpu().reshape(-1, 3)
    scores = torch.as_tensor(weights).detach().float().cpu().reshape(-1)
    if values.shape[0] != scores.shape[0]:
        raise ValueError("Point and weight counts differ.")
    if values.shape[0] <= int(limit):
        return values, scores
    order = torch.argsort(scores, descending=True, stable=True)[: int(limit)]
    return values.index_select(0, order), scores.index_select(0, order)


def _select_history(history: torch.Tensor, limit: int) -> torch.Tensor:
    values = torch.as_tensor(history).detach().float().cpu().reshape(-1, 3)
    if values.shape[0] <= int(limit):
        return values
    indices = torch.linspace(
        0,
        values.shape[0] - 1,
        steps=int(limit),
    ).round().long()
    return values.index_select(0, indices)


def _nearest_neighbors(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    chunk_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    source = torch.as_tensor(source).detach().float().cpu().reshape(-1, 3)
    target = torch.as_tensor(target).detach().float().cpu().reshape(-1, 3)
    if not source.numel() or not target.numel():
        return (
            torch.full((source.shape[0],), float("inf")),
            torch.zeros(source.shape[0], dtype=torch.long),
        )
    distance_chunks: list[torch.Tensor] = []
    index_chunks: list[torch.Tensor] = []
    for start in range(0, int(source.shape[0]), int(chunk_size)):
        end = min(int(source.shape[0]), start + int(chunk_size))
        distances = torch.cdist(source[start:end], target)
        values, indices = distances.min(dim=1)
        distance_chunks.append(values)
        index_chunks.append(indices)
    return torch.cat(distance_chunks), torch.cat(index_chunks)


def _trim_indices(
    distances: torch.Tensor,
    valid: torch.Tensor,
    *,
    trim_ratio: float,
    min_count: int,
) -> torch.Tensor:
    candidates = valid.nonzero(as_tuple=False).flatten()
    if int(candidates.numel()) < int(min_count):
        return torch.empty(0, dtype=torch.long)
    keep_count = max(int(min_count), int(math.ceil(candidates.numel() * trim_ratio)))
    keep_count = min(keep_count, int(candidates.numel()))
    order = torch.argsort(distances.index_select(0, candidates), stable=True)
    return candidates.index_select(0, order[:keep_count])


def _weighted_rmse(residual: torch.Tensor, weights: torch.Tensor) -> float:
    if not residual.numel() or float(weights.sum()) <= 0.0:
        return float("inf")
    value = torch.sqrt((weights * residual.square()).sum() / weights.sum())
    return float(value)


def _transform_magnitude(transform: torch.Tensor) -> tuple[float, float]:
    rotation = transform[:3, :3]
    cosine = ((torch.trace(rotation) - 1.0) * 0.5).clamp(-1.0, 1.0)
    angle = math.degrees(float(torch.arccos(cosine)))
    translation = float(torch.linalg.vector_norm(transform[:3, 3]))
    return angle, translation


def _finite_or_none(value: float | None) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def _mean_or_none(values) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return None if not finite else float(sum(finite) / len(finite))


__all__ = [
    "InstancePointAlignmentConfig",
    "InstancePointAlignmentDecision",
    "InstancePointAlignmentMemory",
    "apply_point_alignment",
]
