"""Causal object-level pose refinement from instance point-cloud loss.

This module is the loss-based counterpart to ``object_pose_refinement``.  It
does not estimate a separate rigid transform for an object and then inject it
into the camera trajectory.  Instead, the current HorizonStream pose is the
initial value of a six degree-of-freedom variable.  SAM persistent-instance
masks select local depth point clouds, those clouds are matched in a common
world frame, and the matching residual directly optimizes the current camera
pose correction.

Only observations from earlier frames are used.  The first few input frames
are treated as fixed coordinate/object anchors; later frames use those
anchors and a small recent history.  The returned result intentionally has the
same pose/result contract as the existing pipeline so raw and refined
object-only maps can be exported side by side.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from typing import Any

import torch

from .contracts import GeometryFrame, SegmentationFrame, SizeHW
from .geometry import geometry_confidence_for_frame, resize_bool_mask
from .object_pose_refinement import PoseRefinementResult


@dataclass(frozen=True)
class ObjectPoseLossRefinementConfig:
    """Conservative settings for causal object-cloud loss optimization."""

    anchor_frame_count: int = 5
    max_anchor_observations: int = 3
    max_history_observations: int = 2
    max_points_per_observation: int = 256
    min_points_per_observation: int = 24

    min_track_score: float = 0.50
    static_score_threshold: float = 0.20
    require_static_score: bool = False
    min_geometry_confidence: float = 0.30
    min_mask_pixels: int = 32
    max_mask_area_ratio: float = 0.85

    max_match_distance_m: float = 0.25
    trim_ratio: float = 0.70
    min_matches_per_pair: int = 8
    min_total_matches: int = 16

    outer_iterations: int = 4
    optimizer_steps: int = 30
    learning_rate: float = 0.03
    huber_delta_m: float = 0.05
    pose_prior_weight: float = 0.02
    anchor_reference_weight: float = 1.0
    local_reference_weight: float = 0.50
    max_correction_rotation_deg: float = 10.0
    max_correction_translation_m: float = 0.25
    min_relative_loss_improvement: float = 0.02
    device: str = "cpu"

    def validate(self) -> "ObjectPoseLossRefinementConfig":
        for name, value in (
            ("anchor_frame_count", self.anchor_frame_count),
            ("max_anchor_observations", self.max_anchor_observations),
            ("max_history_observations", self.max_history_observations),
            ("max_points_per_observation", self.max_points_per_observation),
            ("min_points_per_observation", self.min_points_per_observation),
            ("min_mask_pixels", self.min_mask_pixels),
            ("min_matches_per_pair", self.min_matches_per_pair),
            ("min_total_matches", self.min_total_matches),
            ("outer_iterations", self.outer_iterations),
            ("optimizer_steps", self.optimizer_steps),
        ):
            if int(value) < 1:
                raise ValueError(f"object_pose_loss.{name} must be positive.")
        if int(self.min_total_matches) < int(self.min_matches_per_pair):
            raise ValueError(
                "object_pose_loss.min_total_matches cannot be smaller than "
                "min_matches_per_pair."
            )
        for name, value in (
            ("min_track_score", self.min_track_score),
            ("static_score_threshold", self.static_score_threshold),
            ("min_geometry_confidence", self.min_geometry_confidence),
            ("trim_ratio", self.trim_ratio),
            ("min_relative_loss_improvement", self.min_relative_loss_improvement),
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"object_pose_loss.{name} must be in [0,1].")
        if float(self.max_mask_area_ratio) <= 0.0 or float(self.max_mask_area_ratio) > 1.0:
            raise ValueError(
                "object_pose_loss.max_mask_area_ratio must be in (0,1]."
            )
        for name, value in (
            ("max_match_distance_m", self.max_match_distance_m),
            ("learning_rate", self.learning_rate),
            ("huber_delta_m", self.huber_delta_m),
            ("anchor_reference_weight", self.anchor_reference_weight),
            ("local_reference_weight", self.local_reference_weight),
            ("max_correction_rotation_deg", self.max_correction_rotation_deg),
            ("max_correction_translation_m", self.max_correction_translation_m),
        ):
            if float(value) <= 0.0:
                raise ValueError(f"object_pose_loss.{name} must be positive.")
        if float(self.pose_prior_weight) < 0.0:
            raise ValueError("object_pose_loss.pose_prior_weight cannot be negative.")
        if not str(self.device).strip():
            raise ValueError("object_pose_loss.device must not be empty.")
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor_frame_count": int(self.anchor_frame_count),
            "max_anchor_observations": int(self.max_anchor_observations),
            "max_history_observations": int(self.max_history_observations),
            "max_points_per_observation": int(self.max_points_per_observation),
            "min_points_per_observation": int(self.min_points_per_observation),
            "min_track_score": float(self.min_track_score),
            "static_score_threshold": float(self.static_score_threshold),
            "require_static_score": bool(self.require_static_score),
            "min_geometry_confidence": float(self.min_geometry_confidence),
            "min_mask_pixels": int(self.min_mask_pixels),
            "max_mask_area_ratio": float(self.max_mask_area_ratio),
            "max_match_distance_m": float(self.max_match_distance_m),
            "trim_ratio": float(self.trim_ratio),
            "min_matches_per_pair": int(self.min_matches_per_pair),
            "min_total_matches": int(self.min_total_matches),
            "outer_iterations": int(self.outer_iterations),
            "optimizer_steps": int(self.optimizer_steps),
            "learning_rate": float(self.learning_rate),
            "huber_delta_m": float(self.huber_delta_m),
            "pose_prior_weight": float(self.pose_prior_weight),
            "anchor_reference_weight": float(self.anchor_reference_weight),
            "local_reference_weight": float(self.local_reference_weight),
            "max_correction_rotation_deg": float(self.max_correction_rotation_deg),
            "max_correction_translation_m": float(self.max_correction_translation_m),
            "min_relative_loss_improvement": float(self.min_relative_loss_improvement),
            "device": str(self.device),
        }


@dataclass(frozen=True)
class ObjectCloudObservation:
    """One SAM instance's fixed camera-space point cloud."""

    frame_id: int
    instance_id: int
    category: str
    points_camera: torch.Tensor
    weights: torch.Tensor
    track_score: float
    geometry_confidence: float
    mask_pixels: int
    static: bool


@dataclass(frozen=True)
class ObjectLossEdge:
    """One accepted object alignment-loss constraint.

    ``frame_i`` is the historical reference and ``frame_j`` is the current
    pose being optimized.  No independent object transform is stored.
    """

    frame_i: int
    frame_j: int
    instance_id: int
    category: str
    reference_role: str
    num_matches_initial: int
    num_matches_final: int
    initial_loss_m: float
    final_loss_m: float
    relative_loss_improvement: float
    edge_weight: float
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_i": int(self.frame_i),
            "frame_j": int(self.frame_j),
            "instance_id": int(self.instance_id),
            "category": str(self.category),
            "reference_role": str(self.reference_role),
            "num_matches_initial": int(self.num_matches_initial),
            "num_matches_final": int(self.num_matches_final),
            "initial_loss_m": float(self.initial_loss_m),
            "final_loss_m": float(self.final_loss_m),
            "relative_loss_improvement": float(self.relative_loss_improvement),
            "edge_weight": float(self.edge_weight),
            "provenance": _json_safe(self.provenance),
        }


@dataclass(frozen=True)
class _StoredObjectCloud:
    frame_id: int
    instance_id: int
    category: str
    points_world: torch.Tensor
    weights: torch.Tensor
    role: str
    quality: float


@dataclass(frozen=True)
class _PairSpec:
    current: ObjectCloudObservation
    reference: _StoredObjectCloud
    weight: float


@dataclass(frozen=True)
class _TensorPair:
    current_points: torch.Tensor
    current_weights: torch.Tensor
    reference_points: torch.Tensor
    reference_weights: torch.Tensor
    pair_weight: float


@dataclass(frozen=True)
class _MatchSet:
    pair_index: int
    current_indices: torch.Tensor
    reference_indices: torch.Tensor
    weights: torch.Tensor


class ObjectPoseLossRefiner:
    """Incrementally optimize only the current pose from object-cloud loss."""

    method_name = "causal_anchor_object_point_alignment_loss"

    def __init__(
        self,
        config: ObjectPoseLossRefinementConfig | None = None,
    ) -> None:
        self.config = (config or ObjectPoseLossRefinementConfig()).validate()
        self.device = torch.device(self.config.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"object_pose_loss requested {self.device}, but CUDA is unavailable."
            )

    def refine(
        self,
        geometry_frames: Sequence[GeometryFrame],
        segmentation_frames: Sequence[SegmentationFrame],
        image_paths: Sequence[str] | Sequence[Any],
    ) -> PoseRefinementResult:
        """Refine poses causally using only earlier same-instance observations."""

        del image_paths  # RGB is not needed: masks and depth define the loss.
        geometry = tuple(geometry_frames)
        segmentation = tuple(segmentation_frames)
        if not geometry:
            raise ValueError("ObjectPoseLossRefiner requires geometry frames.")
        if len(geometry) != len(segmentation):
            raise ValueError("Geometry and segmentation counts must agree.")
        frame_ids = tuple(int(frame.frame_id) for frame in geometry)
        if frame_ids != tuple(sorted(frame_ids)):
            raise ValueError("ObjectPoseLossRefiner requires increasing frame IDs.")
        segmentation_by_id = {int(frame.frame_id): frame for frame in segmentation}
        if set(segmentation_by_id) != set(frame_ids):
            raise ValueError("Geometry and segmentation frame IDs do not match.")

        observations_by_frame, tracked_ids, filter_stats = _collect_observations(
            geometry,
            segmentation_by_id,
            config=self.config,
        )
        raw_observation_count = sum(
            len(segmentation_by_id[int(frame_id)].observations)
            for frame_id in frame_ids
        )
        retained_observation_count = sum(
            len(observations) for observations in observations_by_frame.values()
        )
        retained_instance_ids = {
            int(observation.instance_id)
            for observations in observations_by_frame.values()
            for observation in observations
        }
        raw_poses = tuple(_pose4(frame.camera_to_world) for frame in geometry)

        anchors: dict[int, list[_StoredObjectCloud]] = {}
        history: dict[int, list[_StoredObjectCloud]] = {}
        refined_poses: list[torch.Tensor] = []
        candidates: list[dict[str, Any]] = []
        accepted_edges: list[ObjectLossEdge] = []
        rejected_edges: list[dict[str, Any]] = []
        frame_diagnostics: list[dict[str, Any]] = []

        for sequence_index, (frame_id, raw_pose) in enumerate(zip(frame_ids, raw_poses)):
            current = observations_by_frame.get(int(frame_id), ())
            is_anchor_frame = sequence_index < int(self.config.anchor_frame_count)
            if is_anchor_frame:
                refined_pose = raw_pose
                frame_diagnostics.append(
                    {
                        "frame_id": int(frame_id),
                        "role": "anchor",
                        "optimization_attempted": False,
                        "accepted": False,
                        "reason": "early_anchor_frame",
                        "observation_count": int(len(current)),
                        "candidate_pair_count": 0,
                        "accepted_edge_count": 0,
                    }
                )
            else:
                pair_specs = []
                for observation in current:
                    references = _select_references(
                        int(observation.instance_id),
                        anchors,
                        history,
                        config=self.config,
                    )
                    for reference in references:
                        quality = math.sqrt(
                            max(0.0, float(observation.track_score))
                            * max(0.0, float(observation.geometry_confidence))
                            * max(0.0, float(reference.quality))
                        )
                        role_weight = (
                            self.config.anchor_reference_weight
                            if reference.role == "anchor"
                            else self.config.local_reference_weight
                        )
                        pair_specs.append(
                            _PairSpec(
                                current=observation,
                                reference=reference,
                                weight=max(1e-6, float(role_weight) * quality),
                            )
                        )
                if pair_specs:
                    outcome = self._optimize_current_pose(raw_pose, pair_specs)
                    refined_pose = outcome["pose"]
                    candidates.extend(outcome["candidates"])
                    accepted_edges.extend(outcome["accepted_edges"])
                    rejected_edges.extend(outcome["rejected_edges"])
                    frame_diagnostics.append(outcome["frame_diagnostic"])
                else:
                    refined_pose = raw_pose
                    frame_diagnostics.append(
                        {
                            "frame_id": int(frame_id),
                            "role": "online",
                            "optimization_attempted": False,
                            "accepted": False,
                            "reason": "no_historical_instance_reference",
                            "observation_count": int(len(current)),
                            "candidate_pair_count": 0,
                            "accepted_edge_count": 0,
                        }
                    )

            refined_poses.append(refined_pose)
            for observation in current:
                stored = _stored_cloud(
                    observation,
                    refined_pose,
                    role="anchor" if is_anchor_frame else "history",
                )
                history.setdefault(int(observation.instance_id), []).append(stored)
                history[int(observation.instance_id)] = history[
                    int(observation.instance_id)
                ][-int(self.config.max_history_observations) :]
                if is_anchor_frame:
                    anchors.setdefault(int(observation.instance_id), []).append(stored)
                    anchors[int(observation.instance_id)] = anchors[
                        int(observation.instance_id)
                    ][: int(self.config.max_anchor_observations)]

        rotation_changes: list[float] = []
        translation_changes: list[float] = []
        for raw, refined in zip(raw_poses, refined_poses):
            delta = _invert_pose(raw) @ refined
            rotation_changes.append(_rotation_angle_deg(delta[:3, :3]))
            translation_changes.append(float(torch.linalg.vector_norm(delta[:3, 3])))

        initial_losses = [
            float(edge.initial_loss_m)
            for edge in accepted_edges
            if math.isfinite(float(edge.initial_loss_m))
        ]
        final_losses = [
            float(edge.final_loss_m)
            for edge in accepted_edges
            if math.isfinite(float(edge.final_loss_m))
        ]
        reasons = Counter(str(row.get("reason", "unknown")) for row in rejected_edges)
        optimizer_frames = [
            row for row in frame_diagnostics if bool(row.get("optimization_attempted"))
        ]
        accepted_frames = [row for row in optimizer_frames if bool(row.get("accepted"))]
        summary = {
            "schema": 1,
            "revision": "sam_instance_guided_horizonstream_object_alignment_loss_r1",
            "method": self.method_name,
            "enabled": True,
            "config": self.config.to_dict(),
            "frame_count": int(len(frame_ids)),
            "frame_ids": [int(value) for value in frame_ids],
            "tracked_instance_count": int(len(tracked_ids)),
            "raw_observation_count": int(raw_observation_count),
            "retained_observation_count": int(retained_observation_count),
            "filtered_observation_count": int(
                max(0, raw_observation_count - retained_observation_count)
            ),
            "retained_instance_count": int(len(retained_instance_ids)),
            # Kept for compatibility with the earlier object-refinement
            # summary schema; this is a count of retained instance IDs, not
            # a count of frames.
            "filtered_observation_instance_count": int(len(retained_instance_ids)),
            "candidate_pair_count": int(len(candidates)),
            "accepted_edge_count": int(len(accepted_edges)),
            "rejected_edge_count": int(len(rejected_edges)),
            "rejected_reason_counts": dict(reasons),
            "anchor_frame_count": int(
                min(len(frame_ids), int(self.config.anchor_frame_count))
            ),
            "optimization_attempted_frame_count": int(len(optimizer_frames)),
            "accepted_frame_count": int(len(accepted_frames)),
            "loss_statistics": {
                "accepted_initial_mean_m": _mean_or_none(initial_losses),
                "accepted_final_mean_m": _mean_or_none(final_losses),
                "accepted_relative_improvement_mean": _mean_or_none(
                    [float(edge.relative_loss_improvement) for edge in accepted_edges]
                ),
            },
            "raw_vs_refined_pose_change": {
                "mean_rotation_correction_deg": _mean_or_zero(rotation_changes),
                "max_rotation_correction_deg": max(rotation_changes, default=0.0),
                "mean_translation_correction_m": _mean_or_zero(translation_changes),
                "max_translation_correction_m": max(translation_changes, default=0.0),
            },
            "observation_filter_reasons": {
                str(key): int(value) for key, value in filter_stats.items()
            },
            "frame_diagnostics": frame_diagnostics,
            "optimizer": {
                "backend": "torch_adam_fixed_match_loss",
                "attempted": bool(optimizer_frames),
                "success": bool(accepted_frames),
                "attempted_frame_count": int(len(optimizer_frames)),
                "accepted_frame_count": int(len(accepted_frames)),
            },
        }
        return PoseRefinementResult(
            frame_ids=frame_ids,
            raw_camera_to_world=tuple(p.detach().float().cpu() for p in raw_poses),
            refined_camera_to_world=tuple(
                p.detach().float().cpu() for p in refined_poses
            ),
            candidates=tuple(candidates),
            accepted_edges=tuple(accepted_edges),  # type: ignore[arg-type]
            rejected_edges=tuple(rejected_edges),
            summary=summary,
        )

    def _optimize_current_pose(
        self,
        raw_pose: torch.Tensor,
        pair_specs: Sequence[_PairSpec],
    ) -> dict[str, Any]:
        raw_device = raw_pose.to(self.device)
        pairs = tuple(
            _TensorPair(
                current_points=pair.current.points_camera.to(self.device),
                current_weights=pair.current.weights.to(self.device),
                reference_points=pair.reference.points_world.to(self.device),
                reference_weights=pair.reference.weights.to(self.device),
                pair_weight=float(pair.weight),
            )
            for pair in pair_specs
        )
        delta = torch.nn.Parameter(
            torch.zeros(6, dtype=torch.float32, device=self.device)
        )
        initial_sets = self._collect_match_sets(raw_device, pairs)
        initial_count = sum(int(match.current_indices.numel()) for match in initial_sets)
        if initial_count < int(self.config.min_total_matches):
            candidates, rejected = self._rows_for_rejection(
                pair_specs,
                initial_sets,
                reason="reject:too_few_object_matches",
                initial_pose=raw_device,
                final_pose=raw_device,
            )
            return {
                "pose": raw_pose,
                "candidates": candidates,
                "accepted_edges": [],
                "rejected_edges": rejected,
                "frame_diagnostic": {
                    "frame_id": int(pair_specs[0].current.frame_id),
                    "role": "online",
                    "optimization_attempted": True,
                    "accepted": False,
                    "reason": "too_few_initial_object_matches",
                    "observation_count": int(
                        len({int(pair.current.instance_id) for pair in pair_specs})
                    ),
                    "candidate_pair_count": int(len(pair_specs)),
                    "accepted_edge_count": 0,
                    "initial_match_count": int(initial_count),
                    "final_match_count": int(initial_count),
                },
            }

        initial_loss = self._loss_from_matches(
            raw_device,
            raw_device,
            pairs,
            initial_sets,
            include_prior=False,
        )
        best_delta = delta.detach().clone()
        best_loss = float(initial_loss)
        optimizer = torch.optim.Adam([delta], lr=float(self.config.learning_rate))

        for _ in range(int(self.config.outer_iterations)):
            match_sets = self._collect_match_sets(
                self._left_updated_pose(delta.detach(), raw_device),
                pairs,
            )
            match_count = sum(int(match.current_indices.numel()) for match in match_sets)
            if match_count < int(self.config.min_total_matches):
                break
            for _ in range(int(self.config.optimizer_steps)):
                optimizer.zero_grad(set_to_none=True)
                loss = self._loss_from_matches(
                    delta,
                    raw_device,
                    pairs,
                    match_sets,
                    include_prior=True,
                )
                if not bool(torch.isfinite(loss)):
                    break
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    _clamp_pose_delta(delta, self.config)
            candidate_pose = self._left_updated_pose(delta.detach(), raw_device)
            candidate_sets = self._collect_match_sets(candidate_pose, pairs)
            candidate_count = sum(
                int(match.current_indices.numel()) for match in candidate_sets
            )
            if candidate_count < int(self.config.min_total_matches):
                continue
            candidate_loss = self._loss_from_matches(
                candidate_pose,
                raw_device,
                pairs,
                candidate_sets,
                include_prior=False,
            )
            candidate_loss_value = float(candidate_loss)
            if math.isfinite(candidate_loss_value) and candidate_loss_value < best_loss:
                best_loss = candidate_loss_value
                best_delta = delta.detach().clone()

        best_pose = self._left_updated_pose(best_delta, raw_device)
        final_sets = self._collect_match_sets(best_pose, pairs)
        final_count = sum(int(match.current_indices.numel()) for match in final_sets)
        improvement = (float(initial_loss) - float(best_loss)) / max(
            float(initial_loss), 1e-6
        )
        final_pair_fraction = len(final_sets) / max(len(initial_sets), 1)
        accepted = (
            final_count >= int(self.config.min_total_matches)
            and final_pair_fraction >= 0.50
            and improvement >= float(self.config.min_relative_loss_improvement)
        )
        frame_id = int(pair_specs[0].current.frame_id)
        if not accepted:
            candidates, rejected = self._rows_for_rejection(
                pair_specs,
                initial_sets,
                reason="reject:insufficient_object_loss_improvement",
                initial_pose=raw_device,
                final_pose=best_pose,
                final_sets=final_sets,
                final_loss=float(best_loss),
            )
            return {
                "pose": raw_pose,
                "candidates": candidates,
                "accepted_edges": [],
                "rejected_edges": rejected,
                "frame_diagnostic": {
                    "frame_id": frame_id,
                    "role": "online",
                    "optimization_attempted": True,
                    "accepted": False,
                    "reason": "insufficient_object_loss_improvement",
                    "observation_count": int(
                        len({int(pair.current.instance_id) for pair in pair_specs})
                    ),
                    "candidate_pair_count": int(len(pair_specs)),
                    "accepted_edge_count": 0,
                    "initial_match_count": int(initial_count),
                    "final_match_count": int(final_count),
                    "initial_loss_m": float(initial_loss),
                    "final_loss_m": float(best_loss),
                    "relative_loss_improvement": float(improvement),
                },
            }

        candidates: list[dict[str, Any]] = []
        accepted_edges: list[ObjectLossEdge] = []
        rejected_edges: list[dict[str, Any]] = []
        initial_by_pair = {int(match.pair_index): match for match in initial_sets}
        final_by_pair = {int(match.pair_index): match for match in final_sets}
        for pair_index, pair in enumerate(pair_specs):
            initial_match = initial_by_pair.get(pair_index)
            final_match = final_by_pair.get(pair_index)
            initial_pair_loss = self._pair_loss(
                raw_device,
                pairs[pair_index],
                initial_match,
            )
            final_pair_loss = self._pair_loss(
                best_pose,
                pairs[pair_index],
                final_match,
            )
            initial_matches = (
                0 if initial_match is None else int(initial_match.current_indices.numel())
            )
            final_matches = (
                0 if final_match is None else int(final_match.current_indices.numel())
            )
            candidate = _candidate_row(
                pair,
                initial_matches=initial_matches,
                final_matches=final_matches,
                initial_loss=initial_pair_loss,
                final_loss=final_pair_loss,
            )
            if final_matches >= int(self.config.min_matches_per_pair):
                candidate["accepted"] = True
                candidates.append(candidate)
                accepted_edges.append(
                    ObjectLossEdge(
                        frame_i=int(pair.reference.frame_id),
                        frame_j=int(pair.current.frame_id),
                        instance_id=int(pair.current.instance_id),
                        category=str(pair.current.category),
                        reference_role=str(pair.reference.role),
                        num_matches_initial=initial_matches,
                        num_matches_final=final_matches,
                        initial_loss_m=float(initial_pair_loss),
                        final_loss_m=float(final_pair_loss),
                        relative_loss_improvement=(
                            float(initial_pair_loss) - float(final_pair_loss)
                        )
                        / max(float(initial_pair_loss), 1e-6),
                        edge_weight=float(pair.weight),
                        provenance=candidate,
                    )
                )
            else:
                candidate["accepted"] = False
                candidate["reason"] = "reject:too_few_final_object_matches"
                rejected_edges.append(candidate)
                candidates.append(candidate)
        return {
            "pose": best_pose.detach().float().cpu(),
            "candidates": candidates,
            "accepted_edges": accepted_edges,
            "rejected_edges": rejected_edges,
            "frame_diagnostic": {
                "frame_id": frame_id,
                "role": "online",
                "optimization_attempted": True,
                "accepted": bool(accepted_edges),
                "reason": "object_loss_accepted" if accepted_edges else "no_pair_survived",
                "observation_count": int(
                    len({int(pair.current.instance_id) for pair in pair_specs})
                ),
                "candidate_pair_count": int(len(pair_specs)),
                "accepted_edge_count": int(len(accepted_edges)),
                "initial_match_count": int(initial_count),
                "final_match_count": int(final_count),
                "initial_loss_m": float(initial_loss),
                "final_loss_m": float(best_loss),
                "relative_loss_improvement": float(improvement),
            },
        }

    def _left_updated_pose(
        self,
        delta: torch.Tensor,
        raw_pose: torch.Tensor,
    ) -> torch.Tensor:
        omega = delta[:3]
        rotation = _so3_exp(omega)
        translation = delta[3:]
        upper = torch.cat((rotation, translation[:, None]), dim=1)
        bottom = torch.tensor(
            [[0.0, 0.0, 0.0, 1.0]],
            dtype=delta.dtype,
            device=delta.device,
        )
        update = torch.cat((upper, bottom), dim=0)
        return update @ raw_pose

    def _collect_match_sets(
        self,
        pose: torch.Tensor,
        pairs: Sequence[_TensorPair],
    ) -> tuple[_MatchSet, ...]:
        output: list[_MatchSet] = []
        for pair_index, pair in enumerate(pairs):
            current_world = _transform_points(pair.current_points, pose)
            current_indices, reference_indices, weights = _mutual_matches(
                current_world,
                pair.reference_points,
                pair.current_weights,
                pair.reference_weights,
                max_distance=float(self.config.max_match_distance_m),
                trim_ratio=float(self.config.trim_ratio),
                min_matches=int(self.config.min_matches_per_pair),
                pair_weight=float(pair.pair_weight),
            )
            if current_indices.numel():
                output.append(
                    _MatchSet(
                        pair_index=int(pair_index),
                        current_indices=current_indices,
                        reference_indices=reference_indices,
                        weights=weights,
                    )
                )
        return tuple(output)

    def _loss_from_matches(
        self,
        pose_or_delta: torch.Tensor,
        raw_pose: torch.Tensor,
        pairs: Sequence[_TensorPair],
        matches: Sequence[_MatchSet],
        *,
        include_prior: bool,
    ) -> torch.Tensor:
        if pose_or_delta.numel() == 6:
            pose = self._left_updated_pose(pose_or_delta, raw_pose)
            delta = pose_or_delta
        else:
            pose = pose_or_delta
            delta = None
        losses: list[torch.Tensor] = []
        weights: list[torch.Tensor] = []
        for match in matches:
            pair = pairs[int(match.pair_index)]
            current_world = _transform_points(pair.current_points, pose)
            residual = current_world.index_select(0, match.current_indices) - pair.reference_points.index_select(
                0, match.reference_indices
            )
            distance = torch.linalg.vector_norm(residual, dim=-1)
            losses.append(_huber_distance(distance, float(self.config.huber_delta_m)))
            weights.append(match.weights)
        if losses:
            values = torch.cat(losses)
            value_weights = torch.cat(weights).clamp_min(1e-8)
            loss = (values * value_weights).sum() / value_weights.sum()
        else:
            loss = torch.zeros((), dtype=torch.float32, device=raw_pose.device)
        if include_prior and delta is not None and float(self.config.pose_prior_weight) > 0.0:
            rotation_scale = math.radians(float(self.config.max_correction_rotation_deg))
            translation_scale = float(self.config.max_correction_translation_m)
            prior = (
                (delta[:3] / rotation_scale).square().mean()
                + (delta[3:] / translation_scale).square().mean()
            )
            loss = loss + float(self.config.pose_prior_weight) * prior
        return loss

    def _pair_loss(
        self,
        pose: torch.Tensor,
        pair: _TensorPair,
        match: _MatchSet | None,
    ) -> float:
        if match is None or not match.current_indices.numel():
            return float("inf")
        with torch.no_grad():
            current_world = _transform_points(pair.current_points, pose)
            residual = current_world.index_select(0, match.current_indices) - pair.reference_points.index_select(
                0, match.reference_indices
            )
            distance = torch.linalg.vector_norm(residual, dim=-1)
            values = _huber_distance(distance, float(self.config.huber_delta_m))
            weights = match.weights.clamp_min(1e-8)
            return float((values * weights).sum() / weights.sum())

    def _rows_for_rejection(
        self,
        pair_specs: Sequence[_PairSpec],
        initial_sets: Sequence[_MatchSet],
        *,
        reason: str,
        initial_pose: torch.Tensor,
        final_pose: torch.Tensor,
        final_sets: Sequence[_MatchSet] = (),
        final_loss: float | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        initial_by_pair = {int(match.pair_index): match for match in initial_sets}
        final_by_pair = {int(match.pair_index): match for match in final_sets}
        candidates: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        for pair_index, pair in enumerate(pair_specs):
            initial_match = initial_by_pair.get(pair_index)
            final_match = final_by_pair.get(pair_index)
            initial_pair = _TensorPair(
                current_points=pair.current.points_camera.to(self.device),
                current_weights=pair.current.weights.to(self.device),
                reference_points=pair.reference.points_world.to(self.device),
                reference_weights=pair.reference.weights.to(self.device),
                pair_weight=float(pair.weight),
            )
            initial_pair_loss = self._pair_loss(initial_pose, initial_pair, initial_match)
            final_pair_loss = self._pair_loss(final_pose, initial_pair, final_match)
            row = _candidate_row(
                pair,
                initial_matches=(
                    0
                    if initial_match is None
                    else int(initial_match.current_indices.numel())
                ),
                final_matches=(
                    0 if final_match is None else int(final_match.current_indices.numel())
                ),
                initial_loss=initial_pair_loss,
                final_loss=final_pair_loss,
            )
            row["accepted"] = False
            row["reason"] = str(reason)
            if final_loss is not None:
                row["frame_final_loss_m"] = float(final_loss)
            candidates.append(row)
            rejected.append(row)
        return candidates, rejected


def _collect_observations(
    geometry: Sequence[GeometryFrame],
    segmentation_by_id: Mapping[int, SegmentationFrame],
    *,
    config: ObjectPoseLossRefinementConfig,
) -> tuple[dict[int, tuple[ObjectCloudObservation, ...]], set[int], Counter[str]]:
    observations_by_frame: dict[int, tuple[ObjectCloudObservation, ...]] = {}
    tracked_ids: set[int] = set()
    filter_stats: Counter[str] = Counter()
    for frame in geometry:
        frame_id = int(frame.frame_id)
        segmentation = segmentation_by_id[frame_id]
        camera_points, valid, confidence = _camera_geometry(frame)
        output: list[ObjectCloudObservation] = []
        height, width = frame.image_size
        for observation in segmentation.observations:
            instance_id = int(observation.instance_id)
            tracked_ids.add(instance_id)
            score = float(observation.score)
            if score < float(config.min_track_score):
                filter_stats["low_track_score"] += 1
                continue
            static = (
                observation.static_score is not None
                and float(observation.static_score) >= float(config.static_score_threshold)
            )
            if observation.static_score is None:
                static = not bool(config.require_static_score)
            if not static:
                filter_stats["dynamic_or_missing_static_score"] += 1
                continue
            mask = resize_bool_mask(observation.mask, frame.image_size)
            mask_pixels = int(mask.sum())
            area_ratio = float(mask_pixels) / float(height * width)
            if mask_pixels < int(config.min_mask_pixels):
                filter_stats["too_few_mask_pixels"] += 1
                continue
            if area_ratio > float(config.max_mask_area_ratio):
                filter_stats["mask_too_large"] += 1
                continue
            selected = mask & valid & (confidence >= float(config.min_geometry_confidence))
            pixels = selected.nonzero(as_tuple=False)
            if int(pixels.shape[0]) < int(config.min_points_per_observation):
                filter_stats["too_few_geometry_points"] += 1
                continue
            points = camera_points[pixels[:, 0], pixels[:, 1]].float()
            weights = (confidence[pixels[:, 0], pixels[:, 1]] * score).float()
            if int(points.shape[0]) > int(config.max_points_per_observation):
                order = torch.argsort(weights, descending=True, stable=True)[
                    : int(config.max_points_per_observation)
                ]
                points = points.index_select(0, order)
                weights = weights.index_select(0, order)
            geometry_confidence = float(weights.mean() / max(score, 1e-6))
            output.append(
                ObjectCloudObservation(
                    frame_id=frame_id,
                    instance_id=instance_id,
                    category=str(observation.category),
                    points_camera=points.detach().cpu(),
                    weights=weights.detach().cpu().clamp_min(1e-6),
                    track_score=score,
                    geometry_confidence=max(0.0, min(1.0, geometry_confidence)),
                    mask_pixels=mask_pixels,
                    static=True,
                )
            )
        observations_by_frame[frame_id] = tuple(output)
    return observations_by_frame, tracked_ids, filter_stats


def _select_references(
    instance_id: int,
    anchors: Mapping[int, Sequence[_StoredObjectCloud]],
    history: Mapping[int, Sequence[_StoredObjectCloud]],
    *,
    config: ObjectPoseLossRefinementConfig,
) -> tuple[_StoredObjectCloud, ...]:
    output: list[_StoredObjectCloud] = []
    seen_frames: set[int] = set()
    for reference in anchors.get(int(instance_id), ()):
        if int(reference.frame_id) in seen_frames:
            continue
        output.append(reference)
        seen_frames.add(int(reference.frame_id))
    recent = list(history.get(int(instance_id), ()))
    for reference in reversed(recent):
        if int(reference.frame_id) in seen_frames:
            continue
        output.append(reference)
        seen_frames.add(int(reference.frame_id))
        if len(output) >= int(config.max_anchor_observations) + int(
            config.max_history_observations
        ):
            break
    return tuple(output)


def _stored_cloud(
    observation: ObjectCloudObservation,
    pose: torch.Tensor,
    *,
    role: str,
) -> _StoredObjectCloud:
    points_world = _transform_points(observation.points_camera, pose)
    return _StoredObjectCloud(
        frame_id=int(observation.frame_id),
        instance_id=int(observation.instance_id),
        category=str(observation.category),
        points_world=points_world.detach().float().cpu(),
        weights=observation.weights.detach().float().cpu(),
        role=str(role),
        quality=math.sqrt(
            max(0.0, float(observation.track_score))
            * max(0.0, float(observation.geometry_confidence))
        ),
    )


def _mutual_matches(
    current_points: torch.Tensor,
    reference_points: torch.Tensor,
    current_weights: torch.Tensor,
    reference_weights: torch.Tensor,
    *,
    max_distance: float,
    trim_ratio: float,
    min_matches: int,
    pair_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not current_points.numel() or not reference_points.numel():
        empty = torch.empty(0, dtype=torch.long, device=current_points.device)
        return empty, empty, torch.empty(0, dtype=torch.float32, device=current_points.device)
    with torch.no_grad():
        distances = torch.cdist(current_points, reference_points)
        forward_distance, forward_index = distances.min(dim=1)
        backward_index = distances.argmin(dim=0)
        current_index = torch.arange(
            int(current_points.shape[0]),
            dtype=torch.long,
            device=current_points.device,
        )
        mutual = backward_index.index_select(0, forward_index) == current_index
        keep = mutual & (forward_distance <= float(max_distance))
        selected = keep.nonzero(as_tuple=False).flatten()
        if int(selected.numel()) < int(min_matches):
            empty = torch.empty(0, dtype=torch.long, device=current_points.device)
            return empty, empty, torch.empty(
                0, dtype=torch.float32, device=current_points.device
            )
        order = torch.argsort(
            forward_distance.index_select(0, selected),
            descending=False,
            stable=True,
        )
        keep_count = max(int(min_matches), int(math.ceil(float(trim_ratio) * selected.numel())))
        keep_count = min(keep_count, int(selected.numel()))
        selected = selected.index_select(0, order[:keep_count])
        matched_reference = forward_index.index_select(0, selected)
        weights = torch.sqrt(
            current_weights.index_select(0, selected).clamp_min(1e-8)
            * reference_weights.index_select(0, matched_reference).clamp_min(1e-8)
        ) * float(pair_weight)
        return selected, matched_reference, weights.float()


def _candidate_row(
    pair: _PairSpec,
    *,
    initial_matches: int,
    final_matches: int,
    initial_loss: float,
    final_loss: float,
) -> dict[str, Any]:
    improvement = (
        (float(initial_loss) - float(final_loss)) / max(float(initial_loss), 1e-6)
        if math.isfinite(float(initial_loss)) and math.isfinite(float(final_loss))
        else 0.0
    )
    return {
        "frame_i": int(pair.reference.frame_id),
        "frame_j": int(pair.current.frame_id),
        "temporal_gap": int(pair.current.frame_id - pair.reference.frame_id),
        "instance_id": int(pair.current.instance_id),
        "category": str(pair.current.category),
        "reference_role": str(pair.reference.role),
        "track_score": float(pair.current.track_score),
        "geometry_confidence": float(pair.current.geometry_confidence),
        "mask_pixels": int(pair.current.mask_pixels),
        "reference_quality": float(pair.reference.quality),
        "edge_weight": float(pair.weight),
        "num_matches_initial": int(initial_matches),
        "num_matches_final": int(final_matches),
        "initial_loss_m": float(initial_loss),
        "final_loss_m": float(final_loss),
        "relative_loss_improvement": float(improvement),
    }


def _camera_geometry(
    frame: GeometryFrame,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return fixed camera-space points, validity, and confidence."""

    frame.validate()
    height, width = frame.image_size
    confidence = geometry_confidence_for_frame(frame)
    if frame.depth is not None:
        depth = frame.depth.detach().float().cpu()
        if depth.ndim == 3:
            depth = depth[..., 0]
        if frame.intrinsics is None:
            raise ValueError("Object loss refinement needs intrinsics with depth.")
        intrinsics = frame.intrinsics.detach().float().cpu()
        yy, xx = torch.meshgrid(
            torch.arange(height, dtype=depth.dtype),
            torch.arange(width, dtype=depth.dtype),
            indexing="ij",
        )
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        if abs(float(fx)) <= 1e-8 or abs(float(fy)) <= 1e-8:
            raise ValueError("Object loss refinement received zero focal length.")
        points = torch.stack(
            (
                (xx - intrinsics[0, 2]) / fx * depth,
                (yy - intrinsics[1, 2]) / fy * depth,
                depth,
            ),
            dim=-1,
        )
        valid = torch.isfinite(points).all(dim=-1) & torch.isfinite(depth) & (depth > 0.0)
    else:
        if frame.world_points is None or frame.camera_to_world is None:
            raise ValueError(
                "Object loss refinement needs depth/intrinsics or world_points/raw pose."
            )
        world = frame.world_points.detach().float().cpu()
        homogeneous = torch.cat(
            (world, torch.ones(*world.shape[:-1], 1, dtype=world.dtype)), dim=-1
        )
        points = torch.einsum(
            "ij,hwj->hwi", _invert_pose(_pose4(frame.camera_to_world)), homogeneous
        )[..., :3]
        valid = torch.isfinite(points).all(dim=-1)
    if frame.valid is not None:
        valid &= frame.valid.detach().bool().cpu()
    return points, valid, confidence


def _transform_points(points: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(points)
    transform = torch.as_tensor(pose)
    if tuple(transform.shape) == (3, 4):
        bottom = torch.zeros(
            (1, 4), dtype=transform.dtype, device=transform.device
        )
        bottom[0, 3] = 1.0
        transform = torch.cat((transform, bottom), dim=0)
    elif tuple(transform.shape) != (4, 4):
        raise ValueError("Pose must have shape [3,4] or [4,4].")
    transform = transform.to(device=value.device, dtype=value.dtype)
    return value @ transform[:3, :3].transpose(0, 1) + transform[:3, 3]


def _pose4(pose: torch.Tensor | None) -> torch.Tensor:
    if pose is None:
        raise ValueError("Object loss refinement requires camera_to_world poses.")
    value = torch.as_tensor(pose).detach().float()
    if tuple(value.shape) == (3, 4):
        output = torch.eye(4, dtype=value.dtype, device=value.device)
        output[:3] = value
        return output
    if tuple(value.shape) == (4, 4):
        return value
    raise ValueError("camera_to_world pose must have shape [3,4] or [4,4].")


def _invert_pose(pose: torch.Tensor) -> torch.Tensor:
    value = _pose4(pose)
    output = torch.eye(4, dtype=value.dtype, device=value.device)
    rotation = value[:3, :3]
    output[:3, :3] = rotation.transpose(0, 1)
    output[:3, 3] = -rotation.transpose(0, 1) @ value[:3, 3]
    return output


def _so3_exp(vector: torch.Tensor) -> torch.Tensor:
    theta2 = (vector * vector).sum()
    theta = torch.sqrt(theta2.clamp_min(1e-12))
    skew = torch.stack(
        (
            torch.stack((torch.zeros_like(vector[0]), -vector[2], vector[1])),
            torch.stack((vector[2], torch.zeros_like(vector[0]), -vector[0])),
            torch.stack((-vector[1], vector[0], torch.zeros_like(vector[0]))),
        )
    )
    small = theta2 < 1e-6
    a = torch.where(
        small,
        1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0,
        torch.sin(theta) / theta,
    )
    b = torch.where(
        small,
        0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0,
        (1.0 - torch.cos(theta)) / theta2.clamp_min(1e-12),
    )
    identity = torch.eye(3, dtype=vector.dtype, device=vector.device)
    return identity + a * skew + b * (skew @ skew)


def _clamp_pose_delta(
    delta: torch.Tensor,
    config: ObjectPoseLossRefinementConfig,
) -> None:
    rotation_limit = math.radians(float(config.max_correction_rotation_deg))
    translation_limit = float(config.max_correction_translation_m)
    rotation_norm = torch.linalg.vector_norm(delta[:3])
    if float(rotation_norm) > rotation_limit:
        delta[:3].mul_(rotation_limit / float(rotation_norm))
    translation_norm = torch.linalg.vector_norm(delta[3:])
    if float(translation_norm) > translation_limit:
        delta[3:].mul_(translation_limit / float(translation_norm))


def _huber_distance(distance: torch.Tensor, delta: float) -> torch.Tensor:
    threshold = float(delta)
    return torch.where(
        distance <= threshold,
        0.5 * distance.square(),
        threshold * (distance - 0.5 * threshold),
    )


def _rotation_angle_deg(rotation: torch.Tensor) -> float:
    value = torch.as_tensor(rotation).detach().float().cpu()
    cosine = torch.clamp((torch.trace(value) - 1.0) * 0.5, -1.0, 1.0)
    return math.degrees(float(torch.acos(cosine)))


def _mean_or_none(values: Sequence[float]) -> float | None:
    return None if not values else float(sum(values) / len(values))


def _mean_or_zero(values: Sequence[float]) -> float:
    return 0.0 if not values else float(sum(values) / len(values))


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


__all__ = [
    "ObjectCloudObservation",
    "ObjectLossEdge",
    "ObjectPoseLossRefinementConfig",
    "ObjectPoseLossRefiner",
]
