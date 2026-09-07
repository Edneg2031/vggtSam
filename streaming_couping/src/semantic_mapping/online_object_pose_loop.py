"""Causal external SAM/HorizonStream pose-correction loop.

This module intentionally sits downstream of HorizonStream.  It consumes
canonical geometry and causal SAM observations one frame at a time, keeps a
small corrected-pose window, and optimizes that window against static-object
point-cloud constraints.  It does *not* write a corrected pose into
HorizonStream's latent/cache state; the corrected trajectory is an external
world-frame state used by subsequent object matching and map fusion.

The implementation is deliberately conservative:

* HorizonStream remains the pose prior.
* Correspondences are fixed during each optimizer pass and rematched only
  between passes.
* Object losses are averaged per constraint so a large object cannot dominate.
* A correction is accepted only when independent-instance and validation gates
  pass; otherwise the predicted corrected pose is retained unchanged.
* Only earlier frames are eligible as references.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from typing import Any

import torch

from .contracts import GeometryFrame, SegmentationFrame
from .object_pose_loss_refinement import (
    ObjectCloudObservation,
    ObjectLossEdge,
    ObjectPoseLossRefinementConfig,
    _collect_observations,
    _huber_distance,
    _invert_pose,
    _mean_or_none,
    _mutual_matches,
    _pose4,
    _rotation_angle_deg,
    _so3_exp,
    _transform_points,
)
from .object_pose_refinement import PoseRefinementResult


@dataclass(frozen=True)
class OnlineObjectPoseLoopConfig:
    """Settings for the external, causal sliding-window pose loop."""

    # The nested configuration owns mask/depth/track filtering and the basic
    # point-cloud matching thresholds.  It is kept separate so the existing
    # offline loss refiner and this online loop can share those settings.
    observation_config: ObjectPoseLossRefinementConfig = field(
        default_factory=ObjectPoseLossRefinementConfig
    )

    window_size: int = 10
    max_reference_frames: int = 5
    max_reference_gap: int = 10
    anchor_frame_count: int = 3

    min_independent_instances: int = 2
    min_matches_per_instance: int = 15
    min_total_matches: int = 48

    rematch_iterations: int = 2
    optimizer_steps: int = 50
    learning_rate: float = 0.01
    huber_delta_m: float = 0.05
    pose_prior_weight: float = 1.0
    temporal_edge_weight: float = 1.0
    correction_smoothness_weight: float = 0.25
    rotation_residual_scale_m: float = 0.50

    max_local_correction_rotation_deg: float = 1.0
    max_local_correction_translation_m: float = 0.03
    min_relative_loss_improvement: float = 0.10
    max_validation_residual_m: float = 0.10

    device: str = "cpu"
    trace_optimization: bool = False

    def validate(self) -> "OnlineObjectPoseLoopConfig":
        self.observation_config.validate()
        for name, value in (
            ("window_size", self.window_size),
            ("max_reference_frames", self.max_reference_frames),
            ("max_reference_gap", self.max_reference_gap),
            ("anchor_frame_count", self.anchor_frame_count),
            ("min_independent_instances", self.min_independent_instances),
            ("min_matches_per_instance", self.min_matches_per_instance),
            ("min_total_matches", self.min_total_matches),
            ("rematch_iterations", self.rematch_iterations),
            ("optimizer_steps", self.optimizer_steps),
        ):
            if int(value) < 1:
                raise ValueError(f"online_object_pose.{name} must be positive.")
        if int(self.min_total_matches) < int(self.min_matches_per_instance):
            raise ValueError(
                "online_object_pose.min_total_matches cannot be smaller than "
                "min_matches_per_instance."
            )
        for name, value in (
            ("learning_rate", self.learning_rate),
            ("huber_delta_m", self.huber_delta_m),
            ("rotation_residual_scale_m", self.rotation_residual_scale_m),
            ("max_local_correction_rotation_deg", self.max_local_correction_rotation_deg),
            ("max_local_correction_translation_m", self.max_local_correction_translation_m),
            ("max_validation_residual_m", self.max_validation_residual_m),
        ):
            if float(value) <= 0.0:
                raise ValueError(f"online_object_pose.{name} must be positive.")
        for name, value in (
            ("pose_prior_weight", self.pose_prior_weight),
            ("temporal_edge_weight", self.temporal_edge_weight),
            ("correction_smoothness_weight", self.correction_smoothness_weight),
        ):
            if float(value) < 0.0:
                raise ValueError(f"online_object_pose.{name} cannot be negative.")
        if not 0.0 <= float(self.min_relative_loss_improvement) <= 1.0:
            raise ValueError(
                "online_object_pose.min_relative_loss_improvement must be in [0,1]."
            )
        if not str(self.device).strip():
            raise ValueError("online_object_pose.device must not be empty.")
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_config": self.observation_config.to_dict(),
            "window_size": int(self.window_size),
            "max_reference_frames": int(self.max_reference_frames),
            "max_reference_gap": int(self.max_reference_gap),
            "anchor_frame_count": int(self.anchor_frame_count),
            "min_independent_instances": int(self.min_independent_instances),
            "min_matches_per_instance": int(self.min_matches_per_instance),
            "min_total_matches": int(self.min_total_matches),
            "rematch_iterations": int(self.rematch_iterations),
            "optimizer_steps": int(self.optimizer_steps),
            "learning_rate": float(self.learning_rate),
            "huber_delta_m": float(self.huber_delta_m),
            "pose_prior_weight": float(self.pose_prior_weight),
            "temporal_edge_weight": float(self.temporal_edge_weight),
            "correction_smoothness_weight": float(self.correction_smoothness_weight),
            "rotation_residual_scale_m": float(self.rotation_residual_scale_m),
            "max_local_correction_rotation_deg": float(
                self.max_local_correction_rotation_deg
            ),
            "max_local_correction_translation_m": float(
                self.max_local_correction_translation_m
            ),
            "min_relative_loss_improvement": float(
                self.min_relative_loss_improvement
            ),
            "max_validation_residual_m": float(self.max_validation_residual_m),
            "device": str(self.device),
            "trace_optimization": bool(self.trace_optimization),
        }


@dataclass(frozen=True)
class _ObjectConstraint:
    frame_i: int
    frame_j: int
    instance_id: int
    category: str
    reference: ObjectCloudObservation
    current: ObjectCloudObservation
    current_indices: torch.Tensor
    reference_indices: torch.Tensor
    weights: torch.Tensor
    pair_weight: float
    reference_role: str


class OnlineObjectPoseLoopRefiner:
    """Run a causal external corrected-pose loop over canonical frames.

    The input may come from a frozen geometry cache for a deterministic
    ablation.  Causality is preserved by processing frames in order and only
    using observations from earlier frame IDs.  A native streaming provider
    can later call the same state machine one frame at a time.
    """

    method_name = "sam_instance_guided_external_online_sliding_window_pose_graph"

    def __init__(
        self,
        config: OnlineObjectPoseLoopConfig | None = None,
    ) -> None:
        self.config = (config or OnlineObjectPoseLoopConfig()).validate()
        self.device = torch.device(self.config.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"online_object_pose requested {self.device}, but CUDA is unavailable."
            )

    def refine(
        self,
        geometry_frames: Sequence[GeometryFrame],
        segmentation_frames: Sequence[SegmentationFrame],
        image_paths: Sequence[str] | Sequence[Any],
    ) -> PoseRefinementResult:
        """Process frames causally and return raw/corrected trajectories."""

        del image_paths
        geometry = tuple(geometry_frames)
        segmentation = tuple(segmentation_frames)
        if not geometry:
            raise ValueError("OnlineObjectPoseLoopRefiner requires geometry frames.")
        if len(geometry) != len(segmentation):
            raise ValueError("Geometry and segmentation counts must agree.")
        frame_ids = tuple(int(frame.frame_id) for frame in geometry)
        if frame_ids != tuple(sorted(frame_ids)):
            raise ValueError("OnlineObjectPoseLoopRefiner requires increasing frame IDs.")
        if len(set(frame_ids)) != len(frame_ids):
            raise ValueError("OnlineObjectPoseLoopRefiner requires unique frame IDs.")
        segmentation_by_id = {int(frame.frame_id): frame for frame in segmentation}
        if len(segmentation_by_id) != len(segmentation):
            raise ValueError("Segmentation frames contain duplicate frame IDs.")
        if set(segmentation_by_id) != set(frame_ids):
            raise ValueError("Geometry and segmentation frame IDs do not match.")

        observations_by_frame, tracked_ids, filter_stats = _collect_observations(
            geometry,
            segmentation_by_id,
            config=self.config.observation_config,
        )
        raw_poses = {
            int(frame_id): _pose4(frame.camera_to_world)
            for frame_id, frame in zip(frame_ids, geometry)
        }
        corrected_poses: dict[int, torch.Tensor] = {}
        frame_positions = {int(frame_id): index for index, frame_id in enumerate(frame_ids)}
        anchors = set(frame_ids[: int(self.config.anchor_frame_count)])
        processed_ids: list[int] = []

        candidates: list[dict[str, Any]] = []
        accepted_edges: list[ObjectLossEdge] = []
        rejected_edges: list[dict[str, Any]] = []
        frame_diagnostics: list[dict[str, Any]] = []
        optimization_trace: list[dict[str, Any]] = []

        for frame_id in frame_ids:
            frame_id = int(frame_id)
            raw_pose = raw_poses[frame_id]
            current_observations = observations_by_frame.get(frame_id, ())
            is_anchor = frame_id in anchors

            if is_anchor or not processed_ids:
                corrected_poses[frame_id] = raw_pose
                frame_diagnostics.append(
                    self._frame_diagnostic(
                        frame_id,
                        is_anchor=is_anchor,
                        observation_count=len(current_observations),
                        reason="early_anchor_frame" if is_anchor else "no_history",
                    )
                )
                if self.config.trace_optimization:
                    optimization_trace.append(
                        self._trace_row(
                            frame_id,
                            phase="frame",
                            status="anchor" if is_anchor else "no_history",
                            observation_count=len(current_observations),
                        )
                    )
                processed_ids.append(frame_id)
                continue

            predicted_pose = self._predicted_pose(
                frame_id,
                raw_poses,
                corrected_poses,
                processed_ids,
            )
            reference_ids = self._reference_ids(
                frame_id,
                current_observations,
                processed_ids,
                observations_by_frame,
                anchors,
                frame_positions,
            )
            variable_ids = self._variable_ids(
                frame_id,
                processed_ids,
                anchors,
            )
            base_poses = {
                node_id: (
                    predicted_pose if node_id == frame_id else corrected_poses[node_id]
                )
                for node_id in variable_ids
            }
            base_poses[frame_id] = predicted_pose
            working_poses = dict(corrected_poses)
            working_poses.update(base_poses)

            if not reference_ids:
                corrected_poses[frame_id] = predicted_pose
                reason = "no_recent_instance_reference"
                frame_diagnostics.append(
                    self._frame_diagnostic(
                        frame_id,
                        is_anchor=False,
                        observation_count=len(current_observations),
                        reason=reason,
                    )
                )
                if self.config.trace_optimization:
                    optimization_trace.append(
                        self._trace_row(
                            frame_id,
                            phase="frame",
                            status=reason,
                            observation_count=len(current_observations),
                        )
                    )
                processed_ids.append(frame_id)
                continue

            best_poses = dict(working_poses)
            best_constraints: tuple[_ObjectConstraint, ...] = ()
            best_loss = float("inf")
            initial_loss = float("inf")
            initial_constraints: tuple[_ObjectConstraint, ...] = ()
            trace_rows: list[dict[str, Any]] = []
            accepted = False
            reject_reason = "no_valid_object_constraint"
            current_iteration_poses = dict(working_poses)

            for iteration in range(int(self.config.rematch_iterations)):
                constraints = self._build_constraints(
                    frame_id,
                    current_observations,
                    reference_ids,
                    observations_by_frame,
                    current_iteration_poses,
                    anchor_ids=anchors,
                )
                stats = self._constraint_stats(constraints)
                if iteration == 0:
                    initial_constraints = constraints
                    initial_loss = self._object_loss(current_iteration_poses, constraints)
                trace = self._trace_row(
                    frame_id,
                    phase="outer_iteration",
                    outer_iteration=iteration,
                    observation_count=len(current_observations),
                    reference_gap_min=(
                        min(frame_id - ref_id for ref_id in reference_ids)
                        if reference_ids
                        else None
                    ),
                    reference_gap_max=(
                        max(frame_id - ref_id for ref_id in reference_ids)
                        if reference_ids
                        else None
                    ),
                    match_pair_count=len(constraints),
                    match_count=stats["total_matches"],
                    independent_instance_count=stats["instance_count"],
                    start_loss=self._object_loss(current_iteration_poses, constraints),
                    status="not_run",
                )
                if not self._passes_match_gate(stats):
                    trace["status"] = "reject:insufficient_independent_matches"
                    trace_rows.append(trace)
                    reject_reason = "insufficient_independent_matches"
                    break

                candidate_poses, delta = self._optimize_window(
                    current_iteration_poses,
                    variable_ids,
                    constraints,
                    raw_poses,
                )
                candidate_constraints = self._build_constraints(
                    frame_id,
                    current_observations,
                    reference_ids,
                    observations_by_frame,
                    candidate_poses,
                    anchor_ids=anchors,
                )
                candidate_stats = self._constraint_stats(candidate_constraints)
                fixed_loss = self._object_loss(candidate_poses, constraints)
                candidate_loss = self._object_loss(candidate_poses, candidate_constraints)
                common_loss = self._object_loss(candidate_poses, initial_constraints)
                correction = _invert_pose(predicted_pose) @ candidate_poses[frame_id]
                trace.update(
                    {
                        "candidate_match_pair_count": len(candidate_constraints),
                        "candidate_match_count": candidate_stats["total_matches"],
                        "candidate_independent_instance_count": candidate_stats[
                            "instance_count"
                        ],
                        "candidate_loss_m": _finite_or_none(candidate_loss),
                        "fixed_correspondence_loss_m": _finite_or_none(fixed_loss),
                        "common_initial_correspondence_loss_m": _finite_or_none(
                            common_loss
                        ),
                        "delta_rotation_deg": _rotation_angle_deg(correction[:3, :3]),
                        "delta_translation_m": float(
                            torch.linalg.vector_norm(correction[:3, 3])
                        ),
                    }
                )
                # Use the first correspondence set as the common acceptance
                # objective.  The rematched loss is still useful for
                # diagnostics/validation, but comparing it across outer
                # iterations would allow a change in correspondence to look
                # like geometric improvement by itself.
                improvement = _relative_improvement(initial_loss, common_loss)
                valid = self._passes_candidate_gate(
                    candidate_stats,
                    candidate_poses,
                    frame_id,
                    predicted_pose,
                    improvement,
                    candidate_constraints,
                )
                if valid and common_loss < best_loss:
                    best_loss = common_loss
                    best_poses = candidate_poses
                    best_constraints = candidate_constraints
                    current_iteration_poses = candidate_poses
                    accepted = True
                    trace["status"] = "best_updated"
                else:
                    trace["status"] = (
                        "reject:candidate_gate" if not valid else "no_improvement"
                    )
                    if not valid:
                        reject_reason = self._candidate_rejection_reason(
                            candidate_stats,
                            improvement,
                            candidate_poses,
                            frame_id,
                            predicted_pose,
                            candidate_constraints,
                        )
                trace["best_loss_m_after"] = _finite_or_none(best_loss)
                trace_rows.append(trace)
                if not accepted:
                    break

            if accepted:
                # ``best_poses`` contains the complete pose dictionary so
                # object constraints can still evaluate historical
                # references.  Only the current sliding-window nodes are
                # allowed to change the committed trajectory; fixed older
                # nodes are deliberately left untouched.
                for node_id in variable_ids:
                    corrected_poses[int(node_id)] = best_poses[int(node_id)]
                final_constraints = self._build_constraints(
                    frame_id,
                    current_observations,
                    reference_ids,
                    observations_by_frame,
                    corrected_poses,
                    anchor_ids=anchors,
                )
                frame_diagnostics.append(
                    self._frame_diagnostic(
                        frame_id,
                        is_anchor=False,
                        observation_count=len(current_observations),
                        reason="online_pose_accepted",
                        attempted=True,
                        accepted=True,
                        candidate_pair_count=len(final_constraints),
                        match_count=self._constraint_stats(final_constraints)[
                            "total_matches"
                        ],
                        initial_loss=initial_loss,
                        final_loss=self._object_loss(corrected_poses, final_constraints),
                    )
                )
                self._append_edges(
                    accepted_edges,
                    candidates,
                    best_constraints,
                    initial_constraints,
                    corrected_poses,
                    initial_poses=working_poses,
                )
            else:
                corrected_poses[frame_id] = predicted_pose
                final_constraints = self._build_constraints(
                    frame_id,
                    current_observations,
                    reference_ids,
                    observations_by_frame,
                    corrected_poses,
                    anchor_ids=anchors,
                )
                reject_rows = self._rejection_rows(
                    final_constraints or initial_constraints,
                    initial_poses=working_poses,
                    final_poses=corrected_poses,
                    reason=f"reject:{reject_reason}",
                )
                candidates.extend(reject_rows)
                rejected_edges.extend(reject_rows)
                frame_diagnostics.append(
                    self._frame_diagnostic(
                        frame_id,
                        is_anchor=False,
                        observation_count=len(current_observations),
                        reason=f"reject:{reject_reason}",
                        attempted=True,
                        accepted=False,
                        candidate_pair_count=len(final_constraints),
                        match_count=self._constraint_stats(final_constraints)[
                            "total_matches"
                        ],
                        initial_loss=initial_loss,
                        final_loss=None,
                    )
                )

            if self.config.trace_optimization:
                optimization_trace.extend(trace_rows)
                final_pose = corrected_poses[frame_id]
                final_base_pose = (
                    predicted_pose
                    if not is_anchor
                    else raw_pose
                )
                final_constraints = self._build_constraints(
                    frame_id,
                    current_observations,
                    reference_ids,
                    observations_by_frame,
                    corrected_poses,
                    anchor_ids=anchors,
                )
                final_stats = self._constraint_stats(final_constraints)
                optimization_trace.append(
                    self._trace_row(
                        frame_id,
                        phase="final",
                        outer_iteration=int(self.config.rematch_iterations),
                        observation_count=len(current_observations),
                        reference_gap_min=(
                            min(frame_id - ref_id for ref_id in reference_ids)
                            if reference_ids
                            else None
                        ),
                        reference_gap_max=(
                            max(frame_id - ref_id for ref_id in reference_ids)
                            if reference_ids
                            else None
                        ),
                        match_pair_count=len(final_constraints),
                        match_count=final_stats["total_matches"],
                        independent_instance_count=final_stats["instance_count"],
                        loss_m=self._object_loss(corrected_poses, final_constraints),
                        carried_rotation_deg=_rotation_angle_deg(
                            (_invert_pose(raw_pose) @ predicted_pose)[:3, :3]
                        ),
                        carried_translation_m=float(
                            torch.linalg.vector_norm(
                                (_invert_pose(raw_pose) @ predicted_pose)[:3, 3]
                            )
                        ),
                        delta_rotation_deg=_rotation_angle_deg(
                            (_invert_pose(final_base_pose) @ final_pose)[:3, :3]
                        ),
                        delta_translation_m=float(
                            torch.linalg.vector_norm(
                                (_invert_pose(final_base_pose) @ final_pose)[:3, 3]
                            )
                        ),
                        status="accepted" if accepted else "rejected",
                    )
                )
            processed_ids.append(frame_id)

        refined_tuple = tuple(corrected_poses[int(frame_id)] for frame_id in frame_ids)
        raw_tuple = tuple(raw_poses[int(frame_id)] for frame_id in frame_ids)
        rotation_changes = []
        translation_changes = []
        for raw, refined in zip(raw_tuple, refined_tuple):
            delta = _invert_pose(raw) @ refined
            rotation_changes.append(_rotation_angle_deg(delta[:3, :3]))
            translation_changes.append(float(torch.linalg.vector_norm(delta[:3, 3])))
        accepted_frames = sum(
            1 for row in frame_diagnostics if bool(row.get("accepted"))
        )
        attempted_frames = sum(
            1 for row in frame_diagnostics if bool(row.get("optimization_attempted"))
        )
        summary = {
            "schema": 1,
            "revision": "sam_instance_guided_external_online_sliding_window_pose_graph_r1",
            "method": self.method_name,
            "enabled": True,
            "causal": True,
            "horizonstream_feedback": False,
            "config": self.config.to_dict(),
            "frame_count": len(frame_ids),
            "frame_ids": list(frame_ids),
            "tracked_instance_count": len(tracked_ids),
            "raw_observation_count": sum(
                len(segmentation_by_id[int(frame_id)].observations)
                for frame_id in frame_ids
            ),
            "retained_observation_count": sum(
                len(observations) for observations in observations_by_frame.values()
            ),
            "filtered_observation_count": sum(
                int(value) for value in filter_stats.values()
            ),
            "candidate_pair_count": len(candidates),
            "accepted_edge_count": len(accepted_edges),
            "rejected_edge_count": len(rejected_edges),
            "accepted_frame_count": int(accepted_frames),
            "optimization_attempted_frame_count": int(attempted_frames),
            "rejected_reason_counts": dict(
                Counter(str(row.get("reason", "unknown")) for row in rejected_edges)
            ),
            "loss_statistics": {
                "accepted_initial_mean_m": _mean_or_none(
                    [edge.initial_loss_m for edge in accepted_edges]
                ),
                "accepted_final_mean_m": _mean_or_none(
                    [edge.final_loss_m for edge in accepted_edges]
                ),
                "accepted_relative_improvement_mean": _mean_or_none(
                    [edge.relative_loss_improvement for edge in accepted_edges]
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
            "optimization_trace_enabled": bool(self.config.trace_optimization),
            "optimization_trace": optimization_trace,
            "optimizer": {
                "backend": "torch_adam_external_sliding_window_fixed_correspondence",
                "attempted": bool(attempted_frames),
                "success": bool(accepted_frames),
                "attempted_frame_count": int(attempted_frames),
                "accepted_frame_count": int(accepted_frames),
            },
        }
        return PoseRefinementResult(
            frame_ids=frame_ids,
            raw_camera_to_world=tuple(p.detach().float().cpu() for p in raw_tuple),
            refined_camera_to_world=tuple(
                p.detach().float().cpu() for p in refined_tuple
            ),
            candidates=tuple(candidates),
            accepted_edges=tuple(accepted_edges),
            rejected_edges=tuple(rejected_edges),
            summary=summary,
        )

    def _predicted_pose(
        self,
        frame_id: int,
        raw_poses: Mapping[int, torch.Tensor],
        corrected_poses: Mapping[int, torch.Tensor],
        processed_ids: Sequence[int],
    ) -> torch.Tensor:
        """Carry the last accepted external world alignment to the next raw pose."""

        if not processed_ids:
            return raw_poses[frame_id]
        previous_id = int(processed_ids[-1])
        raw_previous = raw_poses[previous_id]
        corrected_previous = corrected_poses[previous_id]
        alignment = corrected_previous @ _invert_pose(raw_previous)
        return alignment @ raw_poses[frame_id]

    def _reference_ids(
        self,
        frame_id: int,
        current: Sequence[ObjectCloudObservation],
        processed_ids: Sequence[int],
        observations_by_frame: Mapping[int, Sequence[ObjectCloudObservation]],
        anchors: set[int],
        frame_positions: Mapping[int, int],
    ) -> tuple[int, ...]:
        current_instances = {int(observation.instance_id) for observation in current}
        candidates: list[int] = []
        for reference_id in reversed(processed_ids):
            gap = int(frame_id - reference_id)
            if gap < 1 or gap > int(self.config.max_reference_gap):
                continue
            reference_instances = {
                int(observation.instance_id)
                for observation in observations_by_frame.get(reference_id, ())
            }
            if not current_instances.intersection(reference_instances):
                continue
            candidates.append(int(reference_id))
            if len(candidates) >= int(self.config.max_reference_frames):
                break
        # If a recent window has no overlap, allow a reliable early anchor only
        # within the same explicit temporal gap.  This prevents frame 0 from
        # influencing frame 99 indefinitely.
        for reference_id in sorted(anchors, key=lambda value: frame_positions[value], reverse=True):
            if reference_id in candidates:
                continue
            gap = int(frame_id - reference_id)
            if gap < 1 or gap > int(self.config.max_reference_gap):
                continue
            reference_instances = {
                int(observation.instance_id)
                for observation in observations_by_frame.get(reference_id, ())
            }
            if current_instances.intersection(reference_instances):
                candidates.append(int(reference_id))
            if len(candidates) >= int(self.config.max_reference_frames):
                break
        return tuple(candidates)

    def _variable_ids(
        self,
        frame_id: int,
        processed_ids: Sequence[int],
        anchors: set[int],
    ) -> tuple[int, ...]:
        history_count = max(0, int(self.config.window_size) - 1)
        recent = (
            list(processed_ids[-history_count:])
            if history_count > 0
            else []
        )
        values = [value for value in recent if value not in anchors]
        values.append(int(frame_id))
        return tuple(dict.fromkeys(values))

    def _build_constraints(
        self,
        frame_id: int,
        current: Sequence[ObjectCloudObservation],
        reference_ids: Sequence[int],
        observations_by_frame: Mapping[int, Sequence[ObjectCloudObservation]],
        poses: Mapping[int, torch.Tensor],
        *,
        anchor_ids: set[int] | None = None,
    ) -> tuple[_ObjectConstraint, ...]:
        current_by_instance = {
            int(observation.instance_id): observation for observation in current
        }
        constraints: list[_ObjectConstraint] = []
        for reference_id in reference_ids:
            reference_pose = poses.get(int(reference_id))
            current_pose = poses.get(int(frame_id))
            if reference_pose is None or current_pose is None:
                continue
            reference_by_instance = {
                int(observation.instance_id): observation
                for observation in observations_by_frame.get(int(reference_id), ())
            }
            for instance_id, current_observation in current_by_instance.items():
                reference_observation = reference_by_instance.get(instance_id)
                if reference_observation is None:
                    continue
                current_world = _transform_points(
                    current_observation.points_camera,
                    current_pose,
                )
                reference_world = _transform_points(
                    reference_observation.points_camera,
                    reference_pose,
                )
                pair_weight = max(
                    1e-6,
                    math.sqrt(
                        max(0.0, float(current_observation.track_score))
                        * max(0.0, float(current_observation.geometry_confidence))
                        * max(0.0, float(reference_observation.track_score))
                        * max(0.0, float(reference_observation.geometry_confidence))
                    ),
                )
                current_indices, reference_indices, weights = _mutual_matches(
                    current_world.to(self.device),
                    reference_world.to(self.device),
                    current_observation.weights.to(self.device),
                    reference_observation.weights.to(self.device),
                    max_distance=float(
                        self.config.observation_config.max_match_distance_m
                    ),
                    trim_ratio=float(self.config.observation_config.trim_ratio),
                    min_matches=int(
                        self.config.observation_config.min_matches_per_pair
                    ),
                    pair_weight=pair_weight,
                )
                if int(current_indices.numel()) < int(
                    self.config.min_matches_per_instance
                ):
                    continue
                constraints.append(
                    _ObjectConstraint(
                        frame_i=int(reference_id),
                        frame_j=int(frame_id),
                        instance_id=int(instance_id),
                        category=str(current_observation.category),
                        reference=reference_observation,
                        current=current_observation,
                        current_indices=current_indices.detach().cpu(),
                        reference_indices=reference_indices.detach().cpu(),
                        weights=weights.detach().cpu(),
                        pair_weight=pair_weight,
                        reference_role=(
                            "anchor"
                            if anchor_ids is not None
                            and int(reference_id) in anchor_ids
                            else "history"
                        ),
                    )
                )
        return tuple(constraints)

    def _optimize_window(
        self,
        base_poses: Mapping[int, torch.Tensor],
        variable_ids: Sequence[int],
        constraints: Sequence[_ObjectConstraint],
        raw_poses: Mapping[int, torch.Tensor],
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        deltas = {
            int(frame_id): torch.nn.Parameter(
                torch.zeros(6, dtype=torch.float32, device=self.device)
            )
            for frame_id in variable_ids
        }
        if not deltas:
            return dict(base_poses), {}
        optimizer = torch.optim.Adam(
            list(deltas.values()),
            lr=float(self.config.learning_rate),
        )
        base_device = {
            int(frame_id): pose.to(self.device)
            for frame_id, pose in base_poses.items()
        }
        for _ in range(int(self.config.optimizer_steps)):
            optimizer.zero_grad(set_to_none=True)
            poses = self._poses_from_deltas(base_device, deltas)
            loss = self._window_loss(poses, deltas, constraints, raw_poses)
            if not bool(torch.isfinite(loss)):
                break
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                for delta in deltas.values():
                    self._clamp_local_delta(delta)
        output = {
            int(frame_id): pose.detach().float().cpu()
            for frame_id, pose in self._poses_from_deltas(base_device, deltas).items()
        }
        return output, {frame_id: value.detach().cpu() for frame_id, value in deltas.items()}

    def _poses_from_deltas(
        self,
        base_poses: Mapping[int, torch.Tensor],
        deltas: Mapping[int, torch.Tensor],
    ) -> dict[int, torch.Tensor]:
        output = dict(base_poses)
        for frame_id, delta in deltas.items():
            update = torch.eye(4, dtype=delta.dtype, device=delta.device)
            update[:3, :3] = _so3_exp(delta[:3])
            update[:3, 3] = delta[3:]
            output[int(frame_id)] = update @ base_poses[int(frame_id)]
        return output

    def _window_loss(
        self,
        poses: Mapping[int, torch.Tensor],
        deltas: Mapping[int, torch.Tensor],
        constraints: Sequence[_ObjectConstraint],
        raw_poses: Mapping[int, torch.Tensor],
    ) -> torch.Tensor:
        losses: list[torch.Tensor] = []
        for constraint in constraints:
            current_pose = poses[int(constraint.frame_j)]
            reference_pose = poses[int(constraint.frame_i)]
            current_world = _transform_points(
                constraint.current.points_camera.to(self.device),
                current_pose,
            ).index_select(0, constraint.current_indices.to(self.device))
            reference_world = _transform_points(
                constraint.reference.points_camera.to(self.device),
                reference_pose,
            ).index_select(0, constraint.reference_indices.to(self.device))
            residual = torch.linalg.vector_norm(current_world - reference_world, dim=-1)
            weights = constraint.weights.to(self.device).clamp_min(1e-6)
            edge_loss = (
                weights * _huber_distance(residual, self.config.huber_delta_m)
            ).sum() / weights.sum()
            losses.append(edge_loss)
        if losses:
            loss = torch.stack(losses).mean()
        else:
            loss = torch.zeros((), dtype=torch.float32, device=self.device)
        if float(self.config.temporal_edge_weight) > 0.0:
            temporal_losses: list[torch.Tensor] = []
            variable_ids = {int(value) for value in deltas}
            ordered_ids = sorted(int(value) for value in poses)
            identity = torch.eye(3, dtype=torch.float32, device=self.device)
            for left_id, right_id in zip(ordered_ids[:-1], ordered_ids[1:]):
                # Pairs whose two poses are fixed contribute a constant and
                # only dilute the gradient of the actual local window as the
                # stream grows.  Keep edges touching a variable node; this
                # preserves the boundary prior without accumulating all past
                # history in the normalization.
                if left_id not in variable_ids and right_id not in variable_ids:
                    continue
                if left_id not in raw_poses or right_id not in raw_poses:
                    continue
                raw_relative = _invert_pose(raw_poses[right_id]).to(self.device) @ raw_poses[
                    left_id
                ].to(self.device)
                current_relative = (
                    _invert_pose_differentiable(poses[right_id]) @ poses[left_id]
                )
                error = _invert_pose(raw_relative) @ current_relative
                rotation_error = error[:3, :3] - identity
                translation_error = error[:3, 3]
                temporal_losses.append(
                    float(self.config.rotation_residual_scale_m) ** 2
                    * rotation_error.square().mean()
                    + translation_error.square().mean()
                )
            if temporal_losses:
                loss = loss + float(self.config.temporal_edge_weight) * torch.stack(
                    temporal_losses
                ).mean()
        if deltas:
            prior_values = []
            for delta in deltas.values():
                prior_vector = torch.cat(
                    (
                        delta[:3] * float(self.config.rotation_residual_scale_m),
                        delta[3:],
                    )
                )
                prior_values.append(prior_vector.square().mean())
            loss = loss + float(self.config.pose_prior_weight) * torch.stack(
                prior_values
            ).mean()
            if len(deltas) > 1 and float(self.config.correction_smoothness_weight) > 0.0:
                ordered = list(deltas.values())
                smooth = torch.stack(
                    [
                        torch.cat(
                            (
                                (left[:3] - right[:3])
                                * float(self.config.rotation_residual_scale_m),
                                left[3:] - right[3:],
                            )
                        ).square().mean()
                        for left, right in zip(ordered[:-1], ordered[1:])
                    ]
                ).mean()
                loss = loss + float(self.config.correction_smoothness_weight) * smooth
        return loss

    def _constraint_stats(
        self,
        constraints: Sequence[_ObjectConstraint],
    ) -> dict[str, Any]:
        per_instance: Counter[int] = Counter()
        for constraint in constraints:
            per_instance[int(constraint.instance_id)] += int(
                constraint.current_indices.numel()
            )
        return {
            "pair_count": int(len(constraints)),
            "total_matches": int(sum(per_instance.values())),
            "instance_count": int(len(per_instance)),
            "matches_per_instance": {str(k): int(v) for k, v in per_instance.items()},
        }

    def _object_loss(
        self,
        poses: Mapping[int, torch.Tensor],
        constraints: Sequence[_ObjectConstraint],
    ) -> float:
        """Average robust loss per object constraint, not per raw point."""

        if not constraints:
            return float("inf")
        values = [
            self._single_constraint_loss(poses, constraint)
            for constraint in constraints
        ]
        finite = [value for value in values if math.isfinite(float(value))]
        return float(sum(finite) / len(finite)) if finite else float("inf")

    def _passes_match_gate(self, stats: Mapping[str, Any]) -> bool:
        if int(stats["instance_count"]) < int(self.config.min_independent_instances):
            return False
        if int(stats["total_matches"]) < int(self.config.min_total_matches):
            return False
        return all(
            int(value) >= int(self.config.min_matches_per_instance)
            for value in stats["matches_per_instance"].values()
        )

    def _passes_candidate_gate(
        self,
        stats: Mapping[str, Any],
        candidate_poses: Mapping[int, torch.Tensor],
        frame_id: int,
        base_pose: torch.Tensor,
        improvement: float,
        constraints: Sequence[_ObjectConstraint],
    ) -> bool:
        if not self._passes_match_gate(stats):
            return False
        if improvement < float(self.config.min_relative_loss_improvement):
            return False
        if int(frame_id) not in candidate_poses:
            return False
        correction = _invert_pose(base_pose) @ candidate_poses[int(frame_id)]
        if _rotation_angle_deg(correction[:3, :3]) > float(
            self.config.max_local_correction_rotation_deg
        ):
            return False
        if float(torch.linalg.vector_norm(correction[:3, 3])) > float(
            self.config.max_local_correction_translation_m
        ):
            return False
        residuals = self._constraint_residuals(candidate_poses, constraints)
        if residuals and float(torch.tensor(residuals).median()) > float(
            self.config.max_validation_residual_m
        ):
            return False
        return True

    def _constraint_residuals(
        self,
        poses: Mapping[int, torch.Tensor],
        constraints: Sequence[_ObjectConstraint],
    ) -> list[float]:
        values: list[float] = []
        for constraint in constraints:
            if (
                int(constraint.frame_i) not in poses
                or int(constraint.frame_j) not in poses
            ):
                continue
            current = _transform_points(
                constraint.current.points_camera,
                poses[int(constraint.frame_j)],
            ).index_select(0, constraint.current_indices)
            reference = _transform_points(
                constraint.reference.points_camera,
                poses[int(constraint.frame_i)],
            ).index_select(
                0, constraint.reference_indices
            )
            values.extend(
                torch.linalg.vector_norm(current - reference, dim=-1).tolist()
            )
        return [float(value) for value in values if math.isfinite(float(value))]

    def _append_edges(
        self,
        accepted_edges: list[ObjectLossEdge],
        candidates: list[dict[str, Any]],
        final_constraints: Sequence[_ObjectConstraint],
        initial_constraints: Sequence[_ObjectConstraint],
        final_poses: Mapping[int, torch.Tensor],
        initial_poses: Mapping[int, torch.Tensor],
    ) -> None:
        initial_by_key = {
            (int(item.frame_i), int(item.frame_j), int(item.instance_id)): item
            for item in initial_constraints
        }
        for constraint in final_constraints:
            key = (
                int(constraint.frame_i),
                int(constraint.frame_j),
                int(constraint.instance_id),
            )
            initial = initial_by_key.get(key, constraint)
            initial_loss = self._single_constraint_loss(
                initial_poses,
                initial,
            )
            final_loss = self._single_constraint_loss(final_poses, constraint)
            initial_matches = int(initial.current_indices.numel())
            final_matches = int(constraint.current_indices.numel())
            improvement = _relative_improvement(initial_loss, final_loss)
            row = self._edge_row(
                constraint,
                initial_matches=initial_matches,
                final_matches=final_matches,
                initial_loss=initial_loss,
                final_loss=final_loss,
                accepted=True,
            )
            candidates.append(row)
            accepted_edges.append(
                ObjectLossEdge(
                    frame_i=int(constraint.frame_i),
                    frame_j=int(constraint.frame_j),
                    instance_id=int(constraint.instance_id),
                    category=str(constraint.category),
                    reference_role=str(constraint.reference_role),
                    num_matches_initial=initial_matches,
                    num_matches_final=final_matches,
                    initial_loss_m=float(initial_loss),
                    final_loss_m=float(final_loss),
                    relative_loss_improvement=float(improvement),
                    edge_weight=float(constraint.pair_weight),
                    provenance=row,
                )
            )

    def _rejection_rows(
        self,
        constraints: Sequence[_ObjectConstraint],
        *,
        initial_poses: Mapping[int, torch.Tensor],
        final_poses: Mapping[int, torch.Tensor],
        reason: str,
    ) -> list[dict[str, Any]]:
        rows = []
        for constraint in constraints:
            initial_loss = self._single_constraint_loss(initial_poses, constraint)
            final_loss = self._single_constraint_loss(final_poses, constraint)
            rows.append(
                self._edge_row(
                    constraint,
                    initial_matches=int(constraint.current_indices.numel()),
                    final_matches=int(constraint.current_indices.numel()),
                    initial_loss=initial_loss,
                    final_loss=final_loss,
                    accepted=False,
                    reason=reason,
                )
            )
        return rows

    def _single_constraint_loss(
        self,
        poses: Mapping[int, torch.Tensor],
        constraint: _ObjectConstraint,
    ) -> float:
        if int(constraint.frame_i) not in poses or int(constraint.frame_j) not in poses:
            return float("inf")
        current = _transform_points(
            constraint.current.points_camera,
            poses[int(constraint.frame_j)],
        ).index_select(0, constraint.current_indices)
        reference = _transform_points(
            constraint.reference.points_camera,
            poses[int(constraint.frame_i)],
        ).index_select(0, constraint.reference_indices)
        residual = torch.linalg.vector_norm(current - reference, dim=-1)
        weights = constraint.weights.clamp_min(1e-6)
        return float(
            (weights * _huber_distance(residual, self.config.huber_delta_m)).sum()
            / weights.sum()
        )

    def _edge_row(
        self,
        constraint: _ObjectConstraint,
        *,
        initial_matches: int,
        final_matches: int,
        initial_loss: float,
        final_loss: float,
        accepted: bool,
        reason: str | None = None,
    ) -> dict[str, Any]:
        row = {
            "frame_i": int(constraint.frame_i),
            "frame_j": int(constraint.frame_j),
            "temporal_gap": int(constraint.frame_j - constraint.frame_i),
            "instance_id": int(constraint.instance_id),
            "category": str(constraint.category),
            "reference_role": str(constraint.reference_role),
            "edge_weight": float(constraint.pair_weight),
            "num_matches_initial": int(initial_matches),
            "num_matches_final": int(final_matches),
            "initial_loss_m": float(initial_loss),
            "final_loss_m": float(final_loss),
            "relative_loss_improvement": _relative_improvement(initial_loss, final_loss),
            "accepted": bool(accepted),
        }
        if reason is not None:
            row["reason"] = str(reason)
        return row

    def _candidate_rejection_reason(
        self,
        stats: Mapping[str, Any],
        improvement: float,
        candidate_poses: Mapping[int, torch.Tensor],
        frame_id: int,
        base_pose: torch.Tensor,
        constraints: Sequence[_ObjectConstraint],
    ) -> str:
        if int(stats["instance_count"]) < int(self.config.min_independent_instances):
            return "too_few_independent_instances"
        if int(stats["total_matches"]) < int(self.config.min_total_matches):
            return "too_few_total_matches"
        if improvement < float(self.config.min_relative_loss_improvement):
            return "insufficient_loss_improvement"
        correction_delta = _invert_pose(base_pose) @ candidate_poses[int(frame_id)]
        if _rotation_angle_deg(correction_delta[:3, :3]) > float(
            self.config.max_local_correction_rotation_deg
        ):
            return "local_rotation_correction_too_large"
        if float(torch.linalg.vector_norm(correction_delta[:3, 3])) > float(
            self.config.max_local_correction_translation_m
        ):
            return "local_translation_correction_too_large"
        residuals = self._constraint_residuals(candidate_poses, constraints)
        if residuals and float(torch.tensor(residuals).median()) > float(
            self.config.max_validation_residual_m
        ):
            return "validation_residual_too_large"
        return "candidate_gate_failed"

    def _clamp_local_delta(self, delta: torch.Tensor) -> None:
        rotation_limit = math.radians(
            float(self.config.max_local_correction_rotation_deg)
        )
        translation_limit = float(self.config.max_local_correction_translation_m)
        rotation_norm = torch.linalg.vector_norm(delta[:3])
        if float(rotation_norm) > rotation_limit:
            delta[:3].mul_(rotation_limit / float(rotation_norm))
        translation_norm = torch.linalg.vector_norm(delta[3:])
        if float(translation_norm) > translation_limit:
            delta[3:].mul_(translation_limit / float(translation_norm))

    @staticmethod
    def _frame_diagnostic(
        frame_id: int,
        *,
        is_anchor: bool,
        observation_count: int,
        reason: str,
        attempted: bool = False,
        accepted: bool = False,
        candidate_pair_count: int = 0,
        match_count: int = 0,
        initial_loss: float = float("nan"),
        final_loss: float | None = None,
    ) -> dict[str, Any]:
        row = {
            "frame_id": int(frame_id),
            "role": "anchor" if is_anchor else "online",
            "optimization_attempted": bool(attempted),
            "accepted": bool(accepted),
            "reason": str(reason),
            "observation_count": int(observation_count),
            "candidate_pair_count": int(candidate_pair_count),
            "accepted_edge_count": int(candidate_pair_count if accepted else 0),
            "match_count": int(match_count),
        }
        if math.isfinite(float(initial_loss)):
            row["initial_loss_m"] = float(initial_loss)
        if final_loss is not None and math.isfinite(float(final_loss)):
            row["final_loss_m"] = float(final_loss)
        return row

    @staticmethod
    def _trace_row(
        frame_id: int,
        *,
        phase: str,
        status: str,
        outer_iteration: int | None = None,
        observation_count: int = 0,
        reference_gap_min: int | None = None,
        reference_gap_max: int | None = None,
        match_pair_count: int = 0,
        match_count: int = 0,
        independent_instance_count: int = 0,
        start_loss: float | None = None,
        loss_m: float | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        row: dict[str, Any] = {
            "frame_id": int(frame_id),
            "phase": str(phase),
            "outer_iteration": outer_iteration,
            "observation_count": int(observation_count),
            "reference_gap_min": reference_gap_min,
            "reference_gap_max": reference_gap_max,
            "match_pair_count": int(match_pair_count),
            "match_count": int(match_count),
            "independent_instance_count": int(independent_instance_count),
            "status": str(status),
        }
        if start_loss is not None:
            row["start_loss_m"] = _finite_or_none(start_loss)
        if loss_m is not None:
            row["loss_m"] = _finite_or_none(loss_m)
        row.update(extra)
        return row


def _relative_improvement(initial: float, final: float) -> float:
    if not math.isfinite(float(initial)) or not math.isfinite(float(final)):
        return 0.0
    return (float(initial) - float(final)) / max(abs(float(initial)), 1e-6)


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _mean_or_zero(values: Sequence[float]) -> float:
    return 0.0 if not values else float(sum(float(value) for value in values) / len(values))


def _invert_pose_differentiable(pose: torch.Tensor) -> torch.Tensor:
    """Invert a homogeneous pose without detaching autograd history."""

    value = pose
    if tuple(value.shape) == (3, 4):
        bottom = torch.zeros(
            (1, 4), dtype=value.dtype, device=value.device
        )
        bottom[0, 3] = 1.0
        value = torch.cat((value, bottom), dim=0)
    elif tuple(value.shape) != (4, 4):
        raise ValueError("Pose must have shape [3,4] or [4,4].")
    rotation = value[:3, :3]
    inverse_rotation = rotation.transpose(0, 1)
    inverse_translation = -inverse_rotation @ value[:3, 3]
    upper = torch.cat((inverse_rotation, inverse_translation[:, None]), dim=1)
    bottom = torch.zeros(
        (1, 4), dtype=value.dtype, device=value.device
    )
    bottom[0, 3] = 1.0
    return torch.cat((upper, bottom), dim=0)


__all__ = ["OnlineObjectPoseLoopConfig", "OnlineObjectPoseLoopRefiner"]
