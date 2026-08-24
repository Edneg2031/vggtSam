"""Projection-first persistent object memory for the V1.1 semantic map.

The frozen StreamVGGT pointmap and camera matrices are used as an image-space
re-identification signal.  A historical object's world points are projected
into the current StreamVGGT grid and compared with the current SAM mask.  A
new SAM source track is kept tentative until the same decision is observed on
multiple frames.  Only confirmed observations write to the persistent map.

This module is deliberately independent of StreamVGGT and SAM model code.  It
consumes cached tensors, so V0 and V1-G remain reproducible and untouched.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from .instance_association import deterministic_points, normalize_category
from .object_memory import estimate_scene_scale


@dataclass(frozen=True)
class ProjectionAssociationConfig:
    """Configuration for projection-first candidate ranking."""

    projection_iou_weight: float = 0.70
    projection_recall_weight: float = 0.12
    projection_precision_weight: float = 0.08
    center_weight: float = 0.05
    category_weight: float = 0.05
    appearance_weight: float = 0.0
    min_projection_iou: float = 0.025
    min_match_score: float = 0.28
    min_projected_pixels: int = 8
    top_k: int = 5
    projection_dilation_radius: int = 3
    max_projection_points: int = 4096
    center_scale_ratio: float = 0.12
    absolute_center_scale: float = 0.02
    category_hard_gate: bool = True
    generic_categories: tuple[str, ...] = ("object", "thing", "item")


@dataclass(frozen=True)
class ProjectionObjectMemoryConfig:
    """Configuration for V1.1 identity and fused-map state."""

    max_points_per_object: int = 4096
    max_fused_voxels_per_object: int = 20000
    min_observation_points: int = 16
    min_mask_pixels: int = 32
    min_track_score: float = 0.50
    min_geometry_confidence: float = 0.30
    center_ema_alpha: float = 0.25
    confirmation_frames: int = 2
    confirmation_window: int = 4
    max_pending_gap: int = 4
    voxel_size_ratio: float = 0.02
    absolute_voxel_size: float = 0.02
    association: ProjectionAssociationConfig = field(
        default_factory=ProjectionAssociationConfig
    )


@dataclass(frozen=True)
class ProjectionCandidate:
    """Auditable score for one historical object."""

    object_id: int
    score: float
    accepted: bool
    projected_pixels: int
    observed_pixels: int
    intersection_pixels: int
    projection_iou: float
    projection_precision: float
    projection_recall: float
    center_distance: float
    center_similarity: float
    category_consistency: float
    appearance_cosine: float
    compared_features: tuple[str, ...]
    rejection_reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "object_id": int(self.object_id),
            "score": float(self.score),
            "accepted": int(self.accepted),
            "projected_pixels": int(self.projected_pixels),
            "observed_pixels": int(self.observed_pixels),
            "intersection_pixels": int(self.intersection_pixels),
            "projection_iou": float(self.projection_iou),
            "projection_precision": float(self.projection_precision),
            "projection_recall": float(self.projection_recall),
            "center_distance": float(self.center_distance),
            "center_similarity": float(self.center_similarity),
            "category_consistency": float(self.category_consistency),
            "appearance_cosine": float(self.appearance_cosine),
            "compared_features": ",".join(self.compared_features),
            "rejection_reason": str(self.rejection_reason),
        }


@dataclass(frozen=True)
class ProjectionObservation:
    """One SAM mask and its corresponding frozen-world geometry."""

    sequence_index: int
    frame_index: int
    source_slot: int
    sam_track_id: int
    category: str
    mask: torch.Tensor
    points: torch.Tensor
    colors: torch.Tensor | None
    center: torch.Tensor
    covariance: torch.Tensor
    extent: torch.Tensor
    confidence: float
    track_score: float
    appearance: torch.Tensor | None = None

    @property
    def quality(self) -> float:
        return float(self.confidence) * float(self.track_score)


@dataclass
class _VoxelState:
    point_sum: torch.Tensor
    color_sum: torch.Tensor
    confidence_sum: float
    count: int


@dataclass
class ProjectionObject:
    """Persistent object with both representative and fused point state."""

    object_id: int
    category: str
    center: torch.Tensor
    covariance: torch.Tensor
    extent: torch.Tensor
    points: torch.Tensor
    point_count: int
    observation_count: int
    first_sequence_index: int
    last_sequence_index: int
    first_frame_index: int
    last_frame_index: int
    confidence: float
    appearance: torch.Tensor | None = None
    observation_timestamps: list[int] = field(default_factory=list)
    sam_track_ids: set[int] = field(default_factory=set)
    source_slots: set[int] = field(default_factory=set)
    voxels: dict[tuple[int, int, int], _VoxelState] = field(
        default_factory=dict
    )

    def fused_arrays(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.voxels:
            empty = torch.empty(0, 3, dtype=torch.float32)
            return (
                empty,
                empty.clone(),
                torch.empty(0, dtype=torch.float32),
                torch.empty(0, dtype=torch.long),
            )
        keys = sorted(self.voxels)
        points = []
        colors = []
        confidence = []
        counts = []
        for key in keys:
            state = self.voxels[key]
            denominator = float(max(1, state.count))
            points.append(state.point_sum / denominator)
            colors.append((state.color_sum / denominator).clamp(0.0, 1.0))
            confidence.append(state.confidence_sum / denominator)
            counts.append(state.count)
        return (
            torch.stack(points).float(),
            torch.stack(colors).float(),
            torch.tensor(confidence, dtype=torch.float32),
            torch.tensor(counts, dtype=torch.long),
        )

    def to_dict(self, *, include_points: bool = False) -> dict[str, object]:
        fused_points, _, _, _ = self.fused_arrays()
        row: dict[str, object] = {
            "object_id": int(self.object_id),
            "category": str(self.category),
            "center": [float(value) for value in self.center.tolist()],
            "covariance": [
                [float(value) for value in line]
                for line in self.covariance.tolist()
            ],
            "extent": [float(value) for value in self.extent.tolist()],
            "point_count": int(self.point_count),
            "stored_point_count": int(self.points.shape[0]),
            "fused_voxel_count": int(len(self.voxels)),
            "fused_point_count": int(fused_points.shape[0]),
            "observation_count": int(self.observation_count),
            "first_sequence_index": int(self.first_sequence_index),
            "last_sequence_index": int(self.last_sequence_index),
            "first_frame_index": int(self.first_frame_index),
            "last_frame_index": int(self.last_frame_index),
            "confidence": float(self.confidence),
            "appearance_dim": (
                int(self.appearance.numel())
                if self.appearance is not None
                else 0
            ),
            "observation_timestamps": [
                int(value) for value in self.observation_timestamps
            ],
            "sam_track_ids": sorted(int(value) for value in self.sam_track_ids),
            "source_slots": sorted(int(value) for value in self.source_slots),
        }
        if include_points:
            row["points"] = [
                [float(value) for value in point]
                for point in self.points.tolist()
            ]
            row["fused_points"] = [
                [float(value) for value in point]
                for point in fused_points.tolist()
            ]
        return row


@dataclass
class _PendingTrack:
    source_key: str
    observations: list[ProjectionObservation] = field(default_factory=list)
    events: list[dict[str, object]] = field(default_factory=list)
    candidate_history: list[int] = field(default_factory=list)
    last_sequence_index: int = -1


def project_world_points_to_mask(
    points: torch.Tensor,
    *,
    world_to_camera: torch.Tensor,
    intrinsics: torch.Tensor,
    image_size: tuple[int, int],
    dilation_radius: int = 0,
    max_points: int = 4096,
) -> torch.Tensor:
    """Project historical world points to a boolean processed-image mask."""

    height, width = (int(value) for value in image_size)
    if height < 1 or width < 1:
        raise ValueError("image_size must be positive.")
    points = deterministic_points(points, limit=int(max_points))
    world_to_camera = world_to_camera.detach().float().cpu()
    intrinsics = intrinsics.detach().float().cpu()
    if tuple(world_to_camera.shape) != (3, 4):
        raise ValueError("world_to_camera must have shape [3,4].")
    if tuple(intrinsics.shape) != (3, 3):
        raise ValueError("intrinsics must have shape [3,3].")
    output = torch.zeros(height, width, dtype=torch.bool)
    if not points.numel():
        return output
    camera = points @ world_to_camera[:, :3].T + world_to_camera[:, 3]
    finite = torch.isfinite(camera).all(dim=1) & (camera[:, 2] > 1e-6)
    if not bool(finite.any()):
        return output
    camera = camera[finite]
    homogeneous = camera @ intrinsics.T
    uv = homogeneous[:, :2] / homogeneous[:, 2:3].clamp_min(1e-6)
    x = uv[:, 0].round().long()
    y = uv[:, 1].round().long()
    inside = (
        torch.isfinite(uv).all(dim=1)
        & (x >= 0)
        & (x < width)
        & (y >= 0)
        & (y < height)
    )
    if bool(inside.any()):
        output[y[inside], x[inside]] = True
    radius = int(dilation_radius)
    if radius < 0:
        raise ValueError("dilation_radius must be non-negative.")
    if radius:
        kernel = 2 * radius + 1
        output = F.max_pool2d(
            output.float()[None, None],
            kernel_size=kernel,
            stride=1,
            padding=radius,
        )[0, 0].bool()
    return output


def mask_overlap_stats(
    observed: torch.Tensor,
    projected: torch.Tensor,
) -> dict[str, float | int]:
    """Return IoU, precision, recall and pixel counts for two masks."""

    observed = observed.detach().cpu().bool()
    projected = projected.detach().cpu().bool()
    if tuple(observed.shape) != tuple(projected.shape):
        raise ValueError("Observed and projected masks must share a shape.")
    intersection = int((observed & projected).sum())
    observed_pixels = int(observed.sum())
    projected_pixels = int(projected.sum())
    union = int((observed | projected).sum())
    return {
        "projected_pixels": projected_pixels,
        "observed_pixels": observed_pixels,
        "intersection_pixels": intersection,
        "projection_iou": intersection / union if union else 0.0,
        "projection_precision": (
            intersection / projected_pixels if projected_pixels else 0.0
        ),
        "projection_recall": (
            intersection / observed_pixels if observed_pixels else 0.0
        ),
    }


class ProjectionObjectMemory:
    """Causal projection-based identity registry with delayed confirmation."""

    def __init__(
        self,
        config: ProjectionObjectMemoryConfig | None = None,
    ) -> None:
        self.config = config or ProjectionObjectMemoryConfig()
        _validate_config(self.config)
        self.registry: dict[int, ProjectionObject] = {}
        self.track_to_object: dict[str, int] = {}
        self.pending: dict[str, _PendingTrack] = {}
        self.events: list[dict[str, object]] = []
        self.scene_scale = 1.0
        self._next_object_id = 0

    @property
    def objects(self) -> tuple[ProjectionObject, ...]:
        return tuple(self.registry[key] for key in sorted(self.registry))

    @property
    def object_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self.registry))

    def rank_candidates(
        self,
        observation: ProjectionObservation,
        *,
        world_to_camera: torch.Tensor,
        intrinsics: torch.Tensor,
        image_size: tuple[int, int],
        excluded_object_ids: Sequence[int] = (),
    ) -> list[ProjectionCandidate]:
        excluded = {int(value) for value in excluded_object_ids}
        ranked: list[ProjectionCandidate] = []
        for obj in self.objects:
            if obj.object_id in excluded:
                continue
            ranked.append(
                self._score_candidate(
                    observation,
                    obj,
                    world_to_camera=world_to_camera,
                    intrinsics=intrinsics,
                    image_size=image_size,
                )
            )
        ranked.sort(
            key=lambda row: (
                -float(row.score),
                -float(row.projection_iou),
                int(row.object_id),
            )
        )
        return ranked[: int(self.config.association.top_k)]

    def process_sequence(
        self,
        *,
        world_points: torch.Tensor,
        confidence: torch.Tensor,
        masks: torch.Tensor,
        track_scores: torch.Tensor,
        frame_indices: Sequence[int],
        track_ids: Sequence[int],
        track_prompts: Sequence[str],
        world_to_camera: torch.Tensor,
        intrinsics: torch.Tensor,
        image_size: tuple[int, int],
        images: torch.Tensor | None = None,
        appearance: torch.Tensor | None = None,
    ) -> dict[str, object]:
        """Process a cached clip in causal frame/slot order."""

        points = world_points.detach().float().cpu()
        confidence = confidence.detach().float().cpu()
        masks = masks.detach().bool().cpu()
        scores = track_scores.detach().float().cpu()
        w2c = world_to_camera.detach().float().cpu()
        intrinsics = intrinsics.detach().float().cpu()
        if points.ndim != 4 or points.shape[-1] != 3:
            raise ValueError("world_points must have shape [S,H,W,3].")
        sequence, height, width = points.shape[:3]
        if confidence.shape != (sequence, height, width):
            raise ValueError("confidence must have shape [S,H,W].")
        if masks.ndim != 4 or masks.shape[0] != sequence or masks.shape[2:] != (
            height,
            width,
        ):
            raise ValueError("masks must have shape [S,K,H,W].")
        tracks = masks.shape[1]
        if tuple(scores.shape) != (sequence, tracks):
            raise ValueError("track_scores must have shape [S,K].")
        if len(frame_indices) != sequence:
            raise ValueError("frame_indices do not match sequence length.")
        if len(track_ids) != tracks or len(track_prompts) != tracks:
            raise ValueError("Track metadata does not match mask slots.")
        if tuple(w2c.shape) != (sequence, 3, 4):
            raise ValueError("world_to_camera must have shape [S,3,4].")
        if tuple(intrinsics.shape) == (3, 3):
            intrinsics = intrinsics.unsqueeze(0).expand(sequence, -1, -1).clone()
        if tuple(intrinsics.shape) != (sequence, 3, 3):
            raise ValueError("intrinsics must have shape [S,3,3] or [3,3].")
        if tuple(int(value) for value in image_size) != (height, width):
            raise ValueError(
                f"image_size={image_size} does not match point grid {(height, width)}."
            )
        if images is not None:
            images = images.detach().float().cpu()
            if tuple(images.shape) != (sequence, 3, height, width):
                raise ValueError("images must have shape [S,3,H,W].")
        if appearance is not None:
            appearance = appearance.detach().float().cpu()
            if appearance.ndim != 3 or tuple(appearance.shape[:2]) != (
                sequence,
                tracks,
            ):
                raise ValueError("appearance must have shape [S,K,D].")

        self.scene_scale = estimate_scene_scale(points)
        persistent_ids = torch.full((sequence, tracks), -1, dtype=torch.long)
        write_mask = torch.zeros(sequence, tracks, dtype=torch.bool)
        local_events: list[dict[str, object]] = []
        for frame in range(sequence):
            occupied: set[int] = set()
            for slot in range(tracks):
                mask = masks[frame, slot]
                area = int(mask.sum())
                score = float(scores[frame, slot])
                finite = torch.isfinite(points[frame]).all(dim=-1)
                valid_geometry = (
                    mask
                    & finite
                    & torch.isfinite(confidence[frame])
                    & (confidence[frame] >= self.config.min_geometry_confidence)
                )
                selected = points[frame][valid_geometry]
                geometry_score = (
                    float(confidence[frame][valid_geometry].mean())
                    if bool(valid_geometry.any())
                    else 0.0
                )
                if (
                    area < self.config.min_mask_pixels
                    or score < self.config.min_track_score
                    or selected.shape[0] < self.config.min_observation_points
                ):
                    event = {
                        "sequence_index": int(frame),
                        "frame_index": int(frame_indices[frame]),
                        "source_slot": int(slot),
                        "sam_track_id": int(track_ids[slot]),
                        "category": str(track_prompts[slot]),
                        "mask_pixels": int(area),
                        "geometry_points": int(selected.shape[0]),
                        "geometry_confidence": float(geometry_score),
                        "track_score": float(score),
                        "persistent_object_id": -1,
                        "map_write": 0,
                        "tentative": 0,
                        "action": _rejection_reason(
                            area=area,
                            score=score,
                            geometry_score=geometry_score,
                            point_count=int(selected.shape[0]),
                            config=self.config,
                        ),
                        "source_key": _source_key(int(track_ids[slot]), slot),
                    }
                    self.events.append(event)
                    local_events.append(event)
                    continue
                colors = None
                if images is not None:
                    colors = images[frame].permute(1, 2, 0)[valid_geometry]
                appearance_value = (
                    appearance[frame, slot] if appearance is not None else None
                )
                observation = make_projection_observation(
                    sequence_index=frame,
                    frame_index=int(frame_indices[frame]),
                    source_slot=slot,
                    sam_track_id=int(track_ids[slot]),
                    category=str(track_prompts[slot]),
                    mask=mask,
                    points=selected,
                    colors=colors,
                    confidence=geometry_score,
                    track_score=score,
                    appearance=appearance_value,
                    max_points=self.config.association.max_projection_points,
                )
                result = self.process_observation(
                    observation,
                    world_to_camera=w2c[frame],
                    intrinsics=intrinsics[frame],
                    image_size=(height, width),
                    occupied_object_ids=tuple(occupied),
                )
                event = result["event"]
                local_events.append(event)
                assigned = int(result["assigned_object_id"])
                if assigned >= 0:
                    persistent_ids[frame, slot] = assigned
                    write_mask[frame, slot] = True
                    occupied.add(assigned)
                for backfill in result["backfilled"]:
                    backfill_frame = int(backfill["sequence_index"])
                    backfill_slot = int(backfill["source_slot"])
                    object_id = int(backfill["persistent_object_id"])
                    persistent_ids[backfill_frame, backfill_slot] = object_id
                    write_mask[backfill_frame, backfill_slot] = True
                    if backfill_frame == frame:
                        occupied.add(object_id)

        # Unconfirmed observations are intentionally not written.  Marking
        # them explicitly makes coverage loss distinguishable from an empty
        # SAM mask in the artifact and CSV event log.
        for pending in self.pending.values():
            for event in pending.events:
                event["action"] = "tentative_unconfirmed_end_of_clip"
                event["tentative"] = 1
                event["map_write"] = 0

        return {
            "persistent_object_ids": persistent_ids,
            "map_write_mask": write_mask,
            "object_ids": list(self.object_ids),
            "objects": [obj.to_dict() for obj in self.objects],
            "events": local_events,
            "all_events": list(self.events),
            "track_to_object": {
                str(key): int(value)
                for key, value in sorted(self.track_to_object.items())
            },
            "scene_scale": float(self.scene_scale),
            "persistent_object_count": int(len(self.registry)),
            "pending_track_count": int(len(self.pending)),
            "confirmed_observation_count": int(write_mask.sum()),
            "fused_map": self.fused_map(),
        }

    def process_observation(
        self,
        observation: ProjectionObservation,
        *,
        world_to_camera: torch.Tensor,
        intrinsics: torch.Tensor,
        image_size: tuple[int, int],
        occupied_object_ids: Sequence[int] = (),
    ) -> dict[str, object]:
        """Process one observation and possibly confirm a pending track."""

        source_key = _source_key(observation.sam_track_id, observation.source_slot)
        direct_id = self.track_to_object.get(source_key)
        if direct_id is not None:
            obj = self.registry[int(direct_id)]
            self._update_object(obj, observation)
            event = _event(
                observation,
                action="direct_active_track",
                object_id=obj.object_id,
                candidate=None,
                top_k=(),
                source_key=source_key,
                tentative=0,
            )
            event["map_write"] = 1
            self.events.append(event)
            return {
                "event": event,
                "assigned_object_id": int(obj.object_id),
                "backfilled": [],
            }

        # Keep occupied objects in the candidate list.  A second SAM slot in
        # the same frame may be a duplicate of an already active object; the
        # temporal confirmation gate, rather than a same-frame exclusion,
        # decides whether it should merge.
        candidates = self.rank_candidates(
            observation,
            world_to_camera=world_to_camera,
            intrinsics=intrinsics,
            image_size=image_size,
            excluded_object_ids=(),
        )
        candidate = next((row for row in candidates if row.accepted), None)
        pending = self.pending.get(source_key)
        if pending is not None and (
            observation.sequence_index - pending.last_sequence_index
            > self.config.max_pending_gap
        ):
            for old_event in pending.events:
                old_event["action"] = "tentative_expired_before_confirmation"
                old_event["tentative"] = 1
                old_event["map_write"] = 0
            self.pending.pop(source_key, None)
            pending = None
        if pending is None:
            pending = _PendingTrack(source_key=source_key)
            self.pending[source_key] = pending
        candidate_id = int(candidate.object_id) if candidate is not None else -1
        pending.observations.append(observation)
        pending.candidate_history.append(candidate_id)
        pending.last_sequence_index = int(observation.sequence_index)
        if len(pending.observations) > self.config.confirmation_window:
            old_event = pending.events.pop(0)
            old_event["action"] = "tentative_window_evicted"
            old_event["tentative"] = 1
            old_event["map_write"] = 0
            pending.observations.pop(0)
            pending.candidate_history.pop(0)

        event = _event(
            observation,
            action="tentative_candidate_existing"
            if candidate is not None
            else "tentative_candidate_new",
            object_id=-1,
            candidate=candidate,
            top_k=candidates,
            source_key=source_key,
            tentative=1,
        )
        event["map_write"] = 0
        pending.events.append(event)
        self.events.append(event)

        confirmed_id = self._confirmation_id(pending)
        if confirmed_id is None:
            return {
                "event": event,
                "assigned_object_id": -1,
                "backfilled": [],
            }

        if confirmed_id >= 0:
            obj = self.registry[confirmed_id]
            action = "confirm_associate_existing_object"
        else:
            obj = self._create_object(pending.observations[0])
            confirmed_id = obj.object_id
            action = "confirm_create_new_object"
        for pending_observation in pending.observations:
            if pending_observation is not pending.observations[0] or action.startswith(
                "confirm_associate"
            ):
                # For an existing object, the first observation is not yet in
                # memory; for a newly created object it is already seeded.
                if not (
                    action == "confirm_create_new_object"
                    and pending_observation is pending.observations[0]
                ):
                    self._update_object(obj, pending_observation)
        self.track_to_object[source_key] = int(confirmed_id)
        backfilled: list[dict[str, object]] = []
        for pending_observation, pending_event in zip(
            pending.observations, pending.events
        ):
            pending_event.update(
                {
                    "persistent_object_id": int(confirmed_id),
                    "action": action if pending_event is event else "tentative_backfilled",
                    "tentative": 0,
                    "map_write": 1,
                    "confirmed_at_sequence_index": int(observation.sequence_index),
                }
            )
            backfilled.append(
                {
                    "sequence_index": int(pending_observation.sequence_index),
                    "source_slot": int(pending_observation.source_slot),
                    "persistent_object_id": int(confirmed_id),
                }
            )
        self.pending.pop(source_key, None)
        return {
            "event": event,
            "assigned_object_id": int(confirmed_id),
            "backfilled": backfilled,
        }

    def fused_map(self, *, max_points: int | None = None) -> dict[str, torch.Tensor]:
        """Return the object-level voxel-fused map tensors."""

        rows: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]] = []
        for obj in self.objects:
            points, colors, confidence, counts = obj.fused_arrays()
            for index in range(points.shape[0]):
                rows.append(
                    (
                        points[index],
                        colors[index],
                        confidence[index],
                        int(obj.object_id),
                        int(counts[index]),
                    )
                )
        if not rows:
            return {
                "world_points": torch.empty(0, 3),
                "rgb": torch.empty(0, 3),
                "confidence": torch.empty(0),
                "object_ids": torch.empty(0, dtype=torch.long),
                "observation_counts": torch.empty(0, dtype=torch.long),
            }
        rows.sort(key=lambda row: (row[3], -row[4]))
        if max_points is not None and len(rows) > int(max_points):
            # Keep high-support voxels first while preserving deterministic
            # object/key order for equal support.
            rows = sorted(
                rows,
                key=lambda row: (-row[4], row[3]),
            )[: int(max_points)]
            rows.sort(key=lambda row: (row[3], -row[4]))
        return {
            "world_points": torch.stack([row[0] for row in rows]).float(),
            "rgb": torch.stack([row[1] for row in rows]).float(),
            "confidence": torch.stack([row[2] for row in rows]).float(),
            "object_ids": torch.tensor([row[3] for row in rows], dtype=torch.long),
            "observation_counts": torch.tensor(
                [row[4] for row in rows], dtype=torch.long
            ),
        }

    def to_dict(self, *, include_points: bool = False) -> dict[str, object]:
        return {
            "schema": 1,
            "revision": "v1_1_projection_temporal_voxel_memory_r1",
            "scene_scale": float(self.scene_scale),
            "persistent_object_count": int(len(self.registry)),
            "pending_track_count": int(len(self.pending)),
            "objects": [
                obj.to_dict(include_points=include_points) for obj in self.objects
            ],
            "track_to_object": {
                str(key): int(value)
                for key, value in sorted(self.track_to_object.items())
            },
            "event_count": int(len(self.events)),
        }

    def save_json(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf8",
        )
        return output

    def save_csv(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        rows = [obj.to_dict() for obj in self.objects]
        fields: list[str] = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
        if not fields:
            fields = ["object_id", "category"]
        with output.open("w", encoding="utf8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({key: _csv_value(row.get(key, "")) for key in fields})
        return output

    def _score_candidate(
        self,
        observation: ProjectionObservation,
        obj: ProjectionObject,
        *,
        world_to_camera: torch.Tensor,
        intrinsics: torch.Tensor,
        image_size: tuple[int, int],
    ) -> ProjectionCandidate:
        config = self.config.association
        projected = project_world_points_to_mask(
            obj.points,
            world_to_camera=world_to_camera,
            intrinsics=intrinsics,
            image_size=image_size,
            dilation_radius=config.projection_dilation_radius,
            max_points=config.max_projection_points,
        )
        overlap = mask_overlap_stats(observation.mask, projected)
        category = _category_consistency(
            observation.category,
            obj.category,
            generic_categories=config.generic_categories,
        )
        center_distance = float(torch.linalg.vector_norm(observation.center - obj.center))
        center_scale = max(
            float(self.scene_scale) * float(config.center_scale_ratio),
            float(config.absolute_center_scale),
            1e-6,
        )
        center_similarity = math.exp(-center_distance / center_scale)
        appearance_cosine = float("nan")
        if observation.appearance is not None and obj.appearance is not None:
            appearance_cosine = _cosine(observation.appearance, obj.appearance)

        terms: list[tuple[float, float]] = [
            (config.projection_iou_weight, float(overlap["projection_iou"])),
            (config.projection_recall_weight, float(overlap["projection_recall"])),
            (config.projection_precision_weight, float(overlap["projection_precision"])),
            (config.center_weight, center_similarity),
            (config.category_weight, category),
        ]
        features = ["projection_iou", "projection_recall", "projection_precision", "center", "category"]
        if math.isfinite(appearance_cosine) and config.appearance_weight > 0.0:
            terms.append((config.appearance_weight, max(0.0, (appearance_cosine + 1.0) * 0.5)))
            features.append("appearance")
        total_weight = sum(max(0.0, float(weight)) for weight, _ in terms)
        score = (
            sum(float(weight) * float(value) for weight, value in terms) / total_weight
            if total_weight > 0.0
            else 0.0
        )
        reason = ""
        accepted = True
        if config.category_hard_gate and category <= 0.0:
            accepted = False
            reason = "category_mismatch"
        elif int(overlap["projected_pixels"]) < config.min_projected_pixels:
            accepted = False
            reason = "too_few_projected_pixels"
        elif float(overlap["projection_iou"]) < config.min_projection_iou:
            accepted = False
            reason = "projection_iou_below_threshold"
        elif score < config.min_match_score:
            accepted = False
            reason = "score_below_threshold"
        return ProjectionCandidate(
            object_id=int(obj.object_id),
            score=float(score),
            accepted=bool(accepted),
            projected_pixels=int(overlap["projected_pixels"]),
            observed_pixels=int(overlap["observed_pixels"]),
            intersection_pixels=int(overlap["intersection_pixels"]),
            projection_iou=float(overlap["projection_iou"]),
            projection_precision=float(overlap["projection_precision"]),
            projection_recall=float(overlap["projection_recall"]),
            center_distance=center_distance,
            center_similarity=float(center_similarity),
            category_consistency=float(category),
            appearance_cosine=appearance_cosine,
            compared_features=tuple(features),
            rejection_reason=reason,
        )

    def _confirmation_id(self, pending: _PendingTrack) -> int | None:
        if len(pending.candidate_history) < self.config.confirmation_frames:
            return None
        history = pending.candidate_history[-self.config.confirmation_window :]
        counts = Counter(history)
        candidate_id, count = counts.most_common(1)[0]
        if count < self.config.confirmation_frames:
            return None
        return int(candidate_id)

    def _create_object(self, observation: ProjectionObservation) -> ProjectionObject:
        object_id = self._next_object_id
        self._next_object_id += 1
        values = deterministic_points(
            observation.points,
            limit=self.config.max_points_per_object,
        )
        obj = ProjectionObject(
            object_id=object_id,
            category=normalize_category(observation.category) or "object",
            center=observation.center.clone(),
            covariance=observation.covariance.clone(),
            extent=observation.extent.clone(),
            points=values.clone(),
            point_count=0,
            observation_count=0,
            first_sequence_index=int(observation.sequence_index),
            last_sequence_index=int(observation.sequence_index),
            first_frame_index=int(observation.frame_index),
            last_frame_index=int(observation.frame_index),
            confidence=0.0,
            appearance=_normalise_appearance(observation.appearance),
        )
        self.registry[object_id] = obj
        self._update_object(obj, observation)
        return obj

    def _update_object(
        self,
        obj: ProjectionObject,
        observation: ProjectionObservation,
    ) -> None:
        alpha = float(self.config.center_ema_alpha)
        if obj.observation_count:
            obj.center = (1.0 - alpha) * obj.center + alpha * observation.center
            obj.extent = torch.maximum(obj.extent, observation.extent)
            obj.confidence = (1.0 - alpha) * obj.confidence + alpha * observation.quality
        else:
            obj.center = observation.center.clone()
            obj.extent = observation.extent.clone()
            obj.confidence = observation.quality
        obj.observation_count += 1
        obj.point_count += int(observation.points.shape[0])
        obj.last_sequence_index = int(observation.sequence_index)
        obj.last_frame_index = int(observation.frame_index)
        obj.observation_timestamps.append(int(observation.frame_index))
        obj.sam_track_ids.add(int(observation.sam_track_id))
        obj.source_slots.add(int(observation.source_slot))
        if len(obj.observation_timestamps) > 1024:
            del obj.observation_timestamps[:-1024]
        self._fuse_observation(obj, observation)
        fused_points, _, _, _ = obj.fused_arrays()
        obj.points = deterministic_points(
            fused_points,
            limit=self.config.max_points_per_object,
        )
        obj.covariance = _point_covariance(obj.points)
        appearance = _normalise_appearance(observation.appearance)
        if appearance is not None:
            if obj.appearance is None or obj.appearance.shape != appearance.shape:
                obj.appearance = appearance
            else:
                obj.appearance = _normalise_appearance(
                    (1.0 - alpha) * obj.appearance + alpha * appearance
                )

    def _fuse_observation(
        self,
        obj: ProjectionObject,
        observation: ProjectionObservation,
    ) -> None:
        voxel_size = max(
            float(self.config.absolute_voxel_size),
            float(self.scene_scale) * float(self.config.voxel_size_ratio),
        )
        points = observation.points.detach().float().cpu().reshape(-1, 3)
        valid = torch.isfinite(points).all(dim=1)
        points = points[valid]
        if not points.numel():
            return
        if observation.colors is None:
            colors = torch.zeros_like(points)
        else:
            colors = observation.colors.detach().float().cpu().reshape(-1, 3)[valid]
            colors = colors.clamp(0.0, 1.0)
        keys = torch.floor(points / voxel_size).long()
        unique, inverse = torch.unique(keys, dim=0, return_inverse=True)
        confidence_value = float(observation.confidence)
        for group in range(unique.shape[0]):
            selected = inverse == group
            key = tuple(int(value) for value in unique[group].tolist())
            point_sum = points[selected].sum(dim=0)
            color_sum = colors[selected].sum(dim=0)
            count = int(selected.sum())
            current = obj.voxels.get(key)
            if current is None:
                obj.voxels[key] = _VoxelState(
                    point_sum=point_sum,
                    color_sum=color_sum,
                    confidence_sum=confidence_value * count,
                    count=count,
                )
            else:
                current.point_sum = current.point_sum + point_sum
                current.color_sum = current.color_sum + color_sum
                current.confidence_sum += confidence_value * count
                current.count += count
        limit = int(self.config.max_fused_voxels_per_object)
        if len(obj.voxels) > limit:
            keep = sorted(
                obj.voxels,
                key=lambda key: (-obj.voxels[key].count, key),
            )[:limit]
            obj.voxels = {key: obj.voxels[key] for key in keep}


def make_projection_observation(
    *,
    sequence_index: int,
    frame_index: int,
    source_slot: int,
    sam_track_id: int,
    category: str,
    mask: torch.Tensor,
    points: torch.Tensor,
    colors: torch.Tensor | None,
    confidence: float,
    track_score: float,
    appearance: torch.Tensor | None = None,
    max_points: int = 4096,
) -> ProjectionObservation:
    values = deterministic_points(points, limit=max_points)
    if values.shape[0] < 1:
        raise ValueError("Projection observations require at least one point.")
    # Fusion keeps all selected points; geometric summaries use a bounded,
    # deterministic sample.  Keep colors aligned with the full point tensor.
    full_points = points.detach().float().cpu().reshape(-1, 3)
    full_valid = torch.isfinite(full_points).all(dim=1)
    full_points = full_points[full_valid]
    full_colors = None
    if colors is not None:
        full_colors = colors.detach().float().cpu().reshape(-1, 3)[full_valid]
    center = values.mean(dim=0)
    covariance = _point_covariance(values)
    extent = values.max(dim=0).values - values.min(dim=0).values
    return ProjectionObservation(
        sequence_index=int(sequence_index),
        frame_index=int(frame_index),
        source_slot=int(source_slot),
        sam_track_id=int(sam_track_id),
        category=str(category),
        mask=mask.detach().bool().cpu(),
        points=full_points,
        colors=full_colors,
        center=center,
        covariance=covariance,
        extent=extent,
        confidence=float(confidence),
        track_score=float(track_score),
        appearance=_normalise_appearance(appearance),
    )


def _event(
    observation: ProjectionObservation,
    *,
    action: str,
    object_id: int,
    candidate: ProjectionCandidate | None,
    top_k: Sequence[ProjectionCandidate],
    source_key: str,
    tentative: int,
) -> dict[str, object]:
    row: dict[str, object] = {
        "sequence_index": int(observation.sequence_index),
        "frame_index": int(observation.frame_index),
        "source_slot": int(observation.source_slot),
        "sam_track_id": int(observation.sam_track_id),
        "category": str(observation.category),
        "track_score": float(observation.track_score),
        "geometry_confidence": float(observation.confidence),
        "observation_quality": float(observation.quality),
        "mask_pixels": int(observation.mask.sum()),
        "geometry_points": int(observation.points.shape[0]),
        "persistent_object_id": int(object_id),
        "action": str(action),
        "source_key": str(source_key),
        "tentative": int(tentative),
        "map_write": 0,
        "association_top_k": json.dumps(
            [row.to_dict() for row in top_k], sort_keys=True
        ),
    }
    if candidate is None:
        row.update(
            {
                "association_object_id": -1,
                "association_score": float("nan"),
                "association_accepted": 0,
                "association_projected_pixels": 0,
                "association_observed_pixels": int(observation.mask.sum()),
                "association_intersection_pixels": 0,
                "association_projection_iou": 0.0,
                "association_projection_precision": 0.0,
                "association_projection_recall": 0.0,
                "association_center_distance": float("nan"),
                "association_center_similarity": float("nan"),
                "association_category_consistency": float("nan"),
                "association_appearance_cosine": float("nan"),
                "association_compared_features": "",
                "association_rejection_reason": "no_existing_object_candidate",
            }
        )
    else:
        row.update(
            {
                f"association_{key}": value
                for key, value in candidate.to_dict().items()
            }
        )
    return row


def _category_consistency(
    left: str,
    right: str,
    *,
    generic_categories: Sequence[str],
) -> float:
    left_key = normalize_category(left)
    right_key = normalize_category(right)
    generic = {normalize_category(value) for value in generic_categories}
    if not left_key or not right_key:
        return 0.5
    if left_key == right_key:
        return 1.0
    if left_key in generic or right_key in generic:
        return 0.5
    if left_key in right_key or right_key in left_key:
        return 0.75
    return 0.0


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.detach().float().cpu().reshape(-1)
    right = right.detach().float().cpu().reshape(-1)
    if left.shape != right.shape or not left.numel():
        return float("nan")
    if not bool(torch.isfinite(left).all() and torch.isfinite(right).all()):
        return float("nan")
    left_norm = float(torch.linalg.vector_norm(left))
    right_norm = float(torch.linalg.vector_norm(right))
    if left_norm <= 1e-8 or right_norm <= 1e-8:
        return float("nan")
    return float(F.cosine_similarity(left[None], right[None]).item())


def _normalise_appearance(value: torch.Tensor | None) -> torch.Tensor | None:
    if value is None:
        return None
    tensor = value.detach().float().cpu().reshape(-1)
    if not tensor.numel() or not bool(torch.isfinite(tensor).all()):
        return None
    norm = float(torch.linalg.vector_norm(tensor))
    return tensor / norm if norm > 1e-8 else None


def _point_covariance(points: torch.Tensor) -> torch.Tensor:
    values = points.detach().float().cpu().reshape(-1, 3)
    values = values[torch.isfinite(values).all(dim=1)]
    if values.shape[0] <= 1:
        return torch.eye(3) * 1e-6
    centered = values - values.mean(dim=0, keepdim=True)
    return centered.T @ centered / float(max(1, values.shape[0] - 1))


def _source_key(sam_track_id: int, source_slot: int) -> str:
    if int(sam_track_id) >= 0:
        return f"sam:{int(sam_track_id)}"
    return f"slot:{int(source_slot)}"


def _rejection_reason(
    *,
    area: int,
    score: float,
    geometry_score: float,
    point_count: int,
    config: ProjectionObjectMemoryConfig,
) -> str:
    if area < config.min_mask_pixels:
        return "reject:too_few_mask_pixels"
    if score < config.min_track_score:
        return "reject:low_track_score"
    if geometry_score < config.min_geometry_confidence:
        return "reject:low_geometry_confidence"
    if point_count < config.min_observation_points:
        return "reject:too_few_geometry_points"
    return "reject:unknown"


def _csv_value(value: object) -> object:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True, default=str)
    return value


def _validate_config(config: ProjectionObjectMemoryConfig) -> None:
    if config.max_points_per_object < 1:
        raise ValueError("max_points_per_object must be positive.")
    if config.max_fused_voxels_per_object < 1:
        raise ValueError("max_fused_voxels_per_object must be positive.")
    if config.min_observation_points < 1 or config.min_mask_pixels < 1:
        raise ValueError("Observation thresholds must be positive.")
    if config.confirmation_frames < 2:
        raise ValueError("confirmation_frames must be at least two.")
    if config.confirmation_window < config.confirmation_frames:
        raise ValueError("confirmation_window must cover confirmation_frames.")
    if config.max_pending_gap < 1:
        raise ValueError("max_pending_gap must be positive.")
    if config.absolute_voxel_size <= 0.0 or config.voxel_size_ratio < 0.0:
        raise ValueError("Voxel sizes must be positive/non-negative.")
    for name, value in (
        ("min_track_score", config.min_track_score),
        ("min_geometry_confidence", config.min_geometry_confidence),
        ("center_ema_alpha", config.center_ema_alpha),
    ):
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be in [0,1].")
    association = config.association
    if association.top_k < 1 or association.max_projection_points < 1:
        raise ValueError("top_k and max_projection_points must be positive.")
    if association.projection_dilation_radius < 0:
        raise ValueError("projection_dilation_radius must be non-negative.")
    if association.min_projected_pixels < 1:
        raise ValueError("min_projected_pixels must be positive.")
    if not 0.0 <= association.min_projection_iou <= 1.0:
        raise ValueError("min_projection_iou must be in [0,1].")
    if not 0.0 <= association.min_match_score <= 1.0:
        raise ValueError("min_match_score must be in [0,1].")
    weights = (
        association.projection_iou_weight,
        association.projection_recall_weight,
        association.projection_precision_weight,
        association.center_weight,
        association.category_weight,
        association.appearance_weight,
    )
    if any(float(value) < 0.0 for value in weights) or sum(weights) <= 0.0:
        raise ValueError("Association weights must be non-negative and non-zero.")
