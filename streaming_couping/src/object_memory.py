"""Auditable short/long object memory for semantic-map observation writes.

This module deliberately does not modify SAM hidden memory.  It consumes a
finished persistent track and decides which 2-D/3-D observations may update an
object-level semantic map.  Every read, rejection, write, and replacement is
returned as tabular provenance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

import torch

from .instance_association import (
    AssociationCandidate,
    InstanceAssociationConfig,
    InstanceAssociator,
    InstanceObservation,
    deterministic_points,
    normalize_category,
    voxel_keys,
)


@dataclass(frozen=True)
class ObjectMemoryConfig:
    short_term_capacity: int = 5
    long_term_capacity: int = 4
    min_track_score: float = 0.50
    min_geometry_confidence: float = 0.30
    min_mask_pixels: int = 32
    min_area_ratio: float = 0.35
    max_area_ratio: float = 2.85
    long_term_min_track_score: float = 0.75
    long_term_min_geometry_confidence: float = 0.50
    reentry_gap: int = 3


@dataclass(frozen=True)
class MemoryEntry:
    sequence_index: int
    frame_index: int
    area: int
    track_score: float
    geometry_confidence: float
    quality: float


def build_object_memory_write_policy(
    *,
    masks: torch.Tensor,
    scores: torch.Tensor,
    geometry_confidence: torch.Tensor,
    frame_indices: Sequence[int] | None = None,
    track_ids: Sequence[int] | None = None,
    variant: str = "",
    config: ObjectMemoryConfig = ObjectMemoryConfig(),
) -> dict[str, object]:
    """Return deterministic map-write gates and full memory-event rows."""

    masks = masks.detach().cpu().bool()
    scores = scores.detach().cpu().float()
    geometry_confidence = geometry_confidence.detach().cpu().float()
    if masks.ndim != 4:
        raise ValueError("masks must have shape [S,K,H,W].")
    sequence, tracks, height, width = masks.shape
    if tuple(scores.shape) != (sequence, tracks):
        raise ValueError("scores must have shape [S,K].")
    if tuple(geometry_confidence.shape) != (sequence, height, width):
        raise ValueError("geometry_confidence must have shape [S,H,W].")
    _validate_config(config)
    frame_indices = tuple(
        range(sequence) if frame_indices is None else frame_indices
    )
    track_ids = tuple(range(tracks) if track_ids is None else track_ids)
    if len(frame_indices) != sequence or len(track_ids) != tracks:
        raise ValueError("Frame/track metadata does not match masks.")

    short_term: list[list[MemoryEntry]] = [[] for _ in range(tracks)]
    long_term: list[list[MemoryEntry]] = [[] for _ in range(tracks)]
    last_visible = [-1 for _ in range(tracks)]
    write_mask = torch.zeros(sequence, tracks, dtype=torch.bool)
    rows: list[dict[str, object]] = []
    for frame in range(sequence):
        for slot in range(tracks):
            mask = masks[frame, slot]
            area = int(mask.sum())
            score = float(scores[frame, slot])
            geometry_score = (
                float(geometry_confidence[frame][mask].mean())
                if area
                else 0.0
            )
            history = short_term[slot] + long_term[slot]
            historical_area = (
                float(median(entry.area for entry in history))
                if history
                else float(area)
            )
            area_ratio = (
                area / historical_area if historical_area > 0.0 else 1.0
            )
            visibility_gap = (
                frame - last_visible[slot] - 1
                if last_visible[slot] >= 0
                else -1
            )
            read_source, read_entry = _select_read_entry(
                short_term[slot],
                long_term[slot],
                visibility_gap=visibility_gap,
                reentry_gap=config.reentry_gap,
            )
            accepted, reason = _write_decision(
                area=area,
                score=score,
                geometry_confidence=geometry_score,
                area_ratio=area_ratio,
                has_history=bool(history),
                config=config,
            )
            short_action = "none"
            long_action = "none"
            replaced_frame = -1
            quality = score * geometry_score
            if accepted:
                entry = MemoryEntry(
                    sequence_index=frame,
                    frame_index=int(frame_indices[frame]),
                    area=area,
                    track_score=score,
                    geometry_confidence=geometry_score,
                    quality=quality,
                )
                short_term[slot].append(entry)
                short_action = "append"
                if len(short_term[slot]) > config.short_term_capacity:
                    short_term[slot].pop(0)
                    short_action = "append_evict_oldest"
                if (
                    score >= config.long_term_min_track_score
                    and geometry_score
                    >= config.long_term_min_geometry_confidence
                ):
                    if len(long_term[slot]) < config.long_term_capacity:
                        long_term[slot].append(entry)
                        long_action = "append_clean_anchor"
                    else:
                        weakest = min(
                            range(len(long_term[slot])),
                            key=lambda index: (
                                long_term[slot][index].quality,
                                long_term[slot][index].sequence_index,
                            ),
                        )
                        if quality > long_term[slot][weakest].quality:
                            replaced_frame = long_term[slot][weakest].frame_index
                            long_term[slot][weakest] = entry
                            long_action = "replace_weakest_anchor"
                        else:
                            long_action = "keep_existing_anchors"
                else:
                    long_action = "not_clean_enough"
                write_mask[frame, slot] = True
            if area:
                last_visible[slot] = frame
            rows.append(
                {
                    "variant": str(variant),
                    "sequence_index": int(frame),
                    "frame_index": int(frame_indices[frame]),
                    "slot": int(slot),
                    "track_id": int(track_ids[slot]),
                    "visible": int(area > 0),
                    "mask_pixels": int(area),
                    "track_score": score,
                    "geometry_confidence": geometry_score,
                    "historical_area": historical_area,
                    "area_ratio": area_ratio,
                    "visibility_gap": int(visibility_gap),
                    "read_source": read_source,
                    "read_anchor_frame": (
                        int(read_entry.frame_index) if read_entry else -1
                    ),
                    "map_write": int(accepted),
                    "write_reason": reason,
                    "short_term_action": short_action,
                    "long_term_action": long_action,
                    "replaced_anchor_frame": int(replaced_frame),
                    "short_term_size": int(len(short_term[slot])),
                    "long_term_size": int(len(long_term[slot])),
                    "entry_quality": float(quality),
                    "memory_scope": "post_tracking_object_map_write_only",
                }
            )
    return {
        "map_write_mask": write_mask,
        "rows": rows,
        "summary": {
            "variant": str(variant),
            "frame_track_occurrences": int(sequence * tracks),
            "visible_observations": int(masks.flatten(2).any(dim=2).sum()),
            "accepted_map_writes": int(write_mask.sum()),
            "write_ratio_of_visible": (
                float(write_mask.sum())
                / max(1, int(masks.flatten(2).any(dim=2).sum()))
            ),
            "short_term_capacity": int(config.short_term_capacity),
            "long_term_capacity": int(config.long_term_capacity),
            "memory_scope": "post_tracking_object_map_write_only",
            "sam_hidden_memory_modified": 0,
        },
    }


def _select_read_entry(
    short_term: Sequence[MemoryEntry],
    long_term: Sequence[MemoryEntry],
    *,
    visibility_gap: int,
    reentry_gap: int,
) -> tuple[str, MemoryEntry | None]:
    if visibility_gap >= int(reentry_gap) and long_term:
        return "long_term_reentry_anchor", max(
            long_term,
            key=lambda entry: (entry.quality, -entry.sequence_index),
        )
    if short_term:
        return "short_term_recent", short_term[-1]
    if long_term:
        return "long_term_anchor", max(
            long_term,
            key=lambda entry: (entry.quality, -entry.sequence_index),
        )
    return "none", None


def _write_decision(
    *,
    area: int,
    score: float,
    geometry_confidence: float,
    area_ratio: float,
    has_history: bool,
    config: ObjectMemoryConfig,
) -> tuple[bool, str]:
    if area == 0:
        return False, "reject:empty_mask"
    if area < config.min_mask_pixels:
        return False, "reject:too_few_mask_pixels"
    if score < config.min_track_score:
        return False, "reject:low_track_score"
    if geometry_confidence < config.min_geometry_confidence:
        return False, "reject:low_geometry_confidence"
    if has_history and area_ratio < config.min_area_ratio:
        return False, "reject:area_too_small_vs_memory"
    if has_history and area_ratio > config.max_area_ratio:
        return False, "reject:area_too_large_vs_memory"
    return True, "accept:quality_and_area_consistent"


def _validate_config(config: ObjectMemoryConfig) -> None:
    if config.short_term_capacity < 1 or config.long_term_capacity < 1:
        raise ValueError("Object-memory capacities must be positive.")
    if config.min_mask_pixels < 1:
        raise ValueError("Object-memory min_mask_pixels must be positive.")
    if not 0.0 < config.min_area_ratio <= 1.0:
        raise ValueError("Object-memory min_area_ratio must be in (0,1].")
    if config.max_area_ratio < 1.0:
        raise ValueError("Object-memory max_area_ratio must be at least one.")
    if config.reentry_gap < 1:
        raise ValueError("Object-memory reentry_gap must be positive.")
    for name, value in (
        ("min_track_score", config.min_track_score),
        ("min_geometry_confidence", config.min_geometry_confidence),
        ("long_term_min_track_score", config.long_term_min_track_score),
        (
            "long_term_min_geometry_confidence",
            config.long_term_min_geometry_confidence,
        ),
    ):
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"Object-memory {name} must be in [0,1].")


@dataclass(frozen=True)
class PersistentObjectMemoryConfig:
    """Configuration for the deployable V1 persistent identity layer.

    ``ObjectMemoryConfig`` above remains the historical short/long write gate
    used by Stage 0.  This separate config makes it impossible for a V1 run to
    accidentally change the old Stage 0 policy.
    """

    max_points_per_object: int = 4096
    min_observation_points: int = 16
    min_mask_pixels: int = 32
    min_track_score: float = 0.50
    min_geometry_confidence: float = 0.30
    center_ema_alpha: float = 0.25
    same_frame_merge_score: float = 0.78
    association: InstanceAssociationConfig = field(
        default_factory=InstanceAssociationConfig
    )


@dataclass
class PersistentObject:
    """Long-lived object state accumulated from multiple SAM observations."""

    object_id: int
    category: str
    center: torch.Tensor
    covariance: torch.Tensor
    extent: torch.Tensor
    points: torch.Tensor
    voxel_keys: set[tuple[int, int, int]]
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

    def to_dict(self, *, include_points: bool = False) -> dict[str, object]:
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
            "voxel_count": int(len(self.voxel_keys)),
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
        return row


class PersistentObjectMemory:
    """Registry mapping causal SAM tracks to persistent 3-D object IDs.

    Existing SAM IDs are trusted for their active lifetime.  Association is
    invoked only for a previously unseen source track (birth/re-entry), which
    keeps the long-term memory from jittering on every frame.
    """

    def __init__(
        self,
        config: PersistentObjectMemoryConfig | None = None,
    ) -> None:
        self.config = config or PersistentObjectMemoryConfig()
        _validate_persistent_config(self.config)
        self.associator = InstanceAssociator(self.config.association)
        self.registry: dict[int, PersistentObject] = {}
        self.track_to_object: dict[str, int] = {}
        self.events: list[dict[str, object]] = []
        self.scene_scale: float = 1.0
        self._next_object_id = 0

    @property
    def objects(self) -> tuple[PersistentObject, ...]:
        return tuple(self.registry[key] for key in sorted(self.registry))

    @property
    def object_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self.registry))

    def process_observation(
        self,
        observation: InstanceObservation,
        *,
        scene_scale: float = 1.0,
        occupied_object_ids: Sequence[int] = (),
    ) -> dict[str, object]:
        """Update memory and return one auditable association event."""

        self.scene_scale = max(float(scene_scale), 1e-6)
        source_key = source_track_key(
            observation.sam_track_id,
            observation.source_slot,
        )
        direct_object_id = self.track_to_object.get(source_key)
        if direct_object_id is not None:
            obj = self.registry[int(direct_object_id)]
            self._update_object(obj, observation)
            event = _memory_event(
                observation,
                action="direct_active_track",
                object_id=obj.object_id,
                candidate=None,
                source_key=source_key,
            )
            self.events.append(event)
            return event

        candidates = self.associator.rank_candidates(
            observation,
            self.objects,
            scene_scale=self.scene_scale,
        )
        candidate = candidates[0] if candidates else None
        occupied = {int(value) for value in occupied_object_ids}
        if (
            candidate is not None
            and candidate.accepted
            and candidate.object_id in occupied
            and candidate.score < self.config.same_frame_merge_score
        ):
            candidate = _first_acceptable(
                candidates,
                occupied=occupied,
                min_score=self.config.same_frame_merge_score,
            )
        if candidate is not None and candidate.accepted:
            obj = self.registry[int(candidate.object_id)]
            self._update_object(obj, observation)
            self.track_to_object[source_key] = obj.object_id
            event = _memory_event(
                observation,
                action="associate_existing_object",
                object_id=obj.object_id,
                candidate=candidate,
                source_key=source_key,
            )
        else:
            obj = self._create_object(observation)
            self.track_to_object[source_key] = obj.object_id
            event = _memory_event(
                observation,
                action="create_new_object",
                object_id=obj.object_id,
                candidate=candidate,
                source_key=source_key,
            )
        self.events.append(event)
        return event

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
        appearance: torch.Tensor | None = None,
    ) -> dict[str, object]:
        """Process a complete cached clip in causal frame/slot order."""

        points = world_points.detach().float().cpu()
        confidence = confidence.detach().float().cpu()
        masks = masks.detach().bool().cpu()
        scores = track_scores.detach().float().cpu()
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
            raise ValueError("frame_indices do not match the sequence.")
        if len(track_ids) != tracks or len(track_prompts) != tracks:
            raise ValueError("Track IDs/prompts do not match mask slots.")
        if appearance is not None:
            appearance = appearance.detach().float().cpu()
            if appearance.ndim != 3 or appearance.shape[:2] != (sequence, tracks):
                raise ValueError("appearance must have shape [S,K,D].")

        self.scene_scale = estimate_scene_scale(points)
        persistent_ids = torch.full(
            (sequence, tracks),
            -1,
            dtype=torch.long,
        )
        write_mask = torch.zeros(sequence, tracks, dtype=torch.bool)
        local_events: list[dict[str, object]] = []
        for frame in range(sequence):
            occupied: dict[int, int] = {}
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
                        "action": _observation_rejection_reason(
                            area=area,
                            score=score,
                            geometry_score=geometry_score,
                            point_count=int(selected.shape[0]),
                            config=self.config,
                        ),
                        "source_key": source_track_key(
                            int(track_ids[slot]), slot
                        ),
                    }
                    self.events.append(event)
                    local_events.append(event)
                    continue
                appearance_value = (
                    appearance[frame, slot]
                    if appearance is not None
                    else None
                )
                observation = make_instance_observation(
                    sequence_index=frame,
                    frame_index=int(frame_indices[frame]),
                    source_slot=slot,
                    sam_track_id=int(track_ids[slot]),
                    category=str(track_prompts[slot]),
                    points=selected,
                    confidence=geometry_score,
                    track_score=score,
                    appearance=appearance_value,
                    max_points=self.config.association.max_points_per_comparison,
                )
                event = self.process_observation(
                    observation,
                    scene_scale=self.scene_scale,
                    occupied_object_ids=tuple(occupied),
                )
                event["mask_pixels"] = int(area)
                event["geometry_points"] = int(selected.shape[0])
                event["geometry_confidence"] = float(geometry_score)
                persistent_ids[frame, slot] = int(event["persistent_object_id"])
                write_mask[frame, slot] = True
                occupied[int(event["persistent_object_id"])] = slot
                local_events.append(event)

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
        }

    def to_dict(self, *, include_points: bool = False) -> dict[str, object]:
        return {
            "schema": 1,
            "revision": "v1_persistent_3d_instance_memory_r1",
            "scene_scale": float(self.scene_scale),
            "persistent_object_count": int(len(self.registry)),
            "objects": [
                obj.to_dict(include_points=include_points) for obj in self.objects
            ],
            "track_to_object": {
                str(key): int(value)
                for key, value in sorted(self.track_to_object.items())
            },
            "event_count": int(len(self.events)),
        }

    def save_json(self, path: str | Path, *, include_points: bool = False) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_dict(include_points=include_points), indent=2) + "\n",
            encoding="utf8",
        )
        return output

    def save_csv(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        rows = [obj.to_dict() for obj in self.objects]
        if not rows:
            output.write_text("object_id,category\n", encoding="utf8")
            return output
        fields = tuple(rows[0])
        with output.open("w", encoding="utf8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return output

    def _create_object(self, observation: InstanceObservation) -> PersistentObject:
        object_id = self._next_object_id
        self._next_object_id += 1
        points = deterministic_points(
            observation.points,
            limit=self.config.max_points_per_object,
        )
        obj = PersistentObject(
            object_id=object_id,
            category=normalize_category(observation.category) or "object",
            center=observation.center.clone(),
            covariance=observation.covariance.clone(),
            extent=observation.extent.clone(),
            points=points.clone(),
            voxel_keys=set(
                voxel_keys(
                    points,
                    voxel_size=_memory_voxel_size(self.scene_scale, self.config),
                )
            ),
            point_count=int(observation.points.shape[0]),
            observation_count=1,
            first_sequence_index=int(observation.sequence_index),
            last_sequence_index=int(observation.sequence_index),
            first_frame_index=int(observation.frame_index),
            last_frame_index=int(observation.frame_index),
            confidence=float(observation.quality),
            appearance=_normalise_appearance(observation.appearance),
            observation_timestamps=[int(observation.frame_index)],
            sam_track_ids={int(observation.sam_track_id)},
            source_slots={int(observation.source_slot)},
        )
        self.registry[object_id] = obj
        return obj

    def _update_object(
        self,
        obj: PersistentObject,
        observation: InstanceObservation,
    ) -> None:
        alpha = float(self.config.center_ema_alpha)
        obj.center = (1.0 - alpha) * obj.center + alpha * observation.center
        obj.extent = torch.maximum(obj.extent, observation.extent)
        obj.confidence = (1.0 - alpha) * obj.confidence + alpha * observation.quality
        obj.observation_count += 1
        obj.point_count += int(observation.points.shape[0])
        obj.last_sequence_index = int(observation.sequence_index)
        obj.last_frame_index = int(observation.frame_index)
        obj.observation_timestamps.append(int(observation.frame_index))
        obj.sam_track_ids.add(int(observation.sam_track_id))
        obj.source_slots.add(int(observation.source_slot))
        if len(obj.observation_timestamps) > 1024:
            del obj.observation_timestamps[:-1024]
        incoming = deterministic_points(
            observation.points,
            limit=self.config.max_points_per_object,
        )
        combined = torch.cat((obj.points, incoming), dim=0)
        obj.points = deterministic_points(
            combined,
            limit=self.config.max_points_per_object,
        )
        obj.voxel_keys = set(
            voxel_keys(
                obj.points,
                voxel_size=_memory_voxel_size(self.scene_scale, self.config),
            )
        )
        obj.covariance = _point_covariance(obj.points)
        appearance = _normalise_appearance(observation.appearance)
        if appearance is not None:
            if obj.appearance is None or obj.appearance.shape != appearance.shape:
                obj.appearance = appearance
            else:
                obj.appearance = (
                    (1.0 - alpha) * obj.appearance + alpha * appearance
                )
                obj.appearance = _normalise_appearance(obj.appearance)


def make_instance_observation(
    *,
    sequence_index: int,
    frame_index: int,
    source_slot: int,
    sam_track_id: int,
    category: str,
    points: torch.Tensor,
    confidence: float,
    track_score: float,
    appearance: torch.Tensor | None = None,
    max_points: int = 256,
) -> InstanceObservation:
    values = deterministic_points(points, limit=max_points)
    if values.shape[0] < 1:
        raise ValueError("Instance observations require at least one point.")
    center = values.mean(dim=0)
    covariance = _point_covariance(values)
    extent = values.max(dim=0).values - values.min(dim=0).values
    return InstanceObservation(
        sequence_index=int(sequence_index),
        frame_index=int(frame_index),
        source_slot=int(source_slot),
        sam_track_id=int(sam_track_id),
        category=str(category),
        points=values,
        center=center,
        covariance=covariance,
        extent=extent,
        confidence=float(confidence),
        track_score=float(track_score),
        appearance=_normalise_appearance(appearance),
    )


def collapse_persistent_tracks(
    *,
    masks: torch.Tensor,
    scores: torch.Tensor,
    persistent_object_ids: torch.Tensor,
    object_ids: Sequence[int] | None = None,
    object_prompts: Mapping[int, str] | None = None,
) -> dict[str, object]:
    """Merge all SAM slots assigned to each persistent object."""

    masks = masks.detach().bool().cpu()
    scores = scores.detach().float().cpu()
    ids = persistent_object_ids.detach().long().cpu()
    if masks.ndim != 4 or ids.shape != masks.shape[:2]:
        raise ValueError("masks and persistent_object_ids have incompatible shapes.")
    if tuple(scores.shape) != masks.shape[:2]:
        raise ValueError("scores and masks have incompatible shapes.")
    values = (
        sorted({int(value) for value in ids.tolist() if int(value) >= 0})
        if object_ids is None
        else [int(value) for value in object_ids]
    )
    output_masks = torch.zeros(
        masks.shape[0], len(values), masks.shape[2], masks.shape[3], dtype=torch.bool
    )
    output_scores = torch.zeros(masks.shape[0], len(values), dtype=torch.float32)
    prompts: list[str] = []
    for output_slot, object_id in enumerate(values):
        members = ids == int(object_id)
        member_masks = masks & members[:, :, None, None]
        output_masks[:, output_slot] = member_masks.any(dim=1)
        masked_scores = torch.where(
            members,
            scores,
            torch.zeros_like(scores),
        )
        output_scores[:, output_slot] = masked_scores.max(dim=1).values
        prompts.append(
            str((object_prompts or {}).get(int(object_id), "object"))
        )
    return {
        "masks": output_masks,
        "scores": output_scores,
        "object_ids": values,
        "prompts": prompts,
    }


def source_track_key(sam_track_id: int, source_slot: int) -> str:
    if int(sam_track_id) >= 0:
        return f"sam:{int(sam_track_id)}"
    return f"slot:{int(source_slot)}"


def estimate_scene_scale(points: torch.Tensor) -> float:
    values = points.detach().float().cpu().reshape(-1, 3)
    values = values[torch.isfinite(values).all(dim=1)]
    if not values.numel():
        return 1.0
    low = torch.quantile(values, 0.05, dim=0)
    high = torch.quantile(values, 0.95, dim=0)
    scale = float(torch.linalg.vector_norm(high - low))
    return max(scale, 1e-6)


def _point_covariance(points: torch.Tensor) -> torch.Tensor:
    values = points.detach().float().cpu().reshape(-1, 3)
    values = values[torch.isfinite(values).all(dim=1)]
    if values.shape[0] <= 1:
        return torch.eye(3, dtype=torch.float32) * 1e-6
    centered = values - values.mean(dim=0, keepdim=True)
    return centered.T @ centered / float(max(1, values.shape[0] - 1))


def _normalise_appearance(value: torch.Tensor | None) -> torch.Tensor | None:
    if value is None:
        return None
    tensor = value.detach().float().cpu().reshape(-1)
    if not tensor.numel() or not bool(torch.isfinite(tensor).all()):
        return None
    norm = float(torch.linalg.vector_norm(tensor))
    return tensor / norm if norm > 1e-8 else None


def _memory_voxel_size(
    scene_scale: float,
    config: PersistentObjectMemoryConfig,
) -> float:
    association = config.association
    return max(
        float(association.absolute_voxel_size),
        float(scene_scale) * float(association.voxel_size_ratio),
    )


def _first_acceptable(
    candidates: Sequence[AssociationCandidate],
    *,
    occupied: set[int],
    min_score: float,
) -> AssociationCandidate | None:
    for candidate in candidates:
        if candidate.accepted and (
            candidate.object_id not in occupied
            or candidate.score >= float(min_score)
        ):
            return candidate
    return None


def _memory_event(
    observation: InstanceObservation,
    *,
    action: str,
    object_id: int,
    candidate: AssociationCandidate | None,
    source_key: str,
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
        "persistent_object_id": int(object_id),
        "action": str(action),
        "source_key": str(source_key),
    }
    if candidate is not None:
        row.update({f"association_{key}": value for key, value in candidate.to_dict().items()})
    else:
        row.update(
            {
                "association_object_id": -1,
                "association_score": float("nan"),
                "association_accepted": 0,
                "association_center_distance": float("nan"),
                "association_center_similarity": float("nan"),
                "association_voxel_iou": float("nan"),
                "association_chamfer_distance": float("nan"),
                "association_chamfer_similarity": float("nan"),
                "association_appearance_cosine": float("nan"),
                "association_category_consistency": float("nan"),
                "association_compared_features": "",
                "association_rejection_reason": "direct_track_or_no_candidate",
            }
        )
    return row


def _observation_rejection_reason(
    *,
    area: int,
    score: float,
    geometry_score: float,
    point_count: int,
    config: PersistentObjectMemoryConfig,
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


def _validate_persistent_config(config: PersistentObjectMemoryConfig) -> None:
    if config.max_points_per_object < 1 or config.min_observation_points < 1:
        raise ValueError("Persistent object point limits must be positive.")
    if config.min_mask_pixels < 1:
        raise ValueError("Persistent object min_mask_pixels must be positive.")
    for name, value in (
        ("min_track_score", config.min_track_score),
        ("min_geometry_confidence", config.min_geometry_confidence),
        ("center_ema_alpha", config.center_ema_alpha),
        ("same_frame_merge_score", config.same_frame_merge_score),
    ):
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"Persistent object {name} must be in [0,1].")
