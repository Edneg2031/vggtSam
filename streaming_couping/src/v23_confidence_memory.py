"""Confidence-aware voxel memory for the V2.3 semantic-map ablation.

V2.3 deliberately runs after the frozen V2.2 recovery decision.  The tracking
mask is never changed here.  This module only decides which world points from
each final observation are allowed to update the long-term map and exposes a
map-only mask for offline evaluation.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

import torch


SOURCE_CONFIDENCE: dict[str, float] = {
    "raw_sam": 1.0,
    "validated_recovery": 0.7,
    "recovery_candidate": 0.3,
}


@dataclass(frozen=True)
class V23ConfidenceMemoryConfig:
    """Inference-time confidence and voxel-fusion policy."""

    voxel_size_m: float = 0.05
    max_voxels_per_object: int = 4096
    max_points_per_observation: int = 2048
    point_confidence_floor: float = 0.30
    min_observation_score: float = 0.50
    raw_observation_confidence: float = 1.0
    validated_recovery_confidence: float = 0.70
    recovery_candidate_confidence: float = 0.30
    confidence_decay: float = 0.98
    recovery_support_distance_m: float = 0.10
    min_recovery_support_ratio: float = 0.20
    min_recovery_supported_points: int = 8
    distance_chunk_size: int = 512
    covariance_epsilon: float = 1.0e-6

    def source_confidence(self, source_type: str) -> float:
        configured = {
            "raw_sam": self.raw_observation_confidence,
            "validated_recovery": self.validated_recovery_confidence,
            "recovery_candidate": self.recovery_candidate_confidence,
        }
        return float(configured.get(str(source_type), SOURCE_CONFIDENCE["recovery_candidate"]))


@dataclass
class _VoxelRecord:
    position: torch.Tensor
    confidence: float
    observation_count: int = 1
    raw_observation_count: int = 0
    recovery_observation_count: int = 0


def _finite_points(points: torch.Tensor) -> torch.Tensor:
    values = points.detach().float().cpu().reshape(-1, 3)
    return values[torch.isfinite(values).all(dim=-1)]


def _nearest_distances(
    source: torch.Tensor,
    target: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    if source.numel() == 0 or target.numel() == 0:
        return torch.empty(0, dtype=torch.float32)
    source = source.detach().float().cpu().reshape(-1, 3)
    target = target.detach().float().cpu().reshape(-1, 3)
    rows: list[torch.Tensor] = []
    for start in range(0, int(source.shape[0]), max(1, int(chunk_size))):
        distance = torch.cdist(source[start : start + int(chunk_size)], target)
        rows.append(distance.min(dim=1).values)
    return torch.cat(rows, dim=0)


def _voxel_key(point: torch.Tensor, voxel_size: float) -> tuple[int, int, int]:
    value = point.detach().float().cpu().reshape(3)
    return tuple(int(math.floor(float(axis) / float(voxel_size))) for axis in value)


def _top_indices(values: torch.Tensor, limit: int) -> torch.Tensor:
    values = values.detach().float().cpu().reshape(-1)
    limit = max(1, int(limit))
    if values.numel() <= limit:
        return torch.arange(values.numel(), dtype=torch.long)
    return torch.topk(values, k=limit, sorted=False).indices.sort().values


def _covariance(points: torch.Tensor, epsilon: float) -> torch.Tensor:
    values = _finite_points(points)
    if int(values.shape[0]) < 2:
        return torch.zeros(3, 3)
    centered = values - values.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(1, int(values.shape[0]) - 1)
    return torch.nan_to_num(covariance, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(
        float(epsilon)
    )


@dataclass
class V23ObjectMemoryState:
    """Voxel memory and confidence counters for one fixed SAM slot."""

    object_id: int
    prompt: str
    sam_track_id: int
    config: V23ConfidenceMemoryConfig
    voxels: dict[tuple[int, int, int], _VoxelRecord] = field(default_factory=dict)
    observation_count: int = 0
    raw_observation_count: int = 0
    recovery_observation_count: int = 0
    rejected_observation_count: int = 0
    low_confidence_reject_count: int = 0
    voxel_conflict_reject_count: int = 0
    last_seen_sequence_index: int = -1
    last_seen_frame_index: int = -1
    source_counts: Counter[str] = field(default_factory=Counter)

    @property
    def world_points(self) -> torch.Tensor:
        if not self.voxels:
            return torch.empty(0, 3, dtype=torch.float32)
        return torch.stack(
            [self.voxels[key].position for key in sorted(self.voxels)], dim=0
        ).float().cpu()

    @property
    def world_weights(self) -> torch.Tensor:
        if not self.voxels:
            return torch.empty(0, dtype=torch.float32)
        return torch.tensor(
            [self.voxels[key].confidence for key in sorted(self.voxels)],
            dtype=torch.float32,
        )

    @property
    def has_history(self) -> bool:
        return bool(self.voxels)

    @property
    def mean_confidence(self) -> float:
        weights = self.world_weights
        return float(weights.mean().item()) if weights.numel() else 0.0

    @property
    def recovery_voxel_count(self) -> int:
        return sum(
            int(record.recovery_observation_count > 0)
            for record in self.voxels.values()
        )

    def support_mask(self, points: torch.Tensor) -> torch.Tensor:
        """Return point-level support from historical voxel centers."""

        values = points.detach().float().cpu().reshape(-1, 3)
        if not self.voxels or not values.numel():
            return torch.zeros(values.shape[0], dtype=torch.bool)
        historical = self.world_points
        nearest = _nearest_distances(
            values,
            historical,
            int(self.config.distance_chunk_size),
        )
        return nearest <= float(self.config.recovery_support_distance_m)

    def update(
        self,
        *,
        points: torch.Tensor,
        point_confidence: torch.Tensor,
        source_type: str,
        sequence_index: int,
        frame_index: int,
    ) -> dict[str, int]:
        """Fuse an accepted observation into the voxel dictionary."""

        values = points.detach().float().cpu().reshape(-1, 3)
        geometry_confidence = point_confidence.detach().float().cpu().reshape(-1)
        if values.shape[0] != geometry_confidence.shape[0]:
            raise ValueError("V2.3 points and point confidence disagree.")
        valid = torch.isfinite(values).all(dim=-1) & torch.isfinite(geometry_confidence)
        valid &= geometry_confidence >= float(self.config.point_confidence_floor)
        values = values[valid]
        geometry_confidence = geometry_confidence[valid].clamp(0.0, 1.0)
        source = str(source_type)
        source_confidence = self.config.source_confidence(source)
        accepted = conflicts = 0
        decay = max(0.0, min(1.0, float(self.config.confidence_decay)))
        for point, geometry_value in zip(values, geometry_confidence):
            key = _voxel_key(point, float(self.config.voxel_size_m))
            effective = max(
                0.0,
                min(1.0, float(geometry_value.item()) * source_confidence),
            )
            previous = self.voxels.get(key)
            if previous is None:
                if len(self.voxels) >= int(self.config.max_voxels_per_object):
                    weakest_key = min(
                        self.voxels,
                        key=lambda item: (
                            self.voxels[item].confidence,
                            self.voxels[item].observation_count,
                            item,
                        ),
                    )
                    weakest = self.voxels[weakest_key]
                    if effective <= float(weakest.confidence):
                        conflicts += 1
                        continue
                    del self.voxels[weakest_key]
                self.voxels[key] = _VoxelRecord(
                    position=point.clone(),
                    confidence=effective,
                    observation_count=1,
                    raw_observation_count=int(source == "raw_sam"),
                    recovery_observation_count=int(source == "validated_recovery"),
                )
                accepted += 1
                continue

            old_count = max(1, int(previous.observation_count))
            old_weight = max(
                float(previous.confidence) * float(old_count) * decay,
                float(self.config.covariance_epsilon),
            )
            new_weight = max(effective, float(self.config.covariance_epsilon))
            previous.position = (
                previous.position * old_weight + point * new_weight
            ) / (old_weight + new_weight)
            previous.confidence = max(
                0.0,
                min(
                    1.0,
                    (
                        float(previous.confidence) * float(old_count) * decay
                        + effective
                    )
                    / (float(old_count) * decay + 1.0),
                ),
            )
            previous.observation_count += 1
            previous.raw_observation_count += int(source == "raw_sam")
            previous.recovery_observation_count += int(
                source == "validated_recovery"
            )
            accepted += 1

        self.observation_count += int(bool(accepted))
        self.source_counts[source] += int(bool(accepted))
        if source == "raw_sam":
            self.raw_observation_count += int(bool(accepted))
        elif source == "validated_recovery":
            self.recovery_observation_count += int(bool(accepted))
        self.voxel_conflict_reject_count += int(conflicts)
        if accepted:
            self.last_seen_sequence_index = int(sequence_index)
            self.last_seen_frame_index = int(frame_index)
        return {"accepted_points": int(accepted), "conflict_points": int(conflicts)}

    def record_rejected(self, *, low_confidence: bool, voxel_conflict: bool) -> None:
        self.rejected_observation_count += 1
        self.low_confidence_reject_count += int(bool(low_confidence))
        self.voxel_conflict_reject_count += int(bool(voxel_conflict))

    def metadata(self) -> dict[str, Any]:
        points = self.world_points
        center = points.mean(dim=0) if points.numel() else torch.zeros(3)
        covariance = _covariance(points, float(self.config.covariance_epsilon))
        return {
            "object_id": int(self.object_id),
            "prompt": str(self.prompt),
            "sam_track_id": int(self.sam_track_id),
            "observation_count": int(self.observation_count),
            "raw_observation_count": int(self.raw_observation_count),
            "recovery_observation_count": int(self.recovery_observation_count),
            "rejected_observation_count": int(self.rejected_observation_count),
            "low_confidence_reject_count": int(self.low_confidence_reject_count),
            "voxel_conflict_reject_count": int(self.voxel_conflict_reject_count),
            "last_seen_sequence_index": int(self.last_seen_sequence_index),
            "last_seen_frame_index": int(self.last_seen_frame_index),
            "voxel_count": int(len(self.voxels)),
            "mean_confidence": float(self.mean_confidence),
            "recovery_voxel_count": int(self.recovery_voxel_count),
            "recovery_voxel_ratio": float(self.recovery_voxel_count / max(1, len(self.voxels))),
            "source_counts": dict(sorted(self.source_counts.items())),
            "world_centroid": [float(value) for value in center.tolist()],
            "world_bbox_min": (
                [float(value) for value in points.min(dim=0).values.tolist()]
                if points.numel()
                else []
            ),
            "world_bbox_max": (
                [float(value) for value in points.max(dim=0).values.tolist()]
                if points.numel()
                else []
            ),
            "world_covariance": [
                [float(value) for value in row] for row in covariance.tolist()
            ],
        }


class V23ConfidenceMemoryBank:
    """One confidence-aware voxel memory entry per frozen SAM slot."""

    def __init__(
        self,
        *,
        track_ids: Sequence[int],
        prompts: Sequence[str],
        config: V23ConfidenceMemoryConfig,
    ) -> None:
        if len(track_ids) != len(prompts):
            raise ValueError("V2.3 track IDs and prompts must have equal length.")
        self.objects = {
            int(slot): V23ObjectMemoryState(
                object_id=int(slot),
                prompt=str(prompts[slot]),
                sam_track_id=int(track_ids[slot]),
                config=config,
            )
            for slot in range(len(track_ids))
        }

    def get(self, slot: int) -> V23ObjectMemoryState:
        return self.objects[int(slot)]

    def metadata(self) -> list[dict[str, Any]]:
        return [self.objects[key].metadata() for key in sorted(self.objects)]

    def point_tensors(self) -> dict[int, dict[str, torch.Tensor]]:
        return {
            int(slot): {
                "world_points": state.world_points,
                "world_weights": state.world_weights,
            }
            for slot, state in sorted(self.objects.items())
        }

    def confidence_summary(self) -> dict[str, Any]:
        objects = self.metadata()
        total_voxels = sum(int(row["voxel_count"]) for row in objects)
        recovery_voxels = sum(int(row["recovery_voxel_count"]) for row in objects)
        return {
            "schema": 1,
            "objects": objects,
            "average_object_confidence": (
                sum(float(row["mean_confidence"]) for row in objects) / len(objects)
                if objects
                else 0.0
            ),
            "recovery_point_ratio": recovery_voxels / max(1, total_voxels),
            "total_voxel_count": int(total_voxels),
            "recovery_voxel_count": int(recovery_voxels),
        }


def process_v23_memory_sequence(
    *,
    world_points: torch.Tensor,
    confidence: torch.Tensor,
    masks_stream: torch.Tensor,
    scores: torch.Tensor,
    events: Sequence[Mapping[str, Any]],
    frame_indices: Sequence[int],
    track_ids: Sequence[int],
    track_prompts: Sequence[str],
    config: V23ConfidenceMemoryConfig,
) -> dict[str, Any]:
    """Run causal confidence-aware map writes on the frozen V2.2 masks."""

    points = world_points.detach().float().cpu()
    point_confidence = confidence.detach().float().cpu()
    masks = masks_stream.detach().bool().cpu()
    scores_cpu = scores.detach().float().cpu()
    if points.ndim != 4 or points.shape[-1] != 3:
        raise ValueError("V2.3 world_points must have shape [S,H,W,3].")
    sequence, height, width = points.shape[:3]
    if point_confidence.shape != (sequence, height, width):
        raise ValueError("V2.3 confidence does not match world_points.")
    if masks.ndim != 4 or tuple(masks.shape[0:1] + masks.shape[2:]) != (
        sequence,
        height,
        width,
    ):
        raise ValueError("V2.3 masks_stream does not match the pointmap grid.")
    tracks = int(masks.shape[1])
    if tuple(scores_cpu.shape) != (sequence, tracks):
        raise ValueError("V2.3 scores do not match masks_stream.")
    if len(events) != sequence * tracks:
        raise ValueError("V2.3 recovery events must contain one row per frame/slot.")
    if len(frame_indices) != sequence:
        raise ValueError("V2.3 frame metadata length mismatch.")

    bank = V23ConfidenceMemoryBank(
        track_ids=track_ids,
        prompts=track_prompts,
        config=config,
    )
    map_masks = torch.zeros_like(masks)
    map_write = torch.zeros(sequence, tracks, dtype=torch.bool)
    map_observation_confidence = torch.zeros(sequence, tracks, dtype=torch.float32)
    updated_events: list[dict[str, Any]] = [dict(row) for row in events]
    stats_counter: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    recovery_points = 0
    total_points = 0
    low_confidence_points = 0
    low_confidence_rejects = 0
    conflict_rejects = 0
    memory_updates = raw_updates = recovery_updates = 0

    for frame in range(sequence):
        for slot in range(tracks):
            event_index = frame * tracks + slot
            event = updated_events[event_index]
            final_mask = masks[frame, slot]
            final_score = float(scores_cpu[frame, slot])
            applied = bool(event.get("recovery_applied", 0))
            if applied:
                source = "validated_recovery"
            elif bool(final_mask.any()):
                source = "raw_sam"
            else:
                source = "recovery_candidate"
            observation_confidence = config.source_confidence(source)
            event.update(
                {
                    "v23_source_type": source,
                    "v23_observation_confidence": float(observation_confidence),
                    "v23_candidate_confidence": float(
                        config.recovery_candidate_confidence
                    ),
                    "v23_map_write": 0,
                    "v23_memory_updated": 0,
                    "v23_memory_reason": "not_processed",
                    "v23_support_ratio": float("nan"),
                    "v23_supported_point_count": 0,
                    "v23_rejected_point_count": 0,
                }
            )
            state = bank.get(slot)
            valid = (
                final_mask
                & torch.isfinite(points[frame]).all(dim=-1)
                & torch.isfinite(point_confidence[frame])
                & (point_confidence[frame] >= float(config.point_confidence_floor))
            )
            all_pixels = int(final_mask.sum())
            valid_pixels = int(valid.sum())
            low_pixels = max(0, all_pixels - valid_pixels)
            low_confidence_points += low_pixels
            total_points += all_pixels
            low_confidence = (
                final_score < float(config.min_observation_score)
                or valid_pixels == 0
            )
            if low_confidence:
                low_confidence_rejects += 1
                state.record_rejected(low_confidence=True, voxel_conflict=False)
                stats_counter["low_confidence_reject"] += 1
                event["v23_memory_reason"] = "low_confidence_or_no_valid_points"
                event["v23_rejected_point_count"] = int(valid_pixels)
                continue

            pixel_indices = torch.nonzero(valid.reshape(-1), as_tuple=False)[:, 0]
            selected_local = _top_indices(
                point_confidence[frame].reshape(-1).index_select(0, pixel_indices),
                int(config.max_points_per_observation),
            )
            selected_pixels = pixel_indices.index_select(0, selected_local)
            selected_points = points[frame].reshape(-1, 3).index_select(
                0, selected_pixels
            )
            selected_confidence = point_confidence[frame].reshape(-1).index_select(
                0, selected_pixels
            )

            if source == "validated_recovery":
                support = state.support_mask(selected_points)
                support_ratio = (
                    float(support.float().mean().item()) if support.numel() else 0.0
                )
                event["v23_support_ratio"] = float(support_ratio)
                event["v23_supported_point_count"] = int(support.sum())
                event["v23_rejected_point_count"] = int((~support).sum())
                if (
                    support_ratio < float(config.min_recovery_support_ratio)
                    or int(support.sum()) < int(config.min_recovery_supported_points)
                ):
                    state.record_rejected(low_confidence=False, voxel_conflict=True)
                    conflict_rejects += 1
                    stats_counter["voxel_conflict_reject"] += 1
                    event["v23_memory_reason"] = "recovery_support_below_threshold"
                    continue
                accepted_points = selected_points[support]
                accepted_confidence = selected_confidence[support]
                map_pixels = selected_pixels[support]
                recovery_points += int(accepted_points.shape[0])
                recovery_updates += 1
            else:
                accepted_points = selected_points
                accepted_confidence = selected_confidence
                map_pixels = pixel_indices
                raw_updates += 1

            updated = state.update(
                points=accepted_points,
                point_confidence=accepted_confidence,
                source_type=source,
                sequence_index=frame,
                frame_index=int(frame_indices[frame]),
            )
            if int(updated["accepted_points"]) <= 0:
                state.record_rejected(low_confidence=False, voxel_conflict=True)
                conflict_rejects += 1
                stats_counter["voxel_conflict_reject"] += 1
                event["v23_memory_reason"] = "voxel_capacity_conflict"
                event["v23_rejected_point_count"] = int(
                    event["v23_rejected_point_count"]
                ) + int(accepted_points.shape[0])
                continue

            flat_map = map_masks[frame, slot].reshape(-1)
            flat_map[map_pixels] = True
            map_write[frame, slot] = True
            map_observation_confidence[frame, slot] = float(observation_confidence)
            event["v23_map_write"] = 1
            event["v23_memory_updated"] = 1
            event["v23_memory_reason"] = "voxel_fused"
            memory_updates += 1
            source_counts[source] += 1
            stats_counter[f"{source}_update"] += 1
            conflict_rejects += int(updated["conflict_points"])

    objects = bank.metadata()
    confidence_summary = bank.confidence_summary()
    confidence_summary["low_confidence_point_ratio"] = (
        low_confidence_points / max(1, total_points)
    )
    confidence_summary["recovery_observation_point_ratio"] = (
        recovery_points / max(1, recovery_points + total_points - low_confidence_points)
    )
    stats = {
        "memory_update_count": int(memory_updates),
        "raw_observation_update": int(raw_updates),
        "validated_recovery_update": int(recovery_updates),
        "low_confidence_reject_count": int(low_confidence_rejects),
        "voxel_conflict_reject_count": int(conflict_rejects),
        "source_update_counts": dict(sorted(source_counts.items())),
        "memory_reject_reason_counts": dict(sorted(stats_counter.items())),
        "average_object_confidence": float(
            confidence_summary["average_object_confidence"]
        ),
        "recovery_point_ratio": float(confidence_summary["recovery_point_ratio"]),
        "recovery_observation_point_ratio": float(
            confidence_summary["recovery_observation_point_ratio"]
        ),
        "low_confidence_point_ratio": float(
            confidence_summary["low_confidence_point_ratio"]
        ),
        "total_memory_voxels": int(confidence_summary["total_voxel_count"]),
        "recovery_memory_voxels": int(confidence_summary["recovery_voxel_count"]),
    }
    return {
        "map_masks_stream": map_masks,
        "map_write_mask": map_write,
        "map_observation_confidence": map_observation_confidence,
        "events": updated_events,
        "objects": objects,
        "object_tensors": bank.point_tensors(),
        "object_memory_confidence": confidence_summary,
        "stats": stats,
    }


__all__ = [
    "SOURCE_CONFIDENCE",
    "V23ConfidenceMemoryConfig",
    "V23ObjectMemoryState",
    "V23ConfidenceMemoryBank",
    "process_v23_memory_sequence",
]
