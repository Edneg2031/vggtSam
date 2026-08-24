"""Auditable short/long object memory for semantic-map observation writes.

This module deliberately does not modify SAM hidden memory.  It consumes a
finished persistent track and decides which 2-D/3-D observations may update an
object-level semantic map.  Every read, rejection, write, and replacement is
returned as tabular provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Sequence

import torch


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
