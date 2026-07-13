"""Separate confidence gates for map updates and low-confidence recovery."""

from __future__ import annotations

from dataclasses import dataclass

from .types import GateDecision


@dataclass(frozen=True)
class GateConfig:
    track_update_threshold: float = 0.7
    track_fallback_threshold: float = 0.5
    geometry_threshold: float = 0.45
    min_persistence: int = 1


def decide_gates(
    *,
    track_confidence: float,
    update_geometry_confidence: float,
    fallback_geometry_confidence: float,
    persistence: int,
    has_object_map: bool,
    config: GateConfig,
) -> GateDecision:
    """Make update and recovery decisions independently.

    A reliable tracker observation may update the map. A weak tracker observation
    may trigger geometry recovery, provided an older reliable map exists. Keeping
    these gates separate avoids requiring high tracker confidence to recover a
    tracker failure.
    """

    update_geometry_ok = update_geometry_confidence >= config.geometry_threshold
    fallback_geometry_ok = fallback_geometry_confidence >= config.geometry_threshold
    persistent = persistence >= config.min_persistence
    update = (
        track_confidence >= config.track_update_threshold
        and update_geometry_ok
        and persistent
    )
    fallback = (
        track_confidence < config.track_fallback_threshold
        and fallback_geometry_ok
        and has_object_map
    )
    if update:
        reason = "reliable tracker and geometry: update object map"
    elif fallback:
        reason = "weak tracker with reliable history: use 3D fallback"
    elif track_confidence < config.track_fallback_threshold and not fallback_geometry_ok:
        reason = "projected-prior geometry confidence below threshold"
    elif not update_geometry_ok:
        reason = "tracker-region geometry confidence below threshold"
    elif not persistent:
        reason = "instance persistence below threshold"
    else:
        reason = "keep SAM3 prediction without map update"
    return GateDecision(
        update_map=update,
        use_fallback=fallback,
        track_confidence=float(track_confidence),
        update_geometry_confidence=float(update_geometry_confidence),
        fallback_geometry_confidence=float(fallback_geometry_confidence),
        persistence=int(persistence),
        reason=reason,
    )
