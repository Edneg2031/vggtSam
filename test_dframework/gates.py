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
    geometry_confidence: float,
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

    geometry_ok = geometry_confidence >= config.geometry_threshold
    persistent = persistence >= config.min_persistence
    update = (
        track_confidence >= config.track_update_threshold
        and geometry_ok
        and persistent
    )
    fallback = (
        track_confidence < config.track_fallback_threshold
        and geometry_ok
        and has_object_map
    )
    if update:
        reason = "reliable tracker and geometry: update object map"
    elif fallback:
        reason = "weak tracker with reliable history: use 3D fallback"
    elif not geometry_ok:
        reason = "geometry confidence below threshold"
    elif not persistent:
        reason = "instance persistence below threshold"
    else:
        reason = "keep SAM3 prediction without map update"
    return GateDecision(
        update_map=update,
        use_fallback=fallback,
        track_confidence=float(track_confidence),
        geometry_confidence=float(geometry_confidence),
        persistence=int(persistence),
        reason=reason,
    )

