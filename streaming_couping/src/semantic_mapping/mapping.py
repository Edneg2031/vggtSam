"""Causal semantic-instance point and voxel fusion."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from typing import Any

import torch

from .contracts import GeometryFrame, ObjectObservation, SegmentationFrame
from .geometry import (
    geometry_confidence_for_frame,
    resize_bool_mask,
    rgb_for_frame,
    world_points_for_frame,
)


@dataclass(frozen=True)
class SemanticMapConfig:
    """Policy for map fusion; model backends do not need to know this type."""

    voxel_size_m: float = 0.05
    min_geometry_confidence: float = 0.30
    min_track_score: float = 0.50
    static_score_threshold: float = 0.20
    require_static_score: bool = False
    include_dynamic_tracks: bool = True
    max_points_per_observation: int = 8_000
    max_points_per_track: int = 250_000
    max_voxels: int = 1_000_000
    max_scene_voxels: int = 1_000_000
    map_write_gate: "MapWriteGateConfig" = field(
        default_factory=lambda: MapWriteGateConfig()
    )

    def validate(self) -> "SemanticMapConfig":
        if float(self.voxel_size_m) <= 0.0:
            raise ValueError("voxel_size_m must be positive.")
        for name, value in (
            ("min_geometry_confidence", self.min_geometry_confidence),
            ("min_track_score", self.min_track_score),
            ("static_score_threshold", self.static_score_threshold),
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0,1].")
        for name, value in (
            ("max_points_per_observation", self.max_points_per_observation),
            ("max_points_per_track", self.max_points_per_track),
            ("max_voxels", self.max_voxels),
            ("max_scene_voxels", self.max_scene_voxels),
        ):
            if int(value) < 1:
                raise ValueError(f"{name} must be positive.")
        self.map_write_gate.validate()
        return self


@dataclass(frozen=True)
class MapWriteGateConfig:
    """Optional causal short/long memory policy for semantic voxel writes.

    This gate is deliberately disabled by default to preserve the established
    raw baseline.  When enabled, every valid observation still remains in the
    object track; only semantic voxel insertion is delayed or down-weighted.
    """

    enabled: bool = False
    short_term_capacity: int = 5
    long_term_capacity: int = 4
    min_observations_before_write: int = 2
    reentry_confirmation_frames: int = 2
    reentry_gap: int = 3
    min_mask_pixels: int = 32
    min_observation_points: int = 16
    min_track_score: float = 0.50
    min_geometry_confidence: float = 0.30
    long_term_min_track_score: float = 0.75
    long_term_min_geometry_confidence: float = 0.50
    min_area_ratio: float = 0.35
    max_area_ratio: float = 2.85
    min_reprojection_consistency: float = 0.20
    min_reliability: float = 0.20
    reprojection_scale_m: float = 0.10
    soft_weight_floor: float = 0.25

    def validate(self) -> "MapWriteGateConfig":
        for name, value in (
            ("short_term_capacity", self.short_term_capacity),
            ("long_term_capacity", self.long_term_capacity),
            ("min_observations_before_write", self.min_observations_before_write),
            ("reentry_confirmation_frames", self.reentry_confirmation_frames),
            ("min_mask_pixels", self.min_mask_pixels),
            ("min_observation_points", self.min_observation_points),
        ):
            if int(value) < 1:
                raise ValueError(f"map_write_gate.{name} must be positive.")
        if int(self.reentry_gap) < 1:
            raise ValueError("map_write_gate.reentry_gap must be positive.")
        if float(self.reprojection_scale_m) <= 0.0:
            raise ValueError("map_write_gate.reprojection_scale_m must be positive.")
        if not 0.0 < float(self.min_area_ratio) <= 1.0:
            raise ValueError("map_write_gate.min_area_ratio must be in (0,1].")
        if float(self.max_area_ratio) < 1.0:
            raise ValueError("map_write_gate.max_area_ratio must be at least one.")
        for name, value in (
            ("min_track_score", self.min_track_score),
            ("min_geometry_confidence", self.min_geometry_confidence),
            ("long_term_min_track_score", self.long_term_min_track_score),
            (
                "long_term_min_geometry_confidence",
                self.long_term_min_geometry_confidence,
            ),
            ("min_reprojection_consistency", self.min_reprojection_consistency),
            ("min_reliability", self.min_reliability),
            ("soft_weight_floor", self.soft_weight_floor),
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"map_write_gate.{name} must be in [0,1].")
        return self

    def to_dict(self) -> dict[str, object]:
        return {
            "enabled": bool(self.enabled),
            "short_term_capacity": int(self.short_term_capacity),
            "long_term_capacity": int(self.long_term_capacity),
            "min_observations_before_write": int(self.min_observations_before_write),
            "reentry_confirmation_frames": int(self.reentry_confirmation_frames),
            "reentry_gap": int(self.reentry_gap),
            "min_mask_pixels": int(self.min_mask_pixels),
            "min_observation_points": int(self.min_observation_points),
            "min_track_score": float(self.min_track_score),
            "min_geometry_confidence": float(self.min_geometry_confidence),
            "long_term_min_track_score": float(self.long_term_min_track_score),
            "long_term_min_geometry_confidence": float(
                self.long_term_min_geometry_confidence
            ),
            "min_area_ratio": float(self.min_area_ratio),
            "max_area_ratio": float(self.max_area_ratio),
            "min_reprojection_consistency": float(
                self.min_reprojection_consistency
            ),
            "min_reliability": float(self.min_reliability),
            "reprojection_scale_m": float(self.reprojection_scale_m),
            "soft_weight_floor": float(self.soft_weight_floor),
        }


@dataclass(frozen=True)
class MapUpdateStats:
    frame_id: int
    observation_count: int
    accepted_observation_count: int
    static_observation_count: int
    dynamic_observation_count: int
    accepted_point_count: int
    static_voxel_count: int
    scene_voxel_count: int
    map_write_observation_count: int = 0
    map_gate_rejected_count: int = 0
    map_write_weight_sum: float = 0.0
    object_memory_count: int = 0


@dataclass(frozen=True)
class ObjectTrackMap:
    """Fused point cloud and provenance for one persistent object identity."""

    instance_id: int
    category: str
    points: torch.Tensor
    weights: torch.Tensor
    frame_indices: torch.Tensor
    observations: int
    is_static: bool | None
    map_write_observations: int = 0
    map_write_points: int = 0


@dataclass(frozen=True)
class SemanticMapResult:
    """Serializable result of the backend-neutral map builder."""

    scene_voxel_points: torch.Tensor
    scene_voxel_rgb: torch.Tensor
    scene_semantic_labels: tuple[str, ...]
    scene_instance_ids: torch.Tensor
    scene_evidence_weights: torch.Tensor
    scene_observation_counts: torch.Tensor
    voxel_points: torch.Tensor
    voxel_rgb: torch.Tensor
    semantic_labels: tuple[str, ...]
    instance_ids: torch.Tensor
    evidence_weights: torch.Tensor
    observation_counts: torch.Tensor
    object_tracks: tuple[ObjectTrackMap, ...]
    metadata: Mapping[str, Any]

    @property
    def voxel_count(self) -> int:
        return int(self.voxel_points.shape[0])

    @property
    def scene_voxel_count(self) -> int:
        return int(self.scene_voxel_points.shape[0])

    @property
    def scene_labeled_voxel_count(self) -> int:
        return int(self.scene_instance_ids.ge(0).sum())

    @property
    def labeled_voxel_count(self) -> int:
        return int(self.instance_ids.ge(0).sum())


@dataclass
class _VoxelEvidence:
    point_sum: torch.Tensor
    rgb_sum: torch.Tensor
    rgb_weight: float = 0.0
    weight: float = 0.0
    observations: int = 0
    last_frame_id: int = -1
    label_weights: dict[tuple[str, int], float] = field(default_factory=dict)


@dataclass
class _TrackEvidence:
    instance_id: int
    label_weights: dict[str, float] = field(default_factory=dict)
    point_chunks: list[torch.Tensor] = field(default_factory=list)
    weight_chunks: list[torch.Tensor] = field(default_factory=list)
    frame_chunks: list[torch.Tensor] = field(default_factory=list)
    observations: int = 0
    static_observations: int = 0
    dynamic_observations: int = 0
    map_write_observations: int = 0
    map_write_points: int = 0

    def add(
        self,
        *,
        label: str,
        points: torch.Tensor,
        weights: torch.Tensor,
        frame_id: int,
        is_static: bool,
        max_points: int,
        map_written: bool = False,
    ) -> None:
        self.label_weights[label] = self.label_weights.get(label, 0.0) + float(
            weights.sum()
        )
        self.point_chunks.append(points.detach().float().cpu())
        self.weight_chunks.append(weights.detach().float().cpu())
        self.frame_chunks.append(
            torch.full(
                (points.shape[0],),
                int(frame_id),
                dtype=torch.long,
            )
        )
        self.observations += 1
        if is_static:
            self.static_observations += 1
        else:
            self.dynamic_observations += 1
        if map_written:
            self.map_write_observations += 1
            self.map_write_points += int(points.shape[0])
        self._compact(max_points)

    def _compact(self, max_points: int) -> None:
        points = torch.cat(self.point_chunks, dim=0)
        weights = torch.cat(self.weight_chunks, dim=0)
        frames = torch.cat(self.frame_chunks, dim=0)
        if points.shape[0] > int(max_points):
            order = torch.argsort(weights, descending=True, stable=True)
            keep = order[: int(max_points)]
            points = points.index_select(0, keep)
            weights = weights.index_select(0, keep)
            frames = frames.index_select(0, keep)
        self.point_chunks = [points]
        self.weight_chunks = [weights]
        self.frame_chunks = [frames]

    def finalize(self) -> ObjectTrackMap:
        points = torch.cat(self.point_chunks, dim=0)
        weights = torch.cat(self.weight_chunks, dim=0)
        frames = torch.cat(self.frame_chunks, dim=0)
        label = sorted(
            self.label_weights.items(),
            key=lambda item: (-item[1], item[0]),
        )[0][0]
        if self.static_observations and not self.dynamic_observations:
            is_static: bool | None = True
        elif self.dynamic_observations and not self.static_observations:
            is_static = False
        else:
            is_static = None
        return ObjectTrackMap(
            instance_id=int(self.instance_id),
            category=label,
            points=points,
            weights=weights,
            frame_indices=frames,
            observations=int(self.observations),
            is_static=is_static,
            map_write_observations=int(self.map_write_observations),
            map_write_points=int(self.map_write_points),
        )


@dataclass(frozen=True)
class _MemoryObservation:
    frame_id: int
    area: int
    center: torch.Tensor
    extent: torch.Tensor
    track_score: float
    geometry_confidence: float
    point_count: int
    quality: float


@dataclass(frozen=True)
class _MapWriteDecision:
    map_write: bool
    weight: float
    reason: str
    state: str
    visibility_gap: int
    area_ratio: float
    area_stability: float
    reprojection_consistency: float
    reliability: float
    short_term_size: int
    long_term_size: int

    def to_dict(self) -> dict[str, object]:
        return {
            "map_write": int(self.map_write),
            "weight": float(self.weight),
            "reason": str(self.reason),
            "state": str(self.state),
            "visibility_gap": int(self.visibility_gap),
            "area_ratio": float(self.area_ratio),
            "area_stability": float(self.area_stability),
            "reprojection_consistency": float(self.reprojection_consistency),
            "reliability": float(self.reliability),
            "short_term_size": int(self.short_term_size),
            "long_term_size": int(self.long_term_size),
        }


@dataclass
class _ObjectMemoryState:
    instance_id: int
    category: str
    recent: list[_MemoryObservation] = field(default_factory=list)
    anchors: list[_MemoryObservation] = field(default_factory=list)
    observation_count: int = 0
    map_write_count: int = 0
    last_seen_frame: int = -1
    reentry_visible_count: int = 0

    def evaluate(
        self,
        *,
        frame_id: int,
        area: int,
        center: torch.Tensor,
        extent: torch.Tensor,
        track_score: float,
        geometry_confidence: float,
        point_count: int,
        config: MapWriteGateConfig,
    ) -> _MapWriteDecision:
        if not bool(config.enabled):
            return _MapWriteDecision(
                map_write=True,
                weight=1.0,
                reason="accept:gate_disabled",
                state="baseline",
                visibility_gap=(
                    -1
                    if self.last_seen_frame < 0
                    else int(frame_id) - self.last_seen_frame - 1
                ),
                area_ratio=1.0,
                area_stability=1.0,
                reprojection_consistency=1.0,
                reliability=1.0,
                short_term_size=len(self.recent),
                long_term_size=len(self.anchors),
            )

        visibility_gap = (
            -1
            if self.last_seen_frame < 0
            else int(frame_id) - self.last_seen_frame - 1
        )
        if self.last_seen_frame < 0:
            state = "new"
        elif visibility_gap >= int(config.reentry_gap):
            state = "reentry"
        elif self.reentry_visible_count > 0 and self.reentry_visible_count < int(
            config.reentry_confirmation_frames
        ):
            state = "reentry_confirmation"
        elif visibility_gap > 0:
            state = "occluded_then_visible"
        else:
            state = "visible"

        history = self.recent or self.anchors
        historical_area = _median([entry.area for entry in history])
        area_ratio = (
            float(area) / historical_area if historical_area > 0.0 else 1.0
        )
        area_stability = (
            math.exp(-abs(math.log(max(area_ratio, 1e-6))))
            if area > 0
            else 0.0
        )
        anchor_center, anchor_extent = self._anchor_geometry()
        if anchor_center is None:
            reprojection_consistency = 1.0
        else:
            scale = max(
                float(config.reprojection_scale_m),
                float(torch.linalg.vector_norm(anchor_extent)) * 0.5,
                float(torch.linalg.vector_norm(extent)) * 0.5,
                1e-6,
            )
            distance = float(torch.linalg.vector_norm(center - anchor_center))
            reprojection_consistency = math.exp(-distance / scale)
        reliability = (
            float(track_score)
            * float(geometry_confidence)
            * float(area_stability)
            * float(reprojection_consistency)
        )

        checks = (
            (area > 0, "reject:empty_mask"),
            (area >= int(config.min_mask_pixels), "reject:too_few_mask_pixels"),
            (
                point_count >= int(config.min_observation_points),
                "reject:too_few_geometry_points",
            ),
            (
                track_score >= float(config.min_track_score),
                "reject:low_track_score",
            ),
            (
                geometry_confidence >= float(config.min_geometry_confidence),
                "reject:low_geometry_confidence",
            ),
        )
        for passed, reason in checks:
            if not passed:
                return self._rejected(
                    reason,
                    state=state,
                    visibility_gap=visibility_gap,
                    area_ratio=area_ratio,
                    area_stability=area_stability,
                    reprojection_consistency=reprojection_consistency,
                    reliability=reliability,
                    config=config,
                )
        if history and (
            area_ratio < float(config.min_area_ratio)
            or area_ratio > float(config.max_area_ratio)
        ):
            return self._rejected(
                "reject:area_inconsistent_with_memory",
                state=state,
                visibility_gap=visibility_gap,
                area_ratio=area_ratio,
                area_stability=area_stability,
                reprojection_consistency=reprojection_consistency,
                reliability=reliability,
                config=config,
            )
        if history and reprojection_consistency < float(
            config.min_reprojection_consistency
        ):
            return self._rejected(
                "reject:reprojection_inconsistent",
                state=state,
                visibility_gap=visibility_gap,
                area_ratio=area_ratio,
                area_stability=area_stability,
                reprojection_consistency=reprojection_consistency,
                reliability=reliability,
                config=config,
            )
        current_observation_count = int(self.observation_count) + 1
        if state == "new" and current_observation_count < int(
            config.min_observations_before_write
        ):
            return self._rejected(
                "defer:confirmation",
                state=state,
                visibility_gap=visibility_gap,
                area_ratio=area_ratio,
                area_stability=area_stability,
                reprojection_consistency=reprojection_consistency,
                reliability=reliability,
                config=config,
            )
        if state in {"reentry", "reentry_confirmation"} and (
            self.reentry_visible_count + 1
            < int(config.reentry_confirmation_frames)
        ):
            return self._rejected(
                "defer:reentry_confirmation",
                state=state,
                visibility_gap=visibility_gap,
                area_ratio=area_ratio,
                area_stability=area_stability,
                reprojection_consistency=reprojection_consistency,
                reliability=reliability,
                config=config,
            )
        if reliability < float(config.min_reliability):
            return self._rejected(
                "reject:low_combined_reliability",
                state=state,
                visibility_gap=visibility_gap,
                area_ratio=area_ratio,
                area_stability=area_stability,
                reprojection_consistency=reprojection_consistency,
                reliability=reliability,
                config=config,
            )
        return _MapWriteDecision(
            map_write=True,
            weight=max(float(config.soft_weight_floor), min(1.0, reliability)),
            reason="accept:memory_consistent",
            state=state,
            visibility_gap=visibility_gap,
            area_ratio=area_ratio,
            area_stability=area_stability,
            reprojection_consistency=reprojection_consistency,
            reliability=reliability,
            short_term_size=len(self.recent),
            long_term_size=len(self.anchors),
        )

    def observe(
        self,
        *,
        frame_id: int,
        category: str,
        area: int,
        center: torch.Tensor,
        extent: torch.Tensor,
        track_score: float,
        geometry_confidence: float,
        point_count: int,
        decision: _MapWriteDecision,
        semantic_map_write: bool,
        config: MapWriteGateConfig,
    ) -> None:
        frame_id = int(frame_id)
        gap = (
            -1
            if self.last_seen_frame < 0
            else frame_id - self.last_seen_frame - 1
        )
        if gap >= int(config.reentry_gap):
            self.reentry_visible_count = 1
        elif self.reentry_visible_count > 0:
            self.reentry_visible_count = min(
                int(config.reentry_confirmation_frames),
                self.reentry_visible_count + 1,
            )
        quality = float(track_score) * float(geometry_confidence)
        entry = _MemoryObservation(
            frame_id=frame_id,
            area=int(area),
            center=center.detach().float().cpu(),
            extent=extent.detach().float().cpu(),
            track_score=float(track_score),
            geometry_confidence=float(geometry_confidence),
            point_count=int(point_count),
            quality=quality,
        )
        if area > 0:
            self.recent.append(entry)
            self.recent = self.recent[-int(config.short_term_capacity) :]
            if (
                track_score >= float(config.long_term_min_track_score)
                and geometry_confidence
                >= float(config.long_term_min_geometry_confidence)
                and decision.reprojection_consistency
                >= float(config.min_reprojection_consistency)
            ):
                self.anchors.append(entry)
                self.anchors.sort(
                    key=lambda value: (-float(value.quality), int(value.frame_id))
                )
                self.anchors = self.anchors[: int(config.long_term_capacity)]
            self.last_seen_frame = frame_id
        self.category = str(category)
        self.observation_count += 1
        # ``decision.map_write`` describes the generic gate decision.  A
        # dynamic observation may pass that gate, but it is still excluded
        # from the semantic voxel map.  Count only the actual semantic write
        # so object-memory diagnostics cannot over-report dynamic writes.
        if semantic_map_write:
            self.map_write_count += 1

    def summary(self) -> dict[str, object]:
        return {
            "instance_id": int(self.instance_id),
            "category": str(self.category),
            "observation_count": int(self.observation_count),
            "map_write_count": int(self.map_write_count),
            "last_seen_frame": int(self.last_seen_frame),
            "short_term_size": int(len(self.recent)),
            "long_term_size": int(len(self.anchors)),
        }

    def _anchor_geometry(self) -> tuple[torch.Tensor | None, torch.Tensor]:
        values = self.anchors or self.recent
        if not values:
            return None, torch.zeros(3)
        weights = torch.tensor(
            [max(float(entry.quality), 1e-6) for entry in values],
            dtype=torch.float32,
        )
        centers = torch.stack([entry.center for entry in values])
        extents = torch.stack([entry.extent for entry in values])
        center = (centers * weights[:, None]).sum(dim=0) / weights.sum()
        extent = extents.max(dim=0).values
        return center, extent

    def _rejected(
        self,
        reason: str,
        *,
        state: str,
        visibility_gap: int,
        area_ratio: float,
        area_stability: float,
        reprojection_consistency: float,
        reliability: float,
        config: MapWriteGateConfig,
    ) -> _MapWriteDecision:
        return _MapWriteDecision(
            map_write=False,
            weight=0.0,
            reason=str(reason),
            state=str(state),
            visibility_gap=int(visibility_gap),
            area_ratio=float(area_ratio),
            area_stability=float(area_stability),
            reprojection_consistency=float(reprojection_consistency),
            reliability=float(reliability),
            short_term_size=len(self.recent),
            long_term_size=len(self.anchors),
        )


class SemanticMapBuilder:
    """Fuse canonical frame observations into a causal semantic map.

    The builder is deliberately unaware of SAM or StreamVGGT classes.  It
    consumes one geometry frame and one segmentation frame at a time, which
    also makes it usable with a native streaming geometry backend.
    """

    def __init__(self, config: SemanticMapConfig | None = None) -> None:
        self.config = (config or SemanticMapConfig()).validate()
        self._scene_voxels: dict[tuple[int, int, int], _VoxelEvidence] = {}
        self._voxels: dict[tuple[int, int, int], _VoxelEvidence] = {}
        self._tracks: dict[int, _TrackEvidence] = {}
        self._object_memory: dict[int, _ObjectMemoryState] = {}
        self._object_memory_events: list[dict[str, object]] = []
        self._last_frame_id: int | None = None
        self._frame_count = 0
        self._last_stats: MapUpdateStats | None = None

    @property
    def last_stats(self) -> MapUpdateStats | None:
        return self._last_stats

    def update(
        self,
        geometry: GeometryFrame,
        segmentation: SegmentationFrame,
    ) -> MapUpdateStats:
        """Consume one frame without consulting any future observation."""

        if int(geometry.frame_id) != int(segmentation.frame_id):
            raise ValueError(
                "Geometry and segmentation frame IDs differ: "
                f"{geometry.frame_id} vs {segmentation.frame_id}."
            )
        frame_id = int(geometry.frame_id)
        if self._last_frame_id is not None and frame_id <= self._last_frame_id:
            raise ValueError(
                "SemanticMapBuilder requires strictly increasing frame IDs; "
                f"received {frame_id} after {self._last_frame_id}."
            )
        geometry = geometry.cpu()
        segmentation = segmentation.cpu()
        points, point_valid = world_points_for_frame(geometry)
        confidence = geometry_confidence_for_frame(geometry)
        valid = point_valid & (confidence >= self.config.min_geometry_confidence)
        rgb = rgb_for_frame(geometry)
        if bool(valid.any()):
            self._add_scene_voxels(
                points=points[valid],
                weights=confidence[valid],
                rgb=rgb[valid] if rgb is not None else None,
                frame_id=frame_id,
            )

        accepted_observations = 0
        static_observations = 0
        dynamic_observations = 0
        accepted_points = 0
        map_write_observations = 0
        map_gate_rejected = 0
        map_write_weight_sum = 0.0
        for observation in segmentation.observations:
            if float(observation.score) < self.config.min_track_score:
                continue
            mask = resize_bool_mask(observation.mask, geometry.image_size)
            selected = valid & mask
            if not bool(selected.any()):
                continue
            selected_points = points[selected]
            selected_weights = confidence[selected] * float(observation.score)
            selected_rgb = rgb[selected] if rgb is not None else None
            (
                selected_points,
                selected_weights,
                selected_rgb,
            ) = _keep_highest_confidence(
                selected_points,
                selected_weights,
                selected_rgb,
                self.config.max_points_per_observation,
            )
            if not selected_points.numel():
                continue
            accepted_observations += 1
            accepted_points += int(selected_points.shape[0])
            is_static = _is_static_observation(
                observation,
                threshold=self.config.static_score_threshold,
                require_score=self.config.require_static_score,
            )
            memory_state = self._object_memory.setdefault(
                int(observation.instance_id),
                _ObjectMemoryState(
                    instance_id=int(observation.instance_id),
                    category=str(observation.category),
                ),
            )
            area = int(mask.sum())
            point_count = int(selected_points.shape[0])
            geometry_score = float(confidence[selected].mean())
            center = selected_points.mean(dim=0)
            extent = selected_points.max(dim=0).values - selected_points.min(dim=0).values
            gate_decision = memory_state.evaluate(
                frame_id=frame_id,
                area=area,
                center=center,
                extent=extent,
                track_score=float(observation.score),
                geometry_confidence=geometry_score,
                point_count=point_count,
                config=self.config.map_write_gate,
            )
            semantic_map_write = bool(is_static and gate_decision.map_write)
            memory_state.observe(
                frame_id=frame_id,
                category=str(observation.category),
                area=area,
                center=center,
                extent=extent,
                track_score=float(observation.score),
                geometry_confidence=geometry_score,
                point_count=point_count,
                decision=gate_decision,
                semantic_map_write=semantic_map_write,
                config=self.config.map_write_gate,
            )
            memory_event = {
                "frame_id": int(frame_id),
                "instance_id": int(observation.instance_id),
                "category": str(observation.category),
                "mask_pixels": int(area),
                "geometry_points": int(point_count),
                "track_score": float(observation.score),
                "geometry_confidence": float(geometry_score),
                "static_observation": int(is_static),
                **gate_decision.to_dict(),
                "semantic_map_write": int(semantic_map_write),
            }
            self._object_memory_events.append(memory_event)
            if is_static:
                static_observations += 1
                if semantic_map_write:
                    map_write_observations += 1
                    map_write_weight_sum += float(gate_decision.weight)
                    self._add_voxels(
                        label=str(observation.category),
                        instance_id=int(observation.instance_id),
                        points=selected_points,
                        weights=selected_weights * float(gate_decision.weight),
                        rgb=selected_rgb,
                        frame_id=frame_id,
                    )
                else:
                    map_gate_rejected += 1
            else:
                dynamic_observations += 1
            if is_static or self.config.include_dynamic_tracks:
                track = self._tracks.setdefault(
                    int(observation.instance_id),
                    _TrackEvidence(instance_id=int(observation.instance_id)),
                )
                track.add(
                    label=str(observation.category),
                    points=selected_points,
                    weights=selected_weights,
                    frame_id=frame_id,
                    is_static=is_static,
                    max_points=self.config.max_points_per_track,
                    map_written=semantic_map_write,
                )

        self._last_frame_id = frame_id
        self._frame_count += 1
        stats = MapUpdateStats(
            frame_id=frame_id,
            observation_count=len(segmentation.observations),
            accepted_observation_count=accepted_observations,
            static_observation_count=static_observations,
            dynamic_observation_count=dynamic_observations,
            accepted_point_count=accepted_points,
            static_voxel_count=len(self._voxels),
            scene_voxel_count=len(self._scene_voxels),
            map_write_observation_count=map_write_observations,
            map_gate_rejected_count=map_gate_rejected,
            map_write_weight_sum=float(map_write_weight_sum),
            object_memory_count=len(self._object_memory),
        )
        self._last_stats = stats
        return stats

    def finalize(self, metadata: Mapping[str, Any] | None = None) -> SemanticMapResult:
        """Materialize deterministic tensors and object-level tracks."""

        keys = _select_voxel_keys(self._voxels, self.config.max_voxels)

        points: list[torch.Tensor] = []
        colors: list[torch.Tensor] = []
        instance_ids: list[int] = []
        labels: list[str] = []
        evidence_weights: list[float] = []
        observation_counts: list[int] = []
        for key in keys:
            evidence = self._voxels[key]
            weight = max(float(evidence.weight), 1e-12)
            points.append(evidence.point_sum / weight)
            if evidence.rgb_weight > 0.0:
                colors.append(evidence.rgb_sum / evidence.rgb_weight)
            else:
                colors.append(torch.zeros(3, dtype=torch.float32))
            label, instance_id = _dominant_label(evidence)
            labels.append(label)
            instance_ids.append(int(instance_id))
            evidence_weights.append(float(evidence.weight))
            observation_counts.append(int(evidence.observations))

        scene_keys = _select_voxel_keys(
            self._scene_voxels,
            self.config.max_scene_voxels,
            preferred_keys=self._voxels,
        )
        scene_points: list[torch.Tensor] = []
        scene_colors: list[torch.Tensor] = []
        scene_labels: list[str] = []
        scene_instance_ids: list[int] = []
        scene_evidence_weights: list[float] = []
        scene_observation_counts: list[int] = []
        for key in scene_keys:
            evidence = self._scene_voxels[key]
            weight = max(float(evidence.weight), 1e-12)
            scene_points.append(evidence.point_sum / weight)
            if evidence.rgb_weight > 0.0:
                scene_colors.append(evidence.rgb_sum / evidence.rgb_weight)
            else:
                scene_colors.append(torch.zeros(3, dtype=torch.float32))
            semantic_evidence = self._voxels.get(key)
            if semantic_evidence is None:
                scene_labels.append("")
                scene_instance_ids.append(-1)
            else:
                label, instance_id = _dominant_label(semantic_evidence)
                scene_labels.append(label)
                scene_instance_ids.append(int(instance_id))
            scene_evidence_weights.append(float(evidence.weight))
            scene_observation_counts.append(int(evidence.observations))

        track_maps = tuple(
            self._tracks[instance_id].finalize()
            for instance_id in sorted(self._tracks)
        )
        result_metadata = dict(metadata or {})
        result_metadata.setdefault("schema", 1)
        result_metadata.setdefault("voxel_size_m", float(self.config.voxel_size_m))
        result_metadata.setdefault("frame_count", int(self._frame_count))
        result_metadata.setdefault("last_frame_id", self._last_frame_id)
        result_metadata.setdefault("coordinate_frame", "world")
        result_metadata.setdefault("map_pose_feedback", False)
        result_metadata.setdefault("pointmap_modified", False)
        result_metadata.setdefault(
            "map_write_gate",
            {
                **self.config.map_write_gate.to_dict(),
                "event_count": int(len(self._object_memory_events)),
                "static_observation_count": int(
                    sum(
                        int(bool(event.get("static_observation", 0)))
                        for event in self._object_memory_events
                    )
                ),
                "dynamic_observation_count": int(
                    sum(
                        int(not bool(event.get("static_observation", 0)))
                        for event in self._object_memory_events
                    )
                ),
                "write_count": int(
                    sum(
                        int(bool(event.get("semantic_map_write", 0)))
                        for event in self._object_memory_events
                    )
                ),
                "rejected_count": int(
                    sum(
                        int(
                            bool(event.get("static_observation", 0))
                            and not bool(event.get("semantic_map_write", 0))
                        )
                        for event in self._object_memory_events
                    )
                ),
            },
        )
        result_metadata.setdefault(
            "object_memory",
            {
                "instance_count": int(len(self._object_memory)),
                "states": [
                    self._object_memory[key].summary()
                    for key in sorted(self._object_memory)
                ],
                "events": list(self._object_memory_events),
            },
        )
        return SemanticMapResult(
            scene_voxel_points=_stack_or_empty(scene_points, (0, 3)),
            scene_voxel_rgb=_stack_or_empty(scene_colors, (0, 3)).clamp(0.0, 1.0),
            scene_semantic_labels=tuple(scene_labels),
            scene_instance_ids=torch.tensor(scene_instance_ids, dtype=torch.long),
            scene_evidence_weights=torch.tensor(
                scene_evidence_weights,
                dtype=torch.float32,
            ),
            scene_observation_counts=torch.tensor(
                scene_observation_counts,
                dtype=torch.long,
            ),
            voxel_points=_stack_or_empty(points, (0, 3)),
            voxel_rgb=_stack_or_empty(colors, (0, 3)).clamp(0.0, 1.0),
            semantic_labels=tuple(labels),
            instance_ids=torch.tensor(instance_ids, dtype=torch.long),
            evidence_weights=torch.tensor(evidence_weights, dtype=torch.float32),
            observation_counts=torch.tensor(observation_counts, dtype=torch.long),
            object_tracks=track_maps,
            metadata=result_metadata,
        )

    def _add_scene_voxels(
        self,
        *,
        points: torch.Tensor,
        weights: torch.Tensor,
        rgb: torch.Tensor | None,
        frame_id: int,
    ) -> None:
        self._accumulate_voxels(
            self._scene_voxels,
            points=points,
            weights=weights,
            rgb=rgb,
            frame_id=frame_id,
        )

    def _add_voxels(
        self,
        *,
        label: str,
        instance_id: int,
        points: torch.Tensor,
        weights: torch.Tensor,
        rgb: torch.Tensor | None,
        frame_id: int,
    ) -> None:
        self._accumulate_voxels(
            self._voxels,
            points=points,
            weights=weights,
            rgb=rgb,
            frame_id=frame_id,
            label=label,
            instance_id=instance_id,
        )

    def _accumulate_voxels(
        self,
        voxels: dict[tuple[int, int, int], _VoxelEvidence],
        *,
        points: torch.Tensor,
        weights: torch.Tensor,
        rgb: torch.Tensor | None,
        frame_id: int,
        label: str | None = None,
        instance_id: int | None = None,
    ) -> None:
        voxel_coords = torch.floor(points / float(self.config.voxel_size_m)).long()
        unique, inverse = torch.unique(
            voxel_coords,
            dim=0,
            sorted=True,
            return_inverse=True,
        )
        group_count = int(unique.shape[0])
        group_weights = torch.zeros(group_count, dtype=torch.float64)
        group_weights.index_add_(0, inverse, weights.double())
        group_points = torch.zeros(group_count, 3, dtype=torch.float64)
        group_points.index_add_(0, inverse, points.double() * weights[:, None].double())
        if rgb is not None:
            group_rgb = torch.zeros(group_count, 3, dtype=torch.float64)
            group_rgb.index_add_(0, inverse, rgb.double() * weights[:, None].double())
        else:
            group_rgb = None
        for index in range(group_count):
            key = tuple(int(value) for value in unique[index].tolist())
            weight = float(group_weights[index])
            evidence = voxels.get(key)
            if evidence is None:
                evidence = _VoxelEvidence(
                    point_sum=torch.zeros(3, dtype=torch.float64),
                    rgb_sum=torch.zeros(3, dtype=torch.float64),
                )
                voxels[key] = evidence
            evidence.point_sum += group_points[index]
            evidence.weight += weight
            evidence.observations += 1
            evidence.last_frame_id = int(frame_id)
            if label is not None and instance_id is not None:
                label_key = (str(label), int(instance_id))
                evidence.label_weights[label_key] = (
                    evidence.label_weights.get(label_key, 0.0) + weight
                )
            if group_rgb is not None:
                evidence.rgb_sum += group_rgb[index]
                evidence.rgb_weight += weight


def _select_voxel_keys(
    voxels: Mapping[tuple[int, int, int], _VoxelEvidence],
    limit: int,
    *,
    preferred_keys: Mapping[tuple[int, int, int], Any] | None = None,
) -> list[tuple[int, int, int]]:
    keys = sorted(voxels)
    if len(keys) <= int(limit):
        return keys
    preferred = set(preferred_keys or ())
    keys = sorted(
        keys,
        key=lambda key: (
            0 if key in preferred else 1,
            -voxels[key].weight,
            key,
        ),
    )[: int(limit)]
    keys.sort()
    return keys


def _dominant_label(evidence: _VoxelEvidence) -> tuple[str, int]:
    if not evidence.label_weights:
        raise ValueError("Semantic voxel has no label evidence.")
    return sorted(
        evidence.label_weights.items(),
        key=lambda item: (-item[1], item[0][0], item[0][1]),
    )[0][0]


def _is_static_observation(
    observation: ObjectObservation,
    *,
    threshold: float,
    require_score: bool,
) -> bool:
    if observation.static_score is None:
        return not require_score
    return float(observation.static_score) >= float(threshold)


def _median(values: list[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def _keep_highest_confidence(
    points: torch.Tensor,
    weights: torch.Tensor,
    rgb: torch.Tensor | None,
    limit: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if points.shape[0] <= int(limit):
        return points, weights, rgb
    order = torch.argsort(weights, descending=True, stable=True)[: int(limit)]
    points = points.index_select(0, order)
    weights = weights.index_select(0, order)
    if rgb is not None:
        rgb = rgb.index_select(0, order)
    return points, weights, rgb


def _stack_or_empty(values: list[torch.Tensor], empty_shape: tuple[int, ...]) -> torch.Tensor:
    if values:
        return torch.stack(values, dim=0).float()
    return torch.empty(empty_shape, dtype=torch.float32)
