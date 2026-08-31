"""Causal semantic-instance point and voxel fusion."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
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
        return self


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

    def add(
        self,
        *,
        label: str,
        points: torch.Tensor,
        weights: torch.Tensor,
        frame_id: int,
        is_static: bool,
        max_points: int,
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
            if is_static:
                static_observations += 1
                self._add_voxels(
                    label=str(observation.category),
                    instance_id=int(observation.instance_id),
                    points=selected_points,
                    weights=selected_weights,
                    rgb=selected_rgb,
                    frame_id=frame_id,
                )
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
