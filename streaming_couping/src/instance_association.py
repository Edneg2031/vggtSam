"""Geometry-first association of short-lived SAM observations.

The SAM video predictor is still treated as a causal short-term tracker.  This
module only decides whether a newly observed SAM track belongs to an existing
long-term object.  It deliberately has no dependency on SAM or StreamVGGT so
that the same association code can be used by the deployable exporter and by
the offline cache evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class InstanceAssociationConfig:
    """Weights and gates for one observation-to-object association."""

    center_weight: float = 0.40
    voxel_weight: float = 0.25
    chamfer_weight: float = 0.20
    appearance_weight: float = 0.10
    category_weight: float = 0.05
    min_match_score: float = 0.50
    max_center_distance_ratio: float = 0.35
    center_scale_ratio: float = 0.12
    chamfer_scale_ratio: float = 0.12
    voxel_size_ratio: float = 0.02
    absolute_voxel_size: float = 0.02
    max_points_per_comparison: int = 256
    category_hard_gate: bool = True
    generic_categories: tuple[str, ...] = ("object", "thing", "item")


@dataclass(frozen=True)
class InstanceObservation:
    """One valid SAM mask projected onto the current world pointmap."""

    sequence_index: int
    frame_index: int
    source_slot: int
    sam_track_id: int
    category: str
    points: torch.Tensor
    center: torch.Tensor
    covariance: torch.Tensor
    extent: torch.Tensor
    confidence: float
    track_score: float
    appearance: torch.Tensor | None = None

    @property
    def quality(self) -> float:
        return float(self.confidence) * float(self.track_score)


@dataclass(frozen=True)
class AssociationCandidate:
    """Auditable score decomposition for one existing object candidate."""

    object_id: int
    score: float
    accepted: bool
    center_distance: float
    center_similarity: float
    voxel_iou: float
    chamfer_distance: float
    chamfer_similarity: float
    appearance_cosine: float
    category_consistency: float
    compared_features: tuple[str, ...]
    rejection_reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "object_id": int(self.object_id),
            "score": float(self.score),
            "accepted": int(self.accepted),
            "center_distance": float(self.center_distance),
            "center_similarity": float(self.center_similarity),
            "voxel_iou": float(self.voxel_iou),
            "chamfer_distance": float(self.chamfer_distance),
            "chamfer_similarity": float(self.chamfer_similarity),
            "appearance_cosine": float(self.appearance_cosine),
            "category_consistency": float(self.category_consistency),
            "compared_features": ",".join(self.compared_features),
            "rejection_reason": str(self.rejection_reason),
        }


class InstanceAssociator:
    """Rank existing persistent objects for a new observation.

    Distances are normalized by ``scene_scale``.  The scale is computed from
    the current clip's frozen pointmap, so the defaults work for both metric
    pointmaps and StreamVGGT's arbitrary scene scale.
    """

    def __init__(self, config: InstanceAssociationConfig | None = None) -> None:
        self.config = config or InstanceAssociationConfig()
        _validate_config(self.config)

    def rank_candidates(
        self,
        observation: InstanceObservation,
        objects: Iterable[object],
        *,
        scene_scale: float = 1.0,
        excluded_object_ids: Sequence[int] = (),
    ) -> list[AssociationCandidate]:
        scale = max(float(scene_scale), 1e-6)
        excluded = {int(value) for value in excluded_object_ids}
        ranked: list[AssociationCandidate] = []
        for obj in objects:
            object_id = int(getattr(obj, "object_id"))
            if object_id in excluded:
                continue
            candidate = self.score_candidate(
                observation,
                obj,
                scene_scale=scale,
            )
            ranked.append(candidate)
        ranked.sort(
            key=lambda row: (
                -float(row.score),
                float(row.center_distance),
                int(row.object_id),
            )
        )
        return ranked

    def score_candidate(
        self,
        observation: InstanceObservation,
        obj: object,
        *,
        scene_scale: float = 1.0,
    ) -> AssociationCandidate:
        config = self.config
        scale = max(float(scene_scale), 1e-6)
        object_id = int(getattr(obj, "object_id"))
        object_category = str(getattr(obj, "category", "object"))
        category = category_consistency(
            observation.category,
            object_category,
            generic_categories=config.generic_categories,
        )
        category_mismatch = category <= 0.0
        center = _as_vector(getattr(obj, "center"), 3)
        distance = float(torch.linalg.vector_norm(observation.center - center))
        max_distance = max(
            scale * float(config.max_center_distance_ratio),
            float(config.absolute_voxel_size),
        )
        center_scale = max(scale * float(config.center_scale_ratio), 1e-6)
        center_similarity = math.exp(-distance / center_scale)
        object_points = _object_points(obj)
        object_voxels = _object_voxel_keys(obj)
        observation_voxels = voxel_keys(
            observation.points,
            voxel_size=_effective_voxel_size(scale, config),
        )
        overlap = voxel_iou(observation_voxels, object_voxels)

        chamfer = float("nan")
        chamfer_similarity = 0.0
        features = ["center", "category"]
        if object_points.numel() and observation.points.numel():
            chamfer = symmetric_chamfer_distance(
                observation.points,
                object_points,
                max_points=config.max_points_per_comparison,
            )
            chamfer_scale = max(scale * float(config.chamfer_scale_ratio), 1e-6)
            chamfer_similarity = math.exp(-chamfer / chamfer_scale)
            features.append("chamfer")
        if observation_voxels and object_voxels:
            features.append("voxel")

        appearance_cosine = float("nan")
        if observation.appearance is not None:
            object_appearance = getattr(obj, "appearance", None)
            if object_appearance is not None:
                appearance_cosine = cosine_similarity(
                    observation.appearance,
                    object_appearance,
                )
                if math.isfinite(appearance_cosine):
                    features.append("appearance")

        if category_mismatch and config.category_hard_gate:
            return AssociationCandidate(
                object_id=object_id,
                score=0.0,
                accepted=False,
                center_distance=distance,
                center_similarity=center_similarity,
                voxel_iou=overlap,
                chamfer_distance=chamfer,
                chamfer_similarity=chamfer_similarity,
                appearance_cosine=appearance_cosine,
                category_consistency=category,
                compared_features=tuple(features),
                rejection_reason="category_mismatch",
            )
        if distance > max_distance:
            return AssociationCandidate(
                object_id=object_id,
                score=0.0,
                accepted=False,
                center_distance=distance,
                center_similarity=center_similarity,
                voxel_iou=overlap,
                chamfer_distance=chamfer,
                chamfer_similarity=chamfer_similarity,
                appearance_cosine=appearance_cosine,
                category_consistency=category,
                compared_features=tuple(features),
                rejection_reason="center_too_far",
            )

        terms: list[tuple[float, float]] = [
            (float(config.center_weight), center_similarity),
            (float(config.category_weight), category),
        ]
        if "voxel" in features:
            terms.append((float(config.voxel_weight), overlap))
        if "chamfer" in features:
            terms.append((float(config.chamfer_weight), chamfer_similarity))
        if "appearance" in features:
            terms.append(
                (
                    float(config.appearance_weight),
                    max(0.0, min(1.0, 0.5 * (appearance_cosine + 1.0))),
                )
            )
        total_weight = sum(weight for weight, _ in terms)
        score = (
            sum(weight * value for weight, value in terms) / total_weight
            if total_weight > 0.0
            else 0.0
        )
        accepted = score >= float(config.min_match_score)
        return AssociationCandidate(
            object_id=object_id,
            score=float(score),
            accepted=bool(accepted),
            center_distance=distance,
            center_similarity=center_similarity,
            voxel_iou=overlap,
            chamfer_distance=chamfer,
            chamfer_similarity=chamfer_similarity,
            appearance_cosine=appearance_cosine,
            category_consistency=category,
            compared_features=tuple(features),
            rejection_reason="" if accepted else "score_below_threshold",
        )

    def match(
        self,
        observation: InstanceObservation,
        objects: Iterable[object],
        *,
        scene_scale: float = 1.0,
        excluded_object_ids: Sequence[int] = (),
    ) -> AssociationCandidate | None:
        candidates = self.rank_candidates(
            observation,
            objects,
            scene_scale=scene_scale,
            excluded_object_ids=excluded_object_ids,
        )
        if not candidates or not candidates[0].accepted:
            return None
        return candidates[0]


def category_consistency(
    left: str,
    right: str,
    *,
    generic_categories: Sequence[str] = ("object", "thing", "item"),
) -> float:
    """Return a prompt/category compatibility score in ``[0, 1]``."""

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


def normalize_category(value: str) -> str:
    return " ".join(str(value).strip().lower().replace("_", " ").split())


def cosine_similarity(left: torch.Tensor, right: torch.Tensor) -> float:
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


def voxel_keys(
    points: torch.Tensor,
    *,
    voxel_size: float,
) -> set[tuple[int, int, int]]:
    points = points.detach().float().cpu().reshape(-1, 3)
    points = points[torch.isfinite(points).all(dim=1)]
    if not points.numel():
        return set()
    if float(voxel_size) <= 0.0:
        raise ValueError("voxel_size must be positive.")
    quantized = torch.floor(points / float(voxel_size)).long()
    return {tuple(int(value) for value in row) for row in quantized.tolist()}


def voxel_iou(
    left: set[tuple[int, int, int]],
    right: set[tuple[int, int, int]],
) -> float:
    union = left | right
    return float(len(left & right) / len(union)) if union else 0.0


def symmetric_chamfer_distance(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    max_points: int = 256,
) -> float:
    """Compute a deterministic, bounded-memory symmetric Chamfer distance."""

    left = deterministic_points(left, limit=max_points)
    right = deterministic_points(right, limit=max_points)
    if not left.numel() or not right.numel():
        return float("nan")
    forward = _nearest_distances(left, right).mean()
    backward = _nearest_distances(right, left).mean()
    return float(0.5 * (forward + backward))


def deterministic_points(points: torch.Tensor, *, limit: int) -> torch.Tensor:
    values = points.detach().float().cpu().reshape(-1, 3)
    values = values[torch.isfinite(values).all(dim=1)]
    if values.shape[0] <= int(limit):
        return values
    indices = torch.linspace(
        0,
        values.shape[0] - 1,
        steps=int(limit),
    ).round().long()
    return values.index_select(0, indices)


def _nearest_distances(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    output = []
    for start in range(0, source.shape[0], 128):
        output.append(
            torch.cdist(source[start : start + 128], target).min(dim=1).values
        )
    return torch.cat(output)


def _object_points(obj: object) -> torch.Tensor:
    value = getattr(obj, "points", None)
    if value is None:
        value = getattr(obj, "point_cloud", None)
    if value is None:
        return torch.empty(0, 3)
    return deterministic_points(torch.as_tensor(value), limit=4096)


def _object_voxel_keys(obj: object) -> set[tuple[int, int, int]]:
    value = getattr(obj, "voxel_keys", None)
    if value is None:
        return set()
    return {
        tuple(int(component) for component in key)
        for key in value
        if len(tuple(key)) == 3
    }


def _effective_voxel_size(
    scene_scale: float,
    config: InstanceAssociationConfig,
) -> float:
    return max(
        float(config.absolute_voxel_size),
        float(scene_scale) * float(config.voxel_size_ratio),
    )


def _as_vector(value: object, dimension: int) -> torch.Tensor:
    tensor = torch.as_tensor(value).detach().float().cpu().reshape(-1)
    if tensor.shape != (dimension,):
        raise ValueError(f"Expected vector with shape [{dimension}].")
    return tensor


def _validate_config(config: InstanceAssociationConfig) -> None:
    weights = (
        config.center_weight,
        config.voxel_weight,
        config.chamfer_weight,
        config.appearance_weight,
        config.category_weight,
    )
    if any(float(value) < 0.0 for value in weights):
        raise ValueError("Association weights must be non-negative.")
    if sum(float(value) for value in weights) <= 0.0:
        raise ValueError("At least one association weight must be positive.")
    if not 0.0 <= float(config.min_match_score) <= 1.0:
        raise ValueError("min_match_score must be in [0,1].")
    for name in (
        "max_center_distance_ratio",
        "center_scale_ratio",
        "chamfer_scale_ratio",
        "voxel_size_ratio",
        "absolute_voxel_size",
    ):
        if float(getattr(config, name)) <= 0.0:
            raise ValueError(f"{name} must be positive.")
    if int(config.max_points_per_comparison) < 1:
        raise ValueError("max_points_per_comparison must be positive.")
