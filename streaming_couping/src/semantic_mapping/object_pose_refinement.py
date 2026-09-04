"""SAM instance-guided, post-geometry pose refinement.

This module is intentionally independent of HorizonStream and SAM3.1 model
implementations.  It consumes frozen canonical geometry/segmentation frames,
uses persistent SAM instance IDs to select long-range frame pairs, estimates
object-relative SE(3) constraints from mask-constrained RGB patch matches, and
optionally refines only the camera poses in a small pose graph.

The module is training-free and does not mutate either model cache.  In
particular, an instance ID is used only to establish that two observations may
belong to the same object; it is never treated as a pixel correspondence.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
import json
import math
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .contracts import GeometryFrame, ObjectObservation, SegmentationFrame, SizeHW
from .geometry import geometry_confidence_for_frame, resize_bool_mask


@dataclass(frozen=True)
class ObjectPoseRefinementConfig:
    """Conservative, training-free settings for object pose refinement."""

    min_temporal_gap: int = 10
    min_track_score: float = 0.50
    min_mask_pixels: int = 32
    max_mask_area_ratio: float = 0.85
    min_geometry_points: int = 24
    min_geometry_confidence: float = 0.30
    max_pairs_per_instance: int = 8
    max_total_candidate_pairs: int = 256

    feature_backend: str = "rgb_patch"
    feature_patch_size: int = 8
    feature_max_points: int = 1024
    feature_cosine_threshold: float = 0.65
    feature_ratio_margin: float = 0.02
    max_feature_matches: int = 256

    ransac_iterations: int = 256
    ransac_inlier_threshold_m: float = 0.10
    min_matches: int = 12
    min_inliers: int = 6
    min_inlier_ratio: float = 0.30
    max_registration_rmse_m: float = 0.12
    ransac_seed: int = 2026

    max_rotation_disagreement_deg: float = 60.0
    max_translation_disagreement_m: float = 1.0
    reject_extreme_disagreement_multiplier: float = 2.0
    low_consistency_weight: float = 0.25

    sequential_edge_weight: float = 10.0
    object_edge_weight: float = 1.0
    huber_delta: float = 0.10
    rotation_residual_scale_m: float = 1.0
    max_pose_delta_rotation_deg: float = 45.0
    max_pose_delta_translation_m: float = 1.0
    optimizer_max_nfev: int = 100

    dinov3_checkpoint: Path | None = None
    dinov3_root: Path = Path(
        "/home/bod/86Nas/95_data_bak/FoundationModels/dinov3"
    )
    dinov3_variant: str = "dinov3-vitl16"
    dinov3_device: str = "cuda:0"
    dinov3_dtype: str = "bfloat16"

    def validate(self) -> "ObjectPoseRefinementConfig":
        for name, value in (
            ("min_temporal_gap", self.min_temporal_gap),
            ("min_mask_pixels", self.min_mask_pixels),
            ("min_geometry_points", self.min_geometry_points),
            ("max_pairs_per_instance", self.max_pairs_per_instance),
            ("max_total_candidate_pairs", self.max_total_candidate_pairs),
            ("feature_patch_size", self.feature_patch_size),
            ("feature_max_points", self.feature_max_points),
            ("max_feature_matches", self.max_feature_matches),
            ("ransac_iterations", self.ransac_iterations),
            ("min_matches", self.min_matches),
            ("min_inliers", self.min_inliers),
            ("optimizer_max_nfev", self.optimizer_max_nfev),
        ):
            if int(value) < 1:
                raise ValueError(f"object_pose.{name} must be positive.")
        for name, value in (
            ("min_track_score", self.min_track_score),
            ("min_geometry_confidence", self.min_geometry_confidence),
            ("feature_cosine_threshold", self.feature_cosine_threshold),
            ("feature_ratio_margin", self.feature_ratio_margin),
            ("min_inlier_ratio", self.min_inlier_ratio),
            ("low_consistency_weight", self.low_consistency_weight),
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"object_pose.{name} must be in [0,1].")
        for name, value in (
            ("max_mask_area_ratio", self.max_mask_area_ratio),
            ("reject_extreme_disagreement_multiplier", self.reject_extreme_disagreement_multiplier),
            ("sequential_edge_weight", self.sequential_edge_weight),
            ("object_edge_weight", self.object_edge_weight),
            ("huber_delta", self.huber_delta),
            ("rotation_residual_scale_m", self.rotation_residual_scale_m),
            ("max_pose_delta_rotation_deg", self.max_pose_delta_rotation_deg),
            ("max_pose_delta_translation_m", self.max_pose_delta_translation_m),
            ("ransac_inlier_threshold_m", self.ransac_inlier_threshold_m),
            ("max_registration_rmse_m", self.max_registration_rmse_m),
            ("max_rotation_disagreement_deg", self.max_rotation_disagreement_deg),
            ("max_translation_disagreement_m", self.max_translation_disagreement_m),
        ):
            if float(value) <= 0.0:
                raise ValueError(f"object_pose.{name} must be positive.")
        if not 0.0 < float(self.max_mask_area_ratio) <= 1.0:
            raise ValueError("object_pose.max_mask_area_ratio must be in (0,1].")
        if str(self.feature_backend).strip().lower() not in {"rgb_patch", "dinov3"}:
            raise ValueError(
                "object_pose.feature_backend must be 'rgb_patch' or 'dinov3'."
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_temporal_gap": int(self.min_temporal_gap),
            "min_track_score": float(self.min_track_score),
            "min_mask_pixels": int(self.min_mask_pixels),
            "max_mask_area_ratio": float(self.max_mask_area_ratio),
            "min_geometry_points": int(self.min_geometry_points),
            "min_geometry_confidence": float(self.min_geometry_confidence),
            "max_pairs_per_instance": int(self.max_pairs_per_instance),
            "max_total_candidate_pairs": int(self.max_total_candidate_pairs),
            "feature_backend": str(self.feature_backend),
            "feature_patch_size": int(self.feature_patch_size),
            "feature_max_points": int(self.feature_max_points),
            "feature_cosine_threshold": float(self.feature_cosine_threshold),
            "feature_ratio_margin": float(self.feature_ratio_margin),
            "max_feature_matches": int(self.max_feature_matches),
            "ransac_iterations": int(self.ransac_iterations),
            "ransac_inlier_threshold_m": float(self.ransac_inlier_threshold_m),
            "min_matches": int(self.min_matches),
            "min_inliers": int(self.min_inliers),
            "min_inlier_ratio": float(self.min_inlier_ratio),
            "max_registration_rmse_m": float(self.max_registration_rmse_m),
            "ransac_seed": int(self.ransac_seed),
            "max_rotation_disagreement_deg": float(self.max_rotation_disagreement_deg),
            "max_translation_disagreement_m": float(self.max_translation_disagreement_m),
            "reject_extreme_disagreement_multiplier": float(
                self.reject_extreme_disagreement_multiplier
            ),
            "low_consistency_weight": float(self.low_consistency_weight),
            "sequential_edge_weight": float(self.sequential_edge_weight),
            "object_edge_weight": float(self.object_edge_weight),
            "huber_delta": float(self.huber_delta),
            "rotation_residual_scale_m": float(self.rotation_residual_scale_m),
            "max_pose_delta_rotation_deg": float(self.max_pose_delta_rotation_deg),
            "max_pose_delta_translation_m": float(self.max_pose_delta_translation_m),
            "optimizer_max_nfev": int(self.optimizer_max_nfev),
            "dinov3_checkpoint": (
                None if self.dinov3_checkpoint is None else str(self.dinov3_checkpoint)
            ),
            "dinov3_root": str(self.dinov3_root),
            "dinov3_variant": str(self.dinov3_variant),
            "dinov3_device": str(self.dinov3_device),
            "dinov3_dtype": str(self.dinov3_dtype),
        }


@dataclass(frozen=True)
class FeatureMatch:
    """A mask-constrained 2D match in the canonical frame grids."""

    u_i: float
    v_i: float
    u_j: float
    v_j: float
    score: float
    similarity: float

    def to_dict(self) -> dict[str, float]:
        return {
            "u_i": float(self.u_i),
            "v_i": float(self.v_i),
            "u_j": float(self.u_j),
            "v_j": float(self.v_j),
            "score": float(self.score),
            "similarity": float(self.similarity),
        }


class ObjectFeatureMatcher(Protocol):
    """Replaceable feature matcher used only inside object masks."""

    backend_name: str

    def match(
        self,
        image_i: str | Path,
        image_j: str | Path,
        mask_i: torch.Tensor,
        mask_j: torch.Tensor,
        *,
        image_size_i: SizeHW,
        image_size_j: SizeHW,
    ) -> Sequence[FeatureMatch]:
        ...


@dataclass(frozen=True)
class _FeatureGrid:
    features: torch.Tensor
    coordinates: torch.Tensor


class RGBPatchFeatureMatcher:
    """Small dependency-free patch descriptor for the first experiment.

    It is deliberately not advertised as a learned correspondence model.  It
    provides a deterministic, auditable baseline: local RGB statistics and
    gradients are pooled on a coarse grid, and only patches touching the SAM
    masks are compared.
    """

    backend_name = "rgb_patch"

    def __init__(
        self,
        *,
        patch_size: int = 8,
        max_points: int = 1024,
        cosine_threshold: float = 0.65,
        ratio_margin: float = 0.02,
        max_matches: int = 256,
    ) -> None:
        self.patch_size = int(patch_size)
        self.max_points = int(max_points)
        self.cosine_threshold = float(cosine_threshold)
        self.ratio_margin = float(ratio_margin)
        self.max_matches = int(max_matches)
        if self.patch_size < 1 or self.max_points < 1 or self.max_matches < 1:
            raise ValueError("RGB patch matcher sizes must be positive.")
        if not -1.0 <= self.cosine_threshold <= 1.0:
            raise ValueError("cosine_threshold must be in [-1,1].")
        if not 0.0 <= self.ratio_margin <= 2.0:
            raise ValueError("ratio_margin must be in [0,2].")
        self._cache: dict[tuple[str, SizeHW], _FeatureGrid] = {}

    def match(
        self,
        image_i: str | Path,
        image_j: str | Path,
        mask_i: torch.Tensor,
        mask_j: torch.Tensor,
        *,
        image_size_i: SizeHW,
        image_size_j: SizeHW,
    ) -> tuple[FeatureMatch, ...]:
        grid_i = self._grid(image_i, image_size_i)
        grid_j = self._grid(image_j, image_size_j)
        return _match_feature_grids(
            grid_i,
            grid_j,
            mask_i,
            mask_j,
            image_size_i=image_size_i,
            image_size_j=image_size_j,
            cosine_threshold=self.cosine_threshold,
            ratio_margin=self.ratio_margin,
            max_matches=self.max_matches,
        )

    def _grid(self, path: str | Path, image_size: SizeHW) -> _FeatureGrid:
        key = (str(Path(path).expanduser().resolve()), tuple(map(int, image_size)))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        image = _load_image(path, image_size)
        height, width = image_size
        patch = min(self.patch_size, height, width)
        values = image.permute(2, 0, 1).unsqueeze(0)
        mean = F.avg_pool2d(values, kernel_size=patch, stride=patch)
        second = F.avg_pool2d(values * values, kernel_size=patch, stride=patch)
        std = (second - mean * mean).clamp_min(0.0).sqrt()
        gray = (
            values[:, 0:1] * 0.299
            + values[:, 1:2] * 0.587
            + values[:, 2:3] * 0.114
        )
        grad_x = F.pad(gray[:, :, :, 1:] - gray[:, :, :, :-1], (0, 1, 0, 0))
        grad_y = F.pad(gray[:, :, 1:, :] - gray[:, :, :-1, :], (0, 0, 0, 1))
        grad = torch.cat(
            (
                F.avg_pool2d(grad_x.abs(), patch, patch),
                F.avg_pool2d(grad_y.abs(), patch, patch),
                F.avg_pool2d(gray, patch, patch),
            ),
            dim=1,
        )
        pooled = torch.cat((mean, std, grad), dim=1)[0]
        grid_height, grid_width = int(pooled.shape[1]), int(pooled.shape[2])
        features = pooled.flatten(1).transpose(0, 1).contiguous()
        features = F.normalize(features, dim=-1)
        yy, xx = torch.meshgrid(
            torch.arange(grid_height, dtype=torch.float32),
            torch.arange(grid_width, dtype=torch.float32),
            indexing="ij",
        )
        coordinates = torch.stack(
            (
                (xx.flatten() * patch + (patch - 1) * 0.5).clamp(0, width - 1),
                (yy.flatten() * patch + (patch - 1) * 0.5).clamp(0, height - 1),
            ),
            dim=-1,
        )
        if int(features.shape[0]) > self.max_points:
            texture = std[0].sum(dim=0).flatten()
            order = torch.argsort(texture, descending=True, stable=True)
            order = order[: self.max_points]
            features = features.index_select(0, order)
            coordinates = coordinates.index_select(0, order)
        result = _FeatureGrid(features=features.cpu(), coordinates=coordinates.cpu())
        self._cache[key] = result
        return result


class DinoV3PatchFeatureMatcher:
    """Optional adapter around the repository's existing DINOv3 encoder."""

    backend_name = "dinov3"

    def __init__(self, config: ObjectPoseRefinementConfig) -> None:
        from ..dinov3_object_features import DinoV3DenseEncoder, DinoV3FeatureConfig

        self.encoder = DinoV3DenseEncoder(
            DinoV3FeatureConfig(
                checkpoint=config.dinov3_checkpoint,
                root=config.dinov3_root,
                preferred_variant=config.dinov3_variant,
                device=config.dinov3_device,
                dtype=config.dinov3_dtype,
            )
        )
        self.max_points = int(config.feature_max_points)
        self.cosine_threshold = float(config.feature_cosine_threshold)
        self.ratio_margin = float(config.feature_ratio_margin)
        self.max_matches = int(config.max_feature_matches)
        self._cache: dict[tuple[str, SizeHW], _FeatureGrid] = {}

    def match(
        self,
        image_i: str | Path,
        image_j: str | Path,
        mask_i: torch.Tensor,
        mask_j: torch.Tensor,
        *,
        image_size_i: SizeHW,
        image_size_j: SizeHW,
    ) -> tuple[FeatureMatch, ...]:
        return _match_feature_grids(
            self._grid(image_i, image_size_i),
            self._grid(image_j, image_size_j),
            mask_i,
            mask_j,
            image_size_i=image_size_i,
            image_size_j=image_size_j,
            cosine_threshold=self.cosine_threshold,
            ratio_margin=self.ratio_margin,
            max_matches=self.max_matches,
        )

    def _grid(self, path: str | Path, image_size: SizeHW) -> _FeatureGrid:
        key = (str(Path(path).expanduser().resolve()), tuple(map(int, image_size)))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        image = _load_image(path, image_size)
        dense, metadata = self.encoder.encode_dense(image.unsqueeze(0))
        dense = F.normalize(dense[0].float(), dim=-1)
        grid_height, grid_width = int(dense.shape[0]), int(dense.shape[1])
        target_height, target_width = image_size
        yy, xx = torch.meshgrid(
            torch.arange(grid_height, dtype=torch.float32),
            torch.arange(grid_width, dtype=torch.float32),
            indexing="ij",
        )
        input_height, input_width = (
            int(metadata["input_size"][0]),
            int(metadata["input_size"][1]),
        )
        coordinates = torch.stack(
            (
                ((xx + 0.5) * input_width / grid_width - 0.5)
                * target_width
                / input_width,
                ((yy + 0.5) * input_height / grid_height - 0.5)
                * target_height
                / input_height,
            ),
            dim=-1,
        ).reshape(-1, 2).clamp_min(0.0)
        coordinates[:, 0].clamp_max_(target_width - 1)
        coordinates[:, 1].clamp_max_(target_height - 1)
        features = dense.reshape(-1, dense.shape[-1])
        if int(features.shape[0]) > self.max_points:
            order = torch.linspace(
                0,
                int(features.shape[0]) - 1,
                steps=self.max_points,
            ).round().long()
            features = features.index_select(0, order)
            coordinates = coordinates.index_select(0, order)
        result = _FeatureGrid(features.cpu(), coordinates.cpu())
        self._cache[key] = result
        return result


def create_object_feature_matcher(
    config: ObjectPoseRefinementConfig,
) -> ObjectFeatureMatcher:
    """Create the requested feature backend lazily."""

    config.validate()
    if str(config.feature_backend).strip().lower() == "dinov3":
        return DinoV3PatchFeatureMatcher(config)
    return RGBPatchFeatureMatcher(
        patch_size=config.feature_patch_size,
        max_points=config.feature_max_points,
        cosine_threshold=config.feature_cosine_threshold,
        ratio_margin=config.feature_ratio_margin,
        max_matches=config.max_feature_matches,
    )


def _match_feature_grids(
    grid_i: _FeatureGrid,
    grid_j: _FeatureGrid,
    mask_i: torch.Tensor,
    mask_j: torch.Tensor,
    *,
    image_size_i: SizeHW,
    image_size_j: SizeHW,
    cosine_threshold: float,
    ratio_margin: float,
    max_matches: int,
) -> tuple[FeatureMatch, ...]:
    if not grid_i.features.numel() or not grid_j.features.numel():
        return ()
    support_i = _feature_mask_support(mask_i, grid_i.coordinates, image_size_i)
    support_j = _feature_mask_support(mask_j, grid_j.coordinates, image_size_j)
    indices_i = torch.nonzero(support_i, as_tuple=False).flatten()
    indices_j = torch.nonzero(support_j, as_tuple=False).flatten()
    if not indices_i.numel() or not indices_j.numel():
        return ()
    features_i = F.normalize(grid_i.features.index_select(0, indices_i).float(), dim=-1)
    features_j = F.normalize(grid_j.features.index_select(0, indices_j).float(), dim=-1)
    similarity = features_i @ features_j.transpose(0, 1)
    best_values, best_j = similarity.max(dim=1)
    best_i_for_j = similarity.argmax(dim=0)
    mutual = best_i_for_j.index_select(0, best_j) == torch.arange(
        int(indices_i.shape[0])
    )
    if int(indices_j.shape[0]) > 1:
        top_values = torch.topk(similarity, k=2, dim=1).values
        ratio_ok = (top_values[:, 0] - top_values[:, 1]) >= float(ratio_margin)
    else:
        ratio_ok = torch.ones(int(indices_i.shape[0]), dtype=torch.bool)
    keep = mutual & ratio_ok & (best_values >= float(cosine_threshold))
    selected_i = torch.nonzero(keep, as_tuple=False).flatten()
    if not selected_i.numel():
        return ()
    order = torch.argsort(best_values.index_select(0, selected_i), descending=True, stable=True)
    selected_i = selected_i.index_select(0, order[: int(max_matches)])
    selected_j = best_j.index_select(0, selected_i)
    output: list[FeatureMatch] = []
    for source_index, target_index in zip(selected_i.tolist(), selected_j.tolist()):
        sim = float(best_values[source_index])
        score = max(0.0, min(1.0, 0.5 * (sim + 1.0)))
        point_i = grid_i.coordinates[int(indices_i[source_index])]
        point_j = grid_j.coordinates[int(indices_j[target_index])]
        output.append(
            FeatureMatch(
                u_i=float(point_i[0]),
                v_i=float(point_i[1]),
                u_j=float(point_j[0]),
                v_j=float(point_j[1]),
                score=score,
                similarity=sim,
            )
        )
    return tuple(output)


def _feature_mask_support(
    mask: torch.Tensor,
    coordinates: torch.Tensor,
    image_size: SizeHW,
) -> torch.Tensor:
    value = resize_bool_mask(mask, image_size)
    height, width = image_size
    u = coordinates[:, 0].round().long().clamp(0, width - 1)
    v = coordinates[:, 1].round().long().clamp(0, height - 1)
    return value[v, u]


@dataclass(frozen=True)
class ObjectObservationRecord:
    """Filtered observation used for pair generation."""

    frame_id: int
    category: str
    instance_id: int
    mask: torch.Tensor
    track_score: float
    mask_area: int
    mask_area_ratio: float
    valid_geometry_points: int
    geometry_confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": int(self.frame_id),
            "category": str(self.category),
            "instance_id": int(self.instance_id),
            "track_score": float(self.track_score),
            "mask_area": int(self.mask_area),
            "mask_area_ratio": float(self.mask_area_ratio),
            "valid_geometry_points": int(self.valid_geometry_points),
            "geometry_confidence": float(self.geometry_confidence),
        }


@dataclass(frozen=True)
class ObjectFramePair:
    """One same-instance long-range candidate with provenance."""

    candidate_id: int
    left: ObjectObservationRecord
    right: ObjectObservationRecord
    priority: float

    @property
    def frame_i(self) -> int:
        return int(self.left.frame_id)

    @property
    def frame_j(self) -> int:
        return int(self.right.frame_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": int(self.candidate_id),
            "instance_id": int(self.left.instance_id),
            "category": str(self.left.category),
            "frame_i": int(self.frame_i),
            "frame_j": int(self.frame_j),
            "temporal_gap": int(self.frame_j - self.frame_i),
            "track_score_i": float(self.left.track_score),
            "track_score_j": float(self.right.track_score),
            "mask_area_i": int(self.left.mask_area),
            "mask_area_j": int(self.right.mask_area),
            "mask_area_ratio_i": float(self.left.mask_area_ratio),
            "mask_area_ratio_j": float(self.right.mask_area_ratio),
            "valid_geometry_points_i": int(self.left.valid_geometry_points),
            "valid_geometry_points_j": int(self.right.valid_geometry_points),
            "geometry_confidence_i": float(self.left.geometry_confidence),
            "geometry_confidence_j": float(self.right.geometry_confidence),
            "pair_priority": float(self.priority),
        }


@dataclass(frozen=True)
class ObjectCorrespondence:
    """One valid 3D-3D pair in the two local camera coordinates."""

    pixel_i: tuple[float, float]
    pixel_j: tuple[float, float]
    point_i: torch.Tensor
    point_j: torch.Tensor
    feature_score: float
    geometry_confidence_i: float
    geometry_confidence_j: float
    weight: float


@dataclass(frozen=True)
class RigidRegistration:
    """Result of weighted RANSAC + Procrustes registration."""

    transform: torch.Tensor
    inlier_mask: torch.Tensor
    num_matches: int
    num_inliers: int
    inlier_ratio: float
    rmse: float
    weighted_inlier_mass: float


@dataclass(frozen=True)
class ObjectPoseEdge:
    """An accepted same-instance relative camera-pose constraint."""

    frame_i: int
    frame_j: int
    instance_id: int
    category: str
    relative_pose: torch.Tensor
    num_matches: int
    num_inliers: int
    inlier_ratio: float
    rmse: float
    edge_weight: float
    raw_rotation_disagreement_deg: float
    raw_translation_disagreement_m: float
    consistency: str
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_i": int(self.frame_i),
            "frame_j": int(self.frame_j),
            "instance_id": int(self.instance_id),
            "category": str(self.category),
            "relative_pose": self.relative_pose.detach().float().cpu().tolist(),
            "num_matches": int(self.num_matches),
            "num_inliers": int(self.num_inliers),
            "inlier_ratio": float(self.inlier_ratio),
            "rmse": float(self.rmse),
            "edge_weight": float(self.edge_weight),
            "raw_rotation_disagreement_deg": float(self.raw_rotation_disagreement_deg),
            "raw_translation_disagreement_m": float(self.raw_translation_disagreement_m),
            "consistency": str(self.consistency),
            "provenance": _json_safe(self.provenance),
        }


@dataclass(frozen=True)
class PoseRefinementResult:
    """Auditable output of the object-pose refinement backend."""

    frame_ids: tuple[int, ...]
    raw_camera_to_world: tuple[torch.Tensor, ...]
    refined_camera_to_world: tuple[torch.Tensor, ...]
    candidates: tuple[Mapping[str, Any], ...]
    accepted_edges: tuple[ObjectPoseEdge, ...]
    rejected_edges: tuple[Mapping[str, Any], ...]
    summary: Mapping[str, Any]

    def pose_by_frame(self) -> dict[int, torch.Tensor]:
        return {
            int(frame_id): pose.detach().float().cpu()
            for frame_id, pose in zip(self.frame_ids, self.refined_camera_to_world)
        }


@dataclass(frozen=True)
class _GeometryView:
    camera_points: torch.Tensor
    valid: torch.Tensor
    confidence: torch.Tensor


class ObjectPoseRefiner:
    """Build and solve a conservative object-augmented pose graph."""

    def __init__(
        self,
        config: ObjectPoseRefinementConfig | None = None,
        *,
        matcher: ObjectFeatureMatcher | None = None,
    ) -> None:
        self.config = (config or ObjectPoseRefinementConfig()).validate()
        self.matcher = matcher or create_object_feature_matcher(self.config)

    def refine(
        self,
        geometry_frames: Sequence[GeometryFrame],
        segmentation_frames: Sequence[SegmentationFrame],
        image_paths: Sequence[str | Path],
    ) -> PoseRefinementResult:
        geometry = tuple(geometry_frames)
        segmentation = tuple(segmentation_frames)
        paths = tuple(Path(path) for path in image_paths)
        if not geometry:
            raise ValueError("ObjectPoseRefiner requires at least one geometry frame.")
        if len(geometry) != len(segmentation) or len(geometry) != len(paths):
            raise ValueError("Geometry, segmentation, and RGB counts must agree.")
        input_frame_ids = tuple(int(frame.frame_id) for frame in geometry)
        if input_frame_ids != tuple(sorted(input_frame_ids)):
            raise ValueError("ObjectPoseRefiner requires increasing geometry frame IDs.")
        segmentation_by_id = {int(frame.frame_id): frame for frame in segmentation}
        if len(segmentation_by_id) != len(segmentation):
            raise ValueError("Segmentation frames contain duplicate frame IDs.")
        if set(segmentation_by_id) != {int(frame.frame_id) for frame in geometry}:
            raise ValueError("Geometry and segmentation frame IDs do not match.")
        if any(not path.is_file() for path in paths):
            missing = next(path for path in paths if not path.is_file())
            raise FileNotFoundError(f"Object feature image does not exist: {missing}")

        frame_ids = tuple(int(frame.frame_id) for frame in geometry)
        geometry_by_id = {int(frame.frame_id): frame for frame in geometry}
        paths_by_id = {
            int(frame_id): paths[index] for index, frame_id in enumerate(frame_ids)
        }
        views = {
            frame_id: _geometry_view(
                frame,
                min_confidence=self.config.min_geometry_confidence,
            )
            for frame_id, frame in geometry_by_id.items()
        }
        records_by_instance, tracked_instance_ids, filter_stats = (
            self._collect_records(geometry, segmentation_by_id, views)
        )
        candidates = self._generate_candidates(records_by_instance)
        accepted: list[ObjectPoseEdge] = []
        rejected: list[dict[str, Any]] = []
        for candidate in candidates:
            candidate_row = candidate.to_dict()
            try:
                matches = tuple(
                    self.matcher.match(
                        paths_by_id[candidate.frame_i],
                        paths_by_id[candidate.frame_j],
                        candidate.left.mask,
                        candidate.right.mask,
                        image_size_i=geometry_by_id[candidate.frame_i].image_size,
                        image_size_j=geometry_by_id[candidate.frame_j].image_size,
                    )
                )
                candidate_row["feature_backend"] = str(
                    getattr(self.matcher, "backend_name", type(self.matcher).__name__)
                )
                candidate_row["num_feature_matches"] = int(len(matches))
                correspondences = _build_3d_correspondences(
                    matches,
                    candidate.left,
                    candidate.right,
                    views[candidate.frame_i],
                    views[candidate.frame_j],
                )
                candidate_row["num_valid_3d_correspondences"] = int(
                    len(correspondences)
                )
                if len(correspondences) < self.config.min_matches:
                    raise _EdgeRejection(
                        "reject:too_few_valid_3d_correspondences"
                    )
                registration, registration_reason = estimate_rigid_transform_ransac(
                    torch.stack([item.point_i for item in correspondences]),
                    torch.stack([item.point_j for item in correspondences]),
                    torch.tensor([item.weight for item in correspondences]),
                    config=self.config,
                )
                if registration is None:
                    raise _EdgeRejection(registration_reason)
                candidate_row.update(
                    {
                        "num_inliers": int(registration.num_inliers),
                        "inlier_ratio": float(registration.inlier_ratio),
                        "registration_rmse": float(registration.rmse),
                    }
                )
                raw_pose_i = _require_pose(geometry_by_id[candidate.frame_i])
                raw_pose_j = _require_pose(geometry_by_id[candidate.frame_j])
                raw_relative = torch.linalg.inv(raw_pose_j) @ raw_pose_i
                rotation_disagreement, translation_disagreement = (
                    _relative_pose_disagreement(raw_relative, registration.transform)
                )
                limit_rotation = self.config.max_rotation_disagreement_deg
                limit_translation = self.config.max_translation_disagreement_m
                multiplier = self.config.reject_extreme_disagreement_multiplier
                if (
                    rotation_disagreement > limit_rotation * multiplier
                    or translation_disagreement > limit_translation * multiplier
                ):
                    raise _EdgeRejection("reject:extreme_pose_disagreement")
                low_consistency = (
                    rotation_disagreement > limit_rotation
                    or translation_disagreement > limit_translation
                )
                consistency = "low" if low_consistency else "high"
                pair_quality = math.sqrt(
                    max(0.0, candidate.left.track_score)
                    * max(0.0, candidate.right.track_score)
                    * max(0.0, candidate.left.geometry_confidence)
                    * max(0.0, candidate.right.geometry_confidence)
                )
                consistency_weight = (
                    self.config.low_consistency_weight if low_consistency else 1.0
                )
                edge_weight = max(
                    1e-6,
                    self.config.object_edge_weight
                    * pair_quality
                    * max(registration.inlier_ratio, 0.05)
                    * consistency_weight,
                )
                accepted.append(
                    ObjectPoseEdge(
                        frame_i=candidate.frame_i,
                        frame_j=candidate.frame_j,
                        instance_id=int(candidate.left.instance_id),
                        category=str(candidate.left.category),
                        relative_pose=registration.transform,
                        num_matches=int(registration.num_matches),
                        num_inliers=int(registration.num_inliers),
                        inlier_ratio=float(registration.inlier_ratio),
                        rmse=float(registration.rmse),
                        edge_weight=float(edge_weight),
                        raw_rotation_disagreement_deg=float(rotation_disagreement),
                        raw_translation_disagreement_m=float(translation_disagreement),
                        consistency=consistency,
                        provenance=candidate_row,
                    )
                )
            except _EdgeRejection as exc:
                candidate_row["reason"] = str(exc)
                rejected.append(candidate_row)
            except Exception as exc:  # optional backend failure must be auditable
                candidate_row["reason"] = "reject:matcher_or_registration_error"
                candidate_row["error_type"] = type(exc).__name__
                candidate_row["error"] = str(exc)
                rejected.append(candidate_row)

        raw_poses = tuple(_require_pose(frame) for frame in geometry)
        refined_poses, optimizer_summary = _optimize_pose_graph(
            frame_ids,
            raw_poses,
            accepted,
            self.config,
        )
        summary = _build_summary(
            config=self.config,
            frame_ids=frame_ids,
            tracked_instance_ids=tracked_instance_ids,
            records_by_instance=records_by_instance,
            candidates=candidates,
            accepted=accepted,
            rejected=rejected,
            filter_stats=filter_stats,
            optimizer_summary=optimizer_summary,
            raw_poses=raw_poses,
            refined_poses=refined_poses,
        )
        return PoseRefinementResult(
            frame_ids=frame_ids,
            raw_camera_to_world=tuple(p.detach().float().cpu() for p in raw_poses),
            refined_camera_to_world=tuple(
                p.detach().float().cpu() for p in refined_poses
            ),
            candidates=tuple(candidate.to_dict() for candidate in candidates),
            accepted_edges=tuple(accepted),
            rejected_edges=tuple(rejected),
            summary=summary,
        )

    def _collect_records(
        self,
        geometry: Sequence[GeometryFrame],
        segmentation_by_id: Mapping[int, SegmentationFrame],
        views: Mapping[int, _GeometryView],
    ) -> tuple[dict[int, list[ObjectObservationRecord]], set[int], dict[str, int]]:
        records: dict[int, list[ObjectObservationRecord]] = {}
        tracked_ids: set[int] = set()
        filter_stats: Counter[str] = Counter()
        for frame in geometry:
            frame_id = int(frame.frame_id)
            segmentation = segmentation_by_id[frame_id]
            view = views[frame_id]
            for observation in segmentation.observations:
                instance_id = int(observation.instance_id)
                tracked_ids.add(instance_id)
                mask = resize_bool_mask(observation.mask, frame.image_size)
                area = int(mask.sum())
                area_ratio = float(area) / float(frame.image_size[0] * frame.image_size[1])
                selected = mask & view.valid
                valid_points = int(selected.sum())
                geometry_confidence = (
                    float(view.confidence[selected].mean()) if valid_points else 0.0
                )
                checks = (
                    (float(observation.score) >= self.config.min_track_score, "low_track_score"),
                    (area >= self.config.min_mask_pixels, "too_few_mask_pixels"),
                    (area_ratio <= self.config.max_mask_area_ratio, "mask_too_large"),
                    (valid_points >= self.config.min_geometry_points, "too_few_geometry_points"),
                    (geometry_confidence >= self.config.min_geometry_confidence, "low_geometry_confidence"),
                )
                failure = next((reason for passed, reason in checks if not passed), None)
                if failure is not None:
                    filter_stats[failure] += 1
                    continue
                record = ObjectObservationRecord(
                    frame_id=frame_id,
                    category=str(observation.category),
                    instance_id=instance_id,
                    mask=mask,
                    track_score=float(observation.score),
                    mask_area=area,
                    mask_area_ratio=area_ratio,
                    valid_geometry_points=valid_points,
                    geometry_confidence=geometry_confidence,
                )
                records.setdefault(instance_id, []).append(record)
        return records, tracked_ids, dict(filter_stats)

    def _generate_candidates(
        self,
        records_by_instance: Mapping[int, Sequence[ObjectObservationRecord]],
    ) -> list[ObjectFramePair]:
        candidates: list[tuple[float, ObjectObservationRecord, ObjectObservationRecord]] = []
        for instance_id in sorted(records_by_instance):
            records = sorted(
                records_by_instance[instance_id], key=lambda record: record.frame_id
            )
            if len(records) < 2:
                continue
            maximum_gap = max(1, records[-1].frame_id - records[0].frame_id)
            instance_pairs = []
            for left_index, left in enumerate(records[:-1]):
                for right in records[left_index + 1 :]:
                    gap = int(right.frame_id - left.frame_id)
                    if gap < self.config.min_temporal_gap:
                        continue
                    quality = math.sqrt(
                        max(0.0, left.track_score)
                        * max(0.0, right.track_score)
                        * max(0.0, left.geometry_confidence)
                        * max(0.0, right.geometry_confidence)
                    )
                    priority = 0.7 * (gap / maximum_gap) + 0.3 * quality
                    instance_pairs.append((priority, left, right))
            instance_pairs.sort(
                key=lambda item: (
                    -float(item[0]),
                    -int(item[2].frame_id - item[1].frame_id),
                    int(item[1].frame_id),
                    int(item[2].frame_id),
                )
            )
            candidates.extend(instance_pairs[: self.config.max_pairs_per_instance])
        candidates.sort(
            key=lambda item: (
                -float(item[0]),
                int(item[1].instance_id),
                int(item[1].frame_id),
                int(item[2].frame_id),
            )
        )
        output = []
        for candidate_id, (priority, left, right) in enumerate(
            candidates[: self.config.max_total_candidate_pairs]
        ):
            output.append(
                ObjectFramePair(
                    candidate_id=int(candidate_id),
                    left=left,
                    right=right,
                    priority=float(priority),
                )
            )
        return output


def estimate_rigid_transform_ransac(
    points_i: torch.Tensor,
    points_j: torch.Tensor,
    weights: torch.Tensor | None = None,
    *,
    config: ObjectPoseRefinementConfig | None = None,
) -> tuple[RigidRegistration | None, str]:
    """Estimate ``X_j ~= R @ X_i + t`` with weighted RANSAC."""

    cfg = (config or ObjectPoseRefinementConfig()).validate()
    left = torch.as_tensor(points_i).detach().float().cpu()
    right = torch.as_tensor(points_j).detach().float().cpu()
    if left.ndim != 2 or right.ndim != 2 or left.shape != right.shape or left.shape[-1] != 3:
        raise ValueError("Rigid registration points must both have shape [N,3].")
    count = int(left.shape[0])
    if count < 3:
        return None, "reject:too_few_points_for_rigid_transform"
    if weights is None:
        weight = torch.ones(count)
    else:
        weight = torch.as_tensor(weights).detach().float().cpu().reshape(-1)
        if int(weight.shape[0]) != count:
            raise ValueError("Rigid registration weights must have length N.")
    finite = torch.isfinite(left).all(dim=1) & torch.isfinite(right).all(dim=1)
    finite &= torch.isfinite(weight) & (weight > 0.0)
    if int(finite.sum()) < 3:
        return None, "reject:too_few_finite_points"
    left = left[finite]
    right = right[finite]
    weight = weight[finite]
    count = int(left.shape[0])
    generator = np.random.default_rng(int(cfg.ransac_seed))
    best_mask: torch.Tensor | None = None
    best_mass = -1.0
    best_rmse = float("inf")
    iterations = max(1, int(cfg.ransac_iterations))
    for iteration in range(iterations):
        if count == 3:
            sample = np.arange(3)
        else:
            sample = generator.choice(count, size=3, replace=False)
        sample_tensor = torch.as_tensor(sample, dtype=torch.long)
        if _rank_deficient(left.index_select(0, sample_tensor)) or _rank_deficient(
            right.index_select(0, sample_tensor)
        ):
            continue
        transform = weighted_rigid_transform(
            left.index_select(0, sample_tensor),
            right.index_select(0, sample_tensor),
            weight.index_select(0, sample_tensor),
        )
        residual = torch.linalg.vector_norm(
            _apply_transform(transform, left) - right, dim=-1
        )
        mask = residual <= float(cfg.ransac_inlier_threshold_m)
        inlier_mass = float(weight[mask].sum())
        rmse = _weighted_rmse(residual[mask], weight[mask]) if bool(mask.any()) else float("inf")
        if inlier_mass > best_mass or (math.isclose(inlier_mass, best_mass) and rmse < best_rmse):
            best_mask = mask
            best_mass = inlier_mass
            best_rmse = rmse
    if best_mask is None or int(best_mask.sum()) < 3:
        return None, "reject:ransac_no_model"
    transform = weighted_rigid_transform(left[best_mask], right[best_mask], weight[best_mask])
    residual = torch.linalg.vector_norm(_apply_transform(transform, left) - right, dim=-1)
    final_mask = residual <= float(cfg.ransac_inlier_threshold_m)
    if int(final_mask.sum()) >= 3:
        transform = weighted_rigid_transform(left[final_mask], right[final_mask], weight[final_mask])
        residual = torch.linalg.vector_norm(_apply_transform(transform, left) - right, dim=-1)
        final_mask = residual <= float(cfg.ransac_inlier_threshold_m)
    num_inliers = int(final_mask.sum())
    inlier_ratio = float(num_inliers) / float(count)
    rmse = _weighted_rmse(residual[final_mask], weight[final_mask]) if num_inliers else float("inf")
    if num_inliers < int(cfg.min_inliers):
        return None, "reject:too_few_ransac_inliers"
    if inlier_ratio < float(cfg.min_inlier_ratio):
        return None, "reject:low_ransac_inlier_ratio"
    if not math.isfinite(rmse) or rmse > float(cfg.max_registration_rmse_m):
        return None, "reject:registration_rmse_too_large"
    return (
        RigidRegistration(
            transform=transform,
            inlier_mask=final_mask,
            num_matches=count,
            num_inliers=num_inliers,
            inlier_ratio=inlier_ratio,
            rmse=rmse,
            weighted_inlier_mass=float(weight[final_mask].sum()),
        ),
        "accept",
    )


def weighted_rigid_transform(
    points_i: torch.Tensor,
    points_j: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Weighted SVD/Procrustes transform mapping points_i to points_j."""

    left = torch.as_tensor(points_i).detach().float().cpu()
    right = torch.as_tensor(points_j).detach().float().cpu()
    if left.ndim != 2 or right.shape != left.shape or left.shape[-1] != 3:
        raise ValueError("Weighted rigid transform expects matching [N,3] tensors.")
    if int(left.shape[0]) < 3:
        raise ValueError("At least three points are required for rigid transform.")
    if weights is None:
        value = torch.ones(left.shape[0])
    else:
        value = torch.as_tensor(weights).detach().float().cpu().reshape(-1)
        if int(value.shape[0]) != int(left.shape[0]):
            raise ValueError("Weighted rigid transform weights must have length N.")
    value = value.clamp_min(0.0)
    if float(value.sum()) <= 0.0:
        raise ValueError("Weighted rigid transform needs positive total weight.")
    normalized = value / value.sum()
    center_left = (left * normalized[:, None]).sum(dim=0)
    center_right = (right * normalized[:, None]).sum(dim=0)
    covariance = (left - center_left).transpose(0, 1) @ (
        normalized[:, None] * (right - center_right)
    )
    u, _, v_transpose = torch.linalg.svd(covariance, full_matrices=False)
    rotation = v_transpose.transpose(0, 1) @ u.transpose(0, 1)
    if float(torch.linalg.det(rotation)) < 0.0:
        v_transpose = v_transpose.clone()
        v_transpose[-1] *= -1.0
        rotation = v_transpose.transpose(0, 1) @ u.transpose(0, 1)
    translation = center_right - rotation @ center_left
    transform = torch.eye(4, dtype=torch.float32)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def apply_refined_camera_poses(
    geometry_frames: Sequence[GeometryFrame],
    refinement: PoseRefinementResult,
) -> tuple[GeometryFrame, ...]:
    """Return geometry frames with only ``camera_to_world`` replaced.

    HorizonStream depth/intrinsics and SAM masks remain untouched.  A legacy
    world-point-only backend is also supported by converting its raw points to
    camera coordinates once and re-applying the refined pose.
    """

    pose_by_frame = refinement.pose_by_frame()
    raw_by_frame = {
        int(frame_id): pose.detach().float().cpu()
        for frame_id, pose in zip(
            refinement.frame_ids, refinement.raw_camera_to_world
        )
    }
    output = []
    refinement_source = str(
        refinement.summary.get(
            "method",
            "sam_instance_object_edges",
        )
    )
    for frame in geometry_frames:
        frame_id = int(frame.frame_id)
        if frame_id not in pose_by_frame:
            raise ValueError(f"Pose refinement has no pose for frame_id={frame_id}.")
        pose = pose_by_frame[frame_id]
        metadata = dict(frame.metadata)
        metadata.update(
            {
                "pose_variant": "object_pose_refined",
                "pose_refinement_applied": True,
                "pose_refinement_source": refinement_source,
            }
        )
        if frame.depth is not None:
            output.append(
                replace(
                    frame,
                    world_points=None,
                    camera_to_world=pose,
                    metadata=metadata,
                )
            )
            continue
        if frame.world_points is None:
            raise ValueError(
                "Refined geometry requires depth or world_points plus a raw pose."
            )
        raw_pose = raw_by_frame.get(frame_id)
        if raw_pose is None:
            raise ValueError(f"Raw pose is missing for frame_id={frame_id}.")
        world = frame.world_points.detach().float().cpu()
        homogeneous = torch.cat(
            (world, torch.ones(*world.shape[:-1], 1, dtype=world.dtype)), dim=-1
        )
        local = torch.einsum(
            "ij,hwj->hwi", torch.linalg.inv(raw_pose), homogeneous
        )[..., :3]
        refined_homogeneous = torch.cat(
            (
                local,
                torch.ones(*local.shape[:-1], 1, dtype=local.dtype),
            ),
            dim=-1,
        )
        refined_world = torch.einsum("ij,hwj->hwi", pose, refined_homogeneous)[..., :3]
        output.append(
            replace(
                frame,
                world_points=refined_world,
                camera_to_world=pose,
                metadata=metadata,
            )
        )
    return tuple(output)


def write_pose_refinement_debug(
    result: PoseRefinementResult,
    output_dir: str | Path,
) -> dict[str, str]:
    """Write candidate/edge/trajectory artifacts for an auditable run."""

    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "candidate_edges": directory / "candidate_edges.json",
        "accepted_edges": directory / "accepted_edges.json",
        "rejected_edges": directory / "rejected_edges.json",
        "raw_trajectory": directory / "raw_trajectory.txt",
        "refined_trajectory": directory / "refined_trajectory.txt",
        "refined_camera_to_world": directory / "refined_camera_to_world.pt",
        "summary": directory / "pose_refinement_summary.json",
    }
    _write_json(paths["candidate_edges"], list(result.candidates))
    _write_json(paths["accepted_edges"], [edge.to_dict() for edge in result.accepted_edges])
    _write_json(paths["rejected_edges"], list(result.rejected_edges))
    _write_trajectory(paths["raw_trajectory"], result.frame_ids, result.raw_camera_to_world)
    _write_trajectory(
        paths["refined_trajectory"], result.frame_ids, result.refined_camera_to_world
    )
    torch.save(
        {
            "frame_ids": list(result.frame_ids),
            "raw_camera_to_world": torch.stack(list(result.raw_camera_to_world)),
            "refined_camera_to_world": torch.stack(
                list(result.refined_camera_to_world)
            ),
        },
        paths["refined_camera_to_world"],
    )
    _write_json(paths["summary"], dict(result.summary))
    return {key: str(value) for key, value in paths.items()}


def _geometry_view(
    frame: GeometryFrame,
    *,
    min_confidence: float = 0.0,
) -> _GeometryView:
    frame.validate()
    height, width = frame.image_size
    confidence = geometry_confidence_for_frame(frame)
    if frame.depth is not None:
        depth = frame.depth.detach().float().cpu()
        if depth.ndim == 3:
            depth = depth[..., 0]
        if frame.intrinsics is None:
            raise ValueError("Depth geometry needs intrinsics for object refinement.")
        yy, xx = torch.meshgrid(
            torch.arange(height, dtype=depth.dtype),
            torch.arange(width, dtype=depth.dtype),
            indexing="ij",
        )
        intrinsics = frame.intrinsics.detach().float().cpu()
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        if abs(float(fx)) <= 1e-8 or abs(float(fy)) <= 1e-8:
            raise ValueError("Object refinement received zero focal length.")
        points = torch.stack(
            (
                (xx - intrinsics[0, 2]) / fx * depth,
                (yy - intrinsics[1, 2]) / fy * depth,
                depth,
            ),
            dim=-1,
        )
        valid = torch.isfinite(points).all(dim=-1) & torch.isfinite(depth) & (depth > 0.0)
    else:
        if frame.world_points is None or frame.camera_to_world is None:
            raise ValueError(
                "Object refinement needs depth/intrinsics or world_points/raw pose."
            )
        world = frame.world_points.detach().float().cpu()
        homogeneous = torch.cat(
            (world, torch.ones(*world.shape[:-1], 1, dtype=world.dtype)), dim=-1
        )
        points = torch.einsum(
            "ij,hwj->hwi", _invert_pose(frame.camera_to_world), homogeneous
        )[..., :3]
        valid = torch.isfinite(points).all(dim=-1)
    if frame.valid is not None:
        valid &= frame.valid.detach().bool().cpu()
    valid &= confidence >= float(min_confidence)
    return _GeometryView(camera_points=points, valid=valid, confidence=confidence)


def _build_3d_correspondences(
    matches: Sequence[FeatureMatch],
    left: ObjectObservationRecord,
    right: ObjectObservationRecord,
    view_i: _GeometryView,
    view_j: _GeometryView,
) -> list[ObjectCorrespondence]:
    height_i, width_i = left.mask.shape
    height_j, width_j = right.mask.shape
    output: list[ObjectCorrespondence] = []
    for match in matches:
        u_i, v_i = int(round(match.u_i)), int(round(match.v_i))
        u_j, v_j = int(round(match.u_j)), int(round(match.v_j))
        if not (0 <= u_i < width_i and 0 <= v_i < height_i):
            continue
        if not (0 <= u_j < width_j and 0 <= v_j < height_j):
            continue
        if not bool(left.mask[v_i, u_i]) or not bool(right.mask[v_j, u_j]):
            continue
        if not bool(view_i.valid[v_i, u_i]) or not bool(view_j.valid[v_j, u_j]):
            continue
        point_i = view_i.camera_points[v_i, u_i]
        point_j = view_j.camera_points[v_j, u_j]
        if not bool(torch.isfinite(point_i).all() and torch.isfinite(point_j).all()):
            continue
        confidence_i = float(view_i.confidence[v_i, u_i])
        confidence_j = float(view_j.confidence[v_j, u_j])
        weight = (
            max(0.0, min(1.0, float(match.score)))
            * max(0.0, min(1.0, confidence_i))
            * max(0.0, min(1.0, confidence_j))
            * max(0.0, min(1.0, left.track_score))
            * max(0.0, min(1.0, right.track_score))
        )
        if weight <= 0.0:
            continue
        output.append(
            ObjectCorrespondence(
                pixel_i=(float(match.u_i), float(match.v_i)),
                pixel_j=(float(match.u_j), float(match.v_j)),
                point_i=point_i.detach().float().cpu(),
                point_j=point_j.detach().float().cpu(),
                feature_score=float(match.score),
                geometry_confidence_i=confidence_i,
                geometry_confidence_j=confidence_j,
                weight=float(weight),
            )
        )
    return output


def _optimize_pose_graph(
    frame_ids: Sequence[int],
    raw_poses: Sequence[torch.Tensor],
    object_edges: Sequence[ObjectPoseEdge],
    config: ObjectPoseRefinementConfig,
) -> tuple[tuple[torch.Tensor, ...], dict[str, Any]]:
    raw = tuple(_as_pose_numpy(pose) for pose in raw_poses)
    count = len(raw)
    if count <= 1 or not object_edges:
        return tuple(torch.from_numpy(pose).float() for pose in raw), {
            "backend": "scipy_least_squares",
            "attempted": False,
            "reason": "no_accepted_object_edges" if not object_edges else "single_frame",
            "success": True,
            "initial_cost": 0.0,
            "final_cost": 0.0,
            "nfev": 0,
        }
    index_by_frame = {int(frame_id): index for index, frame_id in enumerate(frame_ids)}
    sequential = []
    for index in range(count - 1):
        measurement = np.linalg.inv(raw[index + 1]) @ raw[index]
        sequential.append(
            (index, index + 1, measurement, float(config.sequential_edge_weight))
        )
    object_constraints = []
    for edge in object_edges:
        if edge.frame_i not in index_by_frame or edge.frame_j not in index_by_frame:
            continue
        object_constraints.append(
            (
                index_by_frame[edge.frame_i],
                index_by_frame[edge.frame_j],
                _as_pose_numpy(edge.relative_pose),
                float(edge.edge_weight),
            )
        )
    constraints = sequential + object_constraints
    variable_count = 6 * (count - 1)
    x0 = np.zeros(variable_count, dtype=np.float64)
    rotation_bound = math.radians(float(config.max_pose_delta_rotation_deg))
    translation_bound = float(config.max_pose_delta_translation_m)
    lower = np.tile(
        np.array(
            [-rotation_bound, -rotation_bound, -rotation_bound,
             -translation_bound, -translation_bound, -translation_bound],
            dtype=np.float64,
        ),
        count - 1,
    )
    upper = -lower

    def current_poses(values: np.ndarray) -> list[np.ndarray]:
        poses = [raw[0].copy()]
        for index in range(1, count):
            start = 6 * (index - 1)
            correction = _se3_exp(values[start : start + 6])
            poses.append(correction @ raw[index])
        return poses

    def residual(values: np.ndarray) -> np.ndarray:
        poses = current_poses(values)
        chunks = []
        for left_index, right_index, measurement, weight in constraints:
            current = np.linalg.inv(poses[right_index]) @ poses[left_index]
            error = np.linalg.inv(measurement) @ current
            rotation = _so3_log(error[:3, :3]) * float(config.rotation_residual_scale_m)
            translation = error[:3, 3]
            chunks.append(math.sqrt(max(weight, 1e-8)) * np.concatenate((rotation, translation)))
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float64)

    initial_residual = residual(x0)
    initial_cost = 0.5 * float(np.dot(initial_residual, initial_residual))
    try:
        from scipy.optimize import least_squares

        optimized = least_squares(
            residual,
            x0,
            bounds=(lower, upper),
            loss="huber",
            f_scale=float(config.huber_delta),
            max_nfev=int(config.optimizer_max_nfev),
            x_scale="jac",
        )
        values = optimized.x if np.isfinite(optimized.x).all() else x0
        final_residual = residual(values)
        final_cost = 0.5 * float(np.dot(final_residual, final_residual))
        poses = tuple(torch.from_numpy(pose).float() for pose in current_poses(values))
        return poses, {
            "backend": "scipy_least_squares",
            "attempted": True,
            "success": bool(optimized.success),
            "status": int(optimized.status),
            "message": str(optimized.message),
            "nfev": int(optimized.nfev),
            "initial_cost": initial_cost,
            "final_cost": final_cost,
            "constraint_count": int(len(constraints)),
            "sequential_edge_count": int(len(sequential)),
            "object_edge_count": int(len(object_constraints)),
        }
    except Exception as exc:
        return tuple(torch.from_numpy(pose).float() for pose in raw), {
            "backend": "scipy_least_squares",
            "attempted": True,
            "success": False,
            "fallback": True,
            "reason": "optimizer_error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "initial_cost": initial_cost,
            "final_cost": initial_cost,
            "constraint_count": int(len(constraints)),
            "sequential_edge_count": int(len(sequential)),
            "object_edge_count": int(len(object_constraints)),
        }


def _build_summary(
    *,
    config: ObjectPoseRefinementConfig,
    frame_ids: Sequence[int],
    tracked_instance_ids: set[int],
    records_by_instance: Mapping[int, Sequence[ObjectObservationRecord]],
    candidates: Sequence[ObjectFramePair],
    accepted: Sequence[ObjectPoseEdge],
    rejected: Sequence[Mapping[str, Any]],
    filter_stats: Mapping[str, int],
    optimizer_summary: Mapping[str, Any],
    raw_poses: Sequence[torch.Tensor],
    refined_poses: Sequence[torch.Tensor],
) -> dict[str, Any]:
    match_values = [int(edge.num_matches) for edge in accepted]
    inlier_values = [int(edge.num_inliers) for edge in accepted]
    inlier_ratio_values = [float(edge.inlier_ratio) for edge in accepted]
    rmse_values = [float(edge.rmse) for edge in accepted]
    candidate_rows = [edge.provenance for edge in accepted] + list(rejected)
    feature_match_values = _present_numeric_values(
        candidate_rows, "num_feature_matches"
    )
    valid_3d_values = _present_numeric_values(
        candidate_rows, "num_valid_3d_correspondences"
    )
    rotation_changes = []
    translation_changes = []
    for raw, refined in zip(raw_poses, refined_poses):
        delta = _invert_pose(raw) @ refined
        rotation_changes.append(math.degrees(float(torch.linalg.norm(_so3_log_torch(delta[:3, :3])))))
        translation_changes.append(float(torch.linalg.vector_norm(delta[:3, 3])))
    reasons = Counter(str(row.get("reason", "unknown")) for row in rejected)
    return {
        "schema": 1,
        "revision": "sam_instance_guided_horizonstream_pose_refinement_r1",
        "enabled": True,
        "config": config.to_dict(),
        "frame_count": int(len(frame_ids)),
        "frame_ids": [int(value) for value in frame_ids],
        "tracked_instance_count": int(len(tracked_instance_ids)),
        "filtered_observation_instance_count": int(len(records_by_instance)),
        "candidate_pair_count": int(len(candidates)),
        "accepted_edge_count": int(len(accepted)),
        "rejected_edge_count": int(len(rejected)),
        "rejected_reason_counts": dict(reasons),
        "match_statistics": _statistics(match_values),
        "inlier_statistics": _statistics(inlier_values),
        "inlier_ratio_statistics": _statistics(inlier_ratio_values),
        "registration_rmse_statistics": _statistics(rmse_values),
        "candidate_feature_match_statistics": _statistics(feature_match_values),
        "candidate_valid_3d_correspondence_statistics": _statistics(valid_3d_values),
        "accepted_consistency_counts": dict(
            Counter(str(edge.consistency) for edge in accepted)
        ),
        "mean_object_registration_rmse": _mean_or_zero(rmse_values),
        "raw_vs_refined_pose_change": {
            "mean_rotation_correction_deg": _mean_or_zero(rotation_changes),
            "max_rotation_correction_deg": max(rotation_changes, default=0.0),
            "mean_translation_correction_m": _mean_or_zero(translation_changes),
            "max_translation_correction_m": max(translation_changes, default=0.0),
        },
        "sequential_edge_count": max(0, len(frame_ids) - 1),
        "object_edge_count": int(len(accepted)),
        "observation_filter_reasons": {str(k): int(v) for k, v in filter_stats.items()},
        "optimizer": _json_safe(optimizer_summary),
    }


def _statistics(values: Sequence[float | int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "max": None}
    numeric = [float(value) for value in values]
    return {
        "count": int(len(numeric)),
        "mean": float(np.mean(numeric)),
        "median": float(np.median(numeric)),
        "max": float(np.max(numeric)),
    }


def _mean_or_zero(values: Sequence[float]) -> float:
    return 0.0 if not values else float(np.mean([float(value) for value in values]))


def _present_numeric_values(
    rows: Sequence[Mapping[str, Any]], key: str
) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float, np.integer, np.floating)):
            if math.isfinite(float(value)):
                values.append(float(value))
    return values


def _relative_pose_disagreement(
    raw_relative: torch.Tensor,
    object_relative: torch.Tensor,
) -> tuple[float, float]:
    delta = torch.linalg.inv(raw_relative) @ object_relative
    rotation = math.degrees(float(torch.linalg.norm(_so3_log_torch(delta[:3, :3]))))
    translation = float(torch.linalg.vector_norm(delta[:3, 3]))
    return rotation, translation


def _weighted_rmse(residual: torch.Tensor, weights: torch.Tensor) -> float:
    if not residual.numel() or float(weights.sum()) <= 0.0:
        return float("inf")
    return float(torch.sqrt((weights * residual * residual).sum() / weights.sum()))


def _rank_deficient(points: torch.Tensor) -> bool:
    centered = points - points.mean(dim=0, keepdim=True)
    return int(torch.linalg.matrix_rank(centered)) < 2


def _apply_transform(transform: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    return points @ transform[:3, :3].transpose(0, 1) + transform[:3, 3]


def _require_pose(frame: GeometryFrame) -> torch.Tensor:
    if frame.camera_to_world is None:
        raise ValueError(
            f"Geometry frame {frame.frame_id} has no camera_to_world pose."
        )
    return _as_pose_tensor(frame.camera_to_world)


def _as_pose_tensor(pose: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(pose).detach().float().cpu()
    if tuple(value.shape) == (3, 4):
        output = torch.eye(4, dtype=value.dtype)
        output[:3] = value
        return output
    if tuple(value.shape) == (4, 4):
        return value
    raise ValueError("Pose must have shape [3,4] or [4,4].")


def _invert_pose(pose: torch.Tensor) -> torch.Tensor:
    value = _as_pose_tensor(pose)
    rotation = value[:3, :3]
    output = torch.eye(4, dtype=value.dtype)
    output[:3, :3] = rotation.transpose(0, 1)
    output[:3, 3] = -rotation.transpose(0, 1) @ value[:3, 3]
    return output


def _as_pose_numpy(pose: torch.Tensor) -> np.ndarray:
    return _as_pose_tensor(pose).numpy().astype(np.float64, copy=True)


def _se3_exp(value: np.ndarray) -> np.ndarray:
    rotation = _so3_exp(value[:3])
    output = np.eye(4, dtype=np.float64)
    output[:3, :3] = rotation
    output[:3, 3] = value[3:6]
    return output


def _so3_exp(vector: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(vector))
    skew = np.array(
        [[0.0, -vector[2], vector[1]], [vector[2], 0.0, -vector[0]], [-vector[1], vector[0], 0.0]],
        dtype=np.float64,
    )
    if theta < 1e-8:
        return np.eye(3) + skew
    return np.eye(3) + (math.sin(theta) / theta) * skew + (
        (1.0 - math.cos(theta)) / (theta * theta)
    ) * (skew @ skew)


def _so3_log(matrix: np.ndarray) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    cosine = float(np.clip((np.trace(value) - 1.0) * 0.5, -1.0, 1.0))
    theta = math.acos(cosine)
    if theta < 1e-8:
        return np.array(
            [value[2, 1] - value[1, 2], value[0, 2] - value[2, 0], value[1, 0] - value[0, 1]],
            dtype=np.float64,
        ) * 0.5
    scale = theta / (2.0 * math.sin(theta))
    return scale * np.array(
        [value[2, 1] - value[1, 2], value[0, 2] - value[2, 0], value[1, 0] - value[0, 1]],
        dtype=np.float64,
    )


def _so3_log_torch(matrix: torch.Tensor) -> torch.Tensor:
    return torch.from_numpy(_so3_log(matrix.detach().float().cpu().numpy())).float()


def _load_image(path: str | Path, image_size: SizeHW) -> torch.Tensor:
    height, width = map(int, image_size)
    with Image.open(path) as image:
        resized = image.convert("RGB").resize(
            (width, height), resample=Image.Resampling.BILINEAR
        )
        array = np.asarray(resized, dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).clamp(0.0, 1.0)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_trajectory(
    path: Path,
    frame_ids: Sequence[int],
    poses: Sequence[torch.Tensor],
) -> None:
    lines = ["# frame_id followed by row-major camera_to_world 4x4"]
    for frame_id, pose in zip(frame_ids, poses):
        values = " ".join(f"{float(item):.9g}" for item in _as_pose_tensor(pose).flatten())
        lines.append(f"{int(frame_id)} {values}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class _EdgeRejection(RuntimeError):
    pass


__all__ = [
    "DinoV3PatchFeatureMatcher",
    "FeatureMatch",
    "ObjectCorrespondence",
    "ObjectFeatureMatcher",
    "ObjectFramePair",
    "ObjectObservationRecord",
    "ObjectPoseEdge",
    "ObjectPoseRefinementConfig",
    "ObjectPoseRefiner",
    "PoseRefinementResult",
    "RGBPatchFeatureMatcher",
    "RigidRegistration",
    "apply_refined_camera_poses",
    "create_object_feature_matcher",
    "estimate_rigid_transform_ransac",
    "weighted_rigid_transform",
    "write_pose_refinement_debug",
]
