"""Offline metrics for exported semantic-map artifacts.

The runtime mapper intentionally never opens annotations.  This module is the
separate evaluation boundary: it takes an exported ``semantic_map.pt`` and a
GT pointmap/mask sequence, aligns the predicted world frame once, and scores
the resulting fused instances.  A shared alignment is supplied to every
branch so raw/candidate comparisons do not get a branch-specific advantage.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from typing import Any

import torch

from ..pointmap_alignment import robust_similarity
from ..semantic_map_metrics import (
    SemanticMapMetricConfig,
    apply_similarity,
    deterministic_limit_points,
    object_point_metrics,
    set_iou,
    voxel_keys,
)
from ..semantic_tracking_metrics import maximum_weight_assignment, prompt_matches_label


@dataclass(frozen=True)
class ExportedMapMetricConfig:
    """Limits and thresholds for evaluating a fused export."""

    object_metrics: SemanticMapMetricConfig = field(
        default_factory=SemanticMapMetricConfig
    )
    max_points_per_scene: int = 20_000
    assignment_min_voxel_iou: float = 0.01

    def validate(self) -> "ExportedMapMetricConfig":
        if int(self.max_points_per_scene) < 1:
            raise ValueError("max_points_per_scene must be positive.")
        if not 0.0 <= float(self.assignment_min_voxel_iou) <= 1.0:
            raise ValueError("assignment_min_voxel_iou must be in [0,1].")
        if self.object_metrics.max_points_per_object < 1:
            raise ValueError("max_points_per_object must be positive.")
        return self


@dataclass(frozen=True)
class SimilarityAlignment:
    """One fixed predicted-world to GT-world similarity transform."""

    scale: float
    rotation: torch.Tensor
    translation: torch.Tensor
    reference_frame_index: int
    fit_inliers: int
    fit_rmse_m: float

    def apply(self, points: torch.Tensor) -> torch.Tensor:
        return apply_similarity(
            points,
            scale=float(self.scale),
            rotation=self.rotation,
            translation=self.translation,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "scale": float(self.scale),
            "rotation": self.rotation.detach().float().cpu().tolist(),
            "translation": self.translation.detach().float().cpu().tolist(),
            "reference_frame_index": int(self.reference_frame_index),
            "fit_inliers": int(self.fit_inliers),
            "fit_rmse_m": float(self.fit_rmse_m),
        }


def fit_reference_alignment(
    predicted_world_points: torch.Tensor,
    target_world_points: torch.Tensor,
    confidence: torch.Tensor,
    *,
    reference_frame_index: int = 0,
    confidence_threshold: float = 0.30,
    max_points: int = 30_000,
    min_points: int = 128,
) -> SimilarityAlignment:
    """Fit one trimmed Sim(3) on a reference frame for evaluation only."""

    predicted = _as_point_sequence(predicted_world_points, "predicted_world_points")
    target = _as_point_sequence(target_world_points, "target_world_points")
    confidence = torch.as_tensor(confidence).detach().float().cpu()
    if tuple(predicted.shape) != tuple(target.shape):
        raise ValueError("Predicted and target pointmaps must have equal shapes.")
    if tuple(confidence.shape) != tuple(predicted.shape[:3]):
        raise ValueError("Confidence does not match the pointmap sequence.")
    frame = int(reference_frame_index)
    if frame < 0 or frame >= predicted.shape[0]:
        raise ValueError(
            f"reference_frame_index={frame} is outside [0,{predicted.shape[0] - 1}]."
        )
    source = predicted[frame].reshape(-1, 3)
    destination = target[frame].reshape(-1, 3)
    valid = (
        torch.isfinite(source).all(dim=1)
        & torch.isfinite(destination).all(dim=1)
        & torch.isfinite(confidence[frame].reshape(-1))
        & (confidence[frame].reshape(-1) >= float(confidence_threshold))
    )
    source = source[valid]
    destination = destination[valid]
    source, destination = _paired_limit(source, destination, max_points=max_points)
    if source.shape[0] < int(min_points):
        raise ValueError(
            f"Reference alignment needs {int(min_points)} valid paired points, "
            f"got {int(source.shape[0])}."
        )
    scale, rotation, translation, inliers, rmse = robust_similarity(
        source,
        destination,
        min_points=int(min_points),
    )
    return SimilarityAlignment(
        scale=float(scale),
        rotation=rotation.detach().float().cpu(),
        translation=translation.detach().float().cpu(),
        reference_frame_index=frame,
        fit_inliers=int(inliers),
        fit_rmse_m=float(rmse),
    )


def evaluate_pointmap_alignment(
    predicted_world_points: torch.Tensor,
    target_world_points: torch.Tensor,
    confidence: torch.Tensor,
    *,
    alignment: SimilarityAlignment,
    confidence_threshold: float = 0.30,
    frame_ids: Sequence[int] | None = None,
) -> dict[str, object]:
    """Report dense pointmap residuals after one fixed alignment."""

    predicted = _as_point_sequence(predicted_world_points, "predicted_world_points")
    target = _as_point_sequence(target_world_points, "target_world_points")
    confidence = torch.as_tensor(confidence).detach().float().cpu()
    if tuple(predicted.shape) != tuple(target.shape):
        raise ValueError("Predicted and target pointmaps must have equal shapes.")
    if tuple(confidence.shape) != tuple(predicted.shape[:3]):
        raise ValueError("Confidence does not match the pointmap sequence.")
    aligned = alignment.apply(predicted)
    if frame_ids is None:
        frame_ids = tuple(range(predicted.shape[0]))
    if len(frame_ids) != predicted.shape[0]:
        raise ValueError("frame_ids does not match pointmap sequence length.")

    frame_rows: list[dict[str, object]] = []
    all_residuals: list[torch.Tensor] = []
    for index in range(predicted.shape[0]):
        source = aligned[index]
        destination = target[index]
        valid = (
            torch.isfinite(source).all(dim=-1)
            & torch.isfinite(destination).all(dim=-1)
            & torch.isfinite(confidence[index])
            & (confidence[index] >= float(confidence_threshold))
        )
        residual = torch.linalg.vector_norm(source - destination, dim=-1)[valid]
        if residual.numel():
            all_residuals.append(residual)
            row = _residual_summary(residual)
        else:
            row = {
                "valid_points": 0,
                "rmse_m": float("nan"),
                "median_m": float("nan"),
                "p90_m": float("nan"),
            }
        row.update(
            {
                "sequence_index": int(index),
                "frame_id": int(frame_ids[index]),
            }
        )
        frame_rows.append(row)
    combined = torch.cat(all_residuals) if all_residuals else torch.empty(0)
    summary = _residual_summary(combined)
    summary.update(
        {
            "status": "ok" if combined.numel() else "no_valid_pairs",
            "reference_frame_index": int(alignment.reference_frame_index),
            "fit_inliers": int(alignment.fit_inliers),
            "fit_rmse_m": float(alignment.fit_rmse_m),
            "scale": float(alignment.scale),
        }
    )
    return {"summary": summary, "frames": frame_rows}


def evaluate_exported_semantic_map(
    payload: Mapping[str, Any],
    *,
    scene_id: str,
    clip_name: str,
    variant: str,
    target_world_points: torch.Tensor,
    gt_masks: torch.Tensor,
    gt_instance_ids: Sequence[int],
    gt_labels: Sequence[str],
    alignment: SimilarityAlignment | None = None,
    map_source: str = "semantic",
    prompt_label_aliases: Mapping[str, Sequence[str]] | None = None,
    config: ExportedMapMetricConfig = ExportedMapMetricConfig(),
) -> dict[str, object]:
    """Score one exported branch against a frozen GT object point set.

    ``map_source='semantic'`` evaluates the policy-dependent voxel map.  The
    optional ``tracks`` source is useful as a diagnostic because it evaluates
    the raw, non-voxel-fused observation cloud.  In both cases the assignment
    is evaluation-only and does not alter the artifact.
    """

    config.validate()
    target = _as_point_sequence(target_world_points, "target_world_points")
    masks = torch.as_tensor(gt_masks).detach().bool().cpu()
    if masks.ndim != 4 or tuple(masks.shape[0:1] + masks.shape[2:]) != tuple(
        target.shape[:3]
    ):
        raise ValueError("GT masks and target pointmaps do not share [S,H,W].")
    if masks.shape[1] != len(gt_instance_ids) or len(gt_labels) != len(gt_instance_ids):
        raise ValueError("GT instance IDs, labels, and masks disagree.")
    source = str(map_source).strip().lower()
    if source not in {"semantic", "tracks"}:
        raise ValueError("map_source must be 'semantic' or 'tracks'.")

    predicted_objects = extract_exported_objects(payload, source=source)
    if alignment is not None:
        predicted_objects = {
            instance_id: {
                **row,
                "points": alignment.apply(row["points"]),
            }
            for instance_id, row in predicted_objects.items()
        }
    target_objects = _target_objects(
        target,
        masks,
        max_points=int(config.object_metrics.max_points_per_object),
    )
    target_voxels = {
        index: voxel_keys(
            points,
            voxel_size=float(config.object_metrics.voxel_size_m),
        )
        for index, points in target_objects.items()
    }
    predicted_ids = sorted(predicted_objects)
    predicted_voxels = {
        instance_id: voxel_keys(
            predicted_objects[instance_id]["points"],
            voxel_size=float(config.object_metrics.voxel_size_m),
        )
        for instance_id in predicted_ids
    }
    assignments = _assign_exported_objects(
        predicted_objects,
        predicted_voxels,
        gt_labels,
        target_voxels,
        aliases=prompt_label_aliases,
        min_iou=float(config.assignment_min_voxel_iou),
    )
    predicted_to_target = {instance_id: target_index for instance_id, target_index in assignments}
    target_to_predicted = {target_index: instance_id for instance_id, target_index in assignments}

    object_rows: list[dict[str, object]] = []
    for target_index, gt_instance_id in enumerate(gt_instance_ids):
        predicted_id = target_to_predicted.get(target_index)
        if predicted_id is None:
            predicted = torch.empty(0, 3)
            predicted_category = ""
        else:
            predicted = deterministic_limit_points(
                predicted_objects[predicted_id]["points"],
                int(config.object_metrics.max_points_per_object),
            )
            predicted_category = str(predicted_objects[predicted_id]["category"])
        target_points = target_objects[target_index]
        metrics = object_point_metrics(
            predicted,
            target_points,
            fscore_thresholds=config.object_metrics.fscore_thresholds_m,
            voxel_size=float(config.object_metrics.voxel_size_m),
            ghost_distance=float(config.object_metrics.ghost_distance_m),
            chunk_size=int(config.object_metrics.distance_chunk_size),
        )
        object_rows.append(
            {
                "scene_id": str(scene_id),
                "clip": str(clip_name),
                "variant": str(variant),
                "map_source": source,
                "gt_instance_id": int(gt_instance_id),
                "gt_label": str(gt_labels[target_index]),
                "predicted_instance_id": (
                    -1 if predicted_id is None else int(predicted_id)
                ),
                "predicted_category": predicted_category,
                "matched": int(predicted_id is not None),
                "predicted_points": int(predicted.shape[0]),
                "target_points": int(target_points.shape[0]),
                **metrics,
            }
        )

    duplicate_rows: list[dict[str, object]] = []
    duplicate_count = 0
    for predicted_id in predicted_ids:
        if predicted_id in predicted_to_target:
            continue
        points = deterministic_limit_points(
            predicted_objects[predicted_id]["points"],
            int(config.object_metrics.max_points_per_object),
        )
        if not points.numel():
            continue
        best_target = -1
        best_iou = 0.0
        for target_index, target_set in target_voxels.items():
            current = set_iou(predicted_voxels[predicted_id], target_set)
            if current > best_iou:
                best_iou = current
                best_target = target_index
        duplicate = bool(
            best_target >= 0
            and best_target in target_to_predicted
            and best_iou >= float(config.object_metrics.duplicate_voxel_iou)
        )
        duplicate_count += int(duplicate)
        duplicate_rows.append(
            {
                "scene_id": str(scene_id),
                "clip": str(clip_name),
                "variant": str(variant),
                "map_source": source,
                "predicted_instance_id": int(predicted_id),
                "predicted_category": str(predicted_objects[predicted_id]["category"]),
                "predicted_points": int(points.shape[0]),
                "best_gt_instance_id": (
                    -1 if best_target < 0 else int(gt_instance_ids[best_target])
                ),
                "best_voxel_iou": float(best_iou),
                "duplicate_object": int(duplicate),
            }
        )

    summary = _summarize_object_rows(
        object_rows,
        scene_id=scene_id,
        clip_name=clip_name,
        variant=variant,
        map_source=source,
        predicted_instance_count=len(predicted_ids),
        duplicate_count=duplicate_count,
        unmatched_nonempty_instances=len(duplicate_rows),
    )
    scene = _evaluate_scene_cloud(
        payload,
        alignment=alignment,
        target=target,
        config=config,
    )
    return {
        "summary": summary,
        "object_rows": object_rows,
        "duplicate_rows": duplicate_rows,
        "scene": scene,
        "assignments": [
            {
                "predicted_instance_id": int(predicted_id),
                "gt_instance_id": int(gt_instance_ids[target_index]),
                "gt_label": str(gt_labels[target_index]),
                "voxel_iou": float(
                    set_iou(predicted_voxels[predicted_id], target_voxels[target_index])
                ),
            }
            for predicted_id, target_index in assignments
        ],
    }


def extract_exported_objects(
    payload: Mapping[str, Any],
    *,
    source: str = "semantic",
) -> dict[int, dict[str, Any]]:
    """Read semantic voxels or raw object tracks from an export payload."""

    source = str(source).strip().lower()
    if source == "tracks":
        tracks = payload.get("object_tracks", ())
        if not isinstance(tracks, Sequence):
            raise ValueError("Export object_tracks must be a sequence.")
        output: dict[int, dict[str, Any]] = {}
        for row in tracks:
            if not isinstance(row, Mapping):
                raise ValueError("Export object_tracks rows must be mappings.")
            instance_id = int(row["instance_id"])
            points = _as_points(row.get("points", ()), f"track[{instance_id}].points")
            output[instance_id] = {
                "instance_id": instance_id,
                "category": str(row.get("category", "object")),
                "points": points,
            }
        return output
    if source != "semantic":
        raise ValueError("source must be 'semantic' or 'tracks'.")

    points = _as_points(payload.get("voxel_points", ()), "voxel_points")
    instance_ids = torch.as_tensor(payload.get("instance_ids", ())).long().cpu().reshape(-1)
    labels = tuple(str(value) for value in payload.get("semantic_labels", ()))
    weights = torch.as_tensor(
        payload.get("evidence_weights", torch.ones(points.shape[0]))
    ).float().cpu().reshape(-1)
    if instance_ids.shape[0] != points.shape[0] or len(labels) != points.shape[0]:
        raise ValueError("Semantic voxel fields have inconsistent lengths.")
    if weights.shape[0] != points.shape[0]:
        raise ValueError("evidence_weights does not match voxel_points.")
    track_categories = {
        int(row["instance_id"]): str(row.get("category", "object"))
        for row in payload.get("object_tracks", ())
        if isinstance(row, Mapping) and "instance_id" in row
    }
    output: dict[int, dict[str, Any]] = {}
    for instance_id in sorted(
        int(value) for value in instance_ids.unique().tolist() if int(value) >= 0
    ):
        selected = instance_ids == instance_id
        selected_points = points[selected]
        selected_weights = weights[selected]
        label_weights: dict[str, float] = {}
        for label, weight in zip(
            (labels[index] for index in selected.nonzero().flatten().tolist()),
            selected_weights.tolist(),
        ):
            label_weights[label] = label_weights.get(label, 0.0) + float(weight)
        category = track_categories.get(instance_id)
        if category is None:
            category = sorted(
                label_weights.items(), key=lambda item: (-item[1], item[0])
            )[0][0] if label_weights else "object"
        output[instance_id] = {
            "instance_id": instance_id,
            "category": category,
            "points": selected_points,
        }
    return output


def _assign_exported_objects(
    predicted_objects: Mapping[int, Mapping[str, Any]],
    predicted_voxels: Mapping[int, set[tuple[int, int, int]]],
    gt_labels: Sequence[str],
    target_voxels: Mapping[int, set[tuple[int, int, int]]],
    *,
    aliases: Mapping[str, Sequence[str]] | None,
    min_iou: float,
) -> list[tuple[int, int]]:
    predicted_ids = sorted(predicted_objects)
    scores = torch.full(
        (len(predicted_ids), len(gt_labels)),
        -1.0,
        dtype=torch.float64,
    )
    for row, predicted_id in enumerate(predicted_ids):
        category = str(predicted_objects[predicted_id]["category"])
        for column, label in enumerate(gt_labels):
            if prompt_matches_label(category, str(label), aliases=aliases):
                scores[row, column] = set_iou(
                    predicted_voxels[predicted_id], target_voxels[column]
                )
    pairs = maximum_weight_assignment(scores)
    return [
        (predicted_ids[row], column)
        for row, column in pairs
        if float(scores[row, column]) >= float(min_iou)
    ]


def _target_objects(
    target: torch.Tensor,
    masks: torch.Tensor,
    *,
    max_points: int,
) -> dict[int, torch.Tensor]:
    finite = torch.isfinite(target).all(dim=-1)
    return {
        index: deterministic_limit_points(
            target[masks[:, index] & finite],
            int(max_points),
        )
        for index in range(masks.shape[1])
    }


def _evaluate_scene_cloud(
    payload: Mapping[str, Any],
    *,
    alignment: SimilarityAlignment | None,
    target: torch.Tensor,
    config: ExportedMapMetricConfig,
) -> dict[str, object]:
    predicted = _as_points(payload.get("scene_voxel_points", ()), "scene_voxel_points")
    if alignment is not None:
        predicted = alignment.apply(predicted)
    predicted = deterministic_limit_points(predicted, int(config.max_points_per_scene))
    target = deterministic_limit_points(target.reshape(-1, 3), int(config.max_points_per_scene))
    return object_point_metrics(
        predicted,
        target,
        fscore_thresholds=config.object_metrics.fscore_thresholds_m,
        voxel_size=float(config.object_metrics.voxel_size_m),
        ghost_distance=float(config.object_metrics.ghost_distance_m),
        chunk_size=int(config.object_metrics.distance_chunk_size),
    )


def _summarize_object_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    scene_id: str,
    clip_name: str,
    variant: str,
    map_source: str,
    predicted_instance_count: int,
    duplicate_count: int,
    unmatched_nonempty_instances: int,
) -> dict[str, object]:
    def mean(name: str) -> float:
        values = [
            float(row[name])
            for row in rows
            if name in row and math.isfinite(float(row[name]))
        ]
        return sum(values) / len(values) if values else float("nan")

    return {
        "scene_id": str(scene_id),
        "clip": str(clip_name),
        "variant": str(variant),
        "map_source": str(map_source),
        "eligible_gt_objects": int(len(rows)),
        "matched_objects": int(sum(int(row["matched"]) for row in rows)),
        "predicted_instance_count": int(predicted_instance_count),
        "object_accuracy_m": mean("object_accuracy_m"),
        "object_completeness_m": mean("object_completeness_m"),
        "symmetric_distance_m": mean("symmetric_distance_m"),
        "fscore_5cm": mean("fscore_5cm"),
        "fscore_10cm": mean("fscore_10cm"),
        "ghost_point_ratio": mean("ghost_point_ratio"),
        "voxel_iou_5cm": mean("voxel_iou"),
        "duplicate_objects": int(duplicate_count),
        "unmatched_nonempty_instances": int(unmatched_nonempty_instances),
        "duplicate_object_rate": (
            duplicate_count / unmatched_nonempty_instances
            if unmatched_nonempty_instances
            else 0.0
        ),
    }


def _residual_summary(residual: torch.Tensor) -> dict[str, object]:
    residual = torch.as_tensor(residual).detach().float().cpu().reshape(-1)
    residual = residual[torch.isfinite(residual)]
    if not residual.numel():
        return {
            "valid_points": 0,
            "rmse_m": float("nan"),
            "median_m": float("nan"),
            "p90_m": float("nan"),
        }
    return {
        "valid_points": int(residual.shape[0]),
        "rmse_m": float(torch.sqrt(residual.square().mean())),
        "median_m": float(torch.quantile(residual, 0.50)),
        "p90_m": float(torch.quantile(residual, 0.90)),
    }


def _paired_limit(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    max_points: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if source.shape[0] <= int(max_points):
        return source.float().cpu(), target.float().cpu()
    indices = torch.linspace(
        0,
        source.shape[0] - 1,
        steps=int(max_points),
    ).round().long()
    return (
        source.index_select(0, indices).float().cpu(),
        target.index_select(0, indices).float().cpu(),
    )


def _as_point_sequence(value: torch.Tensor, name: str) -> torch.Tensor:
    output = torch.as_tensor(value).detach().float().cpu()
    if output.ndim != 4 or output.shape[-1] != 3:
        raise ValueError(f"{name} must have shape [S,H,W,3].")
    return output


def _as_points(value: Any, name: str) -> torch.Tensor:
    output = torch.as_tensor(value).detach().float().cpu()
    if output.numel() == 0:
        return torch.empty(0, 3, dtype=torch.float32)
    if output.ndim != 2 or output.shape[-1] != 3:
        raise ValueError(f"{name} must contain points with shape [N,3].")
    return output


__all__ = [
    "ExportedMapMetricConfig",
    "SimilarityAlignment",
    "evaluate_exported_semantic_map",
    "evaluate_pointmap_alignment",
    "extract_exported_objects",
    "fit_reference_alignment",
]
