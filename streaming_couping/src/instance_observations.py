"""Instance tracking/cache utilities used by the final joint method.

This module contains only the persistent-instance observation machinery needed
by the learned pointmap/pose path.  The retired standalone pose-refinement
experiment previously owned these helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from test_sam.data import load_mask_tracking_sequence

from .config import ExperimentConfig
from .recovery import output_mask_to_stream, resize_target_masks
from .types import GeometrySequence, TrackingSequence


@dataclass(frozen=True)
class InstanceRefinementConfig:
    min_instance_points: int = 128
    icp_max_points: int = 1024
    map_max_points: int = 4096
    icp_iterations: int = 4
    icp_trim_quantile: float = 0.70
    min_icp_fitness: float = 0.25
    max_icp_rmse: float = 0.03
    correspondence_min_distance: float = 0.02
    correspondence_object_ratio: float = 0.05
    max_proposal_translation: float = 0.15
    min_participating_instances: int = 2
    consensus_distance: float = 0.02
    compute_device: str = "cpu"


@dataclass(frozen=True)
class TranslationProposal:
    instance_id: int
    translation: torch.Tensor
    accepted: bool
    reason: str
    current_points: int
    map_points: int
    correspondences: int
    fitness: float
    rmse: float
    correspondence_distance: float
    object_scale: float
    iterations: int
    initialization: str


def load_instance_sequences(
    config: ExperimentConfig,
    *,
    instance_ids: Sequence[int],
    reference_sequence_index: int,
) -> tuple[dict[int, object], dict[int, torch.Tensor]]:
    """Load the common RGB sequence and per-instance reference masks."""

    sequences = {}
    target_masks = {}
    for instance_id in instance_ids:
        sequence = load_mask_tracking_sequence(
            config.manifest,
            scene_id=config.scene_id,
            frame_indices=config.frame_indices,
            sequence_length=len(config.frame_indices),
            frame_stride=1,
            window_index=0,
            instance_id=int(instance_id),
            min_pixels=config.min_pixels,
            max_area_ratio=config.max_area_ratio,
            min_visible_frames=1,
            excluded_labels=config.excluded_labels,
            seed=0,
        )
        resized = resize_target_masks(sequence.target_masks, config.output_size)
        reference_index = int(reference_sequence_index)
        if not 0 <= reference_index < len(sequence.frame_indices):
            raise ValueError("reference_sequence_index is outside the sequence.")
        if not resized[reference_index].any():
            raise ValueError(
                f"Instance {instance_id} is absent at reference frame "
                f"{sequence.frame_indices[reference_index]}."
            )
        sequence = replace(sequence, reference_frame_idx=reference_index)
        sequences[int(instance_id)] = sequence
        target_masks[int(instance_id)] = resized

    first = sequences[int(instance_ids[0])]
    for instance_id in instance_ids[1:]:
        current = sequences[int(instance_id)]
        if (
            tuple(current.frame_indices) != tuple(first.frame_indices)
            or [str(path) for path in current.image_paths]
            != [str(path) for path in first.image_paths]
        ):
            raise RuntimeError("All instances must share the same RGB sequence.")
    return sequences, target_masks


def translation_icp(
    current: torch.Tensor,
    object_map: torch.Tensor,
    *,
    instance_id: int,
    config: InstanceRefinementConfig,
) -> TranslationProposal:
    """Estimate a bounded translation-only ICP proposal for one instance."""

    device = torch.device(config.compute_device)
    current = deterministic_limit(current.float(), config.icp_max_points).to(device)
    object_map = deterministic_limit(object_map.float(), config.map_max_points).to(
        device
    )
    if current.shape[0] < config.min_instance_points:
        return _rejected_proposal(
            instance_id,
            "insufficient current instance points",
            current_points=int(current.shape[0]),
            map_points=int(object_map.shape[0]),
        )
    if object_map.shape[0] < config.min_instance_points:
        return _rejected_proposal(
            instance_id,
            "insufficient persistent map points",
            current_points=int(current.shape[0]),
            map_points=int(object_map.shape[0]),
        )

    low = torch.quantile(object_map, 0.05, dim=0)
    high = torch.quantile(object_map, 0.95, dim=0)
    object_scale = float(torch.linalg.vector_norm(high - low))
    max_distance = max(
        float(config.correspondence_min_distance),
        float(config.correspondence_object_ratio) * object_scale,
    )
    zero = torch.zeros(3, dtype=torch.float32, device=device)
    initializations = [("zero", zero)]
    centroid = torch.quantile(object_map, 0.50, dim=0) - torch.quantile(
        current, 0.50, dim=0
    )
    if float(torch.linalg.vector_norm(centroid)) <= float(
        config.max_proposal_translation
    ):
        initializations.append(("robust_centroid", centroid))
    fits = [
        _translation_icp_from_start(
            current,
            object_map,
            initialization=name,
            initial_translation=initial,
            max_distance=max_distance,
            config=config,
        )
        for name, initial in initializations
    ]
    bounded = [
        fit
        for fit in fits
        if float(torch.linalg.vector_norm(fit["translation"]))
        <= float(config.max_proposal_translation)
    ]
    best = max(
        bounded if bounded else fits,
        key=lambda fit: (
            fit["fitness"],
            -fit["rmse"] if math.isfinite(fit["rmse"]) else float("-inf"),
        ),
    )
    translation = best["translation"]
    correspondences = int(best["correspondences"])
    fitness = float(best["fitness"])
    rmse = float(best["rmse"])
    translation_norm = float(torch.linalg.vector_norm(translation))
    accepted = (
        correspondences >= config.min_instance_points
        and fitness >= config.min_icp_fitness
        and rmse <= config.max_icp_rmse
        and translation_norm <= config.max_proposal_translation
        and math.isfinite(rmse)
    )
    reasons = []
    if correspondences < config.min_instance_points:
        reasons.append("too few correspondences")
    if fitness < config.min_icp_fitness:
        reasons.append("low fitness")
    if rmse > config.max_icp_rmse:
        reasons.append("high rmse")
    if translation_norm > config.max_proposal_translation:
        reasons.append("translation too large")
    if not math.isfinite(rmse):
        reasons.append("non-finite rmse")
    return TranslationProposal(
        instance_id=instance_id,
        translation=translation.double().cpu(),
        accepted=accepted,
        reason="accepted" if accepted else "; ".join(reasons),
        current_points=int(current.shape[0]),
        map_points=int(object_map.shape[0]),
        correspondences=correspondences,
        fitness=fitness,
        rmse=rmse,
        correspondence_distance=max_distance,
        object_scale=object_scale,
        iterations=int(best["iterations"]),
        initialization=str(best["initialization"]),
    )


def _rejected_proposal(
    instance_id: int,
    reason: str,
    *,
    current_points: int,
    map_points: int,
) -> TranslationProposal:
    return TranslationProposal(
        instance_id=instance_id,
        translation=torch.zeros(3, dtype=torch.float64),
        accepted=False,
        reason=reason,
        current_points=current_points,
        map_points=map_points,
        correspondences=0,
        fitness=0.0,
        rmse=float("nan"),
        correspondence_distance=float("nan"),
        object_scale=float("nan"),
        iterations=0,
        initialization="none",
    )


def _translation_icp_from_start(
    current: torch.Tensor,
    object_map: torch.Tensor,
    *,
    initialization: str,
    initial_translation: torch.Tensor,
    max_distance: float,
    config: InstanceRefinementConfig,
) -> dict:
    translation = initial_translation.clone()
    iterations = 0
    for _ in range(config.icp_iterations):
        shifted = current + translation
        distances = torch.cdist(shifted, object_map)
        nearest_distance, nearest_index = distances.min(dim=1)
        cutoff = min(
            max_distance,
            float(torch.quantile(nearest_distance, config.icp_trim_quantile)),
        )
        keep = nearest_distance <= cutoff
        if int(keep.sum()) < config.min_instance_points:
            break
        matched = object_map.index_select(0, nearest_index[keep])
        step = torch.quantile(matched - shifted[keep], 0.50, dim=0)
        translation += step
        iterations += 1
        if float(torch.linalg.vector_norm(step)) <= 1e-4:
            break

    shifted = current + translation
    nearest_distance = torch.cdist(shifted, object_map).min(dim=1).values
    inliers = nearest_distance <= max_distance
    correspondences = int(inliers.sum())
    fitness = float(inliers.float().mean())
    rmse = (
        float(torch.sqrt(nearest_distance[inliers].square().mean()))
        if correspondences
        else float("nan")
    )
    return {
        "initialization": initialization,
        "translation": translation,
        "iterations": iterations,
        "correspondences": correspondences,
        "fitness": fitness,
        "rmse": rmse,
    }


def proposal_consensus(
    proposals: Sequence[TranslationProposal],
    *,
    min_instances: int,
    max_distance: float,
) -> tuple[torch.Tensor | None, tuple[int, ...], float]:
    """Find a robust shared translation across accepted static instances."""

    cluster = [proposal for proposal in proposals if proposal.accepted]
    if len(cluster) < int(min_instances):
        return None, (), float("nan")
    disagreement = float("nan")
    while len(cluster) >= int(min_instances):
        translations = torch.stack([proposal.translation for proposal in cluster])
        shared = torch.quantile(translations, 0.50, dim=0)
        residuals = torch.linalg.vector_norm(translations - shared, dim=1)
        disagreement = float(residuals.max())
        if disagreement <= float(max_distance):
            participating = tuple(
                sorted(proposal.instance_id for proposal in cluster)
            )
            return shared, participating, disagreement
        cluster.pop(int(torch.argmax(residuals)))
    return None, (), disagreement


def merge_map_points(
    previous: torch.Tensor,
    current: torch.Tensor,
    *,
    max_points: int,
) -> torch.Tensor:
    return deterministic_limit(torch.cat([previous, current], dim=0), max_points)


def deterministic_limit(values: torch.Tensor, limit: int) -> torch.Tensor:
    if values.shape[0] <= int(limit):
        return values
    positions = torch.linspace(
        0,
        values.shape[0] - 1,
        steps=int(limit),
        dtype=torch.float64,
    ).round().long()
    return values.index_select(0, positions)


def tracking_masks_to_geometry_grid(
    trackings: Mapping[int, TrackingSequence],
    *,
    geometry: GeometrySequence,
    image_mode: str,
) -> dict[int, torch.Tensor]:
    return {
        int(instance_id): torch.stack(
            [
                output_mask_to_stream(
                    mask,
                    source_size=geometry.source_sizes[index],
                    processed_size=geometry.processed_size,
                    image_mode=image_mode,
                )
                for index, mask in enumerate(tracking.masks)
            ]
        ).bool()
        for instance_id, tracking in trackings.items()
    }


def save_tracking_cache(
    path: Path,
    *,
    config: ExperimentConfig,
    instance_ids: Sequence[int],
    frame_indices: Sequence[int],
    original: Mapping[int, TrackingSequence],
    recovered: Mapping[int, TrackingSequence],
    tracking_rows: Sequence[dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "cache_version": np.asarray([2], dtype=np.int64),
        "instance_ids": np.asarray(instance_ids, dtype=np.int64),
        "frame_indices": np.asarray(frame_indices, dtype=np.int64),
        "output_size": np.asarray(config.output_size, dtype=np.int64),
        "tracking_signature": np.asarray([_tracking_cache_signature(config)]),
        "tracking_rows_json": np.asarray(
            [json.dumps(list(tracking_rows), ensure_ascii=False, allow_nan=True)]
        ),
    }
    for instance_id in instance_ids:
        for name, values in (
            ("original", original[int(instance_id)]),
            ("recovered", recovered[int(instance_id)]),
        ):
            _validate_cached_tracking_shape(
                values,
                frame_count=len(frame_indices),
                output_size=config.output_size,
                name=f"{name}_{int(instance_id)}",
            )
            prefix = f"{name}_{int(instance_id)}"
            arrays[f"{prefix}_masks"] = values.masks.cpu().numpy().astype(np.uint8)
            arrays[f"{prefix}_scores"] = (
                values.scores.reshape(-1).cpu().numpy().astype(np.float32)
            )
            arrays[f"{prefix}_obj_id"] = np.asarray(
                [
                    -1
                    if values.selected_obj_id is None
                    else int(values.selected_obj_id)
                ],
                dtype=np.int64,
            )
    np.savez_compressed(path, **arrays)


def load_tracking_cache(
    path: Path,
    *,
    config: ExperimentConfig,
    instance_ids: Sequence[int],
    frame_indices: Sequence[int],
) -> tuple[
    dict[int, TrackingSequence],
    dict[int, TrackingSequence],
    list[dict],
] | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as values:
            if (
                values["cache_version"].tolist() != [2]
                or values["instance_ids"].tolist()
                != [int(value) for value in instance_ids]
                or values["frame_indices"].tolist()
                != [int(value) for value in frame_indices]
                or values["output_size"].tolist()
                != [int(value) for value in config.output_size]
                or str(values["tracking_signature"][0])
                != _tracking_cache_signature(config)
            ):
                return None
            output = {}
            for name in ("original", "recovered"):
                trackings = {}
                for instance_id in instance_ids:
                    prefix = f"{name}_{int(instance_id)}"
                    selected = int(values[f"{prefix}_obj_id"][0])
                    tracking = TrackingSequence(
                        masks=torch.from_numpy(
                            values[f"{prefix}_masks"].copy()
                        ).bool(),
                        scores=torch.from_numpy(
                            values[f"{prefix}_scores"].copy()
                        ).float(),
                        selected_obj_id=None if selected < 0 else selected,
                    )
                    _validate_cached_tracking_shape(
                        tracking,
                        frame_count=len(frame_indices),
                        output_size=config.output_size,
                        name=prefix,
                    )
                    trackings[int(instance_id)] = tracking
                output[name] = trackings
            tracking_rows = json.loads(str(values["tracking_rows_json"][0]))
            if (
                not isinstance(tracking_rows, list)
                or len(tracking_rows) != len(instance_ids)
                or not all(isinstance(row, dict) for row in tracking_rows)
                or [int(row.get("instance_id", -1)) for row in tracking_rows]
                != [int(value) for value in instance_ids]
            ):
                return None
            return output["original"], output["recovered"], tracking_rows
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _tracking_cache_signature(config: ExperimentConfig) -> str:
    excluded = {"output_dir", "sam3_device", "geometry_device"}
    values = {
        key: _json_cache_value(value)
        for key, value in config.__dict__.items()
        if key not in excluded
    }
    return json.dumps(
        values,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _json_cache_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_cache_value(item) for item in value]
    return value


def _validate_cached_tracking_shape(
    tracking: TrackingSequence,
    *,
    frame_count: int,
    output_size: Sequence[int],
    name: str,
) -> None:
    expected_masks = (
        int(frame_count),
        int(output_size[0]),
        int(output_size[1]),
    )
    if tuple(tracking.masks.shape) != expected_masks:
        raise ValueError(
            f"Tracking cache {name} masks have shape "
            f"{tuple(tracking.masks.shape)}, expected {expected_masks}."
        )
    if tracking.scores.numel() != int(frame_count):
        raise ValueError(
            f"Tracking cache {name} scores contain "
            f"{tracking.scores.numel()} values, expected {frame_count}."
        )
