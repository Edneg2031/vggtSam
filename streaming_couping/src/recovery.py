"""Minimal geometry-gated recovery mining and mask coordinate conversion."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from test_sam.coordinates import streamvggt_label_to_grid

from .aggregation.mine_revisit_segments import mine_revisit_candidate
from .aggregation.point_map_fusion import ObjectPointMap, sample_masked_observation
from .bridge.gating import decide_correction
from .config import ExperimentConfig
from .types import GeometrySequence, RevisitCandidate


def mine_recovery(
    config: ExperimentConfig,
    *,
    sequence,
    reference_mask: torch.Tensor,
    original_masks: torch.Tensor,
    original_scores: torch.Tensor,
    geometry: GeometrySequence,
    map_update_policy: str = "reference_only",
) -> dict:
    """Mine causal geometry events without consulting post-reference GT."""

    if map_update_policy not in {"reference_only", "joint_reliable"}:
        raise ValueError(
            "map_update_policy must be 'reference_only' or 'joint_reliable'."
        )
    object_map = ObjectPointMap(
        max_points_per_object=config.max_points_per_object
    )
    candidates: list[RevisitCandidate] = []
    rows: list[dict] = []
    reference = int(sequence.reference_frame_idx)
    reference_grid_mask = output_mask_to_stream(
        reference_mask,
        source_size=geometry.source_sizes[reference],
        processed_size=geometry.processed_size,
        image_mode=config.image_mode,
    )
    points, weights = sample_masked_observation(
        geometry.world_points[reference],
        geometry.confidence[reference],
        reference_grid_mask,
        max_points=config.max_points_per_observation,
    )
    object_map.update(
        instance_id=sequence.instance_id,
        label=sequence.label,
        points=points,
        weights=weights,
        frame_idx=reference,
    )

    for sequence_index, frame_index in enumerate(sequence.frame_indices):
        original_mask = original_masks[sequence_index]
        original_score = float(original_scores[sequence_index])
        if sequence_index == reference:
            candidate = empty_candidate(config.output_size, "reference frame")
        else:
            entry = object_map.get(sequence.instance_id)
            candidate = (
                empty_candidate(
                    config.output_size,
                    "object map is unavailable",
                )
                if entry is None
                else mine_revisit_candidate(
                    entry.points,
                    current_world_points=geometry.world_points[sequence_index],
                    world_to_camera=geometry.world_to_camera[sequence_index],
                    intrinsics=geometry.intrinsics[sequence_index],
                    source_size=geometry.source_sizes[sequence_index],
                    processed_size=geometry.processed_size,
                    output_size=config.output_size,
                    image_mode=config.image_mode,
                    box_quantile=config.box_quantile,
                    box_padding_ratio=config.box_padding_ratio,
                    min_projected_points=config.min_projected_points,
                    min_projected_fraction=config.min_projected_fraction,
                    min_supported_points=config.min_supported_points,
                    min_support_ratio=config.min_support_ratio,
                    support_abs_distance=config.support_abs_distance,
                    support_relative_distance=config.support_relative_distance,
                )
            )
        candidates.append(candidate)
        tracker_geometry_coverage = _coverage(
            candidate.supported_mask,
            original_mask,
        )
        decision = decide_correction(
            tracker_mask=original_mask,
            tracker_score=original_score,
            candidate=candidate,
            tracker_low_score=config.tracker_low_score,
            fallback_on_missing_mask=config.fallback_on_missing_mask,
            tracker_geometry_coverage=tracker_geometry_coverage,
            tracker_min_geometry_coverage=(
                config.tracker_min_geometry_coverage
            ),
            fallback_on_geometry_disagreement=(
                config.fallback_on_geometry_disagreement
            ),
        )
        update_map = (
            map_update_policy == "joint_reliable"
            and sequence_index != reference
            and bool(original_mask.any())
            and original_score >= config.map_update_min_score
            and candidate.accepted
            and tracker_geometry_coverage
            >= config.map_update_min_geometry_coverage
            and not decision.use_correction
        )
        if update_map:
            update_grid_mask = output_mask_to_stream(
                original_mask,
                source_size=geometry.source_sizes[sequence_index],
                processed_size=geometry.processed_size,
                image_mode=config.image_mode,
            )
            update_points, update_weights = sample_masked_observation(
                geometry.world_points[sequence_index],
                geometry.confidence[sequence_index],
                update_grid_mask,
                max_points=config.max_points_per_observation,
            )
            object_map.update(
                instance_id=sequence.instance_id,
                label=sequence.label,
                points=update_points,
                weights=update_weights,
                frame_idx=sequence_index,
            )
        entry_after = object_map.get(sequence.instance_id)
        rows.append(
            {
                "sequence_index": sequence_index,
                "frame_index": int(frame_index),
                "sam3_score": original_score,
                "tracker_geometry_coverage": tracker_geometry_coverage,
                "candidate_accepted": int(candidate.accepted),
                "use_correction": int(decision.use_correction),
                "map_updated": int(update_map),
                "map_observations": (
                    entry_after.observations
                    if entry_after is not None
                    else 0
                ),
                "map_points": (
                    int(entry_after.points.shape[0])
                    if entry_after is not None
                    else 0
                ),
                "gate_reason": decision.reason,
            }
        )
    return {"rows": rows, "candidates": candidates}


def resize_target_masks(
    masks: list[np.ndarray],
    output_size: tuple[int, int],
) -> torch.Tensor:
    tensor = torch.from_numpy(np.stack(masks)).float()[:, None]
    return F.interpolate(
        tensor,
        size=output_size,
        mode="nearest",
    )[:, 0].bool()


def output_mask_to_stream(
    mask: torch.Tensor,
    *,
    source_size: tuple[int, int],
    processed_size: tuple[int, int],
    image_mode: str,
) -> torch.Tensor:
    source = F.interpolate(
        mask.float()[None, None],
        size=source_size,
        mode="nearest",
    )[0, 0]
    labels = streamvggt_label_to_grid(
        source.cpu().numpy().astype(np.uint8),
        processed_size,
        mode=image_mode,
    )
    return torch.from_numpy(labels > 0)


def empty_candidate(
    output_size: tuple[int, int],
    reason: str,
) -> RevisitCandidate:
    empty = torch.zeros(output_size, dtype=torch.bool)
    return RevisitCandidate(
        mask=empty,
        projected_mask=empty.clone(),
        supported_mask=empty.clone(),
        box_xyxy=None,
        projected_points=0,
        supported_points=0,
        projected_fraction=0.0,
        support_ratio=0.0,
        accepted=False,
        reason=reason,
    )


def _coverage(evidence: torch.Tensor, mask: torch.Tensor) -> float:
    evidence = evidence.detach().cpu().bool()
    mask = mask.detach().cpu().bool()
    denominator = int(evidence.sum())
    if denominator == 0:
        return 0.0
    return float((evidence & mask).sum()) / denominator
