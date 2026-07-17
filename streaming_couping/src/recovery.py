"""Geometry-guided recovery candidates and mask evaluation metrics."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from test_sam.coordinates import streamvggt_label_to_grid

from .aggregation.mine_revisit_segments import mine_revisit_candidate
from .aggregation.point_map_fusion import ObjectPointMap, sample_masked_observation
from .bridge.gating import binary_iou, decide_correction
from .config import ExperimentConfig
from .types import GeometrySequence, RevisitCandidate


def mine_recovery(
    config: ExperimentConfig,
    *,
    sequence,
    target_masks: torch.Tensor,
    original_masks: torch.Tensor,
    original_scores: torch.Tensor,
    geometry: GeometrySequence,
) -> dict:
    """Find geometry-supported frames where the original tracker is weak."""

    object_map = ObjectPointMap(
        max_points_per_object=config.max_points_per_object
    )
    candidates: list[RevisitCandidate] = []
    rows: list[dict] = []

    reference = int(sequence.reference_frame_idx)
    reference_mask = output_mask_to_stream(
        target_masks[reference],
        source_size=geometry.source_sizes[reference],
        processed_size=geometry.processed_size,
        image_mode=config.image_mode,
    )
    points, weights = sample_masked_observation(
        geometry.world_points[reference],
        geometry.confidence[reference],
        reference_mask,
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
            if entry is None:
                candidate = empty_candidate(
                    config.output_size,
                    "object map is unavailable",
                )
            else:
                candidate = mine_revisit_candidate(
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
        candidates.append(candidate)
        decision = decide_correction(
            tracker_mask=original_mask,
            tracker_score=original_score,
            candidate=candidate,
            tracker_low_score=config.tracker_low_score,
            fallback_on_missing_mask=config.fallback_on_missing_mask,
        )
        rows.append(
            {
                "sequence_index": sequence_index,
                "frame_index": frame_index,
                "geometry_index": sequence_index,
                "gt_visible": int(target_masks[sequence_index].any()),
                "sam3_score": original_score,
                "sam3_iou": binary_iou(
                    original_mask,
                    target_masks[sequence_index],
                ),
                "candidate_iou": binary_iou(
                    candidate.mask,
                    target_masks[sequence_index],
                ),
                "candidate_area_ratio": float(candidate.mask.float().mean()),
                "candidate_centroid_error": centroid_error(
                    candidate.mask,
                    target_masks[sequence_index],
                ),
                "projected_points": candidate.projected_points,
                "supported_points": candidate.supported_points,
                "projected_fraction": candidate.projected_fraction,
                "support_ratio": candidate.support_ratio,
                "candidate_accepted": int(candidate.accepted),
                "use_correction": int(decision.use_correction),
                "gate_reason": decision.reason,
            }
        )
    return {"rows": rows, "candidates": candidates}


def summarize_masks(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    reference_frame_idx: int,
) -> dict[str, float]:
    prediction = prediction.detach().cpu()
    target = target.detach().cpu()
    ious = torch.tensor(
        [binary_iou(pred, gt) for pred, gt in zip(prediction, target)],
        dtype=torch.float32,
    )
    visible = target.flatten(1).any(dim=1)
    cross_view = visible.clone()
    cross_view[int(reference_frame_idx)] = False
    absent = ~visible
    return {
        "mean_iou": float(ious.mean()),
        "positive_iou": float(ious[visible].mean()) if visible.any() else 0.0,
        "cross_view_iou": (
            float(ious[cross_view].mean()) if cross_view.any() else 0.0
        ),
        "cross_view_recall": (
            float((ious[cross_view] >= 0.5).float().mean())
            if cross_view.any()
            else 0.0
        ),
        "absent_fp_ratio": (
            float(prediction[absent].float().mean()) if absent.any() else 0.0
        ),
    }


def summarize_visible_after(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    recovery_frame_idx: int | None,
) -> dict[str, float | int]:
    """Measure future visible frames without counting the recovery frame."""

    if recovery_frame_idx is None:
        return {"iou": 0.0, "recall": 0.0, "visible_frames": 0}
    visible = target.flatten(1).any(dim=1)
    selected = visible & (
        torch.arange(len(target)) > int(recovery_frame_idx)
    )
    if not selected.any():
        return {"iou": 0.0, "recall": 0.0, "visible_frames": 0}
    ious = torch.tensor(
        [
            binary_iou(prediction[index], target[index])
            for index in selected.nonzero(as_tuple=False).flatten().tolist()
        ],
        dtype=torch.float32,
    )
    return {
        "iou": float(ious.mean()),
        "recall": float((ious >= 0.5).float().mean()),
        "visible_frames": int(selected.sum()),
    }


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


def centroid_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> float:
    if not prediction.any() or not target.any():
        return float("nan")
    pred_y, pred_x = prediction.nonzero(as_tuple=True)
    target_y, target_x = target.nonzero(as_tuple=True)
    height, width = prediction.shape
    dx = (
        pred_x.float().mean() - target_x.float().mean()
    ) / max(width, 1)
    dy = (
        pred_y.float().mean() - target_y.float().mean()
    ) / max(height, 1)
    return float(torch.sqrt(dx * dx + dy * dy))
