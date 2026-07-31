"""V6 geometry-prompted SAM3.1 segmentation.

StreamVGGT geometry is converted into a box before SAM3 predicts a mask.  GT
is deliberately absent from this module; callers may use it only afterwards
for evaluation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from .aggregation.mine_revisit_segments import mine_revisit_candidate
from .aggregation.point_map_fusion import ObjectPointMap, sample_masked_observation
from .backbones.sam3_wrapper import SAM3Wrapper
from .config import ExperimentConfig
from .instance_observations import (
    InstanceRefinementConfig,
    TranslationProposal,
    translation_icp,
)
from .recovery import output_mask_to_stream
from .types import SAM3MaskCandidate, TrackingSequence

V6_SEGMENTATION_VARIANTS = (
    "raw_sam31",
    "current_sam3_late_geometry",
    "v6_sam31_geometry_box",
    "v6_sam31_geometry_points",
    "v6_sam31_geometry_box_3d",
    "v6_sam31_geometry_points_3d",
)


@dataclass(frozen=True)
class V6GeometrySegmentationConfig:
    point_confidence_threshold: float = 0.30
    min_support_recall: float = 0.25
    min_box_precision: float = 0.50
    support_dilation: int = 5
    point_positive_samples: int = 6
    point_negative_samples: int = 4
    point_negative_exclusion_radius: int = 5


@torch.no_grad()
def segment_instance_with_geometry_prompts(
    *,
    recovery: ExperimentConfig,
    sequence,
    reference_mask: torch.Tensor,
    raw_tracking: TrackingSequence,
    current_late_tracking: TrackingSequence,
    world_points: torch.Tensor,
    confidence: torch.Tensor,
    world_to_camera: torch.Tensor,
    intrinsics: torch.Tensor,
    source_sizes: Sequence[tuple[int, int]],
    processed_size: tuple[int, int],
    sam3: SAM3Wrapper,
    refinement: InstanceRefinementConfig,
    config: V6GeometrySegmentationConfig,
) -> dict[str, object]:
    """Run reference-only geometry prompting for one persistent instance."""

    sequence_length = len(sequence.frame_indices)
    output_size = tuple(int(value) for value in recovery.output_size)
    _validate_inputs(
        sequence_length=sequence_length,
        output_size=output_size,
        reference_mask=reference_mask,
        raw_tracking=raw_tracking,
        current_late_tracking=current_late_tracking,
        world_points=world_points,
        confidence=confidence,
        world_to_camera=world_to_camera,
        intrinsics=intrinsics,
        source_sizes=source_sizes,
    )
    reference = int(sequence.reference_frame_idx)
    raw_masks = raw_tracking.masks.detach().cpu().bool().clone()
    late_masks = current_late_tracking.masks.detach().cpu().bool().clone()
    box_masks = raw_masks.clone()
    registration_masks = raw_masks.clone()
    point_masks = raw_masks.clone()
    point_registration_masks = raw_masks.clone()
    box_scores = raw_tracking.scores.detach().cpu().float().clone()
    registration_scores = box_scores.clone()
    point_scores = box_scores.clone()
    point_registration_scores = box_scores.clone()
    box_masks[reference] = reference_mask.detach().cpu().bool()
    registration_masks[reference] = reference_mask.detach().cpu().bool()
    point_masks[reference] = reference_mask.detach().cpu().bool()
    point_registration_masks[reference] = reference_mask.detach().cpu().bool()
    box_scores[reference] = 1.0
    registration_scores[reference] = 1.0
    point_scores[reference] = 1.0
    point_registration_scores[reference] = 1.0

    reference_grid = output_mask_to_stream(
        reference_mask,
        source_size=source_sizes[reference],
        processed_size=processed_size,
        image_mode=recovery.image_mode,
    )
    reference_grid = reference_grid & (
        confidence[reference] >= float(config.point_confidence_threshold)
    )
    reference_points, reference_weights = sample_masked_observation(
        world_points[reference],
        confidence[reference],
        reference_grid,
        max_points=recovery.max_points_per_object,
    )
    object_map = ObjectPointMap(
        max_points_per_object=recovery.max_points_per_object
    )
    object_map.update(
        instance_id=int(sequence.instance_id),
        label=str(sequence.label),
        points=reference_points,
        weights=reference_weights,
        frame_idx=reference,
    )
    entry = object_map.get(int(sequence.instance_id))
    diagnostics: list[dict[str, object]] = []

    for frame, frame_index in enumerate(sequence.frame_indices):
        if frame == reference:
            if entry is None:
                diagnostics.append(
                    _fallback_diagnostic(
                        frame=frame,
                        frame_index=int(frame_index),
                        reason=(
                            "reference_mask_empty_or_no_confident_points"
                        ),
                    )
                )
            else:
                diagnostics.append(
                    _reference_diagnostic(
                        frame=frame,
                        frame_index=int(frame_index),
                        reference_points=int(reference_points.shape[0]),
                    )
                )
            continue
        if entry is None:
            diagnostics.append(
                _fallback_diagnostic(
                    frame=frame,
                    frame_index=int(frame_index),
                    reason="reference_object_map_unavailable",
                )
            )
            continue
        geometry = mine_revisit_candidate(
            entry.points,
            current_world_points=world_points[frame],
            world_to_camera=world_to_camera[frame],
            intrinsics=intrinsics[frame],
            source_size=source_sizes[frame],
            processed_size=processed_size,
            output_size=output_size,
            image_mode=recovery.image_mode,
            box_quantile=recovery.box_quantile,
            box_padding_ratio=recovery.box_padding_ratio,
            min_projected_points=recovery.min_projected_points,
            min_projected_fraction=recovery.min_projected_fraction,
            min_supported_points=recovery.min_supported_points,
            min_support_ratio=recovery.min_support_ratio,
            support_abs_distance=recovery.support_abs_distance,
            support_relative_distance=recovery.support_relative_distance,
        )
        if not geometry.accepted or not geometry.supported_mask.any():
            diagnostics.append(
                _fallback_diagnostic(
                    frame=frame,
                    frame_index=int(frame_index),
                    reason=f"geometry_rejected:{geometry.reason}",
                    geometry=geometry,
                )
            )
            continue
        box_candidates = sam3.propose_geometry_prompt_masks(
            Path(sequence.image_paths[frame]),
            prompt=str(sequence.label),
            output_size=output_size,
            geometry_prompt=geometry.mask,
        )
        evaluated_box = [
            evaluate_geometry_prompt_candidate(
                candidate,
                geometry=geometry,
                current_world_points=world_points[frame],
                current_confidence=confidence[frame],
                object_points=entry.points,
                source_size=source_sizes[frame],
                processed_size=processed_size,
                image_mode=recovery.image_mode,
                instance_id=int(sequence.instance_id),
                refinement=refinement,
                config=config,
            )
            for candidate in box_candidates
        ]
        negative_prompt = geometry.projected_mask & ~_dilate(
            geometry.supported_mask,
            int(config.point_negative_exclusion_radius),
        )
        point_candidates = sam3.propose_geometry_point_refined_masks(
            Path(sequence.image_paths[frame]),
            prompt=str(sequence.label),
            output_size=output_size,
            geometry_prompt=geometry.mask,
            positive_prompt=geometry.supported_mask,
            negative_prompt=negative_prompt,
            max_positive_points=int(config.point_positive_samples),
            max_negative_points=int(config.point_negative_samples),
        )
        evaluated_points = [
            evaluate_geometry_prompt_candidate(
                candidate,
                geometry=geometry,
                current_world_points=world_points[frame],
                current_confidence=confidence[frame],
                object_points=entry.points,
                source_size=source_sizes[frame],
                processed_size=processed_size,
                image_mode=recovery.image_mode,
                instance_id=int(sequence.instance_id),
                refinement=refinement,
                config=config,
            )
            for candidate in point_candidates
        ]
        selected_box = select_geometry_prompt_candidate(
            evaluated_box,
            require_registration=False,
            config=config,
        )
        selected_registration = select_geometry_prompt_candidate(
            evaluated_box,
            require_registration=True,
            config=config,
        )
        selected_point = select_geometry_prompt_candidate(
            evaluated_points,
            require_registration=False,
            config=config,
        )
        selected_point_registration = select_geometry_prompt_candidate(
            evaluated_points,
            require_registration=True,
            config=config,
        )
        if selected_box is not None:
            box_masks[frame] = selected_box["candidate"].mask
            box_scores[frame] = float(selected_box["candidate"].score)
        if selected_registration is not None:
            registration_masks[frame] = selected_registration["candidate"].mask
            registration_scores[frame] = float(
                selected_registration["candidate"].score
            )
        if selected_point is not None:
            point_masks[frame] = selected_point["candidate"].mask
            point_scores[frame] = float(
                selected_point["candidate"].score
            )
        if selected_point_registration is not None:
            point_registration_masks[frame] = (
                selected_point_registration["candidate"].mask
            )
            point_registration_scores[frame] = float(
                selected_point_registration["candidate"].score
            )
        diagnostics.append(
            _frame_diagnostic(
                frame=frame,
                frame_index=int(frame_index),
                geometry=geometry,
                evaluated_box=evaluated_box,
                evaluated_points=evaluated_points,
                selected_box=selected_box,
                selected_registration=selected_registration,
                selected_point=selected_point,
                selected_point_registration=selected_point_registration,
            )
        )

    return {
        "masks": {
            "raw_sam31": raw_masks,
            "current_sam3_late_geometry": late_masks,
            "v6_sam31_geometry_box": box_masks,
            "v6_sam31_geometry_points": point_masks,
            "v6_sam31_geometry_box_3d": registration_masks,
            "v6_sam31_geometry_points_3d": point_registration_masks,
        },
        "scores": {
            "raw_sam31": raw_tracking.scores.detach().cpu().float(),
            "current_sam3_late_geometry": (
                current_late_tracking.scores.detach().cpu().float()
            ),
            "v6_sam31_geometry_box": box_scores,
            "v6_sam31_geometry_points": point_scores,
            "v6_sam31_geometry_box_3d": registration_scores,
            "v6_sam31_geometry_points_3d": point_registration_scores,
        },
        "diagnostics": diagnostics,
    }


def evaluate_geometry_prompt_candidate(
    candidate: SAM3MaskCandidate,
    *,
    geometry,
    current_world_points: torch.Tensor,
    current_confidence: torch.Tensor,
    object_points: torch.Tensor,
    source_size: tuple[int, int],
    processed_size: tuple[int, int],
    image_mode: str,
    instance_id: int,
    refinement: InstanceRefinementConfig,
    config: V6GeometrySegmentationConfig,
) -> dict[str, object]:
    """Measure a prompted SAM3 mask using geometry in both directions."""

    mask = candidate.mask.detach().cpu().bool()
    supported = geometry.supported_mask.detach().cpu().bool()
    coarse_box = geometry.mask.detach().cpu().bool()
    dilated_support = _dilate(supported, int(config.support_dilation))
    support_recall = _coverage(supported, mask)
    box_precision = _precision(mask, coarse_box)
    support_precision = _precision(mask, dilated_support)
    grid_mask = output_mask_to_stream(
        mask,
        source_size=source_size,
        processed_size=processed_size,
        image_mode=image_mode,
    )
    grid_mask = grid_mask & (
        current_confidence >= float(config.point_confidence_threshold)
    )
    current_points, _ = sample_masked_observation(
        current_world_points,
        current_confidence,
        grid_mask,
        max_points=refinement.map_max_points,
    )
    registration = translation_icp(
        current_points,
        object_points,
        instance_id=int(instance_id),
        config=refinement,
    )
    registration_score = _registration_score(registration)
    two_d_score = (
        0.55 * support_recall
        + 0.30 * box_precision
        + 0.15 * float(candidate.score)
    )
    three_d_score = (
        0.35 * support_recall
        + 0.20 * box_precision
        + 0.35 * registration_score
        + 0.10 * float(candidate.score)
    )
    return {
        "candidate": candidate,
        "support_recall": support_recall,
        "box_precision": box_precision,
        "support_precision": support_precision,
        "registration": registration,
        "registration_score": registration_score,
        "two_d_score": two_d_score,
        "three_d_score": three_d_score,
    }


def select_geometry_prompt_candidate(
    evaluated: list[dict[str, object]],
    *,
    require_registration: bool,
    config: V6GeometrySegmentationConfig,
) -> dict[str, object] | None:
    """Select without GT; return ``None`` for safe raw-SAM3 fallback."""

    eligible = [
        row
        for row in evaluated
        if float(row["support_recall"]) >= float(config.min_support_recall)
        and float(row["box_precision"]) >= float(config.min_box_precision)
        and (
            not require_registration
            or bool(row["registration"].accepted)
        )
    ]
    if not eligible:
        return None
    score_name = "three_d_score" if require_registration else "two_d_score"
    return max(
        eligible,
        key=lambda row: (
            float(row[score_name]),
            float(row["support_precision"]),
            float(row["candidate"].score),
            -int(row["candidate"].obj_id),
        ),
    )


def _registration_score(proposal: TranslationProposal) -> float:
    if proposal.correspondences <= 0 or not math.isfinite(proposal.rmse):
        return 0.0
    scale = max(float(proposal.correspondence_distance), 1e-6)
    return float(proposal.fitness) * math.exp(
        -min(float(proposal.rmse) / scale, 10.0)
    )


def _coverage(evidence: torch.Tensor, mask: torch.Tensor) -> float:
    denominator = int(evidence.sum())
    if denominator == 0:
        return 0.0
    return float((evidence & mask).sum()) / denominator


def _precision(mask: torch.Tensor, region: torch.Tensor) -> float:
    denominator = int(mask.sum())
    if denominator == 0:
        return 0.0
    return float((mask & region).sum()) / denominator


def _dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask.bool()
    kernel = 2 * radius + 1
    return F.max_pool2d(
        mask.float()[None, None],
        kernel_size=kernel,
        stride=1,
        padding=radius,
    )[0, 0].bool()


def _reference_diagnostic(
    *,
    frame: int,
    frame_index: int,
    reference_points: int,
) -> dict[str, object]:
    return {
        "sequence_index": frame,
        "frame_index": frame_index,
        "geometry_accepted": 1,
        "geometry_reason": "reference_mask",
        "projected_points": reference_points,
        "supported_points": reference_points,
        "box_candidates": 0,
        "point_candidates": 0,
        "box_selected": 1,
        "box_3d_selected": 1,
        "points_selected": 1,
        "points_3d_selected": 1,
        **_reference_selection_diagnostic("box"),
        **_reference_selection_diagnostic("points"),
        "selection_reason": "reference_mask",
    }


def _fallback_diagnostic(
    *,
    frame: int,
    frame_index: int,
    reason: str,
    geometry=None,
) -> dict[str, object]:
    return {
        "sequence_index": frame,
        "frame_index": frame_index,
        "geometry_accepted": int(
            geometry is not None and bool(geometry.accepted)
        ),
        "geometry_reason": (
            geometry.reason if geometry is not None else reason
        ),
        "projected_points": (
            int(geometry.projected_points) if geometry is not None else 0
        ),
        "supported_points": (
            int(geometry.supported_points) if geometry is not None else 0
        ),
        "box_candidates": 0,
        "point_candidates": 0,
        "box_selected": 0,
        "box_3d_selected": 0,
        "points_selected": 0,
        "points_3d_selected": 0,
        **_empty_selection_diagnostic("box"),
        **_empty_selection_diagnostic("points"),
        "selection_reason": reason,
    }


def _frame_diagnostic(
    *,
    frame: int,
    frame_index: int,
    geometry,
    evaluated_box: list[dict[str, object]],
    evaluated_points: list[dict[str, object]],
    selected_box: dict[str, object] | None,
    selected_registration: dict[str, object] | None,
    selected_point: dict[str, object] | None,
    selected_point_registration: dict[str, object] | None,
) -> dict[str, object]:
    box_diagnostic = selected_registration or selected_box
    point_diagnostic = selected_point_registration or selected_point
    return {
        "sequence_index": frame,
        "frame_index": frame_index,
        "geometry_accepted": int(geometry.accepted),
        "geometry_reason": geometry.reason,
        "projected_points": int(geometry.projected_points),
        "supported_points": int(geometry.supported_points),
        "box_candidates": len(evaluated_box),
        "point_candidates": len(evaluated_points),
        "box_selected": int(selected_box is not None),
        "box_3d_selected": int(selected_registration is not None),
        "points_selected": int(selected_point is not None),
        "points_3d_selected": int(
            selected_point_registration is not None
        ),
        **_selection_diagnostic("box", box_diagnostic),
        **_selection_diagnostic("points", point_diagnostic),
        "selection_reason": (
            "box="
            + (
                "3d"
                if selected_registration is not None
                else "2d"
                if selected_box is not None
                else "raw_fallback"
            )
            + ";points="
            + (
                "3d"
                if selected_point_registration is not None
                else "2d"
                if selected_point is not None
                else "raw_fallback"
            )
        ),
    }


def _reference_selection_diagnostic(prefix: str) -> dict[str, object]:
    return {
        f"{prefix}_support_recall": 1.0,
        f"{prefix}_precision": 1.0,
        f"{prefix}_registration_fitness": 1.0,
        f"{prefix}_registration_rmse": 0.0,
    }


def _empty_selection_diagnostic(prefix: str) -> dict[str, object]:
    return {
        f"{prefix}_support_recall": 0.0,
        f"{prefix}_precision": 0.0,
        f"{prefix}_registration_fitness": 0.0,
        f"{prefix}_registration_rmse": float("nan"),
    }


def _selection_diagnostic(
    prefix: str,
    selected: dict[str, object] | None,
) -> dict[str, object]:
    if selected is None:
        return _empty_selection_diagnostic(prefix)
    registration = selected["registration"]
    return {
        f"{prefix}_support_recall": float(
            selected["support_recall"]
        ),
        f"{prefix}_precision": float(selected["box_precision"]),
        f"{prefix}_registration_fitness": float(
            registration.fitness
        ),
        f"{prefix}_registration_rmse": float(registration.rmse),
    }


def _validate_inputs(
    *,
    sequence_length: int,
    output_size: tuple[int, int],
    reference_mask: torch.Tensor,
    raw_tracking: TrackingSequence,
    current_late_tracking: TrackingSequence,
    world_points: torch.Tensor,
    confidence: torch.Tensor,
    world_to_camera: torch.Tensor,
    intrinsics: torch.Tensor,
    source_sizes: Sequence[tuple[int, int]],
) -> None:
    expected_masks = (sequence_length, *output_size)
    if tuple(reference_mask.shape) != output_size:
        raise ValueError("V6 reference mask/output size mismatch.")
    if tuple(raw_tracking.masks.shape) != expected_masks:
        raise ValueError("V6 raw SAM3 mask shape mismatch.")
    if tuple(current_late_tracking.masks.shape) != expected_masks:
        raise ValueError("V6 current-late mask shape mismatch.")
    if tuple(raw_tracking.scores.shape) != (sequence_length,):
        raise ValueError("V6 raw SAM3 score shape mismatch.")
    if tuple(current_late_tracking.scores.shape) != (sequence_length,):
        raise ValueError("V6 current-late score shape mismatch.")
    if world_points.shape[:3] != confidence.shape:
        raise ValueError("V6 pointmap/confidence shape mismatch.")
    if world_points.shape[0] != sequence_length:
        raise ValueError("V6 pointmap/frame count mismatch.")
    if world_to_camera.shape != (sequence_length, 3, 4):
        raise ValueError("V6 world_to_camera shape mismatch.")
    if intrinsics.shape != (sequence_length, 3, 3):
        raise ValueError("V6 intrinsics shape mismatch.")
    if len(source_sizes) != sequence_length:
        raise ValueError("V6 source-size/frame count mismatch.")
