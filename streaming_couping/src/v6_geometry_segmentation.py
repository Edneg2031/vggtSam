"""Lightweight geometry-assisted SAM3.1 segmentation.

The segmentation policy consumes only three image-space masks: a coarse box,
positive support, and optional negative evidence. StreamVGGT is the current
producer of those masks, but the policy has no dependency on its hidden
features or on a specific 3D registration implementation. GT is evaluation
only and is deliberately absent from this module.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from .backbones.sam3_wrapper import SAM3Wrapper
from .types import SAM3MaskCandidate, TrackingSequence

V6_SEGMENTATION_VARIANTS = (
    "raw_sam31",
    "current_sam3_late_geometry",
    "v6_sam31_points_positive",
    "v6_sam31_points_posneg",
    "v6_sam31_adaptive_positive",
    "v6_sam31_adaptive_posneg",
)


@dataclass(frozen=True)
class GeometrySegmentationPrompt:
    """Backend-neutral image-space contract consumed by SAM3.1."""

    box_mask: torch.Tensor
    positive_mask: torch.Tensor
    negative_mask: torch.Tensor


@dataclass(frozen=True)
class V6GeometrySegmentationConfig:
    min_candidate_support_recall: float = 0.25
    support_dilation: int = 5
    point_positive_samples: int = 6
    point_negative_samples: int = 4
    adaptive_raw_support_recall: float = 0.70
    adaptive_raw_box_precision: float = 0.10
    adaptive_score_margin: float = 0.05
    adaptive_support_margin: float = 0.05
    adaptive_min_area_ratio: float = 0.50
    adaptive_max_area_ratio: float = 4.00


@torch.no_grad()
def segment_instance_with_geometry_prompts(
    *,
    sequence,
    reference_mask: torch.Tensor,
    raw_tracking: TrackingSequence,
    current_late_tracking: TrackingSequence,
    geometry_prompts: Sequence[GeometrySegmentationPrompt | None],
    output_size: tuple[int, int],
    sam3: SAM3Wrapper,
    config: V6GeometrySegmentationConfig,
) -> dict[str, object]:
    """Refine one tracked instance from backend-neutral image-space prompts."""

    sequence_length = len(sequence.frame_indices)
    output_size = tuple(int(value) for value in output_size)
    _validate_inputs(
        sequence_length=sequence_length,
        output_size=output_size,
        reference_mask=reference_mask,
        raw_tracking=raw_tracking,
        current_late_tracking=current_late_tracking,
        geometry_prompts=geometry_prompts,
    )
    reference = int(sequence.reference_frame_idx)
    raw_masks = raw_tracking.masks.detach().cpu().bool().clone()
    raw_scores = raw_tracking.scores.detach().cpu().float().clone()
    masks_by_variant = {
        variant: raw_masks.clone()
        for variant in V6_SEGMENTATION_VARIANTS
    }
    scores_by_variant = {
        variant: raw_scores.clone()
        for variant in V6_SEGMENTATION_VARIANTS
    }
    masks_by_variant["current_sam3_late_geometry"] = (
        current_late_tracking.masks.detach().cpu().bool().clone()
    )
    scores_by_variant["current_sam3_late_geometry"] = (
        current_late_tracking.scores.detach().cpu().float().clone()
    )
    for variant in V6_SEGMENTATION_VARIANTS:
        masks_by_variant[variant][reference] = (
            reference_mask.detach().cpu().bool()
        )
        scores_by_variant[variant][reference] = 1.0

    diagnostics: list[dict[str, object]] = []

    for frame, frame_index in enumerate(sequence.frame_indices):
        if frame == reference:
            diagnostics.append(
                _reference_diagnostic(
                    frame=frame,
                    frame_index=int(frame_index),
                    available=bool(reference_mask.any()),
                )
            )
            continue
        source_prompt = geometry_prompts[frame]
        if source_prompt is None:
            diagnostics.append(
                _fallback_diagnostic(
                    frame=frame,
                    frame_index=int(frame_index),
                    reason="geometry_prompt_unavailable",
                )
            )
            continue
        prompt = _normalize_prompt(source_prompt)

        raw_row = evaluate_mask_against_geometry(
            SAM3MaskCandidate(
                obj_id=-1,
                mask=raw_masks[frame],
                score=float(raw_scores[frame]),
            ),
            prompt=prompt,
            config=config,
        )
        positive_rows = _run_point_prompt(
            sam3=sam3,
            image_path=Path(sequence.image_paths[frame]),
            label=str(sequence.label),
            output_size=output_size,
            prompt=prompt,
            include_negative=False,
            config=config,
        )
        posneg_rows = _run_point_prompt(
            sam3=sam3,
            image_path=Path(sequence.image_paths[frame]),
            label=str(sequence.label),
            output_size=output_size,
            prompt=prompt,
            include_negative=True,
            config=config,
        )
        selected_positive = select_geometry_prompt_candidate(
            positive_rows,
            config=config,
        )
        selected_posneg = select_geometry_prompt_candidate(
            posneg_rows,
            config=config,
        )
        _apply_selected(
            masks_by_variant,
            scores_by_variant,
            variant="v6_sam31_points_positive",
            frame=frame,
            selected=selected_positive,
        )
        _apply_selected(
            masks_by_variant,
            scores_by_variant,
            variant="v6_sam31_points_posneg",
            frame=frame,
            selected=selected_posneg,
        )

        adaptive_positive, positive_reason = select_adaptive_correction(
            raw_row=raw_row,
            prompted_row=selected_positive,
            config=config,
        )
        adaptive_posneg, posneg_reason = select_adaptive_correction(
            raw_row=raw_row,
            prompted_row=selected_posneg,
            config=config,
        )
        _apply_selected(
            masks_by_variant,
            scores_by_variant,
            variant="v6_sam31_adaptive_positive",
            frame=frame,
            selected=adaptive_positive,
        )
        _apply_selected(
            masks_by_variant,
            scores_by_variant,
            variant="v6_sam31_adaptive_posneg",
            frame=frame,
            selected=adaptive_posneg,
        )
        diagnostics.append(
            _frame_diagnostic(
                frame=frame,
                frame_index=int(frame_index),
                prompt=prompt,
                raw_row=raw_row,
                selected_positive=selected_positive,
                selected_posneg=selected_posneg,
                adaptive_positive=adaptive_positive,
                adaptive_posneg=adaptive_posneg,
                positive_reason=positive_reason,
                posneg_reason=posneg_reason,
            )
        )

    return {
        "masks": masks_by_variant,
        "scores": scores_by_variant,
        "diagnostics": diagnostics,
    }


def evaluate_mask_against_geometry(
    candidate: SAM3MaskCandidate,
    *,
    prompt: GeometrySegmentationPrompt,
    config: V6GeometrySegmentationConfig,
) -> dict[str, object]:
    """Score a raw or prompted mask using image-space geometry only."""

    mask = candidate.mask.detach().cpu().bool()
    support_recall = _coverage(prompt.positive_mask, mask)
    box_precision = _precision(mask, prompt.box_mask)
    support_precision = _precision(
        mask,
        _dilate(prompt.positive_mask, int(config.support_dilation)),
    )
    geometry_score = (
        0.55 * support_recall
        + 0.30 * box_precision
        + 0.15 * float(candidate.score)
    )
    return {
        "candidate": candidate,
        "support_recall": support_recall,
        "box_precision": box_precision,
        "support_precision": support_precision,
        "geometry_score": geometry_score,
        "mask_pixels": int(mask.sum()),
        "box_pixels": int(prompt.box_mask.sum()),
    }


def select_geometry_prompt_candidate(
    evaluated: list[dict[str, object]],
    *,
    config: V6GeometrySegmentationConfig,
) -> dict[str, object] | None:
    """Choose one prompted mask, or return ``None`` for raw fallback."""

    eligible = [
        row
        for row in evaluated
        if float(row["support_recall"])
        >= float(config.min_candidate_support_recall)
    ]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda row: (
            float(row["geometry_score"]),
            float(row["support_precision"]),
            float(row["candidate"].score),
            -int(row["candidate"].obj_id),
        ),
    )


def select_adaptive_correction(
    *,
    raw_row: dict[str, object],
    prompted_row: dict[str, object] | None,
    config: V6GeometrySegmentationConfig,
) -> tuple[dict[str, object] | None, str]:
    """Use geometry as a conservative correction trigger, never a takeover."""

    if prompted_row is None:
        return None, "keep_raw:no_prompt_candidate"
    raw_pixels = int(raw_row["mask_pixels"])
    raw_support = float(raw_row["support_recall"])
    raw_box_precision = float(raw_row["box_precision"])
    trigger = (
        raw_pixels == 0
        or raw_support < float(config.adaptive_raw_support_recall)
        or raw_box_precision < float(config.adaptive_raw_box_precision)
    )
    if not trigger:
        return None, "keep_raw:raw_geometry_reliable"

    prompted_pixels = int(prompted_row["mask_pixels"])
    if raw_pixels > 0:
        area_ratio = prompted_pixels / raw_pixels
        if area_ratio < float(config.adaptive_min_area_ratio):
            return None, "keep_raw:prompt_too_small"
        if area_ratio > float(config.adaptive_max_area_ratio):
            return None, "keep_raw:prompt_too_large"
    else:
        box_pixels = int(raw_row["box_pixels"])
        if (
            box_pixels > 0
            and prompted_pixels / box_pixels
            > float(config.adaptive_max_area_ratio)
        ):
            return None, "keep_raw:prompt_too_large_for_box"

    support_gain = float(prompted_row["support_recall"]) - raw_support
    score_gain = (
        float(prompted_row["geometry_score"])
        - float(raw_row["geometry_score"])
    )
    if raw_pixels > 0 and support_gain < float(
        config.adaptive_support_margin
    ):
        return None, "keep_raw:insufficient_support_gain"
    if score_gain < float(config.adaptive_score_margin):
        return None, "keep_raw:insufficient_score_gain"
    return prompted_row, "apply_prompt:geometry_improved"


def _run_point_prompt(
    *,
    sam3: SAM3Wrapper,
    image_path: Path,
    label: str,
    output_size: tuple[int, int],
    prompt: GeometrySegmentationPrompt,
    include_negative: bool,
    config: V6GeometrySegmentationConfig,
) -> list[dict[str, object]]:
    candidates = sam3.propose_geometry_point_refined_masks(
        image_path,
        prompt=label,
        output_size=output_size,
        geometry_prompt=prompt.box_mask,
        positive_prompt=prompt.positive_mask,
        negative_prompt=(
            prompt.negative_mask if include_negative else None
        ),
        max_positive_points=int(config.point_positive_samples),
        max_negative_points=(
            int(config.point_negative_samples)
            if include_negative
            else 0
        ),
    )
    return [
        evaluate_mask_against_geometry(
            candidate,
            prompt=prompt,
            config=config,
        )
        for candidate in candidates
    ]


def _apply_selected(
    masks_by_variant: dict[str, torch.Tensor],
    scores_by_variant: dict[str, torch.Tensor],
    *,
    variant: str,
    frame: int,
    selected: dict[str, object] | None,
) -> None:
    if selected is None:
        return
    candidate = selected["candidate"]
    masks_by_variant[variant][frame] = candidate.mask
    scores_by_variant[variant][frame] = float(candidate.score)


def _normalize_prompt(
    prompt: GeometrySegmentationPrompt,
) -> GeometrySegmentationPrompt:
    return GeometrySegmentationPrompt(
        box_mask=prompt.box_mask.detach().cpu().bool(),
        positive_mask=prompt.positive_mask.detach().cpu().bool(),
        negative_mask=prompt.negative_mask.detach().cpu().bool(),
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
    available: bool,
) -> dict[str, object]:
    return {
        "sequence_index": frame,
        "frame_index": frame_index,
        "prompt_available": int(available),
        "box_pixels": int(available),
        "positive_pixels": int(available),
        "negative_pixels": 0,
        "raw_support_recall": 1.0 if available else 0.0,
        "raw_box_precision": 1.0 if available else 0.0,
        "positive_support_recall": 1.0 if available else 0.0,
        "posneg_support_recall": 1.0 if available else 0.0,
        "adaptive_positive_applied": 0,
        "adaptive_posneg_applied": 0,
        "adaptive_positive_reason": "reference_mask",
        "adaptive_posneg_reason": "reference_mask",
    }


def _fallback_diagnostic(
    *,
    frame: int,
    frame_index: int,
    reason: str,
) -> dict[str, object]:
    return {
        "sequence_index": frame,
        "frame_index": frame_index,
        "prompt_available": 0,
        "box_pixels": 0,
        "positive_pixels": 0,
        "negative_pixels": 0,
        "raw_support_recall": float("nan"),
        "raw_box_precision": float("nan"),
        "positive_support_recall": float("nan"),
        "posneg_support_recall": float("nan"),
        "adaptive_positive_applied": 0,
        "adaptive_posneg_applied": 0,
        "adaptive_positive_reason": reason,
        "adaptive_posneg_reason": reason,
    }


def _frame_diagnostic(
    *,
    frame: int,
    frame_index: int,
    prompt: GeometrySegmentationPrompt,
    raw_row: dict[str, object],
    selected_positive: dict[str, object] | None,
    selected_posneg: dict[str, object] | None,
    adaptive_positive: dict[str, object] | None,
    adaptive_posneg: dict[str, object] | None,
    positive_reason: str,
    posneg_reason: str,
) -> dict[str, object]:
    return {
        "sequence_index": frame,
        "frame_index": frame_index,
        "prompt_available": 1,
        "box_pixels": int(prompt.box_mask.sum()),
        "positive_pixels": int(prompt.positive_mask.sum()),
        "negative_pixels": int(prompt.negative_mask.sum()),
        "raw_support_recall": float(raw_row["support_recall"]),
        "raw_box_precision": float(raw_row["box_precision"]),
        "positive_support_recall": _selected_value(
            selected_positive,
            "support_recall",
        ),
        "posneg_support_recall": _selected_value(
            selected_posneg,
            "support_recall",
        ),
        "adaptive_positive_applied": int(
            adaptive_positive is not None
        ),
        "adaptive_posneg_applied": int(adaptive_posneg is not None),
        "adaptive_positive_reason": positive_reason,
        "adaptive_posneg_reason": posneg_reason,
    }


def _selected_value(
    selected: dict[str, object] | None,
    key: str,
) -> float:
    return float(selected[key]) if selected is not None else float("nan")


def _validate_inputs(
    *,
    sequence_length: int,
    output_size: tuple[int, int],
    reference_mask: torch.Tensor,
    raw_tracking: TrackingSequence,
    current_late_tracking: TrackingSequence,
    geometry_prompts: Sequence[GeometrySegmentationPrompt | None],
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
    if len(geometry_prompts) != sequence_length:
        raise ValueError("V6 geometry-prompt/frame count mismatch.")
    for prompt in geometry_prompts:
        if prompt is None:
            continue
        shapes = {
            tuple(prompt.box_mask.shape),
            tuple(prompt.positive_mask.shape),
            tuple(prompt.negative_mask.shape),
        }
        if shapes != {output_size}:
            raise ValueError("V6 geometry-prompt/output size mismatch.")
