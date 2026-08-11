"""Retained lightweight StreamVGGT-assisted SAM3.1 segmentation policy.

The deployed policy consumes only a coarse image-space box and positive
geometry support. It never reads GT and does not depend on geometry-model
hidden features, ICP, or an object-map implementation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from .backbones.sam3_wrapper import SAM3Wrapper
from .types import SAM3MaskCandidate, TrackingSequence


RAW_SAM31_VARIANT = "raw_sam31"
V6_DEPLOYED_VARIANT = "v6_sam31_adaptive_positive_compete_010"
V6_SEGMENTATION_VARIANTS = (
    RAW_SAM31_VARIANT,
    V6_DEPLOYED_VARIANT,
)


def causal_prompts_after_birth(
    prompts: Sequence[GeometrySegmentationPrompt | None],
    *,
    birth_index: int,
) -> tuple[GeometrySegmentationPrompt | None, ...]:
    """Suppress geometry corrections until strictly after a track birth."""

    birth = int(birth_index)
    if birth < 0 or birth >= len(prompts):
        raise ValueError("birth_index is outside the prompt sequence.")
    return tuple(
        None if frame <= birth else prompt
        for frame, prompt in enumerate(prompts)
    )


@dataclass(frozen=True)
class GeometrySegmentationPrompt:
    """Backend-neutral image-space contract consumed by SAM3.1."""

    box_mask: torch.Tensor
    positive_mask: torch.Tensor


@dataclass(frozen=True)
class V6GeometrySegmentationConfig:
    min_candidate_support_recall: float = 0.25
    support_dilation: int = 5
    point_positive_samples: int = 6
    adaptive_raw_support_recall: float = 0.70
    adaptive_raw_box_precision: float = 0.10
    adaptive_support_margin: float = 0.05
    adaptive_score_margin: float = 0.05
    reliable_support_margin: float = 0.10
    reliable_score_margin: float = 0.10
    adaptive_min_area_ratio: float = 0.50
    adaptive_max_area_ratio: float = 4.00


@torch.no_grad()
def segment_instance_with_geometry_prompts(
    *,
    sequence,
    reference_mask: torch.Tensor,
    raw_tracking: TrackingSequence,
    geometry_prompts: Sequence[GeometrySegmentationPrompt | None],
    output_size: tuple[int, int],
    sam3: SAM3Wrapper,
    config: V6GeometrySegmentationConfig,
) -> dict[str, object]:
    """Return raw SAM3.1 and the single selected V6 correction."""

    sequence_length = len(sequence.frame_indices)
    output_size = tuple(int(value) for value in output_size)
    _validate_inputs(
        sequence_length=sequence_length,
        output_size=output_size,
        reference_mask=reference_mask,
        raw_tracking=raw_tracking,
        geometry_prompts=geometry_prompts,
    )
    reference = int(sequence.reference_frame_idx)
    raw_masks = raw_tracking.masks.detach().cpu().bool().clone()
    raw_scores = raw_tracking.scores.detach().cpu().float().clone()
    corrected_masks = raw_masks.clone()
    corrected_scores = raw_scores.clone()
    reference_mask = reference_mask.detach().cpu().bool()
    raw_masks[reference] = reference_mask
    raw_scores[reference] = 1.0
    corrected_masks[reference] = reference_mask
    corrected_scores[reference] = 1.0

    diagnostics: list[dict[str, object]] = []
    for frame, frame_index in enumerate(sequence.frame_indices):
        if frame == reference:
            diagnostics.append(
                _reference_diagnostic(
                    frame=frame,
                    frame_index=int(frame_index),
                    reference_pixels=int(reference_mask.sum()),
                )
            )
            continue
        source_prompt = geometry_prompts[frame]
        if source_prompt is None:
            diagnostics.append(
                _fallback_diagnostic(
                    frame=frame,
                    frame_index=int(frame_index),
                    raw_pixels=int(raw_masks[frame].sum()),
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
        prompted_row = select_geometry_prompt_candidate(
            _run_positive_point_prompt(
                sam3=sam3,
                image_path=Path(sequence.image_paths[frame]),
                label=str(sequence.label),
                output_size=output_size,
                prompt=prompt,
                config=config,
            ),
            config=config,
        )
        selected, reason = select_adaptive_correction(
            raw_row=raw_row,
            prompted_row=prompted_row,
            config=config,
        )
        if selected is not None:
            candidate = selected["candidate"]
            corrected_masks[frame] = candidate.mask
            corrected_scores[frame] = float(candidate.score)
        diagnostics.append(
            _frame_diagnostic(
                frame=frame,
                frame_index=int(frame_index),
                prompt=prompt,
                raw_row=raw_row,
                prompted_row=prompted_row,
                correction_applied=selected is not None,
                correction_reason=reason,
            )
        )

    return {
        "masks": {
            RAW_SAM31_VARIANT: raw_masks,
            V6_DEPLOYED_VARIANT: corrected_masks,
        },
        "scores": {
            RAW_SAM31_VARIANT: raw_scores,
            V6_DEPLOYED_VARIANT: corrected_scores,
        },
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
    """Apply the selected compete-0.10 policy without a strategy sweep."""

    if prompted_row is None:
        return None, "keep_raw:no_prompt_candidate"
    raw_pixels = int(raw_row["mask_pixels"])
    raw_support = float(raw_row["support_recall"])
    raw_box_precision = float(raw_row["box_precision"])
    raw_unreliable = (
        raw_pixels == 0
        or raw_support < float(config.adaptive_raw_support_recall)
        or raw_box_precision < float(config.adaptive_raw_box_precision)
    )

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
    support_margin = (
        float(config.adaptive_support_margin)
        if raw_unreliable
        else float(config.reliable_support_margin)
    )
    score_margin = (
        float(config.adaptive_score_margin)
        if raw_unreliable
        else float(config.reliable_score_margin)
    )
    if raw_pixels > 0 and support_gain < support_margin:
        return None, "keep_raw:insufficient_support_gain"
    if score_gain < score_margin:
        return None, "keep_raw:insufficient_score_gain"
    return prompted_row, (
        "apply_prompt:raw_unreliable_geometry_improved"
        if raw_unreliable
        else "apply_prompt:competitive_geometry_improved"
    )


def _run_positive_point_prompt(
    *,
    sam3: SAM3Wrapper,
    image_path: Path,
    label: str,
    output_size: tuple[int, int],
    prompt: GeometrySegmentationPrompt,
    config: V6GeometrySegmentationConfig,
) -> list[dict[str, object]]:
    candidates = sam3.propose_geometry_point_refined_masks(
        image_path,
        prompt=label,
        output_size=output_size,
        geometry_prompt=prompt.box_mask,
        positive_prompt=prompt.positive_mask,
        max_positive_points=int(config.point_positive_samples),
    )
    return [
        evaluate_mask_against_geometry(
            candidate,
            prompt=prompt,
            config=config,
        )
        for candidate in candidates
    ]


def _normalize_prompt(
    prompt: GeometrySegmentationPrompt,
) -> GeometrySegmentationPrompt:
    return GeometrySegmentationPrompt(
        box_mask=prompt.box_mask.detach().cpu().bool(),
        positive_mask=prompt.positive_mask.detach().cpu().bool(),
    )


def _coverage(evidence: torch.Tensor, mask: torch.Tensor) -> float:
    denominator = int(evidence.sum())
    return float((evidence & mask).sum()) / denominator if denominator else 0.0


def _precision(mask: torch.Tensor, region: torch.Tensor) -> float:
    denominator = int(mask.sum())
    return float((mask & region).sum()) / denominator if denominator else 0.0


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
    reference_pixels: int,
) -> dict[str, object]:
    return {
        "sequence_index": frame,
        "frame_index": frame_index,
        "prompt_available": int(reference_pixels > 0),
        "box_pixels": reference_pixels,
        "positive_pixels": reference_pixels,
        "raw_mask_pixels": reference_pixels,
        "prompted_mask_pixels": reference_pixels,
        "correction_applied": 0,
        "correction_reason": "reference_mask",
    }


def _fallback_diagnostic(
    *,
    frame: int,
    frame_index: int,
    raw_pixels: int,
) -> dict[str, object]:
    return {
        "sequence_index": frame,
        "frame_index": frame_index,
        "prompt_available": 0,
        "box_pixels": 0,
        "positive_pixels": 0,
        "raw_mask_pixels": raw_pixels,
        "prompted_mask_pixels": 0,
        "correction_applied": 0,
        "correction_reason": "keep_raw:geometry_prompt_unavailable",
    }


def _frame_diagnostic(
    *,
    frame: int,
    frame_index: int,
    prompt: GeometrySegmentationPrompt,
    raw_row: dict[str, object],
    prompted_row: dict[str, object] | None,
    correction_applied: bool,
    correction_reason: str,
) -> dict[str, object]:
    return {
        "sequence_index": frame,
        "frame_index": frame_index,
        "prompt_available": 1,
        "box_pixels": int(prompt.box_mask.sum()),
        "positive_pixels": int(prompt.positive_mask.sum()),
        "raw_support_recall": float(raw_row["support_recall"]),
        "raw_box_precision": float(raw_row["box_precision"]),
        "raw_geometry_score": float(raw_row["geometry_score"]),
        "raw_mask_pixels": int(raw_row["mask_pixels"]),
        "prompted_support_recall": _selected_value(
            prompted_row,
            "support_recall",
        ),
        "prompted_box_precision": _selected_value(
            prompted_row,
            "box_precision",
        ),
        "prompted_geometry_score": _selected_value(
            prompted_row,
            "geometry_score",
        ),
        "prompted_mask_pixels": _selected_int(
            prompted_row,
            "mask_pixels",
        ),
        "correction_applied": int(correction_applied),
        "correction_reason": correction_reason,
    }


def _selected_value(
    selected: dict[str, object] | None,
    key: str,
) -> float:
    return float(selected[key]) if selected is not None else float("nan")


def _selected_int(
    selected: dict[str, object] | None,
    key: str,
) -> int:
    return int(selected[key]) if selected is not None else 0


def _validate_inputs(
    *,
    sequence_length: int,
    output_size: tuple[int, int],
    reference_mask: torch.Tensor,
    raw_tracking: TrackingSequence,
    geometry_prompts: Sequence[GeometrySegmentationPrompt | None],
) -> None:
    if tuple(reference_mask.shape) != output_size:
        raise ValueError("V6 reference mask/output size mismatch.")
    if tuple(raw_tracking.masks.shape) != (sequence_length, *output_size):
        raise ValueError("V6 raw SAM3.1 mask shape mismatch.")
    if tuple(raw_tracking.scores.shape) != (sequence_length,):
        raise ValueError("V6 raw SAM3.1 score shape mismatch.")
    if len(geometry_prompts) != sequence_length:
        raise ValueError("V6 geometry-prompt/frame count mismatch.")
    for prompt in geometry_prompts:
        if prompt is None:
            continue
        if {
            tuple(prompt.box_mask.shape),
            tuple(prompt.positive_mask.shape),
        } != {output_size}:
            raise ValueError("V6 geometry-prompt/output size mismatch.")
