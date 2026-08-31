"""Pure helpers for the frozen-V0 real SAM temporal-prompt ablation.

The actual SAM invocation intentionally lives in the experiment runner.  This
module contains only deterministic candidate bookkeeping, prompt-mask
construction, score fallback decisions, and evaluation summaries so those
parts can be tested without a CUDA/SAM installation.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import torch


WORSENED_IOU_MARGIN = 0.05
RAW_CORRECT_IOU_THRESHOLD = 0.50


@dataclass(frozen=True)
class PromptBranchSpec:
    """Mapping from an A--E candidate family to one real-SAM branch."""

    name: str
    candidate_branch: str
    tolerance: str
    max_points: int


REAL_SAM_BRANCHES: tuple[PromptBranchSpec, ...] = (
    PromptBranchSpec(
        name="A_center",
        candidate_branch="A_center",
        tolerance="",
        max_points=1,
    ),
    PromptBranchSpec(
        name="C_surface_5",
        candidate_branch="C_surface_5",
        tolerance="",
        max_points=5,
    ),
    PromptBranchSpec(
        name="E_depth_gate_rel_0.15",
        candidate_branch="E_depth_gate",
        tolerance="0.150000",
        max_points=5,
    ),
    PromptBranchSpec(
        name="E_depth_gate_rel_0.20",
        candidate_branch="E_depth_gate",
        tolerance="0.200000",
        max_points=5,
    ),
)


def query_key(row: Mapping[str, object]) -> tuple[int, int]:
    """Return the stable frame-slot key used by all prompt branches."""

    return int(row["sequence_index"]), int(row["slot"])


def deterministic_query_subset(
    keys: Sequence[tuple[int, int]],
    limit: int,
) -> tuple[tuple[int, int], ...]:
    """Select a fixed, approximately uniform subset without annotations.

    ``limit <= 0`` means all keys.  The input is de-duplicated and sorted, so
    repeated runs and different CSV row ordering produce the same SAM calls.
    """

    ordered = tuple(sorted(set((int(frame), int(slot)) for frame, slot in keys)))
    if int(limit) <= 0 or len(ordered) <= int(limit):
        return ordered
    if int(limit) < 1:
        return ()
    if int(limit) == 1:
        return (ordered[0],)
    selected: list[tuple[int, int]] = []
    for index in range(int(limit)):
        position = round(index * (len(ordered) - 1) / float(int(limit) - 1))
        key = ordered[int(position)]
        if not selected or selected[-1] != key:
            selected.append(key)
    return tuple(selected)


def rows_for_branch(
    rows: Sequence[Mapping[str, object]],
    *,
    key: tuple[int, int],
    spec: PromptBranchSpec,
) -> list[Mapping[str, object]]:
    """Return accepted, causal point rows for one frame-slot branch."""

    selected = [
        row
        for row in rows
        if query_key(row) == key
        and str(row.get("branch", "")) == spec.candidate_branch
        and str(row.get("tolerance", "")) == spec.tolerance
        and int(row.get("accepted_prompt", 0)) == 1
    ]
    selected.sort(key=lambda row: int(row.get("candidate_index", 0)))
    return selected[: int(spec.max_points)]


def build_prompt_mask(
    rows: Sequence[Mapping[str, object]],
    *,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> tuple[torch.Tensor, tuple[tuple[float, float], ...]]:
    """Rasterize causal candidate coordinates onto the SAM output grid.

    Candidate coordinates are pixel-center coordinates in ``source_size``.
    Pixel-center scaling is used when the cached projection grid and the SAM
    comparison grid differ.  No annotation or image content is consulted.
    """

    source_h, source_w = _positive_size(source_size, "source_size")
    target_h, target_w = _positive_size(target_size, "target_size")
    mask = torch.zeros(target_h, target_w, dtype=torch.bool)
    coordinates: list[tuple[float, float]] = []
    seen: set[tuple[int, int]] = set()
    for row in rows:
        try:
            u = float(row["projected_u"])
            v = float(row["projected_v"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(u) or not math.isfinite(v):
            continue
        target_u = (u + 0.5) * target_w / float(source_w) - 0.5
        target_v = (v + 0.5) * target_h / float(source_h) - 0.5
        x = int(math.floor(target_u + 0.5))
        y = int(math.floor(target_v + 0.5))
        if not (0 <= x < target_w and 0 <= y < target_h):
            continue
        if (x, y) in seen:
            continue
        seen.add((x, y))
        mask[y, x] = True
        coordinates.append((float(target_u), float(target_v)))
    return mask, tuple(coordinates)


def decide_score_fallback(
    raw_score: float,
    prompted_score: float,
    *,
    absolute_margin: float = 0.10,
    relative_ratio: float = 0.80,
) -> tuple[bool, str]:
    """Apply a fixed, annotation-free SAM-score safety rule.

    A prompted result is rejected when its score is invalid, or is both
    materially below the raw score under either predeclared test.  A missing
    raw score (non-positive) does not suppress a valid prompted result.
    """

    margin = float(absolute_margin)
    ratio = float(relative_ratio)
    if margin < 0.0:
        raise ValueError("absolute_margin must be non-negative.")
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("relative_ratio must be in [0,1].")
    raw = float(raw_score)
    prompted = float(prompted_score)
    if not math.isfinite(prompted):
        return True, "prompt_score_invalid"
    if not math.isfinite(raw) or raw <= 0.0:
        return False, "raw_score_unavailable"
    if prompted < raw - margin:
        return True, "prompt_score_below_absolute_margin"
    if prompted < raw * ratio:
        return True, "prompt_score_below_relative_ratio"
    return False, "score_ok"


def summarize_prompt_events(
    rows: Sequence[Mapping[str, object]],
    *,
    branch: str,
) -> dict[str, object]:
    """Summarize causal SAM attempts and raw-mask fallbacks."""

    selected = [row for row in rows if str(row.get("branch", "")) == branch]
    attempted = [row for row in selected if int(row.get("sam_called", 0)) == 1]
    available = [row for row in selected if int(row.get("prompt_available", 0)) == 1]
    # ``final_source`` is stored as a string in the causal event CSV.
    applied = [
        row for row in selected if str(row.get("final_source", "")) == "prompt"
    ]
    fallback = [row for row in selected if int(row.get("fallback", 0)) == 1]
    score_fallback = [
        row for row in selected if int(row.get("score_fallback", 0)) == 1
    ]
    errors = [row for row in selected if int(row.get("sam_error", 0)) == 1]
    changed = [row for row in selected if int(row.get("final_mask_changed", 0)) == 1]
    return {
        "branch": str(branch),
        "query_count": int(len(selected)),
        "prompt_available_count": int(len(available)),
        "sam_call_count": int(len(attempted)),
        "sam_error_count": int(len(errors)),
        "prompt_applied_count": int(len(applied)),
        "fallback_count": int(len(fallback)),
        "score_fallback_count": int(len(score_fallback)),
        "mask_changed_count": int(len(changed)),
        "prompt_availability": _ratio(len(available), len(selected)),
        "sam_success_rate": _ratio(len(attempted) - len(errors), len(attempted)),
        "fallback_rate_of_calls": _ratio(len(fallback), len(attempted)),
        "fallback_rate_of_available": _ratio(len(fallback), len(available)),
        "score_fallback_rate": _ratio(len(score_fallback), len(attempted)),
        "mask_changed_rate": _ratio(len(changed), len(selected)),
        "mean_raw_score": _mean_float(attempted, "raw_score"),
        "mean_prompted_score": _mean_float(attempted, "prompted_score"),
        "mean_final_score": _mean_float(selected, "final_score"),
    }


def summarize_worsened_frames(
    frame_rows: Sequence[Mapping[str, object]],
    *,
    branch: str,
    margin: float = WORSENED_IOU_MARGIN,
    raw_correct_threshold: float = RAW_CORRECT_IOU_THRESHOLD,
) -> dict[str, object]:
    """Compute object-frame and literal scene-frame worsening rates.

    The primary ``worsened_frame_ratio`` is the requested scene-frame ratio:
    for each sequence frame, average IoU over visible GT objects with a frozen
    raw assignment, then compare prompted versus raw by ``margin``.  The
    object-frame ratio is also reported because it exposes whether one object
    dominates a frame-level average.
    """

    delta = float(margin)
    if delta < 0.0:
        raise ValueError("margin must be non-negative.")
    eligible = [
        row
        for row in frame_rows
        if str(row.get("variant", "")) == str(branch)
        and int(row.get("slot", -1)) >= 0
        and int(row.get("gt_visible", 0)) == 1
        and _finite_number(row.get("iou"))
        and _finite_number(row.get("raw_iou"))
    ]
    object_worse = [
        row
        for row in eligible
        if float(row["iou"]) < float(row["raw_iou"]) - delta
    ]
    object_improved = [
        row
        for row in eligible
        if float(row["iou"]) > float(row["raw_iou"]) + delta
    ]
    raw_correct = [
        row
        for row in eligible
        if float(row["raw_iou"]) >= float(raw_correct_threshold)
    ]
    raw_correct_worse = [
        row
        for row in raw_correct
        if float(row["iou"]) < float(row["raw_iou"]) - delta
    ]

    grouped: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for row in eligible:
        grouped[int(row["sequence_index"])].append(row)
    scene_deltas: list[float] = []
    for values in grouped.values():
        raw_mean = sum(float(row["raw_iou"]) for row in values) / len(values)
        prompted_mean = sum(float(row["iou"]) for row in values) / len(values)
        scene_deltas.append(prompted_mean - raw_mean)
    scene_worse = [value for value in scene_deltas if value < -delta]
    scene_improved = [value for value in scene_deltas if value > delta]
    return {
        "branch": str(branch),
        "worsened_iou_margin": delta,
        "raw_correct_iou_threshold": float(raw_correct_threshold),
        "object_frame_denominator": int(len(eligible)),
        "worsened_object_frame_count": int(len(object_worse)),
        "worsened_object_frame_ratio": _ratio(len(object_worse), len(eligible)),
        "improved_object_frame_count": int(len(object_improved)),
        "improved_object_frame_ratio": _ratio(len(object_improved), len(eligible)),
        "raw_correct_object_frame_denominator": int(len(raw_correct)),
        "worsened_raw_correct_count": int(len(raw_correct_worse)),
        "worsened_raw_correct_ratio": _ratio(
            len(raw_correct_worse), len(raw_correct)
        ),
        "scene_frame_denominator": int(len(scene_deltas)),
        "worsened_scene_frame_count": int(len(scene_worse)),
        "worsened_frame_ratio": _ratio(len(scene_worse), len(scene_deltas)),
        "improved_scene_frame_count": int(len(scene_improved)),
        "improved_scene_frame_ratio": _ratio(
            len(scene_improved), len(scene_deltas)
        ),
        "scene_frame_delta_mean": (
            sum(scene_deltas) / len(scene_deltas) if scene_deltas else float("nan")
        ),
    }


def _positive_size(value: Sequence[int], name: str) -> tuple[int, int]:
    result = tuple(int(item) for item in value)
    if len(result) != 2 or any(item <= 0 for item in result):
        raise ValueError(f"{name} must be a positive (H,W) pair.")
    return result[0], result[1]


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if int(denominator) else 0.0


def _finite_number(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _mean_float(rows: Sequence[Mapping[str, object]], key: str) -> float:
    values = [float(row[key]) for row in rows if _finite_number(row.get(key))]
    return sum(values) / len(values) if values else float("nan")
