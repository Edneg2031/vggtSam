"""Scene-disjoint identity metrics for persistent semantic-map tracks.

Ground truth is consumed only by this frozen evaluation module.  The raw SAM
branch defines one prompt-aware track-to-instance assignment; every geometry
and memory ablation is scored with that same assignment so a variant cannot
hide an identity failure by rematching itself after seeing annotations.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

from .data import extract_object_labels, read_mask, resolve_manifest_path


@dataclass(frozen=True)
class TrackingMetricConfig:
    assignment_min_iou: float = 0.01
    identity_iou_threshold: float = 0.50
    reentry_iou_threshold: float = 0.25
    reentry_min_gap: int = 2
    reentry_window: int = 3
    improvement_epsilon: float = 1e-6
    fragmentation_iou_threshold: float = 0.05


@dataclass(frozen=True)
class GroundTruthInstances:
    masks: torch.Tensor
    instance_ids: tuple[int, ...]
    labels: tuple[str, ...]
    all_visible_instance_ids: tuple[int, ...]


def load_ground_truth_instances(
    manifest_path: str | Path,
    *,
    scene_id: str,
    frame_indices: Sequence[int],
    output_size: tuple[int, int],
    prompts: Sequence[str],
    prompt_label_aliases: Mapping[str, Sequence[str]] | None = None,
    include_all_instances: bool = False,
) -> GroundTruthInstances:
    """Open GT masks after candidate generation.

    By default only instances compatible with ``prompts`` are returned, which
    preserves the historical prompt-scope metric.  ``include_all_instances``
    exposes every positive instance visible in the selected clip and is used
    by class-agnostic discovery diagnostics.  In both cases the masks are
    opened only by the evaluation caller; proposal generation never calls
    this function.
    """

    manifest_path = Path(manifest_path).expanduser().resolve()
    with manifest_path.open("r", encoding="utf8") as handle:
        manifest = json.load(handle)
    scene = next(
        (
            row
            for row in manifest.get("scenes", ())
            if str(row.get("scene_id")) == str(scene_id)
        ),
        None,
    )
    if scene is None:
        raise ValueError(f"Scene {scene_id!r} is absent from {manifest_path}.")
    frames = scene.get("frames", ())
    native = []
    for index in frame_indices:
        frame = frames[int(index)]
        value = frame.get("instance_mask")
        if not value:
            raise ValueError(
                f"Scene {scene_id} frame {index} lacks an instance mask."
            )
        native.append(
            torch.from_numpy(
                read_mask(resolve_manifest_path(value, manifest_path)).copy()
            ).long()
        )
    if not native:
        raise ValueError("Tracking evaluation requires at least one frame.")
    labels = extract_object_labels(scene.get("objects", {}))
    all_ids = sorted(
        {
            int(value)
            for mask in native
            for value in torch.unique(mask).tolist()
            if int(value) > 0
        }
    )
    if include_all_instances:
        eligible = list(all_ids)
    else:
        eligible = [
            instance_id
            for instance_id in all_ids
            if any(
                prompt_matches_label(
                    prompt,
                    labels.get(instance_id, "object"),
                    aliases=prompt_label_aliases,
                )
                for prompt in prompts
            )
        ]
    height, width = (int(value) for value in output_size)
    if eligible:
        masks = torch.stack(
            [
                torch.stack([mask == instance_id for mask in native])
                for instance_id in eligible
            ],
            dim=1,
        ).float()
        masks = F.interpolate(masks, size=(height, width), mode="nearest").bool()
    else:
        masks = torch.zeros(
            len(native),
            0,
            height,
            width,
            dtype=torch.bool,
        )
    return GroundTruthInstances(
        masks=masks,
        instance_ids=tuple(eligible),
        labels=tuple(labels.get(value, "object") for value in eligible),
        all_visible_instance_ids=tuple(all_ids),
    )


def evaluate_tracking_variants(
    *,
    scene_id: str,
    clip_name: str,
    frame_indices: Sequence[int],
    variant_masks: Mapping[str, torch.Tensor],
    variant_scores: Mapping[str, torch.Tensor],
    raw_variant: str,
    track_ids: Sequence[int],
    track_prompts: Sequence[str],
    ground_truth: GroundTruthInstances,
    config: TrackingMetricConfig = TrackingMetricConfig(),
    prompt_label_aliases: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, object]:
    """Evaluate all variants under one assignment frozen from ``raw_variant``."""

    if raw_variant not in variant_masks or raw_variant not in variant_scores:
        raise ValueError(f"Raw tracking variant {raw_variant!r} is missing.")
    raw = variant_masks[raw_variant].detach().cpu().bool()
    if raw.ndim != 4:
        raise ValueError("Tracking masks must have shape [S,K,H,W].")
    sequence, tracks, height, width = raw.shape
    if sequence != len(frame_indices):
        raise ValueError("Tracking masks/frame_indices disagree.")
    if len(track_ids) != tracks or len(track_prompts) != tracks:
        raise ValueError("Track metadata does not match mask slots.")
    gt = ground_truth.masks.detach().cpu().bool()
    if gt.shape[:1] + gt.shape[2:] != (sequence, height, width):
        raise ValueError("Ground-truth and prediction mask grids disagree.")
    for variant, masks in variant_masks.items():
        if tuple(masks.shape) != tuple(raw.shape):
            raise ValueError(f"Variant {variant!r} has a mismatched mask shape.")
        scores = variant_scores.get(variant)
        if scores is None or tuple(scores.shape) != (sequence, tracks):
            raise ValueError(f"Variant {variant!r} has invalid scores.")

    compatibility = torch.zeros(tracks, gt.shape[1], dtype=torch.bool)
    for slot, prompt in enumerate(track_prompts):
        for target, label in enumerate(ground_truth.labels):
            compatibility[slot, target] = bool(prompt) and prompt_matches_label(
                prompt,
                label,
                aliases=prompt_label_aliases,
            )
    assignment_scores = global_iou_matrix(raw, gt)
    assignment_scores = torch.where(
        compatibility,
        assignment_scores,
        torch.full_like(assignment_scores, -1.0),
    )
    pairs = maximum_weight_assignment(assignment_scores)
    pairs = [
        (slot, target)
        for slot, target in pairs
        if bool(compatibility[slot, target])
        and float(assignment_scores[slot, target])
        >= float(config.assignment_min_iou)
    ]
    slot_to_target = {slot: target for slot, target in pairs}
    target_to_slot = {target: slot for slot, target in pairs}
    assignments = [
        {
            "scene_id": scene_id,
            "clip": clip_name,
            "slot": int(slot),
            "track_id": int(track_ids[slot]),
            "track_prompt": str(track_prompts[slot]),
            "gt_index": int(target),
            "gt_instance_id": int(ground_truth.instance_ids[target]),
            "gt_label": str(ground_truth.labels[target]),
            "raw_global_iou": float(assignment_scores[slot, target]),
        }
        for slot, target in sorted(pairs)
    ]
    raw_pair_iou = _pair_frame_ious(raw, gt, target_to_slot)
    summary_rows: list[dict[str, object]] = []
    object_rows: list[dict[str, object]] = []
    frame_rows: list[dict[str, object]] = []
    for variant in variant_masks:
        result = _evaluate_one_variant(
            scene_id=scene_id,
            clip_name=clip_name,
            frame_indices=frame_indices,
            variant=variant,
            masks=variant_masks[variant].detach().cpu().bool(),
            scores=variant_scores[variant].detach().cpu().float(),
            raw_pair_iou=raw_pair_iou,
            gt=gt,
            ground_truth=ground_truth,
            track_ids=track_ids,
            track_prompts=track_prompts,
            compatibility=compatibility,
            slot_to_target=slot_to_target,
            target_to_slot=target_to_slot,
            config=config,
        )
        summary_rows.append(result["summary"])
        object_rows.extend(result["objects"])
        frame_rows.extend(result["frames"])
    return {
        "assignments": assignments,
        "summary_rows": summary_rows,
        "object_rows": object_rows,
        "frame_rows": frame_rows,
        "eligible_gt_instance_ids": ground_truth.instance_ids,
        "all_visible_gt_instance_ids": ground_truth.all_visible_instance_ids,
        "raw_variant": raw_variant,
    }


def global_iou_matrix(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return sequence-accumulated IoU for every track/GT pair."""

    predicted = predicted.detach().cpu().bool()
    target = target.detach().cpu().bool()
    if predicted.ndim != 4 or target.ndim != 4:
        raise ValueError("Masks must have shapes [S,K,H,W] and [S,G,H,W].")
    if predicted.shape[0] != target.shape[0] or predicted.shape[2:] != target.shape[2:]:
        raise ValueError("Prediction and target grids disagree.")
    tracks, objects = predicted.shape[1], target.shape[1]
    output = torch.zeros(tracks, objects, dtype=torch.float64)
    for slot in range(tracks):
        left = predicted[:, slot]
        for obj in range(objects):
            right = target[:, obj]
            union = int((left | right).sum())
            output[slot, obj] = (
                float((left & right).sum()) / union if union else 0.0
            )
    return output


def maximum_weight_assignment(scores: torch.Tensor) -> list[tuple[int, int]]:
    """Pure-Python/torch rectangular Hungarian assignment with dummy nodes."""

    values = scores.detach().double().cpu()
    if values.ndim != 2:
        raise ValueError("Assignment scores must be a matrix.")
    rows, columns = values.shape
    if not rows or not columns:
        return []
    size = max(rows, columns)
    padded = torch.zeros(size, size, dtype=torch.float64)
    padded[:rows, :columns] = values
    maximum = float(padded.max())
    cost = (maximum - padded).tolist()
    assignment = _hungarian_minimum(cost)
    return [
        (row, column)
        for row, column in enumerate(assignment[:rows])
        if 0 <= column < columns
    ]


def prompt_matches_label(
    prompt: str,
    label: str,
    *,
    aliases: Mapping[str, Sequence[str]] | None = None,
) -> bool:
    prompt_key = _normalize_label(prompt)
    label_key = _normalize_label(label)
    if not prompt_key or not label_key:
        return False
    if prompt_key in {"object", "thing", "item"}:
        return True
    candidates = {prompt_key}
    for alias in (aliases or {}).get(prompt, ()):  # preserve user-facing key
        candidates.add(_normalize_label(alias))
    for key, values in (aliases or {}).items():
        if _normalize_label(key) == prompt_key:
            candidates.update(_normalize_label(value) for value in values)
    return any(
        candidate == label_key
        or candidate in label_key
        or label_key in candidate
        for candidate in candidates
        if candidate
    )


def _evaluate_one_variant(
    *,
    scene_id: str,
    clip_name: str,
    frame_indices: Sequence[int],
    variant: str,
    masks: torch.Tensor,
    scores: torch.Tensor,
    raw_pair_iou: torch.Tensor,
    gt: torch.Tensor,
    ground_truth: GroundTruthInstances,
    track_ids: Sequence[int],
    track_prompts: Sequence[str],
    compatibility: torch.Tensor,
    slot_to_target: Mapping[int, int],
    target_to_slot: Mapping[int, int],
    config: TrackingMetricConfig,
) -> dict[str, object]:
    sequence, tracks = masks.shape[:2]
    objects = gt.shape[1]
    pair_iou = _pair_frame_ious(masks, gt, target_to_slot)
    gt_visible_values: list[float] = []
    positive_values: list[float] = []
    improvements = worsened = unchanged = 0
    frame_idtp = frame_idfp = frame_idfn = 0
    pixel_idtp = pixel_idfp = pixel_idfn = 0
    object_rows: list[dict[str, object]] = []
    frame_rows: list[dict[str, object]] = []
    total_switches = 0
    total_reentries = 0
    successful_reentries = 0

    for target in range(objects):
        slot = target_to_slot.get(target, -1)
        target_masks = gt[:, target]
        target_visible = target_masks.flatten(1).any(dim=1)
        current_iou = (
            pair_iou[:, target]
            if slot >= 0
            else torch.zeros(sequence, dtype=torch.float64)
        )
        raw_iou = (
            raw_pair_iou[:, target]
            if slot >= 0
            else torch.zeros(sequence, dtype=torch.float64)
        )
        pred_masks = (
            masks[:, slot]
            if slot >= 0
            else torch.zeros_like(target_masks)
        )
        pred_visible = pred_masks.flatten(1).any(dim=1)
        visible_ious = current_iou[target_visible]
        gt_visible_values.extend(float(value) for value in visible_ious)
        positive = target_visible & pred_visible
        positive_values.extend(float(value) for value in current_iou[positive])
        differences = current_iou[target_visible] - raw_iou[target_visible]
        improvements += int((differences > config.improvement_epsilon).sum())
        worsened += int((differences < -config.improvement_epsilon).sum())
        unchanged += int(
            (differences.abs() <= config.improvement_epsilon).sum()
        )

        successful = (
            current_iou >= float(config.identity_iou_threshold)
        ) & target_visible & pred_visible
        frame_idtp += int(successful.sum())
        frame_idfp += int((pred_visible & ~successful).sum())
        frame_idfn += int((target_visible & ~successful).sum())
        intersection = (pred_masks & target_masks).flatten(1).sum(dim=1)
        pred_area = pred_masks.flatten(1).sum(dim=1)
        target_area = target_masks.flatten(1).sum(dim=1)
        pixel_idtp += int(intersection.sum())
        pixel_idfp += int((pred_area - intersection).sum())
        pixel_idfn += int((target_area - intersection).sum())

        switches = _identity_switches(
            masks,
            target_masks,
            compatibility[:, target],
            threshold=float(config.identity_iou_threshold),
        )
        reentries, reentry_success = _reentry_metrics(
            target_visible,
            current_iou,
            min_gap=config.reentry_min_gap,
            window=config.reentry_window,
            threshold=config.reentry_iou_threshold,
        )
        total_switches += switches
        total_reentries += reentries
        successful_reentries += reentry_success
        union_pixels = int((pred_masks | target_masks).sum())
        object_rows.append(
            {
                "scene_id": scene_id,
                "clip": clip_name,
                "variant": variant,
                "gt_instance_id": int(ground_truth.instance_ids[target]),
                "gt_label": str(ground_truth.labels[target]),
                "slot": int(slot),
                "track_id": int(track_ids[slot]) if slot >= 0 else -1,
                "track_prompt": str(track_prompts[slot]) if slot >= 0 else "",
                "matched": int(slot >= 0),
                "gt_visible_frames": int(target_visible.sum()),
                "pred_visible_frames": int(pred_visible.sum()),
                "mean_frame_iou": _safe_mean(visible_ious),
                "positive_iou": _safe_mean(current_iou[positive]),
                "global_pixel_iou": (
                    float((pred_masks & target_masks).sum()) / union_pixels
                    if union_pixels
                    else 0.0
                ),
                "tracking_recall_025": _threshold_rate(
                    visible_ious,
                    0.25,
                ),
                "tracking_recall_050": _threshold_rate(
                    visible_ious,
                    0.50,
                ),
                "id_switches": int(switches),
                "reentry_events": int(reentries),
                "reentry_successes": int(reentry_success),
                "reentry_success_rate": (
                    reentry_success / reentries if reentries else float("nan")
                ),
            }
        )
        for frame, frame_index in enumerate(frame_indices):
            difference = float(current_iou[frame] - raw_iou[frame])
            frame_rows.append(
                {
                    "scene_id": scene_id,
                    "clip": clip_name,
                    "variant": variant,
                    "sequence_index": int(frame),
                    "frame_index": int(frame_index),
                    "gt_instance_id": int(ground_truth.instance_ids[target]),
                    "gt_label": str(ground_truth.labels[target]),
                    "slot": int(slot),
                    "track_id": int(track_ids[slot]) if slot >= 0 else -1,
                    "gt_visible": int(target_visible[frame]),
                    "pred_visible": int(pred_visible[frame]),
                    "track_score": (
                        float(scores[frame, slot]) if slot >= 0 else 0.0
                    ),
                    "iou": float(current_iou[frame]),
                    "raw_iou": float(raw_iou[frame]),
                    "iou_delta_vs_raw": difference,
                    "improved_vs_raw": int(
                        bool(target_visible[frame])
                        and difference > config.improvement_epsilon
                    ),
                    "worsened_vs_raw": int(
                        bool(target_visible[frame])
                        and difference < -config.improvement_epsilon
                    ),
                }
            )

    matched_slots = set(slot_to_target)
    for slot in range(tracks):
        if slot not in matched_slots:
            pixels = int(masks[:, slot].sum())
            pixel_idfp += pixels
            frame_idfp += int(masks[:, slot].flatten(1).any(dim=1).sum())
    global_scores = global_iou_matrix(masks, gt)
    fragmentation_count = 0
    for target in range(objects):
        compatible_scores = global_scores[:, target][compatibility[:, target]]
        count = int(
            (compatible_scores >= float(config.fragmentation_iou_threshold)).sum()
        )
        fragmentation_count += max(0, count - 1)
    merge_error_count = 0
    for slot in range(tracks):
        compatible_scores = global_scores[slot][compatibility[slot]]
        count = int(
            (compatible_scores >= float(config.fragmentation_iou_threshold)).sum()
        )
        merge_error_count += max(0, count - 1)
    denominator = improvements + worsened + unchanged
    summary = {
        "scene_id": scene_id,
        "clip": clip_name,
        "variant": variant,
        "frame_occurrences": int(sequence),
        "eligible_gt_objects": int(objects),
        "all_visible_gt_objects": int(
            len(ground_truth.all_visible_instance_ids)
        ),
        "matched_tracks": int(len(slot_to_target)),
        "unmatched_tracks": int(tracks - len(slot_to_target)),
        "unmatched_gt_objects": int(objects - len(target_to_slot)),
        "mean_frame_iou": _safe_mean_list(gt_visible_values),
        "positive_iou": _safe_mean_list(positive_values),
        "tracking_recall_025": _threshold_rate_list(gt_visible_values, 0.25),
        "tracking_recall_050": _threshold_rate_list(gt_visible_values, 0.50),
        "frame_idf1": _f1(frame_idtp, frame_idfp, frame_idfn),
        "pixel_idf1": _f1(pixel_idtp, pixel_idfp, pixel_idfn),
        "frame_idtp": int(frame_idtp),
        "frame_idfp": int(frame_idfp),
        "frame_idfn": int(frame_idfn),
        "pixel_idtp": int(pixel_idtp),
        "pixel_idfp": int(pixel_idfp),
        "pixel_idfn": int(pixel_idfn),
        "id_switches": int(total_switches),
        "fragmentation_count": int(fragmentation_count),
        "merge_error_count": int(merge_error_count),
        "reentry_events": int(total_reentries),
        "reentry_successes": int(successful_reentries),
        "reentry_success_rate": (
            successful_reentries / total_reentries
            if total_reentries
            else float("nan")
        ),
        "improved_frame_ratio_vs_raw": (
            improvements / denominator if denominator else float("nan")
        ),
        "worsened_frame_ratio_vs_raw": (
            worsened / denominator if denominator else float("nan")
        ),
        "unchanged_frame_ratio_vs_raw": (
            unchanged / denominator if denominator else float("nan")
        ),
    }
    return {"summary": summary, "objects": object_rows, "frames": frame_rows}


def _pair_frame_ious(
    masks: torch.Tensor,
    gt: torch.Tensor,
    target_to_slot: Mapping[int, int],
) -> torch.Tensor:
    sequence, objects = gt.shape[:2]
    output = torch.zeros(sequence, objects, dtype=torch.float64)
    for target, slot in target_to_slot.items():
        left = masks[:, slot]
        right = gt[:, target]
        intersection = (left & right).flatten(1).sum(dim=1).double()
        union = (left | right).flatten(1).sum(dim=1).double()
        output[:, target] = torch.where(
            union > 0,
            intersection / union.clamp_min(1),
            torch.zeros_like(union),
        )
    return output


def _identity_switches(
    masks: torch.Tensor,
    target: torch.Tensor,
    compatible_slots: torch.Tensor,
    *,
    threshold: float,
) -> int:
    last_slot: int | None = None
    switches = 0
    for frame in range(masks.shape[0]):
        if not bool(target[frame].any()):
            continue
        best_slot = None
        best_iou = float(threshold)
        for slot in range(masks.shape[1]):
            if not bool(compatible_slots[slot]):
                continue
            union = int((masks[frame, slot] | target[frame]).sum())
            iou = (
                float((masks[frame, slot] & target[frame]).sum()) / union
                if union
                else 0.0
            )
            if iou >= best_iou:
                if iou > best_iou or best_slot is None or slot < best_slot:
                    best_iou = iou
                    best_slot = slot
        if best_slot is None:
            continue
        if last_slot is not None and best_slot != last_slot:
            switches += 1
        last_slot = best_slot
    return switches


def _reentry_metrics(
    visible: torch.Tensor,
    iou: torch.Tensor,
    *,
    min_gap: int,
    window: int,
    threshold: float,
) -> tuple[int, int]:
    events = successes = 0
    last_visible = -1
    for frame in range(len(visible)):
        if not bool(visible[frame]):
            continue
        if last_visible >= 0 and frame - last_visible - 1 >= int(min_gap):
            events += 1
            stop = min(len(visible), frame + max(1, int(window)))
            if any(
                bool(visible[index]) and float(iou[index]) >= float(threshold)
                for index in range(frame, stop)
            ):
                successes += 1
        last_visible = frame
    return events, successes


def _hungarian_minimum(cost: list[list[float]]) -> list[int]:
    size = len(cost)
    if any(len(row) != size for row in cost):
        raise ValueError("Hungarian helper expects a square matrix.")
    u = [0.0] * (size + 1)
    v = [0.0] * (size + 1)
    p = [0] * (size + 1)
    way = [0] * (size + 1)
    for row in range(1, size + 1):
        p[0] = row
        column0 = 0
        minimum = [float("inf")] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[column0] = True
            row0 = p[column0]
            delta = float("inf")
            column1 = 0
            for column in range(1, size + 1):
                if used[column]:
                    continue
                current = cost[row0 - 1][column - 1] - u[row0] - v[column]
                if current < minimum[column]:
                    minimum[column] = current
                    way[column] = column0
                if minimum[column] < delta:
                    delta = minimum[column]
                    column1 = column
            for column in range(size + 1):
                if used[column]:
                    u[p[column]] += delta
                    v[column] -= delta
                else:
                    minimum[column] -= delta
            column0 = column1
            if p[column0] == 0:
                break
        while True:
            column1 = way[column0]
            p[column0] = p[column1]
            column0 = column1
            if column0 == 0:
                break
    assignment = [-1] * size
    for column in range(1, size + 1):
        if p[column]:
            assignment[p[column] - 1] = column - 1
    return assignment


def _normalize_label(value: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", str(value).lower())
    normalized = []
    for token in tokens:
        if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        normalized.append(token)
    return " ".join(normalized)


def _safe_mean(values: torch.Tensor) -> float:
    return float(values.double().mean()) if values.numel() else float("nan")


def _safe_mean_list(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _threshold_rate(values: torch.Tensor, threshold: float) -> float:
    return (
        float((values >= float(threshold)).double().mean())
        if values.numel()
        else float("nan")
    )


def _threshold_rate_list(values: Sequence[float], threshold: float) -> float:
    return (
        sum(float(value) >= float(threshold) for value in values) / len(values)
        if values
        else float("nan")
    )


def _f1(true_positive: int, false_positive: int, false_negative: int) -> float:
    denominator = 2 * true_positive + false_positive + false_negative
    return 2.0 * true_positive / denominator if denominator else float("nan")
