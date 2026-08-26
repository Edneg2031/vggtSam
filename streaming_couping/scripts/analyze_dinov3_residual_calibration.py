#!/usr/bin/env python3
"""Diagnose whether frozen DINOv3 residuals are safe and useful.

This is an analysis-only companion to ``run_dinov3_object_geometry.py``.  It
does not train a model and does not use its measurements to select a model.
For every frozen learned branch it compares the raw point error with the
candidate point error, grouped by residual magnitude and by conditions that
matter for a semantic map.  The output is intended to decide whether the next
method should keep learning XYZ residuals or move DINOv3 to correspondence and
world-space object memory.

The analysis reads GT because it is an offline diagnostic.  It never feeds GT
to a model, gate, or map construction path.
"""

from __future__ import annotations

from collections import defaultdict
import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.nn import functional as F

from streaming_couping.src.dinov3_object_geometry import (
    ObjectConditionedResidualHead,
    apply_similarity,
)
from streaming_couping.src.learned_pose.config import load_learned_pose_config
from streaming_couping.scripts.run_dinov3_object_geometry import (
    BRANCHES,
    LEARNED_BRANCHES,
    _load_clip_data,
    _load_ground_truth_bundle,
    _predict_clip_native,
    _resolve_device,
    _validate_model_compatibility,
)


REVISION = "dinov3_residual_calibration_diagnostic_r1"
DEFAULT_STORAGE_ROOT = "/data184/open_source/vggtSam"
DEFAULT_PROTOCOL = "outputs/streaming_couping_multiclip/protocol.yaml"
DEFAULT_FEATURE_DIR = "outputs/streaming_couping_dinov3_object_features"
DEFAULT_MODEL_PATH = "outputs/streaming_couping_dinov3_object_geometry/models.pt"
DEFAULT_OUTPUT_DIR = "outputs/streaming_couping_dinov3_residual_calibration"

CORRECTION_BINS: tuple[tuple[float, float, str], ...] = (
    (0.0, 0.005, "0-5mm"),
    (0.005, 0.01, "5-10mm"),
    (0.01, 0.02, "10-20mm"),
    (0.02, 0.04, "20-40mm"),
    (0.04, 0.08, "40-80mm"),
    (0.08, 0.16, "80-160mm"),
    (0.16, float("inf"), ">160mm"),
)


def main() -> None:
    args = _parse_args()
    storage_root = Path(
        os.environ.get("VGGT_SAM_STORAGE_ROOT", DEFAULT_STORAGE_ROOT)
    ).expanduser().resolve()
    protocol = _resolve_path(args.protocol, storage_root, DEFAULT_PROTOCOL)
    feature_dir = _resolve_path(args.feature_dir, storage_root, DEFAULT_FEATURE_DIR)
    model_path = _resolve_path(args.model_path, storage_root, DEFAULT_MODEL_PATH)
    output_dir = _resolve_path(args.output_dir, storage_root, DEFAULT_OUTPUT_DIR)

    if not protocol.is_file():
        raise FileNotFoundError(f"Multi-scene protocol is missing: {protocol}")
    if not feature_dir.is_dir():
        raise FileNotFoundError(f"DINOv3 feature cache directory is missing: {feature_dir}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Frozen geometry model is missing: {model_path}")

    config = load_learned_pose_config(protocol)
    validation_clips = tuple(
        clip for clip in config.clips if clip.split == "validation"
    )
    test_clips = tuple(clip for clip in config.clips if clip.split == "test")
    if not validation_clips or not test_clips:
        raise ValueError(
            "Residual calibration requires validation and test clips; got "
            f"validation={len(validation_clips)} test={len(test_clips)}."
        )

    model_payload = torch.load(model_path, map_location="cpu", weights_only=False)
    if not isinstance(model_payload, Mapping):
        raise ValueError(f"Frozen model payload is not a mapping: {model_path}")
    model_spec = model_payload.get("model")
    states = model_payload.get("branches")
    if not isinstance(model_spec, Mapping) or not isinstance(states, Mapping):
        raise ValueError("Frozen model payload lacks model specification or branches.")
    for branch in LEARNED_BRANCHES:
        if branch not in states or not isinstance(states[branch], Mapping):
            raise ValueError(f"Frozen model payload lacks branch {branch!r}.")

    device = _resolve_device(args.device)
    print("DINOv3 RESIDUAL CALIBRATION DIAGNOSTIC")
    print(f"protocol={protocol}")
    print(f"feature_dir={feature_dir}")
    print(f"frozen_model={model_path}")
    print(
        "models are not retrained; StreamVGGT/SAM3/DINOv3 caches are reused; "
        "GT is opened only for offline diagnosis"
    )
    print(f"device={device} output={output_dir}")

    model_states = {
        branch: states[branch]["state_dict"] for branch in LEARNED_BRANCHES
    }
    validation_data = [
        _load_clip_data(config, clip, feature_dir, confidence_threshold=0.30)
        for clip in validation_clips
    ]
    test_data = [
        _load_clip_data(config, clip, feature_dir, confidence_threshold=0.30)
        for clip in test_clips
    ]
    _validate_model_compatibility((*validation_data, *test_data))

    condition_rows: list[dict[str, Any]] = []
    bin_rows: list[dict[str, Any]] = []
    object_rows: list[dict[str, Any]] = []
    aggregate_values: dict[tuple[str, str, str], list[list[torch.Tensor]]] = {}
    clip_summaries: list[dict[str, Any]] = []

    for split, split_data in (
        ("validation", validation_data),
        ("test", test_data),
    ):
        for data in split_data:
            ground_truth = _load_ground_truth_bundle(data)
            predictions = _predict_aligned_predictions(
                data,
                model_states=model_states,
                model_spec=model_spec,
                device=device,
            )
            result = _analyze_clip(
                data,
                ground_truth=ground_truth,
                predictions=predictions,
                split=split,
            )
            condition_rows.extend(result["condition_rows"])
            bin_rows.extend(result["bin_rows"])
            object_rows.extend(result["object_rows"])
            clip_summaries.append(result["summary"])
            for key, tensors in result["aggregate_values"].items():
                aggregate_values.setdefault(key, [[], [], [], []])
                for index, values in enumerate(tensors):
                    aggregate_values[key][index].extend(values)
            _print_clip_summary(result["summary"])

    aggregate_rows: list[dict[str, Any]] = []
    for (split, branch, condition), tensors in sorted(aggregate_values.items()):
        values = tuple(_concat(value) for value in tensors)
        row = _summary_row(
            split=split,
            clip="ALL",
            branch=branch,
            condition=condition,
            raw_error=values[0],
            candidate_error=values[1],
            correction=values[2],
            direction=values[3],
        )
        aggregate_rows.append(row)

    interpretation = _interpret_results(aggregate_rows, object_rows)
    summary = {
        "schema": 1,
        "revision": REVISION,
        "protocol": str(protocol),
        "feature_dir": str(feature_dir),
        "frozen_model": str(model_path),
        "branches": list(BRANCHES),
        "validation_scenes": [clip.scene_id for clip in validation_clips],
        "test_scenes": [clip.scene_id for clip in test_clips],
        "diagnostic_only": 1,
        "clip_summaries": clip_summaries,
        "aggregate_condition_summaries": aggregate_rows,
        "interpretation": interpretation,
        "data_policy": {
            "streamvggt_rerun": 0,
            "sam_rerun": 0,
            "dinov3_rerun": 0,
            "parameters_updated": 0,
            "gt_used_for_model_input": 0,
            "gt_used_for_diagnostic": 1,
            "recommended_next_role_for_dino": (
                "object_correspondence_and_world_space_memory"
            ),
        },
        "outputs": {},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output_dir / "summary.json",
        "condition_csv": output_dir / "condition_summary.csv",
        "correction_bins_csv": output_dir / "correction_bins.csv",
        "object_csv": output_dir / "object_calibration.csv",
        "copyable": output_dir / "copyable_result.txt",
    }
    _write_json(paths["summary"], summary)
    _write_csv(paths["condition_csv"], condition_rows + aggregate_rows)
    _write_csv(paths["correction_bins_csv"], bin_rows)
    _write_csv(paths["object_csv"], object_rows)
    summary["outputs"] = {key: str(value) for key, value in paths.items()}
    _write_json(paths["summary"], summary)
    _write_copyable(paths["copyable"], summary)
    print(f"summary={paths['summary']}")
    print(f"condition_summary={paths['condition_csv']}")
    print(f"correction_bins={paths['correction_bins_csv']}")
    print(f"object_calibration={paths['object_csv']}")
    print("decision=DIAGNOSTIC_ONLY")


def _analyze_clip(
    data: Any,
    *,
    ground_truth: Any,
    predictions: Mapping[str, torch.Tensor],
    split: str,
) -> dict[str, Any]:
    gt_masks = ground_truth.stream_masks.detach().bool().cpu()
    support = data.support.detach().bool().cpu()
    sam_union = data.masks.detach().bool().cpu().any(dim=1)
    slot_map, slot_present = _best_slot_map(data.masks, data.scores)
    track_age = _track_age(data.masks)
    pixel_track_age = _pixel_track_age(track_age, slot_map, slot_present)
    object_rows: list[dict[str, Any]] = []
    condition_rows: list[dict[str, Any]] = []
    bin_rows: list[dict[str, Any]] = []
    aggregate_values: dict[tuple[str, str, str], list[list[torch.Tensor]]] = {}

    for branch in LEARNED_BRANCHES:
        raw = predictions["raw"]
        candidate = predictions[branch]
        correction = candidate - raw
        raw_error = torch.linalg.vector_norm(raw - data.target_metric, dim=-1)
        candidate_error = torch.linalg.vector_norm(
            candidate - data.target_metric, dim=-1
        )
        correction_norm = torch.linalg.vector_norm(correction, dim=-1)
        target_residual = data.target_metric - raw
        direction = _direction_cosine(correction, target_residual)

        groups: dict[str, list[list[torch.Tensor]]] = defaultdict(
            lambda: [[], [], [], []]
        )
        feature_groups: list[list[torch.Tensor]] = [[], [], [], []]
        feature_valid_values: list[torch.Tensor] = []
        feature_invalid_values: list[torch.Tensor] = []
        age_groups: dict[str, list[list[torch.Tensor]]] = defaultdict(
            lambda: [[], [], [], []]
        )

        for object_index in range(int(gt_masks.shape[1])):
            gt_object = gt_masks[:, object_index]
            valid = (
                gt_object
                & support
                & torch.isfinite(raw_error)
                & torch.isfinite(candidate_error)
                & torch.isfinite(correction_norm)
            )
            overlap = valid & sam_union
            object_boundary = _boundary_mask(gt_object)
            boundary = valid & object_boundary
            interior = valid & gt_object & ~object_boundary
            _append_group(
                groups["all_gt_object"],
                raw_error,
                candidate_error,
                correction_norm,
                direction,
                valid,
            )
            _append_group(
                groups["sam_gt_overlap"],
                raw_error,
                candidate_error,
                correction_norm,
                direction,
                overlap,
            )
            _append_group(
                groups["boundary"],
                raw_error,
                candidate_error,
                correction_norm,
                direction,
                boundary,
            )
            _append_group(
                groups["interior"],
                raw_error,
                candidate_error,
                correction_norm,
                direction,
                interior,
            )

            object_feature_valid = torch.empty(0, dtype=torch.bool)
            feature_overlap = overlap & slot_present
            _append_group(
                feature_groups,
                raw_error,
                candidate_error,
                correction_norm,
                direction,
                feature_overlap,
            )
            if bool(feature_overlap.any()):
                frame_slot = slot_map.clone()
                frame_slot[~slot_present] = -1
                for label, age_mask in _age_masks(
                    track_age,
                    frame_slot,
                    slot_present,
                ).items():
                    _append_group(
                        age_groups[label],
                        raw_error,
                        candidate_error,
                        correction_norm,
                        direction,
                        feature_overlap & age_mask,
                    )
                if branch != "geometry_only":
                    feature_valid = _feature_valid_map(
                        data,
                        branch,
                        frame_slot,
                        slot_present,
                    )
                    object_feature_valid = feature_valid[feature_overlap]
                    feature_valid_values.append(object_feature_valid)
                    feature_invalid_values.append(
                        (~object_feature_valid).bool()
                    )

            object_rows.append(
                _object_row(
                    data=data,
                    split=split,
                    branch=branch,
                    object_index=object_index,
                    label=str(ground_truth.instances.labels[object_index]),
                    valid=valid,
                    overlap=overlap,
                    boundary=boundary,
                    raw_error=raw_error,
                    candidate_error=candidate_error,
                    correction_norm=correction_norm,
                    direction=direction,
                    feature_valid=object_feature_valid,
                    track_age=pixel_track_age[overlap]
                    if bool(overlap.any())
                    else torch.empty(0),
                )
            )

        if feature_valid_values and branch != "geometry_only":
            feature_valid = torch.cat(feature_valid_values)
            feature_invalid = torch.cat(feature_invalid_values)
            _append_group(
                groups["dino_valid"],
                *_condition_values(feature_groups),
                feature_valid,
            )
            _append_group(
                groups["dino_invalid"],
                *_condition_values(feature_groups),
                feature_invalid,
            )

        for condition, tensors in groups.items():
            values = tuple(_concat(value) for value in tensors)
            if not values[0].numel():
                continue
            row = _summary_row(
                split=split,
                clip=data.name,
                branch=branch,
                condition=condition,
                raw_error=values[0],
                candidate_error=values[1],
                correction=values[2],
                direction=values[3],
            )
            condition_rows.append(row)
            aggregate_values.setdefault((split, branch, condition), [[], [], [], []])
            for index, value in enumerate(values):
                aggregate_values[(split, branch, condition)][index].append(value)

        for condition, tensors in age_groups.items():
            values = tuple(_concat(value) for value in tensors)
            if not values[0].numel():
                continue
            row = _summary_row(
                split=split,
                clip=data.name,
                branch=branch,
                condition=condition,
                raw_error=values[0],
                candidate_error=values[1],
                correction=values[2],
                direction=values[3],
            )
            condition_rows.append(row)
            aggregate_values.setdefault((split, branch, condition), [[], [], [], []])
            for index, value in enumerate(values):
                aggregate_values[(split, branch, condition)][index].append(value)

        for scope in ("all_gt_object", "sam_gt_overlap"):
            values = tuple(_concat(value) for value in groups[scope])
            if not values[0].numel():
                continue
            for low, high, label in CORRECTION_BINS:
                selected = (values[2] >= low) & (values[2] < high)
                if not bool(selected.any()):
                    continue
                bin_rows.append(
                    {
                        "split": split,
                        "clip": data.name,
                        "branch": branch,
                        "scope": scope,
                        "correction_bin": label,
                        "lower_m": low,
                        "upper_m": high if math.isfinite(high) else "inf",
                        **_summary_fields(
                            values[0][selected],
                            values[1][selected],
                            values[2][selected],
                            values[3][selected],
                        ),
                    }
                )

    summary_rows = [
        row
        for row in condition_rows
        if row["condition"] in {"all_gt_object", "sam_gt_overlap"}
    ]
    return {
        "summary": {
            "split": split,
            "clip": data.name,
            "scene_id": data.scene_id,
            "conditions": summary_rows,
        },
        "condition_rows": condition_rows,
        "bin_rows": bin_rows,
        "object_rows": object_rows,
        "aggregate_values": aggregate_values,
    }


def _predict_aligned_predictions(
    data: Any,
    *,
    model_states: Mapping[str, Mapping[str, torch.Tensor]],
    model_spec: Mapping[str, Any],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    predictions: dict[str, torch.Tensor] = {
        "raw": apply_similarity(
            data.raw_native,
            scale=data.scale,
            rotation=data.rotation,
            translation=data.translation,
        ).float()
    }
    for branch in LEARNED_BRANCHES:
        model = ObjectConditionedResidualHead(**dict(model_spec)).to(device)
        model.load_state_dict(model_states[branch], strict=True)
        model.eval()
        native = _predict_clip_native(model, data, branch, device)
        predictions[branch] = apply_similarity(
            native,
            scale=data.scale,
            rotation=data.rotation,
            translation=data.translation,
        ).float()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return predictions


def _object_row(
    *,
    data: Any,
    split: str,
    branch: str,
    object_index: int,
    label: str,
    valid: torch.Tensor,
    overlap: torch.Tensor,
    boundary: torch.Tensor,
    raw_error: torch.Tensor,
    candidate_error: torch.Tensor,
    correction_norm: torch.Tensor,
    direction: torch.Tensor,
    feature_valid: torch.Tensor,
    track_age: torch.Tensor,
) -> dict[str, Any]:
    all_values = _values_for_mask(
        raw_error, candidate_error, correction_norm, direction, valid
    )
    overlap_values = _values_for_mask(
        raw_error, candidate_error, correction_norm, direction, overlap
    )
    boundary_values = _values_for_mask(
        raw_error, candidate_error, correction_norm, direction, boundary
    )
    row: dict[str, Any] = {
        "split": split,
        "clip": data.name,
        "scene_id": data.scene_id,
        "branch": branch,
        "object_index": int(object_index),
        "gt_label": label,
        "all_points": int(all_values[0].numel()),
        "sam_overlap_points": int(overlap_values[0].numel()),
        "sam_coverage": float(overlap_values[0].numel())
        / max(1, int(all_values[0].numel())),
    }
    row.update({f"all_{key}": value for key, value in _summary_fields(*all_values).items()})
    row.update(
        {
            f"sam_overlap_{key}": value
            for key, value in _summary_fields(*overlap_values).items()
        }
    )
    row.update(
        {
            f"boundary_{key}": value
            for key, value in _summary_fields(*boundary_values).items()
        }
    )
    row["dino_valid_fraction"] = (
        float(feature_valid.float().mean()) if feature_valid.numel() else float("nan")
    )
    row["track_age_median"] = (
        float(track_age.float().median()) if track_age.numel() else float("nan")
    )
    return row


def _summary_row(
    *,
    split: str,
    clip: str,
    branch: str,
    condition: str,
    raw_error: torch.Tensor,
    candidate_error: torch.Tensor,
    correction: torch.Tensor,
    direction: torch.Tensor,
) -> dict[str, Any]:
    return {
        "split": split,
        "clip": clip,
        "branch": branch,
        "condition": condition,
        **_summary_fields(raw_error, candidate_error, correction, direction),
    }


def _summary_fields(
    raw_error: torch.Tensor,
    candidate_error: torch.Tensor,
    correction: torch.Tensor,
    direction: torch.Tensor,
) -> dict[str, Any]:
    raw_error = raw_error.detach().float().cpu().reshape(-1)
    candidate_error = candidate_error.detach().float().cpu().reshape(-1)
    correction = correction.detach().float().cpu().reshape(-1)
    direction = direction.detach().float().cpu().reshape(-1)
    finite = (
        torch.isfinite(raw_error)
        & torch.isfinite(candidate_error)
        & torch.isfinite(correction)
    )
    raw_error = raw_error[finite]
    candidate_error = candidate_error[finite]
    correction = correction[finite]
    direction = direction[finite]
    if not raw_error.numel():
        return {
            "count": 0,
            "raw_rmse_m": float("nan"),
            "candidate_rmse_m": float("nan"),
            "raw_mean_error_m": float("nan"),
            "candidate_mean_error_m": float("nan"),
            "mean_gain_m": float("nan"),
            "gain_percent": float("nan"),
            "improved_fraction": float("nan"),
            "correction_mean_m": float("nan"),
            "correction_median_m": float("nan"),
            "correction_q90_m": float("nan"),
            "direction_cosine_mean": float("nan"),
            "correction_gain_pearson": float("nan"),
        }
    gain = raw_error - candidate_error
    finite_direction = torch.isfinite(direction)
    return {
        "count": int(raw_error.numel()),
        "raw_rmse_m": _rmse(raw_error),
        "candidate_rmse_m": _rmse(candidate_error),
        "raw_mean_error_m": float(raw_error.mean()),
        "candidate_mean_error_m": float(candidate_error.mean()),
        "mean_gain_m": float(gain.mean()),
        "gain_percent": 100.0 * float(gain.mean()) / max(float(raw_error.mean()), 1e-12),
        "improved_fraction": float((gain > 0.0).float().mean()),
        "correction_mean_m": float(correction.mean()),
        "correction_median_m": float(correction.median()),
        "correction_q90_m": float(torch.quantile(correction, 0.90)),
        "direction_cosine_mean": (
            float(direction[finite_direction].mean())
            if bool(finite_direction.any())
            else float("nan")
        ),
        "correction_gain_pearson": _pearson(correction, gain),
    }


def _direction_cosine(
    correction: torch.Tensor,
    target_residual: torch.Tensor,
) -> torch.Tensor:
    left = correction.float()
    right = target_residual.float()
    denominator = torch.linalg.vector_norm(left, dim=-1) * torch.linalg.vector_norm(
        right, dim=-1
    )
    numerator = (left * right).sum(dim=-1)
    output = numerator / denominator.clamp_min(1e-12)
    return torch.where(denominator > 1e-8, output, torch.full_like(output, float("nan")))


def _best_slot_map(
    masks: torch.Tensor,
    scores: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = masks.detach().bool().cpu()
    score_values = torch.nan_to_num(
        scores.detach().float().cpu(), nan=float("-inf")
    )
    masked_scores = score_values[:, :, None, None].expand_as(values).masked_fill(
        ~values, float("-inf")
    )
    best_score, best_slot = masked_scores.max(dim=1)
    present = values.any(dim=1) & torch.isfinite(best_score)
    return best_slot.long(), present.bool()


def _track_age(masks: torch.Tensor) -> torch.Tensor:
    values = masks.detach().bool().cpu()
    sequence, tracks = values.shape[:2]
    output = torch.full((sequence, tracks), -1, dtype=torch.long)
    for slot in range(tracks):
        visible = values[:, slot].flatten(1).any(dim=1)
        positions = torch.nonzero(visible, as_tuple=False).flatten()
        if not positions.numel():
            continue
        birth = int(positions[0])
        output[:, slot] = torch.arange(sequence) - birth
    return output


def _feature_valid_map(
    data: Any,
    branch: str,
    slot_map: torch.Tensor,
    slot_present: torch.Tensor,
) -> torch.Tensor:
    if branch == "single_view_dino":
        values = data.single_valid
    elif branch == "persistent_dino":
        values = data.persistent_valid
    elif branch == "shuffled_persistent_dino":
        values = data.shuffled_valid
    else:
        return slot_present.clone()
    output = torch.zeros_like(slot_present)
    for frame in range(int(slot_map.shape[0])):
        current = slot_map[frame].clamp_min(0)
        output[frame] = values[frame].index_select(0, current.reshape(-1)).reshape(
            current.shape
        ) & slot_present[frame]
    return output.bool()


def _age_masks(
    ages: torch.Tensor,
    slot_map: torch.Tensor,
    slot_present: torch.Tensor,
) -> dict[str, torch.Tensor]:
    pixel_age = _pixel_track_age(ages, slot_map, slot_present)
    return {
        "track_age_0_1": slot_present & (pixel_age >= 0) & (pixel_age < 2),
        "track_age_2_4": slot_present & (pixel_age >= 2) & (pixel_age < 5),
        "track_age_5_plus": slot_present & (pixel_age >= 5),
    }


def _pixel_track_age(
    ages: torch.Tensor,
    slot_map: torch.Tensor,
    slot_present: torch.Tensor,
) -> torch.Tensor:
    pixel_age = torch.full_like(slot_map, -1)
    for frame in range(int(slot_map.shape[0])):
        current = slot_map[frame].clamp_min(0)
        pixel_age[frame] = ages[frame].index_select(0, current.reshape(-1)).reshape(
            current.shape
        )
    return pixel_age.where(slot_present, torch.full_like(pixel_age, -1))


def _boundary_mask(mask: torch.Tensor) -> torch.Tensor:
    values = mask.float().unsqueeze(1)
    eroded = -F.max_pool2d(-values, kernel_size=3, stride=1, padding=1)
    return mask.bool() & (eroded[:, 0] < 0.5)


def _append_group(
    group: list[list[torch.Tensor]],
    raw_error: torch.Tensor,
    candidate_error: torch.Tensor,
    correction: torch.Tensor,
    direction: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    if not bool(mask.any()):
        return
    group[0].append(raw_error[mask].detach().float().cpu())
    group[1].append(candidate_error[mask].detach().float().cpu())
    group[2].append(correction[mask].detach().float().cpu())
    group[3].append(direction[mask].detach().float().cpu())


def _condition_values(
    group: list[list[torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return tuple(_concat(value) for value in group)  # type: ignore[return-value]


def _values_for_mask(
    raw_error: torch.Tensor,
    candidate_error: torch.Tensor,
    correction: torch.Tensor,
    direction: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        raw_error[mask].detach().float().cpu(),
        candidate_error[mask].detach().float().cpu(),
        correction[mask].detach().float().cpu(),
        direction[mask].detach().float().cpu(),
    )


def _concat(values: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.cat(list(values)) if values else torch.empty(0)


def _rmse(values: torch.Tensor) -> float:
    return float(torch.sqrt(values.square().mean())) if values.numel() else float("nan")


def _pearson(left: torch.Tensor, right: torch.Tensor) -> float:
    finite = torch.isfinite(left) & torch.isfinite(right)
    if int(finite.sum()) < 2:
        return float("nan")
    x = left[finite].float()
    y = right[finite].float()
    x = x - x.mean()
    y = y - y.mean()
    denominator = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    if float(denominator) <= 1e-12:
        return float("nan")
    return float((x * y).sum() / denominator)


def _interpret_results(
    aggregate_rows: Sequence[Mapping[str, Any]],
    object_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    def find(split: str, branch: str, condition: str) -> Mapping[str, Any] | None:
        for row in aggregate_rows:
            if (
                row["split"] == split
                and row["branch"] == branch
                and row["condition"] == condition
            ):
                return row
        return None

    persistent = find("test", "persistent_dino", "sam_gt_overlap")
    single = find("test", "single_view_dino", "sam_gt_overlap")
    shuffled = find("test", "shuffled_persistent_dino", "sam_gt_overlap")
    identity_signal = bool(
        persistent
        and single
        and shuffled
        and float(persistent["candidate_rmse_m"])
        < float(single["candidate_rmse_m"])
        and float(persistent["candidate_rmse_m"])
        < float(shuffled["candidate_rmse_m"])
    )
    object_gain = {}
    for branch in LEARNED_BRANCHES:
        rows = [
            row
            for row in object_rows
            if row["split"] == "test" and row["branch"] == branch
        ]
        gains = [
            float(row["sam_overlap_mean_gain_m"])
            for row in rows
            if math.isfinite(float(row["sam_overlap_mean_gain_m"]))
        ]
        object_gain[branch] = {
            "objects": len(gains),
            "positive_gain_objects": sum(value > 0.0 for value in gains),
            "positive_gain_fraction": (
                sum(value > 0.0 for value in gains) / len(gains) if gains else float("nan")
            ),
            "mean_object_gain_m": sum(gains) / len(gains) if gains else float("nan"),
        }
    return {
        "persistent_beats_single_and_shuffled_on_test_overlap": int(identity_signal),
        "test_object_gain_by_branch": object_gain,
        "interpretation": (
            "A positive mean_gain means the learned correction improves raw on "
            "the same GT-supported points. Use correction_bins.csv to decide "
            "whether a magnitude-based trust model is calibratable; use the "
            "condition rows to see whether gains are confined to boundaries, "
            "young tracks, or valid DINO observations. This file is diagnostic, "
            "not a deployment decision."
        ),
    }


def _print_clip_summary(summary: Mapping[str, Any]) -> None:
    print(f"  {summary['split']} clip={summary['clip']} scene={summary['scene_id']}")
    for row in summary["conditions"]:
        print(
            f"    {row['branch']} condition={row['condition']} "
            f"n={row['count']} raw_rmse={row['raw_rmse_m']:.5f} "
            f"candidate_rmse={row['candidate_rmse_m']:.5f} "
            f"gain={row['mean_gain_m']:.5f} "
            f"improved={row['improved_fraction']:.3f} "
            f"corr_q90={row['correction_q90_m']:.5f}"
        )


def _resolve_path(value: str | None, root: Path, default: str) -> Path:
    path = Path(value).expanduser() if value else root / default
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf8")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_copyable(path: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "===== DINOV3_RESIDUAL_CALIBRATION_BEGIN =====",
        f"revision={summary['revision']}",
        "diagnostic_only=1",
        "branches=" + ",".join(summary["branches"]),
        "recommended_next_role_for_dino=object_correspondence_and_world_space_memory",
        "",
        "split,branch,condition,count,raw_rmse_m,candidate_rmse_m,mean_gain_m,improved_fraction,correction_q90_m,direction_cosine_mean,correction_gain_pearson",
    ]
    for row in summary["aggregate_condition_summaries"]:
        lines.append(
            ",".join(
                str(row.get(key, ""))
                for key in (
                    "split",
                    "branch",
                    "condition",
                    "count",
                    "raw_rmse_m",
                    "candidate_rmse_m",
                    "mean_gain_m",
                    "improved_fraction",
                    "correction_q90_m",
                    "direction_cosine_mean",
                    "correction_gain_pearson",
                )
            )
        )
    lines.extend(
        (
            "",
            "interpretation=" + json.dumps(summary["interpretation"], sort_keys=True),
            f"summary={path.with_name('summary.json')}",
            f"condition_summary={path.with_name('condition_summary.csv')}",
            f"correction_bins={path.with_name('correction_bins.csv')}",
            f"object_calibration={path.with_name('object_calibration.csv')}",
            "===== DINOV3_RESIDUAL_CALIBRATION_END =====",
        )
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        return value.tolist()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol")
    parser.add_argument("--feature-dir")
    parser.add_argument("--model-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


if __name__ == "__main__":
    main()
