"""One-shot visualization and GT accounting for cached V6 tracking masks."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw


_COLORS = (
    (230, 57, 70),
    (42, 157, 143),
    (69, 123, 157),
    (255, 183, 3),
    (131, 56, 236),
    (0, 160, 220),
)
_RESAMPLING = getattr(Image, "Resampling", Image)


def evaluate_cached_tracking(
    *,
    clip_name: str,
    split: str,
    frame_indices: Sequence[int],
    instance_ids: Sequence[int],
    reference_sequence_index: int,
    predicted_masks: torch.Tensor,
    target_masks: torch.Tensor,
    tracking_scores: torch.Tensor,
    identity_valid: torch.Tensor,
    identity_unknown: torch.Tensor,
    identity_mismatch: torch.Tensor,
    associated_valid: torch.Tensor,
    trusted_valid: torch.Tensor,
    segmentation_diagnostics: Iterable[Mapping[str, object]] = (),
    success_iou_threshold: float = 0.50,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Return instance-frame, frame, and compact success summaries.

    A successful visible-object track is a non-reference instance-frame whose
    GT mask is visible and whose final selected V6 mask has IoU at least
    ``success_iou_threshold``. GT-absent frames are reported separately so a
    true negative cannot inflate the visible-object success rate.
    """

    frame_indices = tuple(int(value) for value in frame_indices)
    instance_ids = tuple(int(value) for value in instance_ids)
    expected = (len(frame_indices), len(instance_ids))
    predicted = _mask_tensor(
        predicted_masks,
        expected_prefix=expected,
        name="predicted_masks",
    )
    target = _mask_tensor(
        target_masks,
        expected_prefix=expected,
        name="target_masks",
    )
    if tuple(predicted.shape) != tuple(target.shape):
        raise ValueError(
            "Predicted and GT tracking masks must have identical shapes, got "
            f"{tuple(predicted.shape)} and {tuple(target.shape)}."
        )
    scores = _matrix_tensor(
        tracking_scores,
        expected=expected,
        name="tracking_scores",
        dtype=torch.float32,
    )
    matched = _matrix_tensor(
        identity_valid,
        expected=expected,
        name="identity_valid",
        dtype=torch.bool,
    )
    unknown = _matrix_tensor(
        identity_unknown,
        expected=expected,
        name="identity_unknown",
        dtype=torch.bool,
    )
    mismatch = _matrix_tensor(
        identity_mismatch,
        expected=expected,
        name="identity_mismatch",
        dtype=torch.bool,
    )
    associated = _matrix_tensor(
        associated_valid,
        expected=expected,
        name="associated_valid",
        dtype=torch.bool,
    )
    trusted = _matrix_tensor(
        trusted_valid,
        expected=expected,
        name="trusted_valid",
        dtype=torch.bool,
    )
    if bool((matched.int() + unknown.int() + mismatch.int()).gt(1).any()):
        raise ValueError("MATCH/UNKNOWN/MISMATCH identity states overlap.")
    diagnostic_lookup = _diagnostic_lookup(segmentation_diagnostics)

    rows: list[dict[str, object]] = []
    for sequence_index, frame_index in enumerate(frame_indices):
        for slot, instance_id in enumerate(instance_ids):
            pred = predicted[sequence_index, slot]
            gt = target[sequence_index, slot]
            intersection = int((pred & gt).sum())
            union = int((pred | gt).sum())
            predicted_pixels = int(pred.sum())
            gt_pixels = int(gt.sum())
            gt_visible = gt_pixels > 0
            predicted_visible = predicted_pixels > 0
            iou = float(intersection) / union if union else 1.0
            precision = (
                float(intersection) / predicted_pixels
                if predicted_pixels
                else float(not gt_visible)
            )
            recall = (
                float(intersection) / gt_pixels
                if gt_pixels
                else float(not predicted_visible)
            )
            is_reference = sequence_index == int(reference_sequence_index)
            visible_success = (
                not is_reference
                and gt_visible
                and iou >= float(success_iou_threshold)
            )
            absent_correct = (
                not is_reference and not gt_visible and not predicted_visible
            )
            diagnostic = diagnostic_lookup.get(
                (int(instance_id), sequence_index),
                {},
            )
            rows.append(
                {
                    "clip": clip_name,
                    "split": split,
                    "sequence_index": sequence_index,
                    "frame_index": frame_index,
                    "is_reference": int(is_reference),
                    "instance_id": instance_id,
                    "gt_visible": int(gt_visible),
                    "predicted_visible": int(predicted_visible),
                    "tracking_success_iou50": int(visible_success),
                    "correct_absence": int(absent_correct),
                    "iou": _short(iou),
                    "precision": _short(precision),
                    "recall": _short(recall),
                    "gt_pixels": gt_pixels,
                    "predicted_pixels": predicted_pixels,
                    "tracking_score": _short(float(scores[sequence_index, slot])),
                    "identity_state": _identity_state(
                        matched[sequence_index, slot],
                        unknown[sequence_index, slot],
                        mismatch[sequence_index, slot],
                        present=predicted_visible,
                    ),
                    "used_by_camera": int(associated[sequence_index, slot]),
                    "used_by_geometry": int(trusted[sequence_index, slot]),
                    "geometry_prompt_available": diagnostic.get(
                        "prompt_available",
                        "",
                    ),
                    "geometry_correction_applied": diagnostic.get(
                        "correction_applied",
                        "",
                    ),
                    "geometry_correction_reason": diagnostic.get(
                        "correction_reason",
                        "",
                    ),
                }
            )

    frame_rows = _frame_rows(
        rows,
        frame_indices=frame_indices,
        reference_sequence_index=reference_sequence_index,
    )
    summary_rows = _summary_rows(
        rows,
        clip_name=clip_name,
        split=split,
        instance_ids=instance_ids,
        success_iou_threshold=success_iou_threshold,
    )
    return rows, frame_rows, summary_rows


def write_tracking_comparisons(
    output_dir: str | Path,
    *,
    image_paths: Sequence[str | Path],
    frame_indices: Sequence[int],
    instance_ids: Sequence[int],
    predicted_masks: torch.Tensor,
    target_masks: torch.Tensor,
    instance_frame_rows: Sequence[Mapping[str, object]],
) -> None:
    """Write RGB | GT | final-V6 panels and a compact sequence overview."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_indices = tuple(int(value) for value in frame_indices)
    instance_ids = tuple(int(value) for value in instance_ids)
    predicted = torch.as_tensor(predicted_masks).detach().bool().cpu()
    target = torch.as_tensor(target_masks).detach().bool().cpu()
    if tuple(predicted.shape) != tuple(target.shape):
        raise ValueError("Visualization predicted/GT mask shapes differ.")
    expected = (len(frame_indices), len(instance_ids))
    if predicted.ndim != 4 or tuple(predicted.shape[:2]) != expected:
        raise ValueError(
            f"Visualization masks require [S,K,H,W] prefix {expected}, got "
            f"{tuple(predicted.shape)}."
        )
    if len(image_paths) != len(frame_indices):
        raise ValueError("Visualization image/frame counts differ.")
    row_lookup = {
        (int(row["sequence_index"]), int(row["instance_id"])): row
        for row in instance_frame_rows
    }

    overview: list[Image.Image] = []
    height, width = int(predicted.shape[-2]), int(predicted.shape[-1])
    for sequence_index, (frame_index, image_path) in enumerate(
        zip(frame_indices, image_paths)
    ):
        with Image.open(image_path) as source:
            rgb = source.convert("RGB").resize(
                (width, height),
                resample=_RESAMPLING.BILINEAR,
            )
        gt_overlay = _overlay_instances(
            rgb,
            target[sequence_index],
            instance_ids=instance_ids,
        )
        predicted_overlay = _overlay_instances(
            rgb,
            predicted[sequence_index],
            instance_ids=instance_ids,
        )
        current_rows = [
            row_lookup[(sequence_index, instance_id)]
            for instance_id in instance_ids
        ]
        panel = _comparison_panel(
            rgb,
            gt_overlay,
            predicted_overlay,
            frame_index=frame_index,
            instance_ids=instance_ids,
            rows=current_rows,
        )
        path = output_dir / (
            f"seq_{sequence_index:03d}_frame_{frame_index:06d}.png"
        )
        panel.save(path)
        thumbnail = panel.copy()
        thumbnail.thumbnail((960, 280), resample=_RESAMPLING.LANCZOS)
        overview.append(thumbnail)
    _save_overview(overview, output_dir / "sequence_overview.png")


def write_csv(path: str | Path, rows: Sequence[Mapping[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [dict(row) for row in rows]
    if not rows:
        path.write_text("", encoding="utf8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _frame_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    frame_indices: Sequence[int],
    reference_sequence_index: int,
) -> list[dict[str, object]]:
    output = []
    for sequence_index, frame_index in enumerate(frame_indices):
        current = [
            row for row in rows
            if int(row["sequence_index"]) == sequence_index
        ]
        visible = [row for row in current if int(row["gt_visible"])]
        absent = [row for row in current if not int(row["gt_visible"])]
        output.append(
            {
                "sequence_index": sequence_index,
                "frame_index": int(frame_index),
                "is_reference": int(
                    sequence_index == int(reference_sequence_index)
                ),
                "gt_visible_instances": len(visible),
                "predicted_visible_instances": sum(
                    int(row["predicted_visible"]) for row in current
                ),
                "successful_visible_instances_iou50": sum(
                    int(row["tracking_success_iou50"]) for row in visible
                ),
                "false_negative_visible_instances": sum(
                    not int(row["predicted_visible"]) for row in visible
                ),
                "false_positive_absent_instances": sum(
                    int(row["predicted_visible"]) for row in absent
                ),
                "camera_associated_instances": sum(
                    int(row["used_by_camera"]) for row in current
                ),
                "geometry_trusted_instances": sum(
                    int(row["used_by_geometry"]) for row in current
                ),
            }
        )
    return output


def _summary_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    clip_name: str,
    split: str,
    instance_ids: Sequence[int],
    success_iou_threshold: float,
) -> list[dict[str, object]]:
    nonreference = [row for row in rows if not int(row["is_reference"])]
    groups: list[tuple[str, int | str, list[Mapping[str, object]]]] = [
        ("clip", "all", nonreference)
    ]
    groups.extend(
        (
            "instance",
            int(instance_id),
            [
                row
                for row in nonreference
                if int(row["instance_id"]) == int(instance_id)
            ],
        )
        for instance_id in instance_ids
    )
    output = []
    for scope, instance_id, current in groups:
        visible = [row for row in current if int(row["gt_visible"])]
        absent = [row for row in current if not int(row["gt_visible"])]
        successes = sum(
            int(row["tracking_success_iou50"]) for row in visible
        )
        output.append(
            {
                "clip": clip_name,
                "split": split,
                "scope": scope,
                "instance_id": instance_id,
                "success_iou_threshold": float(success_iou_threshold),
                "nonreference_instance_frames": len(current),
                "gt_visible_frames": len(visible),
                "tracking_success_frames": successes,
                "tracking_success_rate_visible": _short(
                    successes / len(visible) if visible else float("nan")
                ),
                "mean_iou_visible": _short(
                    _mean(visible, "iou")
                ),
                "predicted_on_visible_frames": sum(
                    int(row["predicted_visible"]) for row in visible
                ),
                "false_negative_visible_frames": sum(
                    not int(row["predicted_visible"]) for row in visible
                ),
                "gt_absent_frames": len(absent),
                "false_positive_absent_frames": sum(
                    int(row["predicted_visible"]) for row in absent
                ),
                "camera_associated_frames": sum(
                    int(row["used_by_camera"]) for row in current
                ),
                "geometry_trusted_frames": sum(
                    int(row["used_by_geometry"]) for row in current
                ),
                "match_frames": sum(
                    row["identity_state"] == "MATCH" for row in current
                ),
                "unknown_frames": sum(
                    row["identity_state"] == "UNKNOWN" for row in current
                ),
                "mismatch_frames": sum(
                    row["identity_state"] == "MISMATCH" for row in current
                ),
            }
        )
    return output


def _diagnostic_lookup(
    rows: Iterable[Mapping[str, object]],
) -> dict[tuple[int, int], Mapping[str, object]]:
    output = {}
    for row in rows:
        if "instance_id" not in row or "sequence_index" not in row:
            continue
        output[(int(row["instance_id"]), int(row["sequence_index"]))] = row
    return output


def _overlay_instances(
    rgb: Image.Image,
    masks: torch.Tensor,
    *,
    instance_ids: Sequence[int],
) -> Image.Image:
    output = np.asarray(rgb, dtype=np.float32).copy()
    draw_labels: list[tuple[int, int, str, tuple[int, int, int]]] = []
    for slot, instance_id in enumerate(instance_ids):
        mask = torch.as_tensor(masks[slot]).detach().bool().cpu().numpy()
        color = np.asarray(_COLORS[slot % len(_COLORS)], dtype=np.float32)
        if not mask.any():
            continue
        output[mask] = 0.52 * output[mask] + 0.48 * color
        ys, xs = np.nonzero(mask)
        draw_labels.append(
            (
                int(xs.min()),
                int(ys.min()),
                str(int(instance_id)),
                tuple(int(value) for value in color),
            )
        )
    image = Image.fromarray(np.clip(output, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    for x, y, label, color in draw_labels:
        draw.rectangle((x, y, x + 24, y + 13), fill=(20, 20, 20))
        draw.text((x + 2, y + 1), label, fill=color)
    return image


def _comparison_panel(
    rgb: Image.Image,
    gt: Image.Image,
    predicted: Image.Image,
    *,
    frame_index: int,
    instance_ids: Sequence[int],
    rows: Sequence[Mapping[str, object]],
) -> Image.Image:
    header = 44
    panels = (rgb, gt, predicted)
    titles = ("RGB", "GT instances", "V6 SAM3.1 final tracking")
    canvas = Image.new(
        "RGB",
        (rgb.width * len(panels), rgb.height + header),
        (245, 245, 245),
    )
    draw = ImageDraw.Draw(canvas)
    for index, (title, panel) in enumerate(zip(titles, panels)):
        x = index * rgb.width
        canvas.paste(panel, (x, header))
        draw.text((x + 5, 4), title, fill=(10, 10, 10))
    details = []
    for instance_id, row in zip(instance_ids, rows):
        if int(row["gt_visible"]):
            status = (
                "OK"
                if int(row["tracking_success_iou50"])
                else "FAIL"
            )
            details.append(
                f"{int(instance_id)} IoU={float(row['iou']):.2f} {status}"
            )
        else:
            status = "FP" if int(row["predicted_visible"]) else "TN"
            details.append(f"{int(instance_id)} absent {status}")
    draw.text(
        (5, 23),
        f"frame={int(frame_index)}  " + " | ".join(details),
        fill=(20, 20, 20),
    )
    return canvas


def _save_overview(panels: Sequence[Image.Image], path: Path) -> None:
    if not panels:
        return
    columns = min(2, len(panels))
    rows = (len(panels) + columns - 1) // columns
    width = max(image.width for image in panels)
    height = max(image.height for image in panels)
    canvas = Image.new(
        "RGB",
        (columns * width, rows * height),
        (18, 20, 24),
    )
    for index, panel in enumerate(panels):
        canvas.paste(
            panel,
            ((index % columns) * width, (index // columns) * height),
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _mask_tensor(
    value: torch.Tensor,
    *,
    expected_prefix: tuple[int, int],
    name: str,
) -> torch.Tensor:
    tensor = torch.as_tensor(value).detach().bool().cpu()
    if tensor.ndim != 4 or tuple(tensor.shape[:2]) != expected_prefix:
        raise ValueError(
            f"{name} must have [S,K,H,W] prefix {expected_prefix}, got "
            f"{tuple(tensor.shape)}."
        )
    return tensor


def _matrix_tensor(
    value: torch.Tensor,
    *,
    expected: tuple[int, int],
    name: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    tensor = torch.as_tensor(value).detach().to(dtype=dtype).cpu()
    if tuple(tensor.shape) != expected:
        raise ValueError(
            f"{name} must have shape {expected}, got {tuple(tensor.shape)}."
        )
    return tensor


def _identity_state(
    matched: torch.Tensor,
    unknown: torch.Tensor,
    mismatch: torch.Tensor,
    *,
    present: bool,
) -> str:
    if bool(matched):
        return "MATCH"
    if bool(unknown):
        return "UNKNOWN"
    if bool(mismatch):
        return "MISMATCH"
    return "ABSENT" if not present else "UNCLASSIFIED"


def _mean(rows: Sequence[Mapping[str, object]], key: str) -> float:
    if not rows:
        return float("nan")
    return sum(float(row[key]) for row in rows) / len(rows)


def _short(value: float) -> float:
    return float(f"{float(value):.8g}")
