"""Tracking metrics shared by the explicit bridge controls."""

from __future__ import annotations

import torch


def binary_iou(prediction: torch.Tensor, target: torch.Tensor) -> float:
    prediction = prediction.bool()
    target = target.bool()
    union = (prediction | target).sum()
    if int(union) == 0:
        return 1.0
    return float((prediction & target).sum().float() / union.float())


def summarize_masks(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    reference_frame_idx: int,
) -> dict[str, float]:
    ious = torch.tensor(
        [binary_iou(pred, gt) for pred, gt in zip(prediction, target)],
        dtype=torch.float32,
    )
    visible = target.flatten(1).any(dim=1)
    cross = visible.clone()
    cross[int(reference_frame_idx)] = False
    absent = ~visible
    return {
        "mean_iou": float(ious.mean()),
        "positive_iou": float(ious[visible].mean()) if visible.any() else 0.0,
        "cross_view_iou": float(ious[cross].mean()) if cross.any() else 0.0,
        "cross_view_recall": float((ious[cross] >= 0.5).float().mean()) if cross.any() else 0.0,
        "absent_fp_ratio": float(prediction[absent].float().mean()) if absent.any() else 0.0,
    }
