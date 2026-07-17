"""Instance descriptors pooled from frozen StreamVGGT feature maps."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..types import SAM3MaskCandidate


SELECTION_MODES = (
    "geometry_only",
    "descriptor_only",
    "geometry_descriptor",
    "shuffled_descriptor",
)


@dataclass
class DescriptorMemory:
    descriptor: torch.Tensor | None = None
    total_weight: float = 0.0
    observations: int = 0

    def update(self, descriptor: torch.Tensor, *, weight: float) -> None:
        descriptor = F.normalize(descriptor.detach().float().cpu(), dim=0)
        weight = max(float(weight), 1e-6)
        if self.descriptor is None:
            merged = descriptor
        else:
            merged = (
                self.descriptor * self.total_weight + descriptor * weight
            ) / (self.total_weight + weight)
        self.descriptor = F.normalize(merged, dim=0)
        self.total_weight += weight
        self.observations += 1


@dataclass(frozen=True)
class PooledDescriptor:
    descriptor: torch.Tensor
    valid_tokens: int
    weight_sum: float


@dataclass(frozen=True)
class CandidateSelection:
    candidate: SAM3MaskCandidate
    rows: tuple[dict, ...]


def pool_segment_descriptor(
    feature_map: torch.Tensor,
    stream_mask: torch.Tensor,
    *,
    confidence: torch.Tensor | None = None,
) -> PooledDescriptor | None:
    """Confidence-weighted mean pooling over one instance region."""

    if feature_map.ndim != 3:
        raise ValueError(
            f"feature_map must be [C,H,W], got {tuple(feature_map.shape)}."
        )
    height, width = feature_map.shape[-2:]
    mask = F.interpolate(
        stream_mask.float()[None, None],
        size=(height, width),
        mode="nearest",
    )[0, 0] > 0.5
    finite = torch.isfinite(feature_map).all(dim=0)
    selected = mask & finite
    if not selected.any():
        return None

    if confidence is None:
        weights = torch.ones((height, width), dtype=torch.float32)
    else:
        weights = F.interpolate(
            confidence.float()[None, None],
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        weights = weights.clamp_min(0.0)
    selected_weights = weights[selected]
    if float(selected_weights.sum()) <= 1e-6:
        selected_weights = torch.ones_like(selected_weights)

    values = feature_map.float()[:, selected]
    descriptor = (values * selected_weights[None]).sum(dim=1)
    descriptor = descriptor / selected_weights.sum().clamp_min(1e-6)
    if not torch.isfinite(descriptor).all() or float(descriptor.norm()) <= 1e-8:
        return None
    return PooledDescriptor(
        descriptor=F.normalize(descriptor, dim=0).cpu(),
        valid_tokens=int(selected.sum()),
        weight_sum=float(selected_weights.sum()),
    )


def select_mask_candidate(
    candidates: list[SAM3MaskCandidate],
    *,
    supported_mask: torch.Tensor,
    candidate_descriptors: list[PooledDescriptor | None],
    history_descriptor: torch.Tensor | None,
    mode: str,
    geometry_weight: float,
    descriptor_weight: float,
) -> CandidateSelection:
    """Rank a fixed SAM3 candidate set while changing only the scoring rule."""

    if mode not in SELECTION_MODES:
        raise ValueError(f"Unsupported candidate selection mode {mode!r}.")
    if len(candidates) != len(candidate_descriptors):
        raise ValueError("Every SAM3 candidate needs one descriptor entry.")
    if not candidates:
        raise ValueError("SAM3 produced no text candidates.")
    if mode != "geometry_only" and history_descriptor is None:
        raise ValueError("Descriptor selection requires a historical descriptor.")

    supported_mask = supported_mask.detach().cpu().bool()
    history = (
        F.normalize(history_descriptor.detach().float().cpu(), dim=0)
        if history_descriptor is not None
        else None
    )
    rows = []
    ranking = []
    for index, (candidate, pooled) in enumerate(
        zip(candidates, candidate_descriptors)
    ):
        geometry_iou = _binary_iou(candidate.mask, supported_mask)
        descriptor_cosine = (
            float(torch.dot(history, pooled.descriptor))
            if history is not None and pooled is not None
            else -1.0
        )
        descriptor_score = 0.5 * (descriptor_cosine + 1.0)
        if mode == "geometry_only":
            selection_score = geometry_iou
        elif mode == "descriptor_only":
            selection_score = descriptor_score
        else:
            denominator = max(
                float(geometry_weight) + float(descriptor_weight),
                1e-8,
            )
            selection_score = (
                float(geometry_weight) * geometry_iou
                + float(descriptor_weight) * descriptor_score
            ) / denominator
        rows.append(
            {
                "candidate_index": index,
                "temporary_obj_id": candidate.obj_id,
                "sam3_score": candidate.score,
                "candidate_pixels": int(candidate.mask.sum()),
                "geometry_iou": geometry_iou,
                "descriptor_cosine": descriptor_cosine,
                "descriptor_score": descriptor_score,
                "descriptor_valid_tokens": (
                    pooled.valid_tokens if pooled is not None else 0
                ),
                "selection_score": selection_score,
            }
        )
        ranking.append(
            (
                selection_score,
                float(candidate.score),
                -int(candidate.obj_id),
                index,
            )
        )

    selected_index = max(ranking)[-1]
    selected_rows = []
    for index, row in enumerate(rows):
        selected_rows.append({**row, "selected": int(index == selected_index)})
    return CandidateSelection(
        candidate=candidates[selected_index],
        rows=tuple(selected_rows),
    )


def _binary_iou(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.detach().cpu().bool()
    right = right.detach().cpu().bool()
    union = (left | right).sum()
    if int(union) == 0:
        return 1.0
    return float((left & right).sum()) / float(union)
