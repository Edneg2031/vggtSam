"""Conservative object-level point-map aggregation."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ObjectMapEntry:
    instance_id: int
    label: str
    points: torch.Tensor
    weights: torch.Tensor
    observations: int
    last_frame_idx: int


class ObjectPointMap:
    """Store StreamVGGT world points for one persistent object identity.

    No ICP is applied here. StreamVGGT already predicts points in a shared
    sequence frame, while an unconstrained per-object ICP can hide pose errors
    or align a wrong instance to the map.
    """

    def __init__(self, *, max_points_per_object: int) -> None:
        self.max_points_per_object = int(max_points_per_object)
        self.entries: dict[int, ObjectMapEntry] = {}

    def get(self, instance_id: int) -> ObjectMapEntry | None:
        return self.entries.get(int(instance_id))

    def update(
        self,
        *,
        instance_id: int,
        label: str,
        points: torch.Tensor,
        weights: torch.Tensor,
        frame_idx: int,
    ) -> ObjectMapEntry | None:
        valid = torch.isfinite(points).all(dim=-1) & torch.isfinite(weights)
        points = points[valid].detach().float().cpu()
        weights = weights[valid].detach().float().cpu().clamp_min(0.0)
        if points.numel() == 0:
            return self.get(instance_id)

        previous = self.get(instance_id)
        if previous is not None:
            points = torch.cat([previous.points, points], dim=0)
            weights = torch.cat([previous.weights, weights], dim=0)
            observations = previous.observations + 1
        else:
            observations = 1
        points, weights = _keep_high_confidence(
            points,
            weights,
            self.max_points_per_object,
        )
        entry = ObjectMapEntry(
            instance_id=int(instance_id),
            label=str(label),
            points=points,
            weights=weights,
            observations=observations,
            last_frame_idx=int(frame_idx),
        )
        self.entries[int(instance_id)] = entry
        return entry


def sample_masked_observation(
    world_points: torch.Tensor,
    confidence: torch.Tensor,
    mask: torch.Tensor,
    *,
    max_points: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = (
        mask.bool()
        & torch.isfinite(world_points).all(dim=-1)
        & torch.isfinite(confidence)
        & (confidence > 0.0)
    )
    points = world_points[valid]
    weights = confidence[valid]
    return _keep_high_confidence(points, weights, int(max_points))


def _keep_high_confidence(
    points: torch.Tensor,
    weights: torch.Tensor,
    limit: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if points.shape[0] <= limit:
        return points, weights
    indices = torch.topk(weights, k=limit, sorted=False).indices
    return points.index_select(0, indices), weights.index_select(0, indices)

