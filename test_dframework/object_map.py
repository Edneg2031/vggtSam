"""Instance-indexed, confidence-weighted world-point memory."""

from __future__ import annotations

import torch

from .types import ObjectMapEntry


class ObjectPointMap:
    def __init__(self, *, max_points_per_object: int = 20000) -> None:
        self.max_points_per_object = int(max_points_per_object)
        self.entries: dict[int, ObjectMapEntry] = {}

    def has(self, instance_id: int) -> bool:
        return int(instance_id) in self.entries

    def get(self, instance_id: int) -> ObjectMapEntry | None:
        return self.entries.get(int(instance_id))

    def update(
        self,
        *,
        instance_id: int,
        label: str,
        points: torch.Tensor,
        weights: torch.Tensor,
        centroid: torch.Tensor,
        frame_idx: int,
    ) -> None:
        if points.numel() == 0:
            return
        key = int(instance_id)
        previous = self.entries.get(key)
        if previous is None:
            all_points = points.detach().float().cpu()
            all_weights = weights.detach().float().cpu()
            observations = 1
            centroids = [centroid.detach().float().cpu()]
        else:
            all_points = torch.cat([previous.points, points.detach().float().cpu()], dim=0)
            all_weights = torch.cat([previous.weights, weights.detach().float().cpu()], dim=0)
            observations = previous.observations + 1
            centroids = [*previous.centroid_history, centroid.detach().float().cpu()]
        if all_points.shape[0] > self.max_points_per_object:
            keep = torch.topk(all_weights, self.max_points_per_object).indices
            all_points = all_points.index_select(0, keep)
            all_weights = all_weights.index_select(0, keep)
        self.entries[key] = ObjectMapEntry(
            instance_id=key,
            label=label,
            points=all_points,
            weights=all_weights,
            observations=observations,
            last_seen=int(frame_idx),
            centroid_history=centroids,
        )

