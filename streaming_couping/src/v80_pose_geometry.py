"""Coordinate-safe primitives for V8 oracle pose validation."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass
class CausalPairIndices:
    current_slots: torch.Tensor
    current_points: torch.Tensor
    history_frames: torch.Tensor
    history_slots: torch.Tensor
    history_points: torch.Tensor
    gt_distances_metric: torch.Tensor

    @property
    def count(self) -> int:
        return int(self.current_points.numel())


def homogeneous(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.shape[-2:] == (4, 4):
        return matrix
    if matrix.shape[-2:] != (3, 4):
        raise ValueError("Pose must end in [3,4] or [4,4].")
    output = torch.eye(
        4, dtype=matrix.dtype, device=matrix.device
    ).expand((*matrix.shape[:-2], 4, 4)).clone()
    output[..., :3, :4] = matrix
    return output


def invert_rigid(matrix: torch.Tensor) -> torch.Tensor:
    value = homogeneous(matrix)
    rotation = value[..., :3, :3]
    translation = value[..., :3, 3]
    output = torch.eye(
        4, dtype=value.dtype, device=value.device
    ).expand(value.shape).clone()
    output[..., :3, :3] = rotation.transpose(-1, -2)
    output[..., :3, 3] = -(
        rotation.transpose(-1, -2) @ translation[..., None]
    )[..., 0]
    return output


def transform_points(matrix: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    if points.ndim < 1 or points.shape[-1] != 3:
        raise ValueError("Points must end in dimension three.")
    value = homogeneous(matrix)
    if value.device != points.device:
        raise ValueError(
            "Pose and points must be on the same device: "
            f"pose={value.device}, points={points.device}."
        )
    common_dtype = torch.promote_types(value.dtype, points.dtype)
    if not (common_dtype.is_floating_point or common_dtype.is_complex):
        common_dtype = torch.get_default_dtype()
    value = value.to(dtype=common_dtype)
    points = points.to(dtype=common_dtype)
    rotation = value[..., :3, :3]
    translation = value[..., :3, 3]
    matrix_leading = rotation.shape[:-2]
    point_leading = points.shape[:-1]
    if len(matrix_leading) > len(point_leading) or any(
        matrix_size not in (1, point_size)
        for matrix_size, point_size in zip(matrix_leading, point_leading)
    ):
        raise ValueError(
            "Pose and point leading dimensions are not broadcast-compatible: "
            f"pose={tuple(matrix.shape)}, points={tuple(points.shape)}."
        )
    for _ in range(len(point_leading) - len(matrix_leading)):
        rotation = rotation.unsqueeze(-3)
        translation = translation.unsqueeze(-2)
    return (rotation @ points[..., None])[..., 0] + translation


def camera_centers(world_to_camera: torch.Tensor) -> torch.Tensor:
    return invert_rigid(world_to_camera)[..., :3, 3]


def rotation_error_degrees(
    predicted: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    predicted_rotation = homogeneous(predicted)[..., :3, :3]
    target_rotation = homogeneous(target)[..., :3, :3]
    relative = predicted_rotation @ target_rotation.transpose(-1, -2)
    cosine = (
        torch.diagonal(relative, dim1=-2, dim2=-1).sum(dim=-1) - 1.0
    ) * 0.5
    return torch.rad2deg(torch.acos(cosine.clamp(-1.0, 1.0)))


def sample_dense_at_local_tokens(
    dense: torch.Tensor,
    *,
    local_features: torch.Tensor,
    local_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Nearest-sample ``[S,H,W,C]`` at token UVs from ``[S,K,P,F]``."""

    if dense.ndim == 3:
        dense = dense[..., None]
    if dense.ndim != 4:
        raise ValueError("Dense token source must be [S,H,W,C] or [S,H,W].")
    if dense.shape[1] < 1 or dense.shape[2] < 1:
        raise ValueError("Dense token source must have non-empty spatial dimensions.")
    if local_features.ndim != 4 or local_valid.shape != local_features.shape[:-1]:
        raise ValueError("Local tokens must be [S,K,P,F]/[S,K,P].")
    if local_features.shape[0] != dense.shape[0] or local_features.shape[-1] < 5:
        raise ValueError("Dense sequence and local token shapes disagree.")
    sequence, height, width = dense.shape[:3]
    uv = local_features[..., 3:5].float()
    uv_finite = torch.isfinite(uv).all(dim=-1)
    safe_uv = torch.where(uv_finite[..., None], uv, torch.zeros_like(uv))
    x = (((safe_uv[..., 0] + 1.0) * 0.5) * max(width - 1, 1)).round().long()
    y = (((safe_uv[..., 1] + 1.0) * 0.5) * max(height - 1, 1)).round().long()
    x = x.clamp(0, width - 1)
    y = y.clamp(0, height - 1)
    frame = torch.arange(sequence, device=dense.device)[:, None, None]
    sampled = dense[frame, y, x]
    valid = (
        local_valid.bool()
        & uv_finite
        & torch.isfinite(sampled).all(dim=-1)
    )
    return torch.where(valid[..., None], sampled, torch.zeros_like(sampled)), valid


def backproject_depth_at_local_tokens(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    local_features: torch.Tensor,
    local_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create StreamVGGT-native camera points without using its L0 pose."""

    sampled, valid = sample_dense_at_local_tokens(
        depth,
        local_features=local_features,
        local_valid=local_valid,
    )
    if sampled.shape[-1] != 1:
        raise ValueError("Depth must have one channel.")
    if intrinsics.shape != (local_features.shape[0], 3, 3):
        raise ValueError("Intrinsics must be [S,3,3].")
    z = sampled[..., 0]
    valid = valid & z.gt(1e-6)
    sequence, instances, points = z.shape
    height, width = depth.shape[1:3]
    uv = local_features[..., 3:5].float()
    u = ((uv[..., 0] + 1.0) * 0.5) * max(width - 1, 1)
    v = ((uv[..., 1] + 1.0) * 0.5) * max(height - 1, 1)
    fx_raw = intrinsics[:, 0, 0].reshape(sequence, 1, 1)
    fy_raw = intrinsics[:, 1, 1].reshape(sequence, 1, 1)
    focal_valid = (
        torch.isfinite(fx_raw)
        & torch.isfinite(fy_raw)
        & fx_raw.gt(1e-8)
        & fy_raw.gt(1e-8)
    )
    fx = torch.where(focal_valid, fx_raw, torch.ones_like(fx_raw))
    fy = torch.where(focal_valid, fy_raw, torch.ones_like(fy_raw))
    cx = intrinsics[:, 0, 2].reshape(sequence, 1, 1)
    cy = intrinsics[:, 1, 2].reshape(sequence, 1, 1)
    camera = torch.stack(
        [(u - cx) * z / fx, (v - cy) * z / fy, z], dim=-1
    )
    valid = valid & focal_valid & torch.isfinite(camera).all(dim=-1)
    camera = torch.where(valid[..., None], camera, torch.zeros_like(camera))
    if camera.shape != (sequence, instances, points, 3):
        raise RuntimeError("Unexpected backprojected camera point shape.")
    return camera, valid


def gt_camera_tokens_native(
    gt_world_tokens_metric: torch.Tensor,
    gt_world_valid: torch.Tensor,
    gt_world_to_camera_metric: torch.Tensor,
    *,
    native_to_metric_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transform metric GT world samples into native-scale camera frames."""

    if gt_world_tokens_metric.ndim != 4 or gt_world_tokens_metric.shape[-1] != 3:
        raise ValueError("GT world tokens must be [S,K,P,3].")
    if gt_world_valid.shape != gt_world_tokens_metric.shape[:-1]:
        raise ValueError("GT world token validity has the wrong shape.")
    if tuple(gt_world_to_camera_metric.shape) not in {
        (gt_world_tokens_metric.shape[0], 3, 4),
        (gt_world_tokens_metric.shape[0], 4, 4),
    }:
        raise ValueError("GT world_to_camera must be [S,3,4] or [S,4,4].")
    scale = float(native_to_metric_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("Native-to-metric scale must be finite and positive.")
    pose = homogeneous(gt_world_to_camera_metric).float()
    camera = torch.einsum(
        "sij,skpj->skpi", pose[:, :3, :3], gt_world_tokens_metric.float()
    )
    camera = camera + pose[:, None, None, :3, 3]
    camera = camera / scale
    valid = gt_world_valid.bool() & torch.isfinite(camera).all(dim=-1)
    return torch.where(valid[..., None], camera, torch.zeros_like(camera)), valid


def causal_history_indices(
    memory_write: torch.Tensor,
    local_valid: torch.Tensor,
) -> torch.Tensor:
    """Return the exact preceding frame read by V8 causal geometry memory."""

    if memory_write.ndim != 2 or local_valid.ndim != 3:
        raise ValueError("Causal history expects [S,K] write and [S,K,P] valid.")
    if memory_write.shape != local_valid.shape[:2]:
        raise ValueError("Causal history write/local shapes disagree.")
    sequence, instances = memory_write.shape
    last = torch.full((instances,), -1, dtype=torch.long, device=memory_write.device)
    output = torch.full_like(memory_write, -1, dtype=torch.long)
    for frame in range(sequence):
        output[frame] = last
        write = memory_write[frame].bool() & local_valid[frame].any(dim=-1)
        last = torch.where(write, torch.full_like(last, frame), last)
    return output


def causal_gt_nearest_pairs(
    *,
    current_frame: int,
    history_indices: torch.Tensor,
    gt_world_metric: torch.Tensor,
    gt_valid: torch.Tensor,
    max_distance_metric: float,
    require_mutual_nearest: bool,
) -> CausalPairIndices:
    """Build per-slot GT-world proximity pseudo-correspondences."""

    if gt_world_metric.ndim != 4 or gt_world_metric.shape[-1] != 3:
        raise ValueError("GT correspondence points must be [S,K,P,3].")
    if gt_valid.shape != gt_world_metric.shape[:-1]:
        raise ValueError("GT correspondence validity has the wrong shape.")
    sequence, instances = gt_world_metric.shape[:2]
    if not 0 <= int(current_frame) < sequence:
        raise IndexError("Current correspondence frame is out of range.")
    if (
        not math.isfinite(float(max_distance_metric))
        or float(max_distance_metric) <= 0.0
    ):
        raise ValueError("GT correspondence radius must be finite and positive.")
    if history_indices.shape != (instances,):
        raise ValueError("Current causal history indices must be [K].")
    rows: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for slot in range(instances):
        history = int(history_indices[slot])
        if history < 0:
            continue
        if history >= sequence:
            raise IndexError("Causal history frame is out of range.")
        current_valid = (
            gt_valid[int(current_frame), slot]
            & torch.isfinite(gt_world_metric[int(current_frame), slot]).all(dim=-1)
        )
        history_valid = (
            gt_valid[history, slot]
            & torch.isfinite(gt_world_metric[history, slot]).all(dim=-1)
        )
        if not bool(current_valid.any()) or not bool(history_valid.any()):
            continue
        current_index = torch.nonzero(current_valid, as_tuple=False).flatten()
        history_index = torch.nonzero(history_valid, as_tuple=False).flatten()
        current = gt_world_metric[int(current_frame), slot].index_select(
            0, current_index
        )
        previous = gt_world_metric[history, slot].index_select(0, history_index)
        distance = torch.cdist(current.float(), previous.float())
        nearest_distance, nearest_local = distance.min(dim=-1)
        keep = nearest_distance.le(float(max_distance_metric))
        if require_mutual_nearest:
            reverse = distance.min(dim=0).indices
            query = torch.arange(distance.shape[0], device=distance.device)
            keep = keep & reverse.index_select(0, nearest_local).eq(query)
        if not bool(keep.any()):
            continue
        query_index = current_index[keep]
        key_index = history_index[nearest_local[keep]]
        rows.append((slot, query_index, key_index, nearest_distance[keep]))
    if not rows:
        empty = torch.empty(0, dtype=torch.long, device=gt_world_metric.device)
        return CausalPairIndices(
            current_slots=empty,
            current_points=empty,
            history_frames=empty,
            history_slots=empty,
            history_points=empty,
            gt_distances_metric=gt_world_metric.new_empty(0),
        )
    slots = torch.cat(
        [
            torch.full_like(query, int(slot))
            for slot, query, _, _ in rows
        ]
    )
    histories = torch.cat(
        [
            torch.full_like(query, int(history_indices[slot]))
            for slot, query, _, _ in rows
        ]
    )
    return CausalPairIndices(
        current_slots=slots,
        current_points=torch.cat([query for _, query, _, _ in rows]),
        history_frames=histories,
        history_slots=slots.clone(),
        history_points=torch.cat([key for _, _, key, _ in rows]),
        gt_distances_metric=torch.cat([distance for _, _, _, distance in rows]),
    )


def gather_pair_points(
    points: torch.Tensor,
    valid: torch.Tensor,
    *,
    current_frame: int,
    pairs: CausalPairIndices,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather current/history camera points and a pair-valid mask."""

    if points.ndim != 4 or points.shape[-1] != 3 or valid.shape != points.shape[:-1]:
        raise ValueError("Pair point source must be [S,K,P,3]/[S,K,P].")
    if not 0 <= int(current_frame) < points.shape[0]:
        raise IndexError("Current pair frame is out of range.")
    current = points[
        int(current_frame), pairs.current_slots, pairs.current_points
    ]
    history = points[
        pairs.history_frames, pairs.history_slots, pairs.history_points
    ]
    pair_valid = (
        valid[int(current_frame), pairs.current_slots, pairs.current_points]
        & valid[pairs.history_frames, pairs.history_slots, pairs.history_points]
        & torch.isfinite(current).all(dim=-1)
        & torch.isfinite(history).all(dim=-1)
    )
    return current, history, pair_valid
