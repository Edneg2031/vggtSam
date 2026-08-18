"""Read-only pointmap diagnosis primitives for the frozen V0 cache.

This module deliberately contains no model loading and no map write-back.  It
maps every native StreamVGGT branch through the single reference Sim(3) stored
in the V0 cache before comparing it with metric dataset geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as functional


@dataclass(frozen=True)
class CoordinateBundle:
    raw_world_to_camera_native: torch.Tensor
    qk_world_to_camera_native: torch.Tensor
    raw_world_to_camera_metric: torch.Tensor
    qk_world_to_camera_metric: torch.Tensor
    target_world_to_camera_metric: torch.Tensor
    calibrated_intrinsics: torch.Tensor
    predicted_intrinsics: torch.Tensor
    raw_depth_metric: torch.Tensor
    raw_depth_pointmap_scale_metric: torch.Tensor
    target_depth_metric: torch.Tensor
    raw_world_points_metric: torch.Tensor
    target_world_points_metric: torch.Tensor
    confidence: torch.Tensor
    native_to_metric_scale: float
    native_to_metric_rotation: torch.Tensor
    native_to_metric_translation: torch.Tensor
    depth_reference_affine_scale: float
    depth_reference_affine_shift: float
    depth_reference_affine_fit_rmse: float


def intrinsics_from_pose_encoding(
    pose_encoding: torch.Tensor,
    image_size: tuple[int, int],
) -> torch.Tensor:
    """Decode only K from StreamVGGT's absT_quaR_FoV representation."""

    encoding = pose_encoding.detach().float().cpu()
    if encoding.ndim != 2 or encoding.shape[-1] < 9:
        raise ValueError("pose_encoding must have shape [S,9].")
    height, width = map(float, image_size)
    fov_h, fov_w = encoding[:, 7], encoding[:, 8]
    intrinsics = torch.zeros(
        encoding.shape[0], 3, 3, dtype=torch.float32
    )
    intrinsics[:, 0, 0] = (width / 2.0) / torch.tan(fov_w / 2.0)
    intrinsics[:, 1, 1] = (height / 2.0) / torch.tan(fov_h / 2.0)
    intrinsics[:, 0, 2] = width / 2.0
    intrinsics[:, 1, 2] = height / 2.0
    intrinsics[:, 2, 2] = 1.0
    if not torch.isfinite(intrinsics).all():
        raise ValueError("Decoded intrinsics contain non-finite values.")
    return intrinsics


def as_homogeneous_world_to_camera(value: torch.Tensor) -> torch.Tensor:
    pose = value.detach().float().cpu()
    if pose.ndim == 4 and pose.shape[0] == 1:
        pose = pose[0]
    if pose.ndim != 3 or pose.shape[-2:] not in {(3, 4), (4, 4)}:
        raise ValueError("world_to_camera must have shape [S,3,4] or [S,4,4].")
    if pose.shape[-2:] == (4, 4):
        return pose.clone()
    output = torch.eye(4, dtype=pose.dtype).expand(pose.shape[0], 4, 4).clone()
    output[:, :3, :4] = pose
    return output


def native_pose_to_metric_world(
    world_to_camera_native: torch.Tensor,
    *,
    scale: float,
    rotation: torch.Tensor,
    translation: torch.Tensor,
) -> torch.Tensor:
    """Change native-world W2C poses into the metric GT world coordinate.

    The stored reference alignment is ``X_metric = s R X_native + t``.
    Camera axes are unchanged and camera-space translations acquire metric
    scale.  This is not a branch-specific pose fit.
    """

    native = as_homogeneous_world_to_camera(world_to_camera_native)
    align_rotation = rotation.detach().float().cpu()
    align_translation = translation.detach().float().cpu().reshape(3)
    if align_rotation.shape != (3, 3) or float(scale) <= 0.0:
        raise ValueError("Invalid native-to-metric Sim(3).")
    output = torch.eye(4).expand(native.shape[0], 4, 4).clone()
    output[:, :3, :3] = native[:, :3, :3] @ align_rotation.T
    output[:, :3, 3] = (
        float(scale) * native[:, :3, 3]
        - torch.einsum(
            "sij,j->si", output[:, :3, :3], align_translation
        )
    )
    return output


def align_native_points(
    points: torch.Tensor,
    *,
    scale: float,
    rotation: torch.Tensor,
    translation: torch.Tensor,
) -> torch.Tensor:
    return float(scale) * (
        points.detach().float().cpu() @ rotation.detach().float().cpu().T
    ) + translation.detach().float().cpu().reshape(3)


def prepare_coordinate_bundle(payload: dict, qk_artifact: dict) -> CoordinateBundle:
    """Build an explicit metric-coordinate view of the V0 artifacts."""

    required = (
        "image_size",
        "baseline_pose_encoding",
        "target_pose_encoding",
        "baseline_depth",
        "target_depth",
        "baseline_world_points",
        "baseline_world_confidence",
        "target_world_to_camera",
        "target_world_points",
        "point_alignment_scale",
        "point_alignment_rotation",
        "point_alignment_translation",
    )
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError(f"V0 cache lacks required diagnosis fields: {missing}")
    raw_pose = qk_artifact.get("raw_world_to_camera")
    qk_pose = qk_artifact.get("selected_world_to_camera")
    if not torch.is_tensor(raw_pose) or not torch.is_tensor(qk_pose):
        raise ValueError("QK artifact lacks raw/selected world_to_camera tensors.")
    scale = float(payload["point_alignment_scale"])
    rotation = payload["point_alignment_rotation"]
    translation = payload["point_alignment_translation"]
    image_size = tuple(int(value) for value in payload["image_size"])
    raw_depth = _squeeze_scalar(payload["baseline_depth"], "baseline_depth")
    target_depth = _squeeze_scalar(payload["target_depth"], "target_depth")
    confidence = _squeeze_scalar(
        payload["baseline_world_confidence"], "baseline_world_confidence"
    )
    reference_index = int(payload.get("reference_sequence_index", 0))
    depth_scale, depth_shift, depth_fit_rmse = _robust_affine_depth_alignment(
        raw_depth[reference_index], target_depth[reference_index]
    )
    raw_pose_native = as_homogeneous_world_to_camera(raw_pose)
    qk_pose_native = as_homogeneous_world_to_camera(qk_pose)
    bundle = CoordinateBundle(
        raw_world_to_camera_native=raw_pose_native,
        qk_world_to_camera_native=qk_pose_native,
        raw_world_to_camera_metric=native_pose_to_metric_world(
            raw_pose_native, scale=scale, rotation=rotation, translation=translation
        ),
        qk_world_to_camera_metric=native_pose_to_metric_world(
            qk_pose_native, scale=scale, rotation=rotation, translation=translation
        ),
        target_world_to_camera_metric=as_homogeneous_world_to_camera(
            payload["target_world_to_camera"]
        ),
        calibrated_intrinsics=intrinsics_from_pose_encoding(
            payload["target_pose_encoding"], image_size
        ),
        predicted_intrinsics=intrinsics_from_pose_encoding(
            payload["baseline_pose_encoding"], image_size
        ),
        raw_depth_metric=raw_depth * depth_scale + depth_shift,
        raw_depth_pointmap_scale_metric=raw_depth * scale,
        target_depth_metric=target_depth,
        raw_world_points_metric=align_native_points(
            payload["baseline_world_points"],
            scale=scale,
            rotation=rotation,
            translation=translation,
        ),
        target_world_points_metric=payload["target_world_points"].detach().float().cpu(),
        confidence=confidence,
        native_to_metric_scale=scale,
        native_to_metric_rotation=rotation.detach().float().cpu(),
        native_to_metric_translation=translation.detach().float().cpu(),
        depth_reference_affine_scale=depth_scale,
        depth_reference_affine_shift=depth_shift,
        depth_reference_affine_fit_rmse=depth_fit_rmse,
    )
    _validate_bundle_shapes(bundle)
    return bundle


def unproject_z_depth(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    world_to_camera: torch.Tensor,
) -> torch.Tensor:
    """Unproject camera-space z-depth to a metric world pointmap."""

    depth = depth.detach().float().cpu()
    intrinsics = intrinsics.detach().float().cpu()
    pose = as_homogeneous_world_to_camera(world_to_camera)
    if depth.ndim != 3:
        raise ValueError("depth must have shape [S,H,W].")
    sequence, height, width = depth.shape
    if intrinsics.shape != (sequence, 3, 3) or pose.shape != (sequence, 4, 4):
        raise ValueError("depth, intrinsics and poses have inconsistent sequence shapes.")
    y, x = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    fx = intrinsics[:, 0, 0, None, None]
    fy = intrinsics[:, 1, 1, None, None]
    cx = intrinsics[:, 0, 2, None, None]
    cy = intrinsics[:, 1, 2, None, None]
    camera = torch.stack(
        (
            (x[None] - cx) * depth / fx,
            (y[None] - cy) * depth / fy,
            depth,
        ),
        dim=-1,
    )
    rotation = pose[:, :3, :3]
    translation = pose[:, :3, 3]
    world = torch.einsum(
        "sij,shwj->shwi", rotation.transpose(-1, -2), camera - translation[:, None, None]
    )
    valid = torch.isfinite(depth) & (depth > 0.0)
    return torch.where(valid[..., None], world, torch.full_like(world, torch.nan))


def d0_branches(bundle: CoordinateBundle) -> dict[str, torch.Tensor]:
    """Generate the six primary calibrated-K cells and direct point baseline."""

    depths = {
        "raw_depth_ref_affine": bundle.raw_depth_metric,
        "gt_depth": bundle.target_depth_metric,
    }
    poses = {
        "raw_pose": bundle.raw_world_to_camera_metric,
        "qk_pose": bundle.qk_world_to_camera_metric,
        "gt_pose": bundle.target_world_to_camera_metric,
    }
    output = {
        f"{depth_name}__{pose_name}__calibrated_k": unproject_z_depth(
            depth, bundle.calibrated_intrinsics, pose
        )
        for depth_name, depth in depths.items()
        for pose_name, pose in poses.items()
    }
    output["direct_raw_pointmap"] = bundle.raw_world_points_metric
    return output


def d0_predicted_k_supplement(bundle: CoordinateBundle) -> dict[str, torch.Tensor]:
    depths = {
        "raw_depth_ref_affine": bundle.raw_depth_metric,
        "gt_depth": bundle.target_depth_metric,
    }
    poses = {
        "raw_pose": bundle.raw_world_to_camera_metric,
        "qk_pose": bundle.qk_world_to_camera_metric,
        "gt_pose": bundle.target_world_to_camera_metric,
    }
    return {
        f"{depth_name}__{pose_name}__predicted_k": unproject_z_depth(
            depth, bundle.predicted_intrinsics, pose
        )
        for depth_name, depth in depths.items()
        for pose_name, pose in poses.items()
    }


def d0_uncalibrated_depth_supplement(
    bundle: CoordinateBundle,
) -> dict[str, torch.Tensor]:
    """Expose the invalid old point-head-scale assumption as report-only."""

    poses = {
        "raw_pose": bundle.raw_world_to_camera_metric,
        "qk_pose": bundle.qk_world_to_camera_metric,
        "gt_pose": bundle.target_world_to_camera_metric,
    }
    output = {}
    for intrinsics_name, intrinsics in (
        ("calibrated_k", bundle.calibrated_intrinsics),
        ("predicted_k", bundle.predicted_intrinsics),
    ):
        for pose_name, pose in poses.items():
            output[
                f"raw_depth_pointmap_scale__{pose_name}__{intrinsics_name}"
            ] = unproject_z_depth(
                bundle.raw_depth_pointmap_scale_metric, intrinsics, pose
            )
    return output


def common_support(
    branches: Iterable[torch.Tensor], target: torch.Tensor
) -> torch.Tensor:
    valid = torch.isfinite(target).all(dim=-1)
    for points in branches:
        valid &= torch.isfinite(points).all(dim=-1)
    return valid


def pointmap_metrics(
    predicted: torch.Tensor,
    target: torch.Tensor,
    support: torch.Tensor,
    *,
    paired_max_points: int,
    symmetric_max_points: int,
) -> tuple[dict[str, float | int], list[dict[str, float | int]]]:
    """Compute paired and sampled symmetric metrics on a fixed pixel support."""

    if predicted.shape != target.shape or support.shape != target.shape[:-1]:
        raise ValueError("pointmap metric tensors are not pixel-aligned.")
    rows: list[dict[str, float | int]] = []
    paired_errors: list[torch.Tensor] = []
    symmetric_values: list[float] = []
    total_supported = 0
    total_pixels = int(support.numel())
    for frame in range(predicted.shape[0]):
        selected = torch.nonzero(support[frame].reshape(-1), as_tuple=False)[:, 0]
        count = int(selected.numel())
        total_supported += count
        if count == 0:
            rows.append(_empty_metric_row(frame))
            continue
        paired_index = _even_limit(selected, int(paired_max_points))
        pred = predicted[frame].reshape(-1, 3).index_select(0, paired_index)
        truth = target[frame].reshape(-1, 3).index_select(0, paired_index)
        errors = torch.linalg.vector_norm(pred - truth, dim=-1)
        paired_errors.append(errors)
        symmetric_index = _even_limit(selected, int(symmetric_max_points))
        symmetric = symmetric_nearest_mean(
            predicted[frame].reshape(-1, 3).index_select(0, symmetric_index),
            target[frame].reshape(-1, 3).index_select(0, symmetric_index),
        )
        symmetric_values.append(symmetric)
        rows.append(
            {
                "sequence_index": frame,
                "supported_points": count,
                "paired_rmse": float(torch.sqrt(errors.square().mean())),
                "paired_median": float(errors.median()),
                "paired_p90": float(torch.quantile(errors, 0.9)),
                "symmetric_mean": symmetric,
            }
        )
    if not paired_errors:
        return {
            "supported_points": 0,
            "valid_ratio": 0.0,
            "paired_rmse": float("nan"),
            "paired_median": float("nan"),
            "paired_p90": float("nan"),
            "symmetric_mean": float("nan"),
        }, rows
    errors = torch.cat(paired_errors)
    return {
        "supported_points": total_supported,
        "valid_ratio": total_supported / max(total_pixels, 1),
        "paired_rmse": float(torch.sqrt(errors.square().mean())),
        "paired_median": float(errors.median()),
        "paired_p90": float(torch.quantile(errors, 0.9)),
        "symmetric_mean": float(sum(symmetric_values) / len(symmetric_values)),
    }, rows


def symmetric_nearest_mean(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() == 0 or right.numel() == 0:
        return float("nan")
    left_min = _chunked_nearest(left.float(), right.float())
    right_min = _chunked_nearest(right.float(), left.float())
    return float(0.5 * (left_min.mean() + right_min.mean()))


def point_errors(bundle: CoordinateBundle) -> tuple[torch.Tensor, torch.Tensor]:
    error = torch.linalg.vector_norm(
        bundle.raw_world_points_metric - bundle.target_world_points_metric, dim=-1
    )
    valid = (
        torch.isfinite(bundle.raw_world_points_metric).all(dim=-1)
        & torch.isfinite(bundle.target_world_points_metric).all(dim=-1)
        & torch.isfinite(bundle.confidence)
    )
    return error, valid


def ownership_map(
    masks: torch.Tensor, scores: torch.Tensor, *, score_threshold: float
) -> torch.Tensor:
    masks = masks.detach().bool().cpu()
    scores = scores.detach().float().cpu()
    if masks.ndim != 4 or scores.shape != masks.shape[:2]:
        raise ValueError("masks/scores must be [S,I,H,W] and [S,I].")
    eligible = masks & (scores[:, :, None, None] >= float(score_threshold))
    weighted = torch.where(
        eligible,
        scores[:, :, None, None].expand_as(masks),
        torch.full(masks.shape, -torch.inf),
    )
    slot = weighted.argmax(dim=1).long()
    return torch.where(eligible.any(dim=1), slot, torch.full_like(slot, -1))


def region_masks(
    ownership: torch.Tensor, *, boundary_width: int
) -> dict[str, torch.Tensor]:
    """Partition owned interiors/boundaries and the unprompted complement."""

    if ownership.ndim != 3 or int(boundary_width) < 1:
        raise ValueError("ownership must be [S,H,W] and boundary_width positive.")
    union = ownership >= 0
    eroded = torch.zeros_like(union)
    for slot in torch.unique(ownership[union]).tolist():
        eroded |= binary_erode(ownership == int(slot), int(boundary_width))
    dilated = binary_dilate(union, int(boundary_width))
    return {
        "instance_interior": eroded,
        "instance_inner_boundary": union & ~eroded,
        "instance_outer_boundary": dilated & ~union,
        "unprompted_complement": ~dilated,
    }


def binary_erode(mask: torch.Tensor, width: int) -> torch.Tensor:
    value = mask.float().unsqueeze(1)
    kernel = 2 * int(width) + 1
    return (
        -functional.max_pool2d(-value, kernel, stride=1, padding=int(width))
    ).squeeze(1) > 0.5


def binary_dilate(mask: torch.Tensor, width: int) -> torch.Tensor:
    value = mask.float().unsqueeze(1)
    kernel = 2 * int(width) + 1
    return functional.max_pool2d(
        value, kernel, stride=1, padding=int(width)
    ).squeeze(1) > 0.5


def distribution_row(
    values: torch.Tensor, support: torch.Tensor, **metadata: object
) -> dict[str, object]:
    selected = values[support & torch.isfinite(values)].float()
    row: dict[str, object] = dict(metadata)
    row["count"] = int(selected.numel())
    if selected.numel() == 0:
        row.update({name: float("nan") for name in ("mean", "median", "p90", "rmse")})
    else:
        row.update(
            {
                "mean": float(selected.mean()),
                "median": float(selected.median()),
                "p90": float(torch.quantile(selected, 0.9)),
                "rmse": float(torch.sqrt(selected.square().mean())),
            }
        )
    return row


def confidence_diagnostics(
    errors: torch.Tensor,
    confidence: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, float | int]]:
    selected_confidence = confidence[valid].float()
    selected_errors = errors[valid].float()
    if selected_errors.numel() < 10:
        raise ValueError("Confidence diagnosis requires at least ten valid points.")
    order = torch.argsort(selected_confidence)
    bins: list[dict[str, object]] = []
    count = int(order.numel())
    for bin_index in range(5):
        start = count * bin_index // 5
        stop = count * (bin_index + 1) // 5
        index = order[start:stop]
        mask = torch.ones(index.numel(), dtype=torch.bool)
        bins.append(
            distribution_row(
                selected_errors.index_select(0, index),
                mask,
                confidence_bin=f"{20 * bin_index}-{20 * (bin_index + 1)}%",
                confidence_min=float(selected_confidence.index_select(0, index).min()),
                confidence_max=float(selected_confidence.index_select(0, index).max()),
            )
        )
    descending = torch.argsort(selected_confidence, descending=True)
    risk_rows: list[dict[str, object]] = []
    for coverage in (0.2, 0.4, 0.6, 0.8, 1.0):
        keep = max(1, int(round(count * coverage)))
        kept = selected_errors.index_select(0, descending[:keep])
        risk_rows.append(
            {
                "coverage": coverage,
                "count": keep,
                "mean_error": float(kept.mean()),
                "rmse": float(torch.sqrt(kept.square().mean())),
                "p90": float(torch.quantile(kept, 0.9)),
            }
        )
    top_count = max(1, count // 5)
    overall_p90 = torch.quantile(selected_errors, 0.9)
    top_errors = selected_errors.index_select(0, descending[:top_count])
    wrong = top_errors > overall_p90
    high_wrong = {
        "valid_points": count,
        "overall_error_p90": float(overall_p90),
        "top_confidence_points": top_count,
        "high_confidence_wrong_points": int(wrong.sum()),
        "high_confidence_wrong_percent": 100.0 * float(wrong.float().mean()),
    }
    return bins, risk_rows, high_wrong


def _validate_bundle_shapes(bundle: CoordinateBundle) -> None:
    points_shape = bundle.target_world_points_metric.shape
    if len(points_shape) != 4 or points_shape[-1] != 3:
        raise ValueError("target_world_points must have shape [S,H,W,3].")
    sequence, height, width, _ = points_shape
    for name, value in (
        ("raw_world_points_metric", bundle.raw_world_points_metric),
        ("target_world_points_metric", bundle.target_world_points_metric),
    ):
        if value.shape != points_shape:
            raise ValueError(f"{name} is not aligned to target pointmap.")
    for name, value in (
        ("raw_depth_metric", bundle.raw_depth_metric),
        ("raw_depth_pointmap_scale_metric", bundle.raw_depth_pointmap_scale_metric),
        ("target_depth_metric", bundle.target_depth_metric),
        ("confidence", bundle.confidence),
    ):
        if value.shape != (sequence, height, width):
            raise ValueError(f"{name} is not aligned to target pointmap.")
    for name, value in (
        ("raw native pose", bundle.raw_world_to_camera_native),
        ("QK native pose", bundle.qk_world_to_camera_native),
        ("raw pose", bundle.raw_world_to_camera_metric),
        ("QK pose", bundle.qk_world_to_camera_metric),
        ("target pose", bundle.target_world_to_camera_metric),
    ):
        if value.shape != (sequence, 4, 4):
            raise ValueError(f"{name} sequence shape is invalid.")


def _squeeze_scalar(value: torch.Tensor, name: str) -> torch.Tensor:
    output = value.detach().float().cpu()
    if output.ndim == 4 and output.shape[-1] == 1:
        output = output[..., 0]
    if output.ndim != 3:
        raise ValueError(f"{name} must have shape [S,H,W] or [S,H,W,1].")
    return output


def _robust_affine_depth_alignment(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    trim_fraction: float = 0.8,
    iterations: int = 4,
) -> tuple[float, float, float]:
    """Fit one reference-frame depth affine and freeze it for the sequence."""

    valid = (
        torch.isfinite(source)
        & torch.isfinite(target)
        & (source > 0.0)
        & (target > 0.0)
    )
    source_values = source[valid].float().cpu()
    target_values = target[valid].float().cpu()
    if source_values.numel() < 1024:
        raise ValueError("Reference depth affine needs at least 1024 valid pixels.")
    if source_values.numel() > 100_000:
        index = torch.linspace(
            0, source_values.numel() - 1, steps=100_000
        ).long()
        source_values = source_values.index_select(0, index)
        target_values = target_values.index_select(0, index)
    keep = torch.ones(source_values.numel(), dtype=torch.bool)
    for _ in range(int(iterations)):
        scale, shift = _least_squares_affine(
            source_values[keep], target_values[keep]
        )
        residual = (scale * source_values + shift - target_values).abs()
        next_keep = residual <= torch.quantile(residual, float(trim_fraction))
        if int(next_keep.sum()) < 1024 or torch.equal(next_keep, keep):
            break
        keep = next_keep
    scale, shift = _least_squares_affine(
        source_values[keep], target_values[keep]
    )
    if not torch.isfinite(scale) or not torch.isfinite(shift) or scale <= 0.0:
        raise ValueError("Reference depth affine fit is invalid.")
    residual = scale * source_values[keep] + shift - target_values[keep]
    return float(scale), float(shift), float(torch.sqrt(residual.square().mean()))


def _least_squares_affine(
    source: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    source_mean = source.mean()
    target_mean = target.mean()
    centered = source - source_mean
    scale = (centered * (target - target_mean)).sum() / centered.square().sum().clamp_min(1e-12)
    shift = target_mean - scale * source_mean
    return scale, shift


def _even_limit(indices: torch.Tensor, limit: int) -> torch.Tensor:
    if indices.numel() <= limit:
        return indices
    positions = torch.linspace(0, indices.numel() - 1, steps=limit).long()
    return indices.index_select(0, positions)


def _chunked_nearest(query: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    output = []
    for start in range(0, query.shape[0], 512):
        distance = torch.cdist(query[start : start + 512], reference)
        output.append(distance.min(dim=1).values)
    return torch.cat(output)


def _empty_metric_row(frame: int) -> dict[str, float | int]:
    return {
        "sequence_index": frame,
        "supported_points": 0,
        "paired_rmse": float("nan"),
        "paired_median": float("nan"),
        "paired_p90": float("nan"),
        "symmetric_mean": float("nan"),
    }
