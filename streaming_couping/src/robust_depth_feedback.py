"""Conservative historical-depth Veto and affine-depth diagnostics.

The functions in this module are deliberately independent of SAM, GT, and
the cache format.  Callers can therefore build all Veto candidates and
causal depth-consistency pairs before opening annotations.

The Veto uses historical object observations as a *reference interval* and
the current predicted depth only as a test value.  This is intentionally
different from using the current-frame depth median as the ownership source:
the latter is retained as an explicit circularity control by the diagnostic
runner.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import torch


DEPTH_EPS = 1e-6
MAD_TO_SIGMA = 1.4826


@dataclass(frozen=True)
class HistoricalDepthVetoConfig:
    """Fixed, annotation-free policy for historical-depth Veto candidates."""

    confidence_threshold: float = 0.30
    track_score_threshold: float = 0.50
    min_history_points: int = 16
    max_history_points: int = 4096
    max_points_per_history_frame: int = 512
    mad_multiplier: float = 3.0
    # StreamVGGT's frozen pointmap/depth are in a native scene gauge.  The
    # legacy ``_m`` suffix is retained for command-line compatibility, but
    # this value is interpreted in the same native units as the two inputs;
    # the GT Sim(3) scale is intentionally unavailable during candidate
    # generation.
    absolute_padding_m: float = 0.05
    lower_quantile: float = 0.05
    upper_quantile: float = 0.95
    min_depth: float = DEPTH_EPS

    def validate(self) -> None:
        if not 0.0 <= float(self.confidence_threshold) <= 1.0:
            raise ValueError("confidence_threshold must be in [0,1].")
        if not 0.0 <= float(self.track_score_threshold) <= 1.0:
            raise ValueError("track_score_threshold must be in [0,1].")
        if int(self.min_history_points) < 1:
            raise ValueError("min_history_points must be positive.")
        if int(self.max_history_points) < int(self.min_history_points):
            raise ValueError(
                "max_history_points must be >= min_history_points."
            )
        if int(self.max_points_per_history_frame) < 1:
            raise ValueError("max_points_per_history_frame must be positive.")
        if float(self.mad_multiplier) < 0.0:
            raise ValueError("mad_multiplier must be non-negative.")
        if float(self.absolute_padding_m) < 0.0:
            raise ValueError("absolute_padding_m must be non-negative.")
        if not 0.0 <= float(self.lower_quantile) < 0.5:
            raise ValueError("lower_quantile must be in [0,0.5).")
        if not 0.5 < float(self.upper_quantile) <= 1.0:
            raise ValueError("upper_quantile must be in (0.5,1].")
        if float(self.lower_quantile) >= float(self.upper_quantile):
            raise ValueError("Depth quantiles must be ordered.")
        if float(self.min_depth) <= 0.0:
            raise ValueError("min_depth must be positive.")


@dataclass(frozen=True)
class DepthVetoResult:
    """A mask candidate plus auditable reference and removal statistics."""

    mask: torch.Tensor
    reference_kind: str
    history_points: int
    input_mask_pixels: int
    valid_current_pixels: int
    removed_pixels: int
    reference_median: float
    reference_mad: float
    lower_depth: float
    upper_depth: float
    fallback_used: bool
    fallback_reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "reference_kind": str(self.reference_kind),
            "history_points": int(self.history_points),
            "input_mask_pixels": int(self.input_mask_pixels),
            "valid_current_pixels": int(self.valid_current_pixels),
            "removed_pixels": int(self.removed_pixels),
            "removed_ratio": (
                float(self.removed_pixels) / float(self.input_mask_pixels)
                if self.input_mask_pixels
                else 0.0
            ),
            "reference_median": float(self.reference_median),
            "reference_mad": float(self.reference_mad),
            "lower_depth": float(self.lower_depth),
            "upper_depth": float(self.upper_depth),
            "fallback_used": int(self.fallback_used),
            "fallback_reason": str(self.fallback_reason),
        }


@dataclass(frozen=True)
class RobustAffineFit:
    """Result of a positive-slope robust affine fit ``target=a*source+b``."""

    accepted: bool
    status: str
    scale: float
    shift: float
    sample_count: int
    inlier_count: int
    iterations: int
    rmse_before: float
    rmse_after: float
    median_before: float
    median_after: float
    p90_before: float
    p90_after: float

    def to_dict(self) -> dict[str, object]:
        return {
            "accepted": int(self.accepted),
            "status": str(self.status),
            "scale": float(self.scale),
            "shift": float(self.shift),
            "sample_count": int(self.sample_count),
            "inlier_count": int(self.inlier_count),
            "iterations": int(self.iterations),
            "rmse_before": float(self.rmse_before),
            "rmse_after": float(self.rmse_after),
            "median_before": float(self.median_before),
            "median_after": float(self.median_after),
            "p90_before": float(self.p90_before),
            "p90_after": float(self.p90_after),
        }


def transform_world_points(
    world_points: torch.Tensor,
    world_to_camera: torch.Tensor,
) -> torch.Tensor:
    """Apply ``X_c = R X_w + t`` to one or a batched pointmap.

    A single pose has shape ``[3,4]`` or ``[4,4]``.  A batched pose has shape
    ``[S,3,4]`` or ``[S,4,4]`` and must match the leading sequence dimension
    of ``world_points``.  Supporting both forms here avoids accidentally
    interpreting a sequence of poses as one matrix in the scale-shift
    diagnostic.
    """

    points = _float_tensor(world_points, "world_points")
    pose = _float_tensor(world_to_camera, "world_to_camera")
    if points.ndim < 2 or tuple(points.shape[-1:]) != (3,):
        raise ValueError("world_points must end with dimension 3.")
    if pose.ndim == 2:
        if tuple(pose.shape) == (4, 4):
            pose = pose[:3]
        if tuple(pose.shape) != (3, 4):
            raise ValueError(
                "world_to_camera must have shape [3,4] or [4,4] for one pose."
            )
        return points @ pose[:, :3].T + pose[:, 3]
    if pose.ndim == 3:
        if tuple(pose.shape[-2:]) == (4, 4):
            pose = pose[:, :3]
        if tuple(pose.shape[-2:]) != (3, 4):
            raise ValueError(
                "Batched world_to_camera must have shape [S,3,4] or [S,4,4]."
            )
        if points.shape[0] != pose.shape[0]:
            raise ValueError(
                "Batched world_points and world_to_camera must share sequence S."
            )
        rotation = pose[:, :, :3]
        translation = pose[:, :, 3]
        translation_shape = (pose.shape[0],) + (1,) * (points.ndim - 2) + (3,)
        return (
            torch.einsum("s...j,sij->s...i", points, rotation)
            + translation.reshape(translation_shape)
        )
    raise ValueError(
        "world_to_camera must have shape [3,4], [4,4], [S,3,4], or [S,4,4]."
    )


def resize_intrinsics(
    intrinsics: torch.Tensor,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> torch.Tensor:
    """Map a pinhole matrix between grids using pixel-center coordinates.

    ``intrinsics`` is expressed in ``source_size`` pixel coordinates.  The
    returned matrix is expressed in ``target_size`` coordinates.  This is the
    same half-pixel convention used by the repository's image transforms and
    is needed when a depth head and a pointmap have different resolutions.
    """

    matrix = _float_tensor(intrinsics, "intrinsics")
    if tuple(matrix.shape) != (3, 3):
        raise ValueError("intrinsics must have shape [3,3].")
    source_height, source_width = _size_2d(source_size, "source_size")
    target_height, target_width = _size_2d(target_size, "target_size")
    scale_x = float(target_width) / float(source_width)
    scale_y = float(target_height) / float(source_height)
    output = matrix.clone()
    output[0, :2] *= scale_x
    output[1, :2] *= scale_y
    output[0, 2] = (matrix[0, 2] + 0.5) * scale_x - 0.5
    output[1, 2] = (matrix[1, 2] + 0.5) * scale_y - 0.5
    return output


def apply_ray_depth_affine(
    raw_world_points: torch.Tensor,
    source_depth: torch.Tensor,
    intrinsics: torch.Tensor,
    world_to_camera: torch.Tensor,
    *,
    scale: float,
    shift: float,
    update_mask: torch.Tensor | None = None,
    min_depth: float = DEPTH_EPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace depth along the existing pixel rays and return world points.

    The source depth is interpreted as camera ``Z`` rather than Euclidean
    range.  For each pixel ``q=(u,v,1)^T`` we compute

    ``r = K^{-1}q / (K^{-1}q)_z`` and ``X_c' = r * (scale*Z + shift)``.

    The output is transformed back with the inverse of the supplied rigid
    ``world_to_camera`` pose.  Only ``update_mask`` pixels are replaced; all
    other pixels, and every invalid result, retain the original pointmap.
    The returned boolean mask records exactly which pixels were replaced.
    """

    points = _float_tensor(raw_world_points, "raw_world_points")
    depth = _float_tensor(source_depth, "source_depth")
    matrix = _float_tensor(intrinsics, "intrinsics")
    w2c = _as_pose34(world_to_camera, "world_to_camera")
    common_dtype = torch.promote_types(
        torch.promote_types(points.dtype, depth.dtype),
        torch.promote_types(matrix.dtype, w2c.dtype),
    )
    points = points.to(dtype=common_dtype)
    depth = depth.to(dtype=common_dtype)
    matrix = matrix.to(dtype=common_dtype)
    w2c = w2c.to(dtype=common_dtype)
    if points.ndim != 3 or tuple(points.shape[-1:]) != (3,):
        raise ValueError("raw_world_points must have shape [H,W,3].")
    if depth.ndim != 2 or tuple(depth.shape) != tuple(points.shape[:2]):
        raise ValueError("source_depth must match raw_world_points [H,W].")
    if tuple(matrix.shape) != (3, 3):
        raise ValueError("intrinsics must have shape [3,3].")
    if not math.isfinite(float(scale)) or float(scale) <= 0.0:
        raise ValueError("scale must be finite and positive.")
    if not math.isfinite(float(shift)):
        raise ValueError("shift must be finite.")
    if float(min_depth) <= 0.0:
        raise ValueError("min_depth must be positive.")

    height, width = (int(value) for value in depth.shape)
    if update_mask is None:
        requested = torch.ones(height, width, dtype=torch.bool)
    else:
        requested = _as_bool_tensor(update_mask, "update_mask")
        if tuple(requested.shape) != (height, width):
            raise ValueError("update_mask must match source_depth [H,W].")

    finite_points = torch.isfinite(points).all(dim=-1)
    raw_camera = transform_world_points(points, w2c)
    finite_raw_camera = torch.isfinite(raw_camera).all(dim=-1)
    source_valid = torch.isfinite(depth) & (depth > float(min_depth))
    raw_valid = finite_points & finite_raw_camera & (raw_camera[..., 2] > float(min_depth))
    corrected_depth = float(scale) * depth + float(shift)
    corrected_valid = torch.isfinite(corrected_depth) & (
        corrected_depth > float(min_depth)
    )

    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=common_dtype),
        torch.arange(width, dtype=common_dtype),
        indexing="ij",
    )
    pixels = torch.stack(
        (
            xx.reshape(-1),
            yy.reshape(-1),
            torch.ones(height * width, dtype=common_dtype),
        ),
        dim=1,
    )
    inverse_intrinsics = torch.linalg.inv(matrix)
    rays = pixels @ inverse_intrinsics.T
    ray_z = rays[:, 2]
    ray_valid = torch.isfinite(rays).all(dim=1) & (
        ray_z.abs() > float(min_depth)
    )
    safe_ray_z = torch.where(
        ray_z.abs() > float(min_depth),
        ray_z,
        torch.ones_like(ray_z),
    )
    rays = rays / safe_ray_z.unsqueeze(1)
    camera_points = rays * corrected_depth.reshape(-1, 1)
    rotation = w2c[:, :3]
    translation = w2c[:, 3]
    world_points = (camera_points - translation.reshape(1, 3)) @ rotation
    world_points = world_points.reshape(height, width, 3)
    reconstructed_valid = torch.isfinite(world_points).all(dim=-1)
    applied = (
        requested
        & source_valid
        & raw_valid
        & corrected_valid
        & ray_valid.reshape(height, width)
        & reconstructed_valid
    )
    output = points.clone()
    output[applied] = world_points[applied]
    return output, applied


def gather_historical_object_depths(
    *,
    points: torch.Tensor,
    confidence: torch.Tensor,
    raw_masks: torch.Tensor,
    scores: torch.Tensor,
    current_frame: int,
    slot: int,
    current_world_to_camera: torch.Tensor,
    config: HistoricalDepthVetoConfig = HistoricalDepthVetoConfig(),
) -> torch.Tensor:
    """Collect causal same-slot depths in the current camera coordinate frame.

    Only frames ``s < current_frame`` are read.  Per-frame and global caps
    keep memory bounded and make the result deterministic through evenly
    spaced flat-index sampling.
    """

    config.validate()
    points = _float_tensor(points, "points")
    confidence = _float_tensor(confidence, "confidence")
    masks = _as_bool_tensor(raw_masks, "raw_masks")
    scores = _float_tensor(scores, "scores")
    _validate_sequence_shapes(points, confidence, masks, scores)
    sequence, height, width = points.shape[:3]
    frame = int(current_frame)
    object_slot = int(slot)
    if not 0 <= frame < sequence:
        raise ValueError("current_frame is outside the sequence.")
    if not 0 <= object_slot < masks.shape[1]:
        raise ValueError("slot is outside the raw mask slots.")

    historical_points = gather_historical_object_points(
        points=points,
        confidence=confidence,
        raw_masks=masks,
        scores=scores,
        current_frame=frame,
        slot=object_slot,
        config=config,
    )
    if not historical_points.numel():
        return torch.empty(0, dtype=torch.float32)
    camera = transform_world_points(historical_points, current_world_to_camera)
    depth = camera[:, 2]
    valid = torch.isfinite(depth) & (depth > float(config.min_depth))
    if not bool(valid.any()):
        return torch.empty(0, dtype=torch.float32)
    return depth[valid].detach().cpu().float()


def gather_historical_object_points(
    *,
    points: torch.Tensor,
    confidence: torch.Tensor,
    raw_masks: torch.Tensor,
    scores: torch.Tensor,
    current_frame: int,
    slot: int,
    config: HistoricalDepthVetoConfig = HistoricalDepthVetoConfig(),
) -> torch.Tensor:
    """Collect a bounded causal same-slot world-point cloud."""

    config.validate()
    points = _float_tensor(points, "points")
    confidence = _float_tensor(confidence, "confidence")
    masks = _as_bool_tensor(raw_masks, "raw_masks")
    scores = _float_tensor(scores, "scores")
    _validate_sequence_shapes(points, confidence, masks, scores)
    sequence = points.shape[0]
    frame = int(current_frame)
    object_slot = int(slot)
    if not 0 <= frame < sequence:
        raise ValueError("current_frame is outside the sequence.")
    if not 0 <= object_slot < masks.shape[1]:
        raise ValueError("slot is outside the raw mask slots.")

    chunks: list[torch.Tensor] = []
    for history_frame in range(frame):
        support = (
            masks[history_frame, object_slot]
            & torch.isfinite(confidence[history_frame])
            & (confidence[history_frame] >= float(config.confidence_threshold))
            & torch.isfinite(points[history_frame]).all(dim=-1)
            & (scores[history_frame, object_slot] >= float(config.track_score_threshold))
        )
        indices = torch.nonzero(support.reshape(-1), as_tuple=False).flatten()
        if not indices.numel():
            continue
        indices = _deterministic_subsample(
            indices,
            int(config.max_points_per_history_frame),
        )
        selected = points[history_frame].reshape(-1, 3).index_select(0, indices)
        chunks.append(selected.detach().cpu())

    if not chunks:
        return torch.empty(0, 3, dtype=torch.float32)
    result = torch.cat(chunks).float().cpu()
    return _deterministic_subsample_rows(result, int(config.max_history_points))


def build_causal_history_cache(
    *,
    points: torch.Tensor,
    confidence: torch.Tensor,
    raw_masks: torch.Tensor,
    scores: torch.Tensor,
    config: HistoricalDepthVetoConfig = HistoricalDepthVetoConfig(),
) -> list[list[torch.Tensor]]:
    """Build all bounded causal same-slot point clouds in one mask scan.

    The returned value is indexed as ``cache[current_frame][slot]``.  A
    history entry contains only frames strictly before ``current_frame`` and
    is already capped using the same deterministic policy as
    :func:`gather_historical_object_points`.  This is primarily an efficiency
    helper for diagnostics that reuse each causal query for several pose or
    Veto branches.
    """

    config.validate()
    points = _float_tensor(points, "points")
    confidence = _float_tensor(confidence, "confidence")
    masks = _as_bool_tensor(raw_masks, "raw_masks")
    scores = _float_tensor(scores, "scores")
    _validate_sequence_shapes(points, confidence, masks, scores)
    sequence = int(points.shape[0])
    slots = int(masks.shape[1])

    frame_points: list[list[torch.Tensor]] = [
        [torch.empty(0, 3, dtype=torch.float32) for _ in range(slots)]
        for _ in range(sequence)
    ]
    for history_frame in range(sequence):
        for slot in range(slots):
            support = (
                masks[history_frame, slot]
                & torch.isfinite(confidence[history_frame])
                & (confidence[history_frame] >= float(config.confidence_threshold))
                & torch.isfinite(points[history_frame]).all(dim=-1)
                & (scores[history_frame, slot] >= float(config.track_score_threshold))
            )
            indices = torch.nonzero(support.reshape(-1), as_tuple=False).flatten()
            if not indices.numel():
                continue
            indices = _deterministic_subsample(
                indices,
                int(config.max_points_per_history_frame),
            )
            frame_points[history_frame][slot] = (
                points[history_frame]
                .reshape(-1, 3)
                .index_select(0, indices)
                .detach()
                .cpu()
                .float()
            )

    cache: list[list[torch.Tensor]] = []
    prefix: list[list[torch.Tensor]] = [[] for _ in range(slots)]
    for current_frame in range(sequence):
        current_entries: list[torch.Tensor] = []
        for slot in range(slots):
            if prefix[slot]:
                combined = torch.cat(prefix[slot], dim=0)
                current_entries.append(
                    _deterministic_subsample_rows(
                        combined,
                        int(config.max_history_points),
                    )
                )
            else:
                current_entries.append(torch.empty(0, 3, dtype=torch.float32))
        cache.append(current_entries)
        for slot in range(slots):
            current = frame_points[current_frame][slot]
            if current.numel():
                prefix[slot].append(current)
    return cache


def apply_depth_veto(
    raw_mask: torch.Tensor,
    current_depth: torch.Tensor,
    reference_depths: torch.Tensor,
    *,
    reference_kind: str,
    config: HistoricalDepthVetoConfig = HistoricalDepthVetoConfig(),
    current_confidence: torch.Tensor | None = None,
) -> DepthVetoResult:
    """Apply a conservative interval Veto to one binary mask.

    ``reference_depths`` must come from the intended reference source.  The
    current depth map is never used to construct a historical reference in
    this function; the runner may explicitly pass current-mask depths for the
    circularity control branch.  All depth values and padding are in the
    native units of the supplied maps, not guaranteed metric metres.
    """

    config.validate()
    mask = _as_bool_tensor(raw_mask, "raw_mask")
    depth = _float_tensor(current_depth, "current_depth")
    reference = _float_tensor(reference_depths, "reference_depths").reshape(-1)
    if mask.ndim != 2 or depth.ndim != 2 or tuple(mask.shape) != tuple(depth.shape):
        raise ValueError("raw_mask and current_depth must share shape [H,W].")
    if current_confidence is not None:
        confidence = _float_tensor(current_confidence, "current_confidence")
        if tuple(confidence.shape) != tuple(mask.shape):
            raise ValueError("current_confidence must match raw_mask shape.")
    else:
        confidence = None

    input_pixels = int(mask.sum())
    valid_reference = reference[torch.isfinite(reference) & (reference > float(config.min_depth))]
    if valid_reference.numel() < int(config.min_history_points):
        return DepthVetoResult(
            mask=mask.clone(),
            reference_kind=str(reference_kind),
            history_points=int(valid_reference.numel()),
            input_mask_pixels=input_pixels,
            valid_current_pixels=0,
            removed_pixels=0,
            reference_median=float("nan"),
            reference_mad=float("nan"),
            lower_depth=float("nan"),
            upper_depth=float("nan"),
            fallback_used=True,
            fallback_reason="insufficient_reference_depths",
        )

    median = torch.median(valid_reference)
    mad = torch.median((valid_reference - median).abs())
    kind = str(reference_kind).strip().lower()
    if kind in {"history_median_mad", "current_depth_reference", "median_mad"}:
        half_width = max(
            float(config.absolute_padding_m),
            float(config.mad_multiplier) * MAD_TO_SIGMA * float(mad),
        )
        lower = float(median) - half_width
        upper = float(median) + half_width
    elif kind in {"history_quantile_interval", "quantile_interval"}:
        lower = float(
            torch.quantile(valid_reference, float(config.lower_quantile))
        ) - float(config.absolute_padding_m)
        upper = float(
            torch.quantile(valid_reference, float(config.upper_quantile))
        ) + float(config.absolute_padding_m)
    else:
        raise ValueError(f"Unknown depth reference kind: {reference_kind!r}.")

    valid_current = (
        mask
        & torch.isfinite(depth)
        & (depth > float(config.min_depth))
    )
    if confidence is not None:
        valid_current = (
            valid_current
            & torch.isfinite(confidence)
            & (confidence >= float(config.confidence_threshold))
        )
    outside = valid_current & ((depth < lower) | (depth > upper))
    refined = mask.clone()
    refined[outside] = False
    return DepthVetoResult(
        mask=refined,
        reference_kind=str(reference_kind),
        history_points=int(valid_reference.numel()),
        input_mask_pixels=input_pixels,
        valid_current_pixels=int(valid_current.sum()),
        removed_pixels=int(outside.sum()),
        reference_median=float(median),
        reference_mad=float(mad),
        lower_depth=float(lower),
        upper_depth=float(upper),
        fallback_used=False,
    )


def fit_robust_affine(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    min_samples: int = 64,
    max_samples: int = 200_000,
    trim_quantile: float = 0.90,
    max_iterations: int = 4,
    min_depth: float = DEPTH_EPS,
) -> RobustAffineFit:
    """Fit ``target = scale * source + shift`` with deterministic trimming."""

    if int(min_samples) < 2:
        raise ValueError("min_samples must be at least 2.")
    if int(max_samples) < int(min_samples):
        raise ValueError("max_samples must be >= min_samples.")
    if not 0.5 <= float(trim_quantile) < 1.0:
        raise ValueError("trim_quantile must be in [0.5,1).")
    if int(max_iterations) < 1:
        raise ValueError("max_iterations must be positive.")
    x = _float_tensor(source, "source").reshape(-1)
    y = _float_tensor(target, "target").reshape(-1)
    if x.shape != y.shape:
        raise ValueError("source and target must have the same number of values.")
    valid = (
        torch.isfinite(x)
        & torch.isfinite(y)
        & (x > float(min_depth))
        & (y > float(min_depth))
    )
    x = x[valid].double().cpu()
    y = y[valid].double().cpu()
    if x.numel() > int(max_samples):
        indices = _deterministic_subsample(
            torch.arange(x.numel(), dtype=torch.long),
            int(max_samples),
        )
        x = x.index_select(0, indices)
        y = y.index_select(0, indices)
    sample_count = int(x.numel())
    if sample_count < int(min_samples):
        return _empty_affine_fit(sample_count, "insufficient_samples")

    active = torch.ones(sample_count, dtype=torch.bool)
    scale = float("nan")
    shift = float("nan")
    iterations = 0
    for iteration in range(int(max_iterations)):
        if int(active.sum()) < int(min_samples):
            return _empty_affine_fit(sample_count, "insufficient_inliers")
        scale, shift = _least_squares_affine(x[active], y[active])
        iterations = iteration + 1
        if not math.isfinite(scale) or not math.isfinite(shift) or scale <= 0.0:
            return _empty_affine_fit(sample_count, "non_positive_scale", iterations)
        residual = (y - (scale * x + shift)).abs()
        threshold = float(torch.quantile(residual[active], float(trim_quantile)))
        next_active = residual <= threshold
        if int(next_active.sum()) < int(min_samples):
            break
        if torch.equal(next_active, active):
            break
        active = next_active

    if not math.isfinite(scale) or not math.isfinite(shift) or scale <= 0.0:
        return _empty_affine_fit(sample_count, "non_positive_scale", iterations)
    before = affine_error_metrics(x, y, scale=1.0, shift=0.0)
    after = affine_error_metrics(x, y, scale=scale, shift=shift)
    return RobustAffineFit(
        accepted=True,
        status="ok",
        scale=float(scale),
        shift=float(shift),
        sample_count=sample_count,
        inlier_count=int(active.sum()),
        iterations=iterations,
        rmse_before=float(before["rmse"]),
        rmse_after=float(after["rmse"]),
        median_before=float(before["median"]),
        median_after=float(after["median"]),
        p90_before=float(before["p90"]),
        p90_after=float(after["p90"]),
    )


def affine_error_metrics(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    scale: float,
    shift: float,
) -> dict[str, float]:
    """Return absolute residual statistics for a fixed affine transform."""

    x = _float_tensor(source, "source").reshape(-1)
    y = _float_tensor(target, "target").reshape(-1)
    if x.shape != y.shape:
        raise ValueError("source and target must share shape.")
    valid = torch.isfinite(x) & torch.isfinite(y)
    if not bool(valid.any()):
        return {"count": 0.0, "rmse": float("nan"), "median": float("nan"), "p90": float("nan")}
    residual = (y[valid].double() - (float(scale) * x[valid].double() + float(shift))).abs()
    return {
        "count": float(residual.numel()),
        "rmse": float(torch.sqrt(residual.square().mean())),
        "median": float(torch.median(residual)),
        "p90": float(torch.quantile(residual, 0.90)),
    }


def _least_squares_affine(source: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    design = torch.stack((source, torch.ones_like(source)), dim=1)
    solution = torch.linalg.lstsq(design, target[:, None]).solution[:, 0]
    return float(solution[0]), float(solution[1])


def _empty_affine_fit(
    sample_count: int,
    status: str,
    iterations: int = 0,
) -> RobustAffineFit:
    nan = float("nan")
    return RobustAffineFit(
        accepted=False,
        status=str(status),
        scale=nan,
        shift=nan,
        sample_count=int(sample_count),
        inlier_count=0,
        iterations=int(iterations),
        rmse_before=nan,
        rmse_after=nan,
        median_before=nan,
        median_after=nan,
        p90_before=nan,
        p90_after=nan,
    )


def _validate_sequence_shapes(
    points: torch.Tensor,
    confidence: torch.Tensor,
    raw_masks: torch.Tensor,
    scores: torch.Tensor,
) -> None:
    if points.ndim != 4 or tuple(points.shape[-1:]) != (3,):
        raise ValueError("points must have shape [S,H,W,3].")
    sequence, height, width = points.shape[:3]
    if tuple(confidence.shape) != (sequence, height, width):
        raise ValueError("confidence must have shape [S,H,W].")
    if raw_masks.ndim != 4 or tuple(raw_masks.shape[0:1]) != (sequence,):
        raise ValueError("raw_masks must have shape [S,K,H,W].")
    if tuple(raw_masks.shape[-2:]) != (height, width):
        raise ValueError("raw_masks spatial size must match points.")
    if tuple(scores.shape) != tuple(raw_masks.shape[:2]):
        raise ValueError("scores must have shape [S,K].")


def _deterministic_subsample(values: torch.Tensor, limit: int) -> torch.Tensor:
    values = values.reshape(-1)
    if int(limit) <= 0:
        raise ValueError("subsample limit must be positive.")
    if values.numel() <= int(limit):
        return values
    positions = torch.linspace(
        0,
        values.numel() - 1,
        steps=int(limit),
        dtype=torch.float64,
    ).round().long()
    return values.index_select(0, positions.to(values.device))


def _deterministic_subsample_rows(values: torch.Tensor, limit: int) -> torch.Tensor:
    if values.ndim != 2:
        raise ValueError("row subsampling expects a matrix.")
    if int(limit) <= 0:
        raise ValueError("subsample limit must be positive.")
    if values.shape[0] <= int(limit):
        return values
    positions = torch.linspace(
        0,
        values.shape[0] - 1,
        steps=int(limit),
        dtype=torch.float64,
    ).round().long()
    return values.index_select(0, positions.to(values.device))


def _float_tensor(value: torch.Tensor, name: str) -> torch.Tensor:
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    if not value.is_floating_point():
        value = value.float()
    return value.detach().cpu()


def _as_bool_tensor(value: torch.Tensor, name: str) -> torch.Tensor:
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    return value.detach().cpu().bool()


def _as_pose34(value: torch.Tensor, name: str) -> torch.Tensor:
    pose = _float_tensor(value, name)
    if tuple(pose.shape) == (4, 4):
        pose = pose[:3]
    if tuple(pose.shape) != (3, 4):
        raise ValueError(f"{name} must have shape [3,4] or [4,4].")
    return pose


def _size_2d(value: tuple[int, int], name: str) -> tuple[int, int]:
    if len(value) != 2:
        raise ValueError(f"{name} must contain (height,width).")
    height, width = int(value[0]), int(value[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"{name} must contain positive dimensions.")
    return height, width
