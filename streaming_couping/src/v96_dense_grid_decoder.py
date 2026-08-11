"""Actual-grid coordinate upper bounds for the V9.6 diagnosis.

This module does not read or synthesize descriptors.  It limits current and
history coordinates to the real resized SAM detector grid and asks whether a
hard key or a four-key bilinear expectation can represent the correspondence.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from streaming_couping.src.v90_epipolar_geometry import (
    LocalTokenReprojection,
    SurfaceCorrespondences,
)
from streaming_couping.src.v95_spatial_support import (
    clone_surface,
    concatenate_surface_rows,
    empty_surface,
    take_surface,
)


GRID_DECODERS = ("continuous_gt", "hard_nearest", "soft_bilinear_k4")
GRID_SUPPORT_SCOPES = (
    "full_grid",
    "sam_mask_balanced",
    "bbox_balanced",
    "random_shifted_mask_balanced",
)


@dataclass(frozen=True)
class GridSupport:
    scope: str
    correspondences: SurfaceCorrespondences
    region_rows: int
    complement_rows: int

    @property
    def count(self) -> int:
        return self.correspondences.count


@dataclass(frozen=True)
class GridDecode:
    mode: str
    correspondences: SurfaceCorrespondences
    errors_pixels: torch.Tensor
    keys_per_query: int


def dense_grid_normalized(grid_size: tuple[int, int]) -> torch.Tensor:
    """Return row-major align-corners UV for the resized SAM feature grid."""

    height, width = (int(value) for value in grid_size)
    if height < 2 or width < 2:
        raise ValueError("V9.6 dense grid dimensions must be at least two.")
    y = torch.linspace(-1.0, 1.0, height, dtype=torch.float64)
    x = torch.linspace(-1.0, 1.0, width, dtype=torch.float64)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([xx, yy], dim=-1).reshape(-1, 2)


def bbox_mask_stream(instance_masks: torch.Tensor) -> torch.Tensor:
    """Convert each frame's union SAM mask into one tight bbox mask."""

    union = _union_masks(instance_masks)
    output = torch.zeros_like(union)
    for frame in range(union.shape[0]):
        yx = torch.nonzero(union[frame, 0], as_tuple=False)
        if not int(yx.shape[0]):
            continue
        low = yx.min(dim=0).values
        high = yx.max(dim=0).values
        output[frame, 0, low[0] : high[0] + 1, low[1] : high[1] + 1] = True
    return output


def random_shifted_mask_stream(
    instance_masks: torch.Tensor, *, seed: int = 96
) -> torch.Tensor:
    """Area/shape-matched deterministic mask control with destroyed alignment."""

    union = _union_masks(instance_masks)
    output = torch.zeros_like(union)
    height, width = union.shape[-2:]
    for frame in range(union.shape[0]):
        if not bool(union[frame, 0].any()):
            continue
        generator = torch.Generator(device="cpu").manual_seed(
            int(seed) + frame * 10_007
        )
        min_y = max(height // 4, 1)
        min_x = max(width // 4, 1)
        span_y = max(height - min_y, 1)
        span_x = max(width - min_x, 1)
        shift_y = min_y + int(torch.randint(span_y, (1,), generator=generator))
        shift_x = min_x + int(torch.randint(span_x, (1,), generator=generator))
        output[frame, 0] = torch.roll(
            union[frame, 0], shifts=(shift_y, shift_x), dims=(0, 1)
        )
    return output


def filter_grid_labels_by_region(
    labels: LocalTokenReprojection, region_masks: torch.Tensor
) -> SurfaceCorrespondences:
    """Retain full-grid GT labels lying in the region in both views."""

    if region_masks.ndim != 4 or region_masks.shape[1] != 1:
        raise ValueError("V9.6 region masks must be [S,1,H,W].")
    current = labels.current_frame
    history = labels.history_frame
    current_region = _nearest_mask(region_masks[current, 0], labels.current_uv)
    history_region = _nearest_mask(
        region_masks[history, 0], labels.history_target_uv
    )
    visible = (
        labels.query_valid.bool()
        & labels.target_visible.bool()
        & current_region
        & history_region
    )
    indices = torch.nonzero(visible, as_tuple=False).flatten()
    return _surface_from_label_indices(labels, indices)


def build_equal_count_grid_supports(
    *,
    full: SurfaceCorrespondences,
    regions: dict[str, tuple[SurfaceCorrespondences, SurfaceCorrespondences]],
    target_count: int,
    image_size: tuple[int, int],
    complement_fraction: float = 0.5,
) -> dict[str, GridSupport]:
    """Build full and balanced actual-grid supports using current UV only."""

    target = int(target_count)
    if target < 0:
        raise ValueError("V9.6 target count cannot be negative.")
    if set(regions) != set(GRID_SUPPORT_SCOPES[1:]):
        raise ValueError("V9.6 balanced region scopes changed.")
    if target == 0:
        return {
            scope: GridSupport(
                scope,
                empty_surface(full.current_frame, full.history_frame),
                0,
                0,
            )
            for scope in GRID_SUPPORT_SCOPES
        }
    fraction = float(complement_fraction)
    if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError("V9.6 complement fraction must lie inside (0,1).")
    selected_full = take_surface(
        full, current_farthest_indices(full, target, image_size)
    )
    if selected_full.count != target:
        raise ValueError(
            f"V9.6 full-grid rows={selected_full.count}, required={target}."
        )
    output = {
        "full_grid": GridSupport("full_grid", selected_full, 0, selected_full.count)
    }
    desired_complement = int(round(target * fraction))
    desired_region = target - desired_complement
    for scope in GRID_SUPPORT_SCOPES[1:]:
        region, complement = regions[scope]
        selected_region = take_surface(
            region, current_farthest_indices(region, desired_region, image_size)
        )
        selected_complement = take_surface(
            complement,
            current_farthest_indices(complement, desired_complement, image_size),
        )
        # Some coarse controls can cover the entire mutually visible image on
        # one edge (for example, a union bbox), leaving no valid complement.
        # Keep the available rows and let equal_count_exact/fold_pass mark that
        # control infeasible.  A missing negative control must not abort the
        # full-grid and SAM-balanced coordinate gates, nor count as a SAM win.
        combined = concatenate_surface_rows(
            [selected_region, selected_complement],
            current_frame=full.current_frame,
            history_frame=full.history_frame,
        )
        output[scope] = GridSupport(
            scope,
            combined,
            selected_region.count,
            selected_complement.count,
        )
    if tuple(output) != GRID_SUPPORT_SCOPES:
        raise RuntimeError("V9.6 grid scope order changed.")
    return output


def decode_history_grid(
    row: SurfaceCorrespondences,
    *,
    mode: str,
    grid_size: tuple[int, int],
    image_size: tuple[int, int],
) -> GridDecode:
    """Decode continuous targets using actual history-grid coordinates."""

    if mode not in GRID_DECODERS:
        raise ValueError(f"Unknown V9.6 grid decoder={mode!r}.")
    if row.count == 0:
        return GridDecode(mode, clone_surface(row), torch.empty(0), 0)
    target = row.history_uv.double()
    if not bool(torch.isfinite(target).all()):
        raise ValueError("V9.6 history targets must be finite.")
    if mode == "continuous_gt":
        predicted = target.clone()
        keys_per_query = 0
    else:
        height, width = (int(value) for value in image_size)
        grid_height, grid_width = (int(value) for value in grid_size)
        x_axis = torch.linspace(0.0, width - 1, grid_width, dtype=torch.float64)
        y_axis = torch.linspace(0.0, height - 1, grid_height, dtype=torch.float64)
        if mode == "hard_nearest":
            x = x_axis[torch.cdist(target[:, :1], x_axis[:, None]).argmin(dim=1)]
            y = y_axis[torch.cdist(target[:, 1:], y_axis[:, None]).argmin(dim=1)]
            predicted = torch.stack([x, y], dim=-1)
            keys_per_query = 1
        else:
            predicted = _bilinear_grid_expectation(target, x_axis, y_axis)
            keys_per_query = 4
    errors = torch.linalg.vector_norm(predicted - target, dim=-1)
    output = clone_surface(row)
    output.history_uv = predicted
    return GridDecode(mode, output, errors, keys_per_query)


def current_farthest_indices(
    row: SurfaceCorrespondences,
    count: int,
    image_size: tuple[int, int],
) -> torch.Tensor:
    """Deterministic deployable FPS using only current-frame coordinates."""

    requested = max(0, min(int(count), row.count))
    if requested == 0:
        return torch.empty(0, dtype=torch.long)
    if requested == row.count:
        return torch.arange(row.count, dtype=torch.long)
    height, width = (int(value) for value in image_size)
    scale = torch.tensor(
        [max(width - 1, 1), max(height - 1, 1)], dtype=torch.float64
    )
    features = row.current_uv.double() / scale
    finite = torch.isfinite(features).all(dim=-1)
    if int(finite.sum()) < requested:
        raise ValueError("V9.6 grid support contains too few finite current UV rows.")
    valid = torch.nonzero(finite, as_tuple=False).flatten()
    features = features.index_select(0, valid)
    center = features.mean(dim=0)
    first = torch.linalg.vector_norm(features - center, dim=-1).argmin()
    selected = torch.empty(requested, dtype=torch.long)
    selected[0] = first
    minimum = (features - features[first]).square().sum(dim=-1)
    minimum[first] = -1.0
    for index in range(1, requested):
        choice = minimum.argmax()
        selected[index] = choice
        distance = (features - features[choice]).square().sum(dim=-1)
        minimum = torch.minimum(minimum, distance)
        minimum[choice] = -1.0
    return valid.index_select(0, selected)


def _bilinear_grid_expectation(
    target: torch.Tensor, x_axis: torch.Tensor, y_axis: torch.Tensor
) -> torch.Tensor:
    x = target[:, 0].clamp(float(x_axis[0]), float(x_axis[-1]))
    y = target[:, 1].clamp(float(y_axis[0]), float(y_axis[-1]))
    upper_x = torch.searchsorted(x_axis, x, right=True).clamp(1, len(x_axis) - 1)
    upper_y = torch.searchsorted(y_axis, y, right=True).clamp(1, len(y_axis) - 1)
    lower_x = upper_x - 1
    lower_y = upper_y - 1
    x0, x1 = x_axis[lower_x], x_axis[upper_x]
    y0, y1 = y_axis[lower_y], y_axis[upper_y]
    amount_x = (x - x0) / (x1 - x0).clamp_min(1e-12)
    amount_y = (y - y0) / (y1 - y0).clamp_min(1e-12)
    keys = torch.stack(
        [
            torch.stack([x0, y0], dim=-1),
            torch.stack([x1, y0], dim=-1),
            torch.stack([x0, y1], dim=-1),
            torch.stack([x1, y1], dim=-1),
        ],
        dim=1,
    )
    weights = torch.stack(
        [
            (1.0 - amount_x) * (1.0 - amount_y),
            amount_x * (1.0 - amount_y),
            (1.0 - amount_x) * amount_y,
            amount_x * amount_y,
        ],
        dim=-1,
    )
    return (weights[..., None] * keys).sum(dim=1)


def _surface_from_label_indices(
    labels: LocalTokenReprojection, indices: torch.Tensor
) -> SurfaceCorrespondences:
    indices = indices.long().cpu()
    return SurfaceCorrespondences(
        current_frame=labels.current_frame,
        history_frame=labels.history_frame,
        slot=-1,
        current_uv=labels.current_uv.index_select(0, indices).double(),
        history_uv=labels.history_target_uv.index_select(0, indices).double(),
        weights=labels.weights.index_select(0, indices).double(),
        depth_residual_metric=labels.depth_residual_metric.index_select(
            0, indices
        ).double(),
        sampled_queries=labels.query_count,
        projected_in_bounds=labels.query_count,
        visible_queries=int(indices.numel()),
    )


def _nearest_mask(mask: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    height, width = mask.shape
    finite = torch.isfinite(uv).all(dim=-1)
    x = uv[:, 0].nan_to_num(0.0).round().long().clamp(0, width - 1)
    y = uv[:, 1].nan_to_num(0.0).round().long().clamp(0, height - 1)
    return finite & mask[y, x].bool()


def _union_masks(instance_masks: torch.Tensor) -> torch.Tensor:
    if instance_masks.ndim != 4:
        raise ValueError("V9.6 instance masks must be [S,K,H,W].")
    return instance_masks.bool().any(dim=1, keepdim=True)
