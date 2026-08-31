"""Explicit spatial-temporal feedback primitives for the V0 diagnostic.

The classes in this module intentionally do not know about SAM3.1 or
StreamVGGT model internals.  They operate on tensors produced by either model
and can therefore be unit-tested with small synthetic inputs.

There are three important scope boundaries:

* :class:`StaticBackgroundPoseOptimizer` masks an already-computed tracking
  loss.  It does not compute that loss and it does not update a camera pose.
* :class:`DepthGuidedMaskRefiner` is a deterministic depth-cluster heuristic.
  A dominant depth mode is not a proof of object ownership, especially for
  articulated or heavily occluded objects.
* :class:`TemporalPromptProjector` only produces causal 2-D point priors.  It
  does not call a SAM predictor and consequently cannot by itself claim to
  remove temporal flicker.

Keeping these boundaries explicit makes the V0 experiment auditable: a
positive result from one primitive cannot be mistaken for a trained,
closed-loop model result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn


class StaticBackgroundPoseOptimizer(nn.Module):
    r"""Restrict a pre-computed tracking loss to the static background.

    Given a loss map :math:`L \in R^{B\times H\times W}` and instance masks
    :math:`M \in \{0,1\}^{B\times N\times H\times W}`, the foreground union
    and its complement are

    .. math::

       M_{fg}(b,h,w)=\bigvee_n M(b,n,h,w),\qquad
       M_{bg}=1-M_{fg}.

    The returned scalar is the mean of ``L`` over ``M_bg`` (and an optional
    validity mask).  Multiplication by the boolean mask leaves gradients with
    respect to the loss map only at background pixels.  The mask itself is a
    discrete routing decision and is not differentiated.
    """

    def __init__(self, *, reduction: str = "mean") -> None:
        super().__init__()
        reduction = str(reduction).strip().lower()
        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be 'mean' or 'sum'.")
        self.reduction = reduction

    @staticmethod
    def merge_foreground_mask(sam_instance_masks: torch.Tensor) -> torch.Tensor:
        """Return the union of all instance masks as ``[B,H,W]`` booleans."""

        masks = _as_tensor(sam_instance_masks, name="sam_instance_masks")
        if masks.ndim != 4:
            raise ValueError(
                "sam_instance_masks must have shape [B,N,H,W], "
                f"got {tuple(masks.shape)}."
            )
        return masks.bool().any(dim=1)

    @classmethod
    def static_background_mask(
        cls,
        sam_instance_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Return the complement of the merged foreground mask."""

        return ~cls.merge_foreground_mask(sam_instance_masks)

    def forward(
        self,
        vggt_tracking_loss_map: torch.Tensor,
        sam_instance_masks: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the background-only scalar tracking loss.

        ``valid_mask`` is optional and is useful for invalid depth/ray pixels.
        It must have shape ``[B,H,W]``.  If no valid background pixel exists,
        the returned value is a differentiable zero (rather than a NaN).
        """

        loss_map = _as_tensor(
            vggt_tracking_loss_map,
            name="vggt_tracking_loss_map",
        )
        if loss_map.ndim != 3:
            raise ValueError(
                "vggt_tracking_loss_map must have shape [B,H,W], "
                f"got {tuple(loss_map.shape)}."
            )
        if not (loss_map.is_floating_point() or loss_map.is_complex()):
            raise TypeError("vggt_tracking_loss_map must be floating point.")

        masks = _as_tensor(sam_instance_masks, name="sam_instance_masks")
        expected_mask_shape = (
            int(loss_map.shape[0]),
            int(loss_map.shape[1]),
            int(loss_map.shape[2]),
        )
        if masks.ndim != 4 or tuple(
            (masks.shape[0], masks.shape[2], masks.shape[3])
        ) != expected_mask_shape:
            raise ValueError(
                "loss map and SAM masks disagree: expected masks [B,N,H,W] "
                f"for loss {tuple(loss_map.shape)}, got {tuple(masks.shape)}."
            )
        if masks.device != loss_map.device:
            masks = masks.to(loss_map.device)

        background = ~masks.bool().any(dim=1)
        effective = background
        if valid_mask is not None:
            validity = _as_tensor(valid_mask, name="valid_mask")
            if tuple(validity.shape) != tuple(loss_map.shape):
                raise ValueError(
                    "valid_mask must have shape [B,H,W] matching the loss map, "
                    f"got {tuple(validity.shape)}."
                )
            effective = effective & validity.to(loss_map.device).bool()

        weights = effective.to(dtype=loss_map.dtype)
        masked = loss_map * weights
        if self.reduction == "sum":
            return masked.sum()
        denominator = weights.sum()
        # Multiplying by zero keeps the output connected to loss_map even for
        # an empty background, while the clamp avoids a NaN divide.
        return masked.sum() / denominator.clamp_min(1.0)

    def masked_loss(
        self,
        vggt_tracking_loss_map: torch.Tensor,
        sam_instance_masks: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Named alias for :meth:`forward` used by training-loop adapters."""

        return self.forward(
            vggt_tracking_loss_map,
            sam_instance_masks,
            valid_mask=valid_mask,
        )


@dataclass(frozen=True)
class DepthRefinementStats:
    """Auditable statistics emitted by one depth-guided refinement."""

    input_mask_pixels: int
    valid_depth_pixels: int
    cluster_count: int
    selected_cluster_pixels: int
    removed_pixels: int
    depth_gap_threshold: float
    fallback_used: bool
    fallback_reason: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "input_mask_pixels": int(self.input_mask_pixels),
            "valid_depth_pixels": int(self.valid_depth_pixels),
            "cluster_count": int(self.cluster_count),
            "selected_cluster_pixels": int(self.selected_cluster_pixels),
            "removed_pixels": int(self.removed_pixels),
            "removed_pixel_ratio": (
                float(self.removed_pixels) / float(self.input_mask_pixels)
                if self.input_mask_pixels
                else 0.0
            ),
            "depth_gap_threshold": float(self.depth_gap_threshold),
            "fallback_used": int(self.fallback_used),
            "fallback_reason": str(self.fallback_reason),
        }


class DepthGuidedMaskRefiner:
    r"""Refine a binary mask using deterministic 1-D depth clustering.

    Depth values inside the mask are sorted and split whenever the adjacent
    depth gap exceeds

    .. math:: ``max(absolute_gap, relative_gap * median(|depth|))``.

    The largest resulting cluster is retained.  This is deliberately a small,
    inspectable heuristic rather than a learned segmentation model.  In
    particular, ``dominant_cluster='largest'`` assumes that the object's
    visible surface occupies the dominant depth mode; callers should retain
    the returned statistics and treat the result as a diagnostic proposal.
    """

    def __init__(
        self,
        *,
        absolute_gap: float = 0.05,
        relative_gap: float = 0.05,
        fallback_on_invalid_depth: bool = True,
    ) -> None:
        if float(absolute_gap) < 0.0 or float(relative_gap) < 0.0:
            raise ValueError("Depth gap thresholds must be non-negative.")
        if float(absolute_gap) == 0.0 and float(relative_gap) == 0.0:
            raise ValueError("At least one depth gap threshold must be positive.")
        self.absolute_gap = float(absolute_gap)
        self.relative_gap = float(relative_gap)
        self.fallback_on_invalid_depth = bool(fallback_on_invalid_depth)

    def __call__(
        self,
        raw_sam_mask: torch.Tensor,
        vggt_depth: torch.Tensor,
        *,
        return_stats: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, DepthRefinementStats]:
        return self.refine(
            raw_sam_mask,
            vggt_depth,
            return_stats=return_stats,
        )

    def refine(
        self,
        raw_sam_mask: torch.Tensor,
        vggt_depth: torch.Tensor,
        *,
        return_stats: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, DepthRefinementStats]:
        mask = _as_tensor(raw_sam_mask, name="raw_sam_mask").bool()
        depth = _as_tensor(vggt_depth, name="vggt_depth")
        if mask.ndim != 2:
            raise ValueError(
                f"raw_sam_mask must have shape [H,W], got {tuple(mask.shape)}."
            )
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if depth.ndim != 2 or tuple(depth.shape) != tuple(mask.shape):
            raise ValueError(
                "vggt_depth must have shape [H,W] (or [H,W,1]) matching the "
                f"mask, got depth={tuple(depth.shape)} mask={tuple(mask.shape)}."
            )
        if depth.device != mask.device:
            depth = depth.to(mask.device)
        if not depth.is_floating_point():
            depth = depth.float()

        input_pixels = int(mask.sum().item())
        if input_pixels == 0:
            stats = DepthRefinementStats(
                input_mask_pixels=0,
                valid_depth_pixels=0,
                cluster_count=0,
                selected_cluster_pixels=0,
                removed_pixels=0,
                depth_gap_threshold=0.0,
                fallback_used=False,
                fallback_reason="empty_mask",
            )
            refined = mask.clone()
            return (refined, stats) if return_stats else refined
        valid = mask & torch.isfinite(depth) & (depth > 0)
        valid_count = int(valid.sum().item())
        if valid_count == 0:
            if self.fallback_on_invalid_depth:
                refined = mask.clone()
                reason = "no_valid_depth"
            else:
                refined = torch.zeros_like(mask)
                reason = "no_valid_depth_empty"
            stats = DepthRefinementStats(
                input_mask_pixels=input_pixels,
                valid_depth_pixels=0,
                cluster_count=0,
                selected_cluster_pixels=0,
                removed_pixels=int((mask & ~refined).sum().item()),
                depth_gap_threshold=0.0,
                fallback_used=True,
                fallback_reason=reason,
            )
            return (refined, stats) if return_stats else refined

        values = depth[valid].float()
        sorted_values, order = torch.sort(values)
        scale = torch.median(sorted_values.abs()).clamp_min(1e-6)
        threshold = max(
            self.absolute_gap,
            self.relative_gap * float(scale.item()),
        )
        if sorted_values.numel() == 1:
            starts = torch.ones(1, dtype=torch.bool, device=sorted_values.device)
        else:
            gaps = sorted_values[1:] - sorted_values[:-1]
            starts = torch.cat(
                (
                    torch.ones(1, dtype=torch.bool, device=sorted_values.device),
                    gaps > threshold,
                )
            )
        cluster_ids_sorted = torch.cumsum(starts.to(torch.long), dim=0) - 1
        cluster_count = int(cluster_ids_sorted[-1].item()) + 1
        counts = torch.bincount(
            cluster_ids_sorted,
            minlength=cluster_count,
        )
        selected_cluster = int(torch.argmax(counts).item())
        # ``order`` maps sorted positions back to the compact valid-value
        # vector.  Scatter the cluster labels into that original order.
        cluster_ids = torch.empty_like(cluster_ids_sorted)
        cluster_ids.scatter_(0, order, cluster_ids_sorted)
        valid_positions = torch.nonzero(valid, as_tuple=False)
        keep_positions = valid_positions[cluster_ids == selected_cluster]
        refined = torch.zeros_like(mask)
        if keep_positions.numel():
            refined[keep_positions[:, 0], keep_positions[:, 1]] = True
        removed = int((mask & ~refined).sum().item())
        stats = DepthRefinementStats(
            input_mask_pixels=input_pixels,
            valid_depth_pixels=valid_count,
            cluster_count=cluster_count,
            selected_cluster_pixels=int(counts[selected_cluster].item()),
            removed_pixels=removed,
            depth_gap_threshold=float(threshold),
            fallback_used=False,
        )
        return (refined, stats) if return_stats else refined


@dataclass(frozen=True)
class TemporalProjection:
    """Result of projecting a set of world-space object centers."""

    points_uv: torch.Tensor
    object_ids: torch.Tensor
    depth: torch.Tensor
    camera_points: torch.Tensor
    valid_mask: torch.Tensor
    all_points_uv: torch.Tensor
    all_depth: torch.Tensor

    @property
    def uv(self) -> torch.Tensor:
        """Alias convenient for passing coordinates to a point-prompt API."""

        return self.points_uv

    def as_tuple(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(valid_uv, valid_object_ids)``."""

        return self.points_uv, self.object_ids

    @property
    def valid_uv(self) -> torch.Tensor:
        """Explicit alias for the valid projected coordinates."""

        return self.points_uv

    @property
    def valid_object_ids(self) -> torch.Tensor:
        """Explicit alias for IDs whose projections passed all gates."""

        return self.object_ids

    def __getitem__(self, index: int) -> torch.Tensor:
        """Support tuple-style ``result[0]``/``result[1]`` access."""

        return self.as_tuple()[index]

    def __iter__(self):
        # A small compatibility convenience: callers can write
        # ``points, ids = projector(...)`` while retaining the auditable
        # fields above for callers that need depth or validity diagnostics.
        yield self.points_uv
        yield self.object_ids


class TemporalPromptProjector:
    r"""Project global 3-D object centers into the current camera image.

    For a camera-to-world matrix ``T_c2w`` and a world point in homogeneous
    coordinates ``X_w``,

    .. math:: ``X_c = T_{c2w}^{-1} X_w``.

    With camera coordinates ``(X_c,Y_c,Z_c)`` and intrinsic matrix ``K``,

    .. math::

       u = f_x X_c/Z_c + c_x,\qquad
       v = f_y Y_c/Z_c + c_y.

    A point is a valid positive prompt only when ``Z_c > min_depth`` and
    ``0 <= u < W, 0 <= v < H``.  The returned coordinates are ordered
    ``(u,v)`` (x/width first), as expected by common SAM point-prompt APIs.
    """

    def __init__(self, *, min_depth: float = 1e-6) -> None:
        if float(min_depth) <= 0.0:
            raise ValueError("min_depth must be positive.")
        self.min_depth = float(min_depth)

    def __call__(
        self,
        object_3d_centers: torch.Tensor,
        current_pose_c2w: torch.Tensor,
        intrinsics: torch.Tensor,
        image_size: Sequence[int],
        *,
        object_ids: torch.Tensor | Sequence[int] | None = None,
    ) -> TemporalProjection:
        return self.project(
            object_3d_centers,
            current_pose_c2w,
            intrinsics,
            image_size,
            object_ids=object_ids,
        )

    def project(
        self,
        object_3d_centers: torch.Tensor,
        current_pose_c2w: torch.Tensor,
        intrinsics: torch.Tensor,
        image_size: Sequence[int],
        *,
        object_ids: torch.Tensor | Sequence[int] | None = None,
    ) -> TemporalProjection:
        centers = _as_tensor(object_3d_centers, name="object_3d_centers")
        pose = _as_tensor(current_pose_c2w, name="current_pose_c2w")
        matrix = _as_tensor(intrinsics, name="intrinsics")
        if centers.ndim != 2 or tuple(centers.shape[1:]) != (3,):
            raise ValueError(
                "object_3d_centers must have shape [N,3], "
                f"got {tuple(centers.shape)}."
            )
        if tuple(pose.shape) != (4, 4):
            raise ValueError(
                f"current_pose_c2w must have shape [4,4], got {tuple(pose.shape)}."
            )
        if tuple(matrix.shape) != (3, 3):
            raise ValueError(
                f"intrinsics must have shape [3,3], got {tuple(matrix.shape)}."
            )
        height, width = _image_size(image_size)
        if not centers.is_floating_point():
            centers = centers.float()
        # Use at least fp32 for the inverse/projection while preserving a
        # differentiable path from floating-point center inputs.
        dtype = torch.float64 if centers.dtype == torch.float64 else torch.float32
        centers = centers.to(dtype=dtype)
        pose = pose.to(device=centers.device, dtype=dtype)
        matrix = matrix.to(device=centers.device, dtype=dtype)
        if object_ids is None:
            ids = torch.arange(
                centers.shape[0],
                device=centers.device,
                dtype=torch.long,
            )
        else:
            if torch.is_tensor(object_ids):
                ids = object_ids.to(
                    device=centers.device,
                    dtype=torch.long,
                ).reshape(-1)
            else:
                ids = torch.as_tensor(
                    object_ids,
                    device=centers.device,
                    dtype=torch.long,
                ).reshape(-1)
            if ids.shape[0] != centers.shape[0]:
                raise ValueError("object_ids must contain one ID per center.")

        # Column-vector convention: X_c = [R|t] X_w, with the inverse of
        # the supplied camera-to-world homogeneous transform.
        world_to_camera = torch.linalg.inv(pose)
        homogeneous = torch.cat(
            (centers, torch.ones_like(centers[:, :1])),
            dim=1,
        )
        camera_h = (world_to_camera @ homogeneous.T).T
        camera_points = camera_h[:, :3]
        depth = camera_points[:, 2]

        projected_h = (matrix @ camera_points.T).T
        denominator = projected_h[:, 2:3]
        safe_denominator = torch.where(
            denominator.abs() > self.min_depth,
            denominator,
            torch.full_like(denominator, self.min_depth),
        )
        all_uv = projected_h[:, :2] / safe_denominator
        valid = (
            torch.isfinite(centers).all(dim=1)
            & torch.isfinite(camera_points).all(dim=1)
            & torch.isfinite(all_uv).all(dim=1)
            & (depth > self.min_depth)
            & (all_uv[:, 0] >= 0.0)
            & (all_uv[:, 0] < float(width))
            & (all_uv[:, 1] >= 0.0)
            & (all_uv[:, 1] < float(height))
        )
        return TemporalProjection(
            points_uv=all_uv[valid],
            object_ids=ids[valid],
            depth=depth[valid],
            camera_points=camera_points[valid],
            valid_mask=valid,
            all_points_uv=all_uv,
            all_depth=depth,
        )


def _as_tensor(value: object, *, name: str) -> torch.Tensor:
    """Convert array-like inputs without detaching an existing tensor."""

    if torch.is_tensor(value):
        return value
    try:
        return torch.as_tensor(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be tensor- or array-like.") from exc


def _image_size(value: Iterable[int]) -> tuple[int, int]:
    values = tuple(int(item) for item in value)
    if len(values) != 2 or values[0] <= 0 or values[1] <= 0:
        raise ValueError(f"image_size must be positive (H,W), got {value!r}.")
    return values[0], values[1]
