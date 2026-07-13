"""Mine coarse revisit candidates from a historical object point map."""

from __future__ import annotations

import torch

from test_sam.coordinates import output_mask_transform, streamvggt_image_transform

from ..types import RevisitCandidate


def mine_revisit_candidate(
    object_points: torch.Tensor,
    *,
    current_world_points: torch.Tensor,
    world_to_camera: torch.Tensor,
    intrinsics: torch.Tensor,
    source_size: tuple[int, int],
    processed_size: tuple[int, int],
    output_size: tuple[int, int],
    image_mode: str,
    box_quantile: float,
    box_padding_ratio: float,
    min_projected_points: int,
    min_projected_fraction: float,
    min_supported_points: int,
    min_support_ratio: float,
    support_abs_distance: float,
    support_relative_distance: float,
) -> RevisitCandidate:
    """Project history points and validate them against the current pointmap.

    The returned mask is a filled candidate box. It is not a predicted object
    mask and must only be used as a prompt for a segmentation model.
    """

    empty = torch.zeros(output_size, dtype=torch.bool)
    total_points = int(object_points.shape[0])
    if total_points == 0:
        return _rejected(empty, "object map is empty")

    rotation = world_to_camera[:3, :3]
    translation = world_to_camera[:3, 3]
    camera_points = object_points @ rotation.T + translation
    depth = camera_points[:, 2]
    finite = torch.isfinite(camera_points).all(dim=-1) & (depth > 1e-5)
    if not finite.any():
        return _rejected(empty, "object map is behind the camera")
    camera_points = camera_points[finite]
    world_points = object_points[finite]
    depth = camera_points[:, 2]
    homogeneous = camera_points @ intrinsics.T
    x = homogeneous[:, 0] / homogeneous[:, 2].clamp_min(1e-6)
    y = homogeneous[:, 1] / homogeneous[:, 2].clamp_min(1e-6)
    grid_h, grid_w = current_world_points.shape[:2]
    x_index = x.round().long()
    y_index = y.round().long()
    inside = (
        (x_index >= 0)
        & (x_index < grid_w)
        & (y_index >= 0)
        & (y_index < grid_h)
    )
    x = x[inside]
    y = y[inside]
    depth = depth[inside]
    world_points = world_points[inside]
    x_index = x_index[inside]
    y_index = y_index[inside]
    projected_points = int(x.shape[0])
    projected_fraction = projected_points / max(total_points, 1)
    if projected_points == 0:
        return _rejected(empty, "no object-map point projects into the frame")

    observed = current_world_points[y_index, x_index]
    observed_valid = torch.isfinite(observed).all(dim=-1)
    distance = torch.linalg.vector_norm(observed - world_points, dim=-1)
    tolerance = float(support_abs_distance) + float(support_relative_distance) * depth
    supported = observed_valid & (distance <= tolerance)
    supported_points = int(supported.sum())
    support_ratio = supported_points / max(projected_points, 1)

    # A supported subset gives a tighter box. If support is weak, keep the raw
    # projection for diagnostics but reject it below instead of returning it as
    # a final segmentation mask.
    box_x = x[supported] if supported_points >= min_supported_points else x
    box_y = y[supported] if supported_points >= min_supported_points else y
    output_x, output_y = _processed_to_output(
        box_x,
        box_y,
        source_size=source_size,
        processed_size=processed_size,
        output_size=output_size,
        image_mode=image_mode,
    )
    projected_x, projected_y = _processed_to_output(
        x,
        y,
        source_size=source_size,
        processed_size=processed_size,
        output_size=output_size,
        image_mode=image_mode,
    )
    supported_x, supported_y = _processed_to_output(
        x[supported],
        y[supported],
        source_size=source_size,
        processed_size=processed_size,
        output_size=output_size,
        image_mode=image_mode,
    )
    projected_mask = _point_mask(projected_x, projected_y, output_size)
    supported_mask = _point_mask(supported_x, supported_y, output_size)
    box = _robust_box(
        output_x,
        output_y,
        output_size=output_size,
        quantile=box_quantile,
        padding_ratio=box_padding_ratio,
    )
    if box is None:
        return _rejected(empty, "projected candidate box is empty")
    x0, y0, x1, y1 = box
    candidate_mask = empty.clone()
    candidate_mask[y0:y1, x0:x1] = True

    checks = [
        (projected_points >= min_projected_points, "too few projected points"),
        (
            projected_fraction >= min_projected_fraction,
            "projected object fraction below threshold",
        ),
        (supported_points >= min_supported_points, "too few pointmap-supported points"),
        (support_ratio >= min_support_ratio, "pointmap support ratio below threshold"),
    ]
    failed = [message for passed, message in checks if not passed]
    accepted = not failed
    return RevisitCandidate(
        mask=candidate_mask,
        projected_mask=projected_mask,
        supported_mask=supported_mask,
        box_xyxy=box,
        projected_points=projected_points,
        supported_points=supported_points,
        projected_fraction=float(projected_fraction),
        support_ratio=float(support_ratio),
        accepted=accepted,
        reason="accepted geometry candidate" if accepted else "; ".join(failed),
    )


def _processed_to_output(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    source_size: tuple[int, int],
    processed_size: tuple[int, int],
    output_size: tuple[int, int],
    image_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    stream = streamvggt_image_transform(source_size, mode=image_mode)
    if tuple(stream.target_size) != tuple(processed_size):
        raise ValueError(
            f"StreamVGGT transform size {stream.target_size} does not match "
            f"pointmap size {processed_size}."
        )
    output = output_mask_transform(source_size, output_size)
    sx, sy = stream.scale_xy
    ox, oy = stream.offset_xy
    source_x = (x - ox + 0.5) / sx - 0.5
    source_y = (y - oy + 0.5) / sy - 0.5
    output_x = (
        (source_x + 0.5) * output.scale_xy[0] - 0.5 + output.offset_xy[0]
    )
    output_y = (
        (source_y + 0.5) * output.scale_xy[1] - 0.5 + output.offset_xy[1]
    )
    return output_x, output_y


def _robust_box(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    output_size: tuple[int, int],
    quantile: float,
    padding_ratio: float,
) -> tuple[int, int, int, int] | None:
    finite = torch.isfinite(x) & torch.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.numel() == 0:
        return None
    q = min(max(float(quantile), 0.0), 0.49)
    x0 = float(torch.quantile(x, q))
    x1 = float(torch.quantile(x, 1.0 - q))
    y0 = float(torch.quantile(y, q))
    y1 = float(torch.quantile(y, 1.0 - q))
    pad_x = max((x1 - x0) * float(padding_ratio), 2.0)
    pad_y = max((y1 - y0) * float(padding_ratio), 2.0)
    height, width = output_size
    ix0 = max(0, min(width - 1, int(x0 - pad_x)))
    iy0 = max(0, min(height - 1, int(y0 - pad_y)))
    ix1 = max(ix0 + 1, min(width, int(x1 + pad_x + 1)))
    iy1 = max(iy0 + 1, min(height, int(y1 + pad_y + 1)))
    return ix0, iy0, ix1, iy1


def _rejected(mask: torch.Tensor, reason: str) -> RevisitCandidate:
    return RevisitCandidate(
        mask=mask,
        projected_mask=mask.clone(),
        supported_mask=mask.clone(),
        box_xyxy=None,
        projected_points=0,
        supported_points=0,
        projected_fraction=0.0,
        support_ratio=0.0,
        accepted=False,
        reason=reason,
    )


def _point_mask(
    x: torch.Tensor,
    y: torch.Tensor,
    output_size: tuple[int, int],
) -> torch.Tensor:
    height, width = output_size
    mask = torch.zeros(output_size, dtype=torch.bool)
    if x.numel() == 0:
        return mask
    x_index = x.round().long()
    y_index = y.round().long()
    inside = (
        torch.isfinite(x)
        & torch.isfinite(y)
        & (x_index >= 0)
        & (x_index < width)
        & (y_index >= 0)
        & (y_index < height)
    )
    mask[y_index[inside], x_index[inside]] = True
    return torch.nn.functional.max_pool2d(
        mask.float()[None, None],
        kernel_size=3,
        stride=1,
        padding=1,
    )[0, 0].bool()
