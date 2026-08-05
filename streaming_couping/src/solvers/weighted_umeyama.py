"""Deterministic weighted Sim(3) alignment for V8 geometry diagnosis."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .weighted_kabsch import KabschConfig


@dataclass
class UmeyamaResult:
    """Similarity mapping ``source`` to ``target``.

    Points follow the row-vector convention
    ``target = scale * (source @ rotation.T) + translation``.  ``transform``
    stores only the rigid camera frame ``[rotation, translation]``; the scale
    is returned separately and must never be folded into an SE(3) pose.
    """

    transform: torch.Tensor
    rotation: torch.Tensor
    translation: torch.Tensor
    scale: torch.Tensor
    accepted: torch.Tensor
    degenerate: torch.Tensor
    point_count: torch.Tensor
    retained_count: torch.Tensor
    rmse: torch.Tensor
    median_residual: torch.Tensor
    p90_residual: torch.Tensor
    inlier_ratio: torch.Tensor
    source_eigenvalues: torch.Tensor
    target_eigenvalues: torch.Tensor
    secondary_eigenvalue_ratio: torch.Tensor
    condition_number: torch.Tensor
    reflection_corrected: torch.Tensor


def weighted_umeyama(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
    valid: torch.Tensor | None = None,
    config: KabschConfig | None = None,
) -> UmeyamaResult:
    """Fit one positive-scale similarity transform per leading batch item."""

    config = config or KabschConfig()
    config.validate()
    if source.shape != target.shape or source.ndim < 2 or source.shape[-1] != 3:
        raise ValueError("Umeyama source/target must share shape [...,N,3].")
    leading = source.shape[:-2]
    points = int(source.shape[-2])
    batch_items = math.prod(leading) if leading else 1
    expected = (*leading, points)
    if weights is None:
        weights = torch.ones(expected, dtype=source.dtype, device=source.device)
    if valid is None:
        valid = torch.ones(expected, dtype=torch.bool, device=source.device)
    if weights.shape != expected or valid.shape != expected:
        raise ValueError("Umeyama weights/valid must have shape [...,N].")

    source_flat = source.reshape(batch_items, points, 3).double()
    target_flat = target.reshape(batch_items, points, 3).double()
    weights_flat = weights.reshape(batch_items, points).double()
    valid_flat = valid.reshape(batch_items, points).bool()
    results = [
        _solve_one(
            source_flat[index],
            target_flat[index],
            weights_flat[index],
            valid_flat[index],
            config,
        )
        for index in range(batch_items)
    ]

    def stack(name: str) -> torch.Tensor:
        value = torch.stack([item[name] for item in results])
        return value.reshape((*leading, *value.shape[1:]))

    transform = stack("transform").to(source.dtype)
    return UmeyamaResult(
        transform=transform,
        rotation=transform[..., :3, :3],
        translation=transform[..., :3, 3],
        scale=stack("scale").to(source.dtype),
        accepted=stack("accepted"),
        degenerate=stack("degenerate"),
        point_count=stack("point_count"),
        retained_count=stack("retained_count"),
        rmse=stack("rmse").to(source.dtype),
        median_residual=stack("median_residual").to(source.dtype),
        p90_residual=stack("p90_residual").to(source.dtype),
        inlier_ratio=stack("inlier_ratio").to(source.dtype),
        source_eigenvalues=stack("source_eigenvalues").to(source.dtype),
        target_eigenvalues=stack("target_eigenvalues").to(source.dtype),
        secondary_eigenvalue_ratio=stack("secondary_eigenvalue_ratio").to(source.dtype),
        condition_number=stack("condition_number").to(source.dtype),
        reflection_corrected=stack("reflection_corrected"),
    )


def _solve_one(
    source: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    valid: torch.Tensor,
    config: KabschConfig,
) -> dict[str, torch.Tensor]:
    finite = (
        torch.isfinite(source).all(dim=-1)
        & torch.isfinite(target).all(dim=-1)
        & torch.isfinite(weights)
    )
    base = valid & finite & weights.gt(0.0)
    point_count = base.sum()
    identity = torch.eye(4, dtype=source.dtype, device=source.device)
    if int(point_count) < int(config.min_points):
        return _empty_result(identity, point_count, point_count)
    source = torch.where(finite[:, None], source, torch.zeros_like(source))
    target = torch.where(finite[:, None], target, torch.zeros_like(target))
    weights = torch.where(base, weights, torch.zeros_like(weights))

    keep = base.clone()
    reflection_corrected = False
    for iteration in range(int(config.trim_iterations)):
        scale, rotation, translation, corrected = _fit_once(
            source, target, weights, keep
        )
        reflection_corrected = reflection_corrected or corrected
        residual = torch.linalg.vector_norm(
            scale * (source @ rotation.T) + translation - target,
            dim=-1,
        )
        if config.trim_quantile >= 1.0 or iteration + 1 == config.trim_iterations:
            break
        cutoff = torch.quantile(residual[base], float(config.trim_quantile))
        next_keep = base & residual.le(cutoff)
        if int(next_keep.sum()) < int(config.min_points) or torch.equal(
            next_keep, keep
        ):
            break
        keep = next_keep

    scale, rotation, translation, corrected = _fit_once(source, target, weights, keep)
    reflection_corrected = reflection_corrected or corrected
    residual = torch.linalg.vector_norm(
        scale * (source @ rotation.T) + translation - target,
        dim=-1,
    )
    retained = residual[keep]
    source_eigenvalues = _weighted_eigenvalues(source, weights, keep)
    target_eigenvalues = _weighted_eigenvalues(target, weights, keep)
    source_ratio = source_eigenvalues[-2] / source_eigenvalues[-1].clamp_min(1e-12)
    target_ratio = target_eigenvalues[-2] / target_eigenvalues[-1].clamp_min(1e-12)
    secondary_ratio = torch.minimum(source_ratio, target_ratio)
    condition = 1.0 / secondary_ratio.clamp_min(1e-12)
    degenerate = (
        ~torch.isfinite(secondary_ratio)
        | secondary_ratio.lt(float(config.min_secondary_eigenvalue_ratio))
        | ~torch.isfinite(scale)
        | scale.le(0.0)
    )
    rmse = torch.sqrt(retained.square().mean())
    accepted = (
        ~degenerate
        & torch.isfinite(rotation).all()
        & torch.isfinite(translation).all()
        & torch.isfinite(rmse)
    )
    transform = identity.clone()
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return {
        "transform": transform,
        "scale": scale,
        "accepted": accepted,
        "degenerate": degenerate,
        "point_count": point_count,
        "retained_count": keep.sum(),
        "rmse": rmse,
        "median_residual": torch.quantile(retained, 0.50),
        "p90_residual": torch.quantile(retained, 0.90),
        "inlier_ratio": residual[base].le(float(config.inlier_distance)).float().mean(),
        "source_eigenvalues": source_eigenvalues,
        "target_eigenvalues": target_eigenvalues,
        "secondary_eigenvalue_ratio": secondary_ratio,
        "condition_number": condition,
        "reflection_corrected": torch.tensor(
            reflection_corrected, dtype=torch.bool, device=source.device
        ),
    }


def _fit_once(
    source: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    keep: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    selected = torch.where(keep, weights, torch.zeros_like(weights))
    total = selected.sum().clamp_min(1e-12)
    source_mean = (source * selected[:, None]).sum(dim=0) / total
    target_mean = (target * selected[:, None]).sum(dim=0) / total
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = source_centered.T @ (selected[:, None] * target_centered) / total
    left, _, right_t = torch.linalg.svd(covariance)
    right = right_t.T
    rotation = right @ left.T
    corrected = bool(torch.det(rotation) < 0.0)
    if corrected:
        right = right.clone()
        right[:, -1] *= -1.0
        rotation = right @ left.T
    rotated_source = source_centered @ rotation.T
    numerator = (selected[:, None] * target_centered * rotated_source).sum()
    denominator = (selected[:, None] * source_centered.square()).sum().clamp_min(1e-12)
    scale = numerator / denominator
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation, corrected


def _weighted_eigenvalues(
    points: torch.Tensor,
    weights: torch.Tensor,
    keep: torch.Tensor,
) -> torch.Tensor:
    selected = torch.where(keep, weights, torch.zeros_like(weights))
    total = selected.sum().clamp_min(1e-12)
    mean = (points * selected[:, None]).sum(dim=0) / total
    centered = points - mean
    covariance = centered.T @ (selected[:, None] * centered) / total
    return torch.linalg.eigvalsh(covariance).clamp_min(0.0)


def _empty_result(
    identity: torch.Tensor,
    point_count: torch.Tensor,
    retained_count: torch.Tensor,
) -> dict[str, torch.Tensor]:
    nan = identity.new_tensor(float("nan"))
    return {
        "transform": identity,
        "scale": nan,
        "accepted": torch.tensor(False, device=identity.device),
        "degenerate": torch.tensor(True, device=identity.device),
        "point_count": point_count,
        "retained_count": retained_count,
        "rmse": nan,
        "median_residual": nan,
        "p90_residual": nan,
        "inlier_ratio": identity.new_tensor(0.0),
        "source_eigenvalues": identity.new_zeros(3),
        "target_eigenvalues": identity.new_zeros(3),
        "secondary_eigenvalue_ratio": identity.new_tensor(0.0),
        "condition_number": identity.new_tensor(math.inf),
        "reflection_corrected": torch.tensor(False, device=identity.device),
    }
