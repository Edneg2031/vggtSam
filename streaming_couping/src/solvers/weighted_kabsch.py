"""Batched weighted Kabsch with deterministic trimming and degeneracy checks."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class KabschConfig:
    min_points: int = 6
    trim_quantile: float = 1.0
    trim_iterations: int = 1
    min_secondary_eigenvalue_ratio: float = 1e-4
    inlier_distance: float = 0.10

    def validate(self) -> None:
        if self.min_points < 3:
            raise ValueError("Kabsch min_points must be at least three.")
        if not 0.0 < self.trim_quantile <= 1.0:
            raise ValueError("Kabsch trim_quantile must be in (0,1].")
        if self.trim_iterations < 1:
            raise ValueError("Kabsch trim_iterations must be positive.")
        if not 0.0 <= self.min_secondary_eigenvalue_ratio < 1.0:
            raise ValueError(
                "Kabsch min_secondary_eigenvalue_ratio must be in [0,1)."
            )
        if self.inlier_distance <= 0.0:
            raise ValueError("Kabsch inlier_distance must be positive.")


@dataclass
class KabschResult:
    """Transform maps row-vector ``source`` to ``target`` as ``x @ R.T + t``."""

    transform: torch.Tensor
    rotation: torch.Tensor
    translation: torch.Tensor
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


def weighted_kabsch(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
    valid: torch.Tensor | None = None,
    config: KabschConfig = KabschConfig(),
) -> KabschResult:
    """Fit one rigid transform per leading batch item.

    ``source`` and ``target`` must have shape ``[...,N,3]``. Invalid, non-finite
    and non-positive-weight pairs are ignored. A planar set is allowed; a
    collinear or point-like set is rejected using the second covariance
    eigenvalue rather than the smallest eigenvalue.
    """

    config.validate()
    if source.shape != target.shape or source.ndim < 2 or source.shape[-1] != 3:
        raise ValueError("Kabsch source/target must share shape [...,N,3].")
    leading = source.shape[:-2]
    points = int(source.shape[-2])
    batch_items = math.prod(leading) if leading else 1
    expected = (*leading, points)
    if weights is None:
        weights = torch.ones(expected, dtype=source.dtype, device=source.device)
    if valid is None:
        valid = torch.ones(expected, dtype=torch.bool, device=source.device)
    if weights.shape != expected or valid.shape != expected:
        raise ValueError("Kabsch weights/valid must have shape [...,N].")

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
        for index in range(source_flat.shape[0])
    ]

    def stack(name: str) -> torch.Tensor:
        value = torch.stack([item[name] for item in results])
        return value.reshape((*leading, *value.shape[1:]))

    transform = stack("transform").to(source.dtype)
    return KabschResult(
        transform=transform,
        rotation=transform[..., :3, :3],
        translation=transform[..., :3, 3],
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
        secondary_eigenvalue_ratio=stack("secondary_eigenvalue_ratio").to(
            source.dtype
        ),
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
        return _empty_result(
            identity,
            point_count=point_count,
            retained_count=point_count,
            degenerate=True,
        )
    source = torch.where(finite[:, None], source, torch.zeros_like(source))
    target = torch.where(finite[:, None], target, torch.zeros_like(target))
    weights = torch.where(base, weights, torch.zeros_like(weights))

    keep = base.clone()
    reflection_corrected = False
    for iteration in range(int(config.trim_iterations)):
        fitted = _fit_once(source, target, weights, keep)
        reflection_corrected = reflection_corrected or fitted[2]
        rotation, translation = fitted[:2]
        residual = torch.linalg.vector_norm(
            source @ rotation.T + translation - target,
            dim=-1,
        )
        if config.trim_quantile >= 1.0 or iteration + 1 == config.trim_iterations:
            break
        candidate = residual[base]
        cutoff = torch.quantile(candidate, float(config.trim_quantile))
        next_keep = base & residual.le(cutoff)
        if int(next_keep.sum()) < int(config.min_points) or torch.equal(
            next_keep, keep
        ):
            break
        keep = next_keep

    rotation, translation, corrected = _fit_once(source, target, weights, keep)
    reflection_corrected = reflection_corrected or corrected
    residual = torch.linalg.vector_norm(
        source @ rotation.T + translation - target,
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
    )
    rmse = torch.sqrt(retained.square().mean())
    median = torch.quantile(retained, 0.50)
    p90 = torch.quantile(retained, 0.90)
    inlier_ratio = residual[base].le(float(config.inlier_distance)).float().mean()
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
        "accepted": accepted,
        "degenerate": degenerate,
        "point_count": point_count,
        "retained_count": keep.sum(),
        "rmse": rmse,
        "median_residual": median,
        "p90_residual": p90,
        "inlier_ratio": inlier_ratio,
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
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    selected_weight = torch.where(keep, weights, torch.zeros_like(weights))
    total = selected_weight.sum().clamp_min(1e-12)
    source_mean = (source * selected_weight[:, None]).sum(dim=0) / total
    target_mean = (target * selected_weight[:, None]).sum(dim=0) / total
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = (
        source_centered.T
        @ (selected_weight[:, None] * target_centered)
        / total
    )
    left, _, right_t = torch.linalg.svd(covariance)
    right = right_t.T
    rotation = right @ left.T
    corrected = bool(torch.det(rotation) < 0.0)
    if corrected:
        right = right.clone()
        right[:, -1] *= -1.0
        rotation = right @ left.T
    translation = target_mean - rotation @ source_mean
    return rotation, translation, corrected


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
    *,
    point_count: torch.Tensor,
    retained_count: torch.Tensor,
    degenerate: bool,
) -> dict[str, torch.Tensor]:
    scalar_nan = identity.new_tensor(float("nan"))
    scalar_zero = identity.new_tensor(0.0)
    return {
        "transform": identity,
        "accepted": torch.tensor(False, device=identity.device),
        "degenerate": torch.tensor(degenerate, device=identity.device),
        "point_count": point_count,
        "retained_count": retained_count,
        "rmse": scalar_nan,
        "median_residual": scalar_nan,
        "p90_residual": scalar_nan,
        "inlier_ratio": scalar_zero,
        "source_eigenvalues": identity.new_zeros(3),
        "target_eigenvalues": identity.new_zeros(3),
        "secondary_eigenvalue_ratio": scalar_zero,
        "condition_number": identity.new_tensor(math.inf),
        "reflection_corrected": torch.tensor(False, device=identity.device),
    }
