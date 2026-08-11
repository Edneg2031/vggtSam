"""Deterministic robust essential-matrix solvers for the V9.4 diagnosis."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from streaming_couping.src.v90_epipolar_geometry import (
    EpipolarConfig,
    EpipolarEstimate,
    _failed_epipolar,
    _signed_sampson_residual,
    _weighted_eight_point,
    estimate_relative_epipolar_pose,
    pixels_to_calibrated_points,
)


ROBUST_SOLVERS = (
    "o_r1",
    "deterministic_ransac_minimal",
    "deterministic_ransac_inlier_refine",
    "spatial_ransac_inlier_refine",
)


@dataclass(frozen=True)
class RobustEpipolarConfig:
    iterations: int = 128
    threshold_pixels: float = 1.5
    sample_size: int = 8
    spatial_candidate_pool: int = 32
    seed: int = 94

    def validate(self) -> None:
        if self.iterations < 1:
            raise ValueError("V9.4 RANSAC iterations must be positive.")
        if self.sample_size != 8:
            raise ValueError("V9.4 locks minimal samples to eight points.")
        if not math.isfinite(self.threshold_pixels) or self.threshold_pixels <= 0.0:
            raise ValueError("V9.4 RANSAC threshold must be positive.")
        if self.spatial_candidate_pool < self.sample_size:
            raise ValueError("V9.4 spatial pool is smaller than the sample.")


@dataclass(frozen=True)
class ConsensusDiagnostics:
    hypotheses_tested: int
    inliers: int
    inlier_fraction: float
    msac_objective: float
    threshold_calibrated: float


@dataclass(frozen=True)
class RobustEstimate:
    estimate: EpipolarEstimate
    consensus: ConsensusDiagnostics


def estimate_robust_relative_pose(
    current_uv: torch.Tensor,
    history_uv: torch.Tensor,
    weights: torch.Tensor,
    current_intrinsics: torch.Tensor,
    history_intrinsics: torch.Tensor,
    l0_current_to_history: torch.Tensor,
    *,
    solver: str,
    epipolar_config: EpipolarConfig,
    robust_config: RobustEpipolarConfig,
    image_size: tuple[int, int],
    seed_offset: int = 0,
) -> RobustEstimate:
    """Estimate one relative pose with a fixed deterministic solver policy."""

    if solver not in ROBUST_SOLVERS:
        raise ValueError(f"Unknown V9.4 solver={solver!r}.")
    robust_config.validate()
    count = int(current_uv.shape[0])
    if solver == "o_r1":
        estimate = estimate_relative_epipolar_pose(
            current_uv,
            history_uv,
            weights,
            current_intrinsics,
            history_intrinsics,
            l0_current_to_history,
            config=epipolar_config,
        )
        return RobustEstimate(
            estimate=estimate,
            consensus=ConsensusDiagnostics(
                hypotheses_tested=0,
                inliers=count if estimate.success else 0,
                inlier_fraction=1.0 if estimate.success and count else 0.0,
                msac_objective=float("nan"),
                threshold_calibrated=float("nan"),
            ),
        )

    _validate_inputs(current_uv, history_uv, weights)
    finite = (
        torch.isfinite(current_uv).all(dim=-1)
        & torch.isfinite(history_uv).all(dim=-1)
        & torch.isfinite(weights)
        & weights.gt(0.0)
    )
    current_uv = current_uv[finite].double()
    history_uv = history_uv[finite].double()
    weights = weights[finite].double()
    count = int(current_uv.shape[0])
    failure = _failed_epipolar(count)
    if count < robust_config.sample_size:
        failure.reason = "fewer_than_ransac_sample"
        return RobustEstimate(
            failure,
            ConsensusDiagnostics(0, 0, 0.0, float("inf"), float("nan")),
        )
    current = pixels_to_calibrated_points(current_uv, current_intrinsics)
    history = pixels_to_calibrated_points(history_uv, history_intrinsics)
    focal = _mean_focal(current_intrinsics, history_intrinsics)
    threshold = float(robust_config.threshold_pixels) / max(focal, 1e-12)
    spatial = solver == "spatial_ransac_inlier_refine"
    generator = torch.Generator(device="cpu").manual_seed(
        int(robust_config.seed) + int(seed_offset)
    )
    best: tuple[float, float, tuple[int, ...], torch.Tensor, torch.Tensor] | None = None
    hypotheses = 0
    for _ in range(int(robust_config.iterations)):
        sample = _sample_indices(
            current_uv,
            history_uv,
            generator=generator,
            sample_size=robust_config.sample_size,
            spatial=spatial,
            candidate_pool=robust_config.spatial_candidate_pool,
            image_size=image_size,
        )
        try:
            essential, _, _ = _weighted_eight_point(
                current.index_select(0, sample),
                history.index_select(0, sample),
                weights.index_select(0, sample),
            )
        except RuntimeError:
            continue
        residual = _signed_sampson_residual(essential, current, history).abs()
        if not torch.isfinite(residual).all():
            continue
        hypotheses += 1
        normalized = residual / max(threshold, 1e-12)
        objective = float(
            (weights * normalized.square().clamp_max(1.0)).sum()
            / weights.sum().clamp_min(1e-12)
        )
        inlier = residual.le(threshold)
        inlier_weight = float(weights[inlier].sum() / weights.sum().clamp_min(1e-12))
        key = (objective, -inlier_weight, tuple(int(v) for v in sample.tolist()))
        if best is None or key[:3] < best[:3]:
            best = (objective, -inlier_weight, key[2], essential, inlier)
    if best is None:
        failure.reason = "no_valid_ransac_hypothesis"
        return RobustEstimate(
            failure,
            ConsensusDiagnostics(hypotheses, 0, 0.0, float("inf"), threshold),
        )

    objective, negative_fraction, sample_tuple, _, inlier = best
    if solver == "deterministic_ransac_minimal":
        fit = torch.tensor(sample_tuple, dtype=torch.long)
    else:
        fit = torch.nonzero(inlier, as_tuple=False).flatten()
    if int(fit.numel()) < robust_config.sample_size:
        failure.reason = "fewer_than_min_consensus_inliers"
        return RobustEstimate(
            failure,
            ConsensusDiagnostics(
                hypotheses,
                int(inlier.sum()),
                float(-negative_fraction),
                objective,
                threshold,
            ),
        )
    estimate = estimate_relative_epipolar_pose(
        current_uv.index_select(0, fit),
        history_uv.index_select(0, fit),
        weights.index_select(0, fit),
        current_intrinsics,
        history_intrinsics,
        l0_current_to_history,
        config=epipolar_config,
    )
    estimate.correspondences = count
    estimate.inlier_ratio = float(-negative_fraction)
    estimate.initialization = f"{solver}:{estimate.initialization}"
    if not estimate.success:
        estimate.reason = f"ransac_refit_{estimate.reason}"
    return RobustEstimate(
        estimate=estimate,
        consensus=ConsensusDiagnostics(
            hypotheses_tested=hypotheses,
            inliers=int(inlier.sum()),
            inlier_fraction=float(-negative_fraction),
            msac_objective=float(objective),
            threshold_calibrated=threshold,
        ),
    )


def _sample_indices(
    current_uv: torch.Tensor,
    history_uv: torch.Tensor,
    *,
    generator: torch.Generator,
    sample_size: int,
    spatial: bool,
    candidate_pool: int,
    image_size: tuple[int, int],
) -> torch.Tensor:
    count = int(current_uv.shape[0])
    if not spatial:
        return torch.randperm(count, generator=generator)[:sample_size]
    pool_count = min(int(candidate_pool), count)
    pool = torch.randperm(count, generator=generator)[:pool_count]
    height, width = (int(value) for value in image_size)
    scale = torch.tensor(
        [max(width - 1, 1), max(height - 1, 1)], dtype=torch.float64
    )
    features = torch.cat(
        [current_uv.index_select(0, pool) / scale, history_uv.index_select(0, pool) / scale],
        dim=-1,
    )
    selected = torch.empty(sample_size, dtype=torch.long)
    selected[0] = 0
    minimum = torch.linalg.vector_norm(features - features[0], dim=-1)
    minimum[0] = -1.0
    for index in range(1, sample_size):
        choice = minimum.argmax()
        selected[index] = choice
        distance = torch.linalg.vector_norm(features - features[choice], dim=-1)
        minimum = torch.minimum(minimum, distance)
        minimum[choice] = -1.0
    return pool.index_select(0, selected)


def _mean_focal(current: torch.Tensor, history: torch.Tensor) -> float:
    values = torch.stack(
        [current.double()[0, 0], current.double()[1, 1], history.double()[0, 0], history.double()[1, 1]]
    ).abs()
    finite = values[torch.isfinite(values) & values.gt(1e-12)]
    if not int(finite.numel()):
        raise ValueError("V9.4 intrinsics contain no valid focal length.")
    return float(finite.mean())


def _validate_inputs(
    current_uv: torch.Tensor, history_uv: torch.Tensor, weights: torch.Tensor
) -> None:
    count = int(current_uv.shape[0])
    if (
        current_uv.ndim != 2
        or current_uv.shape[-1] != 2
        or history_uv.shape != current_uv.shape
        or weights.shape != (count,)
    ):
        raise ValueError("V9.4 robust inputs must be [N,2], [N,2], [N].")
