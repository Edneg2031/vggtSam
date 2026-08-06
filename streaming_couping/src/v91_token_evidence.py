"""Discrete-token oracle and matcher diagnostics for V9.1.

The helpers in this module never predict camera pose.  They operate only on
the fixed current/history local-token support and return correspondence rows
that the frozen V9 O-R1 epipolar solver may consume.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F

from streaming_couping.src.v90_epipolar_geometry import LocalTokenReprojection
from streaming_couping.src.v90_explicit_matcher import (
    SoftMatchTarget,
    canonicalize_descriptor_channels,
)


PREDICTION_MODES = ("top1", "soft_expectation")


@dataclass
class TokenPrediction:
    """A decoded correspondence distribution on the fixed token support."""

    predicted_uv: torch.Tensor
    accepted: torch.Tensor
    weights: torch.Tensor
    real_key_mass: torch.Tensor
    concentration: torch.Tensor


@dataclass
class TokenAuditMetrics:
    """Additive counts/sums so pair metrics can be aggregated exactly."""

    supervised_queries: int
    visible_queries: int
    visible_supported_queries: int
    accepted_correspondences: int
    visible_pck_correct: int
    visible_epe_sum_pixels: float
    supported_pck_correct: int
    supported_epe_sum_pixels: float
    dustbin_queries: int
    dustbin_correct: int
    visible_ce_sum: float
    visible_ce_queries: int
    dustbin_ce_sum: float
    entropy_sum: float
    real_key_mass_sum: float
    max_probability_sum: float

    def as_match_stats(self) -> dict[str, float]:
        """Return the fields expected by the frozen V9 pose reporter."""

        return {
            "supervised_queries": float(self.supervised_queries),
            "visible_queries": float(self.visible_queries),
            "visible_key_supported_queries": float(
                self.visible_supported_queries
            ),
            "accepted_correspondences": float(self.accepted_correspondences),
            "dustbin_correct": float(self.dustbin_correct),
            "dustbin_queries": float(self.dustbin_queries),
            "pck_correct": float(self.visible_pck_correct),
            "epe_sum_pixels": float(self.visible_epe_sum_pixels),
            "metrics_precomputed": 1.0,
        }


def normalized_uv_to_pixels(
    uv_normalized: torch.Tensor, image_size: tuple[int, int]
) -> torch.Tensor:
    """Convert ``[-1,1]`` UV to pixel coordinates without changing rows."""

    if uv_normalized.ndim != 2 or uv_normalized.shape[-1] != 2:
        raise ValueError("V9.1 normalized UV must have shape [N,2].")
    height, width = (int(value) for value in image_size)
    uv = uv_normalized.double()
    return torch.stack(
        [
            (uv[:, 0] + 1.0) * 0.5 * max(width - 1, 1),
            (uv[:, 1] + 1.0) * 0.5 * max(height - 1, 1),
        ],
        dim=-1,
    )


def hard_discrete_oracle_probability(target: SoftMatchTarget) -> torch.Tensor:
    """Choose the nearest actual history key, otherwise choose dustbin.

    ``build_soft_match_target`` already applies GT visibility and the locked
    support radius.  Taking its argmax converts that continuous GT projection
    into a hard oracle over the *actual* cached history keys.
    """

    if target.probability.ndim != 2:
        raise ValueError("V9.1 discrete oracle expects an unbatched target.")
    probability = torch.zeros_like(target.probability)
    classes = target.probability.argmax(dim=-1)
    probability.scatter_(1, classes[:, None], 1.0)
    invalid = ~target.supervised.to(
        device=probability.device, dtype=torch.bool
    )
    if bool(invalid.any()):
        probability[invalid] = 0.0
        probability[invalid, -1] = 1.0
    return probability


def raw_cosine_probability(
    query_features: torch.Tensor,
    key_features: torch.Tensor,
    query_valid: torch.Tensor,
    key_valid: torch.Tensor,
    *,
    canonical_dim: int,
    temperature: float,
    dustbin_logit: float = 0.0,
) -> torch.Tensor:
    """Parameter-free descriptor cosine with one fixed dustbin logit."""

    if query_features.ndim != 2 or key_features.ndim != 2:
        raise ValueError("V9.1 raw descriptors must be [Q,C] and [R,C].")
    if query_valid.shape != query_features.shape[:1]:
        raise ValueError("V9.1 raw query validity shape is wrong.")
    if key_valid.shape != key_features.shape[:1]:
        raise ValueError("V9.1 raw key validity shape is wrong.")
    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError("V9.1 raw cosine temperature must be positive.")
    query = canonicalize_descriptor_channels(
        query_features.float(), int(canonical_dim)
    )
    key = canonicalize_descriptor_channels(key_features.float(), int(canonical_dim))
    query = F.normalize(query, dim=-1, eps=1e-6)
    key = F.normalize(key, dim=-1, eps=1e-6)
    logits = (query @ key.transpose(0, 1)) / float(temperature)
    query_valid_device = query_valid.to(device=logits.device, dtype=torch.bool)
    key_valid_device = key_valid.to(device=logits.device, dtype=torch.bool)
    pair_valid = query_valid_device[:, None] & key_valid_device[None, :]
    logits = logits.masked_fill(~pair_valid, -1e4)
    dustbin = torch.full(
        (query.shape[0], 1),
        float(dustbin_logit),
        dtype=logits.dtype,
        device=logits.device,
    )
    logits = torch.cat([logits, dustbin], dim=-1)
    invalid = ~query_valid_device
    if bool(invalid.any()):
        invalid_logits = torch.full_like(logits, -1e4)
        invalid_logits[..., -1] = 0.0
        logits = torch.where(invalid[:, None], invalid_logits, logits)
    return torch.softmax(logits, dim=-1)


def decode_token_probability(
    probability: torch.Tensor,
    *,
    history_uv_normalized: torch.Tensor,
    query_valid: torch.Tensor,
    key_valid: torch.Tensor,
    image_size: tuple[int, int],
    mode: str,
) -> TokenPrediction:
    """Decode a token distribution as Top-1 or a soft UV expectation."""

    if mode not in PREDICTION_MODES:
        raise ValueError(f"Unknown V9.1 prediction mode={mode!r}.")
    if probability.ndim != 2:
        raise ValueError("V9.1 probability must be [Q,R+1].")
    if probability.shape != (query_valid.shape[0], key_valid.shape[0] + 1):
        raise ValueError("V9.1 probability/support shapes disagree.")
    device = probability.device
    query_valid_device = query_valid.to(device=device, dtype=torch.bool)
    key_valid_device = key_valid.to(device=device, dtype=torch.bool)
    key_uv = normalized_uv_to_pixels(history_uv_normalized, image_size).to(device)
    real = probability[:, :-1].double() * key_valid_device.double()[None, :]
    real_mass = real.sum(dim=-1)
    normalized = real / real_mass[:, None].clamp_min(1e-12)
    concentration = normalized.max(dim=-1).values
    if mode == "soft_expectation":
        predicted_uv = normalized @ key_uv
        weights = real_mass * concentration
    else:
        key_index = real.argmax(dim=-1)
        predicted_uv = key_uv.index_select(0, key_index)
        weights = real.gather(1, key_index[:, None])[:, 0]
    predicted_class = probability.argmax(dim=-1)
    accepted = (
        query_valid_device
        & key_valid_device.any()
        & predicted_class.ne(probability.shape[-1] - 1)
        & real_mass.gt(1e-12)
        & torch.isfinite(predicted_uv).all(dim=-1)
    )
    predicted_uv = torch.where(
        accepted[:, None], predicted_uv, torch.full_like(predicted_uv, torch.nan)
    )
    weights = torch.where(accepted, weights, torch.zeros_like(weights))
    return TokenPrediction(
        predicted_uv=predicted_uv,
        accepted=accepted,
        weights=weights,
        real_key_mass=real_mass,
        concentration=concentration,
    )


def audit_token_probability(
    probability: torch.Tensor,
    *,
    labels: LocalTokenReprojection,
    target: SoftMatchTarget,
    prediction: TokenPrediction,
    pck_threshold_pixels: float,
    image_size: tuple[int, int],
) -> TokenAuditMetrics:
    """Measure matching without allowing pose metrics to hide failed rows."""

    if probability.shape != target.probability.shape:
        raise ValueError("V9.1 prediction/target shapes disagree.")
    if prediction.accepted.shape != target.supervised.shape:
        raise ValueError("V9.1 decoded prediction/query shapes disagree.")
    threshold = float(pck_threshold_pixels)
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("V9.1 PCK threshold must be positive.")
    device = probability.device
    supervised = target.supervised.to(device=device, dtype=torch.bool)
    visible = labels.target_visible.to(device=device, dtype=torch.bool) & supervised
    supported = (
        target.visible_with_key_support.to(device=device, dtype=torch.bool) & visible
    )
    diagonal = math.hypot(*image_size)
    prediction_accepted = prediction.accepted.to(device=device, dtype=torch.bool)
    prediction_uv = prediction.predicted_uv.to(device)
    finite = prediction_accepted & torch.isfinite(prediction_uv).all(dim=-1)
    error = torch.full(
        supervised.shape,
        float(diagonal),
        dtype=torch.float64,
        device=device,
    )
    history_target_uv = labels.history_target_uv.to(device)
    valid_error = finite & torch.isfinite(history_target_uv).all(dim=-1)
    error[valid_error] = torch.linalg.vector_norm(
        prediction_uv[valid_error] - history_target_uv[valid_error],
        dim=-1,
    )
    pck = error.le(threshold) & finite
    per_query_ce = -(
        target.probability.to(probability.device, probability.dtype)
        * probability.clamp_min(1e-8).log()
    ).sum(dim=-1)
    target_class = target.probability.argmax(dim=-1).to(device)
    predicted_class = probability.argmax(dim=-1)
    dustbin_index = probability.shape[-1] - 1
    dustbin = supervised & target_class.eq(dustbin_index)
    class_count = max(int(probability.shape[-1]), 2)
    entropy = -(
        probability.clamp_min(1e-12) * probability.clamp_min(1e-12).log()
    ).sum(dim=-1) / math.log(class_count)
    return TokenAuditMetrics(
        supervised_queries=int(supervised.sum()),
        visible_queries=int(visible.sum()),
        visible_supported_queries=int(supported.sum()),
        accepted_correspondences=int(prediction_accepted.sum()),
        visible_pck_correct=int((pck & visible).sum()),
        visible_epe_sum_pixels=float(error[visible].sum()),
        supported_pck_correct=int((pck & supported).sum()),
        supported_epe_sum_pixels=float(error[supported].sum()),
        dustbin_queries=int(dustbin.sum()),
        dustbin_correct=int((dustbin & predicted_class.eq(dustbin_index)).sum()),
        visible_ce_sum=float(per_query_ce[supported].sum()),
        visible_ce_queries=int(supported.sum()),
        dustbin_ce_sum=float(per_query_ce[dustbin].sum()),
        entropy_sum=float(entropy[supervised].sum()),
        real_key_mass_sum=float(
            prediction.real_key_mass.to(device)[supervised].sum()
        ),
        max_probability_sum=float(
            probability.max(dim=-1).values[supervised].sum()
        ),
    )
