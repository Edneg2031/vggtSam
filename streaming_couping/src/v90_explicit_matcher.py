"""Explicit local-token correspondence models for V9 Stage A.

The module predicts only a row-stochastic 2D correspondence distribution.
It has no camera tokens, pose head, depth, pointmap, frame-index embedding or
GT-pose input.  A fixed epipolar solver consumes the frozen matcher's output
after training.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from streaming_couping.src.v90_epipolar_geometry import LocalTokenReprojection


MATCHER_ARCHITECTURES = (
    "sam_match",
    "stream_patch_match",
    "mask_uv_uniform",
    "sam_train_off",
)

MATCHER_EVAL_VARIANTS = (
    "normal",
    "sam_off",
    "wrong_identity",
    "shuffle_time",
    "channel_permute",
)


@dataclass(frozen=True)
class MatcherConfig:
    canonical_dim: int = 256
    projection_dim: int = 64
    temperature: float = 0.07
    target_sigma_pixels: float = 6.0
    target_radius_pixels: float = 12.0
    pck_threshold_pixels: float = 8.0
    cycle_weight: float = 0.10

    def validate(self) -> None:
        if self.canonical_dim < 2 or self.projection_dim < 2:
            raise ValueError("V9 matcher dimensions must be at least two.")
        positive = (
            self.temperature,
            self.target_sigma_pixels,
            self.target_radius_pixels,
            self.pck_threshold_pixels,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("V9 matcher positive settings must be finite.")
        if not math.isfinite(self.cycle_weight) or self.cycle_weight < 0.0:
            raise ValueError("V9 cycle weight must be finite and non-negative.")


@dataclass
class SoftMatchTarget:
    probability: torch.Tensor
    supervised: torch.Tensor
    visible_with_key_support: torch.Tensor
    target_uv: torch.Tensor
    nearest_key_distance_pixels: torch.Tensor


@dataclass
class MatchLoss:
    loss: torch.Tensor
    cross_entropy: torch.Tensor
    cycle: torch.Tensor
    supervised_queries: int


class ExplicitLocalMatcher(nn.Module):
    """Parameter-matched Q/K projection shared by SAM and patch controls."""

    def __init__(self, config: MatcherConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.query_projection = nn.Linear(
            int(config.canonical_dim), int(config.projection_dim), bias=False
        )
        self.key_projection = nn.Linear(
            int(config.canonical_dim), int(config.projection_dim), bias=False
        )
        self.dustbin_logit = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        query_features: torch.Tensor,
        key_features: torch.Tensor,
        query_valid: torch.Tensor,
        key_valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if query_features.ndim != 3 or key_features.ndim != 3:
            raise ValueError("V9 matcher features must be [B,Q,C] and [B,R,C].")
        if query_features.shape[0] != key_features.shape[0]:
            raise ValueError("V9 matcher feature batches disagree.")
        if query_valid.shape != query_features.shape[:2]:
            raise ValueError("V9 matcher query validity shape is wrong.")
        if key_valid.shape != key_features.shape[:2]:
            raise ValueError("V9 matcher key validity shape is wrong.")
        query = canonicalize_descriptor_channels(
            query_features.float(), int(self.config.canonical_dim)
        )
        key = canonicalize_descriptor_channels(
            key_features.float(), int(self.config.canonical_dim)
        )
        query = F.normalize(query, dim=-1, eps=1e-6)
        key = F.normalize(key, dim=-1, eps=1e-6)
        projected_query = F.normalize(
            self.query_projection(query), dim=-1, eps=1e-6
        )
        projected_key = F.normalize(
            self.key_projection(key), dim=-1, eps=1e-6
        )
        logits = torch.einsum("bqd,brd->bqr", projected_query, projected_key)
        logits = logits / float(self.config.temperature)
        pair_valid = query_valid.bool()[..., None] & key_valid.bool()[:, None, :]
        logits = logits.masked_fill(~pair_valid, -1e4)
        dustbin = self.dustbin_logit.expand(*logits.shape[:-1], 1)
        # Invalid padded queries are deterministic dustbin rows and are not
        # supervised.  This keeps tensor shapes fixed without fabricating a
        # match from padding.
        logits = torch.cat([logits, dustbin], dim=-1)
        invalid_query = ~query_valid.bool()
        if bool(invalid_query.any()):
            invalid_logits = torch.full_like(logits, -1e4)
            invalid_logits[..., -1] = 0.0
            logits = torch.where(invalid_query[..., None], invalid_logits, logits)
        probability = torch.softmax(logits, dim=-1)
        return {
            "logits": logits,
            "probability": probability,
            "query_embedding": projected_query,
            "key_embedding": projected_key,
        }


def canonicalize_descriptor_channels(
    descriptors: torch.Tensor, target_dim: int
) -> torch.Tensor:
    """Parameter-free channel canonicalization for a fair Q/K parameter count."""

    if descriptors.ndim < 2:
        raise ValueError("V9 descriptors need a channel dimension.")
    channels = int(descriptors.shape[-1])
    target = int(target_dim)
    if channels == target:
        return descriptors
    if channels < target:
        return F.pad(descriptors, (0, target - channels))
    leading = descriptors.shape[:-1]
    flattened = descriptors.reshape(-1, 1, channels)
    reduced = F.adaptive_avg_pool1d(flattened, target)
    return reduced.reshape(*leading, target)


def sample_stream_patch_descriptors(
    token_levels: torch.Tensor,
    *,
    patch_start_idx: int,
    patch_shape: tuple[int, int],
    local_uv_normalized: torch.Tensor,
) -> torch.Tensor:
    """Bilinearly sample the last frozen StreamVGGT DPT patch level."""

    if token_levels.ndim != 4:
        raise ValueError("V9 token_levels must be [L,S,N,C].")
    if local_uv_normalized.ndim != 4 or local_uv_normalized.shape[-1] != 2:
        raise ValueError("V9 local UV must be [S,K,P,2].")
    sequence, instances, points = local_uv_normalized.shape[:3]
    patch_h, patch_w = (int(value) for value in patch_shape)
    patches = token_levels[-1, :, int(patch_start_idx) :].float()
    if patches.shape[:2] != (sequence, patch_h * patch_w):
        raise ValueError(
            "V9 patch token shape disagrees with patch_shape: "
            f"{tuple(patches.shape)} vs {(sequence, patch_h, patch_w)}."
        )
    feature = patches.reshape(sequence, patch_h, patch_w, -1).permute(0, 3, 1, 2)
    grid = local_uv_normalized.float().reshape(sequence, 1, instances * points, 2)
    sampled = F.grid_sample(
        feature,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    sampled = sampled[:, :, 0].transpose(1, 2)
    return sampled.reshape(sequence, instances, points, -1).cpu()


def build_soft_match_target(
    labels: LocalTokenReprojection,
    *,
    history_uv_normalized: torch.Tensor,
    history_valid: torch.Tensor,
    image_size: tuple[int, int],
    config: MatcherConfig,
) -> SoftMatchTarget:
    """Create a visibility-aware continuous-UV target plus one dustbin."""

    config.validate()
    if history_uv_normalized.ndim != 2 or history_uv_normalized.shape[-1] != 2:
        raise ValueError("V9 target history UV must be [R,2].")
    if history_valid.shape != history_uv_normalized.shape[:-1]:
        raise ValueError("V9 target history UV/valid shapes disagree.")
    height, width = (int(value) for value in image_size)
    key_uv = torch.stack(
        [
            (history_uv_normalized[:, 0].double() + 1.0)
            * 0.5
            * max(width - 1, 1),
            (history_uv_normalized[:, 1].double() + 1.0)
            * 0.5
            * max(height - 1, 1),
        ],
        dim=-1,
    )
    queries = int(labels.current_uv.shape[0])
    keys = int(key_uv.shape[0])
    distance = torch.linalg.vector_norm(
        labels.history_target_uv[:, None, :] - key_uv[None, :, :], dim=-1
    )
    pair_valid = (
        labels.target_visible[:, None]
        & history_valid.bool()[None, :]
        & torch.isfinite(distance)
    )
    eligible = pair_valid & distance.le(float(config.target_radius_pixels))
    logits = (-0.5 * (distance / float(config.target_sigma_pixels)).square()).masked_fill(
        ~eligible, -1e4
    )
    key_probability = torch.softmax(logits, dim=-1) * eligible.to(logits.dtype)
    key_probability = key_probability / key_probability.sum(
        dim=-1, keepdim=True
    ).clamp_min(1e-12)
    has_key_support = eligible.any(dim=-1)
    dustbin = (~has_key_support).to(key_probability.dtype)[:, None]
    probability = torch.cat([key_probability, dustbin], dim=-1)
    probability = probability * labels.query_valid[:, None].to(probability.dtype)
    nearest = torch.where(
        pair_valid,
        distance,
        torch.full_like(distance, float("inf")),
    ).min(dim=-1).values
    return SoftMatchTarget(
        probability=probability,
        supervised=labels.query_valid.bool(),
        visible_with_key_support=has_key_support & labels.query_valid.bool(),
        target_uv=labels.history_target_uv,
        nearest_key_distance_pixels=nearest,
    )


def uniform_match_probability(
    query_valid: torch.Tensor, key_valid: torch.Tensor
) -> torch.Tensor:
    """No-descriptor lower bound over valid UV candidates and dustbin."""

    if query_valid.ndim != 2 or key_valid.ndim != 2:
        raise ValueError("V9 uniform validity must be [B,Q] and [B,R].")
    batch, queries = query_valid.shape
    keys = key_valid.shape[1]
    logits = torch.zeros(
        batch,
        queries,
        keys + 1,
        dtype=torch.float32,
        device=query_valid.device,
    )
    logits[..., :-1] = logits[..., :-1].masked_fill(
        ~key_valid.bool()[:, None, :], -1e4
    )
    invalid = ~query_valid.bool()
    if bool(invalid.any()):
        invalid_logits = torch.full_like(logits, -1e4)
        invalid_logits[..., -1] = 0.0
        logits = torch.where(invalid[..., None], invalid_logits, logits)
    return torch.softmax(logits, dim=-1)


def correspondence_loss(
    forward_probability: torch.Tensor,
    forward_target: SoftMatchTarget,
    *,
    reverse_probability: torch.Tensor | None = None,
    reverse_target: SoftMatchTarget | None = None,
    cycle_weight: float = 0.0,
) -> MatchLoss:
    """Bidirectional soft CE plus descriptor-only cycle consistency."""

    forward_ce = _soft_cross_entropy(forward_probability, forward_target)
    losses = [forward_ce]
    supervised = int(forward_target.supervised.sum())
    if reverse_probability is not None and reverse_target is not None:
        losses.append(_soft_cross_entropy(reverse_probability, reverse_target))
        supervised += int(reverse_target.supervised.sum())
    cross_entropy = torch.stack(losses).mean()
    cycle = forward_probability.new_zeros(())
    if reverse_probability is not None and float(cycle_weight) > 0.0:
        forward_real = forward_probability[..., :-1]
        reverse_real = reverse_probability[..., :-1]
        cycle_probability = torch.bmm(forward_real, reverse_real)
        diagonal = torch.diagonal(cycle_probability, dim1=-2, dim2=-1)
        cycle_mask = forward_target.visible_with_key_support
        if bool(cycle_mask.any()):
            cycle = -torch.log(diagonal.clamp_min(1e-8))[cycle_mask].mean()
    return MatchLoss(
        loss=cross_entropy + float(cycle_weight) * cycle,
        cross_entropy=cross_entropy,
        cycle=cycle,
        supervised_queries=supervised,
    )


def probability_to_correspondences(
    probability: torch.Tensor,
    *,
    current_uv: torch.Tensor,
    history_uv_normalized: torch.Tensor,
    query_valid: torch.Tensor,
    key_valid: torch.Tensor,
    image_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert soft rows to continuous UV without a hand-tuned confidence gate."""

    if probability.ndim != 2:
        raise ValueError("V9 probability must be [Q,R+1].")
    height, width = (int(value) for value in image_size)
    key_uv = torch.stack(
        [
            (history_uv_normalized[:, 0].double() + 1.0)
            * 0.5
            * max(width - 1, 1),
            (history_uv_normalized[:, 1].double() + 1.0)
            * 0.5
            * max(height - 1, 1),
        ],
        dim=-1,
    )
    real = probability[:, :-1].double() * key_valid.double()[None, :]
    real_mass = real.sum(dim=-1)
    normalized = real / real_mass[:, None].clamp_min(1e-12)
    predicted_uv = normalized @ key_uv
    predicted_class = probability.argmax(dim=-1)
    accepted = (
        query_valid.bool()
        & key_valid.bool().any()
        & predicted_class.ne(probability.shape[-1] - 1)
        & torch.isfinite(predicted_uv).all(dim=-1)
    )
    concentration = normalized.max(dim=-1).values
    weights = real_mass * concentration
    return current_uv[accepted], predicted_uv[accepted], weights[accepted], accepted


def _soft_cross_entropy(
    probability: torch.Tensor, target: SoftMatchTarget
) -> torch.Tensor:
    if probability.shape != target.probability.shape:
        raise ValueError("V9 prediction/target probability shapes disagree.")
    per_query = -(
        target.probability.to(probability.dtype)
        * probability.clamp_min(1e-8).log()
    ).sum(dim=-1)
    mask = target.supervised.bool()
    if not bool(mask.any()):
        return probability.sum() * 0.0
    return per_query[mask].mean()
