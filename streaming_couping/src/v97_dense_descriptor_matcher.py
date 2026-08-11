"""Dense descriptor matcher for the V9.7 causal correspondence test.

The matcher has no pose input or pose head.  It predicts one coarse history
cell on the real SAM 72x72 grid and a bounded sub-cell offset.  Camera pose is
recovered later by the frozen O-R1 epipolar solver.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from streaming_couping.src.v90_explicit_matcher import (
    canonicalize_descriptor_channels,
)


DENSE_ARCHITECTURES = (
    "sam_dense",
    "stream_dense",
    "coordinate_only",
    "sam_train_off",
)

SAM_EVAL_VARIANTS = (
    "normal",
    "sam_off",
    "channel_permute",
    "shuffle_time",
)


@dataclass(frozen=True)
class DenseMatcherConfig:
    canonical_dim: int = 256
    projection_dim: int = 64
    offset_hidden_dim: int = 128
    temperature: float = 0.07
    offset_weight: float = 1.0
    pck_threshold_pixels: float = 1.0

    def validate(self) -> None:
        if min(
            int(self.canonical_dim),
            int(self.projection_dim),
            int(self.offset_hidden_dim),
        ) < 2:
            raise ValueError("V9.7 matcher dimensions must be at least two.")
        positive = (
            float(self.temperature),
            float(self.offset_weight),
            float(self.pck_threshold_pixels),
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("V9.7 positive matcher settings must be finite.")


@dataclass(frozen=True)
class DenseMatchTarget:
    coarse_index: torch.Tensor
    offset_normalized: torch.Tensor
    query_valid: torch.Tensor
    visible: torch.Tensor
    target_uv_pixels: torch.Tensor


@dataclass(frozen=True)
class DenseMatchLoss:
    loss: torch.Tensor
    classification: torch.Tensor
    offset: torch.Tensor
    supervised_queries: int
    visible_queries: int


@dataclass(frozen=True)
class DenseDecode:
    history_uv_pixels: torch.Tensor
    accepted: torch.Tensor
    selected_key_indices: torch.Tensor
    confidence: torch.Tensor
    offset_normalized: torch.Tensor


class DenseSubgridMatcher(nn.Module):
    """Parameter-matched dense Q/K classifier plus bounded offset decoder."""

    def __init__(self, config: DenseMatcherConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        canonical = int(config.canonical_dim)
        projected = int(config.projection_dim)
        hidden = int(config.offset_hidden_dim)
        self.query_projection = nn.Linear(canonical, projected, bias=False)
        self.key_projection = nn.Linear(canonical, projected, bias=False)
        self.offset_head = nn.Sequential(
            nn.Linear(2 * projected, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
            nn.Tanh(),
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
            raise ValueError("V9.7 features must be [B,Q,C] and [B,R,C].")
        if query_features.shape[0] != key_features.shape[0]:
            raise ValueError("V9.7 query/key batches disagree.")
        if query_valid.shape != query_features.shape[:2]:
            raise ValueError("V9.7 query validity has the wrong shape.")
        if key_valid.shape != key_features.shape[:2]:
            raise ValueError("V9.7 key validity has the wrong shape.")
        query = canonicalize_descriptor_channels(
            query_features.float(), int(self.config.canonical_dim)
        )
        key = canonicalize_descriptor_channels(
            key_features.float(), int(self.config.canonical_dim)
        )
        query = F.normalize(query, dim=-1, eps=1e-6)
        key = F.normalize(key, dim=-1, eps=1e-6)
        query_embedding = F.normalize(
            self.query_projection(query), dim=-1, eps=1e-6
        )
        key_embedding = F.normalize(
            self.key_projection(key), dim=-1, eps=1e-6
        )
        logits = torch.einsum(
            "bqd,brd->bqr", query_embedding, key_embedding
        ) / float(self.config.temperature)
        pair_valid = query_valid.bool()[..., None] & key_valid.bool()[:, None, :]
        logits = logits.masked_fill(~pair_valid, -1e4)
        logits = torch.cat(
            [logits, self.dustbin_logit.expand(*logits.shape[:-1], 1)], dim=-1
        )
        invalid = ~query_valid.bool()
        if bool(invalid.any()):
            fallback = torch.full_like(logits, -1e4)
            fallback[..., -1] = 0.0
            logits = torch.where(invalid[..., None], fallback, logits)
        return {
            "logits": logits,
            "probability": torch.softmax(logits, dim=-1),
            "query_embedding": query_embedding,
            "key_embedding": key_embedding,
        }

    def offsets_for_keys(
        self,
        query_embedding: torch.Tensor,
        key_embedding: torch.Tensor,
        key_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Predict [-1,1] sub-cell offsets for one key per query."""

        if query_embedding.ndim != 3 or key_embedding.ndim != 3:
            raise ValueError("V9.7 projected embeddings must be rank three.")
        if key_indices.shape != query_embedding.shape[:2]:
            raise ValueError("V9.7 selected key indices have the wrong shape.")
        bounded = key_indices.long().clamp(0, max(key_embedding.shape[1] - 1, 0))
        selected = torch.gather(
            key_embedding,
            1,
            bounded[..., None].expand(-1, -1, key_embedding.shape[-1]),
        )
        return self.offset_head(torch.cat([query_embedding, selected], dim=-1))


def build_dense_match_target(
    target_uv_pixels: torch.Tensor,
    *,
    query_valid: torch.Tensor,
    visible: torch.Tensor,
    grid_size: tuple[int, int],
    image_size: tuple[int, int],
) -> DenseMatchTarget:
    """Assign each continuous target to its nearest cell and residual offset."""

    if target_uv_pixels.ndim != 3 or target_uv_pixels.shape[-1] != 2:
        raise ValueError("V9.7 target UV must be [B,Q,2].")
    if query_valid.shape != target_uv_pixels.shape[:2]:
        raise ValueError("V9.7 target/query validity shapes disagree.")
    if visible.shape != query_valid.shape:
        raise ValueError("V9.7 target visibility shape is wrong.")
    grid_h, grid_w = (int(value) for value in grid_size)
    image_h, image_w = (int(value) for value in image_size)
    step_x = float(max(image_w - 1, 1)) / float(max(grid_w - 1, 1))
    step_y = float(max(image_h - 1, 1)) / float(max(grid_h - 1, 1))
    x_index = torch.round(target_uv_pixels[..., 0].double() / step_x).long()
    y_index = torch.round(target_uv_pixels[..., 1].double() / step_y).long()
    x_index = x_index.clamp(0, grid_w - 1)
    y_index = y_index.clamp(0, grid_h - 1)
    coarse = y_index * grid_w + x_index
    center = torch.stack(
        [x_index.to(torch.float64) * step_x, y_index.to(torch.float64) * step_y],
        dim=-1,
    )
    half_step = torch.tensor(
        [0.5 * step_x, 0.5 * step_y],
        dtype=torch.float64,
        device=target_uv_pixels.device,
    )
    offset = (target_uv_pixels.double() - center) / half_step.clamp_min(1e-12)
    offset = offset.clamp(-1.0, 1.0).to(torch.float32)
    supervised = query_valid.bool()
    visible_valid = supervised & visible.bool()
    dustbin = grid_h * grid_w
    coarse = torch.where(visible_valid, coarse, torch.full_like(coarse, dustbin))
    return DenseMatchTarget(
        coarse_index=coarse,
        offset_normalized=offset,
        query_valid=supervised,
        visible=visible_valid,
        target_uv_pixels=target_uv_pixels.double(),
    )


def dense_match_loss(
    model: DenseSubgridMatcher,
    output: dict[str, torch.Tensor],
    target: DenseMatchTarget,
) -> DenseMatchLoss:
    """Coarse-cell CE plus GT-cell offset loss; no pose loss is present."""

    supervised = target.query_valid.bool()
    if not bool(supervised.any()):
        zero = output["logits"].sum() * 0.0
        return DenseMatchLoss(zero, zero, zero, 0, 0)
    classification = F.cross_entropy(
        output["logits"][supervised], target.coarse_index.long()[supervised]
    )
    visible = target.visible.bool()
    offset = classification.new_zeros(())
    if bool(visible.any()):
        predicted = model.offsets_for_keys(
            output["query_embedding"],
            output["key_embedding"],
            target.coarse_index,
        )
        offset = F.smooth_l1_loss(
            predicted[visible], target.offset_normalized.to(predicted.device)[visible]
        )
    return DenseMatchLoss(
        loss=classification + float(model.config.offset_weight) * offset,
        classification=classification,
        offset=offset,
        supervised_queries=int(supervised.sum()),
        visible_queries=int(visible.sum()),
    )


def decode_dense_matches(
    model: DenseSubgridMatcher,
    output: dict[str, torch.Tensor],
    *,
    grid_size: tuple[int, int],
    image_size: tuple[int, int],
    query_valid: torch.Tensor,
) -> DenseDecode:
    """Decode a hard coarse cell and bounded learned sub-cell coordinate."""

    probability = output["probability"]
    if probability.shape[:2] != query_valid.shape:
        raise ValueError("V9.7 probability/query validity shapes disagree.")
    real_keys = probability.shape[-1] - 1
    selected = probability.argmax(dim=-1)
    accepted = query_valid.bool() & selected.lt(real_keys)
    bounded = selected.clamp(0, max(real_keys - 1, 0))
    offset = model.offsets_for_keys(
        output["query_embedding"], output["key_embedding"], bounded
    )
    grid_h, grid_w = (int(value) for value in grid_size)
    image_h, image_w = (int(value) for value in image_size)
    if real_keys != grid_h * grid_w:
        raise ValueError(
            f"V9.7 key count={real_keys}, expected {grid_h}x{grid_w}."
        )
    step_x = float(max(image_w - 1, 1)) / float(max(grid_w - 1, 1))
    step_y = float(max(image_h - 1, 1)) / float(max(grid_h - 1, 1))
    x = bounded.remainder(grid_w).to(torch.float64) * step_x
    y = torch.div(bounded, grid_w, rounding_mode="floor").to(torch.float64) * step_y
    half_step = torch.tensor(
        [0.5 * step_x, 0.5 * step_y],
        dtype=torch.float64,
        device=probability.device,
    )
    center = torch.stack([x, y], dim=-1)
    uv = center + offset.double() * half_step
    uv[..., 0].clamp_(0.0, float(max(image_w - 1, 0)))
    uv[..., 1].clamp_(0.0, float(max(image_h - 1, 0)))
    confidence = torch.gather(probability[..., :-1], -1, bounded[..., None])[..., 0]
    return DenseDecode(
        history_uv_pixels=uv,
        accepted=accepted,
        selected_key_indices=bounded,
        confidence=confidence,
        offset_normalized=offset,
    )


def coordinate_descriptors(
    grid_uv_normalized: torch.Tensor, *, channels: int
) -> torch.Tensor:
    """Deterministic positional control with no learned appearance signal."""

    if grid_uv_normalized.ndim != 2 or grid_uv_normalized.shape[-1] != 2:
        raise ValueError("V9.7 normalized grid UV must be [R,2].")
    target = int(channels)
    if target < 4:
        raise ValueError("V9.7 coordinate control needs at least four channels.")
    frequencies = torch.arange(
        1, max(target // 4, 1) + 1, dtype=torch.float32
    )
    phase = math.pi * grid_uv_normalized.float()[..., None] * frequencies
    encoded = torch.cat(
        [phase[..., 0, :].sin(), phase[..., 0, :].cos(),
         phase[..., 1, :].sin(), phase[..., 1, :].cos()],
        dim=-1,
    )
    if encoded.shape[-1] < target:
        encoded = F.pad(encoded, (0, target - encoded.shape[-1]))
    return encoded[..., :target].contiguous()


def deterministic_channel_permutation(channels: int, *, seed: int = 97) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randperm(int(channels), generator=generator)

