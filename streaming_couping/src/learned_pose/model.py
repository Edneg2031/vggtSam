"""Persistent instance encoder and zero-initialized pose fusion modules."""

from __future__ import annotations

from dataclasses import asdict
from typing import Mapping

import torch
import torch.nn as nn

from .config import FusionConfig


class CausalInstanceTokenizer(nn.Module):
    """Turn per-frame observations into causal persistent instance tokens."""

    def __init__(
        self,
        appearance_dim: int,
        geometry_dim: int,
        config: FusionConfig,
    ) -> None:
        super().__init__()
        self.appearance_dim = int(appearance_dim)
        self.geometry_dim = int(geometry_dim)
        self.config = config
        input_dim = 3 * self.appearance_dim + 3 * self.geometry_dim + 4
        hidden = max(config.instance_dim, min(1024, input_dim))
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, config.instance_dim),
            nn.LayerNorm(config.instance_dim),
        )

    def forward(
        self,
        appearance: torch.Tensor,
        geometry: torch.Tensor,
        quality: torch.Tensor,
        observed: torch.Tensor,
        *,
        mode: str,
        perturbation: str = "aligned",
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        _validate_observation_shapes(appearance, geometry, quality, observed)
        appearance, geometry, quality, observed, gate_quality = _perturb_observations(
            appearance,
            geometry,
            quality,
            observed,
            perturbation=perturbation,
        )
        use_appearance = mode in {
            "camera_sam_only",
            "camera_token_fusion",
            "all_token_fusion",
        }
        use_geometry = mode in {
            "camera_geometry_only",
            "camera_token_fusion",
            "all_token_fusion",
        }
        if not use_appearance:
            appearance = torch.zeros_like(appearance)
        if not use_geometry:
            geometry = torch.zeros_like(geometry)
            # A real SAM-only control must not receive geometry/static scores
            # through a side channel. Only tracker confidence remains active.
            quality = torch.stack(
                [quality[..., 0], torch.ones_like(quality[..., 1]), torch.ones_like(quality[..., 2])],
                dim=-1,
            )
            gate_quality = quality

        batch, sequence, instances = observed.shape
        app_memory = torch.zeros_like(appearance[:, 0])
        geo_memory = torch.zeros_like(geometry[:, 0])
        has_memory = torch.zeros(batch, instances, dtype=torch.bool, device=observed.device)
        age = torch.zeros(batch, instances, dtype=appearance.dtype, device=appearance.device)
        token_rows = []
        valid_rows = []
        update_rows = []
        momentum = float(self.config.memory_momentum)
        for frame in range(sequence):
            current_app = appearance[:, frame]
            current_geo = geometry[:, frame]
            current_quality = quality[:, frame]
            current_observed = observed[:, frame].bool()
            trusted = (
                current_observed
                & (gate_quality[:, frame, :, 0] >= self.config.min_track_confidence)
                & (gate_quality[:, frame, :, 1] >= self.config.min_geometry_confidence)
                & (gate_quality[:, frame, :, 2] >= self.config.min_static_score)
            )
            token_valid = trusted & has_memory
            features = torch.cat(
                [
                    current_app,
                    app_memory,
                    current_app - app_memory,
                    current_geo,
                    geo_memory,
                    current_geo - geo_memory,
                    current_quality,
                    torch.log1p(age)[..., None] / 4.0,
                ],
                dim=-1,
            )
            token = self.encoder(features)
            token_rows.append(token)
            valid_rows.append(token_valid)
            update_rows.append(trusted)

            update = trusted[..., None]
            first = update & (~has_memory)[..., None]
            app_candidate = momentum * app_memory + (1.0 - momentum) * current_app
            geo_candidate = momentum * geo_memory + (1.0 - momentum) * current_geo
            app_memory = torch.where(first, current_app, torch.where(update, app_candidate, app_memory))
            geo_memory = torch.where(first, current_geo, torch.where(update, geo_candidate, geo_memory))
            has_memory = has_memory | trusted
            age = torch.where(has_memory, age + 1.0, age)

        tokens = torch.stack(token_rows, dim=1)
        valid = torch.stack(valid_rows, dim=1)
        updates = torch.stack(update_rows, dim=1)
        return tokens, valid, {
            "effective_instances": valid.float().sum(dim=-1),
            "memory_updates": updates.float().sum(dim=-1),
        }


class ZeroInitializedCrossAttention(nn.Module):
    """Query persistent instances and write an exactly-zero initial residual."""

    def __init__(self, query_dim: int, instance_dim: int, config: FusionConfig) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(query_dim)
        self.instance_norm = nn.LayerNorm(instance_dim)
        self.query_proj = nn.Linear(query_dim, config.attention_dim)
        self.instance_proj = nn.Linear(instance_dim, config.attention_dim)
        self.attention = nn.MultiheadAttention(
            config.attention_dim,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        # No bias: a frame with no valid instance must stay exactly unchanged
        # even after training.
        self.zero_proj = nn.Linear(config.attention_dim, query_dim, bias=False)
        nn.init.zeros_(self.zero_proj.weight)
        self.gate_logit = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        queries: torch.Tensor,
        instance_tokens: torch.Tensor,
        instance_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Fuse ``[B,S,Q,D]`` queries with ``[B,S,K,Di]`` instances."""

        if queries.ndim != 4 or instance_tokens.ndim != 4 or instance_valid.ndim != 3:
            raise ValueError("Expected queries [B,S,Q,D], instances [B,S,K,D], valid [B,S,K].")
        batch, sequence, query_count, query_dim = queries.shape
        if instance_tokens.shape[:3] != instance_valid.shape:
            raise ValueError("Instance token and validity shapes disagree.")
        if instance_tokens.shape[:2] != (batch, sequence):
            raise ValueError("Query and instance [B,S] dimensions disagree.")
        flat_queries = queries.reshape(batch * sequence, query_count, query_dim)
        flat_instances = instance_tokens.reshape(
            batch * sequence,
            instance_tokens.shape[2],
            instance_tokens.shape[3],
        )
        flat_valid = instance_valid.reshape(batch * sequence, instance_valid.shape[2])
        active = flat_valid.any(dim=1)
        attention_output = torch.zeros(
            batch * sequence,
            query_count,
            self.query_proj.out_features,
            dtype=queries.dtype,
            device=queries.device,
        )
        attention_entropy = torch.zeros(batch * sequence, dtype=queries.dtype, device=queries.device)
        if bool(active.any()):
            active_indices = torch.nonzero(active, as_tuple=False).flatten()
            q = self.query_proj(self.query_norm(flat_queries.index_select(0, active_indices)))
            kv = self.instance_proj(
                self.instance_norm(flat_instances.index_select(0, active_indices))
            )
            key_padding = ~flat_valid.index_select(0, active_indices)
            update, weights = self.attention(
                q,
                kv,
                kv,
                key_padding_mask=key_padding,
                need_weights=True,
                average_attn_weights=False,
            )
            attention_output.index_copy_(
                0,
                active_indices,
                update.to(dtype=attention_output.dtype),
            )
            probabilities = weights.float().clamp_min(1e-8)
            entropy = -(probabilities * probabilities.log()).sum(dim=-1).mean(dim=(1, 2))
            attention_entropy.index_copy_(0, active_indices, entropy.to(attention_entropy.dtype))
        residual = self.zero_proj(attention_output)
        gate = torch.sigmoid(self.gate_logit)
        refined = flat_queries + gate * residual
        residual = residual.reshape(batch, sequence, query_count, query_dim)
        residual_mean_square = residual.float().square().mean()
        return refined.reshape_as(queries), {
            "gate": gate,
            # RMS is detached logging only: differentiating sqrt(x)^2 at x=0
            # can create a 0/0 gradient on the exactly-zero first step.
            "residual_rms": residual_mean_square.detach().sqrt(),
            "residual_mean_square": residual_mean_square,
            "attention_entropy": attention_entropy.reshape(batch, sequence).mean(),
            "active_frame_fraction": active.float().mean(),
        }


class InstancePoseAdapter(nn.Module):
    """Learned modules only; SAM3 and StreamVGGT stay outside this state dict."""

    def __init__(
        self,
        *,
        appearance_dim: int,
        geometry_dim: int,
        token_dim: int,
        config: FusionConfig,
    ) -> None:
        super().__init__()
        self.appearance_dim = int(appearance_dim)
        self.geometry_dim = int(geometry_dim)
        self.token_dim = int(token_dim)
        self.config = config
        self.tokenizer = CausalInstanceTokenizer(appearance_dim, geometry_dim, config)
        self.camera_fusion = ZeroInitializedCrossAttention(
            token_dim,
            config.instance_dim,
            config,
        )
        self.all_token_fusions = nn.ModuleDict(
            {
                str(layer): ZeroInitializedCrossAttention(
                    token_dim,
                    config.instance_dim,
                    config,
                )
                for layer in config.dpt_layer_indices
            }
        )

    def forward_camera(
        self,
        camera_hidden: torch.Tensor,
        *,
        appearance: torch.Tensor,
        geometry: torch.Tensor,
        quality: torch.Tensor,
        observed: torch.Tensor,
        mode: str,
        perturbation: str = "aligned",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if perturbation == "module_off" or mode == "baseline":
            zero = camera_hidden.new_zeros(())
            return camera_hidden, {
                "gate": zero,
                "residual_rms": zero,
                "residual_mean_square": zero,
                "attention_entropy": zero,
                "active_frame_fraction": zero,
                "effective_instances": camera_hidden.new_zeros(camera_hidden.shape[:2]),
                "memory_updates": camera_hidden.new_zeros(camera_hidden.shape[:2]),
            }
        instance_tokens, valid, memory_log = self.tokenizer(
            appearance,
            geometry,
            quality,
            observed,
            mode=mode,
            perturbation=perturbation,
        )
        refined, log = self.camera_fusion(
            camera_hidden.unsqueeze(2),
            instance_tokens,
            valid,
        )
        return refined[:, :, 0], {**memory_log, **log}

    def forward_all_tokens(
        self,
        token_levels: Mapping[int, torch.Tensor],
        *,
        appearance: torch.Tensor,
        geometry: torch.Tensor,
        quality: torch.Tensor,
        observed: torch.Tensor,
        perturbation: str = "aligned",
    ) -> tuple[dict[int, torch.Tensor], dict[str, torch.Tensor]]:
        if perturbation == "module_off":
            return dict(token_levels), {}
        instance_tokens, valid, memory_log = self.tokenizer(
            appearance,
            geometry,
            quality,
            observed,
            mode="all_token_fusion",
            perturbation=perturbation,
        )
        output: dict[int, torch.Tensor] = {}
        logs: dict[str, torch.Tensor] = dict(memory_log)
        for layer, tokens in token_levels.items():
            key = str(int(layer))
            if key not in self.all_token_fusions:
                output[int(layer)] = tokens
                continue
            updated, current = self.all_token_fusions[key](tokens, instance_tokens, valid)
            output[int(layer)] = updated
            logs.update({f"layer_{key}_{name}": value for name, value in current.items()})
        return output, logs

    def metadata(self) -> dict:
        return {
            "appearance_dim": self.appearance_dim,
            "geometry_dim": self.geometry_dim,
            "token_dim": self.token_dim,
            "fusion_config": asdict(self.config),
        }


def _perturb_observations(
    appearance: torch.Tensor,
    geometry: torch.Tensor,
    quality: torch.Tensor,
    observed: torch.Tensor,
    *,
    perturbation: str,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    if perturbation in {"aligned", "module_off"}:
        return appearance, geometry, quality, observed, quality
    if perturbation == "zero_appearance":
        return torch.zeros_like(appearance), geometry, quality, observed, quality
    if perturbation == "zero_geometry":
        gate_quality = quality
        geometry = torch.zeros_like(geometry)
        quality = quality.clone()
        # Preserve the aligned checkpoint's trusted-instance mask while
        # removing geometry/static scores from the token itself.  This keeps
        # zero-geometry distinct from module-off without admitting observations
        # that the aligned branch rejected.
        quality[..., 1:] = 0.0
        return appearance, geometry, quality, observed, gate_quality
    if perturbation == "shuffle_instance_ids":
        if appearance.shape[2] <= 1 or appearance.shape[1] <= 1:
            return appearance, geometry, quality, observed, quality
        appearance = appearance.clone()
        geometry = geometry.clone()
        quality = quality.clone()
        observed = observed.clone()
        appearance[:, 1:] = torch.roll(appearance[:, 1:], shifts=1, dims=2)
        geometry[:, 1:] = torch.roll(geometry[:, 1:], shifts=1, dims=2)
        quality[:, 1:] = torch.roll(quality[:, 1:], shifts=1, dims=2)
        observed[:, 1:] = torch.roll(observed[:, 1:], shifts=1, dims=2)
        return appearance, geometry, quality, observed, quality
    if perturbation == "shuffle_time":
        if appearance.shape[1] <= 2:
            return appearance, geometry, quality, observed, quality
        appearance = appearance.clone()
        geometry = geometry.clone()
        quality = quality.clone()
        observed = observed.clone()
        appearance[:, 1:] = torch.roll(appearance[:, 1:], shifts=1, dims=1)
        geometry[:, 1:] = torch.roll(geometry[:, 1:], shifts=1, dims=1)
        quality[:, 1:] = torch.roll(quality[:, 1:], shifts=1, dims=1)
        observed[:, 1:] = torch.roll(observed[:, 1:], shifts=1, dims=1)
        return appearance, geometry, quality, observed, quality
    raise ValueError(f"Unknown instance perturbation: {perturbation}")


def _validate_observation_shapes(
    appearance: torch.Tensor,
    geometry: torch.Tensor,
    quality: torch.Tensor,
    observed: torch.Tensor,
) -> None:
    if appearance.ndim != 4 or geometry.ndim != 4 or quality.ndim != 4 or observed.ndim != 3:
        raise ValueError("Observation tensors must be [B,S,K,D] and observed [B,S,K].")
    if (
        appearance.shape[:3] != observed.shape
        or geometry.shape[:3] != observed.shape
        or quality.shape[:3] != observed.shape
    ):
        raise ValueError("Observation tensors do not share [B,S,K].")
    if quality.shape[-1] != 3:
        raise ValueError("quality must contain track, geometry, and static scores.")
