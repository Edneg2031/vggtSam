"""V7 fusion ablations from global pooling to local geometry matching."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .v6_camera_fusion import (
    PersistentInstanceEncoder,
    V6FusionConfig,
    homogeneous_world_to_camera,
    se3_exp,
    v6_effective_identity_states,
)


V7_ARCHITECTURES = (
    "l0_camera_only",
    "l1_monolithic_weighted_pool",
    "l2_monolithic_cross_attention",
    "l3_identity_key_geometry_value",
    "l4_local32_hierarchical",
)
V7_INPUT_VARIANTS = (
    "normal",
    "instance_off",
    "wrong_geometry",
    "shuffle_time",
    "appearance_only",
    "geometry_only",
)


@dataclass(frozen=True)
class V7ArchitectureDescription:
    level: int
    mechanism: str
    key_source: str
    value_source: str
    spatial_tokens: int


V7_DESCRIPTIONS = {
    "l0_camera_only": V7ArchitectureDescription(
        level=0,
        mechanism="camera_control",
        key_source="none",
        value_source="none",
        spatial_tokens=0,
    ),
    "l1_monolithic_weighted_pool": V7ArchitectureDescription(
        level=1,
        mechanism="reliability_weighted_pool",
        key_source="none",
        value_source="monolithic_instance_vector",
        spatial_tokens=0,
    ),
    "l2_monolithic_cross_attention": V7ArchitectureDescription(
        level=2,
        mechanism="camera_to_instance_cross_attention",
        key_source="monolithic_instance_vector",
        value_source="monolithic_instance_vector",
        spatial_tokens=0,
    ),
    "l3_identity_key_geometry_value": V7ArchitectureDescription(
        level=3,
        mechanism="decoupled_key_value_cross_attention",
        key_source="appearance_identity_memory",
        value_source="signed_global_geometry_residual",
        spatial_tokens=0,
    ),
    "l4_local32_hierarchical": V7ArchitectureDescription(
        level=4,
        mechanism="local_match_then_instance_cross_attention",
        key_source="appearance_identity_memory",
        value_source="global_plus_local_world_geometry_residual",
        spatial_tokens=32,
    ),
}


class DecoupledPersistentEncoder(nn.Module):
    """Build identity Keys and geometry Values without early modality mixing."""

    def __init__(
        self,
        *,
        appearance_dim: int,
        geometry_dim: int,
        config: V6FusionConfig,
    ) -> None:
        super().__init__()
        self.config = config
        hidden = int(config.hidden_dim)
        # current, causal memory, difference, then explicit state variables.
        identity_dim = 3 * int(appearance_dim) + 5
        geometry_input_dim = 3 * int(geometry_dim) + 7
        self.identity_encoder = _mlp(identity_dim, hidden)
        self.geometry_encoder = _mlp(geometry_input_dim, hidden)

    def forward(
        self,
        *,
        appearance: torch.Tensor,
        geometry: torch.Tensor,
        quality: torch.Tensor,
        observed: torch.Tensor,
        identity_valid: torch.Tensor,
        identity_unknown: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, sequence, instances = observed.shape
        app_memory = torch.zeros_like(appearance[:, 0])
        geo_memory = torch.zeros_like(geometry[:, 0])
        has_memory = torch.zeros(
            batch,
            instances,
            dtype=torch.bool,
            device=appearance.device,
        )
        age = torch.zeros(
            batch,
            instances,
            dtype=appearance.dtype,
            device=appearance.device,
        )
        keys = []
        values = []
        valid_rows = []
        reliability_rows = []
        momentum = float(self.config.memory_momentum)

        for frame in range(sequence):
            current_app = appearance[:, frame]
            current_geo = geometry[:, frame]
            current_observed = observed[:, frame].bool()
            current_match, current_unknown, softened_mismatch = (
                v6_effective_identity_states(
                    observed=current_observed,
                    identity_valid=identity_valid[:, frame],
                    identity_unknown=identity_unknown[:, frame],
                    policy=self.config.identity_gate_policy,
                )
            )
            track = quality[:, frame, :, 0].clamp(0.0, 1.0)
            track_ok = track >= float(self.config.min_track_confidence)
            associated = (
                current_observed
                & track_ok
                & (current_match | current_unknown)
            )
            # A camera residual requires a current observation. Missing masks
            # may preserve memory but cannot produce new geometric evidence.
            usable = has_memory & associated
            identity_weight = torch.where(
                current_match,
                torch.ones_like(track),
                torch.full_like(
                    track,
                    float(self.config.unknown_reliability),
                ),
            )
            identity_weight = torch.where(
                softened_mismatch,
                torch.full_like(
                    identity_weight,
                    float(self.config.softened_mismatch_reliability),
                ),
                identity_weight,
            )
            geometry_strength = quality[:, frame, :, 1].clamp(0.0, 1.0)
            reliability = track * identity_weight * geometry_strength
            reliability = torch.where(
                usable,
                reliability,
                torch.zeros_like(reliability),
            )
            normalized_age = (torch.log1p(age) / 4.0)[..., None]
            key_features = torch.cat(
                [
                    current_app,
                    app_memory,
                    current_app - app_memory,
                    track[..., None],
                    current_observed.to(appearance.dtype)[..., None],
                    current_match.to(appearance.dtype)[..., None],
                    current_unknown.to(appearance.dtype)[..., None],
                    normalized_age,
                ],
                dim=-1,
            )
            value_features = torch.cat(
                [
                    current_geo,
                    geo_memory,
                    current_geo - geo_memory,
                    quality[:, frame],
                    current_observed.to(appearance.dtype)[..., None],
                    current_match.to(appearance.dtype)[..., None],
                    current_unknown.to(appearance.dtype)[..., None],
                    normalized_age,
                ],
                dim=-1,
            )
            keys.append(self.identity_encoder(key_features))
            values.append(self.geometry_encoder(value_features))
            valid_rows.append(usable & reliability.gt(0))
            reliability_rows.append(reliability)

            memory_write = current_observed & track_ok & current_match
            update = memory_write[..., None]
            first = update & (~has_memory)[..., None]
            app_candidate = (
                momentum * app_memory + (1.0 - momentum) * current_app
            )
            geo_candidate = (
                momentum * geo_memory + (1.0 - momentum) * current_geo
            )
            app_memory = torch.where(
                first,
                current_app,
                torch.where(update, app_candidate, app_memory),
            )
            geo_memory = torch.where(
                first,
                current_geo,
                torch.where(update, geo_candidate, geo_memory),
            )
            has_memory = has_memory | memory_write
            age = torch.where(has_memory, age + 1.0, age)

        return (
            torch.stack(keys, dim=1),
            torch.stack(values, dim=1),
            torch.stack(valid_rows, dim=1),
            torch.stack(reliability_rows, dim=1),
        )


class LocalGeometryMatcher(nn.Module):
    """Match current local world-geometry tokens to the reference instance."""

    def __init__(
        self,
        *,
        feature_dim: int,
        config: V6FusionConfig,
    ) -> None:
        super().__init__()
        hidden = int(config.hidden_dim)
        self.num_heads = int(config.num_heads)
        self.point_encoder = _mlp(int(feature_dim), hidden)
        self.attention = nn.MultiheadAttention(
            hidden,
            int(config.num_heads),
            dropout=float(config.dropout),
            batch_first=True,
        )
        self.residual_encoder = nn.Sequential(
            nn.LayerNorm(4 * hidden),
            nn.Linear(4 * hidden, 2 * hidden),
            nn.GELU(),
            nn.Linear(2 * hidden, hidden),
            nn.LayerNorm(hidden),
        )

    def forward(
        self,
        local_features: torch.Tensor,
        local_valid: torch.Tensor,
        *,
        reference_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if local_features.ndim != 5 or local_valid.ndim != 4:
            raise ValueError(
                "Local V7 inputs require [B,S,K,P,F] and [B,S,K,P]."
            )
        if local_features.shape[:-1] != local_valid.shape:
            raise ValueError("Local V7 feature/valid shapes disagree.")
        batch, sequence, instances, points, _ = local_features.shape
        encoded = self.point_encoder(local_features)
        reference = encoded[:, int(reference_index)]
        reference_valid = local_valid[:, int(reference_index)]
        reference = reference[:, None].expand(
            batch,
            sequence,
            instances,
            points,
            encoded.shape[-1],
        )
        reference_valid = reference_valid[:, None].expand(
            batch,
            sequence,
            instances,
            points,
        )
        flat_current = encoded.reshape(
            batch * sequence * instances,
            points,
            encoded.shape[-1],
        )
        flat_reference = reference.reshape_as(flat_current)
        flat_current_valid = local_valid.reshape(
            batch * sequence * instances,
            points,
        )
        flat_reference_valid = reference_valid.reshape_as(
            flat_current_valid
        )
        active = (
            flat_current_valid.any(dim=-1)
            & flat_reference_valid.any(dim=-1)
        )
        constraint = torch.zeros(
            batch * sequence * instances,
            encoded.shape[-1],
            dtype=encoded.dtype,
            device=encoded.device,
        )
        if bool(active.any()):
            indices = torch.nonzero(active, as_tuple=False).flatten()
            current = flat_current.index_select(0, indices)
            memory = flat_reference.index_select(0, indices)
            current_valid = flat_current_valid.index_select(0, indices)
            memory_valid = flat_reference_valid.index_select(0, indices)
            matched, _ = self.attention(
                current,
                memory,
                memory,
                key_padding_mask=~memory_valid,
                need_weights=False,
            )
            point_residual = self.residual_encoder(
                torch.cat(
                    [
                        current,
                        matched,
                        current - matched,
                        current * matched,
                    ],
                    dim=-1,
                )
            )
            weight = current_valid.to(point_residual.dtype)[..., None]
            pooled = (point_residual * weight).sum(dim=1)
            pooled = pooled / weight.sum(dim=1).clamp_min(1.0)
            constraint.index_copy_(0, indices, pooled)
        return (
            constraint.reshape(
                batch,
                sequence,
                instances,
                encoded.shape[-1],
            ),
            active.reshape(batch, sequence, instances),
        )


class V7PoseFusion(nn.Module):
    """Common bounded SE(3) head with progressively heavier fusion frontends."""

    def __init__(
        self,
        *,
        architecture: str,
        camera_dim: int,
        appearance_dim: int,
        geometry_dim: int,
        local_feature_dim: int,
        config: V6FusionConfig,
    ) -> None:
        super().__init__()
        if architecture not in V7_ARCHITECTURES:
            raise ValueError(f"Unknown V7 architecture: {architecture!r}.")
        self.architecture = architecture
        self.config = config
        hidden = int(config.hidden_dim)
        self.camera_projector = _mlp(int(camera_dim), hidden)
        self.monolithic_encoder = None
        self.monolithic_norm = None
        self.monolithic_attention = None
        self.decoupled_encoder = None
        self.decoupled_attention = None
        self.local_matcher = None
        self.local_value_merger = None

        if architecture in {
            "l1_monolithic_weighted_pool",
            "l2_monolithic_cross_attention",
        }:
            self.monolithic_encoder = PersistentInstanceEncoder(
                appearance_dim=int(appearance_dim),
                geometry_dim=int(geometry_dim),
                config=config,
            )
        if architecture == "l2_monolithic_cross_attention":
            self.monolithic_norm = nn.LayerNorm(hidden)
            self.monolithic_attention = nn.MultiheadAttention(
                hidden,
                int(config.num_heads),
                dropout=float(config.dropout),
                batch_first=True,
            )
        if architecture in {
            "l3_identity_key_geometry_value",
            "l4_local32_hierarchical",
        }:
            self.decoupled_encoder = DecoupledPersistentEncoder(
                appearance_dim=int(appearance_dim),
                geometry_dim=int(geometry_dim),
                config=config,
            )
            self.decoupled_attention = nn.MultiheadAttention(
                hidden,
                int(config.num_heads),
                dropout=float(config.dropout),
                batch_first=True,
            )
        if architecture == "l4_local32_hierarchical":
            self.local_matcher = LocalGeometryMatcher(
                feature_dim=int(local_feature_dim),
                config=config,
            )
            self.local_value_merger = nn.Sequential(
                nn.LayerNorm(4 * hidden),
                nn.Linear(4 * hidden, 2 * hidden),
                nn.GELU(),
                nn.Linear(2 * hidden, hidden),
                nn.LayerNorm(hidden),
            )

        self.feature_merger = nn.Sequential(
            nn.LayerNorm(4 * hidden),
            nn.Linear(4 * hidden, 2 * hidden),
            nn.GELU(),
            nn.Linear(2 * hidden, hidden),
            nn.LayerNorm(hidden),
        )
        self.se3_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 6),
        )
        nn.init.zeros_(self.se3_head[-1].weight)
        nn.init.zeros_(self.se3_head[-1].bias)

    def forward(
        self,
        *,
        camera_hidden: torch.Tensor,
        baseline_world_to_camera: torch.Tensor,
        appearance: torch.Tensor,
        geometry: torch.Tensor,
        quality: torch.Tensor,
        observed: torch.Tensor,
        identity_valid: torch.Tensor,
        identity_unknown: torch.Tensor,
        local_features: torch.Tensor,
        local_valid: torch.Tensor,
        reference_index: int,
    ) -> dict[str, torch.Tensor]:
        camera = self.camera_projector(camera_hidden)
        if self.architecture == "l0_camera_only":
            evidence = torch.zeros_like(camera)
            active = torch.ones(
                camera.shape[:2],
                dtype=torch.bool,
                device=camera.device,
            )
        elif self.architecture == "l1_monolithic_weighted_pool":
            tokens, valid, reliability = self._monolithic(
                appearance=appearance,
                geometry=geometry,
                quality=quality,
                observed=observed,
                identity_valid=identity_valid,
                identity_unknown=identity_unknown,
            )
            weights = torch.where(
                valid,
                reliability,
                torch.zeros_like(reliability),
            )
            evidence = (
                tokens * weights[..., None]
            ).sum(dim=2) / weights.sum(dim=2, keepdim=True).clamp_min(1e-8)
            active = weights.gt(0).any(dim=2)
            evidence = torch.where(
                active[..., None],
                evidence,
                torch.zeros_like(evidence),
            )
        elif self.architecture == "l2_monolithic_cross_attention":
            tokens, valid, reliability = self._monolithic(
                appearance=appearance,
                geometry=geometry,
                quality=quality,
                observed=observed,
                identity_valid=identity_valid,
                identity_unknown=identity_unknown,
            )
            evidence, active = self._monolithic_cross_attention(
                camera,
                tokens,
                valid,
                reliability,
            )
        else:
            if self.decoupled_encoder is None:
                raise RuntimeError("V7 decoupled encoder is unavailable.")
            keys, values, valid, reliability = self.decoupled_encoder(
                appearance=appearance,
                geometry=geometry,
                quality=quality,
                observed=observed,
                identity_valid=identity_valid,
                identity_unknown=identity_unknown,
            )
            if self.architecture == "l4_local32_hierarchical":
                if self.local_matcher is None or self.local_value_merger is None:
                    raise RuntimeError("V7 local matcher is unavailable.")
                local, local_available = self.local_matcher(
                    local_features,
                    local_valid,
                    reference_index=reference_index,
                )
                values = self.local_value_merger(
                    torch.cat(
                        [
                            values,
                            local,
                            values - local,
                            values * local,
                        ],
                        dim=-1,
                    )
                )
                valid = valid & local_available
                reliability = torch.where(
                    valid,
                    reliability,
                    torch.zeros_like(reliability),
                )
            evidence, active = self._decoupled_cross_attention(
                camera,
                keys,
                values,
                valid,
                reliability,
            )

        merged = self.feature_merger(
            torch.cat(
                [
                    camera,
                    evidence,
                    camera * evidence,
                    camera - evidence,
                ],
                dim=-1,
            )
        )
        raw_delta = self.se3_head(merged)
        update_mask = active.clone()
        update_mask[:, int(reference_index)] = False
        omega = torch.tanh(raw_delta[..., :3]) * torch.deg2rad(
            raw_delta.new_tensor(float(self.config.max_rotation_degrees))
        )
        rho = (
            torch.tanh(raw_delta[..., 3:])
            * float(self.config.max_translation_native)
        )
        omega = torch.where(
            update_mask[..., None],
            omega,
            torch.zeros_like(omega),
        )
        rho = torch.where(
            update_mask[..., None],
            rho,
            torch.zeros_like(rho),
        )
        baseline_h = homogeneous_world_to_camera(
            baseline_world_to_camera
        )
        composed = se3_exp(torch.cat([omega, rho], dim=-1)) @ baseline_h
        final = torch.where(
            update_mask[..., None, None],
            composed,
            baseline_h,
        )
        return {
            "world_to_camera": final[..., :3, :4],
            "active_frames": update_mask,
            "evidence": evidence,
            "fused_camera": merged,
        }

    def _monolithic(
        self,
        *,
        appearance: torch.Tensor,
        geometry: torch.Tensor,
        quality: torch.Tensor,
        observed: torch.Tensor,
        identity_valid: torch.Tensor,
        identity_unknown: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.monolithic_encoder is None:
            raise RuntimeError("V7 monolithic encoder is unavailable.")
        return self.monolithic_encoder(
            appearance=appearance,
            geometry=geometry,
            quality=quality,
            observed=observed,
            identity_valid=identity_valid,
            identity_unknown=identity_unknown,
        )

    def _monolithic_cross_attention(
        self,
        camera: torch.Tensor,
        tokens: torch.Tensor,
        valid: torch.Tensor,
        reliability: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.monolithic_attention is None or self.monolithic_norm is None:
            raise RuntimeError("V7 monolithic attention is unavailable.")
        batch, sequence, hidden = camera.shape
        flat_camera = camera.reshape(batch * sequence, 1, hidden)
        flat_tokens = tokens.reshape(
            batch * sequence,
            tokens.shape[2],
            hidden,
        )
        flat_valid = valid.reshape(batch * sequence, valid.shape[2])
        flat_reliability = reliability.reshape_as(flat_valid)
        active = flat_valid.any(dim=-1)
        output = torch.zeros_like(flat_camera)
        if bool(active.any()):
            indices = torch.nonzero(active, as_tuple=False).flatten()
            key_value = self.monolithic_norm(
                flat_tokens.index_select(0, indices)
            ) * flat_reliability.index_select(
                0,
                indices,
            )[..., None].to(camera.dtype)
            attended, _ = self.monolithic_attention(
                flat_camera.index_select(0, indices),
                key_value,
                key_value,
                key_padding_mask=~flat_valid.index_select(0, indices),
                need_weights=False,
            )
            output.index_copy_(0, indices, attended.to(output.dtype))
        return (
            output.reshape(batch, sequence, hidden),
            active.reshape(batch, sequence),
        )

    def _decoupled_cross_attention(
        self,
        camera: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        valid: torch.Tensor,
        reliability: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.decoupled_attention is None:
            raise RuntimeError("V7 decoupled attention is unavailable.")
        batch, sequence, hidden = camera.shape
        flat_camera = camera.reshape(batch * sequence, 1, hidden)
        flat_keys = keys.reshape(
            batch * sequence,
            keys.shape[2],
            hidden,
        )
        flat_values = values.reshape_as(flat_keys)
        flat_valid = valid.reshape(batch * sequence, valid.shape[2])
        flat_reliability = reliability.reshape_as(flat_valid).float()
        active = (flat_valid & flat_reliability.gt(0)).any(dim=-1)
        output = torch.zeros_like(flat_camera)
        if bool(active.any()):
            indices = torch.nonzero(active, as_tuple=False).flatten()
            current_valid = flat_valid.index_select(0, indices)
            current_reliability = flat_reliability.index_select(
                0,
                indices,
            )
            bias = torch.where(
                current_valid,
                torch.log(current_reliability.clamp_min(1e-8)),
                torch.full_like(current_reliability, float("-inf")),
            )
            attention_mask = (
                bias[:, None, None, :]
                .expand(
                    -1,
                    int(self.config.num_heads),
                    1,
                    -1,
                )
                .reshape(-1, 1, bias.shape[-1])
            )
            attended, _ = self.decoupled_attention(
                flat_camera.index_select(0, indices),
                flat_keys.index_select(0, indices),
                flat_values.index_select(0, indices),
                attn_mask=attention_mask.to(camera.dtype),
                need_weights=False,
            )
            # Preserve absolute evidence strength. Softmax alone would erase
            # the distinction between one weak and one strong instance.
            strength = current_reliability.max(dim=-1).values[:, None, None]
            attended = attended * strength.to(attended.dtype)
            output.index_copy_(0, indices, attended.to(output.dtype))
        return (
            output.reshape(batch, sequence, hidden),
            active.reshape(batch, sequence),
        )


def perturb_v7_inputs(
    batch: dict[str, torch.Tensor],
    variant: str,
) -> dict[str, torch.Tensor]:
    """Apply a fixed-checkpoint semantic perturbation without changing gates."""

    if variant not in V7_INPUT_VARIANTS:
        raise ValueError(f"Unknown V7 input variant: {variant!r}.")
    output = {
        key: batch[key]
        for key in (
            "camera_hidden",
            "appearance",
            "pose_geometry",
            "quality",
            "observed",
            "identity_valid",
            "identity_unknown",
            "local_features",
            "local_valid",
        )
    }
    if variant == "instance_off":
        output["observed"] = torch.zeros_like(output["observed"])
        output["identity_valid"] = torch.zeros_like(
            output["identity_valid"]
        )
        output["identity_unknown"] = torch.zeros_like(
            output["identity_unknown"]
        )
        output["local_valid"] = torch.zeros_like(output["local_valid"])
    elif variant in {"wrong_geometry", "shuffle_time"}:
        sequence = output["pose_geometry"].shape[1]
        permutation = torch.roll(
            torch.arange(
                sequence,
                device=output["pose_geometry"].device,
            ),
            shifts=1,
        )
        output["pose_geometry"] = output["pose_geometry"].index_select(
            1,
            permutation,
        )
        output["local_features"] = output["local_features"].index_select(
            1,
            permutation,
        )
        output["local_valid"] = output["local_valid"].index_select(
            1,
            permutation,
        )
        if variant == "shuffle_time":
            output["appearance"] = output["appearance"].index_select(
                1,
                permutation,
            )
    elif variant == "appearance_only":
        output["pose_geometry"] = torch.zeros_like(
            output["pose_geometry"]
        )
        output["local_features"] = torch.zeros_like(
            output["local_features"]
        )
        output["local_valid"] = torch.zeros_like(output["local_valid"])
        quality = output["quality"].clone()
        quality[..., 1:] = 0
        output["quality"] = quality
    elif variant == "geometry_only":
        output["appearance"] = torch.zeros_like(output["appearance"])
    return output


def _mlp(input_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(int(input_dim)),
        nn.Linear(int(input_dim), int(hidden_dim)),
        nn.GELU(),
        nn.Linear(int(hidden_dim), int(hidden_dim)),
        nn.LayerNorm(int(hidden_dim)),
    )
