"""Causal V7.1 instance residuals on top of a frozen camera baseline."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .v6_camera_fusion import (
    V6FusionConfig,
    homogeneous_world_to_camera,
    se3_exp,
    v6_effective_identity_states,
)
from .v7_fusion import LocalGeometryMatcher, V7PoseFusion


V71_RESIDUAL_ARCHITECTURES = (
    "camera_extra_all",
    "camera_extra_common_gate",
    "gate_only",
    "appearance_pool",
    "geometry_pool",
    "decoupled_global",
    "local32_decoupled",
)


@dataclass(frozen=True)
class V71ArchitectureDescription:
    evidence: str
    instance_content: bool
    spatial_tokens: int


V71_DESCRIPTIONS = {
    "camera_extra_all": V71ArchitectureDescription(
        evidence="extra_camera_capacity_control",
        instance_content=False,
        spatial_tokens=0,
    ),
    "camera_extra_common_gate": V71ArchitectureDescription(
        evidence="extra_camera_capacity_with_shared_instance_gate",
        instance_content=False,
        spatial_tokens=0,
    ),
    "gate_only": V71ArchitectureDescription(
        evidence="shared_instance_gate_statistics_only",
        instance_content=False,
        spatial_tokens=0,
    ),
    "appearance_pool": V71ArchitectureDescription(
        evidence="persistent_appearance_weighted_pool",
        instance_content=True,
        spatial_tokens=0,
    ),
    "geometry_pool": V71ArchitectureDescription(
        evidence="persistent_geometry_weighted_pool",
        instance_content=True,
        spatial_tokens=0,
    ),
    "decoupled_global": V71ArchitectureDescription(
        evidence="appearance_key_geometry_value",
        instance_content=True,
        spatial_tokens=0,
    ),
    "local32_decoupled": V71ArchitectureDescription(
        evidence="appearance_key_global_plus_local_geometry_value",
        instance_content=True,
        spatial_tokens=32,
    ),
}


@dataclass
class CommonInstanceState:
    appearance_features: torch.Tensor
    geometry_features: torch.Tensor
    valid: torch.Tensor
    reliability: torch.Tensor
    gate_features: torch.Tensor


def build_common_instance_state(
    *,
    appearance: torch.Tensor,
    geometry: torch.Tensor,
    quality: torch.Tensor,
    observed: torch.Tensor,
    identity_valid: torch.Tensor,
    identity_unknown: torch.Tensor,
    config: V6FusionConfig,
) -> CommonInstanceState:
    """Build one gate shared by every V7.1 instance architecture.

    Geometry confidence is retained as a feature but deliberately does not
    decide whether a frame is active. This prevents gate coverage from being
    confused with the evidence architecture under comparison.
    """

    if appearance.ndim != 4 or geometry.ndim != 4:
        raise ValueError("V7.1 appearance/geometry must be [B,S,K,D].")
    if quality.shape[:3] != appearance.shape[:3] or quality.shape[-1] != 3:
        raise ValueError("V7.1 quality must be [B,S,K,3].")
    if geometry.shape[:3] != appearance.shape[:3]:
        raise ValueError("V7.1 appearance/geometry leading shapes differ.")
    expected = appearance.shape[:3]
    if any(
        value.shape != expected
        for value in (observed, identity_valid, identity_unknown)
    ):
        raise ValueError("V7.1 identity-state shapes disagree.")

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
    app_rows = []
    geo_rows = []
    valid_rows = []
    reliability_rows = []
    gate_rows = []
    momentum = float(config.memory_momentum)

    for frame in range(sequence):
        current_app = appearance[:, frame]
        current_geo = geometry[:, frame]
        current_observed = observed[:, frame].bool()
        current_match, current_unknown, softened_mismatch = (
            v6_effective_identity_states(
                observed=current_observed,
                identity_valid=identity_valid[:, frame],
                identity_unknown=identity_unknown[:, frame],
                policy=config.identity_gate_policy,
            )
        )
        track = quality[:, frame, :, 0].clamp(0.0, 1.0)
        track_ok = track >= float(config.min_track_confidence)
        associated = (
            current_observed
            & track_ok
            & (current_match | current_unknown)
        )
        usable = has_memory & associated
        identity_weight = torch.where(
            current_match,
            torch.ones_like(track),
            torch.full_like(track, float(config.unknown_reliability)),
        )
        identity_weight = torch.where(
            softened_mismatch,
            torch.full_like(
                identity_weight,
                float(config.softened_mismatch_reliability),
            ),
            identity_weight,
        )
        reliability = torch.where(
            usable,
            track * identity_weight,
            torch.zeros_like(track),
        )
        normalized_age = (torch.log1p(age) / 4.0)[..., None]
        identity_state_features = torch.cat(
            [
                track[..., None],
                current_observed.to(appearance.dtype)[..., None],
                current_match.to(appearance.dtype)[..., None],
                current_unknown.to(appearance.dtype)[..., None],
                normalized_age,
            ],
            dim=-1,
        )
        geometry_state_features = torch.cat(
            [
                quality[:, frame],
                current_observed.to(appearance.dtype)[..., None],
                current_match.to(appearance.dtype)[..., None],
                current_unknown.to(appearance.dtype)[..., None],
                normalized_age,
            ],
            dim=-1,
        )
        app_rows.append(
            torch.cat(
                [
                    current_app,
                    app_memory,
                    current_app - app_memory,
                    identity_state_features,
                ],
                dim=-1,
            )
        )
        geo_rows.append(
            torch.cat(
                [
                    current_geo,
                    geo_memory,
                    current_geo - geo_memory,
                    geometry_state_features,
                ],
                dim=-1,
            )
        )
        valid_rows.append(usable)
        reliability_rows.append(reliability)
        valid_float = usable.to(appearance.dtype)
        valid_count = valid_float.sum(dim=-1).clamp_min(1.0)
        gate_rows.append(
            torch.stack(
                [
                    valid_float.mean(dim=-1),
                    reliability.sum(dim=-1) / valid_count,
                    reliability.max(dim=-1).values,
                    current_observed.to(appearance.dtype).mean(dim=-1),
                ],
                dim=-1,
            )
        )

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

    return CommonInstanceState(
        appearance_features=torch.stack(app_rows, dim=1),
        geometry_features=torch.stack(geo_rows, dim=1),
        valid=torch.stack(valid_rows, dim=1),
        reliability=torch.stack(reliability_rows, dim=1),
        gate_features=torch.stack(gate_rows, dim=1),
    )


class V71FrozenResidualFusion(nn.Module):
    """Train only an incremental instance residual over a frozen V7 L0."""

    def __init__(
        self,
        *,
        base_model: V7PoseFusion,
        architecture: str,
        appearance_dim: int,
        geometry_dim: int,
        local_feature_dim: int,
        config: V6FusionConfig,
    ) -> None:
        super().__init__()
        if architecture not in V71_RESIDUAL_ARCHITECTURES:
            raise ValueError(
                f"Unknown V7.1 residual architecture: {architecture!r}."
            )
        if base_model.architecture != "l0_camera_only":
            raise ValueError("V7.1 requires an L0 camera-only base model.")
        self.base_model = base_model
        self.architecture = architecture
        self.config = config
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)
        self.base_model.eval()

        hidden = int(config.hidden_dim)
        app_dim = 3 * int(appearance_dim) + 5
        geo_dim = 3 * int(geometry_dim) + 7
        self.camera_evidence = None
        self.gate_encoder = None
        self.appearance_encoder = None
        self.geometry_encoder = None
        self.cross_attention = None
        self.local_matcher = None
        self.local_merger = None

        if architecture in {
            "camera_extra_all",
            "camera_extra_common_gate",
        }:
            self.camera_evidence = _mlp(hidden, hidden)
        elif architecture == "gate_only":
            self.gate_encoder = _mlp(4, hidden)
        elif architecture == "appearance_pool":
            self.appearance_encoder = _mlp(app_dim, hidden)
        elif architecture == "geometry_pool":
            self.geometry_encoder = _mlp(geo_dim, hidden)
        else:
            self.appearance_encoder = _mlp(app_dim, hidden)
            self.geometry_encoder = _mlp(geo_dim, hidden)
            self.cross_attention = nn.MultiheadAttention(
                hidden,
                int(config.num_heads),
                dropout=float(config.dropout),
                batch_first=True,
            )
            if architecture == "local32_decoupled":
                self.local_matcher = LocalGeometryMatcher(
                    feature_dim=int(local_feature_dim),
                    config=config,
                )
                self.local_merger = nn.Sequential(
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

    def train(self, mode: bool = True):
        super().train(mode)
        self.base_model.eval()
        return self

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
        with torch.no_grad():
            base = self.base_model(
                camera_hidden=camera_hidden,
                baseline_world_to_camera=baseline_world_to_camera,
                appearance=appearance,
                geometry=geometry,
                quality=quality,
                observed=observed,
                identity_valid=identity_valid,
                identity_unknown=identity_unknown,
                local_features=local_features,
                local_valid=local_valid,
                reference_index=reference_index,
            )
        camera = base["fused_camera"].detach()
        base_pose = base["world_to_camera"].detach()
        state = build_common_instance_state(
            appearance=appearance,
            geometry=geometry,
            quality=quality,
            observed=observed,
            identity_valid=identity_valid,
            identity_unknown=identity_unknown,
            config=self.config,
        )
        common_active = state.valid.any(dim=2)

        if self.architecture in {
            "camera_extra_all",
            "camera_extra_common_gate",
        }:
            if self.camera_evidence is None:
                raise RuntimeError("V7.1 camera control is unavailable.")
            evidence = self.camera_evidence(camera)
            active = (
                torch.ones_like(common_active)
                if self.architecture == "camera_extra_all"
                else common_active
            )
        elif self.architecture == "gate_only":
            if self.gate_encoder is None:
                raise RuntimeError("V7.1 gate encoder is unavailable.")
            evidence = self.gate_encoder(state.gate_features)
            active = common_active
        elif self.architecture == "appearance_pool":
            if self.appearance_encoder is None:
                raise RuntimeError("V7.1 appearance encoder is unavailable.")
            evidence = self._weighted_pool(
                self.appearance_encoder(state.appearance_features),
                state.valid,
                state.reliability,
            )
            active = common_active
        elif self.architecture == "geometry_pool":
            if self.geometry_encoder is None:
                raise RuntimeError("V7.1 geometry encoder is unavailable.")
            evidence = self._weighted_pool(
                self.geometry_encoder(state.geometry_features),
                state.valid,
                state.reliability,
            )
            active = common_active
        else:
            if (
                self.appearance_encoder is None
                or self.geometry_encoder is None
            ):
                raise RuntimeError("V7.1 decoupled encoders are unavailable.")
            keys = self.appearance_encoder(state.appearance_features)
            values = self.geometry_encoder(state.geometry_features)
            if self.architecture == "local32_decoupled":
                if self.local_matcher is None or self.local_merger is None:
                    raise RuntimeError("V7.1 local matcher is unavailable.")
                local, local_available = self.local_matcher(
                    local_features,
                    local_valid,
                    reference_index=reference_index,
                )
                merged_local = self.local_merger(
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
                values = torch.where(
                    local_available[..., None],
                    merged_local,
                    values,
                )
            evidence = self._cross_attend(
                camera,
                keys,
                values,
                state.valid,
                state.reliability,
            )
            active = common_active

        evidence = torch.where(
            active[..., None],
            evidence,
            torch.zeros_like(evidence),
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
        base_h = homogeneous_world_to_camera(base_pose)
        composed = se3_exp(torch.cat([omega, rho], dim=-1)) @ base_h
        final = torch.where(
            update_mask[..., None, None],
            composed,
            base_h,
        )
        return {
            "world_to_camera": final[..., :3, :4],
            "active_frames": update_mask,
            "evidence": evidence,
            "fused_camera": merged,
            "base_world_to_camera": base_pose,
        }

    @staticmethod
    def _weighted_pool(
        tokens: torch.Tensor,
        valid: torch.Tensor,
        reliability: torch.Tensor,
    ) -> torch.Tensor:
        weights = torch.where(
            valid,
            reliability,
            torch.zeros_like(reliability),
        )
        pooled = (tokens * weights[..., None]).sum(dim=2)
        return pooled / weights.sum(dim=2, keepdim=True).clamp_min(1e-8)

    def _cross_attend(
        self,
        camera: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        valid: torch.Tensor,
        reliability: torch.Tensor,
    ) -> torch.Tensor:
        if self.cross_attention is None:
            raise RuntimeError("V7.1 cross-attention is unavailable.")
        batch, sequence, hidden = camera.shape
        flat_camera = camera.reshape(batch * sequence, 1, hidden)
        flat_keys = keys.reshape(batch * sequence, keys.shape[2], hidden)
        flat_values = values.reshape_as(flat_keys)
        flat_valid = valid.reshape(batch * sequence, valid.shape[2])
        flat_reliability = reliability.reshape_as(flat_valid).float()
        active = flat_valid.any(dim=-1)
        output = torch.zeros_like(flat_camera)
        if bool(active.any()):
            indices = torch.nonzero(active, as_tuple=False).flatten()
            current_valid = flat_valid.index_select(0, indices)
            current_reliability = flat_reliability.index_select(0, indices)
            bias = torch.where(
                current_valid,
                torch.log(current_reliability.clamp_min(1e-8)),
                torch.full_like(current_reliability, float("-inf")),
            )
            attention_mask = (
                bias[:, None, None, :]
                .expand(-1, int(self.config.num_heads), 1, -1)
                .reshape(-1, 1, bias.shape[-1])
            )
            attended, _ = self.cross_attention(
                flat_camera.index_select(0, indices),
                flat_keys.index_select(0, indices),
                flat_values.index_select(0, indices),
                attn_mask=attention_mask.to(camera.dtype),
                need_weights=False,
            )
            strength = current_reliability.max(dim=-1).values[:, None, None]
            output.index_copy_(
                0,
                indices,
                (attended * strength.to(attended.dtype)).to(output.dtype),
            )
        return output.reshape(batch, sequence, hidden)


def _mlp(input_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(int(input_dim)),
        nn.Linear(int(input_dim), int(hidden_dim)),
        nn.GELU(),
        nn.Linear(int(hidden_dim), int(hidden_dim)),
        nn.LayerNorm(int(hidden_dim)),
    )
