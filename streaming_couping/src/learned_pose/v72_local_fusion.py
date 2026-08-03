"""V7.2 residuals using true SAM3.1 mask-local descriptor sets."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .v6_camera_fusion import (
    V6FusionConfig,
    homogeneous_world_to_camera,
    se3_exp,
)
from .v7_fusion import LocalGeometryMatcher, V7PoseFusion
from .v71_causal_fusion import build_common_instance_state


V72_ARCHITECTURES = (
    "sam_local_pool",
    "sam_local_match",
    "geometry_local_match",
    "dual_local_match",
    "sam_key_global_geometry",
    "dual_key_global_geometry",
)

V72_INPUT_VARIANTS = (
    "normal",
    "instance_off",
    "local_off",
    "wrong_local_identity",
    "shuffle_local_time",
    "wrong_geometry",
)


@dataclass(frozen=True)
class V72ArchitectureDescription:
    mechanism: str
    identity_source: str
    value_source: str
    uses_sam_local: bool
    uses_geometry_local: bool


V72_DESCRIPTIONS = {
    "sam_local_pool": V72ArchitectureDescription(
        mechanism="learned_pool_over_current_mask_tokens",
        identity_source="sam31_mask_local_fpn",
        value_source="sam31_current_local_pool",
        uses_sam_local=True,
        uses_geometry_local=False,
    ),
    "sam_local_match": V72ArchitectureDescription(
        mechanism="current_to_reference_local_attention",
        identity_source="sam31_mask_local_fpn",
        value_source="sam31_local_match_residual",
        uses_sam_local=True,
        uses_geometry_local=False,
    ),
    "geometry_local_match": V72ArchitectureDescription(
        mechanism="current_to_reference_local_geometry_attention",
        identity_source="common_instance_gate",
        value_source="streamvggt_local_geometry_residual",
        uses_sam_local=False,
        uses_geometry_local=True,
    ),
    "dual_local_match": V72ArchitectureDescription(
        mechanism="separate_sam_and_geometry_local_matching",
        identity_source="sam31_mask_local_fpn",
        value_source="sam_plus_geometry_local_residual",
        uses_sam_local=True,
        uses_geometry_local=True,
    ),
    "sam_key_global_geometry": V72ArchitectureDescription(
        mechanism="local_identity_key_global_geometry_value",
        identity_source="sam31_local_match_residual",
        value_source="persistent_global_geometry",
        uses_sam_local=True,
        uses_geometry_local=False,
    ),
    "dual_key_global_geometry": V72ArchitectureDescription(
        mechanism="local_identity_key_global_plus_local_geometry_value",
        identity_source="sam31_local_match_residual",
        value_source="persistent_global_plus_local_geometry",
        uses_sam_local=True,
        uses_geometry_local=True,
    ),
}


class SAMLocalMatcher(nn.Module):
    """Match a current mask-local SAM token set to the reference set."""

    def __init__(
        self,
        *,
        feature_dim: int,
        config: V6FusionConfig,
    ) -> None:
        super().__init__()
        hidden = int(config.hidden_dim)
        self.point_encoder = _mlp(int(feature_dim) + 2, hidden)
        self.attention = nn.MultiheadAttention(
            hidden,
            int(config.num_heads),
            dropout=float(config.dropout),
            batch_first=True,
        )
        self.residual_encoder = _mlp(4 * hidden, hidden)

    def encode(
        self,
        features: torch.Tensor,
        uv: torch.Tensor,
    ) -> torch.Tensor:
        if features.ndim != 5 or uv.shape != (*features.shape[:-1], 2):
            raise ValueError(
                "SAM local inputs require [B,S,K,P,C] plus [B,S,K,P,2]."
            )
        return self.point_encoder(torch.cat([features.float(), uv.float()], dim=-1))

    def pool_current(
        self,
        features: torch.Tensor,
        uv: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encode(features, uv)
        if valid.shape != encoded.shape[:-1]:
            raise ValueError("SAM local feature/valid shapes disagree.")
        weight = valid.to(encoded.dtype)[..., None]
        pooled = (encoded * weight).sum(dim=-2)
        pooled = pooled / weight.sum(dim=-2).clamp_min(1.0)
        return pooled, valid.any(dim=-1)

    def forward(
        self,
        features: torch.Tensor,
        uv: torch.Tensor,
        valid: torch.Tensor,
        *,
        reference_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.encode(features, uv)
        if valid.shape != encoded.shape[:-1]:
            raise ValueError("SAM local feature/valid shapes disagree.")
        batch, sequence, instances, points, hidden = encoded.shape
        reference = encoded[:, int(reference_index)]
        reference_valid = valid[:, int(reference_index)]
        reference = reference[:, None].expand(
            batch, sequence, instances, points, hidden
        )
        reference_valid = reference_valid[:, None].expand(
            batch, sequence, instances, points
        )
        current_flat = encoded.reshape(-1, points, hidden)
        reference_flat = reference.reshape_as(current_flat)
        current_valid = valid.reshape(-1, points)
        memory_valid = reference_valid.reshape_as(current_valid)
        active = current_valid.any(dim=-1) & memory_valid.any(dim=-1)
        output = torch.zeros(
            current_flat.shape[0],
            hidden,
            dtype=encoded.dtype,
            device=encoded.device,
        )
        entropy_output = torch.zeros(
            current_flat.shape[0],
            dtype=encoded.dtype,
            device=encoded.device,
        )
        maximum_output = torch.zeros_like(entropy_output)
        if bool(active.any()):
            indices = torch.nonzero(active, as_tuple=False).flatten()
            current = current_flat.index_select(0, indices)
            memory = reference_flat.index_select(0, indices)
            current_mask = current_valid.index_select(0, indices)
            memory_mask = memory_valid.index_select(0, indices)
            matched, attention_weights = self.attention(
                current,
                memory,
                memory,
                key_padding_mask=~memory_mask,
                need_weights=True,
                average_attn_weights=False,
            )
            residual = self.residual_encoder(
                torch.cat(
                    [current, matched, current - matched, current * matched],
                    dim=-1,
                )
            )
            weight = current_mask.to(residual.dtype)[..., None]
            pooled = (residual * weight).sum(dim=1)
            pooled = pooled / weight.sum(dim=1).clamp_min(1.0)
            output.index_copy_(0, indices, pooled)
            probability = attention_weights.float().clamp_min(1e-12)
            query_weight = current_mask[:, None].to(probability.dtype)
            entropy = -(probability * probability.log()).sum(dim=-1)
            memory_count = memory_mask.sum(dim=-1).clamp_min(2).float()
            entropy = entropy / memory_count.log()[:, None, None]
            entropy = (entropy * query_weight).sum(dim=(1, 2))
            entropy = entropy / (
                query_weight.sum(dim=(1, 2)).clamp_min(1.0)
                * attention_weights.shape[1]
            )
            maximum = probability.max(dim=-1).values
            maximum = (maximum * query_weight).sum(dim=(1, 2))
            maximum = maximum / (
                query_weight.sum(dim=(1, 2)).clamp_min(1.0)
                * attention_weights.shape[1]
            )
            entropy_output.index_copy_(0, indices, entropy.to(encoded.dtype))
            maximum_output.index_copy_(0, indices, maximum.to(encoded.dtype))
        return (
            output.reshape(batch, sequence, instances, hidden),
            active.reshape(batch, sequence, instances),
            entropy_output.reshape(batch, sequence, instances),
            maximum_output.reshape(batch, sequence, instances),
        )


class V72FrozenLocalResidual(nn.Module):
    """Zero-initialized local-token residual over one frozen camera L0."""

    def __init__(
        self,
        *,
        base_model: V7PoseFusion,
        architecture: str,
        sam_local_dim: int,
        geometry_dim: int,
        geometry_local_dim: int,
        config: V6FusionConfig,
    ) -> None:
        super().__init__()
        if architecture not in V72_ARCHITECTURES:
            raise ValueError(f"Unknown V7.2 architecture: {architecture!r}.")
        if base_model.architecture != "l0_camera_only":
            raise ValueError("V7.2 requires a V7 L0 camera-only base model.")
        self.base_model = base_model
        self.architecture = architecture
        self.config = config
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)
        self.base_model.eval()

        hidden = int(config.hidden_dim)
        description = V72_DESCRIPTIONS[architecture]
        self.sam_matcher = (
            SAMLocalMatcher(feature_dim=sam_local_dim, config=config)
            if description.uses_sam_local
            else None
        )
        self.geometry_matcher = (
            LocalGeometryMatcher(feature_dim=geometry_local_dim, config=config)
            if description.uses_geometry_local
            else None
        )
        self.geometry_encoder = (
            _mlp(3 * int(geometry_dim) + 7, hidden)
            if architecture in {
                "sam_key_global_geometry",
                "dual_key_global_geometry",
            }
            else None
        )
        self.instance_attention = (
            nn.MultiheadAttention(
                hidden,
                int(config.num_heads),
                dropout=float(config.dropout),
                batch_first=True,
            )
            if self.geometry_encoder is not None
            else None
        )
        self.dual_merger = (
            _mlp(4 * hidden, hidden)
            if architecture in {
                "dual_local_match",
                "dual_key_global_geometry",
            }
            else None
        )
        self.camera_merger = _mlp(4 * hidden, hidden)
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
        sam_local_features: torch.Tensor,
        sam_local_uv: torch.Tensor,
        sam_local_valid: torch.Tensor,
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
        common_valid = state.valid
        common_active = common_valid.any(dim=2)

        sam_evidence = None
        sam_available = torch.zeros_like(common_valid)
        sam_attention_entropy = torch.zeros_like(
            state.reliability, dtype=camera.dtype
        )
        sam_attention_max = torch.zeros_like(sam_attention_entropy)
        if self.sam_matcher is not None:
            if self.architecture == "sam_local_pool":
                sam_evidence, sam_available = self.sam_matcher.pool_current(
                    sam_local_features,
                    sam_local_uv,
                    sam_local_valid,
                )
            else:
                (
                    sam_evidence,
                    sam_available,
                    sam_attention_entropy,
                    sam_attention_max,
                ) = self.sam_matcher(
                    sam_local_features,
                    sam_local_uv,
                    sam_local_valid,
                    reference_index=reference_index,
                )

        geometry_evidence = None
        geometry_available = torch.zeros_like(common_valid)
        if self.geometry_matcher is not None:
            geometry_evidence, geometry_available = self.geometry_matcher(
                local_features,
                local_valid,
                reference_index=reference_index,
            )

        if self.architecture in {"sam_local_pool", "sam_local_match"}:
            if sam_evidence is None:
                raise RuntimeError("SAM local evidence is unavailable.")
            valid = common_valid & sam_available
            evidence = _weighted_instance_pool(
                sam_evidence, valid, state.reliability
            )
            active = valid.any(dim=2)
        elif self.architecture == "geometry_local_match":
            if geometry_evidence is None:
                raise RuntimeError("Geometry local evidence is unavailable.")
            valid = common_valid & geometry_available
            evidence = _weighted_instance_pool(
                geometry_evidence, valid, state.reliability
            )
            active = valid.any(dim=2)
        elif self.architecture == "dual_local_match":
            if (
                sam_evidence is None
                or geometry_evidence is None
                or self.dual_merger is None
            ):
                raise RuntimeError("Dual local evidence is unavailable.")
            dual = self.dual_merger(
                torch.cat(
                    [
                        sam_evidence,
                        geometry_evidence,
                        sam_evidence - geometry_evidence,
                        sam_evidence * geometry_evidence,
                    ],
                    dim=-1,
                )
            )
            valid = common_valid & sam_available & geometry_available
            evidence = _weighted_instance_pool(dual, valid, state.reliability)
            active = valid.any(dim=2)
        else:
            if (
                sam_evidence is None
                or self.geometry_encoder is None
                or self.instance_attention is None
            ):
                raise RuntimeError("V7.2 decoupled local/global path is unavailable.")
            values = self.geometry_encoder(state.geometry_features)
            valid = common_valid & sam_available
            if self.architecture == "dual_key_global_geometry":
                if geometry_evidence is None or self.dual_merger is None:
                    raise RuntimeError("V7.2 local geometry is unavailable.")
                merged = self.dual_merger(
                    torch.cat(
                        [
                            values,
                            geometry_evidence,
                            values - geometry_evidence,
                            values * geometry_evidence,
                        ],
                        dim=-1,
                    )
                )
                values = torch.where(
                    geometry_available[..., None], merged, values
                )
            evidence = self._attend_instances(
                camera,
                sam_evidence,
                values,
                valid,
                state.reliability,
            )
            active = valid.any(dim=2)

        # The common gate remains a hard upper bound for every architecture.
        active = active & common_active
        evidence = torch.where(active[..., None], evidence, torch.zeros_like(evidence))
        fused = self.camera_merger(
            torch.cat(
                [camera, evidence, camera * evidence, camera - evidence],
                dim=-1,
            )
        )
        raw_delta = self.se3_head(fused)
        update_mask = active.clone()
        update_mask[:, int(reference_index)] = False
        omega = torch.tanh(raw_delta[..., :3]) * torch.deg2rad(
            raw_delta.new_tensor(float(self.config.max_rotation_degrees))
        )
        rho = torch.tanh(raw_delta[..., 3:]) * float(
            self.config.max_translation_native
        )
        omega = torch.where(update_mask[..., None], omega, torch.zeros_like(omega))
        rho = torch.where(update_mask[..., None], rho, torch.zeros_like(rho))
        base_h = homogeneous_world_to_camera(base_pose)
        composed = se3_exp(torch.cat([omega, rho], dim=-1)) @ base_h
        final = torch.where(update_mask[..., None, None], composed, base_h)
        return {
            "world_to_camera": final[..., :3, :4],
            "active_frames": update_mask,
            "evidence": evidence,
            "fused_camera": fused,
            "base_world_to_camera": base_pose,
            "sam_local_available": sam_available,
            "geometry_local_available": geometry_available,
            "sam_attention_entropy": sam_attention_entropy,
            "sam_attention_max": sam_attention_max,
        }

    def _attend_instances(
        self,
        camera: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        valid: torch.Tensor,
        reliability: torch.Tensor,
    ) -> torch.Tensor:
        if self.instance_attention is None:
            raise RuntimeError("V7.2 instance attention is unavailable.")
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
            selected_valid = flat_valid.index_select(0, indices)
            selected_reliability = flat_reliability.index_select(0, indices)
            bias = torch.where(
                selected_valid,
                torch.log(selected_reliability.clamp_min(1e-8)),
                torch.full_like(selected_reliability, float("-inf")),
            )
            attention_mask = (
                bias[:, None, None, :]
                .expand(-1, int(self.config.num_heads), 1, -1)
                .reshape(-1, 1, bias.shape[-1])
            )
            attended, _ = self.instance_attention(
                flat_camera.index_select(0, indices),
                flat_keys.index_select(0, indices),
                flat_values.index_select(0, indices),
                attn_mask=attention_mask.to(camera.dtype),
                need_weights=False,
            )
            strength = selected_reliability.max(dim=-1).values[:, None, None]
            output.index_copy_(0, indices, attended * strength.to(attended.dtype))
        return output.reshape(batch, sequence, hidden)


def perturb_v72_inputs(
    batch: dict[str, torch.Tensor],
    variant: str,
) -> dict[str, torch.Tensor]:
    """Apply content perturbations without changing the shared gate by accident."""

    if variant not in V72_INPUT_VARIANTS:
        raise ValueError(f"Unknown V7.2 input variant: {variant!r}.")
    output = {name: value for name, value in batch.items()}
    if variant == "normal":
        return output
    if variant == "instance_off":
        output["observed"] = torch.zeros_like(batch["observed"])
        output["identity_valid"] = torch.zeros_like(batch["identity_valid"])
        output["identity_unknown"] = torch.zeros_like(batch["identity_unknown"])
        output["local_valid"] = torch.zeros_like(batch["local_valid"])
        output["sam_local_valid"] = torch.zeros_like(batch["sam_local_valid"])
    elif variant == "local_off":
        output["local_valid"] = torch.zeros_like(batch["local_valid"])
        output["sam_local_valid"] = torch.zeros_like(batch["sam_local_valid"])
    elif variant == "wrong_local_identity":
        # Preserve the reference identity slots, but associate every later
        # local descriptor set with the wrong persistent instance.
        for name in ("sam_local_features", "sam_local_uv", "sam_local_valid"):
            changed = batch[name].clone()
            changed[:, 1:] = torch.roll(batch[name][:, 1:], shifts=1, dims=2)
            output[name] = changed
    elif variant == "shuffle_local_time":
        sequence = int(batch["sam_local_features"].shape[1])
        order = torch.cat(
            [
                torch.zeros(
                    1,
                    dtype=torch.long,
                    device=batch["sam_local_features"].device,
                ),
                torch.arange(
                    sequence - 1,
                    0,
                    -1,
                    device=batch["sam_local_features"].device,
                ),
            ]
        )
        for name in ("sam_local_features", "sam_local_uv", "sam_local_valid"):
            output[name] = batch[name].index_select(1, order)
    elif variant == "wrong_geometry":
        for name in ("pose_geometry", "local_features", "local_valid"):
            changed = batch[name].clone()
            changed[:, 1:] = torch.roll(batch[name][:, 1:], shifts=1, dims=2)
            output[name] = changed
    return output


def _weighted_instance_pool(
    tokens: torch.Tensor,
    valid: torch.Tensor,
    reliability: torch.Tensor,
) -> torch.Tensor:
    weights = torch.where(valid, reliability, torch.zeros_like(reliability))
    pooled = (tokens * weights[..., None]).sum(dim=2)
    return pooled / weights.sum(dim=2, keepdim=True).clamp_min(1e-8)


def _mlp(input_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(int(input_dim)),
        nn.Linear(int(input_dim), int(hidden_dim)),
        nn.GELU(),
        nn.Linear(int(hidden_dim), int(hidden_dim)),
        nn.LayerNorm(int(hidden_dim)),
    )
