"""V7.3 SAM3.1-weighted local geometry correspondence residuals."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .v6_camera_fusion import (
    V6FusionConfig,
    homogeneous_world_to_camera,
    se3_exp,
)
from .v7_fusion import V7PoseFusion
from .v71_causal_fusion import build_common_instance_state


V73_ARCHITECTURES = (
    "uniform_transport",
    "geometry_transport",
    "sam_transport",
    "sam_geometry_transport",
)

V73_INPUT_VARIANTS = (
    "normal",
    "instance_off",
    "geometry_off",
    "sam_off",
    "uniform_sam",
    "wrong_sam_identity",
    "shuffle_sam_time",
    "wrong_local_geometry",
)


@dataclass(frozen=True)
class V73ArchitectureDescription:
    mechanism: str
    correspondence_source: str
    uses_sam: bool
    uses_geometry_affinity: bool


V73_DESCRIPTIONS = {
    "uniform_transport": V73ArchitectureDescription(
        mechanism="uniform_reference_geometry_transport",
        correspondence_source="uniform_valid_reference_points",
        uses_sam=False,
        uses_geometry_affinity=False,
    ),
    "geometry_transport": V73ArchitectureDescription(
        mechanism="geometry_similarity_weighted_transport",
        correspondence_source="streamvggt_local_geometry",
        uses_sam=False,
        uses_geometry_affinity=True,
    ),
    "sam_transport": V73ArchitectureDescription(
        mechanism="sam_identity_weighted_geometry_transport",
        correspondence_source="sam31_local_identity",
        uses_sam=True,
        uses_geometry_affinity=False,
    ),
    "sam_geometry_transport": V73ArchitectureDescription(
        mechanism="sam_and_geometry_product_of_experts_transport",
        correspondence_source="sam31_identity_plus_streamvggt_geometry",
        uses_sam=True,
        uses_geometry_affinity=True,
    ),
}


class SAMWeightedGeometryMatcher(nn.Module):
    """Use SAM identity affinity to transport reference geometry Values."""

    def __init__(
        self,
        *,
        architecture: str,
        sam_feature_dim: int,
        geometry_feature_dim: int,
        config: V6FusionConfig,
        memory_mode: str = "fixed_reference",
    ) -> None:
        super().__init__()
        if architecture not in V73_ARCHITECTURES:
            raise ValueError(f"Unknown V7.3 architecture: {architecture!r}.")
        self.architecture = architecture
        if memory_mode not in {"fixed_reference", "causal_last_observation"}:
            raise ValueError(f"Unknown V7.3 memory mode: {memory_mode!r}.")
        self.memory_mode = memory_mode
        self.description = V73_DESCRIPTIONS[architecture]
        hidden = int(config.hidden_dim)
        self.hidden_dim = hidden
        self.geometry_encoder = _mlp(int(geometry_feature_dim), hidden)
        self.geometry_query = (
            nn.Linear(hidden, hidden, bias=False)
            if self.description.uses_geometry_affinity
            else None
        )
        self.geometry_key = (
            nn.Linear(hidden, hidden, bias=False)
            if self.description.uses_geometry_affinity
            else None
        )
        self.sam_encoder = (
            # UV is used only to interpolate the SAM descriptor onto a
            # StreamVGGT sample.  It is deliberately not encoded into the
            # identity descriptor, otherwise absolute image position could
            # become a shortcut for the claimed SAM identity affinity.
            _mlp(int(sam_feature_dim), hidden)
            if self.description.uses_sam
            else None
        )
        self.sam_query = (
            nn.Linear(hidden, hidden, bias=False)
            if self.description.uses_sam
            else None
        )
        self.sam_key = (
            nn.Linear(hidden, hidden, bias=False)
            if self.description.uses_sam
            else None
        )
        self.residual_encoder = nn.Sequential(
            nn.LayerNorm(4 * hidden),
            nn.Linear(4 * hidden, 2 * hidden),
            nn.GELU(),
            nn.Linear(2 * hidden, hidden),
            nn.LayerNorm(hidden),
        )
        # UV is normalized to [-1, 1]. This fixed temperature only transfers
        # SAM descriptors to nearby StreamVGGT geometry samples; it is not a
        # learned pose shortcut.
        self.spatial_temperature = 0.02
        self.affinity_temperature = 0.10

    def forward(
        self,
        *,
        local_features: torch.Tensor,
        local_valid: torch.Tensor,
        sam_local_features: torch.Tensor,
        sam_local_uv: torch.Tensor,
        sam_local_valid: torch.Tensor,
        reference_index: int,
        uniform_sam: bool = False,
        memory_write: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if local_features.ndim != 5 or local_valid.shape != local_features.shape[:-1]:
            raise ValueError(
                "V7.3 geometry inputs require [B,S,K,P,F] and [B,S,K,P]."
            )
        if sam_local_features.ndim != 5:
            raise ValueError("V7.3 SAM features require [B,S,K,P,C].")
        if sam_local_uv.shape != (*sam_local_features.shape[:-1], 2):
            raise ValueError("V7.3 SAM UV/features shapes disagree.")
        if sam_local_valid.shape != sam_local_features.shape[:-1]:
            raise ValueError("V7.3 SAM valid/features shapes disagree.")
        if local_features.shape[:3] != sam_local_features.shape[:3]:
            raise ValueError("V7.3 SAM and geometry sequence/instance shapes differ.")
        if local_features.shape[-1] < 5:
            raise ValueError("V7.3 geometry tokens must contain normalized UV.")
        if memory_write is not None and memory_write.shape != local_features.shape[:3]:
            raise ValueError("V7.3 memory_write must have shape [B,S,K].")

        geometry = self.geometry_encoder(local_features.float())
        geometry_uv = local_features[..., 3:5].float().clamp(-1.0, 1.0)
        batch, sequence, instances, points, hidden = geometry.shape
        reference = int(reference_index)
        if reference < 0 or reference >= sequence:
            raise ValueError("V7.3 reference index is outside the sequence.")

        current_sam_identity = None
        current_sam_available = torch.zeros_like(local_valid)
        if self.description.uses_sam:
            if self.sam_encoder is None:
                raise RuntimeError("V7.3 SAM encoder is unavailable.")
            encoded_sam = self.sam_encoder(sam_local_features.float())
            current_sam_identity, current_sam_available = (
                self._interpolate_sam_to_geometry(
                    encoded_sam,
                    sam_local_uv.float(),
                    sam_local_valid,
                    geometry_uv,
                    local_valid,
                )
            )
        if self.memory_mode == "causal_last_observation":
            reference_geometry, reference_valid = _causal_previous_memory(
                geometry,
                local_valid,
                write_mask=memory_write,
            )
            if current_sam_identity is not None:
                reference_sam_identity, reference_sam_available = (
                    _causal_previous_memory(
                        current_sam_identity,
                        current_sam_available,
                        write_mask=memory_write,
                    )
                )
            else:
                reference_sam_identity = None
                reference_sam_available = torch.zeros_like(reference_valid)
        else:
            reference_geometry = geometry[:, reference][:, None].expand(
                batch, sequence, instances, points, hidden
            )
            reference_valid = local_valid[:, reference][:, None].expand(
                batch, sequence, instances, points
            )
            reference_sam_identity = None
            reference_sam_available = torch.zeros_like(reference_valid)
            if current_sam_identity is not None:
                reference_sam_identity = current_sam_identity[:, reference][
                    :, None
                ].expand(batch, sequence, instances, points, hidden)
                reference_sam_available = current_sam_available[:, reference][
                    :, None
                ].expand(batch, sequence, instances, points)

        flat_geometry = geometry.reshape(-1, points, hidden)
        flat_reference_geometry = reference_geometry.reshape_as(flat_geometry)
        flat_current_valid = local_valid.reshape(-1, points)
        flat_reference_valid = reference_valid.reshape(-1, points)
        flat_sam_current = (
            current_sam_identity.reshape(-1, points, hidden)
            if current_sam_identity is not None
            else None
        )
        flat_sam_reference = (
            reference_sam_identity.reshape(-1, points, hidden)
            if reference_sam_identity is not None
            else None
        )
        flat_sam_current_valid = current_sam_available.reshape(-1, points)
        flat_sam_reference_valid = reference_sam_available.reshape(-1, points)

        geometry_logits = torch.zeros(
            flat_geometry.shape[0],
            points,
            points,
            dtype=flat_geometry.dtype,
            device=flat_geometry.device,
        )
        if self.description.uses_geometry_affinity:
            if self.geometry_query is None or self.geometry_key is None:
                raise RuntimeError("V7.3 geometry projections are unavailable.")
            geometry_logits = self._cosine_logits(
                self.geometry_query(flat_geometry),
                self.geometry_key(flat_reference_geometry),
            )
        sam_logits = torch.zeros_like(geometry_logits)
        sam_pair_valid = torch.zeros_like(geometry_logits, dtype=torch.bool)
        if flat_sam_current is not None and flat_sam_reference is not None:
            if self.sam_query is None or self.sam_key is None:
                raise RuntimeError("V7.3 SAM projections are unavailable.")
            sam_logits = self._cosine_logits(
                self.sam_query(flat_sam_current),
                self.sam_key(flat_sam_reference),
            )
            sam_pair_valid = (
                flat_sam_current_valid[..., None]
                & flat_sam_reference_valid[:, None, :]
            )
            sam_logits = torch.where(
                sam_pair_valid,
                sam_logits,
                torch.zeros_like(sam_logits),
            )
        if uniform_sam:
            sam_logits = torch.zeros_like(sam_logits)

        query_valid = flat_current_valid
        key_valid = flat_reference_valid
        if self.architecture == "uniform_transport":
            logits = torch.zeros_like(geometry_logits)
        elif self.architecture == "geometry_transport":
            logits = geometry_logits
        elif self.architecture == "sam_transport":
            logits = sam_logits
            query_valid = query_valid & flat_sam_current_valid
            key_valid = key_valid & flat_sam_reference_valid
        else:
            # When SAM is absent, the sum reduces exactly to geometry-only
            # correspondence instead of inventing an identity match.
            logits = geometry_logits + sam_logits

        probability = _masked_transport_probability(
            logits,
            query_valid=query_valid,
            key_valid=key_valid,
        )
        matched = probability @ flat_reference_geometry
        point_residual = self.residual_encoder(
            torch.cat(
                [
                    flat_geometry,
                    matched,
                    flat_geometry - matched,
                    flat_geometry * matched,
                ],
                dim=-1,
            )
        )
        point_weight = query_valid.to(point_residual.dtype)
        # The final cached geometry channel is normalized point confidence.
        point_weight = point_weight * local_features[..., -1].reshape(
            -1, points
        ).float().clamp(0.0, 2.0)
        pooled = (point_residual * point_weight[..., None]).sum(dim=1)
        pooled = pooled / point_weight.sum(dim=1, keepdim=True).clamp_min(1e-8)
        active = query_valid.any(dim=-1) & key_valid.any(dim=-1)
        pooled = torch.where(active[..., None], pooled, torch.zeros_like(pooled))

        entropy, maximum = _transport_statistics(
            probability,
            query_valid=query_valid,
            key_valid=key_valid,
        )
        affinity_delta = torch.zeros_like(entropy)
        sam_used = (
            flat_sam_current_valid.any(dim=-1)
            & flat_sam_reference_valid.any(dim=-1)
        )
        if self.description.uses_sam:
            baseline_logits = (
                geometry_logits
                if self.description.uses_geometry_affinity
                else torch.zeros_like(geometry_logits)
            )
            baseline_probability = _masked_transport_probability(
                baseline_logits,
                query_valid=query_valid,
                key_valid=key_valid,
            )
            difference = (probability - baseline_probability).abs().sum(dim=-1)
            affinity_delta = (
                difference * query_valid.to(difference.dtype)
            ).sum(dim=-1) / query_valid.sum(dim=-1).clamp_min(1).to(
                difference.dtype
            )
            affinity_delta = torch.where(
                sam_used,
                affinity_delta,
                torch.zeros_like(affinity_delta),
            )

        shape = (batch, sequence, instances)
        return {
            "instance_evidence": pooled.reshape(*shape, hidden),
            "available": active.reshape(shape),
            "transport_entropy": entropy.reshape(shape),
            "transport_max": maximum.reshape(shape),
            "sam_affinity_delta": affinity_delta.reshape(shape),
            "sam_used": sam_used.reshape(shape),
            "memory_mature": reference_valid.any(dim=-1),
        }

    def _interpolate_sam_to_geometry(
        self,
        sam_encoded: torch.Tensor,
        sam_uv: torch.Tensor,
        sam_valid: torch.Tensor,
        geometry_uv: torch.Tensor,
        geometry_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sam_points = sam_encoded.shape[-2]
        geometry_points = geometry_uv.shape[-2]
        hidden = sam_encoded.shape[-1]
        flat_sam = sam_encoded.reshape(-1, sam_points, hidden)
        flat_sam_uv = sam_uv.reshape(-1, sam_points, 2)
        flat_sam_valid = sam_valid.reshape(-1, sam_points)
        flat_geometry_uv = geometry_uv.reshape(-1, geometry_points, 2)
        flat_geometry_valid = geometry_valid.reshape(-1, geometry_points)
        distance = (
            flat_geometry_uv[:, :, None, :] - flat_sam_uv[:, None, :, :]
        ).square().sum(dim=-1)
        logits = -distance / float(self.spatial_temperature)
        probability = _masked_transport_probability(
            logits,
            query_valid=flat_geometry_valid,
            key_valid=flat_sam_valid,
        )
        interpolated = probability @ flat_sam
        available = flat_geometry_valid & flat_sam_valid.any(dim=-1)[:, None]
        interpolated = torch.where(
            available[..., None],
            interpolated,
            torch.zeros_like(interpolated),
        )
        return (
            interpolated.reshape(*geometry_uv.shape[:-1], hidden),
            available.reshape(geometry_valid.shape),
        )

    def _cosine_logits(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> torch.Tensor:
        query = F.normalize(query.float(), dim=-1)
        key = F.normalize(key.float(), dim=-1)
        return (query @ key.transpose(-1, -2)) / float(
            self.affinity_temperature
        )


class V73FrozenCorrespondenceResidual(nn.Module):
    """Zero-initialized pose residual driven by transported local geometry."""

    def __init__(
        self,
        *,
        base_model: V7PoseFusion,
        architecture: str,
        sam_local_dim: int,
        geometry_local_dim: int,
        config: V6FusionConfig,
        memory_mode: str = "fixed_reference",
    ) -> None:
        super().__init__()
        if architecture not in V73_ARCHITECTURES:
            raise ValueError(f"Unknown V7.3 architecture: {architecture!r}.")
        if base_model.architecture != "l0_camera_only":
            raise ValueError("V7.3 requires a V7 L0 camera-only base model.")
        self.base_model = base_model
        self.architecture = architecture
        self.memory_mode = memory_mode
        self.config = config
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)
        self.base_model.eval()
        hidden = int(config.hidden_dim)
        self.matcher = SAMWeightedGeometryMatcher(
            architecture=architecture,
            sam_feature_dim=sam_local_dim,
            geometry_feature_dim=geometry_local_dim,
            config=config,
            memory_mode=memory_mode,
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
        uniform_sam: bool = False,
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
        matched = self.matcher(
            local_features=local_features,
            local_valid=local_valid,
            sam_local_features=sam_local_features,
            sam_local_uv=sam_local_uv,
            sam_local_valid=sam_local_valid,
            reference_index=reference_index,
            uniform_sam=uniform_sam,
            memory_write=(
                observed.bool()
                & identity_valid.bool()
                & (
                    quality[..., 0]
                    >= float(self.config.min_track_confidence)
                )
            ),
        )
        valid = state.valid & matched["available"]
        evidence = _weighted_instance_pool(
            matched["instance_evidence"], valid, state.reliability
        )
        active = valid.any(dim=2)
        evidence = torch.where(
            active[..., None], evidence, torch.zeros_like(evidence)
        )
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
        omega = torch.where(
            update_mask[..., None], omega, torch.zeros_like(omega)
        )
        rho = torch.where(update_mask[..., None], rho, torch.zeros_like(rho))
        base_h = homogeneous_world_to_camera(base_pose)
        composed = se3_exp(torch.cat([omega, rho], dim=-1)) @ base_h
        final = torch.where(update_mask[..., None, None], composed, base_h)
        return {
            "world_to_camera": final[..., :3, :4],
            "active_frames": update_mask,
            "usable_instance_mask": valid,
            "evidence": evidence,
            "fused_camera": fused,
            "base_world_to_camera": base_pose,
            **matched,
        }


def perturb_v73_inputs(
    batch: dict[str, torch.Tensor],
    variant: str,
) -> tuple[dict[str, torch.Tensor], bool]:
    """Perturb SAM/geometry content while preserving the common gate."""

    if variant not in V73_INPUT_VARIANTS:
        raise ValueError(f"Unknown V7.3 input variant: {variant!r}.")
    output = {name: value for name, value in batch.items()}
    uniform_sam = variant == "uniform_sam"
    if variant == "instance_off":
        output["observed"] = torch.zeros_like(batch["observed"])
        output["identity_valid"] = torch.zeros_like(batch["identity_valid"])
        output["identity_unknown"] = torch.zeros_like(batch["identity_unknown"])
        output["local_valid"] = torch.zeros_like(batch["local_valid"])
        output["sam_local_valid"] = torch.zeros_like(batch["sam_local_valid"])
    elif variant == "geometry_off":
        output["local_valid"] = torch.zeros_like(batch["local_valid"])
    elif variant == "sam_off":
        output["sam_local_valid"] = torch.zeros_like(batch["sam_local_valid"])
    elif variant == "wrong_sam_identity":
        for name in ("sam_local_features", "sam_local_uv", "sam_local_valid"):
            changed = batch[name].clone()
            changed[:, 1:] = torch.roll(batch[name][:, 1:], shifts=1, dims=2)
            output[name] = changed
    elif variant == "shuffle_sam_time":
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
    elif variant == "wrong_local_geometry":
        for name in ("local_features", "local_valid"):
            changed = batch[name].clone()
            changed[:, 1:] = torch.roll(batch[name][:, 1:], shifts=1, dims=2)
            output[name] = changed
    return output, uniform_sam


def _causal_previous_memory(
    values: torch.Tensor,
    valid: torch.Tensor,
    *,
    write_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expose only the latest observation strictly before each frame.

    The update is intentionally performed after appending the current memory.
    Consequently an instance birth has no Key/Value and cannot change that
    frame's pose; its first possible contribution is its next observation.
    """

    if values.ndim != 5 or valid.shape != values.shape[:-1]:
        raise ValueError("Causal instance memory expects [B,S,K,P,D]/[B,S,K,P].")
    if write_mask is not None and write_mask.shape != values.shape[:3]:
        raise ValueError("Causal memory write mask must have shape [B,S,K].")
    memory = torch.zeros_like(values[:, 0])
    memory_valid = torch.zeros_like(valid[:, 0])
    value_rows = []
    valid_rows = []
    for frame in range(values.shape[1]):
        value_rows.append(memory)
        valid_rows.append(memory_valid)
        current_valid = valid[:, frame]
        write = current_valid.any(dim=-1)
        if write_mask is not None:
            write = write & write_mask[:, frame].bool()
        memory = torch.where(
            write[..., None, None],
            values[:, frame],
            memory,
        )
        memory_valid = torch.where(
            write[..., None],
            current_valid,
            memory_valid,
        )
    return torch.stack(value_rows, dim=1), torch.stack(valid_rows, dim=1)


def _masked_transport_probability(
    logits: torch.Tensor,
    *,
    query_valid: torch.Tensor,
    key_valid: torch.Tensor,
) -> torch.Tensor:
    if logits.ndim != 3:
        raise ValueError("Transport logits must be [N,Q,K].")
    if query_valid.shape != logits.shape[:2]:
        raise ValueError("Transport query mask has the wrong shape.")
    if key_valid.shape != (logits.shape[0], logits.shape[2]):
        raise ValueError("Transport key mask has the wrong shape.")
    masked = logits.float().masked_fill(~key_valid[:, None, :], -1e4)
    probability = torch.softmax(masked, dim=-1)
    probability = probability * key_valid[:, None, :].to(probability.dtype)
    probability = probability / probability.sum(dim=-1, keepdim=True).clamp_min(
        1e-8
    )
    probability = probability * query_valid[..., None].to(probability.dtype)
    return probability.to(logits.dtype)


def _transport_statistics(
    probability: torch.Tensor,
    *,
    query_valid: torch.Tensor,
    key_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    safe = probability.float().clamp_min(1e-12)
    entropy = -(safe * safe.log()).sum(dim=-1)
    key_count = key_valid.sum(dim=-1).clamp_min(2).float()
    entropy = entropy / key_count.log()[:, None]
    maximum = probability.float().max(dim=-1).values
    weight = query_valid.to(entropy.dtype)
    denominator = weight.sum(dim=-1).clamp_min(1.0)
    entropy = (entropy * weight).sum(dim=-1) / denominator
    maximum = (maximum * weight).sum(dim=-1) / denominator
    return entropy.to(probability.dtype), maximum.to(probability.dtype)


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
