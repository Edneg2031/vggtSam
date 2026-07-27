"""V6 camera-only instance fusion and SE(3) pose correction.

This module is deliberately independent from the retained V4/V5 adapter.  It
uses frozen cached observations, replaces (rather than residually edits) the
camera feature presented to a small correction head, and never touches depth
or pointmap tokens.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

V6_VARIANTS = (
    "normal",
    "instance_off",
    "camera_off",
    "shuffle_time",
    "wrong_geometry",
    "appearance_only",
    "geometry_only",
)
V6_TRAINING_MODES = ("camera_only", "instance_only", "fusion")
V6_HEAD_COMPONENTS = ("se3", "rotation", "center", "translation")


@dataclass(frozen=True)
class V6FusionConfig:
    hidden_dim: int = 256
    num_heads: int = 8
    dropout: float = 0.0
    memory_momentum: float = 0.90
    min_track_confidence: float = 0.25
    unknown_reliability: float = 0.50
    max_rotation_degrees: float = 15.0
    max_translation_native: float = 0.50


class PersistentInstanceEncoder(nn.Module):
    """Encode current observations against trusted causal instance memory."""

    def __init__(
        self,
        *,
        appearance_dim: int,
        geometry_dim: int,
        config: V6FusionConfig,
    ) -> None:
        super().__init__()
        self.appearance_dim = int(appearance_dim)
        self.geometry_dim = int(geometry_dim)
        self.config = config
        # current, memory and difference for appearance and geometry; then
        # three quality values, OBSERVED, MATCH, UNKNOWN and memory age.
        input_dim = 3 * self.appearance_dim + 3 * self.geometry_dim + 7
        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
        )

    def forward(
        self,
        *,
        appearance: torch.Tensor,
        geometry: torch.Tensor,
        quality: torch.Tensor,
        observed: torch.Tensor,
        identity_valid: torch.Tensor,
        identity_unknown: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _validate_instance_inputs(
            appearance=appearance,
            geometry=geometry,
            quality=quality,
            observed=observed,
            identity_valid=identity_valid,
            identity_unknown=identity_unknown,
        )
        batch, sequence, instances = observed.shape
        app_memory = torch.zeros_like(appearance[:, 0])
        geo_memory = torch.zeros_like(geometry[:, 0])
        has_memory = torch.zeros(
            batch,
            instances,
            dtype=torch.bool,
            device=observed.device,
        )
        age = torch.zeros(
            batch,
            instances,
            dtype=appearance.dtype,
            device=appearance.device,
        )
        tokens = []
        valid_rows = []
        reliability_rows = []
        momentum = float(self.config.memory_momentum)

        for frame in range(sequence):
            current_app = appearance[:, frame]
            current_geo = geometry[:, frame]
            current_observed = observed[:, frame].bool()
            current_match = identity_valid[:, frame].bool()
            current_unknown = identity_unknown[:, frame].bool()
            track_ok = (
                quality[:, frame, :, 0]
                >= float(self.config.min_track_confidence)
            )
            associated_now = (
                current_observed
                & track_ok
                & (current_match | current_unknown)
            )
            # The trusted token remains available through a temporary miss.
            # A hard observed identity mismatch is excluded.
            usable = has_memory & (associated_now | ~current_observed)
            reliability = quality[:, frame, :, 0].clamp(0.0, 1.0)
            reliability = reliability * torch.where(
                current_match,
                torch.ones_like(reliability),
                torch.full_like(
                    reliability,
                    float(self.config.unknown_reliability),
                ),
            )
            reliability = torch.where(
                current_observed,
                reliability,
                torch.full_like(
                    reliability,
                    0.5 * float(self.config.unknown_reliability),
                ),
            )
            reliability = torch.where(
                usable,
                reliability,
                torch.zeros_like(reliability),
            )
            features = torch.cat(
                [
                    current_app,
                    app_memory,
                    current_app - app_memory,
                    current_geo,
                    geo_memory,
                    current_geo - geo_memory,
                    quality[:, frame],
                    current_observed.to(dtype=appearance.dtype)[..., None],
                    current_match.to(dtype=appearance.dtype)[..., None],
                    current_unknown.to(dtype=appearance.dtype)[..., None],
                    (torch.log1p(age) / 4.0)[..., None],
                ],
                dim=-1,
            )
            tokens.append(self.encoder(features))
            valid_rows.append(usable)
            reliability_rows.append(reliability)

            # Only a geometrically accepted identity may alter persistent
            # memory. UNKNOWN observations can be compared with memory but
            # cannot poison it.
            memory_write = current_observed & track_ok & current_match
            update = memory_write[..., None]
            first = update & (~has_memory)[..., None]
            app_candidate = momentum * app_memory + (1.0 - momentum) * current_app
            geo_candidate = momentum * geo_memory + (1.0 - momentum) * current_geo
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
            torch.stack(tokens, dim=1),
            torch.stack(valid_rows, dim=1),
            torch.stack(reliability_rows, dim=1),
        )


class V6CameraFusion(nn.Module):
    """Feature merger followed by a bounded pose-component head."""

    def __init__(
        self,
        *,
        camera_dim: int,
        appearance_dim: int,
        geometry_dim: int,
        config: V6FusionConfig,
        head_component: str = "se3",
    ) -> None:
        super().__init__()
        if config.hidden_dim % config.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        if head_component not in V6_HEAD_COMPONENTS:
            raise ValueError(f"Unknown V6 head component: {head_component!r}.")
        self.config = config
        self.head_component = head_component
        self.instance_encoder = PersistentInstanceEncoder(
            appearance_dim=appearance_dim,
            geometry_dim=geometry_dim,
            config=config,
        )
        self.camera_projector = nn.Sequential(
            nn.LayerNorm(camera_dim),
            nn.Linear(camera_dim, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
        )
        self.instance_norm = nn.LayerNorm(config.hidden_dim)
        self.cross_attention = nn.MultiheadAttention(
            config.hidden_dim,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.instance_query = nn.Parameter(
            torch.zeros(1, 1, config.hidden_dim)
        )
        nn.init.normal_(self.instance_query, std=0.02)
        # This is a feature replacement: the correction head sees only the
        # output of this merger, never ``camera + residual``.
        self.feature_merger = nn.Sequential(
            nn.LayerNorm(4 * config.hidden_dim),
            nn.Linear(4 * config.hidden_dim, 2 * config.hidden_dim),
            nn.GELU(),
            nn.Linear(2 * config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
        )
        self.se3_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(
                config.hidden_dim,
                6 if head_component == "se3" else 3,
            ),
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
        reference_index: int,
        mode: str = "fusion",
    ) -> dict[str, torch.Tensor]:
        if mode not in V6_TRAINING_MODES:
            raise ValueError(f"Unknown V6 training mode: {mode!r}.")
        if camera_hidden.ndim != 3:
            raise ValueError("camera_hidden must have shape [B,S,D].")
        if baseline_world_to_camera.shape != camera_hidden.shape[:2] + (3, 4):
            raise ValueError("baseline_world_to_camera must have shape [B,S,3,4].")
        camera = self.camera_projector(camera_hidden)
        if mode == "camera_only":
            attended = torch.zeros_like(camera)
            active = torch.ones(
                camera.shape[:2],
                dtype=torch.bool,
                device=camera.device,
            )
            camera_feature = camera
        else:
            instance, valid, reliability = self.instance_encoder(
                appearance=appearance,
                geometry=geometry,
                quality=quality,
                observed=observed,
                identity_valid=identity_valid,
                identity_unknown=identity_unknown,
            )
            query = (
                camera
                if mode == "fusion"
                else self.instance_query.expand(
                    camera.shape[0],
                    camera.shape[1],
                    -1,
                )
            )
            attended, active = self._cross_attend(
                query,
                instance,
                valid,
                reliability,
            )
            camera_feature = (
                camera if mode == "fusion" else torch.zeros_like(camera)
            )
        merged = self.feature_merger(
            torch.cat(
                [
                    camera_feature,
                    attended,
                    camera_feature * attended,
                    camera_feature - attended,
                ],
                dim=-1,
            )
        )
        raw_delta = self.se3_head(merged)
        update_mask = active.clone()
        update_mask[:, int(reference_index)] = False
        baseline_h = homogeneous_world_to_camera(baseline_world_to_camera)
        if self.head_component == "se3":
            omega = torch.tanh(raw_delta[..., :3]) * torch.deg2rad(
                raw_delta.new_tensor(float(self.config.max_rotation_degrees))
            )
            rho = torch.tanh(raw_delta[..., 3:]) * float(
                self.config.max_translation_native
            )
            omega = torch.where(
                update_mask[..., None], omega, torch.zeros_like(omega)
            )
            rho = torch.where(
                update_mask[..., None], rho, torch.zeros_like(rho)
            )
            correction = se3_exp(torch.cat([omega, rho], dim=-1))
            composed = correction @ baseline_h
            center_delta = torch.zeros_like(rho)
        elif self.head_component == "rotation":
            omega = torch.tanh(raw_delta) * torch.deg2rad(
                raw_delta.new_tensor(float(self.config.max_rotation_degrees))
            )
            omega = torch.where(
                update_mask[..., None], omega, torch.zeros_like(omega)
            )
            rho = torch.zeros_like(omega)
            center_delta = torch.zeros_like(omega)
            rotation_correction = torch.eye(
                4,
                dtype=omega.dtype,
                device=omega.device,
            ).expand(omega.shape[:-1] + (4, 4)).clone()
            rotation_correction[..., :3, :3] = so3_exp(omega)
            composed = rotation_correction @ baseline_h
        elif self.head_component == "center":
            center_delta = torch.tanh(raw_delta) * float(
                self.config.max_translation_native
            )
            center_delta = torch.where(
                update_mask[..., None],
                center_delta,
                torch.zeros_like(center_delta),
            )
            omega = torch.zeros_like(center_delta)
            rho = torch.zeros_like(center_delta)
            rotation = baseline_world_to_camera[..., :3, :3]
            translation = (
                baseline_world_to_camera[..., :3, 3:]
                - rotation @ center_delta[..., None]
            )
            composed = homogeneous_world_to_camera(
                torch.cat([rotation, translation], dim=-1)
            )
        else:
            local_translation = torch.tanh(raw_delta) * float(
                self.config.max_translation_native
            )
            local_translation = torch.where(
                update_mask[..., None],
                local_translation,
                torch.zeros_like(local_translation),
            )
            omega = torch.zeros_like(local_translation)
            rho = local_translation
            rotation = baseline_world_to_camera[..., :3, :3]
            translation = (
                baseline_world_to_camera[..., :3, 3:]
                + local_translation[..., None]
            )
            center_delta = -(
                rotation.transpose(-1, -2)
                @ local_translation[..., None]
            ).squeeze(-1)
            composed = homogeneous_world_to_camera(
                torch.cat([rotation, translation], dim=-1)
            )
        # Preserve inactive and reference frames bit-for-bit.
        final_h = torch.where(
            update_mask[..., None, None],
            composed,
            baseline_h,
        )
        return {
            "world_to_camera": final_h[..., :3, :4],
            "twist": torch.cat([omega, rho], dim=-1),
            "center_delta": center_delta,
            "active_frames": update_mask,
            "fused_camera": merged,
        }

    def _cross_attend(
        self,
        camera: torch.Tensor,
        instance: torch.Tensor,
        valid: torch.Tensor,
        reliability: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, sequence, hidden = camera.shape
        flat_camera = camera.reshape(batch * sequence, 1, hidden)
        flat_instance = instance.reshape(
            batch * sequence,
            instance.shape[2],
            hidden,
        )
        flat_valid = valid.reshape(batch * sequence, valid.shape[2])
        flat_reliability = reliability.reshape_as(flat_valid).to(camera.dtype)
        active = flat_valid.any(dim=-1)
        output = torch.zeros_like(flat_camera)
        if bool(active.any()):
            indices = torch.nonzero(active, as_tuple=False).flatten()
            query = flat_camera.index_select(0, indices)
            weights = flat_reliability.index_select(0, indices)[..., None]
            key_value = self.instance_norm(
                flat_instance.index_select(0, indices)
            ) * weights
            key_padding_mask = ~flat_valid.index_select(0, indices)
            attended, _ = self.cross_attention(
                query,
                key_value,
                key_value,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            output.index_copy_(0, indices, attended)
        return output.reshape(batch, sequence, hidden), active.reshape(batch, sequence)
def perturb_instance_inputs(batch: dict[str, torch.Tensor], variant: str) -> dict[str, torch.Tensor]:
    """Return one deterministic same-checkpoint V6 ablation input."""

    if variant not in V6_VARIANTS:
        raise ValueError(f"Unknown V6 variant: {variant!r}.")
    fields = {
        name: batch[name]
        for name in (
            "camera_hidden",
            "appearance",
            "pose_geometry",
            "quality",
            "observed",
            "identity_valid",
            "identity_unknown",
        )
    }
    if variant == "instance_off":
        fields["observed"] = torch.zeros_like(fields["observed"])
        fields["identity_valid"] = torch.zeros_like(fields["identity_valid"])
        fields["identity_unknown"] = torch.zeros_like(fields["identity_unknown"])
    elif variant == "camera_off":
        fields["camera_hidden"] = torch.zeros_like(fields["camera_hidden"])
    elif variant == "shuffle_time":
        sequence = fields["appearance"].shape[1]
        permutation = torch.roll(
            torch.arange(sequence, device=fields["appearance"].device),
            shifts=1,
        )
        # Keep identity/visibility gates unchanged so any degradation comes
        # from wrong token content, not a different number of active frames.
        fields["appearance"] = fields["appearance"].index_select(
            1,
            permutation,
        )
        fields["pose_geometry"] = fields["pose_geometry"].index_select(
            1,
            permutation,
        )
    elif variant == "wrong_geometry":
        sequence = fields["pose_geometry"].shape[1]
        permutation = torch.roll(
            torch.arange(sequence, device=fields["pose_geometry"].device),
            shifts=1,
        )
        fields["pose_geometry"] = fields["pose_geometry"].index_select(
            1,
            permutation,
        )
    elif variant == "appearance_only":
        fields["pose_geometry"] = torch.zeros_like(fields["pose_geometry"])
    elif variant == "geometry_only":
        fields["appearance"] = torch.zeros_like(fields["appearance"])
    return fields


def homogeneous_world_to_camera(world_to_camera: torch.Tensor) -> torch.Tensor:
    if world_to_camera.shape[-2:] != (3, 4):
        raise ValueError("world_to_camera must end in [3,4].")
    result = torch.eye(
        4,
        dtype=world_to_camera.dtype,
        device=world_to_camera.device,
    ).expand(world_to_camera.shape[:-2] + (4, 4)).clone()
    result[..., :3, :4] = world_to_camera
    return result


def so3_exp(omega: torch.Tensor) -> torch.Tensor:
    angle = torch.linalg.vector_norm(omega, dim=-1, keepdim=True)
    x, y, z = omega.unbind(dim=-1)
    zero = torch.zeros_like(x)
    skew = torch.stack(
        [zero, -z, y, z, zero, -x, -y, x, zero],
        dim=-1,
    ).reshape(omega.shape[:-1] + (3, 3))
    identity = torch.eye(
        3,
        dtype=omega.dtype,
        device=omega.device,
    ).expand(omega.shape[:-1] + (3, 3))
    angle_squared = angle.square()
    coefficient_a = torch.sinc(angle / torch.pi)
    coefficient_b = torch.where(
        angle < 1e-4,
        0.5 - angle_squared / 24.0 + angle_squared.square() / 720.0,
        (1.0 - torch.cos(angle)) / angle_squared.clamp_min(1e-12),
    )
    return (
        identity
        + coefficient_a[..., None] * skew
        + coefficient_b[..., None] * (skew @ skew)
    )


def se3_exp(twist: torch.Tensor) -> torch.Tensor:
    """Exponential map for twists stored as ``[omega, rho]``."""

    if twist.shape[-1] != 6:
        raise ValueError("twist must end in six values [omega,rho].")
    omega, rho = twist[..., :3], twist[..., 3:]
    rotation = so3_exp(omega)
    angle = torch.linalg.vector_norm(omega, dim=-1, keepdim=True)
    angle_squared = angle.square()
    x, y, z = omega.unbind(dim=-1)
    zero = torch.zeros_like(x)
    skew = torch.stack(
        [zero, -z, y, z, zero, -x, -y, x, zero],
        dim=-1,
    ).reshape(omega.shape[:-1] + (3, 3))
    identity = torch.eye(
        3,
        dtype=twist.dtype,
        device=twist.device,
    ).expand(twist.shape[:-1] + (3, 3))
    coefficient_b = torch.where(
        angle < 1e-4,
        0.5 - angle_squared / 24.0 + angle_squared.square() / 720.0,
        (1.0 - torch.cos(angle)) / angle_squared.clamp_min(1e-12),
    )
    coefficient_c = torch.where(
        angle < 1e-4,
        1.0 / 6.0 - angle_squared / 120.0 + angle_squared.square() / 5040.0,
        (angle - torch.sin(angle))
        / (angle_squared * angle).clamp_min(1e-12),
    )
    left_jacobian = (
        identity
        + coefficient_b[..., None] * skew
        + coefficient_c[..., None] * (skew @ skew)
    )
    translation = (left_jacobian @ rho[..., None]).squeeze(-1)
    output = torch.eye(
        4,
        dtype=twist.dtype,
        device=twist.device,
    ).expand(twist.shape[:-1] + (4, 4)).clone()
    output[..., :3, :3] = rotation
    output[..., :3, 3] = translation
    return output


def _validate_instance_inputs(
    *,
    appearance: torch.Tensor,
    geometry: torch.Tensor,
    quality: torch.Tensor,
    observed: torch.Tensor,
    identity_valid: torch.Tensor,
    identity_unknown: torch.Tensor,
) -> None:
    if appearance.ndim != 4 or geometry.ndim != 4:
        raise ValueError("appearance and geometry must have shape [B,S,K,D].")
    leading = appearance.shape[:3]
    if geometry.shape[:3] != leading:
        raise ValueError("appearance and geometry leading dimensions disagree.")
    if quality.shape != leading + (3,):
        raise ValueError("quality must have shape [B,S,K,3].")
    for name, value in (
        ("observed", observed),
        ("identity_valid", identity_valid),
        ("identity_unknown", identity_unknown),
    ):
        if value.shape != leading:
            raise ValueError(f"{name} must have shape [B,S,K].")
