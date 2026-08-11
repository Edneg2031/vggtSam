"""Causal dynamic-instance V0 pose candidates.

SAM3.1 contributes masks, persistent identities and slot lifecycle.  The pose
branch deliberately does not consume SAM appearance tokens.  The accepted V0
pose is a V7.1-style camera L0 conditioned on the raw relative StreamVGGT pose;
the V7.4-style geometry-only transport is retained as a separately scored
candidate and never silently replaces L0.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


V0_CAMERA_INPUT_MODES = (
    "camera_pose",
    "pose_only",
    "camera_only",
    "time_only",
)


@dataclass(frozen=True)
class BaselineModelConfig:
    hidden_dim: int = 256
    min_track_confidence: float = 0.50
    min_geometry_confidence: float = 0.20
    min_static_score: float = 0.20
    max_rotation_degrees: float = 15.0
    max_translation_native: float = 0.50
    affinity_temperature: float = 0.10


def eligible_pose_instances(
    *,
    quality: torch.Tensor,
    observed: torch.Tensor,
    identity_valid: torch.Tensor,
    config: BaselineModelConfig,
) -> torch.Tensor:
    """Return the deployable mature-memory write/use gate.

    Maturity itself is enforced by the strictly-previous geometry memory.  A
    birth observation may be written after the frame, but cannot be read by
    the pose update of that same frame.
    """

    if quality.ndim != 4 or quality.shape[-1] != 3:
        raise ValueError("quality must be [B,S,K,3].")
    if observed.shape != quality.shape[:3]:
        raise ValueError("observed must be [B,S,K].")
    if identity_valid.shape != observed.shape:
        raise ValueError("identity_valid must match observed.")
    return (
        observed.bool()
        & identity_valid.bool()
        & quality[..., 0].ge(float(config.min_track_confidence))
        & quality[..., 1].ge(float(config.min_geometry_confidence))
        & quality[..., 2].ge(float(config.min_static_score))
    )


class CameraPoseBaseline(nn.Module):
    """V7.1-style L0 with a deployable raw-relative-pose conditioning path.

    The original V7.1 head used only ``camera_hidden``.  On the retained
    dynamic cache its zero-initialized output is an exact stationary point
    (all parameter gradients are zero).  The raw StreamVGGT pose is already
    available at inference time and provides a gauge-relative per-frame signal
    without reading SAM, GT, or future frames.
    """

    def __init__(
        self,
        *,
        camera_dim: int,
        config: BaselineModelConfig,
        input_mode: str = "camera_pose",
    ) -> None:
        super().__init__()
        if input_mode not in V0_CAMERA_INPUT_MODES:
            raise ValueError(f"Unknown V0 camera input mode: {input_mode!r}.")
        self.config = config
        self.input_mode = input_mode
        hidden = int(config.hidden_dim)
        self.camera_projector = _mlp(int(camera_dim) + 12, hidden)
        self.feature_merger = _mlp(4 * hidden, hidden)
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
        reference_index: int,
    ) -> dict[str, torch.Tensor]:
        if camera_hidden.ndim != 3:
            raise ValueError("camera_hidden must be [B,S,C].")
        if baseline_world_to_camera.shape != (*camera_hidden.shape[:2], 3, 4):
            raise ValueError("baseline pose must be [B,S,3,4].")
        relative_pose = relative_pose_to_reference(
            baseline_world_to_camera.float(), reference_index=reference_index
        )
        camera_input = camera_hidden.float()
        pose_input = relative_pose.reshape(*camera_hidden.shape[:2], 12)
        if self.input_mode == "pose_only":
            camera_input = torch.zeros_like(camera_input)
        elif self.input_mode == "camera_only":
            pose_input = torch.zeros_like(pose_input)
        elif self.input_mode == "time_only":
            camera_input = torch.zeros_like(camera_input)
            pose_input = torch.zeros_like(pose_input)
            position = torch.arange(
                camera_hidden.shape[1],
                dtype=pose_input.dtype,
                device=pose_input.device,
            )
            pose_input[..., 0] = position[None]
        camera = self.camera_projector(
            torch.cat(
                [
                    camera_input,
                    pose_input,
                ],
                dim=-1,
            )
        )
        evidence = torch.zeros_like(camera)
        fused = self.feature_merger(
            torch.cat(
                [camera, evidence, camera * evidence, camera - evidence],
                dim=-1,
            )
        )
        raw_delta = self.se3_head(fused)
        update = torch.ones(
            camera.shape[:2], dtype=torch.bool, device=camera.device
        )
        update[:, int(reference_index)] = False
        pose = _bounded_pose_update(
            baseline_world_to_camera,
            raw_delta,
            update,
            self.config,
        )
        return {
            "world_to_camera": pose,
            "active_frames": update,
            "fused_camera": fused,
        }


class CausalGeometryTransport(nn.Module):
    """Transport latest reliable StreamVGGT geometry with cosine affinity."""

    def __init__(
        self,
        *,
        geometry_dim: int,
        config: BaselineModelConfig,
    ) -> None:
        super().__init__()
        hidden = int(config.hidden_dim)
        self.temperature = float(config.affinity_temperature)
        self.geometry_encoder = _mlp(int(geometry_dim), hidden)
        self.query = nn.Linear(hidden, hidden, bias=False)
        self.key = nn.Linear(hidden, hidden, bias=False)
        self.evidence_encoder = _mlp(4 * hidden, hidden)

    def forward(
        self,
        *,
        local_features: torch.Tensor,
        local_valid: torch.Tensor,
        memory_write: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if local_features.ndim != 5:
            raise ValueError("local_features must be [B,S,K,P,F].")
        if local_valid.shape != local_features.shape[:-1]:
            raise ValueError("local_valid must be [B,S,K,P].")
        if memory_write.shape != local_features.shape[:3]:
            raise ValueError("memory_write must be [B,S,K].")

        encoded = self.geometry_encoder(local_features.float())
        previous, previous_valid = causal_previous_memory(
            encoded,
            local_valid.bool(),
            write_mask=memory_write.bool(),
        )
        points = int(encoded.shape[-2])
        hidden = int(encoded.shape[-1])
        current_flat = encoded.reshape(-1, points, hidden)
        previous_flat = previous.reshape(-1, points, hidden)
        query_valid = local_valid.reshape(-1, points).bool()
        key_valid = previous_valid.reshape(-1, points).bool()
        logits = (
            F.normalize(self.query(current_flat).float(), dim=-1)
            @ F.normalize(self.key(previous_flat).float(), dim=-1).transpose(-1, -2)
        ) / max(self.temperature, 1e-6)
        probability = _masked_probability(
            logits,
            query_valid=query_valid,
            key_valid=key_valid,
        )
        matched = probability @ previous_flat
        point_evidence = self.evidence_encoder(
            torch.cat(
                [
                    current_flat,
                    matched,
                    current_flat - matched,
                    current_flat * matched,
                ],
                dim=-1,
            )
        )
        # Last local channel is normalized StreamVGGT confidence.
        weight = query_valid.to(point_evidence.dtype)
        weight = weight * local_features[..., -1].reshape(-1, points).float().clamp(0, 2)
        pooled = (point_evidence * weight[..., None]).sum(dim=1)
        pooled = pooled / weight.sum(dim=1, keepdim=True).clamp_min(1e-8)
        available = query_valid.any(dim=-1) & key_valid.any(dim=-1)
        pooled = torch.where(available[..., None], pooled, torch.zeros_like(pooled))
        leading = local_features.shape[:3]
        return {
            "instance_evidence": pooled.reshape(*leading, hidden),
            "available": available.reshape(leading),
            "memory_mature": previous_valid.any(dim=-1),
        }


class DynamicInstanceGeometryRefiner(nn.Module):
    """V7.4-style geometry-only transport residual over frozen V7.1 L0."""

    def __init__(
        self,
        *,
        base_model: CameraPoseBaseline,
        geometry_dim: int,
        config: BaselineModelConfig,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.config = config
        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)
        self.base_model.eval()
        hidden = int(config.hidden_dim)
        self.transport = CausalGeometryTransport(
            geometry_dim=int(geometry_dim), config=config
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
        quality: torch.Tensor,
        observed: torch.Tensor,
        identity_valid: torch.Tensor,
        local_features: torch.Tensor,
        local_valid: torch.Tensor,
        reference_index: int,
    ) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            base = self.base_model(
                camera_hidden=camera_hidden,
                baseline_world_to_camera=baseline_world_to_camera,
                reference_index=reference_index,
            )
        eligible = eligible_pose_instances(
            quality=quality,
            observed=observed,
            identity_valid=identity_valid,
            config=self.config,
        )
        matched = self.transport(
            local_features=local_features,
            local_valid=local_valid,
            memory_write=eligible,
        )
        usable = eligible & matched["available"]
        reliability = quality.float().clamp(0, 1).prod(dim=-1)
        weight = torch.where(usable, reliability, torch.zeros_like(reliability))
        evidence = (matched["instance_evidence"] * weight[..., None]).sum(dim=2)
        evidence = evidence / weight.sum(dim=2, keepdim=True).clamp_min(1e-8)
        active = usable.any(dim=2)
        evidence = torch.where(active[..., None], evidence, torch.zeros_like(evidence))
        camera = base["fused_camera"].detach()
        fused = self.camera_merger(
            torch.cat([camera, evidence, camera * evidence, camera - evidence], dim=-1)
        )
        raw_delta = self.se3_head(fused)
        update = active.clone()
        update[:, int(reference_index)] = False
        pose = _bounded_pose_update(
            base["world_to_camera"].detach(),
            raw_delta,
            update,
            self.config,
        )
        return {
            "world_to_camera": pose,
            "active_frames": update,
            "eligible_instances": eligible,
            "usable_instance_mask": usable,
            "memory_mature": matched["memory_mature"],
            "base_world_to_camera": base["world_to_camera"].detach(),
            "evidence": evidence,
        }


def causal_previous_memory(
    values: torch.Tensor,
    valid: torch.Tensor,
    *,
    write_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expose the latest accepted value strictly before each frame."""

    if values.ndim != 5 or valid.shape != values.shape[:-1]:
        raise ValueError("memory values/valid must be [B,S,K,P,D]/[B,S,K,P].")
    if write_mask.shape != values.shape[:3]:
        raise ValueError("write_mask must be [B,S,K].")
    memory = torch.zeros_like(values[:, 0])
    memory_valid = torch.zeros_like(valid[:, 0])
    value_rows = []
    valid_rows = []
    for frame in range(values.shape[1]):
        value_rows.append(memory)
        valid_rows.append(memory_valid)
        write = write_mask[:, frame].bool() & valid[:, frame].any(dim=-1)
        memory = torch.where(write[..., None, None], values[:, frame], memory)
        memory_valid = torch.where(write[..., None], valid[:, frame], memory_valid)
    return torch.stack(value_rows, dim=1), torch.stack(valid_rows, dim=1)


def _bounded_pose_update(
    base_pose: torch.Tensor,
    raw_delta: torch.Tensor,
    update_mask: torch.Tensor,
    config: BaselineModelConfig,
) -> torch.Tensor:
    omega = torch.tanh(raw_delta[..., :3]) * torch.deg2rad(
        raw_delta.new_tensor(float(config.max_rotation_degrees))
    )
    rho = torch.tanh(raw_delta[..., 3:]) * float(config.max_translation_native)
    omega = torch.where(update_mask[..., None], omega, torch.zeros_like(omega))
    rho = torch.where(update_mask[..., None], rho, torch.zeros_like(rho))
    composed = se3_exp(torch.cat([omega, rho], dim=-1)) @ homogeneous_pose(base_pose)
    output = torch.where(
        update_mask[..., None, None], composed, homogeneous_pose(base_pose)
    )
    return output[..., :3, :4]


def homogeneous_pose(world_to_camera: torch.Tensor) -> torch.Tensor:
    if world_to_camera.shape[-2:] != (3, 4):
        raise ValueError("world_to_camera must end in [3,4].")
    output = torch.eye(
        4, dtype=world_to_camera.dtype, device=world_to_camera.device
    ).expand(*world_to_camera.shape[:-2], 4, 4).clone()
    output[..., :3, :4] = world_to_camera
    return output


def relative_pose_to_reference(
    world_to_camera: torch.Tensor, *, reference_index: int
) -> torch.Tensor:
    """Return current-from-reference transforms using only frozen raw poses."""

    pose = homogeneous_pose(world_to_camera)
    reference = pose[:, int(reference_index)]
    rotation = reference[..., :3, :3]
    translation = reference[..., :3, 3]
    inverse = torch.eye(
        4, dtype=pose.dtype, device=pose.device
    ).expand(reference.shape).clone()
    inverse[..., :3, :3] = rotation.transpose(-1, -2)
    inverse[..., :3, 3] = -(
        rotation.transpose(-1, -2) @ translation[..., None]
    ).squeeze(-1)
    return (pose @ inverse[:, None])[..., :3, :4]


def se3_exp(twist: torch.Tensor) -> torch.Tensor:
    """Original V6/V7 exponential map for twists stored as ``[omega,rho]``.

    In particular, the SO(3) coefficient uses ``torch.sinc`` at the origin.
    That is the numerically stable implementation used by the successful
    V7.1/V7.4 runs and avoids changing the zero-initialized training path.
    """

    if twist.shape[-1] != 6:
        raise ValueError("SE(3) twist must end in six values.")
    omega = twist[..., :3]
    rho = twist[..., 3:]
    rotation = so3_exp(omega)
    angle = torch.linalg.vector_norm(omega, dim=-1, keepdim=True)
    skew = _skew(omega)
    angle2 = angle.square()
    identity = torch.eye(3, dtype=twist.dtype, device=twist.device).expand(
        *twist.shape[:-1], 3, 3
    )
    b = torch.where(
        angle < 1e-4,
        0.5 - angle2 / 24.0 + angle2.square() / 720.0,
        (1.0 - torch.cos(angle)) / angle2.clamp_min(1e-12),
    )
    c = torch.where(
        angle < 1e-4,
        1.0 / 6.0 - angle2 / 120.0 + angle2.square() / 5040.0,
        (angle - torch.sin(angle)) / (angle2 * angle).clamp_min(1e-12),
    )
    jacobian = identity + b[..., None] * skew + c[..., None] * (skew @ skew)
    translation = (jacobian @ rho[..., None]).squeeze(-1)
    output = torch.eye(4, dtype=twist.dtype, device=twist.device).expand(
        *twist.shape[:-1], 4, 4
    ).clone()
    output[..., :3, :3] = rotation
    output[..., :3, 3] = translation
    return output


def so3_exp(omega: torch.Tensor) -> torch.Tensor:
    """Stable Rodrigues map copied from the successful V6/V7 implementation."""

    if omega.shape[-1] != 3:
        raise ValueError("SO(3) vector must end in three values.")
    angle = torch.linalg.vector_norm(omega, dim=-1, keepdim=True)
    angle2 = angle.square()
    skew = _skew(omega)
    identity = torch.eye(3, dtype=omega.dtype, device=omega.device).expand(
        *omega.shape[:-1], 3, 3
    )
    coefficient_a = torch.sinc(angle / torch.pi)
    coefficient_b = torch.where(
        angle < 1e-4,
        0.5 - angle2 / 24.0 + angle2.square() / 720.0,
        (1.0 - torch.cos(angle)) / angle2.clamp_min(1e-12),
    )
    return (
        identity
        + coefficient_a[..., None] * skew
        + coefficient_b[..., None] * (skew @ skew)
    )


def _skew(vector: torch.Tensor) -> torch.Tensor:
    x, y, z = vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        [zero, -z, y, z, zero, -x, -y, x, zero], dim=-1
    ).reshape(*vector.shape[:-1], 3, 3)


def _masked_probability(
    logits: torch.Tensor,
    *,
    query_valid: torch.Tensor,
    key_valid: torch.Tensor,
) -> torch.Tensor:
    masked = logits.float().masked_fill(~key_valid[:, None, :], -1e4)
    probability = torch.softmax(masked, dim=-1)
    probability = probability * key_valid[:, None, :].to(probability.dtype)
    probability = probability / probability.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return probability * query_valid[..., None].to(probability.dtype)


def _mlp(input_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(int(input_dim)),
        nn.Linear(int(input_dim), int(hidden_dim)),
        nn.GELU(),
        nn.Linear(int(hidden_dim), int(hidden_dim)),
        nn.LayerNorm(int(hidden_dim)),
    )
