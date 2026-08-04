"""V8 supervised causal correspondence built on the retained V7 matcher.

V8 deliberately separates two questions that V7 could not distinguish:

1. do frozen SAM3.1 descriptors improve held-out local 3D correspondence;
2. does a frozen correspondence model improve a constrained pose residual.

GT is used only to supervise/evaluate correspondence during training.  The
inference model still consumes frozen StreamVGGT geometry and SAM3.1 tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

from .v73_correspondence_fusion import V73FrozenCorrespondenceResidual


V80_VALUE_MODES = ("geometry_only", "geometry_sam_dual")
V80_TRAINING_STAGES = ("matching", "pose")


@dataclass(frozen=True)
class V80MatchingConfig:
    """Conservative labels for sparse cross-view object samples."""

    max_distance: float = 0.10
    temperature: float = 0.025
    require_mutual_nearest: bool = True

    def validate(self) -> None:
        if self.max_distance <= 0.0:
            raise ValueError("V8 match max_distance must be positive.")
        if self.temperature <= 0.0:
            raise ValueError("V8 match temperature must be positive.")


@dataclass
class V80MatchingResult:
    loss: torch.Tensor
    supervised_queries: torch.Tensor
    candidate_queries: torch.Tensor
    top1_exact: torch.Tensor
    top1_within_radius: torch.Tensor
    expected_distance: torch.Tensor
    valid_key_probability_mass: torch.Tensor
    target_candidates: torch.Tensor
    active_frames: torch.Tensor

    def detached_metrics(self) -> dict[str, float]:
        supervised = float(self.supervised_queries.detach().cpu())
        candidates = float(self.candidate_queries.detach().cpu())
        denominator = max(supervised, 1.0)
        return {
            "match_loss": float(self.loss.detach().cpu()),
            "supervised_queries": supervised,
            "candidate_queries": candidates,
            "match_coverage": supervised / max(candidates, 1.0),
            "top1_exact_accuracy": float(self.top1_exact.detach().cpu())
            / denominator,
            "top1_within_radius": float(
                self.top1_within_radius.detach().cpu()
            )
            / denominator,
            "expected_gt_distance": float(
                self.expected_distance.detach().cpu()
            )
            / denominator,
            "valid_key_probability_mass": float(
                self.valid_key_probability_mass.detach().cpu()
            )
            / denominator,
            "mean_target_candidates": float(
                self.target_candidates.detach().cpu()
            )
            / denominator,
            "match_active_frames": int(self.active_frames.detach().sum().cpu()),
        }


class V80SupervisedCorrespondenceResidual(V73FrozenCorrespondenceResidual):
    """Expose correspondence tensors and remove the camera-feature shortcut."""

    def __init__(
        self,
        *,
        base_model,
        architecture: str,
        sam_local_dim: int,
        geometry_local_dim: int,
        config,
        value_mode: str = "geometry_only",
        pose_input_mode: str = "evidence_only",
    ) -> None:
        if value_mode not in V80_VALUE_MODES:
            raise ValueError(f"Unknown V8 Value mode: {value_mode!r}.")
        super().__init__(
            base_model=base_model,
            architecture=architecture,
            sam_local_dim=sam_local_dim,
            geometry_local_dim=geometry_local_dim,
            config=config,
            memory_mode="causal_last_observation",
            value_mode=value_mode,
            pose_input_mode=pose_input_mode,
            return_transport_details=True,
            separate_geometry_value_encoder=True,
        )
        self.value_mode = value_mode


def configure_v80_training_stage(
    model: V80SupervisedCorrespondenceResidual,
    stage: str,
) -> list[nn.Parameter]:
    """Select parameters without allowing the pose head to train the matcher."""

    if stage not in V80_TRAINING_STAGES:
        raise ValueError(f"Unknown V8 training stage: {stage!r}.")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if stage == "matching":
        modules: Iterable[nn.Module | None] = (
            model.matcher.geometry_encoder,
            model.matcher.geometry_query,
            model.matcher.geometry_key,
            model.matcher.sam_encoder,
            model.matcher.sam_query,
            model.matcher.sam_key,
        )
    else:
        modules = (
            model.matcher.geometry_value_encoder,
            model.matcher.residual_encoder,
            model.camera_merger,
            model.se3_head,
        )
    for module in modules:
        if module is None:
            continue
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError(f"V8 stage={stage} selected no trainable parameters.")
    return parameters


def sample_gt_world_at_local_tokens(
    *,
    local_features: torch.Tensor,
    local_valid: torch.Tensor,
    target_world_points: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample mesh-rasterized GT world points at cached geometry-token UVs."""

    if local_features.ndim != 5 or local_valid.shape != local_features.shape[:-1]:
        raise ValueError("V8 local geometry must be [B,S,K,P,F]/[B,S,K,P].")
    if local_features.shape[-1] < 5:
        raise ValueError("V8 local geometry lacks normalized UV coordinates.")
    target = target_world_points
    if target.ndim == 4:
        target = target.unsqueeze(0)
    if target.ndim != 5 or target.shape[-1] != 3:
        raise ValueError("V8 target_world_points must be [B,S,H,W,3] or [S,H,W,3].")
    if target.shape[0] == 1 and local_features.shape[0] != 1:
        target = target.expand(local_features.shape[0], -1, -1, -1, -1)
    if target.shape[:2] != local_features.shape[:2]:
        raise ValueError("V8 target pointmap and local-token sequence shapes differ.")

    height, width = target.shape[2:4]
    uv = local_features[..., 3:5].float()
    x = (((uv[..., 0] + 1.0) * 0.5) * max(width - 1, 1)).round().long()
    y = (((uv[..., 1] + 1.0) * 0.5) * max(height - 1, 1)).round().long()
    x = x.clamp(0, width - 1)
    y = y.clamp(0, height - 1)
    batch = torch.arange(target.shape[0], device=target.device)[:, None, None, None]
    sequence = torch.arange(target.shape[1], device=target.device)[None, :, None, None]
    sampled = target[batch, sequence, y, x].float()
    valid = local_valid.bool() & torch.isfinite(sampled).all(dim=-1)
    sampled = torch.where(valid[..., None], sampled, torch.zeros_like(sampled))
    return sampled, valid


def causal_reference_targets(
    values: torch.Tensor,
    target_valid: torch.Tensor,
    *,
    source_valid: torch.Tensor,
    memory_write: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replay the matcher memory while preserving invalid GT overwrites."""

    if values.ndim != 5 or values.shape[-1] != 3:
        raise ValueError("V8 target memory values must be [B,S,K,P,3].")
    expected = values.shape[:-1]
    if target_valid.shape != expected or source_valid.shape != expected:
        raise ValueError("V8 target/source validity shapes disagree.")
    if memory_write.shape != values.shape[:3]:
        raise ValueError("V8 memory_write must be [B,S,K].")
    memory = torch.zeros_like(values[:, 0])
    memory_valid = torch.zeros_like(target_valid[:, 0])
    rows = []
    valid_rows = []
    for frame in range(values.shape[1]):
        rows.append(memory)
        valid_rows.append(memory_valid)
        write = source_valid[:, frame].any(dim=-1) & memory_write[:, frame].bool()
        memory = torch.where(write[..., None, None], values[:, frame], memory)
        memory_valid = torch.where(
            write[..., None], target_valid[:, frame], memory_valid
        )
    return torch.stack(rows, dim=1), torch.stack(valid_rows, dim=1)


def compute_v80_matching_loss(
    output: dict[str, torch.Tensor],
    *,
    current_gt_world: torch.Tensor,
    current_gt_valid: torch.Tensor,
    source_local_valid: torch.Tensor,
    memory_write: torch.Tensor,
    config: V80MatchingConfig,
    sequence_indices: list[int] | tuple[int, ...] | None = None,
) -> V80MatchingResult:
    """Supervise p_ij with conservative causal GT-world correspondences."""

    config.validate()
    probability = output["transport_probability"].float()
    query_valid = output["transport_query_valid"].bool()
    key_valid = output["transport_key_valid"].bool()
    if probability.ndim != 6:
        raise ValueError("V8 probability must be [B,S,K,Q,R].")
    if query_valid.shape != probability.shape[:-1]:
        raise ValueError("V8 query validity has the wrong shape.")
    if key_valid.shape != (*probability.shape[:3], probability.shape[-1]):
        raise ValueError("V8 key validity has the wrong shape.")
    reference_gt, reference_gt_valid = causal_reference_targets(
        current_gt_world,
        current_gt_valid,
        source_valid=source_local_valid,
        memory_write=memory_write,
    )
    if current_gt_world.shape[:-1] != query_valid.shape:
        raise ValueError("V8 current GT token shape disagrees with probability.")

    pair_valid = (
        query_valid[..., :, None]
        & key_valid[..., None, :]
        & current_gt_valid[..., :, None]
        & reference_gt_valid[..., None, :]
    )
    distance = torch.linalg.vector_norm(
        current_gt_world[..., :, None, :] - reference_gt[..., None, :, :],
        dim=-1,
    )
    infinite = torch.full_like(distance, float("inf"))
    masked_distance = torch.where(pair_valid, distance, infinite)
    nearest_distance, nearest_key = masked_distance.min(dim=-1)
    has_key = pair_valid.any(dim=-1)
    supervised = has_key & nearest_distance.le(float(config.max_distance))

    if config.require_mutual_nearest:
        nearest_query = masked_distance.min(dim=-2).indices
        gathered = torch.gather(nearest_query, -1, nearest_key)
        query_index = torch.arange(
            probability.shape[-2], device=probability.device
        ).reshape(*([1] * (supervised.ndim - 1)), probability.shape[-2])
        supervised = supervised & gathered.eq(query_index)

    evaluation_mask = torch.ones_like(supervised)
    if sequence_indices is not None:
        selected = torch.zeros(
            probability.shape[1], dtype=torch.bool, device=probability.device
        )
        if sequence_indices:
            index = torch.as_tensor(
                sequence_indices, dtype=torch.long, device=probability.device
            )
            if bool((index < 0).any()) or bool((index >= probability.shape[1]).any()):
                raise ValueError("V8 sequence_indices are outside the sequence.")
            selected[index] = True
        evaluation_mask = selected[None, :, None, None].expand_as(supervised)
        supervised = supervised & evaluation_mask

    eligible = pair_valid & distance.le(float(config.max_distance))
    target_logits = (-distance / float(config.temperature)).masked_fill(
        ~eligible, -1e4
    )
    target_probability = torch.softmax(target_logits, dim=-1)
    target_probability = target_probability * eligible.to(target_probability.dtype)
    target_probability = target_probability / target_probability.sum(
        dim=-1, keepdim=True
    ).clamp_min(1e-8)
    target_probability = target_probability * supervised[..., None].to(
        target_probability.dtype
    )

    safe_prediction = probability.clamp_min(1e-8)
    safe_target = target_probability.clamp_min(1e-8)
    per_query_kl = (
        target_probability * (safe_target.log() - safe_prediction.log())
    ).sum(dim=-1)
    supervised_count = supervised.sum()
    loss = (per_query_kl * supervised.to(per_query_kl.dtype)).sum()
    loss = loss / supervised_count.clamp_min(1).to(loss.dtype)
    loss = torch.where(supervised_count > 0, loss, probability.sum() * 0.0)

    predicted_key = probability.argmax(dim=-1)
    predicted_distance = torch.gather(
        masked_distance, -1, predicted_key[..., None]
    )[..., 0]
    top1_exact = (predicted_key.eq(nearest_key) & supervised).sum()
    top1_within = (
        predicted_distance.le(float(config.max_distance)) & supervised
    ).sum()

    gt_key_mask = pair_valid.to(probability.dtype)
    gt_key_mass = (probability * gt_key_mask).sum(dim=-1)
    renormalized = probability * gt_key_mask
    renormalized = renormalized / renormalized.sum(dim=-1, keepdim=True).clamp_min(
        1e-8
    )
    finite_distance = torch.where(pair_valid, distance, torch.zeros_like(distance))
    expected_distance = (renormalized * finite_distance).sum(dim=-1)
    target_candidates = eligible.sum(dim=-1)
    active_frames = supervised.any(dim=-1).any(dim=-1)
    weight = supervised.to(probability.dtype)
    return V80MatchingResult(
        loss=loss,
        supervised_queries=supervised_count,
        candidate_queries=(
            has_key & query_valid & current_gt_valid & evaluation_mask
        ).sum(),
        top1_exact=top1_exact,
        top1_within_radius=top1_within,
        expected_distance=(expected_distance * weight).sum(),
        valid_key_probability_mass=(gt_key_mass * weight).sum(),
        target_candidates=(target_candidates * supervised).sum(),
        active_frames=active_frames,
    )


def v80_memory_write(batch: dict[str, torch.Tensor], *, min_confidence: float) -> torch.Tensor:
    return (
        batch["observed"].bool()
        & batch["identity_valid"].bool()
        & batch["quality"][..., 0].ge(float(min_confidence))
    )


def affinity_health(output: dict[str, torch.Tensor]) -> dict[str, float]:
    """Report masked, centered affinity strength rather than raw logit norms."""

    pair_valid = (
        output["transport_query_valid"][..., :, None]
        & output["transport_key_valid"][..., None, :]
    )

    def rms(name: str) -> torch.Tensor:
        value = output[name].float()
        mask = pair_valid.to(value.dtype)
        mean = (value * mask).sum(dim=-1, keepdim=True) / mask.sum(
            dim=-1, keepdim=True
        ).clamp_min(1.0)
        centered = (value - mean) * mask
        return torch.sqrt(
            centered.square().sum() / mask.sum().clamp_min(1.0) + 1e-12
        )

    geometry = rms("geometry_affinity_logits")
    sam = rms("sam_affinity_logits")
    return {
        "geometry_logit_rms": float(geometry.detach().cpu()),
        "sam_logit_rms": float(sam.detach().cpu()),
        "sam_to_geometry_logit_ratio": float(
            (sam / geometry.clamp_min(1e-6)).detach().cpu()
        ),
        "transport_entropy": float(
            output["transport_entropy"].float().mean().detach().cpu()
        ),
        "sam_affinity_delta": float(
            output["sam_affinity_delta"].float().mean().detach().cpu()
        ),
    }


def projection_gradient_norms(
    model: V80SupervisedCorrespondenceResidual,
) -> dict[str, float]:
    def norm(modules: tuple[nn.Module | None, ...]) -> float:
        squares = []
        for module in modules:
            if module is None:
                continue
            for parameter in module.parameters():
                if parameter.grad is not None:
                    squares.append(parameter.grad.detach().float().square().sum())
        if not squares:
            return 0.0
        return float(torch.sqrt(torch.stack(squares).sum()).cpu())

    geometry = norm((model.matcher.geometry_query, model.matcher.geometry_key))
    sam = norm((model.matcher.sam_query, model.matcher.sam_key))
    return {
        "geometry_projection_grad_norm": geometry,
        "sam_projection_grad_norm": sam,
        "sam_to_geometry_grad_ratio": sam / max(geometry, 1e-12),
    }
