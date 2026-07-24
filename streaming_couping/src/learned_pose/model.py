"""Persistent instance encoder and zero-initialized pose fusion modules."""

from __future__ import annotations

from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        identity_valid: torch.Tensor | None = None,
        identity_unknown: torch.Tensor | None = None,
        *,
        branch: str,
        memory_ablation: str = "normal",
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        _validate_observation_shapes(appearance, geometry, quality, observed)
        if identity_valid is None:
            identity_valid = torch.ones_like(observed, dtype=torch.bool)
        if identity_valid.shape != observed.shape:
            raise ValueError("identity_valid must have shape [B,S,K].")
        if identity_unknown is None:
            identity_unknown = torch.zeros_like(
                observed,
                dtype=torch.bool,
            )
        if identity_unknown.shape != observed.shape:
            raise ValueError("identity_unknown must have shape [B,S,K].")
        memory_quality = quality
        if branch == "pose":
            feature_mode = self.config.pose_feature_mode
            if feature_mode == "appearance_only":
                geometry = torch.zeros_like(geometry)
                quality = torch.stack(
                    [
                        quality[..., 0],
                        torch.ones_like(quality[..., 1]),
                        torch.ones_like(quality[..., 2]),
                    ],
                    dim=-1,
                )
            elif feature_mode == "residual_only":
                appearance = torch.zeros_like(appearance)
            elif feature_mode != "appearance_and_residual":
                raise ValueError(
                    f"Unknown pose feature mode: {feature_mode!r}."
                )
            # Pose use is gated only by tracker/identity. Geometry/static
            # values remain detached token features for residual modes.
            gate_quality = quality
        elif branch == "geometry":
            # The selected geometry branch keeps geometry/static quality as
            # token features but hard-gates only on tracker confidence.
            gate_quality = torch.stack(
                [
                    quality[..., 0],
                    torch.ones_like(quality[..., 1]),
                    torch.ones_like(quality[..., 2]),
                ],
                dim=-1,
            )
        else:
            raise ValueError(f"Unknown final adapter branch: {branch!r}")
        if memory_ablation not in {"normal", "off", "wrong_id"}:
            raise ValueError(
                f"Unknown memory ablation: {memory_ablation!r}."
            )

        batch, sequence, instances = observed.shape
        app_memory = torch.zeros_like(appearance[:, 0])
        geo_memory = torch.zeros_like(geometry[:, 0])
        has_memory = torch.zeros(batch, instances, dtype=torch.bool, device=observed.device)
        age = torch.zeros(batch, instances, dtype=appearance.dtype, device=appearance.device)
        token_rows = []
        valid_rows = []
        update_rows = []
        reliability_rows = []
        momentum = float(self.config.memory_momentum)
        for frame in range(sequence):
            current_app = appearance[:, frame]
            current_geo = geometry[:, frame]
            current_quality = quality[:, frame]
            current_observed = observed[:, frame].bool()
            token_weight = torch.ones_like(
                current_observed,
                dtype=current_app.dtype,
            )
            if self.config.strict_identity_gate:
                current_match = identity_valid[:, frame].bool()
                current_unknown = identity_unknown[:, frame].bool()
                if branch == "pose":
                    identity_usable = current_match | current_unknown
                    token_weight = torch.where(
                        current_match,
                        torch.ones_like(token_weight),
                        torch.full_like(
                            token_weight,
                            float(self.config.unknown_camera_weight),
                        ),
                    )
                else:
                    identity_usable = current_match
                trusted = (
                    current_observed
                    & identity_usable
                    & (
                        gate_quality[:, frame, :, 0]
                        >= self.config.min_track_confidence
                    )
                )
                memory_trusted = (
                    trusted
                    & current_match
                    & (
                        memory_quality[:, frame, :, 1]
                        >= self.config.min_geometry_confidence
                    )
                    & (
                        memory_quality[:, frame, :, 2]
                        >= self.config.min_static_score
                    )
                )
            else:
                trusted = (
                    current_observed
                    & (gate_quality[:, frame, :, 0] >= self.config.min_track_confidence)
                    & (gate_quality[:, frame, :, 1] >= self.config.min_geometry_confidence)
                    & (gate_quality[:, frame, :, 2] >= self.config.min_static_score)
                )
                memory_trusted = trusted
            token_valid = trusted & has_memory
            feature_app_memory = app_memory
            feature_geo_memory = geo_memory
            if memory_ablation == "off":
                feature_app_memory = torch.zeros_like(app_memory)
                feature_geo_memory = torch.zeros_like(geo_memory)
            elif memory_ablation == "wrong_id" and instances > 1:
                feature_app_memory = torch.roll(
                    app_memory,
                    shifts=1,
                    dims=1,
                )
                feature_geo_memory = torch.roll(
                    geo_memory,
                    shifts=1,
                    dims=1,
                )
            features = torch.cat(
                [
                    current_app,
                    feature_app_memory,
                    current_app - feature_app_memory,
                    current_geo,
                    feature_geo_memory,
                    current_geo - feature_geo_memory,
                    current_quality,
                    torch.log1p(age)[..., None] / 4.0,
                ],
                dim=-1,
            )
            token = self.encoder(features)
            token_rows.append(token)
            valid_rows.append(token_valid)
            update_rows.append(memory_trusted)
            reliability_rows.append(
                torch.where(
                    token_valid,
                    token_weight,
                    torch.zeros_like(token_weight),
                )
            )

            update = memory_trusted[..., None]
            first = update & (~has_memory)[..., None]
            app_candidate = momentum * app_memory + (1.0 - momentum) * current_app
            geo_candidate = momentum * geo_memory + (1.0 - momentum) * current_geo
            app_memory = torch.where(first, current_app, torch.where(update, app_candidate, app_memory))
            geo_memory = torch.where(first, current_geo, torch.where(update, geo_candidate, geo_memory))
            has_memory = has_memory | memory_trusted
            age = torch.where(has_memory, age + 1.0, age)

        tokens = torch.stack(token_rows, dim=1)
        valid = torch.stack(valid_rows, dim=1)
        updates = torch.stack(update_rows, dim=1)
        reliability = torch.stack(reliability_rows, dim=1)
        return tokens, valid, {
            "effective_instances": valid.float().sum(dim=-1),
            "memory_updates": updates.float().sum(dim=-1),
            "_instance_reliability": reliability,
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
        *,
        instance_reliability: torch.Tensor | None = None,
        spatial_weight: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Fuse ``[B,S,Q,D]`` queries with ``[B,S,K,Di]`` instances."""

        if queries.ndim != 4 or instance_tokens.ndim != 4 or instance_valid.ndim != 3:
            raise ValueError("Expected queries [B,S,Q,D], instances [B,S,K,D], valid [B,S,K].")
        batch, sequence, query_count, query_dim = queries.shape
        if instance_tokens.shape[:3] != instance_valid.shape:
            raise ValueError("Instance token and validity shapes disagree.")
        if instance_tokens.shape[:2] != (batch, sequence):
            raise ValueError("Query and instance [B,S] dimensions disagree.")
        if instance_reliability is None:
            instance_reliability = instance_valid.to(dtype=queries.dtype)
        if instance_reliability.shape != instance_valid.shape:
            raise ValueError(
                "instance_reliability must have shape [B,S,K]."
            )
        if spatial_weight is not None and spatial_weight.shape != (
            batch,
            sequence,
            query_count,
            instance_valid.shape[2],
        ):
            raise ValueError(
                "spatial_weight must have shape [B,S,Q,K], got "
                f"{tuple(spatial_weight.shape)}."
            )
        flat_queries = queries.reshape(batch * sequence, query_count, query_dim)
        flat_instances = instance_tokens.reshape(
            batch * sequence,
            instance_tokens.shape[2],
            instance_tokens.shape[3],
        )
        flat_valid = instance_valid.reshape(batch * sequence, instance_valid.shape[2])
        flat_reliability = instance_reliability.reshape(
            batch * sequence,
            instance_valid.shape[2],
        ).to(dtype=queries.dtype)
        pair_weight = flat_reliability[:, None, :].expand(
            -1,
            query_count,
            -1,
        )
        if spatial_weight is not None:
            pair_weight = pair_weight * spatial_weight.reshape(
                batch * sequence,
                query_count,
                instance_valid.shape[2],
            ).to(dtype=queries.dtype)
        pair_weight = torch.where(
            flat_valid[:, None, :],
            pair_weight.clamp(0.0, 1.0),
            torch.zeros_like(pair_weight),
        )
        active_query = pair_weight.gt(0).any(dim=-1)
        active = active_query.any(dim=1)
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
            active_valid = flat_valid.index_select(0, active_indices)
            key_padding = torch.where(
                active_valid,
                torch.zeros_like(active_valid, dtype=q.dtype),
                torch.full_like(
                    active_valid,
                    float("-inf"),
                    dtype=q.dtype,
                ),
            )
            active_pair_weight = pair_weight.index_select(
                0,
                active_indices,
            )
            active_query_mask = active_query.index_select(
                0,
                active_indices,
            )
            attention_bias = torch.where(
                active_pair_weight > 0,
                torch.log(
                    active_pair_weight.float().clamp_min(1e-8)
                ),
                torch.full_like(
                    active_pair_weight,
                    float("-inf"),
                    dtype=torch.float32,
                ),
            )
            # MultiheadAttention cannot accept an all-masked query. Give
            # inactive queries a harmless finite row, then force their update
            # back to exactly zero below.
            attention_bias = torch.where(
                active_query_mask[..., None],
                attention_bias,
                torch.zeros_like(attention_bias),
            )
            attention_bias = (
                attention_bias[:, None]
                .expand(-1, self.attention.num_heads, -1, -1)
                .reshape(
                    -1,
                    query_count,
                    instance_valid.shape[2],
                )
            )
            update, weights = self.attention(
                q,
                kv,
                kv,
                key_padding_mask=key_padding,
                attn_mask=attention_bias.to(dtype=q.dtype),
                need_weights=True,
                average_attn_weights=False,
            )
            effective_reliability = (
                weights.float()
                * active_pair_weight[:, None].float()
            ).sum(dim=-1).mean(dim=1)
            effective_reliability = effective_reliability * (
                active_query_mask.float()
            )
            update = (
                update
                * effective_reliability[..., None].to(dtype=update.dtype)
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
        pose_geometry_dim: int | None = None,
        token_dim: int,
        config: FusionConfig,
    ) -> None:
        super().__init__()
        self.appearance_dim = int(appearance_dim)
        self.geometry_dim = int(geometry_dim)
        self.pose_geometry_dim = int(
            geometry_dim
            if pose_geometry_dim is None
            else pose_geometry_dim
        )
        self.token_dim = int(token_dim)
        self.config = config
        self.tokenizer = CausalInstanceTokenizer(
            appearance_dim,
            self.pose_geometry_dim,
            config,
        )
        # V2 deliberately separates learned pose and geometry projections so
        # their losses cannot fight through a shared persistent-token encoder.
        self.geometry_tokenizer = CausalInstanceTokenizer(
            appearance_dim,
            geometry_dim,
            config,
        )
        self.camera_fusion = ZeroInitializedCrossAttention(
            token_dim,
            config.instance_dim,
            config,
        )
        self.patch_token_fusions = nn.ModuleDict(
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
        pose_geometry: torch.Tensor,
        quality: torch.Tensor,
        observed: torch.Tensor,
        identity_valid: torch.Tensor | None = None,
        identity_unknown: torch.Tensor | None = None,
        memory_ablation: str = "normal",
        module_off: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if module_off:
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
            pose_geometry,
            quality,
            observed,
            identity_valid,
            identity_unknown,
            branch="pose",
            memory_ablation=memory_ablation,
        )
        instance_reliability = memory_log.pop("_instance_reliability")
        refined, log = self.camera_fusion(
            camera_hidden.unsqueeze(2),
            instance_tokens,
            valid,
            instance_reliability=instance_reliability,
        )
        return refined[:, :, 0], {**memory_log, **log}

    def forward_patch_tokens(
        self,
        token_levels: dict[int, torch.Tensor],
        *,
        patch_start_idx: int,
        appearance: torch.Tensor,
        geometry: torch.Tensor,
        quality: torch.Tensor,
        observed: torch.Tensor,
        identity_valid: torch.Tensor | None = None,
        identity_unknown: torch.Tensor | None = None,
        memory_ablation: str = "normal",
        shuffle_instance_tokens: bool = False,
        spatial_mask: torch.Tensor | None = None,
        patch_shape: tuple[int, int] | None = None,
        module_off: bool = False,
    ) -> tuple[dict[int, torch.Tensor], dict[str, torch.Tensor]]:
        """Update DPT patch tokens while preserving camera/register tokens."""

        if module_off:
            return dict(token_levels), {}
        instance_tokens, valid, memory_log = self.geometry_tokenizer(
            appearance,
            geometry,
            quality,
            observed,
            identity_valid,
            identity_unknown,
            branch="geometry",
            memory_ablation=memory_ablation,
        )
        instance_reliability = memory_log.pop("_instance_reliability")
        if shuffle_instance_tokens and instance_tokens.shape[2] > 1:
            instance_tokens = torch.roll(
                instance_tokens,
                shifts=1,
                dims=2,
            )
        start = int(patch_start_idx)
        output: dict[int, torch.Tensor] = {}
        logs: dict[str, torch.Tensor] = dict(memory_log)
        patch_gate = None
        pairwise_gate = None
        if self.config.strict_identity_gate:
            if spatial_mask is None or patch_shape is None:
                raise ValueError(
                    "Strict identity fusion requires a spatial mask and patch_shape."
                )
            instance_spatial_mask = None
            if spatial_mask.ndim == 5:
                instance_spatial_mask = spatial_mask
                union_spatial_mask = spatial_mask.any(dim=2)
            elif spatial_mask.ndim == 4:
                union_spatial_mask = spatial_mask
            else:
                raise ValueError(
                    "spatial_mask must have shape [B,S,H,W] or "
                    "[B,S,K,H,W]."
                )
            batch, sequence, height, width = union_spatial_mask.shape
            patch_h, patch_w = (int(value) for value in patch_shape)
            resized = F.interpolate(
                union_spatial_mask.float().reshape(
                    batch * sequence,
                    1,
                    height,
                    width,
                ),
                size=(patch_h, patch_w),
                mode="nearest",
            )
            dilation = int(self.config.patch_mask_dilation)
            if dilation:
                kernel = 2 * dilation + 1
                resized = F.max_pool2d(
                    resized,
                    kernel_size=kernel,
                    stride=1,
                    padding=dilation,
                )
            patch_gate = resized.reshape(
                batch,
                sequence,
                patch_h * patch_w,
            ).bool()
            if self.config.spatial_attention_mode == "per_instance":
                if instance_spatial_mask is None:
                    raise ValueError(
                        "Per-instance spatial attention requires "
                        "spatial_mask [B,S,K,H,W]."
                    )
                instance_count = instance_spatial_mask.shape[2]
                pairwise = F.interpolate(
                    instance_spatial_mask.float().reshape(
                        batch * sequence * instance_count,
                        1,
                        height,
                        width,
                    ),
                    size=(patch_h, patch_w),
                    mode="nearest",
                )
                if dilation:
                    pairwise = F.max_pool2d(
                        pairwise,
                        kernel_size=kernel,
                        stride=1,
                        padding=dilation,
                    )
                pairwise_gate = (
                    pairwise.reshape(
                        batch,
                        sequence,
                        instance_count,
                        patch_h * patch_w,
                    )
                    .permute(0, 1, 3, 2)
                    .contiguous()
                )
        for layer, tokens in token_levels.items():
            key = str(int(layer))
            if start < 0 or start >= tokens.shape[2]:
                raise ValueError(
                    f"patch_start_idx={start} is invalid for {tokens.shape[2]} tokens."
                )
            if key not in self.patch_token_fusions:
                output[int(layer)] = tokens
                continue
            prefix = tokens[:, :, :start]
            patches = tokens[:, :, start:]
            updated, current = self.patch_token_fusions[key](
                patches,
                instance_tokens,
                valid,
                instance_reliability=instance_reliability,
                spatial_weight=pairwise_gate,
            )
            if patch_gate is not None:
                if patch_gate.shape != patches.shape[:3]:
                    raise ValueError(
                        "Patch mask/token shape mismatch: "
                        f"{tuple(patch_gate.shape)} vs {tuple(patches.shape[:3])}."
                    )
                updated = torch.where(
                    patch_gate[..., None],
                    updated,
                    patches,
                )
            output[int(layer)] = torch.cat([prefix, updated], dim=2)
            logs.update({f"layer_{key}_{name}": value for name, value in current.items()})
        return output, logs

    def metadata(self) -> dict:
        return {
            "appearance_dim": self.appearance_dim,
            "geometry_dim": self.geometry_dim,
            "pose_geometry_dim": self.pose_geometry_dim,
            "token_dim": self.token_dim,
            "fusion_config": asdict(self.config),
        }


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
