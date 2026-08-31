"""Exact-causal layer-wise model parallelism for StreamVGGT.

This is inference-only model parallelism, not DDP.  Consecutive aggregator
layers are placed on different GPUs and every layer keeps its own full-history
KV cache on that GPU.  No temporal chunking or history reset is introduced.
"""

from __future__ import annotations

import copy
from contextlib import nullcontext
from types import MethodType
from typing import Any, Callable, Sequence

import torch
import torch.nn.functional as F


def partition_layers(depth: int, device_count: int) -> list[int]:
    """Assign contiguous, nearly equal layer ranges to the requested GPUs."""

    if depth <= 0 or device_count < 2 or device_count > depth:
        raise ValueError(
            f"Invalid StreamVGGT depth/device_count: {depth}/{device_count}"
        )
    base, remainder = divmod(depth, device_count)
    assignments: list[int] = []
    for device_index in range(device_count):
        count = base + (1 if device_index < remainder else 0)
        assignments.extend([device_index] * count)
    return assignments


def parse_cuda_devices(values: Sequence[str]) -> tuple[torch.device, ...]:
    devices = tuple(torch.device(str(value).strip()) for value in values)
    if len(devices) < 2:
        raise ValueError(
            "StreamVGGT model parallelism requires at least two CUDA devices."
        )
    if len(set(devices)) != len(devices):
        raise ValueError(f"StreamVGGT devices must be distinct, got {devices}.")
    if any(device.type != "cuda" for device in devices):
        raise ValueError(
            f"StreamVGGT model-parallel devices must be CUDA, got {devices}."
        )
    return devices


def resolve_amp_dtype(value: str) -> torch.dtype | None:
    normalized = str(value).strip().lower()
    if normalized in {"float32", "fp32", "none"}:
        return None
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float16", "fp16"}:
        return torch.float16
    raise ValueError(
        "streamvggt_amp_dtype must be float32, bfloat16, or float16; "
        f"got {value!r}."
    )


def _install_processed_key_cache(attention: torch.nn.Module) -> None:
    """Store normalized/RoPE keys so historical keys are not recomputed."""

    if getattr(attention, "_streaming_couping_processed_key_cache", False):
        return
    object.__setattr__(
        attention,
        "_streaming_couping_processed_key_cache",
        True,
    )
    object.__setattr__(
        attention,
        "forward",
        MethodType(_processed_key_cache_forward, attention),
    )


def _processed_key_cache_forward(
    attention,
    x: torch.Tensor,
    pos=None,
    attn_mask=None,
    past_key_values=None,
    use_cache=False,
):
    """Upstream-equivalent attention with already-processed cached keys."""

    if not use_cache:
        return type(attention).forward(
            attention,
            x,
            pos=pos,
            attn_mask=attn_mask,
            past_key_values=past_key_values,
            use_cache=False,
        )

    batch, query_tokens, channels = x.shape
    qkv = (
        attention.qkv(x)
        .reshape(
            batch,
            query_tokens,
            3,
            attention.num_heads,
            attention.head_dim,
        )
        .permute(2, 0, 3, 1, 4)
    )
    query, key, value = qkv.unbind(0)
    query = attention.q_norm(query)
    key = attention.k_norm(key)
    if attention.rope is not None:
        query = attention.rope(query, pos)
        key = attention.rope(key, pos)

    key = key.unsqueeze(2)
    value = value.unsqueeze(2)
    if past_key_values is not None:
        past_key, past_value = past_key_values
        key = torch.cat([past_key, key], dim=2)
        value = torch.cat([past_value, value], dim=2)
    new_key_values = (key, value)

    key = key.reshape(
        batch,
        attention.num_heads,
        -1,
        attention.head_dim,
    )
    value = value.reshape(
        batch,
        attention.num_heads,
        -1,
        attention.head_dim,
    )
    if attention.fused_attn:
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=attention.attn_drop.p if attention.training else 0.0,
        )
    else:
        scores = (query * attention.scale) @ key.transpose(-2, -1)
        if attn_mask is not None:
            scores = scores + attn_mask
        weights = attention.attn_drop(scores.softmax(dim=-1))
        output = weights @ value

    output = output.transpose(1, 2).reshape(
        batch,
        query_tokens,
        channels,
    )
    output = attention.proj_drop(attention.proj(output))
    return output, new_key_values


def assert_processed_key_cache_equivalence() -> None:
    """Verify the optimized KV representation against upstream attention."""

    from streamvggt.layers.attention import Attention
    from streamvggt.layers.rope import RotaryPositionEmbedding2D

    with torch.random.fork_rng(devices=[]), torch.inference_mode():
        torch.manual_seed(2026)
        reference = Attention(
            dim=32,
            num_heads=4,
            qk_norm=True,
            rope=RotaryPositionEmbedding2D(frequency=100),
        ).eval()
        optimized = copy.deepcopy(reference).eval()
        _install_processed_key_cache(optimized)
        positions = torch.tensor(
            [[[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]],
            dtype=torch.long,
        )
        reference_cache = None
        optimized_cache = None
        for _ in range(3):
            tokens = torch.randn(1, 6, 32)
            reference_output, reference_cache = reference(
                tokens,
                pos=positions,
                past_key_values=reference_cache,
                use_cache=True,
            )
            optimized_output, optimized_cache = optimized(
                tokens,
                pos=positions,
                past_key_values=optimized_cache,
                use_cache=True,
            )
            torch.testing.assert_close(
                optimized_output,
                reference_output,
                rtol=1e-5,
                atol=1e-6,
            )


def assert_frame_repository_cache_equivalence() -> None:
    """Verify that gathering per-frame KV reproduces a rolling full cache."""

    from streamvggt.layers.attention import Attention
    from streamvggt.layers.rope import RotaryPositionEmbedding2D

    with torch.random.fork_rng(devices=[]), torch.inference_mode():
        torch.manual_seed(2027)
        rolling = Attention(
            dim=32,
            num_heads=4,
            qk_norm=True,
            rope=RotaryPositionEmbedding2D(frequency=100),
        ).eval()
        repository = copy.deepcopy(rolling).eval()
        _install_processed_key_cache(rolling)
        _install_processed_key_cache(repository)
        positions = torch.tensor(
            [[[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]],
            dtype=torch.long,
        )
        rolling_cache = None
        frame_repository: list[tuple[torch.Tensor, torch.Tensor]] = []
        for _ in range(4):
            tokens = torch.randn(1, 6, 32)
            rolling_output, rolling_cache = rolling(
                tokens,
                pos=positions,
                past_key_values=rolling_cache,
                use_cache=True,
            )
            selected_cache = _gather_frame_key_values(
                frame_repository,
                tuple(range(len(frame_repository))),
            )
            repository_output, combined = repository(
                tokens,
                pos=positions,
                past_key_values=selected_cache,
                use_cache=True,
            )
            frame_repository.append(_last_frame_key_values(combined))
            torch.testing.assert_close(
                repository_output,
                rolling_output,
                rtol=1e-5,
                atol=1e-6,
            )
            torch.testing.assert_close(
                torch.cat([value[0] for value in frame_repository], dim=2),
                rolling_cache[0],
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                torch.cat([value[1] for value in frame_repository], dim=2),
                rolling_cache[1],
                rtol=0,
                atol=0,
            )


class LayerShardedStreamVGGT:
    """Run full-history StreamVGGT with aggregator layers split across GPUs."""

    def __init__(
        self,
        model: Any,
        devices: Sequence[str],
        *,
        selected_layer_indices: Sequence[int] = (),
        amp_dtype: str = "bfloat16",
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "StreamVGGT model parallelism requires CUDA; use the existing "
                "single-device path on CPU."
            )
        self.model = model.eval()
        self.devices = parse_cuda_devices(devices)
        self.amp_dtype = resolve_amp_dtype(amp_dtype)
        self.aggregator = model.aggregator
        if self.aggregator.aa_order != ["frame", "global"]:
            raise ValueError(
                "Layer sharding requires aa_order=['frame', 'global'], got "
                f"{self.aggregator.aa_order}."
            )
        if self.aggregator.aa_block_size != 1:
            raise ValueError(
                "Layer sharding requires aa_block_size=1, got "
                f"{self.aggregator.aa_block_size}."
            )

        self.layer_devices = partition_layers(
            self.aggregator.depth,
            len(self.devices),
        )
        requested = {
            self._resolve_layer_index(index)
            for index in selected_layer_indices
        }
        self.selected_layers = sorted(
            requested
            | set(model.depth_head.intermediate_layer_idx)
            | set(model.point_head.intermediate_layer_idx)
            | {self.aggregator.depth - 1}
        )
        self._distribute_model()
        self.reset()

    @property
    def first_device(self) -> torch.device:
        return self.devices[0]

    @property
    def patch_start_idx(self) -> int:
        return int(self.aggregator.patch_start_idx)

    def reset(self) -> None:
        """Release the previous clip history before starting a new clip."""

        self.past_key_values: list[Any] = [None] * self.aggregator.depth
        self.frame_key_values: list[
            list[tuple[torch.Tensor, torch.Tensor]]
        ] = [[] for _ in range(self.aggregator.depth)]
        self.past_key_values_camera: list[Any] = [
            None
        ] * self.model.camera_head.trunk_depth
        self.last_retrieval: dict[str, Any] | None = None

    def layout_summary(self) -> tuple[str, ...]:
        lines = []
        for device_index, device in enumerate(self.devices):
            layers = [
                index
                for index, assignment in enumerate(self.layer_devices)
                if assignment == device_index
            ]
            roles = []
            if device_index == 0:
                roles.append("patch/depth")
            if device_index == 1:
                roles.append("point")
            if device_index == len(self.devices) - 1:
                roles.append("camera")
            lines.append(
                f"{device}=layers {layers[0]}-{layers[-1]}"
                + (f" + {','.join(roles)}" if roles else "")
            )
        return tuple(lines)

    def aggregate_frame(
        self,
        images: torch.Tensor,
        frame_index: int,
        *,
        history_selector: Callable[
            [int, torch.Tensor],
            Sequence[int],
        ]
        | None = None,
    ) -> dict[int, torch.Tensor]:
        """Run one frame and retain only levels consumed by downstream heads."""

        if images.ndim != 5 or images.shape[:2] != (1, 1):
            raise ValueError(
                f"Expected one image [1, 1, 3, H, W], got {images.shape}."
            )
        context = (
            torch.autocast(
                device_type="cuda",
                dtype=self.amp_dtype,
            )
            if self.amp_dtype is not None
            else nullcontext()
        )
        with context:
            return self._aggregate_frame_impl(
                images,
                frame_index,
                history_selector=history_selector,
            )

    def _aggregate_frame_impl(
        self,
        images: torch.Tensor,
        frame_index: int,
        *,
        history_selector: Callable[
            [int, torch.Tensor],
            Sequence[int],
        ]
        | None,
    ) -> dict[int, torch.Tensor]:
        aggregator = self.aggregator
        first = self.first_device
        images = images.to(first, non_blocking=True)
        _, _, channels, height, width = images.shape
        if channels != 3:
            raise ValueError(f"Expected three input channels, got {channels}.")

        normalized = (images - aggregator._resnet_mean) / aggregator._resnet_std
        patch_tokens = aggregator.patch_embed(
            normalized.reshape(1, 3, height, width)
        )
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]
        token_slot = 0 if frame_index == 0 else 1
        camera_token = aggregator.camera_token[:, token_slot]
        register_token = aggregator.register_token[:, token_slot]
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        position = None
        if aggregator.rope is not None:
            position = aggregator.position_getter(
                1,
                height // aggregator.patch_size,
                width // aggregator.patch_size,
                device=first,
            )
            position = position + 1
            special = torch.zeros(
                1,
                aggregator.patch_start_idx,
                2,
                device=first,
                dtype=position.dtype,
            )
            position = torch.cat([special, position], dim=1)

        selected: dict[int, torch.Tensor] = {}
        selected_history: tuple[int, ...] | None = None
        self.last_retrieval = None
        for layer_index, device_index in enumerate(self.layer_devices):
            device = self.devices[device_index]
            if tokens.device != device:
                tokens = tokens.to(device, non_blocking=True)
            layer_position = (
                None
                if position is None
                else position.to(device, non_blocking=True)
            )
            frame_tokens = aggregator.frame_blocks[layer_index](
                tokens,
                pos=layer_position,
            )
            if history_selector is not None and layer_index == 0:
                scores = _first_global_layer_frame_scores(
                    aggregator.global_blocks[layer_index],
                    frame_tokens,
                    position=layer_position,
                    frame_key_values=self.frame_key_values[layer_index],
                    patch_start_idx=self.patch_start_idx,
                )
                requested = tuple(
                    int(value)
                    for value in history_selector(
                        frame_index,
                        scores.clone(),
                    )
                )
                _validate_history_selection(
                    requested,
                    history_frames=len(self.frame_key_values[layer_index]),
                    frame_index=frame_index,
                )
                selected_history = requested
                self.last_retrieval = {
                    "frame_index": int(frame_index),
                    "history_frames": len(self.frame_key_values[layer_index]),
                    "qk_scores": scores.clone(),
                    "selected_history_indices": selected_history,
                }
            if history_selector is not None:
                if selected_history is None:
                    raise RuntimeError(
                        "Frame retrieval selection was not initialized at layer 0."
                    )
                past_key_values = _gather_frame_key_values(
                    self.frame_key_values[layer_index],
                    selected_history,
                )
            else:
                past_key_values = self.past_key_values[layer_index]
            tokens, new_key_values = aggregator.global_blocks[layer_index](
                frame_tokens,
                pos=layer_position,
                past_key_values=past_key_values,
                use_cache=True,
            )
            if history_selector is None:
                self.past_key_values[layer_index] = new_key_values
            else:
                self.frame_key_values[layer_index].append(
                    _last_frame_key_values(new_key_values)
                )
            if layer_index in self.selected_layers:
                selected[layer_index] = torch.cat(
                    [frame_tokens[:, None], tokens[:, None]],
                    dim=-1,
                )
        return selected

    def camera(self, selected: dict[int, torch.Tensor]) -> torch.Tensor:
        device = self.devices[-1]
        token_list: list[Any] = [None] * self.aggregator.depth
        token_list[-1] = selected[self.aggregator.depth - 1].to(
            device,
            non_blocking=True,
        ).float()
        with torch.autocast(device_type="cuda", enabled=False):
            pose_list, self.past_key_values_camera = self.model.camera_head(
                token_list,
                past_key_values_camera=self.past_key_values_camera,
                use_cache=True,
            )
        return pose_list[-1]

    def depth(
        self,
        selected: dict[int, torch.Tensor],
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.devices[0]
        token_list = self._tokens_for_head(
            selected,
            self.model.depth_head,
            device,
        )
        with torch.autocast(device_type="cuda", enabled=False):
            return self.model.depth_head(
                token_list,
                images=images.to(device, non_blocking=True).float(),
                patch_start_idx=self.patch_start_idx,
            )

    def points(
        self,
        selected: dict[int, torch.Tensor],
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.devices[1]
        token_list = self._tokens_for_head(
            selected,
            self.model.point_head,
            device,
        )
        with torch.autocast(device_type="cuda", enabled=False):
            return self.model.point_head(
                token_list,
                images=images.to(device, non_blocking=True).float(),
                patch_start_idx=self.patch_start_idx,
            )

    def _tokens_for_head(
        self,
        selected: dict[int, torch.Tensor],
        head: Any,
        device: torch.device,
    ) -> list[Any]:
        token_list: list[Any] = [None] * self.aggregator.depth
        for layer_index in head.intermediate_layer_idx:
            token_list[layer_index] = selected[layer_index].to(
                device,
                non_blocking=True,
            ).float()
        return token_list

    def _resolve_layer_index(self, index: int) -> int:
        resolved = int(index)
        if resolved < 0:
            resolved += int(self.aggregator.depth)
        if not 0 <= resolved < int(self.aggregator.depth):
            raise ValueError(
                f"StreamVGGT layer index {index} is outside "
                f"[-{self.aggregator.depth}, {self.aggregator.depth - 1}]."
            )
        return resolved

    def resolved_layer_index(self, index: int) -> int:
        return self._resolve_layer_index(index)

    def _distribute_model(self) -> None:
        first = self.devices[0]
        point_device = self.devices[1]
        camera_device = self.devices[-1]
        self.aggregator.patch_embed.to(first)
        _move_parameter(self.aggregator.camera_token, first)
        _move_parameter(self.aggregator.register_token, first)
        for name in ("_resnet_mean", "_resnet_std"):
            buffer = self.aggregator._buffers.get(name)
            if buffer is not None:
                self.aggregator._buffers[name] = buffer.to(first)
        for layer_index, device_index in enumerate(self.layer_devices):
            device = self.devices[device_index]
            self.aggregator.frame_blocks[layer_index].to(device)
            self.aggregator.global_blocks[layer_index].to(device)
            _install_processed_key_cache(
                self.aggregator.global_blocks[layer_index].attn
            )
        self.model.depth_head.to(first)
        self.model.point_head.to(point_device)
        self.model.camera_head.to(camera_device)


def _first_global_layer_frame_scores(
    block: torch.nn.Module,
    frame_tokens: torch.Tensor,
    *,
    position: torch.Tensor | None,
    frame_key_values: Sequence[tuple[torch.Tensor, torch.Tensor]],
    patch_start_idx: int,
) -> torch.Tensor:
    """Return exact processed-Q/processed-K frame relevance on CPU."""

    if not frame_key_values:
        return torch.empty(0, dtype=torch.float32)
    attention = block.attn
    batch, token_count, channels = frame_tokens.shape
    qkv = (
        attention.qkv(block.norm1(frame_tokens))
        .reshape(
            batch,
            token_count,
            3,
            attention.num_heads,
            attention.head_dim,
        )
        .permute(2, 0, 3, 1, 4)
    )
    query = attention.q_norm(qkv[0])
    if attention.rope is not None:
        query = attention.rope(query, position)
    start = int(patch_start_idx)
    if not 0 <= start < token_count:
        raise ValueError(
            f"Invalid patch_start_idx={start} for {token_count} tokens."
        )
    query_descriptor = query[:, :, start:, :].mean(dim=2)
    key_descriptors = torch.stack(
        [
            key[:, :, 0, start:, :].mean(dim=2)
            for key, _ in frame_key_values
        ],
        dim=2,
    )
    scores = (
        query_descriptor.unsqueeze(2)
        .mul(key_descriptors)
        .sum(dim=-1)
        .mean(dim=(0, 1))
    )
    return scores.detach().float().cpu()


def _gather_frame_key_values(
    repository: Sequence[tuple[torch.Tensor, torch.Tensor]],
    indices: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not indices:
        return None
    return (
        torch.cat([repository[int(index)][0] for index in indices], dim=2),
        torch.cat([repository[int(index)][1] for index in indices], dim=2),
    )


def _last_frame_key_values(
    values: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    key, value = values
    if key.ndim != 5 or value.ndim != 5 or key.shape[2] < 1:
        raise ValueError(
            "Frame KV repository requires [B,H,F,P,D] key/value tensors."
        )
    # clone() is required: a detached last-frame view would still retain the
    # storage for the full selected-history tensor and grow quadratically.
    return (
        key[:, :, -1:, :, :].detach().clone(),
        value[:, :, -1:, :, :].detach().clone(),
    )


def _validate_history_selection(
    indices: Sequence[int],
    *,
    history_frames: int,
    frame_index: int,
) -> None:
    values = tuple(int(value) for value in indices)
    if len(values) != len(set(values)):
        raise ValueError(f"Retrieval selected duplicate history frames: {values}.")
    if tuple(sorted(values)) != values:
        raise ValueError(
            f"Retrieval history frames must be time ordered, got {values}."
        )
    if any(value < 0 or value >= int(history_frames) for value in values):
        raise ValueError(
            "Retrieval selected a non-causal or unavailable frame: "
            f"frame={frame_index} history={history_frames} selected={values}."
        )


def _move_parameter(
    parameter: torch.nn.Parameter,
    device: torch.device,
) -> None:
    parameter.data = parameter.data.to(device)
    if parameter.grad is not None:
        parameter.grad.data = parameter.grad.data.to(device)
