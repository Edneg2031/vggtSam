"""Small, auditable building blocks for StreamVGGT feature TTA.

The formal V0 model is never mutated on disk.  These helpers wrap a small
set of in-memory global-attention projections with zero-output LoRA branches
and define the deterministic masked-history protocol used by T1.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class LoRAConfig:
    layer_indices: tuple[int, ...] = (4, 11, 17, 23)
    rank: int = 4
    alpha: float = 4.0
    dropout: float = 0.0

    def validate(self, *, depth: int | None = None) -> None:
        if not self.layer_indices or len(set(self.layer_indices)) != len(
            self.layer_indices
        ):
            raise ValueError("LoRA layer_indices must be nonempty and unique.")
        if depth is not None and any(
            index < 0 or index >= int(depth) for index in self.layer_indices
        ):
            raise ValueError("A LoRA layer index is outside the aggregator depth.")
        if int(self.rank) < 1:
            raise ValueError("LoRA rank must be positive.")
        if float(self.alpha) <= 0.0:
            raise ValueError("LoRA alpha must be positive.")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ValueError("LoRA dropout must be in [0,1).")


class LoRALinear(nn.Module):
    """Frozen linear layer plus a zero-output low-rank residual."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float,
        seed: int,
    ) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRALinear can only wrap torch.nn.Linear.")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / float(self.rank)
        self.dropout = nn.Dropout(float(dropout))
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.lora_a = nn.Parameter(
            torch.empty(
                self.rank,
                int(base.in_features),
                device=base.weight.device,
                dtype=base.weight.dtype,
            )
        )
        self.lora_b = nn.Parameter(
            torch.empty(
                int(base.out_features),
                self.rank,
                device=base.weight.device,
                dtype=base.weight.dtype,
            )
        )
        self.reset_lora(seed=seed)

    def reset_lora(self, *, seed: int) -> None:
        # Generate on CPU so the same seed is identical across sharded GPUs.
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        bound = 1.0 / math.sqrt(max(int(self.base.in_features), 1))
        initial = torch.empty(
            self.lora_a.shape,
            dtype=torch.float32,
            device="cpu",
        ).uniform_(-bound, bound, generator=generator)
        with torch.no_grad():
            self.lora_a.copy_(
                initial.to(device=self.lora_a.device, dtype=self.lora_a.dtype)
            )
            self.lora_b.zero_()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base = self.base(inputs)
        residual = F.linear(F.linear(self.dropout(inputs), self.lora_a), self.lora_b)
        return base + residual * self.scale


def inject_global_attention_lora(
    aggregator: nn.Module,
    config: LoRAConfig,
    *,
    seed: int,
) -> dict[str, LoRALinear]:
    """Freeze the model and wrap qkv/proj at the requested global blocks."""

    depth = len(aggregator.global_blocks)
    config.validate(depth=depth)
    for parameter in aggregator.parameters():
        parameter.requires_grad_(False)
    modules: dict[str, LoRALinear] = {}
    module_seed = int(seed)
    for layer_index in config.layer_indices:
        attention = aggregator.global_blocks[int(layer_index)].attn
        for projection_name in ("qkv", "proj"):
            base = getattr(attention, projection_name)
            if isinstance(base, LoRALinear):
                raise ValueError(
                    f"Layer {layer_index} {projection_name} already has LoRA."
                )
            wrapped = LoRALinear(
                base,
                rank=config.rank,
                alpha=config.alpha,
                dropout=config.dropout,
                seed=module_seed,
            )
            setattr(attention, projection_name, wrapped)
            name = f"global_blocks.{int(layer_index)}.attn.{projection_name}"
            modules[name] = wrapped
            module_seed += 1
    return modules


def reset_lora_modules(modules: Mapping[str, LoRALinear], *, seed: int) -> None:
    for offset, name in enumerate(sorted(modules)):
        modules[name].reset_lora(seed=int(seed) + offset)


def lora_parameters(modules: Mapping[str, LoRALinear]) -> list[nn.Parameter]:
    output: list[nn.Parameter] = []
    for name in sorted(modules):
        output.extend((modules[name].lora_a, modules[name].lora_b))
    return output


def lora_state_dict(modules: Mapping[str, LoRALinear]) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {}
    for name in sorted(modules):
        output[f"{name}.lora_a"] = modules[name].lora_a.detach().float().cpu()
        output[f"{name}.lora_b"] = modules[name].lora_b.detach().float().cpu()
    return output


def load_lora_state_dict(
    modules: Mapping[str, LoRALinear], state: Mapping[str, torch.Tensor]
) -> None:
    expected = {
        f"{name}.{suffix}"
        for name in modules
        for suffix in ("lora_a", "lora_b")
    }
    if set(state) != expected:
        raise ValueError("LoRA state keys do not match injected modules.")
    with torch.no_grad():
        for name, module in modules.items():
            for suffix in ("lora_a", "lora_b"):
                parameter = getattr(module, suffix)
                value = torch.as_tensor(state[f"{name}.{suffix}"])
                if tuple(value.shape) != tuple(parameter.shape):
                    raise ValueError(f"LoRA state shape mismatch for {name}.{suffix}.")
                parameter.copy_(value.to(parameter.device, parameter.dtype))


def deterministic_history_keep(
    prefix_index: int,
    *,
    drop_probability: float,
    seed: int,
) -> tuple[int, ...]:
    """Select retained history for one causal prefix.

    Frame zero is the fixed anchor, while the current frame is not part of
    history and is therefore always evaluated.  Intermediate history is
    independently retained with probability ``1-drop_probability``.
    """

    current = int(prefix_index)
    if current < 1:
        raise ValueError("Masked-history TTA starts at prefix index 1.")
    if not 0.0 <= float(drop_probability) < 1.0:
        raise ValueError("drop_probability must be in [0,1).")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    selected = [0]
    if current > 1:
        samples = torch.rand(current - 1, generator=generator)
        selected.extend(
            index + 1
            for index, value in enumerate(samples.tolist())
            if float(value) >= float(drop_probability)
        )
    return tuple(selected)


def history_for_replay_frame(
    replay_frame_index: int,
    retained_for_target: Sequence[int],
) -> tuple[int, ...]:
    current = int(replay_frame_index)
    return tuple(
        int(index)
        for index in retained_for_target
        if 0 <= int(index) < current
    )


def shuffled_teacher_index(prefix_index: int, adaptation_prefix_count: int) -> int:
    """Fixed half-cycle control permutation over prefixes 1..N."""

    current = int(prefix_index)
    count = int(adaptation_prefix_count)
    if count < 2 or not 1 <= current <= count:
        raise ValueError("Invalid prefix for shuffled teacher permutation.")
    shift = count // 2
    return 1 + ((current - 1 + shift) % count)


def global_patch_tokens(
    selected: Mapping[int, torch.Tensor],
    *,
    layer_index: int,
    patch_start_idx: int,
) -> torch.Tensor:
    value = selected[int(layer_index)]
    if value.ndim != 4 or value.shape[:2] != (1, 1):
        raise ValueError(
            "Selected StreamVGGT tokens must have shape [1,1,T,2C]."
        )
    channels = int(value.shape[-1])
    if channels % 2:
        raise ValueError("Concatenated frame/global token channels must be even.")
    patch_start = int(patch_start_idx)
    if not 0 <= patch_start < int(value.shape[-2]):
        raise ValueError("patch_start_idx is outside selected tokens.")
    return value[0, 0, patch_start:, channels // 2 :]


def cosine_feature_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    if student.shape != teacher.shape or student.ndim != 2:
        raise ValueError(
            f"Student/teacher features must share [P,C], got "
            f"{tuple(student.shape)}/{tuple(teacher.shape)}."
        )
    student_normalized = F.normalize(student.float(), dim=-1, eps=1e-6)
    teacher_normalized = F.normalize(
        teacher.detach().to(student.device).float(), dim=-1, eps=1e-6
    )
    return (1.0 - (student_normalized * teacher_normalized).sum(dim=-1)).mean()


def gradient_norm(parameters: Iterable[nn.Parameter]) -> float:
    squared = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            value = parameter.grad.detach().float().norm()
            squared += float(value.cpu()) ** 2
    return math.sqrt(squared)


def clip_gradients(parameters: Iterable[nn.Parameter], maximum_norm: float) -> float:
    parameters = list(parameters)
    norm = gradient_norm(parameters)
    if norm > float(maximum_norm) > 0.0:
        scale = float(maximum_norm) / (norm + 1e-12)
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.mul_(scale)
    return norm


def assert_only_lora_trainable(model: nn.Module) -> None:
    trainable = [name for name, value in model.named_parameters() if value.requires_grad]
    if not trainable or any(
        not (name.endswith("lora_a") or name.endswith("lora_b"))
        for name in trainable
    ):
        raise RuntimeError(f"Unexpected trainable parameters: {trainable[:20]}")
