#!/usr/bin/env python3
"""CPU smoke checks for deterministic masked-history LoRA TTA."""

from __future__ import annotations

import torch
import torch.nn as nn

from streaming_couping.src.feature_tta import (
    LoRAConfig,
    cosine_feature_loss,
    deterministic_history_keep,
    history_for_replay_frame,
    inject_global_attention_lora,
    lora_parameters,
    lora_state_dict,
    reset_lora_modules,
    shuffled_teacher_index,
)


def main() -> None:
    test_zero_output_and_nonzero_gradient()
    test_deterministic_history_dropout()
    test_fixed_shuffled_teacher_permutation()
    print("T1 masked-history LoRA feature consistency smoke passed")


def test_zero_output_and_nonzero_gradient() -> None:
    torch.manual_seed(12)
    aggregator = _FakeAggregator()
    reference = aggregator.global_blocks[1].attn.qkv
    inputs = torch.randn(2, 5, 8)
    expected = reference(inputs).detach()
    modules = inject_global_attention_lora(
        aggregator,
        LoRAConfig(layer_indices=(1,), rank=2, alpha=2.0, dropout=0.0),
        seed=0,
    )
    actual = aggregator.global_blocks[1].attn.qkv(inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    target = torch.randn_like(actual)
    loss = cosine_feature_loss(actual.reshape(-1, 8), target.reshape(-1, 8))
    loss.backward()
    parameters = lora_parameters(modules)
    assert any(
        parameter.grad is not None and float(parameter.grad.abs().sum()) > 0.0
        for parameter in parameters
    )
    state = lora_state_dict(modules)
    assert all(float(value.abs().sum()) == 0.0 for key, value in state.items() if key.endswith("lora_b"))
    reset_lora_modules(modules, seed=0)
    reset = aggregator.global_blocks[1].attn.qkv(inputs)
    torch.testing.assert_close(reset, expected, rtol=0, atol=0)


def test_deterministic_history_dropout() -> None:
    first = deterministic_history_keep(17, drop_probability=0.5, seed=2026)
    second = deterministic_history_keep(17, drop_probability=0.5, seed=2026)
    different = deterministic_history_keep(17, drop_probability=0.5, seed=2027)
    assert first == second
    assert first != different
    assert first[0] == 0
    assert all(0 <= index < 17 for index in first)
    assert history_for_replay_frame(9, first) == tuple(
        index for index in first if index < 9
    )
    assert deterministic_history_keep(1, drop_probability=0.5, seed=0) == (0,)


def test_fixed_shuffled_teacher_permutation() -> None:
    permutation = tuple(shuffled_teacher_index(index, 17) for index in range(1, 18))
    assert permutation == tuple(range(9, 18)) + tuple(range(1, 9))
    assert len(set(permutation)) == 17
    assert all(source != target for source, target in zip(range(1, 18), permutation))


class _FakeAggregator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.global_blocks = nn.ModuleList(
            [
                _FakeBlock(),
                _FakeBlock(),
            ]
        )


class _FakeBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = _FakeAttention()


class _FakeAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qkv = nn.Linear(8, 8)
        self.proj = nn.Linear(8, 8)


if __name__ == "__main__":
    main()
