import pytest
import torch

from streaming_couping.src.backbones.streamvggt_parallel import (
    parse_cuda_devices,
    partition_layers,
    resolve_amp_dtype,
)


def test_two_gpu_partition_is_contiguous_and_balanced() -> None:
    assignments = partition_layers(24, 2)

    assert assignments == [0] * 12 + [1] * 12


def test_partition_rejects_single_device() -> None:
    with pytest.raises(ValueError, match="depth/device_count"):
        partition_layers(24, 1)


def test_runtime_value_parsing() -> None:
    assert parse_cuda_devices(("cuda:0", "cuda:1")) == (
        torch.device("cuda:0"),
        torch.device("cuda:1"),
    )
    assert resolve_amp_dtype("bfloat16") is torch.bfloat16
    assert resolve_amp_dtype("float32") is None
