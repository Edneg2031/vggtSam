import torch

from streaming_couping.scripts.audit_v7_cache import (
    _tensor_health,
    _uniform_sample_indices,
)


def test_large_tensor_sampling_never_rounds_past_last_element() -> None:
    elements = 30 * 8 * 256 * 384
    indices = _uniform_sample_indices(elements, 1_000_000)

    assert int(indices[0]) == 0
    assert int(indices[-1]) == elements - 1
    assert int(indices.min()) >= 0
    assert int(indices.max()) < elements


def test_large_boolean_mask_health_uses_bounded_sample() -> None:
    mask = torch.zeros(11, 100_003, dtype=torch.bool)
    health = _tensor_health(mask, full_scan=False)

    assert health["elements"] == mask.numel()
    assert health["sampled_elements"] == 1_000_000
    assert health["finite_sample"] == 1
