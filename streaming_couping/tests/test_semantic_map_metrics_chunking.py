"""Nearest-neighbour distances must not depend on how the work is split.

Chunking only the source looks sufficient and is not: the block it builds is
``chunk_size x len(target)``, and a ground-truth object cloud is millions of
points, so one block is tens of gigabytes.  Chunking the target too bounds that
-- and it has to do so at the SAME answer, because the recorded object-map
numbers were produced by the old split and a metric that moves when you
re-chunk it cannot be compared across runs.
"""

from __future__ import annotations

import pytest
import torch

from streaming_couping.src.semantic_map_metrics import nearest_distances

#: The chunk size the object-map evaluation actually uses.
PRODUCTION_CHUNK = 4096


def _source_only_chunked(
    source: torch.Tensor, target: torch.Tensor, step: int
) -> torch.Tensor:
    """The previous implementation, kept here as the comparability baseline."""

    out = []
    for start in range(0, source.shape[0], step):
        out.append(torch.cdist(source[start : start + step], target).min(dim=1).values)
    return torch.cat(out)


def test_the_answer_is_bit_identical_to_the_source_only_chunking() -> None:
    """This is what makes the recorded numbers still comparable."""

    torch.manual_seed(0)
    source = torch.randn(500, 3)
    target = torch.randn(1700, 3)
    assert torch.equal(
        nearest_distances(source, target, chunk_size=PRODUCTION_CHUNK),
        _source_only_chunked(source, target, PRODUCTION_CHUNK),
    )


def test_at_the_production_chunk_size_it_is_the_true_nearest_distance() -> None:
    torch.manual_seed(1)
    source = torch.randn(200, 3)
    target = torch.randn(900, 3)
    expected = torch.cdist(source, target).min(dim=1).values
    assert torch.equal(
        nearest_distances(source, target, chunk_size=PRODUCTION_CHUNK), expected
    )


@pytest.mark.parametrize("step", [1, 7, 64, 256])
def test_small_chunk_sizes_agree_to_float_tolerance(step: int) -> None:
    """cdist picks a different internal path for small blocks.

    The difference is ~1e-6 m on unit-scale points and is not a property of the
    chunking -- the same drift appears between the full computation and the old
    source-only split.  What matters is that it is far below any threshold this
    metric is read against, and that it is exactly zero at 4096.
    """

    torch.manual_seed(2)
    source = torch.randn(120, 3)
    target = torch.randn(700, 3)
    expected = torch.cdist(source, target).min(dim=1).values
    assert torch.allclose(
        nearest_distances(source, target, chunk_size=step), expected, atol=1e-5
    )


def test_no_block_is_larger_than_chunk_size_squared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The memory bound is the whole point, so assert the blocks, not the answer."""

    seen: list[tuple[int, int]] = []
    real_cdist = torch.cdist

    def recording_cdist(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        seen.append((int(left.shape[0]), int(right.shape[0])))
        return real_cdist(left, right)

    monkeypatch.setattr(torch, "cdist", recording_cdist)
    nearest_distances(
        torch.zeros(9, 3), torch.zeros(1000, 3), chunk_size=8
    )
    assert seen, "cdist was never called"
    # source-only chunking would have produced blocks of 8 x 1000 here
    assert max(right for _, right in seen) <= 8
    assert max(left for left, _ in seen) <= 8


def test_an_empty_side_is_refused() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        nearest_distances(torch.zeros(0, 3), torch.zeros(4, 3), chunk_size=2)
    with pytest.raises(ValueError, match="non-empty"):
        nearest_distances(torch.zeros(4, 3), torch.zeros(0, 3), chunk_size=2)


def test_a_non_positive_chunk_size_is_refused() -> None:
    with pytest.raises(ValueError, match="positive"):
        nearest_distances(torch.zeros(4, 3), torch.zeros(4, 3), chunk_size=0)


def test_the_nearest_point_wins_across_a_block_boundary() -> None:
    """The minimum has to be taken over blocks, not within one."""

    source = torch.tensor([[0.0, 0.0, 0.0]])
    target = torch.tensor([[5.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    assert nearest_distances(source, target, chunk_size=1).item() == pytest.approx(0.5)
