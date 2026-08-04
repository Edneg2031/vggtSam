import torch

from streaming_couping.src.solvers.weighted_kabsch import (
    KabschConfig,
    weighted_kabsch,
)


def test_weighted_kabsch_recovers_batched_rigid_transforms():
    torch.manual_seed(3)
    source = torch.randn(2, 16, 3, dtype=torch.float64)
    rotation = torch.eye(3, dtype=torch.float64).repeat(2, 1, 1)
    rotation[0, :2, :2] = torch.tensor(
        [[0.0, -1.0], [1.0, 0.0]], dtype=torch.float64
    )
    translation = torch.tensor([[0.1, 0.2, -0.3], [-0.2, 0.0, 0.4]]).double()
    target = torch.einsum("bij,bnj->bni", rotation, source) + translation[:, None]
    result = weighted_kabsch(source, target, config=KabschConfig(min_points=6))
    assert bool(result.accepted.all())
    assert torch.allclose(result.rotation, rotation, atol=1e-8)
    assert torch.allclose(result.translation, translation, atol=1e-8)


def test_weighted_kabsch_rejects_collinear_geometry():
    x = torch.linspace(-1.0, 1.0, 10)
    source = torch.stack([x, torch.zeros_like(x), torch.zeros_like(x)], dim=-1)
    result = weighted_kabsch(
        source,
        source + torch.tensor([0.1, 0.2, 0.3]),
        config=KabschConfig(min_points=6),
    )
    assert bool(result.degenerate)
    assert not bool(result.accepted)


def test_weighted_kabsch_empty_input_is_exact_identity():
    result = weighted_kabsch(
        torch.empty(0, 3),
        torch.empty(0, 3),
        config=KabschConfig(min_points=6),
    )
    assert not bool(result.accepted)
    assert int(result.point_count) == 0
    assert torch.equal(result.transform, torch.eye(4))


def test_weighted_kabsch_ignores_nonfinite_pairs():
    torch.manual_seed(11)
    source = torch.randn(12, 3, dtype=torch.float64)
    target = source + torch.tensor([0.1, -0.2, 0.3], dtype=torch.float64)
    source[:2] = float("nan")
    result = weighted_kabsch(source, target, config=KabschConfig(min_points=6))
    assert bool(result.accepted)
    assert int(result.point_count) == 10
    assert torch.allclose(
        result.translation,
        torch.tensor([0.1, -0.2, 0.3], dtype=torch.float64),
        atol=1e-8,
    )
