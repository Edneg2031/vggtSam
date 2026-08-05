import torch

from streaming_couping.src.solvers.weighted_kabsch import KabschConfig
from streaming_couping.src.solvers.weighted_umeyama import weighted_umeyama


def test_weighted_umeyama_recovers_similarity_without_embedding_scale_in_pose():
    torch.manual_seed(17)
    source = torch.randn(40, 3, dtype=torch.double)
    angle = torch.tensor(0.37, dtype=torch.double)
    rotation = torch.tensor(
        [
            [torch.cos(angle), -torch.sin(angle), 0.0],
            [torch.sin(angle), torch.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.double,
    )
    scale = 1.7
    translation = torch.tensor([0.2, -0.4, 1.1], dtype=torch.double)
    target = scale * (source @ rotation.T) + translation

    result = weighted_umeyama(
        source,
        target,
        config=KabschConfig(min_points=6),
    )

    assert bool(result.accepted)
    assert torch.allclose(
        result.scale, torch.tensor(scale, dtype=torch.double), atol=1e-10
    )
    assert torch.allclose(result.rotation, rotation, atol=1e-10)
    assert torch.allclose(result.translation, translation, atol=1e-10)
    assert torch.allclose(result.transform[:3, :3], rotation, atol=1e-10)
    assert torch.allclose(result.transform[:3, 3], translation, atol=1e-10)
    assert torch.allclose(
        result.transform[3], torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.double)
    )


def test_weighted_umeyama_trimming_rejects_large_outlier():
    torch.manual_seed(23)
    source = torch.randn(30, 3, dtype=torch.double)
    target = 0.8 * source + torch.tensor([1.0, 2.0, -0.5], dtype=torch.double)
    target[-1] += 100.0
    result = weighted_umeyama(
        source,
        target,
        config=KabschConfig(
            min_points=6,
            trim_quantile=0.8,
            trim_iterations=3,
        ),
    )
    assert bool(result.accepted)
    assert abs(float(result.scale) - 0.8) < 1e-8
    assert int(result.retained_count) < int(result.point_count)


def test_weighted_umeyama_empty_input_is_exact_rigid_identity():
    source = torch.empty(0, 3)
    result = weighted_umeyama(source, source)
    assert not bool(result.accepted)
    assert torch.equal(result.transform, torch.eye(4))
    assert torch.isnan(result.scale)
