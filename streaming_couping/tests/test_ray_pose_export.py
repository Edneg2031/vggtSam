import torch

from streaming_couping.src.learned_pose.export import (
    _align_camera_pose,
    _world_confidence,
)


def test_align_camera_pose_matches_pointmap_similarity() -> None:
    native = torch.eye(4, dtype=torch.float64).repeat(2, 1, 1)
    native[0, :3, 3] = torch.tensor([1.0, 0.0, 0.0])
    native[1, :3, 3] = torch.tensor([0.0, 2.0, 0.0])
    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    translation = torch.tensor([0.5, -0.25, 2.0], dtype=torch.float64)

    c2w, w2c = _align_camera_pose(
        native,
        scale=2.0,
        rotation=rotation,
        translation=translation,
    )

    expected_centers = 2.0 * (native[:, :3, 3] @ rotation.T) + translation
    assert torch.allclose(c2w[:, :3, 3], expected_centers)
    assert torch.allclose(c2w[:, :3, :3], rotation.expand(2, -1, -1))
    assert torch.allclose(w2c @ c2w, torch.eye(4).double().expand(2, -1, -1))


def test_world_confidence_preserves_single_frame_axis() -> None:
    points = torch.zeros(1, 2, 3, 3)
    confidence = torch.ones(1, 1, 2, 3, 1)

    normalized = _world_confidence(confidence, points)

    assert normalized.shape == (1, 2, 3)
