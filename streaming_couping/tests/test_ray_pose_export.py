import torch

from streaming_couping.src.learned_pose.export import (
    _align_camera_pose,
    _camera_matrices_from_world_to_camera,
    _paired_distance_statistics,
    _world_confidence,
)
from vggtsam.utils.imports import maybe_add_repo_to_path


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


def test_repo_path_registration_adds_src_layout(tmp_path, monkeypatch) -> None:
    repo = tmp_path / "streamvggt"
    source = repo / "src"
    source.mkdir(parents=True)
    monkeypatch.setattr("sys.path", [])

    resolved = maybe_add_repo_to_path(repo)

    assert resolved == repo.resolve()
    assert str(source.resolve()) in __import__("sys").path
    assert str(repo.resolve()) in __import__("sys").path


def test_ground_truth_world_to_camera_conversion() -> None:
    w2c = torch.eye(4, dtype=torch.float64).repeat(2, 1, 1)
    w2c[1, 0, 3] = -2.0

    c2w, recovered_w2c = _camera_matrices_from_world_to_camera(
        w2c,
        frame_indices=(10, 20),
    )

    assert torch.allclose(
        c2w[1, :3, 3],
        torch.tensor([2.0, 0.0, 0.0], dtype=torch.float64),
    )
    assert torch.allclose(recovered_w2c, w2c)


def test_paired_distance_statistics_are_metric() -> None:
    predicted = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    target = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    statistics = _paired_distance_statistics(predicted, target)

    assert statistics["paired_distance_mean"] == 1.0
    assert torch.isclose(
        torch.tensor(statistics["paired_distance_rmse"]),
        torch.sqrt(torch.tensor(2.0)),
    )
