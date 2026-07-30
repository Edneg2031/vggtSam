from pathlib import Path

import numpy as np
from PIL import Image

from long_sequence_baselines.common import (
    TemporalPointSampler,
    camera_centers_from_w2c,
    discover_images,
    fit_similarity_transform,
    natural_sort_key,
    save_trajectory_overlay_plot,
    save_trajectory_plot,
    transform_w2c_world_similarity,
    write_binary_ply,
)


def test_natural_sort_orders_numeric_frame_names() -> None:
    names = ["frame_10.png", "frame_2.png", "frame_1.jpg"]
    assert sorted(names, key=natural_sort_key) == [
        "frame_1.jpg",
        "frame_2.png",
        "frame_10.png",
    ]


def test_discover_images_filters_and_limits(tmp_path: Path) -> None:
    for name in ("10.png", "2.jpg", "1.jpeg"):
        Image.new("RGB", (2, 2)).save(tmp_path / name)
    (tmp_path / "ignore.txt").write_text("x")
    assert [path.name for path in discover_images(tmp_path, 2)] == ["1.jpeg", "2.jpg"]


def test_temporal_sampler_is_bounded_and_covers_frames() -> None:
    sampler = TemporalPointSampler(max_points=6, total_frames=3)
    for frame in range(3):
        points = np.full((5, 3), frame, dtype=np.float32)
        colors = np.full((5, 3), frame, dtype=np.uint8)
        sampler.add(points, colors)
    points, colors = sampler.arrays()
    assert points.shape == colors.shape == (6, 3)
    assert set(points[:, 0].tolist()) == {0.0, 1.0, 2.0}


def test_binary_ply_header_has_expected_vertex_count(tmp_path: Path) -> None:
    path = tmp_path / "points.ply"
    write_binary_ply(
        path,
        np.zeros((3, 3), dtype=np.float32),
        np.zeros((3, 3), dtype=np.uint8),
    )
    assert b"element vertex 3\n" in path.read_bytes()[:256]


def test_horizon_style_trajectory_plot_is_written(tmp_path: Path) -> None:
    world_to_camera = np.repeat(
        np.eye(4, dtype=np.float64)[None, :3, :],
        2,
        axis=0,
    )
    world_to_camera[1, 0, 3] = -1.0
    path = tmp_path / "trajectory_compare.png"
    save_trajectory_plot(path, world_to_camera, "test")
    assert path.is_file()
    assert path.stat().st_size > 0

    overlay = tmp_path / "trajectory_overlay.png"
    save_trajectory_overlay_plot(
        overlay,
        {"raw": world_to_camera, "refined": world_to_camera.copy()},
        "test overlay",
    )
    assert overlay.is_file()
    assert overlay.stat().st_size > 0


def test_similarity_fit_and_w2c_transform_share_one_world_gauge() -> None:
    source_centers = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 3.0))
    )
    angle = np.deg2rad(30.0)
    expected_rotation = np.asarray(
        (
            (np.cos(angle), -np.sin(angle), 0.0),
            (np.sin(angle), np.cos(angle), 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    expected_scale = 2.5
    expected_translation = np.asarray((4.0, -2.0, 1.0))
    target_centers = (
        expected_scale * (source_centers @ expected_rotation.T)
        + expected_translation
    )
    scale, rotation, translation = fit_similarity_transform(
        source_centers,
        target_centers,
    )
    np.testing.assert_allclose(scale, expected_scale, atol=1e-10)
    np.testing.assert_allclose(rotation, expected_rotation, atol=1e-10)
    np.testing.assert_allclose(translation, expected_translation, atol=1e-10)

    source_w2c = np.repeat(np.eye(4)[None, :3], len(source_centers), axis=0)
    source_w2c[:, :3, 3] = -source_centers
    aligned = transform_w2c_world_similarity(
        source_w2c,
        scale=scale,
        rotation=rotation,
        translation=translation,
    )
    np.testing.assert_allclose(
        camera_centers_from_w2c(aligned),
        target_centers,
        atol=1e-10,
    )
