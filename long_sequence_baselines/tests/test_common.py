from pathlib import Path

import numpy as np
from PIL import Image

from long_sequence_baselines.common import (
    TemporalPointSampler,
    discover_images,
    natural_sort_key,
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
