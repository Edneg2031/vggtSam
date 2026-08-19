from pathlib import Path

import numpy as np
import pytest

from streaming_couping.scripts.prepare_r3_cross_scene_geometry import (
    available_colmap_images,
    camera_intrinsics,
    fixed_source_indices,
    select_fixed_clip,
)
from vggtsam.data.scannetpp.colmap import Camera, Image


def _image(identifier: int, name: str) -> Image:
    return Image(
        id=identifier,
        qvec=np.asarray([1.0, 0.0, 0.0, 0.0]),
        tvec=np.zeros(3),
        camera_id=1,
        name=name,
    )


def test_r3_protocol_matches_development_source_indices() -> None:
    assert fixed_source_indices() == tuple(range(90, 526, 15))
    assert len(fixed_source_indices()) == 30


def test_available_colmap_images_filters_before_sequence_indexing(
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for name in ("000.jpg", "002.jpg", "003.jpg"):
        (image_dir / name).write_bytes(b"image")
    images = [
        _image(1, "000.jpg"),
        _image(2, "missing.jpg"),
        _image(3, "nested/002.jpg"),
        _image(4, "003.jpg"),
    ]
    available = available_colmap_images(images, image_dir=image_dir)
    assert [row[0] for row in available] == [0, 1, 2]
    assert [row[1].id for row in available] == [1, 3, 4]
    assert [row[2].name for row in available] == ["000.jpg", "002.jpg", "003.jpg"]
    selected = select_fixed_clip(available, source_indices=(0, 2))
    assert [row[1].id for row in selected] == [1, 4]


def test_fixed_clip_rejects_short_scene() -> None:
    available = [(0, _image(1, "000.jpg"), Path("000.jpg"))]
    with pytest.raises(ValueError, match="only 1"):
        select_fixed_clip(available, source_indices=(0, 1))


def test_camera_intrinsics_matches_pinhole_parameters() -> None:
    camera = Camera(
        id=1,
        model="PINHOLE",
        width=1920,
        height=1080,
        params=np.asarray([1000.0, 900.0, 960.0, 540.0]),
    )
    assert np.array_equal(
        camera_intrinsics(camera),
        np.asarray(
            [
                [1000.0, 0.0, 960.0],
                [0.0, 900.0, 540.0],
                [0.0, 0.0, 1.0],
            ]
        ),
    )
