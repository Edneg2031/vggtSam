from pathlib import Path
import sys
import types

import pytest
import torch
from PIL import Image

from streaming_couping.src.horizonstream_cache import (
    HORIZONSTREAM_CACHE_SCHEMA,
    HORIZONSTREAM_CACHE_VERSION,
    horizonstream_cache_matches,
    image_file_signatures,
    materialize_horizonstream_rgb,
    normalize_horizonstream_confidence,
    validate_horizonstream_cache,
)
from streaming_couping.src.semantic_mapping.adapters import (
    HorizonStreamGeometryCacheAdapter,
)
from streaming_couping.src.backbones.streamvggt_wrapper import (
    materialize_streamvggt_rgb,
)
from streaming_couping.src.semantic_mapping.geometry import world_points_for_frame
from streaming_couping.src.rgb_inputs import selected_positions
from streaming_couping.scripts.generate_horizonstream_geometry_cache import (
    _chunk_schedule,
)


def _payload(image_paths: list[Path]) -> dict:
    world_to_camera = torch.eye(4).repeat(2, 1, 1)[:, :3]
    world_to_camera[1, 0, 3] = -1.0
    intrinsics = torch.eye(3).repeat(2, 1, 1)
    return {
        "schema": HORIZONSTREAM_CACHE_SCHEMA,
        "schema_version": HORIZONSTREAM_CACHE_VERSION,
        "backend": "horizonstream",
        "image_paths": [str(path.resolve()) for path in image_paths],
        "frame_ids": [0, 1],
        "depth": torch.ones(2, 2, 3),
        "confidence": torch.full((2, 2, 3), 0.75),
        "world_to_camera": world_to_camera,
        "intrinsics": intrinsics,
        "processed_rgb": torch.full((2, 2, 3, 3), 128, dtype=torch.uint8),
        "processed_size": (2, 3),
        "source_sizes": [(4, 6), (4, 6)],
        "scale_type": "metric",
        "pose_source": "online_motion_averaged",
        "request": {
            "checkpoint": "/tmp/HorizonStream.pt",
            "settings": {"window_size": 10},
            "image_signatures": image_file_signatures(image_paths),
        },
    }


def test_horizonstream_cache_adapter_and_materialized_rgb(tmp_path: Path) -> None:
    image_paths = []
    for index in range(2):
        path = tmp_path / f"source_{index}.jpg"
        Image.new("RGB", (6, 4), color=(index, 0, 0)).save(path)
        image_paths.append(path)
    payload = _payload(image_paths)

    validate_horizonstream_cache(payload, expected_image_paths=image_paths)
    adapter = HorizonStreamGeometryCacheAdapter(payload)
    aligned_paths = materialize_horizonstream_rgb(payload, tmp_path / "aligned")
    frames = adapter.infer(aligned_paths)

    assert adapter.backend_name == "horizonstream"
    assert adapter.image_size == (2, 3)
    assert tuple(frame.frame_id for frame in frames) == (0, 1)
    assert all(path.is_file() for path in aligned_paths)
    points, valid = world_points_for_frame(frames[1])
    assert bool(valid.all())
    assert torch.allclose(points[0, 0], torch.tensor([1.0, 0.0, 1.0]))
    assert torch.allclose(frames[0].rgb[0, 0], torch.full((3,), 128 / 255))

    assert horizonstream_cache_matches(
        payload,
        image_paths=image_paths,
        checkpoint="/tmp/HorizonStream.pt",
        settings={"window_size": 10},
    )
    assert not horizonstream_cache_matches(
        payload,
        image_paths=image_paths,
        checkpoint="/tmp/HorizonStream.pt",
        settings={"window_size": 20},
    )


def test_horizonstream_cache_rejects_path_order(tmp_path: Path) -> None:
    image_paths = [tmp_path / "a.jpg", tmp_path / "b.jpg"]
    for path in image_paths:
        Image.new("RGB", (6, 4)).save(path)
    payload = _payload(image_paths)
    with pytest.raises(ValueError, match="RGB order differs"):
        validate_horizonstream_cache(
            payload,
            expected_image_paths=list(reversed(image_paths)),
        )


def test_streamvggt_materialize_uses_processed_grid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_paths = []
    for index in range(2):
        path = tmp_path / f"source_{index}.jpg"
        Image.new("RGB", (12, 8), color=(index, 10, 20)).save(path)
        source_paths.append(path)

    processed = torch.zeros(2, 3, 4, 6)
    processed[0, 0] = 1.0
    calls: dict[str, object] = {}

    def fake_loader(paths, *, mode):
        calls["paths"] = tuple(paths)
        calls["mode"] = mode
        return processed.clone()

    fake_package = types.ModuleType("streamvggt")
    fake_package.__path__ = []
    fake_utils = types.ModuleType("streamvggt.utils")
    fake_utils.__path__ = []
    fake_load_fn = types.ModuleType("streamvggt.utils.load_fn")
    fake_load_fn.load_and_preprocess_images = fake_loader
    monkeypatch.setitem(sys.modules, "streamvggt", fake_package)
    monkeypatch.setitem(sys.modules, "streamvggt.utils", fake_utils)
    monkeypatch.setitem(sys.modules, "streamvggt.utils.load_fn", fake_load_fn)

    aligned_paths, processed_size = materialize_streamvggt_rgb(
        source_paths,
        tmp_path / "aligned",
        image_mode="crop",
    )

    assert processed_size == (4, 6)
    assert calls == {
        "paths": tuple(str(path.resolve()) for path in source_paths),
        "mode": "crop",
    }
    assert len(aligned_paths) == 2
    with Image.open(aligned_paths[0]) as image:
        assert image.size == (6, 4)
        assert image.mode == "RGB"


def test_horizonstream_confidence_normalization_is_per_frame() -> None:
    confidence = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[10.0, 20.0], [30.0, 40.0]],
        ]
    )
    normalized, low, high = normalize_horizonstream_confidence(confidence)
    assert tuple(normalized.shape) == (2, 2, 2)
    assert torch.allclose(normalized[0], normalized[1])
    assert torch.all(high > low)
    assert float(normalized.min()) == 0.0
    assert float(normalized.max()) == 1.0


def test_default_frame_selection_and_chunk_schedule() -> None:
    positions = selected_positions(795, start=90, stride=14, count=50)
    assert len(positions) == 50
    assert positions[0] == 90
    assert positions[-1] == 776
    assert _chunk_schedule(50, window_size=10, sliding_size=21) == [
        (0, 10),
        (10, 31),
        (31, 50),
    ]
    assert len(_chunk_schedule(50, window_size=10, sliding_size=1)) == 41
