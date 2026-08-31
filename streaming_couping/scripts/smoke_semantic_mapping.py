#!/usr/bin/env python3
"""CPU smoke checks for the backend-neutral semantic mapping pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile

import torch

from streaming_couping.src.semantic_mapping.adapters import (
    SAM31SegmentationAdapter,
    V0CacheGeometryAdapter,
    V0CacheSegmentationAdapter,
)
from streaming_couping.src.semantic_mapping.contracts import (
    GeometryFrame,
    ObjectObservation,
    SegmentationFrame,
)
from streaming_couping.src.semantic_mapping.export import export_semantic_map
from streaming_couping.src.semantic_mapping.geometry import world_points_for_frame
from streaming_couping.src.semantic_mapping.mapping import (
    SemanticMapBuilder,
    SemanticMapConfig,
)
from streaming_couping.src.semantic_mapping.pipeline import SemanticMapPipeline


@dataclass
class _FakeGeometryProvider:
    frames: tuple[GeometryFrame, ...]
    backend_name: str = "fake_geometry"

    def infer(self, image_paths):
        assert len(image_paths) == len(self.frames)
        return self.frames


@dataclass
class _FakeSegmentationProvider:
    frames: tuple[SegmentationFrame, ...]
    backend_name: str = "fake_segmentation"

    def infer(self, image_paths, prompts=None):
        assert len(image_paths) == len(self.frames)
        return self.frames


class _FakeSAMWrapper:
    def __init__(self, masks, scores, obj_ids, birth_indices):
        self.masks = masks
        self.scores = scores
        self.obj_ids = obj_ids
        self.birth_indices = birth_indices

    def track_all_forward(self, image_paths, *, prompt, output_size, max_objects):
        del image_paths, prompt, output_size, max_objects
        return self


def main() -> None:
    height, width = 2, 3
    points = torch.tensor(
        [
            [
                [[0.10, 0.00, 1.00], [0.20, 0.00, 1.00], [1.10, 0.00, 1.00]],
                [[0.10, 0.20, 1.00], [0.20, 0.20, 1.00], [1.10, 0.20, 1.00]],
            ],
            [
                [[0.10, 0.00, 1.00], [0.20, 0.00, 1.00], [1.10, 0.00, 1.00]],
                [[0.10, 0.20, 1.00], [0.20, 0.20, 1.00], [1.10, 0.20, 1.00]],
            ],
        ],
        dtype=torch.float32,
    )
    rgb = torch.full((height, width, 3), 0.5)
    geometry_frames = tuple(
        GeometryFrame(
            frame_id=frame_id,
            image_size=(height, width),
            world_points=points[frame_id],
            confidence=torch.ones(height, width),
            valid=torch.ones(height, width, dtype=torch.bool),
            rgb=rgb,
            backend="fake_geometry",
        )
        for frame_id in range(2)
    )
    static_mask = torch.tensor([[1, 1, 0], [1, 1, 0]], dtype=torch.bool)
    dynamic_mask = torch.tensor([[0, 0, 1], [0, 0, 1]], dtype=torch.bool)
    segmentation_frames = tuple(
        SegmentationFrame(
            frame_id=frame_id,
            image_size=(height, width),
            observations=(
                ObjectObservation(
                    category="chair",
                    instance_id=7,
                    mask=static_mask,
                    score=0.9,
                    static_score=0.9,
                ),
                ObjectObservation(
                    category="person",
                    instance_id=9,
                    mask=dynamic_mask,
                    score=0.9,
                    static_score=0.1,
                ),
            ),
            backend="fake_segmentation",
        )
        for frame_id in range(2)
    )
    pipeline = SemanticMapPipeline(
        geometry=_FakeGeometryProvider(geometry_frames),
        segmentation=_FakeSegmentationProvider(segmentation_frames),
        mapper=SemanticMapBuilder(
            SemanticMapConfig(
                voxel_size_m=0.5,
                min_geometry_confidence=0.5,
                min_track_score=0.5,
                static_score_threshold=0.5,
                include_dynamic_tracks=True,
                max_points_per_observation=32,
                max_points_per_track=64,
            )
        ),
    )
    result = pipeline.run(
        (Path("frame_0.jpg"), Path("frame_1.jpg")),
        prompts=("chair", "person"),
    )
    assert result.voxel_count > 0
    assert result.scene_voxel_count == 2
    assert result.scene_labeled_voxel_count == 1
    assert set(result.scene_instance_ids.tolist()) == {-1, 7}
    assert result.labeled_voxel_count == result.voxel_count
    assert set(result.semantic_labels) == {"chair"}
    assert [track.instance_id for track in result.object_tracks] == [7, 9]
    assert result.object_tracks[0].is_static is True
    assert result.object_tracks[1].is_static is False
    assert set(result.instance_ids.tolist()) == {7}
    assert result.metadata["causal_fusion"] is True

    depth_frame = GeometryFrame(
        frame_id=0,
        image_size=(2, 2),
        depth=torch.ones(2, 2),
        intrinsics=torch.eye(3),
        camera_to_world=torch.eye(4),
        confidence=torch.ones(2, 2),
        backend="fake_depth_geometry",
    )
    depth_points, depth_valid = world_points_for_frame(depth_frame)
    assert torch.allclose(depth_points[0, 0], torch.tensor([0.0, 0.0, 1.0]))
    assert torch.allclose(depth_points[0, 1], torch.tensor([1.0, 0.0, 1.0]))
    assert bool(depth_valid.all())

    fake_masks = torch.zeros(2, 1, 2, 2, dtype=torch.bool)
    fake_masks[:, 0, 0, 0] = True
    sam_adapter = SAM31SegmentationAdapter(
        _FakeSAMWrapper(
            fake_masks,
            torch.ones(2, 1),
            obj_ids=(4,),
            birth_indices=(0,),
        ),
        output_size=(2, 2),
        max_objects_per_prompt=2,
        max_total_objects=2,
        min_birth_pixels=1,
    )
    sam_frames = sam_adapter.infer((Path("frame_0.jpg"), Path("frame_1.jpg")), ("box",))
    assert len(sam_frames) == 2
    assert sam_frames[0].observations[0].category == "box"
    assert sam_frames[0].observations[0].source_track_id == 4

    cache_payload = {
        "frame_indices": [90, 105],
        "baseline_world_points": points,
        "baseline_world_confidence": torch.ones(2, height, width),
        "stream_images": rgb.permute(2, 0, 1)[None].expand(2, -1, -1, -1),
        "tracking_masks_stream": torch.stack(
            [torch.stack((static_mask, dynamic_mask), dim=0)] * 2,
            dim=0,
        ),
        "tracking_scores": torch.full((2, 2), 0.9),
        "quality": torch.tensor(
            [
                [[0.9, 0.9, 0.9], [0.9, 0.9, 0.1]],
                [[0.9, 0.9, 0.9], [0.9, 0.9, 0.1]],
            ]
        ),
        "instance_ids": [7, 9],
        "sam_track_ids": [101, 102],
        "sam_track_prompts": ["chair", "person"],
        "sam_birth_indices": [0, 0],
    }
    cached_geometry = V0CacheGeometryAdapter(cache_payload)
    cached_segmentation = V0CacheSegmentationAdapter(cache_payload)
    assert tuple(frame.frame_id for frame in cached_geometry.infer((Path("a"), Path("b")))) == (90, 105)
    assert len(cached_segmentation.infer((Path("a"), Path("b")))[0].observations) == 2

    with tempfile.TemporaryDirectory(prefix="semantic_mapping_smoke_") as directory:
        summary = export_semantic_map(result, directory)
        assert summary["voxel_count"] == result.voxel_count
        assert summary["scene_voxel_count"] == result.scene_voxel_count
        for name in (
            "semantic_map.pt",
            "scene_rgb_map.ply",
            "scene_semantic_map.ply",
            "semantic_map.ply",
            "rgb_map.ply",
            "object_tracks.ply",
            "object_tracks.json",
            "map_summary.json",
        ):
            assert (Path(directory) / name).is_file(), name
        try:
            artifact = torch.load(
                Path(directory) / "semantic_map.pt",
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            artifact = torch.load(
                Path(directory) / "semantic_map.pt",
                map_location="cpu",
            )
        assert artifact["schema"] == 2
        assert artifact["scene_voxel_points"].shape[0] == result.scene_voxel_count
        assert "frame_id" in (
            Path(directory) / "object_tracks.ply"
        ).read_bytes().split(b"end_header", 1)[0].decode("ascii")
    print(
        "semantic mapping backend-neutral smoke passed "
        f"frames={result.metadata['frame_count']} "
        f"voxels={result.voxel_count} objects={len(result.object_tracks)}"
    )


if __name__ == "__main__":
    main()
