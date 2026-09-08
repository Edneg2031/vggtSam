"""Tests for reusing V1 pose corrections on object points only."""

from __future__ import annotations

import torch

from streaming_couping.src.semantic_mapping.contracts import (
    GeometryFrame,
    ObjectObservation,
    SegmentationFrame,
)
from streaming_couping.src.semantic_mapping.pipeline import SemanticMapPipeline
from streaming_couping.src.semantic_mapping.v1_object_point_alignment import (
    V1ObjectPointPoseAlignment,
    apply_object_point_pose_transform,
)


def _geometry(frame_id: int, *, x: float = 0.0) -> GeometryFrame:
    points = torch.tensor(
        [
            [[x, 0.0, 1.0], [x + 0.1, 0.0, 1.0]],
            [[x, 0.1, 1.0], [x + 0.1, 0.1, 1.0]],
        ]
    )
    return GeometryFrame(
        frame_id=frame_id,
        image_size=(2, 2),
        world_points=points,
        camera_to_world=torch.eye(4),
        confidence=torch.ones(2, 2),
        valid=torch.ones(2, 2, dtype=torch.bool),
        rgb=torch.ones(2, 2, 3),
        backend="test",
    )


def _segmentation(frame_id: int) -> SegmentationFrame:
    return SegmentationFrame(
        frame_id=frame_id,
        image_size=(2, 2),
        observations=(
            ObjectObservation(
                category="chair",
                instance_id=3,
                mask=torch.ones(2, 2, dtype=torch.bool),
                score=1.0,
                static_score=1.0,
            ),
        ),
        backend="test",
    )


class _GeometryProvider:
    backend_name = "test"

    def __init__(self, frames):
        self.frames = tuple(frames)

    def infer(self, image_paths):
        del image_paths
        return self.frames


class _SegmentationProvider:
    backend_name = "test"

    def __init__(self, frames):
        self.frames = tuple(frames)

    def infer(self, image_paths, prompts=None):
        del image_paths, prompts
        return self.frames


def test_v1_transform_is_refined_times_inverse_raw(tmp_path) -> None:
    artifact = tmp_path / "refined_camera_to_world.pt"
    refined = torch.eye(4).repeat(2, 1, 1)
    refined[1, 0, 3] = 0.5
    torch.save(
        {
            "frame_ids": [0, 1],
            "raw_camera_to_world": torch.eye(4).repeat(2, 1, 1),
            "refined_camera_to_world": refined,
        },
        artifact,
    )
    alignment = V1ObjectPointPoseAlignment.from_artifact(
        artifact,
        (_geometry(0), _geometry(1)),
    )
    point = torch.tensor([[1.0, 2.0, 3.0]])
    assert torch.allclose(
        apply_object_point_pose_transform(alignment.correction_by_frame[1], point),
        torch.tensor([[1.5, 2.0, 3.0]]),
    )
    assert alignment.to_dict()["camera_pose_modified"] is False


def test_v1_alignment_moves_only_static_object_points_and_keeps_raw_pose(tmp_path) -> None:
    artifact = tmp_path / "refined_camera_to_world.pt"
    refined = torch.eye(4).repeat(2, 1, 1)
    refined[1, 0, 3] = 0.5
    torch.save(
        {
            "frame_ids": [0, 1],
            "raw_camera_to_world": torch.eye(4).repeat(2, 1, 1),
            "refined_camera_to_world": refined,
        },
        artifact,
    )
    geometry = (_geometry(0), _geometry(1))
    segmentation = (_segmentation(0), _segmentation(1))
    pipeline = SemanticMapPipeline(
        geometry=_GeometryProvider(geometry),
        segmentation=_SegmentationProvider(segmentation),
    )

    run = pipeline.run_with_v1_object_point_alignment(
        ("frame0.jpg", "frame1.jpg"),
        pose_artifact=artifact,
        prompts=("chair",),
    )
    raw = run.results["raw"]
    aligned = run.results["v1_object_point_alignment"]

    assert raw.metadata["camera_pose_modified"] is False
    assert aligned.metadata["camera_pose_modified"] is False
    assert raw.metadata["pointmap_modified"] is False
    assert aligned.metadata["pointmap_modified"] is True
    assert torch.allclose(raw.object_tracks[0].points[:4], aligned.object_tracks[0].points[:4])
    assert torch.allclose(
        aligned.object_tracks[0].points[4:],
        raw.object_tracks[0].points[4:] + torch.tensor([0.5, 0.0, 0.0]),
    )
    assert torch.equal(geometry[1].camera_to_world, torch.eye(4))
    assert torch.allclose(raw.scene_voxel_points, aligned.scene_voxel_points)
