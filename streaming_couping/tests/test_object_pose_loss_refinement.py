"""Synthetic tests for causal object point-cloud alignment loss."""

from __future__ import annotations

import torch

from streaming_couping.src.semantic_mapping.contracts import (
    GeometryFrame,
    ObjectObservation,
    SegmentationFrame,
)
from streaming_couping.src.semantic_mapping.mapping import (
    SemanticMapBuilder,
    SemanticMapConfig,
)
from streaming_couping.src.semantic_mapping.object_pose_loss_refinement import (
    ObjectPoseLossRefinementConfig,
    ObjectPoseLossRefiner,
)
from streaming_couping.src.semantic_mapping.pipeline import SemanticMapPipeline


def _object_mask(size: tuple[int, int] = (6, 6)) -> torch.Tensor:
    mask = torch.zeros(size, dtype=torch.bool)
    mask[1:5, 1:4] = True
    return mask


def _world_points(shift: torch.Tensor | None = None) -> torch.Tensor:
    height, width = 6, 6
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    points = torch.stack((0.20 * xx, 0.20 * yy, torch.ones_like(xx)), dim=-1)
    if shift is not None:
        points[_object_mask()] += shift
    return points


def _geometry(frame_id: int, points: torch.Tensor) -> GeometryFrame:
    size = tuple(int(value) for value in points.shape[:2])
    return GeometryFrame(
        frame_id=frame_id,
        image_size=size,
        world_points=points,
        camera_to_world=torch.eye(4),
        confidence=torch.ones(size),
        valid=torch.ones(size, dtype=torch.bool),
        rgb=torch.ones(*size, 3),
        backend="synthetic_horizonstream",
    )


def _segmentation(frame_id: int, instance_id: int = 7) -> SegmentationFrame:
    return SegmentationFrame(
        frame_id=frame_id,
        image_size=(6, 6),
        observations=(
            ObjectObservation(
                category="chair",
                instance_id=instance_id,
                mask=_object_mask(),
                score=0.95,
                static_score=1.0,
            ),
        ),
        backend="synthetic_sam3",
    )


def _config(**overrides) -> ObjectPoseLossRefinementConfig:
    values = dict(
        anchor_frame_count=1,
        max_anchor_observations=1,
        max_history_observations=1,
        max_points_per_observation=64,
        min_points_per_observation=4,
        min_mask_pixels=4,
        min_matches_per_pair=6,
        min_total_matches=6,
        max_match_distance_m=0.30,
        trim_ratio=0.80,
        outer_iterations=3,
        optimizer_steps=80,
        learning_rate=0.02,
        huber_delta_m=0.10,
        pose_prior_weight=0.0,
        max_correction_rotation_deg=5.0,
        max_correction_translation_m=0.30,
        min_relative_loss_improvement=0.02,
        device="cpu",
    )
    values.update(overrides)
    return ObjectPoseLossRefinementConfig(**values)


def test_loss_refiner_pulls_current_pose_toward_fixed_anchor() -> None:
    shift = torch.tensor([0.08, -0.05, 0.03])
    geometry = (
        _geometry(0, _world_points()),
        _geometry(1, _world_points(shift)),
    )
    segmentation = (_segmentation(0), _segmentation(1))

    result = ObjectPoseLossRefiner(_config()).refine(
        geometry,
        segmentation,
        image_paths=("unused_0.jpg", "unused_1.jpg"),
    )

    assert len(result.accepted_edges) == 1
    assert result.summary["anchor_frame_count"] == 1
    assert result.summary["raw_observation_count"] == 2
    assert result.summary["retained_observation_count"] == 2
    assert result.summary["filtered_observation_count"] == 0
    assert torch.allclose(
        result.refined_camera_to_world[0],
        result.raw_camera_to_world[0],
    )
    correction = result.refined_camera_to_world[1][:3, 3]
    assert torch.allclose(correction, -shift, atol=2e-2)
    edge = result.accepted_edges[0]
    assert edge.frame_i == 0
    assert edge.frame_j == 1
    assert edge.instance_id == 7
    assert edge.final_loss_m < edge.initial_loss_m
    assert edge.relative_loss_improvement > 0.02


def test_loss_refiner_keeps_pose_when_instance_has_no_history() -> None:
    geometry = (
        _geometry(0, _world_points()),
        _geometry(1, _world_points(torch.tensor([0.08, 0.0, 0.0]))),
    )
    segmentation = (_segmentation(0, instance_id=7), _segmentation(1, instance_id=8))

    result = ObjectPoseLossRefiner(_config()).refine(
        geometry,
        segmentation,
        image_paths=(),
    )

    assert not result.accepted_edges
    assert result.summary["accepted_edge_count"] == 0
    assert result.summary["optimization_attempted_frame_count"] == 0
    assert torch.equal(
        result.refined_camera_to_world[1],
        result.raw_camera_to_world[1],
    )
    assert (
        result.summary["frame_diagnostics"][-1]["reason"]
        == "no_historical_instance_reference"
    )


class _StaticGeometryProvider:
    backend_name = "synthetic_horizonstream"

    def __init__(self, frames: tuple[GeometryFrame, ...]) -> None:
        self.frames = frames

    def infer(self, image_paths):
        assert len(image_paths) == len(self.frames)
        return self.frames


class _StaticSegmentationProvider:
    backend_name = "synthetic_sam3"

    def __init__(self, frames: tuple[SegmentationFrame, ...]) -> None:
        self.frames = frames

    def infer(self, image_paths, prompts=None):
        del prompts
        assert len(image_paths) == len(self.frames)
        return self.frames


def test_loss_refinement_pipeline_keeps_object_only_raw_and_refined_maps() -> None:
    shift = torch.tensor([0.08, 0.0, 0.0])
    geometry = (
        _geometry(0, _world_points()),
        _geometry(1, _world_points(shift)),
    )
    segmentation = (_segmentation(0), _segmentation(1))
    pipeline = SemanticMapPipeline(
        geometry=_StaticGeometryProvider(geometry),
        segmentation=_StaticSegmentationProvider(segmentation),
        mapper=SemanticMapBuilder(
            SemanticMapConfig(
                fusion_policy="raw",
                object_only=True,
                voxel_size_m=0.02,
                max_points_per_observation=64,
            )
        ),
    )

    run = pipeline.run_with_object_pose_refinement(
        ("frame_0.jpg", "frame_1.jpg"),
        refiner=ObjectPoseLossRefiner(_config()),
        prompts=("chair",),
    )

    assert set(run.raw_results) == {"raw"}
    assert set(run.refined_results) == {"raw"}
    raw = run.raw_results["raw"]
    refined = run.refined_results["raw"]
    assert raw.scene_voxel_count == 0
    assert refined.scene_voxel_count == 0
    assert len(raw.object_tracks) == 1
    assert len(refined.object_tracks) == 1
    assert raw.metadata["object_only"] is True
    assert refined.metadata["pose_variant"] == "object_pose_refined"
    assert refined.metadata["object_pose_refinement"]["enabled"] is True

