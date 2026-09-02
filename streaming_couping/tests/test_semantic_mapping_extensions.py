"""Tests for the opt-in causal guidance and map-write memory layers."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from streaming_couping.src.semantic_mapping.adapters import (
    GeometryAwareSAM31SegmentationAdapter,
)
from streaming_couping.src.semantic_mapping.contracts import (
    GeometryFrame,
    ObjectObservation,
    SegmentationFrame,
)
from streaming_couping.src.semantic_mapping.geometry_guidance import (
    CausalObjectGeometryMemory,
    GeometryGuidanceConfig,
    build_geometry_prompt,
)
from streaming_couping.src.semantic_mapping.mapping import (
    MapWriteGateConfig,
    SemanticMapBuilder,
    SemanticMapConfig,
)
from streaming_couping.src.types import SAM3MaskCandidate


def _geometry_frames(count: int = 3, size: tuple[int, int] = (8, 8)):
    height, width = size
    points = torch.zeros(height, width, 3)
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    points[..., 0] = 2.0 * xx
    points[..., 1] = 2.0 * yy
    points[..., 2] = 2.0
    return tuple(
        GeometryFrame(
            frame_id=frame,
            image_size=size,
            world_points=points,
            intrinsics=torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            ),
            camera_to_world=torch.eye(4),
            confidence=torch.ones(size),
            valid=torch.ones(size, dtype=torch.bool),
            backend="fake_geometry",
        )
        for frame in range(count)
    )


def _mask(box: tuple[slice, slice], size: tuple[int, int] = (8, 8)) -> torch.Tensor:
    output = torch.zeros(size, dtype=torch.bool)
    output[box] = True
    return output


def test_geometry_prompt_uses_only_previous_world_points() -> None:
    config = GeometryGuidanceConfig(
        min_history_points=4,
        max_positive_points=4,
    )
    memory = CausalObjectGeometryMemory(config)
    frames = _geometry_frames(3)
    object_mask = _mask((slice(2, 5), slice(2, 5)))
    memory.update(4, frames[0], object_mask, frame_id=0, score=0.9)

    history = memory.get(4, current_frame_id=1)
    prompt = build_geometry_prompt(history, frames[1], config=config)

    assert prompt is not None
    assert prompt.source_frame_ids == (0,)
    assert prompt.projected_points == 9
    assert int(prompt.positive_mask.sum()) > 0
    assert bool(prompt.box_mask[2:5, 2:5].any())

    # A future frame is not inserted into the history before frame 1 is
    # queried.  This is the invariant the adapter relies on for causality.
    memory.update(4, frames[1], object_mask, frame_id=1, score=0.9)
    later = memory.get(4, current_frame_id=2)
    assert later is not None
    assert max(later.source_frame_ids) == 1


class _FakeGeometryAwareSAM:
    def __init__(self, raw_masks: torch.Tensor, candidate_mask: torch.Tensor):
        self.raw_masks = raw_masks
        self.candidate_mask = candidate_mask
        self.calls: list[dict[str, object]] = []

    def track_all_forward(self, image_paths, *, prompt, output_size, max_objects):
        del image_paths, prompt, output_size, max_objects
        return SimpleNamespace(
            masks=self.raw_masks,
            scores=torch.full(self.raw_masks.shape[:2], 0.9),
            obj_ids=(7,),
            birth_indices=(0,),
        )

    def propose_geometry_prompted_masks(self, image_path, **kwargs):
        self.calls.append({"image_path": str(image_path), **kwargs})
        return [SAM3MaskCandidate(99, self.candidate_mask, 0.9)]


def test_geometry_aware_adapter_keeps_ids_and_replaces_only_after_gate() -> None:
    target = _mask((slice(2, 5), slice(2, 5)))
    wrong = _mask((slice(5, 8), slice(5, 8)))
    raw_masks = torch.stack((target, wrong, target), dim=0)[:, None]
    wrapper = _FakeGeometryAwareSAM(raw_masks, target)
    adapter = GeometryAwareSAM31SegmentationAdapter(
        wrapper,
        output_size=(8, 8),
        max_objects_per_prompt=2,
        max_total_objects=2,
        min_birth_pixels=1,
        geometry_config=GeometryGuidanceConfig(
            min_history_points=4,
            min_candidate_support_recall=0.2,
        ),
    )

    frames = adapter.infer_with_geometry(
        ("frame0.jpg", "frame1.jpg", "frame2.jpg"),
        _geometry_frames(3),
        prompts=("chair",),
    )

    assert len(wrapper.calls) == 2
    assert [observation.instance_id for frame in frames for observation in frame.observations] == [
        0,
        0,
        0,
    ]
    assert torch.equal(frames[1].observations[0].mask, target)
    assert torch.equal(frames[2].observations[0].mask, target)
    assert adapter.last_summary["applied_count"] == 1
    assert adapter.last_diagnostics[1]["history_last_frame_id"] == 0
    assert adapter.last_diagnostics[1]["history_last_frame_id"] < 1


class _FailingGeometryAwareSAM(_FakeGeometryAwareSAM):
    def propose_geometry_prompted_masks(self, image_path, **kwargs):
        del image_path, kwargs
        raise RuntimeError("fake prompt failure")


def test_geometry_prompt_failure_falls_back_to_raw_mask() -> None:
    target = _mask((slice(2, 5), slice(2, 5)))
    wrong = _mask((slice(5, 8), slice(5, 8)))
    raw_masks = torch.stack((target, wrong), dim=0)[:, None]
    adapter = GeometryAwareSAM31SegmentationAdapter(
        _FailingGeometryAwareSAM(raw_masks, target),
        output_size=(8, 8),
        max_objects_per_prompt=2,
        max_total_objects=2,
        min_birth_pixels=1,
        geometry_config=GeometryGuidanceConfig(min_history_points=4),
    )

    frames = adapter.infer_with_geometry(
        ("frame0.jpg", "frame1.jpg"),
        _geometry_frames(2),
        prompts=("chair",),
    )

    assert torch.equal(frames[1].observations[0].mask, wrong)
    assert adapter.last_diagnostics[1]["reason"] == "keep_raw:prompt_error"
    assert adapter.last_diagnostics[1]["correction_applied"] == 0


def test_map_write_gate_defers_first_observation_but_preserves_track() -> None:
    size = (4, 4)
    points = torch.zeros(*size, 3)
    yy, xx = torch.meshgrid(
        torch.arange(size[0], dtype=torch.float32),
        torch.arange(size[1], dtype=torch.float32),
        indexing="ij",
    )
    points[..., 0] = xx
    points[..., 1] = yy
    points[..., 2] = 1.0
    geometry = tuple(
        GeometryFrame(
            frame_id=frame,
            image_size=size,
            world_points=points,
            confidence=torch.ones(size),
            valid=torch.ones(size, dtype=torch.bool),
            rgb=torch.ones(*size, 3),
            backend="fake_geometry",
        )
        for frame in range(2)
    )
    observation = ObjectObservation(
        category="chair",
        instance_id=3,
        mask=torch.ones(size, dtype=torch.bool),
        score=0.9,
    )
    segmentation = tuple(
        SegmentationFrame(
            frame_id=frame,
            image_size=size,
            observations=(observation,),
            backend="fake_segmentation",
        )
        for frame in range(2)
    )
    builder = SemanticMapBuilder(
        SemanticMapConfig(
            voxel_size_m=0.5,
            map_write_gate=MapWriteGateConfig(
                enabled=True,
                min_mask_pixels=1,
                min_observation_points=1,
                min_observations_before_write=2,
            ),
        )
    )

    first = builder.update(geometry[0], segmentation[0])
    second = builder.update(geometry[1], segmentation[1])
    result = builder.finalize()

    assert first.map_write_observation_count == 0
    assert second.map_write_observation_count == 1
    assert first.map_gate_rejected_count == 1
    assert result.voxel_count > 0
    assert result.object_tracks[0].observations == 2
    assert result.object_tracks[0].map_write_observations == 1
    assert result.metadata["map_write_gate"]["enabled"] is True
    assert result.metadata["object_memory"]["instance_count"] == 1


def test_dynamic_observation_is_not_counted_as_semantic_map_write() -> None:
    size = (4, 4)
    points = torch.zeros(*size, 3)
    yy, xx = torch.meshgrid(
        torch.arange(size[0], dtype=torch.float32),
        torch.arange(size[1], dtype=torch.float32),
        indexing="ij",
    )
    points[..., 0] = xx
    points[..., 1] = yy
    points[..., 2] = 1.0
    geometry = GeometryFrame(
        frame_id=0,
        image_size=size,
        world_points=points,
        confidence=torch.ones(size),
        valid=torch.ones(size, dtype=torch.bool),
        backend="fake_geometry",
    )
    segmentation = SegmentationFrame(
        frame_id=0,
        image_size=size,
        observations=(
            ObjectObservation(
                category="chair",
                instance_id=3,
                mask=torch.ones(size, dtype=torch.bool),
                score=0.9,
                static_score=1.0,
            ),
            ObjectObservation(
                category="person",
                instance_id=4,
                mask=torch.ones(size, dtype=torch.bool),
                score=0.9,
                static_score=0.0,
            ),
        ),
        backend="fake_segmentation",
    )
    builder = SemanticMapBuilder(
        SemanticMapConfig(
            static_score_threshold=0.5,
            map_write_gate=MapWriteGateConfig(enabled=False),
        )
    )
    builder.update(geometry, segmentation)
    map_result = builder.finalize()

    assert map_result.metadata["map_write_gate"]["write_count"] == 1
    assert map_result.metadata["map_write_gate"]["dynamic_observation_count"] == 1
    states = {
        int(state["instance_id"]): state
        for state in map_result.metadata["object_memory"]["states"]
    }
    assert states[3]["map_write_count"] == 1
    assert states[4]["map_write_count"] == 0
