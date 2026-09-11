"""The mask-oracle control: does GT identity/mask explain the proposal error?

The oracle deliberately bypasses the segmentation model, so these tests pin two
things: that it really does feed the GT instances through (identity and label,
visibility per frame), and that its output is labelled as a diagnostic upper
bound so it cannot be mistaken for a method result.
"""

from __future__ import annotations

import torch

from streaming_couping.scripts.run_object_pose_loss_oracle import (
    ORACLE_MASK_SOURCE,
    build_segmentation_frames,
)
from streaming_couping.src.semantic_mapping.object_pose_loss_refinement import (
    ObjectPoseLossRefiner,
)
from streaming_couping.src.semantic_tracking_metrics import GroundTruthInstances
from streaming_couping.tests.test_object_pose_loss_replay import (
    _config,
    _geometry,
    _mask,
)


def _ground_truth(
    *, frame_count: int = 8, second_instance_from: int = 3
) -> GroundTruthInstances:
    masks = torch.zeros(frame_count, 2, 6, 6, dtype=torch.bool)
    for frame in range(frame_count):
        masks[frame, 0] = _mask(slice(0, 3))
        if frame >= second_instance_from:
            masks[frame, 1] = _mask(slice(3, 6))
    return GroundTruthInstances(
        masks=masks,
        instance_ids=(101, 202),
        labels=("bed", "rug"),
        all_visible_instance_ids=(101, 202),
    )


def test_gt_instances_become_observations_with_their_identity() -> None:
    frames = build_segmentation_frames(
        ground_truth=_ground_truth(),
        frame_ids=tuple(range(8)),
        image_size=(6, 6),
    )
    assert len(frames) == 8
    assert all(frame.backend == ORACLE_MASK_SOURCE for frame in frames)
    # the second instance appears only from its first visible frame
    assert tuple(o.instance_id for o in frames[0].observations) == (101,)
    assert tuple(o.instance_id for o in frames[4].observations) == (101, 202)
    assert tuple(o.category for o in frames[4].observations) == ("bed", "rug")


def test_scores_are_maximal_so_refiner_thresholds_cannot_reject_the_oracle() -> None:
    """A low score would let the refiner's own gates filter GT masks and make
    the control uninterpretable."""

    frames = build_segmentation_frames(
        ground_truth=_ground_truth(), frame_ids=tuple(range(8)), image_size=(6, 6)
    )
    for observation in frames[4].observations:
        assert observation.score == 1.0
        assert observation.static_score == 1.0
        assert observation.mask.dtype == torch.bool


def test_invisible_instances_are_omitted() -> None:
    frames = build_segmentation_frames(
        ground_truth=_ground_truth(), frame_ids=tuple(range(8)), image_size=(6, 6)
    )
    # frame 0 has only instance 101 present
    assert len(frames[0].observations) == 1


def test_no_eligible_instance_yields_empty_frames() -> None:
    empty = GroundTruthInstances(
        masks=torch.zeros(4, 0, 6, 6, dtype=torch.bool),
        instance_ids=(),
        labels=(),
        all_visible_instance_ids=(),
    )
    frames = build_segmentation_frames(
        ground_truth=empty, frame_ids=tuple(range(4)), image_size=(6, 6)
    )
    assert len(frames) == 4
    assert all(frame.observations == () for frame in frames)


def test_refine_runs_on_oracle_masks_and_labels_the_output() -> None:
    """End to end through the production refiner, with the audit flags."""

    frame_count = 8
    geometry = [_geometry(frame_id, 0.02 * frame_id) for frame_id in range(frame_count)]
    segmentation = build_segmentation_frames(
        ground_truth=_ground_truth(frame_count=frame_count),
        frame_ids=tuple(range(frame_count)),
        image_size=(6, 6),
    )
    result = ObjectPoseLossRefiner(_config()).refine(
        geometry, segmentation, [f"frame_{index}.png" for index in range(frame_count)]
    )
    diagnostics = result.feedback_diagnostics
    assert diagnostics is not None
    # the oracle fed the refiner: identity comes from GT, so both instances are
    # tracked from the frames they appear in
    assert sum(len(rows) for rows in [diagnostics["observations"]]) > 0
    instance_ids = {int(row["instance_id"]) for row in diagnostics["observations"]}
    assert instance_ids == {101, 202}
    assert diagnostics["proposals"], "oracle run produced no proposals"


def test_oracle_audit_block_is_attached_by_the_script() -> None:
    """The flag that keeps an oracle from being reported as a method result."""

    from streaming_couping.scripts.run_object_pose_loss_oracle import main as _  # noqa: F401

    # The block is composed in main(); assert the contract it must carry so a
    # future edit cannot quietly drop the disclosure.
    required = {
        "mask_source": ORACLE_MASK_SOURCE,
        "gt_used_for_proposals": True,
        "purpose": "diagnostic_upper_bound_only",
        "not_a_method_result": True,
        "segmentation_model_bypassed": True,
        "gpu_used": False,
    }
    import inspect

    from streaming_couping.scripts import run_object_pose_loss_oracle as module

    source = inspect.getsource(module.main)
    for key in required:
        assert key in source, f"oracle audit field {key!r} missing from main()"
