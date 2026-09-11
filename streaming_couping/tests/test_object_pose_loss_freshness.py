"""Reference-freshness and factorized-proposal behavior in the loss refiner.

The 100-frame attribution found the proposal error is dominated by how old the
reference set is (the permanent frame-0 anchors, median age 37.5 frames, carry
rho = +0.785 with the GT correction error inside every object category), so
these knobs are the proposal-side fix.  Both default to the original behavior,
which the first test pins down.
"""

from __future__ import annotations

import torch

from streaming_couping.src.semantic_mapping.object_pose_loss_refinement import (
    PROPOSAL_MODES,
    ObjectPoseLossRefinementConfig,
    ObjectPoseLossRefiner,
    _MatchSet,
    _reference_is_fresh,
    _select_references,
    _StoredObjectCloud,
    _TensorPair,
)


def _config(**overrides) -> ObjectPoseLossRefinementConfig:
    return ObjectPoseLossRefinementConfig(**overrides).validate()


def _cloud(frame_id: int, *, role: str = "anchor", instance_id: int = 0):
    generator = torch.Generator().manual_seed(frame_id)
    points = torch.rand((24, 3), generator=generator) * 0.2
    return _StoredObjectCloud(
        frame_id=int(frame_id),
        instance_id=int(instance_id),
        category="bed",
        points_world=points,
        weights=torch.ones(24),
        role=role,
        quality=0.9,
    )


# ---------------------------------------------------------------------------
# Reference freshness.
# ---------------------------------------------------------------------------


def test_defaults_keep_the_permanent_anchor_behavior() -> None:
    """max_reference_age_frames = 0 must not drop anything."""

    config = _config()
    stale = _cloud(0)
    assert _reference_is_fresh(stale, current_frame_id=500, config=config)
    references = _select_references(
        0,
        {0: (_cloud(0), _cloud(1))},
        {0: (_cloud(48, role="history"),)},
        current_frame_id=50,
        config=config,
    )
    assert [reference.frame_id for reference in references] == [0, 1, 48]


def test_age_cap_drops_stale_references() -> None:
    config = _config(max_reference_age_frames=10)
    assert _reference_is_fresh(_cloud(45), current_frame_id=50, config=config)
    assert not _reference_is_fresh(_cloud(39), current_frame_id=50, config=config)
    references = _select_references(
        0,
        {0: (_cloud(0), _cloud(2))},
        {0: (_cloud(45, role="history"),)},
        current_frame_id=50,
        config=config,
    )
    # only the fresh history survives; both frame-0 anchors are dropped
    assert [reference.frame_id for reference in references] == [45]


def test_age_cap_leaves_no_references_when_everything_is_stale() -> None:
    config = _config(max_reference_age_frames=2)
    references = _select_references(
        0,
        {0: (_cloud(0),)},
        {0: (_cloud(1, role="history"),)},
        current_frame_id=50,
        config=config,
    )
    assert references == ()


def test_anchor_refresh_moves_the_anchor_epoch() -> None:
    """With a refresh interval the anchors are re-taken, not pinned."""

    config = _config(anchor_frame_count=3, anchor_refresh_interval_frames=20)
    refiner = ObjectPoseLossRefiner(config)
    frame_ids = list(range(0, 45))
    poses = [torch.eye(4) for _ in frame_ids]

    # Drive the storage loop the way refine() does, without needing proposals:
    # an observation per frame on instance 0, all rejected (no references yet).
    anchors: dict[int, list[_StoredObjectCloud]] = {}
    anchor_refresh = int(config.anchor_refresh_interval_frames)
    seen_anchor_frames: list[int] = []
    for sequence_index in frame_ids:
        if anchor_refresh > 0:
            phase = sequence_index % anchor_refresh
            is_anchor = phase < int(config.anchor_frame_count)
            if phase == 0 and sequence_index >= int(config.anchor_frame_count):
                anchors.clear()
        else:
            is_anchor = sequence_index < int(config.anchor_frame_count)
        if is_anchor:
            seen_anchor_frames.append(sequence_index)
            anchors.setdefault(0, []).append(_cloud(sequence_index))
            anchors[0] = anchors[0][: int(config.max_anchor_observations)]

    # epochs start at 0 and 20, each contributing its first frames
    assert seen_anchor_frames == [0, 1, 2, 20, 21, 22, 40, 41, 42]
    # the surviving anchors come from the newest epoch, not from frame 0
    assert [cloud.frame_id for cloud in anchors[0]] == [40, 41, 42]
    assert refiner is not None


def test_anchor_refresh_requires_at_least_anchor_frame_count() -> None:
    import pytest

    with pytest.raises(ValueError, match="anchor_refresh_interval_frames"):
        _config(anchor_frame_count=5, anchor_refresh_interval_frames=4)


def test_negative_freshness_values_are_rejected() -> None:
    import pytest

    with pytest.raises(ValueError, match="max_reference_age_frames"):
        _config(max_reference_age_frames=-1)


# ---------------------------------------------------------------------------
# Proposal parameterization.
# ---------------------------------------------------------------------------


def test_proposal_mode_must_be_known() -> None:
    import pytest

    with pytest.raises(ValueError, match="proposal_mode"):
        _config(proposal_mode="rot_then_trans")
    assert _config(proposal_mode="rotation_then_translation").proposal_mode in (
        PROPOSAL_MODES
    )


def _tensor_pair(*, offset: float = 0.05) -> _TensorPair:
    generator = torch.Generator().manual_seed(0)
    reference = torch.rand((60, 3), generator=generator) * 0.4
    return _TensorPair(
        current_points=reference + offset,
        current_weights=torch.ones(60),
        reference_points=reference,
        reference_weights=torch.ones(60),
        pair_weight=1.0,
    )


def _stage_fixture(mode: str):
    config = _config(proposal_mode=mode, outer_iterations=2, optimizer_steps=15)
    refiner = ObjectPoseLossRefiner(config)
    pairs = (_tensor_pair(),)
    raw_pose = torch.eye(4)
    delta = torch.nn.Parameter(torch.zeros(6, dtype=torch.float32))
    match_sets = refiner._collect_match_sets(
        refiner._left_updated_pose(delta.detach(), raw_pose), pairs
    )
    return refiner, delta, raw_pose, pairs, match_sets


def test_rotation_stage_freezes_translation() -> None:
    """Adam keeps warm moments, so freezing must be structural, not gradient."""

    refiner, delta, raw_pose, pairs, match_sets = _stage_fixture("joint")
    delta.data[3:] = 0.01  # a non-zero translation the stage must not touch
    before = delta.detach()[3:].clone()
    refiner._run_optimizer_stage(
        delta, raw_pose, pairs, match_sets, indices=(0, 1, 2)
    )
    assert torch.equal(delta.detach()[3:], before)


def test_translation_stage_freezes_rotation() -> None:
    refiner, delta, raw_pose, pairs, match_sets = _stage_fixture("joint")
    delta.data[:3] = 0.02
    before = delta.detach()[:3].clone()
    refiner._run_optimizer_stage(
        delta, raw_pose, pairs, match_sets, indices=(3, 4, 5)
    )
    assert torch.equal(delta.detach()[:3], before)


def test_both_stages_actually_move_their_half() -> None:
    """A stage that changes nothing would make the two modes indistinguishable."""

    config = _config(optimizer_steps=15)
    refiner = ObjectPoseLossRefiner(config)
    pairs = (_tensor_pair(offset=0.05),)
    raw_pose = torch.eye(4)

    for indices in ((0, 1, 2), (3, 4, 5)):
        delta = torch.nn.Parameter(torch.zeros(6, dtype=torch.float32))
        match_sets = refiner._collect_match_sets(
            refiner._left_updated_pose(delta.detach(), raw_pose), pairs
        )
        refiner._run_optimizer_stage(
            delta, raw_pose, pairs, match_sets, indices=indices
        )
        moved = float(torch.linalg.vector_norm(delta.detach()[list(indices)]))
        assert moved > 0.0, indices
        assert delta.detach()[list({0, 1, 2, 3, 4, 5} - set(indices))].abs().max() == 0


def test_factorized_schedule_differs_from_joint() -> None:
    """The two parameterizations must not silently coincide."""

    config = _config(outer_iterations=4, optimizer_steps=25)
    refiner = ObjectPoseLossRefiner(config)
    pairs = (_tensor_pair(offset=0.05),)
    raw_pose = torch.eye(4)
    joint = torch.nn.Parameter(torch.zeros(6, dtype=torch.float32))
    factored = torch.nn.Parameter(torch.zeros(6, dtype=torch.float32))

    for _ in range(int(config.outer_iterations)):
        for delta, factorized in ((joint, False), (factored, True)):
            match_sets = refiner._collect_match_sets(
                refiner._left_updated_pose(delta.detach(), raw_pose), pairs
            )
            if factorized:
                refiner._run_factorized_steps(delta, raw_pose, pairs, match_sets)
            else:
                refiner._run_optimizer_stage(
                    delta,
                    raw_pose,
                    pairs,
                    match_sets,
                    indices=(0, 1, 2, 3, 4, 5),
                )
    assert not torch.allclose(joint.detach(), factored.detach())
