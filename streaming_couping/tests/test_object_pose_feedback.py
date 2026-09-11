"""Synthetic tests for the object-consensus camera-pose feedback baseline."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pytest
import torch

from streaming_couping.src.semantic_mapping.object_pose_feedback import (
    CONSENSUS_VARIANTS,
    FrameAlignmentContext,
    MAIN_VARIANT_NAME,
    RPE_ROTATION_BOUNDARY_EXCLUDED_KEY,
    RPE_TRANSLATION_BOUNDARY_EXCLUDED_KEY,
    ObjectPoseFeedbackConfig,
    ObjectPoseProposal,
    ReferenceCloud,
    aggregate_alignment_loss,
    build_frame_contexts,
    build_proposals,
    check_refiner_settings,
    classify_geometry,
    compare_trajectories,
    compute_track_stats,
    consensus_weight,
    covariance_eigenvalues,
    decide_object_feedback,
    gate_frame,
    gt_correction_error,
    pose_to_xi,
    robust_consensus,
    rotation_angle_deg,
    rotation_log,
    semantic_reliability,
    xi_to_pose,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _config(**overrides: Any) -> ObjectPoseFeedbackConfig:
    return ObjectPoseFeedbackConfig(**overrides).validate()


def _variant(name: str):
    for variant in CONSENSUS_VARIANTS:
        if variant.name == name:
            return variant
    raise KeyError(name)


def _cloud(point_count: int = 120, scale: float = 0.5, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.rand((point_count, 3), generator=generator) * float(scale)


def _proposal(**overrides: Any) -> ObjectPoseProposal:
    values: dict[str, Any] = dict(
        frame_id=10,
        sequence_index=10,
        instance_id=0,
        category="bed",
        correction=torch.eye(4),
        accepted_by_refiner=True,
        refiner_reason="object_loss_accepted",
        track_length=10,
        visibility_ratio=1.0,
        track_score=0.9,
        mask_pixels=4000,
        point_count=100,
        overlap_count=50,
        alignment_loss_before=0.08,
        alignment_loss_after=0.02,
        relative_improvement=0.5,
        inlier_ratio=0.5,
        eigenvalue_1=1.0,
        eigenvalue_2=1.0,
        eigenvalue_3=1.0,
        geometry_type="volumetric",
        degeneracy_factor=1.0,
        reference_frames=(1, 2),
        reference_roles=("anchor", "anchor"),
        pair_weights=(1.0, 1.0),
        semantic_confidence=0.8,
        semantic_reject_reason=None,
        geometry_confidence=0.8,
        geometry_reject_reason=None,
    )
    values.update(overrides)
    return ObjectPoseProposal(**values)


def _translation_pose(delta: Sequence[float]) -> torch.Tensor:
    pose = torch.eye(4)
    pose[0, 3] = float(delta[0])
    pose[1, 3] = float(delta[1])
    pose[2, 3] = float(delta[2])
    return pose


def _context(
    *,
    sequence_index: int = 10,
    instances: tuple[int, ...] = (0, 1),
    reference_shift: tuple[float, float, float] = (0.0, 0.0, 0.0),
    raw_pose: torch.Tensor | None = None,
    reference_frames: tuple[int, ...] = (1,),
) -> FrameAlignmentContext:
    shift = torch.tensor(reference_shift, dtype=torch.float32)
    points: dict[int, torch.Tensor] = {}
    weights: dict[int, torch.Tensor] = {}
    references: dict[int, tuple[ReferenceCloud, ...]] = {}
    for instance in instances:
        cloud = _cloud(seed=instance)
        points[instance] = cloud
        weights[instance] = torch.ones(cloud.shape[0])
        references[instance] = tuple(
            ReferenceCloud(
                frame_id=frame,
                sequence_index=frame,
                role="anchor",
                quality=0.9,
                points_world=cloud + shift[None, :],
                weights=torch.ones(cloud.shape[0]),
            )
            for frame in reference_frames
        )
    return FrameAlignmentContext(
        sequence_index=sequence_index,
        raw_pose=raw_pose if raw_pose is not None else torch.eye(4),
        points_camera=points,
        weights=weights,
        references=references,
    )


# ---------------------------------------------------------------------------
# Lie algebra and geometry classification.
# ---------------------------------------------------------------------------


def test_rotation_angle_is_stable_near_identity() -> None:
    """Regression: acos((trace-1)/2) reports ~0.03 deg for R against itself.

    A float32 rotation matrix is orthonormal only to ~1e-7, and acos amplifies
    that as sqrt(2*eps).  The comparator therefore flagged two bit-identical
    corrections as 0.028 degrees apart, which no tolerance in the replay check
    could fix without also masking real differences.
    """

    rotation = xi_to_pose(
        torch.tensor([0.02, -0.01, 0.03, 0.0, 0.0, 0.0]).numpy()
    )[:3, :3]
    # genuinely not orthonormal to more than float32 precision
    assert float(torch.trace(rotation)) < 3.0
    assert rotation_angle_deg(rotation.transpose(0, 1) @ rotation) == 0.0


def test_rotation_angle_matches_known_values() -> None:
    assert rotation_angle_deg(torch.eye(3)) == 0.0
    half = math.radians(90.0)
    quarter = torch.tensor(
        [
            [math.cos(half), -math.sin(half), 0.0],
            [math.sin(half), math.cos(half), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    assert rotation_angle_deg(quarter) == pytest.approx(90.0, abs=1e-4)
    assert rotation_angle_deg(torch.diag(torch.tensor([1.0, -1.0, -1.0]))) == (
        pytest.approx(180.0, abs=1e-4)
    )
    # 179 degrees stays accurate: that is where the acos form is well behaved
    nearly_pi = math.radians(179.0)
    rotation = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(nearly_pi), -math.sin(nearly_pi)],
            [0.0, math.sin(nearly_pi), math.cos(nearly_pi)],
        ]
    )
    assert rotation_angle_deg(rotation) == pytest.approx(179.0, abs=1e-3)


def test_rotation_angle_rejects_non_rotation_input() -> None:
    with pytest.raises(ValueError, match="3x3"):
        rotation_angle_deg(torch.eye(4))


def test_so3_log_exp_roundtrip() -> None:
    rng = np.random.default_rng(0)
    for angle in (0.0, 0.01, 0.175, 1.2, 3.0):
        axis = rng.normal(size=3)
        norm = float(np.linalg.norm(axis))
        if norm < 1e-9:
            continue
        vector = axis / norm * angle
        rotation = xi_to_pose(np.concatenate((vector, np.zeros(3))))[:3, :3]
        recovered = rotation_log(rotation).numpy()
        assert np.allclose(recovered, vector, atol=1e-4)


def test_covariance_classification() -> None:
    config = _config()
    volumetric = _cloud(seed=1, scale=0.5)
    planar = volumetric.clone()
    planar[:, 2] *= 1e-3
    linear = volumetric.clone()
    linear[:, 1:] *= 1e-3
    weights = torch.ones(volumetric.shape[0])
    cases = (
        (volumetric, "volumetric"),
        (planar, "planar"),
        (linear, "linear"),
    )
    for points, expected in cases:
        eigenvalues = covariance_eigenvalues(points, weights)
        geometry_type, _ = classify_geometry(
            eigenvalues, point_count=int(points.shape[0]), config=config
        )
        assert geometry_type == expected
    eigenvalues = covariance_eigenvalues(
        volumetric[:2], torch.ones(2)
    )
    geometry_type, _ = classify_geometry(
        eigenvalues, point_count=2, config=config
    )
    assert geometry_type == "degenerate"


# ---------------------------------------------------------------------------
# Reliability rules.
# ---------------------------------------------------------------------------


def test_semantic_reliability_rules() -> None:
    config = _config()
    good = compute_track_stats(
        {0: list(range(10))},
        instance_id=0,
        sequence_index=9,
        config=config,
    )
    score, reason = semantic_reliability(
        good, mask_pixels=4000, track_score=0.9, config=config
    )
    assert reason is None and score is not None and 0.0 < score <= 1.0
    short = compute_track_stats(
        {0: [0, 1]}, instance_id=0, sequence_index=1, config=config
    )
    score, reason = semantic_reliability(
        short, mask_pixels=4000, track_score=0.9, config=config
    )
    assert score is None and reason == "low_track_confidence"
    sparse = compute_track_stats(
        {0: list(range(0, 20, 4))}, instance_id=0, sequence_index=19, config=config
    )
    score, reason = semantic_reliability(
        sparse, mask_pixels=4000, track_score=0.9, config=config
    )
    assert score is None and reason == "low_track_confidence"
    score, reason = semantic_reliability(
        good, mask_pixels=4000, track_score=0.2, config=config
    )
    assert score is None and reason == "low_track_confidence"


def test_geometric_reliability_rules() -> None:
    from streaming_couping.src.semantic_mapping.object_pose_feedback import (
        geometric_reliability,
    )

    config = _config()
    score, reason = geometric_reliability(
        relative_improvement=0.4,
        inlier_ratio=0.4,
        overlap_count=40,
        correction=_translation_pose((0.05, 0.0, 0.0)),
        geometry_type="volumetric",
        degeneracy_factor=0.8,
        config=config,
    )
    assert reason is None and score is not None and score > 0.0
    score, reason = geometric_reliability(
        relative_improvement=0.4,
        inlier_ratio=0.4,
        overlap_count=40,
        correction=_translation_pose((0.05, 0.0, 0.0)),
        geometry_type="degenerate",
        degeneracy_factor=0.0,
        config=config,
    )
    assert score is None and reason == "degenerate_geometry"
    score, reason = geometric_reliability(
        relative_improvement=0.4,
        inlier_ratio=0.4,
        overlap_count=40,
        correction=_translation_pose((0.30, 0.0, 0.0)),
        geometry_type="volumetric",
        degeneracy_factor=0.8,
        config=config,
    )
    assert score is None and reason == "correction_too_large"
    score, reason = geometric_reliability(
        relative_improvement=0.001,
        inlier_ratio=0.4,
        overlap_count=40,
        correction=torch.eye(4),
        geometry_type="volumetric",
        degeneracy_factor=0.8,
        config=config,
    )
    assert score is None and reason == "low_geometry_confidence"
    planar_score, _ = geometric_reliability(
        relative_improvement=0.4,
        inlier_ratio=0.4,
        overlap_count=40,
        correction=torch.eye(4),
        geometry_type="planar",
        degeneracy_factor=0.001,
        config=config,
    )
    good_score, _ = geometric_reliability(
        relative_improvement=0.4,
        inlier_ratio=0.4,
        overlap_count=40,
        correction=torch.eye(4),
        geometry_type="volumetric",
        degeneracy_factor=0.8,
        config=config,
    )
    assert good_score is not None
    assert planar_score is not None and planar_score < good_score


# ---------------------------------------------------------------------------
# Consensus.
# ---------------------------------------------------------------------------


def test_robust_consensus_identical_proposals() -> None:
    config = _config()
    correction = _translation_pose((0.06, 0.0, 0.0))
    proposals = [
        _proposal(instance_id=k, correction=correction) for k in range(3)
    ]
    result = robust_consensus(proposals, _variant(MAIN_VARIANT_NAME), config=config)
    assert result.delta is not None
    t_error, r_error = gt_correction_error(result.delta, correction)
    assert t_error < 1e-6 and r_error < 1e-6
    assert all(result.inlier_flags)


def test_robust_consensus_flags_outlier() -> None:
    config = _config()
    cluster = _translation_pose((0.05, 0.0, 0.0))
    outlier = _translation_pose((-0.18, 0.0, 0.0))
    proposals = [
        _proposal(instance_id=0, correction=cluster),
        _proposal(instance_id=1, correction=cluster),
        _proposal(instance_id=2, correction=outlier),
    ]
    result = robust_consensus(proposals, _variant(MAIN_VARIANT_NAME), config=config)
    assert result.delta is not None
    t_error, _ = gt_correction_error(result.delta, cluster)
    # The consensus must stay much closer to the cluster (0.05) than the
    # outlier (-0.18); IRLS keeps the outlier influence near 20%.
    assert t_error < 0.03
    assert result.inlier_flags == (True, True, False)


def test_mean_and_single_variants() -> None:
    config = _config()
    proposals = [
        _proposal(
            instance_id=0,
            correction=_translation_pose((0.02, 0.0, 0.0)),
            semantic_confidence=0.9,
            geometry_confidence=0.9,
        ),
        _proposal(
            instance_id=1,
            correction=_translation_pose((0.06, 0.0, 0.0)),
            semantic_confidence=0.3,
            geometry_confidence=0.3,
        ),
    ]
    mean = robust_consensus(proposals, _variant("mean"), config=config)
    assert mean.delta is not None
    assert math.isclose(float(mean.delta[0, 3]), 0.04, abs_tol=1e-9)
    single = robust_consensus(proposals, _variant("single"), config=config)
    assert single.delta is not None
    t_error, _ = gt_correction_error(single.delta, proposals[0].correction)
    assert t_error < 1e-9
    weight_best = consensus_weight(proposals[0], _variant("single"))
    weight_other = consensus_weight(proposals[1], _variant("single"))
    assert weight_best > weight_other


# ---------------------------------------------------------------------------
# Aggregate alignment loss.
# ---------------------------------------------------------------------------


def test_aggregate_alignment_loss_improves_with_true_correction() -> None:
    config = _config()
    shift = (0.12, 0.0, 0.0)
    context = _context(reference_shift=shift)
    proposals = [
        _proposal(instance_id=k, correction=_translation_pose(shift))
        for k in (0, 1)
    ]
    before, after = aggregate_alignment_loss(
        context, proposals, None, config=config
    )
    assert before > 0.0
    _, after_corrected = aggregate_alignment_loss(
        context, proposals, _translation_pose(shift), config=config
    )
    assert after_corrected < before * (1.0 - 0.5)
    identity_before, identity_after = aggregate_alignment_loss(
        context, proposals, torch.eye(4), config=config
    )
    assert identity_after > identity_before * 0.99


# ---------------------------------------------------------------------------
# Gating.
# ---------------------------------------------------------------------------


def test_gate_frame_accept() -> None:
    config = _config()
    shift = (0.12, 0.0, 0.0)
    context = _context(reference_shift=shift)
    proposals = [
        _proposal(
            instance_id=k,
            sequence_index=10,
            frame_id=10,
            correction=_translation_pose(shift),
        )
        for k in (0, 1)
    ]
    result = gate_frame(
        proposals, _variant(MAIN_VARIANT_NAME), context=context, config=config
    )
    assert result.accepted, result.reason
    assert result.reason is None
    assert result.target_c2w is not None
    assert torch.allclose(
        result.target_c2w, _translation_pose(shift), atol=1e-5
    )
    assert result.num_reliable == 2
    assert result.aggregate_loss_after is not None
    assert result.aggregate_loss_before is not None
    assert result.aggregate_loss_after < result.aggregate_loss_before


def test_gate_frame_reject_reasons() -> None:
    config = _config()
    shift = (0.12, 0.0, 0.0)
    good = [
        _proposal(instance_id=k, correction=_translation_pose(shift))
        for k in (0, 1)
    ]
    cases = {
        "insufficient_objects": ([good[0]], _context(reference_shift=shift)),
        "low_track_confidence": (
            [
                _proposal(
                    instance_id=k,
                    correction=_translation_pose(shift),
                    track_length=2,
                    semantic_confidence=None,
                    semantic_reject_reason="low_track_confidence",
                )
                for k in (0, 1)
            ],
            _context(reference_shift=shift),
        ),
        "degenerate_geometry": (
            [
                _proposal(
                    instance_id=k,
                    correction=_translation_pose(shift),
                    geometry_type="degenerate",
                    degeneracy_factor=0.0,
                    geometry_confidence=None,
                    geometry_reject_reason="degenerate_geometry",
                )
                for k in (0, 1)
            ],
            _context(reference_shift=shift),
        ),
        "low_geometry_confidence": (
            [
                _proposal(
                    instance_id=k,
                    correction=_translation_pose(shift),
                    relative_improvement=0.001,
                    geometry_confidence=None,
                    geometry_reject_reason="low_geometry_confidence",
                )
                for k in (0, 1)
            ],
            _context(reference_shift=shift),
        ),
        "correction_too_large": (
            [
                _proposal(instance_id=k, correction=_translation_pose((0.24, 0.0, 0.0)))
                for k in (0, 1)
            ],
            _context(reference_shift=shift),
        ),
        "no_consensus": (
            [
                _proposal(instance_id=0, correction=_translation_pose((0.22, 0.0, 0.0))),
                _proposal(instance_id=1, correction=_translation_pose((-0.22, 0.0, 0.0))),
            ],
            _context(reference_shift=shift),
        ),
        "no_alignment_improvement": (
            [
                _proposal(instance_id=k, correction=torch.eye(4))
                for k in (0, 1)
            ],
            _context(reference_shift=(0.0, 0.0, 0.0)),
        ),
    }
    variant = _variant(MAIN_VARIANT_NAME)
    for expected_reason, (proposals, context) in cases.items():
        result = gate_frame(proposals, variant, context=context, config=config)
        assert not result.accepted
        assert result.reason == expected_reason, (
            expected_reason,
            result.reason,
        )


def test_gate_frame_keeps_consensus_delta_on_post_consensus_rejection() -> None:
    """A rejected gate must still expose its consensus correction.

    The offline evaluation scores rejected frames against GT to decide whether
    the gate rejects the right frames; that is impossible if the delta is
    dropped on rejection.  Injection is driven by ``target_c2w``/``accepted``,
    so carrying the delta here cannot change the replay.
    """

    config = _config()
    shift = (0.12, 0.0, 0.0)
    # Identity consensus that cannot improve the aggregate alignment loss.
    proposals = [
        _proposal(instance_id=k, correction=torch.eye(4)) for k in (0, 1)
    ]
    result = gate_frame(
        proposals,
        _variant(MAIN_VARIANT_NAME),
        context=_context(reference_shift=shift),
        config=config,
    )
    assert not result.accepted
    assert result.reason == "no_alignment_improvement"
    assert result.delta is not None
    assert result.target_c2w is None
    assert result.consensus_translation_norm is not None


def test_gate_frame_has_no_delta_without_a_consensus() -> None:
    config = _config()
    shift = (0.12, 0.0, 0.0)
    result = gate_frame(
        [_proposal(instance_id=0, correction=_translation_pose(shift))],
        _variant(MAIN_VARIANT_NAME),
        context=_context(reference_shift=shift),
        config=config,
    )
    assert result.reason == "insufficient_objects"
    assert result.delta is None
    assert result.consensus_translation_norm is None


# ---------------------------------------------------------------------------
# Diagnostics payload consumption.
# ---------------------------------------------------------------------------


def test_build_proposals_from_diagnostics() -> None:
    config = _config()
    cloud = _cloud(point_count=100, seed=3)
    cloud[:, 2] *= 1e-3  # planar support
    frame_count = 8
    observations = [
        {
            "frame_id": index,
            "sequence_index": index,
            "instance_id": 0,
            "category": "bed",
            "points_camera": cloud,
            "weights": torch.ones(cloud.shape[0]),
            "track_score": 0.9,
            "geometry_confidence": 0.8,
            "mask_pixels": 4000,
            "static": True,
        }
        for index in range(frame_count)
    ]
    diagnostics = {
        "schema": "object_pose_loss_feedback_diagnostics_r1",
        "frame_ids": list(range(frame_count)),
        "raw_camera_to_world": torch.eye(4)[None].repeat(frame_count, 1, 1),
        "observations": observations,
        "pairing_snapshots": [
            {
                "frame_id": 5,
                "sequence_index": 5,
                "instance_id": 0,
                "references": [
                    {
                        "frame_id": 1,
                        "sequence_index": 1,
                        "role": "anchor",
                        "quality": 0.8,
                        "points_world": cloud,
                        "weights": torch.ones(cloud.shape[0]),
                    }
                ],
            }
        ],
        "proposals": [
            {
                "frame_id": 5,
                "sequence_index": 5,
                "instance_id": 0,
                "category": "bed",
                "accepted": True,
                "reason": "object_loss_accepted",
                "correction": torch.eye(4),
                "initial_loss_m": 0.08,
                "final_loss_m": 0.02,
                "initial_match_count": 60,
                "final_match_count": 60,
                "relative_loss_improvement": 0.75,
                "reference_frames": [1],
                "reference_roles": ["anchor"],
                "pair_weights": [0.5],
            },
            {
                "frame_id": 7,
                "sequence_index": 7,
                "instance_id": 0,
                "category": "bed",
                "accepted": False,
                "reason": "no_historical_instance_reference",
                "correction": None,
                "initial_loss_m": None,
                "final_loss_m": None,
                "initial_match_count": None,
                "final_match_count": None,
                "relative_loss_improvement": None,
                "reference_frames": [],
                "reference_roles": [],
                "pair_weights": [],
            },
        ],
    }
    proposals = build_proposals(diagnostics, config=config)
    assert len(proposals) == 2
    first = proposals[0]
    assert first.geometry_type == "planar"
    assert first.track_length == 6
    assert math.isclose(first.visibility_ratio, 1.0)
    assert first.inlier_ratio == 0.6
    assert first.semantic_reject_reason is None
    assert first.geometry_reject_reason is None
    assert first.is_reliable
    second = proposals[1]
    assert second.correction is None
    assert second.object_reject_reason == "no_historical_instance_reference"
    contexts = build_frame_contexts(diagnostics, config=config)
    assert set(contexts) == set(range(frame_count))
    assert 0 in contexts[5].references
    assert contexts[5].raw_pose.shape == (4, 4)


# ---------------------------------------------------------------------------
# Replay engine (requires the horizonstream runtime).
# ---------------------------------------------------------------------------


def _synthetic_chunk_cam_maps(
    *,
    frame_count: int = 16,
    window: int = 4,
    sliding: int = 1,
    seed: int = 0,
):
    pytest.importorskip("horizonstream.runtime.motion_averaging")
    from horizonstream.utils.vendor.models.components.utils.rotation import (
        mat_to_quat,
    )

    rng = np.random.default_rng(seed)
    c2w: list[np.ndarray] = []
    center = np.zeros(3)
    for _ in range(frame_count):
        center = center + rng.normal(scale=0.05, size=3)
        axis = rng.normal(size=3)
        axis = axis / max(float(np.linalg.norm(axis)), 1e-9)
        angle = float(rng.uniform(0.0, math.radians(3.0)))
        pose = np.eye(4)
        pose[:3, :3] = (
            xi_to_pose(np.concatenate((axis * angle, np.zeros(3))))[:3, :3]
            .numpy()
            .astype(np.float64)
        )
        pose[:3, 3] = center
        c2w.append(pose)

    def encode(relative: np.ndarray) -> torch.Tensor:
        translation = torch.from_numpy(relative[:3, 3]).float()
        quat = mat_to_quat(torch.from_numpy(relative[:3, :3]).float())
        focal = torch.tensor([300.0, 300.0])
        return torch.cat((translation, quat.float(), focal))

    schedule = [(0, window)]
    start = window
    while start < frame_count:
        end = min(start + sliding, frame_count)
        schedule.append((start, end))
        start = end
    chunks: list[torch.Tensor] = []
    for chunk_index, (start, end) in enumerate(schedule):
        if chunk_index == 0:
            new_frames = list(range(0, end))
        else:
            new_frames = list(range(start, end))
        rows = []
        for t in new_frames:
            j_start = max(0, t - window + 1)
            entries = [
                encode(np.linalg.inv(c2w[j]) @ c2w[t])
                for j in range(j_start, t + 1)
            ]
            while len(entries) < window:
                entries.insert(0, entries[0])
            rows.append(torch.stack(entries))
        chunks.append(torch.stack(rows))
    expected_public = np.stack(
        [np.linalg.inv(c2w[0]) @ pose for pose in c2w], axis=0
    )
    return chunks, expected_public


def _replay(
    chunks: list[torch.Tensor],
    *,
    frame_count: int,
    window: int,
    injections: dict[int, np.ndarray] | None = None,
):
    from streaming_couping.scripts.run_object_pose_feedback import (
        replay_trajectory_with_injections,
    )

    return replay_trajectory_with_injections(
        chunks,
        frame_count=frame_count,
        window_size=window,
        injections=injections or {},
        horizon_repo=REPO_ROOT / "externals" / "horizonstream",
    )


def test_replay_equivalence_with_motion_averaging() -> None:
    pytest.importorskip("horizonstream.runtime.motion_averaging")
    from horizonstream.runtime.motion_averaging import (
        compute_motion_averaged_camera_maps,
    )
    from streaming_couping.scripts.run_scannet_horizonstream_gt_feedback_poc import (
        _cam_map_to_c2w,
    )

    frame_count, window = 16, 4
    chunks, expected_public = _synthetic_chunk_cam_maps(
        frame_count=frame_count, window=window
    )
    motion = compute_motion_averaged_camera_maps(
        chunks,
        frames_num=frame_count,
        window_size=window,
        dtype=torch.float32,
        enable_offline=False,
    )
    reference_c2w = _cam_map_to_c2w(
        motion["online_cam_map"], image_hw=(48, 48)
    )
    replay_c2w, payload = _replay(
        chunks, frame_count=frame_count, window=window
    )
    assert payload["injection_count"] == 0
    assert replay_c2w.shape == (frame_count, 4, 4)
    for index in range(frame_count):
        assert np.allclose(replay_c2w[index], reference_c2w[index], atol=1e-6)
        translation_gap = float(
            np.linalg.norm(replay_c2w[index, :3, 3] - expected_public[index, :3, 3])
        )
        assert translation_gap < 1e-4, index


def test_replay_injection_semantics() -> None:
    pytest.importorskip("horizonstream.runtime.motion_averaging")
    frame_count, window = 16, 4
    chunks, _expected = _synthetic_chunk_cam_maps(
        frame_count=frame_count, window=window, seed=1
    )
    raw_c2w, _ = _replay(chunks, frame_count=frame_count, window=window)
    delta = _translation_pose((0.05, 0.0, 0.0)).numpy().astype(np.float64)
    injections = {
        frame: raw_c2w[frame] @ delta for frame in (8, 9, 10)
    }
    corrected_c2w, payload = _replay(
        chunks,
        frame_count=frame_count,
        window=window,
        injections=injections,
    )
    assert payload["injection_count"] == 3
    assert payload["applied_frames"] == [8, 9, 10]
    for frame in range(8):
        assert np.allclose(corrected_c2w[frame], raw_c2w[frame], atol=1e-9)
    for frame in (8, 9, 10):
        assert np.allclose(corrected_c2w[frame], injections[frame], atol=1e-6)
    propagated = any(
        float(
            np.linalg.norm(
                corrected_c2w[frame, :3, 3] - raw_c2w[frame, :3, 3]
            )
        )
        > 1e-6
        for frame in range(11, frame_count)
    )
    assert propagated, "accumulator correction must reach later frames"


# ---------------------------------------------------------------------------
# Metric-key contract and stage-2a/stage-2b threshold mirroring.
# ---------------------------------------------------------------------------


def _branch_metrics(
    *,
    ate: float,
    rpe_translation: float,
    rpe_rotation: float,
) -> dict[str, Any]:
    return {
        "ate_rmse_m": ate,
        RPE_TRANSLATION_BOUNDARY_EXCLUDED_KEY: rpe_translation,
        RPE_ROTATION_BOUNDARY_EXCLUDED_KEY: rpe_rotation,
    }


def _mirrored_refiner_settings(
    config: ObjectPoseFeedbackConfig,
    **overrides: Any,
) -> dict[str, Any]:
    settings: dict[str, Any] = {
        "anchor_frame_count": int(config.anchor_frame_count),
        "max_correction_rotation_deg": float(config.max_feedback_rotation_deg),
        "max_correction_translation_m": float(config.max_feedback_translation_m),
        "max_match_distance_m": float(config.aggregate_max_match_distance_m),
        "trim_ratio": float(config.aggregate_trim_ratio),
        "min_matches_per_pair": int(config.aggregate_min_matches_per_pair),
    }
    settings.update(overrides)
    return settings


def test_compare_trajectories_reads_the_rotation_metric_key() -> None:
    """Regression: the rotation key must not be string-derived from the
    translation key (``_m`` vs ``_deg`` made every rotation ratio ``None``)."""

    raw = _branch_metrics(ate=0.40, rpe_translation=0.020, rpe_rotation=0.10)
    feedback = _branch_metrics(ate=0.34, rpe_translation=0.0195, rpe_rotation=0.08)
    comparison = compare_trajectories(
        raw,
        feedback,
        raw_rpe_translation_key=RPE_TRANSLATION_BOUNDARY_EXCLUDED_KEY,
        feedback_rpe_translation_key=RPE_TRANSLATION_BOUNDARY_EXCLUDED_KEY,
        raw_rpe_rotation_key=RPE_ROTATION_BOUNDARY_EXCLUDED_KEY,
        feedback_rpe_rotation_key=RPE_ROTATION_BOUNDARY_EXCLUDED_KEY,
    )
    assert comparison.ate_improvement_ratio == pytest.approx(0.15)
    assert comparison.rpe_translation_ratio == pytest.approx(0.025)
    assert comparison.rpe_rotation_ratio == pytest.approx(0.20)


def test_decide_object_feedback_reports_rotation_ratio_informationally() -> None:
    decision = decide_object_feedback(
        variant_name=MAIN_VARIANT_NAME,
        raw_metrics=_branch_metrics(
            ate=0.40, rpe_translation=0.020, rpe_rotation=0.10
        ),
        feedback_metrics=_branch_metrics(
            ate=0.34, rpe_translation=0.0195, rpe_rotation=0.08
        ),
        future_translation_gains=[0.01] * 10,
        future_rotation_gains=[0.001] * 10,
        accepted_ratio=0.5,
        config=_config(),
    )
    assert decision["decision"] == "OBJECT_FEEDBACK_GO"
    assert decision[
        "rpe_rotation_ratio_boundary_excluded_informational"
    ] == pytest.approx(0.20)
    assert (
        "rpe_rotation_ratio_boundary_excluded_informational"
        not in decision["criteria"]
    )


def test_check_refiner_settings_accepts_mirrored_thresholds() -> None:
    config = _config()
    report = check_refiner_settings(
        _mirrored_refiner_settings(config), config=config
    )
    assert report["matches"] is True
    assert report["mismatches"] == {}
    assert report["missing_refiner_settings"] == []


def test_check_refiner_settings_flags_each_divergent_threshold() -> None:
    config = _config(anchor_frame_count=8, aggregate_trim_ratio=0.50)
    settings = _mirrored_refiner_settings(
        config, anchor_frame_count=5, trim_ratio=0.70
    )
    report = check_refiner_settings(settings, config=config)
    assert report["matches"] is False
    assert set(report["mismatches"]) == {"anchor_frame_count", "aggregate_trim_ratio"}
    assert report["mismatches"]["anchor_frame_count"] == {
        "refiner_setting": "anchor_frame_count",
        "refiner_value": 5.0,
        "feedback_value": 8.0,
    }
    assert report["mismatches"]["aggregate_trim_ratio"]["refiner_value"] == 0.70


def test_check_refiner_settings_reports_missing_exported_fields() -> None:
    report = check_refiner_settings({"anchor_frame_count": 5}, config=_config())
    assert report["matches"] is False
    assert set(report["missing_refiner_settings"]) == {
        "max_correction_rotation_deg",
        "max_correction_translation_m",
        "max_match_distance_m",
        "min_matches_per_pair",
        "trim_ratio",
    }
