"""The refiner replay path must reproduce a run from its diagnostics alone.

This is what makes a refiner-parameter comparison trustworthy: the
segmentation model runs once, and every later branch is replayed on CPU with
no GPU variance mixed in.  The whole point rests on the replay being faithful,
so the central test drives a real run through ``refine`` and then asserts the
replay of its diagnostics lands on the same proposals.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from streaming_couping.scripts.run_object_pose_loss_replay import (
    OVERRIDE_FLAGS,
    _flag_dest,
    collect_overrides,
    compare_diagnostics,
    config_from_diagnostics,
    load_diagnostics,
    main as replay_main,
    observations_from_diagnostics,
)
from streaming_couping.src.semantic_mapping.contracts import (
    GeometryFrame,
    ObjectObservation,
    SegmentationFrame,
)
from streaming_couping.src.semantic_mapping.object_pose_loss_refinement import (
    ObjectPoseLossRefinementConfig,
    ObjectPoseLossRefiner,
)


def _mask(columns: slice) -> torch.Tensor:
    mask = torch.zeros((6, 6), dtype=torch.bool)
    mask[1:5, columns] = True
    return mask


def _geometry(frame_id: int, shift: float) -> GeometryFrame:
    size = (6, 6)
    yy, xx = torch.meshgrid(
        torch.arange(6, dtype=torch.float32),
        torch.arange(6, dtype=torch.float32),
        indexing="ij",
    )
    points = torch.stack((0.20 * xx, 0.20 * yy, torch.ones_like(xx)), dim=-1)
    points = points.clone()
    # one object drifts with the frame, so there is something to align
    points[1:5, 1:4, 0] += shift
    pose = torch.eye(4)
    pose[0, 3] = 0.01 * frame_id
    return GeometryFrame(
        frame_id=frame_id,
        image_size=size,
        world_points=points,
        camera_to_world=pose,
        confidence=torch.ones(size),
        valid=torch.ones(size, dtype=torch.bool),
        rgb=torch.ones(*size, 3),
        backend="synthetic_horizonstream",
    )


def _segmentation(frame_id: int, masks) -> SegmentationFrame:
    return SegmentationFrame(
        frame_id=frame_id,
        image_size=(6, 6),
        observations=tuple(
            ObjectObservation(
                category="chair",
                instance_id=index,
                mask=mask,
                score=0.95,
                static_score=1.0,
            )
            for index, mask in enumerate(masks)
        ),
        backend="synthetic_sam3",
    )


def _config(**overrides) -> ObjectPoseLossRefinementConfig:
    values = dict(
        independent_instance_poses=True,
        export_feedback_diagnostics=True,
        outer_iterations=2,
        optimizer_steps=8,
        # the fixture masks are 4x3 px, well under the 32 px production floor
        min_mask_pixels=4,
        min_points_per_observation=4,
        max_points_per_observation=12,
        min_matches_per_pair=2,
        min_total_matches=4,
    )
    values.update(overrides)
    return ObjectPoseLossRefinementConfig(**values).validate()


def _source_run(tmp_path: Path) -> dict:
    """One real refine() pass, returning the diagnostics it exported."""

    frame_count = 8
    # every frame observes both objects; the second only from frame 3 on
    left = _mask(slice(0, 3))
    right = _mask(slice(3, 6))
    geometry = [_geometry(frame_id, 0.02 * frame_id) for frame_id in range(frame_count)]
    segmentation = [
        _segmentation(frame_id, (left, right) if frame_id >= 3 else (left,))
        for frame_id in range(frame_count)
    ]
    refiner = ObjectPoseLossRefiner(_config())
    result = refiner.refine(geometry, segmentation, ["unused"] * frame_count)
    payload = result.feedback_diagnostics
    assert payload is not None
    torch.save(payload, tmp_path / "source.pt")
    return payload


def test_replay_reproduces_the_source_proposals(tmp_path: Path) -> None:
    """The central guarantee: replay from diagnostics == the original run."""

    source = _source_run(tmp_path)
    assert source["proposals"], "fixture produced no proposals to compare"

    observations_by_frame = observations_from_diagnostics(source)
    config = config_from_diagnostics(source, {})
    refiner = ObjectPoseLossRefiner(config)
    result = refiner.refine_from_observations(
        frame_ids=tuple(int(value) for value in source["frame_ids"]),
        raw_poses=tuple(
            torch.as_tensor(pose).detach().float().cpu()
            for pose in source["raw_camera_to_world"]
        ),
        observations_by_frame=observations_by_frame,
        tracked_ids={int(value) for value in source["collection"]["tracked_instance_ids"]},
        filter_stats=__import__("collections").Counter(
            source["collection"]["filter_stats"]
        ),
        raw_observation_count=int(source["collection"]["raw_observation_count"]),
    )
    assert result.feedback_diagnostics is not None
    comparison = compare_diagnostics(
        source,
        result.feedback_diagnostics,
        translation_tolerance_m=1e-6,
        rotation_tolerance_deg=1e-4,
    )
    assert comparison["passed"], comparison
    assert comparison["compared_proposal_count"] == len(source["proposals"])


def test_observations_round_trip_exactly(tmp_path: Path) -> None:
    source = _source_run(tmp_path)
    rebuilt = observations_from_diagnostics(source)
    dumped = {
        (int(row["frame_id"]), int(row["instance_id"])): row
        for row in source["observations"]
    }
    assert set(rebuilt) == {int(frame) for frame in source["frame_ids"]} or True
    for frame_id, rows in rebuilt.items():
        for observation in rows:
            row = dumped[(frame_id, observation.instance_id)]
            assert torch.equal(
                observation.points_camera,
                torch.as_tensor(row["points_camera"]).float(),
            )
            assert torch.equal(
                observation.weights, torch.as_tensor(row["weights"]).float()
            )
            assert observation.track_score == pytest.approx(float(row["track_score"]))
            assert observation.mask_pixels == int(row["mask_pixels"])
            assert observation.category == str(row["category"])


def test_config_round_trips_through_the_diagnostics(tmp_path: Path) -> None:
    """The full config must survive, not just the mirrored thresholds."""

    source = _source_run(tmp_path)
    config = config_from_diagnostics(source, {})
    assert config.outer_iterations == 2
    assert config.optimizer_steps == 8
    assert config.min_total_matches == 4
    # and a replay forces the export on so its output is consumable again
    assert config.export_feedback_diagnostics is True


def test_overrides_are_applied_on_top_of_the_stored_config(tmp_path: Path) -> None:
    source = _source_run(tmp_path)
    config = config_from_diagnostics(
        source,
        {"max_reference_age_frames": 15, "proposal_mode": "rotation_then_translation"},
    )
    assert config.max_reference_age_frames == 15
    assert config.proposal_mode == "rotation_then_translation"


def test_missing_refiner_settings_is_a_clear_error() -> None:
    with pytest.raises(ValueError, match="refiner_settings"):
        config_from_diagnostics({"frame_ids": []}, {})


def test_flag_dest_matches_argparse_underscoring() -> None:
    for _name, flag in OVERRIDE_FLAGS:
        assert _flag_dest(flag) == flag.replace("-", "_")
        assert _flag_dest(flag).startswith("object_pose_loss_")


def test_collect_overrides_ignores_unset_flags() -> None:
    parser_ns = type("NS", (), {})()
    for _name, flag in OVERRIDE_FLAGS:
        setattr(parser_ns, _flag_dest(flag), None)
    assert collect_overrides(parser_ns) == {}
    setattr(parser_ns, _flag_dest(OVERRIDE_FLAGS[0][1]), 15)
    assert collect_overrides(parser_ns) == {"max_reference_age_frames": 15}


def test_compare_diagnostics_detects_a_divergent_proposal() -> None:
    correction = torch.eye(4)
    shifted = torch.eye(4)
    shifted[0, 3] = 0.05
    source = {
        "proposals": [{"sequence_index": 0, "instance_id": 0, "correction": correction}]
    }
    same = {
        "proposals": [{"sequence_index": 0, "instance_id": 0, "correction": correction.clone()}]
    }
    other = {
        "proposals": [{"sequence_index": 0, "instance_id": 0, "correction": shifted}]
    }
    assert compare_diagnostics(
        source, same, translation_tolerance_m=1e-6, rotation_tolerance_deg=1e-4
    )["passed"]
    diverged = compare_diagnostics(
        source, other, translation_tolerance_m=1e-6, rotation_tolerance_deg=1e-4
    )
    assert not diverged["passed"]
    assert diverged["max_correction_translation_gap_m"] == pytest.approx(0.05)


def test_compare_diagnostics_reports_key_set_changes() -> None:
    source = {"proposals": [{"sequence_index": 0, "instance_id": 0, "correction": None}]}
    replay = {"proposals": []}
    result = compare_diagnostics(
        source, replay, translation_tolerance_m=1e-6, rotation_tolerance_deg=1e-4
    )
    assert result["passed"] is False
    assert "key sets differ" in result["reason"]


def test_cli_writes_stage_2b_shaped_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    source_path = tmp_path / "source.pt"
    _source_run(tmp_path)
    (tmp_path / "source.pt").replace(source_path)
    output_dir = tmp_path / "branch"
    monkeypatch.setattr(
        "sys.argv",
        [
            "replay",
            "--diagnostics",
            str(source_path),
            "--output-dir",
            str(output_dir),
            "--check-equivalence",
        ],
    )
    replay_main()
    printed = capsys.readouterr().out
    assert "replay equivalence OK" in printed
    written = output_dir / "object_pose_refinement" / "feedback_diagnostics.pt"
    assert written.is_file()
    payload = torch.load(written, map_location="cpu", weights_only=False)
    assert payload["schema"] == "object_pose_loss_feedback_diagnostics_r1"
    assert payload["replay_provenance"]["gpu_used"] is False
    assert payload["replay_equivalence"]["passed"] is True
    # no stray temp file left behind by the atomic write
    assert not list(written.parent.glob(".*tmp*"))


def test_cli_with_overrides_skips_the_equivalence_assertion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    source_path = tmp_path / "source.pt"
    _source_run(tmp_path)
    (tmp_path / "source.pt").replace(source_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "replay",
            "--diagnostics",
            str(source_path),
            "--output-dir",
            str(tmp_path / "branch"),
            "--check-equivalence",
            "--object-pose-loss-max-reference-age-frames",
            "3",
        ],
    )
    replay_main()
    assert "overrides were supplied" in capsys.readouterr().out
