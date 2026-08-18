#!/usr/bin/env python3
"""CPU smoke checks for the frozen T0.2 confirmation protocol."""

from __future__ import annotations

from pathlib import Path
import tempfile

import torch

from streaming_couping.scripts.run_t01_triangulation_reliability import (
    _candidate_metrics,
    _load_and_validate_scores,
    _validate_and_select_candidates,
)

from streaming_couping.scripts.run_t02_multiview_confirmation import (
    T02Run,
    T02Gate,
    confirmation_gate_pass,
    interpret_sam_geometry,
    t02_gate_mask,
    validate_temporal_holdout_source,
)


def main() -> None:
    test_strict_condition_and_multiview_gate()
    test_empty_control_branch_is_valid()
    test_same_scene_unseen_time_protocol()
    test_confirmation_go_rule()
    test_sam_interpretation()
    print("T0.2 frozen same-scene temporal-holdout smoke passed")


def test_strict_condition_and_multiview_gate() -> None:
    metrics = {
        "condition_number": torch.tensor([99.0, 100.0, 20.0, 20.0]),
        "num_views": torch.tensor([4, 4, 3, 5]),
        "positive_depth": torch.tensor([True, True, True, False]),
    }
    gate = T02Gate(maximum_condition_number_exclusive=100.0, minimum_views=4)
    assert t02_gate_mask(metrics, gate).tolist() == [True, False, False, False]


def test_empty_control_branch_is_valid() -> None:
    empty = {
        "point_world": torch.empty(0, 3),
        "current_sequence_index": torch.empty(0, dtype=torch.long),
        "current_frame_index": torch.empty(0, dtype=torch.long),
        "slot": torch.empty(0, dtype=torch.long),
        "sam_track_id": torch.empty(0, dtype=torch.long),
        "query_patch_flat_index": torch.empty(0, dtype=torch.long),
        "query_pixel_xy": torch.empty(0, 2),
        "num_views": torch.empty(0, dtype=torch.long),
        "condition_number": torch.empty(0),
        "maximum_ray_angle_degrees": torch.empty(0),
        "mean_reprojection_error_px": torch.empty(0),
    }
    candidate = {
        "schema": 1,
        "revision": "t0_sam_indexed_independent_3d_anchor_probe_r1",
        "artifact_role": "frozen_independent_3d_anchor_candidates",
        "candidate_generation_raw_pointmap_fields": 0,
        "candidate_generation_gt_fields": 0,
        "branches": {"shuffled_persistent_id": empty},
    }
    try:
        _validate_and_select_candidates(candidate, "shuffled_persistent_id")
    except ValueError:
        pass
    else:
        raise AssertionError("Empty candidate unexpectedly passed strict T0.1 mode.")
    anchors = _validate_and_select_candidates(
        candidate,
        "shuffled_persistent_id",
        allow_empty=True,
    )
    metrics = _candidate_metrics(anchors)
    gate_mask = t02_gate_mask(metrics, T02Gate(100.0, 4))
    assert gate_mask.numel() == 0
    with tempfile.TemporaryDirectory() as directory:
        score_path = Path(directory) / "scores.csv"
        score_path.write_text("", encoding="utf8")
        rows = _load_and_validate_scores(
            score_path,
            branch="shuffled_persistent_id",
            anchors=anchors,
            metrics=metrics,
            gate_masks={"frozen_gate": gate_mask},
            allow_empty=True,
        )
    assert rows == []


def test_same_scene_unseen_time_protocol() -> None:
    run = T02Run(
        source_path=Path("config.yaml"),
        discovery_clip="scene_90_525",
        discovery_scene_id="scene",
        discovery_last_frame=525,
        validation_scope="same_scene_unseen_temporal_holdout",
        confirmation_clip="scene_533_591",
        confirmation_frames=tuple(range(533, 592, 2)),
        source_t0_output_dir=Path("source"),
        output_dir=Path("output"),
        gate=T02Gate(100.0, 4),
        minimum_correct_anchor_count=30,
        sam_comparison_tolerance_percent=2.0,
        minimum_control_anchor_count=10,
    )
    metadata = validate_temporal_holdout_source(
        run,
        {
            "clip_name": "scene_533_591",
            "scene_id": "scene",
            "frame_indices": tuple(range(533, 592, 2)),
        },
    )
    assert metadata["same_scene_as_discovery"] == 1
    assert metadata["frame_overlap_with_discovery"] == 0
    try:
        validate_temporal_holdout_source(
            run,
            {
                "clip_name": "scene_500_558",
                "scene_id": "scene",
                "frame_indices": tuple(range(500, 559, 2)),
            },
        )
    except ValueError:
        pass
    else:
        raise AssertionError("T0.2 accepted frames overlapping discovery time.")


def test_confirmation_go_rule() -> None:
    passing = {
        "anchor_count": 30,
        "tri_rmse": 0.4,
        "raw_rmse": 0.6,
        "tri_p90": 0.7,
        "raw_p90": 0.8,
    }
    assert confirmation_gate_pass(passing, minimum_anchor_count=30)
    failing = {**passing, "tri_p90": 0.9}
    assert not confirmation_gate_pass(failing, minimum_anchor_count=30)


def test_sam_interpretation() -> None:
    def row(gain: float) -> dict[str, float | int]:
        return {"anchor_count": 40, "tri_gain_vs_raw_percent": gain}

    identity = {
        "correct_persistent_id": row(20.0),
        "foreground_union": row(10.0),
        "shuffled_persistent_id": row(0.0),
    }
    assert interpret_sam_geometry(
        identity, tolerance_percent=2.0, minimum_anchor_count=10
    ) == "persistent_identity_geometry_evidence"
    foreground = {
        "correct_persistent_id": row(20.0),
        "foreground_union": row(19.0),
        "shuffled_persistent_id": row(0.0),
    }
    assert interpret_sam_geometry(
        foreground, tolerance_percent=2.0, minimum_anchor_count=10
    ) == "foreground_gating_geometry_evidence"
    no_identity = {
        "correct_persistent_id": row(10.0),
        "foreground_union": row(0.0),
        "shuffled_persistent_id": row(9.0),
    }
    assert interpret_sam_geometry(
        no_identity, tolerance_percent=2.0, minimum_anchor_count=10
    ) == "no_persistent_identity_geometry_evidence"


if __name__ == "__main__":
    main()
