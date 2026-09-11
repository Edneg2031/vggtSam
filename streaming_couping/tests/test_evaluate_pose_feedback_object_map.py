"""Does the corrected trajectory put the objects closer to the truth?

The whole point of the object-map check is that it must NOT be circular: if the
reference cloud were built through the raw poses, a correction that moves away
from the drifted trajectory would read as moving away from truth.  The first
test pins that down by building a scene where the raw pose is wrong and the
corrected one is right.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from streaming_couping.scripts.evaluate_pose_feedback_object_map import (
    METRIC_KEYS,
    _camera_points,
    _delta,
    _transform,
    predicted_clouds,
    reference_clouds,
    render,
    score_trajectory,
)


def _intrinsics() -> torch.Tensor:
    return torch.tensor([[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]])


def test_backprojection_inverts_the_pinhole_model() -> None:
    depth = torch.full((3, 3), 4.0)
    pixels = torch.tensor([[0, 0], [1, 1], [2, 2]])
    points = _camera_points(depth, _intrinsics(), pixels)
    # fx=2, cx=1: column 0 -> x = (0-1)/2*4 = -2, and so on
    assert torch.allclose(points[0], torch.tensor([-2.0, -2.0, 4.0]))
    assert torch.allclose(points[1], torch.tensor([0.0, 0.0, 4.0]))
    assert torch.allclose(points[2], torch.tensor([2.0, 2.0, 4.0]))


def test_reference_clouds_use_the_ground_truth_poses() -> None:
    """The reference must be built from GT poses, not from the raw ones."""

    mask = torch.zeros(1, 1, 3, 3, dtype=torch.bool)
    mask[0, 0, 1, 1] = True
    depth = torch.full((1, 3, 3), 4.0)
    identity = torch.eye(4)
    shifted = torch.eye(4)
    shifted[0, 3] = 10.0  # a trajectory that is badly wrong

    clouds = reference_clouds(
        masks=mask,
        labels=["bed"],
        depth=depth,
        intrinsics=_intrinsics()[None],
        gt_c2w=identity[None],
    )
    # through the identity the single pixel at (1,1) sits at (0,0,4)
    assert torch.allclose(clouds["bed"], torch.tensor([[0.0, 0.0, 4.0]]))

    shifted_clouds = reference_clouds(
        masks=mask,
        labels=["bed"],
        depth=depth,
        intrinsics=_intrinsics()[None],
        gt_c2w=shifted[None],
    )
    # the reference follows the poses it is given, which is why the caller must
    # pass the ground-truth ones
    assert torch.allclose(shifted_clouds["bed"], torch.tensor([[10.0, 0.0, 4.0]]))


def test_a_corrected_trajectory_scores_better_than_raw() -> None:
    """The measured quantity: same points, only the pose changes."""

    mask = torch.zeros(1, 1, 3, 3, dtype=torch.bool)
    mask[0, 0, 1, 1] = True
    depth = torch.full((1, 3, 3), 4.0)
    identity = torch.eye(4)
    reference = reference_clouds(
        masks=mask,
        labels=["bed"],
        depth=depth,
        intrinsics=_intrinsics()[None],
        gt_c2w=identity[None],
    )
    observations = [
        {
            "frame_id": 0,
            "category": "bed",
            "points_camera": torch.tensor([[0.0, 0.0, 4.0]]),
        }
    ]
    drifted = torch.eye(4)
    drifted[0, 3] = 0.5
    corrected = torch.eye(4)

    raw_scores = score_trajectory(
        predicted=predicted_clouds(observations=observations, trajectory=drifted[None]),
        reference=reference,
    )
    fixed_scores = score_trajectory(
        predicted=predicted_clouds(observations=observations, trajectory=corrected[None]),
        reference=reference,
    )
    assert raw_scores["bed"]["object_accuracy_m"] == pytest.approx(0.5)
    assert fixed_scores["bed"]["object_accuracy_m"] == pytest.approx(0.0)
    assert (
        fixed_scores["bed"]["fscore_5cm"] > raw_scores["bed"]["fscore_5cm"]
    )
    assert "all" in fixed_scores


def test_a_category_without_a_prediction_is_marked_not_scored() -> None:
    mask = torch.zeros(1, 1, 3, 3, dtype=torch.bool)
    mask[0, 0, 1, 1] = True
    reference = reference_clouds(
        masks=mask,
        labels=["bed"],
        depth=torch.full((1, 3, 3), 4.0),
        intrinsics=_intrinsics()[None],
        gt_c2w=torch.eye(4)[None],
    )
    scores = score_trajectory(predicted={}, reference=reference)
    assert scores["bed"]["skipped"] == "no_prediction_for_category"
    assert not any(key in scores for key in ("all",))


def test_predictions_are_pooled_per_category() -> None:
    observations = [
        {
            "frame_id": 0,
            "category": "bed",
            "points_camera": torch.tensor([[0.0, 0.0, 1.0]]),
        },
        {
            "frame_id": 0,
            "category": "bed",
            "points_camera": torch.tensor([[1.0, 0.0, 1.0]]),
        },
        {
            "frame_id": 0,
            "category": "rug",
            "points_camera": torch.tensor([[2.0, 0.0, 1.0]]),
        },
    ]
    clouds = predicted_clouds(observations=observations, trajectory=torch.eye(4)[None])
    assert set(clouds) == {"bed", "rug"}
    assert clouds["bed"].shape[0] == 2
    assert clouds["rug"].shape[0] == 1


def test_an_out_of_range_frame_is_a_clear_error() -> None:
    with pytest.raises(ValueError, match="outside the trajectory"):
        predicted_clouds(
            observations=[
                {
                    "frame_id": 9,
                    "category": "bed",
                    "points_camera": torch.zeros(1, 3),
                }
            ],
            trajectory=torch.eye(4)[None],
        )


def test_render_shows_raw_first_then_deltas() -> None:
    block = {key: 0.1 for key in METRIC_KEYS}
    results = {
        "raw": {"bed": dict(block)},
        "robust_semantic": {"bed": {**block, "object_accuracy_m": 0.08}},
    }
    text = render(results)
    assert text.index("raw") < text.index("robust_semantic")
    assert "Δ vs raw robust_semantic" in text
    assert _delta(0.08, 0.1) == pytest.approx(-0.02)
    assert _delta(None, 0.1) is None


def test_transform_applies_rotation_then_translation() -> None:
    pose = torch.eye(4)
    pose[0, 3] = 1.0
    points = torch.tensor([[0.0, 0.0, 0.0]])
    assert torch.allclose(_transform(points, pose), torch.tensor([[1.0, 0.0, 0.0]]))


def test_cli_scores_a_synthetic_run(tmp_path: Path) -> None:
    from streaming_couping.scripts.evaluate_pose_feedback_object_map import main

    size = 4
    run_dir = tmp_path / "run"
    feedback = run_dir / "object_pose_feedback"
    refinement = run_dir / "object_pose_refinement"
    feedback.mkdir(parents=True)
    refinement.mkdir(parents=True)

    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    labels = np.zeros((size, size), dtype=np.uint16)
    labels[1:3, 1:3] = 7
    Image.fromarray(labels).save(mask_dir / "frame0.png")
    manifest = {
        "scenes": [
            {
                "scene_id": "scene",
                "objects": {"7": {"label": "bed"}},
                "frames": [
                    {"instance_mask": str(mask_dir / "frame0.png"),
                     "world_to_camera": np.eye(4)[:3].tolist()}
                ],
            }
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    cache = {
        "schema": "horizonstream_semantic_geometry",
        "schema_version": 1,
        "backend": "horizonstream",
        "depth": torch.full((1, size, size), 4.0),
        "confidence": torch.ones(1, size, size),
        "world_to_camera": torch.eye(3, 4)[None].expand(1, 3, 4).contiguous(),
        "intrinsics": _intrinsics()[None],
        "processed_rgb": torch.zeros(1, size, size, 3, dtype=torch.uint8),
        "processed_size": (size, size),
        "scale_type": "metric",
        "image_paths": ["frame0.png"],
        "source_sizes": [(size, size)],
        "source_positions": [0],
        "frame_ids": [0],
    }
    cache_path = tmp_path / "geometry.pt"
    torch.save(cache, cache_path)

    drifted = torch.eye(4)
    drifted[0, 3] = 0.30
    torch.save(
        {
            "raw_c2w": drifted[None],
            "gt_c2w": torch.eye(4)[None],
            "gt_deltas": torch.eye(4)[None],
            "variant_c2w": {"robust_semantic": torch.eye(4)[None]},
        },
        feedback / "poses.pt",
    )
    torch.save(
        {
            "schema": "object_pose_loss_feedback_diagnostics_r1",
            "observations": [
                {
                    "frame_id": 0,
                    "category": "bed",
                    "points_camera": torch.tensor([[0.0, 0.0, 4.0]]),
                }
            ],
        },
        refinement / "feedback_diagnostics.pt",
    )

    argv = [
        "evaluate",
        "--run-dir", str(run_dir),
        "--geometry-cache", str(cache_path),
        "--manifest", str(manifest_path),
        "--scene-id", "scene",
        "--prompts", "bed",
        "--json-out", str(tmp_path / "out.json"),
    ]
    import sys

    original = sys.argv
    sys.argv = argv
    try:
        main()
    finally:
        sys.argv = original
    payload = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert set(payload) == {"raw", "robust_semantic"}
    # the raw trajectory is 0.30 m off, the corrected one is exact
    assert payload["raw"]["bed"]["object_accuracy_m"] == pytest.approx(0.30, abs=1e-5)
    assert payload["robust_semantic"]["bed"]["object_accuracy_m"] == pytest.approx(
        0.0, abs=1e-5
    )
