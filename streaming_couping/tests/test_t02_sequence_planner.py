import json
from pathlib import Path

from streaming_couping.scripts.plan_t02_confirmation_sequence import (
    plan_confirmation_sequences,
)


def test_t02_sequence_planner_selects_centered_unseen_time_holdout(
    tmp_path: Path,
) -> None:
    paths = {}
    for field in ("image_path", "instance_mask", "pointmap"):
        path = tmp_path / f"{field}.bin"
        path.write_bytes(b"x")
        paths[field] = str(path)
    frames = []
    for _ in range(600):
        frames.append(
            {
                **paths,
                "world_to_camera": [[1.0]],
                "intrinsics": [[1.0]],
            }
        )
    manifest = {
        "scenes": [
            {"scene_id": "discovery", "frames": frames},
            {"scene_id": "unrelated", "frames": frames},
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf8")
    rows, rejected = plan_confirmation_sequences(
        manifest,
        manifest_path=manifest_path,
        discovery_scene_id="discovery",
        discovery_last_frame=525,
        frame_count=30,
        frame_stride=2,
    )
    assert rejected == 0
    assert len(rows) == 1
    assert rows[0]["scene_id"] == "discovery"
    assert int(rows[0]["eligible"]) == 1
    assert rows[0]["first_frame"] == 533
    assert rows[0]["last_frame"] == 591
    assert rows[0]["frame_stride"] == 2
    assert len(str(rows[0]["frame_indices"]).split()) == 30


def test_t02_sequence_planner_reports_missing_holdout_files(
    tmp_path: Path,
) -> None:
    existing = tmp_path / "value.bin"
    existing.write_bytes(b"x")
    prefix = [
        {
            "image_path": str(existing),
            "instance_mask": str(existing),
            "pointmap": str(existing),
            "world_to_camera": [[1.0]],
            "intrinsics": [[1.0]],
        }
        for _ in range(526)
    ]
    incomplete = [{"image_path": str(existing)} for _ in range(74)]
    manifest = {
        "scenes": [
            {"scene_id": "discovery", "frames": prefix + incomplete},
        ]
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf8")
    rows, rejected = plan_confirmation_sequences(
        manifest,
        manifest_path=path,
        discovery_scene_id="discovery",
        discovery_last_frame=525,
        frame_count=30,
        frame_stride=2,
    )
    assert len(rows) == 1
    assert rows[0]["eligible"] == 0
    assert str(rows[0]["rejection_reason"]).startswith("missing_fields")
    assert rejected == 1
