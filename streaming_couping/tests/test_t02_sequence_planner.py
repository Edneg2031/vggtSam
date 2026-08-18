import json
from pathlib import Path

from streaming_couping.scripts.plan_t02_confirmation_sequence import (
    plan_confirmation_sequences,
)


def test_t02_sequence_planner_is_independent_and_deterministic(tmp_path: Path) -> None:
    paths = {}
    for field in ("image_path", "instance_mask", "pointmap"):
        path = tmp_path / f"{field}.bin"
        path.write_bytes(b"x")
        paths[field] = str(path)
    frames = []
    for _ in range(500):
        frames.append(
            {
                **paths,
                "world_to_camera": [[1.0]],
                "intrinsics": [[1.0]],
            }
        )
    manifest = {
        "scenes": [
            {"scene_id": "old", "frames": frames},
            {"scene_id": "scene_b", "frames": frames},
            {"scene_id": "scene_a", "frames": frames},
        ]
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf8")
    rows, rejected = plan_confirmation_sequences(
        manifest,
        manifest_path=manifest_path,
        discovery_scene_id="old",
        frame_count=30,
        frame_stride=15,
    )
    assert rejected == 0
    assert [row["scene_id"] for row in rows] == ["scene_a", "scene_b"]
    assert all(int(row["eligible"]) == 1 for row in rows)
    assert len(str(rows[0]["frame_indices"]).split()) == 30


def test_t02_sequence_planner_adapts_stride_and_reports_rejection(
    tmp_path: Path,
) -> None:
    existing = tmp_path / "value.bin"
    existing.write_bytes(b"x")
    complete = [
        {
            "image_path": str(existing),
            "instance_mask": str(existing),
            "pointmap": str(existing),
            "world_to_camera": [[1.0]],
            "intrinsics": [[1.0]],
        }
        for _ in range(60)
    ]
    incomplete = [{"image_path": str(existing)} for _ in range(60)]
    manifest = {
        "scenes": [
            {"scene_id": "short_but_valid", "frames": complete},
            {"scene_id": "missing", "frames": incomplete},
        ]
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf8")
    rows, rejected = plan_confirmation_sequences(
        manifest,
        manifest_path=path,
        discovery_scene_id="old",
        frame_count=30,
        frame_stride=15,
    )
    lookup = {row["scene_id"]: row for row in rows}
    assert lookup["short_but_valid"]["frame_stride"] == 2
    assert lookup["short_but_valid"]["eligible"] == 1
    assert lookup["missing"]["eligible"] == 0
    assert str(lookup["missing"]["rejection_reason"]).startswith("missing_fields")
    assert rejected == 1
