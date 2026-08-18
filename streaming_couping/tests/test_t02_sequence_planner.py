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
