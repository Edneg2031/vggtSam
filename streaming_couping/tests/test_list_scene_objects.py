"""The scene inventory: what exists, what is visible, and what the run misses.

Prompts are a hard ceiling -- an object no prompt matches cannot be proposed,
corrected or scored -- so the check has to name the ones that fall outside,
both on the ground-truth side and on the segmenter side.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from streaming_couping.scripts.list_scene_objects import (
    frame_count,
    main,
    sam_inventory,
    scene_objects,
    selection,
)

GRID = 24


def _write_manifest(root: Path, *, frames: int = 3) -> Path:
    masks = root / "masks"
    masks.mkdir(parents=True, exist_ok=True)
    entries = []
    for index in range(frames):
        labels = np.zeros((GRID, GRID), dtype=np.uint16)
        labels[2:14, 2:14] = 1  # bed: 144 px, comfortably above the floor
        labels[16:18, 16:18] = 2  # chair: 4 px, below the 32 px floor
        if index >= 1:
            labels[20:23, 2:8] = 3  # pet bed: 18 px, appears from frame 1
        path = masks / f"f{index}.png"
        Image.fromarray(labels).save(path)
        entries.append(
            {
                "instance_mask": str(path),
                "world_to_camera": np.eye(4)[:3].tolist(),
            }
        )
    manifest = {
        "scenes": [
            {
                "scene_id": "s",
                "objects": {
                    "1": {"label": "bed"},
                    "2": {"label": "chair"},
                    "3": {"label": "pet bed"},
                    "4": {"label": "lamp"},  # never visible in these frames
                },
                "frames": entries,
            }
        ]
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _diagnostics(root: Path) -> Path:
    path = root / "diag.pt"
    torch.save(
        {
            "schema": "object_pose_loss_feedback_diagnostics_r1",
            "observations": [
                {
                    "frame_id": 0,
                    "instance_id": 0,
                    "category": "bed",
                    "points_camera": torch.zeros(100, 3),
                },
                {
                    "frame_id": 0,
                    "instance_id": 0,
                    "category": "bed",
                    "points_camera": torch.zeros(50, 3),
                },
                {
                    "frame_id": 0,
                    "instance_id": 1,
                    "category": "rug",
                    "points_camera": torch.zeros(20, 3),
                },
            ],
        },
        path,
    )
    return path


def test_scene_objects_reads_the_manifest_labels(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    assert scene_objects(manifest, "s") == {
        1: "bed",
        2: "chair",
        3: "pet bed",
        4: "lamp",
    }
    assert frame_count(manifest, "s") == 3


def test_scene_objects_rejects_an_unknown_scene(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    with pytest.raises(ValueError, match="absent"):
        scene_objects(manifest, "nope")


def test_selection_without_a_cache_follows_the_frame_flags(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, frames=10)

    class Args:
        geometry_cache = None
        scene_id = "s"
        frame_start = 2
        frame_stride = 3
        frame_count = 4
        image_size = GRID

    args = Args()
    args.manifest = manifest
    positions, grid = selection(args)
    # count is a number of frames: 2, 5, 8, 11 truncated to the scene length
    assert positions == [2, 5, 8]
    assert grid == (GRID, GRID)


def test_selection_with_a_cache_uses_its_own_frames_and_grid(tmp_path: Path) -> None:
    """The inventory must describe the run's frames, not an assumed window."""

    cache = tmp_path / "geometry.pt"
    torch.save(
        {
            "source_positions": [90, 91, 95],
            "depth": torch.zeros(3, 8, 12),
        },
        cache,
    )

    class Args:
        geometry_cache = cache
        scene_id = "s"
        manifest = tmp_path / "unused.json"
        frame_start = 0
        frame_stride = 1
        frame_count = 0
        image_size = 518

    positions, grid = selection(Args())
    assert positions == [90, 91, 95]
    assert grid == (8, 12)


def test_sam_inventory_aggregates_by_category(tmp_path: Path) -> None:
    inventory = sam_inventory(_diagnostics(tmp_path))
    assert set(inventory) == {"bed", "rug"}
    assert inventory["bed"]["observations"] == 2
    assert inventory["bed"]["points"] == 150
    assert inventory["bed"]["instances"] == [0]


def test_cli_names_every_side_of_the_mismatch(
    tmp_path: Path, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _write_manifest(tmp_path)
    diagnostics = _diagnostics(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "list_scene_objects",
            "--manifest", str(manifest),
            "--scene-id", "s",
            "--geometry-cache", str(_write_cache(tmp_path)),
            "--feedback-diagnostics", str(diagnostics),
            "--prompts", "bed", "wardrobe", "chair", "rug", "dustbin",
        ],
    )
    main()
    output = capsys.readouterr().out

    # the run's grid and frames are reported
    assert "grid=(24,24)" in output
    # an object defined but never visible is named
    assert "never visible" in output
    assert "'lamp'" in output
    # an object no prompt reaches would be named here; "bed" also matches
    # "pet bed", so this scene has none, and the script says so
    assert "every visible ground-truth object is reached by a prompt" in output
    # the mask filter that would drop an object is attributed to the object,
    # with the reason (its size) rather than only a count
    assert "would drop in every frame" in output
    assert "chair#2(4px)" in output
    # a segmenter category with no ground-truth counterpart is called out
    assert "no ground-truth object" in output
    assert "'rug'" in output


def _write_cache(root: Path, *, frame_count: int = 3) -> Path:
    path = root / "geometry.pt"
    torch.save(
        {
            "source_positions": list(range(frame_count)),
            "depth": torch.zeros(frame_count, GRID, GRID),
        },
        path,
    )
    return path


def test_cli_reports_an_object_no_prompt_matches(
    tmp_path: Path, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """'pet bed' is matched by 'bed' only through substring matching; drop that
    prompt and the object becomes unreachable."""

    manifest = _write_manifest(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "list_scene_objects",
            "--manifest", str(manifest),
            "--scene-id", "s",
            "--geometry-cache", str(_write_cache(tmp_path)),
            "--prompts", "wardrobe", "chair",
        ],
    )
    main()
    output = capsys.readouterr().out
    assert "no prompt reaches" in output
    assert "'bed'" in output
    assert "'pet bed'" in output


def test_report_out_puts_the_table_in_a_file_and_summarises_stdout(
    tmp_path: Path, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A seventy-object scene is reference material, not a headline."""

    manifest = _write_manifest(tmp_path)
    report = tmp_path / "out" / "reachability.txt"
    monkeypatch.setattr(
        "sys.argv",
        [
            "list_scene_objects",
            "--manifest", str(manifest),
            "--scene-id", "s",
            "--geometry-cache", str(_write_cache(tmp_path)),
            "--prompts", "bed", "wardrobe",
            "--report-out", str(report),
        ],
    )
    main()
    printed = capsys.readouterr().out
    body = report.read_text(encoding="utf-8")

    # a row that only exists in the table, and the grid it was measured on
    assert "pet bed" not in printed
    assert "pet bed" in body
    assert "grid=(24,24)" in body
    # what stays on stdout is the count that says whether to act
    assert "visible object(s) of" in printed
    assert "reached by a prompt" in printed
    assert str(report) in printed
