import json
from pathlib import Path

from streaming_couping.scripts.materialize_standard_scannetpp_scene import (
    materialize_scene,
)


def test_materialize_r3_clip_matches_standard_scene_layout(tmp_path: Path) -> None:
    source = tmp_path / "source" / "data" / "0a184cf634"
    images = source / "images"
    r3 = tmp_path / "r3"
    images.mkdir(parents=True)
    r3_pointmaps = r3 / "0a184cf634" / "pointmaps"
    r3_pointmaps.mkdir(parents=True)

    frames = []
    for index in range(2):
        image = images / f"DSC{index:05d}.JPG"
        image.write_bytes(b"rgb")
        pointmap = r3_pointmaps / f"{index:02d}_{image.stem}.npz"
        pointmap.write_bytes(b"pointmap")
        frames.append(
            {
                "sequence_index": index,
                "source_sequence_index": index * 11,
                "image_name": image.name,
                "image_path": str(image),
                "image_source_path": str(image),
                "pointmap": str(pointmap),
                "width": 2,
                "height": 2,
            }
        )
    r3_manifest = r3 / "manifest.json"
    r3_manifest.write_text(
        json.dumps({"scenes": [{"scene_id": "0a184cf634", "frames": frames}]}),
        encoding="utf8",
    )

    processed = tmp_path / "processed" / "scannetpp_pinhole_2d"
    existing = {
        "dataset": "scannetpp",
        "scenes": [{"scene_id": "00a231a370", "frames": []}],
    }
    processed.mkdir(parents=True)
    (processed / "manifest.json").write_text(
        json.dumps(existing), encoding="utf8"
    )

    summary = materialize_scene(
        r3_manifest=r3_manifest,
        processed_root=processed,
        scene_id="0a184cf634",
    )
    scene_dir = processed / "0a184cf634"
    assert summary["frame_count"] == 2
    assert (scene_dir / "scene_manifest.json").is_file()
    assert (scene_dir / "semantic_masks").is_dir()
    assert (scene_dir / "instance_masks").is_dir()
    assert (scene_dir / "raster").is_dir()
    assert (scene_dir / "visualizations").is_dir()
    assert (scene_dir / "images" / "DSC00000.JPG").is_symlink()
    assert (scene_dir / "pointmaps" / "DSC00000.npz").is_symlink()

    manifest = json.loads((processed / "manifest.json").read_text(encoding="utf8"))
    assert [scene["scene_id"] for scene in manifest["scenes"]] == [
        "00a231a370",
        "0a184cf634",
    ]
    new_scene = manifest["scenes"][1]
    assert "instance_mask" not in new_scene["frames"][0]
    assert new_scene["frames"][0]["image_storage"] == "symlink"
    assert new_scene["frames"][0]["pointmap_storage"] == "symlink"


def test_materialize_is_idempotent_for_existing_links(tmp_path: Path) -> None:
    image = tmp_path / "image.jpg"
    pointmap = tmp_path / "pointmap.npz"
    image.write_bytes(b"rgb")
    pointmap.write_bytes(b"xyz")
    r3 = tmp_path / "r3.json"
    r3.write_text(
        json.dumps(
            {
                "scenes": [
                    {
                        "scene_id": "heldout",
                        "frames": [
                            {
                                "sequence_index": 0,
                                "image_name": image.name,
                                "image_path": str(image),
                                "pointmap": str(pointmap),
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf8",
    )
    processed = tmp_path / "processed"
    materialize_scene(
        r3_manifest=r3,
        processed_root=processed,
        scene_id="heldout",
    )
    materialize_scene(
        r3_manifest=r3,
        processed_root=processed,
        scene_id="heldout",
    )
    assert (processed / "heldout" / "images" / image.name).is_symlink()
