#!/usr/bin/env python3
"""Materialize an R3 clip in the standard ScanNet++ processed layout.

The R3 geometry preparation intentionally writes an isolated, annotation-free
manifest under ``residual_cross_scene_r3``.  This utility exposes the same clip
under the normal ``scannetpp_pinhole_2d/<scene_id>`` layout without duplicating
RGB images or pointmaps.  RGB and pointmaps are linked when the filesystem
allows it; a source reference is used as a safe fallback on restrictive
mounts.  No mask is fabricated when ScanNet++ annotation assets are absent.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
from typing import Any


REVISION = "standard_scannetpp_scene_materialization_r1"
DEFAULT_SCENE = "0a184cf634"


def materialize_scene(
    *,
    r3_manifest: Path,
    processed_root: Path,
    scene_id: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create/update one standard processed scene and the global manifest."""

    r3_manifest = Path(r3_manifest).expanduser().resolve()
    processed_root = Path(processed_root).expanduser().resolve()
    if not r3_manifest.is_file():
        raise FileNotFoundError(f"Missing R3 manifest: {r3_manifest}")
    payload = _load_json(r3_manifest)
    scenes = payload.get("scenes", [])
    if len(scenes) != 1:
        raise ValueError(f"R3 manifest must contain one scene, got {len(scenes)}")
    source_scene = scenes[0]
    actual_scene = str(source_scene.get("scene_id", "")).strip()
    scene = str(scene_id).strip()
    if actual_scene != scene:
        raise ValueError(
            f"Scene mismatch: requested={scene!r}, manifest={actual_scene!r}"
        )
    frames = list(source_scene.get("frames", ()))
    if not frames:
        raise ValueError(f"R3 manifest has no frames for {scene}")
    sequence_indices = [int(row.get("sequence_index", -1)) for row in frames]
    if sequence_indices != list(range(len(frames))):
        raise ValueError("R3 sequence_index values must be contiguous")

    scene_dir = processed_root / scene
    dirs = {
        name: scene_dir / name
        for name in (
            "images",
            "semantic_masks",
            "instance_masks",
            "pointmaps",
            "raster",
            "visualizations",
        )
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    output_frames: list[dict[str, Any]] = []
    storage_counts: dict[str, int] = {}
    pointmap_storage_counts: dict[str, int] = {}
    for row in frames:
        image_source = _source_path(row, "image_source_path", "image_path")
        pointmap_source = _source_path(row, "pointmap")
        if not image_source.is_file():
            raise FileNotFoundError(f"Missing RGB source: {image_source}")
        if not pointmap_source.is_file():
            raise FileNotFoundError(f"Missing R3 pointmap: {pointmap_source}")

        image_name = Path(str(row.get("image_name") or image_source.name)).name
        image_destination = dirs["images"] / image_name
        image_path, image_storage = _link_or_reference(
            image_source, image_destination, overwrite=overwrite
        )

        stem = Path(image_name).stem
        pointmap_destination = dirs["pointmaps"] / f"{stem}.npz"
        pointmap_path, pointmap_storage = _link_or_reference(
            pointmap_source, pointmap_destination, overwrite=overwrite
        )

        frame = dict(row)
        frame.update(
            {
                "image_name": image_name,
                "image_path": str(image_path),
                "image_source_path": str(image_source),
                "image_storage": image_storage,
                "pointmap": str(pointmap_path),
                "pointmap_source_path": str(pointmap_source),
                "pointmap_storage": pointmap_storage,
                "raster": None,
            }
        )
        # Do not add semantic/instance fields unless real files are present.
        # A missing mask is preferable to silently turning an annotation-free
        # geometry clip into a fake semantic dataset.
        for field, directory in (
            ("semantic_mask", dirs["semantic_masks"]),
            ("instance_mask", dirs["instance_masks"]),
        ):
            candidate = directory / f"{stem}.png"
            if candidate.is_file():
                frame[field] = str(candidate)
        output_frames.append(frame)
        storage_counts[image_storage] = storage_counts.get(image_storage, 0) + 1
        pointmap_storage_counts[pointmap_storage] = (
            pointmap_storage_counts.get(pointmap_storage, 0) + 1
        )

    scene_manifest = dict(source_scene)
    scene_manifest.update(
        {
            "scene_id": scene,
            "output_dir": str(scene_dir),
            "scene_root": str(
                Path(str(source_scene.get("scene_root", ""))).expanduser()
            ),
            "layout_revision": REVISION,
            "layout_role": "standard_scannetpp_processed_scene_from_r3_clip",
            "annotation_mesh_path": None,
            "pinhole_mesh_path": source_scene.get("mesh_path"),
            "segments_path": None,
            "annotations_path": None,
            "semantic_source": "unavailable_annotation_assets",
            "objects": {},
            "semantic_masks_generated": 0,
            "instance_masks_generated": 0,
            "frames": output_frames,
        }
    )
    scene_manifest_path = scene_dir / "scene_manifest.json"
    _write_json(scene_manifest_path, scene_manifest)

    global_manifest_path = processed_root / "manifest.json"
    global_manifest = _load_json(global_manifest_path) if global_manifest_path.is_file() else {}
    existing = {
        str(item.get("scene_id")): item
        for item in global_manifest.get("scenes", [])
        if isinstance(item, dict) and item.get("scene_id")
    }
    existing[scene] = scene_manifest
    global_manifest.update(
        {
            "dataset": "scannetpp",
            "created_at": global_manifest.get("created_at") or dt.datetime.now().isoformat(timespec="seconds"),
            "updated_at": dt.datetime.now().isoformat(timespec="seconds"),
            "output_root": str(processed_root),
            "layout_revision": REVISION,
            "scenes": [existing[key] for key in sorted(existing)],
        }
    )
    _write_json(global_manifest_path, global_manifest)

    summary = {
        "schema": 1,
        "revision": REVISION,
        "status": "READY",
        "scene_id": scene,
        "frame_count": len(output_frames),
        "processed_root": str(processed_root),
        "scene_dir": str(scene_dir),
        "scene_manifest": str(scene_manifest_path),
        "global_manifest": str(global_manifest_path),
        "r3_manifest": str(r3_manifest),
        "rgb_link_or_reference_counts": storage_counts,
        "pointmap_link_or_reference_counts": pointmap_storage_counts,
        "semantic_masks_present": int(
            all("semantic_mask" in frame for frame in output_frames)
        ),
        "instance_masks_present": int(
            all("instance_mask" in frame for frame in output_frames)
        ),
        "annotations_read": 0,
        "sam_loaded_or_run": 0,
        "gt_values_used_for_layout": 0,
    }
    _write_json(scene_dir / "layout_summary.json", summary)
    _write_copyable(scene_dir / "copyable_layout.txt", summary)
    return summary


def _source_path(row: dict[str, Any], *fields: str) -> Path:
    for field in fields:
        value = row.get(field)
        if value:
            return Path(str(value)).expanduser().resolve()
    raise ValueError(f"Frame has no path field among {fields}: {row}")


def _link_or_reference(
    source: Path,
    destination: Path,
    *,
    overwrite: bool,
) -> tuple[Path, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() and destination.resolve() == source.resolve():
            return destination, "symlink"
        if not overwrite:
            raise FileExistsError(
                f"Destination exists and does not match source: {destination}"
            )
        if destination.is_dir() and not destination.is_symlink():
            raise IsADirectoryError(destination)
        destination.unlink()
    try:
        destination.symlink_to(source)
        return destination, "symlink"
    except OSError:
        # Some network mounts reject symlinks. A hardlink is still zero-copy;
        # if that also fails, keep an absolute source reference in the manifest.
        try:
            os.link(source, destination)
            return destination, "hardlink"
        except OSError:
            return source, "source_reference"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf8"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf8")
    temporary.replace(path)


def _write_copyable(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "===== COPYABLE_STANDARD_SCANNETPP_LAYOUT_BEGIN =====",
        f"revision={summary['revision']}",
        f"status={summary['status']}",
        f"scene_id={summary['scene_id']}",
        f"frame_count={summary['frame_count']}",
        f"processed_root={summary['processed_root']}",
        f"scene_dir={summary['scene_dir']}",
        f"scene_manifest={summary['scene_manifest']}",
        f"global_manifest={summary['global_manifest']}",
        f"r3_manifest={summary['r3_manifest']}",
        f"rgb_link_or_reference_counts={json.dumps(summary['rgb_link_or_reference_counts'], sort_keys=True)}",
        f"pointmap_link_or_reference_counts={json.dumps(summary['pointmap_link_or_reference_counts'], sort_keys=True)}",
        f"semantic_masks_present={summary['semantic_masks_present']}",
        f"instance_masks_present={summary['instance_masks_present']}",
        "annotations_read=0",
        "sam_loaded_or_run=0",
        "gt_values_used_for_layout=0",
        "===== COPYABLE_STANDARD_SCANNETPP_LAYOUT_END =====",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r3-manifest", required=True)
    parser.add_argument("--processed-root", required=True)
    parser.add_argument("--scene-id", default=DEFAULT_SCENE)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = materialize_scene(
        r3_manifest=Path(args.r3_manifest),
        processed_root=Path(args.processed_root),
        scene_id=args.scene_id,
        overwrite=args.overwrite,
    )
    print("STANDARD SCANNET++ SCENE LAYOUT READY")
    print(
        f"  scene={result['scene_id']} frames={result['frame_count']} "
        f"rgb={result['rgb_link_or_reference_counts']} "
        f"pointmaps={result['pointmap_link_or_reference_counts']}"
    )
    print(f"  scene_manifest={result['scene_manifest']}")
    print(f"  global_manifest={result['global_manifest']}")
