#!/usr/bin/env python3
"""Relocate copied ScanNet++ manifests without reading array contents."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import shutil
from typing import Any

from streaming_couping.src.storage import expand_storage_path


REVISION = "scannetpp_processed_manifest_relocation_r1"


def main() -> None:
    args = _parse_args()
    processed_root = expand_storage_path(args.processed_root)
    pinhole_root = Path(args.pinhole_root).expanduser().resolve()
    annotation_root = Path(args.annotation_root).expanduser().resolve()
    scene_dirs = sorted(
        path
        for path in processed_root.iterdir()
        if path.is_dir() and (path / "scene_manifest.json").is_file()
    )
    if not scene_dirs:
        raise RuntimeError(f"No scene_manifest.json found below {processed_root}")

    relocated_scenes = []
    scene_rows = []
    for scene_dir in scene_dirs:
        scene, row = _relocate_scene(
            scene_dir,
            pinhole_root=pinhole_root,
            annotation_root=annotation_root,
            write=not args.dry_run,
        )
        relocated_scenes.append(scene)
        scene_rows.append(row)

    manifest_path = processed_root / "manifest.json"
    manifest = _load_json(manifest_path) if manifest_path.is_file() else {}
    manifest.update(
        {
            "dataset": "scannetpp",
            "relocated_at": dt.datetime.now().isoformat(timespec="seconds"),
            "relocation_revision": REVISION,
            "data_root": str(pinhole_root / "data"),
            "output_root": str(processed_root),
            "scenes": relocated_scenes,
        }
    )
    if not args.dry_run:
        _backup_once(manifest_path)
        _write_json(manifest_path, manifest)

    incomplete = sum(int(row["incomplete_frames"]) for row in scene_rows)
    summary = {
        "schema": 1,
        "revision": REVISION,
        "processed_root": str(processed_root),
        "manifest": str(manifest_path),
        "pinhole_root": str(pinhole_root),
        "annotation_root": str(annotation_root),
        "scene_count": len(scene_rows),
        "incomplete_frame_count": incomplete,
        "geometry_values_read": 0,
        "mask_values_read": 0,
        "dry_run": int(args.dry_run),
        "scenes": scene_rows,
    }
    summary_path = processed_root / "relocation_summary.json"
    if not args.dry_run:
        _write_json(summary_path, summary)

    print("SCANNET++ PROCESSED MANIFEST RELOCATION")
    for row in scene_rows:
        print(
            f"  {row['scene_id']}: frames={row['frame_count']} "
            f"complete={row['complete_frames']} "
            f"incomplete={row['incomplete_frames']} "
            f"images={row['image_storage']}"
        )
    print(f"  manifest={manifest_path}")
    print(f"  incomplete_frames={incomplete}")
    if incomplete and not args.allow_incomplete:
        raise RuntimeError(
            "Relocated manifest still contains incomplete frames; inspect "
            f"{summary_path} or rerun with --allow-incomplete for planning only."
        )


def _relocate_scene(
    scene_dir: Path,
    *,
    pinhole_root: Path,
    annotation_root: Path,
    write: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = scene_dir / "scene_manifest.json"
    scene = _load_json(path)
    scene_id = str(scene.get("scene_id") or scene_dir.name)
    raw_scene = pinhole_root / "data" / scene_id
    annotation_scan = annotation_root / "data" / scene_id / "scans"
    complete = 0
    incomplete = 0
    storage_counts: dict[str, int] = {}
    missing_examples = []

    frames = list(scene.get("frames", ()))
    for index, frame in enumerate(frames):
        image_name = str(frame.get("image_name") or "")
        basename = Path(image_name).name
        if not basename:
            existing = frame.get("image_path") or frame.get("image_source_path")
            basename = Path(str(existing)).name
        stem = Path(basename).stem

        processed_image = scene_dir / "images" / basename
        raw_image = raw_scene / "images" / basename
        current_image = Path(str(frame.get("image_path", ""))).expanduser()
        # Prefer the original RGB path, matching the no-copy/no-link policy.
        image = _first_file(raw_image, processed_image, current_image)
        if image is None:
            image = raw_image
        if image == processed_image:
            image_storage = "symlink" if image.is_symlink() else "processed_file"
        else:
            image_storage = "source_reference"

        semantic = scene_dir / "semantic_masks" / f"{stem}.png"
        instance = scene_dir / "instance_masks" / f"{stem}.png"
        pointmap = scene_dir / "pointmaps" / f"{stem}.npz"
        raster = scene_dir / "raster" / f"{stem}.npz"
        frame.update(
            {
                "image_path": str(image.resolve() if image.is_symlink() else image),
                "image_source_path": str(raw_image if raw_image.is_file() else image),
                "image_storage": image_storage,
                "semantic_mask": str(semantic),
                "instance_mask": str(instance),
                "pointmap": str(pointmap),
                "raster": str(raster) if raster.is_file() else None,
            }
        )
        missing = [
            name
            for name, asset in (
                ("image", image),
                ("semantic_mask", semantic),
                ("instance_mask", instance),
                ("pointmap", pointmap),
            )
            if not asset.is_file()
        ]
        storage_counts[image_storage] = storage_counts.get(image_storage, 0) + 1
        if missing:
            incomplete += 1
            if len(missing_examples) < 20:
                missing_examples.append({"frame_index": index, "missing": missing})
        else:
            complete += 1

    scene.update(
        {
            "scene_id": scene_id,
            "scene_root": str(raw_scene),
            "output_dir": str(scene_dir),
            "annotation_mesh_path": str(
                annotation_scan / "mesh_aligned_0.05_semantic.ply"
            ),
            "pinhole_mesh_path": str(raw_scene / "mesh_aligned_0.05.ply"),
            "segments_path": str(annotation_scan / "segments.json"),
            "annotations_path": str(annotation_scan / "segments_anno.json"),
            "frames": frames,
            "relocation_revision": REVISION,
        }
    )
    if write:
        _backup_once(path)
        _write_json(path, scene)
    return scene, {
        "scene_id": scene_id,
        "frame_count": len(frames),
        "complete_frames": complete,
        "incomplete_frames": incomplete,
        "image_storage": storage_counts,
        "missing_examples": missing_examples,
    }


def _first_file(*paths: Path) -> Path | None:
    return next((path for path in paths if str(path) and path.is_file()), None)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf8"))


def _backup_once(path: Path) -> None:
    if not path.is_file():
        return
    backup = path.with_suffix(path.suffix + ".before_relocation")
    if not backup.exists():
        shutil.copy2(path, backup)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf8",
    )
    temporary.replace(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-root",
        default=(
            "${VGGT_SAM_STORAGE_ROOT}/data/processed/"
            "scannetpp_pinhole_2d"
        ),
    )
    parser.add_argument(
        "--pinhole-root",
        default="/data184/open_source/scannet_pp_pinhole",
    )
    parser.add_argument(
        "--annotation-root",
        default="/data184/open_source/scannet_pp",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
