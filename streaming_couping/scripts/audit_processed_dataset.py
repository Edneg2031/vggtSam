#!/usr/bin/env python3
"""Audit processed ScanNet++ assets without reading geometry values."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from streaming_couping.src.data import resolve_manifest_path
from streaming_couping.src.storage import expand_storage_path


REVISION = "scannetpp_processed_storage_audit_r1"


def main() -> None:
    args = _parse_args()
    manifest_path = expand_storage_path(args.manifest)
    output_dir = expand_storage_path(args.output_dir)
    payload = json.loads(manifest_path.read_text(encoding="utf8"))
    rows = [
        _audit_scene(scene, manifest_path, minimum_frames=args.minimum_frames)
        for scene in payload.get("scenes", ())
    ]
    eligible = [row["scene_id"] for row in rows if row["geometry_ready"]]
    split = _fixed_split(eligible)
    summary: dict[str, Any] = {
        "schema": 1,
        "revision": REVISION,
        "manifest": str(manifest_path),
        "manifest_metadata_only": 1,
        "pointmap_values_read": 0,
        "instance_mask_values_read": 0,
        "minimum_geometry_frames": int(args.minimum_frames),
        "scene_count": len(rows),
        "geometry_ready_scene_count": len(eligible),
        "geometry_ready_scene_ids": eligible,
        "estimated_listed_asset_bytes": sum(
            int(row["listed_asset_bytes"]) for row in rows
        ),
        "scene_rows": rows,
        "proposed_scene_split": split,
        "cross_scene_split_ready": int(
            bool(split["train"] and split["validation"] and split["test"])
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf8")
    print("PROCESSED SCANNET++ STORAGE AUDIT")
    print(
        f"  scenes={len(rows)} geometry_ready={len(eligible)} "
        f"split_ready={summary['cross_scene_split_ready']}"
    )
    for row in rows:
        print(
            f"  {row['scene_id']}: frames={row['frame_count']} "
            f"rgb+pointmap={row['geometry_complete_frames']} "
            f"mask={row['instance_mask_frames']} ready={row['geometry_ready']}"
        )
    print(f"  result={path}")


def _audit_scene(
    scene: dict[str, Any],
    manifest_path: Path,
    *,
    minimum_frames: int,
) -> dict[str, Any]:
    frames = list(scene.get("frames", ()))
    geometry_complete = 0
    instance_masks = 0
    missing_rgb = 0
    missing_pointmap = 0
    listed_paths: set[Path] = set()
    for frame in frames:
        image = _existing(frame.get("image_path"), manifest_path)
        pointmap = _existing(frame.get("pointmap"), manifest_path)
        mask = _existing(frame.get("instance_mask"), manifest_path)
        missing_rgb += int(image is None)
        missing_pointmap += int(pointmap is None)
        geometry_complete += int(image is not None and pointmap is not None)
        instance_masks += int(mask is not None)
        listed_paths.update(path for path in (image, pointmap, mask) if path is not None)
    size = sum(path.stat().st_size for path in listed_paths)
    return {
        "scene_id": str(scene.get("scene_id", "")),
        "frame_count": len(frames),
        "geometry_complete_frames": geometry_complete,
        "instance_mask_frames": instance_masks,
        "missing_rgb_frames": missing_rgb,
        "missing_pointmap_frames": missing_pointmap,
        "listed_asset_bytes": int(size),
        "geometry_ready": int(geometry_complete >= int(minimum_frames)),
    }


def _existing(value: Any, manifest_path: Path) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        path = resolve_manifest_path(value, manifest_path)
    except FileNotFoundError:
        return None
    return path if path.is_file() else None


def _fixed_split(scene_ids: list[str]) -> dict[str, list[str]]:
    ordered = sorted(dict.fromkeys(scene_ids))
    if len(ordered) < 3:
        return {"train": [], "validation": [], "test": []}
    test_count = max(1, round(len(ordered) * 0.2))
    validation_count = max(1, round(len(ordered) * 0.2))
    if test_count + validation_count >= len(ordered):
        test_count = validation_count = 1
    return {
        "train": ordered[: -(test_count + validation_count)],
        "validation": ordered[-(test_count + validation_count) : -test_count],
        "test": ordered[-test_count:],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default=(
            "${VGGT_SAM_STORAGE_ROOT}/data/processed/"
            "scannetpp_pinhole_2d/manifest.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="${VGGT_SAM_STORAGE_ROOT}/outputs/processed_dataset_audit",
    )
    parser.add_argument("--minimum-frames", type=int, default=30)
    return parser.parse_args()


if __name__ == "__main__":
    main()
