#!/usr/bin/env python3
"""Build a processed RGB cache and a manifest that points to it.

This is separate from 3D rasterization. Existing masks, pointmaps, and camera
metadata are copied from the input manifest without recomputation.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from PIL import Image


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    with manifest_path.open("r", encoding="utf8") as handle:
        manifest = json.load(handle)

    selected_scenes = set(args.scene_ids or [])
    selected_indices = set(args.frame_indices or [])
    cached = 0
    summary_fallbacks = 0
    failures: list[str] = []

    for scene in manifest.get("scenes", []):
        scene_id = str(scene.get("scene_id", ""))
        if selected_scenes and scene_id not in selected_scenes:
            continue
        scene_dir = resolve_scene_dir(scene, manifest_path)
        cache_dir = scene_dir / "images"
        cache_dir.mkdir(parents=True, exist_ok=True)

        for frame_index, frame in enumerate(scene.get("frames", [])):
            if selected_indices and frame_index not in selected_indices:
                continue
            image_name = str(
                frame.get("image_name") or Path(frame["image_path"]).name
            )
            destination = cache_dir / Path(image_name).name
            source = Path(frame["image_path"]).expanduser()
            source_kind = "raw_copy"
            try:
                if destination.exists() and not args.overwrite:
                    source_kind = str(
                        frame.get("image_cache_source", "existing_processed_cache")
                    )
                else:
                    shutil.copyfile(source, destination)
            except OSError as exc:
                if not args.allow_summary_fallback:
                    failures.append(
                        f"frame={scene_id}:{frame_index} source={source}: {exc}"
                    )
                    continue
                try:
                    recover_from_summary(
                        scene_dir=scene_dir,
                        frame=frame,
                        destination=destination,
                    )
                except (OSError, ValueError) as fallback_exc:
                    failures.append(
                        f"frame={scene_id}:{frame_index} source={source}; "
                        f"summary fallback failed: {fallback_exc}"
                    )
                    continue
                source_kind = "summary_rgb_panel_debug_fallback"
                summary_fallbacks += 1

            destination.chmod(0o644)
            frame["image_source_path"] = str(source)
            frame["image_path"] = str(destination.resolve())
            frame["image_cache_source"] = source_kind
            cached += 1

    if failures:
        preview = "\n".join(f"  - {item}" for item in failures[:10])
        raise PermissionError(
            f"Could not cache {len(failures)} RGB frame(s):\n{preview}\n"
            "Grant read permission to the raw images, or rerun with "
            "--allow-summary-fallback for a debug-only RGB recovery."
        )

    output = args.output_manifest or manifest_path.with_name(
        f"{manifest_path.stem}_rgb_cache.json"
    )
    output = output.expanduser().resolve()
    manifest["rgb_cache"] = {
        "cached_frames": cached,
        "summary_debug_fallbacks": summary_fallbacks,
        "source_manifest": str(manifest_path),
    }
    with output.open("w", encoding="utf8") as handle:
        json.dump(manifest, handle, indent=2)
    print(
        f"RGB cache complete cached={cached} "
        f"summary_fallbacks={summary_fallbacks} manifest={output}"
    )
    if summary_fallbacks:
        print(
            "Warning: summary fallbacks contain the visualization title strip. "
            "Use them only to unblock debugging, not for final quantitative results."
        )


def resolve_scene_dir(scene: dict[str, Any], manifest_path: Path) -> Path:
    candidates = []
    output_dir = scene.get("output_dir")
    if output_dir:
        output_path = Path(output_dir).expanduser()
        candidates.extend(
            [
                output_path,
                Path.cwd() / output_path,
                manifest_path.parent / output_path,
            ]
        )
    candidates.append(manifest_path.parent / str(scene.get("scene_id", "")))
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not resolve processed scene directory for {scene.get('scene_id')!r}."
    )


def recover_from_summary(
    *,
    scene_dir: Path,
    frame: dict[str, Any],
    destination: Path,
) -> None:
    stem = Path(str(frame.get("image_name") or frame["image_path"])).stem
    summary = scene_dir / "visualizations" / "summary" / f"{stem}.jpg"
    with Image.open(summary) as image:
        image = image.convert("RGB")
        width = int(frame.get("width") or image.width // 3)
        height = int(frame.get("height") or image.height)
        if image.width < width or image.height < height:
            raise ValueError(
                f"Summary {summary} is smaller than expected RGB size {(width, height)}."
            )
        image.crop((0, 0, width, height)).save(destination, quality=95)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/scannetpp_pinhole_2d/manifest.json"),
    )
    parser.add_argument("--scene-ids", nargs="+")
    parser.add_argument("--frame-indices", type=int, nargs="+")
    parser.add_argument("--output-manifest", type=Path)
    parser.add_argument("--allow-summary-fallback", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
