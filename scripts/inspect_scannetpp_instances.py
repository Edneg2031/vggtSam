#!/usr/bin/env python3
"""List ScanNet++ instances by label, visibility, and mask area."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from vggtsam.data.scannetpp.io import read_json
from vggtsam.data.scannetpp.object_sequence import extract_object_labels, read_mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/scannetpp_pinhole_2d/manifest.json"),
    )
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--label", default=None, help="Substring filter, e.g. picture")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--min-visible-frames", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--window-size", type=int, default=4)
    args = parser.parse_args()

    manifest = read_json(args.manifest)
    scenes = [
        scene
        for scene in manifest.get("scenes", [])
        if scene.get("scene_id") == args.scene_id
    ]
    if not scenes:
        raise SystemExit(f"scene_id={args.scene_id!r} not found in {args.manifest}")
    scene = scenes[0]
    frames = scene.get("frames", [])
    if args.max_frames is not None:
        frames = frames[: args.max_frames]
    object_labels = extract_object_labels(scene.get("objects", {}))

    stats = defaultdict(lambda: {"frames": 0, "pixels": 0, "max_pixels": 0, "first_frame": None})
    for frame_idx, frame in enumerate(frames):
        mask = read_mask(frame["instance_mask"])
        ids, counts = np.unique(mask, return_counts=True)
        for instance_id, count in zip(ids, counts):
            instance_id = int(instance_id)
            if instance_id <= 0:
                continue
            count = int(count)
            item = stats[instance_id]
            item["frames"] += 1
            item["pixels"] += count
            item["max_pixels"] = max(item["max_pixels"], count)
            if item["first_frame"] is None:
                item["first_frame"] = frame_idx

    label_filter = args.label.strip().lower() if args.label else None
    rows = []
    for instance_id, item in stats.items():
        label = object_labels.get(instance_id, "")
        if label_filter and label_filter not in label.lower():
            continue
        if int(item["frames"]) < args.min_visible_frames:
            continue
        rows.append(
            {
                "instance_id": instance_id,
                "label": label,
                "frames": int(item["frames"]),
                "pixels": int(item["pixels"]),
                "max_pixels": int(item["max_pixels"]),
                "first_frame": int(item["first_frame"]),
            }
        )

    rows.sort(key=lambda row: (row["max_pixels"], row["pixels"]), reverse=True)
    rows = rows[: max(1, args.top_k)]
    if not rows:
        raise SystemExit("No matching instances found.")

    print(
        "instance_id\tlabel\tvisible_frames\ttotal_pixels\tmax_pixels\t"
        "first_frame\twindow_index"
    )
    for row in rows:
        window_index = max(
            0,
            min(
                row["first_frame"],
                max(0, len(frames) - max(1, args.window_size)),
            ),
        )
        print(
            f"{row['instance_id']}\t{row['label']}\t{row['frames']}\t"
            f"{row['pixels']}\t{row['max_pixels']}\t{row['first_frame']}\t"
            f"{window_index}"
        )

    best = rows[0]
    best_window = max(
        0,
        min(
            best["first_frame"],
            max(0, len(frames) - max(1, args.window_size)),
        ),
    )
    print("\nSuggested overfit args:")
    print(f"  --instance-id {best['instance_id']} --window-index {best_window}")


if __name__ == "__main__":
    main()
