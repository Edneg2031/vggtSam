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
    parser.add_argument("--sequence-length", type=int, default=4)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--instance-id", type=int, default=None)
    parser.add_argument("--frame-indices", type=int, nargs="+", default=None)
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

    per_frame_counts = []
    stats = defaultdict(lambda: {"frames": 0, "pixels": 0, "max_pixels": 0, "first_frame": None})
    for frame_idx, frame in enumerate(frames):
        mask = read_mask(frame["instance_mask"])
        ids, counts = np.unique(mask, return_counts=True)
        frame_counts = {}
        for instance_id, count in zip(ids, counts):
            instance_id = int(instance_id)
            if instance_id <= 0:
                continue
            count = int(count)
            frame_counts[instance_id] = count
            item = stats[instance_id]
            item["frames"] += 1
            item["pixels"] += count
            item["max_pixels"] = max(item["max_pixels"], count)
            if item["first_frame"] is None:
                item["first_frame"] = frame_idx
        per_frame_counts.append(frame_counts)

    if args.instance_id is not None:
        print_instance_timeline(
            per_frame_counts,
            object_labels,
            instance_id=int(args.instance_id),
            frame_indices=args.frame_indices,
        )
        return

    label_filter = args.label.strip().lower() if args.label else None
    rows = []
    for instance_id, item in stats.items():
        label = object_labels.get(instance_id, "")
        if label_filter and label_filter not in label.lower():
            continue
        if int(item["frames"]) < args.min_visible_frames:
            continue
        window = best_window_for_instance(
            per_frame_counts,
            instance_id=instance_id,
            sequence_length=args.sequence_length,
            frame_stride=args.frame_stride,
            min_visible_frames=args.min_visible_frames,
        )
        if window is None:
            continue
        rows.append(
            {
                "instance_id": instance_id,
                "label": label,
                "frames": int(item["frames"]),
                "pixels": int(item["pixels"]),
                "max_pixels": int(item["max_pixels"]),
                "first_frame": int(item["first_frame"]),
                "window_index": int(window["window_index"]),
                "window_pixels": int(window["window_pixels"]),
                "window_visible_frames": int(window["window_visible_frames"]),
                "frame_indices": window["frame_indices"],
            }
        )

    rows.sort(
        key=lambda row: (
            row["window_visible_frames"],
            row["window_pixels"],
            row["max_pixels"],
        ),
        reverse=True,
    )
    rows = rows[: max(1, args.top_k)]
    if not rows:
        raise SystemExit("No matching instances found.")

    print(
        "instance_id\tlabel\tvisible_frames\ttotal_pixels\tmax_pixels\t"
        "window_visible\twindow_pixels\twindow_index\tframe_indices"
    )
    for row in rows:
        print(
            f"{row['instance_id']}\t{row['label']}\t{row['frames']}\t"
            f"{row['pixels']}\t{row['max_pixels']}\t"
            f"{row['window_visible_frames']}\t{row['window_pixels']}\t"
            f"{row['window_index']}\t{row['frame_indices']}"
        )

    best = rows[0]
    print("\nSuggested overfit args:")
    print(
        f"  --instance-id {best['instance_id']} "
        f"--window-index {best['window_index']} "
        f"--sequence-length {args.sequence_length} "
        f"--frame-stride {args.frame_stride}"
    )


def print_instance_timeline(
    per_frame_counts: list[dict[int, int]],
    object_labels: dict[int, str],
    *,
    instance_id: int,
    frame_indices: list[int] | None,
) -> None:
    if frame_indices is None:
        frame_indices = list(range(len(per_frame_counts)))
    label = object_labels.get(instance_id, "")
    print(f"instance_id={instance_id} label='{label}'")
    print("frame_idx\tpixels\tpresent")
    for frame_idx in frame_indices:
        if frame_idx < 0 or frame_idx >= len(per_frame_counts):
            print(f"{frame_idx}\tOUT_OF_RANGE\tFalse")
            continue
        pixels = int(per_frame_counts[frame_idx].get(instance_id, 0))
        print(f"{frame_idx}\t{pixels}\t{pixels > 0}")


def best_window_for_instance(
    per_frame_counts: list[dict[int, int]],
    *,
    instance_id: int,
    sequence_length: int,
    frame_stride: int,
    min_visible_frames: int,
) -> dict | None:
    sequence_length = max(1, int(sequence_length))
    frame_stride = max(1, int(frame_stride))
    window_size = (sequence_length - 1) * frame_stride + 1
    if len(per_frame_counts) < window_size:
        return None

    best = None
    for start in range(0, len(per_frame_counts) - window_size + 1):
        frame_indices = [start + i * frame_stride for i in range(sequence_length)]
        counts = [
            int(per_frame_counts[frame_idx].get(instance_id, 0))
            for frame_idx in frame_indices
        ]
        visible = sum(count > 0 for count in counts)
        if visible < min_visible_frames:
            continue
        pixels = sum(counts)
        candidate = {
            "window_index": start,
            "frame_indices": frame_indices,
            "window_pixels": pixels,
            "window_visible_frames": visible,
        }
        if best is None:
            best = candidate
            continue
        if (visible, pixels) > (best["window_visible_frames"], best["window_pixels"]):
            best = candidate
    return best


if __name__ == "__main__":
    main()
