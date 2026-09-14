#!/usr/bin/env python3
"""What objects does this scene have, and which ones is the run missing?

The prompts are a hard ceiling on everything downstream: an object that no
prompt matches cannot be proposed, corrected, or scored, no matter how well it
is seen.  The 100-frame run showed both directions of that mismatch -- the
ground truth has a "pet bed" that no prompt reached, and the segmenter produced
a "rug" the manifest has no object for.

This prints the inventory, read-only and CPU only:

* every object the manifest defines for the scene, and which of them are
  visible in the selected frames (with the mask size the pipeline would see);
* which prompt matches each visible object, and which visible objects no prompt
  reaches at all;
* what the segmenter actually produced, when a run's diagnostics are given, so
  the segmenter side can be compared against the ground-truth side.

    python -m streaming_couping.scripts.list_scene_objects \
        --manifest <manifest.json> --scene-id 00a231a370 \
        --prompts bed wardrobe chair rug dustbin
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from streaming_couping.src.data import extract_object_labels
from streaming_couping.src.semantic_tracking_metrics import (
    load_ground_truth_instances,
    prompt_matches_label,
)

#: The run's own mask filters, so a listed object says whether it would survive.
PIPELINE_MIN_MASK_PIXELS = 32
PIPELINE_MAX_MASK_AREA_RATIO = 0.85


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "yes" if value else "NO"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return "nan"
    return f"{float(value):.{digits}f}"


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    rendered = [[_fmt(cell) for cell in row] for row in rows]
    widths = [
        max(len(str(header)), *(len(row[index]) for row in rendered))
        for index, header in enumerate(headers)
    ]
    lines = [
        "  ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)),
        "  ".join("-" * width for width in widths),
    ]
    for row in rendered:
        lines.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
    return "\n".join(lines)


def scene_objects(manifest_path: Path, scene_id: str) -> dict[int, str]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    scene = next(
        (
            row
            for row in manifest.get("scenes", ())
            if str(row.get("scene_id")) == str(scene_id)
        ),
        None,
    )
    if scene is None:
        raise ValueError(f"scene {scene_id!r} is absent from {manifest_path}")
    return extract_object_labels(scene.get("objects", {}) or {})


def frame_count(manifest_path: Path, scene_id: str) -> int:
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    for row in manifest.get("scenes", ()):
        if str(row.get("scene_id")) == str(scene_id):
            return len(row.get("frames", ()))
    return 0


def selection(args: argparse.Namespace) -> tuple[list[int], tuple[int, int]]:
    """Frame indices to inspect, and the grid their masks are placed on."""

    if args.geometry_cache is not None:
        payload = torch.load(
            args.geometry_cache.expanduser().resolve(),
            map_location="cpu",
            weights_only=False,
        )
        positions = [int(value) for value in payload["source_positions"]]
        depth = torch.as_tensor(payload["depth"]).detach().cpu()
        if depth.ndim == 4 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        return positions, (int(depth.shape[1]), int(depth.shape[2]))
    manifest_path = args.manifest.expanduser().resolve()
    total = frame_count(manifest_path, args.scene_id)
    start = int(args.frame_start)
    stride = int(args.frame_stride)
    # Mirror the pipeline's own selection: the count is how many frames to take
    # and the stride steps past them, so starting at 2 with stride 3 and count
    # 4 selects 2, 5, 8, 11 -- not 2, 5.
    count = int(args.frame_count)
    if count <= 0:
        count = max(0, (total - start + stride - 1) // stride)
    positions = [
        start + offset * stride
        for offset in range(count)
        if start + offset * stride < total
    ]
    return positions, (int(args.image_size), int(args.image_size))


def sam_inventory(diagnostics_path: Path) -> dict[str, dict[str, Any]]:
    payload = torch.load(
        diagnostics_path.expanduser().resolve(), map_location="cpu", weights_only=False
    )
    inventory: dict[str, dict[str, Any]] = {}
    for row in payload.get("observations", ()):
        label = str(row["category"])
        entry = inventory.setdefault(
            label, {"observations": 0, "instance_ids": set(), "points": 0}
        )
        entry["observations"] += 1
        entry["instance_ids"].add(int(row["instance_id"]))
        entry["points"] += int(torch.as_tensor(row["points_camera"]).shape[0])
    return {
        label: {
            "observations": entry["observations"],
            "instances": sorted(entry["instance_ids"]),
            "points": entry["points"],
        }
        for label, entry in inventory.items()
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument(
        "--geometry-cache",
        type=Path,
        default=None,
        help=(
            "Use the exact frames and image grid of a run. Without it the "
            "selection has to be given explicitly."
        ),
    )
    parser.add_argument("--feedback-diagnostics", type=Path, default=None)
    parser.add_argument("--prompts", nargs="*", default=None)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--frame-count", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=518)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    positions, (height, width) = selection(args)
    if not positions:
        raise RuntimeError("the frame selection is empty")
    print(
        f"scene={args.scene_id} manifest={manifest_path}\n"
        f"frames={len(positions)} (positions {positions[0]}..{positions[-1]}) "
        f"grid=({height},{width})"
    )

    defined = scene_objects(manifest_path, args.scene_id)
    print(f"\nmanifest defines {len(defined)} objects for this scene")

    ground_truth = load_ground_truth_instances(
        manifest_path,
        scene_id=str(args.scene_id),
        frame_indices=positions,
        output_size=(height, width),
        prompts=tuple(args.prompts or ()),
        include_all_instances=True,
    )
    prompts = tuple(args.prompts or ())
    total_pixels = float(height * width)

    rows: list[list[Any]] = []
    matched_by: dict[int, str] = {}
    for index, instance_id in enumerate(ground_truth.instance_ids):
        label = str(ground_truth.labels[index])
        mask = ground_truth.masks[:, index]
        per_frame = mask.flatten(1).sum(dim=1)
        visible = per_frame > 0
        visible_frames = int(visible.sum())
        pixels_total = int(mask.sum())
        pixels_median = (
            float(per_frame[visible].float().median()) if visible_frames else 0.0
        )
        area_ratio = pixels_median / total_pixels
        matches = [prompt for prompt in prompts if prompt_matches_label(prompt, label)]
        if matches:
            matched_by[int(instance_id)] = ",".join(matches)
        # The pipeline filters per observation, not on a summary statistic, so
        # count the frames whose mask would actually be kept.
        kept = int(
            (
                (per_frame >= PIPELINE_MIN_MASK_PIXELS)
                & ((per_frame.float() / total_pixels) <= PIPELINE_MAX_MASK_AREA_RATIO)
            ).sum()
        )
        rows.append(
            [
                int(instance_id),
                label,
                visible_frames,
                pixels_median,
                kept,
                ",".join(matches) if matches else "NONE",
            ]
        )
    rows.sort(key=lambda row: (-int(row[4]), -int(row[2]), str(row[1])))
    print(
        f"\nvisible instances: {len(rows)} "
        f"(of {len(defined)} defined in the scene)\n"
        f"  px = median mask pixels over the frames where the object is visible, on\n"
        f"  the run's own grid; 'kept' counts the frames whose mask passes the\n"
        f"  pipeline's own >= {PIPELINE_MIN_MASK_PIXELS} px and "
        f"<= {PIPELINE_MAX_MASK_AREA_RATIO} area filters, so kept=0 means the\n"
        f"  object contributes no observation at all"
    )
    print(
        _table(
            ["id", "label", "frames", "px", "kept", "prompt"],
            rows,
        )
    )

    never = sorted(set(defined) - {int(value) for value in ground_truth.instance_ids})
    if never:
        print(
            f"\n{len(never)} object(s) defined in the scene are never visible in "
            "these frames:"
        )
        for instance_id in never:
            print(f"  id={instance_id} label={defined[instance_id]!r}")

    # The table above already carries the prompt column and the kept count, so
    # the rows are not repeated here -- only what the table cannot show: the
    # labels behind a NONE, and which objects are unreachable *and* well seen,
    # since a small object nobody can use is not the same finding as a large one.
    unmatched = [row for row in rows if row[5] == "NONE"]
    if unmatched:
        labels = sorted({str(row[1]) for row in unmatched})
        print(
            f"\n{len(unmatched)} visible object(s) no prompt reaches, across "
            f"{len(labels)} distinct labels:"
        )
        print(f"  {labels}")
    else:
        print("\nevery visible ground-truth object is reached by a prompt")

    filtered = [row for row in rows if int(row[4]) == 0]
    if filtered:
        print(
            f"\n{len(filtered)} visible object(s) the pipeline's own mask filters "
            "would drop in every frame (px_median below the 32 px floor):"
        )
        print(
            "  "
            + ", ".join(f"{row[1]}#{row[0]}({row[3]:.0f}px)" for row in filtered)
        )

    if args.feedback_diagnostics is not None:
        sam = sam_inventory(args.feedback_diagnostics)
        print(
            f"\nsegmenter produced {len(sam)} categories from "
            f"{args.feedback_diagnostics.expanduser().resolve()}"
        )
        print(
            _table(
                ["category", "instances", "observations", "points", "in_gt"],
                [
                    [
                        label,
                        len(entry["instances"]),
                        entry["observations"],
                        entry["points"],
                        any(
                            prompt_matches_label(label, str(row[1]))
                            for row in rows
                        ),
                    ]
                    for label, entry in sorted(sam.items())
                ],
            )
        )
        produced = {label for label in sam}
        gt_labels = {str(row[1]) for row in rows}
        orphan = sorted(
            label
            for label in produced
            if not any(
                prompt_matches_label(label, gt_label) for gt_label in gt_labels
            )
        )
        if orphan:
            print(
                "\nthe segmenter produced categories with no ground-truth object "
                f"to score against: {orphan}"
            )
    print(
        "\nAdding a prompt is free at this stage: it changes which masks are "
        "requested,\nso it needs a stage-2a re-run, but no model changes and no "
        "new labels."
    )


if __name__ == "__main__":
    main()
