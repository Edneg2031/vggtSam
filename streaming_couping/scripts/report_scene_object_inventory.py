#!/usr/bin/env python3
"""List objects visible in a configured clip for manual prompt planning only."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

import numpy as np

from streaming_couping.src.data import (
    extract_object_labels,
    read_mask,
    resolve_manifest_path,
)


REVISION = "scene_annotation_object_inventory_r1"
DEFAULT_EXCLUDED_LABELS = (
    "wall",
    "floor",
    "ceiling",
    "room",
    "object",
    "unknown",
    "bed",
)


def main() -> None:
    args = _parse_args()
    config, clip = _load_clip(args.config, args.clip)
    manifest_path = config.manifest
    scene = _load_scene(manifest_path, clip.scene_id)
    source_frames = scene.get("frames", ())
    masks = []
    for frame_index in clip.frame_indices:
        if frame_index < 0 or frame_index >= len(source_frames):
            raise ValueError(
                f"Frame {frame_index} is outside manifest scene range "
                f"[0,{len(source_frames) - 1}]."
            )
        frame = source_frames[frame_index]
        if "instance_mask" not in frame:
            raise ValueError(
                f"Manifest frame {frame_index} has no instance_mask field."
            )
        masks.append(
            read_mask(
                resolve_manifest_path(frame["instance_mask"], manifest_path)
            )
        )
    labels = extract_object_labels(scene.get("objects", {}))
    label_rows, instance_rows = summarize_visible_objects(
        masks=masks,
        frame_indices=clip.frame_indices,
        labels=labels,
        configured_prompts=clip.instance_prompts,
        excluded_labels=args.exclude_label,
        min_visible_frames=args.min_visible_frames,
        min_pixels=args.min_pixels,
        max_area_ratio=args.max_area_ratio,
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    result = _write_outputs(
        output_dir=output_dir,
        config_path=Path(args.config).expanduser().resolve(),
        manifest_path=manifest_path,
        clip_name=clip.name,
        scene_id=clip.scene_id,
        frame_indices=clip.frame_indices,
        configured_prompts=clip.instance_prompts,
        excluded_labels=tuple(args.exclude_label),
        min_visible_frames=args.min_visible_frames,
        min_pixels=args.min_pixels,
        max_area_ratio=args.max_area_ratio,
        label_rows=label_rows,
        instance_rows=instance_rows,
    )
    print(f"scene object inventory result={result}")


def summarize_visible_objects(
    *,
    masks: Sequence[np.ndarray],
    frame_indices: Sequence[int],
    labels: dict[int, str],
    configured_prompts: Sequence[str],
    excluded_labels: Iterable[str] = DEFAULT_EXCLUDED_LABELS,
    min_visible_frames: int = 4,
    min_pixels: int = 128,
    max_area_ratio: float = 0.25,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Summarize annotation instances without exposing them to a model path."""

    if not masks or len(masks) != len(frame_indices):
        raise ValueError("Masks and frame_indices must be non-empty and aligned.")
    if min_visible_frames < 1 or min_pixels < 1:
        raise ValueError("Visibility and pixel thresholds must be positive.")
    if not 0.0 < max_area_ratio <= 1.0:
        raise ValueError("max_area_ratio must be in (0,1].")
    excluded = {_normalize_label(value) for value in excluded_labels}
    configured = {_normalize_label(value) for value in configured_prompts}
    instance_stats: dict[int, dict[str, Any]] = {}
    for frame_index, mask in zip(frame_indices, masks):
        value = np.asarray(mask)
        if value.ndim != 2:
            raise ValueError(
                f"Instance mask at frame {frame_index} is not two-dimensional."
            )
        image_pixels = int(value.size)
        object_ids, counts = np.unique(value, return_counts=True)
        for raw_id, raw_count in zip(object_ids, counts):
            instance_id = int(raw_id)
            count = int(raw_count)
            if instance_id == 0 or count <= 0:
                continue
            row = instance_stats.setdefault(
                instance_id,
                {
                    "frame_indices": [],
                    "pixels": [],
                    "area_ratios": [],
                },
            )
            row["frame_indices"].append(int(frame_index))
            row["pixels"].append(count)
            row["area_ratios"].append(float(count) / image_pixels)

    instance_rows = []
    for instance_id, stats in instance_stats.items():
        label = str(labels.get(instance_id, f"unlabeled_{instance_id}"))
        normalized = _normalize_label(label)
        pixels = np.asarray(stats["pixels"], dtype=np.int64)
        ratios = np.asarray(stats["area_ratios"], dtype=np.float64)
        eligible = bool(
            len(stats["frame_indices"]) >= int(min_visible_frames)
            and int(pixels.max()) >= int(min_pixels)
            and float(ratios.max()) <= float(max_area_ratio)
            and normalized not in excluded
            and not normalized.startswith("unlabeled ")
        )
        instance_rows.append(
            {
                "instance_id": int(instance_id),
                "label": label,
                "normalized_label": normalized,
                "visible_frames": int(len(stats["frame_indices"])),
                "first_frame": int(min(stats["frame_indices"])),
                "last_frame": int(max(stats["frame_indices"])),
                "total_pixels": int(pixels.sum()),
                "median_visible_pixels": int(np.median(pixels)),
                "maximum_pixels": int(pixels.max()),
                "maximum_area_percent": 100.0 * float(ratios.max()),
                "configured_prompt_exact_match": int(normalized in configured),
                "local_geometry_prompt_candidate": int(eligible),
                "frame_indices": tuple(int(v) for v in stats["frame_indices"]),
            }
        )
    instance_rows.sort(
        key=lambda row: (
            -int(row["local_geometry_prompt_candidate"]),
            -int(row["visible_frames"]),
            -int(row["total_pixels"]),
            str(row["normalized_label"]),
            int(row["instance_id"]),
        )
    )

    grouped: dict[str, list[dict[str, object]]] = {}
    for row in instance_rows:
        grouped.setdefault(str(row["normalized_label"]), []).append(row)
    label_rows = []
    for normalized, rows in grouped.items():
        visible_union = {
            frame
            for row in rows
            for frame in row["frame_indices"]
        }
        eligible_count = sum(
            int(row["local_geometry_prompt_candidate"]) for row in rows
        )
        label_rows.append(
            {
                "prompt_label": normalized,
                "visible_instance_count": len(rows),
                "eligible_instance_count": int(eligible_count),
                "union_visible_frames": len(visible_union),
                "summed_instance_visible_frames": sum(
                    int(row["visible_frames"]) for row in rows
                ),
                "total_pixels": sum(int(row["total_pixels"]) for row in rows),
                "maximum_instance_pixels": max(
                    int(row["maximum_pixels"]) for row in rows
                ),
                "maximum_instance_area_percent": max(
                    float(row["maximum_area_percent"]) for row in rows
                ),
                "configured_prompt_exact_match": int(normalized in configured),
                "local_geometry_prompt_candidate": int(eligible_count > 0),
                "instance_ids": " ".join(
                    str(row["instance_id"]) for row in rows
                ),
            }
        )
    label_rows.sort(
        key=lambda row: (
            -int(row["local_geometry_prompt_candidate"]),
            -int(row["eligible_instance_count"]),
            -int(row["union_visible_frames"]),
            -int(row["total_pixels"]),
            str(row["prompt_label"]),
        )
    )
    return label_rows, instance_rows


def _write_outputs(
    *,
    output_dir: Path,
    config_path: Path,
    manifest_path: Path,
    clip_name: str,
    scene_id: str,
    frame_indices: Sequence[int],
    configured_prompts: Sequence[str],
    excluded_labels: Sequence[str],
    min_visible_frames: int,
    min_pixels: int,
    max_area_ratio: float,
    label_rows: list[dict[str, object]],
    instance_rows: list[dict[str, object]],
) -> Path:
    if not label_rows or not instance_rows:
        raise RuntimeError("No annotated instance is visible in the clip.")
    output_dir.mkdir(parents=True, exist_ok=True)
    label_csv = output_dir / "visible_labels.csv"
    instance_csv = output_dir / "visible_instances.csv"
    _write_csv(label_csv, label_rows)
    _write_csv(instance_csv, instance_rows)
    candidates = tuple(
        str(row["prompt_label"])
        for row in label_rows
        if int(row["local_geometry_prompt_candidate"])
    )
    output = {
        "schema": 1,
        "revision": REVISION,
        "role": "manual_prompt_planning_only",
        "source": "dataset_instance_annotations_and_masks",
        "annotation_gt_used": 1,
        "consumed_by_v0_or_v1_candidate_generation": 0,
        "pose_or_geometry_evidence": 0,
        "config": str(config_path),
        "manifest": str(manifest_path),
        "clip": clip_name,
        "scene_id": scene_id,
        "frame_indices": tuple(int(value) for value in frame_indices),
        "configured_prompts": tuple(str(value) for value in configured_prompts),
        "excluded_labels": tuple(str(value) for value in excluded_labels),
        "candidate_filter": {
            "min_visible_frames": int(min_visible_frames),
            "min_pixels": int(min_pixels),
            "max_area_ratio": float(max_area_ratio),
        },
        "visible_label_count": len(label_rows),
        "visible_instance_count": len(instance_rows),
        "candidate_prompt_count": len(candidates),
        "candidate_prompts": candidates,
        "labels": label_rows,
        "instances": instance_rows,
        "outputs": {
            "label_csv": str(label_csv),
            "instance_csv": str(instance_csv),
        },
    }
    summary_path = output_dir / "scene_object_inventory.json"
    summary_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf8",
    )
    report_path = output_dir / "copyable_result.txt"
    _write_copyable(report_path, output)
    print("SCENE OBJECT INVENTORY (ANNOTATION-BASED, PLANNING ONLY)")
    print(
        f"  labels={len(label_rows)} instances={len(instance_rows)} "
        f"candidates={len(candidates)}"
    )
    print(f"  copyable_report={report_path}")
    return summary_path


def _write_copyable(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "===== COPYABLE_SCENE_OBJECT_INVENTORY_BEGIN =====",
        f"revision={summary['revision']}",
        f"clip={summary['clip']}",
        f"frames={len(summary['frame_indices'])}",
        "source=dataset_instance_annotations_and_masks",
        "role=manual_prompt_planning_only",
        "annotation_gt_used=1",
        "consumed_by_v0_or_v1_candidate_generation=0",
        f"visible_label_count={summary['visible_label_count']}",
        f"visible_instance_count={summary['visible_instance_count']}",
        f"candidate_prompt_count={summary['candidate_prompt_count']}",
        "candidate_prompts=" + " | ".join(summary["candidate_prompts"]),
        "",
        "prompt_label,visible_instance_count,eligible_instance_count,union_visible_frames,total_pixels,maximum_instance_area_percent,configured_prompt_exact_match,local_geometry_prompt_candidate,instance_ids",
    ]
    for row in summary["labels"]:
        lines.append(
            ",".join(
                str(row[name])
                for name in (
                    "prompt_label",
                    "visible_instance_count",
                    "eligible_instance_count",
                    "union_visible_frames",
                    "total_pixels",
                    "maximum_instance_area_percent",
                    "configured_prompt_exact_match",
                    "local_geometry_prompt_candidate",
                    "instance_ids",
                )
            )
        )
    lines.extend(
        [
            "",
            f"label_csv={summary['outputs']['label_csv']}",
            f"instance_csv={summary['outputs']['instance_csv']}",
            f"summary={path.with_name('scene_object_inventory.json')}",
            "===== COPYABLE_SCENE_OBJECT_INVENTORY_END =====",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _normalize_label(value: object) -> str:
    text = re.sub(r"[_-]+", " ", str(value).strip().lower())
    return " ".join(text.split())


def _load_clip(config_path: str | Path, requested_clip: str | None):
    # Delayed import keeps the pure inventory summarizer independent of YAML.
    from streaming_couping.src.learned_pose.config import (
        load_learned_pose_config,
    )

    config = load_learned_pose_config(config_path)
    if requested_clip is None:
        if len(config.clips) != 1:
            raise ValueError("Use --clip when the config contains several clips.")
        return config, config.clips[0]
    selected = [clip for clip in config.clips if clip.name == requested_clip]
    if len(selected) != 1:
        raise ValueError(f"Clip {requested_clip!r} was not found exactly once.")
    return config, selected[0]


def _load_scene(manifest_path: Path, scene_id: str) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    selected = [
        scene
        for scene in manifest.get("scenes", ())
        if str(scene.get("scene_id")) == str(scene_id)
    ]
    if len(selected) != 1:
        raise ValueError(f"Scene {scene_id!r} was not found exactly once.")
    return selected[0]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="streaming_couping/configs/v0_baseline.yaml",
    )
    parser.add_argument("--clip")
    parser.add_argument(
        "--output-dir",
        default="outputs/streaming_couping_v0/scene_object_inventory",
    )
    parser.add_argument("--min-visible-frames", type=int, default=4)
    parser.add_argument("--min-pixels", type=int, default=128)
    parser.add_argument("--max-area-ratio", type=float, default=0.25)
    parser.add_argument(
        "--exclude-label",
        action="append",
        default=list(DEFAULT_EXCLUDED_LABELS),
        help="Repeat to exclude a structural or scene-dominant label.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
