#!/usr/bin/env python3
"""Plan a same-scene unseen-time T0.2 holdout from manifest metadata only."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence


REVISION = "t02_manifest_only_temporal_holdout_planner_r3"
PATH_FIELDS = ("image_path", "instance_mask", "pointmap")
MATRIX_FIELDS = ("world_to_camera", "intrinsics")
REQUIRED_FIELDS = PATH_FIELDS + MATRIX_FIELDS


def main() -> None:
    args = _parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    candidates, rejected = plan_confirmation_sequences(
        manifest,
        manifest_path=manifest_path,
        discovery_scene_id=args.discovery_scene_id,
        discovery_last_frame=args.discovery_last_frame,
        frame_count=args.frame_count,
        frame_stride=args.frame_stride,
    )
    maximum = int(args.maximum_candidates)
    if maximum < 1:
        raise ValueError("maximum_candidates must be positive.")
    eligible = [row for row in candidates if int(row["eligible"])]
    selected = eligible[:maximum]
    for rank, row in enumerate(selected, start=1):
        row["selection_rank"] = rank
        row["recommended"] = int(rank == 1)
    output_dir = Path(args.output_dir).expanduser().resolve()
    summary_path = _write_outputs(
        output_dir,
        manifest_path=manifest_path,
        discovery_scene_id=args.discovery_scene_id,
        discovery_last_frame=args.discovery_last_frame,
        frame_count=args.frame_count,
        frame_stride=args.frame_stride,
        eligible_holdout_count=len(eligible),
        rejected_holdout_count=rejected,
        candidates=selected,
        diagnostics=candidates,
    )
    print("T0.2 MANIFEST-ONLY SAME-SCENE TEMPORAL HOLDOUT PLANNER")
    print(
        f"  eligible_holdouts={len(eligible)} "
        f"reported={len(selected)} rejected={rejected}"
    )
    if selected:
        primary = selected[0]
        print(
            f"  recommended={primary['scene_id']} "
            f"clip={primary['clip_name']}"
        )
    else:
        print("  recommended=none; inspect rejection diagnostics")
    print(f"  result={summary_path}")


def plan_confirmation_sequences(
    manifest: dict[str, Any],
    *,
    manifest_path: Path,
    discovery_scene_id: str,
    discovery_last_frame: int,
    frame_count: int,
    frame_stride: int,
) -> tuple[list[dict[str, Any]], int]:
    """Return one centered, metadata-complete post-discovery holdout clip."""

    count = int(frame_count)
    maximum_stride = int(frame_stride)
    if count < 2 or maximum_stride < 1:
        raise ValueError("T0.2 frame count/stride are invalid.")
    scene_id = str(discovery_scene_id).strip()
    scene = next(
        (
            value
            for value in manifest.get("scenes", ())
            if str(value.get("scene_id", "")).strip() == scene_id
        ),
        None,
    )
    if scene is None:
        return [_rejected_row(scene_id, count, "discovery_scene_missing")], 1
    frames = scene.get("frames", ())
    first_allowed = int(discovery_last_frame) + 1
    span = (count - 1) * maximum_stride
    maximum_start = len(frames) - 1 - span
    if maximum_start < first_allowed:
        reason = (
            "insufficient_post_discovery_frames:"
            f"need_last_at_most_{len(frames) - 1}"
        )
        return [
            _rejected_row(
                scene_id,
                count,
                reason,
                scene_frame_count=len(frames),
            )
        ], 1
    start = first_allowed + (maximum_start - first_allowed) // 2
    indices = tuple(start + offset * maximum_stride for offset in range(count))
    selected = [frames[index] for index in indices]
    field_counts = {
        field: sum(int(frame.get(field) is not None) for frame in selected)
        for field in REQUIRED_FIELDS
    }
    fields_complete = all(value == count for value in field_counts.values())
    existing_counts = {
        field: sum(
            int(_resolve_manifest_path(frame[field], manifest_path).is_file())
            for frame in selected
            if frame.get(field)
        )
        for field in PATH_FIELDS
    }
    files_complete = all(value == count for value in existing_counts.values())
    eligible = int(fields_complete and files_complete)
    missing_fields = [
        field for field, value in field_counts.items() if value != count
    ]
    missing_files = [
        field for field, value in existing_counts.items() if value != count
    ]
    rejection_reason = ""
    if missing_fields:
        rejection_reason = "missing_fields:" + " ".join(missing_fields)
    elif missing_files:
        rejection_reason = "missing_files:" + " ".join(missing_files)
    row = {
        "scene_id": scene_id,
        "scene_frame_count": len(frames),
        "clip_name": (
            f"{scene_id}_{indices[0]}_{indices[-1]}_step"
            f"{maximum_stride}_t02_holdout"
        ),
        "frame_count": count,
        "frame_stride": maximum_stride,
        "first_frame": indices[0],
        "last_frame": indices[-1],
        "frame_indices": " ".join(str(value) for value in indices),
        "same_scene_as_discovery": 1,
        "frame_overlap_with_discovery": 0,
        "all_frames_after_discovery": 1,
        **{
            f"{field}_field_count": field_counts[field]
            for field in REQUIRED_FIELDS
        },
        **{
            f"{field}_existing_file_count": existing_counts[field]
            for field in PATH_FIELDS
        },
        "required_manifest_fields_complete": int(fields_complete),
        "required_files_exist": int(files_complete),
        "eligible": eligible,
        "rejection_reason": rejection_reason,
    }
    return [row], int(not eligible)


def _rejected_row(
    scene_id: str,
    frame_count: int,
    reason: str,
    *,
    scene_frame_count: int = 0,
) -> dict[str, Any]:
    return {
        "scene_id": scene_id,
        "scene_frame_count": scene_frame_count,
        "clip_name": "",
        "frame_count": frame_count,
        "frame_stride": 0,
        "first_frame": "",
        "last_frame": "",
        "frame_indices": "",
        "same_scene_as_discovery": 1,
        "frame_overlap_with_discovery": 0,
        "all_frames_after_discovery": 0,
        "required_manifest_fields_complete": 0,
        "required_files_exist": 0,
        "eligible": 0,
        "rejection_reason": reason,
    }


def _write_outputs(
    output_dir: Path,
    *,
    manifest_path: Path,
    discovery_scene_id: str,
    discovery_last_frame: int,
    frame_count: int,
    frame_stride: int,
    eligible_holdout_count: int,
    rejected_holdout_count: int,
    candidates: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "candidate_sequences.csv"
    _write_csv(csv_path, candidates)
    diagnostics_path = output_dir / "scene_diagnostics.csv"
    _write_csv(diagnostics_path, diagnostics)
    primary = candidates[0] if candidates else None
    summary = {
        "schema": 1,
        "revision": REVISION,
        "role": "t02_same_scene_temporal_holdout_planning_only",
        "source": "manifest_metadata_and_file_existence_only",
        "manifest": str(manifest_path),
        "discovery_scene_id": str(discovery_scene_id),
        "discovery_last_frame": int(discovery_last_frame),
        "validation_scope": "same_scene_unseen_temporal_holdout",
        "same_scene_required": 1,
        "frame_count": int(frame_count),
        "frame_stride": int(frame_stride),
        "selection_rule": "post_discovery_centered_fixed_stride",
        "required_frame_fields": REQUIRED_FIELDS,
        "gt_geometry_values_read": 0,
        "pointmap_file_content_read": 0,
        "instance_mask_file_content_read": 0,
        "holdout_selection_gt_fields": 0,
        "model_loaded_or_run": 0,
        "eligible_holdout_count": int(eligible_holdout_count),
        "reported_candidate_count": len(candidates),
        "rejected_holdout_count": int(rejected_holdout_count),
        "planner_ready": int(primary is not None),
        "recommended_scene_id": (
            str(primary["scene_id"]) if primary is not None else None
        ),
        "recommended_clip_name": (
            str(primary["clip_name"]) if primary is not None else None
        ),
        "recommended_frame_indices": (
            tuple(int(value) for value in str(primary["frame_indices"]).split())
            if primary is not None
            else ()
        ),
        "candidates": candidates,
        "diagnostics": diagnostics,
        "outputs": {
            "candidate_csv": str(csv_path),
            "diagnostics_csv": str(diagnostics_path),
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf8"
    )
    report_path = output_dir / "copyable_result.txt"
    _write_copyable(report_path, summary)
    return summary_path


def _write_copyable(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "===== COPYABLE_T02_SEQUENCE_PLANNER_BEGIN =====",
        f"revision={summary['revision']}",
        f"validation_scope={summary['validation_scope']}",
        f"discovery_scene_id={summary['discovery_scene_id']}",
        f"discovery_last_frame={summary['discovery_last_frame']}",
        f"frame_count={summary['frame_count']}",
        f"frame_stride={summary['frame_stride']}",
        f"selection_rule={summary['selection_rule']}",
        "source=manifest_metadata_and_file_existence_only",
        "gt_geometry_values_read=0",
        "pointmap_file_content_read=0",
        "instance_mask_file_content_read=0",
        "holdout_selection_gt_fields=0",
        "model_loaded_or_run=0",
        f"eligible_holdout_count={summary['eligible_holdout_count']}",
        f"reported_candidate_count={summary['reported_candidate_count']}",
        f"planner_ready={summary['planner_ready']}",
        f"recommended_scene_id={summary['recommended_scene_id']}",
        f"recommended_clip_name={summary['recommended_clip_name']}",
        "recommended_frame_indices="
        + " ".join(str(value) for value in summary["recommended_frame_indices"]),
        "",
        "selection_rank,scene_id,scene_frame_count,clip_name,first_frame,last_frame,frame_stride,frame_indices",
    ]
    fields = (
        "selection_rank",
        "scene_id",
        "scene_frame_count",
        "clip_name",
        "first_frame",
        "last_frame",
        "frame_stride",
        "frame_indices",
    )
    for row in summary["candidates"]:
        lines.append(",".join(str(row[field]) for field in fields))
    lines.extend(
        (
            "",
            "scene diagnostics:",
            "scene_id,scene_frame_count,frame_stride,eligible,rejection_reason",
        )
    )
    diagnostic_fields = (
        "scene_id",
        "scene_frame_count",
        "frame_stride",
        "eligible",
        "rejection_reason",
    )
    for row in summary["diagnostics"]:
        lines.append(",".join(str(row.get(field, "")) for field in diagnostic_fields))
    lines.extend(
        (
            "",
            f"candidate_csv={summary['outputs']['candidate_csv']}",
            f"diagnostics_csv={summary['outputs']['diagnostics_csv']}",
            f"summary={path.with_name('summary.json')}",
            "===== COPYABLE_T02_SEQUENCE_PLANNER_END =====",
        )
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def _resolve_manifest_path(value: str | Path, manifest_path: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidates = (Path.cwd() / path, manifest_path.parent / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf8")
        return
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="data/processed/scannetpp_pinhole_2d/manifest.json",
    )
    parser.add_argument("--discovery-scene-id", default="00a231a370")
    parser.add_argument("--discovery-last-frame", type=int, default=525)
    parser.add_argument("--frame-count", type=int, default=30)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--maximum-candidates", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        default="outputs/streaming_couping_t02_sequence_planner",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
