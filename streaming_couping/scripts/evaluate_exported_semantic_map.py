#!/usr/bin/env python3
"""Evaluate exported HorizonStream semantic-map branches offline.

The runtime command never reads annotations.  This command is intentionally a
separate process/stage: it opens ``semantic_map.pt`` artifacts, reads the
ScanNet++ pointmaps and instance masks, fits one reference-frame Sim(3), and
reports the same metrics for every branch under one shared alignment.

Typical use after ``commands_run_semantic_map.txt``::

    python -m streaming_couping.scripts.evaluate_exported_semantic_map \
      --input-dir /data184/open_source/vggtSam/outputs/semantic_map_50frames_horizonstream \
      --output-dir /data184/open_source/vggtSam/outputs/semantic_map_50frames_horizonstream/evaluation

The manifest, scene, frame positions, and geometry cache are inferred from
the exported metadata when possible.  They may be supplied explicitly when an
artifact was moved to another machine.  When a HorizonStream geometry cache is
available, ScanNet++ ``frames[].world_to_camera`` is also used offline to
report raw/refined ATE and adjacent-frame RPE.  The optional refined trajectory
is discovered from ``object_pose_refinement/refined_camera_to_world.pt``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import torch

from streaming_couping.src.horizonstream_cache import load_horizonstream_cache
from streaming_couping.src.semantic_map_metrics import (
    SemanticMapMetricConfig,
    load_ground_truth_stream_masks,
)
from streaming_couping.src.semantic_mapping.adapters import (
    HorizonStreamGeometryCacheAdapter,
)
from streaming_couping.src.semantic_mapping.evaluation import (
    ExportedMapMetricConfig,
    evaluate_camera_trajectory,
    evaluate_exported_semantic_map,
    evaluate_pointmap_alignment,
    fit_reference_alignment,
)
from streaming_couping.src.semantic_mapping.geometry import world_points_for_frame
from streaming_couping.src.pointmap_alignment import load_ground_truth_pointmaps
from streaming_couping.src.semantic_tracking_metrics import (
    load_ground_truth_instances,
)


REVISION = "exported_horizonstream_semantic_map_evaluation_r2"


def main() -> None:
    args = _parse_args()
    artifacts = _discover_artifacts(args.input_dir or args.artifact)
    first_payload = _load_artifact(next(iter(artifacts.values())))
    metadata = _artifact_metadata(first_payload)

    manifest = _resolve_path(
        args.manifest,
        metadata.get("manifest"),
        description="ScanNet++ manifest",
    )
    scene_id = str(args.scene_id or metadata.get("scene_id") or "").strip()
    if not scene_id:
        raise ValueError(
            "Scene ID is unavailable; pass --scene-id or use an artifact made "
            "from --manifest."
        )

    geometry_path = _resolve_optional_path(
        args.geometry_cache,
        metadata.get("geometry_cache"),
    )
    geometry_payload = (
        load_horizonstream_cache(geometry_path)
        if geometry_path is not None
        else None
    )
    processed_size, image_mode, target_size, patch_size = _resolve_grid(
        args,
        metadata=metadata,
        geometry_payload=geometry_payload,
    )
    frame_positions = _resolve_frame_positions(
        args,
        metadata=metadata,
        geometry_payload=geometry_payload,
        manifest=manifest,
        scene_id=scene_id,
    )
    if not frame_positions:
        raise ValueError("No dataset frame positions were selected for evaluation.")

    prompts = _resolve_prompts(args.prompts, first_payload)
    aliases = _load_aliases(args.prompt_aliases_json)
    gt_info = load_ground_truth_instances(
        manifest,
        scene_id=scene_id,
        frame_indices=frame_positions,
        output_size=processed_size,
        prompts=prompts,
        prompt_label_aliases=aliases,
        include_all_instances=bool(args.all_gt_instances),
    )
    gt_masks = load_ground_truth_stream_masks(
        manifest,
        scene_id=scene_id,
        frame_indices=frame_positions,
        instance_ids=gt_info.instance_ids,
        processed_size=processed_size,
        image_mode=image_mode,
        target_size=target_size,
        patch_size=patch_size,
    )
    target_world_points = load_ground_truth_pointmaps(
        manifest,
        scene_id=scene_id,
        frame_indices=frame_positions,
        processed_size=processed_size,
        image_mode=image_mode,
        target_size=target_size,
        patch_size=patch_size,
    )
    target_world_to_camera = _load_ground_truth_world_to_camera(
        manifest,
        scene_id=scene_id,
        frame_indices=frame_positions,
    )

    predicted_world_points = None
    confidence = None
    pointmap_alignment: dict[str, object]
    alignment = None
    if geometry_payload is not None:
        adapter = HorizonStreamGeometryCacheAdapter(geometry_payload)
        cached_paths = tuple(adapter.image_paths)
        geometry_frames = adapter.infer(cached_paths)
        predicted_world_points = torch.stack(
            [world_points_for_frame(frame)[0] for frame in geometry_frames],
            dim=0,
        )
        confidence = adapter.confidence.detach().float().cpu()
        if predicted_world_points.shape[0] != len(frame_positions):
            raise ValueError(
                "Geometry cache frame count does not match the selected GT "
                f"frames: {predicted_world_points.shape[0]} vs {len(frame_positions)}."
            )
        if args.alignment == "reference":
            alignment = fit_reference_alignment(
                predicted_world_points,
                target_world_points,
                confidence,
                reference_frame_index=args.alignment_reference,
                confidence_threshold=args.confidence_threshold,
                max_points=args.max_alignment_points,
                min_points=args.min_alignment_points,
            )
            pointmap_alignment = evaluate_pointmap_alignment(
                predicted_world_points,
                target_world_points,
                confidence,
                alignment=alignment,
                confidence_threshold=args.confidence_threshold,
                frame_ids=frame_positions,
            )
            pointmap_alignment["alignment"] = alignment.to_dict()
        else:
            pointmap_alignment = {
                "status": "skipped",
                "reason": "alignment_none",
                "frames": [],
            }
    elif args.alignment == "reference":
        raise ValueError(
            "Reference alignment requires --geometry-cache or an artifact "
            "metadata.geometry_cache path."
        )
    else:
        pointmap_alignment = {
            "status": "skipped",
            "reason": "geometry_cache_not_provided",
            "frames": [],
        }

    pose_evaluation = _build_pose_evaluation(
        input_source=args.input_dir or args.artifact,
        geometry_payload=geometry_payload,
        target_world_to_camera=target_world_to_camera,
        frame_ids=frame_positions,
        alignment=alignment,
        refined_pose_path=args.refined_pose,
    )

    metric_config = ExportedMapMetricConfig(
        object_metrics=SemanticMapMetricConfig(
            confidence_threshold=args.confidence_threshold,
            track_score_threshold=0.0,
            max_points_per_object=args.max_points_per_object,
            distance_chunk_size=args.distance_chunk_size,
            fscore_thresholds_m=(0.05, 0.10),
            voxel_size_m=args.voxel_size_m,
            ghost_distance_m=args.ghost_distance_m,
            duplicate_voxel_iou=args.duplicate_voxel_iou,
        ),
        max_points_per_scene=args.max_points_per_scene,
        assignment_min_voxel_iou=args.assignment_min_voxel_iou,
    ).validate()

    output_dir = _resolve_output_dir(args, artifacts)
    output_dir.mkdir(parents=True, exist_ok=True)
    branch_summaries: dict[str, dict[str, object]] = {}
    map_summary_rows: list[dict[str, object]] = []
    map_object_rows: list[dict[str, object]] = []
    duplicate_rows: list[dict[str, object]] = []
    scene_rows: list[dict[str, object]] = []
    for branch, artifact_path in artifacts.items():
        payload = _load_artifact(artifact_path)
        result = evaluate_exported_semantic_map(
            payload,
            scene_id=scene_id,
            clip_name=str(branch),
            variant=str(branch),
            target_world_points=target_world_points,
            gt_masks=gt_masks,
            gt_instance_ids=gt_info.instance_ids,
            gt_labels=gt_info.labels,
            alignment=alignment,
            map_source=args.map_source,
            prompt_label_aliases=aliases,
            config=metric_config,
        )
        summary = dict(result["summary"])
        summary["artifact"] = str(artifact_path)
        summary["pointmap_alignment_status"] = pointmap_alignment["status"]
        summary["temporal_consensus"] = _artifact_metadata(payload).get(
            "temporal_consensus", {}
        )
        scene = dict(result["scene"])
        scene_row = {
            "scene_id": scene_id,
            "clip": str(branch),
            "variant": str(branch),
            "artifact": str(artifact_path),
            **scene,
        }
        branch_summaries[branch] = {
            "artifact": str(artifact_path),
            "summary": summary,
            "scene": scene,
            "assignments": result["assignments"],
            "object_count": len(result["object_rows"]),
            "duplicate_row_count": len(result["duplicate_rows"]),
        }
        pose_branch = pose_evaluation.get("branches", {}).get(branch)
        if isinstance(pose_branch, dict):
            branch_summaries[branch]["pose"] = pose_branch.get("summary", {})
        map_summary_rows.append(summary)
        map_object_rows.extend(result["object_rows"])
        duplicate_rows.extend(result["duplicate_rows"])
        scene_rows.append(scene_row)

    summary = {
        "schema": 1,
        "revision": REVISION,
        "input": str(args.input_dir or args.artifact),
        "manifest": str(manifest),
        "scene_id": scene_id,
        "frame_indices": [int(value) for value in frame_positions],
        "frame_count": len(frame_positions),
        "processed_size": list(processed_size),
        "image_mode": image_mode,
        "map_source": args.map_source,
        "candidate_generation_gt_fields": 0,
        "evaluation_gt_fields": 1,
        "pointmap_evaluation_gt_fields": 1,
        "instance_evaluation_gt_fields": 1,
        "pose_evaluation_gt_fields": 1,
        "geometry_cache": None if geometry_path is None else str(geometry_path),
        "alignment": (
            "reference_frame_sim3"
            if args.alignment == "reference"
            else "none"
        ),
        "alignment_shared_across_branches": True,
        "pointmap_alignment": pointmap_alignment,
        "ground_truth": {
            "pointmap_source": "manifest.frames[].pointmap",
            "instance_mask_source": "manifest.frames[].instance_mask",
            "object_metadata_source": "manifest.scenes[].objects",
            "pose_source": "manifest.frames[].world_to_camera",
            "eligible_instance_ids": [int(value) for value in gt_info.instance_ids],
            "eligible_labels": [str(value) for value in gt_info.labels],
            "all_visible_instance_ids": [
                int(value) for value in gt_info.all_visible_instance_ids
            ],
            "prompts": list(prompts),
        },
        "pose_evaluation": pose_evaluation,
        "branches": branch_summaries,
        "decision": "EVALUATION_ONLY; no runtime branch is promoted automatically",
        "outputs": {},
    }
    summary_path = output_dir / "summary.json"
    pose_summary_rows = _pose_summary_rows(pose_evaluation, scene_id)
    pose_frame_rows = _pose_frame_rows(pose_evaluation, scene_id)
    pose_rpe_rows = _pose_rpe_rows(pose_evaluation, scene_id)
    outputs = {
        "summary": summary_path,
        "map_summary": _write_csv(output_dir / "map_summary.csv", map_summary_rows),
        "map_objects": _write_csv(output_dir / "map_objects.csv", map_object_rows),
        "map_duplicates": _write_csv(
            output_dir / "map_duplicates.csv", duplicate_rows
        ),
        "scene_summary": _write_csv(output_dir / "scene_summary.csv", scene_rows),
        "pointmap_alignment_frames": _write_csv(
            output_dir / "pointmap_alignment_frames.csv",
            _pointmap_frame_rows(pointmap_alignment, scene_id),
        ),
        "pose_summary": _write_csv(
            output_dir / "pose_summary.csv",
            pose_summary_rows,
        ),
        "pose_frames": _write_csv(
            output_dir / "pose_frames.csv",
            pose_frame_rows,
        ),
        "pose_rpe": _write_csv(
            output_dir / "pose_rpe.csv",
            pose_rpe_rows,
        ),
    }
    summary["outputs"] = {name: str(path) for name, path in outputs.items()}
    summary_path.write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf8",
    )
    copyable = output_dir / "copyable_result.txt"
    _write_copyable(
        copyable,
        summary,
        map_summary_rows,
        pointmap_alignment,
        pose_evaluation,
    )
    _print_summary(summary, map_summary_rows, pointmap_alignment, pose_evaluation)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--input-dir",
        type=Path,
        help="Runtime output root or one branch directory containing semantic_map.pt.",
    )
    source.add_argument(
        "--artifact",
        type=Path,
        help="One exported semantic_map.pt artifact.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--scene-id", default=None)
    parser.add_argument("--geometry-cache", type=Path, default=None)
    parser.add_argument(
        "--refined-pose",
        type=Path,
        default=None,
        help=(
            "Optional refined_camera_to_world.pt. By default the evaluator "
            "looks under input-dir/object_pose_refinement/."
        ),
    )
    parser.add_argument(
        "--frame-indices",
        nargs="+",
        type=int,
        default=None,
        help="Explicit manifest positions; otherwise use exported source_positions.",
    )
    parser.add_argument("--prompts", nargs="+", default=None)
    parser.add_argument("--prompt-aliases-json", type=Path, default=None)
    parser.add_argument("--all-gt-instances", action="store_true")
    parser.add_argument(
        "--map-source",
        choices=("semantic", "tracks"),
        default="semantic",
        help="Evaluate policy-dependent semantic voxels or raw object tracks.",
    )
    parser.add_argument(
        "--alignment",
        choices=("reference", "none"),
        default="reference",
    )
    parser.add_argument("--alignment-reference", type=int, default=0)
    parser.add_argument("--confidence-threshold", type=float, default=0.30)
    parser.add_argument("--max-alignment-points", type=int, default=30_000)
    parser.add_argument("--min-alignment-points", type=int, default=128)
    parser.add_argument("--max-points-per-object", type=int, default=4_096)
    parser.add_argument("--max-points-per-scene", type=int, default=20_000)
    parser.add_argument("--distance-chunk-size", type=int, default=512)
    parser.add_argument("--voxel-size-m", type=float, default=0.05)
    parser.add_argument("--ghost-distance-m", type=float, default=0.10)
    parser.add_argument("--duplicate-voxel-iou", type=float, default=0.25)
    parser.add_argument("--assignment-min-voxel-iou", type=float, default=0.01)
    parser.add_argument(
        "--processed-size",
        type=_parse_size,
        default=None,
        help="Required only when no geometry cache metadata provides a grid.",
    )
    parser.add_argument(
        "--image-mode",
        choices=("auto", "crop", "pad"),
        default="auto",
    )
    parser.add_argument("--target-size", type=int, default=None)
    parser.add_argument("--patch-size", type=int, default=None)
    args = parser.parse_args()
    if args.confidence_threshold < 0.0 or args.confidence_threshold > 1.0:
        parser.error("--confidence-threshold must be in [0,1].")
    for name in (
        "max_alignment_points",
        "min_alignment_points",
        "max_points_per_object",
        "max_points_per_scene",
        "distance_chunk_size",
    ):
        if int(getattr(args, name)) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    return args


def _discover_artifacts(source: Path) -> dict[str, Path]:
    source = Path(source).expanduser().resolve()
    if source.is_file():
        if source.name != "semantic_map.pt":
            raise ValueError(f"Expected semantic_map.pt, got {source}.")
        return {source.parent.name: source}
    if not source.is_dir():
        raise FileNotFoundError(f"Semantic-map input does not exist: {source}")
    direct = source / "semantic_map.pt"
    if direct.is_file():
        return {source.name: direct}
    branches = {
        child.name: child / "semantic_map.pt"
        for child in sorted(source.iterdir(), key=lambda path: path.name)
        if child.is_dir() and (child / "semantic_map.pt").is_file()
    }
    if not branches:
        raise FileNotFoundError(
            f"No semantic_map.pt artifact found in {source} or its branch directories."
        )
    return branches


def _load_artifact(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a dictionary artifact, got {type(payload)!r}.")
    return payload


def _artifact_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    value = payload.get("metadata", {})
    return dict(value) if isinstance(value, dict) else {}


def _resolve_path(
    explicit: Path | None,
    inferred: Any,
    *,
    description: str,
) -> Path:
    path = explicit if explicit is not None else inferred
    if path is None or not str(path).strip():
        raise ValueError(f"{description} is unavailable; pass it explicitly.")
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} does not exist: {resolved}")
    return resolved


def _resolve_optional_path(explicit: Path | None, inferred: Any) -> Path | None:
    value = explicit if explicit is not None else inferred
    if value is None or not str(value).strip():
        return None
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Geometry cache does not exist: {path}")
    return path


def _resolve_grid(
    args: argparse.Namespace,
    *,
    metadata: dict[str, Any],
    geometry_payload: dict[str, Any] | None,
) -> tuple[tuple[int, int], str, int, int]:
    if geometry_payload is not None:
        processed_size = tuple(
            int(value) for value in geometry_payload["processed_size"]
        )
        preprocess = geometry_payload.get("preprocess", {})
        target_size = int(
            args.target_size
            or (preprocess.get("image_size", 518) if isinstance(preprocess, dict) else 518)
        )
        patch_size = int(
            args.patch_size
            or (preprocess.get("patch_size", 14) if isinstance(preprocess, dict) else 14)
        )
        inferred_mode = (
            "crop"
            if not isinstance(preprocess, dict) or bool(preprocess.get("crop", True))
            else "pad"
        )
    else:
        processed_size = args.processed_size
        if processed_size is None:
            candidate = metadata.get("processed_size")
            if candidate is not None:
                processed_size = tuple(int(value) for value in candidate)
        if processed_size is None:
            raise ValueError(
                "Processed grid is unavailable; pass --processed-size when "
                "--geometry-cache is not used."
            )
        target_size = int(args.target_size or 518)
        patch_size = int(args.patch_size or 14)
        inferred_mode = "crop"
    image_mode = inferred_mode if args.image_mode == "auto" else args.image_mode
    if len(processed_size) != 2 or any(int(value) <= 0 for value in processed_size):
        raise ValueError(f"Invalid processed size: {processed_size!r}")
    if target_size < 1 or patch_size < 1:
        raise ValueError("target_size and patch_size must be positive.")
    return (
        (int(processed_size[0]), int(processed_size[1])),
        str(image_mode),
        target_size,
        patch_size,
    )


def _resolve_frame_positions(
    args: argparse.Namespace,
    *,
    metadata: dict[str, Any],
    geometry_payload: dict[str, Any] | None,
    manifest: Path,
    scene_id: str,
) -> tuple[int, ...]:
    if args.frame_indices is not None:
        positions = tuple(int(value) for value in args.frame_indices)
    else:
        candidate = metadata.get("source_positions")
        if candidate is None and geometry_payload is not None:
            candidate = geometry_payload.get("source_positions")
        if candidate is None:
            raise ValueError(
                "Frame positions are unavailable; pass --frame-indices explicitly."
            )
        positions = tuple(int(value) for value in candidate)
    if len(set(positions)) != len(positions):
        raise ValueError("Frame positions must be unique.")
    if any(value < 0 for value in positions):
        raise ValueError("Frame positions must be non-negative.")
    with manifest.open("r", encoding="utf8") as handle:
        manifest_payload = json.load(handle)
    scene = next(
        (
            row
            for row in manifest_payload.get("scenes", ())
            if str(row.get("scene_id")) == str(scene_id)
        ),
        None,
    )
    if scene is None:
        raise ValueError(f"Scene {scene_id!r} is absent from {manifest}.")
    invalid = [value for value in positions if value >= len(scene.get("frames", ()))]
    if invalid:
        raise ValueError(
            f"Manifest frame positions {invalid} are outside "
            f"[0,{len(scene.get('frames', ())) - 1}]."
        )
    if geometry_payload is not None and len(positions) != len(geometry_payload["image_paths"]):
        raise ValueError(
            "Frame positions and geometry cache have different lengths: "
            f"{len(positions)} vs {len(geometry_payload['image_paths'])}."
        )
    return positions


def _resolve_prompts(values: list[str] | None, payload: dict[str, Any]) -> tuple[str, ...]:
    if values:
        prompts: list[str] = []
        for value in values:
            prompts.extend(
                piece.strip() for piece in str(value).split(",") if piece.strip()
            )
        if prompts:
            return tuple(dict.fromkeys(prompts))
    metadata = _artifact_metadata(payload)
    candidate = metadata.get("prompts", ())
    if isinstance(candidate, (list, tuple)) and candidate:
        return tuple(dict.fromkeys(str(value) for value in candidate if str(value).strip()))
    categories = [
        str(row.get("category", ""))
        for row in payload.get("object_tracks", ())
        if isinstance(row, dict) and str(row.get("category", "")).strip()
    ]
    return tuple(dict.fromkeys(categories))


def _load_aliases(path: Path | None) -> dict[str, tuple[str, ...]]:
    if path is None:
        return {}
    with path.expanduser().resolve().open("r", encoding="utf8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Prompt aliases JSON must be an object.")
    output: dict[str, tuple[str, ...]] = {}
    for key, value in payload.items():
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, (list, tuple)):
            raise ValueError("Prompt alias values must be strings or string lists.")
        output[str(key)] = tuple(str(item) for item in value)
    return output


def _resolve_output_dir(args: argparse.Namespace, artifacts: dict[str, Path]) -> Path:
    if args.output_dir is not None:
        return args.output_dir.expanduser().resolve()
    common = Path(next(iter(artifacts.values()))).parent
    return common / "evaluation"


def _load_ground_truth_world_to_camera(
    manifest: Path,
    *,
    scene_id: str,
    frame_indices: tuple[int, ...],
) -> torch.Tensor:
    """Load ScanNet++ camera-pose GT for evaluation only."""

    with manifest.open("r", encoding="utf8") as handle:
        payload = json.load(handle)
    scene = next(
        (
            row
            for row in payload.get("scenes", ())
            if str(row.get("scene_id")) == str(scene_id)
        ),
        None,
    )
    if scene is None:
        raise ValueError(f"Scene {scene_id!r} is absent from {manifest}.")
    frames = scene.get("frames", ())
    poses: list[torch.Tensor] = []
    for frame_index in frame_indices:
        frame = frames[int(frame_index)]
        value = frame.get("world_to_camera")
        if value is None:
            raise ValueError(
                f"ScanNet++ frame {int(frame_index)} has no world_to_camera GT."
            )
        poses.append(
            _pose_batch(
                torch.as_tensor(value, dtype=torch.float64),
                name=f"GT world_to_camera frame {int(frame_index)}",
            )[0]
        )
    return torch.stack(poses, dim=0)


def _build_pose_evaluation(
    *,
    input_source: Path,
    geometry_payload: dict[str, Any] | None,
    target_world_to_camera: torch.Tensor,
    frame_ids: tuple[int, ...],
    alignment: Any,
    refined_pose_path: Path | None,
) -> dict[str, object]:
    """Evaluate raw and optional refined trajectories under one shared gauge."""

    output: dict[str, object] = {
        "status": "skipped",
        "gt_source": "manifest.frames[].world_to_camera",
        "alignment": (
            "shared_reference_pointmap_sim3" if alignment is not None else "none"
        ),
        "branches": {},
    }
    if geometry_payload is None:
        output["reason"] = "geometry_cache_not_provided"
        return output

    raw_world_to_camera = _pose_batch(
        torch.as_tensor(geometry_payload["world_to_camera"]),
        name="HorizonStream world_to_camera",
    )
    raw_camera_to_world = torch.linalg.inv(raw_world_to_camera)
    if raw_camera_to_world.shape[0] != len(frame_ids):
        raise ValueError(
            "Raw HorizonStream pose count differs from selected frame count: "
            f"{raw_camera_to_world.shape[0]} vs {len(frame_ids)}."
        )
    raw_result = evaluate_camera_trajectory(
        raw_camera_to_world,
        target_world_to_camera,
        alignment=alignment,
        frame_ids=frame_ids,
    )
    branches: dict[str, object] = {
        "raw_pose": {
            "trajectory_source": "geometry_cache.world_to_camera",
            **raw_result,
        }
    }

    input_root = _pose_input_root(input_source)
    default_refined_path = input_root / "object_pose_refinement" / "refined_camera_to_world.pt"
    if refined_pose_path is None:
        refined_path = default_refined_path if default_refined_path.is_file() else None
    else:
        refined_path = refined_pose_path.expanduser().resolve()
        if not refined_path.is_file():
            raise FileNotFoundError(f"Refined pose artifact does not exist: {refined_path}")

    if refined_path is None:
        output.update(
            {
                "status": "raw_only",
                "reason": "refined_camera_to_world_not_found",
                "refined_pose_path": str(default_refined_path),
                "branches": branches,
            }
        )
        return output

    refined_payload = _load_torch_payload(refined_path)
    refined_value = refined_payload.get("refined_camera_to_world")
    if refined_value is None:
        raise ValueError(
            f"Refined pose artifact has no refined_camera_to_world: {refined_path}"
        )
    stored_frame_ids = tuple(
        int(value) for value in refined_payload.get("frame_ids", ())
    )
    expected_sequence_ids = tuple(range(len(frame_ids)))
    if stored_frame_ids and stored_frame_ids not in {
        expected_sequence_ids,
        tuple(frame_ids),
    }:
        raise ValueError(
            "Refined pose frame_ids do not match the selected sequence: "
            f"stored={stored_frame_ids[:5]}... expected={expected_sequence_ids[:5]}..."
        )
    refined_camera_to_world = _pose_batch(
        torch.as_tensor(refined_value),
        name="refined_camera_to_world",
    )
    if refined_camera_to_world.shape[0] != len(frame_ids):
        raise ValueError(
            "Refined pose count differs from selected frame count: "
            f"{refined_camera_to_world.shape[0]} vs {len(frame_ids)}."
        )
    refined_result = evaluate_camera_trajectory(
        refined_camera_to_world,
        target_world_to_camera,
        alignment=alignment,
        frame_ids=frame_ids,
    )
    branches["object_pose_refined"] = {
        "trajectory_source": str(refined_path),
        **refined_result,
    }
    output.update(
        {
            "status": "ok",
            "refined_pose_path": str(refined_path),
            "branches": branches,
        }
    )
    return output


def _pose_input_root(source: Path) -> Path:
    path = Path(source).expanduser().resolve()
    if path.is_file():
        return path.parent
    if (path / "semantic_map.pt").is_file():
        return path
    return path


def _load_torch_payload(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a dictionary pose artifact, got {type(payload)!r}.")
    return payload


def _pose_batch(value: torch.Tensor, *, name: str) -> torch.Tensor:
    poses = torch.as_tensor(value).detach().double().cpu()
    if poses.ndim == 2:
        poses = poses.unsqueeze(0)
    if poses.ndim != 3:
        raise ValueError(f"{name} must have shape [S,3,4] or [S,4,4].")
    if tuple(poses.shape[-2:]) == (3, 4):
        homogeneous = torch.eye(4, dtype=poses.dtype).repeat(poses.shape[0], 1, 1)
        homogeneous[:, :3] = poses
        poses = homogeneous
    elif tuple(poses.shape[-2:]) != (4, 4):
        raise ValueError(f"{name} must have shape [S,3,4] or [S,4,4].")
    if not bool(torch.isfinite(poses).all()):
        raise ValueError(f"{name} contains non-finite values.")
    return poses


def _pose_summary_rows(
    pose_evaluation: dict[str, object],
    scene_id: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    branches = pose_evaluation.get("branches", {})
    if not isinstance(branches, dict):
        return rows
    for branch, value in branches.items():
        if not isinstance(value, dict):
            continue
        summary = value.get("summary", {})
        if not isinstance(summary, dict):
            continue
        rows.append(
            {
                "scene_id": scene_id,
                "branch": str(branch),
                "trajectory_source": value.get("trajectory_source", ""),
                **summary,
            }
        )
    return rows


def _pose_frame_rows(
    pose_evaluation: dict[str, object],
    scene_id: str,
) -> list[dict[str, object]]:
    return _pose_detail_rows(pose_evaluation, scene_id, detail_key="frames")


def _pose_rpe_rows(
    pose_evaluation: dict[str, object],
    scene_id: str,
) -> list[dict[str, object]]:
    return _pose_detail_rows(pose_evaluation, scene_id, detail_key="rpe")


def _pose_detail_rows(
    pose_evaluation: dict[str, object],
    scene_id: str,
    *,
    detail_key: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    branches = pose_evaluation.get("branches", {})
    if not isinstance(branches, dict):
        return rows
    for branch, value in branches.items():
        if not isinstance(value, dict):
            continue
        details = value.get(detail_key, ())
        if not isinstance(details, list):
            continue
        for detail in details:
            if isinstance(detail, dict):
                rows.append(
                    {
                        "scene_id": scene_id,
                        "branch": str(branch),
                        **detail,
                    }
                )
    return rows


def _pointmap_frame_rows(
    pointmap_alignment: dict[str, object],
    scene_id: str,
) -> list[dict[str, object]]:
    rows = pointmap_alignment.get("frames", ())
    if not isinstance(rows, list):
        return []
    return [{"scene_id": scene_id, **dict(row)} for row in rows if isinstance(row, dict)]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf8")
        return path
    fieldnames = sorted({str(key) for row in rows for key in row})
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})
    return path


def _csv_value(value: object) -> object:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_json_safe(value), sort_keys=True, ensure_ascii=False)
    return value


def _write_copyable(
    path: Path,
    summary: dict[str, object],
    rows: list[dict[str, object]],
    pointmap_alignment: dict[str, object],
    pose_evaluation: dict[str, object],
) -> None:
    lines = [
        "===== EXPORTED_HORIZONSTREAM_SEMANTIC_MAP_EVALUATION_BEGIN =====",
        f"revision={REVISION}",
        "decision=EVALUATION_ONLY",
        f"scene_id={summary['scene_id']}",
        f"frame_count={summary['frame_count']}",
        f"map_source={summary['map_source']}",
        f"alignment={summary['alignment']}",
        f"pointmap_alignment_status={pointmap_alignment.get('status')}",
    ]
    for row in rows:
        lines.append(
            "branch="
            + str(row.get("variant", ""))
            + " "
            + " ".join(
                f"{key}={row.get(key)}"
                for key in (
                    "matched_objects",
                    "object_accuracy_m",
                    "object_completeness_m",
                    "fscore_5cm",
                    "fscore_10cm",
                    "voxel_iou_5cm",
                    "ghost_point_ratio",
                    "duplicate_object_rate",
                )
            )
        )
    point_summary = pointmap_alignment.get("summary", {})
    if isinstance(point_summary, dict):
        lines.append(
            "pointmap "
            + " ".join(
                f"{key}={point_summary.get(key)}"
                for key in ("rmse_m", "median_m", "p90_m", "fit_rmse_m", "scale")
            )
        )
    lines.append(
        "pose_gt_source=manifest.frames[].world_to_camera "
        f"pose_evaluation_status={pose_evaluation.get('status')}"
    )
    pose_branches = pose_evaluation.get("branches", {})
    if isinstance(pose_branches, dict):
        for branch, value in pose_branches.items():
            if not isinstance(value, dict):
                continue
            pose_summary = value.get("summary", {})
            if not isinstance(pose_summary, dict):
                continue
            lines.append(
                "pose_branch="
                + str(branch)
                + " "
                + " ".join(
                    f"{key}={pose_summary.get(key)}"
                    for key in (
                        "ate_rmse_m",
                        "ate_mean_m",
                        "ate_median_m",
                        "rotation_error_mean_deg",
                        "rpe_translation_rmse_m",
                        "rpe_rotation_rmse_deg",
                        "rpe_pair_count",
                    )
                )
            )
    lines.append("===== EXPORTED_HORIZONSTREAM_SEMANTIC_MAP_EVALUATION_END =====")
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def _print_summary(
    summary: dict[str, object],
    rows: list[dict[str, object]],
    pointmap_alignment: dict[str, object],
    pose_evaluation: dict[str, object],
) -> None:
    print("EXPORTED HORIZONSTREAM SEMANTIC-MAP OFFLINE EVALUATION")
    print(f"scene={summary['scene_id']} frames={summary['frame_count']}")
    print(f"map_source={summary['map_source']} alignment={summary['alignment']}")
    for row in rows:
        print(
            f"branch={row['variant']} "
            f"object_accuracy={_format(row['object_accuracy_m'])} "
            f"object_completeness={_format(row['object_completeness_m'])} "
            f"F5cm={_format(row['fscore_5cm'])} "
            f"F10cm={_format(row['fscore_10cm'])} "
            f"voxelIoU5cm={_format(row['voxel_iou_5cm'])} "
            f"ghost={_format(row['ghost_point_ratio'])} "
            f"duplicate_rate={_format(row['duplicate_object_rate'])}"
        )
    point_summary = pointmap_alignment.get("summary", {})
    if isinstance(point_summary, dict):
        print(
            f"pointmap_rmse={_format(point_summary.get('rmse_m'))} "
            f"median={_format(point_summary.get('median_m'))} "
            f"p90={_format(point_summary.get('p90_m'))}"
        )
    print(
        f"pose_gt=manifest.frames[].world_to_camera "
        f"pose_status={pose_evaluation.get('status')}"
    )
    pose_branches = pose_evaluation.get("branches", {})
    if isinstance(pose_branches, dict):
        for branch, value in pose_branches.items():
            if not isinstance(value, dict):
                continue
            pose_summary = value.get("summary", {})
            if not isinstance(pose_summary, dict):
                continue
            print(
                f"pose_branch={branch} "
                f"ATE_RMSE={_format(pose_summary.get('ate_rmse_m'))} "
                f"ATE_mean={_format(pose_summary.get('ate_mean_m'))} "
                f"RPE_t_RMSE={_format(pose_summary.get('rpe_translation_rmse_m'))} "
                f"RPE_r_RMSE={_format(pose_summary.get('rpe_rotation_rmse_deg'))}"
            )
    print(f"summary={summary['outputs']['summary']}")
    print(f"copyable_result={Path(summary['outputs']['summary']).with_name('copyable_result.txt')}")


def _format(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "nan" if not math.isfinite(number) else f"{number:.6f}"


def _parse_size(value: str) -> tuple[int, int]:
    pieces = value.lower().replace("x", ",").split(",")
    if len(pieces) != 2:
        raise argparse.ArgumentTypeError("Expected SIZE as H,W or HxW.")
    try:
        height, width = (int(piece) for piece in pieces)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("SIZE values must be integers.") from exc
    if height <= 0 or width <= 0:
        raise argparse.ArgumentTypeError("SIZE values must be positive.")
    return height, width


def _json_safe(value: object) -> object:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if torch.is_tensor(value):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


if __name__ == "__main__":
    main()
