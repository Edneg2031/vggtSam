#!/usr/bin/env python3
"""Create a deterministic multi-scene semantic-map experiment protocol.

The planner reads only manifest metadata and asset existence.  It writes a
new YAML configuration and optional per-clip configs; it never runs a model.
Prompt selection is explicitly marked annotation-assisted, matching the
existing V0 protocol.  Ground-truth masks are not copied into any runtime
configuration.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from streaming_couping.src.data import extract_object_labels, resolve_manifest_path
from streaming_couping.src.storage import expand_storage_path


REVISION = "semantic_map_multiclip_protocol_planner_r1"
DEFAULT_EXCLUDED_LABELS = ("wall", "floor", "ceiling", "room", "building")


def main() -> None:
    args = _parse_args()
    base_config_path = Path(args.base_config).expanduser().resolve()
    with base_config_path.open("r", encoding="utf8") as handle:
        base = yaml.safe_load(handle) or {}
    manifest_value = (base.get("dataset", {}) or {}).get("manifest")
    manifest_path = expand_storage_path(manifest_value, base=base_config_path.parent)
    with manifest_path.open("r", encoding="utf8") as handle:
        manifest = json.load(handle)

    requested = _split_values(args.scene_ids)
    scenes = [
        scene
        for scene in manifest.get("scenes", ())
        if not requested or str(scene.get("scene_id")) in requested
    ]
    if requested:
        found = {str(scene.get("scene_id")) for scene in scenes}
        missing = [scene_id for scene_id in requested if scene_id not in found]
        if missing:
            raise ValueError(f"Requested scenes are absent from manifest: {missing}")
    scenes = sorted(scenes, key=lambda row: str(row.get("scene_id", "")))
    if not scenes:
        raise ValueError("No scenes remain after scene selection.")

    excluded = {
        value.strip().lower()
        for value in (args.exclude_label or DEFAULT_EXCLUDED_LABELS)
        if value.strip()
    }
    clips: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for scene in scenes:
        clip, reason = _plan_scene_clip(
            scene,
            manifest_path=manifest_path,
            clip_length=args.clip_length,
            max_prompts=args.max_prompts,
            excluded_labels=excluded,
            prefix=args.name_prefix,
        )
        if clip is None:
            skipped.append(
                {"scene_id": str(scene.get("scene_id", "")), "reason": reason}
            )
        else:
            clips.append(clip)
    if not clips:
        raise ValueError(f"No eligible clips were planned; skipped={skipped}")

    split_map = _assign_splits([str(clip["scene_id"]) for clip in clips])
    for clip in clips:
        clip["split"] = split_map[str(clip["scene_id"])]

    planned = copy.deepcopy(base)
    dataset = planned.setdefault("dataset", {})
    dataset["clips"] = clips
    # This makes the generated protocol usable for cache construction.  The
    # per-clip configs below override it before any baseline audit/export.
    planned.setdefault("baseline", {})["clip_name"] = clips[0]["name"]
    planned["baseline"].setdefault("pose", {})[
        "selected_branch"
    ] = "raw_streamvggt"
    planned["baseline"]["pose"]["allow_raw_fallback"] = True
    planned["baseline"].setdefault("frames", {})["evaluation"] = list(
        _baseline_evaluation_frames(clips[0]["frame_indices"])
    )
    _absolutize_static_paths(planned, base_config_path.parent)

    output_config = Path(args.output_config).expanduser().resolve()
    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_config.write_text(
        yaml.safe_dump(planned, sort_keys=False),
        encoding="utf8",
    )
    protocol = {
        "schema": 1,
        "revision": REVISION,
        "manifest": str(manifest_path),
        "base_config": str(base_config_path),
        "clip_length": int(args.clip_length),
        "prompt_selection_annotation_gt_used": 1,
        "runtime_prompt_selection_gt_fields": 0,
        "excluded_labels": sorted(excluded),
        "pose_policy": "raw_streamvggt_control_qk_diagnostic_only",
        "clips": clips,
        "skipped_scenes": skipped,
        "split": split_map,
        "generated_config": str(output_config),
    }
    protocol_path = output_config.with_name("semantic_mapping_multiclip_protocol.json")
    protocol_path.write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n",
        encoding="utf8",
    )

    if args.per_clip_dir:
        _write_per_clip_configs(
            planned,
            clips,
            output_dir=Path(args.per_clip_dir).expanduser().resolve(),
        )

    print("MULTI-CLIP SEMANTIC-MAP PROTOCOL")
    print(f"manifest={manifest_path}")
    print(f"clips={len(clips)} skipped={len(skipped)} length={args.clip_length}")
    for clip in clips:
        print(
            f"  clip={clip['name']} scene={clip['scene_id']} split={clip['split']} "
            f"frames={clip['frame_indices'][0]}..{clip['frame_indices'][-1]} "
            f"prompts={','.join(clip['instance_prompts'])}"
        )
    for row in skipped:
        print(f"  skipped scene={row['scene_id']} reason={row['reason']}")
    print(f"config={output_config}")
    print(f"protocol={protocol_path}")
    if args.per_clip_dir:
        print(f"per_clip_configs={Path(args.per_clip_dir).expanduser().resolve()}")


def _plan_scene_clip(
    scene: Mapping[str, Any],
    *,
    manifest_path: Path,
    clip_length: int,
    max_prompts: int,
    excluded_labels: set[str],
    prefix: str,
) -> tuple[dict[str, Any] | None, str]:
    scene_id = str(scene.get("scene_id", ""))
    frames = list(scene.get("frames", ()))
    valid_positions = [
        index
        for index, frame in enumerate(frames)
        if _asset_exists(frame.get("image_path"), manifest_path)
        and _asset_exists(frame.get("pointmap"), manifest_path)
        and _asset_exists(frame.get("instance_mask"), manifest_path)
    ]
    if len(valid_positions) < int(clip_length):
        return None, f"complete_frames={len(valid_positions)}<{clip_length}"
    selected = _uniform_indices(valid_positions, clip_length)
    labels = extract_object_labels(dict(scene.get("objects", {}) or {}))
    prompts = sorted(
        {
            label.strip()
            for label in labels.values()
            if label.strip() and label.strip().lower() not in excluded_labels
        },
        key=lambda value: (value.lower(), value),
    )[: int(max_prompts)]
    if not prompts:
        prompts = ["object"]
    name = f"{prefix}{scene_id}_uniform{int(clip_length)}_r1"
    return {
        "name": name,
        "scene_id": scene_id,
        "frame_indices": selected,
        "instance_ids": list(range(16)),
        "instance_source": "sam31_online",
        "instance_prompt": prompts[0],
        "instance_prompts": prompts,
        "reference_sequence_index": 0,
        "split": "unassigned",
    }, ""


def _uniform_indices(values: list[int], length: int) -> list[int]:
    if length < 2:
        raise ValueError("clip_length must be at least 2.")
    if len(values) < length:
        raise ValueError("Not enough valid frame positions.")
    last = len(values) - 1
    selected_positions = [round(index * last / (length - 1)) for index in range(length)]
    selected = [values[position] for position in selected_positions]
    if len(set(selected)) != len(selected):
        raise ValueError("Uniform clip sampling produced duplicate frame indices.")
    return selected


def _assign_splits(scene_ids: list[str]) -> dict[str, str]:
    ordered = sorted(dict.fromkeys(scene_ids))
    if len(ordered) == 1:
        return {ordered[0]: "development_audit"}
    test_count = max(1, round(len(ordered) * 0.2))
    validation_count = max(1, round(len(ordered) * 0.2))
    if test_count + validation_count >= len(ordered):
        test_count = validation_count = 1
    train = ordered[: -(test_count + validation_count)]
    validation = ordered[-(test_count + validation_count) : -test_count]
    test = ordered[-test_count:]
    return {
        scene_id: "train" if scene_id in train else "validation" if scene_id in validation else "test"
        for scene_id in ordered
    }


def _asset_exists(value: Any, manifest_path: Path) -> bool:
    if value is None or not str(value).strip():
        return False
    try:
        return resolve_manifest_path(value, manifest_path).is_file()
    except FileNotFoundError:
        return False


def _absolutize_static_paths(payload: dict[str, Any], base_dir: Path) -> None:
    if payload.get("recovery_config"):
        payload["recovery_config"] = str(
            expand_storage_path(payload["recovery_config"], base=base_dir)
        )
    dataset = payload.get("dataset", {}) or {}
    if dataset.get("manifest"):
        dataset["manifest"] = str(
            expand_storage_path(dataset["manifest"], base=base_dir)
        )
    features = payload.get("features", {}) or {}
    if features.get("cache_dir"):
        features["cache_dir"] = str(
            expand_storage_path(features["cache_dir"], base=base_dir)
        )


def _write_per_clip_configs(
    planned: Mapping[str, Any],
    clips: Iterable[Mapping[str, Any]],
    *,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for clip in clips:
        payload = copy.deepcopy(dict(planned))
        name = str(clip["name"])
        run_root = output_dir
        baseline = payload.setdefault("baseline", {})
        baseline["clip_name"] = name
        baseline["output_dir"] = str(run_root / "v0" / name)
        baseline.setdefault("pose", {})["selected_branch"] = "raw_streamvggt"
        baseline["pose"]["allow_raw_fallback"] = True
        baseline.setdefault("frames", {})["evaluation"] = list(
            _baseline_evaluation_frames(clip["frame_indices"])
        )
        baseline.setdefault("pose", {})["qk_pose_output"] = str(
            run_root / "v0" / name / "qk_pose_retrieval" / "qk_pose_output.pt"
        )
        payload.setdefault("qk_pose_retrieval", {})["output_dir"] = str(
            run_root / "v0" / name / "qk_pose_retrieval"
        )
        semantic = payload.setdefault("semantic_map", {})
        semantic["output_dir"] = str(run_root / "v0" / name / "semantic_map")
        v21 = semantic.setdefault("v2_1_geometry_recovery", {})
        v22 = semantic.setdefault("v2_2_geometry_recovery", {})
        v23 = semantic.setdefault("v2_3_geometry_recovery", {})
        v21["output_dir"] = str(run_root / "v21" / name / "semantic_map")
        v22["output_dir"] = str(run_root / "v22" / name / "semantic_map")
        v23["output_dir"] = str(run_root / "v23" / name / "semantic_map")
        v23["v22_map_dir"] = str(run_root / "v22" / name / "semantic_map")
        path = output_dir / f"{name}.yaml"
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf8")


def _split_values(value: str | None) -> list[str]:
    if value is None:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _baseline_evaluation_frames(frame_indices: Iterable[int]) -> list[int]:
    """Return the twelve future frames required by the frozen V0 protocol."""

    future = [int(value) for value in frame_indices][1:]
    if len(future) < 12:
        raise ValueError(
            "Each multi-clip protocol must contain at least twelve future "
            f"frames; got {len(future)}."
        )
    return _uniform_indices(future, 12)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-config",
        default="streaming_couping/configs/v0_baseline.yaml",
    )
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--per-clip-dir")
    parser.add_argument("--scene-ids", help="Comma-separated scene IDs; default=all manifest scenes.")
    parser.add_argument("--clip-length", type=int, default=30)
    parser.add_argument("--max-prompts", type=int, default=12)
    parser.add_argument("--name-prefix", default="")
    parser.add_argument(
        "--exclude-label",
        action="append",
        default=None,
        help="Repeatable; replaces the default structural-label exclusions.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
