#!/usr/bin/env python3
"""Generate class-agnostic SAM3.1 visual-point tracks without opening GT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from streaming_couping.src.backbones.sam3_wrapper import SAM3Wrapper
from streaming_couping.src.config import load_config
from streaming_couping.src.learned_pose.cache import cache_path, load_feature_cache
from streaming_couping.src.learned_pose.config import (
    ClipConfig,
    load_learned_pose_config,
)
from streaming_couping.src.recovery import output_mask_to_stream
from streaming_couping.src.storage import expand_storage_path


REVISION = "sam31_class_agnostic_visual_point_proposal_r1"
ARTIFACT_NAME = "auto_proposal.pt"


def main() -> None:
    args = _parse_args()
    config_path = Path(args.config).expanduser().resolve()
    data = load_learned_pose_config(config_path)
    clips = _select_clips(data.clips, args.clip)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else expand_storage_path(
            "${VGGT_SAM_STORAGE_ROOT}/outputs/"
            "streaming_couping_sam31_auto_proposal",
            base=config_path.parent,
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    payloads = [
        (clip, _load_generation_payload(data, clip))
        for clip in clips
    ]
    first_clip = clips[0]
    recovery = load_config(
        data.recovery_config,
        {
            "manifest": data.manifest,
            "scene_id": first_clip.scene_id,
            "frame_indices": first_clip.frame_indices,
            "sam3_device": args.device or data.sam3_device,
            "geometry_device": data.geometry_device,
            "output_dir": output_dir,
        },
    )
    if recovery.sam3_version != "sam3.1":
        raise ValueError("Automatic visual-point discovery requires SAM3.1.")
    # Capacity comes from the frozen downstream cache contract, not from the
    # configured semantic IDs (which may themselves have been annotation
    # derived).  This keeps class-agnostic proposal generation independent of
    # the noun list and GT instance registry.
    maximum_slots = max(
        int(_raw_output_masks(payload, args.raw_cache_variant).shape[1])
        for _, payload in payloads
    )
    requested_max_objects = (
        maximum_slots
        if args.max_objects is None
        else int(args.max_objects)
    )
    if requested_max_objects < 1:
        raise ValueError("--max-objects must be positive.")
    if requested_max_objects > maximum_slots:
        raise ValueError(
            f"--max-objects={requested_max_objects} exceeds frozen cache "
            f"capacity={maximum_slots}."
        )
    model_capacity = max(int(recovery.sam3_max_num_objects), maximum_slots)
    tracker = SAM3Wrapper(
        repo_path=recovery.sam3_repo,
        checkpoint_path=recovery.sam3_checkpoint,
        device=recovery.sam3_device,
        output_threshold=recovery.sam3_output_threshold,
        prompt_with_box=False,
        version=recovery.sam3_version,
        use_fa3=recovery.sam3_use_fa3,
        max_num_objects=model_capacity,
        multiplex_count=max(
            int(recovery.sam3_multiplex_count),
            model_capacity,
        ),
    ).load()

    policy = {
        "max_objects": int(requested_max_objects),
        "discovery_stride": int(args.discovery_stride),
        "grid_rows": int(args.grid_rows),
        "grid_columns": int(args.grid_columns),
        "points_per_discovery": int(args.points_per_discovery),
        "min_mask_pixels": int(args.min_mask_pixels),
        "max_mask_area_ratio": float(args.max_mask_area_ratio),
        "duplicate_iou": float(args.duplicate_iou),
        "duplicate_intersection_over_smaller": float(
            args.duplicate_intersection_over_smaller
        ),
    }
    print("SAM3.1 CLASS-AGNOSTIC VISUAL-POINT DISCOVERY")
    print(f"config={config_path}")
    print(f"device={recovery.sam3_device} clips={len(clips)}")
    print(f"policy={policy}")
    print(
        "candidate_generation=RGB plus fixed visual point grid only; "
        "semantic text prompts and GT are not read"
    )

    clip_rows: list[dict[str, object]] = []
    for clip, payload in payloads:
        artifact_path = output_dir / clip.name / ARTIFACT_NAME
        if artifact_path.is_file() and not args.overwrite:
            artifact = _load_artifact(artifact_path)
            _validate_reusable_artifact(
                artifact,
                clip=clip,
                policy=policy,
                raw_cache_variant=args.raw_cache_variant,
            )
            clip_rows.append(_artifact_summary(artifact, artifact_path))
            print(f"  reused clip={clip.name} artifact={artifact_path}")
            continue
        raw_output = _raw_output_masks(payload, args.raw_cache_variant)
        slot_capacity = int(raw_output.shape[1])
        max_objects = min(requested_max_objects, slot_capacity)
        output_size = (
            int(raw_output.shape[-2]),
            int(raw_output.shape[-1]),
        )
        tracking_policy = {
            key: value
            for key, value in policy.items()
            if key != "max_objects"
        }
        tracked = tracker.track_auto_points_forward(
            payload["image_paths"],
            output_size=output_size,
            max_objects=max_objects,
            **tracking_policy,
        )
        artifact = _build_artifact(
            clip=clip,
            payload=payload,
            tracked=tracked,
            output_size=output_size,
            slot_capacity=slot_capacity,
            policy=policy,
            raw_cache_variant=args.raw_cache_variant,
        )
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(artifact, artifact_path)
        row = _artifact_summary(artifact, artifact_path)
        clip_rows.append(row)
        diagnostics = artifact["proposal_diagnostics"]
        accepted = sum(int(value["accepted"]) for value in diagnostics)
        print(
            f"  frozen clip={clip.name} tracks={row['retained_tracks']} "
            f"attempts={row['prompt_attempts']} accepted={accepted} "
            f"reasons={row['proposal_reason_counts']} "
            f"artifact={artifact_path}"
        )

    generation_summary = {
        "schema": 1,
        "revision": REVISION,
        "candidate_generation_gt_fields": 0,
        "semantic_text_prompt_used": 0,
        "visual_point_prompt_used": 1,
        "policy": policy,
        "clips": clip_rows,
    }
    summary_path = output_dir / "generation_summary.json"
    summary_path.write_text(
        json.dumps(generation_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf8",
    )
    print(f"generation_summary={summary_path}")


def _load_generation_payload(data, clip: ClipConfig) -> Mapping[str, Any]:
    payload = load_feature_cache(cache_path(data, clip))
    required = (
        "clip_name",
        "scene_id",
        "frame_indices",
        "image_paths",
        "source_sizes",
        "image_size",
        "image_mode",
        "tracking_variant_masks_output",
    )
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError(
            f"Frozen cache for {clip.name!r} lacks generation fields {missing}."
        )
    if str(payload["clip_name"]) != clip.name:
        raise ValueError(f"Cache identity differs from clip {clip.name!r}.")
    frames = tuple(int(value) for value in payload["frame_indices"])
    if frames != tuple(int(value) for value in clip.frame_indices):
        raise ValueError(f"Cache frame order differs for {clip.name!r}.")
    if len(payload["image_paths"]) != len(frames):
        raise ValueError(f"Cache image count differs for {clip.name!r}.")
    return payload


def _raw_output_masks(
    payload: Mapping[str, Any],
    raw_cache_variant: str,
) -> torch.Tensor:
    variants = payload.get("tracking_variant_masks_output")
    if not isinstance(variants, Mapping) or raw_cache_variant not in variants:
        raise ValueError(
            f"Cache lacks raw tracking variant {raw_cache_variant!r}."
        )
    masks = torch.as_tensor(variants[raw_cache_variant]).bool()
    if masks.ndim != 4:
        raise ValueError("Raw tracking masks must have shape [S,K,H,W].")
    return masks


def _build_artifact(
    *,
    clip: ClipConfig,
    payload: Mapping[str, Any],
    tracked,
    output_size: tuple[int, int],
    slot_capacity: int,
    policy: Mapping[str, object],
    raw_cache_variant: str,
) -> dict[str, object]:
    sequence = len(clip.frame_indices)
    slots = int(slot_capacity)
    if slots < 1:
        raise ValueError("Auto-proposal slot capacity must be positive.")
    tracked_masks = torch.as_tensor(tracked.masks).detach().cpu().bool()
    tracked_scores = torch.as_tensor(tracked.scores).detach().cpu().float()
    if tracked_masks.ndim != 4 or tracked_scores.ndim != 2:
        raise ValueError("SAM auto-proposal output tensors have invalid ranks.")
    if tracked_masks.shape[:2] != tracked_scores.shape:
        raise ValueError("SAM auto-proposal masks and scores disagree.")
    if tracked_masks.shape[0] != sequence:
        raise ValueError(
            f"SAM auto-proposal returned {tracked_masks.shape[0]} frames; "
            f"expected {sequence}."
        )
    retained = min(slots, int(tracked_masks.shape[1]))
    output_masks = torch.zeros(
        sequence,
        slots,
        output_size[0],
        output_size[1],
        dtype=torch.bool,
    )
    scores = torch.zeros(sequence, slots, dtype=torch.float32)
    if retained:
        output_masks[:, :retained] = tracked_masks[:, :retained]
        scores[:, :retained] = tracked_scores[:, :retained]

    processed_size = tuple(int(value) for value in payload["image_size"])
    source_sizes = tuple(
        tuple(int(item) for item in value)
        for value in payload["source_sizes"]
    )
    stream_masks = torch.zeros(
        sequence,
        slots,
        processed_size[0],
        processed_size[1],
        dtype=torch.bool,
    )
    for frame in range(sequence):
        for slot in range(retained):
            if not bool(output_masks[frame, slot].any()):
                continue
            stream_masks[frame, slot] = output_mask_to_stream(
                output_masks[frame, slot],
                source_size=source_sizes[frame],
                processed_size=processed_size,
                image_mode=str(payload["image_mode"]),
            )

    track_ids = [-1] * slots
    births = [-1] * slots
    prompts = [""] * slots
    for slot in range(retained):
        track_ids[slot] = int(tracked.obj_ids[slot])
        births[slot] = int(tracked.birth_indices[slot])
        # Generic "object" is evaluation metadata only. It was not passed to
        # SAM3 during proposal generation.
        prompts[slot] = "object"
    if len(tracked.obj_ids) != retained or len(tracked.birth_indices) != retained:
        raise ValueError("SAM auto-proposal track metadata disagrees with masks.")
    diagnostics = []
    for row in tracked.aux.get("proposal_diagnostics", ()):
        current = dict(row)
        sequence_index = int(current["sequence_index"])
        current.update(
            {
                "clip": clip.name,
                "scene_id": clip.scene_id,
                "frame_index": int(clip.frame_indices[sequence_index]),
            }
        )
        diagnostics.append(current)
    return {
        "schema": 1,
        "revision": REVISION,
        "clip_name": clip.name,
        "scene_id": clip.scene_id,
        "split": clip.split,
        "frame_indices": list(clip.frame_indices),
        "candidate_generation_gt_fields": 0,
        "semantic_text_prompt_used": 0,
        "proposal_source": "class_agnostic_visual_point_grid",
        "raw_cache_variant_for_shape_only": str(raw_cache_variant),
        "policy": dict(policy),
        "tracking_masks_output": output_masks,
        "tracking_masks_stream": stream_masks,
        "tracking_scores": scores,
        "track_ids": track_ids,
        "track_prompts": prompts,
        "birth_indices": births,
        "proposal_diagnostics": diagnostics,
        "retained_tracks": int(retained),
        "slot_capacity": int(slots),
    }


def _artifact_summary(
    artifact: Mapping[str, Any],
    path: Path,
) -> dict[str, object]:
    diagnostics = list(artifact.get("proposal_diagnostics", ()))
    reason_counts: dict[str, int] = {}
    for row in diagnostics:
        reason = str(row.get("reason", "unknown"))
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return {
        "clip": str(artifact["clip_name"]),
        "scene_id": str(artifact["scene_id"]),
        "split": str(artifact.get("split", "")),
        "retained_tracks": int(artifact.get("retained_tracks", 0)),
        "slot_capacity": int(artifact.get("slot_capacity", 0)),
        "prompt_attempts": int(
            sum(int(row.get("obj_id", -1)) >= 0 for row in diagnostics)
        ),
        "accepted_proposals": int(
            sum(int(row.get("accepted", 0)) for row in diagnostics)
        ),
        "proposal_reason_counts": reason_counts,
        "covered_points_skipped": int(
            sum(
                str(row.get("reason", ""))
                == "covered_by_retained_mask"
                for row in diagnostics
            )
        ),
        "artifact": str(path),
    }


def _validate_reusable_artifact(
    artifact: Mapping[str, Any],
    *,
    clip: ClipConfig,
    policy: Mapping[str, object],
    raw_cache_variant: str,
) -> None:
    """Reject stale artifacts instead of silently mixing experiment policies."""

    if str(artifact.get("revision", "")) != REVISION:
        raise ValueError(
            f"Artifact revision differs for {clip.name}; rerun with "
            "--overwrite."
        )
    if str(artifact.get("clip_name", "")) != clip.name or str(
        artifact.get("scene_id", "")
    ) != clip.scene_id:
        raise ValueError(
            f"Artifact identity differs for {clip.name}; rerun with "
            "--overwrite."
        )
    artifact_frames = tuple(
        int(value) for value in artifact.get("frame_indices", ())
    )
    if artifact_frames != tuple(int(value) for value in clip.frame_indices):
        raise ValueError(
            f"Artifact frame order differs for {clip.name}; rerun with "
            "--overwrite."
        )
    stored_policy = artifact.get("policy")
    if not isinstance(stored_policy, Mapping) or dict(stored_policy) != dict(
        policy
    ):
        raise ValueError(
            f"Artifact proposal policy differs for {clip.name}; rerun with "
            "--overwrite."
        )
    if str(artifact.get("raw_cache_variant_for_shape_only", "")) != str(
        raw_cache_variant
    ):
        raise ValueError(
            f"Artifact raw-cache contract differs for {clip.name}; rerun "
            "with --overwrite."
        )


def _load_artifact(path: Path) -> Mapping[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, Mapping):
        raise ValueError(f"Auto-proposal artifact is invalid: {path}")
    return value


def _select_clips(
    clips: Sequence[ClipConfig],
    requested: Sequence[str] | None,
) -> tuple[ClipConfig, ...]:
    if not requested:
        return tuple(clips)
    names = {str(value) for value in requested}
    selected = tuple(
        clip
        for clip in clips
        if clip.name in names or clip.scene_id in names
    )
    missing = names - {
        value
        for clip in selected
        for value in (clip.name, clip.scene_id)
    }
    if missing:
        raise ValueError(f"Requested clips/scenes are absent: {sorted(missing)}")
    return selected


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="streaming_couping/configs/v0_baseline.yaml",
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--clip", action="append")
    parser.add_argument("--device")
    parser.add_argument("--raw-cache-variant", default="sam31_online_forward")
    parser.add_argument("--discovery-stride", type=int, default=5)
    parser.add_argument("--grid-rows", type=int, default=8)
    parser.add_argument("--grid-columns", type=int, default=12)
    parser.add_argument("--points-per-discovery", type=int, default=24)
    parser.add_argument("--min-mask-pixels", type=int, default=128)
    parser.add_argument("--max-mask-area-ratio", type=float, default=0.35)
    parser.add_argument("--duplicate-iou", type=float, default=0.80)
    parser.add_argument(
        "--duplicate-intersection-over-smaller",
        type=float,
        default=0.90,
    )
    parser.add_argument(
        "--max-objects",
        type=int,
        default=None,
        help=(
            "Maximum retained visual-point tracks. Defaults to the frozen "
            "raw-cache slot capacity."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
