#!/usr/bin/env python3
"""Cache the genuine causal SAM3.1 tracker memory-read feature for V9.8."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import gc
import hashlib
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from streaming_couping.scripts.run_v98_temporal_memory_causality import (
    load_v98_config,
)
from streaming_couping.src.backbones.sam3_wrapper import SAM3Wrapper
from streaming_couping.src.config import load_config
from streaming_couping.src.learned_pose.cache import cache_path, load_feature_cache
from streaming_couping.src.learned_pose.config import load_learned_pose_config
from streaming_couping.src.v98_temporal_memory import (
    SAM31MemoryReadCapture,
    combine_prompt_feature_runs,
    resolve_sam31_memory_model,
)


TEMPORAL_CACHE_VERSION = 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="streaming_couping/configs/v98_temporal_memory_causality.yaml",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--temporal-cache-path", default=None)
    parser.add_argument("--v97-dense-cache-path", default=None)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    config = load_v98_config(args.config)
    if args.temporal_cache_path:
        config = replace(
            config,
            temporal_cache_path=Path(args.temporal_cache_path).expanduser().resolve(),
        )
    if args.v97_dense_cache_path:
        config = replace(
            config,
            v97_dense_cache_path=Path(args.v97_dense_cache_path).expanduser().resolve(),
        )
    path = build_temporal_memory_cache(
        config,
        device=args.device or config.cache_device,
        rebuild=bool(args.rebuild),
    )
    print(f"V9.8 temporal memory cache={path}")


def build_temporal_memory_cache(config, *, device: str, rebuild: bool) -> Path:
    learned = load_learned_pose_config(config.data_config)
    clip = next(
        (value for value in learned.clips if value.name == config.clip_name), None
    )
    if clip is None:
        raise ValueError(f"V9.8 clip={config.clip_name!r} is not configured.")
    source_path = cache_path(learned, clip)
    if not source_path.is_file():
        raise FileNotFoundError(
            "V9.8 requires the retained V9.2 observation cache: " f"{source_path}"
        )
    if not config.v97_dense_cache_path.is_file():
        raise FileNotFoundError(
            "V9.8 requires the retained V9.7 dense control cache: "
            f"{config.v97_dense_cache_path}"
        )
    recovery = load_config(
        learned.recovery_config,
        overrides={
            "sam3_device": str(device),
            "frame_indices": clip.frame_indices,
            "scene_id": clip.scene_id,
            "manifest": learned.manifest,
        },
    )
    prompts = tuple(clip.instance_prompts) or (str(clip.instance_prompt),)
    if not prompts or any(not str(value).strip() for value in prompts):
        raise ValueError("V9.8 requires the configured dynamic-instance prompts.")
    provenance = {
        "source_cache": _file_provenance(source_path),
        "v97_dense_cache": _file_provenance(config.v97_dense_cache_path),
        "sam31_checkpoint": _file_provenance(recovery.sam3_checkpoint),
    }
    output = config.temporal_cache_path
    if output.is_file() and not rebuild:
        cached = _torch_load(output)
        if _cache_valid(cached, config=config, provenance=provenance, prompts=prompts):
            diagnostics = output.parent.parent / "v98_temporal_track_diagnostics.csv"
            diagnostics.parent.mkdir(parents=True, exist_ok=True)
            _write_track_diagnostics(
                diagnostics,
                cached["prompt_metadata"],
                tuple(int(value) for value in cached["frame_indices"]),
            )
            print("V9.8 reusing provenance-compatible temporal memory cache")
            return output
        print(f"V9.8 rebuilding stale temporal memory cache={output}")

    payload = load_feature_cache(source_path)
    frames = tuple(int(value) for value in payload["frame_indices"])
    if frames != config.expected_frames:
        raise ValueError("V9.8 source cache does not contain the locked 30 frames.")
    image_paths = [str(value) for value in payload["image_paths"]]
    image_size = tuple(int(value) for value in payload["image_size"])
    del payload

    tracker_max_objects = max(len(clip.instance_ids), 1)
    wrapper = SAM3Wrapper(
        repo_path=recovery.sam3_repo,
        checkpoint_path=recovery.sam3_checkpoint,
        device=str(device),
        output_threshold=recovery.sam3_output_threshold,
        prompt_with_box=recovery.prompt_with_box,
        version=recovery.sam3_version,
        use_fa3=recovery.sam3_use_fa3,
        max_num_objects=tracker_max_objects,
        multiplex_count=recovery.sam3_multiplex_count,
    ).load()
    if wrapper.predictor is None or wrapper.predictor.model is None:
        raise RuntimeError("V9.8 failed to load the SAM3.1 multiplex predictor.")
    memory_model = resolve_sam31_memory_model(wrapper.predictor.model)

    memory_runs = []
    raw_runs = []
    valid_runs = []
    delta_runs = []
    bucket_runs = []
    call_runs = []
    past_count_runs = []
    same_frame_runs = []
    mask_runs = []
    prompt_metadata = []
    per_prompt_limit = tracker_max_objects
    for prompt in prompts:
        with SAM31MemoryReadCapture(
            memory_model,
            grid_size=config.grid_size,
            canonical_dim=config.matcher.canonical_dim,
        ) as capture:
            tracked = wrapper.track_all_forward(
                image_paths,
                prompt=str(prompt),
                output_size=recovery.output_size,
                max_objects=per_prompt_limit,
            )
        captured = capture.finalized(len(frames))
        union = tracked.masks.any(dim=1, keepdim=True).float()
        grid_mask = F.interpolate(
            union,
            size=config.grid_size,
            mode="nearest",
        )[:, 0].bool()
        memory_runs.append(captured["memory"])
        raw_runs.append(captured["raw"])
        valid_runs.append(captured["valid"])
        delta_runs.append(captured["memory_delta_l2"])
        bucket_runs.append(captured["bucket_count"])
        call_runs.append(captured["call_count"])
        past_count_runs.append(captured["past_memory_frame_count"])
        same_frame_runs.append(captured["same_frame_memory_present"])
        mask_runs.append(grid_mask.cpu())
        prompt_metadata.append(
            {
                "prompt": str(prompt),
                "obj_ids": [int(value) for value in tracked.obj_ids],
                "birth_indices": [int(value) for value in tracked.birth_indices],
                "detected_object_count": int(
                    tracked.aux.get("detected_object_count", len(tracked.obj_ids))
                ),
                "retained_object_count": int(len(tracked.obj_ids)),
            }
        )
        print(
            "V9.8 causal prompt "
            f"{prompt!r} objects={len(tracked.obj_ids)} "
            f"captured_frames={int(captured['valid'].sum())}/{len(frames)}"
        )
        del capture

    memory_stack = torch.stack(memory_runs)
    raw_stack = torch.stack(raw_runs)
    valid_stack = torch.stack(valid_runs)
    del memory_runs, raw_runs, valid_runs
    memory, raw, valid = combine_prompt_feature_runs(
        memory_stack, raw_stack, valid_stack
    )
    # Frame zero is the text-conditioning/write frame and is not used by any
    # V9.8 train/test pair.  Every later frame must have actually read the
    # tracker path; silent zero filling would invalidate the experiment.
    if not bool(valid[1:].all()):
        missing = [frames[index] for index in torch.nonzero(~valid, as_tuple=False).flatten()]
        raise RuntimeError(f"V9.8 failed to capture temporal frames={missing}.")
    delta = torch.stack(delta_runs)
    past_count = torch.stack(past_count_runs)
    same_frame_stack = torch.stack(same_frame_runs)
    if bool((same_frame_stack & valid_stack)[:, 1:].any()):
        raise RuntimeError(
            "V9.8 could not isolate past-only memory from same-frame conditioning."
        )
    past_read_valid = (past_count > 0).any(dim=0)
    if not bool(past_read_valid[1:].all()):
        missing = [
            frames[index]
            for index in torch.nonzero(~past_read_valid, as_tuple=False).flatten()
            if index > 0
        ]
        raise RuntimeError(
            f"V9.8 frames did not read strictly earlier mask memory: {missing}."
        )
    if not bool((delta[:, 1:] > 1e-8).any()):
        raise RuntimeError(
            "V9.8 captured no numerical history effect; refusing to label raw "
            "propagation features as temporal memory."
        )
    raw_disagreement = _prompt_raw_disagreement(raw_stack, valid_stack)
    cached: dict[str, Any] = {
        "cache_version": TEMPORAL_CACHE_VERSION,
        "complete": True,
        "source_provenance": provenance,
        "frame_indices": list(frames),
        "image_size": list(image_size),
        "grid_size": list(config.grid_size),
        "prompts": list(prompts),
        "prompt_metadata": prompt_metadata,
        "memory_conditioned_features": memory.to(torch.float16),
        "memory_off_raw_features": raw.to(torch.float16),
        "memory_read_valid": valid,
        "per_prompt_memory_read_valid": valid_stack,
        "per_prompt_memory_delta_l2": delta,
        "per_prompt_bucket_count": torch.stack(bucket_runs),
        "per_prompt_capture_call_count": torch.stack(call_runs),
        "per_prompt_past_memory_frame_count": past_count,
        "per_prompt_same_frame_memory_present": same_frame_stack,
        "past_memory_read_valid": past_read_valid,
        "per_prompt_grid_masks": torch.stack(mask_runs),
        "raw_prompt_max_disagreement": raw_disagreement,
        "feature_source": "sam31_multiplex_prepare_memory_conditioned_features_return",
        "memory_off_source": "same_call_current_propagation_vision_feature_before_memory",
        "prompt_combination": "equal_mean_over_available_independent_forward_sessions",
        "max_objects_per_prompt": tracker_max_objects,
        "multiplex_semantics": "bucket_joint_feature_not_object_demux",
        "propagation_direction": "forward",
        "causal_confirmation": True,
        "hotstart_delay": 0,
        "uses_future_frames": False,
        "strictly_earlier_memory_observed": True,
        "same_frame_memory_excluded": True,
        "uses_pose_loss": False,
        "contains_pose_model": False,
        "image_paths_sha256": _string_digest(image_paths),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cached, output)
    diagnostics = output.parent.parent / "v98_temporal_track_diagnostics.csv"
    _write_track_diagnostics(diagnostics, prompt_metadata, frames)
    print(
        "V9.8 cached causal memory maps "
        f"frames={len(frames)} grid={config.grid_size[0]}x{config.grid_size[1]} "
        f"prompts={len(prompts)} raw_prompt_max_diff={raw_disagreement:.6g}"
    )
    print(f"V9.8 temporal track diagnostics={diagnostics}")
    del wrapper, cached, memory_stack, raw_stack, memory, raw
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output


def _cache_valid(value: Any, *, config, provenance, prompts) -> bool:
    if not isinstance(value, dict):
        return False
    required = {
        "cache_version",
        "complete",
        "source_provenance",
        "frame_indices",
        "grid_size",
        "prompts",
        "prompt_metadata",
        "memory_conditioned_features",
        "memory_off_raw_features",
        "memory_read_valid",
        "past_memory_read_valid",
        "feature_source",
        "memory_off_source",
        "propagation_direction",
        "causal_confirmation",
        "uses_future_frames",
        "strictly_earlier_memory_observed",
        "same_frame_memory_excluded",
    }
    if required - set(value):
        return False
    memory = torch.as_tensor(value["memory_conditioned_features"])
    raw = torch.as_tensor(value["memory_off_raw_features"])
    valid = torch.as_tensor(value["memory_read_valid"]).bool()
    past_valid = torch.as_tensor(value["past_memory_read_valid"]).bool()
    expected = (
        len(config.expected_frames),
        config.grid_size[0] * config.grid_size[1],
        config.matcher.canonical_dim,
    )
    return bool(
        int(value["cache_version"]) == TEMPORAL_CACHE_VERSION
        and value["complete"]
        and value["source_provenance"] == provenance
        and tuple(int(item) for item in value["frame_indices"]) == config.expected_frames
        and tuple(int(item) for item in value["grid_size"]) == config.grid_size
        and tuple(str(item) for item in value["prompts"]) == tuple(prompts)
        and tuple(memory.shape) == expected
        and tuple(raw.shape) == expected
        and tuple(valid.shape) == (len(config.expected_frames),)
        and tuple(past_valid.shape) == (len(config.expected_frames),)
        and bool(valid[1:].all())
        and bool(past_valid[1:].all())
        and bool(torch.isfinite(memory.float()).all())
        and bool(torch.isfinite(raw.float()).all())
        and str(value["propagation_direction"]) == "forward"
        and bool(value["causal_confirmation"])
        and not bool(value["uses_future_frames"])
        and bool(value["strictly_earlier_memory_observed"])
        and bool(value["same_frame_memory_excluded"])
    )


def _prompt_raw_disagreement(raw: torch.Tensor, valid: torch.Tensor) -> float:
    if raw.shape[0] < 2:
        return 0.0
    maximum = 0.0
    for left in range(raw.shape[0]):
        for right in range(left + 1, raw.shape[0]):
            shared = valid[left] & valid[right]
            if bool(shared.any()):
                maximum = max(
                    maximum,
                    float((raw[left, shared].float() - raw[right, shared].float()).abs().max()),
                )
    return maximum


def _write_track_diagnostics(path: Path, metadata, frames) -> None:
    columns = (
        "prompt", "source_obj_id", "birth_sequence_index", "birth_frame_index",
        "retained_objects_for_prompt", "detected_objects_for_prompt",
    )
    rows = []
    for prompt in metadata:
        obj_ids = list(prompt["obj_ids"])
        births = list(prompt["birth_indices"])
        if not obj_ids:
            obj_ids, births = [-1], [-1]
        for obj_id, birth in zip(obj_ids, births):
            rows.append({
                "prompt": prompt["prompt"],
                "source_obj_id": int(obj_id),
                "birth_sequence_index": int(birth),
                "birth_frame_index": int(frames[birth]) if int(birth) >= 0 else -1,
                "retained_objects_for_prompt": prompt["retained_object_count"],
                "detected_objects_for_prompt": prompt["detected_object_count"],
            })
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _file_provenance(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    stat = path.stat()
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _string_digest(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


if __name__ == "__main__":
    main()
