#!/usr/bin/env python3
"""Build and audit V0 dynamic SAM tracking with unmodified StreamVGGT pose."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import torch

from streaming_couping.src.config import load_config
from streaming_couping.src.external_repos import maybe_add_repo_to_path
from streaming_couping.src.learned_pose.baseline_runtime import (
    BaselineRunConfig,
    camera_centers,
    decode_cached_poses,
    load_baseline_run_config,
    pose_metrics,
    tracking_audit,
)
from streaming_couping.src.learned_pose.cache import (
    build_feature_caches,
    cache_path,
    load_feature_cache,
)
from streaming_couping.src.learned_pose.config import (
    ClipConfig,
    LearnedPoseConfig,
    load_learned_pose_config,
)


V0_IMPLEMENTATION_REVISION = "raw_streamvggt_dynamic_tracking_r4"

FRAME_COLUMNS = (
    "sequence_index",
    "frame_index",
    "sam_tracks_discovered",
    "observed_instances",
    "mature_instances",
    "identity_valid_instances",
    "associated_instances",
    "birth_slots",
    "birth_prompts",
    "geometry_birth_slots",
    "geometry_birth_prompts",
    "selected_exact_raw",
    "raw_rotation_error_deg",
    "selected_rotation_error_deg",
    "raw_center_error_native",
    "selected_center_error_native",
)


def main() -> None:
    args = _parse_args()
    data = load_learned_pose_config(args.config)
    run = load_baseline_run_config(args.config)
    if args.output_dir:
        run = replace(
            run,
            output_dir=Path(args.output_dir).expanduser().resolve(),
        )
    if args.audit_device:
        run = replace(run, audit_device=str(args.audit_device))
    if args.sam3_device or args.geometry_device or args.streamvggt_devices:
        data = replace(
            data,
            sam3_device=args.sam3_device or data.sam3_device,
            geometry_device=args.geometry_device or data.geometry_device,
            streamvggt_devices=(
                tuple(
                    value.strip()
                    for value in args.streamvggt_devices.split(",")
                    if value.strip()
                )
                if args.streamvggt_devices
                else data.streamvggt_devices
            ),
        )
    if args.rebuild_cache:
        data = replace(data, features=replace(data.features, rebuild=True))
    if args.stage in {"all", "cache"}:
        build_feature_caches(data)
    if args.stage in {"all", "audit"}:
        result = run_baseline(data, run)
        print(f"dynamic-tracking baseline result={result}")


def run_baseline(data: LearnedPoseConfig, run: BaselineRunConfig) -> Path:
    clip = _find_clip(data, run.clip_name)
    path = cache_path(data, clip)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing baseline cache {path}; run --stage cache or --stage all."
        )
    payload = load_feature_cache(path)
    _validate_payload(payload, clip=clip)
    recovery = load_config(data.recovery_config)
    maybe_add_repo_to_path(recovery.streamvggt_repo)
    from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri

    raw_pose, target_pose = decode_cached_poses(
        payload,
        pose_decoder=pose_encoding_to_extri_intri,
        device=run.audit_device,
    )
    frames = tuple(int(value) for value in payload["frame_indices"])
    positions = {frame: index for index, frame in enumerate(frames)}
    missing = set(run.evaluation_frames) - set(frames)
    if missing:
        raise ValueError(f"Baseline cache lacks evaluation frames={sorted(missing)}.")
    reference = int(payload["reference_sequence_index"])
    evaluation_indices = [positions[frame] for frame in run.evaluation_frames]
    tracking_result = tracking_audit(payload, reference_index=reference)
    if not int(tracking_result["tracking_audit_pass"]):
        raise RuntimeError(f"V0 dynamic tracking audit failed: {tracking_result}")

    run.output_dir.mkdir(parents=True, exist_ok=True)
    _clear_stale_result_artifacts(run.output_dir)
    return _write_outputs(
        run=run,
        payload=payload,
        cache_path_value=path,
        signature=_signature(run=run, payload=payload, cache_path_value=path),
        raw_pose=raw_pose,
        target_pose=target_pose,
        reference=reference,
        evaluation_indices=evaluation_indices,
        tracking_audit=tracking_result,
    )


def _write_outputs(
    *,
    run: BaselineRunConfig,
    payload: dict,
    cache_path_value: Path,
    signature: str,
    raw_pose: torch.Tensor,
    target_pose: torch.Tensor,
    reference: int,
    evaluation_indices: list[int],
    tracking_audit: dict[str, object],
) -> Path:
    frames = tuple(int(value) for value in payload["frame_indices"])
    selected_pose = raw_pose.detach().clone()
    if not torch.equal(selected_pose, raw_pose):
        raise RuntimeError("V0 r4 selected pose must be exactly raw StreamVGGT.")
    raw_metrics = pose_metrics(
        raw_pose,
        target_pose,
        reference_index=reference,
        evaluation_indices=evaluation_indices,
    )
    fold_results = _raw_temporal_fold_results(
        raw_pose=raw_pose,
        target_pose=target_pose,
        frame_indices=frames,
        reference_index=reference,
        evaluation_indices=evaluation_indices,
    )
    dynamic_rows = list(payload.get("dynamic_instance_diagnostics", ()))
    dynamic = {int(row["sequence_index"]): row for row in dynamic_rows}
    frame_rows = []
    for index, frame in enumerate(frames):
        rotation, center = _frame_errors(raw_pose, target_pose, index)
        diag = dynamic[index]
        frame_rows.append(
            {
                "sequence_index": index,
                "frame_index": frame,
                "sam_tracks_discovered": int(diag["discovered_tracks"]),
                "observed_instances": int(diag["observed_tracks"]),
                "mature_instances": int(diag["mature_tracks"]),
                "identity_valid_instances": int(diag["identity_valid_tracks"]),
                "associated_instances": int(diag["associated_tracks"]),
                "birth_slots": str(diag.get("birth_slots", "")),
                "birth_prompts": str(diag.get("birth_prompts", "")),
                "geometry_birth_slots": str(
                    diag.get("geometry_birth_slots", "")
                ),
                "geometry_birth_prompts": str(
                    diag.get("geometry_birth_prompts", "")
                ),
                "selected_exact_raw": 1,
                "raw_rotation_error_deg": rotation,
                "selected_rotation_error_deg": rotation,
                "raw_center_error_native": center,
                "selected_center_error_native": center,
            }
        )
    _write_csv(run.output_dir / "frame_diagnostics.csv", frame_rows, FRAME_COLUMNS)
    if dynamic_rows:
        _write_csv(
            run.output_dir / "dynamic_instance_diagnostics.csv",
            dynamic_rows,
            tuple(dynamic_rows[0]),
        )

    correction_rows = [
        row
        for row in payload.get("segmentation_diagnostics", ())
        if str(row.get("selected_variant", ""))
        == "sam31_online_geometry_compete"
    ]
    summary = {
        "schema": 4,
        "baseline_version": run.version,
        "implementation_revision": V0_IMPLEMENTATION_REVISION,
        "method": "v0_dynamic_sam_tracking_raw_streamvggt_pose",
        "claim_level": "engineering_tracking_baseline_raw_pose_reference",
        "config": str(run.source_path),
        "cache": str(cache_path_value),
        "signature": signature,
        "clip": payload["clip_name"],
        "frames": frames,
        "evaluation_frames": tuple(frames[index] for index in evaluation_indices),
        "sam_role": "prompted_multi_instance_discovery_mask_persistent_id",
        "configured_instance_prompts": tuple(
            str(value) for value in payload.get("instance_prompts", ())
        ),
        "configured_permanent_slot_capacity": len(payload["instance_ids"]),
        "geometry_guidance_role": "causal_mask_prompt_and_competition_only",
        "geometry_guidance_applied_frames": sum(
            int(row.get("correction_applied", 0)) for row in correction_rows
        ),
        "geometry_correction_memory_writeback": False,
        "sam_appearance_cached": bool(payload.get("cache_sam_appearance", True)),
        "tracking_audit": tracking_audit,
        "tracking_baseline_acceptance_pass": int(
            tracking_audit["tracking_audit_pass"]
        ),
        "pose_role": "raw_streamvggt_unmodified",
        "selected_pose_branch": "raw_streamvggt",
        "selected_pose_exact_raw": 1,
        "pose_modification_applied": False,
        "pose_improvement_claim": False,
        "pose_candidate_status": "sam_memory_retrieval_implemented_not_selected",
        "historical_direct_pose_validation_pass": 0,
        "removed_pose_paths": (
            "v71_camera_pose_direct_se3",
            "v74_geometry_transport_pose_refiner",
        ),
        "future_pose_factor_backend": (
            "retrievevggt_style_full_frame_kv_candidate_not_selected"
        ),
        "raw_metrics": raw_metrics,
        "selected_pose_metrics": raw_metrics,
        "selected_gain_vs_raw_percent": 0.0,
        "temporal_fold_results": fold_results,
    }
    result = run.output_dir / "baseline_summary.json"
    result.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf8",
    )
    torch.save(
        {
            "frame_indices": frames,
            "raw_world_to_camera": raw_pose.detach().cpu(),
            "selected_world_to_camera": raw_pose.detach().cpu(),
            "selected_pose_branch": "raw_streamvggt",
            "selected_pose_exact_raw": True,
        },
        run.output_dir / "poses.pt",
    )
    print("V0 RAW-POSE DYNAMIC TRACKING BASELINE")
    print(
        f"  tracks={tracking_audit['discovered_track_count']} "
        f"late_births={tracking_audit['late_birth_count']} "
        f"tracking_audit={tracking_audit['tracking_audit_pass']}"
    )
    print(
        "  selected_pose=raw_streamvggt exact_raw=1 "
        "pose_improvement_claim=0"
    )
    print(f"  output={result}")
    return result


def _raw_temporal_fold_results(
    *,
    raw_pose: torch.Tensor,
    target_pose: torch.Tensor,
    frame_indices: tuple[int, ...],
    reference_index: int,
    evaluation_indices: list[int],
) -> list[dict[str, object]]:
    if len(evaluation_indices) != 12:
        raise ValueError("V0 requires exactly twelve future evaluation frames.")
    output = []
    for fold_index, fold in enumerate(("short", "medium", "long")):
        indices = evaluation_indices[4 * fold_index : 4 * (fold_index + 1)]
        metrics = pose_metrics(
            raw_pose,
            target_pose,
            reference_index=reference_index,
            evaluation_indices=indices,
        )
        output.append(
            {
                "fold": fold,
                "sequence_indices": tuple(indices),
                "frame_indices": tuple(frame_indices[index] for index in indices),
                "raw_metrics": metrics,
                "selected_metrics": metrics,
                "selected_exact_raw": 1,
                "pose_improvement_claim": 0,
            }
        )
    return output


def _frame_errors(
    predicted: torch.Tensor,
    target: torch.Tensor,
    index: int,
) -> tuple[float, float]:
    left = predicted[:, index]
    right = target[:, index]
    relative = left[..., :3, :3] @ right[..., :3, :3].transpose(-1, -2)
    cosine = (
        torch.diagonal(relative, dim1=-2, dim2=-1).sum(dim=-1) - 1.0
    ) * 0.5
    rotation = torch.rad2deg(torch.acos(cosine.clamp(-1, 1))).mean()
    center = torch.linalg.vector_norm(
        camera_centers(left) - camera_centers(right),
        dim=-1,
    ).mean()
    return float(rotation.cpu()), float(center.cpu())


def _signature(
    *,
    run: BaselineRunConfig,
    payload: dict,
    cache_path_value: Path,
) -> str:
    identity = {
        "schema": 4,
        "implementation_revision": V0_IMPLEMENTATION_REVISION,
        "purpose": "v0_dynamic_sam_tracking_raw_streamvggt_pose",
        "config": asdict(run),
        "cache": str(cache_path_value),
        "clip": payload.get("clip_name"),
        "frames": payload.get("frame_indices"),
        "sam_checkpoint": payload.get("sam_checkpoint"),
        "sam_track_ids": payload.get("sam_track_ids"),
        "sam_birth_indices": payload.get("sam_birth_indices"),
        "instance_birth_indices": payload.get("instance_birth_indices"),
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, default=str).encode("utf8")
    ).hexdigest()


def _clear_stale_result_artifacts(output_dir: Path) -> None:
    """Remove outputs that could be mistaken for active V0 r4 evidence."""

    for name in (
        "baseline_summary.json",
        "frame_diagnostics.csv",
        "dynamic_instance_diagnostics.csv",
        "poses.pt",
        # Retired V0 r3 learned-pose checkpoints.
        "camera_baseline.pt",
        "geometry_pose_refiner.pt",
        # Retired V0 r3 parameter-matched control outputs.
        "validation/v0_pose_control_validation.csv",
        "validation/v0_pose_control_decision.json",
    ):
        path = output_dir / name
        if path.is_file():
            path.unlink()


def _validate_payload(payload: dict, *, clip: ClipConfig) -> None:
    if str(payload.get("instance_source")) != "sam31_online":
        raise ValueError("Baseline requires instance_source=sam31_online.")
    if str(payload.get("sam_version")) != "sam3.1":
        raise ValueError("Baseline cache is not SAM3.1.")
    if bool(payload.get("cache_sam_appearance", True)):
        raise ValueError("V0 must disable SAM appearance extraction.")
    if payload.get("appearance") is not None:
        raise ValueError("V0 cache unexpectedly contains SAM appearance.")
    if str(payload.get("sam_segmentation_variant")) != (
        "sam31_online_geometry_compete"
    ):
        raise ValueError("Baseline requires causal online geometry mask competition.")
    if tuple(int(value) for value in payload.get("instance_ids", ())) != (
        clip.instance_ids
    ):
        raise ValueError("Baseline cache slot layout differs from config.")
    expected_prompts = tuple(clip.instance_prompts) or (clip.instance_prompt,)
    cached_prompts = tuple(
        str(value) for value in payload.get("instance_prompts", ())
    )
    if cached_prompts != expected_prompts:
        raise ValueError("Baseline cache prompts differ from config; rebuild it.")
    if not payload.get("dynamic_instance_diagnostics"):
        raise ValueError("Baseline cache lacks dynamic instance diagnostics.")
    if not any(int(value) > 0 for value in payload.get("sam_birth_indices", ())):
        raise ValueError("This audit requires at least one late-born SAM track.")


def _find_clip(config: LearnedPoseConfig, name: str) -> ClipConfig:
    selected = [clip for clip in config.clips if clip.name == name]
    if len(selected) != 1:
        raise ValueError(f"Clip {name!r} was not found exactly once.")
    return selected[0]


def _write_csv(
    path: Path,
    rows: list[dict],
    columns: tuple[str, ...],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in columns})


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="streaming_couping/configs/v0_baseline.yaml",
    )
    parser.add_argument(
        "--stage",
        choices=("all", "cache", "audit"),
        default="all",
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--sam3-device")
    parser.add_argument("--geometry-device")
    parser.add_argument("--streamvggt-devices")
    parser.add_argument("--audit-device")
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
