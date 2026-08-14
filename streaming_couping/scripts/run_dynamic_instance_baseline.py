#!/usr/bin/env python3
"""Build and audit V0 SAM tracking with clean QK-retrieved StreamVGGT pose."""

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


V0_IMPLEMENTATION_REVISION = "qk_retrieved_pose_semantic_tracking_r5"
QK_RETRIEVAL_REVISION = "v0_clean_streamvggt_qk_pose_retrieval_r1"

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
    pose_evaluation_indices = [
        index for index in range(len(frames)) if index != reference
    ]
    tracking_result = tracking_audit(payload, reference_index=reference)
    if not int(tracking_result["tracking_audit_pass"]):
        raise RuntimeError(f"V0 dynamic tracking audit failed: {tracking_result}")

    pose_selection = _load_pose_selection(
        run=run,
        payload=payload,
        raw_pose=raw_pose,
        frames=frames,
    )
    run.output_dir.mkdir(parents=True, exist_ok=True)
    _clear_stale_result_artifacts(run.output_dir)
    return _write_outputs(
        run=run,
        payload=payload,
        cache_path_value=path,
        signature=_signature(
            run=run,
            payload=payload,
            cache_path_value=path,
            pose_selection=pose_selection,
        ),
        raw_pose=raw_pose,
        target_pose=target_pose,
        reference=reference,
        evaluation_indices=evaluation_indices,
        pose_evaluation_indices=pose_evaluation_indices,
        tracking_audit=tracking_result,
        pose_selection=pose_selection,
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
    pose_evaluation_indices: list[int],
    tracking_audit: dict[str, object],
    pose_selection: dict[str, object],
) -> Path:
    frames = tuple(int(value) for value in payload["frame_indices"])
    selected_pose = pose_selection["selected_pose"]
    if not torch.is_tensor(selected_pose):
        raise TypeError("V0 pose selection did not return a tensor.")
    raw_metrics = pose_metrics(
        raw_pose,
        target_pose,
        reference_index=reference,
        evaluation_indices=pose_evaluation_indices,
    )
    selected_metrics = pose_metrics(
        selected_pose,
        target_pose,
        reference_index=reference,
        evaluation_indices=pose_evaluation_indices,
    )
    center_gain = _gain_percent(
        raw_metrics["center_error_native"],
        selected_metrics["center_error_native"],
    )
    rotation_gain = _gain_percent(
        raw_metrics["rotation_degrees"],
        selected_metrics["rotation_degrees"],
    )
    pose_improvement = bool(center_gain > 0.0 and rotation_gain > 0.0)
    window_results = _temporal_window_results(
        raw_pose=raw_pose,
        selected_pose=selected_pose,
        target_pose=target_pose,
        frame_indices=frames,
        reference_index=reference,
        evaluation_indices=evaluation_indices,
    )
    dynamic_rows = list(payload.get("dynamic_instance_diagnostics", ()))
    dynamic = {int(row["sequence_index"]): row for row in dynamic_rows}
    frame_rows = []
    for index, frame in enumerate(frames):
        raw_rotation, raw_center = _frame_errors(raw_pose, target_pose, index)
        selected_rotation, selected_center = _frame_errors(
            selected_pose,
            target_pose,
            index,
        )
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
                "selected_exact_raw": int(
                    torch.equal(selected_pose[:, index], raw_pose[:, index])
                ),
                "raw_rotation_error_deg": raw_rotation,
                "selected_rotation_error_deg": selected_rotation,
                "raw_center_error_native": raw_center,
                "selected_center_error_native": selected_center,
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
        "schema": 5,
        "baseline_version": run.version,
        "implementation_revision": V0_IMPLEMENTATION_REVISION,
        "method": "v0_sam_semantic_tracking_qk_retrieved_streamvggt_pose",
        "claim_level": (
            "single_sequence_training_free_qk_pose_improvement"
            if pose_improvement
            else "engineering_tracking_with_raw_pose_fallback"
        ),
        "config": str(run.source_path),
        "cache": str(cache_path_value),
        "signature": signature,
        "clip": payload["clip_name"],
        "frames": frames,
        "evaluation_frames": tuple(frames[index] for index in evaluation_indices),
        "pose_evaluation_frames": tuple(
            frames[index] for index in pose_evaluation_indices
        ),
        "pose_evaluation_role": "full_sequence_except_gauge_reference",
        "diagnostic_window_role": "not_train_test_folds",
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
        "pose_role": "training_free_streamvggt_native_qk_history_retrieval",
        "selected_pose_branch": pose_selection["selected_pose_branch"],
        "selected_pose_exact_raw": int(pose_selection["selected_pose_exact_raw"]),
        "pose_modification_applied": bool(
            not pose_selection["selected_pose_exact_raw"]
        ),
        "pose_improvement_claim": pose_improvement,
        "pose_claim_scope": "this_single_30_frame_sequence",
        "pose_candidate_status": pose_selection["status"],
        "pose_selection_fallback_used": int(pose_selection["fallback_used"]),
        "raw_pose_fallback_available": 1,
        "pose_candidate_provenance": pose_selection["provenance"],
        "historical_direct_pose_validation_pass": int(pose_improvement),
        "removed_pose_paths": (
            "v71_camera_pose_direct_se3",
            "v74_geometry_transport_pose_refiner",
        ),
        "future_pose_factor_backend": (
            "training_free_native_qk_retrieval_selected"
        ),
        "sam_pose_inputs": 0,
        "candidate_pointmap_used": False,
        "depth_source": "raw_streamvggt",
        "intrinsics_source": "raw_streamvggt",
        "raw_metrics": raw_metrics,
        "selected_pose_metrics": selected_metrics,
        "selected_center_gain_vs_raw_percent": center_gain,
        "selected_rotation_gain_vs_raw_percent": rotation_gain,
        "selected_gain_vs_raw_percent": center_gain,
        "diagnostic_window_results": window_results,
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
            "selected_world_to_camera": selected_pose.detach().cpu(),
            "selected_pose_branch": pose_selection["selected_pose_branch"],
            "selected_pose_exact_raw": bool(
                pose_selection["selected_pose_exact_raw"]
            ),
            "depth_source": "raw_streamvggt",
            "intrinsics_source": "raw_streamvggt",
            "candidate_pointmap_used": False,
        },
        run.output_dir / "poses.pt",
    )
    print("V0 SAM TRACKING + QK-RETRIEVED POSE BASELINE")
    print(
        f"  tracks={tracking_audit['discovered_track_count']} "
        f"late_births={tracking_audit['late_birth_count']} "
        f"tracking_audit={tracking_audit['tracking_audit_pass']}"
    )
    print(
        f"  selected_pose={pose_selection['selected_pose_branch']} "
        f"fallback={int(pose_selection['fallback_used'])} "
        f"center_gain={center_gain:.4f}% R_gain={rotation_gain:.4f}% "
        f"pose_improvement_claim={int(pose_improvement)}"
    )
    print(f"  output={result}")
    return result


def _temporal_window_results(
    *,
    raw_pose: torch.Tensor,
    selected_pose: torch.Tensor,
    target_pose: torch.Tensor,
    frame_indices: tuple[int, ...],
    reference_index: int,
    evaluation_indices: list[int],
) -> list[dict[str, object]]:
    if len(evaluation_indices) != 12:
        raise ValueError("V0 requires exactly twelve future evaluation frames.")
    output = []
    for window_index, window in enumerate(("short", "medium", "long")):
        indices = evaluation_indices[
            4 * window_index : 4 * (window_index + 1)
        ]
        raw_metrics = pose_metrics(
            raw_pose,
            target_pose,
            reference_index=reference_index,
            evaluation_indices=indices,
        )
        selected_metrics = pose_metrics(
            selected_pose,
            target_pose,
            reference_index=reference_index,
            evaluation_indices=indices,
        )
        output.append(
            {
                "window": window,
                "sequence_indices": tuple(indices),
                "frame_indices": tuple(frame_indices[index] for index in indices),
                "raw_metrics": raw_metrics,
                "selected_metrics": selected_metrics,
                "center_gain_vs_raw_percent": _gain_percent(
                    raw_metrics["center_error_native"],
                    selected_metrics["center_error_native"],
                ),
                "rotation_gain_vs_raw_percent": _gain_percent(
                    raw_metrics["rotation_degrees"],
                    selected_metrics["rotation_degrees"],
                ),
            }
        )
    return output


def _load_pose_selection(
    *,
    run: BaselineRunConfig,
    payload: dict,
    raw_pose: torch.Tensor,
    frames: tuple[int, ...],
) -> dict[str, object]:
    """Load a fixed RGB/QK candidate without using GT to choose a branch."""

    if run.selected_pose_branch == "raw_streamvggt":
        return _raw_pose_selection(raw_pose, status="raw_pose_selected_by_config")

    path = run.qk_pose_output
    summary_path = path.with_name("clean_qk_pose_summary.json")
    missing = [value for value in (path, summary_path) if not value.is_file()]
    if missing:
        if not run.allow_raw_pose_fallback:
            raise FileNotFoundError(
                f"Missing required clean QK pose artifacts: {missing}."
            )
        return _raw_pose_selection(
            raw_pose,
            status="raw_pose_fallback_missing_qk_artifact",
            fallback_used=True,
        )

    candidate = torch.load(path, map_location="cpu", weights_only=False)
    summary = json.loads(summary_path.read_text(encoding="utf8"))
    _validate_qk_pose_artifacts(
        candidate=candidate,
        summary=summary,
        payload=payload,
        raw_pose=raw_pose,
        frames=frames,
    )
    selected = candidate["selected_world_to_camera"].to(
        device=raw_pose.device,
        dtype=raw_pose.dtype,
    )
    raw_copy = candidate["raw_world_to_camera"].to(
        device=raw_pose.device,
        dtype=raw_pose.dtype,
    )
    raw_max_abs_diff = float((raw_copy - raw_pose).abs().max().detach().cpu())
    if not torch.allclose(raw_copy, raw_pose, atol=2e-5, rtol=1e-5):
        raise RuntimeError(
            "Clean QK artifact was generated from a different raw pose; "
            f"maximum absolute difference={raw_max_abs_diff}."
        )
    exact_raw = bool(torch.equal(selected, raw_pose))
    if exact_raw:
        raise RuntimeError("Clean QK selected pose is unexpectedly exact raw pose.")
    return {
        "selected_pose": selected,
        "selected_pose_branch": "retrieve_qk",
        "selected_pose_exact_raw": False,
        "fallback_used": False,
        "status": "selected_fixed_clean_qk_retrieval",
        "provenance": {
            "revision": QK_RETRIEVAL_REVISION,
            "pose_output": str(path),
            "pose_output_sha256": _sha256_file(path),
            "summary": str(summary_path),
            "summary_sha256": _sha256_file(summary_path),
            "candidate_generation_fields": summary[
                "candidate_generation_fields"
            ],
            "candidate_generation_gt_fields": 0,
            "sam_pose_inputs": 0,
            "model_trained": 0,
            "point_head_run": 0,
            "raw_pose_max_abs_difference": raw_max_abs_diff,
        },
    }


def _validate_qk_pose_artifacts(
    *,
    candidate: dict,
    summary: dict,
    payload: dict,
    raw_pose: torch.Tensor,
    frames: tuple[int, ...],
) -> None:
    expected_summary = {
        "revision": QK_RETRIEVAL_REVISION,
        "clip": payload["clip_name"],
        "method": "retrieve_qk",
        "candidate_generation_fields": ["stream_images", "frame_indices"],
        "candidate_generation_gt_fields": 0,
        "sam_pose_inputs": 0,
        "model_trained": 0,
        "point_head_run": 0,
    }
    for name, expected in expected_summary.items():
        if summary.get(name) != expected:
            raise ValueError(
                f"Clean QK summary field {name!r}={summary.get(name)!r}; "
                f"expected {expected!r}."
            )
    expected_candidate = {
        "revision": QK_RETRIEVAL_REVISION,
        "selected_pose_branch": "retrieve_qk",
    }
    for name, expected in expected_candidate.items():
        if candidate.get(name) != expected:
            raise ValueError(
                f"Clean QK pose field {name!r}={candidate.get(name)!r}; "
                f"expected {expected!r}."
            )
    if tuple(int(value) for value in summary.get("frames", ())) != frames:
        raise ValueError("Clean QK summary frame order differs from V0 cache.")
    if tuple(int(value) for value in candidate.get("frame_indices", ())) != frames:
        raise ValueError("Clean QK pose frame order differs from V0 cache.")
    for name in ("selected_world_to_camera", "raw_world_to_camera"):
        value = candidate.get(name)
        if not torch.is_tensor(value) or value.shape != raw_pose.shape:
            raise ValueError(
                f"Clean QK {name} must have shape {tuple(raw_pose.shape)}."
            )
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"Clean QK {name} contains non-finite values.")


def _raw_pose_selection(
    raw_pose: torch.Tensor,
    *,
    status: str,
    fallback_used: bool = False,
) -> dict[str, object]:
    return {
        "selected_pose": raw_pose.detach().clone(),
        "selected_pose_branch": "raw_streamvggt",
        "selected_pose_exact_raw": True,
        "fallback_used": bool(fallback_used),
        "status": status,
        "provenance": {
            "revision": None,
            "candidate_generation_fields": (),
            "candidate_generation_gt_fields": 0,
            "sam_pose_inputs": 0,
            "model_trained": 0,
            "point_head_run": 0,
        },
    }


def _gain_percent(raw: float, selected: float) -> float:
    if raw <= 0.0:
        raise ValueError("Raw pose metric must be positive.")
    return 100.0 * (float(raw) - float(selected)) / float(raw)


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
    pose_selection: dict[str, object],
) -> str:
    identity = {
        "schema": 5,
        "implementation_revision": V0_IMPLEMENTATION_REVISION,
        "purpose": "v0_sam_tracking_qk_retrieved_streamvggt_pose",
        "config": asdict(run),
        "cache": str(cache_path_value),
        "clip": payload.get("clip_name"),
        "frames": payload.get("frame_indices"),
        "sam_checkpoint": payload.get("sam_checkpoint"),
        "sam_track_ids": payload.get("sam_track_ids"),
        "sam_birth_indices": payload.get("sam_birth_indices"),
        "instance_birth_indices": payload.get("instance_birth_indices"),
        "selected_pose_branch": run.selected_pose_branch,
        "qk_pose_output": str(run.qk_pose_output),
        "pose_candidate_provenance": pose_selection["provenance"],
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, default=str).encode("utf8")
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _clear_stale_result_artifacts(output_dir: Path) -> None:
    """Remove outputs that could be mistaken for active V0 evidence."""

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
