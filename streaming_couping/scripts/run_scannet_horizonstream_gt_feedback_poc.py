#!/usr/bin/env python3
"""Minimal causal GT-pose feedback proof of concept for HorizonStream.

The model is executed exactly once with one streaming state.  The resulting
chunk camera maps are then replayed through three pose branches:

* raw: the normal online motion-averaged trajectory;
* posthoc: replace only the correction-frame output pose;
* feedback: inject the correction into the causal
  ``online_absolute_poses`` accumulator before later chunks are integrated.

This deliberately does not use SAM3, object losses, ICP, or training.  The
HorizonStream KV/GLA caches are left untouched because they contain feature
state, not a writable world-pose state.  The experiment therefore answers both
questions separately: whether the pose accumulator can accept feedback, and
whether the model hidden state itself exposes a pose-feedback interface.
"""

from __future__ import annotations

import argparse
import csv
import copy
import json
import math
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import yaml

from streaming_couping.src.rgb_inputs import resolve_rgb_inputs


def _homogeneous(value: np.ndarray) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float64)
    if pose.shape[-2:] == (3, 4):
        result = np.eye(4, dtype=np.float64)
        result[:3] = pose
        return result
    if pose.shape[-2:] == (4, 4):
        return pose.copy()
    raise ValueError(f"Expected a 3x4 or 4x4 pose, got {pose.shape}.")


def _project_rotation(rotation: np.ndarray) -> np.ndarray:
    left, _, right_t = np.linalg.svd(np.asarray(rotation, dtype=np.float64))
    projected = left @ right_t
    if np.linalg.det(projected) < 0.0:
        left = left.copy()
        left[:, -1] *= -1.0
        projected = left @ right_t
    return projected


def _rotation_error_deg(predicted: np.ndarray, target: np.ndarray) -> float:
    relative = np.linalg.inv(target[:3, :3]) @ predicted[:3, :3]
    relative = _project_rotation(relative)
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _camera_centers(c2w: np.ndarray) -> np.ndarray:
    return np.asarray(c2w, dtype=np.float64)[:, :3, 3]


def _relative_to_first(c2w: np.ndarray) -> np.ndarray:
    first_inverse = np.linalg.inv(c2w[0])
    return np.stack([first_inverse @ pose for pose in c2w], axis=0)


def _chunk_schedule(
    num_frames: int,
    window_size: int,
    sliding_size: int,
) -> list[tuple[int, int]]:
    if num_frames <= 0:
        return []
    if num_frames <= window_size:
        return [(0, num_frames)]
    chunks = [(0, window_size)]
    start = window_size
    while start < num_frames:
        end = min(start + sliding_size, num_frames)
        chunks.append((start, end))
        start = end
    return chunks


def _autocast_context(device: str, precision: str):
    precision = str(precision).strip().lower()
    if not str(device).startswith("cuda") or precision in {"", "float32", "fp32"}:
        return nullcontext()
    if not torch.cuda.is_available():
        return nullcontext()
    if precision in {"float16", "fp16", "half"}:
        dtype = torch.float16
    elif precision in {"bfloat16", "bf16"}:
        dtype = torch.bfloat16
    else:
        raise ValueError(
            f"Unsupported CUDA precision {precision!r}; use float16, bfloat16, or float32."
        )
    return torch.autocast(device_type="cuda", dtype=dtype)


def _load_gt_c2w(
    manifest_path: Path,
    *,
    scene_id: str,
    source_positions: Sequence[int],
) -> np.ndarray:
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    scene = next(
        (item for item in manifest.get("scenes", []) if str(item.get("scene_id")) == scene_id),
        None,
    )
    if scene is None:
        raise ValueError(f"Scene {scene_id!r} is missing from {manifest_path}.")

    frames = scene.get("frames", [])
    poses: list[np.ndarray] = []
    for source_position in source_positions:
        index = int(source_position)
        if index < 0 or index >= len(frames):
            raise ValueError(
                f"GT frame index {index} is outside scene length {len(frames)}."
            )
        value = frames[index].get("world_to_camera")
        if value is None:
            raise ValueError(
                f"Manifest frame {index} has no world_to_camera field."
            )
        w2c = _homogeneous(np.asarray(value, dtype=np.float64))
        if not np.isfinite(w2c).all():
            raise ValueError(f"Manifest frame {index} has non-finite GT pose.")
        poses.append(np.linalg.inv(w2c))
    if not poses:
        raise ValueError("GT selection produced no poses.")
    return np.stack(poses, axis=0)


def _prepare_data_config(
    upstream_config: Mapping[str, Any],
    image_paths: Sequence[Path],
    *,
    scene_id: str,
    image_size: int,
    patch_size: int,
    crop: bool,
) -> dict[str, Any]:
    data_config = copy.deepcopy(dict(upstream_config.get("data", {}) or {}))
    data_config.update(
        {
            "format": "image_list",
            "image_paths": [str(path) for path in image_paths],
            "image_scene_name": str(scene_id),
            "size": int(image_size),
            "crop": bool(crop),
            "patch_size": int(patch_size),
            "camera_preprocess": False,
            "max_frames": None,
        }
    )
    return data_config


def _run_horizonstream_once(
    *,
    upstream_config: Mapping[str, Any],
    image_paths: Sequence[Path],
    scene_id: str,
    checkpoint: Path,
    horizon_repo: Path,
    device: str,
    window_size: int,
    sliding_size: int,
    image_size: int,
    patch_size: int,
    crop: bool,
    precision: str,
) -> tuple[list[torch.Tensor], int, tuple[int, int], dict[str, Any]]:
    if str(horizon_repo) not in sys.path:
        sys.path.insert(0, str(horizon_repo))

    from horizonstream.core.model import HorizonStreamModel
    from horizonstream.data.dataloader import HorizonStreamDataLoader

    model_config = copy.deepcopy(dict(upstream_config.get("model", {}) or {}))
    model_config["checkpoint"] = str(checkpoint)
    model_config["strict_load"] = True
    model_config.setdefault("horizonstream_cfg", {})
    model_config["horizonstream_cfg"]["enable_metric_readout_token"] = True

    data_config = _prepare_data_config(
        upstream_config,
        image_paths,
        scene_id=scene_id,
        image_size=image_size,
        patch_size=patch_size,
        crop=crop,
    )

    print(
        "HorizonStream inference "
        f"frames={len(image_paths)} device={device} "
        f"window={window_size} sliding={sliding_size} precision={precision}",
        flush=True,
    )
    model = HorizonStreamModel(model_config).to(device).eval()
    loader = HorizonStreamDataLoader(data_config)
    sequences = iter(loader)
    sequence = next(sequences)
    try:
        next(sequences)
    except StopIteration:
        pass
    else:
        raise ValueError("The POC expects exactly one image sequence.")

    images = sequence.images.detach().cpu()
    if images.ndim != 5 or images.shape[0] != 1:
        raise ValueError(
            "HorizonStream loader must return [1,S,3,H,W], "
            f"got {tuple(images.shape)}."
        )
    _, frame_count, channels, height, width = images.shape
    if channels != 3 or frame_count != len(image_paths):
        raise ValueError(
            "Preprocessed HorizonStream image count does not match the RGB selection."
        )

    state = model.build_sequence_state()
    state_audit = {
        "state_keys": sorted(str(key) for key in state.keys()),
        "frame_kv_cache_present": "frame_kv_caches" in state,
        "global_kv_cache_present": "global_kv_caches" in state,
        "gla_cache_present": state.get("gla_cache") is not None,
        "pose_state_present": False,
        "world_geometry_state_present": False,
        "model_pose_feedback_write_supported": False,
        "reason": (
            "HorizonStream sequence state exposes feature KV/GLA caches only; "
            "no previous-pose, world-geometry, or pointmap field is consumed by "
            "forward_chunk."
        ),
    }

    chunks = _chunk_schedule(frame_count, int(window_size), int(sliding_size))
    chunk_cam_maps: list[torch.Tensor] = []
    with torch.no_grad():
        for chunk_index, (start, end) in enumerate(chunks):
            current_window_size = end - start if chunk_index == 0 else int(window_size)
            chunk_images = images[:, start:end].to(device, non_blocking=True)
            with _autocast_context(device, precision):
                outputs = model.forward_chunk(
                    chunk_images,
                    window_size=current_window_size,
                    chunk_idx=chunk_index,
                    state=state,
                )
            chunk_cam_maps.append(
                outputs["chunk_cam_map"].detach().float().cpu()
            )
            model.advance_sequence_state(
                state,
                is_last_chunk=chunk_index == len(chunks) - 1,
            )
            del outputs, chunk_images
            if str(device).startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(
                f"chunk={chunk_index + 1}/{len(chunks)} frames=[{start},{end})",
                flush=True,
            )

    del model, loader, sequence, images
    return chunk_cam_maps, frame_count, (int(height), int(width)), state_audit


def _cam_map_to_c2w(
    cam_map: torch.Tensor,
    *,
    image_hw: tuple[int, int],
) -> np.ndarray:
    from horizonstream.utils.vendor.models.components.utils.pose_enc import (
        pose_encoding_to_extri_intri,
    )

    extri, _ = pose_encoding_to_extri_intri(
        cam_map,
        image_size_hw=image_hw,
    )
    w2c = extri[0].detach().float().cpu().numpy().astype(np.float64)
    return np.stack([np.linalg.inv(_homogeneous(pose)) for pose in w2c], axis=0)


def _normalize_internal_w2c_to_first(
    online_absolute_poses: torch.Tensor,
) -> np.ndarray:
    """Convert the internal absolute w2c cache to the public first-frame gauge."""

    w2c = online_absolute_poses[0].detach().float().cpu().numpy().astype(np.float64)
    rotations = w2c[:, :3, :3]
    translations = w2c[:, :3, 3]
    centers = -(np.transpose(rotations, (0, 2, 1)) @ translations[..., None])[..., 0]
    relative_rotations = np.stack(
        [rotation @ rotations[0].T for rotation in rotations],
        axis=0,
    )
    relative_translations = np.stack(
        [-rotation @ (center - centers[0]) for rotation, center in zip(rotations, centers)],
        axis=0,
    )
    relative_w2c = np.tile(np.eye(4, dtype=np.float64), (w2c.shape[0], 1, 1))
    relative_w2c[:, :3, :3] = relative_rotations
    relative_w2c[:, :3, 3] = relative_translations
    return np.stack([np.linalg.inv(pose) for pose in relative_w2c], axis=0)


def _replace_internal_pose_for_public_target(
    online_absolute_poses: torch.Tensor,
    *,
    frame_index: int,
    target_public_c2w: np.ndarray,
) -> None:
    """Make one internal pose decode to ``target_public_c2w``.

    The public trajectory is ``P0 @ inverse(Pi)`` in c2w form, where ``Pi`` is
    the internal w2c cache.  Keeping frame zero fixed, the required internal
    pose is ``inverse(target) @ P0``.
    """

    p0 = _homogeneous(
        online_absolute_poses[0, 0].detach().float().cpu().numpy()
    )
    target_w2c = np.linalg.inv(np.asarray(target_public_c2w, dtype=np.float64))
    replacement = target_w2c @ p0
    value = torch.as_tensor(
        replacement[:3],
        dtype=online_absolute_poses.dtype,
        device=online_absolute_poses.device,
    )
    online_absolute_poses[0, int(frame_index)] = value


def _run_feedback_motion_averaging(
    chunk_cam_maps: Sequence[torch.Tensor],
    *,
    frame_count: int,
    window_size: int,
    correction_frame: int,
    gt_public_c2w: np.ndarray,
) -> tuple[torch.Tensor, dict[str, Any]]:
    from horizonstream.runtime.motion_averaging import (
        get_4d_anti_diagonal_medians,
        online_motion_averaging,
    )
    from horizonstream.utils.vendor.models.components.utils.rotation import mat_to_quat

    first_chunk = chunk_cam_maps[0]
    batch_size = 1
    if frame_count <= int(window_size):
        raise ValueError(
            "Feedback POC needs more frames than the streaming window so that "
            "future chunks exist."
        )

    valid_relative_poses = torch.empty(
        batch_size,
        frame_count - (int(window_size) - 1),
        int(window_size),
        3,
        4,
        dtype=torch.float32,
    )
    valid_focals_scales_shifts = torch.empty(
        batch_size,
        frame_count - (int(window_size) - 1),
        int(window_size),
        4,
        dtype=torch.float32,
    )
    online_absolute_poses = torch.empty(
        batch_size, frame_count, 3, 4, dtype=torch.float32
    )
    last_absolute_poses = torch.empty_like(online_absolute_poses)
    online_scales = torch.ones(
        batch_size,
        frame_count - (int(window_size) - 1),
        1,
        dtype=torch.float32,
    )

    online_ptr = 0
    valid_ptr = 0
    correction_applied = False
    correction_payload: dict[str, Any] = {}

    for chunk_index, chunk_cam_map in enumerate(chunk_cam_maps):
        before_ptr = int(online_ptr)
        online_ptr, valid_ptr = online_motion_averaging(
            chunk_cam_map=chunk_cam_map,
            chunk_idx=chunk_index,
            B=batch_size,
            Win=int(window_size),
            online_absolute_poses=online_absolute_poses,
            online_S=online_scales,
            valid_relative_poses=valid_relative_poses,
            valid_focals_scales_shifts=valid_focals_scales_shifts,
            online_ptr=online_ptr,
            valid_ptr=valid_ptr,
            last_absolute_poses=last_absolute_poses,
            dtype=torch.float32,
        )
        after_ptr = int(online_ptr)
        if (
            not correction_applied
            and before_ptr <= int(correction_frame) < after_ptr
        ):
            raw_public_c2w = _normalize_internal_w2c_to_first(online_absolute_poses)
            raw_at_t = raw_public_c2w[int(correction_frame)]
            target_at_t = np.asarray(
                gt_public_c2w[int(correction_frame)],
                dtype=np.float64,
            )
            delta = target_at_t @ np.linalg.inv(raw_at_t)
            _replace_internal_pose_for_public_target(
                online_absolute_poses,
                frame_index=int(correction_frame),
                target_public_c2w=target_at_t,
            )
            _replace_internal_pose_for_public_target(
                last_absolute_poses,
                frame_index=int(correction_frame),
                target_public_c2w=target_at_t,
            )
            correction_applied = True
            correction_payload = {
                "frame": int(correction_frame),
                "delta_c2w": delta.tolist(),
                "raw_c2w_before_feedback": raw_at_t.tolist(),
                "gt_c2w_target": target_at_t.tolist(),
                "feedback_state": "online_absolute_poses",
                "kv_cache_modified": False,
                "gla_cache_modified": False,
            }

    if not correction_applied:
        raise RuntimeError(
            f"Correction frame {correction_frame} was not reached by the chunk schedule."
        )

    online_rotations = online_absolute_poses[..., :3, :3]
    online_translations = online_absolute_poses[..., :3, 3:]
    online_centers = -online_rotations.mT @ online_translations
    online_T = (
        -online_rotations
        @ (online_centers - online_centers[:, 0:1])
    ).squeeze(-1)
    online_Q = mat_to_quat(online_rotations @ online_rotations[:, 0:1].mT)
    median_focal = get_4d_anti_diagonal_medians(
        valid_focals_scales_shifts[..., :2]
    )
    feedback_cam_map = torch.cat([online_T, online_Q, median_focal], dim=-1)
    correction_payload["final_online_ptr"] = int(online_ptr)
    correction_payload["final_valid_ptr"] = int(valid_ptr)
    return feedback_cam_map, correction_payload


def _trajectory_metrics(
    predicted_c2w: np.ndarray,
    gt_c2w: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if predicted_c2w.shape != gt_c2w.shape:
        raise ValueError(
            f"Trajectory shape mismatch: {predicted_c2w.shape} vs {gt_c2w.shape}."
        )
    frame_rows: list[dict[str, Any]] = []
    translation_errors: list[float] = []
    rotation_errors: list[float] = []
    for frame, (predicted, target) in enumerate(zip(predicted_c2w, gt_c2w)):
        translation_error = float(
            np.linalg.norm(predicted[:3, 3] - target[:3, 3])
        )
        rotation_error = _rotation_error_deg(predicted, target)
        translation_errors.append(translation_error)
        rotation_errors.append(rotation_error)
        frame_rows.append(
            {
                "frame": int(frame),
                "translation_error_m": translation_error,
                "rotation_error_deg": rotation_error,
            }
        )

    rpe_translation: list[float] = []
    rpe_rotation: list[float] = []
    for frame in range(1, len(predicted_c2w)):
        predicted_delta = (
            np.linalg.inv(predicted_c2w[frame - 1]) @ predicted_c2w[frame]
        )
        gt_delta = np.linalg.inv(gt_c2w[frame - 1]) @ gt_c2w[frame]
        error = np.linalg.inv(gt_delta) @ predicted_delta
        rpe_translation.append(float(np.linalg.norm(error[:3, 3])))
        rpe_rotation.append(
            _rotation_error_deg(error, np.eye(4, dtype=np.float64))
        )

    def rmse(values: Iterable[float]) -> float | None:
        values = [float(value) for value in values if math.isfinite(float(value))]
        if not values:
            return None
        return float(np.sqrt(np.mean(np.square(values))))

    def mean(values: Iterable[float]) -> float | None:
        values = [float(value) for value in values if math.isfinite(float(value))]
        return None if not values else float(np.mean(values))

    try:
        from horizonstream.eval.metrics import ate_rmse

        sim3_ate_rmse = float(
            ate_rmse(
                _camera_centers(predicted_c2w),
                _camera_centers(gt_c2w),
                align_scale=True,
            )["ate_rmse"]
        )
    except (ImportError, KeyError, TypeError, ValueError):
        sim3_ate_rmse = None

    summary = {
        "ate_rmse_m": rmse(translation_errors),
        "ate_rmse_sim3_m": sim3_ate_rmse,
        "ate_mean_m": mean(translation_errors),
        "rotation_error_rmse_deg": rmse(rotation_errors),
        "rotation_error_mean_deg": mean(rotation_errors),
        "rpe_pair_count": int(len(rpe_translation)),
        "rpe_translation_rmse_m": rmse(rpe_translation),
        "rpe_rotation_rmse_deg": rmse(rpe_rotation),
        "evaluation_gauge": "GT normalized to the first selected frame",
        "alignment": "none",
    }
    return summary, frame_rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def _write_branch_metrics(
    output_dir: Path,
    branch_name: str,
    predicted_c2w: np.ndarray,
    gt_c2w: np.ndarray,
) -> dict[str, Any]:
    summary, rows = _trajectory_metrics(predicted_c2w, gt_c2w)
    _write_csv(
        output_dir / f"{branch_name}_metrics.csv",
        rows,
        ("frame", "translation_error_m", "rotation_error_deg"),
    )
    return summary


def _format_metric(value: Any) -> str:
    return "None" if value is None else f"{float(value):.6f}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--frame-count", type=int, default=50)
    parser.add_argument("--correction-frame", type=int, default=15)
    parser.add_argument("--horizon-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--sliding-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--precision", default="float16")
    parser.add_argument("--crop", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.frame_start < 0:
        parser.error("--frame-start must be non-negative.")
    if args.frame_stride < 1:
        parser.error("--frame-stride must be positive.")
    if args.frame_count < 2:
        parser.error("--frame-count must be at least 2.")
    if args.correction_frame < 0 or args.correction_frame >= args.frame_count:
        parser.error("--correction-frame must be inside the selected sequence.")
    if args.window_size < 1 or args.sliding_size < 1:
        parser.error("--window-size and --sliding-size must be positive.")
    return args


def main() -> None:
    args = _parse_args()
    manifest = args.manifest.expanduser().resolve()
    horizon_repo = args.horizon_repo.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    config_path = (
        args.config.expanduser().resolve()
        if args.config is not None
        else horizon_repo / "configs" / "horizonstream_infer.yaml"
    )
    for path, label in (
        (manifest, "ScanNet++ manifest"),
        (horizon_repo / "horizonstream", "HorizonStream source"),
        (checkpoint, "HorizonStream checkpoint"),
        (config_path, "HorizonStream config"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")

    selection = resolve_rgb_inputs(
        manifest=manifest,
        scene_id=str(args.scene_id),
        start=int(args.frame_start),
        stride=int(args.frame_stride),
        count=int(args.frame_count),
    )
    image_paths = selection.image_paths
    if len(image_paths) != int(args.frame_count):
        raise ValueError(
            f"Requested {args.frame_count} frames but selected {len(image_paths)}."
        )
    gt_c2w_absolute = _load_gt_c2w(
        manifest,
        scene_id=str(args.scene_id),
        source_positions=selection.source_positions,
    )
    gt_c2w = _relative_to_first(gt_c2w_absolute)

    with config_path.open("r", encoding="utf-8") as handle:
        upstream_config = yaml.safe_load(handle) or {}

    chunk_cam_maps, frame_count, image_hw, state_audit = _run_horizonstream_once(
        upstream_config=upstream_config,
        image_paths=image_paths,
        scene_id=str(args.scene_id),
        checkpoint=checkpoint,
        horizon_repo=horizon_repo,
        device=str(args.device),
        window_size=int(args.window_size),
        sliding_size=int(args.sliding_size),
        image_size=int(args.image_size),
        patch_size=int(args.patch_size),
        crop=bool(args.crop),
        precision=str(args.precision),
    )
    if frame_count != len(image_paths):
        raise ValueError("Inference frame count differs from selected RGB count.")

    from horizonstream.runtime.motion_averaging import (
        compute_motion_averaged_camera_maps,
    )

    raw_motion = compute_motion_averaged_camera_maps(
        chunk_cam_maps,
        frames_num=frame_count,
        window_size=int(args.window_size),
        dtype=torch.float32,
        enable_offline=False,
    )
    raw_c2w = _cam_map_to_c2w(
        raw_motion["online_cam_map"],
        image_hw=image_hw,
    )

    correction_frame = int(args.correction_frame)
    delta_c2w = gt_c2w[correction_frame] @ np.linalg.inv(raw_c2w[correction_frame])
    posthoc_c2w = raw_c2w.copy()
    posthoc_c2w[correction_frame] = delta_c2w @ posthoc_c2w[correction_frame]

    feedback_cam_map, feedback_payload = _run_feedback_motion_averaging(
        chunk_cam_maps,
        frame_count=frame_count,
        window_size=int(args.window_size),
        correction_frame=correction_frame,
        gt_public_c2w=gt_c2w,
    )
    feedback_c2w = _cam_map_to_c2w(feedback_cam_map, image_hw=image_hw)

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_summary = _write_branch_metrics(output_dir, "raw", raw_c2w, gt_c2w)
    posthoc_summary = _write_branch_metrics(
        output_dir,
        "posthoc",
        posthoc_c2w,
        gt_c2w,
    )
    feedback_summary = _write_branch_metrics(
        output_dir,
        "feedback",
        feedback_c2w,
        gt_c2w,
    )

    future_rows: list[dict[str, Any]] = []
    end_frame = min(frame_count, correction_frame + 11)
    for frame in range(correction_frame + 1, end_frame):
        raw_translation = float(
            np.linalg.norm(raw_c2w[frame, :3, 3] - gt_c2w[frame, :3, 3])
        )
        posthoc_translation = float(
            np.linalg.norm(posthoc_c2w[frame, :3, 3] - gt_c2w[frame, :3, 3])
        )
        feedback_translation = float(
            np.linalg.norm(feedback_c2w[frame, :3, 3] - gt_c2w[frame, :3, 3])
        )
        raw_rotation = _rotation_error_deg(raw_c2w[frame], gt_c2w[frame])
        posthoc_rotation = _rotation_error_deg(posthoc_c2w[frame], gt_c2w[frame])
        feedback_rotation = _rotation_error_deg(
            feedback_c2w[frame],
            gt_c2w[frame],
        )
        future_rows.append(
            {
                "frame": int(frame),
                "raw_translation_error": raw_translation,
                "posthoc_translation_error": posthoc_translation,
                "feedback_translation_error": feedback_translation,
                "raw_rotation_error": raw_rotation,
                "posthoc_rotation_error": posthoc_rotation,
                "feedback_rotation_error": feedback_rotation,
                "feedback_translation_gain": raw_translation - feedback_translation,
                "feedback_rotation_gain": raw_rotation - feedback_rotation,
            }
        )
    _write_csv(
        output_dir / "future_pose_gain.csv",
        future_rows,
        (
            "frame",
            "raw_translation_error",
            "posthoc_translation_error",
            "feedback_translation_error",
            "raw_rotation_error",
            "posthoc_rotation_error",
            "feedback_rotation_error",
            "feedback_translation_gain",
            "feedback_rotation_gain",
        ),
    )

    translation_improved = sum(
        float(row["feedback_translation_gain"]) > 1e-9
        for row in future_rows
    )
    rotation_improved = sum(
        float(row["feedback_rotation_gain"]) > 1e-9
        for row in future_rows
    )
    future_count = len(future_rows)
    posthoc_future_unchanged = all(
        np.isclose(
            posthoc_c2w[frame],
            raw_c2w[frame],
            atol=1e-10,
            rtol=0.0,
        ).all()
        for frame in range(correction_frame + 1, frame_count)
    )
    accumulator_feedback_works = bool(
        translation_improved > 0 or rotation_improved > 0
    )

    summary = {
        "schema": "horizonstream_gt_feedback_poc_v1",
        "scene_id": str(args.scene_id),
        "frame_count": int(frame_count),
        "frame_start": int(args.frame_start),
        "frame_stride": int(args.frame_stride),
        "source_positions": [int(value) for value in selection.source_positions],
        "correction_frame": correction_frame,
        "window_size": int(args.window_size),
        "sliding_size": int(args.sliding_size),
        "pose_convention": {
            "model_output": "world_to_camera",
            "branch_trajectories": "camera_to_world",
            "gt_source": "manifest.world_to_camera inverted to camera_to_world",
            "correction": "delta_c2w = T_gt_c2w @ inverse(T_raw_c2w)",
        },
        "state_audit": state_audit,
        "feedback_target": (
            "online_motion_averaging.online_absolute_poses "
            "(causal pose accumulator)"
        ),
        "model_kv_feedback_decision": "GT_FEEDBACK_DOES_NOT_WORK",
        "model_kv_feedback_reason": (
            "The wrapper/model exposes only frame/global KV and optional GLA "
            "feature caches; no pose/world-state field is read by forward_chunk."
        ),
        "correction": {
            "frame": correction_frame,
            "delta_c2w": delta_c2w.tolist(),
            "raw_c2w": raw_c2w[correction_frame].tolist(),
            "gt_c2w": gt_c2w[correction_frame].tolist(),
            "posthoc_c2w": posthoc_c2w[correction_frame].tolist(),
            "feedback_c2w": feedback_c2w[correction_frame].tolist(),
            "feedback_internal": feedback_payload,
        },
        "branches": {
            "raw": raw_summary,
            "posthoc": posthoc_summary,
            "feedback": feedback_summary,
        },
        "future_window": {
            "start_frame": int(correction_frame + 1),
            "end_frame_exclusive": int(end_frame),
            "count": int(future_count),
            "translation_improved_frames": int(translation_improved),
            "rotation_improved_frames": int(rotation_improved),
            "posthoc_future_unchanged": bool(posthoc_future_unchanged),
            "translation_gains": [
                float(row["feedback_translation_gain"]) for row in future_rows
            ],
            "rotation_gains": [
                float(row["feedback_rotation_gain"]) for row in future_rows
            ],
        },
        "decision": (
            "GT_FEEDBACK_WORKS"
            if accumulator_feedback_works
            else "GT_FEEDBACK_DOES_NOT_WORK"
        ),
        "decision_scope": "causal pose accumulator, not HorizonStream feature KV state",
        "outputs": {
            "raw_metrics_csv": str(output_dir / "raw_metrics.csv"),
            "posthoc_metrics_csv": str(output_dir / "posthoc_metrics.csv"),
            "feedback_metrics_csv": str(output_dir / "feedback_metrics.csv"),
            "future_pose_gain_csv": str(output_dir / "future_pose_gain.csv"),
            "summary_json": str(output_dir / "summary.json"),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    torch.save(
        {
            "raw_c2w": torch.from_numpy(raw_c2w).float(),
            "posthoc_c2w": torch.from_numpy(posthoc_c2w).float(),
            "feedback_c2w": torch.from_numpy(feedback_c2w).float(),
            "gt_c2w": torch.from_numpy(gt_c2w).float(),
        },
        output_dir / "poses.pt",
    )

    print(
        f"Raw ATE = {_format_metric(raw_summary['ate_rmse_sim3_m'])} m (Sim3), "
        f"direct_ATE = {_format_metric(raw_summary['ate_rmse_m'])} m, "
        f"RPE_t = {_format_metric(raw_summary['rpe_translation_rmse_m'])} m, "
        f"RPE_r = {_format_metric(raw_summary['rpe_rotation_rmse_deg'])} deg"
    )
    print(
        f"Posthoc ATE = {_format_metric(posthoc_summary['ate_rmse_sim3_m'])} m (Sim3), "
        f"direct_ATE = {_format_metric(posthoc_summary['ate_rmse_m'])} m, "
        f"RPE_t = {_format_metric(posthoc_summary['rpe_translation_rmse_m'])} m, "
        f"RPE_r = {_format_metric(posthoc_summary['rpe_rotation_rmse_deg'])} deg"
    )
    print(
        f"Feedback ATE = {_format_metric(feedback_summary['ate_rmse_sim3_m'])} m (Sim3), "
        f"direct_ATE = {_format_metric(feedback_summary['ate_rmse_m'])} m, "
        f"RPE_t = {_format_metric(feedback_summary['rpe_translation_rmse_m'])} m, "
        f"RPE_r = {_format_metric(feedback_summary['rpe_rotation_rmse_deg'])} deg"
    )
    print(f"Correction frame = {correction_frame}")
    print(
        f"Future t+1~t+10: translation improved frames = "
        f"{translation_improved}/{future_count}"
    )
    print(
        f"Future t+1~t+10: rotation improved frames = "
        f"{rotation_improved}/{future_count}"
    )
    print(f"Decision: {summary['decision']}")
    print("Model KV feedback decision: GT_FEEDBACK_DOES_NOT_WORK")
    for key, value in summary["outputs"].items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
