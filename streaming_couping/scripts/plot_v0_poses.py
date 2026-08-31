#!/usr/bin/env python3
"""Plot the frozen V0 raw and QK-retrieved camera trajectories.

The V0 pose artifact stores world-to-camera matrices in the StreamVGGT native
gauge. This script converts them to camera centers, normalizes every branch
to the first frame, and optionally overlays the ScanNet++ ground-truth
trajectory in the same native scale. The output is diagnostic only: it does
not change either V0 pose artifact.

Typical usage on the experiment server::

    PYTHONPATH=. python -m streaming_couping.scripts.plot_v0_poses \
        --config streaming_couping/configs/v0_baseline.yaml

The output directory receives ``pose_trajectory.png``,
``pose_trajectory.svg``, ``pose_metrics.csv`` and
``pose_plot_summary.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from streaming_couping.src.learned_pose.baseline_runtime import (
    load_baseline_run_config,
)
from streaming_couping.src.learned_pose.config import (
    load_learned_pose_config,
)


REVISION = "v0_pose_trajectory_plot_r1"
RAW_COLOR = "#1f77b4"
SELECTED_COLOR = "#d95f02"
GT_COLOR = "#222222"
IMPROVED_COLOR = "#2ca25f"
WORSE_COLOR = "#d62728"


def main() -> None:
    args = _parse_args()
    inputs = _resolve_inputs(args)

    pose_payload = _load_torch_dict(inputs["poses"])
    frame_indices = _read_frame_indices(
        pose_payload,
        fallback=inputs.get("config_frame_indices"),
    )
    raw_w2c = _pose_tensor(
        pose_payload.get("raw_world_to_camera"),
        field="raw_world_to_camera",
    )
    selected_w2c = _pose_tensor(
        pose_payload.get("selected_world_to_camera"),
        field="selected_world_to_camera",
    )
    if raw_w2c.shape[0] != len(frame_indices):
        raise ValueError(
            "raw_world_to_camera length does not match frame_indices: "
            f"{raw_w2c.shape[0]} vs {len(frame_indices)}."
        )
    if selected_w2c.shape != raw_w2c.shape:
        raise ValueError(
            "raw and selected pose shapes differ: "
            f"{raw_w2c.shape} vs {selected_w2c.shape}."
        )

    reference_index = int(inputs.get("reference_index", 0))
    _validate_reference_index(reference_index, len(frame_indices))
    raw_w2c = _make_reference_relative(raw_w2c, reference_index)
    selected_w2c = _make_reference_relative(selected_w2c, reference_index)

    native_to_metric_scale = _resolve_native_to_metric_scale(
        cache_path=inputs.get("cache"),
        explicit_scale=args.native_to_metric_scale,
    )
    gt_w2c = None
    if not args.no_gt:
        manifest = inputs.get("manifest")
        scene_id = inputs.get("scene_id")
        if manifest is None or scene_id is None:
            raise ValueError(
                "GT plotting requires --manifest and --scene-id, or --config."
            )
        gt_absolute = _load_manifest_poses(
            manifest,
            scene_id=str(scene_id),
            frame_indices=frame_indices,
        )
        gt_w2c = _make_reference_relative(gt_absolute, reference_index)
        gt_w2c[:, :3, 3] /= native_to_metric_scale

    output_dir = Path(inputs["output_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_payload = _load_optional_json(inputs.get("summary"))

    result = _build_result(
        frame_indices=frame_indices,
        raw_w2c=raw_w2c,
        selected_w2c=selected_w2c,
        gt_w2c=gt_w2c,
        reference_index=reference_index,
        native_to_metric_scale=native_to_metric_scale,
        pose_payload=pose_payload,
        summary_payload=summary_payload,
    )
    _write_metrics_csv(output_dir / "pose_metrics.csv", result)
    _write_plot(output_dir, result)
    result_summary = _summary_for_json(
        result,
        poses_path=inputs["poses"],
        manifest_path=inputs.get("manifest"),
        scene_id=inputs.get("scene_id"),
        cache_path=inputs.get("cache"),
        summary_path=inputs.get("summary"),
        native_to_metric_scale=native_to_metric_scale,
    )
    summary_path = output_dir / "pose_plot_summary.json"
    summary_path.write_text(
        json.dumps(result_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf8",
    )

    print("V0 POSE TRAJECTORY PLOT")
    print(f"  frames={len(frame_indices)} reference_frame={frame_indices[reference_index]}")
    print(f"  selected_pose_branch={pose_payload.get('selected_pose_branch', 'unknown')}")
    print(f"  gt={'enabled' if gt_w2c is not None else 'disabled'}")
    if result["has_gt"]:
        print(
            "  mean_center_error_native="
            f"raw={result['raw_center_error_mean']:.6f} "
            f"selected={result['selected_center_error_mean']:.6f} "
            f"gain={result['center_gain_percent']:.4f}%"
        )
        print(
            "  mean_rotation_error_deg="
            f"raw={result['raw_rotation_error_mean']:.6f} "
            f"selected={result['selected_rotation_error_mean']:.6f} "
            f"gain={result['rotation_gain_percent']:.4f}%"
        )
    print(f"  plot={output_dir / 'pose_trajectory.png'}")
    print(f"  metrics={output_dir / 'pose_metrics.csv'}")
    print(f"  summary={summary_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot V0 raw/QK camera trajectories and, when available, GT "
            "pose errors."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "V0 YAML config. It supplies poses, manifest, scene, cache and "
            "output paths; explicit flags override it."
        ),
    )
    parser.add_argument("--poses", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--scene-id", default=None)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--native-to-metric-scale",
        type=float,
        default=None,
        help=(
            "Scale mapping StreamVGGT native coordinates to metric GT. "
            "Defaults to point_alignment_scale in the cache, then 1.0."
        ),
    )
    parser.add_argument(
        "--no-gt",
        action="store_true",
        help="Only draw raw/QK trajectories and their mutual differences.",
    )
    return parser.parse_args()


def _resolve_inputs(args: argparse.Namespace) -> dict[str, Any]:
    values: dict[str, Any] = {}
    if args.config is not None:
        config_path = Path(args.config).expanduser().resolve()
        learned = load_learned_pose_config(config_path)
        baseline = load_baseline_run_config(config_path)
        clip = next(
            (item for item in learned.clips if item.name == baseline.clip_name),
            None,
        )
        if clip is None:
            raise ValueError(
                f"Configured baseline clip {baseline.clip_name!r} is missing "
                f"from {config_path}."
            )
        values.update(
            {
                "poses": baseline.output_dir / "poses.pt",
                "manifest": learned.manifest,
                "scene_id": clip.scene_id,
                "cache": learned.features.cache_dir / f"{clip.name}.pt",
                "summary": baseline.output_dir / "baseline_summary.json",
                "output_dir": baseline.output_dir / "pose_plots",
                "config_frame_indices": clip.frame_indices,
                "reference_index": clip.reference_sequence_index,
            }
        )

    for name in ("poses", "manifest", "scene_id", "cache", "summary", "output_dir"):
        override = getattr(args, name, None)
        if override is not None:
            values[name] = override

    if values.get("poses") is None:
        raise ValueError("Provide --poses or --config.")
    values["poses"] = Path(values["poses"]).expanduser().resolve()
    if not values["poses"].is_file():
        raise FileNotFoundError(f"Pose artifact does not exist: {values['poses']}")

    for name in ("manifest", "cache", "summary"):
        if values.get(name) is not None:
            values[name] = Path(values[name]).expanduser().resolve()
    if values.get("output_dir") is None:
        values["output_dir"] = values["poses"].parent / "pose_plots"
    else:
        values["output_dir"] = Path(values["output_dir"]).expanduser().resolve()
    return values


def _load_torch_dict(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a dictionary pose artifact, got {type(payload)!r}.")
    return payload


def _pose_tensor(value: Any, *, field: str) -> np.ndarray:
    if value is None:
        raise ValueError(f"Pose artifact lacks required field {field!r}.")
    if torch.is_tensor(value):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    array = np.asarray(array, dtype=np.float64)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3 or array.shape[1:] not in ((3, 4), (4, 4)):
        raise ValueError(
            f"{field} must have shape (T,3,4), (T,4,4), (1,T,3,4) or "
            f"(1,T,4,4); got {array.shape}."
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{field} contains non-finite values.")
    output = np.zeros((array.shape[0], 4, 4), dtype=np.float64)
    output[:, 3, 3] = 1.0
    output[:, :3, :4] = array[:, :3, :4]
    return output


def _read_frame_indices(
    payload: dict[str, Any],
    *,
    fallback: Any,
) -> tuple[int, ...]:
    values = payload.get("frame_indices")
    if values is None:
        values = fallback
    if values is None:
        raise ValueError("Pose artifact lacks frame_indices; provide --config.")
    frames = tuple(int(value) for value in values)
    if not frames:
        raise ValueError("frame_indices is empty.")
    if len(set(frames)) != len(frames):
        raise ValueError("frame_indices contains duplicates.")
    return frames


def _validate_reference_index(reference_index: int, length: int) -> None:
    if reference_index < 0 or reference_index >= length:
        raise ValueError(
            f"reference_index={reference_index} is outside a sequence of length {length}."
        )


def _make_reference_relative(poses: np.ndarray, reference_index: int) -> np.ndarray:
    reference_c2w = np.linalg.inv(poses[reference_index])
    return poses @ reference_c2w


def _load_manifest_poses(
    manifest_path: Path,
    *,
    scene_id: str,
    frame_indices: tuple[int, ...],
) -> np.ndarray:
    with manifest_path.open("r", encoding="utf8") as handle:
        manifest = json.load(handle)
    scene = next(
        (item for item in manifest.get("scenes", []) if item.get("scene_id") == scene_id),
        None,
    )
    if scene is None:
        raise ValueError(f"Scene {scene_id!r} is missing from {manifest_path}.")
    frames = scene.get("frames", [])
    matrices = []
    for frame_index in frame_indices:
        if frame_index < 0 or frame_index >= len(frames):
            raise ValueError(
                f"Frame {frame_index} is outside scene {scene_id!r} with "
                f"{len(frames)} frames."
            )
        value = frames[frame_index].get("world_to_camera")
        if value is None:
            raise ValueError(
                f"Scene {scene_id!r} frame {frame_index} lacks world_to_camera."
            )
        matrices.append(
            _pose_tensor(
                np.asarray(value)[None, ...],
                field=f"GT frame {frame_index}",
            )[0]
        )
    return np.stack(matrices, axis=0)


def _resolve_native_to_metric_scale(
    *,
    cache_path: Path | None,
    explicit_scale: float | None,
) -> float:
    if explicit_scale is not None:
        scale = float(explicit_scale)
    elif cache_path is not None and cache_path.is_file():
        payload = _load_torch_dict(cache_path)
        scale = float(payload.get("point_alignment_scale", 1.0))
    else:
        scale = 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"native-to-metric scale must be positive, got {scale}.")
    return scale


def _camera_centers(w2c: np.ndarray) -> np.ndarray:
    c2w = np.linalg.inv(w2c)
    return c2w[:, :3, 3]


def _rotation_error_degrees(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    relative = predicted[:, :3, :3] @ np.swapaxes(target[:, :3, :3], 1, 2)
    cosine = (np.trace(relative, axis1=1, axis2=2) - 1.0) * 0.5
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _build_result(
    *,
    frame_indices: tuple[int, ...],
    raw_w2c: np.ndarray,
    selected_w2c: np.ndarray,
    gt_w2c: np.ndarray | None,
    reference_index: int,
    native_to_metric_scale: float,
    pose_payload: dict[str, Any],
    summary_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    raw_centers = _camera_centers(raw_w2c)
    selected_centers = _camera_centers(selected_w2c)
    result: dict[str, Any] = {
        "revision": REVISION,
        "frame_indices": np.asarray(frame_indices, dtype=np.int64),
        "sequence_indices": np.arange(len(frame_indices), dtype=np.int64),
        "raw_w2c": raw_w2c,
        "selected_w2c": selected_w2c,
        "raw_centers": raw_centers,
        "selected_centers": selected_centers,
        "qk_raw_center_delta": np.linalg.norm(
            selected_centers - raw_centers,
            axis=1,
        ),
        "qk_raw_rotation_delta": _rotation_error_degrees(selected_w2c, raw_w2c),
        "reference_index": reference_index,
        "native_to_metric_scale": native_to_metric_scale,
        "selected_pose_branch": str(pose_payload.get("selected_pose_branch", "unknown")),
        "has_gt": gt_w2c is not None,
        "summary_payload": summary_payload or {},
    }
    if gt_w2c is not None:
        gt_centers = _camera_centers(gt_w2c)
        raw_center_error = np.linalg.norm(raw_centers - gt_centers, axis=1)
        selected_center_error = np.linalg.norm(
            selected_centers - gt_centers,
            axis=1,
        )
        raw_rotation_error = _rotation_error_degrees(raw_w2c, gt_w2c)
        selected_rotation_error = _rotation_error_degrees(selected_w2c, gt_w2c)
        result.update(
            {
                "gt_w2c": gt_w2c,
                "gt_centers": gt_centers,
                "raw_center_error": raw_center_error,
                "selected_center_error": selected_center_error,
                "raw_rotation_error": raw_rotation_error,
                "selected_rotation_error": selected_rotation_error,
                "center_improvement": raw_center_error - selected_center_error,
                "rotation_improvement": raw_rotation_error - selected_rotation_error,
                "raw_center_error_mean": _evaluation_mean(
                    raw_center_error,
                    reference_index,
                ),
                "selected_center_error_mean": _evaluation_mean(
                    selected_center_error,
                    reference_index,
                ),
                "raw_rotation_error_mean": _evaluation_mean(
                    raw_rotation_error,
                    reference_index,
                ),
                "selected_rotation_error_mean": _evaluation_mean(
                    selected_rotation_error,
                    reference_index,
                ),
            }
        )
        result["center_gain_percent"] = _gain_percent(
            result["raw_center_error_mean"],
            result["selected_center_error_mean"],
        )
        result["rotation_gain_percent"] = _gain_percent(
            result["raw_rotation_error_mean"],
            result["selected_rotation_error_mean"],
        )
    return result


def _evaluation_mean(values: np.ndarray, reference_index: int) -> float:
    mask = np.ones(values.shape[0], dtype=bool)
    mask[reference_index] = False
    return float(values[mask].mean())


def _gain_percent(raw: float, selected: float) -> float:
    if raw <= 0.0:
        return 0.0
    return 100.0 * (float(raw) - float(selected)) / float(raw)


def _set_equal_3d_axes(ax, points: np.ndarray) -> None:
    points = np.asarray(points, dtype=np.float64)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = max(0.5 * float(np.max(maxs - mins)), 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1.0, 1.0, 1.0))


def _set_equal_2d_axes(ax, points: np.ndarray, dims: tuple[int, int]) -> None:
    values = points[:, list(dims)]
    mins = values.min(axis=0)
    maxs = values.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = max(0.5 * float(np.max(maxs - mins)), 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_aspect("equal", adjustable="box")


def _plot_path(ax, points: np.ndarray, *, label: str, color: str, linestyle: str):
    ax.plot(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        color=color,
        linewidth=2.0,
        linestyle=linestyle,
        label=label,
    )
    ax.scatter(points[0, 0], points[0, 1], points[0, 2], color=color, marker="o", s=38)
    ax.scatter(points[-1, 0], points[-1, 1], points[-1, 2], color=color, marker="^", s=45)


def _plot_path_2d(
    ax,
    points: np.ndarray,
    *,
    dims: tuple[int, int],
    label: str,
    color: str,
    linestyle: str,
) -> None:
    ax.plot(
        points[:, dims[0]],
        points[:, dims[1]],
        color=color,
        linewidth=2.0,
        linestyle=linestyle,
        label=label,
    )
    ax.scatter(points[0, dims[0]], points[0, dims[1]], color=color, marker="o", s=38)
    ax.scatter(points[-1, dims[0]], points[-1, dims[1]], color=color, marker="^", s=45)


def _write_plot(output_dir: Path, result: dict[str, Any]) -> None:
    raw = result["raw_centers"]
    selected = result["selected_centers"]
    paths = [raw, selected]
    labels = ["raw StreamVGGT", "QK-retrieved V0"]
    colors = [RAW_COLOR, SELECTED_COLOR]
    linestyles = ["--", "-"]
    if result["has_gt"]:
        paths.append(result["gt_centers"])
        labels.append("GT")
        colors.append(GT_COLOR)
        linestyles.append(":")
    all_points = np.concatenate(paths, axis=0)

    figure = plt.figure(figsize=(18, 10), dpi=150)
    ax3d = figure.add_subplot(2, 3, 1, projection="3d")
    ax_xy = figure.add_subplot(2, 3, 2)
    ax_xz = figure.add_subplot(2, 3, 3)
    ax_metric_1 = figure.add_subplot(2, 3, 4)
    ax_metric_2 = figure.add_subplot(2, 3, 5)
    ax_metric_3 = figure.add_subplot(2, 3, 6)

    for points, label, color, linestyle in zip(paths, labels, colors, linestyles):
        _plot_path(ax3d, points, label=label, color=color, linestyle=linestyle)
        _plot_path_2d(
            ax_xy,
            points,
            dims=(0, 1),
            label=label,
            color=color,
            linestyle=linestyle,
        )
        _plot_path_2d(
            ax_xz,
            points,
            dims=(0, 2),
            label=label,
            color=color,
            linestyle=linestyle,
        )

    _set_equal_3d_axes(ax3d, all_points)
    _set_equal_2d_axes(ax_xy, all_points, (0, 1))
    _set_equal_2d_axes(ax_xz, all_points, (0, 2))
    ax3d.set_title("3D camera trajectory")
    ax3d.set_xlabel("X (native units)")
    ax3d.set_ylabel("Y (native units)")
    ax3d.set_zlabel("Z (native units)")
    ax3d.view_init(elev=25, azim=-60)
    ax_xy.set_title("XY projection")
    ax_xy.set_xlabel("X (native units)")
    ax_xy.set_ylabel("Y (native units)")
    ax_xz.set_title("XZ projection")
    ax_xz.set_xlabel("X (native units)")
    ax_xz.set_ylabel("Z (native units)")
    for axis in (ax3d, ax_xy, ax_xz):
        axis.grid(True, alpha=0.35)
        axis.legend(loc="best", fontsize=8)

    frames = result["frame_indices"]
    if result["has_gt"]:
        ax_metric_1.plot(
            frames,
            result["raw_center_error"],
            color=RAW_COLOR,
            linestyle="--",
            linewidth=1.8,
            label="raw",
        )
        ax_metric_1.plot(
            frames,
            result["selected_center_error"],
            color=SELECTED_COLOR,
            linewidth=1.8,
            label="QK",
        )
        ax_metric_1.set_title("Camera-center error vs GT")
        ax_metric_1.set_ylabel("error (native units)")
        ax_metric_2.plot(
            frames,
            result["raw_rotation_error"],
            color=RAW_COLOR,
            linestyle="--",
            linewidth=1.8,
            label="raw",
        )
        ax_metric_2.plot(
            frames,
            result["selected_rotation_error"],
            color=SELECTED_COLOR,
            linewidth=1.8,
            label="QK",
        )
        ax_metric_2.set_title("Rotation error vs GT")
        ax_metric_2.set_ylabel("error (degrees)")
        ax_metric_3.plot(
            frames,
            result["center_improvement"],
            color=IMPROVED_COLOR,
            linewidth=1.8,
            label="raw − QK",
        )
        ax_metric_3.fill_between(
            frames,
            0.0,
            result["center_improvement"],
            where=result["center_improvement"] >= 0.0,
            color=IMPROVED_COLOR,
            alpha=0.20,
        )
        ax_metric_3.fill_between(
            frames,
            0.0,
            result["center_improvement"],
            where=result["center_improvement"] < 0.0,
            color=WORSE_COLOR,
            alpha=0.20,
        )
        ax_metric_3.axhline(0.0, color="#555555", linewidth=0.8)
        ax_metric_3.set_title("Per-frame center improvement")
        ax_metric_3.set_ylabel("raw error − QK error")
    else:
        ax_metric_1.plot(
            frames,
            result["qk_raw_center_delta"],
            color=SELECTED_COLOR,
            linewidth=1.8,
            label="QK − raw",
        )
        ax_metric_1.set_title("QK/raw camera-center difference")
        ax_metric_1.set_ylabel("distance (native units)")
        ax_metric_2.plot(
            frames,
            result["qk_raw_rotation_delta"],
            color=SELECTED_COLOR,
            linewidth=1.8,
            label="QK vs raw",
        )
        ax_metric_2.set_title("QK/raw rotation difference")
        ax_metric_2.set_ylabel("angle (degrees)")
        ax_metric_3.axis("off")
        ax_metric_3.text(
            0.05,
            0.80,
            "GT not loaded\n\nPass --manifest and --scene-id\n"
            "to add absolute error curves.",
            transform=ax_metric_3.transAxes,
            va="top",
        )

    for axis in (ax_metric_1, ax_metric_2, ax_metric_3):
        if axis.axison:
            axis.set_xlabel("dataset frame index")
            axis.grid(True, alpha=0.35)
            if axis is not ax_metric_3 or result["has_gt"]:
                axis.legend(loc="best", fontsize=8)

    title = "V0 pose trajectory: raw StreamVGGT vs QK retrieval"
    subtitle = (
        f"{len(frames)} frames | first-frame-relative native gauge | "
        f"native→metric scale={result['native_to_metric_scale']:.6g}"
    )
    if result["has_gt"]:
        subtitle += (
            f" | center gain={result['center_gain_percent']:.2f}%"
            f" | rotation gain={result['rotation_gain_percent']:.2f}%"
        )
    figure.suptitle(f"{title}\n{subtitle}", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    for suffix, kwargs in (("png", {"dpi": 180}), ("svg", {})):
        figure.savefig(
            output_dir / f"pose_trajectory.{suffix}",
            bbox_inches="tight",
            **kwargs,
        )
    plt.close(figure)


def _write_metrics_csv(path: Path, result: dict[str, Any]) -> None:
    frames = result["frame_indices"]
    rows = []
    for index, frame in enumerate(frames):
        row = {
            "sequence_index": int(index),
            "frame_index": int(frame),
            "raw_center_error_native": "",
            "selected_center_error_native": "",
            "center_improvement_native": "",
            "raw_rotation_error_deg": "",
            "selected_rotation_error_deg": "",
            "rotation_improvement_deg": "",
            "qk_raw_center_delta_native": float(result["qk_raw_center_delta"][index]),
            "qk_raw_rotation_delta_deg": float(result["qk_raw_rotation_delta"][index]),
        }
        if result["has_gt"]:
            row.update(
                {
                    "raw_center_error_native": float(result["raw_center_error"][index]),
                    "selected_center_error_native": float(
                        result["selected_center_error"][index]
                    ),
                    "center_improvement_native": float(
                        result["center_improvement"][index]
                    ),
                    "raw_rotation_error_deg": float(result["raw_rotation_error"][index]),
                    "selected_rotation_error_deg": float(
                        result["selected_rotation_error"][index]
                    ),
                    "rotation_improvement_deg": float(
                        result["rotation_improvement"][index]
                    ),
                }
            )
        rows.append(row)
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_optional_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    with path.open("r", encoding="utf8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _summary_for_json(
    result: dict[str, Any],
    *,
    poses_path: Path,
    manifest_path: Path | None,
    scene_id: str | None,
    cache_path: Path | None,
    summary_path: Path | None,
    native_to_metric_scale: float,
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "revision": REVISION,
        "poses": str(poses_path),
        "manifest": str(manifest_path) if manifest_path is not None else None,
        "scene_id": scene_id,
        "cache": str(cache_path) if cache_path is not None else None,
        "baseline_summary": str(summary_path) if summary_path is not None else None,
        "frame_count": int(len(result["frame_indices"])),
        "frame_indices": [int(value) for value in result["frame_indices"]],
        "reference_index": int(result["reference_index"]),
        "reference_frame": int(result["frame_indices"][result["reference_index"]]),
        "selected_pose_branch": result["selected_pose_branch"],
        "native_to_metric_scale": float(native_to_metric_scale),
        "has_gt": bool(result["has_gt"]),
    }
    if result["has_gt"]:
        mask = np.ones(len(result["frame_indices"]), dtype=bool)
        mask[result["reference_index"]] = False
        output.update(
            {
                "raw_center_error_mean_native": float(result["raw_center_error_mean"]),
                "selected_center_error_mean_native": float(
                    result["selected_center_error_mean"]
                ),
                "center_gain_percent": float(result["center_gain_percent"]),
                "raw_rotation_error_mean_deg": float(result["raw_rotation_error_mean"]),
                "selected_rotation_error_mean_deg": float(
                    result["selected_rotation_error_mean"]
                ),
                "rotation_gain_percent": float(result["rotation_gain_percent"]),
                "improved_center_frame_ratio": float(
                    np.mean(result["center_improvement"][mask] > 0.0)
                ),
                "improved_rotation_frame_ratio": float(
                    np.mean(result["rotation_improvement"][mask] > 0.0)
                ),
            }
        )
    return output


if __name__ == "__main__":
    main()
