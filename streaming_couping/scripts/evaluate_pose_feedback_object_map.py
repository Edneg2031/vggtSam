#!/usr/bin/env python3
"""Does the pose improvement reach the object map?  (CPU only)

The feedback experiment has only ever been scored on trajectory metrics: ATE,
RPE, future gains.  Whether a corrected trajectory produces a *better object
point cloud* -- the actual product -- was never measured, because the mapper
runs on the raw poses and the corrected trajectories were never mapped.

This closes that gap without re-running any model.  ``feedback_diagnostics.pt``
holds one camera-space point cloud per (frame, instance); ``poses.pt`` holds the
raw trajectory and every variant's corrected one.  Placing the SAME points with
a different trajectory is exactly the ablation wanted: identity errors, mask
quality, sampling, and configuration all cancel, and the only thing that changes
is the pose that puts the points in the world.

The reference cloud is built from the ground-truth instance masks, the depth,
and the GROUND-TRUTH poses -- not the raw ones.  Using raw poses for the target
would make the metric circular: the raw trajectory is the drifted one, so a
correction moving away from it would read as moving away from truth.

Objects are pooled per category rather than matched instance-to-instance.  The
comparison is between trajectories over an identical point set, so an assignment
error cancels on both sides, and category pooling avoids re-implementing the
evaluator's SAM-to-GT assignment.

    python -m streaming_couping.scripts.evaluate_pose_feedback_object_map \
        --run-dir <base>.baseline \
        --geometry-cache <base>/horizonstream_geometry.pt \
        --manifest <manifest.json> --scene-id 00a231a370 \
        --prompts bed wardrobe chair rug dustbin
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from streaming_couping.src.horizonstream_cache import load_horizonstream_cache
from streaming_couping.src.semantic_map_metrics import object_point_metrics
from streaming_couping.src.semantic_tracking_metrics import (
    load_ground_truth_instances,
)

#: Match the pipeline's object-only evaluation so the numbers are comparable.
FSCORE_THRESHOLD_M = 0.05
VOXEL_SIZE_M = 0.05
GHOST_DISTANCE_M = 0.05
CHUNK_SIZE = 4096

METRIC_KEYS = (
    "object_accuracy_m",
    "object_completeness_m",
    "fscore_5cm",
    "voxel_iou_5cm",
    "ghost_point_ratio",
)


def _load_poses(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    trajectories: dict[str, torch.Tensor] = {
        "raw": torch.as_tensor(payload["raw_c2w"]).detach().float().cpu()
    }
    for name, value in (payload.get("variant_c2w") or {}).items():
        trajectories[str(name)] = torch.as_tensor(value).detach().float().cpu()
    return trajectories


def _camera_points(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    pixels: torch.Tensor,
) -> torch.Tensor:
    """Backproject ``pixels`` (row, col) through depth and intrinsics."""

    rows = pixels[:, 0].float()
    columns = pixels[:, 1].float()
    z = depth[pixels[:, 0], pixels[:, 1]].float()
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    if abs(fx) <= 1e-8 or abs(fy) <= 1e-8:
        raise ValueError("intrinsics carry a zero focal length")
    return torch.stack(((columns - cx) / fx * z, (rows - cy) / fy * z, z), dim=-1)


def _transform(points: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
    return (
        points @ pose[:3, :3].transpose(0, 1) + pose[:3, 3]
    )


def reference_clouds(
    *,
    masks: torch.Tensor,
    labels: Sequence[str],
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    gt_c2w: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """GT object cloud per category, placed by the GROUND-TRUTH poses."""

    clouds: dict[str, list[torch.Tensor]] = {}
    for index, label in enumerate(labels):
        collected: list[torch.Tensor] = []
        for frame in range(int(masks.shape[0])):
            mask = masks[frame, index]
            z = depth[frame]
            valid = mask & torch.isfinite(z) & (z > 0.0)
            pixels = valid.nonzero(as_tuple=False)
            if pixels.numel() == 0:
                continue
            camera = _camera_points(z, intrinsics[frame], pixels)
            collected.append(_transform(camera, gt_c2w[frame]))
        if collected:
            clouds.setdefault(str(label), []).append(torch.cat(collected, dim=0))
    return {label: torch.cat(parts, dim=0) for label, parts in clouds.items()}


def predicted_clouds(
    *,
    observations: Sequence[Mapping[str, Any]],
    trajectory: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """SAM observations placed by ``trajectory``, pooled per category."""

    clouds: dict[str, list[torch.Tensor]] = {}
    for row in observations:
        frame_id = int(row["frame_id"])
        if not 0 <= frame_id < int(trajectory.shape[0]):
            raise ValueError(f"observation frame {frame_id} is outside the trajectory")
        points = torch.as_tensor(row["points_camera"]).detach().float().cpu()
        if points.numel() == 0:
            continue
        clouds.setdefault(str(row["category"]), []).append(
            _transform(points, trajectory[frame_id])
        )
    return {label: torch.cat(parts, dim=0) for label, parts in clouds.items()}


def score_trajectory(
    *,
    predicted: Mapping[str, torch.Tensor],
    reference: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Per-category metrics plus a pooled 'all' row."""

    per_category: dict[str, dict[str, float]] = {}
    pooled_predicted: list[torch.Tensor] = []
    pooled_reference: list[torch.Tensor] = []
    for label, target in sorted(reference.items()):
        source = predicted.get(label)
        if source is None or not source.numel():
            per_category[label] = {
                key: float("nan") for key in METRIC_KEYS
            }
            per_category[label]["skipped"] = "no_prediction_for_category"
            continue
        raw = object_point_metrics(
            source,
            target,
            fscore_thresholds=(FSCORE_THRESHOLD_M,),
            voxel_size=VOXEL_SIZE_M,
            ghost_distance=GHOST_DISTANCE_M,
            chunk_size=CHUNK_SIZE,
        )
        per_category[label] = {
            "object_accuracy_m": float(raw["object_accuracy_m"]),
            "object_completeness_m": float(raw["object_completeness_m"]),
            "fscore_5cm": float(raw["fscore_5cm"]),
            "voxel_iou_5cm": float(raw["voxel_iou"]),
            "ghost_point_ratio": float(raw["ghost_point_ratio"]),
            "predicted_points": int(source.shape[0]),
            "reference_points": int(target.shape[0]),
        }
        pooled_predicted.append(source)
        pooled_reference.append(target)
    if pooled_predicted:
        raw = object_point_metrics(
            torch.cat(pooled_predicted, dim=0),
            torch.cat(pooled_reference, dim=0),
            fscore_thresholds=(FSCORE_THRESHOLD_M,),
            voxel_size=VOXEL_SIZE_M,
            ghost_distance=GHOST_DISTANCE_M,
            chunk_size=CHUNK_SIZE,
        )
        per_category["all"] = {
            "object_accuracy_m": float(raw["object_accuracy_m"]),
            "object_completeness_m": float(raw["object_completeness_m"]),
            "fscore_5cm": float(raw["fscore_5cm"]),
            "voxel_iou_5cm": float(raw["voxel_iou"]),
            "ghost_point_ratio": float(raw["ghost_point_ratio"]),
            "predicted_points": int(sum(p.shape[0] for p in pooled_predicted)),
            "reference_points": int(sum(p.shape[0] for p in pooled_reference)),
        }
    return per_category


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return "nan"
    return f"{float(value):.{digits}f}"


def render(
    results: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> str:
    """One block per trajectory, raw first, so the deltas are readable."""

    lines: list[str] = []
    trajectories = list(results)
    categories = sorted(
        {label for block in results.values() for label in block},
        key=lambda label: (label != "all", label),
    )
    for category in categories:
        header = f"  {'trajectory':32s}" + "".join(
            f"{key[:16]:>18s}" for key in METRIC_KEYS
        )
        lines.append(f"[{category}]")
        lines.append(header)
        baseline = results.get("raw", {}).get(category, {})
        for name in trajectories:
            row = results[name].get(category, {})
            rendered = "".join(
                f"{_fmt(row.get(key)):>18s}" for key in METRIC_KEYS
            )
            lines.append(f"  {name:32s}{rendered}")
        if "raw" in results:
            for name in trajectories:
                if name == "raw":
                    continue
                row = results[name].get(category, {})
                delta = "".join(
                    f"{_fmt(_delta(row.get(key), baseline.get(key))):>18s}"
                    for key in METRIC_KEYS
                )
                lines.append(f"  {'  Δ vs raw ' + name:32s}{delta}")
        lines.append("")
    return "\n".join(lines)


def _delta(value: Any, baseline: Any) -> float | None:
    if value is None or baseline is None:
        return None
    try:
        left, right = float(value), float(baseline)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(left) and math.isfinite(right)):
        return None
    return left - right


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--geometry-cache", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--prompts", nargs="+", required=True)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    feedback = run_dir / "object_pose_feedback"
    poses_path = feedback / "poses.pt"
    diagnostics_path = run_dir / "object_pose_refinement" / "feedback_diagnostics.pt"
    for required in (poses_path, diagnostics_path):
        if not required.is_file():
            raise FileNotFoundError(f"missing: {required}")

    cache = load_horizonstream_cache(args.geometry_cache.expanduser().resolve())
    depth = torch.as_tensor(cache["depth"]).detach().float().cpu()
    intrinsics = torch.as_tensor(cache["intrinsics"]).detach().float().cpu()
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    height, width = int(depth.shape[1]), int(depth.shape[2])
    source_positions = [int(value) for value in cache["source_positions"]]

    diagnostics = torch.load(diagnostics_path, map_location="cpu", weights_only=False)
    observations = list(diagnostics.get("observations", ()))
    poses = _load_poses(poses_path)
    gt_c2w = torch.as_tensor(
        torch.load(poses_path, map_location="cpu", weights_only=False)["gt_c2w"]
    ).detach().float().cpu()
    print(
        f"frames={int(depth.shape[0])} size=({height},{width}) "
        f"observations={len(observations)} trajectories={list(poses)}"
    )

    ground_truth = load_ground_truth_instances(
        args.manifest.expanduser().resolve(),
        scene_id=str(args.scene_id),
        frame_indices=source_positions,
        output_size=(height, width),
        prompts=tuple(args.prompts),
    )
    print(
        f"gt instances={len(ground_truth.instance_ids)} "
        f"labels={sorted(set(ground_truth.labels))}"
    )
    reference = reference_clouds(
        masks=ground_truth.masks,
        labels=ground_truth.labels,
        depth=depth,
        intrinsics=intrinsics,
        gt_c2w=gt_c2w,
    )
    print(
        "reference clouds: "
        + ", ".join(f"{label}={int(cloud.shape[0])}" for label, cloud in sorted(reference.items()))
    )

    results: dict[str, dict[str, dict[str, float]]] = {}
    for name, trajectory in poses.items():
        if int(trajectory.shape[0]) != int(depth.shape[0]):
            print(f"skipping {name}: {int(trajectory.shape[0])} poses for {int(depth.shape[0])} frames")
            continue
        predicted = predicted_clouds(observations=observations, trajectory=trajectory)
        results[name] = score_trajectory(predicted=predicted, reference=reference)

    print()
    print("object-map metrics by trajectory (same points, only the pose changes)")
    print("  lower is better: accuracy, completeness, ghost_point_ratio")
    print("  higher is better: fscore_5cm, voxel_iou_5cm")
    print()
    print(render(results))
    print(
        "The reference cloud is built from the GT masks through the GROUND-TRUTH\n"
        "poses, so a trajectory is scored on where it puts the objects, not on how\n"
        "close it stays to the raw one.  Categories are pooled: the ablation only\n"
        "changes the trajectory, so any identity error cancels on both sides."
    )

    if args.json_out is not None:
        args.json_out.expanduser().resolve().write_text(
            json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"json_out={args.json_out.expanduser().resolve()}")


if __name__ == "__main__":
    main()
