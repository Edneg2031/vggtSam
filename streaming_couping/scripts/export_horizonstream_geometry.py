#!/usr/bin/env python3
"""Export a frozen HorizonStream geometry cache as a scene point cloud.

This exporter deliberately does not import or run SAM3, DINO, or any semantic
association module.  It only converts the cached HorizonStream depth, pose,
intrinsics, confidence, and RGB fields into a confidence-weighted voxel-fused
geometry point cloud and a camera-center trajectory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from streaming_couping.src.horizonstream_cache import load_horizonstream_cache
from streaming_couping.src.semantic_mapping.adapters import (
    HorizonStreamGeometryCacheAdapter,
)
from streaming_couping.src.semantic_mapping.contracts import SegmentationFrame
from streaming_couping.src.semantic_mapping.export import _write_ply
from streaming_couping.src.semantic_mapping.mapping import (
    SemanticMapBuilder,
    SemanticMapConfig,
)


def main() -> None:
    args = _parse_args()
    cache_path = args.cache.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = load_horizonstream_cache(cache_path)
    adapter = HorizonStreamGeometryCacheAdapter(payload)
    image_paths = tuple(str(value) for value in payload["image_paths"])
    geometry_frames = adapter.infer(image_paths)
    frame_count = len(geometry_frames)
    height, width = adapter.image_size

    config = SemanticMapConfig(
        fusion_policy="raw",
        voxel_size_m=float(args.voxel_size_m),
        min_geometry_confidence=float(args.min_geometry_confidence),
        max_voxels=int(args.max_voxels),
        max_scene_voxels=int(args.max_scene_voxels),
    )
    builder = SemanticMapBuilder(config)
    for geometry in geometry_frames:
        builder.update(
            geometry,
            SegmentationFrame(
                frame_id=int(geometry.frame_id),
                image_size=geometry.image_size,
                observations=(),
                backend="none",
                metadata={"geometry_only": True},
            ),
        )
    result = builder.finalize(
        metadata={
            "backend": "horizonstream",
            "geometry_only": True,
            "sam_used": False,
            "dino_used": False,
            "pose_modified": False,
            "pointmap_modified": False,
            "cache": str(cache_path),
        }
    )

    points = result.scene_voxel_points.detach().float().cpu()
    colors = result.scene_voxel_rgb.detach().float().cpu()
    point_count = int(points.shape[0])
    category_ids = torch.full((point_count,), -1, dtype=torch.long)
    instance_ids = torch.full((point_count,), -1, dtype=torch.long)
    geometry_ply = output_dir / "horizonstream_geometry.ply"
    _write_ply(
        geometry_ply,
        points=points,
        colors=colors,
        category_ids=category_ids,
        instance_ids=instance_ids,
        weights=result.scene_evidence_weights,
        observations=result.scene_observation_counts,
    )

    geometry_artifact = output_dir / "horizonstream_geometry_map.pt"
    torch.save(
        {
            "schema": 1,
            "revision": "horizonstream_geometry_only_v1",
            "backend": "horizonstream",
            "geometry_only": True,
            "sam_used": False,
            "dino_used": False,
            "pose_modified": False,
            "pointmap_modified": False,
            "cache": str(cache_path),
            "world_points": points,
            "rgb": colors,
            "confidence": result.scene_evidence_weights.detach()
            .float()
            .cpu(),
            "observation_counts": result.scene_observation_counts.detach()
            .long()
            .cpu(),
            "voxel_size_m": float(args.voxel_size_m),
            "min_geometry_confidence": float(args.min_geometry_confidence),
            "frame_count": frame_count,
            "image_size": (height, width),
            "image_paths": image_paths,
        },
        geometry_artifact,
    )

    trajectory_path = output_dir / "camera_trajectory.txt"
    camera_centers = _camera_centers(payload["world_to_camera"])
    _write_trajectory(
        trajectory_path,
        camera_centers=camera_centers,
        source_positions=payload.get("source_positions"),
    )

    depth = torch.as_tensor(payload["depth"]).detach().float().cpu()
    confidence = torch.as_tensor(payload["confidence"]).detach().float().cpu()
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if confidence.ndim == 4 and confidence.shape[-1] == 1:
        confidence = confidence[..., 0]
    valid_depth = torch.isfinite(depth) & (depth > 0.0)
    accepted_depth = valid_depth & (
        confidence >= float(args.min_geometry_confidence)
    )
    path_length = 0.0
    if camera_centers.shape[0] > 1:
        path_length = float(
            torch.linalg.vector_norm(
                camera_centers[1:] - camera_centers[:-1], dim=-1
            ).sum()
        )
    summary: dict[str, Any] = {
        "schema": 1,
        "revision": "horizonstream_geometry_only_v1",
        "backend": "horizonstream",
        "geometry_only": True,
        "sam_used": False,
        "dino_used": False,
        "pose_modified": False,
        "pointmap_modified": False,
        "cache": str(cache_path),
        "image_paths": image_paths,
        "source_positions": list(payload.get("source_positions", range(frame_count))),
        "source_count": int(payload.get("source_count", frame_count)),
        "frame_count": frame_count,
        "image_size": [height, width],
        "frame_selection": dict(
            payload.get("input", {}).get("frame_selection", {})
        ),
        "window_size": int(payload.get("window_size", 0)),
        "sliding_size": int(payload.get("sliding_size", 0)),
        "pose_source": str(payload.get("pose_source", "unknown")),
        "scale_type": str(payload.get("scale_type", "unknown")),
        "voxel_size_m": float(args.voxel_size_m),
        "min_geometry_confidence": float(args.min_geometry_confidence),
        "raw_valid_depth_pixels": int(valid_depth.sum()),
        "accepted_depth_pixels": int(accepted_depth.sum()),
        "accepted_depth_ratio": float(accepted_depth.float().mean()),
        "mean_confidence": float(confidence.mean()),
        "geometry_voxel_count": point_count,
        "camera_path_length_m": path_length,
        "outputs": {
            "geometry_ply": str(geometry_ply),
            "geometry_artifact": str(geometry_artifact),
            "camera_trajectory": str(trajectory_path),
        },
    }
    summary_path = output_dir / "geometry_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(
        "HorizonStream geometry export completed "
        f"frames={frame_count} size=({height},{width}) "
        f"voxels={point_count} "
        f"accepted_depth_ratio={summary['accepted_depth_ratio']:.4f}"
    )
    print(f"geometry_ply={geometry_ply}")
    print(f"geometry_artifact={geometry_artifact}")
    print(f"camera_trajectory={trajectory_path}")
    print(f"geometry_summary={summary_path}")


def _camera_centers(world_to_camera: torch.Tensor) -> torch.Tensor:
    pose = torch.as_tensor(world_to_camera).detach().float().cpu()
    if pose.ndim != 3 or tuple(pose.shape[1:]) != (3, 4):
        raise ValueError("world_to_camera must have shape [S,3,4].")
    homogeneous = torch.eye(4, dtype=pose.dtype).expand(pose.shape[0], -1, -1).clone()
    homogeneous[:, :3] = pose
    return torch.linalg.inv(homogeneous)[:, :3, 3]


def _write_trajectory(
    path: Path,
    *,
    camera_centers: torch.Tensor,
    source_positions: Any,
) -> None:
    positions = list(source_positions) if source_positions is not None else []
    rows = ["sequence_index\tsource_position\tcamera_x\tcamera_y\tcamera_z"]
    for index, center in enumerate(camera_centers.tolist()):
        source_position = positions[index] if index < len(positions) else index
        rows.append(
            f"{index}\t{int(source_position)}\t"
            f"{float(center[0]):.9f}\t{float(center[1]):.9f}\t{float(center[2]):.9f}"
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--voxel-size-m", type=float, default=0.05)
    parser.add_argument("--min-geometry-confidence", type=float, default=0.30)
    parser.add_argument("--max-voxels", type=int, default=1_000_000)
    parser.add_argument("--max-scene-voxels", type=int, default=1_000_000)
    args = parser.parse_args()
    if float(args.voxel_size_m) <= 0.0:
        parser.error("--voxel-size-m must be positive")
    if not 0.0 <= float(args.min_geometry_confidence) <= 1.0:
        parser.error("--min-geometry-confidence must be in [0,1]")
    if int(args.max_voxels) < 1 or int(args.max_scene_voxels) < 1:
        parser.error("voxel limits must be positive")
    return args


if __name__ == "__main__":
    main()
