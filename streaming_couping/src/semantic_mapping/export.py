"""Portable artifact export for semantic-instance maps."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .mapping import ObjectTrackMap, SemanticMapResult


INSTANCE_PALETTE_RGB8: tuple[tuple[int, int, int], ...] = (
    (230, 25, 75),
    (60, 180, 75),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 212),
    (0, 128, 128),
    (220, 190, 255),
    (170, 110, 40),
    (255, 250, 200),
    (128, 0, 0),
    (170, 255, 195),
    (128, 128, 0),
)


def export_semantic_map(
    result: SemanticMapResult,
    output_dir: str | Path,
    *,
    revision: str = "semantic_mapping_backend_neutral_v2",
) -> dict[str, Any]:
    """Write a map artifact and return a JSON-compatible summary."""

    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    labels = tuple(
        sorted(
            set(result.semantic_labels)
            | {str(track.category) for track in result.object_tracks}
        )
    )
    category_to_id = {label: index for index, label in enumerate(labels)}
    semantic_rgb = _instance_colors(result.instance_ids)
    category_ids = torch.tensor(
        [category_to_id[label] for label in result.semantic_labels],
        dtype=torch.long,
    )
    scene_category_ids = torch.tensor(
        [category_to_id.get(label, -1) for label in result.scene_semantic_labels],
        dtype=torch.long,
    )
    scene_rgb = _visible_rgb(result.scene_voxel_rgb)
    scene_semantic_rgb = scene_rgb * 0.35
    scene_labeled = result.scene_instance_ids.ge(0)
    if bool(scene_labeled.any()):
        scene_semantic_rgb[scene_labeled] = _instance_colors(
            result.scene_instance_ids[scene_labeled]
        )

    artifact_path = directory / "semantic_map.pt"
    payload = {
        "schema": 2,
        "revision": revision,
        "scene_voxel_points": result.scene_voxel_points.detach().float().cpu(),
        "scene_voxel_rgb": result.scene_voxel_rgb.detach().float().cpu(),
        "scene_semantic_rgb": scene_semantic_rgb,
        "scene_semantic_labels": list(result.scene_semantic_labels),
        "scene_category_ids": scene_category_ids,
        "scene_instance_ids": result.scene_instance_ids.detach().long().cpu(),
        "scene_evidence_weights": result.scene_evidence_weights.detach().float().cpu(),
        "scene_observation_counts": result.scene_observation_counts.detach().long().cpu(),
        "voxel_points": result.voxel_points.detach().float().cpu(),
        "voxel_rgb": result.voxel_rgb.detach().float().cpu(),
        "semantic_rgb": semantic_rgb,
        "semantic_labels": list(result.semantic_labels),
        "category_names": list(labels),
        "category_ids": category_ids,
        "instance_ids": result.instance_ids.detach().long().cpu(),
        "evidence_weights": result.evidence_weights.detach().float().cpu(),
        "observation_counts": result.observation_counts.detach().long().cpu(),
        "object_tracks": _track_payload(result.object_tracks),
        "metadata": _json_safe(result.metadata),
    }
    torch.save(payload, artifact_path)

    semantic_ply_path = directory / "semantic_map.ply"
    _write_ply(
        semantic_ply_path,
        points=result.voxel_points,
        colors=semantic_rgb,
        category_ids=category_ids,
        instance_ids=result.instance_ids,
        weights=result.evidence_weights,
        observations=result.observation_counts,
    )
    rgb_ply_path = directory / "rgb_map.ply"
    rgb_colors = result.voxel_rgb
    if result.voxel_count and not bool(rgb_colors.abs().sum() > 0):
        rgb_colors = semantic_rgb
    _write_ply(
        rgb_ply_path,
        points=result.voxel_points,
        colors=rgb_colors,
        category_ids=category_ids,
        instance_ids=result.instance_ids,
        weights=result.evidence_weights,
        observations=result.observation_counts,
    )

    scene_rgb_ply_path = directory / "scene_rgb_map.ply"
    _write_ply(
        scene_rgb_ply_path,
        points=result.scene_voxel_points,
        colors=scene_rgb,
        category_ids=scene_category_ids,
        instance_ids=result.scene_instance_ids,
        weights=result.scene_evidence_weights,
        observations=result.scene_observation_counts,
    )
    scene_semantic_ply_path = directory / "scene_semantic_map.ply"
    _write_ply(
        scene_semantic_ply_path,
        points=result.scene_voxel_points,
        colors=scene_semantic_rgb,
        category_ids=scene_category_ids,
        instance_ids=result.scene_instance_ids,
        weights=result.scene_evidence_weights,
        observations=result.scene_observation_counts,
    )

    tracks_ply_path = directory / "object_tracks.ply"
    (
        track_points,
        track_colors,
        track_categories,
        track_ids,
        track_weights,
        track_frames,
    ) = _flatten_tracks(result.object_tracks, category_to_id)
    _write_ply(
        tracks_ply_path,
        points=track_points,
        colors=track_colors,
        category_ids=track_categories,
        instance_ids=track_ids,
        weights=track_weights,
        observations=torch.ones(track_points.shape[0], dtype=torch.long),
        frame_ids=track_frames,
    )

    tracks_path = directory / "object_tracks.json"
    tracks_summary = [_track_summary(track) for track in result.object_tracks]
    tracks_path.write_text(
        json.dumps(tracks_summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    summary = {
        "schema": 2,
        "revision": revision,
        "scene_voxel_count": result.scene_voxel_count,
        "scene_labeled_voxel_count": result.scene_labeled_voxel_count,
        "voxel_count": result.voxel_count,
        "labeled_voxel_count": result.labeled_voxel_count,
        "category_names": list(labels),
        "instance_count": len(result.object_tracks),
        "static_instance_count": sum(
            int(track.is_static is True) for track in result.object_tracks
        ),
        "dynamic_instance_count": sum(
            int(track.is_static is False) for track in result.object_tracks
        ),
        "metadata": _json_safe(result.metadata),
        "outputs": {
            "artifact": str(artifact_path),
            "scene_rgb_ply": str(scene_rgb_ply_path),
            "scene_semantic_ply": str(scene_semantic_ply_path),
            "semantic_ply": str(semantic_ply_path),
            "rgb_ply": str(rgb_ply_path),
            "object_tracks_ply": str(tracks_ply_path),
            "object_tracks_json": str(tracks_path),
        },
        "output_descriptions": {
            "scene_rgb_ply": (
                "All valid geometry from all selected frames, confidence-weighted "
                "and voxel-fused, colored with observed RGB."
            ),
            "scene_semantic_ply": (
                "The same full scene; static prompted object voxels use instance "
                "colors and unlabeled context uses dimmed RGB."
            ),
            "semantic_ply": (
                "Static prompted object voxels only, multi-frame voxel-fused and "
                "colored by persistent instance ID."
            ),
            "rgb_ply": (
                "The same object-only fused voxels as semantic_ply, colored with RGB."
            ),
            "object_tracks_ply": (
                "Accepted per-frame object observations concatenated by track; not "
                "voxel-fused, and each point retains its frame_id."
            ),
        },
        "objects": tracks_summary,
    }
    summary_path = directory / "map_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def _track_payload(tracks: Sequence[ObjectTrackMap]) -> list[dict[str, Any]]:
    return [
        {
            "instance_id": int(track.instance_id),
            "category": str(track.category),
            "points": track.points.detach().float().cpu(),
            "weights": track.weights.detach().float().cpu(),
            "frame_indices": track.frame_indices.detach().long().cpu(),
            "observations": int(track.observations),
            "is_static": track.is_static,
        }
        for track in tracks
    ]


def _flatten_tracks(
    tracks: Sequence[ObjectTrackMap],
    category_to_id: Mapping[str, int],
) -> tuple[torch.Tensor, ...]:
    point_chunks = []
    color_chunks = []
    category_chunks = []
    instance_chunks = []
    weight_chunks = []
    frame_chunks = []
    for track in tracks:
        if not track.points.numel():
            continue
        count = int(track.points.shape[0])
        color = torch.tensor(
            INSTANCE_PALETTE_RGB8[track.instance_id % len(INSTANCE_PALETTE_RGB8)],
            dtype=torch.float32,
        ) / 255.0
        point_chunks.append(track.points.detach().float().cpu())
        color_chunks.append(color.expand(count, 3))
        category_chunks.append(
            torch.full(
                (count,),
                int(category_to_id[track.category]),
                dtype=torch.long,
            )
        )
        instance_chunks.append(
            torch.full((count,), int(track.instance_id), dtype=torch.long)
        )
        weight_chunks.append(track.weights.detach().float().cpu())
        frame_chunks.append(track.frame_indices.detach().long().cpu())
    if not point_chunks:
        return (
            torch.empty(0, 3, dtype=torch.float32),
            torch.empty(0, 3, dtype=torch.float32),
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.float32),
            torch.empty(0, dtype=torch.long),
        )
    return (
        torch.cat(point_chunks, dim=0),
        torch.cat(color_chunks, dim=0),
        torch.cat(category_chunks, dim=0),
        torch.cat(instance_chunks, dim=0),
        torch.cat(weight_chunks, dim=0),
        torch.cat(frame_chunks, dim=0),
    )


def _instance_colors(instance_ids: torch.Tensor) -> torch.Tensor:
    values = instance_ids.detach().long().cpu()
    if not values.numel():
        return torch.empty(0, 3, dtype=torch.float32)
    palette = torch.tensor(INSTANCE_PALETTE_RGB8, dtype=torch.float32) / 255.0
    return palette[torch.remainder(values, len(INSTANCE_PALETTE_RGB8))]


def _visible_rgb(colors: torch.Tensor) -> torch.Tensor:
    value = colors.detach().float().cpu().clamp(0.0, 1.0)
    if value.numel() and not bool(value.abs().sum() > 0):
        return torch.full_like(value, 0.5)
    return value


def _track_summary(track: ObjectTrackMap) -> dict[str, Any]:
    points = track.points.detach().float().cpu()
    if points.numel():
        bounds_min = points.min(dim=0).values.tolist()
        bounds_max = points.max(dim=0).values.tolist()
    else:
        bounds_min = None
        bounds_max = None
    return {
        "instance_id": int(track.instance_id),
        "category": str(track.category),
        "point_count": int(points.shape[0]),
        "observations": int(track.observations),
        "is_static": track.is_static,
        "bounds_min": bounds_min,
        "bounds_max": bounds_max,
        "frame_count": int(track.frame_indices.unique().numel()),
        "first_frame_id": (
            int(track.frame_indices.min()) if track.frame_indices.numel() else None
        ),
        "last_frame_id": (
            int(track.frame_indices.max()) if track.frame_indices.numel() else None
        ),
    }


def _write_ply(
    path: Path,
    *,
    points: torch.Tensor,
    colors: torch.Tensor,
    category_ids: torch.Tensor,
    instance_ids: torch.Tensor,
    weights: torch.Tensor,
    observations: torch.Tensor,
    frame_ids: torch.Tensor | None = None,
) -> None:
    points = points.detach().float().cpu()
    colors = colors.detach().float().cpu().clamp(0.0, 1.0)
    category_ids = category_ids.detach().long().cpu()
    instance_ids = instance_ids.detach().long().cpu()
    weights = weights.detach().float().cpu()
    observations = observations.detach().long().cpu()
    count = int(points.shape[0])
    if frame_ids is None:
        frame_ids = torch.full((count,), -1, dtype=torch.long)
    else:
        frame_ids = frame_ids.detach().long().cpu()
    if any(
        tuple(value.shape) != (count,)
        for value in (category_ids, instance_ids, weights, observations, frame_ids)
    ) or tuple(colors.shape) != (count, 3):
        raise ValueError("PLY fields have inconsistent lengths.")
    array = np.empty(
        count,
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("category_id", "<i4"),
            ("instance_id", "<i4"),
            ("evidence_weight", "<f4"),
            ("observations", "<i4"),
            ("frame_id", "<i4"),
        ],
    )
    if count:
        array["x"], array["y"], array["z"] = points.numpy().T
        rgb = colors.mul(255.0).round().byte().numpy()
        array["red"], array["green"], array["blue"] = rgb.T
        array["category_id"] = category_ids.numpy()
        array["instance_id"] = instance_ids.numpy()
        array["evidence_weight"] = weights.numpy()
        array["observations"] = observations.numpy()
        array["frame_id"] = frame_ids.numpy()
    header = "\n".join(
        (
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {count}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "property int category_id",
            "property int instance_id",
            "property float evidence_weight",
            "property int observations",
            "property int frame_id",
            "end_header",
            "",
        )
    ).encode("ascii")
    with path.open("wb") as handle:
        handle.write(header)
        array.tofile(handle)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value
