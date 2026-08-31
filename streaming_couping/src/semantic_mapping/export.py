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
    revision: str = "semantic_mapping_backend_neutral_v1",
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
    semantic_rgb = torch.stack(
        [
            torch.tensor(
                INSTANCE_PALETTE_RGB8[int(instance_id) % len(INSTANCE_PALETTE_RGB8)],
                dtype=torch.float32,
            )
            / 255.0
            for instance_id in result.instance_ids.tolist()
        ],
        dim=0,
    ) if result.voxel_count else torch.empty(0, 3, dtype=torch.float32)
    category_ids = torch.tensor(
        [category_to_id[label] for label in result.semantic_labels],
        dtype=torch.long,
    )

    artifact_path = directory / "semantic_map.pt"
    payload = {
        "schema": 1,
        "revision": revision,
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

    tracks_ply_path = directory / "object_tracks.ply"
    track_points, track_colors, track_categories, track_ids, track_weights = (
        _flatten_tracks(result.object_tracks, category_to_id)
    )
    _write_ply(
        tracks_ply_path,
        points=track_points,
        colors=track_colors,
        category_ids=track_categories,
        instance_ids=track_ids,
        weights=track_weights,
        observations=torch.ones(track_points.shape[0], dtype=torch.long),
    )

    tracks_path = directory / "object_tracks.json"
    tracks_summary = [_track_summary(track) for track in result.object_tracks]
    tracks_path.write_text(
        json.dumps(tracks_summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    summary = {
        "schema": 1,
        "revision": revision,
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
            "semantic_ply": str(semantic_ply_path),
            "rgb_ply": str(rgb_ply_path),
            "object_tracks_ply": str(tracks_ply_path),
            "object_tracks_json": str(tracks_path),
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
    if not point_chunks:
        return (
            torch.empty(0, 3, dtype=torch.float32),
            torch.empty(0, 3, dtype=torch.float32),
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.float32),
        )
    return (
        torch.cat(point_chunks, dim=0),
        torch.cat(color_chunks, dim=0),
        torch.cat(category_chunks, dim=0),
        torch.cat(instance_chunks, dim=0),
        torch.cat(weight_chunks, dim=0),
    )


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
) -> None:
    points = points.detach().float().cpu()
    colors = colors.detach().float().cpu().clamp(0.0, 1.0)
    category_ids = category_ids.detach().long().cpu()
    instance_ids = instance_ids.detach().long().cpu()
    weights = weights.detach().float().cpu()
    observations = observations.detach().long().cpu()
    count = int(points.shape[0])
    if any(
        tuple(value.shape) != (count,)
        for value in (category_ids, instance_ids, weights, observations)
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
