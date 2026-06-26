"""I/O helpers for ScanNet++ preprocessing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


def read_text_list(path: Path) -> List[str]:
    with path.open("r", encoding="utf8") as handle:
        return [
            line.strip()
            for line in handle
            if line.strip() and not line.lstrip().startswith("#")
        ]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def discover_scene_ids(data_root: Path) -> List[str]:
    scene_ids: List[str] = []
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")
    for path in sorted(data_root.iterdir()):
        if not path.is_dir():
            continue
        if (
            (path / "images").is_dir()
            and (path / "colmap").is_dir()
            and (path / "mesh_aligned_0.05.ply").is_file()
        ):
            scene_ids.append(path.name)
    return scene_ids


def load_ply_mesh(
    path: Path,
    label_property: str = "label",
    instance_property: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """Load vertices, triangle faces and optional per-vertex labels."""
    from plyfile import PlyData

    if not path.is_file():
        raise FileNotFoundError(f"Missing mesh: {path}")

    ply = PlyData.read(str(path))
    vertices_raw = ply["vertex"].data
    names = vertices_raw.dtype.names or ()
    vertices = np.stack(
        [vertices_raw["x"], vertices_raw["y"], vertices_raw["z"]], axis=1
    ).astype(np.float32)

    labels = None
    if label_property in names:
        labels = np.asarray(vertices_raw[label_property], dtype=np.int32)
    else:
        semantic_property = _first_property(
            names,
            [
                "semantic_label",
                "semantic_id",
                "semantic",
                "label_id",
                "class_id",
                "category_id",
                "nyu40id",
            ],
        )
        if semantic_property is not None:
            labels = np.asarray(vertices_raw[semantic_property], dtype=np.int32)

    instances = None
    if instance_property and instance_property in names:
        instances = np.asarray(vertices_raw[instance_property], dtype=np.int32)
    else:
        mesh_instance_property = _first_property(
            names,
            [
                "instance_id",
                "instance",
                "object_id",
                "objectId",
                "objectid",
                "segment_id",
                "segmentId",
                "segmentid",
            ],
        )
        if mesh_instance_property is not None:
            instances = np.asarray(vertices_raw[mesh_instance_property], dtype=np.int32)

    colors = None
    if all(name in names for name in ("red", "green", "blue")):
        colors = np.stack(
            [vertices_raw["red"], vertices_raw["green"], vertices_raw["blue"]],
            axis=1,
        ).astype(np.uint8)

    faces_raw = ply["face"].data
    face_field = "vertex_indices"
    if face_field not in (faces_raw.dtype.names or ()):
        face_field = "vertex_index"
    faces = np.asarray([row for row in faces_raw[face_field]], dtype=np.int32)

    return {
        "vertices": vertices,
        "faces": faces,
        "labels": labels,
        "instances": instances,
        "colors": colors,
        "vertex_properties": list(names),
    }


def _first_property(names: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    names_set = set(names)
    lowered = {name.lower(): name for name in names}
    for candidate in candidates:
        if candidate in names_set:
            return candidate
        matched = lowered.get(candidate.lower())
        if matched is not None:
            return matched
    return None


def face_property_from_vertices(
    vertex_property: Optional[np.ndarray],
    faces: np.ndarray,
    *,
    ignore_values: Iterable[int],
    default_value: int,
) -> np.ndarray:
    """Assign a face property from its three vertex properties.

    For triangles where two valid vertices agree, that value wins. Otherwise the
    first valid vertex value is used. This mirrors the official simple transfer
    behavior while avoiding obvious ignore-label dominance.
    """
    if vertex_property is None:
        return np.full(len(faces), default_value, dtype=np.int32)

    vals = np.asarray(vertex_property, dtype=np.int32)[faces]
    ignore = np.zeros(vals.shape, dtype=bool)
    for ignore_value in ignore_values:
        ignore |= vals == int(ignore_value)
    valid = ~ignore

    result = np.full(vals.shape[0], default_value, dtype=np.int32)
    for col in range(3):
        use = (result == default_value) & valid[:, col]
        result[use] = vals[use, col]

    a, b, c = vals[:, 0], vals[:, 1], vals[:, 2]
    va, vb, vc = valid[:, 0], valid[:, 1], valid[:, 2]
    ab = va & vb & (a == b)
    ac = va & vc & (a == c)
    bc = vb & vc & (b == c)
    result[ab | ac] = a[ab | ac]
    result[bc] = b[bc]
    return result
