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
        if (path / "scans").is_dir() and (path / "dslr").is_dir():
            scene_ids.append(path.name)
    return scene_ids


def load_ply_mesh(path: Path, label_property: str = "label") -> Dict[str, np.ndarray]:
    """Load vertices, triangle faces and optional per-vertex semantic labels."""
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
    elif "semantic_label" in names:
        labels = np.asarray(vertices_raw["semantic_label"], dtype=np.int32)

    faces_raw = ply["face"].data
    face_field = "vertex_indices"
    if face_field not in (faces_raw.dtype.names or ()):
        face_field = "vertex_index"
    faces = np.asarray([row for row in faces_raw[face_field]], dtype=np.int32)

    return {"vertices": vertices, "faces": faces, "labels": labels}


def load_vertex_instances(
    segments_path: Path,
    annotation_path: Path,
    num_vertices: int,
) -> Dict[str, Any]:
    """Create per-vertex object IDs from ScanNet++ segments annotations."""
    if not segments_path.is_file():
        raise FileNotFoundError(f"Missing segments file: {segments_path}")
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Missing annotation file: {annotation_path}")

    segments = read_json(segments_path)
    annotations = read_json(annotation_path)
    seg_indices = np.asarray(segments["segIndices"], dtype=np.int32)
    if len(seg_indices) != num_vertices:
        raise ValueError(
            f"segments.json length {len(seg_indices)} does not match "
            f"mesh vertex count {num_vertices}"
        )

    vertex_instance_ids = np.zeros(num_vertices, dtype=np.int32)
    objects: Dict[int, Dict[str, Any]] = {}
    for group in annotations.get("segGroups", []):
        object_id = int(group.get("objectId", group.get("id", 0)))
        if object_id <= 0:
            continue
        object_segments = np.asarray(group.get("segments", []), dtype=np.int32)
        if object_segments.size == 0:
            continue

        mask = np.isin(seg_indices, object_segments)
        vertex_instance_ids[mask] = object_id
        objects[object_id] = {
            key: value for key, value in group.items() if key not in {"segments"}
        }
        objects[object_id]["num_vertices"] = int(mask.sum())

    return {"vertex_instance_ids": vertex_instance_ids, "objects": objects}


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


def load_train_test_lists(path: Path) -> Dict[str, List[str]]:
    if not path.is_file():
        return {}
    payload = read_json(path)
    if not isinstance(payload, dict):
        return {}

    result: Dict[str, List[str]] = {}
    for key, value in payload.items():
        if isinstance(value, list):
            result[key.lower()] = [str(item) for item in value]
    return result


def filter_by_split(
    image_names: List[str], train_test_lists: Dict[str, List[str]], split: str
) -> List[str]:
    split = split.lower()
    if split == "all" or not train_test_lists:
        return image_names

    candidates = train_test_lists.get(split)
    if candidates is None:
        candidates = train_test_lists.get(f"{split}_frames")
    if candidates is None:
        candidates = train_test_lists.get(f"{split}_images")
    if candidates is None:
        raise ValueError(
            f"Split '{split}' not found in train_test_lists.json. "
            f"Available keys: {sorted(train_test_lists.keys())}"
        )

    allowed = set(candidates)
    allowed.update(Path(name).name for name in candidates)
    return [
        name for name in image_names if name in allowed or Path(name).name in allowed
    ]
