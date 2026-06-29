"""ScanNet++ pinhole 3D-to-2D semantic/instance preprocessing."""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from .colmap import image_lookup, ordered_images, read_colmap_text_model
from .io import (
    discover_scene_ids,
    face_property_from_vertices,
    load_ply_mesh,
    load_vertex_instances,
    read_text_list,
    write_json,
)
from .rasterize import has_numba, rasterize_labels_and_points
from .visualize import colorize_ids, overlay_labels, save_labeled_summary, save_rgb


@dataclass
class ScanNetPP2DConfig:
    data_root: Path
    output_root: Path
    scene_ids: List[str]
    scene_list: Optional[Path] = None
    limit_scenes: Optional[int] = None
    mesh_filename: str = "mesh_aligned_0.05.ply"
    annotation_root: Optional[Path] = None
    annotation_scan_subdir: str = "scans"
    semantic_mesh_filename: str = "mesh_aligned_0.05_semantic.ply"
    segments_filename: str = "segments.json"
    annotations_filename: str = "segments_anno.json"
    metadata_root: Optional[Path] = None
    semantic_classes_file: Optional[Path] = None
    frame_step: int = 1
    max_frames: Optional[int] = None
    near: float = 0.001
    semantic_ignore_label: int = 65535
    save_visualizations: bool = False
    save_raster: bool = True
    save_pointmaps: bool = True
    skip_existing: bool = True
    dry_run: bool = False


def prepare_scannetpp_2d(config: ScanNetPP2DConfig) -> Dict[str, Any]:
    config.data_root = Path(config.data_root)
    config.output_root = Path(config.output_root)
    config.output_root.mkdir(parents=True, exist_ok=True)

    scene_ids = _resolve_scene_ids(config)
    print(f"scannetpp_preprocess scenes={len(scene_ids)} root={config.data_root}")
    if not has_numba():
        print(
            "Warning: numba is not installed; CPU rasterization will work but can be slow. "
            "Install it with `pip install numba` on the server."
        )

    manifest: Dict[str, Any] = {
        "dataset": "scannetpp",
        "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "data_root": str(config.data_root),
        "output_root": str(config.output_root),
        "config": _jsonable_config(config),
        "scenes": [],
    }

    for scene_id in tqdm(scene_ids, desc="scene"):
        if config.dry_run:
            scene_manifest = _inspect_scene(scene_id, config)
        else:
            scene_manifest = _process_scene(scene_id, config)
        manifest["scenes"].append(scene_manifest)

    write_json(config.output_root / "manifest.json", manifest)
    print(f"Wrote manifest: {config.output_root / 'manifest.json'}")
    return manifest


def _inspect_scene(scene_id: str, config: ScanNetPP2DConfig) -> Dict[str, Any]:
    assets = _resolve_scene_assets(scene_id, config)
    scene_root = assets["scene_root"]
    image_dir = assets["image_dir"]
    colmap_dir = assets["colmap_dir"]
    pinhole_mesh_path = assets["pinhole_mesh_path"]
    annotation_mesh_path = assets["annotation_mesh_path"]
    segments_path = assets["segments_path"]
    annotations_path = assets["annotations_path"]

    required = {
        "annotation_mesh": annotation_mesh_path,
        "segments": segments_path,
        "segments_anno": annotations_path,
        "images": image_dir,
        "colmap": colmap_dir,
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if "colmap" in missing or "images" in missing:
        cameras = {}
        colmap_images = {}
        frame_names = []
    else:
        cameras, colmap_images = read_colmap_text_model(colmap_dir)
        frame_names = _select_frame_names(
            image_dir,
            ordered_images(colmap_images),
            config,
        )
    pinhole_mesh_info = _inspect_mesh(pinhole_mesh_path)
    annotation_mesh_info = _inspect_mesh(annotation_mesh_path)
    print(
        f"[dry-run:{scene_id}] missing={missing} cameras={len(cameras)} "
        f"colmap_images={len(colmap_images)} selected_frames={len(frame_names)} "
        f"pinhole_mesh_info={pinhole_mesh_info} "
        f"annotation_mesh_info={annotation_mesh_info}"
    )
    return {
        "scene_id": scene_id,
        "scene_root": str(scene_root),
        "dry_run": True,
        "missing": missing,
        "pinhole_mesh_path": str(pinhole_mesh_path),
        "annotation_mesh_path": str(annotation_mesh_path),
        "pinhole_mesh_info": pinhole_mesh_info,
        "annotation_mesh_info": annotation_mesh_info,
        "num_cameras": int(len(cameras)),
        "num_colmap_images": int(len(colmap_images)),
        "num_selected_frames": int(len(frame_names)),
        "selected_frame_preview": frame_names[:10],
    }


def _inspect_mesh(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    mesh = load_ply_mesh(path)
    return {
        "vertex_properties": mesh.get("vertex_properties", []),
        "has_semantic_labels": mesh.get("labels") is not None,
        "has_instance_ids": mesh.get("instances") is not None,
        "has_vertex_colors": mesh.get("colors") is not None,
        "num_vertices": int(len(mesh["vertices"])),
        "num_faces": int(len(mesh["faces"])),
    }


def _process_scene(scene_id: str, config: ScanNetPP2DConfig) -> Dict[str, Any]:
    assets = _resolve_scene_assets(scene_id, config)
    scene_root = assets["scene_root"]
    image_dir = assets["image_dir"]
    colmap_dir = assets["colmap_dir"]
    semantic_mesh_path = assets["annotation_mesh_path"]
    segments_path = assets["segments_path"]
    annotations_path = assets["annotations_path"]

    out_scene_dir = config.output_root / scene_id
    semantic_dir = out_scene_dir / "semantic_masks"
    instance_dir = out_scene_dir / "instance_masks"
    pointmap_dir = out_scene_dir / "pointmaps"
    raster_dir = out_scene_dir / "raster"
    viz_dir = out_scene_dir / "visualizations"
    for path in [semantic_dir, instance_dir, pointmap_dir, raster_dir, viz_dir]:
        path.mkdir(parents=True, exist_ok=True)

    print(f"\n[{scene_id}] loading mesh and annotations")
    semantic_class_names = _load_semantic_class_names(config)
    mesh = load_ply_mesh(semantic_mesh_path)
    vertices = mesh["vertices"]
    faces = mesh["faces"]
    raw_vertex_semantic_ids = mesh["labels"]
    if raw_vertex_semantic_ids is None:
        raise ValueError(
            f"{semantic_mesh_path} does not contain a vertex 'label' property."
        )

    vertex_instance_ids, objects = _load_instance_data(
        mesh,
        segments_path,
        annotations_path,
        vertex_semantic_ids=raw_vertex_semantic_ids,
        semantic_class_names=semantic_class_names,
        num_vertices=len(vertices),
    )
    vertex_semantic_ids, semantic_source, objects = _resolve_vertex_semantics(
        raw_vertex_semantic_ids,
        vertex_instance_ids,
        objects,
        semantic_class_names=semantic_class_names,
        semantic_ignore_label=config.semantic_ignore_label,
    )
    print(
        f"[{scene_id}] semantic_source={semantic_source} "
        f"semantic_classes={len(semantic_class_names)}"
    )

    face_semantic_ids = face_property_from_vertices(
        vertex_semantic_ids,
        faces,
        ignore_values=(-100, -1, config.semantic_ignore_label),
        default_value=config.semantic_ignore_label,
    )
    face_instance_ids = face_property_from_vertices(
        vertex_instance_ids,
        faces,
        ignore_values=(-1, 0),
        default_value=0,
    )

    cameras, colmap_images = read_colmap_text_model(colmap_dir)
    ordered_colmap_images = ordered_images(colmap_images)
    image_by_name = image_lookup(ordered_colmap_images)
    frame_names = _select_frame_names(image_dir, ordered_colmap_images, config)
    print(f"[{scene_id}] frames={len(frame_names)}")

    scene_manifest: Dict[str, Any] = {
        "scene_id": scene_id,
        "scene_root": str(scene_root),
        "output_dir": str(out_scene_dir),
        "annotation_mesh_path": str(semantic_mesh_path),
        "pinhole_mesh_path": str(assets["pinhole_mesh_path"]),
        "segments_path": str(segments_path),
        "annotations_path": str(annotations_path),
        "semantic_source": semantic_source,
        "num_semantic_classes": int(len(semantic_class_names)),
        "num_vertices": int(len(vertices)),
        "num_faces": int(len(faces)),
        "vertex_properties": mesh.get("vertex_properties", []),
        "objects": {
            str(object_id): value
            for object_id, value in sorted(objects.items(), key=lambda item: item[0])
        },
        "frames": [],
    }

    for frame_name in tqdm(frame_names, desc=f"{scene_id} frame", leave=False):
        frame_record = _process_frame(
            frame_name=frame_name,
            image_dir=image_dir,
            image_by_name=image_by_name,
            cameras=cameras,
            vertices=vertices,
            faces=faces,
            face_semantic_ids=face_semantic_ids,
            face_instance_ids=face_instance_ids,
            semantic_dir=semantic_dir,
            instance_dir=instance_dir,
            pointmap_dir=pointmap_dir,
            raster_dir=raster_dir,
            viz_dir=viz_dir,
            objects=objects,
            config=config,
        )
        if frame_record is not None:
            scene_manifest["frames"].append(frame_record)

    write_json(out_scene_dir / "scene_manifest.json", scene_manifest)
    return scene_manifest


def _process_frame(
    *,
    frame_name: str,
    image_dir: Path,
    image_by_name: Dict[str, Any],
    cameras: Dict[int, Any],
    vertices: np.ndarray,
    faces: np.ndarray,
    face_semantic_ids: np.ndarray,
    face_instance_ids: np.ndarray,
    semantic_dir: Path,
    instance_dir: Path,
    pointmap_dir: Path,
    raster_dir: Path,
    viz_dir: Path,
    objects: Dict[int, Any],
    config: ScanNetPP2DConfig,
) -> Optional[Dict[str, Any]]:
    import cv2

    image_path = image_dir / Path(frame_name).name
    if not image_path.is_file():
        image_path = image_dir / frame_name
    if not image_path.is_file():
        print(f"Skipping missing image: {frame_name}")
        return None

    stem = Path(frame_name).stem
    semantic_path = semantic_dir / f"{stem}.png"
    instance_path = instance_dir / f"{stem}.png"
    pointmap_path = pointmap_dir / f"{stem}.npz"
    raster_path = raster_dir / f"{stem}.npz"
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        print(f"Skipping unreadable image: {image_path}")
        return None
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = image_rgb.shape[:2]

    colmap_image = image_by_name.get(frame_name) or image_by_name.get(
        Path(frame_name).name
    )
    if colmap_image is None:
        print(f"Skipping image not present in COLMAP images.txt: {frame_name}")
        return None
    camera = cameras[colmap_image.camera_id].scaled_to(width, height)
    camera_record = _camera_record(colmap_image, camera)

    pointmap_exists = (not config.save_pointmaps) or pointmap_path.exists()
    if (
        config.skip_existing
        and semantic_path.exists()
        and instance_path.exists()
        and pointmap_exists
    ):
        return {
            "image_name": frame_name,
            "image_path": str(image_path),
            "semantic_mask": str(semantic_path),
            "instance_mask": str(instance_path),
            "pointmap": str(pointmap_path) if pointmap_path.exists() else None,
            "raster": str(raster_path) if raster_path.exists() else None,
            "width": int(width),
            "height": int(height),
            **camera_record,
            "skipped_existing": True,
        }

    semantic, instance, pix_to_face, zbuf, pointmap = rasterize_labels_and_points(
        vertices,
        faces,
        face_semantic_ids,
        face_instance_ids,
        colmap_image.world_to_camera,
        camera,
        height=height,
        width=width,
        near=config.near,
        semantic_ignore_label=config.semantic_ignore_label,
    )

    _write_uint16_png(semantic_path, semantic, config.semantic_ignore_label)
    _write_uint16_png(instance_path, instance, 0)

    if config.save_pointmaps:
        pointmap_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            pointmap_path,
            pointmap=pointmap.astype(np.float32),
            valid=np.isfinite(pointmap).all(axis=-1),
        )

    if config.save_raster:
        raster_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            raster_path,
            pix_to_face=pix_to_face.astype(np.int32),
            zbuf=zbuf.astype(np.float32),
        )

    if config.save_visualizations:
        sem_color = colorize_ids(semantic, ignore_values=(config.semantic_ignore_label,))
        inst_color = colorize_ids(instance, ignore_values=(0,))
        save_rgb(
            viz_dir / "semantic" / f"{stem}.jpg",
            overlay_labels(image_rgb, semantic, sem_color, alpha=0.55),
        )
        save_rgb(
            viz_dir / "instance" / f"{stem}.jpg",
            overlay_labels(image_rgb, instance, inst_color, alpha=0.55),
        )
        save_labeled_summary(
            viz_dir / "summary" / f"{stem}.jpg",
            image_rgb,
            semantic,
            instance,
            objects=objects,
            semantic_ignore_label=config.semantic_ignore_label,
            alpha=0.55,
        )

    valid_semantic = semantic != config.semantic_ignore_label
    valid_instance = instance > 0
    valid_pointmap = np.isfinite(pointmap).all(axis=-1)
    visible_instances = sorted(int(v) for v in np.unique(instance[valid_instance]))
    return {
        "image_name": frame_name,
        "image_path": str(image_path),
        "semantic_mask": str(semantic_path),
        "instance_mask": str(instance_path),
        "pointmap": str(pointmap_path) if config.save_pointmaps else None,
        "raster": str(raster_path) if config.save_raster else None,
        "width": int(width),
        "height": int(height),
        **camera_record,
        "semantic_pixels": int(valid_semantic.sum()),
        "instance_pixels": int(valid_instance.sum()),
        "pointmap_pixels": int(valid_pointmap.sum()),
        "visible_instance_ids": visible_instances,
    }


def _camera_record(colmap_image: Any, camera: Any) -> Dict[str, Any]:
    return {
        "camera_id": int(colmap_image.camera_id),
        "camera_model": str(camera.model),
        "camera_width": int(camera.width),
        "camera_height": int(camera.height),
        "camera_params": [float(v) for v in np.asarray(camera.params).tolist()],
        "intrinsics": _camera_intrinsics(camera).astype(float).tolist(),
        "world_to_camera": colmap_image.world_to_camera.astype(float).tolist(),
    }


def _camera_intrinsics(camera: Any) -> np.ndarray:
    params = np.asarray(camera.params, dtype=np.float64)
    if camera.model == "SIMPLE_PINHOLE":
        f, cx, cy = params[:3]
        fx = fy = f
    elif camera.model in {
        "PINHOLE",
        "OPENCV",
        "OPENCV_FISHEYE",
        "FULL_OPENCV",
        "FOV",
        "THIN_PRISM_FISHEYE",
    }:
        fx, fy, cx, cy = params[:4]
    elif camera.model in {
        "SIMPLE_RADIAL",
        "RADIAL",
        "SIMPLE_RADIAL_FISHEYE",
        "RADIAL_FISHEYE",
    }:
        f, cx, cy = params[:3]
        fx = fy = f
    else:
        raise NotImplementedError(f"Unsupported camera model: {camera.model}")
    return np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _resolve_scene_assets(scene_id: str, config: ScanNetPP2DConfig) -> Dict[str, Any]:
    scene_root = config.data_root / scene_id
    annotation_root = (
        Path(config.annotation_root) / scene_id
        if config.annotation_root is not None
        else scene_root
    )
    annotation_scan_dir = annotation_root / config.annotation_scan_subdir
    image_dir = scene_root / "images"
    colmap_dir = scene_root / "colmap"
    pinhole_mesh_path = scene_root / config.mesh_filename
    annotation_mesh_path = annotation_scan_dir / config.semantic_mesh_filename
    segments_path = annotation_scan_dir / config.segments_filename
    annotations_path = annotation_scan_dir / config.annotations_filename
    return {
        "scene_root": scene_root,
        "annotation_root": annotation_root,
        "image_dir": image_dir,
        "colmap_dir": colmap_dir,
        "pinhole_mesh_path": pinhole_mesh_path,
        "annotation_mesh_path": annotation_mesh_path,
        "segments_path": segments_path,
        "annotations_path": annotations_path,
    }


def _load_instance_data(
    mesh: Dict[str, np.ndarray],
    segments_path: Path,
    annotations_path: Path,
    *,
    vertex_semantic_ids: np.ndarray,
    semantic_class_names: Sequence[str],
    num_vertices: int,
) -> Tuple[np.ndarray, Dict[int, Dict[str, Any]]]:
    mesh_instances = mesh.get("instances")
    if mesh_instances is not None:
        vertex_instance_ids = np.asarray(mesh_instances, dtype=np.int32)
        if len(vertex_instance_ids) != num_vertices:
            raise ValueError(
                f"Mesh instance property length {len(vertex_instance_ids)} does not "
                f"match vertex count {num_vertices}"
            )
        objects = _objects_from_instance_semantics(
            vertex_instance_ids,
            vertex_semantic_ids,
            semantic_class_names,
        )
        return vertex_instance_ids, objects

    instance_data = load_vertex_instances(
        segments_path,
        annotations_path,
        num_vertices,
    )
    return instance_data["vertex_instance_ids"], instance_data["objects"]


def _resolve_vertex_semantics(
    raw_vertex_semantic_ids: np.ndarray,
    vertex_instance_ids: np.ndarray,
    objects: Dict[int, Dict[str, Any]],
    *,
    semantic_class_names: Sequence[str],
    semantic_ignore_label: int,
) -> Tuple[np.ndarray, str, Dict[int, Dict[str, Any]]]:
    """Prefer ScanNet++ object labels as compact semantic class ids.

    The semantic PLY `label` property can contain large sparse annotation ids in
    some releases. Training wants class ids, so we map each annotated object
    label from segments_anno.json into semantic_classes.txt when possible.
    """
    label_to_id = _semantic_label_to_id(semantic_class_names)
    if not label_to_id:
        return raw_vertex_semantic_ids, "mesh_label_property", objects

    vertex_semantic_ids = np.full(
        len(vertex_instance_ids),
        int(semantic_ignore_label),
        dtype=np.int32,
    )
    updated_objects: Dict[int, Dict[str, Any]] = {}
    assigned_objects = 0
    assigned_vertices = 0
    for instance_id, metadata in objects.items():
        item = dict(metadata)
        label = _metadata_label(item)
        semantic_id = label_to_id.get(_normalize_label(label))
        if semantic_id is not None:
            mask = vertex_instance_ids == int(instance_id)
            vertex_semantic_ids[mask] = int(semantic_id)
            assigned_objects += 1
            assigned_vertices += int(mask.sum())
            item["semantic_id"] = int(semantic_id)
            item["label"] = label
        updated_objects[int(instance_id)] = item

    if assigned_objects == 0 or assigned_vertices == 0:
        return raw_vertex_semantic_ids, "mesh_label_property", objects
    return vertex_semantic_ids, "segments_anno_label_to_semantic_classes", updated_objects


def _semantic_label_to_id(names: Sequence[str]) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for idx, name in enumerate(names):
        normalized = _normalize_label(name)
        if normalized and normalized not in mapping:
            mapping[normalized] = int(idx)
    return mapping


def _metadata_label(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in (
            "label",
            "label_name",
            "labelName",
            "class",
            "class_name",
            "category",
            "category_name",
            "rawLabel",
        ):
            item = value.get(key)
            if isinstance(item, str) and item.strip():
                return item.strip()
    return ""


def _normalize_label(value: str) -> str:
    return " ".join(str(value).replace("_", " ").replace("-", " ").lower().split())


def _objects_from_instance_semantics(
    vertex_instance_ids: np.ndarray,
    vertex_semantic_ids: np.ndarray,
    semantic_class_names: Sequence[str],
) -> Dict[int, Dict[str, Any]]:
    objects: Dict[int, Dict[str, Any]] = {}
    for instance_id in sorted(int(v) for v in np.unique(vertex_instance_ids) if v > 0):
        mask = vertex_instance_ids == instance_id
        semantic_values = vertex_semantic_ids[mask]
        semantic_values = semantic_values[
            (semantic_values >= 0) & (semantic_values != 65535)
        ]
        semantic_id = -1
        if semantic_values.size:
            values, counts = np.unique(semantic_values, return_counts=True)
            semantic_id = int(values[int(np.argmax(counts))])
        label = _semantic_name(semantic_id, semantic_class_names)
        objects[instance_id] = {
            "objectId": int(instance_id),
            "semantic_id": int(semantic_id),
            "label": label or f"instance_{instance_id}",
            "num_vertices": int(mask.sum()),
        }
    return objects


def _load_semantic_class_names(config: ScanNetPP2DConfig) -> List[str]:
    candidates: List[Path] = []
    if config.semantic_classes_file is not None:
        candidates.append(Path(config.semantic_classes_file))
    metadata_root = (
        Path(config.metadata_root)
        if config.metadata_root is not None
        else config.data_root.parent / "metadata"
    )
    candidates.append(metadata_root / "semantic_classes.txt")
    for path in candidates:
        if path.is_file():
            return read_text_list(path)
    return []


def _semantic_name(semantic_id: int, names: Sequence[str]) -> str:
    if semantic_id < 0 or not names:
        return ""
    if semantic_id < len(names):
        return names[semantic_id]
    if 1 <= semantic_id <= len(names):
        return names[semantic_id - 1]
    return ""


def _resolve_scene_ids(config: ScanNetPP2DConfig) -> List[str]:
    if config.scene_ids:
        scene_ids = [_clean_scene_id(scene_id) for scene_id in config.scene_ids]
    elif config.scene_list:
        scene_ids = [
            _clean_scene_id(scene_id)
            for scene_id in read_text_list(Path(config.scene_list))
        ]
    else:
        scene_ids = discover_scene_ids(config.data_root)

    seen = set()
    unique = []
    for scene_id in scene_ids:
        if not scene_id:
            continue
        if scene_id in seen:
            continue
        seen.add(scene_id)
        unique.append(scene_id)
    if config.limit_scenes is not None:
        unique = unique[: config.limit_scenes]
    if not unique:
        raise ValueError("No scenes selected for preprocessing.")
    return unique


def _clean_scene_id(value: str) -> str:
    return str(value).strip().split()[0] if str(value).strip() else ""


def _select_frame_names(
    image_dir: Path,
    colmap_images: Sequence[Any],
    config: ScanNetPP2DConfig,
) -> List[str]:
    names = [image.name for image in colmap_images]
    seen = set()
    existing = []
    for name in names:
        key = Path(name).name
        if key in seen:
            continue
        seen.add(key)
        if (image_dir / Path(name).name).is_file() or (image_dir / name).is_file():
            existing.append(name)

    if config.frame_step <= 0:
        raise ValueError(f"frame_step must be positive, got {config.frame_step}")
    existing = existing[:: config.frame_step]
    if config.max_frames is not None:
        existing = existing[: config.max_frames]
    return existing


def _write_uint16_png(path: Path, values: np.ndarray, invalid_value: int) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(values)
    if values.min(initial=0) < 0:
        values = np.where(values < 0, invalid_value, values)
    max_value = int(values.max(initial=0))
    if max_value > np.iinfo(np.uint16).max:
        raise ValueError(
            f"Cannot write {path} as uint16 PNG; max label {max_value} is too large."
        )
    ok = cv2.imwrite(str(path), values.astype(np.uint16))
    if not ok:
        raise IOError(f"Failed to write PNG: {path}")


def _jsonable_config(config: ScanNetPP2DConfig) -> Dict[str, Any]:
    result = asdict(config)
    for key, value in list(result.items()):
        if isinstance(value, Path):
            result[key] = str(value)
        elif isinstance(value, list):
            result[key] = [
                str(item) if isinstance(item, Path) else item for item in value
            ]
    return result


def _load_yaml_config(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    import yaml

    with Path(path).open("r", encoding="utf8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return payload


def _build_config(args: argparse.Namespace) -> ScanNetPP2DConfig:
    payload = _load_yaml_config(args.config)
    valid_fields = {field.name for field in fields(ScanNetPP2DConfig)}
    payload = {key: value for key, value in payload.items() if key in valid_fields}

    for key in valid_fields:
        if not hasattr(args, key):
            continue
        value = getattr(args, key)
        if value is not None:
            payload[key] = value

    if "data_root" not in payload or payload["data_root"] is None:
        raise ValueError("--data-root is required unless provided by --config")
    if "output_root" not in payload or payload["output_root"] is None:
        raise ValueError("--output-root is required unless provided by --config")

    payload["data_root"] = Path(payload["data_root"])
    payload["output_root"] = Path(payload["output_root"])
    if payload.get("annotation_root"):
        payload["annotation_root"] = Path(payload["annotation_root"])
    if payload.get("scene_list"):
        payload["scene_list"] = Path(payload["scene_list"])
    if payload.get("metadata_root"):
        payload["metadata_root"] = Path(payload["metadata_root"])
    if payload.get("semantic_classes_file"):
        payload["semantic_classes_file"] = Path(payload["semantic_classes_file"])
    payload["scene_ids"] = list(payload.get("scene_ids") or [])

    return ScanNetPP2DConfig(**payload)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rasterize ScanNet++ pinhole mesh semantic/instance labels to images."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--scene-ids", nargs="*", default=None)
    parser.add_argument("--scene-list", type=Path, default=None)
    parser.add_argument("--limit-scenes", type=int, default=None)
    parser.add_argument("--mesh-filename", default=None)
    parser.add_argument("--annotation-root", type=Path, default=None)
    parser.add_argument("--annotation-scan-subdir", default=None)
    parser.add_argument("--semantic-mesh-filename", default=None)
    parser.add_argument("--segments-filename", default=None)
    parser.add_argument("--annotations-filename", default=None)
    parser.add_argument("--metadata-root", type=Path, default=None)
    parser.add_argument("--semantic-classes-file", type=Path, default=None)
    parser.add_argument("--frame-step", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--near", type=float, default=None)
    parser.add_argument("--semantic-ignore-label", type=int, default=None)

    viz = parser.add_mutually_exclusive_group()
    viz.add_argument(
        "--save-visualizations", dest="save_visualizations", action="store_true"
    )
    viz.add_argument(
        "--no-visualizations", dest="save_visualizations", action="store_false"
    )
    parser.set_defaults(save_visualizations=None)

    raster = parser.add_mutually_exclusive_group()
    raster.add_argument("--save-raster", dest="save_raster", action="store_true")
    raster.add_argument("--no-raster", dest="save_raster", action="store_false")
    parser.set_defaults(save_raster=None)

    pointmaps = parser.add_mutually_exclusive_group()
    pointmaps.add_argument(
        "--save-pointmaps", dest="save_pointmaps", action="store_true"
    )
    pointmaps.add_argument(
        "--no-pointmaps", dest="save_pointmaps", action="store_false"
    )
    parser.set_defaults(save_pointmaps=None)

    skip = parser.add_mutually_exclusive_group()
    skip.add_argument("--skip-existing", dest="skip_existing", action="store_true")
    skip.add_argument("--overwrite", dest="skip_existing", action="store_false")
    parser.set_defaults(skip_existing=None)
    parser.add_argument("--dry-run", action="store_true", default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    config = _build_config(args)
    print(json.dumps(_jsonable_config(config), indent=2, sort_keys=True))
    prepare_scannetpp_2d(config)
