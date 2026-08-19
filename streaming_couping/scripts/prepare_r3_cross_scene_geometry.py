#!/usr/bin/env python3
"""Prepare one annotation-free ScanNet++ scene for frozen r3 validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image as PILImage

from vggtsam.data.scannetpp.colmap import (
    Camera,
    Image as ColmapImage,
    ordered_images,
    read_colmap_text_model,
)
from vggtsam.data.scannetpp.io import load_ply_mesh
from vggtsam.data.scannetpp.rasterize import (
    has_numba,
    rasterize_labels_and_points,
)


REVISION = "r3_frozen_cross_scene_geometry_preprocess_r1"
DEVELOPMENT_SCENE = "00a231a370"
DEFAULT_SCENE = "0a184cf634"
DEFAULT_START = 90
DEFAULT_STRIDE = 15
DEFAULT_FRAME_COUNT = 30
MINIMUM_VALID_PIXELS = 128


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        summary = prepare_geometry_scene(
            source_root=Path(args.source_root).expanduser().resolve(),
            output_dir=output_dir,
            scene_id=str(args.scene_id),
            start=int(args.start),
            stride=int(args.stride),
            frame_count=int(args.frame_count),
            near=float(args.near),
            overwrite=bool(args.overwrite),
        )
    except Exception as error:
        _write_failure_report(output_dir, error)
        raise
    print("R3 FROZEN CROSS-SCENE GEOMETRY PREPARATION")
    print(
        f"  scene={summary['scene_id']} frames={summary['frame_count']} "
        f"generated={summary['generated_pointmaps']} "
        f"reused={summary['reused_pointmaps']}"
    )
    print(f"  manifest={summary['outputs']['manifest']}")
    print(f"  copyable_report={summary['outputs']['copyable_report']}")


def prepare_geometry_scene(
    *,
    source_root: Path,
    output_dir: Path,
    scene_id: str,
    start: int,
    stride: int,
    frame_count: int,
    near: float,
    overwrite: bool,
) -> dict[str, Any]:
    """Rasterize a fixed RGB/COLMAP clip without annotation or SAM inputs."""

    scene = str(scene_id).strip()
    if not scene or scene == DEVELOPMENT_SCENE:
        raise ValueError("r3 requires a non-empty scene different from development.")
    source_indices = fixed_source_indices(
        start=start,
        stride=stride,
        frame_count=frame_count,
    )
    if near <= 0.0:
        raise ValueError("near must be positive.")
    if not has_numba():
        raise RuntimeError(
            "numba is required for the 30-frame geometry rasterization."
        )

    scene_root = source_root / "data" / scene
    image_dir = scene_root / "images"
    colmap_dir = scene_root / "colmap"
    mesh_path = scene_root / "mesh_aligned_0.05.ply"
    for label, path in (
        ("scene", scene_root),
        ("images", image_dir),
        ("colmap", colmap_dir),
    ):
        if not path.is_dir():
            raise FileNotFoundError(f"Missing source {label} directory: {path}")
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Missing source mesh: {mesh_path}")

    cameras, image_lookup = read_colmap_text_model(colmap_dir)
    available = available_colmap_images(
        ordered_images(image_lookup),
        image_dir=image_dir,
    )
    selected = select_fixed_clip(available, source_indices=source_indices)
    mesh = load_ply_mesh(mesh_path)
    vertices = np.asarray(mesh["vertices"], dtype=np.float32)
    faces = np.asarray(mesh["faces"], dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError(f"Source mesh has invalid vertices: {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError(f"Source mesh has invalid triangle faces: {faces.shape}")
    dummy_face_labels = np.zeros(len(faces), dtype=np.int32)

    pointmap_dir = output_dir / scene / "pointmaps"
    pointmap_dir.mkdir(parents=True, exist_ok=True)
    frame_rows: list[dict[str, Any]] = []
    generated = 0
    reused = 0
    for sequence_index, (source_index, colmap_image, image_path) in enumerate(selected):
        with PILImage.open(image_path) as image:
            width, height = image.size
        camera = cameras.get(int(colmap_image.camera_id))
        if camera is None:
            raise ValueError(
                f"COLMAP image {colmap_image.id} references missing camera "
                f"{colmap_image.camera_id}."
            )
        camera = camera.scaled_to(width, height)
        stem = Path(colmap_image.name).stem
        pointmap_path = pointmap_dir / f"{sequence_index:02d}_{stem}.npz"
        valid_pixels = None
        if pointmap_path.is_file() and not overwrite:
            valid_pixels = _validate_existing_pointmap(
                pointmap_path,
                expected_shape=(height, width, 3),
            )
            reused += 1
        else:
            _, _, _, _, pointmap = rasterize_labels_and_points(
                vertices,
                faces,
                dummy_face_labels,
                dummy_face_labels,
                colmap_image.world_to_camera,
                camera,
                height=height,
                width=width,
                near=near,
                semantic_ignore_label=0,
            )
            valid = np.isfinite(pointmap).all(axis=-1)
            valid_pixels = int(valid.sum())
            if valid_pixels < MINIMUM_VALID_PIXELS:
                raise RuntimeError(
                    f"Frame source_index={source_index} produced only "
                    f"{valid_pixels} valid pointmap pixels."
                )
            _write_pointmap(pointmap_path, pointmap, valid)
            generated += 1
        frame_rows.append(
            {
                "sequence_index": sequence_index,
                "source_sequence_index": source_index,
                "source_colmap_image_id": int(colmap_image.id),
                "image_name": str(colmap_image.name),
                "image_path": str(image_path),
                "image_source_path": str(image_path),
                "image_storage": "source_reference",
                "pointmap": str(pointmap_path),
                "width": width,
                "height": height,
                "valid_pointmap_pixels": int(valid_pixels),
                "valid_pointmap_ratio": float(valid_pixels / (height * width)),
                **camera_record(colmap_image, camera),
            }
        )

    manifest = {
        "schema": 1,
        "dataset": "scannetpp",
        "revision": REVISION,
        "role": "r3_frozen_cross_scene_geometry_only",
        "source_root": str(source_root),
        "annotations_read": 0,
        "semantic_masks_generated": 0,
        "instance_masks_generated": 0,
        "sam_loaded_or_run": 0,
        "scenes": [
            {
                "scene_id": scene,
                "scene_root": str(scene_root),
                "mesh_path": str(mesh_path),
                "source_available_frame_count": len(available),
                "source_frame_indices": source_indices,
                # Downstream isolated-manifest indices. Their source positions
                # remain explicit above and in every frame record.
                "frame_indices": tuple(range(len(frame_rows))),
                "frames": frame_rows,
            }
        ],
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    summary = {
        "schema": 1,
        "revision": REVISION,
        "status": "READY",
        "role": "r3_frozen_cross_scene_geometry_only",
        "scene_id": scene,
        "development_scene_id": DEVELOPMENT_SCENE,
        "heldout_scene_overlap": 0,
        "frame_count": len(frame_rows),
        "source_frame_start": start,
        "source_frame_stride": stride,
        "source_frame_indices": source_indices,
        "manifest_frame_indices": tuple(range(len(frame_rows))),
        "selection_rule": "fixed_colmap_order_indices_matching_development_clip",
        "source_available_frame_count": len(available),
        "source_total_colmap_image_count": len(image_lookup),
        "mesh_path": str(mesh_path),
        "mesh_size_bytes": mesh_path.stat().st_size,
        "mesh_sha256": _sha256_file(mesh_path),
        "mesh_vertices": int(len(vertices)),
        "mesh_faces": int(len(faces)),
        "generated_pointmaps": generated,
        "reused_pointmaps": reused,
        "minimum_valid_pointmap_pixels": min(
            int(row["valid_pointmap_pixels"]) for row in frame_rows
        ),
        "minimum_valid_pointmap_ratio": min(
            float(row["valid_pointmap_ratio"]) for row in frame_rows
        ),
        "annotations_read": 0,
        "semantic_masks_generated": 0,
        "instance_masks_generated": 0,
        "sam_loaded_or_run": 0,
        "streamvggt_loaded_or_run": 0,
        "gt_pointmap_values_used_for_frame_selection": 0,
        "model_parameters_updated": 0,
        "frames": frame_rows,
        "outputs": {
            "manifest": str(manifest_path),
            "summary": str(output_dir / "summary.json"),
            "copyable_report": str(output_dir / "copyable_result.txt"),
            "run_log": str(output_dir / "run.log"),
            "pointmap_dir": str(pointmap_dir),
        },
    }
    _write_json(output_dir / "summary.json", summary)
    _write_copyable(output_dir / "copyable_result.txt", summary)
    return summary


def fixed_source_indices(
    *,
    start: int = DEFAULT_START,
    stride: int = DEFAULT_STRIDE,
    frame_count: int = DEFAULT_FRAME_COUNT,
) -> tuple[int, ...]:
    if start < 0 or stride < 1 or frame_count < 2:
        raise ValueError("Invalid fixed source-frame protocol.")
    return tuple(start + offset * stride for offset in range(frame_count))


def available_colmap_images(
    images: Sequence[ColmapImage],
    *,
    image_dir: Path,
) -> list[tuple[int, ColmapImage, Path]]:
    """Mirror the existing preprocessor's COLMAP order and file filtering."""

    available = []
    seen: set[str] = set()
    for image in images:
        key = Path(image.name).name
        if key in seen:
            continue
        seen.add(key)
        path = image_dir / key
        if not path.is_file():
            path = image_dir / image.name
        if path.is_file():
            available.append((len(available), image, path.resolve()))
    return available


def select_fixed_clip(
    available: Sequence[tuple[int, ColmapImage, Path]],
    *,
    source_indices: Sequence[int],
) -> list[tuple[int, ColmapImage, Path]]:
    if not source_indices:
        raise ValueError("The fixed source-frame list is empty.")
    maximum = max(int(value) for value in source_indices)
    if maximum >= len(available):
        raise ValueError(
            f"Fixed r3 protocol needs source index {maximum}, but only "
            f"{len(available)} ordered RGB/COLMAP frames are available."
        )
    return [available[int(index)] for index in source_indices]


def camera_record(image: ColmapImage, camera: Camera) -> dict[str, Any]:
    return {
        "camera_id": int(image.camera_id),
        "camera_model": str(camera.model),
        "camera_width": int(camera.width),
        "camera_height": int(camera.height),
        "camera_params": [float(value) for value in camera.params.tolist()],
        "intrinsics": camera_intrinsics(camera).tolist(),
        "world_to_camera": image.world_to_camera.astype(float).tolist(),
    }


def camera_intrinsics(camera: Camera) -> np.ndarray:
    params = np.asarray(camera.params, dtype=np.float64)
    if camera.model == "SIMPLE_PINHOLE":
        focal, cx, cy = params[:3]
        fx = fy = focal
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
        focal, cx, cy = params[:3]
        fx = fy = focal
    else:
        raise NotImplementedError(f"Unsupported camera model: {camera.model}")
    return np.asarray(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _validate_existing_pointmap(
    path: Path,
    *,
    expected_shape: tuple[int, int, int],
) -> int:
    with np.load(path) as payload:
        if "pointmap" not in payload.files:
            raise ValueError(f"Existing pointmap lacks 'pointmap': {path}")
        pointmap = np.asarray(payload["pointmap"])
    if pointmap.shape != expected_shape:
        raise ValueError(
            f"Existing pointmap shape {pointmap.shape} != {expected_shape}: {path}"
        )
    valid_pixels = int(np.isfinite(pointmap).all(axis=-1).sum())
    if valid_pixels < MINIMUM_VALID_PIXELS:
        raise ValueError(
            f"Existing pointmap has only {valid_pixels} valid pixels: {path}"
        )
    return valid_pixels


def _write_pointmap(path: Path, pointmap: np.ndarray, valid: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            pointmap=pointmap.astype(np.float32),
            valid=valid.astype(bool),
        )
    temporary.replace(path)


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf8",
    )
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_copyable(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "===== COPYABLE_R3_GEOMETRY_PREPARE_BEGIN =====",
        f"revision={summary['revision']}",
        f"status={summary['status']}",
        f"scene_id={summary['scene_id']}",
        f"development_scene_id={summary['development_scene_id']}",
        "heldout_scene_overlap=0",
        f"frame_count={summary['frame_count']}",
        f"source_frame_start={summary['source_frame_start']}",
        f"source_frame_stride={summary['source_frame_stride']}",
        "source_frame_indices="
        + " ".join(str(value) for value in summary["source_frame_indices"]),
        "manifest_frame_indices="
        + " ".join(str(value) for value in summary["manifest_frame_indices"]),
        f"selection_rule={summary['selection_rule']}",
        "annotations_read=0",
        "semantic_masks_generated=0",
        "instance_masks_generated=0",
        "sam_loaded_or_run=0",
        "streamvggt_loaded_or_run=0",
        "gt_pointmap_values_used_for_frame_selection=0",
        "model_parameters_updated=0",
        f"mesh_sha256={summary['mesh_sha256']}",
        f"mesh_vertices={summary['mesh_vertices']}",
        f"mesh_faces={summary['mesh_faces']}",
        f"generated_pointmaps={summary['generated_pointmaps']}",
        f"reused_pointmaps={summary['reused_pointmaps']}",
        f"minimum_valid_pointmap_pixels={summary['minimum_valid_pointmap_pixels']}",
        f"minimum_valid_pointmap_ratio={summary['minimum_valid_pointmap_ratio']}",
        "",
        "sequence_index,source_sequence_index,colmap_image_id,image_name,valid_pixels,valid_ratio",
    ]
    for row in summary["frames"]:
        lines.append(
            ",".join(
                str(value)
                for value in (
                    row["sequence_index"],
                    row["source_sequence_index"],
                    row["source_colmap_image_id"],
                    row["image_name"],
                    row["valid_pointmap_pixels"],
                    row["valid_pointmap_ratio"],
                )
            )
        )
    lines.extend(
        (
            "",
            "outputs:",
            f"manifest={summary['outputs']['manifest']}",
            f"summary={summary['outputs']['summary']}",
            f"pointmap_dir={summary['outputs']['pointmap_dir']}",
            f"run_log={summary['outputs']['run_log']}",
            f"copyable_report={summary['outputs']['copyable_report']}",
            "===== COPYABLE_R3_GEOMETRY_PREPARE_END =====",
        )
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def _write_failure_report(output_dir: Path, error: Exception) -> None:
    path = output_dir / "copyable_result.txt"
    lines = [
        "===== COPYABLE_R3_GEOMETRY_PREPARE_BEGIN =====",
        f"revision={REVISION}",
        "status=FAILED",
        f"error_type={type(error).__name__}",
        f"error={str(error).replace(chr(10), ' ')}",
        f"run_log={output_dir / 'run.log'}",
        f"copyable_report={path}",
        "===== COPYABLE_R3_GEOMETRY_PREPARE_END =====",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        default="/data184/open_source/vggtSam/source/scanet_pp_pinhole",
    )
    parser.add_argument("--scene-id", default=DEFAULT_SCENE)
    parser.add_argument(
        "--output-dir",
        default=(
            "/data184/open_source/vggtSam/data/processed/"
            "residual_cross_scene_r3"
        ),
    )
    parser.add_argument("--start", type=int, default=DEFAULT_START)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--frame-count", type=int, default=DEFAULT_FRAME_COUNT)
    parser.add_argument("--near", type=float, default=0.001)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
