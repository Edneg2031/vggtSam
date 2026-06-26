"""Small COLMAP text-model reader and projector.

The ScanNet++ DSLR release stores aligned COLMAP cameras in text format. COLMAP
poses are world-to-camera transforms, so a mesh vertex X is projected as
R @ X + t.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


@dataclass(frozen=True)
class Camera:
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray

    def scaled_to(self, width: int, height: int) -> "Camera":
        """Return a camera whose intrinsics match a resized image."""
        if width == self.width and height == self.height:
            return self

        params = self.params.astype(np.float64).copy()
        sx = width / float(self.width)
        sy = height / float(self.height)

        if self.model in {
            "PINHOLE",
            "OPENCV",
            "OPENCV_FISHEYE",
            "FULL_OPENCV",
            "FOV",
            "THIN_PRISM_FISHEYE",
        }:
            params[0] *= sx
            params[1] *= sy
            params[2] *= sx
            params[3] *= sy
        elif self.model in {
            "SIMPLE_PINHOLE",
            "SIMPLE_RADIAL",
            "RADIAL",
            "SIMPLE_RADIAL_FISHEYE",
            "RADIAL_FISHEYE",
        }:
            if abs(sx - sy) > 1e-6:
                raise ValueError(
                    f"Cannot anisotropically resize {self.model}: "
                    f"{self.width}x{self.height} -> {width}x{height}"
                )
            params[0] *= sx
            params[1] *= sx
            params[2] *= sy
        else:
            raise NotImplementedError(f"Unsupported camera model: {self.model}")

        return replace(self, width=width, height=height, params=params)


@dataclass(frozen=True)
class Image:
    id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str

    @property
    def world_to_camera(self) -> np.ndarray:
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = qvec_to_rotmat(self.qvec)
        transform[:3, 3] = self.tvec
        return transform


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    qvec = np.asarray(qvec, dtype=np.float64)
    qvec = qvec / np.linalg.norm(qvec)
    w, x, y, z = qvec
    return np.array(
        [
            [
                1.0 - 2.0 * y * y - 2.0 * z * z,
                2.0 * x * y - 2.0 * w * z,
                2.0 * z * x + 2.0 * w * y,
            ],
            [
                2.0 * x * y + 2.0 * w * z,
                1.0 - 2.0 * x * x - 2.0 * z * z,
                2.0 * y * z - 2.0 * w * x,
            ],
            [
                2.0 * z * x - 2.0 * w * y,
                2.0 * y * z + 2.0 * w * x,
                1.0 - 2.0 * x * x - 2.0 * y * y,
            ],
        ],
        dtype=np.float64,
    )


def read_cameras_text(path: Path) -> Dict[int, Camera]:
    cameras: Dict[int, Camera] = {}
    with path.open("r", encoding="utf8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = np.array([float(v) for v in parts[4:]], dtype=np.float64)
            cameras[camera_id] = Camera(camera_id, model, width, height, params)
    return cameras


def read_images_text(path: Path) -> Dict[int, Image]:
    images: Dict[int, Image] = {}
    with path.open("r", encoding="utf8") as handle:
        lines = [
            line.strip()
            for line in handle
            if line.strip() and not line.startswith("#")
        ]

    i = 0
    while i < len(lines):
        parts = lines[i].split()
        if len(parts) < 10:
            i += 1
            continue

        image_id = int(parts[0])
        qvec = np.array([float(v) for v in parts[1:5]], dtype=np.float64)
        tvec = np.array([float(v) for v in parts[5:8]], dtype=np.float64)
        camera_id = int(parts[8])
        name = " ".join(parts[9:])
        images[image_id] = Image(image_id, qvec, tvec, camera_id, name)
        i += 2
    return images


def read_colmap_text_model(
    colmap_dir: Path,
) -> Tuple[Dict[int, Camera], Dict[int, Image]]:
    colmap_dir = resolve_colmap_text_dir(colmap_dir)
    cameras_path = colmap_dir / "cameras.txt"
    images_path = colmap_dir / "images.txt"
    if not cameras_path.is_file():
        raise FileNotFoundError(f"Missing COLMAP cameras file: {cameras_path}")
    if not images_path.is_file():
        raise FileNotFoundError(f"Missing COLMAP images file: {images_path}")
    return read_cameras_text(cameras_path), read_images_text(images_path)


def resolve_colmap_text_dir(colmap_dir: Path) -> Path:
    """Find a COLMAP text model directory under common ScanNet++ layouts."""
    candidates = [
        colmap_dir,
        colmap_dir / "0",
        colmap_dir / "sparse" / "0",
        colmap_dir / "text",
    ]
    for candidate in candidates:
        if (candidate / "cameras.txt").is_file() and (
            candidate / "images.txt"
        ).is_file():
            return candidate
    if colmap_dir.is_dir():
        for candidate in sorted(colmap_dir.rglob("cameras.txt")):
            model_dir = candidate.parent
            if (model_dir / "images.txt").is_file():
                return model_dir
    return colmap_dir


def ordered_images(images: Dict[int, Image]) -> List[Image]:
    return [images[k] for k in sorted(images.keys())]


def project_points(
    points_world: np.ndarray,
    world_to_camera: np.ndarray,
    camera: Camera,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project world-space points to image coordinates.

    Returns:
        uv: float array shaped [N, 2].
        z: camera-space depth shaped [N].
    """
    points_world = np.asarray(points_world, dtype=np.float64)
    points_h = np.concatenate(
        [points_world, np.ones((points_world.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    points_cam = (world_to_camera @ points_h.T).T[:, :3]
    z = points_cam[:, 2]

    eps = 1e-12
    x = points_cam[:, 0] / np.where(np.abs(z) < eps, eps, z)
    y = points_cam[:, 1] / np.where(np.abs(z) < eps, eps, z)

    uv = _project_normalized(x, y, camera)
    return uv.astype(np.float32), z.astype(np.float32)


def _project_normalized(x: np.ndarray, y: np.ndarray, camera: Camera) -> np.ndarray:
    model = camera.model
    p = camera.params

    if model == "PINHOLE":
        fx, fy, cx, cy = p[:4]
        u = fx * x + cx
        v = fy * y + cy
    elif model == "SIMPLE_PINHOLE":
        f, cx, cy = p[:3]
        u = f * x + cx
        v = f * y + cy
    elif model == "OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2 = p[:8]
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2 * r2
        xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        u = fx * xd + cx
        v = fy * yd + cy
    elif model == "OPENCV_FISHEYE":
        fx, fy, cx, cy, k1, k2, k3, k4 = p[:8]
        r = np.sqrt(x * x + y * y)
        theta = np.arctan(r)
        theta2 = theta * theta
        theta4 = theta2 * theta2
        theta6 = theta4 * theta2
        theta8 = theta4 * theta4
        theta_d = theta * (
            1.0 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8
        )
        scale = np.ones_like(r)
        valid = r > 1e-12
        scale[valid] = theta_d[valid] / r[valid]
        u = fx * x * scale + cx
        v = fy * y * scale + cy
    else:
        raise NotImplementedError(f"Unsupported camera model: {model}")

    return np.stack([u, v], axis=1)


def image_lookup(images: Iterable[Image]) -> Dict[str, Image]:
    """Map both full COLMAP names and basenames to image entries."""
    lookup: Dict[str, Image] = {}
    for image in images:
        lookup[image.name] = image
        lookup[Path(image.name).name] = image
    return lookup
