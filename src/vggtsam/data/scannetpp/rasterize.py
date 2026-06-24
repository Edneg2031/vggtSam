"""CPU z-buffer rasterization for projected ScanNet++ meshes."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .colmap import Camera, project_points

try:
    from numba import njit
except Exception:  # pragma: no cover - optional dependency
    njit = None


def rasterize_labels(
    vertices: np.ndarray,
    faces: np.ndarray,
    face_semantic_ids: np.ndarray,
    face_instance_ids: np.ndarray,
    world_to_camera: np.ndarray,
    camera: Camera,
    *,
    height: int,
    width: int,
    near: float,
    semantic_ignore_label: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rasterize face semantic and instance IDs to an image grid."""
    uv, z = project_points(vertices, world_to_camera, camera)
    tri_uv = uv[faces]
    tri_z = z[faces]

    finite = np.isfinite(tri_uv).all(axis=(1, 2)) & np.isfinite(tri_z).all(axis=1)
    in_front = (tri_z > near).all(axis=1)

    min_xy = tri_uv.min(axis=1)
    max_xy = tri_uv.max(axis=1)
    intersects = (
        (max_xy[:, 0] >= 0)
        & (max_xy[:, 1] >= 0)
        & (min_xy[:, 0] < width)
        & (min_xy[:, 1] < height)
    )

    keep = finite & in_front & intersects
    if not np.any(keep):
        return (
            np.full((height, width), semantic_ignore_label, dtype=np.int32),
            np.zeros((height, width), dtype=np.int32),
            np.full((height, width), -1, dtype=np.int32),
            np.full((height, width), np.inf, dtype=np.float32),
        )

    face_ids = np.nonzero(keep)[0].astype(np.int32)
    tri_data = np.concatenate([tri_uv[keep], tri_z[keep, :, None]], axis=2).astype(
        np.float32
    )
    sem = face_semantic_ids[keep].astype(np.int32)
    inst = face_instance_ids[keep].astype(np.int32)

    return _rasterize_dispatch(
        tri_data,
        sem,
        inst,
        face_ids,
        height,
        width,
        int(semantic_ignore_label),
    )


def _rasterize_dispatch(
    triangles: np.ndarray,
    semantic_ids: np.ndarray,
    instance_ids: np.ndarray,
    face_ids: np.ndarray,
    height: int,
    width: int,
    semantic_ignore_label: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    zbuf = np.full((height, width), np.inf, dtype=np.float32)
    semantic = np.full((height, width), semantic_ignore_label, dtype=np.int32)
    instance = np.zeros((height, width), dtype=np.int32)
    pix_to_face = np.full((height, width), -1, dtype=np.int32)

    if _rasterize_triangles_numba is not None:
        _rasterize_triangles_numba(
            triangles,
            semantic_ids,
            instance_ids,
            face_ids,
            zbuf,
            semantic,
            instance,
            pix_to_face,
        )
    else:
        _rasterize_triangles_python(
            triangles,
            semantic_ids,
            instance_ids,
            face_ids,
            zbuf,
            semantic,
            instance,
            pix_to_face,
        )
    return semantic, instance, pix_to_face, zbuf


def _rasterize_triangles_python(
    triangles: np.ndarray,
    semantic_ids: np.ndarray,
    instance_ids: np.ndarray,
    face_ids: np.ndarray,
    zbuf: np.ndarray,
    semantic: np.ndarray,
    instance: np.ndarray,
    pix_to_face: np.ndarray,
) -> None:
    height, width = zbuf.shape
    for tri_idx in range(triangles.shape[0]):
        _rasterize_one(
            triangles[tri_idx],
            int(semantic_ids[tri_idx]),
            int(instance_ids[tri_idx]),
            int(face_ids[tri_idx]),
            zbuf,
            semantic,
            instance,
            pix_to_face,
            height,
            width,
        )


def _rasterize_one(
    tri: np.ndarray,
    semantic_id: int,
    instance_id: int,
    face_id: int,
    zbuf: np.ndarray,
    semantic: np.ndarray,
    instance: np.ndarray,
    pix_to_face: np.ndarray,
    height: int,
    width: int,
) -> None:
    x0, y0, z0 = float(tri[0, 0]), float(tri[0, 1]), float(tri[0, 2])
    x1, y1, z1 = float(tri[1, 0]), float(tri[1, 1]), float(tri[1, 2])
    x2, y2, z2 = float(tri[2, 0]), float(tri[2, 1]), float(tri[2, 2])

    min_x = max(int(np.floor(min(x0, x1, x2))), 0)
    max_x = min(int(np.ceil(max(x0, x1, x2))), width - 1)
    min_y = max(int(np.floor(min(y0, y1, y2))), 0)
    max_y = min(int(np.ceil(max(y0, y1, y2))), height - 1)
    if min_x > max_x or min_y > max_y:
        return

    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(denom) < 1e-8:
        return

    for py in range(min_y, max_y + 1):
        yy = py + 0.5
        for px in range(min_x, max_x + 1):
            xx = px + 0.5
            w0 = ((y1 - y2) * (xx - x2) + (x2 - x1) * (yy - y2)) / denom
            w1 = ((y2 - y0) * (xx - x2) + (x0 - x2) * (yy - y2)) / denom
            w2 = 1.0 - w0 - w1
            if w0 < -1e-5 or w1 < -1e-5 or w2 < -1e-5:
                continue
            depth = w0 * z0 + w1 * z1 + w2 * z2
            if depth <= 0.0 or depth >= zbuf[py, px]:
                continue
            zbuf[py, px] = depth
            semantic[py, px] = semantic_id
            instance[py, px] = instance_id
            pix_to_face[py, px] = face_id


if njit is not None:

    @njit(cache=True)
    def _rasterize_triangles_numba(
        triangles,
        semantic_ids,
        instance_ids,
        face_ids,
        zbuf,
        semantic,
        instance,
        pix_to_face,
    ):
        height = zbuf.shape[0]
        width = zbuf.shape[1]
        for tri_idx in range(triangles.shape[0]):
            tri = triangles[tri_idx]
            x0 = float(tri[0, 0])
            y0 = float(tri[0, 1])
            z0 = float(tri[0, 2])
            x1 = float(tri[1, 0])
            y1 = float(tri[1, 1])
            z1 = float(tri[1, 2])
            x2 = float(tri[2, 0])
            y2 = float(tri[2, 1])
            z2 = float(tri[2, 2])

            min_x_f = min(x0, min(x1, x2))
            max_x_f = max(x0, max(x1, x2))
            min_y_f = min(y0, min(y1, y2))
            max_y_f = max(y0, max(y1, y2))
            min_x = max(int(np.floor(min_x_f)), 0)
            max_x = min(int(np.ceil(max_x_f)), width - 1)
            min_y = max(int(np.floor(min_y_f)), 0)
            max_y = min(int(np.ceil(max_y_f)), height - 1)
            if min_x > max_x or min_y > max_y:
                continue

            denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
            if abs(denom) < 1e-8:
                continue

            for py in range(min_y, max_y + 1):
                yy = py + 0.5
                for px in range(min_x, max_x + 1):
                    xx = px + 0.5
                    w0 = ((y1 - y2) * (xx - x2) + (x2 - x1) * (yy - y2)) / denom
                    w1 = ((y2 - y0) * (xx - x2) + (x0 - x2) * (yy - y2)) / denom
                    w2 = 1.0 - w0 - w1
                    if w0 < -1e-5 or w1 < -1e-5 or w2 < -1e-5:
                        continue
                    depth = w0 * z0 + w1 * z1 + w2 * z2
                    if depth <= 0.0 or depth >= zbuf[py, px]:
                        continue
                    zbuf[py, px] = depth
                    semantic[py, px] = semantic_ids[tri_idx]
                    instance[py, px] = instance_ids[tri_idx]
                    pix_to_face[py, px] = face_ids[tri_idx]

else:
    _rasterize_triangles_numba = None


def has_numba() -> bool:
    return _rasterize_triangles_numba is not None
