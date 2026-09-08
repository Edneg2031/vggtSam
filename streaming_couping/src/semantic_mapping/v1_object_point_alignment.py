"""Reuse a V1 camera correction as an object-only point transform.

The V1 pose refinement artifact stores two camera-to-world trajectories:
the original HorizonStream trajectory and the refined trajectory.  For a
world point produced with the original pose, the equivalent V1 correction is

    T_correction = T_refined @ inverse(T_raw)

This module applies that correction only to selected static SAM object points.
It deliberately does not replace camera poses, full-scene geometry, or any
future HorizonStream state.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import torch

from .contracts import GeometryFrame


@dataclass(frozen=True)
class V1ObjectPointPoseAlignment:
    """Validated per-frame transforms derived from a V1 pose artifact."""

    artifact_path: Path
    frame_ids: tuple[int, ...]
    raw_camera_to_world: tuple[torch.Tensor, ...]
    refined_camera_to_world: tuple[torch.Tensor, ...]
    correction_by_frame: Mapping[int, torch.Tensor]
    raw_pose_max_abs_error: float

    @classmethod
    def from_artifact(
        cls,
        artifact_path: str | Path,
        geometry_frames: Sequence[GeometryFrame],
        *,
        raw_pose_tolerance: float = 1e-3,
    ) -> "V1ObjectPointPoseAlignment":
        """Load and validate a V1 artifact against current raw geometry poses.

        Matching the stored raw trajectory is important: a correction from a
        different scene, frame range, stride, or HorizonStream cache must not
        silently move the current object points.
        """

        path = Path(artifact_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"V1 pose artifact does not exist: {path}")
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(
                "V1 pose artifact must contain a dictionary, got "
                f"{type(payload)!r}."
            )

        frames = tuple(geometry_frames)
        expected_frame_ids = tuple(int(frame.frame_id) for frame in frames)
        if not frames:
            raise ValueError("V1 object-point alignment requires at least one frame.")
        if expected_frame_ids != tuple(sorted(expected_frame_ids)):
            raise ValueError("Geometry frames must be ordered by increasing frame_id.")

        stored_ids_value = payload.get("frame_ids")
        stored_frame_ids = (
            tuple(int(value) for value in stored_ids_value)
            if stored_ids_value is not None
            else ()
        )
        if stored_frame_ids and stored_frame_ids != expected_frame_ids:
            raise ValueError(
                "V1 pose artifact frame_ids do not match the current sequence: "
                f"stored={stored_frame_ids[:5]}... current={expected_frame_ids[:5]}..."
            )

        raw = _pose_batch(
            payload.get("raw_camera_to_world"),
            name="raw_camera_to_world",
        )
        refined = _pose_batch(
            payload.get("refined_camera_to_world"),
            name="refined_camera_to_world",
        )
        expected_count = len(expected_frame_ids)
        if raw.shape[0] != expected_count or refined.shape[0] != expected_count:
            raise ValueError(
                "V1 pose artifact count does not match the current sequence: "
                f"raw={raw.shape[0]} refined={refined.shape[0]} "
                f"current={expected_count}. Use a V1 artifact generated from "
                "the same frame_start/stride/count."
            )

        current_raw = torch.stack(
            [_homogeneous_pose(frame.camera_to_world, frame.frame_id) for frame in frames]
        )
        raw_pose_error = float((raw - current_raw).abs().max())
        if raw_pose_error > float(raw_pose_tolerance):
            raise ValueError(
                "V1 raw poses do not match the current HorizonStream poses: "
                f"max_abs_error={raw_pose_error:.6g} > tolerance="
                f"{float(raw_pose_tolerance):.6g}. Use the artifact from the "
                "same geometry cache and frame sequence."
            )

        corrections = refined @ torch.linalg.inv(raw)
        if not bool(torch.isfinite(corrections).all()):
            raise ValueError("V1 pose correction contains non-finite values.")
        correction_by_frame = {
            int(frame_id): corrections[index].detach().float().cpu()
            for index, frame_id in enumerate(expected_frame_ids)
        }
        return cls(
            artifact_path=path,
            frame_ids=expected_frame_ids,
            raw_camera_to_world=tuple(
                value.detach().float().cpu() for value in raw
            ),
            refined_camera_to_world=tuple(
                value.detach().float().cpu() for value in refined
            ),
            correction_by_frame=correction_by_frame,
            raw_pose_max_abs_error=raw_pose_error,
        )

    @property
    def transforms_by_frame(self) -> dict[int, torch.Tensor]:
        """Return a detached CPU transform table for the map builder."""

        return {
            int(frame_id): transform.detach().float().cpu()
            for frame_id, transform in self.correction_by_frame.items()
        }

    def to_dict(self) -> dict[str, object]:
        """Return compact JSON-safe provenance and correction statistics."""

        rotations: list[float] = []
        translations: list[float] = []
        for frame_id in self.frame_ids:
            transform = self.correction_by_frame[int(frame_id)]
            rotations.append(_rotation_angle_deg(transform[:3, :3]))
            translations.append(float(torch.linalg.vector_norm(transform[:3, 3])))
        return {
            "enabled": True,
            "source": "v1_refined_camera_to_world",
            "artifact_path": str(self.artifact_path),
            "frame_count": int(len(self.frame_ids)),
            "frame_ids": [int(value) for value in self.frame_ids],
            "raw_pose_max_abs_error": float(self.raw_pose_max_abs_error),
            "mean_correction_rotation_deg": _mean(rotations),
            "max_correction_rotation_deg": max(rotations) if rotations else 0.0,
            "mean_correction_translation_m": _mean(translations),
            "max_correction_translation_m": max(translations) if translations else 0.0,
            "application_scope": "static_sam_object_points_only",
            "camera_pose_modified": False,
            "full_scene_geometry_modified": False,
        }


def apply_object_point_pose_transform(
    transform: torch.Tensor,
    points: torch.Tensor,
) -> torch.Tensor:
    """Apply one homogeneous world-space correction to an ``[N,3]`` cloud."""

    matrix = torch.as_tensor(transform).detach().float().cpu()
    value = torch.as_tensor(points).detach().float().cpu().reshape(-1, 3)
    if tuple(matrix.shape) != (4, 4):
        raise ValueError(
            f"Object-point pose transform must have shape [4,4], got {tuple(matrix.shape)}."
        )
    if not bool(torch.isfinite(matrix).all()) or not bool(torch.isfinite(value).all()):
        raise ValueError("Object-point pose transform and points must be finite.")
    homogeneous = torch.cat(
        (value, torch.ones((value.shape[0], 1), dtype=value.dtype)),
        dim=1,
    )
    return (homogeneous @ matrix.transpose(0, 1))[:, :3]


def _pose_batch(value: Any, *, name: str) -> torch.Tensor:
    if value is None:
        raise ValueError(f"V1 pose artifact is missing {name}.")
    poses = torch.as_tensor(value).detach().double().cpu()
    if poses.ndim == 2:
        poses = poses.unsqueeze(0)
    if poses.ndim != 3:
        raise ValueError(f"{name} must have shape [S,3,4] or [S,4,4].")
    if tuple(poses.shape[-2:]) == (3, 4):
        homogeneous = torch.eye(4, dtype=poses.dtype).repeat(poses.shape[0], 1, 1)
        homogeneous[:, :3] = poses
        poses = homogeneous
    elif tuple(poses.shape[-2:]) != (4, 4):
        raise ValueError(f"{name} must have shape [S,3,4] or [S,4,4].")
    if not bool(torch.isfinite(poses).all()):
        raise ValueError(f"{name} contains non-finite values.")
    return poses


def _homogeneous_pose(value: torch.Tensor | None, frame_id: int) -> torch.Tensor:
    if value is None:
        raise ValueError(
            "V1 object-point alignment needs camera_to_world for every frame; "
            f"frame_id={int(frame_id)} has no pose."
        )
    pose = torch.as_tensor(value).detach().double().cpu()
    if tuple(pose.shape) == (3, 4):
        homogeneous = torch.eye(4, dtype=pose.dtype)
        homogeneous[:3] = pose
        pose = homogeneous
    if tuple(pose.shape) != (4, 4):
        raise ValueError(
            f"camera_to_world for frame_id={int(frame_id)} must have shape [3,4] or [4,4]."
        )
    if not bool(torch.isfinite(pose).all()):
        raise ValueError(f"camera_to_world for frame_id={int(frame_id)} is non-finite.")
    return pose


def _rotation_angle_deg(rotation: torch.Tensor) -> float:
    cosine = ((torch.trace(rotation) - 1.0) * 0.5).clamp(-1.0, 1.0)
    return float(math.degrees(math.acos(float(cosine))))


def _mean(values: Sequence[float]) -> float:
    return float(sum(float(value) for value in values) / len(values)) if values else 0.0


__all__ = [
    "V1ObjectPointPoseAlignment",
    "apply_object_point_pose_transform",
]
