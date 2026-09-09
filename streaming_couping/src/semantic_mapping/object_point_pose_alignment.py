"""Apply pose-refinement corrections only to SAM-selected object points."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import torch

from .contracts import GeometryFrame
from .object_pose_refinement import PoseRefinementResult


@dataclass(frozen=True)
class ObjectPointPoseAlignment:
    """Per-frame world correction used only after an object mask is selected."""

    source: str
    frame_ids: tuple[int, ...]
    raw_camera_to_world: tuple[torch.Tensor, ...]
    refined_camera_to_world: tuple[torch.Tensor, ...]
    correction_by_frame: dict[int, torch.Tensor]
    raw_pose_max_abs_error: float
    artifact_path: Path | None = None
    correction_by_frame_instance: dict[int, dict[int, torch.Tensor]] | None = None

    @classmethod
    def from_refinement(
        cls,
        refinement: PoseRefinementResult,
        geometry_frames: Sequence[GeometryFrame],
        *,
        source: str,
        raw_pose_tolerance: float = 1e-3,
    ) -> "ObjectPointPoseAlignment":
        frames = tuple(geometry_frames)
        expected_ids = tuple(int(frame.frame_id) for frame in frames)
        if not frames:
            raise ValueError("Object-point pose alignment requires frames.")
        if expected_ids != tuple(int(value) for value in refinement.frame_ids):
            raise ValueError(
                "Pose refinement frame IDs do not match geometry frames: "
                f"refinement={tuple(refinement.frame_ids)[:5]}... "
                f"geometry={expected_ids[:5]}..."
            )

        raw = tuple(_pose4(value) for value in refinement.raw_camera_to_world)
        refined = tuple(_pose4(value) for value in refinement.refined_camera_to_world)
        current_raw = tuple(
            _pose4(frame.camera_to_world, frame_id=int(frame.frame_id))
            for frame in frames
        )
        raw_error = max(
            float((stored - current).abs().max())
            for stored, current in zip(raw, current_raw)
        )
        if raw_error > float(raw_pose_tolerance):
            raise ValueError(
                "Pose refinement raw poses do not match current geometry: "
                f"max_abs_error={raw_error:.6g} > "
                f"tolerance={float(raw_pose_tolerance):.6g}."
            )

        per_instance_payload = getattr(refinement, "object_point_corrections", None)
        if per_instance_payload is not None:
            correction_by_frame_instance: dict[int, dict[int, torch.Tensor]] = {}
            for frame_id in expected_ids:
                raw_frame = per_instance_payload.get(int(frame_id), {})
                if not isinstance(raw_frame, Mapping):
                    raise ValueError(
                        "Per-instance object-point corrections must be mappings."
                    )
                correction_by_frame_instance[int(frame_id)] = {
                    int(instance_id): torch.as_tensor(transform)
                    .detach()
                    .float()
                    .cpu()
                    for instance_id, transform in raw_frame.items()
                }
            corrections = {
                int(frame_id): torch.eye(4, dtype=torch.float32)
                for frame_id in expected_ids
            }
        else:
            correction_by_frame_instance = None
            corrections = {
                int(frame_id): refined[index] @ torch.linalg.inv(raw[index])
                for index, frame_id in enumerate(expected_ids)
            }
        if not all(bool(torch.isfinite(value).all()) for value in corrections.values()):
            raise ValueError("Object-point pose correction contains non-finite values.")
        if correction_by_frame_instance is not None:
            for frame_map in correction_by_frame_instance.values():
                for instance_id, transform in frame_map.items():
                    if int(instance_id) < 0 or tuple(transform.shape) != (4, 4):
                        raise ValueError(
                            "Per-instance object-point corrections must map "
                            "non-negative IDs to [4,4] transforms."
                        )
                    if not bool(torch.isfinite(transform).all()):
                        raise ValueError(
                            "Per-instance object-point correction contains "
                            "non-finite values."
                        )
        return cls(
            source=str(source),
            frame_ids=expected_ids,
            raw_camera_to_world=raw,
            refined_camera_to_world=refined,
            correction_by_frame={
                int(frame_id): value.detach().float().cpu()
                for frame_id, value in corrections.items()
            },
            raw_pose_max_abs_error=float(raw_error),
            correction_by_frame_instance=correction_by_frame_instance,
        )

    def to_dict(self) -> dict[str, Any]:
        rotations: list[float] = []
        translations: list[float] = []
        if self.correction_by_frame_instance is not None:
            for frame_map in self.correction_by_frame_instance.values():
                for transform in frame_map.values():
                    rotations.append(_rotation_angle_deg(transform[:3, :3]))
                    translations.append(
                        float(torch.linalg.vector_norm(transform[:3, 3]))
                    )
        else:
            for frame_id in self.frame_ids:
                transform = self.correction_by_frame[int(frame_id)]
                rotations.append(_rotation_angle_deg(transform[:3, :3]))
                translations.append(float(torch.linalg.vector_norm(transform[:3, 3])))
        return {
            "enabled": True,
            "source": str(self.source),
            "artifact_path": None
            if self.artifact_path is None
            else str(self.artifact_path),
            "frame_count": int(len(self.frame_ids)),
            "frame_ids": [int(value) for value in self.frame_ids],
            "raw_pose_max_abs_error": float(self.raw_pose_max_abs_error),
            "mean_correction_rotation_deg": _mean(rotations),
            "max_correction_rotation_deg": max(rotations, default=0.0),
            "mean_correction_translation_m": _mean(translations),
            "max_correction_translation_m": max(translations, default=0.0),
            "application_scope": (
                "static_sam_object_points_per_instance_only"
                if self.correction_by_frame_instance is not None
                else "static_sam_object_points_only"
            ),
            "per_instance": self.correction_by_frame_instance is not None,
            "per_instance_frame_count": (
                0
                if self.correction_by_frame_instance is None
                else sum(
                    bool(value)
                    for value in self.correction_by_frame_instance.values()
                )
            ),
            "per_instance_correction_count": (
                0
                if self.correction_by_frame_instance is None
                else sum(
                    len(value)
                    for value in self.correction_by_frame_instance.values()
                )
            ),
            "per_instance_ids": (
                []
                if self.correction_by_frame_instance is None
                else sorted(
                    {
                        int(instance_id)
                        for frame_map in self.correction_by_frame_instance.values()
                        for instance_id in frame_map
                    }
                )
            ),
            "camera_pose_modified": False,
            "full_scene_geometry_modified": False,
        }


def apply_object_point_pose_transform(
    transform: torch.Tensor,
    points: torch.Tensor,
) -> torch.Tensor:
    """Apply a homogeneous world-space correction to an ``[N,3]`` cloud."""

    matrix = torch.as_tensor(transform).detach().float().cpu()
    value = torch.as_tensor(points).detach().float().cpu().reshape(-1, 3)
    if tuple(matrix.shape) != (4, 4):
        raise ValueError(
            "Object-point pose transform must have shape [4,4], got "
            f"{tuple(matrix.shape)}."
        )
    if not bool(torch.isfinite(matrix).all()) or not bool(torch.isfinite(value).all()):
        raise ValueError("Object-point pose transform and points must be finite.")
    homogeneous = torch.cat(
        (value, torch.ones((value.shape[0], 1), dtype=value.dtype)),
        dim=1,
    )
    return (homogeneous @ matrix.transpose(0, 1))[:, :3]


def _pose4(value: Any, *, frame_id: int | None = None) -> torch.Tensor:
    if value is None:
        suffix = "" if frame_id is None else f" for frame_id={int(frame_id)}"
        raise ValueError(f"Camera pose is missing{suffix}.")
    pose = torch.as_tensor(value).detach().float().cpu()
    if tuple(pose.shape) == (3, 4):
        output = torch.eye(4, dtype=pose.dtype)
        output[:3] = pose
        pose = output
    if tuple(pose.shape) != (4, 4):
        raise ValueError(f"Camera pose must have shape [3,4] or [4,4], got {tuple(pose.shape)}.")
    if not bool(torch.isfinite(pose).all()):
        raise ValueError("Camera pose contains non-finite values.")
    return pose


def _rotation_angle_deg(rotation: torch.Tensor) -> float:
    cosine = ((torch.trace(rotation) - 1.0) * 0.5).clamp(-1.0, 1.0)
    return float(math.degrees(math.acos(float(cosine))))


def _mean(values: Sequence[float]) -> float:
    return float(sum(float(value) for value in values) / len(values)) if values else 0.0


__all__ = [
    "ObjectPointPoseAlignment",
    "apply_object_point_pose_transform",
]
