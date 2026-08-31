"""Canonical contracts shared by geometry, segmentation, and map backends."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import torch


SizeHW = tuple[int, int]


@dataclass(frozen=True)
class GeometryFrame:
    """One frame of backend-independent scene geometry.

    ``world_points`` is preferred because it avoids repeating a backprojection
    when a geometry model already predicts a shared world pointmap.  A future
    backend may provide only ``depth``, ``intrinsics``, and
    ``camera_to_world``; the mapper then reconstructs world points through the
    same contract.

    All tensors are expected to describe the canonical ``image_size`` grid.
    Adapters are responsible for converting model-specific resolutions and
    pose conventions before constructing this object.
    """

    frame_id: int
    image_size: SizeHW
    world_points: torch.Tensor | None = None
    depth: torch.Tensor | None = None
    intrinsics: torch.Tensor | None = None
    camera_to_world: torch.Tensor | None = None
    confidence: torch.Tensor | None = None
    valid: torch.Tensor | None = None
    rgb: torch.Tensor | None = None
    coordinate_frame: str = "world"
    scale_type: str = "metric"
    backend: str = "unknown"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> "GeometryFrame":
        if int(self.frame_id) < 0:
            raise ValueError("frame_id must be non-negative.")
        height, width = _size(self.image_size)
        if str(self.coordinate_frame).strip() != "world":
            raise ValueError(
                "GeometryFrame.coordinate_frame must be 'world'; adapters "
                "must convert local model coordinates first."
            )
        if str(self.scale_type).strip() not in {"metric", "relative", "unknown"}:
            raise ValueError(
                "scale_type must be 'metric', 'relative', or 'unknown'."
            )
        if self.world_points is None:
            if self.depth is None or self.intrinsics is None or self.camera_to_world is None:
                raise ValueError(
                    "A GeometryFrame needs world_points, or depth plus "
                    "intrinsics and camera_to_world."
                )
        else:
            _require_shape(self.world_points, (height, width, 3), "world_points")
        if self.depth is not None:
            depth_shape = tuple(int(value) for value in self.depth.shape)
            if depth_shape == (height, width, 1):
                depth_shape = depth_shape[:2]
            if depth_shape != (height, width):
                raise ValueError(
                    f"depth must have shape {(height, width)}, got {tuple(self.depth.shape)}."
                )
        if self.intrinsics is not None:
            _require_shape(self.intrinsics, (3, 3), "intrinsics")
        if self.camera_to_world is not None:
            if tuple(self.camera_to_world.shape) not in {(3, 4), (4, 4)}:
                raise ValueError(
                    "camera_to_world must have shape [3,4] or [4,4], got "
                    f"{tuple(self.camera_to_world.shape)}."
                )
        for name, value in (
            ("confidence", self.confidence),
            ("valid", self.valid),
        ):
            if value is not None:
                _require_shape(value, (height, width), name)
        if self.rgb is not None:
            _require_shape(self.rgb, (height, width, 3), "rgb")
        return self

    def cpu(self) -> "GeometryFrame":
        """Return a detached CPU copy suitable for deterministic map fusion."""

        def _cpu(value: torch.Tensor | None) -> torch.Tensor | None:
            return None if value is None else value.detach().float().cpu()

        return replace(
            self,
            world_points=_cpu(self.world_points),
            depth=_cpu(self.depth),
            intrinsics=_cpu(self.intrinsics),
            camera_to_world=_cpu(self.camera_to_world),
            confidence=_cpu(self.confidence),
            valid=(
                None
                if self.valid is None
                else self.valid.detach().bool().cpu()
            ),
            rgb=_cpu(self.rgb),
        )


@dataclass(frozen=True)
class ObjectObservation:
    """One prompt/category observation of a persistent object track."""

    category: str
    instance_id: int
    mask: torch.Tensor
    score: float = 1.0
    static_score: float | None = None
    source_track_id: int | None = None
    birth_index: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        category = str(self.category).strip()
        if not category:
            raise ValueError("ObjectObservation.category must not be empty.")
        if int(self.instance_id) < 0:
            raise ValueError("ObjectObservation.instance_id must be non-negative.")
        if not torch.is_tensor(self.mask) or self.mask.ndim != 2:
            raise ValueError(
                "ObjectObservation.mask must be a two-dimensional tensor."
            )
        score = float(self.score)
        if not torch.isfinite(torch.tensor(score)) or not 0.0 <= score <= 1.0:
            raise ValueError("ObjectObservation.score must be finite and in [0,1].")
        if self.static_score is not None:
            static_score = float(self.static_score)
            if not torch.isfinite(torch.tensor(static_score)) or not 0.0 <= static_score <= 1.0:
                raise ValueError(
                    "ObjectObservation.static_score must be finite and in [0,1]."
                )
        if self.birth_index is not None and int(self.birth_index) < 0:
            raise ValueError("birth_index must be non-negative when provided.")

    def cpu(self) -> "ObjectObservation":
        return replace(self, mask=self.mask.detach().bool().cpu())


@dataclass(frozen=True)
class SegmentationFrame:
    """All object observations emitted for one frame."""

    frame_id: int
    image_size: SizeHW
    observations: tuple[ObjectObservation, ...] = ()
    backend: str = "unknown"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.frame_id) < 0:
            raise ValueError("frame_id must be non-negative.")
        _size(self.image_size)
        for observation in self.observations:
            if not isinstance(observation, ObjectObservation):
                raise TypeError("observations must contain ObjectObservation values.")

    def cpu(self) -> "SegmentationFrame":
        return replace(
            self,
            observations=tuple(observation.cpu() for observation in self.observations),
        )


@runtime_checkable
class GeometryProvider(Protocol):
    """Batch geometry interface implemented by each geometry backend."""

    backend_name: str

    def infer(
        self,
        image_paths: Sequence[str | Path],
    ) -> Sequence[GeometryFrame]:
        ...


@runtime_checkable
class SegmentationProvider(Protocol):
    """Prompted segmentation interface implemented by each SAM backend."""

    backend_name: str

    def infer(
        self,
        image_paths: Sequence[str | Path],
        prompts: Sequence[str] | None = None,
    ) -> Sequence[SegmentationFrame]:
        ...


@runtime_checkable
class StreamingGeometryProvider(Protocol):
    """Optional frame-at-a-time interface for a native streaming backend."""

    backend_name: str

    def reset(self) -> None:
        ...

    def push(
        self,
        frame: torch.Tensor,
        *,
        frame_id: int,
    ) -> GeometryFrame | None:
        ...

    def finish(self) -> Sequence[GeometryFrame]:
        ...


def _size(value: Sequence[int]) -> SizeHW:
    if len(value) != 2:
        raise ValueError(f"Expected (height, width), got {value!r}.")
    height, width = int(value[0]), int(value[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"Image dimensions must be positive, got {(height, width)}.")
    return height, width


def _require_shape(value: torch.Tensor, shape: tuple[int, ...], name: str) -> None:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if tuple(int(item) for item in value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}.")
