"""Composition of interchangeable geometry and segmentation providers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .contracts import (
    GeometryFrame,
    GeometryProvider,
    SegmentationFrame,
    SegmentationProvider,
)
from .mapping import MapUpdateStats, SemanticMapBuilder, SemanticMapResult


class SemanticMapPipeline:
    """Run geometry, prompted segmentation, and causal map fusion.

    The pipeline contains no StreamVGGT/SAM-specific branches.  Providers may
    use batch inference internally or expose a streaming implementation behind
    the same boundary.  Frame fusion always happens in increasing frame order.
    """

    def __init__(
        self,
        *,
        geometry: GeometryProvider,
        segmentation: SegmentationProvider,
        mapper: SemanticMapBuilder | None = None,
    ) -> None:
        self.geometry = geometry
        self.segmentation = segmentation
        self.mapper = mapper or SemanticMapBuilder()

    def run(
        self,
        image_paths: Sequence[str | Path],
        *,
        prompts: Sequence[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> SemanticMapResult:
        paths = tuple(Path(path).expanduser() for path in image_paths)
        if not paths:
            raise ValueError("SemanticMapPipeline requires at least one RGB frame.")
        geometry_frames = tuple(self.geometry.infer(paths))
        if len(geometry_frames) != len(paths):
            raise ValueError(
                "Geometry provider returned a different number of frames: "
                f"{len(geometry_frames)} vs {len(paths)}."
            )
        infer_with_geometry = getattr(self.segmentation, "infer_with_geometry", None)
        if callable(infer_with_geometry):
            segmentation_frames = tuple(
                infer_with_geometry(paths, geometry_frames, prompts)
            )
        else:
            segmentation_frames = tuple(self.segmentation.infer(paths, prompts))
        if len(segmentation_frames) != len(paths):
            raise ValueError(
                "Segmentation provider returned a different number of frames: "
                f"{len(segmentation_frames)} vs {len(paths)}."
            )
        geometry_by_id = _unique_frame_map(geometry_frames, "geometry")
        segmentation_by_id = _unique_frame_map(segmentation_frames, "segmentation")
        if set(geometry_by_id) != set(segmentation_by_id):
            raise ValueError(
                "Geometry and segmentation providers returned different frame IDs: "
                f"geometry={sorted(geometry_by_id)}, "
                f"segmentation={sorted(segmentation_by_id)}."
            )
        ordered_ids = tuple(int(frame.frame_id) for frame in geometry_frames)
        if ordered_ids != tuple(sorted(ordered_ids)):
            raise ValueError("Geometry provider must return frames in increasing order.")

        for frame_id in ordered_ids:
            self.update(
                geometry_by_id[frame_id],
                segmentation_by_id[frame_id],
            )

        result_metadata = dict(metadata or {})
        result_metadata.setdefault("geometry_backend", str(self.geometry.backend_name))
        result_metadata.setdefault(
            "segmentation_backend",
            str(self.segmentation.backend_name),
        )
        result_metadata.setdefault("frame_ids", list(ordered_ids))
        result_metadata.setdefault(
            "prompts",
            [str(prompt) for prompt in prompts] if prompts is not None else [],
        )
        result_metadata.setdefault("causal_fusion", True)
        guidance_summary = getattr(self.segmentation, "last_summary", None)
        if guidance_summary is not None:
            result_metadata.setdefault(
                "segmentation_guidance_summary",
                dict(guidance_summary),
            )
        guidance_diagnostics = getattr(self.segmentation, "last_diagnostics", None)
        if guidance_diagnostics is not None:
            result_metadata.setdefault(
                "segmentation_guidance_diagnostics",
                list(guidance_diagnostics),
            )
        return self.finalize(result_metadata)

    def update(
        self,
        geometry: GeometryFrame,
        segmentation: SegmentationFrame,
    ) -> MapUpdateStats:
        """Fuse one already-aligned pair for a native streaming caller."""

        return self.mapper.update(geometry, segmentation)

    def finalize(self, metadata: Mapping[str, Any] | None = None) -> SemanticMapResult:
        """Finalize a stream that was consumed through :meth:`update`."""

        result_metadata = dict(metadata or {})
        result_metadata.setdefault("geometry_backend", str(self.geometry.backend_name))
        result_metadata.setdefault(
            "segmentation_backend",
            str(self.segmentation.backend_name),
        )
        result_metadata.setdefault("causal_fusion", True)
        return self.mapper.finalize(result_metadata)


def _unique_frame_map(frames: Sequence[Any], name: str) -> dict[int, Any]:
    output: dict[int, Any] = {}
    for frame in frames:
        frame_id = int(frame.frame_id)
        if frame_id in output:
            raise ValueError(f"{name} provider returned duplicate frame_id={frame_id}.")
        output[frame_id] = frame
    return output
