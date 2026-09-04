"""Composition of interchangeable geometry and segmentation providers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .contracts import (
    GeometryFrame,
    GeometryProvider,
    SegmentationFrame,
    SegmentationProvider,
)
from .mapping import MapUpdateStats, SemanticMapBuilder, SemanticMapResult
from .object_pose_refinement import (
    ObjectPoseRefiner,
    PoseRefinementResult,
    apply_refined_camera_poses,
)


@dataclass(frozen=True)
class SemanticMapPoseRefinementRun:
    """Raw and pose-refined map results from one shared model inference."""

    raw_results: Mapping[str, SemanticMapResult]
    refined_results: Mapping[str, SemanticMapResult]
    refinement: PoseRefinementResult


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
        geometry_frames, segmentation_frames, ordered_ids = self._infer(
            paths,
            prompts,
        )
        return self._fuse(
            geometry_frames,
            segmentation_frames,
            ordered_ids=ordered_ids,
            metadata=metadata,
            prompts=prompts,
            mapper=self.mapper,
        )

    def run_branches(
        self,
        image_paths: Sequence[str | Path],
        *,
        prompts: Sequence[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
        policies: Sequence[str] = ("raw", "temporal_consensus"),
    ) -> dict[str, SemanticMapResult]:
        """Run model providers once and fuse the frozen result per policy.

        This is the comparison path for causal map experiments.  Geometry and
        segmentation inference is shared across all branches, so a branch
        difference is attributable to the downstream map policy rather than a
        second stochastic model invocation.
        """

        paths = tuple(Path(path).expanduser() for path in image_paths)
        if not paths:
            raise ValueError("SemanticMapPipeline requires at least one RGB frame.")
        normalized = tuple(str(policy).strip().lower() for policy in policies)
        if not normalized:
            raise ValueError("run_branches requires at least one map policy.")
        if len(set(normalized)) != len(normalized):
            raise ValueError("run_branches policies must be unique.")
        unsupported = [
            policy
            for policy in normalized
            if policy not in {
                "raw",
                "temporal_consensus",
                "instance_point_consistency",
            }
        ]
        if unsupported:
            raise ValueError(
                "Unsupported semantic-map branch policy/policies: "
                f"{unsupported!r}."
            )
        if not isinstance(self.mapper, SemanticMapBuilder):
            raise TypeError(
                "run_branches requires a SemanticMapBuilder so each policy "
                "can receive an isolated mapper."
            )

        geometry_frames, segmentation_frames, ordered_ids = self._infer(
            paths,
            prompts,
        )
        results: dict[str, SemanticMapResult] = {}
        for policy in normalized:
            if policy == "instance_point_consistency":
                branch_config = replace(
                    self.mapper.config,
                    fusion_policy="instance_point_consistency",
                    instance_point_consistency=replace(
                        self.mapper.config.instance_point_consistency,
                        enabled=True,
                    ),
                )
            else:
                branch_config = replace(
                    self.mapper.config,
                    fusion_policy=policy,
                    instance_point_consistency=replace(
                        self.mapper.config.instance_point_consistency,
                        enabled=False,
                    ),
                )
            branch_mapper = SemanticMapBuilder(
                branch_config
            )
            branch_metadata = dict(metadata or {})
            branch_metadata["fusion_policy"] = policy
            branch_metadata["branch_shared_model_inference"] = True
            branch_metadata["instance_point_consistency_requested"] = (
                policy == "instance_point_consistency"
            )
            results[policy] = self._fuse(
                geometry_frames,
                segmentation_frames,
                ordered_ids=ordered_ids,
                metadata=branch_metadata,
                prompts=prompts,
                mapper=branch_mapper,
            )
        return results

    def run_with_object_pose_refinement(
        self,
        image_paths: Sequence[str | Path],
        *,
        refiner: ObjectPoseRefiner,
        prompts: Sequence[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
        policies: Sequence[str] = ("raw",),
    ) -> SemanticMapPoseRefinementRun:
        """Run raw and SAM-object pose-refined maps from shared frozen outputs.

        Geometry and SAM inference happen exactly once.  The refiner sees the
        resulting canonical frames, and only the refined map receives the
        replacement camera poses.  Depth, intrinsics, masks, and persistent
        IDs are reused unchanged.  This method is opt-in; :meth:`run` and
        :meth:`run_branches` retain their existing behavior.
        """

        paths = tuple(Path(path).expanduser() for path in image_paths)
        if not paths:
            raise ValueError(
                "SemanticMapPipeline requires at least one RGB frame."
            )
        normalized = _validate_policies(policies)
        geometry_frames, segmentation_frames, ordered_ids = self._infer(
            paths,
            prompts,
        )
        refinement = refiner.refine(
            geometry_frames,
            segmentation_frames,
            paths,
        )
        refined_geometry_frames = apply_refined_camera_poses(
            geometry_frames,
            refinement,
        )
        refinement_metadata = {
            "enabled": True,
            "candidate_generation_gt_fields": 0,
            "evaluation_gt_fields": 0,
            "raw_pose_unchanged": True,
            "summary": dict(refinement.summary),
        }
        raw_results: dict[str, SemanticMapResult] = {}
        refined_results: dict[str, SemanticMapResult] = {}
        for policy in normalized:
            raw_mapper = SemanticMapBuilder(
                replace(self.mapper.config, fusion_policy=policy)
            )
            refined_mapper = SemanticMapBuilder(
                replace(self.mapper.config, fusion_policy=policy)
            )
            raw_metadata = dict(metadata or {})
            raw_metadata.update(
                {
                    "fusion_policy": policy,
                    "pose_variant": "raw_horizonstream",
                    "object_pose_refinement": refinement_metadata,
                }
            )
            refined_metadata = dict(metadata or {})
            refined_metadata.update(
                {
                    "fusion_policy": policy,
                    "pose_variant": "object_pose_refined",
                    "object_pose_refinement": refinement_metadata,
                }
            )
            raw_results[policy] = self._fuse(
                geometry_frames,
                segmentation_frames,
                ordered_ids=ordered_ids,
                metadata=raw_metadata,
                prompts=prompts,
                mapper=raw_mapper,
            )
            refined_results[policy] = self._fuse(
                refined_geometry_frames,
                segmentation_frames,
                ordered_ids=ordered_ids,
                metadata=refined_metadata,
                prompts=prompts,
                mapper=refined_mapper,
            )
        return SemanticMapPoseRefinementRun(
            raw_results=raw_results,
            refined_results=refined_results,
            refinement=refinement,
        )

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

    def _infer(
        self,
        paths: Sequence[Path],
        prompts: Sequence[str] | None,
    ) -> tuple[tuple[GeometryFrame, ...], tuple[SegmentationFrame, ...], tuple[int, ...]]:
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
        return geometry_frames, segmentation_frames, ordered_ids

    def _fuse(
        self,
        geometry_frames: Sequence[GeometryFrame],
        segmentation_frames: Sequence[SegmentationFrame],
        *,
        ordered_ids: Sequence[int],
        metadata: Mapping[str, Any] | None,
        prompts: Sequence[str] | None,
        mapper: SemanticMapBuilder,
    ) -> SemanticMapResult:
        geometry_by_id = _unique_frame_map(geometry_frames, "geometry")
        segmentation_by_id = _unique_frame_map(segmentation_frames, "segmentation")
        for frame_id in ordered_ids:
            mapper.update(
                geometry_by_id[int(frame_id)],
                segmentation_by_id[int(frame_id)],
            )
        result_metadata = self._result_metadata(
            metadata,
            ordered_ids=ordered_ids,
            prompts=prompts,
            mapper=mapper,
        )
        return mapper.finalize(result_metadata)

    def _result_metadata(
        self,
        metadata: Mapping[str, Any] | None,
        *,
        ordered_ids: Sequence[int],
        prompts: Sequence[str] | None,
        mapper: SemanticMapBuilder,
    ) -> dict[str, Any]:
        result_metadata = dict(metadata or {})
        result_metadata.setdefault("geometry_backend", str(self.geometry.backend_name))
        result_metadata.setdefault(
            "segmentation_backend",
            str(self.segmentation.backend_name),
        )
        result_metadata.setdefault("frame_ids", [int(value) for value in ordered_ids])
        result_metadata.setdefault(
            "prompts",
            [str(prompt) for prompt in prompts] if prompts is not None else [],
        )
        result_metadata.setdefault("causal_fusion", True)
        result_metadata.setdefault("fusion_policy", str(mapper.config.fusion_policy))
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
        return result_metadata


def _unique_frame_map(frames: Sequence[Any], name: str) -> dict[int, Any]:
    output: dict[int, Any] = {}
    for frame in frames:
        frame_id = int(frame.frame_id)
        if frame_id in output:
            raise ValueError(f"{name} provider returned duplicate frame_id={frame_id}.")
        output[frame_id] = frame
    return output


def _validate_policies(policies: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(str(policy).strip().lower() for policy in policies)
    if not normalized:
        raise ValueError("At least one semantic-map policy is required.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("Semantic-map policies must be unique.")
    unsupported = [
        policy
        for policy in normalized
        if policy not in {
            "raw",
            "temporal_consensus",
            "instance_point_consistency",
        }
    ]
    if unsupported:
        raise ValueError(f"Unsupported semantic-map policy/policies: {unsupported!r}.")
    return normalized
