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
from .online_object_pose_loop import OnlineObjectPoseLoopRefiner
from .object_point_pose_alignment import ObjectPointPoseAlignment
from .v1_object_point_alignment import V1ObjectPointPoseAlignment


@dataclass(frozen=True)
class SemanticMapPoseRefinementRun:
    """Raw and pose-refined map results from one shared model inference."""

    raw_results: Mapping[str, SemanticMapResult]
    refined_results: Mapping[str, SemanticMapResult]
    refinement: PoseRefinementResult
    object_only: bool = False


@dataclass(frozen=True)
class SemanticMapV1ObjectPointAlignmentRun:
    """Raw and V1-corrected object-point maps from shared inference."""

    results: Mapping[str, SemanticMapResult]
    alignment: V1ObjectPointPoseAlignment


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
                "instance_point_alignment",
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
                    instance_point_alignment=replace(
                        self.mapper.config.instance_point_alignment,
                        enabled=False,
                    ),
                )
            elif policy == "instance_point_alignment":
                branch_config = replace(
                    self.mapper.config,
                    fusion_policy="instance_point_alignment",
                    instance_point_consistency=replace(
                        self.mapper.config.instance_point_consistency,
                        enabled=False,
                    ),
                    instance_point_alignment=replace(
                        self.mapper.config.instance_point_alignment,
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
                    instance_point_alignment=replace(
                        self.mapper.config.instance_point_alignment,
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
            branch_metadata["instance_point_alignment_requested"] = (
                policy == "instance_point_alignment"
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

    def run_with_v1_object_point_alignment(
        self,
        image_paths: Sequence[str | Path],
        *,
        pose_artifact: str | Path,
        raw_pose_tolerance: float = 1e-3,
        prompts: Sequence[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> SemanticMapV1ObjectPointAlignmentRun:
        """Reuse V1 pose corrections only for static SAM object points.

        Both branches fuse the original geometry frames, so their camera poses
        and full-scene pointmaps remain HorizonStream's raw outputs.  The
        aligned branch carries a per-frame transform through the geometry
        contract; the mapper applies it only after selecting a static SAM
        instance mask.
        """

        paths = tuple(Path(path).expanduser() for path in image_paths)
        if not paths:
            raise ValueError("SemanticMapPipeline requires at least one RGB frame.")
        geometry_frames, segmentation_frames, ordered_ids = self._infer(
            paths,
            prompts,
        )
        alignment = V1ObjectPointPoseAlignment.from_artifact(
            pose_artifact,
            geometry_frames,
            raw_pose_tolerance=float(raw_pose_tolerance),
        )

        base_config = replace(
            self.mapper.config,
            fusion_policy="raw",
            instance_point_consistency=replace(
                self.mapper.config.instance_point_consistency,
                enabled=False,
            ),
            instance_point_alignment=replace(
                self.mapper.config.instance_point_alignment,
                enabled=False,
            ),
        )
        raw_mapper = SemanticMapBuilder(base_config)
        aligned_mapper = SemanticMapBuilder(base_config)
        aligned_geometry_frames = tuple(
            replace(
                frame,
                object_point_transform=alignment.correction_by_frame[
                    int(frame.frame_id)
                ],
            )
            for frame in geometry_frames
        )

        alignment_metadata = alignment.to_dict()
        raw_metadata = dict(metadata or {})
        raw_metadata.update(
            {
                "fusion_policy": "raw",
                "pose_variant": "raw_horizonstream",
                "branch_shared_model_inference": True,
                "camera_pose_modified": False,
                "full_scene_geometry_modified": False,
                "pointmap_modified": False,
                "pointmap_modified_scope": "none",
                "v1_object_point_alignment": {
                    **alignment_metadata,
                    "applied": False,
                },
            }
        )
        aligned_metadata = dict(metadata or {})
        aligned_metadata.update(
            {
                "fusion_policy": "raw",
                "pose_variant": "raw_horizonstream",
                "branch_shared_model_inference": True,
                "camera_pose_modified": False,
                "full_scene_geometry_modified": False,
                "pointmap_modified": True,
                "pointmap_modified_scope": "static_sam_object_points_only",
                "v1_object_point_alignment": {
                    **alignment_metadata,
                    "applied": True,
                },
            }
        )
        results = {
            "raw": self._fuse(
                geometry_frames,
                segmentation_frames,
                ordered_ids=ordered_ids,
                metadata=raw_metadata,
                prompts=prompts,
                mapper=raw_mapper,
            ),
            "v1_object_point_alignment": self._fuse(
                aligned_geometry_frames,
                segmentation_frames,
                ordered_ids=ordered_ids,
                metadata=aligned_metadata,
                prompts=prompts,
                mapper=aligned_mapper,
            ),
        }
        return SemanticMapV1ObjectPointAlignmentRun(
            results=results,
            alignment=alignment,
        )

    def run_with_object_pose_refinement(
        self,
        image_paths: Sequence[str | Path],
        *,
        refiner: ObjectPoseRefiner,
        prompts: Sequence[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
        policies: Sequence[str] = ("raw",),
        object_only: bool = False,
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
        object_point_alignment = None
        if object_only:
            object_point_alignment = ObjectPointPoseAlignment.from_refinement(
                refinement,
                geometry_frames,
                source=str(getattr(refiner, "method_name", type(refiner).__name__)),
            )
            refined_geometry_frames = tuple(
                replace(
                    frame,
                    object_point_transform=object_point_alignment.correction_by_frame[
                        int(frame.frame_id)
                    ],
                )
                for frame in geometry_frames
            )
        else:
            refined_geometry_frames = apply_refined_camera_poses(
                geometry_frames,
                refinement,
            )
        refined_pose_variant = (
            "raw_horizonstream_object_only"
            if object_only
            else (
                "object_pose_online_loop"
                if str(getattr(refiner, "method_name", "")).startswith(
                    "sam_instance_guided_external_online"
                )
                else "object_pose_refined"
            )
        )
        refinement_metadata = {
            "enabled": True,
            "candidate_generation_gt_fields": 0,
            "evaluation_gt_fields": 0,
            "raw_pose_unchanged": True,
            "summary": dict(refinement.summary),
            "object_only": bool(object_only),
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
                    "pose_variant": refined_pose_variant,
                    "object_pose_refinement": refinement_metadata,
                }
            )
            if object_only:
                alignment_metadata = object_point_alignment.to_dict()
                raw_metadata.update(
                    {
                        "camera_pose_modified": False,
                        "full_scene_geometry_modified": False,
                        "pointmap_modified": False,
                        "pointmap_modified_scope": "none",
                        "object_pose_refinement_object_only": True,
                        "object_point_pose_alignment": {
                            **alignment_metadata,
                            "applied": False,
                        },
                    }
                )
                refined_metadata.update(
                    {
                        "camera_pose_modified": False,
                        "full_scene_geometry_modified": False,
                        "pointmap_modified": True,
                        "pointmap_modified_scope": "static_sam_object_points_only",
                        "object_pose_refinement_object_only": True,
                        "object_point_pose_alignment": {
                            **alignment_metadata,
                            "applied": True,
                        },
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
            object_only=bool(object_only),
        )

    def run_with_online_object_pose_loop(
        self,
        image_paths: Sequence[str | Path],
        *,
        refiner: OnlineObjectPoseLoopRefiner,
        prompts: Sequence[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
        policies: Sequence[str] = ("raw",),
        object_only: bool = False,
    ) -> SemanticMapPoseRefinementRun:
        """Run the causal external SAM pose-correction loop.

        Geometry and SAM providers are evaluated once, then the refiner
        consumes their canonical frames in increasing frame order.  Unlike
        the legacy offline refiner, it predicts the next pose from the
        previously corrected external pose, optimizes a recent sliding
        window, and returns the corrected trajectory for downstream fusion.
        HorizonStream's internal latent/cache state is intentionally not
        mutated; this is the external-feedback ablation.
        """

        run = self.run_with_object_pose_refinement(
            image_paths,
            refiner=refiner,  # type: ignore[arg-type]
            prompts=prompts,
            metadata=metadata,
            policies=policies,
            object_only=object_only,
        )
        return run

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
            "instance_point_alignment",
        }
    ]
    if unsupported:
        raise ValueError(f"Unsupported semantic-map policy/policies: {unsupported!r}.")
    return normalized
