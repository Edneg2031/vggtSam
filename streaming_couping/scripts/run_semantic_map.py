#!/usr/bin/env python3
"""Build a semantic-instance map from RGB prompts or a frozen V0 cache.

The RGB path accepts either an isolated HorizonStream geometry cache or the
legacy in-process StreamVGGT wrapper, then runs SAM3.1.  The V0 cache path
replays both frozen providers.  Every path converges on the same
backend-neutral ``SemanticMapPipeline``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
from typing import Any

import torch

from streaming_couping.src.config import load_config
from streaming_couping.src.horizonstream_cache import (
    load_horizonstream_cache,
    materialize_horizonstream_rgb,
    validate_horizonstream_cache,
)
from streaming_couping.src.rgb_inputs import (
    expand_image_paths,
    expand_manifest_paths,
    resolve_rgb_inputs,
    select_image_paths,
    selected_positions,
)
from streaming_couping.src.semantic_mapping.adapters import (
    GeometryAwareSAM31SegmentationAdapter,
    HorizonStreamGeometryCacheAdapter,
    SAM31SegmentationAdapter,
    StreamVGGTGeometryAdapter,
    V0CacheGeometryAdapter,
    V0CacheSegmentationAdapter,
)
from streaming_couping.src.semantic_mapping.export import export_semantic_map
from streaming_couping.src.semantic_mapping.mapping import (
    MapWriteGateConfig,
    SemanticMapBuilder,
    SemanticMapConfig,
)
from streaming_couping.src.semantic_mapping.instance_point_consistency import (
    InstancePointConsistencyConfig,
)
from streaming_couping.src.semantic_mapping.object_pose_refinement import (
    ObjectPoseRefinementConfig,
    ObjectPoseRefiner,
    create_object_feature_matcher,
    write_pose_refinement_debug,
)
from streaming_couping.src.semantic_mapping.object_pose_loss_refinement import (
    ObjectPoseLossRefinementConfig,
    ObjectPoseLossRefiner,
)
from streaming_couping.src.semantic_mapping.pipeline import SemanticMapPipeline
from streaming_couping.src.semantic_mapping.pipeline import (
    SemanticMapPoseRefinementRun,
)
from streaming_couping.src.semantic_mapping.temporal_consensus import (
    TemporalConsensusConfig,
)

def main() -> None:
    args = _parse_args()
    prompts = _normalize_prompts(args.prompts)
    builder_policy = (
        "raw" if args.fusion_policy == "both" else args.fusion_policy
    )
    mapper = SemanticMapBuilder(
        SemanticMapConfig(
            fusion_policy=builder_policy,
            voxel_size_m=args.voxel_size_m,
            min_geometry_confidence=args.min_geometry_confidence,
            min_track_score=args.min_track_score,
            static_score_threshold=args.static_score_threshold,
            require_static_score=args.require_static_score,
            include_dynamic_tracks=not args.exclude_dynamic_tracks,
            object_only=args.object_only,
            max_points_per_observation=args.max_points_per_observation,
            max_points_per_track=args.max_points_per_track,
            max_voxels=args.max_voxels,
            max_scene_voxels=args.max_scene_voxels,
            map_write_gate=MapWriteGateConfig(
                enabled=args.map_write_gate,
                min_observations_before_write=args.map_write_min_observations,
                reentry_confirmation_frames=args.map_write_reentry_confirmation,
                reentry_gap=args.map_write_reentry_gap,
                min_mask_pixels=args.map_write_min_mask_pixels,
                min_observation_points=args.map_write_min_observation_points,
                min_reliability=args.map_write_min_reliability,
                min_reprojection_consistency=(
                    args.map_write_min_reprojection_consistency
                ),
                min_area_ratio=args.map_write_min_area_ratio,
                max_area_ratio=args.map_write_max_area_ratio,
                soft_weight_floor=args.map_write_soft_weight_floor,
            ),
            temporal_consensus=TemporalConsensusConfig(
                history_frames=args.temporal_consensus_history_frames,
                max_history_points=args.temporal_consensus_max_history_points,
                min_history_points=args.temporal_consensus_min_history_points,
                support_radius_m=args.temporal_consensus_support_radius_m,
                min_support_points=args.temporal_consensus_min_support_points,
                min_support_ratio=args.temporal_consensus_min_support_ratio,
                max_novel_points=args.temporal_consensus_max_novel_points,
                novel_weight=args.temporal_consensus_novel_weight,
            ),
            instance_point_consistency=InstancePointConsistencyConfig(
                enabled=bool(args.instance_point_consistency),
                history_frames=args.instance_consistency_history_frames,
                max_history_points=args.instance_consistency_max_history_points,
                min_history_points=args.instance_consistency_min_history_points,
                support_radius_m=args.instance_consistency_support_radius_m,
                min_support_points=args.instance_consistency_min_support_points,
                min_support_ratio=args.instance_consistency_min_support_ratio,
                bounds_margin_m=args.instance_consistency_bounds_margin_m,
                max_novel_points=args.instance_consistency_max_novel_points,
                novel_weight=args.instance_consistency_novel_weight,
            ),
        )
    )
    if args.cache:
        result = _run_from_cache(args, prompts, mapper)
    else:
        result = _run_from_rgb(args, prompts, mapper)
    if isinstance(result, SemanticMapPoseRefinementRun):
        _export_pose_refinement_run(result, args.output_dir)
    elif isinstance(result, dict):
        _export_branch_results(result, args.output_dir)
    else:
        _print_export_summary(export_semantic_map(result, args.output_dir))


def _run_from_cache(
    args: argparse.Namespace,
    prompts: tuple[str, ...],
    mapper: SemanticMapBuilder,
):
    if args.object_pose_refinement or args.object_pose_loss_refinement:
        raise ValueError(
            "object pose refinement currently requires RGB mode with a "
            "geometry provider that exposes camera poses; frozen V0 cache "
            "replay has no canonical intrinsics/pose contract."
        )
    payload = _load_payload(args.cache)
    payload = _select_cache_frames(
        payload,
        start=args.frame_start,
        stride=args.frame_stride,
        count=args.frame_count,
    )
    geometry = V0CacheGeometryAdapter(payload, scale_type=args.scale_type)
    segmentation = V0CacheSegmentationAdapter(
        payload,
        mask_field=args.mask_field,
    )
    image_paths = tuple(
        Path(path)
        for path in payload.get(
            "image_paths",
            [str(index) for index in range(len(geometry.frame_ids))],
        )
    )
    pipeline = SemanticMapPipeline(
        geometry=geometry,
        segmentation=segmentation,
        mapper=mapper,
    )
    return _execute_pipeline(
        pipeline,
        image_paths,
        prompts=prompts or None,
        metadata={
            "input_mode": "frozen_v0_cache",
            "cache": str(Path(args.cache).expanduser().resolve()),
            "scale_type": args.scale_type,
            "frame_selection": {
                "start": int(args.frame_start),
                "stride": int(args.frame_stride),
                "count": int(args.frame_count),
            },
        },
        fusion_policy=args.fusion_policy,
        instance_point_consistency=args.instance_point_consistency,
    )


def _run_from_rgb(
    args: argparse.Namespace,
    prompts: tuple[str, ...],
    mapper: SemanticMapBuilder,
):
    if not prompts:
        raise ValueError("--prompts is required when --cache is not used.")
    selection = resolve_rgb_inputs(
        frames=args.frames,
        manifest=args.manifest,
        scene_id=args.scene_id,
        dataset_frame_indices=args.dataset_frame_indices,
        start=args.frame_start,
        stride=args.frame_stride,
        count=args.frame_count,
    )
    image_paths = selection.image_paths
    if not image_paths:
        raise ValueError("No RGB images were selected from the requested input.")
    overrides = {}
    if args.sam_device:
        overrides["sam3_device"] = args.sam_device
    if args.geometry_device:
        overrides["geometry_device"] = args.geometry_device
    recovery = load_config(args.recovery_config, overrides)

    geometry_payload = None
    if args.geometry_cache is not None:
        geometry_payload = load_horizonstream_cache(args.geometry_cache)
        validate_horizonstream_cache(
            geometry_payload,
            expected_image_paths=image_paths,
        )
        geometry = HorizonStreamGeometryCacheAdapter(geometry_payload)
        geometry_backend = geometry.backend_name
        geometry_scale_type = "metric"
        segmentation_output_size = geometry.image_size
    else:
        from streaming_couping.src.backbones.streamvggt_wrapper import (
            StreamVGGTWrapper,
        )

        stream = StreamVGGTWrapper(
            repo_path=recovery.streamvggt_repo,
            checkpoint_path=recovery.streamvggt_checkpoint,
            device=recovery.geometry_device,
            image_mode=recovery.image_mode,
            streaming_cache=recovery.streaming_cache,
        ).load()
        geometry = StreamVGGTGeometryAdapter(
            stream,
            scale_type=args.scale_type,
        )
        geometry_backend = geometry.backend_name
        geometry_scale_type = args.scale_type
        segmentation_output_size = tuple(args.output_size or recovery.output_size)

    print(
        "semantic map runtime "
        f"frames={len(image_paths)} "
        f"geometry_backend={geometry_backend} "
        f"sam_device={recovery.sam3_device} "
        f"sam_grounding_batch={args.sam_grounding_batch_size} "
        f"sam_video_cpu_offload={int(args.sam_offload_video_to_cpu)} "
        f"object_pose_refinement={int(args.object_pose_refinement or args.object_pose_loss_refinement)} "
        f"object_pose_loss_refinement={int(args.object_pose_loss_refinement)}"
    )
    from streaming_couping.src.backbones.sam3_wrapper import SAM3Wrapper

    sam = SAM3Wrapper(
        repo_path=recovery.sam3_repo,
        checkpoint_path=recovery.sam3_checkpoint,
        device=recovery.sam3_device,
        output_threshold=recovery.sam3_output_threshold,
        prompt_with_box=recovery.prompt_with_box,
        version=recovery.sam3_version,
        use_fa3=recovery.sam3_use_fa3,
        max_num_objects=recovery.sam3_max_num_objects,
        multiplex_count=recovery.sam3_multiplex_count,
        grounding_batch_size=args.sam_grounding_batch_size,
        offload_video_to_cpu=args.sam_offload_video_to_cpu,
    ).load()
    segmentation_kwargs = {
        "output_size": segmentation_output_size,
        "max_objects_per_prompt": recovery.sam3_max_num_objects,
        "max_total_objects": args.max_objects,
        "min_birth_pixels": recovery.min_pixels,
    }
    if args.geometry_guidance:
        segmentation = GeometryAwareSAM31SegmentationAdapter(
            sam,
            **segmentation_kwargs,
            geometry_config=args.geometry_guidance_config,
        )
    else:
        segmentation = SAM31SegmentationAdapter(sam, **segmentation_kwargs)
    object_pose_refiner = None
    if args.object_pose_loss_refinement:
        object_pose_config = _object_pose_loss_config(args)
        object_pose_refiner = ObjectPoseLossRefiner(object_pose_config)
    elif args.object_pose_refinement:
        object_pose_config = _object_pose_config(args)
        object_pose_refiner = ObjectPoseRefiner(
            object_pose_config,
            matcher=create_object_feature_matcher(object_pose_config),
        )
    pipeline = SemanticMapPipeline(
        geometry=geometry,
        segmentation=segmentation,
        mapper=mapper,
    )
    metadata = {
        "input_mode": "rgb_and_text_prompts",
        "scale_type": geometry_scale_type,
        "recovery_config": str(Path(args.recovery_config).resolve()),
        "geometry_cache": (
            None
            if args.geometry_cache is None
            else str(args.geometry_cache.expanduser().resolve())
        ),
        "source_image_paths": [str(path) for path in image_paths],
        "source_positions": list(selection.source_positions),
        "source_count": int(selection.source_count),
        "frame_selection": {
            "start": int(args.frame_start),
            "stride": int(args.frame_stride),
            "count": int(args.frame_count),
        },
        "geometry_guidance": bool(args.geometry_guidance),
        "map_write_gate": bool(args.map_write_gate),
        "instance_point_consistency_requested": bool(
            args.instance_point_consistency
        ),
        **selection.metadata,
    }
    if geometry_payload is None:
        return _execute_pipeline(
            pipeline,
            image_paths,
            prompts=prompts,
            metadata=metadata,
            fusion_policy=args.fusion_policy,
            object_pose_refiner=object_pose_refiner,
            instance_point_consistency=args.instance_point_consistency,
        )

    # SAM must see the exact center-cropped pixels used for cached geometry.
    with tempfile.TemporaryDirectory(prefix="horizonstream_sam_rgb_") as directory:
        aligned_paths = materialize_horizonstream_rgb(geometry_payload, directory)
        return _execute_pipeline(
            pipeline,
            aligned_paths,
            prompts=prompts,
            metadata=metadata,
            fusion_policy=args.fusion_policy,
            object_pose_refiner=object_pose_refiner,
            instance_point_consistency=args.instance_point_consistency,
        )


def _execute_pipeline(
    pipeline: SemanticMapPipeline,
    image_paths: tuple[Path, ...] | list[Path],
    *,
    prompts: tuple[str, ...] | None,
    metadata: dict[str, Any],
    fusion_policy: str,
    object_pose_refiner: ObjectPoseRefiner | None = None,
    instance_point_consistency: bool = False,
):
    if instance_point_consistency:
        if object_pose_refiner is not None:
            raise ValueError(
                "--instance-point-consistency cannot be combined with "
                "--object-pose-refinement in the first map-only experiment."
            )
        if fusion_policy != "raw":
            raise ValueError(
                "--instance-point-consistency requires --fusion-policy raw; "
                "it automatically exports raw and instance-consistency branches."
            )
        return pipeline.run_branches(
            image_paths,
            prompts=prompts,
            metadata=metadata,
            policies=("raw", "instance_point_consistency"),
        )
    if object_pose_refiner is not None:
        policies = (
            ("raw", "temporal_consensus")
            if fusion_policy == "both"
            else (fusion_policy,)
        )
        return pipeline.run_with_object_pose_refinement(
            image_paths,
            refiner=object_pose_refiner,
            prompts=prompts,
            metadata=metadata,
            policies=policies,
        )
    if fusion_policy == "both":
        return pipeline.run_branches(
            image_paths,
            prompts=prompts,
            metadata=metadata,
            policies=("raw", "temporal_consensus"),
        )
    return pipeline.run(
        image_paths,
        prompts=prompts,
        metadata=metadata,
    )


def _print_export_summary(summary: dict[str, Any]) -> None:
    print(
        "semantic map completed "
        f"policy={summary['metadata'].get('fusion_policy', 'raw')} "
        f"frames={summary['metadata'].get('frame_count', 0)} "
        f"scene_voxels={summary['scene_voxel_count']} "
        f"voxels={summary['voxel_count']} "
        f"objects={summary['instance_count']}"
    )
    print(f"semantic_map={summary['outputs']['artifact']}")
    print(f"scene_rgb_ply={summary['outputs']['scene_rgb_ply']}")
    print(f"scene_semantic_ply={summary['outputs']['scene_semantic_ply']}")
    print(f"semantic_ply={summary['outputs']['semantic_ply']}")
    print(f"object_plys={summary['outputs']['objects_dir']}")


def _export_branch_results(
    results: dict[str, Any],
    output_dir: Path,
) -> None:
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict[str, Any]] = {}
    for policy, result in results.items():
        summaries[policy] = export_semantic_map(
            result,
            root / policy,
            revision=f"semantic_mapping_{policy}_shared_inference_v1",
        )
        _print_export_summary(summaries[policy])
    comparison = {
        "schema": 1,
        "fusion_policy": list(summaries),
        "shared_model_inference": True,
        "gt_evaluation": "not_run_at_runtime",
        "branches": {
            policy: _branch_comparison_summary(summary)
            for policy, summary in summaries.items()
        },
        "outputs": {
            "branch_directories": {
                policy: str(root / policy)
                for policy in summaries
            },
            "comparison": str(root / "comparison.json"),
        },
    }
    comparison_path = root / "comparison.json"
    comparison_path.write_text(
        json.dumps(comparison, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"semantic map branch comparison={comparison_path}")


def _export_pose_refinement_run(
    run: SemanticMapPoseRefinementRun,
    output_dir: Path,
) -> None:
    """Export raw/refined maps and all pose-refinement provenance."""

    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    debug_dir = root / "object_pose_refinement"
    debug_paths = write_pose_refinement_debug(run.refinement, debug_dir)
    map_summaries: dict[str, dict[str, Any]] = {}
    policy_names = tuple(run.raw_results.keys())
    single_policy = len(policy_names) == 1
    for policy in policy_names:
        raw_result = run.raw_results[policy]
        refined_result = run.refined_results[policy]
        raw_name = "raw_pose" if single_policy else f"{policy}_raw_pose"
        refined_name = (
            "object_pose_refined"
            if single_policy
            else f"{policy}_object_pose_refined"
        )
        raw_summary = export_semantic_map(
            raw_result,
            root / raw_name,
            revision=f"semantic_mapping_{policy}_raw_pose_object_refinement_ablation_r1",
        )
        refined_summary = export_semantic_map(
            refined_result,
            root / refined_name,
            revision=f"semantic_mapping_{policy}_sam_object_pose_refined_r1",
        )
        map_summaries[raw_name] = raw_summary
        map_summaries[refined_name] = refined_summary
        print(f"[{raw_name}]")
        _print_export_summary(raw_summary)
        print(f"[{refined_name}]")
        _print_export_summary(refined_summary)
    comparison = {
        "schema": 1,
        "revision": "sam_instance_guided_horizonstream_pose_refinement_ablation_r1",
        "raw_pose_map_is_baseline": True,
        "shared_geometry_segmentation_inference": True,
        "gt_used_for_candidate_generation_or_optimization": False,
        "pose_refinement": {
            "debug_directory": str(debug_dir),
            "debug_outputs": debug_paths,
            "summary": dict(run.refinement.summary),
        },
        "maps": {
            name: _branch_comparison_summary(summary)
            for name, summary in map_summaries.items()
        },
        "outputs": {
            "debug_directory": str(debug_dir),
            "comparison": str(root / "pose_refinement_comparison.json"),
        },
    }
    comparison_path = root / "pose_refinement_comparison.json"
    comparison_path.write_text(
        json.dumps(comparison, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        "object pose refinement "
        f"candidates={run.refinement.summary['candidate_pair_count']} "
        f"accepted_edges={run.refinement.summary['accepted_edge_count']} "
        f"rejected_edges={run.refinement.summary['rejected_edge_count']}"
    )
    print(f"pose_refinement_summary={debug_paths['summary']}")
    print(f"pose_refinement_comparison={comparison_path}")


def _branch_comparison_summary(summary: dict[str, Any]) -> dict[str, Any]:
    metadata = summary.get("metadata", {})
    return {
        "fusion_policy": metadata.get("fusion_policy", "unknown"),
        "artifact": summary["outputs"]["artifact"],
        "semantic_ply": summary["outputs"]["semantic_ply"],
        "object_tracks_ply": summary["outputs"]["object_tracks_ply"],
        "scene_voxel_count": int(summary["scene_voxel_count"]),
        "voxel_count": int(summary["voxel_count"]),
        "labeled_voxel_count": int(summary["labeled_voxel_count"]),
        "instance_count": int(summary["instance_count"]),
        "static_instance_count": int(summary["static_instance_count"]),
        "temporal_consensus": metadata.get("temporal_consensus", {}),
        "instance_point_consistency": metadata.get(
            "instance_point_consistency", {}
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--cache",
        type=Path,
        help="Frozen V0 feature cache; skips StreamVGGT and SAM inference.",
    )
    source.add_argument(
        "--frames",
        nargs="+",
        type=Path,
        help="RGB files and/or directories, sorted within each directory.",
    )
    source.add_argument(
        "--manifest",
        type=Path,
        help="Processed ScanNet++ manifest containing scene RGB paths.",
    )
    parser.add_argument(
        "--scene-id",
        default=None,
        help="Scene ID used with --manifest.",
    )
    parser.add_argument(
        "--dataset-frame-indices",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Frame positions in the manifest scene. Omit to use every scene "
            "frame; selection is then applied in manifest order."
        ),
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        help="Text prompts, either as separate arguments or comma-separated.",
    )
    parser.add_argument(
        "--geometry-cache",
        type=Path,
        default=None,
        help=(
            "HorizonStream geometry cache for RGB mode. Geometry is not rerun "
            "and StreamVGGT is not loaded."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for semantic_map.pt, PLY, and JSON artifacts.",
    )
    parser.add_argument(
        "--fusion-policy",
        choices=("raw", "temporal_consensus", "both"),
        default="raw",
        help=(
            "Map write policy. 'both' runs raw and temporal_consensus using "
            "one shared geometry/SAM inference."
        ),
    )
    parser.add_argument(
        "--recovery-config",
        type=Path,
        default=Path("streaming_couping/configs/recovery_dynamic_instance.yaml"),
    )
    parser.add_argument("--geometry-device", default=None)
    parser.add_argument("--sam-device", default=None)
    parser.add_argument(
        "--sam-grounding-batch-size",
        type=int,
        default=4,
        help="Frames per SAM3.1 grounding batch; lower values reduce peak VRAM.",
    )
    parser.add_argument(
        "--sam-offload-video-to-cpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep decoded SAM video frames in CPU memory until they are needed.",
    )
    parser.add_argument("--output-size", type=_parse_size, default=None)
    parser.add_argument(
        "--frame-start",
        type=int,
        default=0,
        help="0-based position in the sorted RGB inputs at which to start.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Take every Nth RGB input after frame-start.",
    )
    parser.add_argument(
        "--frame-count",
        type=int,
        default=0,
        help="Number of selected RGB inputs; 0 means all remaining inputs.",
    )
    parser.add_argument("--max-objects", type=int, default=16)
    parser.add_argument("--mask-field", default="tracking_masks_stream")
    parser.add_argument(
        "--scale-type",
        choices=("metric", "relative", "unknown"),
        default="unknown",
    )
    parser.add_argument("--voxel-size-m", type=float, default=0.05)
    parser.add_argument("--min-geometry-confidence", type=float, default=0.30)
    parser.add_argument("--min-track-score", type=float, default=0.50)
    parser.add_argument("--static-score-threshold", type=float, default=0.20)
    parser.add_argument("--require-static-score", action="store_true")
    parser.add_argument("--exclude-dynamic-tracks", action="store_true")
    parser.add_argument("--max-points-per-observation", type=int, default=8_000)
    parser.add_argument("--max-points-per-track", type=int, default=250_000)
    parser.add_argument("--max-voxels", type=int, default=1_000_000)
    parser.add_argument(
        "--max-scene-voxels",
        type=int,
        default=1_000_000,
        help="Maximum full-scene voxels retained in exported artifacts.",
    )
    parser.add_argument(
        "--object-only",
        action="store_true",
        help=(
            "Do not accumulate the parallel full-scene voxel map; retain only "
            "prompted object-level outputs."
        ),
    )
    parser.add_argument(
        "--geometry-guidance",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use causal previous-world-point prompts to compete with raw SAM; "
            "disabled by default."
        ),
    )
    parser.add_argument(
        "--geometry-guidance-prompt-mode",
        choices=("box_points", "box_only", "points_only"),
        default="box_points",
    )
    parser.add_argument("--geometry-guidance-min-history-points", type=int, default=16)
    parser.add_argument("--geometry-guidance-max-history-points", type=int, default=4096)
    parser.add_argument("--geometry-guidance-max-history-frames", type=int, default=8)
    parser.add_argument("--geometry-guidance-max-history-gap", type=int, default=30)
    parser.add_argument("--geometry-guidance-max-positive-points", type=int, default=6)
    parser.add_argument(
        "--geometry-guidance-min-candidate-support",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--geometry-guidance-reliable-support-margin",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--geometry-guidance-reliable-score-margin",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--geometry-guidance-max-area-ratio",
        type=float,
        default=4.0,
    )
    parser.add_argument(
        "--object-pose-refinement",
        action="store_true",
        help=(
            "Opt in to SAM persistent-instance pose refinement. The command "
            "exports raw and refined maps separately; default is unchanged."
        ),
    )
    parser.add_argument(
        "--object-pose-loss-refinement",
        action="store_true",
        help=(
            "Opt in to causal SAM-instance object point-cloud alignment loss. "
            "Only the current pose is optimized against earlier anchors; the "
            "default baseline is unchanged."
        ),
    )
    parser.add_argument("--object-pose-loss-anchor-frames", type=int, default=5)
    parser.add_argument("--object-pose-loss-max-anchor-observations", type=int, default=3)
    parser.add_argument("--object-pose-loss-max-history-observations", type=int, default=2)
    parser.add_argument("--object-pose-loss-max-points-per-observation", type=int, default=256)
    parser.add_argument("--object-pose-loss-min-points-per-observation", type=int, default=24)
    parser.add_argument("--object-pose-loss-min-track-score", type=float, default=0.50)
    parser.add_argument("--object-pose-loss-static-score-threshold", type=float, default=0.20)
    parser.add_argument("--object-pose-loss-require-static-score", action="store_true")
    parser.add_argument("--object-pose-loss-min-geometry-confidence", type=float, default=0.30)
    parser.add_argument("--object-pose-loss-min-mask-pixels", type=int, default=32)
    parser.add_argument("--object-pose-loss-max-mask-area-ratio", type=float, default=0.85)
    parser.add_argument("--object-pose-loss-max-match-distance-m", type=float, default=0.25)
    parser.add_argument("--object-pose-loss-trim-ratio", type=float, default=0.70)
    parser.add_argument("--object-pose-loss-min-matches-per-pair", type=int, default=8)
    parser.add_argument("--object-pose-loss-min-total-matches", type=int, default=16)
    parser.add_argument("--object-pose-loss-outer-iterations", type=int, default=4)
    parser.add_argument("--object-pose-loss-optimizer-steps", type=int, default=30)
    parser.add_argument("--object-pose-loss-learning-rate", type=float, default=0.03)
    parser.add_argument("--object-pose-loss-huber-delta-m", type=float, default=0.05)
    parser.add_argument("--object-pose-loss-pose-prior-weight", type=float, default=0.02)
    parser.add_argument("--object-pose-loss-anchor-reference-weight", type=float, default=1.0)
    parser.add_argument("--object-pose-loss-local-reference-weight", type=float, default=0.50)
    parser.add_argument("--object-pose-loss-max-correction-rotation-deg", type=float, default=10.0)
    parser.add_argument("--object-pose-loss-max-correction-translation-m", type=float, default=0.25)
    parser.add_argument("--object-pose-loss-min-relative-improvement", type=float, default=0.02)
    parser.add_argument("--object-pose-loss-device", default="cpu")
    parser.add_argument("--object-pose-min-gap", type=int, default=10)
    parser.add_argument("--object-pose-min-track-score", type=float, default=0.50)
    parser.add_argument("--object-pose-min-mask-pixels", type=int, default=32)
    parser.add_argument("--object-pose-max-mask-area-ratio", type=float, default=0.85)
    parser.add_argument("--object-pose-min-geometry-points", type=int, default=24)
    parser.add_argument(
        "--object-pose-min-geometry-confidence", type=float, default=0.30
    )
    parser.add_argument("--object-pose-max-pairs-per-instance", type=int, default=8)
    parser.add_argument("--object-pose-max-total-candidate-pairs", type=int, default=256)
    parser.add_argument(
        "--object-pose-feature-backend",
        choices=("rgb_patch", "dinov3"),
        default="rgb_patch",
        help="Mask-constrained feature backend; rgb_patch is model-free.",
    )
    parser.add_argument("--object-pose-feature-patch-size", type=int, default=8)
    parser.add_argument("--object-pose-feature-max-points", type=int, default=1024)
    parser.add_argument(
        "--object-pose-feature-cosine-threshold", type=float, default=0.65
    )
    parser.add_argument("--object-pose-feature-ratio-margin", type=float, default=0.02)
    parser.add_argument("--object-pose-max-feature-matches", type=int, default=256)
    parser.add_argument("--object-pose-ransac-iterations", type=int, default=256)
    parser.add_argument(
        "--object-pose-ransac-inlier-threshold", type=float, default=0.10
    )
    parser.add_argument("--object-pose-min-matches", type=int, default=12)
    parser.add_argument("--object-pose-min-inliers", type=int, default=6)
    parser.add_argument("--object-pose-min-inlier-ratio", type=float, default=0.30)
    parser.add_argument(
        "--object-pose-max-registration-rmse", type=float, default=0.12
    )
    parser.add_argument(
        "--object-pose-max-rotation-disagreement", type=float, default=60.0,
        help="Low-confidence gate in degrees; extreme disagreement is rejected.",
    )
    parser.add_argument(
        "--object-pose-max-translation-disagreement", type=float, default=1.0,
        help="Low-confidence gate in meters; extreme disagreement is rejected.",
    )
    parser.add_argument(
        "--object-pose-reject-extreme-disagreement-multiplier",
        type=float,
        default=2.0,
    )
    parser.add_argument("--object-pose-low-consistency-weight", type=float, default=0.25)
    parser.add_argument("--object-pose-weight", type=float, default=1.0)
    parser.add_argument("--object-pose-sequential-weight", type=float, default=10.0)
    parser.add_argument("--object-pose-huber-delta", type=float, default=0.10)
    parser.add_argument(
        "--object-pose-rotation-residual-scale", type=float, default=1.0
    )
    parser.add_argument(
        "--object-pose-max-pose-correction-rotation", type=float, default=45.0
    )
    parser.add_argument(
        "--object-pose-max-pose-correction-translation", type=float, default=1.0
    )
    parser.add_argument("--object-pose-optimizer-max-nfev", type=int, default=100)
    parser.add_argument("--object-pose-dinov3-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--object-pose-dinov3-root",
        type=Path,
        default=Path("/home/bod/86Nas/95_data_bak/FoundationModels/dinov3"),
    )
    parser.add_argument("--object-pose-dinov3-variant", default="dinov3-vitl16")
    parser.add_argument("--object-pose-dinov3-device", default="cuda:0")
    parser.add_argument("--object-pose-dinov3-dtype", default="bfloat16")
    parser.add_argument(
        "--map-write-gate",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable causal short/long object memory for semantic voxel writes; "
            "tracks remain available when a write is deferred."
        ),
    )
    parser.add_argument("--map-write-min-observations", type=int, default=2)
    parser.add_argument("--map-write-reentry-confirmation", type=int, default=2)
    parser.add_argument("--map-write-reentry-gap", type=int, default=3)
    parser.add_argument("--map-write-min-mask-pixels", type=int, default=32)
    parser.add_argument("--map-write-min-observation-points", type=int, default=16)
    parser.add_argument("--map-write-min-reliability", type=float, default=0.20)
    parser.add_argument(
        "--map-write-min-reprojection-consistency",
        type=float,
        default=0.20,
    )
    parser.add_argument("--map-write-min-area-ratio", type=float, default=0.35)
    parser.add_argument("--map-write-max-area-ratio", type=float, default=2.85)
    parser.add_argument("--map-write-soft-weight-floor", type=float, default=0.25)
    parser.add_argument("--temporal-consensus-history-frames", type=int, default=8)
    parser.add_argument(
        "--temporal-consensus-max-history-points",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--temporal-consensus-min-history-points",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--temporal-consensus-support-radius-m",
        type=float,
        default=0.08,
    )
    parser.add_argument(
        "--temporal-consensus-min-support-points",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--temporal-consensus-min-support-ratio",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--temporal-consensus-max-novel-points",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--temporal-consensus-novel-weight",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--instance-point-consistency",
        action="store_true",
        help=(
            "Opt in to causal persistent-instance point consistency. This "
            "exports raw and instance_point_consistency branches from one "
            "shared HorizonStream/SAM inference."
        ),
    )
    parser.add_argument(
        "--instance-point-consistency-history-frames",
        dest="instance_consistency_history_frames",
        type=int,
        default=12,
        help="Number of previous observations retained per persistent instance.",
    )
    parser.add_argument(
        "--instance-point-consistency-max-history-points",
        dest="instance_consistency_max_history_points",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--instance-point-consistency-min-history-points",
        dest="instance_consistency_min_history_points",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--instance-point-consistency-support-radius-m",
        dest="instance_consistency_support_radius_m",
        type=float,
        default=0.10,
        help="Historical 3D support radius in meters.",
    )
    parser.add_argument(
        "--instance-point-consistency-min-support-points",
        dest="instance_consistency_min_support_points",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--instance-point-consistency-min-support-ratio",
        dest="instance_consistency_min_support_ratio",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--instance-point-consistency-bounds-margin-m",
        dest="instance_consistency_bounds_margin_m",
        type=float,
        default=0.20,
        help="Margin around the historical object extent for new surfaces.",
    )
    parser.add_argument(
        "--instance-point-consistency-max-novel-points",
        dest="instance_consistency_max_novel_points",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--instance-point-consistency-novel-weight",
        dest="instance_consistency_novel_weight",
        type=float,
        default=0.25,
        help="Weight multiplier applied to retained novel points.",
    )
    args = parser.parse_args()
    if args.object_pose_refinement and args.object_pose_loss_refinement:
        parser.error(
            "--object-pose-refinement and --object-pose-loss-refinement "
            "are mutually exclusive."
        )
    from streaming_couping.src.semantic_mapping.geometry_guidance import (
        GeometryGuidanceConfig,
    )

    args.geometry_guidance_config = GeometryGuidanceConfig(
        min_history_points=args.geometry_guidance_min_history_points,
        max_history_points=args.geometry_guidance_max_history_points,
        max_history_frames=args.geometry_guidance_max_history_frames,
        max_history_gap=args.geometry_guidance_max_history_gap,
        prompt_mode=args.geometry_guidance_prompt_mode,
        max_positive_points=args.geometry_guidance_max_positive_points,
        min_candidate_support_recall=args.geometry_guidance_min_candidate_support,
        reliable_support_margin=args.geometry_guidance_reliable_support_margin,
        reliable_score_margin=args.geometry_guidance_reliable_score_margin,
        max_area_ratio=args.geometry_guidance_max_area_ratio,
    ).validate()
    if args.cache is not None and args.geometry_cache is not None:
        parser.error("--geometry-cache cannot be combined with the frozen V0 --cache.")
    return args


def _object_pose_config(args: argparse.Namespace) -> ObjectPoseRefinementConfig:
    return ObjectPoseRefinementConfig(
        min_temporal_gap=args.object_pose_min_gap,
        min_track_score=args.object_pose_min_track_score,
        min_mask_pixels=args.object_pose_min_mask_pixels,
        max_mask_area_ratio=args.object_pose_max_mask_area_ratio,
        min_geometry_points=args.object_pose_min_geometry_points,
        min_geometry_confidence=args.object_pose_min_geometry_confidence,
        max_pairs_per_instance=args.object_pose_max_pairs_per_instance,
        max_total_candidate_pairs=args.object_pose_max_total_candidate_pairs,
        feature_backend=args.object_pose_feature_backend,
        feature_patch_size=args.object_pose_feature_patch_size,
        feature_max_points=args.object_pose_feature_max_points,
        feature_cosine_threshold=args.object_pose_feature_cosine_threshold,
        feature_ratio_margin=args.object_pose_feature_ratio_margin,
        max_feature_matches=args.object_pose_max_feature_matches,
        ransac_iterations=args.object_pose_ransac_iterations,
        ransac_inlier_threshold_m=args.object_pose_ransac_inlier_threshold,
        min_matches=args.object_pose_min_matches,
        min_inliers=args.object_pose_min_inliers,
        min_inlier_ratio=args.object_pose_min_inlier_ratio,
        max_registration_rmse_m=args.object_pose_max_registration_rmse,
        max_rotation_disagreement_deg=args.object_pose_max_rotation_disagreement,
        max_translation_disagreement_m=args.object_pose_max_translation_disagreement,
        reject_extreme_disagreement_multiplier=(
            args.object_pose_reject_extreme_disagreement_multiplier
        ),
        low_consistency_weight=args.object_pose_low_consistency_weight,
        sequential_edge_weight=args.object_pose_sequential_weight,
        object_edge_weight=args.object_pose_weight,
        huber_delta=args.object_pose_huber_delta,
        rotation_residual_scale_m=args.object_pose_rotation_residual_scale,
        max_pose_delta_rotation_deg=args.object_pose_max_pose_correction_rotation,
        max_pose_delta_translation_m=args.object_pose_max_pose_correction_translation,
        optimizer_max_nfev=args.object_pose_optimizer_max_nfev,
        dinov3_checkpoint=args.object_pose_dinov3_checkpoint,
        dinov3_root=args.object_pose_dinov3_root,
        dinov3_variant=args.object_pose_dinov3_variant,
        dinov3_device=args.object_pose_dinov3_device,
        dinov3_dtype=args.object_pose_dinov3_dtype,
    ).validate()


def _object_pose_loss_config(
    args: argparse.Namespace,
) -> ObjectPoseLossRefinementConfig:
    return ObjectPoseLossRefinementConfig(
        anchor_frame_count=args.object_pose_loss_anchor_frames,
        max_anchor_observations=args.object_pose_loss_max_anchor_observations,
        max_history_observations=args.object_pose_loss_max_history_observations,
        max_points_per_observation=args.object_pose_loss_max_points_per_observation,
        min_points_per_observation=args.object_pose_loss_min_points_per_observation,
        min_track_score=args.object_pose_loss_min_track_score,
        static_score_threshold=args.object_pose_loss_static_score_threshold,
        require_static_score=args.object_pose_loss_require_static_score,
        min_geometry_confidence=args.object_pose_loss_min_geometry_confidence,
        min_mask_pixels=args.object_pose_loss_min_mask_pixels,
        max_mask_area_ratio=args.object_pose_loss_max_mask_area_ratio,
        max_match_distance_m=args.object_pose_loss_max_match_distance_m,
        trim_ratio=args.object_pose_loss_trim_ratio,
        min_matches_per_pair=args.object_pose_loss_min_matches_per_pair,
        min_total_matches=args.object_pose_loss_min_total_matches,
        outer_iterations=args.object_pose_loss_outer_iterations,
        optimizer_steps=args.object_pose_loss_optimizer_steps,
        learning_rate=args.object_pose_loss_learning_rate,
        huber_delta_m=args.object_pose_loss_huber_delta_m,
        pose_prior_weight=args.object_pose_loss_pose_prior_weight,
        anchor_reference_weight=args.object_pose_loss_anchor_reference_weight,
        local_reference_weight=args.object_pose_loss_local_reference_weight,
        max_correction_rotation_deg=args.object_pose_loss_max_correction_rotation_deg,
        max_correction_translation_m=args.object_pose_loss_max_correction_translation_m,
        min_relative_loss_improvement=args.object_pose_loss_min_relative_improvement,
        device=args.object_pose_loss_device,
    ).validate()


def _load_payload(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"V0 cache does not exist: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a dictionary V0 cache, got {type(payload)!r}.")
    return payload


def _expand_image_paths(values: list[Path] | None) -> tuple[Path, ...]:
    return expand_image_paths(values)


def _expand_manifest_paths(
    manifest_path: Path,
    *,
    scene_id: str | None,
    frame_indices: list[int] | None,
) -> tuple[Path, ...]:
    paths, _, _ = expand_manifest_paths(
        manifest_path,
        scene_id=scene_id,
        frame_indices=frame_indices,
    )
    return paths


def _select_cache_frames(
    payload: dict[str, Any],
    *,
    start: int,
    stride: int,
    count: int,
) -> dict[str, Any]:
    """Apply the same deterministic frame selection to a V0 cache replay."""

    frame_indices = payload.get("frame_indices")
    if frame_indices is None:
        total = _cache_frame_count(payload)
        frame_indices = list(range(total))
    total = len(frame_indices)
    selected_positions = _selected_positions(total, start, stride, count)
    if not selected_positions:
        raise ValueError("Frame selection produced no cache frames.")
    if len(selected_positions) == total and selected_positions == list(range(total)):
        return payload

    selected = dict(payload)
    frame_fields = (
        "frame_indices",
        "image_paths",
        "stream_images",
        "images",
        "baseline_world_points",
        "baseline_world_confidence",
        "world_points",
        "confidence",
        "tracking_masks_stream",
        "tracking_scores",
        "quality",
    )
    for name in frame_fields:
        if name not in selected:
            continue
        value = selected[name]
        try:
            if len(value) != total:
                continue
            if isinstance(value, (list, tuple)):
                selected[name] = type(value)(
                    value[index] for index in selected_positions
                )
            elif hasattr(value, "__getitem__"):
                selected[name] = value[selected_positions]
        except (TypeError, IndexError, KeyError):
            continue
    return selected


def _cache_frame_count(payload: dict[str, Any]) -> int:
    for name in (
        "baseline_world_points",
        "world_points",
        "tracking_masks_stream",
        "stream_images",
        "images",
    ):
        value = payload.get(name)
        if value is not None:
            return int(len(value))
    raise ValueError("V0 cache does not expose a frame-indexed field.")


def _select_image_paths(
    paths: tuple[Path, ...],
    *,
    start: int,
    stride: int,
    count: int,
) -> tuple[Path, ...]:
    """Select a deterministic subsequence from already sorted RGB inputs."""

    start = int(start)
    stride = int(stride)
    count = int(count)
    return select_image_paths(paths, start=start, stride=stride, count=count)


def _selected_positions(
    total: int,
    start: int,
    stride: int,
    count: int,
) -> list[int]:
    return selected_positions(total, start, stride, count)


def _normalize_prompts(values: list[str] | None) -> tuple[str, ...]:
    if not values:
        return ()
    output = []
    for value in values:
        output.extend(piece.strip() for piece in value.split(",") if piece.strip())
    return tuple(output)


def _parse_size(value: str) -> tuple[int, int]:
    pieces = value.lower().replace("x", ",").split(",")
    if len(pieces) != 2:
        raise argparse.ArgumentTypeError("Expected SIZE as H,W or HxW.")
    try:
        height, width = (int(piece) for piece in pieces)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("SIZE values must be integers.") from exc
    if height <= 0 or width <= 0:
        raise argparse.ArgumentTypeError("SIZE values must be positive.")
    return height, width


if __name__ == "__main__":
    main()
