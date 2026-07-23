"""Cache frozen SAM3/StreamVGGT observations for lightweight adapter training."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Mapping

import torch

from vggtsam.adapters.sam3_intermediate import (
    SAM3IntermediateAdapter,
    load_sam3_image_model,
)
from vggtsam.adapters.streamvggt_latent import (
    StreamVGGTLatentAdapter,
    load_streamvggt_latent_model,
)

from ..backbones.sam3_wrapper import SAM3Wrapper
from ..backbones.streamvggt_wrapper import StreamVGGTWrapper
from ..config import load_config
from ..instance_observations import (
    InstanceRefinementConfig,
    load_instance_sequences,
    load_tracking_cache,
    save_tracking_cache,
    tracking_masks_to_geometry_grid,
)
from ..pointmap_alignment import prepare_map_evaluation
from ..pose_evaluation import _load_ground_truth_sequence
from ..recovery import output_mask_to_stream
from ..tracking_recovery import run_natural_recovery_tracking
from ..types import TrackingSequence
from .config import ClipConfig, LearnedPoseConfig
from .observations import (
    build_geometry_observations,
    pool_sam_instance_features,
    sample_instance_uvd,
)


CACHE_VERSION = 2


def build_feature_caches(config: LearnedPoseConfig) -> list[Path]:
    """Build geometry/tracking first, unload video SAM3, then pool SAM features."""

    config.features.cache_dir.mkdir(parents=True, exist_ok=True)
    paths = [cache_path(config, clip) for clip in config.clips]
    pending = [
        (clip, path)
        for clip, path in zip(config.clips, paths)
        if config.features.rebuild or not _cache_complete(path, clip=clip)
    ]
    if not pending:
        print("learned-pose feature caches are complete")
        return paths

    stream_model = load_streamvggt_latent_model(
        repo_path=load_config(config.recovery_config).streamvggt_repo,
        checkpoint_path=load_config(config.recovery_config).streamvggt_checkpoint,
        device=config.geometry_device,
        strict=True,
    )
    stream_adapter = StreamVGGTLatentAdapter(
        stream_model,
        device=config.geometry_device,
        image_mode=load_config(config.recovery_config).image_mode,
        dpt_layer_indices=config.fusion.dpt_layer_indices,
    )
    sam_video_holder: dict[str, SAM3Wrapper] = {}
    for clip, path in pending:
        print(f"caching frozen geometry/tracking clip={clip.name}")
        partial = _build_geometry_cache(
            config,
            clip,
            stream_adapter=stream_adapter,
            sam_video_holder=sam_video_holder,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(partial, path)

    # The image model and video predictor are both large. Never keep both on
    # the SAM device while extracting pooled appearance descriptors.
    sam_video_holder.clear()
    gc.collect()
    _empty_cuda_cache()
    recovery = load_config(config.recovery_config)
    sam_image_model = load_sam3_image_model(
        repo_path=recovery.sam3_repo,
        checkpoint_path=recovery.sam3_checkpoint,
        device=config.sam3_device,
        enable_segmentation=False,
        enable_inst_interactivity=False,
    )
    sam_adapter = SAM3IntermediateAdapter(
        sam_image_model,
        device=config.sam3_device,
        resolution=config.features.sam_resolution,
        source=config.features.sam_source,
        text_conditioning="none",
        token_grid=config.features.sam_grid,
    )
    for clip, path in pending:
        print(f"pooling frozen SAM3 observations clip={clip.name}")
        payload = load_feature_cache(path, require_complete=False)
        output = sam_adapter.extract_from_paths(
            payload["image_paths"],
            prompt="object",
        )
        height, width = config.features.sam_grid
        tokens = output.semantic.tokens[0]
        sequence = len(payload["frame_indices"])
        if tokens.shape[0] != sequence * height * width:
            raise RuntimeError(
                "SAM3 token count does not match clip/grid: "
                f"{tokens.shape[0]} vs {sequence}*{height}*{width}."
            )
        spatial = tokens.reshape(sequence, height, width, -1).permute(0, 3, 1, 2)
        appearance = pool_sam_instance_features(
            spatial,
            payload["tracking_masks_output"],
        )
        payload["appearance"] = appearance.float()
        payload["appearance_dim"] = int(appearance.shape[-1])
        payload["complete"] = True
        torch.save(payload, path)
    del sam_adapter, sam_image_model, stream_adapter, stream_model
    gc.collect()
    _empty_cuda_cache()
    return paths


def cache_path(config: LearnedPoseConfig, clip: ClipConfig) -> Path:
    return config.features.cache_dir / f"{clip.name}.pt"


def load_feature_cache(path: str | Path, *, require_complete: bool = True) -> dict:
    path = Path(path)
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict) or int(value.get("cache_version", -1)) != CACHE_VERSION:
        raise ValueError(f"Unsupported learned-pose cache: {path}")
    if require_complete and not bool(value.get("complete", False)):
        raise ValueError(f"Learned-pose cache is incomplete: {path}")
    return value


def _build_geometry_cache(
    config: LearnedPoseConfig,
    clip: ClipConfig,
    *,
    stream_adapter: StreamVGGTLatentAdapter,
    sam_video_holder: dict[str, SAM3Wrapper],
) -> dict:
    recovery = load_config(
        config.recovery_config,
        {
            "manifest": config.manifest,
            "scene_id": clip.scene_id,
            "frame_indices": clip.frame_indices,
            "sam3_device": config.sam3_device,
            "geometry_device": config.geometry_device,
            "output_dir": config.features.cache_dir / clip.name,
        },
    )
    sequences, target_masks = load_instance_sequences(
        recovery,
        instance_ids=clip.instance_ids,
        reference_sequence_index=clip.reference_sequence_index,
    )
    shared = sequences[int(clip.instance_ids[0])]
    output = stream_adapter.extract_from_paths(
        shared.image_paths,
        return_pointmap=True,
        streaming_cache=recovery.streaming_cache,
    )
    geometry_sequence = StreamVGGTWrapper._geometry_from_output(
        output,
        shared.image_paths,
    )
    recovered = _load_or_run_tracking(
        recovery,
        clip,
        sequences=sequences,
        target_masks=target_masks,
        geometry=geometry_sequence,
        sam_video_holder=sam_video_holder,
    )
    grid_masks_by_id = tracking_masks_to_geometry_grid(
        recovered,
        geometry=geometry_sequence,
        image_mode=recovery.image_mode,
    )
    for instance_id in clip.instance_ids:
        grid_masks_by_id[int(instance_id)][clip.reference_sequence_index] = output_mask_to_stream(
            target_masks[int(instance_id)][clip.reference_sequence_index],
            source_size=geometry_sequence.source_sizes[clip.reference_sequence_index],
            processed_size=geometry_sequence.processed_size,
            image_mode=recovery.image_mode,
        )
    grid_masks = torch.stack(
        [grid_masks_by_id[int(instance_id)] for instance_id in clip.instance_ids],
        dim=1,
    )
    tracking_masks_output = torch.stack(
        [recovered[int(instance_id)].masks for instance_id in clip.instance_ids],
        dim=1,
    )
    for slot, instance_id in enumerate(clip.instance_ids):
        tracking_masks_output[
            clip.reference_sequence_index,
            slot,
        ] = target_masks[int(instance_id)][clip.reference_sequence_index]
    scores = torch.stack(
        [recovered[int(instance_id)].scores for instance_id in clip.instance_ids],
        dim=1,
    )
    refinement = InstanceRefinementConfig(
        min_instance_points=config.features.min_instance_points,
        compute_device=config.geometry_device,
    )
    observations = build_geometry_observations(
        world_points=geometry_sequence.world_points,
        confidence=geometry_sequence.confidence,
        masks=grid_masks,
        scores=scores,
        instance_ids=clip.instance_ids,
        frame_indices=clip.frame_indices,
        reference_index=clip.reference_sequence_index,
        confidence_threshold=config.features.point_confidence_threshold,
        refinement=refinement,
        sampled_instance_points=config.features.sampled_instance_points,
    )
    depth = output.geometry.aux["depth_dense"].detach().float().cpu()
    depth_confidence = output.geometry.aux["depth_confidence_dense"].detach().float().cpu()
    instance_uvd, uvd_valid, rigid_weight = sample_instance_uvd(
        depth,
        depth_confidence,
        grid_masks,
        observations["quality"],
        max_points=config.features.sampled_instance_points,
    )
    trusted_for_rigid = (
        (observations["quality"][..., 0] >= config.fusion.min_track_confidence)
        & (observations["quality"][..., 1] >= config.fusion.min_geometry_confidence)
        & (observations["quality"][..., 2] >= config.fusion.min_static_score)
    )
    rigid_weight = rigid_weight * trusted_for_rigid.float()
    ground_truth = _load_ground_truth_sequence(
        config.manifest,
        scene_id=clip.scene_id,
        frame_indices=clip.frame_indices,
    )
    point_alignment = prepare_map_evaluation(
        recovery,
        scene_id=clip.scene_id,
        frame_indices=clip.frame_indices,
        geometry=geometry_sequence,
        reference_frame_idx=clip.reference_sequence_index,
    )
    processed_intrinsics = _processed_intrinsics(
        ground_truth.intrinsics,
        geometry_sequence.source_sizes,
        image_mode=recovery.image_mode,
    )
    target_pose_encoding = _target_pose_encoding(
        ground_truth.world_to_camera,
        processed_intrinsics,
        image_size=geometry_sequence.processed_size,
        reference_index=clip.reference_sequence_index,
        native_to_metric_scale=float(point_alignment.sim3_scale),
    )
    target_depth = _target_depth(
        point_alignment.gt_pointmaps,
        ground_truth.world_to_camera,
    )
    dpt_tokens = output.geometry.aux["stream_dpt_tokens"]
    camera_hidden = output.geometry.aux.get("stream_camera_hidden")
    if camera_hidden is None:
        raise RuntimeError(
            "StreamVGGT adapter did not expose the exact CameraHead input."
        )
    camera_hidden = camera_hidden.detach().float().cpu()[0]
    payload: dict[str, Any] = {
        "cache_version": CACHE_VERSION,
        "complete": False,
        "clip_name": clip.name,
        "split": clip.split,
        "scene_id": clip.scene_id,
        "frame_indices": list(clip.frame_indices),
        "instance_ids": list(clip.instance_ids),
        "reference_sequence_index": clip.reference_sequence_index,
        "image_paths": [str(path) for path in shared.image_paths],
        "image_size": list(geometry_sequence.processed_size),
        "patch_start_idx": int(output.geometry.aux["patch_start_idx"]),
        # Keep the frozen-head inputs in fp32.  The module-off control is
        # required to reproduce the actual StreamVGGT outputs, not an fp16
        # cache approximation of them.
        "camera_hidden": camera_hidden.float(),
        "baseline_pose_encoding": output.geometry.camera_tokens.detach().float().cpu()[0],
        "baseline_depth": depth.float(),
        "baseline_world_points": geometry_sequence.world_points.float(),
        "geometry": observations["geometry"].float(),
        "quality": observations["quality"].float(),
        "observed": observations["observed"].bool(),
        "geometry_feature_names": observations["geometry_feature_names"],
        "quality_names": observations["quality_names"],
        "geometry_dim": int(observations["geometry"].shape[-1]),
        "scene_origin": observations["scene_origin"].float(),
        "scene_scale": float(observations["scene_scale"]),
        "tracking_masks_output": tracking_masks_output.bool(),
        "tracking_masks_stream": grid_masks.bool(),
        "tracking_scores": scores.float(),
        "instance_uvd": instance_uvd.float(),
        "instance_uvd_valid": uvd_valid.bool(),
        "instance_rigid_weight": rigid_weight.float(),
        "target_pose_encoding": target_pose_encoding.float(),
        "target_world_to_camera": ground_truth.world_to_camera.float(),
        "target_world_points": point_alignment.gt_pointmaps.float(),
        "target_depth": target_depth.float(),
        "point_alignment_scale": float(point_alignment.sim3_scale),
        "point_alignment_rotation": point_alignment.sim3_rotation.float(),
        "point_alignment_translation": point_alignment.sim3_translation.float(),
    }
    payload["dpt_layer_indices"] = list(config.fusion.dpt_layer_indices)
    payload["token_levels"] = torch.stack(
        [value.detach().float().cpu()[0] for value in dpt_tokens],
        dim=0,
    )
    payload["stream_images"] = output.geometry.aux["stream_images"].detach().float().cpu()
    return payload


def _load_or_run_tracking(
    recovery,
    clip: ClipConfig,
    *,
    sequences: Mapping[int, object],
    target_masks: Mapping[int, torch.Tensor],
    geometry,
    sam_video_holder: dict[str, SAM3Wrapper],
) -> dict[int, TrackingSequence]:
    path = clip.tracking_cache or (recovery.output_dir / "tracking_cache.npz")
    cached = load_tracking_cache(
        path,
        config=recovery,
        instance_ids=clip.instance_ids,
        frame_indices=clip.frame_indices,
    )
    if cached is not None:
        print(f"reusing tracking cache: {path}")
        return cached[1]
    if "model" not in sam_video_holder:
        sam_video_holder["model"] = SAM3Wrapper(
            repo_path=recovery.sam3_repo,
            checkpoint_path=recovery.sam3_checkpoint,
            device=recovery.sam3_device,
            output_threshold=recovery.sam3_output_threshold,
            prompt_with_box=recovery.prompt_with_box,
        ).load()
    original: dict[int, TrackingSequence] = {}
    recovered: dict[int, TrackingSequence] = {}
    rows = []
    for instance_id in clip.instance_ids:
        result = run_natural_recovery_tracking(
            recovery,
            sequence=sequences[int(instance_id)],
            target_masks=target_masks[int(instance_id)],
            geometry=geometry,
            sam3=sam_video_holder["model"],
        )
        original[int(instance_id)] = result["original"]
        recovered[int(instance_id)] = result["recovered"]
        rows.append(
            {
                "instance_id": int(instance_id),
                "recovery_applied": int(result["recovery_applied"]),
                "recovery_sequence_index": result["recovery_sequence_index"],
                "recovery_frame_index": result["recovery_frame_index"],
                "recovery_reason": result["recovery_reason"],
                "selected_support_coverage": result["selected_support_coverage"],
                "selected_candidate_gt_iou": result["selected_candidate_gt_iou"],
            }
        )
    save_tracking_cache(
        path,
        config=recovery,
        instance_ids=clip.instance_ids,
        frame_indices=clip.frame_indices,
        original=original,
        recovered=recovered,
        tracking_rows=rows,
    )
    return recovered


def _processed_intrinsics(
    intrinsics: torch.Tensor,
    source_sizes,
    *,
    image_mode: str,
) -> torch.Tensor:
    from test_sam.coordinates import streamvggt_image_transform

    output = intrinsics.clone().double()
    for index, source_size in enumerate(source_sizes):
        transform = streamvggt_image_transform(source_size, mode=image_mode)
        sx, sy = transform.scale_xy
        ox, oy = transform.offset_xy
        output[index, 0, 0] *= sx
        output[index, 1, 1] *= sy
        output[index, 0, 2] = (output[index, 0, 2] + 0.5) * sx - 0.5 + ox
        output[index, 1, 2] = (output[index, 1, 2] + 0.5) * sy - 0.5 + oy
    return output


def _target_pose_encoding(
    world_to_camera: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    image_size: tuple[int, int],
    reference_index: int,
    native_to_metric_scale: float,
) -> torch.Tensor:
    from streamvggt.utils.pose_enc import extri_intri_to_pose_encoding

    world_to_camera = world_to_camera.double()
    reference_c2w = torch.linalg.inv(world_to_camera[int(reference_index)])
    relative = world_to_camera @ reference_c2w
    # StreamVGGT pose translations and predicted depths share its native
    # arbitrary scale.  The reference-frame pointmap Sim(3) maps that native
    # scale to metric GT, so pose supervision must use the inverse scale.
    # Keeping metric translations here would teach the adapter to destroy the
    # fixed-reference alignment used by every evaluation metric.
    scale = max(float(native_to_metric_scale), 1e-8)
    relative = relative.clone()
    relative[:, :3, 3] /= scale
    return extri_intri_to_pose_encoding(
        relative[None, :, :3, :4].float(),
        intrinsics[None].float(),
        image_size_hw=image_size,
    )[0]


def _target_depth(
    world_points: torch.Tensor,
    world_to_camera: torch.Tensor,
) -> torch.Tensor:
    rotation = world_to_camera[:, :3, :3].float()
    translation = world_to_camera[:, :3, 3].float()
    camera_points = torch.einsum("sij,shwj->shwi", rotation, world_points.float())
    camera_points = camera_points + translation[:, None, None, :]
    depth = camera_points[..., 2:3]
    return torch.where(torch.isfinite(world_points).all(dim=-1, keepdim=True), depth, torch.nan)


def _cache_complete(path: Path, *, clip: ClipConfig | None = None) -> bool:
    if not path.exists():
        return False
    try:
        payload = load_feature_cache(path, require_complete=False)
    except (OSError, RuntimeError, TypeError, ValueError):
        return False
    if not bool(payload.get("complete", False)):
        return False
    if clip is None:
        return True
    return (
        str(payload.get("clip_name")) == clip.name
        and str(payload.get("scene_id")) == clip.scene_id
        and tuple(int(value) for value in payload.get("frame_indices", ()))
        == clip.frame_indices
        and tuple(int(value) for value in payload.get("instance_ids", ()))
        == clip.instance_ids
        and int(payload.get("reference_sequence_index", -1))
        == clip.reference_sequence_index
    )


def _empty_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
