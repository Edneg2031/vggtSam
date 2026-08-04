"""Cache frozen SAM3/StreamVGGT observations for lightweight adapter training."""

from __future__ import annotations

import gc
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import torch

from ..backbones.sam3_intermediate import (
    SAM3IntermediateAdapter,
    load_sam3_image_model,
)
from ..backbones.streamvggt_latent import (
    StreamVGGTLatentAdapter,
    load_streamvggt_latent_model,
)
from ..backbones.streamvggt_parallel import (
    LayerShardedStreamVGGT,
    assert_processed_key_cache_equivalence,
)

from ..backbones.sam3_wrapper import SAM3Wrapper
from ..backbones.streamvggt_wrapper import StreamVGGTWrapper
from ..config import load_config
from ..data import load_rgb_sequence
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
from ..streamvggt_geometry_prompt import build_streamvggt_geometry_prompts
from ..tracking_recovery import run_natural_recovery_tracking
from ..types import TrackingSequence
from ..v6_geometry_segmentation import (
    V6_DEPLOYED_VARIANT,
    V6GeometrySegmentationConfig,
    segment_instance_with_geometry_prompts,
)
from .config import ClipConfig, LearnedPoseConfig
from .observations import (
    build_geometry_observations,
    build_pose_residual_observations,
    pool_sam_instance_features,
    sample_sam_instance_tokens,
    sample_instance_uvd,
)


CACHE_VERSION = 3


def build_feature_caches(config: LearnedPoseConfig) -> list[Path]:
    """Build geometry/tracking first, unload video SAM3, then pool SAM features."""

    config.features.cache_dir.mkdir(parents=True, exist_ok=True)
    paths = [cache_path(config, clip) for clip in config.clips]
    pending = [
        (clip, path)
        for clip, path in zip(config.clips, paths)
        if config.features.rebuild
        or not _cache_complete(
            path,
            config=config,
            clip=clip,
            require_identity=config.fusion.strict_identity_gate,
        )
    ]
    if not pending:
        print("learned-pose feature caches are complete")
        return paths

    _preflight_pending_clips(config, pending)
    recovery = load_config(config.recovery_config)
    geometry_pending = [
        (clip, path)
        for clip, path in pending
        if config.features.rebuild
        or not _geometry_cache_reusable(
            path,
            config=config,
            clip=clip,
            require_identity=config.fusion.strict_identity_gate,
        )
    ]
    sam_video_holder: dict[str, SAM3Wrapper] = {}
    if geometry_pending:
        if config.streamvggt_devices and not recovery.streaming_cache:
            raise ValueError(
                "Layer-sharded StreamVGGT requires "
                "streamvggt.streaming_cache=true in "
                f"{config.recovery_config}."
            )
        stream_model = load_streamvggt_latent_model(
            repo_path=recovery.streamvggt_repo,
            checkpoint_path=recovery.streamvggt_checkpoint,
            device=(
                "cpu"
                if config.streamvggt_devices
                else config.geometry_device
            ),
            strict=True,
        )
        parallel_runner = None
        if config.streamvggt_devices:
            assert_processed_key_cache_equivalence()
            print("StreamVGGT processed-key cache equivalence passed")
            parallel_runner = LayerShardedStreamVGGT(
                stream_model,
                config.streamvggt_devices,
                selected_layer_indices=config.fusion.dpt_layer_indices,
                amp_dtype=config.streamvggt_amp_dtype,
            )
            print(
                "StreamVGGT full-history model parallelism "
                f"amp={config.streamvggt_amp_dtype}"
            )
            for line in parallel_runner.layout_summary():
                print(f"  {line}")
        stream_adapter = StreamVGGTLatentAdapter(
            stream_model,
            device=config.geometry_device,
            image_mode=recovery.image_mode,
            dpt_layer_indices=config.fusion.dpt_layer_indices,
            parallel_runner=parallel_runner,
        )
        for clip, path in geometry_pending:
            print(f"caching frozen geometry/tracking clip={clip.name}")
            partial = _build_geometry_cache(
                config,
                clip,
                stream_adapter=stream_adapter,
                sam_video_holder=sam_video_holder,
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(partial, path)
        del partial, stream_adapter, parallel_runner, stream_model
        gc.collect()
        _empty_cuda_cache()
    else:
        print(
            "reusing frozen StreamVGGT/tracking observations; "
            "only SAM appearance cache fields will be augmented"
        )

    # The image model and video predictor are both large. Never keep both on
    # the SAM device while extracting pooled appearance descriptors.
    sam_video_holder.clear()
    gc.collect()
    _empty_cuda_cache()
    sam_image_model = load_sam3_image_model(
        repo_path=recovery.sam3_repo,
        checkpoint_path=recovery.sam3_checkpoint,
        device=config.sam3_device,
        version=recovery.sam3_version,
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
        tokens = _extract_sam_tokens_batched(
            sam_adapter,
            payload["image_paths"],
            token_grid=config.features.sam_grid,
            batch_size=config.features.sam_batch_size,
        )
        height, width = config.features.sam_grid
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
        if config.features.cache_sam_local_tokens:
            local, local_uv, local_valid = sample_sam_instance_tokens(
                spatial,
                payload["tracking_masks_output"],
                max_tokens=config.features.sam_local_token_count,
            )
            storage_dtype = (
                torch.float16
                if config.features.sam_local_storage_dtype
                in {"float16", "fp16"}
                else torch.float32
            )
            payload["sam_local_features"] = local.to(storage_dtype)
            payload["sam_local_uv"] = local_uv.float()
            payload["sam_local_valid"] = local_valid.bool()
            payload["sam_local_source"] = config.features.sam_source
            payload["sam_local_sampling"] = (
                config.features.sam_local_sampling
            )
            payload["sam_local_token_count"] = int(
                config.features.sam_local_token_count
            )
            payload["sam_local_feature_dim"] = int(local.shape[-1])
            payload["sam_local_storage_dtype"] = str(storage_dtype).replace(
                "torch.", ""
            )
        payload["complete"] = True
        torch.save(payload, path)
    del sam_adapter, sam_image_model
    gc.collect()
    _empty_cuda_cache()
    return paths


def _extract_sam_tokens_batched(
    adapter: SAM3IntermediateAdapter,
    image_paths: list[str],
    *,
    token_grid: tuple[int, int],
    batch_size: int,
) -> torch.Tensor:
    """Extract SAM appearance tokens without retaining all images on GPU."""

    if not image_paths:
        raise ValueError("SAM appearance extraction requires image paths.")
    effective_batch = (
        len(image_paths)
        if int(batch_size) <= 0
        else min(int(batch_size), len(image_paths))
    )
    height, width = token_grid
    chunks = []
    for start in range(0, len(image_paths), effective_batch):
        paths = image_paths[start : start + effective_batch]
        output = adapter.extract_from_paths(paths, prompt="object")
        tokens = output.semantic.tokens[0].detach().float().cpu()
        expected = len(paths) * height * width
        if tokens.shape[0] != expected:
            raise RuntimeError(
                "SAM3 token count does not match appearance batch: "
                f"{tokens.shape[0]} vs {len(paths)}*{height}*{width}."
            )
        chunks.append(tokens)
        del output, tokens
        _empty_cuda_cache()
    return torch.cat(chunks, dim=0)


def _preflight_pending_clips(
    config: LearnedPoseConfig,
    pending: list[tuple[ClipConfig, Path]],
) -> None:
    """Reject bad frame ranges and missing RGB files before model loading."""

    for clip, _ in pending:
        sequence = load_rgb_sequence(
            config.manifest,
            scene_id=clip.scene_id,
            frame_indices=clip.frame_indices,
        )
        if len(sequence.image_paths) != len(clip.frame_indices):
            raise RuntimeError(
                f"Clip preflight returned {len(sequence.image_paths)} images "
                f"for {len(clip.frame_indices)} requested frames: {clip.name}."
            )
    print(
        "cache preflight passed: "
        + ", ".join(
            f"{clip.name}={len(clip.frame_indices)}"
            for clip, _ in pending
        )
    )


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
    if clip.instance_source in {"sam3_reference", "sam31_online"}:
        sequences: dict[int, object] = {}
        target_masks: dict[int, torch.Tensor] = {}
        shared = load_rgb_sequence(
            config.manifest,
            scene_id=clip.scene_id,
            frame_indices=clip.frame_indices,
        )
    else:
        sequences, target_masks = load_instance_sequences(
            recovery,
            instance_ids=clip.instance_ids,
            reference_sequence_index=clip.reference_sequence_index,
            allow_missing_reference=(
                clip.allow_missing_reference_instances
            ),
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
    from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri

    baseline_w2c, baseline_intrinsics = pose_encoding_to_extri_intri(
        output.geometry.camera_tokens.detach().float(),
        image_size_hw=geometry_sequence.processed_size,
    )
    segmentation_diagnostics: list[dict[str, object]] = []
    if clip.instance_source == "sam31_online":
        recovered, segmentation_diagnostics = (
            _load_or_run_sam31_online_tracking(
                recovery,
                clip,
                shared=shared,
                sam_video_holder=sam_video_holder,
            )
        )
    elif config.features.sam_segmentation_variant != "legacy_recovery":
        if clip.instance_source != "configured_gt_reference":
            raise ValueError(
                "V6 SAM3.1 geometry prompting currently requires "
                "instance_source=configured_gt_reference."
            )
        recovered, segmentation_diagnostics = (
            _run_v6_sam31_geometry_prompt_tracking(
                recovery,
                config,
                clip,
                sequences=sequences,
                target_masks=target_masks,
                geometry=geometry_sequence,
                world_to_camera=baseline_w2c[0].detach().float().cpu(),
                intrinsics=baseline_intrinsics[0].detach().float().cpu(),
                sam_video_holder=sam_video_holder,
            )
        )
    elif clip.instance_source == "sam3_reference":
        recovered, target_masks = _load_or_run_sam3_reference_tracking(
            recovery,
            clip,
            shared=shared,
            geometry=geometry_sequence,
            sam_video_holder=sam_video_holder,
        )
    else:
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
    if clip.instance_source != "sam31_online":
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
    if clip.instance_source != "sam31_online":
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
        hard_mismatch_min_points=(
            config.fusion.identity_hard_mismatch_min_points
        ),
        hard_mismatch_max_fitness=(
            config.fusion.identity_hard_mismatch_max_fitness
        ),
        causal_instance_memory=(clip.instance_source == "sam31_online"),
    )
    depth = output.geometry.aux["depth_dense"].detach().float().cpu()
    depth_confidence = output.geometry.aux["depth_confidence_dense"].detach().float().cpu()
    pose_observations = build_pose_residual_observations(
        world_points=geometry_sequence.world_points,
        confidence=geometry_sequence.confidence,
        masks=grid_masks,
        world_to_camera=baseline_w2c[0],
        intrinsics=baseline_intrinsics[0],
        identity_valid=observations["identity_valid"],
        quality=observations["quality"],
        geometry=observations["geometry"],
        confidence_threshold=config.features.point_confidence_threshold,
        min_instance_points=config.features.min_instance_points,
        max_map_points=refinement.map_max_points,
        scene_scale=float(observations["scene_scale"]),
        min_geometry_confidence=config.fusion.min_geometry_confidence,
        min_static_score=config.fusion.min_static_score,
        causal_instance_memory=(clip.instance_source == "sam31_online"),
    )
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
    if config.fusion.strict_identity_gate:
        trusted_for_rigid = (
            trusted_for_rigid & observations["identity_valid"].bool()
        )
    rigid_weight = rigid_weight * trusted_for_rigid.float()
    trusted_instance_valid = (
        observations["observed"].bool()
        & observations["identity_valid"].bool()
        & (
            observations["quality"][..., 0]
            >= config.fusion.min_track_confidence
        )
    )
    associated_instance_valid = (
        observations["observed"].bool()
        & ~observations["identity_mismatch"].bool()
        & (
            observations["quality"][..., 0]
            >= config.fusion.min_track_confidence
        )
    )
    trusted_masks_output = (
        tracking_masks_output
        & trusted_instance_valid[..., None, None]
    )
    trusted_masks_stream = (
        grid_masks
        & trusted_instance_valid[..., None, None]
    )
    associated_masks_output = (
        tracking_masks_output
        & associated_instance_valid[..., None, None]
    )
    associated_masks_stream = (
        grid_masks
        & associated_instance_valid[..., None, None]
    )
    identity_diagnostics = []
    instance_slot = {
        int(instance_id): slot
        for slot, instance_id in enumerate(clip.instance_ids)
    }
    for row in observations["identity_diagnostics"]:
        current = dict(row)
        slot = instance_slot[int(current["instance_id"])]
        frame = int(current["sequence_index"])
        current["trusted_instance"] = int(
            trusted_instance_valid[frame, slot]
        )
        current["associated_instance"] = int(
            associated_instance_valid[frame, slot]
        )
        current["used_by_strict_method"] = int(
            associated_instance_valid[frame, slot]
            if config.fusion.strict_identity_gate
            else observations["observed"][frame, slot]
        )
        identity_diagnostics.append(current)
    sam_track_ids = [
        (
            -1
            if recovered[int(instance_id)].selected_obj_id is None
            else int(recovered[int(instance_id)].selected_obj_id)
        )
        for instance_id in clip.instance_ids
    ]
    sam_birth_indices = []
    for instance_id in clip.instance_ids:
        visible = recovered[int(instance_id)].masks.flatten(1).any(dim=1)
        positions = torch.nonzero(visible, as_tuple=False).flatten()
        sam_birth_indices.append(int(positions[0]) if positions.numel() else -1)
    geometry_birth_indices = [
        int(value) for value in observations["instance_birth_indices"]
    ]
    dynamic_instance_diagnostics = []
    if clip.instance_source == "sam31_online":
        for frame, frame_index in enumerate(clip.frame_indices):
            born_slots = [
                slot
                for slot, birth in enumerate(sam_birth_indices)
                if birth == frame
            ]
            geometry_born_slots = [
                slot
                for slot, birth in enumerate(geometry_birth_indices)
                if birth == frame
            ]
            discovered = sum(
                birth >= 0 and birth <= frame for birth in sam_birth_indices
            )
            observed_count = int(observations["observed"][frame].sum())
            mature = sum(
                birth >= 0
                and birth < frame
                and bool(observations["observed"][frame, slot])
                for slot, birth in enumerate(geometry_birth_indices)
            )
            dynamic_instance_diagnostics.append(
                {
                    "clip": clip.name,
                    "sequence_index": frame,
                    "frame_index": int(frame_index),
                    "discovered_tracks": int(discovered),
                    "observed_tracks": observed_count,
                    "mature_tracks": int(mature),
                    "identity_valid_tracks": int(
                        observations["identity_valid"][frame].sum()
                    ),
                    "associated_tracks": int(
                        associated_instance_valid[frame].sum()
                    ),
                    "birth_slots": " ".join(str(slot) for slot in born_slots),
                    "birth_sam_track_ids": " ".join(
                        str(sam_track_ids[slot]) for slot in born_slots
                    ),
                    "geometry_birth_slots": " ".join(
                        str(slot) for slot in geometry_born_slots
                    ),
                    "geometry_birth_sam_track_ids": " ".join(
                        str(sam_track_ids[slot])
                        for slot in geometry_born_slots
                    ),
                }
            )
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
        "instance_source": clip.instance_source,
        "instance_prompt": clip.instance_prompt,
        "sam_track_ids": sam_track_ids,
        "sam_birth_indices": sam_birth_indices,
        "instance_birth_indices": geometry_birth_indices,
        "dynamic_instance_diagnostics": dynamic_instance_diagnostics,
        "allow_missing_reference_instances": bool(
            clip.allow_missing_reference_instances
        ),
        "reference_sequence_index": clip.reference_sequence_index,
        "strict_identity_gate": bool(config.fusion.strict_identity_gate),
        "sam_version": str(recovery.sam3_version),
        "sam_checkpoint": str(recovery.sam3_checkpoint),
        "sam_appearance_source": config.features.sam_source,
        "sam_appearance_batch_size": config.features.sam_batch_size,
        "sam_segmentation_variant": (
            config.features.sam_segmentation_variant
        ),
        "streamvggt_execution": (
            "layer_sharded_full_history"
            if config.streamvggt_devices
            else "single_device_full_history"
        ),
        "streamvggt_devices": list(config.streamvggt_devices),
        "streamvggt_amp_dtype": config.streamvggt_amp_dtype,
        "segmentation_diagnostics": segmentation_diagnostics,
        "image_paths": [str(path) for path in shared.image_paths],
        "image_size": list(geometry_sequence.processed_size),
        "patch_start_idx": int(output.geometry.aux["patch_start_idx"]),
        "patch_shape": list(output.geometry.aux["patch_shape"]),
        # Keep the frozen-head inputs in fp32.  The module-off control is
        # required to reproduce the actual StreamVGGT outputs, not an fp16
        # cache approximation of them.
        "camera_hidden": camera_hidden.float(),
        "baseline_pose_encoding": output.geometry.camera_tokens.detach().float().cpu()[0],
        "baseline_depth": depth.float(),
        "baseline_depth_confidence": depth_confidence.float(),
        "baseline_world_points": geometry_sequence.world_points.float(),
        "baseline_world_confidence": geometry_sequence.confidence.float(),
        "geometry": observations["geometry"].float(),
        "pose_geometry": pose_observations["pose_geometry"].float(),
        "pose_geometry_valid": pose_observations[
            "pose_geometry_valid"
        ].bool(),
        "pose_geometry_feature_names": pose_observations[
            "pose_geometry_feature_names"
        ],
        "quality": observations["quality"].float(),
        "observed": observations["observed"].bool(),
        "identity_valid": observations["identity_valid"].bool(),
        "identity_unknown": observations["identity_unknown"].bool(),
        "identity_mismatch": observations["identity_mismatch"].bool(),
        "trusted_instance_valid": trusted_instance_valid.bool(),
        "associated_instance_valid": associated_instance_valid.bool(),
        "identity_diagnostics": identity_diagnostics,
        "geometry_feature_names": observations["geometry_feature_names"],
        "quality_names": observations["quality_names"],
        "geometry_dim": int(observations["geometry"].shape[-1]),
        "pose_geometry_dim": int(
            pose_observations["pose_geometry"].shape[-1]
        ),
        "scene_origin": observations["scene_origin"].float(),
        "scene_scale": float(observations["scene_scale"]),
        "tracking_masks_output": tracking_masks_output.bool(),
        "tracking_masks_stream": grid_masks.bool(),
        "trusted_tracking_masks_output": trusted_masks_output.bool(),
        "trusted_tracking_masks_stream": trusted_masks_stream.bool(),
        "associated_tracking_masks_output": associated_masks_output.bool(),
        "associated_tracking_masks_stream": associated_masks_stream.bool(),
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


def _run_v6_sam31_geometry_prompt_tracking(
    recovery,
    config: LearnedPoseConfig,
    clip: ClipConfig,
    *,
    sequences: Mapping[int, object],
    target_masks: Mapping[int, torch.Tensor],
    geometry,
    world_to_camera: torch.Tensor,
    intrinsics: torch.Tensor,
    sam_video_holder: dict[str, SAM3Wrapper],
) -> tuple[dict[int, TrackingSequence], list[dict[str, object]]]:
    """Build the deployed V6 masks from SAM3.1 plus simple geometry prompts."""

    variant = config.features.sam_segmentation_variant
    if recovery.sam3_version != "sam3.1":
        raise ValueError(
            f"{variant} requires sam3.version=sam3.1, "
            f"got {recovery.sam3_version!r}."
        )
    if variant != V6_DEPLOYED_VARIANT:
        raise ValueError(
            f"Unsupported V6 segmentation variant: {variant!r}."
        )

    sam3: SAM3Wrapper | None = None
    recovered: dict[int, TrackingSequence] = {}
    diagnostics: list[dict[str, object]] = []
    segmentation_config = V6GeometrySegmentationConfig()
    for instance_id in clip.instance_ids:
        instance_id = int(instance_id)
        reference = int(clip.reference_sequence_index)
        reference_mask = target_masks[instance_id][reference].bool()
        if not bool(reference_mask.any()):
            recovered[instance_id] = _empty_tracking(
                sequence=len(clip.frame_indices),
                output_size=recovery.output_size,
            )
            diagnostics.append(
                {
                    "clip": clip.name,
                    "instance_id": instance_id,
                    "sequence_index": reference,
                    "frame_index": int(clip.frame_indices[reference]),
                    "selected_variant": variant,
                    "status": "empty_reference_slot",
                }
            )
            continue

        if sam3 is None:
            sam3 = _sam_video_model(recovery, sam_video_holder)
        sequence = sequences[instance_id]
        raw = sam3.track(
            sequence.image_paths,
            prompt=sequence.label,
            output_size=recovery.output_size,
            reference_frame_idx=reference,
            reference_mask=reference_mask,
        )
        prompt_batch = build_streamvggt_geometry_prompts(
            recovery=recovery,
            sequence=sequence,
            reference_mask=reference_mask,
            world_points=geometry.world_points,
            confidence=geometry.confidence,
            world_to_camera=world_to_camera,
            intrinsics=intrinsics,
            source_sizes=geometry.source_sizes,
            processed_size=geometry.processed_size,
            point_confidence_threshold=(
                config.features.geometry_prompt_point_confidence_threshold
            ),
        )
        result = segment_instance_with_geometry_prompts(
            sequence=sequence,
            reference_mask=reference_mask,
            raw_tracking=raw,
            geometry_prompts=prompt_batch.prompts,
            output_size=recovery.output_size,
            sam3=sam3,
            config=segmentation_config,
        )
        recovered[instance_id] = TrackingSequence(
            masks=result["masks"][variant].bool(),
            scores=result["scores"][variant].float(),
            selected_obj_id=raw.selected_obj_id,
        )
        for segmentation_row, backend_row in zip(
            result["diagnostics"],
            prompt_batch.diagnostics,
        ):
            diagnostics.append(
                {
                    "clip": clip.name,
                    "instance_id": instance_id,
                    "selected_variant": variant,
                    **backend_row,
                    **segmentation_row,
                }
            )
    return recovered, diagnostics


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
    sam3 = None
    original: dict[int, TrackingSequence] = {}
    recovered: dict[int, TrackingSequence] = {}
    rows = []
    for instance_id in clip.instance_ids:
        reference_mask = target_masks[int(instance_id)][
            int(clip.reference_sequence_index)
        ]
        if not bool(reference_mask.any()):
            print(
                "reference instance unavailable; using empty slot "
                f"clip={clip.name} frame="
                f"{clip.frame_indices[clip.reference_sequence_index]} "
                f"instance={int(instance_id)}"
            )
            empty = _empty_tracking(
                sequence=len(clip.frame_indices),
                output_size=recovery.output_size,
            )
            original[int(instance_id)] = empty
            recovered[int(instance_id)] = empty
            rows.append(
                {
                    "instance_id": int(instance_id),
                    "recovery_applied": 0,
                    "recovery_sequence_index": None,
                    "recovery_frame_index": None,
                    "recovery_reason": "instance absent at reference frame",
                    "selected_support_coverage": 0.0,
                    "selected_candidate_gt_iou": None,
                }
            )
            continue
        if sam3 is None:
            sam3 = _sam_video_model(recovery, sam_video_holder)
        result = run_natural_recovery_tracking(
            recovery,
            sequence=sequences[int(instance_id)],
            target_masks=target_masks[int(instance_id)],
            geometry=geometry,
            sam3=sam3,
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


def _load_or_run_sam31_online_tracking(
    recovery,
    clip: ClipConfig,
    *,
    shared,
    sam_video_holder: dict[str, SAM3Wrapper],
) -> tuple[dict[int, TrackingSequence], list[dict[str, object]]]:
    """Discover persistent SAM3.1 IDs throughout a forward-only sequence."""

    if recovery.sam3_version != "sam3.1":
        raise ValueError("instance_source=sam31_online requires SAM3.1.")
    path = clip.tracking_cache or (
        recovery.output_dir / "tracking_cache_sam31_online.npz"
    )
    cached = load_tracking_cache(
        path,
        config=recovery,
        instance_ids=clip.instance_ids,
        frame_indices=clip.frame_indices,
    )
    if cached is not None and all(
        str(row.get("instance_prompt", "")) == clip.instance_prompt
        and str(row.get("propagation_direction", "")) == "forward"
        and int(row.get("causal_confirmation", 0)) == 1
        for row in cached[2]
    ):
        print(f"reusing SAM3.1 online tracking cache: {path}")
        return cached[1], [dict(row) for row in cached[2]]

    sam3 = _sam_video_model(recovery, sam_video_holder)
    tracked = sam3.track_all_forward(
        shared.image_paths,
        prompt=clip.instance_prompt,
        output_size=recovery.output_size,
        max_objects=max(
            int(recovery.sam3_max_num_objects),
            len(clip.instance_ids),
        ),
    )
    candidates = []
    for source_slot, obj_id in enumerate(tracked.obj_ids):
        masks = tracked.masks[:, source_slot]
        birth = int(tracked.birth_indices[source_slot])
        birth_pixels = int(masks[birth].sum())
        birth_ratio = float(masks[birth].float().mean())
        # Slot admission consults only the birth frame. Later observations
        # may diagnose the track but can never retroactively create its slot.
        if birth_pixels < int(recovery.min_pixels) or birth_ratio > 0.90:
            continue
        candidates.append(source_slot)
        if len(candidates) == len(clip.instance_ids):
            break

    originals: dict[int, TrackingSequence] = {}
    recovered: dict[int, TrackingSequence] = {}
    rows: list[dict[str, object]] = []
    for slot, instance_id in enumerate(clip.instance_ids):
        instance_id = int(instance_id)
        if slot >= len(candidates):
            sequence = _empty_tracking(
                sequence=len(clip.frame_indices),
                output_size=recovery.output_size,
            )
            birth = -1
            sam_obj_id = -1
        else:
            source_slot = candidates[slot]
            sam_obj_id = int(tracked.obj_ids[source_slot])
            birth = int(tracked.birth_indices[source_slot])
            sequence = TrackingSequence(
                masks=tracked.masks[:, source_slot].detach().cpu().bool(),
                scores=tracked.scores[:, source_slot].detach().cpu().float(),
                selected_obj_id=sam_obj_id,
            )
        originals[instance_id] = sequence
        recovered[instance_id] = sequence
        rows.append(
            {
                "instance_id": instance_id,
                "instance_source": "sam31_online",
                "instance_prompt": clip.instance_prompt,
                "sam_track_id": sam_obj_id,
                "birth_sequence_index": birth,
                "birth_frame_index": (
                    int(clip.frame_indices[birth]) if birth >= 0 else -1
                ),
                "observed_frames": int(
                    sequence.masks.flatten(1).any(dim=1).sum()
                ),
                "maximum_pixels": int(
                    sequence.masks.flatten(1).sum(dim=1).max()
                ),
                "propagation_direction": "forward",
                "causal_confirmation": 1,
            }
        )
    print(
        "SAM3.1 online instances "
        f"clip={clip.name} discovered={len(tracked.obj_ids)} "
        f"retained={len(candidates)} slots={len(clip.instance_ids)}"
    )
    save_tracking_cache(
        path,
        config=recovery,
        instance_ids=clip.instance_ids,
        frame_indices=clip.frame_indices,
        original=originals,
        recovered=recovered,
        tracking_rows=rows,
    )
    return recovered, rows


def _load_or_run_sam3_reference_tracking(
    recovery,
    clip: ClipConfig,
    *,
    shared,
    geometry,
    sam_video_holder: dict[str, SAM3Wrapper],
) -> tuple[dict[int, TrackingSequence], dict[int, torch.Tensor]]:
    """Fill exchangeable slots from a deployable SAM3 reference-frame query."""

    path = clip.tracking_cache or (
        recovery.output_dir / "tracking_cache_sam3_reference.npz"
    )
    cached = load_tracking_cache(
        path,
        config=recovery,
        instance_ids=clip.instance_ids,
        frame_indices=clip.frame_indices,
    )
    reference = int(clip.reference_sequence_index)
    if cached is not None:
        print(f"reusing SAM3 reference tracking cache: {path}")
        recovered = cached[1]
        targets = {
            int(instance_id): _reference_only_masks(
                recovered[int(instance_id)].masks[reference],
                sequence=len(clip.frame_indices),
                reference_index=reference,
            )
            for instance_id in clip.instance_ids
        }
        return recovered, targets

    sam3 = _sam_video_model(recovery, sam_video_holder)
    proposals = sam3.propose_text_masks(
        shared.image_paths[reference],
        prompt="object",
        output_size=recovery.output_size,
    )
    selected = _select_reference_candidates(
        proposals,
        max_instances=len(clip.instance_ids),
        min_pixels=recovery.min_pixels,
    )
    print(
        "SAM3 reference instances "
        f"clip={clip.name} detected={len(selected)} slots={len(clip.instance_ids)}"
    )

    originals: dict[int, TrackingSequence] = {}
    recovered: dict[int, TrackingSequence] = {}
    targets: dict[int, torch.Tensor] = {}
    rows: list[dict[str, object]] = []
    for slot, instance_id in enumerate(clip.instance_ids):
        instance_id = int(instance_id)
        if slot >= len(selected):
            empty = _empty_tracking(
                sequence=len(clip.frame_indices),
                output_size=recovery.output_size,
            )
            originals[instance_id] = empty
            recovered[instance_id] = empty
            targets[instance_id] = _reference_only_masks(
                empty.masks[reference],
                sequence=len(clip.frame_indices),
                reference_index=reference,
            )
            rows.append(
                {
                    "instance_id": instance_id,
                    "instance_source": "sam3_reference_empty_slot",
                    "reference_detection_score": 0.0,
                    "reference_pixels": 0,
                }
            )
            continue

        proposal = selected[slot]
        target = _reference_only_masks(
            proposal.mask,
            sequence=len(clip.frame_indices),
            reference_index=reference,
        )
        sequence = SimpleNamespace(
            scene_id=clip.scene_id,
            frame_indices=list(clip.frame_indices),
            image_paths=list(shared.image_paths),
            instance_id=instance_id,
            label="object",
            reference_frame_idx=reference,
        )
        result = run_natural_recovery_tracking(
            recovery,
            sequence=sequence,
            target_masks=target,
            geometry=geometry,
            sam3=sam3,
        )
        originals[instance_id] = result["original"]
        recovered[instance_id] = result["recovered"]
        targets[instance_id] = target
        rows.append(
            {
                "instance_id": instance_id,
                "instance_source": "sam3_reference",
                "reference_detection_score": float(proposal.score),
                "reference_pixels": int(proposal.mask.sum()),
                "recovery_applied": int(result["recovery_applied"]),
                "recovery_sequence_index": result["recovery_sequence_index"],
                "recovery_frame_index": result["recovery_frame_index"],
                "recovery_reason": result["recovery_reason"],
            }
        )
    save_tracking_cache(
        path,
        config=recovery,
        instance_ids=clip.instance_ids,
        frame_indices=clip.frame_indices,
        original=originals,
        recovered=recovered,
        tracking_rows=rows,
    )
    return recovered, targets


def _sam_video_model(
    recovery,
    holder: dict[str, SAM3Wrapper],
) -> SAM3Wrapper:
    if "model" not in holder:
        holder["model"] = SAM3Wrapper(
            repo_path=recovery.sam3_repo,
            checkpoint_path=recovery.sam3_checkpoint,
            device=recovery.sam3_device,
            output_threshold=recovery.sam3_output_threshold,
            prompt_with_box=recovery.prompt_with_box,
            version=recovery.sam3_version,
            use_fa3=recovery.sam3_use_fa3,
            max_num_objects=recovery.sam3_max_num_objects,
            multiplex_count=recovery.sam3_multiplex_count,
        ).load()
    return holder["model"]


def _select_reference_candidates(
    candidates,
    *,
    max_instances: int,
    min_pixels: int,
) -> list:
    """Select deterministic, non-duplicate SAM3 masks without consulting GT."""

    ranked = sorted(
        (
            candidate
            for candidate in candidates
            if int(candidate.mask.sum()) >= int(min_pixels)
            and float(candidate.mask.float().mean()) <= 0.90
        ),
        key=lambda candidate: (
            -float(candidate.score),
            -int(candidate.mask.sum()),
            int(candidate.obj_id),
        ),
    )
    selected = []
    for candidate in ranked:
        if any(_binary_iou(candidate.mask, item.mask) > 0.85 for item in selected):
            continue
        selected.append(candidate)
        if len(selected) == int(max_instances):
            break
    return selected


def _binary_iou(first: torch.Tensor, second: torch.Tensor) -> float:
    first = first.detach().cpu().bool()
    second = second.detach().cpu().bool()
    union = int((first | second).sum())
    return float((first & second).sum()) / union if union else 0.0


def _empty_tracking(
    *,
    sequence: int,
    output_size,
) -> TrackingSequence:
    return TrackingSequence(
        masks=torch.zeros(
            int(sequence),
            int(output_size[0]),
            int(output_size[1]),
            dtype=torch.bool,
        ),
        scores=torch.zeros(int(sequence), dtype=torch.float32),
        selected_obj_id=None,
    )


def _reference_only_masks(
    mask: torch.Tensor,
    *,
    sequence: int,
    reference_index: int,
) -> torch.Tensor:
    output = torch.zeros(
        int(sequence),
        int(mask.shape[0]),
        int(mask.shape[1]),
        dtype=torch.bool,
    )
    output[int(reference_index)] = mask.detach().cpu().bool()
    return output


def _processed_intrinsics(
    intrinsics: torch.Tensor,
    source_sizes,
    *,
    image_mode: str,
) -> torch.Tensor:
    from ..coordinates import streamvggt_image_transform

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


def _cache_complete(
    path: Path,
    *,
    config: LearnedPoseConfig | None = None,
    clip: ClipConfig | None = None,
    require_identity: bool = False,
) -> bool:
    if not path.exists():
        return False
    try:
        payload = load_feature_cache(path, require_complete=False)
    except (OSError, RuntimeError, TypeError, ValueError):
        return False
    if not bool(payload.get("complete", False)):
        return False
    if config is not None and config.features.cache_sam_local_tokens:
        required_local = (
            "sam_local_features",
            "sam_local_uv",
            "sam_local_valid",
            "sam_local_feature_dim",
        )
        if any(name not in payload for name in required_local):
            return False
        if (
            int(payload.get("sam_local_token_count", -1))
            != config.features.sam_local_token_count
            or str(payload.get("sam_local_source", ""))
            != config.features.sam_source
            or str(payload.get("sam_local_sampling", ""))
            != config.features.sam_local_sampling
            or str(payload.get("sam_local_storage_dtype", ""))
            not in {
                "float16"
                if config.features.sam_local_storage_dtype
                in {"float16", "fp16"}
                else "float32"
            }
        ):
            return False
    if (
        config is not None
        and config.features.sam_segmentation_variant != "legacy_recovery"
    ):
        recovery = load_config(config.recovery_config)
        if (
            str(payload.get("sam_version", "")) != recovery.sam3_version
            or str(payload.get("sam_checkpoint", ""))
            != str(recovery.sam3_checkpoint)
            or str(payload.get("sam_appearance_source", ""))
            != config.features.sam_source
            or str(payload.get("sam_segmentation_variant", ""))
            != config.features.sam_segmentation_variant
        ):
            return False
    if config is not None and config.streamvggt_devices:
        if (
            str(payload.get("streamvggt_execution", ""))
            != "layer_sharded_full_history"
            or tuple(
                str(value)
                for value in payload.get("streamvggt_devices", ())
            )
            != config.streamvggt_devices
            or str(payload.get("streamvggt_amp_dtype", ""))
            != config.streamvggt_amp_dtype
        ):
            return False
    if require_identity and any(
        name not in payload
        for name in (
            "identity_valid",
            "identity_unknown",
            "identity_mismatch",
            "trusted_instance_valid",
            "associated_instance_valid",
            "trusted_tracking_masks_output",
            "trusted_tracking_masks_stream",
            "associated_tracking_masks_output",
            "associated_tracking_masks_stream",
            "identity_diagnostics",
            "patch_shape",
            "pose_geometry",
            "pose_geometry_dim",
            "baseline_depth_confidence",
            "baseline_world_confidence",
        )
    ):
        return False
    if clip is None:
        return True
    if clip.instance_source == "sam31_online" and any(
        name not in payload
        for name in (
            "sam_track_ids",
            "sam_birth_indices",
            "instance_birth_indices",
            "dynamic_instance_diagnostics",
        )
    ):
        return False
    return (
        str(payload.get("clip_name")) == clip.name
        and str(payload.get("scene_id")) == clip.scene_id
        and tuple(int(value) for value in payload.get("frame_indices", ()))
        == clip.frame_indices
        and tuple(int(value) for value in payload.get("instance_ids", ()))
        == clip.instance_ids
        and str(
            payload.get("instance_source", "configured_gt_reference")
        )
        == clip.instance_source
        and str(payload.get("instance_prompt", "object"))
        == clip.instance_prompt
        and bool(payload.get("allow_missing_reference_instances", False))
        == bool(clip.allow_missing_reference_instances)
        and int(payload.get("reference_sequence_index", -1))
        == clip.reference_sequence_index
    )


def _geometry_cache_reusable(
    path: Path,
    *,
    config: LearnedPoseConfig,
    clip: ClipConfig,
    require_identity: bool,
) -> bool:
    """Check whether only the SAM appearance extension needs refreshing.

    In particular, changing V7.2 local-token count or storage dtype must not
    rerun StreamVGGT, geometry-guided SAM tracking, or GT alignment.  This
    validator intentionally ignores global/local appearance cache fields.
    """

    if not path.is_file():
        return False
    try:
        payload = load_feature_cache(path, require_complete=False)
    except (OSError, RuntimeError, TypeError, ValueError):
        return False
    core_fields = {
        "image_paths",
        "image_size",
        "camera_hidden",
        "baseline_pose_encoding",
        "baseline_world_points",
        "baseline_world_confidence",
        "pose_geometry",
        "quality",
        "observed",
        "tracking_masks_output",
        "target_pose_encoding",
    }
    if require_identity:
        core_fields.update(
            {
                "identity_valid",
                "identity_unknown",
                "identity_mismatch",
                "associated_instance_valid",
            }
        )
    if clip.instance_source == "sam31_online":
        core_fields.update(
            {
                "sam_track_ids",
                "sam_birth_indices",
                "instance_birth_indices",
                "dynamic_instance_diagnostics",
            }
        )
    if any(name not in payload for name in core_fields):
        return False
    recovery = load_config(config.recovery_config)
    if (
        str(payload.get("sam_version", "")) != recovery.sam3_version
        or str(payload.get("sam_checkpoint", ""))
        != str(recovery.sam3_checkpoint)
        or str(payload.get("sam_segmentation_variant", ""))
        != config.features.sam_segmentation_variant
        or str(payload.get("clip_name", "")) != clip.name
        or str(payload.get("scene_id", "")) != clip.scene_id
        or tuple(int(value) for value in payload.get("frame_indices", ()))
        != clip.frame_indices
        or tuple(int(value) for value in payload.get("instance_ids", ()))
        != clip.instance_ids
        or str(payload.get("instance_source", "configured_gt_reference"))
        != clip.instance_source
        or str(payload.get("instance_prompt", "object"))
        != clip.instance_prompt
        or int(payload.get("reference_sequence_index", -1))
        != clip.reference_sequence_index
    ):
        return False
    expected_execution = (
        "layer_sharded_full_history"
        if config.streamvggt_devices
        else "single_device_full_history"
    )
    if str(payload.get("streamvggt_execution", "")) != expected_execution:
        return False
    if config.streamvggt_devices and (
        tuple(str(value) for value in payload.get("streamvggt_devices", ()))
        != config.streamvggt_devices
        or str(payload.get("streamvggt_amp_dtype", ""))
        != config.streamvggt_amp_dtype
    ):
        return False
    return True


def _empty_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
