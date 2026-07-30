"""Evaluate fixed V6 pose checkpoints on an image-only long sequence.

SAM3 text prompts initialize two exchangeable persistent instance slots.  No
ground-truth pose, mask, or instance ID is read.  The script exports raw/V6
trajectories plus per-object point clouds for raw StreamVGGT, three V6 variants,
HorizonStream, and StreamVGGT's direct world point head.
"""

from __future__ import annotations

import argparse
import csv
import gc
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from streaming_couping.src.backbones.sam3_intermediate import (
    SAM3IntermediateAdapter,
    load_sam3_image_model,
)
from streaming_couping.src.backbones.sam3_wrapper import SAM3Wrapper
from streaming_couping.src.instance_observations import InstanceRefinementConfig
from streaming_couping.src.learned_pose.observations import (
    build_geometry_observations,
    build_pose_residual_observations,
    pool_sam_instance_features,
)
from streaming_couping.src.learned_pose.v6_camera_fusion import (
    V6CameraFusion,
    V6FusionConfig,
)

from .common import (
    camera_centers_from_w2c,
    fit_similarity_transform,
    natural_sort_key,
    save_trajectory_overlay_plot,
    save_trajectory_plot,
    transform_w2c_world_similarity,
    write_json,
    write_w2c_txt,
)
from .pointcloud_products import (
    PointCloudProductAccumulator,
    PointCloudProtocol,
    camera_to_world,
    read_intrinsics_txt,
    read_w2c_txt,
    unproject_depth,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-root",
        default="outputs/long_sequence_baselines/frames_300",
    )
    parser.add_argument("--scene-name", default="meeting_room_a02")
    parser.add_argument(
        "--output-root",
        default="outputs/long_sequence_baselines/frames_300/v6_instance_pose",
    )
    parser.add_argument("--sam3-repo", default="externals/sam3")
    parser.add_argument("--sam3-checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompts", nargs="+", default=("chair", "monitor"))
    parser.add_argument(
        "--reference-frame",
        type=int,
        default=-1,
        help="Sequence index; -1 scans forward without consulting GT.",
    )
    parser.add_argument("--reference-search-stride", type=int, default=10)
    parser.add_argument("--sam-batch-size", type=int, default=4)
    parser.add_argument("--sam-output-threshold", type=float, default=0.5)
    parser.add_argument("--min-reference-pixels", type=int, default=128)
    parser.add_argument("--max-reference-area-ratio", type=float, default=0.50)
    parser.add_argument("--point-confidence-threshold", type=float, default=0.30)
    parser.add_argument(
        "--v6-fusion-checkpoint",
        default=(
            "outputs/streaming_couping_v6_camera_overfit/"
            "v6_checkpoint_fusion.pt"
        ),
    )
    parser.add_argument(
        "--v6-rotation-checkpoint",
        default=(
            "outputs/streaming_couping_v6_camera_overfit/"
            "v6_checkpoint_specialized_camera_rotation.pt"
        ),
    )
    parser.add_argument(
        "--v6-center-checkpoint",
        default=(
            "outputs/streaming_couping_v6_camera_overfit/"
            "v6_checkpoint_specialized_instance_center_local.pt"
        ),
    )
    parser.add_argument(
        "--v6-aux-center-checkpoint",
        default=(
            "outputs/streaming_couping_v6_camera_overfit/"
            "v6_checkpoint_instance_se3_aux_0p1.pt"
        ),
    )
    parser.add_argument("--object-max-points", type=int, default=500_000)
    parser.add_argument("--confidence-percentile", type=float, default=50.0)
    parser.add_argument("--depth-percentile-low", type=float, default=1.0)
    parser.add_argument("--depth-percentile-high", type=float, default=99.0)
    parser.add_argument("--voxel-size-ratio", type=float, default=0.01)
    parser.add_argument("--min-voxel-observations", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_args(args)
    baseline_root = Path(args.baseline_root).expanduser().resolve()
    stream_dir = baseline_root / "streamvggt" / args.scene_name
    horizon_dir = baseline_root / "horizonstream" / args.scene_name
    output_dir = Path(args.output_root).expanduser().resolve() / args.scene_name
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = _read_image_list(stream_dir / "input_images.txt")
    horizon_images = _read_image_list(horizon_dir.parent / "input_images.txt")
    if image_paths != horizon_images:
        raise ValueError("HorizonStream and StreamVGGT input image lists differ.")
    stream_ids, stream_w2c = read_w2c_txt(stream_dir / "poses" / "abs_pose.txt")
    intri_ids, stream_intrinsics = read_intrinsics_txt(
        stream_dir / "poses" / "intri.txt"
    )
    horizon_ids, horizon_w2c = read_w2c_txt(
        horizon_dir / "poses" / "abs_pose.txt"
    )
    horizon_intri_ids, horizon_intrinsics = read_intrinsics_txt(
        horizon_dir / "poses" / "intri.txt"
    )
    frame_count = len(image_paths)
    if not all(
        len(values) == frame_count
        for values in (
            stream_ids,
            intri_ids,
            horizon_ids,
            horizon_intri_ids,
            stream_w2c,
            stream_intrinsics,
            horizon_w2c,
            horizon_intrinsics,
        )
    ):
        raise ValueError("Image, pose, and intrinsic frame counts disagree.")
    if not (
        stream_ids == intri_ids == horizon_ids == horizon_intri_ids
    ):
        raise ValueError("StreamVGGT/HorizonStream pose and intrinsic frame IDs differ.")

    features_dir = stream_dir / "features"
    camera_hidden = _load_tensor(features_dir / "camera_hidden.pt").float()
    if camera_hidden.shape[0] != frame_count or camera_hidden.ndim != 2:
        raise ValueError(
            f"Expected camera hidden [S,D] for {frame_count} frames, got "
            f"{tuple(camera_hidden.shape)}."
        )
    world_files = _sorted_npy(features_dir / "world_points")
    confidence_files = _sorted_npy(features_dir / "world_confidence")
    if len(world_files) != frame_count or len(confidence_files) != frame_count:
        raise ValueError(
            "Dense StreamVGGT deployment features are incomplete. Re-run the "
            "five-GPU baseline with --save-deployment-features."
        )
    point_shape = _spatial_shape(np.load(world_files[0], mmap_mode="r"))

    sam_payload = _load_or_build_sam_observations(
        args,
        image_paths=image_paths,
        output_size=point_shape,
        output_dir=output_dir,
    )
    reference_global = int(sam_payload["reference_frame"])
    frame_ids = stream_ids[reference_global:]
    segment_slice = slice(reference_global, None)
    masks = sam_payload["masks"].bool()
    scores = sam_payload["scores"].float()
    appearance = sam_payload["appearance"].float()
    expected_sequence = frame_count - reference_global
    if masks.shape[:2] != (expected_sequence, len(args.prompts)):
        raise ValueError("Cached SAM masks disagree with the selected segment.")

    print("loading dense StreamVGGT point-head observations", flush=True)
    world_points = torch.from_numpy(
        np.stack(
            [np.load(path) for path in world_files[reference_global:]],
            axis=0,
        )
    ).float()
    world_confidence = torch.from_numpy(
        np.stack(
            [_as_spatial(np.load(path)) for path in confidence_files[reference_global:]],
            axis=0,
        )
    ).float()
    if tuple(world_points.shape[1:3]) != tuple(masks.shape[-2:]):
        raise ValueError("SAM masks and StreamVGGT dense pointmaps have different grids.")

    refinement = InstanceRefinementConfig(
        min_instance_points=128,
        compute_device=args.device,
    )
    instance_ids = tuple(range(len(args.prompts)))
    numeric_frame_ids = tuple(range(reference_global, frame_count))
    geometry = build_geometry_observations(
        world_points=world_points,
        confidence=world_confidence,
        masks=masks,
        scores=scores,
        instance_ids=instance_ids,
        frame_indices=numeric_frame_ids,
        reference_index=0,
        confidence_threshold=args.point_confidence_threshold,
        refinement=refinement,
        sampled_instance_points=128,
        hard_mismatch_min_points=512,
        hard_mismatch_max_fitness=0.02,
    )
    pose_geometry = build_pose_residual_observations(
        world_points=world_points,
        confidence=world_confidence,
        masks=masks,
        world_to_camera=torch.from_numpy(stream_w2c[segment_slice]).float(),
        intrinsics=torch.from_numpy(stream_intrinsics[segment_slice]).float(),
        identity_valid=geometry["identity_valid"],
        quality=geometry["quality"],
        geometry=geometry["geometry"],
        confidence_threshold=args.point_confidence_threshold,
        min_instance_points=128,
        max_map_points=refinement.map_max_points,
        scene_scale=float(geometry["scene_scale"]),
        min_geometry_confidence=0.20,
        min_static_score=0.20,
    )["pose_geometry"]
    del world_points, world_confidence
    gc.collect()

    identity_rows = []
    prompt_by_slot = {slot: prompt for slot, prompt in enumerate(args.prompts)}
    for row in geometry["identity_diagnostics"]:
        current = dict(row)
        current["prompt"] = prompt_by_slot[int(current["instance_id"])]
        identity_rows.append(current)
    _write_csv(output_dir / "identity_diagnostics.csv", identity_rows)

    batch = {
        "camera_hidden": camera_hidden[segment_slice][None].to(args.device),
        "appearance": appearance[None].to(args.device),
        "pose_geometry": pose_geometry[None].to(args.device),
        "quality": geometry["quality"][None].to(args.device),
        "observed": geometry["observed"][None].to(args.device),
        "identity_valid": geometry["identity_valid"][None].to(args.device),
        "identity_unknown": geometry["identity_unknown"][None].to(args.device),
    }
    baseline_w2c = torch.from_numpy(stream_w2c[segment_slice]).float()[None].to(
        args.device
    )
    fusion_model, fusion_mode = _load_v6_model(
        args.v6_fusion_checkpoint,
        batch=batch,
        device=args.device,
    )
    rotation_model, rotation_mode = _load_v6_model(
        args.v6_rotation_checkpoint,
        batch=batch,
        device=args.device,
    )
    center_model, center_mode = _load_v6_model(
        args.v6_center_checkpoint,
        batch=batch,
        device=args.device,
    )
    aux_center_model, aux_center_mode = _load_v6_model(
        args.v6_aux_center_checkpoint,
        batch=batch,
        device=args.device,
    )
    # The upstream SAM3 video predictor enters a long-lived BF16 autocast
    # context during construction.  V6 checkpoints were trained/evaluated in
    # FP32, so establish an explicit precision boundary for reproducible pose
    # corrections even on a fresh run that just executed SAM3.
    with torch.inference_mode(), torch.autocast(
        device_type=torch.device(args.device).type,
        enabled=False,
    ):
        fusion_output = _run_v6(
            fusion_model,
            fusion_mode,
            batch,
            baseline_w2c,
        )
        rotation_output = _run_v6(
            rotation_model,
            rotation_mode,
            batch,
            baseline_w2c,
        )
        center_output = _run_v6(
            center_model,
            center_mode,
            batch,
            baseline_w2c,
        )
        aux_center_output = _run_v6(
            aux_center_model,
            aux_center_mode,
            batch,
            baseline_w2c,
        )
        candidate_a_output = _compose_decoupled(
            rotation_output,
            center_output,
            baseline_w2c,
        )
        candidate_b_output = _compose_decoupled(
            rotation_output,
            aux_center_output,
            baseline_w2c,
        )

    raw_np = baseline_w2c[0].cpu().numpy()
    fusion_np = fusion_output["world_to_camera"][0].cpu().numpy()
    candidate_a_np = candidate_a_output["world_to_camera"][0].cpu().numpy()
    candidate_b_np = candidate_b_output["world_to_camera"][0].cpu().numpy()
    horizon_np = horizon_w2c[segment_slice]
    pose_dir = output_dir / "poses"
    write_w2c_txt(pose_dir / "streamvggt_raw.txt", raw_np, frame_ids)
    write_w2c_txt(pose_dir / "v6_fusion.txt", fusion_np, frame_ids)
    write_w2c_txt(pose_dir / "v6_candidate_a_local3dof.txt", candidate_a_np, frame_ids)
    write_w2c_txt(pose_dir / "v6_candidate_b_aux0p1.txt", candidate_b_np, frame_ids)
    write_w2c_txt(pose_dir / "horizonstream.txt", horizon_np, frame_ids)
    plot_dir = output_dir / "plots"
    save_trajectory_plot(
        plot_dir / "streamvggt_raw.png",
        raw_np,
        "StreamVGGT raw (no GT/alignment)",
    )
    save_trajectory_plot(
        plot_dir / "v6_fusion.png",
        fusion_np,
        "V6 fusion (no GT/alignment)",
    )
    save_trajectory_plot(
        plot_dir / "v6_candidate_a_local3dof.png",
        candidate_a_np,
        "V6 candidate A: camera-R + local-3DoF-C",
    )
    save_trajectory_plot(
        plot_dir / "v6_candidate_b_aux0p1.png",
        candidate_b_np,
        "V6 candidate B: camera-R + aux-0.1-SE3-C",
    )
    save_trajectory_plot(
        plot_dir / "horizonstream.png",
        horizon_np,
        "HorizonStream (independent gauge; no GT/alignment)",
    )
    save_trajectory_overlay_plot(
        plot_dir / "streamvggt_raw_vs_v6.png",
        {
            "StreamVGGT raw": raw_np,
            "V6 fusion": fusion_np,
            "V6 candidate A": candidate_a_np,
            "V6 candidate B": candidate_b_np,
        },
        "Same-gauge StreamVGGT / V6 trajectories",
    )

    raw_centers = camera_centers_from_w2c(raw_np)
    horizon_centers = camera_centers_from_w2c(horizon_np)
    shared_scale, shared_rotation, shared_translation = fit_similarity_transform(
        raw_centers,
        horizon_centers,
    )
    horizon_aligned = {
        name: transform_w2c_world_similarity(
            poses,
            scale=shared_scale,
            rotation=shared_rotation,
            translation=shared_translation,
        )
        for name, poses in {
            "streamvggt_raw": raw_np,
            "v6_fusion": fusion_np,
            "v6_candidate_a": candidate_a_np,
            "v6_candidate_b": candidate_b_np,
        }.items()
    }
    save_trajectory_overlay_plot(
        plot_dir / "streamvggt_v6_vs_horizon_shared_sim3.png",
        {
            "HorizonStream pseudo-reference": horizon_np,
            "StreamVGGT raw (shared Sim3)": horizon_aligned["streamvggt_raw"],
            "V6 fusion (same Sim3)": horizon_aligned["v6_fusion"],
            "V6 candidate A (same Sim3)": horizon_aligned["v6_candidate_a"],
            "V6 candidate B (same Sim3)": horizon_aligned["v6_candidate_b"],
        },
        "Shared raw-fitted Sim(3); HorizonStream is not GT",
    )

    stability_rows = [
        _trajectory_proxy_row(
            "streamvggt_raw",
            raw_np,
            raw_np,
            active_frames=0,
        ),
        _trajectory_proxy_row(
            "v6_fusion",
            fusion_np,
            raw_np,
            active_frames=int(fusion_output["active_frames"].sum().cpu()),
        ),
        _trajectory_proxy_row(
            "v6_candidate_a",
            candidate_a_np,
            raw_np,
            active_frames=int(candidate_a_output["active_frames"].sum().cpu()),
        ),
        _trajectory_proxy_row(
            "v6_candidate_b",
            candidate_b_np,
            raw_np,
            active_frames=int(candidate_b_output["active_frames"].sum().cpu()),
        ),
        _trajectory_proxy_row(
            "horizonstream",
            horizon_np,
            None,
            active_frames=0,
        ),
    ]
    _write_csv(output_dir / "trajectory_stability_proxies.csv", stability_rows)
    horizon_rows = [
        _horizon_disagreement_row(
            variant,
            aligned,
            horizon_np,
            shared_scale=shared_scale,
        )
        for variant, aligned in horizon_aligned.items()
    ]
    raw_horizon = horizon_rows[0]
    for row in horizon_rows:
        row["center_closer_than_raw"] = int(
            float(row["center_rmse_horizon_native"])
            < float(raw_horizon["center_rmse_horizon_native"])
        )
        row["rotation_closer_than_raw"] = int(
            float(row["rotation_mean_degrees"])
            < float(raw_horizon["rotation_mean_degrees"])
        )
    _write_csv(output_dir / "trajectory_horizon_proxy.csv", horizon_rows)
    _write_csv(
        output_dir / "comparison_summary.csv",
        _comparison_summary_rows(stability_rows, horizon_rows),
    )

    del fusion_model, rotation_model, center_model, aux_center_model
    del batch, baseline_w2c
    del fusion_output, rotation_output, center_output, aux_center_output
    del candidate_a_output, candidate_b_output
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    object_rows = _export_object_pointclouds(
        args,
        output_dir=output_dir,
        stream_dir=stream_dir,
        horizon_dir=horizon_dir,
        reference_global=reference_global,
        masks=masks,
        prompts=args.prompts,
        stream_intrinsics=stream_intrinsics[segment_slice],
        horizon_intrinsics=horizon_intrinsics[segment_slice],
        trajectories={
            "streamvggt_raw": raw_np,
            "v6_fusion": fusion_np,
            "v6_candidate_a": candidate_a_np,
            "v6_candidate_b": candidate_b_np,
            "horizonstream": horizon_np,
        },
        stream_world_files=world_files[reference_global:],
        stream_world_confidence_files=confidence_files[reference_global:],
    )
    _write_csv(output_dir / "object_pointcloud_summary.csv", object_rows)

    summary = {
        "scene": args.scene_name,
        "input_frames": frame_count,
        "reference_sequence_index": reference_global,
        "evaluated_frames": expected_sequence,
        "prompts": list(args.prompts),
        "uses_ground_truth": False,
        "v6_variants": {
            "fusion": str(Path(args.v6_fusion_checkpoint).resolve()),
            "candidate_a_and_b_rotation": str(
                Path(args.v6_rotation_checkpoint).resolve()
            ),
            "candidate_a_local_3dof_center": str(
                Path(args.v6_center_checkpoint).resolve()
            ),
            "candidate_b_aux_center": str(
                Path(args.v6_aux_center_checkpoint).resolve()
            ),
        },
        "horizon_proxy_alignment": {
            "fit_source": "streamvggt_raw_all_camera_centers",
            "scale_raw_to_horizon": float(shared_scale),
            "rotation_raw_to_horizon": shared_rotation.tolist(),
            "translation_raw_to_horizon": shared_translation.tolist(),
            "same_transform_applied_to_all_v6_variants": True,
            "horizonstream_is_ground_truth": False,
        },
        "interpretation": (
            "No GT is available. Stability, shared-Sim3 Horizon disagreement, and "
            "object point-cloud consistency are diagnostic proxies, not pose-accuracy "
            "metrics."
        ),
        "outputs": {
            "reference_mask_overlay": str(
                output_dir / "masks" / "reference_overlay.png"
            ),
            "tracking_mask_overview": str(
                output_dir / "masks" / "tracking_overview.png"
            ),
            "trajectory_overlay": str(
                plot_dir / "streamvggt_raw_vs_v6.png"
            ),
            "horizon_shared_sim3_overlay": str(
                plot_dir / "streamvggt_v6_vs_horizon_shared_sim3.png"
            ),
            "trajectory_proxies": str(
                output_dir / "trajectory_stability_proxies.csv"
            ),
            "comparison_summary": str(output_dir / "comparison_summary.csv"),
            "object_summary": str(
                output_dir / "object_pointcloud_summary.csv"
            ),
        },
    }
    write_json(output_dir / "run_summary.json", summary)
    print(f"V6 meeting-room evaluation complete: {output_dir}", flush=True)
    print("Copy these two short tables:", flush=True)
    print(output_dir / "comparison_summary.csv", flush=True)
    print(output_dir / "object_pointcloud_summary.csv", flush=True)


def _validate_args(args: argparse.Namespace) -> None:
    if len(args.prompts) != 2 or len(set(args.prompts)) != 2:
        raise ValueError("This fixed comparison expects exactly two distinct prompts.")
    if args.reference_search_stride <= 0 or args.sam_batch_size <= 0:
        raise ValueError("Reference stride and SAM batch size must be positive.")
    if not 0.0 < args.max_reference_area_ratio <= 1.0:
        raise ValueError("--max-reference-area-ratio must be in (0,1].")


def _load_or_build_sam_observations(
    args: argparse.Namespace,
    *,
    image_paths: Sequence[Path],
    output_size: tuple[int, int],
    output_dir: Path,
) -> dict[str, Any]:
    cache_path = output_dir / "cache" / "sam3_observations.pt"
    cache_key = {
        "sam3_checkpoint": str(Path(args.sam3_checkpoint).expanduser().resolve()),
        "prompts": list(args.prompts),
        "output_size": list(output_size),
        "reference_frame_request": int(args.reference_frame),
        "reference_search_stride": int(args.reference_search_stride),
        "sam_output_threshold": float(args.sam_output_threshold),
        "min_reference_pixels": int(args.min_reference_pixels),
        "max_reference_area_ratio": float(args.max_reference_area_ratio),
    }
    if cache_path.is_file():
        payload = _load_tensor(cache_path)
        if (
            isinstance(payload, dict)
            and payload.get("image_paths") == [str(path) for path in image_paths]
            and payload.get("cache_key") == cache_key
        ):
            print(f"reusing SAM3 observations: {cache_path}", flush=True)
            _ensure_mask_outputs(
                output_dir,
                prompts=args.prompts,
                masks=payload["masks"],
                reference_frame=int(payload["reference_frame"]),
                reference_rgb=_reference_rgb_path(
                    args,
                    int(payload["reference_frame"]),
                ),
            )
            return payload

    print("loading SAM3 video predictor", flush=True)
    sam3 = SAM3Wrapper(
        repo_path=args.sam3_repo,
        checkpoint_path=args.sam3_checkpoint,
        device=args.device,
        output_threshold=args.sam_output_threshold,
        prompt_with_box=True,
    ).load()
    reference, candidates = _find_reference_frame(
        sam3,
        image_paths=image_paths,
        prompts=args.prompts,
        output_size=output_size,
        requested=args.reference_frame,
        stride=args.reference_search_stride,
        min_pixels=args.min_reference_pixels,
        max_area_ratio=args.max_reference_area_ratio,
    )
    segment_paths = image_paths[reference:]
    masks = []
    scores = []
    for prompt, candidate in zip(args.prompts, candidates):
        print(
            f"SAM3 tracking prompt={prompt!r} reference={reference} "
            f"pixels={int(candidate.mask.sum())} score={candidate.score:.4f}",
            flush=True,
        )
        tracking = sam3.track(
            segment_paths,
            prompt=prompt,
            output_size=output_size,
            reference_frame_idx=0,
            reference_mask=candidate.mask,
        )
        masks.append(tracking.masks)
        scores.append(tracking.scores)
    masks_tensor = torch.stack(masks, dim=1).bool()
    scores_tensor = torch.stack(scores, dim=1).float()
    del sam3
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _save_masks_and_reference_overlay(
        output_dir,
        prompts=args.prompts,
        masks=masks_tensor,
        reference_frame=reference,
        reference_rgb=_reference_rgb_path(args, reference),
    )
    appearance = _extract_sam_appearance(
        args,
        image_paths=segment_paths,
        masks=masks_tensor,
    )
    payload = {
        "image_paths": [str(path) for path in image_paths],
        "cache_key": cache_key,
        "prompts": list(args.prompts),
        "output_size": list(output_size),
        "reference_frame": reference,
        "masks": masks_tensor.cpu(),
        "scores": scores_tensor.cpu(),
        "appearance": appearance.cpu(),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    return payload


def _find_reference_frame(
    sam3: SAM3Wrapper,
    *,
    image_paths: Sequence[Path],
    prompts: Sequence[str],
    output_size: tuple[int, int],
    requested: int,
    stride: int,
    min_pixels: int,
    max_area_ratio: float,
) -> tuple[int, list[Any]]:
    if requested >= len(image_paths):
        raise ValueError("--reference-frame is outside the input sequence.")
    latest_reference = len(image_paths) - 3
    if latest_reference < 0:
        raise ValueError("At least three frames are required for this comparison.")
    if requested >= 0:
        if requested > latest_reference:
            raise ValueError(
                "--reference-frame must leave at least three frames for trajectory "
                "comparison."
            )
        indices = [requested]
    else:
        indices = list(range(0, latest_reference + 1, stride))
        if indices[-1] != latest_reference:
            indices.append(latest_reference)
    total_pixels = int(output_size[0] * output_size[1])
    for index in indices:
        selected = []
        for prompt in prompts:
            proposals = sam3.propose_text_masks(
                image_paths[index],
                prompt=prompt,
                output_size=output_size,
            )
            candidate = _best_reference_candidate(
                proposals,
                min_pixels=min_pixels,
                max_pixels=int(max_area_ratio * total_pixels),
            )
            if candidate is None:
                break
            selected.append(candidate)
        if len(selected) == len(prompts) and _mask_iou(
            selected[0].mask,
            selected[1].mask,
        ) > 0.50:
            print(
                f"reference scan frame={index}: prompts selected the same region",
                flush=True,
            )
            continue
        print(
            f"reference scan frame={index}: detected={len(selected)}/{len(prompts)}",
            flush=True,
        )
        if len(selected) == len(prompts):
            return index, selected
    raise RuntimeError(
        "No scanned frame contains valid chair and monitor reference masks. "
        "Inspect the images and retry with --reference-frame INDEX or adjust "
        "the English prompts."
    )


def _best_reference_candidate(
    proposals: Sequence[Any],
    *,
    min_pixels: int,
    max_pixels: int,
) -> Any | None:
    valid = [
        item
        for item in proposals
        if min_pixels <= int(item.mask.sum()) <= max_pixels
    ]
    if not valid:
        return None
    return max(
        valid,
        key=lambda item: (float(item.score), int(item.mask.sum()), -int(item.obj_id)),
    )


def _mask_iou(first: torch.Tensor, second: torch.Tensor) -> float:
    first = first.detach().cpu().bool()
    second = second.detach().cpu().bool()
    union = int((first | second).sum())
    return float((first & second).sum()) / union if union else 0.0


def _extract_sam_appearance(
    args: argparse.Namespace,
    *,
    image_paths: Sequence[Path],
    masks: torch.Tensor,
) -> torch.Tensor:
    print("extracting frozen SAM3 FPN appearance in batches", flush=True)
    model = load_sam3_image_model(
        repo_path=args.sam3_repo,
        checkpoint_path=args.sam3_checkpoint,
        device=args.device,
        enable_segmentation=False,
        enable_inst_interactivity=False,
    )
    adapter = SAM3IntermediateAdapter(
        model,
        device=args.device,
        resolution=1008,
        source="detector_fpn2",
        text_conditioning="none",
        token_grid=(72, 72),
    )
    rows = []
    for start in range(0, len(image_paths), args.sam_batch_size):
        end = min(start + args.sam_batch_size, len(image_paths))
        output = adapter.extract_from_paths(image_paths[start:end], prompt="object")
        tokens = output.semantic.tokens[0]
        spatial = tokens.reshape(end - start, 72, 72, -1).permute(0, 3, 1, 2)
        rows.append(
            pool_sam_instance_features(spatial, masks[start:end]).float()
        )
        del output, tokens, spatial
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"SAM3 appearance {end}/{len(image_paths)}", flush=True)
    del adapter, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return torch.cat(rows, dim=0)


def _save_masks_and_reference_overlay(
    output_dir: Path,
    *,
    prompts: Sequence[str],
    masks: torch.Tensor,
    reference_frame: int,
    reference_rgb: Path,
) -> None:
    for slot, prompt in enumerate(prompts):
        mask_dir = output_dir / "masks" / _safe_name(prompt)
        mask_dir.mkdir(parents=True, exist_ok=True)
        for stale in mask_dir.glob("frame_*.png"):
            stale.unlink()
        for frame, mask in enumerate(masks[:, slot]):
            Image.fromarray(mask.numpy().astype(np.uint8) * 255).save(
                mask_dir / f"frame_{reference_frame + frame:06d}.png"
            )
    rgb = np.asarray(Image.open(reference_rgb).convert("RGB"), dtype=np.uint8)
    if tuple(rgb.shape[:2]) != tuple(masks.shape[-2:]):
        rgb = np.asarray(
            Image.fromarray(rgb).resize(
                (int(masks.shape[-1]), int(masks.shape[-2])),
                Image.Resampling.BILINEAR,
            )
        )
    colors = np.asarray(((255, 64, 64), (64, 255, 64)), dtype=np.float32)
    overlay = rgb.astype(np.float32)
    for slot in range(len(prompts)):
        mask = masks[0, slot].numpy()
        overlay[mask] = 0.45 * overlay[mask] + 0.55 * colors[slot]
    overlay_image = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(overlay_image)
    legend_y = 8
    for slot, prompt in enumerate(prompts):
        color = tuple(int(value) for value in colors[slot])
        draw.rectangle((8, legend_y, 22, legend_y + 14), fill=color)
        draw.text((28, legend_y), str(prompt), fill=(255, 255, 255))
        legend_y += 20
    draw.text(
        (8, legend_y + 2),
        f"sequence frame {reference_frame}",
        fill=(255, 255, 255),
    )
    overlay_image.save(output_dir / "masks" / "reference_overlay.png")

    sample_count = min(12, int(masks.shape[0]))
    sample_indices = np.linspace(
        0,
        int(masks.shape[0]) - 1,
        num=sample_count,
        dtype=np.int64,
    )
    tiles = []
    for local_index in sample_indices:
        global_index = reference_frame + int(local_index)
        frame_rgb = np.asarray(
            Image.open(
                reference_rgb.parent / f"frame_{global_index:06d}.png"
            ).convert("RGB"),
            dtype=np.uint8,
        )
        if tuple(frame_rgb.shape[:2]) != tuple(masks.shape[-2:]):
            frame_rgb = np.asarray(
                Image.fromarray(frame_rgb).resize(
                    (int(masks.shape[-1]), int(masks.shape[-2])),
                    Image.Resampling.BILINEAR,
                )
            )
        tile = frame_rgb.astype(np.float32)
        for slot in range(len(prompts)):
            mask = masks[int(local_index), slot].numpy()
            tile[mask] = 0.45 * tile[mask] + 0.55 * colors[slot]
        tile_image = Image.fromarray(np.clip(tile, 0, 255).astype(np.uint8))
        tile_draw = ImageDraw.Draw(tile_image)
        tile_draw.rectangle((4, 4, 106, 23), fill=(0, 0, 0))
        tile_draw.text((8, 6), f"frame {global_index}", fill=(255, 255, 255))
        tiles.append(tile_image)
    columns = min(4, len(tiles))
    rows = (len(tiles) + columns - 1) // columns
    tile_width, tile_height = tiles[0].size
    overview = Image.new(
        "RGB",
        (columns * tile_width, rows * tile_height),
        color=(0, 0, 0),
    )
    for index, tile in enumerate(tiles):
        overview.paste(
            tile,
            ((index % columns) * tile_width, (index // columns) * tile_height),
        )
    overview.save(output_dir / "masks" / "tracking_overview.png")


def _ensure_mask_outputs(
    output_dir: Path,
    *,
    prompts: Sequence[str],
    masks: torch.Tensor,
    reference_frame: int,
    reference_rgb: Path,
) -> None:
    final_frame = reference_frame + int(masks.shape[0]) - 1
    expected = [
        output_dir / "masks" / "reference_overlay.png",
        output_dir / "masks" / "tracking_overview.png",
    ]
    for prompt in prompts:
        directory = output_dir / "masks" / _safe_name(prompt)
        expected.extend(
            (
                directory / f"frame_{reference_frame:06d}.png",
                directory / f"frame_{final_frame:06d}.png",
            )
        )
    if all(path.is_file() for path in expected):
        return
    _save_masks_and_reference_overlay(
        output_dir,
        prompts=prompts,
        masks=masks.detach().cpu().bool(),
        reference_frame=reference_frame,
        reference_rgb=reference_rgb,
    )


def _reference_rgb_path(args: argparse.Namespace, reference_frame: int) -> Path:
    return (
        Path(args.baseline_root).expanduser().resolve()
        / "streamvggt"
        / args.scene_name
        / "images"
        / "rgb"
        / f"frame_{reference_frame:06d}.png"
    )


def _load_v6_model(
    checkpoint_path: str | Path,
    *,
    batch: dict[str, torch.Tensor],
    device: str,
) -> tuple[V6CameraFusion, str]:
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"V6 checkpoint not found: {path}. Run "
            "zsh streaming_couping/commands_v6_camera_overfit.txt first."
        )
    checkpoint = _load_tensor(path)
    config = V6FusionConfig(**checkpoint["fusion"])
    expected = {
        "camera_dim": int(batch["camera_hidden"].shape[-1]),
        "appearance_dim": int(batch["appearance"].shape[-1]),
        "geometry_dim": int(batch["pose_geometry"].shape[-1]),
    }
    for name, value in expected.items():
        if int(checkpoint[name]) != value:
            raise ValueError(
                f"V6 checkpoint {path.name} {name}={checkpoint[name]} but "
                f"meeting-room input has {value}."
            )
    model = V6CameraFusion(
        **expected,
        config=config,
        head_component=str(checkpoint["head_component"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, str(checkpoint["mode"])


def _run_v6(
    model: V6CameraFusion,
    mode: str,
    batch: dict[str, torch.Tensor],
    baseline_w2c: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return model(
        camera_hidden=batch["camera_hidden"],
        baseline_world_to_camera=baseline_w2c,
        appearance=batch["appearance"],
        geometry=batch["pose_geometry"],
        quality=batch["quality"],
        observed=batch["observed"],
        identity_valid=batch["identity_valid"],
        identity_unknown=batch["identity_unknown"],
        reference_index=0,
        mode=mode,
    )


def _compose_decoupled(
    rotation_output: dict[str, torch.Tensor],
    center_output: dict[str, torch.Tensor],
    baseline_w2c: torch.Tensor,
) -> dict[str, torch.Tensor]:
    rotation = rotation_output["world_to_camera"][..., :3, :3]
    center_pose = center_output["world_to_camera"]
    center = _torch_camera_centers(center_pose)
    translation = -(rotation @ center[..., None])
    composed = torch.cat([rotation, translation], dim=-1)
    active = rotation_output["active_frames"] & center_output["active_frames"]
    composed = torch.where(active[..., None, None], composed, baseline_w2c).clone()
    composed[:, 0] = baseline_w2c[:, 0]
    return {"world_to_camera": composed, "active_frames": active}


def _torch_camera_centers(w2c: torch.Tensor) -> torch.Tensor:
    rotation = w2c[..., :3, :3]
    translation = w2c[..., :3, 3]
    return -(rotation.transpose(-1, -2) @ translation[..., None]).squeeze(-1)


def _trajectory_proxy_row(
    variant: str,
    world_to_camera: np.ndarray,
    raw_world_to_camera: np.ndarray | None,
    *,
    active_frames: int,
) -> dict[str, object]:
    rotations = world_to_camera[:, :3, :3]
    centers = camera_centers_from_w2c(world_to_camera)
    steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    acceleration = np.linalg.norm(np.diff(centers, n=2, axis=0), axis=1)
    relative_rotation = rotations[1:] @ rotations[:-1].transpose(0, 2, 1)
    rotation_steps = _rotation_angles_deg(relative_rotation)
    angular_acceleration = (
        _rotation_angles_deg(
            relative_rotation[1:]
            @ relative_rotation[:-1].transpose(0, 2, 1)
        )
        if len(relative_rotation) > 1
        else np.empty(0)
    )
    if raw_world_to_camera is None:
        correction_rotation = np.empty(0)
        correction_center = np.empty(0)
    else:
        correction_rotation = _rotation_angles_deg(
            rotations
            @ raw_world_to_camera[:, :3, :3].transpose(0, 2, 1)
        )
        correction_center = np.linalg.norm(
            centers - camera_centers_from_w2c(raw_world_to_camera),
            axis=1,
        )
    return {
        "variant": variant,
        "frames": len(world_to_camera),
        "active_frames": int(active_frames),
        "active_ratio": float(active_frames) / max(1, len(world_to_camera) - 1),
        "path_length_native": float(steps.sum()),
        "translation_step_mean": _mean(steps),
        "translation_acceleration_p95": _percentile(acceleration, 95),
        "rotation_step_mean_degrees": _mean(rotation_steps),
        "rotation_acceleration_p95_degrees": _percentile(
            angular_acceleration,
            95,
        ),
        "correction_rotation_mean_degrees": _mean(correction_rotation),
        "correction_rotation_max_degrees": _max(correction_rotation),
        "correction_center_mean_native": _mean(correction_center),
        "correction_center_max_native": _max(correction_center),
        "metric_role": "no_gt_stability_proxy",
    }


def _horizon_disagreement_row(
    variant: str,
    aligned_world_to_camera: np.ndarray,
    horizon_world_to_camera: np.ndarray,
    *,
    shared_scale: float,
) -> dict[str, object]:
    aligned_centers = camera_centers_from_w2c(aligned_world_to_camera)
    horizon_centers = camera_centers_from_w2c(horizon_world_to_camera)
    center_error = np.linalg.norm(aligned_centers - horizon_centers, axis=1)
    rotation_error = _rotation_angles_deg(
        aligned_world_to_camera[:, :3, :3]
        @ horizon_world_to_camera[:, :3, :3].transpose(0, 2, 1)
    )
    return {
        "variant": variant,
        "frames": len(aligned_world_to_camera),
        "center_rmse_horizon_native": float(
            np.sqrt(np.mean(np.square(center_error)))
        ),
        "center_mean_horizon_native": _mean(center_error),
        "center_p95_horizon_native": _percentile(center_error, 95),
        "rotation_mean_degrees": _mean(rotation_error),
        "rotation_p95_degrees": _percentile(rotation_error, 95),
        "shared_sim3_scale_raw_to_horizon": float(shared_scale),
        "alignment_fit_source": "streamvggt_raw_all_camera_centers",
        "metric_role": "horizonstream_pseudo_reference_not_ground_truth",
    }


def _comparison_summary_rows(
    stability_rows: Sequence[dict[str, object]],
    horizon_rows: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    horizon_by_variant = {str(row["variant"]): row for row in horizon_rows}
    rows = []
    for stability in stability_rows:
        variant = str(stability["variant"])
        horizon = horizon_by_variant.get(variant)
        rows.append(
            {
                "variant": variant,
                "frames": stability["frames"],
                "v6_active_ratio": stability["active_ratio"],
                "translation_acceleration_p95": stability[
                    "translation_acceleration_p95"
                ],
                "rotation_acceleration_p95_degrees": stability[
                    "rotation_acceleration_p95_degrees"
                ],
                "horizon_center_rmse_shared_raw_sim3": (
                    horizon["center_rmse_horizon_native"] if horizon else None
                ),
                "horizon_rotation_mean_degrees": (
                    horizon["rotation_mean_degrees"] if horizon else None
                ),
                "center_closer_to_horizon_than_raw": (
                    horizon["center_closer_than_raw"] if horizon else None
                ),
                "rotation_closer_to_horizon_than_raw": (
                    horizon["rotation_closer_than_raw"] if horizon else None
                ),
                "interpretation": (
                    "diagnostic_only_horizon_is_not_gt"
                    if horizon
                    else "horizonstream_pseudo_reference"
                ),
            }
        )
    return rows


def _export_object_pointclouds(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    stream_dir: Path,
    horizon_dir: Path,
    reference_global: int,
    masks: torch.Tensor,
    prompts: Sequence[str],
    stream_intrinsics: np.ndarray,
    horizon_intrinsics: np.ndarray,
    trajectories: dict[str, np.ndarray],
    stream_world_files: Sequence[Path],
    stream_world_confidence_files: Sequence[Path],
) -> list[dict[str, object]]:
    protocol = PointCloudProtocol(
        confidence_percentile=args.confidence_percentile,
        depth_percentile_low=args.depth_percentile_low,
        depth_percentile_high=args.depth_percentile_high,
        voxel_size_ratio=args.voxel_size_ratio,
        min_voxel_observations=args.min_voxel_observations,
        max_points=args.object_max_points,
    )
    protocol.validate()
    stream_depth_files = _sorted_npy(stream_dir / "depth" / "dpt")[reference_global:]
    stream_conf_files = _sorted_npy(stream_dir / "depth" / "conf")[reference_global:]
    horizon_depth_files = _sorted_npy(horizon_dir / "depth" / "dpt")[reference_global:]
    horizon_conf_files = _sorted_npy(horizon_dir / "depth" / "conf")[reference_global:]
    sequence = masks.shape[0]
    for values in (
        stream_depth_files,
        stream_conf_files,
        horizon_depth_files,
        horizon_conf_files,
        stream_world_files,
        stream_world_confidence_files,
    ):
        if len(values) != sequence:
            raise ValueError("Object point-cloud input frame counts disagree.")
    stream_scale = _positive_depth_median(np.load(stream_depth_files[0]))
    horizon_scale = _positive_depth_median(np.load(horizon_depth_files[0]))
    method_names = (
        "streamvggt_raw",
        "v6_fusion",
        "v6_candidate_a",
        "v6_candidate_b",
        "horizonstream",
        "streamvggt_pointhead",
    )
    accumulators: dict[tuple[str, str], PointCloudProductAccumulator] = {}
    for prompt in prompts:
        for method in method_names:
            scale = horizon_scale if method == "horizonstream" else stream_scale
            accumulators[(prompt, method)] = PointCloudProductAccumulator(
                total_frames=sequence,
                protocol=protocol,
                scale_reference=scale,
            )

    for local_index in range(sequence):
        global_index = reference_global + local_index
        stream_depth = _as_spatial(np.load(stream_depth_files[local_index]))
        stream_conf = _as_spatial(np.load(stream_conf_files[local_index]))
        stream_rgb = _load_rgb(
            stream_dir / "images" / "rgb" / f"frame_{global_index:06d}.png",
            stream_depth.shape,
        )
        stream_camera = unproject_depth(
            stream_depth,
            stream_intrinsics[local_index],
        )
        stream_world_by_method = {
            name: camera_to_world(
                stream_camera,
                trajectories[name][local_index],
            ).reshape(stream_depth.shape + (3,))
            for name in (
                "streamvggt_raw",
                "v6_fusion",
                "v6_candidate_a",
                "v6_candidate_b",
            )
        }
        pointhead = np.asarray(np.load(stream_world_files[local_index]), dtype=np.float32)
        pointhead_conf = _as_spatial(
            np.load(stream_world_confidence_files[local_index])
        )

        horizon_depth = _as_spatial(np.load(horizon_depth_files[local_index]))
        horizon_conf = _as_spatial(np.load(horizon_conf_files[local_index]))
        horizon_rgb = _load_rgb(
            horizon_dir / "images" / "rgb" / f"frame_{global_index:06d}.png",
            horizon_depth.shape,
        )
        horizon_camera = unproject_depth(
            horizon_depth,
            horizon_intrinsics[local_index],
        )
        horizon_world = camera_to_world(
            horizon_camera,
            trajectories["horizonstream"][local_index],
        ).reshape(horizon_depth.shape + (3,))
        stream_positive = np.isfinite(stream_depth) & (stream_depth > 0)
        horizon_positive = np.isfinite(horizon_depth) & (horizon_depth > 0)
        stream_depth_range = _depth_range_mask(
            stream_depth,
            stream_positive,
            args.depth_percentile_low,
            args.depth_percentile_high,
        )
        horizon_depth_range = _depth_range_mask(
            horizon_depth,
            horizon_positive,
            args.depth_percentile_low,
            args.depth_percentile_high,
        )
        for slot, prompt in enumerate(prompts):
            stream_mask = _resize_mask(masks[local_index, slot], stream_depth.shape)
            horizon_mask = _resize_mask(masks[local_index, slot], horizon_depth.shape)
            for method, points in stream_world_by_method.items():
                accumulators[(prompt, method)].add_frame(
                    points[stream_mask],
                    stream_rgb[stream_mask],
                    stream_conf[stream_mask],
                    raw_valid=stream_positive[stream_mask],
                    filtered_valid=stream_depth_range[stream_mask],
                )
            accumulators[(prompt, "streamvggt_pointhead")].add_frame(
                pointhead[stream_mask],
                stream_rgb[stream_mask],
                pointhead_conf[stream_mask],
                raw_valid=stream_positive[stream_mask],
                filtered_valid=stream_depth_range[stream_mask],
            )
            accumulators[(prompt, "horizonstream")].add_frame(
                horizon_world[horizon_mask],
                horizon_rgb[horizon_mask],
                horizon_conf[horizon_mask],
                raw_valid=horizon_positive[horizon_mask],
                filtered_valid=horizon_depth_range[horizon_mask],
            )
        if (
            local_index == 0
            or (local_index + 1) % 25 == 0
            or local_index + 1 == sequence
        ):
            print(f"object point clouds {local_index + 1}/{sequence}", flush=True)

    rows = []
    for prompt in prompts:
        object_dir = output_dir / "objects" / _safe_name(prompt)
        for method in method_names:
            result = accumulators[(prompt, method)].write(object_dir, method)
            rows.append(
                {
                    "object": prompt,
                    "method": method,
                    "raw_points_seen": result["raw_points_seen"],
                    "filtered_points_seen": result["filtered_points_seen"],
                    "raw_ply_points": result["raw_ply_points"],
                    "filtered_ply_points": result["filtered_ply_points"],
                    "fused_ply_points": result["fused_ply_points"],
                    "voxel_size_native": result["voxel_size"],
                    "fused_mean_observations": result["fused_observation_mean"],
                    "coordinate_gauge": (
                        "horizonstream_native"
                        if method == "horizonstream"
                        else "streamvggt_native"
                    ),
                }
            )
    return rows


def _rotation_angles_deg(rotation: np.ndarray) -> np.ndarray:
    if not len(rotation):
        return np.empty(0, dtype=np.float64)
    trace = np.trace(rotation, axis1=-2, axis2=-1)
    cosine = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def _read_image_list(path: Path) -> list[Path]:
    if not path.is_file():
        raise FileNotFoundError(path)
    values = [
        Path(line.strip()).expanduser().resolve()
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    if not values:
        raise ValueError(f"Empty image list: {path}")
    return values


def _sorted_npy(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    values = sorted(directory.glob("*.npy"), key=natural_sort_key)
    if not values:
        raise ValueError(f"No npy files in {directory}")
    return values


def _spatial_shape(value: np.ndarray) -> tuple[int, int]:
    array = np.asarray(value)
    if array.ndim < 2:
        raise ValueError(f"Expected a spatial array, got {array.shape}")
    return int(array.shape[0]), int(array.shape[1])


def _as_spatial(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    while array.ndim > 2 and array.shape[-1] == 1:
        array = array[..., 0]
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"Expected spatial map [H,W], got {array.shape}")
    return array


def _load_rgb(path: Path, shape: tuple[int, int]) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if (image.height, image.width) != tuple(shape):
        image = image.resize((shape[1], shape[0]), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def _resize_mask(mask: torch.Tensor, shape: tuple[int, int]) -> np.ndarray:
    value = mask.detach().cpu().bool()
    if tuple(value.shape) != tuple(shape):
        value = F.interpolate(
            value.float()[None, None],
            size=shape,
            mode="nearest",
        )[0, 0].bool()
    return value.numpy()


def _positive_depth_median(depth: np.ndarray) -> float:
    value = _as_spatial(depth)
    valid = value[np.isfinite(value) & (value > 0)]
    if not len(valid):
        raise ValueError("Reference frame has no positive finite depth.")
    return float(np.median(valid))


def _depth_range_mask(
    depth: np.ndarray,
    valid: np.ndarray,
    low_percentile: float,
    high_percentile: float,
) -> np.ndarray:
    result = np.zeros_like(valid, dtype=bool)
    if not np.any(valid):
        return result
    low, high = np.percentile(
        depth[valid],
        (float(low_percentile), float(high_percentile)),
    )
    return valid & (depth >= low) & (depth <= high)


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)


def _mean(values: np.ndarray) -> float | None:
    return float(np.mean(values)) if len(values) else None


def _max(values: np.ndarray) -> float | None:
    return float(np.max(values)) if len(values) else None


def _percentile(values: np.ndarray, percentile: float) -> float | None:
    return float(np.percentile(values, percentile)) if len(values) else None


def _load_tensor(path: str | Path) -> Any:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
