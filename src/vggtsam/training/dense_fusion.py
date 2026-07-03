"""Dense image-grid SAM3/StreamVGGT fusion training."""

from __future__ import annotations

import csv
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from vggtsam.adapters.sam3_intermediate import (
    SAM3IntermediateAdapter,
    ensure_bchw_tensor,
    load_sam3_image_model,
    pool_language_features,
)
from vggtsam.adapters.sam3_video import (
    SAM3VideoTrackerAdapter,
    load_sam3_video_predictor,
)
from vggtsam.adapters.streamvggt_latent import (
    StreamVGGTLatentAdapter,
    load_streamvggt_latent_model,
)
from vggtsam.data.scannetpp.object_sequence import (
    ObjectSamplingConfig,
    ObjectSequence,
    ScanNetPPObjectSequenceDataset,
    keep_instances_visible_in_multiple_frames,
)
from vggtsam.models.dense_fusion import DenseSAMVGGTModel
from vggtsam.training.latent_fusion import (
    ObjectPromptSelection,
    label_is_excluded,
    label_matches,
    normalize_label_filters,
    select_object_prompt,
    slice_camera_tokens,
    split_sequence_tokens,
)
from vggtsam.training.plotting import plot_training_curves


@dataclass
class DenseFusionTrainConfig:
    manifest: Path
    scene_id: str | None
    sequence_length: int
    frame_stride: int
    frame_indices: List[int] | None
    min_pixels: int
    max_area_ratio: float
    min_visible_frames: int
    max_objects_per_frame: int
    ignore_instance_id: int
    semantic_ignore_label: int
    excluded_semantic_labels: List[int]
    target_object_labels: List[str]
    excluded_object_labels: List[str]
    target_mode: str
    sam3_repo: Path
    sam3_checkpoint: Path
    sam3_prompt: str
    sam3_prompt_mode: str
    sam3_resolution: int
    sam3_feature_source: str
    sam3_text_conditioning: str
    sam3_enable_inst_interactivity: bool
    sam3_device: str
    sam3_frame_chunk_size: int
    sam3_direct_device: str
    sam3_direct_prompt_with_box: bool
    sam3_direct_output_prob_thresh: float
    sam3_direct_async_loading_frames: bool
    sam3_compare_direct: bool
    streamvggt_repo: Path
    streamvggt_checkpoint: Path
    geometry_device: str
    geometry_streaming_cache: bool
    feature_grid: tuple[int, int]
    context_grid: tuple[int, int]
    streamvggt_layer_index: int
    streamvggt_dpt_layer_indices: List[int]
    streamvggt_image_mode: str
    use_camera_tokens: bool
    output_size: tuple[int, int]
    d_fuse: int
    num_heads: int
    embedding_dim: int
    num_classes: int
    dropout: float
    point_decoder: str
    point_mask_condition: str
    fusion_type: str
    primary_mask_source: str
    stream_dpt_use_pretrained: bool
    stream_dpt_freeze: bool
    mask_weight: float
    dice_weight: float
    point_weight: float
    point_valid_source: str
    point_valid_threshold: float
    chamfer_weight: float
    reprojection_weight: float
    text_weight: float
    aux_cls_weight: float
    match_weight: float
    temperature: float
    max_match_pixels: int
    max_chamfer_points: int
    negative_ratio: int
    history_enabled: bool
    history_update_source: str
    history_pred_threshold: float
    fused_sam_prompt_source: str
    fused_sam_feature_mode: str
    fused_sam_mask_weight: float
    fused_sam_dice_weight: float
    device: str
    iterations: int
    lr: float
    train_scope: str
    seed: int
    log_every: int
    save_every: int
    visualize_every: int
    visualize_threshold: float
    overfit: bool
    overfit_window_index: int
    overfit_instance_id: int | None
    max_visual_points: int
    output_dir: Path


def select_training_prompt(
    sequence: ObjectSequence,
    *,
    config: DenseFusionTrainConfig,
    rng: random.Random,
) -> ObjectPromptSelection | None:
    if config.overfit_instance_id is not None and config.overfit_instance_id > 0:
        instance_id = int(config.overfit_instance_id)
        label = sequence.object_labels.get(instance_id)
        if not label:
            return None
        if not any(np.any(mask == instance_id) for mask in sequence.instance_masks):
            return None
        return ObjectPromptSelection(
            prompt=label,
            target_object_labels=[label],
            sampled_instance_id=instance_id,
            sampled_label=label,
        )
    return select_object_prompt(
        sequence.visible_instance_ids,
        sequence.object_labels,
        rng=rng,
        min_visible_frames=config.min_visible_frames,
        mode=config.sam3_prompt_mode,
        fallback_prompt=config.sam3_prompt,
        target_object_labels=config.target_object_labels,
        excluded_object_labels=config.excluded_object_labels,
    )


def select_overfit_sequence(
    dataset: ScanNetPPObjectSequenceDataset,
    *,
    config: DenseFusionTrainConfig,
    rng: random.Random,
) -> tuple[ObjectSequence, ObjectPromptSelection, int]:
    start = int(config.overfit_window_index)
    for offset in range(len(dataset)):
        window_index = (start + offset) % len(dataset)
        sequence = dataset[window_index]
        prompt_selection = select_training_prompt(
            sequence,
            config=config,
            rng=rng,
        )
        if prompt_selection is not None:
            return sequence, prompt_selection, window_index
    raise RuntimeError(
        "Could not select an overfit target from any window. Try lowering "
        "object filters or clearing excluded_object_labels."
    )


def extract_sam3_sequence(
    sam3: SAM3IntermediateAdapter,
    image_paths: Sequence[Path],
    *,
    prompt: str,
    chunk_size: int,
    sam_tracker_for_features: Any | None = None,
):
    chunk_size = int(chunk_size)
    if chunk_size <= 0 or chunk_size >= len(image_paths):
        out = sam3.extract_from_paths(image_paths, prompt=prompt)
        if sam_tracker_for_features is not None:
            out.sam_tracker_features = extract_sam3_tracker_decoder_features(
                out.backbone_out,
                sam_tracker_for_features,
            )
        return out

    token_chunks = []
    feature_chunks: Dict[str, List[torch.Tensor]] = {}
    text_out = None
    spatial_shape = None
    aux = None
    for start in range(0, len(image_paths), chunk_size):
        out = sam3.extract_from_paths(
            image_paths[start : start + chunk_size],
            prompt=prompt,
        )
        token_chunks.append(out.semantic.tokens.detach().cpu())
        if sam_tracker_for_features is not None:
            chunk_features = extract_sam3_tracker_decoder_features(
                out.backbone_out,
                sam_tracker_for_features,
            )
            for key, value in chunk_features.items():
                feature_chunks.setdefault(key, []).append(value)
        if text_out is None:
            text_out = detach_to_cpu(out.text_out)
            spatial_shape = out.semantic.spatial_shape
            aux = dict(out.semantic.aux)
        del out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return SimpleNamespace(
        semantic=SimpleNamespace(
            tokens=torch.cat(token_chunks, dim=1),
            spatial_shape=spatial_shape,
            aux=aux or {},
        ),
        text_out=text_out or {},
        sam_tracker_features={
            key: torch.cat(values, dim=0)
            for key, values in feature_chunks.items()
        } if feature_chunks else None,
    )


def detach_to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: detach_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [detach_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(detach_to_cpu(item) for item in value)
    return value


def uses_fused_sam_decoder(config: DenseFusionTrainConfig) -> bool:
    return (
        config.history_update_source.strip().lower() in {"fused_sam", "sam_decoder"}
        or config.primary_mask_source.strip().lower() in {"fused_sam", "sam_decoder"}
        or config.fused_sam_mask_weight > 0.0
        or config.fused_sam_dice_weight > 0.0
    )


def uses_fused_sam_residual_features(config: DenseFusionTrainConfig) -> bool:
    return uses_fused_sam_decoder(config) and (
        config.fused_sam_feature_mode.strip().lower() == "residual"
    )


def uses_sam3_direct_masks(config: DenseFusionTrainConfig) -> bool:
    direct_sources = {
        "sam3_direct",
        "sam3",
        "sam_video",
        "sam3_video",
    }
    return (
        config.history_update_source.strip().lower() in direct_sources
        or config.point_valid_source.strip().lower() in direct_sources
        or config.primary_mask_source.strip().lower() in direct_sources
        or config.sam3_compare_direct
    )


def configure_trainable_parameters(
    model: DenseSAMVGGTModel,
    *,
    config: DenseFusionTrainConfig,
    use_fused_sam: bool,
) -> List[torch.nn.Parameter]:
    """Select which parameters should be optimized for the requested ablation."""
    scope = config.train_scope.strip().lower()
    if scope == "all":
        params = [param for param in model.parameters() if param.requires_grad]
    elif scope == "sam_adapter":
        if not use_fused_sam:
            raise RuntimeError(
                "training.train_scope='sam_adapter' requires the fused SAM decoder. "
                "Set history.update_source='fused_sam' or give fused_sam losses "
                "positive weights."
            )
        primary_uses_fused_sam = config.primary_mask_source.strip().lower() in {
            "fused_sam",
            "sam_decoder",
        }
        has_primary_sam_loss = primary_uses_fused_sam and (
            config.mask_weight > 0.0 or config.dice_weight > 0.0
        )
        if (
            config.fused_sam_mask_weight <= 0.0
            and config.fused_sam_dice_weight <= 0.0
            and not has_primary_sam_loss
        ):
            raise RuntimeError(
                "training.train_scope='sam_adapter' needs a trainable fused SAM "
                "loss. Set --fused-sam-mask-weight/--fused-sam-dice-weight, "
                "or use --primary-mask-source fused_sam with positive "
                "--mask-weight/--dice-weight."
            )
        for param in model.parameters():
            param.requires_grad_(False)
        trainable_names = []
        trainable_prefixes = (
            "fused_sam_",
            "proj_sam",
            "proj_geometry",
            "proj_camera",
            "context_norm",
            "cross_attention",
            "fusion_norm",
            "camera_guided_fusion",
        )
        for name, param in model.named_parameters():
            if name.startswith(trainable_prefixes):
                param.requires_grad_(True)
                trainable_names.append(name)
        if not trainable_names:
            raise RuntimeError("No SAM/fusion adapter parameters were found to train.")
        params = [param for param in model.parameters() if param.requires_grad]
    else:
        raise ValueError(
            "training.train_scope must be 'all' or 'sam_adapter', "
            f"got {config.train_scope!r}"
        )

    num_params = sum(param.numel() for param in params)
    print(f"train_scope={scope} trainable_parameters={num_params}")
    return params


@torch.no_grad()
def extract_sam3_tracker_decoder_features(
    backbone_out: Dict[str, Any],
    sam_tracker,
) -> Dict[str, torch.Tensor]:
    """Build the original SAM3 tracker decoder features for residual injection."""
    sam2 = backbone_out.get("sam2_backbone_out")
    if sam2 is None:
        raise RuntimeError(
            "SAM3 backbone_out does not contain sam2_backbone_out. "
            "Build SAM3 with enable_inst_interactivity=True."
        )
    fpn = sam2.get("backbone_fpn")
    if fpn is None or len(fpn) < 3:
        raise RuntimeError("SAM3 sam2_backbone_out is missing 3 FPN levels.")
    decoder = sam_tracker.sam_mask_decoder
    decoder_param = next(decoder.parameters())
    decoder_device = decoder_param.device
    decoder_dtype = decoder_param.dtype
    high_s0_input = ensure_bchw_tensor(fpn[0]).to(
        device=decoder_device,
        dtype=decoder_dtype,
    )
    high_s1_input = ensure_bchw_tensor(fpn[1]).to(
        device=decoder_device,
        dtype=decoder_dtype,
    )
    image_embed = ensure_bchw_tensor(fpn[2]).to(
        device=decoder_device,
        dtype=decoder_dtype,
    )
    high_s0 = decoder.conv_s0(high_s0_input)
    high_s1 = decoder.conv_s1(high_s1_input)
    return {
        "image_embed": image_embed.detach().float().cpu(),
        "high_s1": high_s1.detach().float().cpu(),
        "high_s0": high_s0.detach().float().cpu(),
    }


def slice_sam3_tracker_features(
    features: Dict[str, torch.Tensor] | None,
    *,
    frame_idx: int,
    device: str,
) -> Dict[str, torch.Tensor] | None:
    if features is None:
        return None
    return {
        key: value[frame_idx : frame_idx + 1].float().to(device)
        for key, value in features.items()
    }


def slice_stream_dpt_tokens(
    tokens: Sequence[torch.Tensor] | None,
    *,
    frame_idx: int,
    device: str,
) -> List[torch.Tensor] | None:
    if tokens is None:
        return None
    return [
        value[:, frame_idx : frame_idx + 1].float().to(device)
        for value in tokens
    ]


def slice_stream_images(
    images: torch.Tensor | None,
    *,
    frame_idx: int,
    device: str,
) -> torch.Tensor | None:
    if images is None:
        return None
    if images.ndim == 4:
        return images[frame_idx : frame_idx + 1].unsqueeze(0).float().to(device)
    if images.ndim == 5:
        return images[:, frame_idx : frame_idx + 1].float().to(device)
    raise ValueError(f"Expected stream images [T, C, H, W], got {tuple(images.shape)}")


def batch_to_cpu(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in batch.items()}


def batch_to_device(
    batch: Dict[str, torch.Tensor],
    device: str,
) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def export_streamvggt_baseline(config: DenseFusionTrainConfig) -> None:
    """Run frozen StreamVGGT once and export its pointmap under the selected GT mask."""
    rng = random.Random(config.seed)
    torch.manual_seed(config.seed)
    object_config = ObjectSamplingConfig(
        min_pixels=config.min_pixels,
        max_area_ratio=config.max_area_ratio,
        min_visible_frames=config.min_visible_frames,
        max_objects_per_frame=config.max_objects_per_frame,
        ignore_instance_id=config.ignore_instance_id,
        semantic_ignore_label=config.semantic_ignore_label,
    )
    dataset = ScanNetPPObjectSequenceDataset(
        config.manifest,
        scene_id=config.scene_id,
        sequence_length=config.sequence_length,
        frame_stride=config.frame_stride,
        frame_indices=config.frame_indices,
        object_config=object_config,
    )

    if config.overfit:
        sequence, prompt_selection, resolved_window_index = select_overfit_sequence(
            dataset,
            config=config,
            rng=rng,
        )
        if config.overfit_window_index != resolved_window_index:
            print(
                "streamvggt baseline window adjusted "
                f"requested={config.overfit_window_index} "
                f"resolved={resolved_window_index}"
            )
    else:
        sequence = dataset.sample(rng)
        prompt_selection = select_training_prompt(sequence, config=config, rng=rng)
        if prompt_selection is None:
            raise RuntimeError("Could not select a prompt for StreamVGGT baseline export.")

    if sequence.pointmaps is None:
        raise RuntimeError(
            "StreamVGGT baseline export requires processed GT pointmaps. "
            "Re-run scripts/prepare_scannetpp_2d.py with --save-pointmaps."
        )

    batch = build_dense_batch(
        sequence,
        prompt_selection,
        config=config,
        device="cpu",
    )
    if batch is None:
        raise RuntimeError(
            "Selected StreamVGGT baseline target produced no prompt_mask. "
            "Try a different --window-index / --instance-id."
        )

    print(
        "streamvggt baseline target "
        f"scene={sequence.scene_id} "
        f"frames={sequence.frame_indices} "
        f"instance={prompt_selection.sampled_instance_id} "
        f"label='{prompt_selection.sampled_label}' "
        f"target_mode={config.target_mode}"
    )

    streamvggt_model = load_streamvggt_latent_model(
        repo_path=config.streamvggt_repo,
        checkpoint_path=config.streamvggt_checkpoint,
        device=config.geometry_device,
        strict=True,
    )
    streamvggt_model.requires_grad_(False)
    geometry = StreamVGGTLatentAdapter(
        streamvggt_model,
        device=config.geometry_device,
        token_grid=config.feature_grid,
        context_grid=config.context_grid,
        layer_index=config.streamvggt_layer_index,
        dpt_layer_indices=config.streamvggt_dpt_layer_indices,
        image_mode=config.streamvggt_image_mode,
    )
    geo_out = geometry.extract_from_paths(
        sequence.image_paths,
        return_pointmap=True,
        streaming_cache=config.geometry_streaming_cache,
    )
    stream_points = detach_to_cpu(geo_out.geometry.aux.get("pointmap_dense"))
    if stream_points is None:
        stream_points = detach_to_cpu(geo_out.pointmap_grid)
    if stream_points is None:
        raise RuntimeError("StreamVGGT did not return a pointmap for baseline export.")
    stream_points = resize_dense_bhwc_tensor(stream_points, config.output_size)

    output_dir = config.output_dir / "streamvggt_baseline"
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb = load_sequence_rgb(sequence.image_paths, batch["prompt_mask"].shape[1:])
    gt_mask = batch["prompt_mask"].detach().cpu()
    gt_points = batch["pointmap"].detach().cpu()
    gt_valid = batch["point_valid"].detach().cpu() & gt_mask
    stream_loss = dense_point_loss(stream_points, gt_points, gt_valid)

    write_pointcloud_ply(
        output_dir / "gt_object.ply",
        gt_points,
        rgb,
        gt_mask,
        max_points=config.max_visual_points,
    )
    write_pointcloud_ply(
        output_dir / "streamvggt_object_gtmask.ply",
        stream_points,
        rgb,
        gt_mask,
        max_points=config.max_visual_points,
    )
    write_streamvggt_baseline_visualization(
        output_dir / "streamvggt_baseline.png",
        sequence=sequence,
        batch=batch,
        prompt=prompt_selection.prompt,
    )
    print(
        "streamvggt baseline exported "
        f"dir={output_dir} "
        f"stream_point_loss={float(stream_loss.detach().cpu()):.6f} "
        f"gt_mask_pixels={int(gt_mask.sum().item())}"
    )


def train_dense_fusion(config: DenseFusionTrainConfig) -> None:
    rng = random.Random(config.seed)
    torch.manual_seed(config.seed)
    use_fused_sam = uses_fused_sam_decoder(config)
    use_fused_sam_residual = uses_fused_sam_residual_features(config)
    use_sam3_direct = uses_sam3_direct_masks(config)
    object_config = ObjectSamplingConfig(
        min_pixels=config.min_pixels,
        max_area_ratio=config.max_area_ratio,
        min_visible_frames=config.min_visible_frames,
        max_objects_per_frame=config.max_objects_per_frame,
        ignore_instance_id=config.ignore_instance_id,
        semantic_ignore_label=config.semantic_ignore_label,
    )
    dataset = ScanNetPPObjectSequenceDataset(
        config.manifest,
        scene_id=config.scene_id,
        sequence_length=config.sequence_length,
        frame_stride=config.frame_stride,
        frame_indices=config.frame_indices,
        object_config=object_config,
    )

    sam3_model = load_sam3_image_model(
        repo_path=config.sam3_repo,
        checkpoint_path=config.sam3_checkpoint,
        device=config.sam3_device,
        enable_inst_interactivity=(
            config.sam3_enable_inst_interactivity or use_fused_sam
        ),
    )
    sam3_model.requires_grad_(False)
    sam_tracker_model = None
    if use_fused_sam:
        inst_predictor = getattr(sam3_model, "inst_interactive_predictor", None)
        if inst_predictor is None:
            raise RuntimeError(
                "history.update_source='fused_sam' requires SAM3 instance "
                "interactivity. The SAM3 model did not expose inst_interactive_predictor."
            )
        sam_tracker_model = inst_predictor.model.to(config.device).eval()
        sam_tracker_model.requires_grad_(False)
    sam3 = SAM3IntermediateAdapter(
        sam3_model,
        device=config.sam3_device,
        resolution=config.sam3_resolution,
        source=config.sam3_feature_source,
        text_conditioning=config.sam3_text_conditioning,
        token_grid=config.feature_grid,
    )
    sam3_direct_tracker = None
    def get_sam3_direct_tracker() -> SAM3VideoTrackerAdapter:
        nonlocal sam3_direct_tracker
        if sam3_direct_tracker is None:
            sam3_direct_predictor = load_sam3_video_predictor(
                repo_path=config.sam3_repo,
                checkpoint_path=config.sam3_checkpoint,
                device=config.sam3_direct_device,
                async_loading_frames=config.sam3_direct_async_loading_frames,
            )
            sam3_direct_tracker = SAM3VideoTrackerAdapter(
                sam3_direct_predictor,
                output_prob_thresh=config.sam3_direct_output_prob_thresh,
                prompt_with_box=config.sam3_direct_prompt_with_box,
            )
        return sam3_direct_tracker

    if use_sam3_direct:
        print(
            "enabled SAM3 direct mask provider "
            f"device={config.sam3_direct_device} "
            f"box={'yes' if config.sam3_direct_prompt_with_box else 'no'} "
            f"threshold={config.sam3_direct_output_prob_thresh:.2f}"
        )

    streamvggt_model = load_streamvggt_latent_model(
        repo_path=config.streamvggt_repo,
        checkpoint_path=config.streamvggt_checkpoint,
        device=config.geometry_device,
        strict=True,
    )
    streamvggt_model.requires_grad_(False)
    stream_point_head_state = None
    if config.point_decoder == "stream_dpt" and config.stream_dpt_use_pretrained:
        point_head = getattr(streamvggt_model, "point_head", None)
        if point_head is None:
            raise RuntimeError(
                "model.point_decoder='stream_dpt' with stream_dpt_use_pretrained=True "
                "requires StreamVGGT to expose point_head."
            )
        stream_point_head_state = {
            key: value.detach().cpu()
            for key, value in point_head.state_dict().items()
        }
    geometry = StreamVGGTLatentAdapter(
        streamvggt_model,
        device=config.geometry_device,
        token_grid=config.feature_grid,
        context_grid=config.context_grid,
        layer_index=config.streamvggt_layer_index,
        dpt_layer_indices=config.streamvggt_dpt_layer_indices,
        image_mode=config.streamvggt_image_mode,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = config.output_dir / "training_history.csv"
    write_metrics_header(metrics_path)

    model: DenseSAMVGGTModel | None = None
    optimizer: torch.optim.Optimizer | None = None
    completed_steps = 0
    overfit_sequence: ObjectSequence | None = None
    overfit_prompt_selection: ObjectPromptSelection | None = None
    overfit_feature_cache: Dict[str, Any] | None = None
    backbones_released = False

    if config.overfit:
        overfit_sequence, overfit_prompt_selection, resolved_window_index = (
            select_overfit_sequence(
                dataset,
                config=config,
                rng=rng,
            )
        )
        if config.overfit_window_index != resolved_window_index:
            print(
                "overfit window adjusted "
                f"requested={config.overfit_window_index} "
                f"resolved={resolved_window_index}"
            )
        if overfit_prompt_selection is None:
            raise RuntimeError(
                "Could not select an overfit target for "
                f"window_index={config.overfit_window_index}. Try a different "
                "--window-index, lower object filters, or pass --instance-id."
            )
        preview_batch = build_dense_batch(
            overfit_sequence,
            overfit_prompt_selection,
            config=config,
            device=config.device,
        )
        if preview_batch is None:
            raise RuntimeError(
                "Selected overfit target produced no prompt_mask. Try a "
                "different --window-index / --instance-id, or lower min_pixels / "
                "min_visible_frames."
            )
        preview_reference_frame_idx = int(
            preview_batch["reference_frame_idx"].detach().cpu().item()
        )
        preview_prompt_pixels = [
            int(value)
            for value in preview_batch["prompt_mask"]
            .flatten(1)
            .sum(dim=1)
            .detach()
            .cpu()
            .tolist()
        ]
        preview_has_camera = "intrinsics" in preview_batch
        del preview_batch
        print(
            "overfit target "
            f"scene={overfit_sequence.scene_id} "
            f"frames={overfit_sequence.frame_indices} "
            f"instance={overfit_prompt_selection.sampled_instance_id} "
            f"label='{overfit_prompt_selection.sampled_label}' "
            f"target_mode={config.target_mode} "
            f"reference_frame={preview_reference_frame_idx} "
            f"prompt_pixels={preview_prompt_pixels} "
            f"camera={'yes' if preview_has_camera else 'no'}"
        )

    for step in range(1, config.iterations + 1):
        if config.overfit:
            assert overfit_sequence is not None and overfit_prompt_selection is not None
            sequence = overfit_sequence
            prompt_selection = overfit_prompt_selection
        else:
            sequence = dataset.sample(rng)
            prompt_selection = select_training_prompt(
                sequence,
                config=config,
                rng=rng,
            )
        if prompt_selection is None:
            continue
        if sequence.pointmaps is None:
            raise RuntimeError(
                "Dense fusion training requires full-resolution pointmaps. "
                "Re-run scripts/prepare_scannetpp_2d.py with --save-pointmaps."
            )

        if config.overfit and overfit_feature_cache is not None:
            sam_tokens_all = overfit_feature_cache["sam_tokens"]
            geometry_tokens_all = overfit_feature_cache["geometry_tokens"]
            geometry_camera_tokens = overfit_feature_cache["camera_tokens"]
            stream_dpt_tokens_all = overfit_feature_cache.get("stream_dpt_tokens")
            stream_images_all = overfit_feature_cache.get("stream_images")
            stream_patch_start_idx = overfit_feature_cache.get("stream_patch_start_idx")
            stream_baseline_pointmaps = overfit_feature_cache.get(
                "stream_baseline_pointmaps"
            )
            sam_tracker_features = overfit_feature_cache.get("sam_tracker_features")
            sam3_direct_masks = overfit_feature_cache.get("sam3_direct_masks")
            sam3_direct_aux = overfit_feature_cache.get("sam3_direct_aux", {})
            text_embedding = overfit_feature_cache["text_embedding"].to(config.device)
            batch = batch_to_device(overfit_feature_cache["batch"], config.device)
        else:
            sam_tracker_features = None
            sam3_direct_masks = None
            sam3_direct_aux: Dict[str, Any] = {}
            with torch.no_grad():
                sam_out = extract_sam3_sequence(
                    sam3,
                    sequence.image_paths,
                    prompt=prompt_selection.prompt,
                    chunk_size=config.sam3_frame_chunk_size,
                    sam_tracker_for_features=(
                        sam_tracker_model if use_fused_sam_residual else None
                    ),
                )
                geo_out = geometry.extract_from_paths(
                    sequence.image_paths,
                    return_pointmap=config.point_decoder == "stream_dpt",
                    streaming_cache=config.geometry_streaming_cache,
                )
                text_embedding = pool_language_features(sam_out.text_out)
                if text_embedding is None:
                    raise RuntimeError("SAM3 did not return language_features for text alignment.")
                text_embedding = text_embedding.to(config.device).float()
                if use_fused_sam_residual:
                    sam_tracker_features = getattr(
                        sam_out,
                        "sam_tracker_features",
                        None,
                    )
                    if sam_tracker_features is None:
                        raise RuntimeError(
                            "fused_sam.feature_mode='residual' requires SAM tracker features."
                        )
                sam_tokens_all = sam_out.semantic.tokens.detach().cpu()
                geometry_tokens_all = geo_out.geometry.tokens.detach().cpu()
                geometry_camera_tokens = (
                    geo_out.geometry.camera_tokens.detach().cpu()
                    if geo_out.geometry.camera_tokens is not None
                    else None
                )
                if config.point_decoder == "stream_dpt":
                    stream_dpt_tokens_all = detach_to_cpu(
                        geo_out.geometry.aux.get("stream_dpt_tokens")
                    )
                    stream_images_all = detach_to_cpu(
                        geo_out.geometry.aux.get("stream_images")
                    )
                    stream_patch_start_idx = geo_out.geometry.aux.get("patch_start_idx")
                    stream_baseline_pointmaps = detach_to_cpu(
                        geo_out.geometry.aux.get("pointmap_dense")
                    )
                    if stream_baseline_pointmaps is None:
                        stream_baseline_pointmaps = detach_to_cpu(geo_out.pointmap_grid)
                    if (
                        stream_dpt_tokens_all is None
                        or stream_images_all is None
                        or stream_patch_start_idx is None
                    ):
                        raise RuntimeError(
                            "StreamVGGT adapter did not return stream_dpt_tokens, "
                            "stream_images, and patch_start_idx required by "
                            "model.point_decoder='stream_dpt'."
                        )
                    stream_patch_start_idx = int(stream_patch_start_idx)
                    if stream_baseline_pointmaps is None:
                        raise RuntimeError(
                            "StreamVGGT adapter did not return pointmap_grid for "
                            "the direct StreamVGGT pointmap comparison."
                        )
                    stream_baseline_pointmaps = resize_dense_bhwc_tensor(
                        stream_baseline_pointmaps,
                        config.output_size,
                    )
                else:
                    stream_dpt_tokens_all = None
                    stream_images_all = None
                    stream_patch_start_idx = None
                    stream_baseline_pointmaps = None

            batch = build_dense_batch(
                sequence,
                prompt_selection,
                config=config,
                device=config.device,
            )
            if use_sam3_direct and batch is not None:
                reference_frame_idx = int(
                    batch["reference_frame_idx"].detach().cpu().item()
                )
                sam3_track = get_sam3_direct_tracker().track_from_paths(
                    sequence.image_paths,
                    prompt=prompt_selection.prompt,
                    output_size=config.output_size,
                    prompt_frame_idx=reference_frame_idx,
                    reference_mask=batch["reference_mask"].detach().cpu(),
                )
                sam3_direct_masks = sam3_track.masks.detach().cpu()
                sam3_direct_aux = {
                    **sam3_track.aux,
                    "selected_obj_id": sam3_track.selected_obj_id,
                    "prompt_frame_idx": sam3_track.prompt_frame_idx,
                    "prompt_box_xywh": sam3_track.prompt_box_xywh,
                }
            if config.overfit and batch is not None:
                overfit_feature_cache = {
                    "sam_tokens": sam_tokens_all,
                    "geometry_tokens": geometry_tokens_all,
                    "camera_tokens": geometry_camera_tokens,
                    "stream_dpt_tokens": stream_dpt_tokens_all,
                    "stream_images": stream_images_all,
                    "stream_patch_start_idx": stream_patch_start_idx,
                    "stream_baseline_pointmaps": stream_baseline_pointmaps,
                    "sam_tracker_features": sam_tracker_features,
                    "sam3_direct_masks": sam3_direct_masks,
                    "sam3_direct_aux": sam3_direct_aux,
                    "text_embedding": text_embedding.detach().cpu(),
                    "batch": batch_to_cpu(batch),
                }
                if not backbones_released:
                    del sam3
                    del sam3_model
                    del geometry
                    del streamvggt_model
                    if sam3_direct_tracker is not None:
                        del sam3_direct_tracker
                        sam3_direct_tracker = None
                    backbones_released = True
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        if batch is None:
            if config.overfit:
                raise RuntimeError(
                    "Overfit target produced no prompt_mask. Try a different "
                    "--window-index / --instance-id, or lower min_pixels / "
                    "min_visible_frames."
                )
            continue

        if model is None:
            sam_dim = int(sam_tokens_all.shape[-1])
            geometry_dim = int(geometry_tokens_all.shape[-1])
            camera_dim = (
                int(geometry_camera_tokens.shape[-1])
                if config.use_camera_tokens
                and geometry_camera_tokens is not None
                else None
            )
            model = DenseSAMVGGTModel(
                sam_dim=sam_dim,
                geometry_dim=geometry_dim,
                text_dim=int(text_embedding.shape[-1]),
                camera_dim=camera_dim,
                d_fuse=config.d_fuse,
                num_heads=config.num_heads,
                output_size=config.output_size,
                feature_grid=config.feature_grid,
                embedding_dim=config.embedding_dim,
                num_classes=config.num_classes,
                dropout=config.dropout,
                point_decoder=config.point_decoder,
                point_mask_condition=config.point_mask_condition,
                stream_dpt_freeze=config.stream_dpt_freeze,
                enable_fused_sam_decoder=use_fused_sam,
                fusion_type=config.fusion_type,
            ).to(config.device)
            if config.point_decoder == "stream_dpt" and stream_point_head_state is not None:
                load_result = model.load_stream_point_decoder_state_dict(
                    stream_point_head_state,
                    strict=False,
                )
                if hasattr(load_result, "missing_keys"):
                    missing_keys = load_result.missing_keys
                    unexpected_keys = load_result.unexpected_keys
                else:
                    missing_keys, unexpected_keys = load_result
                print(
                    "loaded StreamVGGT point_head into stream_dpt decoder "
                    f"missing={len(missing_keys)} unexpected={len(unexpected_keys)}"
                )
            trainable_params = configure_trainable_parameters(
                model,
                config=config,
                use_fused_sam=use_fused_sam,
            )
            optimizer = torch.optim.AdamW(trainable_params, lr=config.lr)
            print(
                "initialized DenseSAMVGGTModel "
                f"sam_dim={sam_dim} geometry_dim={geometry_dim} "
                f"text_dim={int(text_embedding.shape[-1])} camera_dim={camera_dim} "
                f"output_size={config.output_size} "
                f"point_decoder={config.point_decoder} "
                f"point_mask_condition={config.point_mask_condition} "
                f"fusion_type={config.fusion_type} "
                f"primary_mask_source={config.primary_mask_source} "
                f"fused_sam_feature_mode={config.fused_sam_feature_mode} "
                f"train_scope={config.train_scope}"
            )

        assert model is not None and optimizer is not None
        num_frames = len(sequence.image_paths)
        feature_tokens_per_frame = config.feature_grid[0] * config.feature_grid[1]
        context_tokens_per_frame = config.context_grid[0] * config.context_grid[1]
        sam_frame_tokens = split_sequence_tokens(
            sam_tokens_all.float().to(config.device),
            num_frames=num_frames,
            tokens_per_frame=feature_tokens_per_frame,
            name="sam_tokens",
        )
        geometry_frame_tokens = split_sequence_tokens(
            geometry_tokens_all.float().to(config.device),
            num_frames=num_frames,
            tokens_per_frame=context_tokens_per_frame,
            name="geometry_tokens",
        )
        camera_tokens = (
            geometry_camera_tokens.float().to(config.device)
            if config.use_camera_tokens and geometry_camera_tokens is not None
            else None
        )
        sam3_direct_masks_device = (
            sam3_direct_masks.to(config.device).bool()
            if sam3_direct_masks is not None
            else None
        )
        stream_baseline_pointmaps_device = (
            stream_baseline_pointmaps.to(config.device).float()
            if stream_baseline_pointmaps is not None
            else None
        )

        fused_frame_tokens: List[torch.Tensor] = []
        for frame_idx in range(num_frames):
            fused_frame_tokens.append(
                model.fuse_tokens(
                    sam_tokens=sam_frame_tokens[frame_idx],
                    geometry_tokens=geometry_frame_tokens[frame_idx],
                    camera_tokens=slice_camera_tokens(
                        camera_tokens,
                        frame_idx=frame_idx,
                        num_frames=num_frames,
                    ),
                )
            )

        reference_frame_idx = int(batch["reference_frame_idx"].detach().cpu().item())
        object_query = model.pool_object_query(
            fused_frame_tokens[reference_frame_idx],
            batch["reference_mask"],
        )

        frame_losses: List[torch.Tensor] = []
        mask_losses: List[torch.Tensor] = []
        dice_losses: List[torch.Tensor] = []
        point_losses: List[torch.Tensor] = []
        stream_baseline_point_losses: List[torch.Tensor] = []
        chamfer_losses: List[torch.Tensor] = []
        reprojection_losses: List[torch.Tensor] = []
        fused_sam_mask_losses: List[torch.Tensor] = []
        fused_sam_dice_losses: List[torch.Tensor] = []
        sam3_direct_ious: List[float] = []
        text_losses: List[torch.Tensor] = []
        aux_losses: List[torch.Tensor] = []
        match_losses: List[torch.Tensor] = []
        history_buffer: List[Dict[str, torch.Tensor]] = []
        outputs = []
        selected_match_pixels = 0
        selected_point_pixels = 0
        gt_point_pixels = int(
            (batch["prompt_mask"] & batch["point_valid"]).sum().item()
        )

        for frame_idx in range(num_frames):
            target = frame_target(batch, frame_idx)
            point_mask_condition = select_point_mask_condition(
                target=target,
                source=config.point_mask_condition,
            )
            output = model.decode(
                fused_tokens=fused_frame_tokens[frame_idx],
                text_embedding=text_embedding,
                object_query=object_query,
                stream_tokens=slice_stream_dpt_tokens(
                    stream_dpt_tokens_all,
                    frame_idx=frame_idx,
                    device=config.device,
                ),
                stream_images=slice_stream_images(
                    stream_images_all,
                    frame_idx=frame_idx,
                    device=config.device,
                ),
                stream_patch_start_idx=stream_patch_start_idx,
                point_mask_condition=point_mask_condition,
            )
            if sam3_direct_masks_device is not None:
                output.sam3_direct_mask = sam3_direct_masks_device[frame_idx]
                sam3_direct_ious.append(
                    binary_iou_tensor(output.sam3_direct_mask, target["prompt_mask"])
                )
            if stream_baseline_pointmaps_device is not None:
                output.streamvggt_pointmap = stream_baseline_pointmaps_device[
                    frame_idx : frame_idx + 1
                ]
            zero = output.mask_logits.sum() * 0.0
            fused_sam_mask_loss = zero
            fused_sam_dice_loss = zero
            if use_fused_sam:
                if sam_tracker_model is None:
                    raise RuntimeError("Fused SAM decoder requested but SAM tracker model is missing.")
                fused_sam_prompt = select_fused_sam_prompt_mask(
                    output=output,
                    target=target,
                    source=config.fused_sam_prompt_source,
                )
                output.fused_sam_mask_logits = model.decode_fused_sam_mask(
                    fused_tokens=fused_frame_tokens[frame_idx],
                    sam_tracker=sam_tracker_model,
                    mask_prompt=fused_sam_prompt,
                    object_query=object_query,
                    sam_features=slice_sam3_tracker_features(
                        sam_tracker_features,
                        frame_idx=frame_idx,
                        device=config.device,
                    ),
                    feature_mode=config.fused_sam_feature_mode,
                )
                fused_sam_mask_loss = dense_mask_bce_loss(
                    output.fused_sam_mask_logits[0],
                    target["prompt_mask"],
                    target["mask_supervision"],
                )
                fused_sam_dice_loss = dense_dice_loss(
                    output.fused_sam_mask_logits[0],
                    target["prompt_mask"],
                    target["mask_supervision"],
                )
            apply_primary_mask_source(output, config.primary_mask_source)
            outputs.append(output)

            mask_loss = dense_mask_bce_loss(
                output.mask_logits[0],
                target["prompt_mask"],
                target["mask_supervision"],
            )
            dice_loss = dense_dice_loss(
                output.mask_logits[0],
                target["prompt_mask"],
                target["mask_supervision"],
            )
            point_valid = select_point_supervision_mask(
                output=output,
                target=target,
                source=config.point_valid_source,
                pred_threshold=config.point_valid_threshold,
            )
            point_loss = dense_point_loss(
                output.pointmap[0],
                target["pointmap"],
                point_valid,
            )
            stream_baseline_point_loss = zero
            if output.streamvggt_pointmap is not None:
                stream_baseline_point_loss = dense_point_loss(
                    output.streamvggt_pointmap[0],
                    target["pointmap"],
                    target["point_valid"] & target["prompt_mask"],
                )
            chamfer_loss = dense_chamfer_loss(
                output.pointmap[0],
                target["pointmap"],
                point_valid,
                max_points=config.max_chamfer_points,
            )
            reprojection_loss = dense_reprojection_mask_loss(
                output.pointmap[0],
                output.mask_logits[0],
                target["prompt_mask"],
                target.get("intrinsics"),
                target.get("world_to_camera"),
            )
            text_loss = dense_mask_bce_loss(
                output.prompt_score[0],
                target["prompt_mask"],
                target["mask_supervision"],
            )
            aux_loss = zero
            if output.aux_logits is not None:
                aux_valid = target["semantic_valid"]
                if aux_valid.any():
                    aux_loss = F.cross_entropy(
                        output.aux_logits[0].permute(1, 2, 0)[aux_valid],
                        target["semantic"][aux_valid].long(),
                    )

            match_loss = zero
            match_pixels = 0
            if history_buffer:
                history = history_buffer[rng.randrange(len(history_buffer))]
                match_loss, match_pixels = dense_instance_match_loss(
                    current_embeddings=output.instance_embedding[0],
                    history_embeddings=history["embeddings"],
                    current_instance_ids=target["instance"],
                    history_instance_ids=history["instance"],
                    current_valid=target["instance_valid"],
                    history_valid=history["valid"],
                    max_pixels=config.max_match_pixels,
                    negative_ratio=config.negative_ratio,
                    temperature=config.temperature,
                )

            frame_loss = (
                config.mask_weight * mask_loss
                + config.dice_weight * dice_loss
                + config.point_weight * point_loss
                + config.chamfer_weight * chamfer_loss
                + config.reprojection_weight * reprojection_loss
                + config.fused_sam_mask_weight * fused_sam_mask_loss
                + config.fused_sam_dice_weight * fused_sam_dice_loss
                + config.text_weight * text_loss
                + config.aux_cls_weight * aux_loss
                + config.match_weight * match_loss
            )
            frame_losses.append(frame_loss)
            mask_losses.append(mask_loss)
            dice_losses.append(dice_loss)
            point_losses.append(point_loss)
            stream_baseline_point_losses.append(stream_baseline_point_loss)
            chamfer_losses.append(chamfer_loss)
            reprojection_losses.append(reprojection_loss)
            fused_sam_mask_losses.append(fused_sam_mask_loss)
            fused_sam_dice_losses.append(fused_sam_dice_loss)
            text_losses.append(text_loss)
            aux_losses.append(aux_loss)
            match_losses.append(match_loss)
            selected_match_pixels += match_pixels
            selected_point_pixels += int(point_valid.sum().detach().cpu().item())
            history_buffer.append(
                {
                    "embeddings": output.instance_embedding[0].detach(),
                    "instance": target["instance"].detach(),
                    "valid": target["instance_valid"].detach(),
                }
            )
            if config.history_enabled:
                update_mask = select_history_update_mask(
                    output=output,
                    target=target,
                    source=config.history_update_source,
                    pred_threshold=config.history_pred_threshold,
                )
                object_query = model.update_object_query(
                    object_query,
                    fused_frame_tokens[frame_idx],
                    update_mask,
                )

        loss = torch.stack(frame_losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        completed_steps += 1

        row = {
            "step": step,
            "loss": float(loss.detach().cpu()),
            "mask_loss": mean_metric(mask_losses),
            "dice_loss": mean_metric(dice_losses),
            "point_loss": mean_metric(point_losses),
            "streamvggt_point_loss": mean_metric(stream_baseline_point_losses),
            "chamfer_loss": mean_metric(chamfer_losses),
            "reprojection_loss": mean_metric(reprojection_losses),
            "fused_sam_mask_loss": mean_metric(fused_sam_mask_losses),
            "fused_sam_dice_loss": mean_metric(fused_sam_dice_losses),
            "sam3_direct_iou": mean_float(sam3_direct_ious),
            "text_loss": mean_metric(text_losses),
            "aux_cls_loss": mean_metric(aux_losses),
            "match_loss": mean_metric(match_losses),
            "num_prompt_pixels": int(batch["prompt_mask"].sum().item()),
            "num_supervised_pixels": int(batch["mask_supervision"].sum().item()),
            "num_point_pixels": selected_point_pixels,
            "num_gt_point_pixels": gt_point_pixels,
            "num_match_pixels": int(selected_match_pixels),
            "point_valid_source": config.point_valid_source,
            "point_mask_condition": config.point_mask_condition,
            "fusion_type": config.fusion_type,
            "primary_mask_source": config.primary_mask_source,
            "use_camera_tokens": int(config.use_camera_tokens),
            "train_scope": config.train_scope,
            "history_update_source": config.history_update_source,
            "fused_sam_prompt_source": config.fused_sam_prompt_source,
            "fused_sam_feature_mode": config.fused_sam_feature_mode,
            "sam3_direct_selected_obj_id": (
                "" if not sam3_direct_aux else sam3_direct_aux.get("selected_obj_id", "")
            ),
            "sam3_direct_prompt_frame_idx": (
                "" if not sam3_direct_aux else sam3_direct_aux.get("prompt_frame_idx", "")
            ),
            "prompt": prompt_selection.prompt,
            "sampled_instance_id": int(prompt_selection.sampled_instance_id),
            "sampled_label": prompt_selection.sampled_label,
            "target_mode": config.target_mode,
            "reference_frame_idx": reference_frame_idx,
        }
        append_metric(metrics_path, row)

        if step % config.log_every == 0 or completed_steps == 1:
            print(
                "step={step} loss={loss:.4f} mask={mask_loss:.4f} "
                "dice={dice_loss:.4f} point={point_loss:.4f} "
                "stream_point={streamvggt_point_loss:.4f} "
                "chamfer={chamfer_loss:.4f} reproj={reprojection_loss:.4f} "
                "fused_sam={fused_sam_mask_loss:.4f}/{fused_sam_dice_loss:.4f} "
                "sam3_direct_iou={sam3_direct_iou:.4f} "
                "text={text_loss:.4f} match={match_loss:.4f} "
                "prompt_pixels={num_prompt_pixels} "
                "prompt='{prompt}'".format(**row)
            )

        if config.visualize_every > 0 and (
            step % config.visualize_every == 0 or completed_steps == 1
        ):
            save_dense_visualization(
                config.output_dir / "visualizations" / f"step_{step:06d}.png",
                sequence=sequence,
                batch=batch,
                outputs=outputs,
                prompt=prompt_selection.prompt,
                step=step,
                threshold=config.visualize_threshold,
                primary_mask_source=config.primary_mask_source,
            )
            export_dense_pointclouds(
                config.output_dir / "pointclouds",
                sequence=sequence,
                batch=batch,
                outputs=outputs,
                step=step,
                threshold=config.visualize_threshold,
                max_points=config.max_visual_points,
                gt_once=config.overfit,
            )

        if step % config.save_every == 0:
            save_checkpoint(
                config.output_dir / f"ckpt_step{step:06d}.pt",
                model,
                optimizer,
                step,
                config,
            )
            plot_training_curves(
                metrics_path,
                config.output_dir / "training_curves.png",
                title="Dense SAM3/StreamVGGT Fusion Training",
            )

    if model is None or optimizer is None:
        raise RuntimeError("No valid dense training batches were produced.")
    save_checkpoint(
        config.output_dir / "ckpt_last.pt",
        model,
        optimizer,
        config.iterations,
        config,
    )
    plot_training_curves(
        metrics_path,
        config.output_dir / "training_curves.png",
        title="Dense SAM3/StreamVGGT Fusion Training",
    )
    print(f"training history: {metrics_path}")
    print(f"training curves: {config.output_dir / 'training_curves.png'}")


def build_dense_batch(
    sequence: ObjectSequence,
    prompt_selection: ObjectPromptSelection,
    *,
    config: DenseFusionTrainConfig,
    device: str,
) -> Dict[str, torch.Tensor] | None:
    output_size = config.output_size
    target_mode = config.target_mode.strip().lower()
    if target_mode not in {"class", "category", "instance", "sampled_instance"}:
        raise ValueError(
            "objects.target_mode must be 'class' or 'instance', "
            f"got {config.target_mode!r}"
        )
    keep_per_frame = keep_instances_visible_in_multiple_frames(
        [list(ids) for ids in sequence.visible_instance_ids],
        min_visible_frames=config.min_visible_frames,
    )
    target_filters = normalize_label_filters(prompt_selection.target_object_labels)
    excluded_filters = normalize_label_filters(config.excluded_object_labels)
    excluded_semantic = set(int(v) for v in config.excluded_semantic_labels)

    prompt_masks = []
    mask_supervision = []
    instance_valids = []
    semantic_valids = []
    point_valids = []
    instances = []
    semantics = []
    pointmaps = []
    intrinsics = []
    world_to_cameras = []
    has_camera = (
        sequence.camera_intrinsics is not None
        and sequence.world_to_camera is not None
    )

    for frame_idx, (inst_np, sem_np, point_np) in enumerate(
        zip(sequence.instance_masks, sequence.semantic_masks, sequence.pointmaps or [])
    ):
        inst = resize_label_mask(inst_np, output_size)
        sem = resize_label_mask(sem_np, output_size)
        point, point_valid = resize_pointmap(point_np, output_size)
        if has_camera:
            intrinsics.append(
                torch.from_numpy(
                    scale_intrinsics(
                        sequence.camera_intrinsics[frame_idx],
                        source_hw=inst_np.shape[:2],
                        output_hw=output_size,
                    )
                ).float()
            )
            world_to_cameras.append(
                torch.from_numpy(sequence.world_to_camera[frame_idx]).float()
            )

        visible = {
            int(instance_id)
            for instance_id in keep_per_frame[frame_idx]
            if not label_is_excluded(
                sequence.object_labels.get(int(instance_id)),
                excluded_filters,
            )
        }
        if target_mode in {"instance", "sampled_instance"}:
            target_id = int(prompt_selection.sampled_instance_id)
            target_instances = {target_id} if np.any(inst == target_id) else set()
        else:
            target_instances = {
                instance_id
                for instance_id in visible
                if label_matches(sequence.object_labels.get(instance_id), target_filters)
            }
            if not target_filters:
                target_instances = set(visible)

        object_valid = np.isin(inst, list(visible)) if visible else np.zeros_like(inst, dtype=bool)
        prompt_mask = (
            np.isin(inst, list(target_instances))
            if target_instances
            else np.zeros_like(inst, dtype=bool)
        )
        semantic_valid = (sem != config.semantic_ignore_label) & (sem >= 0)
        semantic_valid &= sem < config.num_classes
        if excluded_semantic:
            semantic_valid &= ~np.isin(sem, list(excluded_semantic))
        if target_mode in {"instance", "sampled_instance"}:
            supervision = semantic_valid
            instance_valid = prompt_mask & point_valid
        else:
            supervision = object_valid & semantic_valid
            instance_valid = object_valid & point_valid

        prompt_masks.append(torch.from_numpy(prompt_mask).bool())
        mask_supervision.append(torch.from_numpy(supervision | prompt_mask).bool())
        instance_valids.append(torch.from_numpy(instance_valid).bool())
        semantic_valids.append(torch.from_numpy(semantic_valid & object_valid).bool())
        point_valids.append(torch.from_numpy(point_valid).bool())
        instances.append(torch.from_numpy(inst.astype(np.int64)).long())
        semantics.append(torch.from_numpy(sem.astype(np.int64)).long())
        pointmaps.append(torch.from_numpy(point).float())

    if not prompt_masks or not torch.stack(prompt_masks).any():
        return None

    def stack(items: Sequence[torch.Tensor]) -> torch.Tensor:
        return torch.stack(list(items), dim=0).to(device)

    prompt_stack = stack(prompt_masks)
    visible_frames = prompt_stack.flatten(1).any(dim=1).nonzero(as_tuple=False).flatten()
    reference_frame_idx = int(visible_frames[0].detach().cpu().item())

    batch = {
        "prompt_mask": prompt_stack,
        "mask_supervision": stack(mask_supervision),
        "instance_valid": stack(instance_valids),
        "semantic_valid": stack(semantic_valids),
        "point_valid": stack(point_valids),
        "instance": stack(instances),
        "semantic": stack(semantics),
        "pointmap": stack(pointmaps),
        "reference_mask": prompt_stack[reference_frame_idx],
        "reference_frame_idx": torch.tensor(reference_frame_idx, device=device),
    }
    if has_camera:
        batch["intrinsics"] = stack(intrinsics)
        batch["world_to_camera"] = stack(world_to_cameras)
    return batch


def scale_intrinsics(
    intrinsics: np.ndarray,
    *,
    source_hw: tuple[int, int],
    output_hw: tuple[int, int],
) -> np.ndarray:
    scaled = np.asarray(intrinsics, dtype=np.float32).copy()
    src_h, src_w = int(source_hw[0]), int(source_hw[1])
    out_h, out_w = int(output_hw[0]), int(output_hw[1])
    scaled[0, 0] *= out_w / float(src_w)
    scaled[0, 2] *= out_w / float(src_w)
    scaled[1, 1] *= out_h / float(src_h)
    scaled[1, 2] *= out_h / float(src_h)
    return scaled


def frame_target(batch: Dict[str, torch.Tensor], frame_idx: int) -> Dict[str, torch.Tensor]:
    num_frames = int(batch["prompt_mask"].shape[0])
    target: Dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if value.ndim > 0 and value.shape[0] == num_frames:
            target[key] = value[frame_idx]
        else:
            target[key] = value
    return target


def resize_label_mask(mask: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(np.asarray(mask))
    image = image.resize((output_size[1], output_size[0]), Image.NEAREST)
    return np.asarray(image)


def resize_pointmap(
    pointmap: np.ndarray,
    output_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    points = torch.from_numpy(np.asarray(pointmap, dtype=np.float32))
    valid = torch.isfinite(points).all(dim=-1)
    points = torch.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)
    points_chw = points.permute(2, 0, 1).unsqueeze(0)
    valid_chw = valid.float().unsqueeze(0).unsqueeze(0)
    weighted = F.interpolate(
        points_chw * valid_chw,
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )
    weights = F.interpolate(
        valid_chw,
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )
    resized = weighted / weights.clamp_min(1e-6)
    resized_valid = weights[0, 0] > 0.5
    return resized[0].permute(1, 2, 0).numpy().astype(np.float32), resized_valid.numpy()


def resize_dense_bhwc_tensor(
    values: torch.Tensor,
    output_size: tuple[int, int],
) -> torch.Tensor:
    if values.ndim != 4:
        raise ValueError(f"Expected dense tensor [B, H, W, C], got {tuple(values.shape)}")
    if tuple(values.shape[1:3]) == tuple(output_size):
        return values.detach().float().cpu()
    x = values.detach().float().permute(0, 3, 1, 2)
    x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
    return x.permute(0, 2, 3, 1).contiguous().cpu()


def dense_mask_bce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    supervision: torch.Tensor,
) -> torch.Tensor:
    if not supervision.any():
        return logits.sum() * 0.0
    target_f = target.to(logits.dtype)
    selected_target = target_f[supervision]
    pos = selected_target.sum()
    neg = selected_target.numel() - pos
    pos_weight = (neg / pos.clamp_min(1.0)).clamp(1.0, 20.0)
    return F.binary_cross_entropy_with_logits(
        logits[supervision],
        selected_target,
        pos_weight=pos_weight,
    )


def dense_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    supervision: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    if not supervision.any() or not target.any():
        return logits.sum() * 0.0
    probs = logits.sigmoid() * supervision.to(logits.dtype)
    target_f = target.to(logits.dtype) * supervision.to(logits.dtype)
    intersection = (probs * target_f).sum()
    union = probs.sum() + target_f.sum()
    return 1.0 - (2.0 * intersection + eps) / (union + eps)


def select_point_supervision_mask(
    *,
    output: Any,
    target: Dict[str, torch.Tensor],
    source: str,
    pred_threshold: float,
) -> torch.Tensor:
    source = source.strip().lower()
    point_valid = target["point_valid"]
    if source in {"gt", "target", "teacher"}:
        return point_valid & target["prompt_mask"]
    if source in {"pred", "prediction"}:
        pred_mask = output.mask_logits[0].detach().sigmoid() > float(pred_threshold)
        return point_valid & pred_mask
    if source in {"sam3_direct", "sam3", "sam_video", "sam3_video"}:
        sam3_mask = getattr(output, "sam3_direct_mask", None)
        if sam3_mask is None:
            raise RuntimeError(
                "loss.point_valid_source='sam3_direct' requires sam3_direct_mask. "
                "Use history.update_source='sam3_direct' or set point_valid_source "
                "to sam3_direct so the SAM3 video tracker is enabled."
            )
        return point_valid & sam3_mask.detach().bool()
    raise ValueError(
        "loss.point_valid_source must be 'gt', 'pred', or 'sam3_direct', "
        f"got {source!r}"
    )


def select_point_mask_condition(
    *,
    target: Dict[str, torch.Tensor],
    source: str,
) -> torch.Tensor | None:
    source = source.strip().lower()
    if source in {"none", "no", "disabled"}:
        return None
    if source in {"gt_soft", "gt", "target", "teacher"}:
        return target["prompt_mask"].float()
    raise ValueError(
        "model.point_mask_condition must be 'none' or 'gt_soft', "
        f"got {source!r}"
    )


def apply_primary_mask_source(output: Any, source: str) -> None:
    """Route the main predicted mask through the selected mask provider."""
    source = source.strip().lower()
    if source in {"dense", "mask_head", "baseline"}:
        return
    if source in {"fused_sam", "sam_decoder"}:
        fused_sam_logits = getattr(output, "fused_sam_mask_logits", None)
        if fused_sam_logits is None:
            raise RuntimeError(
                "model.primary_mask_source='fused_sam' requires fused_sam_mask_logits. "
                "Enable the fused SAM decoder path."
            )
        output.mask_logits = fused_sam_logits
        return
    if source in {"sam3_direct", "sam3", "sam_video", "sam3_video"}:
        sam3_mask = getattr(output, "sam3_direct_mask", None)
        if sam3_mask is None:
            raise RuntimeError(
                "model.primary_mask_source='sam3_direct' requires sam3_direct_mask. "
                "Enable the SAM3 video tracker path."
            )
        output.mask_logits = mask_to_logits(sam3_mask).unsqueeze(0)
        return
    raise ValueError(
        "model.primary_mask_source must be 'dense', 'fused_sam', or 'sam3_direct', "
        f"got {source!r}"
    )


def mask_to_logits(mask: torch.Tensor, *, eps: float = 1e-4) -> torch.Tensor:
    prob = mask.float().clamp(float(eps), 1.0 - float(eps))
    return torch.logit(prob)


def dense_point_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    if not valid.any():
        return prediction.sum() * 0.0
    return F.smooth_l1_loss(prediction[valid], target[valid])


def dense_chamfer_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    *,
    max_points: int,
) -> torch.Tensor:
    if not valid.any():
        return prediction.sum() * 0.0
    pred_points = prediction[valid]
    target_points = target[valid]
    if pred_points.shape[0] < 2 or target_points.shape[0] < 2:
        return prediction.sum() * 0.0
    pred_points = sample_rows(pred_points, max_points)
    target_points = sample_rows(target_points, max_points)
    distances = torch.cdist(pred_points[None], target_points[None], p=2).squeeze(0)
    return 0.5 * (
        distances.min(dim=1).values.mean() + distances.min(dim=0).values.mean()
    )


def sample_rows(values: torch.Tensor, max_rows: int) -> torch.Tensor:
    max_rows = int(max_rows)
    if max_rows <= 0 or values.shape[0] <= max_rows:
        return values
    indices = torch.randperm(values.shape[0], device=values.device)[:max_rows]
    return values[indices]


def dense_reprojection_mask_loss(
    pointmap: torch.Tensor,
    mask_logits: torch.Tensor,
    target_mask: torch.Tensor,
    intrinsics: torch.Tensor | None,
    world_to_camera: torch.Tensor | None,
) -> torch.Tensor:
    if intrinsics is None or world_to_camera is None:
        return pointmap.sum() * 0.0
    projected = project_weighted_pointmap_mask(
        pointmap,
        weights=mask_logits.sigmoid(),
        intrinsics=intrinsics,
        world_to_camera=world_to_camera,
    )
    target = target_mask.to(projected.dtype)
    logits = torch.logit(projected.clamp(1e-4, 1.0 - 1e-4))
    pos = target.sum()
    neg = target.numel() - pos
    pos_weight = (neg / pos.clamp_min(1.0)).clamp(1.0, 20.0)
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
    dice = probability_dice_loss(projected, target)
    return bce + dice


def project_weighted_pointmap_mask(
    pointmap: torch.Tensor,
    *,
    weights: torch.Tensor,
    intrinsics: torch.Tensor,
    world_to_camera: torch.Tensor,
) -> torch.Tensor:
    height, width = pointmap.shape[:2]
    points = pointmap.reshape(-1, 3)
    weights_flat = weights.reshape(-1).to(points.dtype)
    ones = torch.ones(points.shape[0], 1, dtype=points.dtype, device=points.device)
    points_h = torch.cat([points, ones], dim=-1)
    cam = (world_to_camera.to(points.dtype) @ points_h.transpose(0, 1)).transpose(0, 1)
    z = cam[:, 2].clamp_min(1e-6)
    fx = intrinsics[0, 0].to(points.dtype)
    fy = intrinsics[1, 1].to(points.dtype)
    cx = intrinsics[0, 2].to(points.dtype)
    cy = intrinsics[1, 2].to(points.dtype)
    u = fx * (cam[:, 0] / z) + cx
    v = fy * (cam[:, 1] / z) + cy
    valid = (
        torch.isfinite(u)
        & torch.isfinite(v)
        & torch.isfinite(z)
        & (cam[:, 2] > 1e-6)
        & (weights_flat > 1e-6)
        & (u >= -1.0)
        & (u <= width)
        & (v >= -1.0)
        & (v <= height)
    )
    if not valid.any():
        return weights.sum() * 0.0 + torch.zeros_like(weights)

    u = u[valid]
    v = v[valid]
    source_weight = weights_flat[valid]
    u0 = torch.floor(u)
    v0 = torch.floor(v)
    du = u - u0
    dv = v - v0
    u0 = u0.long()
    v0 = v0.long()

    flat = torch.zeros(height * width, dtype=points.dtype, device=points.device)
    flat = splat_to_flat(
        flat,
        u0,
        v0,
        (1.0 - du) * (1.0 - dv) * source_weight,
        width,
        height,
    )
    flat = splat_to_flat(
        flat,
        u0 + 1,
        v0,
        du * (1.0 - dv) * source_weight,
        width,
        height,
    )
    flat = splat_to_flat(
        flat,
        u0,
        v0 + 1,
        (1.0 - du) * dv * source_weight,
        width,
        height,
    )
    flat = splat_to_flat(
        flat,
        u0 + 1,
        v0 + 1,
        du * dv * source_weight,
        width,
        height,
    )
    return (1.0 - torch.exp(-flat.reshape(height, width))).clamp(0.0, 1.0)


def splat_to_flat(
    flat: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    values: torch.Tensor,
    width: int,
    height: int,
) -> torch.Tensor:
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    if not valid.any():
        return flat
    indices = y[valid] * width + x[valid]
    return flat.scatter_add(0, indices, values[valid])


def probability_dice_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    if not target.any():
        return prediction.mean()
    intersection = (prediction * target).sum()
    union = prediction.sum() + target.sum()
    return 1.0 - (2.0 * intersection + eps) / (union + eps)


def dense_instance_match_loss(
    *,
    current_embeddings: torch.Tensor,
    history_embeddings: torch.Tensor,
    current_instance_ids: torch.Tensor,
    history_instance_ids: torch.Tensor,
    current_valid: torch.Tensor,
    history_valid: torch.Tensor,
    max_pixels: int,
    negative_ratio: int,
    temperature: float,
) -> tuple[torch.Tensor, int]:
    zero = current_embeddings.sum() * 0.0
    current_indices = current_valid.reshape(-1).nonzero(as_tuple=False).flatten()
    history_indices = history_valid.reshape(-1).nonzero(as_tuple=False).flatten()
    if current_indices.numel() == 0 or history_indices.numel() == 0:
        return zero, 0

    current_ids_all = current_instance_ids.reshape(-1)
    history_ids_all = history_instance_ids.reshape(-1)
    current_ids = current_ids_all[current_indices]
    history_ids = history_ids_all[history_indices]
    shared = torch.isin(current_ids, history_ids.unique())
    current_indices = current_indices[shared]
    if current_indices.numel() == 0:
        return zero, 0

    max_pixels = max(1, int(max_pixels))
    if current_indices.numel() > max_pixels:
        perm = torch.randperm(current_indices.numel(), device=current_indices.device)
        current_indices = current_indices[perm[:max_pixels]]
    if history_indices.numel() > max_pixels:
        perm = torch.randperm(history_indices.numel(), device=history_indices.device)
        history_indices = history_indices[perm[:max_pixels]]

    current_flat = current_embeddings.reshape(-1, current_embeddings.shape[-1])
    history_flat = history_embeddings.reshape(-1, history_embeddings.shape[-1])
    current_feat = F.normalize(current_flat[current_indices], dim=-1)
    history_feat = F.normalize(history_flat[history_indices], dim=-1)
    logits = torch.matmul(current_feat, history_feat.transpose(0, 1)) / max(
        float(temperature),
        1e-6,
    )
    current_ids = current_ids_all[current_indices]
    history_ids = history_ids_all[history_indices]
    targets = (current_ids[:, None] == history_ids[None, :]).to(logits.dtype)
    positive = targets > 0.5
    negative = ~positive
    if not positive.any() or not negative.any():
        return zero, 0
    pos_idx = positive.reshape(-1).nonzero(as_tuple=False).flatten()
    neg_idx = negative.reshape(-1).nonzero(as_tuple=False).flatten()
    max_neg = pos_idx.numel() * max(1, int(negative_ratio))
    if neg_idx.numel() > max_neg:
        perm = torch.randperm(neg_idx.numel(), device=neg_idx.device)
        neg_idx = neg_idx[perm[:max_neg]]
    selected = torch.cat([pos_idx, neg_idx], dim=0)
    return (
        F.binary_cross_entropy_with_logits(
            logits.reshape(-1)[selected],
            targets.reshape(-1)[selected],
        ),
        int(current_indices.numel() + history_indices.numel()),
    )


def select_history_update_mask(
    *,
    output: Any,
    target: Dict[str, torch.Tensor],
    source: str,
    pred_threshold: float,
) -> torch.Tensor:
    source = source.strip().lower()
    if source in {"gt", "target", "teacher"}:
        return target["prompt_mask"]
    if source in {"pred", "prediction"}:
        return output.mask_logits[0].detach().sigmoid() > float(pred_threshold)
    if source in {"fused_sam", "sam_decoder"}:
        fused_sam_logits = getattr(output, "fused_sam_mask_logits", None)
        if fused_sam_logits is None:
            raise RuntimeError(
                "history.update_source='fused_sam' requires fused_sam_mask_logits. "
                "Enable the fused SAM decoder path."
            )
        return fused_sam_logits[0].detach().sigmoid() > float(pred_threshold)
    if source in {"sam3_direct", "sam3", "sam_video", "sam3_video"}:
        sam3_mask = getattr(output, "sam3_direct_mask", None)
        if sam3_mask is None:
            raise RuntimeError(
                "history.update_source='sam3_direct' requires sam3_direct_mask. "
                "The SAM3 video tracker path was not enabled."
            )
        return sam3_mask.detach().bool()
    if source in {"gt_or_pred", "teacher_or_pred"}:
        gt = target["prompt_mask"]
        if gt.any():
            return gt
        return output.mask_logits[0].detach().sigmoid() > float(pred_threshold)
    raise ValueError(
        "history.update_source must be 'gt', 'pred', 'gt_or_pred', 'fused_sam', "
        "or 'sam3_direct', "
        f"got {source!r}"
    )


def select_fused_sam_prompt_mask(
    *,
    output: Any,
    target: Dict[str, torch.Tensor],
    source: str,
) -> torch.Tensor | None:
    source = source.strip().lower()
    if source in {"none", "no", "disabled"}:
        return None
    if source in {"pred", "prediction"}:
        return output.mask_logits.detach()
    if source in {"gt", "target", "teacher"}:
        return target["prompt_mask"].float()
    if source in {"gt_or_pred", "teacher_or_pred"}:
        gt = target["prompt_mask"]
        if gt.any():
            return gt.float()
        return output.mask_logits.detach()
    raise ValueError(
        "history.fused_sam_prompt_source must be 'none', 'pred', 'gt', or 'gt_or_pred', "
        f"got {source!r}"
    )


def save_dense_visualization(
    path: Path,
    *,
    sequence: ObjectSequence,
    batch: Dict[str, torch.Tensor],
    outputs: Sequence[Any],
    prompt: str,
    step: int,
    threshold: float,
    primary_mask_source: str,
) -> None:
    frames = len(sequence.image_paths)
    panel_w = 320
    panel_h = int(round(panel_w * batch["prompt_mask"].shape[1] / batch["prompt_mask"].shape[2]))
    title_h = 26
    margin = 6
    primary = primary_mask_source.strip().lower()
    show_fused_sam = primary not in {"fused_sam", "sam_decoder"} and any(
        getattr(output, "fused_sam_mask_logits", None) is not None
        for output in outputs
    )
    show_sam3_direct = any(
        getattr(output, "sam3_direct_mask", None) is not None
        for output in outputs
    )
    show_score = primary in {"dense", "mask_head", "baseline"}
    columns = 3 + int(show_sam3_direct) + int(show_fused_sam) + int(show_score)
    canvas = Image.new(
        "RGB",
        (columns * panel_w + (columns + 1) * margin, title_h + frames * (panel_h + margin) + margin),
        color=(20, 20, 20),
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (margin, 5),
        (
            f"step={step} prompt='{prompt}' threshold={threshold:.2f} "
            f"pred_source={primary_mask_source}"
        ),
        fill=(240, 240, 240),
    )
    pred_heading = (
        "Pred (fused SAM)"
        if primary in {"fused_sam", "sam_decoder"}
        else "Pred mask"
    )
    headings = ["RGB", "GT prompt", pred_heading]
    if show_sam3_direct:
        headings.append("SAM3 original")
    if show_fused_sam:
        headings.append("Fused SAM")
    if show_score:
        headings.append("Pred score")
    for col, heading in enumerate(headings):
        draw.text((margin + col * (panel_w + margin), title_h - 14), heading, fill=(220, 220, 220))

    for frame_idx, output in enumerate(outputs):
        image = load_rgb(sequence.image_paths[frame_idx], batch["prompt_mask"].shape[1:])
        gt = batch["prompt_mask"][frame_idx].detach().cpu().numpy()
        pred = output.mask_logits[0].sigmoid().detach().float().cpu().numpy()
        panels = [
            image,
            overlay_mask(image, gt, (230, 57, 70), threshold=0.5),
            overlay_mask(image, pred, (69, 123, 157), threshold=threshold),
        ]
        sam3_direct_mask = getattr(output, "sam3_direct_mask", None)
        if show_sam3_direct:
            if sam3_direct_mask is None:
                panels.append(image)
            else:
                direct = sam3_direct_mask.detach().float().cpu().numpy()
                panels.append(overlay_mask(image, direct, (255, 183, 3), threshold=0.5))
        fused_sam_logits = getattr(output, "fused_sam_mask_logits", None)
        if show_fused_sam:
            if fused_sam_logits is None:
                panels.append(image)
            else:
                fused_sam = fused_sam_logits[0].sigmoid().detach().float().cpu().numpy()
                panels.append(overlay_mask(image, fused_sam, (42, 157, 143), threshold=threshold))
        if show_score:
            score = output.prompt_score[0].sigmoid().detach().float().cpu().numpy()
            panels.append(heatmap_overlay(image, score))
        row_y = title_h + margin + frame_idx * (panel_h + margin)
        for col, panel in enumerate(panels):
            panel = panel.resize((panel_w, panel_h), Image.BILINEAR)
            canvas.paste(panel, (margin + col * (panel_w + margin), row_y))

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def export_dense_pointclouds(
    output_dir: Path,
    *,
    sequence: ObjectSequence,
    batch: Dict[str, torch.Tensor],
    outputs: Sequence[Any],
    step: int,
    threshold: float,
    max_points: int,
    gt_once: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb = load_sequence_rgb(sequence.image_paths, batch["prompt_mask"].shape[1:])
    gt_mask = batch["prompt_mask"].detach().cpu()
    gt_points = batch["pointmap"].detach().cpu()
    pred_masks = torch.stack(
        [output.mask_logits[0].sigmoid().detach().float().cpu() > threshold for output in outputs],
        dim=0,
    )
    pred_points = torch.stack(
        [output.pointmap[0].detach().float().cpu() for output in outputs],
        dim=0,
    )
    gt_path = (
        output_dir / "gt_object.ply"
        if gt_once
        else output_dir / f"step_{step:06d}_gt_object.ply"
    )
    write_pointcloud_ply(
        gt_path,
        gt_points,
        rgb,
        gt_mask,
        max_points=max_points,
    )
    write_pointcloud_ply(
        output_dir / f"step_{step:06d}_pred_object.ply",
        pred_points,
        rgb,
        pred_masks,
        max_points=max_points,
    )
    write_pointcloud_ply(
        output_dir / f"step_{step:06d}_pred_object_gtmask.ply",
        pred_points,
        rgb,
        gt_mask,
        max_points=max_points,
    )
    stream_pointmaps = [
        getattr(output, "streamvggt_pointmap", None) for output in outputs
    ]
    if all(pointmap is not None for pointmap in stream_pointmaps):
        stream_points = torch.cat(
            [pointmap.detach().float().cpu() for pointmap in stream_pointmaps],
            dim=0,
        )
        write_pointcloud_ply(
            output_dir / f"step_{step:06d}_streamvggt_object_gtmask.ply",
            stream_points,
            rgb,
            gt_mask,
            max_points=max_points,
        )
        write_pointcloud_ply(
            output_dir / f"step_{step:06d}_streamvggt_object_predmask.ply",
            stream_points,
            rgb,
            pred_masks,
            max_points=max_points,
        )
    sam3_direct_masks = [
        getattr(output, "sam3_direct_mask", None) for output in outputs
    ]
    if all(mask is not None for mask in sam3_direct_masks):
        direct_masks = torch.stack(
            [mask.detach().bool().cpu() for mask in sam3_direct_masks],
            dim=0,
        )
        write_pointcloud_ply(
            output_dir / f"step_{step:06d}_pred_object_sam3_direct.ply",
            pred_points,
            rgb,
            direct_masks,
            max_points=max_points,
        )
    fused_sam_logits = [
        getattr(output, "fused_sam_mask_logits", None) for output in outputs
    ]
    if all(logits is not None for logits in fused_sam_logits):
        fused_sam_masks = torch.stack(
            [
                logits[0].sigmoid().detach().float().cpu() > threshold
                for logits in fused_sam_logits
            ],
            dim=0,
        )
        write_pointcloud_ply(
            output_dir / f"step_{step:06d}_pred_object_fused_sam.ply",
            pred_points,
            rgb,
            fused_sam_masks,
            max_points=max_points,
        )


def write_streamvggt_baseline_visualization(
    path: Path,
    *,
    sequence: ObjectSequence,
    batch: Dict[str, torch.Tensor],
    prompt: str,
) -> None:
    frames = len(sequence.image_paths)
    panel_w = 320
    panel_h = int(round(panel_w * batch["prompt_mask"].shape[1] / batch["prompt_mask"].shape[2]))
    title_h = 26
    margin = 6
    columns = 2
    canvas = Image.new(
        "RGB",
        (columns * panel_w + (columns + 1) * margin, title_h + frames * (panel_h + margin) + margin),
        color=(20, 20, 20),
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 5), f"StreamVGGT baseline prompt='{prompt}'", fill=(240, 240, 240))
    headings = ["RGB", "GT mask"]
    for col, heading in enumerate(headings):
        draw.text((margin + col * (panel_w + margin), title_h - 14), heading, fill=(220, 220, 220))

    for frame_idx in range(frames):
        image = load_rgb(sequence.image_paths[frame_idx], batch["prompt_mask"].shape[1:])
        gt = batch["prompt_mask"][frame_idx].detach().cpu().numpy()
        panels = [
            image,
            overlay_mask(image, gt, (230, 57, 70), threshold=0.5),
        ]
        row_y = title_h + margin + frame_idx * (panel_h + margin)
        for col, panel in enumerate(panels):
            panel = panel.resize((panel_w, panel_h), Image.BILINEAR)
            canvas.paste(panel, (margin + col * (panel_w + margin), row_y))

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def load_sequence_rgb(
    image_paths: Sequence[Path],
    output_hw: Sequence[int],
) -> torch.Tensor:
    return torch.stack(
        [
            torch.from_numpy(np.asarray(load_rgb(path, output_hw)).copy()).float()
            / 255.0
            for path in image_paths
        ],
        dim=0,
    )


def load_rgb(path: Path, output_hw: Sequence[int]) -> Image.Image:
    image = Image.open(path).convert("RGB")
    return image.resize((int(output_hw[1]), int(output_hw[0])), Image.BILINEAR)


def overlay_mask(
    image: Image.Image,
    mask: np.ndarray,
    color: tuple[int, int, int],
    *,
    threshold: float,
    alpha: float = 0.55,
) -> Image.Image:
    arr = np.asarray(image).astype(np.float32)
    mask_bool = np.asarray(mask) > threshold
    if mask_bool.any():
        arr[mask_bool] = arr[mask_bool] * (1.0 - alpha) + np.asarray(color, dtype=np.float32) * alpha
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def heatmap_overlay(image: Image.Image, heatmap: np.ndarray) -> Image.Image:
    arr = np.asarray(image).astype(np.float32)
    heat = np.clip(np.asarray(heatmap, dtype=np.float32), 0.0, 1.0)
    color = np.stack([255.0 * heat, 60.0 * (1.0 - heat), 255.0 * (1.0 - heat)], axis=-1)
    out = 0.55 * arr + 0.45 * color
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def write_pointcloud_ply(
    path: Path,
    points: torch.Tensor,
    rgb: torch.Tensor,
    mask: torch.Tensor,
    *,
    max_points: int,
) -> None:
    valid = mask.bool() & torch.isfinite(points).all(dim=-1)
    pts = points[valid]
    colors = rgb[valid]
    if pts.numel() == 0:
        return
    if pts.shape[0] > max_points:
        indices = torch.randperm(pts.shape[0])[:max_points]
        pts = pts[indices]
        colors = colors[indices]
    pts_np = pts.numpy()
    colors_np = (colors.numpy() * 255.0).clip(0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {pts_np.shape[0]}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("end_header\n")
        for point, color in zip(pts_np, colors_np):
            handle.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def mean_metric(values: Sequence[torch.Tensor]) -> float:
    if not values:
        return 0.0
    return float(torch.stack([value.detach() for value in values]).mean().cpu())


def mean_float(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(value) for value in values) / len(values))


def binary_iou_tensor(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = pred.detach().bool()
    target = target.detach().bool()
    union = (pred | target).sum().item()
    if union == 0:
        return 1.0
    return float((pred & target).sum().item() / union)


def write_metrics_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "step",
        "loss",
        "mask_loss",
        "dice_loss",
        "point_loss",
        "streamvggt_point_loss",
        "chamfer_loss",
        "reprojection_loss",
        "fused_sam_mask_loss",
        "fused_sam_dice_loss",
        "sam3_direct_iou",
        "text_loss",
        "aux_cls_loss",
        "match_loss",
        "num_prompt_pixels",
        "num_supervised_pixels",
        "num_point_pixels",
        "num_gt_point_pixels",
        "num_match_pixels",
        "point_valid_source",
        "point_mask_condition",
        "fusion_type",
        "primary_mask_source",
        "use_camera_tokens",
        "train_scope",
        "history_update_source",
        "fused_sam_prompt_source",
        "fused_sam_feature_mode",
        "sam3_direct_selected_obj_id",
        "sam3_direct_prompt_frame_idx",
        "prompt",
        "sampled_instance_id",
        "sampled_label",
        "target_mode",
        "reference_frame_idx",
    ]
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()


def append_metric(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writerow(row)


def save_checkpoint(
    path: Path,
    model: DenseSAMVGGTModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: DenseFusionTrainConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(config),
        },
        path,
    )
