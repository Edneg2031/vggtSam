"""Dense image-grid SAM3/StreamVGGT fusion training."""

from __future__ import annotations

import csv
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from vggtsam.adapters.sam3_intermediate import (
    SAM3IntermediateAdapter,
    load_sam3_image_model,
    pool_language_features,
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
    streamvggt_repo: Path
    streamvggt_checkpoint: Path
    feature_grid: tuple[int, int]
    context_grid: tuple[int, int]
    streamvggt_layer_index: int
    streamvggt_image_mode: str
    use_camera_tokens: bool
    output_size: tuple[int, int]
    d_fuse: int
    num_heads: int
    embedding_dim: int
    num_classes: int
    dropout: float
    mask_weight: float
    dice_weight: float
    point_weight: float
    text_weight: float
    aux_cls_weight: float
    match_weight: float
    temperature: float
    max_match_pixels: int
    negative_ratio: int
    device: str
    iterations: int
    lr: float
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


def train_dense_fusion(config: DenseFusionTrainConfig) -> None:
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

    sam3_model = load_sam3_image_model(
        repo_path=config.sam3_repo,
        checkpoint_path=config.sam3_checkpoint,
        device=config.device,
        enable_inst_interactivity=config.sam3_enable_inst_interactivity,
    )
    sam3_model.requires_grad_(False)
    sam3 = SAM3IntermediateAdapter(
        sam3_model,
        device=config.device,
        resolution=config.sam3_resolution,
        source=config.sam3_feature_source,
        text_conditioning=config.sam3_text_conditioning,
        token_grid=config.feature_grid,
    )

    streamvggt_model = load_streamvggt_latent_model(
        repo_path=config.streamvggt_repo,
        checkpoint_path=config.streamvggt_checkpoint,
        device=config.device,
        strict=True,
    )
    streamvggt_model.requires_grad_(False)
    geometry = StreamVGGTLatentAdapter(
        streamvggt_model,
        device=config.device,
        token_grid=config.feature_grid,
        context_grid=config.context_grid,
        layer_index=config.streamvggt_layer_index,
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
        del preview_batch
        print(
            "overfit target "
            f"scene={overfit_sequence.scene_id} "
            f"frames={overfit_sequence.frame_indices} "
            f"instance={overfit_prompt_selection.sampled_instance_id} "
            f"label='{overfit_prompt_selection.sampled_label}' "
            f"target_mode={config.target_mode}"
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

        with torch.no_grad():
            sam_out = sam3.extract_from_paths(
                sequence.image_paths,
                prompt=prompt_selection.prompt,
            )
            geo_out = geometry.extract_from_paths(
                sequence.image_paths,
                return_pointmap=False,
            )
            text_embedding = pool_language_features(sam_out.text_out)
            if text_embedding is None:
                raise RuntimeError("SAM3 did not return language_features for text alignment.")
            text_embedding = text_embedding.to(config.device).float()

        batch = build_dense_batch(
            sequence,
            prompt_selection,
            config=config,
            device=config.device,
        )
        if batch is None:
            if config.overfit:
                raise RuntimeError(
                    "Overfit target produced no prompt_mask. Try a different "
                    "--window-index / --instance-id, or lower min_pixels / "
                    "min_visible_frames."
                )
            continue

        if model is None:
            sam_dim = int(sam_out.semantic.tokens.shape[-1])
            geometry_dim = int(geo_out.geometry.tokens.shape[-1])
            camera_dim = (
                int(geo_out.geometry.camera_tokens.shape[-1])
                if config.use_camera_tokens
                and geo_out.geometry.camera_tokens is not None
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
            ).to(config.device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
            print(
                "initialized DenseSAMVGGTModel "
                f"sam_dim={sam_dim} geometry_dim={geometry_dim} "
                f"text_dim={int(text_embedding.shape[-1])} camera_dim={camera_dim} "
                f"output_size={config.output_size}"
            )

        assert model is not None and optimizer is not None
        num_frames = len(sequence.image_paths)
        feature_tokens_per_frame = config.feature_grid[0] * config.feature_grid[1]
        context_tokens_per_frame = config.context_grid[0] * config.context_grid[1]
        sam_frame_tokens = split_sequence_tokens(
            sam_out.semantic.tokens.float(),
            num_frames=num_frames,
            tokens_per_frame=feature_tokens_per_frame,
            name="sam_tokens",
        )
        geometry_frame_tokens = split_sequence_tokens(
            geo_out.geometry.tokens.float(),
            num_frames=num_frames,
            tokens_per_frame=context_tokens_per_frame,
            name="geometry_tokens",
        )
        camera_tokens = (
            geo_out.geometry.camera_tokens.float()
            if config.use_camera_tokens and geo_out.geometry.camera_tokens is not None
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
        text_losses: List[torch.Tensor] = []
        aux_losses: List[torch.Tensor] = []
        match_losses: List[torch.Tensor] = []
        history_buffer: List[Dict[str, torch.Tensor]] = []
        outputs = []
        selected_match_pixels = 0

        for frame_idx in range(num_frames):
            output = model.decode(
                fused_tokens=fused_frame_tokens[frame_idx],
                text_embedding=text_embedding,
                object_query=object_query,
            )
            outputs.append(output)
            target = frame_target(batch, frame_idx)
            zero = output.mask_logits.sum() * 0.0

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
            point_loss = dense_point_loss(
                output.pointmap[0],
                target["pointmap"],
                target["point_valid"] & target["prompt_mask"],
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
                + config.text_weight * text_loss
                + config.aux_cls_weight * aux_loss
                + config.match_weight * match_loss
            )
            frame_losses.append(frame_loss)
            mask_losses.append(mask_loss)
            dice_losses.append(dice_loss)
            point_losses.append(point_loss)
            text_losses.append(text_loss)
            aux_losses.append(aux_loss)
            match_losses.append(match_loss)
            selected_match_pixels += match_pixels
            history_buffer.append(
                {
                    "embeddings": output.instance_embedding[0].detach(),
                    "instance": target["instance"].detach(),
                    "valid": target["instance_valid"].detach(),
                }
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
            "text_loss": mean_metric(text_losses),
            "aux_cls_loss": mean_metric(aux_losses),
            "match_loss": mean_metric(match_losses),
            "num_prompt_pixels": int(batch["prompt_mask"].sum().item()),
            "num_supervised_pixels": int(batch["mask_supervision"].sum().item()),
            "num_point_pixels": int(
                (batch["prompt_mask"] & batch["point_valid"]).sum().item()
            ),
            "num_match_pixels": int(selected_match_pixels),
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
                "dice={dice_loss:.4f} point={point_loss:.4f} text={text_loss:.4f} "
                "match={match_loss:.4f} prompt_pixels={num_prompt_pixels} "
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
            )
            export_dense_pointclouds(
                config.output_dir / "pointclouds",
                sequence=sequence,
                batch=batch,
                outputs=outputs,
                step=step,
                threshold=config.visualize_threshold,
                max_points=config.max_visual_points,
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

    for frame_idx, (inst_np, sem_np, point_np) in enumerate(
        zip(sequence.instance_masks, sequence.semantic_masks, sequence.pointmaps or [])
    ):
        inst = resize_label_mask(inst_np, output_size)
        sem = resize_label_mask(sem_np, output_size)
        point, point_valid = resize_pointmap(point_np, output_size)

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

    return {
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


def dense_point_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    if not valid.any():
        return prediction.sum() * 0.0
    return F.smooth_l1_loss(prediction[valid], target[valid])


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


def save_dense_visualization(
    path: Path,
    *,
    sequence: ObjectSequence,
    batch: Dict[str, torch.Tensor],
    outputs: Sequence[Any],
    prompt: str,
    step: int,
    threshold: float,
) -> None:
    frames = len(sequence.image_paths)
    panel_w = 320
    panel_h = int(round(panel_w * batch["prompt_mask"].shape[1] / batch["prompt_mask"].shape[2]))
    title_h = 26
    margin = 6
    columns = 4
    canvas = Image.new(
        "RGB",
        (columns * panel_w + (columns + 1) * margin, title_h + frames * (panel_h + margin) + margin),
        color=(20, 20, 20),
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 5), f"step={step} prompt='{prompt}' threshold={threshold:.2f}", fill=(240, 240, 240))
    headings = ["RGB", "GT prompt", "Pred mask", "Pred score"]
    for col, heading in enumerate(headings):
        draw.text((margin + col * (panel_w + margin), title_h - 14), heading, fill=(220, 220, 220))

    for frame_idx, output in enumerate(outputs):
        image = load_rgb(sequence.image_paths[frame_idx], batch["prompt_mask"].shape[1:])
        gt = batch["prompt_mask"][frame_idx].detach().cpu().numpy()
        pred = output.mask_logits[0].sigmoid().detach().cpu().numpy()
        score = output.prompt_score[0].sigmoid().detach().cpu().numpy()
        panels = [
            image,
            overlay_mask(image, gt, (230, 57, 70), threshold=0.5),
            overlay_mask(image, pred, (69, 123, 157), threshold=threshold),
            heatmap_overlay(image, score),
        ]
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
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb = torch.stack(
        [
            torch.from_numpy(
                np.asarray(load_rgb(path, batch["prompt_mask"].shape[1:])).copy()
            ).float()
            / 255.0
            for path in sequence.image_paths
        ],
        dim=0,
    )
    gt_mask = batch["prompt_mask"].detach().cpu()
    gt_points = batch["pointmap"].detach().cpu()
    pred_masks = torch.stack(
        [output.mask_logits[0].sigmoid().detach().cpu() > threshold for output in outputs],
        dim=0,
    )
    pred_points = torch.stack(
        [output.pointmap[0].detach().cpu() for output in outputs],
        dim=0,
    )
    write_pointcloud_ply(
        output_dir / f"step_{step:06d}_gt_object.ply",
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


def write_metrics_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "step",
        "loss",
        "mask_loss",
        "dice_loss",
        "point_loss",
        "text_loss",
        "aux_cls_loss",
        "match_loss",
        "num_prompt_pixels",
        "num_supervised_pixels",
        "num_point_pixels",
        "num_match_pixels",
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
