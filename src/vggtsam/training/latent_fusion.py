"""Training loop for SAM3 intermediate token + StreamVGGT latent fusion."""

from __future__ import annotations

import csv
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from vggtsam.adapters.sam3_intermediate import (
    SAM3IntermediateAdapter,
    load_sam3_image_model,
)
from vggtsam.adapters.streamvggt_latent import (
    StreamVGGTLatentAdapter,
    load_streamvggt_latent_model,
)
from vggtsam.data.scannetpp.object_sequence import (
    ObjectSamplingConfig,
    ScanNetPPObjectSequenceDataset,
    keep_instances_visible_in_multiple_frames,
)
from vggtsam.models.latent_fusion import LatentSAMVGGTModel
from vggtsam.training.plotting import plot_training_curves


@dataclass
class LatentFusionTrainConfig:
    manifest: Path
    scene_id: str | None
    sequence_length: int
    frame_stride: int
    min_pixels: int
    max_area_ratio: float
    min_visible_frames: int
    max_objects_per_frame: int
    ignore_instance_id: int
    semantic_ignore_label: int
    excluded_semantic_labels: List[int]
    target_object_labels: List[str]
    excluded_object_labels: List[str]
    min_token_majority: float
    min_tokens_per_instance: int
    max_match_tokens: int
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
    token_grid: tuple[int, int]
    context_grid: tuple[int, int]
    streamvggt_layer_index: int
    streamvggt_image_mode: str
    point_target_source: str
    use_camera_tokens: bool
    d_fuse: int
    num_heads: int
    num_classes: int
    dropout: float
    semantic_weight: float
    point_weight: float
    match_weight: float
    temperature: float
    device: str
    iterations: int
    lr: float
    seed: int
    log_every: int
    save_every: int
    visualize_every: int
    visualize_threshold: float
    output_dir: Path


@dataclass(frozen=True)
class ObjectPromptSelection:
    prompt: str
    target_object_labels: List[str]
    sampled_instance_id: int
    sampled_label: str


def train_latent_fusion(config: LatentFusionTrainConfig) -> None:
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
        token_grid=config.token_grid,
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
        token_grid=config.token_grid,
        context_grid=config.context_grid,
        layer_index=config.streamvggt_layer_index,
        image_mode=config.streamvggt_image_mode,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = config.output_dir / "training_history.csv"
    write_metrics_header(metrics_path)

    model: LatentSAMVGGTModel | None = None
    optimizer: torch.optim.Optimizer | None = None
    completed_steps = 0

    for step in range(1, config.iterations + 1):
        sequence = dataset.sample(rng)
        prompt_selection = select_object_prompt(
            sequence.visible_instance_ids,
            sequence.object_labels,
            rng=rng,
            min_visible_frames=config.min_visible_frames,
            mode=config.sam3_prompt_mode,
            fallback_prompt=config.sam3_prompt,
            target_object_labels=config.target_object_labels,
            excluded_object_labels=config.excluded_object_labels,
        )
        if prompt_selection is None:
            continue
        need_streamvggt_pointmap = should_request_streamvggt_pointmap(
            config.point_target_source,
            has_gt_pointmaps=sequence.pointmaps is not None,
        )
        with torch.no_grad():
            sam_out = sam3.extract_from_paths(
                sequence.image_paths,
                prompt=prompt_selection.prompt,
            )
            geo_out = geometry.extract_from_paths(
                sequence.image_paths,
                return_pointmap=need_streamvggt_pointmap,
            )
            pointmap_grid = resolve_point_targets(
                sequence.pointmaps,
                geo_out.pointmap_grid,
                token_grid=config.token_grid,
                source=config.point_target_source,
            )

        batch = build_latent_batch(
            sequence.instance_masks,
            sequence.semantic_masks,
            sequence.visible_instance_ids,
            sequence.object_labels,
            pointmap_grid=pointmap_grid,
            token_grid=config.token_grid,
            min_visible_frames=config.min_visible_frames,
            ignore_instance_id=config.ignore_instance_id,
            semantic_ignore_label=config.semantic_ignore_label,
            excluded_semantic_labels=config.excluded_semantic_labels,
            target_object_labels=prompt_selection.target_object_labels,
            excluded_object_labels=config.excluded_object_labels,
            min_token_majority=config.min_token_majority,
            min_tokens_per_instance=config.min_tokens_per_instance,
            max_area_ratio=config.max_area_ratio,
            num_classes=config.num_classes,
            device=config.device,
        )
        if batch is None:
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
            model = LatentSAMVGGTModel(
                sam_dim=sam_dim,
                geometry_dim=geometry_dim,
                camera_dim=camera_dim,
                d_fuse=config.d_fuse,
                num_heads=config.num_heads,
                num_classes=config.num_classes,
                dropout=config.dropout,
                token_grid=config.token_grid,
            ).to(config.device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
            print(
                "initialized LatentSAMVGGTModel "
                f"sam_dim={sam_dim} geometry_dim={geometry_dim} "
                f"camera_dim={camera_dim} use_camera_tokens={config.use_camera_tokens}"
            )

        assert optimizer is not None
        output = model(
            sam_tokens=sam_out.semantic.tokens.float(),
            geometry_tokens=geo_out.geometry.tokens.float(),
            camera_tokens=(
                geo_out.geometry.camera_tokens.float()
                if config.use_camera_tokens
                and geo_out.geometry.camera_tokens is not None
                else None
            ),
        )

        valid = batch["valid_tokens"]
        labels = batch["semantic_labels"]
        semantic_loss = F.cross_entropy(output.logits[0, valid], labels[valid])
        point_loss = F.smooth_l1_loss(
            output.pointmap[0, valid],
            batch["point_targets"][valid],
        )

        match_indices = select_match_indices(
            valid,
            batch["instance_ids"],
            batch["frame_ids"],
            max_tokens=config.max_match_tokens,
        )
        match_loss = cross_frame_correspondence_loss(
            output.embeddings[0, match_indices],
            batch["instance_ids"][match_indices],
            batch["frame_ids"][match_indices],
            temperature=config.temperature,
        )
        loss = (
            config.semantic_weight * semantic_loss
            + config.point_weight * point_loss
            + config.match_weight * match_loss
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        completed_steps += 1

        row = {
            "step": step,
            "loss": float(loss.detach().cpu()),
            "semantic_loss": float(semantic_loss.detach().cpu()),
            "point_loss": float(point_loss.detach().cpu()),
            "match_loss": float(match_loss.detach().cpu()),
            "num_tokens": int(valid.sum().item()),
            "num_match_tokens": int(match_indices.numel()),
            "num_instances": int(batch["instance_ids"][valid].unique().numel()),
            "prompt": prompt_selection.prompt,
            "sampled_instance_id": int(prompt_selection.sampled_instance_id),
            "sampled_label": prompt_selection.sampled_label,
        }
        append_metric(metrics_path, row)

        if step % config.log_every == 0 or completed_steps == 1:
            print(
                "step={step} loss={loss:.4f} semantic={semantic_loss:.4f} "
                "point={point_loss:.4f} match={match_loss:.4f} "
                "tokens={num_tokens} match_tokens={num_match_tokens} "
                "instances={num_instances} prompt='{prompt}'".format(**row)
            )
        if config.visualize_every > 0 and (
            step % config.visualize_every == 0 or completed_steps == 1
        ):
            viz_path = config.output_dir / "visualizations" / f"step_{step:06d}.png"
            save_correspondence_visualization(
                viz_path,
                image_paths=sequence.image_paths,
                instance_masks=sequence.instance_masks,
                object_labels=sequence.object_labels,
                embeddings=output.embeddings[0].detach(),
                batch=batch,
                prompt=prompt_selection.prompt,
                preferred_instance_id=prompt_selection.sampled_instance_id,
                step=step,
                threshold=config.visualize_threshold,
                temperature=config.temperature,
            )
            save_correspondence_crop_visualization(
                viz_path.with_name(f"step_{step:06d}_crops.png"),
                image_paths=sequence.image_paths,
                instance_masks=sequence.instance_masks,
                object_labels=sequence.object_labels,
                embeddings=output.embeddings[0].detach(),
                batch=batch,
                prompt=prompt_selection.prompt,
                preferred_instance_id=prompt_selection.sampled_instance_id,
                step=step,
                threshold=config.visualize_threshold,
                temperature=config.temperature,
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
                title="Latent SAM3/StreamVGGT Fusion Training",
            )

    if model is None or optimizer is None:
        raise RuntimeError("No valid training batches were produced.")

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
        title="Latent SAM3/StreamVGGT Fusion Training",
    )
    print(f"training history: {metrics_path}")
    print(f"training curves: {config.output_dir / 'training_curves.png'}")


def build_latent_batch(
    instance_masks: Sequence[np.ndarray],
    semantic_masks: Sequence[np.ndarray],
    visible_instance_ids: Sequence[Sequence[int]],
    object_labels: Dict[int, str],
    *,
    pointmap_grid: torch.Tensor,
    token_grid: tuple[int, int],
    min_visible_frames: int,
    ignore_instance_id: int,
    semantic_ignore_label: int,
    excluded_semantic_labels: Sequence[int],
    target_object_labels: Sequence[str],
    excluded_object_labels: Sequence[str],
    min_token_majority: float,
    min_tokens_per_instance: int,
    max_area_ratio: float,
    num_classes: int,
    device: str,
) -> Dict[str, torch.Tensor] | None:
    token_h, token_w = token_grid
    expected_shape = (len(instance_masks), token_h, token_w, 3)
    if tuple(pointmap_grid.shape) != expected_shape:
        raise ValueError(
            f"pointmap_grid must have shape {expected_shape}, got {tuple(pointmap_grid.shape)}"
        )
    keep_per_frame = keep_instances_visible_in_multiple_frames(
        [list(ids) for ids in visible_instance_ids],
        min_visible_frames=min_visible_frames,
    )
    target_label_filters = normalize_label_filters(target_object_labels)
    excluded = set(int(label) for label in excluded_semantic_labels)
    excluded_label_filters = normalize_label_filters(excluded_object_labels)
    target_instance_ids = [
        instance_id
        for instance_id, label in object_labels.items()
        if label_matches(label, target_label_filters)
    ]
    excluded_instance_ids = [
        instance_id
        for instance_id, label in object_labels.items()
        if label_is_excluded(label, excluded_label_filters)
    ]

    semantic_labels = []
    instance_ids = []
    frame_ids = []
    valid_masks = []
    instance_grids = []
    semantic_grids = []

    for frame_idx, (inst_np, sem_np) in enumerate(zip(instance_masks, semantic_masks)):
        inst_grid, inst_ratio = majority_pool_mask(inst_np, token_grid)
        sem_grid, sem_ratio = majority_pool_mask(sem_np, token_grid)
        instance_grids.append(inst_grid)
        semantic_grids.append(sem_grid)

        valid = inst_grid != ignore_instance_id
        valid &= inst_ratio >= min_token_majority
        valid &= sem_ratio >= min_token_majority
        valid &= sem_grid != semantic_ignore_label
        valid &= (sem_grid >= 0) & (sem_grid < num_classes)
        if excluded:
            valid &= ~np.isin(sem_grid, list(excluded))
        if target_label_filters:
            if target_instance_ids:
                valid &= np.isin(inst_grid, target_instance_ids)
            else:
                valid &= False
        if excluded_instance_ids:
            valid &= ~np.isin(inst_grid, excluded_instance_ids)

        visible = keep_per_frame[frame_idx]
        if visible:
            valid &= np.isin(inst_grid, list(visible))
        else:
            valid &= False

        valid = filter_token_instances(
            valid,
            inst_grid,
            min_tokens_per_instance=min_tokens_per_instance,
            max_area_ratio=max_area_ratio,
        )

        semantic_labels.append(torch.from_numpy(sem_grid.reshape(-1)).long())
        instance_ids.append(torch.from_numpy(inst_grid.reshape(-1)).long())
        frame_ids.append(torch.full((token_h * token_w,), frame_idx, dtype=torch.long))
        valid_masks.append(torch.from_numpy(valid.reshape(-1)).bool())

    labels = torch.cat(semantic_labels, dim=0).to(device)
    instances = torch.cat(instance_ids, dim=0).to(device)
    frames = torch.cat(frame_ids, dim=0).to(device)
    valid = torch.cat(valid_masks, dim=0).to(device)

    points = pointmap_grid.reshape(-1, pointmap_grid.shape[-1]).to(device).float()
    valid &= torch.isfinite(points).all(dim=-1)
    if not valid.any():
        return None

    return {
        "semantic_labels": labels,
        "instance_ids": instances,
        "frame_ids": frames,
        "valid_tokens": valid,
        "point_targets": points,
        "token_grid": token_grid,
    }


def select_object_prompt(
    visible_instance_ids: Sequence[Sequence[int]],
    object_labels: Dict[int, str],
    *,
    rng: random.Random,
    min_visible_frames: int,
    mode: str,
    fallback_prompt: str,
    target_object_labels: Sequence[str],
    excluded_object_labels: Sequence[str],
) -> Optional[ObjectPromptSelection]:
    mode = mode.lower()
    target_filters = normalize_label_filters(target_object_labels)
    excluded_filters = normalize_label_filters(excluded_object_labels)

    if mode in {"fixed", "constant", "static"}:
        target_labels = list(target_object_labels)
        return ObjectPromptSelection(
            prompt=fallback_prompt,
            target_object_labels=target_labels,
            sampled_instance_id=-1,
            sampled_label=fallback_prompt,
        )
    if mode not in {"random_instance", "sample_instance", "dynamic"}:
        raise ValueError(
            "sam3.prompt_mode must be 'random_instance' or 'fixed', "
            f"got {mode!r}"
        )

    keep_per_frame = keep_instances_visible_in_multiple_frames(
        [list(ids) for ids in visible_instance_ids],
        min_visible_frames=min_visible_frames,
    )
    candidate_ids = sorted(
        {
            int(instance_id)
            for frame_ids in keep_per_frame
            for instance_id in frame_ids
            if label_matches(object_labels.get(int(instance_id)), target_filters)
            and not label_is_excluded(
                object_labels.get(int(instance_id)),
                excluded_filters,
            )
        }
    )
    if not candidate_ids:
        return None

    sampled_instance_id = int(rng.choice(candidate_ids))
    sampled_label = object_labels.get(sampled_instance_id)
    if not sampled_label:
        return None
    return ObjectPromptSelection(
        prompt=sampled_label,
        target_object_labels=[sampled_label],
        sampled_instance_id=sampled_instance_id,
        sampled_label=sampled_label,
    )


def normalize_label_filters(labels: Sequence[str]) -> List[str]:
    return [str(label).strip().lower() for label in labels if str(label).strip()]


def label_matches(label: str | None, filters: Sequence[str]) -> bool:
    if not filters:
        return True
    if label is None:
        return False
    normalized = label.strip().lower()
    return any(item in normalized for item in filters)


def label_is_excluded(label: str | None, filters: Sequence[str]) -> bool:
    if label is None or not filters:
        return False
    normalized = label.strip().lower()
    return any(item in normalized for item in filters)


def save_correspondence_visualization(
    path: Path,
    *,
    image_paths: Sequence[Path],
    instance_masks: Sequence[np.ndarray],
    object_labels: Dict[int, str],
    embeddings: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    prompt: str,
    preferred_instance_id: int,
    step: int,
    threshold: float,
    temperature: float,
) -> None:
    from PIL import Image, ImageDraw

    reference = select_propagation_reference(batch, preferred_instance_id)
    if reference is None:
        return
    target_instance_id, ref_frame = reference
    pred_probs = propagate_instance_from_reference(
        embeddings,
        batch,
        target_instance_id=target_instance_id,
        ref_frame=ref_frame,
        temperature=temperature,
    )
    if pred_probs is None:
        return

    frames = len(image_paths)
    first_image = Image.open(image_paths[0]).convert("RGB")
    image_w, image_h = first_image.size
    panel_w = min(640, image_w)
    panel_h = max(1, int(round(image_h * panel_w / max(image_w, 1))))
    title_h = 24
    footer_h = 18
    margin = 6
    columns = 3
    canvas_w = columns * panel_w + (columns + 1) * margin
    canvas_h = title_h + frames * (panel_h + footer_h + margin) + margin
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    label = object_labels.get(int(target_instance_id), "unknown")
    draw.text(
        (margin, 4),
        f"step={step} prompt='{prompt}' id={target_instance_id} label='{label}' "
        f"ref_frame={ref_frame} threshold={threshold:.2f}",
        fill=(240, 240, 240),
    )

    headings = ["RGB", "GT instance", "Propagated mask"]
    for col, heading in enumerate(headings):
        draw.text(
            (margin + col * (panel_w + margin), title_h - 15),
            heading,
            fill=(220, 220, 220),
        )

    gt_color, pred_color = visualization_palette()[:2]

    for frame_idx in range(frames):
        base = Image.open(image_paths[frame_idx]).convert("RGB")
        gt_mask = np.asarray(instance_masks[frame_idx]) == target_instance_id
        gt_overlay = overlay_single_mask(
            base,
            gt_mask,
            color=gt_color,
            threshold=0.5,
        )
        pred_overlay = overlay_single_mask(
            base,
            pred_probs[frame_idx].numpy(),
            color=pred_color,
            threshold=threshold,
        )
        base = base.resize((panel_w, panel_h), Image.BILINEAR)
        gt_overlay = gt_overlay.resize((panel_w, panel_h), Image.NEAREST)
        pred_overlay = pred_overlay.resize((panel_w, panel_h), Image.NEAREST)
        row_y = title_h + margin + frame_idx * (panel_h + footer_h + margin)
        for col, panel in enumerate([base, gt_overlay, pred_overlay]):
            x = margin + col * (panel_w + margin)
            canvas.paste(panel, (x, row_y))
        footer = f"frame {frame_idx}"
        if frame_idx == ref_frame:
            footer += " (reference)"
        draw.text((margin, row_y + panel_h + 2), footer[:160], fill=(220, 220, 220))

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def save_correspondence_crop_visualization(
    path: Path,
    *,
    image_paths: Sequence[Path],
    instance_masks: Sequence[np.ndarray],
    object_labels: Dict[int, str],
    embeddings: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    prompt: str,
    preferred_instance_id: int,
    step: int,
    threshold: float,
    temperature: float,
    crop_size: int = 192,
) -> None:
    from PIL import Image, ImageDraw

    reference = select_propagation_reference(batch, preferred_instance_id)
    if reference is None:
        return
    target_instance_id, ref_frame = reference
    pred_probs = propagate_instance_from_reference(
        embeddings,
        batch,
        target_instance_id=target_instance_id,
        ref_frame=ref_frame,
        temperature=temperature,
    )
    if pred_probs is None:
        return

    gt_color, pred_color = visualization_palette()[:2]
    crop_specs = []
    frames = len(image_paths)
    for frame_idx in range(frames):
        image = Image.open(image_paths[frame_idx]).convert("RGB")
        image_w, image_h = image.size
        gt = np.asarray(instance_masks[frame_idx]) == target_instance_id
        if gt.shape != (image_h, image_w):
            gt = resize_bool_mask(gt, (image_h, image_w))
        pred = (
            resize_probability_mask(pred_probs[frame_idx].numpy(), (image_h, image_w))
            > threshold
        )
        bbox = mask_union_bbox(gt, pred, pad=16)
        if bbox is None:
            continue
        crop_specs.append((frame_idx, bbox))
    if not crop_specs:
        return

    columns = 3
    title_h = 26
    label_h = 18
    margin = 8
    rows = len(crop_specs)
    canvas_w = columns * crop_size + (columns + 1) * margin
    canvas_h = title_h + rows * (crop_size + label_h + margin) + margin
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    label = object_labels.get(int(target_instance_id), "unknown")
    draw.text(
        (margin, 5),
        f"step={step} prompt='{prompt}' id={target_instance_id} label='{label}' "
        f"ref_frame={ref_frame}",
        fill=(240, 240, 240),
    )
    headings = ["RGB crop", "GT crop", "Prop crop"]
    for col, heading in enumerate(headings):
        draw.text(
            (margin + col * (crop_size + margin), title_h - 14),
            heading,
            fill=(220, 220, 220),
        )

    for row_idx, (frame_idx, bbox) in enumerate(crop_specs):
        image = Image.open(image_paths[frame_idx]).convert("RGB")
        gt_mask = np.asarray(instance_masks[frame_idx]) == target_instance_id
        pred_mask = pred_probs[frame_idx].numpy()
        gt_overlay = overlay_single_mask(
            image,
            gt_mask,
            color=gt_color,
            threshold=0.5,
        )
        pred_overlay = overlay_single_mask(
            image,
            pred_mask,
            color=pred_color,
            threshold=threshold,
        )
        panels = [
            image.crop(bbox).resize((crop_size, crop_size), Image.BILINEAR),
            gt_overlay.crop(bbox).resize((crop_size, crop_size), Image.NEAREST),
            pred_overlay.crop(bbox).resize((crop_size, crop_size), Image.NEAREST),
        ]
        y = title_h + margin + row_idx * (crop_size + label_h + margin)
        for col, panel in enumerate(panels):
            x = margin + col * (crop_size + margin)
            canvas.paste(panel, (x, y))
        suffix = " reference" if frame_idx == ref_frame else ""
        draw.text(
            (margin, y + crop_size + 2),
            f"frame={frame_idx}{suffix}",
            fill=(220, 220, 220),
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def select_propagation_reference(
    batch: Dict[str, torch.Tensor],
    preferred_instance_id: int,
) -> tuple[int, int] | None:
    valid = batch["valid_tokens"].detach().cpu()
    instance_ids = batch["instance_ids"].detach().cpu()
    frame_ids = batch["frame_ids"].detach().cpu()
    if not bool(valid.any()):
        return None

    def reference_for(instance_id: int) -> tuple[int, int] | None:
        mask = valid & (instance_ids == int(instance_id))
        if not bool(mask.any()):
            return None
        frames, counts = torch.unique(frame_ids[mask], return_counts=True)
        best = int(torch.argmax(counts).item())
        return int(instance_id), int(frames[best].item())

    if preferred_instance_id > 0:
        preferred = reference_for(preferred_instance_id)
        if preferred is not None:
            return preferred

    best_score: tuple[int, int] | None = None
    best_reference: tuple[int, int] | None = None
    for instance_id in torch.unique(instance_ids[valid]).tolist():
        mask = valid & (instance_ids == int(instance_id))
        frames, counts = torch.unique(frame_ids[mask], return_counts=True)
        score = (int(frames.numel()), int(counts.sum().item()))
        if best_score is None or score > best_score:
            best_score = score
            best = int(torch.argmax(counts).item())
            best_reference = (int(instance_id), int(frames[best].item()))
    return best_reference


def propagate_instance_from_reference(
    embeddings: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    *,
    target_instance_id: int,
    ref_frame: int,
    temperature: float,
) -> torch.Tensor | None:
    valid = batch["valid_tokens"]
    instance_ids = batch["instance_ids"]
    frame_ids = batch["frame_ids"]
    ref_mask = (
        valid
        & (instance_ids == int(target_instance_id))
        & (frame_ids == int(ref_frame))
    )
    if not bool(ref_mask.any()):
        return None

    embeddings = F.normalize(embeddings.float(), dim=-1)
    prototype = F.normalize(embeddings[ref_mask].mean(dim=0), dim=0)
    logits = embeddings @ prototype / max(float(temperature), 1e-6)
    probs = torch.sigmoid(logits)
    probs = probs * valid.to(probs.dtype)

    frames = int(frame_ids.max().item()) + 1
    if "token_grid" in batch:
        token_h, token_w = tuple(int(v) for v in batch["token_grid"])
    else:
        tokens_per_frame = int(probs.numel() // max(frames, 1))
        token_h = int(round(math.sqrt(tokens_per_frame)))
        if token_h * token_h != tokens_per_frame:
            raise ValueError(
                "Cannot infer token grid from a non-square token count; pass token_grid in batch."
            )
        token_w = token_h
    return probs.detach().cpu().reshape(frames, token_h, token_w)


def overlay_single_mask(
    image,
    mask: np.ndarray,
    *,
    color: tuple[int, int, int],
    threshold: float,
    alpha: float = 0.55,
):
    from PIL import Image

    arr = np.asarray(image).astype(np.float32)
    image_h, image_w = arr.shape[:2]
    mask = np.asarray(mask)
    if mask.shape != (image_h, image_w):
        if mask.dtype == np.bool_:
            mask = resize_bool_mask(mask, (image_h, image_w))
        else:
            mask = resize_probability_mask(mask, (image_h, image_w))
    mask = mask > threshold
    if not mask.any():
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    color_arr = np.asarray(color, dtype=np.float32)
    arr[mask] = arr[mask] * (1.0 - alpha) + color_arr * alpha
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def resize_probability_mask(mask: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    mask = np.asarray(mask, dtype=np.float32)
    image = Image.fromarray(np.clip(mask * 255.0, 0, 255).astype(np.uint8))
    image = image.resize((out_hw[1], out_hw[0]), Image.BILINEAR)
    return np.asarray(image).astype(np.float32) / 255.0


def resize_bool_mask(mask: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    from PIL import Image

    image = Image.fromarray(mask.astype(np.uint8) * 255)
    image = image.resize((out_hw[1], out_hw[0]), Image.NEAREST)
    return np.asarray(image) > 0


def mask_union_bbox(
    gt: np.ndarray,
    pred: np.ndarray,
    *,
    pad: int,
) -> tuple[int, int, int, int] | None:
    union = gt | pred
    if not union.any():
        union = gt
    if not union.any():
        return None
    ys, xs = np.where(union)
    h, w = union.shape
    x0 = max(int(xs.min()) - pad, 0)
    y0 = max(int(ys.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, w)
    y1 = min(int(ys.max()) + pad + 1, h)
    side = max(x1 - x0, y1 - y0, 1)
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2
    x0 = max(cx - side // 2, 0)
    y0 = max(cy - side // 2, 0)
    x1 = min(x0 + side, w)
    y1 = min(y0 + side, h)
    x0 = max(x1 - side, 0)
    y0 = max(y1 - side, 0)
    return x0, y0, x1, y1


def visualization_palette() -> List[tuple[int, int, int]]:
    return [
        (230, 57, 70),
        (69, 123, 157),
        (42, 157, 143),
        (244, 162, 97),
        (131, 56, 236),
        (255, 190, 11),
        (58, 134, 255),
        (6, 214, 160),
        (255, 0, 110),
        (138, 201, 38),
    ]


def resolve_point_targets(
    gt_pointmaps: Sequence[np.ndarray] | None,
    streamvggt_pointmap_grid: torch.Tensor | None,
    *,
    token_grid: tuple[int, int],
    source: str,
) -> torch.Tensor:
    source = source.lower()
    if source in {"gt", "colmap", "mesh"}:
        if gt_pointmaps is None:
            raise RuntimeError(
                "point_target_source is set to 'gt', but the selected frames do not "
                "have pointmap entries. Re-run scripts/prepare_scannetpp_2d.py with "
                "--save-pointmaps or set geometry.point_target_source to 'streamvggt' "
                "for the old pseudo-target baseline."
            )
        return torch.from_numpy(pool_pointmaps_to_grid(gt_pointmaps, token_grid))
    if source in {"streamvggt", "pseudo"}:
        if streamvggt_pointmap_grid is None:
            raise RuntimeError("StreamVGGT did not return a pointmap_grid.")
        return streamvggt_pointmap_grid.detach().cpu()
    if source == "auto":
        if gt_pointmaps is not None:
            return torch.from_numpy(pool_pointmaps_to_grid(gt_pointmaps, token_grid))
        if streamvggt_pointmap_grid is not None:
            return streamvggt_pointmap_grid.detach().cpu()
        raise RuntimeError(
            "No point targets are available: missing dataset pointmaps and "
            "StreamVGGT pointmap_grid."
        )
    raise ValueError(
        "geometry.point_target_source must be one of 'gt', 'streamvggt', or 'auto', "
        f"got {source!r}"
    )


def should_request_streamvggt_pointmap(
    source: str,
    *,
    has_gt_pointmaps: bool,
) -> bool:
    source = source.lower()
    if source in {"streamvggt", "pseudo"}:
        return True
    if source == "auto":
        return not has_gt_pointmaps
    return False


def pool_pointmaps_to_grid(
    pointmaps: Sequence[np.ndarray],
    out_hw: tuple[int, int],
) -> np.ndarray:
    out_h, out_w = out_hw
    pooled = np.full((len(pointmaps), out_h, out_w, 3), np.nan, dtype=np.float32)
    for frame_idx, pointmap in enumerate(pointmaps):
        pointmap = np.asarray(pointmap, dtype=np.float32)
        if pointmap.ndim != 3 or pointmap.shape[-1] != 3:
            raise ValueError(
                f"Pointmap must have shape [H, W, 3], got {pointmap.shape}"
            )
        src_h, src_w = pointmap.shape[:2]
        for y in range(out_h):
            y0 = int(math.floor(y * src_h / out_h))
            y1 = int(math.floor((y + 1) * src_h / out_h))
            y1 = max(y1, y0 + 1)
            for x in range(out_w):
                x0 = int(math.floor(x * src_w / out_w))
                x1 = int(math.floor((x + 1) * src_w / out_w))
                x1 = max(x1, x0 + 1)
                patch = pointmap[y0:y1, x0:x1]
                valid = np.isfinite(patch).all(axis=-1)
                if valid.any():
                    pooled[frame_idx, y, x] = patch[valid].mean(axis=0)
    return pooled


def majority_pool_mask(
    mask: np.ndarray,
    out_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    src_h, src_w = mask.shape[:2]
    out_h, out_w = out_hw
    labels = np.zeros((out_h, out_w), dtype=np.int64)
    ratios = np.zeros((out_h, out_w), dtype=np.float32)

    for y in range(out_h):
        y0 = int(math.floor(y * src_h / out_h))
        y1 = int(math.floor((y + 1) * src_h / out_h))
        y1 = max(y1, y0 + 1)
        for x in range(out_w):
            x0 = int(math.floor(x * src_w / out_w))
            x1 = int(math.floor((x + 1) * src_w / out_w))
            x1 = max(x1, x0 + 1)
            patch = mask[y0:y1, x0:x1].reshape(-1)
            values, counts = np.unique(patch, return_counts=True)
            best = int(counts.argmax())
            labels[y, x] = int(values[best])
            ratios[y, x] = float(counts[best]) / float(max(patch.size, 1))
    return labels, ratios


def filter_token_instances(
    valid: np.ndarray,
    instance_grid: np.ndarray,
    *,
    min_tokens_per_instance: int,
    max_area_ratio: float,
) -> np.ndarray:
    if not valid.any():
        return valid
    total = float(valid.size)
    out = valid.copy()
    ids, counts = np.unique(instance_grid[valid], return_counts=True)
    for instance_id, count in zip(ids, counts):
        if count < min_tokens_per_instance:
            out[instance_grid == instance_id] = False
        if count / total > max_area_ratio:
            out[instance_grid == instance_id] = False
    return out


def select_match_indices(
    valid: torch.Tensor,
    instance_ids: torch.Tensor,
    frame_ids: torch.Tensor,
    *,
    max_tokens: int,
) -> torch.Tensor:
    indices = valid.nonzero(as_tuple=False).flatten()
    if indices.numel() <= 1:
        return indices

    usable = []
    for instance_id in instance_ids[indices].unique():
        inst_indices = indices[instance_ids[indices] == instance_id]
        if frame_ids[inst_indices].unique().numel() >= 2:
            usable.append(inst_indices)
    if not usable:
        return indices[:0]
    indices = torch.cat(usable, dim=0)
    if indices.numel() > max_tokens:
        perm = torch.randperm(indices.numel(), device=indices.device)[:max_tokens]
        indices = indices[perm]
    return indices


def cross_frame_correspondence_loss(
    embeddings: torch.Tensor,
    instance_ids: torch.Tensor,
    frame_ids: torch.Tensor,
    *,
    temperature: float,
    negative_ratio: int = 8,
) -> torch.Tensor:
    n = embeddings.shape[0]
    if n <= 1:
        return embeddings.sum() * 0.0
    embeddings = F.normalize(embeddings, dim=-1)
    logits = embeddings @ embeddings.T / max(float(temperature), 1e-6)
    eye = torch.eye(n, dtype=torch.bool, device=embeddings.device)
    positive = (instance_ids[:, None] == instance_ids[None, :]) & (
        frame_ids[:, None] != frame_ids[None, :]
    )
    positive &= ~eye
    if not positive.any():
        return embeddings.sum() * 0.0
    cross_frame = frame_ids[:, None] != frame_ids[None, :]
    negative = cross_frame & ~positive

    flat_logits = logits.reshape(-1)
    positive_indices = positive.reshape(-1).nonzero(as_tuple=False).flatten()
    negative_indices = negative.reshape(-1).nonzero(as_tuple=False).flatten()
    if negative_indices.numel() > positive_indices.numel() * negative_ratio:
        choice = torch.randperm(
            negative_indices.numel(),
            device=negative_indices.device,
        )[: positive_indices.numel() * negative_ratio]
        negative_indices = negative_indices[choice]

    selected_indices = torch.cat([positive_indices, negative_indices], dim=0)
    targets = torch.cat(
        [
            torch.ones_like(positive_indices, dtype=flat_logits.dtype),
            torch.zeros_like(negative_indices, dtype=flat_logits.dtype),
        ],
        dim=0,
    )
    return F.binary_cross_entropy_with_logits(flat_logits[selected_indices], targets)


def write_metrics_header(path: Path) -> None:
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "step",
                "loss",
                "semantic_loss",
                "point_loss",
                "match_loss",
                "num_tokens",
                "num_match_tokens",
                "num_instances",
                "prompt",
                "sampled_instance_id",
                "sampled_label",
            ],
        )
        writer.writeheader()


def append_metric(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writerow(row)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: LatentFusionTrainConfig,
) -> None:
    payload = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
    }
    torch.save(payload, path)
