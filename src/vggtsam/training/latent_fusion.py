"""Training loop for SAM3 intermediate token + StreamVGGT latent fusion."""

from __future__ import annotations

import csv
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

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
    min_token_majority: float
    min_tokens_per_instance: int
    max_match_tokens: int
    sam3_repo: Path
    sam3_checkpoint: Path
    sam3_prompt: str
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
    output_dir: Path


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
        need_streamvggt_pointmap = should_request_streamvggt_pointmap(
            config.point_target_source,
            has_gt_pointmaps=sequence.pointmaps is not None,
        )
        with torch.no_grad():
            sam_out = sam3.extract_from_paths(
                sequence.image_paths,
                prompt=config.sam3_prompt,
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
            pointmap_grid=pointmap_grid,
            token_grid=config.token_grid,
            min_visible_frames=config.min_visible_frames,
            ignore_instance_id=config.ignore_instance_id,
            semantic_ignore_label=config.semantic_ignore_label,
            excluded_semantic_labels=config.excluded_semantic_labels,
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
                if geo_out.geometry.camera_tokens is not None
                else 9
            )
            model = LatentSAMVGGTModel(
                sam_dim=sam_dim,
                geometry_dim=geometry_dim,
                camera_dim=camera_dim,
                d_fuse=config.d_fuse,
                num_heads=config.num_heads,
                num_classes=config.num_classes,
                dropout=config.dropout,
            ).to(config.device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
            print(
                "initialized LatentSAMVGGTModel "
                f"sam_dim={sam_dim} geometry_dim={geometry_dim} camera_dim={camera_dim}"
            )

        assert optimizer is not None
        output = model(
            sam_tokens=sam_out.semantic.tokens.float(),
            geometry_tokens=geo_out.geometry.tokens.float(),
            camera_tokens=(
                geo_out.geometry.camera_tokens.float()
                if geo_out.geometry.camera_tokens is not None
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
        match_loss = cross_frame_contrastive_loss(
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
        }
        append_metric(metrics_path, row)

        if step % config.log_every == 0 or completed_steps == 1:
            print(
                "step={step} loss={loss:.4f} semantic={semantic_loss:.4f} "
                "point={point_loss:.4f} match={match_loss:.4f} "
                "tokens={num_tokens} match_tokens={num_match_tokens} "
                "instances={num_instances}".format(**row)
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
    *,
    pointmap_grid: torch.Tensor,
    token_grid: tuple[int, int],
    min_visible_frames: int,
    ignore_instance_id: int,
    semantic_ignore_label: int,
    excluded_semantic_labels: Sequence[int],
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
    excluded = set(int(label) for label in excluded_semantic_labels)

    semantic_labels = []
    instance_ids = []
    frame_ids = []
    valid_masks = []

    for frame_idx, (inst_np, sem_np) in enumerate(zip(instance_masks, semantic_masks)):
        inst_grid, inst_ratio = majority_pool_mask(inst_np, token_grid)
        sem_grid, sem_ratio = majority_pool_mask(sem_np, token_grid)

        valid = inst_grid != ignore_instance_id
        valid &= inst_ratio >= min_token_majority
        valid &= sem_ratio >= min_token_majority
        valid &= sem_grid != semantic_ignore_label
        valid &= (sem_grid >= 0) & (sem_grid < num_classes)
        if excluded:
            valid &= ~np.isin(sem_grid, list(excluded))

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
    }


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


def cross_frame_contrastive_loss(
    embeddings: torch.Tensor,
    instance_ids: torch.Tensor,
    frame_ids: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    n = embeddings.shape[0]
    if n <= 1:
        return embeddings.sum() * 0.0
    embeddings = F.normalize(embeddings, dim=-1)
    logits = embeddings @ embeddings.T / temperature
    eye = torch.eye(n, dtype=torch.bool, device=embeddings.device)
    positive = (instance_ids[:, None] == instance_ids[None, :]) & (
        frame_ids[:, None] != frame_ids[None, :]
    )
    positive &= ~eye
    if not positive.any():
        return embeddings.sum() * 0.0
    logits = logits.masked_fill(eye, -1e9)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    per_anchor = -(log_prob * positive.float()).sum(dim=1) / positive.float().sum(
        dim=1
    ).clamp_min(1.0)
    valid_anchor = positive.any(dim=1)
    return per_anchor[valid_anchor].mean()


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
