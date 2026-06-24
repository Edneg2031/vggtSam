"""Training loop for object-level ScanNet++ fusion."""

from __future__ import annotations

import csv
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from vggtsam.adapters.vggt import load_streamvggt_model, run_streamvggt_inference
from vggtsam.data.scannetpp.object_sequence import (
    ObjectSamplingConfig,
    ScanNetPPObjectSequenceDataset,
    keep_instances_visible_in_multiple_frames,
)
from vggtsam.models.object_fusion import ObjectFusionModel


@dataclass
class ObjectFusionTrainConfig:
    manifest: Path
    scene_id: str | None
    sequence_length: int
    frame_stride: int
    min_pixels: int
    max_area_ratio: float
    min_visible_frames: int
    max_objects_per_frame: int
    semantic_ignore_label: int
    streamvggt_repo: Path
    streamvggt_checkpoint: Path
    token_grid: tuple[int, int]
    d_fuse: int
    num_heads: int
    num_classes: int
    semantic_weight: float
    centroid_weight: float
    contrastive_weight: float
    temperature: float
    device: str
    iterations: int
    lr: float
    seed: int
    log_every: int
    save_every: int
    output_dir: Path


def train_object_fusion(config: ObjectFusionTrainConfig) -> None:
    rng = random.Random(config.seed)
    torch.manual_seed(config.seed)

    object_config = ObjectSamplingConfig(
        min_pixels=config.min_pixels,
        max_area_ratio=config.max_area_ratio,
        min_visible_frames=config.min_visible_frames,
        max_objects_per_frame=config.max_objects_per_frame,
        semantic_ignore_label=config.semantic_ignore_label,
    )
    dataset = ScanNetPPObjectSequenceDataset(
        config.manifest,
        scene_id=config.scene_id,
        sequence_length=config.sequence_length,
        frame_stride=config.frame_stride,
        object_config=object_config,
    )

    geometry_model = load_streamvggt_model(
        repo_path=config.streamvggt_repo,
        checkpoint_path=config.streamvggt_checkpoint,
        device=config.device,
        strict=True,
    )
    geometry_model.requires_grad_(False)

    geometry_dim = 6
    object_dim = 9
    model = ObjectFusionModel(
        geometry_dim=geometry_dim,
        object_dim=object_dim,
        camera_dim=9,
        d_fuse=config.d_fuse,
        num_heads=config.num_heads,
        num_classes=config.num_classes,
    ).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = config.output_dir / "training_history.csv"
    with metrics_path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "step",
                "loss",
                "semantic_loss",
                "centroid_loss",
                "contrastive_loss",
                "num_objects",
            ],
        )
        writer.writeheader()

    for step in range(1, config.iterations + 1):
        sequence = dataset.sample(rng)
        with torch.no_grad():
            geometry_output = run_streamvggt_inference(
                geometry_model,
                sequence.image_paths,
                device=config.device,
            )

        batch = build_object_batch(
            geometry_output,
            sequence.instance_masks,
            sequence.semantic_masks,
            sequence.visible_instance_ids,
            token_grid=config.token_grid,
            min_visible_frames=config.min_visible_frames,
            semantic_ignore_label=config.semantic_ignore_label,
            num_classes=config.num_classes,
            device=config.device,
        )
        if batch is None:
            continue

        output = model(
            geometry_tokens=batch["geometry_tokens"],
            object_tokens=batch["object_tokens"],
            camera_tokens=batch["camera_tokens"],
        )

        labels = batch["semantic_labels"]
        valid_semantic = (labels >= 0) & (labels < config.num_classes)
        if valid_semantic.any():
            semantic_loss = F.cross_entropy(
                output.logits[0, valid_semantic], labels[valid_semantic]
            )
        else:
            semantic_loss = output.logits.sum() * 0.0

        centroid_loss = F.smooth_l1_loss(
            output.centroids_3d[0], batch["centroids_3d"]
        )
        contrastive_loss = supervised_contrastive_loss(
            output.embeddings[0],
            batch["instance_ids"],
            temperature=config.temperature,
        )
        loss = (
            config.semantic_weight * semantic_loss
            + config.centroid_weight * centroid_loss
            + config.contrastive_weight * contrastive_loss
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        row = {
            "step": step,
            "loss": float(loss.detach().cpu()),
            "semantic_loss": float(semantic_loss.detach().cpu()),
            "centroid_loss": float(centroid_loss.detach().cpu()),
            "contrastive_loss": float(contrastive_loss.detach().cpu()),
            "num_objects": int(batch["instance_ids"].numel()),
        }
        append_metric(metrics_path, row)
        if step % config.log_every == 0 or step == 1:
            print(
                "step={step} loss={loss:.4f} semantic={semantic_loss:.4f} "
                "centroid={centroid_loss:.4f} contrastive={contrastive_loss:.4f} "
                "objects={num_objects}".format(**row)
            )
        if step % config.save_every == 0:
            save_checkpoint(config.output_dir / f"ckpt_step{step:06d}.pt", model, optimizer, step, config)

    save_checkpoint(config.output_dir / "ckpt_last.pt", model, optimizer, config.iterations, config)
    print(f"training history: {metrics_path}")


def build_object_batch(
    geometry_output: Any,
    instance_masks: Sequence[np.ndarray],
    semantic_masks: Sequence[np.ndarray],
    visible_instance_ids: Sequence[Sequence[int]],
    *,
    token_grid: tuple[int, int],
    min_visible_frames: int,
    semantic_ignore_label: int,
    num_classes: int,
    device: str,
) -> Dict[str, torch.Tensor] | None:
    keep_per_frame = keep_instances_visible_in_multiple_frames(
        [list(ids) for ids in visible_instance_ids],
        min_visible_frames=min_visible_frames,
    )

    geometry_maps = []
    camera_tokens = []
    object_tokens = []
    centroids_3d = []
    semantic_labels = []
    instance_ids = []

    for frame_idx, res in enumerate(geometry_output.ress):
        points = res["pts3d_in_other_view"].float().to(device)  # [1, H, W, 3]
        depth = res["depth"].float().to(device)  # [1, H, W, 1]
        conf = res["conf"].float().to(device).unsqueeze(-1)
        depth_conf = res["depth_conf"].float().to(device).unsqueeze(-1)
        camera = res["camera_pose"].float().to(device)  # [1, 9]
        geom = torch.cat([points, depth, conf, depth_conf], dim=-1)[0]  # [H, W, 6]
        geometry_maps.append(geom)
        camera_tokens.append(camera)

        h, w = geom.shape[:2]
        inst = resize_mask_nearest(instance_masks[frame_idx], (h, w), device=device)
        sem = resize_mask_nearest(semantic_masks[frame_idx], (h, w), device=device)
        for instance_id in sorted(keep_per_frame[frame_idx]):
            mask = inst == int(instance_id)
            if not mask.any():
                continue
            masked_geom = geom[mask]
            point_values = points[0][mask]
            object_feature = masked_geom.mean(dim=0)
            centroid_2d = mask.nonzero().float().mean(dim=0)
            centroid_2d = torch.stack(
                [centroid_2d[1] / max(w - 1, 1), centroid_2d[0] / max(h - 1, 1)]
            )
            area = mask.float().mean().view(1)
            object_tokens.append(torch.cat([object_feature, centroid_2d, area], dim=0))
            centroids_3d.append(point_values.mean(dim=0))
            semantic_labels.append(mode_semantic_label(sem[mask], semantic_ignore_label, num_classes))
            instance_ids.append(int(instance_id))

    if not object_tokens:
        return None

    geometry_tokens = torch.stack(
        [pool_geometry_tokens(geom, token_grid) for geom in geometry_maps],
        dim=0,
    ).reshape(1, -1, geometry_maps[0].shape[-1])
    camera_tokens_tensor = torch.stack(camera_tokens, dim=1)  # [1, T, 9]

    return {
        "geometry_tokens": geometry_tokens,
        "camera_tokens": camera_tokens_tensor,
        "object_tokens": torch.stack(object_tokens, dim=0).unsqueeze(0),
        "centroids_3d": torch.stack(centroids_3d, dim=0),
        "semantic_labels": torch.tensor(semantic_labels, dtype=torch.long, device=device),
        "instance_ids": torch.tensor(instance_ids, dtype=torch.long, device=device),
    }


def pool_geometry_tokens(geometry: torch.Tensor, token_grid: tuple[int, int]) -> torch.Tensor:
    h, w = token_grid
    x = geometry.permute(2, 0, 1).unsqueeze(0)
    pooled = F.adaptive_avg_pool2d(x, output_size=(h, w))
    return pooled.squeeze(0).permute(1, 2, 0).reshape(h * w, geometry.shape[-1])


def resize_mask_nearest(mask: np.ndarray, size_hw: tuple[int, int], *, device: str) -> torch.Tensor:
    tensor = torch.from_numpy(np.asarray(mask).astype(np.int64)).to(device)
    tensor = tensor[None, None].float()
    resized = F.interpolate(tensor, size=size_hw, mode="nearest")
    return resized[0, 0].long()


def mode_semantic_label(
    labels: torch.Tensor,
    semantic_ignore_label: int,
    num_classes: int,
) -> int:
    valid = labels[(labels != semantic_ignore_label) & (labels >= 0) & (labels < num_classes)]
    if valid.numel() == 0:
        return -1
    values, counts = torch.unique(valid, return_counts=True)
    return int(values[counts.argmax()].item())


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    instance_ids: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    n = embeddings.shape[0]
    if n <= 1:
        return embeddings.sum() * 0.0
    embeddings = F.normalize(embeddings, dim=-1)
    logits = embeddings @ embeddings.T / temperature
    eye = torch.eye(n, dtype=torch.bool, device=embeddings.device)
    same = instance_ids[:, None] == instance_ids[None, :]
    positive = same & ~eye
    if not positive.any():
        return embeddings.sum() * 0.0
    logits = logits.masked_fill(eye, -1e9)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    per_anchor = -(log_prob * positive.float()).sum(dim=1) / positive.float().sum(dim=1).clamp_min(1.0)
    valid = positive.any(dim=1)
    return per_anchor[valid].mean()


def append_metric(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writerow(row)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: ObjectFusionTrainConfig,
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
