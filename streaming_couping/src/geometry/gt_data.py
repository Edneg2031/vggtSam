"""Load ScanNet++ GT pointmaps and poses on the StreamVGGT image grid."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image

from test_sam.coordinates import streamvggt_image_transform, streamvggt_label_to_grid
from test_sam.data import read_mask, resolve_manifest_path
from vggtsam.data.scannetpp.object_sequence import read_pointmap


@dataclass(frozen=True)
class GTGeometrySequence:
    pointmaps: torch.Tensor
    world_to_camera: torch.Tensor
    intrinsics: torch.Tensor
    instance_masks: torch.Tensor
    colors: np.ndarray


def load_gt_geometry_sequence(
    manifest_path: str | Path,
    *,
    scene_id: str,
    frame_indices: Sequence[int],
    instance_id: int,
    processed_size: tuple[int, int],
    image_mode: str,
) -> GTGeometrySequence:
    manifest_path = Path(manifest_path).expanduser().resolve()
    with manifest_path.open("r", encoding="utf8") as handle:
        manifest = json.load(handle)
    scene = next(
        (item for item in manifest.get("scenes", []) if item.get("scene_id") == scene_id),
        None,
    )
    if scene is None:
        raise ValueError(f"Scene {scene_id!r} is missing from {manifest_path}.")
    frames = scene.get("frames", [])
    selected = [frames[int(index)] for index in frame_indices]

    pointmaps = []
    poses = []
    intrinsics = []
    masks = []
    colors = []
    for frame in selected:
        pointmap_value = frame.get("pointmap")
        if not pointmap_value:
            raise ValueError(
                "The GT-mask pose experiment requires mesh-rasterized pointmaps."
            )
        pointmap_path = resolve_manifest_path(pointmap_value, manifest_path)
        image_path = resolve_manifest_path(frame["image_path"], manifest_path)
        mask_path = resolve_manifest_path(frame["instance_mask"], manifest_path)
        pointmaps.append(
            transform_dense_map(
                read_pointmap(pointmap_path),
                processed_size,
                mode=image_mode,
            )
        )
        instance_labels = read_mask(mask_path)
        masks.append(
            streamvggt_label_to_grid(
                instance_labels,
                processed_size,
                mode=image_mode,
            )
            == int(instance_id)
        )
        colors.append(process_rgb(image_path, processed_size, mode=image_mode))

        pose = np.asarray(frame.get("world_to_camera"), dtype=np.float32)
        intrinsic = np.asarray(frame.get("intrinsics"), dtype=np.float32)
        if pose.shape != (4, 4) or intrinsic.shape != (3, 3):
            raise ValueError(
                "The GT-mask pose experiment requires COLMAP world_to_camera and intrinsics."
            )
        poses.append(pose)
        intrinsics.append(intrinsic)

    return GTGeometrySequence(
        pointmaps=torch.from_numpy(np.stack(pointmaps)).float(),
        world_to_camera=torch.from_numpy(np.stack(poses)).float(),
        intrinsics=torch.from_numpy(np.stack(intrinsics)).float(),
        instance_masks=torch.from_numpy(np.stack(masks)).bool(),
        colors=np.stack(colors),
    )


def transform_dense_map(
    values: np.ndarray,
    output_size: tuple[int, int],
    *,
    mode: str,
) -> np.ndarray:
    """Sample an original-pixel dense map on StreamVGGT's processed grid."""

    if values.ndim != 3:
        raise ValueError(f"Expected [H,W,C] dense map, got {values.shape}.")
    transform = streamvggt_image_transform(values.shape[:2], mode=mode)
    output_height, output_width = (int(output_size[0]), int(output_size[1]))
    target_y, target_x = np.meshgrid(
        np.arange(output_height, dtype=np.float32),
        np.arange(output_width, dtype=np.float32),
        indexing="ij",
    )
    processed_to_native_x = transform.target_size[1] / float(output_width)
    processed_to_native_y = transform.target_size[0] / float(output_height)
    native_x = (target_x + 0.5) * processed_to_native_x - 0.5
    native_y = (target_y + 0.5) * processed_to_native_y - 0.5
    source_x = (
        (native_x - transform.offset_xy[0] + 0.5) / transform.scale_xy[0] - 0.5
    )
    source_y = (
        (native_y - transform.offset_xy[1] + 0.5) / transform.scale_xy[1] - 0.5
    )
    x_index = np.floor(source_x + 0.5).astype(np.int64)
    y_index = np.floor(source_y + 0.5).astype(np.int64)
    valid = (
        (x_index >= 0)
        & (x_index < values.shape[1])
        & (y_index >= 0)
        & (y_index < values.shape[0])
    )
    output = np.full(
        (output_height, output_width, values.shape[-1]),
        np.nan,
        dtype=np.float32,
    )
    output[valid] = values[y_index[valid], x_index[valid]].astype(np.float32)
    return output


def process_rgb(
    image_path: Path,
    output_size: tuple[int, int],
    *,
    mode: str,
) -> np.ndarray:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        transform = streamvggt_image_transform((image.height, image.width), mode=mode)
        resized_width = int(round(image.width * transform.scale_xy[0]))
        resized_height = int(round(image.height * transform.scale_xy[1]))
        image = image.resize(
            (resized_width, resized_height),
            resample=Image.Resampling.BICUBIC,
        )
        if mode == "crop":
            crop_top = max(0, (resized_height - 518) // 2)
            crop_bottom = crop_top + min(518, resized_height)
            image = image.crop((0, crop_top, resized_width, crop_bottom))
        else:
            canvas = Image.new("RGB", (518, 518), "white")
            canvas.paste(
                image,
                ((518 - resized_width) // 2, (518 - resized_height) // 2),
            )
            image = canvas
        image = image.resize(
            (int(output_size[1]), int(output_size[0])),
            resample=Image.Resampling.BICUBIC,
        )
        return np.asarray(image, dtype=np.uint8).copy()
