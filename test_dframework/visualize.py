"""Compact visual reports for geometry-bridge controls."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw


COLORS = {
    "gt": (40, 210, 80),
    "sam3": (55, 130, 255),
    "prior": (255, 180, 30),
    "bridge": (230, 55, 90),
}


def save_report(
    path: Path,
    *,
    image_paths: Sequence[Path],
    frame_indices: Sequence[int],
    gt_masks: torch.Tensor,
    sam_masks: torch.Tensor,
    priors: torch.Tensor,
    bridged_masks: torch.Tensor,
    scores: torch.Tensor,
    decisions: Sequence[str],
    output_size: tuple[int, int],
    mode: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, image_path in enumerate(image_paths):
        with Image.open(image_path) as image:
            rgb = image.convert("RGB").resize((output_size[1], output_size[0]))
        columns = [
            annotate(rgb, f"RGB frame={frame_indices[index]}\nscore={float(scores[index]):.3f}"),
            overlay(rgb, gt_masks[index], COLORS["gt"], "GT instance"),
            overlay(rgb, sam_masks[index], COLORS["sam3"], "SAM3 original"),
            overlay(rgb, priors[index], COLORS["prior"], "Projected 3D prior"),
            overlay(rgb, bridged_masks[index], COLORS["bridge"], f"Bridge ({mode})\n{decisions[index]}"),
        ]
        rows.append(concat_horizontal(columns))
    concat_vertical(rows).save(path)


def overlay(image: Image.Image, mask: torch.Tensor, color: tuple[int, int, int], title: str) -> Image.Image:
    pixels = np.asarray(image).copy()
    mask_np = mask.detach().cpu().numpy().astype(bool)
    pixels[mask_np] = (0.55 * pixels[mask_np] + 0.45 * np.asarray(color)).astype(np.uint8)
    output = Image.fromarray(pixels)
    draw = ImageDraw.Draw(output)
    draw.rectangle((0, 0, output.width, 38), fill=(0, 0, 0))
    draw.text((6, 5), title, fill=(255, 255, 255))
    draw.text((6, 21), f"pixels={int(mask_np.sum())}", fill=color)
    return output


def annotate(image: Image.Image, text: str) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    draw.rectangle((0, 0, output.width, 38), fill=(0, 0, 0))
    for line_idx, line in enumerate(text.splitlines()):
        draw.text((6, 5 + 16 * line_idx), line, fill=(255, 255, 255))
    return output


def concat_horizontal(images: Sequence[Image.Image]) -> Image.Image:
    output = Image.new("RGB", (sum(image.width for image in images), max(image.height for image in images)))
    left = 0
    for image in images:
        output.paste(image, (left, 0))
        left += image.width
    return output


def concat_vertical(images: Sequence[Image.Image]) -> Image.Image:
    output = Image.new("RGB", (max(image.width for image in images), sum(image.height for image in images)))
    top = 0
    for image in images:
        output.paste(image, (0, top))
        top += image.height
    return output

