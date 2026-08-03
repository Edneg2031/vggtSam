#!/usr/bin/env python3
"""Render cached SAM3.1 masks and farthest-UV local-token samples once."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from streaming_couping.src.learned_pose.cache import cache_path, load_feature_cache
from streaming_couping.src.learned_pose.config import load_learned_pose_config


COLORS = (
    (235, 72, 72),
    (72, 205, 116),
    (70, 130, 240),
    (245, 180, 55),
    (190, 90, 220),
    (35, 200, 205),
)


def main() -> None:
    args = _parse_args()
    config = load_learned_pose_config(args.config)
    output_dir = Path(args.output_dir).expanduser().resolve()
    complete = output_dir / "COMPLETE.json"
    if complete.is_file() and not args.force:
        print(f"V7.2 local-token visualization already complete: {output_dir}")
        return
    rows = render_local_tokens(
        config,
        output_dir=output_dir,
        point_radius=int(args.point_radius),
        mask_alpha=float(args.mask_alpha),
    )
    complete.write_text(
        json.dumps(
            {
                "config": str(config.source_path),
                "frames": len({(row["clip"], row["frame_index"]) for row in rows}),
                "instance_frames": len(rows),
            },
            indent=2,
        )
        + "\n",
        encoding="utf8",
    )
    print(f"V7.2 local-token visualization={output_dir}")


def render_local_tokens(
    config,
    *,
    output_dir: Path,
    point_radius: int,
    mask_alpha: float,
) -> list[dict[str, object]]:
    if point_radius < 1:
        raise ValueError("point_radius must be positive.")
    if not 0.0 <= mask_alpha <= 1.0:
        raise ValueError("mask_alpha must be in [0,1].")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    html_entries: list[tuple[str, str, str]] = []
    for clip in config.clips:
        payload = load_feature_cache(cache_path(config, clip))
        for field in ("sam_local_uv", "sam_local_valid", "tracking_masks_output"):
            if not torch.is_tensor(payload.get(field)):
                raise ValueError(
                    f"Cache {clip.name} lacks {field}; run V7.2 cache first."
                )
        uv = payload["sam_local_uv"].detach().float().cpu()
        valid = payload["sam_local_valid"].detach().bool().cpu()
        masks = payload["tracking_masks_output"].detach().bool().cpu()
        scores = torch.as_tensor(payload.get("tracking_scores", torch.zeros(valid.shape[:2])))
        clip_dir = output_dir / clip.name
        clip_dir.mkdir(exist_ok=True)
        for sequence_index, (frame_index, image_path) in enumerate(
            zip(clip.frame_indices, payload["image_paths"])
        ):
            image = Image.open(image_path).convert("RGB")
            overlay = image.copy()
            overlay = _overlay_masks(
                overlay,
                masks[sequence_index],
                alpha=mask_alpha,
            )
            draw = ImageDraw.Draw(overlay)
            width, height = overlay.size
            for slot, instance_id in enumerate(clip.instance_ids):
                selected = uv[sequence_index, slot, valid[sequence_index, slot]]
                color = COLORS[slot % len(COLORS)]
                for point in selected:
                    x = (float(point[0]) + 1.0) * 0.5 * max(width - 1, 1)
                    y = (float(point[1]) + 1.0) * 0.5 * max(height - 1, 1)
                    radius = max(int(point_radius), 1)
                    draw.ellipse(
                        (x - radius, y - radius, x + radius, y + radius),
                        fill=(255, 255, 255),
                        outline=(0, 0, 0),
                        width=1,
                    )
                    inner = max(radius - 2, 1)
                    draw.ellipse(
                        (x - inner, y - inner, x + inner, y + inner),
                        fill=color,
                    )
                mask_pixels = int(masks[sequence_index, slot].sum())
                rows.append(
                    {
                        "clip": clip.name,
                        "split": clip.split,
                        "sequence_index": sequence_index,
                        "frame_index": int(frame_index),
                        "is_reference": int(
                            sequence_index == clip.reference_sequence_index
                        ),
                        "instance_id": int(instance_id),
                        "tracking_score": _short(scores[sequence_index, slot]),
                        "mask_pixels": mask_pixels,
                        "local_tokens": int(selected.shape[0]),
                        "token_coverage_ratio": _short(
                            float(selected.shape[0]) / max(mask_pixels, 1)
                        ),
                    }
                )
            label = (
                f"{clip.split} | {clip.name} | frame {frame_index} | "
                + " ".join(
                    f"id={instance_id}:{int(valid[sequence_index, slot].sum())}tok"
                    for slot, instance_id in enumerate(clip.instance_ids)
                )
            )
            overlay = _add_title(overlay, label)
            destination = clip_dir / f"frame_{int(frame_index):06d}.png"
            overlay.save(destination)
            html_entries.append(
                (
                    clip.name,
                    label,
                    str(destination.relative_to(output_dir)),
                )
            )
    _write_csv(output_dir / "local_token_visualization.csv", rows)
    (output_dir / "index.html").write_text(
        _html_index(html_entries), encoding="utf8"
    )
    return rows


def _overlay_masks(
    image: Image.Image,
    masks: torch.Tensor,
    *,
    alpha: float,
) -> Image.Image:
    width, height = image.size
    output = image.convert("RGBA")
    for slot in range(int(masks.shape[0])):
        mask = Image.fromarray(
            (masks[slot].numpy().astype("uint8") * 255), mode="L"
        ).resize(
            (width, height),
            resample=(
                Image.Resampling.NEAREST
                if hasattr(Image, "Resampling")
                else Image.NEAREST
            ),
        )
        color = COLORS[slot % len(COLORS)]
        layer = Image.new("RGBA", (width, height), (*color, int(255 * alpha)))
        transparent = Image.new("L", (width, height), 0)
        effective_alpha = Image.blend(transparent, mask, alpha)
        layer.putalpha(effective_alpha)
        output = Image.alpha_composite(output, layer)
    return output.convert("RGB")


def _add_title(image: Image.Image, title: str) -> Image.Image:
    title_height = 30
    output = Image.new("RGB", (image.width, image.height + title_height), "white")
    output.paste(image, (0, title_height))
    ImageDraw.Draw(output).text((8, 8), title, fill="black")
    return output


def _html_index(entries: list[tuple[str, str, str]]) -> str:
    blocks = []
    last_clip = None
    for clip, title, path in entries:
        if clip != last_clip:
            blocks.append(f"<h2>{html.escape(clip)}</h2>")
            last_clip = clip
        blocks.append(
            "<figure><img loading='lazy' src='{path}'><figcaption>{title}</figcaption></figure>".format(
                path=html.escape(path), title=html.escape(title)
            )
        )
    return """<!doctype html>
<html><head><meta charset="utf-8"><title>V7.2 local tokens</title>
<style>body{font-family:sans-serif;margin:24px}figure{margin:20px 0}img{max-width:100%;height:auto;border:1px solid #bbb}figcaption{margin-top:6px;color:#333}</style>
</head><body><h1>V7.2 SAM3.1 mask-local token audit</h1>{blocks}</body></html>
""".format(blocks="\n".join(blocks))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _short(value) -> str:
    return f"{float(value):.8g}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="streaming_couping/configs/v72_local_token_data.yaml",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/streaming_couping_v72_local_token_visualization",
    )
    parser.add_argument("--point-radius", type=int, default=5)
    parser.add_argument("--mask-alpha", type=float, default=0.28)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
