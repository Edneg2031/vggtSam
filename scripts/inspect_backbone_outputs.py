#!/usr/bin/env python3
"""Inspect SAM3 and VGGT outputs on processed ScanNet++ frames.

This script is intentionally diagnostic. It does not decide which backbone layer
is final; it prints and saves tensor shapes so we can choose the fusion inputs
from real server outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from vggtsam.adapters.introspection import summarize_object, tensor_candidates
from vggtsam.adapters.sam3 import (
    load_sam3_video_predictor,
    prepare_video_frame_dir,
    run_sam3_text_prompt,
)
from vggtsam.adapters.vggt import load_vggt_model, run_vggt_forward


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    args = apply_config(args)

    frame_paths = select_manifest_frames(
        manifest_path=args.manifest,
        scene_id=args.scene_id,
        num_frames=args.num_frames,
    )
    print(f"selected_frames={len(frame_paths)}")
    for path in frame_paths:
        print(f"  {path}")

    report: Dict[str, Any] = {
        "manifest": str(args.manifest),
        "scene_id": args.scene_id,
        "frame_paths": [str(p) for p in frame_paths],
        "sam3": None,
        "vggt": None,
    }

    if not args.skip_sam3 and args.sam3_checkpoint:
        report["sam3"] = inspect_sam3(args, frame_paths)

    if not args.skip_vggt and args.vggt_checkpoint:
        report["vggt"] = inspect_vggt(args, frame_paths)

    if args.output_json:
        write_json(args.output_json, report)
        print(f"wrote {args.output_json}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/scannetpp_2d/manifest.json"),
    )
    parser.add_argument("--scene-id", default=None)
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-json", type=Path, default=None)

    parser.add_argument("--sam3-repo", type=Path, default=Path("externals/sam3"))
    parser.add_argument("--sam3-checkpoint", type=Path, default=None)
    parser.add_argument("--sam3-prompt", default="chair")
    parser.add_argument("--sam3-gpus", default="0")
    parser.add_argument("--sam3-tmp-dir", type=Path, default=Path("outputs/tmp/sam3_frames"))
    parser.add_argument("--skip-sam3", action="store_true")

    parser.add_argument("--vggt-repo", type=Path, default=Path("externals/vggt"))
    parser.add_argument("--vggt-checkpoint", type=Path, default=None)
    parser.add_argument("--vggt-patch-multiple", type=int, default=14)
    parser.add_argument(
        "--vggt-value-scale",
        type=float,
        default=1.0,
        help="Use 1.0 to match the previous smoke test; use 255.0 to normalize 0-255 images.",
    )
    parser.add_argument("--skip-vggt", action="store_true")
    return parser


def apply_config(args: argparse.Namespace) -> argparse.Namespace:
    if args.config is None:
        return args

    import yaml

    with args.config.open("r", encoding="utf8") as handle:
        config = yaml.safe_load(handle) or {}

    dataset = config.get("dataset", {})
    sam3 = config.get("sam3", {})
    vggt = config.get("vggt", {})
    runtime = config.get("runtime", {})

    defaults = build_arg_parser().parse_args([])

    maybe_set(args, defaults, "manifest", dataset.get("manifest"), Path)
    maybe_set(args, defaults, "scene_id", dataset.get("scene_id"))
    maybe_set(args, defaults, "num_frames", dataset.get("num_frames"), int)
    maybe_set(args, defaults, "device", runtime.get("device"))
    maybe_set(args, defaults, "output_json", runtime.get("output_json"), Path)

    maybe_set(args, defaults, "sam3_repo", sam3.get("repo"), Path)
    maybe_set(args, defaults, "sam3_checkpoint", sam3.get("checkpoint"), Path)
    maybe_set(args, defaults, "sam3_prompt", sam3.get("prompt"))
    if getattr(args, "sam3_gpus") == getattr(defaults, "sam3_gpus"):
        gpus = sam3.get("gpus_to_use")
        if gpus is not None:
            args.sam3_gpus = ",".join(str(gpu) for gpu in gpus)

    maybe_set(args, defaults, "vggt_repo", vggt.get("repo"), Path)
    maybe_set(args, defaults, "vggt_checkpoint", vggt.get("checkpoint"), Path)
    maybe_set(args, defaults, "vggt_patch_multiple", vggt.get("patch_multiple"), int)
    maybe_set(args, defaults, "vggt_value_scale", vggt.get("value_scale"), float)
    return args


def maybe_set(
    args: argparse.Namespace,
    defaults: argparse.Namespace,
    name: str,
    value,
    caster=None,
) -> None:
    if value is None:
        return
    if getattr(args, name) != getattr(defaults, name):
        return
    setattr(args, name, caster(value) if caster is not None else value)


def select_manifest_frames(
    *,
    manifest_path: Path,
    scene_id: Optional[str],
    num_frames: int,
) -> List[Path]:
    manifest = read_json(manifest_path)
    scenes = manifest.get("scenes", [])
    if not scenes:
        raise ValueError(f"No scenes found in manifest: {manifest_path}")

    scene = None
    if scene_id is None:
        scene = scenes[0]
    else:
        for candidate in scenes:
            if candidate.get("scene_id") == scene_id:
                scene = candidate
                break
    if scene is None:
        raise ValueError(f"Scene {scene_id!r} not found in {manifest_path}")

    frames = scene.get("frames", [])
    frame_paths = [Path(frame["image_path"]) for frame in frames if "image_path" in frame]
    frame_paths = [path for path in frame_paths if path.is_file()]
    if not frame_paths:
        raise ValueError(f"No readable image paths found for scene {scene.get('scene_id')}")
    return frame_paths[:num_frames]


def inspect_sam3(args: argparse.Namespace, frame_paths: List[Path]) -> Dict[str, Any]:
    print("loading SAM3...")
    predictor = load_sam3_video_predictor(
        repo_path=args.sam3_repo,
        checkpoint_path=args.sam3_checkpoint,
        gpus_to_use=parse_gpu_list(args.sam3_gpus),
    )
    frame_dir = prepare_video_frame_dir(frame_paths, args.sam3_tmp_dir)
    print(f"running SAM3 prompt={args.sam3_prompt!r} frame_dir={frame_dir}")
    results = run_sam3_text_prompt(
        predictor,
        frame_dir=frame_dir,
        prompt=args.sam3_prompt,
        frame_idx=0,
    )
    first = results[0] if results else None
    summary = summarize_object(first)
    candidates = tensor_candidates(first)
    print_section("SAM3 first output summary", summary)
    print_candidates("SAM3 tensor candidates", candidates)
    return {
        "num_results": len(results),
        "first_summary": summary,
        "first_tensor_candidates": candidates,
    }


def inspect_vggt(args: argparse.Namespace, frame_paths: List[Path]) -> Dict[str, Any]:
    print("loading VGGT...")
    import torch

    model = load_vggt_model(
        repo_path=args.vggt_repo,
        checkpoint_path=args.vggt_checkpoint,
        device=args.device,
        strict=False,
    )
    images = load_images_as_tensor(frame_paths).to(args.device)
    print(f"running VGGT images={tuple(images.shape)}")
    output = run_vggt_forward(
        model,
        images,
        patch_multiple=args.vggt_patch_multiple,
        value_scale=args.vggt_value_scale,
    )
    if str(args.device).startswith("cuda"):
        torch.cuda.empty_cache()
    summary = summarize_object(output)
    candidates = tensor_candidates(output)
    print_section("VGGT output summary", summary)
    print_candidates("VGGT tensor candidates", candidates)
    return {"summary": summary, "tensor_candidates": candidates}


def load_images_as_tensor(frame_paths: List[Path]):
    import numpy as np
    import torch
    from PIL import Image

    tensors = []
    for path in frame_paths:
        image = Image.open(path).convert("RGB")
        array = np.asarray(image)
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        tensors.append(tensor)
    return torch.stack(tensors, dim=0).float()


def parse_gpu_list(text: str) -> List[int]:
    if text.strip() == "":
        return []
    return [int(item) for item in text.split(",")]


def print_section(title: str, payload: Any) -> None:
    print(f"\n=== {title} ===")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def print_candidates(title: str, candidates: List[Dict[str, Any]]) -> None:
    print(f"\n=== {title} ===")
    if not candidates:
        print("(none)")
        return
    for item in candidates:
        print(
            f"{item['path']}: shape={item['shape']} dtype={item['dtype']} "
            f"rank={item['rank']}"
        )


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
