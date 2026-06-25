#!/usr/bin/env python3
"""Inspect SAM3/StreamVGGT latent fusion adapter outputs."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_latent_fusion import build_train_config, load_config
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
)
from vggtsam.training.latent_fusion import select_object_prompt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/latent_fusion_train.yaml"),
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--prompt", default=None)
    args = parser.parse_args()

    raw = load_config(args.config)
    if args.device is not None:
        raw["training"]["device"] = args.device
    if args.prompt is not None:
        raw["sam3"]["prompt"] = args.prompt
        raw["sam3"]["prompt_mode"] = "fixed"
        if args.prompt.strip().lower() != "object":
            raw["objects"]["target_object_labels"] = [args.prompt]
    config = build_train_config(raw)

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
    sequence = dataset.sample(random.Random(config.seed))
    print(f"scene={sequence.scene_id} frames={sequence.frame_indices}")
    for path in sequence.image_paths:
        print(f"  {path}")
    print(f"pointmaps_available={sequence.pointmaps is not None}")
    label_preview = sorted(sequence.object_labels.items())[:20]
    print(f"object_label_preview={label_preview}")
    prompt_selection = select_object_prompt(
        sequence.visible_instance_ids,
        sequence.object_labels,
        rng=random.Random(config.seed),
        min_visible_frames=config.min_visible_frames,
        mode=config.sam3_prompt_mode,
        fallback_prompt=config.sam3_prompt,
        target_object_labels=config.target_object_labels,
        excluded_object_labels=config.excluded_object_labels,
    )
    if prompt_selection is None:
        raise RuntimeError(
            "No valid prompt candidate found in the sampled clip. "
            "Try lowering object filters or clearing excluded_object_labels."
        )
    print(
        "sampled_prompt="
        f"{prompt_selection.prompt!r} sampled_instance_id={prompt_selection.sampled_instance_id}"
    )

    sam3_model = load_sam3_image_model(
        repo_path=config.sam3_repo,
        checkpoint_path=config.sam3_checkpoint,
        device=config.device,
        enable_inst_interactivity=config.sam3_enable_inst_interactivity,
    )
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
    geometry = StreamVGGTLatentAdapter(
        streamvggt_model,
        device=config.device,
        token_grid=config.token_grid,
        context_grid=config.context_grid,
        layer_index=config.streamvggt_layer_index,
        image_mode=config.streamvggt_image_mode,
    )

    with torch.no_grad():
        sam_out = sam3.extract_from_paths(
            sequence.image_paths,
            prompt=prompt_selection.prompt,
        )
        geo_out = geometry.extract_from_paths(sequence.image_paths)

    print("SAM3:")
    print(f"  tokens={tuple(sam_out.semantic.tokens.shape)}")
    print(f"  spatial_shape={sam_out.semantic.spatial_shape}")
    print(f"  aux={sam_out.semantic.aux}")
    print("StreamVGGT:")
    print(f"  geometry_tokens={tuple(geo_out.geometry.tokens.shape)}")
    print(f"  use_camera_tokens_in_training={config.use_camera_tokens}")
    print(
        "  camera_tokens="
        f"{None if geo_out.geometry.camera_tokens is None else tuple(geo_out.geometry.camera_tokens.shape)}"
    )
    print(
        "  pointmap_grid="
        f"{None if geo_out.pointmap_grid is None else tuple(geo_out.pointmap_grid.shape)}"
    )
    print(f"  aux={geo_out.aux}")


if __name__ == "__main__":
    main()
