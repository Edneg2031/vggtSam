#!/usr/bin/env python3
"""Small no-data smoke test for the multi-clip ablation protocol."""

from __future__ import annotations

from pathlib import Path
import tempfile

import torch

from streaming_couping.scripts.run_multiclip_semantic_ablation import (
    _aggregate_results,
    _parse_variants,
    _resolve_artifact_path,
)
from streaming_couping.src.learned_pose.config import ClipConfig


def main() -> None:
    variants = _parse_variants("raw,v23_all_visible,v23")
    assert variants == ("raw", "v23_all_visible", "v23")
    oracle_variants = _parse_variants("raw,oracle")
    assert oracle_variants == ("raw", "oracle")

    raw_tracking = {
        "variant": "raw_sam",
        "mean_frame_iou": 0.40,
        "frame_idf1": 0.50,
    }
    candidate_tracking = {
        "variant": "v2_3_failure_only_confidence_aware_voxel_memory",
        "mean_frame_iou": 0.45,
        "frame_idf1": 0.55,
    }
    raw_map = {
        "variant": "raw_sam",
        "voxel_iou_5cm": 0.10,
        "fscore_5cm": 0.30,
        "ghost_point_ratio": 0.20,
    }
    candidate_map = {
        "variant": "v2_3_failure_only_confidence_aware_voxel_memory",
        "voxel_iou_5cm": 0.11,
        "fscore_5cm": 0.31,
        "ghost_point_ratio": 0.19,
    }
    clip = {
        "scene_id": "scene",
        "clip_name": "clip",
        "variants": {
            "raw": {"tracking": raw_tracking, "map": raw_map},
            "v23": {"tracking": candidate_tracking, "map": candidate_map},
        },
    }
    aggregate = _aggregate_results(
        clip_results=[clip],
        tracking_rows=[raw_tracking, candidate_tracking],
        map_rows=[raw_map, candidate_map],
        primary_variant="v23",
    )
    assert aggregate["decision"] == "GO", aggregate
    assert aggregate["comparisons_vs_raw"][0]["tracking_nonregression"] == 1

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        clip_config = ClipConfig(
            name="scene_clip",
            scene_id="scene",
            frame_indices=(0,),
            instance_ids=(0,),
        )
        nested = root / clip_config.name
        nested.mkdir()
        torch.save({"clip": clip_config.name}, nested / "semantic_map.pt")
        assert _resolve_artifact_path(root, clip_config) == (
            nested / "semantic_map.pt"
        ).resolve()

        exporter_nested = root / "exporter_clip" / "semantic_map"
        exporter_nested.mkdir(parents=True)
        exporter_artifact = exporter_nested / "semantic_map.pt"
        torch.save({"clip": "exporter_clip"}, exporter_artifact)
        exporter_config = ClipConfig(
            name="exporter_clip",
            scene_id="scene",
            frame_indices=(0,),
            instance_ids=(0,),
        )
        assert _resolve_artifact_path(root, exporter_config) == (
            exporter_artifact
        ).resolve()

    print("multi-clip semantic ablation smoke passed variants=3 decision=GO")


if __name__ == "__main__":
    main()
