#!/usr/bin/env python3
"""Run the isolated E0 masked-edge pose feasibility diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import torch

from streaming_couping.src.edge_pose_feasibility import (
    EDGE_FEASIBILITY_REVISION,
    csv_columns,
    load_edge_feasibility_config,
    run_edge_feasibility,
)
from streaming_couping.src.external_repos import maybe_add_repo_to_path
from streaming_couping.src.learned_pose.cache import (
    build_feature_caches,
    cache_path,
    load_feature_cache,
)
from streaming_couping.src.learned_pose.config import (
    ClipConfig,
    load_learned_pose_config,
)


def main() -> None:
    args = _parse_args()
    e0 = load_edge_feasibility_config(args.config)
    if args.output_dir:
        e0 = replace(e0, output_dir=Path(args.output_dir).expanduser().resolve())
    data = load_learned_pose_config(e0.base_config)
    clip = _find_clip(data.clips, e0.clip_name)
    if args.rebuild_cache:
        data = replace(data, features=replace(data.features, rebuild=True))
    path = cache_path(data, clip)
    if args.stage in {"all", "cache"}:
        build_feature_caches(data)
    if args.stage not in {"all", "audit"}:
        return
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing feature cache {path}; run --stage cache or --stage all."
        )
    payload = load_feature_cache(path)
    if payload.get("clip_name") != clip.name:
        raise ValueError(
            f"Cache clip mismatch: expected {clip.name!r}, "
            f"got {payload.get('clip_name')!r}."
        )
    maybe_add_repo_to_path(data.recovery_config.parent)
    maybe_add_repo_to_path(_recovery_streamvggt_repo(data.recovery_config))
    from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri

    pose, intrinsics = pose_encoding_to_extri_intri(
        payload["baseline_pose_encoding"].unsqueeze(0).float(),
        image_size_hw=tuple(int(value) for value in payload["image_size"]),
    )
    result = run_edge_feasibility(
        payload=payload,
        world_to_camera=pose.detach().cpu(),
        intrinsics=intrinsics.detach().cpu(),
        config=e0,
    )
    _write_outputs(e0.output_dir, result)
    print("E0 MASKED-EDGE POSE FEASIBILITY")
    print(f"  revision={EDGE_FEASIBILITY_REVISION}")
    for row in result["summary"]:
        print(
            "  "
            f"branch={row['branch']} "
            f"pairs={row['pair_count']} "
            f"mean_dist_px={row['mean_truncated_edge_distance_px']:.4f} "
            f"cycle={row['mean_depth_cycle_pass_rate']:.4f} "
            f"in_bounds={row['mean_in_bounds_rate']:.4f}"
        )
    print(f"  output={e0.output_dir / 'edge_feasibility_summary.json'}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="streaming_couping/configs/e0_edge_pose_feasibility.yaml",
    )
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--stage",
        choices=("all", "cache", "audit"),
        default="all",
    )
    parser.add_argument("--rebuild-cache", action="store_true")
    return parser.parse_args()


def _find_clip(clips: tuple[ClipConfig, ...], name: str) -> ClipConfig:
    for clip in clips:
        if clip.name == name:
            return clip
    raise ValueError(f"Unknown clip {name!r}.")


def _recovery_streamvggt_repo(path: Path) -> Path:
    import yaml

    with path.open("r", encoding="utf8") as handle:
        raw = yaml.safe_load(handle) or {}
    repo = raw.get("streamvggt", {}).get("repo")
    if repo is None:
        repo = raw.get("streamvggt_repo")
    if repo is None:
        return Path("externals/streamvggt").resolve()
    repo_path = Path(repo).expanduser()
    return repo_path if repo_path.is_absolute() else (path.parent / repo_path).resolve()


def _write_outputs(output_dir: Path, result: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(result["rows"])
    with (output_dir / "edge_pair_diagnostics.csv").open(
        "w",
        newline="",
        encoding="utf8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_columns(rows))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "edge_branch_summary.csv").open(
        "w",
        newline="",
        encoding="utf8",
    ) as handle:
        summary = list(result["summary"])
        writer = csv.DictWriter(handle, fieldnames=csv_columns(summary))
        writer.writeheader()
        writer.writerows(summary)
    payload = {
        key: value
        for key, value in result.items()
        if key not in {"rows", "summary"}
    }
    payload["summary"] = result["summary"]
    (output_dir / "edge_feasibility_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf8",
    )


if __name__ == "__main__":
    main()
