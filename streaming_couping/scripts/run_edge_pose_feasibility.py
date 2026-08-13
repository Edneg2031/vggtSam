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
    report_path = _write_outputs(e0.output_dir, result)
    print("E0-r2 DIFFERENTIABLE EDGE POSE-BASIN FEASIBILITY")
    print(f"  revision={EDGE_FEASIBILITY_REVISION}")
    for row in result["summary"]:
        print(
            "  "
            f"branch={row['branch']} "
            f"pairs={row['pair_count']} "
            f"mean_dist_px={row['mean_truncated_edge_distance_px']:.4f} "
            f"depth_consistency={row['mean_depth_consistency_pass_rate']:.4f} "
            f"in_bounds={row['mean_in_bounds_rate']:.4f}"
        )
    recovery = result["recovery_summary"]
    print(
        "  "
        f"recovery_active={recovery['active_trials']}/"
        f"{recovery['recovery_trials']} "
        f"frames={recovery['passing_frames']}/{recovery['evaluation_frames']} "
        f"ordering={recovery['loss_order_pass_rate']:.4f} "
        f"gradient_probe={recovery['gradient_probe_pass_rate']:.4f} "
        f"rotation_recovery={recovery['mean_rotation_recovery_fraction']:.4f} "
        f"translation_recovery={recovery['mean_translation_recovery_fraction']:.4f} "
        f"pose_basin_pass={recovery['pose_basin_pass']}"
    )
    print(f"  output={e0.output_dir / 'edge_feasibility_summary.json'}")
    print(f"  copyable_report={report_path}")


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
    return repo_path if repo_path.is_absolute() else repo_path.resolve()


def _write_outputs(output_dir: Path, result: dict[str, object]) -> Path:
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
    recovery_rows = list(result["recovery_rows"])
    with (output_dir / "perturbation_recovery.csv").open(
        "w",
        newline="",
        encoding="utf8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=csv_columns(recovery_rows),
        )
        writer.writeheader()
        writer.writerows(recovery_rows)
    payload = {
        key: value
        for key, value in result.items()
        if key not in {"rows", "summary", "recovery_rows"}
    }
    payload["summary"] = result["summary"]
    (output_dir / "edge_feasibility_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf8",
    )
    report = _copyable_report(result, output_dir)
    report_path = output_dir / "copyable_result.txt"
    report_path.write_text(report, encoding="utf8")
    return report_path


def _copyable_report(result: dict[str, object], output_dir: Path) -> str:
    lines = [
        "===== COPYABLE_E0_RESULT_BEGIN =====",
        f"revision={result['revision']}",
        f"clip={result['clip']}",
        "branches=" + ",".join(str(value) for value in result["branches"]),
        f"exclusion_mask_source={result['exclusion_mask_source']}",
        f"exclusion_mask_semantics={result['exclusion_mask_semantics']}",
        "evaluation_frames=" + " ".join(str(value) for value in result["evaluation_frames"]),
        "",
        "branch,pair_count,mean_dist_px,depth_consistency_pass_rate,in_bounds_rate",
    ]
    for row in result["summary"]:
        lines.append(
            ",".join(
                (
                    str(row["branch"]),
                    str(row["pair_count"]),
                    _fmt(row["mean_truncated_edge_distance_px"]),
                    _fmt(row["mean_depth_consistency_pass_rate"]),
                    _fmt(row["mean_in_bounds_rate"]),
                )
            )
        )
    recovery = result["recovery_summary"]
    lines.extend(
        [
            "",
            "recovery_trials,active_trials,passing_frames,evaluation_frames,loss_order_pass_rate,gradient_probe_pass_rate,mean_rotation_recovery_fraction,mean_translation_recovery_fraction,pose_basin_pass",
            ",".join(
                (
                    str(recovery["recovery_trials"]),
                    str(recovery["active_trials"]),
                    str(recovery["passing_frames"]),
                    str(recovery["evaluation_frames"]),
                    _fmt(recovery["loss_order_pass_rate"]),
                    _fmt(recovery["gradient_probe_pass_rate"]),
                    _fmt(recovery["mean_rotation_recovery_fraction"]),
                    _fmt(recovery["mean_translation_recovery_fraction"]),
                    str(recovery["pose_basin_pass"]),
                )
            ),
            "minimum_frame_loss_order_pass_rate,minimum_frame_gradient_probe_pass_rate,minimum_frame_rotation_recovery_fraction,minimum_frame_translation_recovery_fraction",
            ",".join(
                (
                    _fmt(recovery["minimum_frame_loss_order_pass_rate"]),
                    _fmt(recovery["minimum_frame_gradient_probe_pass_rate"]),
                    _fmt(recovery["minimum_frame_rotation_recovery_fraction"]),
                    _fmt(recovery["minimum_frame_translation_recovery_fraction"]),
                )
            ),
        ]
    )
    lines.extend(
        [
            "",
            "outputs:",
            f"summary={output_dir / 'edge_feasibility_summary.json'}",
            f"branch_csv={output_dir / 'edge_branch_summary.csv'}",
            f"pair_csv={output_dir / 'edge_pair_diagnostics.csv'}",
            f"recovery_csv={output_dir / 'perturbation_recovery.csv'}",
            f"copyable_report={output_dir / 'copyable_result.txt'}",
            "===== COPYABLE_E0_RESULT_END =====",
            "",
        ]
    )
    return "\n".join(lines)


def _fmt(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:
        return "nan"
    return f"{number:.6f}"


if __name__ == "__main__":
    main()
