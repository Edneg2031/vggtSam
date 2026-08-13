#!/usr/bin/env python3
"""Run G0 causal static projective point-to-plane ICP candidate audit."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

from streaming_couping.src.edge_pose_feasibility import csv_columns
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
from streaming_couping.src.static_projective_icp import (
    G0_REVISION,
    generate_projective_icp_candidates,
    load_static_projective_icp_config,
    objective_payload_from_cache,
    score_projective_icp_candidates,
)


def main() -> None:
    args = _parse_args()
    g0 = load_static_projective_icp_config(args.config)
    if args.output_dir:
        g0 = replace(
            g0, output_dir=Path(args.output_dir).expanduser().resolve()
        )
    data = load_learned_pose_config(g0.base_config)
    clip = _find_clip(data.clips, g0.clip_name)
    if args.rebuild_cache:
        data = replace(data, features=replace(data.features, rebuild=True))
    path = cache_path(data, clip)
    if args.stage in {"all", "cache"}:
        build_feature_caches(data)
    if args.stage not in {"all", "audit"}:
        return
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing V0 feature cache {path}; run --stage cache or --stage all."
        )
    payload = load_feature_cache(path)
    if payload.get("clip_name") != clip.name:
        raise ValueError("G0 cache clip mismatch.")

    maybe_add_repo_to_path(data.recovery_config.parent)
    maybe_add_repo_to_path(_recovery_streamvggt_repo(data.recovery_config))
    from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri

    image_size = tuple(int(value) for value in payload["image_size"])
    raw_pose, intrinsics = pose_encoding_to_extri_intri(
        payload["baseline_pose_encoding"].unsqueeze(0).float(),
        image_size_hw=image_size,
    )

    # Phase 1: GT fields are physically absent from this payload and the
    # candidate function has no target-pose parameter.
    objective = objective_payload_from_cache(payload)
    candidates, generation_audit = generate_projective_icp_candidates(
        payload=objective,
        raw_world_to_camera=raw_pose.detach(),
        intrinsics=intrinsics.detach(),
        config=g0,
        device=args.device or g0.device,
    )

    # Phase 2: decode GT only after all candidates are immutable CPU tensors.
    target_pose, _ = pose_encoding_to_extri_intri(
        payload["target_pose_encoding"].unsqueeze(0).float(),
        image_size_hw=image_size,
    )
    scored = score_projective_icp_candidates(
        candidates,
        target_world_to_camera=target_pose.detach(),
        config=g0,
    )
    result = {
        "revision": G0_REVISION,
        "clip": payload["clip_name"],
        "frames": tuple(int(value) for value in payload["frame_indices"]),
        "evaluation_frames": g0.evaluation_frames,
        "branches": g0.branches,
        "primary_branch": g0.primary_branch,
        "generation_audit": generation_audit,
        **scored,
    }
    report = _write_outputs(g0.output_dir, result)
    print("G0 CAUSAL STATIC PROJECTIVE POINT-TO-PLANE ICP")
    print(f"  revision={G0_REVISION}")
    for row in result["fold_summary"]:
        print(
            "  "
            f"fold={row['fold']} branch={row['branch']} "
            f"active={row['active_frames']}/4 "
            f"corr={row['mean_correspondences']:.1f} "
            f"R_gain_deg={row['rotation_gain_deg']:.6f} "
            f"center_gain={row['center_gain_native']:.6f} "
            f"pass={row['fold_pass']}"
        )
    print(f"  decision={json.dumps(result['decision'], sort_keys=True)}")
    print(f"  copyable_report={report}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="streaming_couping/configs/g0_static_projective_icp.yaml",
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--device")
    parser.add_argument(
        "--stage", choices=("all", "cache", "audit"), default="all"
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
    repo = raw.get("streamvggt", {}).get("repo") or raw.get("streamvggt_repo")
    if repo is None:
        return Path("externals/streamvggt").resolve()
    repo_path = Path(repo).expanduser()
    return repo_path if repo_path.is_absolute() else repo_path.resolve()


def _write_outputs(output_dir: Path, result: dict[str, object]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(result["rows"])
    folds = list(result["fold_summary"])
    _write_csv(output_dir / "frame_diagnostics.csv", rows)
    _write_csv(output_dir / "fold_summary.csv", folds)
    payload = {
        key: value
        for key, value in result.items()
        if key not in {"rows", "fold_summary"}
    }
    payload["fold_summary"] = folds
    (output_dir / "candidate_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf8",
    )
    report = output_dir / "copyable_result.txt"
    report.write_text(_copyable_report(result, output_dir), encoding="utf8")
    return report


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty G0 CSV: {path.name}")
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_columns(rows))
        writer.writeheader()
        writer.writerows(rows)


def _copyable_report(result: dict[str, object], output_dir: Path) -> str:
    audit = result["generation_audit"]
    lines = [
        "===== COPYABLE_G0_RESULT_BEGIN =====",
        f"revision={result['revision']}",
        f"clip={result['clip']}",
        "branches=" + ",".join(result["branches"]),
        f"primary_branch={result['primary_branch']}",
        f"solver={audit['solver']}",
        f"source_offsets={' '.join(str(v) for v in audit['source_offsets'])}",
        f"pyramid_factors={' '.join(str(v) for v in audit['pyramid_factors'])}",
        f"candidate_generation_gt_fields={audit['candidate_generation_gt_fields']}",
        f"gt_role={audit['gt_role']}",
        f"exclusion_mask_source={audit['exclusion_mask_source']}",
        f"exclusion_mask_semantics={audit['exclusion_mask_semantics']}",
        f"sam_appearance_tokens_used={audit['sam_appearance_tokens_used']}",
        f"mask_area_control_exact={audit['mask_area_control_exact']}",
        "",
        "fold,branch,active_frames,mean_correspondences,inlier_fraction,energy_decrease,R_gain_deg,R_worse,center_gain_native,center_worse,fold_pass",
    ]
    for row in result["fold_summary"]:
        lines.append(
            ",".join(
                (
                    str(row["fold"]),
                    str(row["branch"]),
                    str(row["active_frames"]),
                    _fmt(row["mean_correspondences"]),
                    _fmt(row["mean_inlier_fraction"]),
                    _fmt(row["mean_energy_decrease_fraction"]),
                    _fmt(row["rotation_gain_deg"]),
                    str(row["rotation_worse_frames"]),
                    _fmt(row["center_gain_native"]),
                    str(row["center_worse_frames"]),
                    str(row["fold_pass"]),
                )
            )
        )
    lines.extend(
        (
            "",
            "decision=" + json.dumps(result["decision"], sort_keys=True),
            "",
            "outputs:",
            f"summary={output_dir / 'candidate_summary.json'}",
            f"fold_csv={output_dir / 'fold_summary.csv'}",
            f"frame_csv={output_dir / 'frame_diagnostics.csv'}",
            f"copyable_report={output_dir / 'copyable_result.txt'}",
            "===== COPYABLE_G0_RESULT_END =====",
            "",
        )
    )
    return "\n".join(lines)


def _fmt(value: object) -> str:
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    main()
