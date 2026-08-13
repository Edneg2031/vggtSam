#!/usr/bin/env python3
"""Run E1 fixed-step edge-gradient directions with scoring-only GT."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

from streaming_couping.src.edge_directional_gt_audit import (
    E1_REVISION,
    generate_directional_candidates,
    load_edge_directional_audit_config,
    objective_payload_from_cache,
    score_directional_candidates,
)
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


def main() -> None:
    args = _parse_args()
    e1 = load_edge_directional_audit_config(args.config)
    if args.output_dir:
        e1 = replace(
            e1, output_dir=Path(args.output_dir).expanduser().resolve()
        )
    data = load_learned_pose_config(e1.base_config)
    clip = _find_clip(data.clips, e1.clip_name)
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
        raise ValueError("E1 cache clip mismatch.")
    maybe_add_repo_to_path(data.recovery_config.parent)
    maybe_add_repo_to_path(_recovery_streamvggt_repo(data.recovery_config))
    from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri

    image_size = tuple(int(value) for value in payload["image_size"])
    # Phase 1: raw inputs only. Target pose is deliberately not decoded yet.
    raw_pose, intrinsics = pose_encoding_to_extri_intri(
        payload["baseline_pose_encoding"].unsqueeze(0).float(),
        image_size_hw=image_size,
    )
    objective_payload = objective_payload_from_cache(payload)
    candidates, generation_audit = generate_directional_candidates(
        payload=objective_payload,
        raw_world_to_camera=raw_pose.detach(),
        intrinsics=intrinsics.detach(),
        config=e1,
        device=args.device or e1.objective.recovery.device,
    )

    # Phase 2: only now is GT decoded and used to score immutable candidates.
    target_pose, _ = pose_encoding_to_extri_intri(
        payload["target_pose_encoding"].unsqueeze(0).float(),
        image_size_hw=image_size,
    )
    scored = score_directional_candidates(
        candidates,
        target_world_to_camera=target_pose.detach(),
        config=e1,
    )
    result = {
        "revision": E1_REVISION,
        "clip": payload["clip_name"],
        "frames": tuple(int(value) for value in payload["frame_indices"]),
        "evaluation_frames": e1.evaluation_frames,
        "branches": e1.branches,
        "primary_branch": e1.primary_branch,
        "generation_audit": generation_audit,
        **scored,
    }
    report = _write_outputs(e1.output_dir, result)
    print("E1 FIXED-STEP EDGE DIRECTIONAL GT AUDIT")
    print(f"  revision={E1_REVISION}")
    for row in result["fold_summary"]:
        print(
            "  "
            f"fold={row['fold']} branch={row['branch']} "
            f"R_gain_deg={row['rotation_gain_deg']:.6f} "
            f"center_gain={row['center_gain_native']:.6f} "
            f"R_worse={row['rotation_worse_frames']} "
            f"center_worse={row['center_worse_frames']} "
            f"pass={row['fold_pass']}"
        )
    print(f"  decision={json.dumps(result['decision'], sort_keys=True)}")
    print(f"  copyable_report={report}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="streaming_couping/configs/e1_edge_directional_gt_audit.yaml",
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
    _write_csv(output_dir / "directional_frame_diagnostics.csv", rows)
    _write_csv(output_dir / "directional_fold_summary.csv", folds)
    payload = {
        key: value
        for key, value in result.items()
        if key not in {"rows", "fold_summary"}
    }
    payload["fold_summary"] = folds
    (output_dir / "directional_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf8",
    )
    report_path = output_dir / "copyable_result.txt"
    report_path.write_text(_copyable_report(result, output_dir), encoding="utf8")
    return report_path


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty E1 CSV: {path.name}")
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_columns(rows))
        writer.writeheader()
        writer.writerows(rows)


def _copyable_report(result: dict[str, object], output_dir: Path) -> str:
    audit = result["generation_audit"]
    decision = result["decision"]
    lines = [
        "===== COPYABLE_E1_RESULT_BEGIN =====",
        f"revision={result['revision']}",
        f"clip={result['clip']}",
        "branches=" + ",".join(result["branches"]),
        f"primary_branch={result['primary_branch']}",
        f"gt_role={decision['gt_role']}",
        f"target_pose_source={decision['target_pose_source']}",
        f"candidate_generation_gt_fields={audit['candidate_generation_gt_fields']}",
        f"fixed_rotation_step_degrees={_fmt(audit['fixed_rotation_step_degrees'])}",
        "fixed_translation_step_scene_fraction="
        + _fmt(audit["fixed_translation_step_scene_fraction"]),
        f"equal_edge_count_per_frame={audit['equal_edge_count_per_frame']}",
        f"exclusion_mask_source={audit['exclusion_mask_source']}",
        "",
        "fold,branch,active_frames,R_gain_deg,center_gain_native,R_worse,center_worse,negative_loss_decrease_rate,positive_loss_increase_rate,rotation_only_gain_deg,rotation_only_worse,rotation_only_pass,translation_only_center_gain,translation_only_worse,translation_only_pass,joint_pass",
    ]
    for row in result["fold_summary"]:
        lines.append(
            ",".join(
                (
                    str(row["fold"]),
                    str(row["branch"]),
                    str(row["active_frames"]),
                    _fmt(row["rotation_gain_deg"]),
                    _fmt(row["center_gain_native"]),
                    str(row["rotation_worse_frames"]),
                    str(row["center_worse_frames"]),
                    _fmt(row["negative_edge_loss_decrease_rate"]),
                    _fmt(row["positive_edge_loss_increase_rate"]),
                    _fmt(row["negative_rotation_only_gain_deg"]),
                    str(row["rotation_only_worse_frames"]),
                    str(row["rotation_only_fold_pass"]),
                    _fmt(row["negative_translation_only_center_gain_native"]),
                    str(row["translation_only_worse_frames"]),
                    str(row["translation_only_fold_pass"]),
                    str(row["fold_pass"]),
                )
            )
        )
    lines.extend(
        [
            "",
            "decision=" + json.dumps(decision, sort_keys=True),
            "",
            "outputs:",
            f"summary={output_dir / 'directional_summary.json'}",
            f"fold_csv={output_dir / 'directional_fold_summary.csv'}",
            f"frame_csv={output_dir / 'directional_frame_diagnostics.csv'}",
            f"copyable_report={output_dir / 'copyable_result.txt'}",
            "===== COPYABLE_E1_RESULT_END =====",
            "",
        ]
    )
    return "\n".join(lines)


def _fmt(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not (number == number):
        return "nan"
    return f"{number:.6f}"


if __name__ == "__main__":
    main()
