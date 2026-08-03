#!/usr/bin/env python3
"""Validate a V7.3 result and render a compact decision table/summary."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


CORE_FIELDS = (
    "architecture",
    "token_count",
    "uses_sam",
    "development_score",
    "v73_development_best",
    "development_loss",
    "validation_loss",
    "future_loss",
    "cross_loss",
    "beats_same_k_geometry_all_splits",
    "beats_retained_v72_geometry_all_splits",
    "beats_camera_controls_report_only",
    "instance_off_exact",
    "sam_perturbation_hurt_count",
    "sam_perturbation_total",
    "causal_sam_pass",
)


def main() -> None:
    args = _parse_args()
    source = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    compact, summary = analyze(source)
    output_dir.mkdir(parents=True, exist_ok=True)
    compact_path = output_dir / "v73_decision_table.csv"
    summary_path = output_dir / "v73_result_summary.md"
    _write_csv(compact_path, compact)
    summary_path.write_text(summary, encoding="utf8")
    print(summary.rstrip())
    print(f"V7.3 compact table={compact_path}")


def analyze(source: Path) -> tuple[list[dict[str, str]], str]:
    if not source.is_file():
        raise FileNotFoundError(f"Missing V7.3 result: {source}")
    with source.open("r", encoding="utf8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty V7.3 result: {source}")
    missing = [field for field in CORE_FIELDS if field not in rows[0]]
    if missing:
        raise ValueError(f"V7.3 CSV lacks fields: {missing}")
    names = [row["architecture"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("V7.3 CSV contains duplicate architecture rows.")
    required_controls = {
        "raw_streamvggt",
        "frozen_l0",
        "camera_extra_all",
        "camera_extra_common_gate",
    }
    absent = sorted(required_controls - set(names))
    if absent:
        raise ValueError(f"V7.3 CSV lacks retained controls: {absent}")
    methods = [
        row for row in rows if row.get("architecture_family") == "v73_correspondence"
    ]
    if not methods:
        raise ValueError("V7.3 CSV contains no correspondence architecture.")
    compact = [
        {field: row.get(field, "") for field in CORE_FIELDS}
        for row in rows
        if row["architecture"] not in {"raw_streamvggt"}
    ]
    selected = min(methods, key=lambda row: float(row["development_score"]))
    causal = [row for row in methods if int(row["causal_sam_pass"] or 0)]
    sam_methods = [row for row in methods if int(row["uses_sam"] or 0)]
    best_sam = min(sam_methods, key=lambda row: float(row["development_score"]))
    summary = _summary(source, selected, best_sam, causal)
    return compact, summary


def _summary(source, selected, best_sam, causal) -> str:
    lines = [
        "# V7.3 result summary",
        "",
        f"Source: `{source}`",
        "",
        "## Development-selected V7.3 architecture",
        "",
        f"- architecture: `{selected['architecture']}`",
        f"- development score: {selected['development_score']}",
        "- development / validation loss: "
        f"{selected['development_loss']} / {selected['validation_loss']}",
        f"- report-only future / cross loss: {selected['future_loss']} / {selected['cross_loss']}",
        "",
        "## Best SAM-weighted candidate",
        "",
        f"- architecture: `{best_sam['architecture']}`",
        f"- development score: {best_sam['development_score']}",
        "- beats same-K geometry on every causal split: "
        f"{best_sam['beats_same_k_geometry_all_splits']}",
        "- beats retained V7.2 geometry on every causal split: "
        f"{best_sam['beats_retained_v72_geometry_all_splits']}",
        f"- beats camera controls on future/cross: {best_sam['beats_camera_controls_report_only']}",
        "- SAM perturbations hurt: "
        f"{best_sam['sam_perturbation_hurt_count']}/"
        f"{best_sam['sam_perturbation_total']}",
        f"- strict causal SAM pass: {best_sam['causal_sam_pass']}",
        "",
        "## Decision",
        "",
    ]
    if causal:
        lines.append(
            "At least one branch passes the predeclared single-seed causal SAM "
            "criterion: " + ", ".join(f"`{row['architecture']}`" for row in causal) + "."
        )
        lines.append(
            "Run multiple V7.3 seeds before treating this as stable evidence."
        )
    else:
        lines.append(
            "No branch passes the strict causal SAM criterion. Treat any lower "
            "normal loss as a capacity or split-specific result until the failed "
            "control/perturbation condition is resolved."
        )
    lines.extend(
        [
            "",
            "The `cross` clip is another segment from the same scene, not a "
            "cross-scene generalization test.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CORE_FIELDS))
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=(
            "outputs/streaming_couping_v73_correspondence/"
            "v73_correspondence_ablation.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/streaming_couping_v73_correspondence/report",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
