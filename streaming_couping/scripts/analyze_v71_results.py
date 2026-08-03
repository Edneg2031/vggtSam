#!/usr/bin/env python3
"""Turn the V7.1 one-table CSV into a concise decision report."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


REQUIRED_ARCHITECTURES = {
    "raw_streamvggt",
    "frozen_l0",
    "camera_extra_all",
    "camera_extra_common_gate",
    "gate_only",
    "appearance_pool",
    "geometry_pool",
    "decoupled_global",
    "local32_decoupled",
}


def analyze(csv_path: str | Path, output_path: str | Path) -> Path:
    source = Path(csv_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    with source.open("r", encoding="utf8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"V7.1 result is empty: {source}")
    by_architecture = {row["architecture"]: row for row in rows}
    missing = REQUIRED_ARCHITECTURES - set(by_architecture)
    if missing:
        raise ValueError(
            "V7.1 result is missing architecture(s): "
            + ", ".join(sorted(missing))
        )
    if len(by_architecture) != len(rows):
        raise ValueError("V7.1 result contains duplicate architecture rows.")

    ranked = sorted(rows, key=lambda row: _float(row, "development_score"))
    content = [row for row in rows if int(row["instance_content"]) == 1]
    content_ranked = sorted(
        content,
        key=lambda row: _float(row, "development_score"),
    )
    causal = [
        row for row in content if int(row["causal_instance_pass"]) == 1
    ]
    best = ranked[0]
    best_content = content_ranked[0]
    l0 = by_architecture["frozen_l0"]
    camera_controls = [
        by_architecture["camera_extra_all"],
        by_architecture["camera_extra_common_gate"],
    ]

    lines = [
        "# V7.1 instance causality result",
        "",
        f"Source: `{source}`",
        "",
        "## Compact ranking",
        "",
        "| Rank | Architecture | Evidence | Dev score ↓ | Dev gain vs L0 | "
        "Future gain vs L0 | Cross gain vs L0 | Causal pass |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(ranked, start=1):
        lines.append(
            f"| {rank} | `{row['architecture']}` | {row['evidence']} | "
            f"{_float(row, 'development_score'):.4f} | "
            f"{_float(row, 'development_gain_vs_frozen_l0_percent'):.2f}% | "
            f"{_float(row, 'future_gain_vs_frozen_l0_percent'):.2f}% | "
            f"{_float(row, 'cross_gain_vs_frozen_l0_percent'):.2f}% | "
            f"{int(row['causal_instance_pass'])} |"
        )

    lines.extend(["", "## Decision", ""])
    lines.append(
        f"- Development best: `{best['architecture']}` "
        f"(score {_float(best, 'development_score'):.4f})."
    )
    lines.append(
        f"- Best instance-content branch: `{best_content['architecture']}` "
        f"(score {_float(best_content, 'development_score'):.4f}, "
        f"future gain "
        f"{_float(best_content, 'future_gain_vs_frozen_l0_percent'):.2f}%, "
        f"cross gain "
        f"{_float(best_content, 'cross_gain_vs_frozen_l0_percent'):.2f}%)."
    )
    if causal:
        names = ", ".join(f"`{row['architecture']}`" for row in causal)
        lines.append(
            f"- Causal instance evidence passed for: {names}. These branches "
            "beat frozen L0 and both camera-capacity controls on every required "
            "split while instance-off returns to L0."
        )
    else:
        lines.append(
            "- No instance-content branch satisfies the strict causal criterion. "
            "Do not claim that instance tokens improve camera pose yet."
        )

    control_best = min(
        camera_controls,
        key=lambda row: _float(row, "development_score"),
    )
    if _float(best_content, "development_score") >= _float(
        control_best,
        "development_score",
    ):
        lines.append(
            "- The best instance-content branch does not beat the best added-"
            "camera-capacity control on development data; any gain can still be "
            "explained by capacity or gating."
        )
    else:
        lines.append(
            "- Instance content beats the best added-camera-capacity control on "
            "development data; inspect future/cross and perturbation damage before "
            "accepting the claim."
        )

    local = by_architecture["local32_decoupled"]
    global_branch = by_architecture["decoupled_global"]
    if _float(local, "development_score") < _float(
        global_branch,
        "development_score",
    ) and _float(local, "future_loss") < _float(global_branch, "future_loss"):
        lines.append(
            "- Local32 improves both development score and future loss over the "
            "global decoupled branch. It is reasonable to implement cached SAM3.1 "
            "mask-local FPN descriptors next."
        )
    else:
        lines.append(
            "- Local32 does not consistently beat global decoupling. Do not add "
            "the cost of SAM3.1 local FPN caching solely from this run."
        )

    geometry = by_architecture["geometry_pool"]
    lines.extend(
        [
            "",
            "## Perturbation checks",
            "",
            f"- Geometry-pool future wrong-geometry damage: "
            f"{_float(geometry, 'future_wrong_geometry_damage_percent'):.2f}%.",
            f"- Best-content future wrong-geometry damage: "
            f"{_float(best_content, 'future_wrong_geometry_damage_percent'):.2f}%.",
            f"- Best-content future shuffle-time damage: "
            f"{_float(best_content, 'future_shuffle_time_damage_percent'):.2f}%.",
            "",
            "Interpret positive damage as evidence that the trained branch actually "
            "uses that input. Improvement without perturbation damage is more likely "
            "to be a gate/capacity effect.",
            "",
            "## Baseline reference",
            "",
            f"Frozen L0 development loss: "
            f"{_float(l0, 'development_loss'):.8g}; future loss: "
            f"{_float(l0, 'future_loss'):.8g}; cross loss: "
            f"{_float(l0, 'cross_loss'):.8g}.",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf8")
    return output


def _float(row: dict[str, str], field: str) -> float:
    value = row.get(field, "")
    if value is None or not str(value).strip():
        raise ValueError(
            f"V7.1 row {row.get('architecture')!r} lacks field {field!r}."
        )
    return float(value)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    path = analyze(args.csv, args.output)
    print(f"V7.1 analysis={path}")
