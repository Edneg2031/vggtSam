#!/usr/bin/env python3
"""Side-by-side table over several object-pose-feedback run directories.

Read-only.  One row per branch, pulling the numbers that decide the next step:
what the proposal-side knobs were, whether the reference-staleness signal
actually went away, and whether the pose metrics moved.

    python -m streaming_couping.scripts.summarize_feedback_branches \
        --run-dir <baseline> --run-dir <fresh> ...

Reads ``<run-dir>/object_pose_feedback/summary.json`` and, when present,
``<run-dir>/object_pose_feedback/attribution.json``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


def _load(path: Path) -> Mapping[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, ValueError):
        return None
    return document if isinstance(document, dict) else None


def _dig(document: Mapping[str, Any] | None, *keys: str, default: Any = None) -> Any:
    current: Any = document
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def _ratio(raw: Any, variant: Any) -> float | None:
    if raw is None or variant is None:
        return None
    raw_value = float(raw)
    if raw_value <= 0.0:
        return None
    return (raw_value - float(variant)) / raw_value


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return "nan"
    if isinstance(value, float) and value == int(value) and abs(value) < 1e6:
        return str(int(value))
    return f"{float(value):.{digits}f}"


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    rendered = [[_fmt(cell) for cell in row] for row in rows]
    widths = [
        max(len(str(header)), *(len(row[index]) for row in rendered))
        for index, header in enumerate(headers)
    ]
    lines = [
        "  ".join(
            str(header).ljust(widths[index]) for index, header in enumerate(headers)
        ),
        "  ".join("-" * width for width in widths),
    ]
    for row in rendered:
        lines.append(
            "  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row))
        )
    return "\n".join(lines)


def branch_row(run_dir: Path) -> tuple[list[Any], dict[str, Any]]:
    """One table row plus the raw numbers behind it."""

    run_dir = run_dir.expanduser().resolve()
    feedback = run_dir / "object_pose_feedback"
    summary = _load(feedback / "summary.json")
    attribution = _load(feedback / "attribution.json")
    # Branch runs are named "<base>.<branch>"; label by the branch so the table
    # stays readable instead of repeating the whole run-directory name.
    name = run_dir.name
    label = name.rsplit(".", 1)[-1] if "." in name else name

    if summary is None:
        # Width derived from HEADERS so adding a column cannot desynchronise it.
        return [label, "no summary.json"] + ["-"] * (len(HEADERS) - 2), {
            "run_dir": str(run_dir),
            "available": False,
        }

    main = str(summary.get("main_variant") or "")
    settings = _dig(summary, "refiner_settings_audit", "exported_by_stage_2a", default={})
    raw_ate = _dig(summary, "branches", "raw", "ate_rmse_m")
    main_ate = _dig(summary, "branches", main, "ate_rmse_m")
    raw_sim3 = _dig(summary, "branches", "raw", "ate_rmse_sim3_m")
    main_sim3 = _dig(summary, "branches", main, "ate_rmse_sim3_m")
    staleness = _dig(
        attribution, "reference_staleness", "anchor_reference_age", default={}
    )
    oldest = _dig(
        attribution, "reference_staleness", "oldest_reference_age", default={}
    )
    gate_reasons = _dig(summary, "gate_stats", main, "reject_reason_counts", default={})
    decision = _dig(summary, "decisions", main, "decision")
    max_age = _dig(settings, "max_reference_age_frames")
    refresh = _dig(settings, "anchor_refresh_interval_frames")
    # The two proposal-side targets: how wrong the proposals are, and how old
    # the reference set they were fitted against is.
    proposal_error = _dig(summary, "answers", "q2_single_proposal_error", "translation_median_m")
    consensus_error = _dig(
        summary,
        "answers",
        "q3_consensus_vs_single_object",
        "main_variant_consensus_translation_median_m",
    )
    future_rotation = _dig(summary, "decisions", main, "future_rotation_gain_median_deg")

    row = [
        label,
        str(_dig(settings, "proposal_mode", default="-")),
        "-" if not (max_age or refresh) else f"{max_age}/{refresh}",
        summary.get("proposal_count"),
        proposal_error,
        consensus_error,
        _dig(summary, "gate_stats", main, "accepted_frame_count"),
        _ratio(raw_ate, main_ate),
        _ratio(raw_sim3, main_sim3),
        future_rotation,
        _dig(staleness, "median_age_frames"),
        _dig(staleness, "spearman_rho_within_category"),
        {
            "OBJECT_FEEDBACK_GO": "GO",
            "OBJECT_FEEDBACK_NO_GO": "NO_GO",
        }.get(str(decision), str(decision)),
    ]
    payload = {
        "run_dir": str(run_dir),
        "available": True,
        "main_variant": main,
        "proposal_mode": _dig(settings, "proposal_mode"),
        "max_reference_age_frames": max_age,
        "anchor_refresh_interval_frames": refresh,
        "proposal_count": summary.get("proposal_count"),
        "proposal_translation_error_median_m": proposal_error,
        "consensus_translation_error_median_m": consensus_error,
        "accepted_frame_count": _dig(
            summary, "gate_stats", main, "accepted_frame_count"
        ),
        "raw_ate_rmse_m": raw_ate,
        "variant_ate_rmse_m": main_ate,
        "direct_ate_improvement_ratio": _ratio(raw_ate, main_ate),
        "raw_ate_rmse_sim3_m": raw_sim3,
        "variant_ate_rmse_sim3_m": main_sim3,
        "sim3_ate_improvement_ratio": _ratio(raw_sim3, main_sim3),
        "future_rotation_gain_median_deg": future_rotation,
        "decision": decision,
        "reject_reason_counts": gate_reasons,
        "anchor_reference_age_median_frames": _dig(staleness, "median_age_frames"),
        "anchor_reference_age_rho_within_category": _dig(
            staleness, "spearman_rho_within_category"
        ),
        "anchor_reference_age_p_within_category": _dig(
            staleness, "permutation_p_within_category"
        ),
        "oldest_reference_age_median_frames": _dig(oldest, "median_age_frames"),
        "oldest_reference_age_rho_within_category": _dig(
            oldest, "spearman_rho_within_category"
        ),
    }
    return row, payload


HEADERS: tuple[str, ...] = (
    "branch",
    "mode",
    "fresh",
    "proposals",
    "prop_err",
    "cons_err",
    "accepted",
    "d_ATE",
    "d_sim3",
    "fut_rot",
    "anchor_age",
    "anchor_rho",
    "decision",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        required=True,
        dest="run_dirs",
        help="Repeat once per branch, in the order to print.",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    rows: list[list[Any]] = []
    payloads: list[dict[str, Any]] = []
    for run_dir in args.run_dirs:
        row, payload = branch_row(run_dir)
        rows.append(row)
        payloads.append(payload)

    print("object pose feedback branches")
    print(
        _table(
            HEADERS,
            rows,
        )
    )
    print()
    print(
        "prop_err  median GT translation error of a single proposal (m) -- the\n"
        "          bottleneck this round targets (baseline 0.086)\n"
        "cons_err  median GT translation error of the consensus (baseline 0.042)\n"
        "d_ATE     improvement of the main variant over raw (positive better)\n"
        "d_sim3    same after a similarity alignment: if d_ATE > 0 but d_sim3 < 0\n"
        "          the gain was a global gauge fix, not better relative geometry\n"
        "fut_rot   median future rotation gain over t+1..t+10 (deg; the\n"
        "          factorized branch targets rotation, so watch this one)\n"
        "fresh     max_reference_age_frames / anchor_refresh_interval_frames\n"
        "anchor_age median age of the anchor reference set (frames)\n"
        "anchor_rho within-category Spearman between anchor age and the GT\n"
        "          correction error (baseline +0.785; see attribution (b6))"
    )
    for payload in payloads:
        if payload.get("available"):
            print(
                f"  {Path(payload['run_dir']).name}: reject reasons "
                f"{payload.get('reject_reason_counts')}"
            )
    if args.json_out is not None:
        args.json_out.expanduser().resolve().write_text(
            json.dumps(payloads, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
