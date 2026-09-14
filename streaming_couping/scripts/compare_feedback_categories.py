#!/usr/bin/env python3
"""Which objects did a prompt change swap in, and did they help?  (CPU only)

The branch table says a generation got better or worse.  It cannot say why,
and for a prompt-set change the "why" is not a knob: the prompt list is NOT
additive.  Asking for more objects changes which objects the segmenter tracks
at all, including ones it tracked before -- in the v1-vs-v2 pair it dropped
``wardrobe`` while adding ``cabinet`` and ``window``.  So the comparison that
matters is not "five prompts vs twelve" but "this set of object tracks vs that
set of object tracks", and that is what this prints:

* per category, how many proposals it contributed and how wrong they were
  against ground truth, side by side across runs;
* per category, how many frames it actually entered the consensus in, and its
  share of all inlier votes -- an object that produces proposals but never
  becomes an inlier changed nothing;
* the net effect of the consensus: whether combining objects beat a single
  proposal, or lost to it.

Ground truth is read here only to score what the run already produced.  No
decision in the pipeline sees it.

    python -m streaming_couping.scripts.compare_feedback_categories \
        --run-dir <base>_v1.baseline --run-dir <base>_v2.baseline
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from streaming_couping.scripts.summarize_feedback_branches import run_label

PROPOSAL_COLUMNS = (
    "frame",
    "instance_id",
    "category",
    "gt_translation_correction_error",
    "consensus_inlier",
)
CONSENSUS_COLUMNS = (
    "variant",
    "frame",
    "num_reliable",
    "inlier_count",
    "gt_consensus_translation_error",
)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _float(value: Any) -> float | None:
    text = "" if value is None else str(value).strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _flag(value: Any) -> bool:
    return str(value or "").strip().lower() in {"true", "1"}


def _median(values: Sequence[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.median(finite)) if finite else None


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return "nan"
    return f"{float(value):.{digits}f}"


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    rendered = [[_fmt(cell) for cell in row] for row in rows]
    widths = [
        max(len(str(header)), *(len(row[index]) for row in rendered))
        for index, header in enumerate(headers)
    ]
    lines = [
        "  ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)),
        "  ".join("-" * width for width in widths),
    ]
    for row in rendered:
        lines.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
    return "\n".join(lines)


def _require_columns(rows: Sequence[Mapping[str, str]], columns: Sequence[str], path: Path) -> None:
    """A renamed column would otherwise make a section silently report nothing."""

    if not rows:
        raise ValueError(f"{path} has no rows")
    missing = [name for name in columns if name not in rows[0]]
    if missing:
        raise ValueError(f"{path} is missing column(s): {', '.join(missing)}")


def category_errors(proposals: Sequence[Mapping[str, str]]) -> dict[str, dict[str, Any]]:
    """Per category: proposal count and the GT error it carries."""

    buckets: dict[str, list[float]] = {}
    counts: dict[str, int] = {}
    for row in proposals:
        label = (row.get("category") or "<empty>").strip() or "<empty>"
        counts[label] = counts.get(label, 0) + 1
        error = _float(row.get("gt_translation_correction_error"))
        if error is not None:
            buckets.setdefault(label, []).append(error)
    return {
        label: {
            "proposals": counts[label],
            "scored": len(buckets.get(label, ())),
            "median_m": _median(buckets.get(label, ())),
        }
        for label in sorted(counts)
    }


def consensus_participation(
    proposals: Sequence[Mapping[str, str]],
) -> dict[str, dict[str, Any]]:
    """Per category: the frames it was a consensus inlier in, and its vote share.

    An object can produce proposals on every frame and never once be an inlier;
    that object is in the pipeline but not in the answer, and the two look the
    same from the branch table.
    """

    frames: dict[str, set[int]] = {}
    inlier_votes: dict[str, int] = {}
    total_votes = 0
    for row in proposals:
        if not _flag(row.get("consensus_inlier")):
            continue
        label = (row.get("category") or "<empty>").strip() or "<empty>"
        frame = _float(row.get("frame"))
        if frame is not None:
            frames.setdefault(label, set()).add(int(frame))
        inlier_votes[label] = inlier_votes.get(label, 0) + 1
        total_votes += 1
    return {
        label: {
            "inlier_frames": len(frames[label]),
            "inlier_votes": inlier_votes[label],
            "vote_share": (inlier_votes[label] / total_votes) if total_votes else None,
        }
        for label in sorted(inlier_votes)
    }


def consensus_frames(
    consensus_rows: Sequence[Mapping[str, str]], variant: str
) -> dict[str, Any]:
    """Inlier counts over the frames where this variant formed a consensus."""

    rows = [
        row
        for row in consensus_rows
        if (row.get("variant") or "").strip() == variant
    ]
    inliers = [
        value
        for value in (_float(row.get("inlier_count")) for row in rows)
        if value is not None
    ]
    reliable = [
        value
        for value in (_float(row.get("num_reliable")) for row in rows)
        if value is not None
    ]
    return {
        "frames": len(rows),
        "inlier_median": _median(inliers),
        "inlier_mean": float(np.mean(inliers)) if inliers else None,
        "reliable_median": _median(reliable),
        "distinct_frames": len({int(_float(row.get("frame")) or -1) for row in rows}),
    }


def within_frame_agreement(
    proposals: Sequence[Mapping[str, str]],
    consensus_rows: Sequence[Mapping[str, str]],
    variant: str,
) -> dict[str, Any]:
    """How much the objects disagree with each other, and what came out.

    A median over objects helps when their errors are independent -- they
    average out -- and hurts when they share a bias, because then every vote
    reinforces the same mistake.  Those two cases look identical from an
    aggregate error: both are "objects that are each somewhat wrong".  They
    come apart here, because the spread among one frame's inlier errors is the
    independent part, and the consensus error is what survives it.

    Tighter agreement with a WORSE consensus is the signature of a shared bias;
    looser agreement with a better consensus is errors cancelling.
    """

    by_frame: dict[int, list[float]] = {}
    for row in proposals:
        if not _flag(row.get("consensus_inlier")):
            continue
        error = _float(row.get("gt_translation_correction_error"))
        frame = _float(row.get("frame"))
        if error is None or frame is None:
            continue
        by_frame.setdefault(int(frame), []).append(error)

    spreads = [
        float(np.std(values, ddof=1))
        for values in by_frame.values()
        if len(values) >= 2
    ]
    ranges = [
        max(values) - min(values)
        for values in by_frame.values()
        if len(values) >= 2
    ]
    consensus_error = {}
    for row in consensus_rows:
        if (row.get("variant") or "").strip() != variant:
            continue
        frame = _float(row.get("frame"))
        error = _float(row.get("gt_consensus_translation_error"))
        if frame is not None and error is not None:
            consensus_error[int(frame)] = error
    shared = sorted(set(by_frame) & set(consensus_error))
    return {
        "frames_with_two_or_more_inliers": len(spreads),
        "inlier_spread_median_m": _median(spreads),
        "inlier_range_median_m": _median(ranges),
        "consensus_error_median_m": _median(
            [consensus_error[frame] for frame in shared]
        ),
        "frames_compared": len(shared),
    }


def consensus_net_effect(
    proposal_error: float | None, consensus_error: float | None
) -> float | None:
    """How much the consensus improved on a single proposal (positive = better).

    This is the number that separates "we added objects and got more evidence"
    from "we added objects and got a worse answer": proposals can be flat while
    this goes negative, and when it does the extra objects are the reason.
    """

    if proposal_error is None or consensus_error is None or proposal_error <= 0.0:
        return None
    return (proposal_error - consensus_error) / proposal_error


def load_run(run_dir: Path, variant: str | None) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    feedback = run_dir / "object_pose_feedback"
    proposals_path = feedback / "object_proposals.csv"
    consensus_path = feedback / "consensus_metrics.csv"
    for path in (proposals_path, consensus_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing stage-2b artifact: {path}")

    proposals = _read_rows(proposals_path)
    consensus = _read_rows(consensus_path)
    _require_columns(proposals, PROPOSAL_COLUMNS, proposals_path)
    _require_columns(consensus, CONSENSUS_COLUMNS, consensus_path)

    summary: Mapping[str, Any] = {}
    summary_path = feedback / "summary.json"
    if summary_path.is_file():
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)

    main = variant or str(summary.get("main_variant") or "robust_semantic")
    proposal_error = _dig(
        summary, "answers", "q2_single_proposal_error", "translation_median_m"
    )
    consensus_error = _dig(
        summary,
        "answers",
        "q3_consensus_vs_single_object",
        "main_variant_consensus_translation_median_m",
    )
    return {
        "run_dir": str(run_dir),
        "label": run_label(run_dir),
        "variant": main,
        "categories": category_errors(proposals),
        "participation": consensus_participation(proposals),
        "frames": consensus_frames(consensus, main),
        "agreement": within_frame_agreement(proposals, consensus, main),
        "proposal_error_m": proposal_error,
        "consensus_error_m": consensus_error,
        "net_effect": consensus_net_effect(proposal_error, consensus_error),
    }


def _dig(document: Mapping[str, Any], *keys: str) -> Any:
    current: Any = document
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def render_categories(runs: Sequence[Mapping[str, Any]]) -> str:
    """Per category: proposal count and GT error, one column pair per run."""

    labels = [str(run["label"]) for run in runs]
    every = sorted({name for run in runs for name in run["categories"]})
    headers = ["category"]
    for label in labels:
        headers += [f"{label} n", f"{label} med_m"]
    rows: list[list[Any]] = []
    for name in every:
        row: list[Any] = [name]
        for run in runs:
            entry = run["categories"].get(name)
            row += [
                entry["proposals"] if entry else "-",
                entry["median_m"] if entry else None,
            ]
        rows.append(row)
    lines = [
        "(1) per category: proposals contributed, and their GT translation error",
        _table(headers, rows),
    ]
    swapped = []
    for run in runs:
        present = set(run["categories"])
        swapped.append(f"{run['label']}: {len(present)}")
    only_in_one = [
        name
        for name in every
        if any(name not in run["categories"] for run in runs)
    ]
    if only_in_one:
        lines.append("")
        lines.append(
            "categories the runs do not share -- the prompt list is not additive, "
            "so this is where it shows:"
        )
        for name in only_in_one:
            where = [
                run["label"] for run in runs if name in run["categories"]
            ]
            missing = [
                run["label"] for run in runs if name not in run["categories"]
            ]
            lines.append(
                f"  {name!r}: present in {', '.join(where)}; "
                f"absent from {', '.join(missing)}"
            )
    lines.append("")
    lines.append("category counts: " + "; ".join(swapped))
    return "\n".join(lines)


def render_participation(runs: Sequence[Mapping[str, Any]]) -> str:
    labels = [str(run["label"]) for run in runs]
    every = sorted({name for run in runs for name in run["participation"]})
    headers = ["category"]
    for label in labels:
        headers += [f"{label} frames", f"{label} votes", f"{label} share"]
    rows: list[list[Any]] = []
    for name in every:
        row: list[Any] = [name]
        for run in runs:
            entry = run["participation"].get(name)
            row += [
                entry["inlier_frames"] if entry else "-",
                entry["inlier_votes"] if entry else "-",
                entry["vote_share"] if entry else None,
            ]
        rows.append(row)
    return "\n".join(
        [
            "(2) per category: frames it was a consensus inlier in, votes cast, "
            "and its share of all inlier votes",
            "    a category with proposals but no frames here never entered the "
            "answer at all",
            _table(headers, rows),
        ]
    )


def render_net_effect(runs: Sequence[Mapping[str, Any]]) -> str:
    headers = [
        "run",
        "variant",
        "prop_err_m",
        "cons_err_m",
        "consensus_gain",
        "consensus_frames",
        "inlier_median",
    ]
    rows = [
        [
            run["label"],
            run["variant"],
            run["proposal_error_m"],
            run["consensus_error_m"],
            run["net_effect"],
            run["frames"]["frames"],
            run["frames"]["inlier_median"],
        ]
        for run in runs
    ]
    return "\n".join(
        [
            "(3) did combining objects beat a single proposal?",
            "    consensus_gain = (prop_err - cons_err) / prop_err; negative means "
            "the consensus is\n    WORSE than one proposal, so the extra objects "
            "actively hurt",
            _table(headers, rows),
        ]
    )


def render_agreement(runs: Sequence[Mapping[str, Any]]) -> str:
    headers = [
        "run",
        "inlier_frames",
        "inlier_spread_m",
        "inlier_range_m",
        "consensus_err_m",
        "consensus_gain",
    ]
    rows = [
        [
            run["label"],
            run["agreement"]["frames_with_two_or_more_inliers"],
            run["agreement"]["inlier_spread_median_m"],
            run["agreement"]["inlier_range_median_m"],
            run["agreement"]["consensus_error_median_m"],
            run["net_effect"],
        ]
        for run in runs
    ]
    return "\n".join(
        [
            "(4) within-frame agreement among the consensus inliers",
            "    inlier_spread_m is the median per-frame spread of the inlier "
            "errors: the part\n    that averaging can cancel.  Read it against "
            "consensus_err_m --",
            "      wide spread + small consensus error = the objects disagree, "
            "and the median wins",
            "      narrow spread + large consensus error = the objects agree on "
            "the same mistake",
            _table(headers, rows),
        ]
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        required=True,
        dest="run_dirs",
        help="Repeat once per run, in the order to print.",
    )
    parser.add_argument(
        "--variant",
        default=None,
        help="Consensus variant to read; defaults to each run's own main variant.",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    runs = [load_run(path, args.variant) for path in args.run_dirs]

    print("object categories across runs")
    for run in runs:
        print(f"  {run['label']}: {run['run_dir']}")
    print()
    print(render_categories(runs))
    print()
    print(render_participation(runs))
    print()
    print(render_net_effect(runs))
    print()
    print(render_agreement(runs))

    if args.json_out is not None:
        args.json_out.expanduser().resolve().write_text(
            json.dumps({"runs": runs}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print()
        print(f"json_out={args.json_out.expanduser().resolve()}")


if __name__ == "__main__":
    main()
