#!/usr/bin/env python3
"""Attribution analysis for the object-consensus pose-feedback run (CPU only).

Reads the four stage-2b CSVs and answers three questions that decide where the
next iteration should go.  Nothing is recomputed and no model runs: this is a
read-only post-mortem over artifacts that already exist.

(a) Did the geometric reliability term collapse the cross-object consensus into
    an effectively single-object estimate?  ``consensus_metrics.csv`` carries a
    per-frame ``inlier_count``; if ``robust_semantic_geometric`` sits at one
    inlier while ``robust_semantic`` sits at two or more, the multiplicative
    degeneracy damping is the mechanism, not the consensus itself.

(b) Do the reliability scores actually predict proposal correctness?  Spearman
    rank correlation (with a permutation p-value) and median splits of the GT
    translation correction error against every recorded proposal feature.  A
    score that does not rank-order correctness cannot improve a weighted
    consensus, however it is combined.

(c) Does the selection filter select correctly?  Proposal-level (the refiner
    verdict and the consensus inlier flag) and frame-level (accepted vs
    rejected gates).  Frame-level consent requires a stage-2b run from a
    revision that records the consensus delta on rejected frames; older CSVs
    leave those cells empty and the script says so instead of guessing.

Ground truth is read only here, after the fact.  It never re-enters any
decision.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

MAIN_VARIANT = "robust_semantic_geometric"
BASELINE_VARIANT = "robust_semantic"
PERMUTATION_ITERATIONS = 4000
SEED = 20260911

#: Columns this analysis reads by name.  A renamed column in the stage-2b
#: writers would otherwise make a section silently report nothing, so the
#: presence of these is checked up front.
REQUIRED_COLUMNS: Mapping[str, tuple[str, ...]] = {
    "object_proposals.csv": (
        "frame",
        "instance_id",
        "category",
        "geometry_type",
        "gt_translation_correction_error",
        "object_reject_reason",
        "accepted_by_refiner",
        "consensus_inlier",
    ),
    "consensus_metrics.csv": (
        "variant",
        "frame",
        "num_reliable",
        "inlier_count",
        "consensus_translation_norm",
        "accepted",
        "gt_consensus_translation_error",
    ),
    "frame_metrics.csv": (
        "variant",
        "frame",
        "raw_translation_error",
        "feedback_translation_error",
        "accepted",
    ),
}

PROPOSAL_FEATURES: tuple[str, ...] = (
    "semantic_confidence",
    "geometry_confidence",
    "inlier_ratio",
    "overlap_count",
    "point_count",
    "track_length",
    "visibility_ratio",
    "degeneracy_factor",
    "delta_translation_norm",
    "delta_rotation_deg",
    "alignment_loss_before",
    "alignment_loss_after",
    "translation_consensus_error",
)


# ---------------------------------------------------------------------------
# CSV parsing (``_write_csv`` writes None as "" and bools as True/False).
# ---------------------------------------------------------------------------


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


def _int(value: Any) -> int | None:
    parsed = _float(value)
    return None if parsed is None else int(parsed)


def _flag(value: Any) -> bool | None:
    text = "" if value is None else str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    return None


def _column(
    rows: Sequence[Mapping[str, str]], name: str
) -> list[float | None]:
    return [_float(row.get(name)) for row in rows]


# ---------------------------------------------------------------------------
# Rank statistics (no scipy dependency).
# ---------------------------------------------------------------------------


def _ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks, ascending (ties share the mean rank)."""

    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    ranks[order] = np.arange(1, values.shape[0] + 1, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < sorted_values.shape[0]:
        stop = start
        while (
            stop + 1 < sorted_values.shape[0]
            and sorted_values[stop + 1] == sorted_values[start]
        ):
            stop += 1
        if stop > start:
            ranks[order[start : stop + 1]] = 0.5 * (start + stop) + 1.0
        start = stop + 1
    return ranks


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    centered_left = left - left.mean()
    centered_right = right - right.mean()
    denominator = float(np.linalg.norm(centered_left) * np.linalg.norm(centered_right))
    if denominator <= 0.0:
        return float("nan")
    return float(centered_left @ centered_right / denominator)


def _spearman(
    features: np.ndarray, target: np.ndarray
) -> tuple[float, float]:
    """(rho, two-sided permutation p-value)."""

    feature_ranks = _ranks(features)
    target_ranks = _ranks(target)
    rho = _pearson(feature_ranks, target_ranks)
    if not math.isfinite(rho):
        return float("nan"), float("nan")
    generator = np.random.default_rng(SEED)
    extreme = 0
    threshold = abs(rho) - 1e-12
    for _ in range(PERMUTATION_ITERATIONS):
        shuffled = generator.permutation(target_ranks)
        if abs(_pearson(feature_ranks, shuffled)) >= threshold:
            extreme += 1
    return rho, (extreme + 1) / (PERMUTATION_ITERATIONS + 1)


def _median_split_p(
    values: np.ndarray, target: np.ndarray, *, iterations: int = PERMUTATION_ITERATIONS
) -> float:
    """Permutation p-value for the difference of group medians."""

    order = np.argsort(values, kind="stable")
    half = values.shape[0] // 2
    if half < 2:
        return float("nan")
    low = order[:half]
    high = order[half:]
    observed = abs(
        float(np.median(target[high])) - float(np.median(target[low]))
    )
    generator = np.random.default_rng(SEED)
    extreme = 0
    indices = np.arange(target.shape[0])
    for _ in range(iterations):
        shuffled = generator.permutation(indices)
        candidate = abs(
            float(np.median(target[shuffled[half:]]))
            - float(np.median(target[shuffled[:half]]))
        )
        if candidate >= observed - 1e-12:
            extreme += 1
    return (extreme + 1) / (iterations + 1)


def _median(values: Iterable[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.median(finite)) if finite else None


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if float(value) != float(value):
        return "nan"
    return f"{float(value):.{digits}f}"


def _table(
    headers: Sequence[str], rows: Sequence[Sequence[Any]], *, digits: int = 4
) -> str:
    rendered = [
        [_fmt(cell, digits) for cell in row]
        for row in rows
    ]
    widths = [
        max(len(str(header)), *(len(row[index]) for row in rendered))
        if rendered
        else len(str(header))
        for index, header in enumerate(headers)
    ]
    lines = [
        "  ".join(
            str(header).ljust(widths[index])
            for index, header in enumerate(headers)
        )
    ]
    lines.append("  ".join("-" * width for width in widths))
    for row in rendered:
        lines.append(
            "  ".join(
                cell.ljust(widths[index]) for index, cell in enumerate(row)
            )
        )
    return "\n".join(lines)


def _paired(values: Sequence[float | None]) -> np.ndarray:
    """Drop rows where the value is missing; return a float array."""

    return np.asarray(
        [float(value) for value in values if value is not None], dtype=np.float64
    )


def _grouped(
    rows: Sequence[Mapping[str, str]],
    *,
    key: str,
    target: str,
) -> list[tuple[str, int, float | None, float | None]]:
    buckets: dict[str, list[float]] = {}
    for row in rows:
        target_value = _float(row.get(target))
        if target_value is None:
            continue
        label = (row.get(key) or "<empty>").strip() or "<empty>"
        buckets.setdefault(label, []).append(target_value)
    return [
        (
            label,
            len(values),
            _median(values),
            float(np.mean(values)),
        )
        for label, values in sorted(buckets.items())
    ]


# ---------------------------------------------------------------------------
# (a) Consensus collapse check.
# ---------------------------------------------------------------------------


def consensus_collapse(
    consensus_rows: Sequence[Mapping[str, str]],
) -> tuple[str, dict[str, Any]]:
    variants = sorted({(row.get("variant") or "").strip() for row in consensus_rows})
    rows: list[list[Any]] = []
    payload: dict[str, Any] = {}
    for variant in variants:
        subset = [
            row for row in consensus_rows if (row.get("variant") or "").strip() == variant
        ]
        # ``consensus_translation_norm`` is written exactly when a consensus
        # was formed, so it is a precise "did consensus run here" flag --
        # independent of each variant's own min_objects rule.
        attempted = [
            row
            for row in subset
            if _float(row.get("consensus_translation_norm")) is not None
        ]
        inliers = [
            int(_int(row.get("inlier_count")) or 0)
            for row in attempted
        ]
        reliable = [
            int(_int(row.get("num_reliable")) or 0) for row in attempted
        ]
        share_one = (
            sum(1 for value in inliers if value <= 1) / len(inliers)
            if inliers
            else None
        )
        rows.append(
            [
                variant,
                len(attempted),
                _median(reliable),
                _median(inliers) if inliers else None,
                float(np.mean(inliers)) if inliers else None,
                share_one,
            ]
        )
        payload[variant] = {
            "frames_with_consensus": len(attempted),
            "num_reliable_median": _median(reliable),
            "inlier_count_median": _median(inliers) if inliers else None,
            "inlier_count_mean": float(np.mean(inliers)) if inliers else None,
            "inlier_count_histogram": {
                str(value): inliers.count(value) for value in sorted(set(inliers))
            },
            "share_inlier_count_at_most_one": share_one,
        }
    title = (
        "(a) consensus collapse: inlier_count over frames where a consensus "
        "was formed"
    )
    table = _table(
        [
            "variant",
            "frames",
            "reliable_med",
            "inlier_med",
            "inlier_mean",
            "frac_inlier<=1",
        ],
        rows,
    )
    baseline = payload.get(BASELINE_VARIANT, {})
    main = payload.get(MAIN_VARIANT, {})
    verdict = "inconclusive"
    if baseline.get("frames_with_consensus") and not main.get(
        "frames_with_consensus"
    ):
        verdict = (
            "supported by another route: the main variant never formed a "
            "consensus at all, while "
            f"{BASELINE_VARIANT} formed one on "
            f"{baseline['frames_with_consensus']} frames"
        )
    elif baseline and main:
        baseline_inlier = baseline.get("inlier_count_mean")
        main_inlier = main.get("inlier_count_mean")
        if baseline_inlier is not None and main_inlier is not None:
            verdict = (
                "supported: main variant collapses toward a single object "
                f"(mean inliers {main_inlier:.2f} vs {baseline_inlier:.2f})"
                if main_inlier < baseline_inlier - 0.25
                else "not supported: main variant keeps >=2 inliers on average "
                f"(mean inliers {main_inlier:.2f} vs {baseline_inlier:.2f})"
            )
    return f"{title}\n{table}", {
        "variants": payload,
        "baseline_variant": BASELINE_VARIANT,
        "main_variant": MAIN_VARIANT,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# (b) Feature predictiveness.
# ---------------------------------------------------------------------------


def feature_predictiveness(
    proposals: Sequence[Mapping[str, str]],
) -> tuple[str, dict[str, Any]]:
    usable = [
        row
        for row in proposals
        if _float(row.get("gt_translation_correction_error")) is not None
    ]
    target = _paired(_column(usable, "gt_translation_correction_error"))
    rows: list[list[Any]] = []
    payload: dict[str, Any] = {"usable_proposals": len(usable)}
    if len(usable) >= 8:
        for feature in PROPOSAL_FEATURES:
            values = _column(usable, feature)
            paired_mask = np.asarray(
                [value is not None for value in values], dtype=bool
            )
            if int(paired_mask.sum()) < 8:
                continue
            feature_values = np.asarray(
                [float(value) for value in values if value is not None],
                dtype=np.float64,
            )
            feature_target = target[paired_mask]
            # Constant features carry no ranking information.  Compare against
            # the scale, not against zero: a repeated value need not have an
            # exactly zero float standard deviation.
            scale = max(1.0, abs(float(feature_values.mean())))
            if float(feature_values.std()) <= 1e-9 * scale:
                continue
            rho, p_value = _spearman(feature_values, feature_target)
            if not math.isfinite(rho):
                continue
            order = np.argsort(feature_values, kind="stable")
            half = feature_values.shape[0] // 2
            low = feature_target[order[:half]]
            high = feature_target[order[half:]]
            split_p = _median_split_p(feature_values, feature_target)
            rows.append(
                [
                    feature,
                    feature_values.shape[0],
                    rho,
                    p_value,
                    _median(low),
                    _median(high),
                    split_p,
                ]
            )
            payload[feature] = {
                "n": int(feature_values.shape[0]),
                "spearman_rho": rho,
                "permutation_p": p_value,
                "median_gt_error_low_half_m": _median(low),
                "median_gt_error_high_half_m": _median(high),
                "median_split_p": split_p,
            }
        rows.sort(key=lambda row: abs(float(row[2])), reverse=True)
    title = (
        "(b) proposal feature vs GT translation correction error "
        "(rho>0 = larger feature means worse proposal)"
    )
    table = _table(
        [
            "feature",
            "n",
            "spearman_rho",
            "p_perm",
            "gt_med_low",
            "gt_med_high",
            "p_split",
        ],
        rows,
    )
    return f"{title}\n{table}", payload


def grouped_error_tables(
    proposals: Sequence[Mapping[str, str]],
) -> tuple[str, dict[str, Any]]:
    usable = [
        row
        for row in proposals
        if _float(row.get("gt_translation_correction_error")) is not None
    ]
    target = "gt_translation_correction_error"
    blocks = [
        f"(b2) GT translation correction error by geometry_type\n"
        + _table(
            ["geometry_type", "n", "median_m", "mean_m"],
            _grouped(usable, key="geometry_type", target=target),
        ),
        f"(b3) GT translation correction error by category\n"
        + _table(
            ["category", "n", "median_m", "mean_m"],
            _grouped(usable, key="category", target=target),
        ),
    ]
    payload = {
        "by_geometry_type": _grouped(usable, key="geometry_type", target=target),
        "by_category": _grouped(usable, key="category", target=target),
    }
    return "\n\n".join(blocks), payload


# ---------------------------------------------------------------------------
# (c) Does the selection filter select correctly?
# ---------------------------------------------------------------------------


def selection_quality(
    proposals: Sequence[Mapping[str, str]],
    consensus_rows: Sequence[Mapping[str, str]],
) -> tuple[str, dict[str, Any]]:
    target = "gt_translation_correction_error"
    with_error = [
        row for row in proposals if _float(row.get(target)) is not None
    ]

    def split(label: str, predicate) -> list[Any]:
        kept = [
            float(_float(row.get(target)))
            for row in with_error
            if predicate(row)
        ]
        dropped = [
            float(_float(row.get(target)))
            for row in with_error
            if not predicate(row)
        ]
        return [label, len(kept), _median(kept), len(dropped), _median(dropped)]

    proposal_rows = [
        split(
            "reliable (object_reject_reason empty)",
            lambda row: not (row.get("object_reject_reason") or "").strip(),
        ),
        split(
            "consensus_inlier",
            lambda row: _flag(row.get("consensus_inlier")) is True,
        ),
        split(
            "refiner accepted",
            lambda row: _flag(row.get("accepted_by_refiner")) is True,
        ),
    ]

    main_rows = [
        row
        for row in consensus_rows
        if (row.get("variant") or "").strip() == MAIN_VARIANT
    ]
    with_gt = [
        row for row in main_rows if _float(row.get("gt_consensus_translation_error")) is not None
    ]
    accepted = [
        row for row in with_gt if _flag(row.get("accepted")) is True
    ]
    rejected = [
        row for row in with_gt if _flag(row.get("accepted")) is False
    ]
    # Rejected frames only carry a GT consensus error from a revision that
    # records the consensus delta on post-consensus rejections.
    rejected_with_gt = [
        row
        for row in rejected
        if (_float(row.get("gt_consensus_translation_error")) is not None)
    ]
    frame_rows = [
        [
            "accepted",
            len(accepted),
            _median(
                float(_float(row.get("gt_consensus_translation_error")))
                for row in accepted
            ),
        ],
        [
            "rejected (post-consensus)",
            len(rejected_with_gt),
            _median(
                float(_float(row.get("gt_consensus_translation_error")))
                for row in rejected_with_gt
            ),
        ],
    ]
    frame_note = (
        ""
        if rejected_with_gt
        else (
            "\nNOTE: no rejected frame carries a GT consensus error. Re-run "
            "stage 2b on the current revision to record the consensus delta "
            "for post-consensus rejections; frame-level selection quality "
            "cannot be judged from this CSV."
        )
    )

    text = (
        "(c1) proposal-level selection: median GT error for kept vs dropped\n"
        + _table(
            ["filter", "kept_n", "kept_med_m", "dropped_n", "dropped_med_m"],
            proposal_rows,
        )
        + "\n\n(c2) frame-level selection, main variant: GT consensus error"
        + frame_note
        + "\n"
        + _table(["gate", "frames", "median_gt_consensus_error_m"], frame_rows)
    )
    payload = {
        "proposal_level": proposal_rows,
        "frame_level": {
            "accepted_frames": len(accepted),
            "rejected_with_gt_frames": len(rejected_with_gt),
            "accepted_median_gt_consensus_error_m": frame_rows[0][2],
            "rejected_median_gt_consensus_error_m": frame_rows[1][2],
            "rejected_gt_recorded": bool(rejected_with_gt),
        },
    }
    return text, payload


def per_frame_effect(
    frame_rows: Sequence[Mapping[str, str]],
) -> tuple[str, dict[str, Any]]:
    """At accepted frames, does the injected pose beat the raw pose locally?"""

    rows: list[list[Any]] = []
    payload: dict[str, Any] = {}
    for variant in sorted({(row.get("variant") or "").strip() for row in frame_rows}):
        if variant == "raw":
            continue
        subset = [
            row for row in frame_rows if (row.get("variant") or "").strip() == variant
        ]
        accepted = [row for row in subset if _flag(row.get("accepted")) is True]
        improved = [
            row
            for row in accepted
            if (_float(row.get("feedback_translation_error")) or 0.0)
            < (_float(row.get("raw_translation_error")) or 0.0)
        ]
        losses = [
            (_float(row.get("raw_translation_error")) or 0.0)
            - (_float(row.get("feedback_translation_error")) or 0.0)
            for row in accepted
        ]
        rows.append(
            [
                variant,
                len(accepted),
                len(improved),
                (len(improved) / len(accepted)) if accepted else None,
                _median(losses) if losses else None,
            ]
        )
        payload[variant] = {
            "accepted_frames": len(accepted),
            "accepted_frames_with_local_improvement": len(improved),
            "local_improvement_ratio": (
                len(improved) / len(accepted) if accepted else None
            ),
            "median_local_gain_m": _median(losses) if losses else None,
        }
    text = (
        "(d) at accepted frames: is the injected pose closer to GT than raw "
        "(same frame, before propagation)?"
    )
    return (
        text
        + "\n"
        + _table(
            [
                "variant",
                "accepted",
                "locally_better",
                "ratio",
                "median_gain_m",
            ],
            rows,
        ),
        payload,
    )


# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feedback-dir",
        type=Path,
        required=True,
        help="Directory holding the stage-2b CSVs (…/object_pose_feedback).",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Defaults to <feedback-dir>/attribution.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    feedback_dir = args.feedback_dir.expanduser().resolve()
    required = (
        "object_proposals.csv",
        "consensus_metrics.csv",
        "frame_metrics.csv",
    )
    missing = [name for name in required if not (feedback_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing stage-2b artifacts in {feedback_dir}: {missing}"
        )
    proposals = _read_rows(feedback_dir / "object_proposals.csv")
    consensus_rows = _read_rows(feedback_dir / "consensus_metrics.csv")
    frame_rows = _read_rows(feedback_dir / "frame_metrics.csv")
    for name, rows in (
        ("object_proposals.csv", proposals),
        ("consensus_metrics.csv", consensus_rows),
        ("frame_metrics.csv", frame_rows),
    ):
        if not rows:
            raise ValueError(f"{name} is empty; stage 2b produced no rows.")
        present = set(rows[0])
        absent = [column for column in REQUIRED_COLUMNS[name] if column not in present]
        if absent:
            raise ValueError(
                f"{name} is missing columns the attribution reads: {absent}. "
                "The stage-2b schema changed; update REQUIRED_COLUMNS and the "
                "analysis together."
            )

    sections: list[str] = []
    payload: dict[str, Any] = {
        "feedback_dir": str(feedback_dir),
        "proposal_row_count": len(proposals),
        "main_variant": MAIN_VARIANT,
        "baseline_variant": BASELINE_VARIANT,
        "notes": "GT is read only here; it never re-enters any decision.",
    }
    for name, (text, block) in (
        ("consensus_collapse", consensus_collapse(consensus_rows)),
        ("feature_predictiveness", feature_predictiveness(proposals)),
        ("grouped_error", grouped_error_tables(proposals)),
        ("selection_quality", selection_quality(proposals, consensus_rows)),
        ("per_frame_effect", per_frame_effect(frame_rows)),
    ):
        sections.append(text)
        payload[name] = block

    print("\n\n".join(sections))
    if payload.get("consensus_collapse", {}).get("verdict"):
        print("\nS_geo collapse verdict: " + str(payload["consensus_collapse"]["verdict"]))
    json_path = (
        args.json_out.expanduser().resolve()
        if args.json_out is not None
        else feedback_dir / "attribution.json"
    )
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"attribution_json={json_path}")


if __name__ == "__main__":
    main()
