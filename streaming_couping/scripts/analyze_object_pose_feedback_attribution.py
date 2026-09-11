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

# The claimed method, and the variant it is compared against in the collapse
# check: whether adding the geometric reliability term collapses the consensus.
MAIN_VARIANT = "robust_semantic"
BASELINE_VARIANT = "robust_semantic_geometric"
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

#: Features the per-category breakdown reports: the two reliability scores the
#: consensus weights are built from, plus the two that feed them.
CATEGORY_STRATIFIED_FEATURES: tuple[str, ...] = (
    "track_length",
    "semantic_confidence",
    "geometry_confidence",
    "inlier_ratio",
    "overlap_count",
    "alignment_loss_before",
    "alignment_loss_after",
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


def _group_masks(labels: Sequence[str]) -> list[np.ndarray]:
    buckets: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        buckets.setdefault(label, []).append(index)
    return [np.asarray(values, dtype=int) for values in buckets.values()]


def _centered_within_ranks(
    values: np.ndarray, groups: Sequence[np.ndarray]
) -> np.ndarray:
    """Ranks computed inside each group, then centred on the group mean."""

    ranks = np.zeros(values.shape[0], dtype=np.float64)
    for group in groups:
        group_ranks = _ranks(values[group])
        ranks[group] = group_ranks - group_ranks.mean()
    return ranks


def _stratified_spearman(
    features: np.ndarray,
    target: np.ndarray,
    groups: Sequence[np.ndarray],
) -> tuple[float, float]:
    """Spearman within each stratum, pooled; the permutation keeps strata.

    Centring each group's ranks removes between-group differences, so this
    measures the association that survives with the grouping variable held
    fixed -- the test for "is this feature predictive, or merely a proxy for
    the group?".  Returns NaN when the feature has no within-group variation.
    """

    if len(groups) < 2:
        return float("nan"), float("nan")
    feature_ranks = _centered_within_ranks(features, groups)
    target_ranks = _centered_within_ranks(target, groups)
    if float(feature_ranks.std()) <= 0.0:
        return float("nan"), float("nan")
    rho = _pearson(feature_ranks, target_ranks)
    if not math.isfinite(rho):
        return float("nan"), float("nan")
    generator = np.random.default_rng(SEED)
    extreme = 0
    threshold = abs(rho) - 1e-12
    shuffled = target_ranks.copy()
    for _ in range(PERMUTATION_ITERATIONS):
        for group in groups:
            shuffled[group] = generator.permutation(target_ranks[group])
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
    categories = [
        (row.get("category") or "<empty>").strip() or "<empty>" for row in usable
    ]
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
            # Does the association survive holding the object category fixed?
            feature_categories = [
                label
                for label, keep in zip(categories, paired_mask)
                if keep
            ]
            groups = _group_masks(feature_categories)
            within_rho, within_p = _stratified_spearman(
                feature_values, feature_target, groups
            )
            rows.append(
                [
                    feature,
                    feature_values.shape[0],
                    rho,
                    p_value,
                    within_rho,
                    within_p,
                    _median(low),
                    _median(high),
                    split_p,
                ]
            )
            payload[feature] = {
                "n": int(feature_values.shape[0]),
                "spearman_rho": rho,
                "permutation_p": p_value,
                "spearman_rho_within_category": within_rho,
                "permutation_p_within_category": within_p,
                "median_gt_error_low_half_m": _median(low),
                "median_gt_error_high_half_m": _median(high),
                "median_split_p": split_p,
            }
        rows.sort(key=lambda row: abs(float(row[2])), reverse=True)
    title = (
        "(b) proposal feature vs GT translation correction error "
        "(rho>0 = larger feature means worse proposal)\n"
        "    rho_within = same correlation computed inside each category, "
        "holding the category fixed"
    )
    table = _table(
        [
            "feature",
            "n",
            "rho_marginal",
            "p_perm",
            "rho_within",
            "p_within",
            "gt_med_low",
            "gt_med_high",
            "p_split",
        ],
        rows,
    )
    return f"{title}\n{table}", payload


# ---------------------------------------------------------------------------
# Reference staleness: is the proposal noisy because it aligns against a
# reference set that is far away in time?
#
# Age is ``frame - reference_frame``, so a larger value means an older
# reference.  ``newest`` is the shortest baseline in the set, ``oldest`` the
# longest.  The refiner weights anchors (frames < anchor_frame_count, fixed at
# the start of the sequence) at 1.0 and recent history at 0.5, so splitting by
# role says whether the stale half of the set is the one driving the error.
# ---------------------------------------------------------------------------

REFERENCE_AGE_FEATURES: tuple[str, ...] = (
    "newest_reference_age",
    "oldest_reference_age",
    "mean_reference_age",
)

ANCHOR_ROLE = "anchor"


def _parse_int_list(value: Any) -> list[int]:
    text = "" if value is None else str(value).strip()
    if not text:
        return []
    frames: list[int] = []
    for part in text.split(";"):
        part = part.strip()
        if not part:
            continue
        try:
            frames.append(int(float(part)))
        except ValueError:
            return []
    return frames


def _parse_str_list(value: Any) -> list[str]:
    text = "" if value is None else str(value).strip()
    if not text:
        return []
    return [part.strip() for part in text.split(";")]


def reference_age_features(row: Mapping[str, str]) -> dict[str, float] | None:
    """Ages of the reference set for one proposal, or None when unusable."""

    frames = _parse_int_list(row.get("reference_frames"))
    frame = _float(row.get("frame"))
    if not frames or frame is None:
        return None
    features = {
        "newest_reference_age": frame - max(frames),
        "oldest_reference_age": frame - min(frames),
        "mean_reference_age": frame - sum(frames) / len(frames),
        "reference_count": float(len(frames)),
    }
    # Role split is only available once the writer records reference_roles.
    roles = _parse_str_list(row.get("reference_roles"))
    if len(roles) == len(frames):
        anchors = [f for f, role in zip(frames, roles) if role == ANCHOR_ROLE]
        history = [f for f, role in zip(frames, roles) if role != ANCHOR_ROLE]
        if anchors:
            features["anchor_reference_age"] = frame - max(anchors)
        if history:
            features["history_reference_age"] = frame - max(history)
    return features


def _age_matrix(
    proposals: Sequence[Mapping[str, str]],
) -> tuple[dict[str, list[float | None]], np.ndarray, list[str]]:
    """Derived-age columns aligned with the target, plus category labels."""

    usable = [
        row
        for row in proposals
        if _float(row.get("gt_translation_correction_error")) is not None
    ]
    target = _paired(_column(usable, "gt_translation_correction_error"))
    categories = [
        (row.get("category") or "<empty>").strip() or "<empty>" for row in usable
    ]
    parsed = [reference_age_features(row) for row in usable]
    names = list(REFERENCE_AGE_FEATURES)
    for optional in ("anchor_reference_age", "history_reference_age"):
        if any(entry is not None and optional in entry for entry in parsed):
            names.append(optional)
    columns = {
        name: [None if entry is None else entry.get(name) for entry in parsed]
        for name in names
    }
    return columns, target, categories


def _quantile_buckets(values: np.ndarray, *, buckets: int = 4) -> list[np.ndarray]:
    """Row indices of roughly equal-count buckets ordered by value."""

    order = np.argsort(values, kind="stable")
    edges = np.linspace(0, values.shape[0], buckets + 1).astype(int)
    return [order[edges[i] : edges[i + 1]] for i in range(buckets)]


def reference_staleness(
    proposals: Sequence[Mapping[str, str]],
) -> tuple[str, dict[str, Any]]:
    """Correlate reference age with proposal error, plus a bucket table."""

    columns, target, categories = _age_matrix(proposals)
    rows: list[list[Any]] = []
    payload: dict[str, Any] = {}
    for name, values in columns.items():
        paired = np.asarray(
            [index for index, value in enumerate(values) if value is not None],
            dtype=int,
        )
        if paired.shape[0] < 8:
            continue
        feature_values = np.asarray(
            [float(values[index]) for index in paired], dtype=np.float64
        )
        feature_target = target[paired]
        if float(feature_values.std()) <= 1e-9:
            continue
        rho, p_value = _spearman(feature_values, feature_target)
        within_rho, within_p = _stratified_spearman(
            feature_values,
            feature_target,
            _group_masks([categories[index] for index in paired]),
        )
        rows.append(
            [
                name,
                feature_values.shape[0],
                _median(feature_values.tolist()),
                rho,
                p_value,
                within_rho,
                within_p,
            ]
        )
        payload[name] = {
            "n": int(feature_values.shape[0]),
            "median_age_frames": _median(feature_values.tolist()),
            "spearman_rho": rho,
            "permutation_p": p_value,
            "spearman_rho_within_category": within_rho,
            "permutation_p_within_category": within_p,
        }
    if not rows:
        return "(b6) reference staleness: no usable reference_frames column", payload

    rows.sort(key=lambda row: abs(float(row[3])), reverse=True)
    text = (
        "(b6) reference staleness: age = frame - reference_frame "
        "(larger = older reference)\n"
        "     newest_reference_age is the SHORTEST baseline in the set, "
        "oldest_reference_age the longest"
    )
    text += "\n" + _table(
        [
            "feature",
            "n",
            "median_age",
            "rho_marginal",
            "p_perm",
            "rho_within",
            "p_within",
        ],
        rows,
    )

    # Bucket view: monotonic rise in GT error across age quartiles is the
    # readable form of the same claim.
    bucket_feature = "oldest_reference_age"
    bucket_rows: list[list[Any]] = []
    values = columns.get(bucket_feature)
    if values is not None:
        paired = np.asarray(
            [index for index, value in enumerate(values) if value is not None],
            dtype=int,
        )
        if paired.shape[0] >= 16:
            feature_values = np.asarray(
                [float(values[index]) for index in paired], dtype=np.float64
            )
            feature_target = target[paired]
            for bucket in _quantile_buckets(feature_values):
                bucket_rows.append(
                    [
                        _median(feature_values[bucket].tolist()),
                        bucket.shape[0],
                        _median(feature_target[bucket].tolist()),
                    ]
                )
            payload[f"{bucket_feature}_quartiles"] = bucket_rows
    if bucket_rows:
        text += (
            "\n\n(b7) GT correction error by oldest-reference-age quartile "
            "(Q1 = freshest reference set)"
        )
        text += "\n" + _table(
            ["median_age", "n", "median_gt_error_m"], bucket_rows
        )
    return text, payload


def category_stratified_correlations(
    proposals: Sequence[Mapping[str, str]],
) -> tuple[str, dict[str, Any]]:
    """Per-category Spearman rho for the features the weights depend on."""

    usable = [
        row
        for row in proposals
        if _float(row.get("gt_translation_correction_error")) is not None
    ]
    target = _paired(_column(usable, "gt_translation_correction_error"))
    categories = [
        (row.get("category") or "<empty>").strip() or "<empty>" for row in usable
    ]
    table_rows: list[list[Any]] = []
    payload: dict[str, Any] = {}
    for label in sorted(set(categories)):
        mask = np.asarray([c == label for c in categories], dtype=bool)
        row: list[Any] = [label, int(mask.sum())]
        entry: dict[str, Any] = {"n": int(mask.sum())}
        for feature in CATEGORY_STRATIFIED_FEATURES:
            values_all = _column(usable, feature)
            # Pair feature and target on the rows where the feature is present:
            # a feature with any missing value in this category must still be
            # scored on the rows that remain, not dropped wholesale.
            present = [
                index
                for index, (keep, value) in enumerate(zip(mask, values_all))
                if keep and value is not None
            ]
            if len(present) < 6:
                row.append(None)
                entry[feature] = None
                continue
            values = np.asarray(
                [float(values_all[index]) for index in present], dtype=np.float64
            )
            feature_target = np.asarray(
                [float(target[index]) for index in present], dtype=np.float64
            )
            if float(values.std()) <= 1e-9:
                row.append(None)
                entry[feature] = None
                continue
            rho, p_value = _spearman(values, feature_target)
            row.append(rho)
            entry[feature] = {
                "n": len(present),
                "rho": rho,
                "p": p_value,
            }
        table_rows.append(row)
        payload[label] = entry
    text = (
        "(b5) per-category Spearman rho vs GT translation correction error\n"
        "     (a category proxy drops to ~0 here while (b)'s rho_marginal "
        "stays large; 'n' is the category size and a feature missing values "
        "is scored on its own paired subset -- see the json for its n)"
    )
    return (
        text
        + "\n"
        + _table(
            ["category", "n", *CATEGORY_STRATIFIED_FEATURES],
            table_rows,
        ),
        payload,
    )


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
        (
            "category_stratified_correlations",
            category_stratified_correlations(proposals),
        ),
        ("reference_staleness", reference_staleness(proposals)),
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
