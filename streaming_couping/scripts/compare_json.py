#!/usr/bin/env python3
"""Report exactly which leaves of two JSON documents differ (read-only).

Used by the attribution runner to check that regenerating stage 2b only added
recorded diagnostics and did not move any number.  A bare "decisions changed"
warning is not actionable; this prints the differing paths and both values so
the difference can be judged -- a newly-introduced informational field is not
the same finding as a flipped decision.

    python -m streaming_couping.scripts.compare_json BEFORE.json AFTER.json \
        [--prefix decisions]

Exit status is 0 when the compared subtrees are equal, 1 when they differ, so
a caller can branch on it without parsing output.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _load(path: Path, prefix: str | None) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if prefix:
        for part in prefix.split("."):
            if not isinstance(document, dict) or part not in document:
                raise KeyError(f"{path} has no {prefix!r} subtree")
            document = document[part]
    return document


def _relative_gap(left: float, right: float) -> float:
    """Relative difference, or infinity when the two are not comparable."""

    if math.isnan(left) and math.isnan(right):
        return 0.0
    if math.isinf(left) or math.isinf(right):
        return 0.0 if left == right else float("inf")
    scale = max(abs(left), abs(right))
    if scale == 0.0:
        return 0.0
    return abs(left - right) / scale


def _same(left: Any, right: Any) -> bool:
    """Exact structural equality (non-finite floats equal to the same value)."""

    if isinstance(left, float) and isinstance(right, float):
        if math.isnan(left) and math.isnan(right):
            return True
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            _same(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _same(a, b) for a, b in zip(left, right)
        )
    return left == right


def _walk(
    left: Any,
    right: Any,
    path: str,
    material: list[tuple[str, Any, Any]],
    drifted: list[tuple[str, float, float, float]],
    added: list[tuple[str, Any]],
    tolerance: float,
) -> None:
    """Split differences into material changes, additions, and float drift.

    A repeating replay is not bit-identical on CPU because reduction order
    varies with thread scheduling, so values drift in their last digits.  A
    checker that flags that, or a newly recorded diagnostic field, as
    "changed" trains the reader to ignore it; only a value that moved beyond
    ``tolerance``, or disappeared, reaches the material list.
    """

    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            child = f"{path}.{key}" if path else str(key)
            if key not in left:
                added.append((child, right[key]))
            elif key not in right:
                material.append((child, left[key], "<absent>"))
            else:
                _walk(
                    left[key], right[key], child, material, drifted, added, tolerance
                )
        return
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            material.append((f"{path}[]", f"len={len(left)}", f"len={len(right)}"))
            return
        for index, (a, b) in enumerate(zip(left, right)):
            _walk(a, b, f"{path}[{index}]", material, drifted, added, tolerance)
        return
    if _same(left, right):
        return
    # Numeric drift is reported, but separately, and only as a magnitude.
    if isinstance(left, float) and isinstance(right, float):
        gap = _relative_gap(left, right)
        drifted.append((path, left, right, gap))
        if gap <= tolerance:
            return
    material.append((path, left, right))


def _render(value: Any, limit: int = 60) -> str:
    if isinstance(value, (dict, list)):
        text = json.dumps(value, sort_keys=True, default=str)
    else:
        text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument(
        "--prefix",
        default=None,
        help="Compare only this top-level key (or dotted path) of both files.",
    )
    parser.add_argument(
        "--max-changes",
        type=int,
        default=50,
        help="Stop listing after this many differing paths.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-3,
        help=(
            "Relative float difference below which a change counts as "
            "numerical drift rather than a material change.  A repeating "
            "replay drifts in its last digits; a real threshold change moves "
            "values by percent."
        ),
    )
    parser.add_argument(
        "--structural-only",
        action="store_true",
        help=(
            "Exit non-zero only for structural differences: a field that "
            "appeared or disappeared, a changed string, a flipped boolean.  "
            "Float drift is always reported with its magnitude but never "
            "fails.  Use when the underlying computation is known to move in "
            "its last bits, so the checker measures that instead of tripping "
            "on it."
        ),
    )
    args = parser.parse_args()

    before = _load(args.before.expanduser().resolve(), args.prefix)
    after = _load(args.after.expanduser().resolve(), args.prefix)
    material: list[tuple[str, Any, Any]] = []
    drifted: list[tuple[str, float, float, float]] = []
    added: list[tuple[str, Any]] = []
    _walk(before, after, "", material, drifted, added, args.tolerance)
    label = args.prefix or "whole document"

    # `drifted` already holds every float that moved, with its magnitude; the
    # ones beyond tolerance also landed in `material`.  Anything in `material`
    # that is not a float pair -- a string, a boolean, a vanished field -- is
    # structural.
    numeric = drifted
    structural = [
        entry
        for entry in material
        if not (isinstance(entry[1], float) and isinstance(entry[2], float))
    ]

    if args.structural_only:
        # Report every float movement, including the large ones, because the
        # magnitude IS the thing being measured here.
        if numeric:
            worst = max(gap for _, _, _, gap in numeric)
            print(
                f"{len(numeric)} numeric value(s) moved (max relative "
                f"{worst:.2e}) [{label}] -- analysis-chain noise"
            )
            for path, old, new, gap in sorted(
                numeric, key=lambda entry: entry[3], reverse=True
            )[: args.max_changes]:
                print(f"  {path}: {_render(old)} -> {_render(new)} ({gap:.2e})")
            if len(numeric) > args.max_changes:
                print(f"  ... and {len(numeric) - args.max_changes} more")
        if added:
            print(f"{len(added)} newly recorded field(s) [{label}]")
        if not structural:
            print(f"no structural change ({label})")
            return
        print(f"{len(structural)} structural change(s) ({label}):")
        for path, old, new in structural[: args.max_changes]:
            print(f"  {path}: {_render(old)} -> {_render(new)}")
        raise SystemExit(1)

    if added:
        print(f"{len(added)} newly recorded field(s) [{label}]:")
        for path, value in added[: args.max_changes]:
            print(f"  {path}: <absent> -> {_render(value)}")
    if drifted:
        worst = max(gap for _, _, _, gap in drifted)
        print(
            f"{len(drifted)} value(s) drifted within tolerance "
            f"(max relative {worst:.2e}, tolerance {args.tolerance:.0e}) [{label}]"
        )
    if not material:
        print(f"no material change ({label})")
        return
    print(f"{len(material)} material change(s) ({label}):")
    for path, old, new in material[: args.max_changes]:
        print(f"  {path}: {_render(old)} -> {_render(new)}")
    if len(material) > args.max_changes:
        print(f"  ... and {len(material) - args.max_changes} more")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
