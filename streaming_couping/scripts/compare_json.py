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


def _same(left: Any, right: Any) -> bool:
    """Equality that treats two non-finite floats as equal to each other."""

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


def _walk(left: Any, right: Any, path: str, out: list[tuple[str, Any, Any]]) -> None:
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            child = f"{path}.{key}" if path else str(key)
            if key not in left:
                out.append((child, "<absent>", right[key]))
            elif key not in right:
                out.append((child, left[key], "<absent>"))
            else:
                _walk(left[key], right[key], child, out)
        return
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            out.append((f"{path}[]", f"len={len(left)}", f"len={len(right)}"))
            return
        for index, (a, b) in enumerate(zip(left, right)):
            _walk(a, b, f"{path}[{index}]", out)
        return
    if not _same(left, right):
        out.append((path, left, right))


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
        default=20,
        help="Stop listing after this many differing paths.",
    )
    args = parser.parse_args()

    before = _load(args.before.expanduser().resolve(), args.prefix)
    after = _load(args.after.expanduser().resolve(), args.prefix)
    differences: list[tuple[str, Any, Any]] = []
    _walk(before, after, "", differences)
    if not differences:
        print(f"identical ({args.prefix or 'whole document'})")
        return
    print(f"{len(differences)} differing path(s) ({args.prefix or 'whole document'}):")
    for path, old, new in differences[: args.max_changes]:
        print(f"  {path}: {_render(old)} -> {_render(new)}")
    if len(differences) > args.max_changes:
        print(f"  ... and {len(differences) - args.max_changes} more")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
