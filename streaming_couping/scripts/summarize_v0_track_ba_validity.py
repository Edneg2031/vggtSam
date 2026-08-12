#!/usr/bin/env python3
"""Print and gate the V0 Track-BA current-frame validity audit."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


REQUIRED_METHODS = ("full_image", "sam_dynamic_excluded")
COUNT_FIELDS = (
    "current_query_count",
    "current_geometry_valid_count",
    "current_track_finite_count",
    "current_track_in_bounds_count",
    "current_visibility_pass_count",
    "current_confidence_pass_count",
    "current_track_gate_pass_count",
    "current_valid_after_geometry_count",
)


def main() -> None:
    args = _parse_args()
    rows = _read_rows(args.csv)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)

    print("V0 Track-BA current-validity factor audit")
    for method, items in grouped.items():
        totals = _totals(items)
        query_total = totals["current_query_count"]
        print(f"\nmethod={method}")
        print(
            "  aggregate "
            f"finite={totals['current_track_finite_count']}/{query_total} "
            f"bounds={totals['current_track_in_bounds_count']}/{query_total} "
            f"vis={totals['current_visibility_pass_count']}/{query_total} "
            f"conf={totals['current_confidence_pass_count']}/{query_total} "
            f"track_gate={totals['current_track_gate_pass_count']}/"
            f"{query_total} "
            f"geometry={totals['current_geometry_valid_count']}/{query_total} "
            f"valid={totals['current_valid_after_geometry_count']}/"
            f"{query_total} "
            f"diagnosis={_diagnosis(totals)}"
        )
        for row in items:
            print(
                "  "
                f"frame={row['frame_index']} "
                f"geometry={row['current_geometry_valid_count']}/"
                f"{row['current_query_count']} "
                f"finite={row['current_track_finite_count']} "
                f"bounds={row['current_track_in_bounds_count']} "
                f"vis={row['current_visibility_pass_count']} "
                f"conf={row['current_confidence_pass_count']} "
                f"track_gate={row['current_track_gate_pass_count']} "
                f"valid={row['current_valid_after_geometry_count']} "
                f"x={row['current_track_x_min']}.."
                f"{row['current_track_x_max']} "
                f"y={row['current_track_y_min']}.."
                f"{row['current_track_y_max']} "
                f"vis_range={row['current_visibility_min']}.."
                f"{row['current_visibility_max']} "
                f"conf_range={row['current_confidence_min']}.."
                f"{row['current_confidence_max']}"
            )

    gate, reason = _gate(
        grouped,
        required_methods=REQUIRED_METHODS,
        expected_frames=args.expected_frames,
        minimum_correspondences=args.minimum_correspondences,
    )
    print(
        "\nV0 Track-BA reaggregated validity gate "
        f"pass={int(gate)} reason={reason}"
    )
    if args.require_pass and not gate:
        raise SystemExit(2)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty Track-BA validity audit: {path}")
    missing = set(COUNT_FIELDS) - set(rows[0])
    if missing:
        raise ValueError(f"Validity audit lacks fields={sorted(missing)}.")
    sources = {row.get("track_token_source", "") for row in rows}
    if sources != {"window_reaggregated"}:
        raise ValueError(
            "Validity audit is stale or has the wrong token source: "
            f"{sorted(sources)}."
        )
    return rows


def _totals(items: list[dict[str, str]]) -> dict[str, int]:
    return {
        name: sum(int(row[name]) for row in items)
        for name in COUNT_FIELDS
    }


def _diagnosis(totals: dict[str, int]) -> str:
    if totals["current_track_finite_count"] == 0:
        return "nonfinite_track_output"
    if totals["current_track_in_bounds_count"] == 0:
        return "coordinate_scale_or_reanchoring_failure"
    if totals["current_visibility_pass_count"] == 0:
        return "visibility_gate_collapse"
    if totals["current_confidence_pass_count"] == 0:
        return "confidence_gate_collapse"
    if totals["current_track_gate_pass_count"] == 0:
        return "gate_intersection_collapse"
    if totals["current_valid_after_geometry_count"] == 0:
        return "track_geometry_intersection_collapse"
    return "current_tracks_survive"


def _gate(
    grouped: dict[str, list[dict[str, str]]],
    *,
    required_methods: tuple[str, ...],
    expected_frames: int,
    minimum_correspondences: int,
) -> tuple[bool, str]:
    for method in required_methods:
        items = grouped.get(method, [])
        if len(items) != int(expected_frames):
            return False, f"{method}_rows_{len(items)}_expected_{expected_frames}"
        failures = [
            row["frame_index"]
            for row in items
            if int(row["current_valid_after_geometry_count"])
            < int(minimum_correspondences)
        ]
        if failures:
            return False, f"{method}_below_min_frames_{'_'.join(failures)}"
    return True, "both_locked_methods_all_frames_meet_min_correspondences"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--expected-frames", type=int, default=12)
    parser.add_argument("--minimum-correspondences", type=int, default=32)
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
