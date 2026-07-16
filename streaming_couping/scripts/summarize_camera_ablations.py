"""Merge camera-refinement ablation summaries into one CSV."""

from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path


def main() -> None:
    args = _parse_args()
    input_paths = [Path(path) for path in sorted(glob.glob(args.input_glob))]
    if not input_paths:
        raise FileNotFoundError(
            f"No summary.csv files matched input glob: {args.input_glob}"
        )

    rows: list[dict[str, str]] = []
    fieldnames = ["source_summary", "experiment_name"]
    for path in input_paths:
        with path.open(newline="", encoding="utf8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                continue
            for fieldname in reader.fieldnames:
                if fieldname not in fieldnames:
                    fieldnames.append(fieldname)
            for row in reader:
                merged = dict(row)
                merged["source_summary"] = path.as_posix()
                merged["experiment_name"] = (
                    merged.get("experiment_name") or path.parent.name
                )
                rows.append(merged)

    if not rows:
        raise RuntimeError("Matched summary files contained no data rows.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"merged {len(rows)} rows from {len(input_paths)} summaries: {args.output}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-glob",
        default="outputs/camera_ablation_*/summary.csv",
        help="Glob matching per-experiment summary files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/camera_ablation_all_summary.csv"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
