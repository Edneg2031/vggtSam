"""Run the controlled camera-refinement experiment for multiple instances."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


SINGLE_EXPERIMENT_MODULE = (
    "streaming_couping.scripts.run_sam3_mask_camera_refinement"
)


def main() -> None:
    args, experiment_args = _parse_args()
    output_root = args.output_dir.expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    statuses: list[dict] = []
    summary_paths: list[Path] = []
    for instance_id in args.instance_ids:
        instance_output = output_root / f"instance_{instance_id}"
        command = [
            sys.executable,
            "-m",
            SINGLE_EXPERIMENT_MODULE,
            "--config",
            str(args.config),
            "--instance-id",
            str(instance_id),
            "--output-dir",
            str(instance_output),
            *experiment_args,
        ]
        print("\n" + "=" * 72, flush=True)
        print(
            f"Running instance_id={instance_id} output={instance_output}",
            flush=True,
        )
        print("=" * 72, flush=True)
        result = subprocess.run(command, check=False)
        summary_path = instance_output / "summary.csv"
        success = result.returncode == 0 and summary_path.is_file()
        statuses.append(
            {
                "instance_id": instance_id,
                "status": "completed" if success else "failed",
                "return_code": result.returncode,
                "output_dir": instance_output.as_posix(),
                "summary": summary_path.as_posix() if success else "",
            }
        )
        if success:
            summary_paths.append(summary_path)
        elif not args.continue_on_error:
            break

    _write_csv(output_root / "batch_status.csv", statuses)
    with (output_root / "batch_settings.json").open("w", encoding="utf8") as handle:
        json.dump(
            {
                "instance_ids": args.instance_ids,
                "config": str(args.config),
                "forwarded_experiment_args": experiment_args,
            },
            handle,
            indent=2,
        )

    if summary_paths:
        merged_rows = _merge_summaries(summary_paths)
        _write_csv(output_root / "summary_all_instances.csv", merged_rows)
        labels = {
            (row.get("instance_id", ""), row.get("instance_label", ""))
            for row in merged_rows
        }
        print(
            f"\nmerged {len(merged_rows)} rows for instances "
            f"{sorted(labels)}: {output_root / 'summary_all_instances.csv'}"
        )

    failed = [row for row in statuses if row["status"] != "completed"]
    if failed:
        failed_ids = ", ".join(str(row["instance_id"]) for row in failed)
        raise SystemExit(
            f"Batch completed with failed instance IDs: {failed_ids}. "
            f"See {output_root / 'batch_status.csv'}."
        )


def _merge_summaries(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="", encoding="utf8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                merged = dict(row)
                merged["source_summary"] = path.as_posix()
                rows.append(merged)
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for fieldname in row:
            if fieldname not in fieldnames:
                fieldnames.append(fieldname)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "All unrecognized options are forwarded unchanged to the single-instance "
            "camera-refinement experiment."
        ),
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--instance-ids", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the remaining instances after one instance fails.",
    )
    args, experiment_args = parser.parse_known_args()
    if len(set(args.instance_ids)) != len(args.instance_ids):
        parser.error("--instance-ids must not contain duplicates")
    forbidden = {"--instance-id", "--instance-ids", "--output-dir", "--config"}
    conflicts = sorted(forbidden.intersection(experiment_args))
    if conflicts:
        parser.error(
            "Batch-owned options cannot be forwarded: " + ", ".join(conflicts)
        )
    return args, experiment_args


if __name__ == "__main__":
    main()
