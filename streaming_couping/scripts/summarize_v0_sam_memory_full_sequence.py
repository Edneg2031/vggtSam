#!/usr/bin/env python3
"""Summarize the completed training-free retrieval run over the full stream."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


REVISION = "v0_sam_memory_full_sequence_pose_eval_r1"


def main() -> None:
    args = _parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    source_summary = _read_json(input_dir / "retrieval_summary.json")
    rows = _read_csv(input_dir / "frame_metrics.csv")
    result = summarize(
        source_summary,
        rows,
        reference_sequence_index=int(args.reference_sequence_index),
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else input_dir / "full_sequence_eval"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "full_sequence_summary.csv", result["method_results"])
    (output_dir / "full_sequence_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf8",
    )
    _write_copyable(output_dir / "copyable_result.txt", result)
    print("V0 TRAINING-FREE FULL-SEQUENCE POSE EVALUATION")
    for row in result["method_results"]:
        if row["method"] == "raw_full_history":
            continue
        print(
            f"  method={row['method']} "
            f"center_gain={float(row['center_gain_percent']):.4f}% "
            f"R_gain={float(row['rotation_gain_percent']):.4f}% "
            f"pose_pass={row['full_sequence_pose_pass']}"
        )
    print(f"  decision={json.dumps(result['decision'], sort_keys=True)}")
    print(f"  copyable_report={output_dir / 'copyable_result.txt'}")


def summarize(
    source_summary: dict,
    rows: list[dict[str, str]],
    *,
    reference_sequence_index: int,
) -> dict:
    if not int(source_summary.get("raw_replay_equivalence", {}).get("pass", 0)):
        raise ValueError("The source retrieval run did not reproduce raw StreamVGGT.")
    methods = tuple(str(value) for value in source_summary.get("methods", ()))
    if not methods or "raw_full_history" not in methods:
        raise ValueError("The source retrieval summary lacks the locked methods.")
    grouped = {method: [] for method in methods}
    for row in rows:
        method = str(row.get("method", ""))
        if method not in grouped:
            raise ValueError(f"Unexpected method in frame metrics: {method!r}.")
        grouped[method].append(row)
    frame_indices = tuple(int(value) for value in source_summary.get("frames", ()))
    if not frame_indices:
        raise ValueError("The source summary contains no stream frames.")
    expected_sequence_indices = tuple(range(len(frame_indices)))
    for method, method_rows in grouped.items():
        method_rows.sort(key=lambda row: int(row["sequence_index"]))
        observed = tuple(int(row["sequence_index"]) for row in method_rows)
        if observed != expected_sequence_indices:
            raise ValueError(
                f"Method {method!r} does not contain exactly the full stream: "
                f"expected={expected_sequence_indices} observed={observed}."
            )
    reference = int(reference_sequence_index)
    if not 0 <= reference < len(frame_indices):
        raise ValueError("reference_sequence_index is outside the stream.")
    evaluation_indices = tuple(
        index for index in expected_sequence_indices if index != reference
    )
    raw = grouped["raw_full_history"]
    raw_center = _mean(raw, evaluation_indices, "center_error_native")
    raw_rotation = _mean(raw, evaluation_indices, "rotation_error_degrees")
    raw_point = _mean(raw, evaluation_indices, "pointmap_paired_rmse")

    method_results = []
    for method in methods:
        current = grouped[method]
        center = _mean(current, evaluation_indices, "center_error_native")
        rotation = _mean(
            current,
            evaluation_indices,
            "rotation_error_degrees",
        )
        point = _mean(current, evaluation_indices, "pointmap_paired_rmse")
        center_gain = _gain(raw_center, center)
        rotation_gain = _gain(raw_rotation, rotation)
        point_gain = _gain(raw_point, point)
        method_results.append(
            {
                "method": method,
                "stream_frames": len(frame_indices),
                "evaluated_pose_frames": len(evaluation_indices),
                "reference_frame": frame_indices[reference],
                "raw_center_error_native": raw_center,
                "candidate_center_error_native": center,
                "center_gain_percent": center_gain,
                "center_worse_frames": _worse_count(
                    raw,
                    current,
                    evaluation_indices,
                    "center_error_native",
                ),
                "raw_rotation_degrees": raw_rotation,
                "candidate_rotation_degrees": rotation,
                "rotation_gain_percent": rotation_gain,
                "rotation_worse_frames": _worse_count(
                    raw,
                    current,
                    evaluation_indices,
                    "rotation_error_degrees",
                ),
                "raw_pointmap_rmse_diagnostic": raw_point,
                "candidate_pointmap_rmse_diagnostic": point,
                "pointmap_gain_percent_diagnostic": point_gain,
                "full_sequence_pose_pass": int(
                    method != "raw_full_history"
                    and center_gain > 0.0
                    and rotation_gain > 0.0
                ),
            }
        )

    primary = str(source_summary.get("primary_method", "sam_hybrid_qk"))
    primary_rows = [row for row in method_results if row["method"] == primary]
    if len(primary_rows) != 1:
        raise ValueError(f"Primary method {primary!r} was not found exactly once.")
    source_decision = source_summary.get("decision", {})
    decision = {
        "primary_method": primary,
        "primary_full_sequence_pose_pass": int(
            primary_rows[0]["full_sequence_pose_pass"]
        ),
        "pipeline_claim": "single_sequence_training_free_pose_help",
        "sam_identity_causal_claim": int(
            source_decision.get("sam_memory_retrieval_causal_pass", 0)
        ),
        "pointmap_head_is_pose_gate": 0,
        "semantic_map_with_shared_raw_depth_k_evaluated": 0,
        "selected_v0_pose_modified": 0,
    }
    return {
        "schema": 1,
        "revision": REVISION,
        "source_revision": source_summary.get("revision"),
        "clip": source_summary.get("clip"),
        "stream_frames": len(frame_indices),
        "frame_indices": frame_indices,
        "reference_sequence_index": reference,
        "reference_frame": frame_indices[reference],
        "evaluated_pose_frames": len(evaluation_indices),
        "evaluation_frame_indices": tuple(
            frame_indices[index] for index in evaluation_indices
        ),
        "model_trained": 0,
        "candidate_generation_gt_fields": 0,
        "gt_role": "aggregate_scoring_only_after_completed_candidates",
        "window_role": "diagnostic_only_not_train_test_folds",
        "method_results": method_results,
        "decision": decision,
    }


def _mean(
    rows: list[dict[str, str]],
    indices: tuple[int, ...],
    field: str,
) -> float:
    values = [float(rows[index][field]) for index in indices]
    finite = [value for value in values if math.isfinite(value)]
    if len(finite) != len(values) or not finite:
        raise ValueError(f"Field {field!r} contains missing/non-finite values.")
    return sum(finite) / len(finite)


def _worse_count(
    raw: list[dict[str, str]],
    candidate: list[dict[str, str]],
    indices: tuple[int, ...],
    field: str,
) -> int:
    return sum(
        float(candidate[index][field]) > float(raw[index][field])
        for index in indices
    )


def _gain(raw: float, candidate: float) -> float:
    if not math.isfinite(raw) or not math.isfinite(candidate) or raw <= 0:
        raise ValueError("Cannot compute a finite percentage gain.")
    return 100.0 * (raw - candidate) / raw


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty full-sequence summary.")
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_copyable(path: Path, result: dict) -> None:
    rows = result["method_results"]
    columns = tuple(rows[0])
    lines = [
        "===== COPYABLE_V0_FULL_SEQUENCE_POSE_BEGIN =====",
        f"revision={REVISION}",
        f"source_revision={result['source_revision']}",
        f"clip={result['clip']}",
        f"stream_frames={result['stream_frames']}",
        f"reference_frame={result['reference_frame']}",
        f"evaluated_pose_frames={result['evaluated_pose_frames']}",
        "model_trained=0",
        "window_role=diagnostic_only_not_train_test_folds",
        "pointmap_role=diagnostic_not_pose_acceptance_gate",
        "",
        ",".join(columns),
    ]
    for row in rows:
        lines.append(",".join(str(row[column]) for column in columns))
    lines.extend(
        (
            "",
            f"decision={json.dumps(result['decision'], sort_keys=True)}",
            "",
            "outputs:",
            f"summary={path.parent / 'full_sequence_summary.json'}",
            f"csv={path.parent / 'full_sequence_summary.csv'}",
            f"copyable_report={path}",
            "===== COPYABLE_V0_FULL_SEQUENCE_POSE_END =====",
        )
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default="outputs/streaming_couping_v0/sam_memory_retrieval",
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--reference-sequence-index", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main()
