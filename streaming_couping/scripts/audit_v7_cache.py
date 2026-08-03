#!/usr/bin/env python3
"""Audit frozen V7/V7.2 caches without loading either backbone."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch

from streaming_couping.src.learned_pose.cache import (
    CACHE_VERSION,
    cache_path,
    load_feature_cache,
)
from streaming_couping.src.learned_pose.config import load_learned_pose_config


CRITICAL_TENSORS = (
    "camera_hidden",
    "baseline_pose_encoding",
    "target_pose_encoding",
    "appearance",
    "pose_geometry",
    "quality",
    "observed",
    "identity_valid",
    "identity_unknown",
    "tracking_masks_output",
    "baseline_world_points",
    "baseline_world_confidence",
)
LOCAL_TENSORS = (
    "sam_local_features",
    "sam_local_uv",
    "sam_local_valid",
)


def main() -> None:
    args = _parse_args()
    config = load_learned_pose_config(args.config)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (config.output_dir / "cache_audit")
    )
    result = audit_caches(
        config,
        output_dir=output_dir,
        require_local=bool(args.require_local),
        full_finite_scan=bool(args.full_finite_scan),
    )
    print(f"cache audit summary={result['summary_csv']}")
    print(
        "cache audit "
        f"clips={result['clips']} passed={result['passed']} "
        f"failed={result['failed']}"
    )
    if args.strict and result["failed"]:
        raise SystemExit(2)


def audit_caches(
    config,
    *,
    output_dir: Path,
    require_local: bool,
    full_finite_scan: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    clip_rows: list[dict[str, Any]] = []
    tensor_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []

    for clip in config.clips:
        path = cache_path(config, clip)
        errors: list[str] = []
        warnings: list[str] = []
        payload: dict[str, Any] | None = None
        if not path.is_file():
            errors.append("missing_file")
        else:
            try:
                payload = load_feature_cache(path, require_complete=False)
            except Exception as exc:  # audit must report every cache in one run
                errors.append(f"load_failed:{type(exc).__name__}:{exc}")

        if payload is not None:
            sequence = len(clip.frame_indices)
            instances = len(clip.instance_ids)
            _check_equal(
                errors,
                "cache_version",
                int(payload.get("cache_version", -1)),
                CACHE_VERSION,
            )
            _check_equal(
                errors,
                "clip_name",
                str(payload.get("clip_name", "")),
                clip.name,
            )
            _check_equal(
                errors,
                "frame_indices",
                tuple(int(v) for v in payload.get("frame_indices", ())),
                clip.frame_indices,
            )
            _check_equal(
                errors,
                "instance_ids",
                tuple(int(v) for v in payload.get("instance_ids", ())),
                clip.instance_ids,
            )
            if not bool(payload.get("complete", False)):
                errors.append("cache_incomplete")
            for field in CRITICAL_TENSORS:
                if field not in payload:
                    errors.append(f"missing_tensor:{field}")
                    continue
                health = _tensor_health(
                    payload[field],
                    full_scan=full_finite_scan,
                )
                tensor_rows.append(
                    {
                        "clip": clip.name,
                        "field": field,
                        **health,
                    }
                )
                if not health["is_tensor"]:
                    errors.append(f"not_tensor:{field}")
                elif not health["finite_sample"]:
                    errors.append(f"nonfinite:{field}")

            expected_prefixes = {
                "camera_hidden": (sequence,),
                "baseline_pose_encoding": (sequence,),
                "target_pose_encoding": (sequence,),
                "appearance": (sequence, instances),
                "pose_geometry": (sequence, instances),
                "quality": (sequence, instances),
                "observed": (sequence, instances),
                "identity_valid": (sequence, instances),
                "identity_unknown": (sequence, instances),
                "tracking_masks_output": (sequence, instances),
            }
            for field, prefix in expected_prefixes.items():
                value = payload.get(field)
                if torch.is_tensor(value) and tuple(value.shape[: len(prefix)]) != prefix:
                    errors.append(
                        f"shape:{field}:{tuple(value.shape)}:prefix={prefix}"
                    )

            has_local = all(field in payload for field in LOCAL_TENSORS)
            if require_local and not has_local:
                errors.append("missing_sam_local_tokens")
            if any(field in payload for field in LOCAL_TENSORS) and not has_local:
                errors.append("partial_sam_local_tokens")
            if has_local:
                local = torch.as_tensor(payload["sam_local_features"])
                uv = torch.as_tensor(payload["sam_local_uv"])
                valid = torch.as_tensor(payload["sam_local_valid"])
                if local.ndim != 4 or local.shape[:2] != (sequence, instances):
                    errors.append(f"shape:sam_local_features:{tuple(local.shape)}")
                if uv.shape != (*local.shape[:3], 2):
                    errors.append(f"shape:sam_local_uv:{tuple(uv.shape)}")
                if valid.shape != local.shape[:3]:
                    errors.append(f"shape:sam_local_valid:{tuple(valid.shape)}")
                if torch.is_floating_point(uv) and bool(valid.any()):
                    selected_uv = uv[valid.bool()]
                    if not bool(
                        ((selected_uv >= -1.0001) & (selected_uv <= 1.0001)).all()
                    ):
                        errors.append("sam_local_uv_out_of_range")
                for field in LOCAL_TENSORS:
                    tensor_rows.append(
                        {
                            "clip": clip.name,
                            "field": field,
                            **_tensor_health(
                                payload[field],
                                full_scan=full_finite_scan,
                            ),
                        }
                    )

            observed = _bool_tensor(payload.get("observed"), sequence, instances)
            identity_valid = _bool_tensor(
                payload.get("identity_valid"), sequence, instances
            )
            identity_unknown = _bool_tensor(
                payload.get("identity_unknown"), sequence, instances
            )
            local_valid = (
                torch.as_tensor(payload["sam_local_valid"]).bool()
                if has_local
                else None
            )
            for sequence_index, frame_index in enumerate(clip.frame_indices):
                frame_observed = observed[sequence_index]
                frame_valid = identity_valid[sequence_index]
                frame_unknown = identity_unknown[sequence_index]
                local_counts = (
                    local_valid[sequence_index].sum(dim=-1)
                    if local_valid is not None
                    else torch.zeros(instances, dtype=torch.long)
                )
                frame_rows.append(
                    {
                        "clip": clip.name,
                        "split": clip.split,
                        "sequence_index": sequence_index,
                        "frame_index": frame_index,
                        "is_reference": int(
                            sequence_index == clip.reference_sequence_index
                        ),
                        "observed_instances": int(frame_observed.sum()),
                        "identity_valid_instances": int(frame_valid.sum()),
                        "identity_unknown_instances": int(frame_unknown.sum()),
                        "usable_instances": int(
                            (frame_observed & (frame_valid | frame_unknown)).sum()
                        ),
                        "sam_local_instances": int(local_counts.gt(0).sum()),
                        "sam_local_tokens": int(local_counts.sum()),
                        "sam_local_min_per_present_instance": int(
                            local_counts[local_counts.gt(0)].min()
                            if bool(local_counts.gt(0).any())
                            else 0
                        ),
                        "sam_local_max_per_instance": int(local_counts.max()),
                    }
                )

            if has_local and not bool(torch.as_tensor(payload["sam_local_valid"]).any()):
                warnings.append("sam_local_tokens_all_empty")

        has_local_payload = bool(
            payload is not None
            and all(field in payload for field in LOCAL_TENSORS)
        )
        clip_rows.append(
            {
                "clip": clip.name,
                "split": clip.split,
                "frames": len(clip.frame_indices),
                "instances": len(clip.instance_ids),
                "path": str(path),
                "size_mib": round(path.stat().st_size / (1024**2), 3)
                if path.is_file()
                else 0,
                "sam_version": "" if payload is None else payload.get("sam_version", ""),
                "sam_source": ""
                if payload is None
                else payload.get("sam_appearance_source", ""),
                "segmentation_variant": ""
                if payload is None
                else payload.get("sam_segmentation_variant", ""),
                "streamvggt_execution": ""
                if payload is None
                else payload.get("streamvggt_execution", ""),
                "has_sam_local_tokens": int(has_local_payload),
                "sam_local_token_count": ""
                if payload is None
                else payload.get("sam_local_token_count", ""),
                "passed": int(not errors),
                "errors": " | ".join(errors),
                "warnings": " | ".join(warnings),
            }
        )

    summary_csv = output_dir / "cache_audit.csv"
    tensor_csv = output_dir / "cache_tensor_audit.csv"
    frame_csv = output_dir / "cache_frame_audit.csv"
    _write_csv(summary_csv, clip_rows)
    _write_csv(tensor_csv, tensor_rows)
    _write_csv(frame_csv, frame_rows)
    report = {
        "config": str(config.source_path),
        "cache_dir": str(config.features.cache_dir),
        "require_local": require_local,
        "full_finite_scan": full_finite_scan,
        "clips": len(clip_rows),
        "passed": sum(int(row["passed"]) for row in clip_rows),
        "failed": sum(not int(row["passed"]) for row in clip_rows),
        "summary_csv": str(summary_csv),
        "tensor_csv": str(tensor_csv),
        "frame_csv": str(frame_csv),
        "clip_results": clip_rows,
    }
    with (output_dir / "cache_audit.json").open("w", encoding="utf8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    return report


def _tensor_health(value: Any, *, full_scan: bool) -> dict[str, Any]:
    if not torch.is_tensor(value):
        return {
            "is_tensor": 0,
            "shape": "",
            "dtype": type(value).__name__,
            "elements": 0,
            "finite_sample": 0,
            "sampled_elements": 0,
            "min": "",
            "max": "",
        }
    tensor = value.detach().cpu()
    flat = tensor.reshape(-1)
    if not full_scan and flat.numel() > 1_000_000:
        indices = torch.linspace(
            0,
            flat.numel() - 1,
            steps=1_000_000,
        ).round().long()
        sample = flat.index_select(0, indices)
    else:
        sample = flat
    if torch.is_floating_point(sample) or torch.is_complex(sample):
        finite = torch.isfinite(sample)
        finite_all = bool(finite.all())
        numeric = sample[finite]
    else:
        finite_all = True
        numeric = sample
    return {
        "is_tensor": 1,
        "shape": "x".join(str(int(v)) for v in tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "elements": int(tensor.numel()),
        "finite_sample": int(finite_all),
        "sampled_elements": int(sample.numel()),
        "min": float(numeric.min()) if numeric.numel() else "",
        "max": float(numeric.max()) if numeric.numel() else "",
    }


def _bool_tensor(value: Any, sequence: int, instances: int) -> torch.Tensor:
    if not torch.is_tensor(value) or value.shape != (sequence, instances):
        return torch.zeros(sequence, instances, dtype=torch.bool)
    return value.detach().bool().cpu()


def _check_equal(errors: list[str], name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        errors.append(f"metadata:{name}:{actual!r}!={expected!r}")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf8")
        return
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="streaming_couping/configs/v7_fusion_data.yaml",
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--require-local", action="store_true")
    parser.add_argument("--full-finite-scan", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
