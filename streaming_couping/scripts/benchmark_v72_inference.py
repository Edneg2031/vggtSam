#!/usr/bin/env python3
"""Benchmark trained V7.2 cached-feature inference (no backbone runtime)."""

from __future__ import annotations

import argparse
import copy
import csv
import time
from pathlib import Path

import torch

from streaming_couping.src.config import load_config
from streaming_couping.src.external_repos import maybe_add_repo_to_path
from streaming_couping.src.learned_pose.config import load_learned_pose_config
from streaming_couping.src.learned_pose.v7_fusion import V7PoseFusion
from streaming_couping.src.learned_pose.v71_causal_fusion import V71FrozenResidualFusion
from streaming_couping.src.learned_pose.v72_local_fusion import (
    V72_ARCHITECTURES,
    V72FrozenLocalResidual,
)
from streaming_couping.scripts.run_v7_fusion_ablation import (
    _find_clip,
    _load_cache,
    load_v7_config,
)
from streaming_couping.scripts.run_v71_instance_causality import (
    _checkpoint_signature as _v71_checkpoint_signature,
    load_v71_config,
)
from streaming_couping.scripts.run_v72_local_token_ablation import (
    CONTROL_ARCHITECTURES,
    _forward_model,
    _limit_local_tokens,
    _load_checkpoint,
    _prepare_payload,
    _signature,
    load_v72_config,
)


def main() -> None:
    args = _parse_args()
    config = load_v72_config(args.config)
    if args.device:
        from dataclasses import replace

        config = replace(config, device=args.device)
    rows = benchmark(
        config,
        result_csv=Path(args.result_csv).expanduser().resolve(),
        selection=str(args.architectures),
        warmup=int(args.warmup),
        repeats=int(args.repeats),
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(output, rows)
    print(f"V7.2 inference speed={output}")
    print(output.read_text(encoding="utf8").rstrip())


def benchmark(
    config,
    *,
    result_csv: Path,
    selection: str,
    warmup: int,
    repeats: int,
) -> list[dict[str, object]]:
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be nonnegative and repeats positive.")
    if not torch.cuda.is_available() and str(config.device).startswith("cuda"):
        raise RuntimeError("V7.2 CUDA benchmark requested without CUDA.")
    v71 = load_v71_config(config.v71_config)
    source_v7 = load_v7_config(v71.v7_config)
    from dataclasses import replace

    v71_base_v7 = replace(
        source_v7,
        device=config.device,
        training=replace(source_v7.training, seed=v71.seed),
    )
    base_v7 = replace(
        source_v7,
        device=config.device,
        training=replace(source_v7.training, seed=config.seed),
    )
    frozen_l0_signature = _v71_checkpoint_signature(
        v71,
        v71_base_v7,
        architecture="frozen_l0",
    )
    data = load_learned_pose_config(config.data_config)
    recovery = load_config(data.recovery_config)
    maybe_add_repo_to_path(recovery.streamvggt_repo)
    from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri

    clips = [
        _find_clip(data, name)
        for name in (
            v71.long_clip_name,
            v71.validation_clip_name,
            v71.external_clip_name,
        )
    ]
    prepared = []
    for clip in clips:
        payload = _load_cache(data, clip)
        batch, baseline, _ = _prepare_payload(
            payload,
            config=base_v7,
            pose_decoder=pose_encoding_to_extri_intri,
            device=config.device,
        )
        prepared.append((clip, payload, batch, baseline))

    base_template = V7PoseFusion(
        architecture="l0_camera_only",
        camera_dim=int(prepared[0][2]["camera_hidden"].shape[-1]),
        appearance_dim=int(prepared[0][2]["appearance"].shape[-1]),
        geometry_dim=int(prepared[0][2]["pose_geometry"].shape[-1]),
        local_feature_dim=int(prepared[0][2]["local_features"].shape[-1]),
        config=base_v7.fusion,
    ).to(config.device)
    base_checkpoint = _load_checkpoint(
        config.output_dir / "frozen_l0.pt",
        _signature(
            config,
            base_v7,
            "frozen_l0",
            token_count=0,
            frozen_l0_signature=frozen_l0_signature,
        ),
    )
    base_template.load_state_dict(base_checkpoint["model"])
    base_template.eval()
    labels = _select_labels(result_csv, selection)
    rows = []
    for label in labels:
        model, token_count = _load_model(
            label,
            base=base_template,
            first_batch=prepared[0][2],
            config=config,
            experiment=base_v7,
            frozen_l0_signature=frozen_l0_signature,
        )
        model.eval()
        parameters = sum(value.numel() for value in model.parameters())
        for clip, payload, full_batch, baseline in prepared:
            batch = (
                _limit_local_tokens(full_batch, token_count)
                if token_count
                else full_batch
            )
            reference = int(payload["reference_sequence_index"])
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                for _ in range(warmup):
                    _forward_model(
                        model,
                        batch=batch,
                        baseline=baseline,
                        reference_index=reference,
                        variant="normal",
                    )
                _synchronize(config.device)
                started = time.perf_counter()
                for _ in range(repeats):
                    _forward_model(
                        model,
                        batch=batch,
                        baseline=baseline,
                        reference_index=reference,
                        variant="normal",
                    )
                _synchronize(config.device)
                elapsed = time.perf_counter() - started
            sequence = int(batch["camera_hidden"].shape[1])
            seconds = elapsed / repeats
            rows.append(
                {
                    "architecture": label,
                    "token_count": token_count,
                    "clip": clip.name,
                    "frames": sequence,
                    "device": config.device,
                    "warmup": warmup,
                    "repeats": repeats,
                    "latency_ms_per_sequence": _short(seconds * 1000),
                    "latency_ms_per_frame": _short(seconds * 1000 / sequence),
                    "frames_per_second": _short(sequence / seconds),
                    "parameters": parameters,
                    "peak_gpu_memory_gib": _short(
                        torch.cuda.max_memory_allocated() / (1024**3)
                        if torch.cuda.is_available()
                        else 0.0
                    ),
                    "scope": "V72_pose_only_cached_features_no_backbones",
                }
            )
        del model
    return rows


def _load_model(
    label,
    *,
    base,
    first_batch,
    config,
    experiment,
    frozen_l0_signature,
):
    if label == "frozen_l0":
        return copy.deepcopy(base), 0
    if label in CONTROL_ARCHITECTURES:
        model = V71FrozenResidualFusion(
            base_model=copy.deepcopy(base),
            architecture=label,
            appearance_dim=int(first_batch["appearance"].shape[-1]),
            geometry_dim=int(first_batch["pose_geometry"].shape[-1]),
            local_feature_dim=int(first_batch["local_features"].shape[-1]),
            config=experiment.fusion,
        ).to(config.device)
        token_count = 0
        architecture = label
    else:
        if "_k" not in label:
            raise ValueError(f"Invalid V7.2 architecture label: {label}")
        architecture, count_text = label.rsplit("_k", 1)
        token_count = int(count_text)
        if architecture not in V72_ARCHITECTURES:
            raise ValueError(f"Unknown V7.2 architecture label: {label}")
        model = V72FrozenLocalResidual(
            base_model=copy.deepcopy(base),
            architecture=architecture,
            sam_local_dim=int(first_batch["sam_local_features"].shape[-1]),
            geometry_dim=int(first_batch["pose_geometry"].shape[-1]),
            geometry_local_dim=int(first_batch["local_features"].shape[-1]),
            config=experiment.fusion,
        ).to(config.device)
    checkpoint = _load_checkpoint(
        config.output_dir / f"{label}.pt",
        _signature(
            config,
            experiment,
            architecture,
            token_count=token_count,
            frozen_l0_signature=frozen_l0_signature,
        ),
    )
    model.load_state_dict(checkpoint["model"])
    return model, token_count


def _select_labels(path: Path, selection: str) -> tuple[str, ...]:
    with path.open(newline="", encoding="utf8") as handle:
        rows = list(csv.DictReader(handle))
    available = {row["architecture"]: row for row in rows}
    if selection == "best":
        candidates = [
            row for row in rows if int(row.get("instance_content", "0")) == 1
        ]
        if not candidates:
            raise ValueError("V7.2 result has no instance-content architecture.")
        best = min(candidates, key=lambda row: float(row["development_score"]))
        labels = ("frozen_l0", *CONTROL_ARCHITECTURES, best["architecture"])
    elif selection == "all":
        labels = tuple(
            row["architecture"]
            for row in rows
            if row["architecture"] != "raw_streamvggt"
        )
    else:
        labels = tuple(value.strip() for value in selection.split(",") if value.strip())
    missing = set(labels) - set(available)
    if missing:
        raise ValueError(f"Requested V7.2 benchmark labels missing: {sorted(missing)}")
    return labels


def _synchronize(device: str) -> None:
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _short(value: float) -> str:
    return f"{float(value):.8g}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="streaming_couping/configs/v72_local_token_ablation.yaml",
    )
    parser.add_argument(
        "--result-csv",
        default="outputs/streaming_couping_v72_local_token_ablation/v72_local_token_ablation.csv",
    )
    parser.add_argument(
        "--output",
        default="outputs/streaming_couping_v72_local_token_ablation/v72_inference_speed.csv",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--architectures",
        default="best",
        help="best, all, or comma-separated result labels.",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=200)
    return parser.parse_args()


if __name__ == "__main__":
    main()
