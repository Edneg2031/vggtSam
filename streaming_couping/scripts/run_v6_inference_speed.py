"""Benchmark cached V6 camera inference without SAM3/StreamVGGT startup."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch

from streaming_couping.scripts.run_v6_camera_overfit import (
    _camera_batch,
    _compose_decoupled_output,
    load_v6_config,
)
from streaming_couping.src.config import load_config
from streaming_couping.src.external_repos import maybe_add_repo_to_path
from streaming_couping.src.learned_pose.cache import cache_path, load_feature_cache
from streaming_couping.src.learned_pose.config import load_learned_pose_config
from streaming_couping.src.learned_pose.v6_camera_fusion import (
    V6CameraFusion,
    V6FusionConfig,
    v6_effective_identity_states,
)

BENCHMARK_VARIANTS = {
    "fusion_6dof": ("v6_checkpoint_fusion.pt",),
    "candidate_A_cameraR_instanceC_local": (
        "v6_checkpoint_specialized_camera_rotation.pt",
        "v6_checkpoint_specialized_instance_center_local.pt",
    ),
    "candidate_B_cameraR_instanceSE3_aux0p1": (
        "v6_checkpoint_specialized_camera_rotation.pt",
        "v6_checkpoint_instance_se3_aux_0p1.pt",
    ),
}


def main() -> None:
    args = _parse_args()
    config = load_v6_config(args.config)
    device = args.device or config.device
    base = load_learned_pose_config(config.base_config)
    selected_names = (
        config.clip_name,
        config.test_clip_name,
        config.validation_clip_name,
    )
    clips = []
    payloads = {}
    cache_load_ms = {}
    for name in selected_names:
        matches = [clip for clip in base.clips if clip.name == name]
        if len(matches) != 1:
            raise ValueError(f"V6 speed clip {name!r} was not found exactly once.")
        clip = matches[0]
        path = cache_path(base, clip)
        start = time.perf_counter()
        payloads[name] = load_feature_cache(path)
        cache_load_ms[name] = 1000.0 * (time.perf_counter() - start)
        clips.append(clip)

    recovery = load_config(base.recovery_config)
    maybe_add_repo_to_path(recovery.streamvggt_repo)
    from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri

    output_dir = args.output_dir or config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    gate_rows = []
    for clip in clips:
        gate_rows.extend(
            _gate_rows(
                payloads[clip.name],
                clip_name=clip.name,
                fusion=config.fusion,
            )
        )
    _write_csv(output_dir / "v6_gate_summary.csv", gate_rows)

    rows = []
    for variant, filenames in BENCHMARK_VARIANTS.items():
        model_start = time.perf_counter()
        loaded = [
            _load_checkpoint_model(
                output_dir / filename,
                device=device,
                gate_config=config.fusion,
            )
            for filename in filenames
        ]
        model_load_ms = 1000.0 * (time.perf_counter() - model_start)
        parameter_count = sum(
            sum(parameter.numel() for parameter in model.parameters())
            for model, _ in loaded
        )
        for clip in clips:
            payload = payloads[clip.name]
            transfer_start = time.perf_counter()
            batch = _camera_batch(payload, device=device)
            image_size = tuple(int(value) for value in payload["image_size"])
            baseline_w2c, _ = pose_encoding_to_extri_intri(
                batch["baseline_pose_encoding"],
                image_size_hw=image_size,
            )
            _synchronize(device)
            input_prepare_ms = 1000.0 * (time.perf_counter() - transfer_start)
            reference_index = int(payload["reference_sequence_index"])

            def run_once(
                loaded=loaded,
                batch=batch,
                baseline_w2c=baseline_w2c,
                reference_index=reference_index,
            ):
                outputs = [
                    _forward_model(
                        model,
                        mode=mode,
                        batch=batch,
                        baseline_w2c=baseline_w2c,
                        reference_index=reference_index,
                    )
                    for model, mode in loaded
                ]
                if len(outputs) == 1:
                    return outputs[0]
                return _compose_decoupled_output(
                    outputs[0],
                    outputs[1],
                    baseline_w2c=baseline_w2c,
                    reference_index=reference_index,
                )

            latency_ms, peak_memory_gib, active_frames = _benchmark(
                run_once,
                device=device,
                warmup=args.warmup,
                repeats=args.repeats,
            )
            frames = len(payload["frame_indices"])
            rows.append(
                {
                    "variant": variant,
                    "clip": clip.name,
                    "frames": frames,
                    "device": device,
                    "dtype": str(batch["camera_hidden"].dtype).removeprefix(
                        "torch."
                    ),
                    "gate_policy": config.fusion.identity_gate_policy,
                    "warmup": args.warmup,
                    "repeats": args.repeats,
                    "latency_ms_per_sequence": _short(latency_ms),
                    "latency_ms_per_frame": _short(latency_ms / frames),
                    "frames_per_second": _short(1000.0 * frames / latency_ms),
                    "active_nonreference_frames": active_frames,
                    "parameters": parameter_count,
                    "peak_gpu_memory_gib": _short(peak_memory_gib),
                    "cache_load_ms_cpu": _short(cache_load_ms[clip.name]),
                    "model_load_ms": _short(model_load_ms),
                    "input_prepare_ms": _short(input_prepare_ms),
                    "scope": "V6_camera_only_cached_features",
                }
            )
        del loaded
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    speed_path = output_dir / "v6_inference_speed.csv"
    _write_csv(speed_path, rows)
    print(speed_path)
    with speed_path.open("r", encoding="utf8") as handle:
        print(handle.read().rstrip())
    print(output_dir / "v6_gate_summary.csv")
    print(
        "scope excludes SAM3, StreamVGGT and disk cache creation; "
        "cache/model/input startup times are reported separately"
    )


def _load_checkpoint_model(
    path: Path,
    *,
    device: str,
    gate_config: V6FusionConfig,
) -> tuple[V6CameraFusion, str]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing V6 checkpoint {path}. Run commands_v6_camera_overfit.txt first."
        )
    artifact = _torch_load(path)
    fusion_values = dict(artifact["fusion"])
    fusion_values["identity_gate_policy"] = gate_config.identity_gate_policy
    fusion_values["unknown_reliability"] = gate_config.unknown_reliability
    fusion_values["softened_mismatch_reliability"] = (
        gate_config.softened_mismatch_reliability
    )
    fusion = V6FusionConfig(**fusion_values)
    model = V6CameraFusion(
        camera_dim=int(artifact["camera_dim"]),
        appearance_dim=int(artifact["appearance_dim"]),
        geometry_dim=int(artifact["geometry_dim"]),
        config=fusion,
        head_component=str(artifact.get("head_component", "se3")),
    ).to(device)
    model.load_state_dict(artifact["model"])
    model.eval()
    return model, str(artifact["mode"])


def _forward_model(
    model: V6CameraFusion,
    *,
    mode: str,
    batch: dict[str, torch.Tensor],
    baseline_w2c: torch.Tensor,
    reference_index: int,
) -> dict[str, torch.Tensor]:
    return model(
        camera_hidden=batch["camera_hidden"],
        baseline_world_to_camera=baseline_w2c,
        appearance=batch["appearance"],
        geometry=batch["pose_geometry"],
        quality=batch["quality"],
        observed=batch["observed"],
        identity_valid=batch["identity_valid"],
        identity_unknown=batch["identity_unknown"],
        reference_index=reference_index,
        mode=mode,
    )


def _benchmark(
    run_once,
    *,
    device: str,
    warmup: int,
    repeats: int,
) -> tuple[float, float, int]:
    with torch.inference_mode():
        output = None
        for _ in range(warmup):
            output = run_once()
        _synchronize(device)
        if str(device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repeats):
                output = run_once()
            end.record()
            end.synchronize()
            elapsed_ms = float(start.elapsed_time(end)) / repeats
            peak_memory_gib = float(torch.cuda.max_memory_allocated(device)) / (
                1024**3
            )
        else:
            start_time = time.perf_counter()
            for _ in range(repeats):
                output = run_once()
            elapsed_ms = 1000.0 * (time.perf_counter() - start_time) / repeats
            peak_memory_gib = 0.0
        if output is None:
            raise RuntimeError("V6 speed benchmark produced no output.")
        active_frames = int(output["active_frames"].sum().cpu())
    return elapsed_ms, peak_memory_gib, active_frames


def _gate_rows(
    payload: dict,
    *,
    clip_name: str,
    fusion: V6FusionConfig,
) -> list[dict[str, object]]:
    observed = payload["observed"].bool()
    identity_valid = payload["identity_valid"].bool()
    identity_unknown = payload["identity_unknown"].bool()
    identity_mismatch = payload["identity_mismatch"].bool()
    match, effective_unknown, softened = v6_effective_identity_states(
        observed=observed,
        identity_valid=identity_valid,
        identity_unknown=identity_unknown,
        policy=fusion.identity_gate_policy,
    )
    track_ok = payload["quality"][..., 0] >= fusion.min_track_confidence
    strict_usable = observed & track_ok & (identity_valid | identity_unknown)
    usable = observed & track_ok & (match | effective_unknown)
    rows = []
    for slot, instance_id in enumerate(payload["instance_ids"]):
        rows.append(
            {
                "clip": clip_name,
                "instance_id": int(instance_id),
                "frames": observed.shape[0],
                "observed": int(observed[:, slot].sum()),
                "cached_match": int(identity_valid[:, slot].sum()),
                "cached_unknown": int(identity_unknown[:, slot].sum()),
                "cached_mismatch": int(identity_mismatch[:, slot].sum()),
                "softened_mismatch": int(softened[:, slot].sum()),
                "effective_unknown": int(effective_unknown[:, slot].sum()),
                "strict_usable_current": int(strict_usable[:, slot].sum()),
                "usable_current": int(usable[:, slot].sum()),
                "recovered_by_soft_gate": int(
                    (usable & ~strict_usable)[:, slot].sum()
                ),
                "strict_memory_writes": int((match & track_ok)[:, slot].sum()),
                "gate_policy": fusion.identity_gate_policy,
            }
        )
    return rows


def _torch_load(path: Path) -> dict:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise TypeError(f"Invalid V6 checkpoint: {path}.")
    return value


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty V6 table: {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _synchronize(device: str) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def _short(value: float) -> str:
    return f"{float(value):.8g}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="streaming_couping/configs/v6_camera_overfit.yaml",
    )
    parser.add_argument("--device")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 1:
        parser.error("--warmup and --repeats must be positive")
    return args


if __name__ == "__main__":
    main()
