"""Run exact full-history StreamVGGT with layer-wise multi-GPU sharding.

This is layer-wise model parallelism, not DDP: every frame still observes the
complete causal history, while each layer's KV cache remains on that layer's GPU.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import sys
import time
from pathlib import Path
from types import MethodType
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .common import (
    discover_images,
    save_depth_visualization,
    save_rgb,
    save_trajectory_plot,
    write_binary_ply,
    write_image_list,
    write_intrinsics_txt,
    write_json,
    write_w2c_txt,
)
from .pointcloud_products import (
    PointCloudProductAccumulator,
    PointCloudProtocol,
    rebuild_depth_pose_products,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = ROOT / "externals" / "streamvggt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--repo", default=str(DEFAULT_REPO))
    parser.add_argument(
        "--output-root",
        default="outputs/long_sequence_baselines/streamvggt",
    )
    parser.add_argument("--scene-name", default="meeting_room_a02")
    parser.add_argument(
        "--devices",
        default="cuda:0,cuda:1,cuda:2,cuda:3,cuda:4",
        help="Three or more logical CUDA devices; select physical GPUs with CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--image-mode", choices=("crop", "pad"), default="crop")
    parser.add_argument(
        "--confidence-percentile",
        type=float,
        default=50.0,
        help="Per-frame percentile used by points/full.ply; full_all.ply remains unfiltered.",
    )
    parser.add_argument("--max-full-pointcloud-points", type=int, default=2_000_000)
    parser.add_argument("--depth-percentile-low", type=float, default=1.0)
    parser.add_argument("--depth-percentile-high", type=float, default=99.0)
    parser.add_argument("--voxel-size-ratio", type=float, default=0.01)
    parser.add_argument("--min-voxel-observations", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()


def partition_layers(depth: int, device_count: int = 3) -> list[int]:
    """Assign contiguous, nearly equal layer ranges to devices."""

    if depth <= 0 or device_count <= 0 or device_count > depth:
        raise ValueError(f"Invalid depth/device_count: {depth}/{device_count}")
    base, remainder = divmod(depth, device_count)
    result: list[int] = []
    for device_index in range(device_count):
        count = base + (1 if device_index < remainder else 0)
        result.extend([device_index] * count)
    return result


def parse_devices(value: str) -> tuple[torch.device, ...]:
    devices = tuple(torch.device(item.strip()) for item in value.split(",") if item.strip())
    if len(devices) < 3 or len(set(devices)) != len(devices):
        raise ValueError(f"--devices must contain at least three distinct devices, got {devices}")
    if any(device.type != "cuda" for device in devices):
        raise ValueError(f"All model-parallel devices must be CUDA devices, got {devices}")
    return devices


def _install_processed_key_cache(attention: torch.nn.Module) -> None:
    """Cache token-wise normalized/RoPE keys instead of recomputing all history."""

    if getattr(attention, "_long_sequence_processed_key_cache", False):
        return
    object.__setattr__(attention, "_long_sequence_processed_key_cache", True)
    object.__setattr__(
        attention,
        "forward",
        MethodType(_processed_key_cache_forward, attention),
    )


def _processed_key_cache_forward(
    attention,
    x: torch.Tensor,
    pos=None,
    attn_mask=None,
    past_key_values=None,
    use_cache=False,
):
    """Equivalent cached attention that never reapplies QK norm/RoPE to old keys."""

    if not use_cache:
        return type(attention).forward(
            attention,
            x,
            pos=pos,
            attn_mask=attn_mask,
            past_key_values=past_key_values,
            use_cache=False,
        )

    batch, query_tokens, channels = x.shape
    qkv = (
        attention.qkv(x)
        .reshape(
            batch,
            query_tokens,
            3,
            attention.num_heads,
            attention.head_dim,
        )
        .permute(2, 0, 3, 1, 4)
    )
    query, key, value = qkv.unbind(0)
    query = attention.q_norm(query)
    key = attention.k_norm(key)
    if attention.rope is not None:
        query = attention.rope(query, pos)
        key = attention.rope(key, pos)

    key = key.unsqueeze(2)
    value = value.unsqueeze(2)
    if past_key_values is not None:
        past_key, past_value = past_key_values
        key = torch.cat([past_key, key], dim=2)
        value = torch.cat([past_value, value], dim=2)
    new_key_values = (key, value)

    key = key.reshape(
        batch,
        attention.num_heads,
        -1,
        attention.head_dim,
    )
    value = value.reshape(
        batch,
        attention.num_heads,
        -1,
        attention.head_dim,
    )
    if attention.fused_attn:
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=attention.attn_drop.p if attention.training else 0.0,
        )
    else:
        scores = (query * attention.scale) @ key.transpose(-2, -1)
        if attn_mask is not None:
            scores = scores + attn_mask
        weights = attention.attn_drop(scores.softmax(dim=-1))
        output = weights @ value

    output = output.transpose(1, 2).reshape(batch, query_tokens, channels)
    output = attention.proj_drop(attention.proj(output))
    return output, new_key_values


def _assert_processed_key_cache_equivalence() -> None:
    """Numerically verify the optimized cache against upstream on three frames."""

    from streamvggt.layers.attention import Attention
    from streamvggt.layers.rope import RotaryPositionEmbedding2D

    with torch.random.fork_rng(devices=[]), torch.inference_mode():
        torch.manual_seed(2026)
        reference = Attention(
            dim=32,
            num_heads=4,
            qk_norm=True,
            rope=RotaryPositionEmbedding2D(frequency=100),
        ).eval()
        optimized = copy.deepcopy(reference).eval()
        _install_processed_key_cache(optimized)
        positions = torch.tensor(
            [[[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]]],
            dtype=torch.long,
        )
        reference_cache = None
        optimized_cache = None
        for _ in range(3):
            tokens = torch.randn(1, 6, 32)
            reference_output, reference_cache = reference(
                tokens,
                pos=positions,
                past_key_values=reference_cache,
                use_cache=True,
            )
            optimized_output, optimized_cache = optimized(
                tokens,
                pos=positions,
                past_key_values=optimized_cache,
                use_cache=True,
            )
            torch.testing.assert_close(
                optimized_output,
                reference_output,
                rtol=1e-5,
                atol=1e-6,
            )


class LayerShardedStreamVGGT:
    """A no-training, exact-causal, layer-sharded execution wrapper."""

    def __init__(self, model: Any, devices: Sequence[torch.device]) -> None:
        if len(devices) < 3:
            raise ValueError("LayerShardedStreamVGGT requires at least three devices")
        self.model = model.eval()
        self.devices = tuple(devices)
        self.aggregator = model.aggregator
        if self.aggregator.aa_order != ["frame", "global"]:
            raise ValueError(
                "This exact sharding runner requires aa_order=['frame', 'global'], got "
                f"{self.aggregator.aa_order}"
            )
        if self.aggregator.aa_block_size != 1:
            raise ValueError(
                "This exact sharding runner requires aa_block_size=1, got "
                f"{self.aggregator.aa_block_size}"
            )
        self.layer_devices = partition_layers(self.aggregator.depth, len(devices))
        self.selected_layers = sorted(
            set(model.depth_head.intermediate_layer_idx)
            | set(model.point_head.intermediate_layer_idx)
            | {self.aggregator.depth - 1}
        )
        self._distribute_model()
        self.past_key_values: list[Any] = [None] * self.aggregator.depth
        self.past_key_values_camera: list[Any] = [None] * model.camera_head.trunk_depth

    def _distribute_model(self) -> None:
        first, middle, last = self.devices[0], self.devices[1], self.devices[-1]
        self.aggregator.patch_embed.to(first)
        _move_parameter(self.aggregator.camera_token, first)
        _move_parameter(self.aggregator.register_token, first)
        for name in ("_resnet_mean", "_resnet_std"):
            buffer = self.aggregator._buffers.get(name)
            if buffer is not None:
                self.aggregator._buffers[name] = buffer.to(first)
        for layer_index, device_index in enumerate(self.layer_devices):
            device = self.devices[device_index]
            self.aggregator.frame_blocks[layer_index].to(device)
            self.aggregator.global_blocks[layer_index].to(device)
            _install_processed_key_cache(
                self.aggregator.global_blocks[layer_index].attn
            )
        self.model.depth_head.to(first)
        self.model.point_head.to(middle)
        self.model.camera_head.to(last)

    def aggregate_frame(
        self,
        images: torch.Tensor,
        frame_index: int,
    ) -> dict[int, torch.Tensor]:
        """Run one frame and retain only head-consumed intermediate levels."""

        if images.ndim != 5 or images.shape[:2] != (1, 1):
            raise ValueError(f"Expected one image [1, 1, 3, H, W], got {images.shape}")
        aggregator = self.aggregator
        first = self.devices[0]
        images = images.to(first, non_blocking=True)
        _, _, channels, height, width = images.shape
        if channels != 3:
            raise ValueError(f"Expected three input channels, got {channels}")

        normalized = (images - aggregator._resnet_mean) / aggregator._resnet_std
        patch_tokens = aggregator.patch_embed(normalized.reshape(1, 3, height, width))
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]
        token_slot = 0 if frame_index == 0 else 1
        camera_token = aggregator.camera_token[:, token_slot]
        register_token = aggregator.register_token[:, token_slot]
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        position = None
        if aggregator.rope is not None:
            position = aggregator.position_getter(
                1,
                height // aggregator.patch_size,
                width // aggregator.patch_size,
                device=first,
            )
            position = position + 1
            special = torch.zeros(
                1,
                aggregator.patch_start_idx,
                2,
                device=first,
                dtype=position.dtype,
            )
            position = torch.cat([special, position], dim=1)

        selected: dict[int, torch.Tensor] = {}
        for layer_index, device_index in enumerate(self.layer_devices):
            device = self.devices[device_index]
            if tokens.device != device:
                tokens = tokens.to(device, non_blocking=True)
            layer_position = (
                None
                if position is None
                else position.to(device, non_blocking=True)
            )
            frame_tokens = aggregator.frame_blocks[layer_index](
                tokens,
                pos=layer_position,
            )
            tokens, new_key_values = aggregator.global_blocks[layer_index](
                frame_tokens,
                pos=layer_position,
                past_key_values=self.past_key_values[layer_index],
                use_cache=True,
            )
            self.past_key_values[layer_index] = new_key_values
            if layer_index in self.selected_layers:
                selected[layer_index] = torch.cat(
                    [frame_tokens[:, None], tokens[:, None]], dim=-1
                )
        return selected

    def camera(self, selected: dict[int, torch.Tensor]) -> torch.Tensor:
        device = self.devices[-1]
        token_list: list[Any] = [None] * self.aggregator.depth
        token_list[-1] = selected[self.aggregator.depth - 1].to(device).float()
        pose_list, self.past_key_values_camera = self.model.camera_head(
            token_list,
            past_key_values_camera=self.past_key_values_camera,
            use_cache=True,
        )
        return pose_list[-1]

    def depth(
        self,
        selected: dict[int, torch.Tensor],
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.devices[0]
        token_list = self._tokens_for_head(selected, self.model.depth_head, device)
        return self.model.depth_head(
            token_list,
            images=images.to(device).float(),
            patch_start_idx=self.aggregator.patch_start_idx,
        )

    def points(
        self,
        selected: dict[int, torch.Tensor],
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.devices[1]
        token_list = self._tokens_for_head(selected, self.model.point_head, device)
        return self.model.point_head(
            token_list,
            images=images.to(device).float(),
            patch_start_idx=self.aggregator.patch_start_idx,
        )

    def _tokens_for_head(
        self,
        selected: dict[int, torch.Tensor],
        head: Any,
        device: torch.device,
    ) -> list[Any]:
        token_list: list[Any] = [None] * self.aggregator.depth
        for layer_index in head.intermediate_layer_idx:
            token_list[layer_index] = selected[layer_index].to(device).float()
        return token_list


def _move_parameter(parameter: torch.nn.Parameter, device: torch.device) -> None:
    parameter.data = parameter.data.to(device)
    if parameter.grad is not None:
        parameter.grad.data = parameter.grad.data.to(device)


def load_model(repo: Path, checkpoint: Path) -> Any:
    sys.path.insert(0, str(repo / "src"))
    from streamvggt.models.streamvggt import StreamVGGT

    print(f"Loading StreamVGGT checkpoint on CPU: {checkpoint}", flush=True)
    model = StreamVGGT()
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=True)
    del state
    model.track_head = None
    gc.collect()
    return model.eval()


def _gpu_summary(devices: Sequence[torch.device]) -> list[dict[str, Any]]:
    summary = []
    for device in devices:
        index = device.index or 0
        properties = torch.cuda.get_device_properties(index)
        summary.append(
            {
                "logical_device": str(device),
                "name": properties.name,
                "total_memory_gib": properties.total_memory / 2**30,
                "peak_allocated_gib": torch.cuda.max_memory_allocated(index) / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved(index) / 2**30,
            }
        )
    return summary


def _sync(devices: Sequence[torch.device]) -> None:
    for device in devices:
        torch.cuda.synchronize(device)


def _layout_lines(layer_devices: Sequence[int], device_count: int) -> list[str]:
    lines = []
    for device_index in range(device_count):
        layers = [
            index
            for index, assigned_device in enumerate(layer_devices)
            if assigned_device == device_index
        ]
        roles = []
        if device_index == 0:
            roles.append("patch/depth")
        if device_index == 1:
            roles.append("point")
        if device_index == device_count - 1:
            roles.append("camera")
        role_text = f" + {','.join(roles)}" if roles else ""
        lines.append(
            f"GPU{device_index}=layers {layers[0]}-{layers[-1]}{role_text}"
        )
    return lines


def _git_commit(repo: Path) -> str | None:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    args = parse_args()
    devices = parse_devices(args.devices)
    if not torch.cuda.is_available():
        raise RuntimeError("StreamVGGT multi-GPU runner requires CUDA")
    if torch.cuda.device_count() != len(devices):
        raise RuntimeError(
            f"This safety check expects exactly {len(devices)} visible GPUs, but PyTorch sees "
            f"{torch.cuda.device_count()}. Start it with "
            "CUDA_VISIBLE_DEVICES containing only the selected free GPUs."
        )
    if not 0 <= args.confidence_percentile <= 100:
        raise ValueError("--confidence-percentile must be in [0, 100]")
    if args.max_full_pointcloud_points <= 0:
        raise ValueError("--max-full-pointcloud-points must be positive")

    repo = Path(args.repo).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not (repo / "src" / "streamvggt").is_dir():
        raise FileNotFoundError(f"StreamVGGT submodule is not initialized: {repo}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"StreamVGGT checkpoint not found: {checkpoint}")
    images = discover_images(args.image_dir, args.max_frames)
    scene_dir = Path(args.output_root).expanduser().resolve() / args.scene_name
    scene_dir.mkdir(parents=True, exist_ok=True)
    write_image_list(scene_dir / "input_images.txt", images)

    model = load_model(repo, checkpoint)
    _assert_processed_key_cache_equivalence()
    print("processed_key_cache_equivalence=passed", flush=True)
    runner = LayerShardedStreamVGGT(model, devices)
    from streamvggt.utils.load_fn import load_and_preprocess_images
    from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri

    for device in devices:
        torch.cuda.reset_peak_memory_stats(device)
    amp_dtype = (
        torch.bfloat16
        if all(torch.cuda.get_device_capability(device)[0] >= 8 for device in devices)
        else torch.float16
    )
    print("StreamVGGT model parallel layout:", flush=True)
    for line in _layout_lines(runner.layer_devices, len(devices)):
        print(f"  {line}", flush=True)
    print(
        f"Frames={len(images)} amp={amp_dtype} full_history=True; no temporal chunk reset",
        flush=True,
    )
    print(
        "processed_key_cache=True; historical QK norm/RoPE is not recomputed",
        flush=True,
    )

    point_protocol = PointCloudProtocol(
        confidence_percentile=args.confidence_percentile,
        depth_percentile_low=args.depth_percentile_low,
        depth_percentile_high=args.depth_percentile_high,
        voxel_size_ratio=args.voxel_size_ratio,
        min_voxel_observations=args.min_voxel_observations,
        max_points=args.max_full_pointcloud_points,
    )
    point_protocol.validate()
    pointhead_products: PointCloudProductAccumulator | None = None
    pose_encodings: list[torch.Tensor] = []
    runtime_rows: list[dict[str, Any]] = []
    processed_shape: tuple[int, int] | None = None
    started = time.perf_counter()

    with torch.inference_mode():
        for frame_index, image_path in enumerate(images):
            frame_started = time.perf_counter()
            image_cpu = load_and_preprocess_images(
                [str(image_path)], mode=args.image_mode
            )
            current_shape = tuple(int(value) for value in image_cpu.shape[-2:])
            if processed_shape is None:
                processed_shape = current_shape
            elif current_shape != processed_shape:
                raise ValueError(
                    "All frames must have one processed shape for a shared KV cache; "
                    f"first={processed_shape}, frame {frame_index}={current_shape}. "
                    "Use images with a consistent resolution or --image-mode pad."
                )
            batch_image = image_cpu.unsqueeze(0).to(devices[0], non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                selected = runner.aggregate_frame(batch_image, frame_index)
            with torch.autocast(device_type="cuda", enabled=False):
                pose_encoding = runner.camera(selected)
                depth, depth_confidence = runner.depth(selected, batch_image)
                points, point_confidence = runner.points(selected, batch_image)

            pose_encodings.append(pose_encoding.detach().float().cpu())
            depth_np = depth[0, 0, ..., 0].detach().float().cpu().numpy()
            depth_conf_np = depth_confidence[0, 0].detach().float().cpu().numpy()
            points_np = points[0, 0].detach().float().cpu().numpy()
            point_conf_np = point_confidence[0, 0].detach().float().cpu().numpy()
            rgb = save_rgb(
                scene_dir / "images" / "rgb" / f"frame_{frame_index:06d}.png",
                image_cpu[0].numpy(),
            )
            depth_dir = scene_dir / "depth"
            (depth_dir / "dpt").mkdir(parents=True, exist_ok=True)
            (depth_dir / "conf").mkdir(parents=True, exist_ok=True)
            np.save(depth_dir / "dpt" / f"frame_{frame_index:06d}.npy", depth_np)
            np.save(depth_dir / "conf" / f"frame_{frame_index:06d}.npy", depth_conf_np)
            save_depth_visualization(
                depth_dir / "dpt_plasma" / f"frame_{frame_index:06d}.png",
                depth_np,
            )

            flat_points = points_np.reshape(-1, 3)
            flat_colors = rgb.reshape(-1, 3)
            flat_confidence = point_conf_np.reshape(-1)
            if pointhead_products is None:
                positive_depth = depth_np[np.isfinite(depth_np) & (depth_np > 0)]
                if not len(positive_depth):
                    raise ValueError("First StreamVGGT frame has no positive finite depth.")
                pointhead_products = PointCloudProductAccumulator(
                    total_frames=len(images),
                    protocol=point_protocol,
                    scale_reference=float(np.median(positive_depth)),
                )
            point_diagnostics = pointhead_products.add_frame(
                flat_points,
                flat_colors,
                flat_confidence,
            )
            finite_count = int(point_diagnostics["raw_points"])
            confident_count = int(point_diagnostics["filtered_points"])
            threshold = float(point_diagnostics["confidence_threshold"])

            del selected, pose_encoding, depth, depth_confidence, points, point_confidence
            del batch_image
            _sync(devices)
            frame_seconds = time.perf_counter() - frame_started
            row: dict[str, Any] = {
                "sequence_index": frame_index,
                "source_image": str(image_path),
                "height": current_shape[0],
                "width": current_shape[1],
                "confidence_threshold": threshold,
                "finite_points": finite_count,
                "confident_points": confident_count,
                "frame_seconds": frame_seconds,
            }
            for logical_index, device in enumerate(devices):
                index = device.index or 0
                row[f"gpu{logical_index}_allocated_gib"] = (
                    torch.cuda.memory_allocated(index) / 2**30
                )
                row[f"gpu{logical_index}_reserved_gib"] = (
                    torch.cuda.memory_reserved(index) / 2**30
                )
            runtime_rows.append(row)
            if (
                frame_index == 0
                or (frame_index + 1) % max(1, args.progress_every) == 0
                or frame_index + 1 == len(images)
            ):
                memory = ", ".join(
                    f"gpu{i}={row[f'gpu{i}_allocated_gib']:.2f}GiB"
                    for i in range(len(devices))
                )
                print(
                    f"[{frame_index + 1}/{len(images)}] {image_path.name} "
                    f"{frame_seconds:.2f}s {memory}",
                    flush=True,
                )

    elapsed = time.perf_counter() - started
    assert processed_shape is not None
    pose_tensor = torch.cat(pose_encodings, dim=1)
    extrinsics, intrinsics = pose_encoding_to_extri_intri(
        pose_tensor,
        image_size_hw=processed_shape,
    )
    extrinsics_np = extrinsics[0].numpy()
    intrinsics_np = intrinsics[0].numpy()
    frame_ids = list(range(len(images)))
    write_w2c_txt(scene_dir / "poses" / "abs_pose.txt", extrinsics_np, frame_ids)
    write_intrinsics_txt(scene_dir / "poses" / "intri.txt", intrinsics_np, frame_ids)
    save_trajectory_plot(
        scene_dir / "plots" / "trajectory.png",
        extrinsics_np,
        "StreamVGGT prediction (no GT/alignment)",
    )
    if pointhead_products is None:
        raise RuntimeError("StreamVGGT produced no point-head products.")
    pointcloud_started = time.perf_counter()
    print("[pointcloud] finalizing StreamVGGT pointhead products", flush=True)
    pointhead_summary = pointhead_products.write(scene_dir / "points", "pointhead")
    points_all, colors_all = pointhead_products.raw.arrays()
    points_conf, colors_conf = pointhead_products.filtered.arrays()
    # Backward-compatible aliases used by the first long-sequence runs.
    write_binary_ply(scene_dir / "points" / "full_all.ply", points_all, colors_all)
    write_binary_ply(scene_dir / "points" / "full.ply", points_conf, colors_conf)
    depthpose_summary = rebuild_depth_pose_products(
        scene_dir,
        protocol=point_protocol,
    )
    pointcloud_seconds = time.perf_counter() - pointcloud_started
    _write_runtime_csv(scene_dir / "runtime_per_frame.csv", runtime_rows)

    summary = {
        "method": "streamvggt_multigpu_full_history",
        "scene": args.scene_name,
        "frames": len(images),
        "image_dir": str(Path(args.image_dir).expanduser().resolve()),
        "checkpoint": str(checkpoint),
        "repo_commit": _git_commit(repo),
        "devices": [str(device) for device in devices],
        "layers_per_device": [
            runner.layer_devices.count(device_index)
            for device_index in range(len(devices))
        ],
        "full_history": True,
        "temporal_chunks": False,
        "processed_key_cache": True,
        "amp_dtype": str(amp_dtype),
        "processed_height": processed_shape[0],
        "processed_width": processed_shape[1],
        "confidence_percentile": args.confidence_percentile,
        "raw_points_seen": pointhead_products.raw.seen_points,
        "confident_points_seen": pointhead_products.filtered_points_seen,
        "full_all_ply_points": len(points_all),
        "full_ply_points": len(points_conf),
        "pointcloud_processing_seconds": pointcloud_seconds,
        "pointhead_products": pointhead_summary,
        "depthpose_products": depthpose_summary,
        "elapsed_seconds": elapsed,
        "frames_per_second": len(images) / elapsed if elapsed else None,
        "gpus": _gpu_summary(devices),
        "output": str(scene_dir),
    }
    write_json(scene_dir / "run_summary.json", summary)
    print(f"StreamVGGT complete: {scene_dir}", flush=True)


def _write_runtime_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
