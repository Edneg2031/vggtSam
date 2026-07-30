"""Run the official HorizonStream baseline on a naturally sorted image directory."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import yaml

from .common import discover_images, write_image_list, write_json
from .pointcloud_products import PointCloudProtocol, rebuild_depth_pose_products


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = ROOT / "externals" / "horizonstream"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--repo", default=str(DEFAULT_REPO))
    parser.add_argument(
        "--output-root",
        default="outputs/long_sequence_baselines/horizonstream",
    )
    parser.add_argument("--scene-name", default="meeting_room_a02")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--sliding-size", type=int, default=10)
    parser.add_argument("--max-full-pointcloud-points", type=int, default=2_000_000)
    parser.add_argument("--confidence-percentile", type=float, default=50.0)
    parser.add_argument("--depth-percentile-low", type=float, default=1.0)
    parser.add_argument("--depth-percentile-high", type=float, default=99.0)
    parser.add_argument("--voxel-size-ratio", type=float, default=0.01)
    parser.add_argument("--min-voxel-observations", type=int, default=2)
    return parser.parse_args()


def build_config(args: argparse.Namespace, images: list[Path]) -> dict:
    repo = Path(args.repo).expanduser().resolve()
    config_path = repo / "configs" / "horizonstream_infer.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"HorizonStream config not found: {config_path}")
    with config_path.open() as handle:
        config = yaml.safe_load(handle)

    config["device"] = args.device
    config["model"]["checkpoint"] = str(Path(args.checkpoint).expanduser().resolve())
    config["inference"].update(
        {
            "window_size": int(args.window_size),
            "sliding_size": int(args.sliding_size),
            "offload_outputs_to_cpu": True,
            "enable_offline_motion_averaging": False,
        }
    )
    config["data"].update(
        {
            "format": "image_list",
            "image_paths": [str(path) for path in images],
            "image_scene_name": args.scene_name,
            "max_frames": None,
            "size": 518,
            "crop": True,
            "patch_size": 14,
            "camera_preprocess": False,
        }
    )
    config["output"].update(
        {
            "root": str(Path(args.output_root).expanduser().resolve()),
            "abs_pose_source": "online",
            "save_videos": False,
            "save_points": True,
            "save_frame_points": False,
            "save_depth": True,
            "save_depth_conf": True,
            "save_depth_vis": True,
            "save_images": True,
            "save_plots": True,
            "mask_sky": False,
            "point_mask_sky": False,
            "point_depth_min": None,
            "point_depth_max": None,
            "point_depth_percentile_min": None,
            "point_depth_percentile_max": None,
            "point_sky_color_filter": False,
            "point_outlier_filter": False,
            "point_voxel_size": None,
            "point_random_sample_ratio": None,
            "max_full_pointcloud_points": int(args.max_full_pointcloud_points),
        }
    )
    return config


def main() -> None:
    args = parse_args()
    pointcloud_protocol = PointCloudProtocol(
        confidence_percentile=args.confidence_percentile,
        depth_percentile_low=args.depth_percentile_low,
        depth_percentile_high=args.depth_percentile_high,
        voxel_size_ratio=args.voxel_size_ratio,
        min_voxel_observations=args.min_voxel_observations,
        max_points=args.max_full_pointcloud_points,
    )
    pointcloud_protocol.validate()
    repo = Path(args.repo).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"HorizonStream checkpoint not found: {checkpoint}")
    if not torch.cuda.is_available():
        raise RuntimeError("HorizonStream baseline requires CUDA")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "HorizonStream runner must see exactly one GPU. Set "
            "CUDA_VISIBLE_DEVICES to one free physical GPU."
        )
    images = discover_images(args.image_dir, args.max_frames)
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    write_image_list(output_root / "input_images.txt", images)
    config = build_config(args, images)
    with (output_root / "resolved_config.yaml").open("w") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    _install_matplotlib_compatibility()
    sys.path.insert(0, str(repo))
    import horizonstream.core.infer as horizon_infer

    # The pinned upstream commit reads this optional diagnostic as a module
    # global but does not initialize it when offline averaging is disabled.
    # Defining it here avoids changing the external submodule.
    if not hasattr(horizon_infer, "offline_error"):
        horizon_infer.offline_error = None

    device_index = torch.device(args.device).index or 0
    torch.cuda.reset_peak_memory_stats(device_index)
    started = time.perf_counter()
    horizon_infer.run_inference_cfg(config)
    torch.cuda.synchronize(device_index)
    elapsed = time.perf_counter() - started
    scene_dir = output_root / args.scene_name
    pointcloud_started = time.perf_counter()
    pointcloud_products = rebuild_depth_pose_products(
        scene_dir,
        protocol=pointcloud_protocol,
    )
    pointcloud_seconds = time.perf_counter() - pointcloud_started
    summary = {
        "method": "horizonstream",
        "scene": args.scene_name,
        "frames": len(images),
        "image_dir": str(Path(args.image_dir).expanduser().resolve()),
        "checkpoint": str(checkpoint),
        "repo_commit": _git_commit(repo),
        "window_size": args.window_size,
        "sliding_size": args.sliding_size,
        "elapsed_seconds": elapsed,
        "frames_per_second": len(images) / elapsed if elapsed else None,
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated(device_index) / 2**30,
        "pointcloud_processing_seconds": pointcloud_seconds,
        "pointcloud_products": pointcloud_products,
        "output": str(scene_dir),
    }
    write_json(scene_dir / "run_summary.json", summary)
    print(f"HorizonStream complete: {summary}", flush=True)


def _install_matplotlib_compatibility() -> None:
    """Restore the cmap API used by the pinned HorizonStream revision."""

    from matplotlib import cm, colormaps

    if hasattr(cm, "get_cmap"):
        return

    def get_cmap(name=None, lut=None):
        color_map = colormaps.get_cmap(name)
        return color_map.resampled(lut) if lut is not None else color_map

    cm.get_cmap = get_cmap


def _git_commit(repo: Path) -> str | None:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


if __name__ == "__main__":
    main()
