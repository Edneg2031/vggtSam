"""Redraw existing HorizonStream and StreamVGGT poses with one plot function."""

from __future__ import annotations

import argparse
from pathlib import Path

from .common import save_trajectory_plot
from .pointcloud_products import read_w2c_txt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default="outputs/long_sequence_baselines/frames_300",
        help="Directory containing horizonstream/ and streamvggt/.",
    )
    parser.add_argument("--scene-name", default="meeting_room_a02")
    return parser.parse_args()


def _redraw(scene_dir: Path, method_name: str) -> Path:
    pose_path = scene_dir / "poses" / "abs_pose.txt"
    if not pose_path.is_file():
        raise FileNotFoundError(f"Pose file not found: {pose_path}")
    _, world_to_camera = read_w2c_txt(pose_path)
    output_path = scene_dir / "plots" / "trajectory_compare.png"
    save_trajectory_plot(
        output_path,
        world_to_camera,
        f"{method_name} prediction (no GT/alignment)",
    )
    return output_path


def main() -> None:
    args = parse_args()
    root = Path(args.output_root).expanduser().resolve()
    outputs = [
        _redraw(root / "horizonstream" / args.scene_name, "HorizonStream"),
        _redraw(root / "streamvggt" / args.scene_name, "StreamVGGT"),
    ]
    print("Unified Horizon-style trajectory plots:", flush=True)
    for output in outputs:
        print(f"  {output}", flush=True)


if __name__ == "__main__":
    main()
