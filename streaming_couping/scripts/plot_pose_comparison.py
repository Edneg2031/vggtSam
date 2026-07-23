#!/usr/bin/env python3
"""Plot GT/raw/ours camera trajectories in the shared GT-world gauge."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class PoseSeries:
    method: str
    label: str
    frame_indices: np.ndarray
    c2w: np.ndarray

    @property
    def centers(self) -> np.ndarray:
        return self.c2w[:, :3, 3]

    @property
    def forward(self) -> np.ndarray:
        # StreamVGGT/ScanNet++ pinhole convention: camera looks along +Z.
        return self.c2w[:, :3, 2]


SERIES = {
    "ground_truth": ("GT", "#2ca02c", "o"),
    "streamvggt_raw": ("Raw StreamVGGT", "#d62728", "x"),
    "ours_v2_pointmap_v3_pose": ("Ours", "#1f77b4", "s"),
}


def main() -> None:
    args = _parse_args()
    comparison_dir = Path(args.comparison_dir).expanduser().resolve()
    series = _read_pose_series(comparison_dir / "camera_poses.csv")
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else comparison_dir / "pose_comparison"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    written = _plot_pose_graph(
        series,
        output=output,
        heldout_frames={int(value) for value in args.heldout_frame_indices},
    )
    print("pose graph written to " + ", ".join(str(path) for path in written))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison-dir",
        required=True,
        help="Directory containing comparison_gt_world/camera_poses.csv.",
    )
    parser.add_argument(
        "--output",
        help="Output path without extension; defaults to <comparison-dir>/pose_comparison.",
    )
    parser.add_argument(
        "--heldout-frame-indices",
        nargs="*",
        type=int,
        default=(),
        help="Optional frames to emphasize with a black outer marker.",
    )
    return parser.parse_args()


def _read_pose_series(path: Path) -> dict[str, PoseSeries]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run commands_final_joint_pointcloud_pose.txt first."
        )
    grouped: dict[str, list[tuple[int, int, np.ndarray]]] = {
        method: [] for method in SERIES
    }
    with path.open("r", encoding="utf8", newline="") as handle:
        for row in csv.DictReader(handle):
            method = str(row["variant"])
            if method not in grouped:
                continue
            matrix = np.fromstring(
                str(row["camera_to_world_4x4"]),
                sep=" ",
                dtype=np.float64,
            )
            if matrix.size != 16 or not np.isfinite(matrix).all():
                raise ValueError(
                    f"Invalid c2w matrix method={method} frame={row['frame_index']}."
                )
            grouped[method].append(
                (
                    int(row["sequence_index"]),
                    int(row["frame_index"]),
                    matrix.reshape(4, 4),
                )
            )
    output = {}
    for method, (label, _, _) in SERIES.items():
        values = sorted(grouped[method], key=lambda item: item[0])
        if not values:
            raise ValueError(f"No camera pose rows found for {method!r} in {path}.")
        output[method] = PoseSeries(
            method=method,
            label=label,
            frame_indices=np.asarray([item[1] for item in values], dtype=np.int64),
            c2w=np.stack([item[2] for item in values]),
        )
    expected = output["ground_truth"].frame_indices
    for method, value in output.items():
        if not np.array_equal(value.frame_indices, expected):
            raise ValueError(f"Frame mismatch between GT and {method}.")
    return output


def _plot_pose_graph(
    series: dict[str, PoseSeries],
    *,
    output: Path,
    heldout_frames: set[int],
) -> tuple[Path, ...]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        del error
        path = output.with_suffix(".svg")
        _write_svg_fallback(series, path=path, heldout_frames=heldout_frames)
        return (path,)

    gt = series["ground_truth"]
    raw = series["streamvggt_raw"]
    ours = series["ours_v2_pointmap_v3_pose"]
    arrow_scale = _trajectory_scale(gt.centers) * 0.08
    projection_axes = _largest_projection_axes(gt.centers)
    axis_names = ("X", "Y", "Z")

    figure = plt.figure(figsize=(14, 10), constrained_layout=True)
    grid = figure.add_gridspec(2, 2)
    trajectory_3d = figure.add_subplot(grid[0, 0], projection="3d")
    trajectory_2d = figure.add_subplot(grid[0, 1])
    translation_axis = figure.add_subplot(grid[1, 0])
    rotation_axis = figure.add_subplot(grid[1, 1])

    for method, value in series.items():
        label, color, marker = SERIES[method]
        centers = value.centers
        trajectory_3d.plot(
            centers[:, 0],
            centers[:, 1],
            centers[:, 2],
            color=color,
            marker=marker,
            linewidth=1.8,
            markersize=5,
            label=label,
        )
        trajectory_3d.quiver(
            centers[:, 0],
            centers[:, 1],
            centers[:, 2],
            value.forward[:, 0],
            value.forward[:, 1],
            value.forward[:, 2],
            length=arrow_scale,
            normalize=True,
            color=color,
            alpha=0.55,
            linewidth=0.8,
        )
        first_axis, second_axis = projection_axes
        trajectory_2d.plot(
            centers[:, first_axis],
            centers[:, second_axis],
            color=color,
            marker=marker,
            linewidth=1.8,
            markersize=5,
            label=label,
        )
        for frame_index, center in zip(value.frame_indices, centers):
            trajectory_2d.annotate(
                str(int(frame_index)),
                (center[first_axis], center[second_axis]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
                color=color,
                alpha=0.85,
            )
        _mark_heldout(
            trajectory_3d,
            value,
            heldout_frames=heldout_frames,
            is_3d=True,
        )
        _mark_heldout(
            trajectory_2d,
            value,
            heldout_frames=heldout_frames,
            is_3d=False,
            projection_axes=projection_axes,
        )

    translation_raw = np.linalg.norm(raw.centers - gt.centers, axis=1)
    translation_ours = np.linalg.norm(ours.centers - gt.centers, axis=1)
    rotation_raw = _rotation_errors(raw.c2w[:, :3, :3], gt.c2w[:, :3, :3])
    rotation_ours = _rotation_errors(ours.c2w[:, :3, :3], gt.c2w[:, :3, :3])
    frames = gt.frame_indices
    for axis, raw_values, ours_values, ylabel in (
        (
            translation_axis,
            translation_raw,
            translation_ours,
            "Camera-center error (m)",
        ),
        (
            rotation_axis,
            rotation_raw,
            rotation_ours,
            "Rotation error (deg)",
        ),
    ):
        axis.plot(
            frames,
            raw_values,
            color=SERIES["streamvggt_raw"][1],
            marker=SERIES["streamvggt_raw"][2],
            linewidth=1.8,
            label="Raw StreamVGGT",
        )
        axis.plot(
            frames,
            ours_values,
            color=SERIES["ours_v2_pointmap_v3_pose"][1],
            marker=SERIES["ours_v2_pointmap_v3_pose"][2],
            linewidth=1.8,
            label="Ours",
        )
        axis.set_xlabel("Source frame index")
        axis.set_ylabel(ylabel)
        axis.set_xticks(frames)
        axis.grid(True, linewidth=0.5, alpha=0.3)
        axis.legend()
        for frame_index in heldout_frames:
            if frame_index in frames:
                axis.axvline(frame_index, color="black", linewidth=0.8, alpha=0.22)

    trajectory_3d.set_title("3D pose graph and camera forward directions")
    trajectory_3d.set_xlabel("X (m)")
    trajectory_3d.set_ylabel("Y (m)")
    trajectory_3d.set_zlabel("Z (m)")
    trajectory_3d.legend()
    _set_equal_3d(trajectory_3d, np.concatenate([v.centers for v in series.values()]))

    first_axis, second_axis = projection_axes
    trajectory_2d.set_title(
        f"Trajectory projection on {axis_names[first_axis]}–{axis_names[second_axis]}"
    )
    trajectory_2d.set_xlabel(f"{axis_names[first_axis]} (m)")
    trajectory_2d.set_ylabel(f"{axis_names[second_axis]} (m)")
    trajectory_2d.set_aspect("equal", adjustable="datalim")
    trajectory_2d.grid(True, linewidth=0.5, alpha=0.3)
    trajectory_2d.legend()

    figure.suptitle(
        "GT vs raw StreamVGGT vs ours — one shared GT-world Sim(3)",
        fontsize=14,
    )
    png_path = output.with_suffix(".png")
    pdf_path = output.with_suffix(".pdf")
    figure.savefig(png_path, dpi=220)
    figure.savefig(pdf_path)
    plt.close(figure)
    return png_path, pdf_path


def _rotation_errors(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    relative = np.einsum("sji,sjk->sik", target, predicted)
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) * 0.5, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def _trajectory_scale(centers: np.ndarray) -> float:
    extent = np.ptp(centers, axis=0)
    return max(float(np.linalg.norm(extent)), 1e-3)


def _largest_projection_axes(centers: np.ndarray) -> tuple[int, int]:
    order = np.argsort(np.ptp(centers, axis=0))[::-1]
    return int(order[0]), int(order[1])


def _set_equal_3d(axis, centers: np.ndarray) -> None:
    minimum = centers.min(axis=0)
    maximum = centers.max(axis=0)
    midpoint = 0.5 * (minimum + maximum)
    radius = max(float((maximum - minimum).max()) * 0.55, 1e-3)
    axis.set_xlim(midpoint[0] - radius, midpoint[0] + radius)
    axis.set_ylim(midpoint[1] - radius, midpoint[1] + radius)
    axis.set_zlim(midpoint[2] - radius, midpoint[2] + radius)


def _mark_heldout(
    axis,
    value: PoseSeries,
    *,
    heldout_frames: set[int],
    is_3d: bool,
    projection_axes: tuple[int, int] = (0, 1),
) -> None:
    selected = np.asarray(
        [int(frame) in heldout_frames for frame in value.frame_indices],
        dtype=bool,
    )
    if not selected.any():
        return
    centers = value.centers[selected]
    if is_3d:
        axis.scatter(
            centers[:, 0],
            centers[:, 1],
            centers[:, 2],
            s=75,
            facecolors="none",
            edgecolors="black",
            linewidths=1.0,
        )
    else:
        first, second = projection_axes
        axis.scatter(
            centers[:, first],
            centers[:, second],
            s=75,
            facecolors="none",
            edgecolors="black",
            linewidths=1.0,
        )


def _write_svg_fallback(
    series: dict[str, PoseSeries],
    *,
    path: Path,
    heldout_frames: set[int],
) -> None:
    """Dependency-free fallback when matplotlib is unavailable."""

    gt = series["ground_truth"]
    raw = series["streamvggt_raw"]
    ours = series["ours_v2_pointmap_v3_pose"]
    projection_axes = _largest_projection_axes(gt.centers)
    first_axis, second_axis = projection_axes
    axis_names = ("X", "Y", "Z")
    panels = {
        "trajectory": (70.0, 65.0, 1060.0, 390.0),
        "translation": (70.0, 525.0, 500.0, 245.0),
        "rotation": (630.0, 525.0, 500.0, 245.0),
    }
    all_centers = np.concatenate([value.centers for value in series.values()])
    x_values = all_centers[:, first_axis]
    y_values = all_centers[:, second_axis]
    trajectory_map = _svg_mapper(
        x_values,
        y_values,
        panels["trajectory"],
        padding=35.0,
    )
    translation_raw = np.linalg.norm(raw.centers - gt.centers, axis=1)
    translation_ours = np.linalg.norm(ours.centers - gt.centers, axis=1)
    rotation_raw = _rotation_errors(raw.c2w[:, :3, :3], gt.c2w[:, :3, :3])
    rotation_ours = _rotation_errors(ours.c2w[:, :3, :3], gt.c2w[:, :3, :3])
    frame_values = gt.frame_indices.astype(np.float64)
    translation_map = _svg_mapper(
        frame_values,
        np.concatenate([translation_raw, translation_ours]),
        panels["translation"],
        padding=34.0,
    )
    rotation_map = _svg_mapper(
        frame_values,
        np.concatenate([rotation_raw, rotation_ours]),
        panels["rotation"],
        padding=34.0,
    )

    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="820" '
        'viewBox="0 0 1200 820">',
        '<rect width="1200" height="820" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222}'
        '.title{font-size:20px;font-weight:600}.panel{font-size:15px;font-weight:600}'
        '.label{font-size:12px}.tick{font-size:10px;fill:#555}'
        '.axis{stroke:#777;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}'
        '.held{fill:none;stroke:#111;stroke-width:1.3}</style>',
        '<text x="600" y="30" text-anchor="middle" class="title">'
        "GT vs raw StreamVGGT vs ours — shared GT-world Sim(3)</text>",
    ]
    for panel in panels.values():
        lines.extend(_svg_panel_axes(panel))
    lines.append(
        f'<text x="70" y="55" class="panel">Trajectory projection '
        f'{axis_names[first_axis]}–{axis_names[second_axis]} with camera forward</text>'
    )
    lines.append(
        '<text x="70" y="515" class="panel">Camera-center error (m)</text>'
    )
    lines.append(
        '<text x="630" y="515" class="panel">Rotation error (deg)</text>'
    )

    for method, value in series.items():
        label, color, marker = SERIES[method]
        points = [
            trajectory_map(center[first_axis], center[second_axis])
            for center in value.centers
        ]
        lines.append(
            f'<polyline points="{_svg_points(points)}" fill="none" '
            f'stroke="{color}" stroke-width="2"/>'
        )
        forward_scale = _trajectory_scale(gt.centers) * 0.06
        for frame_index, center, forward, point in zip(
            value.frame_indices,
            value.centers,
            value.forward,
            points,
        ):
            end = trajectory_map(
                center[first_axis] + forward_scale * forward[first_axis],
                center[second_axis] + forward_scale * forward[second_axis],
            )
            lines.append(
                f'<line x1="{point[0]:.2f}" y1="{point[1]:.2f}" '
                f'x2="{end[0]:.2f}" y2="{end[1]:.2f}" '
                f'stroke="{color}" stroke-width="1" opacity="0.65"/>'
            )
            lines.append(_svg_marker(point, color=color, marker=marker))
            lines.append(
                f'<text x="{point[0] + 5:.2f}" y="{point[1] - 5:.2f}" '
                f'class="tick">{int(frame_index)}</text>'
            )
            if int(frame_index) in heldout_frames:
                lines.append(
                    f'<circle cx="{point[0]:.2f}" cy="{point[1]:.2f}" '
                    'r="8" class="held"/>'
                )
        legend_y = 80 + 22 * list(SERIES).index(method)
        lines.append(_svg_marker((965.0, legend_y), color=color, marker=marker))
        lines.append(
            f'<text x="980" y="{legend_y + 4}" class="label">{label}</text>'
        )

    for mapper, raw_values, ours_values in (
        (translation_map, translation_raw, translation_ours),
        (rotation_map, rotation_raw, rotation_ours),
    ):
        for values, method in (
            (raw_values, "streamvggt_raw"),
            (ours_values, "ours_v2_pointmap_v3_pose"),
        ):
            _, color, marker = SERIES[method]
            points = [
                mapper(float(frame), float(value))
                for frame, value in zip(frame_values, values)
            ]
            lines.append(
                f'<polyline points="{_svg_points(points)}" fill="none" '
                f'stroke="{color}" stroke-width="2"/>'
            )
            lines.extend(
                _svg_marker(point, color=color, marker=marker) for point in points
            )
        for frame in frame_values:
            x, y = mapper(float(frame), 0.0)
            lines.append(
                f'<text x="{x:.2f}" y="{mapper.bottom + 18:.2f}" '
                f'text-anchor="middle" class="tick">{int(frame)}</text>'
            )
            if int(frame) in heldout_frames:
                lines.append(
                    f'<line x1="{x:.2f}" y1="{mapper.top:.2f}" '
                    f'x2="{x:.2f}" y2="{mapper.bottom:.2f}" '
                    'stroke="#111" stroke-width="1" opacity="0.2"/>'
                )
    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


@dataclass(frozen=True)
class _SvgMapper:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    left: float
    right: float
    top: float
    bottom: float

    def __call__(self, x: float, y: float) -> tuple[float, float]:
        x_fraction = (float(x) - self.x_min) / max(self.x_max - self.x_min, 1e-9)
        y_fraction = (float(y) - self.y_min) / max(self.y_max - self.y_min, 1e-9)
        return (
            self.left + x_fraction * (self.right - self.left),
            self.bottom - y_fraction * (self.bottom - self.top),
        )


def _svg_mapper(
    x_values: np.ndarray,
    y_values: np.ndarray,
    panel: tuple[float, float, float, float],
    *,
    padding: float,
) -> _SvgMapper:
    x, y, width, height = panel
    x_min, x_max = float(np.min(x_values)), float(np.max(x_values))
    y_min = min(0.0, float(np.min(y_values)))
    y_max = float(np.max(y_values))
    x_margin = max((x_max - x_min) * 0.07, 1e-6)
    y_margin = max((y_max - y_min) * 0.10, 1e-6)
    return _SvgMapper(
        x_min=x_min - x_margin,
        x_max=x_max + x_margin,
        y_min=y_min - y_margin,
        y_max=y_max + y_margin,
        left=x + padding,
        right=x + width - padding,
        top=y + padding,
        bottom=y + height - padding,
    )


def _svg_panel_axes(
    panel: tuple[float, float, float, float],
) -> list[str]:
    x, y, width, height = panel
    return [
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
        'fill="none" stroke="#bbb" stroke-width="1"/>',
    ]


def _svg_points(points: Iterable[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def _svg_marker(
    point: tuple[float, float],
    *,
    color: str,
    marker: str,
) -> str:
    x, y = point
    if marker == "x":
        return (
            f'<path d="M {x - 4:.2f} {y - 4:.2f} L {x + 4:.2f} {y + 4:.2f} '
            f'M {x - 4:.2f} {y + 4:.2f} L {x + 4:.2f} {y - 4:.2f}" '
            f'stroke="{color}" stroke-width="2" fill="none"/>'
        )
    if marker == "s":
        return (
            f'<rect x="{x - 3.5:.2f}" y="{y - 3.5:.2f}" width="7" height="7" '
            f'fill="{color}"/>'
        )
    return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.8" fill="{color}"/>'


if __name__ == "__main__":
    main()
