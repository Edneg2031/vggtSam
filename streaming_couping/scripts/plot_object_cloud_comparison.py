#!/usr/bin/env python3
"""Object clouds three ways: ground truth, raw, corrected.  (CPU only)

The pose figure shows the trajectory.  This shows what the trajectory was for:
the same object points placed three ways -- by the ground-truth poses, by the
raw HorizonStream poses, and by the corrected ones.

The points are identical in the last two panels; only the pose that puts them
in the world changes.  So identity errors, mask quality and sampling cancel,
and what is left is exactly the thing being claimed.

Reads the same artifacts the object-map evaluation does and reuses its cloud
building, so the picture and the numbers cannot describe different clouds.

    python -m streaming_couping.scripts.plot_object_cloud_comparison \
        --run-dir <run>.baseline --geometry-cache <cache> \
        --manifest <manifest.json> --scene-id 00a231a370 \
        --prompts bed wardrobe chair --out objects.png
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from streaming_couping.scripts.evaluate_pose_feedback_object_map import (
    _load_poses,
    predicted_clouds,
    reference_clouds,
)
from streaming_couping.src.horizonstream_cache import load_horizonstream_cache
from streaming_couping.src.semantic_tracking_metrics import (
    load_ground_truth_instances,
)

GT_COLOR = "#202124"
RAW_COLOR = "#e8710a"
METHOD_COLOR = "#1a73e8"

#: Enough to read the shape, few enough to draw quickly.
DEFAULT_POINTS_PER_CLOUD = 1500
DEFAULT_MAX_OBJECTS = 6


def _subsample(points: torch.Tensor, limit: int, seed: int = 0) -> torch.Tensor:
    """Evenly spaced points, so the sample is deterministic and not clustered."""

    if points.shape[0] <= limit:
        return points
    indices = torch.linspace(0, points.shape[0] - 1, steps=int(limit)).round().long()
    return points.index_select(0, indices)


def _mean_distance(points: torch.Tensor, reference: torch.Tensor) -> float:
    """Mean distance from each point to its nearest reference point.

    The mean, not the median: this is the same quantity the object-map
    evaluation reports as object_accuracy_m, so the number in the panel title
    is the number in the metrics file.  (torch.median also returns the lower of
    two middle values, which would quietly be a different statistic.)
    """

    if points.shape[0] == 0 or reference.shape[0] == 0:
        return float("nan")
    distances = torch.cdist(points.float(), reference.float()).min(dim=1).values
    return float(distances.mean())


def _spread(points: torch.Tensor) -> torch.Tensor:
    """A robust extent, so a few stray points cannot flatten the view."""

    if points.shape[0] == 0:
        return torch.zeros(3)
    low = torch.quantile(points, 0.02, dim=0)
    high = torch.quantile(points, 0.98, dim=0)
    return high - low


def plot_objects(
    *,
    reference: Mapping[str, torch.Tensor],
    raw: Mapping[str, torch.Tensor],
    corrected: Mapping[str, torch.Tensor],
    variant: str,
    title: str,
    out_path: Path,
    max_objects: int,
    points_per_cloud: int,
) -> list[str]:
    """One panel per object: ground truth, raw, corrected, drawn together."""

    labels = [
        label
        for label, cloud in sorted(
            reference.items(), key=lambda item: -int(item[1].shape[0])
        )
        if label in raw and label in corrected
    ][: int(max_objects)]
    if not labels:
        raise ValueError("no object has both a reference cloud and a prediction")

    columns = min(3, len(labels))
    rows = math.ceil(len(labels) / columns)
    figure, axes = plt.subplots(
        rows, columns, figsize=(5.0 * columns, 4.6 * rows), squeeze=False
    )

    for index, label in enumerate(labels):
        axis = axes[index // columns][index % columns]
        truth = _subsample(reference[label], points_per_cloud)
        before = _subsample(raw[label], points_per_cloud)
        after = _subsample(corrected[label], points_per_cloud)

        for points, colour, name, size in (
            (truth, GT_COLOR, "ground truth", 5.0),
            (before, RAW_COLOR, "raw", 4.0),
            (after, METHOD_COLOR, f"corrected ({variant})", 4.0),
        ):
            axis.scatter(
                points[:, 0],
                points[:, 2],
                s=size,
                c=colour,
                alpha=0.45,
                linewidths=0,
                label=name,
            )

        # One scale per panel, taken from ground truth so the panels are
        # comparable to each other rather than each auto-fitting its own error.
        extent = _spread(truth)
        centre = truth.mean(dim=0)
        span = float(extent[[0, 2]].max()) * 0.6 + 1e-6
        axis.set_xlim(float(centre[0]) - span, float(centre[0]) + span)
        axis.set_ylim(float(centre[2]) - span, float(centre[2]) + span)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.2)
        # The top view collapses a vertical face to a line, so a panel can look
        # unchanged while the cloud moved.  The numbers in the title are the
        # same quantity the object-map evaluation reports, and they carry what
        # the projection can hide.
        axis.set_title(
            f"{label}\n"
            f"mean distance to GT   raw {_mean_distance(before, truth):.3f} m"
            f"  →  corrected {_mean_distance(after, truth):.3f} m",
            fontsize=9,
        )
        axis.set_xlabel("x (m)")
        axis.set_ylabel("z (m)")

    for index in range(len(labels), rows * columns):
        axes[index // columns][index % columns].axis("off")

    handles, names = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, names, loc="lower center", ncol=3, fontsize=10)
    figure.suptitle(f"{title}   ·   object clouds, top view (x–z)", fontsize=12)
    figure.tight_layout(rect=(0, 0.07, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=140)
    plt.close(figure)
    return labels


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--geometry-cache", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--prompts", nargs="+", required=True)
    parser.add_argument("--variant", default="robust_semantic")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-objects", type=int, default=DEFAULT_MAX_OBJECTS)
    parser.add_argument("--points-per-cloud", type=int, default=DEFAULT_POINTS_PER_CLOUD)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    poses_path = run_dir / "object_pose_feedback" / "poses.pt"
    diagnostics_path = run_dir / "object_pose_refinement" / "feedback_diagnostics.pt"
    for required in (poses_path, diagnostics_path):
        if not required.is_file():
            raise FileNotFoundError(f"missing {required}")

    cache = load_horizonstream_cache(args.geometry_cache.expanduser().resolve())
    depth = torch.as_tensor(cache["depth"]).detach().float().cpu()
    intrinsics = torch.as_tensor(cache["intrinsics"]).detach().float().cpu()
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    height, width = int(depth.shape[1]), int(depth.shape[2])
    positions = [int(value) for value in cache["source_positions"]]

    poses = _load_poses(poses_path)
    if args.variant not in poses:
        raise KeyError(
            f"{poses_path} has no trajectory {args.variant!r}; "
            f"available: {sorted(poses)}"
        )
    # _load_poses returns the raw trajectory and the variants; the ground truth
    # is a separate key in the payload, not a trajectory it hands back.
    gt_c2w = (
        torch.as_tensor(
            torch.load(poses_path, map_location="cpu", weights_only=False)["gt_c2w"]
        )
        .detach()
        .float()
        .cpu()
    )

    diagnostics = torch.load(diagnostics_path, map_location="cpu", weights_only=False)
    observations = list(diagnostics.get("observations", ()))

    ground_truth = load_ground_truth_instances(
        args.manifest.expanduser().resolve(),
        scene_id=str(args.scene_id),
        frame_indices=positions,
        output_size=(height, width),
        prompts=tuple(args.prompts),
    )
    reference = reference_clouds(
        masks=ground_truth.masks,
        labels=ground_truth.labels,
        depth=depth,
        intrinsics=intrinsics,
        gt_c2w=gt_c2w,
    )

    labels = plot_objects(
        reference=reference,
        raw=predicted_clouds(observations=observations, trajectory=poses["raw"]),
        corrected=predicted_clouds(
            observations=observations, trajectory=poses[args.variant]
        ),
        variant=args.variant,
        title=run_dir.name,
        out_path=args.out.expanduser().resolve(),
        max_objects=args.max_objects,
        points_per_cloud=args.points_per_cloud,
    )
    print(f"run_dir={run_dir}")
    print(f"variant={args.variant}  objects={', '.join(labels)}")
    print(f"figure={args.out.expanduser().resolve()}")


if __name__ == "__main__":
    main()
