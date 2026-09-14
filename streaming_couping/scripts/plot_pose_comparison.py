#!/usr/bin/env python3
"""Draw what the pose correction actually changed.  (CPU only)

The reported number is a percentage, and a percentage does not show what
happened.  This plots the same trajectory three ways -- ground truth, the raw
HorizonStream output, and the corrected one -- so the drift and the correction
are visible rather than asserted.

Three panels:

* the top-down path, where accumulated drift shows up as the raw path pulling
  away from ground truth while the corrected one stays on it;
* translation error against frame index, which shows WHERE the correction
  acts and that it does not simply rescale the whole sequence;
* rotation error the same way, because the translation and rotation halves of
  a correction are not the same story.

Reads ``<run>/object_pose_feedback/poses.pt``, which stage 2b already writes.

    python -m streaming_couping.scripts.plot_pose_comparison \
        --run-dir <run>.baseline --out pose_comparison.png
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")  # no display on a compute node

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

#: Drawn in this order, so the corrected path sits on top of the raw one.
RAW_COLOR = "#9aa0a6"
GT_COLOR = "#202124"
VARIANT_COLOR = "#1a73e8"


def load_poses(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    for required in ("raw_c2w", "gt_c2w"):
        if required not in payload:
            raise KeyError(f"{path} has no {required!r}; cannot plot a comparison")
    trajectories = {
        "raw": torch.as_tensor(payload["raw_c2w"]).detach().float().cpu().numpy(),
        "gt": torch.as_tensor(payload["gt_c2w"]).detach().float().cpu().numpy(),
    }
    for name, value in (payload.get("variant_c2w") or {}).items():
        trajectories[str(name)] = (
            torch.as_tensor(value).detach().float().cpu().numpy()
        )
    return trajectories


def rotation_error_deg(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Per-frame rotation difference in degrees.

    atan2(sin, cos), not acos of the trace: near identity the trace is flat, so
    acos amplifies float noise into whole tenths of a degree.  Same reason the
    pipeline's own metric uses it.
    """

    relative = np.einsum("nij,nkj->nik", left[:, :3, :3], right[:, :3, :3])
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) * 0.5, -1.0, 1.0)
    skew = 0.5 * (relative - np.transpose(relative, (0, 2, 1)))
    sine = np.linalg.norm(
        np.stack((skew[:, 2, 1], skew[:, 0, 2], skew[:, 1, 0]), axis=-1), axis=-1
    )
    return np.degrees(np.arctan2(sine, cosine))


def translation_error_m(trajectory: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.linalg.norm(trajectory[:, :3, 3] - reference[:, :3, 3], axis=-1)


def plot(
    trajectories: Mapping[str, np.ndarray],
    *,
    variant: str,
    title: str,
    out_path: Path,
) -> dict[str, float]:
    ground_truth = trajectories["gt"]
    raw = trajectories["raw"]
    corrected = trajectories[variant]
    frames = np.arange(raw.shape[0])

    raw_translation = translation_error_m(raw, ground_truth)
    fixed_translation = translation_error_m(corrected, ground_truth)
    raw_rotation = rotation_error_deg(raw, ground_truth)
    fixed_rotation = rotation_error_deg(corrected, ground_truth)

    figure, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))

    # --- the path itself, looking down ------------------------------------
    path_axis = axes[0]
    path_axis.plot(
        ground_truth[:, 0, 3],
        ground_truth[:, 2, 3],
        color=GT_COLOR,
        linewidth=2.0,
        linestyle="--",
        label="ground truth",
        zorder=3,
    )
    path_axis.plot(
        raw[:, 0, 3],
        raw[:, 2, 3],
        color=RAW_COLOR,
        linewidth=2.0,
        label="raw (HorizonStream)",
        zorder=2,
    )
    path_axis.plot(
        corrected[:, 0, 3],
        corrected[:, 2, 3],
        color=VARIANT_COLOR,
        linewidth=2.0,
        label=f"corrected ({variant})",
        zorder=4,
    )
    path_axis.scatter(
        ground_truth[0, 0, 3], ground_truth[0, 2, 3], color=GT_COLOR, s=30, zorder=5
    )
    # The paths overlap when the drift is small against the path length -- which
    # is the normal case -- so the panel orients the reader and the numbers in
    # the title carry the magnitude.
    path_axis.set_title(
        "path, top view (x–z)\n"
        f"translation RMSE  raw {np.sqrt(np.mean(raw_translation**2)):.4f} m"
        f"  →  corrected {np.sqrt(np.mean(fixed_translation**2)):.4f} m",
        fontsize=10,
    )
    path_axis.set_xlabel("x (m)")
    path_axis.set_ylabel("z (m)")
    path_axis.set_aspect("equal", adjustable="datalim")
    path_axis.grid(alpha=0.25)
    path_axis.legend(fontsize=8, loc="best")
    # Draw the raw path again, dashed and thin, so overlap with the corrected
    # path is visible as overlap rather than inferred from two solid lines.
    path_axis.plot(
        raw[:, 0, 3],
        raw[:, 2, 3],
        color=RAW_COLOR,
        linewidth=1.0,
        linestyle=":",
        zorder=6,
    )

    # --- error against frame index ----------------------------------------
    for axis, raw_error, fixed_error, name, unit in (
        (axes[1], raw_translation, fixed_translation, "translation", "m"),
        (axes[2], raw_rotation, fixed_rotation, "rotation", "deg"),
    ):
        axis.plot(frames, raw_error, color=RAW_COLOR, linewidth=1.6, label="raw")
        axis.plot(
            frames,
            fixed_error,
            color=VARIANT_COLOR,
            linewidth=1.6,
            label=f"corrected ({variant})",
        )
        axis.set_title(
            f"{name} error against ground truth\n"
            f"RMSE  raw {np.sqrt(np.mean(raw_error**2)):.4f} {unit}"
            f"  →  corrected {np.sqrt(np.mean(fixed_error**2)):.4f} {unit}",
            fontsize=10,
        )
        axis.set_xlabel("frame index")
        axis.set_ylabel(f"{name} error ({unit})")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, loc="best")
        axis.axhline(0.0, color="#dadce0", linewidth=0.8, zorder=0)

    figure.suptitle(title, fontsize=12)
    figure.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=140)
    plt.close(figure)

    def improvement(before: np.ndarray, after: np.ndarray) -> float:
        baseline = float(np.sqrt(np.mean(before**2)))
        if baseline <= 0.0:
            return float("nan")
        return (baseline - float(np.sqrt(np.mean(after**2)))) / baseline

    return {
        "raw_translation_rmse_m": float(np.sqrt(np.mean(raw_translation**2))),
        "corrected_translation_rmse_m": float(np.sqrt(np.mean(fixed_translation**2))),
        "raw_rotation_rmse_deg": float(np.sqrt(np.mean(raw_rotation**2))),
        "corrected_rotation_rmse_deg": float(np.sqrt(np.mean(fixed_rotation**2))),
        "translation_improvement_ratio": improvement(raw_translation, fixed_translation),
        "rotation_improvement_ratio": improvement(raw_rotation, fixed_rotation),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--variant",
        default="robust_semantic",
        help="Which corrected trajectory to draw; defaults to the main variant.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    poses_path = run_dir / "object_pose_feedback" / "poses.pt"
    if not poses_path.is_file():
        raise FileNotFoundError(f"missing {poses_path}")
    trajectories = load_poses(poses_path)
    if args.variant not in trajectories:
        available = sorted(name for name in trajectories if name not in {"gt"})
        raise KeyError(
            f"{poses_path} has no trajectory {args.variant!r}; available: {available}"
        )

    stats = plot(
        trajectories,
        variant=args.variant,
        title=f"{run_dir.name}  ·  {args.variant}",
        out_path=args.out.expanduser().resolve(),
    )
    print(f"run_dir={run_dir}")
    print(f"frames={trajectories['raw'].shape[0]}  variant={args.variant}")
    for key, value in stats.items():
        label = f"{value:+.2%}" if key.endswith("improvement_ratio") else f"{value:.5f}"
        print(f"  {key}={label}")
    print(f"figure={args.out.expanduser().resolve()}")
    if args.json_out is not None:
        import json

        args.json_out.expanduser().resolve().write_text(
            json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
