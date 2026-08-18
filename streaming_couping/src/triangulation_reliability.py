"""Fixed, GT-blind reliability gates for frozen triangulated anchors."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch


@dataclass(frozen=True)
class ReliabilityGate:
    """A deployable gate whose inputs are available before geometry scoring."""

    name: str
    maximum_mean_reprojection_px: float
    minimum_triangulation_angle_degrees: float
    maximum_condition_number: float
    minimum_views: int
    require_positive_depth: bool = True

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("Reliability gate name cannot be empty.")
        if self.maximum_mean_reprojection_px <= 0.0:
            raise ValueError("Maximum reprojection error must be positive.")
        if self.minimum_triangulation_angle_degrees <= 0.0:
            raise ValueError("Minimum triangulation angle must be positive.")
        if self.maximum_condition_number <= 1.0:
            raise ValueError("Maximum condition number must exceed one.")
        if self.minimum_views < 2:
            raise ValueError("A triangulation gate requires at least two views.")


FIXED_BIN_LABELS = {
    "mean_reprojection_error_px": (
        "0_to_1_px",
        "1_to_2_px",
        "2_to_4_px",
        "4_to_8_px",
        "8_plus_px",
    ),
    "maximum_ray_angle_degrees": (
        "lt_1_deg",
        "1_to_2_deg",
        "2_to_5_deg",
        "5_to_10_deg",
        "10_plus_deg",
    ),
    "condition_number": (
        "lt_1e2",
        "1e2_to_5e2",
        "5e2_to_1e3",
        "1e3_to_3e3",
        "3e3_plus",
    ),
    "num_views": ("2_views", "3_views", "4_plus_views"),
}


def reliability_gate_mask(
    metrics: Mapping[str, torch.Tensor], gate: ReliabilityGate
) -> torch.Tensor:
    """Return a boolean mask using non-GT reliability fields only."""

    gate.validate()
    required = {
        "mean_reprojection_error_px",
        "maximum_ray_angle_degrees",
        "condition_number",
        "num_views",
        "positive_depth",
    }
    missing = required.difference(metrics)
    if missing:
        raise ValueError(f"Reliability metrics are missing {sorted(missing)}.")
    values = {name: torch.as_tensor(metrics[name]).reshape(-1) for name in required}
    lengths = {int(value.numel()) for value in values.values()}
    if len(lengths) != 1:
        raise ValueError("Reliability metric tensors must have equal length.")
    reprojection = values["mean_reprojection_error_px"].float()
    angle = values["maximum_ray_angle_degrees"].float()
    condition = values["condition_number"].float()
    views = values["num_views"].long()
    positive = values["positive_depth"].bool()
    finite = (
        torch.isfinite(reprojection)
        & torch.isfinite(angle)
        & torch.isfinite(condition)
    )
    accepted = (
        finite
        & (reprojection <= gate.maximum_mean_reprojection_px)
        & (angle >= gate.minimum_triangulation_angle_degrees)
        & (condition <= gate.maximum_condition_number)
        & (views >= gate.minimum_views)
    )
    if gate.require_positive_depth:
        accepted &= positive
    return accepted


def fixed_bin_label(metric: str, value: float | int) -> str:
    """Assign one protocol-defined reliability bin with fixed boundaries."""

    number = float(value)
    if not math.isfinite(number):
        return "nonfinite"
    if metric == "mean_reprojection_error_px":
        return _edge_label(number, (1.0, 2.0, 4.0, 8.0), FIXED_BIN_LABELS[metric])
    if metric == "maximum_ray_angle_degrees":
        return _edge_label(number, (1.0, 2.0, 5.0, 10.0), FIXED_BIN_LABELS[metric])
    if metric == "condition_number":
        return _edge_label(
            number,
            (100.0, 500.0, 1000.0, 3000.0),
            FIXED_BIN_LABELS[metric],
        )
    if metric == "num_views":
        if int(number) <= 2:
            return "2_views"
        if int(number) == 3:
            return "3_views"
        return "4_plus_views"
    raise ValueError(f"Unknown reliability bin metric {metric!r}.")


def paired_error_metrics(
    triangulated_errors: torch.Tensor,
    raw_errors: torch.Tensor,
) -> dict[str, float | int]:
    """Compute paired error statistics for one frozen subset."""

    tri = torch.as_tensor(triangulated_errors).detach().float().reshape(-1)
    raw = torch.as_tensor(raw_errors).detach().float().reshape(-1)
    if tri.shape != raw.shape:
        raise ValueError("Triangulated and raw error vectors must have equal shape.")
    finite = torch.isfinite(tri) & torch.isfinite(raw)
    tri = tri[finite]
    raw = raw[finite]
    if tri.numel() == 0:
        return {
            "anchor_count": 0,
            "tri_rmse": float("nan"),
            "tri_median": float("nan"),
            "tri_p90": float("nan"),
            "raw_rmse": float("nan"),
            "raw_median": float("nan"),
            "raw_p90": float("nan"),
            "tri_gain_vs_raw_percent": float("nan"),
            "improved_anchor_ratio": float("nan"),
            "median_delta_raw_minus_tri": float("nan"),
        }
    tri_rmse = float(torch.sqrt(tri.square().mean()))
    raw_rmse = float(torch.sqrt(raw.square().mean()))
    delta = raw - tri
    return {
        "anchor_count": int(tri.numel()),
        "tri_rmse": tri_rmse,
        "tri_median": float(tri.median()),
        "tri_p90": float(torch.quantile(tri, 0.90)),
        "raw_rmse": raw_rmse,
        "raw_median": float(raw.median()),
        "raw_p90": float(torch.quantile(raw, 0.90)),
        "tri_gain_vs_raw_percent": 100.0
        * (raw_rmse - tri_rmse)
        / max(raw_rmse, 1e-12),
        "improved_anchor_ratio": float((tri < raw).float().mean()),
        "median_delta_raw_minus_tri": float(delta.median()),
    }


def reliability_gate_pass(
    metrics: Mapping[str, float | int],
    *,
    minimum_anchor_count: int,
    minimum_retained_ratio: float,
) -> bool:
    """Apply the predeclared T0.1 GO rule to one scored gate."""

    return bool(
        int(metrics["anchor_count"]) >= int(minimum_anchor_count)
        and float(metrics["retained_ratio"]) >= float(minimum_retained_ratio)
        and float(metrics["tri_rmse"]) < float(metrics["raw_rmse"])
        and float(metrics["tri_p90"]) <= float(metrics["raw_p90"])
        and float(metrics["improved_anchor_ratio"]) > 0.5
    )


def _edge_label(
    value: float, edges: tuple[float, ...], labels: tuple[str, ...]
) -> str:
    for edge, label in zip(edges, labels):
        if value < edge:
            return label
    return labels[-1]
