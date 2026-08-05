#!/usr/bin/env python3
"""Synthetic end-to-end tensor smoke for the V8 O2.5 runner."""

from __future__ import annotations

import torch

from streaming_couping.src.v80_pose_geometry import (
    causal_gt_nearest_pairs_multi_history,
    causal_history_bank_indices,
)
from streaming_couping.scripts.run_v80_geometry_support_ablation import (
    BRANCHES,
    PreparedTokens,
    SupportSpec,
    _frame_row,
    load_o25_config,
)


def main() -> None:
    torch.manual_seed(805)
    config = load_o25_config(
        "streaming_couping/configs/v80_geometry_support_ablation.yaml"
    )
    sequence, instances, points = 2, 2, 8
    one_frame = torch.randn(instances, points, 3, dtype=torch.float64)
    one_frame[..., 2] += 3.0
    camera = one_frame[None].repeat(sequence, 1, 1, 1)
    valid = torch.ones(sequence, instances, points, dtype=torch.bool)
    write = torch.ones(sequence, instances, dtype=torch.bool)
    history = causal_history_bank_indices(write, valid, max_history=1)
    tokens = PreparedTokens(
        local_valid=valid,
        gt_world=camera,
        gt_world_valid=valid,
        gt_camera=camera,
        gt_camera_valid=valid,
        depth_camera=camera,
        depth_camera_valid=valid,
        point_head_camera=camera,
        point_head_camera_valid=valid,
        history_bank=history,
    )
    pairs = causal_gt_nearest_pairs_multi_history(
        current_frame=1,
        history_indices=history[1],
        gt_world_metric=camera,
        gt_valid=valid,
        max_distance_metric=0.15,
        require_mutual_nearest=True,
    )
    baseline = torch.eye(4, dtype=torch.float64)[None].repeat(sequence, 1, 1)
    quality = torch.ones(sequence, instances, 3)
    row = _frame_row(
        phase="smoke",
        fold="smoke",
        frame_index=105,
        sequence_index=1,
        frames=(90, 105),
        spec=SupportSpec(64, 1, "mutual", 0.15),
        branch=BRANCHES[0],
        solver=config.theory.solvers[0],
        pairs=pairs,
        tokens=tokens,
        baseline=baseline,
        target=baseline,
        quality=quality,
        scale=1.0,
        theory=config.theory,
        gates=config.gates,
    )
    _require(int(row["correspondences"]) == instances * points, "pair count")
    _require(int(row["bounded_accept"]) == 1, "bounded candidate")
    _require(int(row["reliable_accept"]) == 1, "reliable candidate")
    _require(int(row["instance_candidates"]) == instances, "instance solves")
    _require(int(row["instance_consensus_pass"]) == 1, "instance consensus")
    _require(int(row["consensus_accept"]) == 1, "consensus candidate")
    _require(float(row["head_consistency_p90_metric"]) == 0.0, "head agreement")
    _require(int(row["consensus_fallback_exact"]) == 1, "fallback integrity")
    print("V8 O2.5 end-to-end tensor smoke passed")


def _require(condition: bool, label: str) -> None:
    if not condition:
        raise RuntimeError(f"V8 O2.5 smoke failed: {label}")


if __name__ == "__main__":
    main()
