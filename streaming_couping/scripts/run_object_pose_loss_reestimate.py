#!/usr/bin/env python3
"""Close the estimation loop one step: re-solve the proposals against corrected
reference clouds.  (CPU only)

The pipeline is two-pass.  Every proposal is solved from un-corrected geometry,
and only then are the accepted corrections replayed through the pose
accumulator.  So a correction at frame t never reaches the proposal at frame
t+1: the loop is closed in the *propagation* and open in the *estimation*.

That open half is the only half that can be closed.  SAM sees RGB and the
model's forward takes no pose; depth is per-frame and camera-frame.  So nothing
the backbone produces depends on the world pose -- what does depend on it is
the reference cloud the refiner aligns each observation against.  This rebuilds
those clouds under a corrected trajectory and solves the proposals again.

The observations are camera-space points, which no pose change touches, so the
same masks and the same points are used: the only thing that moves is where the
references sit in the world.  This reuses the replay's own loaders and refiner
call, so the second-round solve differs from the first in the base and nothing
else.

THE FALSE POSITIVE TO WATCH FOR

After a correction has been applied, the next round's proposal should be near
identity and the GT correction measured against the corrected base should also
be near identity.  "They agree" is therefore not evidence of anything.  What
this prints is the error of each round's proposals against the base THAT ROUND
was solved from, so the question is whether the residual shrinks or round two
merely restates round one.

    python -m streaming_couping.scripts.run_object_pose_loss_reestimate \
        --diagnostics <run>/object_pose_refinement/feedback_diagnostics.pt \
        --poses <run>/object_pose_feedback/poses.pt \
        --trajectory robust_semantic \
        --output-dir <run>_round2
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from streaming_couping.src.semantic_mapping.object_pose_loss_refinement import (
    ObjectPoseLossRefiner,
)
from streaming_couping.scripts.run_object_pose_loss_replay import (
    SCHEMA,
    config_from_diagnostics,
    load_diagnostics,
    observations_from_diagnostics,
)

#: How far the gauge may vary across frames before this refuses to guess.
GAUGE_TOLERANCE = 1e-3


def resolve_gauge(
    diagnostics_poses: torch.Tensor,
    run_poses: torch.Tensor,
    *,
    tolerance: float = GAUGE_TOLERANCE,
) -> torch.Tensor:
    """The constant transform between two c2w conventions.

    ``feedback_diagnostics.pt`` stores c2w obtained by inverting the cache's
    w2c; ``poses.pt`` stores the c2w the accumulator replays.  Same trajectory,
    but nothing guarantees a shared world origin, and writing one into the
    other's slot unchecked would put every reference cloud in the wrong frame
    -- a mistake that looks like a result rather than an error.

    If ``run[i] == D @ diagnostics[i]`` for one constant ``D``, the conventions
    differ by that gauge.  Anything else is refused.
    """

    if diagnostics_poses.shape != run_poses.shape:
        raise ValueError(
            "pose stacks differ in shape: "
            f"{tuple(diagnostics_poses.shape)} against {tuple(run_poses.shape)}"
        )
    candidates = [
        run.double() @ torch.linalg.inv(diagnostic.double())
        for run, diagnostic in zip(run_poses, diagnostics_poses)
    ]
    gauge = candidates[0]
    worst = max(
        float(torch.linalg.vector_norm(candidate - gauge)) for candidate in candidates
    )
    if worst > float(tolerance):
        raise ValueError(
            "the two representations of the SAME trajectory are not related by a "
            f"single constant transform (worst deviation {worst:.3e}); refusing "
            "to guess which is which"
        )
    return gauge.to(torch.float32)


def rebase(trajectory: torch.Tensor, *, gauge: torch.Tensor) -> torch.Tensor:
    """Express a run-convention c2w trajectory in the diagnostics' convention.

    ``resolve_gauge`` returns ``D`` with ``run = D @ diagnostics``, so going the
    other way left-multiplies by ``inv(D)``.
    """

    return (torch.linalg.inv(gauge.double()) @ trajectory.double()).to(torch.float32)


def residual_proposal_error(
    proposals: Any,
    *,
    gt_c2w: torch.Tensor,
    base_c2w: torch.Tensor,
    frame_ids: tuple[int, ...],
) -> dict[str, float]:
    """Median GT translation/rotation error of this round's proposals.

    The GT correction is taken RELATIVE TO THE BASE THIS ROUND WAS SOLVED FROM.
    Against the raw base it would be a different question, and for a second
    round it would be the wrong one.
    """

    index_of = {int(frame_id): index for index, frame_id in enumerate(frame_ids)}
    translation: list[float] = []
    rotation: list[float] = []
    for row in proposals or ():
        frame_id = int(row.get("frame_id", -1))
        index = index_of.get(frame_id)
        if index is None:
            continue
        # "correction" is best_pose @ inv(raw_pose), i.e. the proposal in the
        # same c2w convention, and it is None on rows that never solved.
        correction = row.get("correction")
        if correction is None:
            continue
        correction = torch.as_tensor(correction).detach().double()
        target = gt_c2w[index].double() @ torch.linalg.inv(base_c2w[index].double())
        relative = torch.linalg.inv(target) @ correction
        translation.append(float(torch.linalg.vector_norm(relative[:3, 3])))
        cosine = float(
            torch.clamp((torch.trace(relative[:3, :3]) - 1.0) * 0.5, -1.0, 1.0)
        )
        skew = 0.5 * (relative[:3, :3] - relative[:3, :3].transpose(0, 1))
        sine = float(
            torch.linalg.vector_norm(
                torch.stack((skew[2, 1], skew[0, 2], skew[1, 0]))
            )
        )
        rotation.append(math.degrees(math.atan2(sine, cosine)))
    if not translation:
        return {"proposal_count": 0}
    translation_tensor = torch.tensor(translation)
    rotation_tensor = torch.tensor(rotation)
    return {
        "proposal_count": len(translation),
        "translation_median_m": float(translation_tensor.median()),
        "rotation_median_deg": float(rotation_tensor.median()),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument(
        "--trajectory",
        default="robust_semantic",
        help="Which corrected trajectory becomes the new base.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=GAUGE_TOLERANCE)
    parser.add_argument(
        "--control",
        action="store_true",
        help=(
            "Re-solve against the UNCHANGED base.  This is the null control: "
            "the result must reproduce the source proposals, and the gap "
            "between it and the loop run is what the re-solve path itself "
            "contributes.  Without it a residual change cannot be attributed "
            "to the loop."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    source_path = args.diagnostics.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    poses_path = args.poses.expanduser().resolve()

    source = load_diagnostics(source_path)
    if source.get("schema") != SCHEMA:
        raise ValueError(
            f"{source_path} has schema {source.get('schema')!r}, expected {SCHEMA!r}"
        )

    poses = torch.load(poses_path, map_location="cpu", weights_only=False)
    variants = poses.get("variant_c2w") or {}
    if args.control:
        # The null control re-solves from the base the source already used, so
        # any difference it shows is the re-solve path and not the loop.
        corrected = torch.as_tensor(poses["raw_c2w"]).detach().float().cpu()
        args.trajectory = "raw (control)"
    else:
        if args.trajectory not in variants:
            raise KeyError(
                f"{poses_path} has no trajectory {args.trajectory!r}; "
                f"available: {sorted(variants)}"
            )
        corrected = torch.as_tensor(variants[args.trajectory]).detach().float().cpu()
    raw_from_poses = torch.as_tensor(poses["raw_c2w"]).detach().float().cpu()
    gt_c2w = torch.as_tensor(poses["gt_c2w"]).detach().float().cpu()

    frame_ids = tuple(int(value) for value in source["frame_ids"])
    base_poses = torch.stack(
        [
            torch.as_tensor(pose).detach().float().cpu()
            for pose in source["raw_camera_to_world"]
        ]
    )
    if len(frame_ids) != int(base_poses.shape[0]):
        raise ValueError("frame_ids and raw_camera_to_world disagree in length.")

    # The gauge comes from the RAW pair -- the same trajectory written two ways.
    # Measuring it against the corrected trajectory would be measuring the
    # correction, which is exactly what must not be assumed to be zero.
    gauge = resolve_gauge(base_poses, raw_from_poses, tolerance=float(args.tolerance))
    new_base = rebase(corrected, gauge=gauge)

    shift = torch.linalg.vector_norm(
        new_base[:, :3, 3] - base_poses[:, :3, 3], dim=-1
    )
    print(f"source={source_path}")
    print(f"base_trajectory={args.trajectory}")
    print(
        f"base_shift_m: median={float(shift.median()):.6f} "
        f"max={float(shift.max()):.6f}"
    )
    if float(shift.max()) <= 0.0 and not args.control:
        raise RuntimeError(
            "the base trajectory is identical to the one the source was solved "
            "from; there is nothing to feed back"
        )

    overrides: dict[str, Any] = {}
    config = config_from_diagnostics(source, overrides)
    observations_by_frame = observations_from_diagnostics(source)
    collection = source.get("collection") or {}
    tracked_ids = {int(value) for value in collection.get("tracked_instance_ids", ())}
    filter_stats = Counter(
        {
            str(key): int(value)
            for key, value in (collection.get("filter_stats") or {}).items()
        }
    )
    raw_observation_count = int(
        collection.get("raw_observation_count")
        or sum(len(rows) for rows in observations_by_frame.values())
    )

    refiner = ObjectPoseLossRefiner(config)
    result = refiner.refine_from_observations(
        frame_ids=frame_ids,
        raw_poses=tuple(new_base[i] for i in range(len(frame_ids))),
        observations_by_frame=observations_by_frame,
        tracked_ids=tracked_ids,
        filter_stats=filter_stats,
        raw_observation_count=raw_observation_count,
    )
    if result.feedback_diagnostics is None:
        raise RuntimeError("re-estimation produced no feedback diagnostics")

    payload = dict(result.feedback_diagnostics)
    payload["reestimate_provenance"] = {
        "source_diagnostics": str(source_path),
        "poses": str(poses_path),
        "base_trajectory": str(args.trajectory),
        "segmentation_reused": True,
        "gpu_used": False,
    }
    refinement_dir = output_dir / "object_pose_refinement"
    refinement_dir.mkdir(parents=True, exist_ok=True)
    torch.save(payload, refinement_dir / "feedback_diagnostics.pt")

    # The base is what stage 2b must score against, so it travels with the
    # diagnostics rather than being rederived from the raw trajectory.
    feedback_dir = output_dir / "object_pose_feedback"
    feedback_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"raw_c2w": new_base, "gt_c2w": gt_c2w, "variant_c2w": {}},
        feedback_dir / "round_base.pt",
    )

    before = residual_proposal_error(
        source.get("proposals"), gt_c2w=gt_c2w, base_c2w=base_poses, frame_ids=frame_ids
    )
    after = residual_proposal_error(
        payload.get("proposals"), gt_c2w=gt_c2w, base_c2w=new_base, frame_ids=frame_ids
    )
    print("")
    print("proposal error against the base that round was solved from:")
    for name, stats in (("round 1", before), ("round 2", after)):
        print(
            f"  {name}: n={stats.get('proposal_count')} "
            f"translation_median={stats.get('translation_median_m', float('nan')):.6f} m "
            f"rotation_median={stats.get('rotation_median_deg', float('nan')):.6f} deg"
        )
    if args.control:
        print(
            "  CONTROL: this re-solved from the UNCHANGED base, so it must "
            "reproduce round 1.  Any gap here is the re-solve path's own "
            "contribution and has to be subtracted from the loop run's change "
            "before that change is attributed to the loop."
        )
    if not after.get("proposal_count"):
        print(
            "  -> round two solved no proposals, so there is nothing to compare. "
            "That means the rebased references produced no usable pairing, which "
            "is itself a result: the corrected base moved the references too far "
            "for the matching thresholds."
        )
    first, second = before.get("translation_median_m"), after.get("translation_median_m")
    if first is not None and second is not None:
        if first <= 0.0:
            # A zero first round means the proposals were already exact, which
            # is a property of the inputs rather than of the loop; there is no
            # ratio to take and reporting one would be dividing by nothing.
            print(
                "  -> round one was already exact, so there is no residual to "
                "shrink; the ratio is undefined and is not reported"
            )
        elif second < first:
            print(
                f"  -> the residual shrank by {(1 - second / first):.1%}; the loop "
                "carries information the first round did not use"
            )
        else:
            print(
                f"  -> the residual did NOT shrink ({(second / first - 1):+.1%}); "
                "round two restates round one"
            )
    print(f"rebased_diagnostics={refinement_dir / 'feedback_diagnostics.pt'}")
    print(f"round_base={feedback_dir / 'round_base.pt'}")


if __name__ == "__main__":
    main()
