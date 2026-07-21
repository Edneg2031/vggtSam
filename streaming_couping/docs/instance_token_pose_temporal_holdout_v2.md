# Decoupled V2 causal temporal holdout

This experiment tests whether the learned instance-token adapters extrapolate
beyond the frames used for supervision.

## Split

- Causal cache context: `90 105 119 130 140 210 240`
- Adapter supervision: `90 105 119 130 140`
- Checkpoint selection: training-supervision loss only
- Held-out evaluation: `210 240`

The seven-frame StreamVGGT cache is safe to reuse because it was generated
with `streaming_cache: true`: aggregator, CameraHead, geometry observations,
and persistent instance memory are causal. Before every optimizer step, all
sequence-shaped tensors are physically sliced to the five-frame prefix, so
frames 210/240 cannot contribute to a loss or checkpoint decision.

At evaluation time the complete seven-frame sequence is replayed. This gives
frames 210/240 the causal history available in deployment, while pose,
pointmap, depth, and instance-rigidity metrics are computed only on those two
held-out frames. Absolute pose alignment and depth/pointmap scale remain fixed
by the original reference frame 90; they are never refit on held-out GT.

## Modes

- `camera_sam_only`: pose-only strong baseline
- `patch_sam_only`: appearance-only geometry strong baseline
- `patch_sam_geometry_tracker_gate`: selected geometry branch
- `decoupled_dual_branch`: proposed V2 method
- `all_token_fusion`: coupled negative architectural control

## Run

```bash
zsh streaming_couping/commands_instance_token_pose_temporal_holdout_v2.txt
```

The run reuses `outputs/streaming_couping_instance_token_pose/cache` and writes
new checkpoints and metrics under
`outputs/streaming_couping_instance_token_pose_temporal_holdout_v2`.

Every evaluation CSV records:

- `evaluation_protocol=causal_temporal_holdout`
- full causal `context_frame_indices`
- supervised `training_frame_indices`
- held-out `evaluated_frame_indices`

Expected pose summaries contain two evaluated frames and one all-pairs/RPE
pair (210 to 240). Geometry summaries contain no frame 90 measurements; the
reference frame is used only to keep the alignment and scale protocol fixed.
