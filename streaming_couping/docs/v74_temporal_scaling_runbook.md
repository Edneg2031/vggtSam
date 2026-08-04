# V7.4 temporal scaling and SAM causality

## Goal

V7.3 established two facts:

1. every correspondence branch can fit all 26 active frames in the sampled
   long sequence;
2. geometry-only and the parameter-matched SAM-off control fit at least as
   well as the SAM branches.

V7.4 no longer tests capacity. It asks whether SAM correspondence learned from
an earlier prefix improves unseen future frames beyond geometry and a
parameter-matched model trained without SAM logits.

## Locked temporal folds

Frame 90 remains only the camera-coordinate gauge. It is no longer an object
reference frame. V7.4 builds its own 30-frame SAM3.1 and StreamVGGT cache, then
trains a fresh camera-only L0 on frames 105 through 255. The resulting L0, K=8
observations and optimization settings are shared by every residual branch.

## Dynamic instances and causal memory

SAM3.1 receives one `object` text prompt and runs its multiplex tracker only in
the forward direction. It may create a new persistent ID on any frame. The
first eight IDs are assigned permanent logical slots in order of first
appearance; the values `[0..7]` in the data YAML are capacities, not GT object
IDs. No GT instance mask is used to create these slots.

For each slot, the first geometry-supported observation is a birth event. That
frame writes SAM local tokens and StreamVGGT local geometry to instance memory,
but cannot update the pose because no historical Key/Value exists yet. From the
second observation onward, current tokens match only the most recent earlier
observation. Memory is updated after matching, so later detections cannot alter
an already-computed prefix. If no mature instance is visible, output falls back
exactly to frozen L0.

| Fold | Residual training | Future test |
|---|---|---|
| `short` | 270, 285, 300, 315, 330, 345 | 360, 375, 390, 405 |
| `medium` | 270 through 405, step 15 | 420, 435, 450, 465 |
| `long` | 270 through 465, step 15 | 480, 495, 510, 525 |

Each fold starts from that same V7.4-owned L0 and a fresh residual
initialization. Training uses a tensor prefix ending at the last training
frame, so future observations and targets do not enter its forward pass.
Evaluation remains streaming: a future frame may use target-free memory
accumulated from earlier observations.

## Controls

The default command trains:

- `uniform_transport`;
- `geometry_transport`;
- `sam_transport`;
- `sam_geometry_transport`;
- `sam_geometry_train_sam_off`.

The last two have identical modules and parameter counts. The final control is
trained and primarily evaluated with SAM validity disabled, rather than merely
turning SAM off after a normal model has been trained.

## Strict fold decision

Only `sam_geometry_transport` can receive `fold_sam_causal_pass=1`. It must:

1. fit every active training frame to loss at most `1e-4` with at least 99%
   mean training-loss reduction;
2. beat `geometry_transport` by at least 1% on the future test frames;
3. beat `sam_geometry_train_sam_off` by at least 1% on the same future frames;
4. worsen by at least 1% under each of `sam_off`, `uniform_sam`,
   `wrong_sam_identity` and `shuffle_sam_time`;
5. use exactly the same active train/test frame support as both controls;
6. preserve the reference and inactive-frame fallback exactly.

`all_folds_sam_causal_pass=1` requires all three folds to pass. This is a
strict same-scene temporal result, not cross-scene generalization.

## Run

```bash
V74_GPU=1 zsh streaming_couping/commands_v74_temporal_scaling.txt
```

Matching checkpoints resume by default. To rerun them:

```bash
V74_GPU=1 V74_FRESH=1 \
  zsh streaming_couping/commands_v74_temporal_scaling.txt
```

V7.4 has no V7.1, V7.2 or V7.3 output dependency. Its cache, fresh L0,
residual checkpoints and CSV all live under
`outputs/streaming_couping_v74_temporal_scaling`. The default physical GPU
layout is StreamVGGT on 1 and 2, SAM3.1 on 3, followed by training on 1.

To change the budget or seed:

```bash
V74_GPU=1 V74_STEPS=5000 V74_SEED=1 \
  V74_OUTPUT=outputs/streaming_couping_v74_temporal_scaling_seed1 \
  zsh streaming_couping/commands_v74_temporal_scaling.txt
```

Copy only this first:

```text
outputs/streaming_couping_v74_temporal_scaling/v74_temporal_scaling.csv
```

If tracking coverage looks suspicious, also copy:

```text
outputs/streaming_couping_v74_temporal_scaling/v74_dynamic_instance_diagnostics.csv
```

That short table reports newly born SAM IDs, all discovered slots, currently
observed slots, mature slots and geometry-associated slots frame by frame.

The command first checks the locked folds, decision thresholds, 54-column CSV
schema and V7.3 tensor behavior without requiring `pytest`.

Run extra seeds only if seed 0 passes or nearly passes the locked criteria.
