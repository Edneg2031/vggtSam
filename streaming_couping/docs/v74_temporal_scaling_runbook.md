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

Frame 90 remains the reference. The exact frozen V7.1 L0, K=8 observations and
optimization settings are shared by every branch.

| Fold | Residual training | Future test |
|---|---|---|
| `short` | 270, 285, 300, 315, 330, 345 | 360, 375, 390, 405 |
| `medium` | 270 through 405, step 15 | 420, 435, 450, 465 |
| `long` | 270 through 465, step 15 | 480, 495, 510, 525 |

Each fold starts from the same L0 and a fresh residual initialization. Training
uses a tensor prefix ending at the last training frame, so future observations
and targets do not enter its forward pass. Evaluation remains streaming: a
future frame may use target-free memory accumulated from earlier observations.

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

The command searches for the exact retained L0 in V7.1, then the provenance-
checked copies in V7.3 and V7.2. An explicit path can be supplied with
`V74_FROZEN_L0=/path/to/frozen_l0.pt`; its V7.1 source signature is still
validated before training.

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

The command first checks the locked folds, decision thresholds, 54-column CSV
schema and V7.3 tensor behavior without requiring `pytest`.

Run extra seeds only if seed 0 passes or nearly passes the locked criteria.
