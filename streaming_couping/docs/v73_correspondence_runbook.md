# V7.3: SAM-weighted StreamVGGT geometry correspondence

## 1. Question

V7.2 showed that mask-local StreamVGGT geometry can help camera refinement,
but it did not show an independent, generalizing contribution from SAM3.1
features. V7.3 tests a narrower claim:

> Can SAM3.1 local identity descriptors improve **correspondence weights** used
> to transport StreamVGGT local geometry from a reference frame, without using
> SAM features themselves as pose-correction Values?

This is different from cross-attending to one pooled persistent token. The Key
set has K local tokens, so attention does not collapse to a length-one softmax.
The pose head receives matched geometric residuals, not a broadcast semantic
vector.

## 2. Data flow

For each persistent instance and current/reference frame pair:

1. take K cached StreamVGGT local tokens containing normalized world XYZ, UV,
   log depth and confidence;
2. take K cached SAM3.1 mask-local FPN descriptors plus their UV positions;
3. interpolate SAM descriptors onto StreamVGGT geometry sample positions using
   fixed UV proximity weights;
4. form current-to-reference correspondence logits from geometry, SAM, both,
   or neither;
5. use the resulting probability matrix to transport **reference StreamVGGT
   geometry Values**;
6. encode current geometry, matched reference geometry, their difference and
   product;
7. reliability-pool valid instances and fuse the evidence with the exact
   frozen V7.1 camera L0 feature;
8. predict a bounded, zero-initialized SE(3) residual. The reference pose is
   never changed.

UV is only used for descriptor interpolation. It is not concatenated into the
SAM identity descriptor, preventing absolute image position from becoming a
shortcut for the claimed SAM affinity.

The existing persistent-instance tracker still determines the coarse instance
slot. V7.3 matches local points only inside the same current/reference slot; it
does not claim that SAM independently solves cross-object data association.
`wrong_sam_identity` swaps descriptor sets across slots specifically to test
whether that assumed identity alignment matters.

## 3. Light-to-heavy ladder

| Architecture | Correspondence logits | Transported Value |
|---|---|---|
| `uniform_transport` | all valid reference points equal | StreamVGGT geometry |
| `geometry_transport` | learned local geometry similarity | StreamVGGT geometry |
| `sam_transport` | SAM3.1 local identity similarity | StreamVGGT geometry |
| `sam_geometry_transport` | geometry + SAM affinity | StreamVGGT geometry |

Only K=8 and K=16 are run. K=32 is intentionally deferred: increasing token
count before proving a SAM-specific effect would mix mechanism and capacity.

The one-table output also reloads, without retraining:

- exact V7.1 `frozen_l0`;
- V7.2 `camera_extra_all` and `camera_extra_common_gate`;
- V7.2 `geometry_local_match_k08/k16` as the previous strong geometry control.

All retained checkpoint signatures must refer to the same exact V7.1 L0.
V7.3 does not depend on V6 checkpoints and does not run SAM3.1 or StreamVGGT.

## 4. Locked causal protocol

The split remains identical to V7.1/V7.2:

```text
90                     reference/context only
105–255                Stage-A camera training (already frozen in V7.1)
270–345                train the V7.3 residual
360–405                development
50–80 clip             independent validation
420–525                report-only temporal future
492–589 clip           report-only second clip from the same scene
```

The second clip is called `cross` for compatibility with existing CSVs. It is
not evidence of cross-scene generalization.

At every logging interval, Stage B saves the lowest-loss logged state and
restores it after training. Future tensors are excluded by slicing the training
sequence at frame 345 before the forward pass.

## 5. Perturbations

Every trained V7.3 branch is evaluated with:

| Variant | Meaning |
|---|---|
| `instance_off` | disable the complete instance path; must return exactly to L0 |
| `geometry_off` | remove transported geometry Values |
| `sam_off` | remove SAM descriptors; combined branch reduces to geometry correspondence |
| `uniform_sam` | keep SAM availability but remove SAM affinity information |
| `wrong_sam_identity` | roll non-reference SAM instance slots |
| `shuffle_sam_time` | reverse non-reference SAM descriptor time |
| `wrong_local_geometry` | roll non-reference local geometry instance slots |

`causal_sam_pass=1` is deliberately strict. A SAM branch must:

1. beat same-K `geometry_transport` on development, validation, future and
   cross;
2. beat both camera capacity controls on future and cross;
3. return exactly to L0 under `instance_off`;
4. worsen under all four SAM perturbations on all four causal splits.

The CSV also reports `sam_perturbation_hurt_count/total`, so a near miss remains
interpretable rather than being reduced to one Boolean.

## 6. Server commands

V7.3 requires the corrected formal V7.2 checkpoints. It reuses the existing
SAM3.1 local-token caches and only needs one GPU for the small fusion models.
The smoke command uses pytest when it is installed in the selected
`STREAMING_COUPING_PYTHON`; otherwise it automatically runs equivalent core
Torch checks without pytest. Installing pytest into an unrelated active conda
environment is therefore unnecessary.

Run the full smoke and formal sweep:

```bash
V73_GPU=1 zsh streaming_couping/commands_v73_monday.txt
```

Or run stages separately:

```bash
V73_GPU=1 zsh streaming_couping/commands_v73_smoke.txt
V73_GPU=1 zsh streaming_couping/commands_v73_correspondence_ablation.txt
zsh streaming_couping/commands_v73_report.txt
```

Formal checkpoints resume by default. To deliberately retrain them:

```bash
V73_FRESH=1 V73_GPU=1 \
  zsh streaming_couping/commands_v73_correspondence_ablation.txt
```

## 7. Outputs to copy back

The primary result is one easy-to-copy table:

```text
outputs/streaming_couping_v73_correspondence/
├── v73_correspondence_ablation.csv   # copy this first
├── v73_frame_diagnostics.csv
├── v73_run_metadata.json
├── report/
│   ├── v73_decision_table.csv
│   └── v73_result_summary.md
├── frozen_l0.pt
└── *_k08.pt / *_k16.pt
```

The formal main table has 14 rows: raw, frozen L0, two retained camera
controls, two retained V7.2 geometry controls, and four V7.3 mechanisms at two
token counts.

The frame table reports usable instances, SAM-used instances, normalized
transport entropy, maximum correspondence probability, SAM-induced affinity
change, activity and per-frame pose error. Copy it only when the primary table
shows an interesting gain or an unexplained split failure.

The formal CSV remains the complete single source of truth to copy back. The
report command additionally creates a short decision table for quick reading;
it does not select on future/cross or alter any metric.

Interpret results in this order:

1. compare `geometry_transport` with the retained V7.2 geometry baseline;
2. compare `sam_transport` and `sam_geometry_transport` with same-K
   `geometry_transport`;
3. inspect future/cross instead of training loss;
4. inspect perturbation sensitivity;
5. only then read `causal_sam_pass`.

A lower loss without sensitivity to SAM removal, identity corruption and time
shuffle is a capacity result, not evidence that SAM correspondence helped the
camera.

`beats_retained_v72_geometry_all_splits` is reported separately from the
strict SAM-causality flag. This distinguishes “SAM changed the new
correspondence mechanism” from the stronger engineering question “the new
mechanism also beats the best retained V7.2 geometry implementation.”

## 8. Multi-seed follow-up

Do not spend three runs on a clearly failed single-seed mechanism. If seed 0
passes or nearly passes the declared controls, run:

```bash
V73_GPU=1 V73_SEEDS="0 1 2" \
  zsh streaming_couping/commands_v73_multiseed.txt
```

Each seed keeps the exact same frozen V7.1 L0 and retained V7.2 controls; only
the new V7.3 residual initialization/training seed changes. The aggregate is:

```text
outputs/streaming_couping_v73_multiseed/v73_seed_aggregate.csv
```

It reports mean/std losses, selection/control win counts, perturbation damage
fraction and strict causal-pass count per architecture/K. A single lucky seed
is not treated as stable evidence.
