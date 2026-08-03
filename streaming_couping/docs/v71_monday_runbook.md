# V7.1 Monday validation runbook

## Goal

V7.1 tests whether instance content provides a causal pose improvement beyond a
frozen camera-only model. It separates additional model capacity, active-gate
coverage, appearance, geometry, global decoupled attention and local geometry.

## Recommended first run

Choose three free physical GPUs and run the smoke plus formal experiment:

```bash
V71_STREAM_GPU0=1 \
V71_STREAM_GPU1=2 \
V71_SAM_GPU=3 \
zsh streaming_couping/commands_v71_monday.txt
```

For a stricter standalone audit before the Monday pipeline:

```bash
zsh streaming_couping/commands_v71_preflight.txt
```

It checks cache metadata/shapes/finite values, runs the causal tensor tests and
executes every architecture for two optimizer steps.

The smoke uses two optimizer steps for the camera stage and every residual
architecture. Formal training starts only after smoke produces all expected
artifacts.

## Resume and restart

The formal command resumes matching checkpoints by default:

```bash
zsh streaming_couping/commands_v71_instance_causality.txt
```

Each checkpoint stores a signature over the seed, training frames, step counts
and fusion configuration. A mismatched checkpoint is rejected instead of being
silently reused. To intentionally retrain and overwrite all checkpoints:

```bash
V71_FRESH=1 zsh streaming_couping/commands_v71_instance_causality.txt
```

Checkpoint writes use a temporary file followed by an atomic rename. A killed
job therefore keeps the last complete checkpoint.

To rerun only one content branch while retaining the camera controls required
for labeling, call the Python entry directly. The controls are inserted
automatically:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. \
/home/huawei/miniconda3/envs/3am/bin/python \
  -m streaming_couping.scripts.run_v71_instance_causality \
  --config streaming_couping/configs/v71_instance_causality.yaml \
  --device cuda:0 \
  --architectures local32_decoupled \
  --resume
```

This produces a partial comparison CSV, so use the full formal command before
running `analyze_v71_results.py` or making a final claim.

## Expected formal artifacts

```text
outputs/streaming_couping_v71_instance_causality/
├── frozen_l0.pt
├── camera_extra_all.pt
├── camera_extra_common_gate.pt
├── gate_only.pt
├── appearance_pool.pt
├── geometry_pool.pt
├── decoupled_global.pt
├── local32_decoupled.pt
├── v71_instance_causality.csv
├── v71_frame_diagnostics.csv
├── v71_result_summary.md
└── run_metadata.json
```

Send `v71_instance_causality.csv` first. If aggregate behavior is unclear, also
send the relevant rows from `v71_frame_diagnostics.csv`.

## Decision order

1. Check `causal_instance_pass`. A value of one is intentionally strict.
2. Compare content branches with both `camera_extra_all` and
   `camera_extra_common_gate`.
3. Check future and cross-clip gains; residual-train fitting is not evidence of
   generalization.
4. Check wrong-geometry and shuffle-time damage. A branch that improves without
   reacting to input perturbations is probably exploiting capacity or gates.
5. Implement SAM3.1 mask-local FPN descriptors only if local32 consistently
   improves over global decoupling.

## Optional multi-seed validation

Do this only after one seed is promising:

```bash
V71_SEEDS="0 1 2" zsh streaming_couping/commands_v71_multiseed.txt
```

The compact stability result is:

```text
outputs/streaming_couping_v71_multiseed/v71_seed_aggregate.csv
```

## Common failures

- Missing V7 cache: the smoke/formal command automatically invokes the V7 data
  stage. If the cache is absent, it will run SAM3.1 and StreamVGGT first.
- Checkpoint signature mismatch: use the same command/config, or set
  `V71_FRESH=1` when a deliberate configuration change should retrain models.
- No usable residual-training frame: inspect the cache identity/tracking fields;
  smoke reports this before the 1200-step formal run.
- CUDA out of memory during the cache stage: select different physical cards;
  residual training itself uses logical `cuda:0` only.
