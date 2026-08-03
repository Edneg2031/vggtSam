# V7.2 true SAM3.1 local-token ablation

## 1. What V7.2 answers

V7.1 established the correct causal protocol: train an early-frame camera-only
model, freeze it, then ask whether an instance residual adds value on later
frames. V7.2 changes only the instance evidence representation.

The old `local32` path is **not** a SAM local descriptor path. It samples
StreamVGGT UVD/world-point geometry inside a tracked mask. V7.2 additionally
caches a set of SAM3.1 `detector_fpn2` descriptors inside every tracked instance
mask. Therefore V7.2 can directly test the claim that retaining a token set is
better than compressing an instance to one pooled vector.

V7.2 is not intended to make a result look better by adding capacity. Its CSV
contains two camera-only capacity controls trained on exactly the same Stage-B
frames as all local-token branches.

## 2. Frozen cache extension

For every clip, frame and configured instance, the additive cache fields are:

```text
sam_local_features  [S, K_instance, P, C]  float16 by default
sam_local_uv        [S, K_instance, P, 2]  float32 in [-1, 1]
sam_local_valid     [S, K_instance, P]     bool
```

The feature grid is the same resized `detector_fpn2` map used by the retained
global mean/std descriptor. The mask is resized with nearest-neighbor sampling.
The sampler is deterministic:

1. choose the mask point nearest the mask centroid;
2. repeatedly choose the UV point farthest from the already selected set;
3. pad to fixed `P`, marking padding invalid.

This makes prefixes meaningful: the first 8 tokens of a 32-token cache are the
same K=8 farthest-point set used in the token-count sweep.

`v72_local_token_data.yaml` points at the existing V7 cache directory. If the
geometry/tracking payload is compatible, enabling local tokens loads only the
SAM3.1 image backbone and augments the `.pt` files. StreamVGGT, video tracking,
GT alignment and geometry construction are not rerun. A message containing
`only SAM appearance cache fields will be augmented` confirms this path.

## 3. Architecture ladder

Every branch predicts a bounded SE(3) residual over one frozen L0. Its final
linear layer is zero initialized, so step zero exactly reproduces L0.

| Architecture | Identity evidence | Value evidence | Purpose |
|---|---|---|---|
| `sam_local_pool` | current SAM token set | learned current-token pool | light control; no reference matching |
| `sam_local_match` | current-to-reference SAM attention | local match residual | tests true local appearance matching |
| `geometry_local_match` | common gate | StreamVGGT local geometry residual | isolates local geometry without SAM content |
| `dual_local_match` | SAM local match | SAM + geometry local residuals | independent local modalities, late merge |
| `sam_key_global_geometry` | SAM local match | persistent global geometry | identity-Key / geometry-Value hypothesis |
| `dual_key_global_geometry` | SAM local match | global + local geometry | heaviest hierarchical path |

The two automatic controls are:

- `camera_extra_all`: extra camera capacity on every non-reference frame;
- `camera_extra_common_gate`: the same extra camera capacity only where the
  shared instance gate is active.

The controls are trained and evaluated inside V7.2, so the decision does not
depend on stale V7.1 checkpoints.

## 4. Time protocol

V7.2 imports the locked V7.1 time partition:

```text
90                     reference/context, never supervised
105–255                train L0 camera-only
270–345                freeze L0; train one residual
360–405                development selection
50–80 clip             independent validation selection
420–525                strict report-only future
492–589 clip           strict report-only cross clip
```

Training tensors are sliced at the last allowed frame before forward, so future
camera, mask, SAM and geometry inputs do not enter the training graph.

## 5. Monday command order

Select three free physical GPUs. StreamVGGT uses the first two only if an old
geometry cache is absent; SAM3.1 uses the third while local descriptors are
added. Training subsequently uses one card.

```bash
V72_STREAM_GPU0=1 \
V72_STREAM_GPU1=2 \
V72_SAM_GPU=3 \
V72_TRAIN_GPU=1 \
zsh streaming_couping/commands_v72_monday.txt
```

The stages are cache augmentation/audit, two-step smoke, formal resumable
sweep, and report generation. If the V7.1 formal CSV is present, the report
compares V7.1 and V7.2; otherwise it creates a valid V7.2-only bundle. To add
the global V7.1 comparison first, run:

```bash
zsh streaming_couping/commands_v71_instance_causality.txt
```

Individual commands are:

```bash
zsh streaming_couping/commands_v72_cache.txt
zsh streaming_couping/commands_v72_smoke.txt
zsh streaming_couping/commands_v72_local_token_ablation.txt
zsh streaming_couping/commands_v72_report.txt
```

Formal training resumes matching atomic checkpoints by default. To deliberately
ignore them:

```bash
V72_FRESH=1 zsh streaming_couping/commands_v72_local_token_ablation.txt
```

To debug only one or two structures without changing YAML:

```bash
CUDA_VISIBLE_DEVICES=1 \
/home/huawei/miniconda3/envs/3am/bin/python \
  -m streaming_couping.scripts.run_v72_local_token_ablation \
  --config streaming_couping/configs/v72_local_token_ablation.yaml \
  --device cuda:0 \
  --architectures sam_local_match,sam_key_global_geometry \
  --token-counts 8,32 \
  --base-steps 20 \
  --residual-steps 20
```

## 6. Outputs

```text
outputs/streaming_couping_v72_local_token_ablation/
├── cache_audit/
│   ├── cache_audit.csv
│   ├── cache_tensor_audit.csv
│   ├── cache_frame_audit.csv
│   └── cache_audit.json
├── frozen_l0.pt
├── camera_extra_all.pt
├── camera_extra_common_gate.pt
├── *_k08.pt / *_k16.pt / *_k32.pt
├── v72_local_token_ablation.csv
├── v72_frame_diagnostics.csv
└── v72_run_metadata.json
```

Copy `v72_local_token_ablation.csv` first. It has raw + frozen L0 + two camera
controls + 6 architectures × 3 token counts, i.e. 22 rows.

The generated report directory contains compact paper CSVs, LaTeX tables,
figures, copied source CSVs and SHA-256 checksums. The `.tar.gz` beside it can
be downloaded as one reproducibility artifact.

Local-token images are deliberately not part of the default formal run. Render
them once after cache creation:

```bash
zsh streaming_couping/commands_v72_local_token_visualization_once.txt
```

Open `outputs/streaming_couping_v72_local_token_visualization/index.html` to
inspect every cached mask and sampled point. The accompanying CSV reports mask
pixels and token counts per instance/frame.

After formal checkpoints exist, benchmark the frozen L0, both camera controls
and the development-best local model:

```bash
zsh streaming_couping/commands_v72_inference_speed.txt
```

The resulting `v72_inference_speed.csv` reports sequence/frame latency, FPS,
parameters and peak allocation. Its scope is explicitly pose refinement from
cached features; it excludes SAM3.1, StreamVGGT and disk-cache construction.
Set `V72_BENCHMARK_ARCHITECTURES=all` only when all 21 trained models need to be
profiled.

For local-attention branches, `v72_frame_diagnostics.csv` also reports
`sam_attention_entropy_normalized` and `sam_attention_max_probability`. A
single Key would force entropy 0 and maximum probability 1; the V7.2 token set
should instead show nontrivial values that change by frame. These statistics do
not prove better pose by themselves, but they verify that the intended local
attention has not mathematically collapsed to broadcasting one Value.

## 7. Decision fields

Read the CSV in this order:

1. `development_score`: average development/validation loss ratio to L0;
2. `future_*` and `cross_*`: report-only generalization;
3. `beats_camera_controls_report_only`: better than both capacity controls on
   both report-only splits;
4. `instance_off_exact`: exact L0 fallback on development, validation, future
   and cross;
5. `causal_local_pass`: all strict conditions simultaneously hold.

Perturbations have different diagnostic meanings:

- `local_off`: local evidence removal should return to L0;
- `wrong_local_identity`: roll instance slots while preserving gate coverage;
- `shuffle_local_time`: reverse local appearance time without changing camera;
- `wrong_geometry`: roll global and local geometry slots;
- `instance_off`: remove the whole instance path and require exact L0.

An apparent gain with no sensitivity to relevant content perturbations is not
evidence that the model used that content.

## 8. Multi-seed validation

Only run this after a single seed produces a plausible report-only gain:

```bash
V72_SEEDS="0 1 2" zsh streaming_couping/commands_v72_multiseed.txt
```

The aggregate reports mean/std, number of development wins, number of camera
control wins and number of strict causal passes per architecture/K pair.

## 9. Failure diagnosis

- `missing sam_local_features`: run `commands_v72_cache.txt`.
- cache audit finite/shape error: do not train; inspect
  `cache_tensor_audit.csv` and `cache_frame_audit.csv`.
- message says it is caching frozen geometry/tracking when an old V7 cache was
  expected: metadata changed or the V7 cache is incomplete. Inspect
  `cache_audit.csv` before accepting a costly rebuild.
- no active residual-training frame: local tokens and the reference identity
  do not overlap the shared gate on `270–345`; inspect per-frame cache audit.
- checkpoint mismatch: config, K, seed or step count changed. Use a new output
  directory or deliberately set `V72_FRESH=1`.
- missing V7.1 CSV: the report remains valid but is marked V7.2-only; rerun the
  report command after V7.1 if a combined comparison is needed.
