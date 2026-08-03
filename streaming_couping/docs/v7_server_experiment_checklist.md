# V7/V7.2 server experiment checklist

## Before occupying GPUs

```bash
git status --short
/home/huawei/miniconda3/envs/3am/bin/python -c \
  'import torch; print(torch.__version__, torch.cuda.device_count())'
zsh streaming_couping/commands_v71_preflight.txt
```

The preflight must finish cache audit, tensor tests and the two-step V7.1 run.
For V7.2, run `commands_v72_cache.txt` before `commands_v72_smoke.txt`.

## Files to copy back after a run

Minimum:

```text
v71_instance_causality.csv
v71_frame_diagnostics.csv
v72_local_token_ablation.csv
v72_frame_diagnostics.csv
cache_audit/cache_audit.csv
cache_audit/cache_frame_audit.csv
v7_result_summary.md
```

For exact reproduction also retain all run metadata, checkpoint files and
`bundle_manifest.json`. The report archive already contains source CSV copies
and generated tables/figures, but not large model checkpoints.

## Stop conditions

Stop a formal run and debug rather than consuming all steps if any is true:

- cache audit has `passed=0`;
- reference pose changes at zero initialization;
- an instance-content branch is active when `instance_off` is applied;
- Stage-B has zero active frames;
- loss becomes non-finite;
- a cache intended for augmentation unexpectedly starts StreamVGGT.

## Result interpretation

Do not select on future/cross and then call the same numbers generalization.
Select architecture/K only with development plus validation. Future/cross are
used once to report whether the preselected choice survives. Multi-seed is
justified only after the one-seed report-only result is promising.

