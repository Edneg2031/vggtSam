# SAM3.1-assisted StreamVGGT

The active implementation is the V0 causal tracking baseline: SAM3.1 provides
prompted multi-instance masks, persistent identities, and future object births;
StreamVGGT provides geometry guidance and the unmodified selected camera pose.
No SAM appearance token or learned pose adapter is active.

Completed V4–V9.8 experiments, retired E0/E1/G0 candidates, positive evidence,
negative evidence, and the next instance-aware pose design are preserved in the
[weekly experiment summary](streaming_couping/docs/weekly_experiment_summary_2026-08-03_to_2026-08-06.md).

Initialize the two upstream model repositories once:

```bash
git submodule update --init --recursive
```

Run V0 from the repository root:

```bash
zsh streaming_couping/commands_v0_baseline.txt
```

The commands use `externals/sam3` and `externals/streamvggt`, while datasets,
checkpoints, caches, and generated evaluations remain local under `data/` and
`outputs/`.

See [the package README](streaming_couping/readme.md) for the exact claim boundary.
The weekly summary is the only retained research-results/design document.
