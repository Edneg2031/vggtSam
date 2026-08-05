# SAM3.1-assisted StreamVGGT

The latest implemented research path is V8.0: GT world pointmaps explicitly
supervise SAM3.1/StreamVGGT correspondence, then the matcher is frozen before
an evidence-only bounded pose residual is trained. Its three-fold run did not
pass the SAM-token causality criteria; V8.1 dual Values also remained
report-only. No existing command is currently designated as the next new
experiment. Consult the validation ledger before extending the pipeline.

Initialize the two upstream model repositories once:

```bash
git submodule update --init --recursive
```

Reproduce the completed V8 matcher-first temporal experiment from the repository root:

```bash
zsh streaming_couping/commands_v80_supervised_correspondence.txt
```

The commands use `externals/sam3` and `externals/streamvggt`, while datasets,
checkpoints, caches, and generated evaluations remain local under `data/` and
`outputs/`.

See [the package README](streaming_couping/readme.md) for historical commands
and output layouts. The single current method description is
[SAM3.1 + StreamVGGT V8](streaming_couping/docs/current_sam31_streamvggt_v80_method.md).
Before proposing another ablation, consult the
[experiment validation ledger](streaming_couping/docs/sam31_streamvggt_experiment_ledger.md),
which records completed controls, measured outcomes, rejected claims, and the
admission rules for genuinely new experiments.
