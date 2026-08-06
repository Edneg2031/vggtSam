# SAM3.1-assisted StreamVGGT

The current experiments separate SAM masks/identity, SAM local descriptors,
StreamVGGT geometry, learned pose adapters, and fixed geometric solvers. The
concise protocol, positive evidence, failed methods, and claim boundaries are
recorded in the
[weekly experiment summary](streaming_couping/docs/weekly_experiment_summary_2026-08-03_to_2026-08-06.md).

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

See [the package README](streaming_couping/readme.md) for the code entry point.
The weekly summary is the only retained research-results document.
