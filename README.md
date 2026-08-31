# SAM3.1-assisted StreamVGGT

The repository currently retains the V0 SAM3.1 × StreamVGGT dynamic-instance
geometry baseline. Its method, measured effect, rejected experiments, and
claim boundaries are recorded in
[the current status](streaming_couping/docs/current_status.md).

Initialize the two upstream model repositories once:

```bash
git submodule update --init --recursive
```

Run the retained V0 baseline from the repository root:

```bash
zsh streaming_couping/commands_v0_baseline.txt
```

The commands use `externals/sam3` and `externals/streamvggt`, while datasets,
checkpoints, caches, and generated evaluations remain local under `data/` and
`outputs/`.

See [the package README](streaming_couping/readme.md) for the code entry point.
