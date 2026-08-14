# SAM3.1-assisted StreamVGGT

The active implementation is the frozen V0 semantic mapping pipeline. StreamVGGT
provides a QK-retrieved camera trajectory and an unmodified full-history world
pointmap. SAM3.1 provides prompted multi-instance masks, persistent identities,
and future object births. No SAM appearance token, learned pose adapter, or
point-cloud refinement is active.

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

See the [package README](streaming_couping/readme.md) for outputs and the
[V0 method](streaming_couping/docs/v0_method.md) for the exact design and claim
boundary.
