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

After the frozen cache is ready, run the offline semantic-map evaluation:

```bash
zsh streaming_couping/commands_semantic_mapping.txt
```

The commands use `externals/sam3` and `externals/streamvggt`, while datasets,
checkpoints, caches, and generated evaluations remain local under `data/` and
`outputs/`.

See the [package README](streaming_couping/readme.md) for outputs, the
[system pipeline](streaming_couping/docs/system_pipeline.md) for the current
data flow, and the [experiment route](streaming_couping/docs/experiment_route.md)
for the next semantic-map evaluations and claim boundary.
