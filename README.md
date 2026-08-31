# SAM3.1-assisted StreamVGGT

The active implementation is the frozen V0 semantic-mapping pipeline. StreamVGGT
provides the camera trajectory and raw full-history world pointmap; SAM3.1
provides prompted multi-instance masks, persistent identities, and future object
births. No rejected temporal point-prompt loop, historical-depth veto, affine
depth correction, or SAM appearance-token pose path is active.

Measured results, rejected experiments, and claim boundaries are recorded in
[the current status](streaming_couping/docs/current_status.md).

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

For the backend-neutral RGB + text-prompt pipeline:

```bash
zsh streaming_couping/commands_run_semantic_map.txt \
  --frames /path/to/rgb_frames \
  --prompts bed wardrobe \
  --output-dir outputs/semantic_map
```

The commands use `externals/sam3` and `externals/streamvggt`, while datasets,
checkpoints, caches, and generated evaluations remain local under `data/` and
`outputs/`.

See the [package README](streaming_couping/readme.md) for outputs, the
[system pipeline](streaming_couping/docs/system_pipeline.md) for the current
data flow, and the [experiment route](streaming_couping/docs/experiment_route.md)
for the next semantic-map evaluations and claim boundary.
