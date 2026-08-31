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

Run the semantic-map pipeline from the repository root:

```bash
zsh streaming_couping/commands_run_semantic_map.txt \
  --frames /path/to/rgb_frames \
  --prompts bed wardrobe \
  --output-dir outputs/semantic_map
```

You can also edit the configuration block at the top of
`streaming_couping/commands_run_semantic_map.txt` and run it without arguments.
`FRAME_COUNT=0` uses all selected RGB inputs; set `FRAME_START` and
`FRAME_STRIDE` for a deterministic subsequence. By default, the command uses
the processed ScanNet++ manifest and the same `00a231a370` frame sequence as the
previous experiments. `CACHE_PATH` is used only when `INPUT_MODE="cache"`; RGB
mode does not automatically reuse a stale cache.

The commands use `externals/sam3` and `externals/streamvggt`, while datasets,
checkpoints, caches, and generated evaluations remain local under `data/` and
`outputs/`.

See the [package README](streaming_couping/readme.md) for configuration and outputs, the
[system pipeline](streaming_couping/docs/system_pipeline.md) for the current
data flow, and the [experiment route](streaming_couping/docs/experiment_route.md)
for the next semantic-map evaluations and claim boundary.
