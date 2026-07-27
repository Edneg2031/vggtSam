# StreamVGGT + SAM3 instance refinement

This repository contains one project package, `streaming_couping`, with two
retained methods:

- V4 coverage-first: the strongest two-sequence pose result and the main line
  for future false-association work.
- V5 adaptive-best: the conservative pose/pointmap method that falls back to
  the raw pointmap when geometric ray support is insufficient.

Initialize the two upstream model repositories once:

```bash
git submodule update --init --recursive
```

Run either retained method from the repository root:

```bash
zsh streaming_couping/commands_v4_coverage_first.txt
zsh streaming_couping/commands_v5_adaptive_best.txt
```

The commands use `externals/sam3` and `externals/streamvggt`, while datasets,
checkpoints, caches, and generated evaluations remain local under `data/` and
`outputs/`.

See [the package README](streaming_couping/readme.md) for the output layout and
[the method record](streaming_couping/docs/final_joint_pointcloud_pose_method.md)
for the retained architecture and metrics.
