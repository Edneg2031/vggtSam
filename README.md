# SAM3.1-assisted StreamVGGT

The current research path is V7.4: SAM3.1 dynamically discovers persistent
instances and supplies local identity affinity, while StreamVGGT supplies the
geometry Values. Causal multi-instance evidence predicts a bounded SE(3)
residual on top of a frozen camera-only baseline. V4-V7.3 remain as historical
ablations and reproduction paths.

Initialize the two upstream model repositories once:

```bash
git submodule update --init --recursive
```

Run the current V7.4 temporal experiment from the repository root:

```bash
zsh streaming_couping/commands_v74_temporal_scaling.txt
```

The commands use `externals/sam3` and `externals/streamvggt`, while datasets,
checkpoints, caches, and generated evaluations remain local under `data/` and
`outputs/`.

See [the package README](streaming_couping/readme.md) for historical commands
and output layouts. The single current method description is
[SAM3.1 + StreamVGGT V7.4](streaming_couping/docs/current_sam31_streamvggt_v74_method.md).
