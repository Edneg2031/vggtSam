# External Foundation Models

This directory is intentionally reserved for local clones or symlinks to large
foundation-model codebases. Do not commit those repositories or checkpoints.

Suggested server layout:

```bash
git submodule update --init --recursive
```

SAM3, StreamVGGT, and HorizonStream are pinned as Git submodules. HorizonStream
must be run in its own Python 3.11 environment; do not install its PyTorch 2.8
and `flash-linear-attention` requirements into the existing `3am` environment.

Expected checkpoints in the current server setup:

```text
SAM3:       /home/bod/86Nas/95_data_bak/FoundationModels/sam3/sam3.pt
StreamVGGT: /home/bod/86Nas/95_data_bak/FoundationModels/StreamVGGT/checkpoints.pth
HorizonStream: set `HORIZONSTREAM_CHECKPOINT` to the downloaded HorizonStream.pt
```
