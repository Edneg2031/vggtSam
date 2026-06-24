# External Foundation Models

This directory is intentionally reserved for local clones or symlinks to large
foundation-model codebases. Do not commit those repositories or checkpoints.

Suggested server layout:

```bash
git submodule update --init --recursive
```

The scripts also accept explicit `--vggt-repo` and `--sam3-repo` paths, so the
external repos can stay outside this project if that is cleaner on the server.

Expected checkpoints in the current server setup:

```text
SAM3:       /home/bod/86Nas/95_data_bak/FoundationModels/sam3/sam3.pt
StreamVGGT: /home/bod/86Nas/95_data_bak/FoundationModels/StreamVGGT/checkpoints.pth
```
