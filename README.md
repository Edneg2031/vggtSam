# vggtSam

This repo is a scratch space for geometry-aware open-vocabulary object tracking.
The first concrete piece is ScanNet++ preprocessing: render 3D mesh semantics and
instances onto selected DSLR frames so later experiments can train on RGB,
2D masks, 3D geometry, and cross-view object identities.

## ScanNet++ 2D Labels

Example:

```bash
PYTHONPATH=src python scripts/prepare_scannetpp_2d.py \
  --data-root /home/bod/184Nas/open_source/scannet_pp/data \
  --output-root data/processed/scannetpp_2d \
  --scene-ids 0a5c013435 \
  --max-frames 20 \
  --frame-step 5 \
  --save-visualizations
```

The output contains one folder per scene with `semantic_masks`,
`instance_masks`, `raster`, `visualizations`, `scene_manifest.json`, plus a
top-level `manifest.json`.

More details are in [docs/scannetpp_preprocessing.md](docs/scannetpp_preprocessing.md).

## Fusion Model Debug

The fusion model is scaffolded around generic SAM3 semantic tokens and
VGGT/StreamVGGT geometry tokens. Before fixing the exact backbone layer, inspect
the real server-side outputs:

```bash
git submodule update --init --recursive

PYTHONPATH=src python scripts/inspect_backbone_outputs.py \
  --config configs/fusion_debug.yaml
```

More details are in [docs/model_fusion.md](docs/model_fusion.md).
