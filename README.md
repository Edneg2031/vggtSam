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
