# vggtSam

Geometry-aware open-vocabulary object tracking experiments built around
ScanNet++, SAM3 intermediate features, and StreamVGGT latent geometry features.

The current mainline is the latent fusion implementation described in:

```text
docs/latent_fusion_training_flow.md
```

## ScanNet++ 2D Labels

Generate projected semantic and instance masks from ScanNet++ 3D annotations:

```bash
PYTHONPATH=src python scripts/prepare_scannetpp_2d.py \
  --data-root /home/bod/184Nas/open_source/scannet_pp/data \
  --output-root data/processed/scannetpp_2d \
  --scene-ids 0a5c013435 \
  --max-frames 20 \
  --frame-step 5 \
  --save-visualizations
```

Output structure:

```text
data/processed/scannetpp_2d/
  manifest.json
  <scene_id>/
    scene_manifest.json
    semantic_masks/
    instance_masks/
    raster/
    visualizations/
```

More details:

```text
docs/scannetpp_preprocessing.md
```

## Latent Fusion

Current model direction:

```text
SAM3 detector FPN-2 + pooled text feature
  -> semantic query tokens

StreamVGGT aggregator patch tokens + camera tokens
  -> geometry key/value context

cross-attention fusion
  -> semantic logits
  -> pointmap prediction
  -> cross-frame matching embeddings
```

Inspect feature shapes first:

```bash
PYTHONPATH=src python scripts/inspect_latent_fusion_features.py \
  --config configs/latent_fusion_train.yaml \
  --device cuda
```

Run a small training job:

```bash
PYTHONPATH=src python scripts/train_latent_fusion.py \
  --config configs/latent_fusion_train.yaml \
  --iterations 20 \
  --device cuda
```

Plot training curves:

```bash
PYTHONPATH=src python scripts/plot_training_curves.py \
  --metrics outputs/latent_fusion_debug/training_history.csv \
  --output outputs/latent_fusion_debug/training_curves.png
```

## Main Files

```text
configs/latent_fusion_train.yaml
scripts/train_latent_fusion.py
scripts/inspect_latent_fusion_features.py
src/vggtsam/adapters/sam3_intermediate.py
src/vggtsam/adapters/streamvggt_latent.py
src/vggtsam/models/latent_fusion.py
src/vggtsam/models/fusion.py
src/vggtsam/training/latent_fusion.py
```
