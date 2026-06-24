# SAM3 and StreamVGGT Fusion

The current target is a latent fusion model rather than a hard-coded tracking
pipeline. SAM3 should provide open-vocabulary object/mask tokens, while
VGGT/StreamVGGT should provide geometry tokens, camera tokens, and later point
or depth predictions. The fusion module lets semantic tokens query geometric
context through cross-attention, then predicts semantic logits, 3D points, and
cross-frame correspondence embeddings.

## Current Implementation

- `externals/` contains SAM3 and StreamVGGT as git submodules.
- `scripts/inspect_backbone_outputs.py` runs a small subset of processed
  ScanNet++ frames and dumps the real SAM3/VGGT output structures.
- `src/vggtsam/models/fusion.py` implements the model core using a clean token
  interface. It does not assume a specific internal SAM3 or VGGT layer yet.
- `scripts/train_object_fusion.py` trains the first object-level version using
  ScanNet++ projected instance masks as oracle cross-frame supervision and
  frozen StreamVGGT geometry outputs as geometric context.

## Why Inspect First

The earlier quick test used VGGT as a global image feature. For the actual idea,
we need denser tokens:

- geometry tokens: patch-level or memory-level VGGT/StreamVGGT tokens;
- camera tokens: pose/camera tokens if exposed by the backbone;
- semantic tokens: SAM3 object or mask tokens if exposed, otherwise mask-pooled
  image features as a temporary adapter.

The layer choice should be made from observed outputs rather than guessed from
names. Run the inspect script on the server and share the JSON/text summary.

## Server Commands

```bash
git submodule update --init --recursive

PYTHONPATH=src python scripts/inspect_backbone_outputs.py \
  --config configs/fusion_debug.yaml
```

The equivalent explicit command is:

```bash
PYTHONPATH=src python scripts/inspect_backbone_outputs.py \
  --manifest data/processed/scannetpp_2d/manifest.json \
  --scene-id 0b031f3119 \
  --num-frames 4 \
  --sam3-checkpoint /home/bod/86Nas/95_data_bak/FoundationModels/sam3/sam3.pt \
  --sam3-repo externals/sam3 \
  --sam3-prompt chair \
  --geometry-backbone streamvggt \
  --vggt-repo externals/streamvggt \
  --vggt-checkpoint /home/bod/86Nas/95_data_bak/FoundationModels/StreamVGGT/checkpoints.pth \
  --device cuda \
  --output-json outputs/debug/backbone_outputs.json
```

You can omit either SAM3 or VGGT arguments to inspect only one side.

## Training V0

The first training version follows the intended supervision path:

```text
ScanNet++ 3D instance ids
  -> rasterized cross-frame-consistent instance_masks
  -> object queries and object correspondence labels

StreamVGGT frozen output
  -> point/depth/conf/camera geometry context

Object fusion model
  -> semantic logits, 3D object centroid, cross-frame match embeddings
```

Run a tiny debug job:

```bash
PYTHONPATH=src python scripts/train_object_fusion.py \
  --config configs/object_fusion_train.yaml \
  --iterations 20 \
  --device cuda
```

This version intentionally does not depend on SAM3 returning masks during
training. SAM3 is the later open-vocabulary query source; ScanNet++ masks are
the stable training target.
