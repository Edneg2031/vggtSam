# ScanNet++ Preprocessing

The downloaded ScanNet++ DSLR folders do not contain ready-to-use 2D semantic
masks. The official ScanNet++ toolbox describes the intended path: rasterize the
3D mesh onto DSLR or iPhone images to get pixel-to-face mappings, then transfer
3D semantic and instance annotations to 2D images.

For this project we implement a local preprocessing entry point that works on a
selected subset of scenes and frames first, instead of forcing the whole dataset
through a large offline job.

## Expected Input

One scene should look like:

```text
<data_root>/<scene_id>/
  dslr/
    colmap/
      cameras.txt
      images.txt
      points3D.txt
    resized_images/
    resized_anon_masks/
    train_test_lists.json
  scans/
    mesh_aligned_0.05.ply
    mesh_aligned_0.05_semantic.ply
    segments.json
    segments_anno.json
```

`resized_anon_masks` are privacy/valid-pixel masks. They are often all 255 and
should not be treated as semantic masks.

## What Gets Generated

For every selected frame:

- `semantic_masks/<image>.png`: 16-bit semantic IDs. Invalid/unlabeled pixels use
  `65535` by default.
- `instance_masks/<image>.png`: 16-bit object IDs from `segments_anno.json`.
  Background/invalid pixels use `0`.
- `raster/<image>.npz`: optional pixel-to-face and z-buffer data.
- `visualizations/*`: optional semantic and instance overlays for quick sanity
  checks.
- `scene_manifest.json` and top-level `manifest.json`: paths and frame metadata
  for later training code.

## Usage

Process one scene and only a few frames:

```bash
PYTHONPATH=src python scripts/prepare_scannetpp_2d.py \
  --data-root /home/bod/184Nas/open_source/scannet_pp/data \
  --output-root data/processed/scannetpp_2d \
  --scene-ids 0a5c013435 \
  --max-frames 20 \
  --frame-step 5 \
  --save-visualizations
```

Process scenes from a text file:

```bash
PYTHONPATH=src python scripts/prepare_scannetpp_2d.py \
  --data-root /home/bod/184Nas/open_source/scannet_pp/data \
  --output-root data/processed/scannetpp_2d \
  --scene-list data/splits/debug_scenes.txt \
  --limit-scenes 3
```

Use the YAML config:

```bash
PYTHONPATH=src python scripts/prepare_scannetpp_2d.py \
  --config configs/scannetpp_2d.yaml \
  --scene-ids 0a5c013435 \
  --max-frames 10
```

Check paths and selected frames before rasterization:

```bash
PYTHONPATH=src python scripts/prepare_scannetpp_2d.py \
  --config configs/scannetpp_2d.yaml \
  --scene-ids 0a5c013435 \
  --max-frames 5 \
  --dry-run
```

## Notes

- The DSLR COLMAP camera is usually `OPENCV_FISHEYE`. The code projects mesh
  vertices with the COLMAP world-to-camera pose and camera distortion model.
- If actual image dimensions differ from COLMAP dimensions, intrinsics are scaled
  to the loaded resized image size.
- CPU rasterization is implemented locally with a z-buffer. Installing `numba`
  is strongly recommended on the server for speed:

```bash
pip install numba
```

- Start with `--max-frames` and `--frame-step` while debugging; full scenes can
  be much slower.

## References

- ScanNet++ toolbox README, semantics section:
  https://github.com/scannetpp/scannetpp/#semantics
- ScanNet++ dataset documentation, file structure and semantic mesh notes:
  https://scannetpp.mlsg.cit.tum.de/scannetpp/documentation
