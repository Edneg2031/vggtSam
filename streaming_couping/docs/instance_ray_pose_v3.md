# Instance-guided pointmap-to-ray pose refinement V3

## 1. Why this is the next optimisation target

The causal temporal-holdout experiment already isolates the current failure:

- the selected patch geometry branch reduces held-out full-scene pointmap mean
  error from `0.15258 m` to `0.06053 m`;
- the decoupled dual branch reduces it to `0.06358 m`;
- the learned camera branch reduces the held-out `210->240` rotation error from
  `3.518 deg` to `1.171 deg` and translation-direction error from `44.22 deg`
  to `34.68 deg`;
- nevertheless, absolute ATE changes from `0.36879 m` to `0.37535 m`.

The geometry and rotation improved, while unconstrained learned camera
translation did not.  V3 therefore keeps the successful outputs and replaces
only camera-center regression with an explicit geometric solve.

This also respects two earlier negative results:

1. fused VGGT tokens are never written into SAM3;
2. all-token fusion is not used as the method because it is much weaker than
   the decoupled patch branch on held-out pointmap quality.

The old translation-only per-instance ICP V3 is not reused either.  Its local
nearest-neighbour minima and conflicting object proposals were already shown
to be fragile.  Persistent instances now improve a dense coherent pointmap;
one camera center is solved from thousands of compatible pixel rays.

## 2. Borrowed ideas from prior work

### Pointmap/ray optimisation

[MASt3R-SLAM](https://arxiv.org/abs/2412.12392) is the closest geometric
precedent.  It performs pointmap matching, camera tracking, local pointmap
fusion, and second-order optimisation.  Its key observation is directly
relevant here: Euclidean 3D point error is easily skewed by incorrect predicted
depth, whereas directional ray error is bounded and more robust.  It therefore
uses Huber-weighted ray errors in an IRLS/Gauss-Newton solve.  With calibration,
it constrains pointmaps to known camera rays and reports improved trajectory
accuracy.

[DROID-SLAM](https://arxiv.org/abs/2108.10869) jointly updates camera pose and
pixelwise depth through dense bundle adjustment.  We borrow the principle that
pose and dense geometry must agree, but use a small analytic translation solve
instead of introducing a new recurrent optimiser.

[VGGT-SLAM](https://arxiv.org/abs/2505.12549) and
[VGGT-SLAM 2.0](https://arxiv.org/abs/2601.19887) show that feed-forward VGGT
geometry benefits from an explicit geometric backend and degeneracy-aware
factor design.  The current seven-frame scope does not need their submap graph,
but it supports separating a learned frontend from a constrained backend.

### Persistent object constraints

[CubeSLAM](https://arxiv.org/abs/1806.00557) shows that objects can provide
long-range geometric and scale constraints and that camera/object estimates can
improve each other through multi-view optimisation.

[DSP-SLAM](https://arxiv.org/abs/2108.09481) uses instance segmentation and an
object-aware pose graph to jointly optimise camera poses, object locations, and
background features.  [DynaSLAM II](https://arxiv.org/abs/2010.07820) likewise
shows that multi-object tracking can benefit camera tracking when static scene
and object motion are handled jointly.

[BundleTrack](https://arxiv.org/abs/2108.00516) combines segmentation, robust
features, memory, and pose-graph optimisation for long-term consistency under
occlusion.  [ObVi-SLAM](https://arxiv.org/abs/2309.15268) uses short-term visual
features together with an uncertainty-aware map of persistent objects for
long-term consistency.  These results motivate our persistent instance memory
and confidence gates, not a semantic token addition to every VGGT feature.

## 3. Optimisation theory

For pixel `u`, let the learned world pointmap predict `X_u` and let the camera
ray in world coordinates be

```text
r_u = normalize(R_c2w K^-1 [u_x, u_y, 1]^T).
```

For a central camera, the true camera center `C` and the world point must lie on
the same line.  The component perpendicular to the ray is therefore

```text
e_line(C) = (I - r_u r_u^T) (X_u - C).
```

The confidence-weighted least-squares objective is

```text
L_line(C) = sum_u w_u ||e_line(C)||^2.
```

It has a three-dimensional closed-form normal equation:

```text
A = sum_u w_u (I - r_u r_u^T)
b = sum_u w_u (I - r_u r_u^T) X_u
C = A^-1 b.
```

The solve is observable when `A` is full rank, which requires a sufficiently
wide set of non-parallel rays.  The implementation rejects too few points and
ill-conditioned `A` instead of applying an arbitrary correction.

To reduce sensitivity to pointmap depth error, the V3 candidate also minimises
the angular approximation

```text
e_angle(C) = ||e_line(C)|| / ||X_u - C||
L_angle(C) = sum_u w_u Huber(e_angle(C)).
```

IRLS freezes the current range and Huber weights, solves the same `3 x 3`
normal equation, and repeats up to six times.  This is the small problem-specific
counterpart of the robust ray optimisation used by MASt3R-SLAM.

## 4. Final causal method

```text
causal SAM3 persistent IDs and masks
    -> decoupled V2 patch geometry adapter
    -> refined world pointmap and point confidence

causal SAM3 appearance
    -> V2 camera adapter
    -> refined camera rotation

reference-frame predicted K + current refined pointmap + refined rotation
    -> angular-Huber ray-center IRLS
    -> one frame-wide camera translation
```

No evaluation GT enters the deployable path.  ScanNet++ uses a fixed physical
camera, so reusing the first predicted intrinsics is a causal calibration
stabiliser.  It is preferable to later per-frame predictions in this clip:
previous diagnostics showed focal error increasing at frame 240.  Frame 90's
camera center is also solved from its own causal pointmap, as in the previously
successful ray baseline.  This defines the pose in the pointmap gauge without
using GT; the cached pointmap Sim(3) and evaluation protocol are never refit on
held-out frames.

Safety gates are:

- at least 1024 confident points;
- normal-matrix condition number at most `1e8`;
- fitted point-to-ray RMSE at most `0.20` native units;
- proposed center shift at most `0.75` native units;
- otherwise exact fallback to the selected V2/baseline center.

## 5. Why success is plausible

This is not a statistical probability claim, but the evidence makes the
mechanism materially better supported than another learned translation head:

1. the input pointmap used by the solve is already 58% better on held-out
   frames;
2. the input rotation is already 67% better on the only held-out pair;
3. the previous all-point ray solve reduced all-pairs translation-direction
   mean from `14.56 deg` to `11.35 deg` before learned pointmap refinement;
4. the solve has only three unknowns and tens of thousands of constraints;
5. failure conditions are measurable without GT and cause fallback.

Expected likelihood for this one scene:

- lower ray residual: very high, because the accepted least-squares solution
  directly optimises it;
- improved relative translation direction: high (`~70-80%` engineering
  expectation);
- improved fixed-reference ATE: moderate (`~55-70%`), because a coherent
  pointmap scale/bias or intrinsics bias can still shift all fitted centers;
- catastrophic degradation: low after shift, residual, conditioning, and
  reference-anchor gates.

## 6. Ablation matrix

| Variant | Pointmap | Rotation | Intrinsics | Pixels | Solver | Question |
|---|---|---|---|---|---|---|
| `raw_baseline_control` | raw | raw | raw | - | none | StreamVGGT baseline |
| `v2_learned_pose_control` | refined | learned | learned | - | none | current V2 pose |
| `ray_baseline_pointmap` | raw | raw | per-frame raw | all | line LS | old ray signal |
| `ray_refined_pointmap_baseline_rotation` | refined | raw | per-frame raw | all | line LS | geometry contribution |
| `ray_refined_pointmap_refined_rotation` | refined | learned | per-frame raw | all | line LS | rotation contribution |
| `ray_refined_pointmap_refined_rk` | refined | learned | per-frame learned | all | line LS | learned K effect |
| `...reference_k` | refined | learned | reference raw | all | line LS | K stabilisation |
| `...reference_k_trimmed` | refined | learned | reference raw | all | trimmed LS | hard trimming |
| `...reference_k_angular_huber` | refined | learned | reference raw | all | angular IRLS | robust-solver ablation |
| `...reference_k_background` | refined | learned | reference raw | background | angular IRLS | instances as conditioners vs fit pixels |
| `...reference_k_instances` | refined | learned | reference raw | instances | angular IRLS | selected V3 |
| `...gt_k_oracle` | refined | learned | GT | all | angular IRLS | intrinsics ceiling only |

The background/instance rows were initially spatial ablations.  The temporal
holdout selected the union of persistent tracked instances: the masks remove
background pointmap/ray inconsistencies, while pooling three independently
located instances retains enough ray diversity to avoid the degeneracy of the
old single-instance centroid/ICP proposals.

## 7. Server command

The V2 checkpoints and feature cache are reused.  The dedicated `ray` stage
loads only the selected dual/aligned checkpoint and does not overwrite the
existing full V2 evaluation CSVs.  No cache building or training is requested:

```bash
zsh streaming_couping/commands_instance_ray_pose_v3.txt
```

New files are written under the existing temporal-holdout evaluation folder:

```text
outputs/streaming_couping_instance_token_pose_temporal_holdout_v2/evaluation/
  ray_pose_summary.csv
  ray_pose_frame_metrics.csv
  ray_pose_rpe.csv
  ray_pose_pair_metrics.csv
  ray_pose_pair_summary.csv
  ray_pose_fit_diagnostics.csv
  ray_pose_compact_summary.csv
  ray_pose_metadata.json
  ray_pose_predictions.pt
```

For handoff, `ray_pose_compact_summary.csv` is sufficient initially; it joins
the primary absolute, relative, all-pairs, fit-acceptance, residual, shift, and
conditioning values for every variant.  `ray_pose_predictions.pt` preserves
all raw pose encodings plus the selected refined pointmap, confidence, tracking
masks, and image paths needed by the later fused/instance PLY export without
rerunning the adapters.

## 8. Temporal-holdout result and final selection

On held-out frames 210 and 240, the selected
`ray_refined_pointmap_refined_rotation_reference_k_instances` row obtained:

| metric | raw StreamVGGT | learned V2 | selected V3 |
|---|---:|---:|---:|
| ATE RMSE | 0.36879 m | 0.37535 m | **0.14387 m** |
| translation mean | 0.35304 m | 0.37342 m | **0.14371 m** |
| translation RPE RMSE | 0.23981 m | 0.17627 m | **0.08986 m** |
| rotation RPE | 3.518 deg | **1.171 deg** | **1.171 deg** |
| all-pair translation direction | 44.22 deg | **34.68 deg** | 40.94 deg |

Both held-out fits were accepted.  Point-to-ray RMSE fell from 0.14058 to
0.02678 in native units and the maximum condition number was 6.69, far below
the configured rejection threshold.  ATE improved by 61.0% over raw
StreamVGGT and 61.7% over V2.  The GT-K oracle reached 0.14615 m ATE, slightly
worse than the selected reference-K result.  GT K does improve the all-pixel
angular-Huber row (0.18165 to 0.14615 m), but the instance mask removes enough
geometric contamination for predicted reference K to match that oracle
without using GT intrinsics.

The claim remains deliberately narrow: V3 substantially repairs absolute and
adjacent translation on this proof-of-concept scene while preserving V2's
rotation gain.  It does **not** improve the all-pair translation-direction
metric over V2, so that metric is recorded as a remaining limitation rather
than hidden behind the ATE result.  Because the final spatial scope was chosen
after inspecting these two held-out frames, this is proof-of-concept model
selection on one scene, not an unbiased generalization claim.

## 9. Final pose and PLY export

The selected artifact can be exported without rerunning SAM3, StreamVGGT,
training, or ray fitting:

```bash
zsh streaming_couping/commands_instance_ray_pose_v3_export.txt
```

This writes `camera_poses.csv`, `camera_poses.npz`, full-scene colored PLYs,
one set of PLYs per persistent instance, and point-selection diagnostics under
`final_instance_ray_pose_v3/`.  The export now includes the rasterized
ScanNet++ GT pointmaps and GT camera poses rather than prediction-only files.
Every prediction is exported in two explicitly named coordinate systems:

- `streamvggt_point_head_native`: deployable but arbitrary-scale;
- `fixed_reference_point_sim3_metric_evaluation_only`: metric visualization
  using the cached GT-fitted reference-frame Sim(3), never presented as a
  deployable inference output.

Native prediction and GT must not be overlaid because they use different
gauges.  For an actual same-coordinate comparison, open
`*_overlay_metric_gt_world.ply`: prediction is red, GT is cyan, and both use
the exact same paired pixels in the ScanNet++ GT world.  The unpaired visible
GT is also exported as `*_gt_visible_all_finite_metric_gt_world.ply`.
`pointcloud_gt_comparison.csv` reports the paired metric distances, while
`camera_comparison_pointmap_sim3.csv` compares predicted and GT cameras after
the same pointmap Sim(3).  This joint reconstruction alignment is intentionally
distinguished from the reference-pose alignment used for the reported ATE.
