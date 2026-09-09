# HorizonStream GT Pose Feedback POC

This experiment answers a narrow causal question:

> If a correct pose is known at frame `t`, can it be written into the
> HorizonStream streaming process so that later pose estimates improve?

The POC uses 50 frames from ScanNet++ scene `00a231a370`, with correction frame
`t=15` by default. It does not load SAM3, object masks, ICP, a learned loss, or
any training code.

## State audit

The active wrapper is `HorizonStreamModel.forward_chunk`. Its sequence state
contains:

- `frame_kv_caches`;
- `global_kv_caches`;
- an optional `gla_cache`.

These caches store transformer feature keys, values, and optional recurrent
attention state. `forward_chunk` does not read a previous pose, world
pointmap, cached world geometry, or a pose anchor. The camera pose is decoded
from the current chunk's pose tokens. Therefore an external 6DoF pose cannot
be inserted into the model KV cache as a semantically valid pose correction.

The current inference path has a separate causal state in
`online_motion_averaging`: `online_absolute_poses`. That state does determine
how the next chunk's relative camera prediction is accumulated into a public
absolute pose. The Feedback branch writes the correction into this pose
accumulator and leaves the model KV/GLA caches unchanged.

The model output convention is `world_to_camera`. Branch trajectories and the
GT correction use `camera_to_world`. The selected GT trajectory is normalized
to the first selected frame so all branches share one evaluation gauge:

```text
delta_c2w = T_gt_c2w @ inverse(T_raw_c2w)
```

## Branches

### Raw

Normal HorizonStream streaming inference and online motion averaging.

### Posthoc

Only the saved output pose at frame `t` is replaced by the GT pose. No state
is changed. Frames after `t` should be identical to Raw.

### Feedback

The same model chunk outputs are replayed causally. At frame `t`, the GT
correction is written into `online_absolute_poses`, which is the pose state
used by later motion averaging. This tests a real pose-accumulator feedback
path without claiming that HorizonStream's hidden KV state was corrected.

## Outputs

The command writes:

- `raw_metrics.csv`;
- `posthoc_metrics.csv`;
- `feedback_metrics.csv`;
- `future_pose_gain.csv`;
- `summary.json`;
- `poses.pt`;
- `run.log`.

`future_pose_gain.csv` contains the raw, posthoc, and feedback translation and
rotation errors for `t+1` through `t+10`, plus:

```text
feedback_translation_gain = raw_translation_error - feedback_translation_error
feedback_rotation_gain = raw_rotation_error - feedback_rotation_error
```

The printed `ATE` is the HorizonStream-style Sim(3)-aligned translation RMSE;
`direct_ATE` is the translation RMSE in the common first-frame GT gauge used
for the causal comparison. RPE is adjacent-frame relative-pose RMSE.
`summary.json` also records the separate model-KV conclusion:

- `GT_FEEDBACK_WORKS` means the causal pose accumulator improved at least one
  future translation or rotation error;
- `GT_FEEDBACK_DOES_NOT_WORK` means no future error improved;
- `model_kv_feedback_decision` is always `GT_FEEDBACK_DOES_NOT_WORK` for the
  current wrapper because no pose/world-state write interface exists there.

## Run

```zsh
zsh streaming_couping/commands_run_scannet_horizonstream_gt_feedback_poc.txt
```

The frame count, correction frame, and output directory can be overridden with
`GT_FEEDBACK_FRAME_COUNT`, `GT_FEEDBACK_CORRECTION_FRAME`, and
`GT_FEEDBACK_OUTPUT_DIR`.
