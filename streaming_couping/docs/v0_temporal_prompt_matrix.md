# V0 temporal prompt A–E diagnostic

This experiment tests whether historical StreamVGGT geometry can provide
useful point or box prompts for a future SAM call. It is a frozen diagnostic,
not a closed-loop implementation.

The runner uses the retained single-scene V0 cache and the raw StreamVGGT
camera trajectory. For each current frame/slot it only uses an observation
from a strictly earlier frame. It writes all causal candidates before opening
ground-truth masks, and it never calls SAM or changes the pointmap.

## Branches

| Branch | Candidate |
| --- | --- |
| `A_center` | One median 3-D center projected into the current frame |
| `B_surface_3` | Three front-depth, 2-D spatially spread surface points |
| `C_surface_5` | Five front-depth, 2-D spatially spread surface points |
| `D_bbox` | Quantile bounding box around the projected historical cloud |
| `E_depth_gate` | The C points retained by current predicted-depth consistency |

E is evaluated at a fixed, predeclared relative-tolerance curve. No tolerance
is selected from test annotations.

## Metrics

For point branches, the primary quantities are kept separate:

- `accepted_prompt_precision`: correct accepted in-bounds points divided by
  accepted in-bounds points for slots with a frozen raw-to-GT assignment;
- `object_frame_coverage`: visible assigned query frame-slots with at least
  one correct point divided by visible assigned query frame-slots;
- `prompt_availability`: visible assigned query frame-slots with at least one
  accepted in-bounds point divided by visible assigned query frame-slots;
- `precision_coverage_f1`: harmonic mean of precision and visible coverage;
- accepted-point mean/min pairwise pixel distance and normalized dispersion.

The strict coverage column also counts assigned frames where the object is not
visible, which is useful for detecting false prompts but is not the primary
coverage denominator.

Box prompts are not converted into point precision. They report box purity,
recall, IoU, IoU-hit rates at 0.25/0.50, availability, and area ratio.

## Run on the V0 server

```bash
zsh streaming_couping/commands_run_v0_temporal_prompt_matrix.txt
```

The command expects the completed V0 cache at
`$VGGT_SAM_STORAGE_ROOT/outputs/streaming_couping_v0/cache`. It produces:

- `point_candidates_causal.csv`, `box_candidates_causal.csv`,
  `query_events_causal.csv`: pre-GT candidate artifacts;
- `point_prompt_metrics.csv`, `box_prompt_metrics.csv`, `query_metrics.csv`:
  evaluated rows;
- `branch_summary.csv`, `per_scene_metrics.csv`, `per_object_metrics.csv`, and
  `depth_gate_curve.csv`;
- `summary.json` and terminal-friendly `copyable_result.txt`.

Only a branch that improves over `A_center` while preserving meaningful
object-frame coverage should be used to justify a subsequent real SAM A/B
rerun. This diagnostic itself always reports `decision=DIAGNOSTIC_ONLY`.
