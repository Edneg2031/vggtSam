# V0 real SAM temporal-prompt A/B

This experiment evaluates whether the causal geometry prompts from the frozen
A--E matrix improve an actual SAM3.1 mask when they are used as positive point
prompts.

It is deliberately a small, controlled step:

```text
frozen V0 cache
    + frozen causal A--E point candidates
    + one single-frame SAM3.1 call per selected frame/slot/branch
    -> same frozen V0 slot, raw pose and raw pointmap
```

StreamVGGT is not rerun. The experiment does not propagate a new video track;
it performs direct single-frame resegmentation on a deterministic common
frame/slot subset. Therefore a positive result is evidence about prompt-side
mask refinement, not yet evidence for a complete temporal closed loop.

## Branches

| Branch | SAM input |
| --- | --- |
| `raw_v0` | Cached V0 mask; no SAM call |
| `A_center` | One point from the frozen A candidate |
| `C_surface_5` | Five points from the frozen C candidate |
| `E_depth_gate_rel_0.15` | E points at relative tolerance 0.15 |
| `E_depth_gate_rel_0.20` | E points at relative tolerance 0.20 |

All prompted branches use the same text prompt from the frozen V0 slot, the
same SAM3.1 checkpoint/session settings, points-only input, and the same output
grid. A prompt is never made from GT.

## Safety fallback

The runner uses fixed values declared on the command line:

```text
prompt score < raw score - 0.10
    or
prompt score < 0.80 * raw score
    -> keep raw V0 mask
```

Missing/empty SAM results and exceptions also keep the raw mask. The output
records both all fallbacks and score-triggered fallbacks. The score field is
`SAM3 out_probs` when exposed; the existing wrapper uses `1.0` as a visibility
fallback when the predictor does not expose a probability, so this semantic
limitation must be considered when interpreting the fallback rate.

## Worsened-frame metrics

The requested threshold is fixed at `0.05`. The primary
`worsened_frame_ratio` is computed over sequence frames that contain at least
one visible GT object under the frozen raw assignment: the prompted mean IoU
must be lower than the raw mean IoU by more than `0.05`. The output also gives
the object-frame version and the ratio restricted to frames where raw IoU was
at least `0.50`.

## Run

First generate the causal matrix:

```bash
zsh streaming_couping/commands_run_v0_temporal_prompt_matrix.txt
```

Then run the real calls:

```bash
zsh streaming_couping/commands_run_v0_temporal_prompt_sam_ab.txt
```

The default invokes a common deterministic subset of 96 frame-slot queries,
which means at most 384 prompted SAM calls. To evaluate every frozen query:

```bash
V0_TEMPORAL_PROMPT_MAX_QUERIES=0 \
  zsh streaming_couping/commands_run_v0_temporal_prompt_sam_ab.txt
```

The output directory contains:

- `prompt_candidates_causal.csv`: selected prompts before SAM is called;
- `sam_events_causal.csv`: call, score, fallback, and final-mask provenance;
- `prompted_masks_causal.pt`: frozen masks/scores for all branches;
- `tracking_metrics.csv`, `tracking_frame_metrics.csv`, and
  `tracking_object_metrics.csv`;
- `map_metrics.csv` and `map_object_metrics.csv`;
- `prompt_branch_metrics.csv`, `worsened_frame_metrics.csv`,
  `branch_summary.csv`, `summary.json`, and `copyable_result.txt`.

GT is opened only after `prompted_masks_causal.pt` has been written. The raw
slot-to-GT assignment is then frozen for every branch. This is a one-scene
pilot; a branch should not be promoted to a full closed-loop implementation
without a new scene-disjoint validation run.
