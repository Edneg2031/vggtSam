# DINOv3-assisted projection memory

This route does not ask DINOv3 to predict metric coordinates. StreamVGGT
continues to provide the frozen world pointmap and camera geometry. SAM3
provides the short-term mask and source-track identity. DINOv3 only adds an
appearance term when an existing source track re-enters after a fixed absence
gap and must be associated with a persistent world-memory object.

The fixed comparison is:

| Branch | Projection memory | Appearance input |
| --- | --- | --- |
| `raw_all_visible` | no persistent fusion | none |
| `world_memory_only` | yes | none |
| `single_view_dino` | yes | current-frame pooled DINO feature |
| `persistent_dino` | yes | causal EMA DINO feature |
| `shuffled_persistent_dino` | yes | deterministic wrong-slot control |

The experiment uses `appearance_weight=0.20`, `reassociation_gap=3`,
`confirmation_frames=2`, `confirmation_window=4`, and shuffle seed `2026`.
These values are constants in the runner and are not test-tuned.

Candidate generation uses only frozen StreamVGGT/SAM/DINO cache fields. Each
branch is serialized and reloaded before GT-backed tracking and map metrics
are computed. If no source track re-enters far enough to trigger ranking, the
result is `NOT_EXERCISED`, not evidence that DINO failed.

该实验命令已归档；本文仅保留 DINO projection-memory 诊断的设置、结果和结论。

Results are written under
`$VGGT_SAM_STORAGE_ROOT/outputs/streaming_couping_dinov3_projection_memory/`.
The main files are `summary.json`, `copyable_result.txt`,
`association_summary.csv`, `tracking_summary.csv`, `map_summary.csv`, and one
`semantic_map.pt` per clip/branch. The fused map rows are marked
`map_type=fused`; the observation rows are marked `map_type=observation`.

## Server result: r1

The fixed four-clip run completed with `decision=NO_GO`. The DINO path was
actually exercised: the branches produced roughly 150 candidate events and
the DINO branches compared appearance on 118--152 events, depending on the
control. Therefore this is not a `NOT_EXERCISED` result.

The important observations are:

- `single_view_dino` and `persistent_dino` were identical to the geometry-only
  association/map readout on most clips; DINO rarely changed the accepted
  object ownership in a way that changed geometry.
- The aggregate fused-map F5cm was about `0.249`, but fused ghost rate was
  about `0.52` for geometry-only and about `0.55` for the DINO branches. The
  direct point fusion is therefore not a safe map representation.
- On the held-out clip, persistent DINO fused F5cm was `0.14577`, below
  geometry-only `0.15979`; the shuffled control was `0.16071`. This rejects
  the claim that the appearance term provides a reliable causal association
  gain.
- The observation map also lost coverage because tentative observations are
  withheld until confirmation. That is an identity-policy side effect, not
  evidence that the frozen StreamVGGT pointmap improved or worsened by itself.

The correct conclusion is:

```text
DINO appearance-assisted re-association: NO-GO
unvalidated direct object-memory fusion:   main failure mode
```

Do not tune `appearance_weight`, EMA, or the DINO backbone from this result.
后续的 evaluation-only multi-scene GT-mask upper-bound 命令也已归档。

If the oracle map is still poor, the bottleneck is StreamVGGT geometry or
world-space fusion. If the oracle is substantially better, the bottleneck is
SAM ownership/coverage and a mask-side method is justified.
