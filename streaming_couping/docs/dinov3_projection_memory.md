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

Run it on the server with:

```zsh
zsh streaming_couping/commands_run_dinov3_projection_memory.txt
```

Results are written under
`$VGGT_SAM_STORAGE_ROOT/outputs/streaming_couping_dinov3_projection_memory/`.
The main files are `summary.json`, `copyable_result.txt`,
`association_summary.csv`, `tracking_summary.csv`, `map_summary.csv`, and one
`semantic_map.pt` per clip/branch. The fused map rows are marked
`map_type=fused`; the observation rows are marked `map_type=observation`.
