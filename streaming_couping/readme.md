# streaming_couping

本目录只保留 `V0` SAM3.1 × StreamVGGT 动态实例几何 baseline。V4–V9.8 的实验设置、有效证据、
失败方法和结论边界统一记录在：

[本周实验总结](docs/weekly_experiment_summary_2026-08-03_to_2026-08-06.md)

运行：

```bash
zsh streaming_couping/commands_v0_baseline.txt
```

该入口支持多 prompt、多实例、未来帧 birth、永久 slot 和几何辅助 mask。V0 r4 不训练 pose 模型；
`selected_world_to_camera` 与 raw StreamVGGT 逐元素完全相同。运行只审计 dynamic registry、late birth、
永久 ID、成熟 observation 的因果性以及几何 mask competition，并输出 raw pose 参考指标。

V0 r3 的 direct SE(3) 与 geometry-transport pose 路线已经三折证伪并从 active code 删除。当前 V0
只声明流式 tracking 工程 baseline，不声明 pose 改善，也不缓存或使用 SAM appearance token。下一条
pose factor 后端通过独立验证前不会覆盖 raw pose。

当前独立 pose 实验：E0-r2 已证明在当前 clip/protocol 下普通 edge-DTF 对 raw pose不形成局部
basin；E1 仅用固定
`0.25° / 0.25% scene-scale` 正负梯度候选判断 edge descent 是否朝向 GT。GT 只在所有候选生成后
评分，实验不输出 pose，也不会修改 V0：

```bash
zsh streaming_couping/commands_e1_edge_directional_gt_audit.txt
```
# Current pose candidate: G0 static projective ICP

V0 remains the accepted engineering baseline: causal multi-instance SAM3.1
tracking (including future object births) plus the unmodified raw StreamVGGT
pose.  G0 is an isolated candidate and never writes selected poses.

G0 follows established RGB-D odometry rather than learning a token-to-camera
adapter.  It uses frozen StreamVGGT depth, confidence, intrinsics and raw pose
to form multi-history projective correspondences, then optimizes only the
current SE(3) with coarse-to-fine robust point-to-plane LM.  SAM is used only
to exclude prompted tracked-object regions.  The controls are full image and
the same mask shifted to an unrelated location.  The cache's prompted-object
mask is not described as complete dynamic ground truth.

Candidate generation receives no GT field.  Native/reference-aligned GT pose
is decoded only after candidates are immutable and is used only for reporting.
Run:

```zsh
zsh streaming_couping/commands_g0_static_projective_icp.txt
```

An all-fold geometry pass requires all four frames in every fold to be active,
the fixed deployable geometry energy to decrease, mean rotation and center
error to improve, and zero worse frames.  A SAM-specific conclusion additionally
requires the SAM exclusion branch to beat the shifted-mask control in every
fold.  Until that gate passes, V0 continues to select raw StreamVGGT pose.
