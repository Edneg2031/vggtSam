# streaming_couping

本目录只保留 `V0` SAM3.1 × StreamVGGT 动态实例几何 baseline。V4–V9.8 的实验设置、有效证据、
失败方法和结论边界统一记录在：

[本周实验总结](docs/weekly_experiment_summary_2026-08-03_to_2026-08-06.md)

运行：

```bash
zsh streaming_couping/commands_v0_baseline.txt
```

该入口支持多 prompt、多实例、未来帧 birth、永久 slot 和几何辅助 mask。最终 pose candidate 使用
V7.1-style camera hidden + raw relative pose；V7.4-style 静态实例 geometry transport 单独计分，
并在无有效实例时 exact fallback 到 camera candidate。运行会拒绝训练 no-op，并按 short/medium/long
三个未来 fold 报告 mean camera-center error 和 worse frames；只有每折均改善且零坏帧才验收。

同一命令还会运行 `normal / raw-pose-only / camera-token-only / time-only` 同参数量控制，输出到
`outputs/streaming_couping_v0/validation/`。当前服务器逐帧结果已经显示 medium fold 下降，因此旧的
aggregate `baseline_acceptance_pass=1` 无效；V0 暂时只是诊断 candidate，不是稳健 pose baseline。
它不缓存 SAM appearance token，也不声称 SAM token 能改善 pose。
