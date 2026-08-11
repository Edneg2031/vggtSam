# streaming_couping

本目录只保留 `V0` SAM3.1 × StreamVGGT 动态实例几何 baseline。V4–V9.8 的实验设置、有效证据、
失败方法和结论边界统一记录在：

[本周实验总结](docs/weekly_experiment_summary_2026-08-03_to_2026-08-06.md)

运行：

```bash
zsh streaming_couping/commands_v0_baseline.txt
```

该入口支持多 prompt、多实例、未来帧 birth、永久 slot、几何辅助 mask、静态实例几何 pose residual
和无有效实例时的 exact camera-baseline fallback。它不缓存 SAM appearance token，也不声称 SAM token
能改善 pose。旧版本 runner、配置、命令和专用测试已删除；模型、数据、`externals/`、`outputs/`
均不在本次清理范围。
