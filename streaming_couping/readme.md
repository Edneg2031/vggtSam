# streaming_couping

本目录只保留 `V0` SAM3.1 × StreamVGGT 动态实例几何 baseline。当前做法、效果、实验结果和结论边界统一记录在：

[当前状态与实验结论](docs/current_status.md)

运行：

```bash
zsh streaming_couping/commands_v0_baseline.txt
```

该入口支持多 prompt、多实例、未来帧 birth、永久 slot、几何辅助 mask、静态实例几何 pose residual
和无有效实例时的 exact camera-baseline fallback。它不缓存 SAM appearance token，也不声称 SAM token
能改善 pose。旧版本 runner、配置、命令和专用测试已删除；模型、数据、`externals/`、`outputs/`
均不在本次清理范围。
