# streaming_couping

本目录只保留 `V0` SAM3.1 × StreamVGGT 动态实例几何 baseline。V4–V9.8 的实验设置、有效证据、
失败方法和结论边界统一记录在：

[本周实验总结](docs/weekly_experiment_summary_2026-08-03_to_2026-08-06.md)

运行：

```bash
zsh streaming_couping/commands_v0_baseline.txt
```

该入口支持多 prompt、多实例、未来帧 birth、永久 slot 和几何辅助 mask。最终 pose 使用恢复的
V7.1 camera L0；V7.4-style 静态实例 geometry transport 作为单独 candidate 输出，并在无有效实例时
exact fallback 到 L0。运行会拒绝零梯度、零参数更新、训练 loss 不下降或 future loss 不优于 raw 的
no-op 结果。它不缓存 SAM appearance token，也不声称 SAM token 能改善 pose。
