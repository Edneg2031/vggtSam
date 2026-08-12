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
