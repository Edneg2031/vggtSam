# streaming_couping

本目录保留 SAM3.1 × StreamVGGT 的实验代码、配置和复现命令。研究设置、有效证据、失败方法和
结论边界统一记录在：

[本周实验总结](docs/weekly_experiment_summary_2026-08-03_to_2026-08-06.md)

历史代码和 CSV 输出继续保留，但旧版本 Markdown、runbook 和重复实验说明已经删除。后续设计实验
时，以总结中的“下一步最小验证”为准，不再重复扩大 pose adapter 或重跑已证伪的 fusion 分支。

V9.2 已证实增加到 256 个 history key 和消除重复 key 都不能恢复 pose。下一步冻结完全相同的
history edge，诊断硬量化、soft-coordinate 表达上界和 O-R1 噪声容忍度；不训练新 matcher 或
pose adapter：

```bash
zsh streaming_couping/commands_v93_quantization_tolerance.txt
```
