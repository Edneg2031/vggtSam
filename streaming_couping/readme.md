# streaming_couping

本目录保留 SAM3.1 × StreamVGGT 的实验代码、配置和复现命令。研究设置、有效证据、失败方法和
结论边界统一记录在：

[本周实验总结](docs/weekly_experiment_summary_2026-08-03_to_2026-08-06.md)

历史代码和 CSV 输出继续保留，但旧版本 Markdown、runbook 和重复实验说明已经删除。后续设计实验
时，以总结中的“下一步最小验证”为准，不再重复扩大 pose adapter 或重跑已证伪的 fusion 分支。

V9.1 已完成并证实实际 local32 history support 本身不足。下一步只分解 token 密度、重复 key
碰撞和空间覆盖，不训练新 matcher 或 pose adapter：

```bash
zsh streaming_couping/commands_v92_support_factorization.txt
```
