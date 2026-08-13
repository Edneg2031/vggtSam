# streaming_couping

本目录只保留 `V0` SAM3.1 × StreamVGGT 因果多实例 tracking baseline。运行：

```bash
zsh streaming_couping/commands_v0_baseline.txt
```

V0 当前提供：

- SAM3.1 多 prompt、多实例、forward-only discovery/tracking；
- 未来帧 late birth、永久 slot 和 persistent ID；
- StreamVGGT 几何辅助 mask prompt/competition；
- 不缓存、不读取 SAM appearance token；
- 不训练 pose 模型；`selected_world_to_camera` 与 raw StreamVGGT 完全相同。

因此 V0 当前只声明可用的流式实例追踪工程基线，不声明位姿改善。当前配置使用
`bed/wardrobe/picture/mat/chair` 五类 prompt 和 16 个永久 registry slots。语义实例也不等于动态实例，
这些 prompts 并不是 class-agnostic object proposal。

E0/E1 edge-DTF、G0 projective ICP、direct SE(3)、TrackHead-BA 和其它失败候选的 active code 已删除。
已有结果与下一条“实例状态分解 → mask-guided StreamVGGT pose”理论设计统一保存在：

[V4–V9.8 实验证据与 V0 理论设计](docs/weekly_experiment_summary_2026-08-03_to_2026-08-06.md)

新 pose 方法在多个场景的独立时间折上通过预锁消融前，不得修改 V0 selected pose。

静态场景的第一个独立 pose 候选使用通用 SIFT 对应，并让 SAM persistent region ID 只负责拒绝
跨区域误匹配；历史 3D 来自冻结的 StreamVGGT world pointmap，当前 pose 由 PnP-RANSAC 求解。运行：

```bash
zsh streaming_couping/commands_v0_sam_region_pose_candidate.txt
```

该命令比较 `full_image_match / sam_region_identity / shuffled_instance_identity` 三组等量对应。
候选 pose 只单独保存和评分，不替换 V0 的 raw StreamVGGT pose。
