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

首次单场景运行已经得到 all-fold pass=0：medium/long 的 SIFT 对应不足，short fold 唯一接受的更新没有
instance correspondence，并与 shuffled-ID control 完全相同。因此该实现只作为已记录的失败候选保留，
不得启用为 V0 pose；完整证据见实验总结第 13 节。

## V0 SAM memory → StreamVGGT KV retrieval probe

当前无训练候选不读取 SAM hidden/appearance token。SAM3.1 的 causal persistent track registry 只给出
same-instance 历史帧集合，StreamVGGT 第一层原生 Q–K 在集合内排序；选中帧的完整原生 KV 在24层复用。
`raw_full_history / retrieve_qk / sam_gated_qk / sam_hybrid_qk /
shuffled_instance_memory` 五支按锁定协议运行，raw selected pose保持不变：

```bash
zsh streaming_couping/commands_v0_sam_memory_retrieval.txt
```

只有 `sam_hybrid_qk` 三折同时改善 center、rotation、固定 raw-reference Sim(3) 与固定 raw-confidence support
下的 paired pointmap RMSE，
并逐折优于同预算 `retrieve_qk`，且 shuffled identity 破坏收益，才允许建立 SAM memory 因果结论。
